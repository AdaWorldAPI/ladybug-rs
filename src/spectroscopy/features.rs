//! Feature extraction from Container bitpatterns.
//!
//! Computes spectral features from 8,192-bit containers:
//! - **Entropy**: Shannon entropy of the bit distribution across word blocks
//! - **Density**: Fraction of set bits (popcount / CONTAINER_BITS)
//! - **Bridgeness**: How many word-boundary bit transitions exist (XOR adjacency)
//! - **Abstraction depth**: Hierarchical structure metric from block-level variance
//!
//! All features are normalised to the [0.0, 1.0] range.

use crate::container::{Container, CONTAINER_BITS, CONTAINER_WORDS};

/// Spectral features extracted from a Container bitpattern.
#[derive(Clone, Debug)]
pub struct SpectralFeatures {
    /// Shannon entropy of word-level popcount distribution [0.0, 1.0].
    pub entropy: f32,

    /// Fraction of set bits: popcount / CONTAINER_BITS [0.0, 1.0].
    pub density: f32,

    /// Normalised count of bit transitions across word boundaries [0.0, 1.0].
    pub bridgeness: f32,

    /// Hierarchical abstraction depth derived from block-level variance [0.0, 1.0].
    pub abstraction_depth: f32,

    /// Spectral energy: sum of squared word popcounts, normalised [0.0, 1.0].
    pub spectral_energy: f32,

    /// Symmetry: correlation between first half and reversed second half [0.0, 1.0].
    pub symmetry: f32,

    /// Clustering coefficient: how tightly set bits cluster together [0.0, 1.0].
    pub clustering: f32,

    /// Run-length complexity: normalised count of bit-run transitions [0.0, 1.0].
    pub run_complexity: f32,
}

impl Default for SpectralFeatures {
    fn default() -> Self {
        Self {
            entropy: 0.0,
            density: 0.0,
            bridgeness: 0.0,
            abstraction_depth: 0.0,
            spectral_energy: 0.0,
            symmetry: 0.5,
            clustering: 0.0,
            run_complexity: 0.0,
        }
    }
}

/// Extract all spectral features from a Container.
pub fn extract(container: &Container) -> SpectralFeatures {
    if container.is_zero() {
        return SpectralFeatures::default();
    }

    let word_popcounts = compute_word_popcounts(container);

    SpectralFeatures {
        entropy: compute_entropy(&word_popcounts),
        density: compute_density(container),
        bridgeness: compute_bridgeness(container),
        abstraction_depth: compute_abstraction_depth(&word_popcounts),
        spectral_energy: compute_spectral_energy(&word_popcounts),
        symmetry: compute_symmetry(&word_popcounts),
        clustering: compute_clustering(container),
        run_complexity: compute_run_complexity(container),
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Popcount per word.
fn compute_word_popcounts(container: &Container) -> [u32; CONTAINER_WORDS] {
    let mut pops = [0u32; CONTAINER_WORDS];
    for (i, &w) in container.words.iter().enumerate() {
        pops[i] = w.count_ones();
    }
    pops
}

/// Shannon entropy of the word-level popcount histogram.
///
/// We treat each word's popcount as a bin value (0..=64) and compute the
/// entropy of the resulting discrete distribution.  The result is normalised
/// by log2(CONTAINER_WORDS) so that a perfectly uniform distribution yields
/// 1.0 and a single-word concentration yields 0.0.
fn compute_entropy(pops: &[u32; CONTAINER_WORDS]) -> f32 {
    let total: f64 = pops.iter().map(|&p| p as f64).sum();
    if total == 0.0 {
        return 0.0;
    }

    let mut entropy: f64 = 0.0;
    for &p in pops.iter() {
        if p > 0 {
            let prob = p as f64 / total;
            entropy -= prob * prob.ln();
        }
    }

    // Normalise by maximum possible entropy (uniform distribution).
    let max_entropy = (CONTAINER_WORDS as f64).ln();
    if max_entropy > 0.0 {
        (entropy / max_entropy) as f32
    } else {
        0.0
    }
}

/// Density: fraction of set bits.
fn compute_density(container: &Container) -> f32 {
    container.popcount() as f32 / CONTAINER_BITS as f32
}

/// Bridgeness: normalised count of bit transitions across word boundaries.
///
/// For each pair of adjacent words we XOR the last bit of word i with the
/// first bit of word i+1.  High bridgeness means frequent cross-word
/// transitions — an indicator of distributed rather than block-local encoding.
fn compute_bridgeness(container: &Container) -> f32 {
    if CONTAINER_WORDS < 2 {
        return 0.0;
    }

    let mut transitions: u32 = 0;
    for i in 0..(CONTAINER_WORDS - 1) {
        let last_bit = (container.words[i] >> 63) & 1;
        let first_bit = container.words[i + 1] & 1;
        if last_bit != first_bit {
            transitions += 1;
        }
    }

    // Also count intra-word transitions (bit-to-bit changes within each word).
    let mut intra_transitions: u32 = 0;
    for &w in container.words.iter() {
        // XOR with self shifted right gives transition map.
        intra_transitions += (w ^ (w >> 1)).count_ones();
    }

    // Combine inter-word and intra-word transitions.
    let max_transitions = (CONTAINER_BITS - 1) as f32;
    let total = transitions as f32 + intra_transitions as f32;
    (total / max_transitions).min(1.0)
}

/// Abstraction depth: derived from variance across hierarchical block sizes.
///
/// We compute popcount variance at multiple block sizes (1 word, 4 words, 16
/// words, 64 words).  Higher variance at coarser levels means more structured
/// content — a proxy for abstraction depth.  The result is mapped to [0, 1].
fn compute_abstraction_depth(pops: &[u32; CONTAINER_WORDS]) -> f32 {
    let block_sizes: &[usize] = &[1, 4, 16, 64];
    let mut weighted_variance: f32 = 0.0;
    let mut total_weight: f32 = 0.0;

    for (level, &bs) in block_sizes.iter().enumerate() {
        if bs > CONTAINER_WORDS {
            break;
        }
        let n_blocks = CONTAINER_WORDS / bs;
        if n_blocks < 2 {
            continue;
        }

        // Compute block-level sums.
        let mut block_sums = Vec::with_capacity(n_blocks);
        for b in 0..n_blocks {
            let start = b * bs;
            let sum: u32 = pops[start..start + bs].iter().sum();
            block_sums.push(sum as f32);
        }

        let mean = block_sums.iter().sum::<f32>() / n_blocks as f32;
        let var = block_sums.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / n_blocks as f32;

        // Weight coarser levels more heavily (they represent deeper structure).
        let weight = (level + 1) as f32;
        let max_possible = (bs as f32 * 64.0).powi(2); // crude upper bound
        let normalised_var = if max_possible > 0.0 {
            (var / max_possible).sqrt()
        } else {
            0.0
        };

        weighted_variance += normalised_var * weight;
        total_weight += weight;
    }

    if total_weight > 0.0 {
        (weighted_variance / total_weight).min(1.0)
    } else {
        0.0
    }
}

/// Spectral energy: L2 norm of per-word popcounts, normalised.
fn compute_spectral_energy(pops: &[u32; CONTAINER_WORDS]) -> f32 {
    let energy: f64 = pops.iter().map(|&p| (p as f64).powi(2)).sum();
    let max_energy = CONTAINER_WORDS as f64 * (64.0f64).powi(2);
    if max_energy > 0.0 {
        (energy / max_energy).sqrt() as f32
    } else {
        0.0
    }
}

/// Symmetry: correlation between the first half and reversed second half.
fn compute_symmetry(pops: &[u32; CONTAINER_WORDS]) -> f32 {
    let half = CONTAINER_WORDS / 2;
    if half == 0 {
        return 0.5;
    }

    let mut dot: f64 = 0.0;
    let mut norm_a: f64 = 0.0;
    let mut norm_b: f64 = 0.0;

    for i in 0..half {
        let a = pops[i] as f64;
        let b = pops[CONTAINER_WORDS - 1 - i] as f64;
        dot += a * b;
        norm_a += a * a;
        norm_b += b * b;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom > 0.0 {
        (dot / denom) as f32
    } else {
        0.5
    }
}

/// Clustering: how tightly set bits cluster (using gap variance).
fn compute_clustering(container: &Container) -> f32 {
    // Count gaps between consecutive set bits using word scanning.
    let mut gaps: Vec<u32> = Vec::new();
    let mut last_set: Option<u32> = None;

    for word_idx in 0..CONTAINER_WORDS {
        let w = container.words[word_idx];
        if w == 0 {
            continue;
        }
        let base = (word_idx as u32) * 64;
        let mut bits = w;
        while bits != 0 {
            let trailing = bits.trailing_zeros();
            let bit_pos = base + trailing;
            if let Some(prev) = last_set {
                gaps.push(bit_pos - prev);
            }
            last_set = Some(bit_pos);
            bits &= bits - 1; // clear lowest set bit
        }
    }

    if gaps.len() < 2 {
        return 0.0;
    }

    let mean_gap: f32 = gaps.iter().sum::<u32>() as f32 / gaps.len() as f32;
    let var_gap: f32 = gaps
        .iter()
        .map(|&g| (g as f32 - mean_gap).powi(2))
        .sum::<f32>()
        / gaps.len() as f32;

    // High variance in gaps means clustered (some tight, some sparse).
    // Normalise against expected mean gap squared.
    let expected_mean = (CONTAINER_BITS as f32) / (container.popcount().max(1) as f32);
    let clustering = (var_gap / (expected_mean * expected_mean + 1.0)).sqrt();
    clustering.min(1.0)
}

/// Run-length complexity: count of transitions between 0-runs and 1-runs.
fn compute_run_complexity(container: &Container) -> f32 {
    let mut transitions: u32 = 0;

    for word_idx in 0..CONTAINER_WORDS {
        let w = container.words[word_idx];
        // Transitions within word: XOR with self shifted right.
        transitions += (w ^ (w >> 1)).count_ones();

        // Transition at word boundary.
        if word_idx + 1 < CONTAINER_WORDS {
            let last_bit = (w >> 63) & 1;
            let next_first = container.words[word_idx + 1] & 1;
            if last_bit != next_first {
                transitions += 1;
            }
        }
    }

    // Maximum transitions is CONTAINER_BITS - 1 (alternating pattern).
    let max = (CONTAINER_BITS - 1) as f32;
    if max > 0.0 {
        transitions as f32 / max
    } else {
        0.0
    }
}

/// Map abstraction depth to a RungLevel index (0-9).
///
/// The mapping uses quantile thresholds calibrated for Container data:
/// - [0.00, 0.05) -> 0 (Surface)
/// - [0.05, 0.10) -> 1 (Shallow)
/// - [0.10, 0.18) -> 2 (Contextual)
/// - [0.18, 0.28) -> 3 (Analogical)
/// - [0.28, 0.38) -> 4 (Abstract)
/// - [0.38, 0.50) -> 5 (Structural)
/// - [0.50, 0.62) -> 6 (Counterfactual)
/// - [0.62, 0.75) -> 7 (Meta)
/// - [0.75, 0.88) -> 8 (Recursive)
/// - [0.88, 1.00] -> 9 (Transcendent)
pub fn depth_to_rung_index(depth: f32) -> u8 {
    match depth {
        d if d < 0.05 => 0,
        d if d < 0.10 => 1,
        d if d < 0.18 => 2,
        d if d < 0.28 => 3,
        d if d < 0.38 => 4,
        d if d < 0.50 => 5,
        d if d < 0.62 => 6,
        d if d < 0.75 => 7,
        d if d < 0.88 => 8,
        _ => 9,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_container_features() {
        let c = Container::zero();
        let f = extract(&c);
        assert_eq!(f.density, 0.0);
        assert_eq!(f.entropy, 0.0);
        assert_eq!(f.bridgeness, 0.0);
    }

    #[test]
    fn test_ones_container_density() {
        let c = Container::ones();
        let f = extract(&c);
        assert!((f.density - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_random_container_features_in_range() {
        let c = Container::random(42);
        let f = extract(&c);
        assert!(f.density >= 0.0 && f.density <= 1.0);
        assert!(f.entropy >= 0.0 && f.entropy <= 1.0);
        assert!(f.bridgeness >= 0.0 && f.bridgeness <= 1.0);
        assert!(f.abstraction_depth >= 0.0 && f.abstraction_depth <= 1.0);
        assert!(f.spectral_energy >= 0.0 && f.spectral_energy <= 1.0);
        assert!(f.symmetry >= 0.0 && f.symmetry <= 1.0);
        assert!(f.clustering >= 0.0 && f.clustering <= 1.0);
        assert!(f.run_complexity >= 0.0 && f.run_complexity <= 1.0);
    }

    #[test]
    fn test_random_density_near_half() {
        let c = Container::random(99);
        let f = extract(&c);
        // Random containers should have density near 0.5.
        assert!(f.density > 0.35 && f.density < 0.65, "density={}", f.density);
    }

    #[test]
    fn test_depth_to_rung_boundaries() {
        assert_eq!(depth_to_rung_index(0.0), 0);
        assert_eq!(depth_to_rung_index(0.04), 0);
        assert_eq!(depth_to_rung_index(0.05), 1);
        assert_eq!(depth_to_rung_index(0.50), 6);
        assert_eq!(depth_to_rung_index(1.0), 9);
    }

    #[test]
    fn test_symmetry_of_palindromic_container() {
        // Build a container that's symmetric: word[i] == word[127-i].
        let mut c = Container::zero();
        for i in 0..64 {
            let val = (i as u64).wrapping_mul(0x517cc1b727220a95);
            c.words[i] = val;
            c.words[CONTAINER_WORDS - 1 - i] = val;
        }
        let f = extract(&c);
        assert!(
            f.symmetry > 0.95,
            "symmetric container should have high symmetry, got {}",
            f.symmetry
        );
    }
}
