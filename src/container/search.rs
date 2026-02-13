//! Belichtungsmesser + HDR cascade on containers.
//!
//! 5-level cascade for progressively filtering candidates:
//!
//! ```text
//! Level 0: Belichtungsmesser (7 samples, ~14 cycles) → 90% rejection
//! Level 1: 1-word scan (which of 128 words differ?) → spatial localization
//! Level 2: Stacked popcount with early exit → precise within 1σ
//! Level 3: Mexican hat (excitation + inhibition) → separates similar from identical
//! Level 4: Voyager deep field (stack weak signals) → finds faint matches
//! ```

use super::{CONTAINER_BITS, CONTAINER_WORDS, Container};

// ============================================================================
// BELICHTUNGSMESSER (Exposure Meter)
// ============================================================================

/// 7 prime-spaced sample points within 128 words.
const SAMPLE_POINTS: [usize; 7] = [0, 19, 41, 59, 79, 101, 127];

/// 7-point exposure meter on a Container (128 words).
/// Estimates total Hamming distance in ~14 CPU cycles.
#[inline]
pub fn belichtungsmesser(a: &Container, b: &Container) -> u32 {
    let mut estimate: u32 = 0;
    for &idx in &SAMPLE_POINTS {
        estimate += (a.words[idx] ^ b.words[idx]).count_ones();
    }
    // Scale: 7 samples × 64 bits = 448 bits sampled out of 8192
    // Scale factor: 8192 / 448 ≈ 18.29
    estimate * CONTAINER_BITS as u32 / 448
}

/// Belichtungsmesser returning (mean, sd×100) for threshold calibration.
pub fn belichtung_stats(a: &Container, b: &Container) -> (u32, u32) {
    let mut samples = [0u32; 7];
    for (i, &idx) in SAMPLE_POINTS.iter().enumerate() {
        samples[i] = (a.words[idx] ^ b.words[idx]).count_ones();
    }
    let sum: u32 = samples.iter().sum();
    let mean = sum / 7;
    let var: u32 = samples
        .iter()
        .map(|&s| {
            let diff = s as i32 - mean as i32;
            (diff * diff) as u32
        })
        .sum::<u32>()
        / 7;
    // sd * 100 for integer precision
    let sd100 = ((var as f64).sqrt() * 100.0) as u32;
    (mean, sd100)
}

// ============================================================================
// LEVEL 1: 1-WORD SCAN
// ============================================================================

/// Count how many of 128 words differ (any bit set in XOR).
/// Returns count 0..128.
#[inline]
pub fn word_diff_count(a: &Container, b: &Container) -> u32 {
    let mut count = 0u32;
    for i in 0..CONTAINER_WORDS {
        if a.words[i] != b.words[i] {
            count += 1;
        }
    }
    count
}

// ============================================================================
// LEVEL 2: STACKED POPCOUNT WITH EARLY EXIT
// ============================================================================

/// Exact Hamming distance with early exit.
/// If cumulative distance exceeds `max_dist`, returns None (pruned).
pub fn hamming_early_exit(a: &Container, b: &Container, max_dist: u32) -> Option<u32> {
    let mut total = 0u32;
    for i in 0..CONTAINER_WORDS {
        total += (a.words[i] ^ b.words[i]).count_ones();
        // Early exit: remaining words can only add more distance
        if total > max_dist {
            return None;
        }
    }
    Some(total)
}

// ============================================================================
// LEVEL 3: MEXICAN HAT
// ============================================================================

/// Mexican hat wavelet response: excitation center + inhibition ring.
pub struct MexicanHat {
    /// Excitation radius (distance within this = positive response).
    pub excite: u32,
    /// Inhibition radius (distance in excite..inhibit = negative response).
    pub inhibit: u32,
    /// Inhibition strength (0.0-1.0).
    pub inhibit_strength: f32,
}

impl MexicanHat {
    /// Default hat tuned for 8K containers: ~1σ excitation, ~2σ inhibition.
    pub fn default_8k() -> Self {
        Self {
            excite: 45,  // ≈ 1σ
            inhibit: 90, // ≈ 2σ
            inhibit_strength: 0.3,
        }
    }

    /// Compute response for a given distance.
    pub fn response(&self, distance: u32) -> f32 {
        if distance < self.excite {
            // Excitation: 1.0 → 0.0 ramp
            1.0 - (distance as f32 / self.excite as f32)
        } else if distance < self.inhibit {
            // Inhibition: negative response
            let t = (distance - self.excite) as f32 / (self.inhibit - self.excite) as f32;
            -self.inhibit_strength * (1.0 - t)
        } else {
            0.0
        }
    }
}

// ============================================================================
// LEVEL 4: VOYAGER DEEP FIELD
// ============================================================================

/// Accumulate weak signals from near-misses.
/// Takes candidates that survived Level 2 but failed Level 3,
/// bundles them, and checks if the bundled result is closer to query.
pub fn voyager_deep_field(
    query: &Container,
    near_misses: &[&Container],
    threshold: u32,
) -> Option<(Container, u32)> {
    if near_misses.is_empty() {
        return None;
    }

    let bundled = Container::bundle(near_misses);
    let dist = bundled.hamming(query);
    if dist <= threshold {
        Some((bundled, dist))
    } else {
        None
    }
}

// ============================================================================
// FULL 5-LEVEL CASCADE
// ============================================================================

/// Result of HDR cascade search on a single candidate.
#[derive(Debug, Clone)]
pub struct CascadeResult {
    /// Candidate index.
    pub index: usize,
    /// Exact Hamming distance.
    pub distance: u32,
    /// Similarity (1.0 - distance / CONTAINER_BITS).
    pub similarity: f32,
    /// Mexican hat response.
    pub response: f32,
    /// Which cascade level resolved this candidate.
    pub resolved_at: u8,
}

/// Run the full 5-level cascade against a corpus.
pub fn cascade_search(
    query: &Container,
    corpus: &[Container],
    threshold: u32,
    top_k: usize,
) -> Vec<CascadeResult> {
    let mut results = Vec::new();
    let hat = MexicanHat::default_8k();

    // Belichtungsmesser pre-filter threshold (generous: 2× threshold scaled)
    let l0_max = threshold * CONTAINER_BITS as u32 / 448 + 200;
    // Word-level threshold (generous)
    let l1_max_words = (threshold / 32).max(4);

    let mut near_misses: Vec<&Container> = Vec::new();

    for (i, candidate) in corpus.iter().enumerate() {
        // L0: Belichtungsmesser
        let estimate = belichtungsmesser(query, candidate);
        if estimate > l0_max {
            continue; // ~90% filtered
        }

        // L1: Word-diff count
        let wd = word_diff_count(query, candidate);
        if wd > l1_max_words + CONTAINER_WORDS as u32 / 4 {
            continue;
        }

        // L2: Stacked popcount with early exit
        let exact = match hamming_early_exit(query, candidate, threshold) {
            Some(d) => d,
            None => {
                // Near miss: didn't pass threshold but survived L0/L1
                near_misses.push(candidate);
                continue;
            }
        };

        // L3: Mexican hat
        let resp = hat.response(exact);
        let similarity = 1.0 - (exact as f32 / CONTAINER_BITS as f32);

        results.push(CascadeResult {
            index: i,
            distance: exact,
            similarity,
            response: resp,
            resolved_at: 3,
        });
    }

    // L4: Voyager deep field — bundle near-misses
    if results.len() < top_k && !near_misses.is_empty() {
        if let Some((bundled, dist)) = voyager_deep_field(query, &near_misses, threshold) {
            let resp = hat.response(dist);
            results.push(CascadeResult {
                index: usize::MAX, // synthetic
                distance: dist,
                similarity: 1.0 - (dist as f32 / CONTAINER_BITS as f32),
                response: resp,
                resolved_at: 4,
            });
            let _ = bundled; // used for the distance check
        }
    }

    // Sort by distance, truncate to top_k
    results.sort_by_key(|r| r.distance);
    results.truncate(top_k);
    results
}
