//! Recursive Thought Expansion — apply rung processing iteratively until convergence.
//!
//! Output fingerprint of rung N becomes input to rung N+1 with measured convergence.
//! Stops when Hamming distance delta drops below Berry-Esseen noise floor.
//!
//! # Science
//! - Hofstadter (1979): Strange loops as recursive self-reference
//! - Schmidhuber (2010): Recursive compression as intelligence measure
//! - Berry-Esseen (1941/42): At d=16384, Normal approximation error < 0.004

use crate::search::hdr_cascade::{hamming_distance, WORDS};

const TOTAL_BITS: f32 = 16384.0;

/// Single step in a recursive expansion trace.
#[derive(Debug, Clone)]
pub struct ExpansionStep {
    /// Recursion depth (0 = initial)
    pub depth: u8,
    /// Normalized Hamming delta from previous step
    pub delta: f32,
    /// Fingerprint at this depth
    pub fingerprint: [u64; WORDS],
}

/// Complete trace of a recursive expansion.
#[derive(Debug, Clone)]
pub struct ExpansionTrace {
    /// Steps taken
    pub steps: Vec<ExpansionStep>,
    /// Whether the expansion converged (delta < threshold)
    pub converged: bool,
    /// Total information accumulated (sum of deltas)
    pub total_information: f32,
}

impl ExpansionTrace {
    /// Final fingerprint after expansion.
    pub fn result(&self) -> Option<&[u64; WORDS]> {
        self.steps.last().map(|s| &s.fingerprint)
    }

    /// Number of steps taken.
    pub fn depth(&self) -> usize {
        self.steps.len()
    }
}

/// Configuration for recursive expansion.
#[derive(Debug, Clone)]
pub struct RecursiveExpansion {
    /// Maximum recursion depth (safety cap, default 7)
    pub max_depth: u8,
    /// Convergence threshold: stop when delta < this (default: Berry-Esseen 0.004)
    pub convergence_threshold: f32,
}

impl Default for RecursiveExpansion {
    fn default() -> Self {
        Self {
            max_depth: 7,
            convergence_threshold: 0.004, // Berry-Esseen noise floor at d=16384
        }
    }
}

impl RecursiveExpansion {
    pub fn new(max_depth: u8, convergence_threshold: f32) -> Self {
        Self { max_depth, convergence_threshold }
    }

    /// Apply a transformation function recursively until convergence or max_depth.
    ///
    /// The `transform` function takes (current_fingerprint, depth) and returns next.
    /// This is generic: the caller decides what "rung processing" means.
    pub fn expand<F>(&self, seed: &[u64; WORDS], transform: F) -> ExpansionTrace
    where
        F: Fn(&[u64; WORDS], u8) -> [u64; WORDS],
    {
        let mut current = *seed;
        let mut steps = Vec::new();
        let mut total_information = 0.0;

        for depth in 0..self.max_depth {
            let next = transform(&current, depth);
            let distance = hamming_distance(&current, &next);
            let delta = distance as f32 / TOTAL_BITS;
            total_information += delta;

            steps.push(ExpansionStep {
                depth,
                delta,
                fingerprint: next,
            });

            if delta < self.convergence_threshold {
                return ExpansionTrace { steps, converged: true, total_information };
            }
            current = next;
        }

        ExpansionTrace { steps, converged: false, total_information }
    }
}

/// Convergent/divergent oscillation protocol.
///
/// Alternates between divergent exploration (bundle with distant neighbors)
/// and convergent exploitation (select nearest match).
///
/// # Science
/// - Guilford (1967): Divergent vs convergent production
/// - Kanerva (2009): BUNDLE = superposition (divergent), SIMILARITY = nearest (convergent)
/// - Sutton & Barto (2018): ε-greedy exploration-exploitation tradeoff
#[derive(Debug, Clone)]
pub struct OscillationResult {
    /// Final fingerprint after oscillation
    pub result: [u64; WORDS],
    /// Exploration ratio at each round (1.0 = diverge, 0.0 = converge)
    pub ratios: Vec<f32>,
    /// Total information traversed
    pub total_delta: f32,
}

/// Run convergent/divergent oscillation on a fingerprint.
///
/// - Even rounds: diverge (XOR with noise = explore neighborhood)
/// - Odd rounds: converge (keep only majority bits = stabilize)
pub fn oscillate(
    seed: &[u64; WORDS],
    rounds: usize,
    noise_magnitude: f32,
) -> OscillationResult {
    let mut current = *seed;
    let mut ratios = Vec::with_capacity(rounds);
    let mut total_delta = 0.0;

    for round in 0..rounds {
        let prev = current;
        if round % 2 == 0 {
            // Diverge: inject noise by flipping bits
            let flip_count = (TOTAL_BITS * noise_magnitude.min(0.5)) as usize;
            let bits_per_word = flip_count / WORDS;
            for word in current.iter_mut() {
                // Deterministic noise based on round + word position
                let mask = if bits_per_word > 0 {
                    let shift = (round * 7 + 3) % 64;
                    u64::MAX.wrapping_shr(64u32.saturating_sub(bits_per_word as u32))
                        .rotate_left(shift as u32)
                } else { 0 };
                *word ^= mask;
            }
            ratios.push(1.0);
        } else {
            // Converge: average with seed (majority vote between current and seed)
            for (w, s) in current.iter_mut().zip(seed.iter()) {
                // Keep bits where current and seed agree; for disagreements, prefer seed
                let agree = !(*w ^ *s); // Bits where they agree
                *w = (*w & agree) | (*s & !agree);
            }
            ratios.push(0.0);
        }
        let delta = hamming_distance(&prev, &current) as f32 / TOTAL_BITS;
        total_delta += delta;
    }

    OscillationResult {
        result: current,
        ratios,
        total_delta,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_fp(seed: u64) -> [u64; WORDS] {
        let mut fp = [0u64; WORDS];
        let mut state = seed;
        for w in fp.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *w = state;
        }
        fp
    }

    #[test]
    fn test_recursive_expansion_converges() {
        let expander = RecursiveExpansion::default();
        let seed = make_fp(42);

        // Identity transform should converge in 1 step
        let trace = expander.expand(&seed, |fp, _depth| *fp);
        assert!(trace.converged);
        assert_eq!(trace.depth(), 1);
    }

    #[test]
    fn test_recursive_expansion_max_depth() {
        let expander = RecursiveExpansion::new(3, 0.0001);
        let seed = make_fp(42);

        // Random transform won't converge
        let trace = expander.expand(&seed, |fp, depth| {
            let mut next = *fp;
            next[depth as usize % WORDS] ^= 0xFFFF_FFFF;
            next
        });
        assert!(!trace.converged);
        assert_eq!(trace.depth(), 3);
    }

    #[test]
    fn test_recursive_expansion_total_information() {
        let expander = RecursiveExpansion::new(5, 0.001);
        let seed = make_fp(42);

        let trace = expander.expand(&seed, |fp, depth| {
            let mut next = *fp;
            // Smaller changes each depth
            let n_words = (5 - depth as usize).max(1);
            for i in 0..n_words {
                next[(depth as usize * 10 + i) % WORDS] ^= 0xFF;
            }
            next
        });
        assert!(trace.total_information > 0.0);
    }

    #[test]
    fn test_oscillation_returns_to_seed() {
        let seed = make_fp(42);
        let result = oscillate(&seed, 10, 0.01);
        // After oscillation with small noise, should be close to seed
        let dist = hamming_distance(&seed, &result.result) as f32 / TOTAL_BITS;
        assert!(dist < 0.1, "Oscillation should stay near seed, got delta={dist}");
    }

    #[test]
    fn test_oscillation_alternating_ratios() {
        let seed = make_fp(42);
        let result = oscillate(&seed, 6, 0.1);
        assert_eq!(result.ratios.len(), 6);
        assert_eq!(result.ratios[0], 1.0); // Diverge
        assert_eq!(result.ratios[1], 0.0); // Converge
        assert_eq!(result.ratios[2], 1.0); // Diverge
    }
}
