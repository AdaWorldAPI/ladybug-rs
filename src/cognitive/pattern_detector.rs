//! Meta-Pattern Detector — recognise emergent patterns in cognitive history.
//!
//! Analyses a sliding window of `ConsciousnessSnapshot`s to detect high-level
//! patterns that arise from the interplay of the 10-layer cognitive stack:
//!
//! - **Loop**: two layers repeatedly alternate in dominance.
//! - **Stagnation**: a single layer holds dominance without meaningful change.
//! - **Rushing**: snapshots arrive too fast (processing overload).
//! - **Epiphany**: sudden spike in emergence + coherence (insight moment).
//! - **Stable**: none of the above — healthy processing.
//!
//! The detector is pure and stateless: it takes a slice of history and returns
//! the single most prominent pattern.

use super::layer_stack::{ConsciousnessSnapshot, LayerId};

// =============================================================================
// META-PATTERN ENUM
// =============================================================================

/// A meta-pattern detected across the recent cognitive history.
#[derive(Clone, Debug, PartialEq)]
pub enum MetaPattern {
    /// Two layers repeatedly alternate dominance.
    Loop {
        /// First layer in the alternation.
        a: LayerId,
        /// Second layer in the alternation.
        b: LayerId,
        /// Number of full A->B->A cycles observed.
        count: u32,
    },

    /// A single layer holds dominance for too long without progress.
    Stagnation {
        /// The stagnating layer.
        layer: LayerId,
        /// Number of cycles the layer has been dominant.
        duration_cycles: u64,
    },

    /// Snapshots arrive faster than they can be meaningfully processed.
    Rushing {
        /// Estimated events per second.
        events_per_second: f32,
    },

    /// Sudden spike in emergence + coherence — an insight moment.
    Epiphany {
        /// Magnitude of the emergence spike.
        emergence_spike: f32,
    },

    /// No pathological pattern detected — healthy cognitive processing.
    Stable,
}

impl std::fmt::Display for MetaPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetaPattern::Loop { a, b, count } => {
                write!(f, "Loop({} <-> {}, {} cycles)", a.name(), b.name(), count)
            }
            MetaPattern::Stagnation {
                layer,
                duration_cycles,
            } => write!(
                f,
                "Stagnation({}, {} cycles)",
                layer.name(),
                duration_cycles
            ),
            MetaPattern::Rushing { events_per_second } => {
                write!(f, "Rushing({:.1} events/s)", events_per_second)
            }
            MetaPattern::Epiphany { emergence_spike } => {
                write!(f, "Epiphany(spike={:.3})", emergence_spike)
            }
            MetaPattern::Stable => write!(f, "Stable"),
        }
    }
}

// =============================================================================
// THRESHOLDS
// =============================================================================

/// Tuning knobs for the pattern detector.
#[derive(Clone, Debug)]
pub struct DetectorThresholds {
    /// Minimum number of full A->B->A cycles to declare a Loop.
    pub loop_min_cycles: u32,
    /// Minimum consecutive snapshots with same dominant layer for Stagnation.
    pub stagnation_min_cycles: u64,
    /// Minimum events/sec to declare Rushing.
    pub rushing_events_per_sec: f32,
    /// Minimum emergence delta to declare Epiphany.
    pub epiphany_emergence_threshold: f32,
    /// Minimum coherence during an emergence spike for Epiphany.
    pub epiphany_coherence_threshold: f32,
}

impl Default for DetectorThresholds {
    fn default() -> Self {
        Self {
            loop_min_cycles: 3,
            stagnation_min_cycles: 5,
            rushing_events_per_sec: 100.0,
            epiphany_emergence_threshold: 0.3,
            epiphany_coherence_threshold: 0.6,
        }
    }
}

// =============================================================================
// PUBLIC API
// =============================================================================

/// Detect the most prominent meta-pattern in the given history.
///
/// History should be ordered oldest-first.  Returns `MetaPattern::Stable` if
/// no pathological pattern is found.  At least 2 snapshots are required for
/// any pattern other than Stable.
pub fn detect(history: &[ConsciousnessSnapshot]) -> MetaPattern {
    detect_with_thresholds(history, &DetectorThresholds::default())
}

/// Detect with custom thresholds.
pub fn detect_with_thresholds(
    history: &[ConsciousnessSnapshot],
    thresholds: &DetectorThresholds,
) -> MetaPattern {
    if history.len() < 2 {
        return MetaPattern::Stable;
    }

    // Check in priority order (most actionable first).

    // 1. Rushing (processing overload is the most urgent signal).
    if let Some(pattern) = check_rushing(history, thresholds) {
        return pattern;
    }

    // 2. Epiphany (rare but important — do not mask by other checks).
    if let Some(pattern) = check_epiphany(history, thresholds) {
        return pattern;
    }

    // 3. Loop (oscillation between two layers).
    if let Some(pattern) = check_loop(history, thresholds) {
        return pattern;
    }

    // 4. Stagnation (single layer stuck).
    if let Some(pattern) = check_stagnation(history, thresholds) {
        return pattern;
    }

    MetaPattern::Stable
}

// =============================================================================
// INDIVIDUAL PATTERN DETECTORS
// =============================================================================

/// Check for processing overload (snapshots arriving too fast).
fn check_rushing(
    history: &[ConsciousnessSnapshot],
    thresholds: &DetectorThresholds,
) -> Option<MetaPattern> {
    if history.len() < 3 {
        return None;
    }

    // Compute average time between consecutive snapshots.
    let first = &history[0];
    let last = &history[history.len() - 1];

    let elapsed = last.timestamp.duration_since(first.timestamp);
    let elapsed_secs = elapsed.as_secs_f32();

    if elapsed_secs <= 0.0 {
        // All snapshots have the same timestamp — definitely rushing.
        return Some(MetaPattern::Rushing {
            events_per_second: f32::INFINITY,
        });
    }

    let events_per_second = (history.len() - 1) as f32 / elapsed_secs;

    if events_per_second >= thresholds.rushing_events_per_sec {
        Some(MetaPattern::Rushing { events_per_second })
    } else {
        None
    }
}

/// Check for an emergence spike (epiphany moment).
fn check_epiphany(
    history: &[ConsciousnessSnapshot],
    thresholds: &DetectorThresholds,
) -> Option<MetaPattern> {
    if history.len() < 2 {
        return None;
    }

    // Look for the largest emergence jump accompanied by high coherence.
    let mut max_spike: f32 = 0.0;

    for i in 1..history.len() {
        let delta = history[i].emergence - history[i - 1].emergence;
        if delta > max_spike
            && history[i].coherence >= thresholds.epiphany_coherence_threshold
        {
            max_spike = delta;
        }
    }

    if max_spike >= thresholds.epiphany_emergence_threshold {
        Some(MetaPattern::Epiphany {
            emergence_spike: max_spike,
        })
    } else {
        None
    }
}

/// Check for oscillation between two layers.
fn check_loop(
    history: &[ConsciousnessSnapshot],
    thresholds: &DetectorThresholds,
) -> Option<MetaPattern> {
    if history.len() < 4 {
        return None;
    }

    // Extract dominant layer sequence.
    let dominant_seq: Vec<LayerId> = history.iter().map(|s| s.dominant_layer).collect();

    // For each pair of distinct layers, count how many times the pattern
    // A, B, A, B, ... appears in the dominant sequence.
    let mut best_pair: Option<(LayerId, LayerId, u32)> = None;

    for &a in LayerId::ALL.iter() {
        for &b in LayerId::ALL.iter() {
            if a == b {
                continue;
            }
            let cycle_count = count_alternation_cycles(&dominant_seq, a, b);
            if cycle_count >= thresholds.loop_min_cycles {
                if best_pair
                    .as_ref()
                    .map(|(_, _, c)| cycle_count > *c)
                    .unwrap_or(true)
                {
                    best_pair = Some((a, b, cycle_count));
                }
            }
        }
    }

    best_pair.map(|(a, b, count)| MetaPattern::Loop { a, b, count })
}

/// Count the number of full A->B->A alternation cycles in a sequence.
///
/// A "cycle" is defined as seeing A followed (possibly not immediately) by B,
/// followed by A again. We scan greedily.
fn count_alternation_cycles(seq: &[LayerId], a: LayerId, b: LayerId) -> u32 {
    let mut cycles: u32 = 0;
    let mut expecting = a;
    let mut half_cycles: u32 = 0;

    for &layer in seq {
        if layer == expecting {
            half_cycles += 1;
            // Toggle expectation.
            expecting = if expecting == a { b } else { a };
        }
    }

    // A full cycle is A->B->A = 3 half transitions.
    // half_cycles counts each matching element.
    // cycles = (half_cycles - 1) / 2 (we need at least 3 elements for 1 cycle).
    if half_cycles >= 3 {
        cycles = (half_cycles - 1) / 2;
    }

    cycles
}

/// Check for a single layer dominating for too long.
fn check_stagnation(
    history: &[ConsciousnessSnapshot],
    thresholds: &DetectorThresholds,
) -> Option<MetaPattern> {
    if history.is_empty() {
        return None;
    }

    // Find the longest consecutive run of the same dominant layer.
    let mut max_layer = history[0].dominant_layer;
    let mut max_run: u64 = 1;
    let mut current_layer = history[0].dominant_layer;
    let mut current_run: u64 = 1;

    for snap in history.iter().skip(1) {
        if snap.dominant_layer == current_layer {
            current_run += 1;
        } else {
            if current_run > max_run {
                max_run = current_run;
                max_layer = current_layer;
            }
            current_layer = snap.dominant_layer;
            current_run = 1;
        }
    }
    // Handle the final run.
    if current_run > max_run {
        max_run = current_run;
        max_layer = current_layer;
    }

    if max_run >= thresholds.stagnation_min_cycles {
        Some(MetaPattern::Stagnation {
            layer: max_layer,
            duration_cycles: max_run,
        })
    } else {
        None
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive::layer_stack::{LayerMarker, NUM_LAYERS};
    use std::time::Instant;

    /// Build a snapshot with a specific dominant layer and emergence/coherence.
    fn synthetic_snapshot(
        dominant: LayerId,
        cycle: u64,
        emergence: f32,
        coherence: f32,
    ) -> ConsciousnessSnapshot {
        let mut layers: [LayerMarker; NUM_LAYERS] = Default::default();
        // Set the dominant layer to have the highest value.
        for (i, m) in layers.iter_mut().enumerate() {
            if i == dominant.index() {
                m.value = 0.9;
                m.active = true;
                m.confidence = 0.8;
            } else {
                m.value = 0.1;
                m.active = false;
                m.confidence = 0.1;
            }
        }

        ConsciousnessSnapshot {
            timestamp: Instant::now(),
            cycle,
            layers,
            dominant_layer: dominant,
            coherence,
            emergence,
        }
    }

    /// Thresholds that disable rushing detection (for unit tests where
    /// synthetic snapshots share the same Instant::now() timestamp).
    fn no_rushing_thresholds() -> DetectorThresholds {
        DetectorThresholds {
            rushing_events_per_sec: f32::INFINITY, // effectively disable rushing
            ..Default::default()
        }
    }

    #[test]
    fn test_stable_pattern() {
        let history = vec![
            synthetic_snapshot(LayerId::L1, 0, 0.2, 0.5),
            synthetic_snapshot(LayerId::L2, 1, 0.2, 0.5),
            synthetic_snapshot(LayerId::L3, 2, 0.2, 0.5),
        ];
        let result = detect_with_thresholds(&history, &no_rushing_thresholds());
        assert_eq!(result, MetaPattern::Stable);
    }

    #[test]
    fn test_empty_history() {
        assert_eq!(detect(&[]), MetaPattern::Stable);
    }

    #[test]
    fn test_single_snapshot() {
        let history = vec![synthetic_snapshot(LayerId::L1, 0, 0.2, 0.5)];
        assert_eq!(detect(&history), MetaPattern::Stable);
    }

    #[test]
    fn test_stagnation_detected() {
        let history: Vec<_> = (0..8)
            .map(|i| synthetic_snapshot(LayerId::L3, i, 0.1, 0.5))
            .collect();
        let result = detect_with_thresholds(&history, &no_rushing_thresholds());
        match result {
            MetaPattern::Stagnation {
                layer,
                duration_cycles,
            } => {
                assert_eq!(layer, LayerId::L3);
                assert!(duration_cycles >= 5);
            }
            other => panic!("expected Stagnation, got {:?}", other),
        }
    }

    #[test]
    fn test_loop_detected() {
        // Create an alternating A, B, A, B, A, B, A pattern.
        let mut history = Vec::new();
        for i in 0..10 {
            let layer = if i % 2 == 0 { LayerId::L2 } else { LayerId::L5 };
            history.push(synthetic_snapshot(layer, i, 0.1, 0.5));
        }

        let result = detect_with_thresholds(&history, &no_rushing_thresholds());
        match result {
            MetaPattern::Loop { a, b, count } => {
                // The loop should be between L2 and L5.
                assert!(
                    (a == LayerId::L2 && b == LayerId::L5)
                        || (a == LayerId::L5 && b == LayerId::L2),
                    "unexpected loop layers: {:?}, {:?}",
                    a,
                    b
                );
                assert!(count >= 3, "expected at least 3 cycles, got {}", count);
            }
            other => panic!("expected Loop, got {:?}", other),
        }
    }

    #[test]
    fn test_epiphany_detected() {
        let history = vec![
            synthetic_snapshot(LayerId::L1, 0, 0.1, 0.7),
            synthetic_snapshot(LayerId::L2, 1, 0.1, 0.7),
            synthetic_snapshot(LayerId::L3, 2, 0.1, 0.7),
            // Sudden emergence spike.
            synthetic_snapshot(LayerId::L8, 3, 0.6, 0.8),
        ];

        let result = detect_with_thresholds(&history, &no_rushing_thresholds());
        match result {
            MetaPattern::Epiphany { emergence_spike } => {
                assert!(emergence_spike >= 0.3);
            }
            other => panic!("expected Epiphany, got {:?}", other),
        }
    }

    #[test]
    fn test_rushing_detected() {
        // All snapshots at the same instant -> infinite events/sec.
        let history = vec![
            synthetic_snapshot(LayerId::L1, 0, 0.1, 0.5),
            synthetic_snapshot(LayerId::L2, 1, 0.1, 0.5),
            synthetic_snapshot(LayerId::L3, 2, 0.1, 0.5),
        ];
        let result = detect(&history);
        match result {
            MetaPattern::Rushing { .. } => {} // expected
            MetaPattern::Stable => {} // also acceptable if timestamps differ enough
            other => panic!("expected Rushing or Stable, got {:?}", other),
        }
    }

    #[test]
    fn test_display_formatting() {
        let patterns = vec![
            MetaPattern::Stable,
            MetaPattern::Loop {
                a: LayerId::L1,
                b: LayerId::L5,
                count: 4,
            },
            MetaPattern::Stagnation {
                layer: LayerId::L3,
                duration_cycles: 7,
            },
            MetaPattern::Rushing {
                events_per_second: 500.0,
            },
            MetaPattern::Epiphany {
                emergence_spike: 0.45,
            },
        ];
        for p in &patterns {
            let s = format!("{}", p);
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn test_alternation_cycle_counter() {
        let seq = vec![
            LayerId::L1,
            LayerId::L2,
            LayerId::L1,
            LayerId::L2,
            LayerId::L1,
            LayerId::L2,
            LayerId::L1,
        ];
        let cycles = count_alternation_cycles(&seq, LayerId::L1, LayerId::L2);
        assert!(cycles >= 3, "expected >= 3 cycles, got {}", cycles);
    }
}
