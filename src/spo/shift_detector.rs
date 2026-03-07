//! Stripe Shift Detector: 0.5σ Migration as Distributional Shift Signal
//!
//! Tracks population migration across 6 σ-stripes per plane across time windows.
//! The migration velocity between adjacent stripes IS the NARS evidence rate:
//!
//! - Population migrating toward foveal → codebook improving → ↑ confidence
//! - Population migrating toward noise → codebook going stale → ↓ confidence
//! - Bimodal migration (both directions) → speciation event → world splitting
//! - Stable → steady state
//!
//! Cost: 5 CMP instructions per plane per candidate. Zero meaningful overhead.
//!
//! # Integration with CollapseGate
//!
//! ```text
//! Shift toward noise  → bias HOLD  (don't commit while ground is moving)
//! Shift toward foveal → bias FLOW  (world is clarifying, commit faster)
//! Bimodal             → bias HOLD  (world is splitting, need more evidence)
//! Stable              → no bias    (use existing gate logic)
//! ```

use rustynum_bnn::causal_trajectory::{
    ShiftDetector as BnnShiftDetector, ShiftDirection, ShiftSignal, StripeHistogram,
};
use rustynum_core::{CollapseGate, SigmaGate};

use super::spo_harvest::SpoDistanceResult;

// =============================================================================
// SPO SHIFT DETECTOR — wraps BNN ShiftDetector with SPO-specific logic
// =============================================================================

/// SPO-aware stripe shift detector.
///
/// Wraps the rustynum-bnn `ShiftDetector` with SPO-specific semantics:
/// per-plane σ-stripe tracking, CollapseGate bias, and integration with
/// `SpoDistanceResult` from the harvest.
pub struct SpoShiftDetector {
    /// Inner BNN shift detector (owns the stripe histograms).
    inner: BnnShiftDetector,
    /// σ-gate for computing continuous σ from raw Hamming distances.
    gate: SigmaGate,
}

impl SpoShiftDetector {
    /// Create a new shift detector with the given σ-gate.
    pub fn new(gate: SigmaGate) -> Self {
        Self {
            inner: BnnShiftDetector::new(),
            gate,
        }
    }

    /// Record a single SPO distance result into the current window.
    ///
    /// Extracts per-plane σ-values from raw Hamming distances and records
    /// them into the appropriate stripe bins.
    pub fn record(&mut self, result: &SpoDistanceResult) {
        let x_sigma = self.raw_sigma(result.x_dist);
        let y_sigma = self.raw_sigma(result.y_dist);
        let z_sigma = self.raw_sigma(result.z_dist);

        self.inner.record(0, x_sigma); // plane 0 = S (X-axis = S⊕P)
        self.inner.record(1, y_sigma); // plane 1 = P (Y-axis = P⊕O)
        self.inner.record(2, z_sigma); // plane 2 = O (Z-axis = S⊕O)
    }

    /// Advance to the next time window.
    ///
    /// Current histograms become previous; current is reset to zero.
    /// Call this after processing a batch of searches.
    pub fn advance_window(&mut self) {
        self.inner.advance_window();
    }

    /// Detect distributional shift between previous and current windows.
    ///
    /// Returns `None` if insufficient data (< 2 windows).
    pub fn detect_shift(&self) -> Option<ShiftSignal> {
        self.inner.detect_shift()
    }

    /// Map the current shift signal to a CollapseGate bias.
    ///
    /// - `TowardNoise` → `Hold` (don't commit while ground is moving)
    /// - `TowardFoveal` → `Flow` (world is clarifying, commit faster)
    /// - `Bimodal` → `Hold` (world is splitting, need more evidence)
    /// - `Stable` → `None` (no bias, use existing gate logic)
    pub fn gate_bias(&self) -> Option<CollapseGate> {
        self.inner.gate_bias()
    }

    /// Get the current window's stripe histograms [X, Y, Z].
    pub fn current_histograms(&self) -> &[StripeHistogram; 3] {
        &self.inner.current
    }

    /// Get the previous window's stripe histograms [X, Y, Z].
    pub fn previous_histograms(&self) -> &[StripeHistogram; 3] {
        &self.inner.previous
    }

    /// Number of completed windows.
    pub fn window_count(&self) -> u32 {
        self.inner.window_count
    }

    /// Compute continuous σ from raw Hamming distance.
    ///
    /// σ = (μ - distance) / σ_unit. Higher = better match.
    #[inline]
    fn raw_sigma(&self, distance: u32) -> f32 {
        let sigma_f = self.gate.sigma_unit as f32;
        if sigma_f > 0.0 {
            (self.gate.mu as f32 - distance as f32) / sigma_f
        } else {
            0.0
        }
    }
}

// =============================================================================
// CONFIRMED SHIFT — cross-validates shift with CHAODA anomaly
// =============================================================================

/// Result of cross-validating a stripe shift with CHAODA anomaly detection.
#[derive(Clone, Debug)]
pub enum ConfirmedShift {
    /// Schaltminute + TowardNoise → global drift. Everything moving.
    GlobalDrift {
        confidence: f32,
    },
    /// Schaltsekunde + TowardFoveal → local refinement. One region sharpening.
    LocalRefinement {
        confidence: f32,
    },
    /// Schaltminute + Bimodal → speciation. World splitting into clusters.
    Speciation {
        confidence: f32,
    },
    /// No confirmed anomaly-shift correlation. Keep monitoring.
    Monitoring {
        confidence: f32,
    },
}

/// Cross-validate a shift signal with a CHAODA anomaly flag.
///
/// `is_schaltminute` = global anomaly (large-scale distributional change).
/// `is_schaltsekunde` = local anomaly (small-scale point anomaly).
///
/// The combination of shift direction + anomaly type gives higher-confidence
/// classification than either signal alone.
pub fn cross_validate_shift(
    shift: &ShiftSignal,
    is_schaltminute: bool,
    is_schaltsekunde: bool,
) -> ConfirmedShift {
    match (is_schaltminute, shift.direction) {
        (true, ShiftDirection::TowardNoise) => ConfirmedShift::GlobalDrift {
            confidence: 0.95,
        },
        (true, ShiftDirection::Bimodal) => ConfirmedShift::Speciation {
            confidence: 0.90,
        },
        _ if is_schaltsekunde && shift.direction == ShiftDirection::TowardFoveal => {
            ConfirmedShift::LocalRefinement {
                confidence: 0.85,
            }
        }
        _ => ConfirmedShift::Monitoring {
            confidence: 0.5,
        },
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spo::spo_harvest::spo_distance;

    fn make_gate() -> SigmaGate {
        SigmaGate::new(16_384)
    }

    fn random_plane(seed: u64) -> Vec<u64> {
        let mut state = seed;
        (0..256)
            .map(|_| {
                state = state.wrapping_add(0x9e3779b97f4a7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                z ^ (z >> 31)
            })
            .collect()
    }

    #[test]
    fn test_shift_detector_recording() {
        let gate = make_gate();
        let mut detector = SpoShiftDetector::new(gate);

        // Record some results.
        for i in 0..20u64 {
            let result = spo_distance(
                &random_plane(i * 6 + 1),
                &random_plane(i * 6 + 2),
                &random_plane(i * 6 + 3),
                &random_plane(i * 6 + 4),
                &random_plane(i * 6 + 5),
                &random_plane(i * 6 + 6),
                &gate,
            );
            detector.record(&result);
        }

        // Should have populated the current histograms.
        let histograms = detector.current_histograms();
        for hist in histograms {
            assert!(hist.total() > 0, "Histogram should have entries");
        }
    }

    #[test]
    fn test_shift_detector_window_advance() {
        let gate = make_gate();
        let mut detector = SpoShiftDetector::new(gate);

        // First window: random.
        for i in 0..10u64 {
            let result = spo_distance(
                &random_plane(i * 6 + 1),
                &random_plane(i * 6 + 2),
                &random_plane(i * 6 + 3),
                &random_plane(i * 6 + 4),
                &random_plane(i * 6 + 5),
                &random_plane(i * 6 + 6),
                &gate,
            );
            detector.record(&result);
        }

        detector.advance_window();
        assert_eq!(detector.window_count(), 1);

        // Second window: also random.
        for i in 100..110u64 {
            let result = spo_distance(
                &random_plane(i * 6 + 1),
                &random_plane(i * 6 + 2),
                &random_plane(i * 6 + 3),
                &random_plane(i * 6 + 4),
                &random_plane(i * 6 + 5),
                &random_plane(i * 6 + 6),
                &gate,
            );
            detector.record(&result);
        }

        // Should be able to detect shift now.
        let shift = detector.detect_shift();
        assert!(shift.is_some(), "Should detect shift after 2 windows");

        // Random→Random should be Stable.
        let signal = shift.unwrap();
        assert_eq!(
            signal.direction,
            ShiftDirection::Stable,
            "Random→Random should be Stable, got {:?}",
            signal.direction
        );
    }

    #[test]
    fn test_gate_bias_stable() {
        let gate = make_gate();
        let detector = SpoShiftDetector::new(gate);

        // No windows recorded → no bias.
        assert_eq!(detector.gate_bias(), None);
    }

    #[test]
    fn test_cross_validate_shift() {
        let signal = ShiftSignal {
            direction: ShiftDirection::TowardNoise,
            plane_directions: [ShiftDirection::TowardNoise; 3],
            com_delta: [-0.5, -0.3, -0.4],
            magnitude: 1.2,
        };

        let confirmed = cross_validate_shift(&signal, true, false);
        match confirmed {
            ConfirmedShift::GlobalDrift { confidence } => {
                assert!(confidence > 0.9);
            }
            _ => panic!("Expected GlobalDrift"),
        }
    }
}
