//! Layer 7: MUL Gate — Binary Gate With 5 Blocking Criteria
//!
//! The gate sits between assessment (L1-L6) and navigation (L8-L9).
//! ALL 5 criteria must pass or the system gathers more evidence.

use super::dk_detector::{DKDetector, DKPosition};
use super::false_flow::FalseFlowDetector;
use super::homeostasis::{CognitiveHomeostasis, HomeostasisState};
use super::trust_qualia::TrustQualia;

/// Why the gate blocked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateBlockReason {
    /// On Mount Stupid — overconfident with insufficient evidence
    MountStupid,
    /// Problem space not yet mapped
    ComplexityUnmapped,
    /// System is depleted (apathy or high allostatic load)
    Depleted,
    /// Trust is below acceptable threshold
    TrustInsufficient,
    /// In severe false flow — spinning wheels
    FalseFlow,
}

/// MUL Gate — 5 blocking criteria that ALL must pass.
pub struct MulGate;

impl MulGate {
    /// Evaluate the 5 gate criteria.
    pub fn check(
        dk: &DKDetector,
        trust: &TrustQualia,
        homeostasis: &CognitiveHomeostasis,
        false_flow: &FalseFlowDetector,
        complexity_mapped: bool,
    ) -> Result<(), GateBlockReason> {
        // 1. NOT on Mount Stupid
        if dk.position == DKPosition::MountStupid {
            return Err(GateBlockReason::MountStupid);
        }

        // 2. Complexity has been mapped
        if !complexity_mapped {
            return Err(GateBlockReason::ComplexityUnmapped);
        }

        // 3. NOT depleted
        if homeostasis.state == HomeostasisState::Apathy
            || homeostasis.allostatic_load > 0.85
        {
            return Err(GateBlockReason::Depleted);
        }

        // 4. Trust is acceptable
        if trust.needs_repair() {
            return Err(GateBlockReason::TrustInsufficient);
        }

        // 5. NOT in severe false flow
        if false_flow.should_disrupt() {
            return Err(GateBlockReason::FalseFlow);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_state() -> (DKDetector, TrustQualia, CognitiveHomeostasis, FalseFlowDetector) {
        let mut dk = DKDetector::new();
        dk.position = DKPosition::SlopeOfEnlightenment;
        let trust = TrustQualia::uniform(0.7);
        let homeostasis = CognitiveHomeostasis::new();
        let false_flow = FalseFlowDetector::new(20);
        (dk, trust, homeostasis, false_flow)
    }

    #[test]
    fn test_gate_passes_nominal() {
        let (dk, trust, homeostasis, false_flow) = default_state();
        assert!(MulGate::check(&dk, &trust, &homeostasis, &false_flow, true).is_ok());
    }

    #[test]
    fn test_gate_blocks_mount_stupid() {
        let (mut dk, trust, homeostasis, false_flow) = default_state();
        dk.position = DKPosition::MountStupid;
        let result = MulGate::check(&dk, &trust, &homeostasis, &false_flow, true);
        assert_eq!(result.unwrap_err(), GateBlockReason::MountStupid);
    }

    #[test]
    fn test_gate_blocks_unmapped() {
        let (dk, trust, homeostasis, false_flow) = default_state();
        let result = MulGate::check(&dk, &trust, &homeostasis, &false_flow, false);
        assert_eq!(result.unwrap_err(), GateBlockReason::ComplexityUnmapped);
    }

    #[test]
    fn test_gate_blocks_low_trust() {
        let (dk, _, homeostasis, false_flow) = default_state();
        let trust = TrustQualia::uniform(0.1);
        let result = MulGate::check(&dk, &trust, &homeostasis, &false_flow, true);
        assert_eq!(result.unwrap_err(), GateBlockReason::TrustInsufficient);
    }
}
