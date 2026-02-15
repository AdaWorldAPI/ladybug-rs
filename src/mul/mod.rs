//! # Meta-Uncertainty Layer (MUL) — 10-Layer Metacognitive Stack
//!
//! The MUL is a metacognitive system that evaluates the system's OWN
//! epistemic state before allowing action. It answers: "How much should
//! I trust my own thinking right now?"
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────── Assessment (L1-L6) ──────────────────┐
//! │  L1 TrustQualia      — 4D felt-sense of knowing      │
//! │  L2 DKDetector        — Dunning-Kruger position       │
//! │  L3 Hysteresis        — State persistence (anti-thrash)│
//! │  L4 RiskVector        — Epistemic × Moral risk axes   │
//! │  L5 FalseFlowDetector — Coherence without progress    │
//! │  L6 CognitiveHomeostasis — Flow/Anxiety/Boredom/Apathy│
//! ├──────────────── Gate (L7) ───────────────────────────┤
//! │  L7 MulGate           — 5 blocking criteria (ALL must pass)│
//! ├──────────────── Navigation (L8-L10) ─────────────────┤
//! │  L8 FreeWillModifier  — Multiplicative confidence     │
//! │  L9 Compass           — 5 ethical/epistemic tests     │
//! │  L10 PostActionLearning — Convert outcomes to knowledge│
//! └─────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust
//! use ladybug::mul::{MetaUncertaintyLayer, MulSnapshot};
//!
//! let mut mul = MetaUncertaintyLayer::new();
//!
//! // Tick every cognitive cycle
//! mul.tick(0.7, 0.3, 0.5, 0.5); // coherence, novelty, challenge, skill
//!
//! // Evaluate before making decisions
//! let snapshot = mul.evaluate(true); // complexity_mapped = true
//! if snapshot.gate_open {
//!     // Safe to proceed
//!     println!("Free will modifier: {}", snapshot.free_will_modifier.value());
//! }
//! ```

pub mod compass;
pub mod dk_detector;
pub mod false_flow;
pub mod free_will_mod;
pub mod gate;
pub mod homeostasis;
pub mod hysteresis;
pub mod learning;
pub mod risk_vector;
pub mod trust_qualia;

// Re-export all public types
pub use compass::{CompassDecision, CompassResult};
pub use dk_detector::{DKDetector, DKPosition};
pub use false_flow::{FalseFlowDetector, FalseFlowSeverity};
pub use free_will_mod::FreeWillModifier;
pub use gate::{GateBlockReason, MulGate};
pub use homeostasis::{CognitiveHomeostasis, HomeostasisAction, HomeostasisState};
pub use hysteresis::{DwellConfig, TemporalHysteresis};
pub use learning::PostActionLearning;
pub use risk_vector::RiskVector;
pub use trust_qualia::{TrustComponent, TrustLevel, TrustQualia};

/// Snapshot of MUL state — the universal output that travels with every decision.
///
/// Consumers (n8n-rs, crewai-rust, ada-rs) receive this and use it to modulate
/// their own behavior. Pack/unpack to CogRecord W64-W65 for storage.
#[derive(Debug, Clone)]
pub struct MulSnapshot {
    /// Trust texture level (L1)
    pub trust_level: TrustLevel,
    /// Dunning-Kruger position (L2)
    pub dk_position: DKPosition,
    /// Cognitive homeostasis state (L6)
    pub homeostasis_state: HomeostasisState,
    /// False flow severity (L5)
    pub false_flow_severity: FalseFlowSeverity,
    /// Free will modifier (L8) — multiplicative confidence
    pub free_will_modifier: FreeWillModifier,
    /// Whether the MUL gate is open (L7)
    pub gate_open: bool,
    /// Why the gate blocked (if closed)
    pub gate_block_reason: Option<GateBlockReason>,
    /// Allostatic load (cumulative stress)
    pub allostatic_load: f32,
}

impl MulSnapshot {
    /// Whether the system should proceed with action.
    pub fn should_proceed(&self) -> bool {
        self.gate_open && self.free_will_modifier.value() > 0.3
    }

    /// Overall confidence level for downstream consumers.
    pub fn confidence(&self) -> f32 {
        if self.gate_open {
            self.free_will_modifier.value()
        } else {
            0.0
        }
    }

    /// Pack into 2 metadata words (W64-W65 of CogRecord).
    ///
    /// W64: `trust_level(3) | dk_position(2) | homeostasis(2) | false_flow(2) | gate(1) | reserved(54)`
    /// W65: `free_will_modifier_f32(32) | allostatic_load_f32(32)`
    pub fn pack(&self) -> [u64; 2] {
        let mut w64: u64 = 0;
        w64 |= (self.trust_level as u64) << 56;
        w64 |= (self.dk_position as u64) << 54;
        w64 |= (self.homeostasis_state as u64) << 52;
        w64 |= (self.false_flow_severity as u64) << 50;
        w64 |= if self.gate_open { 1u64 << 49 } else { 0 };

        let modifier_bits = self.free_will_modifier.value().to_bits() as u64;
        let load_bits = self.allostatic_load.to_bits() as u64;
        let w65 = (modifier_bits << 32) | load_bits;

        [w64, w65]
    }

    /// Unpack from 2 metadata words (W64-W65 of CogRecord).
    pub fn unpack(words: [u64; 2]) -> Self {
        let w64 = words[0];
        let w65 = words[1];

        let modifier_f32 = f32::from_bits(((w65 >> 32) & 0xFFFF_FFFF) as u32);
        let load_f32 = f32::from_bits((w65 & 0xFFFF_FFFF) as u32);

        Self {
            trust_level: TrustLevel::from_bits(((w64 >> 56) & 0x07) as u8),
            dk_position: DKPosition::from_bits(((w64 >> 54) & 0x03) as u8),
            homeostasis_state: HomeostasisState::from_bits(((w64 >> 52) & 0x03) as u8),
            false_flow_severity: FalseFlowSeverity::from_bits(((w64 >> 50) & 0x03) as u8),
            free_will_modifier: FreeWillModifier::from_value(modifier_f32),
            gate_open: (w64 >> 49) & 1 == 1,
            gate_block_reason: if (w64 >> 49) & 1 == 1 {
                None
            } else {
                Some(GateBlockReason::TrustInsufficient) // generic block on unpack
            },
            allostatic_load: load_f32,
        }
    }
}

/// The full MUL integrator — holds all 10 layers and orchestrates them.
///
/// Consumers call `evaluate()` to get the current metacognitive state,
/// `tick()` each cognitive cycle, and `learn()` after actions complete.
#[derive(Debug, Clone)]
pub struct MetaUncertaintyLayer {
    /// L1: Trust qualia — 4D felt-sense of knowing
    pub trust_qualia: TrustQualia,
    /// L2: Dunning-Kruger detector
    pub dk_detector: DKDetector,
    /// L3: Temporal hysteresis (anti-thrash)
    pub hysteresis: TemporalHysteresis,
    /// L4: Risk vector (epistemic × moral)
    pub risk_vector: RiskVector,
    /// L5: False flow detector
    pub false_flow: FalseFlowDetector,
    /// L6: Cognitive homeostasis
    pub homeostasis: CognitiveHomeostasis,
    // L7 (Gate) is stateless — just a function
    // L8 (FreeWillModifier) is computed on demand
    // L9 (Compass) is computed per-proposal
    // L10 (Learning) is applied post-action
    /// Monotonic tick counter
    tick_count: u64,
}

impl MetaUncertaintyLayer {
    pub fn new() -> Self {
        Self {
            trust_qualia: TrustQualia::default(),
            dk_detector: DKDetector::new(),
            hysteresis: TemporalHysteresis::new(),
            risk_vector: RiskVector::default(),
            false_flow: FalseFlowDetector::new(20),
            homeostasis: CognitiveHomeostasis::new(),
            tick_count: 0,
        }
    }

    /// Evaluate current metacognitive state. O(1) — reads current state.
    ///
    /// `complexity_mapped`: whether the problem space has been sufficiently explored.
    pub fn evaluate(&self, complexity_mapped: bool) -> MulSnapshot {
        let modifier = FreeWillModifier::compute(
            &self.dk_detector,
            &self.trust_qualia,
            &self.risk_vector,
            &self.homeostasis,
        );

        let gate_result = MulGate::check(
            &self.dk_detector,
            &self.trust_qualia,
            &self.homeostasis,
            &self.false_flow,
            complexity_mapped,
        );

        MulSnapshot {
            trust_level: self.trust_qualia.texture_level(),
            dk_position: self.dk_detector.position,
            homeostasis_state: self.homeostasis.state,
            false_flow_severity: self.false_flow.severity,
            free_will_modifier: modifier,
            gate_open: gate_result.is_ok(),
            gate_block_reason: gate_result.err(),
            allostatic_load: self.homeostasis.allostatic_load,
        }
    }

    /// Tick — called once per cognitive cycle.
    ///
    /// Updates false flow detection and homeostasis tracking.
    pub fn tick(&mut self, coherence: f32, novelty: f32, challenge: f32, skill: f32) {
        self.tick_count += 1;
        self.false_flow.tick(coherence, novelty);
        self.homeostasis.update(challenge, skill);
    }

    /// Post-action learning update.
    ///
    /// Called after an action completes to update trust and DK calibration.
    pub fn learn(&mut self, learning: &PostActionLearning) {
        // Update DK detector with prediction accuracy
        self.dk_detector.observe(
            learning.predicted_confidence,
            learning.actual_outcome > 0.5,
        );

        // Apply trust delta
        let delta = learning.trust_delta();
        if delta > 0 {
            // Well-calibrated → nudge trust up
            self.trust_qualia.competence =
                (self.trust_qualia.competence + 0.02).min(1.0);
            self.trust_qualia.calibration =
                (self.trust_qualia.calibration + 0.02).min(1.0);
        } else if delta < 0 {
            // Badly miscalibrated → nudge trust down
            self.trust_qualia.competence =
                (self.trust_qualia.competence - 0.05).max(0.0);
            self.trust_qualia.calibration =
                (self.trust_qualia.calibration - 0.05).max(0.0);
        }
    }

    /// Navigate a novel situation using the compass (L9).
    ///
    /// Returns both the compass decision and the free will modifier
    /// that was used to compute it.
    pub fn navigate(&self, compass: &CompassResult) -> (CompassDecision, f32) {
        let modifier = FreeWillModifier::compute(
            &self.dk_detector,
            &self.trust_qualia,
            &self.risk_vector,
            &self.homeostasis,
        );
        let decision = compass.decide(modifier.value());
        (decision, modifier.value())
    }

    /// Pack MUL state into CogRecord metadata words W64-W65.
    pub fn pack_to_metadata(&self, words: &mut [u64], offset: usize) {
        let snapshot = self.evaluate(true);
        let packed = snapshot.pack();
        if offset + 1 < words.len() {
            words[offset] = packed[0];
            words[offset + 1] = packed[1];
        }
    }

    /// Current tick count.
    pub fn tick_count(&self) -> u64 {
        self.tick_count
    }

    /// Update risk vector.
    pub fn set_risk(&mut self, epistemic: f32, moral: f32) {
        self.risk_vector = RiskVector::new(epistemic, moral);
    }

    /// Update trust from measurements.
    pub fn update_trust(
        &mut self,
        brier_score: f32,
        nars_confidence: f32,
        input_entropy: f32,
        calibration_error: f32,
    ) {
        self.trust_qualia =
            TrustQualia::from_measurements(brier_score, nars_confidence, input_entropy, calibration_error);
    }
}

impl Default for MetaUncertaintyLayer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_mul_evaluates() {
        let mul = MetaUncertaintyLayer::new();
        let snapshot = mul.evaluate(true);
        // Default state: Fuzzy trust, SlopeOfEnlightenment, Flow, no false flow
        assert_eq!(snapshot.trust_level, TrustLevel::Fuzzy);
        assert_eq!(snapshot.dk_position, DKPosition::SlopeOfEnlightenment);
        assert_eq!(snapshot.homeostasis_state, HomeostasisState::Flow);
        assert_eq!(snapshot.false_flow_severity, FalseFlowSeverity::None);
        assert!(snapshot.gate_open);
    }

    #[test]
    fn test_tick_updates_state() {
        let mut mul = MetaUncertaintyLayer::new();
        // Sustained anxiety: high challenge, low skill
        for _ in 0..20 {
            mul.tick(0.8, 0.1, 0.9, 0.2);
        }
        let snapshot = mul.evaluate(true);
        assert_eq!(snapshot.homeostasis_state, HomeostasisState::Anxiety);
        assert!(snapshot.allostatic_load > 0.0);
    }

    #[test]
    fn test_learn_updates_dk() {
        let mut mul = MetaUncertaintyLayer::new();
        // Feed many correct observations with high confidence
        for _ in 0..60 {
            let learning = PostActionLearning::new(
                CompassDecision::ExecuteWithLearning,
                0.8,
                0.85, // close to predicted → well-calibrated
            );
            mul.learn(&learning);
        }
        // DK should trend toward mastery
        assert!(mul.dk_detector.sample_count >= 60);
        assert!(mul.dk_detector.demonstrated_competence > 0.5);
    }

    #[test]
    fn test_gate_blocks_mount_stupid() {
        let mut mul = MetaUncertaintyLayer::new();
        mul.dk_detector.position = DKPosition::MountStupid;
        let snapshot = mul.evaluate(true);
        assert!(!snapshot.gate_open);
        assert_eq!(
            snapshot.gate_block_reason,
            Some(GateBlockReason::MountStupid)
        );
    }

    #[test]
    fn test_snapshot_pack_unpack_roundtrip() {
        let mul = MetaUncertaintyLayer::new();
        let snapshot = mul.evaluate(true);
        let packed = snapshot.pack();
        let unpacked = MulSnapshot::unpack(packed);

        assert_eq!(unpacked.trust_level, snapshot.trust_level);
        assert_eq!(unpacked.dk_position, snapshot.dk_position);
        assert_eq!(unpacked.homeostasis_state, snapshot.homeostasis_state);
        assert_eq!(unpacked.false_flow_severity, snapshot.false_flow_severity);
        assert_eq!(unpacked.gate_open, snapshot.gate_open);
        // f32 roundtrip should be exact (bit-level)
        assert!(
            (unpacked.free_will_modifier.value() - snapshot.free_will_modifier.value()).abs()
                < f32::EPSILON
        );
        assert!((unpacked.allostatic_load - snapshot.allostatic_load).abs() < f32::EPSILON);
    }

    #[test]
    fn test_navigate_with_compass() {
        // Use high-trust MUL so modifier is strong enough
        let mut mul = MetaUncertaintyLayer::new();
        mul.trust_qualia = TrustQualia::uniform(0.9);
        mul.risk_vector = RiskVector::low();
        let compass = CompassResult::routine();
        let (decision, modifier) = mul.navigate(&compass);
        assert_eq!(decision, CompassDecision::ExecuteWithLearning);
        assert!(modifier > 0.5);
    }

    #[test]
    fn test_should_proceed() {
        // Use high-trust MUL so modifier > 0.3
        let mut mul = MetaUncertaintyLayer::new();
        mul.trust_qualia = TrustQualia::uniform(0.9);
        mul.risk_vector = RiskVector::low();
        let snapshot = mul.evaluate(true);
        assert!(snapshot.should_proceed());

        // With gate blocked
        let mut mul2 = MetaUncertaintyLayer::new();
        mul2.dk_detector.position = DKPosition::MountStupid;
        let snapshot2 = mul2.evaluate(true);
        assert!(!snapshot2.should_proceed());
    }

    #[test]
    fn test_confidence_when_gate_closed() {
        let mut mul = MetaUncertaintyLayer::new();
        mul.dk_detector.position = DKPosition::MountStupid;
        let snapshot = mul.evaluate(true);
        assert_eq!(snapshot.confidence(), 0.0);
    }

    #[test]
    fn test_set_risk() {
        let mut mul = MetaUncertaintyLayer::new();
        mul.set_risk(0.8, 0.9);
        assert!((mul.risk_vector.epistemic - 0.8).abs() < 0.01);
        assert!((mul.risk_vector.moral - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_false_flow_escalation() {
        let mut mul = MetaUncertaintyLayer::new();
        // 25 ticks of high coherence, low novelty → severe false flow
        for _ in 0..25 {
            mul.tick(0.9, 0.01, 0.5, 0.5);
        }
        let snapshot = mul.evaluate(true);
        assert!(mul.false_flow.should_disrupt());
        assert!(!snapshot.gate_open);
        assert_eq!(
            snapshot.gate_block_reason,
            Some(GateBlockReason::FalseFlow)
        );
    }
}
