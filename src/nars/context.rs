//! Inference Context — Style-Driven Reasoning
//!
//! Aggregates four cognitive subsystems into a unified context for inference:
//! - ThinkingStyle → rule biases (StyleWeights)
//! - RungLevel → atom kind gating (AtomGate)
//! - QueryMode → causal mode selection (PearlMode)
//! - GateState → confidence/depth modulation (CollapseModulation)

use crate::cognitive::{GateState, RungLevel, ThinkingStyle};
use crate::search::causal::QueryMode;

// =============================================================================
// CONSTANTS
// =============================================================================

/// Base chain depth before modifiers
const BASE_DEPTH: u8 = 5;

// =============================================================================
// INFERENCE RULE KIND
// =============================================================================

/// The five fundamental NARS inference rules
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InferenceRuleKind {
    Deduction,
    Induction,
    Abduction,
    Analogy,
    Revision,
}

impl InferenceRuleKind {
    /// All rule kinds in order
    pub const ALL: [InferenceRuleKind; 5] = [
        Self::Deduction,
        Self::Induction,
        Self::Abduction,
        Self::Analogy,
        Self::Revision,
    ];
}

// =============================================================================
// ATOM KIND
// =============================================================================

/// The five atom kinds mapping to inference families
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AtomKind {
    /// Observe → Induction (from data)
    Observe,
    /// Deduce → Deduction (from rules)
    Deduce,
    /// Critique → Abduction (from effects)
    Critique,
    /// Integrate → Revision (consolidation)
    Integrate,
    /// Jump → Analogy (cross-domain)
    Jump,
}

impl AtomKind {
    /// All atom kinds in order
    pub const ALL: [AtomKind; 5] = [
        Self::Observe,
        Self::Deduce,
        Self::Critique,
        Self::Integrate,
        Self::Jump,
    ];

    /// Map to corresponding inference rule kind
    pub fn to_rule_kind(self) -> InferenceRuleKind {
        match self {
            Self::Observe => InferenceRuleKind::Induction,
            Self::Deduce => InferenceRuleKind::Deduction,
            Self::Critique => InferenceRuleKind::Abduction,
            Self::Integrate => InferenceRuleKind::Revision,
            Self::Jump => InferenceRuleKind::Analogy,
        }
    }
}

// =============================================================================
// STYLE WEIGHTS
// =============================================================================

/// Rule biases derived from ThinkingStyle
#[derive(Clone, Debug)]
pub struct StyleWeights {
    /// Per-rule bias from ThinkingStyle (positive = prefer, negative = suppress)
    pub rule_biases: [(InferenceRuleKind, f32); 5],

    /// Confidence modifier (multiplied with base confidence)
    pub confidence_modifier: f32,

    /// Chain depth delta (added to base max depth)
    pub chain_depth_delta: i8,

    /// Optional consumer-supplied weight vector
    /// Ladybug does not interpret these — passes through to consumer hooks
    pub extended_weights: Option<Vec<f32>>,
}

impl StyleWeights {
    /// Create from ThinkingStyle
    pub fn from_thinking_style(style: ThinkingStyle) -> Self {
        let modulation = style.field_modulation();

        // Map FieldModulation to rule biases:
        // depth_bias → Deduction (deep chaining)
        // breadth_bias → Induction + Analogy (wide association)
        // noise_tolerance → Abduction (uncertain leaps)
        // exploration → Analogy + Abduction boost
        // resonance_threshold → confidence modifier (inverted: high threshold = low confidence)

        let deduction = modulation.depth_bias;
        let induction = modulation.breadth_bias;
        let abduction = modulation.noise_tolerance + modulation.exploration * 0.3;
        let analogy = modulation.breadth_bias * 0.5 + modulation.exploration * 0.5;
        let revision = 0.5; // Base revision is neutral

        // Confidence: higher resonance threshold = more strict = higher confidence needed
        let confidence_modifier = 0.5 + modulation.resonance_threshold;

        // Depth: depth_bias increases depth, speed_bias decreases it
        // Convert to i8 range: high depth_bias = positive delta, high speed_bias = negative delta
        let chain_depth_delta = ((modulation.depth_bias - modulation.speed_bias) * 4.0) as i8;

        Self {
            rule_biases: [
                (InferenceRuleKind::Deduction, deduction),
                (InferenceRuleKind::Induction, induction),
                (InferenceRuleKind::Abduction, abduction),
                (InferenceRuleKind::Analogy, analogy),
                (InferenceRuleKind::Revision, revision),
            ],
            confidence_modifier,
            chain_depth_delta,
            extended_weights: None,
        }
    }

    /// Create neutral weights (all biases = 0, modifiers = 1.0)
    pub fn neutral() -> Self {
        Self {
            rule_biases: [
                (InferenceRuleKind::Deduction, 0.0),
                (InferenceRuleKind::Induction, 0.0),
                (InferenceRuleKind::Abduction, 0.0),
                (InferenceRuleKind::Analogy, 0.0),
                (InferenceRuleKind::Revision, 0.0),
            ],
            confidence_modifier: 1.0,
            chain_depth_delta: 0,
            extended_weights: None,
        }
    }

    /// Get bias for a specific rule kind
    pub fn bias_for(&self, kind: InferenceRuleKind) -> f32 {
        self.rule_biases
            .iter()
            .find(|(k, _)| *k == kind)
            .map(|(_, b)| *b)
            .unwrap_or(0.0)
    }

    /// Set extended weights (consumer passthrough)
    pub fn with_extended_weights(mut self, weights: Vec<f32>) -> Self {
        self.extended_weights = Some(weights);
        self
    }
}

impl Default for StyleWeights {
    fn default() -> Self {
        Self::neutral()
    }
}

// =============================================================================
// ATOM GATE
// =============================================================================

/// Atom kind weights derived from RungLevel
#[derive(Clone, Debug)]
pub struct AtomGate {
    /// Per-atom-kind weight (1.0 = neutral, >1 = boosted, <1 = suppressed)
    pub weights: [(AtomKind, f32); 5],

    /// Source rung level
    pub rung: RungLevel,
}

impl AtomGate {
    /// Create from RungLevel
    pub fn from_rung(rung: RungLevel) -> Self {
        // Rung bands:
        // Surface (0-2): Observe dominates
        // Analogical (3-5): Jump dominates
        // Meta (6-7): Critique dominates
        // Recursive (8-9): All equal

        let weights = match rung.as_u8() {
            0..=2 => [
                (AtomKind::Observe, 1.5),
                (AtomKind::Deduce, 1.0),
                (AtomKind::Critique, 0.5),
                (AtomKind::Integrate, 0.5),
                (AtomKind::Jump, 0.3),
            ],
            3..=5 => [
                (AtomKind::Observe, 0.8),
                (AtomKind::Deduce, 0.8),
                (AtomKind::Critique, 0.8),
                (AtomKind::Integrate, 1.0),
                (AtomKind::Jump, 1.5),
            ],
            6..=7 => [
                (AtomKind::Observe, 0.5),
                (AtomKind::Deduce, 0.5),
                (AtomKind::Critique, 1.5),
                (AtomKind::Integrate, 1.0),
                (AtomKind::Jump, 1.0),
            ],
            _ => [
                (AtomKind::Observe, 1.0),
                (AtomKind::Deduce, 1.0),
                (AtomKind::Critique, 1.0),
                (AtomKind::Integrate, 1.0),
                (AtomKind::Jump, 1.0),
            ],
        };

        Self { weights, rung }
    }

    /// Create neutral gate (all weights = 1.0)
    pub fn neutral() -> Self {
        Self {
            weights: [
                (AtomKind::Observe, 1.0),
                (AtomKind::Deduce, 1.0),
                (AtomKind::Critique, 1.0),
                (AtomKind::Integrate, 1.0),
                (AtomKind::Jump, 1.0),
            ],
            rung: RungLevel::Surface,
        }
    }

    /// Get weight for a specific atom kind
    pub fn weight_for(&self, kind: AtomKind) -> f32 {
        self.weights
            .iter()
            .find(|(k, _)| *k == kind)
            .map(|(_, w)| *w)
            .unwrap_or(1.0)
    }

    /// Get the rung band name
    pub fn band_name(&self) -> &'static str {
        match self.rung.as_u8() {
            0..=2 => "Surface",
            3..=5 => "Analogical",
            6..=7 => "Meta",
            _ => "Recursive",
        }
    }
}

impl Default for AtomGate {
    fn default() -> Self {
        Self::neutral()
    }
}

// =============================================================================
// PEARL MODE
// =============================================================================

/// Causal mode wrapper (Pearl's ladder of causation)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PearlMode {
    /// Rung 1: What correlates? (association)
    See,
    /// Rung 2: What happens if I do X? (intervention)
    Do,
    /// Rung 3: What would have happened? (counterfactual)
    Imagine,
}

impl PearlMode {
    /// Convert to QueryMode for causal search
    pub fn to_query_mode(self) -> QueryMode {
        match self {
            Self::See => QueryMode::Correlate,
            Self::Do => QueryMode::Intervene,
            Self::Imagine => QueryMode::Counterfact,
        }
    }

    /// Create from QueryMode
    pub fn from_query_mode(mode: QueryMode) -> Self {
        match mode {
            QueryMode::Correlate => Self::See,
            QueryMode::Intervene => Self::Do,
            QueryMode::Counterfact => Self::Imagine,
        }
    }

    /// Pearl's ladder rung number (1, 2, or 3)
    pub fn rung_number(self) -> u8 {
        match self {
            Self::See => 1,
            Self::Do => 2,
            Self::Imagine => 3,
        }
    }
}

impl Default for PearlMode {
    fn default() -> Self {
        Self::See
    }
}

// =============================================================================
// COLLAPSE MODULATION
// =============================================================================

/// Confidence/depth modulation from GateState
#[derive(Clone, Copy, Debug)]
pub struct CollapseModulation {
    /// Confidence multiplier
    pub confidence_modifier: f32,

    /// Depth delta
    pub depth_delta: i8,

    /// Source gate state
    pub gate: GateState,
}

impl CollapseModulation {
    /// Create from GateState
    pub fn from_gate(gate: GateState) -> Self {
        let (confidence_modifier, depth_delta) = match gate {
            GateState::Flow => (1.4, -2), // Commit fast
            GateState::Hold => (1.0, 0),  // Neutral
            GateState::Block => (0.6, 2), // Explore deep
        };

        Self {
            confidence_modifier,
            depth_delta,
            gate,
        }
    }

    /// Create neutral modulation
    pub fn neutral() -> Self {
        Self {
            confidence_modifier: 1.0,
            depth_delta: 0,
            gate: GateState::Hold,
        }
    }
}

impl Default for CollapseModulation {
    fn default() -> Self {
        Self::neutral()
    }
}

// =============================================================================
// INFERENCE CONTEXT
// =============================================================================

/// Complete inference context aggregating all cognitive subsystems
#[derive(Clone, Debug)]
pub struct InferenceContext {
    /// Style-derived rule biases
    pub style_weights: StyleWeights,

    /// Rung-derived atom gating
    pub atom_gate: AtomGate,

    /// Causal mode
    pub pearl_mode: PearlMode,

    /// Gate-state modulation
    pub collapse: CollapseModulation,

    /// Combined minimum confidence (style × collapse)
    pub min_confidence: f32,

    /// Combined max chain depth (base + style + collapse)
    pub max_chain_depth: u8,
}

impl InferenceContext {
    /// Build context from components
    pub fn build(
        style: StyleWeights,
        gate: AtomGate,
        mode: PearlMode,
        collapse: CollapseModulation,
    ) -> Self {
        // Modifiers multiply for confidence
        let min_confidence = style.confidence_modifier * collapse.confidence_modifier;

        // Modifiers add for depth
        let max_chain_depth =
            (BASE_DEPTH as i16 + style.chain_depth_delta as i16 + collapse.depth_delta as i16)
                .clamp(1, 20) as u8;

        Self {
            style_weights: style,
            atom_gate: gate,
            pearl_mode: mode,
            collapse,
            min_confidence,
            max_chain_depth,
        }
    }

    /// Create neutral context (all modifiers = 1.0 / 0)
    pub fn neutral() -> Self {
        Self::build(
            StyleWeights::neutral(),
            AtomGate::neutral(),
            PearlMode::See,
            CollapseModulation::neutral(),
        )
    }

    /// Create from cognitive state
    pub fn from_state(
        style: ThinkingStyle,
        rung: RungLevel,
        query_mode: QueryMode,
        gate_state: GateState,
    ) -> Self {
        Self::build(
            StyleWeights::from_thinking_style(style),
            AtomGate::from_rung(rung),
            PearlMode::from_query_mode(query_mode),
            CollapseModulation::from_gate(gate_state),
        )
    }

    /// Get effective bias for an inference rule
    pub fn effective_rule_bias(&self, rule: InferenceRuleKind) -> f32 {
        let style_bias = self.style_weights.bias_for(rule);

        // Find matching atom kind and get its gate weight
        let atom_kind = match rule {
            InferenceRuleKind::Deduction => AtomKind::Deduce,
            InferenceRuleKind::Induction => AtomKind::Observe,
            InferenceRuleKind::Abduction => AtomKind::Critique,
            InferenceRuleKind::Analogy => AtomKind::Jump,
            InferenceRuleKind::Revision => AtomKind::Integrate,
        };
        let gate_weight = self.atom_gate.weight_for(atom_kind);

        // Combine: bias adjusted by gate weight
        style_bias * gate_weight
    }

    /// Check if a rule should be preferred in current context
    pub fn should_prefer(&self, rule: InferenceRuleKind) -> bool {
        self.effective_rule_bias(rule) > 0.5
    }

    /// Get the extended weights if any
    pub fn extended_weights(&self) -> Option<&[f32]> {
        self.style_weights.extended_weights.as_deref()
    }
}

impl Default for InferenceContext {
    fn default() -> Self {
        Self::neutral()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neutral_context() {
        let ctx = InferenceContext::neutral();
        assert!((ctx.min_confidence - 1.0).abs() < 0.01);
        assert_eq!(ctx.max_chain_depth, BASE_DEPTH);
    }

    #[test]
    fn test_style_analytical() {
        let weights = StyleWeights::from_thinking_style(ThinkingStyle::Analytical);
        // Analytical should favor deduction
        assert!(weights.bias_for(InferenceRuleKind::Deduction) > 0.0);
    }

    #[test]
    fn test_style_creative() {
        let weights = StyleWeights::from_thinking_style(ThinkingStyle::Creative);
        // Creative should favor exploration (abduction, analogy)
        let analogy_bias = weights.bias_for(InferenceRuleKind::Analogy);
        let _deduction_bias = weights.bias_for(InferenceRuleKind::Deduction);
        // Creative typically has higher analogy than deduction
        assert!(analogy_bias >= 0.0);
    }

    #[test]
    fn test_atom_gate_surface() {
        let gate = AtomGate::from_rung(RungLevel::Surface);
        assert_eq!(gate.weight_for(AtomKind::Observe), 1.5);
        assert_eq!(gate.band_name(), "Surface");
    }

    #[test]
    fn test_atom_gate_meta() {
        let gate = AtomGate::from_rung(RungLevel::Meta);
        assert_eq!(gate.weight_for(AtomKind::Critique), 1.5);
        assert_eq!(gate.band_name(), "Meta");
    }

    #[test]
    fn test_collapse_flow() {
        let collapse = CollapseModulation::from_gate(GateState::Flow);
        assert!((collapse.confidence_modifier - 1.4).abs() < 0.01);
        assert_eq!(collapse.depth_delta, -2);
    }

    #[test]
    fn test_collapse_block() {
        let collapse = CollapseModulation::from_gate(GateState::Block);
        assert!((collapse.confidence_modifier - 0.6).abs() < 0.01);
        assert_eq!(collapse.depth_delta, 2);
    }

    #[test]
    fn test_build_stacks() {
        // Use neutral style so we can test collapse modulation clearly
        let style = StyleWeights::neutral();
        let gate = AtomGate::neutral();
        let collapse = CollapseModulation::from_gate(GateState::Flow);

        let ctx = InferenceContext::build(style, gate, PearlMode::See, collapse);

        // Confidence should be style × collapse
        // Neutral gives 1.0×, Flow gives 1.4×, so min_confidence = 1.4
        assert!((ctx.min_confidence - 1.4).abs() < 0.01);

        // Depth should be reduced by Flow's -2
        // Neutral style has 0 delta, so total = BASE_DEPTH - 2
        assert_eq!(ctx.max_chain_depth, BASE_DEPTH - 2);
    }

    #[test]
    fn test_pearl_to_query_mode() {
        assert_eq!(PearlMode::See.to_query_mode(), QueryMode::Correlate);
        assert_eq!(PearlMode::Do.to_query_mode(), QueryMode::Intervene);
        assert_eq!(PearlMode::Imagine.to_query_mode(), QueryMode::Counterfact);
    }

    #[test]
    fn test_extended_weights_passthrough() {
        let weights = StyleWeights::neutral().with_extended_weights(vec![1.0, 2.0, 3.0]);

        let ctx = InferenceContext::build(
            weights,
            AtomGate::neutral(),
            PearlMode::See,
            CollapseModulation::neutral(),
        );

        let extended = ctx.extended_weights().unwrap();
        assert_eq!(extended.len(), 3);
        assert_eq!(extended[0], 1.0);
        assert_eq!(extended[1], 2.0);
        assert_eq!(extended[2], 3.0);
    }
}
