//! NARS Inference Rules
//!
//! Type-safe inference rules that can be composed and applied.

use crate::nars::TruthValue;

/// Trait for inference rules
pub trait InferenceRule {
    /// Apply the inference rule to two truth values
    fn apply(premise1: &TruthValue, premise2: &TruthValue) -> TruthValue;

    /// Name of the rule
    fn name() -> &'static str;

    /// Description of what this rule does
    fn description() -> &'static str;
}

/// Deduction: A→B, B→C ⊢ A→C
///
/// Forward chaining through implications.
/// "If birds fly, and Tweety is a bird, then Tweety flies."
pub struct Deduction;

impl InferenceRule for Deduction {
    fn apply(p1: &TruthValue, p2: &TruthValue) -> TruthValue {
        p1.deduction(p2)
    }

    fn name() -> &'static str {
        "deduction"
    }

    fn description() -> &'static str {
        "A→B, B→C ⊢ A→C (forward chaining)"
    }
}

/// Induction: A→B, A→C ⊢ B→C
///
/// Generalizing from shared premise.
/// "Ravens are black, ravens are birds → birds are black."
pub struct Induction;

impl InferenceRule for Induction {
    fn apply(p1: &TruthValue, p2: &TruthValue) -> TruthValue {
        p1.induction(p2)
    }

    fn name() -> &'static str {
        "induction"
    }

    fn description() -> &'static str {
        "A→B, A→C ⊢ B→C (generalization)"
    }
}

/// Abduction: A→B, C→B ⊢ A→C
///
/// Inferring cause from shared effect.
/// "Rain causes wet grass, sprinklers cause wet grass → rain ~ sprinklers."
pub struct Abduction;

impl InferenceRule for Abduction {
    fn apply(p1: &TruthValue, p2: &TruthValue) -> TruthValue {
        p1.abduction(p2)
    }

    fn name() -> &'static str {
        "abduction"
    }

    fn description() -> &'static str {
        "A→B, C→B ⊢ A→C (cause inference)"
    }
}

/// Analogy: A→B, A↔C ⊢ C→B
///
/// Transferring relation via similarity.
/// "Cats have fur, cats are similar to dogs → dogs have fur."
pub struct Analogy;

impl InferenceRule for Analogy {
    fn apply(p1: &TruthValue, p2: &TruthValue) -> TruthValue {
        p1.analogy(p2)
    }

    fn name() -> &'static str {
        "analogy"
    }

    fn description() -> &'static str {
        "A→B, A↔C ⊢ C→B (similarity transfer)"
    }
}

/// Comparison: A→B, C→B ⊢ A↔C
///
/// Inferring similarity from shared property.
pub struct Comparison;

impl InferenceRule for Comparison {
    fn apply(p1: &TruthValue, p2: &TruthValue) -> TruthValue {
        p1.comparison(p2)
    }

    fn name() -> &'static str {
        "comparison"
    }

    fn description() -> &'static str {
        "A→B, C→B ⊢ A↔C (similarity from shared property)"
    }
}

/// Apply inference rule by name
pub fn apply_rule(
    rule_name: &str,
    premise1: &TruthValue,
    premise2: &TruthValue,
) -> Option<TruthValue> {
    match rule_name {
        "deduction" => Some(Deduction::apply(premise1, premise2)),
        "induction" => Some(Induction::apply(premise1, premise2)),
        "abduction" => Some(Abduction::apply(premise1, premise2)),
        "analogy" => Some(Analogy::apply(premise1, premise2)),
        "comparison" => Some(Comparison::apply(premise1, premise2)),
        _ => None,
    }
}

/// All available inference rules
pub const INFERENCE_RULES: &[&str] = &[
    "deduction",
    "induction",
    "abduction",
    "analogy",
    "comparison",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deduction_reduces_confidence() {
        let p1 = TruthValue::new(0.9, 0.9);
        let p2 = TruthValue::new(0.9, 0.9);

        let conclusion = Deduction::apply(&p1, &p2);

        // Chained inference should reduce confidence
        assert!(conclusion.confidence < p1.confidence);
        assert!(conclusion.confidence < p2.confidence);
    }

    #[test]
    fn test_rule_dispatch() {
        let p1 = TruthValue::new(0.8, 0.8);
        let p2 = TruthValue::new(0.7, 0.7);

        let result = apply_rule("deduction", &p1, &p2);
        assert!(result.is_some());

        let result = apply_rule("invalid_rule", &p1, &p2);
        assert!(result.is_none());
    }
}
