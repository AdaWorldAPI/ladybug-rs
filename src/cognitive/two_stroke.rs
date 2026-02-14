//! 2-Stroke Cognitive Engine
//!
//! All layers fire every cycle against shared metadata from the previous cycle.
//! No sequential wave dependency — L8 Integration starts forming a partial
//! picture from L5's last-cycle results while L6 Delegation is still fanning out.
//!
//! Three implicit mechanisms replace all explicit dispatch:
//!
//! 1. **Resonance-gated layer activation**: each layer fires only if its
//!    resonance exceeds the effective threshold (satisfaction gate + style).
//!
//! 2. **Resonance-gated style selection**: the system's satisfaction state
//!    is a fingerprint; ThinkingStyles are fingerprints; the most resonant
//!    style activates. No orchestrator decides. The field modulates itself.
//!
//! 3. **Crystallization as implicit learning**: L10 writes surviving results
//!    to BindSpace as bound fingerprints. Future inputs resonate with them.
//!    The crystallized fingerprint IS the learned rule.
//!
//! ```text
//! ┌──────────── CYCLE N ────────────┐  ┌──────────── CYCLE N+1 ──────────┐
//! │ L1-L10 read prev_scores[N-1]    │  │ L1-L10 read prev_scores[N]      │
//! │ Each fires if resonance > thresh │  │ L8 integrates L5's N output     │
//! │ Write new scores to current      │  │ while L6 fans out with N+1 input│
//! └─────────────────────────────────┘  └──────────────────────────────────┘
//! ```

use super::layer_stack::{
    LayerId, LayerNode, LayerResult, NUM_LAYERS, apply_layer_result, process_layer,
};
use super::satisfaction_gate::LayerSatisfaction;
use super::style::{FieldModulation, ThinkingStyle};
use crate::core::Fingerprint;

/// Decay factor for score blending across cycles.
/// Higher = more inertia (slower to change). 0.7 = 70% old, 30% new.
const DECAY: f32 = 0.7;

// =============================================================================
// 2-STROKE ENGINE
// =============================================================================

/// Process all layers in 2-stroke mode: read previous-cycle state, fire if
/// resonance exceeds effective threshold (satisfaction gate + style modulation).
///
/// Returns layers that fired this cycle.
pub fn process_2stroke(
    node: &mut LayerNode,
    input: &Fingerprint,
    satisfaction: &mut LayerSatisfaction,
    prev_scores: &[f32; NUM_LAYERS],
    modulation: &FieldModulation,
    cycle: u64,
) -> Vec<LayerResult> {
    let mut results = Vec::with_capacity(NUM_LAYERS);

    for layer in LayerId::ALL {
        // Threshold = max(satisfaction gate, style resonance threshold)
        // The satisfaction gate raises it if lower layers are weak.
        // The style threshold provides a floor — analytical agents
        // never go below 0.85 even if all lower layers are satisfied.
        let satisfaction_threshold =
            satisfaction.effective_threshold_async(layer, prev_scores);
        let effective_threshold = satisfaction_threshold.max(modulation.resonance_threshold);

        // Resonance: how well does this input match this layer's pattern?
        let resonance = input.similarity(&node.vsa_core);

        if resonance >= effective_threshold {
            let result = process_layer(node, layer, input, cycle);
            apply_layer_result(node, &result);

            // Decay + blend: accumulate across cycles
            let new_score = prev_scores[layer.index()] * DECAY
                + result.output_activation * (1.0 - DECAY);
            satisfaction.update(layer, new_score);

            results.push(result);
        } else {
            // Layer didn't fire — mark as gated
            let marker = node.marker_mut(layer);
            marker.flags |= super::layer_stack::LayerMarker::FLAG_GATED;
        }
    }

    results
}

/// Copy current satisfaction scores to a fixed array for next cycle's prev_scores.
pub fn snapshot_scores(satisfaction: &LayerSatisfaction) -> [f32; NUM_LAYERS] {
    satisfaction.scores
}

// =============================================================================
// RESONANCE-GATED STYLE SELECTION
// =============================================================================

/// Precomputed fingerprint for a ThinkingStyle.
///
/// Created once, reused for resonance comparison.
pub struct StyleFingerprint {
    pub style: ThinkingStyle,
    pub fingerprint: Fingerprint,
}

/// Build fingerprints for all 12 thinking styles.
///
/// Each style's fingerprint encodes its FieldModulation parameters
/// as a content string. Structurally similar modulations produce
/// similar fingerprints — this is what makes resonance-based
/// selection work.
pub fn build_style_fingerprints() -> Vec<StyleFingerprint> {
    ThinkingStyle::ALL
        .iter()
        .map(|&style| {
            let m = style.field_modulation();
            // Encode modulation parameters into a deterministic string
            let encoded = format!(
                "style:{}|threshold:{:.2}|fan_out:{}|depth:{:.2}|breadth:{:.2}|noise:{:.2}|speed:{:.2}|explore:{:.2}",
                style, m.resonance_threshold, m.fan_out, m.depth_bias,
                m.breadth_bias, m.noise_tolerance, m.speed_bias, m.exploration
            );
            StyleFingerprint {
                style,
                fingerprint: Fingerprint::from_content(&encoded),
            }
        })
        .collect()
}

/// Select thinking style by resonance with current cognitive state.
///
/// The system's satisfaction state IS a fingerprint. Each ThinkingStyle
/// IS a fingerprint. The most resonant style activates.
///
/// When satisfaction is high (system grounded), the state fingerprint
/// resonates with Analytical/Focused. When lower layers are shaky,
/// it resonates with Exploratory/Creative. The system doesn't "decide"
/// to explore — exploration emerges from the resonance.
pub fn select_style_by_resonance(
    satisfaction: &LayerSatisfaction,
    style_fps: &[StyleFingerprint],
) -> ThinkingStyle {
    let state_fp = satisfaction.to_fingerprint();

    style_fps
        .iter()
        .max_by(|a, b| {
            let sim_a = state_fp.similarity(&a.fingerprint);
            let sim_b = state_fp.similarity(&b.fingerprint);
            sim_a.partial_cmp(&sim_b).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|sf| sf.style)
        .unwrap_or(ThinkingStyle::Deliberate)
}

// =============================================================================
// RESONANCE-GATED INFERENCE RULE SELECTION
// =============================================================================

/// A NARS inference rule with its fingerprint.
pub struct RuleFingerprint {
    pub name: &'static str,
    pub fingerprint: Fingerprint,
}

/// Build fingerprints for NARS inference rules.
///
/// Each rule's fingerprint is generated from its name. The gestalt
/// superposition from the cognitive state will naturally resonate
/// more with certain rules depending on the triangle position:
/// - High coherence → deduction-like fingerprints
/// - High novelty → abduction-like fingerprints
/// - Balanced → analogy/induction
pub fn build_rule_fingerprints() -> Vec<RuleFingerprint> {
    // These must match the rule names accepted by nars::inference::apply_rule
    ["deduction", "induction", "abduction", "analogy", "revision"]
        .iter()
        .map(|&name| RuleFingerprint {
            name,
            fingerprint: Fingerprint::from_content(name),
        })
        .collect()
}

/// Select inference rules by resonance with the gestalt superposition.
///
/// Instead of `apply_rule("deduction", ...)`, the gestalt fingerprint
/// resonates with each rule's fingerprint. Rules that cross the threshold
/// fire. fan_out caps how many.
///
/// This is the core "implicit thinking adaptation": no if/else dispatch.
/// The resonance algebra IS the routing.
pub fn select_rules_by_resonance(
    gestalt: &Fingerprint,
    rule_fps: &[RuleFingerprint],
    modulation: &FieldModulation,
) -> Vec<&'static str> {
    let mut scored: Vec<_> = rule_fps
        .iter()
        .map(|rf| (rf.name, gestalt.similarity(&rf.fingerprint)))
        .filter(|(_, sim)| *sim >= modulation.resonance_threshold)
        .collect();

    // Sort by resonance score descending
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Cap by fan_out
    scored.truncate(modulation.fan_out);

    // If nothing crossed threshold, fall back to the single strongest
    if scored.is_empty() {
        if let Some(best) = rule_fps
            .iter()
            .max_by(|a, b| {
                let sa = gestalt.similarity(&a.fingerprint);
                let sb = gestalt.similarity(&b.fingerprint);
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            })
        {
            return vec![best.name];
        }
    }

    scored.into_iter().map(|(name, _)| name).collect()
}

// =============================================================================
// CRYSTALLIZATION FEEDBACK
// =============================================================================

/// Crystallize a validated result back into the resonance field.
///
/// The content fingerprint gets bound with the modulation that produced it.
/// Next cycle, when L2 searches, this crystallized fingerprint is in the index.
/// L4 can unbind it to recover what modulation worked. No rule table. No
/// explicit learning step. The fingerprint IS the learned rule.
pub fn crystallize(content_fp: &Fingerprint, modulation: &FieldModulation) -> Fingerprint {
    let modulation_fp = modulation_to_fingerprint(modulation);
    content_fp.bind(&modulation_fp)
}

/// Recover the modulation fingerprint from a crystallized result.
///
/// Since XOR is self-inverse: crystal.bind(content) = modulation.
/// The recovered fingerprint can then be compared against ThinkingStyle
/// fingerprints to find which style produced the original result.
pub fn recover_modulation(crystal_fp: &Fingerprint, content_fp: &Fingerprint) -> Fingerprint {
    crystal_fp.bind(content_fp) // XOR is self-inverse
}

/// Convert FieldModulation to a fingerprint for crystallization binding.
fn modulation_to_fingerprint(m: &FieldModulation) -> Fingerprint {
    let encoded = format!(
        "mod|t:{:.2}|f:{}|d:{:.2}|b:{:.2}|n:{:.2}|s:{:.2}|e:{:.2}",
        m.resonance_threshold,
        m.fan_out,
        m.depth_bias,
        m.breadth_bias,
        m.noise_tolerance,
        m.speed_bias,
        m.exploration
    );
    Fingerprint::from_content(&encoded)
}

// =============================================================================
// L9 VALIDATION RESULT
// =============================================================================

/// Result of L9 Validation — three orthogonal checks.
#[derive(Clone, Debug)]
pub enum ValidationResult {
    /// All checks passed — ready for L10 crystallization
    Pass {
        /// Merged NARS truth confidence
        nars_confidence: f32,
        /// MetaCognition meta-confidence
        meta_confidence: f32,
    },
    /// Truth hardened but calibration poor — gather more outcomes
    Hold {
        reason: &'static str,
    },
    /// Dunning-Kruger or insufficient evidence
    Reject {
        reason: &'static str,
    },
}

/// L9 Validation: triple check before anything reaches L10.
///
/// 1. NARS: is the merged truth value strong enough?
/// 2. Calibration: is our confidence estimate reliable?
/// 3. Dunning-Kruger: high confidence + high uncertainty = suspicious
pub fn validate_l9(
    nars_confidence: f32,
    nars_frequency: f32,
    calibration_error: f32,
    should_admit_ignorance: bool,
    uncertainty_score: f32,
) -> ValidationResult {
    // 1. NARS: is the truth value strong enough?
    let nars_pass = nars_confidence > 0.5 && nars_frequency > 0.3;

    // 2. Calibration: is our confidence reliable?
    let calibrated = !should_admit_ignorance && calibration_error < 0.2;

    // 3. Dunning-Kruger: high confidence + high uncertainty = suspicious
    let dunning_kruger = nars_confidence > 0.8 && uncertainty_score > 0.4;

    if dunning_kruger {
        return ValidationResult::Reject {
            reason: "Dunning-Kruger: high confidence with high cross-branch disagreement",
        };
    }

    if nars_pass && calibrated {
        ValidationResult::Pass {
            nars_confidence,
            meta_confidence: 1.0 - calibration_error,
        }
    } else if nars_pass && !calibrated {
        ValidationResult::Hold {
            reason: "Truth hardened but calibration poor — gather more outcomes",
        }
    } else {
        ValidationResult::Reject {
            reason: "Insufficient evidence or agreement",
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_style_fingerprints_deterministic() {
        let fps1 = build_style_fingerprints();
        let fps2 = build_style_fingerprints();

        for (a, b) in fps1.iter().zip(fps2.iter()) {
            assert_eq!(a.style, b.style);
            assert_eq!(a.fingerprint.similarity(&b.fingerprint), 1.0);
        }
    }

    #[test]
    fn test_style_fingerprints_distinct() {
        let fps = build_style_fingerprints();
        // Each style should produce a unique fingerprint
        for i in 0..fps.len() {
            for j in (i + 1)..fps.len() {
                let sim = fps[i].fingerprint.similarity(&fps[j].fingerprint);
                assert!(
                    sim < 1.0,
                    "Styles {:?} and {:?} should have distinct fingerprints (sim={})",
                    fps[i].style,
                    fps[j].style,
                    sim
                );
            }
        }
    }

    #[test]
    fn test_select_style_returns_something() {
        let sat = LayerSatisfaction::new();
        let fps = build_style_fingerprints();
        let style = select_style_by_resonance(&sat, &fps);
        // Should return one of the 12 styles
        assert!(ThinkingStyle::ALL.contains(&style));
    }

    #[test]
    fn test_rule_fingerprints() {
        let fps = build_rule_fingerprints();
        assert!(fps.len() >= 4); // At least deduction, induction, abduction, analogy
    }

    #[test]
    fn test_select_rules_always_returns_something() {
        let gestalt = Fingerprint::from_content("test gestalt");
        let rule_fps = build_rule_fingerprints();
        let modulation = ThinkingStyle::Analytical.field_modulation();

        let rules = select_rules_by_resonance(&gestalt, &rule_fps, &modulation);
        assert!(!rules.is_empty(), "Should always select at least one rule");
    }

    #[test]
    fn test_creative_selects_more_rules() {
        let gestalt = Fingerprint::from_content("exploratory input");
        let rule_fps = build_rule_fingerprints();

        let analytical = ThinkingStyle::Analytical.field_modulation();
        let creative = ThinkingStyle::Creative.field_modulation();

        let analytical_rules = select_rules_by_resonance(&gestalt, &rule_fps, &analytical);
        let creative_rules = select_rules_by_resonance(&gestalt, &rule_fps, &creative);

        // Creative (threshold=0.35, fan_out=12) should select >= analytical (threshold=0.85, fan_out=3)
        assert!(
            creative_rules.len() >= analytical_rules.len(),
            "Creative ({}) should select >= Analytical ({}) rules",
            creative_rules.len(),
            analytical_rules.len()
        );
    }

    #[test]
    fn test_crystallize_and_recover() {
        let content = Fingerprint::from_content("important result");
        let modulation = ThinkingStyle::Analytical.field_modulation();

        let crystal = crystallize(&content, &modulation);
        let recovered = recover_modulation(&crystal, &content);

        // recovered should be the modulation fingerprint
        let original_mod_fp = modulation_to_fingerprint(&modulation);
        assert_eq!(
            recovered.similarity(&original_mod_fp),
            1.0,
            "XOR self-inverse should perfectly recover the modulation"
        );
    }

    #[test]
    fn test_crystallize_different_content_different_crystal() {
        let content_a = Fingerprint::from_content("result A");
        let content_b = Fingerprint::from_content("result B");
        let modulation = ThinkingStyle::Creative.field_modulation();

        let crystal_a = crystallize(&content_a, &modulation);
        let crystal_b = crystallize(&content_b, &modulation);

        assert!(
            crystal_a.similarity(&crystal_b) < 1.0,
            "Different content should produce different crystals"
        );
    }

    #[test]
    fn test_2stroke_basic() {
        let mut node = LayerNode::new("test");
        let input = Fingerprint::from_content("stimulus");
        let mut satisfaction = LayerSatisfaction::new();
        let prev_scores = snapshot_scores(&satisfaction);
        let modulation = ThinkingStyle::Creative.field_modulation();

        let results = process_2stroke(
            &mut node,
            &input,
            &mut satisfaction,
            &prev_scores,
            &modulation,
            0,
        );

        // With Creative's low threshold (0.35), many layers should fire
        assert!(!results.is_empty(), "At least some layers should fire");
    }

    #[test]
    fn test_validate_l9_pass() {
        let result = validate_l9(0.8, 0.7, 0.1, false, 0.1);
        assert!(matches!(result, ValidationResult::Pass { .. }));
    }

    #[test]
    fn test_validate_l9_dunning_kruger() {
        // High confidence but high cross-branch disagreement
        let result = validate_l9(0.9, 0.8, 0.1, false, 0.5);
        assert!(matches!(result, ValidationResult::Reject { .. }));
    }

    #[test]
    fn test_validate_l9_hold_poor_calibration() {
        let result = validate_l9(0.7, 0.6, 0.3, false, 0.1);
        assert!(matches!(result, ValidationResult::Hold { .. }));
    }

    #[test]
    fn test_validate_l9_reject_low_evidence() {
        let result = validate_l9(0.2, 0.1, 0.1, false, 0.1);
        assert!(matches!(result, ValidationResult::Reject { .. }));
    }
}
