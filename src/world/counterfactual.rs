//! Counterfactual reasoning — Pearl's Rung 3: "What if I had...?"
//!
//! Implements do-calculus interventions on fingerprint world states.
//! An intervention replaces one causal variable with a counterfactual value,
//! then measures how the world state diverges from the baseline.
//!
//! # Science
//! - Pearl (2009): "Causality" ch.7 — structural counterfactuals
//! - Halpern & Pearl (2005): Actual causality definition
//! - Lewis (1973): Counterfactual conditionals

use crate::FINGERPRINT_BITS as TOTAL_BITS;
use crate::core::Fingerprint;

/// A counterfactual world is a BindSpace state where one or more
/// fingerprints have been replaced with intervened values.
#[derive(Debug, Clone)]
pub struct CounterfactualWorld {
    /// The intervention applied
    pub intervention: Intervention,
    /// Fingerprint of the world state AFTER intervention
    pub state: Fingerprint,
    /// Divergence from baseline (Hamming distance / total bits)
    pub divergence: f32,
}

/// An intervention replaces one causal node with a counterfactual value.
#[derive(Debug, Clone)]
pub struct Intervention {
    /// What was changed (identity of the variable)
    pub target: Fingerprint,
    /// What it was (original binding)
    pub original: Fingerprint,
    /// What it became (counterfactual binding)
    pub counterfactual: Fingerprint,
}

/// Create a counterfactual world by intervening on a variable.
///
/// Pearl Rung 3: "What would have happened if X were x'?"
///
/// Method: unbind the original variable from the world state,
/// bind the counterfactual value in its place.
///
/// ```text
/// world' = world ⊗ original ⊗ counterfactual
///        = (base ⊗ original) ⊗ original ⊗ counterfactual
///        = base ⊗ counterfactual
/// ```
pub fn intervene(world: &Fingerprint, intervention: &Intervention) -> CounterfactualWorld {
    // Unbind original, bind counterfactual
    let new_state = world
        .bind(&intervention.original) // Unbind: cancels original via XOR
        .bind(&intervention.counterfactual); // Bind: installs replacement

    let divergence = world.hamming(&new_state) as f32 / TOTAL_BITS as f32;

    CounterfactualWorld {
        intervention: intervention.clone(),
        state: new_state,
        divergence,
    }
}

/// Compare two counterfactual worlds.
///
/// Returns normalized Hamming distance between the two world states.
pub fn worlds_differ(w1: &CounterfactualWorld, w2: &CounterfactualWorld) -> f32 {
    w1.state.hamming(&w2.state) as f32 / TOTAL_BITS as f32
}

/// Apply multiple interventions to a world state.
///
/// Each intervention is applied sequentially, so later interventions
/// operate on the already-modified world.
pub fn multi_intervene(world: &Fingerprint, interventions: &[Intervention]) -> CounterfactualWorld {
    let mut current = world.clone();
    for intervention in interventions {
        let cf = intervene(&current, intervention);
        current = cf.state;
    }
    let divergence = world.hamming(&current) as f32 / TOTAL_BITS as f32;
    CounterfactualWorld {
        intervention: if let Some(last) = interventions.last() {
            last.clone()
        } else {
            Intervention {
                target: Fingerprint::zero(),
                original: Fingerprint::zero(),
                counterfactual: Fingerprint::zero(),
            }
        },
        state: current,
        divergence,
    }
}

// Keep the original structs for backward compatibility
/// High-level counterfactual metadata (for world versioning).
pub struct Counterfactual {
    pub baseline_version: u64,
    pub hypothesis_version: u64,
    pub affected_nodes: Vec<String>,
}

/// A change applied to create a counterfactual world.
#[derive(Clone, Debug)]
pub enum Change {
    Remove(String),
    UpdateTruth {
        id: String,
        frequency: f32,
        confidence: f32,
    },
    AddEdge {
        from: String,
        to: String,
        edge_type: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intervene_diverges() {
        let base = Fingerprint::from_content("base_world_state");
        let variable = Fingerprint::from_content("the_variable");
        let world = base.bind(&variable);

        let intervention = Intervention {
            target: variable.clone(),
            original: variable.clone(),
            counterfactual: Fingerprint::from_content("counterfactual_variable"),
        };

        let cf_world = intervene(&world, &intervention);

        // Counterfactual world should differ from original
        assert!(
            cf_world.divergence > 0.3,
            "Counterfactual should diverge >30% from baseline: {:.3}",
            cf_world.divergence
        );
    }

    #[test]
    fn test_intervene_recovers_base() {
        let base = Fingerprint::from_content("base_world_state");
        let variable = Fingerprint::from_content("the_variable");
        let world = base.bind(&variable);

        let cf_var = Fingerprint::from_content("counterfactual_variable");
        let intervention = Intervention {
            target: variable.clone(),
            original: variable.clone(),
            counterfactual: cf_var.clone(),
        };

        let cf_world = intervene(&world, &intervention);

        // The intervened variable should be recoverable from new world
        // world' = base ⊗ cf_var, so world' ⊗ cf_var = base
        let recovered = cf_world.state.bind(&cf_var);
        assert_eq!(
            recovered.as_raw(),
            base.as_raw(),
            "Should recover base world after unbinding counterfactual"
        );
    }

    #[test]
    fn test_identity_intervention() {
        let base = Fingerprint::from_content("base_state");
        let variable = Fingerprint::from_content("unchanged");
        let world = base.bind(&variable);

        // Intervening with same value should produce identical world
        let identity = Intervention {
            target: variable.clone(),
            original: variable.clone(),
            counterfactual: variable.clone(),
        };

        let cf = intervene(&world, &identity);
        assert_eq!(
            cf.divergence, 0.0,
            "Identity intervention should produce zero divergence"
        );
        assert_eq!(cf.state.as_raw(), world.as_raw());
    }

    #[test]
    fn test_worlds_differ() {
        let base = Fingerprint::from_content("base");
        let var = Fingerprint::from_content("variable");
        let world = base.bind(&var);

        let i1 = Intervention {
            target: var.clone(),
            original: var.clone(),
            counterfactual: Fingerprint::from_content("counterfactual_A"),
        };
        let i2 = Intervention {
            target: var.clone(),
            original: var.clone(),
            counterfactual: Fingerprint::from_content("counterfactual_B"),
        };

        let w1 = intervene(&world, &i1);
        let w2 = intervene(&world, &i2);

        let diff = worlds_differ(&w1, &w2);
        assert!(
            diff > 0.3,
            "Different interventions should produce different worlds: {:.3}",
            diff
        );
    }

    #[test]
    fn test_multi_intervene() {
        let world = Fingerprint::from_content("complex_world");
        let var_a = Fingerprint::from_content("var_a");
        let var_b = Fingerprint::from_content("var_b");
        let world = world.bind(&var_a).bind(&var_b);

        let interventions = vec![
            Intervention {
                target: var_a.clone(),
                original: var_a,
                counterfactual: Fingerprint::from_content("cf_a"),
            },
            Intervention {
                target: var_b.clone(),
                original: var_b,
                counterfactual: Fingerprint::from_content("cf_b"),
            },
        ];

        let cf = multi_intervene(&world, &interventions);
        assert!(
            cf.divergence > 0.3,
            "Multi-intervention should diverge: {:.3}",
            cf.divergence
        );
    }
}
