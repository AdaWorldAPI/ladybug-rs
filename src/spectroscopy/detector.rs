//! Spectroscopy Detector — classify a Container into cognitive coordinates.
//!
//! Given a Container (8,192-bit bitpattern) and a ConsciousnessSnapshot
//! (current cognitive state), produce the triple:
//!   (RungLevel, ThinkingStyle, FieldModulation)
//!
//! The classification pipeline:
//! 1. Extract spectral features from the Container bitpattern.
//! 2. Use the ConsciousnessSnapshot to bias the classification toward
//!    the currently active cognitive mode.
//! 3. Map features -> RungLevel via abstraction depth.
//! 4. Map features -> ThinkingStyle via feature-space nearest-neighbour.
//! 5. Compute FieldModulation from the combined feature + context signal.

use crate::cognitive::layer_stack::{ConsciousnessSnapshot, NUM_LAYERS};
use crate::cognitive::{FieldModulation, RungLevel, ThinkingStyle};
use crate::container::Container;

use super::features::{self, SpectralFeatures};

// =============================================================================
// PUBLIC API
// =============================================================================

/// Classify a Container given the current cognitive context.
///
/// Returns (RungLevel, ThinkingStyle, FieldModulation).
pub fn classify(
    input: &Container,
    context: &ConsciousnessSnapshot,
) -> (RungLevel, ThinkingStyle, FieldModulation) {
    let feats = features::extract(input);
    let ctx = ContextSignal::from_snapshot(context);

    let rung = classify_rung(&feats, &ctx);
    let style = classify_style(&feats, &ctx);
    let modulation = compute_modulation(&feats, &ctx, &style);

    (rung, style, modulation)
}

/// Classify only the RungLevel (cheap path when style is not needed).
pub fn classify_rung_only(input: &Container) -> RungLevel {
    let feats = features::extract(input);
    RungLevel::from_u8(features::depth_to_rung_index(feats.abstraction_depth))
}

/// Classify only the ThinkingStyle.
pub fn classify_style_only(input: &Container, context: &ConsciousnessSnapshot) -> ThinkingStyle {
    let feats = features::extract(input);
    let ctx = ContextSignal::from_snapshot(context);
    classify_style(&feats, &ctx)
}

// =============================================================================
// CONTEXT SIGNAL (derived from ConsciousnessSnapshot)
// =============================================================================

/// Compressed context signal extracted from a ConsciousnessSnapshot.
#[derive(Clone, Debug)]
struct ContextSignal {
    /// Average activation across all layers [0.0, 1.0].
    avg_activation: f32,
    /// Average confidence across all layers [0.0, 1.0].
    avg_confidence: f32,
    /// Coherence from snapshot.
    coherence: f32,
    /// Emergence from snapshot.
    emergence: f32,
    /// Dominant layer index (0-9).
    dominant_index: usize,
    /// Ratio of multi-agent activation (L6-L10) to total.
    multi_agent_ratio: f32,
    /// Maximum single-layer activation.
    peak_activation: f32,
}

impl ContextSignal {
    fn from_snapshot(snap: &ConsciousnessSnapshot) -> Self {
        let active_vals: Vec<f32> = snap.layers.iter().map(|m| m.value).collect();
        let total_activation: f32 = active_vals.iter().sum();
        let avg_activation = total_activation / NUM_LAYERS as f32;

        let confidences: Vec<f32> = snap.layers.iter().map(|m| m.confidence).collect();
        let avg_confidence = confidences.iter().sum::<f32>() / NUM_LAYERS as f32;

        let _single_agent: f32 = active_vals[..5].iter().sum();
        let multi_agent: f32 = active_vals[5..].iter().sum();
        let multi_agent_ratio = if total_activation > 0.0 {
            multi_agent / total_activation
        } else {
            0.0
        };

        let peak_activation = active_vals
            .iter()
            .copied()
            .fold(0.0f32, f32::max);

        Self {
            avg_activation,
            avg_confidence,
            coherence: snap.coherence,
            emergence: snap.emergence,
            dominant_index: snap.dominant_layer.index(),
            multi_agent_ratio,
            peak_activation,
        }
    }

    /// Neutral context for when no snapshot is available.
    fn neutral() -> Self {
        Self {
            avg_activation: 0.5,
            avg_confidence: 0.5,
            coherence: 0.5,
            emergence: 0.2,
            dominant_index: 0,
            multi_agent_ratio: 0.0,
            peak_activation: 0.5,
        }
    }
}

// =============================================================================
// RUNG CLASSIFICATION
// =============================================================================

/// Map spectral features to a RungLevel.
///
/// Primary driver: `abstraction_depth`.
/// Secondary bias: context coherence and emergence can shift the rung +-1.
fn classify_rung(feats: &SpectralFeatures, ctx: &ContextSignal) -> RungLevel {
    let base_rung = features::depth_to_rung_index(feats.abstraction_depth);

    // Context-based adjustment:
    // - Low coherence + high emergence suggests we should step up (novelty).
    // - High coherence + low emergence suggests things are stable.
    let adjustment = if ctx.emergence > 0.5 && ctx.coherence < 0.4 {
        1i8 // step up: emerging pattern needs deeper processing
    } else if ctx.coherence > 0.8 && ctx.emergence < 0.1 {
        -1i8 // step down: well-understood, no need for depth
    } else {
        0i8
    };

    let adjusted = (base_rung as i8 + adjustment).clamp(0, 9) as u8;
    RungLevel::from_u8(adjusted)
}

// =============================================================================
// STYLE CLASSIFICATION
// =============================================================================

/// Style profile: a point in feature-space that characterises each ThinkingStyle.
struct StyleProfile {
    style: ThinkingStyle,
    /// Expected density range.
    density_center: f32,
    /// Expected entropy.
    entropy_center: f32,
    /// Expected bridgeness.
    bridgeness_center: f32,
    /// Expected clustering.
    clustering_center: f32,
    /// Expected run complexity.
    run_complexity_center: f32,
}

/// Reference profiles for each ThinkingStyle.
///
/// These are hand-tuned centroids.  In production they would be learned from
/// labelled data, but for now they capture the intuitive mapping:
///
/// - Analytical / Focused / Systematic: high density, high clustering, low bridgeness
/// - Creative / Divergent / Exploratory: high entropy, high bridgeness, low clustering
/// - Intuitive / Peripheral: moderate everything, high run complexity
/// - Metacognitive: high entropy, high symmetry
const STYLE_PROFILES: &[StyleProfile] = &[
    StyleProfile {
        style: ThinkingStyle::Analytical,
        density_center: 0.55,
        entropy_center: 0.60,
        bridgeness_center: 0.30,
        clustering_center: 0.70,
        run_complexity_center: 0.40,
    },
    StyleProfile {
        style: ThinkingStyle::Convergent,
        density_center: 0.50,
        entropy_center: 0.55,
        bridgeness_center: 0.35,
        clustering_center: 0.65,
        run_complexity_center: 0.45,
    },
    StyleProfile {
        style: ThinkingStyle::Systematic,
        density_center: 0.50,
        entropy_center: 0.65,
        bridgeness_center: 0.40,
        clustering_center: 0.60,
        run_complexity_center: 0.50,
    },
    StyleProfile {
        style: ThinkingStyle::Creative,
        density_center: 0.50,
        entropy_center: 0.80,
        bridgeness_center: 0.70,
        clustering_center: 0.30,
        run_complexity_center: 0.65,
    },
    StyleProfile {
        style: ThinkingStyle::Divergent,
        density_center: 0.48,
        entropy_center: 0.75,
        bridgeness_center: 0.65,
        clustering_center: 0.35,
        run_complexity_center: 0.60,
    },
    StyleProfile {
        style: ThinkingStyle::Exploratory,
        density_center: 0.45,
        entropy_center: 0.85,
        bridgeness_center: 0.75,
        clustering_center: 0.25,
        run_complexity_center: 0.70,
    },
    StyleProfile {
        style: ThinkingStyle::Focused,
        density_center: 0.60,
        entropy_center: 0.40,
        bridgeness_center: 0.20,
        clustering_center: 0.80,
        run_complexity_center: 0.30,
    },
    StyleProfile {
        style: ThinkingStyle::Diffuse,
        density_center: 0.50,
        entropy_center: 0.70,
        bridgeness_center: 0.55,
        clustering_center: 0.40,
        run_complexity_center: 0.55,
    },
    StyleProfile {
        style: ThinkingStyle::Peripheral,
        density_center: 0.40,
        entropy_center: 0.75,
        bridgeness_center: 0.60,
        clustering_center: 0.35,
        run_complexity_center: 0.70,
    },
    StyleProfile {
        style: ThinkingStyle::Intuitive,
        density_center: 0.50,
        entropy_center: 0.55,
        bridgeness_center: 0.50,
        clustering_center: 0.50,
        run_complexity_center: 0.60,
    },
    StyleProfile {
        style: ThinkingStyle::Deliberate,
        density_center: 0.52,
        entropy_center: 0.60,
        bridgeness_center: 0.45,
        clustering_center: 0.55,
        run_complexity_center: 0.45,
    },
    StyleProfile {
        style: ThinkingStyle::Metacognitive,
        density_center: 0.50,
        entropy_center: 0.80,
        bridgeness_center: 0.55,
        clustering_center: 0.45,
        run_complexity_center: 0.55,
    },
];

/// Euclidean distance between features and a style profile.
fn style_distance(feats: &SpectralFeatures, profile: &StyleProfile) -> f32 {
    let d_density = feats.density - profile.density_center;
    let d_entropy = feats.entropy - profile.entropy_center;
    let d_bridge = feats.bridgeness - profile.bridgeness_center;
    let d_cluster = feats.clustering - profile.clustering_center;
    let d_run = feats.run_complexity - profile.run_complexity_center;

    (d_density * d_density
        + d_entropy * d_entropy
        + d_bridge * d_bridge
        + d_cluster * d_cluster
        + d_run * d_run)
        .sqrt()
}

/// Classify features into a ThinkingStyle by nearest-neighbour in feature space.
///
/// Context biases the result: if the dominant layer is in the multi-agent
/// range (L6-L10), styles that support exploration are weighted more heavily.
fn classify_style(feats: &SpectralFeatures, ctx: &ContextSignal) -> ThinkingStyle {
    let mut best_style = ThinkingStyle::Deliberate;
    let mut best_score = f32::MAX;

    for profile in STYLE_PROFILES.iter() {
        let mut dist = style_distance(feats, profile);

        // Context bias: reduce distance for styles that match the cognitive state.
        dist *= context_bias_for_style(profile.style, ctx);

        if dist < best_score {
            best_score = dist;
            best_style = profile.style;
        }
    }

    best_style
}

/// Compute a multiplicative bias (< 1.0 = prefer, > 1.0 = disfavour).
fn context_bias_for_style(style: ThinkingStyle, ctx: &ContextSignal) -> f32 {
    let mut bias = 1.0f32;

    // When multi-agent layers are active, prefer exploratory / creative styles.
    if ctx.multi_agent_ratio > 0.3 {
        match style {
            ThinkingStyle::Creative | ThinkingStyle::Divergent | ThinkingStyle::Exploratory => {
                bias *= 0.7;
            }
            ThinkingStyle::Focused => {
                bias *= 1.3;
            }
            _ => {}
        }
    }

    // When coherence is high, prefer systematic / analytical.
    if ctx.coherence > 0.7 {
        match style {
            ThinkingStyle::Analytical | ThinkingStyle::Systematic | ThinkingStyle::Convergent => {
                bias *= 0.8;
            }
            _ => {}
        }
    }

    // When emergence is high, prefer metacognitive.
    if ctx.emergence > 0.5 {
        match style {
            ThinkingStyle::Metacognitive => {
                bias *= 0.7;
            }
            _ => {}
        }
    }

    // When peak activation is very low, prefer diffuse / peripheral.
    if ctx.peak_activation < 0.2 {
        match style {
            ThinkingStyle::Diffuse | ThinkingStyle::Peripheral => {
                bias *= 0.75;
            }
            _ => {}
        }
    }

    bias
}

// =============================================================================
// FIELD MODULATION
// =============================================================================

/// Compute FieldModulation from spectral features and context.
///
/// Rather than returning the style's default modulation, we blend it with
/// feature-derived parameters to produce a modulation tuned to the specific
/// input.
fn compute_modulation(
    feats: &SpectralFeatures,
    ctx: &ContextSignal,
    style: &ThinkingStyle,
) -> FieldModulation {
    let base = style.field_modulation();

    // Resonance threshold: lower when entropy is high (more uncertain data
    // needs a looser matching threshold).
    let resonance_threshold = (base.resonance_threshold * (1.0 - feats.entropy * 0.3))
        .clamp(0.1, 0.95);

    // Fan-out: increase when bridgeness is high (distributed encoding benefits
    // from wider search).
    let fan_out_f = base.fan_out as f32 * (1.0 + feats.bridgeness * 0.5);
    let fan_out = (fan_out_f as usize).max(1).min(30);

    // Depth bias: increase with abstraction depth and clustering.
    let depth_bias = (base.depth_bias + feats.abstraction_depth * 0.3 + feats.clustering * 0.1)
        .clamp(0.0, 1.0);

    // Breadth bias: increase with entropy and bridgeness.
    let breadth_bias = (base.breadth_bias + feats.entropy * 0.2 + feats.bridgeness * 0.1)
        .clamp(0.0, 1.0);

    // Noise tolerance: scale with context confidence.
    let noise_tolerance =
        (base.noise_tolerance * (1.0 + (1.0 - ctx.avg_confidence) * 0.5)).clamp(0.01, 0.8);

    // Speed bias: faster when coherence is high (well-understood territory).
    let speed_bias = (base.speed_bias + ctx.coherence * 0.2 - ctx.emergence * 0.1)
        .clamp(0.0, 1.0);

    // Exploration: higher when emergence is strong and clustering is low.
    let exploration =
        (base.exploration + ctx.emergence * 0.3 - feats.clustering * 0.15).clamp(0.0, 1.0);

    FieldModulation {
        resonance_threshold,
        fan_out,
        depth_bias,
        breadth_bias,
        noise_tolerance,
        speed_bias,
        exploration,
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive::layer_stack::{LayerNode, process_layers_wave, snapshot_consciousness};
    use crate::core::Fingerprint;

    fn make_snapshot() -> ConsciousnessSnapshot {
        let mut node = LayerNode::new("test");
        let input = Fingerprint::from_content("test signal");
        process_layers_wave(&mut node, &input, 0);
        snapshot_consciousness(&node, 0)
    }

    #[test]
    fn test_classify_returns_valid_triple() {
        let c = Container::random(42);
        let snap = make_snapshot();
        let (rung, _style, modulation) = classify(&c, &snap);

        assert!(rung.as_u8() <= 9);
        assert!(modulation.resonance_threshold >= 0.0 && modulation.resonance_threshold <= 1.0);
        assert!(modulation.fan_out >= 1);
        assert!(modulation.depth_bias >= 0.0 && modulation.depth_bias <= 1.0);
    }

    #[test]
    fn test_classify_zero_container() {
        let c = Container::zero();
        let snap = make_snapshot();
        let (rung, _style, _mod) = classify(&c, &snap);
        // Zero container has zero abstraction depth -> Surface rung.
        assert_eq!(rung, RungLevel::Surface);
    }

    #[test]
    fn test_classify_rung_only() {
        let c = Container::random(77);
        let rung = classify_rung_only(&c);
        assert!(rung.as_u8() <= 9);
    }

    #[test]
    fn test_classify_style_only() {
        let c = Container::random(123);
        let snap = make_snapshot();
        let style = classify_style_only(&c, &snap);
        // Just verify it returns a valid variant.
        let _ = style.field_modulation();
    }

    #[test]
    fn test_different_containers_can_yield_different_styles() {
        let snap = make_snapshot();
        let mut styles = std::collections::HashSet::new();
        // Try many different seeds to get variety.
        for seed in 0..100 {
            let c = Container::random(seed);
            let (_, style, _) = classify(&c, &snap);
            styles.insert(format!("{:?}", style));
        }
        // We should get at least 2 distinct styles across 100 random containers.
        assert!(
            styles.len() >= 2,
            "expected style variety, got only {:?}",
            styles
        );
    }

    #[test]
    fn test_modulation_parameters_in_range() {
        let snap = make_snapshot();
        for seed in 0..20 {
            let c = Container::random(seed * 31);
            let (_, _, m) = classify(&c, &snap);
            assert!(m.resonance_threshold >= 0.1 && m.resonance_threshold <= 0.95);
            assert!(m.fan_out >= 1 && m.fan_out <= 30);
            assert!(m.depth_bias >= 0.0 && m.depth_bias <= 1.0);
            assert!(m.breadth_bias >= 0.0 && m.breadth_bias <= 1.0);
            assert!(m.noise_tolerance >= 0.01 && m.noise_tolerance <= 0.8);
            assert!(m.speed_bias >= 0.0 && m.speed_bias <= 1.0);
            assert!(m.exploration >= 0.0 && m.exploration <= 1.0);
        }
    }
}
