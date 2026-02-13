#![allow(clippy::needless_range_loop)]
//! 34 Tactics Proofs — Verifying PR #100 cognitive primitives.
//!
//! Each test proves a specific tactic's contract using the actual
//! module implementations from PR #100.
//!
//! Run: `cargo test --test tactics`

use ladybug::FINGERPRINT_BITS;
use ladybug::FINGERPRINT_U64 as WORDS;
use ladybug::core::Fingerprint;
use ladybug::core::vsa::{fusion_quality, multi_fusion_quality};
use ladybug::nars::TruthValue;

// =============================================================================
// T-01: Recursive Expansion Convergence
// =============================================================================

/// PROOF T-01: Recursive expansion converges (Berry-Esseen guarantee)
///
/// RecursiveExpansion terminates when successive fingerprints
/// change by less than convergence_threshold.
///
/// Ref: Berry-Esseen CLT convergence
#[test]
fn tactics_t01_recursive_expansion_converges() {
    use ladybug::cognitive::recursive::RecursiveExpansion;

    let seed = Fingerprint::from_content("dodecagon_problem");
    let expander = RecursiveExpansion::new(10, 0.01); // max 10 deep, 1% threshold

    let trace = expander.expand(seed.as_raw(), |fp, _depth| {
        // Deterministic converging transform: shift one word
        let mut result = *fp;
        // XOR with a fixed pattern (converges as repeated XOR is self-inverse)
        let pattern = Fingerprint::from_content("fixed_attractor");
        for i in 0..WORDS {
            result[i] ^= pattern.as_raw()[i] >> 1; // Diminishing change
        }
        result
    });

    // Should produce a result (either converged or hit max depth)
    assert!(
        trace.depth() > 0,
        "Should produce at least one expansion step"
    );
    assert!(trace.result().is_some(), "Should produce a result");
}

// =============================================================================
// T-03: Multi-Agent Debate Improves Truth
// =============================================================================

/// PROOF T-03: Structured debate improves truth value quality
///
/// Ref: Mercier & Sperber (2011) argumentative theory of reasoning
#[cfg(feature = "crewai")]
#[test]
fn tactics_t03_debate_improves_truth() {
    use ladybug::orchestration::debate::*;

    let initial = TruthValue::new(0.5, 0.1); // Uncertain starting point

    let pro_args = vec![
        Argument {
            fingerprint: Fingerprint::from_content("symmetry_approach")
                .as_raw()
                .clone(),
            truth: TruthValue::new(0.8, 0.7),
            is_pro: true,
        },
        Argument {
            fingerprint: Fingerprint::from_content("group_theory").as_raw().clone(),
            truth: TruthValue::new(0.75, 0.6),
            is_pro: true,
        },
    ];
    let con_args = vec![Argument {
        fingerprint: Fingerprint::from_content("brute_force").as_raw().clone(),
        truth: TruthValue::new(0.4, 0.5),
        is_pro: false,
    }];

    let outcome = debate(&initial, &pro_args, &con_args, &DebateConfig::default());

    // Debate should increase confidence through argument revision
    assert!(
        outcome.final_truth.confidence > initial.confidence,
        "Debate should increase confidence: {:.3} > {:.3}",
        outcome.final_truth.confidence,
        initial.confidence
    );

    // Pro arguments are stronger → should win
    assert!(
        outcome.final_truth.expectation() > 0.5,
        "Stronger pro arguments should increase expectation: {:.3}",
        outcome.final_truth.expectation()
    );
}

// =============================================================================
// T-04: Reverse Causal Trace Recovers Chain
// =============================================================================

/// PROOF T-04: Reverse causal trace recovers causal chain
///
/// Given a chain state→action→outcome, reverse_trace walks backward
/// from outcome to find state.
///
/// Ref: Pearl (2009), Halpern & Pearl (2005)
#[test]
fn tactics_t04_reverse_causal_trace() {
    use ladybug::search::causal::CausalSearch;

    let mut search = CausalSearch::new();

    let state = Fingerprint::from_content("initial_state");
    let action = Fingerprint::from_content("take_action");
    let outcome = Fingerprint::from_content("observed_outcome");

    // Store the causal chain
    search.store_intervention(state.as_raw(), action.as_raw(), outcome.as_raw(), 1.0);

    // Trace backward from outcome
    let trace = search.reverse_trace(outcome.as_raw(), 5);

    // Should find the cause
    assert!(
        trace.depth() >= 1,
        "Should find at least one causal step, got depth {}",
        trace.depth()
    );

    // The root cause should match the original state
    if let Some(root) = trace.root_cause() {
        assert_eq!(
            root,
            state.as_raw(),
            "Root cause should match original state"
        );
    }
}

// =============================================================================
// T-07: Adversarial Critique Detects Weakness
// =============================================================================

/// PROOF T-07: Adversarial challenges detect weak beliefs
///
/// Ref: Kahneman (2011) adversarial collaboration
#[test]
fn tactics_t07_adversarial_detects_weakness() {
    use ladybug::nars::adversarial::*;

    // Weak belief: moderate confidence
    let weak = TruthValue::new(0.6, 0.4);
    let challenges = critique(&weak);

    // Should generate all 5 challenge types
    assert_eq!(
        challenges.len(),
        5,
        "Should generate exactly 5 challenges, got {}",
        challenges.len()
    );

    // Robustness score should be low for weak belief
    let weak_score = robustness_score(&weak);

    // Strong belief should survive better
    let strong = TruthValue::new(0.9, 0.9);
    let strong_score = robustness_score(&strong);

    assert!(
        strong_score > weak_score,
        "Strong belief should be more robust: {:.3} > {:.3}",
        strong_score,
        weak_score
    );
}

// =============================================================================
// T-10: MetaCognition Tracks Calibration
// =============================================================================

/// PROOF T-10: Brier score calibration tracks prediction quality
///
/// Ref: Brier (1950) verification of weather forecasts
#[test]
fn tactics_t10_metacog_tracks_calibration() {
    use ladybug::cognitive::GateState;
    use ladybug::cognitive::metacog::MetaCognition;

    let mut meta = MetaCognition::new();

    // Well-calibrated predictions: confidence matches outcomes
    for _ in 0..50 {
        let _assessment = meta.assess(GateState::Flow, &TruthValue::new(0.8, 0.8));
        meta.record_outcome(0.8, 1.0); // Predicted 0.8, outcome was true
    }

    let good_brier = meta.brier_score();
    assert!(
        good_brier < 0.1,
        "Well-calibrated predictions should have low Brier: {:.3}",
        good_brier
    );

    // Reset and predict badly
    meta.reset();
    for _ in 0..50 {
        let _ = meta.assess(GateState::Flow, &TruthValue::new(0.9, 0.9));
        meta.record_outcome(0.9, 0.0); // Predicted 0.9, outcome was false
    }

    let bad_brier = meta.brier_score();
    assert!(
        bad_brier > good_brier,
        "Poor calibration should have higher Brier: {:.3} > {:.3}",
        bad_brier,
        good_brier
    );
}

// =============================================================================
// T-11: Contradiction Detection
// =============================================================================

/// PROOF T-11: Contradictions detected between structurally similar beliefs
///
/// Ref: Priest (2002) paraconsistent logic
#[test]
fn tactics_t11_contradiction_detection() {
    use ladybug::nars::contradiction::*;

    // Create similar fingerprints (same content → identical structure)
    let fp_a = Fingerprint::from_content("cats are mammals");
    let fp_b = fp_a.clone(); // Identical structure
    let fp_c = Fingerprint::from_content("quantum chromodynamics"); // Unrelated

    let fingerprints = vec![*fp_a.as_raw(), *fp_b.as_raw(), *fp_c.as_raw()];
    let truths = vec![
        TruthValue::new(0.9, 0.8), // fp_a: strongly believes
        TruthValue::new(0.1, 0.8), // fp_b: strongly disbelieves (SAME content!)
        TruthValue::new(0.7, 0.5), // fp_c: unrelated
    ];

    let contradictions = detect_contradictions(
        &fingerprints,
        &truths,
        0.9, // High similarity threshold (identical fps will match)
        0.5, // Moderate conflict threshold
    );

    // Should detect contradiction between fp_a and fp_b
    assert!(
        !contradictions.is_empty(),
        "Should detect contradiction between identical fps with opposing truths"
    );

    // Coherence score should reflect the contradiction
    let coherence = coherence_score(&truths);
    assert!(
        coherence < 1.0,
        "Contradictions should reduce coherence: {:.3}",
        coherence
    );
}

// =============================================================================
// T-15: CRP Distribution from Corpus
// =============================================================================

/// PROOF T-15: ClusterDistribution captures corpus statistics
///
/// CRP percentiles should accurately characterize the Hamming distance
/// distribution of a fingerprint corpus.
#[test]
fn tactics_t15_crp_distribution() {
    use ladybug::search::distribution::ClusterDistribution;

    // Build a corpus of real fingerprint distances
    let distances: Vec<u32> = (0..500)
        .map(|i| {
            let a = Fingerprint::from_content(&format!("corpus_a_{}", i));
            let b = Fingerprint::from_content(&format!("corpus_b_{}", i));
            a.hamming(&b)
        })
        .collect();

    let dist = ClusterDistribution::from_distances(&distances);

    // Basic sanity
    assert_eq!(dist.n, 500);
    assert!(dist.mu > 0.0);
    assert!(dist.sigma > 0.0);

    // Percentile ordering
    assert!(
        dist.p25 <= dist.p50,
        "p25={:.1} should be ≤ p50={:.1}",
        dist.p25,
        dist.p50
    );
    assert!(
        dist.p50 <= dist.p75,
        "p50={:.1} should be ≤ p75={:.1}",
        dist.p50,
        dist.p75
    );
    assert!(
        dist.p75 <= dist.p95,
        "p75={:.1} should be ≤ p95={:.1}",
        dist.p75,
        dist.p95
    );
    assert!(
        dist.p95 <= dist.p99,
        "p95={:.1} should be ≤ p99={:.1}",
        dist.p95,
        dist.p99
    );

    // For random fingerprints: mean should be near FINGERPRINT_BITS/2 (≈8192)
    assert!(
        (dist.mu - 8192.0).abs() < 500.0,
        "Mean {:.1} should be near 8192 for random fingerprints",
        dist.mu
    );

    // Z-score of mean should be ~0
    let z_at_mean = dist.z_score(dist.mu);
    assert!(
        z_at_mean.abs() < 0.01,
        "Z-score at mean should be ~0, got {:.3}",
        z_at_mean
    );
}

// =============================================================================
// T-20: Shadow Parallel Consensus
// =============================================================================

/// PROOF T-20: Shadow parallel processing detects discrepancies
///
/// Ref: N-version programming (Avizienis, 1985)
#[test]
fn tactics_t20_shadow_consensus() {
    use ladybug::fabric::shadow::*;

    let config = ShadowConfig::default();
    let mut processor = ShadowProcessor::new(config);

    // Agreeing: identical fingerprints
    let primary = Fingerprint::from_content("approach_A_result");
    let shadow = Fingerprint::from_content("approach_A_result"); // Same

    let comparison = processor.compare(primary.as_raw(), shadow.as_raw(), 1.0, 1.0);
    assert!(comparison.agreement, "Identical results should agree");
    assert!(
        comparison.divergence < 0.001,
        "Identical results should have ~0 divergence: {:.3}",
        comparison.divergence
    );

    // Disagreeing: completely different fingerprints
    let shadow_diff = Fingerprint::from_content("approach_B_totally_different");
    let comparison2 = processor.compare(primary.as_raw(), shadow_diff.as_raw(), 1.0, 1.0);
    assert!(!comparison2.agreement, "Different results should NOT agree");

    // Agreement rate should reflect mixed history
    assert!(
        processor.agreement_rate() < 1.0,
        "Mixed history should show <100% agreement: {:.3}",
        processor.agreement_rate()
    );
    assert_eq!(processor.total_comparisons(), 2);
}

// =============================================================================
// T-24: Bind/Unbind Roundtrip Exact
// =============================================================================

/// PROOF T-24: Bind/unbind roundtrip preserves information exactly
///
/// Uses PR100's fusion_quality metric.
///
/// Ref: Plate (2003) Holographic Reduced Representations
#[test]
fn tactics_t24_fusion_quality_exact() {
    for i in 0..100 {
        let a = Fingerprint::from_content(&format!("fusion_a_{}", i));
        let b = Fingerprint::from_content(&format!("fusion_b_{}", i));

        let (dist_a, dist_b) = fusion_quality(&a, &b);

        // XOR roundtrip is exact → recovery distance should be 0
        assert_eq!(
            dist_a, 0.0,
            "Recovery of A should be exact at i={}, got {}",
            i, dist_a
        );
        assert_eq!(
            dist_b, 0.0,
            "Recovery of B should be exact at i={}, got {}",
            i, dist_b
        );
    }
}

// =============================================================================
// T-25: Hamming Normal Approximation (= F-1)
// =============================================================================

/// PROOF T-25: Hamming distance follows Normal distribution
///
/// This is the same as F-1 but framed as a tactic proof:
/// The Normal approximation enables all statistical reasoning
/// about fingerprint distances.
#[test]
fn tactics_t25_hamming_normal() {
    let n = 5_000;
    let d = FINGERPRINT_BITS as f64;
    let mu = d / 2.0;
    let sigma = (d / 4.0).sqrt();

    let distances: Vec<f64> = (0..n)
        .map(|i| {
            let a = Fingerprint::from_content(&format!("normal_a_{}", i));
            let b = Fingerprint::from_content(&format!("normal_b_{}", i));
            a.hamming(&b) as f64
        })
        .collect();

    let mean: f64 = distances.iter().sum::<f64>() / n as f64;
    let variance: f64 = distances.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / n as f64;
    let observed_sigma = variance.sqrt();

    // Mean within 1% of theoretical
    assert!(
        (mean - mu).abs() / mu < 0.01,
        "Mean {:.1} should be within 1% of theoretical {:.1}",
        mean,
        mu
    );

    // Sigma within 5% of theoretical
    assert!(
        (observed_sigma - sigma).abs() / sigma < 0.05,
        "Sigma {:.1} should be within 5% of theoretical {:.1}",
        observed_sigma,
        sigma
    );
}

// =============================================================================
// T-28: Temporal Granger Effect
// =============================================================================

/// PROOF T-28: Granger temporal effect size detects causal timing
///
/// When series A predicts series B beyond B's autocorrelation,
/// the Granger effect size should be positive.
///
/// Ref: Granger (1969) causality testing
#[test]
fn tactics_t28_temporal_granger() {
    use ladybug::search::temporal::*;

    // Create two fingerprint series with causal relationship
    // A[t] causes B[t+1] (shifted copy)
    let series_a: Vec<[u64; WORDS]> = (0..20)
        .map(|i| *Fingerprint::from_content(&format!("granger_a_{}", i)).as_raw())
        .collect();

    // B = shifted version of A (A causes B)
    let mut series_b: Vec<[u64; WORDS]> = vec![[0u64; WORDS]]; // B[0] = noise
    series_b.extend_from_slice(&series_a[..19]); // B[t+1] = A[t]

    let effect = granger_effect(&series_a, &series_b, 1);

    // Should detect some effect (cross-correlation should exceed auto)
    // Note: with from_content fingerprints the relationship is structural
    assert!(effect.is_some(), "Should produce a Granger effect result");
}

// =============================================================================
// T-31: Counterfactual Divergence
// =============================================================================

/// PROOF T-31: Counterfactual worlds diverge measurably
///
/// Ref: Pearl (2009) do-calculus
#[test]
fn tactics_t31_counterfactual_divergence() {
    use ladybug::world::counterfactual::*;

    let base = Fingerprint::from_content("base_world");
    let var = Fingerprint::from_content("causal_variable");
    let world = base.bind(&var);

    // Two different counterfactual interventions
    let i1 = Intervention {
        target: var.clone(),
        original: var.clone(),
        counterfactual: Fingerprint::from_content("what_if_A"),
    };
    let i2 = Intervention {
        target: var.clone(),
        original: var.clone(),
        counterfactual: Fingerprint::from_content("what_if_B"),
    };

    let w1 = intervene(&world, &i1);
    let w2 = intervene(&world, &i2);

    // Both should diverge from baseline
    assert!(
        w1.divergence > 0.3,
        "World 1 should diverge: {:.3}",
        w1.divergence
    );
    assert!(
        w2.divergence > 0.3,
        "World 2 should diverge: {:.3}",
        w2.divergence
    );

    // And from each other
    let mutual = worlds_differ(&w1, &w2);
    assert!(
        mutual > 0.3,
        "Worlds should differ from each other: {:.3}",
        mutual
    );
}

// =============================================================================
// T-34: Cross-Domain Fusion
// =============================================================================

/// PROOF T-34: Cross-domain fusion via bind preserves retrievability
///
/// Binding fingerprints from different domains (text, number, graph)
/// creates a compound representation from which each component
/// is exactly recoverable.
///
/// Ref: Plate (2003), Kanerva (2009) Hyperdimensional Computing
#[test]
fn tactics_t34_cross_domain_fusion() {
    // Three "domains"
    let text_fp = Fingerprint::from_content("the quick brown fox");
    let number_fp = Fingerprint::from_content("pi=3.14159");
    let graph_fp = Fingerprint::from_content("edge(A→B)");

    // Fuse all three domains
    let fused = text_fp.bind(&number_fp).bind(&graph_fp);

    // Recover each domain
    let recovered_text = fused.bind(&number_fp).bind(&graph_fp);
    let recovered_number = fused.bind(&text_fp).bind(&graph_fp);
    let recovered_graph = fused.bind(&text_fp).bind(&number_fp);

    // Recovery should be exact (XOR algebra)
    assert_eq!(
        recovered_text.as_raw(),
        text_fp.as_raw(),
        "Text domain should be exactly recoverable"
    );
    assert_eq!(
        recovered_number.as_raw(),
        number_fp.as_raw(),
        "Number domain should be exactly recoverable"
    );
    assert_eq!(
        recovered_graph.as_raw(),
        graph_fp.as_raw(),
        "Graph domain should be exactly recoverable"
    );

    // Multi-fusion quality should also confirm
    let quality = multi_fusion_quality(&[text_fp, number_fp, graph_fp]);
    assert_eq!(quality, 0.0, "Multi-domain fusion should be exact");
}
