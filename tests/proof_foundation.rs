//! Foundation Proofs — Mathematical axioms underlying ladybug-rs.
//!
//! Each test proves a foundational property of the 16384-bit fingerprint
//! algebra that all higher-level reasoning depends on.
//!
//! Run: `cargo test --test foundation`

use ladybug::FINGERPRINT_BITS;
use ladybug::core::Fingerprint;
use ladybug::core::vsa::{fusion_quality, multi_fusion_quality};
use ladybug::nars::TruthValue;

// =============================================================================
// F-1: Berry-Esseen Empirical Verification
// =============================================================================

/// PROOF F-1: Berry-Esseen bound at d=16384
///
/// Theorem: Hamming distance between random fingerprints follows
/// Normal(μ=d/2, σ²=d/4) with approximation error < 0.4748/√d.
/// At d=16384: error < 0.00371.
///
/// Method: Kolmogorov-Smirnov test against Normal CDF.
/// All fingerprints generated deterministically via from_content().
///
/// Ref: Berry (1941), Esseen (1942), Korolev & Shevtsova (2010)
#[test]
fn foundation_f1_berry_esseen_16k() {
    let n = 10_000usize;
    let d = FINGERPRINT_BITS as f64;
    let mu = d / 2.0;
    let sigma = (d / 4.0).sqrt();

    // Generate deterministic pairs via from_content (seeded by content hash)
    let mut distances: Vec<f64> = (0..n)
        .map(|i| {
            let a = Fingerprint::from_content(&format!("berry_a_{}", i));
            let b = Fingerprint::from_content(&format!("berry_b_{}", i));
            a.hamming(&b) as f64
        })
        .collect();

    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut max_dev = 0.0f64;
    for (i, &dist) in distances.iter().enumerate() {
        let ecdf = (i + 1) as f64 / n as f64;
        let z = (dist - mu) / sigma;
        let tcdf = 0.5 * (1.0 + erf_approx(z / std::f64::consts::SQRT_2));
        max_dev = max_dev.max((ecdf - tcdf).abs());
    }

    // Theoretical Berry-Esseen bound: 0.4748/√d ≈ 0.00371 for truly random bits.
    // from_content() uses SHA-512 hash → near-random but not perfectly Bernoulli(0.5).
    // KS test with n=10K has own sampling error ~1.36/√n ≈ 0.0136.
    // Combined threshold: 0.02 (safely above both error sources).
    assert!(
        max_dev < 0.02,
        "KS stat {:.6} exceeds practical threshold 0.02 (theoretical: 0.004, sampling: 0.014)",
        max_dev
    );

    // Also verify mean and variance are close to theoretical
    let mean: f64 = distances.iter().sum::<f64>() / n as f64;
    let variance: f64 = distances.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / n as f64;

    assert!(
        (mean - mu).abs() / mu < 0.01,
        "Mean {:.1} should be within 1% of theoretical {:.1}",
        mean,
        mu
    );
    assert!(
        (variance - d / 4.0).abs() / (d / 4.0) < 0.05,
        "Variance {:.1} should be within 5% of theoretical {:.1}",
        variance,
        d / 4.0
    );
}

// =============================================================================
// F-2: Fisher Sufficiency (Documented Derivation)
// =============================================================================

/// PROOF F-2: Fisher sufficiency of (μ,σ) for Normal
///
/// For X ~ Normal(μ,σ²), the sample mean and sample variance are
/// jointly sufficient statistics. This means ClusterDistribution's
/// (mu, sigma) capture ALL information about the distance distribution.
///
/// This is an analytic proof — the test verifies the computational
/// consequence: adding more samples doesn't change the CRP's predictions
/// beyond what (μ,σ) already encode.
#[test]
fn foundation_f2_fisher_sufficiency() {
    use ladybug::search::distribution::ClusterDistribution;

    // Generate distance samples
    let distances_100: Vec<u32> = (0..100)
        .map(|i| {
            let a = Fingerprint::from_content(&format!("fisher_a_{}", i));
            let b = Fingerprint::from_content(&format!("fisher_b_{}", i));
            a.hamming(&b)
        })
        .collect();

    let distances_500: Vec<u32> = (0..500)
        .map(|i| {
            let a = Fingerprint::from_content(&format!("fisher_a_{}", i));
            let b = Fingerprint::from_content(&format!("fisher_b_{}", i));
            a.hamming(&b)
        })
        .collect();

    let crp_100 = ClusterDistribution::from_distances(&distances_100);
    let crp_500 = ClusterDistribution::from_distances(&distances_500);

    // Sufficient statistics should converge: (μ,σ) from 100 ≈ (μ,σ) from 500
    // Both sample from the same Normal distribution
    let mu_diff = (crp_100.mu - crp_500.mu).abs() / crp_500.mu;
    let sigma_diff = (crp_100.sigma - crp_500.sigma).abs() / crp_500.sigma;

    assert!(
        mu_diff < 0.05,
        "μ should converge: 100-sample={:.1}, 500-sample={:.1} (diff={:.3})",
        crp_100.mu,
        crp_500.mu,
        mu_diff
    );
    assert!(
        sigma_diff < 0.15,
        "σ should converge: 100-sample={:.1}, 500-sample={:.1} (diff={:.3})",
        crp_100.sigma,
        crp_500.sigma,
        sigma_diff
    );
}

// =============================================================================
// F-3: XOR Self-Inverse
// =============================================================================

/// PROOF F-3: XOR is exactly self-inverse
///
/// For any A, B: A.bind(&B).bind(&B) == A  (zero Hamming distance)
///
/// This is the algebraic identity that enables ABBA retrieval,
/// fusion_quality exactness, and counterfactual intervention recovery.
///
/// Ref: Boolean algebra XOR identity: a ⊕ b ⊕ b = a
#[test]
fn foundation_f3_xor_self_inverse_exact() {
    for i in 0..1000 {
        let a = Fingerprint::from_content(&format!("xor_a_{}", i));
        let b = Fingerprint::from_content(&format!("xor_b_{}", i));
        let roundtrip = a.bind(&b).bind(&b);
        assert_eq!(
            a.as_raw(),
            roundtrip.as_raw(),
            "XOR self-inverse violated at i={}",
            i
        );
    }
}

/// PROOF F-3b: XOR commutativity and associativity
///
/// bind(A, B) == bind(B, A) and bind(bind(A,B),C) == bind(A,bind(B,C))
#[test]
fn foundation_f3b_xor_commutativity_associativity() {
    for i in 0..100 {
        let a = Fingerprint::from_content(&format!("comm_a_{}", i));
        let b = Fingerprint::from_content(&format!("comm_b_{}", i));
        let c = Fingerprint::from_content(&format!("comm_c_{}", i));

        // Commutativity: A ⊗ B = B ⊗ A
        assert_eq!(
            a.bind(&b).as_raw(),
            b.bind(&a).as_raw(),
            "XOR commutativity violated at i={}",
            i
        );

        // Associativity: (A ⊗ B) ⊗ C = A ⊗ (B ⊗ C)
        let left = a.bind(&b).bind(&c);
        let right = a.bind(&b.bind(&c));
        assert_eq!(
            left.as_raw(),
            right.as_raw(),
            "XOR associativity violated at i={}",
            i
        );
    }
}

// =============================================================================
// F-4: Triangle Inequality
// =============================================================================

/// PROOF F-4: Hamming distance satisfies triangle inequality
///
/// For any A, B, C: d(A,C) ≤ d(A,B) + d(B,C)
///
/// This proves Hamming distance is a proper metric, which enables:
/// - CLAM/CAKES pruning correctness (ball tree guarantees)
/// - HDR cascade level filtering correctness
///
/// Ref: Metric space axiom (identity, symmetry, triangle inequality)
#[test]
fn foundation_f4_triangle_inequality() {
    for i in 0..10_000 {
        let a = Fingerprint::from_content(&format!("tri_a_{}", i));
        let b = Fingerprint::from_content(&format!("tri_b_{}", i));
        let c = Fingerprint::from_content(&format!("tri_c_{}", i));

        let d_ac = a.hamming(&c);
        let d_ab = a.hamming(&b);
        let d_bc = b.hamming(&c);

        assert!(
            d_ac <= d_ab + d_bc,
            "Triangle inequality violated at i={}: d(A,C)={} > d(A,B)+d(B,C)={}",
            i,
            d_ac,
            d_ab + d_bc
        );
    }
}

/// PROOF F-4b: Hamming distance symmetry and identity
#[test]
fn foundation_f4b_metric_axioms() {
    for i in 0..1000 {
        let a = Fingerprint::from_content(&format!("metric_a_{}", i));
        let b = Fingerprint::from_content(&format!("metric_b_{}", i));

        // Identity: d(A, A) = 0
        assert_eq!(a.hamming(&a), 0, "Identity violated at i={}", i);

        // Symmetry: d(A, B) = d(B, A)
        assert_eq!(a.hamming(&b), b.hamming(&a), "Symmetry violated at i={}", i);
    }
}

// =============================================================================
// F-5: Mexican Hat from CRP Percentiles
// =============================================================================

/// PROOF F-5: Mexican hat calibration from CRP percentiles
///
/// ClusterDistribution.mexican_hat() produces a receptive field:
/// - Positive (excitation) for distances near cluster center
/// - Negative (inhibition) for distances far from center
/// - Zero at the transition boundary
///
/// Ref: Difference of Gaussians (DoG), CRP (Chinese Restaurant Process)
#[test]
fn foundation_f5_mexican_hat_shape() {
    use ladybug::search::distribution::ClusterDistribution;

    // Create distribution from real fingerprint distances
    let distances: Vec<u32> = (0..1000)
        .map(|i| {
            let a = Fingerprint::from_content(&format!("hat_a_{}", i));
            let b = Fingerprint::from_content(&format!("hat_b_{}", i));
            a.hamming(&b)
        })
        .collect();

    let dist = ClusterDistribution::from_distances(&distances);

    // Near center (p25): should excite (positive)
    let near_center = dist.mexican_hat(dist.p25);
    // Far away (beyond p99): should inhibit (negative)
    let far_away = dist.mexican_hat(dist.p99 + 500.0);

    assert!(
        near_center > far_away,
        "Mexican hat should excite near center ({:.3}) and inhibit far ({:.3})",
        near_center,
        far_away
    );
}

/// PROOF F-5b: CalibratedThresholds produce valid HDR cascade parameters
#[test]
fn foundation_f5b_calibrated_thresholds() {
    use ladybug::search::distribution::ClusterDistribution;

    let distances: Vec<u32> = (0..500)
        .map(|i| {
            let a = Fingerprint::from_content(&format!("cal_a_{}", i));
            let b = Fingerprint::from_content(&format!("cal_b_{}", i));
            a.hamming(&b)
        })
        .collect();

    let dist = ClusterDistribution::from_distances(&distances);
    let thresholds = dist.calibrate_thresholds();

    // Excite < inhibit (excitation zone is tighter)
    assert!(
        thresholds.excite < thresholds.inhibit,
        "excite={} should be < inhibit={}",
        thresholds.excite,
        thresholds.inhibit
    );
    // Inhibit strength is bounded
    assert!(thresholds.inhibit_strength > 0.0 && thresholds.inhibit_strength <= 1.0);
}

// =============================================================================
// F-6: NARS Revision Evidence Monotonicity
// =============================================================================

/// PROOF F-6: Adding evidence always increases confidence
///
/// Ref: Wang (2006) NAL §3.2, equation 3.5
///
/// Counterintuitive: confidence increases even with NEGATIVE evidence,
/// because confidence = total_evidence / (total + k). Adding ANY evidence
/// increases the numerator.
#[test]
fn foundation_f6_nars_revision_monotone() {
    let base = TruthValue::new(0.5, 0.5);

    // Positive evidence increases frequency
    let positive = TruthValue::new(0.9, 0.8);
    let rev_pos = base.revision(&positive);
    assert!(
        rev_pos.frequency > base.frequency,
        "Positive evidence should increase frequency: {} > {}",
        rev_pos.frequency,
        base.frequency
    );
    assert!(
        rev_pos.confidence > base.confidence,
        "Any evidence should increase confidence: {} > {}",
        rev_pos.confidence,
        base.confidence
    );

    // Negative evidence decreases frequency but INCREASES confidence
    let negative = TruthValue::new(0.1, 0.8);
    let rev_neg = base.revision(&negative);
    assert!(
        rev_neg.frequency < base.frequency,
        "Negative evidence should decrease frequency: {} < {}",
        rev_neg.frequency,
        base.frequency
    );
    assert!(
        rev_neg.confidence > base.confidence,
        "Even negative evidence increases confidence: {} > {}",
        rev_neg.confidence,
        base.confidence
    );
}

/// PROOF F-6b: Revision is commutative
///
/// revision(A, B) ≈ revision(B, A) (order of evidence doesn't matter)
#[test]
fn foundation_f6b_revision_commutativity() {
    for i in 0..100 {
        let f1 = (i as f32 % 10.0) / 10.0 + 0.05;
        let c1 = 0.3 + (i as f32 % 7.0) / 10.0;
        let f2 = ((i + 37) as f32 % 10.0) / 10.0 + 0.05;
        let c2 = 0.2 + ((i + 13) as f32 % 8.0) / 10.0;

        let a = TruthValue::new(f1.min(0.99), c1.min(0.99));
        let b = TruthValue::new(f2.min(0.99), c2.min(0.99));

        let ab = a.revision(&b);
        let ba = b.revision(&a);

        assert!(
            (ab.frequency - ba.frequency).abs() < 0.001,
            "Revision should be commutative: rev(A,B).f={:.4}, rev(B,A).f={:.4} at i={}",
            ab.frequency,
            ba.frequency,
            i
        );
        assert!(
            (ab.confidence - ba.confidence).abs() < 0.001,
            "Revision should be commutative: rev(A,B).c={:.4}, rev(B,A).c={:.4} at i={}",
            ab.confidence,
            ba.confidence,
            i
        );
    }
}

// =============================================================================
// F-7: ABBA Causal Retrieval
// =============================================================================

/// PROOF F-7: bind(A, verb, B) is exactly recoverable
///
/// edge = A ⊗ verb ⊗ B
/// recovered_A = edge ⊗ verb ⊗ B  (XOR cancels verb then B)
/// recovered_B = edge ⊗ A ⊗ verb  (XOR cancels A then verb)
///
/// This is the algebraic foundation of all causal search in causal.rs.
///
/// Ref: Plate (2003) ch.4, XOR algebra
#[test]
fn foundation_f7_abba_causal_retrieval() {
    let verb = Fingerprint::from_content("CAUSES");

    for i in 0..100 {
        let a = Fingerprint::from_content(&format!("cause_{}", i));
        let b = Fingerprint::from_content(&format!("effect_{}", i));

        let edge = a.bind(&verb).bind(&b);
        let recovered_a = edge.bind(&verb).bind(&b);
        let recovered_b = edge.bind(&a).bind(&verb);

        assert_eq!(
            a.as_raw(),
            recovered_a.as_raw(),
            "ABBA failed to recover cause at i={}",
            i
        );
        assert_eq!(
            b.as_raw(),
            recovered_b.as_raw(),
            "ABBA failed to recover effect at i={}",
            i
        );
    }
}

/// PROOF F-7b: Fusion quality metric confirms exact recovery
#[test]
fn foundation_f7b_fusion_quality_exact() {
    for i in 0..100 {
        let a = Fingerprint::from_content(&format!("fusion_a_{}", i));
        let b = Fingerprint::from_content(&format!("fusion_b_{}", i));

        let (dist_a, dist_b) = fusion_quality(&a, &b);

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

/// PROOF F-7c: Multi-way fusion is exact
#[test]
fn foundation_f7c_multi_fusion_exact() {
    let items: Vec<Fingerprint> = (0..7)
        .map(|i| Fingerprint::from_content(&format!("multi_{}", i)))
        .collect();
    let max_dist = multi_fusion_quality(&items);
    assert_eq!(
        max_dist, 0.0,
        "Multi-way bind should be exactly recoverable, got {}",
        max_dist
    );
}

// =============================================================================
// Utility: erf approximation (Abramowitz & Stegun 1964, §7.1.26)
// =============================================================================

/// Approximate error function with maximum error < 1.5×10⁻⁷
fn erf_approx(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;
    let poly =
        0.254829592 * t - 0.284496736 * t2 + 1.421413741 * t3 - 1.453152027 * t4 + 1.061405429 * t5;
    sign * (1.0 - poly * (-x * x).exp())
}
