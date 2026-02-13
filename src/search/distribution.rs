//! Cluster Radius Profile (CRP) — distribution statistics for fingerprint clusters.
//!
//! Implements the CRP struct from CLAM_HARDENING.md §6.
//! Provides calibrated distance thresholds via Normal approximation
//! at d=16384 (Berry-Esseen error < 0.004).
//!
//! # Science
//! - Fisher (1925): (μ, σ) are sufficient statistics for Normal family
//! - Berry-Esseen (1941/42): Normal approximation error < C·ρ/(σ³·√n)
//! - Cohen (1988): CRP percentiles → calibrated effect size thresholds

/// Cluster Radius Profile: percentile-based distance distribution.
///
/// Computed from observed Hamming distances within a cluster.
/// Mexican hat response is calibrated from these percentiles.
#[derive(Debug, Clone)]
pub struct ClusterDistribution {
    /// Mean Hamming distance
    pub mu: f32,
    /// Standard deviation
    pub sigma: f32,
    /// 25th percentile (first quartile)
    pub p25: f32,
    /// 50th percentile (median)
    pub p50: f32,
    /// 75th percentile (third quartile)
    pub p75: f32,
    /// 95th percentile
    pub p95: f32,
    /// 99th percentile
    pub p99: f32,
    /// 16-bin INT4 histogram of distances
    pub histogram_int4: [u16; 16],
    /// Sample count
    pub n: usize,
}

/// Berry-Esseen noise floor at d=16384.
/// Below this normalized distance, differences are indistinguishable from random.
pub const BERRY_ESSEEN_NOISE_FLOOR: f32 = 0.004;

/// Total bits in a fingerprint.
const TOTAL_BITS: f32 = 16384.0;

impl ClusterDistribution {
    /// Compute CRP from observed Hamming distances.
    ///
    /// Uses Normal approximation for percentiles (valid at d=16384
    /// by Berry-Esseen theorem, error < 0.004).
    pub fn from_distances(distances: &[u32]) -> Self {
        if distances.is_empty() {
            return Self::empty();
        }

        let n = distances.len() as f32;
        let mu = distances.iter().sum::<u32>() as f32 / n;
        let sigma = if distances.len() > 1 {
            (distances
                .iter()
                .map(|d| (*d as f32 - mu).powi(2))
                .sum::<f32>()
                / (n - 1.0))
                .sqrt()
        } else {
            0.0
        };

        // CRP from Normal(μ, σ) — z-scores for standard percentiles
        Self {
            mu,
            sigma,
            p25: (mu - 0.6745 * sigma).max(0.0),
            p50: mu,
            p75: mu + 0.6745 * sigma,
            p95: mu + 1.6449 * sigma,
            p99: mu + 2.3263 * sigma,
            histogram_int4: Self::build_histogram_int4(distances),
            n: distances.len(),
        }
    }

    /// Empty distribution (no data).
    pub fn empty() -> Self {
        Self {
            mu: 0.0,
            sigma: 0.0,
            p25: 0.0,
            p50: 0.0,
            p75: 0.0,
            p95: 0.0,
            p99: 0.0,
            histogram_int4: [0; 16],
            n: 0,
        }
    }

    /// Coefficient of variation: σ/μ.
    /// High CV → spread-out cluster → more resolution needed.
    #[inline]
    pub fn coefficient_of_variation(&self) -> f32 {
        if self.mu > 0.0 {
            self.sigma / self.mu
        } else {
            0.0
        }
    }

    /// Interquartile range: p75 - p25.
    #[inline]
    pub fn iqr(&self) -> f32 {
        self.p75 - self.p25
    }

    /// Mexican hat response calibrated from CRP percentiles.
    ///
    /// ```text
    /// distance < p25  → 1.0   (excite: strong match)
    /// distance < p75  → 0.5   (accept: moderate)
    /// distance < p95  → 0.0   (neutral)
    /// distance < p99  → -0.5  (inhibit)
    /// distance >= p99 → -1.0  (reject)
    /// ```
    pub fn mexican_hat(&self, distance: f32) -> f32 {
        if distance < self.p25 {
            1.0
        } else if distance < self.p75 {
            0.5
        } else if distance < self.p95 {
            0.0
        } else if distance < self.p99 {
            -0.5
        } else {
            -1.0
        }
    }

    /// Z-score: how many standard deviations from mean.
    #[inline]
    pub fn z_score(&self, distance: f32) -> f32 {
        if self.sigma > 0.0 {
            (distance - self.mu) / self.sigma
        } else {
            0.0
        }
    }

    /// Cohen's d effect size between this cluster and another.
    pub fn cohens_d(&self, other: &ClusterDistribution) -> f32 {
        let pooled_sigma = ((self.sigma.powi(2) + other.sigma.powi(2)) / 2.0).sqrt();
        if pooled_sigma > 0.0 {
            (self.mu - other.mu).abs() / pooled_sigma
        } else {
            0.0
        }
    }

    /// Shannon entropy of the INT4 histogram (bits).
    pub fn entropy(&self) -> f32 {
        let total: f32 = self.histogram_int4.iter().sum::<u16>() as f32;
        if total == 0.0 {
            return 0.0;
        }
        self.histogram_int4
            .iter()
            .filter(|&&count| count > 0)
            .map(|&count| {
                let p = count as f32 / total;
                -p * p.log2()
            })
            .sum()
    }

    /// Build 16-bin histogram from distances.
    /// Bin width = max_distance / 16, capped at TOTAL_BITS.
    fn build_histogram_int4(distances: &[u32]) -> [u16; 16] {
        let mut hist = [0u16; 16];
        if distances.is_empty() {
            return hist;
        }
        let max_d = *distances.iter().max().unwrap_or(&1) as f32;
        let bin_width = if max_d > 0.0 { max_d / 16.0 } else { 1.0 };
        for &d in distances {
            let bin = ((d as f32 / bin_width) as usize).min(15);
            hist[bin] = hist[bin].saturating_add(1);
        }
        hist
    }
}

/// Calibrated thresholds for HDR cascade from CRP percentiles.
///
/// Maps CRP percentiles to HDR cascade thresholds:
/// - excite = p25 (strong match zone)
/// - inhibit = p95 (rejection zone)
/// - inhibit_strength derived from IQR width
///
/// # Science
/// - Cohen (1988): CRP percentiles → calibrated effect size thresholds
/// - Hartigan (1975): Density estimation from percentile statistics
#[derive(Debug, Clone)]
pub struct CalibratedThresholds {
    /// Excitation threshold (p25 as u32)
    pub excite: u32,
    /// Inhibition threshold (p95 as u32)
    pub inhibit: u32,
    /// Inhibition strength (derived from IQR width)
    pub inhibit_strength: f32,
    /// L1 sketch threshold (p75 / 64, for per-word 1-bit check)
    pub sketch_l1: u32,
    /// L2 sketch threshold (p50 as u32, for 4-bit check)
    pub sketch_l2: u32,
}

impl ClusterDistribution {
    /// Derive calibrated HDR cascade thresholds from this CRP.
    ///
    /// Translates statistical percentiles into concrete search parameters.
    pub fn calibrate_thresholds(&self) -> CalibratedThresholds {
        let iqr = self.iqr();
        let inhibit_strength = if self.mu > 0.0 {
            (iqr / self.mu).min(1.0)
        } else {
            0.5
        };

        CalibratedThresholds {
            excite: self.p25 as u32,
            inhibit: self.p95 as u32,
            inhibit_strength,
            sketch_l1: (self.p75 / 64.0) as u32, // Per-word threshold for 1-bit sketch
            sketch_l2: self.p50 as u32,
        }
    }
}

/// Distortion report: measures information loss and semantic drift
/// between an original fingerprint and a transformed version.
///
/// # Science
/// - Shannon (1948): Channel capacity and noise detection
/// - Berry-Esseen: Noise floor at 0.004 distinguishes distortion from natural variation
/// - Cohen (1988): Z-score > 2.0 = statistically significant distortion (p < 0.05)
#[derive(Debug, Clone)]
pub struct DistortionReport {
    /// Bits lost beyond noise floor, normalized to [0, 1]
    pub information_loss: f32,
    /// Distance relative to cluster mean (structural drift)
    pub structural_drift: f32,
    /// Z-score: how many σ from cluster center
    pub semantic_z: f32,
    /// Whether distortion exceeds statistical significance (|z| > 2.0)
    pub significant: bool,
}

/// Detect distortion between original and transformed fingerprints,
/// calibrated against a cluster's CRP distribution.
pub fn detect_distortion(raw_distance: u32, corpus_dist: &ClusterDistribution) -> DistortionReport {
    let noise_floor = corpus_dist.sigma * BERRY_ESSEEN_NOISE_FLOOR;
    let d = raw_distance as f32;
    let z = corpus_dist.z_score(d);

    DistortionReport {
        information_loss: (d - noise_floor).max(0.0) / TOTAL_BITS,
        structural_drift: if corpus_dist.mu > 0.0 {
            d / corpus_dist.mu
        } else {
            0.0
        },
        semantic_z: z,
        significant: z.abs() > 2.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cluster_distribution_basic() {
        let distances = vec![100, 200, 300, 400, 500];
        let dist = ClusterDistribution::from_distances(&distances);
        assert!((dist.mu - 300.0).abs() < 0.1);
        assert!(dist.sigma > 0.0);
        assert!(dist.p25 < dist.p50);
        assert!(dist.p50 < dist.p75);
        assert!(dist.p75 < dist.p95);
        assert!(dist.p95 < dist.p99);
    }

    #[test]
    fn test_cluster_distribution_empty() {
        let dist = ClusterDistribution::from_distances(&[]);
        assert_eq!(dist.mu, 0.0);
        assert_eq!(dist.n, 0);
    }

    #[test]
    fn test_mexican_hat_zones() {
        let distances: Vec<u32> = (1000..2000).collect();
        let dist = ClusterDistribution::from_distances(&distances);
        // Well within p25 → excite
        assert_eq!(dist.mexican_hat(dist.p25 - 100.0), 1.0);
        // Between p75 and p95 → neutral
        assert_eq!(dist.mexican_hat((dist.p75 + dist.p95) / 2.0), 0.0);
        // Beyond p99 → reject
        assert_eq!(dist.mexican_hat(dist.p99 + 100.0), -1.0);
    }

    #[test]
    fn test_coefficient_of_variation() {
        let dist = ClusterDistribution::from_distances(&[100, 100, 100]);
        assert_eq!(dist.coefficient_of_variation(), 0.0);
        let dist2 = ClusterDistribution::from_distances(&[100, 200, 300]);
        assert!(dist2.coefficient_of_variation() > 0.0);
    }

    #[test]
    fn test_entropy() {
        let dist = ClusterDistribution::from_distances(&[100, 200, 300, 400, 500]);
        assert!(dist.entropy() > 0.0);
    }

    #[test]
    fn test_cohens_d() {
        let a = ClusterDistribution::from_distances(&[100, 110, 120, 130, 140]);
        let b = ClusterDistribution::from_distances(&[900, 910, 920, 930, 940]);
        let d = a.cohens_d(&b);
        assert!(d > 5.0); // Large effect size
    }

    #[test]
    fn test_distortion_detection() {
        let dist = ClusterDistribution::from_distances(&(4000..5000u32).collect::<Vec<_>>());
        // Distance within normal range
        let report = detect_distortion(4500, &dist);
        assert!(!report.significant);
        // Distance far outside range
        let report2 = detect_distortion(8000, &dist);
        assert!(report2.significant);
        assert!(report2.semantic_z > 2.0);
    }

    #[test]
    fn test_calibrated_thresholds() {
        let distances: Vec<u32> = (1000..2000).collect();
        let dist = ClusterDistribution::from_distances(&distances);
        let thresholds = dist.calibrate_thresholds();

        // excite should be p25
        assert_eq!(thresholds.excite, dist.p25 as u32);
        // inhibit should be p95
        assert_eq!(thresholds.inhibit, dist.p95 as u32);
        // excite < inhibit
        assert!(thresholds.excite < thresholds.inhibit);
        // inhibit_strength should be reasonable
        assert!(thresholds.inhibit_strength > 0.0);
        assert!(thresholds.inhibit_strength <= 1.0);
    }

    #[test]
    fn test_z_score() {
        let dist = ClusterDistribution::from_distances(&[4000, 4100, 4200, 4300, 4400]);
        let z_at_mean = dist.z_score(dist.mu);
        assert!(z_at_mean.abs() < 0.01);
    }
}
