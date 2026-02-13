//! Temporal Effect Size — Granger-style temporal causality on fingerprint sequences.
//!
//! Measures whether one fingerprint series predicts another beyond autocorrelation.
//!
//! # Science
//! - Granger (1969): Temporal causality test
//! - Cohen (1988): Effect size d = (μ₁ - μ₂)/σ_pooled
//! - Plate (2003): Permutation for temporal position binding

use crate::search::hdr_cascade::{WORDS, hamming_distance};

/// Temporal effect size between two fingerprint series.
#[derive(Debug, Clone)]
pub struct TemporalEffectSize {
    /// Cohen's d across time lag
    pub effect_d: f32,
    /// Granger signal: asymmetric temporal prediction strength
    /// Positive = series_a predicts series_b beyond autocorrelation
    pub granger_signal: f32,
    /// Optimal time lag (in steps)
    pub lag: usize,
    /// Standard error of the estimate
    pub std_error: f32,
}

/// Compute Granger-style temporal effect size.
///
/// For each lag τ: `d(A_t, B_{t+τ}) - d(B_t, B_{t+τ})` = Granger signal.
/// If positive: A's past predicts B's future beyond B's own autocorrelation.
pub fn granger_effect(
    series_a: &[[u64; WORDS]],
    series_b: &[[u64; WORDS]],
    max_lag: usize,
) -> Option<TemporalEffectSize> {
    if series_a.len() < 3 || series_b.len() < 3 {
        return None;
    }
    let min_len = series_a.len().min(series_b.len());
    let max_lag = max_lag.min(min_len / 2);
    if max_lag == 0 {
        return None;
    }

    let mut best_signal = 0.0f32;
    let mut best_lag = 1;
    let mut best_effect_d = 0.0f32;
    let mut best_std_error = 0.0f32;

    for lag in 1..=max_lag {
        let n = min_len - lag;
        if n == 0 {
            continue;
        }

        // Cross-series distances: d(A_t, B_{t+lag})
        let cross: Vec<f32> = (0..n)
            .map(|t| hamming_distance(&series_a[t], &series_b[t + lag]) as f32)
            .collect();

        // Auto-correlation distances: d(B_t, B_{t+lag})
        let auto: Vec<f32> = (0..n)
            .map(|t| hamming_distance(&series_b[t], &series_b[t + lag]) as f32)
            .collect();

        let cross_mean = cross.iter().sum::<f32>() / n as f32;
        let auto_mean = auto.iter().sum::<f32>() / n as f32;

        let cross_var = if n > 1 {
            cross.iter().map(|d| (d - cross_mean).powi(2)).sum::<f32>() / (n - 1) as f32
        } else {
            1.0
        };
        let auto_var = if n > 1 {
            auto.iter().map(|d| (d - auto_mean).powi(2)).sum::<f32>() / (n - 1) as f32
        } else {
            1.0
        };

        let pooled_sigma = ((cross_var + auto_var) / 2.0).sqrt();

        // Granger signal: if cross_mean < auto_mean, A predicts B better than B predicts itself
        let signal = auto_mean - cross_mean;
        let effect_d = if pooled_sigma > 0.0 {
            signal / pooled_sigma
        } else {
            0.0
        };
        let std_error = if n > 1 {
            pooled_sigma / (n as f32).sqrt()
        } else {
            f32::MAX
        };

        if signal > best_signal {
            best_signal = signal;
            best_lag = lag;
            best_effect_d = effect_d;
            best_std_error = std_error;
        }
    }

    Some(TemporalEffectSize {
        effect_d: best_effect_d,
        granger_signal: best_signal,
        lag: best_lag,
        std_error: best_std_error,
    })
}

/// Compute temporal autocorrelation profile for a single series.
///
/// Returns (lag, mean_distance) pairs showing how quickly the series decorrelates.
pub fn autocorrelation_profile(series: &[[u64; WORDS]], max_lag: usize) -> Vec<(usize, f32)> {
    let max_lag = max_lag.min(series.len() / 2);
    (1..=max_lag)
        .map(|lag| {
            let n = series.len() - lag;
            let mean_dist: f32 = (0..n)
                .map(|t| hamming_distance(&series[t], &series[t + lag]) as f32)
                .sum::<f32>()
                / n as f32;
            (lag, mean_dist)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_series(n: usize, seed: u64) -> Vec<[u64; WORDS]> {
        let mut series = Vec::with_capacity(n);
        let mut state = seed;
        for _ in 0..n {
            let mut fp = [0u64; WORDS];
            for w in fp.iter_mut() {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *w = state;
            }
            series.push(fp);
        }
        series
    }

    #[test]
    fn test_granger_effect_random() {
        let a = make_series(20, 42);
        let b = make_series(20, 99);
        let result = granger_effect(&a, &b, 5);
        assert!(result.is_some());
        let tes = result.unwrap();
        assert!(tes.lag >= 1);
        assert!(tes.lag <= 5);
        // Random series should have weak signal
        assert!(tes.effect_d.abs() < 3.0);
    }

    #[test]
    fn test_granger_effect_self_prediction() {
        // A series should predict itself with lag 0 perfectly,
        // and weakly at lag > 0 for random data
        let a = make_series(20, 42);
        let result = granger_effect(&a, &a, 5);
        assert!(result.is_some());
    }

    #[test]
    fn test_granger_too_short() {
        let a = make_series(2, 42);
        let b = make_series(2, 99);
        assert!(granger_effect(&a, &b, 5).is_none());
    }

    #[test]
    fn test_autocorrelation_profile() {
        let series = make_series(20, 42);
        let profile = autocorrelation_profile(&series, 5);
        assert_eq!(profile.len(), 5);
        // Random series: autocorrelation should be ~flat (all distances ~8192)
        for (lag, dist) in &profile {
            assert!(*lag >= 1 && *lag <= 5);
            assert!(*dist > 0.0);
        }
    }
}
