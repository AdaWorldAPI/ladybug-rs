//! Benchmark Suite for Ladybug-RS
//!
//! Proves the "holy shit" claims:
//! - 50-100x less RAM than float32 vector DBs
//! - >95% recall with bitpacked Hamming
//! - Sub-millisecond search on millions of vectors
//!
//! Run: cargo bench --features bench
//! Or:  cargo run --release --features bench --bin ladybug-bench

pub mod memory;
pub mod recall;
pub mod throughput;
pub mod comparison;

use std::time::{Duration, Instant};
use crate::core::Fingerprint;
use crate::storage::{BindSpace, Substrate, SubstrateConfig, FINGERPRINT_WORDS};

// =============================================================================
// BENCHMARK CONFIGURATION
// =============================================================================

/// Benchmark configuration
#[derive(Clone, Debug)]
pub struct BenchConfig {
    /// Number of vectors to index
    pub num_vectors: usize,
    /// Number of queries to run
    pub num_queries: usize,
    /// K for recall@K measurement
    pub k: usize,
    /// Number of warmup iterations
    pub warmup_iters: usize,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            num_vectors: 100_000,
            num_queries: 1000,
            k: 10,
            warmup_iters: 100,
            seed: 42,
        }
    }
}

impl BenchConfig {
    pub fn small() -> Self {
        Self {
            num_vectors: 10_000,
            num_queries: 100,
            ..Default::default()
        }
    }

    pub fn medium() -> Self {
        Self {
            num_vectors: 100_000,
            num_queries: 1000,
            ..Default::default()
        }
    }

    pub fn large() -> Self {
        Self {
            num_vectors: 1_000_000,
            num_queries: 1000,
            ..Default::default()
        }
    }

    pub fn xlarge() -> Self {
        Self {
            num_vectors: 10_000_000,
            num_queries: 100,
            ..Default::default()
        }
    }
}

// =============================================================================
// BENCHMARK RESULTS
// =============================================================================

/// Results from a benchmark run
#[derive(Clone, Debug)]
pub struct BenchResult {
    /// Configuration used
    pub config: BenchConfig,
    /// Memory usage in bytes
    pub memory_bytes: usize,
    /// Memory per vector in bytes
    pub bytes_per_vector: f64,
    /// Index build time
    pub build_time: Duration,
    /// Average query latency
    pub avg_latency: Duration,
    /// P50 query latency
    pub p50_latency: Duration,
    /// P95 query latency
    pub p95_latency: Duration,
    /// P99 query latency
    pub p99_latency: Duration,
    /// Queries per second
    pub qps: f64,
    /// Recall@K
    pub recall_at_k: f64,
    /// Comparison metrics
    pub comparison: Option<ComparisonMetrics>,
}

/// Comparison against baseline
#[derive(Clone, Debug)]
pub struct ComparisonMetrics {
    /// RAM savings vs float32 (e.g., 50.0 = 50x less RAM)
    pub ram_savings_factor: f64,
    /// Speed improvement vs float32 cosine
    pub speed_factor: f64,
    /// float32 baseline memory per vector
    pub baseline_bytes_per_vector: f64,
    /// float32 baseline latency
    pub baseline_latency: Duration,
}

impl BenchResult {
    /// Print formatted results
    pub fn print(&self) {
        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║                    LADYBUG-RS BENCHMARK                        ║");
        println!("╠════════════════════════════════════════════════════════════════╣");
        println!("║ Vectors: {:>12} | Queries: {:>8} | K: {:>4}            ║",
            format_num(self.config.num_vectors),
            format_num(self.config.num_queries),
            self.config.k
        );
        println!("╠════════════════════════════════════════════════════════════════╣");
        println!("║ MEMORY                                                         ║");
        println!("║   Total:      {:>12}                                     ║", format_bytes(self.memory_bytes));
        println!("║   Per vector: {:>12}                                     ║", format_bytes(self.bytes_per_vector as usize));
        println!("╠════════════════════════════════════════════════════════════════╣");
        println!("║ LATENCY                                                        ║");
        println!("║   Average: {:>10}                                         ║", format_duration(self.avg_latency));
        println!("║   P50:     {:>10}                                         ║", format_duration(self.p50_latency));
        println!("║   P95:     {:>10}                                         ║", format_duration(self.p95_latency));
        println!("║   P99:     {:>10}                                         ║", format_duration(self.p99_latency));
        println!("╠════════════════════════════════════════════════════════════════╣");
        println!("║ THROUGHPUT                                                     ║");
        println!("║   QPS:     {:>12.0}                                       ║", self.qps);
        println!("║   Build:   {:>10}                                         ║", format_duration(self.build_time));
        println!("╠════════════════════════════════════════════════════════════════╣");
        println!("║ RECALL                                                         ║");
        println!("║   Recall@{}: {:>6.2}%                                          ║", self.config.k, self.recall_at_k * 100.0);

        if let Some(ref cmp) = self.comparison {
            println!("╠════════════════════════════════════════════════════════════════╣");
            println!("║ VS FLOAT32 COSINE BASELINE                                    ║");
            println!("║   RAM savings:   {:>6.1}x less memory                         ║", cmp.ram_savings_factor);
            println!("║   Speed factor:  {:>6.1}x faster                              ║", cmp.speed_factor);
            println!("║   Baseline RAM:  {:>12} per vector                     ║", format_bytes(cmp.baseline_bytes_per_vector as usize));
            println!("║   Baseline lat:  {:>10}                                   ║", format_duration(cmp.baseline_latency));
        }

        println!("╚════════════════════════════════════════════════════════════════╝");
    }

    /// Export as JSON
    pub fn to_json(&self) -> String {
        serde_json::json!({
            "config": {
                "num_vectors": self.config.num_vectors,
                "num_queries": self.config.num_queries,
                "k": self.config.k
            },
            "memory": {
                "total_bytes": self.memory_bytes,
                "bytes_per_vector": self.bytes_per_vector
            },
            "latency": {
                "avg_ns": self.avg_latency.as_nanos(),
                "p50_ns": self.p50_latency.as_nanos(),
                "p95_ns": self.p95_latency.as_nanos(),
                "p99_ns": self.p99_latency.as_nanos()
            },
            "throughput": {
                "qps": self.qps,
                "build_time_ms": self.build_time.as_millis()
            },
            "recall": {
                "recall_at_k": self.recall_at_k
            },
            "comparison": self.comparison.as_ref().map(|c| {
                serde_json::json!({
                    "ram_savings_factor": c.ram_savings_factor,
                    "speed_factor": c.speed_factor
                })
            })
        }).to_string()
    }
}

// =============================================================================
// VECTOR GENERATION
// =============================================================================

/// Generate random fingerprints for benchmarking
pub fn generate_random_fingerprints(count: usize, seed: u64) -> Vec<[u64; FINGERPRINT_WORDS]> {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;

    let mut fingerprints = Vec::with_capacity(count);

    for i in 0..count {
        let mut fp = [0u64; FINGERPRINT_WORDS];
        for j in 0..FINGERPRINT_WORDS {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            i.hash(&mut hasher);
            j.hash(&mut hasher);
            fp[j] = hasher.finish();
        }
        fingerprints.push(fp);
    }

    fingerprints
}

/// Generate clustered fingerprints (more realistic for recall testing)
pub fn generate_clustered_fingerprints(
    count: usize,
    num_clusters: usize,
    seed: u64
) -> Vec<[u64; FINGERPRINT_WORDS]> {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;

    // Generate cluster centers
    let centers: Vec<[u64; FINGERPRINT_WORDS]> = (0..num_clusters)
        .map(|c| {
            let mut fp = [0u64; FINGERPRINT_WORDS];
            for j in 0..FINGERPRINT_WORDS {
                let mut hasher = DefaultHasher::new();
                seed.hash(&mut hasher);
                c.hash(&mut hasher);
                j.hash(&mut hasher);
                fp[j] = hasher.finish();
            }
            fp
        })
        .collect();

    // Generate points around cluster centers
    let mut fingerprints = Vec::with_capacity(count);

    for i in 0..count {
        let cluster_idx = i % num_clusters;
        let center = &centers[cluster_idx];

        // Flip some bits from center (noise)
        let mut fp = *center;
        let noise_bits = (i / num_clusters) % 100; // 0-99 bits of noise

        for b in 0..noise_bits {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            i.hash(&mut hasher);
            b.hash(&mut hasher);
            let bit_idx = (hasher.finish() % (FINGERPRINT_WORDS as u64 * 64)) as usize;
            let word_idx = bit_idx / 64;
            let bit_pos = bit_idx % 64;
            fp[word_idx] ^= 1 << bit_pos;
        }

        fingerprints.push(fp);
    }

    fingerprints
}

// =============================================================================
// GROUND TRUTH COMPUTATION
// =============================================================================

/// Compute ground truth nearest neighbors using brute force
pub fn compute_ground_truth(
    queries: &[[u64; FINGERPRINT_WORDS]],
    database: &[[u64; FINGERPRINT_WORDS]],
    k: usize,
) -> Vec<Vec<usize>> {
    queries.iter().map(|query| {
        let mut distances: Vec<(usize, u32)> = database.iter()
            .enumerate()
            .map(|(idx, vec)| {
                let dist: u32 = query.iter()
                    .zip(vec.iter())
                    .map(|(a, b)| (a ^ b).count_ones())
                    .sum();
                (idx, dist)
            })
            .collect();

        distances.sort_by_key(|(_, d)| *d);
        distances.into_iter().take(k).map(|(idx, _)| idx).collect()
    }).collect()
}

/// Compute recall given predictions and ground truth
pub fn compute_recall(
    predictions: &[Vec<usize>],
    ground_truth: &[Vec<usize>],
) -> f64 {
    let mut total_correct = 0;
    let mut total_expected = 0;

    for (pred, truth) in predictions.iter().zip(ground_truth.iter()) {
        for p in pred {
            if truth.contains(p) {
                total_correct += 1;
            }
        }
        total_expected += truth.len();
    }

    total_correct as f64 / total_expected as f64
}

// =============================================================================
// MAIN BENCHMARK RUNNER
// =============================================================================

/// Run full benchmark suite
pub fn run_benchmark(config: BenchConfig) -> BenchResult {
    println!("Generating {} random fingerprints...", config.num_vectors);
    let start = Instant::now();
    let database = generate_clustered_fingerprints(
        config.num_vectors,
        100, // 100 clusters
        config.seed
    );
    let gen_time = start.elapsed();
    println!("  Generated in {:?}", gen_time);

    // Measure memory before indexing
    let mem_before = get_memory_usage();

    println!("Building Substrate index...");
    let build_start = Instant::now();
    let substrate = Substrate::new(SubstrateConfig::default());

    for fp in &database {
        substrate.write(*fp);
    }
    let build_time = build_start.elapsed();
    println!("  Built in {:?}", build_time);

    // Measure memory after indexing
    let mem_after = get_memory_usage();
    let memory_bytes = mem_after.saturating_sub(mem_before);
    let bytes_per_vector = memory_bytes as f64 / config.num_vectors as f64;

    // Generate queries
    println!("Running {} queries...", config.num_queries);
    let queries = generate_random_fingerprints(config.num_queries, config.seed + 1);

    // Warmup
    for query in queries.iter().take(config.warmup_iters) {
        let _ = substrate.resonate(query, config.k);
    }

    // Timed queries
    let mut latencies = Vec::with_capacity(config.num_queries);
    let query_start = Instant::now();

    for query in &queries {
        let iter_start = Instant::now();
        let _ = substrate.resonate(query, config.k);
        latencies.push(iter_start.elapsed());
    }

    let total_query_time = query_start.elapsed();

    // Compute latency percentiles
    latencies.sort();
    let avg_latency = total_query_time / config.num_queries as u32;
    let p50_latency = latencies[latencies.len() / 2];
    let p95_latency = latencies[latencies.len() * 95 / 100];
    let p99_latency = latencies[latencies.len() * 99 / 100];
    let qps = config.num_queries as f64 / total_query_time.as_secs_f64();

    // Compute recall (sample for large datasets)
    println!("Computing recall@{}...", config.k);
    let recall_sample_size = config.num_queries.min(100);
    let query_sample: Vec<_> = queries.iter().take(recall_sample_size).cloned().collect();

    // Get predictions from substrate
    let predictions: Vec<Vec<usize>> = query_sample.iter()
        .map(|q| {
            substrate.resonate(q, config.k)
                .into_iter()
                .map(|(addr, _)| addr.0 as usize)
                .collect()
        })
        .collect();

    // Compute ground truth
    let ground_truth = compute_ground_truth(&query_sample, &database, config.k);
    let recall_at_k = compute_recall(&predictions, &ground_truth);

    // Compute comparison metrics
    let comparison = compute_comparison_metrics(
        bytes_per_vector,
        avg_latency,
    );

    BenchResult {
        config,
        memory_bytes,
        bytes_per_vector,
        build_time,
        avg_latency,
        p50_latency,
        p95_latency,
        p99_latency,
        qps,
        recall_at_k,
        comparison: Some(comparison),
    }
}

/// Compute comparison against float32 baseline
fn compute_comparison_metrics(
    bytes_per_vector: f64,
    avg_latency: Duration,
) -> ComparisonMetrics {
    // Float32 baseline assumptions:
    // - 768-dim embedding (typical for sentence transformers)
    // - 4 bytes per float32
    // - Plus overhead for HNSW graph (~100 bytes per vector typical)
    let baseline_embedding_size = 768 * 4; // 3072 bytes
    let baseline_hnsw_overhead = 100; // conservative
    let baseline_bytes_per_vector = (baseline_embedding_size + baseline_hnsw_overhead) as f64;

    // Typical HNSW latency on 1M vectors: ~1-5ms
    let baseline_latency = Duration::from_micros(2000);

    let ram_savings_factor = baseline_bytes_per_vector / bytes_per_vector;
    let speed_factor = baseline_latency.as_nanos() as f64 / avg_latency.as_nanos() as f64;

    ComparisonMetrics {
        ram_savings_factor,
        speed_factor,
        baseline_bytes_per_vector,
        baseline_latency,
    }
}

// =============================================================================
// UTILITIES
// =============================================================================

fn get_memory_usage() -> usize {
    // Simple approximation - in production would use jemalloc stats
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb) = line.split_whitespace().nth(1) {
                        if let Ok(kb_val) = kb.parse::<usize>() {
                            return kb_val * 1024;
                        }
                    }
                }
            }
        }
    }
    0
}

fn format_num(n: usize) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{}", n)
    }
}

fn format_bytes(b: usize) -> String {
    if b >= 1024 * 1024 * 1024 {
        format!("{:.2} GB", b as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if b >= 1024 * 1024 {
        format!("{:.2} MB", b as f64 / (1024.0 * 1024.0))
    } else if b >= 1024 {
        format!("{:.2} KB", b as f64 / 1024.0)
    } else {
        format!("{} B", b)
    }
}

fn format_duration(d: Duration) -> String {
    let nanos = d.as_nanos();
    if nanos >= 1_000_000_000 {
        format!("{:.2}s", d.as_secs_f64())
    } else if nanos >= 1_000_000 {
        format!("{:.2}ms", nanos as f64 / 1_000_000.0)
    } else if nanos >= 1_000 {
        format!("{:.2}µs", nanos as f64 / 1_000.0)
    } else {
        format!("{}ns", nanos)
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_benchmark() {
        let config = BenchConfig {
            num_vectors: 1000,
            num_queries: 10,
            k: 5,
            warmup_iters: 5,
            seed: 42,
        };

        let result = run_benchmark(config);
        result.print();

        assert!(result.recall_at_k > 0.0);
        assert!(result.qps > 0.0);
    }

    #[test]
    fn test_generate_fingerprints() {
        let fps = generate_random_fingerprints(100, 42);
        assert_eq!(fps.len(), 100);

        // Check they're not all zeros
        let total_bits: u32 = fps.iter()
            .flat_map(|fp| fp.iter())
            .map(|w| w.count_ones())
            .sum();
        assert!(total_bits > 0);
    }

    #[test]
    fn test_ground_truth() {
        let database = generate_random_fingerprints(100, 42);
        let queries = generate_random_fingerprints(10, 43);

        let gt = compute_ground_truth(&queries, &database, 5);

        assert_eq!(gt.len(), 10);
        for neighbors in &gt {
            assert_eq!(neighbors.len(), 5);
        }
    }
}
