//! Throughput benchmarks - proves sub-millisecond search
//!
//! Measures:
//! - Queries per second (QPS)
//! - Latency percentiles (p50, p95, p99)
//! - Batch throughput

use super::*;
use std::time::Instant;

/// Throughput measurement results
#[derive(Debug, Clone)]
pub struct ThroughputReport {
    pub num_vectors: usize,
    pub num_queries: usize,
    pub avg_latency_ns: u64,
    pub p50_latency_ns: u64,
    pub p95_latency_ns: u64,
    pub p99_latency_ns: u64,
    pub min_latency_ns: u64,
    pub max_latency_ns: u64,
    pub qps: f64,
    pub batch_qps: f64,
}

impl ThroughputReport {
    pub fn print(&self) {
        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║                    THROUGHPUT MEASUREMENT                      ║");
        println!("║                    {} vectors, {} queries               ║",
            format_num(self.num_vectors), format_num(self.num_queries));
        println!("╠════════════════════════════════════════════════════════════════╣");
        println!("║ LATENCY                                                        ║");
        println!("║   Average:  {:>12}                                       ║", format_ns(self.avg_latency_ns));
        println!("║   P50:      {:>12}                                       ║", format_ns(self.p50_latency_ns));
        println!("║   P95:      {:>12}                                       ║", format_ns(self.p95_latency_ns));
        println!("║   P99:      {:>12}                                       ║", format_ns(self.p99_latency_ns));
        println!("║   Min:      {:>12}                                       ║", format_ns(self.min_latency_ns));
        println!("║   Max:      {:>12}                                       ║", format_ns(self.max_latency_ns));
        println!("╠════════════════════════════════════════════════════════════════╣");
        println!("║ THROUGHPUT                                                     ║");
        println!("║   Single-query QPS:  {:>12.0}                             ║", self.qps);
        println!("║   Batch QPS:         {:>12.0}                             ║", self.batch_qps);
        println!("╚════════════════════════════════════════════════════════════════╝");

        // Performance tier
        let tier = if self.p99_latency_ns < 1_000_000 {
            "SUB-MILLISECOND (EXCELLENT)"
        } else if self.p99_latency_ns < 10_000_000 {
            "LOW LATENCY (GOOD)"
        } else {
            "STANDARD"
        };
        println!("\nPerformance tier: {}", tier);
    }
}

/// Measure throughput with detailed latency breakdown
pub fn measure_throughput(
    database: &[[u64; FINGERPRINT_WORDS]],
    queries: &[[u64; FINGERPRINT_WORDS]],
    k: usize,
    warmup_iters: usize,
) -> ThroughputReport {
    // Build index
    let substrate = Substrate::new(SubstrateConfig::default());
    for fp in database {
        substrate.write(*fp);
    }

    // Warmup
    for query in queries.iter().take(warmup_iters) {
        let _ = substrate.resonate(query, k);
    }

    // Measure single-query latencies
    let mut latencies: Vec<u64> = Vec::with_capacity(queries.len());

    for query in queries {
        let start = Instant::now();
        let _ = substrate.resonate(query, k);
        latencies.push(start.elapsed().as_nanos() as u64);
    }

    // Sort for percentiles
    latencies.sort();

    let avg_latency_ns = latencies.iter().sum::<u64>() / latencies.len() as u64;
    let p50_latency_ns = latencies[latencies.len() / 2];
    let p95_latency_ns = latencies[latencies.len() * 95 / 100];
    let p99_latency_ns = latencies[latencies.len() * 99 / 100];
    let min_latency_ns = *latencies.first().unwrap();
    let max_latency_ns = *latencies.last().unwrap();

    let total_time_ns: u64 = latencies.iter().sum();
    let qps = queries.len() as f64 / (total_time_ns as f64 / 1_000_000_000.0);

    // Measure batch throughput
    let batch_size = 100;
    let batch_start = Instant::now();
    let batch_queries: Vec<_> = queries.iter().take(batch_size).collect();

    for query in &batch_queries {
        let _ = substrate.resonate(query, k);
    }

    let batch_time = batch_start.elapsed();
    let batch_qps = batch_size as f64 / batch_time.as_secs_f64();

    ThroughputReport {
        num_vectors: database.len(),
        num_queries: queries.len(),
        avg_latency_ns,
        p50_latency_ns,
        p95_latency_ns,
        p99_latency_ns,
        min_latency_ns,
        max_latency_ns,
        qps,
        batch_qps,
    }
}

/// Compare throughput at different index sizes
pub fn throughput_scaling_test() {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║                    THROUGHPUT SCALING TEST                     ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║ Index Size    │ Avg Latency   │ P99 Latency   │ QPS           ║");
    println!("╠═══════════════╪═══════════════╪═══════════════╪═══════════════╣");

    for &size in &[1_000, 10_000, 100_000] {
        let database = generate_random_fingerprints(size, 42);
        let queries = generate_random_fingerprints(100, 43);

        let report = measure_throughput(&database, &queries, 10, 10);

        println!("║ {:>13} │ {:>13} │ {:>13} │ {:>12.0} ║",
            format_num(size),
            format_ns(report.avg_latency_ns),
            format_ns(report.p99_latency_ns),
            report.qps
        );
    }

    println!("╚═══════════════╧═══════════════╧═══════════════╧═══════════════╝");
}

fn format_num(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.0}K", n as f64 / 1_000.0)
    } else {
        format!("{}", n)
    }
}

fn format_ns(ns: u64) -> String {
    if ns >= 1_000_000_000 {
        format!("{:.2}s", ns as f64 / 1_000_000_000.0)
    } else if ns >= 1_000_000 {
        format!("{:.2}ms", ns as f64 / 1_000_000.0)
    } else if ns >= 1_000 {
        format!("{:.2}µs", ns as f64 / 1_000.0)
    } else {
        format!("{}ns", ns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_throughput_measurement() {
        let database = generate_random_fingerprints(1000, 42);
        let queries = generate_random_fingerprints(100, 43);

        let report = measure_throughput(&database, &queries, 10, 10);
        report.print();

        assert!(report.qps > 0.0);
        assert!(report.p50_latency_ns <= report.p95_latency_ns);
        assert!(report.p95_latency_ns <= report.p99_latency_ns);
    }
}
