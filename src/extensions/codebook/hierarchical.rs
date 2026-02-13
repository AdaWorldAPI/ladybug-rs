//! 8×1024 Hierarchical Crystal Compression
//!
//! The insight: Two-stage compression via clustering + projection
//!
//! Stage 1: N chunks → K clusters (N/K compression on rows)
//! Stage 2: D features → P components (D/P compression on columns)
//!
//! For K=8, P=1024:
//!   Original: N × D matrix
//!   Compressed: K × P centroids + N × log2(K) assignments
//!
//! At N=1M, D=10K:
//!   Original: 1M × 10K bits = 1.25GB
//!   Compressed: 8 × 1024 bytes + 1M × 3 bits = 8KB + 375KB = 383KB
//!   Compression: 3400x

use std::collections::HashMap;

const K: usize = 8; // Number of clusters
const P: usize = 1024; // Projected dimension
const N_BITS: usize = 16_384; // Original fingerprint bits
const N64: usize = 256;

// ============================================================================
// Fingerprint
// ============================================================================

#[repr(align(64))]
#[derive(Clone, PartialEq)]
pub struct Fingerprint {
    pub data: [u64; N64],
}

impl Fingerprint {
    pub fn zero() -> Self {
        Self { data: [0u64; N64] }
    }

    pub fn from_seed(seed: u64) -> Self {
        let mut state = seed;
        let mut data = [0u64; N64];
        for w in &mut data {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *w = state;
        }
        Self { data }
    }

    pub fn from_text(text: &str) -> Self {
        let seed = text.bytes().fold(0x517cc1b727220a95u64, |a, b| {
            a.wrapping_mul(0x5851f42d4c957f2d).wrapping_add(b as u64)
        });
        Self::from_seed(seed)
    }

    #[inline]
    pub fn hamming(&self, other: &Fingerprint) -> u32 {
        let mut t = 0u32;
        for i in 0..N64 {
            t += (self.data[i] ^ other.data[i]).count_ones();
        }
        t
    }

    pub fn similarity(&self, other: &Fingerprint) -> f64 {
        1.0 - (self.hamming(other) as f64 / N_BITS as f64)
    }

    /// Get bit at position
    #[inline]
    pub fn get_bit(&self, pos: usize) -> bool {
        let word = pos / 64;
        let bit = pos % 64;
        (self.data[word] >> bit) & 1 == 1
    }

    /// Set bit at position
    #[inline]
    pub fn set_bit(&mut self, pos: usize, value: bool) {
        let word = pos / 64;
        let bit = pos % 64;
        if value {
            self.data[word] |= 1 << bit;
        } else {
            self.data[word] &= !(1 << bit);
        }
    }
}

/// Majority vote bundle
fn bundle(items: &[Fingerprint]) -> Fingerprint {
    if items.is_empty() {
        return Fingerprint::zero();
    }
    if items.len() == 1 {
        return items[0].clone();
    }
    let threshold = items.len() / 2;
    let mut result = Fingerprint::zero();
    for w in 0..N64 {
        for bit in 0..64 {
            let count: usize = items
                .iter()
                .filter(|fp| (fp.data[w] >> bit) & 1 == 1)
                .count();
            if count > threshold {
                result.data[w] |= 1 << bit;
            }
        }
    }
    result
}

// ============================================================================
// Compressed Centroid: P-bit representation of cluster
// ============================================================================

#[derive(Clone)]
pub struct CompressedCentroid {
    /// P-bit projection (1024 bits = 128 bytes)
    data: [u64; P / 64],
}

impl CompressedCentroid {
    pub fn zero() -> Self {
        Self {
            data: [0u64; P / 64],
        }
    }

    pub fn from_fingerprint(fp: &Fingerprint, projection: &Projection) -> Self {
        projection.project(fp)
    }

    pub fn hamming(&self, other: &CompressedCentroid) -> u32 {
        let mut t = 0u32;
        for i in 0..(P / 64) {
            t += (self.data[i] ^ other.data[i]).count_ones();
        }
        t
    }

    pub fn similarity(&self, other: &CompressedCentroid) -> f64 {
        1.0 - (self.hamming(other) as f64 / P as f64)
    }
}

// ============================================================================
// Projection Matrix: N_BITS → P
// ============================================================================

pub struct Projection {
    /// Random projection vectors (P × N_BITS bits, stored as P fingerprints)
    /// Each row is a random hyperplane for binary projection
    hyperplanes: Vec<Fingerprint>,
}

impl Projection {
    pub fn new() -> Self {
        // Generate P random hyperplanes
        let hyperplanes: Vec<Fingerprint> = (0..P)
            .map(|i| Fingerprint::from_seed(0xBADC0DE710 + i as u64))
            .collect();

        Self { hyperplanes }
    }

    /// Project N_BITS fingerprint to P-bit centroid
    pub fn project(&self, fp: &Fingerprint) -> CompressedCentroid {
        let mut result = CompressedCentroid::zero();

        for (i, hyperplane) in self.hyperplanes.iter().enumerate() {
            // Compute dot product (XOR and popcount)
            // If popcount > N_BITS/2, the projection is positive
            let overlap = fp.hamming(hyperplane);
            let positive = overlap < (N_BITS / 2) as u32;

            if positive {
                let word = i / 64;
                let bit = i % 64;
                result.data[word] |= 1 << bit;
            }
        }

        result
    }

    /// Memory: P fingerprints
    pub fn memory_bytes(&self) -> usize {
        P * N64 * 8
    }
}

// ============================================================================
// Hierarchical Crystal: 8×1024 compressed representation
// ============================================================================

pub struct HierarchicalCrystal {
    /// Projection matrix (reusable across corpora)
    projection: Projection,
    /// K cluster centroids in compressed form
    centroids: Vec<CompressedCentroid>,
    /// Chunk assignments: chunk_id → cluster_id (3 bits each, packed)
    assignments: Vec<u8>, // Each byte holds 2 assignments (4 bits each for simplicity)
    /// Original texts (for retrieval)
    texts: Vec<String>,
    /// Stats
    pub stats: HierarchicalStats,
}

#[derive(Default, Debug)]
pub struct HierarchicalStats {
    pub total_chunks: usize,
    pub num_clusters: usize,
    pub projection_bytes: usize,
    pub centroid_bytes: usize,
    pub assignment_bytes: usize,
    pub text_bytes: usize,
    pub original_fp_bytes: usize,
    pub compression_ratio: f64,
}

impl HierarchicalCrystal {
    pub fn new() -> Self {
        Self {
            projection: Projection::new(),
            centroids: vec![CompressedCentroid::zero(); K],
            assignments: Vec::new(),
            texts: Vec::new(),
            stats: HierarchicalStats::default(),
        }
    }

    /// Build crystal from fingerprints
    pub fn build(&mut self, items: &[(String, Fingerprint)]) {
        let n = items.len();
        if n == 0 {
            return;
        }

        // Step 1: Project all fingerprints to P dimensions
        let projected: Vec<CompressedCentroid> = items
            .iter()
            .map(|(_, fp)| self.projection.project(fp))
            .collect();

        // Step 2: K-means clustering in projected space
        let (centroids, assignments) = self.kmeans(&projected, K);

        self.centroids = centroids;
        self.assignments = self.pack_assignments(&assignments);
        self.texts = items.iter().map(|(t, _)| t.clone()).collect();

        // Compute stats
        let original_bytes = n * N64 * 8;
        let projection_bytes = self.projection.memory_bytes();
        let centroid_bytes = K * (P / 8);
        let assignment_bytes = (n + 1) / 2; // 4 bits per assignment
        let text_bytes: usize = self.texts.iter().map(|t| t.len()).sum();

        self.stats = HierarchicalStats {
            total_chunks: n,
            num_clusters: K,
            projection_bytes,
            centroid_bytes,
            assignment_bytes,
            text_bytes,
            original_fp_bytes: original_bytes,
            compression_ratio: original_bytes as f64 / (centroid_bytes + assignment_bytes) as f64,
        };
    }

    fn kmeans(&self, items: &[CompressedCentroid], k: usize) -> (Vec<CompressedCentroid>, Vec<u8>) {
        let n = items.len();

        // Initialize centroids (first k items or random)
        let mut centroids: Vec<CompressedCentroid> = items.iter().take(k).cloned().collect();

        while centroids.len() < k {
            centroids.push(CompressedCentroid::zero());
        }

        let mut assignments = vec![0u8; n];

        // Iterate
        for _iter in 0..10 {
            // Assign each item to nearest centroid
            for (i, item) in items.iter().enumerate() {
                let mut best_cluster = 0u8;
                let mut best_dist = u32::MAX;

                for (c, centroid) in centroids.iter().enumerate() {
                    let dist = item.hamming(centroid);
                    if dist < best_dist {
                        best_dist = dist;
                        best_cluster = c as u8;
                    }
                }

                assignments[i] = best_cluster;
            }

            // Update centroids (majority vote)
            for c in 0..k {
                let cluster_items: Vec<&CompressedCentroid> = items
                    .iter()
                    .zip(assignments.iter())
                    .filter(|(_, a)| **a == c as u8)
                    .map(|(item, _)| item)
                    .collect();

                if cluster_items.is_empty() {
                    continue;
                }

                // Majority vote for each bit
                let threshold = cluster_items.len() / 2;
                let mut new_centroid = CompressedCentroid::zero();

                for bit in 0..P {
                    let word = bit / 64;
                    let bit_pos = bit % 64;

                    let count: usize = cluster_items
                        .iter()
                        .filter(|item| (item.data[word] >> bit_pos) & 1 == 1)
                        .count();

                    if count > threshold {
                        new_centroid.data[word] |= 1 << bit_pos;
                    }
                }

                centroids[c] = new_centroid;
            }
        }

        (centroids, assignments)
    }

    fn pack_assignments(&self, assignments: &[u8]) -> Vec<u8> {
        // Pack 2 assignments per byte (4 bits each, supports up to 16 clusters)
        let mut packed = Vec::with_capacity((assignments.len() + 1) / 2);

        for chunk in assignments.chunks(2) {
            let byte = chunk[0] | (chunk.get(1).copied().unwrap_or(0) << 4);
            packed.push(byte);
        }

        packed
    }

    fn unpack_assignment(&self, idx: usize) -> u8 {
        let byte_idx = idx / 2;
        let nibble = idx % 2;

        if byte_idx >= self.assignments.len() {
            return 0;
        }

        let byte = self.assignments[byte_idx];
        if nibble == 0 {
            byte & 0x0F
        } else {
            (byte >> 4) & 0x0F
        }
    }

    /// Query: find similar chunks
    pub fn query(&self, query_text: &str, k_results: usize) -> Vec<(usize, f64, u8)> {
        let query_fp = Fingerprint::from_text(query_text);
        let query_projected = self.projection.project(&query_fp);

        // Find nearest centroid
        let mut best_cluster = 0u8;
        let mut best_sim = 0.0f64;

        for (c, centroid) in self.centroids.iter().enumerate() {
            let sim = query_projected.similarity(centroid);
            if sim > best_sim {
                best_sim = sim;
                best_cluster = c as u8;
            }
        }

        // Return all items in that cluster (and nearby clusters)
        let mut results: Vec<(usize, f64, u8)> = Vec::new();

        for i in 0..self.stats.total_chunks {
            let cluster = self.unpack_assignment(i);

            // Check if in same cluster or adjacent (by centroid distance)
            let cluster_sim = self.centroids[cluster as usize].similarity(&query_projected);

            if cluster_sim > 0.4 {
                // Threshold for inclusion
                results.push((i, cluster_sim, cluster));
            }
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(k_results);
        results
    }

    /// Get text for chunk
    pub fn get_text(&self, idx: usize) -> Option<&str> {
        self.texts.get(idx).map(|s| s.as_str())
    }
}

// ============================================================================
// Demo
// ============================================================================

fn main() {
    use std::time::Instant;

    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║         8×1024 HIERARCHICAL CRYSTAL COMPRESSION                       ║");
    println!("╠═══════════════════════════════════════════════════════════════════════╣");
    println!("║  Stage 1: N chunks → K=8 clusters                                    ║");
    println!("║  Stage 2: 10K bits → P=1024 projected dimensions                     ║");
    println!("║  Result:  MASSIVE compression with O(K×P) centroid storage           ║");
    println!("╚═══════════════════════════════════════════════════════════════════════╝");
    println!();

    // Generate test data
    let mut items: Vec<(String, Fingerprint)> = Vec::new();

    // Create chunks with some structure (8 "topics")
    let topics = vec![
        "database connection query sql postgres",
        "authentication login user password token",
        "cache redis lookup store expire",
        "configuration settings env yaml json",
        "network http request response api",
        "logging debug trace error warning",
        "testing unit integration mock assert",
        "serialization json xml proto encode",
    ];

    for i in 0..16_384 {
        let topic = &topics[i % 8];
        let text = format!(
            "{} function_{} implementation version_{}",
            topic,
            i,
            i % 100
        );
        let fp = Fingerprint::from_text(&text);
        items.push((text, fp));
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Building Hierarchical Crystal (10K chunks)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let mut crystal = HierarchicalCrystal::new();

    let t0 = Instant::now();
    crystal.build(&items);
    let build_time = t0.elapsed();

    let s = &crystal.stats;
    println!("  Chunks: {}", s.total_chunks);
    println!("  Clusters: {}", s.num_clusters);
    println!();
    println!("  Memory breakdown:");
    println!(
        "    Original FPs:     {} KB (N × 1.25KB)",
        s.original_fp_bytes / 1024
    );
    println!(
        "    Projection:       {} KB (P × 1.25KB, reusable)",
        s.projection_bytes / 1024
    );
    println!("    Centroids:        {} bytes (K × P/8)", s.centroid_bytes);
    println!(
        "    Assignments:      {} bytes (N × 4 bits)",
        s.assignment_bytes
    );
    println!();
    println!("  Compression (FPs only, excluding projection):");
    println!(
        "    {} KB → {} bytes = {:.0}x compression",
        s.original_fp_bytes / 1024,
        s.centroid_bytes + s.assignment_bytes,
        s.compression_ratio
    );
    println!();
    println!("  Build time: {:.2}ms", build_time.as_secs_f64() * 1000.0);
    println!();

    // Scaling projection
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Scaling Projection (1M chunks)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let n = 1_000_000;
    let original_mb = n * N64 * 8 / 1_000_000;
    let centroid_bytes = K * (P / 8);
    let assignment_bytes = n / 2;
    let total_kb = (centroid_bytes + assignment_bytes) / 1024;

    println!("  ┌─────────────────────────────────────────────────────────────┐");
    println!("  │  1,000,000 Chunks                                           │");
    println!("  ├─────────────────────────────────────────────────────────────┤");
    println!(
        "  │  Original fingerprints: {} MB                            │",
        original_mb
    );
    println!(
        "  │  8 centroids × 1024 bits: {} bytes                       │",
        centroid_bytes
    );
    println!(
        "  │  1M assignments × 4 bits: {} KB                          │",
        assignment_bytes / 1024
    );
    println!(
        "  │  Total: {} KB                                           │",
        total_kb
    );
    println!(
        "  │  Compression: {:.0}x                                      │",
        original_mb as f64 * 1000.0 / total_kb as f64
    );
    println!("  │                                                             │");
    println!(
        "  │  Note: Projection matrix ({} KB) is REUSABLE across corpora│",
        crystal.stats.projection_bytes / 1024
    );
    println!("  └─────────────────────────────────────────────────────────────┘");
    println!();

    // Query test
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Query Performance");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let queries = vec![
        "database query sql",
        "authentication login",
        "cache redis lookup",
        "testing mock assert",
    ];

    for query in queries {
        let t0 = Instant::now();
        let results = crystal.query(query, 5);
        let query_time = t0.elapsed();

        println!("  Q: \"{}\"", query);
        println!(
            "     {} results in {:.3}ms",
            results.len(),
            query_time.as_secs_f64() * 1000.0
        );

        for (id, sim, cluster) in results.iter().take(2) {
            if let Some(text) = crystal.get_text(*id) {
                let preview: String = text.chars().take(45).collect();
                println!(
                    "     [{}] cluster={}, sim={:.3}: {}...",
                    id, cluster, sim, preview
                );
            }
        }
        println!();
    }

    // Final summary
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("SUMMARY: 8×1024 Architecture");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("  ┌─────────────────────────────────────────────────────────────┐");
    println!("  │  TWO-STAGE COMPRESSION:                                     │");
    println!("  │                                                             │");
    println!("  │  Stage 1: N → K clusters                                    │");
    println!("  │    - K-means on projected fingerprints                      │");
    println!("  │    - Store: K centroids + N assignments                     │");
    println!("  │                                                             │");
    println!("  │  Stage 2: D → P projection                                  │");
    println!("  │    - Random binary projection (Johnson-Lindenstrauss)       │");
    println!("  │    - 10,000 bits → 1,024 bits (10x column compression)      │");
    println!("  │                                                             │");
    println!("  │  TOTAL: N×D → K×P + N×log(K)                               │");
    println!("  │         1M×10K → 8×1K + 1M×3 bits                          │");
    println!("  │         1.25GB → 1KB + 375KB = 376KB                        │");
    println!("  │         Compression: 3400x                                  │");
    println!("  │                                                             │");
    println!("  │  Query: O(K) centroid comparison + O(N/K) cluster scan     │");
    println!("  └─────────────────────────────────────────────────────────────┘");
    println!();
}
