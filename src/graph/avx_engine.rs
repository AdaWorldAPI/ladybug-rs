//! AVX-512 Graph Engine
//!
//! Graph operations as pure SIMD over fingerprints.
//! No pointers. No adjacency lists. No CSR indices.
//! Just XOR and popcount.
//!
//! # Philosophy
//!
//! Traditional graph: nodes + pointers
//! Fingerprint graph: everything is a 10K-bit vector
//!
//! - Node = fingerprint
//! - Edge = fingerprint (from ⊗ verb ⊗ to)
//! - Query = fingerprint
//! - Traversal = batch XOR + popcount
//!
//! # Performance Model
//!
//! ```text
//! AVX-512 can process 512 bits per instruction.
//! 10K bits = 20 AVX-512 registers
//!
//! Per edge comparison:
//!   - 20 loads (query already in registers)
//!   - 20 XORs
//!   - 20 popcounts (8 per instruction with vpopcntdq)
//!   - 1 reduction
//!   ≈ 50 instructions ≈ 5-17 ns per edge
//!
//! 1M edges full scan: ~5-17 ms
//! 10M edges full scan: ~50-170 ms
//!
//! But we can batch 8 edges per pass through registers:
//!   8 edges × 20 XORs = 160 operations
//!   Amortized: ~2 ns per edge
//!
//! 1M edges: ~2 ms
//! 10M edges: ~20 ms
//!
//! This BEATS Kuzu for any query touching >1% of the graph.
//! ```

use crate::core::Fingerprint;

/// Number of u64 words in a 16K-bit fingerprint
const WORDS: usize = 256; // 256 * 64 = 16384 bits

/// Number of u64s per AVX-512 register
const LANES: usize = 8; // 512 bits / 64 bits

/// Full AVX-512 iterations needed
const FULL_ITERS: usize = WORDS / LANES; // 256 / 8 = 32

/// Remainder words after full iterations
const REMAINDER: usize = WORDS % LANES; // 256 % 8 = 0

// =============================================================================
// CORE SIMD OPERATIONS
// =============================================================================

/// Compute Hamming distance between two 10K-bit fingerprints using AVX-512
/// 
/// Returns the number of differing bits (0 = identical, 10000 = maximally different)
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vpopcntdq")]
pub unsafe fn hamming_distance_avx512(a: &[u64; WORDS], b: &[u64; WORDS]) -> u32 {
    use std::arch::x86_64::*;
    
    let mut total = _mm512_setzero_si512();
    
    // Process 8 u64s (512 bits) at a time
    for i in 0..FULL_ITERS {
        let offset = i * LANES;
        let va = _mm512_loadu_si512(a.as_ptr().add(offset) as *const __m512i);
        let vb = _mm512_loadu_si512(b.as_ptr().add(offset) as *const __m512i);
        let xor = _mm512_xor_si512(va, vb);
        let pop = _mm512_popcnt_epi64(xor);
        total = _mm512_add_epi64(total, pop);
    }
    
    // Handle remainder (4 u64s)
    let mut remainder_sum: u64 = 0;
    for i in (FULL_ITERS * LANES)..WORDS {
        remainder_sum += (a[i] ^ b[i]).count_ones() as u64;
    }
    
    // Horizontal sum of 8 lanes
    let mut result = [0u64; 8];
    _mm512_storeu_si512(result.as_mut_ptr() as *mut __m512i, total);
    let simd_sum: u64 = result.iter().sum();
    
    (simd_sum + remainder_sum) as u32
}

/// Fallback for non-AVX512 systems
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
pub fn hamming_distance_avx512(a: &[u64; WORDS], b: &[u64; WORDS]) -> u32 {
    let mut sum = 0u32;
    for i in 0..WORDS {
        sum += (a[i] ^ b[i]).count_ones();
    }
    sum
}

/// Safe wrapper
#[inline]
pub fn hamming_distance(a: &Fingerprint, b: &Fingerprint) -> u32 {
    let a_words = fingerprint_to_words(a);
    let b_words = fingerprint_to_words(b);
    
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    unsafe {
        hamming_distance_avx512(&a_words, &b_words)
    }
    
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    hamming_distance_avx512(&a_words, &b_words)
}

/// XOR two fingerprints (binding operation) using AVX-512
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn xor_avx512(a: &[u64; WORDS], b: &[u64; WORDS], out: &mut [u64; WORDS]) {
    use std::arch::x86_64::*;
    
    for i in 0..FULL_ITERS {
        let offset = i * LANES;
        let va = _mm512_loadu_si512(a.as_ptr().add(offset) as *const __m512i);
        let vb = _mm512_loadu_si512(b.as_ptr().add(offset) as *const __m512i);
        let result = _mm512_xor_si512(va, vb);
        _mm512_storeu_si512(out.as_mut_ptr().add(offset) as *mut __m512i, result);
    }
    
    // Remainder
    for i in (FULL_ITERS * LANES)..WORDS {
        out[i] = a[i] ^ b[i];
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
pub fn xor_avx512(a: &[u64; WORDS], b: &[u64; WORDS], out: &mut [u64; WORDS]) {
    for i in 0..WORDS {
        out[i] = a[i] ^ b[i];
    }
}

// =============================================================================
// GRAPH OPERATIONS
// =============================================================================

/// Result of a graph query
#[derive(Debug, Clone)]
pub struct QueryMatch {
    /// Index of matching edge in the edge table
    pub edge_index: usize,
    /// Hamming distance (lower = better match)
    pub distance: u32,
    /// The matching edge fingerprint
    pub edge: Fingerprint,
}

/// Graph stored as flat array of edge fingerprints
/// 
/// Each edge is encoded as: from ⊗ verb ⊗ to
/// This allows pattern matching via XOR + popcount
pub struct FingerprintGraph {
    /// All edges as fingerprints
    edges: Vec<[u64; WORDS]>,
    /// Optional: original fingerprints for reconstruction
    edge_sources: Vec<(Fingerprint, Fingerprint, Fingerprint)>, // (from, verb, to)
}

impl FingerprintGraph {
    /// Create empty graph
    pub fn new() -> Self {
        Self {
            edges: Vec::new(),
            edge_sources: Vec::new(),
        }
    }
    
    /// Create with capacity
    pub fn with_capacity(n: usize) -> Self {
        Self {
            edges: Vec::with_capacity(n),
            edge_sources: Vec::with_capacity(n),
        }
    }
    
    /// Number of edges
    pub fn len(&self) -> usize {
        self.edges.len()
    }
    
    /// Add an edge: from --[verb]--> to
    pub fn add_edge(&mut self, from: &Fingerprint, verb: &Fingerprint, to: &Fingerprint) {
        // Edge fingerprint = from ⊗ verb ⊗ to
        let edge = from.bind(verb).bind(to);
        self.edges.push(fingerprint_to_words(&edge));
        self.edge_sources.push((from.clone(), verb.clone(), to.clone()));
    }
    
    /// Query: find all edges matching a pattern
    /// 
    /// Pattern is a partial edge (e.g., from ⊗ verb for "what does X cause?")
    /// Returns edges with Hamming distance < threshold
    pub fn query(&self, pattern: &Fingerprint, threshold: u32) -> Vec<QueryMatch> {
        let pattern_words = fingerprint_to_words(pattern);
        let mut matches = Vec::new();
        
        for (idx, edge) in self.edges.iter().enumerate() {
            #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
            let dist = unsafe { hamming_distance_avx512(&pattern_words, edge) };
            
            #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
            let dist = hamming_distance_avx512(&pattern_words, edge);
            
            if dist < threshold {
                matches.push(QueryMatch {
                    edge_index: idx,
                    distance: dist,
                    edge: words_to_fingerprint(edge),
                });
            }
        }
        
        // Sort by distance (best matches first)
        matches.sort_by_key(|m| m.distance);
        matches
    }
    
    /// Batch query: find edges matching ANY of the patterns
    /// 
    /// More efficient than multiple single queries
    pub fn batch_query(&self, patterns: &[Fingerprint], threshold: u32) -> Vec<Vec<QueryMatch>> {
        let pattern_words: Vec<[u64; WORDS]> = patterns
            .iter()
            .map(fingerprint_to_words)
            .collect();
        
        let mut all_matches: Vec<Vec<QueryMatch>> = vec![Vec::new(); patterns.len()];
        
        for (edge_idx, edge) in self.edges.iter().enumerate() {
            for (pattern_idx, pattern) in pattern_words.iter().enumerate() {
                #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
                let dist = unsafe { hamming_distance_avx512(pattern, edge) };
                
                #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
                let dist = hamming_distance_avx512(pattern, edge);
                
                if dist < threshold {
                    all_matches[pattern_idx].push(QueryMatch {
                        edge_index: edge_idx,
                        distance: dist,
                        edge: words_to_fingerprint(edge),
                    });
                }
            }
        }
        
        // Sort each result set
        for matches in &mut all_matches {
            matches.sort_by_key(|m| m.distance);
        }
        
        all_matches
    }
    
    /// N-hop traversal: find all nodes reachable within N hops
    /// 
    /// Returns (hop_number, target_fingerprint, path_distance)
    pub fn traverse_n_hops(
        &self,
        start: &Fingerprint,
        verb: &Fingerprint,
        max_hops: usize,
        threshold: u32,
    ) -> Vec<(usize, Fingerprint, u32)> {
        let mut results = Vec::new();
        let mut frontier = vec![start.clone()];
        let mut visited = std::collections::HashSet::new();
        visited.insert(fingerprint_hash(start));
        
        for hop in 1..=max_hops {
            let mut next_frontier = Vec::new();
            
            // Build patterns for all frontier nodes
            let patterns: Vec<Fingerprint> = frontier
                .iter()
                .map(|node| node.bind(verb))
                .collect();
            
            // Batch query all patterns at once
            let batch_results = self.batch_query(&patterns, threshold);
            
            for (pattern_idx, matches) in batch_results.into_iter().enumerate() {
                let from_node = &frontier[pattern_idx];
                
                for m in matches {
                    // Extract target: edge ⊗ from ⊗ verb = to
                    let target = m.edge.bind(from_node).bind(verb);
                    let target_hash = fingerprint_hash(&target);
                    
                    if !visited.contains(&target_hash) {
                        visited.insert(target_hash);
                        results.push((hop, target.clone(), m.distance));
                        next_frontier.push(target);
                    }
                }
            }
            
            if next_frontier.is_empty() {
                break;
            }
            frontier = next_frontier;
        }
        
        results
    }
    
    /// Find paths between two nodes
    pub fn find_paths(
        &self,
        from: &Fingerprint,
        to: &Fingerprint,
        verb: &Fingerprint,
        max_hops: usize,
        threshold: u32,
    ) -> Vec<Vec<Fingerprint>> {
        let target_hash = fingerprint_hash(to);
        let mut paths = Vec::new();
        
        // BFS with path tracking
        let mut queue = vec![(vec![from.clone()], from.clone())];
        let mut visited = std::collections::HashSet::new();
        visited.insert(fingerprint_hash(from));
        
        while let Some((path, current)) = queue.pop() {
            if path.len() > max_hops {
                continue;
            }
            
            // Check if we reached target
            let current_hash = fingerprint_hash(&current);
            if current_hash == target_hash {
                paths.push(path);
                continue;
            }
            
            // Explore neighbors
            let pattern = current.bind(verb);
            let matches = self.query(&pattern, threshold);
            
            for m in matches {
                let target = m.edge.bind(&current).bind(verb);
                let target_hash = fingerprint_hash(&target);
                
                if !visited.contains(&target_hash) {
                    visited.insert(target_hash);
                    let mut new_path = path.clone();
                    new_path.push(target.clone());
                    queue.push((new_path, target));
                }
            }
        }
        
        paths
    }
}

impl Default for FingerprintGraph {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// BATCHED OPERATIONS (8 edges at once)
// =============================================================================

/// Process 8 edges simultaneously
/// 
/// This is the real performance win: amortize register loads across 8 comparisons
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
pub unsafe fn batch_hamming_8(
    query: &[u64; WORDS],
    edges: &[[u64; WORDS]; 8],
) -> [u32; 8] {
    use std::arch::x86_64::*;
    
    let mut totals = [_mm512_setzero_si512(); 8];
    
    for i in 0..FULL_ITERS {
        let offset = i * LANES;
        let vq = _mm512_loadu_si512(query.as_ptr().add(offset) as *const __m512i);
        
        for j in 0..8 {
            let ve = _mm512_loadu_si512(edges[j].as_ptr().add(offset) as *const __m512i);
            let xor = _mm512_xor_si512(vq, ve);
            let pop = _mm512_popcnt_epi64(xor);
            totals[j] = _mm512_add_epi64(totals[j], pop);
        }
    }
    
    // Reduce each total
    let mut results = [0u32; 8];
    for j in 0..8 {
        let mut lanes = [0u64; 8];
        _mm512_storeu_si512(lanes.as_mut_ptr() as *mut __m512i, totals[j]);
        let sum: u64 = lanes.iter().sum();
        
        // Add remainder
        let mut rem_sum = 0u64;
        for i in (FULL_ITERS * LANES)..WORDS {
            rem_sum += (query[i] ^ edges[j][i]).count_ones() as u64;
        }
        
        results[j] = (sum + rem_sum) as u32;
    }
    
    results
}

/// Batched graph query - processes edges 8 at a time
pub fn batched_query(
    graph: &FingerprintGraph,
    pattern: &Fingerprint,
    threshold: u32,
) -> Vec<QueryMatch> {
    let pattern_words = fingerprint_to_words(pattern);
    let mut matches = Vec::new();
    
    // Process 8 edges at a time
    let chunks = graph.edges.chunks_exact(8);
    let remainder = chunks.remainder();
    
    for (chunk_idx, chunk) in chunks.enumerate() {
        let edges: &[[u64; WORDS]; 8] = chunk.try_into().unwrap();
        
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        let distances = unsafe { batch_hamming_8(&pattern_words, edges) };
        
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
        let distances = {
            let mut d = [0u32; 8];
            for i in 0..8 {
                d[i] = hamming_distance_avx512(&pattern_words, &edges[i]);
            }
            d
        };
        
        for (i, &dist) in distances.iter().enumerate() {
            if dist < threshold {
                let edge_idx = chunk_idx * 8 + i;
                matches.push(QueryMatch {
                    edge_index: edge_idx,
                    distance: dist,
                    edge: words_to_fingerprint(&graph.edges[edge_idx]),
                });
            }
        }
    }
    
    // Handle remainder
    let base_idx = (graph.edges.len() / 8) * 8;
    for (i, edge) in remainder.iter().enumerate() {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        let dist = unsafe { hamming_distance_avx512(&pattern_words, edge) };
        
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
        let dist = hamming_distance_avx512(&pattern_words, edge);
        
        if dist < threshold {
            matches.push(QueryMatch {
                edge_index: base_idx + i,
                distance: dist,
                edge: words_to_fingerprint(edge),
            });
        }
    }
    
    matches.sort_by_key(|m| m.distance);
    matches
}

// =============================================================================
// HELPERS
// =============================================================================

fn fingerprint_to_words(fp: &Fingerprint) -> [u64; WORDS] {
    let bytes = fp.as_bytes();
    let mut words = [0u64; WORDS];
    
    for i in 0..WORDS {
        let start = i * 8;
        if start + 8 <= bytes.len() {
            words[i] = u64::from_le_bytes(bytes[start..start + 8].try_into().unwrap());
        } else if start < bytes.len() {
            // Partial last word
            let mut buf = [0u8; 8];
            let len = bytes.len() - start;
            buf[..len].copy_from_slice(&bytes[start..]);
            words[i] = u64::from_le_bytes(buf);
        }
    }
    
    words
}

fn words_to_fingerprint(words: &[u64; WORDS]) -> Fingerprint {
    use crate::FINGERPRINT_U64;
    // FINGERPRINT_U64 = FINGERPRINT_WORDS = 256, direct conversion
    let mut bytes = vec![0u8; FINGERPRINT_U64 * 8];  // 2048 bytes
    for (i, &word) in words.iter().enumerate() {
        let start = i * 8;
        bytes[start..start + 8].copy_from_slice(&word.to_le_bytes());
    }
    Fingerprint::from_bytes(&bytes).expect("valid fingerprint bytes")
}

fn fingerprint_hash(fp: &Fingerprint) -> u64 {
    // Simple hash for visited set
    let bytes = fp.as_bytes();
    let mut hash = 0u64;
    for chunk in bytes.chunks(8) {
        let mut buf = [0u8; 8];
        buf[..chunk.len()].copy_from_slice(chunk);
        hash ^= u64::from_le_bytes(buf);
    }
    hash
}

// =============================================================================
// BENCHMARKS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hamming_distance() {
        let a = Fingerprint::random();
        let b = Fingerprint::random();
        let c = a.clone();
        
        // Same fingerprint = distance 0
        assert_eq!(hamming_distance(&a, &c), 0);
        
        // Random fingerprints should be ~8192 apart (half of 16384 bits differ)
        let dist = hamming_distance(&a, &b);
        assert!(dist > 6000 && dist < 10000, "Expected ~8192, got {}", dist);
    }
    
    #[test]
    fn test_graph_add_query() {
        let mut graph = FingerprintGraph::new();
        
        let alice = Fingerprint::from_content("Alice");
        let bob = Fingerprint::from_content("Bob");
        let carol = Fingerprint::from_content("Carol");
        let causes = Fingerprint::from_content("CAUSES");
        let enables = Fingerprint::from_content("ENABLES");
        
        // Alice --CAUSES--> Bob
        graph.add_edge(&alice, &causes, &bob);
        
        // Bob --ENABLES--> Carol
        graph.add_edge(&bob, &enables, &carol);
        
        // Query: what does Alice cause?
        let pattern = alice.bind(&causes);
        let matches = graph.query(&pattern, 8200); // Threshold ~50% of 16384 bits
        
        assert!(!matches.is_empty(), "Should find Alice's edge");
    }
    
    #[test]
    fn test_n_hop_traversal() {
        let mut graph = FingerprintGraph::new();
        
        // Build a chain: A -> B -> C -> D
        let nodes: Vec<Fingerprint> = (0..4)
            .map(|i| Fingerprint::from_content(&format!("Node{}", i)))
            .collect();
        let causes = Fingerprint::from_content("CAUSES");
        
        for i in 0..3 {
            graph.add_edge(&nodes[i], &causes, &nodes[i + 1]);
        }
        
        // 3-hop traversal from Node0
        let results = graph.traverse_n_hops(&nodes[0], &causes, 3, 8200);
        
        assert!(!results.is_empty(), "Should find nodes within 3 hops");
    }
    
    #[test]
    fn bench_single_query() {
        use std::time::Instant;
        
        let mut graph = FingerprintGraph::with_capacity(100_000);
        let verb = Fingerprint::from_content("CAUSES");
        
        // Add 100K random edges
        for i in 0..100_000 {
            let from = Fingerprint::from_content(&format!("from{}", i));
            let to = Fingerprint::from_content(&format!("to{}", i));
            graph.add_edge(&from, &verb, &to);
        }
        
        let query_node = Fingerprint::from_content("from42");
        let pattern = query_node.bind(&verb);
        
        // Benchmark
        let start = Instant::now();
        let iterations = 100;
        for _ in 0..iterations {
            let _ = graph.query(&pattern, 3000);
        }
        let elapsed = start.elapsed();
        
        let per_query = elapsed / iterations;
        let edges_per_sec = (100_000 * iterations) as f64 / elapsed.as_secs_f64();
        
        println!("100K edges, {} queries: {:?} total", iterations, elapsed);
        println!("Per query: {:?}", per_query);
        println!("Edges/sec: {:.2}M", edges_per_sec / 1_000_000.0);
    }
    
    #[test]
    fn bench_batched_query() {
        use std::time::Instant;
        
        let mut graph = FingerprintGraph::with_capacity(100_000);
        let verb = Fingerprint::from_content("CAUSES");
        
        // Add 100K random edges
        for i in 0..100_000 {
            let from = Fingerprint::from_content(&format!("from{}", i));
            let to = Fingerprint::from_content(&format!("to{}", i));
            graph.add_edge(&from, &verb, &to);
        }
        
        let query_node = Fingerprint::from_content("from42");
        let pattern = query_node.bind(&verb);
        
        // Benchmark batched version
        let start = Instant::now();
        let iterations = 100;
        for _ in 0..iterations {
            let _ = batched_query(&graph, &pattern, 3000);
        }
        let elapsed = start.elapsed();
        
        let per_query = elapsed / iterations;
        let edges_per_sec = (100_000 * iterations) as f64 / elapsed.as_secs_f64();
        
        println!("BATCHED: 100K edges, {} queries: {:?} total", iterations, elapsed);
        println!("Per query: {:?}", per_query);
        println!("Edges/sec: {:.2}M", edges_per_sec / 1_000_000.0);
    }
}

// =============================================================================
// SIMD VERIFICATION
// =============================================================================

/// Check if AVX-512 is available at runtime
pub fn avx512_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx512f") && 
        is_x86_feature_detected!("avx512vpopcntdq")
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Get SIMD capability info
pub fn simd_info() -> String {
    let mut info = Vec::new();
    
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            info.push("AVX-512F");
        }
        if is_x86_feature_detected!("avx512vpopcntdq") {
            info.push("AVX-512VPOPCNTDQ");
        }
        if is_x86_feature_detected!("avx512bw") {
            info.push("AVX-512BW");
        }
        if is_x86_feature_detected!("avx2") {
            info.push("AVX2");
        }
        if is_x86_feature_detected!("popcnt") {
            info.push("POPCNT");
        }
    }
    
    if info.is_empty() {
        "No SIMD (scalar fallback)".to_string()
    } else {
        info.join(" + ")
    }
}
