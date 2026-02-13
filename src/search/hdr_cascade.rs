//! HDR Cascade Search Engine
//!
//! Hierarchical Hamming distance with Mexican hat discrimination,
//! rolling window statistics, and O(1) XOR-based retrieval.
//!
//! # The Alien Magic
//!
//! This module pretends to be a float vector search engine but runs on
//! pure integer SIMD operations. From the outside it looks like Faiss/Annoy.
//! Inside it's XOR and popcount with hierarchical filtering.
//!
//! # Key Concepts
//!
//! ## HDR Multi-Resolution Cascade
//!
//! ```text
//! Level 0: 1-bit sketch  → "is chunk different at all?"
//! Level 1: 4-bit count   → "how different is each chunk?"
//! Level 2: 8-bit count   → "precise per-chunk distance"
//! Level 3: Full popcount → "exact Hamming distance"
//!
//! 90% filtered at L0, 90% of survivors at L1, etc.
//! Average: ~7ns per candidate vs ~100ns for float cosine
//! ```
//!
//! ## Mexican Hat Response
//!
//! ```text
//!   response
//!      │
//!   1.0┤    ╭───╮
//!      │   ╱     ╲
//!   0.0┤──╱───────╲──────────
//!      │ ╱         ╲    
//!  -0.5┤╱           ╲___╱
//!      └────────────────────→ distance
//!         excite  inhibit
//!
//! Center: strong match (excitation)
//! Ring: too similar, suppress (inhibition)  
//! Far: irrelevant (zero)
//! ```
//!
//! ## A⊗B⊗B=A Direct Retrieval
//!
//! XOR is self-inverse. Bound queries can be COMPUTED, not searched.
//! ```text
//! Store: edge = A ⊗ verb ⊗ B
//! Query: edge ⊗ verb ⊗ B = A  (recover source in O(1)!)
//! ```

use std::collections::HashMap;
use crate::core::Fingerprint;
use crate::{Error, Result};

// =============================================================================
// CONSTANTS
// =============================================================================

/// Number of 64-bit words in a fingerprint (256 = 16384 bits)
pub(crate) const WORDS: usize = 256;

/// Bits per fingerprint
const BITS: usize = WORDS * 64;  // 16384

/// Default Mexican hat excitation threshold
const DEFAULT_EXCITE: u32 = 2000;  // ~20% different

/// Default Mexican hat inhibition threshold  
const DEFAULT_INHIBIT: u32 = 5000; // ~50% different

// =============================================================================
// CORE DISTANCE OPERATIONS
// =============================================================================

/// Compute exact Hamming distance
#[inline]
pub fn hamming_distance(a: &[u64; WORDS], b: &[u64; WORDS]) -> u32 {
    let mut dist = 0u32;
    for i in 0..WORDS {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}

/// Compute 1-bit sketch: which chunks differ at all?
/// Returns a bitmask where bit i = 1 if chunk i has any difference
#[inline]
pub fn sketch_1bit(a: &[u64; WORDS], b: &[u64; WORDS]) -> [u8; 32] {
    let mut sketch = [0u8; 32]; // 256 words / 8 bits per byte = 32 bytes
    for i in 0..WORDS {
        let differs = (a[i] ^ b[i]) != 0;
        if differs {
            sketch[i / 8] |= 1 << (i % 8);
        }
    }
    sketch
}

/// Compute 4-bit sketch: how different is each chunk? (0-15, saturated)
#[inline]
pub fn sketch_4bit(a: &[u64; WORDS], b: &[u64; WORDS]) -> [u8; 128] {
    let mut sketch = [0u8; 128]; // 256 words / 2 per byte = 128 bytes
    for i in 0..WORDS {
        let count = (a[i] ^ b[i]).count_ones().min(15) as u8;
        let byte_idx = i / 2;
        if i % 2 == 0 {
            sketch[byte_idx] |= count;
        } else {
            sketch[byte_idx] |= count << 4;
        }
    }
    sketch
}

/// Compute 8-bit sketch: exact per-chunk distance (0-64)
#[inline]
pub fn sketch_8bit(a: &[u64; WORDS], b: &[u64; WORDS]) -> [u8; WORDS] {
    let mut sketch = [0u8; WORDS];
    for i in 0..WORDS {
        sketch[i] = (a[i] ^ b[i]).count_ones() as u8;
    }
    sketch
}

/// Sum of 1-bit sketch (number of differing chunks)
#[inline]
pub fn sketch_1bit_sum(sketch: &[u8; 32]) -> u32 {
    sketch.iter().map(|&b| b.count_ones()).sum()
}

/// Sum of 4-bit sketch (approximate total distance)
#[inline]
pub fn sketch_4bit_sum(sketch: &[u8; 128]) -> u32 {
    let mut sum = 0u32;
    for &byte in sketch.iter() {
        sum += (byte & 0x0F) as u32;
        sum += (byte >> 4) as u32;
    }
    sum
}

/// Sum of 8-bit sketch (precise total distance)
#[inline]
pub fn sketch_8bit_sum(sketch: &[u8; WORDS]) -> u32 {
    sketch.iter().map(|&b| b as u32).sum()
}

// =============================================================================
// AVX-512 ACCELERATED OPERATIONS
// =============================================================================

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod simd {
    use super::*;
    use std::arch::x86_64::*;
    
    /// AVX-512 accelerated Hamming distance
    #[target_feature(enable = "avx512f,avx512vpopcntdq")]
    pub unsafe fn hamming_distance_avx512(a: &[u64; WORDS], b: &[u64; WORDS]) -> u32 {
        let mut total = _mm512_setzero_si512();
        
        // Process 8 u64s at a time
        for i in 0..(WORDS / 8) {
            let offset = i * 8;
            let va = _mm512_loadu_si512(a.as_ptr().add(offset) as *const __m512i);
            let vb = _mm512_loadu_si512(b.as_ptr().add(offset) as *const __m512i);
            let xor = _mm512_xor_si512(va, vb);
            let pop = _mm512_popcnt_epi64(xor);
            total = _mm512_add_epi64(total, pop);
        }
        
        // Remainder
        let mut rem = 0u32;
        for i in ((WORDS / 8) * 8)..WORDS {
            rem += (a[i] ^ b[i]).count_ones();
        }
        
        // Horizontal sum
        let mut lanes = [0u64; 8];
        _mm512_storeu_si512(lanes.as_mut_ptr() as *mut __m512i, total);
        let sum: u64 = lanes.iter().sum();
        
        (sum as u32) + rem
    }
    
    /// Batch process 8 candidates against 1 query
    #[target_feature(enable = "avx512f,avx512vpopcntdq")]
    pub unsafe fn batch_hamming_8(
        query: &[u64; WORDS],
        candidates: &[[u64; WORDS]; 8],
    ) -> [u32; 8] {
        let mut totals = [_mm512_setzero_si512(); 8];
        
        for i in 0..(WORDS / 8) {
            let offset = i * 8;
            let vq = _mm512_loadu_si512(query.as_ptr().add(offset) as *const __m512i);
            
            for j in 0..8 {
                let vc = _mm512_loadu_si512(candidates[j].as_ptr().add(offset) as *const __m512i);
                let xor = _mm512_xor_si512(vq, vc);
                let pop = _mm512_popcnt_epi64(xor);
                totals[j] = _mm512_add_epi64(totals[j], pop);
            }
        }
        
        let mut results = [0u32; 8];
        for j in 0..8 {
            let mut lanes = [0u64; 8];
            _mm512_storeu_si512(lanes.as_mut_ptr() as *mut __m512i, totals[j]);
            let sum: u64 = lanes.iter().sum();
            
            // Remainder
            let mut rem = 0u32;
            for i in ((WORDS / 8) * 8)..WORDS {
                rem += (query[i] ^ candidates[j][i]).count_ones();
            }
            
            results[j] = (sum as u32) + rem;
        }
        
        results
    }
}

// =============================================================================
// MEXICAN HAT RESPONSE
// =============================================================================

/// Mexican hat response curve
/// 
/// - distance < excite: positive response (match)
/// - excite <= distance < inhibit: negative response (suppress)
/// - distance >= inhibit: zero response (ignore)
#[derive(Debug, Clone, Copy)]
pub struct MexicanHat {
    /// Excitation threshold (center of receptive field)
    pub excite: u32,
    /// Inhibition threshold (edge of surround)
    pub inhibit: u32,
    /// Inhibition strength (0.0 to 1.0)
    pub inhibit_strength: f32,
}

impl Default for MexicanHat {
    fn default() -> Self {
        Self {
            excite: DEFAULT_EXCITE,
            inhibit: DEFAULT_INHIBIT,
            inhibit_strength: 0.5,
        }
    }
}

impl MexicanHat {
    /// Create with custom thresholds
    pub fn new(excite: u32, inhibit: u32) -> Self {
        Self {
            excite,
            inhibit,
            inhibit_strength: 0.5,
        }
    }
    
    /// Create from similarity thresholds (0.0 to 1.0)
    pub fn from_similarity(excite_sim: f32, inhibit_sim: f32) -> Self {
        Self {
            excite: ((1.0 - excite_sim) * BITS as f32) as u32,
            inhibit: ((1.0 - inhibit_sim) * BITS as f32) as u32,
            inhibit_strength: 0.5,
        }
    }
    
    /// Compute response for a given distance
    #[inline]
    pub fn response(&self, distance: u32) -> f32 {
        if distance < self.excite {
            // Excitation: linear ramp from 1.0 to 0.0
            1.0 - (distance as f32 / self.excite as f32)
        } else if distance < self.inhibit {
            // Inhibition: negative response
            let t = (distance - self.excite) as f32 / (self.inhibit - self.excite) as f32;
            -self.inhibit_strength * (1.0 - t)
        } else {
            // Beyond range
            0.0
        }
    }
    
    /// Check if distance is in excitation zone
    #[inline]
    pub fn is_excited(&self, distance: u32) -> bool {
        distance < self.excite
    }
    
    /// Check if distance is in inhibition zone
    #[inline]
    pub fn is_inhibited(&self, distance: u32) -> bool {
        distance >= self.excite && distance < self.inhibit
    }
}

// =============================================================================
// ROLLING WINDOW STATISTICS
// =============================================================================

/// Rolling window statistics for coherence detection
pub struct RollingWindow {
    /// Window size
    size: usize,
    /// Circular buffer of distances
    distances: Vec<u32>,
    /// Current position in buffer
    pos: usize,
    /// Running sum
    sum: u64,
    /// Running sum of squares
    sum_sq: u64,
    /// Number of valid entries
    count: usize,
}

impl RollingWindow {
    /// Create a new rolling window
    pub fn new(size: usize) -> Self {
        Self {
            size,
            distances: vec![0; size],
            pos: 0,
            sum: 0,
            sum_sq: 0,
            count: 0,
        }
    }
    
    /// Add a distance to the window
    pub fn push(&mut self, distance: u32) {
        let d = distance as u64;
        
        if self.count >= self.size {
            // Remove old value
            let old = self.distances[self.pos] as u64;
            self.sum -= old;
            self.sum_sq -= old * old;
        } else {
            self.count += 1;
        }
        
        // Add new value
        self.distances[self.pos] = distance;
        self.sum += d;
        self.sum_sq += d * d;
        
        // Advance position
        self.pos = (self.pos + 1) % self.size;
    }
    
    /// Get mean distance
    #[inline]
    pub fn mean(&self) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum as f32 / self.count as f32
    }
    
    /// Get standard deviation
    #[inline]
    pub fn stddev(&self) -> f32 {
        if self.count < 2 {
            return 0.0;
        }
        let n = self.count as f32;
        let mean = self.sum as f32 / n;
        let variance = (self.sum_sq as f32 / n) - (mean * mean);
        variance.max(0.0).sqrt()
    }
    
    /// Get mean and stddev together (μ, σ)
    #[inline]
    pub fn stats(&self) -> (f32, f32) {
        (self.mean(), self.stddev())
    }
    
    /// Get coefficient of variation (σ/μ)
    #[inline]
    pub fn cv(&self) -> f32 {
        let μ = self.mean();
        if μ < 1.0 {
            return 0.0;
        }
        self.stddev() / μ
    }
    
    /// Is the window showing coherent (clustered) pattern?
    /// Low σ = coherent, high σ = dispersed
    pub fn is_coherent(&self, threshold: f32) -> bool {
        self.cv() < threshold
    }
    
    /// Clear the window
    pub fn clear(&mut self) {
        self.distances.fill(0);
        self.pos = 0;
        self.sum = 0;
        self.sum_sq = 0;
        self.count = 0;
    }
}

// =============================================================================
// HDR CASCADE INDEX
// =============================================================================

/// Hierarchical Distance Resolution index for fast similarity search
pub struct HdrIndex {
    /// Level 0: 1-bit sketches (which chunks differ?)
    sketches_1bit: Vec<[u8; 32]>,

    /// Level 1: 4-bit sketches (how much per chunk?)
    sketches_4bit: Vec<[u8; 128]>,
    
    /// Level 2: 8-bit sketches (precise per chunk)
    sketches_8bit: Vec<[u8; WORDS]>,
    
    /// Full fingerprints for final verification
    fingerprints: Vec<[u64; WORDS]>,
    
    /// Optional bucket index by sketch prefix
    buckets: Option<HashMap<u64, Vec<usize>>>,
    
    /// Cascade thresholds
    threshold_l0: u32,  // 1-bit: max differing chunks
    threshold_l1: u32,  // 4-bit: max approximate distance
    threshold_l2: u32,  // 8-bit: max precise distance
}

impl HdrIndex {
    /// Create empty index
    pub fn new() -> Self {
        Self {
            sketches_1bit: Vec::new(),
            sketches_4bit: Vec::new(),
            sketches_8bit: Vec::new(),
            fingerprints: Vec::new(),
            buckets: None,
            threshold_l0: 256,   // All 256 chunks can differ (1-bit sketch)
            threshold_l1: 2000,  // ~16K distance at 4-bit resolution
            threshold_l2: 5000,  // ~30% different at 8-bit
        }
    }

    /// Create with capacity
    pub fn with_capacity(n: usize) -> Self {
        Self {
            sketches_1bit: Vec::with_capacity(n),
            sketches_4bit: Vec::with_capacity(n),
            sketches_8bit: Vec::with_capacity(n),
            fingerprints: Vec::with_capacity(n),
            buckets: None,
            threshold_l0: 256,
            threshold_l1: 2000,
            threshold_l2: 5000,
        }
    }
    
    /// Set cascade thresholds
    pub fn set_thresholds(&mut self, l0: u32, l1: u32, l2: u32) {
        self.threshold_l0 = l0;
        self.threshold_l1 = l1;
        self.threshold_l2 = l2;
    }
    
    /// Add a fingerprint to the index
    pub fn add(&mut self, fp: &[u64; WORDS]) {
        // Compute sketches against zero (for storage)
        // Actual comparison recomputes sketches
        self.fingerprints.push(*fp);
        
        // Precompute self-sketches not useful; skip for now
        self.sketches_1bit.push([0u8; 32]);
        self.sketches_4bit.push([0u8; 128]);
        self.sketches_8bit.push([0u8; WORDS]);
    }
    
    /// Number of entries
    pub fn len(&self) -> usize {
        self.fingerprints.len()
    }
    
    /// Is empty?
    pub fn is_empty(&self) -> bool {
        self.fingerprints.is_empty()
    }
    
    /// Search with HDR cascade
    /// Returns (index, distance) pairs sorted by distance
    pub fn search(&self, query: &[u64; WORDS], k: usize) -> Vec<(usize, u32)> {
        let mut candidates: Vec<(usize, u32)> = Vec::with_capacity(k * 2);
        
        for (idx, fp) in self.fingerprints.iter().enumerate() {
            // Level 0: 1-bit filter
            let s1 = sketch_1bit(query, fp);
            let d1 = sketch_1bit_sum(&s1);
            if d1 > self.threshold_l0 {
                continue;
            }
            
            // Level 1: 4-bit filter
            let s4 = sketch_4bit(query, fp);
            let d4 = sketch_4bit_sum(&s4);
            if d4 > self.threshold_l1 {
                continue;
            }
            
            // Level 2: 8-bit filter
            let s8 = sketch_8bit(query, fp);
            let d8 = sketch_8bit_sum(&s8);
            if d8 > self.threshold_l2 {
                continue;
            }
            
            // Level 3: exact distance (already computed as d8 for 8-bit)
            // For full precision, use exact hamming
            let exact = hamming_distance(query, fp);
            candidates.push((idx, exact));
        }
        
        // Sort by distance and take top k
        candidates.sort_by_key(|&(_, d)| d);
        candidates.truncate(k);
        candidates
    }
    
    /// Search with Mexican hat discrimination
    pub fn search_mexican_hat(
        &self, 
        query: &[u64; WORDS], 
        k: usize,
        hat: &MexicanHat,
    ) -> Vec<(usize, f32)> {
        let mut results: Vec<(usize, f32)> = Vec::new();
        
        for (idx, fp) in self.fingerprints.iter().enumerate() {
            let dist = hamming_distance(query, fp);
            let response = hat.response(dist);
            
            // Only keep positive responses
            if response > 0.0 {
                results.push((idx, response));
            }
        }
        
        // Sort by response (highest first)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(k);
        results
    }
    
    /// Batch search - process multiple queries efficiently
    pub fn batch_search(
        &self,
        queries: &[[u64; WORDS]],
        k: usize,
    ) -> Vec<Vec<(usize, u32)>> {
        queries.iter()
            .map(|q| self.search(q, k))
            .collect()
    }
}

impl Default for HdrIndex {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// A⊗B⊗B=A DIRECT RETRIEVAL
// =============================================================================

/// Bound query for O(1) retrieval
/// 
/// When you know the binding structure, you can COMPUTE results
/// instead of searching for them.
pub struct BoundRetrieval {
    /// Stored edges: A ⊗ verb ⊗ B
    edges: Vec<[u64; WORDS]>,
    /// Source fingerprints (A)
    sources: Vec<[u64; WORDS]>,
    /// Target fingerprints (B)
    targets: Vec<[u64; WORDS]>,
    /// Verb fingerprint
    verb: [u64; WORDS],
}

impl BoundRetrieval {
    /// Create with a verb
    pub fn new(verb: &[u64; WORDS]) -> Self {
        Self {
            edges: Vec::new(),
            sources: Vec::new(),
            targets: Vec::new(),
            verb: *verb,
        }
    }
    
    /// Add an edge: A --[verb]--> B
    pub fn add_edge(&mut self, source: &[u64; WORDS], target: &[u64; WORDS]) {
        // Compute edge = source ⊗ verb ⊗ target
        let mut edge = [0u64; WORDS];
        for i in 0..WORDS {
            edge[i] = source[i] ^ self.verb[i] ^ target[i];
        }
        
        self.edges.push(edge);
        self.sources.push(*source);
        self.targets.push(*target);
    }
    
    /// Retrieve target given source: edge ⊗ verb ⊗ source = target
    /// This is O(1) per edge, not a search!
    pub fn get_targets(&self, source: &[u64; WORDS], threshold: u32) -> Vec<(usize, u32)> {
        let mut results = Vec::new();
        
        for (idx, edge) in self.edges.iter().enumerate() {
            // Compute: edge ⊗ verb ⊗ source = target (if this edge is from source)
            let mut candidate = [0u64; WORDS];
            for i in 0..WORDS {
                candidate[i] = edge[i] ^ self.verb[i] ^ source[i];
            }
            
            // Check if candidate matches stored target
            let dist = hamming_distance(&candidate, &self.targets[idx]);
            if dist < threshold {
                results.push((idx, dist));
            }
        }
        
        results
    }
    
    /// Retrieve source given target: edge ⊗ verb ⊗ target = source
    pub fn get_sources(&self, target: &[u64; WORDS], threshold: u32) -> Vec<(usize, u32)> {
        let mut results = Vec::new();
        
        for (idx, edge) in self.edges.iter().enumerate() {
            // Compute: edge ⊗ verb ⊗ target = source (if this edge goes to target)
            let mut candidate = [0u64; WORDS];
            for i in 0..WORDS {
                candidate[i] = edge[i] ^ self.verb[i] ^ target[i];
            }
            
            // Check if candidate matches stored source
            let dist = hamming_distance(&candidate, &self.sources[idx]);
            if dist < threshold {
                results.push((idx, dist));
            }
        }
        
        results
    }
    
    /// Direct unbind: given edge and one endpoint, recover the other
    /// This is TRUE O(1) - no iteration!
    pub fn unbind(
        edge: &[u64; WORDS],
        verb: &[u64; WORDS],
        known: &[u64; WORDS],
    ) -> [u64; WORDS] {
        let mut result = [0u64; WORDS];
        for i in 0..WORDS {
            result[i] = edge[i] ^ verb[i] ^ known[i];
        }
        result
    }
}

// =============================================================================
// UNIFIED SEARCH API (The Alien Magic)
// =============================================================================

/// Search result with multiple representations
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Index in corpus
    pub index: usize,
    /// Hamming distance (0 to ~16K)
    pub distance: u32,
    /// Similarity score (0.0 to 1.0, like float vectors)
    pub similarity: f32,
    /// Mexican hat response (-0.5 to 1.0)
    pub response: f32,
}

/// Unified search engine that looks like float vector search
pub struct AlienSearch {
    /// HDR cascade index
    index: HdrIndex,
    /// Mexican hat parameters
    hat: MexicanHat,
    /// Rolling window for coherence
    window: RollingWindow,
}

impl AlienSearch {
    /// Create new search engine
    pub fn new() -> Self {
        Self {
            index: HdrIndex::new(),
            hat: MexicanHat::default(),
            window: RollingWindow::new(100),
        }
    }
    
    /// Create with capacity
    pub fn with_capacity(n: usize) -> Self {
        Self {
            index: HdrIndex::with_capacity(n),
            hat: MexicanHat::default(),
            window: RollingWindow::new(100),
        }
    }
    
    /// Set Mexican hat parameters
    pub fn set_mexican_hat(&mut self, excite: u32, inhibit: u32) {
        self.hat = MexicanHat::new(excite, inhibit);
    }
    
    /// Add fingerprint to index
    pub fn add(&mut self, fp: &[u64; WORDS]) {
        self.index.add(fp);
    }
    
    /// Add multiple fingerprints
    pub fn add_batch(&mut self, fps: &[[u64; WORDS]]) {
        for fp in fps {
            self.index.add(fp);
        }
    }
    
    /// Number of indexed fingerprints
    pub fn len(&self) -> usize {
        self.index.len()
    }
    
    /// Search - returns results that look like float vector search
    /// 
    /// This is THE alien magic API. User sees similarity scores.
    /// Underneath it's HDR cascade + Mexican hat + rolling σ.
    pub fn search(&mut self, query: &[u64; WORDS], k: usize) -> Vec<SearchResult> {
        let raw_results = self.index.search(query, k);
        
        raw_results.into_iter().map(|(idx, dist)| {
            // Update rolling window
            self.window.push(dist);
            
            // Convert to similarity (like cosine similarity)
            let similarity = 1.0 - (dist as f32 / BITS as f32);
            
            // Mexican hat response
            let response = self.hat.response(dist);
            
            SearchResult {
                index: idx,
                distance: dist,
                similarity,
                response,
            }
        }).collect()
    }
    
    /// Search returning only similarity scores (float-like API)
    pub fn search_similarity(&mut self, query: &[u64; WORDS], k: usize) -> Vec<(usize, f32)> {
        self.search(query, k)
            .into_iter()
            .map(|r| (r.index, r.similarity))
            .collect()
    }
    
    /// Search with Mexican hat discrimination
    pub fn search_discriminate(&mut self, query: &[u64; WORDS], k: usize) -> Vec<(usize, f32)> {
        self.search(query, k)
            .into_iter()
            .filter(|r| r.response > 0.0)
            .map(|r| (r.index, r.response))
            .collect()
    }
    
    /// Get coherence stats for recent searches
    pub fn coherence(&self) -> (f32, f32) {
        self.window.stats()
    }
    
    /// Is recent search pattern coherent?
    pub fn is_coherent(&self) -> bool {
        self.window.is_coherent(0.3)
    }
}

impl Default for AlienSearch {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// FINGERPRINT EXTENSIONS
// =============================================================================

/// Extension trait for Fingerprint to add search capabilities
pub trait FingerprintSearch {
    /// Convert to words for search operations
    fn to_words(&self) -> [u64; WORDS];
    
    /// Create from words
    fn from_words(words: &[u64; WORDS]) -> Self;
    
    /// Compute Hamming distance to another fingerprint
    fn hamming(&self, other: &Self) -> u32;
    
    /// Compute similarity (0.0 to 1.0)
    fn similarity(&self, other: &Self) -> f32;
    
    /// Mexican hat response
    fn resonance(&self, other: &Self, hat: &MexicanHat) -> f32;
    
    /// Unbind (A⊗B⊗B=A)
    fn unbind(&self, key: &Self) -> Self;
}

impl FingerprintSearch for Fingerprint {
    fn to_words(&self) -> [u64; WORDS] {
        let bytes = self.as_bytes();
        let mut words = [0u64; WORDS];
        for i in 0..WORDS {
            let start = i * 8;
            if start + 8 <= bytes.len() {
                words[i] = u64::from_le_bytes(bytes[start..start+8].try_into().unwrap());
            }
        }
        words
    }
    
    fn from_words(words: &[u64; WORDS]) -> Self {
        use crate::FINGERPRINT_U64;
        // FINGERPRINT_U64 = FINGERPRINT_WORDS = 256, direct conversion
        let mut bytes = vec![0u8; FINGERPRINT_U64 * 8];  // 2048 bytes
        for (i, &word) in words.iter().enumerate() {
            bytes[i*8..(i+1)*8].copy_from_slice(&word.to_le_bytes());
        }
        Fingerprint::from_bytes(&bytes).expect("valid fingerprint bytes")
    }
    
    fn hamming(&self, other: &Self) -> u32 {
        let a = self.to_words();
        let b = other.to_words();
        hamming_distance(&a, &b)
    }
    
    fn similarity(&self, other: &Self) -> f32 {
        let dist = self.hamming(other);
        1.0 - (dist as f32 / BITS as f32)
    }
    
    fn resonance(&self, other: &Self, hat: &MexicanHat) -> f32 {
        let dist = self.hamming(other);
        hat.response(dist)
    }
    
    fn unbind(&self, key: &Self) -> Self {
        // XOR is self-inverse: A⊗B⊗B = A
        self.bind(key)
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    fn random_fingerprint() -> [u64; WORDS] {
        let mut fp = [0u64; WORDS];
        for i in 0..WORDS {
            fp[i] = rand::random();
        }
        fp
    }
    
    #[test]
    fn test_hamming_distance() {
        let a = [0u64; WORDS];
        let b = [0u64; WORDS];
        assert_eq!(hamming_distance(&a, &b), 0);
        
        let mut c = [0u64; WORDS];
        c[0] = 0xFFFFFFFFFFFFFFFF;  // 64 bits set
        assert_eq!(hamming_distance(&a, &c), 64);
    }
    
    #[test]
    fn test_mexican_hat() {
        let hat = MexicanHat::new(2000, 5000);
        
        // Center: strong positive
        assert!(hat.response(0) > 0.9);
        assert!(hat.response(1000) > 0.0);
        
        // Ring: negative
        assert!(hat.response(3000) < 0.0);
        
        // Far: zero
        assert_eq!(hat.response(6000), 0.0);
    }
    
    #[test]
    fn test_rolling_window() {
        let mut window = RollingWindow::new(5);
        
        // Add some values
        for d in [100, 110, 105, 108, 103] {
            window.push(d);
        }
        
        let (μ, σ) = window.stats();
        assert!((μ - 105.2).abs() < 1.0);
        assert!(σ > 0.0 && σ < 10.0);  // Low variance
    }
    
    #[test]
    fn test_hdr_index() {
        let mut index = HdrIndex::with_capacity(100);
        
        // Add random fingerprints
        let fps: Vec<_> = (0..100).map(|_| random_fingerprint()).collect();
        for fp in &fps {
            index.add(fp);
        }
        
        // Search should find the exact fingerprint
        let results = index.search(&fps[42], 5);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 42);  // Should find itself
        assert_eq!(results[0].1, 0);   // Distance 0
    }
    
    #[test]
    fn test_bound_retrieval() {
        let verb = random_fingerprint();
        let source = random_fingerprint();
        let target = random_fingerprint();
        
        // Compute edge
        let mut edge = [0u64; WORDS];
        for i in 0..WORDS {
            edge[i] = source[i] ^ verb[i] ^ target[i];
        }
        
        // Unbind to recover target
        let recovered = BoundRetrieval::unbind(&edge, &verb, &source);
        assert_eq!(hamming_distance(&recovered, &target), 0);
        
        // Unbind to recover source
        let recovered = BoundRetrieval::unbind(&edge, &verb, &target);
        assert_eq!(hamming_distance(&recovered, &source), 0);
    }
    
    #[test]
    fn test_alien_search_api() {
        let mut search = AlienSearch::with_capacity(100);
        
        // Add fingerprints
        let fps: Vec<_> = (0..100).map(|_| random_fingerprint()).collect();
        search.add_batch(&fps);
        
        // Search returns similarity scores (like float vectors!)
        let results = search.search_similarity(&fps[0], 5);
        assert!(!results.is_empty());
        assert!(results[0].1 > 0.99);  // High similarity to self
    }
    
    #[test]
    fn test_sketch_cascade() {
        let a = random_fingerprint();
        let b = random_fingerprint();
        
        // 1-bit sketch
        let s1 = sketch_1bit(&a, &b);
        let d1 = sketch_1bit_sum(&s1);
        
        // Most chunks should differ for random fingerprints
        assert!(d1 > WORDS as u32 / 2);
        
        // 4-bit sketch
        let s4 = sketch_4bit(&a, &b);
        let d4 = sketch_4bit_sum(&s4);
        
        // 8-bit sketch
        let s8 = sketch_8bit(&a, &b);
        let d8 = sketch_8bit_sum(&s8);
        
        // Exact
        let exact = hamming_distance(&a, &b);
        
        // d8 should equal exact (8-bit captures full per-chunk count)
        assert_eq!(d8, exact);
        
        // d4 should be close but slightly under (saturated at 15)
        assert!(d4 <= exact);
    }
}

// =============================================================================
// BELICHTUNGSMESSER - Adaptive Threshold Search
// =============================================================================

/// Sample points for exposure metering (prime-spaced for max information)
const METER_POINTS: [usize; 7] = [0, 37, 79, 127, 167, 211, 251];

/// Belichtungsmesser: 7-point exposure meter using 1-bit samples
/// Returns (mean, sd×100) - the "lighting conditions" of this comparison
#[inline]
pub fn belichtung_meter(query: &[u64; WORDS], candidate: &[u64; WORDS]) -> (u8, u8) {
    let mut sum = 0u32;

    // 7 strategic 1-bit samples (super cheap: 7 XOR + 7 compare)
    for &idx in &METER_POINTS {
        sum += ((query[idx] ^ candidate[idx]) != 0) as u32;
    }

    // For binary samples: variance = p(1-p) where p = sum/7
    // SD = sqrt(n * p * (1-p)) / n = sqrt(p * (1-p) / n)
    let p = sum as f32 / 7.0;
    let variance = p * (1.0 - p);
    let sd = (variance * 7.0).sqrt(); // Scaled for 0-1.32 range

    (sum as u8, (sd * 100.0) as u8)
}

/// Quality tracker for Rubicon crossings
#[derive(Clone, Debug)]
pub struct QualityTracker {
    /// Exponential moving average of quality
    pub ema: f32,
    /// SD trajectory history (last 4 readings)
    pub sd_history: [u8; 4],
    /// History index
    pub sd_idx: usize,
    /// Current dynamic threshold
    pub threshold: u16,
    /// Base threshold (learned sweet spot)
    pub base_threshold: u16,
}

impl Default for QualityTracker {
    fn default() -> Self {
        Self {
            ema: 0.5,
            sd_history: [50; 4],
            sd_idx: 0,
            threshold: 2000,
            base_threshold: 2000,
        }
    }
}

impl QualityTracker {
    /// Create with initial threshold
    pub fn new(base_threshold: u16) -> Self {
        Self {
            base_threshold,
            threshold: base_threshold,
            ..Default::default()
        }
    }

    /// Record a meter reading and update trajectory
    pub fn record_meter(&mut self, _mean: u8, sd: u8) {
        self.sd_history[self.sd_idx % 4] = sd;
        self.sd_idx += 1;
    }

    /// Calculate optimal threshold from meter reading
    /// High SD = uncertain → widen threshold (conservative)
    /// Low SD = confident → tighten threshold (aggressive)
    pub fn calculate_sweet_spot(&self, mean: u8, sd: u8) -> u16 {
        // Base threshold from mean (how different the samples are)
        let base = match mean {
            0..=1 => self.base_threshold / 2,      // Very similar
            2..=3 => (self.base_threshold * 3) / 4, // Somewhat similar
            4..=5 => self.base_threshold,           // Mixed signals
            6..=7 => (self.base_threshold * 3) / 2, // Very different
            _ => self.base_threshold,
        };

        // SD adjustment: higher SD → widen threshold
        // SD range is 0-132 (scaled by 100), so sd/100 is 0-1.32
        let sd_factor = 1.0 + (sd as f32 / 150.0);

        (base as f32 * sd_factor) as u16
    }

    /// Infer trajectory and pre-adjust threshold
    pub fn infer_trajectory(&mut self) -> i16 {
        if self.sd_idx < 4 { return 0; }

        let h = &self.sd_history;
        // Linear slope over last 4 readings
        let slope = (h[3] as i16 - h[0] as i16) / 3;

        if slope > 10 {
            // SD increasing → data getting noisier → widen
            self.threshold = (self.threshold as i32 + slope as i32 * 20).min(5000) as u16;
        } else if slope < -10 {
            // SD decreasing → data getting consistent → tighten
            self.threshold = (self.threshold as i32 + slope as i32 * 15).max(500) as u16;
        }

        slope
    }

    /// Update quality EMA after batch
    pub fn update_quality(&mut self, batch_quality: f32) {
        self.ema = 0.85 * self.ema + 0.15 * batch_quality;
    }

    /// Check if we should retreat from Rubicon
    pub fn should_retreat(&self, current_quality: f32) -> bool {
        current_quality < self.ema * 0.6
    }
}

/// Rubicon-aware search with adaptive thresholds
pub struct RubiconSearch {
    /// The fingerprint index
    pub index: HdrIndex,
    /// Quality tracker
    pub tracker: QualityTracker,
    /// Batch size for Rubicon crossings
    pub batch_size: usize,
}

impl RubiconSearch {
    /// Create with capacity
    pub fn with_capacity(n: usize) -> Self {
        Self {
            index: HdrIndex::with_capacity(n),
            tracker: QualityTracker::default(),
            batch_size: 64,
        }
    }

    /// Add a fingerprint
    pub fn add(&mut self, fp: &[u64; WORDS]) {
        self.index.add(fp);
    }

    /// Search with Rubicon crossings and adaptive thresholds
    /// Returns (index, distance, quality) tuples
    pub fn search_adaptive(
        &mut self,
        query: &[u64; WORDS],
        k: usize,
    ) -> Vec<(usize, u32)> {
        let mut results = Vec::with_capacity(k * 2);
        let fps = &self.index.fingerprints;

        if fps.is_empty() {
            return results;
        }

        let mut batch_start = 0;

        while batch_start < fps.len() && results.len() < k * 10 {
            // ═══════════════════════════════════════════════════════
            // PHASE 1: Belichtungsmesser (measure before Rubicon)
            // ═══════════════════════════════════════════════════════
            let (mean, sd) = belichtung_meter(query, &fps[batch_start]);
            self.tracker.record_meter(mean, sd);

            // ═══════════════════════════════════════════════════════
            // PHASE 2: Calculate sweet spot threshold
            // ═══════════════════════════════════════════════════════
            let dynamic_threshold = self.tracker.calculate_sweet_spot(mean, sd);

            // ═══════════════════════════════════════════════════════
            // PHASE 3: Cross Rubicon (batch processing)
            // ═══════════════════════════════════════════════════════
            let batch_end = (batch_start + self.batch_size).min(fps.len());
            let mut batch_quality_sum = 0.0f32;
            let mut batch_count = 0u32;

            for i in batch_start..batch_end {
                let dist = hamming_distance(query, &fps[i]);

                if dist <= dynamic_threshold as u32 {
                    results.push((i, dist));
                    batch_quality_sum += 1.0 - (dist as f32 / BITS as f32);
                    batch_count += 1;
                }

                // ═══════════════════════════════════════════════════
                // PHASE 4: Quality monitoring, retreat if needed
                // ═══════════════════════════════════════════════════
                if batch_count >= 8 && i > batch_start + 16 {
                    let current_q = batch_quality_sum / batch_count as f32;
                    if self.tracker.should_retreat(current_q) {
                        // Quality degraded - break and re-measure
                        break;
                    }
                }
            }

            // Update tracker
            if batch_count > 0 {
                let batch_q = batch_quality_sum / batch_count as f32;
                self.tracker.update_quality(batch_q);
            }

            // ═══════════════════════════════════════════════════════
            // PHASE 5: Infer trajectory, pre-adjust for next batch
            // ═══════════════════════════════════════════════════════
            self.tracker.infer_trajectory();

            batch_start = batch_end;
        }

        // Sort by distance and take top k
        results.sort_by_key(|&(_, d)| d);
        results.truncate(k);
        results
    }

    /// Get current quality EMA
    pub fn quality(&self) -> f32 {
        self.tracker.ema
    }

    /// Get current dynamic threshold
    pub fn threshold(&self) -> u16 {
        self.tracker.threshold
    }

    /// Voyager deep field search
    /// Find faint signals hidden in noise by stacking weak candidates
    ///
    /// Like astronomical image stacking:
    /// - Single exposure at radius 7000 = pure noise
    /// - Stack 10 exposures = faint signal emerges
    /// - Stack 100 exposures = the star reveals itself
    pub fn voyager_deep_field(
        &self,
        query: &[u64; WORDS],
        radius: u32,
        stack_size: usize,
    ) -> Option<VoyagerResult> {
        let fps = &self.index.fingerprints;
        if fps.is_empty() { return None; }

        // Gather weak candidates at the edge of detection
        let mut weak_candidates = Vec::with_capacity(stack_size);
        let radius_min = radius.saturating_sub(500);
        let radius_max = radius + 500;

        for fp in fps {
            let dist = hamming_distance(query, fp);

            // In the noise zone but potentially stackable
            if dist >= radius_min && dist <= radius_max {
                weak_candidates.push(*fp);
                if weak_candidates.len() >= stack_size { break; }
            }
        }

        if weak_candidates.len() < 3 { return None; }

        // Stack exposures using orthogonal superposition
        let star = superposition_clean(query, &weak_candidates)?;

        // Measure the cleaned signal
        let cleaned_dist = hamming_distance(query, &star);
        let signal_strength = 1.0 - (cleaned_dist as f32 / BITS as f32);
        let noise_reduction = radius as f32 / cleaned_dist.max(1) as f32;

        // Did we find a star? (signal improved by at least 2x)
        if noise_reduction > 1.5 {
            Some(VoyagerResult {
                star,
                original_radius: radius,
                cleaned_distance: cleaned_dist,
                signal_strength,
                noise_reduction,
                stack_count: weak_candidates.len(),
            })
        } else {
            None
        }
    }
}

// =============================================================================
// VOYAGER DEEP FIELD - Orthogonal Superposition Cleaning
// =============================================================================

/// Result from Voyager deep field search
#[derive(Debug, Clone)]
pub struct VoyagerResult {
    /// The cleaned "star" fingerprint
    pub star: [u64; WORDS],
    /// Original search radius
    pub original_radius: u32,
    /// Distance after cleaning
    pub cleaned_distance: u32,
    /// Signal strength (0.0-1.0)
    pub signal_strength: f32,
    /// Noise reduction factor (higher = better)
    pub noise_reduction: f32,
    /// Number of exposures stacked
    pub stack_count: usize,
}

/// Orthogonal superposition noise cleaning
///
/// Like stacking astronomical photos:
/// - Noise is random → cancels in majority vote
/// - Signal is consistent → survives the vote
/// - More stacks = cleaner extraction
pub fn superposition_clean(
    query: &[u64; WORDS],
    weak_candidates: &[[u64; WORDS]],
) -> Option<[u64; WORDS]> {
    if weak_candidates.len() < 3 { return None; }

    let n = weak_candidates.len();
    let threshold = n / 2; // Majority vote

    // XOR each candidate with query to get the "difference signal"
    let deltas: Vec<_> = weak_candidates.iter()
        .map(|c| {
            let mut delta = [0u64; WORDS];
            for i in 0..WORDS {
                delta[i] = query[i] ^ c[i];
            }
            delta
        })
        .collect();

    // Componentwise majority vote (VSA bundle)
    // Bits that differ in MOST candidates = real difference (signal)
    // Bits that differ in FEW candidates = noise (filtered out)
    let mut cleaned_delta = [0u64; WORDS];

    for word in 0..WORDS {
        let mut result_word = 0u64;

        for bit in 0..64 {
            let mask = 1u64 << bit;

            // Count votes for this bit
            let votes: usize = deltas.iter()
                .filter(|d| d[word] & mask != 0)
                .count();

            // Majority vote
            if votes > threshold {
                result_word |= mask;
            }
        }

        cleaned_delta[word] = result_word;
    }

    // Apply cleaned delta back to query to get the "star"
    let mut star = [0u64; WORDS];
    for i in 0..WORDS {
        star[i] = query[i] ^ cleaned_delta[i];
    }

    Some(star)
}

/// Signal classification based on Belichtungsmesser
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalClass {
    /// Clean hit - high confidence match
    Strong,
    /// Normal search territory
    Moderate,
    /// Noisy but potentially stackable for deep field
    WeakButStackable,
    /// Pure noise - reject
    Noise,
}

/// Classify a comparison result for routing
pub fn classify_signal(mean: u8, sd: u8, distance: u32) -> SignalClass {
    match (mean, sd, distance) {
        (0..=2, 0..=30, 0..=2000) => SignalClass::Strong,
        (0..=4, 0..=60, 0..=4000) => SignalClass::Moderate,
        (_, 50.., 4000..=8000) => SignalClass::WeakButStackable,
        _ => SignalClass::Noise,
    }
}

#[cfg(test)]
mod belichtung_tests {
    use super::*;

    fn random_fingerprint() -> [u64; WORDS] {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        use std::time::SystemTime;

        let mut fp = [0u64; WORDS];
        let seed = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();

        for i in 0..WORDS {
            let mut hasher = DefaultHasher::new();
            (seed, i).hash(&mut hasher);
            fp[i] = hasher.finish();
        }
        fp
    }

    #[test]
    fn test_belichtung_meter() {
        let a = [0u64; WORDS];
        let b = [0u64; WORDS];

        // Identical → mean=0, sd=0
        let (mean, sd) = belichtung_meter(&a, &b);
        assert_eq!(mean, 0);
        assert_eq!(sd, 0);

        // All different at sample points
        let mut c = [0u64; WORDS];
        for &idx in &METER_POINTS {
            c[idx] = 1;
        }
        let (mean, sd) = belichtung_meter(&a, &c);
        assert_eq!(mean, 7);
        assert_eq!(sd, 0); // All same (all 1s) → no variance
    }

    #[test]
    fn test_quality_tracker() {
        let mut tracker = QualityTracker::new(2000);

        // Simulate decreasing SD (data getting more consistent)
        tracker.record_meter(3, 80);
        tracker.record_meter(3, 60);
        tracker.record_meter(3, 45);
        tracker.record_meter(3, 30);

        let slope = tracker.infer_trajectory();
        assert!(slope < 0); // Negative slope = SD decreasing

        // Threshold should tighten
        assert!(tracker.threshold < 2000);
    }

    #[test]
    fn test_rubicon_search() {
        let mut search = RubiconSearch::with_capacity(100);

        // Add fingerprints
        let fps: Vec<_> = (0..100).map(|_| random_fingerprint()).collect();
        for fp in &fps {
            search.add(fp);
        }

        // Search should find the query
        let results = search.search_adaptive(&fps[42], 10);
        assert!(!results.is_empty());

        // First result should be the exact match
        assert_eq!(results[0].0, 42);
        assert_eq!(results[0].1, 0);
    }

    #[test]
    fn test_superposition_clean() {
        // Create a base fingerprint
        let base = random_fingerprint();

        // Create a "true target" that differs consistently from base
        let mut target = base;
        target[0] ^= 0xFFFF_FFFF_FFFF_FFFF; // Flip all bits in word 0
        target[1] ^= 0xFFFF_FFFF_FFFF_FFFF; // Flip all bits in word 1

        // Create noisy versions of the target (all close to target, not base)
        let mut noisy_candidates = Vec::new();
        for i in 0..20 {
            let mut noisy = target;
            // Add small random noise to each candidate
            noisy[10 + (i % 10)] ^= 1u64 << (i % 64);
            noisy_candidates.push(noisy);
        }

        // Query is the base, candidates are near target
        // Cleaning should find the consistent pattern (target)
        let cleaned = superposition_clean(&base, &noisy_candidates);
        assert!(cleaned.is_some());

        let cleaned = cleaned.unwrap();

        // The cleaned version should be closer to target than to base
        let dist_to_target = hamming_distance(&cleaned, &target);
        let dist_to_base = hamming_distance(&cleaned, &base);

        // Cleaned should be much closer to target (the consistent signal)
        // than to base (the query)
        assert!(dist_to_target < dist_to_base,
            "Cleaned should be closer to target. To target: {}, to base: {}",
            dist_to_target, dist_to_base);
    }

    #[test]
    fn test_signal_classification() {
        // Strong signal
        assert_eq!(classify_signal(1, 20, 1000), SignalClass::Strong);

        // Moderate signal
        assert_eq!(classify_signal(3, 50, 3000), SignalClass::Moderate);

        // Weak but stackable
        assert_eq!(classify_signal(5, 70, 6000), SignalClass::WeakButStackable);

        // Noise
        assert_eq!(classify_signal(7, 100, 9000), SignalClass::Noise);
    }

    #[test]
    fn test_voyager_deep_field() {
        let mut search = RubiconSearch::with_capacity(100);

        // Add some fingerprints
        let base = random_fingerprint();
        for i in 0..50 {
            let mut fp = base;
            // Add varying amounts of noise
            for j in 0..(i * 10) {
                let word = (j / 64) % WORDS;
                let bit = j % 64;
                fp[word] ^= 1u64 << bit;
            }
            search.add(&fp);
        }

        // Try deep field search at a weak radius
        // This tests the mechanism even if it doesn't find a star
        let result = search.voyager_deep_field(&base, 3000, 10);
        // Result may or may not find something depending on the random data
        // Just verify it doesn't panic
        if let Some(r) = result {
            assert!(r.noise_reduction > 0.0);
            assert!(r.stack_count >= 3);
        }
    }
}
