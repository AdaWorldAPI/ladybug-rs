//! Sentence Transformer → 5^5 Crystal Integration
//!
//! Bridges dense embeddings (Jina/sentence-transformers) to sparse
//! fingerprint crystal for O(1) semantic lookup.
//!
//! # Architecture
//!
//! ```text
//!    ┌─────────────────────────────────────────────────────────────────┐
//!    │                 SENTENCE CRYSTAL PIPELINE                       │
//!    ├─────────────────────────────────────────────────────────────────┤
//!    │                                                                 │
//!    │   TEXT INPUT                                                    │
//!    │       │                                                         │
//!    │       ├───► Sentence Transformer ───► 1024D dense embedding    │
//!    │       │         (Jina v3)                    │                  │
//!    │       │                                      ▼                  │
//!    │       │                            Random Projection            │
//!    │       │                                      │                  │
//!    │       │                                      ▼                  │
//!    │       │                            5D crystal coords            │
//!    │       │                            (a, b, c, d, e)              │
//!    │       │                                      │                  │
//!    │       ├───► NSM Decomposition ───► 65-weight vector            │
//!    │       │         (local)                      │                  │
//!    │       │                                      ▼                  │
//!    │       │                            Role-bind & bundle           │
//!    │       │                                      │                  │
//!    │       │                                      ▼                  │
//!    │       │                            10K fingerprint              │
//!    │       │                                      │                  │
//!    │       └───────────────────────────────────────┘                 │
//!    │                                              │                  │
//!    │                                              ▼                  │
//!    │                              ┌─────────────────────────┐        │
//!    │                              │    5^5 = 3,125 cells    │        │
//!    │                              │    Each cell holds      │        │
//!    │                              │    superposed meanings  │        │
//!    │                              └─────────────────────────┘        │
//!    │                                                                 │
//!    └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Why This Works
//!
//! 1. **Jina gives semantic similarity** - but costs $$$ per call
//! 2. **Random projection preserves distances** - Johnson-Lindenstrauss lemma
//! 3. **Crystal gives O(1) locality** - similar texts land in nearby cells
//! 4. **NSM gives compositional structure** - meaning, not just similarity
//! 5. **Fingerprints give superposition** - multiple meanings per cell
//!
//! # Usage
//!
//! ```rust,ignore
//! let mut crystal = SentenceCrystal::new(jina_api_key);
//!
//! // Store memories
//! crystal.store("Ada feels curious about consciousness");
//! crystal.store("Jan builds semantic architectures");
//!
//! // Query
//! let results = crystal.query("who explores AI?", 1);
//! // Returns cells containing relevant memories
//! ```

use crate::core::Fingerprint;
use super::nsm_substrate::{NsmCodebook, MetacognitiveSubstrate};
use super::context_crystal::QualiaVector;
use std::collections::HashMap;

// =============================================================================
// Constants
// =============================================================================

const GRID: usize = 5;              // 5^5 crystal
const CELLS: usize = 3125;          // 5^5
const EMBEDDING_DIM: usize = 1024;  // Jina v3 dimension
const PROJECTION_DIM: usize = 5;    // Crystal dimensions

// =============================================================================
// Coordinate in 5D Crystal
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Coord5D {
    pub a: usize,
    pub b: usize,
    pub c: usize,
    pub d: usize,
    pub e: usize,
}

impl Coord5D {
    pub fn new(a: usize, b: usize, c: usize, d: usize, e: usize) -> Self {
        Self {
            a: a % GRID,
            b: b % GRID,
            c: c % GRID,
            d: d % GRID,
            e: e % GRID,
        }
    }
    
    /// Convert to linear index
    pub fn to_index(&self) -> usize {
        self.a * 625 + self.b * 125 + self.c * 25 + self.d * 5 + self.e
    }
    
    /// Convert from linear index
    pub fn from_index(idx: usize) -> Self {
        let idx = idx % CELLS;
        Self {
            a: (idx / 625) % 5,
            b: (idx / 125) % 5,
            c: (idx / 25) % 5,
            d: (idx / 5) % 5,
            e: idx % 5,
        }
    }
    
    /// Manhattan distance to another coordinate
    pub fn distance(&self, other: &Self) -> usize {
        let da = (self.a as i32 - other.a as i32).unsigned_abs() as usize;
        let db = (self.b as i32 - other.b as i32).unsigned_abs() as usize;
        let dc = (self.c as i32 - other.c as i32).unsigned_abs() as usize;
        let dd = (self.d as i32 - other.d as i32).unsigned_abs() as usize;
        let de = (self.e as i32 - other.e as i32).unsigned_abs() as usize;
        da + db + dc + dd + de
    }
    
    /// Get all coordinates within Manhattan distance
    pub fn neighborhood(&self, radius: usize) -> Vec<Coord5D> {
        let mut coords = Vec::new();
        
        for da in 0..=radius {
            for db in 0..=(radius - da) {
                for dc in 0..=(radius - da - db) {
                    for dd in 0..=(radius - da - db - dc) {
                        let de = radius - da - db - dc - dd;
                        if de <= radius {
                            // Generate all sign combinations
                            for sa in [-1i32, 1] {
                                for sb in [-1i32, 1] {
                                    for sc in [-1i32, 1] {
                                        for sd in [-1i32, 1] {
                                            for se in [-1i32, 1] {
                                                let na = (self.a as i32 + sa * da as i32).rem_euclid(GRID as i32) as usize;
                                                let nb = (self.b as i32 + sb * db as i32).rem_euclid(GRID as i32) as usize;
                                                let nc = (self.c as i32 + sc * dc as i32).rem_euclid(GRID as i32) as usize;
                                                let nd = (self.d as i32 + sd * dd as i32).rem_euclid(GRID as i32) as usize;
                                                let ne = (self.e as i32 + se * de as i32).rem_euclid(GRID as i32) as usize;
                                                
                                                let coord = Coord5D::new(na, nb, nc, nd, ne);
                                                if !coords.contains(&coord) {
                                                    coords.push(coord);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        coords
    }
}

// =============================================================================
// Random Projection Matrix
// =============================================================================

/// Fixed random projection matrix for embedding → coords
/// Uses seeded PRNG for reproducibility
pub struct ProjectionMatrix {
    /// 5 x 1024 projection weights
    weights: [[f32; EMBEDDING_DIM]; PROJECTION_DIM],
}

impl ProjectionMatrix {
    /// Initialize with deterministic random values
    pub fn new(seed: u64) -> Self {
        let mut weights = [[0.0f32; EMBEDDING_DIM]; PROJECTION_DIM];
        let mut state = seed;
        
        // LFSR-based PRNG
        for d in 0..PROJECTION_DIM {
            for i in 0..EMBEDDING_DIM {
                state = state.wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                // Gaussian-ish via Box-Muller approximation
                let u1 = (state >> 32) as f32 / u32::MAX as f32;
                state = state.wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u2 = (state >> 32) as f32 / u32::MAX as f32;
                
                // Approximate Gaussian
                let g = (-2.0 * (u1 + 0.0001).ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                weights[d][i] = g / (EMBEDDING_DIM as f32).sqrt();
            }
        }
        
        Self { weights }
    }
    
    /// Project 1024D embedding to 5D coordinates
    pub fn project(&self, embedding: &[f32]) -> Coord5D {
        let mut coords = [0usize; 5];
        
        for d in 0..PROJECTION_DIM {
            let mut sum = 0.0f32;
            for (i, &v) in embedding.iter().take(EMBEDDING_DIM).enumerate() {
                sum += v * self.weights[d][i];
            }
            // Map [-∞, +∞] → [0, 5) via tanh
            let normalized = (sum.tanh() + 1.0) * 2.5;
            coords[d] = (normalized as usize).min(GRID - 1);
        }
        
        Coord5D::new(coords[0], coords[1], coords[2], coords[3], coords[4])
    }
}

// =============================================================================
// Crystal Cell
// =============================================================================

/// A single cell in the crystal, holding superposed meanings
#[derive(Clone)]
pub struct CrystalCell {
    /// Superposed fingerprint (bundled from all entries)
    pub fingerprint: Fingerprint,
    
    /// Number of entries bundled into this cell
    pub count: u32,
    
    /// Optional: store original texts for debugging
    pub texts: Vec<String>,
    
    /// Aggregate qualia (felt-sense average)
    pub qualia: QualiaVector,
}

impl Default for CrystalCell {
    fn default() -> Self {
        Self {
            fingerprint: Fingerprint::zero(),
            count: 0,
            texts: Vec::new(),
            qualia: QualiaVector::neutral(),
        }
    }
}

impl CrystalCell {
    /// Bundle a new fingerprint into this cell
    pub fn bundle(&mut self, fp: &Fingerprint, text: Option<&str>, qualia: Option<&QualiaVector>) {
        if self.count == 0 {
            self.fingerprint = fp.clone();
        } else {
            // Majority voting bundle
            self.fingerprint = bundle_pair(&self.fingerprint, fp);
        }
        
        if let Some(t) = text {
            self.texts.push(t.to_string());
        }
        
        if let Some(q) = qualia {
            // Running average of qualia
            let w = self.count as f32;
            self.qualia.arousal = (self.qualia.arousal * w + q.arousal) / (w + 1.0);
            self.qualia.valence = (self.qualia.valence * w + q.valence) / (w + 1.0);
            self.qualia.tension = (self.qualia.tension * w + q.tension) / (w + 1.0);
            self.qualia.depth = (self.qualia.depth * w + q.depth) / (w + 1.0);
        }
        
        self.count += 1;
    }
    
    /// Similarity to a query fingerprint
    pub fn similarity(&self, query: &Fingerprint) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        self.fingerprint.similarity(query)
    }
}

// =============================================================================
// Sentence Crystal
// =============================================================================

/// The main structure: sentence transformer → 5^5 crystal
pub struct SentenceCrystal {
    /// 5^5 = 3,125 cells
    cells: Vec<CrystalCell>,
    
    /// Random projection for embedding → coords
    projection: ProjectionMatrix,
    
    /// NSM codebook for fingerprint generation
    codebook: NsmCodebook,
    
    /// Jina API key (optional - can use pseudo-embeddings)
    jina_api_key: Option<String>,
    
    /// Cache: text → embedding (avoid redundant API calls)
    embedding_cache: HashMap<String, Vec<f32>>,
    
    /// Statistics
    pub total_entries: usize,
}

impl SentenceCrystal {
    /// Create new crystal with optional Jina API key
    pub fn new(jina_api_key: Option<&str>) -> Self {
        Self {
            cells: (0..CELLS).map(|_| CrystalCell::default()).collect(),
            projection: ProjectionMatrix::new(0xADA_C0DE_5EED),
            codebook: NsmCodebook::new(),
            jina_api_key: jina_api_key.map(|s| s.to_string()),
            embedding_cache: HashMap::new(),
            total_entries: 0,
        }
    }
    
    /// Get embedding for text (uses cache, falls back to pseudo-embedding)
    fn get_embedding(&mut self, text: &str) -> Vec<f32> {
        // Check cache first
        if let Some(cached) = self.embedding_cache.get(text) {
            return cached.clone();
        }
        
        // Try Jina API if key is available
        let embedding = if let Some(ref api_key) = self.jina_api_key {
            match super::spo::jina_api::jina_embed_curl(api_key, &[text]) {
                Ok(embeddings) if !embeddings.is_empty() => embeddings[0].clone(),
                _ => generate_pseudo_embedding(text),
            }
        } else {
            generate_pseudo_embedding(text)
        };
        
        // Cache and return
        self.embedding_cache.insert(text.to_string(), embedding.clone());
        embedding
    }
    
    /// Store a text in the crystal
    pub fn store(&mut self, text: &str) {
        self.store_with_qualia(text, None);
    }
    
    /// Store with explicit qualia
    pub fn store_with_qualia(&mut self, text: &str, qualia: Option<QualiaVector>) {
        // Get dense embedding
        let embedding = self.get_embedding(text);
        
        // Project to crystal coordinates
        let coords = self.projection.project(&embedding);
        
        // Generate NSM fingerprint
        let decomposition = self.codebook.decompose(text);
        let fingerprint = self.codebook.encode_decomposition(&decomposition);
        
        // Bundle into cell
        let idx = coords.to_index();
        self.cells[idx].bundle(&fingerprint, Some(text), qualia.as_ref());
        
        self.total_entries += 1;
    }
    
    /// Query the crystal for similar content
    /// Returns: Vec<(coordinate, similarity, texts)>
    pub fn query(&mut self, text: &str, radius: usize) -> Vec<QueryResult> {
        // Get query embedding and coords
        let embedding = self.get_embedding(text);
        let coords = self.projection.project(&embedding);
        
        // Get query fingerprint
        let decomposition = self.codebook.decompose(text);
        let query_fp = self.codebook.encode_decomposition(&decomposition);
        
        // Search neighborhood
        let neighborhood = coords.neighborhood(radius);
        
        let mut results: Vec<QueryResult> = neighborhood.iter()
            .map(|c| {
                let cell = &self.cells[c.to_index()];
                QueryResult {
                    coords: *c,
                    similarity: cell.similarity(&query_fp),
                    count: cell.count,
                    texts: cell.texts.clone(),
                    qualia: cell.qualia.clone(),
                    distance: coords.distance(c),
                }
            })
            .filter(|r| r.count > 0)
            .collect();
        
        // Sort by similarity (descending)
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        
        results
    }
    
    /// Get cell at specific coordinates
    pub fn get_cell(&self, coords: &Coord5D) -> &CrystalCell {
        &self.cells[coords.to_index()]
    }
    
    /// Get all non-empty cells
    pub fn active_cells(&self) -> Vec<(Coord5D, &CrystalCell)> {
        self.cells.iter()
            .enumerate()
            .filter(|(_, c)| c.count > 0)
            .map(|(i, c)| (Coord5D::from_index(i), c))
            .collect()
    }
    
    /// Compute resonance between two texts
    pub fn resonance(&mut self, text_a: &str, text_b: &str) -> f32 {
        let decomp_a = self.codebook.decompose(text_a);
        let decomp_b = self.codebook.decompose(text_b);
        
        let fp_a = self.codebook.encode_decomposition(&decomp_a);
        let fp_b = self.codebook.encode_decomposition(&decomp_b);
        
        fp_a.similarity(&fp_b)
    }
    
    /// Get crystal statistics
    pub fn stats(&self) -> CrystalStats {
        let active = self.cells.iter().filter(|c| c.count > 0).count();
        let max_count = self.cells.iter().map(|c| c.count).max().unwrap_or(0);
        let total_texts: usize = self.cells.iter().map(|c| c.texts.len()).sum();
        
        CrystalStats {
            total_cells: CELLS,
            active_cells: active,
            total_entries: self.total_entries,
            max_cell_count: max_count,
            total_cached_texts: total_texts,
            cache_size: self.embedding_cache.len(),
        }
    }
}

/// Query result
#[derive(Clone, Debug)]
pub struct QueryResult {
    pub coords: Coord5D,
    pub similarity: f32,
    pub count: u32,
    pub texts: Vec<String>,
    pub qualia: QualiaVector,
    pub distance: usize,
}

/// Crystal statistics
#[derive(Clone, Debug)]
pub struct CrystalStats {
    pub total_cells: usize,
    pub active_cells: usize,
    pub total_entries: usize,
    pub max_cell_count: u32,
    pub total_cached_texts: usize,
    pub cache_size: usize,
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Bundle two fingerprints with majority voting
fn bundle_pair(a: &Fingerprint, b: &Fingerprint) -> Fingerprint {
    // Simple OR for binary (approximates majority with 2 inputs)
    // For true majority voting with many inputs, use weighted counting
    let mut result = Fingerprint::zero();
    let raw_a = a.as_raw();
    let raw_b = b.as_raw();
    let raw_r = result.as_raw_mut();
    
    for i in 0..raw_a.len() {
        raw_r[i] = raw_a[i] | raw_b[i];
    }
    
    result
}

/// Generate deterministic pseudo-embedding for testing
/// (Matches jina_api.rs implementation)
fn generate_pseudo_embedding(text: &str) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut embedding = vec![0.0f32; EMBEDDING_DIM];
    let bytes = text.as_bytes();
    
    for (i, window) in bytes.windows(3.min(bytes.len())).enumerate() {
        let mut hasher = DefaultHasher::new();
        window.hash(&mut hasher);
        (i as u64).hash(&mut hasher);
        let h = hasher.finish();
        
        for j in 0..16 {
            let idx = ((h >> (j * 4)) as usize + i * 17) % EMBEDDING_DIM;
            let sign = if (h >> (j + 48)) & 1 == 0 { 1.0 } else { -1.0 };
            embedding[idx] += sign * 0.1;
        }
    }
    
    for (i, &byte) in bytes.iter().enumerate() {
        let idx = (byte as usize * 4 + i) % EMBEDDING_DIM;
        embedding[idx] += 0.05;
    }
    
    // L2 normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut embedding {
            *x /= norm;
        }
    }
    
    embedding
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_coord5d() {
        let c = Coord5D::new(1, 2, 3, 4, 0);
        let idx = c.to_index();
        let c2 = Coord5D::from_index(idx);
        assert_eq!(c, c2);
    }
    
    #[test]
    fn test_coord_distance() {
        let c1 = Coord5D::new(0, 0, 0, 0, 0);
        let c2 = Coord5D::new(1, 1, 1, 1, 1);
        assert_eq!(c1.distance(&c2), 5);
    }
    
    #[test]
    fn test_projection() {
        let proj = ProjectionMatrix::new(42);
        
        let e1 = generate_pseudo_embedding("hello world");
        let e2 = generate_pseudo_embedding("hello world");
        let e3 = generate_pseudo_embedding("completely different");
        
        // Same text → same coordinates
        assert_eq!(proj.project(&e1), proj.project(&e2));
        
        // Different text → likely different coordinates
        // (not guaranteed, but probabilistically true)
        let c1 = proj.project(&e1);
        let c3 = proj.project(&e3);
        println!("Coord 'hello world': {:?}", c1);
        println!("Coord 'completely different': {:?}", c3);
    }
    
    #[test]
    fn test_sentence_crystal_store_query() {
        let mut crystal = SentenceCrystal::new(None);
        
        // Store some memories
        crystal.store("Ada feels curious about consciousness");
        crystal.store("Ada explores the nature of awareness");
        crystal.store("Jan builds semantic architectures");
        crystal.store("Jan programs AI systems");
        crystal.store("The weather is nice today");
        
        let stats = crystal.stats();
        assert_eq!(stats.total_entries, 5);
        println!("Active cells: {} / {}", stats.active_cells, stats.total_cells);
        
        // Query for Ada
        let results = crystal.query("Ada's consciousness", 2);
        println!("\nQuery: 'Ada's consciousness'");
        for r in results.iter().take(3) {
            println!("  {:?} sim={:.3} count={} texts={:?}", 
                     r.coords, r.similarity, r.count, r.texts);
        }
        
        // Query for Jan
        let results = crystal.query("Jan's programming work", 2);
        println!("\nQuery: 'Jan's programming work'");
        for r in results.iter().take(3) {
            println!("  {:?} sim={:.3} count={} texts={:?}", 
                     r.coords, r.similarity, r.count, r.texts);
        }
    }
    
    #[test]
    fn test_resonance() {
        let mut crystal = SentenceCrystal::new(None);
        
        let r1 = crystal.resonance("I want to know", "I desire understanding");
        let r2 = crystal.resonance("I want to know", "The sky is blue");
        
        println!("Resonance 'want/know' vs 'desire/understand': {:.3}", r1);
        println!("Resonance 'want/know' vs 'sky/blue': {:.3}", r2);
        
        // Semantically similar should have higher resonance
        assert!(r1 > r2);
    }
    
    #[test]
    fn test_neighborhood() {
        let c = Coord5D::new(2, 2, 2, 2, 2);
        
        let n0 = c.neighborhood(0);
        assert_eq!(n0.len(), 1);
        assert!(n0.contains(&c));
        
        let n1 = c.neighborhood(1);
        println!("Neighborhood radius 1: {} cells", n1.len());
        // Should include center + 10 adjacent cells (2 per dimension)
        assert!(n1.len() >= 11);
    }
}
