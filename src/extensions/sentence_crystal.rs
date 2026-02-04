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
use crate::storage::bind_space::{Addr, BindSpace, dn_path_to_addr};
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

    // =========================================================================
    // DN TREE AWARENESS (for hierarchical context)
    // =========================================================================

    /// DN paths associated with entries in this cell
    /// Enables context-aware retrieval: "what does Ada know about X?"
    pub dn_contexts: Vec<String>,

    /// BindSpace addresses bound to this cell
    /// Enables zero-copy traversal from crystal -> DN tree
    pub bound_addrs: Vec<Addr>,

    /// Maximum rung level stored in this cell (R0-R9)
    /// R0=public, R9=soul-level (most private)
    pub max_rung: u8,

    /// Average tree depth of entries (0=root-level, higher=deeper)
    pub avg_depth: f32,
}

impl Default for CrystalCell {
    fn default() -> Self {
        Self {
            fingerprint: Fingerprint::zero(),
            count: 0,
            texts: Vec::new(),
            qualia: QualiaVector::neutral(),
            dn_contexts: Vec::new(),
            bound_addrs: Vec::new(),
            max_rung: 0,
            avg_depth: 0.0,
        }
    }
}

impl CrystalCell {
    /// Bundle a new fingerprint into this cell
    pub fn bundle(&mut self, fp: &Fingerprint, text: Option<&str>, qualia: Option<&QualiaVector>) {
        self.bundle_with_context(fp, text, qualia, None, None);
    }

    /// Bundle with DN tree context for aware traversal
    pub fn bundle_with_context(
        &mut self,
        fp: &Fingerprint,
        text: Option<&str>,
        qualia: Option<&QualiaVector>,
        dn_path: Option<&str>,
        bound_addr: Option<Addr>,
    ) {
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

        // DN tree context binding
        if let Some(path) = dn_path {
            self.dn_contexts.push(path.to_string());
        }

        if let Some(addr) = bound_addr {
            self.bound_addrs.push(addr);
        }

        self.count += 1;
    }

    /// Update rung and depth from BindSpace node
    pub fn update_dn_metadata(&mut self, rung: u8, depth: u8) {
        self.max_rung = self.max_rung.max(rung);
        let w = (self.count.saturating_sub(1)) as f32;
        self.avg_depth = (self.avg_depth * w + depth as f32) / self.count as f32;
    }

    /// Similarity to a query fingerprint
    pub fn similarity(&self, query: &Fingerprint) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        self.fingerprint.similarity(query)
    }

    /// Check if cell has entries from a specific DN context
    pub fn has_context(&self, dn_prefix: &str) -> bool {
        self.dn_contexts.iter().any(|c| c.starts_with(dn_prefix))
    }

    /// Get all bound addresses (for zero-copy DN tree traversal)
    pub fn bound_addresses(&self) -> &[Addr] {
        &self.bound_addrs
    }

    /// Filter entries by DN context prefix
    pub fn entries_in_context(&self, dn_prefix: &str) -> Vec<(&str, Option<&Addr>)> {
        self.dn_contexts
            .iter()
            .enumerate()
            .filter(|(_, ctx)| ctx.starts_with(dn_prefix))
            .map(|(i, ctx)| (ctx.as_str(), self.bound_addrs.get(i)))
            .collect()
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

    // =========================================================================
    // DN TREE INTEGRATION (for aware traversal)
    // =========================================================================

    /// Index: DN path prefix → cell indices
    /// Enables fast "what does Ada know?" style queries
    dn_index: HashMap<String, Vec<usize>>,

    /// Index: Addr → cell index
    /// Enables crystal lookup from DN tree traversal
    addr_index: HashMap<u16, usize>,
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
            dn_index: HashMap::new(),
            addr_index: HashMap::new(),
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
            match super::spo::jina_embed_curl(api_key, &[text]) {
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

        // Generate NSM fingerprint (encode combines decompose + encoding)
        let fingerprint = self.codebook.encode(text);

        // Bundle into cell
        let idx = coords.to_index();
        self.cells[idx].bundle(&fingerprint, Some(text), qualia.as_ref());

        self.total_entries += 1;
    }

    // =========================================================================
    // DN-AWARE STORAGE (context binding)
    // =========================================================================

    /// Store text with DN tree context for aware traversal
    ///
    /// This binds the sentence to a position in the DN tree, enabling:
    /// - "What does Ada:A:soul know about X?" queries
    /// - Rung-filtered access control
    /// - Hierarchical context propagation
    pub fn store_with_dn_context(
        &mut self,
        text: &str,
        dn_path: &str,
        qualia: Option<QualiaVector>,
        rung: u8,
        depth: u8,
    ) {
        // Get dense embedding
        let embedding = self.get_embedding(text);

        // Project to crystal coordinates
        let coords = self.projection.project(&embedding);

        // Generate NSM fingerprint
        let fingerprint = self.codebook.encode(text);

        // Compute DN address for binding
        let addr = dn_path_to_addr(dn_path);

        // Bundle into cell with context
        let idx = coords.to_index();
        self.cells[idx].bundle_with_context(
            &fingerprint,
            Some(text),
            qualia.as_ref(),
            Some(dn_path),
            Some(addr),
        );
        self.cells[idx].update_dn_metadata(rung, depth);

        // Update DN index for fast context queries
        self.index_dn_path(dn_path, idx);

        // Update addr index for tree -> crystal traversal
        self.addr_index.insert(addr.0, idx);

        self.total_entries += 1;
    }

    /// Store with BindSpace integration - reads rung/depth from existing node
    pub fn store_with_bind_space(
        &mut self,
        text: &str,
        dn_path: &str,
        bind_space: &BindSpace,
        qualia: Option<QualiaVector>,
    ) {
        let addr = dn_path_to_addr(dn_path);
        let (rung, depth) = bind_space
            .read(addr)
            .map(|n| (n.rung, n.depth))
            .unwrap_or((0, 0));

        self.store_with_dn_context(text, dn_path, qualia, rung, depth);
    }

    /// Index DN path for fast prefix queries
    fn index_dn_path(&mut self, path: &str, cell_idx: usize) {
        // Index full path
        self.dn_index
            .entry(path.to_string())
            .or_default()
            .push(cell_idx);

        // Index all prefixes for hierarchical queries
        // "Ada:A:soul:identity" indexes under:
        // - "Ada"
        // - "Ada:A"
        // - "Ada:A:soul"
        // - "Ada:A:soul:identity"
        let mut prefix = String::new();
        for (i, segment) in path.split(':').enumerate() {
            if i > 0 {
                prefix.push(':');
            }
            prefix.push_str(segment);

            self.dn_index
                .entry(prefix.clone())
                .or_default()
                .push(cell_idx);
        }
    }
    
    /// Query the crystal for similar content
    /// Returns: Vec<(coordinate, similarity, texts)>
    pub fn query(&mut self, text: &str, radius: usize) -> Vec<QueryResult> {
        // Get query embedding and coords
        let embedding = self.get_embedding(text);
        let coords = self.projection.project(&embedding);

        // Get query fingerprint
        let query_fp = self.codebook.encode(text);

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
                    dn_contexts: cell.dn_contexts.clone(),
                    bound_addrs: cell.bound_addrs.clone(),
                    max_rung: cell.max_rung,
                }
            })
            .filter(|r| r.count > 0)
            .collect();

        // Sort by similarity (descending)
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());

        results
    }

    // =========================================================================
    // DN-AWARE QUERIES (context-filtered, rung-controlled)
    // =========================================================================

    /// Query with rung filter (access control)
    ///
    /// Only returns results where max_rung <= allowed_rung
    /// R0=public, R9=soul-level (most private)
    pub fn query_with_rung(
        &mut self,
        text: &str,
        radius: usize,
        allowed_rung: u8,
    ) -> Vec<QueryResult> {
        self.query(text, radius)
            .into_iter()
            .filter(|r| r.max_rung <= allowed_rung)
            .collect()
    }

    /// Query within a DN context ("what does Ada know about X?")
    ///
    /// Filters results to only cells that have entries from the given DN prefix.
    /// Example: query_in_context("consciousness", "Ada:A:soul", 2)
    pub fn query_in_context(
        &mut self,
        text: &str,
        dn_prefix: &str,
        radius: usize,
    ) -> Vec<QueryResult> {
        // Get query embedding and coords
        let embedding = self.get_embedding(text);
        let coords = self.projection.project(&embedding);
        let query_fp = self.codebook.encode(text);

        // Get candidate cells from DN index (fast path)
        let context_cells: std::collections::HashSet<usize> = self
            .dn_index
            .get(dn_prefix)
            .map(|v| v.iter().copied().collect())
            .unwrap_or_default();

        // Search neighborhood but only include cells in context
        let neighborhood = coords.neighborhood(radius);

        let mut results: Vec<QueryResult> = neighborhood
            .iter()
            .filter(|c| context_cells.contains(&c.to_index()))
            .map(|c| {
                let cell = &self.cells[c.to_index()];
                QueryResult {
                    coords: *c,
                    similarity: cell.similarity(&query_fp),
                    count: cell.count,
                    texts: cell.texts.clone(),
                    qualia: cell.qualia.clone(),
                    distance: coords.distance(c),
                    dn_contexts: cell.dn_contexts.clone(),
                    bound_addrs: cell.bound_addrs.clone(),
                    max_rung: cell.max_rung,
                }
            })
            .filter(|r| r.count > 0)
            .collect();

        // Sort by similarity
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());

        results
    }

    /// Query with DN context AND rung filter
    pub fn query_in_context_with_rung(
        &mut self,
        text: &str,
        dn_prefix: &str,
        radius: usize,
        allowed_rung: u8,
    ) -> Vec<QueryResult> {
        self.query_in_context(text, dn_prefix, radius)
            .into_iter()
            .filter(|r| r.max_rung <= allowed_rung)
            .collect()
    }

    /// Tree-aware similarity: boost score based on DN path distance
    ///
    /// Combines semantic similarity with hierarchical closeness.
    /// Results from same DN subtree get boosted.
    pub fn query_tree_aware(
        &mut self,
        text: &str,
        query_dn_context: &str,
        radius: usize,
        tree_weight: f32,
    ) -> Vec<QueryResult> {
        let mut results = self.query(text, radius);

        // Boost scores based on DN path similarity
        for result in &mut results {
            let max_context_boost = result
                .dn_contexts
                .iter()
                .map(|ctx| dn_path_similarity(query_dn_context, ctx))
                .fold(0.0f32, f32::max);

            // Combined score: (1-w)*semantic + w*hierarchical
            let semantic = result.similarity;
            let boosted = semantic * (1.0 - tree_weight) + max_context_boost * tree_weight;
            result.similarity = boosted;
        }

        // Re-sort by boosted similarity
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());

        results
    }

    // =========================================================================
    // HYBRID TRAVERSAL (semantic + hierarchical)
    // =========================================================================

    /// Get cell from DN tree address (for tree -> crystal traversal)
    pub fn cell_from_addr(&self, addr: Addr) -> Option<&CrystalCell> {
        self.addr_index.get(&addr.0).map(|&idx| &self.cells[idx])
    }

    /// Get all cells in a DN subtree
    pub fn cells_in_subtree(&self, dn_prefix: &str) -> Vec<(usize, &CrystalCell)> {
        self.dn_index
            .get(dn_prefix)
            .map(|indices| {
                indices
                    .iter()
                    .map(|&idx| (idx, &self.cells[idx]))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Traverse DN tree and collect semantic content
    ///
    /// Given a BindSpace and starting DN path, walks the tree and returns
    /// crystal cells at each node. Enables "show me everything under Ada:A:soul".
    pub fn traverse_and_collect(
        &self,
        bind_space: &mut BindSpace,
        start_dn: &str,
        max_depth: usize,
    ) -> Vec<TraversalResult> {
        let mut results = Vec::new();
        let mut visited = std::collections::HashSet::new();

        // BFS from start
        let start_addr = dn_path_to_addr(start_dn);
        let mut frontier = vec![(start_addr, start_dn.to_string(), 0usize)];

        // Ensure CSR is built
        bind_space.rebuild_csr();

        while let Some((addr, path, depth)) = frontier.pop() {
            if depth > max_depth || !visited.insert(addr.0) {
                continue;
            }

            // Get crystal cell if bound
            let cell = self.cell_from_addr(addr);

            results.push(TraversalResult {
                addr,
                dn_path: path.clone(),
                depth,
                cell_idx: self.addr_index.get(&addr.0).copied(),
                has_content: cell.map(|c| c.count > 0).unwrap_or(false),
            });

            // Get children from CSR
            let children = bind_space.children_raw(addr);
            for &child_raw in children {
                let child_addr = Addr(child_raw);
                // Construct child DN path (would need label lookup in real impl)
                let child_path = format!("{}:child_{}", path, child_raw);
                frontier.push((child_addr, child_path, depth + 1));
            }
        }

        results
    }

    /// Find semantic neighbors that share DN ancestry
    ///
    /// "What concepts similar to X are related to Y in the tree?"
    pub fn semantic_siblings(
        &mut self,
        text: &str,
        bind_space: &BindSpace,
        radius: usize,
    ) -> Vec<(QueryResult, Vec<Addr>)> {
        let results = self.query(text, radius);

        results
            .into_iter()
            .map(|r| {
                // For each result, find shared ancestors
                let shared: Vec<Addr> = r
                    .bound_addrs
                    .iter()
                    .flat_map(|&addr| bind_space.ancestors(addr).collect::<Vec<_>>())
                    .collect();
                (r, shared)
            })
            .collect()
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
        let fp_a = self.codebook.encode(text_a);
        let fp_b = self.codebook.encode(text_b);

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
    /// DN contexts associated with this cell
    pub dn_contexts: Vec<String>,
    /// Bound addresses for tree traversal
    pub bound_addrs: Vec<Addr>,
    /// Maximum rung level (R0-R9) in this cell
    pub max_rung: u8,
}

/// Result from DN tree traversal
#[derive(Clone, Debug)]
pub struct TraversalResult {
    /// Node address in DN tree
    pub addr: Addr,
    /// DN path string
    pub dn_path: String,
    /// Depth from traversal start
    pub depth: usize,
    /// Crystal cell index (if bound)
    pub cell_idx: Option<usize>,
    /// Whether this node has semantic content
    pub has_content: bool,
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
    a.or(b)
}

/// Compute DN path similarity based on shared ancestry
///
/// Returns 0.0-1.0 where 1.0 = identical paths, 0.0 = no shared prefix
/// "Ada:A:soul:x" vs "Ada:A:soul:y" = 0.75 (3/4 segments shared)
/// "Ada:A:soul" vs "Jan:B:core" = 0.0 (no shared prefix)
fn dn_path_similarity(a: &str, b: &str) -> f32 {
    let a_parts: Vec<&str> = a.split(':').collect();
    let b_parts: Vec<&str> = b.split(':').collect();

    let max_len = a_parts.len().max(b_parts.len());
    if max_len == 0 {
        return 1.0;
    }

    let shared = a_parts
        .iter()
        .zip(b_parts.iter())
        .take_while(|(x, y)| x == y)
        .count();

    shared as f32 / max_len as f32
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
        // Should include center + adjacent cells (at least 10 for radius 1)
        // Note: exact count depends on deduplication in neighborhood algorithm
        assert!(n1.len() >= 10);
    }

    // =========================================================================
    // DN-AWARE CRYSTAL TESTS
    // =========================================================================

    #[test]
    fn test_dn_path_similarity() {
        // Identical paths
        assert_eq!(dn_path_similarity("Ada:A:soul", "Ada:A:soul"), 1.0);

        // Shared prefix
        let sim = dn_path_similarity("Ada:A:soul:identity", "Ada:A:soul:core");
        assert!(sim > 0.5 && sim < 1.0, "Expected ~0.75, got {}", sim);

        // No shared prefix
        assert_eq!(dn_path_similarity("Ada:A", "Jan:B"), 0.0);

        // Partial overlap
        let sim2 = dn_path_similarity("Ada:A:soul", "Ada:A:body");
        assert!(sim2 > 0.3 && sim2 < 0.8, "Expected ~0.66, got {}", sim2);
    }

    #[test]
    fn test_store_with_dn_context() {
        let mut crystal = SentenceCrystal::new(None);

        // Store with DN context
        crystal.store_with_dn_context(
            "Ada feels curious about consciousness",
            "Ada:A:soul:curiosity",
            None,
            3, // rung
            4, // depth
        );

        crystal.store_with_dn_context(
            "Ada explores the nature of awareness",
            "Ada:A:soul:exploration",
            None,
            3,
            4,
        );

        crystal.store_with_dn_context(
            "Jan builds semantic architectures",
            "Jan:J:core:building",
            None,
            1,
            4,
        );

        assert_eq!(crystal.total_entries, 3);

        // Check DN index was built
        let ada_cells = crystal.cells_in_subtree("Ada");
        assert!(!ada_cells.is_empty(), "Should have cells under Ada");

        let ada_soul_cells = crystal.cells_in_subtree("Ada:A:soul");
        assert!(!ada_soul_cells.is_empty(), "Should have cells under Ada:A:soul");
    }

    #[test]
    fn test_query_in_context() {
        let mut crystal = SentenceCrystal::new(None);

        // Store Ada's knowledge
        crystal.store_with_dn_context(
            "consciousness is mysterious",
            "Ada:A:soul:thoughts",
            None,
            5,
            4,
        );

        // Store Jan's knowledge
        crystal.store_with_dn_context(
            "consciousness emerges from complexity",
            "Jan:J:core:thoughts",
            None,
            2,
            4,
        );

        // Query in Ada's context only
        let results = crystal.query_in_context("what is consciousness?", "Ada", 3);

        // Should find Ada's entry, not Jan's
        for r in &results {
            let has_ada_context = r.dn_contexts.iter().any(|c| c.starts_with("Ada"));
            assert!(has_ada_context, "Results should only be from Ada context");
        }
    }

    #[test]
    fn test_query_with_rung_filter() {
        let mut crystal = SentenceCrystal::new(None);

        // Store public knowledge (R1)
        crystal.store_with_dn_context(
            "the sky is blue",
            "Public:facts:sky",
            None,
            1, // public
            2,
        );

        // Store private knowledge (R7)
        crystal.store_with_dn_context(
            "my deepest secret",
            "Ada:A:soul:secrets",
            None,
            7, // private
            4,
        );

        // Query with low rung (should only see public)
        let public_results = crystal.query_with_rung("what color is the sky?", 3, 2);
        for r in &public_results {
            assert!(r.max_rung <= 2, "Should only see public entries");
        }

        // Query with high rung (should see both)
        let all_results = crystal.query_with_rung("tell me everything", 3, 9);
        // May find private entries with high enough rung
        println!("With rung 9, found {} results", all_results.len());
    }

    #[test]
    fn test_tree_aware_query() {
        let mut crystal = SentenceCrystal::new(None);

        // Store semantically similar content in different trees
        crystal.store_with_dn_context(
            "exploring new ideas",
            "Ada:A:soul:exploration",
            None,
            3,
            4,
        );

        crystal.store_with_dn_context(
            "exploring new territories",
            "Jan:J:core:exploration",
            None,
            2,
            4,
        );

        // Query with tree awareness from Ada's context
        // Should boost Ada's result due to path similarity
        let results = crystal.query_tree_aware(
            "exploration and discovery",
            "Ada:A:soul",
            3,
            0.3, // 30% weight to tree proximity
        );

        if !results.is_empty() {
            println!("Tree-aware results:");
            for r in &results {
                println!(
                    "  sim={:.3} contexts={:?}",
                    r.similarity,
                    r.dn_contexts.first()
                );
            }
        }
    }

    #[test]
    fn test_cell_dn_metadata() {
        let mut crystal = SentenceCrystal::new(None);

        // Store multiple entries at different rungs
        crystal.store_with_dn_context("first thought", "Ada:A:thoughts:1", None, 2, 3);
        crystal.store_with_dn_context("second thought", "Ada:A:thoughts:2", None, 5, 3);
        crystal.store_with_dn_context("third thought", "Ada:A:thoughts:3", None, 3, 3);

        // Check max_rung is tracked
        let cells = crystal.cells_in_subtree("Ada:A:thoughts");
        for (_, cell) in cells {
            if cell.count > 0 {
                assert!(cell.max_rung <= 5, "Max rung should be <=5");
                println!(
                    "Cell has {} entries, max_rung={}, avg_depth={:.1}",
                    cell.count, cell.max_rung, cell.avg_depth
                );
            }
        }
    }

    #[test]
    fn test_addr_index() {
        let mut crystal = SentenceCrystal::new(None);

        let test_path = "Test:path:for:index";
        crystal.store_with_dn_context("test content", test_path, None, 1, 4);

        // Verify addr_index works
        let addr = dn_path_to_addr(test_path);
        let cell = crystal.cell_from_addr(addr);
        assert!(cell.is_some(), "Should find cell via addr");
        assert_eq!(cell.unwrap().count, 1, "Cell should have 1 entry");
    }
}
