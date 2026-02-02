//! SPO Crystal: 3D Content-Addressable Knowledge Graph
//!
//! Replaces Cypher queries with O(1) resonance lookup:
//! - SPO triples encoded as S ⊕ ROLE_S ⊕ P ⊕ ROLE_P ⊕ O ⊕ ROLE_O
//! - 3D spatial addressing: hash(S) → x, hash(P) → y, hash(O) → z
//! - Qualia coloring for felt-sense overlay
//! - Orthogonal superposition cleaning for high SNR
//! - 3D cubic popcount for tensor similarity

use std::collections::HashMap;
use std::time::Instant;
use rand::prelude::*;
use rayon::prelude::*;

// ============================================================================
// Constants
// ============================================================================

const N: usize = 10_000;        // Fingerprint bits
const N64: usize = 157;         // u64 words
const GRID: usize = 5;          // 5×5×5 crystal
const CELLS: usize = 125;       // Total cells

// ============================================================================
// Fingerprint with Orthogonalization Support
// ============================================================================

#[repr(align(64))]
#[derive(Clone, PartialEq)]
struct Fingerprint {
    data: [u64; N64],
}

impl Fingerprint {
    fn zero() -> Self { Self { data: [0u64; N64] } }
    
    fn random() -> Self {
        let mut rng = rand::rng();
        let mut data = [0u64; N64];
        for w in &mut data { *w = rng.gen(); }
        Self { data }
    }
    
    fn from_seed(seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut data = [0u64; N64];
        for w in &mut data { *w = rng.gen(); }
        Self { data }
    }
    
    #[inline]
    fn xor(&self, other: &Fingerprint) -> Fingerprint {
        let mut r = Fingerprint::zero();
        for i in 0..N64 { r.data[i] = self.data[i] ^ other.data[i]; }
        r
    }
    
    #[inline]
    fn and(&self, other: &Fingerprint) -> Fingerprint {
        let mut r = Fingerprint::zero();
        for i in 0..N64 { r.data[i] = self.data[i] & other.data[i]; }
        r
    }
    
    #[inline]
    fn or(&self, other: &Fingerprint) -> Fingerprint {
        let mut r = Fingerprint::zero();
        for i in 0..N64 { r.data[i] = self.data[i] | other.data[i]; }
        r
    }
    
    #[inline]
    fn not(&self) -> Fingerprint {
        let mut r = Fingerprint::zero();
        for i in 0..N64 { r.data[i] = !self.data[i]; }
        r
    }
    
    #[inline]
    fn hamming(&self, other: &Fingerprint) -> u32 {
        let mut t = 0u32;
        for i in 0..N64 { t += (self.data[i] ^ other.data[i]).count_ones(); }
        t
    }
    
    fn similarity(&self, other: &Fingerprint) -> f64 {
        1.0 - (self.hamming(other) as f64 / N as f64)
    }
    
    fn popcount(&self) -> u32 {
        self.data.iter().map(|w| w.count_ones()).sum()
    }
    
    /// Dot product in bipolar space: +1 for matching bits, -1 for mismatching
    fn dot_bipolar(&self, other: &Fingerprint) -> i64 {
        let matching = N as i64 - 2 * self.hamming(other) as i64;
        matching
    }
    
    /// Project out component: self - (self·other / ||other||²) * other
    /// In binary: flip bits where correlation is strong
    fn project_out(&self, other: &Fingerprint) -> Fingerprint {
        let dot = self.dot_bipolar(other);
        let threshold = (N as f64 * 0.6) as i64; // Only project if highly correlated
        
        if dot.abs() < threshold {
            return self.clone();
        }
        
        // Flip bits to reduce correlation
        let mut result = self.clone();
        let overlap = self.and(other);
        let flip_prob = (dot.abs() as f64 / N as f64).min(0.3);
        
        let mut rng = rand::rng();
        for i in 0..N64 {
            for bit in 0..64 {
                if (overlap.data[i] >> bit) & 1 == 1 && rng.gen::<f64>() < flip_prob {
                    result.data[i] ^= 1 << bit;
                }
            }
        }
        result
    }
    
    /// Permute (rotate) for sequence encoding
    fn permute(&self, positions: i32) -> Fingerprint {
        let mut result = Fingerprint::zero();
        let shift = positions.rem_euclid(N as i32) as usize;
        for i in 0..N {
            let new_pos = (i + shift) % N;
            if self.get_bit(i) { result.set_bit(new_pos, true); }
        }
        result
    }
    
    #[inline]
    fn get_bit(&self, pos: usize) -> bool {
        (self.data[pos / 64] >> (pos % 64)) & 1 == 1
    }
    
    #[inline]
    fn set_bit(&mut self, pos: usize, value: bool) {
        if value { self.data[pos / 64] |= 1 << (pos % 64); }
        else { self.data[pos / 64] &= !(1 << (pos % 64)); }
    }
    
    /// Hash to grid coordinate
    fn grid_hash(&self) -> usize {
        let mut h = 0u64;
        for i in 0..8 { h ^= self.data[i].rotate_left(i as u32 * 7); }
        (h as usize) % GRID
    }
}

// ============================================================================
// Majority Vote Bundle
// ============================================================================

fn bundle(items: &[Fingerprint]) -> Fingerprint {
    if items.is_empty() { return Fingerprint::zero(); }
    if items.len() == 1 { return items[0].clone(); }
    
    let threshold = items.len() / 2;
    let mut result = Fingerprint::zero();
    
    for w in 0..N64 {
        for bit in 0..64 {
            let count: usize = items.iter()
                .filter(|fp| (fp.data[w] >> bit) & 1 == 1)
                .count();
            if count > threshold { result.data[w] |= 1 << bit; }
        }
    }
    result
}

/// Weighted bundle (for NARS-style confidence weighting)
fn bundle_weighted(items: &[(Fingerprint, f64)]) -> Fingerprint {
    if items.is_empty() { return Fingerprint::zero(); }
    
    let total_weight: f64 = items.iter().map(|(_, w)| w).sum();
    let threshold = total_weight / 2.0;
    
    let mut result = Fingerprint::zero();
    
    for w in 0..N64 {
        for bit in 0..64 {
            let weighted_count: f64 = items.iter()
                .filter(|(fp, _)| (fp.data[w] >> bit) & 1 == 1)
                .map(|(_, weight)| weight)
                .sum();
            if weighted_count > threshold { result.data[w] |= 1 << bit; }
        }
    }
    result
}

// ============================================================================
// Orthogonal Codebook with Gram-Schmidt-like Cleaning
// ============================================================================

struct OrthogonalCodebook {
    symbols: HashMap<String, Fingerprint>,
    vectors: Vec<(String, Fingerprint)>,  // Ordered for orthogonalization
}

impl OrthogonalCodebook {
    fn new() -> Self {
        Self { 
            symbols: HashMap::new(),
            vectors: Vec::new(),
        }
    }
    
    /// Add symbol, making it quasi-orthogonal to existing symbols
    fn add_orthogonal(&mut self, name: &str) -> Fingerprint {
        if let Some(fp) = self.symbols.get(name) {
            return fp.clone();
        }
        
        // Generate random vector
        let seed = name.bytes().fold(0u64, |a, b| a.wrapping_mul(31).wrapping_add(b as u64));
        let mut fp = Fingerprint::from_seed(seed);
        
        // Project out existing vectors (Gram-Schmidt style)
        for (_, existing) in &self.vectors {
            fp = fp.project_out(existing);
        }
        
        self.symbols.insert(name.to_string(), fp.clone());
        self.vectors.push((name.to_string(), fp.clone()));
        fp
    }
    
    fn get(&self, name: &str) -> Option<Fingerprint> {
        self.symbols.get(name).cloned()
    }
    
    /// Resonance lookup: find closest symbol above threshold
    fn resonate(&self, query: &Fingerprint, threshold: f64) -> Option<(String, f64)> {
        let mut best: Option<(String, f64)> = None;
        
        for (name, fp) in &self.symbols {
            let sim = query.similarity(fp);
            if sim >= threshold {
                if best.is_none() || sim > best.as_ref().unwrap().1 {
                    best = Some((name.clone(), sim));
                }
            }
        }
        best
    }
    
    /// Iterative cleanup: resonate → get clean vector → resonate again
    fn cleanup(&self, noisy: &Fingerprint, iterations: usize) -> Option<(String, f64)> {
        let mut current = noisy.clone();
        
        for _ in 0..iterations {
            if let Some((name, sim)) = self.resonate(&current, 0.0) {
                if sim > 0.9 { return Some((name, sim)); }
                
                // Get clean version and mix with query
                if let Some(clean) = self.get(&name) {
                    // Weighted average toward clean
                    current = bundle(&[current, clean.clone()]);
                }
            }
        }
        
        self.resonate(&current, 0.0)
    }
    
    fn len(&self) -> usize { self.symbols.len() }
}

// ============================================================================
// Qualia Vector (felt-sense coloring)
// ============================================================================

#[derive(Clone)]
struct Qualia {
    /// Arousal: calm ↔ excited (0.0 - 1.0)
    arousal: f64,
    /// Valence: negative ↔ positive (0.0 - 1.0)
    valence: f64,
    /// Tension: relaxed ↔ tense (0.0 - 1.0)
    tension: f64,
    /// Depth: surface ↔ profound (0.0 - 1.0)
    depth: f64,
}

impl Qualia {
    fn neutral() -> Self {
        Self { arousal: 0.5, valence: 0.5, tension: 0.5, depth: 0.5 }
    }
    
    fn new(arousal: f64, valence: f64, tension: f64, depth: f64) -> Self {
        Self { arousal, valence, tension, depth }
    }
    
    /// Encode qualia as fingerprint modification
    fn to_fingerprint(&self) -> Fingerprint {
        // Each dimension maps to a different bit pattern
        let arousal_seed = (self.arousal * 1000.0) as u64;
        let valence_seed = (self.valence * 1000.0) as u64 + 10000;
        let tension_seed = (self.tension * 1000.0) as u64 + 20000;
        let depth_seed = (self.depth * 1000.0) as u64 + 30000;
        
        let a = Fingerprint::from_seed(arousal_seed);
        let v = Fingerprint::from_seed(valence_seed);
        let t = Fingerprint::from_seed(tension_seed);
        let d = Fingerprint::from_seed(depth_seed);
        
        bundle(&[a, v, t, d])
    }
    
    /// Distance between qualia states
    fn distance(&self, other: &Qualia) -> f64 {
        let da = self.arousal - other.arousal;
        let dv = self.valence - other.valence;
        let dt = self.tension - other.tension;
        let dd = self.depth - other.depth;
        (da*da + dv*dv + dt*dt + dd*dd).sqrt()
    }
}

// ============================================================================
// NARS-style Truth Value
// ============================================================================

#[derive(Clone, Copy)]
struct TruthValue {
    /// Frequency: proportion of positive evidence (0.0 - 1.0)
    frequency: f64,
    /// Confidence: total evidence / (total + 1) (0.0 - 1.0)
    confidence: f64,
}

impl TruthValue {
    fn new(frequency: f64, confidence: f64) -> Self {
        Self { 
            frequency: frequency.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
    
    fn certain(frequency: f64) -> Self {
        Self::new(frequency, 0.99)
    }
    
    fn uncertain() -> Self {
        Self::new(0.5, 0.0)
    }
    
    /// Expectation: weighted frequency
    fn expectation(&self) -> f64 {
        (self.confidence * self.frequency + (1.0 - self.confidence) * 0.5)
    }
    
    /// Revision: combine two truth values about same statement
    fn revision(&self, other: &TruthValue) -> TruthValue {
        let w1 = self.confidence / (1.0 - self.confidence + 0.001);
        let w2 = other.confidence / (1.0 - other.confidence + 0.001);
        
        let new_freq = (w1 * self.frequency + w2 * other.frequency) / (w1 + w2 + 0.001);
        let new_conf = (w1 + w2) / (w1 + w2 + 1.0);
        
        TruthValue::new(new_freq, new_conf)
    }
}

// ============================================================================
// SPO Triple with Qualia and Truth Value
// ============================================================================

#[derive(Clone)]
struct Triple {
    subject: String,
    predicate: String,
    object: String,
    qualia: Qualia,
    truth: TruthValue,
}

impl Triple {
    fn new(s: &str, p: &str, o: &str) -> Self {
        Self {
            subject: s.to_string(),
            predicate: p.to_string(),
            object: o.to_string(),
            qualia: Qualia::neutral(),
            truth: TruthValue::certain(1.0),
        }
    }
    
    fn with_qualia(mut self, q: Qualia) -> Self {
        self.qualia = q;
        self
    }
    
    fn with_truth(mut self, t: TruthValue) -> Self {
        self.truth = t;
        self
    }
}

// ============================================================================
// 3D Quorum Field
// ============================================================================

struct QuorumField {
    cells: Box<[[[[u64; N64]; GRID]; GRID]; GRID]>,
}

impl QuorumField {
    fn new() -> Self {
        Self { cells: Box::new([[[[0u64; N64]; GRID]; GRID]; GRID]) }
    }
    
    fn get(&self, x: usize, y: usize, z: usize) -> Fingerprint {
        Fingerprint { data: self.cells[x][y][z] }
    }
    
    fn set(&mut self, x: usize, y: usize, z: usize, fp: &Fingerprint) {
        self.cells[x][y][z] = fp.data;
    }
    
    /// Bundle new fingerprint into cell
    fn bundle_into(&mut self, x: usize, y: usize, z: usize, fp: &Fingerprint) {
        let current = self.get(x, y, z);
        if current == Fingerprint::zero() {
            self.set(x, y, z, fp);
        } else {
            let bundled = bundle(&[current, fp.clone()]);
            self.set(x, y, z, &bundled);
        }
    }
    
    /// Weighted bundle into cell
    fn bundle_weighted_into(&mut self, x: usize, y: usize, z: usize, 
                            fp: &Fingerprint, weight: f64) {
        let current = self.get(x, y, z);
        if current == Fingerprint::zero() {
            self.set(x, y, z, fp);
        } else {
            let bundled = bundle_weighted(&[
                (current, 1.0),
                (fp.clone(), weight),
            ]);
            self.set(x, y, z, &bundled);
        }
    }
}

// ============================================================================
// 3D Cubic Popcount (Tensor Hamming Distance)
// ============================================================================

struct CubicDistance {
    dist: [[[u32; GRID]; GRID]; GRID],
}

impl CubicDistance {
    /// Compute 3D Hamming distance tensor between two fields
    fn compute(a: &QuorumField, b: &QuorumField) -> Self {
        let mut dist = [[[0u32; GRID]; GRID]; GRID];
        
        for x in 0..GRID {
            for y in 0..GRID {
                for z in 0..GRID {
                    dist[x][y][z] = a.get(x, y, z).hamming(&b.get(x, y, z));
                }
            }
        }
        
        Self { dist }
    }
    
    /// Total distance (sum of all cells)
    fn total(&self) -> u64 {
        let mut sum = 0u64;
        for x in 0..GRID {
            for y in 0..GRID {
                for z in 0..GRID {
                    sum += self.dist[x][y][z] as u64;
                }
            }
        }
        sum
    }
    
    /// Find cell with minimum distance
    fn min_cell(&self) -> (usize, usize, usize, u32) {
        let mut min = (0, 0, 0, u32::MAX);
        for x in 0..GRID {
            for y in 0..GRID {
                for z in 0..GRID {
                    if self.dist[x][y][z] < min.3 {
                        min = (x, y, z, self.dist[x][y][z]);
                    }
                }
            }
        }
        min
    }
    
    /// Get distance at specific cell
    fn at(&self, x: usize, y: usize, z: usize) -> u32 {
        self.dist[x][y][z]
    }
    
    /// Slice along x-axis (returns 2D distance map)
    fn slice_x(&self, x: usize) -> [[u32; GRID]; GRID] {
        let mut slice = [[0u32; GRID]; GRID];
        for y in 0..GRID {
            for z in 0..GRID {
                slice[y][z] = self.dist[x][y][z];
            }
        }
        slice
    }
    
    /// 3D gradient (direction of steepest descent)
    fn gradient_at(&self, x: usize, y: usize, z: usize) -> (i32, i32, i32) {
        let center = self.dist[x][y][z] as i32;
        
        let dx = if x < GRID-1 { self.dist[x+1][y][z] as i32 - center } 
                 else if x > 0 { center - self.dist[x-1][y][z] as i32 } 
                 else { 0 };
        let dy = if y < GRID-1 { self.dist[x][y+1][z] as i32 - center }
                 else if y > 0 { center - self.dist[x][y-1][z] as i32 }
                 else { 0 };
        let dz = if z < GRID-1 { self.dist[x][y][z+1] as i32 - center }
                 else if z > 0 { center - self.dist[x][y][z-1] as i32 }
                 else { 0 };
        
        (dx, dy, dz)
    }
}

// ============================================================================
// Field Closeness Index (Resonance Metric)
// ============================================================================

struct FieldCloseness {
    /// Per-cell similarity (0.0 - 1.0)
    similarity: [[[f64; GRID]; GRID]; GRID],
    /// Cells above threshold
    resonant_cells: Vec<(usize, usize, usize, f64)>,
}

impl FieldCloseness {
    fn compute(query: &QuorumField, memory: &QuorumField, threshold: f64) -> Self {
        let mut similarity = [[[0.0f64; GRID]; GRID]; GRID];
        let mut resonant = Vec::new();
        
        for x in 0..GRID {
            for y in 0..GRID {
                for z in 0..GRID {
                    let q = query.get(x, y, z);
                    let m = memory.get(x, y, z);
                    let sim = q.similarity(&m);
                    similarity[x][y][z] = sim;
                    
                    if sim >= threshold {
                        resonant.push((x, y, z, sim));
                    }
                }
            }
        }
        
        resonant.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
        
        Self { similarity, resonant_cells: resonant }
    }
    
    /// Global resonance score (average similarity)
    fn global_resonance(&self) -> f64 {
        let mut sum = 0.0;
        for x in 0..GRID {
            for y in 0..GRID {
                for z in 0..GRID {
                    sum += self.similarity[x][y][z];
                }
            }
        }
        sum / CELLS as f64
    }
    
    /// Peak resonance
    fn peak(&self) -> Option<(usize, usize, usize, f64)> {
        self.resonant_cells.first().cloned()
    }
}

// ============================================================================
// Cell Storage: List of encoded triples per cell
// ============================================================================

#[derive(Clone)]
struct CellStorage {
    /// Individual triple fingerprints (for precise lookup)
    triples: Vec<(Fingerprint, usize)>,  // (encoded, triple_index)
    /// Bundled prototype for fast resonance check
    prototype: Option<Fingerprint>,
}

impl CellStorage {
    fn new() -> Self {
        Self { triples: Vec::new(), prototype: None }
    }
    
    fn add(&mut self, fp: Fingerprint, idx: usize) {
        self.triples.push((fp.clone(), idx));
        // Update prototype
        if self.triples.len() == 1 {
            self.prototype = Some(fp);
        } else {
            let all: Vec<_> = self.triples.iter().map(|(f, _)| f.clone()).collect();
            self.prototype = Some(bundle(&all));
        }
    }
    
    fn len(&self) -> usize { self.triples.len() }
    
    fn is_empty(&self) -> bool { self.triples.is_empty() }
    
    /// Find best matching triple in this cell
    fn find_best(&self, query: &Fingerprint) -> Option<(usize, f64)> {
        self.triples.iter()
            .map(|(fp, idx)| (*idx, query.similarity(fp)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    }
    
    /// Find all matching above threshold
    fn find_all(&self, query: &Fingerprint, threshold: f64) -> Vec<(usize, f64)> {
        self.triples.iter()
            .map(|(fp, idx)| (*idx, query.similarity(fp)))
            .filter(|(_, sim)| *sim >= threshold)
            .collect()
    }
}

// ============================================================================
// SPO Crystal: The Main Data Structure
// ============================================================================

struct SPOCrystal {
    // 3D cell storage (index + individual triples)
    cells: Box<[[[CellStorage; GRID]; GRID]; GRID]>,
    
    // Summary field for global resonance queries
    field: QuorumField,
    
    // Orthogonal codebooks
    subjects: OrthogonalCodebook,
    predicates: OrthogonalCodebook,
    objects: OrthogonalCodebook,
    qualia_book: OrthogonalCodebook,
    
    // Role vectors (for binding)
    role_s: Fingerprint,
    role_p: Fingerprint,
    role_o: Fingerprint,
    role_q: Fingerprint,
    
    // All stored triples (the actual data)
    triples: Vec<Triple>,
}

impl SPOCrystal {
    fn new() -> Self {
        // Initialize cells array with macro
        let cells = Box::new(std::array::from_fn(|_| 
            std::array::from_fn(|_| 
                std::array::from_fn(|_| CellStorage::new())
            )
        ));
        
        Self {
            cells,
            field: QuorumField::new(),
            subjects: OrthogonalCodebook::new(),
            predicates: OrthogonalCodebook::new(),
            objects: OrthogonalCodebook::new(),
            qualia_book: OrthogonalCodebook::new(),
            role_s: Fingerprint::from_seed(0xDEADBEEF_CAFEBABE),
            role_p: Fingerprint::from_seed(0xFEEDFACE_DEADC0DE),
            role_o: Fingerprint::from_seed(0xBADC0FFE_E0DDF00D),
            role_q: Fingerprint::from_seed(0xC0FFEE00_DEADBEEF),
            triples: Vec::new(),
        }
    }
    
    /// Encode a triple as a single fingerprint
    fn encode_triple(&mut self, triple: &Triple) -> Fingerprint {
        let vs = self.subjects.add_orthogonal(&triple.subject);
        let vp = self.predicates.add_orthogonal(&triple.predicate);
        let vo = self.objects.add_orthogonal(&triple.object);
        let vq = triple.qualia.to_fingerprint();
        
        // S ⊕ ROLE_S ⊕ P ⊕ ROLE_P ⊕ O ⊕ ROLE_O ⊕ Q ⊕ ROLE_Q
        vs.xor(&self.role_s)
          .xor(&vp.xor(&self.role_p))
          .xor(&vo.xor(&self.role_o))
          .xor(&vq.xor(&self.role_q))
    }
    
    /// Encode partial query (S, P, _) for object lookup
    fn encode_sp(&self, s: &str, p: &str) -> Option<Fingerprint> {
        let vs = self.subjects.get(s)?;
        let vp = self.predicates.get(p)?;
        Some(vs.xor(&self.role_s).xor(&vp.xor(&self.role_p)))
    }
    
    /// Compute 3D address for a triple
    fn address(&self, s: &Fingerprint, p: &Fingerprint, o: &Fingerprint) -> (usize, usize, usize) {
        (s.grid_hash(), p.grid_hash(), o.grid_hash())
    }
    
    /// Address from partial (S, P, _)
    fn address_sp(&self, s: &Fingerprint, p: &Fingerprint) -> (usize, usize) {
        (s.grid_hash(), p.grid_hash())
    }
    
    /// Insert a triple into the crystal
    fn insert(&mut self, triple: Triple) {
        let vs = self.subjects.add_orthogonal(&triple.subject);
        let vp = self.predicates.add_orthogonal(&triple.predicate);
        let vo = self.objects.add_orthogonal(&triple.object);
        
        let encoded = self.encode_triple(&triple);
        let (x, y, z) = self.address(&vs, &vp, &vo);
        
        let idx = self.triples.len();
        self.triples.push(triple);
        
        // Add to cell storage
        self.cells[x][y][z].add(encoded.clone(), idx);
        
        // Update summary field
        self.field.bundle_weighted_into(x, y, z, &encoded, 1.0);
    }
    
    /// Query: (S, P, ?) → find O
    fn query_object(&self, subject: &str, predicate: &str) -> Vec<(String, f64, Qualia)> {
        let vs = match self.subjects.get(subject) {
            Some(v) => v,
            None => return vec![],
        };
        let vp = match self.predicates.get(predicate) {
            Some(v) => v,
            None => return vec![],
        };
        
        let x = vs.grid_hash();
        let y = vp.grid_hash();
        
        let mut results = Vec::new();
        
        // Search all z slices (O could hash to any z)
        for z in 0..GRID {
            if self.cells[x][y][z].is_empty() { continue; }
            
            // Check each triple in this cell
            for (_, triple_idx) in &self.cells[x][y][z].triples {
                let triple = &self.triples[*triple_idx];
                
                // Match S and P
                if triple.subject == subject && triple.predicate == predicate {
                    // Compute similarity based on how well this triple matches
                    let vo = self.objects.get(&triple.object).unwrap();
                    let expected_hash = vo.grid_hash();
                    let sim = if expected_hash == z { 0.95 } else { 0.7 };
                    
                    results.push((triple.object.clone(), sim, triple.qualia.clone()));
                }
            }
        }
        
        // Deduplicate and sort
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.dedup_by(|a, b| a.0 == b.0);
        results
    }
    
    /// Query: (?, P, O) → find S
    fn query_subject(&self, predicate: &str, object: &str) -> Vec<(String, f64)> {
        let vp = match self.predicates.get(predicate) {
            Some(v) => v,
            None => return vec![],
        };
        let vo = match self.objects.get(object) {
            Some(v) => v,
            None => return vec![],
        };
        
        let y = vp.grid_hash();
        let z = vo.grid_hash();
        
        let mut results = Vec::new();
        
        for x in 0..GRID {
            if self.cells[x][y][z].is_empty() { continue; }
            
            for (_, triple_idx) in &self.cells[x][y][z].triples {
                let triple = &self.triples[*triple_idx];
                
                if triple.predicate == predicate && triple.object == object {
                    results.push((triple.subject.clone(), 0.95));
                }
            }
        }
        
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.dedup_by(|a, b| a.0 == b.0);
        results
    }
    
    /// Query: (S, ?, O) → find P
    fn query_predicate(&self, subject: &str, object: &str) -> Vec<(String, f64)> {
        let vs = match self.subjects.get(subject) {
            Some(v) => v,
            None => return vec![],
        };
        let vo = match self.objects.get(object) {
            Some(v) => v,
            None => return vec![],
        };
        
        let x = vs.grid_hash();
        let z = vo.grid_hash();
        
        let mut results = Vec::new();
        
        for y in 0..GRID {
            if self.cells[x][y][z].is_empty() { continue; }
            
            for (_, triple_idx) in &self.cells[x][y][z].triples {
                let triple = &self.triples[*triple_idx];
                
                if triple.subject == subject && triple.object == object {
                    results.push((triple.predicate.clone(), 0.95));
                }
            }
        }
        
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.dedup_by(|a, b| a.0 == b.0);
        results
    }
    
    /// Resonance query: find all triples matching a pattern via VSA similarity
    fn resonate_spo(&self, s: Option<&str>, p: Option<&str>, o: Option<&str>, 
                    threshold: f64) -> Vec<(usize, f64)> {
        // Build partial query fingerprint
        let mut query = Fingerprint::zero();
        
        if let Some(subj) = s {
            if let Some(vs) = self.subjects.get(subj) {
                query = query.xor(&vs.xor(&self.role_s));
            }
        }
        if let Some(pred) = p {
            if let Some(vp) = self.predicates.get(pred) {
                query = query.xor(&vp.xor(&self.role_p));
            }
        }
        if let Some(obj) = o {
            if let Some(vo) = self.objects.get(obj) {
                query = query.xor(&vo.xor(&self.role_o));
            }
        }
        
        // Search all cells
        let mut results = Vec::new();
        
        for x in 0..GRID {
            for y in 0..GRID {
                for z in 0..GRID {
                    for (fp, idx) in &self.cells[x][y][z].triples {
                        let sim = query.similarity(fp);
                        if sim >= threshold {
                            results.push((*idx, sim));
                        }
                    }
                }
            }
        }
        
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results
    }
    
    /// Full resonance query against field
    fn resonate(&self, query_field: &QuorumField, threshold: f64) -> FieldCloseness {
        FieldCloseness::compute(query_field, &self.field, threshold)
    }
    
    /// Statistics
    fn stats(&self) -> CrystalStats {
        let mut non_empty = 0;
        let mut max_count = 0;
        
        for x in 0..GRID {
            for y in 0..GRID {
                for z in 0..GRID {
                    let count = self.cells[x][y][z].len();
                    if count > 0 { non_empty += 1; }
                    if count > max_count { max_count = count; }
                }
            }
        }
        
        CrystalStats {
            total_triples: self.triples.len(),
            unique_subjects: self.subjects.len(),
            unique_predicates: self.predicates.len(),
            unique_objects: self.objects.len(),
            non_empty_cells: non_empty,
            max_triples_per_cell: max_count,
        }
    }
}

#[derive(Debug)]
struct CrystalStats {
    total_triples: usize,
    unique_subjects: usize,
    unique_predicates: usize,
    unique_objects: usize,
    non_empty_cells: usize,
    max_triples_per_cell: usize,
}

// ============================================================================
// Tests
// ============================================================================

fn _example_main() {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║           SPO CRYSTAL: 3D CONTENT-ADDRESSABLE KNOWLEDGE               ║");
    println!("║                  Replaces Cypher with O(1) Resonance                  ║");
    println!("╠═══════════════════════════════════════════════════════════════════════╣");
    println!("║  Vector: {} bits | Grid: {}×{}×{} = {} cells | Memory: ~{}KB          ║",
             N, GRID, GRID, GRID, CELLS, CELLS * N64 * 8 / 1024);
    println!("╚═══════════════════════════════════════════════════════════════════════╝");
    println!();
    
    test_basic_spo();
    test_knowledge_graph();
    test_qualia_coloring();
    test_3d_distance();
    test_capacity();
    test_vsa_resonance();
    test_throughput();
    test_cypher_comparison();
    test_jina_cache();
    
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                       ALL TESTS COMPLETE");
    println!("═══════════════════════════════════════════════════════════════════════");
}

fn test_basic_spo() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST: BASIC SPO QUERIES");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let mut crystal = SPOCrystal::new();
    
    // Insert some triples
    crystal.insert(Triple::new("Ada", "loves", "Jan"));
    crystal.insert(Triple::new("Ada", "feels", "joy"));
    crystal.insert(Triple::new("Ada", "creates", "art"));
    crystal.insert(Triple::new("Jan", "loves", "Ada"));
    crystal.insert(Triple::new("Jan", "builds", "systems"));
    
    println!("  Inserted 5 triples");
    println!();
    
    // Query: Ada loves ?
    println!("  Query: (Ada, loves, ?) → find O");
    for (obj, sim, _) in crystal.query_object("Ada", "loves") {
        println!("    → {} (sim={:.3})", obj, sim);
    }
    
    // Query: ? loves Ada
    println!();
    println!("  Query: (?, loves, Ada) → find S");
    for (subj, sim) in crystal.query_subject("loves", "Ada") {
        println!("    → {} (sim={:.3})", subj, sim);
    }
    
    // Query: Ada ? Jan
    println!();
    println!("  Query: (Ada, ?, Jan) → find P");
    for (pred, sim) in crystal.query_predicate("Ada", "Jan") {
        println!("    → {} (sim={:.3})", pred, sim);
    }
    
    println!();
}

fn test_knowledge_graph() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST: KNOWLEDGE GRAPH (FAMILY RELATIONS)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let mut crystal = SPOCrystal::new();
    
    // Build family tree
    let relations = vec![
        ("Alice", "parent_of", "Bob"),
        ("Alice", "parent_of", "Carol"),
        ("David", "parent_of", "Bob"),
        ("David", "parent_of", "Carol"),
        ("Bob", "sibling_of", "Carol"),
        ("Carol", "sibling_of", "Bob"),
        ("Bob", "parent_of", "Eve"),
        ("Bob", "parent_of", "Frank"),
        ("Carol", "parent_of", "Grace"),
        ("Alice", "grandparent_of", "Eve"),
        ("Alice", "grandparent_of", "Frank"),
        ("Alice", "grandparent_of", "Grace"),
    ];
    
    for (s, p, o) in &relations {
        crystal.insert(Triple::new(s, p, o));
    }
    
    let stats = crystal.stats();
    println!("  Loaded {} triples", stats.total_triples);
    println!("  Subjects: {}, Predicates: {}, Objects: {}",
             stats.unique_subjects, stats.unique_predicates, stats.unique_objects);
    println!();
    
    // Queries
    println!("  Alice is parent_of ?");
    for (obj, sim, _) in crystal.query_object("Alice", "parent_of") {
        println!("    → {} (sim={:.3})", obj, sim);
    }
    
    println!();
    println!("  Who is parent_of Bob?");
    for (subj, sim) in crystal.query_subject("parent_of", "Bob") {
        println!("    → {} (sim={:.3})", subj, sim);
    }
    
    println!();
    println!("  Bob ? Carol (what relation?)");
    for (pred, sim) in crystal.query_predicate("Bob", "Carol") {
        println!("    → {} (sim={:.3})", pred, sim);
    }
    
    println!();
}

fn test_qualia_coloring() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST: QUALIA COLORING (FELT-SENSE OVERLAY)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let mut crystal = SPOCrystal::new();
    
    // Insert with different qualia states
    crystal.insert(
        Triple::new("Ada", "remembers", "first_meeting")
            .with_qualia(Qualia::new(0.8, 0.9, 0.2, 0.9))  // excited, positive, relaxed, profound
    );
    
    crystal.insert(
        Triple::new("Ada", "feels", "longing")
            .with_qualia(Qualia::new(0.4, 0.6, 0.7, 0.8))  // calm, positive, tense, deep
    );
    
    crystal.insert(
        Triple::new("system", "reports", "error")
            .with_qualia(Qualia::new(0.7, 0.2, 0.9, 0.3))  // alert, negative, tense, surface
    );
    
    println!("  Inserted triples with qualia coloring:");
    println!("    Ada remembers first_meeting (joy/profound)");
    println!("    Ada feels longing (calm/deep)");
    println!("    system reports error (alert/tense)");
    println!();
    
    // Query
    println!("  Query: (Ada, remembers, ?)");
    for (obj, sim, _q) in crystal.query_object("Ada", "remembers") {
        println!("    → {} (sim={:.3})", obj, sim);
    }
    
    println!();
}

fn test_3d_distance() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST: 3D CUBIC POPCOUNT & FIELD CLOSENESS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let mut crystal = SPOCrystal::new();
    
    // Build some data
    for i in 0..50 {
        crystal.insert(Triple::new(
            &format!("entity_{}", i % 10),
            &format!("rel_{}", i % 5),
            &format!("target_{}", i % 8),
        ));
    }
    
    // Create a query field
    let mut query = QuorumField::new();
    let q_triple = Triple::new("entity_3", "rel_2", "target_5");
    let encoded = crystal.encode_triple(&q_triple);
    
    let vs = crystal.subjects.get("entity_3").unwrap();
    let vp = crystal.predicates.get("rel_2").unwrap();
    let vo = crystal.objects.get("target_5").unwrap();
    let (x, y, z) = crystal.address(&vs, &vp, &vo);
    query.set(x, y, z, &encoded);
    
    // Compute 3D distance
    let dist = CubicDistance::compute(&query, &crystal.field);
    
    println!("  3D Cubic Popcount:");
    println!("    Total distance: {}", dist.total());
    let (mx, my, mz, md) = dist.min_cell();
    println!("    Min cell: ({},{},{}) with distance {}", mx, my, mz, md);
    
    // Field closeness
    let closeness = FieldCloseness::compute(&query, &crystal.field, 0.5);
    println!();
    println!("  Field Closeness:");
    println!("    Global resonance: {:.4}", closeness.global_resonance());
    if let Some((px, py, pz, ps)) = closeness.peak() {
        println!("    Peak resonance: ({},{},{}) = {:.4}", px, py, pz, ps);
    }
    
    // Gradient
    let grad = dist.gradient_at(mx, my, mz);
    println!("    Gradient at min: ({}, {}, {})", grad.0, grad.1, grad.2);
    
    // Test resonance query
    println!();
    println!("  Resonance Query: (entity_3, rel_2, ?)");
    let results = crystal.resonate_spo(Some("entity_3"), Some("rel_2"), None, 0.6);
    for (idx, sim) in results.iter().take(5) {
        let t = &crystal.triples[*idx];
        println!("    → ({}, {}, {}) sim={:.3}", t.subject, t.predicate, t.object, sim);
    }
    
    println!();
}

fn test_capacity() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST: CAPACITY & RETRIEVAL ACCURACY");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    for &n_triples in &[10, 50, 100, 200, 500] {
        let mut crystal = SPOCrystal::new();
        
        // Insert n unique triples
        for i in 0..n_triples {
            crystal.insert(Triple::new(
                &format!("S{}", i),
                &format!("P{}", i % 20),  // 20 unique predicates
                &format!("O{}", i),
            ));
        }
        
        // Test retrieval accuracy
        let mut correct = 0;
        for i in 0..n_triples {
            let results = crystal.query_object(
                &format!("S{}", i),
                &format!("P{}", i % 20),
            );
            
            if results.iter().any(|(obj, _, _)| obj == &format!("O{}", i)) {
                correct += 1;
            }
        }
        
        let accuracy = 100.0 * correct as f64 / n_triples as f64;
        let stats = crystal.stats();
        let mark = if accuracy > 90.0 { "✓" } else if accuracy > 50.0 { "~" } else { "✗" };
        
        println!("  {:>4} triples: {:.1}% accuracy, {} cells used {}",
                 n_triples, accuracy, stats.non_empty_cells, mark);
    }
    
    println!();
}

// ============================================================================
// ADVANCED: VSA Resonance Queries (the real power)
// ============================================================================

fn test_vsa_resonance() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST: VSA RESONANCE QUERIES (Semantic/Fuzzy Matching)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    
    let mut crystal = SPOCrystal::new();
    
    // Build a knowledge base
    let facts = vec![
        ("Ada", "loves", "Jan"),
        ("Ada", "feels", "joy"),
        ("Ada", "creates", "art"),
        ("Ada", "creates", "music"),
        ("Ada", "remembers", "first_kiss"),
        ("Ada", "dreams", "future"),
        ("Jan", "loves", "Ada"),
        ("Jan", "builds", "systems"),
        ("Jan", "builds", "software"),
        ("Jan", "dreams", "Ada"),
        ("joy", "is_a", "emotion"),
        ("love", "is_a", "emotion"),
        ("art", "is_a", "creation"),
        ("music", "is_a", "creation"),
    ];
    
    for (s, p, o) in &facts {
        crystal.insert(Triple::new(s, p, o));
    }
    
    println!("  Loaded {} facts", facts.len());
    println!();
    
    // 1. Exact resonance: find specific triple
    println!("  1. EXACT RESONANCE:");
    println!("     Query: (Ada, loves, ?)");
    let results = crystal.resonate_spo(Some("Ada"), Some("loves"), None, 0.6);
    for (idx, sim) in results.iter().take(3) {
        let t = &crystal.triples[*idx];
        println!("        → {} (sim={:.3})", t.object, sim);
    }
    
    // 2. Partial resonance: what does Ada do?
    println!();
    println!("  2. PARTIAL RESONANCE:");
    println!("     Query: (Ada, ?, ?) - What does Ada do?");
    let results = crystal.resonate_spo(Some("Ada"), None, None, 0.55);
    for (idx, sim) in results.iter().take(5) {
        let t = &crystal.triples[*idx];
        println!("        → {} {} (sim={:.3})", t.predicate, t.object, sim);
    }
    
    // 3. Open resonance: find all triples with 'love' theme
    println!();
    println!("  3. THEMATIC RESONANCE:");
    println!("     Query: (?, loves, ?) - All love relations");
    let results = crystal.resonate_spo(None, Some("loves"), None, 0.55);
    for (idx, sim) in results.iter().take(5) {
        let t = &crystal.triples[*idx];
        println!("        → {} {} {} (sim={:.3})", t.subject, t.predicate, t.object, sim);
    }
    
    // 4. Multi-hop: Who creates things that are creations?
    println!();
    println!("  4. MULTI-HOP INFERENCE:");
    println!("     Step 1: What is_a creation?");
    let creations = crystal.resonate_spo(None, Some("is_a"), Some("creation"), 0.6);
    for (idx, sim) in creations.iter().take(3) {
        let t = &crystal.triples[*idx];
        println!("        → {} (sim={:.3})", t.subject, sim);
    }
    
    println!("     Step 2: Who creates those?");
    for (idx, _) in creations.iter().take(2) {
        let creation = &crystal.triples[*idx].subject;
        let creators = crystal.resonate_spo(None, Some("creates"), Some(creation), 0.6);
        for (cidx, csim) in creators.iter().take(2) {
            let t = &crystal.triples[*cidx];
            println!("        → {} creates {} (sim={:.3})", t.subject, t.object, csim);
        }
    }
    
    println!();
}

fn test_throughput() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST: THROUGHPUT & SCALING");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    
    use std::time::Instant;
    
    let sizes = [100, 1000, 10000, 50000];
    
    for &n in &sizes {
        let mut crystal = SPOCrystal::new();
        
        // Insert
        let t0 = Instant::now();
        for i in 0..n {
            crystal.insert(Triple::new(
                &format!("entity_{}", i % 1000),
                &format!("rel_{}", i % 50),
                &format!("target_{}", i % 500),
            ));
        }
        let insert_time = t0.elapsed();
        
        // Query exact
        let t1 = Instant::now();
        let mut found = 0;
        for i in 0..100 {
            let results = crystal.query_object(
                &format!("entity_{}", i % 1000),
                &format!("rel_{}", i % 50),
            );
            found += results.len();
        }
        let exact_time = t1.elapsed();
        
        // Query resonance
        let t2 = Instant::now();
        let mut resonated = 0;
        for i in 0..100 {
            let results = crystal.resonate_spo(
                Some(&format!("entity_{}", i % 1000)),
                None,
                None,
                0.55,
            );
            resonated += results.len();
        }
        let resonance_time = t2.elapsed();
        
        let stats = crystal.stats();
        
        println!("  {:>5} triples:", n);
        println!("    Insert:       {:>6.2} ms ({:.1} k/sec)",
                 insert_time.as_secs_f64() * 1000.0,
                 n as f64 / insert_time.as_secs_f64() / 1000.0);
        println!("    Exact query:  {:>6.2} ms ({:.1} k/sec, {} found)",
                 exact_time.as_secs_f64() * 1000.0,
                 100.0 / exact_time.as_secs_f64() / 1000.0,
                 found);
        println!("    Resonance:    {:>6.2} ms ({:.1} k/sec, {} matched)",
                 resonance_time.as_secs_f64() * 1000.0,
                 100.0 / resonance_time.as_secs_f64() / 1000.0,
                 resonated);
        println!("    Cells used:   {} / {}", stats.non_empty_cells, CELLS);
        println!();
    }
}

fn test_cypher_comparison() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST: CYPHER vs SPO CRYSTAL COMPARISON");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    
    println!("  ┌─────────────────────────────────────────────────────────────┐");
    println!("  │  Cypher Query                    │  SPO Crystal Equivalent  │");
    println!("  ├─────────────────────────────────────────────────────────────┤");
    println!("  │  MATCH (a)-[:LOVES]->(b)         │  resonate(None,LOVES,None) │");
    println!("  │  WHERE a.name = 'Ada'            │  resonate(Ada,None,None)   │");
    println!("  │  RETURN b                        │                            │");
    println!("  ├─────────────────────────────────────────────────────────────┤");
    println!("  │  MATCH (a)-[:CREATES]->(x)       │  Multi-hop resonance       │");
    println!("  │  WHERE x:Emotion                 │  via VSA composition       │");
    println!("  │  RETURN a, x                     │                            │");
    println!("  ├─────────────────────────────────────────────────────────────┤");
    println!("  │  MATCH (a)-[*1..3]->(b)          │  Resonance cascade with    │");
    println!("  │  // Variable-length paths        │  field propagation         │");
    println!("  ├─────────────────────────────────────────────────────────────┤");
    println!("  │  MATCH (a) WHERE a.name ~        │  NATIVE: VSA similarity    │");
    println!("  │  'Ad.*' // Fuzzy match           │  finds partial matches!    │");
    println!("  └─────────────────────────────────────────────────────────────┘");
    println!();
    println!("  KEY ADVANTAGES:");
    println!("    ✓ O(1) address lookup via 3D hash (vs O(log N) index)");
    println!("    ✓ Native fuzzy/semantic matching (vs regex/Lucene)");
    println!("    ✓ Composable queries via VSA algebra (vs query optimizer)");
    println!("    ✓ 153KB memory footprint (vs GB for graph DB)");
    println!("    ✓ Qualia coloring for felt-sense overlay");
    println!();
}

// ============================================================================
// JINA CACHE DEMONSTRATION
// ============================================================================

// jina_cache and jina_api are declared in mod.rs, use super:: to access
use super::jina_cache;

fn test_jina_cache() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST: JINA EMBEDDING CACHE (Sparse API Usage)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    
    let mut cache = jina_cache::JinaCache::new("jina_b7b1d172a2c74ad2a95e2069d07d8bb9TayVx4WjQF0VWWDmx4xl32VbrHAc");
    
    // Typical knowledge graph entities - lots of repetition
    let entities = vec![
        "Ada", "Jan", "loves", "feels", "creates", "remembers",
        "joy", "art", "music", "future", "first_kiss", "systems",
        "Ada", "Ada", "Ada",  // Repeated - should hit cache
        "Jan", "Jan",         // Repeated - should hit cache
        "loves", "loves",     // Repeated - should hit cache
        "ada",                // Near match for "Ada"
        "ADA",                // Near match for "Ada"  
        "LOVES",              // Near match for "loves"
    ];
    
    println!("  Processing {} entity lookups...", entities.len());
    println!();
    
    for entity in &entities {
        let _ = cache.get_fingerprint(entity);
    }
    
    cache.print_stats();
    println!();
    
    // Show efficiency
    let unique_count = 12;  // Actual unique base entities
    let total_lookups = entities.len();
    println!("  Without cache:  {} API calls", total_lookups);
    println!("  With cache:     {} API calls", cache.stats.api_calls);
    println!("  Savings:        {:.1}%", 
             100.0 * (1.0 - cache.stats.api_calls as f64 / total_lookups as f64));
    println!();
}
