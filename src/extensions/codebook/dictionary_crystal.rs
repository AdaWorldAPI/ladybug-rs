//! Dictionary-Indexed Crystal: Correct Compression Architecture
//!
//! Key insight: Fingerprints ARE compression. Don't compress fingerprints.
//! Instead: store DICTIONARY + INDICES
//!
//! ┌──────────────────────────────────────────────────────────────────┐
//! │  ARCHITECTURE                                                    │
//! ├──────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  SYMBOL CODEBOOK (learned, fixed size)                          │
//! │  ┌─────┬─────┬─────┬─────┬─────┐                                │
//! │  │ S_0 │ S_1 │ S_2 │ ... │S_1023│  1024 × 10Kbit = 1.25MB      │
//! │  └─────┴─────┴─────┴─────┴─────┘                                │
//! │                                                                  │
//! │  ROLE CODEBOOK (fixed, orthogonal)                              │
//! │  ┌──────┬──────┬──────┬──────┐                                  │
//! │  │ROLE_S│ROLE_P│ROLE_O│ROLE_Q│  4 × 10Kbit = 5KB               │
//! │  └──────┴──────┴──────┴──────┘                                  │
//! │                                                                  │
//! │  CHUNK INDEX (sparse, tiny)                                     │
//! │  ┌────────────────────────────────────────┐                     │
//! │  │ chunk_0: (sym=42, role=S, cell=(2,3,1))│                     │
//! │  │ chunk_1: (sym=17, role=P, cell=(0,4,2))│  N × 16bit = 2N bytes│
//! │  │ ...                                    │                     │
//! │  └────────────────────────────────────────┘                     │
//! │                                                                  │
//! │  RECONSTRUCTION:                                                 │
//! │  FP(chunk_i) = CODEBOOK[sym_i] ⊕ ROLE[role_i]                   │
//! │                                                                  │
//! │  For 100K chunks: 1.25MB + 200KB = 1.45MB (vs 125MB raw)        │
//! │  Compression: 86x                                                │
//! └──────────────────────────────────────────────────────────────────┘

use std::collections::HashMap;

// Import from parent module
const N: usize = 10_000;
const N64: usize = 157;
const GRID: usize = 5;

const CODEBOOK_SIZE: usize = 1024;  // 2^10 symbols
const CODEBOOK_BITS: usize = 10;

#[repr(align(64))]
#[derive(Clone, PartialEq)]
pub struct Fingerprint {
    pub data: [u64; N64],
}

impl Fingerprint {
    pub fn zero() -> Self { Self { data: [0u64; N64] } }
    
    pub fn from_seed(seed: u64) -> Self {
        // LCG for deterministic generation
        let mut state = seed;
        let mut data = [0u64; N64];
        for w in &mut data {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
    pub fn xor(&self, other: &Fingerprint) -> Fingerprint {
        let mut r = Fingerprint::zero();
        for i in 0..N64 { r.data[i] = self.data[i] ^ other.data[i]; }
        r
    }
    
    #[inline]
    pub fn hamming(&self, other: &Fingerprint) -> u32 {
        let mut t = 0u32;
        for i in 0..N64 { t += (self.data[i] ^ other.data[i]).count_ones(); }
        t
    }
    
    pub fn similarity(&self, other: &Fingerprint) -> f64 {
        1.0 - (self.hamming(other) as f64 / N as f64)
    }
    
    pub fn to_xyz(&self) -> (usize, usize, usize) {
        let mut h = [0u64; 3];
        for i in 0..N64 { h[i % 3] ^= self.data[i].rotate_left((i * 7) as u32 % 64); }
        ((h[0] as usize) % GRID, (h[1] as usize) % GRID, (h[2] as usize) % GRID)
    }
}

/// Majority vote bundle
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

// ============================================================================
// Symbol Codebook: Learned dictionary of semantic patterns
// ============================================================================

pub struct SymbolCodebook {
    /// Codebook entries (quasi-orthogonal fingerprints)
    symbols: Vec<Fingerprint>,
    /// Reverse lookup: fingerprint hash → symbol index
    lookup: HashMap<u64, u16>,
}

impl SymbolCodebook {
    /// Create codebook with N quasi-orthogonal symbols
    pub fn new(size: usize) -> Self {
        let mut symbols = Vec::with_capacity(size);
        
        // Generate quasi-orthogonal fingerprints using prime-based seeds
        for i in 0..size {
            let seed = (i as u64).wrapping_mul(0x9E3779B97F4A7C15);  // Golden ratio
            symbols.push(Fingerprint::from_seed(seed));
        }
        
        Self { symbols, lookup: HashMap::new() }
    }
    
    /// Find closest symbol to fingerprint (or add if novel)
    pub fn encode(&mut self, fp: &Fingerprint, threshold: f64) -> u16 {
        // Quick hash lookup first
        let hash = Self::fp_hash(fp);
        if let Some(&idx) = self.lookup.get(&hash) {
            return idx;
        }
        
        // Linear search for similar (could use LSH for large codebooks)
        let mut best_idx = 0u16;
        let mut best_sim = 0.0f64;
        
        for (i, sym) in self.symbols.iter().enumerate() {
            let sim = fp.similarity(sym);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i as u16;
            }
        }
        
        // If similar enough, use existing symbol
        if best_sim >= threshold {
            self.lookup.insert(hash, best_idx);
            return best_idx;
        }
        
        // Otherwise, try to add new symbol (if space available)
        if self.symbols.len() < CODEBOOK_SIZE {
            let new_idx = self.symbols.len() as u16;
            self.symbols.push(fp.clone());
            self.lookup.insert(hash, new_idx);
            return new_idx;
        }
        
        // Codebook full, use best match
        self.lookup.insert(hash, best_idx);
        best_idx
    }
    
    /// Decode symbol index to fingerprint
    pub fn decode(&self, idx: u16) -> &Fingerprint {
        &self.symbols[idx as usize % self.symbols.len()]
    }
    
    /// Number of symbols in codebook
    pub fn len(&self) -> usize { self.symbols.len() }
    
    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.symbols.len() * N64 * 8
    }
    
    fn fp_hash(fp: &Fingerprint) -> u64 {
        let mut h = 0u64;
        for i in 0..8 { h ^= fp.data[i].rotate_left(i as u32 * 8); }
        h
    }
}

// ============================================================================
// Role Codebook: Fixed orthogonal role vectors
// ============================================================================

pub struct RoleCodebook {
    pub subject: Fingerprint,
    pub predicate: Fingerprint,
    pub object: Fingerprint,
    pub qualia: Fingerprint,
}

impl RoleCodebook {
    pub fn new() -> Self {
        // Fixed seeds for reproducibility
        Self {
            subject:   Fingerprint::from_seed(0xDEADBEEF_CAFEBABE),
            predicate: Fingerprint::from_seed(0xFEEDFACE_DEADBEEF),
            object:    Fingerprint::from_seed(0xCAFEBABE_FEEDFACE),
            qualia:    Fingerprint::from_seed(0xBAADF00D_DEADC0DE),
        }
    }
    
    pub fn get(&self, role: Role) -> &Fingerprint {
        match role {
            Role::Subject => &self.subject,
            Role::Predicate => &self.predicate,
            Role::Object => &self.object,
            Role::Qualia => &self.qualia,
        }
    }
    
    /// Memory usage: 4 fingerprints
    pub fn memory_bytes(&self) -> usize { 4 * N64 * 8 }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Subject = 0,
    Predicate = 1,
    Object = 2,
    Qualia = 3,
}

impl Role {
    pub fn from_u8(v: u8) -> Self {
        match v & 0x03 {
            0 => Role::Subject,
            1 => Role::Predicate,
            2 => Role::Object,
            _ => Role::Qualia,
        }
    }
}

// ============================================================================
// Chunk Index: Sparse mapping from chunks to codebook
// ============================================================================

#[derive(Clone, Copy)]
pub struct ChunkEntry {
    /// Symbol index (10 bits)
    pub symbol: u16,
    /// Role (2 bits)
    pub role: Role,
    /// Crystal cell coordinates (3 × 3 bits = 9 bits for 8×8×8)
    pub cell: (u8, u8, u8),
}

impl ChunkEntry {
    /// Pack into 24 bits (3 bytes)
    pub fn pack(&self) -> u32 {
        let sym = (self.symbol as u32) & 0x3FF;  // 10 bits
        let role = (self.role as u32) & 0x03;    // 2 bits
        let x = (self.cell.0 as u32) & 0x07;     // 3 bits
        let y = (self.cell.1 as u32) & 0x07;     // 3 bits
        let z = (self.cell.2 as u32) & 0x07;     // 3 bits
        
        sym | (role << 10) | (x << 12) | (y << 15) | (z << 18)
    }
    
    /// Unpack from 24 bits
    pub fn unpack(packed: u32) -> Self {
        Self {
            symbol: (packed & 0x3FF) as u16,
            role: Role::from_u8(((packed >> 10) & 0x03) as u8),
            cell: (
                ((packed >> 12) & 0x07) as u8,
                ((packed >> 15) & 0x07) as u8,
                ((packed >> 18) & 0x07) as u8,
            ),
        }
    }
}

// ============================================================================
// Dictionary Crystal: The main structure
// ============================================================================

pub struct DictionaryCrystal {
    /// Symbol codebook (learned)
    pub symbols: SymbolCodebook,
    /// Role codebook (fixed)
    pub roles: RoleCodebook,
    /// Chunk index (sparse)
    pub chunks: Vec<ChunkEntry>,
    /// Original text storage (optional, for retrieval)
    texts: Vec<String>,
    /// Cell prototype cache for fast resonance
    cell_prototypes: Box<[[[Option<Fingerprint>; GRID]; GRID]; GRID]>,
    /// Stats
    pub stats: DictionaryStats,
}

#[derive(Default, Debug)]
pub struct DictionaryStats {
    pub total_chunks: usize,
    pub unique_symbols: usize,
    pub codebook_memory_kb: usize,
    pub index_memory_kb: usize,
    pub total_memory_kb: usize,
    pub compression_ratio: f64,
}

impl DictionaryCrystal {
    pub fn new() -> Self {
        Self {
            symbols: SymbolCodebook::new(256),  // Start small, grow as needed
            roles: RoleCodebook::new(),
            chunks: Vec::new(),
            texts: Vec::new(),
            cell_prototypes: Box::new(std::array::from_fn(|_| 
                std::array::from_fn(|_| 
                    std::array::from_fn(|_| None)
                )
            )),
            stats: DictionaryStats::default(),
        }
    }
    
    /// Add chunk with automatic codebook learning
    pub fn add(&mut self, text: &str, role: Role) -> usize {
        let fp = Fingerprint::from_text(text);
        
        // Encode to symbol (may add to codebook if novel)
        let symbol = self.symbols.encode(&fp, 0.85);
        
        // Compute cell from reconstructed fingerprint
        let reconstructed = self.symbols.decode(symbol).xor(self.roles.get(role));
        let xyz = reconstructed.to_xyz();
        
        let entry = ChunkEntry {
            symbol,
            role,
            cell: (xyz.0 as u8, xyz.1 as u8, xyz.2 as u8),
        };
        
        let chunk_id = self.chunks.len();
        self.chunks.push(entry);
        self.texts.push(text.to_string());
        
        // Update cell prototype
        self.update_cell_prototype(xyz, &reconstructed);
        
        chunk_id
    }
    
    fn update_cell_prototype(&mut self, xyz: (usize, usize, usize), fp: &Fingerprint) {
        let (x, y, z) = xyz;
        match &mut self.cell_prototypes[x][y][z] {
            Some(proto) => {
                // Bundle with existing
                *proto = bundle(&[proto.clone(), fp.clone()]);
            }
            None => {
                self.cell_prototypes[x][y][z] = Some(fp.clone());
            }
        }
    }
    
    /// Reconstruct fingerprint for chunk
    pub fn reconstruct(&self, chunk_id: usize) -> Fingerprint {
        let entry = &self.chunks[chunk_id];
        self.symbols.decode(entry.symbol).xor(self.roles.get(entry.role))
    }
    
    /// Query: find chunks similar to query
    pub fn query(&self, query_text: &str, k: usize, threshold: f64) -> Vec<(usize, f64)> {
        let query_fp = Fingerprint::from_text(query_text);
        let query_xyz = query_fp.to_xyz();
        
        // 1. Check cell prototypes for hot cells
        let mut hot_cells = Vec::new();
        for x in 0..GRID {
            for y in 0..GRID {
                for z in 0..GRID {
                    if let Some(proto) = &self.cell_prototypes[x][y][z] {
                        let sim = query_fp.similarity(proto);
                        if sim > threshold * 0.8 {
                            hot_cells.push(((x, y, z), sim));
                        }
                    }
                }
            }
        }
        hot_cells.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // 2. Gather candidates from hot cells
        let mut candidates: Vec<(usize, f64)> = Vec::new();
        
        for ((x, y, z), _) in hot_cells.iter().take(10) {
            for (chunk_id, entry) in self.chunks.iter().enumerate() {
                if entry.cell.0 as usize == *x 
                    && entry.cell.1 as usize == *y 
                    && entry.cell.2 as usize == *z {
                    let fp = self.reconstruct(chunk_id);
                    let sim = query_fp.similarity(&fp);
                    if sim >= threshold {
                        candidates.push((chunk_id, sim));
                    }
                }
            }
        }
        
        // 3. Also check by symbol similarity
        // Encode query to nearest symbol
        let mut temp_symbols = self.symbols.clone();
        let query_symbol = temp_symbols.encode(&query_fp, 0.7);
        
        for (chunk_id, entry) in self.chunks.iter().enumerate() {
            if entry.symbol == query_symbol {
                let fp = self.reconstruct(chunk_id);
                let sim = query_fp.similarity(&fp);
                if sim >= threshold && !candidates.iter().any(|(id, _)| *id == chunk_id) {
                    candidates.push((chunk_id, sim));
                }
            }
        }
        
        // Sort and truncate
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        candidates.truncate(k);
        candidates
    }
    
    /// Get text for chunk
    pub fn get_text(&self, chunk_id: usize) -> Option<&str> {
        self.texts.get(chunk_id).map(|s| s.as_str())
    }
    
    /// Compute stats
    pub fn compute_stats(&mut self, original_bytes: usize) {
        let codebook_bytes = self.symbols.memory_bytes() + self.roles.memory_bytes();
        let index_bytes = self.chunks.len() * 3;  // 24 bits per entry
        let total = codebook_bytes + index_bytes;
        
        self.stats = DictionaryStats {
            total_chunks: self.chunks.len(),
            unique_symbols: self.symbols.len(),
            codebook_memory_kb: codebook_bytes / 1024,
            index_memory_kb: index_bytes / 1024,
            total_memory_kb: total / 1024,
            compression_ratio: original_bytes as f64 / total.max(1) as f64,
        };
    }
}

impl Clone for SymbolCodebook {
    fn clone(&self) -> Self {
        Self {
            symbols: self.symbols.clone(),
            lookup: self.lookup.clone(),
        }
    }
}

// ============================================================================
// Demo
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dictionary_crystal() {
        let mut crystal = DictionaryCrystal::new();
        
        // Add some chunks
        let chunks = vec![
            ("fn process_data(input: &[u8]) -> Result<Vec<u8>, Error>", Role::Subject),
            ("fn authenticate(user: &str, pass: &str) -> Token", Role::Subject),
            ("struct Config { url: String, timeout: u64 }", Role::Object),
            ("impl Config { fn new() -> Self }", Role::Predicate),
            ("fn process_data(data: &[u8]) -> Vec<u8>", Role::Subject),  // Similar to first
            ("fn validate_input(input: &str) -> bool", Role::Subject),
        ];
        
        let mut total_bytes = 0;
        for (text, role) in &chunks {
            crystal.add(text, *role);
            total_bytes += text.len();
        }
        
        crystal.compute_stats(total_bytes);
        
        println!("Dictionary Crystal Stats:");
        println!("  Chunks: {}", crystal.stats.total_chunks);
        println!("  Unique symbols: {}", crystal.stats.unique_symbols);
        println!("  Codebook: {} KB", crystal.stats.codebook_memory_kb);
        println!("  Index: {} KB", crystal.stats.index_memory_kb);
        println!("  Total: {} KB", crystal.stats.total_memory_kb);
        println!("  Compression: {:.1}x", crystal.stats.compression_ratio);
        
        // Test reconstruction quality
        for i in 0..chunks.len() {
            let original_fp = Fingerprint::from_text(chunks[i].0);
            let reconstructed_fp = crystal.reconstruct(i);
            let sim = original_fp.similarity(&reconstructed_fp);
            println!("  Chunk {}: reconstruction sim = {:.4}", i, sim);
        }
        
        // Test query
        let results = crystal.query("process_data function", 3, 0.5);
        println!("\nQuery: 'process_data function'");
        for (id, sim) in results {
            println!("  [{}] sim={:.3}: {:?}", id, sim, crystal.get_text(id));
        }
    }
    
    #[test]
    fn test_scaling() {
        let mut crystal = DictionaryCrystal::new();
        
        // Simulate 10K chunks
        let mut total_bytes = 0;
        for i in 0..10_000 {
            let text = format!("fn function_{}(arg: Type{}) -> Result<Output{}, Error>", i, i % 100, i % 50);
            let role = match i % 4 {
                0 => Role::Subject,
                1 => Role::Predicate,
                2 => Role::Object,
                _ => Role::Qualia,
            };
            crystal.add(&text, role);
            total_bytes += text.len();
        }
        
        crystal.compute_stats(total_bytes);
        
        println!("\n10K Chunk Scaling Test:");
        println!("  Original: {} KB", total_bytes / 1024);
        println!("  Chunks: {}", crystal.stats.total_chunks);
        println!("  Unique symbols: {} (of max {})", crystal.stats.unique_symbols, CODEBOOK_SIZE);
        println!("  Codebook: {} KB", crystal.stats.codebook_memory_kb);
        println!("  Index: {} KB (3 bytes × {})", crystal.stats.index_memory_kb, crystal.stats.total_chunks);
        println!("  Total: {} KB", crystal.stats.total_memory_kb);
        println!("  Compression: {:.1}x", crystal.stats.compression_ratio);
        
        // What we expect:
        // Original: ~600KB of text
        // Codebook: 1024 × 1.25KB = 1.25MB (but we might use fewer symbols)
        // Index: 10K × 3 bytes = 30KB
        // With 256 symbols: 256 × 1.25KB + 30KB = 320KB + 30KB = 350KB
        // Compression: 600KB / 350KB = 1.7x
        //
        // But the REAL win is when we have MILLIONS of chunks:
        // 1M chunks × 600 bytes = 600MB original
        // Codebook: 1024 × 1.25KB = 1.25MB (FIXED!)
        // Index: 1M × 3 bytes = 3MB
        // Total: 4.25MB
        // Compression: 141x
    }
}
