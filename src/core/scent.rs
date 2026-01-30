//! Scent Index - Hierarchical Content-Addressable Filtering
//!
//! Petabyte-scale resonance search via scent nodes.
//! 
//! Query: "Siamese cat videos" in 7 PB
//! Time: ~100 ns to eliminate 99.997% of corpus
//!
//! See docs/SCENT_INDEX.md for full architecture.

use std::path::Path;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

/// Fingerprint size: 10K bits = 1250 bytes
pub const FP_BYTES: usize = 1250;

/// Scent size: 5 bytes = 40 bits
pub const SCENT_BYTES: usize = 5;

/// Buckets per level
pub const BUCKETS: usize = 256;

/// Chunk header with embedded scent and cognitive markers
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct ChunkHeader {
    /// Chunk ID (0-255)
    pub chunk_id: u8,
    /// Start offset in data file
    pub offset: u64,
    /// Number of fingerprints in chunk
    pub count: u32,
    /// Compressed representative (scent)
    pub scent: [u8; SCENT_BYTES],
    /// Learning rate for this region (Ada cognitive)
    pub plasticity: f32,
    /// Cached decision/classification (Ada cognitive)
    pub decision: u8,
    /// Last access timestamp (attention tracking)
    pub last_access: u64,
}

impl ChunkHeader {
    pub fn new(chunk_id: u8) -> Self {
        Self {
            chunk_id,
            offset: 0,
            count: 0,
            scent: [0; SCENT_BYTES],
            plasticity: 1.0,
            decision: 0,
            last_access: 0,
        }
    }
}

impl Default for ChunkHeader {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Extract 5-byte scent from full fingerprint
#[inline]
pub fn extract_scent(fp: &[u8]) -> [u8; SCENT_BYTES] {
    if fp.len() < SCENT_BYTES {
        let mut scent = [0u8; SCENT_BYTES];
        scent[..fp.len()].copy_from_slice(fp);
        return scent;
    }
    
    // XOR-fold: captures global structure in 5 bytes
    let mut scent = [0u8; SCENT_BYTES];
    for (i, &b) in fp.iter().enumerate() {
        scent[i % SCENT_BYTES] ^= b;
    }
    scent
}

/// Compute scent distance (Hamming on 40 bits)
#[inline]
pub fn scent_distance(a: &[u8; SCENT_BYTES], b: &[u8; SCENT_BYTES]) -> u32 {
    let mut dist = 0u32;
    for i in 0..SCENT_BYTES {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}

/// Check if scents match within threshold
#[inline]
pub fn scent_matches(a: &[u8; SCENT_BYTES], b: &[u8; SCENT_BYTES], threshold: u32) -> bool {
    scent_distance(a, b) <= threshold
}

/// Single-level scent index (up to ~7 TB)
pub struct ScentIndexL1 {
    pub headers: Box<[ChunkHeader; BUCKETS]>,
}

impl ScentIndexL1 {
    /// Create empty L1 index
    pub fn new() -> Self {
        let headers: [ChunkHeader; BUCKETS] = std::array::from_fn(|i| ChunkHeader::new(i as u8));
        Self {
            headers: Box::new(headers),
        }
    }
    
    /// Extract scents-only view (1.25 KB, L1 cache friendly)
    pub fn scents(&self) -> [[u8; SCENT_BYTES]; BUCKETS] {
        std::array::from_fn(|i| self.headers[i].scent)
    }
    
    /// Find matching chunks via scent scan
    pub fn find_chunks(&self, query_scent: &[u8; SCENT_BYTES], threshold: u32) -> Vec<u8> {
        self.headers
            .iter()
            .filter(|h| h.count > 0)
            .filter(|h| scent_matches(&h.scent, query_scent, threshold))
            .map(|h| h.chunk_id)
            .collect()
    }
    
    /// Find chunks filtered by plasticity (cognitive search)
    pub fn find_chunks_plastic(
        &self,
        query_scent: &[u8; SCENT_BYTES],
        threshold: u32,
        min_plasticity: f32,
    ) -> Vec<u8> {
        self.headers
            .iter()
            .filter(|h| h.count > 0)
            .filter(|h| h.plasticity >= min_plasticity)
            .filter(|h| scent_matches(&h.scent, query_scent, threshold))
            .map(|h| h.chunk_id)
            .collect()
    }
    
    /// Assign fingerprint to chunk (returns chunk ID)
    #[inline]
    pub fn assign(&self, fp: &[u8]) -> u8 {
        // First byte of fingerprint = chunk ID (locality preserving)
        fp[0]
    }
    
    /// Update chunk on append
    pub fn on_append(&mut self, chunk: u8, fp: &[u8], offset: u64) {
        let h = &mut self.headers[chunk as usize];
        
        if h.count == 0 {
            h.offset = offset;
            h.scent = extract_scent(fp);
        } else {
            // Rolling scent update (EWMA)
            let new_scent = extract_scent(fp);
            for i in 0..SCENT_BYTES {
                h.scent[i] = ((h.scent[i] as u16 * 15 + new_scent[i] as u16) / 16) as u8;
            }
        }
        
        h.count += 1;
        h.last_access = timestamp();
    }
    
    /// Set decision for a chunk (O(1), affects millions of fps)
    pub fn set_decision(&mut self, chunk: u8, decision: u8) {
        self.headers[chunk as usize].decision = decision;
    }
    
    /// Set plasticity for a chunk (O(1), affects millions of fps)
    pub fn set_plasticity(&mut self, chunk: u8, plasticity: f32) {
        self.headers[chunk as usize].plasticity = plasticity;
    }
    
    /// Get chunk statistics
    pub fn stats(&self) -> ScentStats {
        let active = self.headers.iter().filter(|h| h.count > 0).count();
        let total_fps: u64 = self.headers.iter().map(|h| h.count as u64).sum();
        let avg_plasticity: f32 = self.headers.iter()
            .filter(|h| h.count > 0)
            .map(|h| h.plasticity)
            .sum::<f32>() / active.max(1) as f32;
        
        ScentStats {
            depth: 1,
            active_buckets: active,
            total_fingerprints: total_fps,
            avg_plasticity,
        }
    }
}

impl Default for ScentIndexL1 {
    fn default() -> Self {
        Self::new()
    }
}

/// Two-level scent index (up to ~1.8 PB)
pub struct ScentIndexL2 {
    pub l1: ScentIndexL1,
    pub l2: Box<[ScentIndexL1; BUCKETS]>,
}

impl ScentIndexL2 {
    pub fn new() -> Self {
        Self {
            l1: ScentIndexL1::new(),
            l2: Box::new(std::array::from_fn(|_| ScentIndexL1::new())),
        }
    }
    
    /// Find matching (l1, l2) pairs
    pub fn find_chunks(
        &self,
        query_scent: &[u8; SCENT_BYTES],
        threshold: u32,
    ) -> Vec<(u8, u8)> {
        let l1_matches = self.l1.find_chunks(query_scent, threshold);
        
        l1_matches
            .iter()
            .flat_map(|&l1| {
                self.l2[l1 as usize]
                    .find_chunks(query_scent, threshold)
                    .into_iter()
                    .map(move |l2| (l1, l2))
            })
            .collect()
    }
    
    /// Assign fingerprint to (l1, l2) bucket
    pub fn assign(&self, fp: &[u8]) -> (u8, u8) {
        let l1 = fp[0];
        let l2 = fp[1];
        (l1, l2)
    }
    
    /// Update on append
    pub fn on_append(&mut self, fp: &[u8], offset: u64) {
        let (l1, l2) = self.assign(fp);
        
        // Update L1
        self.l1.on_append(l1, fp, offset);
        
        // Update L2
        self.l2[l1 as usize].on_append(l2, fp, offset);
    }
    
    /// Set decision at L1 level (affects ~27 TB)
    pub fn set_decision_l1(&mut self, l1: u8, decision: u8) {
        self.l1.set_decision(l1, decision);
    }
    
    /// Set decision at L2 level (affects ~107 GB)
    pub fn set_decision_l2(&mut self, l1: u8, l2: u8, decision: u8) {
        self.l2[l1 as usize].set_decision(l2, decision);
    }
    
    /// Set plasticity at L1 level
    pub fn set_plasticity_l1(&mut self, l1: u8, plasticity: f32) {
        self.l1.set_plasticity(l1, plasticity);
    }
    
    /// Set plasticity at L2 level
    pub fn set_plasticity_l2(&mut self, l1: u8, l2: u8, plasticity: f32) {
        self.l2[l1 as usize].set_plasticity(l2, plasticity);
    }
}

impl Default for ScentIndexL2 {
    fn default() -> Self {
        Self::new()
    }
}

/// Unified scent index (auto-scales by depth)
pub enum ScentIndex {
    L1(ScentIndexL1),
    L2(ScentIndexL2),
    // L3, L4 can be added as needed
}

impl ScentIndex {
    /// Create single-level index
    pub fn new() -> Self {
        ScentIndex::L1(ScentIndexL1::new())
    }
    
    /// Create two-level index
    pub fn new_l2() -> Self {
        ScentIndex::L2(ScentIndexL2::new())
    }
    
    /// Depth of index
    pub fn depth(&self) -> usize {
        match self {
            ScentIndex::L1(_) => 1,
            ScentIndex::L2(_) => 2,
        }
    }
    
    /// Find matching bucket addresses
    pub fn find(&self, query_fp: &[u8], threshold: u32) -> Vec<BucketAddr> {
        let query_scent = extract_scent(query_fp);
        
        match self {
            ScentIndex::L1(idx) => {
                idx.find_chunks(&query_scent, threshold)
                    .into_iter()
                    .map(|l1| BucketAddr::L1(l1))
                    .collect()
            }
            ScentIndex::L2(idx) => {
                idx.find_chunks(&query_scent, threshold)
                    .into_iter()
                    .map(|(l1, l2)| BucketAddr::L2(l1, l2))
                    .collect()
            }
        }
    }
    
    /// Update on append
    pub fn on_append(&mut self, fp: &[u8], offset: u64) {
        match self {
            ScentIndex::L1(idx) => {
                let chunk = idx.assign(fp);
                idx.on_append(chunk, fp, offset);
            }
            ScentIndex::L2(idx) => {
                idx.on_append(fp, offset);
            }
        }
    }
    
    /// Get statistics
    pub fn stats(&self) -> ScentStats {
        match self {
            ScentIndex::L1(idx) => idx.stats(),
            ScentIndex::L2(idx) => {
                let l1_stats = idx.l1.stats();
                let l2_total: u64 = idx.l2.iter().map(|l| l.stats().total_fingerprints).sum();
                ScentStats {
                    depth: 2,
                    active_buckets: l1_stats.active_buckets * BUCKETS, // Approximate
                    total_fingerprints: l2_total,
                    avg_plasticity: l1_stats.avg_plasticity,
                }
            }
        }
    }
    
    // ========== Persistence ==========
    
    /// Save to file
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut w = BufWriter::new(file);
        
        // Magic + version + depth
        w.write_all(b"SCNT")?;
        w.write_all(&1u32.to_le_bytes())?;
        w.write_all(&(self.depth() as u8).to_le_bytes())?;
        
        match self {
            ScentIndex::L1(idx) => {
                self.write_headers(&mut w, &idx.headers)?;
            }
            ScentIndex::L2(idx) => {
                self.write_headers(&mut w, &idx.l1.headers)?;
                for l2 in idx.l2.iter() {
                    self.write_headers(&mut w, &l2.headers)?;
                }
            }
        }
        
        w.flush()
    }
    
    /// Load from file
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mut r = BufReader::new(file);
        
        // Magic
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != b"SCNT" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid scent index magic",
            ));
        }
        
        // Version
        let mut version = [0u8; 4];
        r.read_exact(&mut version)?;
        
        // Depth
        let mut depth = [0u8; 1];
        r.read_exact(&mut depth)?;
        
        match depth[0] {
            1 => {
                let mut idx = ScentIndexL1::new();
                Self::read_headers(&mut r, &mut idx.headers)?;
                Ok(ScentIndex::L1(idx))
            }
            2 => {
                let mut idx = ScentIndexL2::new();
                Self::read_headers(&mut r, &mut idx.l1.headers)?;
                for l2 in idx.l2.iter_mut() {
                    Self::read_headers(&mut r, &mut l2.headers)?;
                }
                Ok(ScentIndex::L2(idx))
            }
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Unsupported scent index depth",
            )),
        }
    }
    
    fn write_headers<W: Write>(&self, w: &mut W, headers: &[ChunkHeader; BUCKETS]) -> std::io::Result<()> {
        for h in headers.iter() {
            w.write_all(&[h.chunk_id])?;
            w.write_all(&h.offset.to_le_bytes())?;
            w.write_all(&h.count.to_le_bytes())?;
            w.write_all(&h.scent)?;
            w.write_all(&h.plasticity.to_le_bytes())?;
            w.write_all(&[h.decision])?;
            w.write_all(&h.last_access.to_le_bytes())?;
        }
        Ok(())
    }
    
    fn read_headers<R: Read>(r: &mut R, headers: &mut [ChunkHeader; BUCKETS]) -> std::io::Result<()> {
        for h in headers.iter_mut() {
            let mut buf1 = [0u8; 1];
            let mut buf4 = [0u8; 4];
            let mut buf8 = [0u8; 8];
            
            r.read_exact(&mut buf1)?;
            h.chunk_id = buf1[0];
            
            r.read_exact(&mut buf8)?;
            h.offset = u64::from_le_bytes(buf8);
            
            r.read_exact(&mut buf4)?;
            h.count = u32::from_le_bytes(buf4);
            
            r.read_exact(&mut h.scent)?;
            
            r.read_exact(&mut buf4)?;
            h.plasticity = f32::from_le_bytes(buf4);
            
            r.read_exact(&mut buf1)?;
            h.decision = buf1[0];
            
            r.read_exact(&mut buf8)?;
            h.last_access = u64::from_le_bytes(buf8);
        }
        Ok(())
    }
}

impl Default for ScentIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Bucket address (supports any depth)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BucketAddr {
    L1(u8),
    L2(u8, u8),
    L3(u8, u8, u8),
}

impl BucketAddr {
    /// Flatten to u32 for hashing/comparison
    pub fn flatten(&self) -> u32 {
        match self {
            BucketAddr::L1(a) => *a as u32,
            BucketAddr::L2(a, b) => ((*a as u32) << 8) | (*b as u32),
            BucketAddr::L3(a, b, c) => ((*a as u32) << 16) | ((*b as u32) << 8) | (*c as u32),
        }
    }
}

/// Statistics
#[derive(Debug)]
pub struct ScentStats {
    pub depth: usize,
    pub active_buckets: usize,
    pub total_fingerprints: u64,
    pub avg_plasticity: f32,
}

/// Current timestamp (milliseconds)
fn timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ========== SIMD Optimized Scent Scan ==========

#[cfg(target_arch = "x86_64")]
mod simd {
    use super::*;
    
    /// SIMD-optimized scent scan (AVX2)
    /// Compares query against 256 scents, returns matching chunk IDs
    #[cfg(target_feature = "avx2")]
    pub fn find_chunks_simd(
        scents: &[[u8; SCENT_BYTES]; BUCKETS],
        query: &[u8; SCENT_BYTES],
        threshold: u32,
    ) -> Vec<u8> {
        // For now, fall back to scalar
        // TODO: Implement AVX2 version
        scents
            .iter()
            .enumerate()
            .filter(|(_, s)| scent_distance(s, query) <= threshold)
            .map(|(i, _)| i as u8)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn make_fp(seed: u8) -> [u8; FP_BYTES] {
        let mut fp = [0u8; FP_BYTES];
        for (i, b) in fp.iter_mut().enumerate() {
            *b = seed.wrapping_add(i as u8);
        }
        fp
    }
    
    #[test]
    fn test_extract_scent() {
        let fp = make_fp(42);
        let scent = extract_scent(&fp);
        assert_eq!(scent.len(), SCENT_BYTES);
    }
    
    #[test]
    fn test_scent_distance() {
        let a = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
        let b = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
        assert_eq!(scent_distance(&a, &b), 0);
        
        let c = [0x00, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(scent_distance(&a, &c), 40); // All 40 bits differ
    }
    
    #[test]
    fn test_l1_append_find() {
        let mut idx = ScentIndexL1::new();
        
        let fp1 = make_fp(0x42);
        let fp2 = make_fp(0x42); // Same bucket
        let fp3 = make_fp(0x99); // Different bucket
        
        idx.on_append(0x42, &fp1, 0);
        idx.on_append(0x42, &fp2, 1250);
        idx.on_append(0x99, &fp3, 2500);
        
        assert_eq!(idx.headers[0x42].count, 2);
        assert_eq!(idx.headers[0x99].count, 1);
        
        // Find should return bucket 0x42 for similar query
        let query = make_fp(0x42);
        let matches = idx.find_chunks(&extract_scent(&query), 10);
        assert!(matches.contains(&0x42));
    }
    
    #[test]
    fn test_l2_append_find() {
        let mut idx = ScentIndexL2::new();
        
        let fp = make_fp(0x42);
        idx.on_append(&fp, 0);
        
        let (l1, l2) = idx.assign(&fp);
        assert_eq!(l1, 0x42);
        
        let matches = idx.find_chunks(&extract_scent(&fp), 10);
        assert!(!matches.is_empty());
    }
    
    #[test]
    fn test_cognitive_markers() {
        let mut idx = ScentIndexL1::new();
        
        let fp = make_fp(0x10);
        idx.on_append(0x10, &fp, 0);
        
        // Set plasticity
        idx.set_plasticity(0x10, 0.5);
        assert_eq!(idx.headers[0x10].plasticity, 0.5);
        
        // Set decision
        idx.set_decision(0x10, 42);
        assert_eq!(idx.headers[0x10].decision, 42);
        
        // Search with plasticity filter
        let matches = idx.find_chunks_plastic(&extract_scent(&fp), 10, 0.3);
        assert!(matches.contains(&0x10));
        
        let no_matches = idx.find_chunks_plastic(&extract_scent(&fp), 10, 0.9);
        assert!(!no_matches.contains(&0x10));
    }
    
    #[test]
    fn test_persistence() {
        let mut idx = ScentIndex::new();
        
        let fp = make_fp(0x55);
        idx.on_append(&fp, 12345);
        
        let tmp = tempfile::NamedTempFile::new().unwrap();
        idx.save(tmp.path()).unwrap();
        
        let loaded = ScentIndex::load(tmp.path()).unwrap();
        
        assert_eq!(loaded.depth(), 1);
        assert_eq!(loaded.stats().total_fingerprints, 1);
    }
    
    #[test]
    fn test_bucket_addr_flatten() {
        assert_eq!(BucketAddr::L1(0x42).flatten(), 0x42);
        assert_eq!(BucketAddr::L2(0x12, 0x34).flatten(), 0x1234);
        assert_eq!(BucketAddr::L3(0x12, 0x34, 0x56).flatten(), 0x123456);
    }
}
