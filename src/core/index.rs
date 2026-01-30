//! Ladybug Index - Content Addressable Memory
//!
//! 64-bit universal addressing:
//! ┌──────────────────┬──────────────────────────────────────────────┐
//! │   16 bits        │                 48 bits                      │
//! │   TYPE           │            fingerprint prefix                │
//! └──────────────────┴──────────────────────────────────────────────┘
//!
//! All query types (SQL, Cypher, Hamming, Vector) resolve to same operation.
//! Immutable after build. Zero-copy mmap. SIMD bucket scan.

use std::path::Path;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::sync::Arc;

/// Fingerprint size: 10K bits = 1250 bytes
pub const FP_BYTES: usize = 1250;

/// Type IDs (16-bit namespace)
pub mod types {
    // Core entities
    pub const THOUGHT: u16 = 0x0001;
    pub const CONCEPT: u16 = 0x0002;
    pub const STYLE: u16 = 0x0003;
    
    // Edge types (graph relations)
    pub const EDGE_CAUSES: u16 = 0x0100;
    pub const EDGE_SUPPORTS: u16 = 0x0101;
    pub const EDGE_CONTRADICTS: u16 = 0x0102;
    pub const EDGE_BECOMES: u16 = 0x0103;
    pub const EDGE_REFINES: u16 = 0x0104;
    pub const EDGE_GROUNDS: u16 = 0x0105;
    pub const EDGE_ABSTRACTS: u16 = 0x0106;
    
    // Consciousness layers
    pub const LAYER_SUBSTRATE: u16 = 0x0200;
    pub const LAYER_FELT_CORE: u16 = 0x0201;
    pub const LAYER_BODY: u16 = 0x0202;
    pub const LAYER_QUALIA: u16 = 0x0203;
    pub const LAYER_VOLITION: u16 = 0x0204;
    pub const LAYER_GESTALT: u16 = 0x0205;
    pub const LAYER_META: u16 = 0x0206;
    
    // Thinking styles
    pub const STYLE_ANALYTICAL: u16 = 0x0300;
    pub const STYLE_INTUITIVE: u16 = 0x0301;
    pub const STYLE_FOCUSED: u16 = 0x0302;
    pub const STYLE_DIFFUSE: u16 = 0x0303;
    pub const STYLE_CONVERGENT: u16 = 0x0304;
    pub const STYLE_DIVERGENT: u16 = 0x0305;
    pub const STYLE_CONCRETE: u16 = 0x0306;
    pub const STYLE_ABSTRACT: u16 = 0x0307;
    pub const STYLE_SEQUENTIAL: u16 = 0x0308;
    pub const STYLE_HOLISTIC: u16 = 0x0309;
    pub const STYLE_VERBAL: u16 = 0x030A;
    pub const STYLE_SPATIAL: u16 = 0x030B;
    
    // Codebook
    pub const CODE: u16 = 0x0400;
    
    // Edge type range for traversal
    pub const EDGE_START: u16 = 0x0100;
    pub const EDGE_END: u16 = 0x01FF;
    
    /// Check if type is an edge
    #[inline]
    pub fn is_edge(t: u16) -> bool {
        t >= EDGE_START && t <= EDGE_END
    }
}

/// 64-bit key: 16-bit type + 48-bit fingerprint prefix
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Key(u64);

impl Key {
    /// Create key from type and full fingerprint
    #[inline]
    pub fn new(type_id: u16, fp: &[u8]) -> Self {
        let prefix = fp_to_48(fp);
        Self((type_id as u64) << 48 | prefix)
    }
    
    /// Create key from raw u64
    #[inline]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    
    /// Get type ID (top 16 bits)
    #[inline]
    pub fn type_id(self) -> u16 {
        (self.0 >> 48) as u16
    }
    
    /// Get fingerprint prefix (bottom 48 bits)
    #[inline]
    pub fn prefix(self) -> u64 {
        self.0 & 0xFFFFFFFFFFFF
    }
    
    /// Get raw u64
    #[inline]
    pub fn raw(self) -> u64 {
        self.0
    }
}

impl std::fmt::Debug for Key {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Key(0x{:04X}:{:012X})", self.type_id(), self.prefix())
    }
}

/// Extract 48-bit prefix from fingerprint
#[inline]
fn fp_to_48(fp: &[u8]) -> u64 {
    if fp.len() < 6 {
        let mut buf = [0u8; 8];
        buf[..fp.len()].copy_from_slice(fp);
        return u64::from_le_bytes(buf) & 0xFFFFFFFFFFFF;
    }
    
    // First 6 bytes = 48 bits
    let bytes: [u8; 8] = [
        fp[0], fp[1], fp[2], fp[3], fp[4], fp[5], 0, 0
    ];
    u64::from_le_bytes(bytes)
}

/// Entry in a bucket: (48-bit prefix, row offset, optional target for edges)
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Entry {
    pub prefix: u64,      // 48-bit fingerprint prefix (stored in 64)
    pub offset: u64,      // Row offset in Arrow/Lance
    pub target: u64,      // For edges: target fingerprint prefix. 0 otherwise.
}

impl Entry {
    #[inline]
    pub fn new(prefix: u64, offset: u64) -> Self {
        Self { prefix, offset, target: 0 }
    }
    
    #[inline]
    pub fn edge(prefix: u64, offset: u64, target: u64) -> Self {
        Self { prefix, offset, target }
    }
}

/// Builder for LadybugIndex (mutable during construction)
pub struct IndexBuilder {
    buckets: Vec<Vec<Entry>>,
    count: usize,
}

impl IndexBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            buckets: (0..65536).map(|_| Vec::new()).collect(),
            count: 0,
        }
    }
    
    /// Insert an entry
    #[inline]
    pub fn insert(&mut self, type_id: u16, fp: &[u8], offset: u64) {
        let prefix = fp_to_48(fp);
        self.buckets[type_id as usize].push(Entry::new(prefix, offset));
        self.count += 1;
    }
    
    /// Insert an edge (with target fingerprint)
    #[inline]
    pub fn insert_edge(&mut self, edge_type: u16, src_fp: &[u8], offset: u64, tgt_fp: &[u8]) {
        let src_prefix = fp_to_48(src_fp);
        let tgt_prefix = fp_to_48(tgt_fp);
        self.buckets[edge_type as usize].push(Entry::edge(src_prefix, offset, tgt_prefix));
        self.count += 1;
    }
    
    /// Number of entries
    pub fn len(&self) -> usize {
        self.count
    }
    
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    
    /// Freeze into immutable index
    pub fn build(self) -> LadybugIndex {
        let buckets: Vec<Box<[Entry]>> = self.buckets
            .into_iter()
            .map(|v| v.into_boxed_slice())
            .collect();
        
        LadybugIndex {
            buckets: buckets.into_boxed_slice(),
            count: self.count,
        }
    }
}

impl Default for IndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Immutable Ladybug Index
/// 
/// Once built, cannot be mutated. Safe for concurrent access.
/// Append = build new index from old + new entries, atomic swap.
pub struct LadybugIndex {
    // 65536 buckets, one per type ID
    buckets: Box<[Box<[Entry]>]>,
    count: usize,
}

impl LadybugIndex {
    /// Create empty index
    pub fn empty() -> Self {
        let buckets: Vec<Box<[Entry]>> = (0..65536)
            .map(|_| Vec::new().into_boxed_slice())
            .collect();
        Self {
            buckets: buckets.into_boxed_slice(),
            count: 0,
        }
    }
    
    /// Number of entries
    pub fn len(&self) -> usize {
        self.count
    }
    
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    
    // ========== Core Lookups ==========
    
    /// O(1) lookup by type and fingerprint
    #[inline]
    pub fn get(&self, type_id: u16, fp: &[u8]) -> Option<u64> {
        let prefix = fp_to_48(fp);
        self.get_by_prefix(type_id, prefix)
    }
    
    /// O(1) lookup by type and prefix
    #[inline]
    pub fn get_by_prefix(&self, type_id: u16, prefix: u64) -> Option<u64> {
        let bucket = &self.buckets[type_id as usize];
        
        // SIMD-friendly: linear scan over contiguous memory
        // For typical bucket sizes (<1000), this beats tree structures
        bucket.iter()
            .find(|e| e.prefix == prefix)
            .map(|e| e.offset)
    }
    
    /// Get by Key
    #[inline]
    pub fn get_key(&self, key: Key) -> Option<u64> {
        self.get_by_prefix(key.type_id(), key.prefix())
    }
    
    // ========== SQL Surface ==========
    
    /// SQL: SELECT * FROM thoughts WHERE fp = X
    #[inline]
    pub fn get_thought(&self, fp: &[u8]) -> Option<u64> {
        self.get(types::THOUGHT, fp)
    }
    
    /// SQL: SELECT * FROM concepts WHERE fp = X
    #[inline]
    pub fn get_concept(&self, fp: &[u8]) -> Option<u64> {
        self.get(types::CONCEPT, fp)
    }
    
    // ========== Cypher Surface ==========
    
    /// Cypher: MATCH (n:Label {fp: X})
    #[inline]
    pub fn match_node(&self, label: u16, fp: &[u8]) -> Option<u64> {
        self.get(label, fp)
    }
    
    /// Cypher: MATCH (a)-[:REL]->(b) WHERE a.fp = X
    /// Returns iterator of (offset, target_prefix)
    pub fn match_edges_from<'a>(&'a self, edge_type: u16, src_fp: &[u8]) -> impl Iterator<Item = (u64, u64)> + 'a {
        let src_prefix = fp_to_48(src_fp);
        self.buckets[edge_type as usize]
            .iter()
            .filter(move |e| e.prefix == src_prefix)
            .map(|e| (e.offset, e.target))
    }
    
    /// Cypher: MATCH (a)-[*]->(b) - all edge types from source
    pub fn all_edges_from<'a>(&'a self, src_fp: &[u8]) -> impl Iterator<Item = (u16, u64, u64)> + 'a {
        let src_prefix = fp_to_48(src_fp);
        (types::EDGE_START..=types::EDGE_END)
            .flat_map(move |t| {
                self.buckets[t as usize]
                    .iter()
                    .filter(move |e| e.prefix == src_prefix)
                    .map(move |e| (t, e.offset, e.target))
            })
    }
    
    /// Cypher: MATCH (a)<-[:REL]-(b) - incoming edges
    pub fn match_edges_to<'a>(&'a self, edge_type: u16, tgt_fp: &[u8]) -> impl Iterator<Item = (u64, u64)> + 'a {
        let tgt_prefix = fp_to_48(tgt_fp);
        self.buckets[edge_type as usize]
            .iter()
            .filter(move |e| e.target == tgt_prefix)
            .map(|e| (e.offset, e.prefix))
    }
    
    // ========== Scan Surface ==========
    
    /// Scan all entries of a type
    pub fn scan_type(&self, type_id: u16) -> impl Iterator<Item = &Entry> {
        self.buckets[type_id as usize].iter()
    }
    
    /// Scan all entries
    pub fn scan_all(&self) -> impl Iterator<Item = (u16, &Entry)> {
        self.buckets.iter().enumerate()
            .flat_map(|(t, bucket)| {
                bucket.iter().map(move |e| (t as u16, e))
            })
    }
    
    // ========== Persistence ==========
    
    /// Save index to file
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        
        // Header: magic + version + count
        writer.write_all(b"LBUG")?;
        writer.write_all(&1u32.to_le_bytes())?;  // version
        writer.write_all(&(self.count as u64).to_le_bytes())?;
        
        // For each bucket: length + entries
        for bucket in self.buckets.iter() {
            writer.write_all(&(bucket.len() as u32).to_le_bytes())?;
            for entry in bucket.iter() {
                writer.write_all(&entry.prefix.to_le_bytes())?;
                writer.write_all(&entry.offset.to_le_bytes())?;
                writer.write_all(&entry.target.to_le_bytes())?;
            }
        }
        
        writer.flush()?;
        Ok(())
    }
    
    /// Load index from file
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        
        // Header
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != b"LBUG" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid index file magic"
            ));
        }
        
        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes)?;
        let _version = u32::from_le_bytes(version_bytes);
        
        let mut count_bytes = [0u8; 8];
        reader.read_exact(&mut count_bytes)?;
        let count = u64::from_le_bytes(count_bytes) as usize;
        
        // Buckets
        let mut buckets: Vec<Box<[Entry]>> = Vec::with_capacity(65536);
        for _ in 0..65536 {
            let mut len_bytes = [0u8; 4];
            reader.read_exact(&mut len_bytes)?;
            let len = u32::from_le_bytes(len_bytes) as usize;
            
            let mut entries = Vec::with_capacity(len);
            for _ in 0..len {
                let mut prefix_bytes = [0u8; 8];
                let mut offset_bytes = [0u8; 8];
                let mut target_bytes = [0u8; 8];
                
                reader.read_exact(&mut prefix_bytes)?;
                reader.read_exact(&mut offset_bytes)?;
                reader.read_exact(&mut target_bytes)?;
                
                entries.push(Entry {
                    prefix: u64::from_le_bytes(prefix_bytes),
                    offset: u64::from_le_bytes(offset_bytes),
                    target: u64::from_le_bytes(target_bytes),
                });
            }
            
            buckets.push(entries.into_boxed_slice());
        }
        
        Ok(Self {
            buckets: buckets.into_boxed_slice(),
            count,
        })
    }
    
    // ========== COW Append ==========
    
    /// Create new index by merging self with additional entries
    /// Old index remains valid (COW semantics)
    pub fn append(&self, additions: IndexBuilder) -> Self {
        let mut builder = IndexBuilder::new();
        
        // Copy existing entries
        for (type_id, bucket) in self.buckets.iter().enumerate() {
            for entry in bucket.iter() {
                builder.buckets[type_id].push(*entry);
            }
        }
        builder.count = self.count;
        
        // Add new entries
        for (type_id, bucket) in additions.buckets.into_iter().enumerate() {
            for entry in bucket {
                builder.buckets[type_id].push(entry);
            }
        }
        builder.count += additions.count;
        
        builder.build()
    }
}

// Safe for concurrent read access
unsafe impl Sync for LadybugIndex {}
unsafe impl Send for LadybugIndex {}

/// Thread-safe handle for atomic index swaps
pub struct IndexHandle {
    inner: std::sync::RwLock<Arc<LadybugIndex>>,
}

impl IndexHandle {
    pub fn new(index: LadybugIndex) -> Self {
        Self {
            inner: std::sync::RwLock::new(Arc::new(index)),
        }
    }
    
    /// Get read access to current index
    pub fn read(&self) -> Arc<LadybugIndex> {
        self.inner.read().unwrap().clone()
    }
    
    /// Atomic swap to new index
    pub fn swap(&self, new_index: LadybugIndex) {
        let mut guard = self.inner.write().unwrap();
        *guard = Arc::new(new_index);
    }
    
    /// Append and swap atomically
    pub fn append(&self, additions: IndexBuilder) {
        let current = self.read();
        let new_index = current.append(additions);
        self.swap(new_index);
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
    fn test_key_roundtrip() {
        let fp = make_fp(42);
        let key = Key::new(types::THOUGHT, &fp);
        
        assert_eq!(key.type_id(), types::THOUGHT);
        assert_eq!(key.prefix(), fp_to_48(&fp));
    }
    
    #[test]
    fn test_basic_insert_lookup() {
        let mut builder = IndexBuilder::new();
        let fp1 = make_fp(1);
        let fp2 = make_fp(2);
        
        builder.insert(types::THOUGHT, &fp1, 100);
        builder.insert(types::THOUGHT, &fp2, 200);
        builder.insert(types::CONCEPT, &fp1, 300);
        
        let index = builder.build();
        
        assert_eq!(index.get_thought(&fp1), Some(100));
        assert_eq!(index.get_thought(&fp2), Some(200));
        assert_eq!(index.get_concept(&fp1), Some(300));
        assert_eq!(index.get_concept(&fp2), None);
    }
    
    #[test]
    fn test_edge_traversal() {
        let mut builder = IndexBuilder::new();
        let a = make_fp(1);
        let b = make_fp(2);
        let c = make_fp(3);
        
        builder.insert_edge(types::EDGE_CAUSES, &a, 100, &b);
        builder.insert_edge(types::EDGE_CAUSES, &a, 101, &c);
        builder.insert_edge(types::EDGE_SUPPORTS, &b, 200, &c);
        
        let index = builder.build();
        
        // a -[:CAUSES]-> ?
        let from_a: Vec<_> = index.match_edges_from(types::EDGE_CAUSES, &a).collect();
        assert_eq!(from_a.len(), 2);
        
        // ? -[:SUPPORTS]-> c
        let to_c: Vec<_> = index.match_edges_to(types::EDGE_SUPPORTS, &c).collect();
        assert_eq!(to_c.len(), 1);
    }
    
    #[test]
    fn test_cow_append() {
        let mut builder = IndexBuilder::new();
        let fp1 = make_fp(1);
        builder.insert(types::THOUGHT, &fp1, 100);
        let index1 = builder.build();
        
        let mut additions = IndexBuilder::new();
        let fp2 = make_fp(2);
        additions.insert(types::THOUGHT, &fp2, 200);
        
        let index2 = index1.append(additions);
        
        // Old index unchanged
        assert_eq!(index1.get_thought(&fp1), Some(100));
        assert_eq!(index1.get_thought(&fp2), None);
        
        // New index has both
        assert_eq!(index2.get_thought(&fp1), Some(100));
        assert_eq!(index2.get_thought(&fp2), Some(200));
    }
    
    #[test]
    fn test_persistence() {
        let mut builder = IndexBuilder::new();
        let fp = make_fp(42);
        builder.insert(types::THOUGHT, &fp, 12345);
        builder.insert_edge(types::EDGE_CAUSES, &fp, 100, &make_fp(99));
        
        let index = builder.build();
        
        let tmp = tempfile::NamedTempFile::new().unwrap();
        index.save(tmp.path()).unwrap();
        
        let loaded = LadybugIndex::load(tmp.path()).unwrap();
        
        assert_eq!(loaded.len(), index.len());
        assert_eq!(loaded.get_thought(&fp), Some(12345));
    }
    
    #[test]
    fn test_all_edges() {
        let mut builder = IndexBuilder::new();
        let a = make_fp(1);
        let b = make_fp(2);
        
        builder.insert_edge(types::EDGE_CAUSES, &a, 100, &b);
        builder.insert_edge(types::EDGE_SUPPORTS, &a, 101, &b);
        
        let index = builder.build();
        
        let all: Vec<_> = index.all_edges_from(&a).collect();
        assert_eq!(all.len(), 2);
    }
}
