//! XOR Delta Chains and Write Cache for 16K Cognitive Records.
//!
//! XOR is the universal operation (HDC binding). Deltas between records
//! are sparse: most words are unchanged. A delta only stores non-zero
//! XOR words, avoiding full 2KB copies.

use super::VECTOR_WORDS;
use std::collections::HashMap;
use std::sync::RwLock;

/// Sparse XOR delta: bitmap of which words changed + the non-zero XOR values.
///
/// For a typical single-field metadata update, this stores 1-2 words instead of 256.
#[derive(Clone, Debug)]
pub struct XorDelta {
    /// 4 x u64 bitmap: bit i = word i has a non-zero XOR
    pub bitmap: [u64; 4],
    /// Non-zero XOR values, in word-index order
    pub values: Vec<u64>,
}

impl XorDelta {
    /// Compute delta between two records
    pub fn compute(old: &[u64; VECTOR_WORDS], new: &[u64; VECTOR_WORDS]) -> Self {
        let mut bitmap = [0u64; 4];
        let mut values = Vec::new();
        for i in 0..VECTOR_WORDS {
            let xor = old[i] ^ new[i];
            if xor != 0 {
                bitmap[i / 64] |= 1u64 << (i % 64);
                values.push(xor);
            }
        }
        Self { bitmap, values }
    }

    /// Apply delta to a record (in-place XOR)
    pub fn apply(&self, words: &mut [u64; VECTOR_WORDS]) {
        let mut vi = 0;
        for i in 0..VECTOR_WORDS {
            if self.bitmap[i / 64] & (1u64 << (i % 64)) != 0 {
                words[i] ^= self.values[vi];
                vi += 1;
            }
        }
    }

    /// Number of changed words
    pub fn changed_words(&self) -> usize {
        self.bitmap.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Is this a resonance-only delta? (no metadata words changed)
    pub fn is_resonance_only(&self) -> bool {
        // Words 224-255 map to bitmap bits 224-255 (bitmap[3] bits 32-63)
        let metadata_mask = 0xFFFFFFFF00000000u64;
        self.bitmap[3] & metadata_mask == 0
    }

    /// Empty delta (no changes)
    pub fn empty() -> Self {
        Self { bitmap: [0; 4], values: Vec::new() }
    }

    /// Is this delta empty?
    pub fn is_empty(&self) -> bool {
        self.bitmap == [0; 4]
    }
}

/// Chain of deltas from an anchor point (for DN tree paths).
///
/// Stores: anchor + sequence of deltas. Reconstructs any version
/// by applying deltas in order from the anchor.
#[derive(Clone, Debug)]
pub struct DeltaChain {
    /// The anchor record (full 256 words)
    pub anchor: Box<[u64; VECTOR_WORDS]>,
    /// Ordered deltas from anchor
    pub deltas: Vec<XorDelta>,
}

impl DeltaChain {
    /// Create a new chain from an anchor record
    pub fn new(anchor: [u64; VECTOR_WORDS]) -> Self {
        Self {
            anchor: Box::new(anchor),
            deltas: Vec::new(),
        }
    }

    /// Push a new version, storing only the delta
    pub fn push(&mut self, new_record: &[u64; VECTOR_WORDS]) {
        let current = self.reconstruct_latest();
        let delta = XorDelta::compute(&current, new_record);
        if !delta.is_empty() {
            self.deltas.push(delta);
        }
    }

    /// Reconstruct the latest version
    pub fn reconstruct_latest(&self) -> [u64; VECTOR_WORDS] {
        let mut record = *self.anchor;
        for delta in &self.deltas {
            delta.apply(&mut record);
        }
        record
    }

    /// Number of versions (anchor + deltas)
    pub fn version_count(&self) -> usize {
        1 + self.deltas.len()
    }
}

/// XOR write cache: accumulates deltas in memory, applies on read.
///
/// Avoids Arrow copy-on-write for small updates. Reads return the
/// patched version by applying the cached delta to the base record.
pub struct XorWriteCache {
    /// Pending deltas keyed by record ID
    deltas: HashMap<u64, XorDelta>,
}

impl XorWriteCache {
    pub fn new() -> Self {
        Self { deltas: HashMap::new() }
    }

    /// Stage a write: compute delta from old to new, merge with existing delta
    pub fn stage(&mut self, id: u64, old: &[u64; VECTOR_WORDS], new: &[u64; VECTOR_WORDS]) {
        let delta = XorDelta::compute(old, new);
        if delta.is_empty() { return; }

        if let Some(existing) = self.deltas.get_mut(&id) {
            // Merge: apply existing delta to old, then compute delta to new
            let mut intermediate = *old;
            existing.apply(&mut intermediate);
            *existing = XorDelta::compute(&intermediate, new);
        } else {
            self.deltas.insert(id, delta);
        }
    }

    /// Read with cache: apply pending delta if present
    pub fn read(&self, id: u64, base: &[u64; VECTOR_WORDS]) -> CacheRead {
        match self.deltas.get(&id) {
            None => CacheRead::Clean,
            Some(delta) => {
                let mut patched = *base;
                delta.apply(&mut patched);
                CacheRead::Patched(Box::new(patched))
            }
        }
    }

    /// Flush all pending deltas (apply to storage)
    pub fn drain(&mut self) -> Vec<(u64, XorDelta)> {
        self.deltas.drain().collect()
    }

    /// Number of dirty records
    pub fn dirty_count(&self) -> usize {
        self.deltas.len()
    }
}

/// Read result: either clean (no pending delta) or patched (delta applied).
/// Owned to avoid lifetime issues with the RwLock wrapper.
pub enum CacheRead {
    Clean,
    Patched(Box<[u64; VECTOR_WORDS]>),
}

/// Thread-safe write cache wrapper.
///
/// Returns owned CacheRead to avoid lifetime entanglement with the RwLock.
/// This is the pattern from doc 04 (race condition fixes): owned enum
/// instead of borrowed references across lock boundaries.
pub struct ConcurrentWriteCache {
    inner: RwLock<XorWriteCache>,
}

impl ConcurrentWriteCache {
    pub fn new() -> Self {
        Self { inner: RwLock::new(XorWriteCache::new()) }
    }

    pub fn stage(&self, id: u64, old: &[u64; VECTOR_WORDS], new: &[u64; VECTOR_WORDS]) {
        self.inner.write().unwrap().stage(id, old, new);
    }

    pub fn read(&self, id: u64, base: &[u64; VECTOR_WORDS]) -> CacheRead {
        self.inner.read().unwrap().read(id, base)
    }

    pub fn dirty_count(&self) -> usize {
        self.inner.read().unwrap().dirty_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(seed: u64) -> [u64; VECTOR_WORDS] {
        let mut r = [0u64; VECTOR_WORDS];
        for i in 0..VECTOR_WORDS {
            r[i] = seed.wrapping_mul(i as u64 + 1);
        }
        r
    }

    #[test]
    fn test_delta_compute_apply() {
        let a = make_record(42);
        let mut b = a;
        b[100] ^= 0xFF; // flip 8 bits in word 100
        b[200] ^= 0x01; // flip 1 bit in word 200 (resonance zone boundary)

        let delta = XorDelta::compute(&a, &b);
        assert_eq!(delta.changed_words(), 2);

        let mut reconstructed = a;
        delta.apply(&mut reconstructed);
        assert_eq!(reconstructed, b);
    }

    #[test]
    fn test_delta_empty() {
        let a = make_record(42);
        let delta = XorDelta::compute(&a, &a);
        assert!(delta.is_empty());
        assert_eq!(delta.changed_words(), 0);
    }

    #[test]
    fn test_delta_resonance_only() {
        let a = make_record(42);
        let mut b = a;
        b[50] ^= 0xFF; // only resonance word changed
        let delta = XorDelta::compute(&a, &b);
        assert!(delta.is_resonance_only());

        let mut c = a;
        c[240] ^= 0xFF; // metadata word changed
        let delta2 = XorDelta::compute(&a, &c);
        assert!(!delta2.is_resonance_only());
    }

    #[test]
    fn test_delta_chain() {
        let v0 = make_record(1);
        let mut chain = DeltaChain::new(v0);

        let mut v1 = v0;
        v1[10] = 0xDEADBEEF;
        chain.push(&v1);

        let mut v2 = v1;
        v2[20] = 0xCAFEBABE;
        chain.push(&v2);

        assert_eq!(chain.version_count(), 3);
        assert_eq!(chain.reconstruct_latest(), v2);
    }

    #[test]
    fn test_write_cache() {
        let base = make_record(42);
        let mut updated = base;
        updated[100] = 0xFFFF;

        let mut cache = XorWriteCache::new();
        cache.stage(1, &base, &updated);
        assert_eq!(cache.dirty_count(), 1);

        match cache.read(1, &base) {
            CacheRead::Patched(record) => assert_eq!(record[100], 0xFFFF),
            CacheRead::Clean => panic!("Expected patched"),
        }

        match cache.read(2, &base) {
            CacheRead::Clean => {} // correct: no delta for ID 2
            CacheRead::Patched(_) => panic!("Expected clean"),
        }
    }

    #[test]
    fn test_concurrent_cache() {
        let cache = ConcurrentWriteCache::new();
        let base = make_record(42);
        let mut updated = base;
        updated[50] = 0xBEEF;

        cache.stage(1, &base, &updated);
        assert_eq!(cache.dirty_count(), 1);

        match cache.read(1, &base) {
            CacheRead::Patched(r) => assert_eq!(r[50], 0xBEEF),
            CacheRead::Clean => panic!("Expected patched"),
        }
    }
}
