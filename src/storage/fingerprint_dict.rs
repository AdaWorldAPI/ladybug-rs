//! Fingerprint Dictionary (Codebook) for Sparse Hydration
//!
//! Maps 16-bit BindSpace addresses to fingerprints with O(1) lookup,
//! supporting sparse representation and batch CPU processing.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                    FINGERPRINT DICTIONARY (CODEBOOK)                        │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │                                                                             │
//! │  Sparse Repr:  [addr₁, addr₂, ... addrₙ] → packed u16 list                │
//! │  Hydrate:      sparse addrs → full fingerprints via O(1) lookup             │
//! │  Batch:        process N fingerprints in cache-friendly order               │
//! │                                                                             │
//! │  Zero-Copy:    Arrow FixedSizeList<UInt64> backing (no serialization)       │
//! │  HDR:          Pre-computed L0/L1 sketches for cascade search               │
//! │                                                                             │
//! │  Integration:  BindSpace → Dictionary → ArrowZeroCopy → DataFusion          │
//! │                                                                             │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! let dict = FingerprintDict::from_bind_space(&bind_space);
//!
//! // Sparse representation: just a list of addresses
//! let sparse = vec![0x8042u16, 0x8100, 0x8201];
//!
//! // Hydrate: O(1) per address, cache-friendly batch
//! let fingerprints = dict.hydrate_batch(&sparse);
//!
//! // Feed directly to DataFusion via Arrow
//! let arrow_batch = dict.to_record_batch(&sparse);
//! ```

use std::sync::Arc;

use arrow::array::*;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;

use super::bind_space::{Addr, BindNode, BindSpace, FINGERPRINT_WORDS, hamming_distance};

// =============================================================================
// CONSTANTS
// =============================================================================

/// Fingerprint size in bytes
const FP_BYTES: usize = FINGERPRINT_WORDS * 8;

/// Total addresses in the system
const TOTAL_ADDRESSES: usize = 65536;

/// Pre-computed L0 sketch size (1 bit per u64 word → 20 bytes for 156 words)
const L0_SKETCH_BYTES: usize = 20;

// =============================================================================
// DICTIONARY ENTRY
// =============================================================================

/// Compact dictionary entry: pre-computed sketch + popcount for fast filtering
#[derive(Clone)]
struct DictEntry {
    /// Pre-computed popcount of the fingerprint
    popcount: u32,
    /// L0 sketch: 1 bit per word (is the word non-zero?)
    l0_sketch: [u8; L0_SKETCH_BYTES],
}

impl DictEntry {
    fn from_fingerprint(fp: &[u64; FINGERPRINT_WORDS]) -> Self {
        let mut popcount = 0u32;
        let mut l0_sketch = [0u8; L0_SKETCH_BYTES];

        for (i, &word) in fp.iter().enumerate() {
            popcount += word.count_ones();
            if word != 0 {
                l0_sketch[i / 8] |= 1 << (i % 8);
            }
        }

        Self { popcount, l0_sketch }
    }
}

// =============================================================================
// FINGERPRINT DICTIONARY
// =============================================================================

/// Sparse fingerprint dictionary for O(1) hydration
///
/// Stores pre-computed sketches alongside the BindSpace reference,
/// enabling fast batch operations without touching the full 10K-bit
/// fingerprints until necessary.
pub struct FingerprintDict {
    /// Pre-computed entries for occupied addresses (indexed by raw addr)
    entries: Vec<Option<DictEntry>>,
    /// Occupied address list (sorted for cache-friendly iteration)
    occupied: Vec<u16>,
    /// Count of occupied entries
    count: usize,
}

impl FingerprintDict {
    /// Build dictionary from BindSpace (scans all zones)
    pub fn from_bind_space(bind_space: &BindSpace) -> Self {
        let mut entries: Vec<Option<DictEntry>> = vec![None; TOTAL_ADDRESSES];
        let mut occupied = Vec::new();

        // Scan all zones
        for prefix in 0u8..=0xFF {
            for slot in 0u8..=0xFF {
                let addr = Addr::new(prefix, slot);
                if let Some(node) = bind_space.read(addr) {
                    let raw = addr.0 as usize;
                    entries[raw] = Some(DictEntry::from_fingerprint(&node.fingerprint));
                    occupied.push(addr.0);
                }
            }
        }

        let count = occupied.len();
        Self { entries, occupied, count }
    }

    /// Number of entries in dictionary
    pub fn len(&self) -> usize {
        self.count
    }

    /// Is dictionary empty?
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Check if address has an entry
    #[inline(always)]
    pub fn contains(&self, addr: u16) -> bool {
        self.entries.get(addr as usize).is_some_and(|e| e.is_some())
    }

    /// Get pre-computed popcount for address (O(1))
    #[inline(always)]
    pub fn popcount(&self, addr: u16) -> Option<u32> {
        self.entries.get(addr as usize)
            .and_then(|e| e.as_ref())
            .map(|e| e.popcount)
    }

    /// Get L0 sketch for address (O(1))
    #[inline(always)]
    pub fn l0_sketch(&self, addr: u16) -> Option<&[u8; L0_SKETCH_BYTES]> {
        self.entries.get(addr as usize)
            .and_then(|e| e.as_ref())
            .map(|e| &e.l0_sketch)
    }

    /// All occupied addresses (sorted)
    pub fn addresses(&self) -> &[u16] {
        &self.occupied
    }

    // =========================================================================
    // SPARSE HYDRATION
    // =========================================================================

    /// Hydrate sparse address list into full fingerprints
    ///
    /// Given a list of 16-bit addresses, returns the corresponding
    /// fingerprints from BindSpace. This is the core "dictionary lookup"
    /// operation: sparse representation → dense fingerprints.
    ///
    /// Returns Vec of (addr, fingerprint) pairs for occupied addresses.
    pub fn hydrate_batch(
        &self,
        addrs: &[u16],
        bind_space: &BindSpace,
    ) -> Vec<(u16, [u64; FINGERPRINT_WORDS])> {
        let mut results = Vec::with_capacity(addrs.len());

        for &raw in addrs {
            if self.contains(raw) {
                let addr = Addr(raw);
                if let Some(node) = bind_space.read(addr) {
                    results.push((raw, node.fingerprint));
                }
            }
        }

        results
    }

    /// Hydrate a single address
    #[inline]
    pub fn hydrate_one(
        &self,
        addr: u16,
        bind_space: &BindSpace,
    ) -> Option<[u64; FINGERPRINT_WORDS]> {
        if self.contains(addr) {
            bind_space.read(Addr(addr)).map(|n| n.fingerprint)
        } else {
            None
        }
    }

    // =========================================================================
    // BATCH POPCOUNT FILTERING
    // =========================================================================

    /// Fast batch popcount filter: find addresses where popcount is within range
    ///
    /// Uses pre-computed popcounts (no fingerprint access needed).
    /// This is the "sweet spot" pre-filter: if query has popcount P,
    /// candidates must have popcount within [P - threshold, P + threshold]
    /// to possibly have Hamming distance < threshold.
    pub fn filter_by_popcount(
        &self,
        query_popcount: u32,
        max_hamming: u32,
    ) -> Vec<u16> {
        // Triangle inequality: |pop(a) - pop(b)| <= hamming(a, b)
        // So if |pop(a) - pop(query)| > max_hamming, skip
        let min_pop = query_popcount.saturating_sub(max_hamming);
        let max_pop = query_popcount.saturating_add(max_hamming);

        self.occupied.iter()
            .filter(|&&addr| {
                if let Some(entry) = &self.entries[addr as usize] {
                    entry.popcount >= min_pop && entry.popcount <= max_pop
                } else {
                    false
                }
            })
            .copied()
            .collect()
    }

    /// Batch Hamming search using dictionary pre-filters
    ///
    /// Three-stage pipeline:
    /// 1. Popcount pre-filter (no fingerprint access)
    /// 2. L0 sketch distance (20 bytes, ~90% filter)
    /// 3. Full Hamming distance (only for survivors)
    pub fn search(
        &self,
        query: &[u64; FINGERPRINT_WORDS],
        max_hamming: u32,
        bind_space: &BindSpace,
    ) -> Vec<(u16, u32)> {
        // Stage 1: Popcount pre-filter
        let query_pop: u32 = query.iter().map(|w| w.count_ones()).sum();
        let candidates = self.filter_by_popcount(query_pop, max_hamming);

        // Stage 2: L0 sketch filter
        let query_l0 = compute_l0_sketch(query);
        let mut survivors = Vec::new();

        for addr in candidates {
            if let Some(entry) = &self.entries[addr as usize] {
                // L0 sketch distance: rough Hamming estimate
                let l0_dist = sketch_distance(&query_l0, &entry.l0_sketch);
                // Each L0 bit represents ~64 bits, so l0_dist * ~32 ≈ min possible Hamming
                if l0_dist * 20 <= max_hamming {
                    survivors.push(addr);
                }
            }
        }

        // Stage 3: Full Hamming distance
        let mut results = Vec::new();
        for addr in survivors {
            if let Some(node) = bind_space.read(Addr(addr)) {
                let dist = hamming_distance(query, &node.fingerprint);
                if dist <= max_hamming {
                    results.push((addr, dist));
                }
            }
        }

        // Sort by distance
        results.sort_by_key(|&(_, d)| d);
        results
    }

    // =========================================================================
    // ARROW INTEGRATION
    // =========================================================================

    /// Convert sparse address list to Arrow RecordBatch with full fingerprints
    ///
    /// This is the bridge to DataFusion: sparse addresses become a queryable
    /// table with fingerprint, label, popcount columns.
    pub fn to_record_batch(
        &self,
        addrs: &[u16],
        bind_space: &BindSpace,
    ) -> Result<RecordBatch, arrow_schema::ArrowError> {
        let schema = dict_schema();

        let mut addr_vals = Vec::with_capacity(addrs.len());
        let mut fp_vals: Vec<Vec<u8>> = Vec::with_capacity(addrs.len());
        let mut label_vals: Vec<Option<String>> = Vec::with_capacity(addrs.len());
        let mut pop_vals = Vec::with_capacity(addrs.len());
        let mut zone_vals = Vec::with_capacity(addrs.len());

        for &raw in addrs {
            if let Some(node) = bind_space.read(Addr(raw)) {
                addr_vals.push(raw);

                let fp_bytes: Vec<u8> = node.fingerprint.iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect();
                fp_vals.push(fp_bytes);

                label_vals.push(node.label.clone());

                let pop = self.popcount(raw).unwrap_or_else(|| {
                    node.fingerprint.iter().map(|w| w.count_ones()).sum()
                });
                pop_vals.push(pop);

                let zone = match raw >> 8 {
                    p if p <= 0x0F => "surface",
                    p if p <= 0x7F => "fluid",
                    _ => "node",
                };
                zone_vals.push(zone.to_string());
            }
        }

        if addr_vals.is_empty() {
            return Ok(RecordBatch::new_empty(schema));
        }

        let addr_array: ArrayRef = Arc::new(UInt16Array::from(addr_vals));
        let mut fp_builder = FixedSizeBinaryBuilder::with_capacity(
            fp_vals.len(),
            FP_BYTES as i32,
        );
        for fp in &fp_vals {
            fp_builder.append_value(fp)?;
        }
        let fp_array: ArrayRef = Arc::new(fp_builder.finish());
        let label_array: ArrayRef = Arc::new(StringArray::from(label_vals));
        let pop_array: ArrayRef = Arc::new(UInt32Array::from(pop_vals));
        let zone_array: ArrayRef = Arc::new(StringArray::from(
            zone_vals.into_iter().map(Some).collect::<Vec<_>>(),
        ));

        RecordBatch::try_new(schema, vec![addr_array, fp_array, label_array, pop_array, zone_array])
    }

    /// Convert all occupied entries to Arrow RecordBatch
    pub fn to_full_record_batch(
        &self,
        bind_space: &BindSpace,
    ) -> Result<RecordBatch, arrow_schema::ArrowError> {
        let addrs = self.occupied.clone();
        self.to_record_batch(&addrs, bind_space)
    }

    // =========================================================================
    // INCREMENTAL UPDATE
    // =========================================================================

    /// Update dictionary entry for a single address
    pub fn update(&mut self, addr: u16, bind_space: &BindSpace) {
        let a = Addr(addr);
        if let Some(node) = bind_space.read(a) {
            let entry = DictEntry::from_fingerprint(&node.fingerprint);
            let was_present = self.entries[addr as usize].is_some();
            self.entries[addr as usize] = Some(entry);

            if !was_present {
                // Insert into sorted occupied list
                match self.occupied.binary_search(&addr) {
                    Ok(_) => {} // already present (shouldn't happen)
                    Err(pos) => {
                        self.occupied.insert(pos, addr);
                        self.count += 1;
                    }
                }
            }
        } else {
            // Address cleared
            if self.entries[addr as usize].take().is_some() {
                if let Ok(pos) = self.occupied.binary_search(&addr) {
                    self.occupied.remove(pos);
                    self.count -= 1;
                }
            }
        }
    }

    /// Batch update from list of modified addresses
    pub fn update_batch(&mut self, addrs: &[u16], bind_space: &BindSpace) {
        for &addr in addrs {
            self.update(addr, bind_space);
        }
    }
}

impl Default for FingerprintDict {
    fn default() -> Self {
        Self {
            entries: vec![None; TOTAL_ADDRESSES],
            occupied: Vec::new(),
            count: 0,
        }
    }
}

// =============================================================================
// SCHEMA
// =============================================================================

/// Schema for dictionary RecordBatch output
fn dict_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("address", DataType::UInt16, false),
        Field::new("fingerprint", DataType::FixedSizeBinary(FP_BYTES as i32), false),
        Field::new("label", DataType::Utf8, true),
        Field::new("popcount", DataType::UInt32, false),
        Field::new("zone", DataType::Utf8, false),
    ]))
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Compute L0 sketch from fingerprint
fn compute_l0_sketch(fp: &[u64; FINGERPRINT_WORDS]) -> [u8; L0_SKETCH_BYTES] {
    let mut sketch = [0u8; L0_SKETCH_BYTES];
    for (i, &word) in fp.iter().enumerate() {
        if word != 0 {
            sketch[i / 8] |= 1 << (i % 8);
        }
    }
    sketch
}

/// Hamming distance between two L0 sketches
fn sketch_distance(a: &[u8; L0_SKETCH_BYTES], b: &[u8; L0_SKETCH_BYTES]) -> u32 {
    let mut d = 0u32;
    for i in 0..L0_SKETCH_BYTES {
        d += (a[i] ^ b[i]).count_ones();
    }
    d
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dict_from_empty_bindspace() {
        let bs = BindSpace::new();
        let dict = FingerprintDict::from_bind_space(&bs);
        // Surface nodes are pre-initialized, so dict is not empty
        assert!(dict.len() > 0);
    }

    #[test]
    fn test_dict_hydration() {
        let mut bs = BindSpace::new();
        let fp = [42u64; FINGERPRINT_WORDS];
        let addr = bs.write_labeled(fp, "test_hydration");

        let dict = FingerprintDict::from_bind_space(&bs);

        // Hydrate single
        let result = dict.hydrate_one(addr.0, &bs);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), fp);

        // Hydrate batch
        let results = dict.hydrate_batch(&[addr.0], &bs);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, fp);
    }

    #[test]
    fn test_popcount_filter() {
        let mut bs = BindSpace::new();

        // Write node with known popcount
        let mut fp = [0u64; FINGERPRINT_WORDS];
        fp[0] = 0xFFFF; // 16 bits set
        let addr = bs.write_labeled(fp, "sixteen_bits");

        let dict = FingerprintDict::from_bind_space(&bs);

        // Popcount should be 16
        assert_eq!(dict.popcount(addr.0), Some(16));

        // Filter: query popcount 16, max hamming 10 → should include
        let candidates = dict.filter_by_popcount(16, 10);
        assert!(candidates.contains(&addr.0));

        // Filter: query popcount 100, max hamming 5 → should exclude
        let candidates = dict.filter_by_popcount(100, 5);
        assert!(!candidates.contains(&addr.0));
    }

    #[test]
    fn test_dict_search() {
        let mut bs = BindSpace::new();

        let mut fp1 = [0u64; FINGERPRINT_WORDS];
        fp1[0] = 0xFF; // 8 bits
        let addr1 = bs.write_labeled(fp1, "close");

        let mut fp2 = [0u64; FINGERPRINT_WORDS];
        fp2[0] = 0xFFFF_FFFF_FFFF_FFFF; // 64 bits
        let _addr2 = bs.write_labeled(fp2, "far");

        let dict = FingerprintDict::from_bind_space(&bs);

        // Search for something close to fp1
        let mut query = [0u64; FINGERPRINT_WORDS];
        query[0] = 0xFE; // 1 bit different from fp1
        let results = dict.search(&query, 50, &bs);

        assert!(!results.is_empty());
        // First result should be addr1 (closest)
        assert_eq!(results[0].0, addr1.0);
    }

    #[test]
    fn test_arrow_record_batch() {
        let mut bs = BindSpace::new();
        let fp = [1u64; FINGERPRINT_WORDS];
        let addr = bs.write_labeled(fp, "arrow_test");

        let dict = FingerprintDict::from_bind_space(&bs);
        let batch = dict.to_record_batch(&[addr.0], &bs).unwrap();

        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 5);
    }

    #[test]
    fn test_incremental_update() {
        let mut bs = BindSpace::new();
        let mut dict = FingerprintDict::from_bind_space(&bs);
        let initial_count = dict.len();

        // Add a new node
        let fp = [99u64; FINGERPRINT_WORDS];
        let addr = bs.write_labeled(fp, "incremental");

        // Dictionary doesn't know about it yet
        assert!(!dict.contains(addr.0));

        // Update
        dict.update(addr.0, &bs);
        assert!(dict.contains(addr.0));
        assert_eq!(dict.len(), initial_count + 1);

        // Delete and update
        bs.delete(addr);
        dict.update(addr.0, &bs);
        assert!(!dict.contains(addr.0));
        assert_eq!(dict.len(), initial_count);
    }
}
