//! Lance Zero-Copy Integration
//!
//! This module provides zero-copy access to Lance storage by sharing
//! the same address space. No serialization boundaries, no copies.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    UNIFIED ADDRESS SPACE                        │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                 │
//! │  HOT (BindSpace)     WARM (Lance mmap)      COLD (Lance file)   │
//! │  ┌───────────────┐   ┌───────────────┐     ┌───────────────┐   │
//! │  │ Array[32K]    │   │ mmap region   │     │ Parquet file  │   │
//! │  │ Direct access │   │ OS page cache │     │ Compressed    │   │
//! │  └───────┬───────┘   └───────┬───────┘     └───────┬───────┘   │
//! │          │                   │                     │            │
//! │          │◀──── BUBBLE UP ───│◀──── BUBBLE UP ─────│            │
//! │          │     (ptr move)    │     (page fault)    │            │
//! │          │                   │                     │            │
//! │          │──── SINK DOWN ───▶│──── SINK DOWN ─────▶│            │
//! │          │     (ptr move)    │     (OS evict)      │            │
//! │                                                                 │
//! │  SCENT INDEX: Awareness layer - tracks where everything lives   │
//! │                                                                 │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Key Principles
//!
//! 1. **No Serialization Boundary**: Lance and Ladybug share memory
//! 2. **Scent as Awareness**: Index knows where data lives without copying
//! 3. **Bubbling Without Copy**: Promote/demote by updating pointers
//! 4. **Arrow Native**: Fingerprints stored as Arrow FixedSizeList

use std::path::PathBuf;
use std::sync::Arc;

#[cfg(feature = "lance")]
use arrow_array::{Array, FixedSizeListArray, UInt64Array};
#[cfg(feature = "lance")]
use arrow_buffer::Buffer;

use super::bind_space::FINGERPRINT_WORDS;

// =============================================================================
// ZERO-COPY VIEW
// =============================================================================

/// A zero-copy view into Lance storage
///
/// This struct holds pointers into mmap'd Arrow columns.
/// No copies, just views into the OS page cache.
pub struct LanceView {
    /// Path to the Lance dataset
    pub path: PathBuf,

    /// Number of fingerprints in this view
    pub len: usize,

    /// Raw pointer to fingerprint data (156 × u64 per fingerprint)
    /// Points directly into mmap'd Arrow buffer
    #[cfg(feature = "lance")]
    fingerprint_ptr: *const u64,

    /// Raw pointer to address data (u16 per entry)
    #[cfg(feature = "lance")]
    address_ptr: *const u16,

    /// Scent index for awareness (which entries are hot/warm/cold)
    scent: ScentAwareness,
}

// Safety: The pointers point to mmap'd memory that outlives this struct
#[cfg(feature = "lance")]
unsafe impl Send for LanceView {}
#[cfg(feature = "lance")]
unsafe impl Sync for LanceView {}

impl LanceView {
    /// Create a new Lance view (placeholder for when Lance is vendored)
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            len: 0,
            #[cfg(feature = "lance")]
            fingerprint_ptr: std::ptr::null(),
            #[cfg(feature = "lance")]
            address_ptr: std::ptr::null(),
            scent: ScentAwareness::new(),
        }
    }

    /// Zero-copy access to fingerprint at index
    ///
    /// # Safety
    /// Caller must ensure index < self.len
    #[cfg(feature = "lance")]
    #[inline]
    pub unsafe fn fingerprint_unchecked(&self, index: usize) -> &[u64; FINGERPRINT_WORDS] {
        &*(self.fingerprint_ptr.add(index * FINGERPRINT_WORDS) as *const [u64; FINGERPRINT_WORDS])
    }

    /// Safe access to fingerprint
    #[cfg(feature = "lance")]
    pub fn fingerprint(&self, index: usize) -> Option<&[u64; FINGERPRINT_WORDS]> {
        if index < self.len {
            Some(unsafe { self.fingerprint_unchecked(index) })
        } else {
            None
        }
    }

    /// Get scent awareness
    pub fn scent(&self) -> &ScentAwareness {
        &self.scent
    }

    /// Get mutable scent awareness
    pub fn scent_mut(&mut self) -> &mut ScentAwareness {
        &mut self.scent
    }

    /// Number of entries
    pub fn len(&self) -> usize {
        self.len
    }

    /// Is empty?
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

// =============================================================================
// SCENT AWARENESS
// =============================================================================

/// Temperature tier for bubbling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Temperature {
    /// In BindSpace array (< 1 cycle access)
    Hot,
    /// In mmap region (page cache, ~100ns)
    Warm,
    /// On disk (needs page fault, ~10μs)
    Cold,
    /// Unknown/not tracked
    Unknown,
}

/// Scent awareness - knows where data lives without copying it
///
/// This is the "awareness" layer that tracks:
/// - Which fingerprints are hot (in BindSpace)
/// - Which are warm (in mmap page cache)
/// - Which are cold (on disk)
///
/// Bubbling happens by updating this index, not by copying data.
pub struct ScentAwareness {
    /// Hot set: indices that are in BindSpace
    hot_indices: Vec<u32>,

    /// Recently accessed (LRU for warmth tracking)
    recent_access: Vec<u32>,

    /// Access counts for promotion decisions
    access_counts: Vec<u8>,

    /// Capacity
    capacity: usize,
}

impl ScentAwareness {
    /// Create new scent awareness
    pub fn new() -> Self {
        Self::with_capacity(1024)
    }

    /// Create with capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            hot_indices: Vec::with_capacity(capacity / 10),
            recent_access: Vec::with_capacity(64),
            access_counts: vec![0; capacity],
            capacity,
        }
    }

    /// Record an access (for warmth tracking)
    pub fn touch(&mut self, index: u32) {
        let idx = index as usize;
        if idx < self.access_counts.len() {
            self.access_counts[idx] = self.access_counts[idx].saturating_add(1);
        }

        // Update recent access (simple ring buffer)
        if self.recent_access.len() >= 64 {
            self.recent_access.remove(0);
        }
        self.recent_access.push(index);
    }

    /// Get temperature of an index
    pub fn temperature(&self, index: u32) -> Temperature {
        if self.hot_indices.binary_search(&index).is_ok() {
            Temperature::Hot
        } else if self.recent_access.contains(&index) {
            Temperature::Warm
        } else if (index as usize) < self.access_counts.len()
            && self.access_counts[index as usize] > 0
        {
            Temperature::Warm
        } else {
            Temperature::Cold
        }
    }

    /// Mark index as hot (bubbled up to BindSpace)
    pub fn mark_hot(&mut self, index: u32) {
        if self.hot_indices.binary_search(&index).is_err() {
            self.hot_indices.push(index);
            self.hot_indices.sort();
        }
    }

    /// Mark index as cold (bubbled down from BindSpace)
    pub fn mark_cold(&mut self, index: u32) {
        if let Ok(pos) = self.hot_indices.binary_search(&index) {
            self.hot_indices.remove(pos);
        }
    }

    /// Get indices that should bubble up (high access count, currently cold)
    pub fn candidates_for_promotion(&self, limit: usize) -> Vec<u32> {
        let mut candidates: Vec<(u32, u8)> = self.access_counts
            .iter()
            .enumerate()
            .filter(|(idx, &count)| {
                count > 5 && self.hot_indices.binary_search(&(*idx as u32)).is_err()
            })
            .map(|(idx, &count)| (idx as u32, count))
            .collect();

        candidates.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by access count desc
        candidates.into_iter().take(limit).map(|(idx, _)| idx).collect()
    }

    /// Get indices that should bubble down (low access count, currently hot)
    pub fn candidates_for_demotion(&self, limit: usize) -> Vec<u32> {
        let mut candidates: Vec<(u32, u8)> = self.hot_indices
            .iter()
            .filter_map(|&idx| {
                let count = self.access_counts.get(idx as usize).copied().unwrap_or(0);
                if count < 2 {
                    Some((idx, count))
                } else {
                    None
                }
            })
            .collect();

        candidates.sort_by(|a, b| a.1.cmp(&b.1)); // Sort by access count asc
        candidates.into_iter().take(limit).map(|(idx, _)| idx).collect()
    }

    /// Decay access counts (call periodically)
    pub fn decay(&mut self) {
        for count in &mut self.access_counts {
            *count = count.saturating_sub(1);
        }
    }

    /// Number of hot entries
    pub fn hot_count(&self) -> usize {
        self.hot_indices.len()
    }
}

impl Default for ScentAwareness {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// BUBBLING OPERATIONS
// =============================================================================

/// Bubble operation result
#[derive(Debug, Clone)]
pub struct BubbleResult {
    /// Index that was bubbled
    pub index: u32,
    /// Previous temperature
    pub from: Temperature,
    /// New temperature
    pub to: Temperature,
    /// Whether data was copied (should be false for zero-copy)
    pub copied: bool,
}

/// Zero-copy bubbler
///
/// Moves data between hot/warm/cold tiers by updating pointers,
/// not by copying data.
pub struct ZeroCopyBubbler {
    /// Maximum hot entries
    max_hot: usize,
    /// Bubble threshold (access count to promote)
    promote_threshold: u8,
    /// Demote threshold (access count to demote)
    demote_threshold: u8,
}

impl Default for ZeroCopyBubbler {
    fn default() -> Self {
        Self {
            max_hot: 32768, // Same as BindSpace node capacity
            promote_threshold: 10,
            demote_threshold: 2,
        }
    }
}

impl ZeroCopyBubbler {
    /// Create with custom thresholds
    pub fn new(max_hot: usize, promote_threshold: u8, demote_threshold: u8) -> Self {
        Self {
            max_hot,
            promote_threshold,
            demote_threshold,
        }
    }

    /// Bubble up: promote cold/warm to hot
    ///
    /// In zero-copy mode, this just updates the scent index.
    /// The actual fingerprint stays in the mmap'd file.
    pub fn bubble_up(&self, scent: &mut ScentAwareness, index: u32) -> BubbleResult {
        let from = scent.temperature(index);
        scent.mark_hot(index);

        BubbleResult {
            index,
            from,
            to: Temperature::Hot,
            copied: false, // Zero-copy!
        }
    }

    /// Bubble down: demote hot to warm/cold
    pub fn bubble_down(&self, scent: &mut ScentAwareness, index: u32) -> BubbleResult {
        let from = scent.temperature(index);
        scent.mark_cold(index);

        BubbleResult {
            index,
            from,
            to: Temperature::Warm, // Goes to warm first (still in page cache)
            copied: false,
        }
    }

    /// Auto-bubble based on access patterns
    pub fn auto_bubble(&self, scent: &mut ScentAwareness) -> Vec<BubbleResult> {
        let mut results = Vec::new();

        // Promote hot candidates
        if scent.hot_count() < self.max_hot {
            let room = self.max_hot - scent.hot_count();
            for idx in scent.candidates_for_promotion(room.min(10)) {
                results.push(self.bubble_up(scent, idx));
            }
        }

        // Demote cold candidates if we're at capacity
        if scent.hot_count() >= self.max_hot {
            for idx in scent.candidates_for_demotion(10) {
                results.push(self.bubble_down(scent, idx));
            }
        }

        results
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scent_awareness() {
        let mut scent = ScentAwareness::with_capacity(100);

        // Initially cold
        assert_eq!(scent.temperature(42), Temperature::Cold);

        // Touch warms it up
        scent.touch(42);
        assert_eq!(scent.temperature(42), Temperature::Warm);

        // Mark hot
        scent.mark_hot(42);
        assert_eq!(scent.temperature(42), Temperature::Hot);

        // Mark cold again
        scent.mark_cold(42);
        assert_eq!(scent.temperature(42), Temperature::Warm); // Still warm from touch
    }

    #[test]
    fn test_bubbler() {
        let bubbler = ZeroCopyBubbler::default();
        let mut scent = ScentAwareness::with_capacity(100);

        // Bubble up
        let result = bubbler.bubble_up(&mut scent, 42);
        assert_eq!(result.to, Temperature::Hot);
        assert!(!result.copied); // Zero-copy!

        // Bubble down
        let result = bubbler.bubble_down(&mut scent, 42);
        assert_eq!(result.to, Temperature::Warm);
        assert!(!result.copied);
    }

    #[test]
    fn test_promotion_candidates() {
        let mut scent = ScentAwareness::with_capacity(100);

        // Touch index 10 many times
        for _ in 0..20 {
            scent.touch(10);
        }

        // Touch index 20 a few times
        for _ in 0..8 {
            scent.touch(20);
        }

        // Index 10 should be candidate for promotion
        let candidates = scent.candidates_for_promotion(5);
        assert!(candidates.contains(&10));
    }
}
