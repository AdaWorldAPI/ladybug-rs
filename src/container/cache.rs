//! Write-through container cache with zero-detection.
//!
//! All writes go through this cache. Direct mutation is forbidden.
//! XOR is self-inverse: `a ⊕ a = 0`. Double-writes silently zero out data.
//! The cache detects this and rejects zero containers.

use super::Container;

/// Error type for cache operations.
#[derive(Debug, Clone)]
pub enum CacheError {
    /// Attempted to write a zero container (would destroy data).
    ZeroContainer { idx: usize },
    /// Index out of bounds.
    OutOfBounds { idx: usize, len: usize },
    /// Spine recomputation produced zero (double-fold detected).
    ZeroSpine { spine_idx: usize },
}

impl std::fmt::Display for CacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CacheError::ZeroContainer { idx } =>
                write!(f, "zero container at index {}", idx),
            CacheError::OutOfBounds { idx, len } =>
                write!(f, "index {} out of bounds (len={})", idx, len),
            CacheError::ZeroSpine { spine_idx } =>
                write!(f, "spine at index {} recomputed to zero", spine_idx),
        }
    }
}

impl std::error::Error for CacheError {}

/// Write-through cache for container records with zero-detection.
pub struct ContainerCache {
    /// Live container slots.
    slots: Vec<Container>,

    /// Generation counter per slot (increments on every write).
    generation: Vec<u64>,

    /// Dirty bitmap (1 bit per slot).
    dirty: Vec<u64>,

    /// Popcount per slot (0 = alarm).
    popcount: Vec<u32>,
}

impl ContainerCache {
    /// Create a cache with the given number of slots.
    pub fn new(num_slots: usize) -> Self {
        let dirty_words = (num_slots + 63) / 64;
        Self {
            slots: vec![Container::zero(); num_slots],
            generation: vec![0u64; num_slots],
            dirty: vec![0u64; dirty_words],
            popcount: vec![0u32; num_slots],
        }
    }

    /// Number of slots.
    #[inline]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// True if no slots.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Write a container to a slot. Rejects zero containers.
    pub fn write(&mut self, idx: usize, data: &Container) -> Result<(), CacheError> {
        if idx >= self.slots.len() {
            return Err(CacheError::OutOfBounds { idx, len: self.slots.len() });
        }

        let pc = data.popcount();
        if pc == 0 {
            return Err(CacheError::ZeroContainer { idx });
        }

        self.slots[idx] = data.clone();
        self.generation[idx] = self.generation[idx].wrapping_add(1);
        self.popcount[idx] = pc;
        self.mark_dirty(idx);
        Ok(())
    }

    /// Read a container (direct reference, no copy).
    #[inline]
    pub fn read(&self, idx: usize) -> &Container {
        &self.slots[idx]
    }

    /// Get generation counter for a slot.
    #[inline]
    pub fn generation(&self, idx: usize) -> u64 {
        self.generation[idx]
    }

    /// Get popcount for a slot.
    #[inline]
    pub fn slot_popcount(&self, idx: usize) -> u32 {
        self.popcount[idx]
    }

    /// Recompute a spine as XOR-fold of children. Rejects zero result.
    pub fn recompute_spine(
        &mut self,
        children: &[usize],
        spine_idx: usize,
    ) -> Result<(), CacheError> {
        if spine_idx >= self.slots.len() {
            return Err(CacheError::OutOfBounds { idx: spine_idx, len: self.slots.len() });
        }

        let mut spine = Container::zero();
        for &child in children {
            if child < self.slots.len() {
                spine = spine.xor(&self.slots[child]);
            }
        }

        let pc = spine.popcount();
        if pc == 0 && !children.is_empty() {
            return Err(CacheError::ZeroSpine { spine_idx });
        }

        self.slots[spine_idx] = spine;
        self.generation[spine_idx] = self.generation[spine_idx].wrapping_add(1);
        self.popcount[spine_idx] = pc;
        self.clear_dirty(spine_idx);
        Ok(())
    }

    /// Scan dirty bitmap and return indices of dirty slots.
    pub fn dirty_indices(&self) -> Vec<usize> {
        let mut result = Vec::new();
        for (word_idx, &word) in self.dirty.iter().enumerate() {
            if word == 0 {
                continue;
            }
            let base = word_idx * 64;
            let mut bits = word;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                let idx = base + bit;
                if idx < self.slots.len() {
                    result.push(idx);
                }
                bits &= bits - 1; // clear lowest set bit
            }
        }
        result
    }

    /// Returns indices of zero or suspicious containers.
    pub fn validate(&self) -> Vec<usize> {
        let mut bad = Vec::new();
        for (i, pc) in self.popcount.iter().enumerate() {
            if *pc == 0 && self.generation[i] > 0 {
                // Was written at least once but is now zero — suspicious
                bad.push(i);
            }
        }
        bad
    }

    // -- Dirty bitmap helpers --

    pub fn mark_dirty(&mut self, idx: usize) {
        let word = idx / 64;
        let bit = idx % 64;
        if word < self.dirty.len() {
            self.dirty[word] |= 1u64 << bit;
        }
    }

    fn clear_dirty(&mut self, idx: usize) {
        let word = idx / 64;
        let bit = idx % 64;
        if word < self.dirty.len() {
            self.dirty[word] &= !(1u64 << bit);
        }
    }

    /// Check if a slot is dirty.
    pub fn is_dirty(&self, idx: usize) -> bool {
        let word = idx / 64;
        let bit = idx % 64;
        if word < self.dirty.len() {
            self.dirty[word] & (1u64 << bit) != 0
        } else {
            false
        }
    }

    /// Clear all dirty bits.
    pub fn clear_all_dirty(&mut self) {
        for w in &mut self.dirty {
            *w = 0;
        }
    }

    /// Append a new container, growing the cache by one slot.
    /// Returns the index of the newly added slot.
    pub fn push(&mut self, data: &Container) -> Result<usize, CacheError> {
        let pc = data.popcount();
        if pc == 0 {
            return Err(CacheError::ZeroContainer { idx: self.slots.len() });
        }

        let idx = self.slots.len();
        self.slots.push(data.clone());
        self.generation.push(1);
        self.popcount.push(pc);

        // Grow dirty bitmap if needed
        let needed_words = (idx + 1 + 63) / 64;
        if needed_words > self.dirty.len() {
            self.dirty.resize(needed_words, 0);
        }

        Ok(idx)
    }

    /// Append an empty (zero) slot for a spine that will be computed lazily.
    /// Returns the index. Marks it dirty so it gets recomputed on first read.
    pub fn push_spine_slot(&mut self) -> usize {
        let idx = self.slots.len();
        self.slots.push(Container::zero());
        self.generation.push(0);
        self.popcount.push(0);

        let needed_words = (idx + 1 + 63) / 64;
        if needed_words > self.dirty.len() {
            self.dirty.resize(needed_words, 0);
        }
        self.mark_dirty(idx);
        idx
    }
}
