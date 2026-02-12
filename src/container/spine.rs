//! Spine declaration + lock-free concurrency.
//!
//! A spine is the XOR-fold of its children. When children are written
//! concurrently, the spine gets recomputed lazily (on next read) from
//! whatever state the children have. No lock needed — XOR is commutative
//! and associative.

use std::collections::HashMap;
use super::Container;
use super::cache::{ContainerCache, CacheError};

/// Spine-aware container cache.
///
/// Extends `ContainerCache` with spine declarations and lazy recomputation.
pub struct SpineCache {
    /// Underlying container cache.
    pub cache: ContainerCache,

    /// Spine → children mapping.
    spine_map: HashMap<usize, Vec<usize>>,

    /// Reverse mapping: child → set of spines it belongs to.
    child_to_spines: HashMap<usize, Vec<usize>>,
}

impl SpineCache {
    /// Create a spine cache with the given number of slots.
    pub fn new(num_slots: usize) -> Self {
        Self {
            cache: ContainerCache::new(num_slots),
            spine_map: HashMap::new(),
            child_to_spines: HashMap::new(),
        }
    }

    /// Declare a spine over a set of children.
    /// Writes to any child mark the spine as dirty.
    pub fn declare_spine(&mut self, spine_idx: usize, children: Vec<usize>) {
        // Update reverse mapping
        for &child in &children {
            self.child_to_spines
                .entry(child)
                .or_default()
                .push(spine_idx);
        }
        self.spine_map.insert(spine_idx, children);
    }

    /// Write a child container. Marks parent spines as dirty.
    pub fn write_child(&mut self, child_idx: usize, data: &Container) -> Result<(), CacheError> {
        self.cache.write(child_idx, data)?;

        // Mark all spines containing this child as dirty
        if let Some(spines) = self.child_to_spines.get(&child_idx) {
            for &spine_idx in spines {
                self.cache.mark_dirty(spine_idx);
            }
        }

        Ok(())
    }

    /// Read a spine. Recomputes if dirty.
    pub fn read_spine(&mut self, spine_idx: usize) -> Result<&Container, CacheError> {
        if self.cache.is_dirty(spine_idx) {
            if let Some(children) = self.spine_map.get(&spine_idx).cloned() {
                self.cache.recompute_spine(&children, spine_idx)?;
            }
        }
        Ok(self.cache.read(spine_idx))
    }

    /// Read any container (non-spine).
    #[inline]
    pub fn read(&self, idx: usize) -> &Container {
        self.cache.read(idx)
    }

    /// Get all spine declarations.
    pub fn spines(&self) -> &HashMap<usize, Vec<usize>> {
        &self.spine_map
    }

    /// Flush all dirty spines.
    pub fn flush_dirty(&mut self) -> Vec<CacheError> {
        let mut errors = Vec::new();
        let spine_indices: Vec<usize> = self.spine_map.keys().copied().collect();

        for spine_idx in spine_indices {
            if self.cache.is_dirty(spine_idx) {
                if let Some(children) = self.spine_map.get(&spine_idx).cloned() {
                    if let Err(e) = self.cache.recompute_spine(&children, spine_idx) {
                        errors.push(e);
                    }
                }
            }
        }

        errors
    }

    // ========================================================================
    // TREE MANAGEMENT: Query + mutate the spine topology
    // ========================================================================

    /// Check if an index is a declared spine.
    #[inline]
    pub fn is_spine(&self, idx: usize) -> bool {
        self.spine_map.contains_key(&idx)
    }

    /// Get the children of a spine.
    pub fn spine_children(&self, spine_idx: usize) -> &[usize] {
        self.spine_map
            .get(&spine_idx)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Append a new leaf container to the cache.
    /// Returns the assigned index.
    pub fn push_leaf(&mut self, data: &Container) -> Result<usize, CacheError> {
        self.cache.push(data)
    }

    /// Allocate a new spine slot (zero, marked dirty) and declare it
    /// over the given children. Returns the spine index.
    pub fn push_spine(&mut self, children: Vec<usize>) -> usize {
        let idx = self.cache.push_spine_slot();
        self.declare_spine(idx, children);
        idx
    }

    /// Add a child to an existing spine. Marks the spine dirty.
    pub fn add_child_to_spine(&mut self, spine_idx: usize, child_idx: usize) {
        // Forward mapping
        self.spine_map
            .entry(spine_idx)
            .or_default()
            .push(child_idx);

        // Reverse mapping
        self.child_to_spines
            .entry(child_idx)
            .or_default()
            .push(spine_idx);

        self.cache.mark_dirty(spine_idx);
    }

    /// Reparent a child: remove it from `old_spine` and add it to `new_spine`.
    /// Marks both spines dirty.
    pub fn reparent(&mut self, child_idx: usize, old_spine: usize, new_spine: usize) {
        // Remove child from old_spine's children
        if let Some(children) = self.spine_map.get_mut(&old_spine) {
            children.retain(|&c| c != child_idx);
        }

        // Remove old_spine from child's reverse mapping
        if let Some(spines) = self.child_to_spines.get_mut(&child_idx) {
            spines.retain(|&s| s != old_spine);
        }

        // Add child to new_spine
        self.add_child_to_spine(new_spine, child_idx);

        // Mark old spine dirty too
        self.cache.mark_dirty(old_spine);
    }

    /// Number of allocated slots.
    #[inline]
    pub fn slot_count(&self) -> usize {
        self.cache.len()
    }

    /// All spine indices.
    pub fn spine_indices(&self) -> Vec<usize> {
        self.spine_map.keys().copied().collect()
    }

    /// Re-declare a spine with a new set of children.
    /// Clears the old child→spine reverse mappings and rebuilds them.
    /// Marks the spine dirty.
    pub fn redeclare_spine(&mut self, spine_idx: usize, new_children: Vec<usize>) {
        // Remove old reverse mappings for this spine
        if let Some(old_children) = self.spine_map.get(&spine_idx) {
            let old: Vec<usize> = old_children.clone();
            for child in old {
                if let Some(spines) = self.child_to_spines.get_mut(&child) {
                    spines.retain(|&s| s != spine_idx);
                }
            }
        }

        // Install new forward + reverse mappings
        for &child in &new_children {
            self.child_to_spines
                .entry(child)
                .or_default()
                .push(spine_idx);
        }
        self.spine_map.insert(spine_idx, new_children);
        self.cache.mark_dirty(spine_idx);
    }
}
