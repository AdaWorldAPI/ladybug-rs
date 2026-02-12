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
}
