//! DnSpineCache — SpineCache with DN addressing for DTO compatibility.
//!
//! Wraps `SpineCache` with a bidirectional `PackedDn ↔ cache slot` mapping.
//! All graph traversal DTOs (CogRedis, BitpackedCsr, Substrate) can address
//! nodes by DN through this single canonical backing store.
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────┐
//! │  DnSpineCache                                                  │
//! │  ┌────────────┐  ┌─────────────────┐  ┌──────────────────┐   │
//! │  │ SpineCache │←→│ dn_to_idx (Map) │←→│ idx_to_dn (Vec)  │   │
//! │  └────────────┘  └─────────────────┘  └──────────────────┘   │
//! │                                                                │
//! │  insert(dn, meta, content) → cache slot                       │
//! │  read(dn) → &Container                                        │
//! │  read_content(dn) → &Container                                │
//! │  children_dns(dn) → Vec<PackedDn>                             │
//! │  dirty_dns() → Vec<PackedDn>                                  │
//! └────────────────────────────────────────────────────────────────┘
//! ```

use std::collections::HashMap;

use super::Container;
use super::adjacency::{PackedDn, InlineEdgeView, InlineEdgeViewMut, CsrOverflowView};
use super::cache::CacheError;
use super::meta::MetaView;
use super::record::CogRecord;
use super::search::belichtungsmesser;
use super::spine::SpineCache;

// ============================================================================
// CONSTANTS
// ============================================================================

/// Distance threshold for plasticity pruning: child too far from recomputed spine.
pub const PRUNE_THRESHOLD: u32 = 5500;

/// Distance threshold for plasticity consolidation: sibling spines too close.
pub const CONSOLIDATION_THRESHOLD: u32 = 800;

// ============================================================================
// DnSpineCache
// ============================================================================

/// SpineCache with DN addressing layer.
///
/// The canonical in-memory graph backing store. CogRedis, BitpackedCsr, and
/// Substrate all delegate their graph operations here.
pub struct DnSpineCache {
    /// Core SpineCache (unchanged).
    pub cache: SpineCache,

    /// DN → cache slot mapping.
    dn_to_idx: HashMap<PackedDn, DnSlot>,

    /// cache slot → DN reverse mapping.
    idx_to_dn: Vec<Option<PackedDn>>,

    /// Labels by DN (codebook or explicit).
    labels: HashMap<PackedDn, String>,

    /// Dirty DN set — tracks which DNs have been modified since last flush.
    dirty_dns: Vec<PackedDn>,
}

impl std::fmt::Debug for DnSpineCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DnSpineCache")
            .field("node_count", &self.dn_to_idx.len())
            .field("slot_count", &self.cache.slot_count())
            .field("dirty_count", &self.dirty_dns.len())
            .finish()
    }
}

/// A DN's slot pair in the cache: meta container + content container.
#[derive(Clone, Copy, Debug)]
pub struct DnSlot {
    /// Cache index of the metadata container (Container 0).
    pub meta_idx: usize,
    /// Cache index of the content container (Container 1).
    pub content_idx: usize,
    /// Whether this DN is a spine node.
    pub is_spine: bool,
}

impl DnSpineCache {
    /// Create a new DnSpineCache with the given initial capacity.
    pub fn new(initial_capacity: usize) -> Self {
        Self {
            cache: SpineCache::new(initial_capacity * 2),
            dn_to_idx: HashMap::with_capacity(initial_capacity),
            idx_to_dn: Vec::with_capacity(initial_capacity * 2),
            labels: HashMap::new(),
            dirty_dns: Vec::new(),
        }
    }

    /// Number of DN-addressed nodes.
    pub fn node_count(&self) -> usize {
        self.dn_to_idx.len()
    }

    /// Check if a DN exists in the cache.
    pub fn contains(&self, dn: PackedDn) -> bool {
        self.dn_to_idx.contains_key(&dn)
    }

    // ========================================================================
    // INSERT
    // ========================================================================

    /// Insert a node with the given DN, metadata container, and content container.
    /// Returns the DnSlot. If the DN already exists, it is overwritten.
    pub fn insert(&mut self, dn: PackedDn, meta: &Container, content: &Container) -> DnSlot {
        if let Some(existing) = self.dn_to_idx.get(&dn).copied() {
            // Overwrite existing slot
            let _ = self.cache.cache.write(existing.meta_idx, meta);
            let _ = self.cache.cache.write(existing.content_idx, content);
            self.mark_dn_dirty(dn);
            return existing;
        }

        // Allocate new slots
        let meta_idx = self.cache.cache.push(meta)
            .expect("push meta should not fail for non-zero container");
        let content_idx = self.cache.cache.push(content)
            .expect("push content should not fail for non-zero container");

        // Grow reverse mapping
        while self.idx_to_dn.len() <= meta_idx.max(content_idx) {
            self.idx_to_dn.push(None);
        }
        self.idx_to_dn[meta_idx] = Some(dn);
        self.idx_to_dn[content_idx] = Some(dn);

        let slot = DnSlot {
            meta_idx,
            content_idx,
            is_spine: false,
        };
        self.dn_to_idx.insert(dn, slot);
        self.mark_dn_dirty(dn);

        // Auto-register as child of parent DN (if parent is a spine)
        if let Some(parent_slot) = dn
            .parent()
            .and_then(|p| self.dn_to_idx.get(&p).copied())
            .filter(|s| s.is_spine)
        {
            self.cache.add_child_to_spine(parent_slot.content_idx, content_idx);
        }

        slot
    }

    /// Insert a node from a CogRecord.
    pub fn insert_record(&mut self, dn: PackedDn, record: &CogRecord) -> DnSlot {
        let content = if record.content.is_empty() {
            &Container::zero()
        } else {
            &record.content[0]
        };
        let slot = self.insert(dn, &record.meta, content);

        // Store DN address in meta
        let meta_idx = slot.meta_idx;
        // We need to write the DN into word 0 of meta
        // Safe: we just wrote this slot
        let meta_container = self.cache.cache.read(meta_idx).clone();
        let mut words = meta_container.words;
        words[0] = dn.raw();
        let updated = Container { words };
        let _ = self.cache.cache.write(meta_idx, &updated);

        slot
    }

    /// Declare a DN as a spine node.
    /// The spine's content slot becomes the XOR-fold of its children.
    pub fn declare_spine(&mut self, dn: PackedDn, child_dns: &[PackedDn]) {
        let slot = match self.dn_to_idx.get_mut(&dn) {
            Some(s) => {
                s.is_spine = true;
                *s
            }
            None => {
                // Create a spine node with zero containers
                let s = self.insert(dn, &Container::zero(), &Container::zero());
                if let Some(entry) = self.dn_to_idx.get_mut(&dn) {
                    entry.is_spine = true;
                }
                s
            }
        };

        // Collect child content indices
        let child_indices: Vec<usize> = child_dns.iter()
            .filter_map(|cdn| self.dn_to_idx.get(cdn).map(|s| s.content_idx))
            .collect();

        self.cache.declare_spine(slot.content_idx, child_indices);
    }

    // ========================================================================
    // READ
    // ========================================================================

    /// Get the slot for a DN.
    pub fn slot(&self, dn: PackedDn) -> Option<DnSlot> {
        self.dn_to_idx.get(&dn).copied()
    }

    /// Read the metadata container for a DN.
    pub fn read_meta(&self, dn: PackedDn) -> Option<&Container> {
        self.dn_to_idx.get(&dn)
            .map(|slot| self.cache.read(slot.meta_idx))
    }

    /// Read the content container for a DN.
    pub fn read_content(&self, dn: PackedDn) -> Option<&Container> {
        self.dn_to_idx.get(&dn)
            .map(|slot| self.cache.read(slot.content_idx))
    }

    /// Read the content container for a DN, recomputing if it's a spine.
    pub fn read_content_fresh(&mut self, dn: PackedDn) -> Option<&Container> {
        let slot = self.dn_to_idx.get(&dn).copied()?;
        if slot.is_spine {
            let _ = self.cache.read_spine(slot.content_idx);
        }
        Some(self.cache.read(slot.content_idx))
    }

    /// Read metadata view for a DN.
    pub fn meta_view(&self, dn: PackedDn) -> Option<MetaView<'_>> {
        self.dn_to_idx.get(&dn)
            .map(|slot| MetaView::new(&self.cache.read(slot.meta_idx).words))
    }

    /// Get the inline edges from Container 0 metadata.
    pub fn inline_edges(&self, dn: PackedDn) -> Option<InlineEdgeView<'_>> {
        self.dn_to_idx.get(&dn)
            .map(|slot| InlineEdgeView::new(&self.cache.read(slot.meta_idx).words))
    }

    /// Get the CSR overflow edges from Container 0 metadata.
    pub fn overflow_edges(&self, dn: PackedDn) -> Option<CsrOverflowView<'_>> {
        self.dn_to_idx.get(&dn)
            .map(|slot| CsrOverflowView::new(&self.cache.read(slot.meta_idx).words))
    }

    /// Get a label for a DN.
    pub fn label(&self, dn: PackedDn) -> Option<&str> {
        self.labels.get(&dn).map(|s| s.as_str())
    }

    /// Set a label for a DN.
    pub fn set_label(&mut self, dn: PackedDn, label: &str) {
        self.labels.insert(dn, label.to_string());
        self.mark_dn_dirty(dn);
    }

    // ========================================================================
    // WRITE / MUTATE
    // ========================================================================

    /// Write a metadata container for a DN. Marks dirty.
    pub fn write_meta(&mut self, dn: PackedDn, meta: &Container) -> Result<(), CacheError> {
        let slot = self.dn_to_idx.get(&dn)
            .ok_or(CacheError::OutOfBounds { idx: 0, len: 0 })?;
        self.cache.cache.write(slot.meta_idx, meta)?;
        self.mark_dn_dirty(dn);
        Ok(())
    }

    /// Write a content container for a DN. If it's a child of a spine, propagates dirty.
    pub fn write_content(&mut self, dn: PackedDn, content: &Container) -> Result<(), CacheError> {
        let slot = self.dn_to_idx.get(&dn)
            .ok_or(CacheError::OutOfBounds { idx: 0, len: 0 })?;
        self.cache.write_child(slot.content_idx, content)?;
        self.mark_dn_dirty(dn);
        Ok(())
    }

    /// Add an inline edge to a DN's metadata container.
    /// Returns the edge index, or None if the edge table is full.
    pub fn add_inline_edge(&mut self, dn: PackedDn, verb: u8, target_hint: u8) -> Option<usize> {
        let slot = self.dn_to_idx.get(&dn).copied()?;
        let mut meta = self.cache.read(slot.meta_idx).clone();
        let result = {
            let mut view = InlineEdgeViewMut::new(&mut meta.words);
            view.add(super::adjacency::InlineEdge { verb, target_hint })
        };
        if result.is_some() {
            let _ = self.cache.cache.write(slot.meta_idx, &meta);
            self.mark_dn_dirty(dn);
        }
        result
    }

    /// Remove an inline edge from a DN's metadata container.
    pub fn remove_inline_edge(&mut self, dn: PackedDn, verb: u8, target_hint: u8) -> bool {
        let slot = match self.dn_to_idx.get(&dn).copied() {
            Some(s) => s,
            None => return false,
        };
        let mut meta = self.cache.read(slot.meta_idx).clone();
        let removed = {
            let mut view = InlineEdgeViewMut::new(&mut meta.words);
            view.remove(verb, target_hint)
        };
        if removed {
            let _ = self.cache.cache.write(slot.meta_idx, &meta);
            self.mark_dn_dirty(dn);
        }
        removed
    }

    // ========================================================================
    // TREE TOPOLOGY QUERIES
    // ========================================================================

    /// Get the child DNs of a DN (from spine mapping).
    pub fn children_dns(&self, dn: PackedDn) -> Vec<PackedDn> {
        let slot = match self.dn_to_idx.get(&dn) {
            Some(s) => s,
            None => return Vec::new(),
        };
        self.cache.spine_children(slot.content_idx)
            .iter()
            .filter_map(|&child_idx| {
                if child_idx < self.idx_to_dn.len() {
                    self.idx_to_dn[child_idx]
                } else {
                    None
                }
            })
            .collect()
    }

    /// Walk from a DN to root, collecting the path.
    pub fn walk_to_root(&self, dn: PackedDn) -> Vec<PackedDn> {
        let mut path = Vec::new();
        let mut current = Some(dn);
        while let Some(cur) = current {
            path.push(cur);
            current = cur.parent().filter(|p| self.contains(*p));
        }
        path
    }

    /// Get all DNs in the subtree rooted at the given DN.
    pub fn subtree(&self, root: PackedDn) -> Vec<PackedDn> {
        let mut result = Vec::new();
        let mut stack = vec![root];
        while let Some(current) = stack.pop() {
            if self.contains(current) {
                result.push(current);
                for child in self.children_dns(current) {
                    stack.push(child);
                }
            }
        }
        result
    }

    /// Find the closest spine to a given container using Belichtungsmesser.
    pub fn find_closest_spine(&self, content: &Container) -> Option<(PackedDn, u32)> {
        let mut best: Option<(PackedDn, u32)> = None;
        for (&dn, slot) in &self.dn_to_idx {
            if !slot.is_spine {
                continue;
            }
            let spine_data = self.cache.read(slot.content_idx);
            let estimate = belichtungsmesser(spine_data, content);
            match &best {
                None => best = Some((dn, estimate)),
                Some((_, best_dist)) if estimate < *best_dist => {
                    best = Some((dn, estimate));
                }
                _ => {}
            }
        }
        best
    }

    /// Merge two sibling spines. The children of `absorbed_dn` are reparented
    /// under `survivor_dn`. The merged content is set on the survivor.
    pub fn merge_spines(
        &mut self,
        survivor_dn: PackedDn,
        absorbed_dn: PackedDn,
        merged_content: &Container,
    ) {
        // Get children of absorbed spine
        let absorbed_children = self.children_dns(absorbed_dn);

        // Reparent children from absorbed to survivor
        if let (Some(absorbed_slot), Some(survivor_slot)) = (
            self.dn_to_idx.get(&absorbed_dn).copied(),
            self.dn_to_idx.get(&survivor_dn).copied(),
        ) {
            for child_dn in &absorbed_children {
                if let Some(child_slot) = self.dn_to_idx.get(child_dn).copied() {
                    self.cache.reparent(
                        child_slot.content_idx,
                        absorbed_slot.content_idx,
                        survivor_slot.content_idx,
                    );
                }
            }

            // Update survivor's content
            let _ = self.cache.cache.write(survivor_slot.content_idx, merged_content);
            self.cache.cache.mark_dirty(survivor_slot.content_idx);
        }

        self.mark_dn_dirty(survivor_dn);
        self.mark_dn_dirty(absorbed_dn);
    }

    // ========================================================================
    // DIRTY TRACKING
    // ========================================================================

    /// Mark a DN as dirty (modified since last flush).
    fn mark_dn_dirty(&mut self, dn: PackedDn) {
        if !self.dirty_dns.contains(&dn) {
            self.dirty_dns.push(dn);
        }
    }

    /// Get all dirty DNs.
    pub fn dirty_dns(&self) -> &[PackedDn] {
        &self.dirty_dns
    }

    /// Clear all dirty flags.
    pub fn clear_dirty(&mut self) {
        self.dirty_dns.clear();
        self.cache.cache.clear_all_dirty();
    }

    /// All DNs in the cache.
    pub fn all_dns(&self) -> Vec<PackedDn> {
        self.dn_to_idx.keys().copied().collect()
    }

    /// Iterate over (DN, DnSlot) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&PackedDn, &DnSlot)> {
        self.dn_to_idx.iter()
    }

    /// Hamming distance between two DNs' content containers.
    pub fn hamming(&self, a: PackedDn, b: PackedDn) -> Option<u32> {
        let ca = self.read_content(a)?;
        let cb = self.read_content(b)?;
        Some(ca.hamming(cb))
    }

    /// Belichtungsmesser estimate between two DNs' content containers.
    pub fn quick_distance(&self, a: PackedDn, b: PackedDn) -> Option<u32> {
        let ca = self.read_content(a)?;
        let cb = self.read_content(b)?;
        Some(belichtungsmesser(ca, cb))
    }
}

// ============================================================================
// LANCE PERSISTENCE STUBS
// ============================================================================

/// Lance table schema for dn_tree.
///
/// ```text
/// dn_tree.lance:
///   dn:        UInt64                 — PackedDn, primary key
///   is_spine:  Boolean
///   depth:     UInt8
///   parent_dn: UInt64 (nullable)
///   meta:      FixedSizeBinary(1024)  — Container 0
///   content:   FixedSizeBinary(1024)  — Container 1
///   label:     Utf8 (nullable)
///   rung:      UInt8
///   qidx:      UInt8
///   updated_at: UInt64 (epoch ms)
/// ```
pub struct DnTreeSchema;

impl DnTreeSchema {
    /// Column names for the dn_tree Lance table.
    pub const DN: &'static str = "dn";
    pub const IS_SPINE: &'static str = "is_spine";
    pub const DEPTH: &'static str = "depth";
    pub const PARENT_DN: &'static str = "parent_dn";
    pub const META: &'static str = "meta";
    pub const CONTENT: &'static str = "content";
    pub const LABEL: &'static str = "label";
    pub const RUNG: &'static str = "rung";
    pub const QIDX: &'static str = "qidx";
    pub const UPDATED_AT: &'static str = "updated_at";
}

impl DnSpineCache {
    /// Convert dirty containers to Arrow-compatible byte vectors.
    /// Returns (dn, is_spine, depth, parent_dn, meta_bytes, content_bytes, label, rung, qidx).
    pub fn dirty_records(&self) -> Vec<DnRecord> {
        self.dirty_dns.iter()
            .filter_map(|&dn| {
                let slot = self.dn_to_idx.get(&dn)?;
                let meta = self.cache.read(slot.meta_idx);
                let content = self.cache.read(slot.content_idx);
                let meta_view = MetaView::new(&meta.words);

                Some(DnRecord {
                    dn: dn.raw(),
                    is_spine: slot.is_spine,
                    depth: dn.depth(),
                    parent_dn: dn.parent().map(|p| p.raw()),
                    meta_bytes: *meta.as_bytes(),
                    content_bytes: *content.as_bytes(),
                    label: self.labels.get(&dn).cloned(),
                    rung: meta_view.rung_level(),
                    qidx: 0, // Default; MetaView doesn't expose qidx directly
                })
            })
            .collect()
    }

    /// Hydrate from pre-fetched records (e.g., from Lance query results).
    /// This inserts nodes without marking them dirty.
    pub fn hydrate(&mut self, records: &[DnRecord]) {
        for rec in records {
            let dn = PackedDn(rec.dn);
            let meta = Container::from_bytes(&rec.meta_bytes);
            let content = Container::from_bytes(&rec.content_bytes);

            // Insert without dirty-marking
            let meta_idx = self.cache.cache.push(&meta)
                .unwrap_or_else(|_| {
                    // On zero container, push a zero anyway (hydration)
                    self.cache.cache.push_spine_slot()
                });
            let content_idx = self.cache.cache.push(&content)
                .unwrap_or_else(|_| self.cache.cache.push_spine_slot());

            while self.idx_to_dn.len() <= meta_idx.max(content_idx) {
                self.idx_to_dn.push(None);
            }
            self.idx_to_dn[meta_idx] = Some(dn);
            self.idx_to_dn[content_idx] = Some(dn);

            let slot = DnSlot {
                meta_idx,
                content_idx,
                is_spine: rec.is_spine,
            };
            self.dn_to_idx.insert(dn, slot);

            if let Some(ref label) = rec.label {
                self.labels.insert(dn, label.clone());
            }
        }
    }
}

/// A serializable record for Lance persistence.
#[derive(Clone, Debug)]
pub struct DnRecord {
    pub dn: u64,
    pub is_spine: bool,
    pub depth: u8,
    pub parent_dn: Option<u64>,
    pub meta_bytes: [u8; super::CONTAINER_BYTES],
    pub content_bytes: [u8; super::CONTAINER_BYTES],
    pub label: Option<String>,
    pub rung: u8,
    pub qidx: u8,
}
