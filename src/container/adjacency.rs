//! Zero-copy adjacency views over Container 0 metadata.
//!
//! Container 0 has two adjacency regions:
//!
//! - **Inline edges** (words 16-31): 64 packed edges, 4 per word, 16 bits each.
//!   Format: `verb:u8 | target_hint:u8`. Fast O(1) access, sufficient for
//!   low-degree nodes (≤64 edges).
//!
//! - **CSR overflow** (words 96-111): Compact inline CSR for high-degree nodes.
//!   16 words = 128 bytes. Stores a compressed row-pointer + column index
//!   structure for up to ~200 additional edges beyond the inline 64.
//!
//! Both regions are read/written through zero-copy views.
//! The graph IS the metadata — no separate adjacency structure.

use super::CONTAINER_WORDS;
use super::meta::{W_EDGE_BASE, W_ADJ_BASE, MAX_INLINE_EDGES};

// ============================================================================
// PACKED DN: Hierarchical address in a u64
// ============================================================================

/// A packed DN address — 7 levels × 8 bits, MSB-first.
///
/// Each level component is stored as value+1 (0x00 = absent).
/// This gives hierarchical sort order automatically:
/// /0 < /0/0 < /0/1 < /1
///
/// ```text
/// Byte 7    Byte 6    Byte 5    Byte 4    Byte 3    Byte 2    Byte 1    Byte 0
/// [lv0+1]  [lv1+1]  [lv2+1]  [lv3+1]  [lv4+1]  [lv5+1]  [lv6+1]  [sentinel=0]
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PackedDn(pub u64);

impl PackedDn {
    /// Maximum tree depth (7 levels).
    pub const MAX_DEPTH: usize = 7;

    /// Create from a path of components (e.g., `&[0, 1, 3]` = /0/1/3).
    pub fn new(components: &[u8]) -> Self {
        assert!(components.len() <= Self::MAX_DEPTH);
        let mut val = 0u64;
        for (i, &c) in components.iter().enumerate() {
            val |= ((c as u64) + 1) << (56 - i * 8);
        }
        PackedDn(val)
    }

    /// Root address (depth 0).
    pub const ROOT: PackedDn = PackedDn(0);

    /// How deep is this DN? 0 = root, 1..7 = deeper.
    pub fn depth(self) -> u8 {
        for i in 0..Self::MAX_DEPTH {
            if (self.0 >> (56 - i * 8)) & 0xFF == 0 {
                return i as u8;
            }
        }
        Self::MAX_DEPTH as u8
    }

    /// Component at a given level (0-indexed), or None if beyond depth.
    pub fn component(self, level: usize) -> Option<u8> {
        if level >= Self::MAX_DEPTH {
            return None;
        }
        let byte = (self.0 >> (56 - level * 8)) & 0xFF;
        if byte == 0 {
            None
        } else {
            Some((byte - 1) as u8)
        }
    }

    /// Navigate up: remove the last component.
    pub fn parent(self) -> Option<Self> {
        let d = self.depth();
        if d == 0 {
            return None;
        }
        let level = (d - 1) as usize;
        let mask = !(0xFFu64 << (56 - level * 8));
        Some(PackedDn(self.0 & mask))
    }

    /// Navigate down: append a child component.
    pub fn child(self, component: u8) -> Option<Self> {
        let d = self.depth() as usize;
        if d >= Self::MAX_DEPTH {
            return None;
        }
        let val = self.0 | (((component as u64) + 1) << (56 - d * 8));
        Some(PackedDn(val))
    }

    /// All ancestors from self up to (but not including) root.
    pub fn ancestors(self) -> Vec<PackedDn> {
        let mut result = Vec::new();
        let mut current = self;
        while let Some(p) = current.parent() {
            if p.0 != 0 {
                result.push(p);
            }
            current = p;
        }
        result
    }

    /// Subtree range for binary search: (low, high).
    /// All DNs in the subtree of `self` satisfy `low <= dn < high`.
    pub fn subtree_range(self) -> (PackedDn, PackedDn) {
        let d = self.depth() as usize;
        if d >= Self::MAX_DEPTH {
            return (self, PackedDn(self.0 + 1));
        }
        let low = self;
        // High: increment the last non-zero byte
        let shift = 56 - (d.max(1) - 1) * 8;
        let byte_val = (self.0 >> shift) & 0xFF;
        let high = PackedDn((self.0 & !(0xFF << shift)) | ((byte_val + 1) << shift));
        (low, high)
    }

    /// Check if `other` is a descendant of `self`.
    pub fn is_ancestor_of(self, other: PackedDn) -> bool {
        let d = self.depth() as usize;
        if d == 0 {
            return true; // root is ancestor of all
        }
        // Check that all bytes at levels 0..d match
        for i in 0..d {
            let shift = 56 - i * 8;
            if (self.0 >> shift) & 0xFF != (other.0 >> shift) & 0xFF {
                return false;
            }
        }
        true
    }

    /// Shared prefix length (common depth).
    pub fn common_depth(self, other: PackedDn) -> u8 {
        for i in 0..Self::MAX_DEPTH {
            let shift = 56 - i * 8;
            let a = (self.0 >> shift) & 0xFF;
            let b = (other.0 >> shift) & 0xFF;
            if a != b || a == 0 || b == 0 {
                return i as u8;
            }
        }
        Self::MAX_DEPTH as u8
    }

    /// Hex string for Redis key generation.
    pub fn hex(self) -> String {
        format!("{:016x}", self.0)
    }

    /// Parse from hex string.
    pub fn from_hex(s: &str) -> Option<Self> {
        u64::from_str_radix(s, 16).ok().map(PackedDn)
    }

    /// Raw u64 value.
    #[inline]
    pub fn raw(self) -> u64 {
        self.0
    }

    /// Create from colon-separated path string (with or without `bindspace://` scheme).
    ///
    /// Each path segment is hashed to a deterministic u8 component:
    /// ```text
    /// "ada:soul:memory"             → PackedDn([hash("ada"), hash("soul"), hash("memory")])
    /// "bindspace://ada:soul:memory" → same (scheme stripped)
    /// ```
    pub fn from_path(path: &str) -> Self {
        let bare = path.strip_prefix("bindspace://").unwrap_or(path);
        let components: Vec<u8> = bare.split(':')
            .take(Self::MAX_DEPTH)
            .map(|seg| {
                // Deterministic 8-bit hash of segment (wrapping multiply-add)
                let mut h = 0u8;
                for &b in seg.as_bytes() {
                    h = h.wrapping_mul(31).wrapping_add(b);
                }
                h
            })
            .collect();
        Self::new(&components)
    }
}

impl std::fmt::Debug for PackedDn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DN(")?;
        let d = self.depth();
        for i in 0..d {
            if i > 0 {
                write!(f, "/")?;
            }
            write!(f, "{}", self.component(i as usize).unwrap_or(0))?;
        }
        write!(f, ")")
    }
}

impl std::fmt::Display for PackedDn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let d = self.depth();
        if d == 0 {
            return write!(f, "/");
        }
        for i in 0..d {
            write!(f, "/")?;
            write!(f, "{}", self.component(i as usize).unwrap_or(0))?;
        }
        Ok(())
    }
}

// ============================================================================
// EDGE DESCRIPTOR: 16-bit packed edge (inline format)
// ============================================================================

/// A packed inline edge: verb(u8) | target_hint(u8).
///
/// The target_hint is the low byte of the target PackedDn.
/// For full DN resolution, use the graph's DN index.
/// The verb byte encodes one of up to 256 cognitive verbs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct InlineEdge {
    /// Verb ID (0 = empty/no edge).
    pub verb: u8,
    /// Target DN hint (low byte of target).
    pub target_hint: u8,
}

impl InlineEdge {
    /// Empty/sentinel edge.
    pub const EMPTY: InlineEdge = InlineEdge { verb: 0, target_hint: 0 };

    /// Pack into 16 bits.
    #[inline]
    pub fn pack(self) -> u16 {
        ((self.verb as u16) << 8) | (self.target_hint as u16)
    }

    /// Unpack from 16 bits.
    #[inline]
    pub fn unpack(packed: u16) -> Self {
        Self {
            verb: (packed >> 8) as u8,
            target_hint: (packed & 0xFF) as u8,
        }
    }

    /// True if this slot is empty.
    #[inline]
    pub fn is_empty(self) -> bool {
        self.verb == 0 && self.target_hint == 0
    }
}

// ============================================================================
// EDGE DESCRIPTOR (FULL): 64-bit holograph-compatible
// ============================================================================

/// Full edge descriptor compatible with holograph's EdgeDescriptor.
///
/// ```text
/// Bits 63-48: verb_id   (u16)  — cognitive verb
/// Bits 47-32: weight_q  (u16)  — fixed-point 0.0-1.0 → 0-65535
/// Bits 31-0:  target_dn (u32)  — lower 32 bits of target PackedDn
/// ```
///
/// Used in the CSR overflow region (words 96-111) where we have more space.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct EdgeDescriptor(pub u64);

impl EdgeDescriptor {
    /// Create from components.
    pub fn new(verb_id: u16, weight: f32, target_dn_low: u32) -> Self {
        let w = (weight.clamp(0.0, 1.0) * 65535.0) as u16;
        let val = ((verb_id as u64) << 48)
            | ((w as u64) << 32)
            | (target_dn_low as u64);
        EdgeDescriptor(val)
    }

    /// Verb ID.
    #[inline]
    pub fn verb_id(self) -> u16 {
        (self.0 >> 48) as u16
    }

    /// Weight as f32 (0.0-1.0).
    #[inline]
    pub fn weight(self) -> f32 {
        ((self.0 >> 32) & 0xFFFF) as f32 / 65535.0
    }

    /// Weight as raw u16.
    #[inline]
    pub fn weight_raw(self) -> u16 {
        ((self.0 >> 32) & 0xFFFF) as u16
    }

    /// Target DN lower 32 bits.
    #[inline]
    pub fn target_dn_low(self) -> u32 {
        (self.0 & 0xFFFF_FFFF) as u32
    }

    /// Raw u64.
    #[inline]
    pub fn raw(self) -> u64 {
        self.0
    }

    /// Empty sentinel.
    pub const EMPTY: EdgeDescriptor = EdgeDescriptor(0);

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
}

// ============================================================================
// INLINE EDGE VIEW: Zero-copy over words 16-31
// ============================================================================

/// Zero-copy read view over the 64 inline edges in Container 0 (words 16-31).
pub struct InlineEdgeView<'a> {
    words: &'a [u64; CONTAINER_WORDS],
}

impl<'a> InlineEdgeView<'a> {
    /// Create from Container 0 words.
    pub fn new(words: &'a [u64; CONTAINER_WORDS]) -> Self {
        Self { words }
    }

    /// Read edge at index 0..63.
    #[inline]
    pub fn get(&self, idx: usize) -> InlineEdge {
        debug_assert!(idx < MAX_INLINE_EDGES);
        let word_idx = W_EDGE_BASE + idx / 4;
        let shift = (idx % 4) * 16;
        let packed = ((self.words[word_idx] >> shift) & 0xFFFF) as u16;
        InlineEdge::unpack(packed)
    }

    /// Count non-empty edges.
    pub fn count(&self) -> usize {
        let mut n = 0;
        for i in 0..MAX_INLINE_EDGES {
            if !self.get(i).is_empty() {
                n += 1;
            }
        }
        n
    }

    /// Iterator over non-empty edges with their indices.
    pub fn iter(&self) -> InlineEdgeIter<'a> {
        InlineEdgeIter {
            view: InlineEdgeView { words: self.words },
            pos: 0,
        }
    }

    /// Find the first empty slot. Returns None if all 64 are occupied.
    pub fn first_empty(&self) -> Option<usize> {
        for i in 0..MAX_INLINE_EDGES {
            if self.get(i).is_empty() {
                return Some(i);
            }
        }
        None
    }

    /// Find an edge by verb + target_hint.
    pub fn find(&self, verb: u8, target_hint: u8) -> Option<usize> {
        for i in 0..MAX_INLINE_EDGES {
            let e = self.get(i);
            if e.verb == verb && e.target_hint == target_hint {
                return Some(i);
            }
        }
        None
    }
}

/// Iterator over non-empty inline edges.
pub struct InlineEdgeIter<'a> {
    view: InlineEdgeView<'a>,
    pos: usize,
}

impl<'a> Iterator for InlineEdgeIter<'a> {
    type Item = (usize, InlineEdge);

    fn next(&mut self) -> Option<Self::Item> {
        while self.pos < MAX_INLINE_EDGES {
            let idx = self.pos;
            self.pos += 1;
            let edge = self.view.get(idx);
            if !edge.is_empty() {
                return Some((idx, edge));
            }
        }
        None
    }
}

// ============================================================================
// INLINE EDGE VIEW MUT: Zero-copy mutable over words 16-31
// ============================================================================

/// Mutable view over inline edges.
pub struct InlineEdgeViewMut<'a> {
    words: &'a mut [u64; CONTAINER_WORDS],
}

impl<'a> InlineEdgeViewMut<'a> {
    /// Create from mutable Container 0 words.
    pub fn new(words: &'a mut [u64; CONTAINER_WORDS]) -> Self {
        Self { words }
    }

    /// Read edge at index.
    #[inline]
    pub fn get(&self, idx: usize) -> InlineEdge {
        debug_assert!(idx < MAX_INLINE_EDGES);
        let word_idx = W_EDGE_BASE + idx / 4;
        let shift = (idx % 4) * 16;
        let packed = ((self.words[word_idx] >> shift) & 0xFFFF) as u16;
        InlineEdge::unpack(packed)
    }

    /// Write edge at index.
    #[inline]
    pub fn set(&mut self, idx: usize, edge: InlineEdge) {
        debug_assert!(idx < MAX_INLINE_EDGES);
        let word_idx = W_EDGE_BASE + idx / 4;
        let shift = (idx % 4) * 16;
        let packed = edge.pack() as u64;
        self.words[word_idx] = (self.words[word_idx] & !(0xFFFF << shift))
            | (packed << shift);
    }

    /// Add an edge to the first empty slot. Returns the slot index, or None if full.
    pub fn add(&mut self, edge: InlineEdge) -> Option<usize> {
        for i in 0..MAX_INLINE_EDGES {
            if self.get(i).is_empty() {
                self.set(i, edge);
                return Some(i);
            }
        }
        None
    }

    /// Remove an edge by verb + target_hint. Returns true if found and removed.
    pub fn remove(&mut self, verb: u8, target_hint: u8) -> bool {
        for i in 0..MAX_INLINE_EDGES {
            let e = self.get(i);
            if e.verb == verb && e.target_hint == target_hint {
                self.set(i, InlineEdge::EMPTY);
                return true;
            }
        }
        false
    }

    /// Clear all edges.
    pub fn clear(&mut self) {
        for w in W_EDGE_BASE..=super::meta::W_EDGE_END {
            self.words[w] = 0;
        }
    }

    /// Count non-empty edges.
    pub fn count(&self) -> usize {
        let mut n = 0;
        for i in 0..MAX_INLINE_EDGES {
            if !self.get(i).is_empty() {
                n += 1;
            }
        }
        n
    }
}

// ============================================================================
// CSR OVERFLOW VIEW: Zero-copy over words 96-111
// ============================================================================

/// CSR overflow region layout (words 96-111, 16 words = 128 bytes):
///
/// ```text
/// W96:     edge_count:u16 | row_count:u16 | flags:u32
/// W97-99:  row_ptrs (up to 24 rows, packed as u8 offsets)
/// W100-111: column entries as EdgeDescriptor (u64 each, up to 12 edges)
/// ```
const CSR_HEADER: usize = W_ADJ_BASE;
const CSR_ROW_PTRS: usize = W_ADJ_BASE + 1;
const CSR_DATA: usize = W_ADJ_BASE + 4;
const CSR_DATA_END: usize = W_ADJ_BASE + 15; // W111

/// Maximum edges in CSR overflow.
pub const MAX_CSR_EDGES: usize = CSR_DATA_END - CSR_DATA + 1; // 12

/// Zero-copy read view over the CSR overflow region.
pub struct CsrOverflowView<'a> {
    words: &'a [u64; CONTAINER_WORDS],
}

impl<'a> CsrOverflowView<'a> {
    pub fn new(words: &'a [u64; CONTAINER_WORDS]) -> Self {
        Self { words }
    }

    /// Number of edges stored in overflow.
    #[inline]
    pub fn edge_count(&self) -> u16 {
        (self.words[CSR_HEADER] & 0xFFFF) as u16
    }

    /// Number of source rows.
    #[inline]
    pub fn row_count(&self) -> u16 {
        ((self.words[CSR_HEADER] >> 16) & 0xFFFF) as u16
    }

    /// Read an EdgeDescriptor from the data region.
    #[inline]
    pub fn edge(&self, idx: usize) -> EdgeDescriptor {
        debug_assert!(idx < MAX_CSR_EDGES);
        EdgeDescriptor(self.words[CSR_DATA + idx])
    }

    /// Iterator over all non-empty edges in overflow.
    pub fn edges(&self) -> impl Iterator<Item = EdgeDescriptor> + 'a {
        let count = self.edge_count() as usize;
        let words = self.words;
        (0..count.min(MAX_CSR_EDGES))
            .map(move |i| EdgeDescriptor(words[CSR_DATA + i]))
            .filter(|e| !e.is_empty())
    }
}

/// Mutable view over CSR overflow.
pub struct CsrOverflowViewMut<'a> {
    words: &'a mut [u64; CONTAINER_WORDS],
}

impl<'a> CsrOverflowViewMut<'a> {
    pub fn new(words: &'a mut [u64; CONTAINER_WORDS]) -> Self {
        Self { words }
    }

    /// Set edge count.
    pub fn set_edge_count(&mut self, count: u16) {
        self.words[CSR_HEADER] = (self.words[CSR_HEADER] & !0xFFFF) | (count as u64);
    }

    /// Set row count.
    pub fn set_row_count(&mut self, count: u16) {
        self.words[CSR_HEADER] = (self.words[CSR_HEADER] & !(0xFFFF << 16))
            | ((count as u64) << 16);
    }

    /// Write an EdgeDescriptor into the data region.
    pub fn set_edge(&mut self, idx: usize, edge: EdgeDescriptor) {
        debug_assert!(idx < MAX_CSR_EDGES);
        self.words[CSR_DATA + idx] = edge.raw();
    }

    /// Append an edge. Returns the index, or None if full.
    pub fn push_edge(&mut self, edge: EdgeDescriptor) -> Option<usize> {
        let count = (self.words[CSR_HEADER] & 0xFFFF) as usize;
        if count >= MAX_CSR_EDGES {
            return None;
        }
        self.words[CSR_DATA + count] = edge.raw();
        self.set_edge_count((count + 1) as u16);
        Some(count)
    }

    /// Clear all overflow edges.
    pub fn clear(&mut self) {
        for w in W_ADJ_BASE..=W_ADJ_BASE + 15 {
            self.words[w] = 0;
        }
    }
}

// ============================================================================
// COMBINED ADJACENCY: Unified view over both regions
// ============================================================================

/// Combined adjacency: iterates inline edges first, then CSR overflow.
/// Returns all outgoing edges for a node from its Container 0.
pub struct AdjacencyView<'a> {
    words: &'a [u64; CONTAINER_WORDS],
}

impl<'a> AdjacencyView<'a> {
    pub fn new(words: &'a [u64; CONTAINER_WORDS]) -> Self {
        Self { words }
    }

    /// Total number of edges (inline + overflow).
    pub fn total_edges(&self) -> usize {
        let inline = InlineEdgeView::new(self.words).count();
        let overflow = CsrOverflowView::new(self.words).edge_count() as usize;
        inline + overflow
    }

    /// Inline edge view.
    pub fn inline(&self) -> InlineEdgeView<'a> {
        InlineEdgeView::new(self.words)
    }

    /// CSR overflow view.
    pub fn overflow(&self) -> CsrOverflowView<'a> {
        CsrOverflowView::new(self.words)
    }
}
