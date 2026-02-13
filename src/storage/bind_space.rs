//! Universal Bind Space - The DTO That All Languages Hit
//!
//! # 8-bit Prefix : 8-bit Address Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                      PREFIX (8-bit) : ADDRESS (8-bit)                       │
//! ├─────────────────┬───────────────────────────────────────────────────────────┤
//! │  0x00-0x0F:XX   │  SURFACE (16 prefixes × 256 = 4,096)                      │
//! │                 │  0x00: Lance/Kuzu    0x08: Concepts                       │
//! │                 │  0x01: SQL/CQL       0x09: Qualia ops                     │
//! │                 │       (0x00-0x7F: SQL, 0x80-0xFF: CQL)                    │
//! │                 │  0x02: Cypher/GQL    0x0A: Memory ops                     │
//! │                 │       (0x00-0x7F: Cypher, 0x80-0xFF: GQL)                 │
//! │                 │  0x03: GraphQL       0x0B: Learning ops                   │
//! │                 │  0x04: NARS/ACT-R    0x0C: Agents (crewai)               │
//! │                 │       (0x00-0x7F: NARS, 0x80-0xFF: ACT-R)                 │
//! │                 │  0x05: Causal        0x0D: Thinking Styles               │
//! │                 │  0x06: Meta          0x0E: Blackboard                    │
//! │                 │  0x07: Verbs         0x0F: A2A Routing                   │
//! ├─────────────────┼───────────────────────────────────────────────────────────┤
//! │  0x10-0x7F:XX   │  FLUID (112 prefixes × 256 = 28,672)                      │
//! │                 │  Edges + Context selector + Working memory                │
//! ├─────────────────┼───────────────────────────────────────────────────────────┤
//! │  0x80-0xFF:XX   │  NODES (128 prefixes × 256 = 32,768)                      │
//! │                 │  THE UNIVERSAL BIND SPACE                                 │
//! │                 │  All languages hit this. Any syntax. Same addresses.      │
//! └─────────────────┴───────────────────────────────────────────────────────────┘
//! ```
//!
//! # Why 8-bit + 8-bit?
//!
//! ```text
//! Operation          HashMap (16-bit)    Array index (8+8)
//! ─────────────────────────────────────────────────────────
//! Hash compute       ~20 cycles          0
//! Bucket lookup      ~10-50 cycles       0  
//! Cache miss risk    High                Low (predictable)
//! Branch prediction  Poor                Perfect (3-way)
//! TOTAL              ~30-100 cycles      ~3-5 cycles
//! ```
//!
//! Works on ANY CPU: No AVX-512, no SIMD, no special instructions.
//! Just shift, mask, array index. Even works on embedded/WASM.

use std::collections::HashMap;

use crate::container::{Container, CONTAINER_WORDS, MetaView, MetaViewMut};
use crate::container::adjacency::PackedDn;

// =============================================================================
// ADDRESS CONSTANTS (8-bit prefix : 8-bit slot)
// =============================================================================

/// Fingerprint words (16K bits = 256 × 64-bit words)
pub const FINGERPRINT_WORDS: usize = 256;

/// Slots per chunk (2^8 = 256)
pub const CHUNK_SIZE: usize = 256;

// -----------------------------------------------------------------------------
// SURFACE: 16 prefixes (0x00-0x0F) × 256 = 4,096 addresses
// -----------------------------------------------------------------------------

/// Surface prefix range
pub const PREFIX_SURFACE_START: u8 = 0x00;
pub const PREFIX_SURFACE_END: u8 = 0x0F;
pub const SURFACE_PREFIXES: usize = 16;
pub const SURFACE_SIZE: usize = 4096;  // 16 × 256

/// Surface compartments (16 prefixes, some with slot subdivision)
pub const PREFIX_LANCE: u8 = 0x00;     // Lance/Kuzu - vector ops
pub const PREFIX_SQL: u8 = 0x01;       // SQL/CQL (columnar languages)
pub const PREFIX_CQL: u8 = 0x01;       // CQL shares prefix with SQL (slot 0x80+)
pub const PREFIX_CYPHER: u8 = 0x02;    // Cypher/GQL (property graph languages)
pub const PREFIX_GQL: u8 = 0x02;       // GQL shares prefix with Cypher (slot 0x80+)
pub const PREFIX_GRAPHQL: u8 = 0x03;   // GraphQL (schema-first, distinct paradigm)
pub const PREFIX_NARS: u8 = 0x04;      // NARS/ACT-R (cognitive architectures)
pub const PREFIX_ACTR: u8 = 0x04;      // ACT-R shares prefix with NARS (slot 0x80+)
pub const PREFIX_CAUSAL: u8 = 0x05;    // Causal reasoning (Pearl)
pub const PREFIX_META: u8 = 0x06;      // Meta-cognition
pub const PREFIX_VERBS: u8 = 0x07;     // Verbs (CAUSES, BECOMES...)
pub const PREFIX_CONCEPTS: u8 = 0x08;  // Core concepts/types
pub const PREFIX_QUALIA: u8 = 0x09;    // Qualia operations
pub const PREFIX_MEMORY: u8 = 0x0A;    // Memory operations
pub const PREFIX_LEARNING: u8 = 0x0B;  // Learning operations
pub const PREFIX_RESERVED_C: u8 = 0x0C;
pub const PREFIX_RESERVED_D: u8 = 0x0D;
pub const PREFIX_RESERVED_E: u8 = 0x0E;
pub const PREFIX_RESERVED_F: u8 = 0x0F;

// --- Orchestration prefixes (crewai feature) ---
// These overlay the reserved 0x0C-0x0F surface addresses.
// Slot subdivision: 0x00-0x7F primary, 0x80-0xFF secondary (same as SQL/Cypher)
pub const PREFIX_AGENTS: u8 = 0x0C;       // Agent registry (cards, capabilities, goals)
pub const PREFIX_THINKING: u8 = 0x0D;     // Thinking style templates (YAML → FieldModulation)
pub const PREFIX_BLACKBOARD: u8 = 0x0E;   // Per-agent blackboard state (ice-caked awareness)
pub const PREFIX_A2A: u8 = 0x0F;          // Agent-to-Agent message routing channels

// Slot subdivision boundary for shared prefixes
// Slots 0x00-0x7F: primary language (SQL, Cypher, NARS)
// Slots 0x80-0xFF: secondary language (CQL, GQL, ACT-R)
pub const SLOT_SUBDIVISION: u8 = 0x80;

// -----------------------------------------------------------------------------
// FLUID: 112 prefixes (0x10-0x7F) × 256 = 28,672 addresses
// -----------------------------------------------------------------------------

pub const PREFIX_FLUID_START: u8 = 0x10;
pub const PREFIX_FLUID_END: u8 = 0x7F;
pub const FLUID_PREFIXES: usize = 112;  // 0x7F - 0x10 + 1
pub const FLUID_SIZE: usize = 28672;    // 112 × 256

// -----------------------------------------------------------------------------
// NODES: 128 prefixes (0x80-0xFF) × 256 = 32,768 addresses
// -----------------------------------------------------------------------------

pub const PREFIX_NODE_START: u8 = 0x80;
pub const PREFIX_NODE_END: u8 = 0xFF;
pub const NODE_PREFIXES: usize = 128;   // 0xFF - 0x80 + 1
pub const NODE_SIZE: usize = 32768;     // 128 × 256

/// Total addressable
pub const TOTAL_ADDRESSES: usize = 65536;  // 256 × 256

// =============================================================================
// ADDRESS TYPE
// =============================================================================

/// 16-bit address as prefix:slot
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Addr(pub u16);

impl Addr {
    /// Create from prefix and slot
    #[inline(always)]
    pub fn new(prefix: u8, slot: u8) -> Self {
        Self(((prefix as u16) << 8) | (slot as u16))
    }
    
    /// Get prefix (high byte)
    #[inline(always)]
    pub fn prefix(self) -> u8 {
        (self.0 >> 8) as u8
    }
    
    /// Get slot (low byte)
    #[inline(always)]
    pub fn slot(self) -> u8 {
        (self.0 & 0xFF) as u8
    }
    
    /// Check if in surface (prefix 0x00-0x0F)
    #[inline(always)]
    pub fn is_surface(self) -> bool {
        self.prefix() <= PREFIX_SURFACE_END
    }
    
    /// Check if in fluid zone (prefix 0x10-0x7F)
    #[inline(always)]
    pub fn is_fluid(self) -> bool {
        let p = self.prefix();
        p >= PREFIX_FLUID_START && p <= PREFIX_FLUID_END
    }
    
    /// Check if in node space (prefix 0x80-0xFF)
    #[inline(always)]
    pub fn is_node(self) -> bool {
        self.prefix() >= PREFIX_NODE_START
    }
    
    /// Get surface compartment (0x00-0x0F) or None
    #[inline(always)]
    pub fn surface_compartment(self) -> Option<SurfaceCompartment> {
        SurfaceCompartment::from_prefix(self.prefix())
    }
}

impl From<u16> for Addr {
    fn from(v: u16) -> Self {
        Self(v)
    }
}

impl From<Addr> for u16 {
    fn from(a: Addr) -> Self {
        a.0
    }
}

// =============================================================================
// SURFACE COMPARTMENTS (16 available, 0x00-0x0F)
// =============================================================================

/// The 16 surface compartments (some with slot subdivision for related languages)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SurfaceCompartment {
    /// 0x00: Lance/Kuzu - vector search, traversal
    Lance = 0x00,
    /// 0x01: SQL/CQL (columnar languages)
    /// Slots 0x00-0x7F: SQL ops, 0x80-0xFF: CQL ops
    Sql = 0x01,
    /// 0x02: Cypher/GQL (property graph languages)
    /// Slots 0x00-0x7F: Cypher ops, 0x80-0xFF: GQL ops
    Cypher = 0x02,
    /// 0x03: GraphQL - schema-first query language (distinct paradigm)
    GraphQL = 0x03,
    /// 0x04: NARS/ACT-R (cognitive architectures)
    /// Slots 0x00-0x7F: NARS ops, 0x80-0xFF: ACT-R ops
    Nars = 0x04,
    /// 0x05: Causal - Pearl's ladder
    Causal = 0x05,
    /// 0x06: Meta - higher-order thinking
    Meta = 0x06,
    /// 0x07: Verbs - CAUSES, BECOMES, etc.
    Verbs = 0x07,
    /// 0x08: Concepts - core types
    Concepts = 0x08,
    /// 0x09: Qualia - felt quality ops
    Qualia = 0x09,
    /// 0x0A: Memory - memory operations
    Memory = 0x0A,
    /// 0x0B: Learning - learning operations
    Learning = 0x0B,
    /// 0x0C-0x0F: Reserved
    Reserved = 0x0C,
}

/// Sublanguage within a shared prefix (slot-based discrimination)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sublanguage {
    // Columnar (prefix 0x01)
    Sql,
    Cql,  // Cassandra Query Language
    // Graph (prefix 0x02)
    Cypher,
    Gql,  // ISO Graph Query Language
    // Cognitive (prefix 0x04)
    Nars,
    ActR, // ACT-R cognitive architecture
}

impl SurfaceCompartment {
    pub fn prefix(self) -> u8 {
        self as u8
    }

    pub fn addr(self, slot: u8) -> Addr {
        Addr::new(self as u8, slot)
    }

    pub fn from_prefix(prefix: u8) -> Option<Self> {
        match prefix {
            0x00 => Some(Self::Lance),
            0x01 => Some(Self::Sql),
            0x02 => Some(Self::Cypher),
            0x03 => Some(Self::GraphQL),
            0x04 => Some(Self::Nars),
            0x05 => Some(Self::Causal),
            0x06 => Some(Self::Meta),
            0x07 => Some(Self::Verbs),
            0x08 => Some(Self::Concepts),
            0x09 => Some(Self::Qualia),
            0x0A => Some(Self::Memory),
            0x0B => Some(Self::Learning),
            0x0C..=0x0F => Some(Self::Reserved),
            _ => None,
        }
    }

    /// Check if this compartment has sublanguages (slot-based subdivision)
    pub fn has_sublanguages(self) -> bool {
        matches!(self, Self::Sql | Self::Cypher | Self::Nars)
    }

    /// Determine sublanguage from address (for shared prefixes)
    /// Returns None for non-subdivided compartments
    pub fn sublanguage_from_addr(addr: Addr) -> Option<Sublanguage> {
        let prefix = addr.prefix();
        let slot = addr.slot();
        let is_secondary = slot >= SLOT_SUBDIVISION;

        match prefix {
            PREFIX_SQL => Some(if is_secondary { Sublanguage::Cql } else { Sublanguage::Sql }),
            PREFIX_CYPHER => Some(if is_secondary { Sublanguage::Gql } else { Sublanguage::Cypher }),
            PREFIX_NARS => Some(if is_secondary { Sublanguage::ActR } else { Sublanguage::Nars }),
            _ => None,
        }
    }
}

// =============================================================================
// CHUNK CONTEXT (What node space means)
// =============================================================================

/// Context that defines how node space is interpreted
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ChunkContext {
    /// Concept space - abstract types and categories
    #[default]
    Concepts,
    /// Memory space - episodic memories
    Memories,
    /// Codebook space - learned patterns
    Codebook,
    /// Meta-awareness - self-model, introspection
    MetaAwareness,
    /// Extended addressing for overflow
    Extended(u8),
}

// =============================================================================
// BIND NODE - Universal content container
// =============================================================================

/// A node in the bind space
///
/// This is what ALL query languages read/write.
#[derive(Clone)]
pub struct BindNode {
    /// 10K-bit fingerprint
    pub fingerprint: [u64; FINGERPRINT_WORDS],
    /// Human-readable label
    pub label: Option<String>,
    /// Qualia index (0-255)
    pub qidx: u8,
    /// Access count
    pub access_count: u32,
    /// Optional payload
    pub payload: Option<Vec<u8>>,

    // =========================================================================
    // DN TREE FIELDS (for zero-copy hierarchical traversal)
    // =========================================================================

    /// Parent node address (None = root). O(1) upward traversal.
    pub parent: Option<Addr>,
    /// Tree depth (0 = root, max 255)
    pub depth: u8,
    /// Access rung (R0=public, R9=soul-level)
    pub rung: u8,
    /// Sigma encoding (reasoning depth Σ)
    pub sigma: u8,
    /// Whether this node is a spine (cluster centroid) in the DN tree.
    pub is_spine: bool,
}

impl BindNode {
    pub fn new(fingerprint: [u64; FINGERPRINT_WORDS]) -> Self {
        Self {
            fingerprint,
            label: None,
            qidx: 0,
            access_count: 0,
            payload: None,
            parent: None,
            depth: 0,
            rung: 0,
            sigma: 0,
            is_spine: false,
        }
    }

    pub fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }

    pub fn with_qidx(mut self, qidx: u8) -> Self {
        self.qidx = qidx;
        self
    }

    pub fn with_parent(mut self, parent: Addr, depth: u8) -> Self {
        self.parent = Some(parent);
        self.depth = depth;
        self
    }

    pub fn with_rung(mut self, rung: u8) -> Self {
        self.rung = rung.min(9); // R0-R9
        self
    }

    pub fn with_sigma(mut self, sigma: u8) -> Self {
        self.sigma = sigma;
        self
    }

    pub fn with_spine(mut self, is_spine: bool) -> Self {
        self.is_spine = is_spine;
        self
    }

    #[inline(always)]
    pub fn touch(&mut self) {
        self.access_count = self.access_count.saturating_add(1);
    }

    // =========================================================================
    // ZERO-COPY CONTAINER VIEWS (Phase 2 — BindSpace Unification)
    // =========================================================================

    /// Container copy of the first 128 words (meta half).
    /// Interpretation: structured metadata (NARS, edges, rung, graph metrics, qualia).
    #[inline]
    pub fn meta_container(&self) -> Container {
        let mut c = Container::zero();
        c.words.copy_from_slice(&self.fingerprint[..CONTAINER_WORDS]);
        c
    }

    /// Container copy of the second 128 words (content half).
    /// Interpretation: semantic content fingerprint for search / Hamming distance.
    #[inline]
    pub fn content_container(&self) -> Container {
        let mut c = Container::zero();
        c.words.copy_from_slice(&self.fingerprint[CONTAINER_WORDS..]);
        c
    }

    /// Direct reference to meta words (first 128 words) for MetaView.
    #[inline(always)]
    pub fn meta_words(&self) -> &[u64; CONTAINER_WORDS] {
        <&[u64; CONTAINER_WORDS]>::try_from(&self.fingerprint[..CONTAINER_WORDS]).unwrap()
    }

    /// Direct reference to content words (second 128 words).
    #[inline(always)]
    pub fn content_words(&self) -> &[u64; CONTAINER_WORDS] {
        <&[u64; CONTAINER_WORDS]>::try_from(&self.fingerprint[CONTAINER_WORDS..]).unwrap()
    }

    /// Mutable reference to meta words (first 128 words) for MetaViewMut.
    #[inline(always)]
    pub fn meta_words_mut(&mut self) -> &mut [u64; CONTAINER_WORDS] {
        <&mut [u64; CONTAINER_WORDS]>::try_from(&mut self.fingerprint[..CONTAINER_WORDS]).unwrap()
    }

    /// Mutable reference to content words (second 128 words).
    #[inline(always)]
    pub fn content_words_mut(&mut self) -> &mut [u64; CONTAINER_WORDS] {
        <&mut [u64; CONTAINER_WORDS]>::try_from(&mut self.fingerprint[CONTAINER_WORDS..]).unwrap()
    }

    /// NARS truth values from meta W4-7 (through MetaView).
    #[inline]
    pub fn nars(&self) -> (f32, f32) {
        let meta = MetaView::new(self.meta_words());
        (meta.nars_frequency(), meta.nars_confidence())
    }
}

impl Default for BindNode {
    fn default() -> Self {
        Self::new([0u64; FINGERPRINT_WORDS])
    }
}

// =============================================================================
// BITPACKED CSR - Zero-copy edge storage
// =============================================================================

/// Bitpacked CSR (Compressed Sparse Row) for zero-copy edge traversal
///
/// Traditional: `Vec<Vec<usize>>` = 64K vectors × ~24 bytes overhead = 1.5MB+ overhead
/// Bitpacked:   `offsets[64K] + edges[N]` = 128KB + N×4 bytes (if <65K edges per node)
///
/// For DN tree with ~32K nodes and avg 2 children each:
/// - Traditional: ~1.5MB overhead + 64K×2×8 = ~2.5MB
/// - Bitpacked:   128KB + 64K×2 = ~256KB
#[derive(Clone)]
pub struct BitpackedCsr {
    /// Offset into edges array for each address (64K entries)
    /// offsets[addr] = start index, offsets[addr+1] = end index
    offsets: Vec<u32>,
    /// Flat array of target addresses (bitpacked as u16)
    edges: Vec<u16>,
    /// Verb for each edge (parallel to edges)
    verbs: Vec<u16>,
}

impl BitpackedCsr {
    pub fn new() -> Self {
        Self {
            offsets: vec![0u32; TOTAL_ADDRESSES + 1],
            edges: Vec::new(),
            verbs: Vec::new(),
        }
    }

    /// Build CSR from edge list (call once after all edges added)
    pub fn build_from_edges(edges: &[BindEdge]) -> Self {
        // Count edges per source
        let mut counts = vec![0u32; TOTAL_ADDRESSES];
        for edge in edges {
            counts[edge.from.0 as usize] += 1;
        }

        // Compute offsets (prefix sum)
        let mut offsets = vec![0u32; TOTAL_ADDRESSES + 1];
        for i in 0..TOTAL_ADDRESSES {
            offsets[i + 1] = offsets[i] + counts[i];
        }

        // Allocate edge storage
        let total_edges = offsets[TOTAL_ADDRESSES] as usize;
        let mut edge_targets = vec![0u16; total_edges];
        let mut edge_verbs = vec![0u16; total_edges];

        // Fill edges (reset counts as write pointers)
        counts.fill(0);
        for edge in edges {
            let src = edge.from.0 as usize;
            let idx = (offsets[src] + counts[src]) as usize;
            edge_targets[idx] = edge.to.0;
            edge_verbs[idx] = edge.verb.0;
            counts[src] += 1;
        }

        Self {
            offsets,
            edges: edge_targets,
            verbs: edge_verbs,
        }
    }

    /// Zero-copy children access: returns slice of target addresses
    #[inline(always)]
    pub fn children(&self, addr: Addr) -> &[u16] {
        let start = self.offsets[addr.0 as usize] as usize;
        let end = self.offsets[addr.0 as usize + 1] as usize;
        &self.edges[start..end]
    }

    /// Zero-copy children with verb filter
    #[inline(always)]
    pub fn children_via(&self, addr: Addr, verb: Addr) -> impl Iterator<Item = Addr> + '_ {
        let start = self.offsets[addr.0 as usize] as usize;
        let end = self.offsets[addr.0 as usize + 1] as usize;
        let verb_raw = verb.0;

        self.edges[start..end]
            .iter()
            .zip(self.verbs[start..end].iter())
            .filter(move |(_, v)| **v == verb_raw)
            .map(|(e, _)| Addr(*e))
    }

    /// Number of outgoing edges from address
    #[inline(always)]
    pub fn out_degree(&self, addr: Addr) -> usize {
        let start = self.offsets[addr.0 as usize] as usize;
        let end = self.offsets[addr.0 as usize + 1] as usize;
        end - start
    }

    /// Total edges in CSR
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.offsets.len() * 4 + self.edges.len() * 2 + self.verbs.len() * 2
    }
}

// =============================================================================
// BIND EDGE - Connection via verb
// =============================================================================

/// An edge connecting nodes via a verb
#[derive(Clone)]
pub struct BindEdge {
    /// Source node address (0x80-0xFF:XX)
    pub from: Addr,
    /// Target node address (0x80-0xFF:XX)
    pub to: Addr,
    /// Verb address (0x03:XX typically)
    pub verb: Addr,
    /// Bound fingerprint: from ⊗ verb ⊗ to
    pub fingerprint: [u64; FINGERPRINT_WORDS],
    /// Edge weight
    pub weight: f32,
}

impl BindEdge {
    pub fn new(from: Addr, verb: Addr, to: Addr) -> Self {
        Self {
            from,
            to,
            verb,
            fingerprint: [0u64; FINGERPRINT_WORDS],
            weight: 1.0,
        }
    }
    
    /// Bind: compute edge fingerprint via XOR
    pub fn bind(
        &mut self,
        from_fp: &[u64; FINGERPRINT_WORDS],
        verb_fp: &[u64; FINGERPRINT_WORDS],
        to_fp: &[u64; FINGERPRINT_WORDS],
    ) {
        for i in 0..FINGERPRINT_WORDS {
            self.fingerprint[i] = from_fp[i] ^ verb_fp[i] ^ to_fp[i];
        }
    }
    
    /// ABBA unbind: recover unknown from edge + known + verb
    pub fn unbind(
        &self,
        known: &[u64; FINGERPRINT_WORDS],
        verb_fp: &[u64; FINGERPRINT_WORDS],
    ) -> [u64; FINGERPRINT_WORDS] {
        let mut result = [0u64; FINGERPRINT_WORDS];
        for i in 0..FINGERPRINT_WORDS {
            result[i] = self.fingerprint[i] ^ known[i] ^ verb_fp[i];
        }
        result
    }
}

// =============================================================================
// DN INDEX - Bidirectional PackedDn ↔ Addr mapping (Phase 3)
// =============================================================================

/// Bidirectional DN ↔ Addr index. Pure addressing — no data storage.
///
/// Tree structure is IMPLICIT in the PackedDn address:
/// - Parent: `dn.parent()` → O(1) bit masking (chop last component)
/// - Depth:  `dn.depth()`  → count non-zero components
/// - Sibling enumeration: `children(parent_addr)`
///
/// DnIndex stores NO fingerprints, NO containers, NO edges.
/// It is an address book, nothing more. All data lives in BindSpace.
pub struct DnIndex {
    dn_to_addr: HashMap<PackedDn, Addr>,
    addr_to_dn: Vec<Option<PackedDn>>,       // indexed by Addr.0, 65536 entries
    children:   HashMap<u16, Vec<Addr>>,      // parent addr.0 → [child_addrs]
}

impl DnIndex {
    pub fn new() -> Self {
        Self {
            dn_to_addr: HashMap::new(),
            addr_to_dn: vec![None; TOTAL_ADDRESSES],
            children: HashMap::new(),
        }
    }

    /// Register a DN ↔ Addr mapping and update the parent's children list.
    pub fn register(&mut self, dn: PackedDn, addr: Addr) {
        self.dn_to_addr.insert(dn, addr);
        self.addr_to_dn[addr.0 as usize] = Some(dn);

        // Automatically maintain children index using PackedDn::parent()
        if let Some(parent_dn) = dn.parent() {
            if let Some(&parent_addr) = self.dn_to_addr.get(&parent_dn) {
                let children = self.children.entry(parent_addr.0).or_default();
                if !children.contains(&addr) {
                    children.push(addr);
                }
            }
        }
    }

    /// Look up Addr for a given PackedDn.
    pub fn addr_for(&self, dn: PackedDn) -> Option<Addr> {
        self.dn_to_addr.get(&dn).copied()
    }

    /// Look up PackedDn for a given Addr.
    pub fn dn_for(&self, addr: Addr) -> Option<PackedDn> {
        self.addr_to_dn.get(addr.0 as usize).and_then(|o| *o)
    }

    /// Children of addr (for downward tree traversal).
    /// Upward traversal doesn't need this — use PackedDn::parent().
    pub fn children(&self, addr: Addr) -> &[Addr] {
        self.children.get(&addr.0).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Number of registered DN paths.
    pub fn len(&self) -> usize {
        self.dn_to_addr.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.dn_to_addr.is_empty()
    }

    /// Iterate all registered (PackedDn, Addr) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (PackedDn, Addr)> + '_ {
        self.dn_to_addr.iter().map(|(&dn, &addr)| (dn, addr))
    }
}

// =============================================================================
// DIRTY BITSET - 65536-bit tracking (Phase 4)
// =============================================================================

/// Compact 65536-bit set using 1024 u64 words.
struct DirtyBits {
    bits: Vec<u64>,
}

impl DirtyBits {
    fn new() -> Self {
        Self {
            bits: vec![0u64; TOTAL_ADDRESSES / 64],
        }
    }

    #[inline(always)]
    fn set(&mut self, idx: u16) {
        let word = idx as usize / 64;
        let bit = idx as usize % 64;
        self.bits[word] |= 1u64 << bit;
    }

    #[inline(always)]
    fn get(&self, idx: u16) -> bool {
        let word = idx as usize / 64;
        let bit = idx as usize % 64;
        self.bits[word] & (1u64 << bit) != 0
    }

    fn clear(&mut self) {
        for w in self.bits.iter_mut() {
            *w = 0;
        }
    }

    /// Iterate all set bit indices.
    fn iter_ones(&self) -> impl Iterator<Item = u16> + '_ {
        self.bits.iter().enumerate().flat_map(|(wi, &word)| {
            let base = (wi * 64) as u16;
            (0..64u16).filter(move |&bit| word & (1u64 << bit) != 0)
                      .map(move |bit| base + bit)
        })
    }
}

// =============================================================================
// BIND SPACE - The Universal DTO (Array-based storage)
// =============================================================================

/// The Universal Bind Space
///
/// Pure array indexing. No HashMap. No SIMD required.
/// Works on any CPU.
pub struct BindSpace {
    // =========================================================================
    // SURFACES: 16 prefixes (0x00-0x0F) × 256 slots = 4,096 addresses
    // =========================================================================
    
    /// All 16 surface compartments
    surfaces: Vec<Box<[Option<BindNode>; CHUNK_SIZE]>>,
    
    // =========================================================================
    // FLUID: 112 prefixes (0x10-0x7F) × 256 slots = 28,672 addresses
    // =========================================================================
    
    /// Fluid chunks for edge storage + working memory
    fluid: Vec<Box<[Option<BindNode>; CHUNK_SIZE]>>,
    
    /// Edges (separate for efficient traversal)
    edges: Vec<BindEdge>,
    
    /// Edge index: from.0 -> edge indices (CSR-style)
    edge_out: Vec<Vec<usize>>,
    
    /// Edge index: to.0 -> edge indices (reverse CSR)
    edge_in: Vec<Vec<usize>>,

    // =========================================================================
    // BITPACKED CSR (zero-copy, Arrow-friendly adjacency)
    // =========================================================================

    /// Bitpacked CSR for zero-copy traversal (built on demand)
    csr: Option<BitpackedCsr>,
    /// CSR dirty flag (rebuild needed after edge modifications)
    csr_dirty: bool,

    // =========================================================================
    // NODES: 128 prefixes (0x80-0xFF) × 256 slots = 32,768 addresses
    // =========================================================================

    /// Node chunks - THE UNIVERSAL BIND SPACE
    nodes: Vec<Box<[Option<BindNode>; CHUNK_SIZE]>>,

    // =========================================================================
    // STATE
    // =========================================================================

    /// Current context
    context: ChunkContext,

    /// Next fluid slot (prefix, slot)
    next_fluid: (u8, u8),

    /// Next node slot (prefix, slot)
    next_node: (u8, u8),

    // =========================================================================
    // UNIFICATION: DnIndex + dirty tracking (Phases 3-4)
    // =========================================================================

    /// Bidirectional PackedDn ↔ Addr index for DN tree navigation.
    pub dn_index: DnIndex,

    /// Dirty bit per address — tracks modifications since last flush.
    dirty: DirtyBits,
}

impl BindSpace {
    /// Allocate a chunk on heap without stack intermediary
    fn alloc_chunk() -> Box<[Option<BindNode>; CHUNK_SIZE]> {
        // Use vec to allocate on heap, then convert to boxed array
        // This avoids stack allocation of ~320KB per chunk
        let mut v: Vec<Option<BindNode>> = Vec::with_capacity(CHUNK_SIZE);
        for _ in 0..CHUNK_SIZE {
            v.push(None);
        }
        // SAFETY: Vec has exactly CHUNK_SIZE elements
        let boxed_slice = v.into_boxed_slice();
        // Convert Box<[T]> to Box<[T; N]>
        let ptr = Box::into_raw(boxed_slice) as *mut [Option<BindNode>; CHUNK_SIZE];
        unsafe { Box::from_raw(ptr) }
    }

    pub fn new() -> Self {
        // Initialize 16 surface compartments (heap allocated)
        let mut surfaces = Vec::with_capacity(SURFACE_PREFIXES);
        for _ in 0..SURFACE_PREFIXES {
            surfaces.push(Self::alloc_chunk());
        }

        // Initialize 112 fluid chunks (heap allocated)
        let mut fluid = Vec::with_capacity(FLUID_PREFIXES);
        for _ in 0..FLUID_PREFIXES {
            fluid.push(Self::alloc_chunk());
        }

        // Initialize 128 node chunks (heap allocated)
        let mut nodes = Vec::with_capacity(NODE_PREFIXES);
        for _ in 0..NODE_PREFIXES {
            nodes.push(Self::alloc_chunk());
        }

        // Edge indices (64K entries for O(1) lookup)
        let edge_out = vec![Vec::new(); TOTAL_ADDRESSES];
        let edge_in = vec![Vec::new(); TOTAL_ADDRESSES];

        let mut space = Self {
            surfaces,
            fluid,
            edges: Vec::new(),
            edge_out,
            edge_in,
            csr: None,
            csr_dirty: false,
            nodes,
            context: ChunkContext::Concepts,
            next_fluid: (PREFIX_FLUID_START, 0),
            next_node: (PREFIX_NODE_START, 0),
            dn_index: DnIndex::new(),
            dirty: DirtyBits::new(),
        };

        space.init_surfaces();
        space
    }
    
    /// Initialize surfaces with core ops
    fn init_surfaces(&mut self) {
        // Surface 0x00: Lance/Kuzu ops
        let lance_ops = [
            (0x00, "VECTOR_SEARCH"),
            (0x01, "TRAVERSE"),
            (0x02, "RESONATE"),
            (0x03, "HAMMING"),
            (0x04, "BIND"),
            (0x05, "UNBIND"),
            (0x06, "BUNDLE"),
            (0x07, "SIMILARITY"),
            (0x08, "KNN"),
            (0x09, "ANN"),
            (0x0A, "CLUSTER"),
            (0x0B, "QUANTIZE"),
        ];
        for (slot, label) in lance_ops {
            self.surfaces[PREFIX_LANCE as usize][slot] = Some(BindNode::new(label_fingerprint(label)).with_label(label));
        }
        
        // Surface 0x01: SQL ops
        let sql_ops = [
            (0x00, "SELECT"),
            (0x01, "INSERT"),
            (0x02, "UPDATE"),
            (0x03, "DELETE"),
            (0x04, "JOIN"),
            (0x05, "WHERE"),
            (0x06, "GROUP"),
            (0x07, "ORDER"),
        ];
        for (slot, label) in sql_ops {
            self.surfaces[PREFIX_SQL as usize][slot] = Some(BindNode::new(label_fingerprint(label)).with_label(label));
        }
        
        // Surface 0x02: Neo4j/Cypher ops
        let cypher_ops = [
            (0x00, "MATCH"),
            (0x01, "CREATE"),
            (0x02, "MERGE"),
            (0x03, "RETURN"),
            (0x04, "WITH"),
            (0x05, "UNWIND"),
            (0x06, "OPTIONAL_MATCH"),
            (0x07, "DETACH_DELETE"),
        ];
        for (slot, label) in cypher_ops {
            self.surfaces[PREFIX_CYPHER as usize][slot] = Some(BindNode::new(label_fingerprint(label)).with_label(label));
        }
        
        // Surface 0x04: NARS inference ops
        let nars_ops = [
            (0x00, "DEDUCE"),
            (0x01, "ABDUCT"),
            (0x02, "INDUCE"),
            (0x03, "REVISE"),
            (0x04, "CHOICE"),
            (0x05, "EXPECTATION"),
        ];
        for (slot, label) in nars_ops {
            self.surfaces[PREFIX_NARS as usize][slot] = Some(BindNode::new(label_fingerprint(label)).with_label(label));
        }
        
        // Surface 0x05: Causal ops (Pearl's ladder)
        let causal_ops = [
            (0x00, "OBSERVE"),    // Rung 1
            (0x01, "INTERVENE"),  // Rung 2 (do)
            (0x02, "IMAGINE"),    // Rung 3 (counterfactual)
            (0x03, "CAUSE"),
            (0x04, "EFFECT"),
            (0x05, "CONFOUND"),
        ];
        for (slot, label) in causal_ops {
            self.surfaces[PREFIX_CAUSAL as usize][slot] = Some(BindNode::new(label_fingerprint(label)).with_label(label));
        }
        
        // Surface 0x06: Meta-cognition ops
        let meta_ops = [
            (0x00, "REFLECT"),
            (0x01, "ABSTRACT"),
            (0x02, "ANALOGIZE"),
            (0x03, "HYPOTHESIZE"),
            (0x04, "BELIEVE"),
            (0x05, "DOUBT"),
            (0x06, "COUNTERFACT"),
        ];
        for (slot, label) in meta_ops {
            self.surfaces[PREFIX_META as usize][slot] = Some(BindNode::new(label_fingerprint(label)).with_label(label));
        }
        
        // Surface 0x07: Verbs (the Go board verbs)
        let verb_ops = [
            (0x00, "CAUSES"),
            (0x01, "BECOMES"),
            (0x02, "ENABLES"),
            (0x03, "PREVENTS"),
            (0x04, "REQUIRES"),
            (0x05, "IMPLIES"),
            (0x06, "CONTAINS"),
            (0x07, "ACTIVATES"),
            (0x08, "INHIBITS"),
            (0x09, "TRANSFORMS"),
            (0x0A, "RESONATES"),
            (0x0B, "AMPLIFIES"),
            (0x0C, "DAMPENS"),
            (0x0D, "OBSERVES"),
            (0x0E, "REMEMBERS"),
            (0x0F, "FORGETS"),
            (0x10, "SHIFT"),
            (0x11, "LEAP"),
            (0x12, "EMERGE"),
            (0x13, "SUBSIDE"),
            (0x14, "OSCILLATE"),
            (0x15, "CRYSTALLIZE"),
            (0x16, "DISSOLVE"),
            (0x17, "GROUNDS"),
            (0x18, "ABSTRACTS"),
            (0x19, "REFINES"),
            (0x1A, "CONTRADICTS"),
            (0x1B, "SUPPORTS"),
            // DN Tree verbs (0x20-0x2F) for zero-copy hierarchical traversal
            (0x20, "PARENT_OF"),     // Tree edge: parent -> child
            (0x21, "CHILD_OF"),      // Reverse: child -> parent
            (0x22, "SIBLING_OF"),    // Horizontal: sibling <-> sibling
            (0x23, "ANCESTOR_OF"),   // Transitive up
            (0x24, "DESCENDANT_OF"), // Transitive down
            (0x25, "ROOT_OF"),       // Points to tree root
            (0x26, "NEXT_SIBLING"),  // Ordered sibling (for Arrow adjacency)
            (0x27, "PREV_SIBLING"),  // Previous sibling
        ];
        for (slot, label) in verb_ops {
            self.surfaces[PREFIX_VERBS as usize][slot] = Some(BindNode::new(label_fingerprint(label)).with_label(label));
        }
        
        // Surface 0x08: Core concepts
        let concept_ops = [
            (0x00, "ENTITY"),
            (0x01, "RELATION"),
            (0x02, "ATTRIBUTE"),
            (0x03, "EVENT"),
            (0x04, "STATE"),
            (0x05, "PROCESS"),
        ];
        for (slot, label) in concept_ops {
            self.surfaces[PREFIX_CONCEPTS as usize][slot] = Some(BindNode::new(label_fingerprint(label)).with_label(label));
        }
        
        // Surface 0x09: Qualia ops
        let qualia_ops = [
            (0x00, "FEEL"),
            (0x01, "INTUIT"),
            (0x02, "SENSE"),
            (0x03, "VALENCE"),
            (0x04, "AROUSAL"),
            (0x05, "TENSION"),
        ];
        for (slot, label) in qualia_ops {
            self.surfaces[PREFIX_QUALIA as usize][slot] = Some(BindNode::new(label_fingerprint(label)).with_label(label));
        }
        
        // Surface 0x0A: Memory ops
        let memory_ops = [
            (0x00, "STORE"),
            (0x01, "RECALL"),
            (0x02, "FORGET"),
            (0x03, "CONSOLIDATE"),
            (0x04, "ASSOCIATE"),
        ];
        for (slot, label) in memory_ops {
            self.surfaces[PREFIX_MEMORY as usize][slot] = Some(BindNode::new(label_fingerprint(label)).with_label(label));
        }
        
        // Surface 0x0B: Learning ops
        let learning_ops = [
            (0x00, "LEARN"),
            (0x01, "UNLEARN"),
            (0x02, "REINFORCE"),
            (0x03, "GENERALIZE"),
            (0x04, "SPECIALIZE"),
        ];
        for (slot, label) in learning_ops {
            self.surfaces[PREFIX_LEARNING as usize][slot] = Some(BindNode::new(label_fingerprint(label)).with_label(label));
        }
    }
    
    // =========================================================================
    // CORE READ/WRITE (Pure array indexing - 3-5 cycles)
    // =========================================================================
    
    /// Read from any address - THE HOT PATH
    /// 
    /// This is what GET, MATCH, SELECT all become.
    /// Pure array indexing, no hash, no search.
    #[inline(always)]
    pub fn read(&self, addr: Addr) -> Option<&BindNode> {
        let prefix = addr.prefix();
        let slot = addr.slot() as usize;
        
        match prefix {
            // Surface: 0x00-0x0F
            p if p <= PREFIX_SURFACE_END => {
                self.surfaces.get(p as usize).and_then(|c| c[slot].as_ref())
            }
            // Fluid: 0x10-0x7F
            p if p >= PREFIX_FLUID_START && p <= PREFIX_FLUID_END => {
                let chunk = (p - PREFIX_FLUID_START) as usize;
                self.fluid.get(chunk).and_then(|c| c[slot].as_ref())
            }
            // Nodes: 0x80-0xFF
            p if p >= PREFIX_NODE_START => {
                let chunk = (p - PREFIX_NODE_START) as usize;
                self.nodes.get(chunk).and_then(|c| c[slot].as_ref())
            }
            _ => None,
        }
    }
    
    /// Read mutable with touch
    #[inline(always)]
    pub fn read_mut(&mut self, addr: Addr) -> Option<&mut BindNode> {
        let prefix = addr.prefix();
        let slot = addr.slot() as usize;
        
        let node = match prefix {
            // Surface: 0x00-0x0F
            p if p <= PREFIX_SURFACE_END => {
                self.surfaces.get_mut(p as usize).and_then(|c| c[slot].as_mut())
            }
            // Fluid: 0x10-0x7F
            p if p >= PREFIX_FLUID_START && p <= PREFIX_FLUID_END => {
                let chunk = (p - PREFIX_FLUID_START) as usize;
                self.fluid.get_mut(chunk).and_then(|c| c[slot].as_mut())
            }
            // Nodes: 0x80-0xFF
            p if p >= PREFIX_NODE_START => {
                let chunk = (p - PREFIX_NODE_START) as usize;
                self.nodes.get_mut(chunk).and_then(|c| c[slot].as_mut())
            }
            _ => None,
        };
        
        if let Some(n) = node {
            n.touch();
            Some(n)
        } else {
            None
        }
    }
    
    /// Write to node space
    ///
    /// This is what SET, CREATE, INSERT all become.
    pub fn write(&mut self, fingerprint: [u64; FINGERPRINT_WORDS]) -> Addr {
        let (prefix, slot) = self.next_node;
        let addr = Addr::new(prefix, slot);

        // Advance next slot
        self.next_node = if slot == 255 {
            if prefix == PREFIX_NODE_END {
                (PREFIX_NODE_START, 0)  // Wrap
            } else {
                (prefix + 1, 0)
            }
        } else {
            (prefix, slot + 1)
        };

        // Write to chunk
        let chunk = (prefix - PREFIX_NODE_START) as usize;
        if let Some(c) = self.nodes.get_mut(chunk) {
            c[slot as usize] = Some(BindNode::new(fingerprint));
            self.dirty.set(addr.0);
        }

        addr
    }
    
    /// Write with label
    pub fn write_labeled(&mut self, fingerprint: [u64; FINGERPRINT_WORDS], label: &str) -> Addr {
        let addr = self.write(fingerprint);
        if let Some(node) = self.read_mut(addr) {
            node.label = Some(label.to_string());
        }
        addr
    }

    /// Write at specific address (fluid, node, and orchestration surface zones)
    pub fn write_at(&mut self, addr: Addr, fingerprint: [u64; FINGERPRINT_WORDS]) -> bool {
        let prefix = addr.prefix();
        let slot = addr.slot() as usize;

        // Surface zone: allow writes to orchestration prefixes (0x0C-0x0F)
        // but reject writes to pre-initialized query language prefixes (0x00-0x0B)
        if prefix <= PREFIX_SURFACE_END {
            if prefix < PREFIX_AGENTS {
                return false; // Query language surfaces are read-only
            }
            // Orchestration surfaces (agents, thinking, blackboard, a2a)
            if let Some(c) = self.surfaces.get_mut(prefix as usize) {
                let node = BindNode::new(fingerprint);
                c[slot] = Some(node);
                self.dirty.set(addr.0);
                return true;
            }
            return false;
        }

        let node = BindNode::new(fingerprint);

        if prefix >= PREFIX_FLUID_START && prefix <= PREFIX_FLUID_END {
            let chunk = (prefix - PREFIX_FLUID_START) as usize;
            if let Some(c) = self.fluid.get_mut(chunk) {
                c[slot] = Some(node);
                self.dirty.set(addr.0);
                return true;
            }
        } else if prefix >= PREFIX_NODE_START {
            let chunk = (prefix - PREFIX_NODE_START) as usize;
            if let Some(c) = self.nodes.get_mut(chunk) {
                c[slot] = Some(node);
                self.dirty.set(addr.0);
                return true;
            }
        }

        false
    }

    /// Delete from address
    pub fn delete(&mut self, addr: Addr) -> Option<BindNode> {
        let prefix = addr.prefix();
        let slot = addr.slot() as usize;
        
        // Can't delete surfaces
        if prefix <= PREFIX_SURFACE_END {
            return None;
        }
        
        if prefix >= PREFIX_FLUID_START && prefix <= PREFIX_FLUID_END {
            let chunk = (prefix - PREFIX_FLUID_START) as usize;
            self.fluid.get_mut(chunk).and_then(|c| c[slot].take())
        } else if prefix >= PREFIX_NODE_START {
            let chunk = (prefix - PREFIX_NODE_START) as usize;
            self.nodes.get_mut(chunk).and_then(|c| c[slot].take())
        } else {
            None
        }
    }
    
    // =========================================================================
    // EDGE OPERATIONS (CSR-style O(1) lookup)
    // =========================================================================
    
    /// Create an edge
    pub fn link(&mut self, from: Addr, verb: Addr, to: Addr) -> usize {
        let mut edge = BindEdge::new(from, verb, to);

        // Bind fingerprints
        if let (Some(from_node), Some(verb_node), Some(to_node)) =
            (self.read(from), self.read(verb), self.read(to))
        {
            let from_fp = from_node.fingerprint;
            let verb_fp = verb_node.fingerprint;
            let to_fp = to_node.fingerprint;
            edge.bind(&from_fp, &verb_fp, &to_fp);
        }

        let idx = self.edges.len();

        // Update CSR indices
        self.edge_out[from.0 as usize].push(idx);
        self.edge_in[to.0 as usize].push(idx);

        // Mark bitpacked CSR as dirty
        self.csr_dirty = true;

        self.edges.push(edge);
        idx
    }
    
    /// Get outgoing edges (O(1) index lookup)
    #[inline(always)]
    pub fn edges_out(&self, from: Addr) -> impl Iterator<Item = &BindEdge> {
        self.edge_out[from.0 as usize]
            .iter()
            .filter_map(|&i| self.edges.get(i))
    }
    
    /// Get incoming edges (O(1) index lookup)
    #[inline(always)]
    pub fn edges_in(&self, to: Addr) -> impl Iterator<Item = &BindEdge> {
        self.edge_in[to.0 as usize]
            .iter()
            .filter_map(|&i| self.edges.get(i))
    }
    
    /// Traverse: from -> via verb -> targets
    pub fn traverse(&self, from: Addr, verb: Addr) -> Vec<Addr> {
        self.edges_out(from)
            .filter(|e| e.verb == verb)
            .map(|e| e.to)
            .collect()
    }
    
    /// Reverse traverse: sources <- via verb <- to
    pub fn traverse_reverse(&self, to: Addr, verb: Addr) -> Vec<Addr> {
        self.edges_in(to)
            .filter(|e| e.verb == verb)
            .map(|e| e.from)
            .collect()
    }
    
    /// N-hop traversal (Kuzu CSR equivalent)
    pub fn traverse_n_hops(&self, start: Addr, verb: Addr, max_hops: usize) -> Vec<(usize, Addr)> {
        let mut results = Vec::new();
        let mut frontier = vec![start];
        let mut visited = std::collections::HashSet::new();
        visited.insert(start.0);
        
        for hop in 1..=max_hops {
            let mut next_frontier = Vec::new();
            
            for &node in &frontier {
                for target in self.traverse(node, verb) {
                    if visited.insert(target.0) {
                        results.push((hop, target));
                        next_frontier.push(target);
                    }
                }
            }
            
            if next_frontier.is_empty() {
                break;
            }
            frontier = next_frontier;
        }
        
        results
    }

    // =========================================================================
    // ZERO-COPY DN TREE TRAVERSAL
    // =========================================================================

    /// Rebuild bitpacked CSR if dirty
    pub fn rebuild_csr(&mut self) {
        if self.csr_dirty || self.csr.is_none() {
            self.csr = Some(BitpackedCsr::build_from_edges(&self.edges));
            self.csr_dirty = false;
        }
    }

    /// Get CSR reference (rebuilds if needed)
    pub fn csr(&mut self) -> &BitpackedCsr {
        self.rebuild_csr();
        self.csr.as_ref().unwrap()
    }

    /// Zero-copy children slice (returns raw u16 addresses)
    /// Use after rebuild_csr() for zero allocation traversal
    #[inline(always)]
    pub fn children_raw(&self, addr: Addr) -> &[u16] {
        self.csr.as_ref().map(|c| c.children(addr)).unwrap_or(&[])
    }

    /// Zero-copy parent lookup via BindNode.parent field
    #[inline(always)]
    pub fn parent(&self, addr: Addr) -> Option<Addr> {
        self.read(addr).and_then(|n| n.parent)
    }

    /// O(1) ancestors via parent chain (returns iterator, no allocation)
    pub fn ancestors(&self, addr: Addr) -> impl Iterator<Item = Addr> + '_ {
        std::iter::successors(self.parent(addr), |&a| self.parent(a))
    }

    /// Tree depth from BindNode (O(1))
    #[inline(always)]
    pub fn depth(&self, addr: Addr) -> u8 {
        self.read(addr).map(|n| n.depth).unwrap_or(0)
    }

    /// Access rung from BindNode (O(1))
    #[inline(always)]
    pub fn rung(&self, addr: Addr) -> u8 {
        self.read(addr).map(|n| n.rung).unwrap_or(0)
    }

    /// Create DN tree node with parent relationship
    pub fn write_dn_node(
        &mut self,
        fingerprint: [u64; FINGERPRINT_WORDS],
        label: &str,
        parent: Option<Addr>,
        rung: u8,
    ) -> Addr {
        let depth = parent
            .and_then(|p| self.read(p))
            .map(|n| n.depth.saturating_add(1))
            .unwrap_or(0);

        let addr = self.write(fingerprint);
        if let Some(node) = self.read_mut(addr) {
            node.label = Some(label.to_string());
            node.parent = parent;
            node.depth = depth;
            node.rung = rung;
        }

        // Auto-link PARENT_OF edge
        if let Some(parent_addr) = parent {
            if let Some(parent_of) = self.verb("PARENT_OF") {
                self.link(parent_addr, parent_of, addr);
            }
        }

        addr
    }

    /// Parse DN path to address: "Ada:A:soul:identity" -> Addr
    /// Returns None if path not found
    pub fn dn_lookup(&self, path: &str) -> Option<Addr> {
        // Hash the full path to get deterministic address
        let addr = dn_path_to_addr(path);
        // Verify it exists
        if self.read(addr).is_some() {
            Some(addr)
        } else {
            None
        }
    }

    /// O(1) parent path extraction: "Ada:A:soul:identity" -> "Ada:A:soul"
    /// Pure string operation, no lookup needed
    #[inline]
    pub fn dn_parent_path(path: &str) -> Option<&str> {
        path.rfind(':').map(|i| &path[..i])
    }

    /// Create node from DN path, auto-creating parent chain
    pub fn write_dn_path(
        &mut self,
        path: &str,
        fingerprint: [u64; FINGERPRINT_WORDS],
        rung: u8,
    ) -> Addr {
        // Strip bindspace:// scheme if present
        let bare = path.strip_prefix("bindspace://").unwrap_or(path);
        let segments: Vec<&str> = bare.split(':').collect();
        let mut current_parent: Option<Addr> = None;
        let mut current_path = String::new();

        for (i, segment) in segments.iter().enumerate() {
            if i > 0 {
                current_path.push(':');
            }
            current_path.push_str(segment);

            let addr = dn_path_to_addr(&current_path);

            // Create node if it doesn't exist
            if self.read(addr).is_none() {
                let is_final = i == segments.len() - 1;
                let fp = if is_final {
                    fingerprint
                } else {
                    label_fingerprint(&current_path)
                };

                // Write at computed address
                self.write_at(addr, fp);
                if let Some(node) = self.read_mut(addr) {
                    node.label = Some(format!("bindspace://{}", current_path));
                    node.parent = current_parent;
                    node.depth = i as u8;
                    node.rung = rung;
                }

                // Link to parent
                if let Some(parent_addr) = current_parent {
                    if let Some(parent_of) = self.verb("PARENT_OF") {
                        self.link(parent_addr, parent_of, addr);
                    }
                }
            }

            // Register in DnIndex for bidirectional PackedDn ↔ Addr lookup
            let packed = PackedDn::from_path(&current_path);
            self.dn_index.register(packed, addr);

            current_parent = Some(addr);
        }

        current_parent.unwrap()
    }

    /// Zero-copy sibling iteration (same parent)
    pub fn siblings(&self, addr: Addr) -> impl Iterator<Item = Addr> + '_ {
        let parent = self.parent(addr);
        let parent_of = self.verb("PARENT_OF");

        parent
            .into_iter()
            .flat_map(move |p| {
                parent_of
                    .into_iter()
                    .flat_map(move |verb| self.traverse(p, verb))
            })
            .filter(move |&a| a != addr)
    }

    // =========================================================================
    // CONTEXT
    // =========================================================================

    pub fn set_context(&mut self, ctx: ChunkContext) {
        self.context = ctx;
    }
    
    pub fn context(&self) -> ChunkContext {
        self.context
    }
    
    // =========================================================================
    // SURFACE HELPERS
    // =========================================================================
    
    /// Get verb address by name (searches PREFIX_VERBS compartment)
    pub fn verb(&self, name: &str) -> Option<Addr> {
        if let Some(verbs) = self.surfaces.get(PREFIX_VERBS as usize) {
            for slot in 0..CHUNK_SIZE {
                if let Some(node) = &verbs[slot] {
                    if node.label.as_deref() == Some(name) {
                        return Some(Addr::new(PREFIX_VERBS, slot as u8));
                    }
                }
            }
        }
        None
    }
    
    /// Get op address by name from any surface compartment
    pub fn surface_op(&self, compartment: u8, name: &str) -> Option<Addr> {
        if compartment > PREFIX_SURFACE_END {
            return None;
        }
        if let Some(surface) = self.surfaces.get(compartment as usize) {
            for slot in 0..CHUNK_SIZE {
                if let Some(node) = &surface[slot] {
                    if node.label.as_deref() == Some(name) {
                        return Some(Addr::new(compartment, slot as u8));
                    }
                }
            }
        }
        None
    }
    
    /// Get verb fingerprint by address
    pub fn verb_fingerprint(&self, verb: Addr) -> Option<&[u64; FINGERPRINT_WORDS]> {
        self.read(verb).map(|n| &n.fingerprint)
    }
    
    // =========================================================================
    // STATS
    // =========================================================================
    
    pub fn stats(&self) -> BindSpaceStats {
        let surface_count: usize = self.surfaces.iter()
            .map(|s| s.iter().filter(|x| x.is_some()).count())
            .sum();

        let fluid_count: usize = self.fluid.iter()
            .map(|c| c.iter().filter(|x| x.is_some()).count())
            .sum();

        let node_count: usize = self.nodes.iter()
            .map(|c| c.iter().filter(|x| x.is_some()).count())
            .sum();

        BindSpaceStats {
            surface_count,
            fluid_count,
            node_count,
            edge_count: self.edges.len(),
            context: self.context,
        }
    }

    // =========================================================================
    // FLUENT API — Zero-copy content/meta access (Phase 2)
    // =========================================================================

    /// Content Container (search fingerprint, words 128-255).
    /// Returns owned `Container` for Hamming distance, similarity, etc.
    #[inline]
    pub fn content(&self, addr: Addr) -> Option<Container> {
        self.read(addr).map(|n| n.content_container())
    }

    /// Meta view (NARS, edges, rung, etc., words 0-127).
    /// Returns `MetaView` for structured metadata access.
    #[inline]
    pub fn meta(&self, addr: Addr) -> Option<MetaView<'_>> {
        self.read(addr).map(|n| MetaView::new(n.meta_words()))
    }

    /// Resolve a `bindspace://` URI (or bare colon path) to an address.
    pub fn resolve(&self, uri: &str) -> Option<Addr> {
        let packed = PackedDn::from_path(uri);
        self.dn_index.addr_for(packed)
    }

    // =========================================================================
    // DIRTY TRACKING (Phase 4)
    // =========================================================================

    /// Mark address as dirty (modified since last flush).
    #[inline]
    pub fn mark_dirty(&mut self, addr: Addr) {
        self.dirty.set(addr.0);
    }

    /// Get all dirty addresses since last clear.
    pub fn dirty_addrs(&self) -> impl Iterator<Item = Addr> + '_ {
        self.dirty.iter_ones().map(Addr)
    }

    /// Clear all dirty bits.
    pub fn clear_dirty(&mut self) {
        self.dirty.clear();
    }

    // =========================================================================
    // SUBSTRATE METHODS (Phase 2 — BindSpace self-contained API)
    // =========================================================================

    /// Iterate all occupied node addresses with their BindNodes.
    pub fn nodes_iter(&self) -> impl Iterator<Item = (Addr, &BindNode)> + '_ {
        let surfaces = self.surfaces.iter().enumerate().flat_map(|(pi, chunk)| {
            chunk.iter().enumerate().filter_map(move |(si, slot)| {
                slot.as_ref().map(|node| (Addr::new(pi as u8, si as u8), node))
            })
        });
        let fluid = self.fluid.iter().enumerate().flat_map(|(ci, chunk)| {
            let prefix = PREFIX_FLUID_START + ci as u8;
            chunk.iter().enumerate().filter_map(move |(si, slot)| {
                slot.as_ref().map(|node| (Addr::new(prefix, si as u8), node))
            })
        });
        let nodes = self.nodes.iter().enumerate().flat_map(|(ci, chunk)| {
            let prefix = PREFIX_NODE_START + ci as u8;
            chunk.iter().enumerate().filter_map(move |(si, slot)| {
                slot.as_ref().map(|node| (Addr::new(prefix, si as u8), node))
            })
        });
        surfaces.chain(fluid).chain(nodes)
    }

    /// XOR-fold all occupied fingerprints into a single 256-word digest.
    /// Useful for integrity checks and snapshot comparison.
    pub fn hash_all(&self) -> [u64; FINGERPRINT_WORDS] {
        let mut acc = [0u64; FINGERPRINT_WORDS];
        for (_, node) in self.nodes_iter() {
            for (i, word) in node.fingerprint.iter().enumerate() {
                acc[i] ^= word;
            }
        }
        acc
    }

    /// NARS revision: update truth value at address with new evidence.
    /// Reads old values, computes revised truth, writes back.
    pub fn nars_revise(&mut self, addr: Addr, evidence_freq: f32, evidence_conf: f32) {
        if let Some(node) = self.read_mut(addr) {
            // Read old values through immutable MetaView
            let old_freq = {
                let view = MetaView::new(node.meta_words());
                view.nars_frequency()
            };
            let old_conf = {
                let view = MetaView::new(node.meta_words());
                view.nars_confidence()
            };

            // NARS revision: weighted average by confidence
            let total_conf = old_conf + evidence_conf;
            if total_conf > 0.0 {
                let w1 = old_conf / total_conf;
                let w2 = evidence_conf / total_conf;
                let new_freq = w1 * old_freq + w2 * evidence_freq;
                let new_conf = total_conf.min(0.99);

                let mut meta_mut = MetaViewMut::new(node.meta_words_mut());
                meta_mut.set_nars_frequency(new_freq);
                meta_mut.set_nars_confidence(new_conf);
            }
            self.dirty.set(addr.0);
        }
    }
}
impl Default for BindSpace {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct BindSpaceStats {
    pub surface_count: usize,
    pub fluid_count: usize,
    pub node_count: usize,
    pub edge_count: usize,
    pub context: ChunkContext,
}

// =============================================================================
// HELPERS
// =============================================================================

/// Generate fingerprint from label (deterministic)
fn label_fingerprint(label: &str) -> [u64; FINGERPRINT_WORDS] {
    let mut fp = [0u64; FINGERPRINT_WORDS];
    let bytes = label.as_bytes();

    for (i, &b) in bytes.iter().enumerate() {
        let word = i % FINGERPRINT_WORDS;
        let bit = (b as usize * 7 + i * 13) % 64;
        fp[word] |= 1u64 << bit;
    }

    // Spread bits
    for i in 0..FINGERPRINT_WORDS {
        let seed = fp[i];
        fp[(i + 1) % FINGERPRINT_WORDS] ^= seed.rotate_left(17);
        fp[(i + 3) % FINGERPRINT_WORDS] ^= seed.rotate_right(23);
    }

    fp
}

/// Convert DN path to deterministic address
///
/// "Ada:A:soul:identity" -> Addr(0x80+prefix_hash, slot_hash)
///
/// This gives O(1) lookup AND implicit hierarchy:
/// - "Ada:A:soul:identity" -> address X
/// - "Ada:A:soul" (parent) -> different address Y (O(1) string truncate)
pub fn dn_path_to_addr(path: &str) -> Addr {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;

    let mut hasher = DefaultHasher::new();
    path.hash(&mut hasher);
    let hash = hasher.finish();

    // Map to node address space (0x80-0xFF:XX)
    let prefix = PREFIX_NODE_START + ((hash >> 8) as u8 & 0x7F); // 0x80-0xFF
    let slot = (hash & 0xFF) as u8;
    Addr::new(prefix, slot)
}

/// Levenshtein distance for DN path horizontal awareness
/// Used to find siblings with similar names
pub fn dn_levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // Use two rows instead of full matrix
    let mut prev = (0..=n).collect::<Vec<_>>();
    let mut curr = vec![0; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1)
                .min(curr[j - 1] + 1)
                .min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

/// Hamming distance
pub fn hamming_distance(a: &[u64; FINGERPRINT_WORDS], b: &[u64; FINGERPRINT_WORDS]) -> u32 {
    let mut d = 0u32;
    for i in 0..FINGERPRINT_WORDS {
        d += (a[i] ^ b[i]).count_ones();
    }
    d
}

// =============================================================================
// QUERY ADAPTER TRAIT
// =============================================================================

/// Trait for query language adapters
pub trait QueryAdapter {
    fn execute(&self, space: &mut BindSpace, query: &str) -> QueryResult;
}

#[derive(Debug)]
pub struct QueryResult {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<QueryValue>>,
    pub affected: usize,
}

impl QueryResult {
    pub fn empty() -> Self {
        Self { columns: Vec::new(), rows: Vec::new(), affected: 0 }
    }
    
    pub fn single(addr: Addr) -> Self {
        Self {
            columns: vec!["addr".to_string()],
            rows: vec![vec![QueryValue::Addr(addr)]],
            affected: 1,
        }
    }
}

#[derive(Debug, Clone)]
pub enum QueryValue {
    Addr(Addr),
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Fingerprint([u64; FINGERPRINT_WORDS]),
    Null,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_addr_split() {
        let addr = Addr::new(0x80, 0x42);
        assert_eq!(addr.prefix(), 0x80);
        assert_eq!(addr.slot(), 0x42);
        assert_eq!(addr.0, 0x8042);
    }
    
    #[test]
    fn test_surface_compartments() {
        let lance = Addr::new(PREFIX_LANCE, 0x05);
        let sql = Addr::new(PREFIX_SQL, 0x10);
        let meta = Addr::new(PREFIX_META, 0x00);
        let verbs = Addr::new(PREFIX_VERBS, 0x01);
        
        assert!(lance.is_surface());
        assert!(sql.is_surface());
        assert!(meta.is_surface());
        assert!(verbs.is_surface());
        
        assert_eq!(lance.surface_compartment(), Some(SurfaceCompartment::Lance));
        assert_eq!(verbs.surface_compartment(), Some(SurfaceCompartment::Verbs));
    }
    
    #[test]
    fn test_fluid_node_ranges() {
        let fluid = Addr::new(0x50, 0x00);
        let node = Addr::new(0x80, 0x00);
        
        assert!(fluid.is_fluid());
        assert!(!fluid.is_node());
        
        assert!(node.is_node());
        assert!(!node.is_fluid());
    }
    
    #[test]
    fn test_bind_space_surfaces() {
        let space = BindSpace::new();
        
        // Check verbs initialized
        let causes = Addr::new(PREFIX_VERBS, 0x00);
        let node = space.read(causes);
        assert!(node.is_some());
        assert_eq!(node.unwrap().label.as_deref(), Some("CAUSES"));
    }
    
    #[test]
    fn test_write_read() {
        let mut space = BindSpace::new();
        let fp = [42u64; FINGERPRINT_WORDS];
        
        let addr = space.write(fp);
        assert!(addr.is_node());
        
        let node = space.read(addr);
        assert!(node.is_some());
        assert_eq!(node.unwrap().fingerprint, fp);
    }
    
    #[test]
    fn test_link_traverse() {
        let mut space = BindSpace::new();
        
        let a = space.write_labeled([1u64; FINGERPRINT_WORDS], "A");
        let b = space.write_labeled([2u64; FINGERPRINT_WORDS], "B");
        
        let causes = Addr::new(PREFIX_VERBS, 0x00);  // CAUSES
        space.link(a, causes, b);
        
        let targets = space.traverse(a, causes);
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0], b);
    }
    
    #[test]
    fn test_n_hop() {
        let mut space = BindSpace::new();
        
        let a = space.write([1u64; FINGERPRINT_WORDS]);
        let b = space.write([2u64; FINGERPRINT_WORDS]);
        let c = space.write([3u64; FINGERPRINT_WORDS]);
        let d = space.write([4u64; FINGERPRINT_WORDS]);
        
        let causes = Addr::new(PREFIX_VERBS, 0x00);
        space.link(a, causes, b);
        space.link(b, causes, c);
        space.link(c, causes, d);
        
        let results = space.traverse_n_hops(a, causes, 3);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0], (1, b));
        assert_eq!(results[1], (2, c));
        assert_eq!(results[2], (3, d));
    }
    
    #[test]
    fn test_verb_lookup() {
        let space = BindSpace::new();

        let causes = space.verb("CAUSES");
        assert!(causes.is_some());
        assert_eq!(causes.unwrap(), Addr::new(PREFIX_VERBS, 0x00));

        let becomes = space.verb("BECOMES");
        assert!(becomes.is_some());
        assert_eq!(becomes.unwrap(), Addr::new(PREFIX_VERBS, 0x01));
    }

    #[test]
    fn test_tree_verbs() {
        let space = BindSpace::new();

        // Check tree verbs initialized
        let parent_of = space.verb("PARENT_OF");
        assert!(parent_of.is_some());
        assert_eq!(parent_of.unwrap(), Addr::new(PREFIX_VERBS, 0x20));

        let child_of = space.verb("CHILD_OF");
        assert!(child_of.is_some());

        let sibling_of = space.verb("SIBLING_OF");
        assert!(sibling_of.is_some());
    }

    #[test]
    fn test_dn_path_to_addr() {
        // Same path should give same address
        let a1 = dn_path_to_addr("Ada:A:soul:identity");
        let a2 = dn_path_to_addr("Ada:A:soul:identity");
        assert_eq!(a1, a2);

        // Different paths should (likely) give different addresses
        let b = dn_path_to_addr("Ada:A:soul:core");
        assert_ne!(a1, b);

        // Parent path is different
        let parent = dn_path_to_addr("Ada:A:soul");
        assert_ne!(a1, parent);
    }

    #[test]
    fn test_dn_parent_path() {
        // O(1) parent extraction from DN path
        assert_eq!(BindSpace::dn_parent_path("Ada:A:soul:identity"), Some("Ada:A:soul"));
        assert_eq!(BindSpace::dn_parent_path("Ada:A:soul"), Some("Ada:A"));
        assert_eq!(BindSpace::dn_parent_path("Ada:A"), Some("Ada"));
        assert_eq!(BindSpace::dn_parent_path("Ada"), None);
    }

    #[test]
    fn test_dn_levenshtein() {
        // Same string
        assert_eq!(dn_levenshtein("Ada:A:soul", "Ada:A:soul"), 0);

        // One char difference
        assert_eq!(dn_levenshtein("Ada:A:soul:x", "Ada:A:soul:y"), 1);

        // Different lengths
        assert_eq!(dn_levenshtein("Ada", ""), 3);
        assert_eq!(dn_levenshtein("", "Ada"), 3);
    }

    #[test]
    fn test_write_dn_path() {
        let mut space = BindSpace::new();
        let fp = [123u64; FINGERPRINT_WORDS];

        // Create DN path - this should create parent chain
        let leaf = space.write_dn_path("Ada:A:soul:identity", fp, 5);
        assert!(leaf.is_node());

        // Check the node was created
        let node = space.read(leaf);
        assert!(node.is_some());
        let node = node.unwrap();
        assert_eq!(node.rung, 5);
        assert_eq!(node.depth, 3); // Ada(0) -> A(1) -> soul(2) -> identity(3)

        // Check parent exists
        assert!(node.parent.is_some());
        let parent_addr = node.parent.unwrap();
        let parent = space.read(parent_addr);
        assert!(parent.is_some());
        assert_eq!(parent.unwrap().depth, 2);
    }

    #[test]
    fn test_bitpacked_csr() {
        let mut space = BindSpace::new();

        let a = space.write([1u64; FINGERPRINT_WORDS]);
        let b = space.write([2u64; FINGERPRINT_WORDS]);
        let c = space.write([3u64; FINGERPRINT_WORDS]);

        let causes = space.verb("CAUSES").unwrap();
        space.link(a, causes, b);
        space.link(a, causes, c);

        // Rebuild CSR
        space.rebuild_csr();

        // Zero-copy children access
        let children = space.children_raw(a);
        assert_eq!(children.len(), 2);
        assert!(children.contains(&b.0));
        assert!(children.contains(&c.0));

        // Check memory efficiency
        let csr = space.csr.as_ref().unwrap();
        assert!(csr.memory_bytes() < 300_000); // Should be ~260KB vs >1.5MB traditional
    }
}
