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
//! │                 │  0x01: SQL           0x09: Qualia ops                     │
//! │                 │  0x02: Neo4j/Cypher  0x0A: Memory ops                     │
//! │                 │  0x03: GraphQL       0x0B: Learning ops                   │
//! │                 │  0x04: NARS          0x0C: Reserved                       │
//! │                 │  0x05: Causal        0x0D: Reserved                       │
//! │                 │  0x06: Meta          0x0E: Reserved                       │
//! │                 │  0x07: Verbs         0x0F: Reserved                       │
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

use std::time::Instant;

// =============================================================================
// ADDRESS CONSTANTS (8-bit prefix : 8-bit slot)
// =============================================================================

/// Fingerprint words (10K bits = 156 × 64-bit words)
pub const FINGERPRINT_WORDS: usize = 156;

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

/// Surface compartments (16 available)
pub const PREFIX_LANCE: u8 = 0x00;     // Lance/Kuzu - vector ops
pub const PREFIX_SQL: u8 = 0x01;       // SQL ops
pub const PREFIX_CYPHER: u8 = 0x02;    // Neo4j/Cypher ops
pub const PREFIX_GRAPHQL: u8 = 0x03;   // GraphQL ops
pub const PREFIX_NARS: u8 = 0x04;      // NARS inference
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

/// The 16 surface compartments
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SurfaceCompartment {
    /// 0x00: Lance/Kuzu - vector search, traversal
    Lance = 0x00,
    /// 0x01: SQL - relational operations
    Sql = 0x01,
    /// 0x02: Neo4j/Cypher - property graph
    Cypher = 0x02,
    /// 0x03: GraphQL - query language
    GraphQL = 0x03,
    /// 0x04: NARS - inference operations
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
}

impl BindNode {
    pub fn new(fingerprint: [u64; FINGERPRINT_WORDS]) -> Self {
        Self {
            fingerprint,
            label: None,
            qidx: 0,
            access_count: 0,
            payload: None,
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
    
    #[inline(always)]
    pub fn touch(&mut self) {
        self.access_count = self.access_count.saturating_add(1);
    }
}

impl Default for BindNode {
    fn default() -> Self {
        Self::new([0u64; FINGERPRINT_WORDS])
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
            nodes,
            context: ChunkContext::Concepts,
            next_fluid: (PREFIX_FLUID_START, 0),
            next_node: (PREFIX_NODE_START, 0),
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
}
