//! Universal Bind Space - Bitpacked Trio Architecture
//!
//! # 16-bit Address = [u16; 3] Edge Format Foundation
//!
//! This layout enables O(1) graph traversal via the "Bitpacked Trio":
//! ```text
//! [Node_u16, Edge_u16, Node_u16] = 48 bits = 6 bytes
//! ```
//! Aligned to 64-bit boundaries with 16-bit padding/metadata.
//!
//! **DO NOT use u24** - breaks hardware alignment and kills cache density.
//! Scale via Shard_ID, not wider pointers.
//!
//! # 8-bit Prefix : 8-bit Slot Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                      PREFIX (8-bit) : SLOT (8-bit)                          │
//! ├─────────────────┬───────────────────────────────────────────────────────────┤
//! │  0x00-0x0F:XX   │  OPS (16 prefixes × 256 = 4,096)                          │
//! │                 │  Instruction set - prefix alone executes                  │
//! │                 │  ┌─────────────────────────────────────────────────────┐  │
//! │                 │  │ 0x00: Lance/Kuzu    0x08: Concepts                 │  │
//! │                 │  │ 0x01: SQL/CQL       0x09: Qualia ops               │  │
//! │                 │  │ 0x02: Cypher/GQL    0x0A: Memory ops               │  │
//! │                 │  │ 0x03: GraphQL       0x0B: Learning ops             │  │
//! │                 │  │ 0x04: NARS/ACT-R    0x0C: USER RESERVED            │  │
//! │                 │  │ 0x05: Causal        0x0D: USER RESERVED            │  │
//! │                 │  │ 0x06: Meta          0x0E: USER RESERVED            │  │
//! │                 │  │ 0x07: Verbs         0x0F: Learned Styles (256)     │  │
//! │                 │  └─────────────────────────────────────────────────────┘  │
//! ├─────────────────┼───────────────────────────────────────────────────────────┤
//! │  0x10-0x3F:XX   │  FLUID (48 prefixes × 256 = 12,288)                       │
//! │                 │  Registers/Cache - fast transient slots                   │
//! │                 │  Context Crystal, stack frames, working memory            │
//! ├─────────────────┼───────────────────────────────────────────────────────────┤
//! │  0x40-0x7F:XX   │  EDGES (64 prefixes × 256 = 16,384)                       │
//! │                 │  The "Verb" slot in Bitpacked Trio [Node, EDGE, Node]     │
//! │                 │  Relationships: verb + temporal + rung + cognitive        │
//! │                 │  Dn tree relative addressing via Arrow                    │
//! ├─────────────────┼───────────────────────────────────────────────────────────┤
//! │  0x80-0xFF:XX   │  NODES (128 prefixes × 256 = 32,768)                      │
//! │                 │  The "Leaf" slots - active working set in Crystal         │
//! │                 │  Dn tree: 2^4 branching (Tree.Branch.Twig.Leaf)           │
//! │                 │  ±3 layer scent radius, Levenshtein as Hamming distance   │
//! └─────────────────┴───────────────────────────────────────────────────────────┘
//! ```
//!
//! # Bitpacked Trio (2^48) - The Superpower
//!
//! ```text
//! Edge triple: [Source_Node, Relationship, Target_Node]
//!              [   u16     ,    u16     ,    u16     ] = 48 bits
//!
//! With u16 padding: 64 bits total = perfect cache line alignment
//! AVX-512 processes 8 triples per instruction
//! ```
//!
//! # Dn Tree Addressing (Nodes 0x80-0xFF)
//!
//! ```text
//! 16 bits = 4 + 4 + 4 + 4 bits
//!         = Tree.Branch.Twig.Leaf
//!         = 16 × 16 × 16 × 16 = 65,536 (but we use 32K in node zone)
//!
//! Hamming distance between Dn addresses = topological distance
//! XOR + popcount = tree distance for free
//! ```
//!
//! # Why NOT u24
//!
//! - No hardware u24 type → pay for u32 or burn cycles masking
//! - Breaks [u16; 3] alignment → kills AVX-512 throughput  
//! - Scale via Shard_ID: [Shard_u16 | Local_u16] not wider pointers

use std::time::Instant;

// =============================================================================
// ADDRESS CONSTANTS (8-bit prefix : 8-bit slot)
// =============================================================================

/// Fingerprint words (10K bits = 156 × 64-bit words)
pub const FINGERPRINT_WORDS: usize = 156;

/// Slots per chunk (2^8 = 256)
pub const CHUNK_SIZE: usize = 256;

/// Dn tree branching factor (2^4 = 16 per level)
pub const DN_BRANCHING: usize = 16;

// =============================================================================
// OPS ZONE: 0x00-0x0F (4,096 addresses)
// Instruction set - prefix alone executes built-in methods
// =============================================================================

pub const PREFIX_OPS_START: u8 = 0x00;
pub const PREFIX_OPS_END: u8 = 0x0F;
pub const OPS_PREFIXES: usize = 16;
pub const OPS_SIZE: usize = 4096;

/// Query language ops (0x00-0x04) - with sublanguage slot subdivision
pub const PREFIX_LANCE: u8 = 0x00;     // Lance/Kuzu - vector ops
pub const PREFIX_SQL: u8 = 0x01;       // SQL/CQL (slots 0x00-0x7F: SQL, 0x80-0xFF: CQL)
pub const PREFIX_CQL: u8 = 0x01;       // CQL shares prefix with SQL
pub const PREFIX_CYPHER: u8 = 0x02;    // Cypher/GQL (slots 0x00-0x7F: Cypher, 0x80-0xFF: GQL)
pub const PREFIX_GQL: u8 = 0x02;       // GQL shares prefix with Cypher
pub const PREFIX_GRAPHQL: u8 = 0x03;   // GraphQL (schema-first, distinct)
pub const PREFIX_NARS: u8 = 0x04;      // NARS/ACT-R (slots 0x00-0x7F: NARS, 0x80-0xFF: ACT-R)
pub const PREFIX_ACTR: u8 = 0x04;      // ACT-R shares prefix with NARS

/// Reasoning ops (0x05-0x07)
pub const PREFIX_CAUSAL: u8 = 0x05;    // Causal reasoning (Pearl's ladder)
pub const PREFIX_META: u8 = 0x06;      // Meta-cognition
pub const PREFIX_VERBS: u8 = 0x07;     // Core verbs (CAUSES, BECOMES, etc.)

/// Cognitive ops (0x08-0x0B)
pub const PREFIX_CONCEPTS: u8 = 0x08;  // Core concepts/types
pub const PREFIX_QUALIA: u8 = 0x09;    // Qualia operations (felt sense)
pub const PREFIX_MEMORY: u8 = 0x0A;    // Memory operations
pub const PREFIX_LEARNING: u8 = 0x0B;  // Learning operations

/// User reserved (0x0C-0x0E) - for custom extensions
pub const PREFIX_USER_C: u8 = 0x0C;    // USER RESERVED
pub const PREFIX_USER_D: u8 = 0x0D;    // USER RESERVED  
pub const PREFIX_USER_E: u8 = 0x0E;    // USER RESERVED

/// Learned/custom thinking styles (0x0F) - 256 slots
pub const PREFIX_LEARNED_STYLES: u8 = 0x0F;  // Custom cognition patterns

/// Slot subdivision boundary for shared prefixes (SQL/CQL, Cypher/GQL, NARS/ACT-R)
pub const SLOT_SUBDIVISION: u8 = 0x80;

// =============================================================================
// FLUID ZONE: 0x10-0x3F (12,288 addresses)
// Registers/Cache - fast transient slots for immediate context
// =============================================================================

pub const PREFIX_FLUID_START: u8 = 0x10;
pub const PREFIX_FLUID_END: u8 = 0x3F;
pub const FLUID_PREFIXES: usize = 48;
pub const FLUID_SIZE: usize = 12288;

/// Context Crystal: temporal SPO grid (0x10-0x14)
pub const PREFIX_CRYSTAL_S_MINUS_2: u8 = 0x10;  // S-2: 2 sentences before
pub const PREFIX_CRYSTAL_S_MINUS_1: u8 = 0x11;  // S-1: 1 sentence before
pub const PREFIX_CRYSTAL_S_CURRENT: u8 = 0x12;  // S0:  Current sentence
pub const PREFIX_CRYSTAL_S_PLUS_1: u8 = 0x13;   // S+1: 1 sentence after
pub const PREFIX_CRYSTAL_S_PLUS_2: u8 = 0x14;   // S+2: 2 sentences after

/// Mexican hat weighting for crystal positions
pub const CRYSTAL_WEIGHTS: [f32; 5] = [0.3, 0.7, 1.0, 0.7, 0.3];

/// Working memory (0x15-0x3F) - stack frames, scratch
pub const PREFIX_WORKING_START: u8 = 0x15;
pub const PREFIX_WORKING_END: u8 = 0x3F;

// =============================================================================
// EDGE ZONE: 0x40-0x7F (16,384 addresses)
// The "Verb" slot in Bitpacked Trio [Node, EDGE, Node]
// =============================================================================

pub const PREFIX_EDGE_START: u8 = 0x40;
pub const PREFIX_EDGE_END: u8 = 0x7F;
pub const EDGE_PREFIXES: usize = 64;
pub const EDGE_SIZE: usize = 16384;

/// Edge addressing encodes relationship + Dn delta to target
/// Prefix: relationship class (verb, temporal, rung, cognitive)
/// Slot: target hint / weight class / qualifier
///
/// Arrow makes any two nodes virtually adjacent - delta is navigation hint

// =============================================================================
// NODE ZONE: 0x80-0xFF (32,768 addresses)
// The "Leaf" slots - active working set loaded in Crystal
// =============================================================================

pub const PREFIX_NODE_START: u8 = 0x80;
pub const PREFIX_NODE_END: u8 = 0xFF;
pub const NODE_PREFIXES: usize = 128;
pub const NODE_SIZE: usize = 32768;

/// Node addressing uses Dn tree structure:
/// - 2^4 branching per level (16 children)
/// - Tree.Branch.Twig.Leaf encoded in 16 bits
/// - Hamming distance = topological distance
/// - ±3 layer scent radius for navigation

// =============================================================================
// ADDRESS TYPE
// =============================================================================

/// 16-bit address: prefix (8-bit) : slot (8-bit)
/// Foundation for Bitpacked Trio [u16; 3]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Addr(u16);

impl Addr {
    #[inline]
    pub const fn new(prefix: u8, slot: u8) -> Self {
        Self(((prefix as u16) << 8) | (slot as u16))
    }
    
    #[inline]
    pub const fn from_u16(raw: u16) -> Self {
        Self(raw)
    }
    
    #[inline]
    pub const fn to_u16(self) -> u16 {
        self.0
    }
    
    #[inline]
    pub const fn prefix(self) -> u8 {
        (self.0 >> 8) as u8
    }
    
    #[inline]
    pub const fn slot(self) -> u8 {
        self.0 as u8
    }
    
    // Zone checks
    
    #[inline]
    pub const fn is_ops(self) -> bool {
        self.prefix() <= PREFIX_OPS_END
    }
    
    #[inline]
    pub const fn is_fluid(self) -> bool {
        self.prefix() >= PREFIX_FLUID_START && self.prefix() <= PREFIX_FLUID_END
    }
    
    #[inline]
    pub const fn is_crystal(self) -> bool {
        self.prefix() >= PREFIX_CRYSTAL_S_MINUS_2 && self.prefix() <= PREFIX_CRYSTAL_S_PLUS_2
    }
    
    #[inline]
    pub const fn is_edge(self) -> bool {
        self.prefix() >= PREFIX_EDGE_START && self.prefix() <= PREFIX_EDGE_END
    }
    
    #[inline]
    pub const fn is_node(self) -> bool {
        self.prefix() >= PREFIX_NODE_START
    }
    
    #[inline]
    pub const fn is_learned_style(self) -> bool {
        self.prefix() == PREFIX_LEARNED_STYLES
    }
    
    #[inline]
    pub const fn is_user_reserved(self) -> bool {
        self.prefix() >= PREFIX_USER_C && self.prefix() <= PREFIX_USER_E
    }
    
    // Crystal helpers
    
    pub fn crystal_position(self) -> Option<i32> {
        if self.is_crystal() {
            Some((self.prefix() as i32) - (PREFIX_CRYSTAL_S_CURRENT as i32))
        } else {
            None
        }
    }
    
    pub fn crystal_weight(self) -> f32 {
        if let Some(pos) = self.crystal_position() {
            let idx = (pos + 2) as usize;
            if idx < CRYSTAL_WEIGHTS.len() {
                CRYSTAL_WEIGHTS[idx]
            } else {
                0.0
            }
        } else {
            0.0
        }
    }
    
    // Constructors
    
    #[inline]
    pub const fn ops(prefix: u8, slot: u8) -> Self {
        debug_assert!(prefix <= PREFIX_OPS_END);
        Self::new(prefix, slot)
    }
    
    #[inline]
    pub const fn learned_style(slot: u8) -> Self {
        Self::new(PREFIX_LEARNED_STYLES, slot)
    }
    
    #[inline]
    pub const fn crystal(position: i32, slot: u8) -> Self {
        let prefix = ((PREFIX_CRYSTAL_S_CURRENT as i32) + position) as u8;
        Self::new(prefix, slot)
    }
    
    #[inline]
    pub const fn edge(prefix_offset: u8, slot: u8) -> Self {
        Self::new(PREFIX_EDGE_START + prefix_offset, slot)
    }
    
    #[inline]
    pub const fn node(prefix_offset: u8, slot: u8) -> Self {
        Self::new(PREFIX_NODE_START + prefix_offset, slot)
    }
    
    // Dn tree helpers for nodes
    
    /// Decode Dn tree address from node
    /// Returns (tree, branch, twig, leaf) each 0-15
    pub fn dn_decode(self) -> Option<(u8, u8, u8, u8)> {
        if !self.is_node() {
            return None;
        }
        // 14 bits available (128 prefixes × 256 slots = 32K)
        // Split as: 4 bits tree, 4 bits branch, 3 bits twig, 3 bits leaf
        let raw = self.0 - ((PREFIX_NODE_START as u16) << 8);
        let tree = ((raw >> 10) & 0x0F) as u8;
        let branch = ((raw >> 6) & 0x0F) as u8;
        let twig = ((raw >> 3) & 0x07) as u8;
        let leaf = (raw & 0x07) as u8;
        Some((tree, branch, twig, leaf))
    }
    
    /// Encode Dn tree address to node
    pub fn dn_encode(tree: u8, branch: u8, twig: u8, leaf: u8) -> Self {
        debug_assert!(tree < 16 && branch < 16 && twig < 8 && leaf < 8);
        let raw = ((tree as u16) << 10) 
                | ((branch as u16) << 6) 
                | ((twig as u16) << 3) 
                | (leaf as u16);
        Self::from_u16(((PREFIX_NODE_START as u16) << 8) + raw)
    }
    
    /// Hamming distance to another address (topological distance for nodes)
    #[inline]
    pub fn hamming_distance(self, other: Addr) -> u32 {
        (self.0 ^ other.0).count_ones()
    }
}

impl std::fmt::Debug for Addr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Addr(0x{:02X}:{:02X})", self.prefix(), self.slot())
    }
}

impl std::fmt::Display for Addr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:02X}:{:02X}", self.prefix(), self.slot())
    }
}

impl From<u16> for Addr {
    fn from(raw: u16) -> Self {
        Self(raw)
    }
}

impl From<Addr> for u16 {
    fn from(addr: Addr) -> Self {
        addr.0
    }
}

// =============================================================================
// SUBLANGUAGE SUPPORT
// =============================================================================

/// Sublanguage within a shared prefix (slot-based discrimination)
/// Slots 0x00-0x7F map to primary, 0x80-0xFF to secondary
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sublanguage {
    // Columnar (prefix 0x01)
    Sql,
    Cql,
    // Graph (prefix 0x02)  
    Cypher,
    Gql,
    // Cognitive (prefix 0x04)
    Nars,
    ActR,
}

impl Sublanguage {
    /// Determine sublanguage from address
    pub fn from_addr(addr: Addr) -> Option<Self> {
        let prefix = addr.prefix();
        let slot = addr.slot();
        let is_secondary = slot >= SLOT_SUBDIVISION;
        
        match prefix {
            PREFIX_SQL => Some(if is_secondary { Self::Cql } else { Self::Sql }),
            PREFIX_CYPHER => Some(if is_secondary { Self::Gql } else { Self::Cypher }),
            PREFIX_NARS => Some(if is_secondary { Self::ActR } else { Self::Nars }),
            _ => None,
        }
    }
}

// =============================================================================
// ZONE CLASSIFICATION
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Zone {
    Ops(OpsCompartment),
    Fluid(FluidType),
    Edge,
    Node,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpsCompartment {
    Lance,
    Sql,
    Cypher,
    GraphQL,
    Nars,
    Causal,
    Meta,
    Verbs,
    Concepts,
    Qualia,
    Memory,
    Learning,
    UserReserved,
    LearnedStyles,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FluidType {
    Crystal { position: i32 },
    Working,
}

impl Addr {
    pub fn zone(self) -> Zone {
        let p = self.prefix();
        
        if p <= PREFIX_OPS_END {
            let comp = match p {
                PREFIX_LANCE => OpsCompartment::Lance,
                PREFIX_SQL => OpsCompartment::Sql,
                PREFIX_CYPHER => OpsCompartment::Cypher,
                PREFIX_GRAPHQL => OpsCompartment::GraphQL,
                PREFIX_NARS => OpsCompartment::Nars,
                PREFIX_CAUSAL => OpsCompartment::Causal,
                PREFIX_META => OpsCompartment::Meta,
                PREFIX_VERBS => OpsCompartment::Verbs,
                PREFIX_CONCEPTS => OpsCompartment::Concepts,
                PREFIX_QUALIA => OpsCompartment::Qualia,
                PREFIX_MEMORY => OpsCompartment::Memory,
                PREFIX_LEARNING => OpsCompartment::Learning,
                PREFIX_USER_C | PREFIX_USER_D | PREFIX_USER_E => OpsCompartment::UserReserved,
                PREFIX_LEARNED_STYLES => OpsCompartment::LearnedStyles,
                _ => unreachable!(),
            };
            Zone::Ops(comp)
        } else if p <= PREFIX_FLUID_END {
            if p <= PREFIX_CRYSTAL_S_PLUS_2 {
                let pos = (p as i32) - (PREFIX_CRYSTAL_S_CURRENT as i32);
                Zone::Fluid(FluidType::Crystal { position: pos })
            } else {
                Zone::Fluid(FluidType::Working)
            }
        } else if p <= PREFIX_EDGE_END {
            Zone::Edge
        } else {
            Zone::Node
        }
    }
}

// =============================================================================
// BITPACKED TRIO - THE SUPERPOWER
// =============================================================================

/// Edge triple: [Source, Relationship, Target]
/// 48 bits total, aligned to 64-bit with padding
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C, align(8))]
pub struct EdgeTriple {
    pub source: Addr,      // u16 - node address
    pub relationship: Addr, // u16 - edge address  
    pub target: Addr,      // u16 - node address
    pub _pad: u16,         // padding for 64-bit alignment
}

impl EdgeTriple {
    #[inline]
    pub const fn new(source: Addr, relationship: Addr, target: Addr) -> Self {
        Self {
            source,
            relationship,
            target,
            _pad: 0,
        }
    }
    
    /// Pack to 64-bit integer (for AVX-512 batch processing)
    #[inline]
    pub const fn pack(self) -> u64 {
        ((self.source.0 as u64) << 48)
            | ((self.relationship.0 as u64) << 32)
            | ((self.target.0 as u64) << 16)
            | (self._pad as u64)
    }
    
    /// Unpack from 64-bit integer
    #[inline]
    pub const fn unpack(packed: u64) -> Self {
        Self {
            source: Addr((packed >> 48) as u16),
            relationship: Addr((packed >> 32) as u16),
            target: Addr((packed >> 16) as u16),
            _pad: packed as u16,
        }
    }
}

impl std::fmt::Debug for EdgeTriple {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{} --{}--> {}]", self.source, self.relationship, self.target)
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_zone_boundaries() {
        // OPS: 0x00-0x0F
        assert!(Addr::new(0x00, 0x00).is_ops());
        assert!(Addr::new(0x0F, 0xFF).is_ops());
        assert!(!Addr::new(0x10, 0x00).is_ops());
        
        // FLUID: 0x10-0x3F
        assert!(Addr::new(0x10, 0x00).is_fluid());
        assert!(Addr::new(0x3F, 0xFF).is_fluid());
        assert!(!Addr::new(0x40, 0x00).is_fluid());
        
        // EDGE: 0x40-0x7F
        assert!(Addr::new(0x40, 0x00).is_edge());
        assert!(Addr::new(0x7F, 0xFF).is_edge());
        assert!(!Addr::new(0x80, 0x00).is_edge());
        
        // NODE: 0x80-0xFF
        assert!(Addr::new(0x80, 0x00).is_node());
        assert!(Addr::new(0xFF, 0xFF).is_node());
    }
    
    #[test]
    fn test_learned_styles() {
        let style = Addr::learned_style(42);
        assert!(style.is_learned_style());
        assert!(style.is_ops());
        assert_eq!(style.prefix(), PREFIX_LEARNED_STYLES);
        assert_eq!(style.slot(), 42);
    }
    
    #[test]
    fn test_user_reserved() {
        assert!(Addr::new(PREFIX_USER_C, 0).is_user_reserved());
        assert!(Addr::new(PREFIX_USER_D, 0).is_user_reserved());
        assert!(Addr::new(PREFIX_USER_E, 0).is_user_reserved());
        assert!(!Addr::new(PREFIX_LEARNED_STYLES, 0).is_user_reserved());
    }
    
    #[test]
    fn test_sublanguage() {
        // SQL vs CQL
        assert_eq!(Sublanguage::from_addr(Addr::new(PREFIX_SQL, 0x00)), Some(Sublanguage::Sql));
        assert_eq!(Sublanguage::from_addr(Addr::new(PREFIX_SQL, 0x7F)), Some(Sublanguage::Sql));
        assert_eq!(Sublanguage::from_addr(Addr::new(PREFIX_SQL, 0x80)), Some(Sublanguage::Cql));
        assert_eq!(Sublanguage::from_addr(Addr::new(PREFIX_SQL, 0xFF)), Some(Sublanguage::Cql));
        
        // Cypher vs GQL
        assert_eq!(Sublanguage::from_addr(Addr::new(PREFIX_CYPHER, 0x00)), Some(Sublanguage::Cypher));
        assert_eq!(Sublanguage::from_addr(Addr::new(PREFIX_CYPHER, 0x80)), Some(Sublanguage::Gql));
        
        // NARS vs ACT-R
        assert_eq!(Sublanguage::from_addr(Addr::new(PREFIX_NARS, 0x00)), Some(Sublanguage::Nars));
        assert_eq!(Sublanguage::from_addr(Addr::new(PREFIX_NARS, 0x80)), Some(Sublanguage::ActR));
    }
    
    #[test]
    fn test_crystal_positions() {
        let crystal = Addr::crystal(-2, 10);
        assert!(crystal.is_crystal());
        assert_eq!(crystal.crystal_position(), Some(-2));
        assert_eq!(crystal.crystal_weight(), 0.3);
        
        let current = Addr::crystal(0, 10);
        assert_eq!(current.crystal_weight(), 1.0);
    }
    
    #[test]
    fn test_dn_encoding() {
        let node = Addr::dn_encode(5, 10, 3, 7);
        assert!(node.is_node());
        
        let (tree, branch, twig, leaf) = node.dn_decode().unwrap();
        assert_eq!(tree, 5);
        assert_eq!(branch, 10);
        assert_eq!(twig, 3);
        assert_eq!(leaf, 7);
    }
    
    #[test]
    fn test_hamming_distance() {
        let a = Addr::new(0x80, 0x00);
        let b = Addr::new(0x80, 0x01);
        assert_eq!(a.hamming_distance(b), 1);
        
        let c = Addr::new(0xFF, 0xFF);
        let d = Addr::new(0x00, 0x00);
        assert_eq!(c.hamming_distance(d), 16); // all bits different
    }
    
    #[test]
    fn test_edge_triple() {
        let source = Addr::node(0, 42);
        let rel = Addr::edge(0, 1); // CAUSES
        let target = Addr::node(10, 100);
        
        let triple = EdgeTriple::new(source, rel, target);
        
        // Test pack/unpack roundtrip
        let packed = triple.pack();
        let unpacked = EdgeTriple::unpack(packed);
        assert_eq!(triple.source, unpacked.source);
        assert_eq!(triple.relationship, unpacked.relationship);
        assert_eq!(triple.target, unpacked.target);
    }
    
    #[test]
    fn test_edge_triple_alignment() {
        // Verify 64-bit alignment
        assert_eq!(std::mem::size_of::<EdgeTriple>(), 8);
        assert_eq!(std::mem::align_of::<EdgeTriple>(), 8);
    }
    
    #[test]
    fn test_zone_sizes() {
        assert_eq!(OPS_SIZE, 4096);
        assert_eq!(FLUID_SIZE, 12288);
        assert_eq!(EDGE_SIZE, 16384);
        assert_eq!(NODE_SIZE, 32768);
        assert_eq!(OPS_SIZE + FLUID_SIZE + EDGE_SIZE + NODE_SIZE, 65536);
    }
}
