//! Universal Bind Space - 16-bit Alien Engine Layout
//!
//! # The Bitpacked Trio Constraint
//!
//! This layout is **mathematically required** for AVX-512 alignment:
//! - Edge Trio = [Source, Target, Verb] = 3 × u16 = 48 bits
//! - AVX-512 processes 32 × u16 per instruction
//! - Widening to u24 breaks alignment, bloats storage 50%, kills throughput
//!
//! # 8-bit Prefix : 8-bit Slot Architecture (u16 total)
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                      PREFIX (8-bit) : SLOT (8-bit)                          │
//! ├─────────────────┬───────────────────────────────────────────────────────────┤
//! │  0x00-0x0F:XX   │  OPS (16 prefixes × 256 = 4,096)                          │
//! │                 │  ├─ 0x00-0x0E:XX  Built-in methods (3,840)                │
//! │                 │  │   0x00: Lance/Kuzu    0x08: Concepts                   │
//! │                 │  │   0x01: SQL/CQL       0x09: Qualia ops                 │
//! │                 │  │   0x02: Cypher/GQL    0x0A: Memory ops                 │
//! │                 │  │   0x03: GraphQL       0x0B: Learning ops               │
//! │                 │  │   0x04: NARS/ACT-R    0x0C: Reserved                   │
//! │                 │  │   0x05: Causal        0x0D: Reserved                   │
//! │                 │  │   0x06: Meta          0x0E: Reserved                   │
//! │                 │  │   0x07: Verbs                                          │
//! │                 │  └─ 0x0F:XX  Learned/custom thinking styles (256)         │
//! ├─────────────────┼───────────────────────────────────────────────────────────┤
//! │  0x10-0x3F:XX   │  FLUID (48 prefixes × 256 = 12,288)                       │
//! │                 │  Working memory, context crystal, ephemeral state         │
//! │                 │  ├─ 0x10-0x14: Crystal (±2 temporal window)               │
//! │                 │  └─ 0x15-0x3F: Working memory (TTL governed)              │
//! ├─────────────────┼───────────────────────────────────────────────────────────┤
//! │  0x40-0x7F:XX   │  EDGES (64 prefixes × 256 = 16,384)                       │
//! │                 │  Relationships with Dn addressing (2^4 branches)          │
//! │                 │  Verb + Temporal + Rung + Cognitive qualifiers            │
//! │                 │  Arrow-relative targeting (virtual adjacency)             │
//! ├─────────────────┼───────────────────────────────────────────────────────────┤
//! │  0x80-0xBF:XX   │  NODES (64 prefixes × 256 = 16,384)                       │
//! │                 │  Dn tree topology: Tree.Branch.Twig.Leaf                  │
//! │                 │  4-bit per level = 16 branches = 2^4 fanout               │
//! │                 │  Hamming distance = topological distance                  │
//! ├─────────────────┼───────────────────────────────────────────────────────────┤
//! │  0xC0-0xFF:XX   │  USER/SHARD (64 prefixes × 256 = 16,384)                  │
//! │                 │  Reserved for sharding, overflow, user extensions         │
//! │                 │  Production scaling: [Shard_ID:u16 | Local_Addr:u16]      │
//! └─────────────────┴───────────────────────────────────────────────────────────┘
//! ```
//!
//! # Why 16-bit is NOT a limitation
//!
//! - 32K active nodes is the **working set**, not total storage
//! - Cold storage (LanceDB) holds billions
//! - Global UUID → Local u16 mapping during "thought" duration
//! - Scale by adding Shards, not widening pointers
//!
//! # Dn Tree Addressing (Nodes 0x80-0xBF)
//!
//! ```text
//! 14 bits for Dn path (64 prefixes × 256 slots, minus overhead):
//! ┌──────────┬──────────┬──────────┬──────────┐
//! │ Tree (4) │Branch (4)│ Twig (4) │ Leaf (2) │
//! │   16     │    16    │    16    │    4     │
//! └──────────┴──────────┴──────────┴──────────┘
//! = 16 × 16 × 16 × 4 = 16,384 addressable nodes
//!
//! XOR two addresses → Hamming distance → topological distance
//! ```

use std::time::Instant;

// =============================================================================
// ADDRESS CONSTANTS (8-bit prefix : 8-bit slot = 16-bit total)
// =============================================================================

/// Fingerprint words (10K bits = 156 × 64-bit words)
pub const FINGERPRINT_WORDS: usize = 156;

/// Slots per prefix (2^8 = 256)
pub const SLOTS_PER_PREFIX: usize = 256;

/// Total address space (2^16 = 65,536)
pub const TOTAL_ADDRESS_SPACE: usize = 65536;

// =============================================================================
// OPS ZONE: 0x00-0x0F (16 prefixes × 256 = 4,096)
// Built-in methods + learned thinking styles
// =============================================================================

pub const PREFIX_OPS_START: u8 = 0x00;
pub const PREFIX_OPS_END: u8 = 0x0F;
pub const OPS_PREFIXES: usize = 16;
pub const OPS_SIZE: usize = 4096;

/// Built-in operation prefixes (0x00-0x0E = 3,840 ops)
pub const PREFIX_LANCE: u8 = 0x00;      // Lance/Kuzu - vector ops
pub const PREFIX_SQL: u8 = 0x01;        // SQL/CQL (columnar languages)
pub const PREFIX_CQL: u8 = 0x01;        // CQL shares prefix with SQL (slot 0x80+)
pub const PREFIX_CYPHER: u8 = 0x02;     // Cypher/GQL (property graph languages)
pub const PREFIX_GQL: u8 = 0x02;        // GQL shares prefix with Cypher (slot 0x80+)
pub const PREFIX_GRAPHQL: u8 = 0x03;    // GraphQL (schema-first)
pub const PREFIX_NARS: u8 = 0x04;       // NARS/ACT-R (cognitive architectures)
pub const PREFIX_ACTR: u8 = 0x04;       // ACT-R shares prefix with NARS (slot 0x80+)
pub const PREFIX_CAUSAL: u8 = 0x05;     // Causal reasoning (Pearl)
pub const PREFIX_META: u8 = 0x06;       // Meta-cognition
pub const PREFIX_VERBS: u8 = 0x07;      // Core verbs (CAUSES, BECOMES...)
pub const PREFIX_CONCEPTS: u8 = 0x08;   // Core concepts/types
pub const PREFIX_QUALIA: u8 = 0x09;     // Qualia operations
pub const PREFIX_MEMORY: u8 = 0x0A;     // Memory operations
pub const PREFIX_LEARNING: u8 = 0x0B;   // Learning operations
pub const PREFIX_RESERVED_C: u8 = 0x0C; // Reserved for user
pub const PREFIX_RESERVED_D: u8 = 0x0D; // Reserved for user
pub const PREFIX_RESERVED_E: u8 = 0x0E; // Reserved for user

/// Learned/custom thinking styles (0x0F:XX = 256 slots)
pub const PREFIX_LEARNED: u8 = 0x0F;
pub const LEARNED_STYLES_COUNT: usize = 256;

// Slot subdivision for shared prefixes (SQL/CQL, Cypher/GQL, NARS/ACT-R)
pub const SLOT_SUBDIVISION: u8 = 0x80;

// =============================================================================
// FLUID ZONE: 0x10-0x3F (48 prefixes × 256 = 12,288)
// Working memory, context crystal, ephemeral state
// =============================================================================

pub const PREFIX_FLUID_START: u8 = 0x10;
pub const PREFIX_FLUID_END: u8 = 0x3F;
pub const FLUID_PREFIXES: usize = 48;
pub const FLUID_SIZE: usize = 12288;

/// Context Crystal: temporal window (±2 sentences)
pub const PREFIX_CRYSTAL_S_MINUS_2: u8 = 0x10;  // S-2: 2 sentences before
pub const PREFIX_CRYSTAL_S_MINUS_1: u8 = 0x11;  // S-1: 1 sentence before
pub const PREFIX_CRYSTAL_S_CURRENT: u8 = 0x12;  // S0:  Current sentence
pub const PREFIX_CRYSTAL_S_PLUS_1: u8 = 0x13;   // S+1: 1 sentence after
pub const PREFIX_CRYSTAL_S_PLUS_2: u8 = 0x14;   // S+2: 2 sentences after

/// Mexican hat weighting for crystal temporal positions
pub const CRYSTAL_WEIGHTS: [f32; 5] = [0.3, 0.7, 1.0, 0.7, 0.3];

/// Working memory (TTL governed)
pub const PREFIX_WORKING_START: u8 = 0x15;
pub const PREFIX_WORKING_END: u8 = 0x3F;

// =============================================================================
// EDGES ZONE: 0x40-0x7F (64 prefixes × 256 = 16,384)
// Relationships with Dn addressing, Arrow-relative targeting
// =============================================================================

pub const PREFIX_EDGE_START: u8 = 0x40;
pub const PREFIX_EDGE_END: u8 = 0x7F;
pub const EDGE_PREFIXES: usize = 64;
pub const EDGE_SIZE: usize = 16384;

// Edge relationship types (encoded in prefix)
pub const EDGE_CAUSES: u8 = 0x40;
pub const EDGE_SUPPORTS: u8 = 0x41;
pub const EDGE_CONTRADICTS: u8 = 0x42;
pub const EDGE_BECOMES: u8 = 0x43;
pub const EDGE_REFINES: u8 = 0x44;
pub const EDGE_GROUNDS: u8 = 0x45;
pub const EDGE_ABSTRACTS: u8 = 0x46;
pub const EDGE_TEMPORAL_BEFORE: u8 = 0x48;
pub const EDGE_TEMPORAL_DURING: u8 = 0x49;
pub const EDGE_TEMPORAL_AFTER: u8 = 0x4A;
pub const EDGE_RUNG_SEE: u8 = 0x50;
pub const EDGE_RUNG_DO: u8 = 0x51;
pub const EDGE_RUNG_IMAGINE: u8 = 0x52;
pub const EDGE_COGNITIVE_BELIEVE: u8 = 0x58;
pub const EDGE_COGNITIVE_DOUBT: u8 = 0x59;
pub const EDGE_COGNITIVE_INFER: u8 = 0x5A;

// =============================================================================
// NODES ZONE: 0x80-0xBF (64 prefixes × 256 = 16,384)
// Dn tree topology with bitpacked Hamming addressing
// =============================================================================

pub const PREFIX_NODE_START: u8 = 0x80;
pub const PREFIX_NODE_END: u8 = 0xBF;
pub const NODE_PREFIXES: usize = 64;
pub const NODE_SIZE: usize = 16384;

/// Dn tree parameters (2^4 = 16 branches per level)
pub const DN_BRANCH_BITS: u8 = 4;
pub const DN_BRANCHES_PER_LEVEL: usize = 16;

// =============================================================================
// USER/SHARD ZONE: 0xC0-0xFF (64 prefixes × 256 = 16,384)
// Reserved for sharding, overflow, user extensions
// =============================================================================

pub const PREFIX_USER_START: u8 = 0xC0;
pub const PREFIX_USER_END: u8 = 0xFF;
pub const USER_PREFIXES: usize = 64;
pub const USER_SIZE: usize = 16384;

// =============================================================================
// ADDRESS TYPE
// =============================================================================

/// A 16-bit address in the bind space
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Addr(u16);

impl Addr {
    /// Create address from prefix and slot
    #[inline]
    pub const fn new(prefix: u8, slot: u8) -> Self {
        Self(((prefix as u16) << 8) | (slot as u16))
    }
    
    /// Create from raw u16
    #[inline]
    pub const fn from_raw(raw: u16) -> Self {
        Self(raw)
    }
    
    /// Get raw u16 value
    #[inline]
    pub const fn raw(self) -> u16 {
        self.0
    }
    
    /// Get prefix byte
    #[inline]
    pub const fn prefix(self) -> u8 {
        (self.0 >> 8) as u8
    }
    
    /// Get slot byte
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
    pub const fn is_learned(self) -> bool {
        self.prefix() == PREFIX_LEARNED
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
        self.prefix() >= PREFIX_NODE_START && self.prefix() <= PREFIX_NODE_END
    }
    
    #[inline]
    pub const fn is_user(self) -> bool {
        self.prefix() >= PREFIX_USER_START
    }
    
    // Constructors for common zones
    
    #[inline]
    pub const fn op(prefix: u8, slot: u8) -> Self {
        debug_assert!(prefix <= PREFIX_OPS_END);
        Self::new(prefix, slot)
    }
    
    #[inline]
    pub const fn learned(slot: u8) -> Self {
        Self::new(PREFIX_LEARNED, slot)
    }
    
    #[inline]
    pub const fn crystal(position: i8, slot: u8) -> Self {
        let prefix = ((PREFIX_CRYSTAL_S_CURRENT as i8) + position) as u8;
        Self::new(prefix, slot)
    }
    
    #[inline]
    pub const fn edge(edge_type: u8, slot: u8) -> Self {
        Self::new(edge_type, slot)
    }
    
    #[inline]
    pub const fn node(prefix: u8, slot: u8) -> Self {
        debug_assert!(prefix >= PREFIX_NODE_START && prefix <= PREFIX_NODE_END);
        Self::new(prefix, slot)
    }
    
    /// Crystal temporal position (-2 to +2)
    pub fn crystal_position(self) -> Option<i8> {
        if self.is_crystal() {
            Some((self.prefix() as i8) - (PREFIX_CRYSTAL_S_CURRENT as i8))
        } else {
            None
        }
    }
    
    /// Crystal weight based on temporal position
    pub fn crystal_weight(self) -> f32 {
        if let Some(pos) = self.crystal_position() {
            CRYSTAL_WEIGHTS[(pos + 2) as usize]
        } else {
            0.0
        }
    }
    
    /// Hamming distance between two addresses (topological distance)
    #[inline]
    pub fn hamming_distance(self, other: Addr) -> u32 {
        (self.0 ^ other.0).count_ones()
    }
}

impl std::fmt::Debug for Addr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Addr({:#04X}:{:#04X})", self.prefix(), self.slot())
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
// SUBLANGUAGE (for shared prefixes)
// =============================================================================

/// Sublanguage within a shared prefix (slot-based discrimination)
/// Slots 0x00-0x7F map to primary, 0x80-0xFF to secondary
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sublanguage {
    Sql,
    Cql,
    Cypher,
    Gql,
    Nars,
    ActR,
}

impl Sublanguage {
    /// Determine sublanguage from address
    pub fn from_addr(addr: Addr) -> Option<Self> {
        let prefix = addr.prefix();
        let is_secondary = addr.slot() >= SLOT_SUBDIVISION;
        
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
    User,
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
    ReservedC,
    ReservedD,
    ReservedE,
    Learned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FluidType {
    Crystal { position: i8 },
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
                PREFIX_RESERVED_C => OpsCompartment::ReservedC,
                PREFIX_RESERVED_D => OpsCompartment::ReservedD,
                PREFIX_RESERVED_E => OpsCompartment::ReservedE,
                PREFIX_LEARNED => OpsCompartment::Learned,
                _ => unreachable!(),
            };
            Zone::Ops(comp)
        } else if p <= PREFIX_FLUID_END {
            if p <= PREFIX_CRYSTAL_S_PLUS_2 {
                let pos = (p as i8) - (PREFIX_CRYSTAL_S_CURRENT as i8);
                Zone::Fluid(FluidType::Crystal { position: pos })
            } else {
                Zone::Fluid(FluidType::Working)
            }
        } else if p <= PREFIX_EDGE_END {
            Zone::Edge
        } else if p <= PREFIX_NODE_END {
            Zone::Node
        } else {
            Zone::User
        }
    }
}

// =============================================================================
// Dn TREE ADDRESSING (for Nodes)
// =============================================================================

/// Dn tree address decomposition
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DnPath {
    pub tree: u8,    // 4 bits (0-15)
    pub branch: u8,  // 4 bits (0-15)
    pub twig: u8,    // 4 bits (0-15)
    pub leaf: u8,    // 2 bits (0-3)
}

impl DnPath {
    /// Extract Dn path from node address
    pub fn from_addr(addr: Addr) -> Option<Self> {
        if !addr.is_node() {
            return None;
        }
        
        // Map 0x80-0xBF (64 prefixes) + 256 slots = 16,384 addresses
        // to Tree(4).Branch(4).Twig(4).Leaf(2) = 16×16×16×4 = 16,384
        let offset = ((addr.prefix() - PREFIX_NODE_START) as u16) << 8 | (addr.slot() as u16);
        
        Some(Self {
            tree: ((offset >> 10) & 0x0F) as u8,
            branch: ((offset >> 6) & 0x0F) as u8,
            twig: ((offset >> 2) & 0x0F) as u8,
            leaf: (offset & 0x03) as u8,
        })
    }
    
    /// Convert Dn path to node address
    pub fn to_addr(self) -> Addr {
        let offset = ((self.tree as u16 & 0x0F) << 10)
                   | ((self.branch as u16 & 0x0F) << 6)
                   | ((self.twig as u16 & 0x0F) << 2)
                   | (self.leaf as u16 & 0x03);
        
        let prefix = PREFIX_NODE_START + ((offset >> 8) as u8);
        let slot = (offset & 0xFF) as u8;
        
        Addr::new(prefix, slot)
    }
    
    /// Hamming distance between two Dn paths (topological distance)
    pub fn distance(self, other: DnPath) -> u32 {
        let a = ((self.tree as u16) << 10) | ((self.branch as u16) << 6) 
              | ((self.twig as u16) << 2) | (self.leaf as u16);
        let b = ((other.tree as u16) << 10) | ((other.branch as u16) << 6)
              | ((other.twig as u16) << 2) | (other.leaf as u16);
        (a ^ b).count_ones()
    }
}

// =============================================================================
// EDGE TRIO (48-bit, AVX-512 aligned)
// =============================================================================

/// Edge as bitpacked trio: [Source:u16, Target:u16, Verb:u16] = 48 bits
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(C, align(2))]
pub struct EdgeTrio {
    pub source: Addr,
    pub target: Addr,
    pub verb: Addr,
}

impl EdgeTrio {
    #[inline]
    pub const fn new(source: Addr, target: Addr, verb: Addr) -> Self {
        Self { source, target, verb }
    }
    
    /// Pack to 48-bit array (for AVX-512 batch processing)
    #[inline]
    pub fn pack(self) -> [u16; 3] {
        [self.source.raw(), self.target.raw(), self.verb.raw()]
    }
    
    /// Unpack from 48-bit array
    #[inline]
    pub fn unpack(packed: [u16; 3]) -> Self {
        Self {
            source: Addr::from_raw(packed[0]),
            target: Addr::from_raw(packed[1]),
            verb: Addr::from_raw(packed[2]),
        }
    }
}

impl std::fmt::Debug for EdgeTrio {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Edge({} --{}--> {})", self.source, self.verb, self.target)
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_zone_sizes() {
        assert_eq!(OPS_SIZE, 4096);
        assert_eq!(FLUID_SIZE, 12288);
        assert_eq!(EDGE_SIZE, 16384);
        assert_eq!(NODE_SIZE, 16384);
        assert_eq!(USER_SIZE, 16384);
        assert_eq!(OPS_SIZE + FLUID_SIZE + EDGE_SIZE + NODE_SIZE + USER_SIZE, TOTAL_ADDRESS_SPACE);
    }
    
    #[test]
    fn test_zone_classification() {
        assert!(Addr::new(0x00, 0x00).is_ops());
        assert!(Addr::new(0x0F, 0x00).is_learned());
        assert!(Addr::new(0x10, 0x00).is_fluid());
        assert!(Addr::new(0x12, 0x00).is_crystal());
        assert!(Addr::new(0x40, 0x00).is_edge());
        assert!(Addr::new(0x80, 0x00).is_node());
        assert!(Addr::new(0xC0, 0x00).is_user());
    }
    
    #[test]
    fn test_crystal_position() {
        assert_eq!(Addr::crystal(-2, 0).crystal_position(), Some(-2));
        assert_eq!(Addr::crystal(0, 0).crystal_position(), Some(0));
        assert_eq!(Addr::crystal(2, 0).crystal_position(), Some(2));
        
        assert_eq!(Addr::crystal(0, 0).crystal_weight(), 1.0);
        assert_eq!(Addr::crystal(-2, 0).crystal_weight(), 0.3);
    }
    
    #[test]
    fn test_sublanguage() {
        assert_eq!(Sublanguage::from_addr(Addr::new(PREFIX_SQL, 0x00)), Some(Sublanguage::Sql));
        assert_eq!(Sublanguage::from_addr(Addr::new(PREFIX_SQL, 0x80)), Some(Sublanguage::Cql));
        assert_eq!(Sublanguage::from_addr(Addr::new(PREFIX_CYPHER, 0x00)), Some(Sublanguage::Cypher));
        assert_eq!(Sublanguage::from_addr(Addr::new(PREFIX_CYPHER, 0x80)), Some(Sublanguage::Gql));
    }
    
    #[test]
    fn test_dn_path_roundtrip() {
        let path = DnPath { tree: 5, branch: 10, twig: 3, leaf: 2 };
        let addr = path.to_addr();
        let recovered = DnPath::from_addr(addr).unwrap();
        assert_eq!(path, recovered);
    }
    
    #[test]
    fn test_dn_distance() {
        let a = DnPath { tree: 0, branch: 0, twig: 0, leaf: 0 };
        let b = DnPath { tree: 0, branch: 0, twig: 0, leaf: 1 };
        assert_eq!(a.distance(b), 1);
        
        let c = DnPath { tree: 15, branch: 15, twig: 15, leaf: 3 };
        assert!(a.distance(c) > 10); // Very different paths
    }
    
    #[test]
    fn test_edge_trio_pack() {
        let edge = EdgeTrio::new(
            Addr::new(0x80, 0x01),  // source node
            Addr::new(0x80, 0x02),  // target node
            Addr::new(0x40, 0x00),  // CAUSES
        );
        
        let packed = edge.pack();
        let unpacked = EdgeTrio::unpack(packed);
        assert_eq!(edge, unpacked);
    }
    
    #[test]
    fn test_hamming_distance() {
        let a = Addr::new(0x80, 0x00);
        let b = Addr::new(0x80, 0x01);
        assert_eq!(a.hamming_distance(b), 1);
        
        let c = Addr::new(0x80, 0xFF);
        assert_eq!(a.hamming_distance(c), 8);
    }
}
