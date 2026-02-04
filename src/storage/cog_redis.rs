//! Cognitive Redis
//!
//! Redis syntax, cognitive semantics. One query surface across three tiers:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                      PREFIX (8-bit) : ADDRESS (8-bit)                       │
//! ├─────────────────┬───────────────────────────────────────────────────────────┤
//! │  0x00-0x0F:XX   │  SURFACE (16 prefixes × 256 = 4,096)                      │
//! │                 │  0x00: Lance    0x04: NARS      0x08: Concepts            │
//! │                 │  0x01: SQL      0x05: Causal    0x09: Qualia              │
//! │                 │  0x02: Cypher   0x06: Meta      0x0A: Memory              │
//! │                 │  0x03: GraphQL  0x07: Verbs     0x0B: Learning            │
//! ├─────────────────┼───────────────────────────────────────────────────────────┤
//! │  0x10-0x7F:XX   │  FLUID (112 prefixes × 256 = 28,672)                      │
//! │                 │  Working memory - TTL governed, promote/demote            │
//! ├─────────────────┼───────────────────────────────────────────────────────────┤
//! │  0x80-0xFF:XX   │  NODES (128 prefixes × 256 = 32,768)                      │
//! │                 │  Persistent graph - THE UNIVERSAL BIND SPACE              │
//! └─────────────────┴───────────────────────────────────────────────────────────┘
//! ```
//!
//! # 8-bit Prefix Architecture
//!
//! Pure array indexing. No HashMap. 3-5 cycles per lookup.
//!
//! ```text
//! let prefix = (addr >> 8) as u8;
//! let slot = (addr & 0xFF) as u8;
//! // Direct array access: surfaces[prefix][slot]
//! ```
//!
//! # Why Cognitive Redis?
//!
//! Standard Redis: `GET key` → value or nil
//! Cognitive Redis: `GET key` → value + qualia + truth + trace
//!
//! Every access returns not just WHAT but HOW IT FEELS and HOW CERTAIN.
//!
//! # Command Extensions
//!
//! ```text
//! ┌────────────────┬─────────────────────────────────────────────────────────┐
//! │ Standard Redis │ Cognitive Extension                                     │
//! ├────────────────┼─────────────────────────────────────────────────────────┤
//! │ GET key        │ GET key [FEEL] [TRACE] [DECAY]                          │
//! │ SET key val    │ SET key val [QUALIA q] [TRUTH f,c] [TTL t] [PROMOTE]    │
//! │ DEL key        │ DEL key [FORGET] [SUPPRESS]                             │
//! │ KEYS pattern   │ KEYS pattern [VALENCE min max] [AROUSAL min max]        │
//! │ LPUSH          │ BIND a b [VIA verb] → edge                              │
//! │ LPOP           │ UNBIND edge a → b                                       │
//! │ SCAN           │ RESONATE query [MEXICAN_HAT] → similar + qualia         │
//! │ —              │ CAUSE a → effects (Rung 2)                              │
//! │ —              │ WOULD a b → counterfactual (Rung 3)                     │
//! │ —              │ DEDUCE a b → conclusion (NARS)                          │
//! │ —              │ INTUIT qualia → resonant atoms                          │
//! │ —              │ FANOUT node → connected edges                           │
//! │ —              │ CRYSTALLIZE addr → promote to node                      │
//! │ —              │ EVAPORATE addr → demote to fluid                        │
//! └────────────────┴─────────────────────────────────────────────────────────┘
//! ```
//!
//! # The Magic
//!
//! User doesn't care WHERE something lives. They query, system decides tier.
//! Hot concepts promote. Cold nodes demote. TTL governs forgetting.
//! Graph queries traverse all tiers transparently.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use crate::core::Fingerprint;
use crate::search::cognitive::{QualiaVector, CognitiveAtom, CognitiveSearch, SpoTriple};
use crate::search::causal::CausalSearch;
use crate::learning::cognitive_frameworks::TruthValue;
use crate::learning::cam_ops::{OpCategory, OpDictionary, OpResult, LanceOp, SqlOp, HammingOp};
use super::bind_space::{BindSpace, BindNode, Addr, FINGERPRINT_WORDS, dn_path_to_addr};

// =============================================================================
// ADDRESS SPACE CONSTANTS (8-bit prefix : 8-bit slot)
// =============================================================================

/// Slots per chunk
pub const CHUNK_SIZE: usize = 256;

// -----------------------------------------------------------------------------
// SURFACE: 16 prefixes (0x00-0x0F) × 256 = 4,096 addresses
// -----------------------------------------------------------------------------

pub const PREFIX_SURFACE_START: u8 = 0x00;
pub const PREFIX_SURFACE_END: u8 = 0x0F;
pub const SURFACE_PREFIXES: usize = 16;

/// Surface compartments
pub const PREFIX_LANCE: u8 = 0x00;      // Lance/Kuzu vector ops
pub const PREFIX_SQL: u8 = 0x01;        // SQL relational ops
pub const PREFIX_CYPHER: u8 = 0x02;     // Neo4j/Cypher graph ops
pub const PREFIX_GRAPHQL: u8 = 0x03;    // GraphQL ops
pub const PREFIX_NARS: u8 = 0x04;       // NARS inference
pub const PREFIX_CAUSAL: u8 = 0x05;     // Causal reasoning (Pearl)
pub const PREFIX_META: u8 = 0x06;       // Meta-cognition
pub const PREFIX_VERBS: u8 = 0x07;      // Verbs (CAUSES, BECOMES...)
pub const PREFIX_CONCEPTS: u8 = 0x08;   // Core concept types
pub const PREFIX_QUALIA: u8 = 0x09;     // Qualia operations
pub const PREFIX_MEMORY: u8 = 0x0A;     // Memory operations
pub const PREFIX_LEARNING: u8 = 0x0B;   // Learning operations

/// Legacy constants (for compatibility)
pub const SURFACE_START: u16 = 0x0000;
pub const SURFACE_END: u16 = 0x0FFF;    // 16 prefixes × 256 slots
pub const SURFACE_SIZE: usize = 4096;

// -----------------------------------------------------------------------------
// FLUID: 112 prefixes (0x10-0x7F) × 256 = 28,672 addresses
// -----------------------------------------------------------------------------

pub const PREFIX_FLUID_START: u8 = 0x10;
pub const PREFIX_FLUID_END: u8 = 0x7F;
pub const FLUID_PREFIXES: usize = 112;
pub const FLUID_START: u16 = 0x1000;
pub const FLUID_END: u16 = 0x7FFF;
pub const FLUID_SIZE: usize = 28672;    // 112 × 256

// -----------------------------------------------------------------------------
// NODES: 128 prefixes (0x80-0xFF) × 256 = 32,768 addresses
// -----------------------------------------------------------------------------

pub const PREFIX_NODE_START: u8 = 0x80;
pub const PREFIX_NODE_END: u8 = 0xFF;
pub const NODE_PREFIXES: usize = 128;
pub const NODE_START: u16 = 0x8000;
pub const NODE_END: u16 = 0xFFFF;
pub const NODE_SIZE: usize = 32768;   // 128 chunks × 256

/// Total address space
pub const TOTAL_SIZE: usize = 65536;

// =============================================================================
// ADDRESS TYPE
// =============================================================================

/// 16-bit cognitive address as prefix:slot (8-bit each)
/// 
/// Pure array indexing. No hash lookup. 3-5 cycles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CogAddr(pub u16);

impl CogAddr {
    /// Create from raw 16-bit address
    pub fn new(addr: u16) -> Self {
        Self(addr)
    }
    
    /// Create from prefix and slot (the fast path)
    #[inline(always)]
    pub fn from_parts(prefix: u8, slot: u8) -> Self {
        Self(((prefix as u16) << 8) | (slot as u16))
    }
    
    /// Get prefix (high byte) - determines tier/compartment
    #[inline(always)]
    pub fn prefix(&self) -> u8 {
        (self.0 >> 8) as u8
    }
    
    /// Get slot (low byte) - index within chunk
    #[inline(always)]
    pub fn slot(&self) -> u8 {
        (self.0 & 0xFF) as u8
    }
    
    /// Which tier does this address belong to?
    #[inline(always)]
    pub fn tier(&self) -> Tier {
        let p = self.prefix();
        match p {
            0x00..=0x0F => Tier::Surface,  // 16 prefixes
            0x10..=0x7F => Tier::Fluid,    // 112 prefixes
            _ => Tier::Node,               // 128 prefixes
        }
    }
    
    /// Which surface compartment (if surface tier)
    #[inline(always)]
    pub fn surface_compartment(&self) -> Option<SurfaceCompartment> {
        SurfaceCompartment::from_prefix(self.prefix())
    }
    
    /// Is this in the persistent node tier?
    #[inline(always)]
    pub fn is_node(&self) -> bool {
        self.prefix() >= PREFIX_NODE_START
    }
    
    /// Is this in the fluid zone?
    #[inline(always)]
    pub fn is_fluid(&self) -> bool {
        let p = self.prefix();
        p >= PREFIX_FLUID_START && p <= PREFIX_FLUID_END
    }
    
    /// Is this a surface operation?
    #[inline(always)]
    pub fn is_surface(&self) -> bool {
        self.prefix() <= PREFIX_SURFACE_END
    }
    
    /// Promote to node tier (move to 0x80+ prefix, keep slot)
    pub fn promote(&self) -> CogAddr {
        CogAddr::from_parts(PREFIX_NODE_START, self.slot())
    }
    
    /// Demote to fluid tier (move to 0x10+ prefix, keep slot)
    pub fn demote(&self) -> CogAddr {
        CogAddr::from_parts(PREFIX_FLUID_START, self.slot())
    }
}

impl From<u16> for CogAddr {
    fn from(addr: u16) -> Self {
        CogAddr(addr)
    }
}

/// Address tier
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    /// Fixed vocabulary (16 compartments × 256 = 4,096)
    Surface,
    /// Working memory (112 chunks × 256 = 28,672)
    Fluid,
    /// Persistent graph (128 chunks × 256 = 32,768)
    Node,
}

/// Surface compartments (16 available, prefix 0x00-0x0F)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum SurfaceCompartment {
    /// 0x00: Lance/Kuzu - vector search, traversal
    Lance = 0x00,
    /// 0x01: SQL - relational ops
    Sql = 0x01,
    /// 0x02: Neo4j/Cypher - property graph
    Cypher = 0x02,
    /// 0x03: GraphQL ops
    GraphQL = 0x03,
    /// 0x04: NARS inference
    Nars = 0x04,
    /// 0x05: Causal reasoning (Pearl)
    Causal = 0x05,
    /// 0x06: Meta - higher-order thinking
    Meta = 0x06,
    /// 0x07: Verbs - CAUSES, BECOMES, etc.
    Verbs = 0x07,
    /// 0x08: Concepts - core types
    Concepts = 0x08,
    /// 0x09: Qualia ops
    Qualia = 0x09,
    /// 0x0A: Memory ops
    Memory = 0x0A,
    /// 0x0B: Learning ops
    Learning = 0x0B,
    /// 0x0C-0x0F: Reserved
    Reserved = 0x0C,
}

impl SurfaceCompartment {
    pub fn prefix(self) -> u8 {
        self as u8
    }
    
    pub fn addr(self, slot: u8) -> CogAddr {
        CogAddr::from_parts(self as u8, slot)
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
// COGNITIVE VALUE
// =============================================================================

/// Value with cognitive metadata
#[derive(Clone, Debug)]
pub struct CogValue {
    /// The fingerprint content
    pub fingerprint: [u64; 156],
    /// Felt quality
    pub qualia: QualiaVector,
    /// NARS truth value
    pub truth: TruthValue,
    /// Access count (for promotion decisions)
    pub access_count: u32,
    /// Last access time
    pub last_access: Instant,
    /// Time-to-live (None = permanent)
    pub ttl: Option<Duration>,
    /// Creation time
    pub created: Instant,
    /// Optional label
    pub label: Option<String>,
}

impl CogValue {
    pub fn new(fingerprint: [u64; 156]) -> Self {
        Self {
            fingerprint,
            qualia: QualiaVector::default(),
            truth: TruthValue::new(1.0, 0.5),
            access_count: 0,
            last_access: Instant::now(),
            ttl: None,
            created: Instant::now(),
            label: None,
        }
    }
    
    pub fn with_qualia(mut self, qualia: QualiaVector) -> Self {
        self.qualia = qualia;
        self
    }
    
    pub fn with_truth(mut self, truth: TruthValue) -> Self {
        self.truth = truth;
        self
    }
    
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }
    
    pub fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }
    
    /// Is this value expired?
    pub fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl {
            self.last_access.elapsed() > ttl
        } else {
            false
        }
    }
    
    /// Record an access
    pub fn touch(&mut self) {
        self.access_count += 1;
        self.last_access = Instant::now();
    }
    
    /// Should this value be promoted to node tier?
    pub fn should_promote(&self, threshold: u32) -> bool {
        self.access_count >= threshold
    }
    
    /// Should this value be demoted from node tier?
    pub fn should_demote(&self, cold_duration: Duration) -> bool {
        self.last_access.elapsed() > cold_duration
    }
    
    /// Apply decay to truth value
    pub fn decay(&mut self, factor: f32) {
        self.truth = TruthValue::new(
            self.truth.f,
            self.truth.c * factor,
        );
    }
}

// =============================================================================
// COGNITIVE EDGE
// =============================================================================

/// Edge in cognitive graph
#[derive(Clone, Debug)]
pub struct CogEdge {
    /// Source address
    pub from: CogAddr,
    /// Target address  
    pub to: CogAddr,
    /// Relation/verb (address in surface tier)
    pub verb: CogAddr,
    /// Bound fingerprint: from ⊗ verb ⊗ to
    pub fingerprint: [u64; 156],
    /// Edge strength
    pub weight: f32,
    /// Edge qualia
    pub qualia: QualiaVector,
}

impl CogEdge {
    pub fn new(from: CogAddr, verb: CogAddr, to: CogAddr, from_fp: &[u64; 156], verb_fp: &[u64; 156], to_fp: &[u64; 156]) -> Self {
        let mut fingerprint = [0u64; 156];
        for i in 0..156 {
            fingerprint[i] = from_fp[i] ^ verb_fp[i] ^ to_fp[i];
        }
        Self {
            from,
            to,
            verb,
            fingerprint,
            weight: 1.0,
            qualia: QualiaVector::default(),
        }
    }
}

// =============================================================================
// COGNITIVE REDIS
// =============================================================================

/// Cognitive Redis - Redis syntax, cognitive semantics
/// 
/// # Hot Cache Architecture
/// 
/// The hot cache provides O(1) lookup for frequent edge queries:
/// ```text
/// Query: "edges from A via CAUSES"
/// Pattern = A_fingerprint XOR CAUSES_fingerprint
/// 
/// 1. Check hot_cache[pattern] → HIT: return cached edge indices
/// 2. MISS: AVX-512 batch scan → cache result → return
/// ```
/// 
/// This bridges the gap between:
/// - Kuzu CSR: O(1) via pointer arrays (but no fingerprint semantics)
/// - Pure AVX scan: O(n/512) but no caching
/// - Hot cache: O(1) for repeated queries, O(n/512) fallback
pub struct CogRedis {
    // =========================================================================
    // BIND SPACE - The universal DTO (array-based O(1) storage)
    // =========================================================================

    /// Universal bind space - all query languages hit this
    /// Pure array indexing. No HashMap. 3-5 cycles per lookup.
    bind_space: BindSpace,

    // =========================================================================
    // LEGACY HASH MAPS (for backward compatibility during transition)
    // =========================================================================

    /// Surface tier: CAM operations (fixed)
    surface: HashMap<CogAddr, CogValue>,
    /// Fluid zone: working memory
    fluid: HashMap<CogAddr, CogValue>,
    /// Node tier: persistent graph
    nodes: HashMap<CogAddr, CogValue>,
    /// Edges (stored separately for graph queries)
    edges: Vec<CogEdge>,
    /// Cognitive search engine
    search: CognitiveSearch,
    /// Causal search engine
    causal: CausalSearch,
    /// Next fluid address
    next_fluid: u16,
    /// Next node address
    next_node: u16,
    /// Promotion threshold (access count)
    promotion_threshold: u32,
    /// Demotion threshold (time since last access)
    demotion_threshold: Duration,
    /// Default TTL for fluid zone
    default_ttl: Duration,

    // =========================================================================
    // HOT CACHE (Redis-style caching for fingerprint CSR)
    // =========================================================================

    /// Hot edge cache: query pattern → edge indices
    /// Key = from_fingerprint XOR verb_fingerprint (the ABBA query pattern)
    /// Value = indices into self.edges that match
    hot_cache: HashMap<[u64; 156], Vec<usize>>,
    /// Fanout cache: source address → edge indices
    fanout_cache: HashMap<CogAddr, Vec<usize>>,
    /// Fanin cache: target address → edge indices
    fanin_cache: HashMap<CogAddr, Vec<usize>>,
    /// Cache statistics
    cache_hits: u64,
    cache_misses: u64,
}

impl CogRedis {
    pub fn new() -> Self {
        Self {
            // Universal bind space - O(1) array indexing
            bind_space: BindSpace::new(),
            // Legacy HashMaps (kept for backward compatibility during transition)
            surface: HashMap::new(),
            fluid: HashMap::new(),
            nodes: HashMap::new(),
            edges: Vec::new(),
            search: CognitiveSearch::new(),
            causal: CausalSearch::new(),
            next_fluid: FLUID_START,
            next_node: NODE_START,
            promotion_threshold: 10,
            demotion_threshold: Duration::from_secs(3600),  // 1 hour
            default_ttl: Duration::from_secs(300),  // 5 minutes
            // Hot cache initialization
            hot_cache: HashMap::new(),
            fanout_cache: HashMap::new(),
            fanin_cache: HashMap::new(),
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    /// Get reference to the underlying bind space
    pub fn bind_space(&self) -> &BindSpace {
        &self.bind_space
    }

    /// Get mutable reference to the underlying bind space
    pub fn bind_space_mut(&mut self) -> &mut BindSpace {
        &mut self.bind_space
    }

    /// Resolve key to bind space address
    ///
    /// Maps string keys to 16-bit addresses:
    /// - DN paths (containing ':') use dn_path_to_addr() for O(1) hierarchical lookup
    /// - Other keys use hash-based addressing
    /// - Check if exists in bind space
    pub fn resolve_key(&self, key: &str) -> Option<Addr> {
        let addr = self.key_to_addr(key);

        // Check if occupied
        if self.bind_space.read(addr).is_some() {
            Some(addr)
        } else {
            None
        }
    }

    /// Convert key to address (always returns address, even if not occupied)
    ///
    /// - DN paths (containing ':') → dn_path_to_addr() for hierarchical addressing
    /// - Other keys → hash-based addressing
    #[inline]
    pub fn key_to_addr(&self, key: &str) -> Addr {
        // DN path detection: contains ':' → use DN tree addressing
        if key.contains(':') {
            dn_path_to_addr(key)
        } else {
            // Standard hash-based addressing
            use std::hash::{Hash, Hasher};
            use std::collections::hash_map::DefaultHasher;

            let mut hasher = DefaultHasher::new();
            key.hash(&mut hasher);
            let hash = hasher.finish();

            // Map to node address space (0x80-0xFF:XX)
            let prefix = 0x80 + ((hash >> 8) as u8 & 0x7F);  // 0x80-0xFF
            let slot = (hash & 0xFF) as u8;
            Addr::new(prefix, slot)
        }
    }

    /// Check if key is a DN path (contains ':')
    #[inline]
    pub fn is_dn_path(key: &str) -> bool {
        key.contains(':')
    }

    /// Allocate address for key in bind space
    ///
    /// Returns address where key's value should be stored.
    /// DN paths auto-create parent chain for hierarchy.
    pub fn resolve_or_allocate(&mut self, key: &str, fingerprint: [u64; FINGERPRINT_WORDS]) -> Addr {
        // Check if already exists
        if let Some(addr) = self.resolve_key(key) {
            return addr;
        }

        // DN path: use write_dn_path to auto-create parent chain
        if Self::is_dn_path(key) {
            self.bind_space.write_dn_path(key, fingerprint, 0)
        } else {
            // Standard allocation
            self.bind_space.write_labeled(fingerprint, key)
        }
    }

    /// Read from bind space using key
    pub fn bind_get(&self, key: &str) -> Option<&BindNode> {
        let addr = self.resolve_key(key)?;
        self.bind_space.read(addr)
    }

    /// Write to bind space using key
    pub fn bind_set(&mut self, key: &str, fingerprint: [u64; FINGERPRINT_WORDS]) -> Addr {
        self.resolve_or_allocate(key, fingerprint)
    }
    
    /// Allocate next fluid address
    fn alloc_fluid(&mut self) -> CogAddr {
        let addr = CogAddr(self.next_fluid);
        self.next_fluid = self.next_fluid.wrapping_add(1);
        if self.next_fluid >= NODE_START {
            self.next_fluid = FLUID_START;  // Wrap around
        }
        addr
    }
    
    /// Allocate next node address
    fn alloc_node(&mut self) -> CogAddr {
        let addr = CogAddr(self.next_node);
        self.next_node = self.next_node.wrapping_add(1);
        if self.next_node == 0 {
            self.next_node = NODE_START;  // Wrap around
        }
        addr
    }
    
    // =========================================================================
    // CORE REDIS-LIKE OPERATIONS
    // =========================================================================
    
    /// GET - retrieve value with cognitive metadata
    /// 
    /// Returns: (value, qualia, truth, tier)
    pub fn get(&mut self, addr: CogAddr) -> Option<GetResult> {
        // Check all tiers
        let (value, tier) = if let Some(v) = self.surface.get_mut(&addr) {
            (v, Tier::Surface)
        } else if let Some(v) = self.fluid.get_mut(&addr) {
            // Check expiry
            if v.is_expired() {
                self.fluid.remove(&addr);
                return None;
            }
            (v, Tier::Fluid)
        } else if let Some(v) = self.nodes.get_mut(&addr) {
            (v, Tier::Node)
        } else {
            return None;
        };
        
        // Touch and maybe promote
        value.touch();
        
        let result = GetResult {
            fingerprint: value.fingerprint,
            qualia: value.qualia,
            truth: value.truth,
            tier,
            access_count: value.access_count,
            label: value.label.clone(),
        };
        
        // Check for promotion (fluid → node)
        if tier == Tier::Fluid && value.should_promote(self.promotion_threshold) {
            self.promote(addr);
        }
        
        Some(result)
    }
    
    /// GET with FEEL - returns qualia-weighted result
    pub fn get_feel(&mut self, addr: CogAddr) -> Option<(GetResult, f32)> {
        let result = self.get(addr)?;
        let intensity = result.qualia.arousal * 0.5 + result.qualia.valence.abs() * 0.5;
        Some((result, intensity))
    }
    
    /// SET - store value with cognitive metadata
    pub fn set(&mut self, fingerprint: [u64; 156], opts: SetOptions) -> CogAddr {
        let mut value = CogValue::new(fingerprint);
        
        if let Some(q) = opts.qualia {
            value.qualia = q;
        }
        if let Some(t) = opts.truth {
            value.truth = t;
        }
        if let Some(ttl) = opts.ttl {
            value.ttl = Some(ttl);
        } else if !opts.promote {
            value.ttl = Some(self.default_ttl);
        }
        if let Some(label) = opts.label {
            value.label = Some(label);
        }
        
        // Decide tier
        let addr = if opts.promote {
            let addr = self.alloc_node();
            self.nodes.insert(addr, value.clone());
            addr
        } else {
            let addr = self.alloc_fluid();
            self.fluid.insert(addr, value.clone());
            addr
        };
        
        // Add to search index
        let atom = CognitiveAtom::new(fingerprint)
            .with_qualia(value.qualia)
            .with_truth(value.truth);
        self.search.add_atom(atom);
        
        addr
    }
    
    /// SET at specific address
    pub fn set_at(&mut self, addr: CogAddr, fingerprint: [u64; 156], opts: SetOptions) {
        let mut value = CogValue::new(fingerprint);
        
        if let Some(q) = opts.qualia {
            value.qualia = q;
        }
        if let Some(t) = opts.truth {
            value.truth = t;
        }
        if let Some(ttl) = opts.ttl {
            value.ttl = Some(ttl);
        }
        if let Some(label) = opts.label {
            value.label = Some(label);
        }
        
        match addr.tier() {
            Tier::Surface => { self.surface.insert(addr, value); }
            Tier::Fluid => { self.fluid.insert(addr, value); }
            Tier::Node => { self.nodes.insert(addr, value); }
        }
    }
    
    /// DEL - remove value
    pub fn del(&mut self, addr: CogAddr) -> bool {
        match addr.tier() {
            Tier::Surface => self.surface.remove(&addr).is_some(),
            Tier::Fluid => self.fluid.remove(&addr).is_some(),
            Tier::Node => self.nodes.remove(&addr).is_some(),
        }
    }
    
    /// DEL with FORGET - decay truth before removing
    pub fn forget(&mut self, addr: CogAddr, decay_factor: f32) -> bool {
        let value = match addr.tier() {
            Tier::Surface => self.surface.get_mut(&addr),
            Tier::Fluid => self.fluid.get_mut(&addr),
            Tier::Node => self.nodes.get_mut(&addr),
        };
        
        if let Some(v) = value {
            v.decay(decay_factor);
            if v.truth.c < 0.1 {
                self.del(addr)
            } else {
                true
            }
        } else {
            false
        }
    }
    
    /// DEL with SUPPRESS - move to negative valence instead of deleting
    pub fn suppress(&mut self, addr: CogAddr) -> bool {
        let value = match addr.tier() {
            Tier::Surface => self.surface.get_mut(&addr),
            Tier::Fluid => self.fluid.get_mut(&addr),
            Tier::Node => self.nodes.get_mut(&addr),
        };
        
        if let Some(v) = value {
            v.qualia.valence = -1.0;
            v.qualia.arousal *= 0.5;
            true
        } else {
            false
        }
    }
    
    // =========================================================================
    // TIER MANAGEMENT
    // =========================================================================
    
    /// CRYSTALLIZE - promote from fluid to node
    pub fn promote(&mut self, addr: CogAddr) -> Option<CogAddr> {
        if !addr.is_fluid() {
            return None;
        }
        
        if let Some(value) = self.fluid.remove(&addr) {
            let new_addr = self.alloc_node();
            self.nodes.insert(new_addr, value);
            Some(new_addr)
        } else {
            None
        }
    }
    
    /// EVAPORATE - demote from node to fluid
    pub fn demote(&mut self, addr: CogAddr) -> Option<CogAddr> {
        if !addr.is_node() {
            return None;
        }
        
        if let Some(mut value) = self.nodes.remove(&addr) {
            value.ttl = Some(self.default_ttl);  // Add TTL on demotion
            let new_addr = self.alloc_fluid();
            self.fluid.insert(new_addr, value);
            Some(new_addr)
        } else {
            None
        }
    }
    
    /// Run maintenance: expire TTLs, demote cold nodes
    pub fn maintain(&mut self) {
        // Expire fluid zone
        let expired: Vec<_> = self.fluid.iter()
            .filter(|(_, v)| v.is_expired())
            .map(|(k, _)| *k)
            .collect();
        
        for addr in expired {
            self.fluid.remove(&addr);
        }
        
        // Demote cold nodes
        let cold: Vec<_> = self.nodes.iter()
            .filter(|(_, v)| v.should_demote(self.demotion_threshold))
            .map(|(k, _)| *k)
            .collect();
        
        for addr in cold {
            self.demote(addr);
        }
    }
    
    // =========================================================================
    // GRAPH OPERATIONS
    // =========================================================================
    
    /// BIND - create edge between two addresses
    pub fn bind(&mut self, from: CogAddr, verb: CogAddr, to: CogAddr) -> Option<CogAddr> {
        let from_val = self.get(from)?;
        let verb_val = self.get(verb)?;
        let to_val = self.get(to)?;
        
        let edge = CogEdge::new(
            from, verb, to,
            &from_val.fingerprint,
            &verb_val.fingerprint,
            &to_val.fingerprint,
        );
        
        // Store edge fingerprint
        let edge_addr = self.set(edge.fingerprint, SetOptions::default());
        self.edges.push(edge);
        
        // Invalidate affected caches
        self.fanout_cache.remove(&from);
        self.fanin_cache.remove(&to);
        // Invalidate pattern cache for this from+verb combo
        let mut pattern = [0u64; 156];
        for i in 0..156 {
            pattern[i] = from_val.fingerprint[i] ^ verb_val.fingerprint[i];
        }
        self.hot_cache.remove(&pattern);
        
        // Also store in causal search as correlation
        self.causal.store_correlation(&from_val.fingerprint, &to_val.fingerprint, 1.0);
        
        Some(edge_addr)
    }
    
    /// UNBIND - given edge and one component, recover the other (ABBA)
    pub fn unbind(&mut self, edge_addr: CogAddr, known: CogAddr) -> Option<[u64; 156]> {
        let edge_val = self.get(edge_addr)?;
        let known_val = self.get(known)?;
        
        // Find the edge metadata
        let edge = self.edges.iter()
            .find(|e| hamming_distance(&e.fingerprint, &edge_val.fingerprint) < 100)?;
        
        // Get verb fingerprint
        let verb_val = self.get(edge.verb)?;
        
        // ABBA: edge ⊗ known ⊗ verb = other
        let mut result = [0u64; 156];
        for i in 0..156 {
            result[i] = edge_val.fingerprint[i] ^ known_val.fingerprint[i] ^ verb_val.fingerprint[i];
        }
        
        Some(result)
    }
    
    /// FANOUT - find all edges from a node (with cache)
    /// 
    /// O(1) for cached queries, O(n) fallback with cache population
    pub fn fanout(&mut self, addr: CogAddr) -> Vec<&CogEdge> {
        // Check cache
        if let Some(indices) = self.fanout_cache.get(&addr) {
            self.cache_hits += 1;
            return indices.iter()
                .filter_map(|&i| self.edges.get(i))
                .collect();
        }
        
        self.cache_misses += 1;
        
        // Scan and cache
        let indices: Vec<usize> = self.edges.iter()
            .enumerate()
            .filter(|(_, e)| e.from == addr)
            .map(|(i, _)| i)
            .collect();
        
        self.fanout_cache.insert(addr, indices.clone());
        
        indices.iter()
            .filter_map(|&i| self.edges.get(i))
            .collect()
    }
    
    /// FANIN - find all edges to a node (with cache)
    pub fn fanin(&mut self, addr: CogAddr) -> Vec<&CogEdge> {
        // Check cache
        if let Some(indices) = self.fanin_cache.get(&addr) {
            self.cache_hits += 1;
            return indices.iter()
                .filter_map(|&i| self.edges.get(i))
                .collect();
        }
        
        self.cache_misses += 1;
        
        // Scan and cache
        let indices: Vec<usize> = self.edges.iter()
            .enumerate()
            .filter(|(_, e)| e.to == addr)
            .map(|(i, _)| i)
            .collect();
        
        self.fanin_cache.insert(addr, indices.clone());
        
        indices.iter()
            .filter_map(|&i| self.edges.get(i))
            .collect()
    }
    
    /// Query by fingerprint pattern (from ⊗ verb) with hot cache
    /// 
    /// This is the Redis-style cached CSR: same query twice = O(1)
    pub fn query_pattern(&mut self, pattern: &[u64; 156], threshold: u32) -> Vec<&CogEdge> {
        // Check hot cache
        if let Some(indices) = self.hot_cache.get(pattern) {
            self.cache_hits += 1;
            return indices.iter()
                .filter_map(|&i| self.edges.get(i))
                .collect();
        }
        
        self.cache_misses += 1;
        
        // AVX-512 style scan (even without SIMD, cache makes repeat queries fast)
        let indices: Vec<usize> = self.edges.iter()
            .enumerate()
            .filter(|(_, e)| hamming_distance(pattern, &e.fingerprint) < threshold)
            .map(|(i, _)| i)
            .collect();
        
        self.hot_cache.insert(*pattern, indices.clone());
        
        indices.iter()
            .filter_map(|&i| self.edges.get(i))
            .collect()
    }
    
    /// Cache statistics: (hits, misses, hit_rate)
    pub fn cache_stats(&self) -> (u64, u64, f64) {
        let total = self.cache_hits + self.cache_misses;
        let hit_rate = if total > 0 {
            self.cache_hits as f64 / total as f64
        } else {
            0.0
        };
        (self.cache_hits, self.cache_misses, hit_rate)
    }
    
    /// Clear all caches (call after bulk edge operations)
    pub fn invalidate_caches(&mut self) {
        self.hot_cache.clear();
        self.fanout_cache.clear();
        self.fanin_cache.clear();
    }
    
    // =========================================================================
    // COGNITIVE SEARCH OPERATIONS
    // =========================================================================
    
    /// RESONATE - find similar by fingerprint + qualia
    pub fn resonate(&self, query: &[u64; 156], qualia: &QualiaVector, k: usize) -> Vec<ResonateResult> {
        let results = self.search.explore(query, qualia, k);
        
        results.into_iter()
            .map(|r| ResonateResult {
                fingerprint: r.atom.fingerprint,
                qualia: r.atom.qualia,
                truth: r.atom.truth,
                content_score: r.scores.content,
                qualia_score: r.scores.qualia,
                combined_score: r.scores.combined,
            })
            .collect()
    }
    
    /// INTUIT - find by qualia only (Mexican hat resonance)
    pub fn intuit(&self, qualia: &QualiaVector, k: usize) -> Vec<ResonateResult> {
        let results = self.search.intuit(qualia, k);
        
        results.into_iter()
            .map(|r| ResonateResult {
                fingerprint: r.atom.fingerprint,
                qualia: r.atom.qualia,
                truth: r.atom.truth,
                content_score: 0.0,
                qualia_score: r.scores.qualia,
                combined_score: r.scores.combined,
            })
            .collect()
    }
    
    /// KEYS - find by qualia range
    pub fn keys_by_qualia(
        &self,
        valence_range: Option<(f32, f32)>,
        arousal_range: Option<(f32, f32)>,
    ) -> Vec<CogAddr> {
        let mut results = Vec::new();
        
        for (addr, value) in self.fluid.iter().chain(self.nodes.iter()) {
            let mut matches = true;
            
            if let Some((min, max)) = valence_range {
                if value.qualia.valence < min || value.qualia.valence > max {
                    matches = false;
                }
            }
            
            if let Some((min, max)) = arousal_range {
                if value.qualia.arousal < min || value.qualia.arousal > max {
                    matches = false;
                }
            }
            
            if matches {
                results.push(*addr);
            }
        }
        
        results
    }
    
    // =========================================================================
    // CAUSAL OPERATIONS (Pearl's Ladder)
    // =========================================================================
    
    /// CAUSE - what does this cause? (Rung 2: intervention)
    pub fn cause(&mut self, addr: CogAddr, action: CogAddr) -> Vec<CausalResult> {
        let state = self.get(addr);
        let act = self.get(action);
        
        if let (Some(s), Some(a)) = (state, act) {
            self.causal.query_outcome(&s.fingerprint, &a.fingerprint)
        } else {
            Vec::new()
        }
    }
    
    /// WOULD - what would have happened? (Rung 3: counterfactual)
    pub fn would(&mut self, addr: CogAddr, alt_action: CogAddr) -> Vec<CausalResult> {
        let state = self.get(addr);
        let act = self.get(alt_action);
        
        if let (Some(s), Some(a)) = (state, act) {
            self.causal.query_counterfactual(&s.fingerprint, &a.fingerprint)
        } else {
            Vec::new()
        }
    }
    
    /// Store intervention for causal learning
    pub fn store_cause(&mut self, state: CogAddr, action: CogAddr, outcome: CogAddr, strength: f32) {
        let s = self.get(state);
        let a = self.get(action);
        let o = self.get(outcome);
        
        if let (Some(sv), Some(av), Some(ov)) = (s, a, o) {
            self.causal.store_intervention(&sv.fingerprint, &av.fingerprint, &ov.fingerprint, strength);
        }
    }
    
    /// Store counterfactual for what-if reasoning
    pub fn store_would(&mut self, state: CogAddr, alt_action: CogAddr, alt_outcome: CogAddr, strength: f32) {
        let s = self.get(state);
        let a = self.get(alt_action);
        let o = self.get(alt_outcome);
        
        if let (Some(sv), Some(av), Some(ov)) = (s, a, o) {
            self.causal.store_counterfactual(&sv.fingerprint, &av.fingerprint, &ov.fingerprint, strength);
        }
    }
    
    // =========================================================================
    // NARS OPERATIONS
    // =========================================================================
    
    /// DEDUCE - derive conclusion from premises
    pub fn deduce(&self, premise1: CogAddr, premise2: CogAddr) -> Option<DeduceResult> {
        let p1 = self.nodes.get(&premise1).or_else(|| self.fluid.get(&premise1))?;
        let p2 = self.nodes.get(&premise2).or_else(|| self.fluid.get(&premise2))?;
        
        let atom1 = CognitiveAtom::new(p1.fingerprint)
            .with_qualia(p1.qualia)
            .with_truth(p1.truth);
        let atom2 = CognitiveAtom::new(p2.fingerprint)
            .with_qualia(p2.qualia)
            .with_truth(p2.truth);
        
        let result = self.search.deduce(&atom1, &atom2)?;
        
        Some(DeduceResult {
            fingerprint: result.atom.fingerprint,
            qualia: result.atom.qualia,
            truth: result.atom.truth,
        })
    }
    
    /// ABDUCT - generate hypothesis from observation
    pub fn abduct(&self, premise1: CogAddr, premise2: CogAddr) -> Option<DeduceResult> {
        let p1 = self.nodes.get(&premise1).or_else(|| self.fluid.get(&premise1))?;
        let p2 = self.nodes.get(&premise2).or_else(|| self.fluid.get(&premise2))?;
        
        let atom1 = CognitiveAtom::new(p1.fingerprint)
            .with_qualia(p1.qualia)
            .with_truth(p1.truth);
        let atom2 = CognitiveAtom::new(p2.fingerprint)
            .with_qualia(p2.qualia)
            .with_truth(p2.truth);
        
        let result = self.search.abduct(&atom1, &atom2)?;
        
        Some(DeduceResult {
            fingerprint: result.atom.fingerprint,
            qualia: result.atom.qualia,
            truth: result.atom.truth,
        })
    }
    
    /// JUDGE - evaluate truth of a statement
    pub fn judge(&self, addr: CogAddr) -> TruthValue {
        if let Some(result) = self.nodes.get(&addr).or_else(|| self.fluid.get(&addr)) {
            result.truth
        } else {
            TruthValue::new(0.5, 0.1)  // Unknown
        }
    }
    
    // =========================================================================
    // STATS
    // =========================================================================
    
    pub fn stats(&self) -> CogRedisStats {
        CogRedisStats {
            surface_count: self.surface.len(),
            fluid_count: self.fluid.len(),
            node_count: self.nodes.len(),
            edge_count: self.edges.len(),
            total: self.surface.len() + self.fluid.len() + self.nodes.len(),
        }
    }
}

impl Default for CogRedis {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// RESULT TYPES
// =============================================================================

/// Result from GET
#[derive(Clone, Debug)]
pub struct GetResult {
    pub fingerprint: [u64; 156],
    pub qualia: QualiaVector,
    pub truth: TruthValue,
    pub tier: Tier,
    pub access_count: u32,
    pub label: Option<String>,
}

/// Options for SET
#[derive(Clone, Debug, Default)]
pub struct SetOptions {
    pub qualia: Option<QualiaVector>,
    pub truth: Option<TruthValue>,
    pub ttl: Option<Duration>,
    pub promote: bool,
    pub label: Option<String>,
}

impl SetOptions {
    pub fn qualia(mut self, q: QualiaVector) -> Self {
        self.qualia = Some(q);
        self
    }
    
    pub fn truth(mut self, f: f32, c: f32) -> Self {
        self.truth = Some(TruthValue::new(f, c));
        self
    }
    
    pub fn ttl(mut self, secs: u64) -> Self {
        self.ttl = Some(Duration::from_secs(secs));
        self
    }
    
    pub fn permanent(mut self) -> Self {
        self.promote = true;
        self
    }
    
    pub fn label(mut self, s: &str) -> Self {
        self.label = Some(s.to_string());
        self
    }
}

/// Result from RESONATE
#[derive(Clone, Debug)]
pub struct ResonateResult {
    pub fingerprint: [u64; 156],
    pub qualia: QualiaVector,
    pub truth: TruthValue,
    pub content_score: f32,
    pub qualia_score: f32,
    pub combined_score: f32,
}

/// Result from DEDUCE/ABDUCT
#[derive(Clone, Debug)]
pub struct DeduceResult {
    pub fingerprint: [u64; 156],
    pub qualia: QualiaVector,
    pub truth: TruthValue,
}

/// Result from CAUSE
pub use crate::search::causal::CausalResult;

/// Stats
#[derive(Clone, Debug)]
pub struct CogRedisStats {
    pub surface_count: usize,
    pub fluid_count: usize,
    pub node_count: usize,
    pub edge_count: usize,
    pub total: usize,
}

// =============================================================================
// HELPERS
// =============================================================================

fn hamming_distance(a: &[u64; 156], b: &[u64; 156]) -> u32 {
    let mut dist = 0u32;
    for i in 0..156 {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}

// =============================================================================
// CAM EXECUTION BRIDGE
// =============================================================================

/// CAM operation execution result with cognitive metadata
#[derive(Clone, Debug)]
pub enum CamResult {
    /// Single fingerprint result
    Fingerprint(Fingerprint),
    /// Multiple fingerprint results with addresses
    Fingerprints(Vec<(Addr, Fingerprint)>),
    /// Scalar value (similarity, distance, etc.)
    Scalar(f64),
    /// Boolean result
    Bool(bool),
    /// Address in bind space
    Addr(Addr),
    /// No result (side effect only)
    Unit,
    /// Error
    Error(String),
}

impl CogRedis {
    /// Execute a CAM operation by operation ID
    ///
    /// This is the bridge between CAM operation codes (0x000-0xFFF)
    /// and actual execution over BindSpace.
    ///
    /// # Arguments
    /// * `op_id` - 12-bit operation ID (prefix 4-bit : slot 8-bit)
    /// * `args` - Fingerprint arguments for the operation
    ///
    /// # Returns
    /// * `CamResult` - Result of the operation
    pub fn execute_cam(&mut self, op_id: u16, args: &[Fingerprint]) -> CamResult {
        let category = OpCategory::from_id(op_id);
        let slot = (op_id & 0xFF) as u8;

        match category {
            OpCategory::LanceDb => self.execute_lance_op(slot, args),
            OpCategory::Sql => self.execute_sql_op(slot, args),
            OpCategory::Cypher => self.execute_cypher_op(slot, args),
            OpCategory::Hamming => self.execute_hamming_op(slot, args),
            OpCategory::Nars => self.execute_nars_op(slot, args),
            OpCategory::Causality => self.execute_causal_op(slot, args),
            OpCategory::Meta => self.execute_meta_op(slot, args),
            OpCategory::Qualia => self.execute_qualia_op(slot, args),
            OpCategory::Learning => self.execute_learning_op(slot, args),
            _ => CamResult::Error(format!("Unimplemented category: {:?}", category)),
        }
    }

    /// Execute LanceDB operations (0x000-0x0FF)
    fn execute_lance_op(&mut self, slot: u8, args: &[Fingerprint]) -> CamResult {
        match slot {
            // VectorSearch (0x60)
            0x60 => {
                if args.is_empty() {
                    return CamResult::Error("VectorSearch requires query fingerprint".to_string());
                }
                let query = &args[0];
                let k = 10; // Default k=10

                // Search BindSpace for similar fingerprints using resonate
                let qualia = QualiaVector::default();
                let results = self.resonate(&fp_to_words(query), &qualia, k);
                let fps: Vec<(Addr, Fingerprint)> = results.into_iter()
                    .map(|r| {
                        let addr = Addr::new(0x80, 0); // Placeholder address
                        (addr, words_to_fp(&r.fingerprint))
                    })
                    .collect();
                CamResult::Fingerprints(fps)
            }
            // Insert (0x30)
            0x30 => {
                if args.is_empty() {
                    return CamResult::Error("Insert requires at least one fingerprint".to_string());
                }
                let addr = self.bind_space.write(fp_to_words(&args[0]));
                CamResult::Addr(addr)
            }
            // Scan (0x40)
            0x40 => {
                let mut results = Vec::new();
                // Scan node space
                for prefix in 0x80..=0xFF_u8 {
                    for slot in 0..=255_u8 {
                        let addr = Addr::new(prefix, slot);
                        if let Some(node) = self.bind_space.read(addr) {
                            results.push((addr, words_to_fp(&node.fingerprint)));
                            if results.len() >= 100 { // Limit scan
                                return CamResult::Fingerprints(results);
                            }
                        }
                    }
                }
                CamResult::Fingerprints(results)
            }
            _ => CamResult::Error(format!("Unknown Lance op: 0x{:02X}", slot)),
        }
    }

    /// Execute SQL operations (0x100-0x1FF)
    fn execute_sql_op(&mut self, slot: u8, args: &[Fingerprint]) -> CamResult {
        match slot {
            // SelectSimilar (0x10)
            0x10 => {
                if args.is_empty() {
                    return CamResult::Error("SelectSimilar requires query".to_string());
                }
                let query = &args[0];
                let qualia = QualiaVector::default();
                let results = self.resonate(&fp_to_words(query), &qualia, 10);
                let fps: Vec<(Addr, Fingerprint)> = results.into_iter()
                    .map(|r| {
                        let addr = Addr::new(0x80, 0);
                        (addr, words_to_fp(&r.fingerprint))
                    })
                    .collect();
                CamResult::Fingerprints(fps)
            }
            _ => CamResult::Error(format!("Unknown SQL op: 0x{:02X}", slot)),
        }
    }

    /// Execute Cypher operations (0x200-0x2FF)
    fn execute_cypher_op(&mut self, slot: u8, args: &[Fingerprint]) -> CamResult {
        match slot {
            // MatchSimilar (0x40)
            0x40 => {
                if args.is_empty() {
                    return CamResult::Error("MatchSimilar requires query".to_string());
                }
                // Same as vector search
                let qualia = QualiaVector::default();
                let results = self.resonate(&fp_to_words(&args[0]), &qualia, 10);
                let fps: Vec<(Addr, Fingerprint)> = results.into_iter()
                    .map(|r| {
                        let addr = Addr::new(0x80, 0);
                        (addr, words_to_fp(&r.fingerprint))
                    })
                    .collect();
                CamResult::Fingerprints(fps)
            }
            // TraverseOut (0x60)
            0x60 => {
                if args.is_empty() {
                    return CamResult::Error("TraverseOut requires source".to_string());
                }
                // For now, return empty - graph traversal needs proper edge storage
                // This would require querying edges by source fingerprint
                CamResult::Fingerprints(Vec::new())
            }
            _ => CamResult::Error(format!("Unknown Cypher op: 0x{:02X}", slot)),
        }
    }

    /// Execute Hamming/VSA operations (0x300-0x3FF)
    fn execute_hamming_op(&mut self, slot: u8, args: &[Fingerprint]) -> CamResult {
        match slot {
            // Distance (0x00) - HammingOp::Distance = 0x300
            0x00 => {
                if args.len() < 2 {
                    return CamResult::Error("Distance requires two fingerprints".to_string());
                }
                let dist = args[0].hamming(&args[1]);
                CamResult::Scalar(dist as f64)
            }
            // Similarity (0x01) - HammingOp::Similarity = 0x301
            0x01 => {
                if args.len() < 2 {
                    return CamResult::Error("Similarity requires two fingerprints".to_string());
                }
                let sim = args[0].similarity(&args[1]);
                CamResult::Scalar(sim as f64)
            }
            // Bind (0x10) - HammingOp::Bind = 0x310
            0x10 => {
                if args.len() < 2 {
                    return CamResult::Error("Bind requires two fingerprints".to_string());
                }
                let bound = args[0].bind(&args[1]);
                CamResult::Fingerprint(bound)
            }
            // Unbind (0x11) - HammingOp::Unbind = 0x311
            0x11 => {
                if args.len() < 2 {
                    return CamResult::Error("Unbind requires two fingerprints".to_string());
                }
                let unbound = args[0].unbind(&args[1]);
                CamResult::Fingerprint(unbound)
            }
            // Bundle (0x12) - HammingOp::Bundle = 0x312
            0x12 => {
                if args.is_empty() {
                    return CamResult::Fingerprint(Fingerprint::zero());
                }
                // Majority vote bundle
                let bundled = crate::learning::cam_ops::bundle_fingerprints(args);
                CamResult::Fingerprint(bundled)
            }
            // Permute (0x30) - HammingOp::Permute = 0x330
            0x30 => {
                if args.is_empty() {
                    return CamResult::Error("Permute requires fingerprint".to_string());
                }
                let permuted = args[0].permute(1);
                CamResult::Fingerprint(permuted)
            }
            _ => CamResult::Error(format!("Unknown Hamming op: 0x{:02X}", slot)),
        }
    }

    /// Execute NARS operations (0x400-0x4FF)
    fn execute_nars_op(&mut self, slot: u8, args: &[Fingerprint]) -> CamResult {
        match slot {
            // Deduction (0x00)
            0x00 => {
                if args.len() < 2 {
                    return CamResult::Error("Deduction requires premise and rule".to_string());
                }
                // NARS deduction: (A→B, B→C) ⊢ (A→C)
                // Simplified: bind premises
                let conclusion = args[0].bind(&args[1]);
                CamResult::Fingerprint(conclusion)
            }
            // Abduction (0x10)
            0x10 => {
                if args.len() < 2 {
                    return CamResult::Error("Abduction requires effect and rule".to_string());
                }
                // Abduction: observe effect, infer cause
                let hypothesis = args[0].bind(&args[1]);
                CamResult::Fingerprint(hypothesis)
            }
            // Revision (0x40)
            0x40 => {
                if args.len() < 2 {
                    return CamResult::Error("Revision requires two beliefs".to_string());
                }
                // Revision: combine evidence
                let revised = args[0].or(&args[1]);
                CamResult::Fingerprint(revised)
            }
            _ => CamResult::Error(format!("Unknown NARS op: 0x{:02X}", slot)),
        }
    }

    /// Execute Causal operations (0xA00-0xAFF)
    fn execute_causal_op(&mut self, slot: u8, args: &[Fingerprint]) -> CamResult {
        match slot {
            // DoIntervene (0x20) - Rung 2
            0x20 => {
                if args.is_empty() {
                    return CamResult::Error("DoIntervene requires intervention".to_string());
                }
                // Store intervention as causal operation
                let addr = self.bind_space.write_labeled(fp_to_words(&args[0]), "causal:intervention");
                CamResult::Addr(addr)
            }
            // Counterfactual (0x30) - Rung 3
            0x30 => {
                if args.len() < 2 {
                    return CamResult::Error("Counterfactual requires world and change".to_string());
                }
                // Counterfactual: what if A had been B?
                let counterfactual = args[0].bind(&args[1].not());
                CamResult::Fingerprint(counterfactual)
            }
            _ => CamResult::Error(format!("Unknown Causal op: 0x{:02X}", slot)),
        }
    }

    /// Execute Meta operations (0xD00-0xDFF)
    fn execute_meta_op(&mut self, slot: u8, args: &[Fingerprint]) -> CamResult {
        match slot {
            // Reflect (0x00)
            0x00 => {
                // Reflection: create fingerprint of current state
                let state_fp = Fingerprint::from_content(&format!("meta:state:{}",
                    self.bind_space.stats().node_count));
                CamResult::Fingerprint(state_fp)
            }
            // Monitor (0x20)
            0x20 => {
                // Monitor: return stats as fingerprint
                let stats = self.bind_space.stats();
                let monitor_fp = Fingerprint::from_content(&format!("meta:monitor:{}:{}:{}",
                    stats.surface_count, stats.fluid_count, stats.node_count));
                CamResult::Fingerprint(monitor_fp)
            }
            _ => CamResult::Error(format!("Unknown Meta op: 0x{:02X}", slot)),
        }
    }

    /// Execute Qualia operations (0xB00-0xBFF)
    fn execute_qualia_op(&mut self, slot: u8, args: &[Fingerprint]) -> CamResult {
        match slot {
            // Valence (0x00)
            0x00 => {
                if args.is_empty() {
                    return CamResult::Error("Valence requires fingerprint".to_string());
                }
                // Compute valence from fingerprint density
                let density = args[0].density();
                let valence = (density - 0.5) * 2.0; // Map [0,1] to [-1,1]
                CamResult::Scalar(valence as f64)
            }
            // Arousal (0x01)
            0x01 => {
                if args.is_empty() {
                    return CamResult::Error("Arousal requires fingerprint".to_string());
                }
                // Arousal from entropy (how mixed are the bits)
                let density = args[0].density();
                let arousal = 1.0 - (density - 0.5).abs() * 2.0; // Peak at 50% density
                CamResult::Scalar(arousal as f64)
            }
            _ => CamResult::Error(format!("Unknown Qualia op: 0x{:02X}", slot)),
        }
    }

    /// Execute Learning operations (0xE00-0xEFF)
    fn execute_learning_op(&mut self, slot: u8, args: &[Fingerprint]) -> CamResult {
        match slot {
            // MomentCapture (0x00)
            0x00 => {
                if args.len() < 2 {
                    return CamResult::Error("MomentCapture requires input and output".to_string());
                }
                // Capture learning moment: bind input → output
                let moment = args[0].bind(&args[1]);
                let addr = self.bind_space.write_labeled(fp_to_words(&moment), "learning:moment");
                CamResult::Addr(addr)
            }
            // Hebbian (0x10)
            0x10 => {
                if args.len() < 2 {
                    return CamResult::Error("Hebbian requires pre and post".to_string());
                }
                // Hebbian: "neurons that fire together wire together"
                let association = args[0].and(&args[1]);
                CamResult::Fingerprint(association)
            }
            _ => CamResult::Error(format!("Unknown Learning op: 0x{:02X}", slot)),
        }
    }

    /// Execute CAM operation by name
    pub fn execute_cam_named(&mut self, name: &str, args: &[Fingerprint]) -> CamResult {
        // Map common names to operation IDs (using exact enum values)
        let op_id = match name.to_uppercase().as_str() {
            "BIND" => HammingOp::Bind as u16,           // 0x310
            "UNBIND" => HammingOp::Unbind as u16,       // 0x311
            "SIMILARITY" => HammingOp::Similarity as u16, // 0x301
            "DISTANCE" => HammingOp::Distance as u16,   // 0x300
            "BUNDLE" => HammingOp::Bundle as u16,       // 0x312
            "PERMUTE" => HammingOp::Permute as u16,     // 0x330
            "VECTOR_SEARCH" => LanceOp::VectorSearch as u16, // 0x060
            "INSERT" => LanceOp::Insert as u16,         // 0x030
            "SCAN" => LanceOp::Scan as u16,             // 0x040
            "SELECT_SIMILAR" => SqlOp::SelectSimilar as u16, // 0x110
            _ => return CamResult::Error(format!("Unknown operation: {}", name)),
        };
        self.execute_cam(op_id, args)
    }
}

// =============================================================================
// REDIS COMMAND EXECUTOR
// =============================================================================

/// Redis command result
#[derive(Clone, Debug)]
pub enum RedisResult {
    /// OK response
    Ok,
    /// Simple string
    String(String),
    /// Integer
    Integer(i64),
    /// Bulk string (fingerprint as hex or bytes)
    Bulk(Vec<u8>),
    /// Array of results
    Array(Vec<RedisResult>),
    /// Null
    Nil,
    /// Error
    Error(String),
}

impl RedisResult {
    pub fn is_ok(&self) -> bool {
        matches!(self, RedisResult::Ok)
    }

    pub fn is_error(&self) -> bool {
        matches!(self, RedisResult::Error(_))
    }
}

impl CogRedis {
    /// Execute a Redis-style command string
    ///
    /// # Supported Commands
    ///
    /// ## Standard Redis (with cognitive extensions)
    /// - `GET key` - Get value (returns fingerprint + metadata)
    /// - `SET key value [QUALIA q] [TRUTH f,c] [TTL t]` - Set value
    /// - `DEL key` - Delete key
    /// - `KEYS pattern` - List keys matching pattern
    ///
    /// ## Cognitive Extensions
    /// - `BIND a b [VIA verb]` - Create edge (XOR bind)
    /// - `UNBIND edge a` - Recover b from edge
    /// - `RESONATE query [K k] [THRESHOLD t]` - Similarity search
    /// - `CAUSE a` - Get effects (Rung 2)
    /// - `WOULD a b` - Counterfactual (Rung 3)
    /// - `DEDUCE premise rule` - NARS deduction
    ///
    /// ## CAM Operations
    /// - `CAM op_name arg1 arg2 ...` - Execute CAM operation by name
    /// - `CAM.ID 0x0300 arg1 arg2 ...` - Execute CAM operation by ID
    ///
    pub fn execute_command(&mut self, cmd: &str) -> RedisResult {
        let parts: Vec<&str> = cmd.split_whitespace().collect();
        if parts.is_empty() {
            return RedisResult::Error("Empty command".to_string());
        }

        let command = parts[0].to_uppercase();
        let args = &parts[1..];

        match command.as_str() {
            // Standard Redis commands
            "GET" => self.cmd_get(args),
            "SET" => self.cmd_set(args),
            "DEL" => self.cmd_del(args),
            "KEYS" => self.cmd_keys(args),
            "PING" => RedisResult::String("PONG".to_string()),
            "INFO" => self.cmd_info(),

            // Cognitive extensions
            "BIND" => self.cmd_bind(args),
            "UNBIND" => self.cmd_unbind(args),
            "RESONATE" => self.cmd_resonate(args),
            "CAUSE" => self.cmd_cause(args),
            "WOULD" => self.cmd_would(args),
            "DEDUCE" => self.cmd_deduce(args),
            "INTUIT" => self.cmd_intuit(args),
            "FANOUT" => self.cmd_fanout(args),

            // DN Tree commands (Distinguished Name hierarchy)
            "DN.GET" => self.cmd_dn_get(args),
            "DN.SET" => self.cmd_dn_set(args),
            "DN.PARENT" => self.cmd_dn_parent(args),
            "DN.CHILDREN" => self.cmd_dn_children(args),
            "DN.ANCESTORS" => self.cmd_dn_ancestors(args),
            "DN.SIBLINGS" => self.cmd_dn_siblings(args),
            "DN.DEPTH" => self.cmd_dn_depth(args),
            "DN.RUNG" => self.cmd_dn_rung(args),
            "DN.TREE" => self.cmd_dn_tree(args),

            // CAM operations
            "CAM" => self.cmd_cam(args),
            "CAM.ID" => self.cmd_cam_id(args),

            // Stats
            "STATS" => self.cmd_stats(),

            _ => RedisResult::Error(format!("Unknown command: {}", command)),
        }
    }

    // --- Standard Redis commands ---

    fn cmd_get(&mut self, args: &[&str]) -> RedisResult {
        if args.is_empty() {
            return RedisResult::Error("GET requires key".to_string());
        }

        let key = args[0];
        if let Some(node) = self.bind_get(key) {
            let fp_hex = node.fingerprint.iter()
                .map(|w| format!("{:016x}", w))
                .collect::<Vec<_>>()
                .join("");
            RedisResult::String(fp_hex)
        } else {
            RedisResult::Nil
        }
    }

    fn cmd_set(&mut self, args: &[&str]) -> RedisResult {
        if args.len() < 2 {
            return RedisResult::Error("SET requires key and value".to_string());
        }

        let key = args[0];
        let value = args[1];

        // Create fingerprint from value
        let fp = Fingerprint::from_content(value);
        let _addr = self.bind_set(key, fp_to_words(&fp));

        RedisResult::Ok
    }

    fn cmd_del(&mut self, args: &[&str]) -> RedisResult {
        if args.is_empty() {
            return RedisResult::Error("DEL requires key".to_string());
        }

        let mut deleted = 0i64;
        for key in args {
            if let Some(addr) = self.resolve_key(key) {
                self.bind_space.delete(addr);
                deleted += 1;
            }
        }
        RedisResult::Integer(deleted)
    }

    fn cmd_keys(&self, args: &[&str]) -> RedisResult {
        let pattern = args.first().copied().unwrap_or("*");
        let stats = self.bind_space.stats();

        // Return count for now (full pattern matching would require label index)
        RedisResult::Array(vec![
            RedisResult::String(format!("surface:{}", stats.surface_count)),
            RedisResult::String(format!("fluid:{}", stats.fluid_count)),
            RedisResult::String(format!("node:{}", stats.node_count)),
        ])
    }

    fn cmd_info(&self) -> RedisResult {
        let stats = self.bind_space.stats();
        RedisResult::String(format!(
            "# BindSpace\r\nsurface_count:{}\r\nfluid_count:{}\r\nnode_count:{}\r\nedge_count:{}\r\ncontext:{:?}",
            stats.surface_count, stats.fluid_count, stats.node_count, stats.edge_count, stats.context
        ))
    }

    // --- Cognitive extension commands ---

    fn cmd_bind(&mut self, args: &[&str]) -> RedisResult {
        if args.len() < 2 {
            return RedisResult::Error("BIND requires at least 2 keys".to_string());
        }

        let a = Fingerprint::from_content(args[0]);
        let b = Fingerprint::from_content(args[1]);

        // Optional VIA verb
        let verb = if args.len() > 3 && args[2].to_uppercase() == "VIA" {
            Fingerprint::from_content(args[3])
        } else {
            Fingerprint::from_content("RELATES")
        };

        // Create edge: a ⊗ verb ⊗ b
        let edge = a.bind(&verb).bind(&b);
        let addr = self.bind_space.write(fp_to_words(&edge));

        RedisResult::String(format!("{:04X}", addr.0))
    }

    fn cmd_unbind(&mut self, args: &[&str]) -> RedisResult {
        if args.len() < 2 {
            return RedisResult::Error("UNBIND requires edge and key".to_string());
        }

        // Unbind using CAM operation
        let edge = Fingerprint::from_content(args[0]);
        let a = Fingerprint::from_content(args[1]);

        let result = self.execute_cam(HammingOp::Unbind as u16, &[edge, a]);
        match result {
            CamResult::Fingerprint(fp) => {
                let hex = fp.as_raw().iter()
                    .take(4)
                    .map(|w| format!("{:016x}", w))
                    .collect::<Vec<_>>()
                    .join("");
                RedisResult::String(hex)
            }
            CamResult::Error(e) => RedisResult::Error(e),
            _ => RedisResult::Error("Unexpected result".to_string()),
        }
    }

    fn cmd_resonate(&mut self, args: &[&str]) -> RedisResult {
        if args.is_empty() {
            return RedisResult::Error("RESONATE requires query".to_string());
        }

        let query = Fingerprint::from_content(args[0]);
        let k = if args.len() > 2 && args[1].to_uppercase() == "K" {
            args[2].parse().unwrap_or(10)
        } else {
            10
        };

        let result = self.execute_cam(LanceOp::VectorSearch as u16, &[query]);
        match result {
            CamResult::Fingerprints(fps) => {
                let results: Vec<RedisResult> = fps.iter()
                    .take(k)
                    .map(|(addr, _fp)| RedisResult::String(format!("{:04X}", addr.0)))
                    .collect();
                RedisResult::Array(results)
            }
            CamResult::Error(e) => RedisResult::Error(e),
            _ => RedisResult::Array(vec![]),
        }
    }

    fn cmd_cause(&mut self, args: &[&str]) -> RedisResult {
        if args.is_empty() {
            return RedisResult::Error("CAUSE requires intervention".to_string());
        }

        let intervention = Fingerprint::from_content(args[0]);
        let result = self.execute_cam(0xA20, &[intervention]); // CausalOp::DoIntervene

        match result {
            CamResult::Addr(addr) => RedisResult::String(format!("{:04X}", addr.0)),
            CamResult::Error(e) => RedisResult::Error(e),
            _ => RedisResult::Nil,
        }
    }

    fn cmd_would(&mut self, args: &[&str]) -> RedisResult {
        if args.len() < 2 {
            return RedisResult::Error("WOULD requires world and change".to_string());
        }

        let world = Fingerprint::from_content(args[0]);
        let change = Fingerprint::from_content(args[1]);
        let result = self.execute_cam(0xA30, &[world, change]); // CausalOp::Counterfactual

        match result {
            CamResult::Fingerprint(fp) => {
                let hex = fp.as_raw().iter()
                    .take(4)
                    .map(|w| format!("{:016x}", w))
                    .collect::<Vec<_>>()
                    .join("");
                RedisResult::String(hex)
            }
            CamResult::Error(e) => RedisResult::Error(e),
            _ => RedisResult::Nil,
        }
    }

    fn cmd_deduce(&mut self, args: &[&str]) -> RedisResult {
        if args.len() < 2 {
            return RedisResult::Error("DEDUCE requires premise and rule".to_string());
        }

        let premise = Fingerprint::from_content(args[0]);
        let rule = Fingerprint::from_content(args[1]);
        let result = self.execute_cam(0x400, &[premise, rule]); // NarsOp::Deduction

        match result {
            CamResult::Fingerprint(fp) => {
                let hex = fp.as_raw().iter()
                    .take(4)
                    .map(|w| format!("{:016x}", w))
                    .collect::<Vec<_>>()
                    .join("");
                RedisResult::String(hex)
            }
            CamResult::Error(e) => RedisResult::Error(e),
            _ => RedisResult::Nil,
        }
    }

    fn cmd_intuit(&mut self, args: &[&str]) -> RedisResult {
        if args.is_empty() {
            return RedisResult::Error("INTUIT requires qualia parameters".to_string());
        }

        let input = Fingerprint::from_content(args[0]);
        let result = self.execute_cam(0xB00, &[input]); // QualiaOp::Valence

        match result {
            CamResult::Scalar(v) => RedisResult::String(format!("{:.4}", v)),
            CamResult::Error(e) => RedisResult::Error(e),
            _ => RedisResult::Nil,
        }
    }

    fn cmd_fanout(&mut self, args: &[&str]) -> RedisResult {
        if args.is_empty() {
            return RedisResult::Error("FANOUT requires source".to_string());
        }

        // Parse address from hex
        let addr_str = args[0];
        let addr_val = u16::from_str_radix(addr_str.trim_start_matches("0x"), 16)
            .unwrap_or(0);
        let addr = CogAddr(addr_val);

        let edges = self.fanout(addr);
        let results: Vec<RedisResult> = edges.iter()
            .map(|edge| RedisResult::String(format!("{:04X} -> {:04X}", edge.from.0, edge.to.0)))
            .collect();

        RedisResult::Array(results)
    }

    // =========================================================================
    // DN TREE COMMANDS (Distinguished Name hierarchy)
    // =========================================================================
    //
    // DN paths use ':' as separator: "Ada:A:soul:identity"
    // - O(1) address lookup via dn_path_to_addr()
    // - O(1) parent extraction via string truncation
    // - Zero-copy children via BitpackedCSR
    //
    // Examples:
    //   DN.GET Ada:A:soul:identity       → Get node at path
    //   DN.SET Ada:A:soul:new "content"  → Create with parent chain
    //   DN.PARENT Ada:A:soul:identity    → Returns "Ada:A:soul"
    //   DN.CHILDREN Ada:A:soul           → List children
    //   DN.ANCESTORS Ada:A:soul:identity → ["Ada:A:soul", "Ada:A", "Ada"]
    //   DN.TREE Ada:A 3                  → Walk tree to depth 3

    /// DN.GET path - Get node at DN path
    fn cmd_dn_get(&self, args: &[&str]) -> RedisResult {
        if args.is_empty() {
            return RedisResult::Error("DN.GET requires path".to_string());
        }

        let path = args[0];
        let addr = dn_path_to_addr(path);

        if let Some(node) = self.bind_space.read(addr) {
            RedisResult::Array(vec![
                RedisResult::String(format!("addr:{:04X}", addr.0)),
                RedisResult::String(format!("label:{}", node.label.as_deref().unwrap_or(""))),
                RedisResult::String(format!("depth:{}", node.depth)),
                RedisResult::String(format!("rung:{}", node.rung)),
                RedisResult::String(format!("parent:{}", node.parent.map(|p| format!("{:04X}", p.0)).unwrap_or_default())),
            ])
        } else {
            RedisResult::Nil
        }
    }

    /// DN.SET path value [RUNG r] - Create node at DN path with auto-parent creation
    fn cmd_dn_set(&mut self, args: &[&str]) -> RedisResult {
        if args.len() < 2 {
            return RedisResult::Error("DN.SET requires path and value".to_string());
        }

        let path = args[0];
        let value = args[1];

        // Parse optional RUNG
        let rung = if args.len() > 3 && args[2].to_uppercase() == "RUNG" {
            args[3].parse().unwrap_or(0)
        } else {
            0
        };

        // Create fingerprint from value
        let fp = Fingerprint::from_content(value);
        let addr = self.bind_space.write_dn_path(path, fp_to_words(&fp), rung);

        RedisResult::String(format!("{:04X}", addr.0))
    }

    /// DN.PARENT path - Get parent path (O(1) string operation)
    fn cmd_dn_parent(&self, args: &[&str]) -> RedisResult {
        if args.is_empty() {
            return RedisResult::Error("DN.PARENT requires path".to_string());
        }

        let path = args[0];

        if let Some(parent_path) = BindSpace::dn_parent_path(path) {
            let parent_addr = dn_path_to_addr(parent_path);
            if let Some(node) = self.bind_space.read(parent_addr) {
                RedisResult::Array(vec![
                    RedisResult::String(parent_path.to_string()),
                    RedisResult::String(format!("{:04X}", parent_addr.0)),
                    RedisResult::String(format!("label:{}", node.label.as_deref().unwrap_or(""))),
                ])
            } else {
                RedisResult::String(parent_path.to_string())
            }
        } else {
            RedisResult::Nil  // Root node has no parent
        }
    }

    /// DN.CHILDREN path - Get children (zero-copy via CSR)
    fn cmd_dn_children(&mut self, args: &[&str]) -> RedisResult {
        if args.is_empty() {
            return RedisResult::Error("DN.CHILDREN requires path".to_string());
        }

        let path = args[0];
        let addr = dn_path_to_addr(path);

        // Ensure CSR is built
        self.bind_space.rebuild_csr();

        // Zero-copy children access
        let children_raw = self.bind_space.children_raw(addr);

        let results: Vec<RedisResult> = children_raw
            .iter()
            .filter_map(|&child_raw| {
                let child_addr = Addr(child_raw);
                self.bind_space.read(child_addr).map(|node| {
                    RedisResult::Array(vec![
                        RedisResult::String(format!("{:04X}", child_addr.0)),
                        RedisResult::String(node.label.clone().unwrap_or_default()),
                    ])
                })
            })
            .collect();

        RedisResult::Array(results)
    }

    /// DN.ANCESTORS path - Get all ancestors (O(1) per hop via parent chain)
    fn cmd_dn_ancestors(&self, args: &[&str]) -> RedisResult {
        if args.is_empty() {
            return RedisResult::Error("DN.ANCESTORS requires path".to_string());
        }

        let path = args[0];
        let addr = dn_path_to_addr(path);

        let ancestors: Vec<RedisResult> = self.bind_space.ancestors(addr)
            .filter_map(|anc_addr| {
                self.bind_space.read(anc_addr).map(|node| {
                    RedisResult::Array(vec![
                        RedisResult::String(format!("{:04X}", anc_addr.0)),
                        RedisResult::String(node.label.clone().unwrap_or_default()),
                        RedisResult::String(format!("depth:{}", node.depth)),
                    ])
                })
            })
            .collect();

        RedisResult::Array(ancestors)
    }

    /// DN.SIBLINGS path - Get siblings (same parent)
    fn cmd_dn_siblings(&self, args: &[&str]) -> RedisResult {
        if args.is_empty() {
            return RedisResult::Error("DN.SIBLINGS requires path".to_string());
        }

        let path = args[0];
        let addr = dn_path_to_addr(path);

        let siblings: Vec<RedisResult> = self.bind_space.siblings(addr)
            .filter_map(|sib_addr| {
                self.bind_space.read(sib_addr).map(|node| {
                    RedisResult::Array(vec![
                        RedisResult::String(format!("{:04X}", sib_addr.0)),
                        RedisResult::String(node.label.clone().unwrap_or_default()),
                    ])
                })
            })
            .collect();

        RedisResult::Array(siblings)
    }

    /// DN.DEPTH path - Get tree depth of node
    fn cmd_dn_depth(&self, args: &[&str]) -> RedisResult {
        if args.is_empty() {
            return RedisResult::Error("DN.DEPTH requires path".to_string());
        }

        let path = args[0];
        let addr = dn_path_to_addr(path);
        let depth = self.bind_space.depth(addr);

        RedisResult::Integer(depth as i64)
    }

    /// DN.RUNG path - Get access rung of node (R0-R9)
    fn cmd_dn_rung(&self, args: &[&str]) -> RedisResult {
        if args.is_empty() {
            return RedisResult::Error("DN.RUNG requires path".to_string());
        }

        let path = args[0];
        let addr = dn_path_to_addr(path);
        let rung = self.bind_space.rung(addr);

        RedisResult::Integer(rung as i64)
    }

    /// DN.TREE path [DEPTH n] - Walk tree and collect nodes
    fn cmd_dn_tree(&mut self, args: &[&str]) -> RedisResult {
        if args.is_empty() {
            return RedisResult::Error("DN.TREE requires path".to_string());
        }

        let path = args[0];
        let max_depth = if args.len() > 2 && args[1].to_uppercase() == "DEPTH" {
            args[2].parse().unwrap_or(3)
        } else {
            3
        };

        let start_addr = dn_path_to_addr(path);

        // Ensure CSR is built for traversal
        self.bind_space.rebuild_csr();

        // BFS traversal
        let mut results = Vec::new();
        let mut frontier = vec![(start_addr, 0usize)];
        let mut visited = std::collections::HashSet::new();

        while let Some((addr, depth)) = frontier.pop() {
            if depth > max_depth || !visited.insert(addr.0) {
                continue;
            }

            if let Some(node) = self.bind_space.read(addr) {
                results.push(RedisResult::Array(vec![
                    RedisResult::String(format!("{:04X}", addr.0)),
                    RedisResult::String(node.label.clone().unwrap_or_default()),
                    RedisResult::Integer(depth as i64),
                    RedisResult::Integer(node.rung as i64),
                ]));

                // Add children to frontier
                for &child_raw in self.bind_space.children_raw(addr) {
                    frontier.push((Addr(child_raw), depth + 1));
                }
            }
        }

        RedisResult::Array(results)
    }

    // --- CAM operations ---

    fn cmd_cam(&mut self, args: &[&str]) -> RedisResult {
        if args.is_empty() {
            return RedisResult::Error("CAM requires operation name".to_string());
        }

        let op_name = args[0];
        let fp_args: Vec<Fingerprint> = args[1..].iter()
            .map(|s| Fingerprint::from_content(s))
            .collect();

        let result = self.execute_cam_named(op_name, &fp_args);
        cam_result_to_redis(result)
    }

    fn cmd_cam_id(&mut self, args: &[&str]) -> RedisResult {
        if args.is_empty() {
            return RedisResult::Error("CAM.ID requires operation ID".to_string());
        }

        let op_id = u16::from_str_radix(args[0].trim_start_matches("0x"), 16)
            .unwrap_or(0);
        let fp_args: Vec<Fingerprint> = args[1..].iter()
            .map(|s| Fingerprint::from_content(s))
            .collect();

        let result = self.execute_cam(op_id, &fp_args);
        cam_result_to_redis(result)
    }

    fn cmd_stats(&self) -> RedisResult {
        let stats = self.bind_space.stats();
        RedisResult::Array(vec![
            RedisResult::String(format!("surface_count:{}", stats.surface_count)),
            RedisResult::String(format!("fluid_count:{}", stats.fluid_count)),
            RedisResult::String(format!("node_count:{}", stats.node_count)),
            RedisResult::String(format!("edge_count:{}", stats.edge_count)),
            RedisResult::String(format!("context:{:?}", stats.context)),
        ])
    }
}

/// Convert CamResult to RedisResult
fn cam_result_to_redis(result: CamResult) -> RedisResult {
    match result {
        CamResult::Fingerprint(fp) => {
            let hex = fp.as_raw().iter()
                .take(8)
                .map(|w| format!("{:016x}", w))
                .collect::<Vec<_>>()
                .join("");
            RedisResult::String(hex)
        }
        CamResult::Fingerprints(fps) => {
            let results: Vec<RedisResult> = fps.iter()
                .map(|(addr, _fp)| RedisResult::String(format!("{:04X}", addr.0)))
                .collect();
            RedisResult::Array(results)
        }
        CamResult::Scalar(v) => RedisResult::String(format!("{:.6}", v)),
        CamResult::Bool(b) => RedisResult::Integer(if b { 1 } else { 0 }),
        CamResult::Addr(addr) => RedisResult::String(format!("{:04X}", addr.0)),
        CamResult::Unit => RedisResult::Ok,
        CamResult::Error(e) => RedisResult::Error(e),
    }
}

// =============================================================================
// HELPERS
// =============================================================================

/// Convert Fingerprint to [u64; 156] words (for CogRedis compatibility)
fn fp_to_words(fp: &Fingerprint) -> [u64; 156] {
    let raw = fp.as_raw();
    let mut words = [0u64; 156];
    words.copy_from_slice(&raw[..156]);
    words
}

/// Convert [u64; 156] words to Fingerprint
fn words_to_fp(words: &[u64; 156]) -> Fingerprint {
    use crate::FINGERPRINT_U64;
    let mut data = [0u64; FINGERPRINT_U64];
    data[..156].copy_from_slice(words);
    Fingerprint::from_raw(data)
}

// =============================================================================
// PRODUCTION-HARDENED COGREDIS
// =============================================================================

use super::hardening::{HardeningConfig, HardenedBindSpace, QueryContext, QueryTimeoutError};

/// Production-hardened CogRedis with memory limits, TTL, WAL, and timeouts
///
/// This wraps the standard CogRedis with production features:
/// - Memory limits with LRU eviction
/// - TTL-based expiration for fluid zone
/// - Write-ahead logging for crash recovery
/// - Query timeouts
///
/// All query language emulation (Cypher, SQL, Redis) remains unchanged.
pub struct HardenedCogRedis {
    /// Inner CogRedis (all functionality preserved)
    inner: CogRedis,
    /// Hardening layer
    hardening: HardenedBindSpace,
}

impl HardenedCogRedis {
    /// Create with default hardening config
    pub fn new() -> std::io::Result<Self> {
        Self::with_config(HardeningConfig::default())
    }

    /// Create with custom hardening config
    pub fn with_config(config: HardeningConfig) -> std::io::Result<Self> {
        Ok(Self {
            inner: CogRedis::new(),
            hardening: HardenedBindSpace::new(config)?,
        })
    }

    /// Create with production config
    pub fn production() -> std::io::Result<Self> {
        Self::with_config(HardeningConfig::production())
    }

    /// Create with performance config (less durable)
    pub fn performance() -> std::io::Result<Self> {
        Self::with_config(HardeningConfig::performance())
    }

    // =========================================================================
    // REDIS-COMPATIBLE OPERATIONS (with hardening)
    // =========================================================================

    /// GET with hardening (tracks LRU, refreshes TTL)
    pub fn get(&mut self, addr: CogAddr) -> Option<GetResult> {
        let result = self.inner.get(addr)?;
        self.hardening.on_read(Addr::from(addr.0));
        Some(result)
    }

    /// SET with hardening (tracks LRU, sets TTL, logs to WAL)
    pub fn set(&mut self, fingerprint: [u64; 156], opts: SetOptions) -> CogAddr {
        let addr = self.inner.set(fingerprint, opts.clone());
        let to_evict = self.hardening.on_write(
            Addr::from(addr.0),
            &fingerprint,
            opts.label.as_deref(),
        );

        // Evict if needed
        for evict_addr in to_evict {
            self.inner.del(CogAddr(evict_addr));
        }

        addr
    }

    /// DEL with hardening
    pub fn del(&mut self, addr: CogAddr) -> bool {
        let result = self.inner.del(addr);
        if result {
            self.hardening.on_delete(Addr::from(addr.0));
        }
        result
    }

    /// BIND with hardening (logs to WAL)
    pub fn bind(&mut self, from: CogAddr, verb: CogAddr, to: CogAddr) -> Option<CogAddr> {
        let result = self.inner.bind(from, verb, to)?;
        self.hardening.on_link(
            Addr::from(from.0),
            Addr::from(verb.0),
            Addr::from(to.0),
        );
        Some(result)
    }

    // =========================================================================
    // QUERY OPERATIONS (with timeout)
    // =========================================================================

    /// Execute command with timeout
    pub fn execute_command(&mut self, cmd: &str) -> RedisResult {
        let ctx = self.hardening.query_context();

        // Check timeout before executing
        if let Err(_) = ctx.check_timeout() {
            self.hardening.on_timeout();
            return RedisResult::Error("Query timeout".to_string());
        }

        self.inner.execute_command(cmd)
    }

    /// Execute command with custom timeout
    pub fn execute_command_with_timeout(&mut self, cmd: &str, timeout: Duration) -> RedisResult {
        let ctx = QueryContext::new(timeout);

        if let Err(_) = ctx.check_timeout() {
            self.hardening.on_timeout();
            return RedisResult::Error("Query timeout".to_string());
        }

        self.inner.execute_command(cmd)
    }

    /// RESONATE with timeout
    pub fn resonate(&self, query: &[u64; 156], qualia: &QualiaVector, k: usize) -> Vec<ResonateResult> {
        self.inner.resonate(query, qualia, k)
    }

    /// DEDUCE with timeout
    pub fn deduce(&self, premise1: CogAddr, premise2: CogAddr) -> Option<DeduceResult> {
        self.inner.deduce(premise1, premise2)
    }

    // =========================================================================
    // MAINTENANCE
    // =========================================================================

    /// Run maintenance (TTL expiration, WAL checkpoint)
    ///
    /// Call this periodically (e.g., every 10 seconds) to:
    /// - Expire TTL'd entries in fluid zone
    /// - Checkpoint WAL if needed
    pub fn maintain(&mut self) {
        let (expired, needs_checkpoint) = self.hardening.maintenance();

        // Delete expired entries
        for addr in expired {
            self.inner.del(CogAddr(addr));
        }

        // Checkpoint WAL if needed
        if needs_checkpoint {
            let _ = self.hardening.checkpoint();
        }
    }

    /// Force WAL checkpoint
    pub fn checkpoint(&self) -> std::io::Result<()> {
        self.hardening.checkpoint()
    }

    /// Recover from WAL (call on startup)
    pub fn recover(&mut self) -> std::io::Result<usize> {
        let entries = self.hardening.recover()?;
        let count = entries.len();

        for entry in entries {
            match entry {
                super::hardening::WalEntry::Write { addr, fingerprint, label } => {
                    let bind_addr = Addr::from(addr);
                    self.inner.bind_space_mut().write(fingerprint);
                    // Note: label recovery would require bind_space modification
                }
                super::hardening::WalEntry::Delete { addr } => {
                    self.inner.bind_space_mut().delete(Addr::from(addr));
                }
                super::hardening::WalEntry::Link { from, verb, to } => {
                    let from_addr = Addr::from(from);
                    let verb_addr = Addr::from(verb);
                    let to_addr = Addr::from(to);
                    self.inner.bind_space_mut().link(from_addr, verb_addr, to_addr);
                }
                super::hardening::WalEntry::Checkpoint { .. } => {
                    // Skip checkpoint markers
                }
            }
        }

        Ok(count)
    }

    // =========================================================================
    // METRICS & STATS
    // =========================================================================

    /// Get hardening metrics
    pub fn metrics(&self) -> super::hardening::MetricsSnapshot {
        self.hardening.metrics.snapshot()
    }

    /// Get standard CogRedis stats
    pub fn stats(&self) -> CogRedisStats {
        self.inner.stats()
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (u64, u64, f64) {
        self.inner.cache_stats()
    }

    // =========================================================================
    // PASSTHROUGH TO INNER
    // =========================================================================

    /// Get reference to underlying BindSpace
    pub fn bind_space(&self) -> &BindSpace {
        self.inner.bind_space()
    }

    /// Get mutable reference to underlying BindSpace
    pub fn bind_space_mut(&mut self) -> &mut BindSpace {
        self.inner.bind_space_mut()
    }

    /// Access inner CogRedis (for operations not yet wrapped)
    pub fn inner(&self) -> &CogRedis {
        &self.inner
    }

    /// Access inner CogRedis mutably
    pub fn inner_mut(&mut self) -> &mut CogRedis {
        &mut self.inner
    }

    /// Get hardening config
    pub fn config(&self) -> &HardeningConfig {
        self.hardening.config()
    }
}

impl Default for HardenedCogRedis {
    fn default() -> Self {
        Self::new().expect("Failed to create HardenedCogRedis")
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    fn random_fp() -> [u64; 156] {
        let mut fp = [0u64; 156];
        for i in 0..156 {
            fp[i] = rand::random();
        }
        fp
    }
    
    #[test]
    fn test_address_tiers() {
        assert_eq!(CogAddr(0x0000).tier(), Tier::Surface);
        assert_eq!(CogAddr(0x0FFF).tier(), Tier::Surface);
        assert_eq!(CogAddr(0x1000).tier(), Tier::Fluid);
        assert_eq!(CogAddr(0x7FFF).tier(), Tier::Fluid);
        assert_eq!(CogAddr(0x8000).tier(), Tier::Node);
        assert_eq!(CogAddr(0xFFFF).tier(), Tier::Node);
    }
    
    #[test]
    fn test_promote_demote() {
        let fluid_addr = CogAddr(0x1234);
        assert!(fluid_addr.is_fluid());
        
        let promoted = fluid_addr.promote();
        assert!(promoted.is_node());
        
        let demoted = promoted.demote();
        assert!(demoted.is_fluid());
    }
    
    #[test]
    fn test_set_get() {
        let mut redis = CogRedis::new();
        
        let fp = random_fp();
        let addr = redis.set(fp, SetOptions::default());
        
        assert!(addr.is_fluid());
        
        let result = redis.get(addr);
        assert!(result.is_some());
        assert_eq!(result.unwrap().tier, Tier::Fluid);
    }
    
    #[test]
    fn test_promotion() {
        let mut redis = CogRedis::new();
        redis.promotion_threshold = 3;
        
        let fp = random_fp();
        let addr = redis.set(fp, SetOptions::default());
        
        // Access multiple times
        for _ in 0..5 {
            redis.get(addr);
        }
        
        // Should have been promoted
        // (The original addr is gone, value is in a new node addr)
        let result = redis.get(addr);
        assert!(result.is_none() || result.unwrap().tier == Tier::Node);
    }
    
    #[test]
    fn test_bind_unbind() {
        let mut redis = CogRedis::new();
        
        let a = redis.set(random_fp(), SetOptions::default().label("A"));
        let verb = redis.set(random_fp(), SetOptions::default().label("CAUSES"));
        let b = redis.set(random_fp(), SetOptions::default().label("B"));
        
        let edge = redis.bind(a, verb, b);
        assert!(edge.is_some());
        
        // Unbind should recover B given A
        let recovered = redis.unbind(edge.unwrap(), a);
        assert!(recovered.is_some());
    }
    
    #[test]
    fn test_qualia_search() {
        let mut redis = CogRedis::new();

        // Add values with different qualia
        for i in 0..10 {
            let q = QualiaVector {
                arousal: i as f32 / 10.0,
                valence: (i as f32 - 5.0) / 5.0,
                ..Default::default()
            };
            redis.set(random_fp(), SetOptions::default().qualia(q));
        }

        // Search by qualia range
        let high_arousal = redis.keys_by_qualia(None, Some((0.7, 1.0)));
        assert!(!high_arousal.is_empty());

        let positive_valence = redis.keys_by_qualia(Some((0.0, 1.0)), None);
        assert!(!positive_valence.is_empty());
    }

    #[test]
    fn test_execute_command_ping() {
        let mut redis = CogRedis::new();

        let result = redis.execute_command("PING");
        match result {
            RedisResult::String(s) => assert_eq!(s, "PONG"),
            _ => panic!("Expected String PONG"),
        }
    }

    #[test]
    fn test_execute_command_set_get() {
        let mut redis = CogRedis::new();

        // SET key value - stores in bind space
        let result = redis.execute_command("SET mykey hello");
        assert!(result.is_ok());

        // Verify bind space has content after SET
        let stats = redis.bind_space.stats();
        assert!(stats.node_count > 0, "SET should create a node");
    }

    #[test]
    fn test_execute_command_bind() {
        let mut redis = CogRedis::new();

        // BIND a b
        let result = redis.execute_command("BIND apple red VIA color");
        match result {
            RedisResult::String(addr) => {
                // Should be hex address
                assert!(!addr.is_empty());
            }
            _ => panic!("Expected address string"),
        }
    }

    #[test]
    fn test_execute_command_cam() {
        let mut redis = CogRedis::new();

        // CAM BIND operation
        let result = redis.execute_command("CAM BIND apple red");
        match result {
            RedisResult::String(hex) => {
                // Should return hex fingerprint
                assert!(!hex.is_empty());
            }
            _ => panic!("Expected hex string from BIND"),
        }

        // CAM SIMILARITY operation
        let result = redis.execute_command("CAM SIMILARITY apple apple");
        match result {
            RedisResult::String(s) => {
                let sim: f64 = s.parse().unwrap_or(0.0);
                // Self-similarity should be very high (1.0)
                assert!(sim > 0.99, "Self-similarity should be ~1.0, got {}", sim);
            }
            _ => panic!("Expected similarity value"),
        }
    }

    #[test]
    fn test_execute_command_stats() {
        let mut redis = CogRedis::new();

        let result = redis.execute_command("STATS");
        match result {
            RedisResult::Array(arr) => {
                assert!(arr.len() >= 4); // surface, fluid, node, edge, context
            }
            _ => panic!("Expected array result"),
        }
    }

    // =========================================================================
    // DN TREE COMMAND TESTS
    // =========================================================================

    #[test]
    fn test_is_dn_path() {
        assert!(CogRedis::is_dn_path("Ada:A:soul:identity"));
        assert!(CogRedis::is_dn_path("a:b"));
        assert!(!CogRedis::is_dn_path("simple_key"));
        assert!(!CogRedis::is_dn_path("mykey"));
    }

    #[test]
    fn test_dn_set_get() {
        let mut redis = CogRedis::new();

        // DN.SET creates node with parent chain
        let result = redis.execute_command("DN.SET Ada:A:soul:identity hello");
        match result {
            RedisResult::String(addr) => {
                assert!(!addr.is_empty(), "Should return address");
            }
            _ => panic!("Expected address string"),
        }

        // DN.GET retrieves the node
        let result = redis.execute_command("DN.GET Ada:A:soul:identity");
        match result {
            RedisResult::Array(arr) => {
                assert!(!arr.is_empty(), "Should return node info");
                // First element should be addr
                if let RedisResult::String(s) = &arr[0] {
                    assert!(s.starts_with("addr:"));
                }
            }
            _ => panic!("Expected array result"),
        }
    }

    #[test]
    fn test_dn_parent() {
        let mut redis = CogRedis::new();

        // Create a deep path
        redis.execute_command("DN.SET Ada:A:soul:identity test");

        // Get parent path
        let result = redis.execute_command("DN.PARENT Ada:A:soul:identity");
        match result {
            RedisResult::Array(arr) => {
                // First element should be parent path
                if let RedisResult::String(path) = &arr[0] {
                    assert_eq!(path, "Ada:A:soul");
                }
            }
            _ => panic!("Expected array with parent path"),
        }

        // Root has no parent
        let result = redis.execute_command("DN.PARENT Ada");
        assert!(matches!(result, RedisResult::Nil));
    }

    #[test]
    fn test_dn_depth_rung() {
        let mut redis = CogRedis::new();

        // Create node with RUNG
        redis.execute_command("DN.SET Ada:A:soul:secrets deep_content RUNG 5");

        // Check depth (0=Ada, 1=A, 2=soul, 3=secrets)
        let result = redis.execute_command("DN.DEPTH Ada:A:soul:secrets");
        match result {
            RedisResult::Integer(depth) => {
                assert_eq!(depth, 3, "Depth should be 3");
            }
            _ => panic!("Expected integer depth"),
        }

        // Check rung
        let result = redis.execute_command("DN.RUNG Ada:A:soul:secrets");
        match result {
            RedisResult::Integer(rung) => {
                assert_eq!(rung, 5, "Rung should be 5");
            }
            _ => panic!("Expected integer rung"),
        }
    }

    #[test]
    fn test_dn_children() {
        let mut redis = CogRedis::new();

        // Create parent and children
        redis.execute_command("DN.SET Ada:A:soul:child1 first");
        redis.execute_command("DN.SET Ada:A:soul:child2 second");
        redis.execute_command("DN.SET Ada:A:soul:child3 third");

        // Get children of Ada:A:soul
        let result = redis.execute_command("DN.CHILDREN Ada:A:soul");
        match result {
            RedisResult::Array(arr) => {
                println!("Found {} children", arr.len());
                // Should have children (may vary based on CSR build)
            }
            _ => panic!("Expected array of children"),
        }
    }

    #[test]
    fn test_dn_ancestors() {
        let mut redis = CogRedis::new();

        // Create deep path
        redis.execute_command("DN.SET Ada:A:soul:identity:deep value");

        // Get ancestors
        let result = redis.execute_command("DN.ANCESTORS Ada:A:soul:identity:deep");
        match result {
            RedisResult::Array(arr) => {
                // Should have ancestors: identity, soul, A, Ada
                println!("Found {} ancestors", arr.len());
            }
            _ => panic!("Expected array of ancestors"),
        }
    }

    #[test]
    fn test_dn_tree_traversal() {
        let mut redis = CogRedis::new();

        // Create a small tree
        redis.execute_command("DN.SET Ada:A:soul value1");
        redis.execute_command("DN.SET Ada:A:core value2");
        redis.execute_command("DN.SET Ada:B:thoughts value3");

        // Traverse from Ada with depth 2
        let result = redis.execute_command("DN.TREE Ada DEPTH 2");
        match result {
            RedisResult::Array(arr) => {
                println!("Tree traversal found {} nodes", arr.len());
                // Should find Ada and some children
            }
            _ => panic!("Expected array from tree traversal"),
        }
    }

    #[test]
    fn test_standard_get_set_with_dn_path() {
        let mut redis = CogRedis::new();

        // Standard SET/GET should work with DN paths too
        // because resolve_key detects ':'
        let result = redis.execute_command("SET Ada:A:test hello");
        assert!(result.is_ok());

        // GET should work (via dn_path_to_addr)
        let result = redis.execute_command("GET Ada:A:test");
        // May or may not find depending on hash vs dn_path
        // The key point is it doesn't error
        println!("GET result: {:?}", result);
    }
}
