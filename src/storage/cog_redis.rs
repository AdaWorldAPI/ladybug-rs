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
use super::bind_space::{BindSpace, BindNode, Addr, FINGERPRINT_WORDS};

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
    /// - Hash key to get deterministic address
    /// - Check if exists in bind space
    pub fn resolve_key(&self, key: &str) -> Option<Addr> {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let hash = hasher.finish();

        // Map to node address space (0x80-0xFF:XX)
        let prefix = 0x80 + ((hash >> 8) as u8 & 0x7F);  // 0x80-0xFF
        let slot = (hash & 0xFF) as u8;
        let addr = Addr::new(prefix, slot);

        // Check if occupied
        if self.bind_space.read(addr).is_some() {
            Some(addr)
        } else {
            None
        }
    }

    /// Allocate address for key in bind space
    ///
    /// Returns address where key's value should be stored.
    pub fn resolve_or_allocate(&mut self, key: &str, fingerprint: [u64; FINGERPRINT_WORDS]) -> Addr {
        // Check if already exists
        if let Some(addr) = self.resolve_key(key) {
            return addr;
        }

        // Allocate new address in node space
        let addr = self.bind_space.write_labeled(fingerprint, key);
        addr
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
}
