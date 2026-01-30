//! Universal Bind Space - The DTO That All Languages Hit
//!
//! # The Insight
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                              16-bit ADDRESS SPACE                           │
//! ├─────────────────┬───────────────────────────────────────────────────────────┤
//! │  0x0000-0x0FFF  │  SURFACE (4K) - VERBS + OPS                               │
//! │                 │                 The language primitives                   │
//! ├─────────────────┼───────────────────────────────────────────────────────────┤
//! │  0x1000-0x7FFF  │  FLUID (28K) - CONTEXT SELECTOR + EDGES                   │
//! │                 │                 Defines WHAT 0x8000-0xFFFF means:         │
//! │                 │                   • Chunk 0: Concept space                │
//! │                 │                   • Chunk 1: Memory space                 │
//! │                 │                   • Chunk 2: Codebook space               │
//! │                 │                   • Chunk 3: Meta-awareness               │
//! │                 │                 EDGES live here too                       │
//! ├─────────────────┼───────────────────────────────────────────────────────────┤
//! │  0x8000-0xFFFF  │  NODES (32K) - THE UNIVERSAL BIND SPACE                   │
//! │                 │                 What it IS depends on fluid context       │
//! │                 │                                                           │
//! │                 │                 CogRedis?  → Reads/writes here            │
//! │                 │                 Cypher?    → Queries here                 │
//! │                 │                 Neo4j?     → Traverses here               │
//! │                 │                 GraphQL?   → Resolves here                │
//! │                 │                 SQL?       → Selects here                 │
//! │                 │                                                           │
//! │                 │                 ONE SPACE. ANY LANGUAGE.                  │
//! └─────────────────┴───────────────────────────────────────────────────────────┘
//! ```
//!
//! # Architecture
//!
//! The fluid zone (0x1000-0x7FFF) acts as a CONTEXT SELECTOR that determines
//! how the node space (0x8000-0xFFFF) is interpreted:
//!
//! - Setting fluid context to Chunk 0 → node space = concepts
//! - Setting fluid context to Chunk 1 → node space = memories  
//! - Setting fluid context to Chunk 2 → node space = codebook entries
//! - Setting fluid context to Chunk 3 → node space = meta-awareness states
//!
//! The node space itself is the UNIVERSAL DTO - all query languages bind here.
//! The fingerprint at any address doesn't care what language asked for it.
//!
//! # Query Language Adapters
//!
//! All adapters implement the same trait and hit the same bind space:
//!
//! ```text
//!                     ┌─────────────┐
//!                     │ Bind Space  │
//!                     │ 0x8000-FFFF │
//!                     └──────┬──────┘
//!                            │
//!        ┌───────────────────┼───────────────────┐
//!        │                   │                   │
//!   ┌────┴────┐        ┌────┴────┐        ┌────┴────┐
//!   │  Redis  │        │  Cypher │        │   SQL   │
//!   │   GET   │        │  MATCH  │        │ SELECT  │
//!   └─────────┘        └─────────┘        └─────────┘
//! ```

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

// =============================================================================
// ADDRESS CONSTANTS
// =============================================================================

/// Surface tier: verbs + ops (fixed vocabulary)
pub const SURFACE_START: u16 = 0x0000;
pub const SURFACE_END: u16 = 0x0FFF;
pub const SURFACE_SIZE: usize = 4096;

/// Fluid tier: context selector + edges
pub const FLUID_START: u16 = 0x1000;
pub const FLUID_END: u16 = 0x7FFF;
pub const FLUID_SIZE: usize = 28672;

/// Node tier: universal bind space
pub const NODE_START: u16 = 0x8000;
pub const NODE_END: u16 = 0xFFFF;
pub const NODE_SIZE: usize = 32768;

/// Total addressable space
pub const TOTAL_SIZE: usize = 65536;

/// Words in a 10K-bit fingerprint
pub const FINGERPRINT_WORDS: usize = 156;

// =============================================================================
// CHUNK CONTEXTS (What the node space means)
// =============================================================================

/// Chunk context - defines interpretation of node space
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum ChunkContext {
    /// Concept space - abstract types and categories
    Concepts = 0,
    /// Memory space - episodic memories, experiences
    Memories = 1,
    /// Codebook space - learned patterns, templates
    Codebook = 2,
    /// Meta-awareness - self-model, introspection states
    MetaAwareness = 3,
    /// Extended node space - overflow when >32K nodes needed
    Extended(u16) = 4,
}

impl ChunkContext {
    /// Get the fluid address that activates this context
    pub fn selector_addr(&self) -> u16 {
        match self {
            ChunkContext::Concepts => FLUID_START,
            ChunkContext::Memories => FLUID_START + 1,
            ChunkContext::Codebook => FLUID_START + 2,
            ChunkContext::MetaAwareness => FLUID_START + 3,
            ChunkContext::Extended(n) => FLUID_START + 4 + n,
        }
    }
    
    /// Parse from fluid address
    pub fn from_selector(addr: u16) -> Option<Self> {
        if addr < FLUID_START || addr > FLUID_END {
            return None;
        }
        let offset = addr - FLUID_START;
        match offset {
            0 => Some(ChunkContext::Concepts),
            1 => Some(ChunkContext::Memories),
            2 => Some(ChunkContext::Codebook),
            3 => Some(ChunkContext::MetaAwareness),
            n => Some(ChunkContext::Extended(n - 4)),
        }
    }
}

// =============================================================================
// BIND NODE - The universal content container
// =============================================================================

/// A node in the universal bind space
/// 
/// This is what ALL query languages ultimately read/write.
/// The fingerprint doesn't care what syntax asked for it.
#[derive(Debug, Clone)]
pub struct BindNode {
    /// 10K-bit fingerprint (the content-addressable identity)
    pub fingerprint: [u64; FINGERPRINT_WORDS],
    /// Human-readable label
    pub label: Option<String>,
    /// Qualia index (0-255, emergent)
    pub qidx: u8,
    /// Creation timestamp
    pub created: Instant,
    /// Last access timestamp  
    pub accessed: Instant,
    /// Access count (for promotion/cache decisions)
    pub access_count: u32,
    /// Optional payload (serialized content)
    pub payload: Option<Vec<u8>>,
}

impl BindNode {
    pub fn new(fingerprint: [u64; FINGERPRINT_WORDS]) -> Self {
        let now = Instant::now();
        Self {
            fingerprint,
            label: None,
            qidx: 0,
            created: now,
            accessed: now,
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
    
    pub fn with_payload(mut self, payload: Vec<u8>) -> Self {
        self.payload = Some(payload);
        self
    }
    
    /// Touch - update access time and count
    pub fn touch(&mut self) {
        self.accessed = Instant::now();
        self.access_count += 1;
    }
}

// =============================================================================
// BIND EDGE - Connection in the fluid zone
// =============================================================================

/// An edge connecting nodes via a verb
/// 
/// Edges live in the fluid zone (0x1000-0x7FFF).
/// They connect nodes in the bind space using verbs from the surface.
#[derive(Debug, Clone)]
pub struct BindEdge {
    /// Source node address (in node space 0x8000-0xFFFF)
    pub from: u16,
    /// Target node address (in node space 0x8000-0xFFFF)
    pub to: u16,
    /// Verb address (in surface space 0x0000-0x0FFF)
    pub verb: u16,
    /// Edge fingerprint: from ⊗ verb ⊗ to (for ABBA retrieval)
    pub fingerprint: [u64; FINGERPRINT_WORDS],
    /// Edge weight/strength
    pub weight: f32,
    /// Creation timestamp
    pub created: Instant,
}

impl BindEdge {
    pub fn new(from: u16, verb: u16, to: u16) -> Self {
        Self {
            from,
            to,
            verb,
            fingerprint: [0u64; FINGERPRINT_WORDS], // Set by bind operation
            weight: 1.0,
            created: Instant::now(),
        }
    }
    
    /// Compute edge fingerprint via XOR binding
    pub fn bind(&mut self, from_fp: &[u64; FINGERPRINT_WORDS], verb_fp: &[u64; FINGERPRINT_WORDS], to_fp: &[u64; FINGERPRINT_WORDS]) {
        for i in 0..FINGERPRINT_WORDS {
            self.fingerprint[i] = from_fp[i] ^ verb_fp[i] ^ to_fp[i];
        }
    }
    
    /// ABBA unbind: given edge and one known, recover the other
    pub fn unbind(&self, known: &[u64; FINGERPRINT_WORDS], verb_fp: &[u64; FINGERPRINT_WORDS]) -> [u64; FINGERPRINT_WORDS] {
        let mut result = [0u64; FINGERPRINT_WORDS];
        for i in 0..FINGERPRINT_WORDS {
            result[i] = self.fingerprint[i] ^ known[i] ^ verb_fp[i];
        }
        result
    }
}

// =============================================================================
// BIND SPACE - The Universal DTO
// =============================================================================

/// The Universal Bind Space
/// 
/// This is the single source of truth that all query languages hit.
/// Whether you speak Redis, Cypher, SQL, or GraphQL - you end up here.
pub struct BindSpace {
    /// Surface tier: verbs and ops (fixed)
    surface: HashMap<u16, BindNode>,
    
    /// Fluid tier: edges and context selectors
    fluid: HashMap<u16, BindNode>,
    
    /// Fluid edges (separate for efficient traversal)
    edges: Vec<BindEdge>,
    
    /// Edge index: from_addr -> edge indices (CSR-style)
    edge_index_out: HashMap<u16, Vec<usize>>,
    
    /// Edge index: to_addr -> edge indices (reverse CSR)
    edge_index_in: HashMap<u16, Vec<usize>>,
    
    /// Node tier: the actual bind space
    nodes: HashMap<u16, BindNode>,
    
    /// Current chunk context
    context: ChunkContext,
    
    /// Next available addresses
    next_fluid: u16,
    next_node: u16,
}

impl BindSpace {
    pub fn new() -> Self {
        let mut space = Self {
            surface: HashMap::new(),
            fluid: HashMap::new(),
            edges: Vec::new(),
            edge_index_out: HashMap::new(),
            edge_index_in: HashMap::new(),
            nodes: HashMap::new(),
            context: ChunkContext::Concepts,
            next_fluid: FLUID_START + 256,  // Reserve first 256 for contexts
            next_node: NODE_START,
        };
        space.init_surface();
        space
    }
    
    /// Initialize surface with core verbs
    fn init_surface(&mut self) {
        // Core verbs (from the 144 Go board intersections)
        let verbs = [
            (0x0060, "CAUSES"),
            (0x0061, "BECOMES"),
            (0x0062, "ENABLES"),
            (0x0063, "PREVENTS"),
            (0x0064, "REQUIRES"),
            (0x0065, "IMPLIES"),
            (0x0066, "CONTAINS"),
            (0x0067, "ACTIVATES"),
            (0x0068, "INHIBITS"),
            (0x0069, "TRANSFORMS"),
            (0x006A, "RESONATES"),
            (0x006B, "AMPLIFIES"),
            (0x006C, "DAMPENS"),
            (0x006D, "OBSERVES"),
            (0x006E, "REMEMBERS"),
            (0x006F, "FORGETS"),
            // Flow verbs
            (0x0070, "SHIFT"),
            (0x0071, "LEAP"),
            (0x0072, "EMERGE"),
            (0x0073, "SUBSIDE"),
            (0x0074, "OSCILLATE"),
            (0x0075, "CRYSTALLIZE"),
            (0x0076, "DISSOLVE"),
            (0x0077, "TRANSFORM"),
        ];
        
        for (addr, label) in verbs {
            let mut node = BindNode::new(verb_fingerprint(label));
            node.label = Some(label.to_string());
            self.surface.insert(addr, node);
        }
    }
    
    // =========================================================================
    // CONTEXT OPERATIONS
    // =========================================================================
    
    /// Set the current chunk context
    /// This changes what the node space (0x8000-0xFFFF) means
    pub fn set_context(&mut self, context: ChunkContext) {
        self.context = context;
    }
    
    /// Get current context
    pub fn context(&self) -> ChunkContext {
        self.context
    }
    
    // =========================================================================
    // UNIVERSAL READ/WRITE (All languages hit these)
    // =========================================================================
    
    /// Read from any address
    /// 
    /// This is what GET (Redis), MATCH (Cypher), SELECT (SQL) all become.
    pub fn read(&mut self, addr: u16) -> Option<&BindNode> {
        let node = match addr {
            a if a <= SURFACE_END => self.surface.get(&a),
            a if a <= FLUID_END => self.fluid.get(&a),
            a => self.nodes.get(&a),
        };
        
        // Touch for access tracking (need mut for this)
        if let Some(_) = node {
            // We'd need interior mutability for proper touch
            // For now, just return the reference
        }
        
        node
    }
    
    /// Read mutable from any address
    pub fn read_mut(&mut self, addr: u16) -> Option<&mut BindNode> {
        let node = match addr {
            a if a <= SURFACE_END => self.surface.get_mut(&a),
            a if a <= FLUID_END => self.fluid.get_mut(&a),
            a => self.nodes.get_mut(&a),
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
    /// This is what SET (Redis), CREATE (Cypher), INSERT (SQL) all become.
    pub fn write(&mut self, fingerprint: [u64; FINGERPRINT_WORDS]) -> u16 {
        let addr = self.next_node;
        self.next_node = self.next_node.wrapping_add(1);
        if self.next_node < NODE_START {
            self.next_node = NODE_START;  // Wrap within node space
        }
        
        let node = BindNode::new(fingerprint);
        self.nodes.insert(addr, node);
        addr
    }
    
    /// Write with label
    pub fn write_labeled(&mut self, fingerprint: [u64; FINGERPRINT_WORDS], label: &str) -> u16 {
        let addr = self.write(fingerprint);
        if let Some(node) = self.nodes.get_mut(&addr) {
            node.label = Some(label.to_string());
        }
        addr
    }
    
    /// Delete from any address
    pub fn delete(&mut self, addr: u16) -> Option<BindNode> {
        match addr {
            a if a <= SURFACE_END => None,  // Can't delete surface
            a if a <= FLUID_END => self.fluid.remove(&a),
            a => self.nodes.remove(&a),
        }
    }
    
    // =========================================================================
    // EDGE OPERATIONS (Fluid zone)
    // =========================================================================
    
    /// Create an edge
    /// 
    /// This is what relationships in Cypher, foreign keys in SQL become.
    pub fn link(&mut self, from: u16, verb: u16, to: u16) -> usize {
        let mut edge = BindEdge::new(from, verb, to);
        
        // Get fingerprints and bind
        if let (Some(from_node), Some(verb_node), Some(to_node)) = 
            (self.nodes.get(&from), self.surface.get(&verb), self.nodes.get(&to)) 
        {
            edge.bind(&from_node.fingerprint, &verb_node.fingerprint, &to_node.fingerprint);
        }
        
        let idx = self.edges.len();
        
        // Update indices (CSR-style)
        self.edge_index_out.entry(from).or_default().push(idx);
        self.edge_index_in.entry(to).or_default().push(idx);
        
        self.edges.push(edge);
        idx
    }
    
    /// Get outgoing edges (CSR-style O(1) index lookup)
    pub fn edges_out(&self, from: u16) -> Vec<&BindEdge> {
        self.edge_index_out
            .get(&from)
            .map(|indices| indices.iter().filter_map(|&i| self.edges.get(i)).collect())
            .unwrap_or_default()
    }
    
    /// Get incoming edges (reverse CSR)
    pub fn edges_in(&self, to: u16) -> Vec<&BindEdge> {
        self.edge_index_in
            .get(&to)
            .map(|indices| indices.iter().filter_map(|&i| self.edges.get(i)).collect())
            .unwrap_or_default()
    }
    
    /// Get edges by verb
    pub fn edges_via(&self, verb: u16) -> Vec<&BindEdge> {
        self.edges.iter().filter(|e| e.verb == verb).collect()
    }
    
    /// Traverse: from -> via verb -> targets
    pub fn traverse(&self, from: u16, verb: u16) -> Vec<u16> {
        self.edges_out(from)
            .into_iter()
            .filter(|e| e.verb == verb)
            .map(|e| e.to)
            .collect()
    }
    
    /// Reverse traverse: targets <- via verb <- to
    pub fn traverse_reverse(&self, to: u16, verb: u16) -> Vec<u16> {
        self.edges_in(to)
            .into_iter()
            .filter(|e| e.verb == verb)
            .map(|e| e.from)
            .collect()
    }
    
    // =========================================================================
    // N-HOP TRAVERSAL (What Kuzu CSR does)
    // =========================================================================
    
    /// N-hop traversal from a node via a verb
    /// 
    /// This is the core graph operation that Kuzu CSR accelerates.
    /// We use edge indices for O(1) neighbor lookup per hop.
    pub fn traverse_n_hops(&self, start: u16, verb: u16, max_hops: usize) -> Vec<(usize, u16)> {
        let mut results = Vec::new();
        let mut frontier = vec![start];
        let mut visited = std::collections::HashSet::new();
        visited.insert(start);
        
        for hop in 1..=max_hops {
            let mut next_frontier = Vec::new();
            
            for &node in &frontier {
                for target in self.traverse(node, verb) {
                    if visited.insert(target) {
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
    // STATISTICS
    // =========================================================================
    
    pub fn stats(&self) -> BindSpaceStats {
        BindSpaceStats {
            surface_count: self.surface.len(),
            fluid_count: self.fluid.len(),
            node_count: self.nodes.len(),
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
// HELPER FUNCTIONS
// =============================================================================

/// Generate a deterministic fingerprint for a verb label
fn verb_fingerprint(label: &str) -> [u64; FINGERPRINT_WORDS] {
    let mut fp = [0u64; FINGERPRINT_WORDS];
    let bytes = label.as_bytes();
    
    // Simple hash spread across fingerprint
    for (i, &b) in bytes.iter().enumerate() {
        let word_idx = i % FINGERPRINT_WORDS;
        let bit_idx = (b as usize * 7 + i * 13) % 64;
        fp[word_idx] |= 1u64 << bit_idx;
    }
    
    // Spread more bits for density
    for i in 0..FINGERPRINT_WORDS {
        let seed = fp[i];
        fp[(i + 1) % FINGERPRINT_WORDS] ^= seed.rotate_left(17);
        fp[(i + 3) % FINGERPRINT_WORDS] ^= seed.rotate_right(23);
    }
    
    fp
}

/// Compute Hamming distance between fingerprints
pub fn hamming_distance(a: &[u64; FINGERPRINT_WORDS], b: &[u64; FINGERPRINT_WORDS]) -> u32 {
    let mut dist = 0u32;
    for i in 0..FINGERPRINT_WORDS {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}

// =============================================================================
// QUERY LANGUAGE TRAIT (What all adapters implement)
// =============================================================================

/// Trait for query language adapters
/// 
/// All languages (Redis, Cypher, SQL, GraphQL) implement this
/// and ultimately call BindSpace methods.
pub trait QueryAdapter {
    /// Execute a query and return results
    fn execute(&self, space: &mut BindSpace, query: &str) -> QueryResult;
}

/// Generic query result
#[derive(Debug)]
pub struct QueryResult {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<QueryValue>>,
    pub affected: usize,
}

impl QueryResult {
    pub fn empty() -> Self {
        Self {
            columns: Vec::new(),
            rows: Vec::new(),
            affected: 0,
        }
    }
    
    pub fn single(addr: u16) -> Self {
        Self {
            columns: vec!["addr".to_string()],
            rows: vec![vec![QueryValue::Addr(addr)]],
            affected: 1,
        }
    }
}

/// Query value types
#[derive(Debug, Clone)]
pub enum QueryValue {
    Addr(u16),
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
    fn test_address_ranges() {
        assert_eq!(SURFACE_SIZE, 4096);
        assert_eq!(FLUID_SIZE, 28672);
        assert_eq!(NODE_SIZE, 32768);
        assert_eq!(SURFACE_SIZE + FLUID_SIZE + NODE_SIZE, TOTAL_SIZE);
    }
    
    #[test]
    fn test_bind_space_creation() {
        let space = BindSpace::new();
        assert!(space.surface.len() > 0);  // Verbs initialized
        assert_eq!(space.context, ChunkContext::Concepts);
    }
    
    #[test]
    fn test_write_read() {
        let mut space = BindSpace::new();
        let fp = [42u64; FINGERPRINT_WORDS];
        
        let addr = space.write(fp);
        assert!(addr >= NODE_START);
        
        let node = space.read(addr);
        assert!(node.is_some());
        assert_eq!(node.unwrap().fingerprint, fp);
    }
    
    #[test]
    fn test_link_traverse() {
        let mut space = BindSpace::new();
        
        // Create two nodes
        let a = space.write_labeled([1u64; FINGERPRINT_WORDS], "Node A");
        let b = space.write_labeled([2u64; FINGERPRINT_WORDS], "Node B");
        
        // Link them with CAUSES
        let verb = 0x0060;  // CAUSES
        space.link(a, verb, b);
        
        // Traverse
        let targets = space.traverse(a, verb);
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0], b);
    }
    
    #[test]
    fn test_n_hop_traversal() {
        let mut space = BindSpace::new();
        
        // Create chain: A -> B -> C -> D
        let a = space.write([1u64; FINGERPRINT_WORDS]);
        let b = space.write([2u64; FINGERPRINT_WORDS]);
        let c = space.write([3u64; FINGERPRINT_WORDS]);
        let d = space.write([4u64; FINGERPRINT_WORDS]);
        
        let verb = 0x0060;  // CAUSES
        space.link(a, verb, b);
        space.link(b, verb, c);
        space.link(c, verb, d);
        
        // 3-hop traversal from A
        let results = space.traverse_n_hops(a, verb, 3);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0], (1, b));  // 1 hop to B
        assert_eq!(results[1], (2, c));  // 2 hops to C
        assert_eq!(results[2], (3, d));  // 3 hops to D
    }
    
    #[test]
    fn test_context_switching() {
        let mut space = BindSpace::new();
        
        assert_eq!(space.context(), ChunkContext::Concepts);
        
        space.set_context(ChunkContext::Memories);
        assert_eq!(space.context(), ChunkContext::Memories);
        
        space.set_context(ChunkContext::MetaAwareness);
        assert_eq!(space.context(), ChunkContext::MetaAwareness);
    }
    
    #[test]
    fn test_edge_indices() {
        let mut space = BindSpace::new();
        
        let a = space.write([1u64; FINGERPRINT_WORDS]);
        let b = space.write([2u64; FINGERPRINT_WORDS]);
        let c = space.write([3u64; FINGERPRINT_WORDS]);
        
        let causes = 0x0060;
        let enables = 0x0062;
        
        space.link(a, causes, b);
        space.link(a, enables, c);
        space.link(b, causes, c);
        
        // A has 2 outgoing edges
        assert_eq!(space.edges_out(a).len(), 2);
        
        // C has 2 incoming edges
        assert_eq!(space.edges_in(c).len(), 2);
        
        // Only 2 CAUSES edges total
        assert_eq!(space.edges_via(causes).len(), 2);
    }
}
