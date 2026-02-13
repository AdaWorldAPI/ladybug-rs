//! Substrate - Unified Interface DTO
//!
//! The Substrate bridges all storage layers into a single coherent interface.
//! BindSpace is the hot cache, Lance (when connected) is persistent storage.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                           SUBSTRATE                                         │
//! │                    (Unified Interface DTO)                                  │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │                                                                             │
//! │   Hot Layer (BindSpace) - SINGLE SOURCE OF TRUTH FOR HOT DATA              │
//! │   ├── Surface (0x00-0x0F): 4,096 CAM operations                            │
//! │   ├── Fluid (0x10-0x7F): 28,672 working memory with TTL                    │
//! │   └── Nodes (0x80-0xFF): 32,768 persistent node cache                      │
//! │                                                                             │
//! │   Lifecycle:                                                                │
//! │   ├── write_fluid() → allocates in 0x10-0x7F with TTL                      │
//! │   ├── tick() → expires fluid, promotes hot, demotes cold                   │
//! │   ├── crystallize() → promotes fluid entry to node space                   │
//! │   └── evaporate() → demotes node entry to fluid space                      │
//! │                                                                             │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, RwLock};
use std::time::{Duration, Instant};

use super::bind_space::{Addr, BindNode, BindSpace, FINGERPRINT_WORDS, hamming_distance};
use super::cog_redis::CogAddr;
use crate::search::RubiconSearch;

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Substrate configuration
#[derive(Debug, Clone)]
pub struct SubstrateConfig {
    /// Path to Lance database (for future use)
    pub lance_path: PathBuf,
    /// Maximum hot nodes in BindSpace
    pub max_hot_nodes: usize,
    /// TTL for fluid zone entries
    pub fluid_ttl: Duration,
    /// Promotion threshold (access count before promoting to node)
    pub promotion_threshold: u32,
    /// Demotion threshold (time since last access before demoting)
    pub demotion_threshold: Duration,
    /// Write buffer size before auto-commit
    pub write_buffer_size: usize,
    /// Enable async commit
    pub async_commit: bool,
    /// Search cascade levels
    pub cascade_levels: usize,
}

impl Default for SubstrateConfig {
    fn default() -> Self {
        Self {
            lance_path: PathBuf::from("./data/lance"),
            max_hot_nodes: 32768,
            fluid_ttl: Duration::from_secs(300),
            promotion_threshold: 10,
            demotion_threshold: Duration::from_secs(3600),
            write_buffer_size: 1000,
            async_commit: true,
            cascade_levels: 4,
        }
    }
}

// =============================================================================
// SUBSTRATE NODE (Unified representation)
// =============================================================================

/// A node in the substrate (unified view)
#[derive(Clone, Debug)]
pub struct SubstrateNode {
    /// 16-bit cognitive address
    pub addr: CogAddr,
    /// 16K-bit fingerprint (256 × 64-bit words)
    pub fingerprint: [u64; FINGERPRINT_WORDS],
    /// Optional label
    pub label: Option<String>,
    /// Qualia index (0-255)
    pub qidx: u8,
    /// Access count
    pub access_count: u32,
    /// Last access time
    pub last_access: Instant,
    /// Creation time
    pub created: Instant,
    /// TTL (None = permanent in node space)
    pub ttl: Option<Duration>,
    /// Lance row ID (if persisted)
    pub lance_id: Option<String>,
    /// Version for MVCC
    pub version: u64,
    /// Is this node dirty (needs Lance sync)?
    pub dirty: bool,
}

impl SubstrateNode {
    /// Create a new node with fingerprint
    pub fn new(addr: CogAddr, fingerprint: [u64; FINGERPRINT_WORDS]) -> Self {
        Self {
            addr,
            fingerprint,
            label: None,
            qidx: 128,
            access_count: 0,
            last_access: Instant::now(),
            created: Instant::now(),
            ttl: None,
            lance_id: None,
            version: 1,
            dirty: true,
        }
    }

    /// Create with label
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set qualia index
    pub fn with_qidx(mut self, qidx: u8) -> Self {
        self.qidx = qidx;
        self
    }

    /// Set TTL for fluid zone
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }

    /// Popcount of fingerprint
    pub fn popcount(&self) -> u32 {
        self.fingerprint.iter().map(|w| w.count_ones()).sum()
    }

    /// Hamming distance to another node
    pub fn distance_to(&self, other: &SubstrateNode) -> u32 {
        hamming_distance(&self.fingerprint, &other.fingerprint)
    }

    /// Similarity to another node (0.0-1.0)
    pub fn similarity_to(&self, other: &SubstrateNode) -> f32 {
        let dist = self.distance_to(other);
        1.0 - (dist as f32 / crate::FINGERPRINT_BITS as f32)
    }
}

impl From<&BindNode> for SubstrateNode {
    fn from(node: &BindNode) -> Self {
        Self {
            addr: CogAddr::new(0),
            fingerprint: node.fingerprint,
            label: node.label.clone(),
            qidx: node.qidx,
            access_count: node.access_count,
            last_access: Instant::now(),
            created: Instant::now(),
            ttl: None,
            lance_id: None,
            version: 1,
            dirty: false,
        }
    }
}

// =============================================================================
// SUBSTRATE EDGE (Unified edge representation)
// =============================================================================

/// An edge in the substrate
#[derive(Clone, Debug)]
pub struct SubstrateEdge {
    /// Source node address
    pub from: CogAddr,
    /// Target node address
    pub to: CogAddr,
    /// Verb/relation address (in surface tier)
    pub verb: CogAddr,
    /// Bound fingerprint: from ⊗ verb ⊗ to
    pub fingerprint: [u64; FINGERPRINT_WORDS],
    /// Edge weight
    pub weight: f32,
    /// Creation time
    pub created: Instant,
    /// Lance ID (if persisted)
    pub lance_id: Option<String>,
    /// Is dirty?
    pub dirty: bool,
}

impl SubstrateEdge {
    /// Create a new edge
    pub fn new(from: CogAddr, verb: CogAddr, to: CogAddr) -> Self {
        Self {
            from,
            to,
            verb,
            fingerprint: [0u64; FINGERPRINT_WORDS],
            weight: 1.0,
            created: Instant::now(),
            lance_id: None,
            dirty: true,
        }
    }

    /// Bind fingerprints
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

    /// Unbind to recover unknown from known
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
// WRITE BUFFER (Pending writes to Lance)
// =============================================================================

/// A pending write operation
#[derive(Clone, Debug)]
pub enum WriteOp {
    /// Insert/update a node
    UpsertNode(SubstrateNode),
    /// Delete a node
    DeleteNode(CogAddr),
    /// Insert/update an edge
    UpsertEdge(SubstrateEdge),
    /// Delete an edge
    DeleteEdge {
        from: CogAddr,
        verb: CogAddr,
        to: CogAddr,
    },
}

/// Write buffer for batching Lance operations
#[derive(Debug)]
pub struct WriteBuffer {
    /// Pending operations
    ops: Vec<WriteOp>,
    /// Maximum size before auto-flush
    max_size: usize,
}

impl WriteBuffer {
    pub fn new(max_size: usize) -> Self {
        Self {
            ops: Vec::with_capacity(max_size),
            max_size,
        }
    }

    /// Add an operation, returns true if buffer is full
    pub fn push(&mut self, op: WriteOp) -> bool {
        self.ops.push(op);
        self.ops.len() >= self.max_size
    }

    /// Take all operations
    pub fn drain(&mut self) -> Vec<WriteOp> {
        std::mem::take(&mut self.ops)
    }

    /// Number of pending operations
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Is buffer empty?
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

// =============================================================================
// SUBSTRATE STATISTICS
// =============================================================================

/// Substrate statistics
#[derive(Debug, Default)]
pub struct SubstrateStats {
    /// Hot cache hits
    pub hot_hits: AtomicU64,
    /// Hot cache misses (fell through to Lance)
    pub hot_misses: AtomicU64,
    /// Nodes in hot cache
    pub hot_nodes: AtomicU64,
    /// Edges in hot cache
    pub hot_edges: AtomicU64,
    /// Pending writes
    pub pending_writes: AtomicU64,
    /// Lance reads
    pub lance_reads: AtomicU64,
    /// Lance writes
    pub lance_writes: AtomicU64,
    /// Promotions (fluid → node)
    pub promotions: AtomicU64,
    /// Demotions (node → fluid)
    pub demotions: AtomicU64,
    /// Evictions (expired from fluid)
    pub evictions: AtomicU64,
}

impl SubstrateStats {
    /// Hit ratio
    pub fn hit_ratio(&self) -> f64 {
        let hits = self.hot_hits.load(Ordering::Relaxed) as f64;
        let misses = self.hot_misses.load(Ordering::Relaxed) as f64;
        if hits + misses > 0.0 {
            hits / (hits + misses)
        } else {
            0.0
        }
    }
}

// =============================================================================
// FLUID TRACKER (TTL management for fluid zone)
// =============================================================================

/// Tracks TTL for fluid zone entries
struct FluidTracker {
    /// Map from address to (creation_time, ttl)
    entries: std::collections::HashMap<u16, (Instant, Duration)>,
}

impl FluidTracker {
    fn new() -> Self {
        Self {
            entries: std::collections::HashMap::new(),
        }
    }

    fn insert(&mut self, addr: u16, ttl: Duration) {
        self.entries.insert(addr, (Instant::now(), ttl));
    }

    fn remove(&mut self, addr: u16) {
        self.entries.remove(&addr);
    }

    fn is_expired(&self, addr: u16) -> bool {
        if let Some((created, ttl)) = self.entries.get(&addr) {
            created.elapsed() > *ttl
        } else {
            false
        }
    }

    fn expired_entries(&self) -> Vec<u16> {
        self.entries
            .iter()
            .filter(|(_, (created, ttl))| created.elapsed() > *ttl)
            .map(|(addr, _)| *addr)
            .collect()
    }
}

// =============================================================================
// SUBSTRATE (The unified interface)
// =============================================================================

/// The Substrate - unified interface to all storage layers
pub struct Substrate {
    /// Configuration
    config: SubstrateConfig,

    /// Hot cache (BindSpace) - SINGLE source of hot data
    bind_space: RwLock<BindSpace>,

    /// Rubicon search with adaptive thresholds (Belichtungsmesser)
    rubicon_search: RwLock<RubiconSearch>,

    /// Address mapping: search index → CogAddr
    search_addrs: RwLock<Vec<CogAddr>>,

    /// Fluid zone TTL tracker
    fluid_tracker: RwLock<FluidTracker>,

    /// Write buffer (pending Lance writes)
    write_buffer: Mutex<WriteBuffer>,

    /// Next fluid zone slot (for allocation)
    next_fluid_slot: AtomicU64,

    /// Version counter
    version: AtomicU64,

    /// Statistics
    stats: SubstrateStats,

    /// Is Lance connected?
    lance_connected: AtomicBool,
}

impl Substrate {
    /// Create a new substrate
    pub fn new(config: SubstrateConfig) -> Self {
        let write_buffer = WriteBuffer::new(config.write_buffer_size);
        // Pre-allocate Rubicon search with Belichtungsmesser
        let rubicon_search = RubiconSearch::with_capacity(config.max_hot_nodes);
        let search_addrs = Vec::with_capacity(config.max_hot_nodes);

        Self {
            config,
            bind_space: RwLock::new(BindSpace::new()),
            rubicon_search: RwLock::new(rubicon_search),
            search_addrs: RwLock::new(search_addrs),
            fluid_tracker: RwLock::new(FluidTracker::new()),
            write_buffer: Mutex::new(write_buffer),
            next_fluid_slot: AtomicU64::new(0x1000), // Start at 0x10:00
            version: AtomicU64::new(1),
            stats: SubstrateStats::default(),
            lance_connected: AtomicBool::new(false),
        }
    }

    /// Create with default config
    pub fn default_new() -> Self {
        Self::new(SubstrateConfig::default())
    }

    // =========================================================================
    // READ OPERATIONS
    // =========================================================================

    /// Read a node by address
    pub fn read(&self, addr: CogAddr) -> Option<SubstrateNode> {
        let bind_space = self.bind_space.read().unwrap();
        let bind_addr = Addr::from(addr.0);

        if let Some(node) = bind_space.read(bind_addr) {
            self.stats.hot_hits.fetch_add(1, Ordering::Relaxed);
            let mut result = SubstrateNode::from(node);
            result.addr = addr;

            // Check if this is a fluid zone entry with TTL
            if addr.is_fluid() {
                let tracker = self.fluid_tracker.read().unwrap();
                if tracker.is_expired(addr.0) {
                    // Expired - return None (will be cleaned up on tick)
                    return None;
                }
            }

            return Some(result);
        }

        self.stats.hot_misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Read a node by label
    pub fn read_by_label(&self, label: &str) -> Option<SubstrateNode> {
        let bind_space = self.bind_space.read().unwrap();

        // Check surfaces first
        for prefix in 0..16u8 {
            for slot in 0..=255u8 {
                let addr = Addr::new(prefix, slot);
                if let Some(node) = bind_space.read(addr) {
                    if node.label.as_deref() == Some(label) {
                        self.stats.hot_hits.fetch_add(1, Ordering::Relaxed);
                        let mut result = SubstrateNode::from(node);
                        result.addr = CogAddr::from_parts(prefix, slot);
                        return Some(result);
                    }
                }
            }
        }

        // Check fluid zone (0x10-0x7F)
        for prefix in 0x10..0x80u8 {
            for slot in 0..=255u8 {
                let addr = Addr::new(prefix, slot);
                if let Some(node) = bind_space.read(addr) {
                    if node.label.as_deref() == Some(label) {
                        self.stats.hot_hits.fetch_add(1, Ordering::Relaxed);
                        let mut result = SubstrateNode::from(node);
                        result.addr = CogAddr::from_parts(prefix, slot);
                        return Some(result);
                    }
                }
            }
        }

        // Check node space (0x80-0xFF)
        for prefix in 0x80..=0xFFu8 {
            for slot in 0..=255u8 {
                let addr = Addr::new(prefix, slot);
                if let Some(node) = bind_space.read(addr) {
                    if node.label.as_deref() == Some(label) {
                        self.stats.hot_hits.fetch_add(1, Ordering::Relaxed);
                        let mut result = SubstrateNode::from(node);
                        result.addr = CogAddr::from_parts(prefix, slot);
                        return Some(result);
                    }
                }
            }
        }

        None
    }

    // =========================================================================
    // WRITE OPERATIONS
    // =========================================================================

    /// Write a node to node space (0x80-0xFF), returns its address
    pub fn write(&self, fingerprint: [u64; FINGERPRINT_WORDS]) -> CogAddr {
        let mut bind_space = self.bind_space.write().unwrap();
        let addr = bind_space.write(fingerprint);
        let cog_addr = CogAddr::from(addr.0);

        // Add to Rubicon search index for adaptive similarity search
        {
            let mut rubicon = self.rubicon_search.write().unwrap();
            let mut addrs = self.search_addrs.write().unwrap();
            rubicon.add(&fingerprint);
            addrs.push(cog_addr);
        }

        let node = SubstrateNode::new(cog_addr, fingerprint);

        // Queue for Lance
        let mut buffer = self.write_buffer.lock().unwrap();
        let should_flush = buffer.push(WriteOp::UpsertNode(node));
        drop(buffer);

        self.stats.hot_nodes.fetch_add(1, Ordering::Relaxed);
        self.stats.pending_writes.fetch_add(1, Ordering::Relaxed);

        if should_flush {
            drop(bind_space);
            self.flush_sync();
        }

        cog_addr
    }

    /// Write a node with label to node space
    pub fn write_labeled(&self, fingerprint: [u64; FINGERPRINT_WORDS], label: &str) -> CogAddr {
        let mut bind_space = self.bind_space.write().unwrap();
        let addr = bind_space.write_labeled(fingerprint, label);
        let cog_addr = CogAddr::from(addr.0);

        // Add to Rubicon search index for adaptive similarity search
        {
            let mut rubicon = self.rubicon_search.write().unwrap();
            let mut addrs = self.search_addrs.write().unwrap();
            rubicon.add(&fingerprint);
            addrs.push(cog_addr);
        }

        let node = SubstrateNode::new(cog_addr, fingerprint).with_label(label);

        // Queue for Lance
        let mut buffer = self.write_buffer.lock().unwrap();
        buffer.push(WriteOp::UpsertNode(node));

        self.stats.hot_nodes.fetch_add(1, Ordering::Relaxed);
        self.stats.pending_writes.fetch_add(1, Ordering::Relaxed);

        cog_addr
    }

    /// Write to fluid zone (0x10-0x7F) with TTL
    pub fn write_fluid(&self, fingerprint: [u64; FINGERPRINT_WORDS], ttl: Duration) -> CogAddr {
        // Allocate in actual fluid zone
        let slot = self.next_fluid_slot.fetch_add(1, Ordering::SeqCst);

        // Wrap around within fluid zone (0x1000-0x7FFF)
        let wrapped = 0x1000 + (slot % 0x7000);
        let prefix = ((wrapped >> 8) & 0xFF) as u8;
        let slot_byte = (wrapped & 0xFF) as u8;

        let addr = Addr::new(prefix, slot_byte);
        let cog_addr = CogAddr::from(addr.0);

        // Write to BindSpace
        let mut bind_space = self.bind_space.write().unwrap();
        bind_space.write_at(addr, fingerprint);
        drop(bind_space);

        // Track TTL
        let mut tracker = self.fluid_tracker.write().unwrap();
        tracker.insert(cog_addr.0, ttl);

        self.stats.hot_nodes.fetch_add(1, Ordering::Relaxed);

        cog_addr
    }

    /// Delete a node
    pub fn delete(&self, addr: CogAddr) -> bool {
        let mut bind_space = self.bind_space.write().unwrap();
        let bind_addr = Addr::from(addr.0);

        let deleted = bind_space.delete(bind_addr).is_some();

        if deleted {
            // Queue delete for Lance
            let mut buffer = self.write_buffer.lock().unwrap();
            buffer.push(WriteOp::DeleteNode(addr));
            self.stats.pending_writes.fetch_add(1, Ordering::Relaxed);

            // Remove from fluid tracker if present
            drop(bind_space);
            let mut tracker = self.fluid_tracker.write().unwrap();
            tracker.remove(addr.0);
        }

        deleted
    }

    // =========================================================================
    // EDGE OPERATIONS
    // =========================================================================

    /// Create an edge
    pub fn link(&self, from: CogAddr, verb: CogAddr, to: CogAddr) -> Option<SubstrateEdge> {
        let mut bind_space = self.bind_space.write().unwrap();

        let from_addr = Addr::from(from.0);
        let verb_addr = Addr::from(verb.0);
        let to_addr = Addr::from(to.0);

        // Get fingerprints
        let from_fp = bind_space.read(from_addr).map(|n| n.fingerprint)?;
        let verb_fp = bind_space.read(verb_addr).map(|n| n.fingerprint)?;
        let to_fp = bind_space.read(to_addr).map(|n| n.fingerprint)?;

        // Create edge in BindSpace
        bind_space.link(from_addr, verb_addr, to_addr);

        // Create substrate edge
        let mut edge = SubstrateEdge::new(from, verb, to);
        edge.bind(&from_fp, &verb_fp, &to_fp);

        // Queue for Lance
        let mut buffer = self.write_buffer.lock().unwrap();
        buffer.push(WriteOp::UpsertEdge(edge.clone()));

        self.stats.hot_edges.fetch_add(1, Ordering::Relaxed);
        self.stats.pending_writes.fetch_add(1, Ordering::Relaxed);

        Some(edge)
    }

    /// Get outgoing edges
    pub fn edges_out(&self, from: CogAddr) -> Vec<SubstrateEdge> {
        let bind_space = self.bind_space.read().unwrap();
        let from_addr = Addr::from(from.0);

        bind_space
            .edges_out(from_addr)
            .map(|e| SubstrateEdge {
                from: CogAddr::from(e.from.0),
                to: CogAddr::from(e.to.0),
                verb: CogAddr::from(e.verb.0),
                fingerprint: e.fingerprint,
                weight: e.weight,
                created: Instant::now(),
                lance_id: None,
                dirty: false,
            })
            .collect()
    }

    /// Get incoming edges
    pub fn edges_in(&self, to: CogAddr) -> Vec<SubstrateEdge> {
        let bind_space = self.bind_space.read().unwrap();
        let to_addr = Addr::from(to.0);

        bind_space
            .edges_in(to_addr)
            .map(|e| SubstrateEdge {
                from: CogAddr::from(e.from.0),
                to: CogAddr::from(e.to.0),
                verb: CogAddr::from(e.verb.0),
                fingerprint: e.fingerprint,
                weight: e.weight,
                created: Instant::now(),
                lance_id: None,
                dirty: false,
            })
            .collect()
    }

    /// Traverse via verb
    pub fn traverse(&self, from: CogAddr, verb: CogAddr) -> Vec<CogAddr> {
        let bind_space = self.bind_space.read().unwrap();
        bind_space
            .traverse(Addr::from(from.0), Addr::from(verb.0))
            .into_iter()
            .map(|a| CogAddr::from(a.0))
            .collect()
    }

    /// N-hop traversal
    pub fn traverse_n_hops(
        &self,
        start: CogAddr,
        verb: CogAddr,
        max_hops: usize,
    ) -> Vec<(usize, CogAddr)> {
        let bind_space = self.bind_space.read().unwrap();
        bind_space
            .traverse_n_hops(Addr::from(start.0), Addr::from(verb.0), max_hops)
            .into_iter()
            .map(|(hop, a)| (hop, CogAddr::from(a.0)))
            .collect()
    }

    // =========================================================================
    // SEARCH OPERATIONS
    // =========================================================================

    /// Search for similar nodes using Belichtungsmesser adaptive thresholds
    ///
    /// Uses 7-point exposure metering to dynamically adjust thresholds:
    /// - Measures SD from cheap 1-bit samples (14 ops)
    /// - Calculates optimal threshold from SD (no quality cliffs)
    /// - Crosses Rubicon with batch processing
    /// - Retreats and re-measures if quality degrades
    /// - Infers SD trajectory for pre-adjustment
    ///
    /// Falls back to full scan if Rubicon returns insufficient results.
    pub fn search(
        &self,
        query_fp: &[u64; FINGERPRINT_WORDS],
        k: usize,
        threshold: f32,
    ) -> Vec<(CogAddr, u32, f32)> {
        let max_distance = ((1.0 - threshold) * crate::FINGERPRINT_BITS as f32) as u32;

        // First try Rubicon search with adaptive thresholds
        let mut results = {
            let mut rubicon = self.rubicon_search.write().unwrap();
            let addrs = self.search_addrs.read().unwrap();

            // Rubicon search with Belichtungsmesser-guided thresholds
            let candidates = rubicon.search_adaptive(query_fp, k * 10);

            // Map indices back to addresses and filter by threshold
            let res: Vec<(CogAddr, u32, f32)> = candidates
                .into_iter()
                .filter_map(|(idx, dist)| {
                    if dist <= max_distance && idx < addrs.len() {
                        let addr = addrs[idx];
                        let sim = 1.0 - (dist as f32 / crate::FINGERPRINT_BITS as f32);
                        Some((addr, dist, sim))
                    } else {
                        None
                    }
                })
                .collect();
            res
        };

        // If Rubicon didn't find enough results, fall back to full scan
        // This handles edge cases or very small indices
        if results.len() < k {
            let bind_space = self.bind_space.read().unwrap();
            results.clear();

            // Scan node space (0x80-0xFF)
            for prefix in 0x80..=0xFF_u8 {
                for slot in 0..=255u8 {
                    let addr = Addr::new(prefix, slot);
                    if let Some(node) = bind_space.read(addr) {
                        let dist = hamming_distance(query_fp, &node.fingerprint);
                        if dist <= max_distance {
                            let sim = 1.0 - (dist as f32 / crate::FINGERPRINT_BITS as f32);
                            results.push((CogAddr::from_parts(prefix, slot), dist, sim));
                        }
                    }
                }
            }
        }

        // Sort by distance and truncate to k
        results.sort_by_key(|(_, d, _)| *d);
        results.truncate(k);

        results
    }

    /// Get current search quality (0.0-1.0)
    /// Higher = more consistent results, lower = noisy/uncertain
    pub fn search_quality(&self) -> f32 {
        self.rubicon_search.read().unwrap().quality()
    }

    /// Get current adaptive threshold
    pub fn search_threshold(&self) -> u16 {
        self.rubicon_search.read().unwrap().threshold()
    }

    /// Resonate: find nodes that resonate with query
    pub fn resonate(&self, query_fp: &[u64; FINGERPRINT_WORDS], k: usize) -> Vec<(CogAddr, f32)> {
        self.search(query_fp, k, 0.0)
            .into_iter()
            .map(|(addr, _, sim)| (addr, sim))
            .collect()
    }

    // =========================================================================
    // LIFECYCLE OPERATIONS
    // =========================================================================

    /// Tick: expire old fluid entries, promote/demote as needed
    pub fn tick(&self) {
        // 1. Find and evict expired fluid entries
        let expired = {
            let tracker = self.fluid_tracker.read().unwrap();
            tracker.expired_entries()
        };

        for addr in expired {
            let mut bind_space = self.bind_space.write().unwrap();
            bind_space.delete(Addr::from(addr));
            drop(bind_space);

            let mut tracker = self.fluid_tracker.write().unwrap();
            tracker.remove(addr);

            self.stats.evictions.fetch_add(1, Ordering::Relaxed);
        }

        // 2. Check for promotion candidates in fluid zone
        // (entries with high access count)
        let bind_space = self.bind_space.read().unwrap();
        let mut promote_candidates = Vec::new();

        for prefix in 0x10..0x80u8 {
            for slot in 0..=255u8 {
                let addr = Addr::new(prefix, slot);
                if let Some(node) = bind_space.read(addr) {
                    if node.access_count >= self.config.promotion_threshold {
                        promote_candidates.push((addr, node.fingerprint, node.label.clone()));
                    }
                }
            }
        }
        drop(bind_space);

        // Promote candidates
        for (old_addr, fingerprint, label) in promote_candidates {
            // Write to node space
            let new_addr = if let Some(ref lbl) = label {
                self.write_labeled(fingerprint, lbl)
            } else {
                self.write(fingerprint)
            };

            // Delete from fluid
            let mut bind_space = self.bind_space.write().unwrap();
            bind_space.delete(old_addr);
            drop(bind_space);

            let mut tracker = self.fluid_tracker.write().unwrap();
            tracker.remove(old_addr.0);

            self.stats.promotions.fetch_add(1, Ordering::Relaxed);
            let _ = new_addr; // Used to create the node
        }
    }

    /// Crystallize: promote a fluid entry to node space
    pub fn crystallize(&self, addr: CogAddr) -> Option<CogAddr> {
        if !addr.is_fluid() {
            return None;
        }

        let bind_space = self.bind_space.read().unwrap();
        let bind_addr = Addr::from(addr.0);

        let node = bind_space.read(bind_addr)?;
        let fingerprint = node.fingerprint;
        let label = node.label.clone();
        drop(bind_space);

        // Write to node space
        let new_addr = if let Some(ref lbl) = label {
            self.write_labeled(fingerprint, lbl)
        } else {
            self.write(fingerprint)
        };

        // Delete from fluid
        self.delete(addr);

        self.stats.promotions.fetch_add(1, Ordering::Relaxed);
        Some(new_addr)
    }

    /// Evaporate: demote a node entry to fluid space with TTL
    pub fn evaporate(&self, addr: CogAddr, ttl: Duration) -> Option<CogAddr> {
        if !addr.is_node() {
            return None;
        }

        let bind_space = self.bind_space.read().unwrap();
        let bind_addr = Addr::from(addr.0);

        let node = bind_space.read(bind_addr)?;
        let fingerprint = node.fingerprint;
        drop(bind_space);

        // Write to fluid zone
        let new_addr = self.write_fluid(fingerprint, ttl);

        // Delete from node space
        self.delete(addr);

        self.stats.demotions.fetch_add(1, Ordering::Relaxed);
        Some(new_addr)
    }

    /// Flush write buffer to Lance (sync version)
    pub fn flush_sync(&self) {
        let mut buffer = self.write_buffer.lock().unwrap();
        let ops = buffer.drain();

        if ops.is_empty() {
            return;
        }

        // In production, this would batch write to Lance
        // For now, just update stats
        self.stats
            .lance_writes
            .fetch_add(ops.len() as u64, Ordering::Relaxed);
        self.stats.pending_writes.store(0, Ordering::Relaxed);
    }

    /// Commit: flush all pending writes
    pub fn commit(&self) {
        self.flush_sync();
        self.version.fetch_add(1, Ordering::SeqCst);
    }

    // =========================================================================
    // SURFACE ACCESS (CAM operations)
    // =========================================================================

    /// Get a verb by name
    pub fn verb(&self, name: &str) -> Option<CogAddr> {
        let bind_space = self.bind_space.read().unwrap();
        bind_space.verb(name).map(|a| CogAddr::from(a.0))
    }

    /// Get a surface operation by compartment and name
    pub fn surface_op(&self, compartment: u8, name: &str) -> Option<CogAddr> {
        let bind_space = self.bind_space.read().unwrap();
        bind_space
            .surface_op(compartment, name)
            .map(|a| CogAddr::from(a.0))
    }

    // =========================================================================
    // STATISTICS
    // =========================================================================

    /// Get statistics
    pub fn stats(&self) -> &SubstrateStats {
        &self.stats
    }

    /// Get configuration
    pub fn config(&self) -> &SubstrateConfig {
        &self.config
    }

    /// Current version
    pub fn version(&self) -> u64 {
        self.version.load(Ordering::SeqCst)
    }

    /// Is Lance connected?
    pub fn is_lance_connected(&self) -> bool {
        self.lance_connected.load(Ordering::Relaxed)
    }
}

impl Default for Substrate {
    fn default() -> Self {
        Self::default_new()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_substrate_write_read() {
        let substrate = Substrate::default_new();

        let fp = [42u64; FINGERPRINT_WORDS];
        let addr = substrate.write(fp);

        assert!(addr.is_node());

        let node = substrate.read(addr);
        assert!(node.is_some());
        assert_eq!(node.unwrap().fingerprint, fp);
    }

    #[test]
    fn test_substrate_labeled() {
        let substrate = Substrate::default_new();

        let fp = [1u64; FINGERPRINT_WORDS];
        let addr = substrate.write_labeled(fp, "test_node");

        let node = substrate.read(addr).unwrap();
        assert_eq!(node.label.as_deref(), Some("test_node"));
    }

    #[test]
    fn test_substrate_link() {
        let substrate = Substrate::default_new();

        let a = substrate.write_labeled([1u64; FINGERPRINT_WORDS], "A");
        let b = substrate.write_labeled([2u64; FINGERPRINT_WORDS], "B");

        let causes = substrate.verb("CAUSES").unwrap();
        let edge = substrate.link(a, causes, b);

        assert!(edge.is_some());

        let targets = substrate.traverse(a, causes);
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0], b);
    }

    #[test]
    fn test_substrate_search() {
        let substrate = Substrate::default_new();

        let fp1 = [0xAAAAAAAAAAAAAAAAu64; FINGERPRINT_WORDS];
        let fp2 = [0xBBBBBBBBBBBBBBBBu64; FINGERPRINT_WORDS];
        let fp3 = [0xAAAAAAAAAAAABBBBu64; FINGERPRINT_WORDS];

        substrate.write_labeled(fp1, "node1");
        substrate.write_labeled(fp2, "node2");
        substrate.write_labeled(fp3, "node3");

        let results = substrate.search(&fp1, 10, 0.5);

        assert!(!results.is_empty());
    }

    #[test]
    fn test_substrate_fluid_zone() {
        let substrate = Substrate::default_new();

        let fp = [99u64; FINGERPRINT_WORDS];
        let addr = substrate.write_fluid(fp, Duration::from_millis(100));

        // Should be in fluid zone
        assert!(addr.is_fluid());

        // Should be readable
        let node = substrate.read(addr);
        assert!(node.is_some());

        // After TTL, should expire on tick
        std::thread::sleep(Duration::from_millis(150));
        substrate.tick();

        // Should now be expired
        let node = substrate.read(addr);
        assert!(node.is_none());
    }

    #[test]
    fn test_substrate_crystallize() {
        let substrate = Substrate::default_new();

        // Write to fluid
        let fp = [77u64; FINGERPRINT_WORDS];
        let fluid_addr = substrate.write_fluid(fp, Duration::from_secs(300));
        assert!(fluid_addr.is_fluid());

        // Crystallize to node
        let node_addr = substrate.crystallize(fluid_addr);
        assert!(node_addr.is_some());
        assert!(node_addr.unwrap().is_node());

        // Old address should be gone
        let old = substrate.read(fluid_addr);
        assert!(old.is_none());

        // New address should have the data
        let new = substrate.read(node_addr.unwrap());
        assert!(new.is_some());
        assert_eq!(new.unwrap().fingerprint, fp);
    }

    #[test]
    fn test_substrate_evaporate() {
        let substrate = Substrate::default_new();

        // Write to node space
        let fp = [88u64; FINGERPRINT_WORDS];
        let node_addr = substrate.write(fp);
        assert!(node_addr.is_node());

        // Evaporate to fluid
        let fluid_addr = substrate.evaporate(node_addr, Duration::from_secs(60));
        assert!(fluid_addr.is_some());
        assert!(fluid_addr.unwrap().is_fluid());

        // Old address should be gone
        let old = substrate.read(node_addr);
        assert!(old.is_none());

        // New address should have the data
        let new = substrate.read(fluid_addr.unwrap());
        assert!(new.is_some());
        assert_eq!(new.unwrap().fingerprint, fp);
    }

    #[test]
    fn test_substrate_stats() {
        let substrate = Substrate::default_new();

        let fp = [1u64; FINGERPRINT_WORDS];
        let addr = substrate.write(fp);

        // Read should hit
        substrate.read(addr);
        assert_eq!(substrate.stats.hot_hits.load(Ordering::Relaxed), 1);

        // Read non-existent should miss
        substrate.read(CogAddr::new(0xFFFF));
        assert_eq!(substrate.stats.hot_misses.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_substrate_commit() {
        let substrate = Substrate::default_new();

        for i in 0..10 {
            let fp = [i as u64; FINGERPRINT_WORDS];
            substrate.write(fp);
        }

        assert!(substrate.stats.pending_writes.load(Ordering::Relaxed) > 0);

        substrate.commit();

        assert_eq!(substrate.version(), 2);
    }
}
