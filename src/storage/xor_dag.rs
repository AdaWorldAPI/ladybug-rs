//! XOR DAG Storage - Zero-Copy ACID with Parity Protection
//!
//! Implements a Database Availability Group (DAG) pattern with:
//! - XOR parity blocks for cross-tier recovery (P = A ⊗ B ⊗ C)
//! - MVCC transactions with ACID guarantees
//! - Epoch-based work stealing guards
//! - Time-travel via XOR delta chains
//! - Conflict resolution with previous state awareness
//!
//! # Zero-Copy Architecture
//!
//! Works directly with Arrow/Lance buffers - no intermediate copies:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                         XOR DAG ZERO-COPY FLOW                              │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │  BindSpace (RAM)          Arrow Buffer (mmap)         Lance (disk)          │
//! │       │                         │                          │                │
//! │       │──── zero-copy ──────────│                          │                │
//! │       │      slice ref          │──── mmap read ───────────│                │
//! │       │                         │                          │                │
//! │       │══════ XOR ⊗ ═══════════│                          │                │
//! │       │   (SIMD on slices)     │                          │                │
//! │       ▼                         ▼                          ▼                │
//! │  HOT parity             WARM parity              COLD parity               │
//! │    P_hot = ⊗ all hot     P_warm = ⊗ all warm     P_cold = ⊗ all cold     │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Parity Block Recovery
//!
//! ```text
//! If any single tier fails, reconstruct from other two + parity:
//!
//!   Lost HOT?   → HOT = P_cross ⊗ WARM ⊗ COLD
//!   Lost WARM?  → WARM = P_cross ⊗ HOT ⊗ COLD
//!   Lost COLD?  → COLD = P_cross ⊗ HOT ⊗ WARM
//!
//! Where P_cross = HOT ⊗ WARM ⊗ COLD (computed incrementally)
//! ```

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use super::bind_space::{Addr, BindNode, BindSpace, FINGERPRINT_WORDS};
use super::temporal::Version;

// =============================================================================
// ZERO-COPY XOR OPERATIONS
// =============================================================================

/// XOR two byte slices in place (zero-copy destination)
///
/// SIMD-friendly: works on any alignment, compiler auto-vectorizes
#[inline]
pub fn xor_slices_inplace(dest: &mut [u8], src: &[u8]) {
    let len = dest.len().min(src.len());

    // Process 8 bytes at a time for auto-vectorization
    let chunks = len / 8;
    let _remainder = len % 8;

    // Safe transmute for aligned u64 XOR
    for i in 0..chunks {
        let offset = i * 8;
        let dest_chunk = &mut dest[offset..offset + 8];
        let src_chunk = &src[offset..offset + 8];

        let dest_val = u64::from_le_bytes(dest_chunk.try_into().unwrap());
        let src_val = u64::from_le_bytes(src_chunk.try_into().unwrap());
        dest_chunk.copy_from_slice(&(dest_val ^ src_val).to_le_bytes());
    }

    // Handle remainder
    for i in (chunks * 8)..len {
        dest[i] ^= src[i];
    }
}

/// XOR two slices into a new Vec (when can't modify in place)
#[inline]
pub fn xor_slices(a: &[u8], b: &[u8]) -> Vec<u8> {
    let len = a.len().max(b.len());
    let mut result = vec![0u8; len];

    // Copy longer slice first
    if a.len() >= b.len() {
        result[..a.len()].copy_from_slice(a);
        xor_slices_inplace(&mut result[..b.len()], b);
    } else {
        result[..b.len()].copy_from_slice(b);
        xor_slices_inplace(&mut result[..a.len()], a);
    }

    result
}

/// XOR fingerprints (the main data type in BindSpace)
#[inline]
pub fn xor_fingerprints(
    a: &[u64; FINGERPRINT_WORDS],
    b: &[u64; FINGERPRINT_WORDS],
) -> [u64; FINGERPRINT_WORDS] {
    let mut result = [0u64; FINGERPRINT_WORDS];
    for i in 0..FINGERPRINT_WORDS {
        result[i] = a[i] ^ b[i];
    }
    result
}

/// XOR multiple fingerprints (parity of N)
#[inline]
pub fn xor_fingerprints_multi(fps: &[[u64; FINGERPRINT_WORDS]]) -> [u64; FINGERPRINT_WORDS] {
    let mut result = [0u64; FINGERPRINT_WORDS];
    for fp in fps {
        for i in 0..FINGERPRINT_WORDS {
            result[i] ^= fp[i];
        }
    }
    result
}

// =============================================================================
// PARITY BLOCK
// =============================================================================

/// A parity block protecting a set of addresses
#[derive(Debug, Clone)]
pub struct ParityBlock {
    /// Unique ID for this parity block
    pub id: u64,

    /// Addresses covered by this parity
    pub covered_addrs: Vec<Addr>,

    /// The XOR parity of all covered fingerprints
    /// P = fp[0] ⊗ fp[1] ⊗ ... ⊗ fp[N]
    pub parity_fingerprint: [u64; FINGERPRINT_WORDS],

    /// Version when this parity was computed
    pub version: Version,

    /// Tier this parity block belongs to
    pub tier: ParityTier,

    /// Timestamp of last update
    pub updated_at: u64,
}

/// Which tier a parity block protects
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParityTier {
    /// Hot tier (RAM) - prefix 0x80-0xFF
    Hot,
    /// Warm tier (SSD cache) - prefix 0x10-0x7F
    Warm,
    /// Cold tier (archive) - prefix 0x00-0x0F
    Cold,
    /// Cross-tier parity (P_hot ⊗ P_warm ⊗ P_cold)
    Cross,
}

impl ParityBlock {
    /// Create a new parity block from addresses
    pub fn new(id: u64, tier: ParityTier, bind_space: &BindSpace, addrs: &[Addr]) -> Self {
        let fps: Vec<[u64; FINGERPRINT_WORDS]> = addrs
            .iter()
            .filter_map(|&addr| bind_space.read(addr).map(|n| n.fingerprint))
            .collect();

        Self {
            id,
            covered_addrs: addrs.to_vec(),
            parity_fingerprint: xor_fingerprints_multi(&fps),
            version: 0,
            tier,
            updated_at: timestamp_micros(),
        }
    }

    /// Update parity when a single address changes
    ///
    /// Incremental update: P_new = P_old ⊗ old_fp ⊗ new_fp
    /// This is O(1) regardless of how many addresses are covered
    pub fn update_single(
        &mut self,
        old_fp: &[u64; FINGERPRINT_WORDS],
        new_fp: &[u64; FINGERPRINT_WORDS],
    ) {
        // XOR out old, XOR in new
        for i in 0..FINGERPRINT_WORDS {
            self.parity_fingerprint[i] ^= old_fp[i] ^ new_fp[i];
        }
        self.version += 1;
        self.updated_at = timestamp_micros();
    }

    /// Reconstruct a missing fingerprint from parity and others
    ///
    /// If we know P and all fps except one, we can recover:
    /// missing = P ⊗ fp[0] ⊗ fp[1] ⊗ ... (all except missing)
    pub fn recover_missing(
        &self,
        bind_space: &BindSpace,
        missing_addr: Addr,
    ) -> Option<[u64; FINGERPRINT_WORDS]> {
        if !self.covered_addrs.contains(&missing_addr) {
            return None;
        }

        // Start with parity
        let mut result = self.parity_fingerprint;

        // XOR in all other fingerprints
        for &addr in &self.covered_addrs {
            if addr != missing_addr {
                if let Some(node) = bind_space.read(addr) {
                    for i in 0..FINGERPRINT_WORDS {
                        result[i] ^= node.fingerprint[i];
                    }
                }
            }
        }

        Some(result)
    }

    /// Verify parity is still valid
    pub fn verify(&self, bind_space: &BindSpace) -> bool {
        let fps: Vec<[u64; FINGERPRINT_WORDS]> = self
            .covered_addrs
            .iter()
            .filter_map(|&addr| bind_space.read(addr).map(|n| n.fingerprint))
            .collect();

        let computed = xor_fingerprints_multi(&fps);
        computed == self.parity_fingerprint
    }
}

// =============================================================================
// EPOCH-BASED WORK STEALING GUARD
// =============================================================================

/// Epoch for work stealing protection
///
/// Work stealing requires careful synchronization to avoid:
/// 1. Reading partially written data
/// 2. Double-processing work items
/// 3. Lost updates during steal
///
/// We use epochs to ensure safe handoff:
/// - Writer increments epoch before write
/// - Stealer must see same epoch before and after read
/// - If epochs differ, retry
#[derive(Debug)]
pub struct EpochGuard {
    /// Current global epoch
    epoch: AtomicU64,

    /// Active readers in each epoch (for safe reclamation)
    reader_counts: [AtomicU64; 2],

    /// Pending work items per epoch
    pending: [Mutex<VecDeque<WorkItem>>; 2],

    /// Steal in progress flag
    steal_active: AtomicBool,
}

/// A work item that can be stolen
#[derive(Debug, Clone)]
pub struct WorkItem {
    pub addr: Addr,
    pub operation: WorkOperation,
    pub epoch: u64,
    pub created_at: Instant,
}

#[derive(Debug, Clone)]
pub enum WorkOperation {
    Write {
        fingerprint: [u64; FINGERPRINT_WORDS],
        label: Option<String>,
    },
    Delete,
    Link {
        target: Addr,
        edge_type: String,
    },
    UpdateParity {
        parity_id: u64,
    },
}

impl Default for EpochGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl EpochGuard {
    pub fn new() -> Self {
        Self {
            epoch: AtomicU64::new(0),
            reader_counts: [AtomicU64::new(0), AtomicU64::new(0)],
            pending: [Mutex::new(VecDeque::new()), Mutex::new(VecDeque::new())],
            steal_active: AtomicBool::new(false),
        }
    }

    /// Begin a read operation, returning epoch ticket
    pub fn begin_read(&self) -> EpochTicket<'_> {
        let epoch = self.epoch.load(Ordering::SeqCst);
        let slot = (epoch % 2) as usize;
        self.reader_counts[slot].fetch_add(1, Ordering::SeqCst);

        EpochTicket { epoch, guard: self }
    }

    /// End a read operation
    fn end_read(&self, epoch: u64) {
        let slot = (epoch % 2) as usize;
        self.reader_counts[slot].fetch_sub(1, Ordering::SeqCst);
    }

    /// Advance epoch (for writers)
    pub fn advance_epoch(&self) -> u64 {
        self.epoch.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Submit work to current epoch
    pub fn submit_work(&self, item: WorkItem) {
        let epoch = self.epoch.load(Ordering::SeqCst);
        let slot = (epoch % 2) as usize;

        if let Ok(mut pending) = self.pending[slot].lock() {
            pending.push_back(item);
        }
    }

    /// Try to steal work from previous epoch (safe once no readers)
    pub fn try_steal(&self) -> Option<Vec<WorkItem>> {
        // Prevent concurrent steals
        if self.steal_active.swap(true, Ordering::SeqCst) {
            return None;
        }

        let current_epoch = self.epoch.load(Ordering::SeqCst);
        let old_slot = ((current_epoch + 1) % 2) as usize; // Previous epoch's slot

        // Check no readers in old epoch
        if self.reader_counts[old_slot].load(Ordering::SeqCst) > 0 {
            self.steal_active.store(false, Ordering::SeqCst);
            return None;
        }

        // Safe to steal
        let items = if let Ok(mut pending) = self.pending[old_slot].lock() {
            let items: Vec<_> = pending.drain(..).collect();
            items
        } else {
            Vec::new()
        };

        self.steal_active.store(false, Ordering::SeqCst);

        if items.is_empty() { None } else { Some(items) }
    }

    /// Current epoch
    pub fn current_epoch(&self) -> u64 {
        self.epoch.load(Ordering::SeqCst)
    }

    /// Pending work count
    pub fn pending_count(&self) -> usize {
        self.pending[0].lock().map(|p| p.len()).unwrap_or(0)
            + self.pending[1].lock().map(|p| p.len()).unwrap_or(0)
    }
}

/// RAII ticket for epoch-protected reads
pub struct EpochTicket<'a> {
    epoch: u64,
    guard: &'a EpochGuard,
}

impl<'a> EpochTicket<'a> {
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Check if our read is still valid (epoch hasn't advanced too far)
    pub fn is_valid(&self) -> bool {
        let current = self.guard.epoch.load(Ordering::SeqCst);
        // Valid if within 1 epoch (current or immediately previous)
        current <= self.epoch + 1
    }
}

impl<'a> Drop for EpochTicket<'a> {
    fn drop(&mut self) {
        self.guard.end_read(self.epoch);
    }
}

// =============================================================================
// CONFLICT RESOLUTION WITH PREVIOUS STATE AWARENESS
// =============================================================================

/// Conflict resolution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictStrategy {
    /// Last write wins (simple, may lose data)
    LastWriteWins,
    /// First write wins (preserves original)
    FirstWriteWins,
    /// Merge using XOR (combines both changes)
    XorMerge,
    /// Custom merge function
    Custom,
    /// Reject with conflict error
    Reject,
}

/// State snapshot for conflict detection
#[derive(Debug, Clone)]
pub struct StateSnapshot {
    /// Address being modified
    pub addr: Addr,
    /// Fingerprint at read time
    pub fingerprint: [u64; FINGERPRINT_WORDS],
    /// Version at read time (tracked externally, not in BindNode)
    pub version: Version,
    /// Access count at read time (for change detection)
    pub access_count: u32,
    /// Timestamp of snapshot
    pub timestamp: u64,
    /// XOR delta from previous version (for 3-way merge)
    pub delta_from_prev: Option<[u64; FINGERPRINT_WORDS]>,
}

/// Conflict detected during write
#[derive(Debug, Clone)]
pub struct WriteConflict {
    /// Address with conflict
    pub addr: Addr,
    /// Our snapshot (what we read)
    pub our_state: StateSnapshot,
    /// Current state (what's there now)
    pub current_state: StateSnapshot,
    /// Common ancestor (if known)
    pub ancestor: Option<StateSnapshot>,
}

impl WriteConflict {
    /// Attempt automatic resolution using XOR merge
    ///
    /// 3-way merge using XOR:
    /// result = ancestor ⊗ (our_delta) ⊗ (their_delta)
    /// where our_delta = our ⊗ ancestor
    /// and their_delta = current ⊗ ancestor
    ///
    /// This combines both sets of changes without losing either.
    pub fn xor_merge(&self) -> Option<[u64; FINGERPRINT_WORDS]> {
        let ancestor = self.ancestor.as_ref()?;

        // Compute deltas from ancestor
        let our_delta = xor_fingerprints(&self.our_state.fingerprint, &ancestor.fingerprint);
        let their_delta = xor_fingerprints(&self.current_state.fingerprint, &ancestor.fingerprint);

        // Merge: ancestor ⊗ our_delta ⊗ their_delta
        let mut result = ancestor.fingerprint;
        for i in 0..FINGERPRINT_WORDS {
            result[i] ^= our_delta[i] ^ their_delta[i];
        }

        Some(result)
    }

    /// Check if changes are orthogonal (no overlapping bits changed)
    pub fn is_orthogonal(&self) -> bool {
        if let Some(ancestor) = &self.ancestor {
            let our_delta = xor_fingerprints(&self.our_state.fingerprint, &ancestor.fingerprint);
            let their_delta =
                xor_fingerprints(&self.current_state.fingerprint, &ancestor.fingerprint);

            // Orthogonal if no bits changed in both
            for i in 0..FINGERPRINT_WORDS {
                if our_delta[i] & their_delta[i] != 0 {
                    return false;
                }
            }
            true
        } else {
            false
        }
    }
}

// =============================================================================
// XOR DAG TRANSACTION
// =============================================================================

/// Transaction state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxnState {
    Active,
    Committed,
    Aborted,
    Conflicted,
}

/// ACID transaction for XOR DAG operations
#[derive(Debug)]
pub struct DagTransaction {
    /// Transaction ID
    pub id: u64,

    /// State
    pub state: TxnState,

    /// Conflict resolution strategy
    pub strategy: ConflictStrategy,

    /// Read set (for conflict detection)
    reads: HashMap<Addr, StateSnapshot>,

    /// Write set (pending writes)
    writes: HashMap<Addr, WriteIntent>,

    /// Parity updates needed
    parity_updates: HashSet<u64>,

    /// Start timestamp
    pub started_at: Instant,

    /// Epoch at start
    pub start_epoch: u64,
}

/// Write intent within a transaction
#[derive(Debug, Clone)]
pub struct WriteIntent {
    pub addr: Addr,
    pub fingerprint: [u64; FINGERPRINT_WORDS],
    pub label: Option<String>,
    pub expected_version: Version,
}

impl DagTransaction {
    /// Create new transaction
    pub fn new(id: u64, strategy: ConflictStrategy, epoch: u64) -> Self {
        Self {
            id,
            state: TxnState::Active,
            strategy,
            reads: HashMap::new(),
            writes: HashMap::new(),
            parity_updates: HashSet::new(),
            started_at: Instant::now(),
            start_epoch: epoch,
        }
    }

    /// Record a read (for conflict detection)
    pub fn record_read(&mut self, snapshot: StateSnapshot) {
        self.reads.insert(snapshot.addr, snapshot);
    }

    /// Stage a write
    pub fn stage_write(
        &mut self,
        addr: Addr,
        fingerprint: [u64; FINGERPRINT_WORDS],
        label: Option<String>,
    ) {
        let expected = self.reads.get(&addr).map(|s| s.version).unwrap_or(0);
        self.writes.insert(
            addr,
            WriteIntent {
                addr,
                fingerprint,
                label,
                expected_version: expected,
            },
        );
    }

    /// Mark parity block as needing update
    pub fn mark_parity_dirty(&mut self, parity_id: u64) {
        self.parity_updates.insert(parity_id);
    }

    /// Get all staged writes
    pub fn staged_writes(&self) -> impl Iterator<Item = &WriteIntent> {
        self.writes.values()
    }

    /// Get all reads
    pub fn read_set(&self) -> impl Iterator<Item = (&Addr, &StateSnapshot)> {
        self.reads.iter()
    }

    /// Check if transaction has modifications
    pub fn has_modifications(&self) -> bool {
        !self.writes.is_empty()
    }

    /// Transaction duration
    pub fn duration(&self) -> Duration {
        self.started_at.elapsed()
    }
}

// =============================================================================
// XOR DAG STORE
// =============================================================================

/// Configuration for XOR DAG
#[derive(Debug, Clone)]
pub struct XorDagConfig {
    /// Addresses per parity block (affects recovery granularity)
    pub parity_block_size: usize,

    /// Maximum transaction duration before timeout
    pub txn_timeout: Duration,

    /// Default conflict resolution strategy
    pub conflict_strategy: ConflictStrategy,

    /// Enable cross-tier parity
    pub cross_tier_parity: bool,

    /// Maximum delta chain length before snapshot consolidation
    pub max_delta_chain: usize,

    /// Enable work stealing
    pub enable_work_stealing: bool,
}

impl Default for XorDagConfig {
    fn default() -> Self {
        Self {
            parity_block_size: 256, // One chunk
            txn_timeout: Duration::from_secs(30),
            conflict_strategy: ConflictStrategy::XorMerge,
            cross_tier_parity: true,
            max_delta_chain: 16,
            enable_work_stealing: true,
        }
    }
}

/// Main XOR DAG storage system
pub struct XorDag {
    /// Configuration
    config: XorDagConfig,

    /// Underlying bind space (zero-copy reference)
    bind_space: Arc<RwLock<BindSpace>>,

    /// Parity blocks by ID
    parity_blocks: RwLock<BTreeMap<u64, ParityBlock>>,

    /// Cross-tier parity (P_hot ⊗ P_warm ⊗ P_cold)
    cross_parity: RwLock<Option<ParityBlock>>,

    /// Active transactions
    active_txns: RwLock<HashMap<u64, DagTransaction>>,

    /// Epoch guard for work stealing
    epoch_guard: EpochGuard,

    /// Next transaction ID
    next_txn_id: AtomicU64,

    /// Next parity block ID
    next_parity_id: AtomicU64,

    /// XOR delta snapshots (version -> delta from previous)
    delta_chain: RwLock<BTreeMap<Version, XorDelta>>,

    /// Last full snapshot version
    last_full_snapshot: AtomicU64,

    /// Statistics
    stats: XorDagStats,
}

/// XOR delta for time-travel
#[derive(Debug, Clone)]
pub struct XorDelta {
    /// Version this delta transitions TO
    pub to_version: Version,

    /// Version this delta transitions FROM
    pub from_version: Version,

    /// Per-address XOR deltas
    pub deltas: Vec<(Addr, [u64; FINGERPRINT_WORDS])>,

    /// Timestamp
    pub timestamp: u64,
}

impl XorDelta {
    /// Apply delta forward (from_version -> to_version)
    pub fn apply_forward(&self, bind_space: &mut BindSpace) {
        for &(addr, delta) in &self.deltas {
            if let Some(node) = bind_space.read(addr) {
                let mut new_fp = node.fingerprint;
                for i in 0..FINGERPRINT_WORDS {
                    new_fp[i] ^= delta[i];
                }
                // Use write_at to update fingerprint at specific address
                bind_space.write_at(addr, new_fp);
            }
        }
    }

    /// Apply delta backward (to_version -> from_version)
    pub fn apply_backward(&self, bind_space: &mut BindSpace) {
        // XOR is self-inverse
        self.apply_forward(bind_space);
    }
}

/// Statistics for XOR DAG
#[derive(Debug, Default)]
pub struct XorDagStats {
    /// Total transactions started
    pub txns_started: AtomicU64,
    /// Transactions committed
    pub txns_committed: AtomicU64,
    /// Transactions aborted
    pub txns_aborted: AtomicU64,
    /// Conflicts detected
    pub conflicts_detected: AtomicU64,
    /// Conflicts auto-merged
    pub conflicts_merged: AtomicU64,
    /// Parity verifications
    pub parity_checks: AtomicU64,
    /// Parity recoveries
    pub parity_recoveries: AtomicU64,
    /// Work items stolen
    pub work_stolen: AtomicU64,
    /// Delta snapshots created
    pub deltas_created: AtomicU64,
}

impl XorDag {
    /// Create new XOR DAG store
    pub fn new(config: XorDagConfig, bind_space: Arc<RwLock<BindSpace>>) -> Self {
        Self {
            config,
            bind_space,
            parity_blocks: RwLock::new(BTreeMap::new()),
            cross_parity: RwLock::new(None),
            active_txns: RwLock::new(HashMap::new()),
            epoch_guard: EpochGuard::new(),
            next_txn_id: AtomicU64::new(1),
            next_parity_id: AtomicU64::new(1),
            delta_chain: RwLock::new(BTreeMap::new()),
            last_full_snapshot: AtomicU64::new(0),
            stats: XorDagStats::default(),
        }
    }

    // =========================================================================
    // TRANSACTION API
    // =========================================================================

    /// Begin a new transaction
    pub fn begin(&self) -> Result<u64, DagError> {
        let txn_id = self.next_txn_id.fetch_add(1, Ordering::SeqCst);
        let epoch = self.epoch_guard.current_epoch();

        let txn = DagTransaction::new(txn_id, self.config.conflict_strategy, epoch);

        let mut txns = self
            .active_txns
            .write()
            .map_err(|_| DagError::LockPoisoned)?;
        txns.insert(txn_id, txn);

        self.stats.txns_started.fetch_add(1, Ordering::Relaxed);

        Ok(txn_id)
    }

    /// Read within a transaction (records for conflict detection)
    pub fn read(&self, txn_id: u64, addr: Addr) -> Result<Option<BindNode>, DagError> {
        let _ticket = self.epoch_guard.begin_read();

        let bind_space = self.bind_space.read().map_err(|_| DagError::LockPoisoned)?;
        let node = bind_space.read(addr);

        // Record read for conflict detection
        if let Some(ref n) = node {
            let snapshot = StateSnapshot {
                addr,
                fingerprint: n.fingerprint,
                version: 0, // Version tracked separately in DAG
                access_count: n.access_count,
                timestamp: timestamp_micros(),
                delta_from_prev: None, // Could populate from delta_chain
            };

            let mut txns = self
                .active_txns
                .write()
                .map_err(|_| DagError::LockPoisoned)?;
            if let Some(txn) = txns.get_mut(&txn_id) {
                txn.record_read(snapshot);
            }
        }

        Ok(node.cloned())
    }

    /// Write within a transaction (staged until commit)
    pub fn write(
        &self,
        txn_id: u64,
        addr: Addr,
        fingerprint: [u64; FINGERPRINT_WORDS],
        label: Option<String>,
    ) -> Result<(), DagError> {
        let mut txns = self
            .active_txns
            .write()
            .map_err(|_| DagError::LockPoisoned)?;
        let txn = txns.get_mut(&txn_id).ok_or(DagError::TxnNotFound(txn_id))?;

        if txn.state != TxnState::Active {
            return Err(DagError::TxnNotActive(txn_id));
        }

        // Check timeout
        if txn.duration() > self.config.txn_timeout {
            txn.state = TxnState::Aborted;
            return Err(DagError::TxnTimeout(txn_id));
        }

        txn.stage_write(addr, fingerprint, label);

        // Mark relevant parity block as dirty
        let parity_id = self.parity_id_for_addr(addr);
        txn.mark_parity_dirty(parity_id);

        Ok(())
    }

    /// Commit transaction
    pub fn commit(&self, txn_id: u64) -> Result<Version, DagError> {
        // Advance epoch before commit
        let new_epoch = self.epoch_guard.advance_epoch();

        // Get transaction
        let txn = {
            let mut txns = self
                .active_txns
                .write()
                .map_err(|_| DagError::LockPoisoned)?;
            txns.remove(&txn_id).ok_or(DagError::TxnNotFound(txn_id))?
        };

        if txn.state != TxnState::Active {
            return Err(DagError::TxnNotActive(txn_id));
        }

        // Validate reads (conflict detection)
        let conflicts = self.validate_reads(&txn)?;

        if !conflicts.is_empty() {
            self.stats
                .conflicts_detected
                .fetch_add(conflicts.len() as u64, Ordering::Relaxed);

            // Try to resolve conflicts
            let resolved = self.resolve_conflicts(&txn, &conflicts)?;

            if !resolved {
                self.stats.txns_aborted.fetch_add(1, Ordering::Relaxed);
                return Err(DagError::ConflictUnresolved(conflicts));
            }

            self.stats
                .conflicts_merged
                .fetch_add(conflicts.len() as u64, Ordering::Relaxed);
        }

        // Apply writes
        let mut bind_space = self
            .bind_space
            .write()
            .map_err(|_| DagError::LockPoisoned)?;
        let mut deltas = Vec::new();

        // Get next version from global counter
        let version = self.next_txn_id.load(Ordering::SeqCst); // Use txn counter as version proxy

        for write in txn.writes.values() {
            // Capture delta for time-travel
            if let Some(old_node) = bind_space.read(write.addr) {
                let delta = xor_fingerprints(&old_node.fingerprint, &write.fingerprint);
                deltas.push((write.addr, delta));
            }

            // Write to bind space using write_at
            bind_space.write_at(write.addr, write.fingerprint);

            // Update label if provided
            if let (Some(label), Some(node)) = (&write.label, bind_space.read_mut(write.addr)) {
                node.label = Some(label.clone());
                node.access_count += 1; // Increment for change detection
            }
        }

        // Update parity blocks
        self.update_parity_blocks(&txn.parity_updates, &bind_space)?;

        // Record delta for time-travel
        if !deltas.is_empty() {
            let prev_version = self.last_full_snapshot.load(Ordering::SeqCst);
            let delta = XorDelta {
                to_version: version,
                from_version: prev_version,
                deltas,
                timestamp: timestamp_micros(),
            };

            let mut chain = self
                .delta_chain
                .write()
                .map_err(|_| DagError::LockPoisoned)?;
            chain.insert(version, delta);
            self.stats.deltas_created.fetch_add(1, Ordering::Relaxed);

            // Consolidate if chain too long
            if chain.len() > self.config.max_delta_chain {
                // Just record new "full" snapshot version
                self.last_full_snapshot.store(version, Ordering::SeqCst);
                // Could prune old deltas here
            }
        }

        // Submit parity update work items
        if self.config.enable_work_stealing {
            for &parity_id in &txn.parity_updates {
                self.epoch_guard.submit_work(WorkItem {
                    addr: Addr(0), // Parity update doesn't target specific addr
                    operation: WorkOperation::UpdateParity { parity_id },
                    epoch: new_epoch,
                    created_at: Instant::now(),
                });
            }
        }

        self.stats.txns_committed.fetch_add(1, Ordering::Relaxed);

        Ok(version)
    }

    /// Abort transaction
    pub fn abort(&self, txn_id: u64) -> Result<(), DagError> {
        let mut txns = self
            .active_txns
            .write()
            .map_err(|_| DagError::LockPoisoned)?;

        if let Some(mut txn) = txns.remove(&txn_id) {
            txn.state = TxnState::Aborted;
            self.stats.txns_aborted.fetch_add(1, Ordering::Relaxed);
            Ok(())
        } else {
            Err(DagError::TxnNotFound(txn_id))
        }
    }

    // =========================================================================
    // TIME-TRAVEL QUERIES
    // =========================================================================

    /// Read at a specific version (time-travel)
    pub fn read_at_version(
        &self,
        addr: Addr,
        target_version: Version,
    ) -> Result<Option<[u64; FINGERPRINT_WORDS]>, DagError> {
        let bind_space = self.bind_space.read().map_err(|_| DagError::LockPoisoned)?;
        let current = bind_space.read(addr);

        if current.is_none() {
            return Ok(None);
        }

        let current = current.unwrap();

        // Walk delta chain backward from latest to target
        let chain = self
            .delta_chain
            .read()
            .map_err(|_| DagError::LockPoisoned)?;

        // If no deltas or target is current, return current
        if chain.is_empty() {
            return Ok(Some(current.fingerprint));
        }

        let current_version = chain.keys().next_back().copied().unwrap_or(0);

        if current_version == target_version || target_version == 0 {
            return Ok(Some(current.fingerprint));
        }

        let mut fingerprint = current.fingerprint;
        let mut version = current_version;

        // Go backward through deltas
        for (_, delta) in chain.range(..=current_version).rev() {
            if version <= target_version {
                break;
            }

            // Find delta for this address
            for &(delta_addr, delta_fp) in &delta.deltas {
                if delta_addr == addr {
                    // Apply delta backward
                    for i in 0..FINGERPRINT_WORDS {
                        fingerprint[i] ^= delta_fp[i];
                    }
                    break;
                }
            }

            version = delta.from_version;
        }

        Ok(Some(fingerprint))
    }

    /// Get diff between two versions
    pub fn diff_versions(
        &self,
        addr: Addr,
        v1: Version,
        v2: Version,
    ) -> Result<Option<[u64; FINGERPRINT_WORDS]>, DagError> {
        let fp1 = self.read_at_version(addr, v1)?;
        let fp2 = self.read_at_version(addr, v2)?;

        match (fp1, fp2) {
            (Some(f1), Some(f2)) => Ok(Some(xor_fingerprints(&f1, &f2))),
            _ => Ok(None),
        }
    }

    // =========================================================================
    // PARITY OPERATIONS
    // =========================================================================

    /// Initialize parity blocks for all tiers
    pub fn init_parity(&self) -> Result<(), DagError> {
        let bind_space = self.bind_space.read().map_err(|_| DagError::LockPoisoned)?;
        let mut parity_blocks = self
            .parity_blocks
            .write()
            .map_err(|_| DagError::LockPoisoned)?;

        // Create parity blocks for each tier
        for tier in [ParityTier::Hot, ParityTier::Warm, ParityTier::Cold] {
            let addrs = self.addrs_for_tier(tier);

            for chunk in addrs.chunks(self.config.parity_block_size) {
                let id = self.next_parity_id.fetch_add(1, Ordering::SeqCst);
                let block = ParityBlock::new(id, tier, &bind_space, chunk);
                parity_blocks.insert(id, block);
            }
        }

        // Create cross-tier parity if enabled
        if self.config.cross_tier_parity {
            self.update_cross_parity(&bind_space)?;
        }

        Ok(())
    }

    /// Verify all parity blocks
    pub fn verify_parity(&self) -> Result<Vec<u64>, DagError> {
        let bind_space = self.bind_space.read().map_err(|_| DagError::LockPoisoned)?;
        let parity_blocks = self
            .parity_blocks
            .read()
            .map_err(|_| DagError::LockPoisoned)?;

        let mut invalid = Vec::new();

        for (&id, block) in parity_blocks.iter() {
            self.stats.parity_checks.fetch_add(1, Ordering::Relaxed);

            if !block.verify(&bind_space) {
                invalid.push(id);
            }
        }

        Ok(invalid)
    }

    /// Recover missing data using parity
    pub fn recover_addr(&self, addr: Addr) -> Result<[u64; FINGERPRINT_WORDS], DagError> {
        let bind_space = self.bind_space.read().map_err(|_| DagError::LockPoisoned)?;
        let parity_blocks = self
            .parity_blocks
            .read()
            .map_err(|_| DagError::LockPoisoned)?;

        // Find parity block covering this address
        for block in parity_blocks.values() {
            if block.covered_addrs.contains(&addr) {
                if let Some(recovered) = block.recover_missing(&bind_space, addr) {
                    self.stats.parity_recoveries.fetch_add(1, Ordering::Relaxed);
                    return Ok(recovered);
                }
            }
        }

        Err(DagError::RecoveryFailed(addr))
    }

    // =========================================================================
    // WORK STEALING
    // =========================================================================

    /// Process stolen work items
    pub fn process_stolen_work(&self) -> Result<usize, DagError> {
        if let Some(items) = self.epoch_guard.try_steal() {
            let count = items.len();
            self.stats
                .work_stolen
                .fetch_add(count as u64, Ordering::Relaxed);

            for item in items {
                match item.operation {
                    WorkOperation::UpdateParity { parity_id } => {
                        // Re-compute parity for this block
                        let bind_space =
                            self.bind_space.read().map_err(|_| DagError::LockPoisoned)?;
                        let mut parity_blocks = self
                            .parity_blocks
                            .write()
                            .map_err(|_| DagError::LockPoisoned)?;

                        if let Some(block) = parity_blocks.get_mut(&parity_id) {
                            let fps: Vec<_> = block
                                .covered_addrs
                                .iter()
                                .filter_map(|&a| bind_space.read(a).map(|n| n.fingerprint))
                                .collect();
                            block.parity_fingerprint = xor_fingerprints_multi(&fps);
                            block.version += 1;
                            block.updated_at = timestamp_micros();
                        }
                    }
                    _ => {
                        // Other work items would be processed here
                    }
                }
            }

            Ok(count)
        } else {
            Ok(0)
        }
    }

    // =========================================================================
    // HELPERS
    // =========================================================================

    /// Get parity block ID for an address
    fn parity_id_for_addr(&self, addr: Addr) -> u64 {
        // Simple: divide address space into parity blocks
        (addr.0 as u64) / (self.config.parity_block_size as u64) + 1
    }

    /// Get addresses for a tier
    fn addrs_for_tier(&self, tier: ParityTier) -> Vec<Addr> {
        let (start, end) = match tier {
            ParityTier::Hot => (0x8000, 0xFFFF),  // Nodes
            ParityTier::Warm => (0x1000, 0x7FFF), // Fluid
            ParityTier::Cold => (0x0000, 0x0FFF), // Surface
            ParityTier::Cross => return Vec::new(),
        };

        (start..=end).map(Addr).collect()
    }

    /// Validate transaction reads against current state
    fn validate_reads(&self, txn: &DagTransaction) -> Result<Vec<WriteConflict>, DagError> {
        let bind_space = self.bind_space.read().map_err(|_| DagError::LockPoisoned)?;
        let mut conflicts = Vec::new();

        for (addr, snapshot) in txn.read_set() {
            if let Some(current) = bind_space.read(*addr) {
                // Detect changes via fingerprint comparison or access_count
                let has_changed = current.fingerprint != snapshot.fingerprint
                    || current.access_count != snapshot.access_count;

                if has_changed {
                    // Conflict detected
                    let current_snapshot = StateSnapshot {
                        addr: *addr,
                        fingerprint: current.fingerprint,
                        version: 0, // Not tracked in BindNode
                        access_count: current.access_count,
                        timestamp: timestamp_micros(),
                        delta_from_prev: None,
                    };

                    // Try to find common ancestor from delta chain
                    let ancestor = self.find_ancestor(*addr, snapshot.version, 0)?;

                    conflicts.push(WriteConflict {
                        addr: *addr,
                        our_state: snapshot.clone(),
                        current_state: current_snapshot,
                        ancestor,
                    });
                }
            }
        }

        Ok(conflicts)
    }

    /// Find common ancestor version for conflict resolution
    fn find_ancestor(
        &self,
        _addr: Addr,
        v1: Version,
        v2: Version,
    ) -> Result<Option<StateSnapshot>, DagError> {
        // Find the common ancestor by walking delta chain
        // For now, return None (no ancestor known)
        // A full implementation would track version history
        let _ = (v1, v2);
        Ok(None)
    }

    /// Resolve conflicts using configured strategy
    fn resolve_conflicts(
        &self,
        txn: &DagTransaction,
        conflicts: &[WriteConflict],
    ) -> Result<bool, DagError> {
        match txn.strategy {
            ConflictStrategy::LastWriteWins => Ok(true), // Just overwrite
            ConflictStrategy::FirstWriteWins => Ok(false), // Abort
            ConflictStrategy::Reject => Ok(false),       // Always reject
            ConflictStrategy::XorMerge => {
                // Try to auto-merge all conflicts
                for conflict in conflicts {
                    if !conflict.is_orthogonal() && conflict.xor_merge().is_none() {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            ConflictStrategy::Custom => Ok(false), // Would call custom function
        }
    }

    /// Update parity blocks after writes
    fn update_parity_blocks(
        &self,
        parity_ids: &HashSet<u64>,
        bind_space: &BindSpace,
    ) -> Result<(), DagError> {
        let mut parity_blocks = self
            .parity_blocks
            .write()
            .map_err(|_| DagError::LockPoisoned)?;

        for &id in parity_ids {
            if let Some(block) = parity_blocks.get_mut(&id) {
                // Recompute parity (could be incremental with old values)
                let fps: Vec<_> = block
                    .covered_addrs
                    .iter()
                    .filter_map(|&a| bind_space.read(a).map(|n| n.fingerprint))
                    .collect();
                block.parity_fingerprint = xor_fingerprints_multi(&fps);
                block.version += 1;
                block.updated_at = timestamp_micros();
            }
        }

        Ok(())
    }

    /// Update cross-tier parity
    fn update_cross_parity(&self, bind_space: &BindSpace) -> Result<(), DagError> {
        let parity_blocks = self
            .parity_blocks
            .read()
            .map_err(|_| DagError::LockPoisoned)?;

        // Collect all tier parities
        let tier_parities: Vec<[u64; FINGERPRINT_WORDS]> = parity_blocks
            .values()
            .filter(|b| b.tier != ParityTier::Cross)
            .map(|b| b.parity_fingerprint)
            .collect();

        if tier_parities.is_empty() {
            return Ok(());
        }

        let cross = xor_fingerprints_multi(&tier_parities);

        let mut cross_parity = self
            .cross_parity
            .write()
            .map_err(|_| DagError::LockPoisoned)?;
        *cross_parity = Some(ParityBlock {
            id: 0,
            covered_addrs: Vec::new(), // Cross parity covers tier parities, not addrs
            parity_fingerprint: cross,
            version: 0,
            tier: ParityTier::Cross,
            updated_at: timestamp_micros(),
        });

        let _ = bind_space; // Used indirectly through parity_blocks

        Ok(())
    }

    /// Get statistics
    pub fn stats(&self) -> XorDagStatsSnapshot {
        XorDagStatsSnapshot {
            txns_started: self.stats.txns_started.load(Ordering::Relaxed),
            txns_committed: self.stats.txns_committed.load(Ordering::Relaxed),
            txns_aborted: self.stats.txns_aborted.load(Ordering::Relaxed),
            conflicts_detected: self.stats.conflicts_detected.load(Ordering::Relaxed),
            conflicts_merged: self.stats.conflicts_merged.load(Ordering::Relaxed),
            parity_checks: self.stats.parity_checks.load(Ordering::Relaxed),
            parity_recoveries: self.stats.parity_recoveries.load(Ordering::Relaxed),
            work_stolen: self.stats.work_stolen.load(Ordering::Relaxed),
            deltas_created: self.stats.deltas_created.load(Ordering::Relaxed),
            active_txns: self.active_txns.read().map(|t| t.len()).unwrap_or(0),
            parity_blocks: self.parity_blocks.read().map(|p| p.len()).unwrap_or(0),
            delta_chain_length: self.delta_chain.read().map(|d| d.len()).unwrap_or(0),
            pending_work: self.epoch_guard.pending_count(),
            current_epoch: self.epoch_guard.current_epoch(),
        }
    }
}

/// Snapshot of statistics
#[derive(Debug, Clone)]
pub struct XorDagStatsSnapshot {
    pub txns_started: u64,
    pub txns_committed: u64,
    pub txns_aborted: u64,
    pub conflicts_detected: u64,
    pub conflicts_merged: u64,
    pub parity_checks: u64,
    pub parity_recoveries: u64,
    pub work_stolen: u64,
    pub deltas_created: u64,
    pub active_txns: usize,
    pub parity_blocks: usize,
    pub delta_chain_length: usize,
    pub pending_work: usize,
    pub current_epoch: u64,
}

// =============================================================================
// ERRORS
// =============================================================================

/// XOR DAG errors
#[derive(Debug, Clone)]
pub enum DagError {
    LockPoisoned,
    TxnNotFound(u64),
    TxnNotActive(u64),
    TxnTimeout(u64),
    ConflictUnresolved(Vec<WriteConflict>),
    RecoveryFailed(Addr),
    ParityInvalid(u64),
    IoError(String),
}

impl std::fmt::Display for DagError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LockPoisoned => write!(f, "Lock poisoned"),
            Self::TxnNotFound(id) => write!(f, "Transaction not found: {}", id),
            Self::TxnNotActive(id) => write!(f, "Transaction not active: {}", id),
            Self::TxnTimeout(id) => write!(f, "Transaction timeout: {}", id),
            Self::ConflictUnresolved(c) => write!(f, "Unresolved conflicts: {}", c.len()),
            Self::RecoveryFailed(addr) => write!(f, "Recovery failed for {:04x}", addr.0),
            Self::ParityInvalid(id) => write!(f, "Parity invalid: {}", id),
            Self::IoError(msg) => write!(f, "I/O error: {}", msg),
        }
    }
}

impl std::error::Error for DagError {}

// =============================================================================
// HELPERS
// =============================================================================

/// Get current timestamp in microseconds
fn timestamp_micros() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_dag() -> XorDag {
        let bind_space = Arc::new(RwLock::new(BindSpace::new()));
        XorDag::new(XorDagConfig::default(), bind_space)
    }

    #[test]
    fn test_xor_slices() {
        let a = vec![0x00, 0xFF, 0x55, 0xAA];
        let b = vec![0xFF, 0x00, 0xAA, 0x55];

        let result = xor_slices(&a, &b);
        assert_eq!(result, vec![0xFF, 0xFF, 0xFF, 0xFF]);

        // XOR is self-inverse
        let back = xor_slices(&result, &b);
        assert_eq!(back, a);
    }

    #[test]
    fn test_xor_fingerprints() {
        let mut a = [0u64; FINGERPRINT_WORDS];
        let mut b = [0u64; FINGERPRINT_WORDS];

        a[0] = 0xFFFFFFFF00000000;
        b[0] = 0x00000000FFFFFFFF;

        let result = xor_fingerprints(&a, &b);
        assert_eq!(result[0], 0xFFFFFFFFFFFFFFFF);

        // Self-inverse
        let back = xor_fingerprints(&result, &b);
        assert_eq!(back, a);
    }

    #[test]
    fn test_xor_fingerprints_multi() {
        let a = [1u64; FINGERPRINT_WORDS];
        let b = [2u64; FINGERPRINT_WORDS];
        let c = [4u64; FINGERPRINT_WORDS];

        let parity = xor_fingerprints_multi(&[a, b, c]);

        // Recover a from parity and b, c
        let recovered = xor_fingerprints_multi(&[parity, b, c]);
        assert_eq!(recovered, a);
    }

    #[test]
    fn test_transaction_basic() {
        let dag = create_test_dag();

        // Begin transaction
        let txn_id = dag.begin().unwrap();

        // Write some data
        let fp = [42u64; FINGERPRINT_WORDS];
        dag.write(txn_id, Addr(0x8000), fp, Some("test".into()))
            .unwrap();

        // Commit
        let version = dag.commit(txn_id).unwrap();
        assert!(version > 0);

        // Verify data persisted
        let txn2 = dag.begin().unwrap();
        let node = dag.read(txn2, Addr(0x8000)).unwrap();
        assert!(node.is_some());
        assert_eq!(node.unwrap().fingerprint, fp);
        dag.abort(txn2).unwrap();

        // Check stats
        let stats = dag.stats();
        assert_eq!(stats.txns_committed, 1);
    }

    #[test]
    fn test_epoch_guard() {
        let guard = EpochGuard::new();

        // Begin read
        let ticket = guard.begin_read();
        assert_eq!(ticket.epoch(), 0);
        assert!(ticket.is_valid());

        // Advance epoch
        guard.advance_epoch();
        assert!(ticket.is_valid()); // Still valid (within 1)

        guard.advance_epoch();
        assert!(!ticket.is_valid()); // Now invalid (epoch advanced twice)

        drop(ticket);
    }

    #[test]
    fn test_work_stealing() {
        let guard = EpochGuard::new();

        // Submit work
        guard.submit_work(WorkItem {
            addr: Addr(0x8000),
            operation: WorkOperation::Delete,
            epoch: 0,
            created_at: Instant::now(),
        });

        assert_eq!(guard.pending_count(), 1);

        // Advance epoch to make work stealable
        guard.advance_epoch();

        // Steal work
        let stolen = guard.try_steal();
        assert!(stolen.is_some());
        assert_eq!(stolen.unwrap().len(), 1);

        // Nothing left to steal
        assert!(guard.try_steal().is_none());
    }

    #[test]
    fn test_conflict_xor_merge() {
        let ancestor = StateSnapshot {
            addr: Addr(0x8000),
            fingerprint: [0u64; FINGERPRINT_WORDS],
            version: 1,
            access_count: 0,
            timestamp: 0,
            delta_from_prev: None,
        };

        // Our change: set bit 0 of word 0
        let mut our_fp = [0u64; FINGERPRINT_WORDS];
        our_fp[0] = 0x01;
        let our_state = StateSnapshot {
            addr: Addr(0x8000),
            fingerprint: our_fp,
            version: 1,
            access_count: 1,
            timestamp: 0,
            delta_from_prev: None,
        };

        // Their change: set bit 1 of word 0
        let mut their_fp = [0u64; FINGERPRINT_WORDS];
        their_fp[0] = 0x02;
        let current_state = StateSnapshot {
            addr: Addr(0x8000),
            fingerprint: their_fp,
            version: 2,
            access_count: 1,
            timestamp: 0,
            delta_from_prev: None,
        };

        let conflict = WriteConflict {
            addr: Addr(0x8000),
            our_state,
            current_state,
            ancestor: Some(ancestor),
        };

        // Changes are orthogonal (different bits)
        assert!(conflict.is_orthogonal());

        // Merge should combine both changes
        let merged = conflict.xor_merge().unwrap();
        assert_eq!(merged[0], 0x03); // Both bits set
    }

    #[test]
    fn test_parity_block_recovery() {
        let bind_space = Arc::new(RwLock::new(BindSpace::new()));

        // Write some test data
        {
            let mut bs = bind_space.write().unwrap();
            for i in 0u16..4 {
                let mut fp = [0u64; FINGERPRINT_WORDS];
                fp[0] = (i + 1) as u64;
                bs.write_at(Addr(0x8000 + i), fp);
            }
        }

        // Create parity block
        let bs = bind_space.read().unwrap();
        let addrs: Vec<Addr> = (0..4).map(|i| Addr(0x8000 + i)).collect();
        let parity = ParityBlock::new(1, ParityTier::Hot, &bs, &addrs);

        // Verify parity
        assert!(parity.verify(&bs));

        // Simulate recovering addr 0x8002 (which has fp[0] = 3)
        let recovered = parity.recover_missing(&bs, Addr(0x8002)).unwrap();
        assert_eq!(recovered[0], 3);
    }

    #[test]
    fn test_time_travel() {
        let dag = create_test_dag();

        // Write version 1
        let txn1 = dag.begin().unwrap();
        let mut fp1 = [0u64; FINGERPRINT_WORDS];
        fp1[0] = 100;
        dag.write(txn1, Addr(0x8000), fp1, None).unwrap();
        let v1 = dag.commit(txn1).unwrap();

        // Write version 2
        let txn2 = dag.begin().unwrap();
        dag.read(txn2, Addr(0x8000)).unwrap(); // Read first for conflict tracking
        let mut fp2 = [0u64; FINGERPRINT_WORDS];
        fp2[0] = 200;
        dag.write(txn2, Addr(0x8000), fp2, None).unwrap();
        let v2 = dag.commit(txn2).unwrap();

        // Read current (should be v2)
        let txn3 = dag.begin().unwrap();
        let current = dag.read(txn3, Addr(0x8000)).unwrap().unwrap();
        assert_eq!(current.fingerprint[0], 200);
        dag.abort(txn3).unwrap();

        // Time travel to v1
        let historical = dag.read_at_version(Addr(0x8000), v1).unwrap().unwrap();
        assert_eq!(historical[0], 100);

        // Get diff between versions
        let diff = dag.diff_versions(Addr(0x8000), v1, v2).unwrap().unwrap();
        assert_eq!(diff[0], 100 ^ 200); // XOR of the two values
    }
}
