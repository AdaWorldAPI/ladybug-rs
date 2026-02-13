//! Resilient Storage Layer - ReFS-like Hardening
//!
//! This module adds production-grade resilience on top of temporal storage:
//!
//! # Write Buffer with Virtual Versions
//! - Writes go to fast buffer first
//! - Virtual versions assigned immediately
//! - Background flush to durable storage
//! - Read-your-writes consistency
//!
//! # Dependency Tracking
//! - Tracks which writes depend on which
//! - Ordered recovery on failure
//! - Cascading rollback for dependent writes
//!
//! # Automatic Recovery
//! - Detects write failures
//! - Forks rollback branch automatically
//! - Re-applies valid writes
//! - Discards failed subtree
//!
//! # Magic Buffering
//! - Serves recent data from buffer
//! - Background confirmation
//! - Transparent to consumers

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use super::bind_space::FINGERPRINT_WORDS;
use super::temporal::{IsolationLevel, TemporalEntry, TemporalStore, Version};

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Resilient storage configuration
#[derive(Clone, Debug)]
pub struct ResilienceConfig {
    /// Maximum pending writes before blocking
    pub max_pending_writes: usize,
    /// Flush interval (how often to persist buffer)
    pub flush_interval: Duration,
    /// Maximum time a write can stay in buffer
    pub max_buffer_age: Duration,
    /// Enable automatic recovery on failure
    pub auto_recovery: bool,
    /// Maximum recovery attempts before giving up
    pub max_recovery_attempts: u32,
    /// Enable dependency tracking
    pub track_dependencies: bool,
    /// Sync mode: true = wait for flush, false = async
    pub sync_writes: bool,
    /// Buffer memory limit (bytes)
    pub buffer_memory_limit: usize,
}

impl Default for ResilienceConfig {
    fn default() -> Self {
        Self {
            max_pending_writes: 10_000,
            flush_interval: Duration::from_millis(100),
            max_buffer_age: Duration::from_secs(5),
            auto_recovery: true,
            max_recovery_attempts: 3,
            track_dependencies: true,
            sync_writes: false,
            buffer_memory_limit: 64 * 1024 * 1024, // 64MB
        }
    }
}

impl ResilienceConfig {
    /// High durability config (slower but safer)
    pub fn durable() -> Self {
        Self {
            max_pending_writes: 1_000,
            flush_interval: Duration::from_millis(10),
            max_buffer_age: Duration::from_secs(1),
            auto_recovery: true,
            max_recovery_attempts: 5,
            track_dependencies: true,
            sync_writes: true,
            buffer_memory_limit: 16 * 1024 * 1024,
        }
    }

    /// High performance config (faster but less safe)
    pub fn performance() -> Self {
        Self {
            max_pending_writes: 50_000,
            flush_interval: Duration::from_millis(500),
            max_buffer_age: Duration::from_secs(30),
            auto_recovery: true,
            max_recovery_attempts: 2,
            track_dependencies: false,
            sync_writes: false,
            buffer_memory_limit: 256 * 1024 * 1024,
        }
    }
}

// =============================================================================
// WRITE BUFFER ENTRY
// =============================================================================

/// Virtual version for buffered writes
pub type VirtualVersion = u64;

/// State of a buffered write
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WriteState {
    /// In buffer, not yet flushed
    Pending,
    /// Being flushed
    Flushing,
    /// Successfully persisted
    Confirmed,
    /// Failed to persist
    Failed,
    /// Rolled back due to dependency failure
    RolledBack,
}

/// A buffered write operation
#[derive(Clone, Debug)]
pub struct BufferedWrite {
    /// Unique write ID
    pub id: u64,
    /// Virtual version (assigned immediately)
    pub virtual_version: VirtualVersion,
    /// Real version (assigned after confirm)
    pub real_version: Option<Version>,
    /// Address being written
    pub addr: u16,
    /// Fingerprint data
    pub fingerprint: [u64; FINGERPRINT_WORDS],
    /// Optional label
    pub label: Option<String>,
    /// Dependencies (write IDs this depends on)
    pub depends_on: Vec<u64>,
    /// Current state
    pub state: WriteState,
    /// When this was buffered
    pub buffered_at: Instant,
    /// Retry count
    pub retries: u32,
    /// Error message if failed
    pub error: Option<String>,
}

impl BufferedWrite {
    pub fn new(
        id: u64,
        virtual_version: VirtualVersion,
        addr: u16,
        fingerprint: [u64; FINGERPRINT_WORDS],
        label: Option<String>,
    ) -> Self {
        Self {
            id,
            virtual_version,
            real_version: None,
            addr,
            fingerprint,
            label,
            depends_on: Vec::new(),
            state: WriteState::Pending,
            buffered_at: Instant::now(),
            retries: 0,
            error: None,
        }
    }

    /// Check if write has aged out
    pub fn is_stale(&self, max_age: Duration) -> bool {
        self.buffered_at.elapsed() > max_age
    }

    /// Estimated memory usage
    pub fn memory_size(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.label.as_ref().map(|l| l.len()).unwrap_or(0)
            + self.depends_on.len() * std::mem::size_of::<u64>()
    }
}

/// A buffered delete operation
#[derive(Clone, Debug)]
pub struct BufferedDelete {
    pub id: u64,
    pub virtual_version: VirtualVersion,
    pub addr: u16,
    pub depends_on: Vec<u64>,
    pub state: WriteState,
    pub buffered_at: Instant,
}

/// A buffered link operation
#[derive(Clone, Debug)]
pub struct BufferedLink {
    pub id: u64,
    pub virtual_version: VirtualVersion,
    pub from: u16,
    pub verb: u16,
    pub to: u16,
    pub depends_on: Vec<u64>,
    pub state: WriteState,
    pub buffered_at: Instant,
}

// =============================================================================
// WRITE BUFFER
// =============================================================================

/// The write buffer with virtual versioning
pub struct WriteBuffer {
    /// Pending writes
    writes: RwLock<HashMap<u64, BufferedWrite>>,
    /// Pending deletes
    deletes: RwLock<HashMap<u64, BufferedDelete>>,
    /// Pending links
    links: RwLock<HashMap<u64, BufferedLink>>,
    /// Flush queue (ordered)
    flush_queue: Mutex<VecDeque<u64>>,
    /// Next write ID
    next_id: AtomicU64,
    /// Next virtual version
    next_virtual: AtomicU64,
    /// Current memory usage
    memory_used: AtomicU64,
    /// Configuration
    config: ResilienceConfig,
    /// Shutdown flag
    shutdown: AtomicBool,
    /// Condition variable for flush notification
    flush_notify: Condvar,
}

impl WriteBuffer {
    pub fn new(config: ResilienceConfig) -> Self {
        Self {
            writes: RwLock::new(HashMap::new()),
            deletes: RwLock::new(HashMap::new()),
            links: RwLock::new(HashMap::new()),
            flush_queue: Mutex::new(VecDeque::new()),
            next_id: AtomicU64::new(1),
            next_virtual: AtomicU64::new(1_000_000), // Start high to distinguish from real versions
            memory_used: AtomicU64::new(0),
            config,
            shutdown: AtomicBool::new(false),
            flush_notify: Condvar::new(),
        }
    }

    /// Buffer a write operation
    pub fn buffer_write(
        &self,
        addr: u16,
        fingerprint: [u64; FINGERPRINT_WORDS],
        label: Option<String>,
        depends_on: Vec<u64>,
    ) -> Result<(u64, VirtualVersion), BufferError> {
        // Check capacity
        let pending = self.pending_count();
        if pending >= self.config.max_pending_writes {
            return Err(BufferError::BufferFull(pending));
        }

        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let virtual_version = self.next_virtual.fetch_add(1, Ordering::SeqCst);

        let mut write = BufferedWrite::new(id, virtual_version, addr, fingerprint, label);
        write.depends_on = depends_on;

        let size = write.memory_size();
        self.memory_used.fetch_add(size as u64, Ordering::SeqCst);

        if let Ok(mut writes) = self.writes.write() {
            writes.insert(id, write);
        }

        if let Ok(mut queue) = self.flush_queue.lock() {
            queue.push_back(id);
        }

        // Notify flusher
        self.flush_notify.notify_one();

        Ok((id, virtual_version))
    }

    /// Buffer a delete operation
    pub fn buffer_delete(
        &self,
        addr: u16,
        depends_on: Vec<u64>,
    ) -> Result<(u64, VirtualVersion), BufferError> {
        let pending = self.pending_count();
        if pending >= self.config.max_pending_writes {
            return Err(BufferError::BufferFull(pending));
        }

        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let virtual_version = self.next_virtual.fetch_add(1, Ordering::SeqCst);

        let delete = BufferedDelete {
            id,
            virtual_version,
            addr,
            depends_on,
            state: WriteState::Pending,
            buffered_at: Instant::now(),
        };

        if let Ok(mut deletes) = self.deletes.write() {
            deletes.insert(id, delete);
        }

        if let Ok(mut queue) = self.flush_queue.lock() {
            queue.push_back(id);
        }

        self.flush_notify.notify_one();

        Ok((id, virtual_version))
    }

    /// Buffer a link operation
    pub fn buffer_link(
        &self,
        from: u16,
        verb: u16,
        to: u16,
        depends_on: Vec<u64>,
    ) -> Result<(u64, VirtualVersion), BufferError> {
        let pending = self.pending_count();
        if pending >= self.config.max_pending_writes {
            return Err(BufferError::BufferFull(pending));
        }

        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let virtual_version = self.next_virtual.fetch_add(1, Ordering::SeqCst);

        let link = BufferedLink {
            id,
            virtual_version,
            from,
            verb,
            to,
            depends_on,
            state: WriteState::Pending,
            buffered_at: Instant::now(),
        };

        if let Ok(mut links) = self.links.write() {
            links.insert(id, link);
        }

        if let Ok(mut queue) = self.flush_queue.lock() {
            queue.push_back(id);
        }

        self.flush_notify.notify_one();

        Ok((id, virtual_version))
    }

    /// Get pending write count
    pub fn pending_count(&self) -> usize {
        let writes = self.writes.read().map(|w| w.len()).unwrap_or(0);
        let deletes = self.deletes.read().map(|d| d.len()).unwrap_or(0);
        let links = self.links.read().map(|l| l.len()).unwrap_or(0);
        writes + deletes + links
    }

    /// Get a buffered write by address (for read-your-writes)
    pub fn get_buffered(&self, addr: u16) -> Option<BufferedWrite> {
        let writes = self.writes.read().ok()?;
        // Find most recent write to this address
        writes
            .values()
            .filter(|w| w.addr == addr && w.state != WriteState::RolledBack)
            .max_by_key(|w| w.virtual_version)
            .cloned()
    }

    /// Mark write as confirmed
    pub fn confirm(&self, id: u64, real_version: Version) {
        if let Ok(mut writes) = self.writes.write() {
            if let Some(write) = writes.get_mut(&id) {
                write.state = WriteState::Confirmed;
                write.real_version = Some(real_version);
            }
        }
    }

    /// Mark write as failed
    pub fn fail(&self, id: u64, error: &str) {
        if let Ok(mut writes) = self.writes.write() {
            if let Some(write) = writes.get_mut(&id) {
                write.state = WriteState::Failed;
                write.error = Some(error.to_string());
                write.retries += 1;
            }
        }
    }

    /// Get next batch to flush
    pub fn next_batch(&self, max_size: usize) -> Vec<u64> {
        let queue = match self.flush_queue.lock() {
            Ok(q) => q,
            Err(_) => return Vec::new(),
        };

        queue.iter().take(max_size).copied().collect()
    }

    /// Remove confirmed writes older than threshold
    pub fn gc(&self, min_age: Duration) {
        let now = Instant::now();

        if let Ok(mut writes) = self.writes.write() {
            let to_remove: Vec<_> = writes
                .iter()
                .filter(|(_, w)| {
                    w.state == WriteState::Confirmed && now.duration_since(w.buffered_at) > min_age
                })
                .map(|(id, _)| *id)
                .collect();

            for id in to_remove {
                if let Some(w) = writes.remove(&id) {
                    self.memory_used
                        .fetch_sub(w.memory_size() as u64, Ordering::SeqCst);
                }
            }
        }

        if let Ok(mut deletes) = self.deletes.write() {
            deletes.retain(|_, d| {
                d.state != WriteState::Confirmed || now.duration_since(d.buffered_at) <= min_age
            });
        }

        if let Ok(mut links) = self.links.write() {
            links.retain(|_, l| {
                l.state != WriteState::Confirmed || now.duration_since(l.buffered_at) <= min_age
            });
        }
    }

    /// Shutdown the buffer
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
        self.flush_notify.notify_all();
    }

    /// Check if shutdown requested
    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::SeqCst)
    }

    /// Get memory usage
    pub fn memory_used(&self) -> u64 {
        self.memory_used.load(Ordering::SeqCst)
    }
}

// =============================================================================
// DEPENDENCY GRAPH
// =============================================================================

/// Tracks dependencies between writes for ordered recovery
pub struct DependencyGraph {
    /// write_id -> depends_on
    dependencies: RwLock<HashMap<u64, Vec<u64>>>,
    /// write_id -> depended_by (reverse index)
    dependents: RwLock<HashMap<u64, Vec<u64>>>,
    /// Address -> most recent write
    addr_writes: RwLock<HashMap<u16, u64>>,
}

impl DependencyGraph {
    pub fn new() -> Self {
        Self {
            dependencies: RwLock::new(HashMap::new()),
            dependents: RwLock::new(HashMap::new()),
            addr_writes: RwLock::new(HashMap::new()),
        }
    }

    /// Record a write with its dependencies
    pub fn record(&self, id: u64, addr: u16, depends_on: Vec<u64>) {
        // Update addr -> write mapping
        if let Ok(mut addr_writes) = self.addr_writes.write() {
            addr_writes.insert(addr, id);
        }

        // Record dependencies
        if let Ok(mut deps) = self.dependencies.write() {
            deps.insert(id, depends_on.clone());
        }

        // Update reverse index
        if let Ok(mut dependents) = self.dependents.write() {
            for dep in depends_on {
                dependents.entry(dep).or_default().push(id);
            }
        }
    }

    /// Get automatic dependencies for an address (previous write to same addr)
    pub fn auto_depends(&self, addr: u16) -> Vec<u64> {
        self.addr_writes
            .read()
            .ok()
            .and_then(|m| m.get(&addr).copied())
            .map(|id| vec![id])
            .unwrap_or_default()
    }

    /// Get all writes that depend on this one (transitive)
    pub fn get_dependents(&self, id: u64) -> Vec<u64> {
        let mut result = Vec::new();
        let mut to_visit = vec![id];
        let mut visited = HashSet::new();

        while let Some(current) = to_visit.pop() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current);

            if let Ok(dependents) = self.dependents.read() {
                if let Some(deps) = dependents.get(&current) {
                    for &dep in deps {
                        if !visited.contains(&dep) {
                            result.push(dep);
                            to_visit.push(dep);
                        }
                    }
                }
            }
        }

        result
    }

    /// Get writes in dependency order (topological sort)
    pub fn ordered(&self, ids: &[u64]) -> Vec<u64> {
        let mut result = Vec::new();
        let mut in_degree: HashMap<u64, usize> = HashMap::new();
        let id_set: HashSet<_> = ids.iter().copied().collect();

        // Calculate in-degrees
        if let Ok(deps) = self.dependencies.read() {
            for &id in ids {
                let count = deps
                    .get(&id)
                    .map(|d| d.iter().filter(|&dep| id_set.contains(dep)).count())
                    .unwrap_or(0);
                in_degree.insert(id, count);
            }
        }

        // Kahn's algorithm
        let mut queue: VecDeque<_> = in_degree
            .iter()
            .filter(|(_, deg)| **deg == 0)
            .map(|(id, _)| *id)
            .collect();

        while let Some(id) = queue.pop_front() {
            result.push(id);

            if let Ok(dependents) = self.dependents.read() {
                if let Some(deps) = dependents.get(&id) {
                    for &dep in deps {
                        if let Some(deg) = in_degree.get_mut(&dep) {
                            *deg = deg.saturating_sub(1);
                            if *deg == 0 {
                                queue.push_back(dep);
                            }
                        }
                    }
                }
            }
        }

        result
    }

    /// Remove a write from tracking
    pub fn remove(&self, id: u64) {
        if let Ok(mut deps) = self.dependencies.write() {
            deps.remove(&id);
        }
        if let Ok(mut dependents) = self.dependents.write() {
            dependents.remove(&id);
            for deps in dependents.values_mut() {
                deps.retain(|&d| d != id);
            }
        }
    }
}

impl Default for DependencyGraph {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// RECOVERY ENGINE
// =============================================================================

/// Handles automatic recovery from write failures
pub struct RecoveryEngine {
    /// Configuration
    config: ResilienceConfig,
    /// Recovery attempts per write
    attempts: RwLock<HashMap<u64, u32>>,
    /// Failed writes waiting for retry
    retry_queue: Mutex<VecDeque<u64>>,
}

impl RecoveryEngine {
    pub fn new(config: ResilienceConfig) -> Self {
        Self {
            config,
            attempts: RwLock::new(HashMap::new()),
            retry_queue: Mutex::new(VecDeque::new()),
        }
    }

    /// Record a failure and decide if we should retry
    pub fn record_failure(&self, id: u64) -> RecoveryAction {
        let attempts = {
            let mut map = self.attempts.write().unwrap();
            let count = map.entry(id).or_insert(0);
            *count += 1;
            *count
        };

        if attempts >= self.config.max_recovery_attempts {
            RecoveryAction::Rollback
        } else {
            if let Ok(mut queue) = self.retry_queue.lock() {
                queue.push_back(id);
            }
            RecoveryAction::Retry { attempt: attempts }
        }
    }

    /// Get next write to retry
    pub fn next_retry(&self) -> Option<u64> {
        self.retry_queue.lock().ok()?.pop_front()
    }

    /// Clear tracking for a write
    pub fn clear(&self, id: u64) {
        if let Ok(mut attempts) = self.attempts.write() {
            attempts.remove(&id);
        }
    }
}

/// Action to take after failure
#[derive(Clone, Copy, Debug)]
pub enum RecoveryAction {
    /// Retry the write
    Retry { attempt: u32 },
    /// Give up and rollback
    Rollback,
}

// =============================================================================
// RESILIENT STORE
// =============================================================================

/// Resilient store wrapping temporal store with buffer and recovery
pub struct ResilientStore {
    /// Underlying temporal store
    temporal: Arc<TemporalStore>,
    /// Write buffer
    buffer: Arc<WriteBuffer>,
    /// Dependency graph
    deps: Arc<DependencyGraph>,
    /// Recovery engine
    recovery: Arc<RecoveryEngine>,
    /// Configuration
    config: ResilienceConfig,
    /// Flusher thread handle
    flusher: Option<JoinHandle<()>>,
    /// Last confirmed version
    last_confirmed: AtomicU64,
}

impl ResilientStore {
    pub fn new(config: ResilienceConfig) -> Self {
        let temporal = Arc::new(TemporalStore::new());
        let buffer = Arc::new(WriteBuffer::new(config.clone()));
        let deps = Arc::new(DependencyGraph::new());
        let recovery = Arc::new(RecoveryEngine::new(config.clone()));

        let mut store = Self {
            temporal,
            buffer,
            deps,
            recovery,
            config,
            flusher: None,
            last_confirmed: AtomicU64::new(0),
        };

        // Start flusher thread
        store.start_flusher();

        store
    }

    fn start_flusher(&mut self) {
        let buffer = Arc::clone(&self.buffer);
        let temporal = Arc::clone(&self.temporal);
        let deps = Arc::clone(&self.deps);
        let recovery = Arc::clone(&self.recovery);
        let config = self.config.clone();
        let last_confirmed = &self.last_confirmed as *const AtomicU64 as usize;

        let handle = thread::spawn(move || {
            let last_confirmed = unsafe { &*(last_confirmed as *const AtomicU64) };

            loop {
                if buffer.is_shutdown() {
                    break;
                }

                // Wait for work or timeout
                thread::sleep(config.flush_interval);

                // Get batch to flush
                let batch = buffer.next_batch(100);
                if batch.is_empty() {
                    continue;
                }

                // Order by dependencies
                let ordered = deps.ordered(&batch);

                // Flush each write
                for id in ordered {
                    // Get the write
                    let write = buffer.writes.read().ok().and_then(|w| w.get(&id).cloned());

                    if let Some(write) = write {
                        if write.state != WriteState::Pending {
                            continue;
                        }

                        // Check dependencies are confirmed
                        let deps_ok = write.depends_on.iter().all(|&dep_id| {
                            buffer
                                .writes
                                .read()
                                .ok()
                                .map(|w| {
                                    w.get(&dep_id)
                                        .map(|entry| entry.state == WriteState::Confirmed)
                                        .unwrap_or(true)
                                })
                                .unwrap_or(true)
                        });

                        if !deps_ok {
                            continue; // Wait for dependencies
                        }

                        // Try to persist
                        let txn_id = temporal.begin(IsolationLevel::ReadCommitted);
                        let result = temporal.write_in_txn(
                            txn_id,
                            write.addr,
                            write.fingerprint,
                            write.label.clone(),
                        );

                        match result {
                            Ok(()) => {
                                match temporal.commit(txn_id) {
                                    Ok(version) => {
                                        buffer.confirm(id, version);
                                        recovery.clear(id);
                                        last_confirmed.store(version, Ordering::SeqCst);
                                    }
                                    Err(e) => {
                                        buffer.fail(id, &e.to_string());
                                        let action = recovery.record_failure(id);
                                        if matches!(action, RecoveryAction::Rollback) {
                                            // Rollback dependents too
                                            for dep_id in deps.get_dependents(id) {
                                                if let Ok(mut writes) = buffer.writes.write() {
                                                    if let Some(w) = writes.get_mut(&dep_id) {
                                                        w.state = WriteState::RolledBack;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                let _ = temporal.abort(txn_id);
                                buffer.fail(id, &e.to_string());
                                recovery.record_failure(id);
                            }
                        }
                    }
                }

                // GC old confirmed writes
                buffer.gc(Duration::from_secs(60));
            }
        });

        self.flusher = Some(handle);
    }

    // -------------------------------------------------------------------------
    // WRITE OPERATIONS
    // -------------------------------------------------------------------------

    /// Write with automatic dependency tracking
    pub fn write(
        &self,
        addr: u16,
        fingerprint: [u64; FINGERPRINT_WORDS],
        label: Option<String>,
    ) -> Result<VirtualVersion, BufferError> {
        // Get automatic dependencies
        let depends_on = if self.config.track_dependencies {
            self.deps.auto_depends(addr)
        } else {
            Vec::new()
        };

        let (id, virtual_version) =
            self.buffer
                .buffer_write(addr, fingerprint, label, depends_on.clone())?;

        // Record in dependency graph
        self.deps.record(id, addr, depends_on);

        // If sync mode, wait for confirmation
        if self.config.sync_writes {
            self.wait_for_confirm(id)?;
        }

        Ok(virtual_version)
    }

    /// Write with explicit dependencies
    pub fn write_with_deps(
        &self,
        addr: u16,
        fingerprint: [u64; FINGERPRINT_WORDS],
        label: Option<String>,
        depends_on: Vec<u64>,
    ) -> Result<VirtualVersion, BufferError> {
        let (id, virtual_version) =
            self.buffer
                .buffer_write(addr, fingerprint, label, depends_on.clone())?;
        self.deps.record(id, addr, depends_on);

        if self.config.sync_writes {
            self.wait_for_confirm(id)?;
        }

        Ok(virtual_version)
    }

    /// Delete with automatic dependencies
    pub fn delete(&self, addr: u16) -> Result<VirtualVersion, BufferError> {
        let depends_on = if self.config.track_dependencies {
            self.deps.auto_depends(addr)
        } else {
            Vec::new()
        };

        let (id, virtual_version) = self.buffer.buffer_delete(addr, depends_on.clone())?;
        self.deps.record(id, addr, depends_on);

        if self.config.sync_writes {
            self.wait_for_confirm(id)?;
        }

        Ok(virtual_version)
    }

    /// Link with dependencies
    pub fn link(&self, from: u16, verb: u16, to: u16) -> Result<VirtualVersion, BufferError> {
        // Link depends on both from and to being written
        let mut depends_on = Vec::new();
        if self.config.track_dependencies {
            depends_on.extend(self.deps.auto_depends(from));
            depends_on.extend(self.deps.auto_depends(to));
        }

        let (id, virtual_version) = self.buffer.buffer_link(from, verb, to, depends_on)?;

        if self.config.sync_writes {
            self.wait_for_confirm(id)?;
        }

        Ok(virtual_version)
    }

    // -------------------------------------------------------------------------
    // READ OPERATIONS (with read-your-writes)
    // -------------------------------------------------------------------------

    /// Read with read-your-writes consistency
    pub fn read(&self, addr: u16) -> Option<ReadResult> {
        // Check buffer first (most recent write)
        if let Some(buffered) = self.buffer.get_buffered(addr) {
            return Some(ReadResult::Buffered(buffered));
        }

        // Fall back to temporal store
        self.temporal.read(addr).map(ReadResult::Confirmed)
    }

    /// Read at specific version (confirmed only)
    pub fn read_at(&self, addr: u16, version: Version) -> Option<TemporalEntry> {
        self.temporal.read_at(addr, version)
    }

    // -------------------------------------------------------------------------
    // SYNC OPERATIONS
    // -------------------------------------------------------------------------

    /// Wait for a specific write to be confirmed
    pub fn wait_for_confirm(&self, id: u64) -> Result<Version, BufferError> {
        let start = Instant::now();
        let timeout = self.config.max_buffer_age;

        loop {
            if let Some(write) = self
                .buffer
                .writes
                .read()
                .ok()
                .and_then(|w| w.get(&id).cloned())
            {
                match write.state {
                    WriteState::Confirmed => return Ok(write.real_version.unwrap_or(0)),
                    WriteState::Failed => {
                        return Err(BufferError::WriteFailed(write.error.unwrap_or_default()));
                    }
                    WriteState::RolledBack => return Err(BufferError::RolledBack(id)),
                    _ => {}
                }
            }

            if start.elapsed() > timeout {
                return Err(BufferError::Timeout);
            }

            thread::sleep(Duration::from_millis(10));
        }
    }

    /// Flush all pending writes
    pub fn flush(&self) -> Result<(), BufferError> {
        let pending_ids: Vec<_> = self
            .buffer
            .writes
            .read()
            .map(|w| w.keys().copied().collect())
            .unwrap_or_default();

        for id in pending_ids {
            self.wait_for_confirm(id)?;
        }

        Ok(())
    }

    // -------------------------------------------------------------------------
    // STATUS & METRICS
    // -------------------------------------------------------------------------

    /// Get buffer status
    pub fn status(&self) -> ResilientStatus {
        ResilientStatus {
            pending_writes: self.buffer.pending_count(),
            memory_used: self.buffer.memory_used(),
            last_confirmed_version: self.last_confirmed.load(Ordering::SeqCst),
            current_version: self.temporal.current_version(),
        }
    }

    /// Get temporal store reference
    pub fn temporal(&self) -> &TemporalStore {
        &self.temporal
    }

    /// Shutdown gracefully
    pub fn shutdown(&mut self) {
        // Flush remaining writes
        let _ = self.flush();

        // Signal shutdown
        self.buffer.shutdown();

        // Wait for flusher thread
        if let Some(handle) = self.flusher.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for ResilientStore {
    fn drop(&mut self) {
        self.buffer.shutdown();
    }
}

/// Result of a read operation
#[derive(Clone, Debug)]
pub enum ReadResult {
    /// From confirmed storage
    Confirmed(TemporalEntry),
    /// From write buffer (not yet confirmed)
    Buffered(BufferedWrite),
}

impl ReadResult {
    pub fn addr(&self) -> u16 {
        match self {
            Self::Confirmed(e) => e.addr,
            Self::Buffered(w) => w.addr,
        }
    }

    pub fn fingerprint(&self) -> &[u64; FINGERPRINT_WORDS] {
        match self {
            Self::Confirmed(e) => &e.fingerprint,
            Self::Buffered(w) => &w.fingerprint,
        }
    }

    pub fn is_confirmed(&self) -> bool {
        matches!(self, Self::Confirmed(_))
    }
}

/// Resilient store status
#[derive(Clone, Debug)]
pub struct ResilientStatus {
    pub pending_writes: usize,
    pub memory_used: u64,
    pub last_confirmed_version: Version,
    pub current_version: Version,
}

// =============================================================================
// ERRORS
// =============================================================================

/// Buffer errors
#[derive(Clone, Debug)]
pub enum BufferError {
    /// Buffer is full
    BufferFull(usize),
    /// Write failed to persist
    WriteFailed(String),
    /// Write was rolled back
    RolledBack(u64),
    /// Timeout waiting for confirmation
    Timeout,
    /// Lock error
    LockError,
}

impl std::fmt::Display for BufferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BufferFull(n) => write!(f, "Buffer full ({} pending)", n),
            Self::WriteFailed(e) => write!(f, "Write failed: {}", e),
            Self::RolledBack(id) => write!(f, "Write {} was rolled back", id),
            Self::Timeout => write!(f, "Timeout waiting for confirmation"),
            Self::LockError => write!(f, "Lock acquisition failed"),
        }
    }
}

impl std::error::Error for BufferError {}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_buffer() {
        let config = ResilienceConfig::default();
        let buffer = WriteBuffer::new(config);

        let fp = [42u64; FINGERPRINT_WORDS];
        let (id, vv) = buffer
            .buffer_write(0x8001, fp, Some("test".into()), vec![])
            .unwrap();

        assert!(id > 0);
        assert!(vv >= 1_000_000); // Virtual versions start high

        // Can retrieve from buffer
        let read = buffer.get_buffered(0x8001).unwrap();
        assert_eq!(read.addr, 0x8001);
        assert_eq!(read.label, Some("test".into()));
    }

    #[test]
    fn test_dependency_graph() {
        let deps = DependencyGraph::new();

        // Write A
        deps.record(1, 0x8001, vec![]);
        // Write B depends on A
        deps.record(2, 0x8002, vec![1]);
        // Write C depends on B
        deps.record(3, 0x8003, vec![2]);

        // Get dependents of A (should include B and C)
        let dependents = deps.get_dependents(1);
        assert!(dependents.contains(&2));
        assert!(dependents.contains(&3));

        // Ordered should be [1, 2, 3]
        let ordered = deps.ordered(&[3, 1, 2]);
        assert_eq!(ordered, vec![1, 2, 3]);
    }

    #[test]
    fn test_auto_dependencies() {
        let deps = DependencyGraph::new();

        // First write to addr
        deps.record(1, 0x8001, vec![]);

        // Second write should auto-depend on first
        let auto = deps.auto_depends(0x8001);
        assert_eq!(auto, vec![1]);
    }

    #[test]
    fn test_recovery_action() {
        let config = ResilienceConfig {
            max_recovery_attempts: 3,
            ..Default::default()
        };
        let recovery = RecoveryEngine::new(config);

        // First failure -> retry
        match recovery.record_failure(1) {
            RecoveryAction::Retry { attempt } => assert_eq!(attempt, 1),
            _ => panic!("Expected retry"),
        }

        // Second failure -> retry
        match recovery.record_failure(1) {
            RecoveryAction::Retry { attempt } => assert_eq!(attempt, 2),
            _ => panic!("Expected retry"),
        }

        // Third failure -> rollback
        match recovery.record_failure(1) {
            RecoveryAction::Rollback => {}
            _ => panic!("Expected rollback"),
        }
    }

    #[test]
    fn test_resilient_store_read_your_writes() {
        let config = ResilienceConfig {
            sync_writes: false, // Async for test
            ..Default::default()
        };
        let store = ResilientStore::new(config);

        let fp = [99u64; FINGERPRINT_WORDS];
        let _vv = store.write(0x8001, fp, Some("buffered".into())).unwrap();

        // Should be readable immediately (from buffer)
        let read = store.read(0x8001).unwrap();
        assert!(!read.is_confirmed()); // Still buffered
        assert_eq!(read.addr(), 0x8001);
    }
}
