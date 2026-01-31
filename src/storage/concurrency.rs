//! Concurrency Layer - MVCC, Memory Pool, and Parallel Execution
//!
//! Addresses production gaps:
//! - OOM protection with bounded memory pool
//! - MVCC concurrent writers (beyond RwLock)
//! - Multi-core query parallelism
//! - Optimistic locking for stale read detection
//!
//! # Concurrent Write Resolution
//!
//! When SQL/Cypher queries make changes:
//! 1. Read captures version at read time
//! 2. Write checks if location changed since read
//! 3. Conflict → retry with fresh read or fail
//!
//! ```text
//! Query A: READ X (v5) → compute → WRITE X (expects v5)
//! Query B: READ X (v5) → compute → WRITE X (expects v5) ← CONFLICT!
//!
//! Resolution:
//! - First writer wins (v5 → v6)
//! - Second writer: ConflictError { expected: v5, actual: v6 }
//! - Caller can retry with fresh read
//! ```

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, AtomicBool, Ordering};
use std::sync::{Arc, RwLock, Mutex, Condvar};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use super::bind_space::FINGERPRINT_WORDS;
use super::temporal::Version;

// =============================================================================
// MEMORY POOL
// =============================================================================

/// Memory pool configuration
#[derive(Clone, Debug)]
pub struct MemoryPoolConfig {
    /// Hard limit in bytes (OOM protection)
    pub max_bytes: usize,
    /// Soft limit - trigger eviction above this
    pub soft_limit_bytes: usize,
    /// Per-allocation limit
    pub max_allocation: usize,
    /// Enable backpressure when near limit
    pub backpressure: bool,
    /// Backpressure threshold (0.0-1.0)
    pub backpressure_threshold: f64,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            max_bytes: 512 * 1024 * 1024,        // 512 MB hard limit
            soft_limit_bytes: 384 * 1024 * 1024, // 384 MB soft limit
            max_allocation: 16 * 1024 * 1024,    // 16 MB max single allocation
            backpressure: true,
            backpressure_threshold: 0.8,
        }
    }
}

impl MemoryPoolConfig {
    /// Embedded/constrained config
    pub fn embedded() -> Self {
        Self {
            max_bytes: 64 * 1024 * 1024,
            soft_limit_bytes: 48 * 1024 * 1024,
            max_allocation: 4 * 1024 * 1024,
            backpressure: true,
            backpressure_threshold: 0.7,
        }
    }

    /// Server config (more headroom)
    pub fn server() -> Self {
        Self {
            max_bytes: 4 * 1024 * 1024 * 1024,   // 4 GB
            soft_limit_bytes: 3 * 1024 * 1024 * 1024,
            max_allocation: 256 * 1024 * 1024,
            backpressure: true,
            backpressure_threshold: 0.85,
        }
    }
}

/// Bounded memory pool with OOM protection
pub struct MemoryPool {
    /// Current usage
    used: AtomicUsize,
    /// Peak usage (for monitoring)
    peak: AtomicUsize,
    /// Allocation count
    alloc_count: AtomicU64,
    /// Deallocation count
    dealloc_count: AtomicU64,
    /// OOM events
    oom_events: AtomicU64,
    /// Backpressure active
    backpressure_active: AtomicBool,
    /// Configuration
    config: MemoryPoolConfig,
    /// Waiters for memory (backpressure)
    waiters: Mutex<VecDeque<Arc<Condvar>>>,
}

impl MemoryPool {
    pub fn new(config: MemoryPoolConfig) -> Self {
        Self {
            used: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            alloc_count: AtomicU64::new(0),
            dealloc_count: AtomicU64::new(0),
            oom_events: AtomicU64::new(0),
            backpressure_active: AtomicBool::new(false),
            config,
            waiters: Mutex::new(VecDeque::new()),
        }
    }

    /// Try to allocate memory
    pub fn try_allocate(&self, bytes: usize) -> Result<MemoryGuard, MemoryError> {
        // Check single allocation limit
        if bytes > self.config.max_allocation {
            return Err(MemoryError::AllocationTooLarge {
                requested: bytes,
                max: self.config.max_allocation,
            });
        }

        // Try to reserve
        loop {
            let current = self.used.load(Ordering::SeqCst);
            let new_used = current + bytes;

            // Hard limit check
            if new_used > self.config.max_bytes {
                self.oom_events.fetch_add(1, Ordering::Relaxed);
                return Err(MemoryError::OutOfMemory {
                    requested: bytes,
                    available: self.config.max_bytes.saturating_sub(current),
                    total: self.config.max_bytes,
                });
            }

            // CAS to reserve
            if self.used.compare_exchange(
                current, new_used,
                Ordering::SeqCst, Ordering::SeqCst
            ).is_ok() {
                // Update peak
                let mut peak = self.peak.load(Ordering::Relaxed);
                while new_used > peak {
                    match self.peak.compare_exchange_weak(
                        peak, new_used,
                        Ordering::Relaxed, Ordering::Relaxed
                    ) {
                        Ok(_) => break,
                        Err(p) => peak = p,
                    }
                }

                self.alloc_count.fetch_add(1, Ordering::Relaxed);

                // Check backpressure
                let ratio = new_used as f64 / self.config.max_bytes as f64;
                self.backpressure_active.store(
                    ratio >= self.config.backpressure_threshold,
                    Ordering::Relaxed
                );

                return Ok(MemoryGuard {
                    pool: self,
                    bytes,
                });
            }
        }
    }

    /// Allocate with backpressure (blocks if over threshold)
    pub fn allocate(&self, bytes: usize, timeout: Duration) -> Result<MemoryGuard, MemoryError> {
        // Fast path: try immediate allocation
        match self.try_allocate(bytes) {
            Ok(guard) => return Ok(guard),
            Err(MemoryError::OutOfMemory { .. }) if self.config.backpressure => {
                // Fall through to wait
            }
            Err(e) => return Err(e),
        }

        // Slow path: wait for memory
        let cv = Arc::new(Condvar::new());
        {
            let mut waiters = self.waiters.lock().unwrap();
            waiters.push_back(Arc::clone(&cv));
        }

        let start = Instant::now();
        let dummy_mutex = Mutex::new(());

        loop {
            // Try allocation again
            match self.try_allocate(bytes) {
                Ok(guard) => return Ok(guard),
                Err(MemoryError::OutOfMemory { .. }) => {
                    // Continue waiting
                }
                Err(e) => return Err(e),
            }

            // Check timeout
            let elapsed = start.elapsed();
            if elapsed >= timeout {
                return Err(MemoryError::Timeout {
                    requested: bytes,
                    waited: elapsed,
                });
            }

            // Wait for notification
            let remaining = timeout - elapsed;
            let guard = dummy_mutex.lock().unwrap();
            let _ = cv.wait_timeout(guard, remaining.min(Duration::from_millis(100)));
        }
    }

    /// Release memory (called by MemoryGuard on drop)
    fn release(&self, bytes: usize) {
        self.used.fetch_sub(bytes, Ordering::SeqCst);
        self.dealloc_count.fetch_add(1, Ordering::Relaxed);

        // Update backpressure
        let current = self.used.load(Ordering::SeqCst);
        let ratio = current as f64 / self.config.max_bytes as f64;
        let was_active = self.backpressure_active.swap(
            ratio >= self.config.backpressure_threshold,
            Ordering::Relaxed
        );

        // Wake waiters if backpressure lifted
        if was_active && ratio < self.config.backpressure_threshold {
            if let Ok(mut waiters) = self.waiters.lock() {
                for waiter in waiters.drain(..) {
                    waiter.notify_one();
                }
            }
        }
    }

    /// Current usage
    pub fn used(&self) -> usize {
        self.used.load(Ordering::SeqCst)
    }

    /// Available memory
    pub fn available(&self) -> usize {
        self.config.max_bytes.saturating_sub(self.used())
    }

    /// Usage ratio (0.0-1.0)
    pub fn usage_ratio(&self) -> f64 {
        self.used() as f64 / self.config.max_bytes as f64
    }

    /// Is backpressure active?
    pub fn is_backpressure_active(&self) -> bool {
        self.backpressure_active.load(Ordering::Relaxed)
    }

    /// Get stats
    pub fn stats(&self) -> MemoryPoolStats {
        MemoryPoolStats {
            used: self.used(),
            peak: self.peak.load(Ordering::Relaxed),
            max: self.config.max_bytes,
            alloc_count: self.alloc_count.load(Ordering::Relaxed),
            dealloc_count: self.dealloc_count.load(Ordering::Relaxed),
            oom_events: self.oom_events.load(Ordering::Relaxed),
            backpressure_active: self.is_backpressure_active(),
        }
    }
}

/// RAII guard for allocated memory
pub struct MemoryGuard<'a> {
    pool: &'a MemoryPool,
    bytes: usize,
}

impl<'a> Drop for MemoryGuard<'a> {
    fn drop(&mut self) {
        self.pool.release(self.bytes);
    }
}

impl<'a> MemoryGuard<'a> {
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

/// Memory pool statistics
#[derive(Clone, Debug)]
pub struct MemoryPoolStats {
    pub used: usize,
    pub peak: usize,
    pub max: usize,
    pub alloc_count: u64,
    pub dealloc_count: u64,
    pub oom_events: u64,
    pub backpressure_active: bool,
}

/// Memory errors
#[derive(Clone, Debug)]
pub enum MemoryError {
    OutOfMemory {
        requested: usize,
        available: usize,
        total: usize,
    },
    AllocationTooLarge {
        requested: usize,
        max: usize,
    },
    Timeout {
        requested: usize,
        waited: Duration,
    },
}

impl std::fmt::Display for MemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfMemory { requested, available, total } => {
                write!(f, "OOM: requested {} bytes, {} available of {} total",
                    requested, available, total)
            }
            Self::AllocationTooLarge { requested, max } => {
                write!(f, "Allocation {} bytes exceeds max {}", requested, max)
            }
            Self::Timeout { requested, waited } => {
                write!(f, "Timeout after {:?} waiting for {} bytes", waited, requested)
            }
        }
    }
}

impl std::error::Error for MemoryError {}

// =============================================================================
// MVCC SLOT
// =============================================================================

/// A versioned slot for MVCC
#[derive(Clone, Debug)]
pub struct MvccSlot {
    /// Current version
    pub version: Version,
    /// Data fingerprint
    pub fingerprint: [u64; FINGERPRINT_WORDS],
    /// Label
    pub label: Option<String>,
    /// Last writer (for debugging)
    pub last_writer: u64,
    /// Timestamp
    pub timestamp: u64,
}

/// Read handle with version tracking
#[derive(Clone, Debug)]
pub struct ReadHandle {
    /// Address read
    pub addr: u16,
    /// Version at read time
    pub version: Version,
    /// Timestamp of read
    pub read_at: Instant,
}

impl ReadHandle {
    pub fn new(addr: u16, version: Version) -> Self {
        Self {
            addr,
            version,
            read_at: Instant::now(),
        }
    }

    /// Check if handle is stale (version changed)
    pub fn is_stale(&self, current_version: Version) -> bool {
        current_version > self.version
    }

    /// Age of this read
    pub fn age(&self) -> Duration {
        self.read_at.elapsed()
    }
}

/// Write intent with expected version
#[derive(Clone, Debug)]
pub struct WriteIntent {
    /// Address to write
    pub addr: u16,
    /// Expected version (from read)
    pub expected_version: Version,
    /// New fingerprint
    pub fingerprint: [u64; FINGERPRINT_WORDS],
    /// New label
    pub label: Option<String>,
    /// Writer ID
    pub writer_id: u64,
}

/// Result of a write attempt
#[derive(Clone, Debug)]
pub enum WriteResult {
    /// Write succeeded, new version
    Success { new_version: Version },
    /// Conflict: version changed since read
    Conflict {
        expected: Version,
        actual: Version,
        conflicting_writer: u64,
    },
    /// Slot doesn't exist
    NotFound,
}

// =============================================================================
// MVCC STORE
// =============================================================================

/// MVCC store with concurrent writers
pub struct MvccStore {
    /// Slots indexed by address
    slots: Vec<RwLock<Option<MvccSlot>>>,
    /// Global version counter
    version: AtomicU64,
    /// Writer ID counter
    next_writer: AtomicU64,
    /// Active writers count
    active_writers: AtomicUsize,
    /// Write conflict count (for monitoring)
    conflicts: AtomicU64,
    /// Memory pool
    pool: Arc<MemoryPool>,
}

impl MvccStore {
    pub fn new(slot_count: usize, pool: Arc<MemoryPool>) -> Self {
        let mut slots = Vec::with_capacity(slot_count);
        for _ in 0..slot_count {
            slots.push(RwLock::new(None));
        }

        Self {
            slots,
            version: AtomicU64::new(0),
            next_writer: AtomicU64::new(1),
            active_writers: AtomicUsize::new(0),
            conflicts: AtomicU64::new(0),
            pool,
        }
    }

    /// Get a writer ID
    pub fn writer_id(&self) -> u64 {
        self.next_writer.fetch_add(1, Ordering::SeqCst)
    }

    /// Read with version tracking
    pub fn read(&self, addr: u16) -> Option<(MvccSlot, ReadHandle)> {
        let idx = addr as usize;
        if idx >= self.slots.len() {
            return None;
        }

        let slot = self.slots[idx].read().ok()?;
        let data = slot.as_ref()?.clone();
        let handle = ReadHandle::new(addr, data.version);
        Some((data, handle))
    }

    /// Read without handle (when you don't plan to write)
    pub fn read_only(&self, addr: u16) -> Option<MvccSlot> {
        let idx = addr as usize;
        if idx >= self.slots.len() {
            return None;
        }

        self.slots[idx].read().ok()?.clone()
    }

    /// Write with optimistic locking
    pub fn write(&self, intent: WriteIntent) -> WriteResult {
        let idx = intent.addr as usize;
        if idx >= self.slots.len() {
            return WriteResult::NotFound;
        }

        // Acquire write lock
        let mut slot = match self.slots[idx].write() {
            Ok(s) => s,
            Err(_) => return WriteResult::Conflict {
                expected: intent.expected_version,
                actual: 0,
                conflicting_writer: 0,
            },
        };

        self.active_writers.fetch_add(1, Ordering::SeqCst);

        // Check version
        let current_version = slot.as_ref().map(|s| s.version).unwrap_or(0);
        if current_version != intent.expected_version {
            self.active_writers.fetch_sub(1, Ordering::SeqCst);
            self.conflicts.fetch_add(1, Ordering::Relaxed);
            return WriteResult::Conflict {
                expected: intent.expected_version,
                actual: current_version,
                conflicting_writer: slot.as_ref().map(|s| s.last_writer).unwrap_or(0),
            };
        }

        // Advance version and write
        let new_version = self.version.fetch_add(1, Ordering::SeqCst) + 1;
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        *slot = Some(MvccSlot {
            version: new_version,
            fingerprint: intent.fingerprint,
            label: intent.label,
            last_writer: intent.writer_id,
            timestamp,
        });

        self.active_writers.fetch_sub(1, Ordering::SeqCst);

        WriteResult::Success { new_version }
    }

    /// Write unconditionally (no version check)
    pub fn write_unconditional(
        &self,
        addr: u16,
        fingerprint: [u64; FINGERPRINT_WORDS],
        label: Option<String>,
        writer_id: u64,
    ) -> Option<Version> {
        let idx = addr as usize;
        if idx >= self.slots.len() {
            return None;
        }

        let mut slot = self.slots[idx].write().ok()?;
        let new_version = self.version.fetch_add(1, Ordering::SeqCst) + 1;
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        *slot = Some(MvccSlot {
            version: new_version,
            fingerprint,
            label,
            last_writer: writer_id,
            timestamp,
        });

        Some(new_version)
    }

    /// Delete with version check
    pub fn delete(&self, addr: u16, expected_version: Version) -> WriteResult {
        let idx = addr as usize;
        if idx >= self.slots.len() {
            return WriteResult::NotFound;
        }

        let mut slot = match self.slots[idx].write() {
            Ok(s) => s,
            Err(_) => return WriteResult::Conflict {
                expected: expected_version,
                actual: 0,
                conflicting_writer: 0,
            },
        };

        let current_version = slot.as_ref().map(|s| s.version).unwrap_or(0);
        if current_version != expected_version {
            self.conflicts.fetch_add(1, Ordering::Relaxed);
            return WriteResult::Conflict {
                expected: expected_version,
                actual: current_version,
                conflicting_writer: slot.as_ref().map(|s| s.last_writer).unwrap_or(0),
            };
        }

        let new_version = self.version.fetch_add(1, Ordering::SeqCst) + 1;
        *slot = None;

        WriteResult::Success { new_version }
    }

    /// Current global version
    pub fn current_version(&self) -> Version {
        self.version.load(Ordering::SeqCst)
    }

    /// Active writer count
    pub fn active_writers(&self) -> usize {
        self.active_writers.load(Ordering::SeqCst)
    }

    /// Conflict count
    pub fn conflicts(&self) -> u64 {
        self.conflicts.load(Ordering::Relaxed)
    }
}

// =============================================================================
// PARALLEL QUERY ENGINE
// =============================================================================

/// Work item for parallel execution
pub trait WorkItem: Send + 'static {
    type Output: Send;
    fn execute(self) -> Self::Output;
}

/// Parallel query configuration
#[derive(Clone, Debug)]
pub struct ParallelConfig {
    /// Number of worker threads
    pub worker_count: usize,
    /// Work queue capacity
    pub queue_capacity: usize,
    /// Batch size for work stealing
    pub steal_batch_size: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        let cpus = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);
        Self {
            worker_count: cpus,
            queue_capacity: 10_000,
            steal_batch_size: 32,
        }
    }
}

/// Simple parallel executor
pub struct ParallelExecutor {
    /// Worker threads
    workers: Vec<JoinHandle<()>>,
    /// Work sender
    sender: crossbeam::channel::Sender<Box<dyn FnOnce() + Send>>,
    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
    /// Active task count
    active: Arc<AtomicUsize>,
}

impl ParallelExecutor {
    pub fn new(config: ParallelConfig) -> Self {
        let (sender, receiver): (
            crossbeam::channel::Sender<Box<dyn FnOnce() + Send>>,
            crossbeam::channel::Receiver<Box<dyn FnOnce() + Send>>,
        ) = crossbeam::channel::bounded(config.queue_capacity);
        let shutdown = Arc::new(AtomicBool::new(false));
        let active = Arc::new(AtomicUsize::new(0));

        let mut workers = Vec::with_capacity(config.worker_count);
        for _ in 0..config.worker_count {
            let receiver: crossbeam::channel::Receiver<Box<dyn FnOnce() + Send>> = receiver.clone();
            let shutdown = Arc::clone(&shutdown);
            let active = Arc::clone(&active);

            let handle = thread::spawn(move || {
                while !shutdown.load(Ordering::Relaxed) {
                    match receiver.recv_timeout(Duration::from_millis(100)) {
                        Ok(work) => {
                            active.fetch_add(1, Ordering::SeqCst);
                            work();
                            active.fetch_sub(1, Ordering::SeqCst);
                        }
                        Err(crossbeam::channel::RecvTimeoutError::Timeout) => continue,
                        Err(crossbeam::channel::RecvTimeoutError::Disconnected) => break,
                    }
                }
            });
            workers.push(handle);
        }

        Self {
            workers,
            sender,
            shutdown,
            active,
        }
    }

    /// Submit work
    pub fn submit<F>(&self, work: F) -> Result<(), ParallelError>
    where
        F: FnOnce() + Send + 'static,
    {
        self.sender.send(Box::new(work))
            .map_err(|_| ParallelError::QueueFull)
    }

    /// Submit and get future result
    pub fn submit_with_result<F, T>(&self, work: F) -> Result<ResultHandle<T>, ParallelError>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let (tx, rx) = crossbeam::channel::bounded(1);
        self.submit(move || {
            let result = work();
            let _ = tx.send(result);
        })?;
        Ok(ResultHandle { receiver: rx })
    }

    /// Execute in parallel, collect results
    pub fn parallel_map<T, U, F, I>(&self, items: I, f: F) -> Vec<U>
    where
        I: IntoIterator<Item = T>,
        T: Send + 'static,
        U: Send + 'static,
        F: Fn(T) -> U + Send + Sync + Clone + 'static,
    {
        let items: Vec<T> = items.into_iter().collect();
        let count = items.len();

        // Use channels instead of shared vec to avoid Clone requirement
        let (tx, rx) = crossbeam::channel::bounded::<(usize, U)>(count);

        let handles: Vec<_> = items.into_iter()
            .enumerate()
            .map(|(i, item)| {
                let f = f.clone();
                let tx = tx.clone();
                self.submit(move || {
                    let result = f(item);
                    let _ = tx.send((i, result));
                })
            })
            .collect();

        // Drop sender so rx knows when done
        drop(tx);

        // Wait for all handles
        for handle in handles.into_iter() {
            let _ = handle;
        }

        // Collect results in order
        let mut results: Vec<Option<U>> = (0..count).map(|_| None).collect();
        for (i, result) in rx.iter() {
            if i < results.len() {
                results[i] = Some(result);
            }
        }

        results.into_iter().flatten().collect()
    }

    /// Active task count
    pub fn active_tasks(&self) -> usize {
        self.active.load(Ordering::SeqCst)
    }

    /// Queue size
    pub fn queue_size(&self) -> usize {
        self.sender.len()
    }

    /// Shutdown executor
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }
}

impl Drop for ParallelExecutor {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Handle to wait for result
pub struct ResultHandle<T> {
    receiver: crossbeam::channel::Receiver<T>,
}

impl<T> ResultHandle<T> {
    /// Wait for result
    pub fn wait(self) -> Result<T, ParallelError> {
        self.receiver.recv().map_err(|_| ParallelError::TaskFailed)
    }

    /// Wait with timeout
    pub fn wait_timeout(self, timeout: Duration) -> Result<T, ParallelError> {
        self.receiver.recv_timeout(timeout)
            .map_err(|_| ParallelError::Timeout)
    }

    /// Try to get result without blocking
    pub fn try_get(&self) -> Option<T> {
        self.receiver.try_recv().ok()
    }
}

/// Parallel execution errors
#[derive(Clone, Debug)]
pub enum ParallelError {
    QueueFull,
    TaskFailed,
    Timeout,
}

impl std::fmt::Display for ParallelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::QueueFull => write!(f, "Work queue is full"),
            Self::TaskFailed => write!(f, "Task execution failed"),
            Self::Timeout => write!(f, "Task timed out"),
        }
    }
}

impl std::error::Error for ParallelError {}

// =============================================================================
// CONCURRENT QUERY CONTEXT
// =============================================================================

/// Context for a concurrent query
pub struct QueryContext {
    /// Reader ID (for tracking)
    pub reader_id: u64,
    /// Read handles (for optimistic locking)
    reads: Mutex<Vec<ReadHandle>>,
    /// Started at
    pub started_at: Instant,
    /// Memory guard
    _memory: Option<MemoryGuard<'static>>,
}

impl QueryContext {
    pub fn new(reader_id: u64) -> Self {
        Self {
            reader_id,
            reads: Mutex::new(Vec::new()),
            started_at: Instant::now(),
            _memory: None,
        }
    }

    /// Record a read for later validation
    pub fn record_read(&self, handle: ReadHandle) {
        if let Ok(mut reads) = self.reads.lock() {
            reads.push(handle);
        }
    }

    /// Validate all reads are still current
    pub fn validate_reads(&self, store: &MvccStore) -> Result<(), ConflictError> {
        let reads = self.reads.lock().map_err(|_| ConflictError {
            addr: 0,
            expected: 0,
            actual: 0,
        })?;

        for read in reads.iter() {
            if let Some(slot) = store.read_only(read.addr) {
                if slot.version > read.version {
                    return Err(ConflictError {
                        addr: read.addr,
                        expected: read.version,
                        actual: slot.version,
                    });
                }
            }
        }

        Ok(())
    }

    /// Get all read handles
    pub fn get_reads(&self) -> Vec<ReadHandle> {
        self.reads.lock().map(|r| r.clone()).unwrap_or_default()
    }

    /// Query duration
    pub fn duration(&self) -> Duration {
        self.started_at.elapsed()
    }
}

/// Conflict error for optimistic locking
#[derive(Clone, Debug)]
pub struct ConflictError {
    pub addr: u16,
    pub expected: Version,
    pub actual: Version,
}

impl std::fmt::Display for ConflictError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Conflict at {:04x}: expected v{}, actual v{}",
            self.addr, self.expected, self.actual)
    }
}

impl std::error::Error for ConflictError {}

// =============================================================================
// CONCURRENT STORE (COMBINES ALL)
// =============================================================================

/// Full concurrent store with all features
pub struct ConcurrentStore {
    /// MVCC storage
    mvcc: MvccStore,
    /// Memory pool
    pool: Arc<MemoryPool>,
    /// Parallel executor
    executor: ParallelExecutor,
    /// Query counter
    query_counter: AtomicU64,
}

impl ConcurrentStore {
    pub fn new(
        slot_count: usize,
        pool_config: MemoryPoolConfig,
        parallel_config: ParallelConfig,
    ) -> Self {
        let pool = Arc::new(MemoryPool::new(pool_config));
        Self {
            mvcc: MvccStore::new(slot_count, Arc::clone(&pool)),
            pool,
            executor: ParallelExecutor::new(parallel_config),
        query_counter: AtomicU64::new(0),
        }
    }

    /// Start a query context
    pub fn query(&self) -> QueryContext {
        let id = self.query_counter.fetch_add(1, Ordering::SeqCst);
        QueryContext::new(id)
    }

    /// Read with tracking
    pub fn read(&self, ctx: &QueryContext, addr: u16) -> Option<MvccSlot> {
        let (slot, handle) = self.mvcc.read(addr)?;
        ctx.record_read(handle);
        Some(slot)
    }

    /// Write with optimistic locking
    pub fn write(
        &self,
        ctx: &QueryContext,
        addr: u16,
        fingerprint: [u64; FINGERPRINT_WORDS],
        label: Option<String>,
    ) -> Result<Version, WriteConflict> {
        // Validate reads first
        ctx.validate_reads(&self.mvcc)
            .map_err(|e| WriteConflict::StaleRead(e))?;

        // Get expected version from context
        let expected = ctx.get_reads()
            .iter()
            .find(|r| r.addr == addr)
            .map(|r| r.version)
            .unwrap_or(0);

        let writer_id = self.mvcc.writer_id();
        let intent = WriteIntent {
            addr,
            expected_version: expected,
            fingerprint,
            label,
            writer_id,
        };

        match self.mvcc.write(intent) {
            WriteResult::Success { new_version } => Ok(new_version),
            WriteResult::Conflict { expected, actual, conflicting_writer } => {
                Err(WriteConflict::VersionMismatch {
                    addr,
                    expected,
                    actual,
                    conflicting_writer,
                })
            }
            WriteResult::NotFound => Err(WriteConflict::NotFound(addr)),
        }
    }

    /// Parallel read (sequential for now - parallel version requires Arc<MvccStore>)
    pub fn parallel_read(&self, addrs: &[u16]) -> Vec<Option<MvccSlot>> {
        // Sequential implementation - parallel would require refactoring MvccStore to be Arc-wrapped
        addrs.iter().map(|&addr| self.mvcc.read_only(addr)).collect()
    }

    /// Memory pool reference
    pub fn pool(&self) -> &MemoryPool {
        &self.pool
    }

    /// MVCC store reference
    pub fn mvcc(&self) -> &MvccStore {
        &self.mvcc
    }

    /// Executor reference
    pub fn executor(&self) -> &ParallelExecutor {
        &self.executor
    }

    /// Combined stats
    pub fn stats(&self) -> ConcurrentStats {
        ConcurrentStats {
            memory: self.pool.stats(),
            mvcc_version: self.mvcc.current_version(),
            mvcc_conflicts: self.mvcc.conflicts(),
            active_writers: self.mvcc.active_writers(),
            executor_active: self.executor.active_tasks(),
            executor_queue: self.executor.queue_size(),
            total_queries: self.query_counter.load(Ordering::Relaxed),
        }
    }
}

/// Write conflict error
#[derive(Clone, Debug)]
pub enum WriteConflict {
    /// Read was stale (another writer modified)
    StaleRead(ConflictError),
    /// Version mismatch on write
    VersionMismatch {
        addr: u16,
        expected: Version,
        actual: Version,
        conflicting_writer: u64,
    },
    /// Address not found
    NotFound(u16),
}

impl std::fmt::Display for WriteConflict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StaleRead(e) => write!(f, "Stale read: {}", e),
            Self::VersionMismatch { addr, expected, actual, conflicting_writer } => {
                write!(f, "Version mismatch at {:04x}: expected v{}, actual v{} (writer {})",
                    addr, expected, actual, conflicting_writer)
            }
            Self::NotFound(addr) => write!(f, "Address {:04x} not found", addr),
        }
    }
}

impl std::error::Error for WriteConflict {}

/// Combined stats
#[derive(Clone, Debug)]
pub struct ConcurrentStats {
    pub memory: MemoryPoolStats,
    pub mvcc_version: Version,
    pub mvcc_conflicts: u64,
    pub active_writers: usize,
    pub executor_active: usize,
    pub executor_queue: usize,
    pub total_queries: u64,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_basic() {
        let config = MemoryPoolConfig {
            max_bytes: 1024,
            soft_limit_bytes: 768,
            max_allocation: 512,
            backpressure: false,
            backpressure_threshold: 0.8,
        };
        let pool = MemoryPool::new(config);

        // Allocate
        let guard1 = pool.try_allocate(256).unwrap();
        assert_eq!(pool.used(), 256);

        let guard2 = pool.try_allocate(256).unwrap();
        assert_eq!(pool.used(), 512);

        // Drop releases
        drop(guard1);
        assert_eq!(pool.used(), 256);

        drop(guard2);
        assert_eq!(pool.used(), 0);
    }

    #[test]
    fn test_memory_pool_oom() {
        let config = MemoryPoolConfig {
            max_bytes: 1024,
            soft_limit_bytes: 768,
            max_allocation: 512,
            backpressure: false,
            backpressure_threshold: 0.8,
        };
        let pool = MemoryPool::new(config);

        let _guard = pool.try_allocate(512).unwrap();
        let _guard2 = pool.try_allocate(256).unwrap();

        // Should OOM
        let result = pool.try_allocate(512);
        assert!(matches!(result, Err(MemoryError::OutOfMemory { .. })));
    }

    #[test]
    fn test_mvcc_basic() {
        let pool = Arc::new(MemoryPool::new(MemoryPoolConfig::default()));
        let store = MvccStore::new(100, pool);

        let writer = store.writer_id();
        let fp = [42u64; FINGERPRINT_WORDS];

        // Initial write (expect v0)
        let intent = WriteIntent {
            addr: 5,
            expected_version: 0,
            fingerprint: fp,
            label: Some("test".into()),
            writer_id: writer,
        };

        match store.write(intent) {
            WriteResult::Success { new_version } => assert!(new_version > 0),
            _ => panic!("Expected success"),
        }

        // Read back
        let (slot, _handle) = store.read(5).unwrap();
        assert_eq!(slot.label, Some("test".into()));
    }

    #[test]
    fn test_mvcc_conflict() {
        let pool = Arc::new(MemoryPool::new(MemoryPoolConfig::default()));
        let store = MvccStore::new(100, pool);

        let fp = [42u64; FINGERPRINT_WORDS];

        // Writer A writes
        let intent_a = WriteIntent {
            addr: 5,
            expected_version: 0,
            fingerprint: fp,
            label: Some("A".into()),
            writer_id: 1,
        };
        let v1 = match store.write(intent_a) {
            WriteResult::Success { new_version } => new_version,
            _ => panic!("Expected success"),
        };

        // Writer B tries with stale version
        let intent_b = WriteIntent {
            addr: 5,
            expected_version: 0, // Stale!
            fingerprint: fp,
            label: Some("B".into()),
            writer_id: 2,
        };

        match store.write(intent_b) {
            WriteResult::Conflict { expected, actual, .. } => {
                assert_eq!(expected, 0);
                assert_eq!(actual, v1);
            }
            _ => panic!("Expected conflict"),
        }
    }

    #[test]
    fn test_parallel_executor() {
        let executor = ParallelExecutor::new(ParallelConfig::default());

        // Submit work
        let handle = executor.submit_with_result(|| 42).unwrap();
        assert_eq!(handle.wait().unwrap(), 42);
    }

    #[test]
    fn test_parallel_map() {
        let executor = ParallelExecutor::new(ParallelConfig::default());

        let results = executor.parallel_map(vec![1, 2, 3, 4, 5], |x| x * 2);
        assert_eq!(results, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn test_concurrent_store_conflict() {
        let store = ConcurrentStore::new(
            1000,
            MemoryPoolConfig::default(),
            ParallelConfig::default(),
        );

        let fp = [99u64; FINGERPRINT_WORDS];

        // Initial write
        store.mvcc.write_unconditional(5, fp, Some("initial".into()), 0);

        // Query A reads
        let ctx_a = store.query();
        let _slot_a = store.read(&ctx_a, 5);

        // Query B reads and writes
        let ctx_b = store.query();
        let _slot_b = store.read(&ctx_b, 5);
        let v2 = store.write(&ctx_b, 5, fp, Some("B".into())).unwrap();
        assert!(v2 > 0);

        // Query A tries to write with stale read
        let result = store.write(&ctx_a, 5, fp, Some("A".into()));
        assert!(matches!(result, Err(WriteConflict::StaleRead(_))));
    }
}
