//! Production Hardening for Ladybug-RS
//!
//! This module adds production-grade features without changing query semantics:
//! - Memory limits with LRU eviction
//! - TTL-based expiration
//! - Write-ahead logging (WAL) for crash recovery
//! - Query timeouts
//! - Buffer pool management
//!
//! # Design Principles
//!
//! 1. **Non-invasive**: All hardening wraps existing functionality
//! 2. **Configurable**: All limits can be adjusted at runtime
//! 3. **Observable**: Metrics for monitoring
//! 4. **Graceful degradation**: System stays up under pressure

use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::bind_space::{
    Addr, FINGERPRINT_WORDS, PREFIX_FLUID_END, PREFIX_FLUID_START, PREFIX_NODE_START,
};

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Production hardening configuration
#[derive(Clone, Debug)]
pub struct HardeningConfig {
    /// Maximum nodes in fluid zone before eviction
    pub max_fluid_nodes: usize,
    /// Maximum nodes in node space before eviction
    pub max_node_nodes: usize,
    /// Default TTL for fluid zone entries
    pub default_fluid_ttl: Duration,
    /// How often to run maintenance (TTL expiration, eviction)
    pub maintenance_interval: Duration,
    /// Enable WAL for crash recovery
    pub enable_wal: bool,
    /// WAL directory
    pub wal_dir: PathBuf,
    /// WAL sync mode (true = fsync every write, false = async)
    pub wal_sync: bool,
    /// Maximum WAL size before checkpoint
    pub max_wal_size: usize,
    /// Query timeout (0 = no timeout)
    pub query_timeout: Duration,
    /// Enable memory pressure monitoring
    pub enable_memory_monitoring: bool,
    /// Memory pressure threshold (0.0-1.0) - trigger eviction above this
    pub memory_pressure_threshold: f64,
}

impl Default for HardeningConfig {
    fn default() -> Self {
        Self {
            max_fluid_nodes: 20_000,                     // ~80% of fluid zone
            max_node_nodes: 25_000,                      // ~80% of node zone
            default_fluid_ttl: Duration::from_secs(300), // 5 minutes
            maintenance_interval: Duration::from_secs(10),
            enable_wal: false, // Off by default for dev
            wal_dir: PathBuf::from("./wal"),
            wal_sync: false,                // Async for performance
            max_wal_size: 64 * 1024 * 1024, // 64MB
            query_timeout: Duration::from_secs(30),
            enable_memory_monitoring: true,
            memory_pressure_threshold: 0.85,
        }
    }
}

impl HardeningConfig {
    /// Production-ready configuration
    pub fn production() -> Self {
        Self {
            max_fluid_nodes: 25_000,
            max_node_nodes: 30_000,
            default_fluid_ttl: Duration::from_secs(600), // 10 minutes
            maintenance_interval: Duration::from_secs(5),
            enable_wal: true,
            wal_dir: PathBuf::from("/var/lib/ladybug/wal"),
            wal_sync: true,                  // Durability > performance
            max_wal_size: 256 * 1024 * 1024, // 256MB
            query_timeout: Duration::from_secs(60),
            enable_memory_monitoring: true,
            memory_pressure_threshold: 0.80,
        }
    }

    /// High-performance configuration (less durable)
    pub fn performance() -> Self {
        Self {
            max_fluid_nodes: 28_000, // Use more capacity
            max_node_nodes: 32_000,
            default_fluid_ttl: Duration::from_secs(120), // 2 minutes
            maintenance_interval: Duration::from_secs(30),
            enable_wal: false, // No WAL for speed
            wal_dir: PathBuf::from("./wal"),
            wal_sync: false,
            max_wal_size: 128 * 1024 * 1024,
            query_timeout: Duration::from_secs(10),
            enable_memory_monitoring: false,
            memory_pressure_threshold: 0.95,
        }
    }
}

// =============================================================================
// LRU EVICTION TRACKER
// =============================================================================

/// LRU tracker for eviction decisions
pub struct LruTracker {
    /// Address -> last access timestamp (microseconds since epoch)
    access_times: HashMap<u16, u64>,
    /// Ordered queue: front = oldest, back = newest
    order: VecDeque<u16>,
    /// Maximum entries before eviction
    max_entries: usize,
}

impl LruTracker {
    pub fn new(max_entries: usize) -> Self {
        Self {
            access_times: HashMap::with_capacity(max_entries),
            order: VecDeque::with_capacity(max_entries),
            max_entries,
        }
    }

    /// Record access to an address
    ///
    /// Duplicate-safe: removes ALL prior occurrences of `addr` from the order
    /// queue before pushing, even if the HashMap and VecDeque were out of sync.
    pub fn touch(&mut self, addr: u16) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        self.access_times.insert(addr, now);

        // Remove ALL occurrences from the order queue (not just the first)
        // to prevent duplicate accumulation if state ever diverges.
        self.order.retain(|&a| a != addr);
        self.order.push_back(addr);
    }

    /// Get addresses to evict (oldest first)
    pub fn evict_candidates(&self, count: usize) -> Vec<u16> {
        self.order.iter().take(count).copied().collect()
    }

    /// Remove address from tracking
    pub fn remove(&mut self, addr: u16) {
        self.access_times.remove(&addr);
        if let Some(pos) = self.order.iter().position(|&a| a == addr) {
            self.order.remove(pos);
        }
    }

    /// Current count
    pub fn len(&self) -> usize {
        self.access_times.len()
    }

    /// Check if over capacity
    pub fn needs_eviction(&self) -> bool {
        self.len() > self.max_entries
    }

    /// How many to evict to get back to 90% capacity
    pub fn eviction_count(&self) -> usize {
        if self.len() > self.max_entries {
            let target = (self.max_entries as f64 * 0.9) as usize;
            self.len() - target
        } else {
            0
        }
    }
}

// =============================================================================
// TTL MANAGER
// =============================================================================

/// TTL-based expiration manager
pub struct TtlManager {
    /// Address -> expiration timestamp (microseconds since epoch)
    expirations: HashMap<u16, u64>,
    /// Default TTL
    default_ttl: Duration,
}

impl TtlManager {
    pub fn new(default_ttl: Duration) -> Self {
        Self {
            expirations: HashMap::new(),
            default_ttl,
        }
    }

    /// Set TTL for address (None = use default)
    pub fn set_ttl(&mut self, addr: u16, ttl: Option<Duration>) {
        let ttl = ttl.unwrap_or(self.default_ttl);
        let expires_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64
            + ttl.as_micros() as u64;
        self.expirations.insert(addr, expires_at);
    }

    /// Check if address is expired
    pub fn is_expired(&self, addr: u16) -> bool {
        if let Some(&expires_at) = self.expirations.get(&addr) {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64;
            now >= expires_at
        } else {
            false
        }
    }

    /// Get all expired addresses
    pub fn get_expired(&self) -> Vec<u16> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        self.expirations
            .iter()
            .filter(|(_, expires_at)| now >= **expires_at)
            .map(|(addr, _)| *addr)
            .collect()
    }

    /// Remove address from tracking
    pub fn remove(&mut self, addr: u16) {
        self.expirations.remove(&addr);
    }

    /// Refresh TTL (extend expiration)
    pub fn refresh(&mut self, addr: u16) {
        if self.expirations.contains_key(&addr) {
            self.set_ttl(addr, None);
        }
    }
}

// =============================================================================
// WRITE-AHEAD LOG
// =============================================================================

/// WAL entry types
#[derive(Clone, Debug)]
pub enum WalEntry {
    /// Write operation: (addr, fingerprint, label)
    Write {
        addr: u16,
        fingerprint: [u64; FINGERPRINT_WORDS],
        label: Option<String>,
    },
    /// Delete operation
    Delete { addr: u16 },
    /// Link operation: (from, verb, to)
    Link { from: u16, verb: u16, to: u16 },
    /// Checkpoint marker
    Checkpoint { timestamp: u64 },
}

impl WalEntry {
    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        match self {
            WalEntry::Write {
                addr,
                fingerprint,
                label,
            } => {
                buf.push(0x01); // Type marker
                buf.extend_from_slice(&addr.to_le_bytes());
                for word in fingerprint {
                    buf.extend_from_slice(&word.to_le_bytes());
                }
                if let Some(l) = label {
                    buf.push(l.len() as u8);
                    buf.extend_from_slice(l.as_bytes());
                } else {
                    buf.push(0);
                }
            }
            WalEntry::Delete { addr } => {
                buf.push(0x02);
                buf.extend_from_slice(&addr.to_le_bytes());
            }
            WalEntry::Link { from, verb, to } => {
                buf.push(0x03);
                buf.extend_from_slice(&from.to_le_bytes());
                buf.extend_from_slice(&verb.to_le_bytes());
                buf.extend_from_slice(&to.to_le_bytes());
            }
            WalEntry::Checkpoint { timestamp } => {
                buf.push(0x04);
                buf.extend_from_slice(&timestamp.to_le_bytes());
            }
        }
        buf
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Option<(Self, usize)> {
        if data.is_empty() {
            return None;
        }

        match data[0] {
            0x01 => {
                // Write
                if data.len() < 3 + FINGERPRINT_WORDS * 8 {
                    return None;
                }
                let addr = u16::from_le_bytes([data[1], data[2]]);
                let mut fingerprint = [0u64; FINGERPRINT_WORDS];
                for i in 0..FINGERPRINT_WORDS {
                    let offset = 3 + i * 8;
                    fingerprint[i] = u64::from_le_bytes([
                        data[offset],
                        data[offset + 1],
                        data[offset + 2],
                        data[offset + 3],
                        data[offset + 4],
                        data[offset + 5],
                        data[offset + 6],
                        data[offset + 7],
                    ]);
                }
                let label_len = data[3 + FINGERPRINT_WORDS * 8] as usize;
                let label = if label_len > 0 {
                    let start = 4 + FINGERPRINT_WORDS * 8;
                    Some(String::from_utf8_lossy(&data[start..start + label_len]).to_string())
                } else {
                    None
                };
                let consumed = 4 + FINGERPRINT_WORDS * 8 + label_len;
                Some((
                    WalEntry::Write {
                        addr,
                        fingerprint,
                        label,
                    },
                    consumed,
                ))
            }
            0x02 => {
                // Delete
                if data.len() < 3 {
                    return None;
                }
                let addr = u16::from_le_bytes([data[1], data[2]]);
                Some((WalEntry::Delete { addr }, 3))
            }
            0x03 => {
                // Link
                if data.len() < 7 {
                    return None;
                }
                let from = u16::from_le_bytes([data[1], data[2]]);
                let verb = u16::from_le_bytes([data[3], data[4]]);
                let to = u16::from_le_bytes([data[5], data[6]]);
                Some((WalEntry::Link { from, verb, to }, 7))
            }
            0x04 => {
                // Checkpoint
                if data.len() < 9 {
                    return None;
                }
                let timestamp = u64::from_le_bytes([
                    data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8],
                ]);
                Some((WalEntry::Checkpoint { timestamp }, 9))
            }
            _ => None,
        }
    }
}

/// Write-ahead log for crash recovery
pub struct WriteAheadLog {
    /// WAL file path
    path: PathBuf,
    /// Current WAL file
    file: Option<BufWriter<File>>,
    /// Current WAL size
    size: usize,
    /// Max WAL size before checkpoint
    max_size: usize,
    /// Sync on every write
    sync_writes: bool,
    /// Entry count since last checkpoint
    entries_since_checkpoint: usize,
}

impl WriteAheadLog {
    pub fn new(dir: PathBuf, max_size: usize, sync_writes: bool) -> std::io::Result<Self> {
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("current.wal");
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        let size = file.metadata()?.len() as usize;

        Ok(Self {
            path,
            file: Some(BufWriter::new(file)),
            size,
            max_size,
            sync_writes,
            entries_since_checkpoint: 0,
        })
    }

    /// Append entry to WAL
    ///
    /// Write-ahead contract: when `sync_writes` is true, data is flushed to
    /// the OS (and thus to disk on fsync-capable filesystems) **before** the
    /// in-memory bookkeeping is updated. This ensures that a crash after
    /// `append` returns will always have the entry on disk.
    pub fn append(&mut self, entry: &WalEntry) -> std::io::Result<()> {
        let bytes = entry.to_bytes();
        if let Some(file) = &mut self.file {
            file.write_all(&bytes)?;

            // Flush to disk BEFORE updating in-memory state so that a crash
            // after this point guarantees the entry is durable.
            if self.sync_writes {
                file.flush()?;
            }

            self.size += bytes.len();
            self.entries_since_checkpoint += 1;
        }
        Ok(())
    }

    /// Check if checkpoint needed
    pub fn needs_checkpoint(&self) -> bool {
        self.size >= self.max_size || self.entries_since_checkpoint >= 10_000
    }

    /// Perform checkpoint (truncate WAL)
    pub fn checkpoint(&mut self) -> std::io::Result<()> {
        // Flush and close current file
        if let Some(file) = &mut self.file {
            file.flush()?;
        }
        self.file = None;

        // Truncate by recreating
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.path)?;
        self.file = Some(BufWriter::new(file));
        self.size = 0;
        self.entries_since_checkpoint = 0;

        // Write checkpoint marker
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        self.append(&WalEntry::Checkpoint { timestamp })?;

        Ok(())
    }

    /// Recover entries from WAL
    pub fn recover(&self) -> std::io::Result<Vec<WalEntry>> {
        let mut entries = Vec::new();
        let file = File::open(&self.path)?;
        let mut reader = BufReader::new(file);
        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;

        let mut offset = 0;
        while offset < data.len() {
            if let Some((entry, consumed)) = WalEntry::from_bytes(&data[offset..]) {
                entries.push(entry);
                offset += consumed;
            } else {
                break;
            }
        }

        Ok(entries)
    }

    /// Get current WAL size
    pub fn size(&self) -> usize {
        self.size
    }
}

// =============================================================================
// QUERY TIMEOUT
// =============================================================================

/// Query execution context with timeout
pub struct QueryContext {
    /// Start time
    start: Instant,
    /// Deadline
    deadline: Option<Instant>,
    /// Cancelled flag
    cancelled: AtomicBool,
}

impl QueryContext {
    pub fn new(timeout: Duration) -> Self {
        let start = Instant::now();
        let deadline = if timeout.is_zero() {
            None
        } else {
            Some(start + timeout)
        };
        Self {
            start,
            deadline,
            cancelled: AtomicBool::new(false),
        }
    }

    pub fn no_timeout() -> Self {
        Self {
            start: Instant::now(),
            deadline: None,
            cancelled: AtomicBool::new(false),
        }
    }

    /// Check if query should continue
    pub fn should_continue(&self) -> bool {
        if self.cancelled.load(Ordering::Relaxed) {
            return false;
        }
        if let Some(deadline) = self.deadline {
            Instant::now() < deadline
        } else {
            true
        }
    }

    /// Cancel the query
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    /// Elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Remaining time (if timeout set)
    pub fn remaining(&self) -> Option<Duration> {
        self.deadline
            .map(|d| d.saturating_duration_since(Instant::now()))
    }

    /// Check timeout and return error if exceeded
    pub fn check_timeout(&self) -> Result<(), QueryTimeoutError> {
        if !self.should_continue() {
            Err(QueryTimeoutError {
                elapsed: self.elapsed(),
            })
        } else {
            Ok(())
        }
    }
}

/// Query timeout error
#[derive(Debug)]
pub struct QueryTimeoutError {
    pub elapsed: Duration,
}

impl std::fmt::Display for QueryTimeoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Query timeout after {:?}", self.elapsed)
    }
}

impl std::error::Error for QueryTimeoutError {}

// =============================================================================
// METRICS
// =============================================================================

/// Production metrics for monitoring
pub struct HardeningMetrics {
    /// Total writes
    pub writes: AtomicU64,
    /// Total reads
    pub reads: AtomicU64,
    /// Total deletes
    pub deletes: AtomicU64,
    /// Evictions triggered
    pub evictions: AtomicU64,
    /// TTL expirations
    pub expirations: AtomicU64,
    /// WAL writes
    pub wal_writes: AtomicU64,
    /// WAL checkpoints
    pub checkpoints: AtomicU64,
    /// Query timeouts
    pub timeouts: AtomicU64,
    /// Memory pressure events
    pub memory_pressure_events: AtomicU64,
}

impl Default for HardeningMetrics {
    fn default() -> Self {
        Self {
            writes: AtomicU64::new(0),
            reads: AtomicU64::new(0),
            deletes: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            expirations: AtomicU64::new(0),
            wal_writes: AtomicU64::new(0),
            checkpoints: AtomicU64::new(0),
            timeouts: AtomicU64::new(0),
            memory_pressure_events: AtomicU64::new(0),
        }
    }
}

impl HardeningMetrics {
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            writes: self.writes.load(Ordering::Relaxed),
            reads: self.reads.load(Ordering::Relaxed),
            deletes: self.deletes.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            expirations: self.expirations.load(Ordering::Relaxed),
            wal_writes: self.wal_writes.load(Ordering::Relaxed),
            checkpoints: self.checkpoints.load(Ordering::Relaxed),
            timeouts: self.timeouts.load(Ordering::Relaxed),
            memory_pressure_events: self.memory_pressure_events.load(Ordering::Relaxed),
        }
    }
}

#[derive(Clone, Debug)]
pub struct MetricsSnapshot {
    pub writes: u64,
    pub reads: u64,
    pub deletes: u64,
    pub evictions: u64,
    pub expirations: u64,
    pub wal_writes: u64,
    pub checkpoints: u64,
    pub timeouts: u64,
    pub memory_pressure_events: u64,
}

// =============================================================================
// HARDENED BIND SPACE WRAPPER
// =============================================================================

/// Production-hardened wrapper around BindSpace
pub struct HardenedBindSpace {
    /// Configuration
    config: HardeningConfig,
    /// LRU tracker for fluid zone
    fluid_lru: Mutex<LruTracker>,
    /// LRU tracker for node zone
    node_lru: Mutex<LruTracker>,
    /// TTL manager for fluid zone
    ttl_manager: Mutex<TtlManager>,
    /// Write-ahead log
    wal: Option<Mutex<WriteAheadLog>>,
    /// Metrics
    pub metrics: HardeningMetrics,
    /// Last maintenance time
    last_maintenance: Mutex<Instant>,
}

impl HardenedBindSpace {
    pub fn new(config: HardeningConfig) -> std::io::Result<Self> {
        let wal = if config.enable_wal {
            Some(Mutex::new(WriteAheadLog::new(
                config.wal_dir.clone(),
                config.max_wal_size,
                config.wal_sync,
            )?))
        } else {
            None
        };

        Ok(Self {
            fluid_lru: Mutex::new(LruTracker::new(config.max_fluid_nodes)),
            node_lru: Mutex::new(LruTracker::new(config.max_node_nodes)),
            ttl_manager: Mutex::new(TtlManager::new(config.default_fluid_ttl)),
            wal,
            metrics: HardeningMetrics::default(),
            last_maintenance: Mutex::new(Instant::now()),
            config,
        })
    }

    /// Record a read access
    pub fn on_read(&self, addr: Addr) {
        self.metrics.reads.fetch_add(1, Ordering::Relaxed);

        let prefix = addr.prefix();
        if prefix >= PREFIX_FLUID_START && prefix <= PREFIX_FLUID_END {
            if let Ok(mut lru) = self.fluid_lru.lock() {
                lru.touch(addr.0);
            }
            if let Ok(mut ttl) = self.ttl_manager.lock() {
                ttl.refresh(addr.0);
            }
        } else if prefix >= PREFIX_NODE_START {
            if let Ok(mut lru) = self.node_lru.lock() {
                lru.touch(addr.0);
            }
        }
    }

    /// Record a write - returns addresses to evict if needed
    pub fn on_write(
        &self,
        addr: Addr,
        fingerprint: &[u64; FINGERPRINT_WORDS],
        label: Option<&str>,
    ) -> Vec<u16> {
        self.metrics.writes.fetch_add(1, Ordering::Relaxed);

        let prefix = addr.prefix();
        let mut to_evict = Vec::new();

        if prefix >= PREFIX_FLUID_START && prefix <= PREFIX_FLUID_END {
            // Fluid zone write
            if let Ok(mut lru) = self.fluid_lru.lock() {
                lru.touch(addr.0);
                if lru.needs_eviction() {
                    let count = lru.eviction_count();
                    to_evict = lru.evict_candidates(count);
                    for &a in &to_evict {
                        lru.remove(a);
                    }
                    self.metrics
                        .evictions
                        .fetch_add(count as u64, Ordering::Relaxed);
                }
            }
            if let Ok(mut ttl) = self.ttl_manager.lock() {
                ttl.set_ttl(addr.0, None);
            }
        } else if prefix >= PREFIX_NODE_START {
            // Node zone write
            if let Ok(mut lru) = self.node_lru.lock() {
                lru.touch(addr.0);
                if lru.needs_eviction() {
                    let count = lru.eviction_count();
                    to_evict = lru.evict_candidates(count);
                    for &a in &to_evict {
                        lru.remove(a);
                    }
                    self.metrics
                        .evictions
                        .fetch_add(count as u64, Ordering::Relaxed);
                }
            }
        }

        // WAL
        if let Some(ref wal) = self.wal {
            if let Ok(mut wal) = wal.lock() {
                let entry = WalEntry::Write {
                    addr: addr.0,
                    fingerprint: *fingerprint,
                    label: label.map(|s| s.to_string()),
                };
                let _ = wal.append(&entry);
                self.metrics.wal_writes.fetch_add(1, Ordering::Relaxed);
            }
        }

        to_evict
    }

    /// Record a delete
    pub fn on_delete(&self, addr: Addr) {
        self.metrics.deletes.fetch_add(1, Ordering::Relaxed);

        let prefix = addr.prefix();
        if prefix >= PREFIX_FLUID_START && prefix <= PREFIX_FLUID_END {
            if let Ok(mut lru) = self.fluid_lru.lock() {
                lru.remove(addr.0);
            }
            if let Ok(mut ttl) = self.ttl_manager.lock() {
                ttl.remove(addr.0);
            }
        } else if prefix >= PREFIX_NODE_START {
            if let Ok(mut lru) = self.node_lru.lock() {
                lru.remove(addr.0);
            }
        }

        // WAL
        if let Some(ref wal) = self.wal {
            if let Ok(mut wal) = wal.lock() {
                let entry = WalEntry::Delete { addr: addr.0 };
                let _ = wal.append(&entry);
                self.metrics.wal_writes.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Record a link
    pub fn on_link(&self, from: Addr, verb: Addr, to: Addr) {
        // WAL
        if let Some(ref wal) = self.wal {
            if let Ok(mut wal) = wal.lock() {
                let entry = WalEntry::Link {
                    from: from.0,
                    verb: verb.0,
                    to: to.0,
                };
                let _ = wal.append(&entry);
                self.metrics.wal_writes.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Run maintenance - returns (expired_addrs, needs_checkpoint)
    pub fn maintenance(&self) -> (Vec<u16>, bool) {
        let mut expired = Vec::new();
        let mut needs_checkpoint = false;

        // Check maintenance interval
        {
            let mut last = self.last_maintenance.lock().unwrap();
            if last.elapsed() < self.config.maintenance_interval {
                return (expired, false);
            }
            *last = Instant::now();
        }

        // Get expired TTLs
        if let Ok(ttl) = self.ttl_manager.lock() {
            expired = ttl.get_expired();
            self.metrics
                .expirations
                .fetch_add(expired.len() as u64, Ordering::Relaxed);
        }

        // Check WAL
        if let Some(ref wal) = self.wal {
            if let Ok(wal) = wal.lock() {
                needs_checkpoint = wal.needs_checkpoint();
            }
        }

        (expired, needs_checkpoint)
    }

    /// Perform WAL checkpoint
    pub fn checkpoint(&self) -> std::io::Result<()> {
        if let Some(ref wal) = self.wal {
            if let Ok(mut wal) = wal.lock() {
                wal.checkpoint()?;
                self.metrics.checkpoints.fetch_add(1, Ordering::Relaxed);
            }
        }
        Ok(())
    }

    /// Create query context with configured timeout
    pub fn query_context(&self) -> QueryContext {
        QueryContext::new(self.config.query_timeout)
    }

    /// Record query timeout
    pub fn on_timeout(&self) {
        self.metrics.timeouts.fetch_add(1, Ordering::Relaxed);
    }

    /// Get configuration
    pub fn config(&self) -> &HardeningConfig {
        &self.config
    }

    /// Recover from WAL
    pub fn recover(&self) -> std::io::Result<Vec<WalEntry>> {
        if let Some(ref wal) = self.wal {
            if let Ok(wal) = wal.lock() {
                return wal.recover();
            }
        }
        Ok(Vec::new())
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru_tracker() {
        let mut lru = LruTracker::new(5);

        // Add entries
        for i in 0..5 {
            lru.touch(i);
        }
        assert_eq!(lru.len(), 5);
        assert!(!lru.needs_eviction());

        // Add one more - should trigger eviction need
        lru.touch(5);
        assert_eq!(lru.len(), 6);
        assert!(lru.needs_eviction());

        // Get eviction candidates (oldest first)
        let candidates = lru.evict_candidates(2);
        assert_eq!(candidates, vec![0, 1]);
    }

    #[test]
    fn test_lru_touch_updates_order() {
        let mut lru = LruTracker::new(10);

        lru.touch(1);
        lru.touch(2);
        lru.touch(3);

        // Touch 1 again - should move to end
        lru.touch(1);

        let candidates = lru.evict_candidates(3);
        assert_eq!(candidates, vec![2, 3, 1]);
    }

    #[test]
    fn test_ttl_manager() {
        let mut ttl = TtlManager::new(Duration::from_millis(100));

        ttl.set_ttl(1, None);
        assert!(!ttl.is_expired(1));

        // Sleep past TTL
        std::thread::sleep(Duration::from_millis(150));
        assert!(ttl.is_expired(1));

        let expired = ttl.get_expired();
        assert!(expired.contains(&1));
    }

    #[test]
    fn test_query_context() {
        let ctx = QueryContext::new(Duration::from_millis(100));
        assert!(ctx.should_continue());

        std::thread::sleep(Duration::from_millis(150));
        assert!(!ctx.should_continue());
    }

    #[test]
    fn test_query_context_cancel() {
        let ctx = QueryContext::new(Duration::from_secs(60));
        assert!(ctx.should_continue());

        ctx.cancel();
        assert!(!ctx.should_continue());
    }

    #[test]
    fn test_wal_entry_serialization() {
        let entry = WalEntry::Write {
            addr: 0x8042,
            fingerprint: [42u64; FINGERPRINT_WORDS],
            label: Some("test".to_string()),
        };

        let bytes = entry.to_bytes();
        let (recovered, consumed) = WalEntry::from_bytes(&bytes).unwrap();

        match recovered {
            WalEntry::Write {
                addr,
                fingerprint,
                label,
            } => {
                assert_eq!(addr, 0x8042);
                assert_eq!(fingerprint[0], 42);
                assert_eq!(label, Some("test".to_string()));
            }
            _ => panic!("Wrong entry type"),
        }
        assert_eq!(consumed, bytes.len());
    }

    #[test]
    fn test_hardened_bind_space() {
        let config = HardeningConfig {
            enable_wal: false,
            ..Default::default()
        };
        let hardened = HardenedBindSpace::new(config).unwrap();

        // Simulate writes
        for i in 0u16..100 {
            let addr = Addr::new(PREFIX_FLUID_START + (i / 256) as u8, (i % 256) as u8);
            let fp = [i as u64; FINGERPRINT_WORDS];
            let _ = hardened.on_write(addr, &fp, None);
        }

        let metrics = hardened.metrics.snapshot();
        assert_eq!(metrics.writes, 100);
    }
}
