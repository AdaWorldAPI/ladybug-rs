//! Snapshots module - VMware-like XOR delta snapshots and cold storage
//!
//! # XOR Delta Snapshots
//!
//! Like VMware's VMDK snapshots, we use XOR deltas to efficiently store
//! differences between versions. Only changed bits are stored.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                         XOR DELTA CHAIN                                     │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │  Base Snapshot (full)                                                       │
//! │       ↓                                                                     │
//! │  Delta 1: XOR(v1, v0) → only changed bits stored                           │
//! │       ↓                                                                     │
//! │  Delta 2: XOR(v2, v1) → sparse, typically <5% of base                      │
//! │       ↓                                                                     │
//! │  Delta N: XOR(vN, vN-1) → chain until consolidation                        │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Cold Storage Tier
//!
//! Inactive data automatically migrates to cold storage:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │  HOT TIER (RAM)          │  WARM TIER (SSD)        │  COLD TIER (Archive)  │
//! │  - Active working set    │  - Recent snapshots     │  - XOR-compressed     │
//! │  - Sub-microsecond       │  - Millisecond access   │  - B-tree indexed     │
//! │  - Full fidelity         │  - Delta chains         │  - Transparent decomp │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, RwLock};
use std::time::{Duration, Instant};

// ============================================================================
// XOR Delta Types
// ============================================================================

/// Snapshot ID - monotonically increasing
pub type SnapshotId = u64;

/// A single XOR delta block
#[derive(Debug, Clone)]
pub struct DeltaBlock {
    /// Address this delta applies to
    pub addr: u16,
    /// XOR mask - apply to previous version to get current
    pub xor_mask: Vec<u8>,
    /// Number of bits changed (for statistics)
    pub bits_changed: u32,
}

impl DeltaBlock {
    /// Create a delta block from two versions
    pub fn from_diff(addr: u16, old: &[u8], new: &[u8]) -> Option<Self> {
        if old == new {
            return None; // No change
        }

        let max_len = old.len().max(new.len());
        let mut xor_mask = vec![0u8; max_len];
        let mut bits_changed = 0u32;

        for i in 0..max_len {
            let old_byte = old.get(i).copied().unwrap_or(0);
            let new_byte = new.get(i).copied().unwrap_or(0);
            let xor = old_byte ^ new_byte;
            xor_mask[i] = xor;
            bits_changed += xor.count_ones();
        }

        // Trim trailing zeros for compression
        while xor_mask.last() == Some(&0) && !xor_mask.is_empty() {
            xor_mask.pop();
        }

        if xor_mask.is_empty() {
            return None;
        }

        Some(Self {
            addr,
            xor_mask,
            bits_changed,
        })
    }

    /// Apply this delta to a previous version
    pub fn apply(&self, previous: &mut [u8]) {
        for (i, &xor) in self.xor_mask.iter().enumerate() {
            if i < previous.len() {
                previous[i] ^= xor;
            }
        }
    }

    /// Reverse apply (go backwards in time)
    pub fn unapply(&self, current: &mut [u8]) {
        // XOR is its own inverse
        self.apply(current);
    }

    /// Compressed size in bytes
    pub fn compressed_size(&self) -> usize {
        // addr (2) + len (4) + mask + bits_changed (4)
        2 + 4 + self.xor_mask.len() + 4
    }
}

/// A complete snapshot - either full or delta
#[derive(Debug, Clone)]
pub enum Snapshot {
    /// Full snapshot - stores complete state
    Full {
        id: SnapshotId,
        created_at: u64,
        /// Address -> serialized data
        data: HashMap<u16, Vec<u8>>,
        /// Total size in bytes
        size_bytes: usize,
    },
    /// Delta snapshot - stores only XOR differences from parent
    Delta {
        id: SnapshotId,
        parent_id: SnapshotId,
        created_at: u64,
        /// Delta blocks for changed addresses
        deltas: Vec<DeltaBlock>,
        /// Addresses that were deleted
        deleted: Vec<u16>,
        /// Addresses that were created (full data stored)
        created: HashMap<u16, Vec<u8>>,
        /// Compression ratio vs full snapshot
        compression_ratio: f32,
    },
}

impl Snapshot {
    /// Get the snapshot ID
    pub fn id(&self) -> SnapshotId {
        match self {
            Snapshot::Full { id, .. } => *id,
            Snapshot::Delta { id, .. } => *id,
        }
    }

    /// Get creation timestamp
    pub fn created_at(&self) -> u64 {
        match self {
            Snapshot::Full { created_at, .. } => *created_at,
            Snapshot::Delta { created_at, .. } => *created_at,
        }
    }

    /// Approximate size in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            Snapshot::Full { size_bytes, .. } => *size_bytes,
            Snapshot::Delta {
                deltas,
                created,
                deleted,
                ..
            } => {
                let delta_size: usize = deltas.iter().map(|d| d.compressed_size()).sum();
                let created_size: usize = created.values().map(|v| v.len() + 2).sum();
                let deleted_size = deleted.len() * 2;
                delta_size + created_size + deleted_size + 32 // header overhead
            }
        }
    }

    /// Is this a full snapshot?
    pub fn is_full(&self) -> bool {
        matches!(self, Snapshot::Full { .. })
    }
}

// ============================================================================
// Snapshot Chain
// ============================================================================

/// A chain of snapshots with efficient reconstruction
#[derive(Debug)]
pub struct SnapshotChain {
    /// All snapshots in order
    snapshots: BTreeMap<SnapshotId, Snapshot>,
    /// Next snapshot ID
    next_id: AtomicU64,
    /// Maximum delta chain length before consolidation
    max_chain_length: usize,
    /// Current chain length since last full snapshot
    chain_length: AtomicU64,
}

impl SnapshotChain {
    /// Create a new snapshot chain
    pub fn new(max_chain_length: usize) -> Self {
        Self {
            snapshots: BTreeMap::new(),
            next_id: AtomicU64::new(1),
            max_chain_length,
            chain_length: AtomicU64::new(0),
        }
    }

    /// Create a full snapshot from current state
    pub fn create_full(&mut self, state: HashMap<u16, Vec<u8>>) -> SnapshotId {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let size_bytes: usize = state.values().map(|v| v.len() + 2).sum();

        let snapshot = Snapshot::Full {
            id,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            data: state,
            size_bytes,
        };

        self.snapshots.insert(id, snapshot);
        self.chain_length.store(0, Ordering::SeqCst);
        id
    }

    /// Create a delta snapshot from current state
    pub fn create_delta(
        &mut self,
        parent_id: SnapshotId,
        old_state: &HashMap<u16, Vec<u8>>,
        new_state: &HashMap<u16, Vec<u8>>,
    ) -> Result<SnapshotId, SnapshotError> {
        // Check if we need to consolidate first
        let chain_len = self.chain_length.load(Ordering::SeqCst) as usize;
        if chain_len >= self.max_chain_length {
            return Err(SnapshotError::ChainTooLong {
                length: chain_len,
                max: self.max_chain_length,
            });
        }

        // Verify parent exists
        if !self.snapshots.contains_key(&parent_id) {
            return Err(SnapshotError::ParentNotFound(parent_id));
        }

        let id = self.next_id.fetch_add(1, Ordering::SeqCst);

        // Compute deltas
        let mut deltas = Vec::new();
        let mut created = HashMap::new();
        let mut deleted = Vec::new();

        // Find modified and deleted entries
        for (addr, old_data) in old_state {
            match new_state.get(addr) {
                Some(new_data) => {
                    if let Some(delta) = DeltaBlock::from_diff(*addr, old_data, new_data) {
                        deltas.push(delta);
                    }
                }
                None => {
                    deleted.push(*addr);
                }
            }
        }

        // Find created entries
        for (addr, new_data) in new_state {
            if !old_state.contains_key(addr) {
                created.insert(*addr, new_data.clone());
            }
        }

        // Calculate compression ratio
        let old_size: usize = old_state.values().map(|v| v.len()).sum();
        let delta_size: usize = deltas.iter().map(|d| d.compressed_size()).sum();
        let created_size: usize = created.values().map(|v| v.len()).sum();
        let new_size = delta_size + created_size + deleted.len() * 2;
        let compression_ratio = if old_size > 0 {
            new_size as f32 / old_size as f32
        } else {
            1.0
        };

        let snapshot = Snapshot::Delta {
            id,
            parent_id,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            deltas,
            deleted,
            created,
            compression_ratio,
        };

        self.snapshots.insert(id, snapshot);
        self.chain_length.fetch_add(1, Ordering::SeqCst);
        Ok(id)
    }

    /// Reconstruct state at a given snapshot
    pub fn reconstruct(
        &self,
        snapshot_id: SnapshotId,
    ) -> Result<HashMap<u16, Vec<u8>>, SnapshotError> {
        // Build chain from snapshot back to full snapshot
        let mut chain = Vec::new();
        let mut current_id = snapshot_id;

        loop {
            let snapshot = self
                .snapshots
                .get(&current_id)
                .ok_or(SnapshotError::NotFound(current_id))?;

            match snapshot {
                Snapshot::Full { data, .. } => {
                    // Start with full snapshot
                    let mut state = data.clone();

                    // Apply deltas in order
                    for delta_snapshot in chain.into_iter().rev() {
                        self.apply_delta(&mut state, delta_snapshot)?;
                    }

                    return Ok(state);
                }
                Snapshot::Delta { parent_id, .. } => {
                    chain.push(snapshot.clone());
                    current_id = *parent_id;
                }
            }
        }
    }

    /// Apply a delta snapshot to state
    fn apply_delta(
        &self,
        state: &mut HashMap<u16, Vec<u8>>,
        delta: Snapshot,
    ) -> Result<(), SnapshotError> {
        if let Snapshot::Delta {
            deltas,
            deleted,
            created,
            ..
        } = delta
        {
            // Apply XOR deltas
            for delta_block in deltas {
                if let Some(data) = state.get_mut(&delta_block.addr) {
                    delta_block.apply(data);
                }
            }

            // Remove deleted entries
            for addr in deleted {
                state.remove(&addr);
            }

            // Add created entries
            for (addr, data) in created {
                state.insert(addr, data);
            }
        }
        Ok(())
    }

    /// Consolidate chain into a new full snapshot
    pub fn consolidate(&mut self, snapshot_id: SnapshotId) -> Result<SnapshotId, SnapshotError> {
        let state = self.reconstruct(snapshot_id)?;
        let new_id = self.create_full(state);
        Ok(new_id)
    }

    /// Get statistics about the chain
    pub fn stats(&self) -> SnapshotStats {
        let total_snapshots = self.snapshots.len();
        let full_snapshots = self.snapshots.values().filter(|s| s.is_full()).count();
        let delta_snapshots = total_snapshots - full_snapshots;

        let total_size: usize = self.snapshots.values().map(|s| s.size_bytes()).sum();
        let full_size: usize = self
            .snapshots
            .values()
            .filter(|s| s.is_full())
            .map(|s| s.size_bytes())
            .sum();

        let avg_compression = if delta_snapshots > 0 {
            self.snapshots
                .values()
                .filter_map(|s| match s {
                    Snapshot::Delta {
                        compression_ratio, ..
                    } => Some(*compression_ratio),
                    _ => None,
                })
                .sum::<f32>()
                / delta_snapshots as f32
        } else {
            1.0
        };

        SnapshotStats {
            total_snapshots,
            full_snapshots,
            delta_snapshots,
            total_size_bytes: total_size,
            full_size_bytes: full_size,
            delta_size_bytes: total_size - full_size,
            average_compression_ratio: avg_compression,
            chain_length: self.chain_length.load(Ordering::SeqCst) as usize,
            max_chain_length: self.max_chain_length,
        }
    }

    /// List all snapshot IDs
    pub fn list(&self) -> Vec<SnapshotId> {
        self.snapshots.keys().copied().collect()
    }

    /// Delete old snapshots, keeping at least one full and its dependents
    pub fn prune(&mut self, keep_after: u64) -> usize {
        let to_remove: Vec<SnapshotId> = self
            .snapshots
            .iter()
            .filter(|(_, s)| s.created_at() < keep_after)
            .map(|(id, _)| *id)
            .collect();

        // Don't remove if it would orphan deltas
        let mut removed = 0;
        for id in to_remove {
            // Check if any delta depends on this
            let has_dependents = self
                .snapshots
                .values()
                .any(|s| matches!(s, Snapshot::Delta { parent_id, .. } if *parent_id == id));

            if !has_dependents {
                self.snapshots.remove(&id);
                removed += 1;
            }
        }

        removed
    }
}

/// Snapshot chain statistics
#[derive(Debug, Clone)]
pub struct SnapshotStats {
    pub total_snapshots: usize,
    pub full_snapshots: usize,
    pub delta_snapshots: usize,
    pub total_size_bytes: usize,
    pub full_size_bytes: usize,
    pub delta_size_bytes: usize,
    pub average_compression_ratio: f32,
    pub chain_length: usize,
    pub max_chain_length: usize,
}

// ============================================================================
// Cold Storage Tier
// ============================================================================

/// Storage tier classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageTier {
    /// Hot tier - in RAM, sub-microsecond access
    Hot,
    /// Warm tier - on SSD, millisecond access
    Warm,
    /// Cold tier - archived, seconds to access
    Cold,
}

/// Configuration for cold storage
#[derive(Debug, Clone)]
pub struct ColdStorageConfig {
    /// Path to warm tier storage
    pub warm_path: PathBuf,
    /// Path to cold tier storage
    pub cold_path: PathBuf,
    /// Time before migrating to warm tier
    pub hot_to_warm_age: Duration,
    /// Time before migrating to cold tier
    pub warm_to_cold_age: Duration,
    /// Maximum hot tier size in bytes
    pub max_hot_size: usize,
    /// Maximum warm tier size in bytes
    pub max_warm_size: usize,
    /// Enable transparent decompression
    pub transparent_decompress: bool,
}

impl Default for ColdStorageConfig {
    fn default() -> Self {
        Self {
            warm_path: PathBuf::from("./data/warm"),
            cold_path: PathBuf::from("./data/cold"),
            hot_to_warm_age: Duration::from_secs(3600), // 1 hour
            warm_to_cold_age: Duration::from_secs(86400 * 7), // 1 week
            max_hot_size: 1024 * 1024 * 1024,           // 1 GB
            max_warm_size: 10 * 1024 * 1024 * 1024,     // 10 GB
            transparent_decompress: true,
        }
    }
}

/// Entry in the tiered storage system
#[derive(Debug, Clone)]
pub struct TieredEntry {
    pub addr: u16,
    pub data: Vec<u8>,
    pub tier: StorageTier,
    pub last_access: Instant,
    pub access_count: u64,
    pub created_at: u64,
}

/// Tiered storage manager
#[derive(Debug)]
pub struct TieredStorage {
    /// Configuration
    config: ColdStorageConfig,
    /// Hot tier entries (in memory)
    hot: RwLock<HashMap<u16, TieredEntry>>,
    /// Warm tier index (metadata only, data on disk)
    warm_index: RwLock<HashMap<u16, WarmMetadata>>,
    /// Cold tier index
    cold_index: RwLock<HashMap<u16, ColdMetadata>>,
    /// Current hot tier size
    hot_size: AtomicU64,
    /// Current warm tier size
    warm_size: AtomicU64,
    /// Migration queue
    migration_queue: Mutex<VecDeque<MigrationTask>>,
}

#[derive(Debug, Clone)]
struct WarmMetadata {
    file_path: PathBuf,
    offset: u64,
    length: u64,
    last_access: Instant,
}

#[derive(Debug, Clone)]
struct ColdMetadata {
    file_path: PathBuf,
    offset: u64,
    length: u64,
    compressed: bool,
    original_size: u64,
}

#[derive(Debug)]
struct MigrationTask {
    addr: u16,
    from_tier: StorageTier,
    to_tier: StorageTier,
}

impl TieredStorage {
    /// Create a new tiered storage manager
    pub fn new(config: ColdStorageConfig) -> io::Result<Self> {
        // Create directories if they don't exist
        fs::create_dir_all(&config.warm_path)?;
        fs::create_dir_all(&config.cold_path)?;

        Ok(Self {
            config,
            hot: RwLock::new(HashMap::new()),
            warm_index: RwLock::new(HashMap::new()),
            cold_index: RwLock::new(HashMap::new()),
            hot_size: AtomicU64::new(0),
            warm_size: AtomicU64::new(0),
            migration_queue: Mutex::new(VecDeque::new()),
        })
    }

    /// Write data to hot tier
    pub fn write(&self, addr: u16, data: Vec<u8>) -> Result<(), SnapshotError> {
        let size = data.len() as u64;

        // Check hot tier capacity
        let current_size = self.hot_size.load(Ordering::SeqCst);
        if current_size + size > self.config.max_hot_size as u64 {
            // Need to evict something first
            self.evict_from_hot(size as usize)?;
        }

        let entry = TieredEntry {
            addr,
            data,
            tier: StorageTier::Hot,
            last_access: Instant::now(),
            access_count: 0,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        let mut hot = self.hot.write().unwrap();
        if let Some(old) = hot.insert(addr, entry) {
            // Subtract old size, add new
            self.hot_size
                .fetch_sub(old.data.len() as u64, Ordering::SeqCst);
        }
        self.hot_size.fetch_add(size, Ordering::SeqCst);

        Ok(())
    }

    /// Read data from any tier (transparent)
    pub fn read(&self, addr: u16) -> Result<Vec<u8>, SnapshotError> {
        // Try hot tier first
        {
            let mut hot = self.hot.write().unwrap();
            if let Some(entry) = hot.get_mut(&addr) {
                entry.last_access = Instant::now();
                entry.access_count += 1;
                return Ok(entry.data.clone());
            }
        }

        // Try warm tier
        {
            let warm_index = self.warm_index.read().unwrap();
            if let Some(meta) = warm_index.get(&addr) {
                let data = self.read_from_file(&meta.file_path, meta.offset, meta.length)?;

                // Promote to hot if accessed frequently
                if self.config.transparent_decompress {
                    drop(warm_index);
                    let _ = self.promote_to_hot(addr, data.clone());
                }

                return Ok(data);
            }
        }

        // Try cold tier
        {
            let cold_index = self.cold_index.read().unwrap();
            if let Some(meta) = cold_index.get(&addr) {
                let data = self.read_from_file(&meta.file_path, meta.offset, meta.length)?;

                // Decompress if needed
                let data = if meta.compressed {
                    self.decompress(&data)?
                } else {
                    data
                };

                return Ok(data);
            }
        }

        Err(SnapshotError::NotFound(addr as u64))
    }

    /// Delete from all tiers
    pub fn delete(&self, addr: u16) -> Result<(), SnapshotError> {
        // Remove from hot
        {
            let mut hot = self.hot.write().unwrap();
            if let Some(entry) = hot.remove(&addr) {
                self.hot_size
                    .fetch_sub(entry.data.len() as u64, Ordering::SeqCst);
            }
        }

        // Remove from warm index (file cleanup happens in background)
        {
            let mut warm = self.warm_index.write().unwrap();
            warm.remove(&addr);
        }

        // Remove from cold index
        {
            let mut cold = self.cold_index.write().unwrap();
            cold.remove(&addr);
        }

        Ok(())
    }

    /// Get the current tier for an address
    pub fn tier_of(&self, addr: u16) -> Option<StorageTier> {
        if self.hot.read().unwrap().contains_key(&addr) {
            return Some(StorageTier::Hot);
        }
        if self.warm_index.read().unwrap().contains_key(&addr) {
            return Some(StorageTier::Warm);
        }
        if self.cold_index.read().unwrap().contains_key(&addr) {
            return Some(StorageTier::Cold);
        }
        None
    }

    /// Run periodic migration tick
    pub fn tick(&self) -> Result<MigrationStats, SnapshotError> {
        let mut stats = MigrationStats::default();
        let now = Instant::now();

        // Find entries to migrate from hot to warm
        let to_warm: Vec<(u16, TieredEntry)> = {
            let hot = self.hot.read().unwrap();
            hot.iter()
                .filter(|(_, e)| now.duration_since(e.last_access) > self.config.hot_to_warm_age)
                .map(|(addr, e)| (*addr, e.clone()))
                .collect()
        };

        for (addr, entry) in to_warm {
            if self.migrate_to_warm(addr, &entry.data).is_ok() {
                let mut hot = self.hot.write().unwrap();
                if let Some(removed) = hot.remove(&addr) {
                    self.hot_size
                        .fetch_sub(removed.data.len() as u64, Ordering::SeqCst);
                    stats.hot_to_warm += 1;
                }
            }
        }

        // Find entries to migrate from warm to cold
        let to_cold: Vec<(u16, WarmMetadata)> = {
            let warm = self.warm_index.read().unwrap();
            warm.iter()
                .filter(|(_, m)| now.duration_since(m.last_access) > self.config.warm_to_cold_age)
                .map(|(addr, m)| (*addr, m.clone()))
                .collect()
        };

        for (addr, meta) in to_cold {
            if let Ok(data) = self.read_from_file(&meta.file_path, meta.offset, meta.length) {
                if self.migrate_to_cold(addr, &data).is_ok() {
                    let mut warm = self.warm_index.write().unwrap();
                    warm.remove(&addr);
                    stats.warm_to_cold += 1;
                }
            }
        }

        Ok(stats)
    }

    /// Force an address to a specific tier
    pub fn force_tier(&self, addr: u16, target: StorageTier) -> Result<(), SnapshotError> {
        let data = self.read(addr)?;

        match target {
            StorageTier::Hot => {
                self.promote_to_hot(addr, data)?;
            }
            StorageTier::Warm => {
                self.migrate_to_warm(addr, &data)?;
                // Remove from hot if present
                let mut hot = self.hot.write().unwrap();
                if let Some(removed) = hot.remove(&addr) {
                    self.hot_size
                        .fetch_sub(removed.data.len() as u64, Ordering::SeqCst);
                }
            }
            StorageTier::Cold => {
                self.migrate_to_cold(addr, &data)?;
                // Remove from hot and warm
                {
                    let mut hot = self.hot.write().unwrap();
                    if let Some(removed) = hot.remove(&addr) {
                        self.hot_size
                            .fetch_sub(removed.data.len() as u64, Ordering::SeqCst);
                    }
                }
                {
                    let mut warm = self.warm_index.write().unwrap();
                    warm.remove(&addr);
                }
            }
        }

        Ok(())
    }

    /// Evict entries from hot tier to make room
    fn evict_from_hot(&self, needed: usize) -> Result<(), SnapshotError> {
        let mut freed = 0usize;

        // Sort by last access time, evict oldest first
        let candidates: Vec<(u16, Instant, usize)> = {
            let hot = self.hot.read().unwrap();
            let mut v: Vec<_> = hot
                .iter()
                .map(|(addr, e)| (*addr, e.last_access, e.data.len()))
                .collect();
            v.sort_by_key(|(_, t, _)| *t);
            v
        };

        for (addr, _, size) in candidates {
            if freed >= needed {
                break;
            }

            // Read and migrate to warm
            let data = {
                let hot = self.hot.read().unwrap();
                hot.get(&addr).map(|e| e.data.clone())
            };

            if let Some(data) = data {
                if self.migrate_to_warm(addr, &data).is_ok() {
                    let mut hot = self.hot.write().unwrap();
                    if let Some(removed) = hot.remove(&addr) {
                        self.hot_size
                            .fetch_sub(removed.data.len() as u64, Ordering::SeqCst);
                        freed += size;
                    }
                }
            }
        }

        Ok(())
    }

    /// Migrate data to warm tier
    fn migrate_to_warm(&self, addr: u16, data: &[u8]) -> Result<(), SnapshotError> {
        let file_path = self.config.warm_path.join(format!("{:04x}.warm", addr));
        let mut file =
            File::create(&file_path).map_err(|e| SnapshotError::IoError(e.to_string()))?;
        file.write_all(data)
            .map_err(|e| SnapshotError::IoError(e.to_string()))?;

        let meta = WarmMetadata {
            file_path,
            offset: 0,
            length: data.len() as u64,
            last_access: Instant::now(),
        };

        self.warm_index.write().unwrap().insert(addr, meta);
        self.warm_size
            .fetch_add(data.len() as u64, Ordering::SeqCst);

        Ok(())
    }

    /// Migrate data to cold tier (with compression)
    fn migrate_to_cold(&self, addr: u16, data: &[u8]) -> Result<(), SnapshotError> {
        // Simple RLE compression for cold storage
        let compressed = self.compress(data);
        let is_compressed = compressed.len() < data.len();
        let to_store = if is_compressed { &compressed } else { data };

        let file_path = self.config.cold_path.join(format!("{:04x}.cold", addr));
        let mut file =
            File::create(&file_path).map_err(|e| SnapshotError::IoError(e.to_string()))?;
        file.write_all(to_store)
            .map_err(|e| SnapshotError::IoError(e.to_string()))?;

        let meta = ColdMetadata {
            file_path,
            offset: 0,
            length: to_store.len() as u64,
            compressed: is_compressed,
            original_size: data.len() as u64,
        };

        self.cold_index.write().unwrap().insert(addr, meta);

        Ok(())
    }

    /// Promote data to hot tier
    fn promote_to_hot(&self, addr: u16, data: Vec<u8>) -> Result<(), SnapshotError> {
        self.write(addr, data)
    }

    /// Read from a file at a specific offset
    fn read_from_file(
        &self,
        path: &Path,
        offset: u64,
        length: u64,
    ) -> Result<Vec<u8>, SnapshotError> {
        let mut file = File::open(path).map_err(|e| SnapshotError::IoError(e.to_string()))?;

        use std::io::Seek;
        file.seek(std::io::SeekFrom::Start(offset))
            .map_err(|e| SnapshotError::IoError(e.to_string()))?;

        let mut data = vec![0u8; length as usize];
        file.read_exact(&mut data)
            .map_err(|e| SnapshotError::IoError(e.to_string()))?;

        Ok(data)
    }

    /// Simple RLE compression
    fn compress(&self, data: &[u8]) -> Vec<u8> {
        if data.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(data.len());
        let mut i = 0;

        while i < data.len() {
            let byte = data[i];
            let mut count = 1u8;

            while (i + count as usize) < data.len()
                && data[i + count as usize] == byte
                && count < 255
            {
                count += 1;
            }

            if count >= 3 || byte == 0xFF {
                // Use RLE: marker, count, byte
                result.push(0xFF);
                result.push(count);
                result.push(byte);
            } else {
                // Store literally
                for _ in 0..count {
                    result.push(byte);
                }
            }

            i += count as usize;
        }

        result
    }

    /// Decompress RLE data
    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>, SnapshotError> {
        let mut result = Vec::with_capacity(data.len() * 2);
        let mut i = 0;

        while i < data.len() {
            if data[i] == 0xFF && i + 2 < data.len() {
                let count = data[i + 1] as usize;
                let byte = data[i + 2];
                for _ in 0..count {
                    result.push(byte);
                }
                i += 3;
            } else {
                result.push(data[i]);
                i += 1;
            }
        }

        Ok(result)
    }

    /// Get storage statistics
    pub fn stats(&self) -> TieredStorageStats {
        let hot = self.hot.read().unwrap();
        let warm = self.warm_index.read().unwrap();
        let cold = self.cold_index.read().unwrap();

        TieredStorageStats {
            hot_entries: hot.len(),
            warm_entries: warm.len(),
            cold_entries: cold.len(),
            hot_size_bytes: self.hot_size.load(Ordering::SeqCst) as usize,
            warm_size_bytes: self.warm_size.load(Ordering::SeqCst) as usize,
            cold_size_bytes: cold.values().map(|m| m.length as usize).sum(),
        }
    }
}

/// Statistics from a migration tick
#[derive(Debug, Default, Clone)]
pub struct MigrationStats {
    pub hot_to_warm: usize,
    pub warm_to_cold: usize,
    pub promoted_to_hot: usize,
}

/// Overall tiered storage statistics
#[derive(Debug, Clone)]
pub struct TieredStorageStats {
    pub hot_entries: usize,
    pub warm_entries: usize,
    pub cold_entries: usize,
    pub hot_size_bytes: usize,
    pub warm_size_bytes: usize,
    pub cold_size_bytes: usize,
}

// ============================================================================
// Errors
// ============================================================================

/// Snapshot-related errors
#[derive(Debug, Clone)]
pub enum SnapshotError {
    NotFound(u64),
    ParentNotFound(SnapshotId),
    ChainTooLong { length: usize, max: usize },
    IoError(String),
    CompressionError(String),
}

impl std::fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "Snapshot not found: {}", id),
            Self::ParentNotFound(id) => write!(f, "Parent snapshot not found: {}", id),
            Self::ChainTooLong { length, max } => {
                write!(f, "Delta chain too long: {} (max {})", length, max)
            }
            Self::IoError(msg) => write!(f, "I/O error: {}", msg),
            Self::CompressionError(msg) => write!(f, "Compression error: {}", msg),
        }
    }
}

impl std::error::Error for SnapshotError {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_block_from_diff() {
        let old = vec![0x00, 0xFF, 0x55, 0xAA];
        let new = vec![0x00, 0xFE, 0x55, 0xAB]; // Bit 0 of byte 1 changed, bit 0 of byte 3 changed

        let delta = DeltaBlock::from_diff(42, &old, &new).unwrap();
        assert_eq!(delta.addr, 42);
        assert_eq!(delta.bits_changed, 2);

        // Apply delta to old, should get new
        let mut reconstructed = old.clone();
        delta.apply(&mut reconstructed);
        assert_eq!(reconstructed, new);
    }

    #[test]
    fn test_delta_block_no_change() {
        let data = vec![0x00, 0xFF, 0x55, 0xAA];
        let delta = DeltaBlock::from_diff(42, &data, &data);
        assert!(delta.is_none());
    }

    #[test]
    fn test_delta_reversible() {
        let old = vec![0x12, 0x34, 0x56, 0x78];
        let new = vec![0xAB, 0xCD, 0xEF, 0x00];

        let delta = DeltaBlock::from_diff(1, &old, &new).unwrap();

        // Forward: old -> new
        let mut forward = old.clone();
        delta.apply(&mut forward);
        assert_eq!(forward, new);

        // Backward: new -> old
        let mut backward = new.clone();
        delta.unapply(&mut backward);
        assert_eq!(backward, old);
    }

    #[test]
    fn test_snapshot_chain() {
        let mut chain = SnapshotChain::new(10);

        // Create initial state
        let mut state_v1 = HashMap::new();
        state_v1.insert(1u16, vec![0x00, 0x01, 0x02, 0x03]);
        state_v1.insert(2u16, vec![0xFF, 0xFE, 0xFD, 0xFC]);

        let snap1 = chain.create_full(state_v1.clone());

        // Modify state
        let mut state_v2 = state_v1.clone();
        state_v2.get_mut(&1).unwrap()[0] = 0x10; // Change one byte
        state_v2.insert(3u16, vec![0xAA, 0xBB]); // Add new entry

        let snap2 = chain.create_delta(snap1, &state_v1, &state_v2).unwrap();

        // Reconstruct and verify
        let reconstructed = chain.reconstruct(snap2).unwrap();
        assert_eq!(reconstructed, state_v2);

        // Original still works
        let original = chain.reconstruct(snap1).unwrap();
        assert_eq!(original, state_v1);
    }

    #[test]
    fn test_chain_too_long() {
        let mut chain = SnapshotChain::new(2); // Max 2 deltas

        let state = HashMap::new();
        let snap1 = chain.create_full(state.clone());
        let snap2 = chain.create_delta(snap1, &state, &state).unwrap();
        let snap3 = chain.create_delta(snap2, &state, &state).unwrap();

        // Third delta should fail
        let result = chain.create_delta(snap3, &state, &state);
        assert!(matches!(result, Err(SnapshotError::ChainTooLong { .. })));
    }

    #[test]
    fn test_consolidation() {
        let mut chain = SnapshotChain::new(5);

        let mut state = HashMap::new();
        state.insert(1u16, vec![0x00]);
        let snap1 = chain.create_full(state.clone());

        state.insert(1u16, vec![0x01]);
        let snap2 = chain.create_delta(snap1, &HashMap::new(), &state).unwrap();

        state.insert(1u16, vec![0x02]);
        let snap3 = chain
            .create_delta(snap2, &HashMap::from([(1u16, vec![0x01])]), &state)
            .unwrap();

        // Consolidate into new full snapshot
        let consolidated = chain.consolidate(snap3).unwrap();

        // Verify consolidated snapshot has correct state
        let recon = chain.reconstruct(consolidated).unwrap();
        assert_eq!(recon.get(&1), Some(&vec![0x02]));

        // Chain length should be reset
        assert_eq!(chain.chain_length.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_snapshot_stats() {
        let mut chain = SnapshotChain::new(10);

        let state = HashMap::from([(1u16, vec![0u8; 100]), (2u16, vec![0u8; 100])]);

        chain.create_full(state.clone());
        chain.create_full(state.clone());

        let stats = chain.stats();
        assert_eq!(stats.total_snapshots, 2);
        assert_eq!(stats.full_snapshots, 2);
        assert_eq!(stats.delta_snapshots, 0);
    }

    #[test]
    fn test_rle_compression() {
        let storage = TieredStorage::new(ColdStorageConfig::default()).unwrap();

        // Data with runs
        let data = vec![0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF];
        let compressed = storage.compress(&data);
        let decompressed = storage.decompress(&compressed).unwrap();

        assert_eq!(decompressed, data);
        assert!(compressed.len() < data.len()); // Should be smaller
    }

    #[test]
    fn test_tiered_storage_write_read() {
        let storage = TieredStorage::new(ColdStorageConfig::default()).unwrap();

        let data = vec![0x01, 0x02, 0x03, 0x04];
        storage.write(42, data.clone()).unwrap();

        let read = storage.read(42).unwrap();
        assert_eq!(read, data);
        assert_eq!(storage.tier_of(42), Some(StorageTier::Hot));
    }

    #[test]
    fn test_tiered_storage_delete() {
        let storage = TieredStorage::new(ColdStorageConfig::default()).unwrap();

        storage.write(42, vec![0x01, 0x02]).unwrap();
        assert!(storage.tier_of(42).is_some());

        storage.delete(42).unwrap();
        assert!(storage.tier_of(42).is_none());
    }
}
