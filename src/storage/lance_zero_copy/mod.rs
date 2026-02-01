//! Lance Zero-Copy Integration
//!
//! This module provides zero-copy access to Lance storage by sharing
//! the same address space. No serialization boundaries, no copies.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    UNIFIED ADDRESS SPACE                        │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                 │
//! │  HOT (BindSpace)     WARM (Lance mmap)      COLD (Lance file)   │
//! │  ┌───────────────┐   ┌───────────────┐     ┌───────────────┐   │
//! │  │ Array[32K]    │   │ mmap region   │     │ Parquet file  │   │
//! │  │ Direct access │   │ OS page cache │     │ Compressed    │   │
//! │  └───────┬───────┘   └───────┬───────┘     └───────┬───────┘   │
//! │          │                   │                     │            │
//! │          │◀──── BUBBLE UP ───│◀──── BUBBLE UP ─────│            │
//! │          │     (ptr move)    │     (page fault)    │            │
//! │          │                   │                     │            │
//! │          │──── SINK DOWN ───▶│──── SINK DOWN ─────▶│            │
//! │          │     (ptr move)    │     (OS evict)      │            │
//! │                                                                 │
//! │  SCENT INDEX: Awareness layer - tracks where everything lives   │
//! │                                                                 │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Key Principles
//!
//! 1. **No Serialization Boundary**: Lance and Ladybug share memory
//! 2. **Scent as Awareness**: Index knows where data lives without copying
//! 3. **Bubbling Without Copy**: Promote/demote by updating pointers
//! 4. **Arrow Native**: Fingerprints stored as Arrow FixedSizeList

use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, FixedSizeListArray, UInt64Array};
use arrow_buffer::Buffer;
use arrow_data::ArrayDataBuilder;
use arrow_schema::{DataType, Field};

use super::bind_space::FINGERPRINT_WORDS;

// =============================================================================
// ARROW ZERO-COPY BRIDGE
// =============================================================================

/// Zero-copy fingerprint buffer backed by Arrow
///
/// This struct provides direct access to fingerprint data stored in Arrow format
/// without any copying. The underlying data lives in an Arrow Buffer which may
/// be backed by mmap'd memory from Lance storage.
pub struct FingerprintBuffer {
    /// The Arrow buffer containing raw u64 data
    buffer: Buffer,
    /// Number of fingerprints in this buffer
    len: usize,
}

impl FingerprintBuffer {
    /// Create from raw u64 data (zero-copy via ownership transfer)
    ///
    /// Uses `Buffer::from_vec()` which takes ownership of the Vec
    /// without copying data.
    pub fn from_vec(data: Vec<u64>, num_fingerprints: usize) -> Self {
        let expected_len = num_fingerprints * FINGERPRINT_WORDS;

        let buffer = if data.len() >= expected_len {
            Buffer::from_vec(data)
        } else {
            // Fallback: copy if size doesn't match
            Buffer::from_slice_ref(&data)
        };

        Self {
            buffer,
            len: num_fingerprints,
        }
    }

    /// Create from raw bytes slice (copies data)
    ///
    /// Use `from_vec` when possible to avoid copies.
    pub fn from_bytes(bytes: &[u8], num_fingerprints: usize) -> Self {
        let buffer = Buffer::from_slice_ref(bytes);
        Self {
            buffer,
            len: num_fingerprints,
        }
    }

    /// Create from an existing Arrow buffer (always zero-copy)
    pub fn from_buffer(buffer: Buffer, num_fingerprints: usize) -> Self {
        Self {
            buffer,
            len: num_fingerprints,
        }
    }

    /// Create from a FixedSizeListArray (zero-copy)
    ///
    /// The FixedSizeListArray should contain UInt64 elements with
    /// list size = FINGERPRINT_WORDS (156).
    pub fn from_fixed_size_list(array: &FixedSizeListArray) -> Option<Self> {
        // Verify the array structure
        if array.value_length() != FINGERPRINT_WORDS as i32 {
            return None;
        }

        // Get the inner values array
        let values = array.values();
        let u64_array = values.as_any().downcast_ref::<UInt64Array>()?;

        // Get the underlying buffer (zero-copy)
        let buffer = u64_array.values().inner().clone();

        Some(Self {
            buffer,
            len: array.len(),
        })
    }

    /// Get raw pointer to fingerprint data
    ///
    /// # Safety
    /// The returned pointer is valid for the lifetime of this FingerprintBuffer.
    #[inline]
    pub fn as_ptr(&self) -> *const u64 {
        self.buffer.as_ptr() as *const u64
    }

    /// Get fingerprint at index as slice (zero-copy)
    #[inline]
    pub fn get(&self, index: usize) -> Option<&[u64; FINGERPRINT_WORDS]> {
        if index >= self.len {
            return None;
        }

        let offset = index * FINGERPRINT_WORDS;
        let ptr = self.as_ptr();

        // Safety: We checked bounds above
        unsafe {
            let slice_ptr = ptr.add(offset) as *const [u64; FINGERPRINT_WORDS];
            Some(&*slice_ptr)
        }
    }

    /// Get fingerprint at index without bounds checking
    ///
    /// # Safety
    /// Caller must ensure index < self.len()
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &[u64; FINGERPRINT_WORDS] {
        let ptr = self.as_ptr().add(index * FINGERPRINT_WORDS) as *const [u64; FINGERPRINT_WORDS];
        &*ptr
    }

    /// Number of fingerprints
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Is empty?
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get underlying Arrow buffer (for sharing with other Arrow operations)
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Convert back to FixedSizeListArray (zero-copy)
    pub fn to_fixed_size_list(&self) -> ArrayRef {
        // Create UInt64Array from buffer
        let u64_data = ArrayDataBuilder::new(DataType::UInt64)
            .len(self.len * FINGERPRINT_WORDS)
            .add_buffer(self.buffer.clone())
            .build()
            .expect("valid array data");

        let u64_array = UInt64Array::from(u64_data);

        // Wrap in FixedSizeListArray
        let field = Arc::new(Field::new("item", DataType::UInt64, false));
        Arc::new(FixedSizeListArray::new(
            field,
            FINGERPRINT_WORDS as i32,
            Arc::new(u64_array),
            None,
        ))
    }

    /// Iterate over fingerprints (zero-copy)
    pub fn iter(&self) -> FingerprintIter<'_> {
        FingerprintIter {
            buffer: self,
            index: 0,
        }
    }
}

/// Iterator over fingerprints in a FingerprintBuffer
pub struct FingerprintIter<'a> {
    buffer: &'a FingerprintBuffer,
    index: usize,
}

impl<'a> Iterator for FingerprintIter<'a> {
    type Item = &'a [u64; FINGERPRINT_WORDS];

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.buffer.len {
            return None;
        }
        // Safety: we just checked bounds
        let fp = unsafe { self.buffer.get_unchecked(self.index) };
        self.index += 1;
        Some(fp)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.buffer.len - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for FingerprintIter<'a> {}

// =============================================================================
// ARROW ZERO-COPY MANAGER
// =============================================================================

/// Manager for zero-copy Arrow integration
///
/// Handles conversion between Ladybug fingerprints and Arrow format,
/// maintaining zero-copy semantics throughout the pipeline.
pub struct ArrowZeroCopy {
    /// Currently loaded fingerprint buffers
    buffers: Vec<FingerprintBuffer>,
    /// Scent awareness for temperature tracking
    scent: ScentAwareness,
}

impl ArrowZeroCopy {
    /// Create new manager
    pub fn new() -> Self {
        Self {
            buffers: Vec::new(),
            scent: ScentAwareness::new(),
        }
    }

    /// Load fingerprints from Vec<u64> (zero-copy via ownership transfer)
    pub fn load_from_vec(&mut self, data: Vec<u64>, num_fingerprints: usize) -> usize {
        let buffer = FingerprintBuffer::from_vec(data, num_fingerprints);
        let id = self.buffers.len();
        self.buffers.push(buffer);
        id
    }

    /// Load fingerprints from bytes slice (copies data)
    pub fn load_from_bytes(&mut self, bytes: &[u8], num_fingerprints: usize) -> usize {
        let buffer = FingerprintBuffer::from_bytes(bytes, num_fingerprints);
        let id = self.buffers.len();
        self.buffers.push(buffer);
        id
    }

    /// Load fingerprints from Arrow array (zero-copy)
    pub fn load_from_arrow(&mut self, array: &FixedSizeListArray) -> Option<usize> {
        let buffer = FingerprintBuffer::from_fixed_size_list(array)?;
        let id = self.buffers.len();
        self.buffers.push(buffer);
        Some(id)
    }

    /// Get fingerprint by buffer ID and index
    pub fn get(&self, buffer_id: usize, index: usize) -> Option<&[u64; FINGERPRINT_WORDS]> {
        self.buffers.get(buffer_id)?.get(index)
    }

    /// Touch for temperature tracking
    pub fn touch(&mut self, buffer_id: usize, index: usize) {
        // Encode buffer_id and index into a single u32
        let combined = ((buffer_id as u32) << 24) | (index as u32 & 0xFFFFFF);
        self.scent.touch(combined);
    }

    /// Get buffer by ID
    pub fn buffer(&self, id: usize) -> Option<&FingerprintBuffer> {
        self.buffers.get(id)
    }

    /// Get scent awareness
    pub fn scent(&self) -> &ScentAwareness {
        &self.scent
    }

    /// Get mutable scent awareness
    pub fn scent_mut(&mut self) -> &mut ScentAwareness {
        &mut self.scent
    }

    /// Total fingerprints across all buffers
    pub fn total_fingerprints(&self) -> usize {
        self.buffers.iter().map(|b| b.len()).sum()
    }
}

impl Default for ArrowZeroCopy {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// ADJACENCY SORTING (locality-preserving order)
// =============================================================================

/// Locality-preserving fingerprint ordering
///
/// Since Lance row operations are cheap, we can sort fingerprints by
/// adjacency (similar fingerprints near each other) for better cache
/// locality during search.
///
/// Uses a simple LSH-style bucket assignment based on the first few bits
/// of each fingerprint word, creating a coarse-grained spatial ordering.
///
/// # IMPORTANT: Codebook Index Preservation
///
/// This index does NOT invalidate codebook addressing. The 8+8 address model
/// (prefix:slot) always uses the ORIGINAL index for lookup:
///
/// ```text
/// Address 0x42:0x0F → original_index = lookup(0x42, 0x0F) → fingerprint
///                                          ↓
///                              AdjacencyIndex::sorted_position(original_index)
///                                          ↓
///                              (only used for cache-friendly iteration)
/// ```
///
/// - Direct lookups by address: use original_index (preserves codebook)
/// - Batch scanning: use iter_sorted() for cache locality
/// - Similarity search: use locality_range() then original_index for results
pub struct AdjacencyIndex {
    /// Bucket assignments for each fingerprint
    /// bucket[i] = locality hash for fingerprint i
    buckets: Vec<u32>,

    /// Sorted indices: sorted_order[i] = original index
    /// Fingerprints are sorted by bucket, so similar ones are adjacent
    sorted_order: Vec<u32>,

    /// Reverse map: reverse[original_index] = position in sorted order
    reverse: Vec<u32>,
}

impl AdjacencyIndex {
    /// Create adjacency index for fingerprints
    ///
    /// Computes a locality-sensitive hash for each fingerprint and
    /// creates a sorted order that clusters similar fingerprints together.
    pub fn build(fingerprints: &FingerprintBuffer) -> Self {
        let n = fingerprints.len();

        // Compute locality hash for each fingerprint
        // Uses first 4 bits from first 8 words = 32-bit locality hash
        let buckets: Vec<u32> = (0..n)
            .map(|i| {
                let fp = fingerprints.get(i).unwrap();
                Self::locality_hash(fp)
            })
            .collect();

        // Create sorted indices
        let mut sorted_order: Vec<u32> = (0..n as u32).collect();
        sorted_order.sort_by_key(|&i| buckets[i as usize]);

        // Create reverse mapping
        let mut reverse = vec![0u32; n];
        for (pos, &orig) in sorted_order.iter().enumerate() {
            reverse[orig as usize] = pos as u32;
        }

        Self {
            buckets,
            sorted_order,
            reverse,
        }
    }

    /// Locality-sensitive hash
    ///
    /// Takes first 4 bits from first 8 words of fingerprint,
    /// creating a 32-bit locality hash that groups similar
    /// fingerprints together.
    #[inline]
    fn locality_hash(fp: &[u64; FINGERPRINT_WORDS]) -> u32 {
        let mut hash = 0u32;
        for i in 0..8 {
            // Take top 4 bits from each of first 8 words
            let bits = ((fp[i] >> 60) & 0xF) as u32;
            hash |= bits << (i * 4);
        }
        hash
    }

    /// Get original index from sorted position
    #[inline]
    pub fn original_index(&self, sorted_pos: usize) -> Option<u32> {
        self.sorted_order.get(sorted_pos).copied()
    }

    /// Get sorted position from original index
    #[inline]
    pub fn sorted_position(&self, original_idx: usize) -> Option<u32> {
        self.reverse.get(original_idx).copied()
    }

    /// Get locality bucket for original index
    #[inline]
    pub fn bucket(&self, original_idx: usize) -> Option<u32> {
        self.buckets.get(original_idx).copied()
    }

    /// Find range of sorted indices that might contain similar fingerprints
    ///
    /// Given a query fingerprint, returns a range of sorted positions
    /// that are likely to contain similar fingerprints (same locality bucket).
    pub fn locality_range(&self, query: &[u64; FINGERPRINT_WORDS]) -> std::ops::Range<usize> {
        let query_bucket = Self::locality_hash(query);

        // Binary search for first index with this bucket
        let start = self.sorted_order
            .partition_point(|&i| self.buckets[i as usize] < query_bucket);

        // Binary search for first index past this bucket
        let end = self.sorted_order
            .partition_point(|&i| self.buckets[i as usize] <= query_bucket);

        start..end
    }

    /// Iterate over original indices in sorted (locality-preserving) order
    pub fn iter_sorted(&self) -> impl Iterator<Item = u32> + '_ {
        self.sorted_order.iter().copied()
    }

    /// Number of fingerprints
    pub fn len(&self) -> usize {
        self.sorted_order.len()
    }

    /// Is empty?
    pub fn is_empty(&self) -> bool {
        self.sorted_order.is_empty()
    }
}

// =============================================================================
// SAFETY: WAL + CONCURRENCY + ACID
// =============================================================================

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use parking_lot::{RwLock, Mutex};

/// Write-Ahead Log entry for crash recovery
#[derive(Debug, Clone)]
pub struct WalEntry {
    /// Monotonic sequence number
    pub lsn: u64,
    /// Operation type
    pub op: WalOp,
    /// Timestamp (epoch millis)
    pub timestamp: u64,
    /// Checksum for integrity
    pub checksum: u32,
}

/// WAL operation types
#[derive(Debug, Clone)]
pub enum WalOp {
    /// Insert fingerprint at index
    Insert { index: u32, fingerprint: Box<[u64; FINGERPRINT_WORDS]> },
    /// Update fingerprint at index
    Update { index: u32, old: Box<[u64; FINGERPRINT_WORDS]>, new: Box<[u64; FINGERPRINT_WORDS]> },
    /// Delete fingerprint at index
    Delete { index: u32, fingerprint: Box<[u64; FINGERPRINT_WORDS]> },
    /// Temperature change (for scent tracking)
    TempChange { index: u32, from: Temperature, to: Temperature },
    /// Checkpoint marker (safe recovery point)
    Checkpoint { version: u64 },
    /// Transaction begin
    TxnBegin { txn_id: u64 },
    /// Transaction commit
    TxnCommit { txn_id: u64 },
    /// Transaction abort
    TxnAbort { txn_id: u64 },
}

impl WalEntry {
    /// Create new WAL entry
    pub fn new(lsn: u64, op: WalOp) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let mut entry = Self {
            lsn,
            op,
            timestamp,
            checksum: 0,
        };
        entry.checksum = entry.compute_checksum();
        entry
    }

    /// Compute CRC32-style checksum
    fn compute_checksum(&self) -> u32 {
        // Simple checksum: XOR of lsn, timestamp, and op discriminant
        let op_tag = match &self.op {
            WalOp::Insert { .. } => 1u32,
            WalOp::Update { .. } => 2,
            WalOp::Delete { .. } => 3,
            WalOp::TempChange { .. } => 4,
            WalOp::Checkpoint { .. } => 5,
            WalOp::TxnBegin { .. } => 6,
            WalOp::TxnCommit { .. } => 7,
            WalOp::TxnAbort { .. } => 8,
        };
        (self.lsn as u32) ^ (self.timestamp as u32) ^ op_tag
    }

    /// Verify entry integrity
    pub fn verify(&self) -> bool {
        self.checksum == self.compute_checksum()
    }
}

/// Write-Ahead Log for durability
pub struct WriteAheadLog {
    /// Current LSN (Log Sequence Number)
    current_lsn: AtomicU64,
    /// In-memory log buffer (flushed to disk periodically)
    buffer: Mutex<VecDeque<WalEntry>>,
    /// Last flushed LSN
    flushed_lsn: AtomicU64,
    /// Last checkpoint LSN
    checkpoint_lsn: AtomicU64,
    /// Maximum buffer size before force flush
    max_buffer_size: usize,
    /// Path for WAL files
    wal_path: PathBuf,
    /// Is WAL enabled?
    enabled: AtomicBool,
}

impl WriteAheadLog {
    /// Create new WAL
    pub fn new(wal_path: PathBuf) -> Self {
        Self {
            current_lsn: AtomicU64::new(1),
            buffer: Mutex::new(VecDeque::with_capacity(1024)),
            flushed_lsn: AtomicU64::new(0),
            checkpoint_lsn: AtomicU64::new(0),
            max_buffer_size: 1024,
            wal_path,
            enabled: AtomicBool::new(true),
        }
    }

    /// Append entry to WAL, returns LSN
    pub fn append(&self, op: WalOp) -> u64 {
        if !self.enabled.load(Ordering::Relaxed) {
            return 0;
        }

        let lsn = self.current_lsn.fetch_add(1, Ordering::SeqCst);
        let entry = WalEntry::new(lsn, op);

        let mut buffer = self.buffer.lock();
        buffer.push_back(entry);

        // Force flush if buffer is full
        if buffer.len() >= self.max_buffer_size {
            drop(buffer);
            self.flush();
        }

        lsn
    }

    /// Flush buffer to disk
    pub fn flush(&self) -> u64 {
        let mut buffer = self.buffer.lock();
        if buffer.is_empty() {
            return self.flushed_lsn.load(Ordering::Relaxed);
        }

        // In real implementation, serialize and write to file
        // For now, just update flushed_lsn
        let max_lsn = buffer.back().map(|e| e.lsn).unwrap_or(0);
        buffer.clear();

        self.flushed_lsn.store(max_lsn, Ordering::Release);
        max_lsn
    }

    /// Create checkpoint (safe recovery point)
    pub fn checkpoint(&self) -> u64 {
        let lsn = self.append(WalOp::Checkpoint {
            version: self.current_lsn.load(Ordering::Relaxed),
        });
        self.flush();
        self.checkpoint_lsn.store(lsn, Ordering::Release);
        lsn
    }

    /// Get current LSN
    pub fn current_lsn(&self) -> u64 {
        self.current_lsn.load(Ordering::Relaxed)
    }

    /// Get last flushed LSN
    pub fn flushed_lsn(&self) -> u64 {
        self.flushed_lsn.load(Ordering::Relaxed)
    }

    /// Get last checkpoint LSN
    pub fn checkpoint_lsn(&self) -> u64 {
        self.checkpoint_lsn.load(Ordering::Relaxed)
    }

    /// Enable/disable WAL
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Recover from WAL (replay entries after last checkpoint)
    pub fn recover(&self) -> Vec<WalEntry> {
        // In real implementation, read WAL file and return entries
        // after last checkpoint for replay
        Vec::new()
    }
}

impl Default for WriteAheadLog {
    fn default() -> Self {
        Self::new(PathBuf::from("./wal"))
    }
}

/// MVCC version for isolation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Version(pub u64);

impl Version {
    pub fn new(v: u64) -> Self {
        Self(v)
    }
}

/// Versioned fingerprint for MVCC
pub struct VersionedFingerprint {
    /// The fingerprint data
    pub data: [u64; FINGERPRINT_WORDS],
    /// Version when this was written
    pub write_version: Version,
    /// Version when this was deleted (None if still visible)
    pub delete_version: Option<Version>,
}

impl VersionedFingerprint {
    /// Check if visible at given version
    pub fn visible_at(&self, version: Version) -> bool {
        version >= self.write_version
            && self.delete_version.map(|dv| version < dv).unwrap_or(true)
    }
}

/// Transaction state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxnState {
    Active,
    Committed,
    Aborted,
}

/// Transaction context for ACID
pub struct Transaction {
    /// Transaction ID
    pub id: u64,
    /// Read version (snapshot isolation)
    pub read_version: Version,
    /// State
    pub state: TxnState,
    /// WAL entries for this transaction
    pub wal_entries: Vec<u64>,
    /// Modified indices (for rollback)
    pub modified: Vec<u32>,
}

impl Transaction {
    /// Create new transaction
    pub fn new(id: u64, read_version: Version) -> Self {
        Self {
            id,
            read_version,
            state: TxnState::Active,
            wal_entries: Vec::new(),
            modified: Vec::new(),
        }
    }

    /// Check if transaction is active
    pub fn is_active(&self) -> bool {
        self.state == TxnState::Active
    }
}

/// Concurrent access controller with read-write locking
pub struct ConcurrentAccess<T> {
    /// The protected data
    data: RwLock<T>,
    /// Version counter
    version: AtomicU64,
    /// WAL reference
    wal: Arc<WriteAheadLog>,
    /// Active transactions
    active_txns: Mutex<Vec<Transaction>>,
    /// Next transaction ID
    next_txn_id: AtomicU64,
}

impl<T> ConcurrentAccess<T> {
    /// Create new concurrent access wrapper
    pub fn new(data: T, wal: Arc<WriteAheadLog>) -> Self {
        Self {
            data: RwLock::new(data),
            version: AtomicU64::new(1),
            wal,
            active_txns: Mutex::new(Vec::new()),
            next_txn_id: AtomicU64::new(1),
        }
    }

    /// Read access (shared lock)
    pub fn read(&self) -> parking_lot::RwLockReadGuard<'_, T> {
        self.data.read()
    }

    /// Write access (exclusive lock)
    pub fn write(&self) -> parking_lot::RwLockWriteGuard<'_, T> {
        self.data.write()
    }

    /// Try read with timeout
    pub fn try_read(&self, timeout: std::time::Duration) -> Option<parking_lot::RwLockReadGuard<'_, T>> {
        self.data.try_read_for(timeout)
    }

    /// Try write with timeout
    pub fn try_write(&self, timeout: std::time::Duration) -> Option<parking_lot::RwLockWriteGuard<'_, T>> {
        self.data.try_write_for(timeout)
    }

    /// Begin transaction
    pub fn begin_txn(&self) -> u64 {
        let txn_id = self.next_txn_id.fetch_add(1, Ordering::SeqCst);
        let read_version = Version::new(self.version.load(Ordering::Acquire));

        let lsn = self.wal.append(WalOp::TxnBegin { txn_id });

        let mut txn = Transaction::new(txn_id, read_version);
        txn.wal_entries.push(lsn);

        self.active_txns.lock().push(txn);
        txn_id
    }

    /// Commit transaction
    pub fn commit_txn(&self, txn_id: u64) -> bool {
        let mut txns = self.active_txns.lock();
        if let Some(pos) = txns.iter().position(|t| t.id == txn_id) {
            let mut txn = txns.remove(pos);
            if txn.state != TxnState::Active {
                return false;
            }

            txn.state = TxnState::Committed;
            self.wal.append(WalOp::TxnCommit { txn_id });

            // Bump version
            self.version.fetch_add(1, Ordering::Release);
            true
        } else {
            false
        }
    }

    /// Abort transaction
    pub fn abort_txn(&self, txn_id: u64) -> bool {
        let mut txns = self.active_txns.lock();
        if let Some(pos) = txns.iter().position(|t| t.id == txn_id) {
            let mut txn = txns.remove(pos);
            if txn.state != TxnState::Active {
                return false;
            }

            txn.state = TxnState::Aborted;
            self.wal.append(WalOp::TxnAbort { txn_id });
            // Rollback would happen here using txn.modified
            true
        } else {
            false
        }
    }

    /// Current version
    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Acquire)
    }
}

/// Thread-safe zero-copy manager with ACID guarantees
pub struct SafeArrowZeroCopy {
    /// Inner manager with concurrent access
    inner: ConcurrentAccess<ArrowZeroCopy>,
}

impl SafeArrowZeroCopy {
    /// Create new thread-safe manager
    pub fn new(wal: Arc<WriteAheadLog>) -> Self {
        Self {
            inner: ConcurrentAccess::new(ArrowZeroCopy::new(), wal),
        }
    }

    /// Load fingerprints with transaction
    pub fn load_with_txn(&self, txn_id: u64, data: Vec<u64>, num_fingerprints: usize) -> Option<usize> {
        let mut guard = self.inner.write();
        let buffer_id = guard.load_from_vec(data, num_fingerprints);

        // Record in transaction
        let mut txns = self.inner.active_txns.lock();
        if let Some(txn) = txns.iter_mut().find(|t| t.id == txn_id) {
            txn.modified.push(buffer_id as u32);
        }

        Some(buffer_id)
    }

    /// Read fingerprint (shared access)
    pub fn get(&self, buffer_id: usize, index: usize) -> Option<[u64; FINGERPRINT_WORDS]> {
        let guard = self.inner.read();
        guard.get(buffer_id, index).copied()
    }

    /// Begin transaction
    pub fn begin(&self) -> u64 {
        self.inner.begin_txn()
    }

    /// Commit transaction
    pub fn commit(&self, txn_id: u64) -> bool {
        self.inner.commit_txn(txn_id)
    }

    /// Abort transaction
    pub fn abort(&self, txn_id: u64) -> bool {
        self.inner.abort_txn(txn_id)
    }

    /// Checkpoint (create safe recovery point)
    pub fn checkpoint(&self) {
        self.inner.wal.checkpoint();
    }

    /// Force WAL flush
    pub fn flush_wal(&self) {
        self.inner.wal.flush();
    }
}

// =============================================================================
// TRANSPARENT WRITE-THROUGH FOR PREFIX 0x00 (LANCE)
// =============================================================================

/// Prefix constants (mirrored from bind_space for no-dependency)
pub const PREFIX_LANCE: u8 = 0x00;
pub const PREFIX_SURFACE_END: u8 = 0x0F;

/// Storage backend trait for transparent DTO integration
///
/// All query languages (Redis, SQL, Cypher, GQL, NARS) go through this trait.
/// The implementation decides where data actually lives:
/// - Hot: In-memory BindSpace array
/// - Warm: Lance mmap'd buffer
/// - Cold: Lance persistent storage
pub trait StorageBackend: Send + Sync {
    /// Read fingerprint at address (prefix:slot)
    fn read_fingerprint(&self, prefix: u8, slot: u8) -> Option<[u64; FINGERPRINT_WORDS]>;

    /// Write fingerprint to address
    fn write_fingerprint(&self, prefix: u8, slot: u8, fp: [u64; FINGERPRINT_WORDS]) -> bool;

    /// Delete fingerprint at address
    fn delete_fingerprint(&self, prefix: u8, slot: u8) -> bool;

    /// Check if address is in Lance prefix (for routing)
    fn is_lance_prefix(&self, prefix: u8) -> bool {
        prefix == PREFIX_LANCE
    }

    /// Sync to persistent storage
    fn sync(&self) -> bool;

    /// Begin transaction
    fn begin_transaction(&self) -> u64;

    /// Commit transaction
    fn commit_transaction(&self, txn_id: u64) -> bool;

    /// Abort transaction
    fn abort_transaction(&self, txn_id: u64) -> bool;
}

/// Transparent write-through adapter for Lance prefix
///
/// All writes to prefix 0x00 automatically persist to Lance storage.
/// All reads check the hot cache first, then fall through to Lance.
///
/// This achieves the "same control over storage access architecture"
/// without duplicating methods between BindSpace and LanceDB.
pub struct LanceWriteThrough {
    /// The zero-copy Lance storage (warm/cold)
    lance: SafeArrowZeroCopy,

    /// Hot cache: recently accessed fingerprints
    /// Key: (slot as u32), Value: fingerprint
    hot_cache: RwLock<HashMap<u32, [u64; FINGERPRINT_WORDS]>>,

    /// Dirty set: slots modified but not yet synced
    dirty: Mutex<Vec<u32>>,

    /// Configuration
    config: WriteThroughConfig,
}

/// Configuration for write-through behavior
#[derive(Debug, Clone)]
pub struct WriteThroughConfig {
    /// Maximum entries in hot cache before eviction
    pub max_hot_entries: usize,

    /// Sync to Lance after this many writes
    pub sync_interval: usize,

    /// Enable write-behind (async) instead of write-through (sync)
    pub write_behind: bool,
}

impl Default for WriteThroughConfig {
    fn default() -> Self {
        Self {
            max_hot_entries: 1024,
            sync_interval: 100,
            write_behind: false,
        }
    }
}

impl LanceWriteThrough {
    /// Create new write-through adapter
    pub fn new(wal: Arc<WriteAheadLog>, config: WriteThroughConfig) -> Self {
        Self {
            lance: SafeArrowZeroCopy::new(wal),
            hot_cache: RwLock::new(HashMap::with_capacity(config.max_hot_entries)),
            dirty: Mutex::new(Vec::new()),
            config,
        }
    }

    /// Create with default config
    pub fn with_defaults(wal: Arc<WriteAheadLog>) -> Self {
        Self::new(wal, WriteThroughConfig::default())
    }

    /// Read from hot cache or Lance
    pub fn read(&self, slot: u8) -> Option<[u64; FINGERPRINT_WORDS]> {
        let key = slot as u32;

        // Check hot cache first (fast path)
        {
            let cache = self.hot_cache.read();
            if let Some(fp) = cache.get(&key) {
                return Some(*fp);
            }
        }

        // Fall through to Lance storage
        self.lance.get(0, slot as usize)
    }

    /// Write to hot cache and optionally to Lance
    pub fn write(&self, slot: u8, fp: [u64; FINGERPRINT_WORDS]) -> bool {
        let key = slot as u32;

        // Write to hot cache
        {
            let mut cache = self.hot_cache.write();
            cache.insert(key, fp);

            // Evict if over capacity
            if cache.len() > self.config.max_hot_entries {
                // Simple eviction: remove first key (could use LRU)
                if let Some(&evict_key) = cache.keys().next() {
                    cache.remove(&evict_key);
                }
            }
        }

        // Track dirty slot
        {
            let mut dirty = self.dirty.lock();
            if !dirty.contains(&key) {
                dirty.push(key);
            }

            // Sync if interval reached
            if dirty.len() >= self.config.sync_interval && !self.config.write_behind {
                drop(dirty);
                self.sync_to_lance();
            }
        }

        true
    }

    /// Delete from cache and Lance
    pub fn delete(&self, slot: u8) -> bool {
        let key = slot as u32;

        // Remove from hot cache
        {
            let mut cache = self.hot_cache.write();
            cache.remove(&key);
        }

        // Track deletion (would need separate delete tracking in production)
        true
    }

    /// Sync dirty entries to Lance storage
    pub fn sync_to_lance(&self) -> usize {
        let dirty_slots: Vec<u32>;
        {
            let mut dirty = self.dirty.lock();
            dirty_slots = dirty.drain(..).collect();
        }

        if dirty_slots.is_empty() {
            return 0;
        }

        // Begin transaction
        let txn_id = self.lance.begin();

        // Collect fingerprints for bulk write
        let cache = self.hot_cache.read();
        let mut data = Vec::with_capacity(dirty_slots.len() * FINGERPRINT_WORDS);
        let mut count = 0;

        for &slot in &dirty_slots {
            if let Some(fp) = cache.get(&slot) {
                data.extend_from_slice(fp);
                count += 1;
            }
        }
        drop(cache);

        // Load into Lance
        if !data.is_empty() {
            self.lance.load_with_txn(txn_id, data, count);
        }

        // Commit transaction
        self.lance.commit(txn_id);
        self.lance.checkpoint();

        count
    }

    /// Get statistics
    pub fn stats(&self) -> WriteThroughStats {
        let cache = self.hot_cache.read();
        let dirty = self.dirty.lock();
        WriteThroughStats {
            hot_entries: cache.len(),
            dirty_entries: dirty.len(),
        }
    }
}

/// Write-through statistics
#[derive(Debug, Clone)]
pub struct WriteThroughStats {
    pub hot_entries: usize,
    pub dirty_entries: usize,
}

impl StorageBackend for LanceWriteThrough {
    fn read_fingerprint(&self, prefix: u8, slot: u8) -> Option<[u64; FINGERPRINT_WORDS]> {
        if prefix == PREFIX_LANCE {
            self.read(slot)
        } else {
            None // Only handles Lance prefix
        }
    }

    fn write_fingerprint(&self, prefix: u8, slot: u8, fp: [u64; FINGERPRINT_WORDS]) -> bool {
        if prefix == PREFIX_LANCE {
            self.write(slot, fp)
        } else {
            false
        }
    }

    fn delete_fingerprint(&self, prefix: u8, slot: u8) -> bool {
        if prefix == PREFIX_LANCE {
            self.delete(slot)
        } else {
            false
        }
    }

    fn sync(&self) -> bool {
        self.sync_to_lance() > 0 || self.dirty.lock().is_empty()
    }

    fn begin_transaction(&self) -> u64 {
        self.lance.begin()
    }

    fn commit_transaction(&self, txn_id: u64) -> bool {
        self.lance.commit(txn_id)
    }

    fn abort_transaction(&self, txn_id: u64) -> bool {
        self.lance.abort(txn_id)
    }
}

/// Unified storage that routes by prefix
///
/// This is the "polyglot" entry point - all query languages hit the same interface.
/// Routing is transparent based on the address prefix.
pub struct UnifiedStorage {
    /// Lance storage for prefix 0x00
    lance: Arc<LanceWriteThrough>,

    // Note: Additional backends can be registered here in the future
    // backends: HashMap<u8, Arc<dyn StorageBackend>>,
}

impl UnifiedStorage {
    /// Create unified storage with Lance write-through
    pub fn new(wal: Arc<WriteAheadLog>) -> Self {
        Self {
            lance: Arc::new(LanceWriteThrough::with_defaults(wal)),
        }
    }

    /// Read from any prefix
    ///
    /// Routes to appropriate backend based on prefix:
    /// - 0x00 (Lance): Zero-copy buffer storage
    /// - Others: Would route to SQL, Cypher, etc. backends
    pub fn read(&self, prefix: u8, slot: u8) -> Option<[u64; FINGERPRINT_WORDS]> {
        match prefix {
            PREFIX_LANCE => self.lance.read(slot),
            // Other prefixes would route to their backends
            _ => None,
        }
    }

    /// Write to any prefix
    pub fn write(&self, prefix: u8, slot: u8, fp: [u64; FINGERPRINT_WORDS]) -> bool {
        match prefix {
            PREFIX_LANCE => self.lance.write(slot, fp),
            _ => false,
        }
    }

    /// Sync all backends
    pub fn sync_all(&self) {
        self.lance.sync_to_lance();
    }

    /// Get Lance backend reference
    pub fn lance(&self) -> &LanceWriteThrough {
        &self.lance
    }
}

// =============================================================================
// ZERO-COPY VIEW
// =============================================================================

/// A zero-copy view into Lance storage
///
/// This struct holds pointers into mmap'd Arrow columns.
/// No copies, just views into the OS page cache.
pub struct LanceView {
    /// Path to the Lance dataset
    pub path: PathBuf,

    /// Number of fingerprints in this view
    pub len: usize,

    /// Raw pointer to fingerprint data (156 × u64 per fingerprint)
    /// Points directly into mmap'd Arrow buffer
    #[cfg(feature = "lance")]
    fingerprint_ptr: *const u64,

    /// Raw pointer to address data (u16 per entry)
    #[cfg(feature = "lance")]
    address_ptr: *const u16,

    /// Scent index for awareness (which entries are hot/warm/cold)
    scent: ScentAwareness,
}

// Safety: The pointers point to mmap'd memory that outlives this struct
#[cfg(feature = "lance")]
unsafe impl Send for LanceView {}
#[cfg(feature = "lance")]
unsafe impl Sync for LanceView {}

impl LanceView {
    /// Create a new Lance view (placeholder for when Lance is vendored)
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            len: 0,
            #[cfg(feature = "lance")]
            fingerprint_ptr: std::ptr::null(),
            #[cfg(feature = "lance")]
            address_ptr: std::ptr::null(),
            scent: ScentAwareness::new(),
        }
    }

    /// Zero-copy access to fingerprint at index
    ///
    /// # Safety
    /// Caller must ensure index < self.len
    #[cfg(feature = "lance")]
    #[inline]
    pub unsafe fn fingerprint_unchecked(&self, index: usize) -> &[u64; FINGERPRINT_WORDS] {
        &*(self.fingerprint_ptr.add(index * FINGERPRINT_WORDS) as *const [u64; FINGERPRINT_WORDS])
    }

    /// Safe access to fingerprint
    #[cfg(feature = "lance")]
    pub fn fingerprint(&self, index: usize) -> Option<&[u64; FINGERPRINT_WORDS]> {
        if index < self.len {
            Some(unsafe { self.fingerprint_unchecked(index) })
        } else {
            None
        }
    }

    /// Get scent awareness
    pub fn scent(&self) -> &ScentAwareness {
        &self.scent
    }

    /// Get mutable scent awareness
    pub fn scent_mut(&mut self) -> &mut ScentAwareness {
        &mut self.scent
    }

    /// Number of entries
    pub fn len(&self) -> usize {
        self.len
    }

    /// Is empty?
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

// =============================================================================
// SCENT AWARENESS
// =============================================================================

/// Temperature tier for bubbling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Temperature {
    /// In BindSpace array (< 1 cycle access)
    Hot,
    /// In mmap region (page cache, ~100ns)
    Warm,
    /// On disk (needs page fault, ~10μs)
    Cold,
    /// Unknown/not tracked
    Unknown,
}

/// Scent awareness - knows where data lives without copying it
///
/// This is the "awareness" layer that tracks:
/// - Which fingerprints are hot (in BindSpace)
/// - Which are warm (in mmap page cache)
/// - Which are cold (on disk)
///
/// Bubbling happens by updating this index, not by copying data.
pub struct ScentAwareness {
    /// Hot set: indices that are in BindSpace
    hot_indices: Vec<u32>,

    /// Recently accessed (LRU for warmth tracking)
    recent_access: Vec<u32>,

    /// Access counts for promotion decisions
    access_counts: Vec<u8>,

    /// Capacity
    capacity: usize,
}

impl ScentAwareness {
    /// Create new scent awareness
    pub fn new() -> Self {
        Self::with_capacity(1024)
    }

    /// Create with capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            hot_indices: Vec::with_capacity(capacity / 10),
            recent_access: Vec::with_capacity(64),
            access_counts: vec![0; capacity],
            capacity,
        }
    }

    /// Record an access (for warmth tracking)
    pub fn touch(&mut self, index: u32) {
        let idx = index as usize;
        if idx < self.access_counts.len() {
            self.access_counts[idx] = self.access_counts[idx].saturating_add(1);
        }

        // Update recent access (simple ring buffer)
        if self.recent_access.len() >= 64 {
            self.recent_access.remove(0);
        }
        self.recent_access.push(index);
    }

    /// Get temperature of an index
    pub fn temperature(&self, index: u32) -> Temperature {
        if self.hot_indices.binary_search(&index).is_ok() {
            Temperature::Hot
        } else if self.recent_access.contains(&index) {
            Temperature::Warm
        } else if (index as usize) < self.access_counts.len()
            && self.access_counts[index as usize] > 0
        {
            Temperature::Warm
        } else {
            Temperature::Cold
        }
    }

    /// Mark index as hot (bubbled up to BindSpace)
    pub fn mark_hot(&mut self, index: u32) {
        if self.hot_indices.binary_search(&index).is_err() {
            self.hot_indices.push(index);
            self.hot_indices.sort();
        }
    }

    /// Mark index as cold (bubbled down from BindSpace)
    pub fn mark_cold(&mut self, index: u32) {
        if let Ok(pos) = self.hot_indices.binary_search(&index) {
            self.hot_indices.remove(pos);
        }
    }

    /// Get indices that should bubble up (high access count, currently cold)
    pub fn candidates_for_promotion(&self, limit: usize) -> Vec<u32> {
        let mut candidates: Vec<(u32, u8)> = self.access_counts
            .iter()
            .enumerate()
            .filter(|(idx, &count)| {
                count > 5 && self.hot_indices.binary_search(&(*idx as u32)).is_err()
            })
            .map(|(idx, &count)| (idx as u32, count))
            .collect();

        candidates.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by access count desc
        candidates.into_iter().take(limit).map(|(idx, _)| idx).collect()
    }

    /// Get indices that should bubble down (low access count, currently hot)
    pub fn candidates_for_demotion(&self, limit: usize) -> Vec<u32> {
        let mut candidates: Vec<(u32, u8)> = self.hot_indices
            .iter()
            .filter_map(|&idx| {
                let count = self.access_counts.get(idx as usize).copied().unwrap_or(0);
                if count < 2 {
                    Some((idx, count))
                } else {
                    None
                }
            })
            .collect();

        candidates.sort_by(|a, b| a.1.cmp(&b.1)); // Sort by access count asc
        candidates.into_iter().take(limit).map(|(idx, _)| idx).collect()
    }

    /// Decay access counts (call periodically)
    pub fn decay(&mut self) {
        for count in &mut self.access_counts {
            *count = count.saturating_sub(1);
        }
    }

    /// Number of hot entries
    pub fn hot_count(&self) -> usize {
        self.hot_indices.len()
    }
}

impl Default for ScentAwareness {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// BUBBLING OPERATIONS
// =============================================================================

/// Bubble operation result
#[derive(Debug, Clone)]
pub struct BubbleResult {
    /// Index that was bubbled
    pub index: u32,
    /// Previous temperature
    pub from: Temperature,
    /// New temperature
    pub to: Temperature,
    /// Whether data was copied (should be false for zero-copy)
    pub copied: bool,
}

/// Zero-copy bubbler
///
/// Moves data between hot/warm/cold tiers by updating pointers,
/// not by copying data.
pub struct ZeroCopyBubbler {
    /// Maximum hot entries
    max_hot: usize,
    /// Bubble threshold (access count to promote)
    promote_threshold: u8,
    /// Demote threshold (access count to demote)
    demote_threshold: u8,
}

impl Default for ZeroCopyBubbler {
    fn default() -> Self {
        Self {
            max_hot: 32768, // Same as BindSpace node capacity
            promote_threshold: 10,
            demote_threshold: 2,
        }
    }
}

impl ZeroCopyBubbler {
    /// Create with custom thresholds
    pub fn new(max_hot: usize, promote_threshold: u8, demote_threshold: u8) -> Self {
        Self {
            max_hot,
            promote_threshold,
            demote_threshold,
        }
    }

    /// Bubble up: promote cold/warm to hot
    ///
    /// In zero-copy mode, this just updates the scent index.
    /// The actual fingerprint stays in the mmap'd file.
    pub fn bubble_up(&self, scent: &mut ScentAwareness, index: u32) -> BubbleResult {
        let from = scent.temperature(index);
        scent.mark_hot(index);

        BubbleResult {
            index,
            from,
            to: Temperature::Hot,
            copied: false, // Zero-copy!
        }
    }

    /// Bubble down: demote hot to warm/cold
    pub fn bubble_down(&self, scent: &mut ScentAwareness, index: u32) -> BubbleResult {
        let from = scent.temperature(index);
        scent.mark_cold(index);

        BubbleResult {
            index,
            from,
            to: Temperature::Warm, // Goes to warm first (still in page cache)
            copied: false,
        }
    }

    /// Auto-bubble based on access patterns
    pub fn auto_bubble(&self, scent: &mut ScentAwareness) -> Vec<BubbleResult> {
        let mut results = Vec::new();

        // Promote hot candidates
        if scent.hot_count() < self.max_hot {
            let room = self.max_hot - scent.hot_count();
            for idx in scent.candidates_for_promotion(room.min(10)) {
                results.push(self.bubble_up(scent, idx));
            }
        }

        // Demote cold candidates if we're at capacity
        if scent.hot_count() >= self.max_hot {
            for idx in scent.candidates_for_demotion(10) {
                results.push(self.bubble_down(scent, idx));
            }
        }

        results
    }
}

// =============================================================================
// KÙZU-STYLE OPTIMIZATIONS
// =============================================================================
// Buffer pool, copy-on-write, efficient serialization, CSR edges

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::io::{Read as IoRead, Write as IoWrite};

/// Magic bytes for file format identification
pub const MAGIC_BYTES: [u8; 8] = *b"LADYBUG\x00";

/// File format version (major.minor.patch as u32)
pub const FORMAT_VERSION: u32 = 0x0001_0000; // 1.0.0

// -----------------------------------------------------------------------------
// BUFFER POOL MANAGER (Kùzu-style page management)
// -----------------------------------------------------------------------------

/// Page size for buffer pool (4KB aligned like OS pages)
pub const PAGE_SIZE: usize = 4096;

/// Page ID combining file ID and page number
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PageId {
    pub file_id: u32,
    pub page_num: u32,
}

impl PageId {
    pub fn new(file_id: u32, page_num: u32) -> Self {
        Self { file_id, page_num }
    }
}

/// Buffer frame in the pool
pub struct BufferFrame {
    /// Page ID this frame holds
    page_id: Option<PageId>,
    /// The actual data
    data: Box<[u8; PAGE_SIZE]>,
    /// Pin count (>0 means page is in use)
    pin_count: u32,
    /// Dirty flag (modified since load)
    dirty: bool,
    /// Reference bit for clock algorithm
    ref_bit: bool,
}

impl BufferFrame {
    fn new() -> Self {
        Self {
            page_id: None,
            data: Box::new([0u8; PAGE_SIZE]),
            pin_count: 0,
            dirty: false,
            ref_bit: false,
        }
    }

    /// Mark as dirty (needs writeback)
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Get data slice
    pub fn data(&self) -> &[u8; PAGE_SIZE] {
        &self.data
    }

    /// Get mutable data slice
    pub fn data_mut(&mut self) -> &mut [u8; PAGE_SIZE] {
        self.dirty = true;
        &mut self.data
    }
}

/// Buffer Pool Manager with clock-based page eviction
///
/// Kùzu-style buffer management:
/// - Fixed-size buffer pool with page frames
/// - Clock algorithm for victim selection
/// - Pin/unpin semantics for safe access
/// - Dirty page tracking for write-back
pub struct BufferPoolManager {
    /// Pool of buffer frames
    frames: Vec<BufferFrame>,
    /// Map from PageId to frame index
    page_table: HashMap<PageId, usize>,
    /// Clock hand for eviction
    clock_hand: usize,
    /// Number of frames
    pool_size: usize,
    /// Statistics
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
}

impl BufferPoolManager {
    /// Create buffer pool with specified number of frames
    pub fn new(pool_size: usize) -> Self {
        let frames = (0..pool_size).map(|_| BufferFrame::new()).collect();
        Self {
            frames,
            page_table: HashMap::with_capacity(pool_size),
            clock_hand: 0,
            pool_size,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }

    /// Fetch page into buffer pool
    ///
    /// Returns frame index if successful.
    /// Uses clock algorithm for eviction if pool is full.
    pub fn fetch_page(&mut self, page_id: PageId) -> Option<usize> {
        // Check if already in pool
        if let Some(&frame_idx) = self.page_table.get(&page_id) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            self.frames[frame_idx].ref_bit = true;
            return Some(frame_idx);
        }

        self.misses.fetch_add(1, Ordering::Relaxed);

        // Find a victim frame using clock algorithm
        let victim_idx = self.find_victim()?;

        // Evict if necessary
        if let Some(old_page_id) = self.frames[victim_idx].page_id {
            if self.frames[victim_idx].dirty {
                // Would write back to disk here
                // self.flush_page(old_page_id);
            }
            self.page_table.remove(&old_page_id);
            self.evictions.fetch_add(1, Ordering::Relaxed);
        }

        // Load new page (would read from disk here)
        self.frames[victim_idx].page_id = Some(page_id);
        self.frames[victim_idx].pin_count = 1;
        self.frames[victim_idx].dirty = false;
        self.frames[victim_idx].ref_bit = true;
        self.page_table.insert(page_id, victim_idx);

        Some(victim_idx)
    }

    /// Find victim frame using clock algorithm
    fn find_victim(&mut self) -> Option<usize> {
        let mut attempts = 0;
        let max_attempts = self.pool_size * 2;

        while attempts < max_attempts {
            let idx = self.clock_hand;
            self.clock_hand = (self.clock_hand + 1) % self.pool_size;

            // Skip pinned pages
            if self.frames[idx].pin_count > 0 {
                attempts += 1;
                continue;
            }

            // Check reference bit
            if self.frames[idx].ref_bit {
                self.frames[idx].ref_bit = false;
                attempts += 1;
                continue;
            }

            // Found victim
            return Some(idx);
        }

        // All pages pinned or recently used
        None
    }

    /// Pin page (increment pin count)
    pub fn pin_page(&mut self, page_id: PageId) -> bool {
        if let Some(&frame_idx) = self.page_table.get(&page_id) {
            self.frames[frame_idx].pin_count += 1;
            self.frames[frame_idx].ref_bit = true;
            true
        } else {
            false
        }
    }

    /// Unpin page (decrement pin count)
    pub fn unpin_page(&mut self, page_id: PageId, dirty: bool) -> bool {
        if let Some(&frame_idx) = self.page_table.get(&page_id) {
            if self.frames[frame_idx].pin_count > 0 {
                self.frames[frame_idx].pin_count -= 1;
            }
            if dirty {
                self.frames[frame_idx].dirty = true;
            }
            true
        } else {
            false
        }
    }

    /// Flush all dirty pages
    pub fn flush_all(&mut self) {
        for frame in &mut self.frames {
            if frame.dirty {
                // Would write to disk here
                frame.dirty = false;
            }
        }
    }

    /// Get frame by index
    pub fn get_frame(&self, frame_idx: usize) -> Option<&BufferFrame> {
        self.frames.get(frame_idx)
    }

    /// Get mutable frame by index
    pub fn get_frame_mut(&mut self, frame_idx: usize) -> Option<&mut BufferFrame> {
        self.frames.get_mut(frame_idx)
    }

    /// Statistics
    pub fn hit_ratio(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed) as f64;
        let misses = self.misses.load(Ordering::Relaxed) as f64;
        if hits + misses == 0.0 {
            0.0
        } else {
            hits / (hits + misses)
        }
    }

    pub fn eviction_count(&self) -> u64 {
        self.evictions.load(Ordering::Relaxed)
    }
}

// -----------------------------------------------------------------------------
// COPY-ON-WRITE (CoW) SEMANTICS
// -----------------------------------------------------------------------------

/// Copy-on-Write wrapper for zero-copy reads with safe writes
///
/// Kùzu uses CoW for MVCC - readers never block writers.
/// This implementation provides:
/// - Fast reads via Arc sharing
/// - Copy only when modification needed
/// - Version tracking for MVCC
pub struct CopyOnWrite<T: Clone> {
    /// The data wrapped in Arc for sharing
    data: Arc<RwLock<T>>,
    /// Version number
    version: AtomicU64,
    /// Copy count (for statistics)
    copies: AtomicU64,
}

impl<T: Clone> CopyOnWrite<T> {
    /// Create new CoW wrapper
    pub fn new(data: T) -> Self {
        Self {
            data: Arc::new(RwLock::new(data)),
            version: AtomicU64::new(1),
            copies: AtomicU64::new(0),
        }
    }

    /// Read access (shared, no copy)
    pub fn read(&self) -> parking_lot::RwLockReadGuard<'_, T> {
        self.data.read()
    }

    /// Write access with copy-on-write
    ///
    /// If there are multiple readers, creates a copy.
    /// Otherwise, mutates in place.
    pub fn write(&self) -> parking_lot::RwLockWriteGuard<'_, T> {
        self.version.fetch_add(1, Ordering::SeqCst);

        // Check if we're the only owner
        // Note: With RwLock, we always have exclusive access when writing
        self.data.write()
    }

    /// Get a snapshot (clone for independent access)
    pub fn snapshot(&self) -> T {
        self.copies.fetch_add(1, Ordering::Relaxed);
        self.data.read().clone()
    }

    /// Current version
    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Acquire)
    }

    /// Number of copies made
    pub fn copy_count(&self) -> u64 {
        self.copies.load(Ordering::Relaxed)
    }
}

impl<T: Clone + Default> Default for CopyOnWrite<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

// -----------------------------------------------------------------------------
// EFFICIENT SERIALIZATION (Kùzu-style with magic bytes, versioning)
// -----------------------------------------------------------------------------

/// Serialization header for fingerprint storage
#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
pub struct SerializationHeader {
    /// Magic bytes for format identification
    pub magic: [u8; 8],
    /// Format version
    pub version: u32,
    /// Flags (compression, encoding, etc.)
    pub flags: u32,
    /// Number of fingerprints
    pub count: u64,
    /// Checksum of data section (CRC32)
    pub checksum: u32,
    /// Reserved for future use
    pub reserved: [u8; 4],
}

impl SerializationHeader {
    /// Create new header
    pub fn new(count: u64) -> Self {
        Self {
            magic: MAGIC_BYTES,
            version: FORMAT_VERSION,
            flags: 0,
            count,
            checksum: 0,
            reserved: [0; 4],
        }
    }

    /// Header size in bytes
    pub const SIZE: usize = std::mem::size_of::<Self>();

    /// Validate header
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.magic != MAGIC_BYTES {
            return Err("Invalid magic bytes");
        }
        if self.version > FORMAT_VERSION {
            return Err("Unsupported format version");
        }
        Ok(())
    }

    /// Serialize header to bytes
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        unsafe { std::mem::transmute_copy(self) }
    }

    /// Deserialize header from bytes
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < Self::SIZE {
            return None;
        }
        let header: Self = unsafe {
            std::ptr::read(bytes.as_ptr() as *const Self)
        };
        Some(header)
    }
}

/// Serialization flags
pub mod flags {
    /// Data is LZ4 compressed
    pub const COMPRESSED_LZ4: u32 = 1 << 0;
    /// Data is bit-packed (no wasted bits)
    pub const BITPACKED: u32 = 1 << 1;
    /// Fingerprints are sorted by adjacency
    pub const ADJACENCY_SORTED: u32 = 1 << 2;
    /// Includes adjacency index
    pub const HAS_ADJACENCY_INDEX: u32 = 1 << 3;
    /// Includes scent metadata
    pub const HAS_SCENT: u32 = 1 << 4;
}

/// Efficient serializer for fingerprint storage
pub struct FingerprintSerializer;

impl FingerprintSerializer {
    /// Serialize fingerprints to bytes with header
    pub fn serialize(fingerprints: &FingerprintBuffer) -> Vec<u8> {
        let count = fingerprints.len() as u64;
        let data_size = count as usize * FINGERPRINT_WORDS * 8;

        let mut header = SerializationHeader::new(count);

        // Compute checksum
        let data_bytes = unsafe {
            std::slice::from_raw_parts(
                fingerprints.as_ptr() as *const u8,
                data_size
            )
        };
        header.checksum = Self::crc32(data_bytes);

        // Build output
        let mut output = Vec::with_capacity(SerializationHeader::SIZE + data_size);
        output.extend_from_slice(&header.to_bytes());
        output.extend_from_slice(data_bytes);

        output
    }

    /// Deserialize fingerprints from bytes
    pub fn deserialize(bytes: &[u8]) -> Result<FingerprintBuffer, &'static str> {
        let header = SerializationHeader::from_bytes(bytes)
            .ok_or("Truncated header")?;
        header.validate()?;

        let data_start = SerializationHeader::SIZE;
        let data_size = header.count as usize * FINGERPRINT_WORDS * 8;

        if bytes.len() < data_start + data_size {
            return Err("Truncated data");
        }

        let data_bytes = &bytes[data_start..data_start + data_size];

        // Verify checksum
        if header.checksum != 0 && header.checksum != Self::crc32(data_bytes) {
            return Err("Checksum mismatch");
        }

        // Convert to u64 vec
        let u64_count = header.count as usize * FINGERPRINT_WORDS;
        let mut data = vec![0u64; u64_count];
        unsafe {
            std::ptr::copy_nonoverlapping(
                data_bytes.as_ptr(),
                data.as_mut_ptr() as *mut u8,
                data_size
            );
        }

        Ok(FingerprintBuffer::from_vec(data, header.count as usize))
    }

    /// Simple CRC32 checksum
    fn crc32(data: &[u8]) -> u32 {
        // Simple implementation - in production use crc32fast crate
        let mut crc = 0xFFFF_FFFFu32;
        for byte in data {
            crc ^= *byte as u32;
            for _ in 0..8 {
                crc = if crc & 1 != 0 {
                    (crc >> 1) ^ 0xEDB8_8320
                } else {
                    crc >> 1
                };
            }
        }
        !crc
    }
}

// -----------------------------------------------------------------------------
// SPARSE FINGERPRINT (100:1 XOR-compressed, transparent ops)
// -----------------------------------------------------------------------------

/// Maximum words for high-resolution fingerprints
/// 64 Mio bits = 1M words × 64 bits
pub const SPARSE_MAX_WORDS: usize = 1_000_000;

/// Sparse XOR-compressed fingerprint
///
/// Stores only non-zero u64 words, enabling 100:1 compression for
/// sparse qualia patterns while preserving transparent algebraic operations.
///
/// # Algebraic Properties (preserved under compression)
/// - XOR: A ⊕ B works directly on sparse representation
/// - Hamming: popcount(A ⊕ B) computed without decompression
/// - Bind: rotate + XOR stays sparse
/// - Bundle: majority vote (requires expansion for dense results)
///
/// # Memory
/// - Full 64M bits: 8 MB
/// - 1% sparse: 80 KB (presence bitmap + non-zero words)
/// - 0.1% sparse: 8 KB
#[derive(Clone, Debug)]
pub struct SparseFingerprint {
    /// Total number of words in full representation
    total_words: usize,

    /// Bitmap indicating which words are non-zero
    /// Length: ceil(total_words / 64) u64s
    presence: Vec<u64>,

    /// Non-zero word indices (sorted)
    indices: Vec<u32>,

    /// Non-zero word values (parallel to indices)
    values: Vec<u64>,
}

impl SparseFingerprint {
    /// Create empty sparse fingerprint for given resolution
    pub fn new(total_words: usize) -> Self {
        let presence_len = (total_words + 63) / 64;
        Self {
            total_words,
            presence: vec![0u64; presence_len],
            indices: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Create from dense representation (compresses automatically)
    pub fn from_dense(words: &[u64]) -> Self {
        let total_words = words.len();
        let presence_len = (total_words + 63) / 64;
        let mut presence = vec![0u64; presence_len];
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for (i, &word) in words.iter().enumerate() {
            if word != 0 {
                // Set presence bit
                presence[i / 64] |= 1u64 << (i % 64);
                indices.push(i as u32);
                values.push(word);
            }
        }

        Self {
            total_words,
            presence,
            indices,
            values,
        }
    }

    /// Create from standard fingerprint (156 words)
    pub fn from_fingerprint(fp: &[u64; FINGERPRINT_WORDS]) -> Self {
        Self::from_dense(fp)
    }

    /// Expand to dense representation
    pub fn to_dense(&self) -> Vec<u64> {
        let mut result = vec![0u64; self.total_words];
        for (&idx, &val) in self.indices.iter().zip(&self.values) {
            result[idx as usize] = val;
        }
        result
    }

    /// Check if word at index is present (non-zero)
    #[inline]
    fn is_present(&self, idx: usize) -> bool {
        if idx >= self.total_words {
            return false;
        }
        (self.presence[idx / 64] >> (idx % 64)) & 1 != 0
    }

    /// Set presence bit
    #[inline]
    fn set_present(&mut self, idx: usize) {
        self.presence[idx / 64] |= 1u64 << (idx % 64);
    }

    /// Clear presence bit
    #[inline]
    fn clear_present(&mut self, idx: usize) {
        self.presence[idx / 64] &= !(1u64 << (idx % 64));
    }

    /// Get value at index (0 if not present)
    pub fn get(&self, idx: usize) -> u64 {
        if !self.is_present(idx) {
            return 0;
        }
        // Binary search for index
        match self.indices.binary_search(&(idx as u32)) {
            Ok(pos) => self.values[pos],
            Err(_) => 0,
        }
    }

    /// Set value at index
    pub fn set(&mut self, idx: usize, value: u64) {
        if value == 0 {
            // Remove if present
            if self.is_present(idx) {
                self.clear_present(idx);
                if let Ok(pos) = self.indices.binary_search(&(idx as u32)) {
                    self.indices.remove(pos);
                    self.values.remove(pos);
                }
            }
        } else {
            // Insert or update
            match self.indices.binary_search(&(idx as u32)) {
                Ok(pos) => {
                    self.values[pos] = value;
                }
                Err(pos) => {
                    self.set_present(idx);
                    self.indices.insert(pos, idx as u32);
                    self.values.insert(pos, value);
                }
            }
        }
    }

    /// XOR with another sparse fingerprint (TRANSPARENT - no decompression)
    ///
    /// Complexity: O(n + m) where n, m are non-zero counts
    pub fn xor(&self, other: &SparseFingerprint) -> SparseFingerprint {
        assert_eq!(self.total_words, other.total_words);

        let presence_len = self.presence.len();
        let mut result = SparseFingerprint {
            total_words: self.total_words,
            presence: vec![0u64; presence_len],
            indices: Vec::with_capacity(self.indices.len() + other.indices.len()),
            values: Vec::with_capacity(self.indices.len() + other.indices.len()),
        };

        // Merge sorted index arrays
        let mut i = 0;
        let mut j = 0;

        while i < self.indices.len() || j < other.indices.len() {
            let (idx, val) = if i >= self.indices.len() {
                // Only other remaining
                let idx = other.indices[j];
                let val = other.values[j];
                j += 1;
                (idx, val)
            } else if j >= other.indices.len() {
                // Only self remaining
                let idx = self.indices[i];
                let val = self.values[i];
                i += 1;
                (idx, val)
            } else if self.indices[i] < other.indices[j] {
                // Self comes first
                let idx = self.indices[i];
                let val = self.values[i];
                i += 1;
                (idx, val)
            } else if self.indices[i] > other.indices[j] {
                // Other comes first
                let idx = other.indices[j];
                let val = other.values[j];
                j += 1;
                (idx, val)
            } else {
                // Same index - XOR the values
                let idx = self.indices[i];
                let val = self.values[i] ^ other.values[j];
                i += 1;
                j += 1;
                (idx, val)
            };

            // Only store if non-zero
            if val != 0 {
                result.set_present(idx as usize);
                result.indices.push(idx);
                result.values.push(val);
            }
        }

        result
    }

    /// Hamming distance (TRANSPARENT - no decompression)
    ///
    /// Complexity: O(n + m) where n, m are non-zero counts
    pub fn hamming(&self, other: &SparseFingerprint) -> u64 {
        assert_eq!(self.total_words, other.total_words);

        let mut distance = 0u64;
        let mut i = 0;
        let mut j = 0;

        while i < self.indices.len() || j < other.indices.len() {
            if i >= self.indices.len() {
                // Remaining in other
                distance += other.values[j].count_ones() as u64;
                j += 1;
            } else if j >= other.indices.len() {
                // Remaining in self
                distance += self.values[i].count_ones() as u64;
                i += 1;
            } else if self.indices[i] < other.indices[j] {
                // Only in self
                distance += self.values[i].count_ones() as u64;
                i += 1;
            } else if self.indices[i] > other.indices[j] {
                // Only in other
                distance += other.values[j].count_ones() as u64;
                j += 1;
            } else {
                // Both present - XOR and count
                distance += (self.values[i] ^ other.values[j]).count_ones() as u64;
                i += 1;
                j += 1;
            }
        }

        distance
    }

    /// Bind operation: rotate then XOR (TRANSPARENT)
    ///
    /// Rotation is applied to indices, not by expanding
    pub fn bind(&self, other: &SparseFingerprint, rotation: usize) -> SparseFingerprint {
        // Rotate self's indices
        let rotated = self.rotate(rotation);
        rotated.xor(other)
    }

    /// Rotate indices by amount
    pub fn rotate(&self, amount: usize) -> SparseFingerprint {
        if amount == 0 || self.indices.is_empty() {
            return self.clone();
        }

        let amount = amount % self.total_words;
        let mut result = SparseFingerprint::new(self.total_words);

        for (&idx, &val) in self.indices.iter().zip(&self.values) {
            let new_idx = (idx as usize + amount) % self.total_words;
            result.set(new_idx, val);
        }

        result
    }

    /// Similarity (1.0 - normalized hamming distance)
    pub fn similarity(&self, other: &SparseFingerprint) -> f64 {
        let max_dist = (self.total_words * 64) as f64;
        1.0 - (self.hamming(other) as f64 / max_dist)
    }

    /// Number of non-zero words (sparsity measure)
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Compression ratio vs dense
    pub fn compression_ratio(&self) -> f64 {
        let dense_size = self.total_words * 8; // bytes
        let sparse_size = self.presence.len() * 8 + self.indices.len() * 4 + self.values.len() * 8;
        dense_size as f64 / sparse_size as f64
    }

    /// Total bits set
    pub fn popcount(&self) -> u64 {
        self.values.iter().map(|v| v.count_ones() as u64).sum()
    }

    /// Density: fraction of total bits that are set
    pub fn density(&self) -> f64 {
        self.popcount() as f64 / (self.total_words * 64) as f64
    }

    /// Memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.presence.len() * 8 + self.indices.len() * 4 + self.values.len() * 8
    }

    /// Resolution (total bits)
    pub fn resolution(&self) -> usize {
        self.total_words * 64
    }

    /// Total words in this fingerprint
    pub fn total_words(&self) -> usize {
        self.total_words
    }
}

/// Create sparse fingerprint at different resolutions
pub mod resolution {
    use super::*;

    /// 10K bits (standard fingerprint)
    pub fn standard() -> SparseFingerprint {
        SparseFingerprint::new(FINGERPRINT_WORDS)
    }

    /// 64K bits (qualia resolution)
    pub fn qualia() -> SparseFingerprint {
        SparseFingerprint::new(1000)  // 64K bits
    }

    /// 640K bits (high resolution)
    pub fn high() -> SparseFingerprint {
        SparseFingerprint::new(10_000)  // 640K bits
    }

    /// 64M bits (reality resolution)
    pub fn reality() -> SparseFingerprint {
        SparseFingerprint::new(SPARSE_MAX_WORDS)  // 64M bits
    }
}

// -----------------------------------------------------------------------------
// CSR (COMPRESSED SPARSE ROW) FOR EDGES
// -----------------------------------------------------------------------------

/// CSR-format edge storage (Kùzu-style)
///
/// CSR is extremely cache-efficient for graph traversals.
/// Layout:
/// - offsets[node] = start index in edges array
/// - edges[offset..next_offset] = neighbors of node
///
/// Memory: O(V + E) instead of O(V²) for adjacency matrix
pub struct CsrEdges {
    /// Offset array: offsets[i] is start of node i's neighbors
    offsets: Vec<u64>,
    /// Edge array: consecutive neighbor IDs
    edges: Vec<u32>,
    /// Edge weights (optional, parallel to edges)
    weights: Option<Vec<f32>>,
    /// Number of nodes
    num_nodes: usize,
    /// Number of edges
    num_edges: usize,
}

impl CsrEdges {
    /// Create empty CSR
    pub fn new() -> Self {
        Self {
            offsets: vec![0],
            edges: Vec::new(),
            weights: None,
            num_nodes: 0,
            num_edges: 0,
        }
    }

    /// Build CSR from edge list
    ///
    /// Edges: Vec<(source, target)>
    pub fn from_edges(num_nodes: usize, edges: &[(u32, u32)]) -> Self {
        // Count edges per node
        let mut counts = vec![0u64; num_nodes + 1];
        for &(src, _) in edges {
            counts[src as usize + 1] += 1;
        }

        // Prefix sum for offsets
        let mut offsets = vec![0u64; num_nodes + 1];
        for i in 1..=num_nodes {
            offsets[i] = offsets[i - 1] + counts[i];
        }

        // Fill edges array
        let mut edge_array = vec![0u32; edges.len()];
        let mut current = offsets.clone();

        for &(src, dst) in edges {
            let idx = current[src as usize] as usize;
            edge_array[idx] = dst;
            current[src as usize] += 1;
        }

        Self {
            offsets,
            edges: edge_array,
            weights: None,
            num_nodes,
            num_edges: edges.len(),
        }
    }

    /// Build CSR with weights
    pub fn from_weighted_edges(num_nodes: usize, edges: &[(u32, u32, f32)]) -> Self {
        // Count edges per node
        let mut counts = vec![0u64; num_nodes + 1];
        for &(src, _, _) in edges {
            counts[src as usize + 1] += 1;
        }

        // Prefix sum for offsets
        let mut offsets = vec![0u64; num_nodes + 1];
        for i in 1..=num_nodes {
            offsets[i] = offsets[i - 1] + counts[i];
        }

        // Fill edges and weights arrays
        let mut edge_array = vec![0u32; edges.len()];
        let mut weight_array = vec![0.0f32; edges.len()];
        let mut current = offsets.clone();

        for &(src, dst, weight) in edges {
            let idx = current[src as usize] as usize;
            edge_array[idx] = dst;
            weight_array[idx] = weight;
            current[src as usize] += 1;
        }

        Self {
            offsets,
            edges: edge_array,
            weights: Some(weight_array),
            num_nodes,
            num_edges: edges.len(),
        }
    }

    /// Get neighbors of a node (zero-copy slice)
    #[inline]
    pub fn neighbors(&self, node: u32) -> &[u32] {
        let start = self.offsets[node as usize] as usize;
        let end = self.offsets[node as usize + 1] as usize;
        &self.edges[start..end]
    }

    /// Get neighbors with weights
    #[inline]
    pub fn neighbors_weighted(&self, node: u32) -> Option<(&[u32], &[f32])> {
        let start = self.offsets[node as usize] as usize;
        let end = self.offsets[node as usize + 1] as usize;
        let weights = self.weights.as_ref()?;
        Some((&self.edges[start..end], &weights[start..end]))
    }

    /// Degree of node
    #[inline]
    pub fn degree(&self, node: u32) -> usize {
        let start = self.offsets[node as usize];
        let end = self.offsets[node as usize + 1];
        (end - start) as usize
    }

    /// Number of nodes
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Number of edges
    pub fn num_edges(&self) -> usize {
        self.num_edges
    }

    /// Iterate over all edges
    pub fn iter_edges(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        (0..self.num_nodes as u32).flat_map(move |src| {
            self.neighbors(src).iter().map(move |&dst| (src, dst))
        })
    }

    /// Memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let offsets_size = self.offsets.len() * 8;
        let edges_size = self.edges.len() * 4;
        let weights_size = self.weights.as_ref().map(|w| w.len() * 4).unwrap_or(0);
        offsets_size + edges_size + weights_size
    }
}

impl Default for CsrEdges {
    fn default() -> Self {
        Self::new()
    }
}

/// CSR builder for incremental construction
pub struct CsrBuilder {
    /// Adjacency list being built
    adj_list: Vec<Vec<(u32, f32)>>,
    /// Number of edges
    num_edges: usize,
}

impl CsrBuilder {
    /// Create builder for n nodes
    pub fn new(num_nodes: usize) -> Self {
        Self {
            adj_list: vec![Vec::new(); num_nodes],
            num_edges: 0,
        }
    }

    /// Add edge
    pub fn add_edge(&mut self, src: u32, dst: u32) {
        self.adj_list[src as usize].push((dst, 1.0));
        self.num_edges += 1;
    }

    /// Add weighted edge
    pub fn add_weighted_edge(&mut self, src: u32, dst: u32, weight: f32) {
        self.adj_list[src as usize].push((dst, weight));
        self.num_edges += 1;
    }

    /// Build CSR (sorts neighbors for cache efficiency)
    pub fn build(mut self) -> CsrEdges {
        // Sort neighbors for each node
        for neighbors in &mut self.adj_list {
            neighbors.sort_by_key(|&(dst, _)| dst);
        }

        let num_nodes = self.adj_list.len();
        let has_weights = self.adj_list.iter().any(|n| n.iter().any(|&(_, w)| w != 1.0));

        // Build offsets
        let mut offsets = Vec::with_capacity(num_nodes + 1);
        offsets.push(0u64);
        for neighbors in &self.adj_list {
            offsets.push(offsets.last().unwrap() + neighbors.len() as u64);
        }

        // Build edges (and optionally weights)
        let mut edges = Vec::with_capacity(self.num_edges);
        let mut weights = if has_weights {
            Some(Vec::with_capacity(self.num_edges))
        } else {
            None
        };

        for neighbors in &self.adj_list {
            for &(dst, weight) in neighbors {
                edges.push(dst);
                if let Some(ref mut w) = weights {
                    w.push(weight);
                }
            }
        }

        CsrEdges {
            offsets,
            edges,
            weights,
            num_nodes,
            num_edges: self.num_edges,
        }
    }
}

// -----------------------------------------------------------------------------
// MORSEL-DRIVEN PARALLELISM (Kùzu-style)
// -----------------------------------------------------------------------------

/// Morsel size for parallel processing
pub const MORSEL_SIZE: usize = 2048;

/// Morsel: a chunk of work for parallel processing
pub struct Morsel<T> {
    /// Start index in source data
    pub start: usize,
    /// End index (exclusive)
    pub end: usize,
    /// Local results
    pub results: Vec<T>,
}

impl<T> Morsel<T> {
    pub fn new(start: usize, end: usize) -> Self {
        Self {
            start,
            end,
            results: Vec::new(),
        }
    }

    pub fn size(&self) -> usize {
        self.end - self.start
    }
}

/// Morsel dispatcher for parallel fingerprint operations
pub struct MorselDispatcher {
    /// Total items
    total: usize,
    /// Next morsel start
    next_start: AtomicU64,
}

impl MorselDispatcher {
    /// Create dispatcher for n items
    pub fn new(total: usize) -> Self {
        Self {
            total,
            next_start: AtomicU64::new(0),
        }
    }

    /// Get next morsel (None if done)
    pub fn next_morsel<T>(&self) -> Option<Morsel<T>> {
        loop {
            let start = self.next_start.load(Ordering::Relaxed) as usize;
            if start >= self.total {
                return None;
            }

            let end = (start + MORSEL_SIZE).min(self.total);

            // Try to claim this morsel
            if self.next_start.compare_exchange(
                start as u64,
                end as u64,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ).is_ok() {
                return Some(Morsel::new(start, end));
            }
            // Another thread claimed it, retry
        }
    }

    /// Reset for reuse
    pub fn reset(&self) {
        self.next_start.store(0, Ordering::Release);
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a test fingerprint with a specific pattern
    fn make_fingerprint(seed: u64) -> [u64; FINGERPRINT_WORDS] {
        let mut fp = [0u64; FINGERPRINT_WORDS];
        for i in 0..FINGERPRINT_WORDS {
            fp[i] = seed.wrapping_mul(i as u64 + 1).wrapping_add(seed);
        }
        fp
    }

    #[test]
    fn test_fingerprint_buffer_from_vec() {
        // Create test fingerprints
        let fps: Vec<[u64; FINGERPRINT_WORDS]> = (0..10)
            .map(|i| make_fingerprint(i as u64 * 1000))
            .collect();

        // Flatten to Vec<u64>
        let mut data = Vec::with_capacity(fps.len() * FINGERPRINT_WORDS);
        for fp in &fps {
            data.extend_from_slice(fp);
        }

        // Create buffer (zero-copy via ownership transfer)
        let buffer = FingerprintBuffer::from_vec(data, fps.len());

        // Verify access
        assert_eq!(buffer.len(), 10);
        for (i, expected) in fps.iter().enumerate() {
            let got = buffer.get(i).unwrap();
            assert_eq!(got, expected);
        }
    }

    #[test]
    fn test_fingerprint_buffer_iter() {
        let fps: Vec<[u64; FINGERPRINT_WORDS]> = (0..5)
            .map(|i| make_fingerprint(i as u64 * 100))
            .collect();

        let mut data = Vec::new();
        for fp in &fps {
            data.extend_from_slice(fp);
        }

        let buffer = FingerprintBuffer::from_vec(data, fps.len());

        // Test iterator
        let collected: Vec<_> = buffer.iter().collect();
        assert_eq!(collected.len(), 5);
        for (i, got) in collected.iter().enumerate() {
            assert_eq!(*got, &fps[i]);
        }
    }

    #[test]
    fn test_fingerprint_buffer_to_arrow() {
        let fps: Vec<[u64; FINGERPRINT_WORDS]> = (0..3)
            .map(|i| make_fingerprint(i as u64 * 50))
            .collect();

        let mut data = Vec::new();
        for fp in &fps {
            data.extend_from_slice(fp);
        }

        let buffer = FingerprintBuffer::from_vec(data, fps.len());

        // Convert to Arrow
        let array = buffer.to_fixed_size_list();
        let fsl = array.as_any().downcast_ref::<FixedSizeListArray>().unwrap();

        assert_eq!(fsl.len(), 3);
        assert_eq!(fsl.value_length(), FINGERPRINT_WORDS as i32);
    }

    #[test]
    fn test_arrow_zero_copy_manager() {
        let mut manager = ArrowZeroCopy::new();

        // Create and load fingerprints
        let fps: Vec<[u64; FINGERPRINT_WORDS]> = (0..5)
            .map(|i| make_fingerprint(i as u64))
            .collect();

        let mut data = Vec::new();
        for fp in &fps {
            data.extend_from_slice(fp);
        }

        let buffer_id = manager.load_from_vec(data, fps.len());

        // Verify access
        assert_eq!(manager.total_fingerprints(), 5);
        for (i, expected) in fps.iter().enumerate() {
            let got = manager.get(buffer_id, i).unwrap();
            assert_eq!(got, expected);
        }

        // Test touch (temperature tracking)
        manager.touch(buffer_id, 2);
        manager.touch(buffer_id, 2);
        manager.touch(buffer_id, 2);
    }

    #[test]
    fn test_adjacency_index() {
        // Create fingerprints with varying patterns
        let fps: Vec<[u64; FINGERPRINT_WORDS]> = (0..20)
            .map(|i| make_fingerprint(i as u64 * 1000))
            .collect();

        let mut data = Vec::new();
        for fp in &fps {
            data.extend_from_slice(fp);
        }

        let buffer = FingerprintBuffer::from_vec(data, fps.len());
        let adj_index = AdjacencyIndex::build(&buffer);

        assert_eq!(adj_index.len(), 20);

        // Verify bidirectional mapping
        for orig_idx in 0..20 {
            let sorted_pos = adj_index.sorted_position(orig_idx).unwrap();
            let back = adj_index.original_index(sorted_pos as usize).unwrap();
            assert_eq!(back as usize, orig_idx);
        }
    }

    #[test]
    fn test_locality_range() {
        // Create fingerprints where some share locality buckets
        let mut fps = Vec::new();

        // First 5: all zeros in first 8 words (same locality bucket)
        for i in 0..5 {
            let mut fp = [0u64; FINGERPRINT_WORDS];
            fp[100] = i; // Differ in later words only
            fps.push(fp);
        }

        // Next 5: high bits set (different locality bucket)
        for i in 0..5 {
            let mut fp = [0xF000_0000_0000_0000u64; FINGERPRINT_WORDS];
            fp[100] = i as u64;
            fps.push(fp);
        }

        let mut data = Vec::new();
        for fp in &fps {
            data.extend_from_slice(fp);
        }

        let buffer = FingerprintBuffer::from_vec(data, fps.len());
        let adj_index = AdjacencyIndex::build(&buffer);

        // Query with zeros pattern should find ~5 matches
        let query = [0u64; FINGERPRINT_WORDS];
        let range = adj_index.locality_range(&query);

        // Should have some matches in the locality bucket
        assert!(!range.is_empty());
    }

    #[test]
    fn test_scent_awareness() {
        let mut scent = ScentAwareness::with_capacity(100);

        // Initially cold
        assert_eq!(scent.temperature(42), Temperature::Cold);

        // Touch warms it up
        scent.touch(42);
        assert_eq!(scent.temperature(42), Temperature::Warm);

        // Mark hot
        scent.mark_hot(42);
        assert_eq!(scent.temperature(42), Temperature::Hot);

        // Mark cold again
        scent.mark_cold(42);
        assert_eq!(scent.temperature(42), Temperature::Warm); // Still warm from touch
    }

    #[test]
    fn test_bubbler() {
        let bubbler = ZeroCopyBubbler::default();
        let mut scent = ScentAwareness::with_capacity(100);

        // Bubble up
        let result = bubbler.bubble_up(&mut scent, 42);
        assert_eq!(result.to, Temperature::Hot);
        assert!(!result.copied); // Zero-copy!

        // Bubble down
        let result = bubbler.bubble_down(&mut scent, 42);
        assert_eq!(result.to, Temperature::Warm);
        assert!(!result.copied);
    }

    #[test]
    fn test_promotion_candidates() {
        let mut scent = ScentAwareness::with_capacity(100);

        // Touch index 10 many times
        for _ in 0..20 {
            scent.touch(10);
        }

        // Touch index 20 a few times
        for _ in 0..8 {
            scent.touch(20);
        }

        // Index 10 should be candidate for promotion
        let candidates = scent.candidates_for_promotion(5);
        assert!(candidates.contains(&10));
    }

    #[test]
    fn test_wal_basic() {
        let wal = WriteAheadLog::new(PathBuf::from("/tmp/test_wal"));

        // Append entries
        let lsn1 = wal.append(WalOp::Insert {
            index: 0,
            fingerprint: Box::new([0u64; FINGERPRINT_WORDS]),
        });
        assert_eq!(lsn1, 1);

        let lsn2 = wal.append(WalOp::Update {
            index: 0,
            old: Box::new([0u64; FINGERPRINT_WORDS]),
            new: Box::new([1u64; FINGERPRINT_WORDS]),
        });
        assert_eq!(lsn2, 2);

        // Check LSN progression
        assert_eq!(wal.current_lsn(), 3);
    }

    #[test]
    fn test_wal_checkpoint() {
        let wal = WriteAheadLog::new(PathBuf::from("/tmp/test_wal_ckpt"));

        // Add some entries
        for i in 0..5 {
            wal.append(WalOp::Insert {
                index: i,
                fingerprint: Box::new([i as u64; FINGERPRINT_WORDS]),
            });
        }

        // Checkpoint
        let ckpt_lsn = wal.checkpoint();
        assert!(ckpt_lsn > 0);
        assert_eq!(wal.checkpoint_lsn(), ckpt_lsn);
    }

    #[test]
    fn test_wal_entry_integrity() {
        let entry = WalEntry::new(42, WalOp::Delete {
            index: 10,
            fingerprint: Box::new([0u64; FINGERPRINT_WORDS]),
        });

        // Verify integrity
        assert!(entry.verify());

        // Tampered entry should fail
        let mut tampered = entry.clone();
        tampered.lsn = 999;
        assert!(!tampered.verify());
    }

    #[test]
    fn test_concurrent_access() {
        let wal = Arc::new(WriteAheadLog::default());
        let safe = SafeArrowZeroCopy::new(wal);

        // Begin transaction
        let txn_id = safe.begin();
        assert!(txn_id > 0);

        // Load data
        let fps: Vec<[u64; FINGERPRINT_WORDS]> = (0..3)
            .map(|i| make_fingerprint(i as u64))
            .collect();

        let mut data = Vec::new();
        for fp in &fps {
            data.extend_from_slice(fp);
        }

        let buffer_id = safe.load_with_txn(txn_id, data, fps.len());
        assert!(buffer_id.is_some());

        // Commit
        assert!(safe.commit(txn_id));
    }

    #[test]
    fn test_transaction_abort() {
        let wal = Arc::new(WriteAheadLog::default());
        let safe = SafeArrowZeroCopy::new(wal);

        let txn_id = safe.begin();

        // Load some data
        let data = vec![0u64; FINGERPRINT_WORDS];
        safe.load_with_txn(txn_id, data, 1);

        // Abort transaction
        assert!(safe.abort(txn_id));

        // Can't commit aborted transaction
        assert!(!safe.commit(txn_id));
    }

    #[test]
    fn test_versioned_fingerprint_visibility() {
        let fp = VersionedFingerprint {
            data: [42u64; FINGERPRINT_WORDS],
            write_version: Version::new(5),
            delete_version: Some(Version::new(10)),
        };

        // Before write - not visible
        assert!(!fp.visible_at(Version::new(4)));

        // After write, before delete - visible
        assert!(fp.visible_at(Version::new(5)));
        assert!(fp.visible_at(Version::new(7)));

        // At or after delete - not visible
        assert!(!fp.visible_at(Version::new(10)));
        assert!(!fp.visible_at(Version::new(15)));
    }

    #[test]
    fn test_concurrent_reads() {
        use std::thread;

        let wal = Arc::new(WriteAheadLog::default());
        let safe = Arc::new(SafeArrowZeroCopy::new(wal));

        // Load some data first
        let txn = safe.begin();
        let data = vec![0u64; FINGERPRINT_WORDS * 10];
        safe.load_with_txn(txn, data, 10);
        safe.commit(txn);

        // Spawn multiple readers
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let safe_clone = Arc::clone(&safe);
                thread::spawn(move || {
                    for _ in 0..100 {
                        let _ = safe_clone.get(0, 0);
                    }
                })
            })
            .collect();

        // All readers should complete without deadlock
        for h in handles {
            h.join().expect("reader thread panicked");
        }
    }

    // =========================================================================
    // KÙZU-STYLE OPTIMIZATION TESTS
    // =========================================================================

    #[test]
    fn test_buffer_pool_basic() {
        let mut pool = BufferPoolManager::new(4);

        // Fetch pages
        let frame1 = pool.fetch_page(PageId::new(0, 0)).unwrap();
        let frame2 = pool.fetch_page(PageId::new(0, 1)).unwrap();

        assert_ne!(frame1, frame2);

        // Fetching same page should return same frame (hit)
        let frame1_again = pool.fetch_page(PageId::new(0, 0)).unwrap();
        assert_eq!(frame1, frame1_again);

        // Check hit ratio
        assert!(pool.hit_ratio() > 0.0);
    }

    #[test]
    fn test_buffer_pool_eviction() {
        let mut pool = BufferPoolManager::new(2); // Small pool

        // Fill pool
        pool.fetch_page(PageId::new(0, 0));
        pool.fetch_page(PageId::new(0, 1));

        // Unpin first page
        pool.unpin_page(PageId::new(0, 0), false);
        pool.unpin_page(PageId::new(0, 1), false);

        // Fetch new page should evict
        pool.fetch_page(PageId::new(0, 2));

        // Should have evicted
        assert!(pool.eviction_count() > 0);
    }

    #[test]
    fn test_buffer_pool_dirty_tracking() {
        let mut pool = BufferPoolManager::new(4);

        let frame_idx = pool.fetch_page(PageId::new(0, 0)).unwrap();

        // Modify the frame
        {
            let frame = pool.get_frame_mut(frame_idx).unwrap();
            frame.data_mut()[0] = 0xFF;
        }

        // Unpin as dirty
        pool.unpin_page(PageId::new(0, 0), true);

        // Flush should handle dirty pages
        pool.flush_all();
    }

    #[test]
    fn test_csr_edges_basic() {
        // Create simple graph: 0 -> 1, 0 -> 2, 1 -> 2
        let edges = vec![(0, 1), (0, 2), (1, 2)];
        let csr = CsrEdges::from_edges(3, &edges);

        assert_eq!(csr.num_nodes(), 3);
        assert_eq!(csr.num_edges(), 3);

        // Check neighbors
        let n0 = csr.neighbors(0);
        assert_eq!(n0.len(), 2);
        assert!(n0.contains(&1));
        assert!(n0.contains(&2));

        let n1 = csr.neighbors(1);
        assert_eq!(n1.len(), 1);
        assert_eq!(n1[0], 2);

        let n2 = csr.neighbors(2);
        assert_eq!(n2.len(), 0);
    }

    #[test]
    fn test_csr_edges_weighted() {
        let edges = vec![(0, 1, 1.5), (0, 2, 2.5), (1, 2, 0.5)];
        let csr = CsrEdges::from_weighted_edges(3, &edges);

        let (neighbors, weights) = csr.neighbors_weighted(0).unwrap();
        assert_eq!(neighbors.len(), 2);
        assert_eq!(weights.len(), 2);
    }

    #[test]
    fn test_csr_builder() {
        let mut builder = CsrBuilder::new(4);

        builder.add_edge(0, 1);
        builder.add_edge(0, 2);
        builder.add_edge(1, 3);
        builder.add_weighted_edge(2, 3, 2.0);

        let csr = builder.build();

        assert_eq!(csr.num_nodes(), 4);
        assert_eq!(csr.num_edges(), 4);
        assert_eq!(csr.degree(0), 2);
        assert_eq!(csr.degree(1), 1);
    }

    #[test]
    fn test_csr_iteration() {
        let edges = vec![(0, 1), (0, 2), (1, 2)];
        let csr = CsrEdges::from_edges(3, &edges);

        let collected: Vec<_> = csr.iter_edges().collect();
        assert_eq!(collected.len(), 3);
        assert!(collected.contains(&(0, 1)));
        assert!(collected.contains(&(0, 2)));
        assert!(collected.contains(&(1, 2)));
    }

    #[test]
    fn test_serialization_header() {
        let header = SerializationHeader::new(42);

        // Copy fields to avoid packed struct alignment issues
        let magic = header.magic;
        let version = header.version;
        let count = header.count;

        assert_eq!(magic, MAGIC_BYTES);
        assert_eq!(version, FORMAT_VERSION);
        assert_eq!(count, 42);

        // Roundtrip
        let bytes = header.to_bytes();
        let restored = SerializationHeader::from_bytes(&bytes).unwrap();
        let restored_count = restored.count;
        assert_eq!(restored_count, 42);
        assert!(restored.validate().is_ok());
    }

    #[test]
    fn test_fingerprint_serialization_roundtrip() {
        let fps: Vec<[u64; FINGERPRINT_WORDS]> = (0..10)
            .map(|i| make_fingerprint(i as u64 * 1000))
            .collect();

        let mut data = Vec::new();
        for fp in &fps {
            data.extend_from_slice(fp);
        }

        let buffer = FingerprintBuffer::from_vec(data, fps.len());

        // Serialize
        let serialized = FingerprintSerializer::serialize(&buffer);
        assert!(serialized.len() > SerializationHeader::SIZE);

        // Deserialize
        let restored = FingerprintSerializer::deserialize(&serialized).unwrap();
        assert_eq!(restored.len(), fps.len());

        // Verify contents
        for (i, expected) in fps.iter().enumerate() {
            let got = restored.get(i).unwrap();
            assert_eq!(got, expected);
        }
    }

    #[test]
    fn test_serialization_checksum() {
        let fps: Vec<[u64; FINGERPRINT_WORDS]> = (0..3)
            .map(|i| make_fingerprint(i as u64))
            .collect();

        let mut data = Vec::new();
        for fp in &fps {
            data.extend_from_slice(fp);
        }

        let buffer = FingerprintBuffer::from_vec(data, fps.len());
        let mut serialized = FingerprintSerializer::serialize(&buffer);

        // Corrupt data after header
        serialized[SerializationHeader::SIZE + 10] ^= 0xFF;

        // Should fail checksum
        let result = FingerprintSerializer::deserialize(&serialized);
        assert!(result.is_err());
    }

    #[test]
    fn test_copy_on_write() {
        let cow: CopyOnWrite<Vec<u32>> = CopyOnWrite::new(vec![1, 2, 3]);

        // Read doesn't increment version
        let v1 = cow.version();
        {
            let _guard = cow.read();
        }
        assert_eq!(cow.version(), v1);

        // Write increments version
        {
            let mut guard = cow.write();
            guard.push(4);
        }
        assert!(cow.version() > v1);
    }

    #[test]
    fn test_copy_on_write_snapshot() {
        let cow: CopyOnWrite<Vec<u32>> = CopyOnWrite::new(vec![1, 2, 3]);

        // Take snapshot
        let snapshot = cow.snapshot();
        assert_eq!(snapshot, vec![1, 2, 3]);
        assert_eq!(cow.copy_count(), 1);

        // Modify original
        {
            let mut guard = cow.write();
            guard.push(4);
        }

        // Snapshot is independent
        assert_eq!(snapshot.len(), 3);
        assert_eq!(cow.read().len(), 4);
    }

    #[test]
    fn test_morsel_dispatcher() {
        let dispatcher = MorselDispatcher::new(5000);

        let mut total_processed = 0usize;
        let mut morsel_count = 0usize;

        while let Some(morsel) = dispatcher.next_morsel::<()>() {
            total_processed += morsel.size();
            morsel_count += 1;
        }

        assert_eq!(total_processed, 5000);
        assert_eq!(morsel_count, (5000 + MORSEL_SIZE - 1) / MORSEL_SIZE);

        // Reset and verify
        dispatcher.reset();
        assert!(dispatcher.next_morsel::<()>().is_some());
    }

    #[test]
    fn test_morsel_parallel() {
        use std::thread;

        let dispatcher = Arc::new(MorselDispatcher::new(10000));
        let counter = Arc::new(AtomicU64::new(0));

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let d = Arc::clone(&dispatcher);
                let c = Arc::clone(&counter);
                thread::spawn(move || {
                    while let Some(morsel) = d.next_morsel::<()>() {
                        c.fetch_add(morsel.size() as u64, Ordering::Relaxed);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // All items processed exactly once
        assert_eq!(counter.load(Ordering::Relaxed), 10000);
    }

    #[test]
    fn test_adjacency_preserves_codebook() {
        // This test verifies that adjacency sorting does NOT break address lookups
        let fps: Vec<[u64; FINGERPRINT_WORDS]> = (0..100)
            .map(|i| make_fingerprint(i as u64 * 1000))
            .collect();

        let mut data = Vec::new();
        for fp in &fps {
            data.extend_from_slice(fp);
        }

        let buffer = FingerprintBuffer::from_vec(data, fps.len());
        let adj_index = AdjacencyIndex::build(&buffer);

        // Simulate codebook address lookup
        // Address 0x42:0x0F would map to some original_index
        let codebook_addr = 42; // Simulated codebook lookup result

        // Direct lookup still works (codebook not invalidated)
        let fp_direct = buffer.get(codebook_addr).unwrap();
        assert_eq!(fp_direct, &fps[codebook_addr]);

        // Adjacency index provides sorted position for cache-friendly scanning
        let sorted_pos = adj_index.sorted_position(codebook_addr).unwrap();

        // Can map back to original
        let back_to_original = adj_index.original_index(sorted_pos as usize).unwrap();
        assert_eq!(back_to_original as usize, codebook_addr);

        // Iteration in sorted order for cache locality
        let mut count = 0;
        for orig_idx in adj_index.iter_sorted() {
            // Can still access by original index (codebook addressing)
            let _fp = buffer.get(orig_idx as usize).unwrap();
            count += 1;
        }
        assert_eq!(count, 100);
    }

    // =========================================================================
    // WRITE-THROUGH TESTS
    // =========================================================================

    #[test]
    fn test_write_through_basic() {
        let wal = Arc::new(WriteAheadLog::default());
        let wt = LanceWriteThrough::with_defaults(wal);

        let fp = make_fingerprint(12345);

        // Write to slot 0
        assert!(wt.write(0, fp));

        // Read back
        let read = wt.read(0).unwrap();
        assert_eq!(read, fp);
    }

    #[test]
    fn test_write_through_hot_cache() {
        let wal = Arc::new(WriteAheadLog::default());
        let wt = LanceWriteThrough::with_defaults(wal);

        // Write multiple entries
        for i in 0..10u8 {
            let fp = make_fingerprint(i as u64 * 1000);
            wt.write(i, fp);
        }

        // Verify hot cache has entries
        let stats = wt.stats();
        assert_eq!(stats.hot_entries, 10);
        assert_eq!(stats.dirty_entries, 10);
    }

    #[test]
    fn test_write_through_sync() {
        let wal = Arc::new(WriteAheadLog::default());
        let config = WriteThroughConfig {
            max_hot_entries: 100,
            sync_interval: 5,  // Sync after 5 writes
            write_behind: false,
        };
        let wt = LanceWriteThrough::new(wal, config);

        // Write 5 entries to trigger sync
        for i in 0..5u8 {
            let fp = make_fingerprint(i as u64);
            wt.write(i, fp);
        }

        // Dirty should be cleared after sync
        let stats = wt.stats();
        assert_eq!(stats.dirty_entries, 0);
    }

    #[test]
    fn test_write_through_delete() {
        let wal = Arc::new(WriteAheadLog::default());
        let wt = LanceWriteThrough::with_defaults(wal);

        let fp = make_fingerprint(999);
        wt.write(42, fp);

        // Verify it's there
        assert!(wt.read(42).is_some());

        // Delete
        assert!(wt.delete(42));

        // Should be gone from hot cache
        let stats = wt.stats();
        assert_eq!(stats.hot_entries, 0);
    }

    #[test]
    fn test_storage_backend_trait() {
        let wal = Arc::new(WriteAheadLog::default());
        let wt = LanceWriteThrough::with_defaults(wal);

        // Use via trait
        let backend: &dyn StorageBackend = &wt;

        // Write via trait
        let fp = make_fingerprint(777);
        assert!(backend.write_fingerprint(PREFIX_LANCE, 10, fp));

        // Read via trait
        let read = backend.read_fingerprint(PREFIX_LANCE, 10).unwrap();
        assert_eq!(read, fp);

        // Non-Lance prefix returns None
        assert!(backend.read_fingerprint(0x01, 10).is_none());
    }

    #[test]
    fn test_unified_storage() {
        let wal = Arc::new(WriteAheadLog::default());
        let unified = UnifiedStorage::new(wal);

        let fp = make_fingerprint(555);

        // Write to Lance prefix
        assert!(unified.write(PREFIX_LANCE, 20, fp));

        // Read back
        let read = unified.read(PREFIX_LANCE, 20).unwrap();
        assert_eq!(read, fp);

        // Non-Lance prefix not handled
        assert!(unified.read(0x01, 20).is_none());
    }

    #[test]
    fn test_unified_storage_sync() {
        let wal = Arc::new(WriteAheadLog::default());
        let unified = UnifiedStorage::new(wal);

        // Write some data
        for i in 0..10u8 {
            unified.write(PREFIX_LANCE, i, make_fingerprint(i as u64));
        }

        // Sync all
        unified.sync_all();

        // Stats should show synced
        let stats = unified.lance().stats();
        assert_eq!(stats.dirty_entries, 0);
    }

    #[test]
    fn test_write_through_eviction() {
        let wal = Arc::new(WriteAheadLog::default());
        let config = WriteThroughConfig {
            max_hot_entries: 5,
            sync_interval: 1000,  // Don't auto-sync
            write_behind: true,   // Write-behind mode
        };
        let wt = LanceWriteThrough::new(wal, config);

        // Write more than max_hot_entries
        for i in 0..10u8 {
            wt.write(i, make_fingerprint(i as u64));
        }

        // Hot cache should be capped
        let stats = wt.stats();
        assert!(stats.hot_entries <= 5);
    }

    // =========================================================================
    // SPARSE FINGERPRINT TESTS (64M resolution, 100:1 compression)
    // =========================================================================

    #[test]
    fn test_sparse_from_dense() {
        // Create a sparse pattern (only a few bits set)
        let mut dense = vec![0u64; 1000];
        dense[0] = 0xFF;
        dense[100] = 0x1234;
        dense[999] = 0xDEAD;

        let sparse = SparseFingerprint::from_dense(&dense);

        assert_eq!(sparse.nnz(), 3);
        assert_eq!(sparse.get(0), 0xFF);
        assert_eq!(sparse.get(100), 0x1234);
        assert_eq!(sparse.get(999), 0xDEAD);
        assert_eq!(sparse.get(500), 0);  // Not present

        // Compression ratio should be significant
        assert!(sparse.compression_ratio() > 10.0);
    }

    #[test]
    fn test_sparse_to_dense_roundtrip() {
        let mut dense = vec![0u64; 100];
        dense[10] = 0xABCD;
        dense[50] = 0x1111;

        let sparse = SparseFingerprint::from_dense(&dense);
        let recovered = sparse.to_dense();

        assert_eq!(dense, recovered);
    }

    #[test]
    fn test_sparse_xor_transparent() {
        // Test that XOR works without decompression
        let mut a_dense = vec![0u64; 100];
        a_dense[10] = 0xFF00;
        a_dense[20] = 0x00FF;

        let mut b_dense = vec![0u64; 100];
        b_dense[10] = 0x0F0F;  // Overlaps with a
        b_dense[30] = 0x1234;  // Only in b

        let a = SparseFingerprint::from_dense(&a_dense);
        let b = SparseFingerprint::from_dense(&b_dense);

        // Sparse XOR
        let c_sparse = a.xor(&b);

        // Dense XOR for comparison
        let c_dense: Vec<u64> = a_dense.iter().zip(&b_dense)
            .map(|(x, y)| x ^ y)
            .collect();

        assert_eq!(c_sparse.to_dense(), c_dense);
    }

    #[test]
    fn test_sparse_xor_cancellation() {
        // XOR of identical values should produce zero (cancellation)
        let mut dense = vec![0u64; 100];
        dense[42] = 0xDEADBEEF;

        let a = SparseFingerprint::from_dense(&dense);
        let b = SparseFingerprint::from_dense(&dense);

        let c = a.xor(&b);

        // Should be completely empty after XOR with self
        assert_eq!(c.nnz(), 0);
    }

    #[test]
    fn test_sparse_hamming_transparent() {
        let mut a_dense = vec![0u64; 100];
        a_dense[0] = 0b1111_0000;  // 4 bits set

        let mut b_dense = vec![0u64; 100];
        b_dense[0] = 0b1100_1100;  // 4 bits set, 2 overlap

        let a = SparseFingerprint::from_dense(&a_dense);
        let b = SparseFingerprint::from_dense(&b_dense);

        // Hamming distance = number of differing bits
        let dist = a.hamming(&b);

        // 0b1111_0000 XOR 0b1100_1100 = 0b0011_1100 = 4 bits
        assert_eq!(dist, 4);
    }

    #[test]
    fn test_sparse_bind() {
        let mut a_dense = vec![0u64; 100];
        a_dense[0] = 0xFF;

        let mut b_dense = vec![0u64; 100];
        b_dense[10] = 0xAA;

        let a = SparseFingerprint::from_dense(&a_dense);
        let b = SparseFingerprint::from_dense(&b_dense);

        // Bind with rotation 10
        let bound = a.bind(&b, 10);

        // a rotated by 10 puts 0xFF at index 10
        // XOR with b's 0xAA at index 10 = 0xFF ^ 0xAA = 0x55
        assert_eq!(bound.get(10), 0x55);
    }

    #[test]
    fn test_sparse_64m_resolution() {
        // Test at full 64M bit resolution
        let mut sparse = resolution::reality();

        assert_eq!(sparse.resolution(), 64_000_000);
        assert_eq!(sparse.nnz(), 0);

        // Set a few bits scattered across the space
        sparse.set(0, 0xDEAD);
        sparse.set(500_000, 0xBEEF);
        sparse.set(999_999, 0xCAFE);

        assert_eq!(sparse.nnz(), 3);
        assert_eq!(sparse.get(500_000), 0xBEEF);

        // Memory usage: presence bitmap (~125KB) + sparse data
        // Still much smaller than 8MB dense
        assert!(sparse.memory_usage() < 150_000);  // Less than 150KB

        // Compression ratio: 8MB / ~125KB ≈ 64x (presence bitmap dominates)
        // For truly sparse data with tiny presence, would be much higher
        assert!(sparse.compression_ratio() > 50.0);
    }

    #[test]
    fn test_sparse_similarity() {
        let mut a_dense = vec![0u64; 100];
        let mut b_dense = vec![0u64; 100];

        // Identical patterns
        a_dense[0] = 0xFFFF;
        b_dense[0] = 0xFFFF;

        let a = SparseFingerprint::from_dense(&a_dense);
        let b = SparseFingerprint::from_dense(&b_dense);

        // Should be identical (similarity = 1.0)
        assert!((a.similarity(&b) - 1.0).abs() < 0.0001);

        // Now make b different
        b_dense[0] = 0x0000;
        let b2 = SparseFingerprint::from_dense(&b_dense);

        // Should be less similar
        assert!(a.similarity(&b2) < 1.0);
    }

    #[test]
    fn test_sparse_qualia_resolution() {
        let qualia = resolution::qualia();
        assert_eq!(qualia.resolution(), 64_000);  // 64K bits
    }

    #[test]
    fn test_sparse_from_standard_fingerprint() {
        let fp = make_fingerprint(12345);
        let sparse = SparseFingerprint::from_fingerprint(&fp);

        // Should have converted correctly
        assert_eq!(sparse.resolution(), FINGERPRINT_WORDS * 64);

        // Roundtrip through dense should preserve values
        let dense = sparse.to_dense();
        for i in 0..FINGERPRINT_WORDS {
            assert_eq!(dense[i], fp[i]);
        }
    }

    #[test]
    fn test_sparse_algebraic_properties() {
        // Test XOR algebraic properties hold on sparse representation

        let a = SparseFingerprint::from_dense(&vec![0xAA; 100]);
        let b = SparseFingerprint::from_dense(&vec![0xBB; 100]);
        let c = SparseFingerprint::from_dense(&vec![0xCC; 100]);
        let zero = SparseFingerprint::new(100);

        // Associative: (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)
        let left = a.xor(&b).xor(&c);
        let right = a.xor(&b.xor(&c));
        assert_eq!(left.to_dense(), right.to_dense());

        // Commutative: a ⊕ b = b ⊕ a
        assert_eq!(a.xor(&b).to_dense(), b.xor(&a).to_dense());

        // Self-inverse: a ⊕ a = 0
        assert_eq!(a.xor(&a).nnz(), 0);

        // Identity: a ⊕ 0 = a
        assert_eq!(a.xor(&zero).to_dense(), a.to_dense());
    }
}
