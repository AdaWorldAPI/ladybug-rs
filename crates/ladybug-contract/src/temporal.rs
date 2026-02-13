//! Temporal types — versioning, diffs, and transaction state.
//!
//! Pure data types only. The actual `TemporalStore` runtime stays in ladybug-rs.

/// Version identifier (monotonically increasing).
pub type Version = u64;

/// Timestamp in microseconds since epoch.
pub type Timestamp = u64;

/// Transaction ID.
pub type TxnId = u64;

/// Transaction isolation level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IsolationLevel {
    /// See committed data from other transactions.
    ReadCommitted,
    /// Snapshot at transaction start.
    RepeatableRead,
    /// Full isolation (conflicts fail).
    Serializable,
}

/// Transaction state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TxnState {
    Active,
    Committed,
    Aborted,
}

/// Diff between two versions of the store.
#[derive(Clone, Debug)]
pub struct VersionDiff {
    pub from: Version,
    pub to: Version,
    pub added_count: usize,
    pub removed_count: usize,
    pub modified_count: usize,
}

impl VersionDiff {
    pub fn change_count(&self) -> usize {
        self.added_count + self.removed_count + self.modified_count
    }

    pub fn is_empty(&self) -> bool {
        self.change_count() == 0
    }
}

/// Temporal error types.
#[derive(Clone, Debug)]
pub enum TemporalError {
    TxnNotFound(TxnId),
    TxnNotActive(TxnId),
    Conflict {
        txn_id: TxnId,
        addr: u16,
        conflicting_version: Version,
    },
    LockError,
    VersionNotFound(Version),
    InvalidOperation(String),
}

impl std::fmt::Display for TemporalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TemporalError::TxnNotFound(id) => write!(f, "transaction {} not found", id),
            TemporalError::TxnNotActive(id) => write!(f, "transaction {} not active", id),
            TemporalError::Conflict {
                txn_id,
                addr,
                conflicting_version,
            } => write!(
                f,
                "conflict in txn {} at addr {} (version {})",
                txn_id, addr, conflicting_version
            ),
            TemporalError::LockError => write!(f, "lock error"),
            TemporalError::VersionNotFound(v) => write!(f, "version {} not found", v),
            TemporalError::InvalidOperation(s) => write!(f, "invalid operation: {}", s),
        }
    }
}

impl std::error::Error for TemporalError {}
