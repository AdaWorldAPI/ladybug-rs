//! Storage module - Persistence layers
//!
//! # 8-bit Prefix : 8-bit Slot Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                      PREFIX (8-bit) : SLOT (8-bit)                          │
//! ├─────────────────┬───────────────────────────────────────────────────────────┤
//! │  0x00-0x0F:XX   │  SURFACE (16 × 256 = 4,096)                               │
//! │                 │  0x00: Lance    0x04: NARS      0x08: Concepts            │
//! │                 │  0x01: SQL      0x05: Causal    0x09: Qualia              │
//! │                 │  0x02: Cypher   0x06: Meta      0x0A: Memory              │
//! │                 │  0x03: GraphQL  0x07: Verbs     0x0B: Learning            │
//! ├─────────────────┼───────────────────────────────────────────────────────────┤
//! │  0x10-0x7F:XX   │  FLUID (112 × 256 = 28,672)                               │
//! ├─────────────────┼───────────────────────────────────────────────────────────┤
//! │  0x80-0xFF:XX   │  NODES (128 × 256 = 32,768) - UNIVERSAL BIND SPACE        │
//! └─────────────────┴───────────────────────────────────────────────────────────┘
//! ```
//!
//! Pure array indexing. No HashMap. 3-5 cycles per lookup.
//! Works on any CPU - no SIMD required.
//!
//! # Layers
//!
//! - **BindSpace**: Universal DTO - all languages hit this
//! - **CogRedis**: Redis syntax adapter with cognitive semantics
//! - **LanceDB**: Vector storage (fingerprints, embeddings)
//! - **Database**: Unified query interface

pub mod bind_space;
pub mod cog_redis;
pub mod concurrency;
pub mod corpus;
#[cfg(feature = "lancedb")]
pub mod database;
pub mod fingerprint_dict;
pub mod hardening;
#[cfg(feature = "lancedb")]
pub mod lance;
pub mod lance_zero_copy;
pub mod redis_adapter;
pub mod resilient;
pub mod service;
pub mod snapshots;
pub mod substrate;
pub mod temporal;
pub mod unified_engine;
pub mod xor_dag;

#[cfg(feature = "lancedb")]
pub use database::Database;
#[cfg(feature = "lancedb")]
pub use lance::{EdgeRecord, LanceStore, NodeRecord};

// CogRedis exports
pub use cog_redis::{
    // Size constants
    CHUNK_SIZE,
    // Address types
    CogAddr,
    CogEdge,

    // Main interface
    CogRedis,
    CogRedisStats,

    // Values and edges
    CogValue,
    DeduceResult,
    FLUID_END,
    FLUID_PREFIXES,

    FLUID_SIZE,
    FLUID_START,
    // Results
    GetResult,
    // Production-hardened version
    HardenedCogRedis,
    NODE_END,

    NODE_PREFIXES,

    NODE_SIZE,
    NODE_START,
    PREFIX_CAUSAL,
    PREFIX_CONCEPTS,
    PREFIX_CYPHER,
    PREFIX_FLUID_END,
    // Fluid prefix constants (112 chunks)
    PREFIX_FLUID_START,
    PREFIX_GRAPHQL,
    PREFIX_LANCE,
    PREFIX_LEARNING,

    PREFIX_MEMORY,
    PREFIX_META,
    PREFIX_NARS,
    PREFIX_NODE_END,
    // Node prefix constants (128 chunks)
    PREFIX_NODE_START,
    PREFIX_QUALIA,
    PREFIX_SQL,
    PREFIX_SURFACE_END,
    // Surface prefix constants (16 compartments)
    PREFIX_SURFACE_START,
    PREFIX_VERBS,
    RedisResult,

    ResonateResult,
    SURFACE_END,
    SURFACE_PREFIXES,
    SURFACE_SIZE,
    // Legacy 16-bit range constants (compatibility)
    SURFACE_START,
    SetOptions,
    SurfaceCompartment,

    TOTAL_SIZE,

    Tier,
};

// BindSpace exports (universal DTO)
pub use bind_space::{
    Addr, BindEdge, BindNode, BindSpace, BindSpaceStats, ChunkContext, FINGERPRINT_WORDS,
    QueryAdapter, QueryResult, QueryValue, hamming_distance,
};

// Hardening exports (production-ready features)
pub use hardening::{
    HardenedBindSpace, HardeningConfig, HardeningMetrics, LruTracker, MetricsSnapshot,
    QueryContext, QueryTimeoutError, TtlManager, WalEntry, WriteAheadLog,
};

// Temporal exports (ACID, time travel, what-if)
pub use temporal::{
    IsolationLevel,
    // Full-featured CogRedis
    TemporalCogRedis,
    TemporalEdge,
    TemporalEntry,
    // Errors
    TemporalError,
    TemporalStats,
    TemporalStore,
    Timestamp,
    Transaction,
    TxnId,
    TxnState,
    // Types
    Version,
    VersionDiff,
    // Stores
    VersionManager,
    // What-if
    WhatIfBranch,
};

// Resilient exports (ReFS-like hardening)
pub use resilient::{
    // Errors
    BufferError,
    BufferedDelete,
    BufferedLink,
    BufferedWrite,
    // Dependency tracking
    DependencyGraph,
    ReadResult,
    RecoveryAction,
    // Recovery
    RecoveryEngine,
    // Config
    ResilienceConfig,
    ResilientStatus,
    // Store
    ResilientStore,
    // Buffer types
    VirtualVersion,
    WriteBuffer,
    WriteState,
};

// Concurrency exports (MVCC, memory pool, parallel execution)
pub use concurrency::{
    ConcurrentStats,
    // Combined store
    ConcurrentStore,
    ConflictError,
    MemoryError,
    MemoryGuard,
    MemoryPool,
    // Memory pool
    MemoryPoolConfig,
    MemoryPoolStats,
    // MVCC
    MvccSlot,
    MvccStore,
    // Parallel execution
    ParallelConfig,
    ParallelError,
    ParallelExecutor,
    // Query context
    QueryContext as ConcurrentQueryContext,
    ReadHandle,
    ResultHandle,
    WriteConflict,
    WriteIntent,
    WriteResult,
};

// Snapshot exports (VMware-like XOR delta snapshots, cold storage)
pub use snapshots::{
    ColdStorageConfig,
    // XOR deltas
    DeltaBlock,
    MigrationStats,
    Snapshot,
    SnapshotChain,
    SnapshotError,
    SnapshotId,
    SnapshotStats,
    // Tiered storage
    StorageTier,
    TieredEntry,
    TieredStorage,
    TieredStorageStats,
};

// Corpus exports (scent-indexed Arrow FS, Gutenberg chunking)
pub use corpus::{
    Chunk,
    // Chunking
    ChunkStrategy,
    // Store
    CorpusConfig,
    // Errors
    CorpusError,
    CorpusStats,
    CorpusStore,
    Document,
    // Documents
    DocumentMeta,
    // Fingerprinting
    Fingerprint,
    chunk_document,
    fingerprint_similarity,
    generate_fingerprint,
    hamming_distance as corpus_hamming_distance,
    load_gutenberg_book,
    // Gutenberg
    parse_gutenberg,
};

// Service exports (container lifecycle, DuckDB-inspired buffering)
pub use service::{
    // Compile hints
    AVX512_RUSTFLAGS,
    // Schema hydration
    AddressSchema,
    BufferPool,
    // Buffer pool (DuckDB-style)
    BufferPoolConfig,
    BufferPoolStats,
    CognitiveService,
    // Column operations (Lance-style)
    ColumnBatch,
    ColumnType,
    // CPU detection
    CpuFeatures,
    DOCKER_BUILD_CMD,
    // Vectorized batching
    DataBatch,
    // Recovery
    RecoveryManifest,
    // Service container
    ServiceConfig,
    // Errors
    ServiceError,
    ServiceHealth,
    ZoneDescriptor,
    batch_hamming_distance,
};

// Substrate exports (unified interface DTO)
pub use substrate::{
    // Main interface
    Substrate,
    // Configuration
    SubstrateConfig,
    SubstrateEdge,
    // Node and edge types
    SubstrateNode,
    // Statistics
    SubstrateStats,
    WriteBuffer as SubstrateWriteBuffer,
    // Write operations
    WriteOp,
};

// Lance zero-copy exports (unified address space)
pub use lance_zero_copy::{
    // Bubbling operations
    BubbleResult,
    // Zero-copy view into Lance
    LanceView,
    ScentAwareness,
    // Temperature tracking for bubbling
    Temperature,
    ZeroCopyBubbler,
};

// XOR DAG exports (ACID transactions, parity protection, work stealing)
pub use xor_dag::{
    // Conflict resolution (aliased to avoid conflict with concurrency module)
    ConflictStrategy,
    // Errors
    DagError,
    DagTransaction,
    // Epoch-based work stealing
    EpochGuard,
    EpochTicket,
    // Parity protection
    ParityBlock,
    ParityTier,
    StateSnapshot,
    // ACID transactions (aliased to avoid conflict with temporal module)
    TxnState as DagTxnState,
    WorkItem,
    WorkOperation,
    WriteConflict as DagWriteConflict,
    WriteIntent as DagWriteIntent,
    XorDag,
    // XOR DAG store
    XorDagConfig,
    XorDagStatsSnapshot,
    XorDelta,
    xor_fingerprints,
    xor_fingerprints_multi,
    // Zero-copy XOR operations
    xor_slices,
    xor_slices_inplace,
};

// Redis Adapter exports (Redis syntax interface)
pub use redis_adapter::{
    CamResult,
    DeleteMode,
    EdgeResult,
    NodeResult,
    // Main adapter
    RedisAdapter,
    // Command types
    RedisCommand,
    // Result types (aliased to avoid conflict with cog_redis::RedisResult)
    RedisResult as AdapterRedisResult,
    SearchHit,
    SetOptions as RedisSetOptions,
};

// Unified Engine exports (all features through BindSpace)
pub use unified_engine::{
    // Configuration
    UnifiedConfig,
    // Main engine
    UnifiedEngine,
    // Errors
    UnifiedError,
    // Statistics
    UnifiedStatsSnapshot,
};

// Fingerprint Dictionary exports (codebook for sparse hydration)
pub use fingerprint_dict::FingerprintDict;
