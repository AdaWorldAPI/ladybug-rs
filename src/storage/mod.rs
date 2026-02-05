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

#[cfg(feature = "lancedb")]
pub mod lance;
#[cfg(feature = "lancedb")]
pub mod database;
pub mod cog_redis;
pub mod bind_space;
pub mod hardening;
pub mod temporal;
pub mod resilient;
pub mod concurrency;
pub mod snapshots;
pub mod corpus;
pub mod service;
pub mod substrate;
pub mod redis_adapter;
pub mod lance_zero_copy;
pub mod xor_dag;
pub mod unified_engine;
pub mod fingerprint_dict;

#[cfg(feature = "lancedb")]
pub use lance::{LanceStore, NodeRecord, EdgeRecord};
#[cfg(feature = "lancedb")]
pub use database::Database;

// CogRedis exports
pub use cog_redis::{
    // Address types
    CogAddr, Tier, SurfaceCompartment,
    
    // Surface prefix constants (16 compartments)
    PREFIX_SURFACE_START, PREFIX_SURFACE_END, SURFACE_PREFIXES,
    PREFIX_LANCE, PREFIX_SQL, PREFIX_CYPHER, PREFIX_GRAPHQL,
    PREFIX_NARS, PREFIX_CAUSAL, PREFIX_META, PREFIX_VERBS,
    PREFIX_CONCEPTS, PREFIX_QUALIA, PREFIX_MEMORY, PREFIX_LEARNING,
    
    // Fluid prefix constants (112 chunks)
    PREFIX_FLUID_START, PREFIX_FLUID_END, FLUID_PREFIXES,
    
    // Node prefix constants (128 chunks)
    PREFIX_NODE_START, PREFIX_NODE_END, NODE_PREFIXES,
    
    // Size constants
    CHUNK_SIZE, SURFACE_SIZE, FLUID_SIZE, NODE_SIZE, TOTAL_SIZE,
    
    // Legacy 16-bit range constants (compatibility)
    SURFACE_START, SURFACE_END,
    FLUID_START, FLUID_END,
    NODE_START, NODE_END,
    
    // Values and edges
    CogValue, CogEdge,
    
    // Main interface
    CogRedis, CogRedisStats,
    
    // Results
    GetResult, SetOptions, ResonateResult, DeduceResult,
    RedisResult,

    // Production-hardened version
    HardenedCogRedis,
};

// BindSpace exports (universal DTO)
pub use bind_space::{
    Addr, BindNode, BindEdge, BindSpace, BindSpaceStats,
    ChunkContext, QueryAdapter, QueryResult, QueryValue,
    hamming_distance, FINGERPRINT_WORDS,
};

// Hardening exports (production-ready features)
pub use hardening::{
    HardeningConfig, HardenedBindSpace,
    LruTracker, TtlManager,
    WriteAheadLog, WalEntry,
    QueryContext, QueryTimeoutError,
    HardeningMetrics, MetricsSnapshot,
};

// Temporal exports (ACID, time travel, what-if)
pub use temporal::{
    // Types
    Version, Timestamp, TxnId,
    IsolationLevel, TxnState,
    TemporalEntry, TemporalEdge,
    Transaction,
    // Stores
    VersionManager, TemporalStore,
    // What-if
    WhatIfBranch, VersionDiff,
    // Errors
    TemporalError,
    // Full-featured CogRedis
    TemporalCogRedis, TemporalStats,
};

// Resilient exports (ReFS-like hardening)
pub use resilient::{
    // Config
    ResilienceConfig,
    // Buffer types
    VirtualVersion, WriteState,
    BufferedWrite, BufferedDelete, BufferedLink,
    WriteBuffer,
    // Dependency tracking
    DependencyGraph,
    // Recovery
    RecoveryEngine, RecoveryAction,
    // Store
    ResilientStore, ReadResult, ResilientStatus,
    // Errors
    BufferError,
};

// Concurrency exports (MVCC, memory pool, parallel execution)
pub use concurrency::{
    // Memory pool
    MemoryPoolConfig, MemoryPool, MemoryGuard, MemoryPoolStats, MemoryError,
    // MVCC
    MvccSlot, ReadHandle, WriteIntent, WriteResult, MvccStore,
    // Parallel execution
    ParallelConfig, ParallelExecutor, ResultHandle, ParallelError,
    // Query context
    QueryContext as ConcurrentQueryContext, ConflictError,
    // Combined store
    ConcurrentStore, WriteConflict, ConcurrentStats,
};

// Snapshot exports (VMware-like XOR delta snapshots, cold storage)
pub use snapshots::{
    // XOR deltas
    DeltaBlock, Snapshot, SnapshotChain, SnapshotStats,
    SnapshotId, SnapshotError,
    // Tiered storage
    StorageTier, ColdStorageConfig, TieredEntry,
    TieredStorage, TieredStorageStats, MigrationStats,
};

// Corpus exports (scent-indexed Arrow FS, Gutenberg chunking)
pub use corpus::{
    // Fingerprinting
    Fingerprint, generate_fingerprint,
    hamming_distance as corpus_hamming_distance,
    fingerprint_similarity,
    // Chunking
    ChunkStrategy, Chunk, chunk_document,
    // Documents
    DocumentMeta, Document,
    // Store
    CorpusConfig, CorpusStore, CorpusStats,
    // Gutenberg
    parse_gutenberg, load_gutenberg_book,
    // Errors
    CorpusError,
};

// Service exports (container lifecycle, DuckDB-inspired buffering)
pub use service::{
    // CPU detection
    CpuFeatures,
    // Buffer pool (DuckDB-style)
    BufferPoolConfig, BufferPool, BufferPoolStats,
    // Schema hydration
    AddressSchema, ZoneDescriptor,
    // Recovery
    RecoveryManifest,
    // Vectorized batching
    DataBatch, batch_hamming_distance,
    // Column operations (Lance-style)
    ColumnBatch, ColumnType,
    // Service container
    ServiceConfig, CognitiveService, ServiceHealth,
    // Compile hints
    AVX512_RUSTFLAGS, DOCKER_BUILD_CMD,
    // Errors
    ServiceError,
};

// Substrate exports (unified interface DTO)
pub use substrate::{
    // Configuration
    SubstrateConfig,
    // Node and edge types
    SubstrateNode, SubstrateEdge,
    // Write operations
    WriteOp, WriteBuffer as SubstrateWriteBuffer,
    // Statistics
    SubstrateStats,
    // Main interface
    Substrate,
};

// Lance zero-copy exports (unified address space)
pub use lance_zero_copy::{
    // Zero-copy view into Lance
    LanceView,
    // Temperature tracking for bubbling
    Temperature, ScentAwareness,
    // Bubbling operations
    BubbleResult, ZeroCopyBubbler,
};

// XOR DAG exports (ACID transactions, parity protection, work stealing)
pub use xor_dag::{
    // Zero-copy XOR operations
    xor_slices, xor_slices_inplace, xor_fingerprints, xor_fingerprints_multi,
    // Parity protection
    ParityBlock, ParityTier,
    // Epoch-based work stealing
    EpochGuard, EpochTicket, WorkItem, WorkOperation,
    // Conflict resolution (aliased to avoid conflict with concurrency module)
    ConflictStrategy, StateSnapshot, WriteConflict as DagWriteConflict,
    // ACID transactions (aliased to avoid conflict with temporal module)
    TxnState as DagTxnState, DagTransaction, WriteIntent as DagWriteIntent,
    // XOR DAG store
    XorDagConfig, XorDag, XorDelta, XorDagStatsSnapshot,
    // Errors
    DagError,
};

// Redis Adapter exports (Redis syntax interface)
pub use redis_adapter::{
    // Result types (aliased to avoid conflict with cog_redis::RedisResult)
    RedisResult as AdapterRedisResult, NodeResult, SearchHit, EdgeResult, CamResult,
    // Command types
    RedisCommand, SetOptions as RedisSetOptions, DeleteMode,
    // Main adapter
    RedisAdapter,
};

// Unified Engine exports (all features through BindSpace)
pub use unified_engine::{
    // Configuration
    UnifiedConfig,
    // Main engine
    UnifiedEngine,
    // Statistics
    UnifiedStatsSnapshot,
    // Errors
    UnifiedError,
};

// Fingerprint Dictionary exports (codebook for sparse hydration)
pub use fingerprint_dict::FingerprintDict;
