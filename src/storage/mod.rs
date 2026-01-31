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
