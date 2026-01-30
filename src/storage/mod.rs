//! Storage module - Persistence layers
//!
//! # 8-bit Prefix Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                      PREFIX (8-bit) : ADDRESS (8-bit)                       │
//! ├─────────────────┬───────────────────────────────────────────────────────────┤
//! │  0x00:XX        │  SURFACE 0 - Lance/Kuzu (256 ops)                         │
//! │  0x01:XX        │  SURFACE 1 - SQL/Neo4j (256 ops)                          │
//! │  0x02:XX        │  SURFACE 2 - Meta/NARS (256 ops)                          │
//! │  0x03:XX        │  SURFACE 3 - Verbs/Cypher (256 verbs)                     │
//! ├─────────────────┼───────────────────────────────────────────────────────────┤
//! │  0x04-0x7F:XX   │  FLUID (124 × 256 = 31,744 edges)                         │
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
//! - **Kuzu**: Optional native graph (Cypher syntax)
//! - **Database**: Unified query interface

pub mod lance;
pub mod database;
pub mod kuzu;
pub mod cog_redis;
pub mod bind_space;

pub use lance::LanceStore;
pub use database::Database;
pub use kuzu::{KuzuStore, NodeRecord, EdgeRecord, PathRecord};

// CogRedis exports
pub use cog_redis::{
    // Address types
    CogAddr, Tier, SurfaceCompartment,
    
    // Prefix constants
    PREFIX_LANCE, PREFIX_SQL, PREFIX_META, PREFIX_VERBS,
    PREFIX_FLUID_START, PREFIX_FLUID_END,
    PREFIX_NODE_START, PREFIX_NODE_END,
    CHUNK_SIZE,
    
    // Legacy constants (compatibility)
    SURFACE_START, SURFACE_END, SURFACE_SIZE,
    FLUID_START, FLUID_END, FLUID_SIZE,
    NODE_START, NODE_END, NODE_SIZE,
    TOTAL_SIZE,
    
    // Values and edges
    CogValue, CogEdge,
    
    // Main interface
    CogRedis, CogRedisStats,
    
    // Results
    GetResult, SetOptions, ResonateResult, DeduceResult,
};

// BindSpace exports (universal DTO)
pub use bind_space::{
    Addr, BindNode, BindEdge, BindSpace, BindSpaceStats,
    ChunkContext, QueryAdapter, QueryResult, QueryValue,
    hamming_distance, FINGERPRINT_WORDS,
};
