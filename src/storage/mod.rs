//! Storage module - Persistence layers
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                              16-bit ADDRESS SPACE                           │
//! ├─────────────────┬───────────────────────────────────────────────────────────┤
//! │  0x0000-0x0FFF  │  SURFACE (4,096) - CAM operations, verbs                  │
//! ├─────────────────┼───────────────────────────────────────────────────────────┤
//! │  0x1000-0x7FFF  │  FLUID (28,672) - Working memory, TTL, promotion/demotion │
//! ├─────────────────┼───────────────────────────────────────────────────────────┤
//! │  0x8000-0xFFFF  │  NODES (32,768) - Persistent graph, 256-way tree          │
//! └─────────────────┴───────────────────────────────────────────────────────────┘
//! ```
//!
//! # Layers
//!
//! - **CogRedis**: Cognitive Redis - Redis syntax, cognitive semantics
//!   - GET/SET with qualia, truth values, tier management
//!   - BIND/UNBIND for graph edges (ABBA retrieval)
//!   - RESONATE/INTUIT for qualia-weighted search
//!   - CAUSE/WOULD for causal reasoning (Pearl's 3 rungs)
//!   - DEDUCE/ABDUCT for NARS inference
//!
//! - **LanceDB**: Content storage (fingerprints, vectors)
//! - **Kuzu**: Optional graph structure (can use CogRedis instead)
//! - **Database**: Unified query interface

pub mod lance;
pub mod database;
pub mod kuzu;
pub mod cog_redis;

pub use lance::LanceStore;
pub use database::Database;
pub use kuzu::{KuzuStore, NodeRecord, EdgeRecord, PathRecord};
pub use cog_redis::{
    // Address space
    CogAddr, Tier,
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
