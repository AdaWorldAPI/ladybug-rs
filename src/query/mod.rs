//! Query layer - SQL, Cypher, and execution
//!
//! Provides unified query interface:
//! - SQL via DataFusion with cognitive UDFs
//! - Cypher via transpilation to recursive CTEs
//! - Custom UDFs for Hamming/similarity/NARS operations
//! - Scent Index integration for predicate pushdown
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    DATAFUSION AS CONSCIOUSNESS                   │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │   Query → Parser → LogicalPlan → PhysicalPlan → Execution       │
//! │                         ↓                                        │
//! │              ┌─────────────────────┐                            │
//! │              │  Cognitive UDFs     │                            │
//! │              │  - hamming()        │                            │
//! │              │  - similarity()     │                            │
//! │              │  - xor_bind()       │                            │
//! │              │  - nars_deduction() │                            │
//! │              │  - extract_scent()  │                            │
//! │              └─────────────────────┘                            │
//! │                         ↓                                        │
//! │              ┌─────────────────────┐                            │
//! │              │  Scent Index        │                            │
//! │              │  L1: 98.8% filter   │                            │
//! │              │  L2: 99.997% filter │                            │
//! │              └─────────────────────┘                            │
//! │                         ↓                                        │
//! │              ┌─────────────────────┐                            │
//! │              │  SIMD Hamming       │                            │
//! │              │  AVX-512: 2ns/cmp   │                            │
//! │              └─────────────────────┘                            │
//! │                                                                  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

mod builder;
pub mod cognitive_udfs;
mod cypher;
mod datafusion;
pub mod dn_tree_provider;
pub mod fingerprint_table;
pub mod graph_provider;
pub mod hybrid;
pub mod scent_scan;

pub use builder::{Query, QueryResult};
pub use cognitive_udfs::{
    ExtractScentUdf, HammingUdf, MembraneDecodeUdf, MembraneEncodeUdf, NarsAbductionUdf,
    NarsDeductionUdf, NarsInductionUdf, NarsRevisionUdf, PopcountUdf, ScentDistanceUdf,
    SimilarityUdf, XorBindUdf, all_cognitive_udfs, register_cognitive_udfs,
};
pub use cypher::{CypherParser, CypherQuery, CypherTranspiler, cypher_to_sql};
pub use datafusion::{QueryBuilder, SqlEngine};
pub use dn_tree_provider::{DnTreeExt, DnTreeTableProvider};
pub use fingerprint_table::{BindSpaceExt, BindSpaceScan, FingerprintTableProvider};
pub use graph_provider::{
    EdgeTableProvider, GraphExt, GraphTraversalExec, TraversalConfig, TraversalDirection,
};
pub use hybrid::{
    CausalMode, HybridEngine, HybridQuery, HybridResult, HybridStats, QualiaFilter,
    TemporalConstraint, TruthFilter, VectorConstraint, execute_hybrid_command, parse_hybrid,
};
pub use scent_scan::{
    HammingDistanceUdf, ScentPredicate, ScentScanExec, ScentUdfExtension,
    SimilarityUdf as ScentSimilarityUdf,
};

#[derive(thiserror::Error, Debug)]
pub enum QueryError {
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Execution error: {0}")]
    Execution(String),
    #[error("Transpile error: {0}")]
    Transpile(String),
}
