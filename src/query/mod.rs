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
pub mod fingerprint_table;
pub mod hybrid;
pub mod scent_scan;
pub mod graph_provider;
pub mod dn_tree_provider;

pub use builder::{Query, QueryResult};
pub use cypher::{
    CypherParser,
    CypherTranspiler,
    CypherQuery,
    cypher_to_sql,
};
pub use datafusion::{
    SqlEngine,
    QueryBuilder,
};
pub use hybrid::{
    HybridQuery,
    HybridResult,
    HybridEngine,
    HybridStats,
    CausalMode,
    TemporalConstraint,
    VectorConstraint,
    QualiaFilter,
    TruthFilter,
    parse_hybrid,
    execute_hybrid_command,
};
pub use cognitive_udfs::{
    register_cognitive_udfs,
    all_cognitive_udfs,
    HammingUdf,
    SimilarityUdf,
    PopcountUdf,
    XorBindUdf,
    ExtractScentUdf,
    ScentDistanceUdf,
    NarsDeductionUdf,
    NarsInductionUdf,
    NarsAbductionUdf,
    NarsRevisionUdf,
    MembraneEncodeUdf,
    MembraneDecodeUdf,
};
pub use fingerprint_table::{
    FingerprintTableProvider,
    BindSpaceScan,
    BindSpaceExt,
};
pub use scent_scan::{
    ScentScanExec,
    ScentPredicate,
    HammingDistanceUdf,
    SimilarityUdf as ScentSimilarityUdf,
    ScentUdfExtension,
};
pub use graph_provider::{
    EdgeTableProvider,
    GraphTraversalExec,
    TraversalConfig,
    TraversalDirection,
    GraphExt,
};
pub use dn_tree_provider::{
    DnTreeTableProvider,
    DnTreeExt,
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
