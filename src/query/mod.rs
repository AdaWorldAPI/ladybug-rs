//! Query Layer
//!
//! Unified query interface supporting:
//! - SQL via DataFusion
//! - Cypher via transpilation to SQL
//! - Custom UDFs for Hamming, NARS, VSA operations

mod builder;
mod sql;
mod cypher;

pub use builder::{Query, QueryResult as SimpleResult};
pub use sql::{SqlExecutor, QueryResult};
pub use cypher::CypherTranspiler;

#[derive(thiserror::Error, Debug)]
pub enum QueryError {
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Execution error: {0}")]
    Execution(String),
    #[error("Transpilation error: {0}")]
    Transpile(String),
}
