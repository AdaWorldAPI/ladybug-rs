//! Query layer - SQL, Cypher, and execution
//!
//! Provides unified query interface:
//! - SQL via DataFusion
//! - Cypher via transpilation to recursive CTEs
//! - Custom UDFs for Hamming/similarity operations

mod builder;
mod cypher;
mod datafusion;

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

#[derive(thiserror::Error, Debug)]
pub enum QueryError {
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Execution error: {0}")]
    Execution(String),
    #[error("Transpile error: {0}")]
    Transpile(String),
}
