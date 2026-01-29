//! Query types

mod builder;

pub use builder::{Query, QueryResult};

#[derive(thiserror::Error, Debug)]
pub enum QueryError {
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Execution error: {0}")]
    Execution(String),
}
