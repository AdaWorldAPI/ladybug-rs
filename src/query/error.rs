//! Error types for the ladybug query engine.
//!
//! Modeled on lance-graph's error.rs — snafu + #[track_caller] helpers.

use snafu::{prelude::*, Location};

pub type Result<T> = std::result::Result<T, QueryError>;

/// Errors that can occur during query processing.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum QueryError {
    /// Error parsing Cypher/SQL query syntax
    #[snafu(display("Parse error at position {position}: {message}"))]
    ParseError {
        message: String,
        position: usize,
        location: Location,
    },

    /// Error during query planning
    #[snafu(display("Query planning error: {message}"))]
    PlanError { message: String, location: Location },

    /// Error during query execution
    #[snafu(display("Query execution error: {message}"))]
    ExecutionError { message: String, location: Location },

    /// Error during Cypher → SQL transpilation
    #[snafu(display("Transpile error: {message}"))]
    TranspileError { message: String, location: Location },

    /// Unsupported query feature
    #[snafu(display("Unsupported feature: {feature}"))]
    UnsupportedFeature { feature: String, location: Location },

    /// Invalid graph pattern in query
    #[snafu(display("Invalid pattern: {message}"))]
    InvalidPattern { message: String, location: Location },

    /// Arrow error
    #[snafu(display("Arrow error: {source}"))]
    Arrow {
        source: arrow::error::ArrowError,
        location: Location,
    },

    /// DataFusion error
    #[snafu(display("DataFusion error: {source}"))]
    DataFusion {
        source: datafusion::error::DataFusionError,
        location: Location,
    },

    /// SPO graph store error
    #[snafu(display("SPO error: {message}"))]
    SpoError { message: String, location: Location },
}

// ============================================================================
// #[track_caller] helper functions
// ============================================================================

#[track_caller]
pub fn parse_err_at(message: impl Into<String>, position: usize) -> QueryError {
    let loc = std::panic::Location::caller();
    QueryError::ParseError {
        message: message.into(),
        position,
        location: Location::new(loc.file(), loc.line(), loc.column()),
    }
}

#[track_caller]
pub fn plan_err_at(message: impl Into<String>) -> QueryError {
    let loc = std::panic::Location::caller();
    QueryError::PlanError {
        message: message.into(),
        location: Location::new(loc.file(), loc.line(), loc.column()),
    }
}

#[track_caller]
pub fn exec_err_at(message: impl Into<String>) -> QueryError {
    let loc = std::panic::Location::caller();
    QueryError::ExecutionError {
        message: message.into(),
        location: Location::new(loc.file(), loc.line(), loc.column()),
    }
}

#[track_caller]
pub fn transpile_err_at(message: impl Into<String>) -> QueryError {
    let loc = std::panic::Location::caller();
    QueryError::TranspileError {
        message: message.into(),
        location: Location::new(loc.file(), loc.line(), loc.column()),
    }
}

// ============================================================================
// Convenience macros
// ============================================================================

#[macro_export]
macro_rules! parse_err {
    ($pos:expr, $($arg:tt)*) => {
        $crate::query::error::parse_err_at(format!($($arg)*), $pos)
    };
}

#[macro_export]
macro_rules! plan_err {
    ($($arg:tt)*) => {
        $crate::query::error::plan_err_at(format!($($arg)*))
    };
}

#[macro_export]
macro_rules! exec_err {
    ($($arg:tt)*) => {
        $crate::query::error::exec_err_at(format!($($arg)*))
    };
}

#[macro_export]
macro_rules! transpile_err {
    ($($arg:tt)*) => {
        $crate::query::error::transpile_err_at(format!($($arg)*))
    };
}

// ============================================================================
// From impls
// ============================================================================

impl From<arrow::error::ArrowError> for QueryError {
    fn from(source: arrow::error::ArrowError) -> Self {
        Self::Arrow {
            source,
            location: Location::new(file!(), line!(), column!()),
        }
    }
}

impl From<datafusion::error::DataFusionError> for QueryError {
    fn from(source: datafusion::error::DataFusionError) -> Self {
        Self::DataFusion {
            source,
            location: Location::new(file!(), line!(), column!()),
        }
    }
}

impl From<crate::graph::spo::sparse::SpoError> for QueryError {
    fn from(source: crate::graph::spo::sparse::SpoError) -> Self {
        Self::SpoError {
            message: source.to_string(),
            location: Location::new(file!(), line!(), column!()),
        }
    }
}

impl From<crate::query::lance_parser::error::GraphError> for QueryError {
    fn from(source: crate::query::lance_parser::error::GraphError) -> Self {
        use crate::query::lance_parser::error::GraphError;
        match source {
            GraphError::ParseError { message, position, location } => Self::ParseError {
                message,
                position,
                location,
            },
            GraphError::PlanError { message, location } => Self::PlanError {
                message,
                location,
            },
            GraphError::ExecutionError { message, location } => Self::ExecutionError {
                message,
                location,
            },
            GraphError::ConfigError { message, location } => Self::PlanError {
                message,
                location,
            },
            GraphError::UnsupportedFeature { feature, location } => Self::UnsupportedFeature {
                feature,
                location,
            },
            GraphError::InvalidPattern { message, location } => Self::InvalidPattern {
                message,
                location,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_err_at_captures_location() {
        let err = parse_err_at("bad token", 42);
        match &err {
            QueryError::ParseError { message, position, location } => {
                assert_eq!(message, "bad token");
                assert_eq!(*position, 42);
                assert!(location.file.ends_with("error.rs"));
            }
            other => panic!("expected ParseError, got {:?}", other),
        }
        let s = err.to_string();
        assert!(s.contains("bad token"), "display: {s}");
        assert!(s.contains("42"), "display: {s}");
    }

    #[test]
    fn test_plan_err_at_captures_location() {
        let err = plan_err_at("missing index");
        match &err {
            QueryError::PlanError { message, location } => {
                assert_eq!(message, "missing index");
                assert!(location.file.ends_with("error.rs"));
            }
            other => panic!("expected PlanError, got {:?}", other),
        }
    }

    #[test]
    fn test_exec_err_at_captures_location() {
        let err = exec_err_at("timeout");
        match &err {
            QueryError::ExecutionError { message, .. } => {
                assert_eq!(message, "timeout");
            }
            other => panic!("expected ExecutionError, got {:?}", other),
        }
    }

    #[test]
    fn test_transpile_err_at_captures_location() {
        let err = transpile_err_at("unsupported clause");
        match &err {
            QueryError::TranspileError { message, .. } => {
                assert_eq!(message, "unsupported clause");
            }
            other => panic!("expected TranspileError, got {:?}", other),
        }
    }

    #[test]
    fn test_from_arrow_error() {
        let arrow_err = arrow::error::ArrowError::ParseError("bad arrow".into());
        let qe: QueryError = arrow_err.into();
        assert!(matches!(qe, QueryError::Arrow { .. }));
    }

    #[test]
    fn test_from_datafusion_error() {
        let df_err = datafusion::error::DataFusionError::Plan("bad plan".into());
        let qe: QueryError = df_err.into();
        assert!(matches!(qe, QueryError::DataFusion { .. }));
    }
}
