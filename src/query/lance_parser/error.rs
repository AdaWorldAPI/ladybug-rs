// Stolen from lance-graph/crates/lance-graph/src/error.rs
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Error types for the stolen Cypher parser.
//!
//! Stripped of DataFusion/LanceCore/Arrow variants — those are mesh-side.
//! Uses snafu with #[track_caller] for zero-cost compile-time location capture.
//! Zero mesh coupling. The bouncer only knows about parse/plan/config errors.
//!
//! ## Error construction macros
//!
//! Use these macros instead of manual `Location::new(file!(), line!(), column!())`:
//!
//! ```ignore
//! // Before (verbose, 200+ chars per error):
//! GraphError::PlanError {
//!     message: format!("..."),
//!     location: snafu::Location::new(file!(), line!(), column!()),
//! }
//!
//! // After (zero-cost #[track_caller] via std::panic::Location):
//! lance_plan_err!("Failed to plan: {}", reason)
//! lance_config_err!("Invalid config: {}", detail)
//! lance_exec_err!("Execution failed: {}", detail)
//! ```

use snafu::{prelude::*, Location};

/// Result type for lance_parser operations
pub type Result<T> = std::result::Result<T, GraphError>;

// =============================================================================
// Zero-cost error construction via #[track_caller]
// =============================================================================

/// Create a PlanError with zero-cost caller location capture.
///
/// `#[track_caller]` makes the compiler insert the call-site location
/// at compile time — 0 runtime cycles. Uses `std::panic::Location`
/// internally, bridged to `snafu::Location` for compatibility
/// with the existing error enum.
#[track_caller]
pub fn plan_err_at(message: String) -> GraphError {
    let loc = std::panic::Location::caller();
    GraphError::PlanError {
        message,
        location: Location::new(loc.file(), loc.line(), loc.column()),
    }
}

/// Create a ConfigError with zero-cost caller location capture.
#[track_caller]
pub fn config_err_at(message: String) -> GraphError {
    let loc = std::panic::Location::caller();
    GraphError::ConfigError {
        message,
        location: Location::new(loc.file(), loc.line(), loc.column()),
    }
}

/// Create an ExecutionError with zero-cost caller location capture.
#[track_caller]
pub fn exec_err_at(message: String) -> GraphError {
    let loc = std::panic::Location::caller();
    GraphError::ExecutionError {
        message,
        location: Location::new(loc.file(), loc.line(), loc.column()),
    }
}

/// Create a PlanError with zero-cost location capture.
///
/// Uses `#[track_caller]` via `plan_err_at()` — the compiler inserts the
/// call-site file/line/column at 0 runtime cycles. No `file!()` / `line!()`
/// macros needed.
///
/// # Example
/// ```ignore
/// let err = lance_plan_err!("Cannot join {} to {}", left, right);
/// ```
#[macro_export]
macro_rules! lance_plan_err {
    ($($arg:tt)*) => {
        $crate::query::lance_parser::error::plan_err_at(format!($($arg)*))
    };
}

/// Create a ConfigError with zero-cost location capture.
#[macro_export]
macro_rules! lance_config_err {
    ($($arg:tt)*) => {
        $crate::query::lance_parser::error::config_err_at(format!($($arg)*))
    };
}

/// Create an ExecutionError with zero-cost location capture.
#[macro_export]
macro_rules! lance_exec_err {
    ($($arg:tt)*) => {
        $crate::query::lance_parser::error::exec_err_at(format!($($arg)*))
    };
}

/// Errors that can occur during graph query parsing and validation.
///
/// This is the star-chart-side error type. It knows about syntax, not geometry.
/// Stripped of DataFusion/LanceCore/Arrow variants — those are mesh-side.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum GraphError {
    /// Error parsing Cypher query syntax
    #[snafu(display("Cypher parse error at position {position}: {message}"))]
    ParseError {
        message: String,
        position: usize,
        location: Location,
    },

    /// Error with graph configuration
    #[snafu(display("Graph configuration error: {message}"))]
    ConfigError { message: String, location: Location },

    /// Error during query planning
    #[snafu(display("Query planning error: {message}"))]
    PlanError { message: String, location: Location },

    /// Error during query execution
    #[snafu(display("Query execution error: {message}"))]
    ExecutionError { message: String, location: Location },

    /// Unsupported Cypher feature
    #[snafu(display("Unsupported Cypher feature: {feature}"))]
    UnsupportedFeature { feature: String, location: Location },

    /// Invalid graph pattern
    #[snafu(display("Invalid graph pattern: {message}"))]
    InvalidPattern { message: String, location: Location },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_err_carries_location() {
        let err = lance_plan_err!("cannot plan join: {} to {}", "left", "right");
        match err {
            GraphError::PlanError { message, location } => {
                assert_eq!(message, "cannot plan join: left to right");
                // #[track_caller] captures the call site — this file.
                assert!(
                    location.file.contains("error.rs"),
                    "location should point to this file, got: {}",
                    location.file
                );
                assert!(location.line > 0, "line should be non-zero");
            }
            other => panic!("expected PlanError, got: {:?}", other),
        }
    }

    #[test]
    fn test_config_err_carries_location() {
        let err = lance_config_err!("invalid config: {}", "missing field");
        match err {
            GraphError::ConfigError { message, location } => {
                assert_eq!(message, "invalid config: missing field");
                assert!(location.file.contains("error.rs"));
            }
            other => panic!("expected ConfigError, got: {:?}", other),
        }
    }

    #[test]
    fn test_exec_err_carries_location() {
        let err = lance_exec_err!("execution failed at step {}", 3);
        match err {
            GraphError::ExecutionError { message, location } => {
                assert_eq!(message, "execution failed at step 3");
                assert!(location.file.contains("error.rs"));
            }
            other => panic!("expected ExecutionError, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_error_display() {
        let err = GraphError::ParseError {
            message: "unexpected token".to_string(),
            position: 42,
            location: Location::new(file!(), line!(), column!()),
        };
        let display = format!("{}", err);
        assert!(display.contains("unexpected token"));
        assert!(display.contains("42"));
    }
}
