// Stolen from lance-graph — hardened Cypher parser (the bouncer).
//
// This module sits OUTSIDE the door. It validates syntax, rejects garbage,
// and hands off clean resolved ASTs. It does NOT know about BindSpace,
// Containers, fingerprints, SPO, or qualia. It checks IDs at the door.
//
// REGIME BOUNDARY: No imports from crate::spo, crate::storage, crate::container,
// crate::graph::spo, or any mesh-side module. If you find yourself adding one, STOP.

pub mod ast;
pub mod case_insensitive;
pub mod config;
pub mod error;
pub mod parameter_substitution;
pub mod parser;
pub mod semantic;

// Re-export the main entry points.
// NOTE: ast::CypherQuery deliberately excluded to avoid collision with
// cypher.rs::CypherQuery. Use lance_parser::ast::CypherQuery if needed.
pub use ast::{
    ArithmeticOperator, BooleanExpression, ComparisonOperator, DistanceMetric,
    FunctionType, GraphPattern, LengthRange, MatchClause, NodePattern, OrderByClause,
    OrderByItem, PathPattern, PathSegment, PropertyRef, PropertyValue, ReadingClause,
    RelationshipDirection, RelationshipPattern, ReturnClause, ReturnItem, SortDirection,
    UnwindClause, ValueExpression, WhereClause, WithClause, classify_function,
};
pub use error::{GraphError, Result};
pub use parameter_substitution::ParamValue;
pub use parser::parse_cypher_query;
