use super::ast::*;
use super::super::error::{QueryError, Result};
use std::collections::HashMap;

/// Query parameter value — no JSON, no Python, no serialization.
/// Comes from MCP tool input or BindSpace property lookup.
///
/// If a parameter isn't set, it's not in the HashMap. No Null variant.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    List(Vec<ParamValue>),
}

/// Substitute parameters with literal values in the AST
pub fn substitute_parameters(
    query: &mut CypherQuery,
    parameters: &HashMap<String, ParamValue>,
) -> Result<()> {
    // Substitute in READING clauses
    for reading_clause in &mut query.reading_clauses {
        substitute_in_reading_clause(reading_clause, parameters)?;
    }

    // Substitute in WHERE clause
    if let Some(where_clause) = &mut query.where_clause {
        substitute_in_where_clause(where_clause, parameters)?;
    }

    // Substitute in WITH clause
    if let Some(with_clause) = &mut query.with_clause {
        substitute_in_with_clause(with_clause, parameters)?;
    }

    // Substitute in post-WITH READING clauses
    for reading_clause in &mut query.post_with_reading_clauses {
        substitute_in_reading_clause(reading_clause, parameters)?;
    }

    // Substitute in post-WITH WHERE clause
    if let Some(post_where) = &mut query.post_with_where_clause {
        substitute_in_where_clause(post_where, parameters)?;
    }

    // Substitute in RETURN clause
    substitute_in_return_clause(&mut query.return_clause, parameters)?;

    // Substitute in ORDER BY clause
    if let Some(order_by) = &mut query.order_by {
        substitute_in_order_by_clause(order_by, parameters)?;
    }

    Ok(())
}

fn substitute_in_reading_clause(
    clause: &mut ReadingClause,
    parameters: &HashMap<String, ParamValue>,
) -> Result<()> {
    match clause {
        ReadingClause::Match(match_clause) => {
            for pattern in &mut match_clause.patterns {
                substitute_in_graph_pattern(pattern, parameters)?;
            }
        }
        ReadingClause::Unwind(unwind_clause) => {
            substitute_in_value_expression(&mut unwind_clause.expression, parameters)?;
        }
    }
    Ok(())
}

fn substitute_in_graph_pattern(
    pattern: &mut GraphPattern,
    parameters: &HashMap<String, ParamValue>,
) -> Result<()> {
    match pattern {
        GraphPattern::Node(node) => {
            for value in node.properties.values_mut() {
                substitute_in_property_value(value, parameters)?;
            }
        }
        GraphPattern::Path(path) => {
            substitute_in_node_pattern(&mut path.start_node, parameters)?;
            for segment in &mut path.segments {
                substitute_in_relationship_pattern(&mut segment.relationship, parameters)?;
                substitute_in_node_pattern(&mut segment.end_node, parameters)?;
            }
        }
    }
    Ok(())
}

fn substitute_in_node_pattern(
    node: &mut NodePattern,
    parameters: &HashMap<String, ParamValue>,
) -> Result<()> {
    for value in node.properties.values_mut() {
        substitute_in_property_value(value, parameters)?;
    }
    Ok(())
}

fn substitute_in_relationship_pattern(
    rel: &mut RelationshipPattern,
    parameters: &HashMap<String, ParamValue>,
) -> Result<()> {
    for value in rel.properties.values_mut() {
        substitute_in_property_value(value, parameters)?;
    }
    Ok(())
}

fn substitute_in_property_value(
    value: &mut PropertyValue,
    parameters: &HashMap<String, ParamValue>,
) -> Result<()> {
    if let PropertyValue::Parameter(name) = value {
        let param_value =
            parameters
                .get(&name.to_lowercase())
                .ok_or_else(|| QueryError::PlanError {
                    message: format!("Missing parameter: ${}", name),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;

        *value = param_to_property_value(param_value)?;
    }
    Ok(())
}

fn substitute_in_where_clause(
    where_clause: &mut WhereClause,
    parameters: &HashMap<String, ParamValue>,
) -> Result<()> {
    substitute_in_boolean_expression(&mut where_clause.expression, parameters)
}

fn substitute_in_with_clause(
    with_clause: &mut WithClause,
    parameters: &HashMap<String, ParamValue>,
) -> Result<()> {
    for item in &mut with_clause.items {
        substitute_in_value_expression(&mut item.expression, parameters)?;
    }
    if let Some(order_by) = &mut with_clause.order_by {
        substitute_in_order_by_clause(order_by, parameters)?;
    }
    Ok(())
}

fn substitute_in_return_clause(
    return_clause: &mut ReturnClause,
    parameters: &HashMap<String, ParamValue>,
) -> Result<()> {
    for item in &mut return_clause.items {
        substitute_in_value_expression(&mut item.expression, parameters)?;
    }
    Ok(())
}

fn substitute_in_order_by_clause(
    order_by: &mut OrderByClause,
    parameters: &HashMap<String, ParamValue>,
) -> Result<()> {
    for item in &mut order_by.items {
        substitute_in_value_expression(&mut item.expression, parameters)?;
    }
    Ok(())
}

fn substitute_in_boolean_expression(
    expr: &mut BooleanExpression,
    parameters: &HashMap<String, ParamValue>,
) -> Result<()> {
    match expr {
        BooleanExpression::Comparison { left, right, .. } => {
            substitute_in_value_expression(left, parameters)?;
            substitute_in_value_expression(right, parameters)?;
        }
        BooleanExpression::And(left, right) | BooleanExpression::Or(left, right) => {
            substitute_in_boolean_expression(left, parameters)?;
            substitute_in_boolean_expression(right, parameters)?;
        }
        BooleanExpression::Not(inner) => {
            substitute_in_boolean_expression(inner, parameters)?;
        }
        BooleanExpression::Exists(_) => {}
        BooleanExpression::In { expression, list } => {
            substitute_in_value_expression(expression, parameters)?;
            for item in list {
                substitute_in_value_expression(item, parameters)?;
            }
        }
        BooleanExpression::Like { expression, .. }
        | BooleanExpression::ILike { expression, .. }
        | BooleanExpression::Contains { expression, .. }
        | BooleanExpression::StartsWith { expression, .. }
        | BooleanExpression::EndsWith { expression, .. }
        | BooleanExpression::IsNull(expression)
        | BooleanExpression::IsNotNull(expression) => {
            substitute_in_value_expression(expression, parameters)?;
        }
    }
    Ok(())
}

fn substitute_in_value_expression(
    expr: &mut ValueExpression,
    parameters: &HashMap<String, ParamValue>,
) -> Result<()> {
    match expr {
        ValueExpression::Parameter(name) => {
            let param_value =
                parameters
                    .get(&name.to_lowercase())
                    .ok_or_else(|| QueryError::PlanError {
                        message: format!("Missing parameter: ${}", name),
                        location: snafu::Location::new(file!(), line!(), column!()),
                    })?;

            // Check for float list → VectorLiteral conversion
            if let ParamValue::List(items) = param_value {
                let mut floats = Vec::new();
                for v in items {
                    match v {
                        ParamValue::Float(f) => floats.push(*f as f32),
                        ParamValue::Int(i) => floats.push(*i as f32),
                        _ => {
                            return Err(QueryError::PlanError {
                                message: format!(
                                    "Parameter ${} is a list but contains non-numeric values. Only float vectors are supported as list parameters currently.",
                                    name
                                ),
                                location: snafu::Location::new(file!(), line!(), column!()),
                            });
                        }
                    }
                }
                *expr = ValueExpression::VectorLiteral(floats);
                return Ok(());
            }

            // Scalar conversion
            let prop_val = param_to_property_value(param_value)?;
            *expr = ValueExpression::Literal(prop_val);
        }
        ValueExpression::ScalarFunction { args, .. }
        | ValueExpression::AggregateFunction { args, .. } => {
            for arg in args {
                substitute_in_value_expression(arg, parameters)?;
            }
        }
        ValueExpression::Arithmetic { left, right, .. } => {
            substitute_in_value_expression(left, parameters)?;
            substitute_in_value_expression(right, parameters)?;
        }
        ValueExpression::VectorDistance { left, right, .. }
        | ValueExpression::VectorSimilarity { left, right, .. } => {
            substitute_in_value_expression(left, parameters)?;
            substitute_in_value_expression(right, parameters)?;
        }
        _ => {}
    }
    Ok(())
}

fn param_to_property_value(value: &ParamValue) -> Result<PropertyValue> {
    match value {
        ParamValue::Bool(b) => Ok(PropertyValue::Boolean(*b)),
        ParamValue::Int(i) => Ok(PropertyValue::Integer(*i)),
        ParamValue::Float(f) => Ok(PropertyValue::Float(*f)),
        ParamValue::String(s) => Ok(PropertyValue::String(s.clone())),
        ParamValue::List(_) => {
            Err(QueryError::PlanError {
                message: "List parameters are only supported as float vectors in value expressions.".to_string(),
                location: snafu::Location::new(file!(), line!(), column!()),
            })
        }
    }
}
