//! Cypher Bridge — lance_parser AST → BindSpace operations
//!
//! This module takes parsed Cypher ASTs (from lance_parser) and executes them
//! directly against BindSpace. No intermediate types. No bridges. No adapters.
//!
//! ```text
//! lance_parser::parse_cypher_query(cypher_str) → CypherQuery AST
//!     → execute_cypher(&mut BindSpace, &CypherQuery) → CypherResult
//! ```

use std::collections::HashMap;

use crate::query::lance_parser::ast::{
    self, BooleanExpression, ComparisonOperator, CypherQuery, GraphPattern, MatchClause,
    NodePattern, PropertyValue, ReadingClause, ReturnClause, ValueExpression,
};
use crate::storage::bind_space::{Addr, BindEdge, BindNode, BindSpace, FINGERPRINT_WORDS};

// =============================================================================
// QUERY RESULT
// =============================================================================

/// Result of a Cypher query execution against BindSpace.
#[derive(Debug, Clone)]
pub struct CypherResult {
    pub columns: Vec<String>,
    pub rows: Vec<HashMap<String, serde_json::Value>>,
    pub nodes_created: usize,
    pub relationships_created: usize,
    pub properties_set: usize,
}

impl CypherResult {
    pub fn empty() -> Self {
        Self {
            columns: Vec::new(),
            rows: Vec::new(),
            nodes_created: 0,
            relationships_created: 0,
            properties_set: 0,
        }
    }
}

// =============================================================================
// EXECUTE: CypherQuery AST → BindSpace mutations/reads
// =============================================================================

/// Execute a parsed Cypher query against a BindSpace.
///
/// Takes a lance_parser CypherQuery AST directly — no intermediate types.
pub fn execute_cypher(
    bs: &mut BindSpace,
    query: &CypherQuery,
) -> Result<CypherResult, String> {
    let mut result = CypherResult::empty();

    // Process reading clauses (MATCH, UNWIND)
    let mut matched_nodes: Vec<(Addr, &BindNode)> = Vec::new();
    let mut has_match = false;

    for clause in &query.reading_clauses {
        match clause {
            ReadingClause::Match(match_clause) => {
                has_match = true;
                execute_match(bs, match_clause, &query.where_clause, &mut matched_nodes)?;
            }
            ReadingClause::Unwind(_) => {
                // UNWIND not yet wired to BindSpace — skip
            }
        }
    }

    // If we had MATCH clauses, build RETURN results
    if has_match {
        // Apply LIMIT
        if let Some(limit) = query.limit {
            matched_nodes.truncate(limit as usize);
        }

        build_return_results(&matched_nodes, &query.return_clause, &mut result);
    }

    // Process update clauses by scanning reading_clauses for write patterns
    // In Cypher, CREATE/MERGE can appear as top-level statements.
    // Since lance_parser models them as reading clauses with specific patterns,
    // we detect write intent from the query structure.
    //
    // For now, we support direct MERGE/CREATE via a convention:
    // If there are no MATCH clauses and there's a single node pattern
    // with properties, treat it as a MERGE/CREATE.
    if !has_match && !query.reading_clauses.is_empty() {
        // Check for node-only patterns that indicate a write
        for clause in &query.reading_clauses {
            if let ReadingClause::Match(match_clause) = clause {
                for pattern in &match_clause.patterns {
                    match pattern {
                        GraphPattern::Node(node_pat) => {
                            execute_merge_node(bs, node_pat, &mut result)?;
                        }
                        GraphPattern::Path(path_pat) => {
                            // CREATE edge: start_node -[rel]-> end_node
                            execute_create_edge_from_path(bs, path_pat, &mut result)?;
                        }
                    }
                }
            }
        }
    }

    Ok(result)
}

/// Execute a MATCH clause — scan BindSpace, filter by labels and WHERE.
fn execute_match<'a>(
    bs: &'a BindSpace,
    match_clause: &MatchClause,
    where_clause: &Option<ast::WhereClause>,
    matched_nodes: &mut Vec<(Addr, &'a BindNode)>,
) -> Result<(), String> {
    // Extract label filters from match patterns
    let mut label_filters: Vec<String> = Vec::new();
    for pattern in &match_clause.patterns {
        match pattern {
            GraphPattern::Node(node) => {
                label_filters.extend(node.labels.clone());
            }
            GraphPattern::Path(path) => {
                label_filters.extend(path.start_node.labels.clone());
                for segment in &path.segments {
                    label_filters.extend(segment.end_node.labels.clone());
                }
            }
        }
    }

    // Scan all nodes, filter by label and WHERE
    for (addr, node) in bs.nodes_iter() {
        // Label filter: if we have label constraints, node must match at least one
        if !label_filters.is_empty() {
            match &node.label {
                Some(node_label) => {
                    if !label_filters.iter().any(|l| l == node_label) {
                        continue;
                    }
                }
                None => continue,
            }
        }

        // WHERE filter
        if let Some(wc) = where_clause {
            if !evaluate_where(node, &wc.expression) {
                continue;
            }
        }

        matched_nodes.push((addr, node));
    }

    Ok(())
}

/// Build RETURN results from matched nodes.
fn build_return_results(
    matched_nodes: &[(Addr, &BindNode)],
    return_clause: &ReturnClause,
    result: &mut CypherResult,
) {
    // Build column names from return items
    let columns: Vec<String> = if return_clause.items.is_empty()
        || (return_clause.items.len() == 1
            && matches!(
                &return_clause.items[0].expression,
                ValueExpression::Variable(_)
            )
            && return_clause.items[0].alias.is_none())
    {
        // RETURN n or RETURN * → return all properties
        vec![
            "addr".to_string(),
            "label".to_string(),
            "properties".to_string(),
        ]
    } else {
        return_clause
            .items
            .iter()
            .map(|item| {
                if let Some(ref alias) = item.alias {
                    alias.clone()
                } else {
                    match &item.expression {
                        ValueExpression::Property(prop_ref) => prop_ref.property.clone(),
                        ValueExpression::Variable(v) => v.clone(),
                        _ => "?".to_string(),
                    }
                }
            })
            .collect()
    };

    result.columns = columns.clone();

    for (addr, node) in matched_nodes {
        let props: HashMap<String, serde_json::Value> = node
            .payload
            .as_ref()
            .and_then(|p| serde_json::from_slice(p).ok())
            .unwrap_or_default();

        let mut row = HashMap::new();

        // Check if we're returning all properties or specific ones
        let is_wildcard = return_clause.items.is_empty()
            || (return_clause.items.len() == 1
                && matches!(
                    &return_clause.items[0].expression,
                    ValueExpression::Variable(_)
                )
                && return_clause.items[0].alias.is_none());

        if is_wildcard {
            row.insert(
                "addr".to_string(),
                serde_json::json!(format!("0x{:04X}", addr.0)),
            );
            row.insert(
                "label".to_string(),
                serde_json::json!(node.label.clone().unwrap_or_else(|| "?".to_string())),
            );
            row.insert(
                "properties".to_string(),
                serde_json::Value::Object(
                    props
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect(),
                ),
            );
        } else {
            for (i, col) in columns.iter().enumerate() {
                let return_item = return_clause.items.get(i);
                let val = match return_item.map(|ri| &ri.expression) {
                    Some(ValueExpression::Property(prop_ref)) => {
                        match prop_ref.property.as_str() {
                            "addr" => serde_json::json!(format!("0x{:04X}", addr.0)),
                            "label" => serde_json::json!(
                                node.label.clone().unwrap_or_else(|| "?".to_string())
                            ),
                            key => props.get(key).cloned().unwrap_or(serde_json::Value::Null),
                        }
                    }
                    _ => {
                        // Try as property key
                        props
                            .get(col.as_str())
                            .cloned()
                            .unwrap_or(serde_json::Value::Null)
                    }
                };
                row.insert(col.clone(), val);
            }
        }

        result.rows.push(row);
    }
}

/// Execute MERGE for a node pattern — upsert into BindSpace.
fn execute_merge_node(
    bs: &mut BindSpace,
    node_pat: &NodePattern,
    result: &mut CypherResult,
) -> Result<(), String> {
    let primary_label = node_pat
        .labels
        .first()
        .map(|s| s.as_str())
        .unwrap_or("Node");

    let name_prop = node_pat
        .properties
        .get("name")
        .or_else(|| node_pat.properties.get("noun_key"));

    // Check for existing node (MERGE semantics)
    if let Some(name_val) = name_prop {
        let name_str = property_value_to_string(name_val);
        let existing = find_node_by_label_and_name(bs, primary_label, &name_str);

        if let Some(addr) = existing {
            // Node exists — update properties
            if let Some(node) = bs.read_mut(addr) {
                let json_props = properties_to_json(&node_pat.properties);
                node.payload = Some(serde_json::to_vec(&json_props).unwrap_or_default());
                result.properties_set += node_pat.properties.len();
            }
            return Ok(());
        }
    }

    // Node doesn't exist — create it
    let fingerprint = node_pattern_to_fingerprint(primary_label, &node_pat.properties);
    let addr = bs.write_labeled(fingerprint, primary_label);

    // Store properties as JSON payload
    if let Some(node) = bs.read_mut(addr) {
        let json_props = properties_to_json(&node_pat.properties);
        node.payload = Some(serde_json::to_vec(&json_props).unwrap_or_default());
    }

    result.nodes_created += 1;
    result.properties_set += node_pat.properties.len();
    Ok(())
}

/// Execute CREATE edge from a PathPattern.
fn execute_create_edge_from_path(
    bs: &mut BindSpace,
    path: &ast::PathPattern,
    result: &mut CypherResult,
) -> Result<(), String> {
    // Ensure start node exists (MERGE)
    execute_merge_node(bs, &path.start_node, result)?;

    for segment in &path.segments {
        // Ensure end node exists
        execute_merge_node(bs, &segment.end_node, result)?;

        // Resolve start and end addresses
        let from_label = path
            .start_node
            .labels
            .first()
            .map(|s| s.as_str())
            .unwrap_or("Node");
        let from_name = path
            .start_node
            .properties
            .get("name")
            .map(|v| property_value_to_string(v))
            .unwrap_or_default();
        let from_addr = find_node_by_label_and_name(bs, from_label, &from_name)
            .ok_or_else(|| format!("Cannot resolve source node: {}:{}", from_label, from_name))?;

        let to_label = segment
            .end_node
            .labels
            .first()
            .map(|s| s.as_str())
            .unwrap_or("Node");
        let to_name = segment
            .end_node
            .properties
            .get("name")
            .map(|v| property_value_to_string(v))
            .unwrap_or_default();
        let to_addr = find_node_by_label_and_name(bs, to_label, &to_name)
            .ok_or_else(|| format!("Cannot resolve target node: {}:{}", to_label, to_name))?;

        // Create verb node for relationship type
        let rel_type = segment
            .relationship
            .types
            .first()
            .map(|s| s.as_str())
            .unwrap_or("RELATED_TO");
        let verb_fp = label_to_fingerprint(rel_type);
        let verb_addr = bs.write_labeled(verb_fp, rel_type);

        let edge = BindEdge::new(from_addr, verb_addr, to_addr);
        bs.link_with_edge(edge);

        result.relationships_created += 1;
    }

    Ok(())
}

// =============================================================================
// WHERE EVALUATION — works directly on P3 BooleanExpression
// =============================================================================

/// Evaluate a lance_parser BooleanExpression against a BindNode.
fn evaluate_where(node: &BindNode, expr: &BooleanExpression) -> bool {
    let props: HashMap<String, serde_json::Value> = node
        .payload
        .as_ref()
        .and_then(|p| serde_json::from_slice(p).ok())
        .unwrap_or_default();

    evaluate_bool_expr(&props, expr)
}

fn evaluate_bool_expr(
    props: &HashMap<String, serde_json::Value>,
    expr: &BooleanExpression,
) -> bool {
    match expr {
        BooleanExpression::Comparison {
            left,
            operator,
            right,
        } => {
            let left_val = resolve_value_expr(props, left);
            let right_val = resolve_value_expr(props, right);
            compare_json_values(&left_val, operator, &right_val)
        }
        BooleanExpression::And(left, right) => {
            evaluate_bool_expr(props, left) && evaluate_bool_expr(props, right)
        }
        BooleanExpression::Or(left, right) => {
            evaluate_bool_expr(props, left) || evaluate_bool_expr(props, right)
        }
        BooleanExpression::Not(inner) => !evaluate_bool_expr(props, inner),
        BooleanExpression::Exists(prop_ref) => {
            props
                .get(&prop_ref.property)
                .map(|v| !v.is_null())
                .unwrap_or(false)
        }
        BooleanExpression::IsNull(expr) => {
            let val = resolve_value_expr(props, expr);
            val.is_null()
        }
        BooleanExpression::IsNotNull(expr) => {
            let val = resolve_value_expr(props, expr);
            !val.is_null()
        }
        BooleanExpression::Contains {
            expression,
            substring,
        } => {
            let val = resolve_value_expr(props, expression);
            val.as_str()
                .map(|s| s.contains(substring.as_str()))
                .unwrap_or(false)
        }
        BooleanExpression::StartsWith { expression, prefix } => {
            let val = resolve_value_expr(props, expression);
            val.as_str()
                .map(|s| s.starts_with(prefix.as_str()))
                .unwrap_or(false)
        }
        BooleanExpression::EndsWith { expression, suffix } => {
            let val = resolve_value_expr(props, expression);
            val.as_str()
                .map(|s| s.ends_with(suffix.as_str()))
                .unwrap_or(false)
        }
        BooleanExpression::In { expression, list } => {
            let val = resolve_value_expr(props, expression);
            list.iter()
                .any(|item| resolve_value_expr(props, item) == val)
        }
        BooleanExpression::Like {
            expression,
            pattern,
        } => {
            let val = resolve_value_expr(props, expression);
            val.as_str()
                .map(|s| simple_like_match(s, pattern, true))
                .unwrap_or(false)
        }
        BooleanExpression::ILike {
            expression,
            pattern,
        } => {
            let val = resolve_value_expr(props, expression);
            val.as_str()
                .map(|s| simple_like_match(s, pattern, false))
                .unwrap_or(false)
        }
    }
}

/// Resolve a ValueExpression to a JSON value using node properties.
fn resolve_value_expr(
    props: &HashMap<String, serde_json::Value>,
    expr: &ValueExpression,
) -> serde_json::Value {
    match expr {
        ValueExpression::Literal(pv) => property_value_to_json(pv),
        ValueExpression::Property(prop_ref) => {
            props
                .get(&prop_ref.property)
                .cloned()
                .unwrap_or(serde_json::Value::Null)
        }
        ValueExpression::Variable(_) => {
            // Variable reference — return the whole props as object
            serde_json::Value::Object(
                props
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect(),
            )
        }
        _ => serde_json::Value::Null,
    }
}

/// Compare two JSON values using a ComparisonOperator.
fn compare_json_values(
    left: &serde_json::Value,
    op: &ComparisonOperator,
    right: &serde_json::Value,
) -> bool {
    match op {
        ComparisonOperator::Equal => left == right,
        ComparisonOperator::NotEqual => left != right,
        ComparisonOperator::LessThan => json_numeric_cmp(left, right).map_or(false, |c| c < 0),
        ComparisonOperator::LessThanOrEqual => {
            json_numeric_cmp(left, right).map_or(false, |c| c <= 0)
        }
        ComparisonOperator::GreaterThan => {
            json_numeric_cmp(left, right).map_or(false, |c| c > 0)
        }
        ComparisonOperator::GreaterThanOrEqual => {
            json_numeric_cmp(left, right).map_or(false, |c| c >= 0)
        }
    }
}

fn json_numeric_cmp(left: &serde_json::Value, right: &serde_json::Value) -> Option<i8> {
    let l = left.as_f64()?;
    let r = right.as_f64()?;
    if l < r {
        Some(-1)
    } else if l > r {
        Some(1)
    } else {
        Some(0)
    }
}

/// Simple LIKE pattern matching (% = any, _ = single char).
fn simple_like_match(s: &str, pattern: &str, case_sensitive: bool) -> bool {
    let (s, pattern) = if case_sensitive {
        (s.to_string(), pattern.to_string())
    } else {
        (s.to_lowercase(), pattern.to_lowercase())
    };

    // Convert SQL LIKE pattern to simple matching
    if pattern.starts_with('%') && pattern.ends_with('%') && pattern.len() > 2 {
        let inner = &pattern[1..pattern.len() - 1];
        s.contains(inner)
    } else if pattern.starts_with('%') {
        s.ends_with(&pattern[1..])
    } else if pattern.ends_with('%') {
        s.starts_with(&pattern[..pattern.len() - 1])
    } else {
        s == pattern
    }
}

// =============================================================================
// HELPERS
// =============================================================================

/// Convert a P3 PropertyValue to a JSON value.
fn property_value_to_json(pv: &PropertyValue) -> serde_json::Value {
    match pv {
        PropertyValue::String(s) => serde_json::Value::String(s.clone()),
        PropertyValue::Integer(i) => serde_json::json!(*i),
        PropertyValue::Float(f) => serde_json::json!(*f),
        PropertyValue::Boolean(b) => serde_json::Value::Bool(*b),
        PropertyValue::Null => serde_json::Value::Null,
        PropertyValue::Parameter(p) => serde_json::Value::String(format!("${}", p)),
        PropertyValue::Property(pr) => {
            serde_json::Value::String(format!("{}.{}", pr.variable, pr.property))
        }
    }
}

/// Convert a P3 PropertyValue to a display string.
fn property_value_to_string(pv: &PropertyValue) -> String {
    match pv {
        PropertyValue::String(s) => s.clone(),
        PropertyValue::Integer(i) => i.to_string(),
        PropertyValue::Float(f) => f.to_string(),
        PropertyValue::Boolean(b) => b.to_string(),
        PropertyValue::Null => "null".to_string(),
        PropertyValue::Parameter(p) => format!("${}", p),
        PropertyValue::Property(pr) => format!("{}.{}", pr.variable, pr.property),
    }
}

/// Convert P3 property map to JSON map.
fn properties_to_json(
    properties: &HashMap<String, PropertyValue>,
) -> HashMap<String, serde_json::Value> {
    properties
        .iter()
        .map(|(k, v)| (k.clone(), property_value_to_json(v)))
        .collect()
}

/// Generate a deterministic fingerprint from label + properties.
fn node_pattern_to_fingerprint(
    label: &str,
    properties: &HashMap<String, PropertyValue>,
) -> [u64; FINGERPRINT_WORDS] {
    let mut content = label.to_string();
    let mut sorted: Vec<_> = properties.iter().collect();
    sorted.sort_by_key(|(k, _)| *k);
    for (k, v) in sorted {
        content.push(':');
        content.push_str(k);
        content.push('=');
        content.push_str(&property_value_to_string(v));
    }
    let fp = crate::core::Fingerprint::from_content(&content);
    let mut words = [0u64; FINGERPRINT_WORDS];
    words.copy_from_slice(fp.as_raw());
    words
}

/// Generate a fingerprint from a label string.
fn label_to_fingerprint(label: &str) -> [u64; FINGERPRINT_WORDS] {
    let fp = crate::core::Fingerprint::from_content(label);
    let mut words = [0u64; FINGERPRINT_WORDS];
    words.copy_from_slice(fp.as_raw());
    words
}

/// Find a node by label + name property.
pub fn find_node_by_label_and_name(bs: &BindSpace, label: &str, name: &str) -> Option<Addr> {
    for (addr, node) in bs.nodes_iter() {
        if node.label.as_deref() != Some(label) {
            continue;
        }
        if let Some(ref payload) = node.payload {
            if let Ok(props) =
                serde_json::from_slice::<HashMap<String, serde_json::Value>>(payload)
            {
                let matches = props
                    .get("name")
                    .and_then(|v| v.as_str())
                    .map(|n| n == name)
                    .unwrap_or(false)
                    || props
                        .get("noun_key")
                        .and_then(|v| v.as_str())
                        .map(|n| n == name)
                        .unwrap_or(false);
                if matches {
                    return Some(addr);
                }
            }
        }
    }
    None
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::lance_parser::parser::parse_cypher_query;

    #[test]
    fn test_execute_merge_and_match() {
        let mut bs = BindSpace::new();

        // MERGE a node via parsed AST
        let merge_ast =
            parse_cypher_query("MATCH (s:System {name: 'Predator', military_use: 'Drone'}) RETURN s")
                .unwrap();
        // Manually create a MERGE-style operation
        let node_pat = NodePattern {
            variable: Some("s".to_string()),
            labels: vec!["System".to_string()],
            properties: {
                let mut m = HashMap::new();
                m.insert(
                    "name".to_string(),
                    PropertyValue::String("Predator".to_string()),
                );
                m.insert(
                    "military_use".to_string(),
                    PropertyValue::String("Drone".to_string()),
                );
                m
            },
        };
        let mut result = CypherResult::empty();
        execute_merge_node(&mut bs, &node_pat, &mut result).unwrap();
        assert_eq!(result.nodes_created, 1);

        // MATCH it back using parsed query
        let match_ast =
            parse_cypher_query("MATCH (s:System) RETURN s.name, s.military_use").unwrap();
        let match_result = execute_cypher(&mut bs, &match_ast).unwrap();
        assert_eq!(match_result.rows.len(), 1);
        assert_eq!(
            match_result.rows[0].get("name"),
            Some(&serde_json::json!("Predator"))
        );
    }

    #[test]
    fn test_merge_upsert() {
        let mut bs = BindSpace::new();

        // First MERGE creates
        let node1 = NodePattern {
            variable: Some("s".to_string()),
            labels: vec!["System".to_string()],
            properties: {
                let mut m = HashMap::new();
                m.insert(
                    "name".to_string(),
                    PropertyValue::String("Predator".to_string()),
                );
                m
            },
        };
        let mut r1 = CypherResult::empty();
        execute_merge_node(&mut bs, &node1, &mut r1).unwrap();
        assert_eq!(r1.nodes_created, 1);

        // Second MERGE with same name should NOT create new node
        let node2 = NodePattern {
            variable: Some("s".to_string()),
            labels: vec!["System".to_string()],
            properties: {
                let mut m = HashMap::new();
                m.insert(
                    "name".to_string(),
                    PropertyValue::String("Predator".to_string()),
                );
                m.insert(
                    "type".to_string(),
                    PropertyValue::String("UAV".to_string()),
                );
                m
            },
        };
        let mut r2 = CypherResult::empty();
        execute_merge_node(&mut bs, &node2, &mut r2).unwrap();
        assert_eq!(r2.nodes_created, 0);

        // Should still be only one System
        let match_ast = parse_cypher_query("MATCH (s:System) RETURN s.name").unwrap();
        let match_result = execute_cypher(&mut bs, &match_ast).unwrap();
        assert_eq!(match_result.rows.len(), 1);
    }

    #[test]
    fn test_evaluate_where_equals() {
        let mut bs = BindSpace::new();
        let node_pat = NodePattern {
            variable: None,
            labels: vec!["Person".to_string()],
            properties: {
                let mut m = HashMap::new();
                m.insert(
                    "name".to_string(),
                    PropertyValue::String("Alice".to_string()),
                );
                m.insert("age".to_string(), PropertyValue::Integer(30));
                m
            },
        };
        let mut result = CypherResult::empty();
        execute_merge_node(&mut bs, &node_pat, &mut result).unwrap();

        let ast = parse_cypher_query(
            "MATCH (p:Person) WHERE p.name = 'Alice' RETURN p.name",
        )
        .unwrap();
        let qr = execute_cypher(&mut bs, &ast).unwrap();
        assert_eq!(qr.rows.len(), 1);
        assert_eq!(qr.rows[0].get("name"), Some(&serde_json::json!("Alice")));
    }
}
