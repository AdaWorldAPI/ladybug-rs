//! Cypher Bridge — Cypher string → BindSpace operations
//!
//! This module parses Cypher-syntax strings (MERGE, MATCH, SET, CREATE)
//! and translates them into BindSpace write/read operations. It is the
//! bridge between neo4j-rs' AST world and ladybug-rs' BindNode/BindEdge world.
//!
//! The user writes Cypher. ladybug-rs executes it. No external Neo4j needed.
//!
//! ```text
//! Cypher String → parse → CypherOp → execute against BindSpace
//!     MERGE (n:System {name: "X"})  →  write_labeled(fingerprint, "System")
//!     SET n.prop = val              →  update payload on BindNode
//!     MATCH (n:System) RETURN n     →  scan nodes_iter, filter by label
//!     CREATE (a)-[:REL]->(b)        →  link_with_edge(BindEdge)
//! ```

use std::collections::HashMap;

use crate::storage::bind_space::{Addr, BindEdge, BindNode, BindSpace, FINGERPRINT_WORDS};

// =============================================================================
// PARSED CYPHER OPERATIONS
// =============================================================================

/// A parsed Cypher operation ready for BindSpace execution.
#[derive(Debug, Clone)]
pub enum CypherOp {
    /// MERGE (n:Label {props...}) — upsert a node
    MergeNode {
        labels: Vec<String>,
        properties: HashMap<String, CypherValue>,
    },
    /// CREATE (n:Label {props...}) — insert a new node
    CreateNode {
        labels: Vec<String>,
        properties: HashMap<String, CypherValue>,
    },
    /// CREATE (a)-[:TYPE {props}]->(b) — insert an edge
    CreateEdge {
        from_ref: NodeRef,
        to_ref: NodeRef,
        rel_type: String,
        properties: HashMap<String, CypherValue>,
    },
    /// SET n.key = value — update a property on a node
    SetProperty {
        node_ref: NodeRef,
        key: String,
        value: CypherValue,
    },
    /// MATCH (n:Label) WHERE ... RETURN ... — read query
    MatchReturn {
        label: Option<String>,
        where_clause: Option<WhereClause>,
        return_items: Vec<String>,
        order_by: Option<String>,
        limit: Option<usize>,
    },
}

/// Reference to a node (by label+properties for MERGE/CREATE, or by address).
#[derive(Debug, Clone)]
pub enum NodeRef {
    /// Reference by label + property key (for MERGE lookups)
    ByKey { label: String, key: String, value: CypherValue },
    /// Reference by BindSpace address (resolved)
    ByAddr(Addr),
}

/// Simple WHERE clause filter.
#[derive(Debug, Clone)]
pub enum WhereClause {
    /// n.key IS NOT NULL
    IsNotNull { key: String },
    /// n.key = value
    Equals { key: String, value: CypherValue },
    /// n.key CONTAINS value
    Contains { key: String, value: String },
    /// AND of two clauses
    And(Box<WhereClause>, Box<WhereClause>),
}

/// Cypher literal value.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub enum CypherValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Null,
}

impl std::fmt::Display for CypherValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CypherValue::String(s) => write!(f, "{}", s),
            CypherValue::Int(i) => write!(f, "{}", i),
            CypherValue::Float(v) => write!(f, "{}", v),
            CypherValue::Bool(b) => write!(f, "{}", b),
            CypherValue::Null => write!(f, "null"),
        }
    }
}

// =============================================================================
// QUERY RESULT
// =============================================================================

/// Result of a Cypher query execution against BindSpace.
#[derive(Debug, Clone)]
pub struct CypherResult {
    pub columns: Vec<String>,
    pub rows: Vec<HashMap<String, CypherValue>>,
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
// PARSE: Simple Cypher string → CypherOp
// =============================================================================

/// Parse a Cypher string into a sequence of operations.
///
/// This is a lightweight parser for the most common Cypher patterns.
/// For full Cypher parsing, use neo4j-rs' parser + planner.
pub fn parse_cypher(cypher: &str) -> Result<Vec<CypherOp>, String> {
    let trimmed = cypher.trim();
    let upper = trimmed.to_uppercase();

    if upper.starts_with("MERGE") {
        parse_merge(trimmed)
    } else if upper.starts_with("CREATE") {
        parse_create(trimmed)
    } else if upper.starts_with("MATCH") {
        parse_match(trimmed)
    } else {
        Err(format!("Unsupported Cypher statement: {}", &trimmed[..trimmed.len().min(40)]))
    }
}

fn parse_merge(cypher: &str) -> Result<Vec<CypherOp>, String> {
    // MERGE (n:Label {key: 'value', ...})
    let (labels, properties) = parse_node_pattern(cypher)
        .map_err(|e| format!("MERGE parse error: {}", e))?;

    Ok(vec![CypherOp::MergeNode { labels, properties }])
}

fn parse_create(cypher: &str) -> Result<Vec<CypherOp>, String> {
    // CREATE (n:Label {key: 'value', ...})
    let (labels, properties) = parse_node_pattern(cypher)
        .map_err(|e| format!("CREATE parse error: {}", e))?;

    Ok(vec![CypherOp::CreateNode { labels, properties }])
}

fn parse_match(cypher: &str) -> Result<Vec<CypherOp>, String> {
    // MATCH (n:Label) WHERE ... RETURN ...
    let upper = cypher.to_uppercase();

    // Extract label from pattern
    let label = extract_label(cypher);

    // Extract WHERE clause
    let where_clause = if let Some(where_pos) = upper.find("WHERE") {
        let return_pos = upper.find("RETURN").unwrap_or(upper.len());
        let where_str = &cypher[where_pos + 5..return_pos].trim();
        parse_where_clause(where_str).ok()
    } else {
        None
    };

    // Extract RETURN items
    let return_items = if let Some(ret_pos) = upper.find("RETURN") {
        let after_return = &cypher[ret_pos + 6..];
        let end = after_return.to_uppercase().find("ORDER BY")
            .or_else(|| after_return.to_uppercase().find("LIMIT"))
            .unwrap_or(after_return.len());
        after_return[..end]
            .split(',')
            .map(|s| s.trim().to_string())
            .collect()
    } else {
        vec!["*".to_string()]
    };

    // Extract ORDER BY
    let order_by = if let Some(pos) = upper.find("ORDER BY") {
        let after = &cypher[pos + 8..];
        let end = after.to_uppercase().find("LIMIT").unwrap_or(after.len());
        Some(after[..end].trim().to_string())
    } else {
        None
    };

    // Extract LIMIT
    let limit = if let Some(pos) = upper.find("LIMIT") {
        cypher[pos + 5..].trim().parse::<usize>().ok()
    } else {
        None
    };

    Ok(vec![CypherOp::MatchReturn {
        label,
        where_clause,
        return_items,
        order_by,
        limit,
    }])
}

// =============================================================================
// EXECUTE: CypherOp → BindSpace mutations/reads
// =============================================================================

/// Execute a sequence of Cypher operations against a BindSpace.
pub fn execute_cypher(
    bs: &mut BindSpace,
    ops: &[CypherOp],
) -> Result<CypherResult, String> {
    let mut result = CypherResult::empty();

    for op in ops {
        match op {
            CypherOp::MergeNode { labels, properties } => {
                execute_merge_node(bs, labels, properties, &mut result)?;
            }
            CypherOp::CreateNode { labels, properties } => {
                execute_create_node(bs, labels, properties, &mut result)?;
            }
            CypherOp::CreateEdge { from_ref, to_ref, rel_type, properties: _ } => {
                execute_create_edge(bs, from_ref, to_ref, rel_type, &mut result)?;
            }
            CypherOp::SetProperty { node_ref, key, value } => {
                execute_set_property(bs, node_ref, key, value, &mut result)?;
            }
            CypherOp::MatchReturn { label, where_clause, return_items, order_by: _, limit } => {
                execute_match_return(bs, label, where_clause, return_items, limit, &mut result)?;
            }
        }
    }

    Ok(result)
}

fn execute_merge_node(
    bs: &mut BindSpace,
    labels: &[String],
    properties: &HashMap<String, CypherValue>,
    result: &mut CypherResult,
) -> Result<(), String> {
    // Check if node with same label + name already exists (upsert)
    let primary_label = labels.first().map(|s| s.as_str()).unwrap_or("Node");
    let name_prop = properties.get("name").or_else(|| properties.get("noun_key"));

    if let Some(name) = name_prop {
        // Search existing nodes for match
        let name_str = name.to_string();
        let existing = find_node_by_label_and_name(bs, primary_label, &name_str);

        if let Some(addr) = existing {
            // Node exists — update properties as payload
            if let Some(node) = bs.read_mut(addr) {
                let payload = serde_json::to_vec(properties).unwrap_or_default();
                node.payload = Some(payload);
                result.properties_set += properties.len();
            }
            return Ok(());
        }
    }

    // Node doesn't exist — create it
    let fingerprint = properties_to_fingerprint(primary_label, properties);
    let addr = bs.write_labeled(fingerprint, primary_label);

    // Store properties as JSON payload
    if let Some(node) = bs.read_mut(addr) {
        let payload = serde_json::to_vec(properties).unwrap_or_default();
        node.payload = Some(payload);
    }

    result.nodes_created += 1;
    result.properties_set += properties.len();
    Ok(())
}

fn execute_create_node(
    bs: &mut BindSpace,
    labels: &[String],
    properties: &HashMap<String, CypherValue>,
    result: &mut CypherResult,
) -> Result<(), String> {
    let primary_label = labels.first().map(|s| s.as_str()).unwrap_or("Node");
    let fingerprint = properties_to_fingerprint(primary_label, properties);
    let addr = bs.write_labeled(fingerprint, primary_label);

    if let Some(node) = bs.read_mut(addr) {
        let payload = serde_json::to_vec(properties).unwrap_or_default();
        node.payload = Some(payload);
    }

    result.nodes_created += 1;
    result.properties_set += properties.len();
    Ok(())
}

fn execute_create_edge(
    bs: &mut BindSpace,
    from_ref: &NodeRef,
    to_ref: &NodeRef,
    rel_type: &str,
    result: &mut CypherResult,
) -> Result<(), String> {
    let from_addr = resolve_node_ref(bs, from_ref)
        .ok_or_else(|| format!("Cannot resolve source node: {:?}", from_ref))?;
    let to_addr = resolve_node_ref(bs, to_ref)
        .ok_or_else(|| format!("Cannot resolve target node: {:?}", to_ref))?;

    // Use verb prefix 0x07 for relationship types
    let verb_fp = label_to_fingerprint(rel_type);
    let verb_addr = bs.write_labeled(verb_fp, rel_type);

    let edge = BindEdge::new(from_addr, verb_addr, to_addr);
    bs.link_with_edge(edge);

    result.relationships_created += 1;
    Ok(())
}

fn execute_set_property(
    bs: &mut BindSpace,
    node_ref: &NodeRef,
    key: &str,
    value: &CypherValue,
    result: &mut CypherResult,
) -> Result<(), String> {
    let addr = resolve_node_ref(bs, node_ref)
        .ok_or_else(|| format!("Cannot resolve node: {:?}", node_ref))?;

    if let Some(node) = bs.read_mut(addr) {
        // Read existing payload, update property, write back
        let mut props: HashMap<String, serde_json::Value> = node.payload
            .as_ref()
            .and_then(|p| serde_json::from_slice(p).ok())
            .unwrap_or_default();

        props.insert(key.to_string(), cypher_value_to_json(value));

        node.payload = Some(serde_json::to_vec(&props).unwrap_or_default());
        result.properties_set += 1;
    }

    Ok(())
}

fn execute_match_return(
    bs: &BindSpace,
    label: &Option<String>,
    where_clause: &Option<WhereClause>,
    return_items: &[String],
    limit: &Option<usize>,
    result: &mut CypherResult,
) -> Result<(), String> {
    // Scan all nodes, filter by label and WHERE
    let mut matching_nodes: Vec<(Addr, &BindNode)> = Vec::new();

    for (addr, node) in bs.nodes_iter() {
        // Label filter
        if let Some(lbl) = &label {
            match &node.label {
                Some(node_label) if node_label == lbl => {}
                _ => continue,
            }
        }

        // WHERE filter
        if let Some(wc) = &where_clause {
            if !evaluate_where(node, wc) {
                continue;
            }
        }

        matching_nodes.push((addr, node));
    }

    // Apply LIMIT
    if let Some(lim) = limit {
        matching_nodes.truncate(*lim);
    }

    // Build result columns from return items
    let columns: Vec<String> = if return_items.len() == 1 && return_items[0] == "*" {
        vec!["addr".to_string(), "label".to_string(), "properties".to_string()]
    } else {
        return_items.iter().map(|item| {
            // Strip alias: "n.name AS name" -> "name", "n.name" -> "name"
            if let Some(alias_pos) = item.to_uppercase().find(" AS ") {
                item[alias_pos + 4..].trim().to_string()
            } else if let Some(dot_pos) = item.find('.') {
                item[dot_pos + 1..].trim().to_string()
            } else {
                item.trim().to_string()
            }
        }).collect()
    };

    result.columns = columns.clone();

    for (addr, node) in &matching_nodes {
        let props: HashMap<String, serde_json::Value> = node.payload
            .as_ref()
            .and_then(|p| serde_json::from_slice(p).ok())
            .unwrap_or_default();

        let mut row = HashMap::new();

        for (i, col) in columns.iter().enumerate() {
            let return_item = return_items.get(i).map(|s| s.as_str()).unwrap_or(col);
            let prop_key = extract_property_key(return_item).unwrap_or(col.as_str());

            let val = match prop_key {
                "addr" => CypherValue::String(format!("0x{:04X}", addr.0)),
                "label" => CypherValue::String(
                    node.label.clone().unwrap_or_else(|| "?".to_string())
                ),
                "properties" => CypherValue::String(
                    serde_json::to_string(&props).unwrap_or_else(|_| "{}".to_string())
                ),
                key => {
                    if let Some(v) = props.get(key) {
                        json_to_cypher_value(v)
                    } else {
                        CypherValue::Null
                    }
                }
            };

            row.insert(col.clone(), val);
        }

        result.rows.push(row);
    }

    Ok(())
}

// =============================================================================
// HELPERS
// =============================================================================

/// Generate a deterministic fingerprint from label + properties.
fn properties_to_fingerprint(
    label: &str,
    properties: &HashMap<String, CypherValue>,
) -> [u64; FINGERPRINT_WORDS] {
    // Hash label into first portion, properties into remaining
    let mut content = label.to_string();
    // Sort properties for determinism
    let mut sorted: Vec<_> = properties.iter().collect();
    sorted.sort_by_key(|(k, _)| k.clone());
    for (k, v) in sorted {
        content.push(':');
        content.push_str(k);
        content.push('=');
        content.push_str(&v.to_string());
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
fn find_node_by_label_and_name(bs: &BindSpace, label: &str, name: &str) -> Option<Addr> {
    for (addr, node) in bs.nodes_iter() {
        if node.label.as_deref() != Some(label) {
            continue;
        }
        if let Some(ref payload) = node.payload {
            if let Ok(props) = serde_json::from_slice::<HashMap<String, serde_json::Value>>(payload) {
                let matches = props.get("name")
                    .and_then(|v| v.as_str())
                    .map(|n| n == name)
                    .unwrap_or(false)
                || props.get("noun_key")
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

/// Resolve a NodeRef to an Addr.
fn resolve_node_ref(bs: &BindSpace, node_ref: &NodeRef) -> Option<Addr> {
    match node_ref {
        NodeRef::ByAddr(addr) => Some(*addr),
        NodeRef::ByKey { label, key, value } => {
            let value_str = value.to_string();
            for (addr, node) in bs.nodes_iter() {
                if node.label.as_deref() != Some(label.as_str()) {
                    continue;
                }
                if let Some(ref payload) = node.payload {
                    if let Ok(props) = serde_json::from_slice::<HashMap<String, serde_json::Value>>(payload) {
                        if props.get(key.as_str())
                            .and_then(|v| v.as_str())
                            .map(|v| v == value_str)
                            .unwrap_or(false)
                        {
                            return Some(addr);
                        }
                    }
                }
            }
            None
        }
    }
}

/// Evaluate a WHERE clause against a BindNode.
fn evaluate_where(node: &BindNode, clause: &WhereClause) -> bool {
    let props: HashMap<String, serde_json::Value> = node.payload
        .as_ref()
        .and_then(|p| serde_json::from_slice(p).ok())
        .unwrap_or_default();

    match clause {
        WhereClause::IsNotNull { key } => {
            props.get(key).map(|v| !v.is_null()).unwrap_or(false)
        }
        WhereClause::Equals { key, value } => {
            props.get(key)
                .map(|v| json_to_cypher_value(v) == *value)
                .unwrap_or(false)
        }
        WhereClause::Contains { key, value } => {
            props.get(key)
                .and_then(|v| v.as_str())
                .map(|s| s.contains(value.as_str()))
                .unwrap_or(false)
        }
        WhereClause::And(left, right) => {
            evaluate_where(node, left) && evaluate_where(node, right)
        }
    }
}

/// Parse a WHERE clause string into a WhereClause.
fn parse_where_clause(s: &str) -> Result<WhereClause, String> {
    let trimmed = s.trim();

    // Handle AND
    if let Some(pos) = trimmed.to_uppercase().find(" AND ") {
        let left = parse_where_clause(&trimmed[..pos])?;
        let right = parse_where_clause(&trimmed[pos + 5..])?;
        return Ok(WhereClause::And(Box::new(left), Box::new(right)));
    }

    // IS NOT NULL
    if trimmed.to_uppercase().ends_with("IS NOT NULL") {
        let key = trimmed[..trimmed.len() - 11].trim();
        let key = strip_variable_prefix(key);
        return Ok(WhereClause::IsNotNull { key: key.to_string() });
    }

    // CONTAINS
    if let Some(pos) = trimmed.to_uppercase().find(" CONTAINS ") {
        let key = strip_variable_prefix(trimmed[..pos].trim());
        let value = trimmed[pos + 10..].trim().trim_matches('\'').trim_matches('"');
        return Ok(WhereClause::Contains {
            key: key.to_string(),
            value: value.to_string(),
        });
    }

    // Equals: n.key = value
    if let Some(pos) = trimmed.find('=') {
        if !trimmed[..pos].ends_with('!') && !trimmed[..pos].ends_with('<') && !trimmed[..pos].ends_with('>') {
            let key = strip_variable_prefix(trimmed[..pos].trim());
            let val_str = trimmed[pos + 1..].trim().trim_matches('\'').trim_matches('"');
            let value = parse_cypher_literal(val_str);
            return Ok(WhereClause::Equals {
                key: key.to_string(),
                value,
            });
        }
    }

    Err(format!("Cannot parse WHERE clause: {}", trimmed))
}

/// Extract label from a MATCH/MERGE/CREATE pattern like "(n:System {...})"
fn extract_label(cypher: &str) -> Option<String> {
    // Find first (variable:Label pattern
    let mut in_parens = false;
    let chars: Vec<char> = cypher.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '(' {
            in_parens = true;
            i += 1;
            // Skip whitespace
            while i < chars.len() && chars[i].is_whitespace() { i += 1; }
            // Skip variable name
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') { i += 1; }
            // Check for colon (label indicator)
            if i < chars.len() && chars[i] == ':' {
                i += 1;
                let start = i;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                if i > start {
                    return Some(cypher[start..i].to_string());
                }
            }
        }
        i += 1;
    }
    None
}

/// Parse node pattern: (alias:Label {key: 'value', ...})
fn parse_node_pattern(cypher: &str) -> Result<(Vec<String>, HashMap<String, CypherValue>), String> {
    let mut labels = Vec::new();
    let mut properties = HashMap::new();

    // Find content between first ( and matching )
    let open = cypher.find('(').ok_or("No opening paren")?;
    let close = cypher.rfind(')').ok_or("No closing paren")?;
    let inner = &cypher[open + 1..close].trim();

    // Extract labels (after colon, before {)
    let brace_start = inner.find('{').unwrap_or(inner.len());
    let label_part = &inner[..brace_start];

    for part in label_part.split(':').skip(1) {
        let label = part.trim().split_whitespace().next().unwrap_or("").to_string();
        if !label.is_empty() {
            labels.push(label);
        }
    }

    // Extract properties from { ... }
    if let (Some(start), Some(end)) = (inner.find('{'), inner.rfind('}')) {
        let props_str = &inner[start + 1..end];
        for pair in split_properties(props_str) {
            let pair = pair.trim();
            if let Some(colon_pos) = pair.find(':') {
                let key = pair[..colon_pos].trim().to_string();
                let val_str = pair[colon_pos + 1..].trim();
                let value = parse_cypher_literal(val_str);
                properties.insert(key, value);
            }
        }
    }

    Ok((labels, properties))
}

/// Split property pairs, respecting quoted strings.
fn split_properties(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut in_quote = false;
    let mut quote_char = '"';

    for ch in s.chars() {
        if !in_quote && (ch == '\'' || ch == '"') {
            in_quote = true;
            quote_char = ch;
            current.push(ch);
        } else if in_quote && ch == quote_char {
            in_quote = false;
            current.push(ch);
        } else if !in_quote && ch == ',' {
            parts.push(current.clone());
            current.clear();
        } else {
            current.push(ch);
        }
    }

    if !current.trim().is_empty() {
        parts.push(current);
    }
    parts
}

/// Parse a Cypher literal value.
fn parse_cypher_literal(s: &str) -> CypherValue {
    let trimmed = s.trim();

    if trimmed.eq_ignore_ascii_case("null") {
        return CypherValue::Null;
    }
    if trimmed.eq_ignore_ascii_case("true") {
        return CypherValue::Bool(true);
    }
    if trimmed.eq_ignore_ascii_case("false") {
        return CypherValue::Bool(false);
    }

    // Quoted string
    if (trimmed.starts_with('\'') && trimmed.ends_with('\''))
        || (trimmed.starts_with('"') && trimmed.ends_with('"'))
    {
        return CypherValue::String(trimmed[1..trimmed.len() - 1].to_string());
    }

    // Integer
    if let Ok(i) = trimmed.parse::<i64>() {
        return CypherValue::Int(i);
    }

    // Float
    if let Ok(f) = trimmed.parse::<f64>() {
        return CypherValue::Float(f);
    }

    // Default to string
    CypherValue::String(trimmed.to_string())
}

/// Strip "n." prefix from property access.
fn strip_variable_prefix(s: &str) -> &str {
    if let Some(dot_pos) = s.find('.') {
        &s[dot_pos + 1..]
    } else {
        s
    }
}

/// Extract property key from a return item like "n.name" or "n.name AS alias".
fn extract_property_key(item: &str) -> Option<&str> {
    let base = if let Some(pos) = item.to_uppercase().find(" AS ") {
        &item[..pos]
    } else {
        item
    };

    if let Some(dot_pos) = base.find('.') {
        Some(base[dot_pos + 1..].trim())
    } else {
        None
    }
}

/// Convert CypherValue to serde_json::Value.
fn cypher_value_to_json(val: &CypherValue) -> serde_json::Value {
    match val {
        CypherValue::String(s) => serde_json::Value::String(s.clone()),
        CypherValue::Int(i) => serde_json::json!(*i),
        CypherValue::Float(f) => serde_json::json!(*f),
        CypherValue::Bool(b) => serde_json::Value::Bool(*b),
        CypherValue::Null => serde_json::Value::Null,
    }
}

/// Convert serde_json::Value to CypherValue.
fn json_to_cypher_value(val: &serde_json::Value) -> CypherValue {
    match val {
        serde_json::Value::String(s) => CypherValue::String(s.clone()),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                CypherValue::Int(i)
            } else if let Some(f) = n.as_f64() {
                CypherValue::Float(f)
            } else {
                CypherValue::Null
            }
        }
        serde_json::Value::Bool(b) => CypherValue::Bool(*b),
        serde_json::Value::Null => CypherValue::Null,
        other => CypherValue::String(other.to_string()),
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_merge() {
        let ops = parse_cypher("MERGE (s:System {name: 'Predator', type: 'UAV'})").unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            CypherOp::MergeNode { labels, properties } => {
                assert_eq!(labels, &["System"]);
                assert_eq!(properties.get("name"), Some(&CypherValue::String("Predator".to_string())));
                assert_eq!(properties.get("type"), Some(&CypherValue::String("UAV".to_string())));
            }
            _ => panic!("Expected MergeNode"),
        }
    }

    #[test]
    fn test_parse_match_return() {
        let ops = parse_cypher(
            "MATCH (s:System) WHERE s.military_use IS NOT NULL RETURN s.name, s.military_use ORDER BY s.name LIMIT 10"
        ).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            CypherOp::MatchReturn { label, where_clause, return_items, order_by, limit } => {
                assert_eq!(label, &Some("System".to_string()));
                assert!(where_clause.is_some());
                assert_eq!(return_items.len(), 2);
                assert!(order_by.is_some());
                assert_eq!(limit, &Some(10));
            }
            _ => panic!("Expected MatchReturn"),
        }
    }

    #[test]
    fn test_extract_label() {
        assert_eq!(extract_label("MATCH (n:Person)"), Some("Person".to_string()));
        assert_eq!(extract_label("MERGE (s:System {name: 'X'})"), Some("System".to_string()));
        assert_eq!(extract_label("MATCH ()"), None);
    }

    #[test]
    fn test_parse_cypher_literal() {
        assert_eq!(parse_cypher_literal("'hello'"), CypherValue::String("hello".to_string()));
        assert_eq!(parse_cypher_literal("42"), CypherValue::Int(42));
        assert_eq!(parse_cypher_literal("3.14"), CypherValue::Float(3.14));
        assert_eq!(parse_cypher_literal("true"), CypherValue::Bool(true));
        assert_eq!(parse_cypher_literal("null"), CypherValue::Null);
    }

    #[test]
    fn test_execute_merge_and_match() {
        let mut bs = BindSpace::new();

        // MERGE a node
        let merge_ops = parse_cypher("MERGE (s:System {name: 'Predator', military_use: 'Drone'})").unwrap();
        let merge_result = execute_cypher(&mut bs, &merge_ops).unwrap();
        assert_eq!(merge_result.nodes_created, 1);

        // MATCH it back
        let match_ops = parse_cypher("MATCH (s:System) RETURN s.name, s.military_use").unwrap();
        let match_result = execute_cypher(&mut bs, &match_ops).unwrap();
        assert_eq!(match_result.rows.len(), 1);
        assert_eq!(
            match_result.rows[0].get("name"),
            Some(&CypherValue::String("Predator".to_string()))
        );
    }

    #[test]
    fn test_merge_upsert() {
        let mut bs = BindSpace::new();

        // First MERGE creates
        let ops1 = parse_cypher("MERGE (s:System {name: 'Predator'})").unwrap();
        let r1 = execute_cypher(&mut bs, &ops1).unwrap();
        assert_eq!(r1.nodes_created, 1);

        // Second MERGE with same name should NOT create a new node
        let ops2 = parse_cypher("MERGE (s:System {name: 'Predator', type: 'UAV'})").unwrap();
        let r2 = execute_cypher(&mut bs, &ops2).unwrap();
        assert_eq!(r2.nodes_created, 0);

        // Should still be only one System
        let match_ops = parse_cypher("MATCH (s:System) RETURN s.name").unwrap();
        let match_result = execute_cypher(&mut bs, &match_ops).unwrap();
        assert_eq!(match_result.rows.len(), 1);
    }
}
