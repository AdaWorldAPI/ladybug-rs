//! Cypher to SQL Transpiler
//!
//! Converts Cypher graph queries to SQL using recursive CTEs.
//! Supports MATCH, WHERE, RETURN, ORDER BY, LIMIT patterns.
//!
//! Example:
//! ```cypher
//! MATCH (a:Thought)-[:CAUSES]->(b:Thought)
//! WHERE a.confidence > 0.7
//! RETURN b.content, b.confidence
//! ORDER BY b.confidence DESC
//! LIMIT 10
//! ```
//!
//! Transpiles to:
//! ```sql
//! SELECT t2.content, t2.confidence
//! FROM thoughts t1
//! JOIN edges e ON t1.id = e.source_id AND e.relation = 'CAUSES'
//! JOIN thoughts t2 ON e.target_id = t2.id
//! WHERE t1.confidence > 0.7
//! ORDER BY t2.confidence DESC
//! LIMIT 10
//! ```

use std::collections::HashMap;

use crate::{Result, Error};

/// Cypher query transpiler
pub struct CypherTranspiler {
    /// Node variable -> table alias mapping
    node_aliases: HashMap<String, String>,
    /// Edge variable -> alias mapping  
    edge_aliases: HashMap<String, String>,
    /// Alias counter
    alias_counter: usize,
}

impl CypherTranspiler {
    pub fn new() -> Self {
        Self {
            node_aliases: HashMap::new(),
            edge_aliases: HashMap::new(),
            alias_counter: 0,
        }
    }
    
    fn next_alias(&mut self, prefix: &str) -> String {
        self.alias_counter += 1;
        format!("{}{}", prefix, self.alias_counter)
    }
    
    /// Transpile Cypher to SQL
    pub fn transpile(&mut self, cypher: &str) -> Result<String> {
        // Reset state
        self.node_aliases.clear();
        self.edge_aliases.clear();
        self.alias_counter = 0;
        
        // Parse into components
        let parsed = self.parse_cypher(cypher)?;
        
        // Generate SQL
        self.generate_sql(&parsed)
    }
    
    /// Parse Cypher into components
    fn parse_cypher(&mut self, cypher: &str) -> Result<CypherQuery> {
        let cypher = cypher.trim();
        
        let mut query = CypherQuery::default();
        
        // Split into clauses (simple approach)
        let upper = cypher.to_uppercase();
        
        // Extract MATCH clause
        if let Some(match_start) = upper.find("MATCH") {
            let match_end = upper[match_start..]
                .find("WHERE")
                .or_else(|| upper[match_start..].find("RETURN"))
                .map(|i| match_start + i)
                .unwrap_or(cypher.len());
            
            let match_clause = &cypher[match_start + 5..match_end].trim();
            query.patterns = self.parse_patterns(match_clause)?;
        }
        
        // Extract WHERE clause
        if let Some(where_start) = upper.find("WHERE") {
            let where_end = upper[where_start..]
                .find("RETURN")
                .map(|i| where_start + i)
                .unwrap_or(cypher.len());
            
            query.where_clause = Some(cypher[where_start + 5..where_end].trim().to_string());
        }
        
        // Extract RETURN clause
        if let Some(return_start) = upper.find("RETURN") {
            let return_end = upper[return_start..]
                .find("ORDER")
                .or_else(|| upper[return_start..].find("LIMIT"))
                .map(|i| return_start + i)
                .unwrap_or(cypher.len());
            
            query.return_clause = cypher[return_start + 6..return_end].trim().to_string();
        }
        
        // Extract ORDER BY
        if let Some(order_start) = upper.find("ORDER BY") {
            let order_end = upper[order_start..]
                .find("LIMIT")
                .map(|i| order_start + i)
                .unwrap_or(cypher.len());
            
            query.order_by = Some(cypher[order_start + 8..order_end].trim().to_string());
        }
        
        // Extract LIMIT
        if let Some(limit_start) = upper.find("LIMIT") {
            let limit_str = cypher[limit_start + 5..].trim();
            query.limit = limit_str.parse().ok();
        }
        
        Ok(query)
    }
    
    /// Parse MATCH patterns like (a:Thought)-[:CAUSES]->(b)
    fn parse_patterns(&mut self, pattern_str: &str) -> Result<Vec<Pattern>> {
        let mut patterns = Vec::new();
        
        // Simple pattern parser: (var:Label)-[edge:REL]->(var2:Label)
        // This is a simplified implementation - a real one would use a proper parser
        
        let pattern_str = pattern_str.trim();
        
        // Find node-edge-node patterns
        let mut chars = pattern_str.chars().peekable();
        let mut current_pos = 0;
        
        while current_pos < pattern_str.len() {
            // Find opening paren for first node
            if let Some(node1_start) = pattern_str[current_pos..].find('(') {
                let abs_start = current_pos + node1_start;
                
                // Find closing paren
                if let Some(node1_end) = pattern_str[abs_start..].find(')') {
                    let node1_str = &pattern_str[abs_start + 1..abs_start + node1_end];
                    let (node1_var, node1_label) = self.parse_node(node1_str)?;
                    
                    current_pos = abs_start + node1_end + 1;
                    
                    // Check for edge
                    let remaining = &pattern_str[current_pos..];
                    if remaining.starts_with('-') {
                        // Parse edge: -[var:REL]-> or -[:REL]->
                        if let Some(edge_start) = remaining.find('[') {
                            if let Some(edge_end) = remaining.find(']') {
                                let edge_str = &remaining[edge_start + 1..edge_end];
                                let (edge_var, edge_rel, direction) = self.parse_edge(edge_str, remaining)?;
                                
                                current_pos += edge_end + 1;
                                
                                // Skip arrow
                                let remaining = &pattern_str[current_pos..];
                                if remaining.starts_with("->") {
                                    current_pos += 2;
                                } else if remaining.starts_with("-") {
                                    current_pos += 1;
                                }
                                
                                // Parse second node
                                let remaining = &pattern_str[current_pos..];
                                if let Some(node2_start) = remaining.find('(') {
                                    if let Some(node2_end) = remaining.find(')') {
                                        let node2_str = &remaining[node2_start + 1..node2_end];
                                        let (node2_var, node2_label) = self.parse_node(node2_str)?;
                                        
                                        patterns.push(Pattern {
                                            source: NodePattern {
                                                variable: node1_var,
                                                label: node1_label,
                                            },
                                            edge: Some(EdgePattern {
                                                variable: edge_var,
                                                rel_type: edge_rel,
                                                direction,
                                            }),
                                            target: Some(NodePattern {
                                                variable: node2_var,
                                                label: node2_label,
                                            }),
                                        });
                                        
                                        current_pos += node2_end + 1;
                                    }
                                }
                            }
                        }
                    } else {
                        // Single node pattern
                        patterns.push(Pattern {
                            source: NodePattern {
                                variable: node1_var,
                                label: node1_label,
                            },
                            edge: None,
                            target: None,
                        });
                    }
                }
            } else {
                break;
            }
        }
        
        Ok(patterns)
    }
    
    /// Parse node: "var:Label" or "var" or ":Label"
    fn parse_node(&mut self, node_str: &str) -> Result<(String, Option<String>)> {
        let node_str = node_str.trim();
        
        if node_str.contains(':') {
            let parts: Vec<&str> = node_str.splitn(2, ':').collect();
            let var = if parts[0].is_empty() {
                self.next_alias("n")
            } else {
                parts[0].to_string()
            };
            let label = Some(parts[1].to_string());
            
            self.node_aliases.insert(var.clone(), self.next_alias("t"));
            Ok((var, label))
        } else if node_str.is_empty() {
            let var = self.next_alias("n");
            self.node_aliases.insert(var.clone(), self.next_alias("t"));
            Ok((var, None))
        } else {
            let var = node_str.to_string();
            self.node_aliases.insert(var.clone(), self.next_alias("t"));
            Ok((var, None))
        }
    }
    
    /// Parse edge: "var:REL_TYPE" or ":REL_TYPE" or "*1..3"
    fn parse_edge(&mut self, edge_str: &str, context: &str) -> Result<(Option<String>, Option<String>, EdgeDirection)> {
        let edge_str = edge_str.trim();
        
        // Determine direction
        let direction = if context.contains("->") {
            EdgeDirection::Outgoing
        } else if context.contains("<-") {
            EdgeDirection::Incoming
        } else {
            EdgeDirection::Both
        };
        
        if edge_str.is_empty() {
            return Ok((None, None, direction));
        }
        
        // Check for variable path *1..3
        if edge_str.starts_with('*') {
            // Variable length path - handled specially
            return Ok((None, Some(edge_str.to_string()), direction));
        }
        
        if edge_str.contains(':') {
            let parts: Vec<&str> = edge_str.splitn(2, ':').collect();
            let var = if parts[0].is_empty() {
                None
            } else {
                let v = parts[0].to_string();
                self.edge_aliases.insert(v.clone(), self.next_alias("e"));
                Some(v)
            };
            let rel_type = Some(parts[1].to_string());
            Ok((var, rel_type, direction))
        } else {
            let var = edge_str.to_string();
            self.edge_aliases.insert(var.clone(), self.next_alias("e"));
            Ok((Some(var), None, direction))
        }
    }
    
    /// Generate SQL from parsed Cypher
    fn generate_sql(&self, query: &CypherQuery) -> Result<String> {
        let mut sql = String::new();
        
        // Determine if we need recursive CTE for variable-length paths
        let needs_cte = query.patterns.iter().any(|p| {
            p.edge.as_ref().map(|e| {
                e.rel_type.as_ref().map(|r| r.starts_with('*')).unwrap_or(false)
            }).unwrap_or(false)
        });
        
        if needs_cte {
            sql.push_str(&self.generate_recursive_sql(query)?);
        } else {
            sql.push_str(&self.generate_simple_sql(query)?);
        }
        
        Ok(sql)
    }
    
    /// Generate simple SQL (no variable-length paths)
    fn generate_simple_sql(&self, query: &CypherQuery) -> Result<String> {
        let mut select_parts = Vec::new();
        let mut from_parts = Vec::new();
        let mut join_parts = Vec::new();
        let mut where_parts = Vec::new();
        
        // Process patterns
        for (idx, pattern) in query.patterns.iter().enumerate() {
            let source_alias = self.node_aliases.get(&pattern.source.variable)
                .cloned()
                .unwrap_or_else(|| format!("t{}", idx * 2));
            
            // Source table
            let source_table = pattern.source.label.as_ref()
                .map(|l| self.label_to_table(l))
                .unwrap_or("thoughts".to_string());
            
            if idx == 0 {
                from_parts.push(format!("{} {}", source_table, source_alias));
            }
            
            // Edge and target
            if let (Some(edge), Some(target)) = (&pattern.edge, &pattern.target) {
                let edge_alias = format!("e{}", idx);
                let target_alias = self.node_aliases.get(&target.variable)
                    .cloned()
                    .unwrap_or_else(|| format!("t{}", idx * 2 + 1));
                
                let target_table = target.label.as_ref()
                    .map(|l| self.label_to_table(l))
                    .unwrap_or("thoughts".to_string());
                
                // Build join conditions
                let mut edge_conditions = vec![
                    format!("{}.id = {}.source_id", source_alias, edge_alias),
                ];
                
                if let Some(rel_type) = &edge.rel_type {
                    if !rel_type.starts_with('*') {
                        edge_conditions.push(format!("{}.relation = '{}'", edge_alias, rel_type));
                    }
                }
                
                join_parts.push(format!(
                    "JOIN edges {} ON {}",
                    edge_alias,
                    edge_conditions.join(" AND ")
                ));
                
                join_parts.push(format!(
                    "JOIN {} {} ON {}.target_id = {}.id",
                    target_table, target_alias, edge_alias, target_alias
                ));
            }
        }
        
        // Process RETURN clause
        let return_clause = self.translate_return(&query.return_clause);
        select_parts.push(return_clause);
        
        // Process WHERE clause
        if let Some(where_clause) = &query.where_clause {
            where_parts.push(self.translate_where(where_clause));
        }
        
        // Build SQL
        let mut sql = format!(
            "SELECT {}\nFROM {}",
            select_parts.join(", "),
            from_parts.join(", ")
        );
        
        for join in join_parts {
            sql.push_str(&format!("\n{}", join));
        }
        
        if !where_parts.is_empty() {
            sql.push_str(&format!("\nWHERE {}", where_parts.join(" AND ")));
        }
        
        if let Some(order) = &query.order_by {
            sql.push_str(&format!("\nORDER BY {}", self.translate_order(order)));
        }
        
        if let Some(limit) = query.limit {
            sql.push_str(&format!("\nLIMIT {}", limit));
        }
        
        Ok(sql)
    }
    
    /// Generate SQL with recursive CTE for variable-length paths
    fn generate_recursive_sql(&self, query: &CypherQuery) -> Result<String> {
        // Find the variable-length pattern
        let var_pattern = query.patterns.iter()
            .find(|p| p.edge.as_ref().map(|e| {
                e.rel_type.as_ref().map(|r| r.starts_with('*')).unwrap_or(false)
            }).unwrap_or(false));
        
        let Some(pattern) = var_pattern else {
            return self.generate_simple_sql(query);
        };
        
        // Parse path length: *1..3 or *..5 or *
        let (min_depth, max_depth) = if let Some(edge) = &pattern.edge {
            if let Some(rel_type) = &edge.rel_type {
                self.parse_path_length(rel_type)?
            } else {
                (1, 10) // Default
            }
        } else {
            (1, 10)
        };
        
        let source_alias = self.node_aliases.get(&pattern.source.variable)
            .cloned()
            .unwrap_or("t1".to_string());
        
        let target_alias = pattern.target.as_ref()
            .and_then(|t| self.node_aliases.get(&t.variable))
            .cloned()
            .unwrap_or("t2".to_string());
        
        let sql = format!(r#"
WITH RECURSIVE paths AS (
    -- Base case: direct edges from source
    SELECT 
        source.id as start_id,
        target.id as end_id,
        1 as depth,
        ARRAY[source.id, target.id] as path
    FROM thoughts source
    JOIN edges e ON source.id = e.source_id
    JOIN thoughts target ON e.target_id = target.id
    WHERE depth >= {min_depth}
    
    UNION ALL
    
    -- Recursive case: extend paths
    SELECT 
        p.start_id,
        target.id as end_id,
        p.depth + 1,
        p.path || target.id
    FROM paths p
    JOIN edges e ON p.end_id = e.source_id
    JOIN thoughts target ON e.target_id = target.id
    WHERE p.depth < {max_depth}
    AND NOT target.id = ANY(p.path)  -- Prevent cycles
)
SELECT DISTINCT 
    {source_alias}.*, 
    {target_alias}.*,
    p.depth,
    p.path
FROM paths p
JOIN thoughts {source_alias} ON p.start_id = {source_alias}.id
JOIN thoughts {target_alias} ON p.end_id = {target_alias}.id
"#, min_depth = min_depth, max_depth = max_depth, 
    source_alias = source_alias, target_alias = target_alias);
        
        Ok(sql)
    }
    
    fn parse_path_length(&self, spec: &str) -> Result<(usize, usize)> {
        // Parse: * or *3 or *1..3 or *..5
        let spec = spec.trim_start_matches('*');
        
        if spec.is_empty() {
            return Ok((1, 10));
        }
        
        if spec.contains("..") {
            let parts: Vec<&str> = spec.split("..").collect();
            let min = if parts[0].is_empty() { 1 } else { parts[0].parse().unwrap_or(1) };
            let max = if parts.len() > 1 && !parts[1].is_empty() {
                parts[1].parse().unwrap_or(10)
            } else {
                10
            };
            Ok((min, max))
        } else {
            let n: usize = spec.parse().unwrap_or(1);
            Ok((n, n))
        }
    }
    
    fn label_to_table(&self, label: &str) -> String {
        match label.to_lowercase().as_str() {
            "thought" | "thoughts" => "thoughts",
            "concept" | "concepts" => "thoughts", // Same table, different semantics
            "edge" | "edges" => "edges",
            _ => "thoughts", // Default
        }.to_string()
    }
    
    fn translate_return(&self, return_clause: &str) -> String {
        // Translate property access: a.content -> t1.content
        let mut result = return_clause.to_string();
        
        for (var, alias) in &self.node_aliases {
            result = result.replace(&format!("{}.", var), &format!("{}.", alias));
        }
        
        result
    }
    
    fn translate_where(&self, where_clause: &str) -> String {
        let mut result = where_clause.to_string();
        
        for (var, alias) in &self.node_aliases {
            result = result.replace(&format!("{}.", var), &format!("{}.", alias));
        }
        
        result
    }
    
    fn translate_order(&self, order_clause: &str) -> String {
        let mut result = order_clause.to_string();
        
        for (var, alias) in &self.node_aliases {
            result = result.replace(&format!("{}.", var), &format!("{}.", alias));
        }
        
        result
    }
}

impl Default for CypherTranspiler {
    fn default() -> Self {
        Self::new()
    }
}

// === Data Structures ===

#[derive(Debug, Default)]
struct CypherQuery {
    patterns: Vec<Pattern>,
    where_clause: Option<String>,
    return_clause: String,
    order_by: Option<String>,
    limit: Option<usize>,
}

#[derive(Debug)]
struct Pattern {
    source: NodePattern,
    edge: Option<EdgePattern>,
    target: Option<NodePattern>,
}

#[derive(Debug)]
struct NodePattern {
    variable: String,
    label: Option<String>,
}

#[derive(Debug)]
struct EdgePattern {
    variable: Option<String>,
    rel_type: Option<String>,
    direction: EdgeDirection,
}

#[derive(Debug, Clone, Copy)]
enum EdgeDirection {
    Outgoing,  // ->
    Incoming,  // <-
    Both,      // -
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simple_match() {
        let mut transpiler = CypherTranspiler::new();
        
        let cypher = "MATCH (a:Thought)-[:CAUSES]->(b:Thought) RETURN b.content";
        let sql = transpiler.transpile(cypher).unwrap();
        
        assert!(sql.contains("SELECT"));
        assert!(sql.contains("JOIN edges"));
        assert!(sql.contains("CAUSES"));
    }
    
    #[test]
    fn test_with_where() {
        let mut transpiler = CypherTranspiler::new();
        
        let cypher = r#"
            MATCH (a:Thought)-[:SUPPORTS]->(b)
            WHERE a.confidence > 0.7
            RETURN b.content, b.confidence
            ORDER BY b.confidence DESC
            LIMIT 10
        "#;
        
        let sql = transpiler.transpile(cypher).unwrap();
        
        assert!(sql.contains("WHERE"));
        assert!(sql.contains("ORDER BY"));
        assert!(sql.contains("LIMIT 10"));
    }
    
    #[test]
    fn test_variable_path() {
        let mut transpiler = CypherTranspiler::new();
        
        let cypher = "MATCH (a)-[*1..3]->(b) RETURN b";
        let sql = transpiler.transpile(cypher).unwrap();
        
        assert!(sql.contains("RECURSIVE"));
        assert!(sql.contains("depth"));
    }
    
    #[test]
    fn test_path_length_parsing() {
        let transpiler = CypherTranspiler::new();
        
        assert_eq!(transpiler.parse_path_length("*").unwrap(), (1, 10));
        assert_eq!(transpiler.parse_path_length("*3").unwrap(), (3, 3));
        assert_eq!(transpiler.parse_path_length("*1..5").unwrap(), (1, 5));
        assert_eq!(transpiler.parse_path_length("*..3").unwrap(), (1, 3));
    }
}
