//! CTE Builder — Recursive CTE generation for graph traversals.
//!
//! Saved from src/query/cypher.rs (P1) before deletion.
//! These functions generated recursive CTEs for variable-length path queries.
//! They need adaptation to use lance_parser::ast types before they can be used.
//!
//! TODO: Adapt to take lance_parser::ast::PathPattern + LengthRange
//!       instead of the deleted P1 Pattern/EdgePattern types.

/// Build a recursive CTE SQL string for variable-length path traversal.
///
/// Takes start label, edge type filter, hop range, end label filter,
/// user WHERE clause, and LIMIT — returns a SQL string with a
/// WITH RECURSIVE that does cycle-detected BFS.
pub fn build_recursive_cte(
    start_label: Option<&str>,
    edge_types: &[String],
    min_hops: u32,
    max_hops: u32,
    end_label: Option<&str>,
    user_where: Option<&str>,
    limit: Option<u64>,
) -> String {
    let edge_type_filter = if !edge_types.is_empty() {
        format!(
            "AND e.type IN ({})",
            edge_types
                .iter()
                .map(|t| format!("'{}'", t))
                .collect::<Vec<_>>()
                .join(", ")
        )
    } else {
        String::new()
    };

    let start_where = match start_label {
        Some(label) => format!("WHERE label = '{}'", label),
        None => String::new(),
    };

    let end_label_filter = match end_label {
        Some(label) => format!("  AND n.label = '{}'", label),
        None => String::new(),
    };

    let user_where_clause = match user_where {
        Some(w) => format!("  AND ({})", w),
        None => String::new(),
    };

    let limit_clause = match limit {
        Some(l) => format!("LIMIT {}", l),
        None => String::new(),
    };

    format!(
        r#"
WITH RECURSIVE traverse AS (
    -- Base case: start nodes
    SELECT
        id,
        ARRAY[id] as path,
        1.0 as amplification,
        0 as depth
    FROM nodes
    {start_where}

    UNION ALL

    -- Recursive case: follow edges
    SELECT
        n.id,
        t.path || n.id,
        t.amplification * COALESCE(e.amplification, e.weight, 1.0),
        t.depth + 1
    FROM traverse t
    JOIN edges e ON t.id = e.from_id {edge_type_filter}
    JOIN nodes n ON e.to_id = n.id
    WHERE t.depth < {max_depth}
      AND n.id != ALL(t.path)  -- Cycle detection
)
SELECT t.*, n.*
FROM traverse t
JOIN nodes n ON t.id = n.id
WHERE t.depth >= {min_depth}
{end_label_filter}
{user_where_clause}
ORDER BY t.depth, t.amplification DESC
{limit_clause}
"#,
        start_where = start_where,
        edge_type_filter = edge_type_filter,
        max_depth = max_hops,
        min_depth = min_hops,
        end_label_filter = end_label_filter,
        user_where_clause = user_where_clause,
        limit_clause = limit_clause,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_cte() {
        let sql = build_recursive_cte(
            Some("Thought"),
            &["CAUSES".to_string()],
            1,
            5,
            None,
            None,
            Some(10),
        );
        assert!(sql.contains("WITH RECURSIVE traverse"));
        assert!(sql.contains("'CAUSES'"));
        assert!(sql.contains("WHERE label = 'Thought'"));
        assert!(sql.contains("LIMIT 10"));
    }
}
