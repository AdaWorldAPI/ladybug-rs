//! Query builder

pub struct Query {
    pub sql: Option<String>,
    pub cypher: Option<String>,
}

impl Query {
    pub fn sql(query: &str) -> Self {
        Self {
            sql: Some(query.to_string()),
            cypher: None,
        }
    }

    pub fn cypher(query: &str) -> Self {
        Self {
            sql: None,
            cypher: Some(query.to_string()),
        }
    }
}

#[derive(Debug)]
pub struct QueryResult {
    pub rows: Vec<Vec<String>>,
    pub columns: Vec<String>,
}
