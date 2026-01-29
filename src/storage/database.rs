//! Main Database API - Unified Interface
//!
//! Combines all operations:
//! - SQL queries (DataFusion)
//! - Cypher queries (transpiled)
//! - Vector search (Lance ANN)
//! - Hamming search (SIMD)
//! - NARS inference
//! - Counterfactual reasoning

use std::path::Path;
use std::sync::Arc;

use parking_lot::RwLock;
use tokio::runtime::Runtime;

use crate::core::{Fingerprint, HammingEngine};
use crate::nars::TruthValue;
use crate::query::{SqlExecutor, QueryResult, CypherTranspiler};
use crate::storage::lance::LanceStore;
use crate::{Result, Error};

/// Main database handle
pub struct Database {
    /// Path to database
    path: String,
    /// Lance storage backend
    lance: Arc<RwLock<Option<LanceStore>>>,
    /// Hamming search engine
    hamming: Arc<RwLock<HammingEngine>>,
    /// SQL executor
    sql_executor: SqlExecutor,
    /// Cypher transpiler
    cypher_transpiler: Arc<RwLock<CypherTranspiler>>,
    /// Async runtime
    runtime: Runtime,
    /// Current version (for COW)
    version: u64,
}

impl Database {
    /// Open or create database at path
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        std::fs::create_dir_all(&path_str)?;
        
        let runtime = Runtime::new()
            .map_err(|e| Error::Storage(format!("Failed to create runtime: {}", e)))?;
        
        // Open Lance storage
        let lance = runtime.block_on(async {
            LanceStore::open(&path_str).await
        })?;
        
        let sql_executor = SqlExecutor::new();
        
        // Register Lance tables with DataFusion
        runtime.block_on(async {
            sql_executor.register_lance_table("thoughts", &format!("{}/thoughts.lance", path_str)).await?;
            sql_executor.register_lance_table("edges", &format!("{}/edges.lance", path_str)).await?;
            Ok::<_, Error>(())
        })?;
        
        Ok(Self {
            path: path_str,
            lance: Arc::new(RwLock::new(Some(lance))),
            hamming: Arc::new(RwLock::new(HammingEngine::new())),
            sql_executor,
            cypher_transpiler: Arc::new(RwLock::new(CypherTranspiler::new())),
            runtime,
            version: 0,
        })
    }
    
    /// In-memory database
    pub fn memory() -> Result<Self> {
        let runtime = Runtime::new()
            .map_err(|e| Error::Storage(format!("Failed to create runtime: {}", e)))?;
        
        Ok(Self {
            path: ":memory:".to_string(),
            lance: Arc::new(RwLock::new(None)),
            hamming: Arc::new(RwLock::new(HammingEngine::new())),
            sql_executor: SqlExecutor::new(),
            cypher_transpiler: Arc::new(RwLock::new(CypherTranspiler::new())),
            runtime,
            version: 0,
        })
    }
    
    // ========== SQL Operations ==========
    
    /// Execute SQL query
    pub fn sql(&self, query: &str) -> Result<QueryResult> {
        self.runtime.block_on(async {
            self.sql_executor.query(query).await
        })
    }
    
    /// Execute SQL returning raw batches
    pub fn sql_raw(&self, query: &str) -> Result<Vec<arrow::array::RecordBatch>> {
        self.runtime.block_on(async {
            self.sql_executor.execute(query).await
        })
    }
    
    // ========== Cypher Operations ==========
    
    /// Execute Cypher query (transpiled to SQL)
    pub fn cypher(&self, query: &str) -> Result<QueryResult> {
        let sql = {
            let mut transpiler = self.cypher_transpiler.write();
            transpiler.transpile(query)?
        };
        
        self.sql(&sql)
    }
    
    /// Transpile Cypher to SQL (for debugging)
    pub fn cypher_to_sql(&self, query: &str) -> Result<String> {
        let mut transpiler = self.cypher_transpiler.write();
        transpiler.transpile(query)
    }
    
    // ========== Vector Operations ==========
    
    /// Vector similarity search (ANN)
    pub fn vector_search(&self, embedding: &[f32], k: usize) -> Result<Vec<(String, f32)>> {
        let lance = self.lance.read();
        let Some(ref store) = *lance else {
            return Ok(vec![]);
        };
        
        self.runtime.block_on(async {
            store.vector_search(embedding, k).await
        })
    }
    
    /// Create vector index
    pub fn create_vector_index(&self) -> Result<()> {
        let mut lance = self.lance.write();
        let Some(ref mut store) = *lance else {
            return Ok(());
        };
        
        self.runtime.block_on(async {
            store.create_vector_index().await
        })
    }
    
    // ========== Hamming Operations ==========
    
    /// Resonance search (Hamming similarity)
    pub fn resonate(
        &self,
        fingerprint: &Fingerprint,
        threshold: f32,
        limit: usize,
    ) -> Vec<(usize, f32)> {
        let engine = self.hamming.read();
        engine.search_threshold(fingerprint, threshold, limit)
            .into_iter()
            .map(|(idx, _, sim)| (idx, sim))
            .collect()
    }
    
    /// Resonate by content
    pub fn resonate_content(
        &self,
        content: &str,
        threshold: f32,
        limit: usize,
    ) -> Vec<(usize, f32)> {
        let fp = Fingerprint::from_content(content);
        self.resonate(&fp, threshold, limit)
    }
    
    /// Index a fingerprint for Hamming search
    pub fn index_fingerprint(&self, fingerprint: &Fingerprint) -> usize {
        let mut engine = self.hamming.write();
        engine.add(fingerprint.clone())
    }
    
    /// Batch index fingerprints
    pub fn index_fingerprints(&self, fingerprints: &[Fingerprint]) -> Vec<usize> {
        let mut engine = self.hamming.write();
        fingerprints.iter().map(|fp| engine.add(fp.clone())).collect()
    }
    
    // ========== Write Operations ==========
    
    /// Insert a thought
    pub fn insert_thought(
        &self,
        id: &str,
        content: &str,
        frequency: f32,
        confidence: f32,
    ) -> Result<()> {
        let fp = Fingerprint::from_content(content);
        
        // Index in Hamming engine
        self.index_fingerprint(&fp);
        
        // Insert in Lance
        let mut lance = self.lance.write();
        if let Some(ref mut store) = *lance {
            self.runtime.block_on(async {
                store.insert_thought(id, content, &fp, None, frequency, confidence).await
            })?;
        }
        
        Ok(())
    }
    
    /// Insert an edge
    pub fn insert_edge(
        &self,
        id: &str,
        source_id: &str,
        target_id: &str,
        relation: &str,
        frequency: f32,
        confidence: f32,
    ) -> Result<()> {
        let mut lance = self.lance.write();
        if let Some(ref mut store) = *lance {
            self.runtime.block_on(async {
                store.insert_edge(id, source_id, target_id, relation, frequency, confidence, None).await
            })?;
        }
        Ok(())
    }
    
    // ========== Graph Operations ==========
    
    /// Get outgoing edges from a node
    pub fn edges_from(&self, source_id: &str) -> Result<Vec<crate::storage::lance::EdgeRow>> {
        let lance = self.lance.read();
        let Some(ref store) = *lance else {
            return Ok(vec![]);
        };
        
        self.runtime.block_on(async {
            store.get_edges_from(source_id).await
        })
    }
    
    /// Graph traversal via Cypher
    pub fn traverse(&self, start: &str, pattern: &str, max_depth: usize) -> Result<QueryResult> {
        let cypher = format!(
            "MATCH (a {{id: '{}'}})-[*1..{}]->(b) RETURN b",
            start, max_depth
        );
        self.cypher(&cypher)
    }
    
    // ========== Counterfactual Operations ==========
    
    /// Fork database for "what if" analysis
    pub fn fork(&self) -> DatabaseFork {
        DatabaseFork {
            parent_version: self.version,
            changes: Vec::new(),
        }
    }
    
    // ========== Info ==========
    
    /// Get database path
    pub fn path(&self) -> &str {
        &self.path
    }
    
    /// Get current version
    pub fn version(&self) -> u64 {
        self.version
    }
}

/// Forked database for counterfactual reasoning
pub struct DatabaseFork {
    parent_version: u64,
    changes: Vec<Change>,
}

#[derive(Debug, Clone)]
pub enum Change {
    Remove(String),      // Remove node by ID
    Modify(String, String, String), // Modify property: (id, key, value)
    Add(String, String), // Add node: (id, content)
}

impl DatabaseFork {
    pub fn apply(mut self, change: Change) -> Self {
        self.changes.push(change);
        self
    }
    
    pub fn propagate(self) -> PropagatedFork {
        // In a real implementation, this would trace causal chains
        PropagatedFork {
            parent_version: self.parent_version,
            changes: self.changes,
            affected: Vec::new(),
        }
    }
}

pub struct PropagatedFork {
    parent_version: u64,
    changes: Vec<Change>,
    affected: Vec<String>,
}

impl PropagatedFork {
    pub fn diff(&self) -> ForkDiff {
        ForkDiff {
            changes: self.changes.clone(),
            affected_nodes: self.affected.clone(),
            broken_chains: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub struct ForkDiff {
    pub changes: Vec<Change>,
    pub affected_nodes: Vec<String>,
    pub broken_chains: Vec<(String, String)>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_db() {
        let db = Database::memory().unwrap();
        assert_eq!(db.path(), ":memory:");
    }
    
    #[test]
    fn test_cypher_to_sql() {
        let db = Database::memory().unwrap();
        let sql = db.cypher_to_sql("MATCH (a)-[:CAUSES]->(b) RETURN b").unwrap();
        assert!(sql.contains("SELECT"));
        assert!(sql.contains("CAUSES"));
    }
    
    #[test]
    fn test_fork() {
        let db = Database::memory().unwrap();
        let diff = db.fork()
            .apply(Change::Remove("node1".into()))
            .propagate()
            .diff();
        
        assert_eq!(diff.changes.len(), 1);
    }
}
