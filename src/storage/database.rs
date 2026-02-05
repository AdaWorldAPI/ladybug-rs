//! Main Database API - unified interface for all operations
//!
//! Provides a single entry point for all LadybugDB operations:
//! - SQL queries (via DataFusion)
//! - Cypher queries (via transpilation)
//! - Vector search (via LanceDB ANN)
//! - Hamming/resonance search (via SIMD engine)
//! - Graph traversal and butterfly detection

use crate::core::{Fingerprint, HammingEngine};
use crate::cognitive::Thought;
use crate::nars::TruthValue;
use crate::graph::{Edge, Traversal};
use crate::query::{Query, QueryResult, cypher_to_sql, SqlEngine, QueryBuilder};
use crate::storage::{LanceStore, NodeRecord, EdgeRecord};
use crate::{Result, Error};

use arrow::record_batch::RecordBatch;
use std::path::Path;
use std::sync::Arc;
use parking_lot::RwLock;

/// Main database handle - unified access to all operations
pub struct Database {
    /// Path to database
    path: String,
    /// Lance storage backend
    lance: Arc<tokio::sync::RwLock<LanceStore>>,
    /// SQL execution engine
    sql_engine: Arc<tokio::sync::RwLock<SqlEngine>>,
    /// Hamming search engine (pre-indexed, in-memory)
    hamming: Arc<RwLock<HammingEngine>>,
    /// Current version (for copy-on-write)
    version: u64,
}

impl Database {
    /// Open or create a database (async)
    pub async fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        
        // Create directory if needed
        std::fs::create_dir_all(&path_str)?;
        
        // Open Lance store
        let lance = LanceStore::open(&path_str).await?;
        
        // Create SQL engine with Lance tables
        let sql_engine = SqlEngine::with_database(&path_str).await?;
        
        Ok(Self {
            path: path_str,
            lance: Arc::new(tokio::sync::RwLock::new(lance)),
            sql_engine: Arc::new(tokio::sync::RwLock::new(sql_engine)),
            hamming: Arc::new(RwLock::new(HammingEngine::new())),
            version: 0,
        })
    }
    
    /// Open synchronously (blocks on runtime)
    pub fn open_sync<P: AsRef<Path>>(path: P) -> Result<Self> {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(Self::open(path))
    }
    
    /// Connect to in-memory database
    pub fn memory() -> Self {
        Self {
            path: ":memory:".to_string(),
            lance: Arc::new(tokio::sync::RwLock::new(LanceStore::memory())),
            sql_engine: Arc::new(tokio::sync::RwLock::new(SqlEngine::default())),
            hamming: Arc::new(RwLock::new(HammingEngine::new())),
            version: 0,
        }
    }
    
    // =========================================================================
    // SQL OPERATIONS
    // =========================================================================
    
    /// Execute SQL query
    pub async fn sql(&self, query: &str) -> Result<Vec<RecordBatch>> {
        let engine = self.sql_engine.read().await;
        engine.execute(query).await
    }
    
    /// Execute SQL query with parameters
    pub async fn sql_params(
        &self,
        query: &str,
        params: &[(&str, datafusion::scalar::ScalarValue)],
    ) -> Result<Vec<RecordBatch>> {
        let engine = self.sql_engine.read().await;
        engine.execute_with_params(query, params).await
    }
    
    /// Build and execute a query
    pub async fn query(&self) -> QueryBuilder {
        QueryBuilder::from("nodes")
    }
    
    // =========================================================================
    // CYPHER OPERATIONS
    // =========================================================================
    
    /// Execute Cypher query (transpiled to SQL)
    pub async fn cypher(&self, query: &str) -> Result<Vec<RecordBatch>> {
        // Transpile Cypher to SQL
        let sql = cypher_to_sql(query)?;
        
        // Execute via SQL engine
        self.sql(&sql).await
    }
    
    // =========================================================================
    // VECTOR OPERATIONS
    // =========================================================================
    
    /// Vector similarity search (ANN)
    pub async fn vector_search(
        &self,
        embedding: &[f32],
        k: usize,
    ) -> Result<Vec<(NodeRecord, f32)>> {
        let mut lance = self.lance.write().await;
        lance.vector_search(embedding, k, None).await
    }
    
    /// Vector search with filter
    pub async fn vector_search_filtered(
        &self,
        embedding: &[f32],
        k: usize,
        filter: &str,
    ) -> Result<Vec<(NodeRecord, f32)>> {
        let mut lance = self.lance.write().await;
        lance.vector_search(embedding, k, Some(filter)).await
    }
    
    // =========================================================================
    // HAMMING/RESONANCE OPERATIONS
    // =========================================================================
    
    /// Resonance search (Hamming similarity) - in-memory indexed
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
    
    /// Resonance search over Lance storage
    pub async fn resonate_lance(
        &self,
        fingerprint: &Fingerprint,
        k: usize,
        threshold: Option<f32>,
    ) -> Result<Vec<(NodeRecord, u32, f32)>> {
        let mut lance = self.lance.write().await;
        lance.hamming_search(fingerprint, k, threshold).await
    }
    
    /// Resonate by content (auto-generates fingerprint)
    pub fn resonate_content(
        &self,
        content: &str,
        threshold: f32,
        limit: usize,
    ) -> Vec<(usize, f32)> {
        let fp = Fingerprint::from_content(content);
        self.resonate(&fp, threshold, limit)
    }
    
    /// Index fingerprints for resonance search (in-memory)
    pub fn index_fingerprints(&self, fingerprints: Vec<Fingerprint>) {
        let mut engine = self.hamming.write();
        engine.index(fingerprints);
    }
    
    // =========================================================================
    // GRAPH OPERATIONS
    // =========================================================================
    
    /// Start a graph traversal query
    pub fn traverse(&self, start_id: &str) -> Traversal {
        Traversal::from(start_id)
    }
    
    /// Detect butterfly effects (causal amplification chains)
    pub async fn detect_butterflies(
        &self,
        source_id: &str,
        threshold: f32,
        max_depth: usize,
    ) -> Result<Vec<RecordBatch>> {
        let cypher = format!(
            "MATCH (source)-[:CAUSES|AMPLIFIES*1..{}]->(target) \
             WHERE source.id = '{}' \
             RETURN target, path, amplification",
            max_depth, source_id
        );
        
        let mut sql = cypher_to_sql(&cypher)?;
        sql.push_str(&format!("\n  AND t.amplification > {}", threshold));
        
        self.sql(&sql).await
    }
    
    /// Impact analysis for a potential change
    pub async fn impact_analysis(&self, change_id: &str) -> Result<ImpactReport> {
        // Get all affected nodes
        let affected = self.cypher(&format!(
            "MATCH (source)-[:CAUSES|AMPLIFIES|ENABLES*1..10]->(affected) \
             WHERE source.id = '{}' \
             RETURN affected",
            change_id
        )).await?;
        
        // Get butterfly effects
        let butterflies = self.detect_butterflies(change_id, 5.0, 10).await?;
        
        let total_affected = affected.iter().map(|b| b.num_rows()).sum();
        let butterfly_count = butterflies.iter().map(|b| b.num_rows()).sum();
        
        Ok(ImpactReport {
            total_affected,
            butterfly_count,
            affected_batches: affected,
            butterfly_batches: butterflies,
        })
    }
    
    // =========================================================================
    // CRUD OPERATIONS
    // =========================================================================
    
    /// Add a node
    pub async fn add_node(&self, node: NodeRecord) -> Result<()> {
        let mut lance = self.lance.write().await;
        lance.insert_node(&node).await
    }
    
    /// Add multiple nodes
    pub async fn add_nodes(&self, nodes: &[NodeRecord]) -> Result<()> {
        let mut lance = self.lance.write().await;
        lance.insert_nodes(nodes).await
    }
    
    /// Add an edge
    pub async fn add_edge(&self, edge: EdgeRecord) -> Result<()> {
        let mut lance = self.lance.write().await;
        lance.insert_edge(&edge).await
    }
    
    /// Get a node by ID
    pub async fn get_node(&self, id: &str) -> Result<Option<NodeRecord>> {
        let mut lance = self.lance.write().await;
        lance.get_node(id).await
    }
    
    /// Get edges from a node
    pub async fn get_edges_from(&self, from_id: &str) -> Result<Vec<EdgeRecord>> {
        let mut lance = self.lance.write().await;
        lance.get_edges_from(from_id).await
    }
    
    /// Add a thought (convenience method)
    pub async fn add_thought(&self, thought: &Thought) -> Result<String> {
        let node = NodeRecord::new(&thought.id, "Thought")
            .with_qidx(thought.qidx)
            .with_content(&thought.content)
            .with_fingerprint(&thought.fingerprint);

        self.add_node(node).await?;
        Ok(thought.id.clone())
    }
    
    /// Create a CAUSES edge
    pub async fn causes(&self, from_id: &str, to_id: &str, amplification: f32) -> Result<()> {
        let edge = EdgeRecord::new(from_id, to_id, "CAUSES")
            .with_amplification(amplification);
        self.add_edge(edge).await
    }
    
    /// Create an ENABLES edge
    pub async fn enables(&self, from_id: &str, to_id: &str) -> Result<()> {
        let edge = EdgeRecord::new(from_id, to_id, "ENABLES");
        self.add_edge(edge).await
    }
    
    /// Create an AMPLIFIES edge
    pub async fn amplifies(&self, from_id: &str, to_id: &str, factor: f32) -> Result<()> {
        let edge = EdgeRecord::new(from_id, to_id, "AMPLIFIES")
            .with_amplification(factor);
        self.add_edge(edge).await
    }
    
    // =========================================================================
    // COUNTERFACTUAL OPERATIONS
    // =========================================================================
    
    /// Fork database for counterfactual reasoning
    pub fn fork(&self) -> Database {
        Database {
            path: self.path.clone(),
            lance: Arc::clone(&self.lance),
            sql_engine: Arc::clone(&self.sql_engine),
            hamming: Arc::clone(&self.hamming),
            version: self.version + 1,
        }
    }
    
    // =========================================================================
    // DATABASE INFO
    // =========================================================================
    
    /// Database path
    pub fn path(&self) -> &str {
        &self.path
    }
    
    /// Current version
    pub fn version(&self) -> u64 {
        self.version
    }
    
    /// Number of indexed fingerprints (in-memory)
    pub fn fingerprint_count(&self) -> usize {
        self.hamming.read().len()
    }
}

/// Impact analysis report
#[derive(Debug)]
pub struct ImpactReport {
    pub total_affected: usize,
    pub butterfly_count: usize,
    pub affected_batches: Vec<RecordBatch>,
    pub butterfly_batches: Vec<RecordBatch>,
}

// Convenience function
pub fn open<P: AsRef<Path>>(path: P) -> Result<Database> {
    Database::open_sync(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_open_memory() {
        let db = Database::memory();
        assert_eq!(db.path(), ":memory:");
    }
    
    #[test]
    fn test_resonate() {
        let db = Database::memory();
        
        // Index some fingerprints
        let fps: Vec<Fingerprint> = (0..100)
            .map(|i| Fingerprint::from_content(&format!("thought_{}", i)))
            .collect();
        db.index_fingerprints(fps);
        
        // Search
        let query = Fingerprint::from_content("thought_50");
        let results = db.resonate(&query, 0.5, 10);
        
        // Should find exact match with similarity 1.0
        assert!(!results.is_empty());
        assert!(results[0].1 > 0.99);
    }
    
    #[test]
    fn test_fork() {
        let db = Database::memory();
        let forked = db.fork();
        
        assert_eq!(forked.version(), db.version() + 1);
    }
    
    #[tokio::test]
    async fn test_cypher_transpile() {
        let cypher = "MATCH (a:Thought)-[:CAUSES]->(b:Thought) RETURN b";
        let sql = cypher_to_sql(cypher).unwrap();
        
        assert!(sql.contains("SELECT"));
        assert!(sql.contains("JOIN edges"));
    }
}
