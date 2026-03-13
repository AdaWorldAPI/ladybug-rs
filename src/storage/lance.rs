//! LanceDB Storage Substrate (lancedb SDK 0.26)
//!
//! Provides the persistent storage layer using LanceDB.
//! All data (nodes, edges, fingerprints) stored in LanceDB tables
//! with native vector/Hamming index support.
//!
//! Versioning comes for free: every write creates a new version.
//! Time travel: table.checkout(version_n) reads any previous state.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     LANCEDB SUBSTRATE                            │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │   nodes table     → id, label, fingerprint, embedding, props    │
//! │   edges table     → from_id, to_id, type, weight, amplification │
//! │   sessions table  → session state, consciousness snapshots      │
//! │                                                                  │
//! │   Indices:                                                       │
//! │     - IVF-PQ on embedding (vector ANN)                          │
//! │     - Scalar on label, type (filtering)                         │
//! │     - Custom Hamming index on fingerprint                       │
//! │                                                                  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use arrow::array::*;
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema, TimeUnit};
use arrow::record_batch::RecordBatch;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::table::AddDataMode;
use std::path::Path;
use std::sync::Arc;

use crate::core::Fingerprint;
use crate::{Error, Result};

/// Fingerprint size in bytes (16,384 bits = 2,048 bytes)
pub const FINGERPRINT_BYTES: usize = 2048;

/// Jina embedding dimension
pub const EMBEDDING_DIM: usize = 1024;

/// Thinking style vector dimension (7 axes)
pub const THINKING_STYLE_DIM: usize = 7;

// =============================================================================
// SCHEMA DEFINITIONS
// =============================================================================

/// Create the nodes table schema
pub fn nodes_schema() -> ArrowSchema {
    ArrowSchema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("label", DataType::Utf8, false),
        Field::new(
            "fingerprint",
            DataType::FixedSizeBinary(FINGERPRINT_BYTES as i32),
            true,
        ),
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                EMBEDDING_DIM as i32,
            ),
            true,
        ),
        Field::new("qidx", DataType::UInt8, false),
        Field::new(
            "thinking_style",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                THINKING_STYLE_DIM as i32,
            ),
            true,
        ),
        Field::new("content", DataType::Utf8, true),
        Field::new("properties", DataType::Utf8, true), // JSON
        Field::new(
            "created_at",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("version", DataType::Int64, false),
    ])
}

/// Create the edges table schema
pub fn edges_schema() -> ArrowSchema {
    ArrowSchema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("from_id", DataType::Utf8, false),
        Field::new("to_id", DataType::Utf8, false),
        Field::new("type", DataType::Utf8, false),
        Field::new("weight", DataType::Float32, false),
        Field::new("amplification", DataType::Float32, false),
        Field::new("properties", DataType::Utf8, true), // JSON
        Field::new(
            "created_at",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
    ])
}

/// Create the sessions table schema (for consciousness snapshots)
pub fn sessions_schema() -> ArrowSchema {
    ArrowSchema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, true),
        Field::new(
            "thinking_style",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                THINKING_STYLE_DIM as i32,
            ),
            true,
        ),
        Field::new("coherence", DataType::Float32, false),
        Field::new("ice_cake_layers", DataType::Int32, false),
        Field::new("state", DataType::Utf8, true), // JSON blob
        Field::new(
            "created_at",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new(
            "updated_at",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
    ])
}

// =============================================================================
// HELPERS
// =============================================================================

/// Wrap a single RecordBatch as a RecordBatchReader for lancedb APIs.
fn batch_reader(
    batch: RecordBatch,
) -> RecordBatchIterator<
    std::vec::IntoIter<std::result::Result<RecordBatch, arrow::error::ArrowError>>,
> {
    let schema = batch.schema();
    RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema)
}

// =============================================================================
// LANCE STORE
// =============================================================================

/// LanceDB-backed storage for LadybugDB.
///
/// Uses the lancedb SDK which wraps Lance 2.x with automatic versioning,
/// time travel, and compaction.
pub struct LanceStore {
    /// LanceDB connection (manages all tables in a directory)
    db: lancedb::Connection,
}

impl LanceStore {
    /// Open or create a LanceDB store at the given path.
    pub async fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        // Create directory if needed
        std::fs::create_dir_all(&path_str)?;

        let db = lancedb::connect(&path_str)
            .execute()
            .await
            .map_err(|e| Error::Storage(format!("lancedb connect: {}", e)))?;

        Ok(Self { db })
    }

    /// Create in-memory store (for testing).
    pub async fn memory() -> Result<Self> {
        let db = lancedb::connect("memory://ladybug")
            .execute()
            .await
            .map_err(|e| Error::Storage(format!("lancedb memory connect: {}", e)))?;
        Ok(Self { db })
    }

    // -------------------------------------------------------------------------
    // TABLE MANAGEMENT
    // -------------------------------------------------------------------------

    /// Get or create the nodes table.
    async fn nodes_table(&self) -> Result<lancedb::Table> {
        match self.db.open_table("nodes").execute().await {
            Ok(table) => Ok(table),
            Err(lancedb::Error::TableNotFound { .. }) => {
                let schema = Arc::new(nodes_schema());
                let batch = RecordBatch::new_empty(schema.clone());
                let table = self
                    .db
                    .create_table("nodes", batch_reader(batch))
                    .execute()
                    .await
                    .map_err(|e| Error::Storage(format!("create nodes table: {}", e)))?;
                Ok(table)
            }
            Err(e) => Err(Error::Storage(format!("open nodes table: {}", e))),
        }
    }

    /// Get or create the edges table.
    async fn edges_table(&self) -> Result<lancedb::Table> {
        match self.db.open_table("edges").execute().await {
            Ok(table) => Ok(table),
            Err(lancedb::Error::TableNotFound { .. }) => {
                let schema = Arc::new(edges_schema());
                let batch = RecordBatch::new_empty(schema.clone());
                let table = self
                    .db
                    .create_table("edges", batch_reader(batch))
                    .execute()
                    .await
                    .map_err(|e| Error::Storage(format!("create edges table: {}", e)))?;
                Ok(table)
            }
            Err(e) => Err(Error::Storage(format!("open edges table: {}", e))),
        }
    }

    /// Get or create the sessions table.
    async fn sessions_table(&self) -> Result<lancedb::Table> {
        match self.db.open_table("sessions").execute().await {
            Ok(table) => Ok(table),
            Err(lancedb::Error::TableNotFound { .. }) => {
                let schema = Arc::new(sessions_schema());
                let batch = RecordBatch::new_empty(schema.clone());
                let table = self
                    .db
                    .create_table("sessions", batch_reader(batch))
                    .execute()
                    .await
                    .map_err(|e| Error::Storage(format!("create sessions table: {}", e)))?;
                Ok(table)
            }
            Err(e) => Err(Error::Storage(format!("open sessions table: {}", e))),
        }
    }

    // -------------------------------------------------------------------------
    // NODE OPERATIONS
    // -------------------------------------------------------------------------

    /// Insert a node. Creates a new version automatically.
    pub async fn insert_node(&self, node: &NodeRecord) -> Result<()> {
        let table = self.nodes_table().await?;
        let batch = node.to_record_batch()?;
        table
            .add(batch_reader(batch))
            .mode(AddDataMode::Append)
            .execute()
            .await
            .map_err(|e| Error::Storage(format!("insert_node: {}", e)))?;
        Ok(())
    }

    /// Insert multiple nodes. Creates a new version automatically.
    pub async fn insert_nodes(&self, nodes: &[NodeRecord]) -> Result<()> {
        if nodes.is_empty() {
            return Ok(());
        }
        let table = self.nodes_table().await?;
        let batch = NodeRecord::batch_to_record_batch(nodes)?;
        table
            .add(batch_reader(batch))
            .mode(AddDataMode::Append)
            .execute()
            .await
            .map_err(|e| Error::Storage(format!("insert_nodes: {}", e)))?;
        Ok(())
    }

    /// Get a node by ID.
    pub async fn get_node(&self, id: &str) -> Result<Option<NodeRecord>> {
        let table = self.nodes_table().await?;

        let results = table
            .query()
            .only_if(format!("id = '{}'", id))
            .execute()
            .await
            .map_err(|e| Error::Storage(format!("get_node query: {}", e)))?;

        use futures::StreamExt;
        let mut stream = results;
        while let Some(batch) = stream.next().await {
            let batch = batch.map_err(|e| Error::Storage(format!("get_node batch: {}", e)))?;
            if batch.num_rows() > 0 {
                return Ok(Some(NodeRecord::from_record_batch(&batch, 0)?));
            }
        }

        Ok(None)
    }

    // -------------------------------------------------------------------------
    // EDGE OPERATIONS
    // -------------------------------------------------------------------------

    /// Insert an edge. Creates a new version automatically.
    pub async fn insert_edge(&self, edge: &EdgeRecord) -> Result<()> {
        let table = self.edges_table().await?;
        let batch = edge.to_record_batch()?;
        table
            .add(batch_reader(batch))
            .mode(AddDataMode::Append)
            .execute()
            .await
            .map_err(|e| Error::Storage(format!("insert_edge: {}", e)))?;
        Ok(())
    }

    /// Get edges from a node.
    pub async fn get_edges_from(&self, from_id: &str) -> Result<Vec<EdgeRecord>> {
        let table = self.edges_table().await?;

        let results = table
            .query()
            .only_if(format!("from_id = '{}'", from_id))
            .execute()
            .await
            .map_err(|e| Error::Storage(format!("get_edges_from query: {}", e)))?;

        use futures::StreamExt;
        let mut records = Vec::new();
        let mut stream = results;
        while let Some(batch) = stream.next().await {
            let batch =
                batch.map_err(|e| Error::Storage(format!("get_edges_from batch: {}", e)))?;
            for i in 0..batch.num_rows() {
                records.push(EdgeRecord::from_record_batch(&batch, i)?);
            }
        }

        Ok(records)
    }

    /// Get edges to a node.
    pub async fn get_edges_to(&self, to_id: &str) -> Result<Vec<EdgeRecord>> {
        let table = self.edges_table().await?;

        let results = table
            .query()
            .only_if(format!("to_id = '{}'", to_id))
            .execute()
            .await
            .map_err(|e| Error::Storage(format!("get_edges_to query: {}", e)))?;

        use futures::StreamExt;
        let mut records = Vec::new();
        let mut stream = results;
        while let Some(batch) = stream.next().await {
            let batch =
                batch.map_err(|e| Error::Storage(format!("get_edges_to batch: {}", e)))?;
            for i in 0..batch.num_rows() {
                records.push(EdgeRecord::from_record_batch(&batch, i)?);
            }
        }

        Ok(records)
    }

    // -------------------------------------------------------------------------
    // VECTOR SEARCH
    // -------------------------------------------------------------------------

    /// Vector similarity search using LanceDB native ANN.
    ///
    /// LanceDB automatically manages IVF-PQ indices.
    pub async fn vector_search(
        &self,
        embedding: &[f32],
        k: usize,
        filter: Option<&str>,
    ) -> Result<Vec<(NodeRecord, f32)>> {
        let table = self.nodes_table().await?;

        let mut query = table
            .vector_search(embedding)
            .map_err(|e| Error::Storage(format!("vector_search setup: {}", e)))?
            .limit(k);

        if let Some(f) = filter {
            query = query.only_if(f);
        }

        let results = query
            .execute()
            .await
            .map_err(|e| Error::Storage(format!("vector_search execute: {}", e)))?;

        use futures::StreamExt;
        let mut nodes = Vec::new();
        let mut stream = results;
        while let Some(batch) = stream.next().await {
            let batch =
                batch.map_err(|e| Error::Storage(format!("vector_search batch: {}", e)))?;
            let distances = batch
                .column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

            for i in 0..batch.num_rows() {
                let node = NodeRecord::from_record_batch(&batch, i)?;
                let dist = distances.map(|d| d.value(i)).unwrap_or(0.0);
                nodes.push((node, dist));
            }
        }

        Ok(nodes)
    }

    // -------------------------------------------------------------------------
    // HAMMING SEARCH (Fingerprint similarity)
    // -------------------------------------------------------------------------

    /// Fingerprint similarity search using Hamming distance.
    pub async fn hamming_search(
        &self,
        query_fp: &Fingerprint,
        k: usize,
        threshold: Option<f32>,
    ) -> Result<Vec<(NodeRecord, u32, f32)>> {
        let table = self.nodes_table().await?;

        let results = table
            .query()
            .select(lancedb::query::Select::Columns(vec![
                "id".into(),
                "label".into(),
                "fingerprint".into(),
                "qidx".into(),
                "content".into(),
                "properties".into(),
                "created_at".into(),
                "version".into(),
            ]))
            .only_if("fingerprint IS NOT NULL")
            .execute()
            .await
            .map_err(|e| Error::Storage(format!("hamming_search query: {}", e)))?;

        use futures::StreamExt;
        let mut candidates: Vec<(NodeRecord, u32)> = Vec::new();
        let mut stream = results;

        while let Some(batch) = stream.next().await {
            let batch =
                batch.map_err(|e| Error::Storage(format!("hamming_search batch: {}", e)))?;
            let fp_col = batch
                .column_by_name("fingerprint")
                .unwrap()
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .unwrap();

            for i in 0..batch.num_rows() {
                if fp_col.is_null(i) {
                    continue;
                }

                let fp_bytes = fp_col.value(i);
                let candidate_fp = Fingerprint::from_bytes(fp_bytes)?;
                let distance = query_fp.hamming(&candidate_fp);

                let node = NodeRecord::from_record_batch(&batch, i)?;
                candidates.push((node, distance));
            }
        }

        // Sort by distance
        candidates.sort_by_key(|(_, d)| *d);

        // Apply threshold and limit
        let max_distance =
            threshold.map(|t| ((1.0 - t) * crate::FINGERPRINT_BITS as f32) as u32);

        let results: Vec<(NodeRecord, u32, f32)> = candidates
            .into_iter()
            .filter(|(_, d)| max_distance.map(|m| *d <= m).unwrap_or(true))
            .take(k)
            .map(|(node, dist)| {
                let similarity = 1.0 - (dist as f32 / crate::FINGERPRINT_BITS as f32);
                (node, dist, similarity)
            })
            .collect();

        Ok(results)
    }

    // -------------------------------------------------------------------------
    // VERSIONING (lancedb gives us this for free)
    // -------------------------------------------------------------------------

    /// Get the current version of the nodes table.
    pub async fn nodes_version(&self) -> Result<u64> {
        let table = self.nodes_table().await?;
        table
            .version()
            .await
            .map_err(|e| Error::Storage(format!("nodes_version: {}", e)))
    }

    // -------------------------------------------------------------------------
    // SQL
    // -------------------------------------------------------------------------

    /// Execute raw SQL via DataFusion (delegated to query module).
    pub async fn sql(&self, _query: &str) -> Result<RecordBatch> {
        todo!("Delegate to DataFusion execution engine")
    }
}

// =============================================================================
// DATA RECORDS
// =============================================================================

/// Node record for insert/query operations
#[derive(Debug, Clone)]
pub struct NodeRecord {
    pub id: String,
    pub label: String,
    pub fingerprint: Option<Vec<u8>>,
    pub embedding: Option<Vec<f32>>,
    pub qidx: u8,
    pub thinking_style: Option<Vec<f32>>,
    pub content: Option<String>,
    pub properties: Option<String>,
    pub created_at: i64, // microseconds since epoch
    pub version: i64,
}

impl NodeRecord {
    /// Create a new node record
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            fingerprint: None,
            embedding: None,
            qidx: 128,
            thinking_style: None,
            content: None,
            properties: None,
            created_at: chrono::Utc::now().timestamp_micros(),
            version: 1,
        }
    }

    /// Set fingerprint
    pub fn with_fingerprint(mut self, fp: &Fingerprint) -> Self {
        self.fingerprint = Some(fp.to_bytes().to_vec());
        self
    }

    /// Set embedding
    pub fn with_embedding(mut self, emb: Vec<f32>) -> Self {
        self.embedding = Some(emb);
        self
    }

    /// Set qualia index
    pub fn with_qidx(mut self, qidx: u8) -> Self {
        self.qidx = qidx;
        self
    }

    /// Set thinking style
    pub fn with_thinking_style(mut self, style: Vec<f32>) -> Self {
        self.thinking_style = Some(style);
        self
    }

    /// Set content
    pub fn with_content(mut self, content: impl Into<String>) -> Self {
        self.content = Some(content.into());
        self
    }

    /// Set properties (JSON)
    pub fn with_properties(mut self, props: impl Into<String>) -> Self {
        self.properties = Some(props.into());
        self
    }

    /// Convert to Arrow RecordBatch
    pub fn to_record_batch(&self) -> Result<RecordBatch> {
        Self::batch_to_record_batch(std::slice::from_ref(self))
    }

    /// Convert multiple nodes to Arrow RecordBatch
    pub fn batch_to_record_batch(nodes: &[Self]) -> Result<RecordBatch> {
        let schema = Arc::new(nodes_schema());

        let ids: StringArray = nodes.iter().map(|n| Some(n.id.as_str())).collect();
        let labels: StringArray = nodes.iter().map(|n| Some(n.label.as_str())).collect();

        // Fingerprints
        let mut fp_array = FixedSizeBinaryBuilder::new(FINGERPRINT_BYTES as i32);
        for node in nodes {
            if let Some(ref fp) = node.fingerprint {
                fp_array.append_value(fp)?;
            } else {
                fp_array.append_null();
            }
        }
        let fingerprints = fp_array.finish();

        // Embeddings (FixedSizeList of Float32)
        let embedding_values: Vec<Option<Vec<f32>>> =
            nodes.iter().map(|n| n.embedding.clone()).collect();
        let embeddings = create_fixed_size_list_f32(&embedding_values, EMBEDDING_DIM)?;

        let qidxs: UInt8Array = nodes.iter().map(|n| Some(n.qidx)).collect();

        // Thinking styles
        let style_values: Vec<Option<Vec<f32>>> =
            nodes.iter().map(|n| n.thinking_style.clone()).collect();
        let thinking_styles = create_fixed_size_list_f32(&style_values, THINKING_STYLE_DIM)?;

        let contents: StringArray = nodes.iter().map(|n| n.content.as_deref()).collect();
        let properties: StringArray = nodes.iter().map(|n| n.properties.as_deref()).collect();
        let created_ats: TimestampMicrosecondArray =
            nodes.iter().map(|n| Some(n.created_at)).collect();
        let versions: Int64Array = nodes.iter().map(|n| Some(n.version)).collect();

        Ok(RecordBatch::try_new(
            schema,
            vec![
                Arc::new(ids),
                Arc::new(labels),
                Arc::new(fingerprints),
                Arc::new(embeddings),
                Arc::new(qidxs),
                Arc::new(thinking_styles),
                Arc::new(contents),
                Arc::new(properties),
                Arc::new(created_ats),
                Arc::new(versions),
            ],
        )?)
    }

    /// Extract from RecordBatch at given row index
    pub fn from_record_batch(batch: &RecordBatch, row: usize) -> Result<Self> {
        let id = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(row)
            .to_string();
        let label = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(row)
            .to_string();

        let fp_col = batch
            .column(2)
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .unwrap();
        let fingerprint = if fp_col.is_null(row) {
            None
        } else {
            Some(fp_col.value(row).to_vec())
        };

        let qidx = batch
            .column(4)
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap()
            .value(row);

        let content_col = batch
            .column(6)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let content = if content_col.is_null(row) {
            None
        } else {
            Some(content_col.value(row).to_string())
        };

        let props_col = batch
            .column(7)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let properties = if props_col.is_null(row) {
            None
        } else {
            Some(props_col.value(row).to_string())
        };

        let created_at = batch
            .column(8)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap()
            .value(row);
        let version = batch
            .column(9)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(row);

        Ok(Self {
            id,
            label,
            fingerprint,
            embedding: None, // TODO: extract from FixedSizeList
            qidx,
            thinking_style: None, // TODO: extract from FixedSizeList
            content,
            properties,
            created_at,
            version,
        })
    }
}

/// Edge record for insert/query operations
#[derive(Debug, Clone)]
pub struct EdgeRecord {
    pub id: String,
    pub from_id: String,
    pub to_id: String,
    pub edge_type: String,
    pub weight: f32,
    pub amplification: f32,
    pub properties: Option<String>,
    pub created_at: i64,
}

impl EdgeRecord {
    /// Create a new edge
    pub fn new(
        from_id: impl Into<String>,
        to_id: impl Into<String>,
        edge_type: impl Into<String>,
    ) -> Self {
        let from = from_id.into();
        let to = to_id.into();
        let etype = edge_type.into();
        Self {
            id: format!("{}->{}:{}", from, to, etype),
            from_id: from,
            to_id: to,
            edge_type: etype,
            weight: 1.0,
            amplification: 1.0,
            properties: None,
            created_at: chrono::Utc::now().timestamp_micros(),
        }
    }

    /// Set weight
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Set amplification factor
    pub fn with_amplification(mut self, amp: f32) -> Self {
        self.amplification = amp;
        self
    }

    /// Convert to RecordBatch
    pub fn to_record_batch(&self) -> Result<RecordBatch> {
        let schema = Arc::new(edges_schema());

        let ids: StringArray = [Some(self.id.as_str())].into_iter().collect();
        let from_ids: StringArray = [Some(self.from_id.as_str())].into_iter().collect();
        let to_ids: StringArray = [Some(self.to_id.as_str())].into_iter().collect();
        let types: StringArray = [Some(self.edge_type.as_str())].into_iter().collect();
        let weights: Float32Array = [Some(self.weight)].into_iter().collect();
        let amplifications: Float32Array = [Some(self.amplification)].into_iter().collect();
        let properties: StringArray = [self.properties.as_deref()].into_iter().collect();
        let created_ats: TimestampMicrosecondArray =
            [Some(self.created_at)].into_iter().collect();

        Ok(RecordBatch::try_new(
            schema,
            vec![
                Arc::new(ids),
                Arc::new(from_ids),
                Arc::new(to_ids),
                Arc::new(types),
                Arc::new(weights),
                Arc::new(amplifications),
                Arc::new(properties),
                Arc::new(created_ats),
            ],
        )?)
    }

    /// Extract from RecordBatch
    pub fn from_record_batch(batch: &RecordBatch, row: usize) -> Result<Self> {
        Ok(Self {
            id: batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value(row)
                .to_string(),
            from_id: batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value(row)
                .to_string(),
            to_id: batch
                .column(2)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value(row)
                .to_string(),
            edge_type: batch
                .column(3)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value(row)
                .to_string(),
            weight: batch
                .column(4)
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .value(row),
            amplification: batch
                .column(5)
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .value(row),
            properties: {
                let col = batch
                    .column(6)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();
                if col.is_null(row) {
                    None
                } else {
                    Some(col.value(row).to_string())
                }
            },
            created_at: batch
                .column(7)
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap()
                .value(row),
        })
    }
}

// =============================================================================
// HELPERS
// =============================================================================

/// Create a FixedSizeList<Float32> array from optional vectors
fn create_fixed_size_list_f32(
    values: &[Option<Vec<f32>>],
    size: usize,
) -> Result<FixedSizeListArray> {
    let inner_field = Field::new("item", DataType::Float32, false);
    let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), size as i32)
        .with_field(Arc::new(inner_field));

    for val in values {
        if let Some(v) = val {
            if v.len() != size {
                return Err(Error::Storage(format!(
                    "Expected {} elements, got {}",
                    size,
                    v.len()
                )));
            }
            for &f in v {
                builder.values().append_value(f);
            }
            builder.append(true);
        } else {
            for _ in 0..size {
                builder.values().append_null();
            }
            builder.append(false);
        }
    }

    Ok(builder.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_node_record_roundtrip() {
        let node = NodeRecord::new("test-1", "Thought")
            .with_qidx(42)
            .with_content("Hello world");

        let batch = node.to_record_batch().unwrap();
        assert_eq!(batch.num_rows(), 1);

        let recovered = NodeRecord::from_record_batch(&batch, 0).unwrap();
        assert_eq!(recovered.id, "test-1");
        assert_eq!(recovered.label, "Thought");
        assert_eq!(recovered.qidx, 42);
        assert_eq!(recovered.content, Some("Hello world".to_string()));
    }

    #[tokio::test]
    async fn test_edge_record_roundtrip() {
        let edge = EdgeRecord::new("a", "b", "CAUSES")
            .with_weight(0.8)
            .with_amplification(1.5);

        let batch = edge.to_record_batch().unwrap();
        let recovered = EdgeRecord::from_record_batch(&batch, 0).unwrap();

        assert_eq!(recovered.from_id, "a");
        assert_eq!(recovered.to_id, "b");
        assert_eq!(recovered.edge_type, "CAUSES");
        assert_eq!(recovered.weight, 0.8);
        assert_eq!(recovered.amplification, 1.5);
    }
}
