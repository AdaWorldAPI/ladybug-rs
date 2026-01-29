//! LanceDB Storage Substrate
//!
//! Provides the persistent storage layer using Lance columnar format.
//! All data (thoughts, edges, fingerprints) stored in Lance tables
//! with native vector/Hamming index support.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, BinaryArray, Float32Array, Int64Array, StringArray,
    UInt64Array, RecordBatch, FixedSizeBinaryArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use lance::dataset::{Dataset, WriteParams, WriteMode};
use lance::index::vector::{VectorIndexParams, IvfPqIndexParams};

use crate::{Result, Error};
use crate::core::Fingerprint;

/// Schema version for migrations
const SCHEMA_VERSION: u32 = 1;

/// Fingerprint size in bytes (10K bits = 1250 bytes)
pub const FINGERPRINT_BYTES: usize = 1250;

/// Lance storage handle
pub struct LanceStore {
    path: String,
    thoughts: Option<Dataset>,
    edges: Option<Dataset>,
    fingerprints: Option<Dataset>,
}

impl LanceStore {
    /// Open or create Lance storage at path
    pub async fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        std::fs::create_dir_all(&path_str)?;
        
        let mut store = Self {
            path: path_str.clone(),
            thoughts: None,
            edges: None,
            fingerprints: None,
        };
        
        // Open or create tables
        store.thoughts = store.open_or_create_table("thoughts", Self::thoughts_schema()).await?;
        store.edges = store.open_or_create_table("edges", Self::edges_schema()).await?;
        store.fingerprints = store.open_or_create_table("fingerprints", Self::fingerprints_schema()).await?;
        
        Ok(store)
    }
    
    /// In-memory store for testing
    pub fn memory() -> Self {
        Self {
            path: ":memory:".to_string(),
            thoughts: None,
            edges: None,
            fingerprints: None,
        }
    }
    
    // === Schema Definitions ===
    
    fn thoughts_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new("fingerprint", DataType::FixedSizeBinary(FINGERPRINT_BYTES as i32), false),
            Field::new("embedding", DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                1536,  // OpenAI embedding dimension
            ), true),
            Field::new("frequency", DataType::Float32, false),
            Field::new("confidence", DataType::Float32, false),
            Field::new("created_at", DataType::Int64, false),
            Field::new("updated_at", DataType::Int64, false),
            Field::new("style", DataType::Utf8, true),  // ThinkingStyle
            Field::new("layer", DataType::Int32, true), // Consciousness layer (0-6)
            Field::new("metadata", DataType::Utf8, true), // JSON
        ]))
    }
    
    fn edges_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("source_id", DataType::Utf8, false),
            Field::new("target_id", DataType::Utf8, false),
            Field::new("relation", DataType::Utf8, false),  // CAUSES, SUPPORTS, BECOMES, etc.
            Field::new("frequency", DataType::Float32, false),
            Field::new("confidence", DataType::Float32, false),
            Field::new("weight", DataType::Float32, true),
            Field::new("created_at", DataType::Int64, false),
            Field::new("metadata", DataType::Utf8, true), // JSON
        ]))
    }
    
    fn fingerprints_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("fingerprint", DataType::FixedSizeBinary(FINGERPRINT_BYTES as i32), false),
            Field::new("source_type", DataType::Utf8, false),  // "thought", "concept", "style"
            Field::new("source_id", DataType::Utf8, false),
        ]))
    }
    
    async fn open_or_create_table(
        &self,
        name: &str,
        schema: Arc<Schema>,
    ) -> Result<Option<Dataset>> {
        if self.path == ":memory:" {
            return Ok(None);
        }
        
        let table_path = format!("{}/{}.lance", self.path, name);
        
        if Path::new(&table_path).exists() {
            let dataset = Dataset::open(&table_path).await
                .map_err(|e| Error::Storage(format!("Failed to open {}: {}", name, e)))?;
            Ok(Some(dataset))
        } else {
            // Create empty dataset with schema
            let batch = RecordBatch::new_empty(schema);
            let dataset = Dataset::write(
                vec![batch].into_iter().map(Ok),
                &table_path,
                Some(WriteParams {
                    mode: WriteMode::Create,
                    ..Default::default()
                }),
            ).await.map_err(|e| Error::Storage(format!("Failed to create {}: {}", name, e)))?;
            Ok(Some(dataset))
        }
    }
    
    // === Thought Operations ===
    
    /// Insert a thought
    pub async fn insert_thought(
        &mut self,
        id: &str,
        content: &str,
        fingerprint: &Fingerprint,
        embedding: Option<&[f32]>,
        frequency: f32,
        confidence: f32,
    ) -> Result<()> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;
        
        let schema = Self::thoughts_schema();
        
        let id_array = StringArray::from(vec![id]);
        let content_array = StringArray::from(vec![content]);
        let fp_array = FixedSizeBinaryArray::try_from_iter(
            vec![Some(fingerprint.as_bytes())]
        ).map_err(|e| Error::Storage(e.to_string()))?;
        
        // Embedding (nullable)
        let embedding_array: ArrayRef = if let Some(emb) = embedding {
            Arc::new(arrow::array::FixedSizeListArray::try_new(
                Arc::new(Field::new("item", DataType::Float32, false)),
                1536,
                Arc::new(Float32Array::from(emb.to_vec())),
                None,
            ).map_err(|e| Error::Storage(e.to_string()))?)
        } else {
            Arc::new(arrow::array::NullArray::new(1))
        };
        
        let freq_array = Float32Array::from(vec![frequency]);
        let conf_array = Float32Array::from(vec![confidence]);
        let created_array = Int64Array::from(vec![now]);
        let updated_array = Int64Array::from(vec![now]);
        let style_array = StringArray::from(vec![None::<&str>]);
        let layer_array = arrow::array::Int32Array::from(vec![None::<i32>]);
        let meta_array = StringArray::from(vec![None::<&str>]);
        
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(id_array),
            Arc::new(content_array),
            Arc::new(fp_array),
            embedding_array,
            Arc::new(freq_array),
            Arc::new(conf_array),
            Arc::new(created_array),
            Arc::new(updated_array),
            Arc::new(style_array),
            Arc::new(layer_array),
            Arc::new(meta_array),
        ]).map_err(|e| Error::Storage(e.to_string()))?;
        
        if let Some(ref mut dataset) = self.thoughts {
            // Append to existing dataset
            let table_path = format!("{}/thoughts.lance", self.path);
            *dataset = Dataset::write(
                vec![batch].into_iter().map(Ok),
                &table_path,
                Some(WriteParams {
                    mode: WriteMode::Append,
                    ..Default::default()
                }),
            ).await.map_err(|e| Error::Storage(e.to_string()))?;
        }
        
        Ok(())
    }
    
    /// Query thoughts by SQL
    pub async fn query_thoughts(&self, filter: &str) -> Result<Vec<ThoughtRow>> {
        let Some(ref dataset) = self.thoughts else {
            return Ok(vec![]);
        };
        
        let scanner = dataset.scan()
            .filter(filter)
            .map_err(|e| Error::Storage(e.to_string()))?;
        
        let batches: Vec<RecordBatch> = scanner.try_into_stream()
            .await
            .map_err(|e| Error::Storage(e.to_string()))?
            .try_collect()
            .await
            .map_err(|e| Error::Storage(e.to_string()))?;
        
        let mut rows = Vec::new();
        for batch in batches {
            for i in 0..batch.num_rows() {
                rows.push(ThoughtRow::from_batch(&batch, i)?);
            }
        }
        
        Ok(rows)
    }
    
    // === Edge Operations ===
    
    /// Insert an edge
    pub async fn insert_edge(
        &mut self,
        id: &str,
        source_id: &str,
        target_id: &str,
        relation: &str,
        frequency: f32,
        confidence: f32,
        weight: Option<f32>,
    ) -> Result<()> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;
        
        let schema = Self::edges_schema();
        
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec![id])),
            Arc::new(StringArray::from(vec![source_id])),
            Arc::new(StringArray::from(vec![target_id])),
            Arc::new(StringArray::from(vec![relation])),
            Arc::new(Float32Array::from(vec![frequency])),
            Arc::new(Float32Array::from(vec![confidence])),
            Arc::new(Float32Array::from(vec![weight])),
            Arc::new(Int64Array::from(vec![now])),
            Arc::new(StringArray::from(vec![None::<&str>])),
        ]).map_err(|e| Error::Storage(e.to_string()))?;
        
        if let Some(ref mut dataset) = self.edges {
            let table_path = format!("{}/edges.lance", self.path);
            *dataset = Dataset::write(
                vec![batch].into_iter().map(Ok),
                &table_path,
                Some(WriteParams {
                    mode: WriteMode::Append,
                    ..Default::default()
                }),
            ).await.map_err(|e| Error::Storage(e.to_string()))?;
        }
        
        Ok(())
    }
    
    /// Get edges from a source node
    pub async fn get_edges_from(&self, source_id: &str) -> Result<Vec<EdgeRow>> {
        let Some(ref dataset) = self.edges else {
            return Ok(vec![]);
        };
        
        let filter = format!("source_id = '{}'", source_id);
        let scanner = dataset.scan()
            .filter(&filter)
            .map_err(|e| Error::Storage(e.to_string()))?;
        
        let batches: Vec<RecordBatch> = scanner.try_into_stream()
            .await
            .map_err(|e| Error::Storage(e.to_string()))?
            .try_collect()
            .await
            .map_err(|e| Error::Storage(e.to_string()))?;
        
        let mut rows = Vec::new();
        for batch in batches {
            for i in 0..batch.num_rows() {
                rows.push(EdgeRow::from_batch(&batch, i)?);
            }
        }
        
        Ok(rows)
    }
    
    // === Fingerprint Index ===
    
    /// Build Hamming index on fingerprints
    pub async fn build_hamming_index(&mut self) -> Result<()> {
        // Lance doesn't natively support Hamming, so we store fingerprints
        // in a table and use batch scan + SIMD comparison
        // For large scale, consider IVF clustering by Hamming prefix
        Ok(())
    }
    
    /// Scan all fingerprints (for Hamming search)
    pub async fn scan_fingerprints(&self) -> Result<Vec<(String, Vec<u8>)>> {
        let Some(ref dataset) = self.fingerprints else {
            return Ok(vec![]);
        };
        
        let scanner = dataset.scan();
        let batches: Vec<RecordBatch> = scanner.try_into_stream()
            .await
            .map_err(|e| Error::Storage(e.to_string()))?
            .try_collect()
            .await
            .map_err(|e| Error::Storage(e.to_string()))?;
        
        let mut results = Vec::new();
        for batch in batches {
            let id_col = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
            let fp_col = batch.column(1).as_any().downcast_ref::<FixedSizeBinaryArray>().unwrap();
            
            for i in 0..batch.num_rows() {
                let id = id_col.value(i).to_string();
                let fp = fp_col.value(i).to_vec();
                results.push((id, fp));
            }
        }
        
        Ok(results)
    }
    
    // === Vector Index ===
    
    /// Create IVF-PQ vector index on embeddings
    pub async fn create_vector_index(&mut self) -> Result<()> {
        let Some(ref mut dataset) = self.thoughts else {
            return Ok(());
        };
        
        let params = VectorIndexParams::with_ivf_pq(
            IvfPqIndexParams::new(256, 8, 96, lance::index::vector::DistanceType::L2)
        );
        
        dataset.create_index(
            &["embedding"],
            lance::index::IndexType::Vector,
            Some("embedding_idx".to_string()),
            &params,
            true,
        ).await.map_err(|e| Error::Storage(format!("Index creation failed: {}", e)))?;
        
        Ok(())
    }
    
    /// Vector similarity search
    pub async fn vector_search(
        &self,
        embedding: &[f32],
        k: usize,
    ) -> Result<Vec<(String, f32)>> {
        let Some(ref dataset) = self.thoughts else {
            return Ok(vec![]);
        };
        
        let scanner = dataset.scan()
            .nearest("embedding", embedding, k)
            .map_err(|e| Error::Storage(e.to_string()))?;
        
        let batches: Vec<RecordBatch> = scanner.try_into_stream()
            .await
            .map_err(|e| Error::Storage(e.to_string()))?
            .try_collect()
            .await
            .map_err(|e| Error::Storage(e.to_string()))?;
        
        let mut results = Vec::new();
        for batch in batches {
            let id_col = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
            let dist_col = batch.column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());
            
            for i in 0..batch.num_rows() {
                let id = id_col.value(i).to_string();
                let dist = dist_col.map(|d| d.value(i)).unwrap_or(0.0);
                results.push((id, dist));
            }
        }
        
        Ok(results)
    }
}

// === Row Types ===

#[derive(Debug, Clone)]
pub struct ThoughtRow {
    pub id: String,
    pub content: String,
    pub fingerprint: Vec<u8>,
    pub frequency: f32,
    pub confidence: f32,
    pub created_at: i64,
    pub style: Option<String>,
    pub layer: Option<i32>,
}

impl ThoughtRow {
    fn from_batch(batch: &RecordBatch, idx: usize) -> Result<Self> {
        let id = batch.column(0).as_any().downcast_ref::<StringArray>()
            .ok_or_else(|| Error::Storage("Invalid id column".into()))?
            .value(idx).to_string();
        let content = batch.column(1).as_any().downcast_ref::<StringArray>()
            .ok_or_else(|| Error::Storage("Invalid content column".into()))?
            .value(idx).to_string();
        let fingerprint = batch.column(2).as_any().downcast_ref::<FixedSizeBinaryArray>()
            .ok_or_else(|| Error::Storage("Invalid fingerprint column".into()))?
            .value(idx).to_vec();
        let frequency = batch.column(4).as_any().downcast_ref::<Float32Array>()
            .ok_or_else(|| Error::Storage("Invalid frequency column".into()))?
            .value(idx);
        let confidence = batch.column(5).as_any().downcast_ref::<Float32Array>()
            .ok_or_else(|| Error::Storage("Invalid confidence column".into()))?
            .value(idx);
        let created_at = batch.column(6).as_any().downcast_ref::<Int64Array>()
            .ok_or_else(|| Error::Storage("Invalid created_at column".into()))?
            .value(idx);
        
        Ok(Self {
            id,
            content,
            fingerprint,
            frequency,
            confidence,
            created_at,
            style: None,  // TODO: extract
            layer: None,  // TODO: extract
        })
    }
}

#[derive(Debug, Clone)]
pub struct EdgeRow {
    pub id: String,
    pub source_id: String,
    pub target_id: String,
    pub relation: String,
    pub frequency: f32,
    pub confidence: f32,
    pub weight: Option<f32>,
}

impl EdgeRow {
    fn from_batch(batch: &RecordBatch, idx: usize) -> Result<Self> {
        Ok(Self {
            id: batch.column(0).as_any().downcast_ref::<StringArray>().unwrap().value(idx).to_string(),
            source_id: batch.column(1).as_any().downcast_ref::<StringArray>().unwrap().value(idx).to_string(),
            target_id: batch.column(2).as_any().downcast_ref::<StringArray>().unwrap().value(idx).to_string(),
            relation: batch.column(3).as_any().downcast_ref::<StringArray>().unwrap().value(idx).to_string(),
            frequency: batch.column(4).as_any().downcast_ref::<Float32Array>().unwrap().value(idx),
            confidence: batch.column(5).as_any().downcast_ref::<Float32Array>().unwrap().value(idx),
            weight: batch.column(6).as_any().downcast_ref::<Float32Array>().map(|a| a.value(idx)),
        })
    }
}

// Need futures for try_collect
use futures::TryStreamExt;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_open_store() {
        let dir = tempfile::tempdir().unwrap();
        let store = LanceStore::open(dir.path()).await.unwrap();
        assert!(store.thoughts.is_some());
        assert!(store.edges.is_some());
    }
    
    #[tokio::test]
    async fn test_insert_thought() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = LanceStore::open(dir.path()).await.unwrap();
        
        let fp = Fingerprint::from_content("test content");
        store.insert_thought(
            "t1",
            "test content",
            &fp,
            None,
            0.9,
            0.8,
        ).await.unwrap();
        
        let rows = store.query_thoughts("id = 't1'").await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].content, "test content");
    }
}
