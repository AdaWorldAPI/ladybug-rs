//! DataFusion SQL Execution Engine
//!
//! Integrates Apache DataFusion for SQL query execution over Lance tables.
//! Registers custom UDFs for Hamming distance and similarity.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                  DATAFUSION EXECUTION                            │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │   Query → Parser → Logical Plan → Physical Plan → Execution     │
//! │                                                                  │
//! │   Custom UDFs:                                                   │
//! │     - hamming(a, b) → distance (0-10000)                        │
//! │     - similarity(a, b) → similarity (0.0-1.0)                   │
//! │     - popcount(x) → count of set bits                           │
//! │     - xor_bind(a, b) → XOR of two fingerprints                  │
//! │                                                                  │
//! │   Data Sources:                                                  │
//! │     - Lance tables (nodes, edges, sessions)                     │
//! │     - In-memory Arrow batches                                   │
//! │                                                                  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use arrow::array::*;
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use datafusion::prelude::*;
use datafusion::execution::context::SessionContext;
use datafusion::logical_expr::{ScalarUDF, Signature, TypeSignature, Volatility};
use std::sync::Arc;

use crate::core::DIM;
use crate::{Error, Result};

// =============================================================================
// HAMMING OPERATIONS (Pure functions for UDFs)
// =============================================================================

/// Compute Hamming distance between two byte slices
fn hamming_distance_bytes(a: &[u8], b: &[u8]) -> u32 {
    let min_len = a.len().min(b.len());
    let mut dist: u32 = 0;
    
    // Process 8 bytes at a time for SIMD-friendly access
    let chunks = min_len / 8;
    for i in 0..chunks {
        let offset = i * 8;
        let a_u64 = u64::from_le_bytes(a[offset..offset + 8].try_into().unwrap());
        let b_u64 = u64::from_le_bytes(b[offset..offset + 8].try_into().unwrap());
        dist += (a_u64 ^ b_u64).count_ones();
    }
    
    // Handle remaining bytes
    for i in (chunks * 8)..min_len {
        dist += (a[i] ^ b[i]).count_ones();
    }
    
    dist
}

/// Compute similarity from Hamming distance
fn hamming_similarity(dist: u32) -> f32 {
    1.0 - (dist as f32 / DIM as f32)
}

/// XOR bind two fingerprints
fn xor_bind_bytes(a: &[u8], b: &[u8]) -> Vec<u8> {
    a.iter().zip(b.iter()).map(|(x, y)| x ^ y).collect()
}

// =============================================================================
// EXECUTION CONTEXT
// =============================================================================

/// DataFusion-based SQL execution engine
pub struct SqlEngine {
    /// DataFusion session context
    ctx: SessionContext,
    /// Path to database (for Lance table registration)
    db_path: Option<String>,
}

impl SqlEngine {
    /// Create a new SQL engine
    pub async fn new() -> Self {
        let ctx = SessionContext::new();
        let mut engine = Self {
            ctx,
            db_path: None,
        };
        
        // Register custom UDFs
        engine.register_udfs();
        
        engine
    }
    
    /// Create engine with database path for Lance tables
    pub async fn with_database(path: impl Into<String>) -> Result<Self> {
        let mut engine = Self::new().await;
        let db_path = path.into();
        engine.db_path = Some(db_path.clone());
        
        // Register Lance tables
        engine.register_lance_tables(&db_path).await?;
        
        Ok(engine)
    }
    
    /// Register all cognitive UDFs (DataFusion 50+ ScalarUDFImpl)
    fn register_udfs(&mut self) {
        use super::cognitive_udfs::register_cognitive_udfs;
        register_cognitive_udfs(&self.ctx);
    }
    
    /// Register Lance tables as DataFusion tables
    #[cfg(feature = "lancedb")]
    async fn register_lance_tables(&mut self, db_path: &str) -> Result<()> {
        use lance::dataset::Dataset;
        use datafusion::datasource::MemTable;

        // Register nodes table
        let nodes_path = format!("{}/nodes.lance", db_path);
        if std::path::Path::new(&nodes_path).exists() {
            let dataset = Dataset::open(&nodes_path).await?;
            let schema = dataset.schema().clone();

            // Read all data into memory (for now - TODO: use Lance TableProvider)
            let batches = dataset
                .scan()
                .try_into_stream()
                .await?;

            use futures::StreamExt;
            let mut all_batches = Vec::new();
            let mut stream = batches;
            while let Some(batch) = stream.next().await {
                all_batches.push(batch?);
            }

            if !all_batches.is_empty() {
                let table = MemTable::try_new(Arc::new(schema.into()), vec![all_batches])?;
                self.ctx.register_table("nodes", Arc::new(table))?;
            }
        }

        // Register edges table
        let edges_path = format!("{}/edges.lance", db_path);
        if std::path::Path::new(&edges_path).exists() {
            let dataset = Dataset::open(&edges_path).await?;
            let schema = dataset.schema().clone();

            let batches = dataset
                .scan()
                .try_into_stream()
                .await?;

            use futures::StreamExt;
            let mut all_batches = Vec::new();
            let mut stream = batches;
            while let Some(batch) = stream.next().await {
                all_batches.push(batch?);
            }

            if !all_batches.is_empty() {
                let table = MemTable::try_new(Arc::new(schema.into()), vec![all_batches])?;
                self.ctx.register_table("edges", Arc::new(table))?;
            }
        }

        Ok(())
    }

    /// No-op when lancedb feature not enabled
    #[cfg(not(feature = "lancedb"))]
    async fn register_lance_tables(&mut self, _db_path: &str) -> Result<()> {
        Ok(())
    }
    
    /// Register an in-memory table
    pub fn register_table(&mut self, name: &str, batches: Vec<RecordBatch>) -> Result<()> {
        if batches.is_empty() {
            return Ok(());
        }
        
        let schema = batches[0].schema();
        let table = datafusion::datasource::MemTable::try_new(schema, vec![batches])?;
        self.ctx.register_table(name, Arc::new(table))?;
        Ok(())
    }
    
    /// Execute a SQL query
    pub async fn execute(&self, sql: &str) -> Result<Vec<RecordBatch>> {
        let df = self.ctx.sql(sql).await?;
        let batches = df.collect().await?;
        Ok(batches)
    }
    
    /// Execute a SQL query and return a DataFrame
    pub async fn query(&self, sql: &str) -> Result<DataFrame> {
        let df = self.ctx.sql(sql).await?;
        Ok(df)
    }
    
    /// Execute with parameters (prepared statement style)
    pub async fn execute_with_params(
        &self,
        sql: &str,
        params: &[(&str, ScalarValue)],
    ) -> Result<Vec<RecordBatch>> {
        // Replace $param with actual values
        let mut processed_sql = sql.to_string();
        for (name, value) in params {
            let placeholder = format!("${}", name);
            let replacement = match value {
                ScalarValue::Utf8(Some(s)) => format!("'{}'", s.replace('\'', "''")),
                ScalarValue::Int64(Some(n)) => n.to_string(),
                ScalarValue::Float64(Some(f)) => f.to_string(),
                ScalarValue::Boolean(Some(b)) => if *b { "TRUE" } else { "FALSE" }.to_string(),
                ScalarValue::Binary(Some(b)) => format!("X'{}'", hex::encode(b)),
                _ => "NULL".to_string(),
            };
            processed_sql = processed_sql.replace(&placeholder, &replacement);
        }
        
        self.execute(&processed_sql).await
    }
    
    /// Get the underlying DataFusion context
    pub fn context(&self) -> &SessionContext {
        &self.ctx
    }
}

impl Default for SqlEngine {
    fn default() -> Self {
        // Create synchronously for Default trait
        let ctx = SessionContext::new();
        let mut engine = Self {
            ctx,
            db_path: None,
        };
        engine.register_udfs();
        engine
    }
}

// =============================================================================
// UDF IMPLEMENTATIONS (disabled - requires DataFusion 43+ ScalarUDFImpl trait)
// =============================================================================

#[allow(unused_imports)]
use datafusion::arrow::datatypes::Field;
#[allow(unused_imports)]
use datafusion::logical_expr::ColumnarValue;
#[allow(unused_imports)]
use datafusion::scalar::ScalarValue;

/* TODO: Rewrite UDFs using ScalarUDFImpl trait for DataFusion 43+
   The old closure-based ScalarUDF::new() API is no longer available.

/// Create hamming distance UDF
fn create_hamming_udf() -> ScalarUDF {
    let signature = Signature::new(
        TypeSignature::Any(2),
        Volatility::Immutable,
    );
    
    ScalarUDF::new(
        "hamming",
        &signature,
        &(|_: &[DataType]| Ok(Arc::new(DataType::UInt32))),
        &(|args: &[ColumnarValue]| {
            match (&args[0], &args[1]) {
                (ColumnarValue::Array(a), ColumnarValue::Array(b)) => {
                    let result = hamming_array(a.clone(), b.clone())?;
                    Ok(ColumnarValue::Array(result))
                }
                (ColumnarValue::Scalar(ScalarValue::Binary(Some(a))), 
                 ColumnarValue::Scalar(ScalarValue::Binary(Some(b)))) => {
                    let dist = hamming_distance_bytes(a, b);
                    Ok(ColumnarValue::Scalar(ScalarValue::UInt32(Some(dist))))
                }
                _ => Err(datafusion::error::DataFusionError::Execution(
                    "hamming requires binary arguments".into()
                ))
            }
        }),
    )
}

/// Create similarity UDF
fn create_similarity_udf() -> ScalarUDF {
    let signature = Signature::new(
        TypeSignature::Any(2),
        Volatility::Immutable,
    );
    
    ScalarUDF::new(
        "similarity",
        &signature,
        &(|_: &[DataType]| Ok(Arc::new(DataType::Float32))),
        &(|args: &[ColumnarValue]| {
            match (&args[0], &args[1]) {
                (ColumnarValue::Array(a), ColumnarValue::Array(b)) => {
                    let result = similarity_array(a.clone(), b.clone())?;
                    Ok(ColumnarValue::Array(result))
                }
                (ColumnarValue::Scalar(ScalarValue::Binary(Some(a))), 
                 ColumnarValue::Scalar(ScalarValue::Binary(Some(b)))) => {
                    let dist = hamming_distance_bytes(a, b);
                    let sim = hamming_similarity(dist);
                    Ok(ColumnarValue::Scalar(ScalarValue::Float32(Some(sim))))
                }
                _ => Err(datafusion::error::DataFusionError::Execution(
                    "similarity requires binary arguments".into()
                ))
            }
        }),
    )
}

/// Create popcount UDF
fn create_popcount_udf() -> ScalarUDF {
    let signature = Signature::new(
        TypeSignature::Any(1),
        Volatility::Immutable,
    );
    
    ScalarUDF::new(
        "popcount",
        &signature,
        &(|_: &[DataType]| Ok(Arc::new(DataType::UInt32))),
        &(|args: &[ColumnarValue]| {
            match &args[0] {
                ColumnarValue::Array(arr) => {
                    let result = popcount_array(arr.clone())?;
                    Ok(ColumnarValue::Array(result))
                }
                ColumnarValue::Scalar(ScalarValue::UInt64(Some(n))) => {
                    Ok(ColumnarValue::Scalar(ScalarValue::UInt32(Some(n.count_ones()))))
                }
                _ => Err(datafusion::error::DataFusionError::Execution(
                    "popcount requires uint64 argument".into()
                ))
            }
        }),
    )
}

/// Create xor_bind UDF
fn create_xor_bind_udf() -> ScalarUDF {
    let signature = Signature::new(
        TypeSignature::Any(2),
        Volatility::Immutable,
    );
    
    ScalarUDF::new(
        "xor_bind",
        &signature,
        &(|_: &[DataType]| Ok(Arc::new(DataType::Binary))),
        &(|args: &[ColumnarValue]| {
            match (&args[0], &args[1]) {
                (ColumnarValue::Array(a), ColumnarValue::Array(b)) => {
                    let result = xor_bind_array(a.clone(), b.clone())?;
                    Ok(ColumnarValue::Array(result))
                }
                (ColumnarValue::Scalar(ScalarValue::Binary(Some(a))), 
                 ColumnarValue::Scalar(ScalarValue::Binary(Some(b)))) => {
                    let result = xor_bind_bytes(a, b);
                    Ok(ColumnarValue::Scalar(ScalarValue::Binary(Some(result))))
                }
                _ => Err(datafusion::error::DataFusionError::Execution(
                    "xor_bind requires binary arguments".into()
                ))
            }
        }),
    )
}
End of disabled UDF code */

// =============================================================================
// ARRAY OPERATIONS
// =============================================================================

/// Compute Hamming distance for arrays
#[allow(dead_code)]
fn hamming_array(a: ArrayRef, b: ArrayRef) -> datafusion::error::Result<ArrayRef> {
    let a_bin = a.as_any().downcast_ref::<BinaryArray>()
        .or_else(|| {
            a.as_any().downcast_ref::<FixedSizeBinaryArray>()
                .map(|_| todo!("Convert FixedSizeBinaryArray"))
        });
    let b_bin = b.as_any().downcast_ref::<BinaryArray>();
    
    match (a_bin, b_bin) {
        (Some(a), Some(b)) => {
            let mut builder = UInt32Builder::new();
            for i in 0..a.len() {
                if a.is_null(i) || b.is_null(i) {
                    builder.append_null();
                } else {
                    let dist = hamming_distance_bytes(a.value(i), b.value(i));
                    builder.append_value(dist);
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        _ => {
            // Try FixedSizeBinaryArray
            let a_fixed = a.as_any().downcast_ref::<FixedSizeBinaryArray>();
            let b_fixed = b.as_any().downcast_ref::<FixedSizeBinaryArray>();
            
            match (a_fixed, b_fixed) {
                (Some(a), Some(b)) => {
                    let mut builder = UInt32Builder::new();
                    for i in 0..a.len() {
                        if a.is_null(i) || b.is_null(i) {
                            builder.append_null();
                        } else {
                            let dist = hamming_distance_bytes(a.value(i), b.value(i));
                            builder.append_value(dist);
                        }
                    }
                    Ok(Arc::new(builder.finish()))
                }
                _ => Err(datafusion::error::DataFusionError::Execution(
                    "hamming_array requires binary arrays".into()
                ))
            }
        }
    }
}

/// Compute similarity for arrays
#[allow(dead_code)]
fn similarity_array(a: ArrayRef, b: ArrayRef) -> datafusion::error::Result<ArrayRef> {
    let a_fixed = a.as_any().downcast_ref::<FixedSizeBinaryArray>();
    let b_fixed = b.as_any().downcast_ref::<FixedSizeBinaryArray>();
    
    match (a_fixed, b_fixed) {
        (Some(a), Some(b)) => {
            let mut builder = Float32Builder::new();
            for i in 0..a.len() {
                if a.is_null(i) || b.is_null(i) {
                    builder.append_null();
                } else {
                    let dist = hamming_distance_bytes(a.value(i), b.value(i));
                    let sim = hamming_similarity(dist);
                    builder.append_value(sim);
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        _ => {
            // Try regular binary
            let a_bin = a.as_any().downcast_ref::<BinaryArray>();
            let b_bin = b.as_any().downcast_ref::<BinaryArray>();
            
            match (a_bin, b_bin) {
                (Some(a), Some(b)) => {
                    let mut builder = Float32Builder::new();
                    for i in 0..a.len() {
                        if a.is_null(i) || b.is_null(i) {
                            builder.append_null();
                        } else {
                            let dist = hamming_distance_bytes(a.value(i), b.value(i));
                            let sim = hamming_similarity(dist);
                            builder.append_value(sim);
                        }
                    }
                    Ok(Arc::new(builder.finish()))
                }
                _ => Err(datafusion::error::DataFusionError::Execution(
                    "similarity_array requires binary arrays".into()
                ))
            }
        }
    }
}

/// Compute popcount for array
#[allow(dead_code)]
fn popcount_array(arr: ArrayRef) -> datafusion::error::Result<ArrayRef> {
    let u64_arr = arr.as_any().downcast_ref::<UInt64Array>();
    
    match u64_arr {
        Some(arr) => {
            let mut builder = UInt32Builder::new();
            for i in 0..arr.len() {
                if arr.is_null(i) {
                    builder.append_null();
                } else {
                    builder.append_value(arr.value(i).count_ones());
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        None => {
            // Try binary array - count all bits
            let bin_arr = arr.as_any().downcast_ref::<BinaryArray>();
            match bin_arr {
                Some(arr) => {
                    let mut builder = UInt32Builder::new();
                    for i in 0..arr.len() {
                        if arr.is_null(i) {
                            builder.append_null();
                        } else {
                            let count: u32 = arr.value(i).iter().map(|b| b.count_ones()).sum();
                            builder.append_value(count);
                        }
                    }
                    Ok(Arc::new(builder.finish()))
                }
                None => Err(datafusion::error::DataFusionError::Execution(
                    "popcount_array requires uint64 or binary array".into()
                ))
            }
        }
    }
}

/// XOR bind two arrays
#[allow(dead_code)]
fn xor_bind_array(a: ArrayRef, b: ArrayRef) -> datafusion::error::Result<ArrayRef> {
    let a_bin = a.as_any().downcast_ref::<BinaryArray>();
    let b_bin = b.as_any().downcast_ref::<BinaryArray>();
    
    match (a_bin, b_bin) {
        (Some(a), Some(b)) => {
            let mut builder = BinaryBuilder::new();
            for i in 0..a.len() {
                if a.is_null(i) || b.is_null(i) {
                    builder.append_null();
                } else {
                    let result = xor_bind_bytes(a.value(i), b.value(i));
                    builder.append_value(&result);
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        _ => {
            // Try FixedSizeBinaryArray
            let a_fixed = a.as_any().downcast_ref::<FixedSizeBinaryArray>();
            let b_fixed = b.as_any().downcast_ref::<FixedSizeBinaryArray>();
            
            match (a_fixed, b_fixed) {
                (Some(a), Some(b)) => {
                    let size = a.value_length() as usize;
                    let mut builder = FixedSizeBinaryBuilder::new(size as i32);
                    for i in 0..a.len() {
                        if a.is_null(i) || b.is_null(i) {
                            builder.append_null();
                        } else {
                            let result = xor_bind_bytes(a.value(i), b.value(i));
                            builder.append_value(&result)?;
                        }
                    }
                    Ok(Arc::new(builder.finish()))
                }
                _ => Err(datafusion::error::DataFusionError::Execution(
                    "xor_bind_array requires binary arrays".into()
                ))
            }
        }
    }
}

// =============================================================================
// QUERY BUILDER
// =============================================================================

/// Helper for building SQL queries programmatically
pub struct QueryBuilder {
    select: Vec<String>,
    from: String,
    joins: Vec<String>,
    where_clauses: Vec<String>,
    order_by: Vec<String>,
    limit: Option<u64>,
    offset: Option<u64>,
}

impl QueryBuilder {
    /// Start building a query from a table
    pub fn from(table: &str) -> Self {
        Self {
            select: Vec::new(),
            from: table.to_string(),
            joins: Vec::new(),
            where_clauses: Vec::new(),
            order_by: Vec::new(),
            limit: None,
            offset: None,
        }
    }
    
    /// Add SELECT columns
    pub fn select(mut self, columns: &[&str]) -> Self {
        self.select.extend(columns.iter().map(|s| s.to_string()));
        self
    }
    
    /// Add a JOIN clause
    pub fn join(mut self, join_type: &str, table: &str, on: &str) -> Self {
        self.joins.push(format!("{} JOIN {} ON {}", join_type, table, on));
        self
    }
    
    /// Add a WHERE condition
    pub fn where_clause(mut self, condition: &str) -> Self {
        self.where_clauses.push(condition.to_string());
        self
    }
    
    /// Add Hamming distance filter
    pub fn where_hamming_lt(mut self, col: &str, param: &str, max_dist: u32) -> Self {
        self.where_clauses.push(format!("hamming({}, {}) < {}", col, param, max_dist));
        self
    }
    
    /// Add similarity filter
    pub fn where_similar(mut self, col: &str, param: &str, min_sim: f32) -> Self {
        let max_dist = ((1.0 - min_sim) * DIM as f32) as u32;
        self.where_clauses.push(format!("hamming({}, {}) < {}", col, param, max_dist));
        self
    }
    
    /// Add ORDER BY
    pub fn order_by(mut self, expr: &str, desc: bool) -> Self {
        let dir = if desc { "DESC" } else { "ASC" };
        self.order_by.push(format!("{} {}", expr, dir));
        self
    }
    
    /// Set LIMIT
    pub fn limit(mut self, n: u64) -> Self {
        self.limit = Some(n);
        self
    }
    
    /// Set OFFSET
    pub fn offset(mut self, n: u64) -> Self {
        self.offset = Some(n);
        self
    }
    
    /// Build the SQL query string
    pub fn build(self) -> String {
        let mut sql = String::new();
        
        // SELECT
        let cols = if self.select.is_empty() {
            "*".to_string()
        } else {
            self.select.join(", ")
        };
        sql.push_str(&format!("SELECT {}\n", cols));
        
        // FROM
        sql.push_str(&format!("FROM {}\n", self.from));
        
        // JOINs
        for join in self.joins {
            sql.push_str(&join);
            sql.push('\n');
        }
        
        // WHERE
        if !self.where_clauses.is_empty() {
            sql.push_str(&format!("WHERE {}\n", self.where_clauses.join(" AND ")));
        }
        
        // ORDER BY
        if !self.order_by.is_empty() {
            sql.push_str(&format!("ORDER BY {}\n", self.order_by.join(", ")));
        }
        
        // LIMIT
        if let Some(n) = self.limit {
            sql.push_str(&format!("LIMIT {}\n", n));
        }
        
        // OFFSET
        if let Some(n) = self.offset {
            sql.push_str(&format!("OFFSET {}\n", n));
        }
        
        sql
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hamming_distance() {
        let a = vec![0xFF, 0x00, 0xFF, 0x00];
        let b = vec![0x00, 0xFF, 0x00, 0xFF];
        // All bits differ: 4 bytes * 8 bits = 32
        assert_eq!(hamming_distance_bytes(&a, &b), 32);
        
        let c = vec![0xFF, 0xFF, 0xFF, 0xFF];
        let d = vec![0xFF, 0xFF, 0xFF, 0xFF];
        assert_eq!(hamming_distance_bytes(&c, &d), 0);
    }
    
    #[test]
    fn test_similarity() {
        let sim = hamming_similarity(0);
        assert_eq!(sim, 1.0);
        
        let sim = hamming_similarity(DIM as u32);
        assert_eq!(sim, 0.0);
        
        let sim = hamming_similarity(DIM as u32 / 2);
        assert!((sim - 0.5).abs() < 0.01);
    }
    
    #[test]
    fn test_query_builder() {
        let sql = QueryBuilder::from("nodes")
            .select(&["id", "label", "content"])
            .where_clause("label = 'Thought'")
            .where_similar("fingerprint", "$fp", 0.8)
            .order_by("created_at", true)
            .limit(10)
            .build();
        
        assert!(sql.contains("SELECT id, label, content"));
        assert!(sql.contains("FROM nodes"));
        assert!(sql.contains("label = 'Thought'"));
        assert!(sql.contains("hamming(fingerprint, $fp)"));
        assert!(sql.contains("ORDER BY created_at DESC"));
        assert!(sql.contains("LIMIT 10"));
    }
    
    #[tokio::test]
    async fn test_sql_engine_basic() {
        let engine = SqlEngine::new().await;
        
        // Register a simple test table
        let schema = arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("id", DataType::Int64, false),
            arrow::datatypes::Field::new("name", DataType::Utf8, false),
        ]);
        
        let ids: Int64Array = vec![1, 2, 3].into_iter().map(Some).collect();
        let names: StringArray = vec!["a", "b", "c"].into_iter().map(Some).collect();
        
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(ids), Arc::new(names)],
        ).unwrap();
        
        let mut engine = engine;
        engine.register_table("test", vec![batch]).unwrap();
        
        let results = engine.execute("SELECT * FROM test WHERE id > 1").await.unwrap();
        assert_eq!(results[0].num_rows(), 2);
    }
}
