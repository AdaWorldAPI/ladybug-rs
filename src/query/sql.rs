//! DataFusion SQL Execution Layer
//!
//! Provides SQL query execution over Lance tables with custom UDFs
//! for Hamming similarity, NARS inference, and VSA operations.

use std::sync::Arc;

use arrow::array::{
    ArrayRef, Float32Array, Int64Array, StringArray, UInt64Array,
    BinaryArray, RecordBatch,
};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::*;
use datafusion::execution::context::SessionContext;
use datafusion::logical_expr::{Volatility, create_udf, create_udaf};
use datafusion::physical_plan::functions::make_scalar_function;
use datafusion::datasource::MemTable;

use crate::{Result, Error};
use crate::core::{Fingerprint, hamming_distance_simd};
use crate::nars::TruthValue;

/// SQL executor with registered UDFs
pub struct SqlExecutor {
    ctx: SessionContext,
}

impl SqlExecutor {
    /// Create new executor
    pub fn new() -> Self {
        let ctx = SessionContext::new();
        let mut executor = Self { ctx };
        executor.register_udfs();
        executor
    }
    
    /// Register a Lance table as a DataFusion table
    pub async fn register_lance_table(&self, name: &str, path: &str) -> Result<()> {
        // Lance integrates with DataFusion via LanceDataset
        let sql = format!(
            "CREATE EXTERNAL TABLE {} STORED AS LANCE LOCATION '{}'",
            name, path
        );
        self.ctx.sql(&sql).await
            .map_err(|e| Error::Query(format!("Failed to register table: {}", e)))?;
        Ok(())
    }
    
    /// Register an Arrow RecordBatch as a table
    pub fn register_batch(&self, name: &str, batch: RecordBatch) -> Result<()> {
        let schema = batch.schema();
        let table = MemTable::try_new(schema, vec![vec![batch]])
            .map_err(|e| Error::Query(e.to_string()))?;
        self.ctx.register_table(name, Arc::new(table))
            .map_err(|e| Error::Query(e.to_string()))?;
        Ok(())
    }
    
    /// Execute SQL and return results
    pub async fn execute(&self, sql: &str) -> Result<Vec<RecordBatch>> {
        let df = self.ctx.sql(sql).await
            .map_err(|e| Error::Query(format!("SQL error: {}", e)))?;
        
        let batches = df.collect().await
            .map_err(|e| Error::Query(format!("Execution error: {}", e)))?;
        
        Ok(batches)
    }
    
    /// Execute and return as rows
    pub async fn query(&self, sql: &str) -> Result<QueryResult> {
        let batches = self.execute(sql).await?;
        
        if batches.is_empty() {
            return Ok(QueryResult {
                columns: vec![],
                rows: vec![],
            });
        }
        
        let schema = batches[0].schema();
        let columns: Vec<String> = schema.fields().iter()
            .map(|f| f.name().clone())
            .collect();
        
        let mut rows = Vec::new();
        for batch in batches {
            for row_idx in 0..batch.num_rows() {
                let mut row = Vec::new();
                for col_idx in 0..batch.num_columns() {
                    let value = Self::array_value_to_string(batch.column(col_idx), row_idx);
                    row.push(value);
                }
                rows.push(row);
            }
        }
        
        Ok(QueryResult { columns, rows })
    }
    
    fn array_value_to_string(array: &ArrayRef, idx: usize) -> String {
        if array.is_null(idx) {
            return "NULL".to_string();
        }
        
        match array.data_type() {
            DataType::Utf8 => {
                array.as_any().downcast_ref::<StringArray>()
                    .map(|a| a.value(idx).to_string())
                    .unwrap_or_default()
            }
            DataType::Int64 => {
                array.as_any().downcast_ref::<Int64Array>()
                    .map(|a| a.value(idx).to_string())
                    .unwrap_or_default()
            }
            DataType::UInt64 => {
                array.as_any().downcast_ref::<UInt64Array>()
                    .map(|a| a.value(idx).to_string())
                    .unwrap_or_default()
            }
            DataType::Float32 => {
                array.as_any().downcast_ref::<Float32Array>()
                    .map(|a| format!("{:.4}", a.value(idx)))
                    .unwrap_or_default()
            }
            DataType::Float64 => {
                array.as_any().downcast_ref::<arrow::array::Float64Array>()
                    .map(|a| format!("{:.4}", a.value(idx)))
                    .unwrap_or_default()
            }
            DataType::Binary | DataType::FixedSizeBinary(_) => {
                "[binary]".to_string()
            }
            _ => format!("{:?}", array.data_type()),
        }
    }
    
    // === UDF Registration ===
    
    fn register_udfs(&mut self) {
        self.register_hamming_udf();
        self.register_nars_udfs();
        self.register_vsa_udfs();
    }
    
    /// Register hamming_similarity(fp1, fp2) -> Float32
    fn register_hamming_udf(&self) {
        let hamming_fn = make_scalar_function(|args: &[ArrayRef]| {
            let fp1 = args[0].as_any().downcast_ref::<BinaryArray>()
                .expect("fp1 must be binary");
            let fp2 = args[1].as_any().downcast_ref::<BinaryArray>()
                .expect("fp2 must be binary");
            
            let mut results = Vec::with_capacity(fp1.len());
            
            for i in 0..fp1.len() {
                if fp1.is_null(i) || fp2.is_null(i) {
                    results.push(None);
                } else {
                    let bytes1 = fp1.value(i);
                    let bytes2 = fp2.value(i);
                    
                    // Use SIMD Hamming
                    let distance = hamming_distance_simd(bytes1, bytes2);
                    let max_bits = (bytes1.len() * 8) as u32;
                    let similarity = 1.0 - (distance as f32 / max_bits as f32);
                    results.push(Some(similarity));
                }
            }
            
            Ok(Arc::new(Float32Array::from(results)) as ArrayRef)
        });
        
        let udf = create_udf(
            "hamming_similarity",
            vec![DataType::Binary, DataType::Binary],
            Arc::new(DataType::Float32),
            Volatility::Immutable,
            hamming_fn,
        );
        
        self.ctx.register_udf(udf);
    }
    
    /// Register NARS truth value functions
    fn register_nars_udfs(&self) {
        // nars_deduction(f1, c1, f2, c2) -> (f, c)
        let deduction_fn = make_scalar_function(|args: &[ArrayRef]| {
            let f1 = args[0].as_any().downcast_ref::<Float32Array>().unwrap();
            let c1 = args[1].as_any().downcast_ref::<Float32Array>().unwrap();
            let f2 = args[2].as_any().downcast_ref::<Float32Array>().unwrap();
            let c2 = args[3].as_any().downcast_ref::<Float32Array>().unwrap();
            
            let mut freq_results = Vec::with_capacity(f1.len());
            let mut conf_results = Vec::with_capacity(f1.len());
            
            for i in 0..f1.len() {
                let tv1 = TruthValue::new(f1.value(i), c1.value(i));
                let tv2 = TruthValue::new(f2.value(i), c2.value(i));
                let result = tv1.deduction(&tv2);
                freq_results.push(result.frequency);
                conf_results.push(result.confidence);
            }
            
            // Return as struct with (frequency, confidence)
            // For simplicity, return frequency only - extend as needed
            Ok(Arc::new(Float32Array::from(freq_results)) as ArrayRef)
        });
        
        let deduction_udf = create_udf(
            "nars_deduction",
            vec![DataType::Float32, DataType::Float32, DataType::Float32, DataType::Float32],
            Arc::new(DataType::Float32),
            Volatility::Immutable,
            deduction_fn,
        );
        self.ctx.register_udf(deduction_udf);
        
        // nars_revision(f1, c1, f2, c2) -> f (revised frequency)
        let revision_fn = make_scalar_function(|args: &[ArrayRef]| {
            let f1 = args[0].as_any().downcast_ref::<Float32Array>().unwrap();
            let c1 = args[1].as_any().downcast_ref::<Float32Array>().unwrap();
            let f2 = args[2].as_any().downcast_ref::<Float32Array>().unwrap();
            let c2 = args[3].as_any().downcast_ref::<Float32Array>().unwrap();
            
            let mut results = Vec::with_capacity(f1.len());
            
            for i in 0..f1.len() {
                let tv1 = TruthValue::new(f1.value(i), c1.value(i));
                let tv2 = TruthValue::new(f2.value(i), c2.value(i));
                let result = tv1.revision(&tv2);
                results.push(result.frequency);
            }
            
            Ok(Arc::new(Float32Array::from(results)) as ArrayRef)
        });
        
        let revision_udf = create_udf(
            "nars_revision",
            vec![DataType::Float32, DataType::Float32, DataType::Float32, DataType::Float32],
            Arc::new(DataType::Float32),
            Volatility::Immutable,
            revision_fn,
        );
        self.ctx.register_udf(revision_udf);
    }
    
    /// Register VSA operations
    fn register_vsa_udfs(&self) {
        // vsa_bind(fp1, fp2) -> Binary (XOR)
        let bind_fn = make_scalar_function(|args: &[ArrayRef]| {
            let fp1 = args[0].as_any().downcast_ref::<BinaryArray>().unwrap();
            let fp2 = args[1].as_any().downcast_ref::<BinaryArray>().unwrap();
            
            let mut results: Vec<Option<Vec<u8>>> = Vec::with_capacity(fp1.len());
            
            for i in 0..fp1.len() {
                if fp1.is_null(i) || fp2.is_null(i) {
                    results.push(None);
                } else {
                    let bytes1 = fp1.value(i);
                    let bytes2 = fp2.value(i);
                    let bound: Vec<u8> = bytes1.iter()
                        .zip(bytes2.iter())
                        .map(|(a, b)| a ^ b)
                        .collect();
                    results.push(Some(bound));
                }
            }
            
            Ok(Arc::new(BinaryArray::from(results)) as ArrayRef)
        });
        
        let bind_udf = create_udf(
            "vsa_bind",
            vec![DataType::Binary, DataType::Binary],
            Arc::new(DataType::Binary),
            Volatility::Immutable,
            bind_fn,
        );
        self.ctx.register_udf(bind_udf);
    }
    
    /// Get the DataFusion context for advanced usage
    pub fn context(&self) -> &SessionContext {
        &self.ctx
    }
}

impl Default for SqlExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Query result
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

impl QueryResult {
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
    
    pub fn len(&self) -> usize {
        self.rows.len()
    }
    
    /// Get column index by name
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c == name)
    }
    
    /// Get value at row, column
    pub fn get(&self, row: usize, col: &str) -> Option<&str> {
        let col_idx = self.column_index(col)?;
        self.rows.get(row).and_then(|r| r.get(col_idx)).map(|s| s.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_sql_basic() {
        let executor = SqlExecutor::new();
        
        // Create test data
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("score", DataType::Float32, false),
        ]));
        
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["alice", "bob", "charlie"])),
            Arc::new(Float32Array::from(vec![0.9, 0.8, 0.7])),
        ]).unwrap();
        
        executor.register_batch("test", batch).unwrap();
        
        let result = executor.query("SELECT * FROM test WHERE score > 0.75").await.unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result.get(0, "name"), Some("alice"));
    }
    
    #[tokio::test]
    async fn test_hamming_udf() {
        let executor = SqlExecutor::new();
        
        // Create test data with fingerprints
        let schema = Arc::new(Schema::new(vec![
            Field::new("fp1", DataType::Binary, false),
            Field::new("fp2", DataType::Binary, false),
        ]));
        
        let fp1 = vec![0xFF_u8; 16];  // All 1s
        let fp2 = vec![0xFF_u8; 16];  // All 1s (identical)
        let fp3 = vec![0x00_u8; 16];  // All 0s (opposite)
        
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(BinaryArray::from(vec![fp1.as_slice(), fp1.as_slice()])),
            Arc::new(BinaryArray::from(vec![fp2.as_slice(), fp3.as_slice()])),
        ]).unwrap();
        
        executor.register_batch("fps", batch).unwrap();
        
        let result = executor.query("SELECT hamming_similarity(fp1, fp2) as sim FROM fps").await.unwrap();
        assert_eq!(result.len(), 2);
        // First row: identical fingerprints = similarity 1.0
        // Second row: opposite fingerprints = similarity 0.0
    }
    
    #[tokio::test]
    async fn test_nars_udf() {
        let executor = SqlExecutor::new();
        
        let schema = Arc::new(Schema::new(vec![
            Field::new("f1", DataType::Float32, false),
            Field::new("c1", DataType::Float32, false),
            Field::new("f2", DataType::Float32, false),
            Field::new("c2", DataType::Float32, false),
        ]));
        
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float32Array::from(vec![0.9])),
            Arc::new(Float32Array::from(vec![0.9])),
            Arc::new(Float32Array::from(vec![0.8])),
            Arc::new(Float32Array::from(vec![0.8])),
        ]).unwrap();
        
        executor.register_batch("truth", batch).unwrap();
        
        let result = executor.query("SELECT nars_deduction(f1, c1, f2, c2) as deduced FROM truth").await.unwrap();
        assert_eq!(result.len(), 1);
    }
}
