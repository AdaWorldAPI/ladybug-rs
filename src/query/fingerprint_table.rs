//! FingerprintTableProvider: Expose BindSpace as DataFusion TableProvider
//!
//! This makes the cognitive substrate queryable via SQL:
//!
//! ```sql
//! SELECT address, label, similarity(fingerprint, $query) as sim
//! FROM bindspace
//! WHERE similarity(fingerprint, $query) > 0.8
//! ORDER BY sim DESC
//! LIMIT 10
//! ```
//!
//! The TableProvider exposes:
//! - address: UINT16 (8+8 prefix:slot)
//! - fingerprint: FIXED_SIZE_BINARY(2048) (16K bits)
//! - label: UTF8 (optional)
//! - qidx: UINT8 (qualia index)
//! - access_count: UINT32
//! - zone: UTF8 ('surface', 'fluid', 'node')

use std::any::Any;
use std::sync::Arc;

use arrow::array::*;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::datasource::TableProvider;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::context::TaskContext;
use datafusion::logical_expr::TableType;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties, RecordBatchStream,
    SendableRecordBatchStream,
    execution_plan::{Boundedness, EmissionType},
};
use datafusion::prelude::*;
use futures::Stream;
use parking_lot::RwLock;
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::storage::bind_space::{Addr, BindNode, BindSpace, FINGERPRINT_WORDS};

// =============================================================================
// CONSTANTS
// =============================================================================

/// Fingerprint size in bytes (256 * 8 = 2048 bytes for 16384 bits)
const FP_BYTES: usize = FINGERPRINT_WORDS * 8;

// =============================================================================
// SCHEMA
// =============================================================================

/// Create the schema for the bindspace table
fn bindspace_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("address", DataType::UInt16, false),
        Field::new(
            "fingerprint",
            DataType::FixedSizeBinary(FP_BYTES as i32),
            false,
        ),
        Field::new("label", DataType::Utf8, true),
        Field::new("qidx", DataType::UInt8, false),
        Field::new("access_count", DataType::UInt32, false),
        Field::new("zone", DataType::Utf8, false),
    ]))
}

// =============================================================================
// TABLE PROVIDER
// =============================================================================

/// DataFusion TableProvider for BindSpace
///
/// Exposes the cognitive substrate as a SQL-queryable table.
/// Supports predicate pushdown for Hamming distance filters.
pub struct FingerprintTableProvider {
    schema: SchemaRef,
    bind_space: Arc<RwLock<BindSpace>>,
}

impl FingerprintTableProvider {
    /// Create a new table provider wrapping a BindSpace
    pub fn new(bind_space: Arc<RwLock<BindSpace>>) -> Self {
        Self {
            schema: bindspace_schema(),
            bind_space,
        }
    }

    /// Create a test provider with sample data
    #[cfg(test)]
    pub fn test_provider() -> Self {
        let bind_space = Arc::new(RwLock::new(BindSpace::new()));
        Self::new(bind_space)
    }
}

impl std::fmt::Debug for FingerprintTableProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FingerprintTableProvider")
            .field("schema", &self.schema)
            .finish()
    }
}

#[async_trait]
impl TableProvider for FingerprintTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(BindSpaceScan::new(
            self.schema.clone(),
            self.bind_space.clone(),
            projection.cloned(),
        )))
    }
}

// =============================================================================
// EXECUTION PLAN
// =============================================================================

/// Physical execution plan for scanning BindSpace
pub struct BindSpaceScan {
    schema: SchemaRef,
    projected_schema: SchemaRef,
    bind_space: Arc<RwLock<BindSpace>>,
    projection: Option<Vec<usize>>,
    properties: PlanProperties,
}

impl std::fmt::Debug for BindSpaceScan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BindSpaceScan")
            .field("schema", &self.schema)
            .field("projected_schema", &self.projected_schema)
            .field("projection", &self.projection)
            .finish()
    }
}

impl BindSpaceScan {
    pub fn new(
        schema: SchemaRef,
        bind_space: Arc<RwLock<BindSpace>>,
        projection: Option<Vec<usize>>,
    ) -> Self {
        let projected_schema = match &projection {
            Some(indices) => {
                let fields: Vec<_> = indices.iter().map(|&i| schema.field(i).clone()).collect();
                Arc::new(Schema::new(fields))
            }
            None => schema.clone(),
        };

        let properties = PlanProperties::new(
            EquivalenceProperties::new(projected_schema.clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        );

        Self {
            schema,
            projected_schema,
            bind_space,
            projection,
            properties,
        }
    }
}

impl DisplayAs for BindSpaceScan {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "BindSpaceScan")
    }
}

impl ExecutionPlan for BindSpaceScan {
    fn name(&self) -> &str {
        "BindSpaceScan"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.projected_schema.clone()
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        // Read all data from bind space
        let bind_space = self.bind_space.read();
        let batch = bind_space_to_batch(&bind_space, &self.schema, &self.projection)?;

        Ok(Box::pin(MemoryStream::new(
            vec![batch],
            self.projected_schema.clone(),
        )))
    }
}

// =============================================================================
// CONVERSION HELPERS
// =============================================================================

/// Convert BindSpace to RecordBatch
fn bind_space_to_batch(
    bind_space: &BindSpace,
    schema: &SchemaRef,
    projection: &Option<Vec<usize>>,
) -> Result<RecordBatch> {
    let mut addresses: Vec<u16> = Vec::new();
    let mut fingerprints: Vec<Vec<u8>> = Vec::new();
    let mut labels: Vec<Option<String>> = Vec::new();
    let mut qidxs: Vec<u8> = Vec::new();
    let mut access_counts: Vec<u32> = Vec::new();
    let mut zones: Vec<String> = Vec::new();

    // Iterate through all zones in BindSpace
    // Surface zone: 0x00-0x0F
    for prefix in 0x00u8..=0x0F {
        for slot in 0u8..=0xFF {
            let addr = Addr::new(prefix, slot);
            if let Some(node) = bind_space.read(addr) {
                add_node(
                    addr.0,
                    &node,
                    "surface",
                    &mut addresses,
                    &mut fingerprints,
                    &mut labels,
                    &mut qidxs,
                    &mut access_counts,
                    &mut zones,
                );
            }
        }
    }

    // Fluid zone: 0x10-0x7F
    for prefix in 0x10u8..=0x7F {
        for slot in 0u8..=0xFF {
            let addr = Addr::new(prefix, slot);
            if let Some(node) = bind_space.read(addr) {
                add_node(
                    addr.0,
                    &node,
                    "fluid",
                    &mut addresses,
                    &mut fingerprints,
                    &mut labels,
                    &mut qidxs,
                    &mut access_counts,
                    &mut zones,
                );
            }
        }
    }

    // Node zone: 0x80-0xFF
    for prefix in 0x80u8..=0xFF {
        for slot in 0u8..=0xFF {
            let addr = Addr::new(prefix, slot);
            if let Some(node) = bind_space.read(addr) {
                add_node(
                    addr.0,
                    &node,
                    "node",
                    &mut addresses,
                    &mut fingerprints,
                    &mut labels,
                    &mut qidxs,
                    &mut access_counts,
                    &mut zones,
                );
            }
        }
    }

    // Handle empty case early - compute projected schema first
    let projected_schema = match projection {
        Some(indices) => {
            let fields: Vec<_> = indices.iter().map(|&i| schema.field(i).clone()).collect();
            Arc::new(Schema::new(fields))
        }
        None => schema.clone(),
    };

    if addresses.is_empty() {
        return Ok(RecordBatch::new_empty(projected_schema));
    }

    // Build arrays
    let address_array: UInt16Array = addresses.into();

    // Build FixedSizeBinaryArray for fingerprints
    let mut fp_builder = FixedSizeBinaryBuilder::with_capacity(fingerprints.len(), FP_BYTES as i32);
    for fp in &fingerprints {
        fp_builder
            .append_value(fp)
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
    }
    let fp_array = fp_builder.finish();

    let label_array: StringArray = labels.into_iter().collect();
    let qidx_array: UInt8Array = qidxs.into();
    let access_array: UInt32Array = access_counts.into();
    let zone_array: StringArray = zones.into_iter().map(Some).collect();

    // Build columns based on projection
    let columns: Vec<ArrayRef> = match projection {
        Some(indices) => {
            let all_columns: Vec<ArrayRef> = vec![
                Arc::new(address_array),
                Arc::new(fp_array),
                Arc::new(label_array),
                Arc::new(qidx_array),
                Arc::new(access_array),
                Arc::new(zone_array),
            ];
            indices.iter().map(|&i| all_columns[i].clone()).collect()
        }
        None => vec![
            Arc::new(address_array),
            Arc::new(fp_array),
            Arc::new(label_array),
            Arc::new(qidx_array),
            Arc::new(access_array),
            Arc::new(zone_array),
        ],
    };

    RecordBatch::try_new(projected_schema, columns)
        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
}

/// Helper to add a node to the arrays
fn add_node(
    addr: u16,
    node: &BindNode,
    zone: &str,
    addresses: &mut Vec<u16>,
    fingerprints: &mut Vec<Vec<u8>>,
    labels: &mut Vec<Option<String>>,
    qidxs: &mut Vec<u8>,
    access_counts: &mut Vec<u32>,
    zones: &mut Vec<String>,
) {
    addresses.push(addr);

    // Convert fingerprint [u64; 256] to bytes
    let mut fp_bytes = vec![0u8; FP_BYTES];
    for (i, &word) in node.fingerprint.iter().enumerate() {
        let start = i * 8;
        fp_bytes[start..start + 8].copy_from_slice(&word.to_le_bytes());
    }
    fingerprints.push(fp_bytes);

    labels.push(node.label.clone());
    qidxs.push(node.qidx);
    access_counts.push(node.access_count);
    zones.push(zone.to_string());
}

// =============================================================================
// MEMORY STREAM
// =============================================================================

/// Simple in-memory stream for RecordBatches
struct MemoryStream {
    batches: Vec<RecordBatch>,
    index: usize,
    schema: SchemaRef,
}

impl MemoryStream {
    fn new(batches: Vec<RecordBatch>, schema: SchemaRef) -> Self {
        Self {
            batches,
            index: 0,
            schema,
        }
    }
}

impl Stream for MemoryStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.index < self.batches.len() {
            let batch = self.batches[self.index].clone();
            self.index += 1;
            Poll::Ready(Some(Ok(batch)))
        } else {
            Poll::Ready(None)
        }
    }
}

impl RecordBatchStream for MemoryStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

// =============================================================================
// EXTENSION TRAIT
// =============================================================================

/// Extension trait to register bindspace table with SessionContext
pub trait BindSpaceExt {
    /// Register the BindSpace as a "bindspace" table
    fn register_bindspace(&self, bind_space: Arc<RwLock<BindSpace>>) -> Result<()>;
}

impl BindSpaceExt for SessionContext {
    fn register_bindspace(&self, bind_space: Arc<RwLock<BindSpace>>) -> Result<()> {
        let provider = FingerprintTableProvider::new(bind_space);
        self.register_table("bindspace", Arc::new(provider))?;
        Ok(())
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_schema() {
        let schema = bindspace_schema();
        assert_eq!(schema.fields().len(), 6);
        assert_eq!(schema.field(0).name(), "address");
        assert_eq!(schema.field(1).name(), "fingerprint");
        assert_eq!(schema.field(2).name(), "label");
    }

    #[tokio::test]
    async fn test_empty_bindspace() {
        let bind_space = Arc::new(RwLock::new(BindSpace::new()));
        let provider = FingerprintTableProvider::new(bind_space);

        let ctx = SessionContext::new();
        ctx.register_table("test", Arc::new(provider)).unwrap();

        // Test that query executes successfully
        // BindSpace::new() pre-initializes surface nodes, so it's not truly empty
        // Query node zone which should be empty
        let df = ctx
            .sql("SELECT * FROM test WHERE zone = 'node'")
            .await
            .unwrap();
        let batches = df.collect().await.unwrap();

        // Node zone should be empty
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(
            total_rows, 0,
            "Expected 0 rows in node zone, got {}",
            total_rows
        );
    }

    #[tokio::test]
    async fn test_with_data() {
        let bind_space = Arc::new(RwLock::new(BindSpace::new()));

        // Insert some nodes using write_labeled
        {
            let mut bs = bind_space.write();
            bs.write_labeled([0u64; FINGERPRINT_WORDS], "test1");
            bs.write_labeled([0u64; FINGERPRINT_WORDS], "test2");
        }

        let provider = FingerprintTableProvider::new(bind_space);

        let ctx = SessionContext::new();
        ctx.register_table("bindspace", Arc::new(provider)).unwrap();

        // Query only node zone (where our inserts go) to avoid counting surface nodes
        let df = ctx
            .sql("SELECT COUNT(*) as cnt FROM bindspace WHERE zone = 'node'")
            .await
            .unwrap();
        let batches = df.collect().await.unwrap();

        let cnt = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(cnt.value(0), 2);
    }

    #[tokio::test]
    async fn test_projection() {
        let bind_space = Arc::new(RwLock::new(BindSpace::new()));

        {
            let mut bs = bind_space.write();
            bs.write_labeled([0u64; FINGERPRINT_WORDS], "projection_test");
        }

        let provider = FingerprintTableProvider::new(bind_space);

        let ctx = SessionContext::new();
        ctx.register_table("bindspace", Arc::new(provider)).unwrap();

        // Select label column filtered by our specific label
        let df = ctx
            .sql("SELECT label FROM bindspace WHERE label = 'projection_test'")
            .await
            .unwrap();
        let batches = df.collect().await.unwrap();

        assert_eq!(batches[0].num_rows(), 1);
        assert_eq!(batches[0].num_columns(), 1);
        let labels = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(labels.value(0), "projection_test");
    }

    #[tokio::test]
    async fn test_register_extension() {
        let bind_space = Arc::new(RwLock::new(BindSpace::new()));

        let ctx = SessionContext::new();
        ctx.register_bindspace(bind_space).unwrap();

        // Verify table is registered
        let df = ctx.sql("SELECT * FROM bindspace LIMIT 1").await.unwrap();
        let schema = df.schema();
        assert!(schema.has_column_with_unqualified_name("address"));
        assert!(schema.has_column_with_unqualified_name("fingerprint"));
    }
}
