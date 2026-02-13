//! DnTreeTableProvider: Expose BindSpace DN tree as DataFusion TableProvider.
//!
//! ```sql
//! SELECT dn, is_spine, depth, label, rung,
//!        hamming(content, $query) as dist
//! FROM dn_tree
//! WHERE depth <= 3
//! ORDER BY dist ASC
//! LIMIT 10
//! ```
//!
//! After BindSpace Unification, this provider reads directly from BindSpace
//! via the DnIndex — no separate DnSpineCache needed.

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

use crate::container::{CONTAINER_BYTES, MetaView};
use crate::storage::bind_space::BindSpace;

// =============================================================================
// CONSTANTS
// =============================================================================

/// Container size in bytes for schema definition
const C_BYTES: i32 = CONTAINER_BYTES as i32;

// =============================================================================
// SCHEMA
// =============================================================================

/// Create the schema for the dn_tree table.
fn dn_tree_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("dn", DataType::UInt64, false),
        Field::new("is_spine", DataType::Boolean, false),
        Field::new("depth", DataType::UInt8, false),
        Field::new("parent_dn", DataType::UInt64, true),
        Field::new("address", DataType::UInt16, false),
        Field::new("meta", DataType::FixedSizeBinary(C_BYTES), false),
        Field::new("content", DataType::FixedSizeBinary(C_BYTES), false),
        Field::new("label", DataType::Utf8, true),
        Field::new("rung", DataType::UInt8, false),
    ]))
}

// =============================================================================
// TABLE PROVIDER
// =============================================================================

/// DataFusion TableProvider that reads DN tree nodes from BindSpace.
///
/// After BindSpace Unification, DN tree data lives directly in BindSpace
/// nodes. The DnIndex provides PackedDn ↔ Addr bidirectional lookup.
/// Meta and content are accessed through BindNode::meta_container() / content_container().
pub struct DnTreeTableProvider {
    schema: SchemaRef,
    bind_space: Arc<RwLock<BindSpace>>,
}

impl std::fmt::Debug for DnTreeTableProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DnTreeTableProvider")
            .field("schema", &self.schema)
            .finish()
    }
}

impl DnTreeTableProvider {
    pub fn new(bind_space: Arc<RwLock<BindSpace>>) -> Self {
        Self {
            schema: dn_tree_schema(),
            bind_space,
        }
    }
}

#[async_trait]
impl TableProvider for DnTreeTableProvider {
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
        let projected_schema = match projection {
            Some(proj) => {
                let fields: Vec<Arc<Field>> = proj
                    .iter()
                    .map(|&i| Arc::new(self.schema.field(i).clone()))
                    .collect();
                Arc::new(Schema::new(fields))
            }
            None => self.schema.clone(),
        };

        let properties = PlanProperties::new(
            EquivalenceProperties::new(projected_schema.clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        );

        Ok(Arc::new(DnTreeScan {
            schema: self.schema.clone(),
            projected_schema,
            bind_space: self.bind_space.clone(),
            projection: projection.cloned(),
            properties,
        }))
    }
}

// =============================================================================
// EXECUTION PLAN
// =============================================================================

struct DnTreeScan {
    schema: SchemaRef,
    projected_schema: SchemaRef,
    bind_space: Arc<RwLock<BindSpace>>,
    projection: Option<Vec<usize>>,
    properties: PlanProperties,
}

impl std::fmt::Debug for DnTreeScan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DnTreeScan")
            .field("schema", &self.schema)
            .field("projected_schema", &self.projected_schema)
            .field("projection", &self.projection)
            .finish()
    }
}

impl DisplayAs for DnTreeScan {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "DnTreeScan(bindspace)")
    }
}

impl ExecutionPlan for DnTreeScan {
    fn name(&self) -> &str {
        "DnTreeScan"
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
        let bs = self.bind_space.read();

        // Collect all DN-indexed nodes into Arrow arrays
        let mut dn_values = Vec::new();
        let mut is_spine_values = Vec::new();
        let mut depth_values = Vec::new();
        let mut parent_dn_values: Vec<Option<u64>> = Vec::new();
        let mut address_values = Vec::new();
        let mut meta_values: Vec<Vec<u8>> = Vec::new();
        let mut content_values: Vec<Vec<u8>> = Vec::new();
        let mut label_values: Vec<Option<String>> = Vec::new();
        let mut rung_values = Vec::new();

        let row_count = bs.dn_index.len();

        for (dn, addr) in bs.dn_index.iter() {
            if let Some(node) = bs.read(addr) {
                dn_values.push(dn.raw());
                is_spine_values.push(node.is_spine);
                depth_values.push(dn.depth());
                parent_dn_values.push(dn.parent().map(|p| p.raw()));
                address_values.push(addr.0);

                // Read meta/content containers from BindNode fingerprint halves
                meta_values.push(node.meta_container().as_bytes().to_vec());
                content_values.push(node.content_container().as_bytes().to_vec());

                label_values.push(node.label.clone());

                let meta_view = MetaView::new(node.meta_words());
                rung_values.push(meta_view.rung_level());
            }
        }

        // Handle empty projection (e.g. SELECT COUNT(*))
        if self.projected_schema.fields().is_empty() {
            let options =
                arrow::record_batch::RecordBatchOptions::new().with_row_count(Some(row_count));
            let batch =
                RecordBatch::try_new_with_options(self.projected_schema.clone(), vec![], &options)?;
            return Ok(Box::pin(DnTreeStream {
                schema: self.projected_schema.clone(),
                batch: Some(batch),
            }));
        }

        // Build arrays
        let dn_array: ArrayRef = Arc::new(UInt64Array::from(dn_values));
        let is_spine_array: ArrayRef = Arc::new(BooleanArray::from(is_spine_values));
        let depth_array: ArrayRef = Arc::new(UInt8Array::from(depth_values));
        let parent_dn_array: ArrayRef = Arc::new(UInt64Array::from(parent_dn_values));
        let address_array: ArrayRef = Arc::new(UInt16Array::from(address_values));

        // Fixed-size binary for meta and content
        let meta_array: ArrayRef = Arc::new(
            FixedSizeBinaryArray::try_from_iter(meta_values.iter().map(|v| v.as_slice()))
                .map_err(|e| DataFusionError::Execution(format!("meta array: {e}")))?,
        );
        let content_array: ArrayRef = Arc::new(
            FixedSizeBinaryArray::try_from_iter(content_values.iter().map(|v| v.as_slice()))
                .map_err(|e| DataFusionError::Execution(format!("content array: {e}")))?,
        );

        let label_array: ArrayRef = Arc::new(StringArray::from(label_values));
        let rung_array: ArrayRef = Arc::new(UInt8Array::from(rung_values));

        let all_columns: Vec<ArrayRef> = vec![
            dn_array,
            is_spine_array,
            depth_array,
            parent_dn_array,
            address_array,
            meta_array,
            content_array,
            label_array,
            rung_array,
        ];

        // Apply projection
        let columns = match &self.projection {
            Some(proj) => proj.iter().map(|&i| all_columns[i].clone()).collect(),
            None => all_columns,
        };

        let batch = RecordBatch::try_new(self.projected_schema.clone(), columns)?;

        Ok(Box::pin(DnTreeStream {
            schema: self.projected_schema.clone(),
            batch: Some(batch),
        }))
    }
}

// =============================================================================
// STREAM
// =============================================================================

struct DnTreeStream {
    schema: SchemaRef,
    batch: Option<RecordBatch>,
}

impl Stream for DnTreeStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Ready(self.batch.take().map(Ok))
    }
}

impl RecordBatchStream for DnTreeStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

// =============================================================================
// EXTENSION TRAIT
// =============================================================================

/// Extension trait for registering BindSpace DN tree with DataFusion.
pub trait DnTreeExt {
    fn register_dn_tree(&self, bind_space: Arc<RwLock<BindSpace>>) -> Result<()>;
}

impl DnTreeExt for SessionContext {
    fn register_dn_tree(&self, bind_space: Arc<RwLock<BindSpace>>) -> Result<()> {
        let provider = DnTreeTableProvider::new(bind_space);
        self.register_table("dn_tree", Arc::new(provider))?;
        Ok(())
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::bind_space::{BindSpace, FINGERPRINT_WORDS};

    #[test]
    fn test_dn_tree_schema() {
        let schema = dn_tree_schema();
        assert_eq!(schema.fields().len(), 9);
        assert_eq!(schema.field(0).name(), "dn");
        assert_eq!(schema.field(1).name(), "is_spine");
        assert_eq!(schema.field(4).name(), "address");
        assert_eq!(schema.field(5).name(), "meta");
        assert_eq!(schema.field(6).name(), "content");
    }

    #[tokio::test]
    async fn test_dn_tree_provider_empty() {
        let bs = Arc::new(RwLock::new(BindSpace::new()));
        let provider = DnTreeTableProvider::new(bs);

        assert_eq!(provider.table_type(), TableType::Base);
        assert_eq!(provider.schema().fields().len(), 9);
    }

    #[tokio::test]
    async fn test_dn_tree_provider_with_data() {
        let mut bs = BindSpace::new();

        // Insert nodes via write_dn_path (registers in DnIndex automatically)
        let fp1 = [1u64; FINGERPRINT_WORDS];
        let fp2 = [2u64; FINGERPRINT_WORDS];
        let fp3 = [3u64; FINGERPRINT_WORDS];
        bs.write_dn_path("root", fp1, 0);
        bs.write_dn_path("root:child1", fp2, 1);
        bs.write_dn_path("root:child2", fp3, 1);

        let bs = Arc::new(RwLock::new(bs));
        let provider = Arc::new(DnTreeTableProvider::new(bs));
        let ctx = SessionContext::new();
        ctx.register_table("dn_tree", provider).unwrap();

        let df = ctx
            .sql("SELECT COUNT(*) as cnt FROM dn_tree")
            .await
            .unwrap();
        let batches = df.collect().await.unwrap();
        assert_eq!(batches.len(), 1);

        let cnt = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0);
        assert_eq!(cnt, 3);
    }
}
