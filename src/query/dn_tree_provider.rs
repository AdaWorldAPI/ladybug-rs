//! DnTreeTableProvider: Expose DnSpineCache as DataFusion TableProvider.
//!
//! ```sql
//! SELECT dn, is_spine, depth, label,
//!        container_hamming(content, $query) as dist
//! FROM dn_tree
//! WHERE depth <= 3
//! ORDER BY dist ASC
//! LIMIT 10
//! ```
//!
//! This is a NEW provider alongside the existing FingerprintTableProvider
//! and EdgeTableProvider. It reads from the DnSpineCache (Container architecture).

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
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties,
    RecordBatchStream, SendableRecordBatchStream, Partitioning,
    execution_plan::{Boundedness, EmissionType},
};
use datafusion::prelude::*;
use futures::Stream;
use parking_lot::RwLock;
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::container::dn_spine_cache::DnSpineCache;
use crate::container::CONTAINER_BYTES;

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
        Field::new("meta", DataType::FixedSizeBinary(C_BYTES), false),
        Field::new("content", DataType::FixedSizeBinary(C_BYTES), false),
        Field::new("label", DataType::Utf8, true),
        Field::new("rung", DataType::UInt8, false),
    ]))
}

// =============================================================================
// TABLE PROVIDER
// =============================================================================

/// DataFusion TableProvider for DnSpineCache.
///
/// Exposes the container tree as a SQL-queryable table alongside
/// the existing FingerprintTableProvider.
pub struct DnTreeTableProvider {
    schema: SchemaRef,
    dn_cache: Arc<RwLock<DnSpineCache>>,
}

impl std::fmt::Debug for DnTreeTableProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DnTreeTableProvider")
            .field("schema", &self.schema)
            .finish()
    }
}

impl DnTreeTableProvider {
    pub fn new(dn_cache: Arc<RwLock<DnSpineCache>>) -> Self {
        Self {
            schema: dn_tree_schema(),
            dn_cache,
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
                let fields: Vec<Arc<Field>> = proj.iter()
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
            dn_cache: self.dn_cache.clone(),
            projection: projection.cloned(),
            properties,
        }))
    }
}

// =============================================================================
// EXECUTION PLAN
// =============================================================================

#[derive(Debug)]
struct DnTreeScan {
    schema: SchemaRef,
    projected_schema: SchemaRef,
    dn_cache: Arc<RwLock<DnSpineCache>>,
    projection: Option<Vec<usize>>,
    properties: PlanProperties,
}

impl DisplayAs for DnTreeScan {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "DnTreeScan")
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
        let cache = self.dn_cache.read();

        // Collect all nodes into Arrow arrays
        let mut dn_values = Vec::new();
        let mut is_spine_values = Vec::new();
        let mut depth_values = Vec::new();
        let mut parent_dn_values: Vec<Option<u64>> = Vec::new();
        let mut meta_values: Vec<Vec<u8>> = Vec::new();
        let mut content_values: Vec<Vec<u8>> = Vec::new();
        let mut label_values: Vec<Option<String>> = Vec::new();
        let mut rung_values = Vec::new();

        let row_count = cache.node_count();

        for (&dn, slot) in cache.iter() {
            dn_values.push(dn.raw());
            is_spine_values.push(slot.is_spine);
            depth_values.push(dn.depth());
            parent_dn_values.push(dn.parent().map(|p| p.raw()));

            let meta = cache.cache.read(slot.meta_idx);
            let content = cache.cache.read(slot.content_idx);

            meta_values.push(meta.as_bytes().to_vec());
            content_values.push(content.as_bytes().to_vec());
            label_values.push(cache.label(dn).map(|s| s.to_string()));

            let meta_view = crate::container::MetaView::new(&meta.words);
            rung_values.push(meta_view.rung_level());
        }

        // Handle empty projection (e.g. SELECT COUNT(*))
        if self.projected_schema.fields().is_empty() {
            let options = arrow::record_batch::RecordBatchOptions::new()
                .with_row_count(Some(row_count));
            let batch = RecordBatch::try_new_with_options(
                self.projected_schema.clone(), vec![], &options,
            )?;
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

        // Fixed-size binary for meta and content
        let meta_array: ArrayRef = Arc::new(
            FixedSizeBinaryArray::try_from_iter(
                meta_values.iter().map(|v| v.as_slice())
            ).map_err(|e| DataFusionError::Execution(format!("meta array: {e}")))?
        );
        let content_array: ArrayRef = Arc::new(
            FixedSizeBinaryArray::try_from_iter(
                content_values.iter().map(|v| v.as_slice())
            ).map_err(|e| DataFusionError::Execution(format!("content array: {e}")))?
        );

        let label_array: ArrayRef = Arc::new(StringArray::from(label_values));
        let rung_array: ArrayRef = Arc::new(UInt8Array::from(rung_values));

        let all_columns: Vec<ArrayRef> = vec![
            dn_array,
            is_spine_array,
            depth_array,
            parent_dn_array,
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

    fn poll_next(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
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

/// Extension trait for registering DnSpineCache with DataFusion.
pub trait DnTreeExt {
    fn register_dn_tree(&self, dn_cache: Arc<RwLock<DnSpineCache>>) -> Result<()>;
}

impl DnTreeExt for SessionContext {
    fn register_dn_tree(&self, dn_cache: Arc<RwLock<DnSpineCache>>) -> Result<()> {
        let provider = DnTreeTableProvider::new(dn_cache);
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
    use crate::container::Container;
    use crate::container::adjacency::PackedDn;
    use crate::container::dn_spine_cache::DnSpineCache;

    #[test]
    fn test_dn_tree_schema() {
        let schema = dn_tree_schema();
        assert_eq!(schema.fields().len(), 8);
        assert_eq!(schema.field(0).name(), "dn");
        assert_eq!(schema.field(1).name(), "is_spine");
        assert_eq!(schema.field(4).name(), "meta");
        assert_eq!(schema.field(5).name(), "content");
    }

    #[tokio::test]
    async fn test_dn_tree_provider_empty() {
        let cache = Arc::new(RwLock::new(DnSpineCache::new(16)));
        let provider = DnTreeTableProvider::new(cache);

        assert_eq!(provider.table_type(), TableType::Base);
        assert_eq!(provider.schema().fields().len(), 8);
    }

    #[tokio::test]
    async fn test_dn_tree_provider_with_data() {
        let mut cache = DnSpineCache::new(16);

        // Insert a few nodes
        let root = PackedDn::new(&[0]);
        let child1 = PackedDn::new(&[0, 1]);
        let child2 = PackedDn::new(&[0, 2]);

        cache.insert(root, &Container::random(1), &Container::random(10));
        cache.insert(child1, &Container::random(2), &Container::random(20));
        cache.insert(child2, &Container::random(3), &Container::random(30));
        cache.set_label(root, "root");

        let provider = Arc::new(DnTreeTableProvider::new(Arc::new(RwLock::new(cache))));
        let ctx = SessionContext::new();
        ctx.register_table("dn_tree", provider).unwrap();

        let df = ctx.sql("SELECT COUNT(*) as cnt FROM dn_tree").await.unwrap();
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
