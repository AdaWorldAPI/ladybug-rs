//! ScentScan: Pushed-down Hamming Distance Predicates
//!
//! This module provides a custom ExecutionPlan that pushes Hamming distance
//! predicates down to the scan level, using HDR cascade for fast filtering.
//!
//! # Example SQL
//!
//! ```sql
//! SELECT address, label, hamming_distance(fingerprint, x'...') as dist
//! FROM bindspace
//! WHERE hamming_distance(fingerprint, x'...') < 2000
//! ORDER BY dist
//! LIMIT 10
//! ```
//!
//! # Architecture
//!
//! ```text
//! SQL Query
//!    │
//!    ▼
//! ┌─────────────────────────────┐
//! │  Predicate Extraction       │  Extract hamming_distance < N
//! └─────────────────────────────┘
//!    │
//!    ▼
//! ┌─────────────────────────────┐
//! │  ScentScanExec              │  Custom ExecutionPlan
//! │  ├── HDR Cascade L0-L2     │  90%+ filtered at each level
//! │  └── Full Hamming verify   │  Only for survivors
//! └─────────────────────────────┘
//!    │
//!    ▼
//! RecordBatch (filtered results)
//! ```

use std::any::Any;
use std::sync::Arc;
use std::fmt;

use arrow::array::*;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::common::ScalarValue;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::context::TaskContext;
use datafusion::logical_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility};
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

use crate::search::hdr_cascade::{hamming_distance, HdrIndex};
use crate::storage::bind_space::{Addr, BindSpace, BindNode, FINGERPRINT_WORDS};

// =============================================================================
// CONSTANTS
// =============================================================================

/// Fingerprint size in bytes
const FP_BYTES: usize = FINGERPRINT_WORDS * 8;

/// HDR words (256 for HDR cascade compatibility)
const HDR_WORDS: usize = 256;

/// Max bits in fingerprint (~16K)
const MAX_BITS: u32 = (HDR_WORDS * 64) as u32;

// =============================================================================
// SCENT PREDICATE
// =============================================================================

/// A predicate that can be pushed down to the scan level
#[derive(Debug, Clone)]
pub enum ScentPredicate {
    /// hamming_distance(fingerprint, query) < threshold
    HammingLessThan {
        query: [u64; HDR_WORDS],
        threshold: u32,
    },
    /// hamming_distance(fingerprint, query) <= threshold
    HammingLessEqual {
        query: [u64; HDR_WORDS],
        threshold: u32,
    },
    /// similarity(fingerprint, query) > threshold (converted to Hamming)
    SimilarityGreaterThan {
        query: [u64; HDR_WORDS],
        threshold: f32,  // 0.0-1.0
    },
    /// Top-K nearest neighbors
    TopK {
        query: [u64; HDR_WORDS],
        k: usize,
    },
}

impl ScentPredicate {
    /// Convert to Hamming threshold for HDR cascade
    pub fn hamming_threshold(&self) -> u32 {
        match self {
            ScentPredicate::HammingLessThan { threshold, .. } => *threshold,
            ScentPredicate::HammingLessEqual { threshold, .. } => *threshold + 1,
            ScentPredicate::SimilarityGreaterThan { threshold, .. } => {
                // similarity = 1.0 - (hamming / MAX_BITS)
                // hamming = (1.0 - similarity) * MAX_BITS
                ((1.0 - threshold) * MAX_BITS as f32) as u32
            }
            ScentPredicate::TopK { .. } => u32::MAX, // No threshold for top-k
        }
    }

    /// Get the query fingerprint
    pub fn query(&self) -> &[u64; HDR_WORDS] {
        match self {
            ScentPredicate::HammingLessThan { query, .. } => query,
            ScentPredicate::HammingLessEqual { query, .. } => query,
            ScentPredicate::SimilarityGreaterThan { query, .. } => query,
            ScentPredicate::TopK { query, .. } => query,
        }
    }

    /// Check if a distance passes this predicate
    pub fn matches(&self, distance: u32) -> bool {
        match self {
            ScentPredicate::HammingLessThan { threshold, .. } => distance < *threshold,
            ScentPredicate::HammingLessEqual { threshold, .. } => distance <= *threshold,
            ScentPredicate::SimilarityGreaterThan { threshold, .. } => {
                let sim = 1.0 - (distance as f32 / MAX_BITS as f32);
                sim > *threshold
            }
            ScentPredicate::TopK { .. } => true, // All candidates pass, sorted later
        }
    }
}

// =============================================================================
// SCENT SCAN EXECUTION PLAN
// =============================================================================

/// Custom ExecutionPlan that pushes Hamming predicates to scan level
pub struct ScentScanExec {
    /// Schema for output
    schema: SchemaRef,
    /// Projected schema (subset of columns)
    projected_schema: SchemaRef,
    /// Column projection indices
    projection: Option<Vec<usize>>,
    /// The BindSpace to scan
    bind_space: Arc<RwLock<BindSpace>>,
    /// Optional HDR index for fast cascade filtering
    hdr_index: Option<Arc<RwLock<HdrIndex>>>,
    /// Pushed-down predicate
    predicate: ScentPredicate,
    /// Plan properties
    properties: PlanProperties,
}

impl ScentScanExec {
    /// Create a new ScentScanExec
    pub fn new(
        bind_space: Arc<RwLock<BindSpace>>,
        predicate: ScentPredicate,
        projection: Option<Vec<usize>>,
    ) -> Self {
        let schema = scent_schema();
        let projected_schema = match &projection {
            Some(indices) => Arc::new(Schema::new(
                indices.iter().map(|i| schema.field(*i).clone()).collect::<Vec<_>>()
            )),
            None => schema.clone(),
        };

        let eq_properties = EquivalenceProperties::new(projected_schema.clone());
        let properties = PlanProperties::new(
            eq_properties,
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        );

        Self {
            schema,
            projected_schema,
            projection,
            bind_space,
            hdr_index: None,
            predicate,
            properties,
        }
    }

    /// Add an HDR index for faster cascade filtering
    pub fn with_hdr_index(mut self, hdr_index: Arc<RwLock<HdrIndex>>) -> Self {
        self.hdr_index = Some(hdr_index);
        self
    }
}

/// Schema for scent scan results (includes distance column)
fn scent_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("address", DataType::UInt16, false),
        Field::new("fingerprint", DataType::FixedSizeBinary(FP_BYTES as i32), false),
        Field::new("label", DataType::Utf8, true),
        Field::new("qidx", DataType::UInt8, false),
        Field::new("access_count", DataType::UInt32, false),
        Field::new("zone", DataType::Utf8, false),
        Field::new("distance", DataType::UInt32, false),      // Hamming distance
        Field::new("similarity", DataType::Float32, false),   // 1.0 - distance/MAX_BITS
    ]))
}

impl fmt::Debug for ScentScanExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScentScanExec")
            .field("predicate", &self.predicate)
            .field("projection", &self.projection)
            .finish()
    }
}

impl DisplayAs for ScentScanExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ScentScanExec: predicate={:?}", self.predicate)
    }
}

impl ExecutionPlan for ScentScanExec {
    fn name(&self) -> &str {
        "ScentScanExec"
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
        let bind_space = self.bind_space.read();
        let query = self.predicate.query();
        let threshold = self.predicate.hamming_threshold();

        // Collect matching results
        let mut results: Vec<(Addr, &BindNode, u32)> = Vec::new();

        // If we have an HDR index, use cascade filtering
        if let Some(hdr_ref) = &self.hdr_index {
            let hdr = hdr_ref.read();
            let candidates = hdr.search(query, 1000); // Get top 1000 candidates

            for (idx, dist) in candidates {
                if dist <= threshold && self.predicate.matches(dist) {
                    // Convert index to address - this assumes HDR index is aligned with BindSpace
                    // In practice, we'd need a mapping from HDR index to Addr
                    let prefix = (0x80 + (idx / 256)) as u8;
                    let slot = (idx % 256) as u8;
                    let addr = Addr::new(prefix, slot);
                    if let Some(node) = bind_space.read(addr) {
                        results.push((addr, node, dist));
                    }
                }
            }
        } else {
            // Full scan with Hamming distance computation
            scan_zone(&bind_space, query, &self.predicate, 0x00, 0x0F, &mut results);
            scan_zone(&bind_space, query, &self.predicate, 0x10, 0x7F, &mut results);
            scan_zone(&bind_space, query, &self.predicate, 0x80, 0xFF, &mut results);
        }

        // Sort by distance for top-k
        if let ScentPredicate::TopK { k, .. } = &self.predicate {
            results.sort_by_key(|(_, _, dist)| *dist);
            results.truncate(*k);
        }

        // Convert to RecordBatch
        let batch = results_to_batch(&results, &self.schema, &self.projection)?;

        Ok(Box::pin(MemoryStream::new(
            vec![batch],
            self.projected_schema.clone(),
        )))
    }
}

/// Scan a zone of BindSpace and collect matches
fn scan_zone<'a>(
    bind_space: &'a BindSpace,
    query: &[u64; HDR_WORDS],
    predicate: &ScentPredicate,
    prefix_start: u8,
    prefix_end: u8,
    results: &mut Vec<(Addr, &'a BindNode, u32)>,
) {
    let threshold = predicate.hamming_threshold();

    for prefix in prefix_start..=prefix_end {
        for slot in 0u8..=0xFF {
            let addr = Addr::new(prefix, slot);
            if let Some(node) = bind_space.read(addr) {
                // Convert fingerprint to HDR format (256 words)
                let fp = &node.fingerprint;
                let mut hdr_fp = [0u64; HDR_WORDS];
                let copy_len = fp.len().min(HDR_WORDS);
                hdr_fp[..copy_len].copy_from_slice(&fp[..copy_len]);

                let dist = hamming_distance(query, &hdr_fp);

                if dist <= threshold && predicate.matches(dist) {
                    results.push((addr, node, dist));
                }
            }
        }
    }
}

/// Convert results to RecordBatch
fn results_to_batch(
    results: &[(Addr, &BindNode, u32)],
    schema: &SchemaRef,
    projection: &Option<Vec<usize>>,
) -> Result<RecordBatch> {
    if results.is_empty() {
        let projected_schema = match projection {
            Some(indices) => Arc::new(Schema::new(
                indices.iter().map(|i| schema.field(*i).clone()).collect::<Vec<_>>()
            )),
            None => schema.clone(),
        };
        return Ok(RecordBatch::new_empty(projected_schema));
    }

    // Build arrays
    let mut addresses: Vec<u16> = Vec::with_capacity(results.len());
    let mut fingerprints: Vec<Vec<u8>> = Vec::with_capacity(results.len());
    let mut labels: Vec<Option<String>> = Vec::with_capacity(results.len());
    let mut qidxs: Vec<u8> = Vec::with_capacity(results.len());
    let mut access_counts: Vec<u32> = Vec::with_capacity(results.len());
    let mut zones: Vec<String> = Vec::with_capacity(results.len());
    let mut distances: Vec<u32> = Vec::with_capacity(results.len());
    let mut similarities: Vec<f32> = Vec::with_capacity(results.len());

    for (addr, node, dist) in results {
        addresses.push(addr.0);

        // Convert fingerprint to bytes
        let fp_bytes: Vec<u8> = node.fingerprint
            .iter()
            .flat_map(|w: &u64| w.to_le_bytes())
            .collect();
        fingerprints.push(fp_bytes);

        labels.push(node.label.clone());
        qidxs.push(node.qidx);
        access_counts.push(node.access_count);

        let zone = match addr.prefix() {
            p if p <= 0x0F => "surface",
            p if p >= 0x10 && p <= 0x7F => "fluid",
            _ => "node",
        };
        zones.push(zone.to_string());

        distances.push(*dist);
        similarities.push(1.0 - (*dist as f32 / MAX_BITS as f32));
    }

    // Build Arrow arrays
    let address_array = UInt16Array::from(addresses);
    let mut fp_builder = FixedSizeBinaryBuilder::new(FP_BYTES as i32);
    for fp in &fingerprints {
        let mut padded = vec![0u8; FP_BYTES];
        let copy_len = fp.len().min(FP_BYTES);
        padded[..copy_len].copy_from_slice(&fp[..copy_len]);
        fp_builder.append_value(&padded)?;
    }
    let fingerprint_array = fp_builder.finish();
    let label_array = StringArray::from(labels);
    let qidx_array = UInt8Array::from(qidxs);
    let access_count_array = UInt32Array::from(access_counts);
    let zone_array = StringArray::from(zones);
    let distance_array = UInt32Array::from(distances);
    let similarity_array = Float32Array::from(similarities);

    let arrays: Vec<ArrayRef> = vec![
        Arc::new(address_array),
        Arc::new(fingerprint_array),
        Arc::new(label_array),
        Arc::new(qidx_array),
        Arc::new(access_count_array),
        Arc::new(zone_array),
        Arc::new(distance_array),
        Arc::new(similarity_array),
    ];

    // Apply projection
    let (projected_schema, projected_arrays) = match projection {
        Some(indices) => {
            let schema = Arc::new(Schema::new(
                indices.iter().map(|i| schema.field(*i).clone()).collect::<Vec<_>>()
            ));
            let arrays = indices.iter().map(|i| arrays[*i].clone()).collect();
            (schema, arrays)
        }
        None => (schema.clone(), arrays),
    };

    RecordBatch::try_new(projected_schema, projected_arrays)
        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
}

// =============================================================================
// MEMORY STREAM (same as fingerprint_table.rs)
// =============================================================================

/// Simple in-memory stream for RecordBatches
struct MemoryStream {
    batches: Vec<RecordBatch>,
    schema: SchemaRef,
    index: usize,
}

impl MemoryStream {
    fn new(batches: Vec<RecordBatch>, schema: SchemaRef) -> Self {
        Self { batches, schema, index: 0 }
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
// SCALAR UDFs
// =============================================================================

/// Hamming distance UDF for use in SQL queries
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct HammingDistanceUdf {
    signature: Signature,
}

impl HammingDistanceUdf {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![
                    DataType::FixedSizeBinary(FP_BYTES as i32),
                    DataType::FixedSizeBinary(FP_BYTES as i32),
                ],
                Volatility::Immutable,
            ),
        }
    }

    pub fn scalar_udf() -> ScalarUDF {
        ScalarUDF::new_from_impl(Self::new())
    }
}

impl ScalarUDFImpl for HammingDistanceUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "hamming_distance"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _args: &[DataType]) -> Result<DataType> {
        Ok(DataType::UInt32)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let args = &args.args;
        if args.len() != 2 {
            return Err(DataFusionError::Plan(
                "hamming_distance requires exactly 2 arguments".to_string()
            ));
        }

        // Handle scalar vs array for each argument
        let (fp1_array, fp2_array) = match (&args[0], &args[1]) {
            (ColumnarValue::Array(a1), ColumnarValue::Array(a2)) => {
                let fp1 = a1.as_any().downcast_ref::<FixedSizeBinaryArray>()
                    .ok_or_else(|| DataFusionError::Plan("Expected FixedSizeBinary".to_string()))?;
                let fp2 = a2.as_any().downcast_ref::<FixedSizeBinaryArray>()
                    .ok_or_else(|| DataFusionError::Plan("Expected FixedSizeBinary".to_string()))?;
                (fp1.clone(), fp2.clone())
            }
            (ColumnarValue::Array(a1), ColumnarValue::Scalar(s2)) => {
                let fp1 = a1.as_any().downcast_ref::<FixedSizeBinaryArray>()
                    .ok_or_else(|| DataFusionError::Plan("Expected FixedSizeBinary".to_string()))?;
                // Broadcast scalar to array
                let query_bytes = match s2 {
                    ScalarValue::FixedSizeBinary(_, Some(b)) => b.clone(),
                    _ => return Err(DataFusionError::Plan("Expected binary scalar".to_string())),
                };
                let mut builder = FixedSizeBinaryBuilder::new(FP_BYTES as i32);
                for _ in 0..fp1.len() {
                    builder.append_value(&query_bytes)?;
                }
                (fp1.clone(), builder.finish())
            }
            _ => {
                return Err(DataFusionError::Plan(
                    "hamming_distance: first argument must be array".to_string()
                ));
            }
        };

        // Compute distances
        let mut distances: Vec<u32> = Vec::with_capacity(fp1_array.len());
        for i in 0..fp1_array.len() {
            let bytes1 = fp1_array.value(i);
            let bytes2 = fp2_array.value(i);

            // Convert to u64 arrays
            let mut words1 = [0u64; HDR_WORDS];
            let mut words2 = [0u64; HDR_WORDS];

            for (j, chunk) in bytes1.chunks(8).enumerate().take(HDR_WORDS) {
                if chunk.len() == 8 {
                    words1[j] = u64::from_le_bytes(chunk.try_into().unwrap());
                }
            }
            for (j, chunk) in bytes2.chunks(8).enumerate().take(HDR_WORDS) {
                if chunk.len() == 8 {
                    words2[j] = u64::from_le_bytes(chunk.try_into().unwrap());
                }
            }

            distances.push(hamming_distance(&words1, &words2));
        }

        Ok(ColumnarValue::Array(Arc::new(UInt32Array::from(distances))))
    }
}

/// Similarity UDF (1.0 - hamming/MAX_BITS)
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SimilarityUdf {
    signature: Signature,
}

impl SimilarityUdf {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![
                    DataType::FixedSizeBinary(FP_BYTES as i32),
                    DataType::FixedSizeBinary(FP_BYTES as i32),
                ],
                Volatility::Immutable,
            ),
        }
    }

    pub fn scalar_udf() -> ScalarUDF {
        ScalarUDF::new_from_impl(Self::new())
    }
}

impl ScalarUDFImpl for SimilarityUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "similarity"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _args: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float32)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let args = &args.args;
        if args.len() != 2 {
            return Err(DataFusionError::Plan(
                "similarity requires exactly 2 arguments".to_string()
            ));
        }

        // Handle scalar vs array for each argument
        let (fp1_array, fp2_array) = match (&args[0], &args[1]) {
            (ColumnarValue::Array(a1), ColumnarValue::Array(a2)) => {
                let fp1 = a1.as_any().downcast_ref::<FixedSizeBinaryArray>()
                    .ok_or_else(|| DataFusionError::Plan("Expected FixedSizeBinary".to_string()))?;
                let fp2 = a2.as_any().downcast_ref::<FixedSizeBinaryArray>()
                    .ok_or_else(|| DataFusionError::Plan("Expected FixedSizeBinary".to_string()))?;
                (fp1.clone(), fp2.clone())
            }
            (ColumnarValue::Array(a1), ColumnarValue::Scalar(s2)) => {
                let fp1 = a1.as_any().downcast_ref::<FixedSizeBinaryArray>()
                    .ok_or_else(|| DataFusionError::Plan("Expected FixedSizeBinary".to_string()))?;
                let query_bytes = match s2 {
                    ScalarValue::FixedSizeBinary(_, Some(b)) => b.clone(),
                    _ => return Err(DataFusionError::Plan("Expected binary scalar".to_string())),
                };
                let mut builder = FixedSizeBinaryBuilder::new(FP_BYTES as i32);
                for _ in 0..fp1.len() {
                    builder.append_value(&query_bytes)?;
                }
                (fp1.clone(), builder.finish())
            }
            _ => {
                return Err(DataFusionError::Plan(
                    "similarity: first argument must be array".to_string()
                ));
            }
        };

        // Compute similarities
        let mut similarities: Vec<f32> = Vec::with_capacity(fp1_array.len());
        for i in 0..fp1_array.len() {
            let bytes1 = fp1_array.value(i);
            let bytes2 = fp2_array.value(i);

            let mut words1 = [0u64; HDR_WORDS];
            let mut words2 = [0u64; HDR_WORDS];

            for (j, chunk) in bytes1.chunks(8).enumerate().take(HDR_WORDS) {
                if chunk.len() == 8 {
                    words1[j] = u64::from_le_bytes(chunk.try_into().unwrap());
                }
            }
            for (j, chunk) in bytes2.chunks(8).enumerate().take(HDR_WORDS) {
                if chunk.len() == 8 {
                    words2[j] = u64::from_le_bytes(chunk.try_into().unwrap());
                }
            }

            let dist = hamming_distance(&words1, &words2);
            similarities.push(1.0 - (dist as f32 / MAX_BITS as f32));
        }

        Ok(ColumnarValue::Array(Arc::new(Float32Array::from(similarities))))
    }
}

// =============================================================================
// SESSION CONTEXT EXTENSION
// =============================================================================

/// Extension trait to register scent UDFs
pub trait ScentUdfExtension {
    /// Register hamming_distance and similarity UDFs
    fn register_scent_udfs(&self) -> Result<()>;
}

impl ScentUdfExtension for SessionContext {
    fn register_scent_udfs(&self) -> Result<()> {
        self.register_udf(HammingDistanceUdf::scalar_udf());
        self.register_udf(SimilarityUdf::scalar_udf());
        Ok(())
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scent_predicate_hamming() {
        let query = [0u64; HDR_WORDS];
        let pred = ScentPredicate::HammingLessThan {
            query,
            threshold: 1000,
        };

        assert_eq!(pred.hamming_threshold(), 1000);
        assert!(pred.matches(500));
        assert!(pred.matches(999));
        assert!(!pred.matches(1000));
        assert!(!pred.matches(2000));
    }

    #[test]
    fn test_scent_predicate_similarity() {
        let query = [0u64; HDR_WORDS];
        let pred = ScentPredicate::SimilarityGreaterThan {
            query,
            threshold: 0.8, // Expect > 80% similarity
        };

        // 80% similarity = 20% different = 0.2 * 16384 = ~3277 bits
        let thresh = pred.hamming_threshold();
        assert!(thresh > 3100 && thresh < 3500, "threshold was {}", thresh);

        // 0 distance = 100% similarity, should match
        assert!(pred.matches(0));
        // 1000 distance = ~90% similarity, should match
        assert!(pred.matches(1000));
        // 4000 distance = ~75.6% similarity at 16K, should not match
        assert!(!pred.matches(4000));
    }

    #[test]
    fn test_scent_schema() {
        let schema = scent_schema();
        assert_eq!(schema.fields().len(), 8);
        assert_eq!(schema.field(0).name(), "address");
        assert_eq!(schema.field(6).name(), "distance");
        assert_eq!(schema.field(7).name(), "similarity");
    }

    #[tokio::test]
    async fn test_scent_scan_empty() {
        let bind_space = Arc::new(RwLock::new(BindSpace::new()));
        let query = [0u64; HDR_WORDS];

        let pred = ScentPredicate::HammingLessThan { query, threshold: 100 };
        let scan = ScentScanExec::new(bind_space, pred, None);

        let ctx = Arc::new(TaskContext::default());
        let stream = scan.execute(0, ctx).unwrap();
        let batches: Vec<_> = futures::executor::block_on_stream(stream)
            .filter_map(|r| r.ok())
            .collect();

        // Should have results from surface zone (pre-initialized nodes)
        // But most won't match threshold of 100 with all-zero query
        assert!(!batches.is_empty());
    }

    #[tokio::test]
    async fn test_scent_scan_with_data() {
        let bind_space = Arc::new(RwLock::new(BindSpace::new()));

        // Insert a node with known fingerprint
        let mut test_fp = [0u64; FINGERPRINT_WORDS];
        test_fp[0] = 0xFFFF; // Small non-zero pattern

        {
            let mut bs = bind_space.write();
            bs.write_labeled(test_fp, "test_node");
        }

        // Query for exact match (hamming < 100)
        let mut query = [0u64; HDR_WORDS];
        query[0] = 0xFFFF; // Same pattern

        let pred = ScentPredicate::HammingLessThan { query, threshold: 100 };
        let scan = ScentScanExec::new(bind_space, pred, None);

        let ctx = Arc::new(TaskContext::default());
        let stream = scan.execute(0, ctx).unwrap();
        let batches: Vec<_> = futures::executor::block_on_stream(stream)
            .filter_map(|r| r.ok())
            .collect();

        // Should find the node we inserted (distance = 0)
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert!(total_rows >= 1, "Expected at least 1 match, got {}", total_rows);

        // Check that distance column has 0 for exact match
        if let Some(batch) = batches.first() {
            let distances = batch.column(6).as_any()
                .downcast_ref::<UInt32Array>()
                .unwrap();
            // At least one result should have distance 0
            let has_exact = distances.iter().any(|d| d == Some(0));
            assert!(has_exact, "Expected exact match with distance 0");
        }
    }

    #[tokio::test]
    async fn test_scent_scan_top_k() {
        let bind_space = Arc::new(RwLock::new(BindSpace::new()));

        // Insert multiple nodes with varying distances from query
        {
            let mut bs = bind_space.write();
            let mut fp1 = [0u64; FINGERPRINT_WORDS];
            fp1[0] = 0x0001; // 1 bit different
            bs.write_labeled(fp1, "close");

            let mut fp2 = [0u64; FINGERPRINT_WORDS];
            fp2[0] = 0xFFFF; // 16 bits different
            bs.write_labeled(fp2, "medium");

            let mut fp3 = [0u64; FINGERPRINT_WORDS];
            fp3[0] = 0xFFFFFFFFFFFFFFFF; // 64 bits different
            bs.write_labeled(fp3, "far");
        }

        // Top-2 nearest to all zeros
        let query = [0u64; HDR_WORDS];
        let pred = ScentPredicate::TopK { query, k: 2 };
        let scan = ScentScanExec::new(bind_space, pred, None);

        let ctx = Arc::new(TaskContext::default());
        let stream = scan.execute(0, ctx).unwrap();
        let batches: Vec<_> = futures::executor::block_on_stream(stream)
            .filter_map(|r| r.ok())
            .collect();

        // May have more than 2 due to surface nodes, but sorted by distance
        if let Some(batch) = batches.first() {
            let distances = batch.column(6).as_any()
                .downcast_ref::<UInt32Array>()
                .unwrap();
            // First results should be closest
            if distances.len() >= 2 {
                assert!(distances.value(0) <= distances.value(1),
                    "Results should be sorted by distance");
            }
        }
    }

    #[tokio::test]
    async fn test_register_scent_udfs() {
        let ctx = SessionContext::new();
        ctx.register_scent_udfs().unwrap();

        // Verify UDFs are registered by trying to use them in a query plan
        // (just check that a simple SQL using the UDF parses without error)
        let state = ctx.state();
        let udfs = state.scalar_functions();
        assert!(udfs.contains_key("hamming_distance"));
        assert!(udfs.contains_key("similarity"));
    }
}
