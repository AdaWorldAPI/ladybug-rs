//! Lance-backed persistence for BindSpace.
//!
//! This module provides durable write-through persistence for the BindSpace
//! using Lance columnar format. The BindSpace remains the hot-path for reads;
//! Lance is the durable ground truth that survives restarts.
//!
//! # Schema
//!
//! **bind_nodes table**: One row per occupied BindSpace address.
//! ```text
//!   addr         u16            — BindSpace address (prefix:slot)
//!   fingerprint  FixedSizeBinary(2048) — 256 × u64, little-endian
//!   label        Utf8 nullable  — human-readable label
//!   qidx         u8             — qualia index
//!   parent       u16 nullable   — parent address (0 = None)
//!   depth        u8             — tree depth
//!   rung         u8             — access rung
//!   sigma        u8             — reasoning depth
//!   is_spine     bool           — cluster centroid flag
//!   dn_path      u64 nullable   — PackedDn (7-level hierarchical path)
//!   payload      LargeBinary nullable — arbitrary payload bytes
//! ```
//!
//! **bind_edges table**: One row per BindEdge.
//! ```text
//!   from_addr    u16            — source address
//!   to_addr      u16            — target address
//!   verb_addr    u16            — verb address
//!   fingerprint  FixedSizeBinary(2048) — edge XOR fingerprint
//!   weight       f32            — edge weight
//! ```
//!
//! **bind_state table**: Single-row metadata.
//! ```text
//!   next_node_prefix  u8
//!   next_node_slot    u8
//!   next_fluid_prefix u8
//!   next_fluid_slot   u8
//! ```
//!
//! # Design
//!
//! - Write-through: every mutation to BindSpace also goes to Lance
//! - Hydrate on startup: Lance → BindSpace on server init
//! - Graceful degradation: if Lance fails, log error, keep running in-memory

use arrow::array::*;
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use arrow::record_batch::RecordBatch;
use lance::Dataset;
use lance::dataset::write::{WriteMode, WriteParams};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::bind_space::{
    Addr, BindEdge, BindNode, BindSpace, FINGERPRINT_WORDS, PREFIX_SURFACE_END,
};
use crate::container::adjacency::PackedDn;

/// Fingerprint bytes (256 u64 × 8 bytes = 2048)
const FP_BYTES: i32 = (FINGERPRINT_WORDS * 8) as i32;

// =============================================================================
// SCHEMAS
// =============================================================================

fn bind_nodes_schema() -> ArrowSchema {
    ArrowSchema::new(vec![
        Field::new("addr", DataType::UInt16, false),
        Field::new("fingerprint", DataType::FixedSizeBinary(FP_BYTES), false),
        Field::new("label", DataType::Utf8, true),
        Field::new("qidx", DataType::UInt8, false),
        Field::new("parent", DataType::UInt16, true),
        Field::new("depth", DataType::UInt8, false),
        Field::new("rung", DataType::UInt8, false),
        Field::new("sigma", DataType::UInt8, false),
        Field::new("is_spine", DataType::Boolean, false),
        Field::new("dn_path", DataType::UInt64, true),
        Field::new("payload", DataType::LargeBinary, true),
    ])
}

fn bind_edges_schema() -> ArrowSchema {
    ArrowSchema::new(vec![
        Field::new("from_addr", DataType::UInt16, false),
        Field::new("to_addr", DataType::UInt16, false),
        Field::new("verb_addr", DataType::UInt16, false),
        Field::new("fingerprint", DataType::FixedSizeBinary(FP_BYTES), false),
        Field::new("weight", DataType::Float32, false),
    ])
}

fn bind_state_schema() -> ArrowSchema {
    ArrowSchema::new(vec![
        Field::new("next_node_prefix", DataType::UInt8, false),
        Field::new("next_node_slot", DataType::UInt8, false),
        Field::new("next_fluid_prefix", DataType::UInt8, false),
        Field::new("next_fluid_slot", DataType::UInt8, false),
    ])
}

// =============================================================================
// HELPERS
// =============================================================================

/// Wrap a RecordBatch as a RecordBatchReader for Lance write API.
fn batch_reader(
    batch: RecordBatch,
) -> RecordBatchIterator<
    std::vec::IntoIter<std::result::Result<RecordBatch, arrow::error::ArrowError>>,
> {
    let schema = batch.schema();
    RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema)
}

/// Convert [u64; FINGERPRINT_WORDS] to little-endian bytes.
fn fp_to_le_bytes(fp: &[u64; FINGERPRINT_WORDS]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(FINGERPRINT_WORDS * 8);
    for &word in fp {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

/// Convert little-endian bytes back to [u64; FINGERPRINT_WORDS].
fn fp_from_le_bytes(bytes: &[u8]) -> [u64; FINGERPRINT_WORDS] {
    let mut fp = [0u64; FINGERPRINT_WORDS];
    for (i, chunk) in bytes.chunks_exact(8).enumerate() {
        fp[i] = u64::from_le_bytes(chunk.try_into().unwrap());
    }
    fp
}

// =============================================================================
// LANCE PERSISTENCE
// =============================================================================

/// Durable persistence layer bridging BindSpace ↔ Lance.
///
/// Write-through: in-memory stays hot, Lance is ground truth.
/// Hydrate: on startup, load Lance → BindSpace.
pub struct LancePersistence {
    /// Path to lance data directory
    data_dir: PathBuf,
    /// Whether persistence is active (false if init failed)
    active: bool,
}

impl LancePersistence {
    /// Create persistence layer. Creates data directory if needed.
    pub fn new(data_dir: impl AsRef<Path>) -> Self {
        let data_dir = data_dir.as_ref().to_path_buf();
        let active = match std::fs::create_dir_all(&data_dir) {
            Ok(()) => true,
            Err(e) => {
                eprintln!("[lance-persist] Failed to create data dir: {}", e);
                false
            }
        };
        Self { data_dir, active }
    }

    /// Path to the nodes lance dataset
    fn nodes_path(&self) -> PathBuf {
        self.data_dir.join("bind_nodes.lance")
    }

    /// Path to the edges lance dataset
    fn edges_path(&self) -> PathBuf {
        self.data_dir.join("bind_edges.lance")
    }

    /// Path to the state lance dataset
    fn state_path(&self) -> PathBuf {
        self.data_dir.join("bind_state.lance")
    }

    /// Check if persisted data exists on disk.
    pub fn has_data(&self) -> bool {
        self.nodes_path().exists()
    }

    // =========================================================================
    // PHASE 2: TABLE CREATION
    // =========================================================================

    /// Ensure all Lance tables exist (create empty if needed).
    /// Called on startup before any reads/writes.
    pub async fn ensure_tables(&self) -> Result<(), String> {
        if !self.active {
            return Err("Persistence not active".into());
        }

        // Create nodes table if missing
        let nodes_path = self.nodes_path();
        if !nodes_path.exists() {
            let schema = Arc::new(bind_nodes_schema());
            let batch = RecordBatch::new_empty(schema);
            Dataset::write(batch_reader(batch), nodes_path.to_str().unwrap(), None)
                .await
                .map_err(|e| format!("Failed to create bind_nodes table: {}", e))?;
            eprintln!("[lance-persist] Created bind_nodes table");
        }

        // Create edges table if missing
        let edges_path = self.edges_path();
        if !edges_path.exists() {
            let schema = Arc::new(bind_edges_schema());
            let batch = RecordBatch::new_empty(schema);
            Dataset::write(batch_reader(batch), edges_path.to_str().unwrap(), None)
                .await
                .map_err(|e| format!("Failed to create bind_edges table: {}", e))?;
            eprintln!("[lance-persist] Created bind_edges table");
        }

        // Create state table if missing
        let state_path = self.state_path();
        if !state_path.exists() {
            let schema = Arc::new(bind_state_schema());
            let batch = RecordBatch::new_empty(schema);
            Dataset::write(batch_reader(batch), state_path.to_str().unwrap(), None)
                .await
                .map_err(|e| format!("Failed to create bind_state table: {}", e))?;
            eprintln!("[lance-persist] Created bind_state table");
        }

        Ok(())
    }

    // =========================================================================
    // PHASE 3+4: WRITE-THROUGH (FULL SNAPSHOT)
    // =========================================================================

    /// Persist the entire BindSpace to Lance (full snapshot).
    ///
    /// Overwrites all tables. Used for:
    /// - Initial persistence after first population
    /// - Periodic checkpoints
    /// - Graceful shutdown
    pub async fn persist_full(&self, space: &BindSpace) -> Result<(), String> {
        if !self.active {
            return Ok(());
        }

        self.persist_nodes(space).await?;
        self.persist_edges(space).await?;
        self.persist_state(space).await?;

        Ok(())
    }

    /// Persist all occupied nodes.
    async fn persist_nodes(&self, space: &BindSpace) -> Result<(), String> {
        let nodes_path = self.nodes_path();

        // Collect all occupied nodes
        let mut addrs = Vec::new();
        let mut fps = Vec::new();
        let mut labels: Vec<Option<String>> = Vec::new();
        let mut qidxs = Vec::new();
        let mut parents: Vec<Option<u16>> = Vec::new();
        let mut depths = Vec::new();
        let mut rungs = Vec::new();
        let mut sigmas = Vec::new();
        let mut spines = Vec::new();
        let mut dn_paths: Vec<Option<u64>> = Vec::new();
        let mut payloads: Vec<Option<Vec<u8>>> = Vec::new();

        for (addr, node) in space.nodes_iter() {
            // Skip zero fingerprints in surface area (init-generated)
            if addr.prefix() <= PREFIX_SURFACE_END {
                continue;
            }

            addrs.push(addr.0);
            fps.push(fp_to_le_bytes(&node.fingerprint));
            labels.push(node.label.clone());
            qidxs.push(node.qidx);
            parents.push(node.parent.map(|a| a.0));
            depths.push(node.depth);
            rungs.push(node.rung);
            sigmas.push(node.sigma);
            spines.push(node.is_spine);
            dn_paths.push(space.dn_index.dn_for(addr).map(|dn| dn.0));
            payloads.push(node.payload.clone());
        }

        if addrs.is_empty() {
            // Write empty table to clear previous data
            let schema = Arc::new(bind_nodes_schema());
            let batch = RecordBatch::new_empty(schema);
            Dataset::write(batch_reader(batch), nodes_path.to_str().unwrap(), None)
                .await
                .map_err(|e| format!("persist_nodes empty: {}", e))?;
            return Ok(());
        }

        // Build Arrow arrays
        let addr_arr = UInt16Array::from(addrs);

        let mut fp_builder = FixedSizeBinaryBuilder::new(FP_BYTES);
        for fp in &fps {
            fp_builder
                .append_value(fp)
                .map_err(|e| format!("fp append: {}", e))?;
        }
        let fp_arr = fp_builder.finish();

        let label_arr: StringArray = labels
            .iter()
            .map(|l| l.as_deref())
            .collect();
        let qidx_arr = UInt8Array::from(qidxs);

        let parent_arr: UInt16Array = parents
            .iter()
            .map(|p| *p)
            .collect();
        let depth_arr = UInt8Array::from(depths);
        let rung_arr = UInt8Array::from(rungs);
        let sigma_arr = UInt8Array::from(sigmas);
        let spine_arr = BooleanArray::from(spines);

        let dn_arr: UInt64Array = dn_paths
            .iter()
            .map(|d| *d)
            .collect();

        let payload_arr: LargeBinaryArray = payloads
            .iter()
            .map(|p| p.as_deref())
            .collect();

        let schema = Arc::new(bind_nodes_schema());
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(addr_arr),
                Arc::new(fp_arr),
                Arc::new(label_arr),
                Arc::new(qidx_arr),
                Arc::new(parent_arr),
                Arc::new(depth_arr),
                Arc::new(rung_arr),
                Arc::new(sigma_arr),
                Arc::new(spine_arr),
                Arc::new(dn_arr),
                Arc::new(payload_arr),
            ],
        )
        .map_err(|e| format!("batch build: {}", e))?;

        // Overwrite (full snapshot)
        let mut params = WriteParams::default();
        params.mode = WriteMode::Overwrite;
        Dataset::write(
            batch_reader(batch),
            nodes_path.to_str().unwrap(),
            Some(params),
        )
        .await
        .map_err(|e| format!("persist_nodes write: {}", e))?;

        Ok(())
    }

    /// Persist all edges.
    async fn persist_edges(&self, space: &BindSpace) -> Result<(), String> {
        let edges_path = self.edges_path();
        let edges: Vec<&BindEdge> = space.edges_iter().collect();

        if edges.is_empty() {
            let schema = Arc::new(bind_edges_schema());
            let batch = RecordBatch::new_empty(schema);
            Dataset::write(batch_reader(batch), edges_path.to_str().unwrap(), None)
                .await
                .map_err(|e| format!("persist_edges empty: {}", e))?;
            return Ok(());
        }

        let from_arr = UInt16Array::from(edges.iter().map(|e| e.from.0).collect::<Vec<_>>());
        let to_arr = UInt16Array::from(edges.iter().map(|e| e.to.0).collect::<Vec<_>>());
        let verb_arr = UInt16Array::from(edges.iter().map(|e| e.verb.0).collect::<Vec<_>>());

        let mut fp_builder = FixedSizeBinaryBuilder::new(FP_BYTES);
        for edge in &edges {
            let bytes = fp_to_le_bytes(&edge.fingerprint);
            fp_builder
                .append_value(&bytes)
                .map_err(|e| format!("edge fp append: {}", e))?;
        }
        let fp_arr = fp_builder.finish();

        let weight_arr = Float32Array::from(edges.iter().map(|e| e.weight).collect::<Vec<_>>());

        let schema = Arc::new(bind_edges_schema());
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(from_arr),
                Arc::new(to_arr),
                Arc::new(verb_arr),
                Arc::new(fp_arr),
                Arc::new(weight_arr),
            ],
        )
        .map_err(|e| format!("edge batch: {}", e))?;

        let mut params = WriteParams::default();
        params.mode = WriteMode::Overwrite;
        Dataset::write(
            batch_reader(batch),
            edges_path.to_str().unwrap(),
            Some(params),
        )
        .await
        .map_err(|e| format!("persist_edges write: {}", e))?;

        Ok(())
    }

    /// Persist allocator state (next_node, next_fluid pointers).
    async fn persist_state(&self, space: &BindSpace) -> Result<(), String> {
        let state_path = self.state_path();
        let (np, ns) = space.next_node_slot();
        let (fp, fs) = space.next_fluid_slot();

        let schema = Arc::new(bind_state_schema());
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt8Array::from(vec![np])),
                Arc::new(UInt8Array::from(vec![ns])),
                Arc::new(UInt8Array::from(vec![fp])),
                Arc::new(UInt8Array::from(vec![fs])),
            ],
        )
        .map_err(|e| format!("state batch: {}", e))?;

        let mut params = WriteParams::default();
        params.mode = WriteMode::Overwrite;
        Dataset::write(
            batch_reader(batch),
            state_path.to_str().unwrap(),
            Some(params),
        )
        .await
        .map_err(|e| format!("persist_state write: {}", e))?;

        Ok(())
    }

    // =========================================================================
    // PHASE 5: HYDRATION (Lance → BindSpace on startup)
    // =========================================================================

    /// Hydrate a BindSpace from Lance data.
    ///
    /// Returns the populated BindSpace, or None if no data exists.
    pub async fn hydrate(&self) -> Result<Option<BindSpace>, String> {
        if !self.active || !self.has_data() {
            return Ok(None);
        }

        eprintln!("[lance-persist] Hydrating BindSpace from Lance...");

        let mut space = BindSpace::new();

        // 1. Load state (allocator pointers)
        self.hydrate_state(&mut space).await?;

        // 2. Load nodes
        let node_count = self.hydrate_nodes(&mut space).await?;

        // 3. Load edges
        let edge_count = self.hydrate_edges(&mut space).await?;

        eprintln!(
            "[lance-persist] Hydrated {} nodes, {} edges",
            node_count, edge_count
        );

        Ok(Some(space))
    }

    /// Hydrate allocator state.
    async fn hydrate_state(&self, space: &mut BindSpace) -> Result<(), String> {
        let state_path = self.state_path();
        if !state_path.exists() {
            return Ok(());
        }

        let dataset = Dataset::open(state_path.to_str().unwrap())
            .await
            .map_err(|e| format!("open state: {}", e))?;

        let batches = self.scan_all(&dataset).await?;
        if batches.is_empty() || batches[0].num_rows() == 0 {
            return Ok(());
        }

        let batch = &batches[0];
        let np = batch.column(0).as_any().downcast_ref::<UInt8Array>().unwrap().value(0);
        let ns = batch.column(1).as_any().downcast_ref::<UInt8Array>().unwrap().value(0);
        let fp = batch.column(2).as_any().downcast_ref::<UInt8Array>().unwrap().value(0);
        let fs = batch.column(3).as_any().downcast_ref::<UInt8Array>().unwrap().value(0);

        space.set_next_node_slot(np, ns);
        space.set_next_fluid_slot(fp, fs);

        Ok(())
    }

    /// Hydrate nodes from Lance.
    async fn hydrate_nodes(&self, space: &mut BindSpace) -> Result<usize, String> {
        let nodes_path = self.nodes_path();
        if !nodes_path.exists() {
            return Ok(0);
        }

        let dataset = Dataset::open(nodes_path.to_str().unwrap())
            .await
            .map_err(|e| format!("open nodes: {}", e))?;

        let batches = self.scan_all(&dataset).await?;
        let mut count = 0usize;

        for batch in &batches {
            let addr_col = batch.column(0).as_any().downcast_ref::<UInt16Array>().unwrap();
            let fp_col = batch.column(1).as_any().downcast_ref::<FixedSizeBinaryArray>().unwrap();
            let label_col = batch.column(2).as_any().downcast_ref::<StringArray>().unwrap();
            let qidx_col = batch.column(3).as_any().downcast_ref::<UInt8Array>().unwrap();
            let parent_col = batch.column(4).as_any().downcast_ref::<UInt16Array>().unwrap();
            let depth_col = batch.column(5).as_any().downcast_ref::<UInt8Array>().unwrap();
            let rung_col = batch.column(6).as_any().downcast_ref::<UInt8Array>().unwrap();
            let sigma_col = batch.column(7).as_any().downcast_ref::<UInt8Array>().unwrap();
            let spine_col = batch.column(8).as_any().downcast_ref::<BooleanArray>().unwrap();
            let dn_col = batch.column(9).as_any().downcast_ref::<UInt64Array>().unwrap();
            let payload_col = batch.column(10).as_any().downcast_ref::<LargeBinaryArray>().unwrap();

            for row in 0..batch.num_rows() {
                let addr = Addr(addr_col.value(row));
                let fp = fp_from_le_bytes(fp_col.value(row));

                let mut node = BindNode::new(fp);

                if !label_col.is_null(row) {
                    node.label = Some(label_col.value(row).to_string());
                }
                node.qidx = qidx_col.value(row);
                if !parent_col.is_null(row) {
                    node.parent = Some(Addr(parent_col.value(row)));
                }
                node.depth = depth_col.value(row);
                node.rung = rung_col.value(row);
                node.sigma = sigma_col.value(row);
                node.is_spine = spine_col.value(row);
                if !payload_col.is_null(row) {
                    node.payload = Some(payload_col.value(row).to_vec());
                }

                // Write at the exact address
                space.write_at(addr, node.fingerprint);
                // Then apply metadata
                if let Some(written) = space.read_mut(addr) {
                    written.label = node.label;
                    written.qidx = node.qidx;
                    written.parent = node.parent;
                    written.depth = node.depth;
                    written.rung = node.rung;
                    written.sigma = node.sigma;
                    written.is_spine = node.is_spine;
                    written.payload = node.payload;
                }

                // Restore DN index
                if !dn_col.is_null(row) {
                    let dn = PackedDn(dn_col.value(row));
                    space.dn_index.register(dn, addr);
                }

                count += 1;
            }
        }

        // Clear dirty bits since we just loaded clean data
        space.clear_dirty();

        Ok(count)
    }

    /// Hydrate edges from Lance.
    async fn hydrate_edges(&self, space: &mut BindSpace) -> Result<usize, String> {
        let edges_path = self.edges_path();
        if !edges_path.exists() {
            return Ok(0);
        }

        let dataset = Dataset::open(edges_path.to_str().unwrap())
            .await
            .map_err(|e| format!("open edges: {}", e))?;

        let batches = self.scan_all(&dataset).await?;
        let mut count = 0usize;

        for batch in &batches {
            let from_col = batch.column(0).as_any().downcast_ref::<UInt16Array>().unwrap();
            let to_col = batch.column(1).as_any().downcast_ref::<UInt16Array>().unwrap();
            let verb_col = batch.column(2).as_any().downcast_ref::<UInt16Array>().unwrap();
            let fp_col = batch.column(3).as_any().downcast_ref::<FixedSizeBinaryArray>().unwrap();
            let weight_col = batch.column(4).as_any().downcast_ref::<Float32Array>().unwrap();

            for row in 0..batch.num_rows() {
                let from = Addr(from_col.value(row));
                let to = Addr(to_col.value(row));
                let verb = Addr(verb_col.value(row));
                let fp = fp_from_le_bytes(fp_col.value(row));
                let weight = weight_col.value(row);

                let mut edge = BindEdge::new(from, verb, to);
                edge.fingerprint = fp;
                edge.weight = weight;

                space.link_with_edge(edge);
                count += 1;
            }
        }

        Ok(count)
    }

    /// Helper: scan all rows from a Dataset into RecordBatches.
    async fn scan_all(&self, dataset: &Dataset) -> Result<Vec<RecordBatch>, String> {
        let stream = dataset
            .scan()
            .try_into_stream()
            .await
            .map_err(|e| format!("scan: {}", e))?;

        use futures::StreamExt;
        let mut batches = Vec::new();
        let mut stream = stream;
        while let Some(batch) = stream.next().await {
            batches.push(batch.map_err(|e| format!("batch read: {}", e))?);
        }
        Ok(batches)
    }

    /// Whether persistence is operational.
    pub fn is_active(&self) -> bool {
        self.active
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_test_space() -> BindSpace {
        let mut space = BindSpace::new();
        let fp1 = [42u64; FINGERPRINT_WORDS];
        let fp2 = [99u64; FINGERPRINT_WORDS];
        let fp3 = [7u64; FINGERPRINT_WORDS];

        let a = space.write_labeled(fp1, "node_alpha");
        let b = space.write_labeled(fp2, "node_beta");
        let _c = space.write(fp3);

        // Link a → b via verb CAUSES (0x07:0x00)
        let causes = Addr::new(0x07, 0x00);
        space.link(a, causes, b);

        space
    }

    #[tokio::test]
    async fn test_ensure_tables() {
        let tmp = TempDir::new().unwrap();
        let persist = LancePersistence::new(tmp.path().join("lance"));

        persist.ensure_tables().await.unwrap();

        assert!(persist.nodes_path().exists());
        assert!(persist.edges_path().exists());
        assert!(persist.state_path().exists());
    }

    #[tokio::test]
    async fn test_persist_and_hydrate_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let persist = LancePersistence::new(tmp.path().join("lance"));

        // Create and persist
        let original = make_test_space();
        persist.ensure_tables().await.unwrap();
        persist.persist_full(&original).await.unwrap();

        // Hydrate into new space
        let hydrated = persist.hydrate().await.unwrap().unwrap();

        // Verify node count matches (excluding surface nodes)
        let orig_nodes: Vec<_> = original
            .nodes_iter()
            .filter(|(a, _)| a.prefix() > PREFIX_SURFACE_END)
            .collect();
        let hydr_nodes: Vec<_> = hydrated
            .nodes_iter()
            .filter(|(a, _)| a.prefix() > PREFIX_SURFACE_END)
            .collect();
        assert_eq!(
            orig_nodes.len(),
            hydr_nodes.len(),
            "Node count mismatch: {} vs {}",
            orig_nodes.len(),
            hydr_nodes.len()
        );

        // Verify edge count
        assert_eq!(
            original.edges_iter().count(),
            hydrated.edges_iter().count(),
            "Edge count mismatch"
        );

        // Verify fingerprint round-trip for first node
        for (addr, orig_node) in &orig_nodes {
            if let Some(hydr_node) = hydrated.read(*addr) {
                assert_eq!(
                    orig_node.fingerprint, hydr_node.fingerprint,
                    "Fingerprint mismatch at {:?}",
                    addr
                );
                assert_eq!(orig_node.label, hydr_node.label, "Label mismatch at {:?}", addr);
            } else {
                panic!("Node at {:?} not found in hydrated space", addr);
            }
        }
    }

    #[tokio::test]
    async fn test_fingerprint_byte_roundtrip() {
        // Verify the critical path: fp → bytes → lance → bytes → fp
        let original = [0xDEADBEEFCAFEBABEu64; FINGERPRINT_WORDS];
        let bytes = fp_to_le_bytes(&original);
        assert_eq!(bytes.len(), FINGERPRINT_WORDS * 8);

        let recovered = fp_from_le_bytes(&bytes);
        assert_eq!(original, recovered);
    }
}
