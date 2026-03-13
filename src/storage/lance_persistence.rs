//! Lance-backed persistence for BindSpace (lancedb SDK 0.26).
//!
//! This module provides durable write-through persistence for the BindSpace
//! using LanceDB. The BindSpace remains the hot-path for reads;
//! LanceDB is the durable ground truth that survives restarts.
//!
//! Versioning comes for free: every write creates a new version (MVCC).
//! Time travel: checkout(version_n) reads any previous state.
//! Compaction: optimize() merges small versions into larger ones.
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
//! - Write-through: every mutation to BindSpace also goes to LanceDB
//! - Hydrate on startup: LanceDB → BindSpace on server init
//! - Graceful degradation: if LanceDB fails, log error, keep running in-memory
//! - Automatic versioning: every persist_full creates a new version

use arrow::array::*;
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use arrow::record_batch::RecordBatch;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::table::AddDataMode;
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
        Field::new("updated_at", DataType::UInt64, false),
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

/// Schema for the HTTP index fingerprints Vec.
fn index_fingerprints_schema() -> ArrowSchema {
    ArrowSchema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("fingerprint", DataType::FixedSizeBinary(FP_BYTES), false),
        Field::new("metadata", DataType::Utf8, true), // JSON-encoded HashMap
    ])
}

// =============================================================================
// HELPERS
// =============================================================================

/// Wrap a RecordBatch as a RecordBatchReader for lancedb write API.
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

/// Durable persistence layer bridging BindSpace ↔ LanceDB.
///
/// Write-through: in-memory stays hot, LanceDB is ground truth.
/// Hydrate: on startup, load LanceDB → BindSpace.
/// Versioning: every write creates a new version (MVCC for free).
pub struct LancePersistence {
    /// Path to lancedb data directory
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

    /// Connect to the LanceDB database.
    /// lancedb::connect is cheap — it doesn't open files until table operations.
    async fn connection(&self) -> Result<lancedb::Connection, String> {
        let path = self.data_dir.to_string_lossy().to_string();
        lancedb::connect(&path)
            .execute()
            .await
            .map_err(|e| format!("lancedb connect: {}", e))
    }

    /// Check if persisted data exists on disk.
    pub fn has_data(&self) -> bool {
        // LanceDB stores tables as subdirectories; check for bind_nodes
        self.data_dir.join("bind_nodes.lance").exists()
    }

    /// Open or create a named table.
    async fn open_or_create_table(
        &self,
        name: &str,
        schema: ArrowSchema,
    ) -> Result<lancedb::Table, String> {
        let db = self.connection().await?;
        match db.open_table(name).execute().await {
            Ok(table) => Ok(table),
            Err(lancedb::Error::TableNotFound { .. }) => {
                let batch = RecordBatch::new_empty(Arc::new(schema));
                db.create_table(name, batch_reader(batch))
                    .execute()
                    .await
                    .map_err(|e| format!("create table {}: {}", name, e))
            }
            Err(e) => Err(format!("open table {}: {}", name, e)),
        }
    }

    // =========================================================================
    // PHASE 2: TABLE CREATION
    // =========================================================================

    /// Ensure all LanceDB tables exist (create empty if needed).
    /// Called on startup before any reads/writes.
    pub async fn ensure_tables(&self) -> Result<(), String> {
        if !self.active {
            return Err("Persistence not active".into());
        }

        self.open_or_create_table("bind_nodes", bind_nodes_schema())
            .await?;
        eprintln!("[lance-persist] Ensured bind_nodes table");

        self.open_or_create_table("bind_edges", bind_edges_schema())
            .await?;
        eprintln!("[lance-persist] Ensured bind_edges table");

        self.open_or_create_table("bind_state", bind_state_schema())
            .await?;
        eprintln!("[lance-persist] Ensured bind_state table");

        Ok(())
    }

    // =========================================================================
    // PHASE 3+4: WRITE-THROUGH (FULL SNAPSHOT)
    // =========================================================================

    /// Persist the entire BindSpace to LanceDB (full snapshot).
    ///
    /// Overwrites all tables. Each overwrite creates a new version.
    /// Used for:
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
        let mut updated_ats = Vec::new();

        for (addr, node) in space.nodes_iter() {
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
            updated_ats.push(node.updated_at);
        }

        let table = self
            .open_or_create_table("bind_nodes", bind_nodes_schema())
            .await?;

        if addrs.is_empty() {
            // Write empty to clear previous data
            let schema = Arc::new(bind_nodes_schema());
            let batch = RecordBatch::new_empty(schema);
            table
                .add(batch_reader(batch))
                .mode(AddDataMode::Overwrite)
                .execute()
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

        let label_arr: StringArray = labels.iter().map(|l| l.as_deref()).collect();
        let qidx_arr = UInt8Array::from(qidxs);

        let parent_arr: UInt16Array = parents.iter().copied().collect();
        let depth_arr = UInt8Array::from(depths);
        let rung_arr = UInt8Array::from(rungs);
        let sigma_arr = UInt8Array::from(sigmas);
        let spine_arr = BooleanArray::from(spines);

        let dn_arr: UInt64Array = dn_paths.iter().copied().collect();

        let payload_arr: LargeBinaryArray = payloads.iter().map(|p| p.as_deref()).collect();

        let updated_at_arr = UInt64Array::from(updated_ats);

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
                Arc::new(updated_at_arr),
            ],
        )
        .map_err(|e| format!("batch build: {}", e))?;

        // Overwrite (full snapshot) — creates a new version
        table
            .add(batch_reader(batch))
            .mode(AddDataMode::Overwrite)
            .execute()
            .await
            .map_err(|e| format!("persist_nodes write: {}", e))?;

        Ok(())
    }

    /// Persist all edges.
    async fn persist_edges(&self, space: &BindSpace) -> Result<(), String> {
        let edges: Vec<&BindEdge> = space.edges_iter().collect();

        let table = self
            .open_or_create_table("bind_edges", bind_edges_schema())
            .await?;

        if edges.is_empty() {
            let schema = Arc::new(bind_edges_schema());
            let batch = RecordBatch::new_empty(schema);
            table
                .add(batch_reader(batch))
                .mode(AddDataMode::Overwrite)
                .execute()
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

        table
            .add(batch_reader(batch))
            .mode(AddDataMode::Overwrite)
            .execute()
            .await
            .map_err(|e| format!("persist_edges write: {}", e))?;

        Ok(())
    }

    /// Persist allocator state (next_node, next_fluid pointers).
    async fn persist_state(&self, space: &BindSpace) -> Result<(), String> {
        let (np, ns) = space.next_node_slot();
        let (fp, fs) = space.next_fluid_slot();

        let table = self
            .open_or_create_table("bind_state", bind_state_schema())
            .await?;

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

        table
            .add(batch_reader(batch))
            .mode(AddDataMode::Overwrite)
            .execute()
            .await
            .map_err(|e| format!("persist_state write: {}", e))?;

        Ok(())
    }

    // =========================================================================
    // INDEX FINGERPRINTS PERSISTENCE (HTTP /api/v1/index store)
    // =========================================================================

    /// Persist the HTTP index fingerprints Vec.
    pub async fn persist_index(
        &self,
        fingerprints: &[(
            String,
            crate::core::Fingerprint,
            std::collections::HashMap<String, String>,
        )],
    ) -> Result<(), String> {
        if !self.active {
            return Ok(());
        }

        let table = self
            .open_or_create_table("index_fingerprints", index_fingerprints_schema())
            .await?;

        if fingerprints.is_empty() {
            let schema = Arc::new(index_fingerprints_schema());
            let batch = RecordBatch::new_empty(schema);
            table
                .add(batch_reader(batch))
                .mode(AddDataMode::Overwrite)
                .execute()
                .await
                .map_err(|e| format!("persist_index empty: {}", e))?;
            return Ok(());
        }

        let ids: StringArray = fingerprints
            .iter()
            .map(|(id, _, _)| Some(id.as_str()))
            .collect();

        let mut fp_builder = FixedSizeBinaryBuilder::new(FP_BYTES);
        for (_, fp, _) in fingerprints {
            fp_builder
                .append_value(fp.as_bytes())
                .map_err(|e| format!("index fp: {}", e))?;
        }
        let fp_arr = fp_builder.finish();

        // Serialize metadata as JSON strings
        let meta_arr: StringArray = fingerprints
            .iter()
            .map(|(_, _, meta)| {
                if meta.is_empty() {
                    None
                } else {
                    Some(serde_json::to_string(meta).unwrap_or_default())
                }
            })
            .collect();

        let schema = Arc::new(index_fingerprints_schema());
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(ids), Arc::new(fp_arr), Arc::new(meta_arr)],
        )
        .map_err(|e| format!("index batch: {}", e))?;

        table
            .add(batch_reader(batch))
            .mode(AddDataMode::Overwrite)
            .execute()
            .await
            .map_err(|e| format!("persist_index write: {}", e))?;

        Ok(())
    }

    /// Hydrate the HTTP index fingerprints Vec from LanceDB.
    pub async fn hydrate_index(
        &self,
    ) -> Result<
        Vec<(
            String,
            crate::core::Fingerprint,
            std::collections::HashMap<String, String>,
        )>,
        String,
    > {
        if !self.active {
            return Ok(Vec::new());
        }

        let table = match self.connection().await?.open_table("index_fingerprints").execute().await
        {
            Ok(t) => t,
            Err(lancedb::Error::TableNotFound { .. }) => return Ok(Vec::new()),
            Err(e) => return Err(format!("open index: {}", e)),
        };

        let batches = scan_all_batches(&table).await?;
        let mut result = Vec::new();

        for batch in &batches {
            let id_col = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let fp_col = batch
                .column(1)
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .unwrap();
            let meta_col = batch
                .column(2)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();

            for row in 0..batch.num_rows() {
                let id = id_col.value(row).to_string();
                let fp = crate::core::Fingerprint::from_bytes(fp_col.value(row))
                    .map_err(|e| format!("fp decode: {}", e))?;
                let meta: std::collections::HashMap<String, String> = if meta_col.is_null(row) {
                    std::collections::HashMap::new()
                } else {
                    serde_json::from_str(meta_col.value(row)).unwrap_or_default()
                };
                result.push((id, fp, meta));
            }
        }

        eprintln!(
            "[lance-persist] Hydrated {} index fingerprints",
            result.len()
        );
        Ok(result)
    }

    // =========================================================================
    // PHASE 5: HYDRATION (LanceDB → BindSpace on startup)
    // =========================================================================

    /// Hydrate a BindSpace from LanceDB data.
    ///
    /// Returns the populated BindSpace, or None if no data exists.
    pub async fn hydrate(&self) -> Result<Option<BindSpace>, String> {
        if !self.active || !self.has_data() {
            return Ok(None);
        }

        eprintln!("[lance-persist] Hydrating BindSpace from LanceDB...");

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
        let table = match self
            .connection()
            .await?
            .open_table("bind_state")
            .execute()
            .await
        {
            Ok(t) => t,
            Err(lancedb::Error::TableNotFound { .. }) => return Ok(()),
            Err(e) => return Err(format!("open state: {}", e)),
        };

        let batches = scan_all_batches(&table).await?;
        if batches.is_empty() || batches[0].num_rows() == 0 {
            return Ok(());
        }

        let batch = &batches[0];
        let np = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap()
            .value(0);
        let ns = batch
            .column(1)
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap()
            .value(0);
        let fp = batch
            .column(2)
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap()
            .value(0);
        let fs = batch
            .column(3)
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap()
            .value(0);

        space.set_next_node_slot(np, ns);
        space.set_next_fluid_slot(fp, fs);

        Ok(())
    }

    /// Hydrate nodes from LanceDB.
    async fn hydrate_nodes(&self, space: &mut BindSpace) -> Result<usize, String> {
        let table = match self
            .connection()
            .await?
            .open_table("bind_nodes")
            .execute()
            .await
        {
            Ok(t) => t,
            Err(lancedb::Error::TableNotFound { .. }) => return Ok(0),
            Err(e) => return Err(format!("open nodes: {}", e)),
        };

        let batches = scan_all_batches(&table).await?;
        let mut count = 0usize;

        for batch in &batches {
            let addr_col = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt16Array>()
                .unwrap();
            let fp_col = batch
                .column(1)
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .unwrap();
            let label_col = batch
                .column(2)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let qidx_col = batch
                .column(3)
                .as_any()
                .downcast_ref::<UInt8Array>()
                .unwrap();
            let parent_col = batch
                .column(4)
                .as_any()
                .downcast_ref::<UInt16Array>()
                .unwrap();
            let depth_col = batch
                .column(5)
                .as_any()
                .downcast_ref::<UInt8Array>()
                .unwrap();
            let rung_col = batch
                .column(6)
                .as_any()
                .downcast_ref::<UInt8Array>()
                .unwrap();
            let sigma_col = batch
                .column(7)
                .as_any()
                .downcast_ref::<UInt8Array>()
                .unwrap();
            let spine_col = batch
                .column(8)
                .as_any()
                .downcast_ref::<BooleanArray>()
                .unwrap();
            let dn_col = batch
                .column(9)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            let payload_col = batch
                .column(10)
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .unwrap();
            // updated_at column (index 11) — may not exist in older datasets
            let updated_at_col = batch
                .column_by_name("updated_at")
                .and_then(|c| c.as_any().downcast_ref::<UInt64Array>());

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
                // Restore timestamp from LanceDB (preserves age for tier management)
                if let Some(ts_col) = updated_at_col {
                    node.updated_at = ts_col.value(row);
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
                    written.updated_at = node.updated_at;
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

    /// Hydrate edges from LanceDB.
    async fn hydrate_edges(&self, space: &mut BindSpace) -> Result<usize, String> {
        let table = match self
            .connection()
            .await?
            .open_table("bind_edges")
            .execute()
            .await
        {
            Ok(t) => t,
            Err(lancedb::Error::TableNotFound { .. }) => return Ok(0),
            Err(e) => return Err(format!("open edges: {}", e)),
        };

        let batches = scan_all_batches(&table).await?;
        let mut count = 0usize;

        for batch in &batches {
            let from_col = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt16Array>()
                .unwrap();
            let to_col = batch
                .column(1)
                .as_any()
                .downcast_ref::<UInt16Array>()
                .unwrap();
            let verb_col = batch
                .column(2)
                .as_any()
                .downcast_ref::<UInt16Array>()
                .unwrap();
            let fp_col = batch
                .column(3)
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .unwrap();
            let weight_col = batch
                .column(4)
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap();

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

    /// Whether persistence is operational.
    pub fn is_active(&self) -> bool {
        self.active
    }

    // =========================================================================
    // AGE-BASED HOT→COLD TIER FLUSH (5-30 min threshold)
    // =========================================================================

    /// Flush aged nodes from BindSpace to LanceDB.
    ///
    /// Persists all nodes older than `threshold_secs` to LanceDB cold tier,
    /// then evicts them from BindSpace to free memory.
    /// Each flush creates a new version — previous data is preserved.
    ///
    /// Returns `(persisted_count, evicted_count)`.
    pub async fn flush_aged(
        &self,
        space: &mut BindSpace,
        threshold_secs: u64,
    ) -> Result<(usize, usize), String> {
        if !self.active {
            return Ok((0, 0));
        }

        // 1. Collect aged nodes (read-only pass)
        let aged = space.collect_aged(threshold_secs);
        if aged.is_empty() {
            return Ok((0, 0));
        }

        let persist_count = aged.len();
        eprintln!(
            "[lance-persist] Flushing {} aged nodes (threshold={}s)",
            persist_count, threshold_secs
        );

        // 2. Build Arrow RecordBatch from aged nodes
        let batch = self.nodes_to_batch(&aged, space)?;

        // 3. Append to LanceDB (not overwrite — preserves existing cold data)
        //    Creates a new version automatically.
        let table = self
            .open_or_create_table("bind_nodes", bind_nodes_schema())
            .await?;
        table
            .add(batch_reader(batch))
            .mode(AddDataMode::Append)
            .execute()
            .await
            .map_err(|e| format!("flush_aged append: {}", e))?;

        // 4. Evict from BindSpace (now safely persisted)
        let evicted = space.evict_aged(threshold_secs);
        let evict_count = evicted.len();

        eprintln!(
            "[lance-persist] Flushed {} nodes to cold tier, evicted {} from hot",
            persist_count, evict_count
        );

        Ok((persist_count, evict_count))
    }

    /// Convert `(Addr, BindNode)` pairs to Arrow RecordBatch.
    fn nodes_to_batch(
        &self,
        items: &[(Addr, BindNode)],
        space: &BindSpace,
    ) -> Result<RecordBatch, String> {
        let mut addrs = Vec::with_capacity(items.len());
        let mut fps = Vec::with_capacity(items.len());
        let mut labels: Vec<Option<&str>> = Vec::with_capacity(items.len());
        let mut qidxs = Vec::with_capacity(items.len());
        let mut parents: Vec<Option<u16>> = Vec::with_capacity(items.len());
        let mut depths = Vec::with_capacity(items.len());
        let mut rungs = Vec::with_capacity(items.len());
        let mut sigmas = Vec::with_capacity(items.len());
        let mut spines = Vec::with_capacity(items.len());
        let mut dn_paths: Vec<Option<u64>> = Vec::with_capacity(items.len());
        let mut payloads: Vec<Option<&[u8]>> = Vec::with_capacity(items.len());
        let mut updated_ats = Vec::with_capacity(items.len());

        for (addr, node) in items {
            addrs.push(addr.0);
            fps.push(fp_to_le_bytes(&node.fingerprint));
            labels.push(node.label.as_deref());
            qidxs.push(node.qidx);
            parents.push(node.parent.map(|a| a.0));
            depths.push(node.depth);
            rungs.push(node.rung);
            sigmas.push(node.sigma);
            spines.push(node.is_spine);
            dn_paths.push(space.dn_index.dn_for(*addr).map(|dn| dn.0));
            payloads.push(node.payload.as_deref());
            updated_ats.push(node.updated_at);
        }

        let addr_arr = UInt16Array::from(addrs);
        let mut fp_builder = FixedSizeBinaryBuilder::new(FP_BYTES);
        for fp in &fps {
            fp_builder
                .append_value(fp)
                .map_err(|e| format!("fp append: {}", e))?;
        }
        let fp_arr = fp_builder.finish();
        let label_arr: StringArray = labels.into_iter().collect();
        let qidx_arr = UInt8Array::from(qidxs);
        let parent_arr: UInt16Array = parents.into_iter().collect();
        let depth_arr = UInt8Array::from(depths);
        let rung_arr = UInt8Array::from(rungs);
        let sigma_arr = UInt8Array::from(sigmas);
        let spine_arr = BooleanArray::from(spines);
        let dn_arr: UInt64Array = dn_paths.into_iter().collect();
        let payload_arr: LargeBinaryArray = payloads.into_iter().collect();
        let updated_at_arr = UInt64Array::from(updated_ats);

        let schema = Arc::new(bind_nodes_schema());
        RecordBatch::try_new(
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
                Arc::new(updated_at_arr),
            ],
        )
        .map_err(|e| format!("batch build: {}", e))
    }
}

// =============================================================================
// SHARED HELPERS
// =============================================================================

/// Scan all rows from a lancedb Table into RecordBatches.
async fn scan_all_batches(table: &lancedb::Table) -> Result<Vec<RecordBatch>, String> {
    let stream = table
        .query()
        .execute()
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

        // Tables are managed by lancedb Connection, not as individual .lance dirs
        assert!(persist.is_active());
    }

    #[tokio::test]
    async fn test_persist_and_hydrate_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let persist = LancePersistence::new(tmp.path().join("lance"));

        // Create and persist
        let original = make_test_space();
        persist.ensure_tables().await.unwrap();
        persist.persist_full(&original).await.unwrap();

        // Hydrate into new space (need fresh LancePersistence to avoid connection caching)
        let persist2 = LancePersistence::new(tmp.path().join("lance"));
        let hydrated = persist2.hydrate().await.unwrap().unwrap();

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
                assert_eq!(
                    orig_node.label, hydr_node.label,
                    "Label mismatch at {:?}",
                    addr
                );
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
