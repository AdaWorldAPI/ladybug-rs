//! MCP Actions for Arrow Flight
//!
//! Implements the MCP tool interface via Flight DoAction using Arrow IPC.
//! All serialization uses Arrow RecordBatch - JSON is NOT used.

use parking_lot::RwLock;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BinaryArray, BooleanArray, FixedSizeBinaryArray, Float32Array, RecordBatch,
    StringArray, UInt8Array, UInt16Array, UInt32Array,
};
use arrow_ipc::{reader::StreamReader, writer::StreamWriter};
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use crate::search::HdrIndex;
use crate::storage::BindSpace;
use crate::storage::bind_space::{Addr, FINGERPRINT_WORDS};

// =============================================================================
// ARROW SCHEMAS FOR MCP RESULTS
// =============================================================================

/// Schema for fingerprint encoding result
fn fingerprint_result_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(
            "fingerprint",
            DataType::FixedSizeBinary((FINGERPRINT_WORDS * 8) as i32),
            false,
        ),
        Field::new("bits_set", DataType::UInt32, false),
        Field::new("encoding_style", DataType::Utf8, false),
    ]))
}

/// Schema for bind result
fn bind_result_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("address", DataType::UInt16, false),
        Field::new("success", DataType::Boolean, false),
    ]))
}

/// Schema for node read result
fn node_result_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("address", DataType::UInt16, false),
        Field::new(
            "fingerprint",
            DataType::FixedSizeBinary((FINGERPRINT_WORDS * 8) as i32),
            false,
        ),
        Field::new("label", DataType::Utf8, true),
        Field::new("zone", DataType::Utf8, false),
    ]))
}

/// Schema for search/resonate results
fn search_result_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("address", DataType::UInt16, false),
        Field::new(
            "fingerprint",
            DataType::FixedSizeBinary((FINGERPRINT_WORDS * 8) as i32),
            false,
        ),
        Field::new("label", DataType::Utf8, true),
        Field::new("distance", DataType::UInt32, false),
        Field::new("similarity", DataType::Float32, false),
        Field::new("cascade_level", DataType::UInt8, false),
        Field::new("query_time_ns", DataType::UInt64, false),
        Field::new("l0_candidates", DataType::UInt32, false),
        Field::new("l1_candidates", DataType::UInt32, false),
        Field::new("l2_candidates", DataType::UInt32, false),
        Field::new("final_candidates", DataType::UInt32, false),
    ]))
}

/// Schema for distance result
fn distance_result_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("distance", DataType::UInt32, false),
        Field::new("similarity", DataType::Float32, false),
        Field::new("max_bits", DataType::UInt32, false),
    ]))
}

/// Schema for XOR bind result
fn xor_result_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(
            "fingerprint",
            DataType::FixedSizeBinary((FINGERPRINT_WORDS * 8) as i32),
            false,
        ),
        Field::new("bits_set", DataType::UInt32, false),
    ]))
}

/// Schema for stats result
fn stats_result_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("total_nodes", DataType::UInt32, false),
        Field::new("surface_nodes", DataType::UInt32, false),
        Field::new("fluid_nodes", DataType::UInt32, false),
        Field::new("node_space_nodes", DataType::UInt32, false),
    ]))
}

/// Schema for error result
fn error_result_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("error", DataType::Boolean, false),
        Field::new("message", DataType::Utf8, false),
    ]))
}

/// Schema for action input parameters (Arrow IPC request)
fn action_input_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("param_name", DataType::Utf8, false),
        Field::new("param_value", DataType::Binary, false),
    ]))
}

// =============================================================================
// ARROW IPC HELPERS
// =============================================================================

/// Encode a RecordBatch to Arrow IPC stream bytes
fn encode_to_ipc(batch: &RecordBatch) -> Result<Vec<u8>, String> {
    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, batch.schema().as_ref())
            .map_err(|e| e.to_string())?;
        writer.write(batch).map_err(|e| e.to_string())?;
        writer.finish().map_err(|e| e.to_string())?;
    }
    Ok(buffer)
}

/// Decode Arrow IPC stream bytes to get parameters
/// Returns a map of param_name -> param_value (as bytes)
fn decode_ipc_params(data: &[u8]) -> Result<std::collections::HashMap<String, Vec<u8>>, String> {
    if data.is_empty() {
        return Ok(std::collections::HashMap::new());
    }

    let cursor = std::io::Cursor::new(data);
    let reader = StreamReader::try_new(cursor, None)
        .map_err(|e| format!("Failed to read Arrow IPC: {}", e))?;

    let mut params = std::collections::HashMap::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| e.to_string())?;

        // Get param_name and param_value columns
        let names = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or("Expected StringArray for param_name")?;
        let values = batch
            .column(1)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .ok_or("Expected BinaryArray for param_value")?;

        for i in 0..names.len() {
            if let (Some(name), Some(value)) = (names.value(i).into(), values.value(i).into()) {
                params.insert(name.to_string(), value.to_vec());
            }
        }
    }

    Ok(params)
}

/// Extract string parameter from decoded params
fn get_string_param(
    params: &std::collections::HashMap<String, Vec<u8>>,
    key: &str,
) -> Option<String> {
    params
        .get(key)
        .and_then(|v| String::from_utf8(v.clone()).ok())
}

/// Extract u16 parameter from decoded params
fn get_u16_param(params: &std::collections::HashMap<String, Vec<u8>>, key: &str) -> Option<u16> {
    params.get(key).and_then(|v| {
        if v.len() >= 2 {
            Some(u16::from_le_bytes([v[0], v[1]]))
        } else {
            None
        }
    })
}

/// Extract usize parameter from decoded params
fn get_usize_param(
    params: &std::collections::HashMap<String, Vec<u8>>,
    key: &str,
) -> Option<usize> {
    params.get(key).and_then(|v| {
        if v.len() >= 8 {
            Some(u64::from_le_bytes(v[..8].try_into().unwrap()) as usize)
        } else if v.len() >= 4 {
            Some(u32::from_le_bytes(v[..4].try_into().unwrap()) as usize)
        } else {
            None
        }
    })
}

/// Extract u32 parameter from decoded params
fn get_u32_param(params: &std::collections::HashMap<String, Vec<u8>>, key: &str) -> Option<u32> {
    params.get(key).and_then(|v| {
        if v.len() >= 4 {
            Some(u32::from_le_bytes(v[..4].try_into().unwrap()))
        } else {
            None
        }
    })
}

/// Extract bytes parameter from decoded params
fn get_bytes_param(
    params: &std::collections::HashMap<String, Vec<u8>>,
    key: &str,
) -> Option<Vec<u8>> {
    params.get(key).cloned()
}

// =============================================================================
// ACTION EXECUTION
// =============================================================================

/// Execute an MCP action and return Arrow IPC encoded result
pub async fn execute_action(
    action_type: &str,
    body: &[u8],
    bind_space: Arc<RwLock<BindSpace>>,
    hdr_cascade: Arc<RwLock<HdrIndex>>,
) -> Result<Vec<u8>, String> {
    // Parse Arrow IPC body to get parameters
    let params = decode_ipc_params(body)?;

    match action_type {
        "encode" => execute_encode(&params),
        "bind" => execute_bind(&params, bind_space),
        "read" => execute_read(&params, bind_space),
        "resonate" => execute_resonate(&params, bind_space, hdr_cascade),
        "hamming" => execute_hamming(&params),
        "xor_bind" => execute_xor_bind(&params),
        "stats" => execute_stats(bind_space),
        "ingest.unified_step" => execute_ingest_step(body, bind_space),
        _ => {
            let schema = error_result_schema();
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(BooleanArray::from(vec![true])) as ArrayRef,
                    Arc::new(StringArray::from(vec![format!(
                        "Unknown action: {}",
                        action_type
                    )])) as ArrayRef,
                ],
            )
            .map_err(|e| e.to_string())?;
            encode_to_ipc(&batch)
        }
    }
}

/// Encode text/data to fingerprint
fn execute_encode(params: &std::collections::HashMap<String, Vec<u8>>) -> Result<Vec<u8>, String> {
    let text = get_string_param(params, "text");
    let data = get_bytes_param(params, "data");
    let style = get_string_param(params, "style").unwrap_or_else(|| "balanced".to_string());

    let input = if let Some(t) = text {
        t.into_bytes()
    } else if let Some(d) = data {
        d
    } else {
        return Err("Either text or data required".to_string());
    };

    // Sigma-10 membrane encoding (simplified: SHA256-based expansion)
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(&input);
    let hash = hasher.finalize();

    // Expand to full fingerprint size (2048 bytes = 256 * 8)
    let mut fingerprint = vec![0u8; FINGERPRINT_WORDS * 8];
    for (i, chunk) in fingerprint.chunks_mut(32).enumerate() {
        let mut h = Sha256::new();
        h.update(&hash);
        h.update(&[i as u8]);
        chunk.copy_from_slice(&h.finalize()[..chunk.len().min(32)]);
    }

    let bits_set: u32 = fingerprint.iter().map(|b| b.count_ones()).sum();

    // Build Arrow result
    let schema = fingerprint_result_schema();
    let fp_array: ArrayRef = Arc::new(
        FixedSizeBinaryArray::try_from_iter(std::iter::once(fingerprint.as_slice()))
            .map_err(|e| e.to_string())?,
    );
    let bits_array: ArrayRef = Arc::new(UInt32Array::from(vec![bits_set]));
    let style_array: ArrayRef = Arc::new(StringArray::from(vec![style]));

    let batch = RecordBatch::try_new(schema, vec![fp_array, bits_array, style_array])
        .map_err(|e| e.to_string())?;

    encode_to_ipc(&batch)
}

/// Bind fingerprint to address
fn execute_bind(
    params: &std::collections::HashMap<String, Vec<u8>>,
    bind_space: Arc<RwLock<BindSpace>>,
) -> Result<Vec<u8>, String> {
    let address = get_u16_param(params, "address").ok_or("missing address")?;
    let fingerprint = get_bytes_param(params, "fingerprint").ok_or("missing fingerprint")?;
    let label = get_string_param(params, "label");

    let addr = Addr(address);

    // Convert bytes to [u64; FINGERPRINT_WORDS]
    let mut fp_array = [0u64; FINGERPRINT_WORDS];
    for (i, chunk) in fingerprint.chunks(8).enumerate() {
        if i >= FINGERPRINT_WORDS {
            break;
        }
        if chunk.len() == 8 {
            fp_array[i] = u64::from_le_bytes(chunk.try_into().unwrap());
        } else {
            let mut buf = [0u8; 8];
            buf[..chunk.len()].copy_from_slice(chunk);
            fp_array[i] = u64::from_le_bytes(buf);
        }
    }

    let mut space = bind_space.write();
    let success = space.write_at(addr, fp_array);

    if success {
        if let Some(lbl) = label {
            if let Some(node) = space.read_mut(addr) {
                node.label = Some(lbl);
            }
        }
    }

    // Build Arrow result
    let schema = bind_result_schema();
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt16Array::from(vec![address])) as ArrayRef,
            Arc::new(BooleanArray::from(vec![success])) as ArrayRef,
        ],
    )
    .map_err(|e| e.to_string())?;

    encode_to_ipc(&batch)
}

/// Read node from address
fn execute_read(
    params: &std::collections::HashMap<String, Vec<u8>>,
    bind_space: Arc<RwLock<BindSpace>>,
) -> Result<Vec<u8>, String> {
    let address = get_u16_param(params, "address").ok_or("missing address")?;

    let addr = Addr(address);
    let space = bind_space.read();

    if let Some(node) = space.read(addr) {
        let fingerprint: Vec<u8> = node
            .fingerprint
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();

        let zone = match addr.prefix() {
            0x00..=0x0F => "surface",
            0x10..=0x7F => "fluid",
            _ => "node",
        };

        let schema = node_result_schema();
        let fp_array: ArrayRef = Arc::new(
            FixedSizeBinaryArray::try_from_iter(std::iter::once(fingerprint.as_slice()))
                .map_err(|e| e.to_string())?,
        );
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt16Array::from(vec![address])) as ArrayRef,
                fp_array,
                Arc::new(StringArray::from(vec![node.label.as_deref()])) as ArrayRef,
                Arc::new(StringArray::from(vec![zone])) as ArrayRef,
            ],
        )
        .map_err(|e| e.to_string())?;

        encode_to_ipc(&batch)
    } else {
        // Return error via error schema
        let schema = error_result_schema();
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(BooleanArray::from(vec![true])) as ArrayRef,
                Arc::new(StringArray::from(vec![format!(
                    "No node at address {:#06x}",
                    address
                )])) as ArrayRef,
            ],
        )
        .map_err(|e| e.to_string())?;
        encode_to_ipc(&batch)
    }
}

/// Find similar fingerprints via HDR cascade
fn execute_resonate(
    params: &std::collections::HashMap<String, Vec<u8>>,
    _bind_space: Arc<RwLock<BindSpace>>,
    _hdr_cascade: Arc<RwLock<HdrIndex>>,
) -> Result<Vec<u8>, String> {
    let _query = get_bytes_param(params, "query").ok_or("missing query")?;
    let _k = get_usize_param(params, "k").unwrap_or(10);
    let _threshold = get_u32_param(params, "threshold");

    let start = std::time::Instant::now();

    // TODO: Implement HDR cascade search
    // Placeholder: return empty results

    let schema = search_result_schema();

    // Empty batch with correct schema
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt16Array::from(Vec::<u16>::new())) as ArrayRef,
            Arc::new(FixedSizeBinaryArray::from(Vec::<Option<&[u8]>>::new())) as ArrayRef,
            Arc::new(StringArray::from(Vec::<Option<&str>>::new())) as ArrayRef,
            Arc::new(UInt32Array::from(Vec::<u32>::new())) as ArrayRef,
            Arc::new(Float32Array::from(Vec::<f32>::new())) as ArrayRef,
            Arc::new(UInt8Array::from(Vec::<u8>::new())) as ArrayRef,
            Arc::new(arrow_array::UInt64Array::from(vec![
                start.elapsed().as_nanos() as u64,
            ])) as ArrayRef,
            Arc::new(UInt32Array::from(vec![0u32])) as ArrayRef,
            Arc::new(UInt32Array::from(vec![0u32])) as ArrayRef,
            Arc::new(UInt32Array::from(vec![0u32])) as ArrayRef,
            Arc::new(UInt32Array::from(vec![0u32])) as ArrayRef,
        ],
    )
    .map_err(|e| e.to_string())?;

    encode_to_ipc(&batch)
}

/// Compute Hamming distance between two fingerprints
fn execute_hamming(params: &std::collections::HashMap<String, Vec<u8>>) -> Result<Vec<u8>, String> {
    let a = get_bytes_param(params, "a").ok_or("missing a")?;
    let b = get_bytes_param(params, "b").ok_or("missing b")?;

    let max_len = a.len().min(b.len());
    let distance: u32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum();

    let max_bits = (max_len * 8) as u32;
    let similarity = if max_bits > 0 {
        1.0 - (distance as f32 / max_bits as f32)
    } else {
        0.0
    };

    let schema = distance_result_schema();
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt32Array::from(vec![distance])) as ArrayRef,
            Arc::new(Float32Array::from(vec![similarity])) as ArrayRef,
            Arc::new(UInt32Array::from(vec![max_bits])) as ArrayRef,
        ],
    )
    .map_err(|e| e.to_string())?;

    encode_to_ipc(&batch)
}

/// XOR bind two fingerprints
fn execute_xor_bind(
    params: &std::collections::HashMap<String, Vec<u8>>,
) -> Result<Vec<u8>, String> {
    let a = get_bytes_param(params, "a").ok_or("missing a")?;
    let b = get_bytes_param(params, "b").ok_or("missing b")?;

    let fingerprint: Vec<u8> = a.iter().zip(b.iter()).map(|(x, y)| x ^ y).collect();

    let bits_set: u32 = fingerprint.iter().map(|b| b.count_ones()).sum();

    // Pad to full fingerprint size if needed
    let mut full_fp = vec![0u8; FINGERPRINT_WORDS * 8];
    let copy_len = fingerprint.len().min(FINGERPRINT_WORDS * 8);
    full_fp[..copy_len].copy_from_slice(&fingerprint[..copy_len]);

    let schema = xor_result_schema();
    let fp_array: ArrayRef = Arc::new(
        FixedSizeBinaryArray::try_from_iter(std::iter::once(full_fp.as_slice()))
            .map_err(|e| e.to_string())?,
    );
    let batch = RecordBatch::try_new(
        schema,
        vec![
            fp_array,
            Arc::new(UInt32Array::from(vec![bits_set])) as ArrayRef,
        ],
    )
    .map_err(|e| e.to_string())?;

    encode_to_ipc(&batch)
}

/// Get BindSpace statistics
fn execute_stats(bind_space: Arc<RwLock<BindSpace>>) -> Result<Vec<u8>, String> {
    let space = bind_space.read();
    let stats = space.stats();

    let schema = stats_result_schema();
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt32Array::from(vec![
                (stats.surface_count + stats.fluid_count + stats.node_count) as u32,
            ])) as ArrayRef,
            Arc::new(UInt32Array::from(vec![stats.surface_count as u32])) as ArrayRef,
            Arc::new(UInt32Array::from(vec![stats.fluid_count as u32])) as ArrayRef,
            Arc::new(UInt32Array::from(vec![stats.node_count as u32])) as ArrayRef,
        ],
    )
    .map_err(|e| e.to_string())?;

    encode_to_ipc(&batch)
}

// =============================================================================
// LEGACY JSON FALLBACK (for backwards compatibility only)
// =============================================================================

#[cfg(feature = "json_fallback")]
mod json_fallback {
    use serde::{Deserialize, Serialize};

    /// MCP Action types (legacy JSON format)
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(tag = "action", rename_all = "snake_case")]
    pub enum McpAction {
        Encode {
            text: Option<String>,
            data: Option<Vec<u8>>,
            style: Option<String>,
        },
        Bind {
            address: u16,
            fingerprint: Vec<u8>,
            label: Option<String>,
        },
        Read {
            address: u16,
        },
        Resonate {
            query: Vec<u8>,
            k: usize,
            threshold: Option<u32>,
        },
        Hamming {
            a: Vec<u8>,
            b: Vec<u8>,
        },
        XorBind {
            a: Vec<u8>,
            b: Vec<u8>,
        },
        Stats,
    }

    /// MCP Action result (legacy JSON format)
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(tag = "type", rename_all = "snake_case")]
    pub enum McpResult {
        Fingerprint {
            fingerprint: Vec<u8>,
            bits_set: u32,
            encoding_style: String,
        },
        Bound {
            address: u16,
            success: bool,
        },
        Node {
            address: u16,
            fingerprint: Vec<u8>,
            label: Option<String>,
            zone: String,
        },
        Matches {
            results: Vec<MatchResult>,
            query_time_ns: u64,
            cascade_stats: CascadeStats,
        },
        Distance {
            distance: u32,
            similarity: f32,
            max_bits: u32,
        },
        Combined {
            fingerprint: Vec<u8>,
            bits_set: u32,
        },
        Stats {
            total_nodes: usize,
            surface_nodes: usize,
            fluid_nodes: usize,
            node_space_nodes: usize,
        },
        Error {
            message: String,
        },
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct MatchResult {
        pub address: u16,
        pub fingerprint: Vec<u8>,
        pub label: Option<String>,
        pub distance: u32,
        pub similarity: f32,
        pub cascade_level: u8,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CascadeStats {
        pub l0_candidates: usize,
        pub l1_candidates: usize,
        pub l2_candidates: usize,
        pub final_candidates: usize,
    }
}

// =============================================================================
// INGEST: UNIFIED STEP (Contract Integration)
// =============================================================================

/// Ingest a UnifiedStep from ada-n8n or crewai-rust via Flight push.
///
/// The action body is raw JSON (not Arrow IPC) containing a serialized
/// `UnifiedStep`. The step's output is fingerprinted and stored in BindSpace.
/// Returns an Arrow IPC batch with the enrichment result.
fn execute_ingest_step(body: &[u8], bind_space: Arc<RwLock<BindSpace>>) -> Result<Vec<u8>, String> {
    use crate::contract::enricher::EnrichmentEngine;
    use crate::contract::types::UnifiedStep;

    // Deserialize the step from JSON body
    let step: UnifiedStep =
        serde_json::from_slice(body).map_err(|e| format!("Invalid UnifiedStep JSON: {}", e))?;

    // Enrich: fingerprint + write to BindSpace
    let engine = EnrichmentEngine::new(bind_space);
    let enrichment = engine.enrich_step(&step);

    // Build Arrow result: address, qidx, step_id
    let schema = Arc::new(Schema::new(vec![
        Field::new("step_id", DataType::Utf8, false),
        Field::new("address", DataType::UInt16, false),
        Field::new("qidx", DataType::UInt8, false),
        Field::new("success", DataType::Boolean, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec![enrichment.step_id.as_str()])) as ArrayRef,
            Arc::new(UInt16Array::from(vec![enrichment.bind_addr.0])) as ArrayRef,
            Arc::new(UInt8Array::from(vec![enrichment.qidx])) as ArrayRef,
            Arc::new(BooleanArray::from(vec![true])) as ArrayRef,
        ],
    )
    .map_err(|e| e.to_string())?;

    encode_to_ipc(&batch)
}
