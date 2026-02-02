//! Ladybug Arrow Flight Server
//!
//! Implements FlightService for MCP-style interactions with BindSpace.
//!
//! This module provides Arrow Flight RPC endpoints for:
//! - Zero-copy fingerprint streaming (DoGet)
//! - Batch fingerprint ingestion (DoPut)
//! - MCP tool execution (DoAction)
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Claude / AI Client                            │
//! └─────────────────────────────────────────────────────────────────┘
//!                               │
//!                               ▼ Arrow Flight (gRPC)
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                  LadybugFlightService                           │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  DoGet(ticket)     → Stream search results as RecordBatches     │
//! │  DoPut(stream)     → Ingest fingerprints (zero-copy)            │
//! │  DoAction(action)  → MCP tools (encode, bind, resonate)         │
//! │  GetFlightInfo()   → Schema discovery for fingerprints          │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Ticket Formats
//!
//! - `all` - stream all fingerprints from BindSpace
//! - `surface` - stream surface zone (0x00-0x0F)
//! - `fluid` - stream fluid zone (0x10-0x7F)
//! - `nodes` - stream node zone (0x80-0xFF)
//! - `search:<query_hex>:<threshold>` - similarity search with HDR cascade
//! - `topk:<query_hex>:<k>` - top-k similar fingerprints
//!
//! # Actions (MCP Tools)
//!
//! - `encode` - Encode text/data to 10K-bit fingerprint
//! - `bind` - Bind fingerprint to BindSpace address
//! - `read` - Read node from BindSpace address
//! - `resonate` - Find similar fingerprints via HDR cascade
//! - `hamming` - Compute Hamming distance between fingerprints
//! - `xor_bind` - XOR bind two fingerprints (holographic composition)
//! - `stats` - Get BindSpace statistics

use std::pin::Pin;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, FixedSizeBinaryArray, Float32Array, RecordBatch,
    StringArray, UInt16Array, UInt32Array, UInt8Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use arrow_ipc::writer::IpcWriteOptions;
use arrow_flight::{
    flight_service_server::FlightService,
    encode::FlightDataEncoderBuilder,
    Action, ActionType, Criteria, Empty, FlightData, FlightDescriptor, FlightInfo,
    HandshakeRequest, HandshakeResponse, PollInfo, PutResult, SchemaAsIpc, SchemaResult,
    Ticket,
};
use futures::{Stream, StreamExt, stream};
use parking_lot::RwLock;
use tonic::{Request, Response, Status, Streaming};

use crate::storage::BindSpace;
use crate::storage::bind_space::{Addr, FINGERPRINT_WORDS};
use crate::search::HdrIndex;

use super::actions::execute_action;

// =============================================================================
// CONSTANTS
// =============================================================================

/// Batch size for streaming (number of fingerprints per RecordBatch)
const BATCH_SIZE: usize = 1000;

/// Maximum results for unbounded search
const MAX_SEARCH_RESULTS: usize = 10000;

// =============================================================================
// SCHEMA DEFINITIONS
// =============================================================================

/// Fingerprint schema for Arrow Flight transfers
///
/// Schema:
/// - address: UInt16 (16-bit BindSpace address)
/// - fingerprint: FixedSizeBinary(1248) (156 * 8 bytes)
/// - label: Utf8 (optional human-readable label)
/// - zone: Utf8 (surface/fluid/node)
/// - distance: UInt32 (optional Hamming distance)
/// - similarity: Float32 (optional 0.0-1.0)
pub fn fingerprint_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("address", DataType::UInt16, false),
        Field::new("fingerprint", DataType::FixedSizeBinary((FINGERPRINT_WORDS * 8) as i32), false),
        Field::new("label", DataType::Utf8, true),
        Field::new("zone", DataType::Utf8, false),
        Field::new("distance", DataType::UInt32, true),
        Field::new("similarity", DataType::Float32, true),
    ]))
}

/// Search result schema for streaming HDR cascade results
pub fn search_result_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("address", DataType::UInt16, false),
        Field::new("fingerprint", DataType::FixedSizeBinary((FINGERPRINT_WORDS * 8) as i32), false),
        Field::new("label", DataType::Utf8, true),
        Field::new("distance", DataType::UInt32, false),
        Field::new("similarity", DataType::Float32, false),
        Field::new("cascade_level", DataType::UInt8, false),
    ]))
}

// =============================================================================
// FLIGHT SERVICE
// =============================================================================

/// Ladybug Flight Service for MCP interactions
pub struct LadybugFlightService {
    bind_space: Arc<RwLock<BindSpace>>,
    hdr_index: Arc<RwLock<HdrIndex>>,
}

impl LadybugFlightService {
    /// Create a new Flight service
    pub fn new(bind_space: Arc<RwLock<BindSpace>>, hdr_index: Arc<RwLock<HdrIndex>>) -> Self {
        Self { bind_space, hdr_index }
    }
}

/// Stream type for tonic responses
type TonicStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;

#[tonic::async_trait]
impl FlightService for LadybugFlightService {
    type HandshakeStream = TonicStream<HandshakeResponse>;
    type ListFlightsStream = TonicStream<FlightInfo>;
    type DoGetStream = TonicStream<FlightData>;
    type DoPutStream = TonicStream<PutResult>;
    type DoActionStream = TonicStream<arrow_flight::Result>;
    type ListActionsStream = TonicStream<ActionType>;
    type DoExchangeStream = TonicStream<FlightData>;

    async fn handshake(
        &self,
        _request: Request<Streaming<HandshakeRequest>>,
    ) -> Result<Response<Self::HandshakeStream>, Status> {
        let output = stream::once(async {
            Ok(HandshakeResponse {
                protocol_version: 1,
                payload: bytes::Bytes::from("ladybug-flight-v1"),
            })
        });
        Ok(Response::new(Box::pin(output)))
    }

    async fn list_flights(
        &self,
        _request: Request<Criteria>,
    ) -> Result<Response<Self::ListFlightsStream>, Status> {
        let schema = fingerprint_schema();
        let stats = self.bind_space.read().stats();

        let flights = vec![
            FlightInfo::new()
                .try_with_schema(&schema)
                .map_err(|e| Status::internal(e.to_string()))?
                .with_descriptor(FlightDescriptor::new_path(vec!["all".to_string()]))
                .with_total_records((stats.surface_count + stats.fluid_count + stats.node_count) as i64),
            FlightInfo::new()
                .try_with_schema(&schema)
                .map_err(|e| Status::internal(e.to_string()))?
                .with_descriptor(FlightDescriptor::new_path(vec!["surface".to_string()]))
                .with_total_records(stats.surface_count as i64),
            FlightInfo::new()
                .try_with_schema(&schema)
                .map_err(|e| Status::internal(e.to_string()))?
                .with_descriptor(FlightDescriptor::new_path(vec!["fluid".to_string()]))
                .with_total_records(stats.fluid_count as i64),
            FlightInfo::new()
                .try_with_schema(&schema)
                .map_err(|e| Status::internal(e.to_string()))?
                .with_descriptor(FlightDescriptor::new_path(vec!["nodes".to_string()]))
                .with_total_records(stats.node_count as i64),
        ];

        let output = stream::iter(flights.into_iter().map(Ok));
        Ok(Response::new(Box::pin(output)))
    }

    async fn get_flight_info(
        &self,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let descriptor = request.into_inner();
        let schema = fingerprint_schema();
        let stats = self.bind_space.read().stats();

        let records = match descriptor.path.first().map(|s| s.as_str()) {
            Some("surface") => stats.surface_count,
            Some("fluid") => stats.fluid_count,
            Some("nodes") => stats.node_count,
            _ => stats.surface_count + stats.fluid_count + stats.node_count,
        };

        let info = FlightInfo::new()
            .try_with_schema(&schema)
            .map_err(|e| Status::internal(e.to_string()))?
            .with_descriptor(descriptor)
            .with_total_records(records as i64);

        Ok(Response::new(info))
    }

    async fn poll_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<PollInfo>, Status> {
        Err(Status::unimplemented("poll_flight_info not implemented"))
    }

    async fn get_schema(
        &self,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<SchemaResult>, Status> {
        let descriptor = request.into_inner();

        // Use search schema for search/topk tickets
        let schema = if descriptor.path.first()
            .map(|s| s.starts_with("search") || s.starts_with("topk"))
            .unwrap_or(false)
        {
            search_result_schema()
        } else {
            fingerprint_schema()
        };

        let options = IpcWriteOptions::default();
        let schema_result = SchemaAsIpc::new(&schema, &options)
            .try_into()
            .map_err(|e: arrow_schema::ArrowError| Status::internal(e.to_string()))?;

        Ok(Response::new(schema_result))
    }

    /// DoGet - Stream fingerprints or search results
    async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        let ticket = request.into_inner();
        let ticket_str = String::from_utf8(ticket.ticket.to_vec())
            .map_err(|_| Status::invalid_argument("Invalid UTF-8 in ticket"))?;

        let bind_space = self.bind_space.clone();
        let hdr_index = self.hdr_index.clone();

        match ticket_str.as_str() {
            "all" => {
                let output = stream_all_fingerprints(bind_space, 0x00..=0xFF);
                Ok(Response::new(Box::pin(output)))
            }
            "surface" => {
                let output = stream_all_fingerprints(bind_space, 0x00..=0x0F);
                Ok(Response::new(Box::pin(output)))
            }
            "fluid" => {
                let output = stream_all_fingerprints(bind_space, 0x10..=0x7F);
                Ok(Response::new(Box::pin(output)))
            }
            "nodes" => {
                let output = stream_all_fingerprints(bind_space, 0x80..=0xFF);
                Ok(Response::new(Box::pin(output)))
            }
            _ if ticket_str.starts_with("search:") => {
                let parts: Vec<&str> = ticket_str.split(':').collect();
                if parts.len() != 3 {
                    return Err(Status::invalid_argument(
                        "Invalid search format. Use: search:<query_hex>:<threshold>"
                    ));
                }

                let query = hex::decode(parts[1])
                    .map_err(|_| Status::invalid_argument("Invalid query hex"))?;
                let threshold: u32 = parts[2].parse()
                    .map_err(|_| Status::invalid_argument("Invalid threshold"))?;

                let output = stream_search_results(bind_space, hdr_index, query, threshold);
                Ok(Response::new(Box::pin(output)))
            }
            _ if ticket_str.starts_with("topk:") => {
                let parts: Vec<&str> = ticket_str.split(':').collect();
                if parts.len() != 3 {
                    return Err(Status::invalid_argument(
                        "Invalid topk format. Use: topk:<query_hex>:<k>"
                    ));
                }

                let query = hex::decode(parts[1])
                    .map_err(|_| Status::invalid_argument("Invalid query hex"))?;
                let k: usize = parts[2].parse()
                    .map_err(|_| Status::invalid_argument("Invalid k"))?;

                let output = stream_topk_results(bind_space, hdr_index, query, k);
                Ok(Response::new(Box::pin(output)))
            }
            _ => Err(Status::invalid_argument(format!(
                "Unknown ticket: {}. Use: all, surface, fluid, nodes, search:..., or topk:...",
                ticket_str
            ))),
        }
    }

    async fn do_put(
        &self,
        request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        let mut input = request.into_inner();
        let bind_space = self.bind_space.clone();
        let mut total = 0usize;

        while let Some(data) = input.next().await {
            let data = data?;

            // Skip schema-only messages
            if data.data_body.is_empty() {
                continue;
            }

            // Decode and ingest batch
            match decode_and_ingest(&bind_space, &data) {
                Ok(count) => total += count,
                Err(e) => eprintln!("Warning: Failed to ingest batch: {}", e),
            }
        }

        let result = PutResult {
            app_metadata: bytes::Bytes::from(format!("{{\"ingested\":{}}}", total)),
        };

        let output = stream::once(async { Ok(result) });
        Ok(Response::new(Box::pin(output)))
    }

    async fn do_action(
        &self,
        request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        let action = request.into_inner();
        let action_type = action.r#type.as_str();
        let body = action.body;

        let result = execute_action(action_type, &body, self.bind_space.clone(), self.hdr_index.clone())
            .await
            .map_err(|e| Status::internal(e))?;

        let flight_result = arrow_flight::Result {
            body: bytes::Bytes::from(result),
        };

        let output = stream::once(async { Ok(flight_result) });
        Ok(Response::new(Box::pin(output)))
    }

    async fn list_actions(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        let actions = vec![
            ActionType {
                r#type: "encode".to_string(),
                description: "Encode text/data to 10K-bit fingerprint. Args: {text?, data?, style?}".to_string(),
            },
            ActionType {
                r#type: "bind".to_string(),
                description: "Bind fingerprint to address. Args: {address, fingerprint, label?}".to_string(),
            },
            ActionType {
                r#type: "read".to_string(),
                description: "Read node from address. Args: {address}".to_string(),
            },
            ActionType {
                r#type: "resonate".to_string(),
                description: "Find similar fingerprints. Args: {query, k?, threshold?}".to_string(),
            },
            ActionType {
                r#type: "hamming".to_string(),
                description: "Compute Hamming distance. Args: {a, b}".to_string(),
            },
            ActionType {
                r#type: "xor_bind".to_string(),
                description: "XOR bind two fingerprints. Args: {a, b}".to_string(),
            },
            ActionType {
                r#type: "stats".to_string(),
                description: "Get BindSpace statistics. Args: {}".to_string(),
            },
        ];

        let output = stream::iter(actions.into_iter().map(Ok));
        Ok(Response::new(Box::pin(output)))
    }

    async fn do_exchange(
        &self,
        _request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        Err(Status::unimplemented("do_exchange not implemented"))
    }
}

// =============================================================================
// STREAMING IMPLEMENTATIONS
// =============================================================================

/// Stream all fingerprints from BindSpace within a prefix range
fn stream_all_fingerprints(
    bind_space: Arc<RwLock<BindSpace>>,
    prefix_range: std::ops::RangeInclusive<u8>,
) -> impl Stream<Item = Result<FlightData, Status>> {
    let schema = fingerprint_schema();

    stream::unfold(
        (bind_space, prefix_range.into_iter(), schema, Vec::new()),
        |(bs, mut prefixes, schema, mut buffer)| async move {
            // Collect fingerprints until we have a batch
            loop {
                // Try to fill buffer from current prefix
                if buffer.len() >= BATCH_SIZE {
                    break;
                }

                match prefixes.next() {
                    Some(prefix) => {
                        let space = bs.read();
                        for slot in 0..=255u8 {
                            let addr = Addr::new(prefix, slot);
                            if let Some(node) = space.read(addr) {
                                let zone = match prefix {
                                    0x00..=0x0F => "surface",
                                    0x10..=0x7F => "fluid",
                                    _ => "node",
                                };
                                buffer.push((addr.0, node.fingerprint, node.label.clone(), zone.to_string()));
                            }
                        }
                    }
                    None => break, // No more prefixes
                }
            }

            if buffer.is_empty() {
                return None; // Done streaming
            }

            // Take a batch from buffer
            let batch_data: Vec<_> = buffer.drain(..buffer.len().min(BATCH_SIZE)).collect();

            // Build RecordBatch
            match build_fingerprint_batch(&batch_data, &schema) {
                Ok(batch) => {
                    // Encode to FlightData
                    let encoder = FlightDataEncoderBuilder::new()
                        .with_schema(schema.clone())
                        .build(stream::once(async { Ok(batch) }));

                    let flight_data: Vec<Result<FlightData, Status>> = encoder
                        .map(|r| r.map_err(|e| Status::internal(e.to_string())))
                        .collect()
                        .await;

                    Some((stream::iter(flight_data), (bs, prefixes, schema, buffer)))
                }
                Err(e) => {
                    Some((stream::iter(vec![Err(e)]), (bs, prefixes, schema, buffer)))
                }
            }
        },
    )
    .flatten()
}

/// Stream search results using HDR cascade
fn stream_search_results(
    bind_space: Arc<RwLock<BindSpace>>,
    hdr_index: Arc<RwLock<HdrIndex>>,
    query: Vec<u8>,
    threshold: u32,
) -> impl Stream<Item = Result<FlightData, Status>> {
    let schema = search_result_schema();

    // Do all synchronous work first (locks must be released before any async)
    let batch_result = {
        // Convert query bytes to [u64; FINGERPRINT_WORDS]
        let mut query_fp = [0u64; FINGERPRINT_WORDS];
        for (i, chunk) in query.chunks(8).enumerate() {
            if i >= FINGERPRINT_WORDS {
                break;
            }
            if chunk.len() == 8 {
                query_fp[i] = u64::from_le_bytes(chunk.try_into().unwrap());
            }
        }

        // Search HDR index (lock released after this block)
        let mut results = {
            let hdr = hdr_index.read();
            hdr.search(&query_fp, MAX_SEARCH_RESULTS)
        };

        // Filter by threshold
        results.retain(|(_, dist)| *dist <= threshold);

        // Build result batch (all locks released after this block)
        if results.is_empty() {
            Ok(RecordBatch::new_empty(schema.clone()))
        } else {
            let batch_data = {
                let space = bind_space.read();
                let hdr = hdr_index.read();
                build_search_result_data(&space, &hdr, &results)
            };
            build_search_batch(&batch_data, &schema)
        }
    };

    // Now do async encoding - no locks held
    let schema_clone = schema.clone();
    stream::once(async move {
        match batch_result {
            Ok(batch) => {
                let encoder = FlightDataEncoderBuilder::new()
                    .with_schema(schema_clone)
                    .build(stream::once(async { Ok(batch) }));

                let flight_data: Vec<FlightData> = encoder
                    .filter_map(|r| async { r.ok() })
                    .collect()
                    .await;

                flight_data
            }
            Err(_) => Vec::new(),
        }
    })
    .flat_map(|data| stream::iter(data.into_iter().map(Ok)))
}

/// Stream top-k search results
fn stream_topk_results(
    bind_space: Arc<RwLock<BindSpace>>,
    hdr_index: Arc<RwLock<HdrIndex>>,
    query: Vec<u8>,
    k: usize,
) -> impl Stream<Item = Result<FlightData, Status>> {
    let schema = search_result_schema();

    // Do all synchronous work first (locks must be released before any async)
    let batch_result = {
        // Convert query bytes to [u64; FINGERPRINT_WORDS]
        let mut query_fp = [0u64; FINGERPRINT_WORDS];
        for (i, chunk) in query.chunks(8).enumerate() {
            if i >= FINGERPRINT_WORDS {
                break;
            }
            if chunk.len() == 8 {
                query_fp[i] = u64::from_le_bytes(chunk.try_into().unwrap());
            }
        }

        // Search HDR index for top-k (lock released after this block)
        let results = {
            let hdr = hdr_index.read();
            hdr.search(&query_fp, k)
        };

        // Build result batch (all locks released after this block)
        if results.is_empty() {
            Ok(RecordBatch::new_empty(schema.clone()))
        } else {
            let batch_data = {
                let space = bind_space.read();
                let hdr = hdr_index.read();
                build_search_result_data(&space, &hdr, &results)
            };
            build_search_batch(&batch_data, &schema)
        }
    };

    // Now do async encoding - no locks held
    let schema_clone = schema.clone();
    stream::once(async move {
        match batch_result {
            Ok(batch) => {
                let encoder = FlightDataEncoderBuilder::new()
                    .with_schema(schema_clone)
                    .build(stream::once(async { Ok(batch) }));

                let flight_data: Vec<FlightData> = encoder
                    .filter_map(|r| async { r.ok() })
                    .collect()
                    .await;

                flight_data
            }
            Err(_) => Vec::new(),
        }
    })
    .flat_map(|data| stream::iter(data.into_iter().map(Ok)))
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Build RecordBatch from fingerprint data
fn build_fingerprint_batch(
    data: &[(u16, [u64; FINGERPRINT_WORDS], Option<String>, String)],
    schema: &SchemaRef,
) -> Result<RecordBatch, Status> {
    let addresses: Vec<u16> = data.iter().map(|(a, _, _, _)| *a).collect();
    let fingerprints: Vec<Vec<u8>> = data.iter()
        .map(|(_, fp, _, _)| fp.iter().flat_map(|w| w.to_le_bytes()).collect())
        .collect();
    let labels: Vec<Option<&str>> = data.iter()
        .map(|(_, _, l, _)| l.as_deref())
        .collect();
    let zones: Vec<&str> = data.iter().map(|(_, _, _, z)| z.as_str()).collect();

    let addr_array: ArrayRef = Arc::new(UInt16Array::from(addresses));
    let fp_array: ArrayRef = Arc::new(
        FixedSizeBinaryArray::try_from_iter(fingerprints.iter().map(|v| v.as_slice()))
            .map_err(|e| Status::internal(e.to_string()))?
    );
    let label_array: ArrayRef = Arc::new(StringArray::from(labels));
    let zone_array: ArrayRef = Arc::new(StringArray::from(zones));
    let dist_array: ArrayRef = Arc::new(UInt32Array::from(vec![None::<u32>; data.len()]));
    let sim_array: ArrayRef = Arc::new(Float32Array::from(vec![None::<f32>; data.len()]));

    RecordBatch::try_new(
        schema.clone(),
        vec![addr_array, fp_array, label_array, zone_array, dist_array, sim_array],
    ).map_err(|e| Status::internal(e.to_string()))
}

/// Build search result data from HDR results
fn build_search_result_data(
    _space: &BindSpace,
    _hdr: &HdrIndex,
    results: &[(usize, u32)],
) -> Vec<(u16, [u64; FINGERPRINT_WORDS], Option<String>, u32, f32, u8)> {
    results.iter()
        .filter_map(|(idx, dist)| {
            // Get fingerprint from HDR index
            // Note: We don't have direct address mapping, so we use index as pseudo-address
            // In a real implementation, HDR index would store (addr, fingerprint) pairs
            let addr = *idx as u16;
            let similarity = 1.0 - (*dist as f32 / 10000.0);
            let cascade_level = if *dist < 1000 { 0 } else if *dist < 3000 { 1 } else { 2 };

            // Return placeholder fingerprint - real impl would look up from index
            Some((addr, [0u64; FINGERPRINT_WORDS], None, *dist, similarity, cascade_level))
        })
        .collect()
}

/// Build search result RecordBatch
fn build_search_batch(
    data: &[(u16, [u64; FINGERPRINT_WORDS], Option<String>, u32, f32, u8)],
    schema: &SchemaRef,
) -> Result<RecordBatch, Status> {
    let addresses: Vec<u16> = data.iter().map(|(a, _, _, _, _, _)| *a).collect();
    let fingerprints: Vec<Vec<u8>> = data.iter()
        .map(|(_, fp, _, _, _, _)| fp.iter().flat_map(|w| w.to_le_bytes()).collect())
        .collect();
    let labels: Vec<Option<&str>> = data.iter()
        .map(|(_, _, l, _, _, _)| l.as_deref())
        .collect();
    let distances: Vec<u32> = data.iter().map(|(_, _, _, d, _, _)| *d).collect();
    let similarities: Vec<f32> = data.iter().map(|(_, _, _, _, s, _)| *s).collect();
    let levels: Vec<u8> = data.iter().map(|(_, _, _, _, _, l)| *l).collect();

    let addr_array: ArrayRef = Arc::new(UInt16Array::from(addresses));
    let fp_array: ArrayRef = Arc::new(
        FixedSizeBinaryArray::try_from_iter(fingerprints.iter().map(|v| v.as_slice()))
            .map_err(|e| Status::internal(e.to_string()))?
    );
    let label_array: ArrayRef = Arc::new(StringArray::from(labels));
    let dist_array: ArrayRef = Arc::new(UInt32Array::from(distances));
    let sim_array: ArrayRef = Arc::new(Float32Array::from(similarities));
    let level_array: ArrayRef = Arc::new(UInt8Array::from(levels));

    RecordBatch::try_new(
        schema.clone(),
        vec![addr_array, fp_array, label_array, dist_array, sim_array, level_array],
    ).map_err(|e| Status::internal(e.to_string()))
}

/// Decode FlightData and ingest into BindSpace
fn decode_and_ingest(
    bind_space: &Arc<RwLock<BindSpace>>,
    _data: &FlightData,
) -> Result<usize, Status> {
    // TODO: Implement proper Arrow IPC decoding
    // For now, return 0 ingested
    let _space = bind_space.write();
    Ok(0)
}
