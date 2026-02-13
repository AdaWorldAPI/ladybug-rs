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
//! - `edges` - stream all edges as (source, target, verb, labels)
//! - `traverse:<source_hex>:<max_hops>[:<verb>]` - BFS graph traversal via CSR
//! - `neighbors:<addr_hex>` - immediate neighbors (1-hop)
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

use std::collections::HashSet;
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
/// - fingerprint: FixedSizeBinary(2048) (256 * 8 bytes, 16K bits)
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

/// Edge schema for graph edge streaming
pub fn edge_flight_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("source", DataType::UInt16, false),
        Field::new("target", DataType::UInt16, false),
        Field::new("verb", DataType::UInt16, false),
        Field::new("source_label", DataType::Utf8, true),
        Field::new("target_label", DataType::Utf8, true),
        Field::new("verb_label", DataType::Utf8, true),
        Field::new("weight", DataType::Float32, false),
    ]))
}

/// Traversal result schema for BFS/DFS via CSR
pub fn traversal_flight_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("source", DataType::UInt16, false),
        Field::new("hop", DataType::UInt32, false),
        Field::new("target", DataType::UInt16, false),
        Field::new("target_label", DataType::Utf8, true),
        Field::new("via_verb", DataType::UInt16, false),
        Field::new("via_verb_label", DataType::Utf8, true),
    ]))
}

// =============================================================================
// FLIGHT SERVICE
// =============================================================================

/// Ladybug Flight Service for MCP interactions
pub struct LadybugFlightService {
    bind_space: Arc<RwLock<BindSpace>>,
    hdr_index: Arc<RwLock<HdrIndex>>,
    #[cfg(feature = "crewai")]
    crew_bridge: Arc<RwLock<crate::orchestration::crew_bridge::CrewBridge>>,
}

impl LadybugFlightService {
    /// Create a new Flight service
    pub fn new(bind_space: Arc<RwLock<BindSpace>>, hdr_index: Arc<RwLock<HdrIndex>>) -> Self {
        Self {
            bind_space,
            hdr_index,
            #[cfg(feature = "crewai")]
            crew_bridge: Arc::new(RwLock::new(crate::orchestration::crew_bridge::CrewBridge::new())),
        }
    }

    /// Create a new Flight service with an existing CrewBridge
    #[cfg(feature = "crewai")]
    pub fn with_crew_bridge(
        bind_space: Arc<RwLock<BindSpace>>,
        hdr_index: Arc<RwLock<HdrIndex>>,
        crew_bridge: Arc<RwLock<crate::orchestration::crew_bridge::CrewBridge>>,
    ) -> Self {
        Self { bind_space, hdr_index, crew_bridge }
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
            "edges" => {
                let output = stream_edges(bind_space);
                Ok(Response::new(Box::pin(output)))
            }
            _ if ticket_str.starts_with("traverse:") => {
                // traverse:<source_hex>:<max_hops>[:<verb_name>]
                let parts: Vec<&str> = ticket_str.split(':').collect();
                if parts.len() < 3 {
                    return Err(Status::invalid_argument(
                        "Invalid traverse format. Use: traverse:<source_hex>:<max_hops>[:<verb>]"
                    ));
                }

                let source: u16 = u16::from_str_radix(parts[1], 16)
                    .map_err(|_| Status::invalid_argument("Invalid source address hex"))?;
                let max_hops: usize = parts[2].parse()
                    .map_err(|_| Status::invalid_argument("Invalid max_hops"))?;
                let verb_name = parts.get(3).map(|s| s.to_string());

                let output = stream_traversal(bind_space, source, max_hops, verb_name);
                Ok(Response::new(Box::pin(output)))
            }
            _ if ticket_str.starts_with("neighbors:") => {
                // neighbors:<addr_hex> — 1-hop convenience
                let parts: Vec<&str> = ticket_str.split(':').collect();
                if parts.len() != 2 {
                    return Err(Status::invalid_argument(
                        "Invalid neighbors format. Use: neighbors:<addr_hex>"
                    ));
                }

                let addr: u16 = u16::from_str_radix(parts[1], 16)
                    .map_err(|_| Status::invalid_argument("Invalid address hex"))?;

                let output = stream_traversal(bind_space, addr, 1, None);
                Ok(Response::new(Box::pin(output)))
            }
            // Orchestration zone tickets (crewai feature)
            "agents" => {
                // Stream prefix 0x0C (agent registry)
                let output = stream_all_fingerprints(bind_space, 0x0C..=0x0C);
                Ok(Response::new(Box::pin(output)))
            }
            "styles" => {
                // Stream prefix 0x0D (thinking templates)
                let output = stream_all_fingerprints(bind_space, 0x0D..=0x0D);
                Ok(Response::new(Box::pin(output)))
            }
            "blackboards" => {
                // Stream prefix 0x0E (agent blackboards)
                let output = stream_all_fingerprints(bind_space, 0x0E..=0x0E);
                Ok(Response::new(Box::pin(output)))
            }
            "a2a" => {
                // Stream prefix 0x0F (A2A channels)
                let output = stream_all_fingerprints(bind_space, 0x0F..=0x0F);
                Ok(Response::new(Box::pin(output)))
            }
            "orchestration" => {
                // Stream all orchestration prefixes 0x0C-0x0F
                let output = stream_all_fingerprints(bind_space, 0x0C..=0x0F);
                Ok(Response::new(Box::pin(output)))
            }
            _ => Err(Status::invalid_argument(format!(
                "Unknown ticket: {}. Use: all, surface, fluid, nodes, edges, agents, styles, blackboards, a2a, orchestration, search:..., topk:..., traverse:..., neighbors:...",
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

        // Route crew/a2a/style/agent actions to CrewBridge when crewai feature is enabled
        #[cfg(feature = "crewai")]
        if action_type.starts_with("crew.")
            || action_type.starts_with("a2a.")
            || action_type.starts_with("style.")
            || action_type.starts_with("agent.")
            || action_type.starts_with("persona.")
            || action_type.starts_with("handover.")
            || action_type.starts_with("orchestrator.")
            || action_type.starts_with("kernel.")
            || action_type.starts_with("filter.")
            || action_type.starts_with("guardrail.")
            || action_type.starts_with("workflow.")
            || action_type.starts_with("memory.")
            || action_type.starts_with("observability.")
            || action_type.starts_with("verification.")
        {
            let result = super::crew_actions::execute_crew_action(
                action_type,
                &body,
                self.crew_bridge.clone(),
                self.bind_space.clone(),
            ).map_err(|e| Status::internal(e))?;

            let flight_result = arrow_flight::Result {
                body: bytes::Bytes::from(result),
            };
            let output = stream::once(async { Ok(flight_result) });
            return Ok(Response::new(Box::pin(output)));
        }

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
            ActionType {
                r#type: "traverse".to_string(),
                description: "BFS graph traversal via CSR. Args: {source, max_hops, verb?}".to_string(),
            },
            ActionType {
                r#type: "hydrate".to_string(),
                description: "Hydrate sparse addresses to fingerprints. Args: {addresses: [u16]}".to_string(),
            },
        ];

        // Append crewAI orchestration actions when the feature is enabled
        #[cfg(feature = "crewai")]
        let actions = {
            let mut actions = actions;
            let crew_actions = vec![
                ActionType {
                    r#type: "crew.register_agent".to_string(),
                    description: "Register agent(s) from YAML. Body: YAML string (agents.yaml format)".to_string(),
                },
                ActionType {
                    r#type: "crew.register_style".to_string(),
                    description: "Register thinking template(s) from YAML. Body: YAML string".to_string(),
                },
                ActionType {
                    r#type: "crew.submit_task".to_string(),
                    description: "Submit task for dispatch. Body: JSON CrewTask".to_string(),
                },
                ActionType {
                    r#type: "crew.dispatch".to_string(),
                    description: "Dispatch full crew. Body: JSON CrewDispatch".to_string(),
                },
                ActionType {
                    r#type: "crew.complete_task".to_string(),
                    description: "Mark task completed. Body: JSON {task_id, outcome}".to_string(),
                },
                ActionType {
                    r#type: "crew.status".to_string(),
                    description: "Get orchestration bridge status. No args".to_string(),
                },
                ActionType {
                    r#type: "crew.bind".to_string(),
                    description: "Bind all orchestration state to BindSpace. No args".to_string(),
                },
                ActionType {
                    r#type: "a2a.send".to_string(),
                    description: "Send A2A message. Body: JSON A2AMessage".to_string(),
                },
                ActionType {
                    r#type: "a2a.receive".to_string(),
                    description: "Receive pending A2A messages. Body: JSON {agent_slot}".to_string(),
                },
                ActionType {
                    r#type: "style.resolve".to_string(),
                    description: "Resolve thinking template by name. Body: template name string".to_string(),
                },
                ActionType {
                    r#type: "style.list".to_string(),
                    description: "List all thinking templates with modulation params. No args".to_string(),
                },
                ActionType {
                    r#type: "agent.list".to_string(),
                    description: "List all registered agents. No args".to_string(),
                },
                ActionType {
                    r#type: "agent.blackboard".to_string(),
                    description: "Get agent blackboard state. Body: JSON {agent_slot}".to_string(),
                },
                ActionType {
                    r#type: "agent.blackboard.yaml".to_string(),
                    description: "Get agent blackboard as YAML handover. Body: JSON {agent_slot}".to_string(),
                },
                ActionType {
                    r#type: "persona.attach".to_string(),
                    description: "Attach persona to agent. Body: JSON {agent_slot, persona}".to_string(),
                },
                ActionType {
                    r#type: "persona.attach_yaml".to_string(),
                    description: "Attach persona from YAML. Body: YAML with agent_slot + persona fields".to_string(),
                },
                ActionType {
                    r#type: "persona.get".to_string(),
                    description: "Get agent persona as JSON. Body: JSON {agent_slot}".to_string(),
                },
                ActionType {
                    r#type: "persona.get_yaml".to_string(),
                    description: "Get agent persona as YAML. Body: JSON {agent_slot}".to_string(),
                },
                ActionType {
                    r#type: "persona.compatible".to_string(),
                    description: "Find compatible agents. Body: JSON {agent_slot, threshold}".to_string(),
                },
                ActionType {
                    r#type: "persona.best_for_task".to_string(),
                    description: "Find best agent for task by volition. Body: task description string".to_string(),
                },
                // Handover actions
                ActionType {
                    r#type: "handover.evaluate".to_string(),
                    description: "Evaluate handover for agent. Body: JSON {agent_slot, task_description?}".to_string(),
                },
                ActionType {
                    r#type: "handover.execute".to_string(),
                    description: "Execute a handover decision. Body: JSON HandoverDecision".to_string(),
                },
                ActionType {
                    r#type: "handover.update_flow".to_string(),
                    description: "Update agent flow state. Body: JSON {agent_slot, gate_state: flow|hold|block}".to_string(),
                },
                // Meta-orchestrator actions
                ActionType {
                    r#type: "orchestrator.status".to_string(),
                    description: "Get orchestrator status (flow states, affinities, events). No args".to_string(),
                },
                ActionType {
                    r#type: "orchestrator.route_task".to_string(),
                    description: "Route task to best agent by resonance. Body: task description string".to_string(),
                },
                ActionType {
                    r#type: "orchestrator.tick".to_string(),
                    description: "Tick orchestrator cycle, evaluate all agents. Returns non-Continue decisions".to_string(),
                },
                ActionType {
                    r#type: "orchestrator.affinity".to_string(),
                    description: "Get affinity between two agents. Body: JSON {agent_a, agent_b}".to_string(),
                },
                ActionType {
                    r#type: "orchestrator.awareness".to_string(),
                    description: "Get total A2A awareness density across all channels. No args".to_string(),
                },
                // Semantic kernel actions
                ActionType {
                    r#type: "kernel.describe".to_string(),
                    description: "Get kernel description (address model, zones, operations, expansions)".to_string(),
                },
                ActionType {
                    r#type: "kernel.introspect".to_string(),
                    description: "Kernel self-observation: population density, hot zones, complexity. No args".to_string(),
                },
                ActionType {
                    r#type: "kernel.zone_density".to_string(),
                    description: "Count populated slots in zone. Body: JSON {prefix}".to_string(),
                },
                ActionType {
                    r#type: "kernel.expansion".to_string(),
                    description: "Get kernel expansion registry summary. No args".to_string(),
                },
                ActionType {
                    r#type: "kernel.prefix_map".to_string(),
                    description: "Get full prefix allocation map. No args".to_string(),
                },
                // Filter pipeline actions (Microsoft SK pattern)
                ActionType {
                    r#type: "filter.add".to_string(),
                    description: "Add filter to pipeline. Body: JSON KernelFilter".to_string(),
                },
                ActionType {
                    r#type: "filter.remove".to_string(),
                    description: "Remove filter by name. Body: filter name string".to_string(),
                },
                ActionType {
                    r#type: "filter.list".to_string(),
                    description: "List all filters in pipeline. No args".to_string(),
                },
                ActionType {
                    r#type: "filter.apply".to_string(),
                    description: "Apply filter pipeline to context. Body: JSON FilterContext".to_string(),
                },
                // Guardrail actions (Amazon Bedrock pattern)
                ActionType {
                    r#type: "guardrail.apply".to_string(),
                    description: "Apply guardrails to fingerprint. Body: JSON {fingerprint, source_addrs?}".to_string(),
                },
                ActionType {
                    r#type: "guardrail.add_topic".to_string(),
                    description: "Add denied topic. Body: JSON DeniedTopic".to_string(),
                },
                ActionType {
                    r#type: "guardrail.enable_grounding".to_string(),
                    description: "Enable grounding checks. Body: JSON {threshold}".to_string(),
                },
                ActionType {
                    r#type: "guardrail.add_content_filter".to_string(),
                    description: "Add content filter. Body: JSON {category, max_severity}".to_string(),
                },
                // Workflow actions (Google ADK + MS Process)
                ActionType {
                    r#type: "workflow.execute".to_string(),
                    description: "Execute workflow DAG. Body: JSON WorkflowNode".to_string(),
                },
                // Memory bank actions (Google Memory Bank)
                ActionType {
                    r#type: "memory.store".to_string(),
                    description: "Store memory. Body: JSON {kind, content, fingerprint, cycle, source_agent?}".to_string(),
                },
                ActionType {
                    r#type: "memory.retrieve".to_string(),
                    description: "Retrieve memories by similarity. Body: JSON {query, kind?, threshold, limit, cycle}".to_string(),
                },
                ActionType {
                    r#type: "memory.extract_semantic".to_string(),
                    description: "Extract semantic memories from episodic. Body: JSON {cycle}".to_string(),
                },
                ActionType {
                    r#type: "memory.list".to_string(),
                    description: "List memories. Body: JSON {kind?}".to_string(),
                },
                // Observability actions (Amazon Session>Trace>Span)
                ActionType {
                    r#type: "observability.start_session".to_string(),
                    description: "Start observability session. Body: JSON {agent_slot?, cycle}".to_string(),
                },
                ActionType {
                    r#type: "observability.start_trace".to_string(),
                    description: "Start trace in active session. Body: JSON {operation, cycle}".to_string(),
                },
                ActionType {
                    r#type: "observability.add_span".to_string(),
                    description: "Add span to trace. Body: JSON {trace_id, span}".to_string(),
                },
                ActionType {
                    r#type: "observability.complete_trace".to_string(),
                    description: "Complete trace. Body: JSON {trace_id, cycle, grounding?}".to_string(),
                },
                ActionType {
                    r#type: "observability.summary".to_string(),
                    description: "Get observability summary. No args".to_string(),
                },
                // Verification actions (Amazon Automated Reasoning)
                ActionType {
                    r#type: "verification.add_rule".to_string(),
                    description: "Add verification rule. Body: JSON VerificationRule".to_string(),
                },
                ActionType {
                    r#type: "verification.verify".to_string(),
                    description: "Verify fingerprint against rules. Body: JSON {fingerprint, addr}".to_string(),
                },
                ActionType {
                    r#type: "verification.list_rules".to_string(),
                    description: "List verification rules. No args".to_string(),
                },
            ];
            actions.extend(crew_actions);
            actions
        };

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
// GRAPH STREAMING (edges + BFS traversal via CSR)
// =============================================================================

/// Stream all edges from BindSpace
fn stream_edges(
    bind_space: Arc<RwLock<BindSpace>>,
) -> impl Stream<Item = Result<FlightData, Status>> {
    let schema = edge_flight_schema();

    // Build all edges synchronously (lock released after block)
    let batch_result = {
        let space = bind_space.read();
        build_edge_batch(&space, &schema)
    };

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

/// Build edge RecordBatch from BindSpace (reads directly from edge list + CSR)
fn build_edge_batch(
    space: &BindSpace,
    schema: &SchemaRef,
) -> Result<RecordBatch, Status> {
    let mut sources = Vec::new();
    let mut targets = Vec::new();
    let mut verbs = Vec::new();
    let mut source_labels = Vec::new();
    let mut target_labels = Vec::new();
    let mut verb_labels = Vec::new();
    let mut weights = Vec::new();

    for prefix in 0u8..=0xFF {
        for slot in 0u8..=0xFF {
            let addr = Addr::new(prefix, slot);
            for edge in space.edges_out(addr) {
                sources.push(edge.from.0);
                targets.push(edge.to.0);
                verbs.push(edge.verb.0);
                source_labels.push(space.read(edge.from).and_then(|n| n.label.clone()));
                target_labels.push(space.read(edge.to).and_then(|n| n.label.clone()));
                verb_labels.push(space.read(edge.verb).and_then(|n| n.label.clone()));
                weights.push(edge.weight);
            }
        }
    }

    if sources.is_empty() {
        return Ok(RecordBatch::new_empty(schema.clone()));
    }

    let columns: Vec<ArrayRef> = vec![
        Arc::new(UInt16Array::from(sources)),
        Arc::new(UInt16Array::from(targets)),
        Arc::new(UInt16Array::from(verbs)),
        Arc::new(StringArray::from(source_labels)),
        Arc::new(StringArray::from(target_labels)),
        Arc::new(StringArray::from(verb_labels)),
        Arc::new(Float32Array::from(weights)),
    ];

    RecordBatch::try_new(schema.clone(), columns)
        .map_err(|e| Status::internal(e.to_string()))
}

/// Stream BFS traversal results from BindSpace via CSR
fn stream_traversal(
    bind_space: Arc<RwLock<BindSpace>>,
    source: u16,
    max_hops: usize,
    verb_name: Option<String>,
) -> impl Stream<Item = Result<FlightData, Status>> {
    let schema = traversal_flight_schema();

    // BFS synchronously (lock released after block)
    let batch_result = {
        let space = bind_space.read();

        // Resolve verb name to address
        let verb_addr = verb_name.as_deref().and_then(|name| space.verb(name));

        build_traversal_batch(&space, source, max_hops, verb_addr, &schema)
    };

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

/// BFS traversal over BindSpace edges, returning results as RecordBatch
fn build_traversal_batch(
    space: &BindSpace,
    source: u16,
    max_hops: usize,
    verb_addr: Option<Addr>,
    schema: &SchemaRef,
) -> Result<RecordBatch, Status> {
    let mut sources_col = Vec::new();
    let mut hops_col = Vec::new();
    let mut targets_col = Vec::new();
    let mut target_labels_col: Vec<Option<String>> = Vec::new();
    let mut via_verbs_col = Vec::new();
    let mut via_verb_labels_col: Vec<Option<String>> = Vec::new();

    let start = Addr(source);
    let mut frontier = vec![start];
    let mut visited = HashSet::new();
    visited.insert(source);

    for hop in 1..=max_hops {
        let mut next_frontier = Vec::new();

        for &node in &frontier {
            for edge in space.edges_out(node) {
                // Verb filter
                if let Some(verb) = verb_addr {
                    if edge.verb != verb {
                        continue;
                    }
                }

                if visited.insert(edge.to.0) {
                    sources_col.push(source);
                    hops_col.push(hop as u32);
                    targets_col.push(edge.to.0);
                    target_labels_col.push(
                        space.read(edge.to).and_then(|n| n.label.clone()),
                    );
                    via_verbs_col.push(edge.verb.0);
                    via_verb_labels_col.push(
                        space.read(edge.verb).and_then(|n| n.label.clone()),
                    );

                    next_frontier.push(edge.to);
                }
            }
        }

        if next_frontier.is_empty() {
            break;
        }
        frontier = next_frontier;
    }

    if sources_col.is_empty() {
        return Ok(RecordBatch::new_empty(schema.clone()));
    }

    let columns: Vec<ArrayRef> = vec![
        Arc::new(UInt16Array::from(sources_col)),
        Arc::new(UInt32Array::from(hops_col)),
        Arc::new(UInt16Array::from(targets_col)),
        Arc::new(StringArray::from(target_labels_col)),
        Arc::new(UInt16Array::from(via_verbs_col)),
        Arc::new(StringArray::from(via_verb_labels_col)),
    ];

    RecordBatch::try_new(schema.clone(), columns)
        .map_err(|e| Status::internal(e.to_string()))
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
            let similarity = 1.0 - (*dist as f32 / crate::FINGERPRINT_BITS as f32);
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
