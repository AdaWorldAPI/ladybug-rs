//! Ladybug Arrow Flight Server
//!
//! Implements FlightService for MCP-style interactions with BindSpace.
//!
//! This module provides Arrow Flight RPC endpoints for:
//! - Zero-copy fingerprint streaming (DoGet)
//! - Batch fingerprint ingestion (DoPut)
//! - MCP tool execution (DoAction)
//!
//! # Ticket Formats
//! - `all` - stream all fingerprints
//! - `search:<query_hex>:<threshold>` - similarity search
//! - `topk:<query_hex>:<k>` - top-k search
//!
//! # Actions (MCP Tools)
//! - `encode` - Encode text to fingerprint
//! - `bind` - Bind fingerprint to address
//! - `resonate` - Find similar fingerprints
//! - `hamming` - Compute Hamming distance
//! - `xor_bind` - XOR bind two fingerprints

use std::pin::Pin;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow_flight::{
    flight_service_server::FlightService,
    Action, ActionType, Criteria, Empty, FlightData, FlightDescriptor, FlightInfo,
    HandshakeRequest, HandshakeResponse, PollInfo, PutResult, SchemaAsIpc, SchemaResult,
    Ticket,
};
use futures::Stream;
use parking_lot::RwLock;
use tonic::{Request, Response, Status, Streaming};

use crate::storage::BindSpace;
use crate::search::HdrIndex;

use super::actions::execute_action;

/// Fingerprint schema for Arrow Flight transfers
pub fn fingerprint_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("address", DataType::UInt16, false),
        Field::new("fingerprint", DataType::FixedSizeBinary(1248), false),
        Field::new("label", DataType::Utf8, true),
        Field::new("zone", DataType::Utf8, false),
        Field::new("distance", DataType::UInt32, true),
        Field::new("similarity", DataType::Float32, true),
    ]))
}

/// Search result schema for streaming results
pub fn search_result_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("address", DataType::UInt16, false),
        Field::new("fingerprint", DataType::FixedSizeBinary(1248), false),
        Field::new("label", DataType::Utf8, true),
        Field::new("distance", DataType::UInt32, false),
        Field::new("similarity", DataType::Float32, false),
        Field::new("cascade_level", DataType::UInt8, false),
    ]))
}

/// Ladybug Flight Service for MCP interactions
pub struct LadybugFlightService {
    bind_space: Arc<RwLock<BindSpace>>,
    hdr_index: Arc<RwLock<HdrIndex>>,
}

impl LadybugFlightService {
    /// Create a new Flight service with the given BindSpace and HDR index
    pub fn new(bind_space: Arc<RwLock<BindSpace>>, hdr_index: Arc<RwLock<HdrIndex>>) -> Self {
        Self { bind_space, hdr_index }
    }
}

// Stream type aliases for tonic
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
        let output = futures::stream::once(async {
            Ok(HandshakeResponse {
                protocol_version: 0,
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

        let surface_info = FlightInfo::new()
            .try_with_schema(&schema)
            .map_err(|e| Status::internal(e.to_string()))?
            .with_descriptor(FlightDescriptor::new_path(vec!["surface".to_string()]))
            .with_total_records(4096)
            .with_total_bytes(4096 * 1248);

        let node_info = FlightInfo::new()
            .try_with_schema(&schema)
            .map_err(|e| Status::internal(e.to_string()))?
            .with_descriptor(FlightDescriptor::new_path(vec!["nodes".to_string()]))
            .with_total_records(32768)
            .with_total_bytes(32768 * 1248);

        let flights = vec![surface_info, node_info];
        let stream = futures::stream::iter(flights.into_iter().map(Ok));
        Ok(Response::new(Box::pin(stream)))
    }

    async fn get_flight_info(
        &self,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let descriptor = request.into_inner();
        let schema = fingerprint_schema();

        let info = FlightInfo::new()
            .try_with_schema(&schema)
            .map_err(|e| Status::internal(e.to_string()))?
            .with_descriptor(descriptor);

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
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<SchemaResult>, Status> {
        let schema = fingerprint_schema();
        let options = arrow::ipc::writer::IpcWriteOptions::default();
        let schema_result = SchemaAsIpc::new(&schema, &options)
            .try_into()
            .map_err(|e: arrow::error::ArrowError| Status::internal(e.to_string()))?;
        Ok(Response::new(schema_result))
    }

    async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        let ticket = request.into_inner();
        let ticket_str = String::from_utf8(ticket.ticket.to_vec())
            .map_err(|_| Status::invalid_argument("Invalid ticket encoding"))?;

        // TODO: Implement actual streaming based on ticket type
        // For now, return empty stream
        if ticket_str == "all" || ticket_str.starts_with("search:") || ticket_str.starts_with("topk:") {
            let stream: TonicStream<FlightData> = Box::pin(futures::stream::empty());
            Ok(Response::new(stream))
        } else {
            Err(Status::invalid_argument("Unknown ticket type"))
        }
    }

    async fn do_put(
        &self,
        _request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        // TODO: Implement batch ingestion
        let result = PutResult {
            app_metadata: bytes::Bytes::from("ingested:0"),
        };
        let stream = futures::stream::once(async { Ok(result) });
        Ok(Response::new(Box::pin(stream)))
    }

    async fn do_action(
        &self,
        request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        let action = request.into_inner();
        let action_type = action.r#type.as_str();
        let body = action.body;

        let bind_space = self.bind_space.clone();
        let hdr_index = self.hdr_index.clone();

        let result = execute_action(action_type, &body, bind_space, hdr_index)
            .await
            .map_err(|e| Status::internal(e))?;

        let flight_result = arrow_flight::Result {
            body: bytes::Bytes::from(result),
        };

        let stream = futures::stream::once(async { Ok(flight_result) });
        Ok(Response::new(Box::pin(stream)))
    }

    async fn list_actions(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        let actions = vec![
            ActionType {
                r#type: "encode".to_string(),
                description: "Encode text/data to 10K-bit fingerprint".to_string(),
            },
            ActionType {
                r#type: "bind".to_string(),
                description: "Bind fingerprint to BindSpace address".to_string(),
            },
            ActionType {
                r#type: "resonate".to_string(),
                description: "Find similar fingerprints via HDR cascade".to_string(),
            },
            ActionType {
                r#type: "hamming".to_string(),
                description: "Compute Hamming distance between fingerprints".to_string(),
            },
            ActionType {
                r#type: "xor_bind".to_string(),
                description: "XOR bind two fingerprints".to_string(),
            },
            ActionType {
                r#type: "stats".to_string(),
                description: "Get BindSpace statistics".to_string(),
            },
        ];

        let stream = futures::stream::iter(actions.into_iter().map(Ok));
        Ok(Response::new(Box::pin(stream)))
    }

    async fn do_exchange(
        &self,
        _request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        Err(Status::unimplemented("do_exchange not implemented"))
    }
}
