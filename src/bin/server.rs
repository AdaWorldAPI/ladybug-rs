#![allow(clippy::needless_range_loop, clippy::type_complexity)]
//! LadybugDB Multi-Protocol Server
//!
//! Three transport layers — one BindSpace:
//!
//! ## HTTP (JSON + Arrow IPC)
//! - REST API with Arrow IPC (default) or JSON (fallback) on /api/*
//! - Graph traversal: /api/v1/graph/{traverse,edges,neighbors,hydrate,search}
//! - Redis-compatible text protocol on /redis/*
//! - SQL/Cypher endpoints on /sql, /cypher
//! - Health/readiness on /health, /ready
//!
//! ## gRPC (Flight Arrow)
//! - Via `flight_server` binary on port 50051
//! - Tickets: all, surface, fluid, nodes, edges, traverse:..., neighbors:...
//! - DoAction: encode, bind, resonate, search, traverse, hydrate
//!
//! ## UDP (Bitpacked Hamming)
//! - Binary protocol for ultra-low-latency fingerprint operations
//! - Header: 222 bytes | Payload: 2048 bytes (SIMD-aligned) | Total: 2270 bytes
//! - Ops: PING, SEARCH, TRAVERSE, HYDRATE, EDGES
//! - Fingerprint: 256×u64 = 2048 bytes (16K bits, naturally 64-byte aligned)
//!
//! # Content Negotiation (HTTP)
//!
//! All API endpoints return Arrow IPC by default (`application/vnd.apache.arrow.stream`).
//! JSON is only returned when explicitly requested via `Accept: application/json` header.
//!
//! # Environment Detection
//!
//! - Railway: detects `RAILWAY_*` env vars → binds 0.0.0.0:8080
//! - Claude Code: detects `CLAUDE_*` env vars → binds 127.0.0.1:5432
//! - Custom: set `LADYBUG_HOST`, `LADYBUG_PORT`, `LADYBUG_UDP_PORT`
//! - Default: 127.0.0.1:8080 (HTTP), 127.0.0.1:8081 (UDP)

use std::collections::HashMap;
use std::env;
use std::io::{BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, UdpSocket};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use arrow_array::{
    ArrayRef, BooleanArray, FixedSizeBinaryArray, Float32Array, RecordBatch, StringArray,
    UInt32Array,
};
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use ladybug::core::Fingerprint;
use ladybug::core::simd::{self, hamming_distance};
use ladybug::nars::TruthValue;
use ladybug::storage::service::{CognitiveService, CpuFeatures, ServiceConfig};
use ladybug::storage::{Addr, BindSpace, CogRedis, FINGERPRINT_WORDS, RedisResult};
use ladybug::{FINGERPRINT_BITS, FINGERPRINT_BYTES, VERSION};

// =============================================================================
// CONSTANTS
// =============================================================================

/// Arrow IPC MIME type
const ARROW_MIME: &str = "application/vnd.apache.arrow.stream";

/// JSON MIME type (legacy fallback only)
const JSON_MIME: &str = "application/json";

// =============================================================================
// UDP BITPACKED HAMMING PROTOCOL
// =============================================================================

/// UDP magic bytes
const UDP_MAGIC: [u8; 4] = *b"LDBG";
/// Protocol version
const UDP_VERSION: u8 = 1;
/// Header size in bytes
const UDP_HEADER_SIZE: usize = 222;
/// Raw fingerprint payload: 256 × u64 = 2048 bytes
const UDP_FP_RAW: usize = FINGERPRINT_WORDS * 8;
/// Fingerprint payload: 2048 bytes (256 words, naturally 64-byte aligned, no padding needed)
const UDP_FP_PADDED: usize = FINGERPRINT_WORDS * 8;
/// Maximum datagram size: header + fingerprint = 2270
const UDP_MAX_DATAGRAM: usize = UDP_HEADER_SIZE + UDP_FP_PADDED;
/// Result slots in header reserved area (190 bytes / 2 = 95 u16 addresses)
const UDP_MAX_RESULT_ADDRS: usize = 95;

/// UDP operation codes
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
enum UdpOp {
    Ping = 0,
    Pong = 1,
    Search = 2,
    Traverse = 3,
    Result = 4,
    Edges = 5,
    Hydrate = 6,
}

impl UdpOp {
    fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::Ping),
            1 => Some(Self::Pong),
            2 => Some(Self::Search),
            3 => Some(Self::Traverse),
            4 => Some(Self::Result),
            5 => Some(Self::Edges),
            6 => Some(Self::Hydrate),
            _ => None,
        }
    }
}

/// UDP header flags
const UDP_FLAG_SIMD_ALIGNED: u16 = 0x0001;
const UDP_FLAG_HAS_FINGERPRINT: u16 = 0x0002;
const UDP_FLAG_HAS_POPCOUNT: u16 = 0x0004;

/// UDP packet header (222 bytes)
///
/// Layout:
///   [0..4]    magic: b"LDBG"
///   [4]       version: u8
///   [5]       op: u8
///   [6..8]    flags: u16 LE
///   [8..12]   sequence: u32 LE
///   [12..14]  source_addr: u16 LE (prefix:slot)
///   [14..16]  query_addr: u16 LE
///   [16]      max_hops: u8
///   [17..19]  verb_filter: u16 LE
///   [19]      top_k: u8
///   [20..24]  max_distance: u32 LE
///   [24..28]  popcount_hint: u32 LE
///   [28..30]  result_count: u16 LE
///   [30..32]  payload_len: u16 LE
///   [32..222] reserved/result_addrs (190 bytes)
struct UdpHeader {
    op: UdpOp,
    flags: u16,
    sequence: u32,
    source_addr: u16,
    query_addr: u16,
    max_hops: u8,
    verb_filter: u16,
    top_k: u8,
    max_distance: u32,
    popcount_hint: u32,
    result_count: u16,
    payload_len: u16,
    result_addrs: Vec<u16>,
}

impl UdpHeader {
    fn parse(buf: &[u8]) -> Option<Self> {
        if buf.len() < UDP_HEADER_SIZE {
            return None;
        }
        if buf[0..4] != UDP_MAGIC {
            return None;
        }
        if buf[4] != UDP_VERSION {
            return None;
        }

        let op = UdpOp::from_byte(buf[5])?;
        let flags = u16::from_le_bytes([buf[6], buf[7]]);
        let sequence = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
        let source_addr = u16::from_le_bytes([buf[12], buf[13]]);
        let query_addr = u16::from_le_bytes([buf[14], buf[15]]);
        let max_hops = buf[16];
        let verb_filter = u16::from_le_bytes([buf[17], buf[18]]);
        let top_k = buf[19];
        let max_distance = u32::from_le_bytes([buf[20], buf[21], buf[22], buf[23]]);
        let popcount_hint = u32::from_le_bytes([buf[24], buf[25], buf[26], buf[27]]);
        let result_count = u16::from_le_bytes([buf[28], buf[29]]);
        let payload_len = u16::from_le_bytes([buf[30], buf[31]]);

        // Parse result addresses from reserved area
        let n = (result_count as usize).min(UDP_MAX_RESULT_ADDRS);
        let mut result_addrs = Vec::with_capacity(n);
        for i in 0..n {
            let off = 32 + i * 2;
            if off + 1 < UDP_HEADER_SIZE {
                result_addrs.push(u16::from_le_bytes([buf[off], buf[off + 1]]));
            }
        }

        Some(Self {
            op,
            flags,
            sequence,
            source_addr,
            query_addr,
            max_hops,
            verb_filter,
            top_k,
            max_distance,
            popcount_hint,
            result_count,
            payload_len,
            result_addrs,
        })
    }

    fn encode(&self, result_buf: &mut [u8]) {
        debug_assert!(result_buf.len() >= UDP_HEADER_SIZE);
        result_buf[0..4].copy_from_slice(&UDP_MAGIC);
        result_buf[4] = UDP_VERSION;
        result_buf[5] = self.op as u8;
        result_buf[6..8].copy_from_slice(&self.flags.to_le_bytes());
        result_buf[8..12].copy_from_slice(&self.sequence.to_le_bytes());
        result_buf[12..14].copy_from_slice(&self.source_addr.to_le_bytes());
        result_buf[14..16].copy_from_slice(&self.query_addr.to_le_bytes());
        result_buf[16] = self.max_hops;
        result_buf[17..19].copy_from_slice(&self.verb_filter.to_le_bytes());
        result_buf[19] = self.top_k;
        result_buf[20..24].copy_from_slice(&self.max_distance.to_le_bytes());
        result_buf[24..28].copy_from_slice(&self.popcount_hint.to_le_bytes());
        result_buf[28..30].copy_from_slice(&self.result_count.to_le_bytes());
        result_buf[30..32].copy_from_slice(&self.payload_len.to_le_bytes());

        // Write result addresses
        for (i, &addr) in self
            .result_addrs
            .iter()
            .enumerate()
            .take(UDP_MAX_RESULT_ADDRS)
        {
            let off = 32 + i * 2;
            if off + 1 < UDP_HEADER_SIZE {
                result_buf[off..off + 2].copy_from_slice(&addr.to_le_bytes());
            }
        }

        // Zero remaining reserved bytes
        let written = 32 + self.result_addrs.len().min(UDP_MAX_RESULT_ADDRS) * 2;
        for b in result_buf[written..UDP_HEADER_SIZE].iter_mut() {
            *b = 0;
        }
    }
}

// =============================================================================
// CONFIGURATION
// =============================================================================

#[derive(Debug, Clone)]
struct ServerConfig {
    host: String,
    port: u16,
    data_dir: String,
    environment: Environment,
    workers: usize,
}

#[derive(Debug, Clone, PartialEq)]
enum Environment {
    Railway,
    ClaudeCode,
    Docker,
    Local,
}

impl ServerConfig {
    fn from_env() -> Self {
        let environment = detect_environment();

        let (default_host, default_port) = match &environment {
            Environment::Railway => ("0.0.0.0", 8080u16),
            Environment::ClaudeCode => ("127.0.0.1", 5432),
            Environment::Docker => ("0.0.0.0", 8080),
            Environment::Local => ("127.0.0.1", 8080),
        };

        let host = env::var("LADYBUG_HOST")
            .or_else(|_| env::var("HOST"))
            .unwrap_or_else(|_| default_host.to_string());

        let port = env::var("LADYBUG_PORT")
            .or_else(|_| env::var("PORT"))
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(default_port);

        let data_dir = env::var("LADYBUG_DATA_DIR").unwrap_or_else(|_| "./data".to_string());

        let cpu = CpuFeatures::detect();

        Self {
            host,
            port,
            data_dir,
            environment,
            workers: cpu.optimal_workers(),
        }
    }
}

fn detect_environment() -> Environment {
    // Railway detection: check for RAILWAY_* env vars or hostname
    if env::var("RAILWAY_ENVIRONMENT").is_ok()
        || env::var("RAILWAY_PROJECT_ID").is_ok()
        || hostname_matches("railway.internal")
    {
        return Environment::Railway;
    }

    // Claude Code detection
    if env::var("CLAUDE_CODE").is_ok() || env::var("CLAUDE_SESSION_ID").is_ok() {
        return Environment::ClaudeCode;
    }

    // Docker detection
    if std::path::Path::new("/.dockerenv").exists() || env::var("DOCKER_CONTAINER").is_ok() {
        return Environment::Docker;
    }

    Environment::Local
}

fn hostname_matches(pattern: &str) -> bool {
    if let Ok(hostname) = std::fs::read_to_string("/etc/hostname") {
        return hostname.trim().contains(pattern);
    }
    // Also check via env
    if let Ok(hostname) = env::var("HOSTNAME") {
        return hostname.contains(pattern);
    }
    // Check RAILWAY_PRIVATE_DOMAIN
    if let Ok(domain) = env::var("RAILWAY_PRIVATE_DOMAIN") {
        return domain.contains(pattern);
    }
    false
}

// =============================================================================
// ARROW SCHEMAS
// =============================================================================

/// Schema for fingerprint responses
fn fingerprint_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(
            "fingerprint",
            DataType::FixedSizeBinary(FINGERPRINT_BYTES as i32),
            false,
        ),
        Field::new("popcount", DataType::UInt32, false),
        Field::new("density", DataType::Float32, false),
        Field::new("bits", DataType::UInt32, false),
    ]))
}

/// Schema for distance/similarity responses
fn distance_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("distance", DataType::UInt32, false),
        Field::new("similarity", DataType::Float32, false),
        Field::new("bits", DataType::UInt32, false),
    ]))
}

/// Schema for search result responses
fn search_result_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("index", DataType::UInt32, false),
        Field::new("id", DataType::Utf8, false),
        Field::new("distance", DataType::UInt32, false),
        Field::new("similarity", DataType::Float32, false),
        Field::new("metadata", DataType::Utf8, true),
    ]))
}

/// Schema for index operation responses
fn index_result_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("success", DataType::Boolean, false),
        Field::new("id", DataType::Utf8, false),
        Field::new("index", DataType::UInt32, false),
        Field::new("total", DataType::UInt32, false),
    ]))
}

/// Schema for NARS inference responses
fn nars_result_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("frequency", DataType::Float32, false),
        Field::new("confidence", DataType::Float32, false),
        Field::new("expectation", DataType::Float32, false),
    ]))
}

/// Schema for health/info responses
fn health_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("status", DataType::Utf8, false),
        Field::new("version", DataType::Utf8, false),
        Field::new("simd_level", DataType::Utf8, false),
        Field::new("uptime_secs", DataType::UInt32, false),
        Field::new("indexed_count", DataType::UInt32, false),
    ]))
}

/// Schema for count responses
fn count_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new(
        "count",
        DataType::UInt32,
        false,
    )]))
}

/// Schema for graph edge responses
fn graph_edge_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("source", DataType::UInt32, false),
        Field::new("target", DataType::UInt32, false),
        Field::new("verb", DataType::UInt32, false),
        Field::new("source_label", DataType::Utf8, true),
        Field::new("target_label", DataType::Utf8, true),
        Field::new("verb_label", DataType::Utf8, true),
        Field::new("weight", DataType::Float32, false),
    ]))
}

/// Schema for graph traversal responses
fn graph_traversal_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("hop", DataType::UInt32, false),
        Field::new("address", DataType::UInt32, false),
        Field::new("label", DataType::Utf8, true),
        Field::new("popcount", DataType::UInt32, false),
        Field::new("distance_from_source", DataType::UInt32, true),
    ]))
}

/// Schema for hydration responses (sparse addr → full fingerprint)
fn hydrate_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("address", DataType::UInt32, false),
        Field::new("label", DataType::Utf8, true),
        Field::new(
            "fingerprint",
            DataType::FixedSizeBinary(FINGERPRINT_BYTES as i32),
            false,
        ),
        Field::new("popcount", DataType::UInt32, false),
    ]))
}

// =============================================================================
// ARROW IPC ENCODING
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

// =============================================================================
// IN-MEMORY DATABASE STATE
// =============================================================================

struct DbState {
    /// Indexed fingerprints with metadata
    fingerprints: Vec<(String, Fingerprint, HashMap<String, String>)>,
    /// Key-value store (simple fallback)
    #[allow(dead_code)]
    kv: HashMap<String, String>,
    /// Full CogRedis interface (DN commands, CAM ops, etc.)
    cog_redis: CogRedis,
    /// Service container
    service: CognitiveService,
    /// CPU features
    cpu: CpuFeatures,
    /// Start time
    start_time: Instant,
}

impl DbState {
    fn new(config: &ServerConfig) -> Self {
        let svc_config = ServiceConfig {
            data_dir: config.data_dir.clone().into(),
            ..Default::default()
        };

        let service = CognitiveService::new(svc_config).expect("Failed to create CognitiveService");

        Self {
            fingerprints: Vec::new(),
            kv: HashMap::new(),
            cog_redis: CogRedis::new(),
            service,
            cpu: CpuFeatures::detect(),
            start_time: Instant::now(),
        }
    }
}

type SharedState = Arc<RwLock<DbState>>;

// =============================================================================
// RESPONSE FORMAT SELECTION
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
enum ResponseFormat {
    Arrow, // Default - Arrow IPC
    Json,  // Legacy fallback only
}

fn parse_accept_header(headers: &HashMap<String, String>) -> ResponseFormat {
    if let Some(accept) = headers.get("accept") {
        // If client explicitly requests JSON, give them JSON
        // Otherwise, default to Arrow
        if accept.contains("application/json") && !accept.contains(ARROW_MIME) {
            return ResponseFormat::Json;
        }
    }
    // Default to Arrow IPC
    ResponseFormat::Arrow
}

// =============================================================================
// HTTP HANDLER
// =============================================================================

fn handle_connection(stream: &mut TcpStream, state: &SharedState) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut request_line = String::new();

    if reader.read_line(&mut request_line).is_err() {
        return;
    }

    // Parse method and path
    let parts: Vec<&str> = request_line.split_whitespace().collect();
    if parts.len() < 2 {
        let resp = http_arrow_error(400, "bad_request");
        let _ = stream.write_all(&resp);
        let _ = stream.flush();
        return;
    }

    let method = parts[0];
    let path = parts[1];

    // Read headers
    let mut headers = HashMap::new();
    let mut content_length: usize = 0;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).is_err() || line.trim().is_empty() {
            break;
        }
        if let Some((key, val)) = line.trim().split_once(':') {
            let key = key.trim().to_lowercase();
            let val = val.trim().to_string();
            if key == "content-length" {
                content_length = val.parse().unwrap_or(0);
            }
            headers.insert(key, val);
        }
    }

    // Read body
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        let _ = std::io::Read::read_exact(&mut reader, &mut body);
    }
    let body_str = String::from_utf8_lossy(&body).to_string();

    // Determine response format from Accept header
    let format = parse_accept_header(&headers);

    // Route
    let response = route(method, path, &body_str, state, format);
    let _ = stream.write_all(&response);
    let _ = stream.flush();
}

fn route(
    method: &str,
    path: &str,
    body: &str,
    state: &SharedState,
    format: ResponseFormat,
) -> Vec<u8> {
    match (method, path) {
        // Health endpoints - always return appropriate format
        ("GET", "/health") | ("GET", "/healthz") => handle_health(state, format),
        ("GET", "/ready") | ("GET", "/readyz") => handle_ready(format),
        ("GET", "/") => handle_root(state, format),

        // Info
        ("GET", "/api/v1/info") => handle_info(state, format),
        ("GET", "/api/v1/simd") => handle_simd(format),

        // Fingerprint operations
        ("POST", "/api/v1/fingerprint") => handle_fingerprint_create(body, format),
        ("POST", "/api/v1/fingerprint/batch") => handle_fingerprint_batch(body, format),
        ("POST", "/api/v1/hamming") => handle_hamming(body, format),
        ("POST", "/api/v1/similarity") => handle_similarity(body, state, format),
        ("POST", "/api/v1/bind") => handle_bind(body, format),
        ("POST", "/api/v1/bundle") => handle_bundle(body, format),

        // Search
        ("POST", "/api/v1/search/topk") => handle_topk(body, state, format),
        ("POST", "/api/v1/search/threshold") => handle_threshold(body, state, format),
        ("POST", "/api/v1/search/resonate") => handle_resonate(body, state, format),

        // Index operations
        ("POST", "/api/v1/index") => handle_index(body, state, format),
        ("GET", "/api/v1/index/count") => handle_index_count(state, format),
        ("DELETE", "/api/v1/index") => handle_index_clear(state, format),

        // NARS inference
        ("POST", "/api/v1/nars/deduction") => handle_nars_deduction(body, format),
        ("POST", "/api/v1/nars/induction") => handle_nars_induction(body, format),
        ("POST", "/api/v1/nars/abduction") => handle_nars_abduction(body, format),
        ("POST", "/api/v1/nars/revision") => handle_nars_revision(body, format),

        // SQL endpoint
        ("POST", "/api/v1/sql") | ("POST", "/sql") => handle_sql(body, state, format),

        // Cypher endpoint
        ("POST", "/api/v1/cypher") | ("POST", "/cypher") => handle_cypher(body, format),

        // CogRedis text protocol - always uses Redis wire protocol
        ("POST", "/redis") => handle_redis_command(body, state),

        // Graph traversal endpoints (BindSpace CSR + edges)
        ("POST", "/api/v1/graph/traverse") => handle_graph_traverse(body, state, format),
        ("POST", "/api/v1/graph/edges") => handle_graph_edges(body, state, format),
        ("GET", "/api/v1/graph/edges") => handle_graph_edges(body, state, format),
        ("POST", "/api/v1/graph/neighbors") => handle_graph_neighbors(body, state, format),
        ("POST", "/api/v1/graph/hydrate") => handle_graph_hydrate(body, state, format),
        ("POST", "/api/v1/graph/search") => handle_graph_search(body, state, format),

        // LanceDB-compatible API
        ("POST", "/api/v1/lance/table") => handle_lance_create_table(body, format),
        ("POST", "/api/v1/lance/add") => handle_lance_add(body, state, format),
        ("POST", "/api/v1/lance/search") => handle_lance_search(body, state, format),

        _ => http_error(404, "not_found", "Unknown endpoint", format),
    }
}

// =============================================================================
// HANDLER IMPLEMENTATIONS
// =============================================================================

fn handle_health(state: &SharedState, format: ResponseFormat) -> Vec<u8> {
    let db = state.read().unwrap();
    let health = db.service.health_check();
    let uptime = db.start_time.elapsed().as_secs() as u32;
    let count = db.fingerprints.len() as u32;

    match format {
        ResponseFormat::Arrow => {
            let schema = health_schema();
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(StringArray::from(vec!["ok"])) as ArrayRef,
                    Arc::new(StringArray::from(vec![VERSION])) as ArrayRef,
                    Arc::new(StringArray::from(vec![health.cpu_features.as_str()])) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![uptime])) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![count])) as ArrayRef,
                ],
            )
            .unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => {
            let json = format!(
                r#"{{"status":"ok","uptime_secs":{},"cpu":"{}","buffer_pool_used":{},"version":"{}"}}"#,
                health.uptime_secs, health.cpu_features, health.buffer_pool_used, VERSION
            );
            http_json(200, &json)
        }
    }
}

fn handle_ready(format: ResponseFormat) -> Vec<u8> {
    match format {
        ResponseFormat::Arrow => {
            let schema = Arc::new(Schema::new(vec![Field::new(
                "status",
                DataType::Utf8,
                false,
            )]));
            let batch = RecordBatch::try_new(
                schema,
                vec![Arc::new(StringArray::from(vec!["ready"])) as ArrayRef],
            )
            .unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => http_json(200, r#"{"status":"ready"}"#),
    }
}

fn handle_root(state: &SharedState, format: ResponseFormat) -> Vec<u8> {
    handle_health(state, format) // Same as health for root
}

fn handle_info(state: &SharedState, format: ResponseFormat) -> Vec<u8> {
    let db = state.read().unwrap();
    let uptime = db.start_time.elapsed().as_secs() as u32;
    let count = db.fingerprints.len() as u32;

    match format {
        ResponseFormat::Arrow => {
            let schema = health_schema();
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec!["ok"])) as ArrayRef,
                    Arc::new(StringArray::from(vec![VERSION])) as ArrayRef,
                    Arc::new(StringArray::from(vec![simd::simd_level()])) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![uptime])) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![count])) as ArrayRef,
                ],
            )
            .unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => {
            let json = format!(
                r#"{{"version":"{}","fingerprint_bits":{},"fingerprint_bytes":{},"simd":"{}","cpu":{{"avx512":{},"avx2":{},"cores":{}}},"indexed_count":{}}}"#,
                VERSION,
                FINGERPRINT_BITS,
                FINGERPRINT_BYTES,
                simd::simd_level(),
                db.cpu.has_avx512f,
                db.cpu.has_avx2,
                db.cpu.physical_cores,
                db.fingerprints.len()
            );
            http_json(200, &json)
        }
    }
}

fn handle_simd(format: ResponseFormat) -> Vec<u8> {
    let cpu = CpuFeatures::detect();

    match format {
        ResponseFormat::Arrow => {
            let schema = Arc::new(Schema::new(vec![
                Field::new("level", DataType::Utf8, false),
                Field::new("avx512f", DataType::Boolean, false),
                Field::new("avx512vpopcntdq", DataType::Boolean, false),
                Field::new("avx2", DataType::Boolean, false),
                Field::new("physical_cores", DataType::UInt32, false),
            ]));
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec![simd::simd_level()])) as ArrayRef,
                    Arc::new(BooleanArray::from(vec![cpu.has_avx512f])) as ArrayRef,
                    Arc::new(BooleanArray::from(vec![cpu.has_avx512vpopcntdq])) as ArrayRef,
                    Arc::new(BooleanArray::from(vec![cpu.has_avx2])) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![cpu.physical_cores as u32])) as ArrayRef,
                ],
            )
            .unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => {
            let json = format!(
                r#"{{"level":"{}","avx512f":{},"avx512vpopcntdq":{},"avx2":{},"sse42":{},"physical_cores":{},"optimal_batch_size":{}}}"#,
                simd::simd_level(),
                cpu.has_avx512f,
                cpu.has_avx512vpopcntdq,
                cpu.has_avx2,
                cpu.has_sse42,
                cpu.physical_cores,
                cpu.optimal_batch_size()
            );
            http_json(200, &json)
        }
    }
}

fn handle_fingerprint_create(body: &str, format: ResponseFormat) -> Vec<u8> {
    // Parse JSON input (input is always JSON for backwards compat)
    let fp = if let Some(content) =
        extract_json_str(body, "text").or_else(|| extract_json_str(body, "content"))
    {
        Fingerprint::from_content(&content)
    } else if let Some(b64) = extract_json_str(body, "bytes") {
        match base64_decode(&b64) {
            Ok(bytes) => match Fingerprint::from_bytes(&bytes) {
                Ok(fp) => fp,
                Err(e) => return http_error(400, "invalid_fingerprint", &e.to_string(), format),
            },
            Err(e) => return http_error(400, "invalid_base64", &e, format),
        }
    } else {
        Fingerprint::random()
    };

    match format {
        ResponseFormat::Arrow => {
            let schema = fingerprint_schema();
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(
                        FixedSizeBinaryArray::try_from_iter(std::iter::once(fp.as_bytes()))
                            .unwrap(),
                    ) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![fp.popcount()])) as ArrayRef,
                    Arc::new(Float32Array::from(vec![fp.density()])) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![FINGERPRINT_BITS as u32])) as ArrayRef,
                ],
            )
            .unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => {
            let b64 = base64_encode(fp.as_bytes());
            let json = format!(
                r#"{{"fingerprint":"{}","popcount":{},"density":{:.4},"bits":{}}}"#,
                b64,
                fp.popcount(),
                fp.density(),
                FINGERPRINT_BITS
            );
            http_json(200, &json)
        }
    }
}

fn handle_fingerprint_batch(body: &str, format: ResponseFormat) -> Vec<u8> {
    let contents = extract_json_str_array(body, "contents");
    if contents.is_empty() {
        return http_error(400, "missing_field", "need contents array", format);
    }

    let fps: Vec<Fingerprint> = contents
        .iter()
        .map(|c| Fingerprint::from_content(c))
        .collect();

    match format {
        ResponseFormat::Arrow => {
            let schema = fingerprint_schema();
            let fp_bytes: Vec<&[u8]> = fps.iter().map(|fp| fp.as_bytes()).collect();
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(FixedSizeBinaryArray::try_from_iter(fp_bytes.into_iter()).unwrap())
                        as ArrayRef,
                    Arc::new(UInt32Array::from(
                        fps.iter().map(|fp| fp.popcount()).collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(Float32Array::from(
                        fps.iter().map(|fp| fp.density()).collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![FINGERPRINT_BITS as u32; fps.len()]))
                        as ArrayRef,
                ],
            )
            .unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => {
            let results: Vec<String> = fps
                .iter()
                .zip(contents.iter())
                .map(|(fp, c)| {
                    let b64 = base64_encode(fp.as_bytes());
                    format!(
                        r#"{{"content":"{}","fingerprint":"{}","popcount":{},"density":{:.4}}}"#,
                        c,
                        b64,
                        fp.popcount(),
                        fp.density()
                    )
                })
                .collect();
            let json = format!(
                r#"{{"fingerprints":[{}],"count":{}}}"#,
                results.join(","),
                results.len()
            );
            http_json(200, &json)
        }
    }
}

fn handle_hamming(body: &str, format: ResponseFormat) -> Vec<u8> {
    let a_str = extract_json_str(body, "a").unwrap_or_default();
    let b_str = extract_json_str(body, "b").unwrap_or_default();

    let fp_a = resolve_fingerprint(&a_str);
    let fp_b = resolve_fingerprint(&b_str);

    let dist = hamming_distance(&fp_a, &fp_b);
    let sim = 1.0 - (dist as f32 / FINGERPRINT_BITS as f32);

    match format {
        ResponseFormat::Arrow => {
            let schema = distance_schema();
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(UInt32Array::from(vec![dist])) as ArrayRef,
                    Arc::new(Float32Array::from(vec![sim])) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![FINGERPRINT_BITS as u32])) as ArrayRef,
                ],
            )
            .unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => {
            let json = format!(
                r#"{{"distance":{},"similarity":{:.6},"bits":{}}}"#,
                dist, sim, FINGERPRINT_BITS
            );
            http_json(200, &json)
        }
    }
}

fn handle_similarity(body: &str, _state: &SharedState, format: ResponseFormat) -> Vec<u8> {
    handle_hamming(body, format)
}

fn handle_bind(body: &str, format: ResponseFormat) -> Vec<u8> {
    let a_str = extract_json_str(body, "a").unwrap_or_default();
    let b_str = extract_json_str(body, "b").unwrap_or_default();

    let fp_a = resolve_fingerprint(&a_str);
    let fp_b = resolve_fingerprint(&b_str);
    let result = fp_a.bind(&fp_b);

    match format {
        ResponseFormat::Arrow => {
            let schema = fingerprint_schema();
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(
                        FixedSizeBinaryArray::try_from_iter(std::iter::once(result.as_bytes()))
                            .unwrap(),
                    ) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![result.popcount()])) as ArrayRef,
                    Arc::new(Float32Array::from(vec![result.density()])) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![FINGERPRINT_BITS as u32])) as ArrayRef,
                ],
            )
            .unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => {
            let json = format!(
                r#"{{"result":"{}","popcount":{},"density":{:.4}}}"#,
                base64_encode(result.as_bytes()),
                result.popcount(),
                result.density()
            );
            http_json(200, &json)
        }
    }
}

fn handle_bundle(body: &str, format: ResponseFormat) -> Vec<u8> {
    let fps_b64 = extract_json_str_array(body, "fingerprints");
    if fps_b64.is_empty() {
        return http_error(400, "missing_field", "need fingerprints array", format);
    }

    let fps: Vec<Fingerprint> = fps_b64.iter().map(|s| resolve_fingerprint(s)).collect();

    // Majority vote bundling
    let threshold = fps.len() / 2;
    let mut result = Fingerprint::zero();
    for bit in 0..FINGERPRINT_BITS {
        let count: usize = fps.iter().filter(|fp| fp.get_bit(bit)).count();
        if count > threshold {
            result.set_bit(bit, true);
        }
    }

    match format {
        ResponseFormat::Arrow => {
            let schema = fingerprint_schema();
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(
                        FixedSizeBinaryArray::try_from_iter(std::iter::once(result.as_bytes()))
                            .unwrap(),
                    ) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![result.popcount()])) as ArrayRef,
                    Arc::new(Float32Array::from(vec![result.density()])) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![FINGERPRINT_BITS as u32])) as ArrayRef,
                ],
            )
            .unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => {
            let json = format!(
                r#"{{"result":"{}","popcount":{},"density":{:.4},"input_count":{}}}"#,
                base64_encode(result.as_bytes()),
                result.popcount(),
                result.density(),
                fps.len()
            );
            http_json(200, &json)
        }
    }
}

fn handle_topk(body: &str, state: &SharedState, format: ResponseFormat) -> Vec<u8> {
    let query_str = extract_json_str(body, "query").unwrap_or_default();
    let k = extract_json_usize(body, "k").unwrap_or(10);
    let style = extract_json_str(body, "style").unwrap_or_else(|| "balanced".to_string());

    let query = resolve_fingerprint(&query_str);
    let db = state.read().unwrap();

    let diversity_boost = match style.as_str() {
        "creative" => 0.1_f32,
        _ => 0.0_f32,
    };

    let mut scored: Vec<(usize, u32, f32)> = db
        .fingerprints
        .iter()
        .enumerate()
        .map(|(i, (_, fp, _))| {
            let dist = hamming_distance(&query, fp);
            let base_sim = 1.0 - (dist as f32 / FINGERPRINT_BITS as f32);
            let sim = base_sim + diversity_boost * ((i % 7) as f32 / 100.0);
            (i, dist, sim)
        })
        .collect();

    scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);

    match format {
        ResponseFormat::Arrow => {
            let schema = search_result_schema();
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(UInt32Array::from(
                        scored.iter().map(|(i, _, _)| *i as u32).collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(StringArray::from(
                        scored
                            .iter()
                            .map(|(i, _, _)| db.fingerprints[*i].0.as_str())
                            .collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(UInt32Array::from(
                        scored.iter().map(|(_, d, _)| *d).collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(Float32Array::from(
                        scored.iter().map(|(_, _, s)| *s).collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(StringArray::from(
                        scored
                            .iter()
                            .map(|(i, _, _)| {
                                let meta = &db.fingerprints[*i].2;
                                if meta.is_empty() {
                                    None
                                } else {
                                    Some(format!("{:?}", meta))
                                }
                            })
                            .collect::<Vec<_>>(),
                    )) as ArrayRef,
                ],
            )
            .unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => {
            let results: Vec<String> = scored.iter().map(|&(idx, dist, sim)| {
                let (id, _, meta) = &db.fingerprints[idx];
                let meta_json = meta.iter()
                    .map(|(k, v)| format!(r#""{}":"{}""#, k, v))
                    .collect::<Vec<_>>().join(",");
                format!(
                    r#"{{"index":{},"id":"{}","distance":{},"similarity":{:.6},"metadata":{{{}}}}}"#,
                    idx, id, dist, sim, meta_json
                )
            }).collect();

            let json = format!(
                r#"{{"results":[{}],"count":{},"style":"{}","total_indexed":{}}}"#,
                results.join(","),
                results.len(),
                style,
                db.fingerprints.len()
            );
            http_json(200, &json)
        }
    }
}

fn handle_threshold(body: &str, state: &SharedState, format: ResponseFormat) -> Vec<u8> {
    let query_str = extract_json_str(body, "query").unwrap_or_default();
    let max_distance = extract_json_usize(body, "max_distance").unwrap_or(2000) as u32;
    let limit = extract_json_usize(body, "limit").unwrap_or(100);

    let query = resolve_fingerprint(&query_str);
    let db = state.read().unwrap();

    let mut results: Vec<(usize, u32, f32)> = Vec::new();
    for (idx, (_, fp, _)) in db.fingerprints.iter().enumerate() {
        let dist = hamming_distance(&query, fp);
        if dist <= max_distance {
            let sim = 1.0 - (dist as f32 / FINGERPRINT_BITS as f32);
            results.push((idx, dist, sim));
            if results.len() >= limit {
                break;
            }
        }
    }

    match format {
        ResponseFormat::Arrow => {
            let schema = search_result_schema();
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(UInt32Array::from(
                        results
                            .iter()
                            .map(|(i, _, _)| *i as u32)
                            .collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(StringArray::from(
                        results
                            .iter()
                            .map(|(i, _, _)| db.fingerprints[*i].0.as_str())
                            .collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(UInt32Array::from(
                        results.iter().map(|(_, d, _)| *d).collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(Float32Array::from(
                        results.iter().map(|(_, _, s)| *s).collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(StringArray::from(vec![None::<&str>; results.len()])) as ArrayRef,
                ],
            )
            .unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => {
            let results_json: Vec<String> = results.iter().map(|&(idx, dist, sim)| {
                let (id, _, meta) = &db.fingerprints[idx];
                let meta_json = meta.iter()
                    .map(|(k, v)| format!(r#""{}":"{}""#, k, v))
                    .collect::<Vec<_>>().join(",");
                format!(
                    r#"{{"index":{},"id":"{}","distance":{},"similarity":{:.6},"metadata":{{{}}}}}"#,
                    idx, id, dist, sim, meta_json
                )
            }).collect();

            let json = format!(
                r#"{{"results":[{}],"count":{},"max_distance":{},"total_indexed":{}}}"#,
                results_json.join(","),
                results.len(),
                max_distance,
                db.fingerprints.len()
            );
            http_json(200, &json)
        }
    }
}

fn handle_resonate(body: &str, state: &SharedState, format: ResponseFormat) -> Vec<u8> {
    let content = extract_json_str(body, "content").unwrap_or_default();
    let threshold = extract_json_f32(body, "threshold").unwrap_or(0.7);
    let limit = extract_json_usize(body, "limit").unwrap_or(10);

    let query = Fingerprint::from_content(&content);
    let db = state.read().unwrap();

    let mut scored: Vec<(usize, f32)> = db
        .fingerprints
        .iter()
        .enumerate()
        .filter_map(|(i, (_, fp, _))| {
            let sim = 1.0 - (hamming_distance(&query, fp) as f32 / FINGERPRINT_BITS as f32);
            if sim >= threshold {
                Some((i, sim))
            } else {
                None
            }
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.truncate(limit);

    match format {
        ResponseFormat::Arrow => {
            let schema = search_result_schema();
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(UInt32Array::from(
                        scored.iter().map(|(i, _)| *i as u32).collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(StringArray::from(
                        scored
                            .iter()
                            .map(|(i, _)| db.fingerprints[*i].0.as_str())
                            .collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![0u32; scored.len()])) as ArrayRef, // Distance not computed
                    Arc::new(Float32Array::from(
                        scored.iter().map(|(_, s)| *s).collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(StringArray::from(vec![None::<&str>; scored.len()])) as ArrayRef,
                ],
            )
            .unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => {
            let results: Vec<String> = scored
                .iter()
                .map(|&(idx, sim)| {
                    let (id, _, _) = &db.fingerprints[idx];
                    format!(
                        r#"{{"index":{},"id":"{}","similarity":{:.6}}}"#,
                        idx, id, sim
                    )
                })
                .collect();

            let json = format!(
                r#"{{"results":[{}],"count":{},"content":"{}","threshold":{}}}"#,
                results.join(","),
                results.len(),
                content,
                threshold
            );
            http_json(200, &json)
        }
    }
}

fn handle_index(body: &str, state: &SharedState, format: ResponseFormat) -> Vec<u8> {
    let id = extract_json_str(body, "id").unwrap_or_else(uuid_v4);

    let fp = if let Some(content) = extract_json_str(body, "content") {
        Fingerprint::from_content(&content)
    } else if let Some(b64) = extract_json_str(body, "fingerprint") {
        resolve_fingerprint(&b64)
    } else {
        return http_error(400, "missing_field", "need content or fingerprint", format);
    };

    let meta = extract_json_object(body, "metadata");

    let mut db = state.write().unwrap();
    let idx = db.fingerprints.len();
    db.fingerprints.push((id.clone(), fp, meta));

    match format {
        ResponseFormat::Arrow => {
            let schema = index_result_schema();
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(BooleanArray::from(vec![true])) as ArrayRef,
                    Arc::new(StringArray::from(vec![id.as_str()])) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![idx as u32])) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![db.fingerprints.len() as u32])) as ArrayRef,
                ],
            )
            .unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => {
            let json = format!(
                r#"{{"success":true,"id":"{}","index":{},"total":{}}}"#,
                id,
                idx,
                db.fingerprints.len()
            );
            http_json(200, &json)
        }
    }
}

fn handle_index_count(state: &SharedState, format: ResponseFormat) -> Vec<u8> {
    let db = state.read().unwrap();
    let count = db.fingerprints.len() as u32;

    match format {
        ResponseFormat::Arrow => {
            let schema = count_schema();
            let batch = RecordBatch::try_new(
                schema,
                vec![Arc::new(UInt32Array::from(vec![count])) as ArrayRef],
            )
            .unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => http_json(200, &format!(r#"{{"count":{}}}"#, count)),
    }
}

fn handle_index_clear(state: &SharedState, format: ResponseFormat) -> Vec<u8> {
    let mut db = state.write().unwrap();
    let was = db.fingerprints.len() as u32;
    db.fingerprints.clear();

    match format {
        ResponseFormat::Arrow => {
            let schema = Arc::new(Schema::new(vec![
                Field::new("cleared", DataType::Boolean, false),
                Field::new("was", DataType::UInt32, false),
            ]));
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(BooleanArray::from(vec![true])) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![was])) as ArrayRef,
                ],
            )
            .unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => http_json(200, &format!(r#"{{"cleared":true,"was":{}}}"#, was)),
    }
}

// NARS handlers
fn handle_nars_deduction(body: &str, format: ResponseFormat) -> Vec<u8> {
    nars_binary_op(body, |a, b| a.deduction(&b), format)
}
fn handle_nars_induction(body: &str, format: ResponseFormat) -> Vec<u8> {
    nars_binary_op(body, |a, b| a.induction(&b), format)
}
fn handle_nars_abduction(body: &str, format: ResponseFormat) -> Vec<u8> {
    nars_binary_op(body, |a, b| a.abduction(&b), format)
}
fn handle_nars_revision(body: &str, format: ResponseFormat) -> Vec<u8> {
    nars_binary_op(body, |a, b| a.revision(&b), format)
}

fn nars_binary_op(
    body: &str,
    op: impl Fn(TruthValue, TruthValue) -> TruthValue,
    format: ResponseFormat,
) -> Vec<u8> {
    let f1 = extract_json_f32(body, "f1").unwrap_or(0.9);
    let c1 = extract_json_f32(body, "c1").unwrap_or(0.9);
    let f2 = extract_json_f32(body, "f2").unwrap_or(0.9);
    let c2 = extract_json_f32(body, "c2").unwrap_or(0.9);

    let a = TruthValue::new(f1, c1);
    let b = TruthValue::new(f2, c2);
    let result = op(a, b);

    match format {
        ResponseFormat::Arrow => {
            let schema = nars_result_schema();
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Float32Array::from(vec![result.frequency])) as ArrayRef,
                    Arc::new(Float32Array::from(vec![result.confidence])) as ArrayRef,
                    Arc::new(Float32Array::from(vec![result.expectation()])) as ArrayRef,
                ],
            )
            .unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => {
            let json = format!(
                r#"{{"frequency":{:.6},"confidence":{:.6},"expectation":{:.6}}}"#,
                result.frequency,
                result.confidence,
                result.expectation()
            );
            http_json(200, &json)
        }
    }
}

// SQL handler
fn handle_sql(body: &str, _state: &SharedState, format: ResponseFormat) -> Vec<u8> {
    let query = extract_json_str(body, "query")
        .or_else(|| Some(body.to_string()))
        .unwrap_or_default();

    match format {
        ResponseFormat::Arrow => {
            let schema = Arc::new(Schema::new(vec![
                Field::new("status", DataType::Utf8, false),
                Field::new("query", DataType::Utf8, false),
                Field::new("note", DataType::Utf8, false),
            ]));
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec!["acknowledged"])) as ArrayRef,
                    Arc::new(StringArray::from(vec![query.as_str()])) as ArrayRef,
                    Arc::new(StringArray::from(vec![
                        "Full DataFusion SQL execution available via Flight API",
                    ])) as ArrayRef,
                ],
            )
            .unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => {
            let json = format!(
                r#"{{"status":"acknowledged","query":"{}","note":"Full DataFusion SQL execution available via Flight API"}}"#,
                query
                    .replace('"', "'")
                    .chars()
                    .take(200)
                    .collect::<String>()
            );
            http_json(200, &json)
        }
    }
}

// Cypher handler
fn handle_cypher(body: &str, format: ResponseFormat) -> Vec<u8> {
    let query = extract_json_str(body, "query").unwrap_or_default();

    match ladybug::query::cypher_to_sql(&query) {
        Ok(sql) => match format {
            ResponseFormat::Arrow => {
                let schema = Arc::new(Schema::new(vec![
                    Field::new("cypher", DataType::Utf8, false),
                    Field::new("transpiled_sql", DataType::Utf8, false),
                    Field::new("status", DataType::Utf8, false),
                ]));
                let batch = RecordBatch::try_new(
                    schema,
                    vec![
                        Arc::new(StringArray::from(vec![query.as_str()])) as ArrayRef,
                        Arc::new(StringArray::from(vec![sql.as_str()])) as ArrayRef,
                        Arc::new(StringArray::from(vec!["transpiled"])) as ArrayRef,
                    ],
                )
                .unwrap();
                http_arrow(200, &batch)
            }
            ResponseFormat::Json => {
                let json = format!(
                    r#"{{"cypher":"{}","transpiled_sql":"{}","status":"transpiled"}}"#,
                    query.replace('"', "'"),
                    sql.replace('"', "'")
                );
                http_json(200, &json)
            }
        },
        Err(e) => http_error(400, "cypher_parse_error", &e.to_string(), format),
    }
}

// CogRedis handler - uses full CogRedis command set including DN.*, CAM.*, DAG.* etc.
fn handle_redis_command(body: &str, state: &SharedState) -> Vec<u8> {
    let cmd = body.trim();
    if cmd.is_empty() {
        return http_json(400, r#"{"error":"empty_command"}"#);
    }

    // Execute through full CogRedis interface (supports DN.*, CAM.*, DAG.*, etc.)
    let result = state.write().unwrap().cog_redis.execute_command(cmd);
    let response = cog_redis_result_to_wire(&result);

    // Wrap Redis response in HTTP
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        response.len(), response
    ).into_bytes()
}

/// Convert CogRedis result to Redis wire protocol (RESP)
fn cog_redis_result_to_wire(result: &RedisResult) -> String {
    match result {
        RedisResult::Ok => "+OK\r\n".to_string(),
        RedisResult::String(s) => format!("+{}\r\n", s),
        RedisResult::Integer(i) => format!(":{}\r\n", i),
        RedisResult::Bulk(bytes) => {
            // Return as hex-encoded bulk string
            let hex = hex::encode(bytes);
            format!("${}\r\n{}\r\n", hex.len(), hex)
        }
        RedisResult::Array(items) => {
            let mut resp = format!("*{}\r\n", items.len());
            for item in items {
                resp.push_str(&cog_redis_result_to_wire(item));
            }
            resp
        }
        RedisResult::Nil => "$-1\r\n".to_string(),
        RedisResult::Error(e) => format!("-ERR {}\r\n", e),
    }
}

// LanceDB-compatible handlers
fn handle_lance_create_table(body: &str, format: ResponseFormat) -> Vec<u8> {
    let name = extract_json_str(body, "name").unwrap_or_else(|| "default".to_string());

    match format {
        ResponseFormat::Arrow => {
            let schema = Arc::new(Schema::new(vec![
                Field::new("table", DataType::Utf8, false),
                Field::new("status", DataType::Utf8, false),
            ]));
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec![name.as_str()])) as ArrayRef,
                    Arc::new(StringArray::from(vec!["created"])) as ArrayRef,
                ],
            )
            .unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => http_json(
            200,
            &format!(
                r#"{{"table":"{}","status":"created","note":"In-memory table backed by indexed fingerprints"}}"#,
                name
            ),
        ),
    }
}

fn handle_lance_add(body: &str, state: &SharedState, format: ResponseFormat) -> Vec<u8> {
    let id = extract_json_str(body, "id").unwrap_or_else(uuid_v4);
    let text = extract_json_str(body, "text").unwrap_or_default();

    let fp = Fingerprint::from_content(&text);
    let mut meta = HashMap::new();
    meta.insert("text".to_string(), text);

    let mut db = state.write().unwrap();
    let idx = db.fingerprints.len();
    db.fingerprints.push((id.clone(), fp, meta));

    match format {
        ResponseFormat::Arrow => {
            let schema = index_result_schema();
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(BooleanArray::from(vec![true])) as ArrayRef,
                    Arc::new(StringArray::from(vec![id.as_str()])) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![idx as u32])) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![db.fingerprints.len() as u32])) as ArrayRef,
                ],
            )
            .unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => http_json(200, &format!(r#"{{"id":"{}","index":{}}}"#, id, idx)),
    }
}

fn handle_lance_search(body: &str, state: &SharedState, format: ResponseFormat) -> Vec<u8> {
    let query_text = extract_json_str(body, "query").unwrap_or_default();
    let limit = extract_json_usize(body, "limit").unwrap_or(10);

    let query = Fingerprint::from_content(&query_text);
    let db = state.read().unwrap();

    let mut scored: Vec<(usize, u32, f32)> = db
        .fingerprints
        .iter()
        .enumerate()
        .map(|(i, (_, fp, _))| {
            let dist = hamming_distance(&query, fp);
            let sim = 1.0 - (dist as f32 / FINGERPRINT_BITS as f32);
            (i, dist, sim)
        })
        .collect();

    scored.sort_by_key(|&(_, d, _)| d);
    scored.truncate(limit);

    match format {
        ResponseFormat::Arrow => {
            let schema = search_result_schema();
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(UInt32Array::from(
                        scored.iter().map(|(i, _, _)| *i as u32).collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(StringArray::from(
                        scored
                            .iter()
                            .map(|(i, _, _)| db.fingerprints[*i].0.as_str())
                            .collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(UInt32Array::from(
                        scored.iter().map(|(_, d, _)| *d).collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(Float32Array::from(
                        scored.iter().map(|(_, _, s)| *s).collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(StringArray::from(
                        scored
                            .iter()
                            .map(|(i, _, _)| db.fingerprints[*i].2.get("text").map(|s| s.as_str()))
                            .collect::<Vec<_>>(),
                    )) as ArrayRef,
                ],
            )
            .unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => {
            let results: Vec<String> = scored
                .iter()
                .map(|&(idx, dist, sim)| {
                    let (id, _, meta) = &db.fingerprints[idx];
                    let text = meta.get("text").cloned().unwrap_or_default();
                    format!(
                        r#"{{"id":"{}","_distance":{},"_similarity":{:.6},"text":"{}"}}"#,
                        id,
                        dist,
                        sim,
                        text.replace('"', "'")
                    )
                })
                .collect();
            http_json(200, &format!(r#"[{}]"#, results.join(",")))
        }
    }
}

// =============================================================================
// GRAPH TRAVERSAL HANDLERS (BindSpace native)
// =============================================================================

/// Helper: compute popcount of a fingerprint word array
fn fp_popcount(fp: &[u64; FINGERPRINT_WORDS]) -> u32 {
    fp.iter().map(|w| w.count_ones()).sum()
}

/// Helper: get label for an address from BindSpace
fn addr_label(bs: &BindSpace, addr: Addr) -> Option<String> {
    bs.read(addr).and_then(|n| n.label.clone())
}

/// POST /api/v1/graph/traverse
/// BFS traversal from source address through BindSpace edges.
/// Body: {"source": "0x8000", "max_hops": 3, "verb": "0x0700", "limit": 100}
fn handle_graph_traverse(body: &str, state: &SharedState, format: ResponseFormat) -> Vec<u8> {
    let source_raw = extract_json_hex_u16(body, "source").unwrap_or(0x8000);
    let max_hops = extract_json_usize(body, "max_hops").unwrap_or(3) as usize;
    let verb_raw = extract_json_hex_u16(body, "verb");
    let limit = extract_json_usize(body, "limit").unwrap_or(1000);

    let db = state.read().unwrap();
    let bs = db.cog_redis.bind_space();
    let source = Addr(source_raw);

    // BFS traversal
    let mut results: Vec<(u32, u16, Option<String>, u32, Option<u32>)> = Vec::new();
    let mut frontier = vec![source];
    let mut visited = std::collections::HashSet::new();
    visited.insert(source_raw);

    let source_fp = bs.read(source).map(|n| n.fingerprint);

    for hop in 0..max_hops {
        let mut next_frontier = Vec::new();
        for &addr in &frontier {
            let edges: Vec<_> = bs.edges_out(addr).collect();
            for edge in edges {
                if visited.contains(&edge.to.0) {
                    continue;
                }
                visited.insert(edge.to.0);

                // If verb filter set, skip non-matching edges
                if let Some(vf) = verb_raw
                    && edge.verb.0 != vf
                {
                    continue;
                }

                let label = addr_label(bs, edge.to);
                let pc = bs
                    .read(edge.to)
                    .map(|n| fp_popcount(&n.fingerprint))
                    .unwrap_or(0);

                // Distance from source (Hamming) if source has fingerprint
                let dist = source_fp.and_then(|sfp| {
                    bs.read(edge.to).map(|n| {
                        let mut d = 0u32;
                        for i in 0..FINGERPRINT_WORDS {
                            d += (sfp[i] ^ n.fingerprint[i]).count_ones();
                        }
                        d
                    })
                });

                results.push(((hop + 1) as u32, edge.to.0, label, pc, dist));
                next_frontier.push(edge.to);

                if results.len() >= limit {
                    break;
                }
            }
            if results.len() >= limit {
                break;
            }
        }
        frontier = next_frontier;
        if frontier.is_empty() || results.len() >= limit {
            break;
        }
    }

    match format {
        ResponseFormat::Arrow => {
            let schema = graph_traversal_schema();
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(UInt32Array::from(
                        results.iter().map(|r| r.0).collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(UInt32Array::from(
                        results.iter().map(|r| r.1 as u32).collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(StringArray::from(
                        results.iter().map(|r| r.2.as_deref()).collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(UInt32Array::from(
                        results.iter().map(|r| r.3).collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(UInt32Array::from(
                        results.iter().map(|r| r.4.unwrap_or(0)).collect::<Vec<_>>(),
                    )) as ArrayRef,
                ],
            )
            .unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => {
            let items: Vec<String> = results.iter().map(|(hop, addr, label, pc, dist)| {
                format!(
                    r#"{{"hop":{},"address":"0x{:04X}","label":{},"popcount":{},"distance":{}}}"#,
                    hop, addr,
                    label.as_ref().map(|l| format!(r#""{}""#, l)).unwrap_or("null".to_string()),
                    pc,
                    dist.map(|d| d.to_string()).unwrap_or("null".to_string()),
                )
            }).collect();
            http_json(
                200,
                &format!(
                    r#"{{"source":"0x{:04X}","max_hops":{},"count":{},"results":[{}]}}"#,
                    source_raw,
                    max_hops,
                    results.len(),
                    items.join(",")
                ),
            )
        }
    }
}

/// POST /api/v1/graph/edges
/// List edges from BindSpace. Optionally filter by source/target/verb.
/// Body: {"source": "0x8000", "verb": "0x0700", "limit": 100}
fn handle_graph_edges(body: &str, state: &SharedState, format: ResponseFormat) -> Vec<u8> {
    let source_raw = extract_json_hex_u16(body, "source");
    let verb_raw = extract_json_hex_u16(body, "verb");
    let limit = extract_json_usize(body, "limit").unwrap_or(500);

    let db = state.read().unwrap();
    let bs = db.cog_redis.bind_space();

    // Collect edges
    let mut edges: Vec<(
        u16,
        u16,
        u16,
        Option<String>,
        Option<String>,
        Option<String>,
        f32,
    )> = Vec::new();

    if let Some(src) = source_raw {
        // Edges from specific source
        for edge in bs.edges_out(Addr(src)) {
            if let Some(vf) = verb_raw
                && edge.verb.0 != vf
            {
                continue;
            }
            edges.push((
                edge.from.0,
                edge.to.0,
                edge.verb.0,
                addr_label(bs, edge.from),
                addr_label(bs, edge.to),
                addr_label(bs, edge.verb),
                edge.weight,
            ));
            if edges.len() >= limit {
                break;
            }
        }
    } else {
        // Scan all node addresses (0x80..0xFF prefixes)
        for prefix in 0x80u8..=0xFF {
            for slot in 0..=255u8 {
                let addr = Addr::new(prefix, slot);
                for edge in bs.edges_out(addr) {
                    if let Some(vf) = verb_raw
                        && edge.verb.0 != vf
                    {
                        continue;
                    }
                    edges.push((
                        edge.from.0,
                        edge.to.0,
                        edge.verb.0,
                        addr_label(bs, edge.from),
                        addr_label(bs, edge.to),
                        addr_label(bs, edge.verb),
                        edge.weight,
                    ));
                    if edges.len() >= limit {
                        break;
                    }
                }
                if edges.len() >= limit {
                    break;
                }
            }
            if edges.len() >= limit {
                break;
            }
        }
    }

    match format {
        ResponseFormat::Arrow => {
            let schema = graph_edge_schema();
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(UInt32Array::from(
                        edges.iter().map(|e| e.0 as u32).collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(UInt32Array::from(
                        edges.iter().map(|e| e.1 as u32).collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(UInt32Array::from(
                        edges.iter().map(|e| e.2 as u32).collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(StringArray::from(
                        edges.iter().map(|e| e.3.as_deref()).collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(StringArray::from(
                        edges.iter().map(|e| e.4.as_deref()).collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(StringArray::from(
                        edges.iter().map(|e| e.5.as_deref()).collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(Float32Array::from(
                        edges.iter().map(|e| e.6).collect::<Vec<_>>(),
                    )) as ArrayRef,
                ],
            )
            .unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => {
            let items: Vec<String> = edges.iter().map(|(src, tgt, vrb, sl, tl, vl, w)| {
                format!(
                    r#"{{"source":"0x{:04X}","target":"0x{:04X}","verb":"0x{:04X}","source_label":{},"target_label":{},"verb_label":{},"weight":{:.4}}}"#,
                    src, tgt, vrb,
                    sl.as_ref().map(|l| format!(r#""{}""#, l)).unwrap_or("null".to_string()),
                    tl.as_ref().map(|l| format!(r#""{}""#, l)).unwrap_or("null".to_string()),
                    vl.as_ref().map(|l| format!(r#""{}""#, l)).unwrap_or("null".to_string()),
                    w,
                )
            }).collect();
            http_json(
                200,
                &format!(
                    r#"{{"count":{},"edges":[{}]}}"#,
                    edges.len(),
                    items.join(",")
                ),
            )
        }
    }
}

/// POST /api/v1/graph/neighbors
/// Get immediate neighbors (1-hop) of an address with their fingerprint popcounts.
/// Body: {"address": "0x8000", "direction": "out|in|both"}
fn handle_graph_neighbors(body: &str, state: &SharedState, format: ResponseFormat) -> Vec<u8> {
    let addr_raw = extract_json_hex_u16(body, "address").unwrap_or(0x8000);
    let direction = extract_json_str(body, "direction").unwrap_or_else(|| "out".to_string());

    let db = state.read().unwrap();
    let bs = db.cog_redis.bind_space();
    let addr = Addr(addr_raw);

    // Collect neighbors: (address, label, popcount, via_verb, weight)
    let mut neighbors: Vec<(u16, Option<String>, u32, u16, f32)> = Vec::new();
    let mut seen = std::collections::HashSet::new();

    if direction == "out" || direction == "both" {
        for edge in bs.edges_out(addr) {
            if seen.insert(edge.to.0) {
                let pc = bs
                    .read(edge.to)
                    .map(|n| fp_popcount(&n.fingerprint))
                    .unwrap_or(0);
                neighbors.push((
                    edge.to.0,
                    addr_label(bs, edge.to),
                    pc,
                    edge.verb.0,
                    edge.weight,
                ));
            }
        }
    }
    if direction == "in" || direction == "both" {
        for edge in bs.edges_in(addr) {
            if seen.insert(edge.from.0) {
                let pc = bs
                    .read(edge.from)
                    .map(|n| fp_popcount(&n.fingerprint))
                    .unwrap_or(0);
                neighbors.push((
                    edge.from.0,
                    addr_label(bs, edge.from),
                    pc,
                    edge.verb.0,
                    edge.weight,
                ));
            }
        }
    }

    match format {
        ResponseFormat::Arrow => {
            let schema = Arc::new(Schema::new(vec![
                Field::new("address", DataType::UInt32, false),
                Field::new("label", DataType::Utf8, true),
                Field::new("popcount", DataType::UInt32, false),
                Field::new("via_verb", DataType::UInt32, false),
                Field::new("weight", DataType::Float32, false),
            ]));
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(UInt32Array::from(
                        neighbors.iter().map(|n| n.0 as u32).collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(StringArray::from(
                        neighbors.iter().map(|n| n.1.as_deref()).collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(UInt32Array::from(
                        neighbors.iter().map(|n| n.2).collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(UInt32Array::from(
                        neighbors.iter().map(|n| n.3 as u32).collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(Float32Array::from(
                        neighbors.iter().map(|n| n.4).collect::<Vec<_>>(),
                    )) as ArrayRef,
                ],
            )
            .unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => {
            let items: Vec<String> = neighbors.iter().map(|(a, l, pc, v, w)| {
                format!(
                    r#"{{"address":"0x{:04X}","label":{},"popcount":{},"via_verb":"0x{:04X}","weight":{:.4}}}"#,
                    a,
                    l.as_ref().map(|s| format!(r#""{}""#, s)).unwrap_or("null".to_string()),
                    pc, v, w,
                )
            }).collect();
            http_json(
                200,
                &format!(
                    r#"{{"address":"0x{:04X}","direction":"{}","count":{},"neighbors":[{}]}}"#,
                    addr_raw,
                    direction,
                    neighbors.len(),
                    items.join(",")
                ),
            )
        }
    }
}

/// POST /api/v1/graph/hydrate
/// Hydrate sparse 16-bit addresses to full 10K-bit fingerprints.
/// Body: {"addresses": ["0x8000", "0x8001"]} or {"address": "0x8000"}
fn handle_graph_hydrate(body: &str, state: &SharedState, format: ResponseFormat) -> Vec<u8> {
    let db = state.read().unwrap();
    let bs = db.cog_redis.bind_space();

    // Single or batch
    let addrs: Vec<u16> = if let Some(single) = extract_json_hex_u16(body, "address") {
        vec![single]
    } else {
        extract_json_str_array(body, "addresses")
            .iter()
            .filter_map(|s| {
                u16::from_str_radix(s.trim_start_matches("0x").trim_start_matches("0X"), 16).ok()
            })
            .collect()
    };

    if addrs.is_empty() {
        return http_error(400, "missing_field", "need address or addresses", format);
    }

    // Hydrate each address
    let mut results: Vec<(u16, Option<String>, Vec<u8>, u32)> = Vec::new();
    for &raw in &addrs {
        let addr = Addr(raw);
        if let Some(node) = bs.read(addr) {
            let mut fp_bytes = vec![0u8; FINGERPRINT_BYTES];
            for (i, &word) in node.fingerprint.iter().enumerate() {
                let offset = i * 8;
                if offset + 8 <= FINGERPRINT_BYTES {
                    fp_bytes[offset..offset + 8].copy_from_slice(&word.to_le_bytes());
                }
            }
            let pc = fp_popcount(&node.fingerprint);
            results.push((raw, node.label.clone(), fp_bytes, pc));
        } else {
            // Address not occupied - return zero fingerprint
            results.push((raw, None, vec![0u8; FINGERPRINT_BYTES], 0));
        }
    }

    match format {
        ResponseFormat::Arrow => {
            let schema = hydrate_schema();
            let fp_refs: Vec<&[u8]> = results.iter().map(|r| r.2.as_slice()).collect();
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(UInt32Array::from(
                        results.iter().map(|r| r.0 as u32).collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(StringArray::from(
                        results.iter().map(|r| r.1.as_deref()).collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(FixedSizeBinaryArray::try_from_iter(fp_refs.into_iter()).unwrap())
                        as ArrayRef,
                    Arc::new(UInt32Array::from(
                        results.iter().map(|r| r.3).collect::<Vec<_>>(),
                    )) as ArrayRef,
                ],
            )
            .unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => {
            let items: Vec<String> = results
                .iter()
                .map(|(addr, label, fp_bytes, pc)| {
                    let b64 = base64_encode(fp_bytes);
                    format!(
                        r#"{{"address":"0x{:04X}","label":{},"fingerprint":"{}","popcount":{}}}"#,
                        addr,
                        label
                            .as_ref()
                            .map(|l| format!(r#""{}""#, l))
                            .unwrap_or("null".to_string()),
                        b64,
                        pc,
                    )
                })
                .collect();
            http_json(
                200,
                &format!(
                    r#"{{"count":{},"results":[{}]}}"#,
                    results.len(),
                    items.join(",")
                ),
            )
        }
    }
}

/// POST /api/v1/graph/search
/// Search BindSpace using popcount pre-filtering + Hamming distance.
/// Body: {"query": "<base64_fp>", "max_distance": 2000, "top_k": 10}
fn handle_graph_search(body: &str, state: &SharedState, format: ResponseFormat) -> Vec<u8> {
    let query_str = extract_json_str(body, "query").unwrap_or_default();
    let max_distance = extract_json_usize(body, "max_distance").map(|d| d as u32);
    let top_k = extract_json_usize(body, "top_k").unwrap_or(10);

    let query = resolve_fingerprint(&query_str);
    let query_words: [u64; FINGERPRINT_WORDS] = {
        let mut w = [0u64; FINGERPRINT_WORDS];
        let bytes = query.as_bytes();
        for i in 0..FINGERPRINT_WORDS {
            let offset = i * 8;
            if offset + 8 <= bytes.len() {
                w[i] = u64::from_le_bytes([
                    bytes[offset],
                    bytes[offset + 1],
                    bytes[offset + 2],
                    bytes[offset + 3],
                    bytes[offset + 4],
                    bytes[offset + 5],
                    bytes[offset + 6],
                    bytes[offset + 7],
                ]);
            }
        }
        w
    };
    let query_pc = fp_popcount(&query_words);

    let db = state.read().unwrap();
    let bs = db.cog_redis.bind_space();

    // 3-stage search: popcount triangle inequality → full Hamming
    let pc_tolerance = max_distance.unwrap_or(2000);
    let mut scored: Vec<(u16, u32, Option<String>)> = Vec::new();

    // Scan all node addresses
    for prefix in 0x80u8..=0xFF {
        for slot in 0..=255u8 {
            let addr = Addr::new(prefix, slot);
            if let Some(node) = bs.read(addr) {
                let node_pc = fp_popcount(&node.fingerprint);

                // Stage 1: popcount triangle inequality pre-filter
                let pc_diff = (query_pc as i64 - node_pc as i64).unsigned_abs() as u32;
                if pc_diff > pc_tolerance {
                    continue;
                }

                // Stage 2: full Hamming distance
                let mut dist = 0u32;
                for i in 0..FINGERPRINT_WORDS {
                    dist += (query_words[i] ^ node.fingerprint[i]).count_ones();
                }

                if let Some(max_d) = max_distance
                    && dist > max_d
                {
                    continue;
                }

                scored.push((addr.0, dist, node.label.clone()));
            }
        }
    }

    // Sort by distance and take top_k
    scored.sort_by_key(|&(_, d, _)| d);
    scored.truncate(top_k);

    match format {
        ResponseFormat::Arrow => {
            let schema = Arc::new(Schema::new(vec![
                Field::new("address", DataType::UInt32, false),
                Field::new("distance", DataType::UInt32, false),
                Field::new("similarity", DataType::Float32, false),
                Field::new("label", DataType::Utf8, true),
            ]));
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(UInt32Array::from(
                        scored.iter().map(|r| r.0 as u32).collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(UInt32Array::from(
                        scored.iter().map(|r| r.1).collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(Float32Array::from(
                        scored
                            .iter()
                            .map(|r| 1.0 - (r.1 as f32 / FINGERPRINT_BITS as f32))
                            .collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(StringArray::from(
                        scored.iter().map(|r| r.2.as_deref()).collect::<Vec<_>>(),
                    )) as ArrayRef,
                ],
            )
            .unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => {
            let items: Vec<String> = scored
                .iter()
                .map(|(addr, dist, label)| {
                    let sim = 1.0 - (*dist as f32 / FINGERPRINT_BITS as f32);
                    format!(
                        r#"{{"address":"0x{:04X}","distance":{},"similarity":{:.6},"label":{}}}"#,
                        addr,
                        dist,
                        sim,
                        label
                            .as_ref()
                            .map(|l| format!(r#""{}""#, l))
                            .unwrap_or("null".to_string()),
                    )
                })
                .collect();
            http_json(
                200,
                &format!(
                    r#"{{"query_popcount":{},"count":{},"results":[{}]}}"#,
                    query_pc,
                    scored.len(),
                    items.join(",")
                ),
            )
        }
    }
}

// =============================================================================
// UDP BITPACKED HAMMING HANDLER
// =============================================================================

/// Handle a single UDP datagram. Returns response bytes to send back.
fn handle_udp_packet(buf: &[u8], len: usize, state: &SharedState) -> Option<Vec<u8>> {
    if len < UDP_HEADER_SIZE {
        return None;
    }

    let header = UdpHeader::parse(buf)?;

    match header.op {
        UdpOp::Ping => {
            // PONG: echo back with op changed
            let mut resp = vec![0u8; UDP_HEADER_SIZE];
            let pong = UdpHeader {
                op: UdpOp::Pong,
                sequence: header.sequence,
                ..UdpHeader {
                    op: UdpOp::Pong,
                    flags: 0,
                    sequence: header.sequence,
                    source_addr: 0,
                    query_addr: 0,
                    max_hops: 0,
                    verb_filter: 0,
                    top_k: 0,
                    max_distance: 0,
                    popcount_hint: 0,
                    result_count: 0,
                    payload_len: 0,
                    result_addrs: Vec::new(),
                }
            };
            pong.encode(&mut resp);
            Some(resp)
        }

        UdpOp::Search => {
            // Extract fingerprint from payload (after header)
            if len < UDP_HEADER_SIZE + UDP_FP_RAW {
                return None;
            }

            let fp_payload = &buf[UDP_HEADER_SIZE..UDP_HEADER_SIZE + UDP_FP_RAW];
            let mut query_words = [0u64; FINGERPRINT_WORDS];
            for i in 0..FINGERPRINT_WORDS {
                let off = i * 8;
                query_words[i] = u64::from_le_bytes([
                    fp_payload[off],
                    fp_payload[off + 1],
                    fp_payload[off + 2],
                    fp_payload[off + 3],
                    fp_payload[off + 4],
                    fp_payload[off + 5],
                    fp_payload[off + 6],
                    fp_payload[off + 7],
                ]);
            }

            let query_pc = fp_popcount(&query_words);
            let max_dist = if header.max_distance > 0 {
                header.max_distance
            } else {
                2000
            };
            let top_k = if header.top_k > 0 {
                header.top_k as usize
            } else {
                10
            };

            let db = state.read().unwrap();
            let bs = db.cog_redis.bind_space();

            // Search with popcount pre-filter
            let mut scored: Vec<(u16, u32)> = Vec::new();
            for prefix in 0x80u8..=0xFF {
                for slot in 0..=255u8 {
                    let addr = Addr::new(prefix, slot);
                    if let Some(node) = bs.read(addr) {
                        let node_pc = fp_popcount(&node.fingerprint);
                        let pc_diff = (query_pc as i64 - node_pc as i64).unsigned_abs() as u32;
                        if pc_diff > max_dist {
                            continue;
                        }

                        let mut dist = 0u32;
                        for i in 0..FINGERPRINT_WORDS {
                            dist += (query_words[i] ^ node.fingerprint[i]).count_ones();
                        }
                        if dist <= max_dist {
                            scored.push((addr.0, dist));
                        }
                    }
                }
            }
            scored.sort_by_key(|&(_, d)| d);
            scored.truncate(top_k.min(UDP_MAX_RESULT_ADDRS));

            // Build result datagram
            let mut resp = vec![0u8; UDP_HEADER_SIZE];
            let result_addrs: Vec<u16> = scored.iter().map(|&(a, _)| a).collect();
            let result_header = UdpHeader {
                op: UdpOp::Result,
                flags: 0,
                sequence: header.sequence,
                source_addr: 0,
                query_addr: 0,
                max_hops: 0,
                verb_filter: 0,
                top_k: top_k as u8,
                max_distance: max_dist,
                popcount_hint: query_pc,
                result_count: result_addrs.len() as u16,
                payload_len: 0,
                result_addrs,
            };
            result_header.encode(&mut resp);
            Some(resp)
        }

        UdpOp::Traverse => {
            let source = Addr(header.source_addr);
            let max_hops = if header.max_hops > 0 {
                header.max_hops as usize
            } else {
                3
            };

            let db = state.read().unwrap();
            let bs = db.cog_redis.bind_space();

            let verb_filter = if header.verb_filter != 0 {
                Some(header.verb_filter)
            } else {
                None
            };

            // BFS
            let mut found: Vec<u16> = Vec::new();
            let mut frontier = vec![source];
            let mut visited = std::collections::HashSet::new();
            visited.insert(header.source_addr);

            for _hop in 0..max_hops {
                let mut next = Vec::new();
                for &addr in &frontier {
                    for edge in bs.edges_out(addr) {
                        if visited.contains(&edge.to.0) {
                            continue;
                        }
                        visited.insert(edge.to.0);
                        if let Some(vf) = verb_filter
                            && edge.verb.0 != vf
                        {
                            continue;
                        }
                        found.push(edge.to.0);
                        next.push(edge.to);
                        if found.len() >= UDP_MAX_RESULT_ADDRS {
                            break;
                        }
                    }
                    if found.len() >= UDP_MAX_RESULT_ADDRS {
                        break;
                    }
                }
                frontier = next;
                if frontier.is_empty() || found.len() >= UDP_MAX_RESULT_ADDRS {
                    break;
                }
            }

            let mut resp = vec![0u8; UDP_HEADER_SIZE];
            let result_header = UdpHeader {
                op: UdpOp::Result,
                flags: 0,
                sequence: header.sequence,
                source_addr: header.source_addr,
                query_addr: 0,
                max_hops: max_hops as u8,
                verb_filter: verb_filter.unwrap_or(0),
                top_k: 0,
                max_distance: 0,
                popcount_hint: 0,
                result_count: found.len() as u16,
                payload_len: 0,
                result_addrs: found,
            };
            result_header.encode(&mut resp);
            Some(resp)
        }

        UdpOp::Hydrate => {
            // Hydrate: send back full fingerprint for the query_addr
            let addr = Addr(header.query_addr);

            let db = state.read().unwrap();
            let bs = db.cog_redis.bind_space();

            let mut resp = vec![0u8; UDP_MAX_DATAGRAM];

            if let Some(node) = bs.read(addr) {
                let pc = fp_popcount(&node.fingerprint);

                let resp_header = UdpHeader {
                    op: UdpOp::Result,
                    flags: UDP_FLAG_SIMD_ALIGNED | UDP_FLAG_HAS_FINGERPRINT | UDP_FLAG_HAS_POPCOUNT,
                    sequence: header.sequence,
                    source_addr: 0,
                    query_addr: header.query_addr,
                    max_hops: 0,
                    verb_filter: 0,
                    top_k: 0,
                    max_distance: 0,
                    popcount_hint: pc,
                    result_count: 1,
                    payload_len: UDP_FP_PADDED as u16,
                    result_addrs: vec![header.query_addr],
                };
                resp_header.encode(&mut resp);

                // Write fingerprint payload (2048 bytes, no padding needed at 16K)
                for (i, &word) in node.fingerprint.iter().enumerate() {
                    let off = UDP_HEADER_SIZE + i * 8;
                    resp[off..off + 8].copy_from_slice(&word.to_le_bytes());
                }

                Some(resp)
            } else {
                // Address not occupied
                let resp_header = UdpHeader {
                    op: UdpOp::Result,
                    flags: 0,
                    sequence: header.sequence,
                    source_addr: 0,
                    query_addr: header.query_addr,
                    max_hops: 0,
                    verb_filter: 0,
                    top_k: 0,
                    max_distance: 0,
                    popcount_hint: 0,
                    result_count: 0,
                    payload_len: 0,
                    result_addrs: Vec::new(),
                };
                resp_header.encode(&mut resp);
                resp.truncate(UDP_HEADER_SIZE);
                Some(resp)
            }
        }

        UdpOp::Edges => {
            let source = Addr(header.source_addr);

            let db = state.read().unwrap();
            let bs = db.cog_redis.bind_space();

            let mut targets: Vec<u16> = Vec::new();
            for edge in bs.edges_out(source) {
                if header.verb_filter != 0 && edge.verb.0 != header.verb_filter {
                    continue;
                }
                targets.push(edge.to.0);
                if targets.len() >= UDP_MAX_RESULT_ADDRS {
                    break;
                }
            }

            let mut resp = vec![0u8; UDP_HEADER_SIZE];
            let result_header = UdpHeader {
                op: UdpOp::Result,
                flags: 0,
                sequence: header.sequence,
                source_addr: header.source_addr,
                query_addr: 0,
                max_hops: 0,
                verb_filter: header.verb_filter,
                top_k: 0,
                max_distance: 0,
                popcount_hint: 0,
                result_count: targets.len() as u16,
                payload_len: 0,
                result_addrs: targets,
            };
            result_header.encode(&mut resp);
            Some(resp)
        }

        _ => None, // Pong/Result are client-side, not server-handled
    }
}

/// Spawn UDP bitpacked Hamming listener on a separate thread.
fn spawn_udp_listener(host: &str, port: u16, state: SharedState) {
    let addr: SocketAddr = format!("{}:{}", host, port)
        .parse()
        .expect("Invalid UDP address");

    std::thread::spawn(move || {
        let socket = match UdpSocket::bind(addr) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("UDP bind failed on {}: {}", addr, e);
                return;
            }
        };

        println!("UDP bitpacked Hamming listener on udp://{}", addr);
        println!(
            "  Header: {} bytes | Payload: {} bytes (SIMD-padded) | Max datagram: {} bytes",
            UDP_HEADER_SIZE, UDP_FP_PADDED, UDP_MAX_DATAGRAM
        );

        let mut buf = vec![0u8; UDP_MAX_DATAGRAM];
        loop {
            match socket.recv_from(&mut buf) {
                Ok((len, src)) => {
                    if let Some(resp) = handle_udp_packet(&buf, len, &state) {
                        let _ = socket.send_to(&resp, src);
                    }
                }
                Err(e) => {
                    eprintln!("UDP recv error: {}", e);
                }
            }
        }
    });
}

// =============================================================================
// HTTP RESPONSE UTILITIES
// =============================================================================

/// Create HTTP response with Arrow IPC body
fn http_arrow(status: u16, batch: &RecordBatch) -> Vec<u8> {
    let status_text = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "Unknown",
    };

    let body = encode_to_ipc(batch).unwrap_or_else(|_| Vec::new());

    let mut response = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type, Accept\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        status, status_text, ARROW_MIME, body.len()
    ).into_bytes();

    response.extend(body);
    response
}

/// Create error response in Arrow IPC format
fn http_arrow_error(status: u16, message: &str) -> Vec<u8> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("error", DataType::Boolean, false),
        Field::new("message", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(BooleanArray::from(vec![true])) as ArrayRef,
            Arc::new(StringArray::from(vec![message])) as ArrayRef,
        ],
    )
    .unwrap();
    http_arrow(status, &batch)
}

/// Create HTTP response with JSON body (legacy fallback)
fn http_json(status: u16, body: &str) -> Vec<u8> {
    let status_text = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "Unknown",
    };

    format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type, Accept\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        status, status_text, JSON_MIME, body.len(), body
    ).into_bytes()
}

/// Create error response in appropriate format
fn http_error(status: u16, error_type: &str, message: &str, format: ResponseFormat) -> Vec<u8> {
    match format {
        ResponseFormat::Arrow => http_arrow_error(status, message),
        ResponseFormat::Json => http_json(
            status,
            &format!(r#"{{"error":"{}","message":"{}"}}"#, error_type, message),
        ),
    }
}

// =============================================================================
// JSON PARSING UTILITIES (for input parsing only)
// =============================================================================

fn resolve_fingerprint(s: &str) -> Fingerprint {
    if let Ok(bytes) = base64_decode(s)
        && let Ok(fp) = Fingerprint::from_bytes(&bytes)
    {
        return fp;
    }
    Fingerprint::from_content(s)
}

fn extract_json_str(json: &str, key: &str) -> Option<String> {
    let pattern = format!(r#""{}":"#, key);
    let start = json.find(&pattern)?;
    let rest = &json[start + pattern.len()..];

    if let Some(inner) = rest.strip_prefix('"') {
        let end = inner.find('"')?;
        Some(inner[..end].to_string())
    } else {
        None
    }
}

fn extract_json_usize(json: &str, key: &str) -> Option<usize> {
    let pattern = format!(r#""{}":"#, key);
    let start = json.find(&pattern)?;
    let rest = &json[start + pattern.len()..];
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

fn extract_json_f32(json: &str, key: &str) -> Option<f32> {
    let pattern = format!(r#""{}":"#, key);
    let start = json.find(&pattern)?;
    let rest = &json[start + pattern.len()..];
    let end = rest
        .find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

fn extract_json_str_array(json: &str, key: &str) -> Vec<String> {
    let pattern = format!(r#""{}":["#, key);
    let start = match json.find(&pattern) {
        Some(s) => s + pattern.len(),
        None => return Vec::new(),
    };
    let rest = &json[start..];
    let end = match rest.find(']') {
        Some(e) => e,
        None => return Vec::new(),
    };
    let inner = &rest[..end];

    inner
        .split(',')
        .filter_map(|s| {
            let s = s.trim().trim_matches('"');
            if s.is_empty() {
                None
            } else {
                Some(s.to_string())
            }
        })
        .collect()
}

fn extract_json_object(json: &str, key: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    if let Some(start) = json.find(&format!(r#""{}":"#, key)) {
        let rest = &json[start..];
        if let Some(obj_start) = rest.find('{')
            && let Some(obj_end) = rest[obj_start..].find('}')
        {
            let inner = &rest[obj_start + 1..obj_start + obj_end];
            for part in inner.split(',') {
                if let Some((k, v)) = part.split_once(':') {
                    let k = k.trim().trim_matches('"');
                    let v = v.trim().trim_matches('"');
                    if !k.is_empty() {
                        map.insert(k.to_string(), v.to_string());
                    }
                }
            }
        }
    }
    map
}

fn extract_json_hex_u16(json: &str, key: &str) -> Option<u16> {
    let s = extract_json_str(json, key)?;
    let s = s.trim_start_matches("0x").trim_start_matches("0X");
    u16::from_str_radix(s, 16).ok()
}

fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    format!(
        "{:08x}-{:04x}-4{:03x}-{:04x}-{:012x}",
        t.as_secs() as u32,
        (t.subsec_nanos() >> 16) & 0xFFFF,
        t.subsec_nanos() & 0xFFF,
        0x8000 | (t.as_nanos() as u16 & 0x3FFF),
        t.as_nanos() as u64 & 0xFFFFFFFFFFFF
    )
}

fn base64_encode(bytes: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn base64_decode(s: &str) -> std::result::Result<Vec<u8>, String> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD
        .decode(s.trim())
        .map_err(|e| e.to_string())
}

// =============================================================================
// MAIN
// =============================================================================

fn main() {
    let config = ServerConfig::from_env();

    let udp_port = env::var("LADYBUG_UDP_PORT")
        .ok()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(config.port + 1);

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║              LadybugDB v{:<38}║", VERSION);
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!(
        "║  Environment: {:>45}  ║",
        format!("{:?}", config.environment)
    );
    println!(
        "║  HTTP:        {:>45}  ║",
        format!("{}:{}", config.host, config.port)
    );
    println!(
        "║  UDP:         {:>45}  ║",
        format!("{}:{}", config.host, udp_port)
    );
    println!("║  Data dir:    {:>45}  ║", config.data_dir);
    println!("║  SIMD:        {:>45}  ║", simd::simd_level());
    println!("║  Workers:     {:>45}  ║", config.workers);
    println!("║  FP bits:     {:>45}  ║", FINGERPRINT_BITS);
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║  Protocols:                                                     ║");
    println!("║    HTTP  → Arrow IPC (default) + JSON (Accept header)          ║");
    println!("║    gRPC  → Flight Arrow (port 50051 via flight_server binary)  ║");
    println!(
        "║    UDP   → Bitpacked Hamming ({}B hdr + {}B fp = {}B)     ║",
        UDP_HEADER_SIZE, UDP_FP_PADDED, UDP_MAX_DATAGRAM
    );
    println!("╚═══════════════════════════════════════════════════════════════╝");

    let addr: SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .expect("Invalid address");

    let state: SharedState = Arc::new(RwLock::new(DbState::new(&config)));

    // Spawn UDP bitpacked Hamming listener
    spawn_udp_listener(&config.host, udp_port, Arc::clone(&state));

    let listener = TcpListener::bind(addr).unwrap_or_else(|e| {
        eprintln!("Failed to bind {}: {}", addr, e);
        std::process::exit(1);
    });

    println!("Listening on http://{}", addr);
    println!("UDP Hamming on udp://{}:{}", config.host, udp_port);

    // Accept connections
    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                let state = Arc::clone(&state);
                std::thread::spawn(move || {
                    handle_connection(&mut stream, &state);
                });
            }
            Err(e) => eprintln!("Connection error: {}", e),
        }
    }
}
