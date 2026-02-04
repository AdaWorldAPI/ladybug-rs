//! LadybugDB HTTP Server
//!
//! Multi-interface cognitive database server exposing:
//! - REST API with Arrow IPC (default) or JSON (fallback) on /api/*
//! - Redis-compatible text protocol on /redis/*
//! - SQL endpoint on /sql
//! - Cypher endpoint on /cypher
//! - Health/readiness on /health, /ready
//!
//! # Content Negotiation
//!
//! All API endpoints return Arrow IPC by default (`application/vnd.apache.arrow.stream`).
//! JSON is only returned when explicitly requested via `Accept: application/json` header.
//!
//! # Environment Detection
//!
//! - Railway: detects `RAILWAY_*` env vars → binds 0.0.0.0:8080
//! - Claude Code: detects `CLAUDE_*` env vars → binds 127.0.0.1:5432
//! - Custom: set `LADYBUG_HOST` and `LADYBUG_PORT`
//! - Default: 127.0.0.1:8080

use std::collections::HashMap;
use std::env;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream, SocketAddr};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use arrow_array::{
    ArrayRef, FixedSizeBinaryArray, Float32Array, RecordBatch,
    StringArray, UInt32Array, BooleanArray,
};
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use ladybug::core::Fingerprint;
use ladybug::core::simd::{self, hamming_distance};
use ladybug::nars::TruthValue;
use ladybug::storage::service::{CognitiveService, ServiceConfig, CpuFeatures};
use ladybug::{FINGERPRINT_BITS, FINGERPRINT_BYTES, VERSION};

// =============================================================================
// CONSTANTS
// =============================================================================

/// Arrow IPC MIME type
const ARROW_MIME: &str = "application/vnd.apache.arrow.stream";

/// JSON MIME type (legacy fallback only)
const JSON_MIME: &str = "application/json";

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

        let data_dir = env::var("LADYBUG_DATA_DIR")
            .unwrap_or_else(|_| "./data".to_string());

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
    if env::var("CLAUDE_CODE").is_ok()
        || env::var("CLAUDE_SESSION_ID").is_ok()
    {
        return Environment::ClaudeCode;
    }

    // Docker detection
    if std::path::Path::new("/.dockerenv").exists()
        || env::var("DOCKER_CONTAINER").is_ok()
    {
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
        Field::new("fingerprint", DataType::FixedSizeBinary(FINGERPRINT_BYTES as i32), false),
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
    Arc::new(Schema::new(vec![
        Field::new("count", DataType::UInt32, false),
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
    /// Key-value store (CogRedis surface)
    kv: HashMap<String, String>,
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

        let service = CognitiveService::new(svc_config)
            .expect("Failed to create CognitiveService");

        Self {
            fingerprints: Vec::new(),
            kv: HashMap::new(),
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
    Arrow,  // Default - Arrow IPC
    Json,   // Legacy fallback only
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
    let parts: Vec<&str> = request_line.trim().split_whitespace().collect();
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

fn route(method: &str, path: &str, body: &str, state: &SharedState, format: ResponseFormat) -> Vec<u8> {
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
            ).unwrap();
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
            let schema = Arc::new(Schema::new(vec![
                Field::new("status", DataType::Utf8, false),
            ]));
            let batch = RecordBatch::try_new(
                schema,
                vec![Arc::new(StringArray::from(vec!["ready"])) as ArrayRef],
            ).unwrap();
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
            ).unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => {
            let json = format!(
                r#"{{"version":"{}","fingerprint_bits":{},"fingerprint_bytes":{},"simd":"{}","cpu":{{"avx512":{},"avx2":{},"cores":{}}},"indexed_count":{}}}"#,
                VERSION, FINGERPRINT_BITS, FINGERPRINT_BYTES,
                simd::simd_level(),
                db.cpu.has_avx512f, db.cpu.has_avx2, db.cpu.physical_cores,
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
            ).unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => {
            let json = format!(
                r#"{{"level":"{}","avx512f":{},"avx512vpopcntdq":{},"avx2":{},"sse42":{},"physical_cores":{},"optimal_batch_size":{}}}"#,
                simd::simd_level(), cpu.has_avx512f, cpu.has_avx512vpopcntdq,
                cpu.has_avx2, cpu.has_sse42, cpu.physical_cores, cpu.optimal_batch_size()
            );
            http_json(200, &json)
        }
    }
}

fn handle_fingerprint_create(body: &str, format: ResponseFormat) -> Vec<u8> {
    // Parse JSON input (input is always JSON for backwards compat)
    let fp = if let Some(content) = extract_json_str(body, "text")
        .or_else(|| extract_json_str(body, "content"))
    {
        Fingerprint::from_content(&content)
    } else if let Some(b64) = extract_json_str(body, "bytes") {
        match base64_decode(&b64) {
            Ok(bytes) => match Fingerprint::from_bytes(&bytes) {
                Ok(fp) => fp,
                Err(e) => return http_error(400, "invalid_fingerprint", &e, format),
            }
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
                    Arc::new(FixedSizeBinaryArray::try_from_iter(
                        std::iter::once(fp.as_bytes())
                    ).unwrap()) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![fp.popcount()])) as ArrayRef,
                    Arc::new(Float32Array::from(vec![fp.density()])) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![FINGERPRINT_BITS as u32])) as ArrayRef,
                ],
            ).unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => {
            let b64 = base64_encode(fp.as_bytes());
            let json = format!(
                r#"{{"fingerprint":"{}","popcount":{},"density":{:.4},"bits":{}}}"#,
                b64, fp.popcount(), fp.density(), FINGERPRINT_BITS
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

    let fps: Vec<Fingerprint> = contents.iter().map(|c| Fingerprint::from_content(c)).collect();

    match format {
        ResponseFormat::Arrow => {
            let schema = fingerprint_schema();
            let fp_bytes: Vec<&[u8]> = fps.iter().map(|fp| fp.as_bytes()).collect();
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(FixedSizeBinaryArray::try_from_iter(fp_bytes.into_iter()).unwrap()) as ArrayRef,
                    Arc::new(UInt32Array::from(fps.iter().map(|fp| fp.popcount()).collect::<Vec<_>>())) as ArrayRef,
                    Arc::new(Float32Array::from(fps.iter().map(|fp| fp.density()).collect::<Vec<_>>())) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![FINGERPRINT_BITS as u32; fps.len()])) as ArrayRef,
                ],
            ).unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => {
            let results: Vec<String> = fps.iter().zip(contents.iter()).map(|(fp, c)| {
                let b64 = base64_encode(fp.as_bytes());
                format!(
                    r#"{{"content":"{}","fingerprint":"{}","popcount":{},"density":{:.4}}}"#,
                    c, b64, fp.popcount(), fp.density()
                )
            }).collect();
            let json = format!(r#"{{"fingerprints":[{}],"count":{}}}"#, results.join(","), results.len());
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
            ).unwrap();
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

fn handle_similarity(body: &str, state: &SharedState, format: ResponseFormat) -> Vec<u8> {
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
                    Arc::new(FixedSizeBinaryArray::try_from_iter(
                        std::iter::once(result.as_bytes())
                    ).unwrap()) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![result.popcount()])) as ArrayRef,
                    Arc::new(Float32Array::from(vec![result.density()])) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![FINGERPRINT_BITS as u32])) as ArrayRef,
                ],
            ).unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => {
            let json = format!(
                r#"{{"result":"{}","popcount":{},"density":{:.4}}}"#,
                base64_encode(result.as_bytes()), result.popcount(), result.density()
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
                    Arc::new(FixedSizeBinaryArray::try_from_iter(
                        std::iter::once(result.as_bytes())
                    ).unwrap()) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![result.popcount()])) as ArrayRef,
                    Arc::new(Float32Array::from(vec![result.density()])) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![FINGERPRINT_BITS as u32])) as ArrayRef,
                ],
            ).unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => {
            let json = format!(
                r#"{{"result":"{}","popcount":{},"density":{:.4},"input_count":{}}}"#,
                base64_encode(result.as_bytes()), result.popcount(), result.density(), fps.len()
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

    let mut scored: Vec<(usize, u32, f32)> = db.fingerprints.iter().enumerate()
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
                    Arc::new(UInt32Array::from(scored.iter().map(|(i, _, _)| *i as u32).collect::<Vec<_>>())) as ArrayRef,
                    Arc::new(StringArray::from(scored.iter().map(|(i, _, _)| db.fingerprints[*i].0.as_str()).collect::<Vec<_>>())) as ArrayRef,
                    Arc::new(UInt32Array::from(scored.iter().map(|(_, d, _)| *d).collect::<Vec<_>>())) as ArrayRef,
                    Arc::new(Float32Array::from(scored.iter().map(|(_, _, s)| *s).collect::<Vec<_>>())) as ArrayRef,
                    Arc::new(StringArray::from(scored.iter().map(|(i, _, _)| {
                        let meta = &db.fingerprints[*i].2;
                        if meta.is_empty() { None } else { Some(format!("{:?}", meta)) }
                    }).collect::<Vec<_>>())) as ArrayRef,
                ],
            ).unwrap();
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

            let json = format!(r#"{{"results":[{}],"count":{},"style":"{}","total_indexed":{}}}"#,
                results.join(","), results.len(), style, db.fingerprints.len());
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
            if results.len() >= limit { break; }
        }
    }

    match format {
        ResponseFormat::Arrow => {
            let schema = search_result_schema();
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(UInt32Array::from(results.iter().map(|(i, _, _)| *i as u32).collect::<Vec<_>>())) as ArrayRef,
                    Arc::new(StringArray::from(results.iter().map(|(i, _, _)| db.fingerprints[*i].0.as_str()).collect::<Vec<_>>())) as ArrayRef,
                    Arc::new(UInt32Array::from(results.iter().map(|(_, d, _)| *d).collect::<Vec<_>>())) as ArrayRef,
                    Arc::new(Float32Array::from(results.iter().map(|(_, _, s)| *s).collect::<Vec<_>>())) as ArrayRef,
                    Arc::new(StringArray::from(vec![None::<&str>; results.len()])) as ArrayRef,
                ],
            ).unwrap();
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

            let json = format!(r#"{{"results":[{}],"count":{},"max_distance":{},"total_indexed":{}}}"#,
                results_json.join(","), results.len(), max_distance, db.fingerprints.len());
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

    let mut scored: Vec<(usize, f32)> = db.fingerprints.iter().enumerate()
        .filter_map(|(i, (_, fp, _))| {
            let sim = 1.0 - (hamming_distance(&query, fp) as f32 / FINGERPRINT_BITS as f32);
            if sim >= threshold { Some((i, sim)) } else { None }
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
                    Arc::new(UInt32Array::from(scored.iter().map(|(i, _)| *i as u32).collect::<Vec<_>>())) as ArrayRef,
                    Arc::new(StringArray::from(scored.iter().map(|(i, _)| db.fingerprints[*i].0.as_str()).collect::<Vec<_>>())) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![0u32; scored.len()])) as ArrayRef, // Distance not computed
                    Arc::new(Float32Array::from(scored.iter().map(|(_, s)| *s).collect::<Vec<_>>())) as ArrayRef,
                    Arc::new(StringArray::from(vec![None::<&str>; scored.len()])) as ArrayRef,
                ],
            ).unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => {
            let results: Vec<String> = scored.iter().map(|&(idx, sim)| {
                let (id, _, _) = &db.fingerprints[idx];
                format!(r#"{{"index":{},"id":"{}","similarity":{:.6}}}"#, idx, id, sim)
            }).collect();

            let json = format!(r#"{{"results":[{}],"count":{},"content":"{}","threshold":{}}}"#,
                results.join(","), results.len(), content, threshold);
            http_json(200, &json)
        }
    }
}

fn handle_index(body: &str, state: &SharedState, format: ResponseFormat) -> Vec<u8> {
    let id = extract_json_str(body, "id").unwrap_or_else(|| uuid_v4());

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
            ).unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => {
            let json = format!(r#"{{"success":true,"id":"{}","index":{},"total":{}}}"#,
                id, idx, db.fingerprints.len());
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
            ).unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => {
            http_json(200, &format!(r#"{{"count":{}}}"#, count))
        }
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
            ).unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => {
            http_json(200, &format!(r#"{{"cleared":true,"was":{}}}"#, was))
        }
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

fn nars_binary_op(body: &str, op: impl Fn(TruthValue, TruthValue) -> TruthValue, format: ResponseFormat) -> Vec<u8> {
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
            ).unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => {
            let json = format!(
                r#"{{"frequency":{:.6},"confidence":{:.6},"expectation":{:.6}}}"#,
                result.frequency, result.confidence, result.expectation()
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
                    Arc::new(StringArray::from(vec!["Full DataFusion SQL execution available via Flight API"])) as ArrayRef,
                ],
            ).unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => {
            let json = format!(
                r#"{{"status":"acknowledged","query":"{}","note":"Full DataFusion SQL execution available via Flight API"}}"#,
                query.replace('"', "'").chars().take(200).collect::<String>()
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
                ).unwrap();
                http_arrow(200, &batch)
            }
            ResponseFormat::Json => {
                let json = format!(
                    r#"{{"cypher":"{}","transpiled_sql":"{}","status":"transpiled"}}"#,
                    query.replace('"', "'"), sql.replace('"', "'")
                );
                http_json(200, &json)
            }
        }
        Err(e) => http_error(400, "cypher_parse_error", &e.to_string(), format),
    }
}

// CogRedis handler - always uses Redis wire protocol (not Arrow or JSON)
fn handle_redis_command(body: &str, state: &SharedState) -> Vec<u8> {
    let parts: Vec<&str> = body.trim().split_whitespace().collect();
    if parts.is_empty() {
        return http_json(400, r#"{"error":"empty_command"}"#);
    }

    let cmd = parts[0].to_uppercase();
    let response = match cmd.as_str() {
        "PING" => "+PONG\r\n".to_string(),
        "SET" if parts.len() >= 3 => {
            let key = parts[1].to_string();
            let val = parts[2..].join(" ");
            state.write().unwrap().kv.insert(key, val);
            "+OK\r\n".to_string()
        }
        "GET" if parts.len() >= 2 => {
            let key = parts[1];
            let db = state.read().unwrap();
            match db.kv.get(key) {
                Some(v) => format!("${}\r\n{}\r\n", v.len(), v),
                None => "$-1\r\n".to_string(),
            }
        }
        "DEL" if parts.len() >= 2 => {
            let key = parts[1];
            let removed = state.write().unwrap().kv.remove(key).is_some();
            format!(":{}\r\n", if removed { 1 } else { 0 })
        }
        "KEYS" => {
            let pattern = if parts.len() >= 2 { parts[1] } else { "*" };
            let db = state.read().unwrap();
            let keys: Vec<&String> = if pattern == "*" {
                db.kv.keys().collect()
            } else {
                db.kv.keys().filter(|k| k.contains(pattern.trim_matches('*'))).collect()
            };
            let mut resp = format!("*{}\r\n", keys.len());
            for k in keys {
                resp.push_str(&format!("${}\r\n{}\r\n", k.len(), k));
            }
            resp
        }
        "INFO" => {
            let db = state.read().unwrap();
            let info = format!(
                "ladybugdb v{}\nsimd:{}\nindexed:{}\nkeys:{}\nuptime:{}s",
                VERSION, simd::simd_level(), db.fingerprints.len(),
                db.kv.len(), db.start_time.elapsed().as_secs()
            );
            format!("${}\r\n{}\r\n", info.len(), info)
        }
        _ => format!("-ERR unknown command '{}'\r\n", cmd),
    };

    // Wrap Redis response in HTTP
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        response.len(), response
    ).into_bytes()
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
            ).unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => {
            http_json(200, &format!(r#"{{"table":"{}","status":"created","note":"In-memory table backed by indexed fingerprints"}}"#, name))
        }
    }
}

fn handle_lance_add(body: &str, state: &SharedState, format: ResponseFormat) -> Vec<u8> {
    let id = extract_json_str(body, "id").unwrap_or_else(|| uuid_v4());
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
            ).unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => {
            http_json(200, &format!(r#"{{"id":"{}","index":{}}}"#, id, idx))
        }
    }
}

fn handle_lance_search(body: &str, state: &SharedState, format: ResponseFormat) -> Vec<u8> {
    let query_text = extract_json_str(body, "query").unwrap_or_default();
    let limit = extract_json_usize(body, "limit").unwrap_or(10);

    let query = Fingerprint::from_content(&query_text);
    let db = state.read().unwrap();

    let mut scored: Vec<(usize, u32, f32)> = db.fingerprints.iter().enumerate()
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
                    Arc::new(UInt32Array::from(scored.iter().map(|(i, _, _)| *i as u32).collect::<Vec<_>>())) as ArrayRef,
                    Arc::new(StringArray::from(scored.iter().map(|(i, _, _)| db.fingerprints[*i].0.as_str()).collect::<Vec<_>>())) as ArrayRef,
                    Arc::new(UInt32Array::from(scored.iter().map(|(_, d, _)| *d).collect::<Vec<_>>())) as ArrayRef,
                    Arc::new(Float32Array::from(scored.iter().map(|(_, _, s)| *s).collect::<Vec<_>>())) as ArrayRef,
                    Arc::new(StringArray::from(scored.iter().map(|(i, _, _)| {
                        db.fingerprints[*i].2.get("text").map(|s| s.as_str())
                    }).collect::<Vec<_>>())) as ArrayRef,
                ],
            ).unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => {
            let results: Vec<String> = scored.iter().map(|&(idx, dist, sim)| {
                let (id, _, meta) = &db.fingerprints[idx];
                let text = meta.get("text").cloned().unwrap_or_default();
                format!(r#"{{"id":"{}","_distance":{},"_similarity":{:.6},"text":"{}"}}"#,
                    id, dist, sim, text.replace('"', "'"))
            }).collect();
            http_json(200, &format!(r#"[{}]"#, results.join(",")))
        }
    }
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
    ).unwrap();
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
        ResponseFormat::Json => {
            http_json(status, &format!(r#"{{"error":"{}","message":"{}"}}"#, error_type, message))
        }
    }
}

// =============================================================================
// JSON PARSING UTILITIES (for input parsing only)
// =============================================================================

fn resolve_fingerprint(s: &str) -> Fingerprint {
    if let Ok(bytes) = base64_decode(s) {
        if let Ok(fp) = Fingerprint::from_bytes(&bytes) {
            return fp;
        }
    }
    Fingerprint::from_content(s)
}

fn extract_json_str(json: &str, key: &str) -> Option<String> {
    let pattern = format!(r#""{}":"#, key);
    let start = json.find(&pattern)?;
    let rest = &json[start + pattern.len()..];

    if rest.starts_with('"') {
        let inner = &rest[1..];
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
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().ok()
}

fn extract_json_f32(json: &str, key: &str) -> Option<f32> {
    let pattern = format!(r#""{}":"#, key);
    let start = json.find(&pattern)?;
    let rest = &json[start + pattern.len()..];
    let end = rest.find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')
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

    inner.split(',')
        .filter_map(|s| {
            let s = s.trim().trim_matches('"');
            if s.is_empty() { None } else { Some(s.to_string()) }
        })
        .collect()
}

fn extract_json_object(json: &str, key: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    if let Some(start) = json.find(&format!(r#""{}":"#, key)) {
        let rest = &json[start..];
        if let Some(obj_start) = rest.find('{') {
            if let Some(obj_end) = rest[obj_start..].find('}') {
                let inner = &rest[obj_start+1..obj_start+obj_end];
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
    }
    map
}

fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    format!("{:08x}-{:04x}-4{:03x}-{:04x}-{:012x}",
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

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║              LadybugDB v{:<38}║", VERSION);
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║  Environment: {:>45}  ║", format!("{:?}", config.environment));
    println!("║  Binding:     {:>45}  ║", format!("{}:{}", config.host, config.port));
    println!("║  Data dir:    {:>45}  ║", config.data_dir);
    println!("║  SIMD:        {:>45}  ║", simd::simd_level());
    println!("║  Workers:     {:>45}  ║", config.workers);
    println!("║  FP bits:     {:>45}  ║", FINGERPRINT_BITS);
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║  Default format: Arrow IPC (application/vnd.apache.arrow.stream) ║");
    println!("║  JSON fallback:  Accept: application/json                       ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");

    let addr: SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .expect("Invalid address");

    let state: SharedState = Arc::new(RwLock::new(DbState::new(&config)));

    let listener = TcpListener::bind(addr)
        .unwrap_or_else(|e| {
            eprintln!("Failed to bind {}: {}", addr, e);
            std::process::exit(1);
        });

    println!("Listening on http://{}", addr);

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
