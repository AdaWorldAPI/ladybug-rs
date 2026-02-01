//! LadybugDB HTTP Server
//!
//! Multi-interface cognitive database server exposing:
//! - REST API (JSON) on /api/*
//! - Redis-compatible text protocol on /redis/*
//! - SQL endpoint on /sql
//! - Cypher endpoint on /cypher
//! - Health/readiness on /health, /ready
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

use ladybug::core::Fingerprint;
use ladybug::core::simd::{self, hamming_distance};
use ladybug::nars::TruthValue;
use ladybug::storage::service::{CognitiveService, ServiceConfig, CpuFeatures};
use ladybug::{FINGERPRINT_BITS, FINGERPRINT_BYTES, VERSION};

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
        let resp = http_json(400, r#"{"error":"bad_request"}"#);
        let _ = stream.write_all(resp.as_bytes());
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

    // Route
    let response = route(method, path, &body_str, state);
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
}

fn route(method: &str, path: &str, body: &str, state: &SharedState) -> String {
    match (method, path) {
        // Health endpoints
        ("GET", "/health") | ("GET", "/healthz") => handle_health(state),
        ("GET", "/ready") | ("GET", "/readyz") => handle_ready(state),
        ("GET", "/") => handle_root(state),

        // Info
        ("GET", "/api/v1/info") => handle_info(state),
        ("GET", "/api/v1/simd") => handle_simd(state),

        // Fingerprint operations
        ("POST", "/api/v1/fingerprint") => handle_fingerprint_create(body, state),
        ("POST", "/api/v1/fingerprint/batch") => handle_fingerprint_batch(body, state),
        ("POST", "/api/v1/hamming") => handle_hamming(body, state),
        ("POST", "/api/v1/similarity") => handle_similarity(body, state),
        ("POST", "/api/v1/bind") => handle_bind(body, state),
        ("POST", "/api/v1/bundle") => handle_bundle(body, state),

        // Search
        ("POST", "/api/v1/search/topk") => handle_topk(body, state),
        ("POST", "/api/v1/search/threshold") => handle_threshold(body, state),
        ("POST", "/api/v1/search/resonate") => handle_resonate(body, state),

        // Index operations
        ("POST", "/api/v1/index") => handle_index(body, state),
        ("GET", "/api/v1/index/count") => handle_index_count(state),
        ("DELETE", "/api/v1/index") => handle_index_clear(state),

        // NARS inference
        ("POST", "/api/v1/nars/deduction") => handle_nars_deduction(body),
        ("POST", "/api/v1/nars/induction") => handle_nars_induction(body),
        ("POST", "/api/v1/nars/abduction") => handle_nars_abduction(body),
        ("POST", "/api/v1/nars/revision") => handle_nars_revision(body),

        // SQL endpoint
        ("POST", "/api/v1/sql") | ("POST", "/sql") => handle_sql(body, state),

        // Cypher endpoint
        ("POST", "/api/v1/cypher") | ("POST", "/cypher") => handle_cypher(body, state),

        // CogRedis text protocol
        ("POST", "/redis") => handle_redis_command(body, state),

        // LanceDB-compatible API
        ("POST", "/api/v1/lance/table") => handle_lance_create_table(body, state),
        ("POST", "/api/v1/lance/add") => handle_lance_add(body, state),
        ("POST", "/api/v1/lance/search") => handle_lance_search(body, state),

        _ => http_json(404, r#"{"error":"not_found","message":"Unknown endpoint"}"#),
    }
}

// =============================================================================
// HANDLER IMPLEMENTATIONS
// =============================================================================

fn handle_health(state: &SharedState) -> String {
    let db = state.read().unwrap();
    let health = db.service.health_check();
    let json = format!(
        r#"{{"status":"ok","uptime_secs":{},"cpu":"{}","buffer_pool_used":{},"version":"{}"}}"#,
        health.uptime_secs, health.cpu_features, health.buffer_pool_used, VERSION
    );
    http_json(200, &json)
}

fn handle_ready(_state: &SharedState) -> String {
    http_json(200, r#"{"status":"ready"}"#)
}

fn handle_root(state: &SharedState) -> String {
    let db = state.read().unwrap();
    let uptime = db.start_time.elapsed().as_secs();
    let json = format!(
        r#"{{
  "name": "LadybugDB",
  "version": "{}",
  "fingerprint_bits": {},
  "fingerprint_bytes": {},
  "simd_level": "{}",
  "uptime_secs": {},
  "indexed_count": {},
  "endpoints": {{
    "health": "/health",
    "info": "/api/v1/info",
    "fingerprint": "POST /api/v1/fingerprint",
    "hamming": "POST /api/v1/hamming",
    "bind": "POST /api/v1/bind",
    "bundle": "POST /api/v1/bundle",
    "topk_search": "POST /api/v1/search/topk",
    "threshold_search": "POST /api/v1/search/threshold",
    "resonate": "POST /api/v1/search/resonate",
    "index": "POST /api/v1/index",
    "sql": "POST /api/v1/sql",
    "cypher": "POST /api/v1/cypher",
    "redis": "POST /redis",
    "lance_search": "POST /api/v1/lance/search",
    "nars_deduction": "POST /api/v1/nars/deduction"
  }}
}}"#,
        VERSION, FINGERPRINT_BITS, FINGERPRINT_BYTES,
        simd::simd_level(), uptime, db.fingerprints.len()
    );
    http_json(200, &json)
}

fn handle_info(state: &SharedState) -> String {
    let db = state.read().unwrap();
    let json = format!(
        r#"{{"version":"{}","fingerprint_bits":{},"fingerprint_bytes":{},"simd":"{}","cpu":{{"avx512":{},"avx2":{},"cores":{}}},"indexed_count":{}}}"#,
        VERSION, FINGERPRINT_BITS, FINGERPRINT_BYTES,
        simd::simd_level(),
        db.cpu.has_avx512f, db.cpu.has_avx2, db.cpu.physical_cores,
        db.fingerprints.len()
    );
    http_json(200, &json)
}

fn handle_simd(_state: &SharedState) -> String {
    let cpu = CpuFeatures::detect();
    let json = format!(
        r#"{{"level":"{}","avx512f":{},"avx512vpopcntdq":{},"avx2":{},"sse42":{},"physical_cores":{},"optimal_batch_size":{}}}"#,
        simd::simd_level(), cpu.has_avx512f, cpu.has_avx512vpopcntdq,
        cpu.has_avx2, cpu.has_sse42, cpu.physical_cores, cpu.optimal_batch_size()
    );
    http_json(200, &json)
}

fn handle_fingerprint_create(body: &str, _state: &SharedState) -> String {
    // Parse JSON: {"content": "hello"} or {"bytes": "base64..."}
    if let Some(content) = extract_json_str(body, "content") {
        let fp = Fingerprint::from_content(&content);
        let bytes = fp.as_bytes();
        let b64 = base64_encode(bytes);
        let json = format!(
            r#"{{"fingerprint":"{}","popcount":{},"density":{:.4},"bits":{}}}"#,
            b64, fp.popcount(), fp.density(), FINGERPRINT_BITS
        );
        http_json(200, &json)
    } else if let Some(b64) = extract_json_str(body, "bytes") {
        match base64_decode(&b64) {
            Ok(bytes) => match Fingerprint::from_bytes(&bytes) {
                Ok(fp) => {
                    let json = format!(
                        r#"{{"fingerprint":"{}","popcount":{},"density":{:.4},"bits":{}}}"#,
                        b64, fp.popcount(), fp.density(), FINGERPRINT_BITS
                    );
                    http_json(200, &json)
                }
                Err(e) => http_json(400, &format!(r#"{{"error":"invalid_fingerprint","message":"{}"}}"#, e)),
            }
            Err(e) => http_json(400, &format!(r#"{{"error":"invalid_base64","message":"{}"}}"#, e)),
        }
    } else {
        // Random fingerprint
        let fp = Fingerprint::random();
        let b64 = base64_encode(fp.as_bytes());
        let json = format!(
            r#"{{"fingerprint":"{}","popcount":{},"density":{:.4},"bits":{},"type":"random"}}"#,
            b64, fp.popcount(), fp.density(), FINGERPRINT_BITS
        );
        http_json(200, &json)
    }
}

fn handle_fingerprint_batch(body: &str, _state: &SharedState) -> String {
    // {"contents": ["hello", "world", ...]}
    let contents = extract_json_str_array(body, "contents");
    if contents.is_empty() {
        return http_json(400, r#"{"error":"missing_field","message":"need contents array"}"#);
    }

    let mut results = Vec::new();
    for c in &contents {
        let fp = Fingerprint::from_content(c);
        let b64 = base64_encode(fp.as_bytes());
        results.push(format!(
            r#"{{"content":"{}","fingerprint":"{}","popcount":{},"density":{:.4}}}"#,
            c, b64, fp.popcount(), fp.density()
        ));
    }

    let json = format!(r#"{{"fingerprints":[{}],"count":{}}}"#, results.join(","), results.len());
    http_json(200, &json)
}

fn handle_hamming(body: &str, _state: &SharedState) -> String {
    let a_str = extract_json_str(body, "a").unwrap_or_default();
    let b_str = extract_json_str(body, "b").unwrap_or_default();

    let fp_a = resolve_fingerprint(&a_str);
    let fp_b = resolve_fingerprint(&b_str);

    let dist = hamming_distance(&fp_a, &fp_b);
    let sim = 1.0 - (dist as f32 / FINGERPRINT_BITS as f32);

    let json = format!(
        r#"{{"distance":{},"similarity":{:.6},"bits":{}}}"#,
        dist, sim, FINGERPRINT_BITS
    );
    http_json(200, &json)
}

fn handle_similarity(body: &str, state: &SharedState) -> String {
    handle_hamming(body, state) // same impl, different name for clarity
}

fn handle_bind(body: &str, _state: &SharedState) -> String {
    let a_str = extract_json_str(body, "a").unwrap_or_default();
    let b_str = extract_json_str(body, "b").unwrap_or_default();

    let fp_a = resolve_fingerprint(&a_str);
    let fp_b = resolve_fingerprint(&b_str);
    let result = fp_a.bind(&fp_b);

    let json = format!(
        r#"{{"result":"{}","popcount":{},"density":{:.4}}}"#,
        base64_encode(result.as_bytes()), result.popcount(), result.density()
    );
    http_json(200, &json)
}

fn handle_bundle(body: &str, _state: &SharedState) -> String {
    let fps_b64 = extract_json_str_array(body, "fingerprints");
    if fps_b64.is_empty() {
        return http_json(400, r#"{"error":"missing_field","message":"need fingerprints array"}"#);
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

    let json = format!(
        r#"{{"result":"{}","popcount":{},"density":{:.4},"input_count":{}}}"#,
        base64_encode(result.as_bytes()), result.popcount(), result.density(), fps.len()
    );
    http_json(200, &json)
}

fn handle_topk(body: &str, state: &SharedState) -> String {
    let query_str = extract_json_str(body, "query").unwrap_or_default();
    let k = extract_json_usize(body, "k").unwrap_or(10);

    let query = resolve_fingerprint(&query_str);
    let db = state.read().unwrap();

    let mut scored: Vec<(usize, u32, f32)> = db.fingerprints.iter().enumerate()
        .map(|(i, (_, fp, _))| {
            let dist = hamming_distance(&query, fp);
            let sim = 1.0 - (dist as f32 / FINGERPRINT_BITS as f32);
            (i, dist, sim)
        })
        .collect();

    scored.sort_by_key(|&(_, d, _)| d);
    scored.truncate(k);

    let results: Vec<String> = scored.iter().map(|&(idx, dist, sim)| {
        let (ref id, _, ref meta): &(String, Fingerprint, HashMap<String, String>) = &db.fingerprints[idx];
        let meta_json = meta.iter()
            .map(|(k, v)| format!(r#""{}":"{}""#, k, v))
            .collect::<Vec<_>>().join(",");
        format!(
            r#"{{"index":{},"id":"{}","distance":{},"similarity":{:.6},"metadata":{{{}}}}}"#,
            idx, id, dist, sim, meta_json
        )
    }).collect();

    let json = format!(r#"{{"results":[{}],"count":{},"total_indexed":{}}}"#,
        results.join(","), results.len(), db.fingerprints.len());
    http_json(200, &json)
}

fn handle_threshold(body: &str, state: &SharedState) -> String {
    let query_str = extract_json_str(body, "query").unwrap_or_default();
    let max_distance = extract_json_usize(body, "max_distance").unwrap_or(2000) as u32;
    let limit = extract_json_usize(body, "limit").unwrap_or(100);

    let query = resolve_fingerprint(&query_str);
    let db = state.read().unwrap();

    let mut results: Vec<String> = Vec::new();
    for (idx, (id, fp, meta)) in db.fingerprints.iter().enumerate() {
        let dist = hamming_distance(&query, fp);
        if dist <= max_distance {
            let sim = 1.0 - (dist as f32 / FINGERPRINT_BITS as f32);
            let meta_json = meta.iter()
                .map(|(k, v)| format!(r#""{}":"{}""#, k, v))
                .collect::<Vec<_>>().join(",");
            results.push(format!(
                r#"{{"index":{},"id":"{}","distance":{},"similarity":{:.6},"metadata":{{{}}}}}"#,
                idx, id, dist, sim, meta_json
            ));
            if results.len() >= limit { break; }
        }
    }

    let json = format!(r#"{{"results":[{}],"count":{},"max_distance":{},"total_indexed":{}}}"#,
        results.join(","), results.len(), max_distance, db.fingerprints.len());
    http_json(200, &json)
}

fn handle_resonate(body: &str, state: &SharedState) -> String {
    // Content-based resonance search
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

    let results: Vec<String> = scored.iter().map(|&(idx, sim)| {
        let (ref id, _, ref _meta): &(String, Fingerprint, HashMap<String, String>) = &db.fingerprints[idx];
        format!(r#"{{"index":{},"id":"{}","similarity":{:.6}}}"#, idx, id, sim)
    }).collect();

    let json = format!(r#"{{"results":[{}],"count":{},"content":"{}","threshold":{}}}"#,
        results.join(","), results.len(), content, threshold);
    http_json(200, &json)
}

fn handle_index(body: &str, state: &SharedState) -> String {
    // {"id": "node1", "content": "hello", "metadata": {"type": "thought"}}
    // or {"id": "node1", "fingerprint": "base64...", "metadata": {...}}
    let id = extract_json_str(body, "id").unwrap_or_else(|| uuid_v4());

    let fp = if let Some(content) = extract_json_str(body, "content") {
        Fingerprint::from_content(&content)
    } else if let Some(b64) = extract_json_str(body, "fingerprint") {
        resolve_fingerprint(&b64)
    } else {
        return http_json(400, r#"{"error":"missing_field","message":"need content or fingerprint"}"#);
    };

    let meta = extract_json_object(body, "metadata");

    let mut db = state.write().unwrap();
    let idx = db.fingerprints.len();
    db.fingerprints.push((id.clone(), fp, meta));

    let json = format!(r#"{{"indexed":true,"id":"{}","index":{},"total":{}}}"#,
        id, idx, db.fingerprints.len());
    http_json(200, &json)
}

fn handle_index_count(state: &SharedState) -> String {
    let db = state.read().unwrap();
    http_json(200, &format!(r#"{{"count":{}}}"#, db.fingerprints.len()))
}

fn handle_index_clear(state: &SharedState) -> String {
    let mut db = state.write().unwrap();
    let was = db.fingerprints.len();
    db.fingerprints.clear();
    http_json(200, &format!(r#"{{"cleared":true,"was":{}}}"#, was))
}

// NARS
fn handle_nars_deduction(body: &str) -> String {
    nars_binary_op(body, |a, b| a.deduction(&b))
}
fn handle_nars_induction(body: &str) -> String {
    nars_binary_op(body, |a, b| a.induction(&b))
}
fn handle_nars_abduction(body: &str) -> String {
    nars_binary_op(body, |a, b| a.abduction(&b))
}
fn handle_nars_revision(body: &str) -> String {
    nars_binary_op(body, |a, b| a.revision(&b))
}

fn nars_binary_op(body: &str, op: impl Fn(TruthValue, TruthValue) -> TruthValue) -> String {
    let f1 = extract_json_f32(body, "f1").unwrap_or(0.9);
    let c1 = extract_json_f32(body, "c1").unwrap_or(0.9);
    let f2 = extract_json_f32(body, "f2").unwrap_or(0.9);
    let c2 = extract_json_f32(body, "c2").unwrap_or(0.9);

    let a = TruthValue::new(f1, c1);
    let b = TruthValue::new(f2, c2);
    let result = op(a, b);

    let json = format!(
        r#"{{"frequency":{:.6},"confidence":{:.6},"expectation":{:.6}}}"#,
        result.frequency, result.confidence, result.expectation()
    );
    http_json(200, &json)
}

// SQL
fn handle_sql(body: &str, _state: &SharedState) -> String {
    let query = extract_json_str(body, "query")
        .or_else(|| Some(body.to_string()))
        .unwrap_or_default();

    // For now, return acknowledgment — full DataFusion integration needs async runtime
    let json = format!(
        r#"{{"status":"acknowledged","query":"{}","note":"Full DataFusion SQL execution available via library API. REST SQL coming in v0.3."}}"#,
        query.replace('"', "'").chars().take(200).collect::<String>()
    );
    http_json(200, &json)
}

// Cypher
fn handle_cypher(body: &str, _state: &SharedState) -> String {
    let query = extract_json_str(body, "query").unwrap_or_default();

    // Transpile to SQL
    match ladybug::query::cypher_to_sql(&query) {
        Ok(sql) => {
            let json = format!(
                r#"{{"cypher":"{}","transpiled_sql":"{}","status":"transpiled"}}"#,
                query.replace('"', "'"), sql.replace('"', "'")
            );
            http_json(200, &json)
        }
        Err(e) => {
            http_json(400, &format!(
                r#"{{"error":"cypher_parse_error","message":"{}"}}"#,
                e.to_string().replace('"', "'")
            ))
        }
    }
}

// CogRedis text commands
fn handle_redis_command(body: &str, state: &SharedState) -> String {
    let parts: Vec<&str> = body.trim().split_whitespace().collect();
    if parts.is_empty() {
        return http_json(400, r#"{"error":"empty_command"}"#);
    }

    let cmd = parts[0].to_uppercase();
    match cmd.as_str() {
        "PING" => http_json(200, r#"{"reply":"PONG"}"#),
        "SET" if parts.len() >= 3 => {
            let key = parts[1].to_string();
            let val = parts[2..].join(" ");
            state.write().unwrap().kv.insert(key.clone(), val.clone());
            http_json(200, &format!(r#"{{"reply":"OK","key":"{}"}}"#, key))
        }
        "GET" if parts.len() >= 2 => {
            let key = parts[1];
            let db = state.read().unwrap();
            match db.kv.get(key) {
                Some(v) => http_json(200, &format!(r#"{{"reply":"{}"}}"#, v)),
                None => http_json(200, r#"{"reply":null}"#),
            }
        }
        "DEL" if parts.len() >= 2 => {
            let key = parts[1];
            let removed = state.write().unwrap().kv.remove(key).is_some();
            http_json(200, &format!(r#"{{"reply":{}}}"#, if removed { 1 } else { 0 }))
        }
        "KEYS" => {
            let pattern = if parts.len() >= 2 { parts[1] } else { "*" };
            let db = state.read().unwrap();
            let keys: Vec<&String> = if pattern == "*" {
                db.kv.keys().collect()
            } else {
                db.kv.keys().filter(|k| k.contains(pattern.trim_matches('*'))).collect()
            };
            let json_keys: Vec<String> = keys.iter().map(|k| format!(r#""{}""#, k)).collect();
            http_json(200, &format!(r#"{{"reply":[{}]}}"#, json_keys.join(",")))
        }
        "INFO" => {
            let db = state.read().unwrap();
            let json = format!(
                r#"{{"reply":"ladybugdb v{}\nsimd:{}\nindexed:{}\nkeys:{}\nuptime:{}s"}}"#,
                VERSION, simd::simd_level(), db.fingerprints.len(),
                db.kv.len(), db.start_time.elapsed().as_secs()
            );
            http_json(200, &json)
        }
        _ => http_json(400, &format!(r#"{{"error":"unknown_command","command":"{}"}}"#, cmd)),
    }
}

// LanceDB-compatible API
fn handle_lance_create_table(body: &str, _state: &SharedState) -> String {
    let name = extract_json_str(body, "name").unwrap_or_else(|| "default".to_string());
    http_json(200, &format!(r#"{{"table":"{}","status":"created","note":"In-memory table backed by indexed fingerprints"}}"#, name))
}

fn handle_lance_add(body: &str, state: &SharedState) -> String {
    // LanceDB-compatible: {"data": [{"vector": [...], "id": "...", "text": "..."}]}
    // We map vector → fingerprint via content hash
    let id = extract_json_str(body, "id").unwrap_or_else(|| uuid_v4());
    let text = extract_json_str(body, "text").unwrap_or_default();

    let fp = Fingerprint::from_content(&text);
    let mut meta = HashMap::new();
    meta.insert("text".to_string(), text);

    let mut db = state.write().unwrap();
    let idx = db.fingerprints.len();
    db.fingerprints.push((id.clone(), fp, meta));

    http_json(200, &format!(r#"{{"id":"{}","index":{}}}"#, id, idx))
}

fn handle_lance_search(body: &str, state: &SharedState) -> String {
    // LanceDB-compatible search: {"query": "text", "limit": 10}
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

    let results: Vec<String> = scored.iter().map(|&(idx, dist, sim)| {
        let (ref id, _, ref meta): &(String, Fingerprint, HashMap<String, String>) = &db.fingerprints[idx];
        let text = meta.get("text").cloned().unwrap_or_default();
        format!(r#"{{"id":"{}","_distance":{},"_similarity":{:.6},"text":"{}"}}"#,
            id, dist, sim, text.replace('"', "'"))
    }).collect();

    http_json(200, &format!(r#"[{}]"#, results.join(",")))
}

// =============================================================================
// UTILITIES
// =============================================================================

fn http_json(status: u16, body: &str) -> String {
    let status_text = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "Unknown",
    };

    format!(
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        status, status_text, body.len(), body
    )
}

fn resolve_fingerprint(s: &str) -> Fingerprint {
    // Try base64 first
    if let Ok(bytes) = base64_decode(s) {
        if let Ok(fp) = Fingerprint::from_bytes(&bytes) {
            return fp;
        }
    }
    // Fall back to content hash
    Fingerprint::from_content(s)
}

// Simple JSON parsing (no serde in server binary to keep it lean)
fn extract_json_str(json: &str, key: &str) -> Option<String> {
    let pattern = format!(r#""{}":"#, key);
    let start = json.find(&pattern)?;
    let rest = &json[start + pattern.len()..];

    if rest.starts_with('"') {
        // String value
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
    let _pattern = format!(r#""{}":{{" "#, key);
    // Simple extraction — just capture key-value pairs
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

    println!("╔═══════════════════════════════════════════════════════╗");
    println!("║              LadybugDB v{:<26}║", VERSION);
    println!("╠═══════════════════════════════════════════════════════╣");
    println!("║  Environment: {:>39}  ║", format!("{:?}", config.environment));
    println!("║  Binding:     {:>39}  ║", format!("{}:{}", config.host, config.port));
    println!("║  Data dir:    {:>39}  ║", config.data_dir);
    println!("║  SIMD:        {:>39}  ║", simd::simd_level());
    println!("║  Workers:     {:>39}  ║", config.workers);
    println!("║  FP bits:     {:>39}  ║", FINGERPRINT_BITS);
    println!("╚═══════════════════════════════════════════════════════╝");

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
