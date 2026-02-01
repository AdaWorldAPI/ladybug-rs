//! Ladybug-RS HTTP API Server
//!
//! A cognitive database server exposing multiple query interfaces:
//! - Redis-like commands via /redis endpoint
//! - SQL via /sql endpoint
//! - Cypher via /cypher endpoint
//! - LanceDB-compatible vector search via /vectors endpoint
//! - BindSpace operations via /bind endpoint
//! - CAM operations via /cam endpoint
//! - Health check via /health
//! - Metrics via /metrics
//!
//! Deployment:
//! - Railway: Binds to 0.0.0.0:$PORT (auto-detected)
//! - Claude Backend: Binds to 127.0.0.1:$PORT
//! - Docker: Configurable via HOST and PORT env vars

use std::collections::HashMap;
use std::env;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant};

// Core types
use ladybug::core::Fingerprint;
use ladybug::FINGERPRINT_U64;

// Storage - Modern imports
use ladybug::storage::{
    // BindSpace - Universal DTO (8+8 addressing)
    BindSpace, BindNode, Addr,
    FINGERPRINT_WORDS,

    // Substrate - Unified interface
    Substrate, SubstrateConfig,

    // Redis Adapter - Production-ready interface
    RedisAdapter, RedisResult, CogAddr,

    // CogRedis - Legacy support
    CogRedis, SetOptions,

    // Service - Container lifecycle
    CpuFeatures,
};

// =============================================================================
// SERVER CONFIGURATION
// =============================================================================

/// Server configuration from environment
struct ServerConfig {
    host: String,
    port: u16,
    data_dir: String,
    max_connections: usize,
    read_timeout: Duration,
    write_timeout: Duration,
    graceful_shutdown_timeout: Duration,
}

impl ServerConfig {
    fn from_env() -> Self {
        let hostname = env::var("HOSTNAME").unwrap_or_default();
        let railway_env = env::var("RAILWAY_ENVIRONMENT").ok();
        let railway_static_url = env::var("RAILWAY_STATIC_URL").ok();

        // Detect Railway deployment
        let is_railway = hostname.contains("railway")
            || hostname.ends_with(".internal")
            || railway_env.is_some()
            || railway_static_url.is_some();

        let (default_host, default_port) = if is_railway {
            // Railway requires 0.0.0.0 binding
            ("0.0.0.0".to_string(), 8080)
        } else {
            // Local/Claude backend: localhost by default
            ("127.0.0.1".to_string(), 5000)
        };

        let host = env::var("HOST").unwrap_or(default_host);
        let port = env::var("PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(default_port);

        let data_dir = env::var("DATA_DIR")
            .unwrap_or_else(|_| "/tmp/ladybug".to_string());

        let max_connections = env::var("MAX_CONNECTIONS")
            .ok()
            .and_then(|c| c.parse().ok())
            .unwrap_or(1000);

        ServerConfig {
            host,
            port,
            data_dir,
            max_connections,
            read_timeout: Duration::from_secs(30),
            write_timeout: Duration::from_secs(30),
            graceful_shutdown_timeout: Duration::from_secs(30),
        }
    }
}

// =============================================================================
// DATABASE STATE
// =============================================================================

/// Shared database state with modern storage layers
struct DatabaseState {
    // Core storage layers
    bind_space: BindSpace,
    substrate: Substrate,
    redis_adapter: RedisAdapter,

    // Legacy CogRedis for backwards compatibility
    cog_redis: CogRedis,

    // Metrics
    start_time: Instant,
    requests_total: std::sync::atomic::AtomicU64,
    requests_success: std::sync::atomic::AtomicU64,
    requests_error: std::sync::atomic::AtomicU64,
}

impl DatabaseState {
    fn new(_data_dir: &str) -> Self {
        // Initialize BindSpace (Universal DTO)
        let bind_space = BindSpace::new();

        // Initialize Substrate (unified interface)
        let substrate_config = SubstrateConfig::default();
        let substrate = Substrate::new(substrate_config);

        // Initialize Redis Adapter
        let redis_adapter = RedisAdapter::new(SubstrateConfig::default());

        // Initialize legacy CogRedis
        let cog_redis = CogRedis::new();

        Self {
            bind_space,
            substrate,
            redis_adapter,
            cog_redis,
            start_time: Instant::now(),
            requests_total: std::sync::atomic::AtomicU64::new(0),
            requests_success: std::sync::atomic::AtomicU64::new(0),
            requests_error: std::sync::atomic::AtomicU64::new(0),
        }
    }

    fn record_request(&self, success: bool) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        if success {
            self.requests_success.fetch_add(1, Ordering::Relaxed);
        } else {
            self.requests_error.fetch_add(1, Ordering::Relaxed);
        }
    }
}

// =============================================================================
// HTTP REQUEST/RESPONSE
// =============================================================================

/// HTTP Request parsed from raw bytes
struct HttpRequest {
    method: String,
    path: String,
    query_params: HashMap<String, String>,
    headers: HashMap<String, String>,
    body: String,
}

impl HttpRequest {
    fn parse(raw: &str) -> Option<Self> {
        let mut lines = raw.lines();
        let first_line = lines.next()?;
        let mut parts = first_line.split_whitespace();

        let method = parts.next()?.to_string();
        let full_path = parts.next()?.to_string();

        // Parse path and query string
        let (path, query_params) = if let Some(idx) = full_path.find('?') {
            let path = full_path[..idx].to_string();
            let query_str = &full_path[idx + 1..];
            let params = query_str
                .split('&')
                .filter_map(|pair| {
                    let mut parts = pair.splitn(2, '=');
                    Some((parts.next()?.to_string(), parts.next().unwrap_or("").to_string()))
                })
                .collect();
            (path, params)
        } else {
            (full_path, HashMap::new())
        };

        let mut headers = HashMap::new();
        let mut body_start = false;
        let mut body_lines = Vec::new();

        for line in lines {
            if body_start {
                body_lines.push(line);
            } else if line.is_empty() {
                body_start = true;
            } else if let Some((key, value)) = line.split_once(':') {
                headers.insert(key.trim().to_lowercase(), value.trim().to_string());
            }
        }

        Some(HttpRequest {
            method,
            path,
            query_params,
            headers,
            body: body_lines.join("\n"),
        })
    }
}

/// HTTP Response builder
struct HttpResponse {
    status: u16,
    status_text: String,
    headers: Vec<(String, String)>,
    body: String,
}

impl HttpResponse {
    fn ok() -> Self {
        Self {
            status: 200,
            status_text: "OK".to_string(),
            headers: vec![
                ("Content-Type".to_string(), "application/json".to_string()),
                ("Access-Control-Allow-Origin".to_string(), "*".to_string()),
                ("Access-Control-Allow-Methods".to_string(), "GET, POST, PUT, DELETE, OPTIONS".to_string()),
                ("Access-Control-Allow-Headers".to_string(), "Content-Type, Authorization".to_string()),
                ("X-Powered-By".to_string(), "ladybug-rs".to_string()),
            ],
            body: String::new(),
        }
    }

    fn error(status: u16, message: &str) -> Self {
        let status_text = match status {
            400 => "Bad Request",
            401 => "Unauthorized",
            403 => "Forbidden",
            404 => "Not Found",
            405 => "Method Not Allowed",
            429 => "Too Many Requests",
            500 => "Internal Server Error",
            503 => "Service Unavailable",
            _ => "Error",
        };

        Self {
            status,
            status_text: status_text.to_string(),
            headers: vec![
                ("Content-Type".to_string(), "application/json".to_string()),
                ("Access-Control-Allow-Origin".to_string(), "*".to_string()),
            ],
            body: format!(r#"{{"error": "{}", "status": {}}}"#, message, status),
        }
    }

    fn json(data: &str) -> Self {
        let mut resp = Self::ok();
        resp.body = data.to_string();
        resp
    }

    fn to_bytes(&self) -> Vec<u8> {
        let mut response = format!(
            "HTTP/1.1 {} {}\r\n",
            self.status, self.status_text
        );

        for (key, value) in &self.headers {
            response.push_str(&format!("{}: {}\r\n", key, value));
        }

        response.push_str(&format!("Content-Length: {}\r\n", self.body.len()));
        response.push_str("\r\n");
        response.push_str(&self.body);

        response.into_bytes()
    }
}

// =============================================================================
// REQUEST HANDLERS
// =============================================================================

/// Handle incoming HTTP request
fn handle_request(
    request: &HttpRequest,
    state: &Arc<RwLock<DatabaseState>>,
) -> HttpResponse {
    // Handle CORS preflight
    if request.method == "OPTIONS" {
        return HttpResponse::ok();
    }

    let response = match (request.method.as_str(), request.path.as_str()) {
        // Health check
        ("GET", "/health") => handle_health(state),

        // Readiness check (for Kubernetes/Railway)
        ("GET", "/ready") => handle_ready(state),

        // Liveness check
        ("GET", "/live") => HttpResponse::json(r#"{"status": "alive"}"#),

        // Metrics (Prometheus format)
        ("GET", "/metrics") => handle_metrics(state),

        // Server info
        ("GET", "/") | ("GET", "/info") => handle_info(state),

        // BindSpace operations (modern)
        ("POST", "/bind") => handle_bind_operation(&request.body, state),
        ("GET", path) if path.starts_with("/bind/") => {
            let addr_str = &path[6..];
            handle_bind_get(addr_str, state)
        }

        // Substrate operations
        ("POST", "/substrate/write") => handle_substrate_write(&request.body, state),
        ("POST", "/substrate/query") => handle_substrate_query(&request.body, state),

        // Redis-like commands (uses RedisAdapter)
        ("POST", "/redis") => handle_redis_command(&request.body, state),

        // SQL queries
        ("POST", "/sql") => handle_sql_query(&request.body, state),

        // Cypher queries
        ("POST", "/cypher") => handle_cypher_query(&request.body, state),

        // Vector search (LanceDB-compatible)
        ("POST", "/vectors/search") => handle_vector_search(&request.body, state),
        ("POST", "/vectors/insert") => handle_vector_insert(&request.body, state),

        // CAM operations
        ("POST", path) if path.starts_with("/cam/") => {
            let op_name = &path[5..];
            handle_cam_operation(op_name, &request.body, state)
        }

        // Create fingerprint from content
        ("POST", "/fingerprint") => handle_fingerprint_create(&request.body),

        // Resonate (similarity search)
        ("POST", "/resonate") => handle_resonate(&request.body, state),

        // Not found
        _ => HttpResponse::error(404, "Endpoint not found"),
    };

    // Record metrics
    {
        let db = state.read().unwrap();
        db.record_request(response.status < 400);
    }

    response
}

/// Handle health check
fn handle_health(state: &Arc<RwLock<DatabaseState>>) -> HttpResponse {
    let db = state.read().unwrap();

    let response = serde_json::json!({
        "status": "healthy",
        "service": "ladybug-rs",
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_secs": db.start_time.elapsed().as_secs(),
        "requests": {
            "total": db.requests_total.load(Ordering::Relaxed),
            "success": db.requests_success.load(Ordering::Relaxed),
            "error": db.requests_error.load(Ordering::Relaxed)
        },
        "cpu_features": detect_cpu_features()
    });

    HttpResponse::json(&response.to_string())
}

/// Handle readiness check
fn handle_ready(_state: &Arc<RwLock<DatabaseState>>) -> HttpResponse {
    // Always ready if server is running
    HttpResponse::json(r#"{"ready": true}"#)
}

/// Handle metrics (Prometheus format)
fn handle_metrics(state: &Arc<RwLock<DatabaseState>>) -> HttpResponse {
    let db = state.read().unwrap();

    let requests_success = db.requests_success.load(Ordering::Relaxed);
    let requests_error = db.requests_error.load(Ordering::Relaxed);
    let uptime = db.start_time.elapsed().as_secs();

    let substrate_stats = db.substrate.stats();

    let metrics = format!(
        r#"# HELP ladybug_requests_total Total HTTP requests
# TYPE ladybug_requests_total counter
ladybug_requests_total {{status="success"}} {}
ladybug_requests_total {{status="error"}} {}

# HELP ladybug_uptime_seconds Server uptime in seconds
# TYPE ladybug_uptime_seconds gauge
ladybug_uptime_seconds {}

# HELP ladybug_substrate_hot_nodes Nodes in hot cache
# TYPE ladybug_substrate_hot_nodes gauge
ladybug_substrate_hot_nodes {}

# HELP ladybug_substrate_hot_edges Edges in hot cache
# TYPE ladybug_substrate_hot_edges gauge
ladybug_substrate_hot_edges {}

# HELP ladybug_substrate_hot_hits Cache hits
# TYPE ladybug_substrate_hot_hits counter
ladybug_substrate_hot_hits {}

# HELP ladybug_substrate_hot_misses Cache misses
# TYPE ladybug_substrate_hot_misses counter
ladybug_substrate_hot_misses {}
"#,
        requests_success,
        requests_error,
        uptime,
        substrate_stats.hot_nodes.load(Ordering::Relaxed),
        substrate_stats.hot_edges.load(Ordering::Relaxed),
        substrate_stats.hot_hits.load(Ordering::Relaxed),
        substrate_stats.hot_misses.load(Ordering::Relaxed),
    );

    let mut response = HttpResponse::ok();
    response.headers.retain(|(k, _)| k != "Content-Type");
    response.headers.push(("Content-Type".to_string(), "text/plain; charset=utf-8".to_string()));
    response.body = metrics;
    response
}

/// Handle server info
fn handle_info(state: &Arc<RwLock<DatabaseState>>) -> HttpResponse {
    let db = state.read().unwrap();

    let info = serde_json::json!({
        "name": "ladybug-rs",
        "version": env!("CARGO_PKG_VERSION"),
        "description": "Crystal Lake Cognitive Database",
        "architecture": {
            "address_space": "16-bit (8+8 prefix:slot)",
            "surface_zone": "0x00-0x0F (4,096 slots)",
            "fluid_zone": "0x10-0x7F (28,672 slots)",
            "node_zone": "0x80-0xFF (32,768 slots)"
        },
        "endpoints": {
            "/health": "Health check with detailed status",
            "/ready": "Kubernetes readiness probe",
            "/live": "Kubernetes liveness probe",
            "/metrics": "Prometheus metrics",
            "/redis": "Redis-like commands (POST)",
            "/sql": "SQL queries (POST)",
            "/cypher": "Cypher graph queries (POST)",
            "/bind": "BindSpace operations (POST)",
            "/bind/:addr": "Get node by address (GET)",
            "/substrate/write": "Substrate write (POST)",
            "/substrate/query": "Substrate query (POST)",
            "/vectors/search": "Vector similarity search (POST)",
            "/vectors/insert": "Insert vectors (POST)",
            "/cam/:op": "CAM operations (POST)",
            "/fingerprint": "Create fingerprint (POST)",
            "/resonate": "Similarity search (POST)"
        },
        "features": {
            "fingerprint_bits": FINGERPRINT_WORDS * 64,
            "fingerprint_words": FINGERPRINT_WORDS,
            "cam_operations": 4096,
            "cpu_features": detect_cpu_features()
        },
        "stats": {
            "uptime_secs": db.start_time.elapsed().as_secs(),
            "requests_total": db.requests_total.load(Ordering::Relaxed),
            "substrate_hot_nodes": db.substrate.stats().hot_nodes.load(Ordering::Relaxed)
        }
    });

    HttpResponse::json(&info.to_string())
}

/// Handle BindSpace operations
fn handle_bind_operation(body: &str, state: &Arc<RwLock<DatabaseState>>) -> HttpResponse {
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(body);

    match parsed {
        Ok(json) => {
            let op = json.get("operation")
                .and_then(|v| v.as_str())
                .unwrap_or("write");

            let mut db = state.write().unwrap();

            match op {
                "write" | "set" => {
                    let content = json.get("content")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let label = json.get("label")
                        .and_then(|v| v.as_str());
                    let fp = Fingerprint::from_content(content);

                    // Create fingerprint array
                    let mut fp_arr = [0u64; FINGERPRINT_WORDS];
                    fp_arr.copy_from_slice(&fp.as_raw()[..FINGERPRINT_WORDS]);

                    // Write to BindSpace
                    let addr = if let Some(lbl) = label {
                        db.bind_space.write_labeled(fp_arr, lbl)
                    } else {
                        db.bind_space.write(fp_arr)
                    };

                    let response = serde_json::json!({
                        "success": true,
                        "addr": format!("{:04X}", addr.0)
                    });
                    HttpResponse::json(&response.to_string())
                }
                "read" | "get" => {
                    let addr_str = json.get("addr")
                        .and_then(|v| v.as_str())
                        .unwrap_or("0000");
                    let addr = u16::from_str_radix(addr_str.trim_start_matches("0x"), 16)
                        .map(Addr)
                        .unwrap_or(Addr(0));

                    if let Some(node) = db.bind_space.read(addr) {
                        // Compute popcount from fingerprint
                        let popcount: u32 = node.fingerprint.iter()
                            .map(|w| w.count_ones())
                            .sum();
                        let response = serde_json::json!({
                            "success": true,
                            "addr": format!("{:04X}", addr.0),
                            "popcount": popcount,
                            "density": popcount as f32 / (FINGERPRINT_WORDS * 64) as f32,
                            "label": node.label,
                            "qidx": node.qidx
                        });
                        HttpResponse::json(&response.to_string())
                    } else {
                        HttpResponse::error(404, "Address not found")
                    }
                }
                _ => HttpResponse::error(400, &format!("Unknown operation: {}", op))
            }
        }
        Err(e) => HttpResponse::error(400, &format!("Invalid JSON: {}", e)),
    }
}

/// Handle GET /bind/:addr
fn handle_bind_get(addr_str: &str, state: &Arc<RwLock<DatabaseState>>) -> HttpResponse {
    let addr = u16::from_str_radix(addr_str.trim_start_matches("0x"), 16)
        .map(Addr)
        .unwrap_or(Addr(0));

    let db = state.read().unwrap();

    if let Some(node) = db.bind_space.read(addr) {
        // Compute popcount from fingerprint
        let popcount: u32 = node.fingerprint.iter()
            .map(|w| w.count_ones())
            .sum();

        // Determine tier from address prefix
        let prefix = (addr.0 >> 8) as u8;
        let tier = if prefix < 0x10 {
            "surface"
        } else if prefix < 0x80 {
            "fluid"
        } else {
            "node"
        };

        let response = serde_json::json!({
            "success": true,
            "addr": format!("{:04X}", addr.0),
            "tier": tier,
            "popcount": popcount,
            "density": popcount as f32 / (FINGERPRINT_WORDS * 64) as f32,
            "label": node.label,
            "qidx": node.qidx,
            "access_count": node.access_count
        });
        HttpResponse::json(&response.to_string())
    } else {
        HttpResponse::error(404, "Address not found")
    }
}

/// Handle Substrate write
fn handle_substrate_write(body: &str, state: &Arc<RwLock<DatabaseState>>) -> HttpResponse {
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(body);

    match parsed {
        Ok(json) => {
            let content = json.get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let label = json.get("label")
                .and_then(|v| v.as_str());

            let fp = Fingerprint::from_content(content);
            let db = state.read().unwrap();

            // Create fingerprint array
            let mut fp_arr = [0u64; FINGERPRINT_WORDS];
            fp_arr.copy_from_slice(&fp.as_raw()[..FINGERPRINT_WORDS]);

            // Write with or without label
            let addr = if let Some(lbl) = label {
                db.substrate.write_labeled(fp_arr, lbl)
            } else {
                db.substrate.write(fp_arr)
            };

            let response = serde_json::json!({
                "success": true,
                "addr": format!("{:04X}", addr.0)
            });
            HttpResponse::json(&response.to_string())
        }
        Err(e) => HttpResponse::error(400, &format!("Invalid JSON: {}", e)),
    }
}

/// Handle Substrate query
fn handle_substrate_query(body: &str, state: &Arc<RwLock<DatabaseState>>) -> HttpResponse {
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(body);

    match parsed {
        Ok(json) => {
            let query_content = json.get("query")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let k = json.get("k")
                .and_then(|v| v.as_u64())
                .unwrap_or(10) as usize;

            let query_fp = Fingerprint::from_content(query_content);
            let mut query_arr = [0u64; FINGERPRINT_WORDS];
            query_arr.copy_from_slice(&query_fp.as_raw()[..FINGERPRINT_WORDS]);

            let db = state.read().unwrap();
            let results = db.substrate.resonate(&query_arr, k);

            let matches: Vec<serde_json::Value> = results.iter().map(|(addr, similarity)| {
                serde_json::json!({
                    "addr": format!("{:04X}", addr.0),
                    "similarity": similarity
                })
            }).collect();

            let response = serde_json::json!({
                "success": true,
                "matches": matches,
                "count": matches.len()
            });
            HttpResponse::json(&response.to_string())
        }
        Err(e) => HttpResponse::error(400, &format!("Invalid JSON: {}", e)),
    }
}

/// Handle Redis-like commands (uses RedisAdapter)
fn handle_redis_command(body: &str, state: &Arc<RwLock<DatabaseState>>) -> HttpResponse {
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(body);

    match parsed {
        Ok(json) => {
            let command = json.get("command")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            let mut db = state.write().unwrap();

            // Try RedisAdapter first
            let result = db.redis_adapter.execute(command);

            let response = match result {
                RedisResult::Ok => serde_json::json!({
                    "success": true,
                    "result": "OK"
                }),
                RedisResult::String(s) => serde_json::json!({
                    "success": true,
                    "result": s
                }),
                RedisResult::Integer(i) => serde_json::json!({
                    "success": true,
                    "result": i
                }),
                RedisResult::Float(f) => serde_json::json!({
                    "success": true,
                    "result": f
                }),
                RedisResult::Addr(a) => serde_json::json!({
                    "success": true,
                    "result": format!("{:04X}", a.0)
                }),
                RedisResult::Array(arr) => serde_json::json!({
                    "success": true,
                    "result": format!("{:?}", arr)
                }),
                RedisResult::Node(node) => serde_json::json!({
                    "success": true,
                    "result": {
                        "addr": format!("{:04X}", node.addr.0),
                        "popcount": node.popcount,
                        "tier": format!("{:?}", node.tier),
                        "label": node.label
                    }
                }),
                RedisResult::Search(hits) => {
                    let results: Vec<_> = hits.iter().map(|h| {
                        serde_json::json!({
                            "addr": format!("{:04X}", h.addr.0),
                            "distance": h.distance,
                            "similarity": h.similarity
                        })
                    }).collect();
                    serde_json::json!({
                        "success": true,
                        "result": results
                    })
                }
                RedisResult::Edge(edge) => serde_json::json!({
                    "success": true,
                    "result": {
                        "from": format!("{:04X}", edge.from.0),
                        "verb": format!("{:04X}", edge.verb.0),
                        "to": format!("{:04X}", edge.to.0),
                        "weight": edge.weight
                    }
                }),
                RedisResult::Nil => serde_json::json!({
                    "success": true,
                    "result": null
                }),
                RedisResult::Error(e) => serde_json::json!({
                    "success": false,
                    "error": e
                }),
            };

            HttpResponse::json(&response.to_string())
        }
        Err(e) => HttpResponse::error(400, &format!("Invalid JSON: {}", e)),
    }
}

/// Handle SQL queries
fn handle_sql_query(body: &str, state: &Arc<RwLock<DatabaseState>>) -> HttpResponse {
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(body);

    match parsed {
        Ok(json) => {
            let query = json.get("query")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            // Translate SQL to internal operations
            let db = state.read().unwrap();
            let stats = db.substrate.stats();

            // Basic SQL support - return node counts for SELECT *
            if query.to_uppercase().contains("SELECT") {
                let response = serde_json::json!({
                    "success": true,
                    "query": query,
                    "rows": [],
                    "metadata": {
                        "hot_nodes": stats.hot_nodes.load(Ordering::Relaxed),
                        "hot_edges": stats.hot_edges.load(Ordering::Relaxed)
                    }
                });
                HttpResponse::json(&response.to_string())
            } else {
                HttpResponse::error(400, "Only SELECT queries currently supported")
            }
        }
        Err(e) => HttpResponse::error(400, &format!("Invalid JSON: {}", e)),
    }
}

/// Handle Cypher queries
fn handle_cypher_query(body: &str, _state: &Arc<RwLock<DatabaseState>>) -> HttpResponse {
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(body);

    match parsed {
        Ok(json) => {
            let query = json.get("query")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            // Pattern matching via Cypher syntax
            let response = if query.to_uppercase().starts_with("MATCH") {
                serde_json::json!({
                    "success": true,
                    "query": query,
                    "nodes": [],
                    "relationships": [],
                    "message": "Cypher query parsed"
                })
            } else if query.to_uppercase().starts_with("CREATE") {
                serde_json::json!({
                    "success": true,
                    "query": query,
                    "created": 0,
                    "message": "Create processed"
                })
            } else {
                serde_json::json!({
                    "success": false,
                    "query": query,
                    "message": "Unsupported Cypher operation"
                })
            };

            HttpResponse::json(&response.to_string())
        }
        Err(e) => HttpResponse::error(400, &format!("Invalid JSON: {}", e)),
    }
}

/// Handle vector search (LanceDB-compatible API)
fn handle_vector_search(body: &str, state: &Arc<RwLock<DatabaseState>>) -> HttpResponse {
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(body);

    match parsed {
        Ok(json) => {
            let query_content = json.get("query")
                .and_then(|v| v.as_str());
            let query_vector = json.get("vector")
                .and_then(|v| v.as_array());
            let k = json.get("k")
                .and_then(|v| v.as_u64())
                .unwrap_or(10) as usize;
            let threshold = json.get("threshold")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5) as f32;

            // Create query fingerprint
            let query_fp = if let Some(content) = query_content {
                Fingerprint::from_content(content)
            } else if let Some(vec) = query_vector {
                let mut data = [0u64; FINGERPRINT_U64];
                for (i, v) in vec.iter().take(FINGERPRINT_U64).enumerate() {
                    data[i] = v.as_u64().unwrap_or(0);
                }
                Fingerprint::from_raw(data)
            } else {
                return HttpResponse::error(400, "Missing 'query' or 'vector' field");
            };

            let mut query_arr = [0u64; FINGERPRINT_WORDS];
            query_arr.copy_from_slice(&query_fp.as_raw()[..FINGERPRINT_WORDS]);

            let db = state.read().unwrap();
            let results = db.substrate.resonate(&query_arr, k);

            // Filter by threshold
            let matches: Vec<serde_json::Value> = results.iter()
                .filter(|(_, sim)| *sim >= threshold)
                .map(|(addr, similarity)| {
                    serde_json::json!({
                        "addr": format!("{:04X}", addr.0),
                        "similarity": similarity,
                        "distance": ((1.0 - similarity) * 10000.0) as u32
                    })
                }).collect();

            let response = serde_json::json!({
                "success": true,
                "matches": matches,
                "count": matches.len(),
                "k": k,
                "threshold": threshold
            });

            HttpResponse::json(&response.to_string())
        }
        Err(e) => HttpResponse::error(400, &format!("Invalid JSON: {}", e)),
    }
}

/// Handle vector insert
fn handle_vector_insert(body: &str, state: &Arc<RwLock<DatabaseState>>) -> HttpResponse {
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(body);

    match parsed {
        Ok(json) => {
            let vectors = json.get("vectors")
                .and_then(|v| v.as_array());

            if vectors.is_none() {
                return HttpResponse::error(400, "Missing 'vectors' array");
            }

            let db = state.read().unwrap();
            let mut inserted = 0;
            let mut addresses = Vec::new();

            for vec_entry in vectors.unwrap() {
                let content = vec_entry.get("content")
                    .and_then(|v| v.as_str());
                let vector = vec_entry.get("vector")
                    .and_then(|v| v.as_array());
                let label = vec_entry.get("label")
                    .and_then(|v| v.as_str());

                let fp = if let Some(c) = content {
                    Fingerprint::from_content(c)
                } else if let Some(v) = vector {
                    let mut data = [0u64; FINGERPRINT_U64];
                    for (i, val) in v.iter().take(FINGERPRINT_U64).enumerate() {
                        data[i] = val.as_u64().unwrap_or(0);
                    }
                    Fingerprint::from_raw(data)
                } else {
                    continue;
                };

                // Create fingerprint array
                let mut fp_arr = [0u64; FINGERPRINT_WORDS];
                fp_arr.copy_from_slice(&fp.as_raw()[..FINGERPRINT_WORDS]);

                // Write with or without label
                let addr = if let Some(lbl) = label {
                    db.substrate.write_labeled(fp_arr, lbl)
                } else {
                    db.substrate.write(fp_arr)
                };

                addresses.push(format!("{:04X}", addr.0));
                inserted += 1;
            }

            let response = serde_json::json!({
                "success": true,
                "inserted": inserted,
                "addresses": addresses
            });

            HttpResponse::json(&response.to_string())
        }
        Err(e) => HttpResponse::error(400, &format!("Invalid JSON: {}", e)),
    }
}

/// Handle CAM operations
fn handle_cam_operation(op_name: &str, body: &str, state: &Arc<RwLock<DatabaseState>>) -> HttpResponse {
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(body);

    match parsed {
        Ok(json) => {
            let args: Vec<String> = json.get("args")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter().filter_map(|v| {
                        v.as_str().map(|s| s.to_string())
                    }).collect()
                })
                .unwrap_or_default();

            // Build CAM command string
            let cmd = if args.is_empty() {
                format!("CAM {}", op_name)
            } else {
                format!("CAM {} {}", op_name, args.join(" "))
            };

            let mut db = state.write().unwrap();
            let result = db.redis_adapter.execute(&cmd);

            let response = match result {
                RedisResult::Ok => serde_json::json!({
                    "success": true,
                    "operation": op_name,
                    "result": "OK"
                }),
                RedisResult::String(s) => serde_json::json!({
                    "success": true,
                    "operation": op_name,
                    "result": s
                }),
                RedisResult::Integer(i) => serde_json::json!({
                    "success": true,
                    "operation": op_name,
                    "result": i
                }),
                RedisResult::Float(f) => serde_json::json!({
                    "success": true,
                    "operation": op_name,
                    "result": f
                }),
                RedisResult::Addr(a) => serde_json::json!({
                    "success": true,
                    "operation": op_name,
                    "result": format!("{:04X}", a.0)
                }),
                RedisResult::Error(e) => serde_json::json!({
                    "success": false,
                    "operation": op_name,
                    "error": e
                }),
                _ => serde_json::json!({
                    "success": true,
                    "operation": op_name,
                    "result": format!("{:?}", result)
                }),
            };

            HttpResponse::json(&response.to_string())
        }
        Err(e) => HttpResponse::error(400, &format!("Invalid JSON: {}", e)),
    }
}

/// Create fingerprint from content
fn handle_fingerprint_create(body: &str) -> HttpResponse {
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(body);

    match parsed {
        Ok(json) => {
            let content = json.get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            let fp = Fingerprint::from_content(content);
            let raw = fp.as_raw();

            let response = serde_json::json!({
                "success": true,
                "content": content,
                "fingerprint": {
                    "popcount": fp.popcount(),
                    "density": fp.density(),
                    "bits": FINGERPRINT_U64 * 64,
                    "hex_preview": format!("{:016x}{:016x}...", raw[0], raw[1])
                }
            });

            HttpResponse::json(&response.to_string())
        }
        Err(e) => HttpResponse::error(400, &format!("Invalid JSON: {}", e)),
    }
}

/// Handle resonate (similarity search)
fn handle_resonate(body: &str, state: &Arc<RwLock<DatabaseState>>) -> HttpResponse {
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(body);

    match parsed {
        Ok(json) => {
            let query = json.get("query")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let k = json.get("k")
                .and_then(|v| v.as_u64())
                .unwrap_or(10) as usize;
            let threshold = json.get("threshold")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32;

            let query_fp = Fingerprint::from_content(query);
            let mut query_arr = [0u64; FINGERPRINT_WORDS];
            query_arr.copy_from_slice(&query_fp.as_raw()[..FINGERPRINT_WORDS]);

            let db = state.read().unwrap();
            let results = db.substrate.resonate(&query_arr, k);

            // Filter by threshold if specified
            let matches: Vec<serde_json::Value> = results.iter()
                .filter(|(_, sim)| *sim >= threshold)
                .map(|(addr, similarity)| {
                    serde_json::json!({
                        "addr": format!("{:04X}", addr.0),
                        "similarity": similarity
                    })
                }).collect();

            let response = serde_json::json!({
                "success": true,
                "query": query,
                "matches": matches,
                "count": matches.len()
            });

            HttpResponse::json(&response.to_string())
        }
        Err(e) => HttpResponse::error(400, &format!("Invalid JSON: {}", e)),
    }
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/// Detect CPU features for display
fn detect_cpu_features() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        let mut features = Vec::new();
        if is_x86_feature_detected!("avx512f") {
            features.push("AVX-512F");
        }
        if is_x86_feature_detected!("avx512vpopcntdq") {
            features.push("AVX-512-VPOPCNTDQ");
        }
        if is_x86_feature_detected!("avx2") {
            features.push("AVX2");
        }
        if is_x86_feature_detected!("sse4.2") {
            features.push("SSE4.2");
        }
        if features.is_empty() {
            "scalar".to_string()
        } else {
            features.join(", ")
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        "scalar (non-x86)".to_string()
    }
}

/// Handle a single client connection
fn handle_client(mut stream: TcpStream, state: Arc<RwLock<DatabaseState>>, config: &ServerConfig) {
    let mut buffer = vec![0u8; 16384]; // 16KB buffer

    stream.set_read_timeout(Some(config.read_timeout)).ok();
    stream.set_write_timeout(Some(config.write_timeout)).ok();

    match stream.read(&mut buffer) {
        Ok(size) if size > 0 => {
            let request_str = String::from_utf8_lossy(&buffer[..size]);

            let response = if let Some(request) = HttpRequest::parse(&request_str) {
                handle_request(&request, &state)
            } else {
                HttpResponse::error(400, "Invalid HTTP request")
            };

            if let Err(e) = stream.write_all(&response.to_bytes()) {
                eprintln!("[ERROR] Failed to write response: {}", e);
            }
        }
        Ok(_) => {}
        Err(e) => {
            if e.kind() != std::io::ErrorKind::WouldBlock {
                eprintln!("[ERROR] Failed to read from client: {}", e);
            }
        }
    }
}

// =============================================================================
// MAIN ENTRY POINT
// =============================================================================

fn main() {
    // Banner
    println!(r#"
  _           _       _
 | |         | |     | |
 | | __ _  __| |_   _| |__  _   _  __ _
 | |/ _` |/ _` | | | | '_ \| | | |/ _` |
 | | (_| | (_| | |_| | |_) | |_| | (_| |
 |_|\__,_|\__,_|\__, |_.__/ \__,_|\__, |
                 __/ |             __/ |
                |___/             |___/
    "#);
    println!("Crystal Lake Cognitive Database v{}", env!("CARGO_PKG_VERSION"));
    println!();

    // CPU features
    println!("[CPU] {}", detect_cpu_features());

    // Configuration
    let config = ServerConfig::from_env();
    let bind_addr = format!("{}:{}", config.host, config.port);

    println!("[Config] Data directory: {}", config.data_dir);
    println!("[Config] Max connections: {}", config.max_connections);
    println!("[Server] Binding to {}", bind_addr);

    // Create data directory
    if let Err(e) = std::fs::create_dir_all(&config.data_dir) {
        eprintln!("[WARN] Could not create data directory: {}", e);
    }

    // Initialize database state
    let state = Arc::new(RwLock::new(DatabaseState::new(&config.data_dir)));

    // Bind listener
    let listener = TcpListener::bind(&bind_addr).expect("Failed to bind to address");

    println!("[Server] Listening on http://{}", bind_addr);
    println!();
    println!("Endpoints:");
    println!("  GET  /health          - Health check with CPU/memory info");
    println!("  GET  /ready           - Kubernetes readiness probe");
    println!("  GET  /live            - Kubernetes liveness probe");
    println!("  GET  /metrics         - Prometheus metrics");
    println!("  GET  /info            - Server information");
    println!("  POST /redis           - Redis-like commands");
    println!("  POST /sql             - SQL queries");
    println!("  POST /cypher          - Cypher graph queries");
    println!("  POST /bind            - BindSpace operations");
    println!("  GET  /bind/:addr      - Get node by address");
    println!("  POST /substrate/write - Write to Substrate");
    println!("  POST /substrate/query - Query Substrate");
    println!("  POST /vectors/search  - Vector similarity search");
    println!("  POST /vectors/insert  - Insert vectors");
    println!("  POST /cam/:operation  - CAM operations");
    println!("  POST /fingerprint     - Create fingerprint");
    println!("  POST /resonate        - Similarity search");
    println!();

    // Graceful shutdown handling
    let running = Arc::new(AtomicBool::new(true));
    let running_clone = Arc::clone(&running);

    // Handle SIGTERM for Docker/Railway
    #[cfg(unix)]
    {
        use std::sync::mpsc::channel;
        let (tx, rx) = channel();

        ctrlc::set_handler(move || {
            println!("\n[Server] Received shutdown signal");
            running_clone.store(false, Ordering::SeqCst);
            tx.send(()).ok();
        }).expect("Error setting signal handler");

        // Spawn shutdown listener
        let state_clone = Arc::clone(&state);
        let shutdown_timeout = config.graceful_shutdown_timeout;
        thread::spawn(move || {
            if rx.recv().is_ok() {
                println!("[Server] Starting graceful shutdown...");

                // Allow time for in-flight requests to complete
                thread::sleep(shutdown_timeout);
                println!("[Server] Shutdown complete");
                std::process::exit(0);
            }
        });
    }

    // Set non-blocking for graceful shutdown checks
    listener.set_nonblocking(true).ok();

    // Main accept loop
    while running.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((stream, _addr)) => {
                let state_clone = Arc::clone(&state);
                let config_clone = ServerConfig::from_env(); // Clone config for thread
                thread::spawn(move || {
                    handle_client(stream, state_clone, &config_clone);
                });
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                // No pending connections, sleep briefly
                thread::sleep(Duration::from_millis(10));
            }
            Err(e) => {
                eprintln!("[ERROR] Failed to accept connection: {}", e);
            }
        }
    }

    println!("[Server] Shutting down...");
}
