//! Ladybug-RS HTTP API Server
//!
//! A cognitive database server exposing multiple query interfaces:
//! - Redis-like commands via /redis endpoint
//! - SQL via /sql endpoint
//! - Cypher via /cypher endpoint
//! - LanceDB-compatible vector search via /vectors endpoint
//! - CAM operations via /cam endpoint
//! - Health check via /health
//!
//! Deployment:
//! - Railway: Binds to 0.0.0.0:8080 when hostname contains "railway"
//! - Claude Backend: Binds to 127.0.0.1:$PORT (default 5000)
//! - Local: Configurable via HOST and PORT env vars

use std::collections::HashMap;
use std::env;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Duration;

use ladybug::core::Fingerprint;
use ladybug::storage::cog_redis::{CogRedis, SetOptions, RedisResult, CamResult};
use ladybug::FINGERPRINT_U64;

/// Server configuration
struct ServerConfig {
    host: String,
    port: u16,
}

impl ServerConfig {
    fn from_env() -> Self {
        let hostname = env::var("HOSTNAME").unwrap_or_default();
        let railway_env = env::var("RAILWAY_ENVIRONMENT").ok();

        // Detect Railway deployment
        let is_railway = hostname.contains("railway")
            || hostname.ends_with(".internal")
            || railway_env.is_some();

        let (default_host, default_port) = if is_railway {
            // Railway requires 0.0.0.0 and prefers 8080
            ("0.0.0.0".to_string(), 8080)
        } else {
            // Local/Claude backend: use localhost
            ("127.0.0.1".to_string(), 5000)
        };

        let host = env::var("HOST").unwrap_or(default_host);
        let port = env::var("PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(default_port);

        ServerConfig { host, port }
    }
}

/// Shared database state
struct DatabaseState {
    cog_redis: CogRedis,
}

impl DatabaseState {
    fn new() -> Self {
        Self {
            cog_redis: CogRedis::new(),
        }
    }
}

/// HTTP Request parsed from raw bytes
struct HttpRequest {
    method: String,
    path: String,
    headers: HashMap<String, String>,
    body: String,
}

impl HttpRequest {
    fn parse(raw: &str) -> Option<Self> {
        let mut lines = raw.lines();
        let first_line = lines.next()?;
        let mut parts = first_line.split_whitespace();

        let method = parts.next()?.to_string();
        let path = parts.next()?.to_string();

        let mut headers = HashMap::new();
        let mut body_start = false;
        let mut body_lines = Vec::new();

        for line in lines {
            if body_start {
                body_lines.push(line);
            } else if line.is_empty() {
                body_start = true;
            } else if let Some((key, value)) = line.split_once(':') {
                headers.insert(
                    key.trim().to_lowercase(),
                    value.trim().to_string(),
                );
            }
        }

        Some(HttpRequest {
            method,
            path,
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
                ("Access-Control-Allow-Methods".to_string(), "GET, POST, OPTIONS".to_string()),
                ("Access-Control-Allow-Headers".to_string(), "Content-Type".to_string()),
            ],
            body: String::new(),
        }
    }

    fn error(status: u16, message: &str) -> Self {
        let status_text = match status {
            400 => "Bad Request",
            404 => "Not Found",
            500 => "Internal Server Error",
            _ => "Error",
        };

        Self {
            status,
            status_text: status_text.to_string(),
            headers: vec![
                ("Content-Type".to_string(), "application/json".to_string()),
                ("Access-Control-Allow-Origin".to_string(), "*".to_string()),
            ],
            body: format!(r#"{{"error": "{}"}}"#, message),
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

/// Handle incoming HTTP request
fn handle_request(
    request: &HttpRequest,
    state: &Arc<RwLock<DatabaseState>>,
) -> HttpResponse {
    // Handle CORS preflight
    if request.method == "OPTIONS" {
        return HttpResponse::ok();
    }

    match (request.method.as_str(), request.path.as_str()) {
        // Health check
        ("GET", "/health") => {
            HttpResponse::json(r#"{"status": "healthy", "service": "ladybug-rs"}"#)
        }

        // Server info
        ("GET", "/") | ("GET", "/info") => {
            let info = serde_json::json!({
                "name": "ladybug-rs",
                "version": env!("CARGO_PKG_VERSION"),
                "description": "Crystal Lake Cognitive Database",
                "endpoints": {
                    "/health": "Health check",
                    "/redis": "Redis-like commands (POST)",
                    "/sql": "SQL queries (POST)",
                    "/cypher": "Cypher graph queries (POST)",
                    "/vectors/search": "Vector similarity search (POST)",
                    "/vectors/insert": "Insert vectors (POST)",
                    "/cam/:op": "CAM operations (POST)",
                    "/fingerprint": "Create fingerprint from content (POST)"
                },
                "features": {
                    "fingerprint_bits": 10000,
                    "cam_operations": 4096,
                    "address_space": "16-bit (8+8 prefix:slot)"
                }
            });
            HttpResponse::json(&info.to_string())
        }

        // Redis-like commands
        ("POST", "/redis") => {
            handle_redis_command(&request.body, state)
        }

        // SQL queries
        ("POST", "/sql") => {
            handle_sql_query(&request.body, state)
        }

        // Cypher queries
        ("POST", "/cypher") => {
            handle_cypher_query(&request.body, state)
        }

        // Vector search (LanceDB-compatible)
        ("POST", "/vectors/search") => {
            handle_vector_search(&request.body, state)
        }

        // Vector insert
        ("POST", "/vectors/insert") => {
            handle_vector_insert(&request.body, state)
        }

        // CAM operations
        ("POST", path) if path.starts_with("/cam/") => {
            let op_name = &path[5..];
            handle_cam_operation(op_name, &request.body, state)
        }

        // Create fingerprint from content
        ("POST", "/fingerprint") => {
            handle_fingerprint_create(&request.body)
        }

        // Not found
        _ => HttpResponse::error(404, "Endpoint not found"),
    }
}

/// Handle Redis-like commands
fn handle_redis_command(body: &str, state: &Arc<RwLock<DatabaseState>>) -> HttpResponse {
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(body);

    match parsed {
        Ok(json) => {
            let command = json.get("command")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            let mut db = state.write().unwrap();
            let result = db.cog_redis.execute_command(command);

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
                RedisResult::Bulk(bytes) => serde_json::json!({
                    "success": true,
                    "result": hex::encode(&bytes)
                }),
                RedisResult::Array(arr) => serde_json::json!({
                    "success": true,
                    "result": format!("{:?}", arr)
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

            // Map SQL to Redis command for now
            // Full SQL support would use DataFusion
            let command = if query.to_uppercase().starts_with("SELECT") {
                "SCAN 0 MATCH * COUNT 100".to_string()
            } else if query.to_uppercase().starts_with("INSERT") {
                "SET placeholder 0.5".to_string()
            } else {
                query.to_string()
            };

            let mut db = state.write().unwrap();
            let result = db.cog_redis.execute_command(&command);

            let (success, message) = match &result {
                RedisResult::Error(e) => (false, e.clone()),
                _ => (true, format!("{:?}", result)),
            };

            let response = serde_json::json!({
                "success": success,
                "query": query,
                "rows": [],
                "message": message
            });

            HttpResponse::json(&response.to_string())
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

            // Map Cypher to cognitive operations
            // Full Cypher support would use the grammar module
            let response = if query.to_uppercase().starts_with("MATCH") {
                // Pattern matching query
                serde_json::json!({
                    "success": true,
                    "query": query,
                    "nodes": [],
                    "relationships": [],
                    "message": "Cypher query processed"
                })
            } else if query.to_uppercase().starts_with("CREATE") {
                // Create node/relationship
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
                // Convert vector to fingerprint (simplified)
                let mut data = [0u64; FINGERPRINT_U64];
                for (i, v) in vec.iter().take(FINGERPRINT_U64).enumerate() {
                    data[i] = v.as_u64().unwrap_or(0);
                }
                Fingerprint::from_raw(data)
            } else {
                return HttpResponse::error(400, "Missing 'query' or 'vector' field");
            };

            let mut db = state.write().unwrap();

            // Use RESONATE command for similarity search
            let threshold_u32 = ((1.0 - threshold) * 10000.0) as u32;
            // Truncate from 157 to 156 u64s for cog_redis
            let raw = query_fp.as_raw();
            let mut pattern = [0u64; 156];
            pattern.copy_from_slice(&raw[..156]);
            let results = db.cog_redis.query_pattern(&pattern, threshold_u32);

            let matches: Vec<serde_json::Value> = results.iter().take(k).map(|edge| {
                // Compute Hamming distance between query and edge fingerprint
                let query_raw = query_fp.as_raw();
                let distance: u32 = query_raw.iter()
                    .zip(edge.fingerprint.iter())
                    .map(|(a, b)| (a ^ b).count_ones())
                    .sum();
                serde_json::json!({
                    "addr": format!("{:04X}", edge.to.0),
                    "distance": distance,
                    "similarity": 1.0 - (distance as f32 / 10000.0)
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

            let mut db = state.write().unwrap();
            let mut inserted = 0;
            let mut addresses = Vec::new();

            for vec_entry in vectors.unwrap() {
                let content = vec_entry.get("content")
                    .and_then(|v| v.as_str());
                let vector = vec_entry.get("vector")
                    .and_then(|v| v.as_array());

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

                let opts = SetOptions::default();
                // Truncate from 157 to 156 u64s for cog_redis
                let raw = fp.as_raw();
                let mut truncated = [0u64; 156];
                truncated.copy_from_slice(&raw[..156]);
                let addr = db.cog_redis.set(truncated, opts);
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
            let args: Vec<Fingerprint> = json.get("args")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter().filter_map(|v| {
                        v.as_str().map(|s| Fingerprint::from_content(s))
                    }).collect()
                })
                .unwrap_or_default();

            let mut db = state.write().unwrap();
            let result = db.cog_redis.execute_cam_named(op_name, &args);

            let response = match result {
                CamResult::Fingerprint(fp) => serde_json::json!({
                    "success": true,
                    "operation": op_name,
                    "result": {
                        "type": "fingerprint",
                        "popcount": fp.popcount(),
                        "density": fp.density()
                    }
                }),
                CamResult::Fingerprints(fps) => serde_json::json!({
                    "success": true,
                    "operation": op_name,
                    "result": {
                        "type": "fingerprints",
                        "count": fps.len()
                    }
                }),
                CamResult::Scalar(v) => serde_json::json!({
                    "success": true,
                    "operation": op_name,
                    "result": {
                        "type": "scalar",
                        "value": v
                    }
                }),
                CamResult::Bool(b) => serde_json::json!({
                    "success": true,
                    "operation": op_name,
                    "result": {
                        "type": "bool",
                        "value": b
                    }
                }),
                CamResult::Addr(a) => serde_json::json!({
                    "success": true,
                    "operation": op_name,
                    "result": {
                        "type": "addr",
                        "value": format!("{:04X}", a.0)
                    }
                }),
                CamResult::Unit => serde_json::json!({
                    "success": true,
                    "operation": op_name,
                    "result": {
                        "type": "unit"
                    }
                }),
                CamResult::Error(e) => serde_json::json!({
                    "success": false,
                    "operation": op_name,
                    "error": e
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

            let response = serde_json::json!({
                "success": true,
                "content": content,
                "fingerprint": {
                    "popcount": fp.popcount(),
                    "density": fp.density(),
                    "hex_preview": format!("{:016x}{:016x}...", fp.as_raw()[0], fp.as_raw()[1])
                }
            });

            HttpResponse::json(&response.to_string())
        }
        Err(e) => HttpResponse::error(400, &format!("Invalid JSON: {}", e)),
    }
}

/// Handle a single client connection
fn handle_client(mut stream: TcpStream, state: Arc<RwLock<DatabaseState>>) {
    let mut buffer = [0u8; 8192];

    stream.set_read_timeout(Some(Duration::from_secs(30))).ok();
    stream.set_write_timeout(Some(Duration::from_secs(30))).ok();

    match stream.read(&mut buffer) {
        Ok(size) if size > 0 => {
            let request_str = String::from_utf8_lossy(&buffer[..size]);

            let response = if let Some(request) = HttpRequest::parse(&request_str) {
                handle_request(&request, &state)
            } else {
                HttpResponse::error(400, "Invalid HTTP request")
            };

            if let Err(e) = stream.write_all(&response.to_bytes()) {
                eprintln!("Failed to write response: {}", e);
            }
        }
        Ok(_) => {}
        Err(e) => {
            eprintln!("Failed to read from client: {}", e);
        }
    }
}

fn main() {
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

    // Print CPU features
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            println!("[CPU] AVX-512 detected - using optimized SIMD");
        } else if is_x86_feature_detected!("avx2") {
            println!("[CPU] AVX2 detected - using SIMD fallback");
        } else {
            println!("[CPU] No AVX detected - using scalar operations");
        }
    }

    let config = ServerConfig::from_env();
    let bind_addr = format!("{}:{}", config.host, config.port);

    println!("[Server] Binding to {}", bind_addr);

    let listener = TcpListener::bind(&bind_addr).expect("Failed to bind to address");

    println!("[Server] Listening on http://{}", bind_addr);
    println!();
    println!("Endpoints:");
    println!("  GET  /health          - Health check");
    println!("  GET  /info            - Server information");
    println!("  POST /redis           - Redis-like commands");
    println!("  POST /sql             - SQL queries");
    println!("  POST /cypher          - Cypher graph queries");
    println!("  POST /vectors/search  - Vector similarity search");
    println!("  POST /vectors/insert  - Insert vectors");
    println!("  POST /cam/:operation  - CAM operations");
    println!("  POST /fingerprint     - Create fingerprint");
    println!();

    let state = Arc::new(RwLock::new(DatabaseState::new()));

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let state_clone = Arc::clone(&state);
                thread::spawn(move || {
                    handle_client(stream, state_clone);
                });
            }
            Err(e) => {
                eprintln!("Failed to accept connection: {}", e);
            }
        }
    }
}
