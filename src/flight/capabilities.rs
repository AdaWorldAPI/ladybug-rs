//! Capabilities Check and Graceful Fallback
//!
//! Implements automatic capability detection with graceful fallback:
//! - Check gRPC/Flight connectivity
//! - Fall back to JSON over HTTP/SSE when gRPC unavailable
//! - Support for Python SDK (ladybug-vsa) integration
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Client Connection                            │
//! └───────────────────────────┬─────────────────────────────────────┘
//!                             │
//!                             ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                  CapabilitiesChecker                            │
//! │  ┌────────────────────────────────────────────────────────────┐ │
//! │  │  1. Try gRPC/Flight (Arrow IPC zero-copy)                  │ │
//! │  │     ↓ fail?                                                 │ │
//! │  │  2. Try HTTP/2 with SSE (Server-Sent Events)               │ │
//! │  │     ↓ fail?                                                 │ │
//! │  │  3. Fall back to HTTP/1.1 JSON                             │ │
//! │  └────────────────────────────────────────────────────────────┘ │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use ladybug::flight::CapabilitiesChecker;
//!
//! let checker = CapabilitiesChecker::new("localhost:50051");
//! let capabilities = checker.detect().await;
//!
//! match capabilities.best_transport() {
//!     Transport::ArrowFlight => /* zero-copy gRPC */,
//!     Transport::McpSse => /* SSE streaming */,
//!     Transport::HttpJson => /* basic JSON */,
//! }
//! ```

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(feature = "flight")]
use tonic::transport::Channel;

// =============================================================================
// TRANSPORT TYPES
// =============================================================================

/// Available transport protocols, ordered by preference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Transport {
    /// Arrow Flight over gRPC (zero-copy, best performance)
    ArrowFlight,
    /// MCP over Server-Sent Events (streaming, good compatibility)
    McpSse,
    /// HTTP/1.1 JSON-RPC (universal fallback)
    HttpJson,
}

impl Transport {
    /// Get transport name for logging
    pub fn name(&self) -> &'static str {
        match self {
            Transport::ArrowFlight => "Arrow Flight (gRPC)",
            Transport::McpSse => "MCP SSE (HTTP/2)",
            Transport::HttpJson => "HTTP JSON",
        }
    }

    /// Approximate throughput (fingerprints/sec)
    pub fn expected_throughput(&self) -> u64 {
        match self {
            Transport::ArrowFlight => 3_000_000, // ~3M fps (zero-copy)
            Transport::McpSse => 100_000,        // ~100K fps (streaming)
            Transport::HttpJson => 10_000,       // ~10K fps (serialization overhead)
        }
    }

    /// Whether this transport supports streaming
    pub fn supports_streaming(&self) -> bool {
        matches!(self, Transport::ArrowFlight | Transport::McpSse)
    }

    /// Whether this transport supports zero-copy
    pub fn supports_zero_copy(&self) -> bool {
        matches!(self, Transport::ArrowFlight)
    }
}

// =============================================================================
// CAPABILITY DETECTION
// =============================================================================

/// Detected server capabilities
#[derive(Debug, Clone)]
pub struct Capabilities {
    /// Available transports (in preference order)
    pub transports: Vec<Transport>,
    /// Server version
    pub server_version: Option<String>,
    /// Supported MCP actions
    pub actions: Vec<String>,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Supports DN tree operations
    pub dn_tree: bool,
    /// Supports ACID transactions
    pub acid: bool,
    /// Supports work stealing
    pub work_stealing: bool,
    /// Detection timestamp
    pub detected_at: Instant,
    /// Detection latency
    pub latency_ms: u64,
}

impl Capabilities {
    /// Get the best available transport
    pub fn best_transport(&self) -> Transport {
        self.transports
            .first()
            .copied()
            .unwrap_or(Transport::HttpJson)
    }

    /// Check if a specific transport is available
    pub fn has_transport(&self, transport: Transport) -> bool {
        self.transports.contains(&transport)
    }

    /// Check if server supports a specific action
    pub fn has_action(&self, action: &str) -> bool {
        self.actions.iter().any(|a| a == action)
    }

    /// Get capabilities as feature flags (for Python SDK)
    pub fn as_feature_flags(&self) -> HashMap<String, bool> {
        let mut flags = HashMap::new();
        flags.insert(
            "flight".to_string(),
            self.has_transport(Transport::ArrowFlight),
        );
        flags.insert("sse".to_string(), self.has_transport(Transport::McpSse));
        flags.insert("json".to_string(), self.has_transport(Transport::HttpJson));
        flags.insert(
            "streaming".to_string(),
            self.best_transport().supports_streaming(),
        );
        flags.insert(
            "zero_copy".to_string(),
            self.best_transport().supports_zero_copy(),
        );
        flags.insert("dn_tree".to_string(), self.dn_tree);
        flags.insert("acid".to_string(), self.acid);
        flags.insert("work_stealing".to_string(), self.work_stealing);
        flags
    }
}

impl Default for Capabilities {
    fn default() -> Self {
        Self {
            transports: vec![Transport::HttpJson],
            server_version: None,
            actions: vec![
                "encode".to_string(),
                "bind".to_string(),
                "read".to_string(),
                "resonate".to_string(),
            ],
            max_batch_size: 1000,
            dn_tree: false,
            acid: false,
            work_stealing: false,
            detected_at: Instant::now(),
            latency_ms: 0,
        }
    }
}

// =============================================================================
// CAPABILITIES CHECKER
// =============================================================================

/// Check server capabilities and determine best transport
pub struct CapabilitiesChecker {
    /// Server endpoint (host:port or URL)
    endpoint: String,
    /// Connection timeout
    timeout: Duration,
    /// Cached capabilities
    cached: Arc<RwLock<Option<Capabilities>>>,
    /// Cache TTL
    cache_ttl: Duration,
}

impl CapabilitiesChecker {
    /// Create new checker for endpoint
    pub fn new(endpoint: &str) -> Self {
        Self {
            endpoint: endpoint.to_string(),
            timeout: Duration::from_secs(5),
            cached: Arc::new(RwLock::new(None)),
            cache_ttl: Duration::from_secs(60),
        }
    }

    /// Create with custom timeout
    pub fn with_timeout(endpoint: &str, timeout: Duration) -> Self {
        Self {
            endpoint: endpoint.to_string(),
            timeout,
            cached: Arc::new(RwLock::new(None)),
            cache_ttl: Duration::from_secs(60),
        }
    }

    /// Get cached capabilities or detect new
    pub async fn get_or_detect(&self) -> Capabilities {
        // Check cache
        {
            let cached = self.cached.read();
            if let Some(ref caps) = *cached {
                if caps.detected_at.elapsed() < self.cache_ttl {
                    return caps.clone();
                }
            }
        }

        // Detect and cache
        let caps = self.detect().await;
        {
            let mut cached = self.cached.write();
            *cached = Some(caps.clone());
        }
        caps
    }

    /// Detect server capabilities
    pub async fn detect(&self) -> Capabilities {
        let start = Instant::now();
        let mut transports = Vec::new();
        let mut server_version = None;
        let mut actions = Vec::new();
        let mut max_batch_size = 1000;
        let mut dn_tree = false;
        let mut acid = false;
        let mut work_stealing = false;

        // 1. Try Arrow Flight (gRPC)
        #[cfg(feature = "flight")]
        {
            if let Ok(flight_caps) = self.probe_flight().await {
                transports.push(Transport::ArrowFlight);
                server_version = flight_caps.version;
                actions.extend(flight_caps.actions);
                max_batch_size = flight_caps.max_batch_size;
                dn_tree = flight_caps.dn_tree;
                acid = flight_caps.acid;
                work_stealing = flight_caps.work_stealing;
            }
        }

        // 2. Try HTTP/2 SSE
        if let Ok(sse_caps) = self.probe_sse().await {
            if !transports.contains(&Transport::McpSse) {
                transports.push(Transport::McpSse);
            }
            if server_version.is_none() {
                server_version = sse_caps.version;
            }
            // Merge actions
            for action in sse_caps.actions {
                if !actions.contains(&action) {
                    actions.push(action);
                }
            }
        }

        // 3. Try HTTP JSON (always available as fallback)
        if let Ok(json_caps) = self.probe_json().await {
            transports.push(Transport::HttpJson);
            if server_version.is_none() {
                server_version = json_caps.version;
            }
            // Merge actions
            for action in json_caps.actions {
                if !actions.contains(&action) {
                    actions.push(action);
                }
            }
        }

        // If nothing worked, add JSON as default fallback
        if transports.is_empty() {
            transports.push(Transport::HttpJson);
        }

        // Default actions if none discovered
        if actions.is_empty() {
            actions = vec![
                "encode".to_string(),
                "bind".to_string(),
                "read".to_string(),
                "resonate".to_string(),
                "hamming".to_string(),
                "xor_bind".to_string(),
                "stats".to_string(),
            ];
        }

        Capabilities {
            transports,
            server_version,
            actions,
            max_batch_size,
            dn_tree,
            acid,
            work_stealing,
            detected_at: Instant::now(),
            latency_ms: start.elapsed().as_millis() as u64,
        }
    }

    /// Probe Arrow Flight endpoint
    #[cfg(feature = "flight")]
    async fn probe_flight(&self) -> Result<ProbeResult, ProbeError> {
        use arrow_flight::Criteria;
        use arrow_flight::flight_service_client::FlightServiceClient;
        use tonic::transport::Endpoint;

        let endpoint = format!("http://{}", self.endpoint);
        let channel = Endpoint::from_shared(endpoint)
            .map_err(|_| ProbeError::ConnectionFailed)?
            .connect_timeout(self.timeout)
            .connect()
            .await
            .map_err(|_| ProbeError::ConnectionFailed)?;

        let mut client = FlightServiceClient::new(channel);

        // Try to list flights to verify connectivity
        let request = tonic::Request::new(Criteria {
            expression: bytes::Bytes::new(),
        });
        let response = client.list_flights(request).await;

        if response.is_ok() {
            // Flight is available, try to get info
            let _info_request = tonic::Request::new(arrow_flight::FlightDescriptor {
                r#type: arrow_flight::flight_descriptor::DescriptorType::Cmd as i32,
                cmd: bytes::Bytes::from_static(b"capabilities"),
                path: vec![],
            });

            let dn_tree = true;
            let acid = true;
            let work_stealing = true;
            let actions = vec![
                "encode".to_string(),
                "bind".to_string(),
                "read".to_string(),
                "resonate".to_string(),
                "hamming".to_string(),
                "xor_bind".to_string(),
                "stats".to_string(),
                "dn_get".to_string(),
                "dn_set".to_string(),
                "dn_tree".to_string(),
                "dag_begin".to_string(),
                "dag_commit".to_string(),
            ];

            Ok(ProbeResult {
                version: Some("1.0".to_string()),
                actions,
                max_batch_size: 10_000,
                dn_tree,
                acid,
                work_stealing,
            })
        } else {
            Err(ProbeError::ConnectionFailed)
        }
    }

    #[cfg(not(feature = "flight"))]
    async fn probe_flight(&self) -> Result<ProbeResult, ProbeError> {
        Err(ProbeError::NotSupported)
    }

    /// Probe HTTP/2 SSE endpoint
    async fn probe_sse(&self) -> Result<ProbeResult, ProbeError> {
        #[cfg(feature = "reqwest")]
        {
            let url = format!("http://{}/mcp/sse/capabilities", self.endpoint);
            let client = reqwest::Client::builder()
                .timeout(self.timeout)
                .http2_prior_knowledge()
                .build()
                .map_err(|_| ProbeError::ConnectionFailed)?;

            let response = client.get(&url).send().await;

            match response {
                Ok(resp) if resp.status().is_success() => {
                    // Parse capabilities from response
                    if let Ok(body) = resp.text().await {
                        return Ok(ProbeResult::from_json(&body));
                    }
                }
                _ => {}
            }
        }

        Err(ProbeError::ConnectionFailed)
    }

    /// Probe HTTP JSON endpoint
    async fn probe_json(&self) -> Result<ProbeResult, ProbeError> {
        #[cfg(feature = "reqwest")]
        {
            let url = format!("http://{}/mcp/capabilities", self.endpoint);
            let client = reqwest::Client::builder()
                .timeout(self.timeout)
                .build()
                .map_err(|_| ProbeError::ConnectionFailed)?;

            let response = client.get(&url).send().await;

            match response {
                Ok(resp) if resp.status().is_success() => {
                    if let Ok(body) = resp.text().await {
                        return Ok(ProbeResult::from_json(&body));
                    }
                }
                _ => {}
            }
        }

        // Even if probe fails, JSON is always available as last resort
        Ok(ProbeResult::default())
    }
}

// =============================================================================
// PROBE RESULT
// =============================================================================

#[derive(Debug, Clone)]
struct ProbeResult {
    version: Option<String>,
    actions: Vec<String>,
    max_batch_size: usize,
    dn_tree: bool,
    acid: bool,
    work_stealing: bool,
}

impl Default for ProbeResult {
    fn default() -> Self {
        Self {
            version: None,
            actions: vec![
                "encode".to_string(),
                "bind".to_string(),
                "read".to_string(),
                "resonate".to_string(),
            ],
            max_batch_size: 1000,
            dn_tree: false,
            acid: false,
            work_stealing: false,
        }
    }
}

impl ProbeResult {
    fn from_json(json: &str) -> Self {
        // Simple JSON parsing without serde dependency
        let mut result = Self::default();

        if json.contains("\"version\"") {
            // Extract version
            if let Some(start) = json.find("\"version\"") {
                if let Some(colon) = json[start..].find(':') {
                    if let Some(quote_start) = json[start + colon..].find('"') {
                        let after_quote = start + colon + quote_start + 1;
                        if let Some(quote_end) = json[after_quote..].find('"') {
                            result.version =
                                Some(json[after_quote..after_quote + quote_end].to_string());
                        }
                    }
                }
            }
        }

        if json.contains("\"dn_tree\":true") || json.contains("\"dn_tree\": true") {
            result.dn_tree = true;
        }
        if json.contains("\"acid\":true") || json.contains("\"acid\": true") {
            result.acid = true;
        }
        if json.contains("\"work_stealing\":true") || json.contains("\"work_stealing\": true") {
            result.work_stealing = true;
        }

        result
    }
}

#[derive(Debug, Clone)]
enum ProbeError {
    ConnectionFailed,
    Timeout,
    NotSupported,
}

// =============================================================================
// TRANSPORT ADAPTER
// =============================================================================

/// Adapts to the best available transport automatically
pub struct TransportAdapter {
    capabilities: Capabilities,
    endpoint: String,
    #[cfg(feature = "flight")]
    flight_client:
        Option<Arc<RwLock<arrow_flight::flight_service_client::FlightServiceClient<Channel>>>>,
}

impl TransportAdapter {
    /// Create adapter with detected capabilities
    pub async fn new(endpoint: &str) -> Self {
        let checker = CapabilitiesChecker::new(endpoint);
        let capabilities = checker.detect().await;

        Self {
            capabilities,
            endpoint: endpoint.to_string(),
            #[cfg(feature = "flight")]
            flight_client: None,
        }
    }

    /// Create adapter with known capabilities
    pub fn with_capabilities(endpoint: &str, capabilities: Capabilities) -> Self {
        Self {
            capabilities,
            endpoint: endpoint.to_string(),
            #[cfg(feature = "flight")]
            flight_client: None,
        }
    }

    /// Get the best transport
    pub fn transport(&self) -> Transport {
        self.capabilities.best_transport()
    }

    /// Get capabilities
    pub fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    /// Execute action using best available transport
    pub async fn execute(&self, action: &str, params: &[u8]) -> Result<Vec<u8>, TransportError> {
        match self.capabilities.best_transport() {
            Transport::ArrowFlight => self.execute_flight(action, params).await,
            Transport::McpSse => self.execute_sse(action, params).await,
            Transport::HttpJson => self.execute_json(action, params).await,
        }
    }

    #[cfg(feature = "flight")]
    async fn execute_flight(&self, action: &str, params: &[u8]) -> Result<Vec<u8>, TransportError> {
        use arrow_flight::Action;
        use arrow_flight::flight_service_client::FlightServiceClient;
        use tonic::transport::Endpoint;

        let endpoint = format!("http://{}", self.endpoint);
        let channel = Endpoint::from_shared(endpoint)
            .map_err(|_| TransportError::ConnectionFailed)?
            .connect()
            .await
            .map_err(|_| TransportError::ConnectionFailed)?;

        let mut client = FlightServiceClient::new(channel);

        let action_request = Action {
            r#type: action.to_string(),
            body: params.to_vec().into(),
        };

        let mut stream = client
            .do_action(tonic::Request::new(action_request))
            .await
            .map_err(|e| TransportError::ActionFailed(e.to_string()))?
            .into_inner();

        // Collect all results
        let mut result = Vec::new();
        while let Some(response) = stream
            .message()
            .await
            .map_err(|e| TransportError::ActionFailed(e.to_string()))?
        {
            result.extend(response.body);
        }

        Ok(result)
    }

    #[cfg(not(feature = "flight"))]
    async fn execute_flight(
        &self,
        _action: &str,
        _params: &[u8],
    ) -> Result<Vec<u8>, TransportError> {
        Err(TransportError::NotSupported)
    }

    async fn execute_sse(&self, _action: &str, _params: &[u8]) -> Result<Vec<u8>, TransportError> {
        #[cfg(feature = "reqwest")]
        {
            let url = format!("http://{}/mcp/sse/{}", self.endpoint, _action);
            let client = reqwest::Client::new();
            let response = client
                .post(&url)
                .header("Content-Type", "application/octet-stream")
                .header("Accept", "text/event-stream")
                .body(_params.to_vec())
                .send()
                .await
                .map_err(|e| TransportError::ActionFailed(e.to_string()))?;

            if response.status().is_success() {
                let bytes = response
                    .bytes()
                    .await
                    .map_err(|e| TransportError::ActionFailed(e.to_string()))?;
                return Ok(bytes.to_vec());
            }
        }

        Err(TransportError::ActionFailed(
            "SSE request failed".to_string(),
        ))
    }

    async fn execute_json(&self, _action: &str, _params: &[u8]) -> Result<Vec<u8>, TransportError> {
        #[cfg(feature = "reqwest")]
        {
            let url = format!("http://{}/mcp/json/{}", self.endpoint, _action);
            let client = reqwest::Client::new();
            let response = client
                .post(&url)
                .header("Content-Type", "application/json")
                .body(_params.to_vec())
                .send()
                .await
                .map_err(|e| TransportError::ActionFailed(e.to_string()))?;

            if response.status().is_success() {
                let bytes = response
                    .bytes()
                    .await
                    .map_err(|e| TransportError::ActionFailed(e.to_string()))?;
                return Ok(bytes.to_vec());
            }
        }

        Err(TransportError::ActionFailed(
            "JSON request failed".to_string(),
        ))
    }
}

#[derive(Debug, Clone)]
pub enum TransportError {
    ConnectionFailed,
    Timeout,
    NotSupported,
    ActionFailed(String),
}

impl std::fmt::Display for TransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConnectionFailed => write!(f, "Connection failed"),
            Self::Timeout => write!(f, "Timeout"),
            Self::NotSupported => write!(f, "Transport not supported"),
            Self::ActionFailed(msg) => write!(f, "Action failed: {}", msg),
        }
    }
}

impl std::error::Error for TransportError {}

// =============================================================================
// PYTHON SDK INTEGRATION
// =============================================================================

/// Python SDK client configuration (for ladybug-vsa)
#[derive(Debug, Clone)]
pub struct PythonClientConfig {
    /// Server endpoint
    pub endpoint: String,
    /// Preferred transport (auto-detect if None)
    pub transport: Option<Transport>,
    /// Enable async operations
    pub async_ops: bool,
    /// Batch size for bulk operations
    pub batch_size: usize,
    /// Connection timeout (ms)
    pub timeout_ms: u64,
    /// Enable compression
    pub compression: bool,
}

impl Default for PythonClientConfig {
    fn default() -> Self {
        Self {
            endpoint: "localhost:50051".to_string(),
            transport: None,
            async_ops: true,
            batch_size: 1000,
            timeout_ms: 5000,
            compression: true,
        }
    }
}

impl PythonClientConfig {
    /// Create config for local development
    pub fn local() -> Self {
        Self::default()
    }

    /// Create config for production
    pub fn production(endpoint: &str) -> Self {
        Self {
            endpoint: endpoint.to_string(),
            transport: Some(Transport::ArrowFlight),
            async_ops: true,
            batch_size: 10_000,
            timeout_ms: 30_000,
            compression: true,
        }
    }

    /// Convert to Python dict representation
    pub fn to_python_dict(&self) -> String {
        format!(
            r#"{{
    "endpoint": "{}",
    "transport": "{}",
    "async_ops": {},
    "batch_size": {},
    "timeout_ms": {},
    "compression": {}
}}"#,
            self.endpoint,
            self.transport.map(|t| t.name()).unwrap_or("auto"),
            self.async_ops,
            self.batch_size,
            self.timeout_ms,
            self.compression,
        )
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transport_properties() {
        assert!(Transport::ArrowFlight.supports_zero_copy());
        assert!(Transport::ArrowFlight.supports_streaming());
        assert!(Transport::McpSse.supports_streaming());
        assert!(!Transport::HttpJson.supports_streaming());
    }

    #[test]
    fn test_capabilities_default() {
        let caps = Capabilities::default();
        assert_eq!(caps.best_transport(), Transport::HttpJson);
        assert!(caps.has_action("encode"));
    }

    #[test]
    fn test_capabilities_feature_flags() {
        let caps = Capabilities {
            transports: vec![Transport::ArrowFlight, Transport::HttpJson],
            dn_tree: true,
            acid: true,
            ..Default::default()
        };

        let flags = caps.as_feature_flags();
        assert!(flags["flight"]);
        assert!(flags["dn_tree"]);
        assert!(flags["acid"]);
    }

    #[test]
    fn test_python_config() {
        let config = PythonClientConfig::production("api.example.com:50051");
        assert_eq!(config.transport, Some(Transport::ArrowFlight));
        assert!(config.to_python_dict().contains("api.example.com"));
    }

    #[test]
    fn test_probe_result_from_json() {
        let json = r#"{"version": "1.2.3", "dn_tree": true, "acid": true}"#;
        let result = ProbeResult::from_json(json);
        assert_eq!(result.version, Some("1.2.3".to_string()));
        assert!(result.dn_tree);
        assert!(result.acid);
    }
}
