//! LadybugDB Arrow Flight gRPC Server
//!
//! High-performance zero-copy Arrow Flight server for cognitive database operations.
//!
//! # Usage
//!
//! ```bash
//! # Start on default port 50051
//! ladybug-flight
//!
//! # Custom port
//! LADYBUG_FLIGHT_PORT=50052 ladybug-flight
//! ```
//!
//! # Endpoints
//!
//! - **DoGet**: Stream fingerprints from BindSpace (all/surface/fluid/nodes/search)
//! - **DoPut**: Ingest fingerprints (zero-copy)
//! - **DoAction**: Execute MCP tools (encode, bind, resonate, hamming, etc.)
//! - **GetFlightInfo**: Schema discovery for fingerprints

use std::env;
use std::net::SocketAddr;
use std::sync::Arc;

use parking_lot::RwLock;
use tonic::transport::Server;

use ladybug::VERSION;
use ladybug::flight::LadybugFlightService;
use ladybug::search::HdrIndex;
use ladybug::storage::BindSpace;

use arrow_flight::flight_service_server::FlightServiceServer;

// =============================================================================
// CONFIGURATION
// =============================================================================

struct FlightConfig {
    host: String,
    port: u16,
}

impl FlightConfig {
    fn from_env() -> Self {
        let host = env::var("LADYBUG_FLIGHT_HOST")
            .or_else(|_| env::var("HOST"))
            .unwrap_or_else(|_| "0.0.0.0".to_string());

        let port = env::var("LADYBUG_FLIGHT_PORT")
            .or_else(|_| env::var("FLIGHT_PORT"))
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(50051);

        Self { host, port }
    }
}

// =============================================================================
// MAIN
// =============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    let config = FlightConfig::from_env();
    let addr: SocketAddr = format!("{}:{}", config.host, config.port).parse()?;

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║         LadybugDB Arrow Flight Server v{:<23}║", VERSION);
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║  Protocol:    Arrow Flight (gRPC)                             ║");
    println!("║  Binding:     {:>45}  ║", addr);
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║  Endpoints:                                                   ║");
    println!("║  • DoGet      → Stream fingerprints (zero-copy)               ║");
    println!("║  • DoPut      → Ingest fingerprints (zero-copy)               ║");
    println!("║  • DoAction   → MCP tools (encode, bind, resonate)            ║");
    println!("║  • GetInfo    → Schema discovery                              ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");

    // Initialize BindSpace and HDR index
    let bind_space = Arc::new(RwLock::new(BindSpace::new()));
    let hdr_index = Arc::new(RwLock::new(HdrIndex::new()));

    // Create Flight service
    let service = LadybugFlightService::new(Arc::clone(&bind_space), Arc::clone(&hdr_index));

    println!("Starting Arrow Flight server on {}", addr);

    // Start the gRPC server
    Server::builder()
        .add_service(FlightServiceServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
