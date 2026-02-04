//! Arrow Flight MCP Server
//!
//! High-performance MCP (Model Context Protocol) connector using Arrow Flight.
//! Zero-copy transfer of fingerprints and streaming search results.
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
//! # Performance
//!
//! | Protocol | Fingerprint Transfer | Bulk Search |
//! |----------|---------------------|-------------|
//! | JSON-RPC | ~100 KB/s (base64)  | Full response |
//! | Flight   | ~3 GB/s (zero-copy) | Streaming    |

#[cfg(feature = "flight")]
mod server;
#[cfg(feature = "flight")]
mod actions;

#[cfg(feature = "flight")]
pub use server::LadybugFlightService;
#[cfg(feature = "flight")]
pub use actions::execute_action;

// Legacy JSON types are only available with json_fallback feature
#[cfg(all(feature = "flight", feature = "json_fallback"))]
pub use actions::json_fallback::{McpAction, McpResult};
