//! Unified Execution Contract — ladybug-rs adapter.
//!
//! Implements the shared contract between ada-n8n, crewai-rust, and ladybug-rs.
//! Ladybug acts as the **cognitive spine**: it ingests execution steps from
//! the other runtimes, fingerprints them into BindSpace, and provides
//! semantic search and what-if analysis across all execution history.
//!
//! # Architecture
//!
//! ```text
//! ada-n8n ──────┐
//!               │  UnifiedStep (JSON)
//! crewai-rust ──┤──────────────────────► EnrichmentEngine
//!               │                              │
//! lb.* steps ───┘                              ▼
//!                                        BindSpace (fingerprint @ Addr)
//!                                              │
//!                                              ▼
//!                                     UnifiedSpectator.resonate_history()
//! ```
//!
//! # Ingestion Modes
//!
//! 1. **Flight push** (preferred): `DoAction("ingest.unified_step")`
//! 2. **PG poll** (fallback, `postgres` feature): periodic read from `unified_steps`

pub mod enricher;
pub mod enrichment;
pub mod envelope;
pub mod spectator;
pub mod types;

// Re-export canonical types
pub use enricher::EnrichmentEngine;
pub use enrichment::StepEnrichment;
pub use spectator::UnifiedSpectator;
pub use types::{DataEnvelope, EnvelopeMetadata, StepStatus, UnifiedExecution, UnifiedStep};

// Re-export from the contract crate (substrate types)
pub use ladybug_contract as kernel;
