//! Enrichment engine — converts UnifiedSteps into BindSpace fingerprints.
//!
//! Two ingestion modes use this engine:
//! - **Flight push**: `ingest.unified_step` DoAction calls `enrich_step()` directly
//! - **PG poll**: background worker reads unenriched steps, calls `enrich_batch()`

use std::sync::Arc;

use chrono::Utc;
use parking_lot::RwLock;

use crate::storage::bind_space::{Addr, BindSpace};

use super::enrichment::{StepEnrichment, compute_qidx, extract_thinking_style, hash_step_id};
use super::types::{DataEnvelope, EnvelopeMetadata, UnifiedStep};

/// Core enrichment engine — stateless, operates on BindSpace.
pub struct EnrichmentEngine {
    bind_space: Arc<RwLock<BindSpace>>,
}

impl EnrichmentEngine {
    pub fn new(bind_space: Arc<RwLock<BindSpace>>) -> Self {
        Self { bind_space }
    }

    /// Enrich a single step: fingerprint its output and write to BindSpace.
    ///
    /// Returns the enrichment metadata.
    pub fn enrich_step(&self, step: &UnifiedStep) -> StepEnrichment {
        // Convert step output to DataEnvelope
        let envelope = DataEnvelope {
            step_id: step.step_id.clone(),
            output_key: format!("{}.output", step.step_id),
            content_type: "application/json".to_string(),
            content: step.output.clone(),
            metadata: EnvelopeMetadata::default(),
        };

        // Compute fingerprint from envelope content
        let fingerprint = envelope.to_fingerprint();

        // Determine BindSpace address based on runtime namespace
        let prefix = runtime_prefix(&step.step_type);
        let slot = hash_step_id(&step.step_id);
        let addr = Addr::new(prefix, slot);

        // Write to BindSpace
        {
            let mut space = self.bind_space.write();
            space.write_at(addr, fingerprint);
        }

        StepEnrichment {
            step_id: step.step_id.clone(),
            fingerprint,
            qidx: compute_qidx(&fingerprint),
            thinking_style: extract_thinking_style(step),
            bind_addr: addr,
            enriched_at: Utc::now(),
        }
    }

    /// Enrich a batch of steps. Returns enrichments for each.
    pub fn enrich_batch(&self, steps: &[UnifiedStep]) -> Vec<StepEnrichment> {
        steps.iter().map(|s| self.enrich_step(s)).collect()
    }
}

/// Map step_type namespace prefix to a BindSpace address prefix.
///
/// - `crew.*` → 0x0E (near surface, agent zone)
/// - `n8n.*`  → 0x10 (fluid zone start)
/// - `lb.*`   → 0x00 (surface, Lance compartment)
/// - other    → 0x10 (fluid zone default)
fn runtime_prefix(step_type: &str) -> u8 {
    match step_type.split('.').next() {
        Some("crew") => 0x0E,
        Some("n8n") => 0x10,
        Some("lb") => 0x00,
        _ => 0x10,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::bind_space::BindSpace;
    use chrono::Utc;
    use serde_json::json;

    use super::super::types::StepStatus;

    fn make_step(step_type: &str, output: serde_json::Value) -> UnifiedStep {
        UnifiedStep {
            step_id: uuid::Uuid::new_v4().to_string(),
            execution_id: "exec-1".to_string(),
            step_type: step_type.to_string(),
            runtime: step_type.split('.').next().unwrap_or("unknown").to_string(),
            name: "test step".to_string(),
            status: StepStatus::Completed,
            input: json!({}),
            output,
            error: None,
            started_at: Utc::now(),
            finished_at: Some(Utc::now()),
            sequence: 0,
            reasoning: None,
            confidence: None,
            alternatives: None,
        }
    }

    #[test]
    fn test_enrich_crew_step() {
        let bs = Arc::new(RwLock::new(BindSpace::new()));
        let engine = EnrichmentEngine::new(bs.clone());

        let step = make_step("crew.agent", json!({"result": "Agent completed analysis"}));
        let enrichment = engine.enrich_step(&step);

        assert_eq!(
            enrichment.bind_addr.prefix(),
            0x0E,
            "crew steps → prefix 0x0E"
        );
        assert!(
            enrichment.fingerprint.iter().any(|&w| w != 0),
            "Fingerprint must be non-zero"
        );
    }

    #[test]
    fn test_enrich_n8n_step() {
        let bs = Arc::new(RwLock::new(BindSpace::new()));
        let engine = EnrichmentEngine::new(bs.clone());

        let step = make_step("n8n.httpRequest", json!({"url": "https://api.example.com"}));
        let enrichment = engine.enrich_step(&step);

        assert_eq!(
            enrichment.bind_addr.prefix(),
            0x10,
            "n8n steps → prefix 0x10"
        );
    }

    #[test]
    fn test_enrich_lb_step() {
        let bs = Arc::new(RwLock::new(BindSpace::new()));
        let engine = EnrichmentEngine::new(bs.clone());

        let step = make_step("lb.resonate", json!({"matches": 5}));
        let enrichment = engine.enrich_step(&step);

        assert_eq!(
            enrichment.bind_addr.prefix(),
            0x00,
            "lb steps → prefix 0x00"
        );
    }

    #[test]
    fn test_enrich_batch() {
        let bs = Arc::new(RwLock::new(BindSpace::new()));
        let engine = EnrichmentEngine::new(bs.clone());

        let steps = vec![
            make_step("crew.agent", json!({"a": 1})),
            make_step("n8n.slack", json!({"b": 2})),
            make_step("lb.collapse", json!({"c": 3})),
        ];
        let enrichments = engine.enrich_batch(&steps);
        assert_eq!(enrichments.len(), 3);

        // Verify prefix assignment
        assert_eq!(enrichments[0].bind_addr.prefix(), 0x0E);
        assert_eq!(enrichments[1].bind_addr.prefix(), 0x10);
        assert_eq!(enrichments[2].bind_addr.prefix(), 0x00);
    }
}
