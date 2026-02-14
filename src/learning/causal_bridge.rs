//! Grammar → Causal Bridge
//!
//! Connects the Grammar Triangle (universal input layer) to the CausalEngine
//! (Pearl's do-calculus). This is the pipe that transforms textual meaning
//! into causal structure.
//!
//! # Architecture
//!
//! ```text
//! Text ──→ GrammarTriangle ──→ CausalBridge ──→ CausalEngine
//!           │                    │                 │
//!           ├─ NSM primes        ├─ to_edge()      ├─ store_intervention()
//!           ├─ CausalityFlow     ├─ to_state()     ├─ store_counterfactual()
//!           └─ QualiaField       └─ ingest()       └─ query_do()
//! ```
//!
//! # Mapping Rules
//!
//! | DependencyType   | EdgeType / Rung           | CausalEdgeType |
//! |------------------|---------------------------|----------------|
//! | Causal           | Do (Rung 2)               | Causes         |
//! | Intentional      | Do (Rung 2)               | Causes         |
//! | Enabling         | Do (Rung 2)               | MayCause       |
//! | Preventing       | Imagine (Rung 3)          | Causes (neg)   |
//! | Correlational    | See (Rung 1)              | Correlated     |
//! | Constitutive     | See (Rung 1)              | Correlated     |
//! | None             | See (Rung 1)              | Correlated     |

use crate::core::Fingerprint;
use crate::grammar::{CausalityFlow, DependencyType, GrammarTriangle};
use crate::search::causal::EdgeType;

use super::causal_ops::{CausalEdgeType, CausalEngine};

// =============================================================================
// FINGERPRINT CONVERSION HELPERS
// =============================================================================

const WORDS: usize = 256;

/// Convert a Fingerprint to the raw [u64; 256] array used by CausalSearch.
#[inline]
fn fp_to_words(fp: &Fingerprint) -> [u64; WORDS] {
    *fp.as_raw()
}

/// Create a deterministic fingerprint from a string label.
fn label_to_fp(label: &str) -> [u64; WORDS] {
    fp_to_words(&Fingerprint::from_content(label))
}

// =============================================================================
// CAUSAL BRIDGE
// =============================================================================

/// A causal edge extracted from grammar analysis, ready for CausalEngine ingestion.
#[derive(Debug, Clone)]
pub struct GrammarCausalEdge {
    /// State fingerprint (context / agent).
    pub state: [u64; WORDS],
    /// Action fingerprint (what was done / the verb).
    pub action: [u64; WORDS],
    /// Outcome fingerprint (patient / result).
    pub outcome: [u64; WORDS],
    /// Which rung of Pearl's ladder this edge belongs to.
    pub edge_type: EdgeType,
    /// Causal graph edge classification.
    pub causal_type: CausalEdgeType,
    /// Strength of the causal link (0.0–1.0).
    pub strength: f32,
    /// Temporal direction (-1 past, 0 present, +1 future).
    pub temporality: f32,
}

/// The Grammar → CausalSearch bridge.
///
/// Converts Grammar Triangle analysis into causal edges and ingests them
/// into a CausalEngine for do-calculus reasoning.
pub struct CausalBridge {
    engine: CausalEngine,
    /// Number of edges ingested so far.
    edge_count: usize,
}

impl CausalBridge {
    /// Create a new bridge with a fresh CausalEngine.
    pub fn new() -> Self {
        Self {
            engine: CausalEngine::new(),
            edge_count: 0,
        }
    }

    /// Create a bridge wrapping an existing CausalEngine.
    pub fn with_engine(engine: CausalEngine) -> Self {
        Self {
            engine,
            edge_count: 0,
        }
    }

    /// Get a reference to the underlying CausalEngine.
    pub fn engine(&self) -> &CausalEngine {
        &self.engine
    }

    /// Get a mutable reference to the underlying CausalEngine.
    pub fn engine_mut(&mut self) -> &mut CausalEngine {
        &mut self.engine
    }

    /// Consume the bridge and return the CausalEngine.
    pub fn into_engine(self) -> CausalEngine {
        self.engine
    }

    /// Number of edges ingested.
    pub fn edge_count(&self) -> usize {
        self.edge_count
    }

    // -------------------------------------------------------------------------
    // EXTRACT: Grammar → GrammarCausalEdge
    // -------------------------------------------------------------------------

    /// Extract a causal edge from a CausalityFlow.
    ///
    /// The agent becomes the state, the action becomes the action fingerprint,
    /// and the patient becomes the outcome. Missing fields get deterministic
    /// placeholder fingerprints based on the available fields.
    pub fn extract_edge(flow: &CausalityFlow) -> GrammarCausalEdge {
        // Build state from agent (or fallback to "unknown_agent")
        let state = match &flow.agent {
            Some(agent) => label_to_fp(agent),
            None => label_to_fp("__causal_bridge::unknown_agent"),
        };

        // Build action from action verb
        let action = match &flow.action {
            Some(act) => label_to_fp(act),
            None => label_to_fp("__causal_bridge::unknown_action"),
        };

        // Build outcome from patient
        let outcome = match &flow.patient {
            Some(patient) => label_to_fp(patient),
            None => label_to_fp("__causal_bridge::unknown_patient"),
        };

        // Map DependencyType → EdgeType + CausalEdgeType
        let (edge_type, causal_type) = Self::map_dependency(flow.dependency);

        // Strength combines causal_strength and agency
        let strength = flow.causal_strength * (0.5 + 0.5 * flow.agency);

        GrammarCausalEdge {
            state,
            action,
            outcome,
            edge_type,
            causal_type,
            strength,
            temporality: flow.temporality,
        }
    }

    /// Extract a causal edge from a full GrammarTriangle.
    ///
    /// Uses the triangle's fingerprint as the state context, the causality
    /// flow's action as the action, and the patient as the outcome.
    /// Qualia valence modulates the edge strength.
    pub fn extract_from_triangle(triangle: &GrammarTriangle) -> GrammarCausalEdge {
        let mut edge = Self::extract_edge(&triangle.causality);

        // Use the full triangle fingerprint as state context
        // This embeds NSM + Qualia + Causality into the state
        edge.state = fp_to_words(&triangle.to_fingerprint());

        // Modulate strength by qualia certainty
        let certainty = triangle.qualia("certainty").unwrap_or(0.5);
        edge.strength *= certainty;

        edge
    }

    /// Map DependencyType to (EdgeType, CausalEdgeType).
    fn map_dependency(dep: DependencyType) -> (EdgeType, CausalEdgeType) {
        match dep {
            DependencyType::Causal => (EdgeType::Do, CausalEdgeType::Causes),
            DependencyType::Intentional => (EdgeType::Do, CausalEdgeType::Causes),
            DependencyType::Enabling => (EdgeType::Do, CausalEdgeType::MayCause),
            DependencyType::Preventing => (EdgeType::Imagine, CausalEdgeType::Causes),
            DependencyType::Correlational => (EdgeType::See, CausalEdgeType::Correlated),
            DependencyType::Constitutive => (EdgeType::See, CausalEdgeType::Correlated),
            DependencyType::None => (EdgeType::See, CausalEdgeType::Correlated),
        }
    }

    // -------------------------------------------------------------------------
    // INGEST: GrammarCausalEdge → CausalEngine
    // -------------------------------------------------------------------------

    /// Ingest a single causal edge into the engine.
    pub fn ingest_edge(&mut self, edge: &GrammarCausalEdge) {
        match edge.edge_type {
            EdgeType::See => {
                // Rung 1: Store correlation (state ↔ outcome)
                self.engine
                    .store_correlation(&edge.state, &edge.outcome, edge.strength);
            }
            EdgeType::Do => {
                // Rung 2: Store intervention (state + action → outcome)
                self.engine.store_intervention(
                    &edge.state,
                    &edge.action,
                    &edge.outcome,
                    edge.strength,
                );
            }
            EdgeType::Imagine => {
                // Rung 3: Store counterfactual (preventing = "what if this didn't happen")
                self.engine.store_counterfactual(
                    &edge.state,
                    &edge.action,
                    &edge.outcome,
                    edge.strength,
                );
            }
        }

        // Also add to the causal graph for structure queries
        self.engine
            .add_edge(&edge.action, &edge.outcome, edge.causal_type, edge.strength);

        self.edge_count += 1;
    }

    /// Ingest text directly: parse → extract → store.
    ///
    /// This is the high-level one-shot API:
    /// `text → GrammarTriangle → GrammarCausalEdge → CausalEngine`
    pub fn ingest_text(&mut self, text: &str) -> GrammarCausalEdge {
        let triangle = GrammarTriangle::from_text(text);
        let edge = Self::extract_from_triangle(&triangle);
        self.ingest_edge(&edge);
        edge
    }

    /// Ingest a CausalityFlow directly (without full triangle).
    pub fn ingest_flow(&mut self, flow: &CausalityFlow) -> GrammarCausalEdge {
        let edge = Self::extract_edge(flow);
        self.ingest_edge(&edge);
        edge
    }

    // -------------------------------------------------------------------------
    // QUERY: CausalEngine → results via Grammar-level concepts
    // -------------------------------------------------------------------------

    /// Query: "What happens if agent does action?"
    ///
    /// Converts string labels → fingerprints, queries CausalEngine at Rung 2.
    pub fn query_do(&self, agent: &str, action: &str) -> Vec<crate::search::causal::CausalResult> {
        let state = label_to_fp(agent);
        let action_fp = label_to_fp(action);
        self.engine.query_do(&state, &action_fp)
    }

    /// Query: "What caused this outcome?"
    pub fn query_cause(
        &self,
        agent: &str,
        outcome: &str,
    ) -> Vec<crate::search::causal::CausalResult> {
        let state = label_to_fp(agent);
        let outcome_fp = label_to_fp(outcome);
        self.engine.query_cause(&state, &outcome_fp)
    }

    /// Query: "What would have happened if agent had done alt_action?"
    pub fn query_counterfactual(
        &self,
        agent: &str,
        alt_action: &str,
    ) -> Vec<crate::search::causal::CausalResult> {
        let state = label_to_fp(agent);
        let alt_fp = label_to_fp(alt_action);
        self.engine.query_counterfactual(&state, &alt_fp)
    }

    /// Query: "What correlates with this concept?"
    pub fn query_correlates(
        &self,
        concept: &str,
        k: usize,
    ) -> Vec<crate::search::causal::CausalResult> {
        let fp = label_to_fp(concept);
        self.engine.query_correlates(&fp, k)
    }
}

impl Default for CausalBridge {
    fn default() -> Self {
        Self::new()
    }
}

// Re-export CausalResult from the parent module for convenience
pub use crate::search::causal::CausalResult;

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_edge_causal() {
        let flow = CausalityFlow {
            agent: Some("rain".into()),
            action: Some("causes".into()),
            patient: Some("flood".into()),
            reason: Some("heavy downpour".into()),
            temporality: -0.5,
            agency: 0.8,
            dependency: DependencyType::Causal,
            causal_strength: 0.9,
        };

        let edge = CausalBridge::extract_edge(&flow);

        assert_eq!(edge.edge_type, EdgeType::Do);
        assert_eq!(edge.causal_type, CausalEdgeType::Causes);
        assert!(edge.strength > 0.0);
        assert_eq!(edge.temporality, -0.5);
    }

    #[test]
    fn test_extract_edge_correlational() {
        let flow = CausalityFlow {
            agent: None,
            action: None,
            patient: None,
            reason: None,
            temporality: 0.0,
            agency: 0.3,
            dependency: DependencyType::Correlational,
            causal_strength: 0.5,
        };

        let edge = CausalBridge::extract_edge(&flow);

        assert_eq!(edge.edge_type, EdgeType::See);
        assert_eq!(edge.causal_type, CausalEdgeType::Correlated);
    }

    #[test]
    fn test_extract_edge_preventing() {
        let flow = CausalityFlow {
            agent: Some("dam".into()),
            action: Some("blocks".into()),
            patient: Some("flood".into()),
            reason: None,
            temporality: 0.0,
            agency: 0.9,
            dependency: DependencyType::Preventing,
            causal_strength: 0.85,
        };

        let edge = CausalBridge::extract_edge(&flow);

        // Preventing maps to Rung 3 (counterfactual)
        assert_eq!(edge.edge_type, EdgeType::Imagine);
        assert_eq!(edge.causal_type, CausalEdgeType::Causes);
    }

    #[test]
    fn test_ingest_and_query() {
        let mut bridge = CausalBridge::new();

        // Ingest: "rain causes flooding"
        let flow = CausalityFlow {
            agent: Some("rain".into()),
            action: Some("causes".into()),
            patient: Some("flooding".into()),
            reason: None,
            temporality: -0.3,
            agency: 0.7,
            dependency: DependencyType::Causal,
            causal_strength: 0.9,
        };

        bridge.ingest_flow(&flow);
        assert_eq!(bridge.edge_count(), 1);

        // Query: what does rain cause?
        let results = bridge.query_do("rain", "causes");
        // Should find the stored edge
        assert!(!results.is_empty());
    }

    #[test]
    fn test_ingest_text_pipeline() {
        let mut bridge = CausalBridge::new();

        let edge = bridge.ingest_text("The rain caused severe flooding because of the storm");

        // Should detect causal dependency
        assert_eq!(edge.edge_type, EdgeType::Do);
        assert!(edge.strength > 0.0);
        assert_eq!(bridge.edge_count(), 1);
    }

    #[test]
    fn test_ingest_multiple_texts() {
        let mut bridge = CausalBridge::new();

        bridge.ingest_text("The rain caused severe flooding because of the storm");
        bridge.ingest_text("Sun correlates with ice cream sales along with temperature");
        bridge.ingest_text("The vaccine prevents infection despite side effects");

        assert_eq!(bridge.edge_count(), 3);
    }

    #[test]
    fn test_extract_from_triangle() {
        let triangle = GrammarTriangle::from_text(
            "The decision caused major changes because of new requirements",
        );

        let edge = CausalBridge::extract_from_triangle(&triangle);

        // Should have a non-zero state (from the full triangle fingerprint)
        let zero = [0u64; WORDS];
        assert_ne!(edge.state, zero);
        assert!(edge.strength > 0.0);
    }

    #[test]
    fn test_dependency_mapping_all_variants() {
        let cases = vec![
            (DependencyType::None, EdgeType::See, CausalEdgeType::Correlated),
            (DependencyType::Causal, EdgeType::Do, CausalEdgeType::Causes),
            (DependencyType::Enabling, EdgeType::Do, CausalEdgeType::MayCause),
            (DependencyType::Preventing, EdgeType::Imagine, CausalEdgeType::Causes),
            (DependencyType::Correlational, EdgeType::See, CausalEdgeType::Correlated),
            (DependencyType::Constitutive, EdgeType::See, CausalEdgeType::Correlated),
            (DependencyType::Intentional, EdgeType::Do, CausalEdgeType::Causes),
        ];

        for (dep, expected_edge, expected_causal) in cases {
            let (edge_type, causal_type) = CausalBridge::map_dependency(dep);
            assert_eq!(edge_type, expected_edge, "Failed for {:?}", dep);
            assert_eq!(causal_type, expected_causal, "Failed for {:?}", dep);
        }
    }

    #[test]
    fn test_roundtrip_ingest_query() {
        let mut bridge = CausalBridge::new();

        // Store: agent="alice", action="sends", patient="message"
        let flow = CausalityFlow {
            agent: Some("alice".into()),
            action: Some("sends".into()),
            patient: Some("message".into()),
            reason: None,
            temporality: 0.0,
            agency: 1.0,
            dependency: DependencyType::Causal,
            causal_strength: 1.0,
        };

        bridge.ingest_flow(&flow);

        // Query the intervention: alice + sends → ?
        let results = bridge.query_do("alice", "sends");
        assert!(!results.is_empty(), "Should find stored intervention");

        // The result fingerprint should match "message"
        let expected_outcome = label_to_fp("message");
        let result_fp = &results[0].fingerprint;
        assert_eq!(
            result_fp, &expected_outcome,
            "ABBA should recover exact outcome"
        );
    }

    #[test]
    fn test_with_engine() {
        let engine = CausalEngine::new();
        let bridge = CausalBridge::with_engine(engine);
        assert_eq!(bridge.edge_count(), 0);
    }

    #[test]
    fn test_into_engine() {
        let mut bridge = CausalBridge::new();
        bridge.ingest_text("Testing because of reasons");

        let engine = bridge.into_engine();
        // Engine should have the stored edges
        let results = engine.query_correlates(&label_to_fp("test"), 5);
        // Results may or may not be empty depending on what was stored,
        // but the engine should not panic
        let _ = results;
    }
}
