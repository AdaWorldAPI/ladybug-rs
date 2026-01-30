//! Causal Search Engine
//!
//! Unified surface for three modes of reasoning:
//! - **Correlate** (Rung 1): What co-occurs? (HDR cascade)
//! - **Intervene** (Rung 2): What happens if I do X? (DO edges + ABBA)
//! - **Counterfact** (Rung 3): What would have happened? (IMAGINE edges)
//!
//! # The Pearl Ladder
//!
//! ```text
//! Rung 3: COUNTERFACTUAL  "What if I had...?"
//!         ↑               Requires: model of world
//!         │               Edge: state ⊗ IMAGINE ⊗ action ⊗ WOULD ⊗ outcome
//!         │
//! Rung 2: INTERVENTION    "What if I do...?"
//!         ↑               Requires: ability to act
//!         │               Edge: state ⊗ DO ⊗ action ⊗ CAUSES ⊗ outcome
//!         │
//! Rung 1: CORRELATION     "What do I see?"
//!         │               Requires: observation only
//!         │               Edge: X ⊗ SEE ⊗ Y (or just similarity)
//! ```
//!
//! # Why This Matters for RL
//!
//! Traditional RL learns Q(s,a) = E[reward] - pure correlation.
//! This fails when:
//! - Confounders exist (ice cream ↔ drowning, both caused by sun)
//! - Distribution shifts (training ≠ deployment)
//! - Explanations needed ("why did you do that?")
//!
//! Causal RL learns Q(s, do(a)) = E[reward | intervention].
//! This enables:
//! - Confounder detection (same cause → different edges)
//! - Transfer learning (causal structure transfers)
//! - Explainability (trace causal chain)
//!
//! # ABBA Retrieval
//!
//! XOR is self-inverse: A ⊗ B ⊗ B = A
//!
//! ```text
//! Store:  edge = state ⊗ DO ⊗ action ⊗ CAUSES ⊗ outcome
//! Query:  "what outcome?" → edge ⊗ state ⊗ DO ⊗ action ⊗ CAUSES = outcome
//! Query:  "what action?"  → edge ⊗ state ⊗ DO ⊗ CAUSES ⊗ outcome = action
//! Query:  "what state?"   → edge ⊗ DO ⊗ action ⊗ CAUSES ⊗ outcome = state
//!
//! O(1) retrieval in any direction!
//! ```

use std::collections::HashMap;
use crate::core::Fingerprint;
use crate::{Error, Result};

use super::hdr_cascade::{
    HdrIndex, MexicanHat, RollingWindow, AlienSearch,
    hamming_distance, SearchResult,
};

// =============================================================================
// CONSTANTS - THE VERBS
// =============================================================================

/// Number of u64 words in fingerprint
const WORDS: usize = 156;

/// The causal verbs - these are fingerprints too
/// Generated deterministically from content
pub struct CausalVerbs {
    /// Rung 1: Correlation/observation
    pub see: [u64; WORDS],
    /// Rung 2: Intervention
    pub do_verb: [u64; WORDS],
    /// Rung 2: Causation
    pub causes: [u64; WORDS],
    /// Rung 3: Counterfactual imagination
    pub imagine: [u64; WORDS],
    /// Rung 3: Would-cause (counterfactual causation)
    pub would: [u64; WORDS],
    /// Confounder detection
    pub confounds: [u64; WORDS],
}

impl CausalVerbs {
    /// Create verbs from deterministic seeds
    pub fn new() -> Self {
        Self {
            see: Self::seed_to_fingerprint("CAUSAL::SEE::RUNG1"),
            do_verb: Self::seed_to_fingerprint("CAUSAL::DO::RUNG2"),
            causes: Self::seed_to_fingerprint("CAUSAL::CAUSES::RUNG2"),
            imagine: Self::seed_to_fingerprint("CAUSAL::IMAGINE::RUNG3"),
            would: Self::seed_to_fingerprint("CAUSAL::WOULD::RUNG3"),
            confounds: Self::seed_to_fingerprint("CAUSAL::CONFOUNDS::DETECT"),
        }
    }
    
    /// Convert seed string to deterministic fingerprint
    fn seed_to_fingerprint(seed: &str) -> [u64; WORDS] {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut fp = [0u64; WORDS];
        for i in 0..WORDS {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            i.hash(&mut hasher);
            fp[i] = hasher.finish();
        }
        fp
    }
}

impl Default for CausalVerbs {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// EDGE TYPES
// =============================================================================

/// Edge type determines how it's stored and queried
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeType {
    /// Rung 1: X co-occurs with Y (correlation)
    See,
    /// Rung 2: doing X causes Y (intervention)
    Do,
    /// Rung 3: had I done X, Y would have happened (counterfactual)
    Imagine,
}

/// A causal edge with full structure
#[derive(Debug, Clone)]
pub struct CausalEdge {
    /// The bound fingerprint: components ⊗ verbs ⊗ ...
    pub fingerprint: [u64; WORDS],
    /// Edge type (which rung)
    pub edge_type: EdgeType,
    /// Source state
    pub state: [u64; WORDS],
    /// Action taken (for Rung 2, 3)
    pub action: Option<[u64; WORDS]>,
    /// Outcome observed
    pub outcome: [u64; WORDS],
    /// Strength/confidence
    pub weight: f32,
    /// Timestamp for decay
    pub timestamp: u64,
}

// =============================================================================
// CAUSAL STORE - THE THREE RUNGS
// =============================================================================

/// Store for Rung 1: Correlations (SEE)
/// Uses HDR cascade for similarity search
pub struct CorrelationStore {
    /// HDR index for fast correlation lookup
    index: HdrIndex,
    /// Stored edges: X ⊗ SEE ⊗ Y
    edges: Vec<CausalEdge>,
    /// Verb fingerprints
    verbs: CausalVerbs,
}

impl CorrelationStore {
    pub fn new() -> Self {
        Self {
            index: HdrIndex::new(),
            edges: Vec::new(),
            verbs: CausalVerbs::new(),
        }
    }
    
    /// Store correlation: X co-occurs with Y
    pub fn store(&mut self, x: &[u64; WORDS], y: &[u64; WORDS], weight: f32) {
        // Edge = X ⊗ SEE ⊗ Y
        let mut edge_fp = [0u64; WORDS];
        for i in 0..WORDS {
            edge_fp[i] = x[i] ^ self.verbs.see[i] ^ y[i];
        }
        
        let edge = CausalEdge {
            fingerprint: edge_fp,
            edge_type: EdgeType::See,
            state: *x,
            action: None,
            outcome: *y,
            weight,
            timestamp: 0,
        };
        
        self.index.add(&edge_fp);
        self.edges.push(edge);
    }
    
    /// Query: what correlates with X?
    pub fn query(&self, x: &[u64; WORDS], k: usize) -> Vec<(&CausalEdge, u32)> {
        // Pattern = X ⊗ SEE (partial match)
        let mut pattern = [0u64; WORDS];
        for i in 0..WORDS {
            pattern[i] = x[i] ^ self.verbs.see[i];
        }
        
        let matches = self.index.search(&pattern, k);
        matches.into_iter()
            .map(|(idx, dist)| (&self.edges[idx], dist))
            .collect()
    }
    
    /// Extract Y from edge given X (ABBA)
    pub fn unbind_outcome(&self, edge: &CausalEdge, x: &[u64; WORDS]) -> [u64; WORDS] {
        let mut result = [0u64; WORDS];
        for i in 0..WORDS {
            // edge = X ⊗ SEE ⊗ Y
            // edge ⊗ X ⊗ SEE = Y
            result[i] = edge.fingerprint[i] ^ x[i] ^ self.verbs.see[i];
        }
        result
    }
}

impl Default for CorrelationStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Store for Rung 2: Interventions (DO → CAUSES)
/// Uses ABBA for direct retrieval
pub struct InterventionStore {
    /// Edges: state ⊗ DO ⊗ action ⊗ CAUSES ⊗ outcome
    edges: Vec<CausalEdge>,
    /// Index by state prefix (for fast lookup)
    state_index: HashMap<u64, Vec<usize>>,
    /// Index by action prefix
    action_index: HashMap<u64, Vec<usize>>,
    /// Index by outcome prefix
    outcome_index: HashMap<u64, Vec<usize>>,
    /// Verb fingerprints
    verbs: CausalVerbs,
}

impl InterventionStore {
    pub fn new() -> Self {
        Self {
            edges: Vec::new(),
            state_index: HashMap::new(),
            action_index: HashMap::new(),
            outcome_index: HashMap::new(),
            verbs: CausalVerbs::new(),
        }
    }
    
    /// Store intervention: in state, doing action causes outcome
    pub fn store(
        &mut self,
        state: &[u64; WORDS],
        action: &[u64; WORDS],
        outcome: &[u64; WORDS],
        weight: f32,
    ) {
        // Edge = state ⊗ DO ⊗ action ⊗ CAUSES ⊗ outcome
        let mut edge_fp = [0u64; WORDS];
        for i in 0..WORDS {
            edge_fp[i] = state[i] 
                ^ self.verbs.do_verb[i] 
                ^ action[i] 
                ^ self.verbs.causes[i] 
                ^ outcome[i];
        }
        
        let idx = self.edges.len();
        
        let edge = CausalEdge {
            fingerprint: edge_fp,
            edge_type: EdgeType::Do,
            state: *state,
            action: Some(*action),
            outcome: *outcome,
            weight,
            timestamp: 0,
        };
        
        // Index by prefixes (first u64 as key)
        self.state_index.entry(state[0]).or_default().push(idx);
        self.action_index.entry(action[0]).or_default().push(idx);
        self.outcome_index.entry(outcome[0]).or_default().push(idx);
        
        self.edges.push(edge);
    }
    
    /// Query: what outcome if I do action in state?
    /// This is O(1) with ABBA when edge exists!
    pub fn query_outcome(
        &self,
        state: &[u64; WORDS],
        action: &[u64; WORDS],
        threshold: u32,
    ) -> Vec<(&CausalEdge, [u64; WORDS], u32)> {
        let mut results = Vec::new();
        
        // Construct query pattern
        // edge = state ⊗ DO ⊗ action ⊗ CAUSES ⊗ outcome
        // edge ⊗ state ⊗ DO ⊗ action ⊗ CAUSES = outcome
        
        // Check edges indexed by state prefix
        if let Some(indices) = self.state_index.get(&state[0]) {
            for &idx in indices {
                let edge = &self.edges[idx];
                
                // ABBA: unbind to get outcome
                let mut candidate_outcome = [0u64; WORDS];
                for i in 0..WORDS {
                    candidate_outcome[i] = edge.fingerprint[i]
                        ^ state[i]
                        ^ self.verbs.do_verb[i]
                        ^ action[i]
                        ^ self.verbs.causes[i];
                }
                
                // Verify by checking distance to stored outcome
                let dist = hamming_distance(&candidate_outcome, &edge.outcome);
                if dist < threshold {
                    results.push((edge, candidate_outcome, dist));
                }
            }
        }
        
        results
    }
    
    /// Query: what action caused this outcome in state?
    pub fn query_action(
        &self,
        state: &[u64; WORDS],
        outcome: &[u64; WORDS],
        threshold: u32,
    ) -> Vec<(&CausalEdge, [u64; WORDS], u32)> {
        let mut results = Vec::new();
        
        // edge = state ⊗ DO ⊗ action ⊗ CAUSES ⊗ outcome
        // edge ⊗ state ⊗ DO ⊗ CAUSES ⊗ outcome = action
        
        if let Some(indices) = self.outcome_index.get(&outcome[0]) {
            for &idx in indices {
                let edge = &self.edges[idx];
                
                // ABBA: unbind to get action
                let mut candidate_action = [0u64; WORDS];
                for i in 0..WORDS {
                    candidate_action[i] = edge.fingerprint[i]
                        ^ state[i]
                        ^ self.verbs.do_verb[i]
                        ^ self.verbs.causes[i]
                        ^ outcome[i];
                }
                
                // Verify
                if let Some(stored_action) = &edge.action {
                    let dist = hamming_distance(&candidate_action, stored_action);
                    if dist < threshold {
                        results.push((edge, candidate_action, dist));
                    }
                }
            }
        }
        
        results
    }
    
    /// Query: what state would lead to this outcome with this action?
    pub fn query_state(
        &self,
        action: &[u64; WORDS],
        outcome: &[u64; WORDS],
        threshold: u32,
    ) -> Vec<(&CausalEdge, [u64; WORDS], u32)> {
        let mut results = Vec::new();
        
        // edge = state ⊗ DO ⊗ action ⊗ CAUSES ⊗ outcome
        // edge ⊗ DO ⊗ action ⊗ CAUSES ⊗ outcome = state
        
        if let Some(indices) = self.action_index.get(&action[0]) {
            for &idx in indices {
                let edge = &self.edges[idx];
                
                // ABBA: unbind to get state
                let mut candidate_state = [0u64; WORDS];
                for i in 0..WORDS {
                    candidate_state[i] = edge.fingerprint[i]
                        ^ self.verbs.do_verb[i]
                        ^ action[i]
                        ^ self.verbs.causes[i]
                        ^ outcome[i];
                }
                
                // Verify
                let dist = hamming_distance(&candidate_state, &edge.state);
                if dist < threshold {
                    results.push((edge, candidate_state, dist));
                }
            }
        }
        
        results
    }
    
    /// Detect confounders: two outcomes with same cause
    pub fn detect_confounders(
        &self,
        outcome1: &[u64; WORDS],
        outcome2: &[u64; WORDS],
        threshold: u32,
    ) -> Vec<[u64; WORDS]> {
        let mut confounders = Vec::new();
        
        // Find causes of outcome1
        let causes1: Vec<_> = self.outcome_index
            .get(&outcome1[0])
            .map(|indices| {
                indices.iter()
                    .map(|&idx| self.edges[idx].state)
                    .collect()
            })
            .unwrap_or_default();
        
        // Find causes of outcome2
        let causes2: Vec<_> = self.outcome_index
            .get(&outcome2[0])
            .map(|indices| {
                indices.iter()
                    .map(|&idx| self.edges[idx].state)
                    .collect()
            })
            .unwrap_or_default();
        
        // Find common causes (confounders)
        for c1 in &causes1 {
            for c2 in &causes2 {
                let dist = hamming_distance(c1, c2);
                if dist < threshold {
                    confounders.push(*c1);
                }
            }
        }
        
        confounders
    }
}

impl Default for InterventionStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Store for Rung 3: Counterfactuals (IMAGINE → WOULD)
/// For "what would have happened if..." queries
pub struct CounterfactualStore {
    /// Edges: actual_state ⊗ IMAGINE ⊗ alt_action ⊗ WOULD ⊗ alt_outcome
    edges: Vec<CausalEdge>,
    /// Index by actual state
    state_index: HashMap<u64, Vec<usize>>,
    /// Verb fingerprints
    verbs: CausalVerbs,
}

impl CounterfactualStore {
    pub fn new() -> Self {
        Self {
            edges: Vec::new(),
            state_index: HashMap::new(),
            verbs: CausalVerbs::new(),
        }
    }
    
    /// Store counterfactual: if in state, had I done action, outcome would have happened
    pub fn store(
        &mut self,
        actual_state: &[u64; WORDS],
        alt_action: &[u64; WORDS],
        alt_outcome: &[u64; WORDS],
        weight: f32,
    ) {
        // Edge = state ⊗ IMAGINE ⊗ action ⊗ WOULD ⊗ outcome
        let mut edge_fp = [0u64; WORDS];
        for i in 0..WORDS {
            edge_fp[i] = actual_state[i]
                ^ self.verbs.imagine[i]
                ^ alt_action[i]
                ^ self.verbs.would[i]
                ^ alt_outcome[i];
        }
        
        let idx = self.edges.len();
        
        let edge = CausalEdge {
            fingerprint: edge_fp,
            edge_type: EdgeType::Imagine,
            state: *actual_state,
            action: Some(*alt_action),
            outcome: *alt_outcome,
            weight,
            timestamp: 0,
        };
        
        self.state_index.entry(actual_state[0]).or_default().push(idx);
        self.edges.push(edge);
    }
    
    /// Query: what would have happened if I had done action in state?
    pub fn query_counterfactual(
        &self,
        state: &[u64; WORDS],
        alt_action: &[u64; WORDS],
        threshold: u32,
    ) -> Vec<(&CausalEdge, [u64; WORDS], u32)> {
        let mut results = Vec::new();
        
        // edge = state ⊗ IMAGINE ⊗ action ⊗ WOULD ⊗ outcome
        // edge ⊗ state ⊗ IMAGINE ⊗ action ⊗ WOULD = outcome
        
        if let Some(indices) = self.state_index.get(&state[0]) {
            for &idx in indices {
                let edge = &self.edges[idx];
                
                // ABBA: unbind to get counterfactual outcome
                let mut cf_outcome = [0u64; WORDS];
                for i in 0..WORDS {
                    cf_outcome[i] = edge.fingerprint[i]
                        ^ state[i]
                        ^ self.verbs.imagine[i]
                        ^ alt_action[i]
                        ^ self.verbs.would[i];
                }
                
                // Verify
                let dist = hamming_distance(&cf_outcome, &edge.outcome);
                if dist < threshold {
                    results.push((edge, cf_outcome, dist));
                }
            }
        }
        
        results
    }
    
    /// Compute regret: actual outcome vs counterfactual outcome
    pub fn compute_regret(
        &self,
        state: &[u64; WORDS],
        actual_outcome: &[u64; WORDS],
        alt_action: &[u64; WORDS],
        threshold: u32,
    ) -> Option<f32> {
        // Get counterfactual outcome
        let cf_results = self.query_counterfactual(state, alt_action, threshold);
        
        if let Some((edge, cf_outcome, _)) = cf_results.first() {
            // Regret = distance between actual and counterfactual
            // (positive = counterfactual was better)
            let actual_dist = hamming_distance(actual_outcome, state);
            let cf_dist = hamming_distance(&cf_outcome, state);
            
            // Normalize to -1.0 to 1.0
            let regret = (actual_dist as f32 - cf_dist as f32) / (WORDS * 64) as f32;
            Some(regret)
        } else {
            None
        }
    }
}

impl Default for CounterfactualStore {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// UNIFIED CAUSAL SEARCH
// =============================================================================

/// Query mode - which rung of the ladder
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryMode {
    /// Rung 1: What correlates?
    Correlate,
    /// Rung 2: What happens if I do X?
    Intervene,
    /// Rung 3: What would have happened?
    Counterfact,
}

/// Unified causal search result
#[derive(Debug, Clone)]
pub struct CausalResult {
    /// Query mode used
    pub mode: QueryMode,
    /// Retrieved/computed fingerprint
    pub fingerprint: [u64; WORDS],
    /// Distance/confidence
    pub distance: u32,
    /// Weight from edge
    pub weight: f32,
    /// Edge type
    pub edge_type: EdgeType,
}

/// Unified Causal Search Engine
/// 
/// One surface, three modes. Clean separation underneath.
pub struct CausalSearch {
    /// Rung 1: Correlation store (HDR cascade)
    correlations: CorrelationStore,
    /// Rung 2: Intervention store (ABBA)
    interventions: InterventionStore,
    /// Rung 3: Counterfactual store (ABBA)
    counterfactuals: CounterfactualStore,
    /// Mexican hat for discrimination
    hat: MexicanHat,
    /// Rolling window for coherence
    window: RollingWindow,
    /// Default query threshold
    threshold: u32,
}

impl CausalSearch {
    /// Create new causal search engine
    pub fn new() -> Self {
        Self {
            correlations: CorrelationStore::new(),
            interventions: InterventionStore::new(),
            counterfactuals: CounterfactualStore::new(),
            hat: MexicanHat::default(),
            window: RollingWindow::new(100),
            threshold: 2000,
        }
    }
    
    /// Set query threshold
    pub fn set_threshold(&mut self, threshold: u32) {
        self.threshold = threshold;
    }
    
    /// Set Mexican hat parameters
    pub fn set_mexican_hat(&mut self, excite: u32, inhibit: u32) {
        self.hat = MexicanHat::new(excite, inhibit);
    }
    
    // -------------------------------------------------------------------------
    // STORE OPERATIONS
    // -------------------------------------------------------------------------
    
    /// Store correlation (Rung 1): X co-occurs with Y
    pub fn store_correlation(&mut self, x: &[u64; WORDS], y: &[u64; WORDS], weight: f32) {
        self.correlations.store(x, y, weight);
    }
    
    /// Store intervention (Rung 2): doing action in state causes outcome
    pub fn store_intervention(
        &mut self,
        state: &[u64; WORDS],
        action: &[u64; WORDS],
        outcome: &[u64; WORDS],
        weight: f32,
    ) {
        self.interventions.store(state, action, outcome, weight);
    }
    
    /// Store counterfactual (Rung 3): had I done action, outcome would have happened
    pub fn store_counterfactual(
        &mut self,
        state: &[u64; WORDS],
        alt_action: &[u64; WORDS],
        alt_outcome: &[u64; WORDS],
        weight: f32,
    ) {
        self.counterfactuals.store(state, alt_action, alt_outcome, weight);
    }
    
    // -------------------------------------------------------------------------
    // QUERY OPERATIONS
    // -------------------------------------------------------------------------
    
    /// Query correlations (Rung 1): what co-occurs with X?
    pub fn query_correlates(&self, x: &[u64; WORDS], k: usize) -> Vec<CausalResult> {
        self.correlations.query(x, k)
            .into_iter()
            .map(|(edge, dist)| CausalResult {
                mode: QueryMode::Correlate,
                fingerprint: edge.outcome,
                distance: dist,
                weight: edge.weight,
                edge_type: EdgeType::See,
            })
            .collect()
    }
    
    /// Query intervention (Rung 2): what happens if I do action in state?
    pub fn query_outcome(
        &self,
        state: &[u64; WORDS],
        action: &[u64; WORDS],
    ) -> Vec<CausalResult> {
        self.interventions.query_outcome(state, action, self.threshold)
            .into_iter()
            .map(|(edge, outcome, dist)| CausalResult {
                mode: QueryMode::Intervene,
                fingerprint: outcome,
                distance: dist,
                weight: edge.weight,
                edge_type: EdgeType::Do,
            })
            .collect()
    }
    
    /// Query intervention (Rung 2): what action caused this outcome?
    pub fn query_action(
        &self,
        state: &[u64; WORDS],
        outcome: &[u64; WORDS],
    ) -> Vec<CausalResult> {
        self.interventions.query_action(state, outcome, self.threshold)
            .into_iter()
            .map(|(edge, action, dist)| CausalResult {
                mode: QueryMode::Intervene,
                fingerprint: action,
                distance: dist,
                weight: edge.weight,
                edge_type: EdgeType::Do,
            })
            .collect()
    }
    
    /// Query counterfactual (Rung 3): what would have happened?
    pub fn query_counterfactual(
        &self,
        state: &[u64; WORDS],
        alt_action: &[u64; WORDS],
    ) -> Vec<CausalResult> {
        self.counterfactuals.query_counterfactual(state, alt_action, self.threshold)
            .into_iter()
            .map(|(edge, outcome, dist)| CausalResult {
                mode: QueryMode::Counterfact,
                fingerprint: outcome,
                distance: dist,
                weight: edge.weight,
                edge_type: EdgeType::Imagine,
            })
            .collect()
    }
    
    // -------------------------------------------------------------------------
    // UNIFIED QUERY API
    // -------------------------------------------------------------------------
    
    /// Unified query: automatically selects mode based on inputs
    pub fn query(
        &mut self,
        mode: QueryMode,
        state: &[u64; WORDS],
        action: Option<&[u64; WORDS]>,
        outcome: Option<&[u64; WORDS]>,
        k: usize,
    ) -> Vec<CausalResult> {
        let results = match mode {
            QueryMode::Correlate => {
                self.query_correlates(state, k)
            }
            QueryMode::Intervene => {
                if let Some(action) = action {
                    if outcome.is_some() {
                        // Have both: query for state (unusual)
                        self.interventions.query_state(
                            action,
                            outcome.unwrap(),
                            self.threshold
                        ).into_iter()
                            .map(|(e, s, d)| CausalResult {
                                mode: QueryMode::Intervene,
                                fingerprint: s,
                                distance: d,
                                weight: e.weight,
                                edge_type: EdgeType::Do,
                            })
                            .collect()
                    } else {
                        // Have state + action: query for outcome
                        self.query_outcome(state, action)
                    }
                } else if let Some(outcome) = outcome {
                    // Have state + outcome: query for action
                    self.query_action(state, outcome)
                } else {
                    Vec::new()
                }
            }
            QueryMode::Counterfact => {
                if let Some(action) = action {
                    self.query_counterfactual(state, action)
                } else {
                    Vec::new()
                }
            }
        };
        
        // Update rolling window with distances
        for r in &results {
            self.window.push(r.distance);
        }
        
        results
    }
    
    // -------------------------------------------------------------------------
    // ANALYSIS
    // -------------------------------------------------------------------------
    
    /// Detect confounders between two outcomes
    pub fn detect_confounders(
        &self,
        outcome1: &[u64; WORDS],
        outcome2: &[u64; WORDS],
    ) -> Vec<[u64; WORDS]> {
        self.interventions.detect_confounders(outcome1, outcome2, self.threshold)
    }
    
    /// Compute regret for a counterfactual
    pub fn compute_regret(
        &self,
        state: &[u64; WORDS],
        actual_outcome: &[u64; WORDS],
        alt_action: &[u64; WORDS],
    ) -> Option<f32> {
        self.counterfactuals.compute_regret(state, actual_outcome, alt_action, self.threshold)
    }
    
    /// Get coherence stats
    pub fn coherence(&self) -> (f32, f32) {
        self.window.stats()
    }
    
    /// Is the search pattern coherent?
    pub fn is_coherent(&self) -> bool {
        self.window.is_coherent(0.3)
    }
    
    /// Get Mexican hat response for a distance
    pub fn response(&self, distance: u32) -> f32 {
        self.hat.response(distance)
    }
}

impl Default for CausalSearch {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    fn random_fp() -> [u64; WORDS] {
        let mut fp = [0u64; WORDS];
        for i in 0..WORDS {
            fp[i] = rand::random();
        }
        fp
    }
    
    #[test]
    fn test_correlation_store() {
        let mut store = CorrelationStore::new();
        
        let x = random_fp();
        let y = random_fp();
        
        store.store(&x, &y, 1.0);
        
        let results = store.query(&x, 5);
        assert!(!results.is_empty());
    }
    
    #[test]
    fn test_intervention_abba() {
        let mut store = InterventionStore::new();
        
        let state = random_fp();
        let action = random_fp();
        let outcome = random_fp();
        
        store.store(&state, &action, &outcome, 1.0);
        
        // Query for outcome
        let results = store.query_outcome(&state, &action, 100);
        assert!(!results.is_empty());
        
        // ABBA should recover exact outcome
        let (_, recovered, dist) = &results[0];
        assert_eq!(*dist, 0);
        assert_eq!(recovered, &outcome);
    }
    
    #[test]
    fn test_intervention_reverse_query() {
        let mut store = InterventionStore::new();
        
        let state = random_fp();
        let action = random_fp();
        let outcome = random_fp();
        
        store.store(&state, &action, &outcome, 1.0);
        
        // Query for action given outcome
        let results = store.query_action(&state, &outcome, 100);
        assert!(!results.is_empty());
        
        // ABBA should recover exact action
        let (_, recovered, dist) = &results[0];
        assert_eq!(*dist, 0);
        assert_eq!(recovered, &action);
    }
    
    #[test]
    fn test_counterfactual() {
        let mut store = CounterfactualStore::new();
        
        let state = random_fp();
        let alt_action = random_fp();
        let alt_outcome = random_fp();
        
        store.store(&state, &alt_action, &alt_outcome, 1.0);
        
        // Query counterfactual
        let results = store.query_counterfactual(&state, &alt_action, 100);
        assert!(!results.is_empty());
        
        // Should recover counterfactual outcome
        let (_, recovered, dist) = &results[0];
        assert_eq!(*dist, 0);
        assert_eq!(recovered, &alt_outcome);
    }
    
    #[test]
    fn test_unified_causal_search() {
        let mut search = CausalSearch::new();
        
        let state = random_fp();
        let action = random_fp();
        let outcome = random_fp();
        
        // Store intervention
        search.store_intervention(&state, &action, &outcome, 1.0);
        
        // Query via unified API
        let results = search.query(
            QueryMode::Intervene,
            &state,
            Some(&action),
            None,
            5,
        );
        
        assert!(!results.is_empty());
        assert_eq!(results[0].mode, QueryMode::Intervene);
        assert_eq!(results[0].fingerprint, outcome);
    }
    
    #[test]
    fn test_confounder_detection() {
        let mut search = CausalSearch::new();
        
        // Sun causes both ice cream and drowning
        let sun = random_fp();
        let ice_cream = random_fp();
        let drowning = random_fp();
        let eat = random_fp();
        let swim = random_fp();
        
        // sun -> ice_cream
        search.store_intervention(&sun, &eat, &ice_cream, 1.0);
        // sun -> drowning
        search.store_intervention(&sun, &swim, &drowning, 1.0);
        
        // Detect: ice_cream and drowning share a cause
        let confounders = search.detect_confounders(&ice_cream, &drowning);
        
        // Sun should be detected as confounder
        assert!(!confounders.is_empty());
        let dist = hamming_distance(&confounders[0], &sun);
        assert_eq!(dist, 0);
    }
}
