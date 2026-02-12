//! Causality Operations (0xA00-0xAFF)
//!
//! Pearl's do-calculus as fingerprint operations.
//! Integrates with search module for ABBA retrieval.
//!
//! # The Three Rungs (implemented here)
//!
//! ```text
//! Rung 1: SEE (0xA00-0xA2F)
//!   - Correlation queries
//!   - Association strength
//!   - Co-occurrence patterns
//!
//! Rung 2: DO (0xA30-0xA5F)
//!   - Intervention queries
//!   - Causal effect estimation
//!   - do-calculus rules
//!
//! Rung 3: IMAGINE (0xA60-0xA8F)
//!   - Counterfactual queries
//!   - Regret computation
//!   - Credit assignment
//! ```
//!
//! # Integration with Search Module
//!
//! All operations use CausalSearch from the search module.
//! ABBA retrieval (A⊗B⊗B=A) enables O(1) queries in any direction.

use crate::core::Fingerprint;
use crate::search::causal::{CausalSearch, CausalVerbs, QueryMode, CausalResult, EdgeType};

// =============================================================================
// CAUSALITY OPERATION CODES (0xA00-0xAFF)
// =============================================================================

#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CausalOp {
    // =========================================================================
    // RUNG 1: ASSOCIATION (0xA00-0xA2F)
    // "What do I see?"
    // =========================================================================
    
    /// P(Y|X) - conditional probability via correlation
    CondProb            = 0xA00,
    /// Correlation strength between X and Y
    Correlation         = 0xA01,
    /// Mutual information I(X;Y)
    MutualInfo          = 0xA02,
    /// Co-occurrence count
    CoOccurrence        = 0xA03,
    /// Store correlation: X co-occurs with Y
    StoreCorrelation    = 0xA04,
    /// Query correlates: what co-occurs with X?
    QueryCorrelates     = 0xA05,
    /// Association rule: X → Y (confidence, support)
    AssociationRule     = 0xA06,
    /// Lift: P(Y|X) / P(Y)
    Lift                = 0xA07,
    
    // Patterns (0xA10-0xA1F)
    /// Frequent itemset detection
    FrequentItemset     = 0xA10,
    /// Sequential pattern mining
    SequentialPattern   = 0xA11,
    /// Temporal correlation
    TemporalCorrelation = 0xA12,
    /// Spatial correlation
    SpatialCorrelation  = 0xA13,
    
    // Simpson's paradox detection (0xA20-0xA2F)
    /// Detect Simpson's paradox
    SimpsonDetect       = 0xA20,
    /// Stratify by confounder
    Stratify            = 0xA21,
    /// Aggregate vs disaggregate comparison
    AggregateCompare    = 0xA22,
    
    // =========================================================================
    // RUNG 2: INTERVENTION (0xA30-0xA5F)
    // "What if I do X?"
    // =========================================================================
    
    /// P(Y | do(X)) - interventional distribution
    DoProb              = 0xA30,
    /// Store intervention: do(X) causes Y
    StoreIntervention   = 0xA31,
    /// Query outcome of intervention
    QueryIntervention   = 0xA32,
    /// Query cause of outcome
    QueryCause          = 0xA33,
    /// Average Treatment Effect: E[Y|do(X=1)] - E[Y|do(X=0)]
    AverageTreatment    = 0xA34,
    /// Conditional ATE: ATE for subgroup
    ConditionalAte      = 0xA35,
    
    // Do-calculus rules (0xA40-0xA4F)
    /// Rule 1: Insertion/deletion of observations
    DoRule1             = 0xA40,
    /// Rule 2: Action/observation exchange
    DoRule2             = 0xA41,
    /// Rule 3: Insertion/deletion of actions
    DoRule3             = 0xA42,
    /// Check identifiability
    Identifiable        = 0xA43,
    /// Compute adjustment formula
    AdjustmentFormula   = 0xA44,
    /// Front-door adjustment
    FrontDoorAdjust     = 0xA45,
    /// Back-door adjustment
    BackDoorAdjust      = 0xA46,
    
    // Confounder handling (0xA50-0xA5F)
    /// Detect confounders
    DetectConfounder    = 0xA50,
    /// List all confounders between X and Y
    ListConfounders     = 0xA51,
    /// Adjust for confounder
    AdjustConfounder    = 0xA52,
    /// Instrumental variable estimation
    InstrumentalVar     = 0xA53,
    /// Propensity score matching
    PropensityMatch     = 0xA54,
    /// Inverse probability weighting
    InverseProbWeight   = 0xA55,
    
    // =========================================================================
    // RUNG 3: COUNTERFACTUAL (0xA60-0xA8F)
    // "What if I had done X differently?"
    // =========================================================================
    
    /// Y_x - counterfactual outcome under intervention
    Counterfactual      = 0xA60,
    /// Store counterfactual
    StoreCf             = 0xA61,
    /// Query counterfactual outcome
    QueryCf             = 0xA62,
    /// Probability of necessity: P(Y'_x' | X=x, Y=y)
    ProbNecessity       = 0xA63,
    /// Probability of sufficiency: P(Y_x | X=x', Y=y')
    ProbSufficiency     = 0xA64,
    /// Probability of necessity and sufficiency
    ProbNS              = 0xA65,
    
    // Regret and blame (0xA70-0xA7F)
    /// Compute regret: actual vs counterfactual
    ComputeRegret       = 0xA70,
    /// Assign blame to action
    AssignBlame         = 0xA71,
    /// Assign credit to action
    AssignCredit        = 0xA72,
    /// Responsibility degree
    Responsibility      = 0xA73,
    /// Actual causation (Halpern-Pearl)
    ActualCausation     = 0xA74,
    
    // Explanation (0xA80-0xA8F)
    /// Contrastive explanation: why X and not Y?
    ContrastiveExplain  = 0xA80,
    /// Counterfactual explanation
    CfExplain           = 0xA81,
    /// Necessary cause explanation
    NecessaryCause      = 0xA82,
    /// Sufficient cause explanation
    SufficientCause     = 0xA83,
    
    // =========================================================================
    // GRAPH OPERATIONS (0xA90-0xABF)
    // Causal graph structure
    // =========================================================================
    
    /// Add node to causal graph
    GraphAddNode        = 0xA90,
    /// Add edge to causal graph
    GraphAddEdge        = 0xA91,
    /// Remove node
    GraphRemoveNode     = 0xA92,
    /// Remove edge
    GraphRemoveEdge     = 0xA93,
    /// Find parents of node
    GraphParents        = 0xA94,
    /// Find children of node
    GraphChildren       = 0xA95,
    /// Find ancestors
    GraphAncestors      = 0xA96,
    /// Find descendants
    GraphDescendants    = 0xA97,
    
    // Path operations (0xAA0-0xAAF)
    /// Find all causal paths
    FindPaths           = 0xAA0,
    /// Find backdoor paths
    FindBackdoor        = 0xAA1,
    /// Find frontdoor paths
    FindFrontdoor       = 0xAA2,
    /// Check d-separation
    DSeparation         = 0xAA3,
    /// Find minimal adjustment set
    MinimalAdjustment   = 0xAA4,
    
    // Discovery (0xAB0-0xABF)
    /// PC algorithm for structure learning
    DiscoverPc          = 0xAB0,
    /// FCI algorithm
    DiscoverFci         = 0xAB1,
    /// GES algorithm
    DiscoverGes         = 0xAB2,
    /// Score-based discovery
    DiscoverScore       = 0xAB3,
    
    // =========================================================================
    // MEDIATION (0xAC0-0xADF)
    // X → M → Y analysis
    // =========================================================================
    
    /// Total effect
    TotalEffect         = 0xAC0,
    /// Direct effect (not through mediator)
    DirectEffect        = 0xAC1,
    /// Indirect effect (through mediator)
    IndirectEffect      = 0xAC2,
    /// Natural direct effect
    NaturalDirect       = 0xAC3,
    /// Natural indirect effect
    NaturalIndirect     = 0xAC4,
    /// Controlled direct effect
    ControlledDirect    = 0xAC5,
    /// Mediation proportion
    MediationProportion = 0xAC6,
    
    // =========================================================================
    // TIME SERIES CAUSALITY (0xAE0-0xAFF)
    // =========================================================================
    
    /// Granger causality test
    GrangerCausality    = 0xAE0,
    /// Transfer entropy
    TransferEntropy     = 0xAE1,
    /// Convergent cross mapping
    ConvergentCrossMap  = 0xAE2,
    /// Intervention time series
    InterventionTs      = 0xAE3,
}

// =============================================================================
// CAUSAL ENGINE
// =============================================================================

/// Causal reasoning engine
/// 
/// Wraps CausalSearch with higher-level operations and do-calculus
pub struct CausalEngine {
    /// Underlying causal search
    search: CausalSearch,
    /// Causal graph (adjacency representation via fingerprints)
    graph_edges: Vec<GraphEdge>,
    /// Verb fingerprints
    verbs: CausalVerbs,
}

/// Edge in causal graph
#[derive(Debug, Clone)]
pub struct GraphEdge {
    pub from: [u64; 256],
    pub to: [u64; 256],
    pub edge_type: CausalEdgeType,
    pub strength: f32,
}

/// Type of causal edge
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalEdgeType {
    /// X causes Y
    Causes,
    /// X possibly causes Y (uncertain)
    MayCause,
    /// X and Y are correlated (no causal direction)
    Correlated,
    /// X and Y share a common cause
    CommonCause,
}

impl CausalEngine {
    /// Create new causal engine
    pub fn new() -> Self {
        Self {
            search: CausalSearch::new(),
            graph_edges: Vec::new(),
            verbs: CausalVerbs::new(),
        }
    }
    
    // -------------------------------------------------------------------------
    // RUNG 1: ASSOCIATION
    // -------------------------------------------------------------------------
    
    /// Store correlation: X co-occurs with Y
    pub fn store_correlation(&mut self, x: &[u64; 256], y: &[u64; 256], strength: f32) {
        self.search.store_correlation(x, y, strength);
    }
    
    /// Query: what correlates with X?
    pub fn query_correlates(&self, x: &[u64; 256], k: usize) -> Vec<CausalResult> {
        self.search.query_correlates(x, k)
    }
    
    // -------------------------------------------------------------------------
    // RUNG 2: INTERVENTION
    // -------------------------------------------------------------------------
    
    /// Store intervention: do(X) in state causes Y
    pub fn store_intervention(
        &mut self,
        state: &[u64; 256],
        action: &[u64; 256],
        outcome: &[u64; 256],
        strength: f32,
    ) {
        self.search.store_intervention(state, action, outcome, strength);
        
        // Also add to graph
        self.graph_edges.push(GraphEdge {
            from: *action,
            to: *outcome,
            edge_type: CausalEdgeType::Causes,
            strength,
        });
    }
    
    /// Query: P(Y | do(X))
    pub fn query_do(
        &self,
        state: &[u64; 256],
        action: &[u64; 256],
    ) -> Vec<CausalResult> {
        self.search.query_outcome(state, action)
    }
    
    /// Query: what action caused this outcome?
    pub fn query_cause(
        &self,
        state: &[u64; 256],
        outcome: &[u64; 256],
    ) -> Vec<CausalResult> {
        self.search.query_action(state, outcome)
    }
    
    /// Detect confounders
    pub fn detect_confounders(
        &self,
        outcome1: &[u64; 256],
        outcome2: &[u64; 256],
    ) -> Vec<[u64; 256]> {
        self.search.detect_confounders(outcome1, outcome2)
    }
    
    /// Average Treatment Effect: E[Y|do(X=1)] - E[Y|do(X=0)]
    pub fn average_treatment_effect(
        &self,
        state: &[u64; 256],
        treatment: &[u64; 256],
        control: &[u64; 256],
    ) -> Option<f32> {
        let treated = self.query_do(state, treatment);
        let untreated = self.query_do(state, control);
        
        if treated.is_empty() || untreated.is_empty() {
            return None;
        }
        
        // Average weights (which represent effect strength)
        let treated_avg: f32 = treated.iter().map(|r| r.weight).sum::<f32>() 
            / treated.len() as f32;
        let untreated_avg: f32 = untreated.iter().map(|r| r.weight).sum::<f32>()
            / untreated.len() as f32;
        
        Some(treated_avg - untreated_avg)
    }
    
    // -------------------------------------------------------------------------
    // RUNG 3: COUNTERFACTUAL
    // -------------------------------------------------------------------------
    
    /// Store counterfactual
    pub fn store_counterfactual(
        &mut self,
        state: &[u64; 256],
        alt_action: &[u64; 256],
        alt_outcome: &[u64; 256],
        strength: f32,
    ) {
        self.search.store_counterfactual(state, alt_action, alt_outcome, strength);
    }
    
    /// Query counterfactual: what would have happened?
    pub fn query_counterfactual(
        &self,
        state: &[u64; 256],
        alt_action: &[u64; 256],
    ) -> Vec<CausalResult> {
        self.search.query_counterfactual(state, alt_action)
    }
    
    /// Compute regret
    pub fn compute_regret(
        &self,
        state: &[u64; 256],
        actual_outcome: &[u64; 256],
        alt_action: &[u64; 256],
    ) -> Option<f32> {
        self.search.compute_regret(state, actual_outcome, alt_action)
    }
    
    /// Probability of Necessity: P(Y'_x' | X=x, Y=y)
    /// "Was X=x necessary for Y=y?"
    pub fn prob_necessity(
        &self,
        state: &[u64; 256],
        x: &[u64; 256],        // Actual action
        y: &[u64; 256],        // Actual outcome  
        x_prime: &[u64; 256],  // Alternative action
    ) -> Option<f32> {
        // Query: if we had done x' instead, would y still have happened?
        let cf_results = self.query_counterfactual(state, x_prime);
        
        if cf_results.is_empty() {
            return None;
        }
        
        // Check if any counterfactual outcome is different from y
        let y_would_change = cf_results.iter()
            .any(|r| {
                // Distance > threshold means different outcome
                hamming_distance(&r.fingerprint, y) > 2000
            });
        
        if y_would_change {
            // X was necessary (changing it changes Y)
            Some(cf_results[0].weight)
        } else {
            // X was not necessary (Y happens anyway)
            Some(0.0)
        }
    }
    
    /// Probability of Sufficiency: P(Y_x | X=x', Y=y')
    /// "Would X=x be sufficient to cause Y=y?"
    pub fn prob_sufficiency(
        &self,
        state: &[u64; 256],
        x: &[u64; 256],        // Action to test
        y: &[u64; 256],        // Desired outcome
    ) -> Option<f32> {
        // Query: if we do x, will y happen?
        let do_results = self.query_do(state, x);
        
        if do_results.is_empty() {
            return None;
        }
        
        // Check if outcome matches y
        let y_would_happen = do_results.iter()
            .any(|r| hamming_distance(&r.fingerprint, y) < 2000);
        
        if y_would_happen {
            Some(do_results[0].weight)
        } else {
            Some(0.0)
        }
    }
    
    // -------------------------------------------------------------------------
    // GRAPH OPERATIONS
    // -------------------------------------------------------------------------
    
    /// Add edge to causal graph
    pub fn add_edge(
        &mut self,
        from: &[u64; 256],
        to: &[u64; 256],
        edge_type: CausalEdgeType,
        strength: f32,
    ) {
        self.graph_edges.push(GraphEdge {
            from: *from,
            to: *to,
            edge_type,
            strength,
        });
    }
    
    /// Find parents of a node (direct causes)
    pub fn parents(&self, node: &[u64; 256]) -> Vec<[u64; 256]> {
        self.graph_edges.iter()
            .filter(|e| hamming_distance(&e.to, node) < 100)
            .map(|e| e.from)
            .collect()
    }
    
    /// Find children of a node (direct effects)
    pub fn children(&self, node: &[u64; 256]) -> Vec<[u64; 256]> {
        self.graph_edges.iter()
            .filter(|e| hamming_distance(&e.from, node) < 100)
            .map(|e| e.to)
            .collect()
    }
    
    /// Find all ancestors (transitive causes)
    pub fn ancestors(&self, node: &[u64; 256]) -> Vec<[u64; 256]> {
        let mut result = Vec::new();
        let mut frontier = self.parents(node);
        let mut visited = std::collections::HashSet::new();
        visited.insert(hash_fp(node));
        
        while let Some(parent) = frontier.pop() {
            let h = hash_fp(&parent);
            if visited.contains(&h) {
                continue;
            }
            visited.insert(h);
            result.push(parent);
            frontier.extend(self.parents(&parent));
        }
        
        result
    }
    
    /// Find all descendants (transitive effects)
    pub fn descendants(&self, node: &[u64; 256]) -> Vec<[u64; 256]> {
        let mut result = Vec::new();
        let mut frontier = self.children(node);
        let mut visited = std::collections::HashSet::new();
        visited.insert(hash_fp(node));
        
        while let Some(child) = frontier.pop() {
            let h = hash_fp(&child);
            if visited.contains(&h) {
                continue;
            }
            visited.insert(h);
            result.push(child);
            frontier.extend(self.children(&child));
        }
        
        result
    }
    
    /// Check d-separation: are X and Y independent given Z?
    pub fn d_separated(
        &self,
        x: &[u64; 256],
        y: &[u64; 256],
        z: &[[u64; 256]],
    ) -> bool {
        // Simplified d-separation check
        // Full implementation would trace all paths and check blocking
        
        let x_ancestors = self.ancestors(x);
        let y_ancestors = self.ancestors(y);
        
        // Check if any Z blocks the path
        for zi in z {
            // If Z is on the path between X and Y, it blocks
            let zi_desc = self.descendants(zi);
            let zi_anc = self.ancestors(zi);
            
            // Very simplified: if Z is an ancestor of both X and Y, and
            // conditioning on Z blocks the path
            let blocks_path = (zi_anc.iter().any(|a| hamming_distance(a, x) < 100) ||
                              zi_desc.iter().any(|d| hamming_distance(d, x) < 100)) &&
                             (zi_anc.iter().any(|a| hamming_distance(a, y) < 100) ||
                              zi_desc.iter().any(|d| hamming_distance(d, y) < 100));
            
            if blocks_path {
                return true;
            }
        }
        
        false
    }
    
    // -------------------------------------------------------------------------
    // MEDIATION ANALYSIS
    // -------------------------------------------------------------------------
    
    /// Total effect: X → Y (direct + indirect)
    pub fn total_effect(
        &self,
        state: &[u64; 256],
        x: &[u64; 256],
        y: &[u64; 256],
    ) -> Option<f32> {
        // Total effect = P(Y | do(X))
        let results = self.query_do(state, x);
        if results.is_empty() {
            return None;
        }
        
        // Return average effect
        Some(results.iter().map(|r| r.weight).sum::<f32>() / results.len() as f32)
    }
    
    /// Estimate natural direct effect (effect not through mediator)
    pub fn natural_direct_effect(
        &self,
        state: &[u64; 256],
        x: &[u64; 256],
        _mediator: &[u64; 256],
    ) -> Option<f32> {
        // NDE = E[Y_{x,M_{x'}}] - E[Y_{x',M_{x'}}]
        // This requires nested counterfactuals
        // Simplified: just return direct edge strength if it exists
        
        for edge in &self.graph_edges {
            if hamming_distance(&edge.from, x) < 100 {
                return Some(edge.strength);
            }
        }
        
        None
    }
}

impl Default for CausalEngine {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// HELPERS
// =============================================================================

fn hamming_distance(a: &[u64; 256], b: &[u64; 256]) -> u32 {
    let mut dist = 0u32;
    for i in 0..256 {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}

fn hash_fp(fp: &[u64; 256]) -> u64 {
    fp[0] ^ fp[1].rotate_left(32) ^ fp[2]
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Convert Fingerprint to 256-word array
    fn fp_to_words(fp: &Fingerprint) -> [u64; 256] {
        let raw = fp.as_raw();
        let mut result = [0u64; 256];
        result.copy_from_slice(&raw[..256]);
        result
    }

    /// Create a content-based fingerprint (proper density)
    fn random_fp() -> [u64; 256] {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        fp_to_words(&Fingerprint::from_content(&format!("concept_{}", id)))
    }

    #[test]
    fn test_store_query_correlation() {
        let mut engine = CausalEngine::new();

        // Use content-based fingerprints that represent actual concepts
        let x = fp_to_words(&Fingerprint::from_content("rain"));
        let y = fp_to_words(&Fingerprint::from_content("wet_ground"));

        engine.store_correlation(&x, &y, 0.9);

        let results = engine.query_correlates(&x, 5);
        assert!(!results.is_empty());
    }
    
    #[test]
    fn test_intervention() {
        let mut engine = CausalEngine::new();
        
        let state = random_fp();
        let action = random_fp();
        let outcome = random_fp();
        
        engine.store_intervention(&state, &action, &outcome, 1.0);
        
        let results = engine.query_do(&state, &action);
        assert!(!results.is_empty());
        
        // Should also be in graph
        assert!(!engine.graph_edges.is_empty());
    }
    
    #[test]
    fn test_counterfactual() {
        let mut engine = CausalEngine::new();
        
        let state = random_fp();
        let alt_action = random_fp();
        let alt_outcome = random_fp();
        
        engine.store_counterfactual(&state, &alt_action, &alt_outcome, 0.8);
        
        let results = engine.query_counterfactual(&state, &alt_action);
        assert!(!results.is_empty());
    }
    
    #[test]
    fn test_graph_operations() {
        let mut engine = CausalEngine::new();
        
        let a = random_fp();
        let b = random_fp();
        let c = random_fp();
        
        // A → B → C
        engine.add_edge(&a, &b, CausalEdgeType::Causes, 1.0);
        engine.add_edge(&b, &c, CausalEdgeType::Causes, 1.0);
        
        // Parents of B should include A
        let parents = engine.parents(&b);
        assert!(!parents.is_empty());
        
        // Children of B should include C
        let children = engine.children(&b);
        assert!(!children.is_empty());
    }
}
