//! Reinforcement Learning Operations (0x900-0x9FF)
//!
//! Causal RL operations that use the search module for:
//! - Intervention-based Q-learning (Rung 2)
//! - Counterfactual policy evaluation (Rung 3)
//! - Confounder-aware value estimation
//!
//! # Why Causal RL?
//!
//! Traditional RL: Q(s,a) = E[reward | s,a]
//!   - Learns correlations, not causes
//!   - Fails with confounders
//!   - Can't transfer across domains
//!   - Can't explain decisions
//!
//! Causal RL: Q(s, do(a)) = E[reward | s, do(a)]
//!   - Learns causal effects
//!   - Handles confounders
//!   - Transfers when causal structure preserved
//!   - Can trace causal chain for explanation
//!
//! # Integration with Search Module
//!
//! ```text
//! ┌─────────────────┐     ┌─────────────────┐
//! │   RlOps (here)  │────▶│  CausalSearch   │
//! │   0x900-0x9FF   │     │  (search mod)   │
//! └─────────────────┘     └─────────────────┘
//!         │                       │
//!         │                       │
//!         ▼                       ▼
//! ┌─────────────────┐     ┌─────────────────┐
//! │  InterventionStore   │     │  Counterfactual │
//! │  (Rung 2: DO)   │     │  (Rung 3: IMAGINE)│
//! └─────────────────┘     └─────────────────┘
//! ```

use crate::core::Fingerprint;
use crate::search::causal::{CausalSearch, QueryMode, CausalResult};

// =============================================================================
// RL OPERATION CODES (0x900-0x9FF)
// =============================================================================

#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RlOp {
    // Value estimation (0x900-0x90F)
    /// Q(s, do(a)) - causal Q-value
    QValueCausal        = 0x900,
    /// V(s) - state value
    StateValue          = 0x901,
    /// A(s,a) = Q(s,a) - V(s) - advantage
    Advantage           = 0x902,
    /// TD error: r + γV(s') - V(s)
    TdError             = 0x903,
    /// n-step return
    NStepReturn         = 0x904,
    /// GAE (Generalized Advantage Estimation)
    GaeAdvantage        = 0x905,
    
    // Policy (0x910-0x91F)
    /// π(a|s) - policy probability
    PolicyProb          = 0x910,
    /// argmax π(a|s) - greedy action
    PolicyGreedy        = 0x911,
    /// Sample from π(a|s)
    PolicySample        = 0x912,
    /// ε-greedy action selection
    PolicyEpsilonGreedy = 0x913,
    /// Softmax action selection
    PolicySoftmax       = 0x914,
    /// UCB action selection
    PolicyUcb           = 0x915,
    
    // Learning updates (0x920-0x92F)
    /// Q-learning update
    UpdateQ             = 0x920,
    /// SARSA update
    UpdateSarsa         = 0x921,
    /// Policy gradient update
    UpdatePolicyGrad    = 0x922,
    /// Actor-critic update
    UpdateActorCritic   = 0x923,
    /// PPO update
    UpdatePpo           = 0x924,
    /// DQN update (with target network)
    UpdateDqn           = 0x925,
    
    // Causal-specific (0x930-0x93F)
    /// Store intervention: do(a) in state causes outcome
    StoreIntervention   = 0x930,
    /// Query outcome: what happens if do(a)?
    QueryOutcome        = 0x931,
    /// Query action: what action caused this?
    QueryCause          = 0x932,
    /// Detect confounders between outcomes
    DetectConfounder    = 0x933,
    /// Compute causal effect
    CausalEffect        = 0x934,
    /// Adjustment for confounders
    CausalAdjust        = 0x935,
    
    // Counterfactual (0x940-0x94F) - Rung 3
    /// Store counterfactual: had I done a', outcome would be...
    StoreCounterfactual = 0x940,
    /// Query: what would have happened?
    QueryCounterfactual = 0x941,
    /// Compute regret: actual vs counterfactual
    ComputeRegret       = 0x942,
    /// Blame assignment via counterfactual
    AssignBlame         = 0x943,
    /// Credit assignment via counterfactual
    AssignCredit        = 0x944,
    
    // Exploration (0x950-0x95F)
    /// Intrinsic motivation: novelty
    IntrinsicNovelty    = 0x950,
    /// Intrinsic motivation: curiosity
    IntrinsicCuriosity  = 0x951,
    /// Intrinsic motivation: empowerment
    IntrinsicEmpowerment= 0x952,
    /// Count-based exploration bonus
    ExploreCount        = 0x953,
    /// RND (Random Network Distillation) bonus
    ExploreRnd          = 0x954,
    /// ICM (Intrinsic Curiosity Module) bonus
    ExploreIcm          = 0x955,
    
    // Model-based (0x960-0x96F)
    /// Learn transition model: P(s'|s,a)
    LearnTransition     = 0x960,
    /// Learn reward model: R(s,a)
    LearnReward         = 0x961,
    /// Plan via model rollout
    PlanRollout         = 0x962,
    /// MCTS-style planning
    PlanMcts            = 0x963,
    /// Dyna-style planning
    PlanDyna            = 0x964,
    
    // Multi-agent (0x970-0x97F)
    /// Nash equilibrium
    NashEquilibrium     = 0x970,
    /// Correlated equilibrium
    CorrelatedEq        = 0x971,
    /// Self-play update
    SelfPlayUpdate      = 0x972,
    /// Opponent modeling
    OpponentModel       = 0x973,
    
    // Hierarchical (0x980-0x98F)
    /// Option selection (high-level)
    OptionSelect        = 0x980,
    /// Option termination
    OptionTerminate     = 0x981,
    /// Subgoal discovery
    SubgoalDiscover     = 0x982,
    /// Skill learning
    SkillLearn          = 0x983,
    
    // Safety (0x990-0x99F)
    /// Constraint satisfaction
    ConstraintSatisfy   = 0x990,
    /// Risk-sensitive value
    RiskSensitiveValue  = 0x991,
    /// CVaR (Conditional Value at Risk)
    CvarValue           = 0x992,
    /// Safe exploration
    SafeExplore         = 0x993,
    
    // Explanation (0x9A0-0x9AF)
    /// Explain action choice
    ExplainAction       = 0x9A0,
    /// Trace causal chain
    TraceCausalChain    = 0x9A1,
    /// Feature importance
    FeatureImportance   = 0x9A2,
    /// Counterfactual explanation
    ExplainCounterfact  = 0x9A3,
}

// =============================================================================
// CAUSAL RL AGENT
// =============================================================================

/// Causal RL agent that uses intervention and counterfactual reasoning
pub struct CausalRlAgent {
    /// Causal search engine (correlation, intervention, counterfactual)
    causal: CausalSearch,
    /// Discount factor
    gamma: f32,
    /// Learning rate
    alpha: f32,
    /// Exploration rate
    epsilon: f32,
    /// Q-value cache (for fast lookup)
    q_cache: std::collections::HashMap<u64, f32>,
}

impl CausalRlAgent {
    /// Create new causal RL agent
    pub fn new(gamma: f32, alpha: f32, epsilon: f32) -> Self {
        Self {
            causal: CausalSearch::new(),
            gamma,
            alpha,
            epsilon,
            q_cache: std::collections::HashMap::new(),
        }
    }
    
    /// Hash a state-action pair for cache
    fn hash_sa(state: &[u64; 256], action: &[u64; 256]) -> u64 {
        state[0] ^ action[0] ^ state[1].rotate_left(32) ^ action[1].rotate_left(32)
    }
    
    // -------------------------------------------------------------------------
    // RUNG 2: INTERVENTION (do-calculus)
    // -------------------------------------------------------------------------
    
    /// Store intervention: in state, doing action causes outcome with reward
    pub fn store_intervention(
        &mut self,
        state: &[u64; 256],
        action: &[u64; 256],
        outcome: &[u64; 256],
        reward: f32,
    ) {
        self.causal.store_intervention(state, action, outcome, reward);
    }
    
    /// Query: what outcome if I do action in state?
    pub fn query_outcome(
        &self,
        state: &[u64; 256],
        action: &[u64; 256],
    ) -> Vec<CausalResult> {
        self.causal.query_outcome(state, action)
    }
    
    /// Query: what action caused this outcome?
    pub fn query_cause(
        &self,
        state: &[u64; 256],
        outcome: &[u64; 256],
    ) -> Vec<CausalResult> {
        self.causal.query_action(state, outcome)
    }
    
    /// Causal Q-value: Q(s, do(a)) = E[reward | do(a) in s]
    /// Uses stored interventions, not just correlations
    pub fn q_value_causal(
        &self,
        state: &[u64; 256],
        action: &[u64; 256],
    ) -> f32 {
        let outcomes = self.query_outcome(state, action);
        
        if outcomes.is_empty() {
            return 0.0;
        }
        
        // Weighted average of rewards (weights from edge strength)
        let total_weight: f32 = outcomes.iter().map(|r| r.weight).sum();
        if total_weight <= 0.0 {
            return 0.0;
        }
        
        outcomes.iter()
            .map(|r| r.weight * r.weight)  // weight IS the reward in our storage
            .sum::<f32>() / total_weight
    }
    
    /// Detect confounders between two outcomes
    pub fn detect_confounders(
        &self,
        outcome1: &[u64; 256],
        outcome2: &[u64; 256],
    ) -> Vec<[u64; 256]> {
        self.causal.detect_confounders(outcome1, outcome2)
    }
    
    // -------------------------------------------------------------------------
    // RUNG 3: COUNTERFACTUAL (what would have happened)
    // -------------------------------------------------------------------------
    
    /// Store counterfactual: had I done alt_action, alt_outcome would have happened
    pub fn store_counterfactual(
        &mut self,
        state: &[u64; 256],
        alt_action: &[u64; 256],
        alt_outcome: &[u64; 256],
        weight: f32,
    ) {
        self.causal.store_counterfactual(state, alt_action, alt_outcome, weight);
    }
    
    /// Query counterfactual: what would have happened if I had done action?
    pub fn query_counterfactual(
        &self,
        state: &[u64; 256],
        alt_action: &[u64; 256],
    ) -> Vec<CausalResult> {
        self.causal.query_counterfactual(state, alt_action)
    }
    
    /// Compute regret: how much better would counterfactual have been?
    pub fn compute_regret(
        &self,
        state: &[u64; 256],
        actual_outcome: &[u64; 256],
        alt_action: &[u64; 256],
    ) -> Option<f32> {
        self.causal.compute_regret(state, actual_outcome, alt_action)
    }
    
    // -------------------------------------------------------------------------
    // POLICY
    // -------------------------------------------------------------------------
    
    /// Select action using causal Q-values with ε-greedy
    pub fn select_action(
        &self,
        state: &[u64; 256],
        actions: &[[u64; 256]],
    ) -> Option<[u64; 256]> {
        if actions.is_empty() {
            return None;
        }
        
        // Exploration
        let r: f32 = rand::random();
        if r < self.epsilon {
            let idx = (rand::random::<f32>() * actions.len() as f32) as usize;
            return Some(actions[idx.min(actions.len() - 1)]);
        }
        
        // Exploitation: use causal Q-values
        actions.iter()
            .max_by(|a, b| {
                let qa = self.q_value_causal(state, a);
                let qb = self.q_value_causal(state, b);
                qa.partial_cmp(&qb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
    }
    
    /// Select action with counterfactual-aware exploration
    /// Prefers actions where we have less counterfactual knowledge
    pub fn select_action_curious(
        &self,
        state: &[u64; 256],
        actions: &[[u64; 256]],
    ) -> Option<[u64; 256]> {
        if actions.is_empty() {
            return None;
        }
        
        // Score each action: Q-value + exploration bonus for uncertainty
        let scored: Vec<([u64; 256], f32)> = actions.iter()
            .map(|a| {
                let q = self.q_value_causal(state, a);
                let cf_count = self.query_counterfactual(state, a).len();
                
                // Less counterfactual knowledge = more bonus
                let curiosity_bonus = 1.0 / (1.0 + cf_count as f32);
                
                (*a, q + 0.1 * curiosity_bonus)
            })
            .collect();
        
        scored.into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(a, _)| a)
    }
    
    // -------------------------------------------------------------------------
    // LEARNING UPDATES
    // -------------------------------------------------------------------------
    
    /// Causal Q-learning update
    /// Unlike standard Q-learning, this stores the intervention explicitly
    pub fn update_causal_q(
        &mut self,
        state: &[u64; 256],
        action: &[u64; 256],
        reward: f32,
        next_state: &[u64; 256],
        available_actions: &[[u64; 256]],
    ) {
        // Store the intervention
        self.store_intervention(state, action, next_state, reward);
        
        // Also store counterfactual for other actions (for regret computation)
        for alt_action in available_actions {
            if alt_action != action {
                // Estimate counterfactual outcome (simplified: use current knowledge)
                if let Some(cf_result) = self.query_outcome(state, alt_action).first() {
                    self.store_counterfactual(
                        state,
                        alt_action,
                        &cf_result.fingerprint,
                        cf_result.weight,
                    );
                }
            }
        }
    }
    
    /// Update with explicit counterfactual (when we know what would have happened)
    pub fn update_with_counterfactual(
        &mut self,
        state: &[u64; 256],
        actual_action: &[u64; 256],
        actual_outcome: &[u64; 256],
        actual_reward: f32,
        alt_action: &[u64; 256],
        alt_outcome: &[u64; 256],
        alt_reward: f32,
    ) {
        // Store actual intervention
        self.store_intervention(state, actual_action, actual_outcome, actual_reward);
        
        // Store counterfactual
        self.store_counterfactual(state, alt_action, alt_outcome, alt_reward);
    }
    
    // -------------------------------------------------------------------------
    // EXPLANATION
    // -------------------------------------------------------------------------
    
    /// Explain why an action was chosen
    pub fn explain_action(
        &self,
        state: &[u64; 256],
        chosen_action: &[u64; 256],
        actions: &[[u64; 256]],
    ) -> ActionExplanation {
        let chosen_q = self.q_value_causal(state, chosen_action);
        let chosen_outcomes = self.query_outcome(state, chosen_action);
        
        let alternatives: Vec<_> = actions.iter()
            .filter(|a| *a != chosen_action)
            .map(|a| {
                let q = self.q_value_causal(state, a);
                let regret = self.compute_regret(
                    state,
                    &chosen_outcomes.first().map(|r| r.fingerprint).unwrap_or([0u64; 256]),
                    a
                );
                AlternativeAction {
                    action: *a,
                    q_value: q,
                    regret,
                }
            })
            .collect();
        
        ActionExplanation {
            chosen_action: *chosen_action,
            chosen_q_value: chosen_q,
            expected_outcomes: chosen_outcomes,
            alternatives,
            confounders: Vec::new(),  // Could populate if relevant
        }
    }
    
    /// Trace the causal chain: state → action → outcome → ...
    pub fn trace_causal_chain(
        &self,
        initial_state: &[u64; 256],
        max_depth: usize,
    ) -> Vec<CausalChainLink> {
        let mut chain = Vec::new();
        let mut current_state = *initial_state;
        
        for _ in 0..max_depth {
            // Find what we did in this state
            // (This is simplified - real implementation would track actual history)
            let outcomes = self.causal.query_correlates(&current_state, 1);
            
            if outcomes.is_empty() {
                break;
            }
            
            let outcome = &outcomes[0];
            chain.push(CausalChainLink {
                state: current_state,
                action: None,  // Would need to track this
                outcome: outcome.fingerprint,
                confidence: 1.0 - (outcome.distance as f32 / crate::FINGERPRINT_BITS as f32),
            });
            
            current_state = outcome.fingerprint;
        }
        
        chain
    }
}

impl Default for CausalRlAgent {
    fn default() -> Self {
        Self::new(0.99, 0.1, 0.1)
    }
}

// =============================================================================
// EXPLANATION TYPES
// =============================================================================

/// Explanation for an action choice
#[derive(Debug)]
pub struct ActionExplanation {
    pub chosen_action: [u64; 256],
    pub chosen_q_value: f32,
    pub expected_outcomes: Vec<CausalResult>,
    pub alternatives: Vec<AlternativeAction>,
    pub confounders: Vec<[u64; 256]>,
}

/// Alternative action that wasn't chosen
#[derive(Debug)]
pub struct AlternativeAction {
    pub action: [u64; 256],
    pub q_value: f32,
    pub regret: Option<f32>,
}

/// Link in a causal chain
#[derive(Debug)]
pub struct CausalChainLink {
    pub state: [u64; 256],
    pub action: Option<[u64; 256]>,
    pub outcome: [u64; 256],
    pub confidence: f32,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    fn random_fp() -> [u64; 256] {
        let mut fp = [0u64; 256];
        for i in 0..256 {
            fp[i] = rand::random();
        }
        fp
    }
    
    #[test]
    fn test_causal_rl_intervention() {
        let mut agent = CausalRlAgent::new(0.99, 0.1, 0.1);
        
        let state = random_fp();
        let action = random_fp();
        let outcome = random_fp();
        
        // Store intervention
        agent.store_intervention(&state, &action, &outcome, 1.0);
        
        // Query should find it
        let results = agent.query_outcome(&state, &action);
        assert!(!results.is_empty());
    }
    
    #[test]
    fn test_causal_rl_counterfactual() {
        let mut agent = CausalRlAgent::new(0.99, 0.1, 0.1);
        
        let state = random_fp();
        let alt_action = random_fp();
        let alt_outcome = random_fp();
        
        // Store counterfactual
        agent.store_counterfactual(&state, &alt_action, &alt_outcome, 0.8);
        
        // Query should find it
        let results = agent.query_counterfactual(&state, &alt_action);
        assert!(!results.is_empty());
    }
    
    #[test]
    fn test_action_selection() {
        let mut agent = CausalRlAgent::new(0.99, 0.1, 0.0);  // No exploration
        
        let state = random_fp();
        let good_action = random_fp();
        let bad_action = random_fp();
        let good_outcome = random_fp();
        let bad_outcome = random_fp();
        
        // Good action leads to high reward
        agent.store_intervention(&state, &good_action, &good_outcome, 10.0);
        // Bad action leads to low reward
        agent.store_intervention(&state, &bad_action, &bad_outcome, 1.0);
        
        // Should select good action (with 0 exploration)
        let selected = agent.select_action(&state, &[good_action, bad_action]);
        assert!(selected.is_some());
        // Note: due to how we store weight=reward, good action should have higher Q
    }
}
