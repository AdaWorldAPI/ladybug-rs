//! Cognitive Frameworks
//!
//! Complete implementations of cognitive architectures as fingerprint operations:
//! - NARS: Non-Axiomatic Reasoning System (inference, truth values, beliefs)
//! - ACT-R: Adaptive Control of Thought (buffers, chunks, productions)
//! - RL: Reinforcement Learning (value, policy, reward)
//! - Causality: Pearl's do-calculus (intervention, counterfactual)
//! - Qualia: Affect/experience channels
//! - Rung: Abstraction ladder (0-9 meaning depth)

use crate::core::Fingerprint;

// =============================================================================
// NARS - Non-Axiomatic Reasoning System
// =============================================================================

/// NARS truth value: <frequency, confidence>
#[derive(Clone, Copy, Debug)]
pub struct TruthValue {
    /// Frequency: positive evidence / total evidence
    pub f: f32,
    /// Confidence: total evidence / (total evidence + k)
    pub c: f32,
}

impl TruthValue {
    pub fn new(f: f32, c: f32) -> Self {
        Self {
            f: f.clamp(0.0, 1.0),
            c: c.clamp(0.0, 1.0),
        }
    }

    /// Expectation: E = c * (f - 0.5) + 0.5
    pub fn expectation(&self) -> f32 {
        self.c * (self.f - 0.5) + 0.5
    }

    /// Evidence counts: w+ = k * f * c / (1 - c)
    pub fn positive_evidence(&self, k: f32) -> f32 {
        if self.c >= 1.0 {
            return f32::INFINITY;
        }
        k * self.f * self.c / (1.0 - self.c)
    }

    /// Total evidence: w = k * c / (1 - c)
    pub fn total_evidence(&self, k: f32) -> f32 {
        if self.c >= 1.0 {
            return f32::INFINITY;
        }
        k * self.c / (1.0 - self.c)
    }

    /// From evidence counts
    pub fn from_evidence(positive: f32, total: f32, k: f32) -> Self {
        let f = if total > 0.0 { positive / total } else { 0.5 };
        let c = total / (total + k);
        Self::new(f, c)
    }
}

/// NARS copula types
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NarsCopula {
    Inheritance,  // S --> P
    Similarity,   // S <-> P
    Implication,  // S ==> P
    Equivalence,  // S <=> P
    Instance,     // {S} --> P
    Property,     // S --> [P]
    InstanceProp, // {S} --> [P]
    PredImpl,     // S =/> P (predictive)
    RetrImpl,     // S =\> P (retrospective)
    ConcImpl,     // S =|> P (concurrent)
}

impl NarsCopula {
    pub fn fingerprint(&self) -> Fingerprint {
        Fingerprint::from_content(&format!("NARS_COPULA::{:?}", self))
    }
}

/// NARS inference rules
pub struct NarsInference;

impl NarsInference {
    const K: f32 = 1.0; // Confidence horizon

    /// Deduction: {M --> P, S --> M} |- S --> P
    pub fn deduction(premise1: TruthValue, premise2: TruthValue) -> TruthValue {
        let f = premise1.f * premise2.f;
        let c = premise1.c * premise2.c * premise1.f * premise2.f;
        TruthValue::new(f, c)
    }

    /// Induction: {M --> P, M --> S} |- S --> P
    pub fn induction(premise1: TruthValue, premise2: TruthValue) -> TruthValue {
        let w_plus = premise1.f * premise2.f * premise1.c * premise2.c;
        let w = premise2.f * premise2.c;
        TruthValue::from_evidence(w_plus, w, Self::K)
    }

    /// Abduction: {P --> M, S --> M} |- S --> P
    pub fn abduction(premise1: TruthValue, premise2: TruthValue) -> TruthValue {
        let w_plus = premise1.f * premise2.f * premise1.c * premise2.c;
        let w = premise1.f * premise1.c;
        TruthValue::from_evidence(w_plus, w, Self::K)
    }

    /// Revision: combine evidence from independent sources
    pub fn revision(belief1: TruthValue, belief2: TruthValue) -> TruthValue {
        let w1 = belief1.total_evidence(Self::K);
        let w2 = belief2.total_evidence(Self::K);
        let w1_plus = belief1.positive_evidence(Self::K);
        let w2_plus = belief2.positive_evidence(Self::K);

        TruthValue::from_evidence(w1_plus + w2_plus, w1 + w2, Self::K)
    }

    /// Negation: --S
    pub fn negation(truth: TruthValue) -> TruthValue {
        TruthValue::new(1.0 - truth.f, truth.c)
    }

    /// Intersection: S && P
    pub fn intersection(truth1: TruthValue, truth2: TruthValue) -> TruthValue {
        TruthValue::new(truth1.f * truth2.f, truth1.c * truth2.c)
    }

    /// Union: S || P
    pub fn union(truth1: TruthValue, truth2: TruthValue) -> TruthValue {
        let f = truth1.f + truth2.f - truth1.f * truth2.f;
        TruthValue::new(f, truth1.c * truth2.c)
    }

    /// Choice: select belief with higher expectation
    pub fn choice(belief1: TruthValue, belief2: TruthValue) -> TruthValue {
        if belief1.expectation() >= belief2.expectation() {
            belief1
        } else {
            belief2
        }
    }

    /// Decision: act if expectation > threshold
    pub fn decision(truth: TruthValue, threshold: f32) -> bool {
        truth.expectation() > threshold
    }
}

/// NARS statement (term copula term with truth value)
#[derive(Clone, Debug)]
pub struct NarsStatement {
    pub subject: Fingerprint,
    pub copula: NarsCopula,
    pub predicate: Fingerprint,
    pub truth: TruthValue,
    pub fingerprint: Fingerprint,
}

impl NarsStatement {
    pub fn new(
        subject: Fingerprint,
        copula: NarsCopula,
        predicate: Fingerprint,
        truth: TruthValue,
    ) -> Self {
        // Statement fingerprint = subject ⊗ copula ⊗ predicate
        let copula_fp = copula.fingerprint();
        let fingerprint = subject.bind(&copula_fp).bind(&predicate);

        Self {
            subject,
            copula,
            predicate,
            truth,
            fingerprint,
        }
    }
}

// =============================================================================
// ACT-R - Adaptive Control of Thought - Rational
// =============================================================================

/// ACT-R buffer types
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ActrBuffer {
    Goal,      // Current goal
    Retrieval, // Retrieved chunk
    Visual,    // Visual percept
    Aural,     // Auditory percept
    Manual,    // Motor action
    Vocal,     // Speech action
    Imaginal,  // Imaginal representation
    Temporal,  // Temporal tracking
}

impl ActrBuffer {
    pub fn fingerprint(&self) -> Fingerprint {
        Fingerprint::from_content(&format!("ACTR_BUFFER::{:?}", self))
    }
}

/// ACT-R chunk - a structured unit of declarative memory
#[derive(Clone, Debug)]
pub struct ActrChunk {
    /// Chunk type
    pub chunk_type: Fingerprint,

    /// Slot-value pairs (slot fingerprint -> value fingerprint)
    pub slots: Vec<(Fingerprint, Fingerprint)>,

    /// Chunk fingerprint (computed from type and slots)
    pub fingerprint: Fingerprint,

    /// Base-level activation
    pub activation: f32,

    /// Creation time
    pub created_at: u64,

    /// Access times (for base-level learning)
    pub access_times: Vec<u64>,
}

impl ActrChunk {
    pub fn new(chunk_type: Fingerprint) -> Self {
        Self {
            chunk_type: chunk_type.clone(),
            slots: Vec::new(),
            fingerprint: chunk_type,
            activation: 0.0,
            created_at: now_millis(),
            access_times: Vec::new(),
        }
    }

    pub fn with_slot(mut self, slot: Fingerprint, value: Fingerprint) -> Self {
        self.slots.push((slot, value));
        self.recompute_fingerprint();
        self
    }

    fn recompute_fingerprint(&mut self) {
        let mut fp = self.chunk_type.clone();
        for (slot, value) in &self.slots {
            fp = fp.bind(slot).bind(value);
        }
        self.fingerprint = fp;
    }

    /// Base-level activation (simplified): B = ln(n) - d * ln(t)
    pub fn base_level_activation(&self, current_time: u64, decay: f32) -> f32 {
        if self.access_times.is_empty() {
            return self.activation;
        }

        let n = self.access_times.len() as f32;
        let t = (current_time - self.created_at) as f32 / 1000.0 + 1.0;

        n.ln() - decay * t.ln()
    }

    /// Record an access
    pub fn record_access(&mut self) {
        self.access_times.push(now_millis());
    }

    /// Get slot value
    pub fn get_slot(&self, slot: &Fingerprint) -> Option<&Fingerprint> {
        self.slots
            .iter()
            .find(|(s, _)| s.similarity(slot) > 0.9)
            .map(|(_, v)| v)
    }

    /// Partial match score
    pub fn partial_match(&self, pattern: &ActrChunk, mismatch_penalty: f32) -> f32 {
        let mut score = 0.0;

        // Type match
        let type_sim = self.chunk_type.similarity(&pattern.chunk_type);
        score += type_sim;

        // Slot matches
        for (slot, value) in &pattern.slots {
            if let Some(my_value) = self.get_slot(slot) {
                let sim = my_value.similarity(value);
                if sim > 0.9 {
                    score += 1.0;
                } else {
                    score -= mismatch_penalty * (1.0 - sim);
                }
            } else {
                score -= mismatch_penalty;
            }
        }

        score
    }
}

/// ACT-R production rule
#[derive(Clone, Debug)]
pub struct ActrProduction {
    /// Production name
    pub name: String,

    /// Condition fingerprint (pattern to match in buffers)
    pub condition: Fingerprint,

    /// Action fingerprint (modifications to make)
    pub action: Fingerprint,

    /// Utility value
    pub utility: f32,

    /// Fingerprint of the production
    pub fingerprint: Fingerprint,
}

impl ActrProduction {
    pub fn new(name: &str, condition: Fingerprint, action: Fingerprint) -> Self {
        let fp = Fingerprint::from_content(&format!("PROD::{}", name))
            .bind(&condition)
            .bind(&action);

        Self {
            name: name.to_string(),
            condition,
            action,
            utility: 0.0,
            fingerprint: fp,
        }
    }

    pub fn with_utility(mut self, utility: f32) -> Self {
        self.utility = utility;
        self
    }

    /// Check if production matches current state
    pub fn matches(&self, state: &Fingerprint) -> f32 {
        state.similarity(&self.condition)
    }
}

// =============================================================================
// RL - Reinforcement Learning
// =============================================================================

/// RL state-action pair
#[derive(Clone, Debug)]
pub struct StateAction {
    pub state: Fingerprint,
    pub action: Fingerprint,
    pub fingerprint: Fingerprint,
}

impl StateAction {
    pub fn new(state: Fingerprint, action: Fingerprint) -> Self {
        let fp = state.bind(&action);
        Self {
            state,
            action,
            fingerprint: fp,
        }
    }
}

/// Q-value entry
#[derive(Clone, Debug)]
pub struct QValue {
    pub state_action: StateAction,
    pub value: f32,
    pub visits: u32,
}

/// RL agent with fingerprint-based state/action representation
pub struct RlAgent {
    /// Q-values: state_action_hash -> QValue
    q_values: std::collections::HashMap<u64, QValue>,

    /// Learning rate
    pub alpha: f32,

    /// Discount factor
    pub gamma: f32,

    /// Exploration rate
    pub epsilon: f32,
}

impl RlAgent {
    pub fn new(alpha: f32, gamma: f32, epsilon: f32) -> Self {
        Self {
            q_values: std::collections::HashMap::new(),
            alpha,
            gamma,
            epsilon,
        }
    }

    /// Get Q-value for state-action pair
    pub fn get_q(&self, state: &Fingerprint, action: &Fingerprint) -> f32 {
        let sa = StateAction::new(state.clone(), action.clone());
        let hash = fp_hash(&sa.fingerprint);
        self.q_values.get(&hash).map_or(0.0, |q| q.value)
    }

    /// Update Q-value (Q-learning)
    pub fn update_q(
        &mut self,
        state: &Fingerprint,
        action: &Fingerprint,
        reward: f32,
        next_state: &Fingerprint,
        available_actions: &[Fingerprint],
    ) {
        let sa = StateAction::new(state.clone(), action.clone());
        let hash = fp_hash(&sa.fingerprint);

        // Max Q for next state
        let max_next_q = available_actions
            .iter()
            .map(|a| self.get_q(next_state, a))
            .fold(f32::NEG_INFINITY, f32::max);
        let max_next_q = if max_next_q.is_finite() {
            max_next_q
        } else {
            0.0
        };

        // TD update
        let current_q = self.get_q(state, action);
        let td_error = reward + self.gamma * max_next_q - current_q;
        let new_q = current_q + self.alpha * td_error;

        // Store
        let entry = self.q_values.entry(hash).or_insert_with(|| QValue {
            state_action: sa,
            value: 0.0,
            visits: 0,
        });
        entry.value = new_q;
        entry.visits += 1;
    }

    /// Select action (ε-greedy)
    pub fn select_action(
        &self,
        state: &Fingerprint,
        actions: &[Fingerprint],
    ) -> Option<Fingerprint> {
        if actions.is_empty() {
            return None;
        }

        // Exploration
        let r: f32 = rand_f32();
        if r < self.epsilon {
            let idx = (rand_f32() * actions.len() as f32) as usize;
            return Some(actions[idx.min(actions.len() - 1)].clone());
        }

        // Exploitation
        actions
            .iter()
            .max_by(|a, b| {
                let qa = self.get_q(state, a);
                let qb = self.get_q(state, b);
                qa.partial_cmp(&qb).unwrap()
            })
            .cloned()
    }

    /// Compute TD error
    pub fn td_error(
        &self,
        state: &Fingerprint,
        action: &Fingerprint,
        reward: f32,
        next_state: &Fingerprint,
        next_action: &Fingerprint,
    ) -> f32 {
        let current_q = self.get_q(state, action);
        let next_q = self.get_q(next_state, next_action);
        reward + self.gamma * next_q - current_q
    }
}

// =============================================================================
// CAUSALITY - Pearl's Do-Calculus
// =============================================================================

/// Causal relation types
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CausalRelation {
    Causes,    // X causes Y
    Enables,   // X enables Y (necessary but not sufficient)
    Prevents,  // X prevents Y
    Maintains, // X maintains Y
    Triggers,  // X triggers Y (sufficient but not necessary)
    Modulates, // X modulates Y
}

/// Causal graph node
#[derive(Clone, Debug)]
pub struct CausalNode {
    pub fingerprint: Fingerprint,
    pub name: Option<String>,
    pub observed: bool,
}

/// Causal graph edge
#[derive(Clone, Debug)]
pub struct CausalEdge {
    pub from: Fingerprint,
    pub to: Fingerprint,
    pub relation: CausalRelation,
    pub strength: f32,
}

/// do-operator: P(Y | do(X=x))
#[derive(Clone, Debug)]
pub struct DoOperator {
    /// Variable being intervened on
    pub variable: Fingerprint,
    /// Value to set
    pub value: Fingerprint,
}

impl DoOperator {
    pub fn new(variable: Fingerprint, value: Fingerprint) -> Self {
        Self { variable, value }
    }

    /// Fingerprint of the intervention
    pub fn fingerprint(&self) -> Fingerprint {
        let do_marker = Fingerprint::from_content("DO_OPERATOR");
        do_marker.bind(&self.variable).bind(&self.value)
    }
}

/// Counterfactual query: Y_x (Y under intervention do(X=x))
#[derive(Clone, Debug)]
pub struct Counterfactual {
    /// Outcome variable
    pub outcome: Fingerprint,
    /// Intervention
    pub intervention: DoOperator,
    /// Observed context
    pub context: Vec<(Fingerprint, Fingerprint)>,
}

impl Counterfactual {
    pub fn fingerprint(&self) -> Fingerprint {
        let cf_marker = Fingerprint::from_content("COUNTERFACTUAL");
        let mut fp = cf_marker
            .bind(&self.outcome)
            .bind(&self.intervention.fingerprint());
        for (var, val) in &self.context {
            fp = fp.bind(var).bind(val);
        }
        fp
    }
}

// =============================================================================
// QUALIA - Affect Channels
// =============================================================================

/// 8 qualia channels (Russell's circumplex + extensions)
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QualiaChannel {
    Arousal = 0,     // Activation level (low..high)
    Valence = 1,     // Hedonic tone (negative..positive)
    Tension = 2,     // Stress level (relaxed..tense)
    Certainty = 3,   // Epistemic (doubt..confidence)
    Agency = 4,      // Control (helpless..empowered)
    Temporality = 5, // Time pressure (patient..urgent)
    Sociality = 6,   // Social orientation (avoidant..approach)
    Novelty = 7,     // Pattern deviation (familiar..surprising)
}

impl QualiaChannel {
    pub fn fingerprint(&self) -> Fingerprint {
        Fingerprint::from_content(&format!("QUALIA::{:?}", self))
    }

    pub fn all() -> [QualiaChannel; 8] {
        [
            QualiaChannel::Arousal,
            QualiaChannel::Valence,
            QualiaChannel::Tension,
            QualiaChannel::Certainty,
            QualiaChannel::Agency,
            QualiaChannel::Temporality,
            QualiaChannel::Sociality,
            QualiaChannel::Novelty,
        ]
    }
}

/// Qualia state - 8-dimensional affect vector
#[derive(Clone, Debug)]
pub struct QualiaState {
    /// Values for each channel (0.0 to 1.0, 0.5 = neutral)
    pub channels: [f32; 8],

    /// Fingerprint encoding the state
    pub fingerprint: Fingerprint,
}

impl QualiaState {
    pub fn neutral() -> Self {
        Self::from_values([0.5; 8])
    }

    pub fn from_values(channels: [f32; 8]) -> Self {
        // Clamp values
        let channels: [f32; 8] = std::array::from_fn(|i| channels[i].clamp(0.0, 1.0));

        // Compute fingerprint by blending channel fingerprints
        let mut fp = Fingerprint::zero();
        for (i, &value) in channels.iter().enumerate() {
            let channel = QualiaChannel::all()[i];
            let channel_fp = channel.fingerprint();

            // Weight channel contribution by deviation from neutral
            let weight = (value - 0.5).abs() * 2.0;
            if weight > 0.01 {
                // Permute by channel index and blend
                let permuted = channel_fp.permute((i as i32) * 100);
                fp = weighted_blend(&fp, &permuted, weight);
            }
        }

        Self {
            channels,
            fingerprint: fp,
        }
    }

    pub fn get(&self, channel: QualiaChannel) -> f32 {
        self.channels[channel as usize]
    }

    pub fn set(&mut self, channel: QualiaChannel, value: f32) {
        self.channels[channel as usize] = value.clamp(0.0, 1.0);
        *self = Self::from_values(self.channels);
    }

    /// Blend two qualia states
    pub fn blend(&self, other: &QualiaState, weight: f32) -> QualiaState {
        let channels: [f32; 8] =
            std::array::from_fn(|i| self.channels[i] * (1.0 - weight) + other.channels[i] * weight);
        Self::from_values(channels)
    }

    /// Distance to another state
    pub fn distance(&self, other: &QualiaState) -> f32 {
        let sum: f32 = self
            .channels
            .iter()
            .zip(other.channels.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        sum.sqrt()
    }

    /// Classify basic emotion (simplified)
    pub fn classify_emotion(&self) -> &'static str {
        let arousal = self.get(QualiaChannel::Arousal);
        let valence = self.get(QualiaChannel::Valence);

        match (arousal > 0.5, valence > 0.5) {
            (true, true) => "excited/happy",
            (true, false) => "angry/afraid",
            (false, true) => "calm/content",
            (false, false) => "sad/depressed",
        }
    }
}

impl Default for QualiaState {
    fn default() -> Self {
        Self::neutral()
    }
}

// =============================================================================
// RUNG - Abstraction Ladder
// =============================================================================

/// Rung levels (0-9 meaning depth)
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub enum Rung {
    Noise = 0,        // Random/meaningless
    Token = 1,        // Lexical unit
    Phrase = 2,       // Phrase/chunk
    Proposition = 3,  // Statement
    Argument = 4,     // Reasoning chain
    Narrative = 5,    // Story/sequence
    Worldview = 6,    // Belief system
    Paradigm = 7,     // Framework
    Episteme = 8,     // Knowledge structure
    Transcendent = 9, // Meta-level
}

impl Rung {
    pub fn fingerprint(&self) -> Fingerprint {
        Fingerprint::from_content(&format!("RUNG::{}", *self as u8))
    }

    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => Rung::Noise,
            1 => Rung::Token,
            2 => Rung::Phrase,
            3 => Rung::Proposition,
            4 => Rung::Argument,
            5 => Rung::Narrative,
            6 => Rung::Worldview,
            7 => Rung::Paradigm,
            8 => Rung::Episteme,
            _ => Rung::Transcendent,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Rung::Noise => "noise",
            Rung::Token => "token",
            Rung::Phrase => "phrase",
            Rung::Proposition => "proposition",
            Rung::Argument => "argument",
            Rung::Narrative => "narrative",
            Rung::Worldview => "worldview",
            Rung::Paradigm => "paradigm",
            Rung::Episteme => "episteme",
            Rung::Transcendent => "transcendent",
        }
    }

    /// Can ascend (abstract up)?
    pub fn can_ascend(&self) -> bool {
        (*self as u8) < 9
    }

    /// Can descend (concretize down)?
    pub fn can_descend(&self) -> bool {
        (*self as u8) > 0
    }

    /// Ascend one level
    pub fn ascend(&self) -> Rung {
        Rung::from_u8((*self as u8).saturating_add(1))
    }

    /// Descend one level
    pub fn descend(&self) -> Rung {
        Rung::from_u8((*self as u8).saturating_sub(1))
    }
}

/// Rung classifier - determine abstraction level of content
pub struct RungClassifier {
    /// Fingerprints for each rung level
    rung_fps: [Fingerprint; 10],
}

impl RungClassifier {
    pub fn new() -> Self {
        Self {
            rung_fps: std::array::from_fn(|i| Rung::from_u8(i as u8).fingerprint()),
        }
    }

    /// Classify content by rung level (simple heuristic)
    pub fn classify(&self, content: &Fingerprint, word_count: usize, has_structure: bool) -> Rung {
        // Simple heuristics
        if word_count == 0 {
            return Rung::Noise;
        }
        if word_count == 1 {
            return Rung::Token;
        }
        if word_count <= 5 {
            return Rung::Phrase;
        }
        if word_count <= 30 && !has_structure {
            return Rung::Proposition;
        }
        if has_structure {
            return Rung::Argument;
        }

        // For longer content, use fingerprint similarity
        let mut best_rung = Rung::Proposition;
        let mut best_sim = 0.0;

        for (i, rung_fp) in self.rung_fps.iter().enumerate() {
            let sim = content.similarity(rung_fp);
            if sim > best_sim {
                best_sim = sim;
                best_rung = Rung::from_u8(i as u8);
            }
        }

        best_rung
    }
}

impl Default for RungClassifier {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

fn fp_hash(fp: &Fingerprint) -> u64 {
    let raw = fp.as_raw();
    let mut hash = 0u64;
    for &word in raw.iter() {
        hash ^= word;
    }
    hash
}

fn now_millis() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

fn rand_f32() -> f32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    ((nanos % 1000000) as f32) / 1000000.0
}

/// Weighted blend of two fingerprints
fn weighted_blend(a: &Fingerprint, b: &Fingerprint, weight: f32) -> Fingerprint {
    // If weight > 0.5, favor b; otherwise favor a
    // Using XOR with probability proportional to weight
    let threshold = (weight * crate::FINGERPRINT_BITS as f32) as usize;

    let mut result = a.clone();
    for bit in 0..crate::FINGERPRINT_BITS {
        if bit < threshold && b.get_bit(bit) != a.get_bit(bit) {
            result.set_bit(bit, b.get_bit(bit));
        }
    }
    result
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nars_truth_value() {
        let tv = TruthValue::new(0.8, 0.9);

        // Expectation should be high
        assert!(tv.expectation() > 0.7);

        // Negation
        let neg = NarsInference::negation(tv);
        assert!((neg.f - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_nars_deduction() {
        // {M --> P <0.8, 0.9>, S --> M <0.9, 0.8>} |- S --> P
        let p1 = TruthValue::new(0.8, 0.9);
        let p2 = TruthValue::new(0.9, 0.8);

        let result = NarsInference::deduction(p1, p2);

        // Frequency should be product
        assert!((result.f - 0.72).abs() < 0.01);
        // Confidence should be lower
        assert!(result.c < p1.c);
    }

    #[test]
    fn test_nars_revision() {
        let b1 = TruthValue::new(0.6, 0.5);
        let b2 = TruthValue::new(0.8, 0.5);

        let revised = NarsInference::revision(b1, b2);

        // Revised should be between the two
        assert!(revised.f > b1.f && revised.f < b2.f);
        // Confidence should increase
        assert!(revised.c > b1.c);
    }

    #[test]
    fn test_actr_chunk() {
        let chunk_type = Fingerprint::from_content("PERSON");
        let name_slot = Fingerprint::from_content("name");
        let name_value = Fingerprint::from_content("Alice");

        let chunk = ActrChunk::new(chunk_type).with_slot(name_slot.clone(), name_value.clone());

        // Should be able to retrieve slot
        let retrieved = chunk.get_slot(&name_slot);
        assert!(retrieved.is_some());
        assert!(retrieved.unwrap().similarity(&name_value) > 0.99);
    }

    #[test]
    fn test_rl_agent() {
        let mut agent = RlAgent::new(0.1, 0.9, 0.1);

        let state = Fingerprint::from_content("state1");
        let action = Fingerprint::from_content("action1");
        let next_state = Fingerprint::from_content("state2");

        // Initial Q should be 0
        assert_eq!(agent.get_q(&state, &action), 0.0);

        // Update with reward
        agent.update_q(&state, &action, 1.0, &next_state, &[]);

        // Q should now be positive
        assert!(agent.get_q(&state, &action) > 0.0);
    }

    #[test]
    fn test_qualia_state() {
        let mut state = QualiaState::neutral();

        // Set high arousal, positive valence
        state.set(QualiaChannel::Arousal, 0.9);
        state.set(QualiaChannel::Valence, 0.8);

        assert_eq!(state.classify_emotion(), "excited/happy");

        // Blend with neutral
        let neutral = QualiaState::neutral();
        let blended = state.blend(&neutral, 0.5);

        // Should be half way
        assert!((blended.get(QualiaChannel::Arousal) - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_rung_levels() {
        assert!(Rung::Token < Rung::Phrase);
        assert!(Rung::Argument.can_ascend());
        assert!(Rung::Token.can_descend());
        assert!(!Rung::Noise.can_descend());

        assert_eq!(Rung::Proposition.ascend(), Rung::Argument);
        assert_eq!(Rung::Proposition.descend(), Rung::Phrase);
    }
}
