//! Agent Blackboard — Per-agent ice-caked awareness via BindSpace 0x0E
//!
//! Each agent gets a dedicated blackboard slot in prefix 0x0E that mirrors
//! their agent slot in 0x0C. This provides persistent, agent-specific
//! state that survives across task executions.
//!
//! # Slot Mapping
//!
//! ```text
//! Agent at 0x0C:03 → Blackboard at 0x0E:03
//! Agent at 0x0C:10 → Blackboard at 0x0E:10
//! ```

use super::handover::FlowState;
use crate::cognitive::ThinkingStyle;
use crate::storage::bind_space::{
    Addr, BindSpace, FINGERPRINT_WORDS, PREFIX_AGENTS, PREFIX_BLACKBOARD,
};
use serde::{Deserialize, Serialize};

/// Agent awareness state — what the agent knows about itself and its context
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentAwareness {
    /// Current thinking style in use
    pub active_style: String,
    /// Coherence level (0.0-1.0) of current reasoning
    pub coherence: f32,
    /// Current task progress (0.0-1.0)
    pub progress: f32,
    /// Ice-caked decisions (frozen commitments)
    pub ice_caked: Vec<String>,
    /// Active goals from agent card
    pub active_goals: Vec<String>,
    /// Tools currently available
    pub available_tools: Vec<String>,
    /// Recent resonance hits (addresses that matched)
    pub resonance_hits: Vec<u16>,
    /// Number of A2A messages pending
    pub pending_messages: u32,
    /// Flow state — mirrors GateState semantics with momentum tracking
    pub flow_state: FlowState,
    /// Self-reported confidence in current task (0.0-1.0)
    /// Used by Dunning-Kruger guard: if confidence >> coherence → metacognitive review
    pub confidence: f32,
}

impl Default for AgentAwareness {
    fn default() -> Self {
        Self {
            active_style: "deliberate".to_string(),
            coherence: 0.0,
            progress: 0.0,
            ice_caked: Vec::new(),
            active_goals: Vec::new(),
            available_tools: Vec::new(),
            resonance_hits: Vec::new(),
            pending_messages: 0,
            flow_state: FlowState::default(),
            confidence: 0.5,
        }
    }
}

/// Per-agent blackboard stored at prefix 0x0E
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentBlackboard {
    /// Agent slot (mirrors 0x0C slot)
    pub agent_slot: u8,
    /// Agent identifier
    pub agent_id: String,
    /// Current awareness state
    pub awareness: AgentAwareness,
    /// Task history (last N completed tasks)
    pub task_history: Vec<TaskRecord>,
    /// Accumulated knowledge fingerprints (addresses learned)
    pub knowledge_addrs: Vec<u16>,
    /// Session cycle counter
    pub cycle: u64,
}

/// Record of a completed task
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaskRecord {
    pub task_id: String,
    pub description: String,
    pub outcome: String,
    pub thinking_style_used: String,
    pub cycle: u64,
}

impl AgentBlackboard {
    pub fn new(agent_slot: u8, agent_id: &str) -> Self {
        Self {
            agent_slot,
            agent_id: agent_id.to_string(),
            awareness: AgentAwareness::default(),
            task_history: Vec::new(),
            knowledge_addrs: Vec::new(),
            cycle: 0,
        }
    }

    /// BindSpace address for this blackboard
    pub fn addr(&self) -> Addr {
        Addr::new(PREFIX_BLACKBOARD, self.agent_slot)
    }

    /// Corresponding agent card address
    pub fn agent_addr(&self) -> Addr {
        Addr::new(PREFIX_AGENTS, self.agent_slot)
    }

    /// Update awareness with current thinking style
    pub fn set_thinking_style(&mut self, style: ThinkingStyle) {
        self.awareness.active_style = format!("{:?}", style).to_lowercase();
    }

    /// Record a completed task
    pub fn record_task(&mut self, task_id: &str, description: &str, outcome: &str, style: &str) {
        self.task_history.push(TaskRecord {
            task_id: task_id.to_string(),
            description: description.to_string(),
            outcome: outcome.to_string(),
            thinking_style_used: style.to_string(),
            cycle: self.cycle,
        });
        // Keep last 50 tasks
        if self.task_history.len() > 50 {
            self.task_history.remove(0);
        }
    }

    /// Ice-cake a decision (freeze a commitment)
    pub fn ice_cake(&mut self, decision: &str) {
        self.awareness.ice_caked.push(decision.to_string());
    }

    /// Learn an address (add to knowledge base)
    pub fn learn_address(&mut self, addr: u16) {
        if !self.knowledge_addrs.contains(&addr) {
            self.knowledge_addrs.push(addr);
        }
    }

    /// Update flow state from a gate evaluation
    pub fn update_flow(&mut self, gate: crate::cognitive::GateState) {
        self.awareness.flow_state = FlowState::from_gate(gate, &self.awareness.flow_state);
    }

    /// Set flow state directly (e.g., from MetaOrchestrator handover)
    pub fn set_flow(&mut self, state: FlowState) {
        self.awareness.flow_state = state;
    }

    /// Set confidence level
    pub fn set_confidence(&mut self, confidence: f32) {
        self.awareness.confidence = confidence.clamp(0.0, 1.0);
    }

    /// Check if the agent is in flow
    pub fn is_in_flow(&self) -> bool {
        self.awareness.flow_state.is_flow()
    }

    /// Check if the agent is blocked
    pub fn is_blocked(&self) -> bool {
        self.awareness.flow_state.is_blocked()
    }

    /// Get flow momentum (0.0 if not in flow)
    pub fn flow_momentum(&self) -> f32 {
        self.awareness.flow_state.momentum()
    }

    /// Dunning-Kruger gap: confidence - coherence
    /// Positive values indicate overconfidence relative to actual coherence
    pub fn dk_gap(&self) -> f32 {
        self.awareness.confidence - self.awareness.coherence
    }

    /// Advance cycle counter
    pub fn tick(&mut self) {
        self.cycle += 1;
    }

    /// Generate a state fingerprint for BindSpace storage.
    /// Encodes awareness + knowledge into a fingerprint that enables
    /// similarity search between agent states.
    pub fn state_fingerprint(&self) -> [u64; FINGERPRINT_WORDS] {
        use sha2::{Digest, Sha256};

        let state = format!(
            "{}:{}:{}:{}:{}:{:?}:{}",
            self.agent_id,
            self.awareness.active_style,
            self.awareness.coherence,
            self.cycle,
            self.knowledge_addrs.len(),
            self.awareness.flow_state,
            self.awareness.confidence,
        );

        let mut hasher = Sha256::new();
        hasher.update(state.as_bytes());
        let hash = hasher.finalize();

        let mut fp = [0u64; FINGERPRINT_WORDS];
        for (i, word) in fp.iter_mut().enumerate() {
            let mut h = Sha256::new();
            h.update(&hash);
            h.update(&(i as u32).to_le_bytes());
            let block = h.finalize();
            *word = u64::from_le_bytes(block[..8].try_into().unwrap());
        }

        // XOR in knowledge addresses for content-based matching
        for &addr in &self.knowledge_addrs {
            let word_idx = (addr as usize) % FINGERPRINT_WORDS;
            fp[word_idx] ^= (addr as u64).wrapping_mul(0x9E3779B97F4A7C15); // Fibonacci hashing
        }

        fp
    }

    /// Serialize to YAML for handover
    pub fn to_yaml(&self) -> String {
        serde_yml::to_string(self).unwrap_or_default()
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}

/// Registry managing all agent blackboards
pub struct BlackboardRegistry {
    blackboards: Vec<AgentBlackboard>,
}

impl BlackboardRegistry {
    pub fn new() -> Self {
        Self {
            blackboards: Vec::new(),
        }
    }

    /// Create a new blackboard for an agent
    pub fn create(&mut self, agent_slot: u8, agent_id: &str) -> &mut AgentBlackboard {
        let bb = AgentBlackboard::new(agent_slot, agent_id);
        self.blackboards.push(bb);
        self.blackboards.last_mut().unwrap()
    }

    /// Get blackboard by agent slot
    pub fn get(&self, agent_slot: u8) -> Option<&AgentBlackboard> {
        self.blackboards.iter().find(|b| b.agent_slot == agent_slot)
    }

    /// Get mutable blackboard by agent slot
    pub fn get_mut(&mut self, agent_slot: u8) -> Option<&mut AgentBlackboard> {
        self.blackboards
            .iter_mut()
            .find(|b| b.agent_slot == agent_slot)
    }

    /// Bind all blackboard states into BindSpace
    pub fn bind_all(&self, space: &mut BindSpace) {
        for bb in &self.blackboards {
            let addr = bb.addr();
            let fp = bb.state_fingerprint();
            space.write_at(addr, fp);
            if let Some(node) = space.read_mut(addr) {
                node.label = Some(format!("blackboard:{}", bb.agent_id));
            }
        }
    }

    /// List all blackboards
    pub fn list(&self) -> &[AgentBlackboard] {
        &self.blackboards
    }
}

impl Default for BlackboardRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blackboard_slot_mapping() {
        let bb = AgentBlackboard::new(0x03, "researcher");
        assert_eq!(bb.addr(), Addr::new(PREFIX_BLACKBOARD, 0x03));
        assert_eq!(bb.agent_addr(), Addr::new(PREFIX_AGENTS, 0x03));
    }

    #[test]
    fn test_ice_cake() {
        let mut bb = AgentBlackboard::new(0, "test");
        bb.ice_cake("Use analytical style for this task");
        assert_eq!(bb.awareness.ice_caked.len(), 1);
    }

    #[test]
    fn test_knowledge_learning() {
        let mut bb = AgentBlackboard::new(0, "test");
        bb.learn_address(0x8001);
        bb.learn_address(0x8001); // Duplicate
        bb.learn_address(0x8002);
        assert_eq!(bb.knowledge_addrs.len(), 2);
    }

    #[test]
    fn test_state_fingerprint_changes_with_state() {
        let mut bb = AgentBlackboard::new(0, "test");
        let fp1 = bb.state_fingerprint();
        bb.learn_address(0x8001);
        let fp2 = bb.state_fingerprint();
        assert_ne!(fp1, fp2);
    }

    #[test]
    fn test_yaml_roundtrip() {
        let bb = AgentBlackboard::new(5, "agent-5");
        let yaml = bb.to_yaml();
        assert!(yaml.contains("agent_slot: 5"));
        assert!(yaml.contains("agent-5"));
    }

    #[test]
    fn test_flow_state_integration() {
        let mut bb = AgentBlackboard::new(0, "test");
        assert!(!bb.is_in_flow());

        // Enter flow via gate
        bb.update_flow(crate::cognitive::GateState::Flow);
        assert!(bb.is_in_flow());
        assert!(bb.flow_momentum() > 0.0);

        // Accumulate momentum
        for _ in 0..5 {
            bb.update_flow(crate::cognitive::GateState::Flow);
        }
        assert!(bb.flow_momentum() >= 0.5);

        // Block
        bb.update_flow(crate::cognitive::GateState::Block);
        assert!(bb.is_blocked());
        assert!(!bb.is_in_flow());
    }

    #[test]
    fn test_dk_gap() {
        let mut bb = AgentBlackboard::new(0, "test");
        bb.awareness.coherence = 0.3;
        bb.set_confidence(0.9);
        // DK gap: 0.9 - 0.3 = 0.6 — overconfident
        assert!((bb.dk_gap() - 0.6).abs() < f32::EPSILON);
    }

    #[test]
    fn test_flow_state_affects_fingerprint() {
        let mut bb = AgentBlackboard::new(0, "test");
        let fp1 = bb.state_fingerprint();

        bb.update_flow(crate::cognitive::GateState::Flow);
        let fp2 = bb.state_fingerprint();
        assert_ne!(fp1, fp2);
    }
}
