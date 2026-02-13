//! Meta-Orchestrator — Flow-Aware Coordination with Resonance Routing
//!
//! The MetaOrchestrator sits above CrewBridge and makes coordination
//! decisions based on:
//!
//! 1. **Personality resonance** — Hamming similarity on persona fingerprints
//! 2. **Flow state** — agents in flow get protected; stalled agents get help
//! 3. **Handover policy** — resonance thresholds that trigger delegation
//! 4. **Collective efficacy** — agents that work well together get preferred
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     META-ORCHESTRATOR                           │
//! │                                                                 │
//! │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
//! │   │ Handover │  │ Resonance│  │  Flow    │  │ Affinity │     │
//! │   │  Policy  │  │  Router  │  │ Monitor  │  │  Graph   │     │
//! │   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘     │
//! │        │              │              │              │           │
//! │        └──────────────┴──────────────┴──────────────┘           │
//! │                          │                                     │
//! │                    CrewBridge                                   │
//! │                          │                                     │
//! │          ┌────────────────┼──────────────┐                     │
//! │          ▼                ▼              ▼                     │
//! │     Agents(0x0C)   Blackboards(0x0E)  A2A(0x0F)              │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Affinity Graph
//!
//! Tracks which agent pairs collaborate well (measured by successful
//! handovers and joint task completion). This creates the "teamwork
//! as reward loop" — agents that resonate together get paired again.

use crate::orchestration::a2a::{A2AMessage, A2AProtocol, DeliveryStatus, MessageKind};
use crate::orchestration::blackboard_agent::BlackboardRegistry;
use crate::orchestration::handover::{
    FlowState, FlowTransition, HandoverAction, HandoverDecision, HandoverPolicy,
};
use crate::orchestration::persona::PersonaRegistry;
use crate::storage::bind_space::BindSpace;
use serde::{Deserialize, Serialize};

// =============================================================================
// AFFINITY EDGE
// =============================================================================

/// An edge in the affinity graph tracking how well two agents work together
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AffinityEdge {
    /// First agent slot
    pub agent_a: u8,
    /// Second agent slot
    pub agent_b: u8,
    /// Persona compatibility (static, from fingerprints)
    pub persona_resonance: f32,
    /// Successful handover count (dynamic, accumulated)
    pub successful_handovers: u32,
    /// Joint task completions (dynamic, accumulated)
    pub joint_completions: u32,
    /// Effective affinity (blended static + dynamic)
    pub effective_affinity: f32,
}

impl AffinityEdge {
    pub fn new(agent_a: u8, agent_b: u8, persona_resonance: f32) -> Self {
        let mut edge = Self {
            agent_a,
            agent_b,
            persona_resonance,
            successful_handovers: 0,
            joint_completions: 0,
            effective_affinity: 0.0,
        };
        edge.recompute_affinity();
        edge
    }

    /// Record a successful handover between these agents
    pub fn record_handover(&mut self) {
        self.successful_handovers += 1;
        self.recompute_affinity();
    }

    /// Record a joint task completion
    pub fn record_joint_completion(&mut self) {
        self.joint_completions += 1;
        self.recompute_affinity();
    }

    /// Recompute effective affinity from static + dynamic components
    fn recompute_affinity(&mut self) {
        // Blend: 60% persona resonance (static identity), 40% collaboration history
        let total = self.successful_handovers + self.joint_completions;
        let history_score = if total == 0 {
            0.5 // neutral prior — no interactions yet
        } else {
            // Start from neutral (0.5) and grow toward 1.0 with each interaction.
            // Each successful interaction adds from the prior.
            // Formula: 0.5 + 0.5 * (1 - 1/(1 + total * 0.5))
            0.5 + 0.5 * (1.0 - 1.0 / (1.0 + total as f32 * 0.5))
        };
        self.effective_affinity = 0.6 * self.persona_resonance + 0.4 * history_score;
    }
}

// =============================================================================
// ORCHESTRATOR EVENT
// =============================================================================

/// Events emitted by the meta-orchestrator for observability
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OrchestratorEvent {
    /// Handover decision was made
    HandoverDecided {
        source_slot: u8,
        decision: HandoverDecision,
    },
    /// Flow state changed for an agent
    FlowTransition(FlowTransition),
    /// Affinity updated between two agents
    AffinityUpdated {
        agent_a: u8,
        agent_b: u8,
        new_affinity: f32,
    },
    /// Task was routed to an agent based on resonance
    ResonanceRouted {
        task_description: String,
        chosen_slot: u8,
        resonance_score: f32,
    },
    /// Escalation: no suitable handover target found
    Escalation { source_slot: u8, reason: String },
}

// =============================================================================
// META-ORCHESTRATOR
// =============================================================================

/// The meta-orchestrator coordinates agents using personality resonance
/// and flow state awareness.
pub struct MetaOrchestrator {
    /// Handover policy configuration
    pub policy: HandoverPolicy,
    /// Per-agent flow states (indexed by slot)
    flow_states: Vec<(u8, FlowState)>,
    /// Affinity graph edges
    affinities: Vec<AffinityEdge>,
    /// Flow transition history
    transitions: Vec<FlowTransition>,
    /// Event log for observability
    events: Vec<OrchestratorEvent>,
    /// Global cycle counter
    cycle: u64,
}

impl MetaOrchestrator {
    pub fn new(policy: HandoverPolicy) -> Self {
        Self {
            policy,
            flow_states: Vec::new(),
            affinities: Vec::new(),
            transitions: Vec::new(),
            events: Vec::new(),
            cycle: 0,
        }
    }

    /// Initialize flow state for an agent
    pub fn register_agent(&mut self, slot: u8) {
        if !self.flow_states.iter().any(|(s, _)| *s == slot) {
            self.flow_states.push((slot, FlowState::default()));
        }
    }

    /// Get flow state for an agent
    pub fn flow_state(&self, slot: u8) -> Option<&FlowState> {
        self.flow_states
            .iter()
            .find(|(s, _)| *s == slot)
            .map(|(_, f)| f)
    }

    /// Update flow state from a gate evaluation result
    pub fn update_flow(&mut self, slot: u8, gate: crate::cognitive::GateState) {
        let previous = self
            .flow_states
            .iter()
            .find(|(s, _)| *s == slot)
            .map(|(_, f)| f.clone())
            .unwrap_or_default();

        let new_state = FlowState::from_gate(gate, &previous);

        // Record transition if state type changed
        let changed = std::mem::discriminant(&previous) != std::mem::discriminant(&new_state);
        if changed {
            let transition = FlowTransition {
                agent_slot: slot,
                from: previous.clone(),
                to: new_state.clone(),
                at_cycle: self.cycle,
                triggered_by: None,
            };
            self.events
                .push(OrchestratorEvent::FlowTransition(transition.clone()));
            self.transitions.push(transition);
        }

        // Update stored state
        if let Some(entry) = self.flow_states.iter_mut().find(|(s, _)| *s == slot) {
            entry.1 = new_state;
        } else {
            self.flow_states.push((slot, new_state));
        }
    }

    /// Build or update affinity edges from persona registry
    pub fn refresh_affinities(&mut self, personas: &PersonaRegistry) {
        let persona_list = personas.list();
        for i in 0..persona_list.len() {
            for j in (i + 1)..persona_list.len() {
                let (slot_a, persona_a) = &persona_list[i];
                let (slot_b, persona_b) = &persona_list[j];
                let resonance = persona_a.compatibility(persona_b);

                // Update or create edge
                let existing = self.affinities.iter_mut().find(|e| {
                    (e.agent_a == *slot_a && e.agent_b == *slot_b)
                        || (e.agent_a == *slot_b && e.agent_b == *slot_a)
                });

                if let Some(edge) = existing {
                    edge.persona_resonance = resonance;
                    edge.recompute_affinity();
                } else {
                    self.affinities
                        .push(AffinityEdge::new(*slot_a, *slot_b, resonance));
                }
            }
        }
    }

    /// Evaluate handover for a specific agent
    pub fn evaluate_handover(
        &mut self,
        source_slot: u8,
        blackboards: &BlackboardRegistry,
        personas: &PersonaRegistry,
        task_description: Option<&str>,
    ) -> HandoverDecision {
        let flow_state = self
            .flow_states
            .iter()
            .find(|(s, _)| *s == source_slot)
            .map(|(_, f)| f.clone())
            .unwrap_or_default();

        let coherence = blackboards
            .get(source_slot)
            .map(|bb| bb.awareness.coherence)
            .unwrap_or(0.5);

        let volition_alignment = match (personas.get(source_slot), task_description) {
            (Some(persona), Some(desc)) => persona.volition_alignment(desc),
            _ => 0.0,
        };

        // Estimate confidence from flow momentum + coherence
        let confidence = (flow_state.momentum() * 0.5 + coherence * 0.5).min(1.0);

        // Find best alternative: highest affinity agent that is NOT blocked
        let best_alternative =
            self.find_best_handover_target(source_slot, personas, task_description);

        let decision = self.policy.evaluate(
            source_slot,
            &flow_state,
            coherence,
            volition_alignment,
            confidence,
            best_alternative,
        );

        self.events.push(OrchestratorEvent::HandoverDecided {
            source_slot,
            decision: decision.clone(),
        });

        decision
    }

    /// Find the best handover target for a given source agent
    fn find_best_handover_target(
        &self,
        source_slot: u8,
        personas: &PersonaRegistry,
        task_description: Option<&str>,
    ) -> Option<(u8, f32)> {
        let source_persona = personas.get(source_slot)?;

        // Score all other agents by:
        // 1. Effective affinity (persona resonance + collaboration history)
        // 2. Volition alignment with task (if task description provided)
        // 3. Current flow state (prefer agents NOT blocked)
        let mut candidates: Vec<(u8, f32)> = Vec::new();

        for (slot, persona) in personas.list() {
            if *slot == source_slot {
                continue;
            }

            // Base score: persona compatibility
            let compatibility = source_persona.compatibility(persona);

            // Boost from affinity graph (collaboration history)
            let affinity_boost = self
                .affinities
                .iter()
                .find(|e| {
                    (e.agent_a == source_slot && e.agent_b == *slot)
                        || (e.agent_a == *slot && e.agent_b == source_slot)
                })
                .map(|e| e.effective_affinity - e.persona_resonance)
                .unwrap_or(0.0);

            // Volition alignment with task
            let task_boost = match task_description {
                Some(desc) => persona.volition_alignment(desc).max(0.0) * 0.2,
                None => 0.0,
            };

            // Penalty for blocked agents
            let flow_penalty = match self.flow_state(*slot) {
                Some(FlowState::Block { .. }) => -0.3,
                Some(FlowState::Handover { .. }) => -0.5, // already handing over
                _ => 0.0,
            };

            let total_score = compatibility + affinity_boost + task_boost + flow_penalty;
            candidates.push((*slot, total_score));
        }

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.first().copied()
    }

    /// Execute a handover: send A2A delegation message and update flow states
    pub fn execute_handover(
        &mut self,
        decision: &HandoverDecision,
        a2a: &mut A2AProtocol,
        space: &mut BindSpace,
    ) -> Option<DeliveryStatus> {
        match &decision.action {
            HandoverAction::Delegate {
                target_slot,
                resonance,
            } => {
                // Transition source to Handover state
                if let Some(entry) = self
                    .flow_states
                    .iter_mut()
                    .find(|(s, _)| *s == decision.source_slot)
                {
                    let previous = entry.1.clone();
                    entry.1 = FlowState::Handover {
                        target_slot: *target_slot,
                        resonance_score: *resonance,
                    };

                    self.transitions.push(FlowTransition {
                        agent_slot: decision.source_slot,
                        from: previous,
                        to: entry.1.clone(),
                        at_cycle: self.cycle,
                        triggered_by: Some(decision.clone()),
                    });
                }

                // Send A2A delegation message
                let msg = A2AMessage {
                    id: format!(
                        "handover-{}-{}-{}",
                        decision.source_slot, target_slot, self.cycle
                    ),
                    sender_slot: decision.source_slot,
                    receiver_slot: *target_slot,
                    kind: MessageKind::Delegate,
                    payload: format!(
                        "{{\"handover\":true,\"resonance\":{:.3},\"reasons\":{:?}}}",
                        resonance,
                        decision
                            .reasons
                            .iter()
                            .map(|r| format!("{:?}", r))
                            .collect::<Vec<_>>()
                    ),
                    fingerprint: None,
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                    status: DeliveryStatus::Pending,
                    thinking_style_hint: None,
                    resonance_weight: *resonance,
                };

                let status = a2a.send(msg, space);

                // Record in affinity graph
                if status == DeliveryStatus::Delivered {
                    if let Some(edge) = self.affinities.iter_mut().find(|e| {
                        (e.agent_a == decision.source_slot && e.agent_b == *target_slot)
                            || (e.agent_a == *target_slot && e.agent_b == decision.source_slot)
                    }) {
                        edge.record_handover();
                        self.events.push(OrchestratorEvent::AffinityUpdated {
                            agent_a: edge.agent_a,
                            agent_b: edge.agent_b,
                            new_affinity: edge.effective_affinity,
                        });
                    }
                }

                Some(status)
            }
            HandoverAction::Escalate => {
                self.events.push(OrchestratorEvent::Escalation {
                    source_slot: decision.source_slot,
                    reason: format!("{:?}", decision.reasons),
                });
                None
            }
            _ => None,
        }
    }

    /// Route a new task to the best available agent using resonance scoring
    pub fn route_task(
        &mut self,
        task_description: &str,
        personas: &PersonaRegistry,
    ) -> Option<(u8, f32)> {
        let mut candidates: Vec<(u8, f32)> = Vec::new();

        for (slot, persona) in personas.list() {
            // Volition alignment
            let alignment = persona.volition_alignment(task_description);

            // Flow availability (prefer agents in flow or hold, avoid blocked)
            let availability = match self.flow_state(*slot) {
                Some(FlowState::Flow { momentum }) => 0.5 + momentum * 0.3,
                Some(FlowState::Hold { hold_cycles }) => {
                    if *hold_cycles > 3 {
                        0.3
                    } else {
                        0.6
                    }
                }
                Some(FlowState::Block { .. }) => 0.1,
                Some(FlowState::Handover { .. }) => 0.0, // not available
                None => 0.5,                             // unknown
            };

            // Feature match — check if agent has relevant capabilities
            let feature_match: f32 = persona
                .features
                .iter()
                .filter(|f| {
                    task_description
                        .to_lowercase()
                        .contains(&f.name.to_lowercase())
                })
                .map(|f| f.proficiency)
                .sum::<f32>()
                .min(1.0);

            let total = alignment.max(0.0) * 0.4 + availability * 0.3 + feature_match * 0.3;
            candidates.push((*slot, total));
        }

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if let Some(&(slot, score)) = candidates.first() {
            if score > 0.0 {
                self.events.push(OrchestratorEvent::ResonanceRouted {
                    task_description: task_description.to_string(),
                    chosen_slot: slot,
                    resonance_score: score,
                });
                return Some((slot, score));
            }
        }

        None
    }

    /// Tick the orchestrator cycle — evaluate all agents
    pub fn tick(
        &mut self,
        blackboards: &BlackboardRegistry,
        personas: &PersonaRegistry,
    ) -> Vec<HandoverDecision> {
        self.cycle += 1;
        let slots: Vec<u8> = self.flow_states.iter().map(|(s, _)| *s).collect();

        let mut decisions = Vec::new();
        for slot in slots {
            let decision = self.evaluate_handover(slot, blackboards, personas, None);
            if decision.action != HandoverAction::Continue {
                decisions.push(decision);
            }
        }
        decisions
    }

    /// Get affinity between two agents
    pub fn affinity(&self, a: u8, b: u8) -> Option<f32> {
        self.affinities
            .iter()
            .find(|e| (e.agent_a == a && e.agent_b == b) || (e.agent_a == b && e.agent_b == a))
            .map(|e| e.effective_affinity)
    }

    /// Get all affinity edges
    pub fn affinities(&self) -> &[AffinityEdge] {
        &self.affinities
    }

    /// Get event log
    pub fn events(&self) -> &[OrchestratorEvent] {
        &self.events
    }

    /// Get flow transition history
    pub fn transitions(&self) -> &[FlowTransition] {
        &self.transitions
    }

    /// Current cycle
    pub fn cycle(&self) -> u64 {
        self.cycle
    }

    /// Status summary
    pub fn status(&self) -> OrchestratorStatus {
        let agents_in_flow = self.flow_states.iter().filter(|(_, f)| f.is_flow()).count();
        let agents_blocked = self
            .flow_states
            .iter()
            .filter(|(_, f)| f.is_blocked())
            .count();
        let agents_in_handover = self
            .flow_states
            .iter()
            .filter(|(_, f)| f.is_handover())
            .count();

        OrchestratorStatus {
            cycle: self.cycle,
            agents_tracked: self.flow_states.len(),
            agents_in_flow,
            agents_blocked,
            agents_in_handover,
            affinity_edges: self.affinities.len(),
            total_transitions: self.transitions.len(),
            total_events: self.events.len(),
        }
    }
}

impl Default for MetaOrchestrator {
    fn default() -> Self {
        Self::new(HandoverPolicy::default())
    }
}

/// Summary of orchestrator state
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OrchestratorStatus {
    pub cycle: u64,
    pub agents_tracked: usize,
    pub agents_in_flow: usize,
    pub agents_blocked: usize,
    pub agents_in_handover: usize,
    pub affinity_edges: usize,
    pub total_transitions: usize,
    pub total_events: usize,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orchestration::persona::{
        CommunicationStyle, FeatureAd, Persona, PersonalityTrait, VolitionDTO,
    };

    fn make_persona(drive: &str, curiosity: f32, affinities: Vec<&str>) -> Persona {
        Persona {
            volition: VolitionDTO {
                drive: drive.to_string(),
                curiosity,
                autonomy: 0.5,
                persistence: 0.7,
                risk_tolerance: 0.5,
                collaboration: 0.6,
                affinities: affinities.into_iter().map(|s| s.to_string()).collect(),
                aversions: Vec::new(),
            },
            traits: vec![PersonalityTrait {
                name: "openness".to_string(),
                value: curiosity,
                frozen: false,
            }],
            communication: CommunicationStyle::default(),
            preferred_styles: vec!["analytical".to_string()],
            features: vec![FeatureAd {
                name: drive.to_string(),
                proficiency: 0.9,
                preference: 0.8,
                cam_opcode: None,
            }],
        }
    }

    #[test]
    fn test_register_and_flow_update() {
        let mut orch = MetaOrchestrator::default();
        orch.register_agent(0);
        orch.register_agent(1);

        // Agent 0 enters flow
        orch.update_flow(0, crate::cognitive::GateState::Flow);
        assert!(orch.flow_state(0).unwrap().is_flow());

        // Agent 1 gets blocked
        orch.update_flow(1, crate::cognitive::GateState::Block);
        assert!(orch.flow_state(1).unwrap().is_blocked());
    }

    #[test]
    fn test_affinity_graph() {
        let mut orch = MetaOrchestrator::default();
        let mut personas = PersonaRegistry::new();

        let p1 = make_persona("research", 0.9, vec!["statistics"]);
        let p2 = make_persona("research", 0.8, vec!["statistics"]); // similar
        let p3 = make_persona("design", 0.3, vec!["art"]); // different

        personas.attach(0, p1);
        personas.attach(1, p2);
        personas.attach(2, p3);

        orch.refresh_affinities(&personas);

        // Agents 0 and 1 should have higher affinity than 0 and 2
        let affinity_01 = orch.affinity(0, 1).unwrap();
        let affinity_02 = orch.affinity(0, 2).unwrap();
        assert!(
            affinity_01 > affinity_02,
            "Similar agents should have higher affinity: {} vs {}",
            affinity_01,
            affinity_02
        );
    }

    #[test]
    fn test_route_task() {
        let mut orch = MetaOrchestrator::default();
        let mut personas = PersonaRegistry::new();

        personas.attach(0, make_persona("research", 0.9, vec!["statistics"]));
        personas.attach(1, make_persona("design", 0.3, vec!["art"]));

        orch.register_agent(0);
        orch.register_agent(1);

        // Research task should route to agent 0
        let result = orch.route_task("Analyze statistics data", &personas);
        assert!(result.is_some());
        assert_eq!(result.unwrap().0, 0);
    }

    #[test]
    fn test_handover_records_affinity() {
        let mut orch = MetaOrchestrator::default();
        orch.affinities.push(AffinityEdge::new(0, 1, 0.7));

        let initial = orch.affinity(0, 1).unwrap();

        // Record a handover
        orch.affinities[0].record_handover();
        let after = orch.affinity(0, 1).unwrap();

        // Affinity should increase after successful handover
        assert!(
            after > initial,
            "Affinity should increase after handover: {} -> {}",
            initial,
            after
        );
    }

    #[test]
    fn test_orchestrator_status() {
        let mut orch = MetaOrchestrator::default();
        orch.register_agent(0);
        orch.register_agent(1);
        orch.update_flow(0, crate::cognitive::GateState::Flow);
        orch.update_flow(1, crate::cognitive::GateState::Block);

        let status = orch.status();
        assert_eq!(status.agents_tracked, 2);
        assert_eq!(status.agents_in_flow, 1);
        assert_eq!(status.agents_blocked, 1);
    }
}
