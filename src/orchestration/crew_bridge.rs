//! Crew Bridge — Zero-copy crewAI integration via Arrow Flight
//!
//! This module bridges crewAI's Python orchestration with ladybug-rs's
//! cognitive substrate. It provides:
//!
//! 1. Task dispatch from crewAI crews to ladybug-rs agents
//! 2. Zero-copy Arrow Flight transport for fingerprint/knowledge transfer
//! 3. sci/v1 endpoint routing for research-capable agents
//! 4. Thinking style resolution from YAML templates
//!
//! # Integration Model
//!
//! ```text
//! crewAI (Python)                  ladybug-rs (Rust)
//! ─────────────────                ─────────────────────
//!
//! Crew.kickoff() ──► Arrow Flight DoAction("crew.dispatch") ──► CrewBridge
//!                                                                    │
//!   agents.yaml  ──► DoAction("crew.register_agent")  ──► AgentRegistry (0x0C)
//!   tasks.yaml   ──► DoAction("crew.submit_task")     ──► TaskQueue
//!   styles.yaml  ──► DoAction("crew.register_style")  ──► ThinkingTemplateRegistry (0x0D)
//!                                                                    │
//!   DoGet("agents")  ◄── zero-copy agent list                       │
//!   DoGet("sci:...")  ◄── sci/v1 statistical results                │
//!   DoAction("a2a.*") ◄──► agent-to-agent messaging (0x0F)         │
//! ```
//!
//! # Non-blocking Design
//!
//! The bridge does NOT embed crewAI logic. It provides endpoints that
//! crewAI's Python process calls via Arrow Flight. ladybug-rs stays
//! standalone — if crewAI is not connected, these endpoints simply
//! return empty results or are not called.

use super::a2a::{A2AMessage, A2AProtocol, DeliveryStatus};
use super::agent_card::{AgentCard, AgentRegistry};
use super::blackboard_agent::{AgentBlackboard, BlackboardRegistry};
use super::handover::{HandoverDecision, HandoverPolicy};
use super::kernel_extensions::{
    FilterPipeline, KernelGuardrail, MemoryBank, ObservabilityManager, VerificationEngine,
};
use super::meta_orchestrator::MetaOrchestrator;
use super::persona::{Persona, PersonaRegistry};
use super::semantic_kernel::SemanticKernel;
use super::thinking_template::{ThinkingTemplate, ThinkingTemplateRegistry};
use crate::storage::bind_space::{Addr, BindSpace, FINGERPRINT_WORDS};
use serde::{Deserialize, Serialize};

/// Task status in the dispatch pipeline
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum TaskStatus {
    Queued,
    Assigned,
    InProgress,
    Completed,
    Failed,
    Delegated,
}

/// A task submitted by crewAI for execution
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CrewTask {
    pub id: String,
    pub description: String,
    pub expected_output: String,
    /// Agent ID to assign this task to
    pub agent_id: Option<String>,
    /// Thinking style override for this task
    pub thinking_style: Option<String>,
    /// Whether to use sci/v1 validation
    pub require_validation: bool,
    /// Task dependencies (IDs that must complete first)
    pub depends_on: Vec<String>,
    pub status: TaskStatus,
    /// Assigned agent slot (set by dispatcher)
    #[serde(skip)]
    pub assigned_slot: Option<u8>,
}

/// Result from a dispatch operation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DispatchResult {
    pub task_id: String,
    pub status: TaskStatus,
    pub agent_slot: Option<u8>,
    pub message: String,
}

/// Crew dispatch coordinator
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CrewDispatch {
    pub crew_id: String,
    pub tasks: Vec<CrewTask>,
    pub process: String, // "sequential" or "hierarchical"
}

/// The main bridge between crewAI and ladybug-rs
pub struct CrewBridge {
    pub agents: AgentRegistry,
    pub templates: ThinkingTemplateRegistry,
    pub blackboards: BlackboardRegistry,
    pub personas: PersonaRegistry,
    pub a2a: A2AProtocol,
    pub orchestrator: MetaOrchestrator,
    pub kernel: SemanticKernel,
    pub filters: FilterPipeline,
    pub guardrail: KernelGuardrail,
    pub memory: MemoryBank,
    pub observability: ObservabilityManager,
    pub verification: VerificationEngine,
    task_queue: Vec<CrewTask>,
    completed: Vec<DispatchResult>,
}

impl CrewBridge {
    pub fn new() -> Self {
        Self {
            agents: AgentRegistry::new(),
            templates: ThinkingTemplateRegistry::default(),
            blackboards: BlackboardRegistry::new(),
            personas: PersonaRegistry::new(),
            a2a: A2AProtocol::new(),
            orchestrator: MetaOrchestrator::default(),
            kernel: SemanticKernel::new(),
            filters: FilterPipeline::new(),
            guardrail: KernelGuardrail::new(),
            // Memory bank uses Fluid zone prefixes 0x20-0x22
            memory: MemoryBank::new(0x20, 0x21, 0x22),
            observability: ObservabilityManager::new(),
            verification: VerificationEngine::new(),
            task_queue: Vec::new(),
            completed: Vec::new(),
        }
    }

    /// Create with a custom handover policy
    pub fn with_policy(policy: HandoverPolicy) -> Self {
        Self {
            orchestrator: MetaOrchestrator::new(policy),
            ..Self::new()
        }
    }

    /// Register an agent from YAML definition
    pub fn register_agent(&mut self, card: AgentCard) -> Result<Addr, String> {
        let id = card.id.clone();
        let persona = card.persona.clone();
        let addr = self.agents.register(card)?;
        let slot = addr.slot();

        // Create matching blackboard
        self.blackboards.create(slot, &id);

        // Register with meta-orchestrator for flow tracking
        self.orchestrator.register_agent(slot);

        // Attach persona if provided
        if let Some(p) = persona {
            self.personas.attach(slot, p);
        }

        // Refresh affinity graph with new persona
        self.orchestrator.refresh_affinities(&self.personas);

        Ok(addr)
    }

    /// Register agents from YAML string (crewAI agents.yaml format)
    pub fn register_agents_yaml(&mut self, yaml: &str) -> Result<Vec<Addr>, String> {
        let cards = AgentRegistry::from_yaml(yaml)?;
        let mut addrs = Vec::new();
        for card in cards {
            addrs.push(self.register_agent(card)?);
        }
        Ok(addrs)
    }

    /// Register a thinking template
    pub fn register_template(&mut self, template: ThinkingTemplate) -> Result<Addr, String> {
        self.templates.register(template)
    }

    /// Register templates from YAML string
    pub fn register_templates_yaml(&mut self, yaml: &str) -> Result<Vec<Addr>, String> {
        let templates = ThinkingTemplateRegistry::from_yaml(yaml)?;
        let mut addrs = Vec::new();
        for t in templates {
            addrs.push(self.register_template(t)?);
        }
        Ok(addrs)
    }

    /// Submit a task for dispatch
    pub fn submit_task(&mut self, task: CrewTask) -> DispatchResult {
        let task_id = task.id.clone();

        // Try to auto-assign if agent_id specified
        if let Some(agent_id) = task.agent_id.clone() {
            if let Some(agent) = self.agents.get_by_id(&agent_id) {
                let slot = agent.slot;
                let mut task = task;
                task.assigned_slot = slot;
                task.status = TaskStatus::Assigned;

                // Update blackboard
                if let Some(s) = slot {
                    if let Some(bb) = self.blackboards.get_mut(s) {
                        bb.awareness.active_goals.push(task.description.clone());

                        // Apply thinking style if specified
                        if let Some(ref style_name) = task.thinking_style {
                            if let Some(template) = self.templates.get(style_name) {
                                bb.set_thinking_style(template.resolve_base());
                            }
                        }
                    }
                }

                let result = DispatchResult {
                    task_id: task_id.clone(),
                    status: TaskStatus::Assigned,
                    agent_slot: slot,
                    message: format!("Assigned to agent {}", agent_id),
                };

                self.task_queue.push(task);
                return result;
            }
        }

        // Queue for later assignment
        let mut task = task;
        task.status = TaskStatus::Queued;
        self.task_queue.push(task);

        DispatchResult {
            task_id,
            status: TaskStatus::Queued,
            agent_slot: None,
            message: "Queued for assignment".to_string(),
        }
    }

    /// Submit a full crew dispatch
    pub fn dispatch_crew(&mut self, dispatch: CrewDispatch) -> Vec<DispatchResult> {
        let mut results = Vec::new();
        match dispatch.process.as_str() {
            "sequential" => {
                for task in dispatch.tasks {
                    results.push(self.submit_task(task));
                }
            }
            "hierarchical" => {
                // In hierarchical mode, first task goes to manager agent,
                // which can then delegate via A2A
                for task in dispatch.tasks {
                    results.push(self.submit_task(task));
                }
            }
            _ => {
                for task in dispatch.tasks {
                    results.push(self.submit_task(task));
                }
            }
        }
        results
    }

    /// Complete a task
    pub fn complete_task(&mut self, task_id: &str, outcome: &str) -> Option<DispatchResult> {
        if let Some(task) = self.task_queue.iter_mut().find(|t| t.id == task_id) {
            task.status = TaskStatus::Completed;

            // Update blackboard
            if let Some(slot) = task.assigned_slot {
                if let Some(bb) = self.blackboards.get_mut(slot) {
                    let style = task.thinking_style.as_deref().unwrap_or("deliberate");
                    bb.record_task(task_id, &task.description, outcome, style);
                    bb.tick();
                }
            }

            let result = DispatchResult {
                task_id: task_id.to_string(),
                status: TaskStatus::Completed,
                agent_slot: task.assigned_slot,
                message: outcome.to_string(),
            };

            self.completed.push(result.clone());
            Some(result)
        } else {
            None
        }
    }

    /// Send an A2A message between agents
    pub fn send_a2a(&mut self, msg: A2AMessage, space: &mut BindSpace) -> DeliveryStatus {
        self.a2a.send(msg, space)
    }

    /// Receive A2A messages for an agent
    pub fn receive_a2a(&mut self, agent_slot: u8) -> Vec<A2AMessage> {
        self.a2a.receive(agent_slot)
    }

    /// Evaluate handover for an agent via the meta-orchestrator
    pub fn evaluate_handover(
        &mut self,
        source_slot: u8,
        task_desc: Option<&str>,
    ) -> HandoverDecision {
        self.orchestrator.evaluate_handover(
            source_slot,
            &self.blackboards,
            &self.personas,
            task_desc,
        )
    }

    /// Execute a handover decision via A2A
    pub fn execute_handover(
        &mut self,
        decision: &HandoverDecision,
        space: &mut BindSpace,
    ) -> Option<DeliveryStatus> {
        self.orchestrator
            .execute_handover(decision, &mut self.a2a, space)
    }

    /// Update an agent's flow state from a gate evaluation
    pub fn update_flow(&mut self, slot: u8, gate: crate::cognitive::GateState) {
        self.orchestrator.update_flow(slot, gate);
        if let Some(bb) = self.blackboards.get_mut(slot) {
            bb.update_flow(gate);
        }
    }

    /// Route a task to the best available agent using resonance scoring
    pub fn route_task(&mut self, task_description: &str) -> Option<(u8, f32)> {
        self.orchestrator
            .route_task(task_description, &self.personas)
    }

    /// Tick the orchestrator — evaluate all agents and return non-Continue decisions
    pub fn tick_orchestrator(&mut self) -> Vec<HandoverDecision> {
        self.orchestrator.tick(&self.blackboards, &self.personas)
    }

    /// Get total awareness across all A2A channels
    pub fn total_awareness(&self) -> f32 {
        self.a2a.total_awareness()
    }

    /// Bind all state into BindSpace (agents, templates, blackboards, personas)
    pub fn bind_all(&self, space: &mut BindSpace) {
        self.agents.bind_all(space);
        self.templates.bind_all(space);
        self.blackboards.bind_all(space);
        self.personas.bind_all(space);
    }

    /// Get task queue
    pub fn task_queue(&self) -> &[CrewTask] {
        &self.task_queue
    }

    /// Get completed results
    pub fn completed(&self) -> &[DispatchResult] {
        &self.completed
    }

    /// Summary of bridge state (for handover / MCP status)
    pub fn status_summary(&self) -> BridgeStatus {
        BridgeStatus {
            agents_registered: self.agents.count(),
            templates_registered: self.templates.list().len(),
            personas_registered: self.personas.list().len(),
            tasks_queued: self
                .task_queue
                .iter()
                .filter(|t| t.status == TaskStatus::Queued)
                .count(),
            tasks_in_progress: self
                .task_queue
                .iter()
                .filter(|t| t.status == TaskStatus::InProgress)
                .count(),
            tasks_completed: self.completed.len(),
            a2a_channels: self.a2a.channels().len(),
            filters_registered: self.filters.count(),
            guardrail_topics: self.guardrail.denied_topics.len(),
            memories_stored: self.memory.count(None),
            verification_rules: self.verification.count(),
            observability: self.observability.summary(),
        }
    }
}

impl Default for CrewBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Bridge status summary
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BridgeStatus {
    pub agents_registered: usize,
    pub templates_registered: usize,
    pub personas_registered: usize,
    pub tasks_queued: usize,
    pub tasks_in_progress: usize,
    pub tasks_completed: usize,
    pub a2a_channels: usize,
    pub filters_registered: usize,
    pub guardrail_topics: usize,
    pub memories_stored: usize,
    pub verification_rules: usize,
    pub observability: super::kernel_extensions::ObservabilitySummary,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orchestration::agent_card::{AgentGoal, AgentRole};

    fn test_agent(id: &str) -> AgentCard {
        AgentCard {
            id: id.to_string(),
            name: id.to_string(),
            role: AgentRole {
                name: id.to_string(),
                description: format!("{} role", id),
            },
            goal: AgentGoal {
                objective: format!("{} objective", id),
                success_criteria: vec![],
                constraints: vec![],
            },
            thinking_style: "analytical".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn test_register_and_dispatch() {
        let mut bridge = CrewBridge::new();

        // Register agents
        let addr = bridge.register_agent(test_agent("researcher")).unwrap();
        assert_eq!(addr.prefix(), 0x0C);
        assert_eq!(addr.slot(), 0);

        // Submit task
        let task = CrewTask {
            id: "task-1".to_string(),
            description: "Research quantum computing".to_string(),
            expected_output: "Summary report".to_string(),
            agent_id: Some("researcher".to_string()),
            thinking_style: None,
            require_validation: false,
            depends_on: vec![],
            status: TaskStatus::Queued,
            assigned_slot: None,
        };

        let result = bridge.submit_task(task);
        assert_eq!(result.status, TaskStatus::Assigned);
        assert_eq!(result.agent_slot, Some(0));
    }

    #[test]
    fn test_full_crew_dispatch() {
        let mut bridge = CrewBridge::new();

        bridge.register_agent(test_agent("researcher")).unwrap();
        bridge.register_agent(test_agent("writer")).unwrap();

        let dispatch = CrewDispatch {
            crew_id: "crew-1".to_string(),
            tasks: vec![
                CrewTask {
                    id: "t1".to_string(),
                    description: "Research topic".to_string(),
                    expected_output: "Notes".to_string(),
                    agent_id: Some("researcher".to_string()),
                    thinking_style: None,
                    require_validation: true,
                    depends_on: vec![],
                    status: TaskStatus::Queued,
                    assigned_slot: None,
                },
                CrewTask {
                    id: "t2".to_string(),
                    description: "Write report".to_string(),
                    expected_output: "Report".to_string(),
                    agent_id: Some("writer".to_string()),
                    thinking_style: Some("creative".to_string()),
                    require_validation: false,
                    depends_on: vec!["t1".to_string()],
                    status: TaskStatus::Queued,
                    assigned_slot: None,
                },
            ],
            process: "sequential".to_string(),
        };

        let results = bridge.dispatch_crew(dispatch);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].status, TaskStatus::Assigned);
        assert_eq!(results[1].status, TaskStatus::Assigned);
    }

    #[test]
    fn test_status_summary() {
        let bridge = CrewBridge::new();
        let status = bridge.status_summary();
        assert_eq!(status.agents_registered, 0);
        assert_eq!(status.templates_registered, 12); // base styles
    }
}
