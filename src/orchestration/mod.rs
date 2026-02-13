//! Orchestration Module — crewAI Integration Layer
//!
//! Non-blocking, modular expansion that turns crewAI into an orchestration layer
//! while ladybug-rs provides ice-caked blackboard awareness with agent-specific
//! goals, tools, thinking styles, and personal awareness.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────────┐
//! │                         crewAI (Python)                                  │
//! │                                                                          │
//! │   agents.yaml ──► AgentCard ──► BindSpace 0x0C:XX (Agent Registry)      │
//! │   tasks.yaml  ──► TaskSpec  ──► Fluid zone (working memory)             │
//! │   styles.yaml ──► Template  ──► BindSpace 0x0D:XX (Thinking Styles)     │
//! │                                                                          │
//! │   Agent A ◄──── A2A ────► Agent B                                       │
//! │       │         (0x0F)        │                                          │
//! │       ▼                       ▼                                          │
//! │   Blackboard A            Blackboard B                                  │
//! │   (0x0E:slot_a)           (0x0E:slot_b)                                 │
//! │       │                       │                                          │
//! │       └───────────┬───────────┘                                          │
//! │                   ▼                                                      │
//! │            Arrow Flight (zero-copy)                                      │
//! │                   │                                                      │
//! └───────────────────┼──────────────────────────────────────────────────────┘
//!                     ▼
//! ┌──────────────────────────────────────────────────────────────────────────┐
//! │                      ladybug-rs (Rust)                                   │
//! │                                                                          │
//! │   BindSpace                                                             │
//! │   ├── 0x0C: Agent Registry (256 agent cards)                            │
//! │   ├── 0x0D: Thinking Templates (12 styles × 21 variants = 252 slots)   │
//! │   ├── 0x0E: Agent Blackboards (256 per-agent state snapshots)           │
//! │   ├── 0x0F: A2A Channels (256 message routing slots)                   │
//! │   │                                                                     │
//! │   ├── Surface 0x00-0x0B: Existing cognitive substrate (unchanged)       │
//! │   ├── Fluid  0x10-0x7F: Working memory + agent task state              │
//! │   └── Nodes  0x80-0xFF: Universal bind space (unchanged)               │
//! │                                                                          │
//! │   sci/v1/* ──► Statistical validation for research agents               │
//! │   HDR cascade ──► Similarity search for RAG agents                     │
//! │   CogRedis ──► DN.*/CAM.*/DAG.* for all agent operations              │
//! └──────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Prefix Allocation (0x0C-0x0F)
//!
//! | Prefix | Purpose | Slot Layout |
//! |--------|---------|-------------|
//! | 0x0C | Agent Registry | 0x00-0x7F: agent cards, 0x80-0xFF: capabilities |
//! | 0x0D | Thinking Styles | 0x00-0x0B: 12 base styles, 0x0C-0xFF: variants |
//! | 0x0E | Blackboard | 0x00-0xFF: per-agent state (matches agent slot) |
//! | 0x0F | A2A Routing | 0x00-0xFF: message channels (sender:receiver pairs) |

pub mod a2a;
pub mod agent_card;
pub mod blackboard_agent;
pub mod crew_bridge;
pub mod debate;
pub mod handover;
pub mod kernel_extensions;
pub mod meta_orchestrator;
pub mod persona;
pub mod semantic_kernel;
pub mod thinking_template;

pub use agent_card::{AgentCapability, AgentCard, AgentGoal, AgentRegistry, AgentRole};

pub use thinking_template::{StyleOverride, ThinkingTemplate, ThinkingTemplateRegistry};

pub use a2a::{A2AChannel, A2AMessage, A2AProtocol, DeliveryStatus, MessageKind};

pub use blackboard_agent::{AgentAwareness, AgentBlackboard, BlackboardRegistry};

pub use crew_bridge::{CrewBridge, CrewDispatch, CrewTask, DispatchResult, TaskStatus};

pub use persona::{
    CommunicationStyle, FeatureAd, Persona, PersonaExchange, PersonaRegistry, PersonalityTrait,
    VolitionDTO, VolitionSummary,
};

pub use handover::{
    FlowState, FlowTransition, HandoverAction, HandoverDecision, HandoverPolicy, HandoverReason,
};

pub use meta_orchestrator::{
    AffinityEdge, MetaOrchestrator, OrchestratorEvent, OrchestratorStatus,
};

pub use semantic_kernel::{
    CausalRung, CollapseStrategy, CrystalPlugin, DataFusionColumn, DataFusionMapping,
    EscalationResult, ExpansionRegistry, ExpansionSummary, KernelDescription, KernelIntrospection,
    KernelOp, KernelOperator, KernelTruth, KernelZone, PrefixAllocation, PrefixStats,
    ProtocolExtension, RungEscalationStrategy, SemanticKernel,
};

pub use kernel_extensions::{
    ClaimGrounding, ContentCategory, DeniedTopic, FilterContext, FilterPhase, FilterPipeline,
    FilterResult, GroundingMetadata, GroundingResult, GroundingSource, GuardrailAction,
    GuardrailResult, GuardrailSeverity, KernelFilter, KernelGuardrail, KernelMemory, KernelSession,
    KernelSpan, KernelTrace, MemoryBank, MemoryKind, ObservabilityManager, ObservabilitySummary,
    VerificationEngine, VerificationKind, VerificationResult, VerificationRule, WorkflowNode,
    WorkflowOp, WorkflowResult, WorkflowStep, execute_workflow,
};
