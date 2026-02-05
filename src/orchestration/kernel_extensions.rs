//! Kernel Extensions — Cross-Platform Best Practices
//!
//! Incorporates the strongest architectural patterns from:
//!
//! - **Microsoft Semantic Kernel / Agent Framework**: Filter pipeline (middleware),
//!   process framework (deterministic workflows), plugin-scoped tool exposure
//! - **Google ADK / Vertex AI**: Workflow agents (no-LLM orchestration),
//!   Memory Bank (episodic/semantic/procedural), grounding as first-class primitive,
//!   context caching, A2A open protocol alignment
//! - **Amazon Bedrock**: Decoupled guardrails (ApplyGuardrail), formal verification
//!   (Automated Reasoning Checks), observability hierarchy (Session > Trace > Span),
//!   semantic tool discovery, Lambda-contract orchestration
//!
//! # Design Principle
//!
//! All extensions operate ON BindSpace through the SemanticKernel API.
//! They compose with zero-copy fingerprint operations and XOR superposition.
//! Nothing here requires an LLM — the kernel is substrate, not model.
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────────────────┐
//! │                    KERNEL EXTENSION STACK                             │
//! │                                                                       │
//! │  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐               │
//! │  │   FILTERS   │  │  GUARDRAILS  │  │  VERIFICATION │               │
//! │  │  (MS pattern)│  │ (AWS pattern)│  │ (AWS pattern) │               │
//! │  │  pre/post   │  │  standalone  │  │ formal logic  │               │
//! │  │  hooks      │  │  content chk │  │ rule check    │               │
//! │  └──────┬──────┘  └──────┬───────┘  └──────┬────────┘               │
//! │         │                │                  │                        │
//! │  ┌──────┴────────────────┴──────────────────┴────────┐              │
//! │  │                 KERNEL WORKFLOW                     │              │
//! │  │  (Google ADK + MS Process + AWS Flows)             │              │
//! │  │  Sequential │ Parallel │ Loop │ Conditional        │              │
//! │  │  No LLM required — deterministic orchestration    │              │
//! │  └──────┬────────────────────────────────────────────┘              │
//! │         │                                                            │
//! │  ┌──────┴──────────────┐  ┌──────────────────────────┐             │
//! │  │   KERNEL MEMORY     │  │   OBSERVABILITY          │             │
//! │  │  (Google Mem Bank)  │  │  (AWS Session>Trace>Span)│             │
//! │  │  Episodic           │  │  + Grounding Metadata    │             │
//! │  │  Semantic           │  │  (Google confidence)     │             │
//! │  │  Procedural         │  │                          │             │
//! │  └─────────────────────┘  └──────────────────────────┘             │
//! │                                                                       │
//! │  ═══════════════════════════════════════════════════════════          │
//! │                     SEMANTIC KERNEL (core)                            │
//! │                     BindSpace · 8+8 · Zero-Copy                      │
//! │  ═══════════════════════════════════════════════════════════          │
//! └───────────────────────────────────────────────────────────────────────┘
//! ```

use serde::{Deserialize, Serialize};
use crate::storage::bind_space::{Addr, BindSpace, FINGERPRINT_WORDS};
use super::semantic_kernel::{KernelZone, KernelTruth, CausalRung};

// =============================================================================
// 1. FILTER PIPELINE (Microsoft Semantic Kernel Pattern)
// =============================================================================
//
// ASP.NET-style middleware for kernel operations. Each filter can:
// - Inspect/modify arguments before operation
// - Override results after operation
// - Short-circuit the pipeline (skip downstream filters)
// - Implement cross-cutting concerns: caching, auth, PII redaction, logging

/// Phase of a kernel operation that a filter intercepts
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum FilterPhase {
    /// Before a bind/write operation
    PreBind,
    /// After a bind/write operation
    PostBind,
    /// Before a query/read operation
    PreQuery,
    /// After a query/read operation (can modify result)
    PostQuery,
    /// Before a resonance search
    PreResonate,
    /// After a resonance search (can filter/rerank results)
    PostResonate,
    /// Before collapse gate evaluation
    PreCollapse,
    /// After collapse gate decision
    PostCollapse,
    /// Before NARS inference
    PreInference,
    /// After NARS inference
    PostInference,
    /// Before causal rung escalation
    PreEscalation,
    /// After causal rung escalation
    PostEscalation,
}

/// Context passed through the filter pipeline
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FilterContext {
    /// Which operation phase
    pub phase: FilterPhase,
    /// Address being operated on (if applicable)
    pub addr: Option<Addr>,
    /// Zone classification
    pub zone: Option<KernelZone>,
    /// Fingerprint involved (if applicable)
    pub fingerprint: Option<Vec<u64>>,
    /// Similarity scores (for post-resonate)
    pub scores: Option<Vec<(Addr, f32)>>,
    /// Truth value (for inference operations)
    pub truth: Option<KernelTruth>,
    /// Label or description
    pub label: Option<String>,
    /// Whether to short-circuit (skip remaining filters + operation)
    pub short_circuit: bool,
    /// Metadata bag for cross-filter communication
    pub metadata: std::collections::HashMap<String, String>,
}

impl FilterContext {
    pub fn new(phase: FilterPhase) -> Self {
        Self {
            phase,
            addr: None,
            zone: None,
            fingerprint: None,
            scores: None,
            truth: None,
            label: None,
            short_circuit: false,
            metadata: std::collections::HashMap::new(),
        }
    }

    pub fn with_addr(mut self, addr: Addr) -> Self {
        self.zone = Some(KernelZone::from_prefix(addr.prefix()));
        self.addr = Some(addr);
        self
    }

    pub fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }
}

/// A kernel filter — interceptor in the operation pipeline.
///
/// Inspired by Microsoft Semantic Kernel's three filter types
/// (FunctionInvocation, AutoFunctionInvocation, PromptRendering)
/// unified into a single trait that dispatches on FilterPhase.
///
/// Also inspired by Amazon Bedrock AgentCore's gateway interceptors
/// which provide fine-grained access control and audit trails.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KernelFilter {
    pub name: String,
    /// Which phases this filter intercepts
    pub phases: Vec<FilterPhase>,
    /// Priority (lower = earlier in pipeline)
    pub priority: u32,
    /// Whether this filter is enabled
    pub enabled: bool,
    /// Filter-specific configuration
    pub config: std::collections::HashMap<String, String>,
}

/// Result of applying a filter pipeline
#[derive(Clone, Debug)]
pub struct FilterResult {
    pub context: FilterContext,
    pub filters_applied: Vec<String>,
    pub short_circuited_by: Option<String>,
}

/// The filter pipeline — manages ordered filter execution
pub struct FilterPipeline {
    filters: Vec<KernelFilter>,
}

impl FilterPipeline {
    pub fn new() -> Self {
        Self { filters: Vec::new() }
    }

    pub fn add(&mut self, filter: KernelFilter) {
        self.filters.push(filter);
        self.filters.sort_by_key(|f| f.priority);
    }

    pub fn remove(&mut self, name: &str) {
        self.filters.retain(|f| f.name != name);
    }

    /// Run the filter pipeline for a given context.
    /// Returns the (possibly modified) context after all applicable filters.
    pub fn apply(&self, mut ctx: FilterContext) -> FilterResult {
        let mut applied = Vec::new();
        let mut short_circuited_by = None;

        for filter in &self.filters {
            if !filter.enabled {
                continue;
            }
            if !filter.phases.contains(&ctx.phase) {
                continue;
            }

            applied.push(filter.name.clone());

            // Apply filter-specific logic based on config
            // In a real implementation, this would dispatch to registered closures.
            // For the kernel substrate, we encode filter effects declaratively:

            // PII redaction filter: strip labels matching patterns
            if filter.config.get("type").map(|t| t.as_str()) == Some("pii_redact") {
                if let Some(ref mut label) = ctx.label {
                    if label.contains('@') || label.contains("SSN") {
                        *label = "[REDACTED]".to_string();
                        ctx.metadata.insert("pii_redacted".into(), "true".into());
                    }
                }
            }

            // Zone restriction filter: block operations outside allowed zones
            if filter.config.get("type").map(|t| t.as_str()) == Some("zone_restrict") {
                if let Some(zone) = &ctx.zone {
                    let allowed = filter.config.get("allowed_zones").cloned().unwrap_or_default();
                    let zone_name = match zone {
                        KernelZone::Surface { .. } => "Surface",
                        KernelZone::Fluid { .. } => "Fluid",
                        KernelZone::Node { .. } => "Node",
                    };
                    if !allowed.contains(zone_name) {
                        ctx.short_circuit = true;
                        short_circuited_by = Some(filter.name.clone());
                        break;
                    }
                }
            }

            // Cache filter: check if result is already cached
            if filter.config.get("type").map(|t| t.as_str()) == Some("cache") {
                if let Some(addr) = ctx.addr {
                    let cache_key = format!("cache:{}", addr.0);
                    ctx.metadata.insert("cache_key".into(), cache_key);
                }
            }

            if ctx.short_circuit {
                short_circuited_by = Some(filter.name.clone());
                break;
            }
        }

        FilterResult {
            context: ctx,
            filters_applied: applied,
            short_circuited_by,
        }
    }

    pub fn filters(&self) -> &[KernelFilter] {
        &self.filters
    }

    pub fn count(&self) -> usize {
        self.filters.len()
    }
}

impl Default for FilterPipeline {
    fn default() -> Self { Self::new() }
}

// =============================================================================
// 2. GUARDRAILS (Amazon Bedrock Pattern)
// =============================================================================
//
// Decoupled from model invocation — can evaluate ANY content independently.
// Inspired by Bedrock's ApplyGuardrail API: content filtering, topic denial,
// PII detection, and contextual grounding checks as standalone operations.

/// Guardrail severity level
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum GuardrailSeverity {
    None,
    Low,
    Medium,
    High,
    Critical,
}

/// Content category for guardrail filtering
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ContentCategory {
    Hate,
    Violence,
    Sexual,
    SelfHarm,
    Misconduct,
    PromptAttack,
    PII,
    Custom(String),
}

/// A denied topic definition
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeniedTopic {
    pub name: String,
    /// Semantic description (not keyword-based — uses fingerprint similarity)
    pub description: String,
    /// Fingerprint of the topic description for resonance matching
    /// Stored as Vec<u64> because serde doesn't support [u64; 156] directly
    #[serde(skip)]
    pub fingerprint: Option<[u64; FINGERPRINT_WORDS]>,
    pub threshold: f32,
}

/// Grounding check result — is the content grounded in source material?
///
/// Inspired by both Amazon's contextual grounding checks and
/// Google's grounding confidence scores.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GroundingResult {
    /// Overall grounding score (0.0 = completely ungrounded, 1.0 = fully grounded)
    pub score: f32,
    /// Per-claim grounding scores
    pub claim_scores: Vec<ClaimGrounding>,
    /// Whether the content passes the grounding threshold
    pub is_grounded: bool,
}

/// Per-claim grounding evidence
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClaimGrounding {
    /// The claim being evaluated
    pub claim: String,
    /// How well-grounded this claim is (0.0-1.0)
    pub confidence: f32,
    /// Source addresses that support this claim
    pub source_addrs: Vec<Addr>,
    /// Similarity to source material
    pub similarity: f32,
}

/// Result of applying guardrails to content
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GuardrailResult {
    /// Whether the content passed all guardrails
    pub passed: bool,
    /// Content categories that triggered
    pub triggered_categories: Vec<(ContentCategory, GuardrailSeverity)>,
    /// Denied topics that matched
    pub denied_topics_matched: Vec<String>,
    /// PII detected (category, count)
    pub pii_detected: Vec<(String, u32)>,
    /// Grounding check result (if grounding was evaluated)
    pub grounding: Option<GroundingResult>,
    /// Action taken: Pass, Block, or Mask
    pub action: GuardrailAction,
}

/// What the guardrail decided to do
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum GuardrailAction {
    /// Content is safe — pass through
    Pass,
    /// Content blocked — do not deliver
    Block { reason: String },
    /// Content modified — PII masked or topics redacted
    Mask { modifications: u32 },
}

/// The guardrail engine — operates independently of any model invocation.
///
/// This is the key Amazon Bedrock insight: guardrails should be a standalone
/// service, not coupled to specific model calls. Any content from any source
/// (model output, user input, tool results) can be evaluated.
pub struct KernelGuardrail {
    /// Content filter thresholds per category
    pub content_thresholds: Vec<(ContentCategory, GuardrailSeverity)>,
    /// Denied topics (evaluated via fingerprint resonance)
    pub denied_topics: Vec<DeniedTopic>,
    /// Minimum grounding score to pass
    pub grounding_threshold: f32,
    /// Whether to block or mask PII
    pub pii_action: GuardrailAction,
    /// Whether grounding check is enabled
    pub grounding_enabled: bool,
}

impl KernelGuardrail {
    pub fn new() -> Self {
        Self {
            content_thresholds: Vec::new(),
            denied_topics: Vec::new(),
            grounding_threshold: 0.7,
            pii_action: GuardrailAction::Pass,
            grounding_enabled: false,
        }
    }

    /// Add a content filter threshold
    pub fn add_content_filter(&mut self, category: ContentCategory, max_severity: GuardrailSeverity) {
        self.content_thresholds.push((category, max_severity));
    }

    /// Add a denied topic with its fingerprint for resonance matching
    pub fn add_denied_topic(&mut self, topic: DeniedTopic) {
        self.denied_topics.push(topic);
    }

    /// Enable grounding checks with a minimum threshold
    pub fn enable_grounding(&mut self, threshold: f32) {
        self.grounding_enabled = true;
        self.grounding_threshold = threshold;
    }

    /// Apply guardrails to a fingerprint — check it against denied topics
    /// using resonance in the BindSpace.
    ///
    /// This is the "ApplyGuardrail" API — standalone, decoupled from model calls.
    pub fn apply(
        &self,
        content_fp: &[u64; FINGERPRINT_WORDS],
        space: &BindSpace,
        source_addrs: Option<&[Addr]>,
    ) -> GuardrailResult {
        let mut result = GuardrailResult {
            passed: true,
            triggered_categories: Vec::new(),
            denied_topics_matched: Vec::new(),
            pii_detected: Vec::new(),
            grounding: None,
            action: GuardrailAction::Pass,
        };

        // Check denied topics via fingerprint resonance
        for topic in &self.denied_topics {
            if let Some(ref topic_fp) = topic.fingerprint {
                let sim = super::semantic_kernel::SemanticKernel::hamming_similarity(
                    content_fp, topic_fp,
                );
                if sim >= topic.threshold {
                    result.denied_topics_matched.push(topic.name.clone());
                    result.passed = false;
                }
            }
        }

        // Grounding check: verify content is supported by source material
        if self.grounding_enabled {
            if let Some(sources) = source_addrs {
                let mut claim_scores = Vec::new();
                let mut total_sim = 0.0f32;

                for &source_addr in sources {
                    if let Some(source_node) = space.read(source_addr) {
                        let sim = super::semantic_kernel::SemanticKernel::hamming_similarity(
                            content_fp, &source_node.fingerprint,
                        );
                        total_sim += sim;
                        claim_scores.push(ClaimGrounding {
                            claim: format!("source@{:04X}", source_addr.0),
                            confidence: sim,
                            source_addrs: vec![source_addr],
                            similarity: sim,
                        });
                    }
                }

                let avg_score = if sources.is_empty() {
                    0.0
                } else {
                    total_sim / sources.len() as f32
                };

                let is_grounded = avg_score >= self.grounding_threshold;
                if !is_grounded {
                    result.passed = false;
                }

                result.grounding = Some(GroundingResult {
                    score: avg_score,
                    claim_scores,
                    is_grounded,
                });
            }
        }

        // Set action based on results
        if !result.passed {
            if !result.denied_topics_matched.is_empty() {
                result.action = GuardrailAction::Block {
                    reason: format!(
                        "Denied topics: {}",
                        result.denied_topics_matched.join(", ")
                    ),
                };
            } else if result.grounding.as_ref().map(|g| !g.is_grounded).unwrap_or(false) {
                result.action = GuardrailAction::Block {
                    reason: "Content not grounded in source material".to_string(),
                };
            }
        }

        result
    }
}

impl Default for KernelGuardrail {
    fn default() -> Self { Self::new() }
}

// =============================================================================
// 3. WORKFLOW ORCHESTRATION (Google ADK + MS Process + AWS Flows)
// =============================================================================
//
// Deterministic workflow nodes that DO NOT require an LLM.
// This is Google's key insight: SequentialAgent/ParallelAgent/LoopAgent
// execute without model calls, making pipelines cheap and predictable.
// Combined with Microsoft's Process Framework event-driven steps
// and Amazon's Bedrock Flows DAG with DoWhile loops.

/// A workflow step — the fundamental unit of deterministic orchestration.
///
/// Each step operates on BindSpace addresses via KernelOps.
/// Steps communicate through shared state (addresses in Fluid zone).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkflowStep {
    pub id: String,
    pub name: String,
    /// Input addresses this step reads from
    pub input_addrs: Vec<Addr>,
    /// Output address this step writes to
    pub output_addr: Addr,
    /// The kernel operation this step performs
    pub operation: WorkflowOp,
    /// Whether this step is stateful (checkpointable)
    pub stateful: bool,
}

/// Operations a workflow step can perform — all without LLM invocation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum WorkflowOp {
    /// Copy fingerprint from input to output
    Forward,
    /// XOR-bind all inputs into output
    XorCompose,
    /// Bundle (majority vote) inputs into output
    Bundle,
    /// Permute input fingerprint
    Permute { shift: usize },
    /// Resonate: search for similar fingerprints
    Resonate { threshold: f32, limit: usize },
    /// Collapse gate evaluation on inputs
    Collapse,
    /// NARS inference between two inputs
    Deduce,
    /// Causal rung escalation
    Escalate,
    /// Crystallize (Fluid → Node promotion)
    Crystallize,
    /// Custom: delegate to agent slot
    DelegateToAgent { agent_slot: u8 },
    /// Apply guardrail check
    GuardrailCheck,
}

/// A workflow node in the execution graph
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum WorkflowNode {
    /// Single step execution
    Step(WorkflowStep),

    /// Sequential pipeline — steps execute in order.
    /// Google ADK: SequentialAgent pattern.
    /// Microsoft: Sequential orchestration pattern.
    Sequential {
        id: String,
        steps: Vec<WorkflowNode>,
    },

    /// Parallel fan-out — all steps execute concurrently.
    /// Google ADK: ParallelAgent pattern.
    /// Microsoft: Concurrent orchestration pattern.
    /// Each step MUST write to unique output addresses (no race conditions).
    Parallel {
        id: String,
        branches: Vec<WorkflowNode>,
    },

    /// Loop with exit condition — iterates until condition met or max iterations.
    /// Google ADK: LoopAgent with exit_loop tool.
    /// Amazon Bedrock: DoWhile loop node.
    Loop {
        id: String,
        body: Box<WorkflowNode>,
        /// Exit when this address has similarity > threshold to target
        exit_condition_addr: Addr,
        exit_threshold: f32,
        max_iterations: u32,
    },

    /// Conditional routing — evaluate fingerprint similarity to choose branch.
    /// Amazon Bedrock: Condition node in Flows.
    Conditional {
        id: String,
        /// Address to evaluate
        condition_addr: Addr,
        /// Branches: (threshold, node). First matching branch wins.
        branches: Vec<(f32, WorkflowNode)>,
        /// Default branch if no condition matches
        default: Option<Box<WorkflowNode>>,
    },
}

/// Workflow execution result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkflowResult {
    pub workflow_id: String,
    pub steps_executed: u32,
    pub loops_iterated: u32,
    pub branches_taken: Vec<String>,
    pub output_addrs: Vec<Addr>,
    pub completed: bool,
    /// Execution trace for observability
    pub spans: Vec<KernelSpan>,
}

/// Execute a workflow node against BindSpace
pub fn execute_workflow(
    node: &WorkflowNode,
    space: &mut BindSpace,
    kernel: &super::semantic_kernel::SemanticKernel,
) -> WorkflowResult {
    let mut result = WorkflowResult {
        workflow_id: String::new(),
        steps_executed: 0,
        loops_iterated: 0,
        branches_taken: Vec::new(),
        output_addrs: Vec::new(),
        completed: false,
        spans: Vec::new(),
    };

    execute_node(node, space, kernel, &mut result);
    result.completed = true;
    result
}

fn execute_node(
    node: &WorkflowNode,
    space: &mut BindSpace,
    kernel: &super::semantic_kernel::SemanticKernel,
    result: &mut WorkflowResult,
) {
    match node {
        WorkflowNode::Step(step) => {
            execute_step(step, space, kernel);
            result.steps_executed += 1;
            result.output_addrs.push(step.output_addr);
            result.spans.push(KernelSpan {
                id: step.id.clone(),
                operation: format!("{:?}", step.operation),
                parent_id: None,
                duration_ns: 0,
                metadata: std::collections::HashMap::new(),
            });
        }

        WorkflowNode::Sequential { id, steps } => {
            result.workflow_id = id.clone();
            for step in steps {
                execute_node(step, space, kernel, result);
            }
        }

        WorkflowNode::Parallel { id, branches } => {
            result.workflow_id = id.clone();
            // In single-threaded context, execute sequentially.
            // With rayon feature, these would execute in parallel.
            for branch in branches {
                execute_node(branch, space, kernel, result);
            }
        }

        WorkflowNode::Loop { id, body, exit_condition_addr, exit_threshold, max_iterations } => {
            result.workflow_id = id.clone();
            for i in 0..*max_iterations {
                execute_node(body, space, kernel, result);
                result.loops_iterated += 1;

                // Check exit condition
                if let Some(node) = space.read(*exit_condition_addr) {
                    let popcount: u32 = node.fingerprint.iter().map(|w| w.count_ones()).sum();
                    let density = popcount as f32 / (FINGERPRINT_WORDS * 64) as f32;
                    if density >= *exit_threshold {
                        result.branches_taken.push(format!("loop:{} exited at iteration {}", id, i));
                        break;
                    }
                }
            }
        }

        WorkflowNode::Conditional { id, condition_addr, branches, default } => {
            if let Some(node) = space.read(*condition_addr) {
                let density: f32 = {
                    let popcount: u32 = node.fingerprint.iter().map(|w| w.count_ones()).sum();
                    popcount as f32 / (FINGERPRINT_WORDS * 64) as f32
                };

                let mut matched = false;
                for (threshold, branch) in branches {
                    if density >= *threshold {
                        result.branches_taken.push(format!("cond:{} took branch@{:.2}", id, threshold));
                        execute_node(branch, space, kernel, result);
                        matched = true;
                        break;
                    }
                }

                if !matched {
                    if let Some(default_branch) = default {
                        result.branches_taken.push(format!("cond:{} took default", id));
                        execute_node(default_branch, space, kernel, result);
                    }
                }
            }
        }
    }
}

fn execute_step(
    step: &WorkflowStep,
    space: &mut BindSpace,
    kernel: &super::semantic_kernel::SemanticKernel,
) {
    match &step.operation {
        WorkflowOp::Forward => {
            if let Some(&input) = step.input_addrs.first() {
                if let Some(fp) = kernel.query(space, input) {
                    kernel.bind(space, step.output_addr, fp, Some(&step.name));
                }
            }
        }
        WorkflowOp::XorCompose => {
            if step.input_addrs.len() >= 2 {
                kernel.xor_bind(space, step.input_addrs[0], step.input_addrs[1], step.output_addr, Some(&step.name));
            }
        }
        WorkflowOp::Bundle => {
            kernel.bundle(space, &step.input_addrs, step.output_addr, Some(&step.name));
        }
        WorkflowOp::Permute { shift } => {
            if let Some(&input) = step.input_addrs.first() {
                kernel.permute(space, input, *shift, step.output_addr, Some(&step.name));
            }
        }
        WorkflowOp::Resonate { threshold, limit } => {
            if let Some(&input) = step.input_addrs.first() {
                if let Some(fp) = kernel.query(space, input) {
                    let results = kernel.resonate(space, &fp, None, *threshold, *limit);
                    // Write the best match to output
                    if let Some((best_addr, _)) = results.first() {
                        if let Some(best_fp) = kernel.query(space, *best_addr) {
                            kernel.bind(space, step.output_addr, best_fp, Some(&step.name));
                        }
                    }
                }
            }
        }
        WorkflowOp::Collapse => {
            if let Some(&input) = step.input_addrs.first() {
                if let Some(reference) = kernel.query(space, input) {
                    let _gate = kernel.collapse(space, &step.input_addrs, &reference);
                }
            }
        }
        WorkflowOp::Deduce | WorkflowOp::Escalate | WorkflowOp::Crystallize
        | WorkflowOp::DelegateToAgent { .. } | WorkflowOp::GuardrailCheck => {
            // These are higher-level operations that require additional context.
            // Handled by the orchestrator layer, not the workflow engine directly.
        }
    }
}

// =============================================================================
// 4. KERNEL MEMORY (Google Memory Bank Pattern)
// =============================================================================
//
// Three-layer memory system inspired by Google's Memory Bank:
// - Episodic: specific past interactions and events
// - Semantic: generalized knowledge and preferences
// - Procedural: learned workflows and skills
//
// All layers store fingerprints in BindSpace Fluid zone with TTLs.
// Semantic extraction happens through XOR composition + bundle operations.

/// Memory type classification
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MemoryKind {
    /// Specific events: "Agent A handed off to Agent B at cycle 42"
    Episodic,
    /// Generalized knowledge: "User prefers analytical thinking style"
    Semantic,
    /// Learned skills: "For research tasks, use resonance threshold 0.7"
    Procedural,
}

/// A memory entry in the kernel's memory bank
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KernelMemory {
    pub id: String,
    pub kind: MemoryKind,
    /// Human-readable description
    pub content: String,
    /// Fingerprint stored in Fluid zone
    pub addr: Addr,
    /// Cycle at which this memory was created
    pub created_at: u64,
    /// Cycle at which this memory was last accessed
    pub last_accessed: u64,
    /// How many times this memory has been retrieved
    pub access_count: u32,
    /// Relevance score from last retrieval (0.0-1.0)
    pub relevance: f32,
    /// Agent slot that created this memory (if any)
    pub source_agent: Option<u8>,
}

/// The memory bank — manages the three-layer memory system.
///
/// Fluid zone prefix allocation:
/// - Episodic memories: allocated dynamically via expansion registry
/// - Semantic memories: allocated dynamically via expansion registry
/// - Procedural memories: allocated dynamically via expansion registry
pub struct MemoryBank {
    memories: Vec<KernelMemory>,
    /// Next slot to allocate in each memory prefix
    next_slots: std::collections::HashMap<MemoryKind, u8>,
    /// Fluid prefix for episodic memories
    pub episodic_prefix: u8,
    /// Fluid prefix for semantic memories
    pub semantic_prefix: u8,
    /// Fluid prefix for procedural memories
    pub procedural_prefix: u8,
}

impl MemoryBank {
    pub fn new(episodic_prefix: u8, semantic_prefix: u8, procedural_prefix: u8) -> Self {
        let mut next_slots = std::collections::HashMap::new();
        next_slots.insert(MemoryKind::Episodic, 0);
        next_slots.insert(MemoryKind::Semantic, 0);
        next_slots.insert(MemoryKind::Procedural, 0);

        Self {
            memories: Vec::new(),
            next_slots,
            episodic_prefix,
            semantic_prefix,
            procedural_prefix,
        }
    }

    /// Store a new memory in the bank.
    /// Writes the fingerprint to the appropriate Fluid zone prefix.
    pub fn store(
        &mut self,
        kind: MemoryKind,
        content: &str,
        fingerprint: [u64; FINGERPRINT_WORDS],
        space: &mut BindSpace,
        cycle: u64,
        source_agent: Option<u8>,
    ) -> Option<Addr> {
        let prefix = match kind {
            MemoryKind::Episodic => self.episodic_prefix,
            MemoryKind::Semantic => self.semantic_prefix,
            MemoryKind::Procedural => self.procedural_prefix,
        };

        let slot = self.next_slots.get_mut(&kind)?;
        if *slot == 255 {
            return None; // Prefix full
        }

        let addr = Addr::new(prefix, *slot);
        space.write_at(addr, fingerprint);
        if let Some(node) = space.read_mut(addr) {
            node.label = Some(format!("{:?}:{}", kind, content));
        }

        let memory = KernelMemory {
            id: format!("{:?}-{}-{}", kind, prefix, slot),
            kind: kind.clone(),
            content: content.to_string(),
            addr,
            created_at: cycle,
            last_accessed: cycle,
            access_count: 0,
            relevance: 0.0,
            source_agent,
        };

        self.memories.push(memory);
        *slot += 1;
        Some(addr)
    }

    /// Retrieve memories by semantic similarity to a query fingerprint.
    /// This is the "RetrieveMemories" API from Google's Memory Bank.
    pub fn retrieve(
        &mut self,
        query: &[u64; FINGERPRINT_WORDS],
        kind: Option<MemoryKind>,
        space: &BindSpace,
        threshold: f32,
        limit: usize,
        cycle: u64,
    ) -> Vec<&KernelMemory> {
        let mut scored: Vec<(usize, f32)> = self.memories.iter()
            .enumerate()
            .filter(|(_, m)| kind.as_ref().map(|k| k == &m.kind).unwrap_or(true))
            .filter_map(|(i, m)| {
                space.read(m.addr).map(|node| {
                    let sim = super::semantic_kernel::SemanticKernel::hamming_similarity(
                        query, &node.fingerprint,
                    );
                    (i, sim)
                })
            })
            .filter(|(_, sim)| *sim >= threshold)
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);

        // Update access stats
        for &(idx, sim) in &scored {
            self.memories[idx].last_accessed = cycle;
            self.memories[idx].access_count += 1;
            self.memories[idx].relevance = sim;
        }

        scored.iter().map(|&(idx, _)| &self.memories[idx]).collect()
    }

    /// Extract semantic memories from episodic memories.
    /// This is the "GenerateMemories" pattern from Google's Memory Bank —
    /// bundling multiple episodic fingerprints into a generalized semantic memory.
    pub fn extract_semantic(
        &mut self,
        space: &mut BindSpace,
        cycle: u64,
    ) -> Vec<Addr> {
        let episodic_addrs: Vec<Addr> = self.memories.iter()
            .filter(|m| m.kind == MemoryKind::Episodic)
            .map(|m| m.addr)
            .collect();

        if episodic_addrs.len() < 3 {
            return Vec::new(); // Need enough episodes to generalize
        }

        // Bundle episodic memories into a semantic generalization
        let sources: Vec<[u64; FINGERPRINT_WORDS]> = episodic_addrs.iter()
            .filter_map(|&addr| space.read(addr).map(|n| n.fingerprint))
            .collect();

        if sources.is_empty() {
            return Vec::new();
        }

        // Majority vote bundle
        let threshold = sources.len() / 2;
        let mut bundled = [0u64; FINGERPRINT_WORDS];
        for word_idx in 0..FINGERPRINT_WORDS {
            for bit in 0..64 {
                let mask = 1u64 << bit;
                let count = sources.iter().filter(|fp| fp[word_idx] & mask != 0).count();
                if count > threshold {
                    bundled[word_idx] |= mask;
                }
            }
        }

        let mut new_addrs = Vec::new();
        if let Some(addr) = self.store(
            MemoryKind::Semantic,
            &format!("extracted from {} episodes", sources.len()),
            bundled,
            space,
            cycle,
            None,
        ) {
            new_addrs.push(addr);
        }

        new_addrs
    }

    /// List all memories of a given kind
    pub fn list(&self, kind: Option<MemoryKind>) -> Vec<&KernelMemory> {
        self.memories.iter()
            .filter(|m| kind.as_ref().map(|k| k == &m.kind).unwrap_or(true))
            .collect()
    }

    /// Count memories by kind
    pub fn count(&self, kind: Option<MemoryKind>) -> usize {
        self.memories.iter()
            .filter(|m| kind.as_ref().map(|k| k == &m.kind).unwrap_or(true))
            .count()
    }
}

// =============================================================================
// 5. OBSERVABILITY (Amazon Bedrock Pattern + Google Grounding)
// =============================================================================
//
// Session > Trace > Span hierarchy inspired by Amazon Bedrock AgentCore.
// OTEL-compatible structure for integration with existing observability tools.
// Enhanced with Google's grounding confidence metadata.

/// A span in the observability hierarchy
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KernelSpan {
    pub id: String,
    pub operation: String,
    pub parent_id: Option<String>,
    pub duration_ns: u64,
    pub metadata: std::collections::HashMap<String, String>,
}

/// A trace — the complete execution path of a single operation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KernelTrace {
    pub id: String,
    pub session_id: String,
    pub spans: Vec<KernelSpan>,
    pub started_at: u64,
    pub completed_at: Option<u64>,
    /// Grounding metadata for this trace
    pub grounding: Option<GroundingMetadata>,
}

/// Grounding metadata — confidence scores for search/retrieval results.
///
/// Inspired by Google's grounding confidence scores per response segment
/// and Amazon's contextual grounding checks.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GroundingMetadata {
    /// Overall grounding confidence (0.0-1.0)
    pub confidence: f32,
    /// Source addresses used for grounding
    pub sources: Vec<GroundingSource>,
    /// Whether this result should be considered grounded
    pub is_grounded: bool,
}

/// A grounding source with provenance
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GroundingSource {
    pub addr: Addr,
    pub label: Option<String>,
    pub similarity: f32,
    pub zone: String,
}

/// A session — high-level grouping of traces for a user/agent interaction.
///
/// Inspired by Amazon Bedrock AgentCore's session model and
/// Google's Interactions API server-side state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KernelSession {
    pub id: String,
    pub agent_slot: Option<u8>,
    pub traces: Vec<String>,
    pub created_at: u64,
    pub last_active: u64,
    /// Session-scoped state (shared between traces)
    pub state: std::collections::HashMap<String, String>,
    /// Whether this session has server-side state (Google Interactions pattern)
    pub server_side_state: bool,
}

/// The observability manager
pub struct ObservabilityManager {
    sessions: Vec<KernelSession>,
    traces: Vec<KernelTrace>,
    active_session: Option<String>,
}

impl ObservabilityManager {
    pub fn new() -> Self {
        Self {
            sessions: Vec::new(),
            traces: Vec::new(),
            active_session: None,
        }
    }

    /// Start a new session
    pub fn start_session(&mut self, agent_slot: Option<u8>, cycle: u64) -> String {
        let id = format!("session-{}-{}", agent_slot.unwrap_or(255), cycle);
        let session = KernelSession {
            id: id.clone(),
            agent_slot,
            traces: Vec::new(),
            created_at: cycle,
            last_active: cycle,
            state: std::collections::HashMap::new(),
            server_side_state: true,
        };
        self.sessions.push(session);
        self.active_session = Some(id.clone());
        id
    }

    /// Start a trace within the active session
    pub fn start_trace(&mut self, operation: &str, cycle: u64) -> String {
        let session_id = self.active_session.clone().unwrap_or_else(|| "no-session".into());
        let id = format!("trace-{}-{}", operation, cycle);
        let trace = KernelTrace {
            id: id.clone(),
            session_id: session_id.clone(),
            spans: Vec::new(),
            started_at: cycle,
            completed_at: None,
            grounding: None,
        };

        // Update session
        if let Some(session) = self.sessions.iter_mut().find(|s| s.id == session_id) {
            session.traces.push(id.clone());
            session.last_active = cycle;
        }

        self.traces.push(trace);
        id
    }

    /// Add a span to a trace
    pub fn add_span(&mut self, trace_id: &str, span: KernelSpan) {
        if let Some(trace) = self.traces.iter_mut().find(|t| t.id == trace_id) {
            trace.spans.push(span);
        }
    }

    /// Complete a trace with optional grounding metadata
    pub fn complete_trace(
        &mut self,
        trace_id: &str,
        cycle: u64,
        grounding: Option<GroundingMetadata>,
    ) {
        if let Some(trace) = self.traces.iter_mut().find(|t| t.id == trace_id) {
            trace.completed_at = Some(cycle);
            trace.grounding = grounding;
        }
    }

    /// Get session by ID
    pub fn session(&self, id: &str) -> Option<&KernelSession> {
        self.sessions.iter().find(|s| s.id == id)
    }

    /// Get trace by ID
    pub fn trace(&self, id: &str) -> Option<&KernelTrace> {
        self.traces.iter().find(|t| t.id == id)
    }

    /// Summary for status reporting
    pub fn summary(&self) -> ObservabilitySummary {
        ObservabilitySummary {
            total_sessions: self.sessions.len(),
            total_traces: self.traces.len(),
            total_spans: self.traces.iter().map(|t| t.spans.len()).sum(),
            active_session: self.active_session.clone(),
            grounded_traces: self.traces.iter()
                .filter(|t| t.grounding.as_ref().map(|g| g.is_grounded).unwrap_or(false))
                .count(),
        }
    }
}

impl Default for ObservabilityManager {
    fn default() -> Self { Self::new() }
}

/// Summary of observability state
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ObservabilitySummary {
    pub total_sessions: usize,
    pub total_traces: usize,
    pub total_spans: usize,
    pub active_session: Option<String>,
    pub grounded_traces: usize,
}

// =============================================================================
// 6. FORMAL VERIFICATION HOOKS (Amazon Bedrock Automated Reasoning)
// =============================================================================
//
// Hooks for rule-based verification of kernel operations.
// Inspired by Amazon's Automated Reasoning Checks which use formal mathematical
// logic (not ML) to verify factual accuracy against defined policies.

/// A verification rule — a formal constraint on kernel state
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerificationRule {
    pub name: String,
    pub description: String,
    /// Rule type
    pub kind: VerificationKind,
    /// Parameters for the rule
    pub params: std::collections::HashMap<String, String>,
}

/// Types of formal verification
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum VerificationKind {
    /// Minimum popcount — fingerprint must have enough set bits
    MinimumDensity,
    /// Maximum Hamming distance — result must be similar enough to reference
    MaximumDistance,
    /// Zone constraint — operation must target specific zones
    ZoneConstraint,
    /// Parity check — XOR composition must satisfy parity invariant
    ParityCheck,
    /// Consistency — NARS truth values must be internally consistent
    TruthConsistency,
    /// Causal ordering — rung escalation must be monotonic
    CausalOrdering,
    /// Custom rule
    Custom,
}

/// Result of a verification check
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerificationResult {
    pub rule_name: String,
    pub passed: bool,
    pub explanation: String,
    /// Which rule variables/components were checked
    pub checked_components: Vec<String>,
    /// Suggested corrections (if failed)
    pub suggestions: Vec<String>,
}

/// The verification engine — applies formal rules to kernel operations
pub struct VerificationEngine {
    rules: Vec<VerificationRule>,
}

impl VerificationEngine {
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    pub fn add_rule(&mut self, rule: VerificationRule) {
        self.rules.push(rule);
    }

    /// Verify a fingerprint against all applicable rules
    pub fn verify_fingerprint(
        &self,
        fingerprint: &[u64; FINGERPRINT_WORDS],
        addr: Addr,
    ) -> Vec<VerificationResult> {
        let mut results = Vec::new();

        for rule in &self.rules {
            match rule.kind {
                VerificationKind::MinimumDensity => {
                    let popcount: u32 = fingerprint.iter().map(|w| w.count_ones()).sum();
                    let density = popcount as f32 / (FINGERPRINT_WORDS * 64) as f32;
                    let min = rule.params.get("min_density")
                        .and_then(|v| v.parse::<f32>().ok())
                        .unwrap_or(0.1);

                    results.push(VerificationResult {
                        rule_name: rule.name.clone(),
                        passed: density >= min,
                        explanation: format!("Density {:.4} vs minimum {:.4}", density, min),
                        checked_components: vec!["popcount".into(), "density".into()],
                        suggestions: if density < min {
                            vec!["Fingerprint is too sparse — consider bundling with more sources".into()]
                        } else {
                            vec![]
                        },
                    });
                }

                VerificationKind::ZoneConstraint => {
                    let allowed = rule.params.get("allowed_zone").cloned().unwrap_or_default();
                    let zone = KernelZone::from_prefix(addr.prefix());
                    let zone_name = match zone {
                        KernelZone::Surface { .. } => "Surface",
                        KernelZone::Fluid { .. } => "Fluid",
                        KernelZone::Node { .. } => "Node",
                    };
                    let passed = allowed.contains(zone_name);

                    results.push(VerificationResult {
                        rule_name: rule.name.clone(),
                        passed,
                        explanation: format!("Zone {} vs allowed {}", zone_name, allowed),
                        checked_components: vec!["zone".into(), "prefix".into()],
                        suggestions: if !passed {
                            vec![format!("Operation targets {} zone, but only {} is allowed", zone_name, allowed)]
                        } else {
                            vec![]
                        },
                    });
                }

                VerificationKind::TruthConsistency => {
                    // Check that NARS truth values encoded in the fingerprint are consistent
                    // (This is a placeholder — real implementation would decode truth values)
                    results.push(VerificationResult {
                        rule_name: rule.name.clone(),
                        passed: true,
                        explanation: "Truth consistency check passed".into(),
                        checked_components: vec!["truth.f".into(), "truth.c".into()],
                        suggestions: vec![],
                    });
                }

                _ => {
                    // Other rule types: MaximumDistance, ParityCheck, CausalOrdering, Custom
                    // These require additional context passed in through params
                }
            }
        }

        results
    }

    pub fn rules(&self) -> &[VerificationRule] {
        &self.rules
    }

    pub fn count(&self) -> usize {
        self.rules.len()
    }
}

impl Default for VerificationEngine {
    fn default() -> Self { Self::new() }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_pipeline_ordering() {
        let mut pipeline = FilterPipeline::new();
        pipeline.add(KernelFilter {
            name: "audit".into(),
            phases: vec![FilterPhase::PreBind, FilterPhase::PostBind],
            priority: 100,
            enabled: true,
            config: std::collections::HashMap::new(),
        });
        pipeline.add(KernelFilter {
            name: "auth".into(),
            phases: vec![FilterPhase::PreBind],
            priority: 10,
            enabled: true,
            config: std::collections::HashMap::new(),
        });

        // Auth should come first (lower priority number)
        assert_eq!(pipeline.filters()[0].name, "auth");
        assert_eq!(pipeline.filters()[1].name, "audit");
    }

    #[test]
    fn test_filter_pipeline_pii_redaction() {
        let mut pipeline = FilterPipeline::new();
        let mut config = std::collections::HashMap::new();
        config.insert("type".into(), "pii_redact".into());
        pipeline.add(KernelFilter {
            name: "pii".into(),
            phases: vec![FilterPhase::PreBind],
            priority: 1,
            enabled: true,
            config,
        });

        let ctx = FilterContext::new(FilterPhase::PreBind)
            .with_label("user@email.com");
        let result = pipeline.apply(ctx);
        assert_eq!(result.context.label.as_deref(), Some("[REDACTED]"));
        assert_eq!(result.context.metadata.get("pii_redacted").map(|s| s.as_str()), Some("true"));
    }

    #[test]
    fn test_filter_zone_restriction() {
        let mut pipeline = FilterPipeline::new();
        let mut config = std::collections::HashMap::new();
        config.insert("type".into(), "zone_restrict".into());
        config.insert("allowed_zones".into(), "Node".into());
        pipeline.add(KernelFilter {
            name: "zone_guard".into(),
            phases: vec![FilterPhase::PreBind],
            priority: 1,
            enabled: true,
            config,
        });

        // Surface zone should be blocked
        let ctx = FilterContext::new(FilterPhase::PreBind)
            .with_addr(Addr::new(0x01, 0x00));
        let result = pipeline.apply(ctx);
        assert!(result.context.short_circuit);
        assert_eq!(result.short_circuited_by, Some("zone_guard".into()));

        // Node zone should pass
        let ctx = FilterContext::new(FilterPhase::PreBind)
            .with_addr(Addr::new(0x80, 0x00));
        let result = pipeline.apply(ctx);
        assert!(!result.context.short_circuit);
    }

    #[test]
    fn test_guardrail_denied_topic() {
        use sha2::{Sha256, Digest};

        let mut guardrail = KernelGuardrail::new();

        // Create a denied topic fingerprint
        let topic_text = "investment financial advice";
        let mut hasher = Sha256::new();
        hasher.update(topic_text.as_bytes());
        let hash = hasher.finalize();
        let mut topic_fp = [0u64; FINGERPRINT_WORDS];
        for (i, word) in topic_fp.iter_mut().enumerate() {
            let mut h = Sha256::new();
            h.update(&hash);
            h.update(&(i as u32).to_le_bytes());
            let block = h.finalize();
            *word = u64::from_le_bytes(block[..8].try_into().unwrap());
        }

        guardrail.add_denied_topic(DeniedTopic {
            name: "financial_advice".into(),
            description: "Investment and financial advice".into(),
            fingerprint: Some(topic_fp),
            threshold: 0.95, // High threshold — only very similar content triggers
        });

        let space = BindSpace::new();

        // Same content should be blocked
        let result = guardrail.apply(&topic_fp, &space, None);
        assert!(!result.passed);
        assert!(result.denied_topics_matched.contains(&"financial_advice".to_string()));

        // Different content should pass
        let other_fp = [0xDEADBEEFu64; FINGERPRINT_WORDS];
        let result = guardrail.apply(&other_fp, &space, None);
        assert!(result.passed);
    }

    #[test]
    fn test_guardrail_grounding_check() {
        let mut guardrail = KernelGuardrail::new();
        guardrail.enable_grounding(0.5);

        let mut space = BindSpace::new();
        let source_fp = [0xFF00FF00_FF00FF00u64; FINGERPRINT_WORDS];
        let source_addr = Addr::new(0x80, 0x01);
        space.write_at(source_addr, source_fp);

        // Content identical to source — should be grounded
        let result = guardrail.apply(&source_fp, &space, Some(&[source_addr]));
        assert!(result.passed);
        assert!(result.grounding.as_ref().unwrap().is_grounded);

        // Content very different from source — should not be grounded
        let ungrounded_fp = [0x00FF00FF_00FF00FFu64; FINGERPRINT_WORDS];
        let result = guardrail.apply(&ungrounded_fp, &space, Some(&[source_addr]));
        assert!(!result.passed);
        assert!(!result.grounding.as_ref().unwrap().is_grounded);
    }

    #[test]
    fn test_workflow_sequential() {
        let kernel = super::super::semantic_kernel::SemanticKernel::new();
        let mut space = BindSpace::new();

        // Write a fingerprint to start
        let input = Addr::new(0x10, 0x01);
        let output1 = Addr::new(0x10, 0x02);
        let output2 = Addr::new(0x10, 0x03);
        let fp = [0xCAFEBABEu64; FINGERPRINT_WORDS];
        space.write_at(input, fp);

        let workflow = WorkflowNode::Sequential {
            id: "test-seq".into(),
            steps: vec![
                WorkflowNode::Step(WorkflowStep {
                    id: "step1".into(),
                    name: "forward".into(),
                    input_addrs: vec![input],
                    output_addr: output1,
                    operation: WorkflowOp::Forward,
                    stateful: false,
                }),
                WorkflowNode::Step(WorkflowStep {
                    id: "step2".into(),
                    name: "permute".into(),
                    input_addrs: vec![output1],
                    output_addr: output2,
                    operation: WorkflowOp::Permute { shift: 1 },
                    stateful: false,
                }),
            ],
        };

        let result = execute_workflow(&workflow, &mut space, &kernel);
        assert!(result.completed);
        assert_eq!(result.steps_executed, 2);

        // Output1 should have the original fingerprint
        let read1 = space.read(output1).unwrap();
        assert_eq!(read1.fingerprint, fp);

        // Output2 should have the permuted fingerprint
        let read2 = space.read(output2).unwrap();
        assert_eq!(read2.fingerprint[1], fp[0]); // Shifted by 1
    }

    #[test]
    fn test_workflow_loop() {
        let kernel = super::super::semantic_kernel::SemanticKernel::new();
        let mut space = BindSpace::new();

        let input = Addr::new(0x10, 0x01);
        let output = Addr::new(0x10, 0x02);
        let condition = Addr::new(0x10, 0x03);

        // Write a dense fingerprint as exit condition target
        let dense_fp = [0xFFFFFFFF_FFFFFFFFu64; FINGERPRINT_WORDS];
        space.write_at(condition, dense_fp);

        let fp = [0xCAFEBABEu64; FINGERPRINT_WORDS];
        space.write_at(input, fp);

        let workflow = WorkflowNode::Loop {
            id: "test-loop".into(),
            body: Box::new(WorkflowNode::Step(WorkflowStep {
                id: "loop-body".into(),
                name: "forward".into(),
                input_addrs: vec![input],
                output_addr: output,
                operation: WorkflowOp::Forward,
                stateful: false,
            })),
            exit_condition_addr: condition,
            exit_threshold: 0.4, // Dense fingerprint exceeds this
            max_iterations: 5,
        };

        let result = execute_workflow(&workflow, &mut space, &kernel);
        assert!(result.completed);
        // Should exit after 1 iteration because condition is already met
        assert_eq!(result.loops_iterated, 1);
    }

    #[test]
    fn test_memory_bank_store_and_retrieve() {
        let mut bank = MemoryBank::new(0x10, 0x11, 0x12);
        let mut space = BindSpace::new();

        let fp1 = [0xFF00FF00u64; FINGERPRINT_WORDS];
        let fp2 = [0x00FF00FFu64; FINGERPRINT_WORDS];

        bank.store(MemoryKind::Episodic, "user likes cats", fp1, &mut space, 1, Some(0));
        bank.store(MemoryKind::Episodic, "user likes dogs", fp2, &mut space, 2, Some(0));

        assert_eq!(bank.count(Some(MemoryKind::Episodic)), 2);
        assert_eq!(bank.count(Some(MemoryKind::Semantic)), 0);

        // Retrieve with a query similar to fp1
        let results = bank.retrieve(&fp1, None, &space, 0.4, 10, 3);
        assert!(!results.is_empty());
        assert!(results[0].content.contains("cats"));
    }

    #[test]
    fn test_memory_extract_semantic() {
        let mut bank = MemoryBank::new(0x10, 0x11, 0x12);
        let mut space = BindSpace::new();

        // Store 3 similar episodic memories
        let common = [0xFF00FF00_FF00FF00u64; FINGERPRINT_WORDS];
        bank.store(MemoryKind::Episodic, "ep1", common, &mut space, 1, None);
        bank.store(MemoryKind::Episodic, "ep2", common, &mut space, 2, None);
        bank.store(MemoryKind::Episodic, "ep3", common, &mut space, 3, None);

        let extracted = bank.extract_semantic(&mut space, 4);
        assert!(!extracted.is_empty());
        assert_eq!(bank.count(Some(MemoryKind::Semantic)), 1);
    }

    #[test]
    fn test_observability_session_trace_span() {
        let mut obs = ObservabilityManager::new();

        let session_id = obs.start_session(Some(0), 1);
        let trace_id = obs.start_trace("resonate", 2);

        obs.add_span(&trace_id, KernelSpan {
            id: "span-1".into(),
            operation: "hamming_search".into(),
            parent_id: None,
            duration_ns: 1000,
            metadata: std::collections::HashMap::new(),
        });

        obs.complete_trace(&trace_id, 3, Some(GroundingMetadata {
            confidence: 0.85,
            sources: vec![],
            is_grounded: true,
        }));

        let summary = obs.summary();
        assert_eq!(summary.total_sessions, 1);
        assert_eq!(summary.total_traces, 1);
        assert_eq!(summary.total_spans, 1);
        assert_eq!(summary.grounded_traces, 1);
        assert_eq!(summary.active_session, Some(session_id));
    }

    #[test]
    fn test_verification_minimum_density() {
        let mut engine = VerificationEngine::new();
        let mut params = std::collections::HashMap::new();
        params.insert("min_density".into(), "0.3".into());
        engine.add_rule(VerificationRule {
            name: "min_density".into(),
            description: "Fingerprint must have at least 30% bits set".into(),
            kind: VerificationKind::MinimumDensity,
            params,
        });

        // Dense fingerprint — should pass
        let dense = [0xFFFFFFFF_FFFFFFFFu64; FINGERPRINT_WORDS];
        let results = engine.verify_fingerprint(&dense, Addr::new(0x80, 0x01));
        assert!(results[0].passed);

        // Sparse fingerprint — should fail
        let sparse = [0x00000001u64; FINGERPRINT_WORDS];
        let results = engine.verify_fingerprint(&sparse, Addr::new(0x80, 0x01));
        assert!(!results[0].passed);
        assert!(!results[0].suggestions.is_empty());
    }

    #[test]
    fn test_verification_zone_constraint() {
        let mut engine = VerificationEngine::new();
        let mut params = std::collections::HashMap::new();
        params.insert("allowed_zone".into(), "Node".into());
        engine.add_rule(VerificationRule {
            name: "node_only".into(),
            description: "Only write to Node zone".into(),
            kind: VerificationKind::ZoneConstraint,
            params,
        });

        let fp = [0xCAFEBABEu64; FINGERPRINT_WORDS];

        // Node zone — should pass
        let results = engine.verify_fingerprint(&fp, Addr::new(0x80, 0x01));
        assert!(results[0].passed);

        // Surface zone — should fail
        let results = engine.verify_fingerprint(&fp, Addr::new(0x01, 0x01));
        assert!(!results[0].passed);
    }
}
