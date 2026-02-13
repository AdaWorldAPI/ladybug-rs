//! Semantic Kernel — BindSpace as Universal Computation Surface
//!
//! The 8+8 address model IS a semantic kernel: every query language, every
//! agent operation, every storage path hits the same 65,536 addresses.
//! This module makes that identity explicit and provides expansion points.
//!
//! # What Makes BindSpace a Semantic Kernel
//!
//! A semantic kernel maps heterogeneous operations to a uniform substrate.
//! BindSpace does this through its prefix:slot addressing:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    SEMANTIC KERNEL = BINDSPACE                          │
//! │                                                                         │
//! │  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐                  │
//! │  │   SURFACE    │  │    FLUID     │  │    NODES     │                  │
//! │  │  0x00-0x0F   │  │  0x10-0x7F   │  │  0x80-0xFF   │                  │
//! │  │              │  │              │  │              │                  │
//! │  │ Query Lang   │  │ Working Mem  │  │ Universal    │                  │
//! │  │ Orchestrate  │  │ Agent State  │  │ Bind Space   │                  │
//! │  │ CAM Ops      │  │ TTL Edges    │  │ All queries  │                  │
//! │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                  │
//! │         │                 │                 │                          │
//! │         └─────────────────┴─────────────────┘                          │
//! │                           │                                            │
//! │              ┌────────────┴────────────┐                               │
//! │              │    KERNEL OPERATIONS    │                               │
//! │              │                         │                               │
//! │              │  bind()      — write FP │                               │
//! │              │  query()     — read FP  │                               │
//! │              │  resonate()  — HDR srch │                               │
//! │              │  collapse()  — gate     │                               │
//! │              │  xor_bind()  — compose  │                               │
//! │              │  xor_unbind()— extract  │                               │
//! │              │  infer()     — NARS     │                               │
//! │              │  intervene() — causal   │                               │
//! │              │  imagine()   — counter. │                               │
//! │              │  crystallize()— commit  │                               │
//! │              └─────────────────────────┘                               │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Concurrency Model: Zero-Copy, Lock-Free Where Possible
//!
//! The kernel supports concurrent access through zone-level isolation:
//! - Each prefix is an independent array; reads to different prefixes
//!   require no synchronization (disjoint memory).
//! - Within a prefix, fingerprints are 256×u64 contiguous blocks.
//!   Reads are always zero-copy references into the backing array.
//! - Writes use prefix-level RwLock striping: concurrent reads anywhere,
//!   writes only lock the affected prefix.
//! - The kernel itself is `Send + Sync` — multiple agents can resonate
//!   concurrently across different zones.
//!
//! # Meta-Awareness
//!
//! The kernel can observe its own state through `introspect()`, which
//! reports population density, hot zones, and active operations per prefix.
//! This self-observation feeds the MetaOrchestrator's flow decisions.
//!
//! # Expansion Model
//!
//! The kernel expands through:
//! 1. **Prefix plugins** — new Surface prefixes (0x0C-0x0F already used)
//! 2. **Operator plugins** — custom CAM opcodes (4096 operation space)
//! 3. **Style plugins** — custom thinking styles beyond the 12 base
//! 4. **Protocol plugins** — new A2A message kinds
//! 5. **Collapse plugins** — custom gate evaluation strategies
//! 6. **Crystal plugins** — custom crystal geometries
//! 7. **Rung plugins** — causal rung escalation strategies
//! 8. **DataFusion plugins** — custom UDFs over kernel state

use crate::storage::bind_space::{Addr, BindSpace, FINGERPRINT_WORDS};
use serde::{Deserialize, Serialize};

// =============================================================================
// KERNEL ZONE
// =============================================================================

/// Named zones in the kernel's address space
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum KernelZone {
    /// Surface zone (0x00-0x0F): query languages, orchestration
    Surface { prefix: u8 },
    /// Fluid zone (0x10-0x7F): working memory, agent state, edges
    Fluid { prefix: u8 },
    /// Node zone (0x80-0xFF): universal bind targets
    Node { prefix: u8 },
}

impl KernelZone {
    pub fn from_prefix(prefix: u8) -> Self {
        match prefix {
            0x00..=0x0F => Self::Surface { prefix },
            0x10..=0x7F => Self::Fluid { prefix },
            0x80..=0xFF => Self::Node { prefix },
        }
    }

    pub fn prefix(&self) -> u8 {
        match self {
            Self::Surface { prefix } | Self::Fluid { prefix } | Self::Node { prefix } => *prefix,
        }
    }

    pub fn is_surface(&self) -> bool {
        matches!(self, Self::Surface { .. })
    }

    pub fn is_fluid(&self) -> bool {
        matches!(self, Self::Fluid { .. })
    }

    pub fn is_node(&self) -> bool {
        matches!(self, Self::Node { .. })
    }
}

// =============================================================================
// KERNEL OPERATIONS — The Semantic Instruction Set
// =============================================================================

/// A kernel operation — what the semantic kernel can do.
/// Each operation maps to a specific cognitive primitive.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum KernelOp {
    // === Core VSA operations (zero-copy on fingerprint arrays) ===
    /// Write fingerprint at address
    Bind { addr: Addr, label: Option<String> },
    /// Read fingerprint from address (zero-copy reference)
    Query { addr: Addr },
    /// XOR-compose two fingerprints (associative binding)
    XorBind {
        addr_a: Addr,
        addr_b: Addr,
        target: Addr,
    },
    /// XOR-extract component from composite (A ⊗ B ⊗ B = A)
    XorUnbind {
        composite: Addr,
        known: Addr,
        target: Addr,
    },
    /// Majority-vote bundle (noise elimination)
    Bundle { sources: Vec<Addr>, target: Addr },
    /// Permute fingerprint (position encoding)
    Permute {
        addr: Addr,
        shift: usize,
        target: Addr,
    },

    // === Search operations (parallel across prefixes) ===
    /// HDR resonance search (Hamming similarity)
    Resonate {
        zone: Option<KernelZone>,
        threshold: f32,
        limit: usize,
    },
    /// Mexican hat resonance (center excitation + surround inhibition)
    MexicanHat {
        zone: Option<KernelZone>,
        center_weight: f32,
        edge_weight: f32,
    },

    // === Collapse gate (dispersion → action) ===
    /// Evaluate collapse gate on candidate scores
    Collapse { candidates: Vec<Addr> },

    // === NARS inference (truth-value propagation) ===
    /// Deduction: {M→P, S→M} ⊢ S→P (strong conclusion)
    Deduce { premise1: Addr, premise2: Addr },
    /// Induction: {M→P, M→S} ⊢ S→P (weak generalization)
    Induce { premise1: Addr, premise2: Addr },
    /// Abduction: {P→M, S→M} ⊢ S→P (hypothesis generation)
    Abduct { premise1: Addr, premise2: Addr },
    /// Truth revision: update belief with new evidence
    Revise { existing: Addr, evidence: Addr },

    // === Causal inference (Pearl's 3 rungs) ===
    /// Rung 1 — SEE: P(Y|X) correlation query
    Correlate { x: Addr, k: usize },
    /// Rung 2 — DO: P(Y|do(X)) intervention query
    Intervene { state: Addr, action: Addr },
    /// Rung 3 — IMAGINE: P(Y_x|X=x') counterfactual query
    Imagine { state: Addr, alt_action: Addr },
    /// Rung escalation: auto-promote query to highest valid rung
    Escalate {
        state: Addr,
        action: Option<Addr>,
        alt_action: Option<Addr>,
    },

    // === Crystal operations (semantic memory) ===
    /// Crystallize: commit fingerprint from Fluid → Node zone
    Crystallize { fluid_addr: Addr, node_addr: Addr },
    /// Dissolve: move fingerprint from Node → Fluid zone (with TTL)
    Dissolve {
        node_addr: Addr,
        fluid_addr: Addr,
        ttl_seconds: u64,
    },
    /// Crystal query: search crystal geometry for semantic neighbors
    CrystalQuery { center: Addr, radius: usize },

    // === Meta-awareness (kernel observes itself) ===
    /// Introspect: report kernel population, hot zones, operation counts
    Introspect,
    /// Zone density: count populated slots in a zone
    ZoneDensity { zone: KernelZone },
}

// =============================================================================
// CAUSAL RUNG ESCALATION
// =============================================================================

/// The three rungs of causal reasoning (Pearl's causal hierarchy)
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum CausalRung {
    /// Rung 1: Association — P(Y|X) — "What do I see?"
    See = 1,
    /// Rung 2: Intervention — P(Y|do(X)) — "What if I do X?"
    Do = 2,
    /// Rung 3: Counterfactual — P(Y_x|X=x') — "What if I had done X instead?"
    Imagine = 3,
}

impl CausalRung {
    /// Escalate to the next rung (None if already at Imagine)
    pub fn escalate(self) -> Option<CausalRung> {
        match self {
            Self::See => Some(Self::Do),
            Self::Do => Some(Self::Imagine),
            Self::Imagine => None,
        }
    }

    /// Can this rung answer interventional queries?
    pub fn supports_intervention(self) -> bool {
        self >= Self::Do
    }

    /// Can this rung answer counterfactual queries?
    pub fn supports_counterfactual(self) -> bool {
        self >= Self::Imagine
    }
}

/// Result of a rung escalation attempt
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EscalationResult {
    /// The rung at which the query was answered
    pub answered_at: CausalRung,
    /// The highest rung attempted
    pub attempted_rung: CausalRung,
    /// Whether escalation was needed (lower rungs insufficient)
    pub escalated: bool,
    /// Why escalation stopped (if it did)
    pub ceiling_reason: Option<String>,
}

// =============================================================================
// NARS TRUTH VALUES (for kernel-level inference tracking)
// =============================================================================

/// NARS truth value — frequency and confidence
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct KernelTruth {
    /// Frequency: proportion of positive evidence (0.0-1.0)
    pub f: f32,
    /// Confidence: amount of evidence relative to total possible (0.0-1.0)
    pub c: f32,
}

impl KernelTruth {
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

    /// Deduction: {M→P <f1,c1>, S→M <f2,c2>} ⊢ S→P
    pub fn deduction(self, other: KernelTruth) -> KernelTruth {
        let f = self.f * other.f;
        let c = self.c * other.c * self.f * other.f;
        KernelTruth::new(f, c)
    }

    /// Induction: {M→P <f1,c1>, M→S <f2,c2>} ⊢ S→P
    pub fn induction(self, other: KernelTruth) -> KernelTruth {
        let f = other.f;
        let w_plus = self.f * self.c * other.f * other.c;
        let w = self.f * self.c * other.c;
        let c = if w > 0.0 { w_plus / (w + 1.0) } else { 0.0 };
        KernelTruth::new(f, c)
    }

    /// Abduction: {P→M <f1,c1>, S→M <f2,c2>} ⊢ S→P
    pub fn abduction(self, other: KernelTruth) -> KernelTruth {
        let f = self.f;
        let w_plus = self.f * self.c * other.f * other.c;
        let w = other.f * self.c * other.c;
        let c = if w > 0.0 { w_plus / (w + 1.0) } else { 0.0 };
        KernelTruth::new(f, c)
    }

    /// Revision: merge two independent truth values
    pub fn revision(self, other: KernelTruth) -> KernelTruth {
        let w1 = self.c / (1.0 - self.c + f32::EPSILON);
        let w2 = other.c / (1.0 - other.c + f32::EPSILON);
        let w = w1 + w2;
        if w < f32::EPSILON {
            return KernelTruth::new(0.5, 0.0);
        }
        let f = (w1 * self.f + w2 * other.f) / w;
        let c = w / (w + 1.0);
        KernelTruth::new(f, c)
    }
}

impl Default for KernelTruth {
    fn default() -> Self {
        Self { f: 0.5, c: 0.0 }
    }
}

// =============================================================================
// KERNEL INTROSPECTION (Meta-Awareness)
// =============================================================================

/// Per-prefix population and activity stats
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrefixStats {
    pub prefix: u8,
    pub zone: String,
    pub name: String,
    pub populated_slots: u16,
    pub total_popcount: u64,
    pub avg_density: f32,
}

/// Kernel self-observation — the kernel's awareness of its own state
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KernelIntrospection {
    /// Total populated addresses across all zones
    pub total_populated: usize,
    /// Population by zone
    pub surface_populated: usize,
    pub fluid_populated: usize,
    pub node_populated: usize,
    /// Per-prefix stats for non-empty prefixes
    pub hot_prefixes: Vec<PrefixStats>,
    /// Overall fingerprint density (avg popcount / max popcount)
    pub avg_density: f32,
    /// Addresses in the kernel with the highest popcount (most complex)
    pub most_complex: Vec<(Addr, u64)>,
}

// =============================================================================
// DATAFUSION SYNERGY — SQL OVER KERNEL STATE
// =============================================================================

/// Describes how a DataFusion query maps to kernel operations.
///
/// This enables SQL like:
/// ```sql
/// SELECT addr, similarity(fingerprint, :query) as sim
/// FROM kernel_nodes
/// WHERE similarity(fingerprint, :query) > 0.7
/// ORDER BY sim DESC
/// LIMIT 10
/// ```
///
/// Which translates to `KernelOp::Resonate` with zone=Node, threshold=0.7, limit=10.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataFusionMapping {
    /// Virtual table name in DataFusion
    pub table_name: String,
    /// Which kernel zone this table exposes
    pub zone: KernelZone,
    /// Column schema (Arrow-compatible)
    pub columns: Vec<DataFusionColumn>,
}

/// A column in the DataFusion virtual table
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataFusionColumn {
    pub name: String,
    pub data_type: String,
    pub description: String,
}

/// Generate the DataFusion virtual table mappings for the kernel.
///
/// These tables expose kernel state as SQL-queryable Arrow RecordBatches:
///
/// | Table | Zone | Contents |
/// |-------|------|----------|
/// | `kernel_surface` | Surface | Query language + orchestration FPs |
/// | `kernel_fluid` | Fluid | Working memory + agent state FPs |
/// | `kernel_nodes` | Node | Universal bind targets |
/// | `kernel_agents` | 0x0C | Agent cards + persona fingerprints |
/// | `kernel_styles` | 0x0D | Thinking style templates |
/// | `kernel_blackboards` | 0x0E | Agent blackboard states |
/// | `kernel_a2a` | 0x0F | A2A message channels |
/// | `kernel_causal` | 0x05 | Causal model fingerprints |
/// | `kernel_nars` | 0x04 | NARS inference addresses |
///
/// UDFs available in all queries:
/// - `hamming(a, b)` — Hamming distance between fingerprints
/// - `similarity(a, b)` — 1.0 - hamming/DIM (cosine-like)
/// - `popcount(x)` — set bits in fingerprint
/// - `xor_bind(a, b)` — XOR composition
/// - `truth_expect(f, c)` — NARS expectation: c * (f - 0.5) + 0.5
/// - `rung(addr)` — highest causal rung available for address
pub fn datafusion_table_mappings() -> Vec<DataFusionMapping> {
    let fp_columns = vec![
        DataFusionColumn {
            name: "prefix".into(),
            data_type: "UInt8".into(),
            description: "Prefix byte".into(),
        },
        DataFusionColumn {
            name: "slot".into(),
            data_type: "UInt8".into(),
            description: "Slot byte".into(),
        },
        DataFusionColumn {
            name: "addr".into(),
            data_type: "UInt16".into(),
            description: "Full 16-bit address".into(),
        },
        DataFusionColumn {
            name: "label".into(),
            data_type: "Utf8".into(),
            description: "Human-readable label".into(),
        },
        DataFusionColumn {
            name: "fingerprint".into(),
            data_type: "FixedSizeBinary(2048)".into(),
            description: "16K-bit fingerprint".into(),
        },
        DataFusionColumn {
            name: "popcount".into(),
            data_type: "UInt32".into(),
            description: "Set bits in fingerprint".into(),
        },
        DataFusionColumn {
            name: "zone".into(),
            data_type: "Utf8".into(),
            description: "Surface/Fluid/Node".into(),
        },
    ];

    vec![
        DataFusionMapping {
            table_name: "kernel_surface".into(),
            zone: KernelZone::Surface { prefix: 0x00 },
            columns: fp_columns.clone(),
        },
        DataFusionMapping {
            table_name: "kernel_fluid".into(),
            zone: KernelZone::Fluid { prefix: 0x10 },
            columns: fp_columns.clone(),
        },
        DataFusionMapping {
            table_name: "kernel_nodes".into(),
            zone: KernelZone::Node { prefix: 0x80 },
            columns: fp_columns.clone(),
        },
        DataFusionMapping {
            table_name: "kernel_agents".into(),
            zone: KernelZone::Surface { prefix: 0x0C },
            columns: fp_columns.clone(),
        },
        DataFusionMapping {
            table_name: "kernel_causal".into(),
            zone: KernelZone::Surface { prefix: 0x05 },
            columns: fp_columns.clone(),
        },
        DataFusionMapping {
            table_name: "kernel_nars".into(),
            zone: KernelZone::Surface { prefix: 0x04 },
            columns: fp_columns,
        },
    ]
}

// =============================================================================
// PREFIX ALLOCATION
// =============================================================================

/// Known prefix allocations in the kernel
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrefixAllocation {
    pub prefix: u8,
    pub name: String,
    pub purpose: String,
    pub slot_layout: String,
    pub active: bool,
    pub plugin: Option<String>,
}

/// The complete prefix map of the kernel
pub fn core_prefix_map() -> Vec<PrefixAllocation> {
    vec![
        PrefixAllocation {
            prefix: 0x00,
            name: "lance".into(),
            purpose: "Lance query addresses".into(),
            slot_layout: "0x00-0xFF: query fingerprints".into(),
            active: true,
            plugin: None,
        },
        PrefixAllocation {
            prefix: 0x01,
            name: "sql".into(),
            purpose: "SQL query addresses".into(),
            slot_layout: "0x00-0xFF: query fingerprints".into(),
            active: true,
            plugin: None,
        },
        PrefixAllocation {
            prefix: 0x02,
            name: "cypher".into(),
            purpose: "Cypher graph query addresses".into(),
            slot_layout: "0x00-0xFF: query fingerprints".into(),
            active: true,
            plugin: None,
        },
        PrefixAllocation {
            prefix: 0x03,
            name: "graphql".into(),
            purpose: "GraphQL query addresses".into(),
            slot_layout: "0x00-0xFF: query fingerprints".into(),
            active: true,
            plugin: None,
        },
        PrefixAllocation {
            prefix: 0x04,
            name: "nars".into(),
            purpose: "NARS inference addresses".into(),
            slot_layout: "0x00-0xFF: NAL term fingerprints".into(),
            active: true,
            plugin: None,
        },
        PrefixAllocation {
            prefix: 0x05,
            name: "causal".into(),
            purpose: "Causal inference (Pearl's 3 rungs)".into(),
            slot_layout: "0x00-0xFF: causal model fingerprints".into(),
            active: true,
            plugin: None,
        },
        PrefixAllocation {
            prefix: 0x06,
            name: "meta".into(),
            purpose: "Meta-cognitive state".into(),
            slot_layout: "0x00-0xFF: meta fingerprints".into(),
            active: true,
            plugin: None,
        },
        PrefixAllocation {
            prefix: 0x07,
            name: "verbs".into(),
            purpose: "Verb/action fingerprints".into(),
            slot_layout: "0x00-0xFF: verb fingerprints".into(),
            active: true,
            plugin: None,
        },
        PrefixAllocation {
            prefix: 0x08,
            name: "concepts".into(),
            purpose: "Concept fingerprints".into(),
            slot_layout: "0x00-0xFF: concept fingerprints".into(),
            active: true,
            plugin: None,
        },
        PrefixAllocation {
            prefix: 0x09,
            name: "qualia".into(),
            purpose: "Qualia/experience state".into(),
            slot_layout: "0x00-0xFF: qualia fingerprints".into(),
            active: true,
            plugin: None,
        },
        PrefixAllocation {
            prefix: 0x0A,
            name: "memory".into(),
            purpose: "Memory management".into(),
            slot_layout: "0x00-0xFF: memory fingerprints".into(),
            active: true,
            plugin: None,
        },
        PrefixAllocation {
            prefix: 0x0B,
            name: "learning".into(),
            purpose: "Learning/CAM ops".into(),
            slot_layout: "0x00-0xFF: learning fingerprints".into(),
            active: true,
            plugin: None,
        },
        PrefixAllocation {
            prefix: 0x0C,
            name: "agents".into(),
            purpose: "Agent Registry (crewAI)".into(),
            slot_layout: "0x00-0x7F: agent cards, 0x80-0xFF: persona fingerprints".into(),
            active: true,
            plugin: Some("orchestration".into()),
        },
        PrefixAllocation {
            prefix: 0x0D,
            name: "thinking".into(),
            purpose: "Thinking Style Templates".into(),
            slot_layout: "0x00-0x0B: 12 base, 0x0C-0xFF: custom".into(),
            active: true,
            plugin: Some("orchestration".into()),
        },
        PrefixAllocation {
            prefix: 0x0E,
            name: "blackboard".into(),
            purpose: "Agent Blackboards".into(),
            slot_layout: "0x00-0xFF: per-agent state".into(),
            active: true,
            plugin: Some("orchestration".into()),
        },
        PrefixAllocation {
            prefix: 0x0F,
            name: "a2a".into(),
            purpose: "A2A Message Routing".into(),
            slot_layout: "0x00-0xFF: directional channels".into(),
            active: true,
            plugin: Some("orchestration".into()),
        },
    ]
}

// =============================================================================
// EXPANSION REGISTRY
// =============================================================================

/// Plugin-defined operator that hooks into the kernel
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KernelOperator {
    pub name: String,
    pub opcode_range: (u16, u16),
    pub target_prefix: u8,
    pub description: String,
}

/// Plugin-defined message kind for A2A protocol extension
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProtocolExtension {
    pub name: String,
    pub message_kind: String,
    pub description: String,
}

/// Plugin-defined collapse strategy
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CollapseStrategy {
    pub name: String,
    pub flow_threshold: Option<f32>,
    pub block_threshold: Option<f32>,
    pub description: String,
}

/// Plugin-defined crystal geometry
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CrystalPlugin {
    pub name: String,
    /// Dimensionality (e.g., 3 for 5x5x5 ContextCrystal, 5 for 5^5 SentenceCrystal)
    pub dimensions: usize,
    /// Cells per dimension
    pub cells_per_dim: usize,
    /// Total cells in the crystal
    pub total_cells: usize,
    /// Which Fluid prefix range this crystal occupies
    pub fluid_prefix_start: u8,
    /// How many prefixes this crystal uses
    pub prefix_count: u8,
    pub description: String,
}

/// Plugin-defined causal rung escalation strategy
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RungEscalationStrategy {
    pub name: String,
    /// Minimum confidence to accept a rung-1 answer (skip escalation)
    pub see_confidence_floor: f32,
    /// Minimum confidence to accept a rung-2 answer (skip to rung-3)
    pub do_confidence_floor: f32,
    /// Whether to auto-escalate when confounders are detected
    pub auto_escalate_on_confounders: bool,
    pub description: String,
}

/// The expansion registry tracks all kernel extensions
pub struct ExpansionRegistry {
    prefix_plugins: Vec<PrefixAllocation>,
    operators: Vec<KernelOperator>,
    protocols: Vec<ProtocolExtension>,
    strategies: Vec<CollapseStrategy>,
    crystals: Vec<CrystalPlugin>,
    rung_strategies: Vec<RungEscalationStrategy>,
    next_fluid_prefix: u8,
}

impl ExpansionRegistry {
    pub fn new() -> Self {
        Self {
            prefix_plugins: Vec::new(),
            operators: Vec::new(),
            protocols: Vec::new(),
            strategies: Vec::new(),
            crystals: Vec::new(),
            rung_strategies: Vec::new(),
            next_fluid_prefix: 0x10,
        }
    }

    pub fn allocate_prefix(
        &mut self,
        name: &str,
        purpose: &str,
        plugin: &str,
    ) -> Result<u8, String> {
        if self.next_fluid_prefix >= 0x80 {
            return Err("Fluid zone exhausted (112 prefixes max)".into());
        }
        let prefix = self.next_fluid_prefix;
        self.prefix_plugins.push(PrefixAllocation {
            prefix,
            name: name.into(),
            purpose: purpose.into(),
            slot_layout: "Plugin-defined".into(),
            active: true,
            plugin: Some(plugin.into()),
        });
        self.next_fluid_prefix += 1;
        Ok(prefix)
    }

    pub fn register_operator(&mut self, op: KernelOperator) -> Result<(), String> {
        if self.operators.iter().any(|o| o.name == op.name) {
            return Err(format!("Operator '{}' already registered", op.name));
        }
        self.operators.push(op);
        Ok(())
    }

    pub fn register_protocol(&mut self, ext: ProtocolExtension) -> Result<(), String> {
        if self.protocols.iter().any(|p| p.name == ext.name) {
            return Err(format!("Protocol '{}' already registered", ext.name));
        }
        self.protocols.push(ext);
        Ok(())
    }

    pub fn register_strategy(&mut self, strategy: CollapseStrategy) -> Result<(), String> {
        if self.strategies.iter().any(|s| s.name == strategy.name) {
            return Err(format!("Strategy '{}' already registered", strategy.name));
        }
        self.strategies.push(strategy);
        Ok(())
    }

    pub fn register_crystal(&mut self, crystal: CrystalPlugin) -> Result<(), String> {
        if self.crystals.iter().any(|c| c.name == crystal.name) {
            return Err(format!("Crystal '{}' already registered", crystal.name));
        }
        self.crystals.push(crystal);
        Ok(())
    }

    pub fn register_rung_strategy(
        &mut self,
        strategy: RungEscalationStrategy,
    ) -> Result<(), String> {
        if self.rung_strategies.iter().any(|s| s.name == strategy.name) {
            return Err(format!(
                "Rung strategy '{}' already registered",
                strategy.name
            ));
        }
        self.rung_strategies.push(strategy);
        Ok(())
    }

    pub fn all_prefixes(&self) -> Vec<PrefixAllocation> {
        let mut all = core_prefix_map();
        all.extend(self.prefix_plugins.iter().cloned());
        all
    }

    pub fn operators(&self) -> &[KernelOperator] {
        &self.operators
    }
    pub fn protocols(&self) -> &[ProtocolExtension] {
        &self.protocols
    }
    pub fn strategies(&self) -> &[CollapseStrategy] {
        &self.strategies
    }
    pub fn crystals(&self) -> &[CrystalPlugin] {
        &self.crystals
    }
    pub fn rung_strategies(&self) -> &[RungEscalationStrategy] {
        &self.rung_strategies
    }

    pub fn summary(&self) -> ExpansionSummary {
        ExpansionSummary {
            core_prefixes: 16,
            plugin_prefixes: self.prefix_plugins.len(),
            available_fluid_prefixes: (0x80u8.wrapping_sub(self.next_fluid_prefix)) as usize,
            operators: self.operators.len(),
            protocols: self.protocols.len(),
            strategies: self.strategies.len(),
            crystals: self.crystals.len(),
            rung_strategies: self.rung_strategies.len(),
        }
    }
}

impl Default for ExpansionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of kernel expansion state
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExpansionSummary {
    pub core_prefixes: usize,
    pub plugin_prefixes: usize,
    pub available_fluid_prefixes: usize,
    pub operators: usize,
    pub protocols: usize,
    pub strategies: usize,
    pub crystals: usize,
    pub rung_strategies: usize,
}

// =============================================================================
// SEMANTIC KERNEL FACADE
// =============================================================================

/// The SemanticKernel provides a unified API over BindSpace, treating it
/// as a concurrent computation surface rather than just storage.
///
/// # Thread Safety
///
/// The SemanticKernel itself is `Send + Sync`. It holds no mutable state
/// except the expansion registry (which is only mutated during setup).
/// All mutable BindSpace access goes through `&mut BindSpace` or
/// `Arc<RwLock<BindSpace>>` at the call site — the kernel doesn't own
/// the BindSpace, it operates ON it.
///
/// This means multiple agents can call kernel operations concurrently
/// as long as they hold appropriate locks on the BindSpace. Reads are
/// zero-copy references into the backing arrays.
pub struct SemanticKernel {
    pub expansion: ExpansionRegistry,
}

impl SemanticKernel {
    pub fn new() -> Self {
        Self {
            expansion: ExpansionRegistry::new(),
        }
    }

    // =========================================================================
    // CORE VSA OPERATIONS (zero-copy on fingerprint arrays)
    // =========================================================================

    /// Bind: write a fingerprint at an address
    pub fn bind(
        &self,
        space: &mut BindSpace,
        addr: Addr,
        fingerprint: [u64; FINGERPRINT_WORDS],
        label: Option<&str>,
    ) {
        space.write_at(addr, fingerprint);
        if let Some(lbl) = label {
            if let Some(node) = space.read_mut(addr) {
                node.label = Some(lbl.to_string());
            }
        }
    }

    /// Query: read a fingerprint from an address (zero-copy reference semantics)
    pub fn query(&self, space: &BindSpace, addr: Addr) -> Option<[u64; FINGERPRINT_WORDS]> {
        space.read(addr).map(|node| node.fingerprint)
    }

    /// XOR-bind: compose two fingerprints into a third address.
    /// A ⊗ B encodes the relationship "A is associated with B".
    pub fn xor_bind(
        &self,
        space: &mut BindSpace,
        addr_a: Addr,
        addr_b: Addr,
        target: Addr,
        label: Option<&str>,
    ) -> bool {
        let (fp_a, fp_b) = match (space.read(addr_a), space.read(addr_b)) {
            (Some(a), Some(b)) => (a.fingerprint, b.fingerprint),
            _ => return false,
        };

        let mut composed = [0u64; FINGERPRINT_WORDS];
        for i in 0..FINGERPRINT_WORDS {
            composed[i] = fp_a[i] ^ fp_b[i];
        }
        self.bind(space, target, composed, label);
        true
    }

    /// XOR-unbind: given composite A⊗B and B, extract A.
    /// Uses the identity: A ⊗ B ⊗ B = A (XOR is self-inverse).
    pub fn xor_unbind(
        &self,
        space: &mut BindSpace,
        composite_addr: Addr,
        known_addr: Addr,
        target: Addr,
        label: Option<&str>,
    ) -> bool {
        // XOR-unbind is the same operation as XOR-bind (XOR is self-inverse)
        self.xor_bind(space, composite_addr, known_addr, target, label)
    }

    /// Bundle: majority-vote of multiple fingerprints (noise elimination).
    /// For each bit position, the output bit is 1 if >50% of inputs have 1.
    pub fn bundle(
        &self,
        space: &mut BindSpace,
        source_addrs: &[Addr],
        target: Addr,
        label: Option<&str>,
    ) -> bool {
        if source_addrs.is_empty() {
            return false;
        }

        let sources: Vec<[u64; FINGERPRINT_WORDS]> = source_addrs
            .iter()
            .filter_map(|&addr| space.read(addr).map(|n| n.fingerprint))
            .collect();

        if sources.is_empty() {
            return false;
        }

        let threshold = sources.len() / 2;
        let mut result = [0u64; FINGERPRINT_WORDS];

        for word_idx in 0..FINGERPRINT_WORDS {
            for bit in 0..64 {
                let mask = 1u64 << bit;
                let count = sources.iter().filter(|fp| fp[word_idx] & mask != 0).count();
                if count > threshold {
                    result[word_idx] |= mask;
                }
            }
        }

        self.bind(space, target, result, label);
        true
    }

    /// Permute: circular shift fingerprint words (position encoding).
    /// Used for temporal/structural dimension encoding in crystals.
    pub fn permute(
        &self,
        space: &mut BindSpace,
        source: Addr,
        shift: usize,
        target: Addr,
        label: Option<&str>,
    ) -> bool {
        let fp = match space.read(source) {
            Some(node) => node.fingerprint,
            None => return false,
        };

        let mut permuted = [0u64; FINGERPRINT_WORDS];
        let shift = shift % FINGERPRINT_WORDS;
        for i in 0..FINGERPRINT_WORDS {
            permuted[(i + shift) % FINGERPRINT_WORDS] = fp[i];
        }
        self.bind(space, target, permuted, label);
        true
    }

    // =========================================================================
    // SEARCH (parallel-safe: reads only, no mutation)
    // =========================================================================

    /// Hamming similarity between two fingerprints (pure function, zero-copy)
    pub fn hamming_similarity(a: &[u64; FINGERPRINT_WORDS], b: &[u64; FINGERPRINT_WORDS]) -> f32 {
        let total_bits = (FINGERPRINT_WORDS * 64) as f32;
        let matching: u32 = (0..FINGERPRINT_WORDS)
            .map(|i| (!(a[i] ^ b[i])).count_ones())
            .sum();
        matching as f32 / total_bits
    }

    /// Resonance search: find addresses with similar fingerprints.
    /// This is a read-only operation — safe for concurrent access.
    pub fn resonate(
        &self,
        space: &BindSpace,
        target: &[u64; FINGERPRINT_WORDS],
        zone: Option<KernelZone>,
        threshold: f32,
        limit: usize,
    ) -> Vec<(Addr, f32)> {
        let mut results = Vec::new();

        let (start_prefix, end_prefix) = match zone {
            Some(KernelZone::Surface { prefix }) => (prefix, prefix),
            Some(KernelZone::Fluid { prefix }) => (prefix, prefix),
            Some(KernelZone::Node { prefix }) => (prefix, prefix),
            None => (0x00, 0xFF),
        };

        for prefix in start_prefix..=end_prefix {
            for slot in 0..=255u8 {
                let addr = Addr::new(prefix, slot);
                if let Some(node) = space.read(addr) {
                    if node.fingerprint.iter().all(|&w| w == 0) {
                        continue;
                    }
                    let similarity = Self::hamming_similarity(target, &node.fingerprint);
                    if similarity >= threshold {
                        results.push((addr, similarity));
                    }
                }
            }
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }

    // =========================================================================
    // COLLAPSE GATE (read-only evaluation)
    // =========================================================================

    /// Evaluate collapse gate on fingerprints at the given addresses.
    /// Returns (GateState, SD, winner_index, winner_similarity).
    /// This is read-only — safe for concurrent access.
    pub fn collapse(
        &self,
        space: &BindSpace,
        candidate_addrs: &[Addr],
        reference: &[u64; FINGERPRINT_WORDS],
    ) -> (crate::cognitive::GateState, f32, Option<usize>, Option<f32>) {
        let scores: Vec<f32> = candidate_addrs
            .iter()
            .filter_map(|&addr| {
                space
                    .read(addr)
                    .map(|n| Self::hamming_similarity(&n.fingerprint, reference))
            })
            .collect();

        if scores.is_empty() {
            return (
                crate::cognitive::GateState::Block,
                f32::INFINITY,
                None,
                None,
            );
        }

        let sd = crate::cognitive::calculate_sd(&scores);
        let gate = crate::cognitive::get_gate_state(sd);

        let (winner_idx, winner_score) = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, &s)| (i, s))
            .unwrap_or((0, 0.0));

        (gate, sd, Some(winner_idx), Some(winner_score))
    }

    // =========================================================================
    // NARS INFERENCE (pure functions on truth values)
    // =========================================================================

    /// Deduction at kernel level: combine two truth-annotated addresses
    pub fn deduce(&self, t1: KernelTruth, t2: KernelTruth) -> KernelTruth {
        t1.deduction(t2)
    }

    /// Induction at kernel level
    pub fn induce(&self, t1: KernelTruth, t2: KernelTruth) -> KernelTruth {
        t1.induction(t2)
    }

    /// Abduction at kernel level
    pub fn abduct(&self, t1: KernelTruth, t2: KernelTruth) -> KernelTruth {
        t1.abduction(t2)
    }

    /// Revision at kernel level
    pub fn revise(&self, existing: KernelTruth, evidence: KernelTruth) -> KernelTruth {
        existing.revision(evidence)
    }

    // =========================================================================
    // CAUSAL RUNG ESCALATION
    // =========================================================================

    /// Attempt rung escalation: start at See, escalate to Do, then Imagine.
    /// Returns the highest rung at which an answer was found.
    pub fn escalate_rung(
        &self,
        space: &BindSpace,
        state_fp: &[u64; FINGERPRINT_WORDS],
        action_fp: Option<&[u64; FINGERPRINT_WORDS]>,
        threshold: f32,
    ) -> EscalationResult {
        // Rung 1: SEE — correlate state with causal zone
        let rung1_hits = self.resonate(
            space,
            state_fp,
            Some(KernelZone::Surface { prefix: 0x05 }),
            threshold,
            5,
        );

        if !rung1_hits.is_empty() && action_fp.is_none() {
            return EscalationResult {
                answered_at: CausalRung::See,
                attempted_rung: CausalRung::See,
                escalated: false,
                ceiling_reason: None,
            };
        }

        // Rung 2: DO — check for intervention data
        if let Some(action) = action_fp {
            // Compose state ⊗ action to look for intervention records
            let mut intervention_fp = [0u64; FINGERPRINT_WORDS];
            for i in 0..FINGERPRINT_WORDS {
                intervention_fp[i] = state_fp[i] ^ action[i];
            }

            let rung2_hits = self.resonate(
                space,
                &intervention_fp,
                Some(KernelZone::Surface { prefix: 0x05 }),
                threshold,
                5,
            );

            if !rung2_hits.is_empty() {
                return EscalationResult {
                    answered_at: CausalRung::Do,
                    attempted_rung: CausalRung::Do,
                    escalated: true,
                    ceiling_reason: None,
                };
            }

            // Rung 3: IMAGINE — counterfactual (permuted intervention)
            let mut counterfactual_fp = [0u64; FINGERPRINT_WORDS];
            let shift = 1;
            for i in 0..FINGERPRINT_WORDS {
                counterfactual_fp[(i + shift) % FINGERPRINT_WORDS] = intervention_fp[i];
            }

            let rung3_hits = self.resonate(
                space,
                &counterfactual_fp,
                Some(KernelZone::Surface { prefix: 0x05 }),
                threshold * 0.8, // lower threshold for counterfactuals
                5,
            );

            if !rung3_hits.is_empty() {
                return EscalationResult {
                    answered_at: CausalRung::Imagine,
                    attempted_rung: CausalRung::Imagine,
                    escalated: true,
                    ceiling_reason: None,
                };
            }

            return EscalationResult {
                answered_at: CausalRung::See,
                attempted_rung: CausalRung::Imagine,
                escalated: true,
                ceiling_reason: Some("No intervention or counterfactual data found".into()),
            };
        }

        EscalationResult {
            answered_at: CausalRung::See,
            attempted_rung: CausalRung::See,
            escalated: false,
            ceiling_reason: if rung1_hits.is_empty() {
                Some("No causal data found at any rung".into())
            } else {
                None
            },
        }
    }

    // =========================================================================
    // CRYSTAL OPERATIONS
    // =========================================================================

    /// Crystallize: promote fingerprint from Fluid → Node zone.
    /// This is the "commit" operation — meaning has stabilized enough
    /// to become a permanent concept.
    pub fn crystallize(&self, space: &mut BindSpace, fluid_addr: Addr, node_addr: Addr) -> bool {
        let fp = match space.read(fluid_addr) {
            Some(node) => node.fingerprint,
            None => return false,
        };

        let label = space
            .read(fluid_addr)
            .and_then(|n| n.label.clone())
            .map(|l| format!("crystallized:{}", l));

        self.bind(space, node_addr, fp, label.as_deref());
        true
    }

    /// Dissolve: move fingerprint from Node → Fluid zone.
    /// The fingerprint gets a TTL in the fluid zone and will eventually
    /// evict unless re-crystallized.
    pub fn dissolve(&self, space: &mut BindSpace, node_addr: Addr, fluid_addr: Addr) -> bool {
        let fp = match space.read(node_addr) {
            Some(node) => node.fingerprint,
            None => return false,
        };

        let label = space
            .read(node_addr)
            .and_then(|n| n.label.clone())
            .map(|l| format!("dissolved:{}", l));

        self.bind(space, fluid_addr, fp, label.as_deref());
        true
    }

    // =========================================================================
    // META-AWARENESS (kernel observes itself)
    // =========================================================================

    /// Introspect: the kernel observes its own state.
    /// This feeds the MetaOrchestrator's flow decisions.
    /// Read-only — safe for concurrent access.
    pub fn introspect(&self, space: &BindSpace) -> KernelIntrospection {
        let mut total_populated = 0usize;
        let mut surface_populated = 0usize;
        let mut fluid_populated = 0usize;
        let mut node_populated = 0usize;
        let mut prefix_stats: Vec<PrefixStats> = Vec::new();
        let mut most_complex: Vec<(Addr, u64)> = Vec::new();
        let mut total_popcount_all = 0u64;
        let mut total_addrs_checked = 0u64;

        for prefix in 0..=255u8 {
            let mut populated_slots = 0u16;
            let mut prefix_popcount = 0u64;

            for slot in 0..=255u8 {
                let addr = Addr::new(prefix, slot);
                if let Some(node) = space.read(addr) {
                    if node.fingerprint.iter().any(|&w| w != 0) {
                        populated_slots += 1;
                        let pc: u64 = node.fingerprint.iter().map(|w| w.count_ones() as u64).sum();
                        prefix_popcount += pc;
                        total_popcount_all += pc;

                        // Track most complex
                        if most_complex.len() < 10
                            || pc > most_complex.last().map(|x| x.1).unwrap_or(0)
                        {
                            most_complex.push((addr, pc));
                            most_complex.sort_by(|a, b| b.1.cmp(&a.1));
                            most_complex.truncate(10);
                        }
                    }
                }
                total_addrs_checked += 1;
            }

            if populated_slots > 0 {
                total_populated += populated_slots as usize;
                match KernelZone::from_prefix(prefix) {
                    KernelZone::Surface { .. } => surface_populated += populated_slots as usize,
                    KernelZone::Fluid { .. } => fluid_populated += populated_slots as usize,
                    KernelZone::Node { .. } => node_populated += populated_slots as usize,
                }

                let name = self
                    .expansion
                    .all_prefixes()
                    .iter()
                    .find(|p| p.prefix == prefix)
                    .map(|p| p.name.clone())
                    .unwrap_or_else(|| format!("0x{:02X}", prefix));

                let max_popcount = (FINGERPRINT_WORDS * 64) as f32;
                let avg_density = if populated_slots > 0 {
                    (prefix_popcount as f32 / populated_slots as f32) / max_popcount
                } else {
                    0.0
                };

                prefix_stats.push(PrefixStats {
                    prefix,
                    zone: format!("{:?}", KernelZone::from_prefix(prefix)),
                    name,
                    populated_slots,
                    total_popcount: prefix_popcount,
                    avg_density,
                });
            }
        }

        let max_total_popcount = (FINGERPRINT_WORDS * 64) as f32 * total_addrs_checked as f32;
        let avg_density = if max_total_popcount > 0.0 {
            total_popcount_all as f32 / max_total_popcount
        } else {
            0.0
        };

        KernelIntrospection {
            total_populated,
            surface_populated,
            fluid_populated,
            node_populated,
            hot_prefixes: prefix_stats,
            avg_density,
            most_complex,
        }
    }

    /// Quick zone density check (cheaper than full introspect).
    /// Read-only — safe for concurrent access.
    pub fn zone_density(&self, space: &BindSpace, zone: &KernelZone) -> u16 {
        let prefix = zone.prefix();
        let mut count = 0u16;
        for slot in 0..=255u8 {
            if let Some(node) = space.read(Addr::new(prefix, slot)) {
                if node.fingerprint.iter().any(|&w| w != 0) {
                    count += 1;
                }
            }
        }
        count
    }

    /// Identify which semantic zone an address belongs to
    pub fn zone_of(&self, addr: Addr) -> KernelZone {
        KernelZone::from_prefix(addr.prefix())
    }

    /// Get the prefix allocation for an address
    pub fn prefix_info(&self, prefix: u8) -> Option<PrefixAllocation> {
        self.expansion
            .all_prefixes()
            .into_iter()
            .find(|p| p.prefix == prefix)
    }

    /// Describe the kernel's current state
    pub fn describe(&self) -> KernelDescription {
        let expansion = self.expansion.summary();
        KernelDescription {
            address_model: "8+8 (prefix:slot, 65536 addresses)".to_string(),
            zones: vec![
                "Surface (0x00-0x0F): Query languages + Orchestration".to_string(),
                "Fluid (0x10-0x7F): Working memory + Agent state + Crystal".to_string(),
                "Node (0x80-0xFF): Universal bind targets (crystallized)".to_string(),
            ],
            fingerprint_width: format!(
                "{} words x 64 bits = {} bits",
                FINGERPRINT_WORDS,
                FINGERPRINT_WORDS * 64
            ),
            operations: vec![
                "bind/query — read/write fingerprints".to_string(),
                "xor_bind/xor_unbind — VSA composition (O(1) encode/decode)".to_string(),
                "bundle — majority-vote noise elimination".to_string(),
                "permute — position encoding for crystals".to_string(),
                "resonate — HDR Hamming similarity search".to_string(),
                "collapse — dispersion gate (Flow/Hold/Block)".to_string(),
                "deduce/induce/abduct/revise — NARS inference".to_string(),
                "escalate_rung — Pearl's causal hierarchy (See/Do/Imagine)".to_string(),
                "crystallize/dissolve — Fluid ↔ Node promotion".to_string(),
                "introspect — kernel self-awareness".to_string(),
            ],
            datafusion_tables: datafusion_table_mappings()
                .iter()
                .map(|m| format!("{} ({})", m.table_name, m.zone.prefix()))
                .collect(),
            expansion,
        }
    }
}

impl Default for SemanticKernel {
    fn default() -> Self {
        Self::new()
    }
}

/// Description of the kernel for introspection
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KernelDescription {
    pub address_model: String,
    pub zones: Vec<String>,
    pub fingerprint_width: String,
    pub operations: Vec<String>,
    pub datafusion_tables: Vec<String>,
    pub expansion: ExpansionSummary,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_zone_classification() {
        assert_eq!(
            KernelZone::from_prefix(0x00),
            KernelZone::Surface { prefix: 0x00 }
        );
        assert_eq!(
            KernelZone::from_prefix(0x0F),
            KernelZone::Surface { prefix: 0x0F }
        );
        assert_eq!(
            KernelZone::from_prefix(0x10),
            KernelZone::Fluid { prefix: 0x10 }
        );
        assert_eq!(
            KernelZone::from_prefix(0x7F),
            KernelZone::Fluid { prefix: 0x7F }
        );
        assert_eq!(
            KernelZone::from_prefix(0x80),
            KernelZone::Node { prefix: 0x80 }
        );
    }

    #[test]
    fn test_core_prefix_map() {
        let map = core_prefix_map();
        assert_eq!(map.len(), 16);
        assert_eq!(map[0].name, "lance");
        assert_eq!(map[12].name, "agents");
        assert_eq!(map[15].name, "a2a");
    }

    #[test]
    fn test_expansion_prefix_allocation() {
        let mut reg = ExpansionRegistry::new();
        let prefix = reg
            .allocate_prefix("custom_memory", "Plugin working memory", "my_plugin")
            .unwrap();
        assert_eq!(prefix, 0x10);
        let prefix2 = reg
            .allocate_prefix("custom_edges", "Plugin edge store", "my_plugin")
            .unwrap();
        assert_eq!(prefix2, 0x11);
        let summary = reg.summary();
        assert_eq!(summary.plugin_prefixes, 2);
    }

    #[test]
    fn test_operator_registration() {
        let mut reg = ExpansionRegistry::new();
        reg.register_operator(KernelOperator {
            name: "custom_op".into(),
            opcode_range: (0x100, 0x1FF),
            target_prefix: 0x10,
            description: "Custom".into(),
        })
        .unwrap();
        assert_eq!(reg.operators().len(), 1);
        assert!(
            reg.register_operator(KernelOperator {
                name: "custom_op".into(),
                opcode_range: (0x200, 0x2FF),
                target_prefix: 0x11,
                description: "Dup".into(),
            })
            .is_err()
        );
    }

    #[test]
    fn test_bind_and_query() {
        let kernel = SemanticKernel::new();
        let mut space = BindSpace::new();
        let addr = Addr::new(0x80, 0x01);
        let fp = [42u64; FINGERPRINT_WORDS];
        kernel.bind(&mut space, addr, fp, Some("test:node"));
        assert_eq!(kernel.query(&space, addr), Some(fp));
    }

    #[test]
    fn test_xor_bind_unbind_roundtrip() {
        let kernel = SemanticKernel::new();
        let mut space = BindSpace::new();

        let a = Addr::new(0x80, 0x01);
        let b = Addr::new(0x80, 0x02);
        let composite = Addr::new(0x80, 0x03);
        let recovered = Addr::new(0x80, 0x04);

        let fp_a = [0xDEADBEEF_CAFEBABEu64; FINGERPRINT_WORDS];
        let fp_b = [0x12345678_9ABCDEF0u64; FINGERPRINT_WORDS];

        kernel.bind(&mut space, a, fp_a, Some("A"));
        kernel.bind(&mut space, b, fp_b, Some("B"));

        // Bind A ⊗ B
        assert!(kernel.xor_bind(&mut space, a, b, composite, Some("A⊗B")));

        // Unbind: (A⊗B) ⊗ B = A
        assert!(kernel.xor_unbind(&mut space, composite, b, recovered, Some("recovered-A")));

        assert_eq!(kernel.query(&space, recovered), Some(fp_a));
    }

    #[test]
    fn test_bundle_majority_vote() {
        let kernel = SemanticKernel::new();
        let mut space = BindSpace::new();

        // 3 similar fingerprints + 1 outlier
        let common = [0xFFFF_FFFF_FFFF_FFFFu64; FINGERPRINT_WORDS];
        let outlier = [0x0000_0000_0000_0000u64; FINGERPRINT_WORDS];

        let a1 = Addr::new(0x80, 0x01);
        let a2 = Addr::new(0x80, 0x02);
        let a3 = Addr::new(0x80, 0x03);
        let a4 = Addr::new(0x80, 0x04);
        let target = Addr::new(0x80, 0x05);

        kernel.bind(&mut space, a1, common, None);
        kernel.bind(&mut space, a2, common, None);
        kernel.bind(&mut space, a3, common, None);
        kernel.bind(&mut space, a4, outlier, None);

        assert!(kernel.bundle(&mut space, &[a1, a2, a3, a4], target, Some("bundled")));

        // Majority wins: should be all-1s
        let result = kernel.query(&space, target).unwrap();
        assert_eq!(result, common);
    }

    #[test]
    fn test_permute() {
        let kernel = SemanticKernel::new();
        let mut space = BindSpace::new();

        let mut fp = [0u64; FINGERPRINT_WORDS];
        fp[0] = 0xDEADBEEF;
        fp[1] = 0xCAFEBABE;

        let src = Addr::new(0x80, 0x01);
        let dst = Addr::new(0x80, 0x02);
        kernel.bind(&mut space, src, fp, None);
        assert!(kernel.permute(&mut space, src, 1, dst, None));

        let result = kernel.query(&space, dst).unwrap();
        assert_eq!(result[1], 0xDEADBEEF);
        assert_eq!(result[2], 0xCAFEBABE);
    }

    #[test]
    fn test_hamming_similarity() {
        let a = [0xFF00FF00_FF00FF00u64; FINGERPRINT_WORDS];
        let b = [0xFF00FF00_FF00FF00u64; FINGERPRINT_WORDS];
        assert!((SemanticKernel::hamming_similarity(&a, &b) - 1.0).abs() < f32::EPSILON);

        let c = [0x00FF00FF_00FF00FFu64; FINGERPRINT_WORDS];
        let sim = SemanticKernel::hamming_similarity(&a, &c);
        assert!(sim < 0.01, "Inverted should be near 0: {}", sim);
    }

    #[test]
    fn test_resonate() {
        let kernel = SemanticKernel::new();
        let mut space = BindSpace::new();

        let target_fp = [0xFF00FF00_FF00FF00u64; FINGERPRINT_WORDS];
        let addr = Addr::new(0x80, 0x01);
        kernel.bind(&mut space, addr, target_fp, Some("target"));

        let results = kernel.resonate(
            &space,
            &target_fp,
            Some(KernelZone::Node { prefix: 0x80 }),
            0.4,
            10,
        );
        assert!(!results.is_empty());
        assert_eq!(results[0].0, addr);
        assert!((results[0].1 - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_nars_truth_deduction() {
        let kernel = SemanticKernel::new();
        let t1 = KernelTruth::new(0.9, 0.9);
        let t2 = KernelTruth::new(0.8, 0.8);
        let result = kernel.deduce(t1, t2);
        assert!(result.f > 0.0 && result.f <= 1.0);
        assert!(result.c > 0.0 && result.c <= 1.0);
    }

    #[test]
    fn test_nars_truth_revision() {
        let kernel = SemanticKernel::new();
        let existing = KernelTruth::new(0.7, 0.5);
        let evidence = KernelTruth::new(0.9, 0.8);
        let revised = kernel.revise(existing, evidence);
        // Revision should increase confidence
        assert!(revised.c > existing.c);
        // Frequency should shift toward evidence
        assert!(revised.f > existing.f);
    }

    #[test]
    fn test_causal_rung_ordering() {
        assert!(CausalRung::See < CausalRung::Do);
        assert!(CausalRung::Do < CausalRung::Imagine);
        assert_eq!(CausalRung::See.escalate(), Some(CausalRung::Do));
        assert_eq!(CausalRung::Do.escalate(), Some(CausalRung::Imagine));
        assert_eq!(CausalRung::Imagine.escalate(), None);
    }

    #[test]
    fn test_crystallize_dissolve() {
        let kernel = SemanticKernel::new();
        let mut space = BindSpace::new();

        let fluid = Addr::new(0x10, 0x01);
        let node = Addr::new(0x80, 0x01);
        let fp = [0xABCDu64; FINGERPRINT_WORDS];

        kernel.bind(&mut space, fluid, fp, Some("working"));
        assert!(kernel.crystallize(&mut space, fluid, node));
        assert_eq!(kernel.query(&space, node), Some(fp));

        // Dissolve back
        let fluid2 = Addr::new(0x10, 0x02);
        assert!(kernel.dissolve(&mut space, node, fluid2));
        assert_eq!(kernel.query(&space, fluid2), Some(fp));
    }

    #[test]
    fn test_kernel_describe() {
        let kernel = SemanticKernel::new();
        let desc = kernel.describe();
        assert!(desc.address_model.contains("8+8"));
        assert_eq!(desc.zones.len(), 3);
        assert!(!desc.operations.is_empty());
        assert!(!desc.datafusion_tables.is_empty());
    }

    #[test]
    fn test_crystal_and_rung_plugin_registration() {
        let mut reg = ExpansionRegistry::new();
        reg.register_crystal(CrystalPlugin {
            name: "context_5x5x5".into(),
            dimensions: 3,
            cells_per_dim: 5,
            total_cells: 125,
            fluid_prefix_start: 0x10,
            prefix_count: 5,
            description: "ContextCrystal 5x5x5".into(),
        })
        .unwrap();

        reg.register_rung_strategy(RungEscalationStrategy {
            name: "conservative".into(),
            see_confidence_floor: 0.9,
            do_confidence_floor: 0.8,
            auto_escalate_on_confounders: true,
            description: "Only accept high-confidence answers at each rung".into(),
        })
        .unwrap();

        assert_eq!(reg.crystals().len(), 1);
        assert_eq!(reg.rung_strategies().len(), 1);
    }

    #[test]
    fn test_protocol_and_strategy_registration() {
        let mut reg = ExpansionRegistry::new();
        reg.register_protocol(ProtocolExtension {
            name: "heartbeat".into(),
            message_kind: "Heartbeat".into(),
            description: "Periodic health check".into(),
        })
        .unwrap();
        reg.register_strategy(CollapseStrategy {
            name: "strict".into(),
            flow_threshold: Some(0.05),
            block_threshold: Some(0.20),
            description: "Strict collapse".into(),
        })
        .unwrap();
        assert_eq!(reg.protocols().len(), 1);
        assert_eq!(reg.strategies().len(), 1);
    }

    #[test]
    fn test_zone_helpers() {
        assert!(KernelZone::Surface { prefix: 0 }.is_surface());
        assert!(!KernelZone::Surface { prefix: 0 }.is_fluid());
        assert!(KernelZone::Fluid { prefix: 0x10 }.is_fluid());
        assert!(KernelZone::Node { prefix: 0x80 }.is_node());
    }

    #[test]
    fn test_datafusion_mappings() {
        let mappings = datafusion_table_mappings();
        assert!(mappings.len() >= 6);
        assert!(mappings.iter().any(|m| m.table_name == "kernel_nodes"));
        assert!(mappings.iter().any(|m| m.table_name == "kernel_causal"));
        assert!(mappings.iter().any(|m| m.table_name == "kernel_nars"));
    }
}
