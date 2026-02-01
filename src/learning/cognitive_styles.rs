//! Cognitive Styles - Scientifically grounded reasoning modes with RL adaptation
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                    FIXED / LEARNED / DISCOVERED TRIANGLE                    │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │                                                                             │
//! │                           DISCOVERED (α)                                    │
//! │                              ╱    ╲                                         │
//! │                             ╱      ╲                                        │
//! │           Novel styles    ╱        ╲   Mutation + recombination            │
//! │           from search    ╱          ╲  of successful patterns              │
//! │                         ╱            ╲                                      │
//! │                        ╱   ACTIVE     ╲                                     │
//! │                       ╱    STYLE       ╲                                    │
//! │                      ╱   w = αD+βL+γF   ╲                                   │
//! │                     ╱                    ╲                                  │
//! │                    ╱                      ╲                                 │
//! │            FIXED (γ) ──────────────────── LEARNED (β)                       │
//! │                                                                             │
//! │   Immutable base styles              RL-adapted weights per context         │
//! │   (15 core modes)                    (TD-learning on task success)          │
//! │                                                                             │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Self-Modification Loop
//!
//! 1. **Select**: Choose style via ε-greedy on Q-values
//! 2. **Execute**: Apply operator/atom biases to reasoning
//! 3. **Observe**: Measure task outcome (success, novelty, efficiency)
//! 4. **Update**: TD-learning on style Q-values
//! 5. **Evolve**: Periodically discover new styles via mutation

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

// ============================================================================
// Core Types
// ============================================================================

/// 9 cognitive operators - what processing to apply
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Operator {
    /// Selective attention allocation
    Attend = 0,
    /// Similarity/pattern recognition
    Match = 1,
    /// Hypothesis generation
    Infer = 2,
    /// Decision/consolidation
    Commit = 3,
    /// Mental simulation/projection
    Project = 4,
    /// Analysis into parts
    Decompose = 5,
    /// Relevance filtering
    Gate = 6,
    /// Irrelevance removal
    Prune = 7,
    /// Synthesis across sources
    Integrate = 8,
}

impl Operator {
    pub const ALL: [Operator; 9] = [
        Self::Attend, Self::Match, Self::Infer, Self::Commit,
        Self::Project, Self::Decompose, Self::Gate, Self::Prune, Self::Integrate,
    ];

    pub fn from_index(i: usize) -> Option<Self> {
        Self::ALL.get(i).copied()
    }
}

/// 9 reasoning atoms - what inference type to use
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Atom {
    /// Syllogistic: A→B, B→C, ∴ A→C
    Deduction = 0,
    /// Generalization from instances
    Induction = 1,
    /// Inference to best explanation
    Abduction = 2,
    /// Logical consequence
    Entailment = 3,
    /// Spreading activation
    Association = 4,
    /// Structural mapping (Gentner)
    Analogy = 5,
    /// Fast pattern match (System 1)
    Recognition = 6,
    /// Cost/benefit, utility
    Evaluation = 7,
    /// Alternative world reasoning
    Counterfactual = 8,
}

impl Atom {
    pub const ALL: [Atom; 9] = [
        Self::Deduction, Self::Induction, Self::Abduction, Self::Entailment,
        Self::Association, Self::Analogy, Self::Recognition, Self::Evaluation,
        Self::Counterfactual,
    ];

    pub fn from_index(i: usize) -> Option<Self> {
        Self::ALL.get(i).copied()
    }
}

/// Style fingerprint: 20 dimensions packed into [i8; 20]
/// [0-8]: operator biases (-128 to 127, scaled from -1.0 to 1.0)
/// [9-17]: atom biases
/// [18]: confidence_threshold (0-255 → 0.0-1.0)
/// [19]: exploration_rate (0-255 → 0.0-1.0)
pub type StyleFingerprint = [i8; 20];

/// A cognitive style definition
#[derive(Debug, Clone)]
pub struct CognitiveStyle {
    /// Unique identifier
    pub id: u16,
    /// Human-readable name
    pub name: String,
    /// When to use this style
    pub use_when: Vec<String>,
    /// Operator biases (-1.0 to 1.0)
    pub operator_bias: [f32; 9],
    /// Atom biases (-1.0 to 1.0)
    pub atom_bias: [f32; 9],
    /// Confidence required before committing (0.0 to 1.0)
    pub confidence_threshold: f32,
    /// Exploration vs exploitation (0.0 to 1.0)
    pub exploration_rate: f32,
    /// Origin: Fixed, Learned, or Discovered
    pub origin: StyleOrigin,
    /// Generation (0 = fixed, 1+ = discovered)
    pub generation: u32,
    /// Parent style IDs (for discovered styles)
    pub parents: Vec<u16>,
}

/// Where a style came from
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StyleOrigin {
    /// Immutable base style
    Fixed,
    /// Adapted through RL
    Learned,
    /// Synthesized through mutation/recombination
    Discovered,
}

impl CognitiveStyle {
    /// Convert to compact fingerprint
    pub fn to_fingerprint(&self) -> StyleFingerprint {
        let mut fp = [0i8; 20];

        // Operator biases (scale -1.0..1.0 to -127..127)
        for (i, &bias) in self.operator_bias.iter().enumerate() {
            fp[i] = (bias.clamp(-1.0, 1.0) * 127.0) as i8;
        }

        // Atom biases
        for (i, &bias) in self.atom_bias.iter().enumerate() {
            fp[9 + i] = (bias.clamp(-1.0, 1.0) * 127.0) as i8;
        }

        // Thresholds (scale 0.0..1.0 to 0..127, stored in i8)
        fp[18] = (self.confidence_threshold.clamp(0.0, 1.0) * 127.0) as i8;
        fp[19] = (self.exploration_rate.clamp(0.0, 1.0) * 127.0) as i8;

        fp
    }

    /// Create from fingerprint
    pub fn from_fingerprint(id: u16, name: String, fp: &StyleFingerprint, origin: StyleOrigin) -> Self {
        let mut operator_bias = [0.0f32; 9];
        let mut atom_bias = [0.0f32; 9];

        for i in 0..9 {
            operator_bias[i] = fp[i] as f32 / 127.0;
            atom_bias[i] = fp[9 + i] as f32 / 127.0;
        }

        Self {
            id,
            name,
            use_when: Vec::new(),
            operator_bias,
            atom_bias,
            confidence_threshold: fp[18] as f32 / 127.0,
            exploration_rate: fp[19] as f32 / 127.0,
            origin,
            generation: if origin == StyleOrigin::Fixed { 0 } else { 1 },
            parents: Vec::new(),
        }
    }

    /// Hamming distance to another style
    pub fn distance(&self, other: &Self) -> u32 {
        let fp1 = self.to_fingerprint();
        let fp2 = other.to_fingerprint();

        fp1.iter()
            .zip(fp2.iter())
            .map(|(a, b)| ((*a as i16 - *b as i16).abs() as u32))
            .sum()
    }

    /// Mutate style (for discovery)
    pub fn mutate(&self, mutation_rate: f32, rng: &mut impl FnMut() -> f32) -> Self {
        let mut child = self.clone();
        child.origin = StyleOrigin::Discovered;
        child.generation = self.generation + 1;
        child.parents = vec![self.id];

        for bias in &mut child.operator_bias {
            if rng() < mutation_rate {
                *bias = (*bias + (rng() - 0.5) * 0.2).clamp(-1.0, 1.0);
            }
        }

        for bias in &mut child.atom_bias {
            if rng() < mutation_rate {
                *bias = (*bias + (rng() - 0.5) * 0.2).clamp(-1.0, 1.0);
            }
        }

        if rng() < mutation_rate {
            child.confidence_threshold = (child.confidence_threshold + (rng() - 0.5) * 0.1).clamp(0.5, 0.99);
        }

        if rng() < mutation_rate {
            child.exploration_rate = (child.exploration_rate + (rng() - 0.5) * 0.1).clamp(0.0, 0.5);
        }

        child
    }

    /// Crossover two styles (for discovery)
    pub fn crossover(&self, other: &Self, rng: &mut impl FnMut() -> f32) -> Self {
        let mut child = self.clone();
        child.origin = StyleOrigin::Discovered;
        child.generation = self.generation.max(other.generation) + 1;
        child.parents = vec![self.id, other.id];
        child.name = format!("{}+{}", self.name.split_whitespace().next().unwrap_or("X"),
                             other.name.split_whitespace().next().unwrap_or("Y"));

        // Uniform crossover
        for i in 0..9 {
            if rng() > 0.5 {
                child.operator_bias[i] = other.operator_bias[i];
            }
            if rng() > 0.5 {
                child.atom_bias[i] = other.atom_bias[i];
            }
        }

        if rng() > 0.5 {
            child.confidence_threshold = other.confidence_threshold;
        }
        if rng() > 0.5 {
            child.exploration_rate = other.exploration_rate;
        }

        child
    }
}

// ============================================================================
// Fixed Base Styles (15 core modes)
// ============================================================================

/// Create the 15 fixed base styles
pub fn create_base_styles() -> Vec<CognitiveStyle> {
    vec![
        // === ANALYSIS MODES ===
        CognitiveStyle {
            id: 1,
            name: "First Principles Decomposition".into(),
            use_when: vec!["debugging".into(), "questioning assumptions".into(), "novel domains".into()],
            operator_bias: [0.15, 0.0, 0.25, 0.15, 0.0, 0.25, 0.0, 0.0, 0.0],
            atom_bias: [0.2, 0.0, 0.15, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            confidence_threshold: 0.8,
            exploration_rate: 0.1,
            origin: StyleOrigin::Fixed,
            generation: 0,
            parents: vec![],
        },
        CognitiveStyle {
            id: 2,
            name: "Constraint Propagation".into(),
            use_when: vec!["planning".into(), "satisfiability".into(), "configuration".into()],
            operator_bias: [0.1, 0.0, 0.2, 0.15, 0.0, 0.0, 0.15, 0.2, 0.0],
            atom_bias: [0.15, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 0.1, 0.0],
            confidence_threshold: 0.85,
            exploration_rate: 0.05,
            origin: StyleOrigin::Fixed,
            generation: 0,
            parents: vec![],
        },
        CognitiveStyle {
            id: 3,
            name: "Causal Inference".into(),
            use_when: vec!["root cause".into(), "intervention effects".into(), "mechanisms".into()],
            operator_bias: [0.1, 0.0, 0.25, 0.0, 0.2, 0.1, 0.0, 0.0, 0.0],
            atom_bias: [0.0, 0.0, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.2],
            confidence_threshold: 0.85,
            exploration_rate: 0.15,
            origin: StyleOrigin::Fixed,
            generation: 0,
            parents: vec![],
        },
        CognitiveStyle {
            id: 4,
            name: "Counterfactual Simulation".into(),
            use_when: vec!["what-if".into(), "risk assessment".into(), "regret minimization".into()],
            operator_bias: [0.1, 0.0, 0.15, 0.0, 0.35, 0.0, 0.1, 0.0, 0.0],
            atom_bias: [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.25],
            confidence_threshold: 0.9,
            exploration_rate: 0.3,
            origin: StyleOrigin::Fixed,
            generation: 0,
            parents: vec![],
        },

        // === TRANSFER MODES ===
        CognitiveStyle {
            id: 5,
            name: "Analogical Transfer".into(),
            use_when: vec!["novel domains".into(), "teaching".into(), "cross-domain".into()],
            operator_bias: [0.1, 0.25, 0.2, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0],
            atom_bias: [0.0, 0.0, 0.1, 0.0, 0.1, 0.3, 0.0, 0.0, 0.0],
            confidence_threshold: 0.85,
            exploration_rate: 0.25,
            origin: StyleOrigin::Fixed,
            generation: 0,
            parents: vec![],
        },
        CognitiveStyle {
            id: 6,
            name: "Cross-Domain Synthesis".into(),
            use_when: vec!["creative problem solving".into(), "innovation".into(), "non-obvious connections".into()],
            operator_bias: [0.0, 0.15, 0.2, 0.0, 0.2, 0.0, 0.0, 0.0, 0.15],
            atom_bias: [0.0, 0.0, 0.15, 0.0, 0.15, 0.15, 0.0, 0.0, 0.0],
            confidence_threshold: 0.9,
            exploration_rate: 0.4,
            origin: StyleOrigin::Fixed,
            generation: 0,
            parents: vec![],
        },

        // === UNCERTAINTY MODES ===
        CognitiveStyle {
            id: 7,
            name: "Bayesian Calibration".into(),
            use_when: vec!["high stakes".into(), "limited data".into(), "uncertainty communication".into()],
            operator_bias: [0.15, 0.0, 0.2, 0.0, 0.0, 0.0, 0.15, 0.1, 0.0],
            atom_bias: [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, -0.1, 0.15, 0.0],
            confidence_threshold: 0.95,
            exploration_rate: 0.1,
            origin: StyleOrigin::Fixed,
            generation: 0,
            parents: vec![],
        },
        CognitiveStyle {
            id: 8,
            name: "Anomaly-Driven Exploration".into(),
            use_when: vec!["model gaps".into(), "unexpected observations".into(), "active learning".into()],
            operator_bias: [0.15, 0.1, 0.0, 0.0, 0.25, 0.15, 0.0, 0.0, 0.0],
            atom_bias: [0.0, 0.1, 0.15, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
            confidence_threshold: 0.95,
            exploration_rate: 0.5,
            origin: StyleOrigin::Fixed,
            generation: 0,
            parents: vec![],
        },

        // === STRUCTURE MODES ===
        CognitiveStyle {
            id: 9,
            name: "Hierarchical Abstraction".into(),
            use_when: vec!["complex systems".into(), "right level of detail".into(), "API design".into()],
            operator_bias: [0.15, 0.0, 0.2, 0.0, 0.0, 0.15, 0.0, 0.0, 0.15],
            atom_bias: [0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0],
            confidence_threshold: 0.85,
            exploration_rate: 0.1,
            origin: StyleOrigin::Fixed,
            generation: 0,
            parents: vec![],
        },
        CognitiveStyle {
            id: 10,
            name: "Graph Structure Detection".into(),
            use_when: vec!["dependency analysis".into(), "network analysis".into(), "knowledge graphs".into()],
            operator_bias: [0.15, 0.15, 0.25, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0],
            atom_bias: [0.15, 0.0, 0.0, 0.15, 0.1, 0.0, 0.0, 0.0, 0.0],
            confidence_threshold: 0.8,
            exploration_rate: 0.1,
            origin: StyleOrigin::Fixed,
            generation: 0,
            parents: vec![],
        },
        CognitiveStyle {
            id: 11,
            name: "Temporal Sequence Modeling".into(),
            use_when: vec!["time series".into(), "narrative".into(), "process modeling".into()],
            operator_bias: [0.1, 0.15, 0.2, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0],
            atom_bias: [0.0, 0.15, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0],
            confidence_threshold: 0.85,
            exploration_rate: 0.15,
            origin: StyleOrigin::Fixed,
            generation: 0,
            parents: vec![],
        },

        // === DECISION MODES ===
        CognitiveStyle {
            id: 12,
            name: "Utility Maximization".into(),
            use_when: vec!["resource allocation".into(), "trade-offs".into(), "optimization".into()],
            operator_bias: [0.1, 0.0, 0.15, 0.25, 0.2, 0.0, 0.0, 0.0, 0.0],
            atom_bias: [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.1],
            confidence_threshold: 0.8,
            exploration_rate: 0.1,
            origin: StyleOrigin::Fixed,
            generation: 0,
            parents: vec![],
        },
        CognitiveStyle {
            id: 13,
            name: "Constraint Relaxation".into(),
            use_when: vec!["over-constrained".into(), "negotiation".into(), "MVP definition".into()],
            operator_bias: [0.1, 0.0, 0.0, 0.1, 0.0, 0.2, 0.15, 0.15, 0.0],
            atom_bias: [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.2, 0.1],
            confidence_threshold: 0.85,
            exploration_rate: 0.2,
            origin: StyleOrigin::Fixed,
            generation: 0,
            parents: vec![],
        },

        // === SOCIAL/COLLABORATIVE MODES ===
        CognitiveStyle {
            id: 14,
            name: "Perspective Simulation".into(),
            use_when: vec!["collaboration".into(), "explaining".into(), "anticipating objections".into()],
            operator_bias: [0.15, 0.2, 0.1, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0],
            atom_bias: [0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.0, 0.1, 0.1],
            confidence_threshold: 0.85,
            exploration_rate: 0.15,
            origin: StyleOrigin::Fixed,
            generation: 0,
            parents: vec![],
        },
        CognitiveStyle {
            id: 15,
            name: "Adversarial Stress Testing".into(),
            use_when: vec!["code review".into(), "argument verification".into(), "security analysis".into()],
            operator_bias: [0.1, 0.0, 0.2, 0.0, 0.2, 0.15, 0.0, 0.0, 0.0],
            atom_bias: [0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0, 0.1, 0.15],
            confidence_threshold: 0.9,
            exploration_rate: 0.3,
            origin: StyleOrigin::Fixed,
            generation: 0,
            parents: vec![],
        },
    ]
}

// ============================================================================
// RL-Based Style Selection
// ============================================================================

/// Q-value entry for a (context, style) pair
#[derive(Debug, Clone)]
struct QEntry {
    /// Expected value
    value: f32,
    /// Update count (for learning rate decay)
    count: u32,
    /// Last update timestamp
    last_update: u64,
}

impl Default for QEntry {
    fn default() -> Self {
        Self {
            value: 0.0,
            count: 0,
            last_update: 0,
        }
    }
}

/// Context features for style selection
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct TaskContext {
    /// Task type hash (debugging, planning, creative, etc.)
    pub task_type: u32,
    /// Uncertainty level (0-255)
    pub uncertainty: u8,
    /// Time pressure (0-255)
    pub time_pressure: u8,
    /// Novelty of domain (0-255)
    pub novelty: u8,
    /// Collaboration mode (0 = solo, 255 = team)
    pub collaboration: u8,
}

impl TaskContext {
    /// Create a context key for Q-table lookup
    pub fn key(&self) -> u64 {
        ((self.task_type as u64) << 32)
            | ((self.uncertainty as u64) << 24)
            | ((self.time_pressure as u64) << 16)
            | ((self.novelty as u64) << 8)
            | (self.collaboration as u64)
    }

    /// Discretize continuous values into buckets
    pub fn discretize(uncertainty: f32, time_pressure: f32, novelty: f32, collaboration: f32) -> Self {
        Self {
            task_type: 0,
            uncertainty: (uncertainty.clamp(0.0, 1.0) * 3.0) as u8, // 4 buckets
            time_pressure: (time_pressure.clamp(0.0, 1.0) * 3.0) as u8,
            novelty: (novelty.clamp(0.0, 1.0) * 3.0) as u8,
            collaboration: (collaboration.clamp(0.0, 1.0) * 3.0) as u8,
        }
    }
}

/// Outcome of applying a style to a task
#[derive(Debug, Clone)]
pub struct TaskOutcome {
    /// Did the task succeed? (0.0 to 1.0)
    pub success: f32,
    /// How novel was the solution? (0.0 to 1.0)
    pub novelty: f32,
    /// How efficient was the process? (0.0 to 1.0)
    pub efficiency: f32,
    /// Any errors or issues?
    pub errors: u32,
}

impl TaskOutcome {
    /// Compute reward signal
    pub fn reward(&self) -> f32 {
        // Weighted combination
        0.5 * self.success
            + 0.2 * self.novelty
            + 0.2 * self.efficiency
            - 0.1 * (self.errors as f32).min(1.0)
    }
}

/// RL configuration
#[derive(Debug, Clone)]
pub struct RLConfig {
    /// Learning rate (α)
    pub learning_rate: f32,
    /// Discount factor (γ)
    pub discount: f32,
    /// Exploration rate (ε)
    pub epsilon: f32,
    /// Epsilon decay per update
    pub epsilon_decay: f32,
    /// Minimum epsilon
    pub epsilon_min: f32,
    /// How much to weight fixed styles vs learned
    pub fixed_weight: f32,
    /// How much to weight discovered styles
    pub discovery_weight: f32,
    /// Mutation rate for discovery
    pub mutation_rate: f32,
    /// Discovery interval (every N selections)
    pub discovery_interval: u32,
    /// Maximum discovered styles to keep
    pub max_discovered: usize,
}

impl Default for RLConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            discount: 0.95,
            epsilon: 0.3,
            epsilon_decay: 0.999,
            epsilon_min: 0.05,
            fixed_weight: 0.5,      // γ in triangle
            discovery_weight: 0.2,  // α in triangle
            // β (learned) = 1.0 - γ - α = 0.3
            mutation_rate: 0.15,
            discovery_interval: 50,
            max_discovered: 20,
        }
    }
}

// ============================================================================
// Style Selector (Main Interface)
// ============================================================================

/// Adaptive style selector with RL
pub struct StyleSelector {
    /// Configuration
    config: RLConfig,
    /// Fixed base styles
    fixed_styles: Vec<CognitiveStyle>,
    /// Learned Q-values: context_key -> style_id -> QEntry
    q_table: RwLock<HashMap<u64, HashMap<u16, QEntry>>>,
    /// Discovered styles
    discovered_styles: RwLock<Vec<CognitiveStyle>>,
    /// Selection count (for discovery interval)
    selection_count: AtomicU64,
    /// Current epsilon
    epsilon: RwLock<f32>,
    /// Performance history for discovered styles
    discovered_performance: RwLock<HashMap<u16, f32>>,
    /// Next discovered style ID
    next_discovered_id: AtomicU64,
    /// Random state (simple LCG)
    rng_state: AtomicU64,
}

impl StyleSelector {
    /// Create a new style selector
    pub fn new(config: RLConfig) -> Self {
        Self {
            fixed_styles: create_base_styles(),
            q_table: RwLock::new(HashMap::new()),
            discovered_styles: RwLock::new(Vec::new()),
            selection_count: AtomicU64::new(0),
            epsilon: RwLock::new(config.epsilon),
            discovered_performance: RwLock::new(HashMap::new()),
            next_discovered_id: AtomicU64::new(1000), // Discovered start at 1000
            rng_state: AtomicU64::new(0x12345678),
            config,
        }
    }

    /// Simple RNG
    fn rand(&self) -> f32 {
        let state = self.rng_state.fetch_add(1, Ordering::Relaxed);
        let x = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.rng_state.store(x, Ordering::Relaxed);
        (x >> 33) as f32 / (1u64 << 31) as f32
    }

    /// Select a style for the given context (ε-greedy)
    pub fn select(&self, context: &TaskContext) -> CognitiveStyle {
        let count = self.selection_count.fetch_add(1, Ordering::Relaxed);

        // Periodic discovery
        if count > 0 && count % self.config.discovery_interval as u64 == 0 {
            self.discover_new_style();
        }

        let epsilon = *self.epsilon.read().unwrap();

        // ε-greedy selection
        if self.rand() < epsilon {
            // Explore: random style
            self.random_style()
        } else {
            // Exploit: best Q-value with triangle weighting
            self.best_style(context)
        }
    }

    /// Get a random style
    fn random_style(&self) -> CognitiveStyle {
        let discovered = self.discovered_styles.read().unwrap();
        let total = self.fixed_styles.len() + discovered.len();

        if total == 0 {
            return self.fixed_styles[0].clone();
        }

        let idx = (self.rand() * total as f32) as usize % total;

        if idx < self.fixed_styles.len() {
            self.fixed_styles[idx].clone()
        } else {
            discovered[idx - self.fixed_styles.len()].clone()
        }
    }

    /// Get best style according to Q-values with triangle weighting
    fn best_style(&self, context: &TaskContext) -> CognitiveStyle {
        let q_table = self.q_table.read().unwrap();
        let discovered = self.discovered_styles.read().unwrap();

        let context_q = q_table.get(&context.key());

        let mut best_style = None;
        let mut best_score = f32::NEG_INFINITY;

        // Score fixed styles
        for style in &self.fixed_styles {
            let q_value = context_q
                .and_then(|m| m.get(&style.id))
                .map(|e| e.value)
                .unwrap_or(0.0);

            // Fixed styles get γ weight
            let score = q_value * self.config.fixed_weight;

            if score > best_score {
                best_score = score;
                best_style = Some(style.clone());
            }
        }

        // Score discovered styles
        let perf = self.discovered_performance.read().unwrap();
        for style in discovered.iter() {
            let q_value = context_q
                .and_then(|m| m.get(&style.id))
                .map(|e| e.value)
                .unwrap_or(0.0);

            // Discovered styles get α weight + performance bonus
            let perf_bonus = perf.get(&style.id).copied().unwrap_or(0.0);
            let score = q_value * self.config.discovery_weight + perf_bonus * 0.1;

            if score > best_score {
                best_score = score;
                best_style = Some(style.clone());
            }
        }

        best_style.unwrap_or_else(|| self.fixed_styles[0].clone())
    }

    /// Update Q-value after task completion (TD-learning)
    pub fn update(&self, context: &TaskContext, style_id: u16, outcome: &TaskOutcome) {
        let reward = outcome.reward();

        let mut q_table = self.q_table.write().unwrap();
        let context_map = q_table.entry(context.key()).or_insert_with(HashMap::new);
        let entry = context_map.entry(style_id).or_default();

        // TD update: Q(s,a) ← Q(s,a) + α * (r - Q(s,a))
        // Simplified: no next state since this is contextual bandit
        let alpha = self.config.learning_rate / (1.0 + entry.count as f32 * 0.01);
        entry.value += alpha * (reward - entry.value);
        entry.count += 1;
        entry.last_update = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Decay epsilon
        {
            let mut eps = self.epsilon.write().unwrap();
            *eps = (*eps * self.config.epsilon_decay).max(self.config.epsilon_min);
        }

        // Update discovered style performance
        if style_id >= 1000 {
            let mut perf = self.discovered_performance.write().unwrap();
            let current = perf.entry(style_id).or_insert(0.0);
            *current = *current * 0.9 + reward * 0.1; // Exponential moving average
        }
    }

    /// Discover a new style through mutation/crossover
    fn discover_new_style(&self) {
        let mut discovered = self.discovered_styles.write().unwrap();

        // Prune if too many
        if discovered.len() >= self.config.max_discovered {
            let perf = self.discovered_performance.read().unwrap();
            // Remove worst performer
            if let Some((worst_idx, _)) = discovered.iter().enumerate()
                .min_by(|(_, a), (_, b)| {
                    let pa = perf.get(&a.id).copied().unwrap_or(0.0);
                    let pb = perf.get(&b.id).copied().unwrap_or(0.0);
                    pa.partial_cmp(&pb).unwrap_or(std::cmp::Ordering::Equal)
                })
            {
                discovered.remove(worst_idx);
            }
        }

        // Generate new style
        let new_id = self.next_discovered_id.fetch_add(1, Ordering::SeqCst) as u16;

        let new_style = if self.rand() < 0.5 && discovered.len() >= 2 {
            // Crossover two discovered or fixed styles
            let all_styles: Vec<_> = self.fixed_styles.iter()
                .chain(discovered.iter())
                .collect();

            let idx1 = (self.rand() * all_styles.len() as f32) as usize % all_styles.len();
            let idx2 = (self.rand() * all_styles.len() as f32) as usize % all_styles.len();

            let mut rng = || self.rand();
            let mut child = all_styles[idx1].crossover(all_styles[idx2], &mut rng);
            child.id = new_id;
            child
        } else {
            // Mutate a random style
            let base_idx = (self.rand() * self.fixed_styles.len() as f32) as usize % self.fixed_styles.len();
            let mut rng = || self.rand();
            let mut child = self.fixed_styles[base_idx].mutate(self.config.mutation_rate, &mut rng);
            child.id = new_id;
            child.name = format!("Discovered-{}", new_id);
            child
        };

        discovered.push(new_style);
    }

    /// Get all styles (fixed + discovered)
    pub fn all_styles(&self) -> Vec<CognitiveStyle> {
        let discovered = self.discovered_styles.read().unwrap();
        self.fixed_styles.iter()
            .chain(discovered.iter())
            .cloned()
            .collect()
    }

    /// Get style by ID
    pub fn get_style(&self, id: u16) -> Option<CognitiveStyle> {
        if id < 1000 {
            self.fixed_styles.iter().find(|s| s.id == id).cloned()
        } else {
            self.discovered_styles.read().unwrap()
                .iter()
                .find(|s| s.id == id)
                .cloned()
        }
    }

    /// Get current triangle weights
    pub fn triangle_weights(&self) -> (f32, f32, f32) {
        let fixed = self.config.fixed_weight;
        let discovered = self.config.discovery_weight;
        let learned = 1.0 - fixed - discovered;
        (fixed, learned, discovered)
    }

    /// Shift triangle weights (for meta-learning)
    pub fn shift_weights(&mut self, fixed: f32, learned: f32, discovered: f32) {
        let total = fixed + learned + discovered;
        self.config.fixed_weight = fixed / total;
        self.config.discovery_weight = discovered / total;
        // learned is implicit
    }

    /// Export learned Q-values
    pub fn export_q_table(&self) -> HashMap<u64, HashMap<u16, f32>> {
        let q_table = self.q_table.read().unwrap();
        q_table.iter()
            .map(|(k, v)| (*k, v.iter().map(|(id, e)| (*id, e.value)).collect()))
            .collect()
    }

    /// Get statistics
    pub fn stats(&self) -> StyleSelectorStats {
        let q_table = self.q_table.read().unwrap();
        let discovered = self.discovered_styles.read().unwrap();

        StyleSelectorStats {
            fixed_count: self.fixed_styles.len(),
            discovered_count: discovered.len(),
            contexts_seen: q_table.len(),
            total_selections: self.selection_count.load(Ordering::Relaxed),
            current_epsilon: *self.epsilon.read().unwrap(),
            triangle_weights: self.triangle_weights(),
        }
    }
}

/// Statistics about the selector
#[derive(Debug, Clone)]
pub struct StyleSelectorStats {
    pub fixed_count: usize,
    pub discovered_count: usize,
    pub contexts_seen: usize,
    pub total_selections: u64,
    pub current_epsilon: f32,
    pub triangle_weights: (f32, f32, f32),
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_styles() {
        let styles = create_base_styles();
        assert_eq!(styles.len(), 15);

        for style in &styles {
            assert_eq!(style.origin, StyleOrigin::Fixed);
            assert_eq!(style.generation, 0);
            assert!(style.confidence_threshold >= 0.0 && style.confidence_threshold <= 1.0);
            assert!(style.exploration_rate >= 0.0 && style.exploration_rate <= 1.0);
        }
    }

    #[test]
    fn test_fingerprint_roundtrip() {
        let styles = create_base_styles();

        for style in &styles {
            let fp = style.to_fingerprint();
            let recovered = CognitiveStyle::from_fingerprint(
                style.id,
                style.name.clone(),
                &fp,
                StyleOrigin::Fixed,
            );

            // Should be close (some precision loss from i8)
            for i in 0..9 {
                assert!((style.operator_bias[i] - recovered.operator_bias[i]).abs() < 0.01);
                assert!((style.atom_bias[i] - recovered.atom_bias[i]).abs() < 0.01);
            }
        }
    }

    #[test]
    fn test_mutation() {
        let style = create_base_styles()[0].clone();

        // Use RNG that produces larger values to ensure visible changes
        let mut counter = 0u32;
        let mut rng = || {
            counter += 1;
            // Alternate between 0.0 and 1.0 to ensure mutations happen
            if counter % 2 == 0 { 0.9 } else { 0.1 }
        };

        let mutated = style.mutate(1.0, &mut rng); // 100% mutation rate

        assert_eq!(mutated.origin, StyleOrigin::Discovered);
        assert_eq!(mutated.generation, 1);
        assert_eq!(mutated.parents, vec![style.id]);

        // Mutation should change fingerprint (larger RNG swing ensures this)
        let fp1 = style.to_fingerprint();
        let fp2 = mutated.to_fingerprint();
        assert_ne!(fp1, fp2, "Mutation should change the fingerprint");
    }

    #[test]
    fn test_crossover() {
        let styles = create_base_styles();
        let parent1 = &styles[0];
        let parent2 = &styles[5];

        let mut rng_state = 123u64;
        let mut rng = || {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            (rng_state >> 16) as f32 / 65536.0
        };

        let child = parent1.crossover(parent2, &mut rng);

        assert_eq!(child.origin, StyleOrigin::Discovered);
        assert_eq!(child.parents.len(), 2);
        assert!(child.parents.contains(&parent1.id));
        assert!(child.parents.contains(&parent2.id));
    }

    #[test]
    fn test_selector_basic() {
        let selector = StyleSelector::new(RLConfig::default());

        let context = TaskContext {
            task_type: 1,
            uncertainty: 128,
            time_pressure: 64,
            novelty: 200,
            collaboration: 0,
        };

        // Should be able to select
        let style = selector.select(&context);
        assert!(!style.name.is_empty());

        // Update with outcome
        let outcome = TaskOutcome {
            success: 0.8,
            novelty: 0.3,
            efficiency: 0.7,
            errors: 0,
        };

        selector.update(&context, style.id, &outcome);

        // Stats should reflect the selection
        let stats = selector.stats();
        assert_eq!(stats.fixed_count, 15);
        assert!(stats.total_selections >= 1);
    }

    #[test]
    fn test_discovery() {
        let config = RLConfig {
            discovery_interval: 2, // Discover every 2 selections
            ..Default::default()
        };

        let selector = StyleSelector::new(config);

        let context = TaskContext::discretize(0.5, 0.5, 0.5, 0.0);

        // Select multiple times to trigger discovery
        for _ in 0..10 {
            let _ = selector.select(&context);
        }

        let stats = selector.stats();
        assert!(stats.discovered_count > 0);
    }

    #[test]
    fn test_triangle_weights() {
        let mut selector = StyleSelector::new(RLConfig::default());

        let (f, l, d) = selector.triangle_weights();
        assert!((f + l + d - 1.0).abs() < 0.001);

        // Shift weights
        selector.shift_weights(0.3, 0.5, 0.2);
        let (f2, l2, d2) = selector.triangle_weights();
        assert!((f2 + l2 + d2 - 1.0).abs() < 0.001);
        assert!((f2 - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_q_learning() {
        let selector = StyleSelector::new(RLConfig {
            epsilon: 0.0, // Pure exploitation for test
            ..Default::default()
        });

        let context = TaskContext::discretize(0.5, 0.5, 0.5, 0.0);

        // Update style 1 with good outcome
        let good_outcome = TaskOutcome {
            success: 1.0,
            novelty: 0.5,
            efficiency: 1.0,
            errors: 0,
        };
        selector.update(&context, 1, &good_outcome);
        selector.update(&context, 1, &good_outcome);
        selector.update(&context, 1, &good_outcome);

        // Update style 2 with bad outcome
        let bad_outcome = TaskOutcome {
            success: 0.0,
            novelty: 0.0,
            efficiency: 0.0,
            errors: 5,
        };
        selector.update(&context, 2, &bad_outcome);

        // Q-table should prefer style 1
        let q_table = selector.export_q_table();
        let context_q = q_table.get(&context.key()).unwrap();

        let q1 = context_q.get(&1).copied().unwrap_or(0.0);
        let q2 = context_q.get(&2).copied().unwrap_or(0.0);

        assert!(q1 > q2);
    }

    #[test]
    fn test_task_outcome_reward() {
        let perfect = TaskOutcome {
            success: 1.0,
            novelty: 1.0,
            efficiency: 1.0,
            errors: 0,
        };
        assert!((perfect.reward() - 0.9).abs() < 0.001);

        let failure = TaskOutcome {
            success: 0.0,
            novelty: 0.0,
            efficiency: 0.0,
            errors: 10,
        };
        assert!(failure.reward() < 0.0);
    }
}
