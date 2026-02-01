//! Learning module - Meta-AGI Learning Loop + CAM Operations + Cognitive Frameworks
//!
//! CAM = Content Addressable Methods
//! 4096 operations as a unified cognitive vocabulary.
//! Everything stays in fingerprint space - no context switching.
//!
//! Cognitive Frameworks:
//! - NARS: Non-Axiomatic Reasoning System
//! - ACT-R: Adaptive Control of Thought
//! - RL: Reinforcement Learning (with causal extensions)
//! - Causality: Pearl's do-calculus
//! - Qualia: Affect channels
//! - Rung: Abstraction ladder
//!
//! Quantum-Inspired:
//! - Operators as linear mappings on fingerprint space
//! - Non-commutative algebra
//! - Measurement as collapse to eigenstates
//!
//! Tree Addressing:
//! - 256-way branching for hierarchical navigation
//! - Like LDAP Distinguished Names
//!
//! NEW: Causal RL Integration
//! - rl_ops: Causal Q-learning with intervention/counterfactual reasoning
//! - causal_ops: Full do-calculus as fingerprint operations

pub mod moment;
pub mod session;
pub mod blackboard;
pub mod resonance;
pub mod concept;
pub mod cam_ops;
pub mod cognitive_frameworks;
pub mod quantum_ops;
pub mod rl_ops;
pub mod causal_ops;
pub mod cognitive_styles;

pub use moment::{Moment, MomentType, Qualia, MomentBuilder};
pub use session::{LearningSession, SessionState, SessionPhase};
pub use blackboard::{Blackboard, Decision, IceCakedLayer};
pub use resonance::{ResonanceCapture, SimilarMoment, ResonanceStats, find_sweet_spot, mexican_hat_resonance};
pub use concept::{ConceptExtractor, ExtractedConcept, RelationType, ConceptRelation};
pub use cam_ops::{
    OpDictionary, OpResult, OpContext, OpCategory, OpSignature, OpType, OpMeta, OpParam,
    LanceOp, SqlOp, CypherOp, HammingOp, LearnOp,
    bundle_fingerprints, fold_to_48,
};
pub use cognitive_frameworks::{
    // NARS
    TruthValue, NarsCopula, NarsInference, NarsStatement,
    // ACT-R
    ActrBuffer, ActrChunk, ActrProduction,
    // RL (basic)
    StateAction, QValue, RlAgent,
    // Causality (basic)
    CausalRelation, CausalNode, CausalEdge, DoOperator, Counterfactual,
    // Qualia
    QualiaChannel, QualiaState,
    // Rung
    Rung, RungClassifier,
};
pub use quantum_ops::{
    // Tree addressing
    TreeAddr, tree_branches,
    // Quantum operator trait
    QuantumOp,
    // Core operators
    IdentityOp, NotOp, BindOp, PermuteOp, ProjectOp, HadamardOp, MeasureOp, TimeEvolutionOp,
    // Cognitive operators
    NarsInferenceOp, ActrRetrievalOp, RlValueOp, CausalDoOp, QualiaShiftOp, RungLadderOp,
    // Operator algebra
    ComposedOp, SumOp, TensorOp,
};

// NEW: Causal RL integration (wired to search module)
pub use rl_ops::{
    RlOp, CausalRlAgent,
    ActionExplanation, AlternativeAction, CausalChainLink,
};
pub use causal_ops::{
    CausalOp, CausalEngine, GraphEdge, CausalEdgeType,
};

// Cognitive styles with RL-based adaptation
pub use cognitive_styles::{
    // Core types
    Operator, Atom, StyleFingerprint, StyleOrigin,
    // Style definition
    CognitiveStyle, create_base_styles,
    // RL components
    TaskContext, TaskOutcome, RLConfig,
    // Selector (main interface)
    StyleSelector, StyleSelectorStats,
};
