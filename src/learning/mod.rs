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

pub mod blackboard;
pub mod cam_ops;
pub mod causal_ops;
pub mod cognitive_frameworks;
pub mod cognitive_styles;
pub mod concept;
pub mod moment;
pub mod quantum_ops;
pub mod resonance;
pub mod rl_ops;
pub mod scm;
pub mod session;

pub use blackboard::{Blackboard, Decision, IceCakedLayer};
pub use cam_ops::{
    CypherOp, HammingOp, LanceOp, LearnOp, OpCategory, OpContext, OpDictionary, OpMeta, OpParam,
    OpResult, OpSignature, OpType, SqlOp, bundle_fingerprints, fold_to_48,
};
pub use cognitive_frameworks::{
    // ACT-R
    ActrBuffer,
    ActrChunk,
    ActrProduction,
    CausalEdge,
    CausalNode,
    // Causality (basic)
    CausalRelation,
    Counterfactual,
    DoOperator,
    NarsCopula,
    NarsInference,
    NarsStatement,
    QValue,
    // Qualia
    QualiaChannel,
    QualiaState,
    RlAgent,
    // Rung
    Rung,
    RungClassifier,
    // RL (basic)
    StateAction,
    // NARS
    TruthValue,
};
pub use concept::{ConceptExtractor, ConceptRelation, ExtractedConcept, RelationType};
pub use moment::{Moment, MomentBuilder, MomentType, Qualia};
pub use quantum_ops::{
    ActrRetrievalOp,
    BindOp,
    CausalDoOp,
    // Operator algebra
    ComposedOp,
    HadamardOp,
    // Core operators
    IdentityOp,
    MeasureOp,
    // Cognitive operators
    NarsInferenceOp,
    NotOp,
    PermuteOp,
    ProjectOp,
    QualiaShiftOp,
    // Quantum operator trait
    QuantumOp,
    RlValueOp,
    RungLadderOp,
    SumOp,
    TensorOp,
    TimeEvolutionOp,
    // Tree addressing
    TreeAddr,
    tree_branches,
};
pub use resonance::{
    ResonanceCapture, ResonanceStats, SimilarMoment, find_sweet_spot, mexican_hat_resonance,
};
pub use session::{LearningSession, SessionPhase, SessionState};

// NEW: Causal RL integration (wired to search module)
pub use causal_ops::{CausalEdgeType, CausalEngine, CausalOp, GraphEdge};
pub use rl_ops::{ActionExplanation, AlternativeAction, CausalChainLink, CausalRlAgent, RlOp};

// Cognitive styles with RL-based adaptation
pub use cognitive_styles::{
    Atom,
    // Style definition
    CognitiveStyle,
    // Core types
    Operator,
    RLConfig,
    StyleFingerprint,
    StyleOrigin,
    // Selector (main interface)
    StyleSelector,
    StyleSelectorStats,
    // RL components
    TaskContext,
    TaskOutcome,
    create_base_styles,
};
