//! Learning module - Meta-AGI Learning Loop + CAM Operations + Cognitive Frameworks
//!
//! CAM = Content Addressable Methods
//! 4096 operations as a unified cognitive vocabulary.
//! Everything stays in fingerprint space - no context switching.
//!
//! Cognitive Frameworks:
//! - NARS: Non-Axiomatic Reasoning System
//! - ACT-R: Adaptive Control of Thought
//! - RL: Reinforcement Learning
//! - Causality: Pearl's do-calculus
//! - Qualia: Affect channels
//! - Rung: Abstraction ladder

pub mod moment;
pub mod session;
pub mod blackboard;
pub mod resonance;
pub mod concept;
pub mod cam_ops;
pub mod cognitive_frameworks;

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
    // RL
    StateAction, QValue, RlAgent,
    // Causality
    CausalRelation, CausalNode, CausalEdge, DoOperator, Counterfactual,
    // Qualia
    QualiaChannel, QualiaState,
    // Rung
    Rung, RungClassifier,
};
