//! Learning module - Meta-AGI Learning Loop

pub mod moment;
pub mod session;
pub mod blackboard;
pub mod resonance;
pub mod concept;

pub use moment::{Moment, MomentType, Qualia, MomentBuilder};
pub use session::{LearningSession, SessionState, SessionPhase};
pub use blackboard::{Blackboard, Decision, IceCakedLayer};
pub use resonance::{ResonanceCapture, SimilarMoment, ResonanceStats, find_sweet_spot, mexican_hat_resonance};
pub use concept::{ConceptExtractor, ExtractedConcept, RelationType, ConceptRelation};
