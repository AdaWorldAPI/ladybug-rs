//! Learning Module - Meta-AGI Learning Loop
//!
//! The learning curve IS the knowledge.
//! Similar problems FEEL similar before you know WHY.
//! Capture the feeling, retrieve the solution.
//!
//! # The 6-Phase Learning Loop
//!
//! ```text
//! 1. ENCOUNTER   → Log to blackboard
//! 2. STRUGGLE    → Capture attempt vectors (resonance)
//! 3. BREAKTHROUGH→ Extract concept (high satisfaction qualia)
//! 4. CONSOLIDATE → Ice-cake decisions (FLOW/HOLD/BLOCK)
//! 5. APPLY       → Query resonance for "felt this before"
//! 6. META-LEARN  → Track what patterns work
//! ```

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
