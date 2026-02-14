//! Search module - Alien Magic Vector Search + Causal Reasoning + Cognitive Search
//!
//! This module provides search APIs that look like float vector search
//! but run on pure integer SIMD operations with cognitive extensions.
//!
//! # Features
//!
//! - **HDR Cascade**: Hierarchical filtering (1-bit → 4-bit → 8-bit → exact)
//! - **Mexican Hat**: Discrimination with excitation and inhibition zones
//! - **Rolling σ**: Window-based coherence detection
//! - **A⊗B⊗B=A**: O(1) direct retrieval via XOR unbinding
//! - **Causal Search**: Three rungs (correlate, intervene, counterfact)
//! - **Cognitive Search**: NARS inference + Qualia resonance + SPO structure
//!
//! # Human-like Cognitive Operations
//!
//! ```text
//! DEDUCE      → What must follow? (NARS deduction)
//! INDUCE      → What pattern emerges? (NARS induction)
//! ABDUCT      → What explains this? (NARS abduction)
//! CONTRADICT  → What conflicts? (NARS negation)
//! INTUIT      → What feels right? (qualia resonance)
//! ASSOCIATE   → What's related? (qualia similarity)
//! FANOUT      → What connects? (SPO expansion)
//! EXTRAPOLATE → What comes next? (sequence prediction)
//! SYNTHESIZE  → How do these combine? (bundle + revision)
//! JUDGE       → Is this true? (truth evaluation)
//! ```
//!
//! # Example
//!
//! ```ignore
//! use ladybug::search::{CognitiveSearch, QualiaVector, TruthValue};
//!
//! let mut search = CognitiveSearch::new();
//!
//! // Add atoms with qualia and truth
//! search.add_with_qualia(fp, qualia, truth);
//!
//! // Intuit: find what "feels" similar
//! let results = search.intuit(&query_qualia, 10);
//!
//! // Deduce: draw conclusions
//! let conclusion = search.deduce(&premise1, &premise2);
//!
//! // Judge: evaluate truth
//! let truth = search.judge(&statement);
//! ```

pub mod causal;
pub mod certificate;
pub mod cognitive;
pub mod distribution;
pub mod hdr_cascade;
pub mod scientific;
pub mod temporal;

pub use hdr_cascade::{
    AlienSearch,

    // Bound retrieval (A⊗B⊗B=A)
    BoundRetrieval,

    // Fingerprint extension trait
    FingerprintSearch,

    // HDR index
    HdrIndex,

    // Mexican hat
    MexicanHat,

    QualityTracker,
    // Rolling window
    RollingWindow,

    RubiconSearch,

    // Unified API (the alien magic)
    SearchResult,
    SignalClass,
    VoyagerResult,
    // Belichtungsmesser (adaptive threshold search)
    belichtung_meter,
    classify_signal,
    // Core operations
    hamming_distance,
    sketch_1bit,
    sketch_1bit_sum,
    sketch_4bit,
    sketch_4bit_sum,
    sketch_8bit,
    sketch_8bit_sum,

    // Voyager deep field (orthogonal superposition cleaning)
    superposition_clean,
};

pub use causal::{
    CausalEdge,

    CausalResult,
    CausalSearch,
    // Verbs
    CausalVerbs,

    // Rung stores
    CorrelationStore,
    CounterfactualStore,

    // Edge types
    EdgeType,
    InterventionStore,
    // Unified API
    QueryMode,
};

pub use certificate::{CausalCertificate, EffectClass};

pub use cognitive::{
    // Cognitive atom
    CognitiveAtom,

    // Results
    CognitiveResult,
    // Unified cognitive search
    CognitiveSearch,
    // Qualia
    QualiaVector,

    RelevanceScores,

    SearchVia,
    // SPO
    SpoTriple,
};
