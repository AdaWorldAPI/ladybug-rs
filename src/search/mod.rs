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

pub mod hdr_cascade;
pub mod causal;
pub mod cognitive;
pub mod scientific;

pub use hdr_cascade::{
    // Core operations
    hamming_distance,
    sketch_1bit,
    sketch_4bit,
    sketch_8bit,
    sketch_1bit_sum,
    sketch_4bit_sum,
    sketch_8bit_sum,

    // Mexican hat
    MexicanHat,

    // Rolling window
    RollingWindow,

    // HDR index
    HdrIndex,

    // Bound retrieval (A⊗B⊗B=A)
    BoundRetrieval,

    // Unified API (the alien magic)
    SearchResult,
    AlienSearch,

    // Fingerprint extension trait
    FingerprintSearch,

    // Belichtungsmesser (adaptive threshold search)
    belichtung_meter,
    QualityTracker,
    RubiconSearch,

    // Voyager deep field (orthogonal superposition cleaning)
    superposition_clean,
    VoyagerResult,
    SignalClass,
    classify_signal,
};

pub use causal::{
    // Verbs
    CausalVerbs,
    
    // Edge types
    EdgeType,
    CausalEdge,
    
    // Rung stores
    CorrelationStore,
    InterventionStore,
    CounterfactualStore,
    
    // Unified API
    QueryMode,
    CausalResult,
    CausalSearch,
};

pub use cognitive::{
    // Qualia
    QualiaVector,
    
    // SPO
    SpoTriple,
    
    // Cognitive atom
    CognitiveAtom,
    
    // Results
    CognitiveResult,
    SearchVia,
    RelevanceScores,
    
    // Unified cognitive search
    CognitiveSearch,
};
