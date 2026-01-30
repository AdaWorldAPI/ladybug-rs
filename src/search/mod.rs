//! Search module - Alien Magic Vector Search
//!
//! This module provides a search API that looks like float vector search
//! (Faiss, Annoy, etc.) but runs on pure integer SIMD operations.
//!
//! # Features
//!
//! - **HDR Cascade**: Hierarchical filtering (1-bit → 4-bit → 8-bit → exact)
//! - **Mexican Hat**: Discrimination with excitation and inhibition zones
//! - **Rolling σ**: Window-based coherence detection
//! - **A⊗B⊗B=A**: O(1) direct retrieval via XOR unbinding
//!
//! # Example
//!
//! ```ignore
//! use ladybug::search::{AlienSearch, MexicanHat};
//!
//! // Create search engine
//! let mut search = AlienSearch::with_capacity(10000);
//!
//! // Add fingerprints (looks like adding vectors)
//! for fp in fingerprints {
//!     search.add(&fp.to_words());
//! }
//!
//! // Search returns similarity scores (looks like cosine similarity!)
//! let results = search.search_similarity(&query.to_words(), 10);
//! // results: [(index, 0.95), (index, 0.91), ...]
//!
//! // Or use Mexican hat for discrimination
//! let results = search.search_discriminate(&query.to_words(), 10);
//! ```

pub mod hdr_cascade;

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
};
