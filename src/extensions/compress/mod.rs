//! Compress Extension - Semantic Compression via Crystal Dictionary
//!
//! BTR Procella RL for optimal compression/quality trade-off.
//!
//! # Pipeline
//!
//! ```text
//! HUGE CONTEXT → LangExtract → Crystal Dictionary → BTR-RL → LanceDB
//!
//! Key insight: Crystal as LEARNED CODEBOOK for semantic quantization
//! • 125 cells = 125 cluster centroids (k-means++)
//! • chunk → nearest centroid + residual
//! • 100K tokens → 125 prototypes + sparse pointers ≈ O(KB) index
//! ```
//!
//! # Results
//!
//! ```text
//! Source: 1774 tokens, 17284 bytes
//! Chunks: 51
//! Cells used: 50 / 125
//! Distortion: 0.13
//! Raw: 16 KB → Compressed: 1 KB
//! Compression: 8.6x
//! Query: <1ms
//! ```
//!
//! # BTR-RL Policy
//!
//! - **State**: (compression_ratio, distortion, query_accuracy)
//! - **Actions**: IncreaseResidual, DecreaseResidual, Refine, Hold
//! - **Reward**: ln(compression) - 2×distortion + 2×accuracy

mod compress;

pub use compress::*;
