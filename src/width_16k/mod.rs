//! 16Kbit (2^14) Cognitive Record — HDC-Aligned Layout
//!
//! Each CogRecord (container) is 16,384 bits = 256 × u64 = 2 KB.
//! A node is composed of separate CogRecords:
//!
//! - **Container 0** (Metadata): W0-W127 defined fields, W128-W255 reserved.
//! - **Container 1** (Content): All 256 words = searchable VSA fingerprint.
//! - **Container N** (Additional): Jina embeddings, etc.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │  Container 0: Metadata (16,384 bits = 256 words)                    │
//! │    W0-127:  defined fields (identity, NARS, edges, qualia, ...)     │
//! │    W128-255: reserved for future expansion                          │
//! ├──────────────────────────────────────────────────────────────────────┤
//! │  Container 1: Content (16,384 bits = 256 words)                     │
//! │    W0-255:  searchable VSA fingerprint (Hamming / XOR-bind)         │
//! └──────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! Distance = popcount(XOR(a[0..256], b[0..256])) over all content words.

pub mod compat;
pub mod schema;
pub mod search;
pub mod xor_bubble;

// ============================================================================
// VECTOR DIMENSIONS
// ============================================================================

/// Number of logical bits in the vector (2^14)
pub const VECTOR_BITS: usize = 16_384;

/// Number of u64 words: 16384/64 = 256 (exact, no remainder)
pub const VECTOR_WORDS: usize = VECTOR_BITS / 64; // 256

/// Raw bytes per vector: 256 x 8 = 2,048
pub const VECTOR_BYTES: usize = VECTOR_WORDS * 8; // 2048

/// Mask for the last word — all bits (no masking needed)
pub const LAST_WORD_MASK: u64 = u64::MAX;

// ============================================================================
// STATISTICAL CONSTANTS (Hamming distribution over full 16K)
// ============================================================================

/// Expected Hamming distance between two random vectors = n/2
pub const EXPECTED_RANDOM_DISTANCE: f64 = VECTOR_BITS as f64 / 2.0; // 8192.0

/// Standard deviation: sigma = sqrt(n/4) = sqrt(4096) = 64 (one u64 word!)
pub const HAMMING_STD_DEV: f64 = 64.0;

/// One sigma threshold
pub const ONE_SIGMA: u32 = 64;

/// Two sigma threshold
pub const TWO_SIGMA: u32 = 128;

/// Three sigma threshold (99.7% confidence)
pub const THREE_SIGMA: u32 = 192;

// ============================================================================
// CONTENT CONTAINER LAYOUT — all 256 words are searchable
// ============================================================================

/// Content words: all 256 words (full container is searchable fingerprint).
pub const CONTENT_WORDS: usize = VECTOR_WORDS; // 256

/// Content offset: starts at word 0.
pub const CONTENT_OFFSET: usize = 0;

// ============================================================================
// BLOCK LAYOUT: 16 blocks of 16 words (1024 bits each)
// ============================================================================

/// Words per block
pub const WORDS_PER_BLOCK: usize = 16;

/// Number of blocks: 256/16 = 16
pub const NUM_BLOCKS: usize = VECTOR_WORDS / WORDS_PER_BLOCK; // 16

/// Bits per block
pub const BITS_PER_BLOCK: usize = WORDS_PER_BLOCK * 64; // 1024

// ============================================================================
// SIMD LAYOUT — All zero remainder
// ============================================================================

/// AVX-512: 256/8 = 32 iterations (exact)
pub const AVX512_ITERATIONS: usize = VECTOR_WORDS / 8; // 32
pub const AVX512_REMAINDER: usize = 0;

/// AVX2: 256/4 = 64 iterations (exact)
pub const AVX2_ITERATIONS: usize = VECTOR_WORDS / 4; // 64
pub const AVX2_REMAINDER: usize = 0;

/// NEON: 256/2 = 128 iterations (exact)
pub const NEON_ITERATIONS: usize = VECTOR_WORDS / 2; // 128
pub const NEON_REMAINDER: usize = 0;

// ============================================================================
// BACKWARD COMPAT — old names (deprecated, use CONTENT_WORDS)
// ============================================================================

/// Deprecated: use `CONTENT_WORDS`. Kept for compat with search/compat modules.
pub const RESONANCE_WORDS: usize = CONTENT_WORDS;

// ============================================================================
// BELICHTUNGSMESSER SAMPLE POINTS
// ============================================================================

/// Strategic 7-point sample indices within the content region (words 0-255).
pub const SAMPLE_POINTS: [usize; 7] = [0, 37, 82, 128, 171, 213, 251];

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimensions() {
        assert_eq!(VECTOR_BITS, 16_384);
        assert_eq!(VECTOR_WORDS, 256);
        assert_eq!(VECTOR_BYTES, 2048);
    }

    #[test]
    fn test_sigma_is_one_word() {
        assert_eq!(ONE_SIGMA, 64);
        assert_eq!(TWO_SIGMA, 128);
        assert_eq!(THREE_SIGMA, 192);
    }

    #[test]
    fn test_content_is_full_container() {
        assert_eq!(CONTENT_WORDS, VECTOR_WORDS);
        assert_eq!(CONTENT_WORDS, 256);
        assert_eq!(CONTENT_OFFSET, 0);
    }

    #[test]
    fn test_block_layout() {
        assert_eq!(NUM_BLOCKS, 16);
        assert_eq!(NUM_BLOCKS * WORDS_PER_BLOCK, VECTOR_WORDS);
    }

    #[test]
    fn test_simd_zero_remainder() {
        assert_eq!(VECTOR_WORDS % 8, 0);
        assert_eq!(AVX512_REMAINDER, 0);
        assert_eq!(AVX2_REMAINDER, 0);
        assert_eq!(NEON_REMAINDER, 0);
    }

    #[test]
    fn test_sample_points_in_content_region() {
        for &p in &SAMPLE_POINTS {
            assert!(p < CONTENT_WORDS, "Sample point {} outside content", p);
        }
    }

    #[test]
    fn test_fingerprint_covers_content() {
        const { assert!(crate::FINGERPRINT_U64 >= VECTOR_WORDS) };
        assert_eq!(CONTENT_WORDS, VECTOR_WORDS);
    }
}
