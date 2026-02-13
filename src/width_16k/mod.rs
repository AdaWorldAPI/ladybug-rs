//! 16Kbit (2^14) Cognitive Record — HDC-Aligned Layout
//!
//! Hyperdimensional computing requires maximally homogeneous vectors.
//! Metadata compresses into 32 words (2,048 bits = 12.5% of the record),
//! leaving 224 words (14,336 bits = 87.5%) for Hamming resonance.
//!
//! The four HDC hallmarks drive this layout:
//! - **Homogeneous**: minimize non-resonance zone (32 words, not 64)
//! - **Holographic**: metadata distributed in compact sidecar, not sprawled
//! - **Binding**: XOR over resonance words = the universal distance/bind op
//! - **Robust**: 14,336 resonance bits with wide noise margin
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │  Words 0-223   (224 words = 14,336 bits)  Hamming resonance         │
//! │    Full 16K fingerprint covers all 256 words (resonance + metadata) │
//! │                                                                      │
//! │  Words 224-255 ( 32 words =  2,048 bits)  Metadata sidecar          │
//! │    Block 14 (224-239): Identity + Reasoning + Learning               │
//! │    Block 15 (240-255): Graph topology + inline edges                 │
//! └──────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! Distance = popcount(XOR(a[0..224], b[0..224])). Metadata excluded.
//! No self_addr / parent_addr fields — the DN address path encodes both.

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
// RESONANCE / METADATA SPLIT
// ============================================================================

/// Resonance words: 224 (14,336 bits = 87.5% of the record)
pub const RESONANCE_WORDS: usize = 224;

/// Metadata words: 32 (2,048 bits = 12.5% of the record)
pub const METADATA_WORDS: usize = 32;

/// First word of the metadata sidecar
pub const METADATA_WORD_START: usize = RESONANCE_WORDS; // 224

// ============================================================================
// BLOCK LAYOUT: 16 blocks of 16 words (1024 bits each)
// ============================================================================

/// Words per block
pub const WORDS_PER_BLOCK: usize = 16;

/// Number of blocks: 256/16 = 16
pub const NUM_BLOCKS: usize = VECTOR_WORDS / WORDS_PER_BLOCK; // 16

/// Bits per block
pub const BITS_PER_BLOCK: usize = WORDS_PER_BLOCK * 64; // 1024

/// Number of resonance blocks (blocks 0-13 = 224 words = 14,336 bits)
pub const RESONANCE_BLOCKS: usize = 14;

/// First metadata block index (block 14 = word 224)
pub const SCHEMA_BLOCK_START: usize = 14;

/// Number of metadata blocks (blocks 14-15 = 32 words = 2,048 bits)
pub const SCHEMA_BLOCK_COUNT: usize = 2;

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

/// Resonance-only AVX-512: 224/8 = 28 iterations (exact)
pub const RESONANCE_AVX512_ITERATIONS: usize = RESONANCE_WORDS / 8; // 28

// ============================================================================
// BELICHTUNGSMESSER SAMPLE POINTS
// ============================================================================

/// Strategic 7-point sample indices within the resonance region (words 0-223).
pub const SAMPLE_POINTS: [usize; 7] = [0, 32, 67, 112, 149, 183, 219];

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
    fn test_resonance_metadata_split() {
        assert_eq!(RESONANCE_WORDS + METADATA_WORDS, VECTOR_WORDS);
        assert_eq!(METADATA_WORD_START, 224);
        assert_eq!(METADATA_WORDS, 32);
        // 87.5% resonance
        assert_eq!(RESONANCE_WORDS * 100 / VECTOR_WORDS, 87);
    }

    #[test]
    fn test_block_layout() {
        assert_eq!(NUM_BLOCKS, 16);
        assert_eq!(NUM_BLOCKS * WORDS_PER_BLOCK, VECTOR_WORDS);
        assert_eq!(RESONANCE_BLOCKS + SCHEMA_BLOCK_COUNT, NUM_BLOCKS);
        assert_eq!(SCHEMA_BLOCK_START * WORDS_PER_BLOCK, METADATA_WORD_START);
    }

    #[test]
    fn test_simd_zero_remainder() {
        assert_eq!(VECTOR_WORDS % 8, 0);
        assert_eq!(RESONANCE_WORDS % 8, 0);
        assert_eq!(AVX512_REMAINDER, 0);
        assert_eq!(AVX2_REMAINDER, 0);
        assert_eq!(NEON_REMAINDER, 0);
        assert_eq!(RESONANCE_AVX512_ITERATIONS, 28);
    }

    #[test]
    fn test_sample_points_in_resonance_region() {
        for &p in &SAMPLE_POINTS {
            assert!(p < RESONANCE_WORDS, "Sample point {} outside resonance", p);
        }
    }

    #[test]
    fn test_fingerprint_covers_resonance() {
        // Fingerprint (256 words) covers full resonance (224 words) + metadata (32 words)
        const { assert!(crate::FINGERPRINT_U64 >= VECTOR_WORDS) };
        assert_eq!(VECTOR_WORDS, RESONANCE_WORDS + METADATA_WORDS);
    }

    #[test]
    fn test_metadata_alignment() {
        // Metadata starts at word 224 = 28 AVX-512 iterations boundary
        assert_eq!(METADATA_WORD_START % 8, 0);
        // Metadata region is exactly 2 blocks
        assert_eq!(METADATA_WORDS / WORDS_PER_BLOCK, 2);
    }
}
