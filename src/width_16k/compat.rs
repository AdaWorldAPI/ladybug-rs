//! 10K ↔ 16K Compatibility Layer
//!
//! Converts between the existing 157-word (10K-bit) Fingerprint and the
//! new 256-word (16K-bit) cognitive record. Phase 1: additive only —
//! this module does not modify existing Fingerprint code.
//!
//! Three conversion strategies:
//! - **zero_extend**: Copy 157 words, zero-pad to 256. Lossless, reversible.
//! - **truncate**: Take first 157 words. Lossy (drops surplus + metadata).
//! - **xor_fold**: XOR surplus words (157-223) back into the first 157.
//!   Preserves information density at the cost of non-reversibility.

use crate::core::Fingerprint;
use crate::core::{DIM_U64 as WORDS_10K};
use super::VECTOR_WORDS as WORDS_16K;
use super::RESONANCE_WORDS;
use super::schema::SchemaSidecar;

/// Zero-extend a 10K fingerprint into a 16K record.
///
/// Words 0-156: copied from the fingerprint (the 10K resonance pattern).
/// Words 157-223: zeroed (surplus, available for upscaling membrane).
/// Words 224-255: zeroed (metadata sidecar, written separately).
pub fn zero_extend(fp: &Fingerprint) -> [u64; WORDS_16K] {
    let mut record = [0u64; WORDS_16K];
    let src = fp.as_raw();
    record[..WORDS_10K].copy_from_slice(src);
    record
}

/// Zero-extend and attach schema metadata in one step.
pub fn zero_extend_with_schema(fp: &Fingerprint, schema: &SchemaSidecar) -> [u64; WORDS_16K] {
    let mut record = zero_extend(fp);
    schema.write_to_words(&mut record);
    record
}

/// Truncate a 16K record back to a 10K fingerprint.
///
/// Takes only the first 157 words. Surplus and metadata are discarded.
/// The raw words are preserved exactly — no masking is applied, matching
/// the behavior of `Fingerprint::from_raw()` which stores verbatim.
pub fn truncate(record: &[u64; WORDS_16K]) -> Fingerprint {
    let mut words = [0u64; WORDS_10K];
    words.copy_from_slice(&record[..WORDS_10K]);
    Fingerprint::from_raw(words)
}

/// XOR-fold surplus words back into the 10K resonance.
///
/// Words 157-223 (the surplus) are XORed back into the first 157 words,
/// preserving their information in the original address space. This is
/// useful when downgrading a record that has been enriched by the
/// upscaling membrane.
///
/// Not reversible: the surplus information is folded into the original
/// words, but cannot be separated out again.
pub fn xor_fold(record: &[u64; WORDS_16K]) -> Fingerprint {
    let mut words = [0u64; WORDS_10K];
    words.copy_from_slice(&record[..WORDS_10K]);

    // Fold surplus words 157-223 back into 0..156 (wrapping)
    let surplus_start = WORDS_10K; // 157
    let surplus_end = RESONANCE_WORDS; // 224
    for i in surplus_start..surplus_end {
        let target = (i - surplus_start) % WORDS_10K;
        words[target] ^= record[i];
    }

    // Mask last word
    words[WORDS_10K - 1] &= crate::core::LAST_MASK;
    Fingerprint::from_raw(words)
}

/// Compute cross-width Hamming distance.
///
/// Compares a 10K fingerprint against a 16K record using only the
/// shared 157 words. The surplus and metadata words are ignored.
pub fn cross_width_distance(fp: &Fingerprint, record: &[u64; WORDS_16K]) -> u32 {
    let src = fp.as_raw();
    let mut dist = 0u32;
    for i in 0..WORDS_10K {
        dist += (src[i] ^ record[i]).count_ones();
    }
    dist
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_extend_truncate_roundtrip() {
        let fp = Fingerprint::from_content("test roundtrip");
        let record = zero_extend(&fp);
        let recovered = truncate(&record);
        assert_eq!(fp.as_raw(), recovered.as_raw());
    }

    #[test]
    fn test_zero_extend_padding() {
        let fp = Fingerprint::from_content("test padding");
        let record = zero_extend(&fp);
        // Words 157-255 should be zero
        for i in WORDS_10K..WORDS_16K {
            assert_eq!(record[i], 0, "Word {} should be zero", i);
        }
    }

    #[test]
    fn test_zero_extend_with_schema() {
        let fp = Fingerprint::from_content("test schema");
        let mut schema = SchemaSidecar::default();
        schema.identity.depth = 5;
        schema.ani_levels.planning = 1000;

        let record = zero_extend_with_schema(&fp, &schema);

        // Resonance preserved
        assert_eq!(&record[..WORDS_10K], fp.as_raw());
        // Schema written
        let recovered = SchemaSidecar::read_from_words(&record);
        assert_eq!(recovered.identity.depth, 5);
        assert_eq!(recovered.ani_levels.planning, 1000);
    }

    #[test]
    fn test_xor_fold() {
        let fp = Fingerprint::from_content("test fold");
        let mut record = zero_extend(&fp);

        // Put some data in the surplus region
        record[160] = 0xDEADBEEF;
        record[200] = 0xCAFEBABE;

        let folded = xor_fold(&record);
        let plain = truncate(&record);

        // Folded should differ from plain truncation
        // (because surplus data was XORed in)
        assert_ne!(folded.as_raw(), plain.as_raw());
    }

    #[test]
    fn test_cross_width_distance_self() {
        let fp = Fingerprint::from_content("test distance");
        let record = zero_extend(&fp);
        assert_eq!(cross_width_distance(&fp, &record), 0);
    }

    #[test]
    fn test_cross_width_distance_ignores_surplus() {
        let fp = Fingerprint::from_content("test surplus");
        let mut record = zero_extend(&fp);
        // Modify surplus and metadata — should not affect distance
        record[160] = 0xFFFF;
        record[240] = 0xFFFF;
        assert_eq!(cross_width_distance(&fp, &record), 0);
    }
}
