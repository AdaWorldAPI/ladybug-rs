//! Fingerprint ↔ 16K Record Compatibility Layer
//!
//! Now that both `Fingerprint` and `Container` use 256 words (16,384 bits),
//! the conversions here are identity operations. This module is retained
//! for API compatibility and for future use if a narrower format is
//! reintroduced (e.g., compressed wire format).
//!
//! Three conversion strategies:
//! - **zero_extend**: Copy Fingerprint words into a 256-word record.
//! - **truncate**: Extract Fingerprint from a 256-word record.
//! - **xor_fold**: Identity (no surplus — Fingerprint = Container = 256 words).

use super::CONTENT_WORDS;
use super::VECTOR_WORDS as WORDS_16K;
use super::schema::SchemaSidecar;
use crate::core::DIM_U64 as WORDS_FP;
use crate::core::Fingerprint;

/// Extend a Fingerprint into a 16K record.
///
/// With the 16K migration, Fingerprint is already 256 words, so this
/// is a direct copy.
pub fn zero_extend(fp: &Fingerprint) -> [u64; WORDS_16K] {
    let mut record = [0u64; WORDS_16K];
    let src = fp.as_raw();
    record[..WORDS_FP].copy_from_slice(src);
    record
}

/// Extend and attach schema metadata in one step.
pub fn zero_extend_with_schema(fp: &Fingerprint, schema: &SchemaSidecar) -> [u64; WORDS_16K] {
    let mut record = zero_extend(fp);
    schema.write_to_words(&mut record);
    record
}

/// Extract a Fingerprint from a 16K record.
///
/// Takes the first WORDS_FP (256) words. With the 16K migration this
/// is a direct copy of the full record.
pub fn truncate(record: &[u64; WORDS_16K]) -> Fingerprint {
    let mut words = [0u64; WORDS_FP];
    words.copy_from_slice(&record[..WORDS_FP]);
    Fingerprint::from_raw(words)
}

/// XOR-fold surplus words back into the core region.
///
/// With the 16K migration (WORDS_FP = 256 = CONTENT_WORDS), there are
/// no surplus words and this is equivalent to `truncate`.
pub fn xor_fold(record: &[u64; WORDS_16K]) -> Fingerprint {
    let mut words = [0u64; WORDS_FP];
    words.copy_from_slice(&record[..WORDS_FP]);

    // No surplus: WORDS_FP (256) >= CONTENT_WORDS (256), loop doesn't execute.
    let surplus_start = WORDS_FP;
    let surplus_end = CONTENT_WORDS;
    for i in surplus_start..surplus_end {
        let target = (i - surplus_start) % WORDS_FP;
        words[target] ^= record[i];
    }

    Fingerprint::from_raw(words)
}

/// Compute Hamming distance between a Fingerprint and a 16K record.
///
/// Uses the shared WORDS_FP (256) words. With the 16K migration this
/// covers the full record.
pub fn cross_width_distance(fp: &Fingerprint, record: &[u64; WORDS_16K]) -> u32 {
    let src = fp.as_raw();
    let mut dist = 0u32;
    for i in 0..WORDS_FP {
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
    fn test_zero_extend_with_schema() {
        let fp = Fingerprint::from_content("test schema");
        let mut schema = SchemaSidecar::default();
        schema.identity.depth = 5;
        schema.ani_levels.planning = 1000;

        let record = zero_extend_with_schema(&fp, &schema);

        // Content zone W0-223 preserved (schema at W224-255)
        assert_eq!(
            &record[..224],
            &fp.as_raw()[..224]
        );
        // Schema written
        let recovered = SchemaSidecar::read_from_words(&record);
        assert_eq!(recovered.identity.depth, 5);
        assert_eq!(recovered.ani_levels.planning, 1000);
    }

    #[test]
    fn test_xor_fold_identity() {
        // With 16K migration, xor_fold and truncate produce the same result
        let fp = Fingerprint::from_content("test fold");
        let record = zero_extend(&fp);

        let folded = xor_fold(&record);
        let truncated = truncate(&record);
        assert_eq!(folded.as_raw(), truncated.as_raw());
    }

    #[test]
    fn test_cross_width_distance_self() {
        let fp = Fingerprint::from_content("test distance");
        let record = zero_extend(&fp);
        assert_eq!(cross_width_distance(&fp, &record), 0);
    }
}
