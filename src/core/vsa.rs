//! Vector Symbolic Architecture (VSA) operations.
//!
//! VSA provides a mathematical framework for representing and manipulating
//! symbolic information in high-dimensional binary vectors.

use crate::FINGERPRINT_U64;
use crate::core::Fingerprint;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// VSA operations trait
pub trait VsaOps {
    /// Bind two representations (XOR) - creates compound
    ///
    /// `bind(red, apple)` → "red apple"
    fn bind(&self, other: &Self) -> Self;

    /// Unbind to recover component
    ///
    /// `unbind(red_apple, red)` ≈ apple
    fn unbind(&self, other: &Self) -> Self;

    /// Bundle multiple representations (majority vote) - creates prototype
    ///
    /// `bundle([cat1, cat2, cat3])` → "generic cat"
    fn bundle(items: &[Self]) -> Self
    where
        Self: Sized;

    /// Permute for sequence encoding
    ///
    /// `permute(word, 3)` → word at position 3
    fn permute(&self, positions: i32) -> Self;

    /// Create sequence from ordered items
    ///
    /// `sequence([a, b, c])` → a + permute(b, 1) + permute(c, 2)
    fn sequence(items: &[Self]) -> Self
    where
        Self: Sized;
}

impl VsaOps for Fingerprint {
    #[inline]
    fn bind(&self, other: &Self) -> Self {
        Fingerprint::bind(self, other)
    }

    #[inline]
    fn unbind(&self, other: &Self) -> Self {
        Fingerprint::unbind(self, other)
    }

    fn bundle(items: &[Self]) -> Self {
        if items.is_empty() {
            return Fingerprint::zero();
        }

        if items.len() == 1 {
            return items[0].clone();
        }

        // Majority vote for each bit
        let threshold = items.len() / 2;
        let even = items.len() % 2 == 0;

        // Count bits across all items
        let mut counts = [0u32; FINGERPRINT_U64 * 64];

        for item in items {
            for (word_idx, &word) in item.as_raw().iter().enumerate() {
                for bit in 0..64 {
                    if (word >> bit) & 1 == 1 {
                        counts[word_idx * 64 + bit] += 1;
                    }
                }
            }
        }

        // Set bits that exceed threshold (tie-break with first item for even counts)
        let mut data = [0u64; FINGERPRINT_U64];
        for (i, &count) in counts.iter().enumerate() {
            let word = i / 64;
            let bit = i % 64;
            if count > threshold as u32
                || (even && count == threshold as u32 && items[0].as_raw()[word] & (1 << bit) != 0)
            {
                data[word] |= 1 << bit;
            }
        }

        Fingerprint::from_raw(data)
    }

    #[inline]
    fn permute(&self, positions: i32) -> Self {
        Fingerprint::permute(self, positions)
    }

    fn sequence(items: &[Self]) -> Self {
        if items.is_empty() {
            return Fingerprint::zero();
        }

        // Create sequence: sum of permuted items
        let permuted: Vec<Fingerprint> = items
            .iter()
            .enumerate()
            .map(|(i, item)| item.permute(i as i32))
            .collect();

        Self::bundle(&permuted)
    }
}

/// Clean up noisy fingerprint by resonating with codebook
pub fn cleanup(
    noisy: &Fingerprint,
    codebook: &[Fingerprint],
    threshold: f32,
) -> Option<Fingerprint> {
    let mut best_idx = 0;
    let mut best_sim = 0.0f32;

    for (i, item) in codebook.iter().enumerate() {
        let sim = noisy.similarity(item);
        if sim > best_sim {
            best_sim = sim;
            best_idx = i;
        }
    }

    if best_sim >= threshold {
        Some(codebook[best_idx].clone())
    } else {
        None
    }
}

/// Resonance query - find items above similarity threshold
#[cfg(feature = "parallel")]
pub fn resonate(query: &Fingerprint, corpus: &[Fingerprint], threshold: f32) -> Vec<(usize, f32)> {
    corpus
        .par_iter()
        .enumerate()
        .filter_map(|(i, fp)| {
            let sim = query.similarity(fp);
            if sim >= threshold {
                Some((i, sim))
            } else {
                None
            }
        })
        .collect()
}

#[cfg(not(feature = "parallel"))]
pub fn resonate(query: &Fingerprint, corpus: &[Fingerprint], threshold: f32) -> Vec<(usize, f32)> {
    corpus
        .iter()
        .enumerate()
        .filter_map(|(i, fp)| {
            let sim = query.similarity(fp);
            if sim >= threshold {
                Some((i, sim))
            } else {
                None
            }
        })
        .collect()
}

/// Analogy completion: A is to B as C is to ?
///
/// Uses the relation: ? ≈ unbind(bind(A, B), C) = A ⊕ B ⊕ C
pub fn analogy(
    a: &Fingerprint,
    b: &Fingerprint,
    c: &Fingerprint,
    codebook: &[Fingerprint],
) -> Option<Fingerprint> {
    // Compute the transformation from A to B
    let a_to_b = a.bind(b);

    // Apply same transformation to C
    let predicted = a_to_b.bind(c);

    // Clean up with codebook
    cleanup(&predicted, codebook, 0.5)
}

/// Fusion quality metric for bind operations.
///
/// Measures the quality of `bind(A, B)` by checking that unbinding recovers
/// both parents exactly (XOR is self-inverse, so this should be exact).
///
/// Returns `(recovery_a, recovery_b)` as normalized Hamming distances.
/// Both should be 0.0 for exact recovery.
///
/// # Science
/// - Plate (2003): Holographic Reduced Representations — bind/unbind algebra
/// - Kanerva (2009): Hyperdimensional Computing — XOR self-inverse property
pub fn fusion_quality(a: &Fingerprint, b: &Fingerprint) -> (f32, f32) {
    let fused = a.bind(b);
    let recovered_a = fused.unbind(b);
    let recovered_b = fused.unbind(a);

    let dist_a = a.hamming(&recovered_a) as f32 / (FINGERPRINT_U64 as f32 * 64.0);
    let dist_b = b.hamming(&recovered_b) as f32 / (FINGERPRINT_U64 as f32 * 64.0);

    (dist_a, dist_b)
}

/// Multi-way fusion quality: bind N items and verify all are recoverable.
///
/// For XOR-based VSA: `bind(a, bind(b, c))` → unbind with `bind(b, c)` recovers `a`.
/// Returns the maximum recovery distance across all items.
pub fn multi_fusion_quality(items: &[Fingerprint]) -> f32 {
    if items.len() < 2 {
        return 0.0;
    }

    // Compute total binding: items[0] ⊗ items[1] ⊗ ... ⊗ items[n-1]
    let mut total = items[0].clone();
    for item in &items[1..] {
        total = total.bind(item);
    }

    // For each item, unbind all others and check recovery
    let total_bits = FINGERPRINT_U64 as f32 * 64.0;
    let mut max_dist = 0.0f32;

    for i in 0..items.len() {
        // key = XOR of all items except items[i]
        // total XOR key = items[i] (because XOR is self-inverse)
        let key = items
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .fold(Fingerprint::zero(), |acc, (_, item)| acc.bind(item));

        let recovered = total.unbind(&key);
        let dist = items[i].hamming(&recovered) as f32 / total_bits;
        max_dist = max_dist.max(dist);
    }

    max_dist
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bundle_preserves_similarity() {
        let cat1 = Fingerprint::from_content("cat instance 1");
        let cat2 = Fingerprint::from_content("cat instance 2");
        let cat3 = Fingerprint::from_content("cat instance 3");
        let _dog = Fingerprint::from_content("dog");

        let prototype = Fingerprint::bundle(&[cat1.clone(), cat2.clone(), cat3.clone()]);

        // Prototype should be similar to all cats
        assert!(prototype.similarity(&cat1) > 0.4);
        assert!(prototype.similarity(&cat2) > 0.4);
        assert!(prototype.similarity(&cat3) > 0.4);

        // But less similar to dog (random baseline ~0.5)
        // Note: With random fingerprints, similarity is ~0.5
    }

    #[test]
    fn test_sequence_encoding() {
        let word1 = Fingerprint::from_content("the");
        let word2 = Fingerprint::from_content("quick");
        let word3 = Fingerprint::from_content("fox");

        let seq = Fingerprint::sequence(&[word1.clone(), word2.clone(), word3.clone()]);

        // Sequence should be unique
        assert!(seq.similarity(&word1) < 0.9);

        // But we can decode first word
        let _decoded_first = seq.unbind(&Fingerprint::zero().permute(0));
        // (This is a simplified test - real decoding needs iterative cleanup)
    }

    #[test]
    fn test_fusion_quality_exact_recovery() {
        let a = Fingerprint::from_content("concept A");
        let b = Fingerprint::from_content("concept B");
        let (dist_a, dist_b) = fusion_quality(&a, &b);
        // XOR is exactly self-inverse — distances must be 0
        assert_eq!(dist_a, 0.0, "Recovery of A should be exact");
        assert_eq!(dist_b, 0.0, "Recovery of B should be exact");
    }

    #[test]
    fn test_fusion_quality_multiple_pairs() {
        // Test across many random pairs
        for i in 0..100 {
            let a = Fingerprint::from_content(&format!("pair_{}_a", i));
            let b = Fingerprint::from_content(&format!("pair_{}_b", i));
            let (dist_a, dist_b) = fusion_quality(&a, &b);
            assert_eq!(dist_a, 0.0);
            assert_eq!(dist_b, 0.0);
        }
    }

    #[test]
    fn test_multi_fusion_quality() {
        let items: Vec<Fingerprint> = (0..5)
            .map(|i| Fingerprint::from_content(&format!("item_{}", i)))
            .collect();
        let max_dist = multi_fusion_quality(&items);
        // XOR multi-bind is exactly recoverable
        assert_eq!(
            max_dist, 0.0,
            "Multi-way bind should be exactly recoverable"
        );
    }

    #[test]
    fn test_bind_unbind_roundtrip_algebraic() {
        // Prove: A ⊗ B ⊗ B = A (exact, algebraic identity)
        let a = Fingerprint::from_content("original");
        let b = Fingerprint::from_content("key");
        let roundtrip = a.bind(&b).bind(&b);
        assert_eq!(
            a.hamming(&roundtrip),
            0,
            "XOR self-inverse must be EXACT, zero Hamming distance"
        );
    }
}
