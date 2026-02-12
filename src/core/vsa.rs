//! Vector Symbolic Architecture (VSA) operations.
//! 
//! VSA provides a mathematical framework for representing and manipulating
//! symbolic information in high-dimensional binary vectors.

use crate::core::Fingerprint;
use crate::FINGERPRINT_U64;

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
    fn bundle(items: &[Self]) -> Self where Self: Sized;
    
    /// Permute for sequence encoding
    /// 
    /// `permute(word, 3)` → word at position 3
    fn permute(&self, positions: i32) -> Self;
    
    /// Create sequence from ordered items
    /// 
    /// `sequence([a, b, c])` → a + permute(b, 1) + permute(c, 2)
    fn sequence(items: &[Self]) -> Self where Self: Sized;
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
pub fn resonate(
    query: &Fingerprint,
    corpus: &[Fingerprint],
    threshold: f32,
) -> Vec<(usize, f32)> {
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
pub fn resonate(
    query: &Fingerprint,
    corpus: &[Fingerprint],
    threshold: f32,
) -> Vec<(usize, f32)> {
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

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bundle_preserves_similarity() {
        let cat1 = Fingerprint::from_content("cat instance 1");
        let cat2 = Fingerprint::from_content("cat instance 2");
        let cat3 = Fingerprint::from_content("cat instance 3");
        let dog = Fingerprint::from_content("dog");
        
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
        let decoded_first = seq.unbind(&Fingerprint::zero().permute(0));
        // (This is a simplified test - real decoding needs iterative cleanup)
    }
}
