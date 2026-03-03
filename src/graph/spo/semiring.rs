//! SpoSemiring — Pluggable traversal algebra for SPO graph walks.
//!
//! Stolen from holograph's GraphBLAS semiring pattern. Just the trait +
//! two concrete implementations, not the full GraphBLAS API.
//!
//! ```text
//! ⊗ = edge step   (how to compose across an edge)
//! ⊕ = accumulate  (how to combine multiple paths)
//!
//! Query                       ⊗ (multiply)        ⊕ (add)
//! "What does X cause?"        XOR bind            Bundle (majority)
//! "Shortest semantic path"    Hamming distance    Min
//! "Best match for X"          Similarity          Max
//! "Is X reachable from Y?"    AND                 OR
//! ```

use ladybug_contract::container::Container;

/// Semiring for SPO graph traversal.
///
/// ⊗ = edge step (how to compose across an edge)
/// ⊕ = accumulate (how to combine multiple paths)
///
/// Implementations must satisfy:
/// - `add(zero, x) == x`        (identity)
/// - `add(a, b) == add(b, a)`   (commutativity)
/// - `multiply(edge, zero)` is semantically neutral
pub trait SpoSemiring: Clone + Send + Sync + 'static {
    /// The accumulator type carried through traversal.
    type Value: Clone + Send + Sync;

    /// Additive identity — no path found.
    fn zero(&self) -> Self::Value;

    /// ⊗: compose input across an edge.
    fn multiply(&self, edge_fp: &Container, input: &Self::Value) -> Self::Value;

    /// ⊕: combine two path results.
    fn add(&self, a: &Self::Value, b: &Self::Value) -> Self::Value;

    /// Is this the identity element (no useful information)?
    fn is_zero(&self, v: &Self::Value) -> bool;
}

// ============================================================================
// XOR-BIND + BUNDLE (default HDC path composition)
// ============================================================================

/// Default semiring: XOR bind for edge traversal, majority-vote bundle for accumulation.
///
/// This is standard HDC/VSA path composition:
/// - Crossing an edge XOR-binds the edge fingerprint into the running state
/// - Multiple paths merge via majority-vote bundle
/// - Zero is the all-zeros container
#[derive(Clone)]
pub struct XorBundle;

impl SpoSemiring for XorBundle {
    type Value = Container;

    #[inline]
    fn zero(&self) -> Container {
        Container::zero()
    }

    #[inline]
    fn multiply(&self, edge_fp: &Container, input: &Container) -> Container {
        input.xor(edge_fp)
    }

    fn add(&self, a: &Container, b: &Container) -> Container {
        if a.is_zero() { return b.clone(); }
        if b.is_zero() { return a.clone(); }
        Container::bundle(&[a, b])
    }

    #[inline]
    fn is_zero(&self, v: &Container) -> bool {
        v.is_zero()
    }
}

// ============================================================================
// HAMMING-MIN (shortest semantic path)
// ============================================================================

/// Shortest-semantic-path semiring: Hamming cost for edges, min for accumulation.
///
/// - Crossing an edge costs `popcount(edge_fp)` (how much information the edge carries)
/// - Multiple paths keep the shortest
/// - Zero is `u32::MAX` (unreachable)
#[derive(Clone)]
pub struct HammingMin;

impl SpoSemiring for HammingMin {
    type Value = u32;

    #[inline]
    fn zero(&self) -> u32 {
        u32::MAX
    }

    #[inline]
    fn multiply(&self, edge_fp: &Container, input: &u32) -> u32 {
        input.saturating_add(edge_fp.popcount())
    }

    #[inline]
    fn add(&self, a: &u32, b: &u32) -> u32 {
        (*a).min(*b)
    }

    #[inline]
    fn is_zero(&self, v: &u32) -> bool {
        *v == u32::MAX
    }
}

// ============================================================================
// SIMILARITY-MAX (best match)
// ============================================================================

/// Best-match semiring: similarity score for edges, max for accumulation.
///
/// - Crossing an edge multiplies running similarity by edge similarity
/// - Multiple paths keep the best (highest similarity)
/// - Zero is `0.0` (no match)
#[derive(Clone)]
pub struct SimilarityMax;

impl SpoSemiring for SimilarityMax {
    type Value = f32;

    #[inline]
    fn zero(&self) -> f32 {
        0.0
    }

    #[inline]
    fn multiply(&self, edge_fp: &Container, input: &f32) -> f32 {
        // Similarity of the edge = normalized popcount density
        let density = edge_fp.popcount() as f32 / ladybug_contract::container::CONTAINER_BITS as f32;
        input * density
    }

    #[inline]
    fn add(&self, a: &f32, b: &f32) -> f32 {
        a.max(*b)
    }

    #[inline]
    fn is_zero(&self, v: &f32) -> bool {
        *v <= f32::EPSILON
    }
}

// ============================================================================
// REACHABILITY (boolean OR/AND)
// ============================================================================

/// Reachability semiring: AND for edges, OR for accumulation.
///
/// - Crossing an edge: reachable if input is reachable AND edge is non-trivial
/// - Multiple paths: reachable if ANY path reaches
/// - Zero is `false` (unreachable)
#[derive(Clone)]
pub struct Reachability;

impl SpoSemiring for Reachability {
    type Value = bool;

    #[inline]
    fn zero(&self) -> bool {
        false
    }

    #[inline]
    fn multiply(&self, edge_fp: &Container, input: &bool) -> bool {
        *input && !edge_fp.is_zero()
    }

    #[inline]
    fn add(&self, a: &bool, b: &bool) -> bool {
        *a || *b
    }

    #[inline]
    fn is_zero(&self, v: &bool) -> bool {
        !*v
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xor_bundle_identity() {
        let s = XorBundle;
        let z = s.zero();
        assert!(s.is_zero(&z));

        let a = Container::random(42);
        let result = s.add(&z, &a);
        assert_eq!(result, a);
    }

    #[test]
    fn test_xor_bundle_multiply() {
        let s = XorBundle;
        let edge = Container::random(1);
        let input = Container::random(2);
        let output = s.multiply(&edge, &input);
        // XOR is self-inverse: XOR again should recover input
        let recovered = s.multiply(&edge, &output);
        assert_eq!(recovered, input);
    }

    #[test]
    fn test_hamming_min_identity() {
        let s = HammingMin;
        assert!(s.is_zero(&s.zero()));
        assert_eq!(s.add(&s.zero(), &100), 100);
        assert_eq!(s.add(&50, &100), 50);
    }

    #[test]
    fn test_hamming_min_multiply() {
        let s = HammingMin;
        let edge = Container::random(42);
        let cost = s.multiply(&edge, &0);
        assert!(cost > 0); // random container has non-zero popcount
        assert_eq!(cost, edge.popcount());
    }

    #[test]
    fn test_similarity_max_identity() {
        let s = SimilarityMax;
        assert!(s.is_zero(&s.zero()));
        assert_eq!(s.add(&0.5, &0.8), 0.8);
    }

    #[test]
    fn test_reachability_logic() {
        let s = Reachability;
        assert!(s.is_zero(&s.zero()));

        let edge = Container::random(1);
        assert!(s.multiply(&edge, &true));
        assert!(!s.multiply(&edge, &false));
        assert!(!s.multiply(&Container::zero(), &true));

        assert!(s.add(&true, &false));
        assert!(!s.add(&false, &false));
    }
}
