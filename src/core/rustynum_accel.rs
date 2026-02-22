//! RustyNum-accelerated operations for Fingerprint and Container.
//!
//! Provides runtime-dispatched AVX-512 kernels when the `rustynum` feature
//! is enabled. Key advantages over ladybug's built-in SIMD:
//!
//! - **Runtime dispatch** (`is_x86_feature_detected!`) — same binary works
//!   on any x86_64 CPU vs compile-time `#[cfg(target_feature)]`
//! - **VNNI int8 dot product** — new capability for embedding containers
//! - **Optimized bundle** — ripple-carry > per-bit counting for large n
//! - **Zero-copy bridge** — `view_u64_as_bytes` reinterprets `[u64; N]` as `&[u8]`
//!   without allocation
//!
//! ## Usage
//!
//! ```rust,ignore
//! use ladybug::core::rustynum_accel::*;
//!
//! let fp = Fingerprint::from_content("hello");
//! let pc = fingerprint_popcount(&fp);   // VPOPCNTDQ when available
//! let hd = fingerprint_hamming(&fp, &fp); // VPOPCNTDQ when available
//! ```

use crate::core::Fingerprint;
use crate::FINGERPRINT_U64;
use ladybug_contract::Container;
use ladybug_contract::container::CONTAINER_WORDS;
use rustynum_rs::simd_ops::avx512::{
    dot_i8i8_dispatch, dot_u8i8_dispatch, hamming_dispatch, popcount_dispatch,
    view_u64_as_bytes,
};

// ────────────────────────────────────────────────────────────────
// Fingerprint operations (16384-bit = 2048 bytes = 256 u64 words)
// ────────────────────────────────────────────────────────────────

/// Popcount a Fingerprint using runtime-dispatched VPOPCNTDQ.
///
/// On AVX-512 VPOPCNTDQ hardware: 32 instructions (vs 256 scalar POPCNT).
#[inline]
pub fn fingerprint_popcount(fp: &Fingerprint) -> u32 {
    popcount_dispatch(view_u64_as_bytes(fp.as_raw())) as u32
}

/// Hamming distance between two Fingerprints using runtime-dispatched VPOPCNTDQ.
///
/// On AVX-512 VPOPCNTDQ hardware: 32 XOR + 32 VPOPCNTDQ = 64 instructions.
#[inline]
pub fn fingerprint_hamming(a: &Fingerprint, b: &Fingerprint) -> u32 {
    hamming_dispatch(
        view_u64_as_bytes(a.as_raw()),
        view_u64_as_bytes(b.as_raw()),
    ) as u32
}

/// Similarity (0.0–1.0) between two Fingerprints.
#[inline]
pub fn fingerprint_similarity(a: &Fingerprint, b: &Fingerprint) -> f32 {
    1.0 - (fingerprint_hamming(a, b) as f32 / (FINGERPRINT_U64 * 64) as f32)
}

/// Signed i8 × i8 dot product on Fingerprint data (both interpreted as i8).
///
/// Uses AVX-512 VNNI (VPDPBUSD) when available: 32 instructions for 2048 bytes.
/// Useful for embedding similarity in CogRecord Container 3.
#[inline]
pub fn fingerprint_dot_i8(a: &Fingerprint, b: &Fingerprint) -> i64 {
    dot_i8i8_dispatch(
        view_u64_as_bytes(a.as_raw()),
        view_u64_as_bytes(b.as_raw()),
    )
}

/// Unsigned u8 × signed i8 dot product on Fingerprint data.
///
/// Native VPDPBUSD semantics: `a` unsigned, `b` signed.
/// Standard format for quantized embeddings (activations × weights).
#[inline]
pub fn fingerprint_dot_u8i8(a: &Fingerprint, b: &Fingerprint) -> i64 {
    dot_u8i8_dispatch(
        view_u64_as_bytes(a.as_raw()),
        view_u64_as_bytes(b.as_raw()),
    )
}

// ────────────────────────────────────────────────────────────────
// Container operations (8192-bit = 1024 bytes = 128 u64 words)
// ────────────────────────────────────────────────────────────────

/// Popcount a Container using runtime-dispatched VPOPCNTDQ.
///
/// On AVX-512 VPOPCNTDQ hardware: 16 instructions (vs 128 scalar POPCNT).
#[inline]
pub fn container_popcount(c: &Container) -> u32 {
    popcount_dispatch(view_u64_as_bytes(&c.words)) as u32
}

/// Hamming distance between two Containers using runtime-dispatched VPOPCNTDQ.
#[inline]
pub fn container_hamming(a: &Container, b: &Container) -> u32 {
    hamming_dispatch(
        view_u64_as_bytes(&a.words),
        view_u64_as_bytes(&b.words),
    ) as u32
}

/// Container similarity (0.0–1.0).
#[inline]
pub fn container_similarity(a: &Container, b: &Container) -> f32 {
    1.0 - (container_hamming(a, b) as f32 / (CONTAINER_WORDS * 64) as f32)
}

/// Signed i8 × i8 dot product on Container data.
///
/// Interprets the container bytes as signed int8 values.
/// For embedding containers (CogRecord Container 3).
#[inline]
pub fn container_dot_i8(a: &Container, b: &Container) -> i64 {
    dot_i8i8_dispatch(
        view_u64_as_bytes(&a.words),
        view_u64_as_bytes(&b.words),
    )
}

/// Unsigned u8 × signed i8 dot product on Container data.
#[inline]
pub fn container_dot_u8i8(a: &Container, b: &Container) -> i64 {
    dot_u8i8_dispatch(
        view_u64_as_bytes(&a.words),
        view_u64_as_bytes(&b.words),
    )
}

/// Bundle multiple Containers using rustynum's optimized ripple-carry algorithm.
///
/// For large n (> 16 vectors): ripple-carry with u64x8 SIMD + thread parallelism.
/// Much faster than ladybug-contract's per-bit counting.
pub fn container_bundle(items: &[&Container]) -> Container {
    if items.is_empty() {
        return Container::zero();
    }
    if items.len() == 1 {
        return items[0].clone();
    }

    // Convert Containers to NumArrayU8 references for bundle
    let arrays: Vec<rustynum_rs::NumArrayU8> = items
        .iter()
        .map(|c| rustynum_rs::NumArrayU8::new(view_u64_as_bytes(&c.words).to_vec()))
        .collect();
    let refs: Vec<&rustynum_rs::NumArrayU8> = arrays.iter().collect();

    let result = rustynum_rs::NumArrayU8::bundle(&refs);
    let result_bytes = result.get_data();

    // Convert back to Container
    let mut container = Container::zero();
    let word_bytes = view_u64_as_bytes(&container.words).len();
    debug_assert_eq!(result_bytes.len(), word_bytes);

    // Write bytes back into container words
    for (i, chunk) in result_bytes.chunks_exact(8).enumerate() {
        container.words[i] = u64::from_ne_bytes(chunk.try_into().unwrap());
    }

    container
}

// ────────────────────────────────────────────────────────────────
// Raw slice operations (for arbitrary-length byte arrays)
// ────────────────────────────────────────────────────────────────

/// Popcount on any `&[u64]` slice (zero-copy). Works for any container size.
#[inline]
pub fn slice_popcount(data: &[u64]) -> u64 {
    popcount_dispatch(view_u64_as_bytes(data))
}

/// Hamming distance on any two `&[u64]` slices (zero-copy).
#[inline]
pub fn slice_hamming(a: &[u64], b: &[u64]) -> u64 {
    debug_assert_eq!(a.len(), b.len());
    hamming_dispatch(view_u64_as_bytes(a), view_u64_as_bytes(b))
}

/// Signed i8 × i8 dot product on any two `&[u64]` slices.
#[inline]
pub fn slice_dot_i8(a: &[u64], b: &[u64]) -> i64 {
    debug_assert_eq!(a.len(), b.len());
    dot_i8i8_dispatch(view_u64_as_bytes(a), view_u64_as_bytes(b))
}

// ────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fingerprint_popcount_zero() {
        let fp = Fingerprint::zero();
        assert_eq!(fingerprint_popcount(&fp), 0);
    }

    #[test]
    fn test_fingerprint_popcount_ones() {
        let fp = Fingerprint::ones();
        assert_eq!(fingerprint_popcount(&fp), (FINGERPRINT_U64 * 64) as u32);
    }

    #[test]
    fn test_fingerprint_popcount_matches_scalar() {
        let fp = Fingerprint::from_content("test popcount");
        let accel = fingerprint_popcount(&fp);
        let scalar = fp.popcount();
        assert_eq!(accel, scalar);
    }

    #[test]
    fn test_fingerprint_hamming_identical() {
        let fp = Fingerprint::from_content("identical");
        assert_eq!(fingerprint_hamming(&fp, &fp), 0);
    }

    #[test]
    fn test_fingerprint_hamming_matches_scalar() {
        let a = Fingerprint::from_content("alpha");
        let b = Fingerprint::from_content("beta");
        let accel = fingerprint_hamming(&a, &b);
        let scalar = a.hamming(&b);
        assert_eq!(accel, scalar);
    }

    #[test]
    fn test_fingerprint_dot_i8_self() {
        let fp = Fingerprint::from_content("dot self");
        let dot = fingerprint_dot_i8(&fp, &fp);
        // Self-dot should always be non-negative (sum of squares)
        assert!(dot >= 0, "Self dot product should be >= 0, got {}", dot);
    }

    #[test]
    fn test_container_popcount_zero() {
        let c = Container::zero();
        assert_eq!(container_popcount(&c), 0);
    }

    #[test]
    fn test_container_popcount_ones() {
        let c = Container::ones();
        assert_eq!(container_popcount(&c), (CONTAINER_WORDS * 64) as u32);
    }

    #[test]
    fn test_container_popcount_matches_scalar() {
        let c = Container::random(42);
        let accel = container_popcount(&c);
        let scalar = c.popcount();
        assert_eq!(accel, scalar);
    }

    #[test]
    fn test_container_hamming_identical() {
        let c = Container::random(42);
        assert_eq!(container_hamming(&c, &c), 0);
    }

    #[test]
    fn test_container_hamming_matches_scalar() {
        let a = Container::random(1);
        let b = Container::random(2);
        let accel = container_hamming(&a, &b);
        let scalar = a.hamming(&b);
        assert_eq!(accel, scalar);
    }

    #[test]
    fn test_container_bundle_unanimous() {
        let a = Container::random(100);
        let result = container_bundle(&[&a, &a, &a]);
        assert_eq!(result, a);
    }

    #[test]
    fn test_container_bundle_majority() {
        let a = Container::ones();
        let b = Container::ones();
        let c = Container::zero();
        let result = container_bundle(&[&a, &b, &c]);
        assert_eq!(result, Container::ones());
    }

    #[test]
    fn test_container_bundle_matches_contract() {
        let a = Container::random(10);
        let b = Container::random(20);
        let c = Container::random(30);
        let accel = container_bundle(&[&a, &b, &c]);
        let contract = Container::bundle(&[&a, &b, &c]);
        assert_eq!(accel, contract);
    }

    #[test]
    fn test_slice_popcount() {
        let data: [u64; 4] = [u64::MAX; 4];
        assert_eq!(slice_popcount(&data), 256);
    }

    #[test]
    fn test_slice_hamming() {
        let a: [u64; 4] = [u64::MAX; 4];
        let b: [u64; 4] = [0; 4];
        assert_eq!(slice_hamming(&a, &b), 256);
    }
}
