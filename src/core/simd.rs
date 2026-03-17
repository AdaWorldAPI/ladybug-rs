//! SIMD operation re-exports and scalar reference implementations.
//!
//! Provides both the hardware-accelerated `hamming_distance` (via rustynum)
//! and a pure-scalar `hamming_scalar` for equivalence testing.

use crate::core::Fingerprint;
use crate::FINGERPRINT_U64;

/// SIMD-accelerated Hamming distance between two Fingerprints.
///
/// Delegates to `rustynum_core::simd::hamming_distance` which uses
/// runtime-dispatched AVX-512 VPOPCNTDQ when available.
#[inline]
pub fn hamming_distance(a: &Fingerprint, b: &Fingerprint) -> u32 {
    super::rustynum_accel::hamming_distance(a, b)
}

/// Pure-scalar Hamming distance (reference implementation).
///
/// Uses `u64::count_ones()` — no SIMD intrinsics. Useful for
/// correctness testing against the SIMD-accelerated path.
#[inline]
pub fn hamming_scalar(a: &Fingerprint, b: &Fingerprint) -> u32 {
    let mut dist = 0u32;
    let ra = a.as_raw();
    let rb = b.as_raw();
    for i in 0..FINGERPRINT_U64 {
        dist += (ra[i] ^ rb[i]).count_ones();
    }
    dist
}
