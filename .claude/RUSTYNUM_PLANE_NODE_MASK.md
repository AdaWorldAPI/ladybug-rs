# RUSTYNUM_PLANE_NODE_MASK.md

## Implement Plane/Node/Mask in rustynum-core. The Foundation.

**Repo:** rustynum (WRITE)
**Read:** ladybug-rs CLAUDE.md + .claude/prompts/25_node_plane_mask.md (reference only)
**Also read:** rustynum-core/src/simd.rs, rustynum-core/src/fingerprint.rs

---

## THE PROBLEM WITH WHAT EXISTS

```rust
// fingerprint.rs: SCALAR LOOPS. Doesn't use simd.rs at all.
pub fn hamming_distance(&self, other: &Self) -> u32 {
    for i in 0..N { dist += (self.words[i] ^ other.words[i]).count_ones(); }
    // This is 256 scalar popcount ops for 16K. 
    // simd.rs has AVX-512 VPOPCNTDQ that does it in 32 ops.
    // They're in the SAME CRATE and don't talk to each other.
}

// Container (in ladybug-contract): FIXED at 16384 bits. No const generic.
// Fingerprint<N> (in rustynum-core): const generic but SCALAR.
// simd.rs: WORKS ON &[u8] SLICES. Any width. Already tiered.
// Nobody connected them.
```

---

## WHAT TO BUILD

### 1. Plane<const N: usize> — The Core Type

```rust
// rustynum-core/src/plane.rs

use crate::simd;  // THE SIMD ARSENAL. USE IT.

/// One dimension of cognition. N × 64 bits of signal.
///
/// The i8 accumulator IS the ground truth. Everything else is derived.
/// bits = sign(acc). alpha = |acc| > threshold. truth = alpha density.
///
/// NaN is structurally impossible: i8 saturating arithmetic, no floats.
/// Width mismatch is handled gracefully: operations on mismatched
/// planes use the SHORTER width and set alpha=0 on the remainder.
///
/// Standard sizes:
///   Plane<256>  = 16,384 bits = 16 KB accumulator (L1 cache resident)
///   Plane<128>  =  8,192 bits =  8 KB accumulator
///   Plane<512>  = 32,768 bits = 32 KB accumulator
///   Plane<1024> = 65,536 bits = 64 KB accumulator
pub struct Plane<const N: usize> {
    /// The ONLY stored state. Everything else is derived.
    /// i8 per bit position. Sign = data. Magnitude = confidence.
    acc: [i8; { N * 64 }],
    /// How many encounters shaped this plane.
    encounters: u32,
}
```

**WAIT — `[i8; { N * 64 }]` requires nightly `generic_const_exprs`.**

Alternative that works on stable Rust:

```rust
/// Plane stores accumulator as bytes, sized to match Fingerprint<N>.
/// N = number of u64 words. Total bits = N × 64. Total i8 slots = N × 64.
///
/// We store acc as [u64; N] pairs: acc_pos and acc_neg.
/// Each bit position k has:
///   positive evidence = popcount of acc_pos at bit k across encounters
///   negative evidence = popcount of acc_neg at bit k across encounters
///
/// BUT SIMPLER: store acc as a Vec<i8> of length N*64 on heap,
/// or store as Box<[i8]>. The Plane struct stays on stack with a pointer.
///
/// SIMPLEST (and correct for SIMD alignment): 
///   Store sign bits as Fingerprint<N> (the data).
///   Store magnitude as a parallel Fingerprint<N> (the alpha).
///   Store raw accumulator as Box<[i8]> for encounter().
///   Derive bits/alpha from acc on demand, OR cache them.

pub struct Plane<const N: usize> {
    /// Raw i8 accumulator. Heap allocated. SIMD-aligned.
    /// Length: N * 64 (one i8 per bit position).
    acc: Box<AlignedI8<N>>,
    /// Cached data bits. Recomputed from acc when dirty.
    bits: Fingerprint<N>,
    /// Cached alpha mask. Recomputed from acc when dirty.
    alpha: Fingerprint<N>,
    /// Whether bits/alpha need recomputing from acc.
    dirty: bool,
    /// Encounter count.
    encounters: u32,
}

/// SIMD-aligned i8 accumulator. 64-byte alignment for AVX-512.
#[repr(C, align(64))]
pub struct AlignedI8<const N: usize> {
    pub data: [i8; N * 64],  // requires generic_const_exprs OR use a fixed max
}
```

**PRAGMATIC SOLUTION (works on stable, no nightly):**

```rust
/// Plane at standard 16K width. The default. The one that fits L1.
/// Other widths use PlaneWide or are implemented as Plane with different N
/// via the methods that operate on &[u8] slices.
pub struct Plane {
    /// i8 accumulator. 16384 positions. 16 KB. Fits L1 cache.
    /// 64-byte aligned for AVX-512 loads.
    acc: Box<Acc16K>,
    /// Cached data bits derived from sign(acc). 
    bits: Fingerprint<256>,
    /// Cached alpha mask derived from |acc| > threshold.
    alpha: Fingerprint<256>,
    /// Whether cache needs refresh.
    dirty: bool,
    /// Encounter count.
    encounters: u32,
}

#[repr(C, align(64))]
pub struct Acc16K {
    pub values: [i8; 16384],
}

// For other widths, Plane delegates to &[u8] SIMD which handles any length.
// The struct itself is fixed at 16K for L1 cache optimization.
// Wider operations (32K, 64K) use PlaneWide which allocates accordingly.
```

**THE DECISION:** Use fixed 16K as the default Plane, provide `PlaneN<const N: usize>` for other widths that compiles on stable by using `Vec<i8>` instead of `[i8; N*64]`. The SIMD layer (`simd.rs`) already operates on `&[u8]` slices of ANY length.

---

### 2. Width Mismatch: Graceful, Not Panic

```rust
impl Plane {
    /// Distance to another Plane. SIMD-accelerated. Alpha-aware.
    /// 
    /// If widths differ (comparing 16K against 32K):
    ///   - Compare on the SHORTER width
    ///   - Treat extra bits in the wider plane as alpha=0 (undefined)
    ///   - Penalty applies to the undefined region
    ///   - Log a warning (tracing::warn!) on first mismatch per pair
    ///   - NEVER panic. NEVER return NaN. Always return a valid Distance.
    pub fn distance(&self, other: &Plane) -> Distance {
        self.distance_slices(
            self.bits_bytes(), self.alpha_bytes(),
            other.bits_bytes(), other.alpha_bytes(),
        )
    }
    
    /// Core distance on raw byte slices. Handles ANY width combination.
    /// This is where SIMD lives.
    fn distance_slices(
        &self,
        a_bits: &[u8], a_alpha: &[u8],
        b_bits: &[u8], b_alpha: &[u8],
    ) -> Distance {
        // Determine shared width
        let shared_len = a_bits.len().min(b_bits.len())
            .min(a_alpha.len()).min(b_alpha.len());
        
        if shared_len == 0 {
            return Distance::Incomparable; // nothing to compare
        }
        
        if a_bits.len() != b_bits.len() {
            // Width mismatch. Compare on shared prefix. Warn once.
            static WARNED: std::sync::atomic::AtomicBool = 
                std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::warn!(
                    "Plane width mismatch: {}b vs {}b. Comparing on shared {}b prefix.",
                    a_bits.len() * 8, b_bits.len() * 8, shared_len * 8
                );
            }
        }
        
        // Compute shared alpha: AND of both alpha channels on shared region
        // Use SIMD — this is just a bulk AND on byte slices
        let hamming_fn = simd::select_hamming_fn();  // resolved ONCE via OnceLock
        
        // shared_alpha = a_alpha[..shared_len] AND b_alpha[..shared_len]
        // XOR of bits WHERE both alpha=1
        // popcount of XOR AND shared_alpha = disagreement
        // popcount of shared_alpha = overlap
        // popcount of NOT a_alpha = penalty
        
        let a = &a_bits[..shared_len];
        let b = &b_bits[..shared_len];
        let aa = &a_alpha[..shared_len];
        let ba = &b_alpha[..shared_len];
        
        // Allocate temp buffers (stack for 16K, heap for larger)
        let mut xor_buf = vec![0u8; shared_len];
        let mut shared_alpha_buf = vec![0u8; shared_len];
        let mut masked_xor_buf = vec![0u8; shared_len];
        let mut not_alpha_buf = vec![0u8; shared_len];
        
        // SIMD bulk operations (LLVM auto-vectorizes these tight loops)
        for i in 0..shared_len {
            xor_buf[i] = a[i] ^ b[i];
            shared_alpha_buf[i] = aa[i] & ba[i];
            masked_xor_buf[i] = xor_buf[i] & shared_alpha_buf[i];
            not_alpha_buf[i] = !aa[i];
        }
        
        // Use the SIMD popcount from simd.rs:
        let disagreement = simd::popcount(&masked_xor_buf) as u32;
        let overlap = simd::popcount(&shared_alpha_buf) as u32;
        let penalty = simd::popcount(&not_alpha_buf) as u32;
        
        // Add penalty for the width mismatch region (wider plane's extra bits)
        let extra_bits = (a_bits.len().max(b_bits.len()) - shared_len) * 8;
        let total_penalty = penalty + extra_bits as u32;
        
        if overlap == 0 {
            return Distance::Incomparable;
        }
        
        Distance::Measured {
            disagreement,
            overlap,
            penalty: total_penalty,
            // Normalized: no float division by zero possible.
            // overlap > 0 guaranteed by check above.
            // Result is u32/u32 ratio, stored as separate fields.
            // The CALLER decides if they want f32 — we don't force it.
        }
    }
}

/// Distance result. No floats. No NaN. Ever.
/// The caller computes a ratio if they need one.
#[derive(Debug, Clone, Copy)]
pub enum Distance {
    /// Enough shared alpha to compare.
    Measured {
        /// Bits that disagree on shared-alpha positions.
        disagreement: u32,
        /// Bits where both planes have alpha=1.
        overlap: u32,
        /// Bits where self has alpha=0 (active penalty).
        penalty: u32,
    },
    /// Not enough shared alpha to compare meaningfully.
    /// This is NOT an error. It's honest: "I can't tell."
    Incomparable,
}

impl Distance {
    /// Normalized distance as f32. ONLY place float appears.
    /// Returns None if Incomparable. Never NaN.
    #[inline]
    pub fn normalized(&self) -> Option<f32> {
        match self {
            Distance::Measured { disagreement, overlap, penalty, .. } => {
                let denom = overlap + penalty;
                if denom == 0 { return None; }  // shouldn't happen but belt+suspenders
                Some((*disagreement + *penalty) as f32 / denom as f32)
            }
            Distance::Incomparable => None,
        }
    }
    
    /// Is this closer than a threshold? Pure integer comparison. No float.
    #[inline]
    pub fn closer_than(&self, max_disagreement: u32) -> bool {
        match self {
            Distance::Measured { disagreement, .. } => *disagreement <= max_disagreement,
            Distance::Incomparable => false,
        }
    }
    
    /// Raw disagreement count. None if incomparable.
    #[inline]
    pub fn raw(&self) -> Option<u32> {
        match self {
            Distance::Measured { disagreement, .. } => Some(*disagreement),
            Distance::Incomparable => None,
        }
    }
}
```

---

### 3. NaN Impossibility — By Construction

```rust
/// NARS truth from accumulator state. Integer arithmetic only.
/// Frequency and confidence are u16 scaled to [0, 65535] = [0.0, 1.0].
/// No float. No NaN. No division by zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Truth {
    /// Frequency: positive defined bits / total defined bits.
    /// Scaled: 0 = never true, 65535 = always true.
    pub frequency: u16,
    /// Confidence: defined bits / total bits.
    /// Scaled: 0 = no evidence, 65535 = fully defined.
    pub confidence: u16,
    /// Raw encounter count.
    pub evidence: u32,
}

impl Plane {
    /// NARS truth derived from accumulator state.
    /// Pure integer. Impossible to produce NaN.
    pub fn truth(&self) -> Truth {
        self.ensure_cache();
        
        let total_bits = Self::BITS as u32;  // 16384 for default Plane
        let defined = simd::popcount(self.alpha.as_bytes()) as u32;
        let positive = {
            // bits AND alpha → positive defined bits
            let mut buf = vec![0u8; Self::BYTES];
            for i in 0..Self::BYTES {
                buf[i] = self.bits.as_bytes()[i] & self.alpha.as_bytes()[i];
            }
            simd::popcount(&buf) as u32
        };
        
        // Integer-only scaling. No division by zero.
        let frequency = if defined == 0 { 
            32768u16  // no evidence → 0.5 (maximum uncertainty)
        } else {
            ((positive as u64 * 65535) / defined as u64) as u16
        };
        
        let confidence = if total_bits == 0 {
            0u16  // empty plane → zero confidence
        } else {
            ((defined as u64 * 65535) / total_bits as u64) as u16
        };
        
        Truth {
            frequency,
            confidence,
            evidence: self.encounters,
        }
    }
}

impl Truth {
    /// Frequency as f32 [0.0, 1.0]. The ONLY float conversion point.
    /// Guaranteed not NaN because u16 / 65535 is always finite.
    #[inline]
    pub fn frequency_f32(&self) -> f32 { self.frequency as f32 / 65535.0 }
    
    /// Confidence as f32 [0.0, 1.0].
    #[inline]
    pub fn confidence_f32(&self) -> f32 { self.confidence as f32 / 65535.0 }
    
    /// Expectation: c * (f - 0.5) + 0.5. Integer version.
    /// Returns u16 scaled [0, 65535].
    #[inline]
    pub fn expectation(&self) -> u16 {
        // All integer: (c * (f - 32768) / 65535) + 32768
        let f = self.frequency as i32;
        let c = self.confidence as i32;
        let centered = f - 32768;  // f - 0.5 scaled
        let weighted = (c * centered) / 65535;  // c * (f - 0.5)
        (weighted + 32768).clamp(0, 65535) as u16
    }
    
    /// Revision: combine two independent evidence sources.
    /// Integer only. No float.
    pub fn revision(&self, other: &Truth) -> Truth {
        let total_evidence = self.evidence.saturating_add(other.evidence);
        if total_evidence == 0 {
            return Truth { frequency: 32768, confidence: 0, evidence: 0 };
        }
        
        // Weighted average by evidence count. Integer arithmetic.
        let f = ((self.frequency as u64 * self.evidence as u64)
               + (other.frequency as u64 * other.evidence as u64))
               / total_evidence as u64;
        let c = ((self.confidence as u64 * self.evidence as u64)
               + (other.confidence as u64 * other.evidence as u64))
               / total_evidence as u64;
        
        Truth {
            frequency: f.min(65535) as u16,
            confidence: c.min(65535) as u16,
            evidence: total_evidence,
        }
    }
}
```

---

### 4. Encounter — Wire to SIMD

```rust
impl Plane {
    /// Evidence arrives. BLAKE3 → expand → i8 saturating ±1.
    /// Uses SIMD for the bulk accumulation.
    pub fn encounter(&mut self, text: &str) {
        let hash = blake3::hash(text.as_bytes());
        let bits = Self::blake3_expand(hash.as_bytes());
        
        // Bulk i8 accumulation: acc[k] += (bit ? +1 : -1), saturating
        // This is a dot_i8 variant — add/sub per byte
        // SIMD: process 64 bytes per AVX-512 iteration
        let acc_bytes = self.acc_as_bytes_mut();
        let bit_bytes = bits.as_bytes();
        
        // SIMD-friendly tight loop (LLVM auto-vectorizes with -C target-cpu=native)
        for k in 0..Self::BITS {
            let byte_idx = k / 8;
            let bit_idx = k % 8;
            let is_set = (bit_bytes[byte_idx] >> bit_idx) & 1 == 1;
            if is_set {
                acc_bytes[k] = acc_bytes[k].saturating_add(1);
            } else {
                acc_bytes[k] = acc_bytes[k].saturating_sub(1);
            }
        }
        
        self.encounters += 1;
        self.dirty = true;  // bits/alpha cache invalidated
    }
    
    /// Recompute cached bits and alpha from accumulator.
    fn ensure_cache(&mut self) {
        if !self.dirty { return; }
        
        let threshold = self.alpha_threshold();
        let acc = &self.acc.values;
        
        for k in 0..Self::BITS {
            let word = k / 64;
            let bit = k % 64;
            
            // Data: sign of accumulator
            if acc[k] > 0 {
                self.bits.words[word] |= 1u64 << bit;
            } else {
                self.bits.words[word] &= !(1u64 << bit);
            }
            
            // Alpha: magnitude above threshold
            if acc[k].unsigned_abs() > threshold {
                self.alpha.words[word] |= 1u64 << bit;
            } else {
                self.alpha.words[word] &= !(1u64 << bit);
            }
        }
        
        self.dirty = false;
    }
    
    fn alpha_threshold(&self) -> u8 {
        match self.encounters {
            0..=1 => 0,
            2..=5 => (self.encounters as u8) / 2,
            6..=20 => (self.encounters as u8) * 2 / 5,
            _ => ((self.encounters as f64).sqrt() * 0.8) as u8,
            // Note: this is the ONE float computation in the entire type.
            // It happens once per ensure_cache, not per comparison.
            // Could be replaced with integer_sqrt * lookup_table.
        }
    }
}
```

---

### 5. Node + Mask — Compose Planes

```rust
/// The cognitive atom. Three planes. Separately addressable.
pub struct Node {
    pub s: Plane,
    pub p: Plane,
    pub o: Plane,
}

/// Attention + borrow boundary + query scope + merge scope.
/// One type. Four meanings. Eight values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mask {
    pub s: bool,
    pub p: bool,
    pub o: bool,
}

pub const SPO: Mask = Mask { s: true,  p: true,  o: true  };
pub const SP_: Mask = Mask { s: true,  p: true,  o: false };
pub const S_O: Mask = Mask { s: true,  p: false, o: true  };
pub const _PO: Mask = Mask { s: false, p: true,  o: true  };
pub const S__: Mask = Mask { s: true,  p: false, o: false };
pub const _P_: Mask = Mask { s: false, p: true,  o: false };
pub const __O: Mask = Mask { s: false, p: false, o: true  };
pub const ___: Mask = Mask { s: false, p: false, o: false };

impl Node {
    /// Distance to another node. Only masked planes participate.
    /// Unmasked planes = zero cost, zero contribution, zero noise.
    /// NaN impossible: returns Distance enum, not float.
    pub fn distance(&self, other: &Node, mask: Mask) -> Distance {
        let mut total_disagreement = 0u32;
        let mut total_overlap = 0u32;
        let mut total_penalty = 0u32;
        let mut any_measured = false;
        
        macro_rules! add_plane {
            ($plane_self:expr, $plane_other:expr, $active:expr) => {
                if $active {
                    match $plane_self.distance(&$plane_other) {
                        Distance::Measured { disagreement, overlap, penalty } => {
                            total_disagreement += disagreement;
                            total_overlap += overlap;
                            total_penalty += penalty;
                            any_measured = true;
                        }
                        Distance::Incomparable => {}
                    }
                }
            }
        }
        
        add_plane!(self.s, other.s, mask.s);
        add_plane!(self.p, other.p, mask.p);
        add_plane!(self.o, other.o, mask.o);
        
        if !any_measured || total_overlap == 0 {
            Distance::Incomparable
        } else {
            Distance::Measured {
                disagreement: total_disagreement,
                overlap: total_overlap,
                penalty: total_penalty,
            }
        }
    }
    
    /// Combined truth across masked planes.
    pub fn truth(&self, mask: Mask) -> Truth {
        let mut total_freq = 0u64;
        let mut total_conf = 0u64;
        let mut total_evidence = 0u32;
        let mut count = 0u32;
        
        if mask.s { 
            let t = self.s.truth(); 
            total_freq += t.frequency as u64;
            total_conf += t.confidence as u64;
            total_evidence += t.evidence;
            count += 1;
        }
        if mask.p {
            let t = self.p.truth();
            total_freq += t.frequency as u64;
            total_conf += t.confidence as u64;
            total_evidence += t.evidence;
            count += 1;
        }
        if mask.o {
            let t = self.o.truth();
            total_freq += t.frequency as u64;
            total_conf += t.confidence as u64;
            total_evidence += t.evidence;
            count += 1;
        }
        
        if count == 0 {
            return Truth { frequency: 32768, confidence: 0, evidence: 0 };
        }
        
        Truth {
            frequency: (total_freq / count as u64) as u16,
            confidence: (total_conf / count as u64) as u16,
            evidence: total_evidence,
        }
    }
}
```

---

### 6. Wire Fingerprint<N> to SIMD (fix existing gap)

While building Plane, also fix `Fingerprint<N>` to actually use simd.rs:

```rust
// In rustynum-core/src/fingerprint.rs, REPLACE scalar loops:

impl<const N: usize> Fingerprint<N> {
    /// Hamming distance using SIMD dispatch. No more scalar loop.
    #[inline]
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        // Delegate to the SIMD arsenal via byte slices.
        // simd::hamming_distance handles AVX-512 → AVX2 → scalar.
        crate::simd::hamming_distance(self.as_bytes(), other.as_bytes()) as u32
    }
    
    /// Popcount using SIMD dispatch.
    #[inline]
    pub fn popcount(&self) -> u32 {
        crate::simd::popcount(self.as_bytes()) as u32
    }
    
    /// Similarity. Returns Option<f32> to avoid NaN on zero-width.
    /// (Breaking change from previous f64 return. Worth it.)
    #[inline]
    pub fn similarity(&self, other: &Self) -> Option<f32> {
        if Self::BITS == 0 { return None; }
        Some(1.0 - self.hamming_distance(other) as f32 / Self::BITS as f32)
    }
}
```

---

### 7. Container ↔ Fingerprint Bridge

```rust
// In rustynum-core or ladybug-contract:

impl From<&Container> for Fingerprint<256> {
    fn from(c: &Container) -> Self {
        Fingerprint::from_words(c.words)
    }
}

impl From<&Fingerprint<256>> for Container {
    fn from(fp: &Fingerprint<256>) -> Self {
        let mut c = Container::zero();
        c.words = fp.words;
        c
    }
}

// For Plane ↔ Container:
impl From<&Plane> for Container {
    fn from(plane: &Plane) -> Self {
        plane.ensure_cache();  // may need &mut self or interior mutability
        Container::from(&plane.bits)
    }
}
```

---

### 8. Seal — Integer Blake3

```rust
/// Integrity seal. blake3 of data masked by alpha.
/// Comparison is byte equality. No float. No NaN.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Seal {
    /// Hash matches stored hash. Stable. Consolidated.
    Wisdom,
    /// Hash differs from stored hash. Changed. Surprising.
    Staunen,
}

/// Truncated blake3 hash for compact storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MerkleRoot([u8; 6]);  // 48 bits. Collision-safe for <10M nodes.

impl Plane {
    /// Compute merkle root from data masked by alpha.
    /// Undefined bits (alpha=0) are zeroed before hashing.
    /// This means noise in undefined regions doesn't change the hash.
    pub fn merkle(&self) -> MerkleRoot {
        self.ensure_cache();
        let mut masked = vec![0u8; Self::BYTES];
        let bits_bytes = self.bits.as_bytes();
        let alpha_bytes = self.alpha.as_bytes();
        for i in 0..Self::BYTES {
            masked[i] = bits_bytes[i] & alpha_bytes[i];
        }
        let hash = blake3::hash(&masked);
        let mut root = [0u8; 6];
        root.copy_from_slice(&hash.as_bytes()[..6]);
        MerkleRoot(root)
    }
    
    /// Verify integrity against a stored root.
    pub fn verify(&self, stored: &MerkleRoot) -> Seal {
        if self.merkle() == *stored {
            Seal::Wisdom
        } else {
            Seal::Staunen
        }
    }
}
```

---

### 9. File Layout in rustynum-core

```
rustynum-core/src/
  lib.rs               ADD: pub mod plane; pub mod node; pub mod seal;
  plane.rs             NEW: Plane, Acc16K, encounter, distance, truth, merkle
  node.rs              NEW: Node, Mask, 8 mask constants, distance, truth
  seal.rs              NEW: Seal, MerkleRoot, verify
  fingerprint.rs       MODIFY: wire hamming/popcount to simd.rs
  simd.rs              UNCHANGED (it already has everything we need)
  
  Total new code: ~600 lines across 3 files
  Modified code: ~20 lines in fingerprint.rs (scalar → SIMD delegation)
```

---

### 10. Tests

```rust
#[test]
fn plane_encounter_builds_signal() {
    let mut p = Plane::new();
    p.encounter("hello");
    p.encounter("hello");
    p.encounter("hello");
    let t = p.truth();
    assert!(t.confidence > 0);  // three encounters → some bits defined
    assert!(t.evidence == 3);
}

#[test]
fn plane_nan_impossible() {
    let empty = Plane::new();
    let t = empty.truth();
    assert_eq!(t.frequency, 32768);  // 0.5 = maximum uncertainty, not NaN
    assert_eq!(t.confidence, 0);     // no evidence, not NaN
    
    let d = empty.distance(&empty);
    assert!(matches!(d, Distance::Incomparable));  // not NaN, not panic
}

#[test]
fn width_mismatch_graceful() {
    // If we support multiple widths:
    // 16K vs 16K → normal comparison
    // 16K vs 32K → compare on 16K prefix, warn, penalty for extra
    // Either way: no panic, no NaN
}

#[test]
fn mask_skips_planes() {
    let a = Node::random(42);
    let b = Node::random(43);
    
    let d_spo = a.distance(&b, SPO);  // all 3 planes
    let d_sp = a.distance(&b, SP_);   // skip O
    let d_s = a.distance(&b, S__);    // skip P and O
    
    // More planes → more overlap → higher potential disagreement
    assert!(d_spo.raw().unwrap() >= d_sp.raw().unwrap());
    assert!(d_sp.raw().unwrap() >= d_s.raw().unwrap());
}

#[test]
fn fingerprint_now_uses_simd() {
    let a = Fingerprint::<256>::ones();
    let b = Fingerprint::<256>::zero();
    assert_eq!(a.hamming_distance(&b), 16384);  // all bits differ
    // This now goes through AVX-512, not scalar loop
}

#[test]
fn seal_stable_despite_noise() {
    let mut p = Plane::new();
    p.encounter("hello");
    p.encounter("hello");
    let root1 = p.merkle();
    
    // Noise in undefined regions doesn't change merkle
    // (because undefined bits are masked out before hashing)
    let root2 = p.merkle();
    assert_eq!(root1, root2);
}

#[test]
fn truth_revision_integer_only() {
    let t1 = Truth { frequency: 60000, confidence: 50000, evidence: 10 };
    let t2 = Truth { frequency: 30000, confidence: 40000, evidence: 5 };
    let revised = t1.revision(&t2);
    
    // Weighted average: (60000*10 + 30000*5) / 15 = 50000
    assert_eq!(revised.frequency, 50000);
    assert_eq!(revised.evidence, 15);
    // No float involved. No NaN possible.
}
```

---

## SUMMARY: What goes wrong if you skip this

```
× Without Plane in rustynum: 31 hand-rolled scalar loops stay. No SIMD. No alpha.
× Without width handling: 64K fingerprint vs 16K → panic in production.
× Without Distance enum: division by zero → NaN → propagates silently.
× Without Truth as u16: every truth comparison needs float → NaN-susceptible.
× Without Fingerprint→SIMD wiring: the 2302-line arsenal sits unused.
× Without Seal: merkle checks need manual blake3 → easy to forget alpha mask.
```

---

*"i8 saturates. u16 scales. u32 counts. Enum distinguishes. Float never touches the hot path."*
*"The type system makes NaN a compilation error, not a runtime bug."*
