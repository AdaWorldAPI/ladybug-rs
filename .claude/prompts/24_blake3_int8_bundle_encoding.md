# 24_BLAKE3_INT8_BUNDLE_ENCODING.md

## Text → Binary → Bundle → Think. No Model. No Float. No GPU. Ever.

**Target:** hexagon/crates/hexagon-core/, hexagon/crates/hexagon-cam/
**Replaces:** 5 separate text→fingerprint encoders with ONE path

---

## The Pipeline

```
Text "Alice loves Bob"
  ↓
Tokenize: ["Alice", "loves", "Bob"]
  ↓
Per token: BLAKE3(token) → 16384 deterministic bits
  ↓
Bundle into i8 accumulator: [i8; 16384]
  per bit k: input=1 → acc[k]+=1, input=0 → acc[k]-=1
  ↓
Threshold: |acc[k]| > f(N) ?
  sign → data bit
  magnitude → alpha bit
  ↓
Output: (data: [u64; 256], alpha: [u64; 256]) = 4 KB per concept
```

## The i8 Accumulator

```rust
struct Accumulator {
    weights: [i8; 16384],  // 16 KB — fits L1 cache
    count: u32,            // how many inputs bundled
}

impl Accumulator {
    fn bundle_token(&mut self, token: &str) {
        let hash = blake3::hash(token.as_bytes());
        let bits = expand_hash_to_bits(hash);  // BLAKE3 → LFSR → 16384 bits
        for k in 0..16384 {
            if bits[k] {
                self.weights[k] = self.weights[k].saturating_add(1);
            } else {
                self.weights[k] = self.weights[k].saturating_sub(1);
            }
        }
        self.count += 1;
    }

    fn extract(&self) -> (Data, Alpha) {
        let threshold = self.alpha_threshold();
        let mut data = [0u64; 256];
        let mut alpha = [0u64; 256];

        for k in 0..16384 {
            let word = k / 64;
            let bit = k % 64;
            
            // Sign → data
            if self.weights[k] > 0 {
                data[word] |= 1 << bit;
            }
            
            // Magnitude → alpha
            if self.weights[k].unsigned_abs() > threshold {
                alpha[word] |= 1 << bit;
            }
        }
        (Data(data), Alpha(alpha))
    }

    fn alpha_threshold(&self) -> u8 {
        // Scale with bundle count
        // N=1: threshold=0 (single input, everything defined)
        // N=5: threshold=2 (need 70%+ agreement)
        // N=10: threshold=4 (need 70%+ agreement)
        // N=100: threshold=30 (need 65%+ agreement)
        match self.count {
            0..=1 => 0,
            2..=5 => (self.count as u8) / 2,
            6..=20 => (self.count as u8) * 2 / 5,
            _ => (self.count as f32 * 0.3) as u8,
        }
    }
}
```

## Why i8

```
i8 range: -128 to +127
Saturating arithmetic: never overflows, clamps at bounds
16384 × 1 byte = 16 KB accumulator → fits L1 cache entirely
Compare: f32 accumulator = 64 KB → spills to L2 (4x slower)
Compare: i16 accumulator = 32 KB → borderline L1

i8 handles up to ~127 bundles before saturation.
That covers: sentences (5-20 words), paragraphs (50-100 words),
concepts accumulated over many encounters.

For corpus-scale accumulation (1000+ encounters):
  Option A: use i16 (32 KB, still L1/L2 boundary)
  Option B: periodically extract → rebundle at half magnitude
  Option C: accept saturation as "maximum confidence"
  Recommend B for production, C for simplicity.
```

## Semantic Quality Without Embeddings

```
"The king rules the kingdom"
  → bundle(BLAKE3("the"), BLAKE3("king"), BLAKE3("rules"), 
           BLAKE3("the"), BLAKE3("kingdom"))

"The queen rules the kingdom"
  → bundle(BLAKE3("the"), BLAKE3("queen"), BLAKE3("rules"),
           BLAKE3("the"), BLAKE3("kingdom"))

Shared: "the"(×2), "rules", "kingdom" = 4/5 terms identical
Different: "king" vs "queen" = 1/5 terms

Bundle Hamming distance: ~10-15% (very similar!)
The CONTEXT overlap makes the BUNDLES similar.

"You shall know a word by the company it keeps" — Firth 1957
Implemented as majority voting on BLAKE3 hashes. No training.
```

## SPO Encoding With i8 Bundles

```
Triple: "Alice loves Bob" in context "Alice has always loved Bob deeply"

S_acc: i8 accumulator
  bundle("Alice", "has", "always")  // subject + context
  extract → (s_data, s_alpha)

P_acc: i8 accumulator
  bundle("loves", "loved", "always", "deeply")  // predicate + context
  extract → (p_data, p_alpha)

O_acc: i8 accumulator
  bundle("Bob", "deeply")  // object + context
  extract → (o_data, o_alpha)

Bind:
  triple_data  = s_data XOR ROLE_S XOR p_data XOR ROLE_P XOR o_data XOR ROLE_O
  triple_alpha = s_alpha AND p_alpha AND o_alpha

Store: 4 KB (2 KB data + 2 KB alpha)
```

## Partial Triple Encoding

```
"Alice loves ???" (unknown object)

S_acc: bundle("Alice", context...) → (s_data, s_alpha) — fully defined
P_acc: bundle("loves", context...) → (p_data, p_alpha) — fully defined
O_acc: empty → (zeros, zeros) — alpha = all zero

SP_ projection:
  data  = s_data XOR ROLE_S XOR p_data XOR ROLE_P
  alpha = s_alpha AND p_alpha = fully defined

Full SPO attempt:
  alpha = s_alpha AND p_alpha AND o_alpha = s_alpha AND p_alpha AND 0...0 = 0...0
  → Belichtungsmesser reads zero → early exit → never stored as complete triple

The alpha channel STRUCTURALLY PREVENTS storing unknowns as knowledge.
```

## What This Eliminates

```
ELIMINATED:                              REPLACED BY:
  core/fingerprint.rs from_content()     BLAKE3 → i8 bundle
  spo/nsm_substrate.rs encode()          BLAKE3 → i8 bundle
  spo/codebook_training.rs encode()      BLAKE3 → i8 bundle
  spo/deepnsm_integration.rs encode()    BLAKE3 → i8 bundle (training optional)
  spo/crystal_lm.rs encode_clean()       BLAKE3 → i8 bundle
  spo/jina_api.rs                        Not needed
  spo/jina_cache.rs                      Not needed
  Any Jina API dependency                Not needed
  Any float embedding anywhere           Not needed
  Any GPU for inference                  Not needed

KEPT AS OPTIONAL ENHANCEMENT:
  NSM prime decomposition as preprocessing:
    text → NSM primes → BLAKE3(prime) per prime → i8 bundle
    Better semantics. Same pipeline. Optional.
    
  Codebook for O(1) known-concept lookup:
    If concept already in codebook → return cached fingerprint
    If not → encode via pipeline above, add to codebook
    Codebook is ACCELERATION, not REQUIREMENT.
```

## Implementation Location

```
hexagon/crates/hexagon-core/src/
  encode.rs (NEW, ~200 lines)
    Accumulator struct
    bundle_token()
    bundle_tokens()  — convenience for word list
    extract() → (Data, Alpha)
    alpha_threshold()
    expand_hash_to_bits() — BLAKE3 output → 16384 bits via LFSR

  Depends on: blake3 crate only. Zero other deps.
  Fits in: hexagon-core (this IS a core instruction: ENCODE)

hexagon/crates/hexagon-cam/src/
  codebook.rs — OPTIONAL known-concept cache
    Uses encode.rs for new concepts
    Returns cached (data, alpha) for known concepts
```

## The Encoding IS the Seventh Instruction

```
XOR.α         bind / unbind
POPCOUNT.α    distance / similarity
MAJORITY.α    bundle / superpose
AND.α/NOT.α   2^3 factorization
BLAKE3.α      seal / verify
THRESHOLD.α   σ-band gating
ENCODE.α      text → (data, alpha) via i8 bundle  ← THIS

Still 6 RISC instructions for COMPUTATION.
ENCODE is INPUT — how data enters the substrate.
Not part of the compute loop. Part of the ingestion path.
```

---

*"BLAKE3 gives you bits. Bundling gives you meaning. Alpha gives you honesty. No model needed."*
