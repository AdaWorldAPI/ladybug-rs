# 23_ALPHA_CHANNEL_RISC_MODIFIER.md

## The Seventh Bit: Alpha Channel as RISC Modifier

**Jan Hübener — Ada Architecture — March 2026**
**Target repo:** hexagon (6 RISC instructions)
**Cross-validated:** Orangutan STP/STD dynamics, autaptic penalty circuit

---

## 1. THE PROBLEM

When you encode "Alice loves ???", the unknown Object gets filled with hash noise.
That noise is INDISTINGUISHABLE from real signal in Hamming distance.
Every comparison pays for bits that mean nothing.
The system literally cannot tell "I know this" from "I made this up."

Every vector database has this problem. Nobody solves it at the substrate level.

---

## 2. THE SOLUTION: Alpha Channel

Every vector carries a second vector of the same size: the alpha mask.

```
vector    Binary(2048)    ← data   (16384 bits, the cognitive content)
alpha     Binary(2048)    ← mask   (16384 bits, which bits are DEFINED)

Alpha 1 = this bit carries real signal
Alpha 0 = this bit is undefined / transparent / noise

All 6 RISC instructions gain an alpha modifier.
One extra AND per operation. One cycle.
```

---

## 3. ALPHA PROPAGATION RULES (Corrected)

### BIND (XOR.α) — Both must be defined

```
result_data  = a_data XOR b_data
result_alpha = a_alpha AND b_alpha

WHY AND not OR:
  If Alice has alpha=1 at bit k and ??? has alpha=0 at bit k,
  then Alice XOR ??? at bit k = real_bit XOR noise_bit = NOISE.
  The result is undefined at that position. AND catches this.
  
  OR would launder noise into signal. AND is honest.
```

### UNBIND (XOR.α) — Same as BIND

```
result_data  = bound_data XOR key_data
result_alpha = bound_alpha AND key_alpha

You can only recover signal where both the bound vector
and the unbinding key are defined.
```

### BUNDLE (MAJORITY.α) — Majority of defined voters

```
result_data  = MAJORITY(inputs_data) on alpha=1 positions
result_alpha = MAJORITY(inputs_alpha)

If 3/5 inputs have alpha=1 at bit k: the vote is meaningful → result alpha=1
If 1/5 inputs have alpha=1 at bit k: noise dominates → result alpha=0
Undefined inputs don't vote. They're not abstentions — they're absent.
```

### POPCOUNT.α — Normalized by defined bits

```
distance = popcount((a_data XOR b_data) AND a_alpha AND b_alpha)
overlap  = popcount(a_alpha AND b_alpha)
normalized_distance = distance / overlap

Only count disagreements where BOTH vectors have definitions.
Normalize by the number of bits where comparison is meaningful.

If overlap = 0 → INCOMPARABLE (not similar, not dissimilar — undefined)
```

### AND.α / NOT.α — Preserve or intersect

```
AND:  result_alpha = a_alpha AND b_alpha  (both must be defined)
NOT:  result_alpha = a_alpha              (negation preserves the mask)
```

### BLAKE3.α — Hash only defined bits

```
hash = blake3(data AND alpha)

Undefined bits are zeroed before hashing.
Same content with same alpha = same hash regardless of noise in undefined regions.
This makes MerkleRoot stable even when noise bits change.
```

### THRESHOLD.α — Minimum overlap guard

```
if popcount(a_alpha AND b_alpha) < MIN_OVERLAP:
    return INCOMPARABLE  // not enough shared definition to compare

distance = POPCOUNT.α(a, b)  // normalized by shared alpha
if distance > threshold:
    return REJECT
else:
    return ADMIT
```

---

## 4. ROLE VECTORS AND SPO BINDING

Role vectors (ROLE_S, ROLE_P, ROLE_O, ROLE_Q) are deterministic from BLAKE3.
They are ALWAYS fully defined: `role_alpha = 1...1` (all ones).

This means role binding preserves entity alpha perfectly:

```
S ⊕ ROLE_S:
  result_data  = s_data XOR role_data
  result_alpha = s_alpha AND role_alpha = s_alpha AND 1...1 = s_alpha

The role binding is transparent to the alpha channel.
Only the entity's certainty matters.
```

Full triple binding:

```
(Alice ⊕ ROLE_S) ⊕ (loves ⊕ ROLE_P) ⊕ (Bob ⊕ ROLE_O)
alpha = alice_α AND loves_α AND bob_α

All known entities → alpha = 1...1 AND 1...1 AND 1...1 = all defined
One unknown entity → alpha collapses to 0...0 → Belichtungsmesser reads zero → skip

The alpha channel enforces: you cannot store triples with unknowns
as if they were complete knowledge. The substrate rejects it structurally.
```

---

## 5. PARTIAL KNOWLEDGE ≠ UNCERTAIN KNOWLEDGE

**Critical distinction.** Two completely different epistemic states:

```
PARTIAL KNOWLEDGE:
  SP_ projection: (Alice ⊕ ROLE_S) ⊕ (loves ⊕ ROLE_P)
  alpha = alice_α AND loves_α = 1...1  (FULLY DEFINED)
  
  "I know Alice loves. I haven't encoded who."
  This is a first-class vector with full alpha.
  It encodes LESS information, not UNCERTAIN information.
  It can match confidently against other SP_ projections.

UNCERTAIN KNOWLEDGE:
  SPO vector where the accumulator has weak agreement on O dimensions.
  data bits for O region: set (majority won)
  alpha bits for O region: 0 (agreement was too weak, below threshold)
  
  "I think Alice loves Bob but the evidence is thin."
  The data says Bob. The alpha says "I wouldn't bet on it."
  Comparisons that reach the O region get zero contribution.
```

The 2^3 factorization produces 8 SEPARATE vectors, each with its own alpha:

```
PROJECTION    WHAT IT ENCODES                    ALPHA STATUS
─────────────────────────────────────────────────────────────
SPO           Full triple                        Entity alphas ANDed
SP_           Who does what                      S_α AND P_α (fully defined)
S_O           Who and whom                       S_α AND O_α (fully defined)
_PO           What to whom                       P_α AND O_α (fully defined)
S__           Entity only                        S_α (fully defined)
_P_           Relation only                      P_α (fully defined)
__O           Patient only                       O_α (fully defined)
___           Nothing                            All zero (skip entirely)
```

Each projection is a first-class citizen. SP_ isn't "SPO with missing O."
It's "the concept of Alice-loves, independent of object." Full alpha.
Stored independently. Queryable independently. No noise.

---

## 6. ALPHA ACCUMULATION AS EVIDENCE (STP/STD Dynamics)

The bundle accumulator tracks magnitude per dimension:

```
Accumulator at dimension k after N bundles:
  +5  → strong agreement  → data=1, alpha=1  (confident)
  +1  → weak agreement    → data=1, alpha=0  (set but untrustworthy)
   0  → perfect cancel    → data=?, alpha=0  (contested, no signal)
  -3  → moderate disagree → data=0, alpha=1  (confident negative)

Alpha threshold = f(N):
  As more vectors bundle, threshold increases.
  Need STRONGER consensus for confidence.
  Marginal dimensions LOSE alpha over time.
  Only strong signal survives.
```

This maps to synaptic dynamics:

```
STP (short-term potentiation):
  Recently reinforced dimension. Accumulator magnitude high. Alpha=1.
  The system RECENTLY saw confirming evidence.

STD (short-term depression):
  Contradicted dimension. Accumulator magnitude dropping. Alpha decays.
  Conflicting evidence is ACTIVELY suppressing confidence.

The alpha threshold function IS the temporal decay constant.
Scaled to the system's temporal resolution (access cycles, not milliseconds).
```

### Connection to NARS confidence

```
NARS confidence ≈ popcount(alpha) / total_bits

But now this is STRUCTURAL, not metadata:
  High alpha density → high NARS confidence → derived from accumulation
  Low alpha density → low NARS confidence → derived from accumulation
  
  You don't STORE confidence. You READ IT from the alpha channel.
  Confidence is an OBSERVABLE PROPERTY of the substrate, not a number someone set.
```

---

## 7. ALPHA-AWARE MEXICAN HAT WITH PENALTY CIRCUIT

Standard Mexican hat:

```
response = excite(center) - inhibit(surround)
```

Alpha-aware with Orangutan penalty:

```
excitation = popcount(center_data AND center_alpha)
inhibition = popcount(surround_data AND surround_alpha)
penalty    = popcount(NOT center_alpha)

response = excitation - inhibition - penalty
```

**The penalty is the key innovation.** Every undefined center bit is ACTIVELY penalized.
Not passive silence — active suppression.

```
Vector with 80% alpha → 20% penalty  → mild cost for partial knowledge
Vector with 50% alpha → 50% penalty  → significant suppression
Vector with 20% alpha → 80% penalty  → nearly killed
Vector with  0% alpha → 100% penalty → maximally suppressed, never matches

This pushes undefined vectors OUT of the resonance.
They can't false-match because they're penalized into the noise floor.
Absence of evidence is treated as weak evidence of absence.
```

This is the autaptic inhibitory neuron: it constantly fires (penalty).
Only ACTIVE INPUT (defined alpha bits with excitatory signal) cancels it.
No input = all penalty = suppressed. The system's default state is doubt.

---

## 8. BELICHTUNGSMESSER WITH ALPHA

The 7-point exposure meter reads alpha density FIRST:

```
For each of 7 sample points across the vector:
  alpha_density = popcount(alpha[sample_range]) / sample_size
  
  if alpha_density < MIN_EXPOSURE:
    skip this sample point (not enough light)
  else:
    data_exposure = popcount(data[sample_range] AND alpha[sample_range]) / popcount(alpha[sample_range])
    record exposure reading

If 5/7 sample points skipped → EARLY EXIT entire comparison
  "Not enough light to expose this frame."
  Cost: 7 popcount operations (~7 cycles) instead of full 16384-bit comparison
```

The HDR cascade with alpha early exit:

```
Stage 1: 1-bit Hamming on 1/16 sample, alpha-masked
  shared_alpha = popcount(a_α AND b_α) on sample
  if shared_alpha < MIN_OVERLAP → INCOMPARABLE → early exit
  if hamming_α > σ₃ → REJECT → early exit
  99.7% eliminated (existing) + additional incomparable exits (new)

Stage 2: 1-bit Hamming on 1/4 sample, alpha-masked
Stage 3: 4-bit INT8 full precision, alpha-masked
Stage 4: BF16/FP32 foveal, alpha-masked

Each stage has TWO exit conditions:
  (a) distance too high → not similar → skip          (existing)
  (b) shared alpha too low → not enough overlap → skip (NEW)
```

---

## 9. STORAGE COST

```
BEFORE ALPHA:
  vector    Binary(2048)    per node    2 KB

AFTER ALPHA:
  vector    Binary(2048)    per node    2 KB
  alpha     Binary(2048)    per node    2 KB
                            total       4 KB per node

Storage doubles. But:
  - Belichtungsmesser early exit SAVES cycles on low-alpha comparisons
  - Penalty circuit PREVENTS false matches (fewer wasted cascade stages)
  - Alpha-normalized distance is MORE ACCURATE (no noise dilution)
  - MerkleRoot is STABLE (undefined noise doesn't change hash)
  - Confidence is STRUCTURAL (no separate NARS confidence field needed)

Net: more bytes, fewer cycles, better accuracy, simpler schema.
```

---

## 10. THE SIX RISC INSTRUCTIONS WITH ALPHA MODIFIER

```
INSTRUCTION     WITHOUT α                WITH .α MODIFIER                 COST
──────────────────────────────────────────────────────────────────────────────────
XOR             a XOR b                  data: a XOR b                    +1 AND
                                         α: a_α AND b_α

POPCOUNT        popcount(a XOR b)        popcount((a XOR b) AND aα AND bα) +2 AND
                / total_bits             / popcount(aα AND bα)

MAJORITY        majority_vote(inputs)    vote on α=1 positions only       +1 AND
                                         result_α = majority(input_αs)      per input

AND/NOT         a AND b / NOT a          data: same                       +1 AND
                                         α: a_α AND b_α / a_α

BLAKE3          blake3(data)             blake3(data AND α)               +1 AND

THRESHOLD       dist > thresh            α-normalized dist > thresh       +overlap
                                         AND shared_α > min_overlap         check
```

Alpha is NOT a seventh instruction. It's a MODIFIER on the existing six.
Like ARM condition codes. Like x86 REP prefix. Every instruction optionally
respects the alpha channel. Same ISA. Richer semantics. One extra AND.

---

## 11. IMPLEMENTATION IN HEXAGON

```
hexagon/crates/hexagon-core/src/
  xor.rs          → add xor_alpha(a, b, a_α, b_α) → (result, result_α)
  popcount.rs     → add hamming_alpha(a, b, a_α, b_α) → (distance, overlap)
  majority.rs     → add bundle_alpha(inputs, alphas) → (result, result_α)
  factorize.rs    → 8 projections, each with its own alpha
  seal.rs         → blake3(data AND alpha) for stable MerkleRoot
  threshold.rs    → add incomparable check (shared_α < min)

hexagon/crates/hexagon-core/src/alpha.rs  (NEW, ~200 lines)
  AlphaVector: Binary(2048) with utility methods
  alpha_density() → f32
  shared_overlap(a_α, b_α) → u32
  penalty(α) → u32 (popcount of NOT α)
  is_fully_defined(α) → bool
  is_fully_transparent(α) → bool

hexagon/crates/hexagon-cam/src/
  crystal.rs      → SPOCrystal stores (data, alpha) pairs
  orthogonal.rs   → OrthogonalCodebook entries have full alpha
  
hexagon/crates/hexagon-bnn/src/
  pentary.rs      → accumulator magnitude → alpha threshold derivation
  hebbian.rs      → co-fire strengthening respects alpha (only defined bits)
```

---

## 12. WHAT THIS CHANGES IN THE ARCHITECTURE

```
BEFORE:
  Confidence is a float field stored alongside the vector.
  Unknown dimensions are filled with noise.
  Comparisons pay for noise bits.
  False matches happen when noise accidentally aligns.
  Merkle roots change when noise changes.
  
AFTER:
  Confidence IS the alpha density — structural, not metadata.
  Unknown dimensions are alpha=0, transparent, zero cost.
  Comparisons skip undefined regions via early exit.
  False matches are penalized by the Orangutan penalty circuit.
  Merkle roots are stable (noise zeroed before hashing).
  NARS confidence can be DERIVED, not stored.
  Partial vs uncertain knowledge is structurally distinguished.
```

---

*"Absence of evidence is weak evidence of absence — and the penalty circuit makes the substrate believe it."*

*"The system's default state is doubt. Only signal cancels it. That's not a bug. That's epistemology."*
