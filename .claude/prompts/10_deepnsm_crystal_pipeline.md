# DeepNSM → Crystal Encoder Pipeline

## The Five Pieces

| Piece | Location | What It Is | Status |
|-------|----------|-----------|--------|
| **Paper** | arXiv:2505.11764 | "Towards Universal Semantics with LLMs" — 65 semantic primes, explications beat dictionary defs, 1B/8B models beat GPT-4o | Published May 2025 |
| **DeepNSM repo** | AdaWorldAPI/DeepNSM | Fork of paper's code: eval pipeline, prompts, train wrappers. Models on HF: `baartmar/DeepNSM-1B`, `baartmar/DeepNSM-8B`, `baartmar/nsm_dataset` | Working (Python, needs GPU) |
| **deepmsm repo** | AdaWorldAPI/deepmsm | Unrelated — "deep Markov State Modeling" for molecular dynamics. Wrong repo. Not ours. | **IGNORE** |
| **nsm.rs** | ladybug-rs `src/grammar/nsm.rs` (448 lines) | 65 NSM primitives as Rust constants. `NSMField` = 65-dim float vector. `from_text()` = keyword match → prime weights. `to_fingerprint_contribution()` = golden-ratio hash projection to fingerprint bits. | Working but primitive — keyword matching, not LLM explications |
| **deepnsm_integration.rs** | ladybug-rs `src/spo/deepnsm_integration.rs` (26KB) | Integration spec: explication → prime weights → role-bind → fingerprint. Training: use DeepNSM model. Inference: pure SIMD, no LLM. | Architecture defined, partially implemented |

## The Pipeline (how they connect)

```
                    TRAINING (one-time, needs GPU)
                    ═══════════════════════════════
                    
Text corpus ──► DeepNSM-8B model (HF: baartmar/DeepNSM-8B)
                    │
                    ▼
            NSM Explications
            "HAPPY = a person feels something good,
             this person thinks: something good happened to me"
                    │
                    ▼
            Parse into Prime Weight Vectors
            HAPPY → [I:0.3, FEEL:0.9, GOOD:0.8, THINK:0.6, 
                      HAPPEN:0.5, SOMETHING:0.7, ...]
                    │
                    ▼
            Build Codebook: 1024 σ₃-distinct prime-weight clusters
            Each centroid = a "semantic kernel locus" in prime space
                    │
                    ▼
            Project each centroid → 16K-bit fingerprint via nsm.rs
            (currently golden-ratio hash, should be learned projection)


                    INFERENCE (forever, no GPU, no LLM)
                    ════════════════════════════════════

New text ──► Keyword → prime weights (nsm.rs `from_text()`)
                │
                ├──► Nearest codebook centroid (1024-way lookup)
                │        Cost: 1024 Hamming distances = ~13K cycles
                │
                ├──► SPO role binding: S⊗subject_primes + P⊗predicate_primes + O⊗object_primes
                │        Cost: 3 XOR operations
                │
                └──► Store as SPO fingerprint with typed halo + NARS truth
                         Cost: 13 cycles per comparison
```

## What's Missing (the gap between paper and production)

### Gap 1: `from_text()` is keyword matching, not explication

Current `nsm.rs`:
```rust
pub fn from_text(text: &str) -> NSMField {
    // Looks for "I", "WANT", "KNOW" etc. as literal words in text
    // This misses: "desire" → WANT, "understand" → KNOW, "beautiful" → GOOD+SEE
}
```

The paper's insight: DeepNSM-1B generates proper explications that decompose ANY word into primes. "Schadenfreude" → "this person feels something good because something bad happened to another person" → [FEEL:0.9, GOOD:0.7, BAD:0.6, HAPPEN:0.8, SOMEONE:0.5, BECAUSE:0.9].

**Fix:** Build a lookup table from DeepNSM-8B explications. Run DeepNSM on vocabulary (e.g., 50K words from WordNet or domain corpus). Parse each explication into prime weights. Store as the codebook. At inference time, tokenize → lookup → sum prime weights. No LLM needed.

### Gap 2: `to_fingerprint_contribution()` uses golden-ratio hash

Current projection:
```rust
let seed = (i as u64).wrapping_mul(0x9E3779B97F4A7C15); // golden ratio
let bit_pos = (seed.wrapping_mul((j + 1) as u64) % FINGERPRINT_BITS as u64) as usize;
```

This is random projection. It works (JL lemma guarantees distance preservation in expectation) but it's not learned. The paper shows that DeepNSM models learn better-than-random representations.

**Fix:** Train the projection matrix. Use DeepNSM explication pairs with known similarity as training data. Optimize the 65→16K projection to maximize σ₃ separation in Hamming space. This is the codebook training pipeline from `codebook_training.rs` (Phase 2: distillation).

### Gap 3: No SPO role binding for primes

Current: NSMField treats all 65 primes as a flat vector. But the paper's explications have structure: "a PERSON FEELS something GOOD" has Subject (PERSON), Predicate (FEEL), Object (GOOD).

**Fix:** Parse explications into SPO structure, then bind primes to roles:
```rust
pub fn explication_to_spo(explication: &str) -> SpoFingerprint {
    let parsed = parse_nsm_explication(explication);
    let s_primes = NSMField::from_primes(&parsed.agent_primes);
    let p_primes = NSMField::from_primes(&parsed.action_primes);
    let o_primes = NSMField::from_primes(&parsed.patient_primes);
    
    SpoFingerprint {
        s_plane: s_primes.to_fingerprint(),
        p_plane: p_primes.to_fingerprint(),
        o_plane: o_primes.to_fingerprint(),
    }
}
```

Now each plane carries NSM-grounded meaning. The Faktorzerlegung decomposes in terms of universal semantic primes, not arbitrary embedding dimensions.

### Gap 4: Evaluation metrics not wired

The paper defines three metrics:
1. **Legality Score**: (primes - molecules) / total_words — how pure is the explication?
2. **Substitutability Score**: log-probability tests — does replacing the word with its explication preserve meaning?
3. **Cross-Translatability**: round-trip BLEU through low-resource languages

These should be wired into the codebook training pipeline as quality gates. An explication with legality < 0.7 doesn't enter the codebook.

## How This Connects to the Crystal Encoder Strategy (prompt 05)

The crystal encoder strategy has three phases:
```
Phase 1: Jina parallel (external API, 1024D dense, ~100ms)
Phase 2: Distillation (Burn/Candle, SPO structural loss)
Phase 3: Pure crystal (no external, codebook only, ~5μs)
```

NSM/DeepNSM provides a **fourth path that may be better than all three**:

```
Phase 0: NSM bootstrap (one-time)
  Run DeepNSM-8B on vocabulary → explications → prime weights → codebook
  No continuous distillation needed. The codebook IS the model.
  
Phase 3b: Pure NSM inference
  Text → tokenize → prime weight lookup → SPO role bind → fingerprint
  Cost: dictionary lookup + 3 XOR operations
  No Jina. No Burn/Candle. No transformer. No API.
  The 65 primes ARE the embedding dimensions.
```

This is even cheaper than the σ₃ codebook lookup because you skip the nearest-centroid step entirely. The prime weights directly encode meaning. The projection to fingerprint space is deterministic.

**The trade-off**: NSM keyword matching (`from_text()`) is cruder than transformer encoding. But:
- The paper shows DeepNSM-1B (1 billion params) beats GPT-4o on explication quality
- If you build the lookup table from DeepNSM explications, you get that quality at dictionary-lookup cost
- 65 primes is an incredibly compact representation (65 floats = 260 bytes vs 1024D = 4KB)
- The primes are **universal** (attested in 90+ languages) — no retraining for new languages

## Connection to Edge Validation (from fork plan prompt 10, Action 2)

The edge validation in the lance-graph fork uses three paths:
- Path A: Grammar verbs (144 in ladybug)
- Path B: NSM primes (65 from this paper)
- Path C: Semantic kernel loci (1024 σ₃ codebook)

With DeepNSM integration:
```
Query: MATCH (a)-[:LOVES]->(b)

Path B resolution:
  "LOVES" → DeepNSM explication → "SOMEONE FEELS something very GOOD 
   about another SOMEONE, this SOMEONE WANTS to be NEAR this other SOMEONE"
  → NSMField: [SOMEONE:1.0, FEEL:0.9, GOOD:0.8, WANT:0.7, NEAR:0.6]
  → fingerprint contribution
  → Hamming search in predicate plane for any edge with similar prime activation
  → Returns: LOVES, ADORES, CHERISHES, IS_FOND_OF (all with confidence scores)
```

This gives **semantic edge validation**, not string matching. A query for `:LOVES` finds `:ADORES` because they decompose into similar primes. This is what makes the Cypher bridge intelligent.

## Action Items

### Immediate
1. **Verify DeepNSM-8B model access**: Download from `baartmar/DeepNSM-8B` on HF. Test locally.
2. **Generate lookup table**: Run DeepNSM on WordNet core vocabulary (~5000 words). Parse explications into prime weights. Store as JSON/bincode codebook.
3. **Upgrade `from_text()`**: Replace keyword matching with codebook lookup. Fall back to keywords for OOV words.

### Short-term
4. **SPO role parsing**: Implement explication → Subject/Predicate/Object prime extraction. Use grammar module's `CausalityFlow` for agent/action/patient parsing.
5. **Quality gates**: Wire legality/substitutability/cross-translatability scores into codebook training.
6. **Learned projection**: Replace golden-ratio hash with trained 65→16K projection matrix optimized for σ₃ separation.

### Medium-term
7. **Full crystal encoder Phase 0**: NSM bootstrap path as alternative to Jina/Burn/Candle.
8. **Edge validation**: Wire NSM resolution into lance-graph fork's `resolve_edge_type()`.
9. **Multi-language**: Test with non-English input. NSM primes are language-universal — the codebook should work across languages without retraining.

## Note on deepmsm

`AdaWorldAPI/deepmsm` is a **molecular dynamics** repo ("Progress in deep Markov State Modeling: Coarse graining and experimental data restraints"). It has nothing to do with Natural Semantic Metalanguage. The naming collision is unfortunate. This repo should be ignored for the current work.
