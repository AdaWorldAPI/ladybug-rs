# Investigation: Distance Granularity Options for SPO Harvest

## Context

The current adaptive cascade uses 3 coarse σ-bands (σ₁, σ₂, σ₃) as thresholds for filtering. The SPO distance harvest prompt claims 238× fewer cycles than cosine by using Hamming distance via VPOPCNTDQ.

**Question:** Between "3 coarse σ-bands" and "full floating-point cosine," what intermediate granularity options exist that give finer distance resolution while staying in the popcount cost class?

## The Constraint

The 238× advantage comes from: 3 × XOR(2048 bytes) + 3 × POPCNT(2048 bytes) = ~192 AVX-512 instructions ≈ 12 cycles.

Any granularity improvement must stay within ~2× of this cost (≤ 24 cycles) to remain in a fundamentally different cost class from cosine (~3100 cycles). At 60% overhead (~19 cycles) we're still 160× faster than cosine, which is fine. At 10× overhead (~120 cycles) we've lost the dramatic advantage.

## Options to Investigate

### Option A: Raw Popcount (No Binning)
```
Current:  popcount → compare to σ thresholds → bin {σ₁, σ₂, σ₃}
Proposed: popcount → keep raw u32 count per plane
```
- Overhead: ZERO. The popcount already produces the raw number. σ-binning is a REDUCTION of information we already have.
- Resolution: u32 per plane = 15 bits of distance precision per plane = 45 bits total
- Question: Is the codebase currently discarding the raw count after binning? If so, this is pure win — just stop throwing away information.

### Option B: Per-Word Popcount Histogram  
```
Each plane = 2048 bytes = 32 × 64-byte AVX-512 words
Instead of summing all 32 popcounts into one u32:
Keep the 32 individual popcounts as a histogram
```
- Overhead: ~0 extra compute (we already compute per-word), ~128 bytes storage (32 × u16 × 2 bytes × 3 planes = 192 bytes per comparison, but can be u8 since max per-word popcount for 512 bits = 512)
- Resolution: 32-bin spatial histogram of where agreement/disagreement lives
- This tells you: "dimensions 0-511 agree strongly, dimensions 512-1023 disagree strongly" — POSITIONAL distance information
- Question: Does the positional structure of Jina embeddings carry semantic meaning? If dimensions are randomly ordered, per-word histograms are noise. If early dims are more salient (which they often are in PCA-based embeddings), this is gold.

### Option C: Multi-Threshold Cascade (More σ-Bands)
```
Current:  3σ reject → 2σ reject → full
Proposed: 3σ → 2.5σ → 2σ → 1.5σ → 1σ → full
```
- Overhead: 2× more comparisons at early stages (5 thresholds vs 3)
- Benefit: finer-grained typing at each stage
- Question: Do the extra thresholds actually change the halo distribution, or does the cross-plane vote already capture finer gradations than the per-plane σ-bands?

### Option D: Quinary L1 Distance (Exploit Signed Values)
```
The SPO planes are quinary: {-2, -1, 0, +1, +2}
Binary XOR + popcount measures Hamming (agree/disagree)
But quinary L1 measures |a[i] - b[i]| which has range [0, 4]

L1 gives you 5 levels per dimension instead of 2 (agree/disagree)
```
- Overhead: Significant. L1 on INT8 uses VPSADBW (sum of absolute differences) = ~32 instructions per plane, ~96 total. Plus accumulation. Probably ~30-40 cycles total.
- Resolution: 5 levels per dimension vs 2. Much richer.
- Question: Is the 3× cost increase (12 → 40 cycles) worth the 2.5× information increase? Still 80× faster than cosine.

### Option E: Hybrid — Popcount for Filter, L1 for Survivors
```
Stage 1: Binary Hamming (popcount) at 1/16 sample → reject 99.7%
Stage 2: Binary Hamming at 1/4 sample → reject 95% of remaining
Stage 3: Quinary L1 (VPSADBW) on full planes for survivors only
```
- Overhead: Almost zero at population level (L1 only runs on ~0.3% of candidates)
- Resolution: Full quinary precision where it matters, binary speed for filtering
- This matches the existing cascade philosophy: cheap broad strokes first, expensive precision last
- Question: Is this already how the cascade works? If Stage 3 already uses INT8, this might already be in place.

## What to Check in the Codebase

1. **Does the cascade currently preserve raw popcount, or does it bin to σ-thresholds?**
   If it bins: Option A is a free upgrade, just stop discarding the count.

2. **Does Stage 3 (foveal) already use INT8/L1, or is it still Hamming?**
   If INT8 is already there: Option E might already be partially implemented.

3. **Are per-word popcounts computed individually and then summed, or is the whole plane popcounted at once?**
   If per-word: Option B's histogram is available at zero compute cost.

4. **What is the actual cycle count for VPSADBW on 2048 bytes?**
   Need this to evaluate Option D's real overhead.

5. **Benchmark all 5 options against cosine on the same dataset.**
   Measure: cycles, recall@10, bits of distance information, halo typing accuracy.

## Recommendation

Start with Option A (free — just expose raw popcount) and Option B (near-free — keep per-word histogram). These have essentially zero overhead and strictly more information.

Then benchmark Option E (hybrid popcount + L1) to see if the quinary precision on survivors is worth the foveal cost.

Option C (more σ-bands) is probably the least valuable because the cross-plane vote already provides finer granularity than single-plane thresholds.

Option D (full L1) is the "nuclear option" — maximum information but highest overhead. Only worth it if Options A+B don't provide enough resolution.
