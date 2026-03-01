# Addendum: 0.5σ Stripe Migration as Distributional Shift Detector

## Add to SPO Distance Harvest

The raw popcount (Option A from investigation) should be compared against **5 σ-thresholds** not 2:

```
1.0σ, 1.5σ, 2.0σ, 2.5σ, 3.0σ
```

Cost: 5 CMP instructions instead of 2. Zero meaningful overhead.

## What to track

Per plane, per search batch, maintain a **stripe histogram**: count of candidates landing in each band.

```rust
struct StripeHistogram {
    below_1s:   u32,  // noise floor
    s1_to_s15:  u32,  // emerging
    s15_to_s2:  u32,  // IQ confidence band  
    s2_to_s25:  u32,  // strong
    s25_to_s3:  u32,  // very strong
    above_3s:   u32,  // foveal
}
```

3 planes × 6 bins = 18 counters. Track across time windows.

## The signal

**Migration between adjacent stripes across time = distributional shift.**

- Population migrating toward foveal → codebook matches improving → increase NARS confidence, expect faster resonator convergence
- Population migrating toward noise → codebook going stale → decrease NARS confidence, increase DN mutation rate, flag codebook for expansion
- Bimodal migration (some toward foveal, some toward noise) → speciation event, world splitting into clusters
- Stable → steady state

The migration velocity between adjacent stripes IS the NARS evidence rate. If the 2σ→2.5σ boundary is seeing net positive flow, confidence should be rising. If net negative, confidence should be falling.

## Implementation

```rust
struct ShiftDetector {
    current: [StripeHistogram; 3],  // S, P, O planes
    previous: [StripeHistogram; 3], // last window
    
    fn detect_shift(&self) -> ShiftSignal {
        // Per plane: compare current vs previous histogram
        // Chi-squared or KL divergence on the 6 bins
        // If statistically significant → shift detected
        // Direction: center of mass moving toward foveal or noise?
    }
}
```

This gives the Belichtungsmesser its temporal dimension — not just "what is the current exposure" but "is the exposure changing and in which direction."

Wire the ShiftSignal into CollapseGate: if shift detected toward noise → bias toward HOLD (don't commit while ground is moving). If shift toward foveal → bias toward FLOW (the world is clarifying, commit faster).
