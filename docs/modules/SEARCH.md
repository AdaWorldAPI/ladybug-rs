# Search Module

The search module provides high-performance similarity search over binary fingerprints.

## HDR Cascade (`hdr_cascade.rs`)

Hierarchical Distance Refinement for O(1)-like search performance.

### Concept

```
Query → Level 0 (1-bit sketch)  → 90% filtered
      → Level 1 (4-bit count)   → 90% of survivors
      → Level 2 (8-bit count)   → 90% of survivors
      → Level 3 (Full popcount) → Exact distance

Result: ~7ns per candidate vs ~100ns for float cosine
```

### HdrIndex

```rust
pub struct HdrIndex {
    fingerprints: Vec<[u64; WORDS]>,
    sketches: Vec<u64>,      // 1-bit per word
    counts_4: Vec<[u8; 39]>, // 4-bit counts
    counts_8: Vec<[u8; 156]>,// 8-bit counts
}

impl HdrIndex {
    pub fn new() -> Self;
    pub fn add(&mut self, fp: &[u64; WORDS]);
    pub fn search(&self, query: &[u64; WORDS], k: usize) -> Vec<(usize, u32)>;
}
```

### Search Algorithm

```rust
pub fn search(&self, query: &[u64; WORDS], k: usize) -> Vec<(usize, u32)> {
    let mut candidates: Vec<(usize, u32)> = Vec::with_capacity(k * 2);

    for (idx, fp) in self.fingerprints.iter().enumerate() {
        // Level 0: Sketch filter (XOR + popcount on 64-bit sketch)
        let sketch_dist = (self.sketches[idx] ^ query_sketch).count_ones();
        if sketch_dist > threshold_0 { continue; }

        // Level 1: 4-bit count filter
        let count_dist = self.estimate_4bit(idx, query);
        if count_dist > threshold_1 { continue; }

        // Level 2: 8-bit count filter
        let count_dist = self.estimate_8bit(idx, query);
        if count_dist > threshold_2 { continue; }

        // Level 3: Full Hamming distance
        let dist = hamming_distance(fp, query);
        candidates.push((idx, dist));
    }

    // Sort and return top-k
    candidates.sort_by_key(|(_, d)| *d);
    candidates.truncate(k);
    candidates
}
```

### Performance

| Level | Operation | Time | Filtering |
|-------|-----------|------|-----------|
| 0 | 64-bit XOR + popcount | ~1ns | 90% |
| 1 | 39-byte comparison | ~2ns | 90% |
| 2 | 156-byte comparison | ~3ns | 90% |
| 3 | Full 1248-byte XOR | ~4ns | Exact |

**Total**: ~7ns average per candidate (with 90% early termination)

## Mexican Hat Filter

Implements difference-of-Gaussians for edge detection in similarity space:

```rust
pub struct MexicanHat {
    sigma_center: f32,
    sigma_surround: f32,
}

impl MexicanHat {
    pub fn response(&self, distance: u32) -> f32 {
        let center = gaussian(distance, self.sigma_center);
        let surround = gaussian(distance, self.sigma_surround);
        center - surround  // Positive for close, negative for far
    }
}
```

## Rolling Statistics

Adaptive thresholds based on recent search statistics:

```rust
pub struct RollingWindow {
    values: VecDeque<f32>,
    capacity: usize,
}

impl RollingWindow {
    pub fn mean(&self) -> f32;
    pub fn stddev(&self) -> f32;
    pub fn stats(&self) -> (f32, f32);
}
```

## Cognitive Search (`cognitive.rs`)

Integrates NARS truth values with similarity search:

```rust
pub struct CognitiveSearch {
    hdr_index: HdrIndex,
    truth_table: HashMap<usize, (f32, f32)>,  // (frequency, confidence)
}

impl CognitiveSearch {
    pub fn search_with_truth(
        &self,
        query: &[u64; WORDS],
        k: usize,
        min_confidence: f32,
    ) -> Vec<SearchResult>;
}
```

## Causal Search (`causal.rs`)

Pearl's three rungs of causation:

```rust
pub enum CausalQuery {
    See(Fingerprint),           // Rung 1: Observation
    Do(Fingerprint, Action),    // Rung 2: Intervention
    Imagine(Fingerprint, Counterfactual), // Rung 3: Imagination
}

pub struct CausalEngine {
    pub fn query(&self, q: CausalQuery) -> Vec<CausalResult>;
}
```

## SIMD Acceleration

Hamming distance uses platform-specific SIMD:

```rust
// AVX-512: 8 fingerprints parallel
#[cfg(target_feature = "avx512vpopcntdq")]
unsafe fn hamming_avx512(a: &[u64; 156], b: &[u64; 156]) -> u32;

// AVX2: 4 fingerprints parallel
#[cfg(target_feature = "avx2")]
unsafe fn hamming_avx2(a: &[u64; 156], b: &[u64; 156]) -> u32;

// NEON: ARM acceleration
#[cfg(target_feature = "neon")]
unsafe fn hamming_neon(a: &[u64; 156], b: &[u64; 156]) -> u32;

// Fallback: Portable
fn hamming_scalar(a: &[u64; 156], b: &[u64; 156]) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
}
```

## Usage Example

```rust
use ladybug::search::HdrIndex;

// Create index
let mut index = HdrIndex::new();

// Add fingerprints
for fp in fingerprints {
    index.add(&fp);
}

// Search
let query = encode_text("hello world");
let results = index.search(&query, 10);

for (idx, distance) in results {
    let similarity = 1.0 - (distance as f32 / 9984.0);
    println!("Index {}: distance={}, similarity={:.3}", idx, distance, similarity);
}
```
