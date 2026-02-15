//! Texture Computation — 8-dimensional phenomenal texture of a fingerprint.
//!
//! A `Texture` captures the "feel" of a Container fingerprint when interpreted
//! in the context of graph-structural metrics.  The eight dimensions are:
//!
//! | Dimension    | Source                    | Meaning                          |
//! |-------------|---------------------------|----------------------------------|
//! | `entropy`    | Bitpattern                | Information density              |
//! | `purity`     | Bitpattern                | How concentrated the encoding is |
//! | `density`    | Bitpattern                | Fraction of set bits             |
//! | `bridgeness` | Graph metrics             | Cross-community connectivity     |
//! | `warmth`     | Bitpattern + graph        | Affective temperature (valence)  |
//! | `edge`       | Bitpattern                | Transition sharpness             |
//! | `depth`      | Bitpattern                | Abstraction level                |
//! | `flow`       | Graph metrics             | Information throughput           |
//!
//! # Usage
//!
//! ```rust,ignore
//! use ladybug::qualia::texture::{Texture, GraphMetrics, compute};
//! use ladybug::container::Container;
//!
//! let fp = Container::random(42);
//! let metrics = GraphMetrics::default();
//! let tex = compute(&fp, &metrics);
//! println!("entropy={:.3}, warmth={:.3}", tex.entropy, tex.warmth);
//! ```

use crate::container::{Container, CONTAINER_BITS, CONTAINER_WORDS};

// =============================================================================
// GRAPH METRICS (minimal struct for external graph-structural data)
// =============================================================================

/// Graph-structural metrics that influence texture computation.
///
/// These would typically be computed from the knowledge graph surrounding
/// the fingerprint's node.  When unavailable, use `GraphMetrics::default()`
/// for reasonable neutral values.
#[derive(Clone, Debug)]
pub struct GraphMetrics {
    /// Degree centrality of the node [0.0, 1.0].
    pub degree_centrality: f32,

    /// Betweenness centrality [0.0, 1.0].
    pub betweenness_centrality: f32,

    /// Clustering coefficient [0.0, 1.0].
    pub clustering_coefficient: f32,

    /// PageRank score [0.0, 1.0].
    pub pagerank: f32,

    /// Number of distinct communities the node bridges.
    pub communities_bridged: u32,

    /// Average edge weight to neighbours [0.0, 1.0].
    pub avg_edge_weight: f32,

    /// In-degree / out-degree ratio (information flow direction).
    /// > 1.0 = sink (more incoming), < 1.0 = source (more outgoing).
    pub flow_ratio: f32,
}

impl Default for GraphMetrics {
    fn default() -> Self {
        Self {
            degree_centrality: 0.5,
            betweenness_centrality: 0.0,
            clustering_coefficient: 0.5,
            pagerank: 0.5,
            communities_bridged: 1,
            avg_edge_weight: 0.5,
            flow_ratio: 1.0,
        }
    }
}

// =============================================================================
// TEXTURE STRUCT
// =============================================================================

/// 8-dimensional phenomenal texture of a fingerprint.
///
/// All dimensions are normalised to [0.0, 1.0].
#[derive(Clone, Debug)]
pub struct Texture {
    /// Shannon entropy of the word-level bit distribution.
    pub entropy: f32,

    /// Purity: 1.0 - entropy.  High purity = concentrated, low-noise signal.
    pub purity: f32,

    /// Fraction of set bits (popcount / CONTAINER_BITS).
    pub density: f32,

    /// Cross-community connectivity (from graph metrics + bitpattern).
    pub bridgeness: f32,

    /// Affective warmth / valence proxy.
    pub warmth: f32,

    /// Transition sharpness / edge energy in the bitpattern.
    pub edge: f32,

    /// Abstraction depth (hierarchical block variance).
    pub depth: f32,

    /// Information throughput / flow indicator.
    pub flow: f32,
}

impl Default for Texture {
    fn default() -> Self {
        Self {
            entropy: 0.0,
            purity: 1.0,
            density: 0.0,
            bridgeness: 0.0,
            warmth: 0.5,
            edge: 0.0,
            depth: 0.0,
            flow: 0.5,
        }
    }
}

impl Texture {
    /// Euclidean distance to another Texture.
    pub fn distance(&self, other: &Texture) -> f32 {
        let d = [
            self.entropy - other.entropy,
            self.purity - other.purity,
            self.density - other.density,
            self.bridgeness - other.bridgeness,
            self.warmth - other.warmth,
            self.edge - other.edge,
            self.depth - other.depth,
            self.flow - other.flow,
        ];
        d.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Cosine similarity to another Texture.
    pub fn similarity(&self, other: &Texture) -> f32 {
        let a = self.as_array();
        let b = other.as_array();

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }

    /// Convert to an 8-element array for linear algebra operations.
    pub fn as_array(&self) -> [f32; 8] {
        [
            self.entropy,
            self.purity,
            self.density,
            self.bridgeness,
            self.warmth,
            self.edge,
            self.depth,
            self.flow,
        ]
    }

    /// Construct from an 8-element array.
    pub fn from_array(arr: [f32; 8]) -> Self {
        Self {
            entropy: arr[0],
            purity: arr[1],
            density: arr[2],
            bridgeness: arr[3],
            warmth: arr[4],
            edge: arr[5],
            depth: arr[6],
            flow: arr[7],
        }
    }
}

impl std::fmt::Display for Texture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Texture(ent={:.2} pur={:.2} den={:.2} brg={:.2} wrm={:.2} edg={:.2} dep={:.2} flw={:.2})",
            self.entropy, self.purity, self.density, self.bridgeness,
            self.warmth, self.edge, self.depth, self.flow
        )
    }
}

// =============================================================================
// COMPUTE
// =============================================================================

/// Compute the Texture of a fingerprint given graph-structural context.
pub fn compute(fingerprint: &Container, graph_metrics: &GraphMetrics) -> Texture {
    if fingerprint.is_zero() {
        return Texture::default();
    }

    let word_pops = word_popcounts(fingerprint);

    let entropy = compute_entropy(&word_pops);
    let purity = 1.0 - entropy;
    let density = compute_density(fingerprint);
    let edge = compute_edge_energy(fingerprint);
    let depth = compute_depth(&word_pops);

    // Bridgeness blends bitpattern transitions with graph betweenness.
    let bit_bridge = compute_bit_bridgeness(fingerprint);
    let bridgeness = blend_bridgeness(bit_bridge, graph_metrics);

    // Warmth: a proxy for affective temperature.  High clustering + moderate
    // density suggests warm, connected content.  Cold = isolated, sparse.
    let warmth = compute_warmth(density, graph_metrics);

    // Flow: information throughput.  High degree + balanced flow ratio = high flow.
    let flow = compute_flow(graph_metrics);

    Texture {
        entropy,
        purity,
        density,
        bridgeness,
        warmth,
        edge,
        depth,
        flow,
    }
}

// =============================================================================
// INTERNAL COMPUTATIONS
// =============================================================================

fn word_popcounts(c: &Container) -> [u32; CONTAINER_WORDS] {
    let mut pops = [0u32; CONTAINER_WORDS];
    for (i, &w) in c.words.iter().enumerate() {
        pops[i] = w.count_ones();
    }
    pops
}

/// Shannon entropy of word-level popcount distribution, normalised to [0, 1].
fn compute_entropy(pops: &[u32; CONTAINER_WORDS]) -> f32 {
    let total: f64 = pops.iter().map(|&p| p as f64).sum();
    if total == 0.0 {
        return 0.0;
    }

    let mut h: f64 = 0.0;
    for &p in pops.iter() {
        if p > 0 {
            let prob = p as f64 / total;
            h -= prob * prob.ln();
        }
    }

    let max_h = (CONTAINER_WORDS as f64).ln();
    if max_h > 0.0 {
        (h / max_h) as f32
    } else {
        0.0
    }
}

fn compute_density(c: &Container) -> f32 {
    c.popcount() as f32 / CONTAINER_BITS as f32
}

/// Edge energy: fraction of bit-to-bit transitions (1->0 or 0->1).
fn compute_edge_energy(c: &Container) -> f32 {
    let mut transitions: u32 = 0;
    for word_idx in 0..CONTAINER_WORDS {
        let w = c.words[word_idx];
        transitions += (w ^ (w >> 1)).count_ones();

        if word_idx + 1 < CONTAINER_WORDS {
            let last_bit = (w >> 63) & 1;
            let next_first = c.words[word_idx + 1] & 1;
            if last_bit != next_first {
                transitions += 1;
            }
        }
    }

    let max = (CONTAINER_BITS - 1) as f32;
    if max > 0.0 {
        transitions as f32 / max
    } else {
        0.0
    }
}

/// Abstraction depth from block-level variance.
fn compute_depth(pops: &[u32; CONTAINER_WORDS]) -> f32 {
    let block_sizes: &[usize] = &[4, 16, 64];
    let mut total_var: f32 = 0.0;
    let mut count: f32 = 0.0;

    for &bs in block_sizes {
        if bs > CONTAINER_WORDS {
            break;
        }
        let n_blocks = CONTAINER_WORDS / bs;
        if n_blocks < 2 {
            continue;
        }

        let mut sums = Vec::with_capacity(n_blocks);
        for b in 0..n_blocks {
            let start = b * bs;
            let s: u32 = pops[start..start + bs].iter().sum();
            sums.push(s as f32);
        }

        let mean = sums.iter().sum::<f32>() / n_blocks as f32;
        let var = sums.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / n_blocks as f32;
        let max_var = (bs as f32 * 64.0).powi(2);
        total_var += if max_var > 0.0 {
            (var / max_var).sqrt()
        } else {
            0.0
        };
        count += 1.0;
    }

    if count > 0.0 {
        (total_var / count).min(1.0)
    } else {
        0.0
    }
}

/// Bitpattern bridgeness: cross-word transition ratio.
fn compute_bit_bridgeness(c: &Container) -> f32 {
    if CONTAINER_WORDS < 2 {
        return 0.0;
    }

    let mut cross_word: u32 = 0;
    for i in 0..(CONTAINER_WORDS - 1) {
        let last = (c.words[i] >> 63) & 1;
        let first = c.words[i + 1] & 1;
        if last != first {
            cross_word += 1;
        }
    }

    cross_word as f32 / (CONTAINER_WORDS - 1) as f32
}

/// Blend bitpattern bridgeness with graph betweenness centrality.
fn blend_bridgeness(bit_bridge: f32, gm: &GraphMetrics) -> f32 {
    let community_factor = (gm.communities_bridged as f32).min(10.0) / 10.0;
    let graph_bridge = gm.betweenness_centrality * 0.5 + community_factor * 0.5;

    // 60% bitpattern, 40% graph.
    (bit_bridge * 0.6 + graph_bridge * 0.4).clamp(0.0, 1.0)
}

/// Warmth: affective temperature proxy.
fn compute_warmth(density: f32, gm: &GraphMetrics) -> f32 {
    // High clustering + moderate density = warm.
    // Low clustering + extreme density = cold.
    let density_contribution = 1.0 - (density - 0.5).abs() * 2.0; // peaks at 0.5
    let cluster_contribution = gm.clustering_coefficient;
    let edge_contribution = gm.avg_edge_weight;

    ((density_contribution * 0.3 + cluster_contribution * 0.4 + edge_contribution * 0.3) as f32)
        .clamp(0.0, 1.0)
}

/// Flow: information throughput.
fn compute_flow(gm: &GraphMetrics) -> f32 {
    // High degree + balanced flow ratio = high flow.
    let degree_factor = gm.degree_centrality;
    let balance_factor = 1.0 - (gm.flow_ratio - 1.0).abs().min(1.0);
    let pr_factor = gm.pagerank;

    (degree_factor * 0.4 + balance_factor * 0.3 + pr_factor * 0.3).clamp(0.0, 1.0)
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_container_texture() {
        let c = Container::zero();
        let gm = GraphMetrics::default();
        let t = compute(&c, &gm);
        assert_eq!(t.density, 0.0);
        assert_eq!(t.purity, 1.0);
    }

    #[test]
    fn test_random_container_texture_in_range() {
        let c = Container::random(42);
        let gm = GraphMetrics::default();
        let t = compute(&c, &gm);

        for &v in t.as_array().iter() {
            assert!(v >= 0.0 && v <= 1.0, "value {} out of range", v);
        }
    }

    #[test]
    fn test_ones_container_has_full_density() {
        let c = Container::ones();
        let gm = GraphMetrics::default();
        let t = compute(&c, &gm);
        assert!((t.density - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_purity_is_complement_of_entropy() {
        let c = Container::random(99);
        let gm = GraphMetrics::default();
        let t = compute(&c, &gm);
        assert!(
            (t.entropy + t.purity - 1.0).abs() < 1e-5,
            "entropy={}, purity={}",
            t.entropy,
            t.purity
        );
    }

    #[test]
    fn test_distance_same_is_zero() {
        let c = Container::random(42);
        let gm = GraphMetrics::default();
        let t = compute(&c, &gm);
        assert!(t.distance(&t) < 1e-6);
    }

    #[test]
    fn test_distance_different() {
        let c1 = Container::random(1);
        let c2 = Container::random(2);
        let gm = GraphMetrics::default();
        let t1 = compute(&c1, &gm);
        let t2 = compute(&c2, &gm);
        // Different random containers should have nonzero distance.
        assert!(t1.distance(&t2) > 0.0);
    }

    #[test]
    fn test_similarity_self_is_near_one() {
        let c = Container::random(42);
        let gm = GraphMetrics::default();
        let t = compute(&c, &gm);
        let s = t.similarity(&t);
        assert!(s > 0.99, "self-similarity should be ~1.0, got {}", s);
    }

    #[test]
    fn test_graph_metrics_affect_bridgeness() {
        let c = Container::random(42);

        let low_bridge = GraphMetrics {
            betweenness_centrality: 0.0,
            communities_bridged: 1,
            ..Default::default()
        };
        let high_bridge = GraphMetrics {
            betweenness_centrality: 0.9,
            communities_bridged: 8,
            ..Default::default()
        };

        let t_low = compute(&c, &low_bridge);
        let t_high = compute(&c, &high_bridge);

        assert!(
            t_high.bridgeness > t_low.bridgeness,
            "high graph bridgeness should increase texture bridgeness"
        );
    }

    #[test]
    fn test_display_formatting() {
        let c = Container::random(42);
        let gm = GraphMetrics::default();
        let t = compute(&c, &gm);
        let s = format!("{}", t);
        assert!(s.starts_with("Texture("));
    }

    #[test]
    fn test_from_array_roundtrip() {
        let c = Container::random(42);
        let gm = GraphMetrics::default();
        let t = compute(&c, &gm);
        let arr = t.as_array();
        let t2 = Texture::from_array(arr);
        assert!(t.distance(&t2) < 1e-6);
    }
}
