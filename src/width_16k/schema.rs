//! Compressed 32-Word Metadata Sidecar (words 224-255)
//!
//! HDC-aligned: minimize non-homogeneous zone. All metadata in 2 blocks.
//! No self_addr / parent_addr — the DN address path encodes both
//! (path[depth-1] = self, path[depth-2] = parent).
//!
//! ## Block 14 (words 224-239): Identity + Reasoning + Learning
//!
//! ```text
//! [224] depth:u8 | rung:u8 | qidx:u16 | access_count:u32
//! [225] ttl:u16 | sigma_q:u16 | node_type:u32
//! [226] label_hash:u64
//! [227] edge_type:u32 | version:u8 | reserved:u24
//! [228-229] ANI levels: 8 x u16 = 128 bits
//! [230] NARS truth:u32 | budget_lo:u32 (priority + durability)
//! [231] budget_hi:u32 (quality + reserved) | reserved:u32
//! [232-233] Q-values: 16 x i8 = 128 bits
//! [234-235] Rewards: 8 x i16 = 128 bits
//! [236-237] STDP: 8 x u16 = 128 bits
//! [238-239] Hebbian: 8 x u16 = 128 bits
//! ```
//!
//! ## Block 15 (words 240-255): Graph Topology + Edges
//!
//! ```text
//! [240-243] DN address: 32 x u8 = 256 bits
//! [244-247] Neighbor bloom: 4 x u64 = 256 bits
//! [248] Graph metrics: packed u64
//! [249-255] Inline edges: 7 words = up to 28 edges at 16 bits each
//! ```

use super::VECTOR_WORDS;

// ============================================================================
// BLOCK 14: IDENTITY + REASONING + LEARNING (words 224-239)
// ============================================================================

/// Operational identity packed into words[224..228].
///
/// No self_addr or parent_addr: the DN address path in block 15
/// already encodes the full traversal path including self and parent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NodeIdentity {
    /// Tree depth (0 = root)
    pub depth: u8,
    /// Pearl's causal rung: 0=SEE, 1=DO, 2=IMAGINE
    pub rung: u8,
    /// Quantization index (codebook entry)
    pub qidx: u16,
    /// Access count for LRU/frequency tracking
    pub access_count: u32,
    /// Time-to-live in ticks (0 = permanent)
    pub ttl: u16,
    /// Uncertainty: sigma * 1000 as u16
    pub sigma_q: u16,
    /// Node type marker
    pub node_type: NodeTypeMarker,
    /// Hash of the string label
    pub label_hash: u64,
    /// Edge type descriptor
    pub edge_type: EdgeTypeMarker,
}

impl NodeIdentity {
    pub fn sigma_from_f32(sigma: f32) -> u16 {
        (sigma.clamp(0.0, 65.535) * 1000.0) as u16
    }

    pub fn sigma_as_f32(&self) -> f32 {
        self.sigma_q as f32 / 1000.0
    }
}

/// ANI reasoning levels (8 x u16 = 128 bits)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct AniLevels {
    pub reactive: u16,
    pub memory: u16,
    pub analogy: u16,
    pub planning: u16,
    pub meta: u16,
    pub social: u16,
    pub creative: u16,
    pub r#abstract: u16,
}

impl AniLevels {
    pub fn dominant(&self) -> u8 {
        let levels = [
            self.reactive, self.memory, self.analogy, self.planning,
            self.meta, self.social, self.creative, self.r#abstract,
        ];
        levels.iter().enumerate().max_by_key(|(_, v)| **v)
            .map(|(i, _)| i as u8).unwrap_or(0)
    }

    pub fn pack(&self) -> u128 {
        (self.reactive as u128)
            | ((self.memory as u128) << 16)
            | ((self.analogy as u128) << 32)
            | ((self.planning as u128) << 48)
            | ((self.meta as u128) << 64)
            | ((self.social as u128) << 80)
            | ((self.creative as u128) << 96)
            | ((self.r#abstract as u128) << 112)
    }

    pub fn unpack(packed: u128) -> Self {
        Self {
            reactive: packed as u16,
            memory: (packed >> 16) as u16,
            analogy: (packed >> 32) as u16,
            planning: (packed >> 48) as u16,
            meta: (packed >> 64) as u16,
            social: (packed >> 80) as u16,
            creative: (packed >> 96) as u16,
            r#abstract: (packed >> 112) as u16,
        }
    }
}

/// NARS truth value: frequency + confidence, quantized to u16
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct NarsTruth {
    pub frequency: u16,
    pub confidence: u16,
}

impl NarsTruth {
    pub fn from_floats(f: f32, c: f32) -> Self {
        Self {
            frequency: (f.clamp(0.0, 1.0) * 65535.0) as u16,
            confidence: (c.clamp(0.0, 0.9999) * 65535.0) as u16,
        }
    }

    pub fn f(&self) -> f32 { self.frequency as f32 / 65535.0 }
    pub fn c(&self) -> f32 { self.confidence as f32 / 65535.0 }

    pub fn revision(&self, other: &Self) -> Self {
        let w1 = self.c() / (1.0 - self.c());
        let w2 = other.c() / (1.0 - other.c());
        let w = w1 + w2;
        let f = if w > 0.0 { (w1 * self.f() + w2 * other.f()) / w } else { 0.5 };
        Self::from_floats(f, w / (w + 1.0))
    }

    pub fn deduction(&self, other: &Self) -> Self {
        let f = self.f() * other.f();
        Self::from_floats(f, f * self.c() * other.c())
    }

    pub fn pack(&self) -> u32 { (self.frequency as u32) | ((self.confidence as u32) << 16) }
    pub fn unpack(v: u32) -> Self { Self { frequency: v as u16, confidence: (v >> 16) as u16 } }
}

/// NARS budget: priority, durability, quality
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct NarsBudget {
    pub priority: u16,
    pub durability: u16,
    pub quality: u16,
    pub _reserved: u16,
}

impl NarsBudget {
    pub fn from_floats(p: f32, d: f32, q: f32) -> Self {
        Self {
            priority: (p.clamp(0.0, 1.0) * 65535.0) as u16,
            durability: (d.clamp(0.0, 1.0) * 65535.0) as u16,
            quality: (q.clamp(0.0, 1.0) * 65535.0) as u16,
            _reserved: 0,
        }
    }

    pub fn pack(&self) -> u64 {
        (self.priority as u64) | ((self.durability as u64) << 16)
            | ((self.quality as u64) << 32) | ((self._reserved as u64) << 48)
    }

    pub fn unpack(v: u64) -> Self {
        Self {
            priority: v as u16, durability: (v >> 16) as u16,
            quality: (v >> 32) as u16, _reserved: (v >> 48) as u16,
        }
    }
}

/// Edge type: cognitive verb + direction + weight + flags
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct EdgeTypeMarker {
    pub verb_id: u8,
    pub direction: u8,
    pub weight: u8,
    pub flags: u8,
}

impl EdgeTypeMarker {
    pub fn pack(&self) -> u32 {
        (self.verb_id as u32) | ((self.direction as u32) << 8)
            | ((self.weight as u32) << 16) | ((self.flags as u32) << 24)
    }
    pub fn unpack(v: u32) -> Self {
        Self { verb_id: v as u8, direction: (v >> 8) as u8,
               weight: (v >> 16) as u8, flags: (v >> 24) as u8 }
    }
    pub fn is_temporal(&self) -> bool { self.flags & 1 != 0 }
    pub fn is_causal(&self) -> bool { self.flags & 2 != 0 }
    pub fn is_hierarchical(&self) -> bool { self.flags & 4 != 0 }
}

/// Node kind
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum NodeKind {
    Entity = 0, Concept = 1, Event = 2, Rule = 3,
    Goal = 4, Query = 5, Hypothesis = 6, Observation = 7,
}
impl Default for NodeKind { fn default() -> Self { Self::Entity } }

/// Node type: kind + subtype + provenance
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct NodeTypeMarker {
    pub kind: u8,
    pub subtype: u8,
    pub provenance: u16,
}

impl NodeTypeMarker {
    pub fn pack(&self) -> u32 {
        (self.kind as u32) | ((self.subtype as u32) << 8) | ((self.provenance as u32) << 16)
    }
    pub fn unpack(v: u32) -> Self {
        Self { kind: v as u8, subtype: (v >> 8) as u8, provenance: (v >> 16) as u16 }
    }
}

/// Q-values: 16 x i8 = 128 bits
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InlineQValues { pub values: [i8; 16] }

impl InlineQValues {
    pub fn q(&self, a: usize) -> f32 { if a < 16 { self.values[a] as f32 / 127.0 } else { 0.0 } }
    pub fn set_q(&mut self, a: usize, v: f32) {
        if a < 16 { self.values[a] = (v.clamp(-1.0, 1.0) * 127.0) as i8; }
    }
    pub fn best_action(&self) -> usize {
        self.values.iter().enumerate().max_by_key(|(_, v)| **v).map(|(i, _)| i).unwrap_or(0)
    }
    pub fn pack(&self) -> [u64; 2] {
        let mut w = [0u64; 2];
        for i in 0..8 { w[0] |= ((self.values[i] as u8) as u64) << (i * 8); }
        for i in 0..8 { w[1] |= ((self.values[i + 8] as u8) as u64) << (i * 8); }
        w
    }
    pub fn unpack(w: [u64; 2]) -> Self {
        let mut v = [0i8; 16];
        for i in 0..8 { v[i] = ((w[0] >> (i * 8)) & 0xFF) as u8 as i8; }
        for i in 0..8 { v[i + 8] = ((w[1] >> (i * 8)) & 0xFF) as u8 as i8; }
        Self { values: v }
    }
}

/// Reward history: 8 x i16 = 128 bits
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InlineRewards { pub rewards: [i16; 8] }

impl InlineRewards {
    pub fn push(&mut self, reward: f32) {
        for i in 0..7 { self.rewards[i] = self.rewards[i + 1]; }
        self.rewards[7] = (reward.clamp(-1.0, 1.0) * 32767.0) as i16;
    }
    pub fn average(&self) -> f32 {
        let s: i32 = self.rewards.iter().map(|&r| r as i32).sum();
        (s as f32 / 8.0) / 32767.0
    }
    pub fn pack(&self) -> [u64; 2] {
        let mut w = [0u64; 2];
        for i in 0..4 { w[0] |= ((self.rewards[i] as u16) as u64) << (i * 16); }
        for i in 0..4 { w[1] |= ((self.rewards[i + 4] as u16) as u64) << (i * 16); }
        w
    }
}

/// STDP timestamps: 8 x u16 = 128 bits
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StdpMarkers { pub timestamps: [u16; 8] }

impl StdpMarkers {
    pub fn record_spike(&mut self, time: u16) {
        for i in 0..7 { self.timestamps[i] = self.timestamps[i + 1]; }
        self.timestamps[7] = time;
    }
}

/// Hebbian weights: 8 x u16 = 128 bits
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InlineHebbian { pub weights: [u16; 8] }

impl InlineHebbian {
    pub fn weight(&self, i: usize) -> f32 {
        if i < 8 { self.weights[i] as f32 / 65535.0 } else { 0.0 }
    }
    pub fn strengthen(&mut self, i: usize, amount: f32) {
        if i < 8 {
            let v = (self.weights[i] as f32 / 65535.0 + amount).clamp(0.0, 1.0);
            self.weights[i] = (v * 65535.0) as u16;
        }
    }
    pub fn decay(&mut self, factor: f32) {
        for w in &mut self.weights { *w = ((*w as f32) * factor) as u16; }
    }
}

// ============================================================================
// BLOCK 15: GRAPH TOPOLOGY + EDGES (words 240-255)
// ============================================================================

/// DN address: 32 x u8 path (256 bits)
///
/// path[0] = root, path[depth-1] = self. Depth stored in block 14.
/// Parent = path[depth-2] (or root if depth <= 1).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompressedDnAddr { pub path: [u8; 32] }

/// Neighbor bloom: 4 x u64 = 256 bits, 3-hash
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NeighborBloom { pub words: [u64; 4] }

impl NeighborBloom {
    pub fn insert(&mut self, id: u64) {
        let h1 = id;
        let h2 = id.wrapping_mul(0x9E3779B97F4A7C15);
        let h3 = id.wrapping_mul(0x517CC1B727220A95);
        self.set_bit(h1 as usize % 256);
        self.set_bit(h2 as usize % 256);
        self.set_bit(h3 as usize % 256);
    }
    pub fn might_contain(&self, id: u64) -> bool {
        let h1 = id;
        let h2 = id.wrapping_mul(0x9E3779B97F4A7C15);
        let h3 = id.wrapping_mul(0x517CC1B727220A95);
        self.get_bit(h1 as usize % 256)
            && self.get_bit(h2 as usize % 256)
            && self.get_bit(h3 as usize % 256)
    }
    fn set_bit(&mut self, i: usize) { self.words[i / 64] |= 1u64 << (i % 64); }
    fn get_bit(&self, i: usize) -> bool { self.words[i / 64] & (1u64 << (i % 64)) != 0 }
}

/// Graph metrics (packed into one u64)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GraphMetrics {
    pub pagerank: u16,
    pub hop_to_root: u8,
    pub cluster_id: u16,
    pub degree: u8,
    pub in_degree: u8,
    pub out_degree: u8,
}

impl GraphMetrics {
    pub fn pack(&self) -> u64 {
        (self.pagerank as u64) | ((self.hop_to_root as u64) << 16)
            | ((self.cluster_id as u64) << 24) | ((self.degree as u64) << 40)
            | ((self.in_degree as u64) << 48) | ((self.out_degree as u64) << 56)
    }
    pub fn unpack(v: u64) -> Self {
        Self {
            pagerank: v as u16, hop_to_root: (v >> 16) as u8,
            cluster_id: (v >> 24) as u16, degree: (v >> 40) as u8,
            in_degree: (v >> 48) as u8, out_degree: (v >> 56) as u8,
        }
    }
}

/// Inline edge: verb(u8) + target_addr(u8) = 16 bits per edge
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InlineEdge {
    pub verb: u8,
    pub target: u8,
}

impl InlineEdge {
    pub fn pack(&self) -> u16 { (self.verb as u16) | ((self.target as u16) << 8) }
    pub fn unpack(v: u16) -> Self { Self { verb: v as u8, target: (v >> 8) as u8 } }
    pub fn is_empty(&self) -> bool { self.verb == 0 && self.target == 0 }
}

/// Inline edge array: 7 words = 28 edge slots (words 249-255)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InlineEdges {
    pub edges: [InlineEdge; 28],
}

impl InlineEdges {
    pub fn count(&self) -> usize {
        self.edges.iter().take_while(|e| !e.is_empty()).count()
    }

    pub fn push(&mut self, verb: u8, target: u8) -> bool {
        for e in &mut self.edges {
            if e.is_empty() {
                *e = InlineEdge { verb, target };
                return true;
            }
        }
        false // full
    }
}

// ============================================================================
// UNIFIED SIDECAR: 32 words (224-255)
// ============================================================================

/// Complete metadata sidecar for one 16K cognitive record.
///
/// 32 words = 2,048 bits. Two 1024-bit blocks:
/// - Block 14 (224-239): Identity + Reasoning + Learning
/// - Block 15 (240-255): Graph + Edges
#[derive(Clone, Debug, Default)]
pub struct SchemaSidecar {
    // Block 14
    pub identity: NodeIdentity,
    pub ani_levels: AniLevels,
    pub nars_truth: NarsTruth,
    pub nars_budget: NarsBudget,
    pub q_values: InlineQValues,
    pub rewards: InlineRewards,
    pub stdp: StdpMarkers,
    pub hebbian: InlineHebbian,

    // Block 15
    pub dn_addr: CompressedDnAddr,
    pub neighbors: NeighborBloom,
    pub metrics: GraphMetrics,
    pub edges: InlineEdges,
}

impl SchemaSidecar {
    pub const WORD_OFFSET: usize = 224;
    pub const WORD_COUNT: usize = 32;
    pub const SCHEMA_VERSION: u8 = 1;

    pub fn write_to_words(&self, words: &mut [u64]) {
        assert!(words.len() >= VECTOR_WORDS);

        // --- Block 14 (words 224-239) ---

        // [224] depth | rung | qidx | access_count
        words[224] = (self.identity.depth as u64)
            | ((self.identity.rung as u64) << 8)
            | ((self.identity.qidx as u64) << 16)
            | ((self.identity.access_count as u64) << 32);

        // [225] ttl | sigma_q | node_type
        words[225] = (self.identity.ttl as u64)
            | ((self.identity.sigma_q as u64) << 16)
            | ((self.identity.node_type.pack() as u64) << 32);

        // [226] label_hash
        words[226] = self.identity.label_hash;

        // [227] edge_type | version | reserved
        words[227] = (self.identity.edge_type.pack() as u64)
            | ((Self::SCHEMA_VERSION as u64) << 32);

        // [228-229] ANI levels
        let ani = self.ani_levels.pack();
        words[228] = ani as u64;
        words[229] = (ani >> 64) as u64;

        // [230] NARS truth (lower 32) + budget_lo (upper 32)
        let budget = self.nars_budget.pack();
        words[230] = (self.nars_truth.pack() as u64) | ((budget & 0xFFFFFFFF) << 32);

        // [231] budget_hi (lower 32) + reserved
        words[231] = budget >> 32;

        // [232-233] Q-values
        let q = self.q_values.pack();
        words[232] = q[0];
        words[233] = q[1];

        // [234-235] Rewards
        let r = self.rewards.pack();
        words[234] = r[0];
        words[235] = r[1];

        // [236-237] STDP
        let mut sw = [0u64; 2];
        for i in 0..4 { sw[0] |= (self.stdp.timestamps[i] as u64) << (i * 16); }
        for i in 0..4 { sw[1] |= (self.stdp.timestamps[i + 4] as u64) << (i * 16); }
        words[236] = sw[0];
        words[237] = sw[1];

        // [238-239] Hebbian
        let mut hw = [0u64; 2];
        for i in 0..4 { hw[0] |= (self.hebbian.weights[i] as u64) << (i * 16); }
        for i in 0..4 { hw[1] |= (self.hebbian.weights[i + 4] as u64) << (i * 16); }
        words[238] = hw[0];
        words[239] = hw[1];

        // --- Block 15 (words 240-255) ---

        // [240-243] DN address
        for i in 0..4 {
            let mut w = 0u64;
            for j in 0..8 { w |= (self.dn_addr.path[i * 8 + j] as u64) << (j * 8); }
            words[240 + i] = w;
        }

        // [244-247] Bloom
        for i in 0..4 { words[244 + i] = self.neighbors.words[i]; }

        // [248] Graph metrics
        words[248] = self.metrics.pack();

        // [249-255] Inline edges: 28 edges packed 4-per-word
        for wi in 0..7 {
            let mut w = 0u64;
            for ei in 0..4 {
                let idx = wi * 4 + ei;
                w |= (self.edges.edges[idx].pack() as u64) << (ei * 16);
            }
            words[249 + wi] = w;
        }
    }

    pub fn read_version(words: &[u64]) -> u8 {
        if words.len() < VECTOR_WORDS { return 0; }
        ((words[227] >> 32) & 0xFF) as u8
    }

    pub fn read_from_words(words: &[u64]) -> Self {
        assert!(words.len() >= VECTOR_WORDS);

        // --- Block 14 ---
        let identity = NodeIdentity {
            depth: words[224] as u8,
            rung: (words[224] >> 8) as u8,
            qidx: (words[224] >> 16) as u16,
            access_count: (words[224] >> 32) as u32,
            ttl: words[225] as u16,
            sigma_q: (words[225] >> 16) as u16,
            node_type: NodeTypeMarker::unpack((words[225] >> 32) as u32),
            label_hash: words[226],
            edge_type: EdgeTypeMarker::unpack(words[227] as u32),
        };

        let ani = (words[228] as u128) | ((words[229] as u128) << 64);
        let ani_levels = AniLevels::unpack(ani);

        let nars_truth = NarsTruth::unpack(words[230] as u32);
        let budget_lo = (words[230] >> 32) as u64;
        let budget_hi = (words[231] & 0xFFFFFFFF) as u64;
        let nars_budget = NarsBudget::unpack(budget_lo | (budget_hi << 32));

        let q_values = InlineQValues::unpack([words[232], words[233]]);

        let mut rewards = InlineRewards::default();
        for i in 0..4 { rewards.rewards[i] = ((words[234] >> (i * 16)) & 0xFFFF) as u16 as i16; }
        for i in 0..4 { rewards.rewards[i + 4] = ((words[235] >> (i * 16)) & 0xFFFF) as u16 as i16; }

        let mut stdp = StdpMarkers::default();
        for i in 0..4 { stdp.timestamps[i] = ((words[236] >> (i * 16)) & 0xFFFF) as u16; }
        for i in 0..4 { stdp.timestamps[i + 4] = ((words[237] >> (i * 16)) & 0xFFFF) as u16; }

        let mut hebbian = InlineHebbian::default();
        for i in 0..4 { hebbian.weights[i] = ((words[238] >> (i * 16)) & 0xFFFF) as u16; }
        for i in 0..4 { hebbian.weights[i + 4] = ((words[239] >> (i * 16)) & 0xFFFF) as u16; }

        // --- Block 15 ---
        let mut dn_addr = CompressedDnAddr::default();
        for i in 0..4 {
            for j in 0..8 {
                dn_addr.path[i * 8 + j] = ((words[240 + i] >> (j * 8)) & 0xFF) as u8;
            }
        }

        let mut neighbors = NeighborBloom::default();
        for i in 0..4 { neighbors.words[i] = words[244 + i]; }

        let metrics = GraphMetrics::unpack(words[248]);

        let mut edges = InlineEdges::default();
        for wi in 0..7 {
            for ei in 0..4 {
                let idx = wi * 4 + ei;
                let v = ((words[249 + wi] >> (ei * 16)) & 0xFFFF) as u16;
                edges.edges[idx] = InlineEdge::unpack(v);
            }
        }

        Self {
            identity, ani_levels, nars_truth, nars_budget,
            q_values, rewards, stdp, hebbian,
            dn_addr, neighbors, metrics, edges,
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_roundtrip() {
        let mut s = SchemaSidecar::default();
        s.identity.depth = 5;
        s.identity.rung = 2; // IMAGINE
        s.identity.qidx = 42;
        s.identity.access_count = 1234;
        s.identity.ttl = 500;
        s.identity.sigma_q = NodeIdentity::sigma_from_f32(12.5);
        s.identity.label_hash = 0xDEAD_BEEF_CAFE_BABE;
        s.identity.node_type.kind = NodeKind::Concept as u8;
        s.identity.edge_type.verb_id = 42;

        let mut w = [0u64; 256];
        s.write_to_words(&mut w);
        let r = SchemaSidecar::read_from_words(&w);

        assert_eq!(r.identity.depth, 5);
        assert_eq!(r.identity.rung, 2);
        assert_eq!(r.identity.qidx, 42);
        assert_eq!(r.identity.access_count, 1234);
        assert_eq!(r.identity.ttl, 500);
        assert_eq!(r.identity.label_hash, 0xDEAD_BEEF_CAFE_BABE);
        assert_eq!(r.identity.node_type.kind, NodeKind::Concept as u8);
        assert_eq!(r.identity.edge_type.verb_id, 42);
    }

    #[test]
    fn test_ani_pack_unpack() {
        let l = AniLevels { reactive: 100, memory: 200, analogy: 300, planning: 400,
            meta: 500, social: 600, creative: 700, r#abstract: 800 };
        assert_eq!(AniLevels::unpack(l.pack()), l);
    }

    #[test]
    fn test_nars_revision() {
        let t1 = NarsTruth::from_floats(0.8, 0.5);
        let t2 = NarsTruth::from_floats(0.6, 0.3);
        let r = t1.revision(&t2);
        assert!(r.c() > t1.c() || r.c() > t2.c());
    }

    #[test]
    fn test_q_values() {
        let mut q = InlineQValues::default();
        q.set_q(3, 0.75);
        q.set_q(7, -0.5);
        assert!((q.q(3) - 0.75).abs() < 0.02);
        assert!((q.q(7) + 0.5).abs() < 0.02);
        assert_eq!(q.best_action(), 3);
        assert_eq!(InlineQValues::unpack(q.pack()).values, q.values);
    }

    #[test]
    fn test_bloom() {
        let mut b = NeighborBloom::default();
        b.insert(42); b.insert(100); b.insert(999);
        assert!(b.might_contain(42));
        assert!(b.might_contain(100));
        assert!(b.might_contain(999));
    }

    #[test]
    fn test_graph_metrics() {
        let m = GraphMetrics { pagerank: 1000, hop_to_root: 3, cluster_id: 42,
            degree: 10, in_degree: 5, out_degree: 5 };
        assert_eq!(GraphMetrics::unpack(m.pack()), m);
    }

    #[test]
    fn test_inline_edges() {
        let mut s = SchemaSidecar::default();
        s.edges.push(7, 0x42); // verb 7, target 0x42
        s.edges.push(12, 0x80); // verb 12, target 0x80

        let mut w = [0u64; 256];
        s.write_to_words(&mut w);
        let r = SchemaSidecar::read_from_words(&w);

        assert_eq!(r.edges.count(), 2);
        assert_eq!(r.edges.edges[0].verb, 7);
        assert_eq!(r.edges.edges[0].target, 0x42);
        assert_eq!(r.edges.edges[1].verb, 12);
        assert_eq!(r.edges.edges[1].target, 0x80);
    }

    #[test]
    fn test_full_roundtrip() {
        let mut s = SchemaSidecar::default();
        s.identity.depth = 3;
        s.ani_levels.planning = 500;
        s.nars_truth = NarsTruth::from_floats(0.8, 0.6);
        s.nars_budget = NarsBudget::from_floats(0.9, 0.5, 0.7);
        s.q_values.set_q(0, 0.5);
        s.rewards.push(0.8);
        s.stdp.record_spike(42);
        s.hebbian.strengthen(0, 0.5);
        s.dn_addr.path[0] = 0x80;
        s.dn_addr.path[1] = 0x42;
        s.neighbors.insert(123);
        s.metrics.pagerank = 999;
        s.edges.push(1, 0x43);

        let mut w = [0u64; 256];
        s.write_to_words(&mut w);
        let r = SchemaSidecar::read_from_words(&w);

        assert_eq!(r.identity.depth, 3);
        assert_eq!(r.ani_levels.planning, 500);
        assert!((r.nars_budget.priority as f32 / 65535.0 - 0.9).abs() < 0.001);
        assert!((r.q_values.q(0) - 0.5).abs() < 0.02);
        assert_eq!(r.stdp.timestamps[7], 42);
        assert!(r.hebbian.weight(0) > 0.4);
        assert_eq!(r.dn_addr.path[0], 0x80);
        assert!(r.neighbors.might_contain(123));
        assert_eq!(r.metrics.pagerank, 999);
        assert_eq!(r.edges.edges[0].verb, 1);
    }

    #[test]
    fn test_version_byte() {
        let mut w = [0u64; 256];
        assert_eq!(SchemaSidecar::read_version(&w), 0);
        SchemaSidecar::default().write_to_words(&mut w);
        assert_eq!(SchemaSidecar::read_version(&w), 1);
    }

    #[test]
    fn test_metadata_stays_in_sidecar() {
        let mut w = [0xAAAAAAAAAAAAAAAAu64; 256];
        let mut s = SchemaSidecar::default();
        s.identity.depth = 255;
        s.ani_levels.planning = 65535;
        s.neighbors.insert(42);
        s.edges.push(7, 0xFF);
        s.write_to_words(&mut w);

        // Words 0-223 (resonance) must be untouched
        for i in 0..224 {
            assert_eq!(w[i], 0xAAAAAAAAAAAAAAAA,
                "Word {} was modified (resonance zone)", i);
        }
    }

    #[test]
    fn test_dn_addr_encodes_self_and_parent() {
        let mut s = SchemaSidecar::default();
        s.identity.depth = 3;
        s.dn_addr.path[0] = 0x80; // root
        s.dn_addr.path[1] = 0x42; // level 1
        s.dn_addr.path[2] = 0x43; // self (depth-1)

        // Self = path[depth-1]
        assert_eq!(s.dn_addr.path[s.identity.depth as usize - 1], 0x43);
        // Parent = path[depth-2]
        assert_eq!(s.dn_addr.path[s.identity.depth as usize - 2], 0x42);
    }
}
