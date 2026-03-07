//! SpoStore — Three-axis content-addressable graph store.
//!
//! Forward:  "What does Jan know?"   → scan X+Y → return Z matches
//! Reverse:  "Who knows Ada?"        → scan Z+Y → return X matches
//! Relation: "How are Jan and Ada related?" → scan X+Z → return Y matches
//! Causal:   "What does this feed?"  → scan Z→X chain links
//!
//! ## CLAM-pruned queries
//!
//! `ClamSpoIndex` replaces the Belichtung 7-point heuristic pre-filter
//! with ClamTree triangle inequality pruning (δ⁻ > ρ → skip cluster).
//! Build once via `ClamSpoIndex::build()`, then query via `sxp2o_clam()`.

use std::collections::BTreeMap;

use ladybug_contract::container::{Container, CONTAINER_BYTES, CONTAINER_WORDS};
use ladybug_contract::nars::TruthValue;
use ladybug_contract::record::CogRecord;

use super::merkle::SpoMerkle;
use super::scent::NibbleScent;
use super::semiring::SpoSemiring;
use super::sparse::{unpack_axes, AxisDescriptors, SparseContainer, SpoError};

// ============================================================================
// BELICHTUNG PREFILTER
// ============================================================================

/// Estimate Hamming distance from up to 7 sampled words. Returns true if the
/// estimated distance exceeds `threshold`, meaning this candidate should
/// be rejected without computing full Hamming distance.
///
/// Samples from BITMAP POSITIONS (words that actually contain data) rather
/// than fixed indices. For a SparseContainer with 30 stored words, sampling
/// from fixed positions like [0, 17, 37, 59, 79, 101, 123] would mostly
/// hit zeros and produce a useless estimate. Sampling from the union of
/// both containers' bitmaps ensures every sample is informative.
#[inline]
fn belichtung_reject(a: &SparseContainer, b: &SparseContainer, threshold: u32) -> bool {
    // Union of occupied positions in both containers
    let union_bm = [a.bitmap[0] | b.bitmap[0], a.bitmap[1] | b.bitmap[1]];
    let total = (union_bm[0].count_ones() + union_bm[1].count_ones()) as usize;

    if total == 0 {
        return false; // both empty → distance 0
    }

    // Step: skip this many set bits between samples. max(1, total/7)
    let step = (total / 7).max(1);

    let mut sample_diff = 0u32;
    let mut sampled = 0u32;
    let mut nth = 0usize;

    for half in 0..2usize {
        let mut w = union_bm[half];
        while w != 0 {
            let bit = w.trailing_zeros() as usize;
            w &= w - 1; // clear lowest set bit

            if nth % step == 0 && sampled < 7 {
                let pos = half * 64 + bit;
                let wa = a.get_word(pos);
                let wb = b.get_word(pos);
                sample_diff += (wa ^ wb).count_ones();
                sampled += 1;
            }
            nth += 1;
        }
    }

    if sampled == 0 {
        return false;
    }

    // Scale: sampled words out of 128 → estimated full distance
    (sample_diff as u64 * 128 / sampled as u64) as u32 > threshold
}

// ============================================================================
// QUERY TYPES
// ============================================================================

/// A query hit with DN, distance, and which axis matched.
#[derive(Clone, Debug)]
pub struct QueryHit {
    pub dn: u64,
    pub distance: u32,
    pub axis: QueryAxis,
}

/// A query hit enriched with NARS truth values.
#[derive(Clone, Debug)]
pub struct SpoHit {
    pub dn: u64,
    pub distance: u32,
    pub axis: QueryAxis,
    pub truth: ladybug_contract::nars::TruthValue,
}

// ============================================================================
// NARS TRUTH GATE
// ============================================================================

/// NARS truth gate — minimum frequency and confidence for an edge to exist
/// in a given truth-view. This is epistemic filtering, not propositional.
///
/// ```text
/// S×P→O(alice, causes, f>0.7, c>0.5) → only edges the system believes
///                                        with ≥70% frequency, ≥50% confidence
/// ```
#[derive(Clone, Copy, Debug)]
pub struct TruthGate {
    pub min_freq: f32,
    pub min_conf: f32,
}

impl TruthGate {
    /// No filtering — all edges pass.
    pub const OPEN: Self = Self { min_freq: 0.0, min_conf: 0.0 };
    /// Strong belief: f≥0.7, c≥0.5.
    pub const STRONG: Self = Self { min_freq: 0.7, min_conf: 0.5 };
    /// Near-certain: f≥0.9, c≥0.8.
    pub const CERTAIN: Self = Self { min_freq: 0.9, min_conf: 0.8 };

    /// Create a custom truth gate.
    pub fn new(min_freq: f32, min_conf: f32) -> Self {
        Self { min_freq, min_conf }
    }

    /// Does this truth value pass the gate?
    #[inline]
    pub fn passes(&self, freq: f32, conf: f32) -> bool {
        freq >= self.min_freq && conf >= self.min_conf
    }
}

/// Which axis (or combination) a query matched on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryAxis {
    X,  // Subject
    Y,  // Predicate
    Z,  // Object
    XY, // Forward query (Subject + Predicate)
    YZ, // Reverse query (Predicate + Object)
    XZ, // Relation query (Subject + Object)
}

// ============================================================================
// SPO STORE
// ============================================================================

/// Three-axis content-addressable graph store.
///
/// POC uses BTreeMap. Production replaces with LanceDB columnar store.
/// Merkle tree maintains authenticated state over the DN address space.
pub struct SpoStore {
    records: BTreeMap<u64, CogRecord>,
    /// XOR-Merkle tree for authenticated query results over the wire.
    merkle: SpoMerkle,
}

impl SpoStore {
    pub fn new() -> Self {
        Self {
            records: BTreeMap::new(),
            merkle: SpoMerkle::new(),
        }
    }

    pub fn insert(&mut self, record: CogRecord) -> Result<(), SpoError> {
        let dn = record.meta.words[0]; // W0 = DN address
        if self.records.contains_key(&dn) {
            return Err(SpoError::DuplicateDn { dn });
        }
        // Update Merkle tree with content fingerprint + NARS truth
        let nars = self.read_nars(&record);
        self.merkle.insert(dn, 0, record.content.as_bytes(), nars.frequency, nars.confidence);
        self.records.insert(dn, record);
        Ok(())
    }

    /// Remove a record by DN, updating the Merkle tree.
    pub fn remove(&mut self, dn: u64) -> Result<CogRecord, SpoError> {
        match self.records.remove(&dn) {
            Some(record) => {
                self.merkle.remove(dn);
                Ok(record)
            }
            None => Err(SpoError::DnNotFound { dn }),
        }
    }

    /// Get the Merkle tree (for authenticated query results).
    pub fn merkle(&self) -> &SpoMerkle {
        &self.merkle
    }

    /// Root hash of the Merkle tree — summarizes entire store state.
    pub fn root_hash(&self) -> super::merkle::MerkleHash {
        self.merkle.root_hash()
    }

    /// Verify a record's integrity against the Merkle tree.
    pub fn verify_integrity(&self, dn: u64) -> bool {
        if let Some(record) = self.records.get(&dn) {
            let nars = self.read_nars(record);
            self.merkle.verify(dn, record.content.as_bytes(), nars.frequency, nars.confidence)
        } else {
            false
        }
    }

    pub fn get(&self, dn: u64) -> Option<&CogRecord> {
        self.records.get(&dn)
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    // ========================================================================
    // THREE-AXIS QUERIES
    // ========================================================================

    /// Forward: "What does <src> <verb>?" → scan X+Y, return Z matches.
    pub fn query_forward(
        &self,
        src_fp: &Container,
        verb_fp: &Container,
        radius: u32,
    ) -> Vec<QueryHit> {
        let src_sparse = SparseContainer::from_dense(src_fp);
        let verb_sparse = SparseContainer::from_dense(verb_fp);
        let mut hits = Vec::new();

        for (&dn, record) in &self.records {
            if let Ok((x, y, _z)) = self.unpack_record(record) {
                // Belichtung: 7-point sample rejects ~90% before full Hamming
                if belichtung_reject(&src_sparse, &x, radius * 2) {
                    continue;
                }
                let dx = SparseContainer::hamming_sparse(&src_sparse, &x);
                let dy = SparseContainer::hamming_sparse(&verb_sparse, &y);
                // Combined distance: both subject and predicate must match
                let combined = dx.saturating_add(dy) / 2;
                if combined <= radius {
                    hits.push(QueryHit {
                        dn,
                        distance: combined,
                        axis: QueryAxis::XY,
                    });
                }
            }
        }

        hits.sort_by_key(|h| h.distance);
        hits
    }

    /// Reverse: "Who <verb>s <tgt>?" → scan Z+Y, return X matches.
    pub fn query_reverse(
        &self,
        tgt_fp: &Container,
        verb_fp: &Container,
        radius: u32,
    ) -> Vec<QueryHit> {
        let tgt_sparse = SparseContainer::from_dense(tgt_fp);
        let verb_sparse = SparseContainer::from_dense(verb_fp);
        let mut hits = Vec::new();

        for (&dn, record) in &self.records {
            if let Ok((_x, y, z)) = self.unpack_record(record) {
                // Belichtung: cheap rejection on primary axis
                if belichtung_reject(&tgt_sparse, &z, radius * 2) {
                    continue;
                }
                let dz = SparseContainer::hamming_sparse(&tgt_sparse, &z);
                let dy = SparseContainer::hamming_sparse(&verb_sparse, &y);
                let combined = dz.saturating_add(dy) / 2;
                if combined <= radius {
                    hits.push(QueryHit {
                        dn,
                        distance: combined,
                        axis: QueryAxis::YZ,
                    });
                }
            }
        }

        hits.sort_by_key(|h| h.distance);
        hits
    }

    /// Relation: "How are <src> and <tgt> related?" → scan X+Z, return Y matches.
    pub fn query_relation(
        &self,
        src_fp: &Container,
        tgt_fp: &Container,
        radius: u32,
    ) -> Vec<QueryHit> {
        let src_sparse = SparseContainer::from_dense(src_fp);
        let tgt_sparse = SparseContainer::from_dense(tgt_fp);
        let mut hits = Vec::new();

        for (&dn, record) in &self.records {
            if let Ok((x, _y, z)) = self.unpack_record(record) {
                // Belichtung: cheap rejection on X axis
                if belichtung_reject(&src_sparse, &x, radius * 2) {
                    continue;
                }
                let dx = SparseContainer::hamming_sparse(&src_sparse, &x);
                let dz = SparseContainer::hamming_sparse(&tgt_sparse, &z);
                let combined = dx.saturating_add(dz) / 2;
                if combined <= radius {
                    hits.push(QueryHit {
                        dn,
                        distance: combined,
                        axis: QueryAxis::XZ,
                    });
                }
            }
        }

        hits.sort_by_key(|h| h.distance);
        hits
    }

    /// Content: match against any axis.
    pub fn query_content(
        &self,
        query: &Container,
        radius: u32,
    ) -> Vec<QueryHit> {
        let q_sparse = SparseContainer::from_dense(query);
        let mut hits = Vec::new();

        for (&dn, record) in &self.records {
            if let Ok((x, y, z)) = self.unpack_record(record) {
                let dx = SparseContainer::hamming_sparse(&q_sparse, &x);
                let dy = SparseContainer::hamming_sparse(&q_sparse, &y);
                let dz = SparseContainer::hamming_sparse(&q_sparse, &z);

                let (best_dist, best_axis) = if dx <= dy && dx <= dz {
                    (dx, QueryAxis::X)
                } else if dy <= dz {
                    (dy, QueryAxis::Y)
                } else {
                    (dz, QueryAxis::Z)
                };

                if best_dist <= radius {
                    hits.push(QueryHit {
                        dn,
                        distance: best_dist,
                        axis: best_axis,
                    });
                }
            }
        }

        hits.sort_by_key(|h| h.distance);
        hits
    }

    // ========================================================================
    // CAUSAL CHAIN DISCOVERY
    // ========================================================================

    /// Find records whose X axis resonates with `record`'s Z axis.
    /// These are causal successors: things this record feeds into.
    pub fn causal_successors(
        &self,
        record: &CogRecord,
        radius: u32,
    ) -> Vec<QueryHit> {
        if let Ok((_x, _y, z)) = self.unpack_record(record) {
            let source_dn = record.meta.words[0];
            let mut hits = Vec::new();

            for (&dn, other) in &self.records {
                if dn == source_dn { continue; } // skip self
                if let Ok((ox, _oy, _oz)) = self.unpack_record(other) {
                    let dist = SparseContainer::hamming_sparse(&z, &ox);
                    if dist <= radius {
                        hits.push(QueryHit {
                            dn,
                            distance: dist,
                            axis: QueryAxis::X,
                        });
                    }
                }
            }

            hits.sort_by_key(|h| h.distance);
            hits
        } else {
            Vec::new()
        }
    }

    /// Find records whose Z axis resonates with `record`'s X axis.
    /// These are causal predecessors: things that feed into this record.
    pub fn causal_predecessors(
        &self,
        record: &CogRecord,
        radius: u32,
    ) -> Vec<QueryHit> {
        if let Ok((x, _y, _z)) = self.unpack_record(record) {
            let source_dn = record.meta.words[0];
            let mut hits = Vec::new();

            for (&dn, other) in &self.records {
                if dn == source_dn { continue; }
                if let Ok((_ox, _oy, oz)) = self.unpack_record(other) {
                    let dist = SparseContainer::hamming_sparse(&x, &oz);
                    if dist <= radius {
                        hits.push(QueryHit {
                            dn,
                            distance: dist,
                            axis: QueryAxis::Z,
                        });
                    }
                }
            }

            hits.sort_by_key(|h| h.distance);
            hits
        } else {
            Vec::new()
        }
    }

    /// Walk a causal chain forward from `start`, max `depth` hops.
    /// Returns one Vec per hop level.
    pub fn walk_chain_forward(
        &self,
        start: &CogRecord,
        radius: u32,
        depth: usize,
    ) -> Vec<Vec<QueryHit>> {
        let mut chain = Vec::new();
        let mut current = vec![start.meta.words[0]]; // start DNs

        for _ in 0..depth {
            let mut level_hits = Vec::new();
            for &dn in &current {
                if let Some(record) = self.get(dn) {
                    let successors = self.causal_successors(record, radius);
                    level_hits.extend(successors);
                }
            }
            if level_hits.is_empty() { break; }
            current = level_hits.iter().map(|h| h.dn).collect();
            chain.push(level_hits);
        }

        chain
    }

    /// Walk a causal chain forward with pluggable semiring algebra.
    ///
    /// Instead of hardcoded XOR-bind + bundle, this uses the semiring's
    /// `multiply` (⊗) for edge traversal and `add` (⊕) for path accumulation.
    ///
    /// Returns (dn, accumulated_value) pairs per hop level.
    pub fn walk_chain_semiring<S: SpoSemiring>(
        &self,
        start: &CogRecord,
        semiring: &S,
        init: S::Value,
        radius: u32,
        depth: usize,
    ) -> Vec<Vec<(u64, S::Value)>> {
        let mut chain = Vec::new();
        let mut frontier: Vec<(u64, S::Value)> = vec![(start.meta.words[0], init)];

        // Reusable buffer — avoids 2KB Container::zero() + to_dense() per iteration.
        let mut edge_buf = Container::zero();

        for _ in 0..depth {
            let mut next_level = Vec::new();
            for (dn, value) in &frontier {
                if let Some(record) = self.get(*dn) {
                    // Get the edge fingerprint (Z axis = what this record feeds)
                    if let Ok((_x, _y, z)) = self.unpack_record(record) {
                        // Write sparse into reusable buffer (zero-alloc)
                        edge_buf.words.fill(0);
                        let mut wi = 0;
                        for i in 0..CONTAINER_WORDS {
                            let half = i / 64;
                            let bit = i % 64;
                            if z.bitmap[half] & (1u64 << bit) != 0 {
                                edge_buf.words[i] = z.words[wi];
                                wi += 1;
                            }
                        }
                        let new_value = semiring.multiply(&edge_buf, value);

                        // Find successors: records whose X resonates with our Z
                        let successors = self.causal_successors(record, radius);
                        for hit in successors {
                            next_level.push((hit.dn, new_value.clone()));
                        }
                    }
                }
            }
            if next_level.is_empty() { break; }

            // Merge duplicate DNs using semiring.add (⊕)
            let mut merged: BTreeMap<u64, S::Value> = BTreeMap::new();
            for (dn, val) in next_level {
                merged
                    .entry(dn)
                    .and_modify(|existing| *existing = semiring.add(existing, &val))
                    .or_insert(val);
            }

            let level: Vec<(u64, S::Value)> = merged.into_iter().collect();
            frontier = level.clone();
            chain.push(level);
        }

        chain
    }

    /// Compute chain coherence: product of normalized link coherences.
    ///
    /// coherence_per_link = 1.0 - (hamming(Z_i, X_{i+1}) / CONTAINER_BITS)
    /// chain_coherence = product of all link coherences
    pub fn chain_coherence(&self, dns: &[u64]) -> f32 {
        if dns.len() < 2 { return 1.0; }

        let mut coherence = 1.0f32;
        for window in dns.windows(2) {
            let a = match self.get(window[0]) { Some(r) => r, None => return 0.0 };
            let b = match self.get(window[1]) { Some(r) => r, None => return 0.0 };

            let (_, _, z_a) = match self.unpack_record(a) { Ok(v) => v, Err(_) => return 0.0 };
            let (x_b, _, _) = match self.unpack_record(b) { Ok(v) => v, Err(_) => return 0.0 };

            let dist = SparseContainer::hamming_sparse(&z_a, &x_b);
            let link_coherence = 1.0 - (dist as f32 / ladybug_contract::container::CONTAINER_BITS as f32);
            coherence *= link_coherence;
        }

        coherence
    }

    // ========================================================================
    // NARS CHAIN DEDUCTION
    // ========================================================================

    /// Deduction along a causal chain with coherence-weighted confidence.
    ///
    /// f_chain = f₁ × f₂ × ... × fₙ
    /// c_chain = c₁ × c₂ × ... × cₙ × coherence₁₂ × coherence₂₃ × ...
    pub fn chain_deduction(&self, dns: &[u64]) -> TruthValue {
        if dns.is_empty() { return TruthValue::unknown(); }
        if dns.len() == 1 {
            return self.get(dns[0])
                .map(|r| self.read_nars(r))
                .unwrap_or(TruthValue::unknown());
        }

        let mut f_chain = 1.0f32;
        let mut c_chain = 1.0f32;

        for (i, &dn) in dns.iter().enumerate() {
            let record = match self.get(dn) { Some(r) => r, None => return TruthValue::unknown() };
            let nars = self.read_nars(record);
            f_chain *= nars.frequency;
            c_chain *= nars.confidence;

            // Multiply by coherence factor for each Z→X link
            if i + 1 < dns.len() {
                let next = match self.get(dns[i + 1]) { Some(r) => r, None => return TruthValue::unknown() };
                let (_, _, z) = match self.unpack_record(record) { Ok(v) => v, Err(_) => return TruthValue::unknown() };
                let (x, _, _) = match self.unpack_record(next) { Ok(v) => v, Err(_) => return TruthValue::unknown() };
                let dist = SparseContainer::hamming_sparse(&z, &x);
                let link_coh = 1.0 - (dist as f32 / ladybug_contract::container::CONTAINER_BITS as f32);
                c_chain *= link_coh;
            }
        }

        TruthValue::new(
            f_chain.clamp(0.0, 1.0),
            c_chain.clamp(0.0, 1.0),
        )
    }

    // ========================================================================
    // SPO META QUERY — NARS-GATED AXIS PROJECTIONS
    // ========================================================================

    /// S×P→O: Fix Subject + Predicate, project Objects.
    /// NARS truth gate filters edges before distance ranking.
    ///
    /// ```text
    /// S×P→O(alice, causes, r=500, f>0.7, c>0.5)
    ///   → only objects that alice causes with strong belief
    /// ```
    pub fn sxp2o(
        &self,
        src_fp: &Container,
        verb_fp: &Container,
        radius: u32,
        gate: TruthGate,
    ) -> Vec<SpoHit> {
        let src_sparse = SparseContainer::from_dense(src_fp);
        let verb_sparse = SparseContainer::from_dense(verb_fp);
        let mut hits = Vec::new();

        for (&dn, record) in &self.records {
            // NARS gate FIRST — cheapest filter (2 float compares)
            let nars = self.read_nars(record);
            if !gate.passes(nars.frequency, nars.confidence) {
                continue;
            }

            if let Ok((x, y, _z)) = self.unpack_record(record) {
                if belichtung_reject(&src_sparse, &x, radius * 2) {
                    continue;
                }
                let dx = SparseContainer::hamming_sparse(&src_sparse, &x);
                let dy = SparseContainer::hamming_sparse(&verb_sparse, &y);
                let combined = dx.saturating_add(dy) / 2;
                if combined <= radius {
                    hits.push(SpoHit {
                        dn,
                        distance: combined,
                        axis: QueryAxis::XY,
                        truth: nars,
                    });
                }
            }
        }

        hits.sort_by_key(|h| h.distance);
        hits
    }

    /// S×O→P: Fix Subject + Object, project Predicates.
    /// "How are alice and bob related?" — with truth gating.
    pub fn sxo2p(
        &self,
        src_fp: &Container,
        tgt_fp: &Container,
        radius: u32,
        gate: TruthGate,
    ) -> Vec<SpoHit> {
        let src_sparse = SparseContainer::from_dense(src_fp);
        let tgt_sparse = SparseContainer::from_dense(tgt_fp);
        let mut hits = Vec::new();

        for (&dn, record) in &self.records {
            let nars = self.read_nars(record);
            if !gate.passes(nars.frequency, nars.confidence) {
                continue;
            }

            if let Ok((x, _y, z)) = self.unpack_record(record) {
                if belichtung_reject(&src_sparse, &x, radius * 2) {
                    continue;
                }
                let dx = SparseContainer::hamming_sparse(&src_sparse, &x);
                let dz = SparseContainer::hamming_sparse(&tgt_sparse, &z);
                let combined = dx.saturating_add(dz) / 2;
                if combined <= radius {
                    hits.push(SpoHit {
                        dn,
                        distance: combined,
                        axis: QueryAxis::XZ,
                        truth: nars,
                    });
                }
            }
        }

        hits.sort_by_key(|h| h.distance);
        hits
    }

    /// P×O→S: Fix Predicate + Object, project Subjects.
    /// "Who/what causes rain?" — with truth gating.
    pub fn pxo2s(
        &self,
        verb_fp: &Container,
        tgt_fp: &Container,
        radius: u32,
        gate: TruthGate,
    ) -> Vec<SpoHit> {
        let verb_sparse = SparseContainer::from_dense(verb_fp);
        let tgt_sparse = SparseContainer::from_dense(tgt_fp);
        let mut hits = Vec::new();

        for (&dn, record) in &self.records {
            let nars = self.read_nars(record);
            if !gate.passes(nars.frequency, nars.confidence) {
                continue;
            }

            if let Ok((_x, y, z)) = self.unpack_record(record) {
                if belichtung_reject(&tgt_sparse, &z, radius * 2) {
                    continue;
                }
                let dy = SparseContainer::hamming_sparse(&verb_sparse, &y);
                let dz = SparseContainer::hamming_sparse(&tgt_sparse, &z);
                let combined = dy.saturating_add(dz) / 2;
                if combined <= radius {
                    hits.push(SpoHit {
                        dn,
                        distance: combined,
                        axis: QueryAxis::YZ,
                        truth: nars,
                    });
                }
            }
        }

        hits.sort_by_key(|h| h.distance);
        hits
    }

    // ========================================================================
    // SCENT PRE-FILTER
    // ========================================================================

    /// Filter records by scent distance before expensive Hamming scan.
    pub fn scent_prefilter(
        &self,
        query_scent: &NibbleScent,
        max_distance: u32,
    ) -> Vec<u64> {
        self.records.iter()
            .filter_map(|(&dn, record)| {
                let record_scent = self.read_scent(record);
                if record_scent.distance(query_scent) <= max_distance {
                    Some(dn)
                } else {
                    None
                }
            })
            .collect()
    }

    // ========================================================================
    // INTERNAL HELPERS
    // ========================================================================

    fn unpack_record(
        &self,
        record: &CogRecord,
    ) -> Result<(SparseContainer, SparseContainer, SparseContainer), SpoError> {
        let desc = self.read_axis_descriptors(record);
        unpack_axes(&record.content, &desc)
    }

    fn read_axis_descriptors(&self, record: &CogRecord) -> AxisDescriptors {
        AxisDescriptors::from_words(&[record.meta.words[34], record.meta.words[35]])
    }

    fn read_nars(&self, record: &CogRecord) -> TruthValue {
        let freq = f32::from_bits(record.meta.words[4] as u32);
        let conf = f32::from_bits(record.meta.words[5] as u32);
        TruthValue::new(
            freq.clamp(0.0, 1.0),
            conf.clamp(0.0, 1.0),
        )
    }

    fn read_scent(&self, record: &CogRecord) -> NibbleScent {
        NibbleScent::from_words(&[
            record.meta.words[12],
            record.meta.words[13],
            record.meta.words[14],
            record.meta.words[15],
            record.meta.words[16],
            record.meta.words[17],
        ])
    }
}

impl Default for SpoStore {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// AUTHENTICATED QUERY WRAPPERS
// ============================================================================

impl SpoStore {
    /// Forward query with authenticated result (includes Merkle proofs).
    pub fn query_forward_authenticated(
        &self,
        src_fp: &Container,
        verb_fp: &Container,
        radius: u32,
    ) -> super::merkle::AuthenticatedResult {
        let hits = self.query_forward(src_fp, verb_fp, radius);
        super::merkle::AuthenticatedResult::from_query(hits, &self.merkle)
    }

    /// Reverse query with authenticated result.
    pub fn query_reverse_authenticated(
        &self,
        tgt_fp: &Container,
        verb_fp: &Container,
        radius: u32,
    ) -> super::merkle::AuthenticatedResult {
        let hits = self.query_reverse(tgt_fp, verb_fp, radius);
        super::merkle::AuthenticatedResult::from_query(hits, &self.merkle)
    }

    /// Relation query with authenticated result.
    pub fn query_relation_authenticated(
        &self,
        src_fp: &Container,
        tgt_fp: &Container,
        radius: u32,
    ) -> super::merkle::AuthenticatedResult {
        let hits = self.query_relation(src_fp, tgt_fp, radius);
        super::merkle::AuthenticatedResult::from_query(hits, &self.merkle)
    }
}

//
// CLAM-PRUNED SPO INDEX
// ============================================================================

/// CLAM tree index over X-axis (Subject) fingerprints.
///
/// Replaces Belichtung 7-point heuristic with ClamTree triangle inequality.
/// rho_nn prunes clusters where δ⁻ = max(0, d(q,center) - radius) > ρ,
/// guaranteeing zero false negatives (exact metric pruning).
///
/// ## Violation gate compliance
///
/// - Gate 1: No shadow structure — index is a read-only view, SpoStore owns data.
/// - Gate 3: No SparseContainer::from_dense() in query path. Query operates on
///   raw bytes via ClamTree SIMD Hamming. Y-axis check uses `hamming_dense_vs_sparse()`.
/// - Gate 4: O(⌈d⌉·log 𝒩) via DFS pruning vs O(n) linear scan.
/// - Gate 7: Uses rustynum_clam::search::rho_nn (existing primitive).
pub struct ClamSpoIndex {
    tree: rustynum_clam::tree::ClamTree,
    /// Flat byte buffer: X-axis dense fingerprints concatenated.
    /// Layout: [record_0: CONTAINER_BYTES] [record_1: CONTAINER_BYTES] ...
    data: Vec<u8>,
    /// Maps ClamTree point index → record DN.
    dn_map: Vec<u64>,
}

/// Number of words the SparseContainer bitmap can track.
/// SparseContainer.bitmap is `[u64; 2]` = 128 bits = 128 word positions.
const SPARSE_BITMAP_CAPACITY: usize = 128;

/// Hamming distance between a dense Container and a SparseContainer.
///
/// No allocation — walks the sparse bitmap directly. This avoids the
/// Gate 3 prohibited `SparseContainer::from_dense()` in the query path.
///
/// Note: SparseContainer bitmap covers 128 word positions (bitmap: [u64; 2]).
/// Words beyond position 127 in the dense Container contribute their full
/// popcount (since the sparse side is zero there).
#[inline]
fn hamming_dense_vs_sparse(dense: &Container, sparse: &SparseContainer) -> u32 {
    let mut dist = 0u32;
    let mut word_idx = 0usize;

    // Walk the bitmap-covered region (128 words)
    for i in 0..SPARSE_BITMAP_CAPACITY {
        let half = i / 64;
        let bit = i % 64;
        if sparse.bitmap[half] & (1u64 << bit) != 0 {
            // Word present in sparse: XOR with dense word
            dist += (dense.words[i] ^ sparse.words[word_idx]).count_ones();
            word_idx += 1;
        } else {
            // Word absent in sparse (=0): all set bits in dense contribute
            dist += dense.words[i].count_ones();
        }
    }

    // Words beyond bitmap range: sparse side is zero, full popcount
    for i in SPARSE_BITMAP_CAPACITY..CONTAINER_WORDS {
        dist += dense.words[i].count_ones();
    }

    dist
}

/// Effective vector length in bytes for the sparse-compatible region.
/// SparseContainer bitmap tracks 128 words × 8 bytes = 1024 bytes.
const SPARSE_VEC_LEN: usize = SPARSE_BITMAP_CAPACITY * 8;

/// Write a SparseContainer's data as dense bytes into a fixed-size buffer.
/// Only covers the 128-word bitmap region (1024 bytes). No to_dense() call.
fn sparse_to_bytes(sparse: &SparseContainer, buf: &mut [u8; SPARSE_VEC_LEN]) {
    buf.fill(0);
    let mut word_idx = 0;
    for i in 0..SPARSE_BITMAP_CAPACITY {
        let half = i / 64;
        let bit = i % 64;
        if sparse.bitmap[half] & (1u64 << bit) != 0 {
            let start = i * 8;
            buf[start..start + 8].copy_from_slice(&sparse.words[word_idx].to_ne_bytes());
            word_idx += 1;
        }
    }
}

impl ClamSpoIndex {
    /// Build a CLAM tree index over X-axis data of all records.
    ///
    /// One-time cost: converts X-axis sparse containers into a flat byte
    /// buffer for ClamTree construction. After build, queries are O(log n).
    ///
    /// Uses SPARSE_VEC_LEN (1024 bytes) per vector — the region tracked by
    /// the SparseContainer bitmap (128 words × 8 bytes).
    pub fn build(store: &SpoStore) -> Self {
        let vec_len = SPARSE_VEC_LEN;
        let mut data = Vec::with_capacity(store.records.len() * vec_len);
        let mut dn_map = Vec::with_capacity(store.records.len());

        for (&dn, record) in &store.records {
            let desc = AxisDescriptors::from_words(&[
                record.meta.words[34],
                record.meta.words[35],
            ]);
            if let Ok((x_sparse, _, _)) = unpack_axes(&record.content, &desc) {
                let mut buf = [0u8; SPARSE_VEC_LEN];
                sparse_to_bytes(&x_sparse, &mut buf);
                data.extend_from_slice(&buf);
                dn_map.push(dn);
            }
        }

        let count = dn_map.len();
        let config = rustynum_clam::tree::BuildConfig::default();
        let tree = if count > 0 {
            rustynum_clam::tree::ClamTree::build(&data, vec_len, count, &config)
        } else {
            // Empty tree for empty store — build with 1 zero vector
            let zero = vec![0u8; vec_len];
            rustynum_clam::tree::ClamTree::build(&zero, vec_len, 1, &config)
        };

        Self { tree, data, dn_map }
    }

    /// Subject × Predicate → Object: CLAM-pruned forward query.
    ///
    /// Phase 1: rho_nn on X-axis ClamTree finds candidates where
    ///          δ⁻ = max(0, d(q,center) - radius) ≤ ρ.
    /// Phase 2: Only surviving candidates are checked for Y-axis proximity.
    ///
    /// Returns hits sorted by combined (X+Y)/2 distance.
    pub fn sxp2o_clam(
        &self,
        store: &SpoStore,
        src_fp: &Container,
        verb_fp: &Container,
        radius: u32,
    ) -> Vec<QueryHit> {
        if self.dn_map.is_empty() {
            return Vec::new();
        }

        // Phase 1: rho_nn on X-axis — prune via triangle inequality.
        // Convert first 128 words of query Container to bytes (stack alloc).
        let mut query_bytes = [0u8; SPARSE_VEC_LEN];
        for i in 0..SPARSE_BITMAP_CAPACITY {
            let start = i * 8;
            query_bytes[start..start + 8].copy_from_slice(&src_fp.words[i].to_ne_bytes());
        }

        // Use 2× radius for X-axis alone since combined is (dx+dy)/2 ≤ radius
        // → dx can be up to 2×radius if dy=0. This ensures no false negatives.
        let x_rho = (radius as u64).saturating_mul(2);
        let rho_result = rustynum_clam::search::rho_nn(
            &self.tree,
            &self.data,
            SPARSE_VEC_LEN,
            &query_bytes,
            x_rho,
        );

        // Phase 2: check combined X+Y distance on candidates only.
        let mut hits = Vec::new();
        for (orig_idx, x_dist) in rho_result.hits {
            let dn = self.dn_map[orig_idx];
            if let Some(record) = store.get(dn) {
                let desc = AxisDescriptors::from_words(&[
                    record.meta.words[34],
                    record.meta.words[35],
                ]);
                if let Ok((_, y_sparse, _)) = unpack_axes(&record.content, &desc) {
                    // Gate 3: No SparseContainer::from_dense(verb_fp).
                    // hamming_dense_vs_sparse walks bitmap directly.
                    let y_dist = hamming_dense_vs_sparse(verb_fp, &y_sparse);
                    let combined = (x_dist as u32).saturating_add(y_dist) / 2;
                    if combined <= radius {
                        hits.push(QueryHit {
                            dn,
                            distance: combined,
                            axis: QueryAxis::XY,
                        });
                    }
                }
            }
        }

        hits.sort_by_key(|h| h.distance);
        hits
    }

    /// Number of records indexed.
    pub fn len(&self) -> usize {
        self.dn_map.len()
    }

    /// Is the index empty?
    pub fn is_empty(&self) -> bool {
        self.dn_map.is_empty()
    }

    /// Pruning statistics from the last rho_nn call (diagnostic).
    /// Re-runs the query to gather stats — use only for diagnostics.
    pub fn pruning_stats(
        &self,
        src_fp: &Container,
        radius: u32,
    ) -> (usize, usize) {
        if self.dn_map.is_empty() {
            return (0, 0);
        }
        let mut query_bytes = [0u8; SPARSE_VEC_LEN];
        for i in 0..SPARSE_BITMAP_CAPACITY {
            let start = i * 8;
            query_bytes[start..start + 8].copy_from_slice(&src_fp.words[i].to_ne_bytes());
        }
        let x_rho = (radius as u64).saturating_mul(2);
        let result = rustynum_clam::search::rho_nn(
            &self.tree,
            &self.data,
            SPARSE_VEC_LEN,
            &query_bytes,
            x_rho,
        );
        (result.distance_calls, result.clusters_pruned)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::sparse::pack_axes;
    use ladybug_contract::container::Container;
    use ladybug_contract::record::CogRecord;

    /// Helper: create a CogRecord with sparse-packed X, Y, Z axes.
    fn make_spo_record(dn: u64, x: &Container, y: &Container, z: &Container) -> CogRecord {
        // Use low-density versions to fit in one content Container
        let sx_low = sparse_low_density(x, 30);
        let sy_low = sparse_low_density(y, 30);
        let sz_low = sparse_low_density(z, 30);

        let (content, desc) = pack_axes(&sx_low, &sy_low, &sz_low).unwrap();
        let desc_words = desc.to_words();

        let mut record = CogRecord::default();
        record.meta.words[0] = dn;
        record.meta.words[34] = desc_words[0];
        record.meta.words[35] = desc_words[1];
        record.content = content;
        record
    }

    /// Create a sparse container with only the first `n` words from a dense container.
    fn sparse_low_density(c: &Container, n: usize) -> SparseContainer {
        let mut bitmap = [0u64; 2];
        let mut words = Vec::new();
        for i in 0..n.min(CONTAINER_WORDS) {
            if c.words[i] != 0 {
                let half = i / 64;
                let bit = i % 64;
                bitmap[half] |= 1u64 << bit;
                words.push(c.words[i]);
            }
        }
        SparseContainer { bitmap, words }
    }

    #[test]
    fn test_hamming_dense_vs_sparse_zero() {
        let dense = Container::zero();
        let sparse = SparseContainer::zero();
        assert_eq!(hamming_dense_vs_sparse(&dense, &sparse), 0);
    }

    #[test]
    fn test_hamming_dense_vs_sparse_equivalence() {
        let a = Container::random(42);
        let b = Container::random(99);
        // Use sparse_low_density (bitmap covers 128 words max)
        let sb = sparse_low_density(&b, SPARSE_BITMAP_CAPACITY);
        let sa = sparse_low_density(&a, SPARSE_BITMAP_CAPACITY);

        // hamming_dense_vs_sparse should equal SparseContainer::hamming_sparse
        // for the 128-word region tracked by the bitmap
        assert_eq!(
            hamming_dense_vs_sparse(&a, &sb),
            SparseContainer::hamming_sparse(&sa, &sb)
                // + popcount of dense words[128..256] (sparse treats as zero)
                + (128..CONTAINER_WORDS).map(|i| a.words[i].count_ones()).sum::<u32>()
        );
    }

    #[test]
    fn test_hamming_dense_vs_sparse_self() {
        let c = Container::random(7);
        let sc = sparse_low_density(&c, SPARSE_BITMAP_CAPACITY);
        // Distance is NOT zero because sparse only captures first 128 words.
        // The remaining 128 words in dense contribute their popcount.
        let tail_popcount: u32 = (128..CONTAINER_WORDS)
            .map(|i| c.words[i].count_ones())
            .sum();
        assert_eq!(hamming_dense_vs_sparse(&c, &sc), tail_popcount);
    }

    #[test]
    fn test_clam_spo_index_empty_store() {
        let store = SpoStore::new();
        let index = ClamSpoIndex::build(&store);
        assert!(index.is_empty());

        let src = Container::random(1);
        let verb = Container::random(2);
        let hits = index.sxp2o_clam(&store, &src, &verb, 10000);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_clam_spo_index_finds_close_record() {
        let mut store = SpoStore::new();

        // Insert a record with known X axis
        let x = Container::random(1);
        let y = Container::random(2);
        let z = Container::random(3);
        let record = make_spo_record(0x100, &x, &y, &z);
        store.insert(record).unwrap();

        let index = ClamSpoIndex::build(&store);
        assert_eq!(index.len(), 1);

        // Query with the same X and Y — should find at distance 0
        let hits = index.sxp2o_clam(&store, &x, &y, 20000);
        // The record should appear (distance depends on sparse truncation)
        assert!(!hits.is_empty(), "should find at least one hit");
        assert_eq!(hits[0].dn, 0x100);
    }

    #[test]
    fn test_clam_spo_index_respects_radius() {
        let mut store = SpoStore::new();

        let x = Container::random(10);
        let y = Container::random(20);
        let z = Container::random(30);
        let record = make_spo_record(0x200, &x, &y, &z);
        store.insert(record).unwrap();

        let index = ClamSpoIndex::build(&store);

        // Query with a very different X — should not find with tight radius
        let far_x = Container::random(999);
        let hits = index.sxp2o_clam(&store, &far_x, &y, 100);
        // With random fingerprints and radius 100, very unlikely to match
        // (expected distance ~8192 for random 16384-bit vectors)
        assert!(hits.is_empty(), "far query should find nothing at radius 100");
    }

    #[test]
    fn test_clam_spo_index_preserves_query_axis() {
        let mut store = SpoStore::new();

        let x = Container::random(1);
        let y = Container::random(2);
        let z = Container::random(3);
        store.insert(make_spo_record(0x300, &x, &y, &z)).unwrap();

        let index = ClamSpoIndex::build(&store);
        let hits = index.sxp2o_clam(&store, &x, &y, 20000);
        for hit in &hits {
            assert_eq!(hit.axis, QueryAxis::XY, "sxp2o_clam always returns XY axis");
        }
    }

    #[test]
    fn test_clam_spo_pruning_stats() {
        let mut store = SpoStore::new();

        // Insert several records
        for i in 0..10u64 {
            let x = Container::random(i);
            let y = Container::random(i + 100);
            let z = Container::random(i + 200);
            store.insert(make_spo_record(i, &x, &y, &z)).unwrap();
        }

        let index = ClamSpoIndex::build(&store);
        let query = Container::random(42);
        let (distance_calls, _clusters_pruned) = index.pruning_stats(&query, 100);

        // With tight radius and random data, tree should prune most clusters
        // At minimum, the root cluster center is checked
        assert!(distance_calls > 0, "should compute at least one distance");
    }
}
