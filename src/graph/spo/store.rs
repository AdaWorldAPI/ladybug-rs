//! SpoStore — Three-axis content-addressable graph store.
//!
//! Forward:  "What does Jan know?"   → scan X+Y → return Z matches
//! Reverse:  "Who knows Ada?"        → scan Z+Y → return X matches
//! Relation: "How are Jan and Ada related?" → scan X+Z → return Y matches
//! Causal:   "What does this feed?"  → scan Z→X chain links

use std::collections::BTreeMap;

use ladybug_contract::container::Container;
use ladybug_contract::nars::TruthValue;
use ladybug_contract::record::CogRecord;

use super::merkle::SpoMerkle;
use super::scent::NibbleScent;
use super::semiring::SpoSemiring;
use super::sparse::{unpack_axes, AxisDescriptors, SparseContainer, SpoError};

// ============================================================================
// BELICHTUNG PREFILTER
// ============================================================================

/// 7 prime-spaced sample points across the 128-word bitmap range.
/// 14 cycles to estimate Hamming distance ± 15%. Rejects ~90% of candidates.
/// (SparseContainer bitmap is [u64; 2] = 128 bits, so indices must be < 128.)
const BELICHTUNG_SAMPLES: [usize; 7] = [0, 17, 37, 59, 79, 101, 123];

/// Estimate Hamming distance from 7 sampled words. Returns true if the
/// estimated distance exceeds `threshold`, meaning this candidate should
/// be rejected without computing full Hamming distance.
///
/// Scale factor: 7 words out of 128 → multiply by 128/7 ≈ 18.3.
/// Use 18 (conservative, slight underestimate) to avoid false negatives.
#[inline]
fn belichtung_reject(a: &SparseContainer, b: &SparseContainer, threshold: u32) -> bool {
    let mut sample_diff = 0u32;
    for &idx in &BELICHTUNG_SAMPLES {
        let wa = a.get_word(idx);
        let wb = b.get_word(idx);
        sample_diff += (wa ^ wb).count_ones();
    }
    (sample_diff * 18) > threshold
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

        for _ in 0..depth {
            let mut next_level = Vec::new();
            for (dn, value) in &frontier {
                if let Some(record) = self.get(*dn) {
                    // Get the edge fingerprint (Z axis = what this record feeds)
                    if let Ok((_x, _y, z)) = self.unpack_record(record) {
                        let edge_fp = z.to_dense();
                        let new_value = semiring.multiply(&edge_fp, value);

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
