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

use super::scent::NibbleScent;
use super::sparse::{unpack_axes, AxisDescriptors, SparseContainer, SpoError};

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
// NARS TRUTH GATING
// ============================================================================

/// Gate that filters SPO hits by NARS truth value thresholds.
///
/// Records whose frequency < min_frequency OR confidence < min_confidence
/// are skipped before the expensive Hamming distance computation.
#[derive(Clone, Debug)]
pub struct TruthGate {
    pub min_frequency: f32,
    pub min_confidence: f32,
}

impl TruthGate {
    /// A gate that passes everything (no filtering).
    pub fn open() -> Self {
        Self { min_frequency: 0.0, min_confidence: 0.0 }
    }

    /// Check if a truth value passes this gate.
    #[inline]
    pub fn passes(&self, tv: &TruthValue) -> bool {
        tv.frequency >= self.min_frequency && tv.confidence >= self.min_confidence
    }
}

impl Default for TruthGate {
    fn default() -> Self { Self::open() }
}

// ============================================================================
// SPO HIT (enriched query result)
// ============================================================================

/// Query hit enriched with NARS truth value.
#[derive(Clone, Debug)]
pub struct SpoHit {
    pub dn: u64,
    pub distance: u32,
    pub axis: QueryAxis,
    pub truth: TruthValue,
}

// ============================================================================
// SPO SEMIRING (chain combination)
// ============================================================================

/// Semiring for combining SPO hits across chain hops.
///
/// - frequency:  product (AND semantics)
/// - confidence: product × link coherence (deduction with attenuation)
pub struct SpoSemiring;

impl SpoSemiring {
    /// Combine two truth values with a link coherence factor.
    ///
    /// f_out = f1 × f2
    /// c_out = c1 × c2 × coherence
    #[inline]
    pub fn combine(a: &TruthValue, b: &TruthValue, coherence: f32) -> TruthValue {
        TruthValue::new(
            (a.frequency * b.frequency).clamp(0.0, 1.0),
            (a.confidence * b.confidence * coherence).clamp(0.0, 1.0),
        )
    }

    /// Fold a chain of (TruthValue, coherence) pairs into a single truth value.
    pub fn fold_chain(steps: &[(TruthValue, f32)]) -> TruthValue {
        if steps.is_empty() { return TruthValue::unknown(); }
        let mut f = 1.0f32;
        let mut c = 1.0f32;
        for (tv, coh) in steps {
            f *= tv.frequency;
            c *= tv.confidence * coh;
        }
        TruthValue::new(f.clamp(0.0, 1.0), c.clamp(0.0, 1.0))
    }
}

// ============================================================================
// SPO STORE
// ============================================================================

/// Three-axis content-addressable graph store.
///
/// POC uses BTreeMap. Production replaces with LanceDB columnar store.
pub struct SpoStore {
    records: BTreeMap<u64, CogRecord>,
}

impl SpoStore {
    pub fn new() -> Self {
        Self {
            records: BTreeMap::new(),
        }
    }

    pub fn insert(&mut self, record: CogRecord) -> Result<(), SpoError> {
        let dn = record.meta.words[0]; // W0 = DN address
        if self.records.contains_key(&dn) {
            return Err(SpoError::DuplicateDn { dn });
        }
        self.records.insert(dn, record);
        Ok(())
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
    // SCENT-PRUNED PROJECTIONS (Module 4)
    // ========================================================================

    /// S × P → O: Forward projection with scent pre-filter and NARS gating.
    ///
    /// Accepts sparse containers directly — no `from_dense()` on query side.
    /// 1. Gate on NARS truth value (skip below threshold).
    /// 2. Pre-filter via per-axis scent distance (skip incompatible profiles).
    /// 3. Full Hamming only on candidates passing both filters.
    pub fn sxp2o(
        &self,
        s: &SparseContainer,
        p: &SparseContainer,
        radius: u32,
        gate: &TruthGate,
    ) -> Vec<SpoHit> {
        let q_scent = NibbleScent::from_axes(s, p, &SparseContainer::zero());
        let mut hits = Vec::new();

        for (&dn, record) in &self.records {
            // Gate 1: NARS truth threshold
            let truth = self.read_nars(record);
            if !gate.passes(&truth) { continue; }

            // Gate 2: Scent pre-filter (per-axis L1 as cheap lower bound)
            let rec_scent = self.read_scent(record);
            let (sx, sy, _) = rec_scent.axis_distances(&q_scent);
            if sx.saturating_add(sy) > radius { continue; }

            // Gate 3: Full Hamming on X + Y axes
            if let Ok((x, y, _z)) = self.unpack_record(record) {
                let dx = SparseContainer::hamming_sparse(s, &x);
                let dy = SparseContainer::hamming_sparse(p, &y);
                let combined = dx.saturating_add(dy) / 2;
                if combined <= radius {
                    hits.push(SpoHit { dn, distance: combined, axis: QueryAxis::XY, truth });
                }
            }
        }

        hits.sort_by_key(|h| h.distance);
        hits
    }

    /// S × O → P: Relation projection — "How are S and O related?"
    ///
    /// Scans X + Z axes, returns predicate matches.
    pub fn sxo2p(
        &self,
        s: &SparseContainer,
        o: &SparseContainer,
        radius: u32,
        gate: &TruthGate,
    ) -> Vec<SpoHit> {
        let q_scent = NibbleScent::from_axes(s, &SparseContainer::zero(), o);
        let mut hits = Vec::new();

        for (&dn, record) in &self.records {
            let truth = self.read_nars(record);
            if !gate.passes(&truth) { continue; }

            let rec_scent = self.read_scent(record);
            let (sx, _, sz) = rec_scent.axis_distances(&q_scent);
            if sx.saturating_add(sz) > radius { continue; }

            if let Ok((x, _y, z)) = self.unpack_record(record) {
                let dx = SparseContainer::hamming_sparse(s, &x);
                let dz = SparseContainer::hamming_sparse(o, &z);
                let combined = dx.saturating_add(dz) / 2;
                if combined <= radius {
                    hits.push(SpoHit { dn, distance: combined, axis: QueryAxis::XZ, truth });
                }
            }
        }

        hits.sort_by_key(|h| h.distance);
        hits
    }

    /// P × O → S: Reverse projection — "Who <verb>s <tgt>?"
    ///
    /// Scans Y + Z axes, returns subject matches.
    pub fn pxo2s(
        &self,
        p: &SparseContainer,
        o: &SparseContainer,
        radius: u32,
        gate: &TruthGate,
    ) -> Vec<SpoHit> {
        let q_scent = NibbleScent::from_axes(&SparseContainer::zero(), p, o);
        let mut hits = Vec::new();

        for (&dn, record) in &self.records {
            let truth = self.read_nars(record);
            if !gate.passes(&truth) { continue; }

            let rec_scent = self.read_scent(record);
            let (_, sy, sz) = rec_scent.axis_distances(&q_scent);
            if sy.saturating_add(sz) > radius { continue; }

            if let Ok((_x, y, z)) = self.unpack_record(record) {
                let dy = SparseContainer::hamming_sparse(p, &y);
                let dz = SparseContainer::hamming_sparse(o, &z);
                let combined = dy.saturating_add(dz) / 2;
                if combined <= radius {
                    hits.push(SpoHit { dn, distance: combined, axis: QueryAxis::YZ, truth });
                }
            }
        }

        hits.sort_by_key(|h| h.distance);
        hits
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
