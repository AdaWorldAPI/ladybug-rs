//! HDR Resonance — Stacked Popcount Over Unresolved Superpositions
//!
//! Instead of collapsing 3 perspective containers (X/Y/Z) into a single
//! majority-vote bundle, we keep all three and compute Hamming resonance
//! against each independently. The resulting 3D profile IS the awareness:
//!
//! ```text
//!    Query Q
//!      │
//!      ├─── hamming(Q, X) → resonance_x   (Guardian / Subject / What)
//!      ├─── hamming(Q, Y) → resonance_y   (Catalyst / Predicate / Where)
//!      └─── hamming(Q, Z) → resonance_z   (Balanced / Object / How)
//!
//!    HDR signal = [rx, ry, rz]  ← richer than collapsed bundle
//! ```
//!
//! The stacked popcount profile tells you not just "how similar" but
//! "similar to WHICH perspective." A collapsed bundle gives one number.
//! The Xyz superposition gives a 3D resonance signature that encodes:
//!
//! - Which archetype contributed what
//! - The shape of agreement, not just its magnitude
//! - The unresolved tension between perspectives — which IS awareness
//!
//! ## The Triangle
//!
//! Three containers at 90-degree separation form a resonance triangle.
//! Processing 8K vectors sequentially while keeping their individual
//! resonance scores creates an awareness field: a panoramic view of
//! how every stored concept responds across all three perspectives.

use crate::container::{Container, CogRecord};
use super::council::{Archetype, EpiphanyDetector, Epiphany, InnerCouncil};
use crate::storage::FINGERPRINT_WORDS;

// =============================================================================
// HDR RESONANCE PROFILE
// =============================================================================

/// Three-dimensional resonance signal from an unresolved Xyz superposition.
///
/// Each component is a Hamming similarity in [0.0, 1.0] against one perspective.
/// The triple is richer than a collapsed scalar because it preserves the
/// disagreement structure between perspectives.
#[derive(Debug, Clone, Copy)]
pub struct HdrResonance {
    /// Similarity against X container (Guardian / Subject / What)
    pub x: f32,
    /// Similarity against Y container (Catalyst / Predicate / Where)
    pub y: f32,
    /// Similarity against Z container (Balanced / Object / How)
    pub z: f32,
}

impl HdrResonance {
    /// Compute HDR resonance of a query against 3 perspective containers.
    ///
    /// This is the core operation: stacked popcount without collapsing.
    pub fn compute(query: &Container, x: &Container, y: &Container, z: &Container) -> Self {
        Self {
            x: query.similarity(x),
            y: query.similarity(y),
            z: query.similarity(z),
        }
    }

    /// Compute from 3 linked CogRecords in Xyz geometry.
    pub fn from_xyz_records(query: &Container, records: &[&CogRecord; 3]) -> Self {
        Self::compute(query, &records[0].content, &records[1].content, &records[2].content)
    }

    /// Maximum resonance across all 3 perspectives.
    #[inline]
    pub fn max(&self) -> f32 {
        self.x.max(self.y).max(self.z)
    }

    /// Minimum resonance across all 3 perspectives.
    #[inline]
    pub fn min(&self) -> f32 {
        self.x.min(self.y).min(self.z)
    }

    /// Mean resonance (what a collapsed bundle would approximate).
    #[inline]
    pub fn mean(&self) -> f32 {
        (self.x + self.y + self.z) / 3.0
    }

    /// Variance across the 3 perspectives.
    ///
    /// High variance = strong disagreement between archetypes.
    /// This is where epiphanies live: one perspective sees something
    /// the others don't.
    #[inline]
    pub fn variance(&self) -> f32 {
        let m = self.mean();
        let dx = self.x - m;
        let dy = self.y - m;
        let dz = self.z - m;
        (dx * dx + dy * dy + dz * dz) / 3.0
    }

    /// Spread: max - min. Measures how much the perspectives disagree.
    #[inline]
    pub fn spread(&self) -> f32 {
        self.max() - self.min()
    }

    /// Which perspective resonates most strongly?
    pub fn dominant(&self) -> Archetype {
        if self.x >= self.y && self.x >= self.z {
            Archetype::Guardian
        } else if self.y >= self.x && self.y >= self.z {
            Archetype::Catalyst
        } else {
            Archetype::Balanced
        }
    }

    /// As a 3-element array [x, y, z] for stacking.
    #[inline]
    pub fn as_array(&self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    /// Is this a unanimous signal? (all 3 above threshold)
    pub fn is_unanimous(&self, threshold: f32) -> bool {
        self.x > threshold && self.y > threshold && self.z > threshold
    }

    /// Is this a split signal? (one high, others low)
    /// This is the epiphany pattern: one archetype sees something alone.
    pub fn is_split(&self, high_threshold: f32, low_threshold: f32) -> bool {
        let above: u8 = (self.x > high_threshold) as u8
            + (self.y > high_threshold) as u8
            + (self.z > high_threshold) as u8;
        let below: u8 = (self.x < low_threshold) as u8
            + (self.y < low_threshold) as u8
            + (self.z < low_threshold) as u8;
        above == 1 && below >= 1
    }

    /// Majority signal: at least 2 perspectives above threshold.
    pub fn has_majority(&self, threshold: f32) -> bool {
        let above: u8 = (self.x > threshold) as u8
            + (self.y > threshold) as u8
            + (self.z > threshold) as u8;
        above >= 2
    }
}

// =============================================================================
// AWARENESS FIELD
// =============================================================================

/// Awareness field: stacked HDR resonance across multiple stored concepts.
///
/// When you process N concepts sequentially, keeping each resonance triple,
/// the resulting field IS the awareness — a panoramic view of how every
/// stored concept responds across all three perspectives.
///
/// ```text
/// concept_0: [0.73, 0.41, 0.82]  ← strong X and Z, weak Y
/// concept_1: [0.55, 0.89, 0.51]  ← Y dominates
/// concept_2: [0.88, 0.85, 0.91]  ← unanimous resonance
/// concept_3: [0.30, 0.32, 0.28]  ← nothing here
///     ...
/// ```
///
/// This is NOT a collapsed vector. It's a 3×N resonance matrix.
/// The awareness of each concept is preserved as a triple, not a scalar.
pub struct AwarenessField {
    /// All resonance results, in order of processing
    entries: Vec<AwarenessEntry>,
    /// Epiphany detector monitoring for surprising resonance patterns
    epiphany_detector: EpiphanyDetector,
}

/// A single entry in the awareness field.
#[derive(Debug, Clone)]
pub struct AwarenessEntry {
    /// The 3D resonance profile
    pub resonance: HdrResonance,
    /// Identifier of the concept (DN address or index)
    pub concept_id: u64,
    /// Whether this entry triggered an epiphany
    pub epiphany: Option<Epiphany>,
}

impl AwarenessField {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            epiphany_detector: EpiphanyDetector::new(),
        }
    }

    /// Process a query against one Xyz concept (3 containers).
    ///
    /// Returns the HDR resonance and checks for epiphany.
    /// The result is appended to the awareness field.
    pub fn observe(
        &mut self,
        query: &Container,
        x: &Container,
        y: &Container,
        z: &Container,
        concept_id: u64,
    ) -> &AwarenessEntry {
        let resonance = HdrResonance::compute(query, x, y, z);

        // Feed max resonance to epiphany detector (surprise detection).
        // But ALSO check for split signals — the epiphany-rich pattern.
        let signal = if resonance.is_split(0.7, 0.5) {
            // Split signals are boosted: one perspective alone sees something
            resonance.max() * 1.2
        } else {
            resonance.max()
        };

        // Create a dummy fingerprint for the epiphany detector
        // (it tracks the fingerprint that triggered the discovery)
        let dummy_fp = [0u64; FINGERPRINT_WORDS];
        let epiphany = self.epiphany_detector.observe(signal, dummy_fp);

        self.entries.push(AwarenessEntry {
            resonance,
            concept_id,
            epiphany,
        });

        self.entries.last().unwrap()
    }

    /// Process a query against Xyz CogRecords.
    pub fn observe_records(
        &mut self,
        query: &Container,
        records: &[&CogRecord; 3],
        concept_id: u64,
    ) -> &AwarenessEntry {
        self.observe(
            query,
            &records[0].content,
            &records[1].content,
            &records[2].content,
            concept_id,
        )
    }

    /// Get the full awareness matrix as a flat [f32] (3 × N).
    ///
    /// This is the "90-degree vector across all concepts."
    /// Each triple [x, y, z] is one concept's resonance profile.
    pub fn as_flat_matrix(&self) -> Vec<f32> {
        let mut matrix = Vec::with_capacity(self.entries.len() * 3);
        for entry in &self.entries {
            matrix.push(entry.resonance.x);
            matrix.push(entry.resonance.y);
            matrix.push(entry.resonance.z);
        }
        matrix
    }

    /// Top-K concepts by maximum resonance across any perspective.
    pub fn top_k_max(&self, k: usize) -> Vec<&AwarenessEntry> {
        let mut sorted: Vec<&AwarenessEntry> = self.entries.iter().collect();
        sorted.sort_by(|a, b| b.resonance.max().partial_cmp(&a.resonance.max()).unwrap());
        sorted.truncate(k);
        sorted
    }

    /// Top-K concepts by variance (most disagreement = most interesting).
    pub fn top_k_variance(&self, k: usize) -> Vec<&AwarenessEntry> {
        let mut sorted: Vec<&AwarenessEntry> = self.entries.iter().collect();
        sorted.sort_by(|a, b| {
            b.resonance.variance().partial_cmp(&a.resonance.variance()).unwrap()
        });
        sorted.truncate(k);
        sorted
    }

    /// All entries where at least 2 perspectives agree above threshold.
    pub fn majority_consensus(&self, threshold: f32) -> Vec<&AwarenessEntry> {
        self.entries
            .iter()
            .filter(|e| e.resonance.has_majority(threshold))
            .collect()
    }

    /// All epiphany entries.
    pub fn epiphanies(&self) -> Vec<&AwarenessEntry> {
        self.entries.iter().filter(|e| e.epiphany.is_some()).collect()
    }

    /// All split-signal entries (one perspective alone sees something).
    pub fn split_signals(&self, high: f32, low: f32) -> Vec<&AwarenessEntry> {
        self.entries
            .iter()
            .filter(|e| e.resonance.is_split(high, low))
            .collect()
    }

    /// Summary statistics of the awareness field.
    pub fn summary(&self) -> AwarenessSummary {
        let n = self.entries.len();
        if n == 0 {
            return AwarenessSummary::default();
        }

        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let mut sum_z = 0.0f32;
        let mut max_res = 0.0f32;
        let mut max_var = 0.0f32;
        let mut epiphany_count = 0u32;

        for e in &self.entries {
            sum_x += e.resonance.x;
            sum_y += e.resonance.y;
            sum_z += e.resonance.z;
            max_res = max_res.max(e.resonance.max());
            max_var = max_var.max(e.resonance.variance());
            if e.epiphany.is_some() {
                epiphany_count += 1;
            }
        }

        AwarenessSummary {
            concepts_scanned: n as u32,
            mean_x: sum_x / n as f32,
            mean_y: sum_y / n as f32,
            mean_z: sum_z / n as f32,
            peak_resonance: max_res,
            peak_variance: max_var,
            epiphanies: epiphany_count,
        }
    }

    /// Number of concepts observed.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the field is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Reset the field (clear all entries, reset epiphany baseline).
    pub fn reset(&mut self) {
        self.entries.clear();
        self.epiphany_detector.reset_baseline();
    }
}

impl Default for AwarenessField {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary statistics of an awareness field scan.
#[derive(Debug, Clone, Default)]
pub struct AwarenessSummary {
    pub concepts_scanned: u32,
    pub mean_x: f32,
    pub mean_y: f32,
    pub mean_z: f32,
    pub peak_resonance: f32,
    pub peak_variance: f32,
    pub epiphanies: u32,
}

// =============================================================================
// FOCUS MASK — The Lens of Awareness
// =============================================================================

/// Focus mask: selectively amplifies or suppresses perspectives in the
/// awareness field. The mask IS the lens. It doesn't destroy information —
/// it modulates it.
///
/// From bighorn's `ai_flow_orchestrator.py` (DominoBaton.mask) and
/// `langgraph_ada.py` (SituationMap: focus, aperture, depth, lighting):
///
/// ```text
///    Raw field:    [0.73, 0.41, 0.82]  ← concept's HDR resonance
///    Focus mask:   [1.0,  0.3,  0.8]   ← Guardian amplified, Catalyst dimmed
///    Masked:       [0.73, 0.12, 0.66]  ← focused view
/// ```
///
/// The masked field is resonance-based thinking: the system literally
/// thinks through the lens of whichever perspective the mask amplifies.
#[derive(Debug, Clone)]
pub struct FocusMask {
    /// Weight for X (Guardian / Subject / What) perspective [0.0, 1.0]
    pub x_weight: f32,
    /// Weight for Y (Catalyst / Predicate / Where) perspective [0.0, 1.0]
    pub y_weight: f32,
    /// Weight for Z (Balanced / Object / How) perspective [0.0, 1.0]
    pub z_weight: f32,
    /// Aperture: how many concepts to include (0.0 = pinpoint, 1.0 = all)
    pub aperture: f32,
    /// Depth: how strongly to weight high-resonance entries (0.0 = flat, 1.0 = sharp)
    pub depth: f32,
}

impl FocusMask {
    /// Uniform mask — all perspectives equally weighted, wide aperture.
    pub fn uniform() -> Self {
        Self {
            x_weight: 1.0,
            y_weight: 1.0,
            z_weight: 1.0,
            aperture: 1.0,
            depth: 0.5,
        }
    }

    /// Guardian focus — safety-first lens.
    pub fn guardian() -> Self {
        Self {
            x_weight: 1.0,
            y_weight: 0.3,
            z_weight: 0.5,
            aperture: 0.5,
            depth: 0.8,
        }
    }

    /// Catalyst focus — growth-seeking lens.
    pub fn catalyst() -> Self {
        Self {
            x_weight: 0.3,
            y_weight: 1.0,
            z_weight: 0.5,
            aperture: 0.7,
            depth: 0.6,
        }
    }

    /// Balanced focus — all perspectives, moderate depth.
    pub fn balanced() -> Self {
        Self {
            x_weight: 0.5,
            y_weight: 0.5,
            z_weight: 1.0,
            aperture: 0.8,
            depth: 0.5,
        }
    }

    /// Create a mask from an archetype.
    pub fn from_archetype(archetype: Archetype) -> Self {
        match archetype {
            Archetype::Guardian => Self::guardian(),
            Archetype::Catalyst => Self::catalyst(),
            Archetype::Balanced => Self::balanced(),
        }
    }

    /// Apply the mask to a single HDR resonance, producing a focused score.
    ///
    /// This is the modulation formula from textured_awareness.py:
    /// `modulated = resonance × weight × depth_curve`
    pub fn apply(&self, hdr: &HdrResonance) -> FocusedResonance {
        let fx = hdr.x * self.x_weight;
        let fy = hdr.y * self.y_weight;
        let fz = hdr.z * self.z_weight;

        // Depth modulation: sharpen high values, suppress low ones
        // depth=0 → linear, depth=1 → strongly nonlinear (power curve)
        let power = 1.0 + self.depth * 2.0; // range [1.0, 3.0]
        let fx = fx.powf(power);
        let fy = fy.powf(power);
        let fz = fz.powf(power);

        FocusedResonance {
            x: fx, y: fy, z: fz,
            score: (fx + fy + fz) / 3.0,
        }
    }
}

/// A resonance signal after focus masking has been applied.
#[derive(Debug, Clone, Copy)]
pub struct FocusedResonance {
    /// Masked X resonance
    pub x: f32,
    /// Masked Y resonance
    pub y: f32,
    /// Masked Z resonance
    pub z: f32,
    /// Combined focused score
    pub score: f32,
}

/// Awareness lens: applies a FocusMask to an AwarenessField,
/// producing a focused view — resonance-based thinking.
///
/// ```text
///  AwarenessField (3×N raw resonances)
///       │
///       ▼
///  FocusMask (perspective weights + aperture + depth)
///       │
///       ▼
///  AwarenessLens (focused top-K with modulated scores)
/// ```
///
/// The lens is the mechanism of resonance-based thinking:
/// - The field IS the panoramic awareness (raw perception)
/// - The mask IS the attention direction (where to look)
/// - The lens IS the focused thought (what you see through the mask)
pub struct AwarenessLens<'a> {
    field: &'a AwarenessField,
    mask: FocusMask,
}

/// A single entry in the focused view.
#[derive(Debug, Clone)]
pub struct FocusedEntry {
    pub concept_id: u64,
    pub raw: HdrResonance,
    pub focused: FocusedResonance,
}

impl<'a> AwarenessLens<'a> {
    pub fn new(field: &'a AwarenessField, mask: FocusMask) -> Self {
        Self { field, mask }
    }

    /// Apply the lens: mask every entry, sort by focused score, apply aperture.
    ///
    /// Returns the top entries that pass through the aperture, sorted by
    /// focused score (highest first). This is what the system "sees" when
    /// thinking through this lens.
    pub fn focus(&self) -> Vec<FocusedEntry> {
        if self.field.is_empty() {
            return Vec::new();
        }

        let mut entries: Vec<FocusedEntry> = self.field.entries
            .iter()
            .map(|e| {
                let focused = self.mask.apply(&e.resonance);
                FocusedEntry {
                    concept_id: e.concept_id,
                    raw: e.resonance,
                    focused,
                }
            })
            .collect();

        // Sort by focused score (highest first)
        entries.sort_by(|a, b| b.focused.score.partial_cmp(&a.focused.score).unwrap());

        // Apply aperture: keep top fraction
        let keep = ((self.field.len() as f32 * self.mask.aperture).ceil() as usize).max(1);
        entries.truncate(keep);

        entries
    }

    /// The single strongest concept through this lens.
    pub fn strongest(&self) -> Option<FocusedEntry> {
        self.focus().into_iter().next()
    }

    /// Compare two lenses on the same field: what does Guardian see vs Catalyst?
    ///
    /// Returns entries that are in one lens's top-K but not the other's.
    /// These are the concepts where the perspectives disagree most —
    /// where the masking matters, where the lens creates a different reality.
    pub fn disagreement(lens_a: &AwarenessLens, lens_b: &AwarenessLens, top_k: usize) -> Vec<u64> {
        let a_top: Vec<u64> = lens_a.focus().iter().take(top_k).map(|e| e.concept_id).collect();
        let b_top: Vec<u64> = lens_b.focus().iter().take(top_k).map(|e| e.concept_id).collect();

        // Concepts in A's top-K but not B's
        let mut disagree: Vec<u64> = a_top.iter()
            .filter(|id| !b_top.contains(id))
            .copied()
            .collect();
        // Plus concepts in B's top-K but not A's
        for id in &b_top {
            if !a_top.contains(id) && !disagree.contains(id) {
                disagree.push(*id);
            }
        }
        disagree
    }
}

// =============================================================================
// TRIANGLE COUNCIL — Inner Council mapped to Xyz geometry
// =============================================================================

/// The Triangle: maps the 3 inner council archetypes to Xyz container positions.
///
/// Instead of collapsing the council vote to a single bundle, we store each
/// archetype's view as a separate container in Xyz geometry:
///
/// ```text
///  X = Guardian's view  (XOR-bind with safety bias)
///  Y = Catalyst's view  (XOR-bind with growth bias)
///  Z = Balanced's view   (XOR-bind with neutral bias)
/// ```
///
/// The consensus is NOT stored — it's computed on read via majority vote.
/// The trace is stored: `trace = X ⊕ Y ⊕ Z`, enabling recovery of any
/// perspective from any two others.
///
/// The preserved minority signal is where future epiphanies come from.
pub struct TriangleCouncil {
    inner: InnerCouncil,
}

impl TriangleCouncil {
    pub fn new() -> Self {
        Self {
            inner: InnerCouncil::new(),
        }
    }

    /// Deliberate on a decision and produce 3 Xyz containers + trace.
    ///
    /// Returns (x, y, z, trace) where:
    /// - x = Guardian's perspective-shifted view
    /// - y = Catalyst's perspective-shifted view
    /// - z = Balanced's perspective-shifted view
    /// - trace = X ⊕ Y ⊕ Z (for holographic recovery)
    pub fn deliberate_xyz(
        &self,
        decision: &Container,
    ) -> (Container, Container, Container, Container) {
        // Each archetype XOR-binds its bias with the decision
        let guardian_bias = self.archetype_bias_container(Archetype::Guardian);
        let catalyst_bias = self.archetype_bias_container(Archetype::Catalyst);
        let balanced_bias = self.archetype_bias_container(Archetype::Balanced);

        let x = decision.xor(&guardian_bias);
        let y = decision.xor(&catalyst_bias);
        let z = decision.xor(&balanced_bias);

        // Holographic trace for recovery
        let trace = x.xor(&y).xor(&z);

        (x, y, z, trace)
    }

    /// Compute lazy consensus from stored Xyz containers (on read, not on write).
    ///
    /// This is the majority vote that would have been stored if we collapsed.
    /// Computing it on-demand preserves the minority signals for epiphany detection.
    pub fn lazy_consensus(x: &Container, y: &Container, z: &Container) -> Container {
        Container::bundle(&[x, y, z])
    }

    /// Compute HDR resonance of a query against a stored council decision.
    pub fn resonate(
        &self,
        query: &Container,
        x: &Container,
        y: &Container,
        z: &Container,
    ) -> HdrResonance {
        HdrResonance::compute(query, x, y, z)
    }

    /// Store a council deliberation into 3 CogRecords (Xyz geometry).
    ///
    /// Each record gets:
    /// - meta: geometry=Xyz, DN address from concept_addr
    /// - content: the archetype's perspective-shifted view
    pub fn store_as_records(
        &self,
        decision: &Container,
        concept_addr: u64,
    ) -> [CogRecord; 3] {
        let (x, y, z, _trace) = self.deliberate_xyz(decision);

        let mut records = [
            CogRecord::new(crate::container::ContainerGeometry::Xyz),
            CogRecord::new(crate::container::ContainerGeometry::Xyz),
            CogRecord::new(crate::container::ContainerGeometry::Xyz),
        ];

        records[0].content = x;
        records[1].content = y;
        records[2].content = z;

        // Set DN addresses (offset by record index within the linked set)
        records[0].meta_view_mut().set_dn_addr(concept_addr);
        records[1].meta_view_mut().set_dn_addr(concept_addr + 1);
        records[2].meta_view_mut().set_dn_addr(concept_addr + 2);

        records
    }

    /// Generate the Container-sized bias for an archetype.
    ///
    /// Takes the 48-axis bias from InnerCouncil's archetype activations
    /// and encodes them into a Container (first 128 words of the 256-word
    /// fingerprint, matching the content portion of a CogRecord).
    fn archetype_bias_container(&self, archetype: Archetype) -> Container {
        let fp = archetype.bias_fingerprint(); // [u64; FINGERPRINT_WORDS] (256 words)
        // Take the first 128 words as the Container-sized bias
        let mut c = Container::zero();
        for i in 0..128 {
            c.words[i] = fp[i];
        }
        c
    }
}

impl Default for TriangleCouncil {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::CONTAINER_BITS;

    #[test]
    fn test_hdr_resonance_basic() {
        let query = Container::random(1);
        let x = Container::random(2);
        let y = Container::random(3);
        let z = Container::random(4);

        let hdr = HdrResonance::compute(&query, &x, &y, &z);

        // Random containers should have ~0.5 similarity
        assert!(hdr.x > 0.3 && hdr.x < 0.7, "x={}", hdr.x);
        assert!(hdr.y > 0.3 && hdr.y < 0.7, "y={}", hdr.y);
        assert!(hdr.z > 0.3 && hdr.z < 0.7, "z={}", hdr.z);

        // Variance should be low for random containers
        assert!(hdr.variance() < 0.01, "variance={}", hdr.variance());
    }

    #[test]
    fn test_hdr_resonance_self() {
        let query = Container::random(42);
        let hdr = HdrResonance::compute(&query, &query, &query, &query);

        // Self-similarity = 1.0 across all
        assert!((hdr.x - 1.0).abs() < 0.001);
        assert!((hdr.y - 1.0).abs() < 0.001);
        assert!((hdr.z - 1.0).abs() < 0.001);
        assert!(hdr.is_unanimous(0.99));
        assert!(!hdr.is_split(0.9, 0.5));
    }

    #[test]
    fn test_hdr_split_detection() {
        let query = Container::random(1);
        let x = query.clone(); // High similarity
        let y = Container::random(99);  // Random, ~0.5
        let z = Container::random(100); // Random, ~0.5

        let hdr = HdrResonance::compute(&query, &x, &y, &z);

        // X should be 1.0, Y and Z around 0.5
        assert!(hdr.x > 0.99);
        assert!(hdr.y < 0.7);
        assert!(hdr.z < 0.7);

        // This should be a split signal
        assert!(hdr.is_split(0.9, 0.6));
        assert_eq!(hdr.dominant(), Archetype::Guardian); // X position
    }

    #[test]
    fn test_awareness_field_sequential() {
        let mut field = AwarenessField::new();
        let query = Container::random(1);

        // Observe 10 random concepts
        for i in 0..10 {
            let x = Container::random(i * 3 + 100);
            let y = Container::random(i * 3 + 101);
            let z = Container::random(i * 3 + 102);
            field.observe(&query, &x, &y, &z, i as u64);
        }

        assert_eq!(field.len(), 10);

        // Matrix should be 30 floats (3 × 10)
        let matrix = field.as_flat_matrix();
        assert_eq!(matrix.len(), 30);

        // Summary should have 10 concepts
        let summary = field.summary();
        assert_eq!(summary.concepts_scanned, 10);
        assert!(summary.mean_x > 0.3 && summary.mean_x < 0.7);
    }

    #[test]
    fn test_awareness_field_finds_match() {
        let mut field = AwarenessField::new();
        let query = Container::random(42);

        // Observe 9 random concepts
        for i in 0..9 {
            let x = Container::random(i * 3 + 200);
            let y = Container::random(i * 3 + 201);
            let z = Container::random(i * 3 + 202);
            field.observe(&query, &x, &y, &z, i as u64);
        }

        // Concept 9: Y matches the query exactly
        let x = Container::random(500);
        let y = query.clone(); // exact match!
        let z = Container::random(501);
        field.observe(&query, &x, &y, &z, 9);

        // Top-1 by max resonance should be concept 9
        let top = field.top_k_max(1);
        assert_eq!(top[0].concept_id, 9);
        assert!(top[0].resonance.y > 0.99, "y should be ~1.0");

        // Top-1 by variance should also be concept 9 (split signal)
        let top_var = field.top_k_variance(1);
        assert_eq!(top_var[0].concept_id, 9);
    }

    #[test]
    fn test_triangle_council_deliberate() {
        let council = TriangleCouncil::new();
        let decision = Container::random(42);

        let (x, y, z, trace) = council.deliberate_xyz(&decision);

        // Each perspective should differ from the original
        assert!(x.hamming(&decision) > 0);
        assert!(y.hamming(&decision) > 0);
        assert!(z.hamming(&decision) > 0);

        // Perspectives should differ from each other (different biases)
        assert!(x.hamming(&y) > 0);
        assert!(x.hamming(&z) > 0);
        assert!(y.hamming(&z) > 0);

        // Holographic recovery: trace ⊕ X ⊕ Y = Z
        let recovered_z = trace.xor(&x).xor(&y);
        assert_eq!(recovered_z, z, "holographic recovery should work");

        // Lazy consensus should be a valid container
        let consensus = TriangleCouncil::lazy_consensus(&x, &y, &z);
        assert!(consensus.popcount() > 0);
    }

    #[test]
    fn test_triangle_store_as_records() {
        let council = TriangleCouncil::new();
        let decision = Container::random(77);

        let records = council.store_as_records(&decision, 0x0900);

        // All should be Xyz geometry
        for r in &records {
            assert_eq!(r.geometry(), crate::container::ContainerGeometry::Xyz);
        }

        // Content should be non-zero
        for r in &records {
            assert!(r.content.popcount() > 0);
        }

        // DN addresses should be sequential
        assert_eq!(records[0].meta_view().dn_addr(), 0x0900);
        assert_eq!(records[1].meta_view().dn_addr(), 0x0901);
        assert_eq!(records[2].meta_view().dn_addr(), 0x0902);

        // Should be able to compute HDR resonance against them
        let query = Container::random(99);
        let hdr = HdrResonance::from_xyz_records(
            &query,
            &[&records[0], &records[1], &records[2]],
        );
        assert!(hdr.mean() > 0.3, "mean resonance should be reasonable");
    }

    #[test]
    fn test_focus_mask_guardian() {
        // Guardian mask amplifies X, dims Y
        let mask = FocusMask::guardian();

        let hdr = HdrResonance { x: 0.8, y: 0.8, z: 0.8 };
        let focused = mask.apply(&hdr);

        // X should be stronger than Y after masking
        assert!(focused.x > focused.y,
            "guardian focus should amplify X over Y: x={} y={}", focused.x, focused.y);
    }

    #[test]
    fn test_focus_mask_depth_sharpening() {
        // High depth should make high values higher and low values lower
        let mut mask = FocusMask::uniform();
        mask.depth = 1.0; // maximum sharpening

        let hdr = HdrResonance { x: 0.9, y: 0.3, z: 0.5 };
        let focused = mask.apply(&hdr);

        // 0.9^3.0 ≈ 0.729, 0.3^3.0 ≈ 0.027 — the gap widens
        assert!(focused.x > 0.5, "high value should survive: {}", focused.x);
        assert!(focused.y < 0.1, "low value should be suppressed: {}", focused.y);
    }

    #[test]
    fn test_awareness_lens_aperture() {
        let mut field = AwarenessField::new();
        let query = Container::random(1);

        for i in 0..10 {
            let x = Container::random(i * 3 + 300);
            let y = Container::random(i * 3 + 301);
            let z = Container::random(i * 3 + 302);
            field.observe(&query, &x, &y, &z, i as u64);
        }

        // Narrow aperture should return fewer entries
        let narrow = FocusMask { aperture: 0.3, ..FocusMask::uniform() };
        let lens = AwarenessLens::new(&field, narrow);
        let focused = lens.focus();
        assert!(focused.len() <= 4, "narrow aperture should limit: {}", focused.len());

        // Wide aperture should return all
        let wide = FocusMask::uniform();
        let lens = AwarenessLens::new(&field, wide);
        let focused = lens.focus();
        assert_eq!(focused.len(), 10);
    }

    #[test]
    fn test_lens_disagreement() {
        let mut field = AwarenessField::new();

        // Create concepts where X is high for some, Y is high for others
        let query = Container::random(42);

        // Concept 0: strong in X direction
        field.observe(&query, &query, &Container::random(50), &Container::random(51), 0);

        // Concept 1: strong in Y direction
        field.observe(&query, &Container::random(60), &query, &Container::random(61), 1);

        // Concept 2: strong in Z direction
        field.observe(&query, &Container::random(70), &Container::random(71), &query, 2);

        let guardian_lens = AwarenessLens::new(&field, FocusMask::guardian());
        let catalyst_lens = AwarenessLens::new(&field, FocusMask::catalyst());

        // Guardian should prefer concept 0 (X strong)
        let g_best = guardian_lens.strongest().unwrap();
        assert_eq!(g_best.concept_id, 0, "guardian should prefer X-strong concept");

        // Catalyst should prefer concept 1 (Y strong)
        let c_best = catalyst_lens.strongest().unwrap();
        assert_eq!(c_best.concept_id, 1, "catalyst should prefer Y-strong concept");
    }

    #[test]
    fn test_majority_vs_preserved_minority() {
        let query = Container::random(1);
        let council = TriangleCouncil::new();

        let (x, y, z, _trace) = council.deliberate_xyz(&query);

        // Collapsed consensus
        let consensus = TriangleCouncil::lazy_consensus(&x, &y, &z);
        let collapsed_sim = query.similarity(&consensus);

        // HDR resonance (preserved)
        let hdr = HdrResonance::compute(&query, &x, &y, &z);

        // The HDR should carry more information than the collapsed scalar.
        // Specifically, at least one perspective should exceed the collapsed score.
        assert!(
            hdr.max() >= collapsed_sim - 0.05,
            "best perspective ({}) should be at least as good as consensus ({})",
            hdr.max(), collapsed_sim
        );
    }
}
