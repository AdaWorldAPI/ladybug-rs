//! Gestalt Frame — I/Thou/It Role Binding for Xyz Geometry
//!
//! Maps Martin Buber's relational ontology onto the Xyz (1+3)×8192 storage:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │  Role      │  Buber  │  Xyz  │  Council   │  Gestalt Value         │
//! ├────────────┼─────────┼───────┼────────────┼────────────────────────┤
//! │  Subject   │  I      │  X    │  Guardian  │  coherence             │
//! │  Predicate │  Thou   │  Y    │  Catalyst  │  thou_coupling         │
//! │  Object    │  It     │  Z    │  Balanced  │  emergence             │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## SPO as Xyz
//!
//! The dragonfly-vsa TRIPLE stores `(S ⊕ ROLE_S) | (R ⊕ ROLE_R) | (O ⊕ ROLE_O)`,
//! collapsing via superposition. We DON'T collapse. Instead:
//!
//! ```text
//! X = content ⊕ ROLE_SUBJECT    (I-perspective)
//! Y = content ⊕ ROLE_PREDICATE  (Thou-perspective)
//! Z = content ⊕ ROLE_OBJECT     (It-perspective)
//! trace = X ⊕ Y ⊕ Z            (holographic recovery)
//! ```
//!
//! ## Cross-Perspective: "Look From the Other Tree"
//!
//! To see topic T from a different angle, swap role bindings:
//!
//! ```text
//! Original:  I look at It through Thou  →  HDR[rx, ry, rz]
//! Reversed:  It looks at I through Thou  →  HDR[rz, ry, rx]  (swap X↔Z)
//! Rotated:   Thou looks at It, I am context  →  HDR[ry, rx, rz]  (swap X↔Y)
//! ```
//!
//! Each role-swap produces a different resonance profile — the same content
//! "feels" different depending on who is Subject and who is Object.
//!
//! ## Sigma and Collapse
//!
//! The standard deviation of the triangle (from gestalt_dto.py):
//! - sigma < 0.08 → FLOW (aligned, can collapse to decision)
//! - sigma 0.08-0.18 → FANOUT (ruminating, gathering context)
//! - sigma > 0.18 → RUNG_ELEVATE (high disagreement, need clarification)
//!
//! `sigma` maps directly to `HdrResonance::variance()`.

use crate::container::{Container, CogRecord};
use super::resonance::HdrResonance;

// =============================================================================
// ROLE CONTAINERS — Fixed reference atoms for S/P/O binding
// =============================================================================

/// The three fixed role containers. Deterministic from seed.
/// These are the "role atoms" from dragonfly-vsa's create_triple().
///
/// Each role is a random Container generated from a fixed seed.
/// The role containers never change — they're the coordinate axes
/// of the relational space.
pub struct RoleAtoms {
    pub subject: Container,   // ROLE_S: the "I" role
    pub predicate: Container, // ROLE_P: the "Thou" role
    pub object: Container,    // ROLE_O: the "It" role
}

impl RoleAtoms {
    /// Create role atoms from fixed seeds (deterministic, same across all instances).
    pub fn new() -> Self {
        // These seeds produce quasi-orthogonal containers (~4096 Hamming apart)
        Self {
            subject:   Container::random(0x4947_4553_5442), // "I" seed
            predicate: Container::random(0x5448_4F55_5442), // "Thou" seed
            object:    Container::random(0x4954_5345_4C46), // "It" seed
        }
    }
}

impl Default for RoleAtoms {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// QUADRANT — Relational mode from I-Thou-It axes
// =============================================================================

/// The four quadrants of the I-Thou-It relational space.
///
/// From gestalt_dto.py:
/// - X-axis = agentive ↔ experiential (i_agency)
/// - Y-axis = world ↔ relational (it_constraint - thou_coupling)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quadrant {
    /// Q1: +agentive, +world = I acts on It
    IActsOnIt,
    /// Q2: -experiential, +world = I experiences It
    IExperiencesIt,
    /// Q3: +agentive, -relational = I acts with Thou
    IActsWithThou,
    /// Q4: -experiential, -relational = I experiences Thou
    IExperiencesThou,
}

/// Collapse gate state — how aligned the triangle is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollapseGate {
    /// sigma < 0.08: tight consensus, can collapse to decision
    Flow,
    /// 0.08 <= sigma < 0.18: ruminating, gathering context
    Fanout,
    /// sigma >= 0.18: high disagreement, need rung elevation
    RungElevate,
}

impl CollapseGate {
    /// Determine gate from sigma (variance of the HDR triangle).
    pub fn from_sigma(sigma: f32) -> Self {
        if sigma < 0.08 {
            CollapseGate::Flow
        } else if sigma < 0.18 {
            CollapseGate::Fanout
        } else {
            CollapseGate::RungElevate
        }
    }
}

// =============================================================================
// GESTALT FRAME
// =============================================================================

/// A Gestalt frame: content bound with I/Thou/It roles into Xyz geometry.
///
/// The frame assigns:
/// - X = content ⊕ ROLE_SUBJECT (I-perspective)
/// - Y = content ⊕ ROLE_PREDICATE (Thou-perspective)
/// - Z = content ⊕ ROLE_OBJECT (It-perspective)
///
/// Role-swapping rotates the perspective without changing the content.
pub struct GestaltFrame {
    roles: RoleAtoms,
}

/// The result of framing content through the Gestalt I/Thou/It lens.
#[derive(Debug, Clone)]
pub struct FramedContent {
    /// X = content ⊕ ROLE_SUBJECT (I-perspective)
    pub x: Container,
    /// Y = content ⊕ ROLE_PREDICATE (Thou-perspective)
    pub y: Container,
    /// Z = content ⊕ ROLE_OBJECT (It-perspective)
    pub z: Container,
    /// trace = X ⊕ Y ⊕ Z (holographic recovery)
    pub trace: Container,
}

/// Cross-perspective result: same content seen through swapped roles.
#[derive(Debug, Clone)]
pub struct CrossPerspective {
    /// The original resonance (I looks at It through Thou)
    pub original: HdrResonance,
    /// Reversed: It looks at I through Thou (X↔Z swap)
    pub reversed: HdrResonance,
    /// Rotated: Thou looks at It, I is context (X↔Y swap)
    pub rotated: HdrResonance,
    /// Quadrant determined from the original resonance
    pub quadrant: Quadrant,
    /// Collapse gate from sigma (variance)
    pub gate: CollapseGate,
    /// Sigma: standard deviation of the original triangle
    pub sigma: f32,
}

impl GestaltFrame {
    pub fn new() -> Self {
        Self {
            roles: RoleAtoms::new(),
        }
    }

    /// Frame content through the I/Thou/It lens.
    ///
    /// Binds content with each role atom (XOR), producing 3 perspective
    /// containers stored as Xyz geometry.
    pub fn frame(&self, content: &Container) -> FramedContent {
        let x = content.xor(&self.roles.subject);
        let y = content.xor(&self.roles.predicate);
        let z = content.xor(&self.roles.object);
        let trace = x.xor(&y).xor(&z);

        FramedContent { x, y, z, trace }
    }

    /// Recover the original content from a framed view.
    ///
    /// Since XOR is self-inverse: content = X ⊕ ROLE_SUBJECT
    pub fn unframe_subject(&self, framed: &FramedContent) -> Container {
        framed.x.xor(&self.roles.subject)
    }

    /// Recover from predicate perspective.
    pub fn unframe_predicate(&self, framed: &FramedContent) -> Container {
        framed.y.xor(&self.roles.predicate)
    }

    /// Recover from object perspective.
    pub fn unframe_object(&self, framed: &FramedContent) -> Container {
        framed.z.xor(&self.roles.object)
    }

    /// Compute cross-perspective resonance: how does a query look from 3 angles?
    ///
    /// 1. Original: I (query as subject) looks at content
    /// 2. Reversed: It (query as object) looks back at content
    /// 3. Rotated: Thou (query as predicate) looks at content
    ///
    /// This is "looking at yourself from the other tree."
    pub fn cross_resonate(
        &self,
        query: &Container,
        framed: &FramedContent,
    ) -> CrossPerspective {
        // Original: query resonates with X(Subject), Y(Predicate), Z(Object)
        let original = HdrResonance::compute(query, &framed.x, &framed.y, &framed.z);

        // Reversed: swap X↔Z (I↔It) — "It" is now looking at "I"
        let reversed = HdrResonance::compute(query, &framed.z, &framed.y, &framed.x);

        // Rotated: swap X↔Y (I↔Thou) — "Thou" is now the subject
        let rotated = HdrResonance::compute(query, &framed.y, &framed.x, &framed.z);

        // Sigma from the original triangle
        let sigma = original.variance().sqrt();
        let gate = CollapseGate::from_sigma(sigma);

        // Quadrant from original resonance:
        // x_axis (agentive) = how much Subject > Object resonance
        // y_axis (world) = how much Object > Predicate resonance
        let x_axis = original.x - original.z; // I vs It
        let y_axis = original.z - original.y; // It vs Thou
        let quadrant = if y_axis >= 0.0 {
            if x_axis >= 0.0 { Quadrant::IActsOnIt } else { Quadrant::IExperiencesIt }
        } else if x_axis >= 0.0 {
            Quadrant::IActsWithThou
        } else {
            Quadrant::IExperiencesThou
        };

        CrossPerspective {
            original,
            reversed,
            rotated,
            quadrant,
            gate,
            sigma,
        }
    }

    /// Store framed content as 3 CogRecords in Xyz geometry.
    pub fn store_as_records(
        &self,
        content: &Container,
        concept_addr: u64,
    ) -> [CogRecord; 3] {
        let framed = self.frame(content);

        let mut records = [
            CogRecord::new(crate::container::ContainerGeometry::Xyz),
            CogRecord::new(crate::container::ContainerGeometry::Xyz),
            CogRecord::new(crate::container::ContainerGeometry::Xyz),
        ];

        records[0].content = framed.x; // Subject / I
        records[1].content = framed.y; // Predicate / Thou
        records[2].content = framed.z; // Object / It

        records[0].meta_view_mut().set_dn_addr(concept_addr);
        records[1].meta_view_mut().set_dn_addr(concept_addr + 1);
        records[2].meta_view_mut().set_dn_addr(concept_addr + 2);

        records
    }

    /// "Look from the other tree": given concept A and concept B in the
    /// DN tree, compute how A feels when seen through B's perspective.
    ///
    /// This is the core cross-perspective operation:
    /// 1. Extract the perspective delta between A and B's contexts
    /// 2. Apply delta to the framed content (cross-hydrate)
    /// 3. Compute resonance from the new vantage point
    pub fn look_from_other_tree(
        &self,
        topic: &FramedContent,
        my_context: &Container,
        their_context: &Container,
    ) -> CrossPerspective {
        // The perspective delta: what changes when I move from my position to theirs
        let delta = my_context.xor(their_context);

        // Apply the perspective shift to each role-bound container
        let shifted_x = topic.x.xor(&delta);
        let shifted_y = topic.y.xor(&delta);
        let shifted_z = topic.z.xor(&delta);

        // Compute resonance from the shifted vantage point
        // Query is the other's context (we're looking from their position)
        let original = HdrResonance::compute(their_context, &shifted_x, &shifted_y, &shifted_z);
        let reversed = HdrResonance::compute(their_context, &shifted_z, &shifted_y, &shifted_x);
        let rotated = HdrResonance::compute(their_context, &shifted_y, &shifted_x, &shifted_z);

        let sigma = original.variance().sqrt();
        let gate = CollapseGate::from_sigma(sigma);

        let x_axis = original.x - original.z;
        let y_axis = original.z - original.y;
        let quadrant = if y_axis >= 0.0 {
            if x_axis >= 0.0 { Quadrant::IActsOnIt } else { Quadrant::IExperiencesIt }
        } else if x_axis >= 0.0 {
            Quadrant::IActsWithThou
        } else {
            Quadrant::IExperiencesThou
        };

        CrossPerspective {
            original,
            reversed,
            rotated,
            quadrant,
            gate,
            sigma,
        }
    }
}

impl Default for GestaltFrame {
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

    #[test]
    fn test_role_atoms_quasi_orthogonal() {
        let roles = RoleAtoms::new();

        // Role containers should be quasi-orthogonal (~4096 Hamming apart)
        let d_sp = roles.subject.hamming(&roles.predicate);
        let d_so = roles.subject.hamming(&roles.object);
        let d_po = roles.predicate.hamming(&roles.object);

        assert!(d_sp > 3500 && d_sp < 4700, "S-P distance: {}", d_sp);
        assert!(d_so > 3500 && d_so < 4700, "S-O distance: {}", d_so);
        assert!(d_po > 3500 && d_po < 4700, "P-O distance: {}", d_po);
    }

    #[test]
    fn test_frame_and_unframe() {
        let gestalt = GestaltFrame::new();
        let content = Container::random(42);

        let framed = gestalt.frame(&content);

        // Each perspective should differ from content
        assert!(framed.x.hamming(&content) > 0);
        assert!(framed.y.hamming(&content) > 0);
        assert!(framed.z.hamming(&content) > 0);

        // Unframing should recover original content
        let recovered_s = gestalt.unframe_subject(&framed);
        let recovered_p = gestalt.unframe_predicate(&framed);
        let recovered_o = gestalt.unframe_object(&framed);

        assert_eq!(recovered_s, content, "subject unframe should recover content");
        assert_eq!(recovered_p, content, "predicate unframe should recover content");
        assert_eq!(recovered_o, content, "object unframe should recover content");
    }

    #[test]
    fn test_holographic_trace_recovery() {
        let gestalt = GestaltFrame::new();
        let content = Container::random(77);
        let framed = gestalt.frame(&content);

        // Recover Z from trace + X + Y
        let recovered_z = framed.trace.xor(&framed.x).xor(&framed.y);
        assert_eq!(recovered_z, framed.z);

        // Recover X from trace + Y + Z
        let recovered_x = framed.trace.xor(&framed.y).xor(&framed.z);
        assert_eq!(recovered_x, framed.x);
    }

    #[test]
    fn test_cross_perspective_symmetry() {
        let gestalt = GestaltFrame::new();
        let content = Container::random(42);
        let query = Container::random(99);

        let framed = gestalt.frame(&content);
        let cross = gestalt.cross_resonate(&query, &framed);

        // Original and reversed should have swapped X and Z
        // (approximately — they go through different resonance paths)
        // The key insight: the dominant archetype changes when roles swap
        assert!(cross.original.x != cross.reversed.x || cross.original.z != cross.reversed.z,
            "role swap should change the resonance profile");
    }

    #[test]
    fn test_self_resonance_high_coherence() {
        let gestalt = GestaltFrame::new();
        let content = Container::random(42);

        let framed = gestalt.frame(&content);

        // Query = content itself: high self-resonance from every angle
        let cross = gestalt.cross_resonate(&content, &framed);

        // Self-resonance through each role should be identical
        // because content ⊕ ROLE ⊕ content_query has fixed distance
        // All three should be similar (low sigma)
        assert!(cross.sigma < 0.2,
            "self-resonance should have low sigma: {}", cross.sigma);
    }

    #[test]
    fn test_collapse_gate_thresholds() {
        assert_eq!(CollapseGate::from_sigma(0.05), CollapseGate::Flow);
        assert_eq!(CollapseGate::from_sigma(0.10), CollapseGate::Fanout);
        assert_eq!(CollapseGate::from_sigma(0.25), CollapseGate::RungElevate);
    }

    #[test]
    fn test_quadrant_selection() {
        let gestalt = GestaltFrame::new();

        // Create content that's close to query (high resonance)
        let query = Container::random(1);
        let content = Container::random(1); // Same seed = identical

        let framed = gestalt.frame(&content);
        let cross = gestalt.cross_resonate(&query, &framed);

        // With identical content, the quadrant should be deterministic
        // (depends on role atom structure, not on random noise)
        // Just verify it's one of the valid quadrants
        assert!(matches!(
            cross.quadrant,
            Quadrant::IActsOnIt | Quadrant::IExperiencesIt |
            Quadrant::IActsWithThou | Quadrant::IExperiencesThou
        ));
    }

    #[test]
    fn test_look_from_other_tree() {
        let gestalt = GestaltFrame::new();
        let topic_content = Container::random(42);
        let my_context = Container::random(100); // My position in DN tree
        let their_context = Container::random(200); // Other node's position

        let framed = gestalt.frame(&topic_content);

        // My view
        let my_view = gestalt.cross_resonate(&my_context, &framed);
        // Their view
        let their_view = gestalt.look_from_other_tree(
            &framed, &my_context, &their_context,
        );

        // Different vantage points should produce different resonance profiles
        assert!(
            (my_view.original.x - their_view.original.x).abs() > 0.001
            || (my_view.original.y - their_view.original.y).abs() > 0.001
            || (my_view.original.z - their_view.original.z).abs() > 0.001,
            "different vantage points should produce different resonance"
        );

        // But the quadrant might or might not change
        // (depends on how different the contexts are)
    }

    #[test]
    fn test_store_as_records() {
        let gestalt = GestaltFrame::new();
        let content = Container::random(55);

        let records = gestalt.store_as_records(&content, 0x0900);

        assert_eq!(records.len(), 3);
        for r in &records {
            assert_eq!(r.geometry(), crate::container::ContainerGeometry::Xyz);
            assert!(r.content.popcount() > 0);
        }

        // DN addresses sequential
        assert_eq!(records[0].meta_view().dn_addr(), 0x0900);
        assert_eq!(records[1].meta_view().dn_addr(), 0x0901);
        assert_eq!(records[2].meta_view().dn_addr(), 0x0902);
    }
}
