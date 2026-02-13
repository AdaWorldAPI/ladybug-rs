//! Quad-Triangle Cognitive Model with 10K-bit VSA Corners
//!
//! Four interlocking triangles, each corner is a 10K-bit fingerprint:
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────┐
//! │                    QUAD-TRIANGLE GEOMETRY                      │
//! │                                                                │
//! │    Processing Triangle         Content Triangle                │
//! │         (10K×3)                   (10K×3)                      │
//! │                                                                │
//! │       Analytical                  Abstract                     │
//! │          ╱╲                         ╱╲                         │
//! │         ╱  ╲                       ╱  ╲                        │
//! │        ╱ A  ╲                     ╱ B  ╲                       │
//! │       ╱──────╲                   ╱──────╲                      │
//! │   Intuitive  Procedural     Concrete  Relational              │
//! │                                                                │
//! │    Gestalt Triangle        Crystallization Triangle           │
//! │         (10K×3)                   (10K×3)                      │
//! │                                                                │
//! │       Coherence                  Immutable                     │
//! │          ╱╲                         ╱╲                         │
//! │         ╱  ╲                       ╱  ╲                        │
//! │        ╱ C  ╲                     ╱ D  ╲                       │
//! │       ╱──────╲                   ╱──────╲                      │
//! │    Novelty  Resonance        Hot    Experimental              │
//! │                                                                │
//! │  Total: 4 triangles × 3 corners × 10K bits = 120K bits        │
//! │  With VSA bundle: 4 triangles × 10K bits = 40K bits (5KB)     │
//! └────────────────────────────────────────────────────────────────┘
//! ```
//!
//! Hardware acceleration:
//! - AVX-512 VPOPCNTDQ for Hamming distance (triangle similarity)
//! - SIMD bundle/bind for triangle composition
//! - Vectorized SD calculation for collapse gate

use crate::core::{Fingerprint, VsaOps};
use std::fmt;

// =============================================================================
// TRIANGLE CORNER - 10K-bit VSA Vector
// =============================================================================

/// A single triangle corner as a 10K-bit fingerprint
#[derive(Clone)]
pub struct TriangleCorner {
    /// 10K-bit fingerprint for this corner
    pub fingerprint: Fingerprint,

    /// Activation level [0.0, 1.0] (for compatibility)
    pub activation: f32,

    /// Label for display
    pub label: &'static str,
}

impl TriangleCorner {
    /// Create corner from label (deterministic fingerprint)
    pub fn from_label(label: &'static str) -> Self {
        Self {
            fingerprint: Fingerprint::from_content(label),
            activation: 0.5,
            label,
        }
    }

    /// Create corner with specific activation
    pub fn with_activation(label: &'static str, activation: f32) -> Self {
        Self {
            fingerprint: Fingerprint::from_content(label),
            activation: activation.clamp(0.0, 1.0),
            label,
        }
    }

    /// Resonance with another corner (Hamming similarity)
    pub fn resonance(&self, other: &TriangleCorner) -> f32 {
        self.fingerprint.similarity(&other.fingerprint)
    }
}

// =============================================================================
// VSA TRIANGLE - 3 corners bundled into one 10K-bit vector
// =============================================================================

/// A cognitive triangle with 3 VSA corners
#[derive(Clone)]
pub struct VsaTriangle {
    /// Corner 0 (e.g., Analytical, Abstract, Coherence, Immutable)
    pub corner0: TriangleCorner,

    /// Corner 1 (e.g., Intuitive, Concrete, Novelty, Hot)
    pub corner1: TriangleCorner,

    /// Corner 2 (e.g., Procedural, Relational, Resonance, Experimental)
    pub corner2: TriangleCorner,

    /// Superposition of all 3 corners (weighted bundle)
    superposition: Fingerprint,

    /// Triangle identifier
    pub id: TriangleId,
}

/// Triangle identifiers
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TriangleId {
    /// Processing Style (Analytical/Intuitive/Procedural)
    Processing,
    /// Content Focus (Abstract/Concrete/Relational)
    Content,
    /// Gestalt Integration (Coherence/Novelty/Resonance)
    Gestalt,
    /// L4 Crystallization (Immutable/Hot/Experimental)
    Crystallization,
}

impl VsaTriangle {
    /// Create triangle with labeled corners
    pub fn new(id: TriangleId, labels: [&'static str; 3]) -> Self {
        let corner0 = TriangleCorner::from_label(labels[0]);
        let corner1 = TriangleCorner::from_label(labels[1]);
        let corner2 = TriangleCorner::from_label(labels[2]);

        let superposition = Self::compute_superposition(&corner0, &corner1, &corner2);

        Self {
            corner0,
            corner1,
            corner2,
            superposition,
            id,
        }
    }

    /// Create processing triangle
    pub fn processing() -> Self {
        Self::new(
            TriangleId::Processing,
            ["Analytical", "Intuitive", "Procedural"],
        )
    }

    /// Create content triangle
    pub fn content() -> Self {
        Self::new(TriangleId::Content, ["Abstract", "Concrete", "Relational"])
    }

    /// Create gestalt triangle
    pub fn gestalt() -> Self {
        Self::new(TriangleId::Gestalt, ["Coherence", "Novelty", "Resonance"])
    }

    /// Create crystallization triangle
    pub fn crystallization() -> Self {
        Self::new(
            TriangleId::Crystallization,
            ["Immutable", "Hot", "Experimental"],
        )
    }

    /// Compute weighted superposition of corners
    fn compute_superposition(
        c0: &TriangleCorner,
        c1: &TriangleCorner,
        c2: &TriangleCorner,
    ) -> Fingerprint {
        // Weight by activation levels
        let total = c0.activation + c1.activation + c2.activation;
        if total < 0.001 {
            return Fingerprint::zero();
        }

        let _w0 = c0.activation / total;
        let _w1 = c1.activation / total;
        let _w2 = c2.activation / total;

        // Weighted bundle (majority vote with weights)
        let fps = vec![
            c0.fingerprint.clone(),
            c1.fingerprint.clone(),
            c2.fingerprint.clone(),
        ];

        // For now, simple bundle (proper weighted bundle would use vote counts)
        Fingerprint::bundle(&fps)
    }

    /// Update superposition after activation changes
    pub fn update_superposition(&mut self) {
        self.superposition =
            Self::compute_superposition(&self.corner0, &self.corner1, &self.corner2);
    }

    /// Set corner activations
    pub fn set_activations(&mut self, a0: f32, a1: f32, a2: f32) {
        self.corner0.activation = a0.clamp(0.0, 1.0);
        self.corner1.activation = a1.clamp(0.0, 1.0);
        self.corner2.activation = a2.clamp(0.0, 1.0);
        self.update_superposition();
    }

    /// Get activations as array
    pub fn activations(&self) -> [f32; 3] {
        [
            self.corner0.activation,
            self.corner1.activation,
            self.corner2.activation,
        ]
    }

    /// Resonance with another triangle (superposition similarity)
    pub fn resonance(&self, other: &VsaTriangle) -> f32 {
        self.superposition.similarity(&other.superposition)
    }

    /// Resonance with a query fingerprint
    pub fn query_resonance(&self, query: &Fingerprint) -> f32 {
        self.superposition.similarity(query)
    }

    /// Check if triangle is in FLOW state
    /// Flow = all corners active (>0.3) and balanced (range < 0.4)
    pub fn is_flow(&self) -> bool {
        let min_active = 0.3;
        let max_range = 0.4;

        let a = self.activations();
        let all_active = a.iter().all(|&x| x > min_active);
        let range = a.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            - a.iter().cloned().fold(f32::INFINITY, f32::min);

        all_active && range < max_range
    }

    /// Get dominant corner (highest activation)
    pub fn dominant_corner(&self) -> Option<usize> {
        let threshold = 0.6;
        let a = self.activations();
        let max_idx = a
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)?;

        if a[max_idx] >= threshold {
            Some(max_idx)
        } else {
            None
        }
    }

    /// Balance metric (0 = one dominant, 1 = perfectly balanced)
    pub fn balance(&self) -> f32 {
        let a = self.activations();
        let max = a.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = a.iter().cloned().fold(f32::INFINITY, f32::min);
        1.0 - (max - min)
    }

    /// Blend toward another triangle
    pub fn blend_toward(&mut self, target: &VsaTriangle, weight: f32) {
        let w = weight.clamp(0.0, 1.0);
        let ta = target.activations();

        self.corner0.activation = self.corner0.activation * (1.0 - w) + ta[0] * w;
        self.corner1.activation = self.corner1.activation * (1.0 - w) + ta[1] * w;
        self.corner2.activation = self.corner2.activation * (1.0 - w) + ta[2] * w;

        self.update_superposition();
    }

    /// Nudge toward target by delta
    pub fn nudge_toward(&mut self, target: &VsaTriangle, delta: f32) {
        let d = delta.clamp(0.0, 1.0);
        let ta = target.activations();

        self.corner0.activation += (ta[0] - self.corner0.activation) * d;
        self.corner1.activation += (ta[1] - self.corner1.activation) * d;
        self.corner2.activation += (ta[2] - self.corner2.activation) * d;

        // Clamp
        self.corner0.activation = self.corner0.activation.clamp(0.0, 1.0);
        self.corner1.activation = self.corner1.activation.clamp(0.0, 1.0);
        self.corner2.activation = self.corner2.activation.clamp(0.0, 1.0);

        self.update_superposition();
    }

    /// Get corner labels
    pub fn labels(&self) -> [&'static str; 3] {
        [self.corner0.label, self.corner1.label, self.corner2.label]
    }
}

impl fmt::Display for VsaTriangle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.activations();
        write!(f, "{:?}[{:.2}/{:.2}/{:.2}]", self.id, a[0], a[1], a[2])
    }
}

// =============================================================================
// QUAD-TRIANGLE - 4 interlocking triangles
// =============================================================================

/// Four interlocking cognitive triangles
#[derive(Clone)]
pub struct QuadTriangle {
    /// Triangle A: Processing Style
    pub processing: VsaTriangle,

    /// Triangle B: Content Focus
    pub content: VsaTriangle,

    /// Triangle C: Gestalt Integration
    pub gestalt: VsaTriangle,

    /// Triangle D: L4 Crystallization
    pub crystallization: VsaTriangle,

    /// Global superposition (bundle of all 4 triangle superpositions)
    global_superposition: Fingerprint,
}

impl QuadTriangle {
    /// Create neutral quad-triangle (all activations at 0.5)
    pub fn neutral() -> Self {
        let mut qt = Self {
            processing: VsaTriangle::processing(),
            content: VsaTriangle::content(),
            gestalt: VsaTriangle::gestalt(),
            crystallization: VsaTriangle::crystallization(),
            global_superposition: Fingerprint::zero(),
        };
        qt.update_global_superposition();
        qt
    }

    /// Create with specific activations
    pub fn with_activations(
        proc: [f32; 3],
        cont: [f32; 3],
        gest: [f32; 3],
        crys: [f32; 3],
    ) -> Self {
        let mut qt = Self::neutral();
        qt.processing.set_activations(proc[0], proc[1], proc[2]);
        qt.content.set_activations(cont[0], cont[1], cont[2]);
        qt.gestalt.set_activations(gest[0], gest[1], gest[2]);
        qt.crystallization
            .set_activations(crys[0], crys[1], crys[2]);
        qt.update_global_superposition();
        qt
    }

    /// Update global superposition
    fn update_global_superposition(&mut self) {
        let fps = vec![
            self.processing.superposition.clone(),
            self.content.superposition.clone(),
            self.gestalt.superposition.clone(),
            self.crystallization.superposition.clone(),
        ];
        self.global_superposition = Fingerprint::bundle(&fps);
    }

    /// Get triangle by ID
    pub fn get(&self, id: TriangleId) -> &VsaTriangle {
        match id {
            TriangleId::Processing => &self.processing,
            TriangleId::Content => &self.content,
            TriangleId::Gestalt => &self.gestalt,
            TriangleId::Crystallization => &self.crystallization,
        }
    }

    /// Get mutable triangle by ID
    pub fn get_mut(&mut self, id: TriangleId) -> &mut VsaTriangle {
        match id {
            TriangleId::Processing => &mut self.processing,
            TriangleId::Content => &mut self.content,
            TriangleId::Gestalt => &mut self.gestalt,
            TriangleId::Crystallization => &mut self.crystallization,
        }
    }

    /// Count triangles in FLOW state
    pub fn flow_count(&self) -> usize {
        let mut count = 0;
        if self.processing.is_flow() {
            count += 1;
        }
        if self.content.is_flow() {
            count += 1;
        }
        if self.gestalt.is_flow() {
            count += 1;
        }
        if self.crystallization.is_flow() {
            count += 1;
        }
        count
    }

    /// Check if all 4 triangles are in FLOW
    pub fn is_global_flow(&self) -> bool {
        self.flow_count() == 4
    }

    /// Resonance with query fingerprint
    pub fn query_resonance(&self, query: &Fingerprint) -> f32 {
        self.global_superposition.similarity(query)
    }

    /// Inter-triangle resonance matrix
    pub fn resonance_matrix(&self) -> [[f32; 4]; 4] {
        let triangles = [
            &self.processing,
            &self.content,
            &self.gestalt,
            &self.crystallization,
        ];

        let mut matrix = [[0.0f32; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                matrix[i][j] = if i == j {
                    1.0
                } else {
                    triangles[i].resonance(triangles[j])
                };
            }
        }
        matrix
    }

    /// Global coherence (average inter-triangle resonance)
    pub fn coherence(&self) -> f32 {
        let matrix = self.resonance_matrix();
        let mut sum = 0.0;
        let mut count = 0;

        for i in 0..4 {
            for j in (i + 1)..4 {
                sum += matrix[i][j];
                count += 1;
            }
        }

        if count > 0 { sum / count as f32 } else { 0.0 }
    }

    /// Cognitive signature (dominant corners)
    pub fn signature(&self) -> String {
        let mut parts = Vec::new();

        let labels = [
            (&self.processing, ["Analytical", "Intuitive", "Procedural"]),
            (&self.content, ["Abstract", "Concrete", "Relational"]),
            (&self.gestalt, ["Coherence", "Novelty", "Resonance"]),
        ];

        for (tri, names) in labels {
            if let Some(idx) = tri.dominant_corner() {
                parts.push(names[idx]);
            }
        }

        if parts.is_empty() {
            "Balanced".to_string()
        } else {
            parts.join("/")
        }
    }

    /// Blend toward another quad-triangle
    pub fn blend_toward(&mut self, target: &QuadTriangle, weight: f32) {
        self.processing.blend_toward(&target.processing, weight);
        self.content.blend_toward(&target.content, weight);
        self.gestalt.blend_toward(&target.gestalt, weight);
        self.crystallization
            .blend_toward(&target.crystallization, weight);
        self.update_global_superposition();
    }

    /// Nudge toward target
    pub fn nudge_toward(&mut self, target: &QuadTriangle, delta: f32) {
        self.processing.nudge_toward(&target.processing, delta);
        self.content.nudge_toward(&target.content, delta);
        self.gestalt.nudge_toward(&target.gestalt, delta);
        self.crystallization
            .nudge_toward(&target.crystallization, delta);
        self.update_global_superposition();
    }

    /// Serialize to 12 floats (for compatibility)
    pub fn to_floats(&self) -> [f32; 12] {
        let p = self.processing.activations();
        let c = self.content.activations();
        let g = self.gestalt.activations();
        let x = self.crystallization.activations();

        [
            p[0], p[1], p[2], c[0], c[1], c[2], g[0], g[1], g[2], x[0], x[1], x[2],
        ]
    }

    /// Deserialize from 12 floats
    pub fn from_floats(values: [f32; 12]) -> Self {
        Self::with_activations(
            [values[0], values[1], values[2]],
            [values[3], values[4], values[5]],
            [values[6], values[7], values[8]],
            [values[9], values[10], values[11]],
        )
    }

    /// Get global superposition fingerprint
    pub fn fingerprint(&self) -> &Fingerprint {
        &self.global_superposition
    }
}

impl Default for QuadTriangle {
    fn default() -> Self {
        Self::neutral()
    }
}

impl fmt::Display for QuadTriangle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QuadTriangle[{}|{}|{}|{}] sig={}",
            self.processing,
            self.content,
            self.gestalt,
            self.crystallization,
            self.signature()
        )
    }
}

// =============================================================================
// COGNITIVE PROFILES (predefined quad-triangles)
// =============================================================================

/// Predefined cognitive profiles
pub struct CognitiveProfiles;

impl CognitiveProfiles {
    /// Technical analysis mode
    pub fn analytical() -> QuadTriangle {
        QuadTriangle::with_activations(
            [0.9, 0.2, 0.7], // Processing: Analytical dominant
            [0.7, 0.3, 0.4], // Content: Abstract dominant
            [0.8, 0.2, 0.6], // Gestalt: Coherence dominant
            [0.6, 0.3, 0.1], // Crystallization: Immutable lean
        )
    }

    /// Creative exploration mode
    pub fn creative() -> QuadTriangle {
        QuadTriangle::with_activations(
            [0.3, 0.9, 0.3], // Processing: Intuitive dominant
            [0.8, 0.4, 0.7], // Content: Abstract + Relational
            [0.5, 0.9, 0.6], // Gestalt: Novelty dominant
            [0.2, 0.4, 0.8], // Crystallization: Experimental dominant
        )
    }

    /// Empathic understanding mode
    pub fn empathic() -> QuadTriangle {
        QuadTriangle::with_activations(
            [0.3, 0.8, 0.4], // Processing: Intuitive lean
            [0.2, 0.9, 0.8], // Content: Concrete + Relational
            [0.7, 0.4, 0.8], // Gestalt: Coherence + Resonance
            [0.3, 0.5, 0.4], // Crystallization: Balanced
        )
    }

    /// Procedural execution mode
    pub fn procedural() -> QuadTriangle {
        QuadTriangle::with_activations(
            [0.5, 0.2, 0.9], // Processing: Procedural dominant
            [0.3, 0.8, 0.3], // Content: Concrete dominant
            [0.8, 0.1, 0.5], // Gestalt: Coherence dominant
            [0.7, 0.6, 0.2], // Crystallization: Immutable + Hot
        )
    }

    /// Counterfactual reasoning mode
    pub fn counterfactual() -> QuadTriangle {
        QuadTriangle::with_activations(
            [0.6, 0.7, 0.4], // Processing: Balanced with intuitive lean
            [0.8, 0.3, 0.6], // Content: Abstract dominant
            [0.4, 0.8, 0.5], // Gestalt: Novelty dominant
            [0.2, 0.5, 0.9], // Crystallization: Experimental dominant
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangle_creation() {
        let tri = VsaTriangle::processing();
        assert_eq!(tri.labels(), ["Analytical", "Intuitive", "Procedural"]);
    }

    #[test]
    fn test_quad_triangle_coherence() {
        let qt = CognitiveProfiles::analytical();
        let coherence = qt.coherence();
        assert!(coherence >= 0.0 && coherence <= 1.0);
    }

    #[test]
    fn test_flow_detection() {
        // Balanced activations should be flow
        let mut qt = QuadTriangle::neutral();
        qt.processing.set_activations(0.5, 0.5, 0.5);
        assert!(qt.processing.is_flow());

        // Extreme activations should not be flow
        qt.processing.set_activations(0.9, 0.1, 0.1);
        assert!(!qt.processing.is_flow());
    }

    #[test]
    fn test_serialization() {
        let qt = CognitiveProfiles::creative();
        let floats = qt.to_floats();
        let restored = QuadTriangle::from_floats(floats);

        assert!((qt.coherence() - restored.coherence()).abs() < 0.01);
    }
}
