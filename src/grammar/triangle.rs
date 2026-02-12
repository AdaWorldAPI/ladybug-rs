//! Grammar Triangle — The Universal Input Layer
//!
//! The Grammar Triangle embeds meaning in a three-way continuous field:
//!
//! ```text
//!         CAUSALITY (flows, not causes)
//!               /\
//!              /  \
//!             /    \
//!     NSM <──⊕──> QUALIA (18D field)
//!   (65 primes)     (continuous)
//!         │
//!         ↓
//!    10Kbit FINGERPRINT
//!    (holds superposition)
//! ```
//!
//! The triangle creates a continuous field where meaning flows without
//! collapsing into discrete categories.

use crate::core::Fingerprint;
use super::nsm::NSMField;
use super::causality::CausalityFlow;
use super::qualia::QualiaField;

/// The complete Grammar Triangle
#[derive(Clone, Debug)]
pub struct GrammarTriangle {
    /// NSM field: weights over 65 semantic primitives
    pub nsm: NSMField,
    
    /// Causality flow: agency, temporality, dependency
    pub causality: CausalityFlow,
    
    /// Qualia field: 18D phenomenal coordinates
    pub qualia: QualiaField,
}

impl Default for GrammarTriangle {
    fn default() -> Self {
        Self {
            nsm: NSMField::default(),
            causality: CausalityFlow::default(),
            qualia: QualiaField::default(),
        }
    }
}

impl GrammarTriangle {
    /// Create empty triangle
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Compute Grammar Triangle from text
    /// 
    /// This is the main entry point for text → meaning
    pub fn from_text(text: &str) -> Self {
        Self {
            nsm: NSMField::from_text(text),
            causality: CausalityFlow::from_text(text),
            qualia: QualiaField::from_text(text),
        }
    }
    
    /// Generate 10Kbit fingerprint from the triangle
    /// 
    /// The fingerprint encodes all three vertices:
    /// - Bits 0-3999: NSM contribution
    /// - Bits 4000-5999: Causality contribution  
    /// - Bits 6000-9999: Qualia contribution
    pub fn to_fingerprint(&self) -> Fingerprint {
        let nsm_fp = self.nsm.to_fingerprint_contribution();
        let causality_fp = self.causality.to_fingerprint_contribution();
        let qualia_fp = self.qualia.to_fingerprint_contribution();
        
        // XOR all contributions together
        let mut result = Fingerprint::zero();
        
        // NSM contribution
        for i in 0..4000 {
            if nsm_fp.get_bit(i % crate::FINGERPRINT_BITS) {
                result.set_bit(i, true);
            }
        }
        
        // Causality contribution (shifted to 4000-5999)
        for i in 0..2000 {
            if causality_fp.get_bit(i % crate::FINGERPRINT_BITS) {
                result.set_bit(4000 + i, true);
            }
        }
        
        // Qualia contribution (shifted to 6000-9999)
        for i in 0..4000 {
            if qualia_fp.get_bit(i % crate::FINGERPRINT_BITS) {
                result.set_bit(6000 + i, true);
            }
        }
        
        result
    }
    
    /// Compute Hamming similarity to another triangle
    pub fn similarity(&self, other: &Self) -> f32 {
        let fp1 = self.to_fingerprint();
        let fp2 = other.to_fingerprint();
        fp1.similarity(&fp2)
    }
    
    /// Compute weighted similarity with explicit vertex weights
    pub fn weighted_similarity(&self, other: &Self, nsm_weight: f32, causality_weight: f32, qualia_weight: f32) -> f32 {
        let total_weight = nsm_weight + causality_weight + qualia_weight;
        if total_weight == 0.0 {
            return 0.0;
        }
        
        let nsm_sim = self.nsm.cosine_similarity(&other.nsm);
        let causality_sim = self.causality.similarity(&other.causality);
        let qualia_sim = self.qualia.similarity(&other.qualia);
        
        (nsm_weight * nsm_sim + causality_weight * causality_sim + qualia_weight * qualia_sim) / total_weight
    }
    
    /// Get the dominant NSM primitive
    pub fn dominant_nsm(&self) -> Option<(&'static str, f32)> {
        self.nsm.top_activations(1).into_iter().next()
    }
    
    /// Get the top N activated NSM primitives
    pub fn top_nsm(&self, n: usize) -> Vec<(&'static str, f32)> {
        self.nsm.top_activations(n)
    }
    
    /// Get the qualia coordinate for a dimension
    pub fn qualia(&self, dimension: &str) -> Option<f32> {
        self.qualia.get(dimension)
    }
    
    /// Get the temporality (-1 past, 0 present, +1 future)
    pub fn temporality(&self) -> f32 {
        self.causality.temporality
    }
    
    /// Get the agency (0 passive, 1 active)
    pub fn agency(&self) -> f32 {
        self.causality.agency
    }
    
    /// Is this triangle predominantly positive valence?
    pub fn is_positive(&self) -> bool {
        self.qualia.get("valence").unwrap_or(0.5) > 0.5
    }
    
    /// Is this triangle high arousal?
    pub fn is_high_arousal(&self) -> bool {
        self.qualia.get("arousal").unwrap_or(0.5) > 0.6
    }
    
    /// Is this triangle future-oriented?
    pub fn is_future_oriented(&self) -> bool {
        self.causality.temporality > 0.2
    }
    
    /// Is this triangle past-oriented?
    pub fn is_past_oriented(&self) -> bool {
        self.causality.temporality < -0.2
    }
    
    /// Get a summary of the triangle's characteristics
    pub fn summary(&self) -> TriangleSummary {
        TriangleSummary {
            top_nsm: self.top_nsm(3),
            temporality: self.temporality(),
            agency: self.agency(),
            valence: self.qualia("valence").unwrap_or(0.5),
            arousal: self.qualia("arousal").unwrap_or(0.5),
            certainty: self.qualia("certainty").unwrap_or(0.5),
        }
    }
}

/// Summary of a triangle's key characteristics
#[derive(Clone, Debug)]
pub struct TriangleSummary {
    pub top_nsm: Vec<(&'static str, f32)>,
    pub temporality: f32,
    pub agency: f32,
    pub valence: f32,
    pub arousal: f32,
    pub certainty: f32,
}

impl std::fmt::Display for TriangleSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let nsm_str: Vec<_> = self.top_nsm.iter()
            .map(|(p, w)| format!("{}:{:.2}", p, w))
            .collect();
        
        write!(f, "NSM[{}] T:{:+.2} A:{:.2} V:{:.2} Ar:{:.2} C:{:.2}",
            nsm_str.join(","),
            self.temporality,
            self.agency,
            self.valence,
            self.arousal,
            self.certainty
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_from_text() {
        let romeo = GrammarTriangle::from_text(
            "But soft! What light through yonder window breaks? It is the east, and Juliet is the sun."
        );
        
        // Should detect some mental/perceptual primitives
        let top = romeo.top_nsm(5);
        assert!(!top.is_empty());
        
        // Should have some qualia
        assert!(romeo.qualia("arousal").is_some());
    }
    
    #[test]
    fn test_similarity() {
        let romeo = GrammarTriangle::from_text(
            "But soft! What light through yonder window breaks? It is the east, and Juliet is the sun."
        );
        
        let juliet = GrammarTriangle::from_text(
            "Lady Juliet gazed longingly at the stars, her heart aching for Romeo."
        );
        
        let financial = GrammarTriangle::from_text(
            "The quarterly financial report shows a 5% increase in revenue."
        );
        
        let sim_rj = romeo.similarity(&juliet);
        let sim_rf = romeo.similarity(&financial);
        
        // Love scenes should be more similar to each other than to financial reports
        assert!(sim_rj > sim_rf, "Romeo-Juliet similarity ({}) should be > Romeo-Financial ({})", sim_rj, sim_rf);
    }
    
    #[test]
    fn test_temporality() {
        let past = GrammarTriangle::from_text("Yesterday I was walking in the park with my old friend");
        let future = GrammarTriangle::from_text("Tomorrow I will travel to the new destination");
        
        assert!(past.is_past_oriented());
        assert!(future.is_future_oriented());
    }
    
    #[test]
    fn test_valence() {
        let positive = GrammarTriangle::from_text("I love this beautiful wonderful amazing day");
        let negative = GrammarTriangle::from_text("I hate this terrible awful horrible situation");
        
        assert!(positive.is_positive());
        assert!(!negative.is_positive());
    }
    
    #[test]
    fn test_fingerprint() {
        let triangle = GrammarTriangle::from_text("I want to know something good");
        let fp = triangle.to_fingerprint();
        
        // Should have some bits set
        assert!(fp.popcount() > 0);
        assert!(fp.popcount() < crate::FINGERPRINT_BITS as u32); // Not all bits
    }
    
    #[test]
    fn test_summary() {
        let triangle = GrammarTriangle::from_text("I feel deeply happy about this wonderful moment");
        let summary = triangle.summary();
        
        println!("Summary: {}", summary);
        assert!(!summary.top_nsm.is_empty());
    }
}
