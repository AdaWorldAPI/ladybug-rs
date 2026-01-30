//! Causality Flow — Agency, Temporality, and Dependency
//!
//! The second vertex of the Grammar Triangle.
//! Captures WHO → DID → WHAT → WHY structure.

use crate::core::Fingerprint;

/// Causality flow structure
#[derive(Clone, Debug, Default)]
pub struct CausalityFlow {
    /// Agent (WHO) - if identified
    pub agent: Option<String>,
    
    /// Action (DID) - if identified
    pub action: Option<String>,
    
    /// Patient (WHAT) - if identified
    pub patient: Option<String>,
    
    /// Reason (WHY) - if identified
    pub reason: Option<String>,
    
    /// Temporal direction: -1.0 = past, 0.0 = present, 1.0 = future
    pub temporality: f32,
    
    /// Agency strength: 0.0 = passive, 1.0 = active
    pub agency: f32,
    
    /// Dependency type
    pub dependency: DependencyType,
    
    /// Causal strength (how certain is the causal link)
    pub causal_strength: f32,
}

/// Types of causal dependency
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DependencyType {
    #[default]
    None,
    /// A causes B
    Causal,
    /// A enables B (necessary but not sufficient)
    Enabling,
    /// A prevents B
    Preventing,
    /// A correlates with B (no direction)
    Correlational,
    /// A is part of B
    Constitutive,
    /// A intends B
    Intentional,
}

impl CausalityFlow {
    /// Create empty causality flow
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Extract causality from text using heuristics
    pub fn from_text(text: &str) -> Self {
        let text_lower = text.to_lowercase();
        
        // Temporal detection
        let temporality = Self::detect_temporality(&text_lower);
        
        // Agency detection
        let agency = Self::detect_agency(&text_lower);
        
        // Dependency detection
        let (dependency, causal_strength) = Self::detect_dependency(&text_lower);
        
        Self {
            agent: None,      // Would need NER for proper extraction
            action: None,
            patient: None,
            reason: None,
            temporality,
            agency,
            dependency,
            causal_strength,
        }
    }
    
    /// Detect temporal orientation from text
    fn detect_temporality(text: &str) -> f32 {
        let past_markers = ["was", "were", "had", "did", "been", "ago", "yesterday", 
                           "last", "before", "previously", "once", "used to"];
        let future_markers = ["will", "shall", "going to", "tomorrow", "next",
                             "soon", "later", "eventually", "plan to", "intend"];
        let present_markers = ["is", "are", "am", "now", "currently", "today",
                              "at present", "right now"];
        
        let past_count: usize = past_markers.iter()
            .map(|m| text.matches(m).count())
            .sum();
        let future_count: usize = future_markers.iter()
            .map(|m| text.matches(m).count())
            .sum();
        let present_count: usize = present_markers.iter()
            .map(|m| text.matches(m).count())
            .sum();
        
        let total = (past_count + future_count + present_count).max(1) as f32;
        
        // Weighted score: -1 (past) to +1 (future)
        let score = (future_count as f32 - past_count as f32) / total;
        score.clamp(-1.0, 1.0)
    }
    
    /// Detect agency level from text
    fn detect_agency(text: &str) -> f32 {
        let active_markers = ["i ", "we ", "he ", "she ", "they ", 
                             "decided", "chose", "made", "created",
                             "did", "performed", "executed", "initiated"];
        let passive_markers = ["was ", "were ", "been ", "being ",
                              "by the", "it was", "got ", "received",
                              "happened to", "occurred"];
        
        let active_count: usize = active_markers.iter()
            .map(|m| text.matches(m).count())
            .sum();
        let passive_count: usize = passive_markers.iter()
            .map(|m| text.matches(m).count())
            .sum();
        
        let total = (active_count + passive_count).max(1) as f32;
        let score = active_count as f32 / total;
        score.clamp(0.0, 1.0)
    }
    
    /// Detect dependency type and strength
    fn detect_dependency(text: &str) -> (DependencyType, f32) {
        // Causal markers
        let causal = ["because", "therefore", "thus", "hence", "so that",
                     "caused", "led to", "resulted in", "due to", "owing to"];
        let enabling = ["allows", "enables", "permits", "lets", "makes possible",
                       "if", "when", "provided that", "as long as"];
        let preventing = ["prevents", "stops", "blocks", "inhibits", "despite",
                         "although", "however", "but", "yet", "unless"];
        let correlational = ["correlates", "associated", "related to", "linked to",
                            "along with", "together with", "and also"];
        let intentional = ["wants to", "intends to", "plans to", "hopes to",
                          "in order to", "so as to", "for the purpose of"];
        
        let count_matches = |markers: &[&str]| -> usize {
            markers.iter().map(|m| text.matches(m).count()).sum()
        };
        
        let causal_count = count_matches(&causal);
        let enabling_count = count_matches(&enabling);
        let preventing_count = count_matches(&preventing);
        let correlational_count = count_matches(&correlational);
        let intentional_count = count_matches(&intentional);
        
        let max_count = [causal_count, enabling_count, preventing_count, 
                        correlational_count, intentional_count]
            .into_iter()
            .max()
            .unwrap_or(0);
        
        if max_count == 0 {
            return (DependencyType::None, 0.0);
        }
        
        let strength = (max_count as f32 * 0.3).min(1.0);
        
        let dep_type = if causal_count == max_count {
            DependencyType::Causal
        } else if enabling_count == max_count {
            DependencyType::Enabling
        } else if preventing_count == max_count {
            DependencyType::Preventing
        } else if correlational_count == max_count {
            DependencyType::Correlational
        } else if intentional_count == max_count {
            DependencyType::Intentional
        } else {
            DependencyType::None
        };
        
        (dep_type, strength)
    }
    
    /// Convert to vector representation
    pub fn to_vector(&self) -> [f32; 6] {
        [
            self.temporality,
            self.agency,
            self.causal_strength,
            self.dependency_to_float(),
            // Reserved for future expansion
            0.0,
            0.0,
        ]
    }
    
    /// Convert dependency type to float for vector representation
    fn dependency_to_float(&self) -> f32 {
        match self.dependency {
            DependencyType::None => 0.0,
            DependencyType::Causal => 0.2,
            DependencyType::Enabling => 0.4,
            DependencyType::Preventing => 0.6,
            DependencyType::Correlational => 0.8,
            DependencyType::Constitutive => 0.9,
            DependencyType::Intentional => 1.0,
        }
    }
    
    /// Similarity to another causality flow
    pub fn similarity(&self, other: &Self) -> f32 {
        let v1 = self.to_vector();
        let v2 = other.to_vector();
        
        // Cosine similarity
        let dot: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = v2.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm1 > 0.0 && norm2 > 0.0 {
            dot / (norm1 * norm2)
        } else {
            0.0
        }
    }
    
    /// Convert to fingerprint contribution
    pub fn to_fingerprint_contribution(&self) -> Fingerprint {
        let mut fp = Fingerprint::zero();
        
        // Encode temporality in bits 0-99
        let temp_bits = ((self.temporality + 1.0) * 50.0) as usize;
        for i in 0..temp_bits.min(100) {
            fp.set_bit(i, true);
        }
        
        // Encode agency in bits 100-199
        let agency_bits = (self.agency * 100.0) as usize;
        for i in 0..agency_bits.min(100) {
            fp.set_bit(100 + i, true);
        }
        
        // Encode dependency type in bits 200-299
        let dep_val = self.dependency_to_float();
        let dep_bits = (dep_val * 100.0) as usize;
        for i in 0..dep_bits.min(100) {
            fp.set_bit(200 + i, true);
        }
        
        // Encode causal strength in bits 300-399
        let strength_bits = (self.causal_strength * 100.0) as usize;
        for i in 0..strength_bits.min(100) {
            fp.set_bit(300 + i, true);
        }
        
        fp
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_temporality_detection() {
        let past = CausalityFlow::from_text("Yesterday I was walking in the park");
        let future = CausalityFlow::from_text("Tomorrow I will go to the store");
        let present = CausalityFlow::from_text("I am currently working on this");
        
        assert!(past.temporality < 0.0);
        assert!(future.temporality > 0.0);
        assert!(present.temporality.abs() < 0.5);
    }
    
    #[test]
    fn test_agency_detection() {
        let active = CausalityFlow::from_text("I decided to make a change and did it");
        let passive = CausalityFlow::from_text("It was done by someone else");
        
        assert!(active.agency > passive.agency);
    }
    
    #[test]
    fn test_dependency_detection() {
        let causal = CausalityFlow::from_text("The rain caused flooding because of the storm");
        assert_eq!(causal.dependency, DependencyType::Causal);
        
        let enabling = CausalityFlow::from_text("This allows us to proceed if conditions are met");
        assert_eq!(enabling.dependency, DependencyType::Enabling);
        
        let preventing = CausalityFlow::from_text("The wall prevents water from entering despite pressure");
        assert_eq!(preventing.dependency, DependencyType::Preventing);
    }
    
    #[test]
    fn test_similarity() {
        let flow1 = CausalityFlow::from_text("I will do this because I want to");
        let flow2 = CausalityFlow::from_text("I shall act since I desire it");
        let flow3 = CausalityFlow::from_text("It was done despite objections");
        
        let sim_12 = flow1.similarity(&flow2);
        let sim_13 = flow1.similarity(&flow3);
        
        // Similar temporal/agency should be closer
        assert!(sim_12 > sim_13);
    }
}
