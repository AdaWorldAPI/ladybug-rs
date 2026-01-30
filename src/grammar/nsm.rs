//! NSM (Natural Semantic Metalanguage) Primitives
//!
//! Based on Anna Wierzbicka's semantic primes - the irreducible
//! building blocks of meaning found across all human languages.
//!
//! These 65 primitives are the foundation of the Grammar Triangle.

use crate::core::Fingerprint;
use std::collections::HashMap;

/// The 65 NSM semantic primitives
pub const NSM_PRIMITIVES: [&str; 65] = [
    // Substantives (6)
    "I", "YOU", "SOMEONE", "SOMETHING", "PEOPLE", "BODY",
    
    // Determiners (3)
    "THIS", "THE_SAME", "OTHER",
    
    // Quantifiers (6)
    "ONE", "TWO", "SOME", "ALL", "MUCH", "MANY",
    
    // Evaluators (2)
    "GOOD", "BAD",
    
    // Descriptors (2)
    "BIG", "SMALL",
    
    // Mental predicates (6)
    "THINK", "KNOW", "WANT", "FEEL", "SEE", "HEAR",
    
    // Speech (3)
    "SAY", "WORDS", "TRUE",
    
    // Actions/Events (4)
    "DO", "HAPPEN", "MOVE", "TOUCH",
    
    // Existence/Possession (2)
    "THERE_IS", "HAVE",
    
    // Life/Death (2)
    "LIVE", "DIE",
    
    // Time (8)
    "WHEN", "NOW", "BEFORE", "AFTER", 
    "A_LONG_TIME", "A_SHORT_TIME", "FOR_SOME_TIME", "MOMENT",
    
    // Space (8)
    "WHERE", "HERE", "ABOVE", "BELOW", 
    "FAR", "NEAR", "SIDE", "INSIDE",
    
    // Logical concepts (5)
    "NOT", "MAYBE", "CAN", "BECAUSE", "IF",
    
    // Intensifier (1)
    "VERY",
    
    // Similarity (1)
    "LIKE",
    
    // Augmentatives (6) - Extended set
    "MORE", "PART", "KIND", "WORD", "SAY", "THINK",
];

/// Keyword activations for each primitive
/// Maps NSM primitive → keywords that activate it
static ACTIVATIONS: &[(&str, &[&str])] = &[
    ("I", &["i", "me", "my", "myself", "mine"]),
    ("YOU", &["you", "your", "yourself", "yours"]),
    ("SOMEONE", &["someone", "person", "one", "who", "character", "individual"]),
    ("SOMETHING", &["something", "thing", "it", "what", "object"]),
    ("PEOPLE", &["people", "they", "them", "everyone", "folks", "humans"]),
    ("BODY", &["body", "physical", "flesh", "corporal", "bodily"]),
    
    ("THIS", &["this", "these", "here", "now"]),
    ("THE_SAME", &["same", "identical", "equal", "equivalent"]),
    ("OTHER", &["other", "another", "else", "different", "alternative"]),
    
    ("ONE", &["one", "single", "a", "an", "alone"]),
    ("TWO", &["two", "both", "pair", "couple", "dual"]),
    ("SOME", &["some", "few", "several", "certain"]),
    ("ALL", &["all", "every", "each", "entire", "whole", "complete"]),
    ("MUCH", &["much", "lot", "plenty", "abundant"]),
    ("MANY", &["many", "numerous", "multiple", "countless"]),
    
    ("GOOD", &["good", "great", "beautiful", "wonderful", "excellent", "positive", "right"]),
    ("BAD", &["bad", "wrong", "terrible", "awful", "negative", "evil", "poor"]),
    
    ("BIG", &["big", "large", "huge", "enormous", "vast", "great"]),
    ("SMALL", &["small", "little", "tiny", "minute", "slight"]),
    
    ("THINK", &["think", "consider", "suppose", "ponder", "wonder", "believe", "imagine"]),
    ("KNOW", &["know", "understand", "realize", "aware", "recognize", "comprehend"]),
    ("WANT", &["want", "desire", "wish", "need", "yearn", "long", "crave"]),
    ("FEEL", &["feel", "emotion", "sense", "experience", "heart", "sensation"]),
    ("SEE", &["see", "look", "gaze", "watch", "observe", "behold", "view", "witness"]),
    ("HEAR", &["hear", "listen", "sound", "audible"]),
    
    ("SAY", &["say", "tell", "speak", "mention", "whisper", "cry", "state", "declare"]),
    ("WORDS", &["words", "language", "speech", "verbal", "written"]),
    ("TRUE", &["true", "truth", "real", "actual", "fact", "genuine"]),
    
    ("DO", &["do", "make", "create", "perform", "act", "execute", "accomplish"]),
    ("HAPPEN", &["happen", "occur", "event", "take place", "transpire"]),
    ("MOVE", &["move", "motion", "go", "travel", "shift", "transfer"]),
    ("TOUCH", &["touch", "contact", "feel", "handle", "reach"]),
    
    ("THERE_IS", &["there is", "exists", "presence", "being"]),
    ("HAVE", &["have", "possess", "own", "hold", "contain"]),
    
    ("LIVE", &["live", "alive", "life", "living", "vital"]),
    ("DIE", &["die", "death", "dead", "dying", "perish", "end"]),
    
    ("WHEN", &["when", "time", "moment", "while", "during"]),
    ("NOW", &["now", "present", "current", "today", "instant"]),
    ("BEFORE", &["before", "past", "once", "ago", "earlier", "previously"]),
    ("AFTER", &["after", "then", "future", "next", "later", "subsequently"]),
    ("A_LONG_TIME", &["long time", "ages", "forever", "extended", "prolonged"]),
    ("A_SHORT_TIME", &["short time", "brief", "moment", "instant", "quick"]),
    ("FOR_SOME_TIME", &["for some time", "while", "period", "duration"]),
    ("MOMENT", &["moment", "instant", "second", "flash"]),
    
    ("WHERE", &["where", "place", "location", "position"]),
    ("HERE", &["here", "this place", "present"]),
    ("ABOVE", &["above", "over", "up", "higher", "top"]),
    ("BELOW", &["below", "under", "down", "lower", "beneath"]),
    ("FAR", &["far", "distant", "remote", "away"]),
    ("NEAR", &["near", "close", "nearby", "adjacent"]),
    ("SIDE", &["side", "beside", "next to", "lateral"]),
    ("INSIDE", &["inside", "within", "interior", "inner"]),
    
    ("NOT", &["not", "no", "never", "none", "neither", "without"]),
    ("MAYBE", &["maybe", "perhaps", "possibly", "might", "could"]),
    ("CAN", &["can", "able", "capable", "possible", "may"]),
    ("BECAUSE", &["because", "for", "since", "reason", "cause", "therefore"]),
    ("IF", &["if", "whether", "condition", "suppose", "assuming"]),
    
    ("VERY", &["very", "extremely", "highly", "really", "quite", "so"]),
    
    ("LIKE", &["like", "similar", "as", "resemble", "same as"]),
];

/// NSM field: continuous weights over 65 primitives
#[derive(Clone, Debug)]
pub struct NSMField {
    weights: [f32; 65],
}

impl Default for NSMField {
    fn default() -> Self {
        Self { weights: [0.0; 65] }
    }
}

impl NSMField {
    /// Create empty field
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Compute NSM field from text using keyword activation
    pub fn from_text(text: &str) -> Self {
        let text_lower = text.to_lowercase();
        let mut weights = [0.0f32; 65];
        
        for (i, primitive) in NSM_PRIMITIVES.iter().enumerate() {
            // Find activation keywords for this primitive
            let keywords = ACTIVATIONS.iter()
                .find(|(p, _)| p == primitive)
                .map(|(_, kws)| *kws)
                .unwrap_or(&[]);
            
            // Count keyword occurrences
            let count: usize = keywords.iter()
                .map(|kw| text_lower.matches(kw).count())
                .sum();
            
            // Soft saturation: asymptotic approach to 1.0
            weights[i] = (count as f32 * 0.25).min(1.0);
        }
        
        Self { weights }
    }
    
    /// Get weight for a specific primitive
    pub fn weight(&self, primitive: &str) -> Option<f32> {
        NSM_PRIMITIVES.iter()
            .position(|p| *p == primitive)
            .map(|i| self.weights[i])
    }
    
    /// Set weight for a specific primitive
    pub fn set_weight(&mut self, primitive: &str, weight: f32) {
        if let Some(i) = NSM_PRIMITIVES.iter().position(|p| *p == primitive) {
            self.weights[i] = weight.clamp(0.0, 1.0);
        }
    }
    
    /// Get all weights as slice
    pub fn weights(&self) -> &[f32; 65] {
        &self.weights
    }
    
    /// Get top N activated primitives
    pub fn top_activations(&self, n: usize) -> Vec<(&'static str, f32)> {
        let mut sorted: Vec<_> = NSM_PRIMITIVES.iter()
            .zip(self.weights.iter())
            .map(|(p, w)| (*p, *w))
            .collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(n);
        sorted
    }
    
    /// Dot product with another NSM field (similarity)
    pub fn dot(&self, other: &Self) -> f32 {
        self.weights.iter()
            .zip(other.weights.iter())
            .map(|(a, b)| a * b)
            .sum()
    }
    
    /// L2 norm
    pub fn norm(&self) -> f32 {
        self.weights.iter()
            .map(|w| w * w)
            .sum::<f32>()
            .sqrt()
    }
    
    /// Cosine similarity with another NSM field
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        let dot = self.dot(other);
        let norm_product = self.norm() * other.norm();
        if norm_product > 0.0 {
            dot / norm_product
        } else {
            0.0
        }
    }
    
    /// Convert to fingerprint contribution
    /// Uses deterministic projection based on primitive index
    pub fn to_fingerprint_contribution(&self) -> Fingerprint {
        let mut fp = Fingerprint::zero();
        
        for (i, weight) in self.weights.iter().enumerate() {
            if *weight > 0.3 {
                // Deterministic "random" projection based on primitive index
                // Uses golden ratio hash for good distribution
                let seed = (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
                
                // Set bits based on weight and seed
                let num_bits = (*weight * 100.0) as usize;
                for j in 0..num_bits.min(157) {
                    let bit_pos = (seed.wrapping_mul((j + 1) as u64) % 10000) as usize;
                    fp.set_bit(bit_pos, true);
                }
            }
        }
        
        fp
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_nsm_from_text() {
        let field = NSMField::from_text("I want to know what you think about this");
        
        assert!(field.weight("I").unwrap() > 0.0);
        assert!(field.weight("WANT").unwrap() > 0.0);
        assert!(field.weight("KNOW").unwrap() > 0.0);
        assert!(field.weight("YOU").unwrap() > 0.0);
        assert!(field.weight("THINK").unwrap() > 0.0);
        assert!(field.weight("THIS").unwrap() > 0.0);
    }
    
    #[test]
    fn test_top_activations() {
        let field = NSMField::from_text("I feel good about this beautiful day");
        let top = field.top_activations(3);
        
        assert!(!top.is_empty());
        // Check that activations are sorted descending
        for i in 1..top.len() {
            assert!(top[i-1].1 >= top[i].1);
        }
    }
    
    #[test]
    fn test_cosine_similarity() {
        let field1 = NSMField::from_text("I want to know");
        let field2 = NSMField::from_text("I desire to understand");
        let field3 = NSMField::from_text("The big tree is above the small house");
        
        let sim_12 = field1.cosine_similarity(&field2);
        let sim_13 = field1.cosine_similarity(&field3);
        
        // Mental predicates should be more similar to each other
        assert!(sim_12 > sim_13);
    }
    
    #[test]
    fn test_fingerprint_contribution() {
        let field = NSMField::from_text("I feel something beautiful");
        let fp = field.to_fingerprint_contribution();
        
        // Should have some bits set
        assert!(fp.popcount() > 0);
    }
}
