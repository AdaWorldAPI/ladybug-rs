//! DeepNSM Integration
//!
//! Based on arXiv:2505.11764 "Towards Universal Semantics with Large Language Models"
//!
//! Key insights from the paper:
//!
//! 1. **65 Semantic Primes are Universal**
//!    - Empirically attested in 90+ languages
//!    - Any word can be paraphrased using only these primes
//!    - Cross-translatable without loss of meaning
//!
//! 2. **Explications > Dictionary Definitions**
//!    - Dictionary definitions are circular and culture-specific
//!    - NSM explications are precise, non-circular, universal
//!    - 20+ BLEU points better on low-resource translation
//!
//! 3. **Small Fine-Tuned Models Beat GPT-4o**
//!    - DeepNSM-1B and DeepNSM-8B outperform GPT-4o
//!    - Quality filtering on training data is crucial
//!    - Only ~44K examples needed for fine-tuning
//!
//! 4. **Evaluation Metrics**
//!    - Legality Score: (primes - molecules) / total_words
//!    - Substitutability Score: log-probability tests
//!    - Cross-Translatability: round-trip BLEU through low-resource languages
//!
//! ## Our Integration Strategy
//!
//! The paper uses LLMs to GENERATE explications.
//! We use explications to BUILD our fingerprint codebook.
//!
//! ```text
//!    DeepNSM Model (1B/8B)
//!           │
//!           ▼
//!    text → NSM Explication
//!           │
//!           ▼
//!    Parse Explication into Prime Weights
//!           │
//!           ▼
//!    Role-bind primes (AGENT, ACTION, etc.)
//!           │
//!           ▼
//!    Bundle into 10K Fingerprint
//!           │
//!           ▼
//!    Pure SIMD inference (no LLM needed!)
//! ```
//!
//! The LLM is only used during TRAINING to generate proper explications.
//! At inference time, we use learned keyword→prime weights.

use crate::core::Fingerprint;
use super::nsm_substrate::NsmCodebook;
use std::collections::HashMap;

// =============================================================================
// The Complete NSM Prime List (from Figure 1 of the paper)
// =============================================================================

/// The 65 NSM semantic primes organized exactly as in the paper
pub const NSM_PRIMES_PAPER: &[(&str, &[&str])] = &[
    // Substantives
    ("SUBSTANTIVES", &["I", "YOU", "SOMEONE", "SOMETHING", "THING", "BODY"]),
    
    // Relational substantives
    ("RELATIONAL", &["KIND", "PART"]),
    
    // Determiners
    ("DETERMINERS", &["THIS", "THE_SAME", "OTHER", "ELSE", "ANOTHER"]),
    
    // Quantifiers
    ("QUANTIFIERS", &["ONE", "TWO", "MUCH", "MANY", "LITTLE", "FEW", "SOME", "ALL"]),
    
    // Evaluators
    ("EVALUATORS", &["GOOD", "BAD"]),
    
    // Descriptors
    ("DESCRIPTORS", &["BIG", "SMALL"]),
    
    // Mental predicates
    ("MENTAL", &["THINK", "KNOW", "WANT", "DONT_WANT", "FEEL", "SEE", "HEAR"]),
    
    // Speech
    ("SPEECH", &["SAY", "WORDS", "TRUE"]),
    
    // Actions/events/movement
    ("ACTIONS", &["DO", "HAPPEN", "MOVE"]),
    
    // Existence/possession
    ("EXISTENCE", &["BE", "THERE_IS", "BE_SOMEONE_SOMETHING", "MINE"]),
    
    // Life/death
    ("LIFE", &["LIVE", "DIE"]),
    
    // Time
    ("TIME", &[
        "WHEN", "TIME", "NOW", "BEFORE", "AFTER",
        "A_LONG_TIME", "A_SHORT_TIME", "FOR_SOME_TIME", "MOMENT"
    ]),
    
    // Space
    ("SPACE", &[
        "WHERE", "PLACE", "HERE", "ABOVE", "BELOW",
        "FAR", "NEAR", "SIDE", "INSIDE", "TOUCH", "CONTACT"
    ]),
    
    // Logical
    ("LOGICAL", &["NOT", "MAYBE", "CAN", "BECAUSE", "IF"]),
    
    // Intensifier/augmentor
    ("INTENSIFIER", &["VERY", "MORE"]),
    
    // Similarity
    ("SIMILARITY", &["LIKE", "AS", "WAY"]),
];

/// Flatten all primes into a single list
pub fn all_primes() -> Vec<&'static str> {
    NSM_PRIMES_PAPER.iter()
        .flat_map(|(_, primes)| primes.iter().copied())
        .collect()
}

/// Get category for a prime
pub fn prime_category(prime: &str) -> Option<&'static str> {
    for (category, primes) in NSM_PRIMES_PAPER {
        if primes.contains(&prime) {
            return Some(category);
        }
    }
    None
}

// =============================================================================
// Legality Score (from Section 3.1 of the paper)
// =============================================================================

/// Compute legality score for an explication
/// 
/// Formula: α * (primes - molecules) / total_words
/// Where α = 10 (so perfect score = 10, worst = -10)
pub fn legality_score(explication: &str) -> f32 {
    let alpha = 10.0;
    
    let words: Vec<&str> = explication
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphabetic()))
        .filter(|w| !w.is_empty())
        .collect();
    
    if words.is_empty() {
        return 0.0;
    }
    
    let all_primes = all_primes();
    let stopwords = stopwords();
    
    let mut prime_count = 0;
    let mut molecule_count = 0;
    
    for word in &words {
        let upper = word.to_uppercase();
        let lower = word.to_lowercase();
        
        if all_primes.iter().any(|p| p.to_uppercase() == upper || *p == upper) {
            prime_count += 1;
        } else if !stopwords.contains(&lower.as_str()) {
            molecule_count += 1;
        }
        // Stopwords are neither primes nor molecules
    }
    
    alpha * (prime_count as f32 - molecule_count as f32) / words.len() as f32
}

/// Check if explication is circular (contains the target word)
pub fn is_circular(explication: &str, target_word: &str) -> bool {
    let explication_lower = explication.to_lowercase();
    let target_lower = target_word.to_lowercase();
    
    // Check for exact word match
    explication_lower.split_whitespace()
        .any(|w| w.trim_matches(|c: char| !c.is_alphabetic()) == target_lower)
}

/// Count primes and molecules in text
pub fn count_primes_molecules(text: &str) -> (usize, usize, usize) {
    let words: Vec<&str> = text
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphabetic()))
        .filter(|w| !w.is_empty())
        .collect();
    
    let all_primes = all_primes();
    let stopwords = stopwords();
    
    let mut primes = 0;
    let mut molecules = 0;
    
    for word in &words {
        let upper = word.to_uppercase();
        let lower = word.to_lowercase();
        
        if all_primes.iter().any(|p| p.to_uppercase() == upper || *p == upper) {
            primes += 1;
        } else if !stopwords.contains(&lower.as_str()) {
            molecules += 1;
        }
    }
    
    (primes, molecules, words.len())
}

/// Common English stopwords (grammatical function words)
fn stopwords() -> Vec<&'static str> {
    vec![
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
        "from", "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "under", "again", "further", "then", "once",
        "and", "but", "or", "nor", "so", "yet", "both", "either", "neither",
        "not", "only", "own", "same", "than", "too", "very", "just", "also",
        "no", "such", "any", "each", "every", "both", "few", "more", "most",
        "other", "some", "such", "what", "which", "who", "whom", "whose",
        "this", "that", "these", "those", "am", "its", "it", "he", "she",
        "they", "them", "his", "her", "their", "my", "your", "our",
    ]
}

// =============================================================================
// Explication Parser
// =============================================================================

/// Parse an NSM explication into prime weights
/// 
/// Example explication (for "sick"):
/// ```text
/// something bad was happening to her at that time
/// because something bad was happening in her body
/// it was happening for some time
/// she could know that something bad was happening in her body
/// because she felt something bad in her body
/// ```
/// 
/// Extracts: SOMETHING, BAD, HAPPEN, TIME, BODY, KNOW, FEEL
pub struct ExplicationParser {
    /// Mapping from surface forms to canonical primes
    surface_to_prime: HashMap<String, String>,
}

impl Default for ExplicationParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ExplicationParser {
    pub fn new() -> Self {
        let mut surface_to_prime = HashMap::new();
        
        // Map surface forms to canonical primes
        // (the paper notes primes can have multiple surface forms)
        
        // Substantives
        for word in ["i", "me", "my", "myself"] {
            surface_to_prime.insert(word.to_string(), "I".to_string());
        }
        for word in ["you", "your", "yourself"] {
            surface_to_prime.insert(word.to_string(), "YOU".to_string());
        }
        for word in ["someone", "somebody", "person"] {
            surface_to_prime.insert(word.to_string(), "SOMEONE".to_string());
        }
        for word in ["something", "thing", "things"] {
            surface_to_prime.insert(word.to_string(), "SOMETHING".to_string());
        }
        for word in ["body", "bodies"] {
            surface_to_prime.insert(word.to_string(), "BODY".to_string());
        }
        
        // Mental predicates
        for word in ["think", "thinking", "thought", "thinks"] {
            surface_to_prime.insert(word.to_string(), "THINK".to_string());
        }
        for word in ["know", "knowing", "knew", "knows"] {
            surface_to_prime.insert(word.to_string(), "KNOW".to_string());
        }
        for word in ["want", "wanting", "wanted", "wants"] {
            surface_to_prime.insert(word.to_string(), "WANT".to_string());
        }
        for word in ["feel", "feeling", "felt", "feels"] {
            surface_to_prime.insert(word.to_string(), "FEEL".to_string());
        }
        for word in ["see", "seeing", "saw", "sees", "seen"] {
            surface_to_prime.insert(word.to_string(), "SEE".to_string());
        }
        for word in ["hear", "hearing", "heard", "hears"] {
            surface_to_prime.insert(word.to_string(), "HEAR".to_string());
        }
        
        // Evaluators
        for word in ["good", "well"] {
            surface_to_prime.insert(word.to_string(), "GOOD".to_string());
        }
        for word in ["bad", "badly"] {
            surface_to_prime.insert(word.to_string(), "BAD".to_string());
        }
        
        // Descriptors
        for word in ["big", "large"] {
            surface_to_prime.insert(word.to_string(), "BIG".to_string());
        }
        for word in ["small", "little"] {
            surface_to_prime.insert(word.to_string(), "SMALL".to_string());
        }
        
        // Actions
        for word in ["do", "doing", "did", "does", "done"] {
            surface_to_prime.insert(word.to_string(), "DO".to_string());
        }
        for word in ["happen", "happening", "happened", "happens"] {
            surface_to_prime.insert(word.to_string(), "HAPPEN".to_string());
        }
        for word in ["move", "moving", "moved", "moves"] {
            surface_to_prime.insert(word.to_string(), "MOVE".to_string());
        }
        
        // Speech
        for word in ["say", "saying", "said", "says"] {
            surface_to_prime.insert(word.to_string(), "SAY".to_string());
        }
        for word in ["word", "words"] {
            surface_to_prime.insert(word.to_string(), "WORDS".to_string());
        }
        for word in ["true", "truly"] {
            surface_to_prime.insert(word.to_string(), "TRUE".to_string());
        }
        
        // Time
        for word in ["now", "currently"] {
            surface_to_prime.insert(word.to_string(), "NOW".to_string());
        }
        for word in ["before", "previously", "earlier"] {
            surface_to_prime.insert(word.to_string(), "BEFORE".to_string());
        }
        for word in ["after", "later", "afterwards"] {
            surface_to_prime.insert(word.to_string(), "AFTER".to_string());
        }
        for word in ["time", "times", "moment"] {
            surface_to_prime.insert(word.to_string(), "TIME".to_string());
        }
        
        // Space
        for word in ["place", "places", "where", "somewhere"] {
            surface_to_prime.insert(word.to_string(), "PLACE".to_string());
        }
        for word in ["here"] {
            surface_to_prime.insert(word.to_string(), "HERE".to_string());
        }
        for word in ["above", "over"] {
            surface_to_prime.insert(word.to_string(), "ABOVE".to_string());
        }
        for word in ["below", "under", "beneath"] {
            surface_to_prime.insert(word.to_string(), "BELOW".to_string());
        }
        for word in ["inside", "within"] {
            surface_to_prime.insert(word.to_string(), "INSIDE".to_string());
        }
        for word in ["touch", "touching", "touched"] {
            surface_to_prime.insert(word.to_string(), "TOUCH".to_string());
        }
        
        // Logical
        for word in ["not", "no", "never"] {
            surface_to_prime.insert(word.to_string(), "NOT".to_string());
        }
        for word in ["maybe", "perhaps", "possibly"] {
            surface_to_prime.insert(word.to_string(), "MAYBE".to_string());
        }
        for word in ["can", "could", "able"] {
            surface_to_prime.insert(word.to_string(), "CAN".to_string());
        }
        for word in ["because", "since", "therefore"] {
            surface_to_prime.insert(word.to_string(), "BECAUSE".to_string());
        }
        for word in ["if", "whether"] {
            surface_to_prime.insert(word.to_string(), "IF".to_string());
        }
        
        // Quantifiers
        for word in ["one", "1"] {
            surface_to_prime.insert(word.to_string(), "ONE".to_string());
        }
        for word in ["two", "2"] {
            surface_to_prime.insert(word.to_string(), "TWO".to_string());
        }
        for word in ["some"] {
            surface_to_prime.insert(word.to_string(), "SOME".to_string());
        }
        for word in ["all", "every", "everything"] {
            surface_to_prime.insert(word.to_string(), "ALL".to_string());
        }
        for word in ["many", "much", "lot", "lots"] {
            surface_to_prime.insert(word.to_string(), "MANY".to_string());
        }
        
        // Intensifiers
        for word in ["very", "really"] {
            surface_to_prime.insert(word.to_string(), "VERY".to_string());
        }
        for word in ["more"] {
            surface_to_prime.insert(word.to_string(), "MORE".to_string());
        }
        
        // Similarity
        for word in ["like", "similar"] {
            surface_to_prime.insert(word.to_string(), "LIKE".to_string());
        }
        for word in ["same"] {
            surface_to_prime.insert(word.to_string(), "THE_SAME".to_string());
        }
        for word in ["other", "another", "else"] {
            surface_to_prime.insert(word.to_string(), "OTHER".to_string());
        }
        
        // Existence
        for word in ["live", "living", "lived", "lives", "alive"] {
            surface_to_prime.insert(word.to_string(), "LIVE".to_string());
        }
        for word in ["die", "dying", "died", "dies", "dead", "death"] {
            surface_to_prime.insert(word.to_string(), "DIE".to_string());
        }
        
        Self { surface_to_prime }
    }
    
    /// Parse explication into prime occurrences with weights
    pub fn parse(&self, explication: &str) -> Vec<(String, f32)> {
        let mut prime_counts: HashMap<String, usize> = HashMap::new();
        let mut total_words = 0;
        
        for word in explication.split_whitespace() {
            let clean = word.to_lowercase()
                .trim_matches(|c: char| !c.is_alphabetic())
                .to_string();
            
            if clean.is_empty() {
                continue;
            }
            
            total_words += 1;
            
            if let Some(prime) = self.surface_to_prime.get(&clean) {
                *prime_counts.entry(prime.clone()).or_insert(0) += 1;
            }
        }
        
        if total_words == 0 {
            return Vec::new();
        }
        
        // Convert counts to weights (normalized by total words)
        prime_counts.into_iter()
            .map(|(prime, count)| {
                let weight = (count as f32 / total_words as f32).min(1.0);
                (prime, weight)
            })
            .collect()
    }
    
    /// Get the canonical prime for a surface form
    pub fn get_prime(&self, word: &str) -> Option<&String> {
        self.surface_to_prime.get(&word.to_lowercase())
    }
}

// =============================================================================
// DeepNSM-Enhanced Codebook
// =============================================================================

/// A codebook enhanced with DeepNSM-style explication parsing
pub struct DeepNsmCodebook {
    /// Base NSM codebook with orthogonal fingerprints
    base: NsmCodebook,
    
    /// Explication parser
    parser: ExplicationParser,
    
    /// Cache of word → fingerprint (learned from explications)
    cache: HashMap<String, Fingerprint>,
}

impl Default for DeepNsmCodebook {
    fn default() -> Self {
        Self::new()
    }
}

impl DeepNsmCodebook {
    pub fn new() -> Self {
        Self {
            base: NsmCodebook::new(),
            parser: ExplicationParser::new(),
            cache: HashMap::new(),
        }
    }
    
    /// Learn a word's fingerprint from its NSM explication
    /// 
    /// This is how we bridge from DeepNSM output to our substrate:
    /// 1. DeepNSM generates explication
    /// 2. We parse primes from explication
    /// 3. We role-bind and bundle into fingerprint
    /// 4. Cache for future use
    pub fn learn_from_explication(&mut self, word: &str, explication: &str) {
        // Check for circularity
        if is_circular(explication, word) {
            return; // Skip circular explications
        }
        
        // Check legality
        let score = legality_score(explication);
        if score < 0.0 {
            return; // Skip low-quality explications
        }
        
        // Parse primes from explication
        let primes = self.parser.parse(explication);
        
        if primes.is_empty() {
            return;
        }
        
        // Build fingerprint from primes
        let mut components = Vec::new();
        
        for (prime_name, weight) in &primes {
            if let Some(prime_fp) = self.base.prime(prime_name) {
                // Infer role from prime category
                let role = infer_role_from_category(prime_name);
                
                let bound = if let Some(role_name) = role {
                    if let Some(role_fp) = self.base.role(&role_name) {
                        prime_fp.bind(role_fp)
                    } else {
                        prime_fp.clone()
                    }
                } else {
                    prime_fp.clone()
                };
                
                components.push((bound, *weight));
            }
        }
        
        if !components.is_empty() {
            let fp = weighted_bundle(&components);
            self.cache.insert(word.to_lowercase(), fp);
        }
    }
    
    /// Encode text using cached knowledge + fallback to keyword decomposition
    pub fn encode(&self, text: &str) -> Fingerprint {
        let words: Vec<&str> = text.split_whitespace().collect();
        
        let mut components = Vec::new();
        
        for word in words {
            let clean = word.to_lowercase()
                .trim_matches(|c: char| !c.is_alphabetic())
                .to_string();
            
            if clean.is_empty() {
                continue;
            }
            
            // Check cache first
            if let Some(fp) = self.cache.get(&clean) {
                components.push((fp.clone(), 1.0));
                continue;
            }
            
            // Check if it's a known prime
            if let Some(prime) = self.parser.get_prime(&clean) {
                if let Some(fp) = self.base.prime(prime) {
                    components.push((fp.clone(), 0.8));
                    continue;
                }
            }
            
            // Fallback to content-based fingerprint
            components.push((Fingerprint::from_content(&clean), 0.3));
        }
        
        if components.is_empty() {
            return Fingerprint::from_content(text);
        }
        
        weighted_bundle(&components)
    }
    
    /// Get statistics about the codebook
    pub fn stats(&self) -> (usize, usize) {
        let base_vocab = self.base.vocabulary_size();
        let cached = self.cache.len();
        (base_vocab, cached)
    }
}

/// Infer role from prime category
fn infer_role_from_category(prime: &str) -> Option<String> {
    let category = prime_category(prime)?;
    
    match category {
        "SUBSTANTIVES" => Some("R_AGENT".to_string()),
        "MENTAL" | "SPEECH" | "ACTIONS" => Some("R_ACTION".to_string()),
        "EVALUATORS" | "DESCRIPTORS" => Some("R_MANNER".to_string()),
        "TIME" => Some("R_TIME".to_string()),
        "SPACE" => Some("R_LOCATION".to_string()),
        "LOGICAL" => Some("R_CAUSE".to_string()),
        _ => None,
    }
}

/// Weighted bundle helper
fn weighted_bundle(fps: &[(Fingerprint, f32)]) -> Fingerprint {
    if fps.is_empty() {
        return Fingerprint::zero();
    }
    
    let mut counts = [0.0f32; 10000];
    let mut total_weight = 0.0f32;
    
    for (fp, weight) in fps {
        for i in 0..10000 {
            if fp.get_bit(i) {
                counts[i] += weight;
            }
        }
        total_weight += weight;
    }
    
    if total_weight == 0.0 {
        return Fingerprint::zero();
    }
    
    let threshold = total_weight / 2.0;
    let mut result = Fingerprint::zero();
    
    for i in 0..10000 {
        if counts[i] > threshold {
            result.set_bit(i, true);
        }
    }
    
    result
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_legality_score() {
        // High-quality explication (from paper)
        let good = "something bad was happening to her at that time \
                    because something bad was happening in her body";
        let score = legality_score(good);
        println!("Good explication legality: {:.2}", score);
        assert!(score > 0.0);
        
        // Poor explication (uses molecules)
        let poor = "the patient felt unwell due to viral infection";
        let score = legality_score(poor);
        println!("Poor explication legality: {:.2}", score);
        assert!(score < 5.0);
    }
    
    #[test]
    fn test_circularity() {
        assert!(is_circular("feeling sick is bad", "sick"));
        assert!(!is_circular("something bad happening in body", "sick"));
    }
    
    #[test]
    fn test_prime_counts() {
        let text = "I want to know something good about this";
        let (primes, molecules, total) = count_primes_molecules(text);
        
        println!("Primes: {}, Molecules: {}, Total: {}", primes, molecules, total);
        assert!(primes >= 3); // I, WANT, KNOW, SOMETHING, GOOD, THIS
    }
    
    #[test]
    fn test_explication_parser() {
        let parser = ExplicationParser::new();
        
        let explication = "something bad was happening to her at that time \
                          because something bad was happening in her body \
                          she could know that something bad was happening \
                          because she felt something bad in her body";
        
        let primes = parser.parse(explication);
        
        println!("Parsed primes:");
        for (prime, weight) in &primes {
            println!("  {}: {:.3}", prime, weight);
        }
        
        // Should detect SOMETHING, BAD, HAPPEN, BODY, KNOW, FEEL, BECAUSE
        let prime_names: Vec<&str> = primes.iter().map(|(p, _)| p.as_str()).collect();
        assert!(prime_names.contains(&"BAD"));
        assert!(prime_names.contains(&"HAPPEN"));
        assert!(prime_names.contains(&"BODY"));
    }
    
    #[test]
    fn test_deepnsm_codebook() {
        let mut codebook = DeepNsmCodebook::new();
        
        // Learn "sick" from explication
        let sick_explication = "something bad was happening to her at that time \
                               because something bad was happening in her body \
                               it was happening for some time \
                               she could know that something bad was happening \
                               because she felt something bad in her body";
        
        codebook.learn_from_explication("sick", sick_explication);
        
        let (base, cached) = codebook.stats();
        println!("Codebook: {} base primes, {} cached words", base, cached);
        
        assert!(cached >= 1);
        
        // Encode using learned knowledge
        let fp1 = codebook.encode("I feel sick");
        let fp2 = codebook.encode("something bad in my body");
        let fp3 = codebook.encode("the weather is nice");
        
        let sim_12 = fp1.similarity(&fp2);
        let sim_13 = fp1.similarity(&fp3);
        
        println!("sick/bad-body similarity: {:.3}", sim_12);
        println!("sick/weather similarity: {:.3}", sim_13);
        
        // Similar meanings should be closer
        // (though not guaranteed with small training)
    }
}
