//! Codebook Training Pipeline
//!
//! Bootstraps the NSM Metacognitive Substrate from Jina embeddings
//! to a self-sufficient, pure-SIMD semantic encoder.
//!
//! ## The Strategy
//!
//! Phase 1: JINA PARALLEL (current)
//!   - Run Jina AND NSM decomposition in parallel
//!   - Use Jina as ground truth
//!   - Train NSM weights to match Jina similarity rankings
//!
//! Phase 2: DISTILLATION
//!   - Use fine-tuned LLM to generate NSM explications
//!   - Build concept clusters from explications
//!   - Mint new codebook entries for clusters
//!
//! Phase 3: PURE SUBSTRATE
//!   - NSM substrate runs independently
//!   - Jina used only for validation
//!   - Eventually Jina dependency removed entirely
//!
//! ## Training Loop
//!
//! ```text
//!                    ┌─────────────────────────────────────────┐
//!                    │         TRAINING CORPUS                 │
//!                    │  (text pairs with semantic similarity)  │
//!                    └────────────────────┬────────────────────┘
//!                                         │
//!                    ┌────────────────────▼────────────────────┐
//!                    │         PARALLEL ENCODING               │
//!                    │                                         │
//!                    │   text ──┬── Jina ────► 1024D dense     │
//!                    │          │                              │
//!                    │          └── NSM  ────► 10K sparse      │
//!                    └────────────────────┬────────────────────┘
//!                                         │
//!                    ┌────────────────────▼────────────────────┐
//!                    │         SIMILARITY LOSS                 │
//!                    │                                         │
//!                    │   L = Σ (sim_jina - sim_nsm)²           │
//!                    │                                         │
//!                    │   Goal: NSM similarity rankings         │
//!                    │         match Jina rankings             │
//!                    └────────────────────┬────────────────────┘
//!                                         │
//!                    ┌────────────────────▼────────────────────┐
//!                    │         GRADIENT UPDATE                 │
//!                    │                                         │
//!                    │   Adjust:                               │
//!                    │   - Keyword → Prime weights             │
//!                    │   - Prime fingerprint bits (permute)    │
//!                    │   - Role binding strengths              │
//!                    └─────────────────────────────────────────┘
//! ```

use crate::core::Fingerprint;
use crate::nars::TruthValue;
use super::nsm_substrate::{NsmCodebook, MetacognitiveSubstrate, NSM_CATEGORIES, ROLES};
use std::collections::HashMap;

// =============================================================================
// Training Data Structures
// =============================================================================

/// A single training example: text pair with known similarity
#[derive(Clone, Debug)]
pub struct TrainingPair {
    pub text_a: String,
    pub text_b: String,
    pub jina_similarity: f32,  // Ground truth from Jina
}

/// Batch of training examples
pub struct TrainingBatch {
    pub pairs: Vec<TrainingPair>,
    pub source: String,  // Where this batch came from
}

/// Training statistics
#[derive(Clone, Debug, Default)]
pub struct TrainingStats {
    pub epoch: usize,
    pub total_pairs: usize,
    pub mean_loss: f32,
    pub rank_correlation: f32,  // Spearman correlation of similarity rankings
    pub top_10_accuracy: f32,   // How often top-10 by NSM matches top-10 by Jina
}

// =============================================================================
// Keyword → Prime Mapping (Trainable)
// =============================================================================

/// Trainable keyword weights
#[derive(Clone)]
pub struct KeywordWeights {
    /// keyword → (prime_name, base_weight, learned_delta)
    mappings: HashMap<String, Vec<(String, f32, f32)>>,
    
    /// Learning rate for weight updates
    learning_rate: f32,
    
    /// Momentum for SGD
    momentum: f32,
    
    /// Previous gradients (for momentum)
    prev_gradients: HashMap<String, f32>,
}

impl Default for KeywordWeights {
    fn default() -> Self {
        Self::new()
    }
}

impl KeywordWeights {
    pub fn new() -> Self {
        let mut mappings = HashMap::new();
        
        // Initialize with default keyword → prime mappings
        // These are the trainable weights
        
        // Mental predicates
        for (kw, prime, base) in [
            ("want", "WANT", 0.9),
            ("desire", "WANT", 0.85),
            ("wish", "WANT", 0.8),
            ("need", "WANT", 0.75),
            ("yearn", "WANT", 0.9),
            ("crave", "WANT", 0.85),
            
            ("know", "KNOW", 0.9),
            ("understand", "KNOW", 0.85),
            ("realize", "KNOW", 0.8),
            ("aware", "KNOW", 0.75),
            ("comprehend", "KNOW", 0.85),
            ("grasp", "KNOW", 0.7),
            
            ("think", "THINK", 0.9),
            ("believe", "THINK", 0.85),
            ("suppose", "THINK", 0.75),
            ("consider", "THINK", 0.8),
            ("assume", "THINK", 0.7),
            
            ("feel", "FEEL", 0.9),
            ("sense", "FEEL", 0.8),
            ("emotion", "FEEL", 0.85),
            ("experience", "FEEL", 0.7),
            
            ("see", "SEE", 0.9),
            ("look", "SEE", 0.8),
            ("watch", "SEE", 0.85),
            ("observe", "SEE", 0.8),
            ("view", "SEE", 0.75),
            
            ("hear", "HEAR", 0.9),
            ("listen", "HEAR", 0.85),
            ("sound", "HEAR", 0.7),
        ] {
            mappings.entry(kw.to_string())
                .or_insert_with(Vec::new)
                .push((prime.to_string(), base, 0.0));
        }
        
        // Agents
        for (kw, prime, base) in [
            ("i", "I", 0.95),
            ("me", "I", 0.9),
            ("my", "I", 0.85),
            ("myself", "I", 0.9),
            ("we", "I", 0.7),  // Partial I
            
            ("you", "YOU", 0.95),
            ("your", "YOU", 0.9),
            ("yourself", "YOU", 0.9),
            
            ("someone", "SOMEONE", 0.9),
            ("person", "SOMEONE", 0.85),
            ("one", "SOMEONE", 0.6),
            ("anyone", "SOMEONE", 0.8),
            ("somebody", "SOMEONE", 0.85),
            
            ("people", "PEOPLE", 0.9),
            ("they", "PEOPLE", 0.75),
            ("everyone", "PEOPLE", 0.85),
            ("everybody", "PEOPLE", 0.85),
            ("humans", "PEOPLE", 0.8),
        ] {
            mappings.entry(kw.to_string())
                .or_insert_with(Vec::new)
                .push((prime.to_string(), base, 0.0));
        }
        
        // Evaluators
        for (kw, prime, base) in [
            ("good", "GOOD", 0.9),
            ("great", "GOOD", 0.85),
            ("wonderful", "GOOD", 0.9),
            ("excellent", "GOOD", 0.85),
            ("beautiful", "GOOD", 0.8),
            ("nice", "GOOD", 0.7),
            ("positive", "GOOD", 0.75),
            
            ("bad", "BAD", 0.9),
            ("terrible", "BAD", 0.9),
            ("awful", "BAD", 0.85),
            ("wrong", "BAD", 0.75),
            ("negative", "BAD", 0.7),
            ("horrible", "BAD", 0.9),
        ] {
            mappings.entry(kw.to_string())
                .or_insert_with(Vec::new)
                .push((prime.to_string(), base, 0.0));
        }
        
        // Time
        for (kw, prime, base) in [
            ("now", "NOW", 0.95),
            ("currently", "NOW", 0.9),
            ("present", "NOW", 0.8),
            ("today", "NOW", 0.85),
            
            ("before", "BEFORE", 0.9),
            ("past", "BEFORE", 0.85),
            ("ago", "BEFORE", 0.8),
            ("previously", "BEFORE", 0.85),
            ("earlier", "BEFORE", 0.8),
            
            ("after", "AFTER", 0.9),
            ("future", "AFTER", 0.85),
            ("later", "AFTER", 0.8),
            ("next", "AFTER", 0.75),
            ("soon", "AFTER", 0.7),
        ] {
            mappings.entry(kw.to_string())
                .or_insert_with(Vec::new)
                .push((prime.to_string(), base, 0.0));
        }
        
        // Logic
        for (kw, prime, base) in [
            ("not", "NOT", 0.95),
            ("no", "NOT", 0.9),
            ("never", "NOT", 0.9),
            ("none", "NOT", 0.85),
            
            ("maybe", "MAYBE", 0.9),
            ("perhaps", "MAYBE", 0.85),
            ("possibly", "MAYBE", 0.85),
            ("might", "MAYBE", 0.75),
            ("could", "MAYBE", 0.7),
            
            ("because", "BECAUSE", 0.9),
            ("since", "BECAUSE", 0.75),
            ("therefore", "BECAUSE", 0.8),
            ("thus", "BECAUSE", 0.75),
            
            ("if", "IF", 0.9),
            ("whether", "IF", 0.8),
            ("suppose", "IF", 0.7),
        ] {
            mappings.entry(kw.to_string())
                .or_insert_with(Vec::new)
                .push((prime.to_string(), base, 0.0));
        }
        
        Self {
            mappings,
            learning_rate: 0.01,
            momentum: 0.9,
            prev_gradients: HashMap::new(),
        }
    }
    
    /// Get effective weight for a keyword → prime mapping
    pub fn weight(&self, keyword: &str, prime: &str) -> f32 {
        if let Some(primes) = self.mappings.get(keyword) {
            for (p, base, delta) in primes {
                if p == prime {
                    return (*base + *delta).clamp(0.0, 1.0);
                }
            }
        }
        0.0
    }
    
    /// Decompose text using trained weights
    pub fn decompose(&self, text: &str) -> Vec<(String, f32, Option<String>)> {
        let text_lower = text.to_lowercase();
        let words: Vec<&str> = text_lower.split_whitespace().collect();
        
        let mut activations: HashMap<String, (f32, Option<String>)> = HashMap::new();
        
        for word in &words {
            // Strip punctuation
            let clean: String = word.chars().filter(|c| c.is_alphabetic()).collect();
            if clean.is_empty() {
                continue;
            }
            
            if let Some(primes) = self.mappings.get(&clean) {
                for (prime, base, delta) in primes {
                    let weight = (*base + *delta).clamp(0.0, 1.0);
                    
                    // Determine role from prime category
                    let role = Self::infer_role(prime);
                    
                    let entry = activations.entry(prime.clone()).or_insert((0.0, role.clone()));
                    entry.0 = entry.0.max(weight);  // Take max if multiple hits
                }
            }
        }
        
        activations.into_iter()
            .map(|(prime, (weight, role))| (prime, weight, role))
            .collect()
    }
    
    /// Infer role from prime category
    fn infer_role(prime: &str) -> Option<String> {
        // Mental predicates → ACTION
        if ["WANT", "KNOW", "THINK", "FEEL", "SEE", "HEAR", "DO", "HAPPEN", "MOVE"].contains(&prime) {
            return Some("R_ACTION".to_string());
        }
        
        // Agents → AGENT
        if ["I", "YOU", "SOMEONE", "PEOPLE", "BODY"].contains(&prime) {
            return Some("R_AGENT".to_string());
        }
        
        // Time → TIME
        if ["NOW", "BEFORE", "AFTER", "WHEN", "A_LONG_TIME", "A_SHORT_TIME"].contains(&prime) {
            return Some("R_TIME".to_string());
        }
        
        // Logic → various
        if prime == "BECAUSE" {
            return Some("R_CAUSE".to_string());
        }
        if prime == "IF" {
            return Some("R_CONDITION".to_string());
        }
        
        // Space → LOCATION
        if ["WHERE", "HERE", "ABOVE", "BELOW", "FAR", "NEAR", "INSIDE"].contains(&prime) {
            return Some("R_LOCATION".to_string());
        }
        
        None
    }
    
    /// Update weights based on gradient
    pub fn update(&mut self, keyword: &str, prime: &str, gradient: f32) {
        if let Some(primes) = self.mappings.get_mut(keyword) {
            for (p, _base, delta) in primes.iter_mut() {
                if p == prime {
                    // SGD with momentum
                    let key = format!("{}:{}", keyword, prime);
                    let prev = self.prev_gradients.get(&key).copied().unwrap_or(0.0);
                    
                    let update = self.momentum * prev + self.learning_rate * gradient;
                    *delta += update;
                    
                    self.prev_gradients.insert(key, update);
                    return;
                }
            }
        }
    }
}

// =============================================================================
// Training Pipeline
// =============================================================================

/// The main training pipeline
pub struct CodebookTrainer {
    /// The substrate being trained
    pub substrate: MetacognitiveSubstrate,
    
    /// Trainable keyword weights
    pub weights: KeywordWeights,
    
    /// Training statistics history
    pub history: Vec<TrainingStats>,
    
    /// Jina API endpoint (for parallel comparison)
    jina_endpoint: Option<String>,
    jina_api_key: Option<String>,
}

impl Default for CodebookTrainer {
    fn default() -> Self {
        Self::new()
    }
}

impl CodebookTrainer {
    pub fn new() -> Self {
        Self {
            substrate: MetacognitiveSubstrate::new(),
            weights: KeywordWeights::new(),
            history: Vec::new(),
            jina_endpoint: None,
            jina_api_key: None,
        }
    }
    
    /// Configure Jina for parallel training
    pub fn with_jina(mut self, endpoint: &str, api_key: &str) -> Self {
        self.jina_endpoint = Some(endpoint.to_string());
        self.jina_api_key = Some(api_key.to_string());
        self
    }
    
    /// Encode text using current trained weights
    pub fn encode(&self, text: &str) -> Fingerprint {
        let decomposition = self.weights.decompose(text);
        
        if decomposition.is_empty() {
            return Fingerprint::from_content(text);
        }
        
        let mut components = Vec::new();
        let codebook = &self.substrate.codebook;
        
        for (primitive, weight, role) in &decomposition {
            if let Some(prime_fp) = codebook.prime(primitive) {
                let bound = if let Some(role_name) = role {
                    if let Some(role_fp) = codebook.role(role_name) {
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
        
        weighted_bundle(&components)
    }
    
    /// Compute loss for a batch (without Jina - using pre-computed similarities)
    pub fn compute_loss(&self, batch: &TrainingBatch) -> (f32, Vec<(String, String, f32)>) {
        let mut total_loss = 0.0;
        let mut gradients = Vec::new();
        
        for pair in &batch.pairs {
            let fp_a = self.encode(&pair.text_a);
            let fp_b = self.encode(&pair.text_b);
            
            let nsm_sim = fp_a.similarity(&fp_b);
            let jina_sim = pair.jina_similarity;
            
            // L2 loss on similarity difference
            let diff = nsm_sim - jina_sim;
            let loss = diff * diff;
            total_loss += loss;
            
            // Gradient: d(loss)/d(sim) = 2 * diff
            let grad = 2.0 * diff;
            
            // Propagate gradient to keyword weights
            // This is approximate - we use finite differences conceptually
            let decomp_a = self.weights.decompose(&pair.text_a);
            let decomp_b = self.weights.decompose(&pair.text_b);
            
            for (prime, weight, _) in decomp_a.iter().chain(decomp_b.iter()) {
                // Gradient flows through prime weight
                gradients.push((
                    format!("{}:{}", pair.text_a, pair.text_b),
                    prime.clone(),
                    grad * *weight,
                ));
            }
        }
        
        let mean_loss = total_loss / batch.pairs.len() as f32;
        (mean_loss, gradients)
    }
    
    /// Train for one epoch
    pub fn train_epoch(&mut self, batches: &[TrainingBatch]) -> TrainingStats {
        let mut total_loss = 0.0;
        let mut total_pairs = 0;
        
        // Accumulate gradients per keyword-prime
        let mut grad_accum: HashMap<(String, String), (f32, usize)> = HashMap::new();
        
        for batch in batches {
            let (loss, gradients) = self.compute_loss(batch);
            total_loss += loss * batch.pairs.len() as f32;
            total_pairs += batch.pairs.len();
            
            for (_context, prime, grad) in gradients {
                // Find which keywords activated this prime
                for (keyword, primes) in self.weights.mappings.iter() {
                    for (p, _, _) in primes {
                        if p == &prime {
                            let key = (keyword.clone(), prime.clone());
                            let entry = grad_accum.entry(key).or_insert((0.0, 0));
                            entry.0 += grad;
                            entry.1 += 1;
                        }
                    }
                }
            }
        }
        
        // Apply accumulated gradients
        for ((keyword, prime), (grad_sum, count)) in grad_accum {
            let avg_grad = grad_sum / count as f32;
            self.weights.update(&keyword, &prime, -avg_grad);  // Negative for gradient descent
        }
        
        let mean_loss = total_loss / total_pairs as f32;
        
        let stats = TrainingStats {
            epoch: self.history.len(),
            total_pairs,
            mean_loss,
            rank_correlation: 0.0,  // TODO: compute
            top_10_accuracy: 0.0,   // TODO: compute
        };
        
        self.history.push(stats.clone());
        stats
    }
    
    /// Generate training pairs from a corpus
    /// Uses content-based fingerprints as weak supervision
    pub fn generate_pairs_from_corpus(texts: &[String], pairs_per_text: usize) -> TrainingBatch {
        let mut pairs = Vec::new();
        
        // Generate content fingerprints for all texts
        let fps: Vec<Fingerprint> = texts.iter()
            .map(|t| Fingerprint::from_content(t))
            .collect();
        
        // For each text, find top-k most similar and create pairs
        for i in 0..texts.len() {
            let mut similarities: Vec<(usize, f32)> = fps.iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(j, fp)| (j, fps[i].similarity(fp)))
                .collect();
            
            similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            for (j, sim) in similarities.into_iter().take(pairs_per_text) {
                pairs.push(TrainingPair {
                    text_a: texts[i].clone(),
                    text_b: texts[j].clone(),
                    jina_similarity: sim,  // Use content similarity as proxy
                });
            }
        }
        
        TrainingBatch {
            pairs,
            source: "corpus_self_supervision".to_string(),
        }
    }
    
    /// Learn new concepts from the current corpus
    pub fn learn_concepts(&mut self, texts: &[String], similarity_threshold: f32) {
        // Encode all texts
        let fps: Vec<Fingerprint> = texts.iter()
            .map(|t| self.encode(t))
            .collect();
        
        // Find clusters of similar fingerprints
        let mut clusters: Vec<Vec<usize>> = Vec::new();
        let mut assigned = vec![false; texts.len()];
        
        for i in 0..texts.len() {
            if assigned[i] {
                continue;
            }
            
            let mut cluster = vec![i];
            assigned[i] = true;
            
            for j in (i+1)..texts.len() {
                if !assigned[j] && fps[i].similarity(&fps[j]) > similarity_threshold {
                    cluster.push(j);
                    assigned[j] = true;
                }
            }
            
            if cluster.len() >= 3 {  // Minimum cluster size
                clusters.push(cluster);
            }
        }
        
        // Mint new concepts from clusters
        for (idx, cluster) in clusters.iter().enumerate() {
            let cluster_fps: Vec<Fingerprint> = cluster.iter()
                .map(|&i| fps[i].clone())
                .collect();
            
            let name = format!("LEARNED_{}", self.substrate.codebook.vocabulary_size() + idx);
            let confidence = (cluster.len() as f32 / texts.len() as f32).min(0.9);
            
            self.substrate.codebook.learn_concept(&name, &cluster_fps, confidence);
        }
    }
    
    /// Export trained weights for persistence
    pub fn export_weights(&self) -> HashMap<String, Vec<(String, f32)>> {
        let mut export = HashMap::new();
        
        for (keyword, primes) in &self.weights.mappings {
            let weights: Vec<(String, f32)> = primes.iter()
                .map(|(p, base, delta)| (p.clone(), (*base + *delta).clamp(0.0, 1.0)))
                .collect();
            export.insert(keyword.clone(), weights);
        }
        
        export
    }
    
    /// Import trained weights
    pub fn import_weights(&mut self, weights: HashMap<String, Vec<(String, f32)>>) {
        for (keyword, primes) in weights {
            if let Some(existing) = self.weights.mappings.get_mut(&keyword) {
                for (prime, weight) in primes {
                    for (p, base, delta) in existing.iter_mut() {
                        if *p == prime {
                            *delta = weight - *base;
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

fn weighted_bundle(fps: &[(Fingerprint, f32)]) -> Fingerprint {
    if fps.is_empty() {
        return Fingerprint::zero();
    }
    
    let mut counts = [0.0f32; 16384];
    let mut total_weight = 0.0f32;

    for (fp, weight) in fps {
        for i in 0..16384 {
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

    for i in 0..16384 {
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
    fn test_keyword_weights() {
        let weights = KeywordWeights::new();
        
        // Should have mappings for common words
        assert!(weights.weight("want", "WANT") > 0.8);
        assert!(weights.weight("know", "KNOW") > 0.8);
        assert!(weights.weight("good", "GOOD") > 0.8);
        
        // Unknown words should return 0
        assert_eq!(weights.weight("xyzzy", "WANT"), 0.0);
    }
    
    #[test]
    fn test_decomposition() {
        let weights = KeywordWeights::new();
        
        let decomp = weights.decompose("I really want to understand this");
        let primes: Vec<&str> = decomp.iter().map(|(p, _, _)| p.as_str()).collect();
        
        assert!(primes.contains(&"I"));
        assert!(primes.contains(&"WANT"));
    }
    
    #[test]
    fn test_training_pipeline() {
        let mut trainer = CodebookTrainer::new();
        
        // Create simple training batch
        let batch = TrainingBatch {
            pairs: vec![
                TrainingPair {
                    text_a: "I want to learn".to_string(),
                    text_b: "I desire knowledge".to_string(),
                    jina_similarity: 0.85,
                },
                TrainingPair {
                    text_a: "I want to learn".to_string(),
                    text_b: "The weather is nice".to_string(),
                    jina_similarity: 0.1,
                },
            ],
            source: "test".to_string(),
        };
        
        // Train one epoch
        let stats = trainer.train_epoch(&[batch]);
        
        println!("Training stats: {:?}", stats);
        assert!(stats.mean_loss >= 0.0);
    }
    
    #[test]
    fn test_concept_learning() {
        let mut trainer = CodebookTrainer::new();
        
        let texts = vec![
            "machine learning algorithm".to_string(),
            "deep learning neural network".to_string(),
            "artificial intelligence model".to_string(),
            "dog cat pet animal".to_string(),
            "weather rain sun cloud".to_string(),
        ];
        
        let initial_vocab = trainer.substrate.codebook.vocabulary_size();
        trainer.learn_concepts(&texts, 0.4);
        
        // May or may not learn concepts depending on similarity
        println!("Vocab size: {} -> {}", initial_vocab, trainer.substrate.codebook.vocabulary_size());
    }
}
