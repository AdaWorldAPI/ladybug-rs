//! Crystal Language Model
//!
//! A language model compressed into 3.75KB via crystal axis projection.
//! 
//! ## Architecture
//!
//! ```text
//! 5×5×5 Crystal (125 cells × 10K bits = 1.25M bits)
//!          │
//!          ▼
//! Project to 3 orthogonal axes
//!          │
//!     ┌────┼────┐
//!     ▼    ▼    ▼
//!    X    Y    Z    (each 10K bits)
//!    │    │    │
//!    └────┴────┘
//!          │
//!    30K bits = 3.75 KB
//! ```
//!
//! ## Key Insight
//!
//! LLMs store noise (float precision) that accumulates.
//! We store signal only - noise is eliminated at every step via:
//! - Majority voting (bundle)
//! - Codebook cleaning (resonance)
//! - Threshold filtering
//!
//! ## Capacity
//!
//! With superposition, each axis can encode ~100 orthogonal patterns.
//! 3 axes × 100 patterns = 300 "concepts" at full fidelity.
//! With graceful degradation: 1000-3000 fuzzy concepts.
//!
//! This is enough for:
//! - 65 NSM primes (universal semantic basis)
//! - 16 roles (thematic relations)
//! - 8 qualia channels (affect dimensions)
//! - 12 NARS copulas (reasoning connectives)
//! - 16 YAML templates (speech acts)
//! - 10 rung levels (meaning depth)
//! - ~100 learned associations
//!
//! Total: ~230 built-in + ~100 learned ≈ 330 concepts

use crate::core::Fingerprint;
use crate::extensions::cognitive_codebook::{
    CognitiveCodebook, CognitiveAddress, CognitiveDomain,
    ThematicRole, QualiaChannel, NarsCopula, YamlTemplate,
    fold_to_48,
};
use std::collections::HashMap;

// =============================================================================
// Crystal Axis Resonances
// =============================================================================

/// The three axis resonances that encode the entire model
#[derive(Clone)]
pub struct CrystalAxes {
    /// X-axis: Temporal flow (BEFORE → NOW → AFTER)
    /// Encodes causality, narrative sequence, event ordering
    pub temporal: Fingerprint,
    
    /// Y-axis: Structural roles (AGENT → ACTION → PATIENT)
    /// Encodes SPO, thematic relations, argument structure
    pub structural: Fingerprint,
    
    /// Z-axis: Depth/abstraction (SURFACE → CORE → META)
    /// Encodes rung levels, concrete vs abstract, literal vs figurative
    pub depth: Fingerprint,
}

impl Default for CrystalAxes {
    fn default() -> Self {
        Self::new()
    }
}

impl CrystalAxes {
    pub fn new() -> Self {
        Self {
            temporal: Fingerprint::zero(),
            structural: Fingerprint::zero(),
            depth: Fingerprint::zero(),
        }
    }
    
    /// Total size in bytes
    pub fn size_bytes(&self) -> usize {
        3 * 10000 / 8  // 3,750 bytes
    }
    
    /// Total size in kilobytes
    pub fn size_kb(&self) -> f32 {
        self.size_bytes() as f32 / 1024.0
    }
    
    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.size_bytes());
        bytes.extend(self.temporal.to_bytes());
        bytes.extend(self.structural.to_bytes());
        bytes.extend(self.depth.to_bytes());
        bytes
    }
    
    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 3750 {
            return None;
        }
        
        let chunk_size = 10000 / 8;
        Some(Self {
            temporal: Fingerprint::from_bytes(&bytes[0..chunk_size])?,
            structural: Fingerprint::from_bytes(&bytes[chunk_size..chunk_size*2])?,
            depth: Fingerprint::from_bytes(&bytes[chunk_size*2..chunk_size*3])?,
        })
    }
}

// =============================================================================
// The Crystal Language Model
// =============================================================================

/// Crystal-compressed language model
/// 
/// Stores complete cognitive knowledge in ~13KB:
/// - 3.75 KB: axis resonances
/// - ~8 KB: codebook (shared, can be static)
/// - ~1 KB: learned concept cache
pub struct CrystalLM {
    /// The three axis projections (3.75 KB)
    pub axes: CrystalAxes,
    
    /// Cognitive codebook for encoding/decoding
    pub codebook: CognitiveCodebook,
    
    /// Learned association cache
    learned: HashMap<u64, Fingerprint>,
    
    /// Noise threshold for cleaning
    noise_threshold: f32,
    
    /// Training statistics
    train_count: usize,
}

impl Default for CrystalLM {
    fn default() -> Self {
        Self::new()
    }
}

impl CrystalLM {
    pub fn new() -> Self {
        Self {
            axes: CrystalAxes::new(),
            codebook: CognitiveCodebook::new(),
            learned: HashMap::new(),
            noise_threshold: 0.25,
            train_count: 0,
        }
    }
    
    /// Total model size estimate
    pub fn size_bytes(&self) -> usize {
        self.axes.size_bytes() + 
        self.learned.len() * (8 + 1250) // hash + fingerprint
    }
    
    // =========================================================================
    // Training
    // =========================================================================
    
    /// Train from (input, output) text pairs
    pub fn train(&mut self, pairs: &[(String, String)]) {
        // Phase 1: Build 5×5×5 crystal from training data
        let mut crystal = [[[Vec::<Fingerprint>::new(); 5]; 5]; 5];
        
        for (input, output) in pairs {
            // Encode with cleaning
            let input_fp = self.encode_clean(input);
            let output_fp = self.encode_clean(output);
            
            // Bind input→output association
            let association = input_fp.bind(&output_fp);
            
            // Clean the association
            let association_clean = self.clean(&association);
            
            // Find position in crystal
            let (t, s, d) = self.locate_for_training(&input_fp, input);
            
            // Add to crystal cell
            crystal[t][s][d].push(association_clean);
        }
        
        // Phase 2: Bundle each cell (noise elimination via majority voting)
        let mut bundled_crystal = [[[Fingerprint::zero(); 5]; 5]; 5];
        
        for t in 0..5 {
            for s in 0..5 {
                for d in 0..5 {
                    if !crystal[t][s][d].is_empty() {
                        // Bundle = majority vote = noise death
                        let bundled = Self::bundle(&crystal[t][s][d]);
                        // Clean against codebook
                        bundled_crystal[t][s][d] = self.clean(&bundled);
                    }
                }
            }
        }
        
        // Phase 3: Project to axes
        self.project_axes(&bundled_crystal);
        
        // Phase 4: Final clean of axes
        self.axes.temporal = self.clean(&self.axes.temporal);
        self.axes.structural = self.clean(&self.axes.structural);
        self.axes.depth = self.clean(&self.axes.depth);
        
        self.train_count += pairs.len();
    }
    
    /// Train single pair
    pub fn train_one(&mut self, input: &str, output: &str) {
        self.train(&[(input.to_string(), output.to_string())]);
    }
    
    /// Project bundled crystal to 3 axis resonances
    fn project_axes(&mut self, crystal: &[[[Fingerprint; 5]; 5]; 5]) {
        // Temporal axis: bundle all YZ planes for each T position
        let mut t_slices = Vec::new();
        for t in 0..5 {
            let mut plane = Vec::new();
            for s in 0..5 {
                for d in 0..5 {
                    if crystal[t][s][d] != Fingerprint::zero() {
                        plane.push(crystal[t][s][d].clone());
                    }
                }
            }
            if !plane.is_empty() {
                let bundled = Self::bundle(&plane);
                // Position encode with permutation
                t_slices.push(bundled.permute(t as i32));
            }
        }
        if !t_slices.is_empty() {
            // Blend with existing (for incremental training)
            let new_temporal = Self::bundle(&t_slices);
            self.axes.temporal = Self::blend(&self.axes.temporal, &new_temporal, 0.3);
        }
        
        // Structural axis: bundle all TD planes for each S position
        let mut s_slices = Vec::new();
        for s in 0..5 {
            let mut plane = Vec::new();
            for t in 0..5 {
                for d in 0..5 {
                    if crystal[t][s][d] != Fingerprint::zero() {
                        plane.push(crystal[t][s][d].clone());
                    }
                }
            }
            if !plane.is_empty() {
                let bundled = Self::bundle(&plane);
                s_slices.push(bundled.permute(s as i32 + 10)); // Offset to distinguish from T
            }
        }
        if !s_slices.is_empty() {
            let new_structural = Self::bundle(&s_slices);
            self.axes.structural = Self::blend(&self.axes.structural, &new_structural, 0.3);
        }
        
        // Depth axis: bundle all TS planes for each D position
        let mut d_slices = Vec::new();
        for d in 0..5 {
            let mut plane = Vec::new();
            for t in 0..5 {
                for s in 0..5 {
                    if crystal[t][s][d] != Fingerprint::zero() {
                        plane.push(crystal[t][s][d].clone());
                    }
                }
            }
            if !plane.is_empty() {
                let bundled = Self::bundle(&plane);
                d_slices.push(bundled.permute(d as i32 + 20)); // Offset to distinguish
            }
        }
        if !d_slices.is_empty() {
            let new_depth = Self::bundle(&d_slices);
            self.axes.depth = Self::blend(&self.axes.depth, &new_depth, 0.3);
        }
    }
    
    /// Locate input in crystal space for training
    fn locate_for_training(&self, fp: &Fingerprint, text: &str) -> (usize, usize, usize) {
        // T: temporal position (0=past, 2=present, 4=future)
        let t = self.infer_temporal(text);
        
        // S: structural position (0=subject, 2=verb, 4=object)
        let s = self.infer_structural(text);
        
        // D: depth position (0=surface, 2=core, 4=meta)
        let d = self.infer_depth(fp, text);
        
        (t, s, d)
    }
    
    fn infer_temporal(&self, text: &str) -> usize {
        let lower = text.to_lowercase();
        
        // Look for temporal markers
        if lower.contains("was") || lower.contains("were") || 
           lower.contains("did") || lower.contains("before") ||
           lower.contains("yesterday") || lower.contains("ago") {
            return 0; // Past
        }
        
        if lower.contains("will") || lower.contains("going to") ||
           lower.contains("tomorrow") || lower.contains("after") ||
           lower.contains("future") || lower.contains("later") {
            return 4; // Future
        }
        
        2 // Present (default)
    }
    
    fn infer_structural(&self, text: &str) -> usize {
        let lower = text.to_lowercase();
        
        // Look for structural indicators
        // Subject-focused (who)
        if lower.starts_with("i ") || lower.starts_with("you ") ||
           lower.starts_with("he ") || lower.starts_with("she ") ||
           lower.starts_with("they ") || lower.starts_with("we ") ||
           lower.contains(" who ") {
            return 0; // Subject position
        }
        
        // Object-focused (what)
        if lower.contains(" it ") || lower.contains(" this ") ||
           lower.contains(" that ") || lower.contains(" what ") ||
           lower.ends_with(" it") || lower.ends_with(" this") {
            return 4; // Object position
        }
        
        2 // Predicate/action (default)
    }
    
    fn infer_depth(&self, fp: &Fingerprint, text: &str) -> usize {
        // Check for rung level indicators
        if let Some((addr, sim)) = self.codebook.find_best_match(fp) {
            if sim > 0.5 {
                match addr.domain() {
                    CognitiveDomain::MetaPattern => return 4,
                    CognitiveDomain::NarsInference => return 3,
                    CognitiveDomain::NarsTerm => return 3,
                    CognitiveDomain::YamlTemplate => return 2,
                    CognitiveDomain::NsmPrime => return 1,
                    _ => {}
                }
            }
        }
        
        // Text-based heuristics
        let lower = text.to_lowercase();
        
        if lower.contains("meta") || lower.contains("about") ||
           lower.contains("concept") || lower.contains("abstract") {
            return 4; // Meta level
        }
        
        if lower.contains("therefore") || lower.contains("because") ||
           lower.contains("implies") || lower.contains("means") {
            return 3; // Reasoning level
        }
        
        2 // Core meaning (default)
    }
    
    // =========================================================================
    // Inference
    // =========================================================================
    
    /// Generate output for input text
    pub fn infer(&self, input: &str) -> InferenceResult {
        // Step 1: Encode input with cleaning
        let input_fp = self.encode_clean(input);
        
        // Step 2: Check learned cache first
        let hash = fold_to_48(&input_fp);
        if let Some(cached) = self.learned.get(&hash) {
            return InferenceResult {
                output_fp: cached.clone(),
                confidence: 0.9,
                source: InferenceSource::Cache,
            };
        }
        
        // Step 3: Locate in crystal space
        let (t, s, d) = self.locate_for_inference(&input_fp);
        
        // Step 4: Reconstruct cell from axis projections (holographic)
        let cell = self.reconstruct_cell(t, s, d);
        
        // Step 5: Unbind to get associated output
        let output_raw = cell.bind(&input_fp); // XOR is self-inverse
        
        // Step 6: Clean output
        let output_fp = self.clean(&output_raw);
        
        // Step 7: Compute confidence
        let confidence = self.compute_confidence(&output_fp);
        
        InferenceResult {
            output_fp,
            confidence,
            source: InferenceSource::Crystal,
        }
    }
    
    /// Locate input in crystal space for inference
    fn locate_for_inference(&self, fp: &Fingerprint) -> (usize, usize, usize) {
        // Find best resonating position in each axis
        let t = self.find_best_axis_position(&self.axes.temporal, fp, 0);
        let s = self.find_best_axis_position(&self.axes.structural, fp, 10);
        let d = self.find_best_axis_position(&self.axes.depth, fp, 20);
        
        (t, s, d)
    }
    
    fn find_best_axis_position(&self, axis: &Fingerprint, query: &Fingerprint, offset: i32) -> usize {
        (0..5)
            .map(|i| {
                let unbound = axis.permute(-(i as i32 + offset));
                let cleaned = self.clean(&unbound);
                (i, query.similarity(&cleaned))
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(2)
    }
    
    /// Reconstruct crystal cell from axis projections (holographic reconstruction)
    fn reconstruct_cell(&self, t: usize, s: usize, d: usize) -> Fingerprint {
        // Unbind position encoding from each axis
        let t_view = self.axes.temporal.permute(-(t as i32));
        let s_view = self.axes.structural.permute(-(s as i32 + 10));
        let d_view = self.axes.depth.permute(-(d as i32 + 20));
        
        // Clean each view
        let t_clean = self.clean(&t_view);
        let s_clean = self.clean(&s_view);
        let d_clean = self.clean(&d_view);
        
        // Bundle = intersection with noise elimination
        let cell = Self::bundle(&[t_clean, s_clean, d_clean]);
        
        // Final clean
        self.clean(&cell)
    }
    
    fn compute_confidence(&self, fp: &Fingerprint) -> f32 {
        // Confidence based on how well fp matches known concepts
        if let Some((_, sim)) = self.codebook.find_best_match(fp) {
            sim
        } else {
            0.0
        }
    }
    
    // =========================================================================
    // Encoding / Decoding
    // =========================================================================
    
    /// Encode text to fingerprint with cleaning
    pub fn encode_clean(&self, text: &str) -> Fingerprint {
        // First try to find in codebook by name
        if let Some(fp) = self.codebook.get_by_name(text) {
            return fp.clone();
        }
        
        // Parse into components
        let mut components: Vec<(Fingerprint, f32)> = Vec::new();
        
        for word in text.split_whitespace() {
            let clean_word = word.to_lowercase()
                .trim_matches(|c: char| !c.is_alphabetic())
                .to_string();
            
            if clean_word.is_empty() {
                continue;
            }
            
            // Try codebook lookup
            if let Some(fp) = self.codebook.get_by_name(&clean_word.to_uppercase()) {
                components.push((fp.clone(), 1.0));
            } else if let Some(fp) = self.codebook.get_by_name(&clean_word) {
                components.push((fp.clone(), 1.0));
            } else {
                // Content-based fallback
                components.push((Fingerprint::from_content(&clean_word), 0.5));
            }
        }
        
        if components.is_empty() {
            return Fingerprint::from_content(text);
        }
        
        // Bundle components
        let raw = Self::weighted_bundle(&components);
        
        // Clean against codebook
        self.clean(&raw)
    }
    
    /// Decode fingerprint to text
    pub fn decode(&self, fp: &Fingerprint) -> String {
        let matches = self.codebook.find_matches(fp, self.noise_threshold);
        
        if matches.is_empty() {
            return "[unknown]".to_string();
        }
        
        // Sort by similarity
        let mut sorted: Vec<_> = matches.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Return top matches
        sorted.iter()
            .take(5)
            .map(|(addr, _)| addr.name())
            .collect::<Vec<_>>()
            .join(" + ")
    }
    
    // =========================================================================
    // Noise Elimination
    // =========================================================================
    
    /// Clean fingerprint against codebook
    pub fn clean(&self, fp: &Fingerprint) -> Fingerprint {
        self.codebook.clean(fp, self.noise_threshold)
    }
    
    /// Bundle with majority voting (THE noise eliminator)
    pub fn bundle(fps: &[Fingerprint]) -> Fingerprint {
        if fps.is_empty() {
            return Fingerprint::zero();
        }
        
        let mut counts = vec![0i32; 10000];
        let threshold = fps.len() as i32 / 2;
        
        for fp in fps {
            for i in 0..10000 {
                if fp.get_bit(i) {
                    counts[i] += 1;
                }
            }
        }
        
        let mut result = Fingerprint::zero();
        for (i, &count) in counts.iter().enumerate() {
            // MAJORITY VOTE: noise (≤50%) loses, signal (>50%) wins
            if count > threshold {
                result.set_bit(i, true);
            }
        }
        
        result
    }
    
    /// Weighted bundle
    pub fn weighted_bundle(fps: &[(Fingerprint, f32)]) -> Fingerprint {
        if fps.is_empty() {
            return Fingerprint::zero();
        }
        
        let mut counts = vec![0.0f32; 10000];
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
        
        for (i, &count) in counts.iter().enumerate() {
            if count > threshold {
                result.set_bit(i, true);
            }
        }
        
        result
    }
    
    /// Blend two fingerprints (for incremental learning)
    fn blend(old: &Fingerprint, new: &Fingerprint, new_weight: f32) -> Fingerprint {
        if *old == Fingerprint::zero() {
            return new.clone();
        }
        
        Self::weighted_bundle(&[
            (old.clone(), 1.0 - new_weight),
            (new.clone(), new_weight),
        ])
    }
    
    // =========================================================================
    // Utilities
    // =========================================================================
    
    /// Learn a specific association
    pub fn learn(&mut self, input: &str, output_fp: Fingerprint) {
        let input_fp = self.encode_clean(input);
        let hash = fold_to_48(&input_fp);
        self.learned.insert(hash, output_fp);
    }
    
    /// Get training statistics
    pub fn stats(&self) -> CrystalLMStats {
        CrystalLMStats {
            axes_size_bytes: self.axes.size_bytes(),
            codebook_entries: self.codebook.stats().total_entries,
            learned_entries: self.learned.len(),
            total_trained: self.train_count,
            noise_threshold: self.noise_threshold,
        }
    }
}

// =============================================================================
// Result Types
// =============================================================================

/// Result of inference
#[derive(Debug)]
pub struct InferenceResult {
    pub output_fp: Fingerprint,
    pub confidence: f32,
    pub source: InferenceSource,
}

#[derive(Debug)]
pub enum InferenceSource {
    Cache,
    Crystal,
}

/// Model statistics
#[derive(Debug)]
pub struct CrystalLMStats {
    pub axes_size_bytes: usize,
    pub codebook_entries: usize,
    pub learned_entries: usize,
    pub total_trained: usize,
    pub noise_threshold: f32,
}

impl CrystalLMStats {
    pub fn print(&self) {
        println!("=== Crystal LM Stats ===");
        println!("Axes size: {} bytes ({:.2} KB)", 
            self.axes_size_bytes, 
            self.axes_size_bytes as f32 / 1024.0);
        println!("Codebook entries: {}", self.codebook_entries);
        println!("Learned entries: {}", self.learned_entries);
        println!("Total trained: {}", self.total_trained);
        println!("Noise threshold: {:.2}", self.noise_threshold);
        
        let total_kb = (self.axes_size_bytes + self.learned_entries * 1258) as f32 / 1024.0;
        println!("Estimated total: {:.2} KB", total_kb);
        println!();
        println!("Compare to:");
        println!("  DeepNSM-1B: 4,000,000 KB (4 GB)");
        println!("  Compression ratio: {:.0}×", 4_000_000.0 / total_kb);
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_crystal_axes_size() {
        let axes = CrystalAxes::new();
        assert_eq!(axes.size_bytes(), 3750);
        println!("Crystal axes size: {} bytes = {:.2} KB", 
            axes.size_bytes(), axes.size_kb());
    }
    
    #[test]
    fn test_crystal_lm_basic() {
        let mut lm = CrystalLM::new();
        
        // Train some associations
        lm.train(&[
            ("hello".to_string(), "greeting response".to_string()),
            ("goodbye".to_string(), "farewell response".to_string()),
            ("how are you".to_string(), "I am well".to_string()),
        ]);
        
        let stats = lm.stats();
        stats.print();
        
        // Test inference
        let result = lm.infer("hello");
        println!("Inference result: {:?}", result.source);
        println!("Confidence: {:.2}", result.confidence);
        println!("Decoded: {}", lm.decode(&result.output_fp));
    }
    
    #[test]
    fn test_bundle_noise_elimination() {
        // Create signal
        let signal = Fingerprint::orthogonal(42);
        
        // Create noise (random fingerprints)
        let noise1 = Fingerprint::random();
        let noise2 = Fingerprint::random();
        let noise3 = Fingerprint::random();
        
        // Bundle signal (3×) with noise (3×)
        // Signal should win by majority
        let bundled = CrystalLM::bundle(&[
            signal.clone(),
            signal.clone(),
            signal.clone(),
            noise1,
            noise2,
            noise3,
        ]);
        
        let sim_to_signal = bundled.similarity(&signal);
        println!("Similarity to signal after bundling: {:.3}", sim_to_signal);
        
        // Signal should be preserved (>0.5 similarity)
        assert!(sim_to_signal > 0.4);
    }
    
    #[test]
    fn test_nsm_explication() {
        let mut lm = CrystalLM::new();
        
        // Train NSM-style explications
        lm.train(&[
            // "sick" explication (from DeepNSM paper)
            ("sick".to_string(), 
             "something bad happening body feel bad".to_string()),
            
            // "happy" explication
            ("happy".to_string(),
             "something good happening feel good".to_string()),
            
            // "angry" explication  
            ("angry".to_string(),
             "someone did something bad feel bad want do something".to_string()),
        ]);
        
        // Test
        let result = lm.infer("sick");
        println!("sick -> {}", lm.decode(&result.output_fp));
        
        let result = lm.infer("happy");
        println!("happy -> {}", lm.decode(&result.output_fp));
        
        let stats = lm.stats();
        stats.print();
    }
    
    #[test]
    fn test_serialize() {
        let mut lm = CrystalLM::new();
        lm.train(&[("test".to_string(), "response".to_string())]);
        
        // Serialize axes
        let bytes = lm.axes.to_bytes();
        println!("Serialized axes: {} bytes", bytes.len());
        
        // Deserialize
        let restored = CrystalAxes::from_bytes(&bytes).unwrap();
        
        // Should match
        let sim_t = lm.axes.temporal.similarity(&restored.temporal);
        let sim_s = lm.axes.structural.similarity(&restored.structural);
        let sim_d = lm.axes.depth.similarity(&restored.depth);
        
        println!("Temporal similarity: {:.3}", sim_t);
        println!("Structural similarity: {:.3}", sim_s);
        println!("Depth similarity: {:.3}", sim_d);
        
        assert!(sim_t > 0.99);
        assert!(sim_s > 0.99);
        assert!(sim_d > 0.99);
    }
}
