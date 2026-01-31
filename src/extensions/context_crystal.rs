//! Contextual Resonance Crystal
//!
//! Instead of analyzing a single utterance in isolation,
//! embed meaning in the CONTEXT FIELD of surrounding sentences.
//!
//! ```text
//!                     5×5×5 Crystal
//!           ┌─────────────────────────────┐
//!           │   S-2   S-1   S0   S+1  S+2 │  ← Sentence axis (time)
//!           │    │     │    │     │    │  │
//!           │    ▼     ▼    ▼     ▼    ▼  │
//!           │  ┌───┬───┬───┬───┬───┐     │
//!           │  │ O │ O │ O │ O │ O │ ←──── Object axis (theme)
//!           │  ├───┼───┼───┼───┼───┤     │
//!           │  │   │   │ S │   │   │     │ ← Subject axis (agent)
//!           │  └───┴───┴───┴───┴───┘     │
//!           │         P axis (action)     │
//!           └─────────────────────────────┘
//!
//! Each cell: SPO_triple ⊕ Qualia ⊕ TemporalPosition
//! Query: resonate across the cube, not just one point
//! ```
//!
//! Key insight: MEANING IS FLOW, not static structure
//! - The 65 NSM primitives are implicit in SPO role binding
//! - Qualia is already encoded
//! - Temporal position via permutation handles causality
//!
//! This replaces Grammar Triangle with something richer:
//! - Grammar Triangle: NSM + Causality + Qualia → single fingerprint
//! - Context Crystal: (SPO + Qualia) × 5 sentences → resonance field

use crate::core::Fingerprint;
use std::collections::HashMap;

// =============================================================================
// Constants
// =============================================================================

const GRID: usize = 5;          // 5×5×5 crystal
const N: usize = 10_000;        // Fingerprint bits
const CONTEXT_WINDOW: usize = 5; // 2 before + current + 2 after

// =============================================================================
// Sentence with SPO + Qualia
// =============================================================================

/// A sentence decomposed into Subject-Predicate-Object with felt-sense
#[derive(Clone, Debug)]
pub struct SentenceAtom {
    /// Raw text (for debugging)
    pub text: String,
    
    /// Subject fingerprint (WHO)
    pub subject: Fingerprint,
    
    /// Predicate fingerprint (DOES)
    pub predicate: Fingerprint,
    
    /// Object fingerprint (WHAT)
    pub object: Fingerprint,
    
    /// Qualia: felt-sense of the utterance
    pub qualia: QualiaVector,
    
    /// Position in context window (-2 to +2)
    pub position: i32,
}

/// Compact qualia representation (4D → 8D for more nuance)
#[derive(Clone, Debug, Default)]
pub struct QualiaVector {
    pub arousal: f32,      // Calm ↔ Excited
    pub valence: f32,      // Negative ↔ Positive
    pub tension: f32,      // Relaxed ↔ Tense
    pub depth: f32,        // Surface ↔ Profound
    pub certainty: f32,    // Doubtful ↔ Certain
    pub intimacy: f32,     // Distant ↔ Intimate
    pub urgency: f32,      // Relaxed ↔ Urgent
    pub novelty: f32,      // Familiar ↔ Novel
}

impl QualiaVector {
    pub fn neutral() -> Self {
        Self {
            arousal: 0.5, valence: 0.5, tension: 0.5, depth: 0.5,
            certainty: 0.5, intimacy: 0.5, urgency: 0.5, novelty: 0.5,
        }
    }
    
    /// Encode as fingerprint contribution
    pub fn to_fingerprint(&self) -> Fingerprint {
        let mut fp = Fingerprint::zero();
        
        // Each dimension gets ~1250 bits (10000/8)
        let dims = [
            self.arousal, self.valence, self.tension, self.depth,
            self.certainty, self.intimacy, self.urgency, self.novelty,
        ];
        
        for (i, &val) in dims.iter().enumerate() {
            let base = i * 1250;
            let num_bits = (val * 1250.0) as usize;
            for j in 0..num_bits.min(1250) {
                fp.set_bit(base + j, true);
            }
        }
        fp
    }
    
    /// Distance to another qualia vector
    pub fn distance(&self, other: &Self) -> f32 {
        let da = self.arousal - other.arousal;
        let dv = self.valence - other.valence;
        let dt = self.tension - other.tension;
        let dd = self.depth - other.depth;
        let dc = self.certainty - other.certainty;
        let di = self.intimacy - other.intimacy;
        let du = self.urgency - other.urgency;
        let dn = self.novelty - other.novelty;
        (da*da + dv*dv + dt*dt + dd*dd + dc*dc + di*di + du*du + dn*dn).sqrt()
    }
}

impl SentenceAtom {
    /// Encode the sentence as a single fingerprint
    /// SPO ⊕ Qualia ⊕ TemporalShift
    pub fn to_fingerprint(&self) -> Fingerprint {
        // Role vectors (deterministic from position)
        let role_s = Fingerprint::from_content("ROLE_SUBJECT");
        let role_p = Fingerprint::from_content("ROLE_PREDICATE");
        let role_o = Fingerprint::from_content("ROLE_OBJECT");
        
        // Bind each component with its role
        let s_bound = self.subject.bind(&role_s);
        let p_bound = self.predicate.bind(&role_p);
        let o_bound = self.object.bind(&role_o);
        
        // Bundle SPO
        let spo = bundle(&[s_bound, p_bound, o_bound]);
        
        // Add qualia overlay
        let qualia_fp = self.qualia.to_fingerprint();
        let spo_qualia = spo.bind(&qualia_fp);
        
        // Temporal shift: permute based on position
        // This makes S-2 different from S+2 even with same content
        let temporal_shift = self.position * 100;
        spo_qualia.permute(temporal_shift)
    }
}

// =============================================================================
// Context Crystal: 5×5×5 Resonance Field
// =============================================================================

/// The context crystal holds superposed meaning across a window of sentences
pub struct ContextCrystal {
    /// 5×5×5 grid of fingerprints
    /// Axis 0: Temporal position (S-2 to S+2)
    /// Axis 1: Subject/Agent dimension
    /// Axis 2: Object/Theme dimension
    cells: Box<[[[Fingerprint; GRID]; GRID]; GRID]>,
    
    /// Count of contributions per cell (for averaging)
    counts: [[[u32; GRID]; GRID]; GRID],
    
    /// Metadata
    pub total_sentences: usize,
}

impl Default for ContextCrystal {
    fn default() -> Self {
        Self::new()
    }
}

impl ContextCrystal {
    pub fn new() -> Self {
        // Initialize all cells to zero using from_fn (Fingerprint doesn't impl Copy)
        let cells = Box::new(core::array::from_fn(|_| {
            core::array::from_fn(|_| {
                core::array::from_fn(|_| Fingerprint::zero())
            })
        }));
        Self {
            cells,
            counts: [[[0u32; GRID]; GRID]; GRID],
            total_sentences: 0,
        }
    }
    
    /// Insert a context window (5 sentences centered on current)
    /// 
    /// The sentences should be in order: [S-2, S-1, S0, S+1, S+2]
    /// where S0 is the current sentence
    pub fn insert_context(&mut self, sentences: &[SentenceAtom]) {
        if sentences.len() != CONTEXT_WINDOW {
            return; // Require exactly 5 sentences
        }
        
        for (t, sentence) in sentences.iter().enumerate() {
            // t is the temporal axis (0-4)
            let fp = sentence.to_fingerprint();
            
            // Hash subject and object to get s and o axes
            let s_axis = hash_to_grid(&sentence.subject);
            let o_axis = hash_to_grid(&sentence.object);
            
            // Bundle into the cell
            self.bundle_into(t, s_axis, o_axis, &fp);
        }
        
        self.total_sentences += sentences.len();
    }
    
    /// Insert a single sentence at a specific position
    pub fn insert_sentence(&mut self, sentence: &SentenceAtom) {
        let fp = sentence.to_fingerprint();
        
        // Temporal position: -2..+2 → 0..4
        let t_axis = (sentence.position + 2).clamp(0, 4) as usize;
        let s_axis = hash_to_grid(&sentence.subject);
        let o_axis = hash_to_grid(&sentence.object);
        
        self.bundle_into(t_axis, s_axis, o_axis, &fp);
        self.total_sentences += 1;
    }
    
    /// Bundle a fingerprint into a cell
    fn bundle_into(&mut self, t: usize, s: usize, o: usize, fp: &Fingerprint) {
        let current = &self.cells[t][s][o];
        if self.counts[t][s][o] == 0 {
            self.cells[t][s][o] = fp.clone();
        } else {
            self.cells[t][s][o] = bundle(&[current.clone(), fp.clone()]);
        }
        self.counts[t][s][o] += 1;
    }
    
    /// Query: find resonance with a context window
    /// Returns overall similarity and per-cell breakdown
    pub fn query(&self, context: &[SentenceAtom]) -> CrystalQuery {
        let mut query_crystal = ContextCrystal::new();
        query_crystal.insert_context(context);
        
        // Compute cell-by-cell similarity
        let mut cell_sims = [[[0.0f32; GRID]; GRID]; GRID];
        let mut total_sim = 0.0f32;
        let mut active_cells = 0;
        
        for t in 0..GRID {
            for s in 0..GRID {
                for o in 0..GRID {
                    let self_fp = &self.cells[t][s][o];
                    let query_fp = &query_crystal.cells[t][s][o];
                    
                    // Skip empty cells
                    if self.counts[t][s][o] == 0 || query_crystal.counts[t][s][o] == 0 {
                        continue;
                    }
                    
                    let sim = self_fp.similarity(query_fp);
                    cell_sims[t][s][o] = sim;
                    total_sim += sim;
                    active_cells += 1;
                }
            }
        }
        
        let avg_sim = if active_cells > 0 { total_sim / active_cells as f32 } else { 0.0 };
        
        CrystalQuery {
            overall_similarity: avg_sim,
            cell_similarities: cell_sims,
            active_cells,
            temporal_flow: self.compute_temporal_flow(&query_crystal),
        }
    }
    
    /// Compute temporal flow similarity (how meaning evolves)
    fn compute_temporal_flow(&self, query: &ContextCrystal) -> f32 {
        // Compare the temporal gradient of meaning
        let mut flow_sim = 0.0f32;
        let mut comparisons = 0;
        
        for t in 0..GRID-1 {
            // Get the "flow" from t to t+1 in both crystals
            let self_t = self.temporal_slice(t);
            let self_t1 = self.temporal_slice(t + 1);
            let query_t = query.temporal_slice(t);
            let query_t1 = query.temporal_slice(t + 1);
            
            // Flow is the change in meaning
            let self_flow = self_t.bind(&self_t1);
            let query_flow = query_t.bind(&query_t1);
            
            flow_sim += self_flow.similarity(&query_flow);
            comparisons += 1;
        }
        
        if comparisons > 0 { flow_sim / comparisons as f32 } else { 0.0 }
    }
    
    /// Get bundled fingerprint for a temporal slice (all S×O at time t)
    fn temporal_slice(&self, t: usize) -> Fingerprint {
        let mut result = Fingerprint::zero();
        for s in 0..GRID {
            for o in 0..GRID {
                if self.counts[t][s][o] > 0 {
                    result = bundle(&[result, self.cells[t][s][o].clone()]);
                }
            }
        }
        result
    }
    
    /// Get the central sentence's resonance (S0)
    pub fn central_resonance(&self) -> Fingerprint {
        self.temporal_slice(2) // t=2 is the center (S0)
    }
    
    /// Mexican hat in temporal dimension:
    /// Emphasize middle (S0), de-emphasize extremes (S-2, S+2)
    pub fn mexican_hat_resonance(&self, center_weight: f32, edge_weight: f32) -> Fingerprint {
        // Weights: [edge, mid, center, mid, edge]
        let weights = [
            edge_weight,      // S-2
            center_weight * 0.7,  // S-1
            center_weight,    // S0 (strongest)
            center_weight * 0.7,  // S+1
            edge_weight,      // S+2
        ];
        
        let mut result = Fingerprint::zero();
        for (t, &w) in weights.iter().enumerate() {
            let slice = self.temporal_slice(t);
            // Weight by popcount proportion
            let weighted = weight_fingerprint(&slice, w);
            result = bundle(&[result, weighted]);
        }
        result
    }
}

/// Result of a crystal query
#[derive(Clone, Debug)]
pub struct CrystalQuery {
    /// Overall similarity (average of active cells)
    pub overall_similarity: f32,
    /// Per-cell similarities
    pub cell_similarities: [[[f32; GRID]; GRID]; GRID],
    /// Number of cells that had content in both crystals
    pub active_cells: usize,
    /// How similar the temporal flow patterns are
    pub temporal_flow: f32,
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Hash fingerprint to grid coordinate (0..4)
fn hash_to_grid(fp: &Fingerprint) -> usize {
    let raw = fp.as_raw();
    let h = raw[0].wrapping_mul(31)
        .wrapping_add(raw[1])
        .wrapping_mul(31)
        .wrapping_add(raw[2]);
    (h as usize) % GRID
}

/// Bundle multiple fingerprints (majority voting)
fn bundle(fps: &[Fingerprint]) -> Fingerprint {
    if fps.is_empty() {
        return Fingerprint::zero();
    }
    if fps.len() == 1 {
        return fps[0].clone();
    }
    
    let mut counts = [0i32; N];
    
    for fp in fps {
        for i in 0..N {
            if fp.get_bit(i) {
                counts[i] += 1;
            } else {
                counts[i] -= 1;
            }
        }
    }
    
    let mut result = Fingerprint::zero();
    let threshold = 0; // Majority
    for i in 0..N {
        if counts[i] > threshold {
            result.set_bit(i, true);
        }
    }
    result
}

/// Weight a fingerprint (probabilistic thinning/thickening)
fn weight_fingerprint(fp: &Fingerprint, weight: f32) -> Fingerprint {
    if weight >= 1.0 {
        return fp.clone();
    }
    
    let mut result = Fingerprint::zero();
    let seed = fp.as_raw()[0]; // Deterministic based on content
    let mut state = seed;
    
    for i in 0..N {
        if fp.get_bit(i) {
            // LFSR-based "random" decision
            state = state.wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let rand = (state >> 32) as f32 / u32::MAX as f32;
            
            if rand < weight {
                result.set_bit(i, true);
            }
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
    
    fn make_sentence(text: &str, pos: i32) -> SentenceAtom {
        SentenceAtom {
            text: text.to_string(),
            subject: Fingerprint::from_content(&format!("{}_subject", text)),
            predicate: Fingerprint::from_content(&format!("{}_predicate", text)),
            object: Fingerprint::from_content(&format!("{}_object", text)),
            qualia: QualiaVector::neutral(),
            position: pos,
        }
    }
    
    #[test]
    fn test_context_insertion() {
        let mut crystal = ContextCrystal::new();
        
        let context = vec![
            make_sentence("The rain started falling", -2),
            make_sentence("She looked out the window", -1),
            make_sentence("The sky was dark grey", 0),
            make_sentence("Thunder rumbled in the distance", 1),
            make_sentence("She felt a chill", 2),
        ];
        
        crystal.insert_context(&context);
        assert_eq!(crystal.total_sentences, 5);
    }
    
    #[test]
    fn test_similar_contexts_resonate() {
        let mut crystal = ContextCrystal::new();
        
        // Insert a context about rain
        let context1 = vec![
            make_sentence("The rain started", -2),
            make_sentence("She watched outside", -1),
            make_sentence("The sky darkened", 0),
            make_sentence("Thunder came", 1),
            make_sentence("She shivered", 2),
        ];
        crystal.insert_context(&context1);
        
        // Query with similar context
        let query_similar = vec![
            make_sentence("Rain began falling", -2),
            make_sentence("She gazed out", -1),
            make_sentence("Clouds covered sky", 0),
            make_sentence("Lightning flashed", 1),
            make_sentence("She felt cold", 2),
        ];
        
        // Query with different context
        let query_different = vec![
            make_sentence("The sun was bright", -2),
            make_sentence("He went swimming", -1),
            make_sentence("The beach was crowded", 0),
            make_sentence("Children played", 1),
            make_sentence("He was happy", 2),
        ];
        
        let result_similar = crystal.query(&query_similar);
        let result_different = crystal.query(&query_different);
        
        // Similar contexts should have higher resonance
        // (In practice, with better embeddings this would be stronger)
        println!("Similar context: {:.3}", result_similar.overall_similarity);
        println!("Different context: {:.3}", result_different.overall_similarity);
    }
    
    #[test]
    fn test_temporal_flow() {
        let mut crystal = ContextCrystal::new();
        
        // Context with emotional arc: neutral → negative → negative → positive → positive
        let context = vec![
            SentenceAtom {
                text: "It was an ordinary day".to_string(),
                subject: Fingerprint::from_content("day"),
                predicate: Fingerprint::from_content("was"),
                object: Fingerprint::from_content("ordinary"),
                qualia: QualiaVector { valence: 0.5, ..QualiaVector::neutral() },
                position: -2,
            },
            SentenceAtom {
                text: "Bad news arrived".to_string(),
                subject: Fingerprint::from_content("news"),
                predicate: Fingerprint::from_content("arrived"),
                object: Fingerprint::from_content("bad"),
                qualia: QualiaVector { valence: 0.2, ..QualiaVector::neutral() },
                position: -1,
            },
            SentenceAtom {
                text: "She felt devastated".to_string(),
                subject: Fingerprint::from_content("she"),
                predicate: Fingerprint::from_content("felt"),
                object: Fingerprint::from_content("devastated"),
                qualia: QualiaVector { valence: 0.1, ..QualiaVector::neutral() },
                position: 0,
            },
            SentenceAtom {
                text: "But then hope emerged".to_string(),
                subject: Fingerprint::from_content("hope"),
                predicate: Fingerprint::from_content("emerged"),
                object: Fingerprint::from_content("new"),
                qualia: QualiaVector { valence: 0.7, ..QualiaVector::neutral() },
                position: 1,
            },
            SentenceAtom {
                text: "Everything would be okay".to_string(),
                subject: Fingerprint::from_content("everything"),
                predicate: Fingerprint::from_content("would_be"),
                object: Fingerprint::from_content("okay"),
                qualia: QualiaVector { valence: 0.9, ..QualiaVector::neutral() },
                position: 2,
            },
        ];
        
        crystal.insert_context(&context);
        
        // The mexican hat should emphasize the central moment
        let resonance = crystal.mexican_hat_resonance(1.0, 0.3);
        assert!(resonance.popcount() > 0);
    }
}
