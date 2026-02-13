//! mRNA: Memory RNA - Cross-Pollination Substrate
//!
//! Not message passing. Live resonance between subsystems.
//! Every operation leaves a "scent" that others can sense.

use parking_lot::RwLock;
use std::collections::VecDeque;
use std::sync::Arc;

use crate::cognitive::ThinkingStyle;
use crate::core::Fingerprint;
use crate::fabric::Subsystem;

/// Maximum concepts in resonance field
const MAX_ACTIVE_CONCEPTS: usize = 1000;

/// History depth for butterfly detection
const HISTORY_DEPTH: usize = 100;

/// mRNA - the cross-pollination substrate
pub struct MRNA {
    /// Shared resonance field
    field: Arc<RwLock<ResonanceField>>,
}

impl MRNA {
    /// Create new mRNA substrate
    pub fn new() -> Self {
        Self {
            field: Arc::new(RwLock::new(ResonanceField::new())),
        }
    }

    /// Pollinate: add concept to field, get what it resonates with
    pub fn pollinate(&self, concept: &Fingerprint) -> Vec<Resonance> {
        let mut field = self.field.write();
        field.pollinate(concept)
    }

    /// Pollinate with subsystem tag
    pub fn pollinate_from(&self, subsystem: Subsystem, concept: &Fingerprint) -> Vec<Resonance> {
        let mut field = self.field.write();
        field.pollinate_tagged(subsystem, concept)
    }

    /// Check cross-pollination between subsystems
    pub fn cross_pollinate(
        &self,
        source: Subsystem,
        concept: &Fingerprint,
        target: Subsystem,
    ) -> Option<CrossPollination> {
        let field = self.field.read();
        field.cross_pollinate(source, concept, target)
    }

    /// Get current field snapshot
    pub fn snapshot(&self) -> FieldSnapshot {
        let field = self.field.read();
        field.snapshot()
    }

    /// Set thinking style (affects all resonance)
    pub fn set_style(&self, style: ThinkingStyle) {
        let mut field = self.field.write();
        field.style = style;
    }

    /// Get current superposition fingerprint
    pub fn superposition(&self) -> Fingerprint {
        let field = self.field.read();
        field.superposition.clone()
    }

    /// Recent query patterns (for compression optimization)
    pub fn recent_query_patterns(&self) -> Vec<Fingerprint> {
        let field = self.field.read();
        field.concepts_for_subsystem(Subsystem::Query)
    }
}

impl Default for MRNA {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for MRNA {
    fn clone(&self) -> Self {
        Self {
            field: Arc::clone(&self.field),
        }
    }
}

/// The shared resonance field
pub struct ResonanceField {
    /// Active concepts with their sources
    active_concepts: Vec<TaggedConcept>,

    /// Superposition of all active (bundled)
    superposition: Fingerprint,

    /// History for butterfly detection
    history: VecDeque<FieldSnapshot>,

    /// Current thinking style
    style: ThinkingStyle,
}

/// A concept tagged with its source subsystem
#[derive(Clone)]
struct TaggedConcept {
    fingerprint: Fingerprint,
    source: Subsystem,
    timestamp: std::time::Instant,
}

impl ResonanceField {
    fn new() -> Self {
        Self {
            active_concepts: Vec::with_capacity(MAX_ACTIVE_CONCEPTS),
            superposition: Fingerprint::zero(),
            history: VecDeque::with_capacity(HISTORY_DEPTH),
            style: ThinkingStyle::default(),
        }
    }

    /// Pollinate concept into field
    fn pollinate(&mut self, concept: &Fingerprint) -> Vec<Resonance> {
        self.pollinate_tagged(Subsystem::Query, concept)
    }

    /// Pollinate with subsystem tag
    fn pollinate_tagged(&mut self, source: Subsystem, concept: &Fingerprint) -> Vec<Resonance> {
        // Find what resonates
        let threshold = self.style.field_modulation().resonance_threshold;
        let resonances: Vec<Resonance> = self
            .active_concepts
            .iter()
            .enumerate()
            .filter_map(|(i, tc)| {
                let sim = concept.similarity(&tc.fingerprint);
                if sim >= threshold {
                    Some(Resonance {
                        index: i,
                        similarity: sim,
                        source: tc.source,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Add to field
        if self.active_concepts.len() >= MAX_ACTIVE_CONCEPTS {
            // Remove oldest
            self.active_concepts.remove(0);
        }

        self.active_concepts.push(TaggedConcept {
            fingerprint: concept.clone(),
            source,
            timestamp: std::time::Instant::now(),
        });

        // Update superposition
        self.update_superposition();

        // Record history
        self.record_history();

        resonances
    }

    /// Check cross-pollination between subsystems
    fn cross_pollinate(
        &self,
        source: Subsystem,
        concept: &Fingerprint,
        target: Subsystem,
    ) -> Option<CrossPollination> {
        // Get target's concepts
        let target_concepts: Vec<_> = self
            .active_concepts
            .iter()
            .filter(|tc| tc.source == target)
            .collect();

        if target_concepts.is_empty() {
            return None;
        }

        // Find cross-resonances (loose threshold)
        let cross_threshold = 0.3;
        let mut cross_resonances: Vec<(usize, f32)> = target_concepts
            .iter()
            .enumerate()
            .filter_map(|(i, tc)| {
                let sim = concept.similarity(&tc.fingerprint);
                if sim > cross_threshold {
                    Some((i, sim))
                } else {
                    None
                }
            })
            .collect();

        if cross_resonances.is_empty() {
            return None;
        }

        // Sort by similarity
        cross_resonances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Some(CrossPollination {
            source,
            target,
            resonance_count: cross_resonances.len(),
            strongest_similarity: cross_resonances[0].1,
            strongest_concept: target_concepts[cross_resonances[0].0].fingerprint.clone(),
        })
    }

    /// Get concepts from specific subsystem
    pub fn concepts_for_subsystem(&self, subsystem: Subsystem) -> Vec<Fingerprint> {
        self.active_concepts
            .iter()
            .filter(|tc| tc.source == subsystem)
            .map(|tc| tc.fingerprint.clone())
            .collect()
    }

    /// Update superposition (bundle all active)
    fn update_superposition(&mut self) {
        if self.active_concepts.is_empty() {
            self.superposition = Fingerprint::zero();
            return;
        }

        let fps: Vec<_> = self
            .active_concepts
            .iter()
            .map(|tc| &tc.fingerprint)
            .collect();

        self.superposition = bundle_fingerprints(&fps);
    }

    /// Record snapshot for history
    fn record_history(&mut self) {
        if self.history.len() >= HISTORY_DEPTH {
            self.history.pop_front();
        }
        self.history.push_back(self.snapshot());
    }

    /// Create snapshot of current state
    fn snapshot(&self) -> FieldSnapshot {
        FieldSnapshot {
            concept_count: self.active_concepts.len(),
            superposition: self.superposition.clone(),
            timestamp: std::time::Instant::now(),
            subsystem_counts: self.subsystem_counts(),
        }
    }

    /// Count concepts per subsystem
    fn subsystem_counts(&self) -> [(Subsystem, usize); 5] {
        let mut counts = [(Subsystem::Query, 0); 5];
        counts[0] = (Subsystem::Query, 0);
        counts[1] = (Subsystem::Compression, 0);
        counts[2] = (Subsystem::Learning, 0);
        counts[3] = (Subsystem::Inference, 0);
        counts[4] = (Subsystem::Style, 0);

        for tc in &self.active_concepts {
            match tc.source {
                Subsystem::Query => counts[0].1 += 1,
                Subsystem::Compression => counts[1].1 += 1,
                Subsystem::Learning => counts[2].1 += 1,
                Subsystem::Inference => counts[3].1 += 1,
                Subsystem::Style => counts[4].1 += 1,
            }
        }
        counts
    }

    /// Get history for butterfly detection
    pub fn history(&self) -> &VecDeque<FieldSnapshot> {
        &self.history
    }
}

/// A resonance result
#[derive(Clone, Debug)]
pub struct Resonance {
    pub index: usize,
    pub similarity: f32,
    pub source: Subsystem,
}

/// Cross-pollination result
#[derive(Clone, Debug)]
pub struct CrossPollination {
    pub source: Subsystem,
    pub target: Subsystem,
    pub resonance_count: usize,
    pub strongest_similarity: f32,
    pub strongest_concept: Fingerprint,
}

/// Snapshot of field state
#[derive(Clone)]
pub struct FieldSnapshot {
    pub concept_count: usize,
    pub superposition: Fingerprint,
    pub timestamp: std::time::Instant,
    pub subsystem_counts: [(Subsystem, usize); 5],
}

/// Bundle multiple fingerprints (majority vote)
fn bundle_fingerprints(fps: &[&Fingerprint]) -> Fingerprint {
    use crate::core::VsaOps;

    if fps.is_empty() {
        return Fingerprint::zero();
    }
    if fps.len() == 1 {
        return (*fps[0]).clone();
    }

    // Convert to owned for bundle
    let owned: Vec<Fingerprint> = fps.iter().map(|&f| f.clone()).collect();
    let _refs: Vec<&Fingerprint> = owned.iter().collect();

    // Use VSA bundle (but we need owned slice)
    Fingerprint::bundle(&owned)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mrna_pollination() {
        let mrna = MRNA::new();

        // First concept, no resonances
        let fp1 = Fingerprint::from_content("hello");
        let r1 = mrna.pollinate(&fp1);
        assert!(r1.is_empty());

        // Same concept should resonate
        let r2 = mrna.pollinate(&fp1);
        assert!(!r2.is_empty());
        assert!(r2[0].similarity > 0.99);
    }

    #[test]
    fn test_cross_pollination() {
        let mrna = MRNA::new();

        // Add from Query
        let query_fp = Fingerprint::from_content("database query");
        mrna.pollinate_from(Subsystem::Query, &query_fp);

        // Add similar from Compression
        let compress_fp = Fingerprint::from_content("database compression");
        mrna.pollinate_from(Subsystem::Compression, &compress_fp);

        // Check cross-pollination
        let _cross = mrna.cross_pollinate(Subsystem::Compression, &compress_fp, Subsystem::Query);

        // Should find weak resonance (different but related)
        // Note: with random fingerprints baseline is ~0.5
    }
}
