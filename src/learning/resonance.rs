//! ResonanceCapture — "Felt this before" via Hamming similarity

use crate::core::Fingerprint;
use crate::learning::moment::{Moment, Qualia};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct SimilarMoment {
    pub moment_id: String,
    pub resonance: f32,
    pub content_similarity: f32,
    pub qualia_distance: f32,
    pub cycle_delta: u64,
}

#[derive(Clone)]
struct StoredResonance {
    content_fp: Fingerprint,
    resonance_fp: Fingerprint,
    qualia: Qualia,
    cycle: u64,
    session_id: String,
}

pub struct ResonanceCapture {
    fingerprints: HashMap<String, StoredResonance>,
    batch_vectors: Vec<(String, Fingerprint)>,
    pub total_captures: u64,
    pub total_queries: u64,
    pub cache_hits: u64,
}

impl ResonanceCapture {
    pub fn new() -> Self {
        Self {
            fingerprints: HashMap::new(),
            batch_vectors: Vec::new(),
            total_captures: 0,
            total_queries: 0,
            cache_hits: 0,
        }
    }

    pub fn capture(&mut self, moment: &Moment, cycle: u64) {
        let stored = StoredResonance {
            content_fp: moment.fingerprint.clone(),
            resonance_fp: moment.resonance_vector.clone(),
            qualia: moment.qualia.clone(),
            cycle,
            session_id: moment.session_id.clone(),
        };

        self.fingerprints.insert(moment.id.clone(), stored);
        self.batch_vectors
            .push((moment.id.clone(), moment.resonance_vector.clone()));
        self.total_captures += 1;
    }

    pub fn find_resonant(
        &mut self,
        query: &Fingerprint,
        threshold: f32,
        limit: usize,
        current_cycle: u64,
    ) -> Vec<SimilarMoment> {
        self.total_queries += 1;

        let mut results: Vec<SimilarMoment> = self
            .batch_vectors
            .iter()
            .filter_map(|(id, fp)| {
                let resonance = query.similarity(fp);
                if resonance >= threshold {
                    let stored = self.fingerprints.get(id)?;
                    let content_similarity = query.similarity(&stored.content_fp);
                    let qualia_distance = Self::qualia_distance(&stored.qualia, &Qualia::default());
                    let cycle_delta = current_cycle.saturating_sub(stored.cycle);

                    Some(SimilarMoment {
                        moment_id: id.clone(),
                        resonance,
                        content_similarity,
                        qualia_distance,
                        cycle_delta,
                    })
                } else {
                    None
                }
            })
            .collect();

        results.sort_by(|a, b| {
            b.resonance
                .partial_cmp(&a.resonance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);
        results
    }

    fn qualia_distance(a: &Qualia, b: &Qualia) -> f32 {
        let dn = (a.novelty - b.novelty).powi(2);
        let de = (a.effort - b.effort).powi(2);
        let ds = (a.satisfaction - b.satisfaction).powi(2);
        let dc = (a.confusion - b.confusion).powi(2);
        let dsu = (a.surprise - b.surprise).powi(2);
        ((dn + de + ds + dc + dsu) / 5.0).sqrt()
    }

    pub fn stats(&self) -> ResonanceStats {
        ResonanceStats {
            total_captures: self.total_captures,
            total_queries: self.total_queries,
            cache_hits: self.cache_hits,
            unique_moments: self.fingerprints.len(),
            hit_rate: if self.total_queries > 0 {
                self.cache_hits as f32 / self.total_queries as f32
            } else {
                0.0
            },
        }
    }
}

impl Default for ResonanceCapture {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub struct ResonanceStats {
    pub total_captures: u64,
    pub total_queries: u64,
    pub cache_hits: u64,
    pub unique_moments: usize,
    pub hit_rate: f32,
}

pub fn mexican_hat_resonance(distances: &[f32], center: f32, width: f32) -> Vec<f32> {
    distances
        .iter()
        .map(|&d| {
            let x = (d - center) / width;
            let x2 = x * x;
            (1.0 - x2) * (-x2 / 2.0).exp()
        })
        .collect()
}

pub fn find_sweet_spot(
    store: &mut ResonanceCapture,
    query: &Fingerprint,
    current_cycle: u64,
) -> Option<SimilarMoment> {
    let candidates = store.find_resonant(query, 0.6, 20, current_cycle);

    let scored: Vec<(SimilarMoment, f32)> = candidates
        .into_iter()
        .map(|m| {
            let mexican = mexican_hat_resonance(&[m.resonance], 0.72, 0.1)[0];
            (m, mexican)
        })
        .collect();

    scored
        .into_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(m, _)| m)
}
