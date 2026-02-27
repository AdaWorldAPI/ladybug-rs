//! SubstrateView implementation for BindSpace.
//!
//! Implements the `SubstrateView` trait from crewai-rust for the actual
//! BindSpace, enabling zero-serde data flow between BindSpace and Blackboard.
//!
//! # Data Flow
//!
//! ```text
//! crewai-rust Blackboard            ladybug-rs BindSpace
//! ══════════════════════            ═══════════════════
//!
//! AwarenessFrame  ◄── hydrate() ──  Hamming search (0x80-0xFF)
//! NarsSemanticState ── writeback()→  Meta words 4-7 (NARS surface 0x04)
//! XOR deltas       ── flush() ──→  fingerprint[i] ^= delta[i]
//! ```
//!
//! crewai-rust is OBLIGATORY — always linked.

use crate::storage::bind_space::{
    Addr, BindSpace, FINGERPRINT_WORDS, PREFIX_NARS, PREFIX_NODE_END, PREFIX_NODE_START,
    PREFIX_SURFACE_END, hamming_distance,
};
use crate::container::MetaViewMut;

use crewai_vendor::blackboard::bind_bridge::{SubstrateMatch, SubstrateView};

// ============================================================================
// SubstrateView for BindSpace
// ============================================================================

impl SubstrateView for BindSpace {
    fn read_fingerprint(&self, addr: u16) -> Option<[u64; FINGERPRINT_WORDS]> {
        self.read(Addr(addr)).map(|node| node.fingerprint)
    }

    fn read_label(&self, addr: u16) -> Option<String> {
        self.read(Addr(addr)).and_then(|node| node.label.clone())
    }

    fn read_truth(&self, addr: u16) -> Option<(f32, f32)> {
        self.read(Addr(addr)).map(|node| node.nars())
    }

    fn write_truth(&mut self, addr: u16, frequency: f32, confidence: f32) {
        if let Some(node) = self.read_mut(Addr(addr)) {
            let mut meta = MetaViewMut::new(node.meta_words_mut());
            meta.set_nars_frequency(frequency);
            meta.set_nars_confidence(confidence);
        }
    }

    fn hamming_search(
        &self,
        query: &[u64; FINGERPRINT_WORDS],
        prefix_range: (u8, u8),
        top_k: usize,
        threshold: f32,
    ) -> Vec<SubstrateMatch> {
        let total_bits = (FINGERPRINT_WORDS * 64) as f32;
        let mut matches = Vec::new();

        for prefix in prefix_range.0..=prefix_range.1 {
            for slot in 0..=255u8 {
                let addr = Addr::new(prefix, slot);
                if let Some(node) = self.read(addr) {
                    let dist = hamming_distance(&node.fingerprint, query);
                    let similarity = 1.0 - (dist as f32 / total_bits);

                    if similarity >= threshold {
                        matches.push(SubstrateMatch {
                            addr: addr.0,
                            similarity,
                            label: node.label.clone(),
                            truth: Some(node.nars()),
                        });
                    }
                }
            }
        }

        // Sort by similarity descending
        matches.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        matches.truncate(top_k);
        matches
    }

    fn write_fingerprint(&mut self, addr: u16, fingerprint: [u64; FINGERPRINT_WORDS]) -> bool {
        self.write_at(Addr(addr), fingerprint)
    }

    fn xor_delta(&mut self, addr: u16, delta: [u64; FINGERPRINT_WORDS]) {
        if let Some(node) = self.read_mut(Addr(addr)) {
            for (w, d) in node.fingerprint.iter_mut().zip(delta.iter()) {
                *w ^= d;
            }
        }
    }

    fn noise_floor(&self, prefix_range: (u8, u8)) -> f32 {
        // Compute noise floor as average similarity of bottom 10% of non-empty slots.
        // Uses a zero fingerprint as baseline query.
        let total_bits = (FINGERPRINT_WORDS * 64) as f32;
        let zero_query = [0u64; FINGERPRINT_WORDS];
        let mut similarities = Vec::new();

        for prefix in prefix_range.0..=prefix_range.1 {
            for slot in 0..=255u8 {
                let addr = Addr::new(prefix, slot);
                if let Some(node) = self.read(addr) {
                    let dist = hamming_distance(&node.fingerprint, &zero_query);
                    let sim = 1.0 - (dist as f32 / total_bits);
                    similarities.push(sim);
                }
            }
        }

        if similarities.is_empty() {
            return 0.0;
        }

        similarities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Bottom 10%
        let bottom_n = (similarities.len() / 10).max(1);
        let sum: f32 = similarities[..bottom_n].iter().sum();
        sum / bottom_n as f32
    }

    fn nars_surface(&self) -> Vec<(u8, String, f32, f32)> {
        let mut results = Vec::new();
        // NARS surface is prefix 0x04, slots 0x00-0x7F (NARS half)
        for slot in 0..=0x7Fu8 {
            let addr = Addr::new(PREFIX_NARS, slot);
            if let Some(node) = self.read(addr) {
                let (freq, conf) = node.nars();
                let label = node.label.clone().unwrap_or_default();
                if freq != 0.0 || conf != 0.0 {
                    results.push((slot, label, freq, conf));
                }
            }
        }
        results
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_substrate_view_read_write() {
        let mut space = BindSpace::new();
        let fp = [42u64; FINGERPRINT_WORDS];
        // Write to orchestration surface (0x0C prefix — agents)
        assert!(space.write_fingerprint(0x0C00, fp));
        let read = space.read_fingerprint(0x0C00);
        assert!(read.is_some());
        assert_eq!(read.unwrap()[0], 42);
    }

    #[test]
    fn test_substrate_view_truth() {
        let mut space = BindSpace::new();
        let fp = [0u64; FINGERPRINT_WORDS];
        space.write_fingerprint(0x0C01, fp);
        space.write_truth(0x0C01, 0.85, 0.72);

        let truth = space.read_truth(0x0C01);
        assert!(truth.is_some());
        let (f, c) = truth.unwrap();
        assert!((f - 0.85).abs() < 0.01);
        assert!((c - 0.72).abs() < 0.01);
    }

    #[test]
    fn test_substrate_view_xor_delta() {
        let mut space = BindSpace::new();
        let mut fp = [0u64; FINGERPRINT_WORDS];
        fp[0] = 0xFF;
        space.write_fingerprint(0x0C02, fp);

        let mut delta = [0u64; FINGERPRINT_WORDS];
        delta[0] = 0x0F; // flip bottom 4 bits
        space.xor_delta(0x0C02, delta);

        let result = space.read_fingerprint(0x0C02).unwrap();
        assert_eq!(result[0], 0xFF ^ 0x0F); // 0xF0
    }

    #[test]
    fn test_substrate_view_hamming_search() {
        let mut space = BindSpace::new();

        // Write a node
        let mut fp = [0u64; FINGERPRINT_WORDS];
        fp[0] = 0xFFFF_FFFF_FFFF_FFFF;
        let addr = space.write_labeled(fp, "test_node");

        // Search with similar query
        let mut query = [0u64; FINGERPRINT_WORDS];
        query[0] = 0xFFFF_FFFF_FFFF_FFFF;
        let prefix = addr.prefix();

        let results = space.hamming_search(&query, (prefix, prefix), 10, 0.5);
        assert!(!results.is_empty());
        assert!(results[0].similarity > 0.9);
        assert_eq!(results[0].label.as_deref(), Some("test_node"));
    }

    #[test]
    fn test_substrate_view_noise_floor_empty() {
        let space = BindSpace::new();
        let nf = space.noise_floor((0x80, 0xFF));
        assert_eq!(nf, 0.0);
    }

    #[test]
    fn test_substrate_view_label() {
        let mut space = BindSpace::new();
        let fp = [1u64; FINGERPRINT_WORDS];
        space.write_fingerprint(0x0C03, fp);

        // No label initially (write_fingerprint doesn't set label)
        assert!(space.read_label(0x0C03).is_none());

        // Use write_labeled via BindSpace API
        let addr = space.write_labeled([2u64; FINGERPRINT_WORDS], "hello");
        assert_eq!(space.read_label(addr.0).as_deref(), Some("hello"));
    }
}
