//! Cognitive Kernel Bridge — 10-Layer Stack ↔ BindSpace
//!
//! Maps 10-layer cognitive stack outputs directly to BindSpace operations.
//! No feature gates — this module operates on the raw BindSpace substrate
//! that is always available. The orchestration::SemanticKernel wraps this
//! for external consumers; internally, layers talk to BindSpace directly.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │                  COGNITIVE KERNEL BRIDGE                             │
//! │                                                                      │
//! │  10-Layer Stack                    BindSpace Operations              │
//! │  ═════════════                    ════════════════════               │
//! │  L1 Recognition  ───resonate───►  HDR similarity search (Node zone) │
//! │  L2 Resonance    ───resonate───►  Wider threshold (Fluid zone)      │
//! │  L3 Appraisal    ───evaluate──►   Gestalt coherence check           │
//! │  L4 Routing      ───style────►    ThinkingStyle resonance select    │
//! │  L5 Execution    ───write────►    Produce fingerprint to Fluid zone │
//! │  ─── single agent boundary ─────────────────────────────────────    │
//! │  L6 Delegation   ───resonate──►   Cross-agent blackboard search     │
//! │  L7 Contingency  ───xor_bind──►   Counterfactual branch             │
//! │  L8 Integration  ───bundle────►   Majority-vote evidence merge      │
//! │  L9 Validation   ───sieve────►    NARS + Brier + XOR + DK check    │
//! │  L10 Crystal.    ───promote──►    Fluid → Node crystallization      │
//! │                                                                      │
//! │  Causal Rung Mapping:                                               │
//! │    L3 (Appraisal) ↔ Rung 1: See (correlation)                      │
//! │    L5 (Execution) ↔ Rung 2: Do  (intervention)                     │
//! │    L7 (Continge.) ↔ Rung 3: Imagine (counterfactual)               │
//! └──────────────────────────────────────────────────────────────────────┘
//! ```

use crate::cognitive::layer_stack::{ConsciousnessSnapshot, LayerId, LayerNode, LayerResult, NUM_LAYERS};
use crate::cognitive::metacog::MetaCognition;
use crate::cognitive::satisfaction_gate::LayerSatisfaction;
use crate::cognitive::sieve::SocraticSieve;
use crate::cognitive::two_stroke::{self, ValidationResult};
use crate::core::Fingerprint;
use crate::nars::TruthValue;
use crate::storage::bind_space::{Addr, BindSpace, FINGERPRINT_WORDS};

use ladybug_contract::container::Container;
use ladybug_contract::wire::{self, CogPacket};

/// BindSpace prefix for working memory (Fluid zone start).
const PREFIX_FLUID: u8 = 0x10;
/// BindSpace prefix for crystallized nodes (Node zone start).
const PREFIX_NODE: u8 = 0x80;
/// BindSpace prefix for agent blackboards.
const PREFIX_BLACKBOARD: u8 = 0x0E;
/// BindSpace prefix for causal model fingerprints.
const PREFIX_CAUSAL: u8 = 0x05;

// =============================================================================
// COGNITIVE KERNEL
// =============================================================================

/// Cognitive kernel bridge — maps 10-layer stack outputs to BindSpace operations.
///
/// Operates directly on BindSpace without feature gates.
/// The semantic kernel in the orchestration module wraps this for external use;
/// internally, layers drive BindSpace reads/writes via fingerprint resonance.
pub struct CognitiveKernel {
    /// MetaCognition tracker (Brier calibration, L9 validation)
    metacog: MetaCognition,

    /// Satisfaction gate state
    satisfaction: LayerSatisfaction,

    /// Previous cycle scores for 2-stroke
    prev_scores: [f32; NUM_LAYERS],

    /// Precomputed rule fingerprints
    rule_fingerprints: Vec<two_stroke::RuleFingerprint>,

    /// Precomputed style fingerprints
    style_fingerprints: Vec<two_stroke::StyleFingerprint>,

    /// Next fluid slot for working memory writes
    next_fluid_slot: u8,

    /// Next node slot for crystallization writes
    next_node_slot: u8,

    /// Processing cycle
    cycle: u64,
}

impl CognitiveKernel {
    /// Create a new cognitive kernel bridge.
    pub fn new() -> Self {
        Self {
            metacog: MetaCognition::new(),
            satisfaction: LayerSatisfaction::new(),
            prev_scores: [0.5; NUM_LAYERS],
            rule_fingerprints: two_stroke::build_rule_fingerprints(),
            style_fingerprints: two_stroke::build_style_fingerprints(),
            next_fluid_slot: 0,
            next_node_slot: 0,
            cycle: 0,
        }
    }

    /// Process layer results through the cognitive kernel bridge.
    ///
    /// Each layer's activation drives specific BindSpace operations.
    /// No explicit dispatch — layer results carry the fingerprints that
    /// resonate with BindSpace addresses. The bridge translates layer
    /// semantics into BindSpace primitives (read/write/similarity).
    pub fn process_layer_results(
        &mut self,
        space: &mut BindSpace,
        results: &[LayerResult],
        snapshot: &ConsciousnessSnapshot,
        input_fp: &Fingerprint,
    ) -> CognitiveKernelResult {
        self.cycle += 1;
        let mut ops_performed = Vec::new();
        let mut crystallized = Vec::new();
        let mut validation = None;

        let input_words = fp_to_words(input_fp);

        for result in results {
            if result.output_activation < 0.1 {
                continue; // Layer didn't fire
            }

            match result.layer {
                LayerId::L1 => {
                    // Recognition → resonate in Node zone (pattern matching)
                    let hits = resonate_in_prefix(space, &input_words, PREFIX_NODE, 0.6, 5);
                    ops_performed.push(KernelLayerOp::Resonated {
                        layer: LayerId::L1,
                        hit_count: hits.len(),
                    });
                }

                LayerId::L2 => {
                    // Resonance → wider search across Fluid zone (association)
                    let threshold = (0.4 * result.output_activation).max(0.2);
                    let hits = resonate_in_prefix(space, &input_words, PREFIX_FLUID, threshold, 10);
                    ops_performed.push(KernelLayerOp::Resonated {
                        layer: LayerId::L2,
                        hit_count: hits.len(),
                    });
                }

                LayerId::L3 => {
                    // Appraisal → check causal zone for correlations (Rung 1: See)
                    let hits = resonate_in_prefix(space, &input_words, PREFIX_CAUSAL, 0.5, 5);
                    ops_performed.push(KernelLayerOp::CausalQuery {
                        layer: LayerId::L3,
                        rung: 1,
                        hit_count: hits.len(),
                    });
                }

                LayerId::L4 => {
                    // Routing → style self-selection via satisfaction resonance
                    let _selected = two_stroke::select_style_by_resonance(
                        &self.satisfaction,
                        &self.style_fingerprints,
                    );
                    ops_performed.push(KernelLayerOp::Routed { layer: LayerId::L4 });
                }

                LayerId::L5 => {
                    // Execution → write result to Fluid zone (working memory)
                    let fluid_addr = Addr::new(PREFIX_FLUID, self.next_fluid_slot);
                    space.write_at(fluid_addr, input_words);
                    self.next_fluid_slot = self.next_fluid_slot.wrapping_add(1);
                    ops_performed.push(KernelLayerOp::Executed {
                        layer: LayerId::L5,
                        addr: fluid_addr,
                    });
                }

                LayerId::L6 => {
                    // Delegation → search across agent blackboards (0x0E)
                    let hits = resonate_in_prefix(space, &input_words, PREFIX_BLACKBOARD, 0.3, 20);
                    ops_performed.push(KernelLayerOp::Delegated {
                        layer: LayerId::L6,
                        agent_count: hits.len(),
                    });
                }

                LayerId::L7 => {
                    // Contingency → counterfactual branch (XOR with snapshot state)
                    let snap_fp = snapshot_to_fp(snapshot);
                    let snap_words = fp_to_words(&snap_fp);
                    // XOR-compose input with snapshot = counterfactual fingerprint
                    let mut counterfactual = [0u64; FINGERPRINT_WORDS];
                    for i in 0..FINGERPRINT_WORDS {
                        counterfactual[i] = input_words[i] ^ snap_words[i];
                    }
                    let hits = resonate_in_prefix(space, &counterfactual, PREFIX_CAUSAL, 0.4, 5);
                    ops_performed.push(KernelLayerOp::CausalQuery {
                        layer: LayerId::L7,
                        rung: 3, // Imagine
                        hit_count: hits.len(),
                    });
                }

                LayerId::L8 => {
                    // Integration → majority-vote bundle of recent evidence
                    let bundled = bundle_recent(
                        space,
                        PREFIX_FLUID,
                        self.next_fluid_slot,
                        3, // merge last 3
                    );
                    if let Some(merged_fp) = bundled {
                        let target = Addr::new(PREFIX_FLUID, self.next_fluid_slot);
                        space.write_at(target, merged_fp);
                        self.next_fluid_slot = self.next_fluid_slot.wrapping_add(1);
                    }
                    ops_performed.push(KernelLayerOp::Integrated {
                        layer: LayerId::L8,
                        merged: bundled.is_some(),
                    });
                }

                LayerId::L9 => {
                    // Validation → NARS + Brier + XOR residual + Dunning-Kruger
                    let nars_tv = TruthValue::new(
                        result.output_activation,
                        result.input_resonance,
                    );
                    let calibration_error = self.metacog.brier_score();

                    // XOR residual: popcount distance from centroid
                    let centroid = Fingerprint::from_content("centroid");
                    let (xor_residual, centroid_pop) =
                        SocraticSieve::xor_residual(input_fp, &centroid);

                    let (pass, reason, adjusted_conf) = SocraticSieve::validate_l9(
                        &nars_tv,
                        calibration_error,
                        xor_residual,
                        centroid_pop,
                    );

                    // Track with MetaCognition
                    let _assessment = self.metacog.assess(
                        crate::cognitive::GateState::Flow,
                        &nars_tv,
                    );

                    let vr = if pass {
                        ValidationResult::Pass {
                            nars_confidence: adjusted_conf,
                            meta_confidence: 1.0 - calibration_error,
                        }
                    } else {
                        ValidationResult::Reject { reason }
                    };

                    validation = Some(vr.clone());
                    ops_performed.push(KernelLayerOp::Validated {
                        layer: LayerId::L9,
                        passed: pass,
                    });
                }

                LayerId::L10 => {
                    // Crystallization → promote from Fluid to Node zone
                    // Only if L9 validation passed
                    let should_crystallize = matches!(
                        &validation,
                        Some(ValidationResult::Pass { .. })
                    );

                    if should_crystallize {
                        let fluid_addr =
                            Addr::new(PREFIX_FLUID, self.next_fluid_slot.wrapping_sub(1));
                        let node_addr = Addr::new(PREFIX_NODE, self.next_node_slot);

                        // Read from fluid, write to node (crystallize)
                        let did_crystallize = if let Some(node) = space.read(fluid_addr) {
                            if node.fingerprint.iter().any(|&w| w != 0) {
                                space.write_at(node_addr, node.fingerprint);
                                true
                            } else {
                                false
                            }
                        } else {
                            false
                        };

                        if did_crystallize {
                            crystallized.push(node_addr);
                            self.next_node_slot = self.next_node_slot.wrapping_add(1);
                        }
                        ops_performed.push(KernelLayerOp::Crystallized {
                            layer: LayerId::L10,
                            addr: if did_crystallize {
                                Some(node_addr)
                            } else {
                                None
                            },
                        });
                    }
                }
            }

            // Update satisfaction scores
            self.satisfaction.update(result.layer, result.output_activation);
        }

        self.prev_scores = two_stroke::snapshot_scores(&self.satisfaction);

        CognitiveKernelResult {
            cycle: self.cycle,
            ops_performed,
            crystallized,
            validation,
            satisfaction: self.satisfaction.clone(),
        }
    }

    /// Record a prediction outcome for Brier calibration.
    pub fn record_outcome(&mut self, predicted_confidence: f32, actual_outcome: f32) {
        self.metacog.record_outcome(predicted_confidence, actual_outcome);
    }

    /// Get metacognition state.
    pub fn metacognition(&self) -> &MetaCognition {
        &self.metacog
    }

    /// Get satisfaction gate state.
    pub fn satisfaction(&self) -> &LayerSatisfaction {
        &self.satisfaction
    }

    // =========================================================================
    // L4 ASSIMILATION — Process CogPackets by resonance, not string dispatch
    // =========================================================================

    /// Process an incoming CogPacket through the 10-layer stack.
    ///
    /// This is the L4 Self-Realization path: the contract IS the layer stack.
    /// Routing happens by fingerprint resonance, not by step_type string parsing.
    /// The packet's content Container resonates against BindSpace to determine
    /// which layers fire and what operations execute.
    ///
    /// Returns a response CogPacket with crystallization results.
    pub fn process_packet(
        &mut self,
        space: &mut BindSpace,
        packet: &CogPacket,
        node: &mut crate::cognitive::layer_stack::LayerNode,
    ) -> (CognitiveKernelResult, CogPacket) {
        // Extract fingerprint from Container payload (Container = 8192 bits,
        // BindSpace = 16384 bits — zero-pad the upper half)
        let content = packet.content();
        let input_fp = container_to_fingerprint(content);

        // Apply field modulation from packet header to satisfaction gate
        let threshold = packet.resonance_threshold();
        if threshold > 0.0 {
            // Packet carries modulation — L4 uses it as style hint
            // This is how agents communicate their FieldModulation through
            // binary protocol instead of JSON metadata
            for layer in LayerId::ALL {
                let sat = packet.satisfaction(layer.index() as u8);
                if sat > 0.0 {
                    self.satisfaction.update(layer, sat);
                }
            }
        }

        // Apply NARS truth value from packet header
        let pkt_tv = packet.truth_value();
        if pkt_tv.confidence > 0.0 {
            self.metacog.record_outcome(pkt_tv.confidence, pkt_tv.frequency);
        }

        // Process through layer stack (deterministic wave processing)
        let results = crate::cognitive::layer_stack::process_layers_wave(
            node, &input_fp, self.cycle,
        );

        // Process layer results through the kernel bridge
        let snapshot = crate::cognitive::layer_stack::snapshot_consciousness(node, self.cycle);
        let ck_result = self.process_layer_results(space, &results, &snapshot, &input_fp);

        // Build response packet
        let mut response = if let Some(crystallized_addr) = ck_result.crystallized.first() {
            // Crystallization occurred — return the crystallized fingerprint
            if let Some(crystal_node) = space.read(*crystallized_addr) {
                let crystal_container = fingerprint_to_container(&crystal_node.fingerprint);
                CogPacket::response(
                    wire::wire_ops::CRYSTALLIZE,
                    packet.target_addr(),
                    packet.source_addr(),
                    crystal_container,
                )
            } else {
                CogPacket::response(
                    packet.opcode(),
                    packet.target_addr(),
                    packet.source_addr(),
                    content.clone(),
                )
            }
        } else {
            // No crystallization — return processed result
            CogPacket::response(
                packet.opcode(),
                packet.target_addr(),
                packet.source_addr(),
                content.clone(),
            )
        };

        // Populate response header with cognitive state
        response.set_cycle(ck_result.cycle);
        response.set_layer(snapshot.dominant_layer.index() as u8);

        // Pack satisfaction scores
        let scores = two_stroke::snapshot_scores(&ck_result.satisfaction);
        let mut sat_array = [0.0f32; 10];
        for (i, &s) in scores.iter().enumerate().take(10) {
            sat_array[i] = s;
        }
        response.set_satisfaction_array(&sat_array);

        // Pack NARS truth value from validation
        if let Some(ref validation) = ck_result.validation {
            match validation {
                ValidationResult::Pass { nars_confidence, meta_confidence } => {
                    let tv = ladybug_contract::TruthValue::new(*meta_confidence, *nars_confidence);
                    response.set_truth_value(&tv);
                    response.set_flags(response.flags() | wire::FLAG_VALIDATED);
                }
                ValidationResult::Reject { .. } | ValidationResult::Hold { .. } => {
                    response.set_flags(response.flags() | wire::FLAG_ERROR);
                }
            }
        }

        // Pack causal rung from dominant layer
        let rung = match snapshot.dominant_layer {
            LayerId::L3 => 1, // See
            LayerId::L5 => 2, // Do
            LayerId::L7 => 3, // Imagine
            _ => 0,
        };
        response.set_rung(rung);

        if !ck_result.crystallized.is_empty() {
            response.set_flags(response.flags() | wire::FLAG_CRYSTALLIZED);
        }

        response.update_checksum();
        (ck_result, response)
    }
}

impl Default for CognitiveKernel {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// CONTAINER ↔ FINGERPRINT BRIDGING
// =============================================================================

/// Convert a Container (8192 bits = 128 × u64) to a Fingerprint-compatible
/// word array (16384 bits = 256 × u64). Upper half zero-padded.
fn container_to_fingerprint(container: &Container) -> Fingerprint {
    let mut words = [0u64; FINGERPRINT_WORDS];
    for (i, &w) in container.words.iter().enumerate() {
        words[i] = w;
    }
    // XOR-fold: replicate lower 128 words into upper 128 with permutation
    // This preserves information density while filling the full 16K space
    for i in 0..128 {
        words[128 + i] = container.words[i].rotate_left(7);
    }
    Fingerprint::from_raw(words)
}

/// Convert a BindSpace fingerprint array back to a Container.
/// Truncates to 128 words (8192 bits).
fn fingerprint_to_container(fp: &[u64; FINGERPRINT_WORDS]) -> Container {
    let mut words = [0u64; 128];
    words.copy_from_slice(&fp[..128]);
    Container { words }
}

// =============================================================================
// RESULT TYPES
// =============================================================================

/// Result of processing layer results through the cognitive kernel.
#[derive(Clone, Debug)]
pub struct CognitiveKernelResult {
    /// Processing cycle
    pub cycle: u64,
    /// Operations performed per layer
    pub ops_performed: Vec<KernelLayerOp>,
    /// Addresses where crystallization occurred
    pub crystallized: Vec<Addr>,
    /// L9 validation result (if L9 fired)
    pub validation: Option<ValidationResult>,
    /// Current satisfaction state
    pub satisfaction: LayerSatisfaction,
}

/// A kernel operation triggered by a layer result.
#[derive(Clone, Debug)]
pub enum KernelLayerOp {
    /// Layer triggered resonance search
    Resonated { layer: LayerId, hit_count: usize },
    /// Layer triggered causal query at a given rung
    CausalQuery { layer: LayerId, rung: u8, hit_count: usize },
    /// Layer performed routing (style selection)
    Routed { layer: LayerId },
    /// Layer wrote to working memory
    Executed { layer: LayerId, addr: Addr },
    /// Layer searched agent blackboards
    Delegated { layer: LayerId, agent_count: usize },
    /// Layer performed evidence merge
    Integrated { layer: LayerId, merged: bool },
    /// Layer performed validation
    Validated { layer: LayerId, passed: bool },
    /// Layer crystallized result to Node zone
    Crystallized { layer: LayerId, addr: Option<Addr> },
}

// =============================================================================
// BINDSPACE HELPERS (no feature gates — always available)
// =============================================================================

/// Convert Fingerprint to raw word array for BindSpace compatibility.
fn fp_to_words(fp: &Fingerprint) -> [u64; FINGERPRINT_WORDS] {
    let raw = fp.as_raw();
    let mut result = [0u64; FINGERPRINT_WORDS];
    let len = raw.len().min(FINGERPRINT_WORDS);
    result[..len].copy_from_slice(&raw[..len]);
    result
}

/// Create a fingerprint from a consciousness snapshot (for causal queries).
fn snapshot_to_fp(snapshot: &ConsciousnessSnapshot) -> Fingerprint {
    let mut encoded = String::with_capacity(128);
    encoded.push_str("snapshot:");
    for (i, layer) in snapshot.layers.iter().enumerate() {
        if layer.active {
            encoded.push_str(&format!("L{}={:.2},", i + 1, layer.value));
        }
    }
    encoded.push_str(&format!(
        "coh={:.2},em={:.2}",
        snapshot.coherence, snapshot.emergence
    ));
    Fingerprint::from_content(&encoded)
}

/// Hamming similarity between two fingerprint word arrays.
fn hamming_similarity(a: &[u64; FINGERPRINT_WORDS], b: &[u64; FINGERPRINT_WORDS]) -> f32 {
    let total_bits = (FINGERPRINT_WORDS * 64) as f32;
    let matching: u32 = (0..FINGERPRINT_WORDS)
        .map(|i| (!(a[i] ^ b[i])).count_ones())
        .sum();
    matching as f32 / total_bits
}

/// Search a BindSpace prefix for fingerprints similar to target.
/// Returns (addr, similarity) pairs sorted by similarity descending.
///
/// Popcount stacking early-exit: skip slots where popcount difference
/// exceeds 2σ from expected — impossible to be within threshold.
fn resonate_in_prefix(
    space: &BindSpace,
    target: &[u64; FINGERPRINT_WORDS],
    prefix: u8,
    threshold: f32,
    limit: usize,
) -> Vec<(Addr, f32)> {
    let mut results = Vec::new();

    // Popcount of target for early-exit (popcount stacking)
    let target_popcount: u32 = target.iter().map(|w| w.count_ones()).sum();
    let total_bits = (FINGERPRINT_WORDS * 64) as f32;

    // 1-2 sigma tolerance: if popcounts differ too much, similarity can't
    // reach threshold. max_hamming_distance = total_bits * (1 - threshold).
    // popcount_diff > 2 * max_hamming gives early exit.
    let max_hamming = (total_bits * (1.0 - threshold)) as u32;

    for slot in 0..=255u8 {
        let addr = Addr::new(prefix, slot);
        if let Some(node) = space.read(addr) {
            if node.fingerprint.iter().all(|&w| w == 0) {
                continue;
            }

            // Early exit: popcount difference check (O(1) — no XOR needed)
            let candidate_popcount: u32 = node.fingerprint.iter().map(|w| w.count_ones()).sum();
            let pop_diff = (target_popcount as i64 - candidate_popcount as i64).unsigned_abs() as u32;
            if pop_diff > max_hamming * 2 {
                continue; // Can't possibly reach threshold
            }

            let similarity = hamming_similarity(target, &node.fingerprint);
            if similarity >= threshold {
                results.push((addr, similarity));
            }
        }
    }

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(limit);
    results
}

/// Majority-vote bundle of the last N fingerprints at a prefix.
/// Returns the bundled fingerprint, or None if insufficient sources.
fn bundle_recent(
    space: &BindSpace,
    prefix: u8,
    current_slot: u8,
    count: u8,
) -> Option<[u64; FINGERPRINT_WORDS]> {
    let mut sources = Vec::new();

    for i in 0..count {
        let slot = current_slot.wrapping_sub(count).wrapping_add(i);
        let addr = Addr::new(prefix, slot);
        if let Some(node) = space.read(addr) {
            if node.fingerprint.iter().any(|&w| w != 0) {
                sources.push(node.fingerprint);
            }
        }
    }

    if sources.len() < 2 {
        return None;
    }

    let threshold = sources.len() / 2;
    let mut result = [0u64; FINGERPRINT_WORDS];

    for word_idx in 0..FINGERPRINT_WORDS {
        for bit in 0..64 {
            let mask = 1u64 << bit;
            let count = sources.iter().filter(|fp| fp[word_idx] & mask != 0).count();
            if count > threshold {
                result[word_idx] |= mask;
            }
        }
    }

    Some(result)
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive::layer_stack::{LayerNode, process_layers_wave, snapshot_consciousness};

    #[test]
    fn test_cognitive_kernel_creation() {
        let ck = CognitiveKernel::new();
        assert_eq!(ck.cycle, 0);
    }

    #[test]
    fn test_cognitive_kernel_process() {
        let mut ck = CognitiveKernel::new();
        let mut space = BindSpace::new();
        let mut node = LayerNode::new("test");
        let input = Fingerprint::from_content("test input");

        let results = process_layers_wave(&mut node, &input, 1);
        let snapshot = snapshot_consciousness(&node, 1);

        let ck_result = ck.process_layer_results(&mut space, &results, &snapshot, &input);
        assert_eq!(ck_result.cycle, 1);
        assert!(!ck_result.ops_performed.is_empty());
    }

    #[test]
    fn test_cognitive_kernel_crystallization_requires_validation() {
        let mut ck = CognitiveKernel::new();
        let mut space = BindSpace::new();
        let mut node = LayerNode::new("test");
        let input = Fingerprint::from_content("data");

        let results = process_layers_wave(&mut node, &input, 1);
        let snapshot = snapshot_consciousness(&node, 1);

        let ck_result = ck.process_layer_results(&mut space, &results, &snapshot, &input);
        assert!(
            ck_result.crystallized.is_empty(),
            "Should not crystallize without L9 validation"
        );
    }

    #[test]
    fn test_hamming_similarity_identical() {
        let a = [0xFFu64; FINGERPRINT_WORDS];
        assert!((hamming_similarity(&a, &a) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_hamming_similarity_opposite() {
        let a = [0xFFFF_FFFF_FFFF_FFFFu64; FINGERPRINT_WORDS];
        let b = [0x0000_0000_0000_0000u64; FINGERPRINT_WORDS];
        let sim = hamming_similarity(&a, &b);
        assert!(sim < 0.01, "Opposite should be near 0: {}", sim);
    }

    #[test]
    fn test_resonate_finds_written_fingerprint() {
        let mut space = BindSpace::new();
        let fp = [0xDEAD_BEEF_CAFE_BABEu64; FINGERPRINT_WORDS];
        let addr = Addr::new(PREFIX_NODE, 0x42);
        space.write_at(addr, fp);

        let results = resonate_in_prefix(&space, &fp, PREFIX_NODE, 0.9, 5);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, addr);
        assert!((results[0].1 - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_popcount_early_exit() {
        let mut space = BindSpace::new();
        // Write a very dense fingerprint
        let dense = [0xFFFF_FFFF_FFFF_FFFFu64; FINGERPRINT_WORDS];
        space.write_at(Addr::new(PREFIX_NODE, 0), dense);

        // Search for a very sparse fingerprint — should early-exit
        let sparse = [0x0000_0000_0000_0001u64; FINGERPRINT_WORDS];
        let results = resonate_in_prefix(&space, &sparse, PREFIX_NODE, 0.9, 5);
        assert!(results.is_empty(), "Extreme popcount difference should be filtered");
    }

    #[test]
    fn test_bundle_majority_vote() {
        let mut space = BindSpace::new();
        let all_ones = [0xFFFF_FFFF_FFFF_FFFFu64; FINGERPRINT_WORDS];
        let all_zeros = [0u64; FINGERPRINT_WORDS];

        // Write 3 ones and 1 zero
        space.write_at(Addr::new(PREFIX_FLUID, 0), all_ones);
        space.write_at(Addr::new(PREFIX_FLUID, 1), all_ones);
        space.write_at(Addr::new(PREFIX_FLUID, 2), all_ones);
        space.write_at(Addr::new(PREFIX_FLUID, 3), all_zeros);

        let bundled = bundle_recent(&space, PREFIX_FLUID, 4, 4);
        assert!(bundled.is_some());
        assert_eq!(bundled.unwrap(), all_ones); // majority wins
    }

    #[test]
    fn test_process_packet_roundtrip() {
        use ladybug_contract::container::Container;
        use ladybug_contract::wire::{self, CogPacket};

        let mut ck = CognitiveKernel::new();
        let mut space = BindSpace::new();
        let mut node = LayerNode::new("test");

        // Create a CogPacket with a random container payload
        let content = Container::random(42);
        let pkt = CogPacket::request(
            wire::wire_ops::EXECUTE,
            0x0C00, // Agent prefix
            0x8001, // Node zone target
            content,
        );

        let (result, response) = ck.process_packet(&mut space, &pkt, &mut node);
        assert_eq!(result.cycle, 1);
        assert!(response.is_response());
        assert!(!result.ops_performed.is_empty());
        // Response carries satisfaction scores
        let sat = response.satisfaction_array();
        assert!(sat.iter().any(|&s| s >= 0.0));
    }

    #[test]
    fn test_container_fingerprint_bridge() {
        use ladybug_contract::container::Container;

        let c = Container::random(99);
        let fp = container_to_fingerprint(&c);
        let c_back = fingerprint_to_container(fp.as_raw());
        // Lower 128 words should roundtrip exactly
        assert_eq!(c.words, c_back.words);
    }

    #[test]
    fn test_snapshot_to_fingerprint_deterministic() {
        let mut node = LayerNode::new("test");
        let input = Fingerprint::from_content("data");
        process_layers_wave(&mut node, &input, 1);
        let snap1 = snapshot_consciousness(&node, 1);
        let snap2 = snapshot_consciousness(&node, 1);

        let fp1 = snapshot_to_fp(&snap1);
        let fp2 = snapshot_to_fp(&snap2);
        assert_eq!(fp1.similarity(&fp2), 1.0);
    }
}
