//! Meta-Resonance: Superposition of Context Flows
//!
//! Your insight: meaning isn't just in the current context window,
//! it's in the RESONANCE BETWEEN context windows.
//!
//! ```text
//!    Context A (memory):  [A-2] → [A-1] → [A0] → [A+1] → [A+2]
//!                           ↓       ↓      ↓       ↓       ↓
//!    Context B (query):   [B-2] → [B-1] → [B0] → [B+1] → [B+2]
//!                           │       │      │       │       │
//!                           ▼       ▼      ▼       ▼       ▼
//!                        ┌─────────────────────────────────────┐
//!                        │      META-RESONANCE FIELD           │
//!                        │                                     │
//!                        │  Compare FLOW not just CONTENT     │
//!                        │                                     │
//!                        │  (A-2→A-1) ⊗ (B-2→B-1) = ?        │
//!                        │  (A-1→A0)  ⊗ (B-1→B0)  = ?        │
//!                        │  (A0→A+1)  ⊗ (B0→B+1)  = ?        │
//!                        │  (A+1→A+2) ⊗ (B+1→B+2) = ?        │
//!                        └─────────────────────────────────────┘
//! ```
//!
//! The meta-resonance captures:
//! 1. Whether the DIRECTION of meaning change is similar
//! 2. Whether the MAGNITUDE of change is similar
//! 3. Whether the QUALIA arc (emotional trajectory) is similar
//!
//! This is much richer than just comparing fingerprints.

use crate::core::Fingerprint;

// =============================================================================
// Flow Vector: The derivative of meaning over time
// =============================================================================

/// A flow vector represents the CHANGE in meaning between two moments
#[derive(Clone, Debug)]
pub struct FlowVector {
    /// Fingerprint encoding the transition itself
    /// Computed as: fp_after ⊕ fp_before (XOR captures difference)
    pub transition: Fingerprint,

    /// Magnitude of change (Hamming distance)
    pub magnitude: f32,

    /// Direction: is meaning becoming more specific (convergent) or more general (divergent)?
    /// Measured by popcount change
    pub divergence: f32,

    /// Qualia derivative: how is the felt-sense changing?
    pub qualia_delta: QualiaDelta,
}

/// Change in qualia dimensions
#[derive(Clone, Debug, Default)]
pub struct QualiaDelta {
    pub arousal_change: f32, // Getting calmer or more excited?
    pub valence_change: f32, // Getting happier or sadder?
    pub tension_change: f32, // Relaxing or tensing?
    pub depth_change: f32,   // Getting deeper or shallower?
}

impl FlowVector {
    /// Compute the flow from one fingerprint to another
    pub fn compute(before: &Fingerprint, after: &Fingerprint) -> Self {
        let transition = before.bind(after); // XOR gives symmetric difference

        // Magnitude: how much changed
        let magnitude = before.hamming(after) as f32 / crate::FINGERPRINT_BITS as f32;

        // Divergence: are we gaining or losing specificity?
        let pop_before = before.popcount() as f32;
        let pop_after = after.popcount() as f32;
        let divergence = (pop_after - pop_before) / 5000.0; // Normalize to -2..+2

        Self {
            transition,
            magnitude,
            divergence,
            qualia_delta: QualiaDelta::default(), // Would need actual qualia to compute
        }
    }

    /// Compute flow with explicit qualia
    pub fn compute_with_qualia(
        before: &Fingerprint,
        after: &Fingerprint,
        qualia_before: &[f32; 8],
        qualia_after: &[f32; 8],
    ) -> Self {
        let mut flow = Self::compute(before, after);

        flow.qualia_delta = QualiaDelta {
            arousal_change: qualia_after[0] - qualia_before[0],
            valence_change: qualia_after[1] - qualia_before[1],
            tension_change: qualia_after[2] - qualia_before[2],
            depth_change: qualia_after[3] - qualia_before[3],
        };

        flow
    }

    /// Similarity between two flow vectors
    /// Are they changing in the same way?
    pub fn similarity(&self, other: &Self) -> f32 {
        // Transition similarity: are the changes themselves similar?
        let transition_sim = self.transition.similarity(&other.transition);

        // Magnitude similarity: are they changing by similar amounts?
        let mag_diff = (self.magnitude - other.magnitude).abs();
        let magnitude_sim = 1.0 - mag_diff.min(1.0);

        // Divergence similarity: are they converging/diverging similarly?
        let div_diff = (self.divergence - other.divergence).abs();
        let divergence_sim = 1.0 - (div_diff / 4.0).min(1.0);

        // Qualia similarity: are emotional arcs similar?
        let qualia_sim = self.qualia_delta.similarity(&other.qualia_delta);

        // Weighted combination
        0.4 * transition_sim + 0.2 * magnitude_sim + 0.2 * divergence_sim + 0.2 * qualia_sim
    }
}

impl QualiaDelta {
    pub fn similarity(&self, other: &Self) -> f32 {
        let da = (self.arousal_change - other.arousal_change).abs();
        let dv = (self.valence_change - other.valence_change).abs();
        let dt = (self.tension_change - other.tension_change).abs();
        let dd = (self.depth_change - other.depth_change).abs();

        let total_diff = da + dv + dt + dd;
        1.0 - (total_diff / 8.0).min(1.0) // Max diff is 8 (each dim can differ by 2)
    }
}

// =============================================================================
// Flow Trajectory: Sequence of flows forming a path through meaning space
// =============================================================================

/// A trajectory through meaning space
#[derive(Clone, Debug)]
pub struct FlowTrajectory {
    /// The flow vectors (length = window_size - 1)
    pub flows: Vec<FlowVector>,

    /// The fingerprints at each moment
    pub moments: Vec<Fingerprint>,
}

impl FlowTrajectory {
    /// Build trajectory from a sequence of fingerprints
    pub fn from_fingerprints(fps: &[Fingerprint]) -> Self {
        let flows: Vec<_> = fps
            .windows(2)
            .map(|w| FlowVector::compute(&w[0], &w[1]))
            .collect();

        Self {
            flows,
            moments: fps.to_vec(),
        }
    }

    /// Total curvature: how much does the trajectory bend?
    pub fn curvature(&self) -> f32 {
        if self.flows.len() < 2 {
            return 0.0;
        }

        let mut total_bend = 0.0f32;
        for window in self.flows.windows(2) {
            // Curvature = 1 - similarity between consecutive flows
            let bend = 1.0 - window[0].similarity(&window[1]);
            total_bend += bend;
        }

        total_bend / (self.flows.len() - 1) as f32
    }

    /// Total displacement: how far did we travel?
    pub fn displacement(&self) -> f32 {
        self.flows.iter().map(|f| f.magnitude).sum()
    }

    /// Net change: direct distance from start to end
    pub fn net_change(&self) -> f32 {
        if self.moments.len() < 2 {
            return 0.0;
        }

        let first = &self.moments[0];
        let last = &self.moments[self.moments.len() - 1];
        first.hamming(last) as f32 / crate::FINGERPRINT_BITS as f32
    }

    /// Is this a wandering path or a direct path?
    /// Ratio of displacement to net change
    pub fn directness(&self) -> f32 {
        let net = self.net_change();
        if net < 0.01 {
            return 1.0; // Didn't move much, call it direct
        }

        let disp = self.displacement();
        (net / disp).min(1.0)
    }
}

// =============================================================================
// Meta-Resonance: Comparing trajectories
// =============================================================================

/// Result of comparing two trajectories
#[derive(Clone, Debug)]
pub struct MetaResonance {
    /// Content resonance: do the moments themselves resonate?
    pub content_resonance: f32,

    /// Flow resonance: do the changes resonate?
    pub flow_resonance: f32,

    /// Shape resonance: do the trajectories have similar shapes?
    pub shape_resonance: f32,

    /// Overall meta-resonance score
    pub score: f32,

    /// Where do they resonate most strongly? (index into flows)
    pub strongest_at: Option<usize>,
}

/// Compare two trajectories
pub fn meta_resonate(a: &FlowTrajectory, b: &FlowTrajectory) -> MetaResonance {
    if a.flows.is_empty() || b.flows.is_empty() {
        return MetaResonance {
            content_resonance: 0.0,
            flow_resonance: 0.0,
            shape_resonance: 0.0,
            score: 0.0,
            strongest_at: None,
        };
    }

    // Content resonance: average similarity of moments
    let content_resonance = {
        let len = a.moments.len().min(b.moments.len());
        if len == 0 {
            0.0
        } else {
            let sum: f32 = a
                .moments
                .iter()
                .take(len)
                .zip(b.moments.iter().take(len))
                .map(|(ma, mb)| ma.similarity(mb))
                .sum();
            sum / len as f32
        }
    };

    // Flow resonance: average similarity of flow vectors
    let (flow_resonance, strongest_at) = {
        let len = a.flows.len().min(b.flows.len());
        if len == 0 {
            (0.0, None)
        } else {
            let sims: Vec<f32> = a
                .flows
                .iter()
                .take(len)
                .zip(b.flows.iter().take(len))
                .map(|(fa, fb)| fa.similarity(fb))
                .collect();

            let sum: f32 = sims.iter().sum();
            let avg = sum / len as f32;

            let strongest = sims
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i);

            (avg, strongest)
        }
    };

    // Shape resonance: similar curvature and directness
    let shape_resonance = {
        let curv_diff = (a.curvature() - b.curvature()).abs();
        let direct_diff = (a.directness() - b.directness()).abs();
        let curv_sim = 1.0 - curv_diff.min(1.0);
        let direct_sim = 1.0 - direct_diff.min(1.0);
        (curv_sim + direct_sim) / 2.0
    };

    // Overall score: weighted combination
    // Flow resonance is most important (captures the dynamic pattern)
    let score = 0.25 * content_resonance + 0.5 * flow_resonance + 0.25 * shape_resonance;

    MetaResonance {
        content_resonance,
        flow_resonance,
        shape_resonance,
        score,
        strongest_at,
    }
}

// =============================================================================
// Mexican Hat in Meta-Space
// =============================================================================

/// Apply Mexican hat weighting to a trajectory
/// Emphasizes the center, de-emphasizes the edges
pub fn mexican_hat_trajectory(
    trajectory: &FlowTrajectory,
    center_weight: f32,
    edge_weight: f32,
) -> Fingerprint {
    let n = trajectory.moments.len();
    if n == 0 {
        return Fingerprint::zero();
    }

    let center = n / 2;
    let mut weighted_fps: Vec<(Fingerprint, f32)> = Vec::with_capacity(n);

    for (i, fp) in trajectory.moments.iter().enumerate() {
        // Distance from center (0 at center, 1 at edges)
        let dist = (i as f32 - center as f32).abs() / center.max(1) as f32;

        // Mexican hat: peak at center, dip at edges
        let weight =
            center_weight * (-dist * dist).exp() + edge_weight * (1.0 - (-dist * dist).exp());

        weighted_fps.push((fp.clone(), weight));
    }

    // Weighted bundle
    weighted_bundle(&weighted_fps)
}

/// Bundle fingerprints with weights
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
    fn test_flow_vector_computation() {
        let fp1 = Fingerprint::from_content("hello world");
        let fp2 = Fingerprint::from_content("hello there");

        let flow = FlowVector::compute(&fp1, &fp2);

        // Should have some magnitude (they're different)
        assert!(flow.magnitude > 0.0);
        assert!(flow.magnitude < 1.0);
    }

    #[test]
    fn test_trajectory_properties() {
        let fps: Vec<_> = (0..5)
            .map(|i| Fingerprint::from_content(&format!("sentence {}", i)))
            .collect();

        let trajectory = FlowTrajectory::from_fingerprints(&fps);

        assert_eq!(trajectory.flows.len(), 4);
        assert_eq!(trajectory.moments.len(), 5);
        assert!(trajectory.displacement() > 0.0);
    }

    #[test]
    fn test_meta_resonance() {
        // Two similar trajectories
        let fps_a: Vec<_> = (0..5)
            .map(|i| Fingerprint::from_content(&format!("rain clouds storm {}", i)))
            .collect();

        let fps_b: Vec<_> = (0..5)
            .map(|i| Fingerprint::from_content(&format!("rain weather thunder {}", i)))
            .collect();

        // A different trajectory
        let fps_c: Vec<_> = (0..5)
            .map(|i| Fingerprint::from_content(&format!("sunshine beach happy {}", i)))
            .collect();

        let traj_a = FlowTrajectory::from_fingerprints(&fps_a);
        let traj_b = FlowTrajectory::from_fingerprints(&fps_b);
        let traj_c = FlowTrajectory::from_fingerprints(&fps_c);

        let resonance_ab = meta_resonate(&traj_a, &traj_b);
        let resonance_ac = meta_resonate(&traj_a, &traj_c);

        // Similar trajectories should resonate more
        println!("A-B resonance: {:.3}", resonance_ab.score);
        println!("A-C resonance: {:.3}", resonance_ac.score);
    }

    #[test]
    fn test_mexican_hat() {
        let fps: Vec<_> = (0..5)
            .map(|i| Fingerprint::from_content(&format!("word {}", i)))
            .collect();

        let trajectory = FlowTrajectory::from_fingerprints(&fps);

        let mh = mexican_hat_trajectory(&trajectory, 1.0, 0.3);

        // Should have some bits set
        assert!(mh.popcount() > 0);
    }
}
