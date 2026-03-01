//! Gestalt Integration Module: Bundling Detection, Tilt Correction, Truth Trajectories
//!
//! Bridges rustynum PRs #74-81 types into ladybug-rs SPO architecture:
//!
//! - **Bundling Detection**: Cross-plane vote analysis detects when two CLAM branches
//!   share 2-of-3 SPO planes (SO/SP/PO halos) → proposes branch merges.
//!
//! - **Tilt Correction**: Per-plane σ calibration detects skewed stripe distributions
//!   and recalibrates SigmaGate thresholds per axis.
//!
//! - **Truth Trajectory**: Temporal tracking of NARS truth values across evidence events,
//!   mapping CollapseGate decisions to tentative/committed/rejected states.
//!
//! - **Gestalt Change Classification**: Crystallizing (GREEN), contested (AMBER),
//!   dissolving (RED), epiphany (BLUE) — from CausalSaliency per-plane analysis.
//!
//! # Architecture
//!
//! ```text
//! rustynum-core types:
//!   SigmaGate, SignificanceLevel, SigmaScore, EnergyConflict
//!   NarsTruthValue, CausalityDirection, CausalityDecomposition
//!   CollapseGate, LayerStack, SpatialCrystal3D, SpatialDistances
//!   QualiaGateLevel (Flow/Hold/Block)
//!
//! rustynum-bnn types:
//!   CrossPlaneVote, HaloType, HaloDistribution
//!   CausalSaliency (crystallizing/dissolving/contested bitmasks)
//!   CausalTrajectory, ResonatorSnapshot, RifDiff
//!   NarsTruth (bnn's version), CausalArrow
//!
//! This module CONSUMES these types without modification.
//! ```

// =============================================================================
// GESTALT CHANGE CLASSIFICATION
// =============================================================================

/// Gestalt state of an SPO edge or bundle, derived from CausalSaliency.
///
/// ```text
/// GREEN GLOW   → truth crystallizing (confidence rising, planes converging)
/// AMBER PULSE  → contested (planes disagree, moderator variable suspected)
/// RED DIM      → truth dissolving (confidence falling, counter-evidence arriving)
/// BLUE SHIMMER → epiphany proposed (new bundling candidate detected)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GestaltState {
    /// Crystallizing: evidence accumulating, planes converging.
    /// CausalSaliency shows crystallizing_count >> dissolving_count.
    Crystallizing,

    /// Contested: planes disagree — one plane's stripes migrated while others stable.
    /// CausalSaliency shows contested_count > threshold in at least one plane.
    Contested,

    /// Dissolving: counter-evidence arriving, confidence dropping.
    /// CausalSaliency shows dissolving_count >> crystallizing_count.
    Dissolving,

    /// Epiphany: new cross-plane evidence suggests bundling or reclassification.
    /// Triggered when a BundlingProposal is first created.
    Epiphany,
}

impl GestaltState {
    /// Derive gestalt state from per-plane crystallizing/dissolving/contested counts.
    ///
    /// Uses the CausalSaliency per-plane counts from rustynum-bnn:
    ///   crystallizing_count: [u32; 3] — per S/P/O plane
    ///   dissolving_count: [u32; 3]
    ///   contested_count: [u32; 3]
    pub fn from_saliency_counts(
        crystallizing: &[u32; 3],
        dissolving: &[u32; 3],
        contested: &[u32; 3],
    ) -> Self {
        let total_cryst: u32 = crystallizing.iter().sum();
        let total_dissolve: u32 = dissolving.iter().sum();
        let total_contest: u32 = contested.iter().sum();

        let total = total_cryst + total_dissolve + total_contest;
        if total == 0 {
            return GestaltState::Crystallizing; // no activity = stable
        }

        // Contested dominates when any single plane has high contest ratio
        let max_contest = *contested.iter().max().unwrap_or(&0);
        let max_per_plane = total / 3;
        if max_per_plane > 0 && max_contest > max_per_plane / 2 {
            return GestaltState::Contested;
        }

        // Compare crystallizing vs dissolving
        if total_cryst > total_dissolve * 2 {
            GestaltState::Crystallizing
        } else if total_dissolve > total_cryst * 2 {
            GestaltState::Dissolving
        } else {
            GestaltState::Contested
        }
    }
}

// =============================================================================
// BUNDLING TYPES
// =============================================================================

/// Which SPO plane is the source of disagreement in a bundling proposal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContestedPlane {
    /// Subjects differ, predicates and objects agree (PO-type halo).
    Subject,
    /// Predicates differ, subjects and objects agree (SO-type halo).
    Predicate,
    /// Objects differ, subjects and predicates agree (SP-type halo).
    Object,
}

/// Type of bundling event, derived from which planes agree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BundlingType {
    /// SO-type: same entities, opposite predicates → predicate inversion.
    /// "letting_go and holding_on share actors and targets."
    PredicateInversion,

    /// PO-type: same actions on same targets, different agents → agent convergence.
    /// "Different tribes, same journey."
    AgentConvergence,

    /// SP-type: same agents doing same things, different targets → target divergence.
    /// "Same actors, same actions, landed on different targets."
    TargetDivergence,
}

impl BundlingType {
    /// Which plane is contested for this bundling type.
    pub fn contested_plane(&self) -> ContestedPlane {
        match self {
            BundlingType::PredicateInversion => ContestedPlane::Predicate,
            BundlingType::AgentConvergence => ContestedPlane::Subject,
            BundlingType::TargetDivergence => ContestedPlane::Object,
        }
    }
}

/// A proposal to bundle two CLAM branches.
///
/// Maps to the three-state tentative/committed/rejected model:
/// - Tentative → CollapseGate::Hold (visible, searchable, not committed)
/// - Committed → CollapseGate::Flow (approved, tree rewritten)
/// - Rejected → CollapseGate::Block (declined, kept in audit trail)
#[derive(Debug, Clone)]
pub struct BundlingProposal {
    /// Identifiers of the two branches proposed for bundling.
    pub branch_a: String,
    pub branch_b: String,

    /// Type of bundling detected.
    pub bundling_type: BundlingType,

    /// Per-plane Hamming distances between branch centers.
    pub s_distance: u32,
    pub p_distance: u32,
    pub o_distance: u32,

    /// NARS truth value: frequency × confidence of the bundling evidence.
    pub nars_frequency: f32,
    pub nars_confidence: f32,

    /// σ-significance level of the cross-plane evidence.
    pub significance: rustynum_core::SignificanceLevel,

    /// Number of cross-matches supporting the bundling.
    pub evidence_count: u32,

    /// Current collapse gate state (tentative lifecycle).
    pub gate: rustynum_core::CollapseGate,

    /// Timestamp of proposal creation (Unix millis).
    pub proposed_at_ms: u64,

    /// Optional: who approved/rejected and why.
    pub review: Option<BundlingReview>,
}

/// Review decision on a bundling proposal.
#[derive(Debug, Clone)]
pub struct BundlingReview {
    /// Who made the decision.
    pub reviewer: String,
    /// When the decision was made (Unix millis).
    pub reviewed_at_ms: u64,
    /// The decision: Flow (approve), Block (reject).
    pub decision: rustynum_core::CollapseGate,
    /// Reason text from the reviewer.
    pub reason: String,
    /// Machine confidence at time of review (may have changed since proposal).
    pub auto_confidence_at_review: f32,
}

impl BundlingProposal {
    /// Create a new tentative proposal (CollapseGate::Hold).
    pub fn new_tentative(
        branch_a: String,
        branch_b: String,
        bundling_type: BundlingType,
        s_distance: u32,
        p_distance: u32,
        o_distance: u32,
        nars_frequency: f32,
        nars_confidence: f32,
        significance: rustynum_core::SignificanceLevel,
        evidence_count: u32,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            branch_a,
            branch_b,
            bundling_type,
            s_distance,
            p_distance,
            o_distance,
            nars_frequency,
            nars_confidence,
            significance,
            evidence_count,
            gate: rustynum_core::CollapseGate::Hold,
            proposed_at_ms: now,
            review: None,
        }
    }

    /// Whether this proposal is still tentative (pending review).
    pub fn is_tentative(&self) -> bool {
        matches!(self.gate, rustynum_core::CollapseGate::Hold)
    }

    /// Whether this proposal was committed (approved).
    pub fn is_committed(&self) -> bool {
        matches!(self.gate, rustynum_core::CollapseGate::Flow)
    }

    /// Whether this proposal was rejected.
    pub fn is_rejected(&self) -> bool {
        matches!(self.gate, rustynum_core::CollapseGate::Block)
    }

    /// Approve the bundling proposal (Flow).
    pub fn approve(&mut self, reviewer: String, reason: String) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        self.gate = rustynum_core::CollapseGate::Flow;
        self.review = Some(BundlingReview {
            reviewer,
            reviewed_at_ms: now,
            decision: rustynum_core::CollapseGate::Flow,
            reason,
            auto_confidence_at_review: self.nars_confidence,
        });
    }

    /// Reject the bundling proposal (Block).
    pub fn reject(&mut self, reviewer: String, reason: String) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        self.gate = rustynum_core::CollapseGate::Block;
        self.review = Some(BundlingReview {
            reviewer,
            reviewed_at_ms: now,
            decision: rustynum_core::CollapseGate::Block,
            reason,
            auto_confidence_at_review: self.nars_confidence,
        });
    }

    /// Update evidence (tentative proposals accumulate evidence while waiting).
    pub fn update_evidence(&mut self, new_frequency: f32, new_confidence: f32, new_count: u32) {
        if self.is_tentative() {
            self.nars_frequency = new_frequency;
            self.nars_confidence = new_confidence;
            self.evidence_count = new_count;
        }
    }
}

// =============================================================================
// BUNDLING DETECTION
// =============================================================================

/// Detect whether two branch centers are bundling candidates.
///
/// Uses per-plane Hamming distance analysis. If 2-of-3 planes are close
/// (below `agreement_threshold`) and 1 plane is distant, that's a bundling signal.
///
/// The halo type determines the bundling type:
/// - SO (S and O close, P far) → PredicateInversion
/// - PO (P and O close, S far) → AgentConvergence
/// - SP (S and P close, O far) → TargetDivergence
///
/// `gate` provides the σ-thresholds for "close" (Evidence level) and "far" (Noise level).
pub fn detect_bundling(
    center_a_s: &[u64],
    center_a_p: &[u64],
    center_a_o: &[u64],
    center_b_s: &[u64],
    center_b_p: &[u64],
    center_b_o: &[u64],
    gate: &rustynum_core::SigmaGate,
) -> Option<(BundlingType, u32, u32, u32)> {
    // Per-plane Hamming distance
    let s_dist = crate::core::rustynum_accel::slice_hamming(center_a_s, center_b_s) as u32;
    let p_dist = crate::core::rustynum_accel::slice_hamming(center_a_p, center_b_p) as u32;
    let o_dist = crate::core::rustynum_accel::slice_hamming(center_a_o, center_b_o) as u32;

    // "Close" = Evidence level or better (2σ below noise)
    let close = gate.evidence;
    // "Far" = at or above Hint level (in the noise region)
    let far = gate.hint;

    let s_close = s_dist < close;
    let p_close = p_dist < close;
    let o_close = o_dist < close;
    let s_far = s_dist >= far;
    let p_far = p_dist >= far;
    let o_far = o_dist >= far;

    // Exactly 2 close + 1 far → bundling candidate
    if s_close && o_close && p_far {
        Some((BundlingType::PredicateInversion, s_dist, p_dist, o_dist))
    } else if p_close && o_close && s_far {
        Some((BundlingType::AgentConvergence, s_dist, p_dist, o_dist))
    } else if s_close && p_close && o_far {
        Some((BundlingType::TargetDivergence, s_dist, p_dist, o_dist))
    } else {
        None
    }
}

// =============================================================================
// TILT DETECTION AND CORRECTION
// =============================================================================

/// Per-plane σ calibration report.
///
/// When one plane's σ diverges significantly from the others,
/// the data is "tilted" in that semantic dimension. The tilt angle
/// tells you which dimension needs recalibration.
#[derive(Debug, Clone, Copy)]
pub struct TiltReport {
    /// S-plane tilt: positive = S is looser than average.
    pub s_tilt: f32,
    /// P-plane tilt: positive = P is looser than average.
    pub p_tilt: f32,
    /// O-plane tilt: positive = O is looser than average.
    pub o_tilt: f32,
    /// Total tilt magnitude (L2 norm of per-plane tilts).
    pub total_tilt: f32,
}

impl TiltReport {
    /// Detect tilt from per-plane standard deviations.
    ///
    /// Each σ value is the standard deviation of Hamming distances within that plane.
    /// Balanced planes have similar σ values. Skewed planes have outlier σ.
    pub fn from_plane_sigmas(s_sigma: f32, p_sigma: f32, o_sigma: f32) -> Self {
        let mean = (s_sigma + p_sigma + o_sigma) / 3.0;
        let s_tilt = s_sigma - mean;
        let p_tilt = p_sigma - mean;
        let o_tilt = o_sigma - mean;
        let total_tilt = (s_tilt * s_tilt + p_tilt * p_tilt + o_tilt * o_tilt).sqrt();

        Self {
            s_tilt,
            p_tilt,
            o_tilt,
            total_tilt,
        }
    }

    /// Whether the data is significantly tilted (any plane > 1σ from mean).
    pub fn is_tilted(&self, threshold: f32) -> bool {
        self.s_tilt.abs() > threshold
            || self.p_tilt.abs() > threshold
            || self.o_tilt.abs() > threshold
    }

    /// Which plane is most tilted.
    pub fn most_tilted_plane(&self) -> ContestedPlane {
        let abs_s = self.s_tilt.abs();
        let abs_p = self.p_tilt.abs();
        let abs_o = self.o_tilt.abs();

        if abs_s >= abs_p && abs_s >= abs_o {
            ContestedPlane::Subject
        } else if abs_p >= abs_s && abs_p >= abs_o {
            ContestedPlane::Predicate
        } else {
            ContestedPlane::Object
        }
    }
}

/// Per-plane SigmaGate calibration — the "rotation correction."
///
/// Each SPO plane gets its OWN σ-thresholds calibrated to its actual distribution,
/// instead of a shared global SigmaGate. This corrects for data arriving "tilted"
/// (e.g., predicates dispersed while entities are tight).
#[derive(Debug, Clone)]
pub struct PlaneCalibration {
    /// S-plane σ thresholds (calibrated to S⊕P distribution).
    pub s_gate: rustynum_core::SigmaGate,
    /// P-plane σ thresholds (calibrated to P⊕O distribution).
    pub p_gate: rustynum_core::SigmaGate,
    /// O-plane σ thresholds (calibrated to S⊕O distribution).
    pub o_gate: rustynum_core::SigmaGate,
}

impl PlaneCalibration {
    /// Create from a single shared gate (no tilt correction).
    pub fn uniform(gate: rustynum_core::SigmaGate) -> Self {
        Self {
            s_gate: gate,
            p_gate: gate,
            o_gate: gate,
        }
    }

    /// Create with per-plane calibration from observed μ and σ values.
    ///
    /// Each plane's gate is derived from its own mean distance (μ) and
    /// standard deviation (σ) rather than the global 16K-bit assumption.
    pub fn from_plane_stats(s_mu: u32, s_sigma: u32, p_mu: u32, p_sigma: u32, o_mu: u32, o_sigma: u32) -> Self {
        Self {
            s_gate: rustynum_core::SigmaGate::custom(s_mu, s_sigma),
            p_gate: rustynum_core::SigmaGate::custom(p_mu, p_sigma),
            o_gate: rustynum_core::SigmaGate::custom(o_mu, o_sigma),
        }
    }

    /// Compute tilt report from current calibration.
    pub fn tilt(&self) -> TiltReport {
        TiltReport::from_plane_sigmas(
            self.s_gate.sigma_unit as f32,
            self.p_gate.sigma_unit as f32,
            self.o_gate.sigma_unit as f32,
        )
    }

    /// Classify a distance on a specific plane using that plane's calibrated gate.
    pub fn classify_plane(
        &self,
        plane: ContestedPlane,
        distance: u32,
    ) -> rustynum_core::SignificanceLevel {
        let gate = match plane {
            ContestedPlane::Subject => &self.s_gate,
            ContestedPlane::Predicate => &self.p_gate,
            ContestedPlane::Object => &self.o_gate,
        };
        crate::search::hdr_cascade::classify_sigma(distance, gate)
    }
}

// =============================================================================
// TRUTH TRAJECTORY
// =============================================================================

/// A single evidence event in a truth trajectory.
#[derive(Debug, Clone)]
pub struct EvidenceEvent {
    /// Timestamp (Unix millis).
    pub timestamp_ms: u64,
    /// What happened: new match, counter-evidence, reviewer action, etc.
    pub event_type: EvidenceEventType,
    /// NARS truth value AFTER this event.
    pub nars_frequency: f32,
    pub nars_confidence: f32,
    /// σ-significance at this moment.
    pub significance: rustynum_core::SignificanceLevel,
    /// Evidence count at this moment.
    pub evidence_count: u32,
    /// Per-plane gestalt state at this moment.
    pub gestalt: GestaltState,
}

/// Type of evidence event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceEventType {
    /// New cross-matches found (evidence for bundling).
    MatchesAdded(u32),
    /// Counter-examples found (evidence against bundling).
    CounterEvidence(u32),
    /// Reviewer approved the proposal.
    ReviewApproved,
    /// Reviewer rejected the proposal.
    ReviewRejected,
    /// Reviewer requested more evidence (biases future searches).
    MoreEvidenceRequested,
    /// σ-stripe migration detected (Schaltsekunde triggered).
    StripeMigration,
}

/// Temporal trajectory of a truth value's evolution.
///
/// Records every evidence event from proposal to decision, enabling
/// forward/backward playback and audit trail visualization.
#[derive(Debug, Clone)]
pub struct TruthTrajectory {
    /// Unique identifier for this trajectory.
    pub trajectory_id: String,
    /// The bundling proposal this trajectory tracks.
    pub proposal: BundlingProposal,
    /// Ordered sequence of evidence events.
    pub events: Vec<EvidenceEvent>,
}

impl TruthTrajectory {
    /// Create a new trajectory for a bundling proposal.
    pub fn new(proposal: BundlingProposal) -> Self {
        let trajectory_id = format!(
            "{}_{}_{}",
            proposal.branch_a, proposal.branch_b, proposal.proposed_at_ms
        );

        // Record the initial proposal as the first event
        let initial = EvidenceEvent {
            timestamp_ms: proposal.proposed_at_ms,
            event_type: EvidenceEventType::MatchesAdded(proposal.evidence_count),
            nars_frequency: proposal.nars_frequency,
            nars_confidence: proposal.nars_confidence,
            significance: proposal.significance,
            evidence_count: proposal.evidence_count,
            gestalt: GestaltState::Epiphany,
        };

        Self {
            trajectory_id,
            proposal,
            events: vec![initial],
        }
    }

    /// Record a new evidence event.
    pub fn record_event(&mut self, event: EvidenceEvent) {
        // Update the proposal's evidence if it's still tentative
        self.proposal.update_evidence(
            event.nars_frequency,
            event.nars_confidence,
            event.evidence_count,
        );
        self.events.push(event);
    }

    /// Current NARS truth value (from the most recent event).
    pub fn current_truth(&self) -> (f32, f32) {
        self.events
            .last()
            .map(|e| (e.nars_frequency, e.nars_confidence))
            .unwrap_or((0.5, 0.0))
    }

    /// Current gestalt state.
    pub fn current_gestalt(&self) -> GestaltState {
        self.events
            .last()
            .map(|e| e.gestalt)
            .unwrap_or(GestaltState::Crystallizing)
    }

    /// Confidence trend: positive = rising, negative = falling.
    pub fn confidence_trend(&self) -> f32 {
        if self.events.len() < 2 {
            return 0.0;
        }
        let recent = &self.events[self.events.len() - 1];
        let previous = &self.events[self.events.len() - 2];
        recent.nars_confidence - previous.nars_confidence
    }

    /// Number of evidence events recorded.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }
}

// =============================================================================
// COLLAPSE MODE (auto/semi-auto/manual threshold)
// =============================================================================

/// Operational mode for bundling decisions.
///
/// Maps directly to CollapseGate auto-FLOW threshold:
/// - Research: auto-bundle above 0.95 confidence
/// - Production: propose above 0.80, require human review
/// - Regulated: propose at any level, always require human gut commit
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollapseMode {
    /// Fully automatic: auto-bundle when confidence > 0.95.
    /// Audit trail records "auto-approved".
    Research,
    /// Semi-automatic: propose when confidence > 0.80, human approves.
    /// Audit trail records both machine and human confidence.
    Production,
    /// Fully manual: propose at ANY confidence, always require human review.
    /// Nothing changes without a gut commit.
    Regulated,
}

impl CollapseMode {
    /// Minimum confidence threshold for auto-approval.
    pub fn auto_threshold(&self) -> f32 {
        match self {
            CollapseMode::Research => 0.95,
            CollapseMode::Production => f32::INFINITY, // never auto-approve
            CollapseMode::Regulated => f32::INFINITY,  // never auto-approve
        }
    }

    /// Minimum confidence threshold for proposal creation.
    pub fn proposal_threshold(&self) -> f32 {
        match self {
            CollapseMode::Research => 0.80,
            CollapseMode::Production => 0.80,
            CollapseMode::Regulated => 0.0, // propose at any confidence
        }
    }

    /// Decide the CollapseGate for a given confidence level.
    pub fn decide(&self, confidence: f32) -> rustynum_core::CollapseGate {
        if confidence >= self.auto_threshold() {
            rustynum_core::CollapseGate::Flow // auto-approve
        } else if confidence >= self.proposal_threshold() {
            rustynum_core::CollapseGate::Hold // tentative, awaiting review
        } else {
            match self {
                CollapseMode::Regulated => rustynum_core::CollapseGate::Hold,
                _ => rustynum_core::CollapseGate::Block, // below proposal threshold
            }
        }
    }
}

// =============================================================================
// ANTIALIASED SIGMA SCORING
// =============================================================================

/// Antialiased σ-band assignment: items near band boundaries get weighted membership.
///
/// Instead of hard "this is 2σ" vs "this is 2.5σ", provides continuous position
/// with soft membership in adjacent bands. This is the "rotation antialiasing" that
/// prevents jagged artifacts when per-plane σ calibration shifts band boundaries.
#[derive(Debug, Clone, Copy)]
pub struct AntialiasedSigma {
    /// Primary significance band.
    pub primary: rustynum_core::SignificanceLevel,
    /// Adjacent significance band (for boundary items).
    pub secondary: rustynum_core::SignificanceLevel,
    /// Weight of primary band (0.0..1.0).
    pub primary_weight: f32,
    /// Weight of secondary band (0.0..1.0, = 1 - primary_weight).
    pub secondary_weight: f32,
    /// Continuous σ position (higher = closer to noise floor = weaker match).
    pub continuous_sigma: f32,
}

impl AntialiasedSigma {
    /// Compute antialiased sigma from raw distance and gate.
    ///
    /// The continuous sigma position is interpolated between band boundaries,
    /// and weights reflect how close the distance is to each boundary.
    pub fn from_distance(distance: u32, gate: &rustynum_core::SigmaGate) -> Self {
        // Continuous sigma: how many σ below the noise floor
        let dist_f = distance as f32;
        let mu_f = gate.mu as f32;
        let sigma_f = gate.sigma_unit as f32;

        let continuous_sigma = if sigma_f > 0.0 {
            (mu_f - dist_f) / sigma_f
        } else {
            0.0
        };

        // Determine primary and secondary bands with weights
        let (primary, secondary, primary_weight) = if distance < gate.discovery {
            // Deep in Discovery zone
            (
                rustynum_core::SignificanceLevel::Discovery,
                rustynum_core::SignificanceLevel::Strong,
                1.0_f32,
            )
        } else if distance < gate.strong {
            // Between Discovery and Strong
            let range = (gate.strong - gate.discovery) as f32;
            let pos = (distance - gate.discovery) as f32;
            let w = 1.0 - (pos / range);
            (
                rustynum_core::SignificanceLevel::Discovery,
                rustynum_core::SignificanceLevel::Strong,
                w,
            )
        } else if distance < gate.evidence {
            let range = (gate.evidence - gate.strong) as f32;
            let pos = (distance - gate.strong) as f32;
            let w = 1.0 - (pos / range);
            (
                rustynum_core::SignificanceLevel::Strong,
                rustynum_core::SignificanceLevel::Evidence,
                w,
            )
        } else if distance < gate.hint {
            let range = (gate.hint - gate.evidence) as f32;
            let pos = (distance - gate.evidence) as f32;
            let w = 1.0 - (pos / range);
            (
                rustynum_core::SignificanceLevel::Evidence,
                rustynum_core::SignificanceLevel::Hint,
                w,
            )
        } else {
            (
                rustynum_core::SignificanceLevel::Noise,
                rustynum_core::SignificanceLevel::Noise,
                1.0,
            )
        };

        Self {
            primary,
            secondary,
            primary_weight,
            secondary_weight: 1.0 - primary_weight,
            continuous_sigma,
        }
    }

    /// NARS confidence derived from continuous sigma position.
    /// Higher sigma = higher confidence. Maps to [0, 1) range.
    pub fn to_nars_confidence(&self) -> f32 {
        // Sigmoid mapping: σ=0 → c≈0.5, σ=3 → c≈0.95, σ=5 → c≈0.99
        let x = self.continuous_sigma;
        if x <= 0.0 {
            0.0
        } else {
            1.0 - 1.0 / (1.0 + x * x / 4.0)
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gestalt_classification() {
        // Strongly crystallizing
        let cryst = [100, 80, 90];
        let dissolve = [10, 5, 8];
        let contest = [5, 3, 4];
        assert_eq!(
            GestaltState::from_saliency_counts(&cryst, &dissolve, &contest),
            GestaltState::Crystallizing
        );

        // Strongly dissolving
        let cryst = [5, 3, 4];
        let dissolve = [100, 80, 90];
        let contest = [5, 3, 4];
        assert_eq!(
            GestaltState::from_saliency_counts(&cryst, &dissolve, &contest),
            GestaltState::Dissolving
        );

        // Contested (high contest in one plane)
        let cryst = [50, 50, 50];
        let dissolve = [50, 50, 50];
        let contest = [10, 10, 200]; // O-plane severely contested
        assert_eq!(
            GestaltState::from_saliency_counts(&cryst, &dissolve, &contest),
            GestaltState::Contested
        );
    }

    #[test]
    fn test_bundling_type_contested_plane() {
        assert_eq!(
            BundlingType::PredicateInversion.contested_plane(),
            ContestedPlane::Predicate
        );
        assert_eq!(
            BundlingType::AgentConvergence.contested_plane(),
            ContestedPlane::Subject
        );
        assert_eq!(
            BundlingType::TargetDivergence.contested_plane(),
            ContestedPlane::Object
        );
    }

    #[test]
    fn test_proposal_lifecycle() {
        let mut proposal = BundlingProposal::new_tentative(
            "1010:1100".to_string(),
            "1010:1101".to_string(),
            BundlingType::PredicateInversion,
            200,  // s_dist: close
            7500, // p_dist: far
            300,  // o_dist: close
            0.78,
            0.87,
            rustynum_core::SignificanceLevel::Strong,
            500,
        );

        assert!(proposal.is_tentative());
        assert!(!proposal.is_committed());
        assert!(!proposal.is_rejected());

        // Evidence accumulates while tentative
        proposal.update_evidence(0.83, 0.91, 612);
        assert_eq!(proposal.evidence_count, 612);
        assert!(proposal.is_tentative());

        // Approve
        proposal.approve(
            "jan.huebener".to_string(),
            "Domain expertise confirms predicate inversion".to_string(),
        );
        assert!(proposal.is_committed());
        assert!(!proposal.is_tentative());
        assert!(proposal.review.is_some());
    }

    #[test]
    fn test_tilt_detection() {
        // Balanced: no tilt
        let tilt = TiltReport::from_plane_sigmas(64.0, 64.0, 64.0);
        assert!(!tilt.is_tilted(10.0));
        assert!(tilt.total_tilt < 0.01);

        // Tilted: P-plane dispersed
        let tilt = TiltReport::from_plane_sigmas(64.0, 890.0, 64.0);
        assert!(tilt.is_tilted(10.0));
        assert_eq!(tilt.most_tilted_plane(), ContestedPlane::Predicate);
        assert!(tilt.p_tilt > 0.0); // P is looser than average
    }

    #[test]
    fn test_collapse_mode() {
        // Research mode: auto-approve above 0.95
        assert_eq!(
            CollapseMode::Research.decide(0.97),
            rustynum_core::CollapseGate::Flow
        );
        assert_eq!(
            CollapseMode::Research.decide(0.85),
            rustynum_core::CollapseGate::Hold
        );
        assert_eq!(
            CollapseMode::Research.decide(0.50),
            rustynum_core::CollapseGate::Block
        );

        // Production mode: never auto-approve, propose above 0.80
        assert_eq!(
            CollapseMode::Production.decide(0.99),
            rustynum_core::CollapseGate::Hold
        );
        assert_eq!(
            CollapseMode::Production.decide(0.50),
            rustynum_core::CollapseGate::Block
        );

        // Regulated mode: always Hold (propose at any confidence)
        assert_eq!(
            CollapseMode::Regulated.decide(0.10),
            rustynum_core::CollapseGate::Hold
        );
    }

    #[test]
    fn test_antialiased_sigma() {
        let gate = rustynum_core::SigmaGate::sku_16k();

        // Deep discovery: should be firmly in Discovery band
        let aa = AntialiasedSigma::from_distance(100, &gate);
        assert_eq!(aa.primary, rustynum_core::SignificanceLevel::Discovery);
        assert!(aa.primary_weight > 0.9);
        assert!(aa.continuous_sigma > 3.0);

        // Deep noise: should be firmly Noise
        let aa = AntialiasedSigma::from_distance(gate.mu + 100, &gate);
        assert_eq!(aa.primary, rustynum_core::SignificanceLevel::Noise);

        // NARS confidence from sigma
        let high_sigma = AntialiasedSigma::from_distance(100, &gate);
        let low_sigma = AntialiasedSigma::from_distance(gate.hint - 1, &gate);
        assert!(high_sigma.to_nars_confidence() > low_sigma.to_nars_confidence());
    }

    #[test]
    fn test_truth_trajectory() {
        let proposal = BundlingProposal::new_tentative(
            "a".to_string(),
            "b".to_string(),
            BundlingType::PredicateInversion,
            200,
            7500,
            300,
            0.78,
            0.87,
            rustynum_core::SignificanceLevel::Strong,
            500,
        );

        let mut trajectory = TruthTrajectory::new(proposal);
        assert_eq!(trajectory.event_count(), 1);
        assert_eq!(trajectory.current_gestalt(), GestaltState::Epiphany);

        // Add evidence event
        trajectory.record_event(EvidenceEvent {
            timestamp_ms: 1000,
            event_type: EvidenceEventType::MatchesAdded(112),
            nars_frequency: 0.83,
            nars_confidence: 0.91,
            significance: rustynum_core::SignificanceLevel::Strong,
            evidence_count: 612,
            gestalt: GestaltState::Crystallizing,
        });

        assert_eq!(trajectory.event_count(), 2);
        assert_eq!(trajectory.current_gestalt(), GestaltState::Crystallizing);
        let (f, c) = trajectory.current_truth();
        assert!((f - 0.83).abs() < 0.001);
        assert!((c - 0.91).abs() < 0.001);
        assert!(trajectory.confidence_trend() > 0.0);

        // Counter-evidence arrives
        trajectory.record_event(EvidenceEvent {
            timestamp_ms: 2000,
            event_type: EvidenceEventType::CounterEvidence(3),
            nars_frequency: 0.79,
            nars_confidence: 0.84,
            significance: rustynum_core::SignificanceLevel::Evidence,
            evidence_count: 615,
            gestalt: GestaltState::Contested,
        });

        assert_eq!(trajectory.event_count(), 3);
        assert_eq!(trajectory.current_gestalt(), GestaltState::Contested);
        assert!(trajectory.confidence_trend() < 0.0); // confidence dropped
    }
}
