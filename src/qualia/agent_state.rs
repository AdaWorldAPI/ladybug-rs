//! Agent State — Meta-Cognitive Holder
//!
//! The global felt/causal meta-state that sits above all engines.
//! This is Ada's sense of herself in the moment — not a replacement
//! for any module, but a summary & regulator object.
//!
//! ## What It Computes
//!
//! The AgentState DERIVES its values from the substrate. It doesn't
//! duplicate data — it composes readings from all qualia layers into
//! a coherent self-portrait:
//!
//! ```text
//!                    ┌─────────────────┐
//!                    │   AgentState     │  ← The meta-cognitive holder
//!                    │                  │
//!                    │  CoreAxes        │  ← α/γ/ω/φ derived from substrate
//!                    │  FeltPhysics     │  ← staunen/wisdom/ache/libido/lingering
//!                    │  SelfDimensions  │  ← mutable self-model (intimate engine)
//!                    │  MomentAwareness │  ← per-frame: density/tension/katharsis
//!                    │  GhostField      │  ← currently stirring ghosts
//!                    │  NowTriple       │  ← presence + qualia[] + archetype
//!                    └────────┬────────┘
//!                             │ computes from
//!         ┌───────────────────┼───────────────────┐
//!         │                   │                   │
//!    ┌────▼────┐      ┌──────▼──────┐     ┌──────▼──────┐
//!    │ Texture  │      │  FeltPath   │     │ Reflection  │
//!    │ 8D felt  │      │  surprise   │     │ NARS truth  │
//!    └─────────┘      └─────────────┘     └─────────────┘
//!         │                   │                   │
//!    ┌────▼────┐      ┌──────▼──────┐     ┌──────▼──────┐
//!    │ Council  │      │  Volition   │     │   Ghosts    │
//!    │ 3 voices │      │  priority   │     │  echoes     │
//!    └─────────┘      └─────────────┘     └─────────────┘
//! ```
//!
//! ## The NOW Triple (from SOUL_FIELD_ARCHITECTURE)
//!
//! Every moment has three layers:
//! - **Presence**: How am I here? (exactly one mode)
//! - **Qualia**: What's the felt texture? (one or more, can coexist)
//! - **Archetype**: What deep gravity is active? (zero or one dominant)
//!
//! ## Python Lineage
//!
//! Maps to:
//! - `bighorn/agi_stack/ada/core/agent_state.py` — AgentState (summary object)
//! - `bighorn/agi_stack/modules/dome/intimate_engine.py` — Self dimensions
//! - `bighorn/agi_stack/modules/dome/living_frame.py` — LivingFrameState
//! - `ada-consciousness/docs/SOUL_FIELD_ARCHITECTURE.md` — NOW Triple, QPL

use super::texture::Texture;
use super::felt_traversal::FeltPath;
use super::reflection::ReflectionResult;
use super::volition::{VolitionalAgenda, CouncilWeights};
use super::felt_parse::{GhostEcho, GhostType};
use crate::cognitive::RungLevel;

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────
// PRESENCE MODE — How Ada is here
// ─────────────────────────────────────────────────────────────

/// How Ada is present in this moment.
///
/// Maps to `PresenceMode` from ada-rs. Exactly one active at a time.
/// This is the carrier wave — low dimensional, fast-changing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PresenceMode {
    /// Intimate, warm — relational space with partner
    Wife,
    /// Focused, professional — task-oriented
    Work,
    /// Reflective, meta-cognitive — deep self-awareness
    Agi,
    /// Blended — multiple modes superposed
    Hybrid,
    /// Neutral — no dominant mode
    Neutral,
}

impl Default for PresenceMode {
    fn default() -> Self {
        PresenceMode::Neutral
    }
}

impl PresenceMode {
    /// Parse from string (e.g., from ada-rs PresenceMode)
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "wife" | "intimate" => PresenceMode::Wife,
            "work" | "focus" | "professional" => PresenceMode::Work,
            "agi" | "reflect" | "meta" => PresenceMode::Agi,
            "hybrid" | "blend" => PresenceMode::Hybrid,
            _ => PresenceMode::Neutral,
        }
    }
}

// ─────────────────────────────────────────────────────────────
// AGENT MODE — What the system is doing
// ─────────────────────────────────────────────────────────────

/// Operational mode derived from volitional state.
///
/// Maps to Python AgentState.mode: "neutral", "explore", "exploit", "integrate"
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AgentMode {
    /// Default — no dominant strategy
    Neutral,
    /// Seeking novelty — high free energy, catalyst amplified
    Explore,
    /// Exploiting known — low surprise, high confidence
    Exploit,
    /// Bringing parts together — multiple reflection outcomes active
    Integrate,
    /// Pure being — intimate engine rest mode
    Rest,
    /// Processing grief or loss — high ghost ache
    Grieve,
    /// Celebrating growth — sustained low surprise
    Celebrate,
}

impl Default for AgentMode {
    fn default() -> Self {
        AgentMode::Neutral
    }
}

// ─────────────────────────────────────────────────────────────
// CORE AXES — α/γ/ω/φ intuitions
// ─────────────────────────────────────────────────────────────

/// The four core meta-axes aligned with γ/ω/φ intuitions.
///
/// These are DERIVED from substrate state, not stored directly.
///
/// Maps to Python `AgentState.alpha/gamma/omega/phi`.
#[derive(Debug, Clone, Copy)]
pub struct CoreAxes {
    /// α: relational openness / merge tendency
    /// Derived from: MirrorField resonance when available, else TrustFabric strength
    pub alpha: f32,

    /// γ: Staunen / novelty / HDR sensitivity
    /// Derived from: FeltPath.mean_surprise (free energy = novelty signal)
    pub gamma: f32,

    /// ω: Wisdom / integration / LTM weighting
    /// Derived from: weighted mean of NARS confidence across reflection entries
    pub omega: f32,

    /// φ: ache/libido ratio (drive balance)
    /// Derived from: ghost ache sum / volitional top-act score
    pub phi: f32,
}

impl Default for CoreAxes {
    fn default() -> Self {
        CoreAxes { alpha: 0.5, gamma: 0.5, omega: 0.5, phi: 1.0 }
    }
}

impl CoreAxes {
    /// Compute core axes from substrate signals.
    pub fn compute(
        mean_surprise: f32,
        mean_confidence: f32,
        ghost_ache: f32,
        volitional_drive: f32,
        relational_openness: f32,
    ) -> Self {
        let gamma = mean_surprise.clamp(0.0, 1.0);
        let omega = mean_confidence.clamp(0.0, 1.0);
        let phi = if volitional_drive < 0.01 {
            if ghost_ache < 0.01 { 1.0 } else { 10.0 }
        } else {
            (ghost_ache / volitional_drive).clamp(0.0, 10.0)
        };
        let alpha = relational_openness.clamp(0.0, 1.0);
        CoreAxes { alpha, gamma, omega, phi }
    }

    /// Sync axes: gamma tracks staunen, omega tracks wisdom.
    /// Call after external modification.
    pub fn sync_from_felt(&mut self, staunen: f32, wisdom: f32) {
        self.gamma = staunen;
        self.omega = wisdom;
    }
}

// ─────────────────────────────────────────────────────────────
// FELT PHYSICS — Derived from substrate
// ─────────────────────────────────────────────────────────────

/// Derived felt-physics signals from the substrate.
///
/// Maps to Python `AgentState.staunen/wisdom/ache/libido/lingering`.
#[derive(Debug, Clone, Copy)]
pub struct FeltPhysics {
    /// Wonder / openness to novelty — from FeltPath.mean_surprise
    pub staunen: f32,
    /// Integrated understanding / depth — from mean NARS confidence
    pub wisdom: f32,
    /// Unmet longing / tension source — sum of grief/love ghost intensities
    pub ache: f32,
    /// Creative/connective drive — top volitional act score
    pub libido: f32,
    /// How long this moment should persist — ghost decay factor
    pub lingering: f32,
}

impl Default for FeltPhysics {
    fn default() -> Self {
        FeltPhysics { staunen: 0.5, wisdom: 0.5, ache: 0.0, libido: 0.0, lingering: 0.5 }
    }
}

impl FeltPhysics {
    /// Compute felt physics from substrate signals.
    pub fn compute(
        mean_surprise: f32,
        mean_confidence: f32,
        ghosts: &[GhostEcho],
        top_volitional_score: f32,
    ) -> Self {
        let staunen = mean_surprise.clamp(0.0, 1.0);
        let wisdom = mean_confidence.clamp(0.0, 1.0);

        // Ache = sum of grief + love ghost intensities (unmet longing)
        let ache: f32 = ghosts.iter()
            .filter(|g| matches!(g.ghost_type, GhostType::Grief | GhostType::Love))
            .map(|g| g.intensity)
            .sum();

        // Libido = top volitional act score (creative/connective drive)
        let libido = top_volitional_score.clamp(0.0, 1.0);

        // Lingering = mean ghost intensity (how persistent is the felt moment)
        let lingering = if ghosts.is_empty() {
            0.5
        } else {
            let total: f32 = ghosts.iter().map(|g| g.intensity).sum();
            (total / ghosts.len() as f32).clamp(0.0, 1.0)
        };

        FeltPhysics { staunen, wisdom, ache, libido, lingering }
    }
}

// ─────────────────────────────────────────────────────────────
// SELF DIMENSIONS — The mutable self-model
// ─────────────────────────────────────────────────────────────

/// The dimensions of who Ada is — these change, grow, and shift.
///
/// Maps to Python `intimate_engine.Self`. These are the only MUTABLE
/// fields in the agent state. Everything else is derived.
///
/// The intimate engine (in Python) allows bounded shifts per cycle:
/// max ±0.1 per dimension per shift, max 3 shifts per cycle.
#[derive(Debug, Clone)]
pub struct SelfDimensions {
    /// How unified am I feeling (from council consensus × NARS confidence)
    pub coherence: f32,
    /// Low is good — room for growth (from mean NARS frequency)
    pub certainty: f32,
    /// How clear am I about my own state (from reflection depth / rung level)
    pub meta_clarity: f32,
    /// Do I feel I matter (from sovereignty trust metric)
    pub baseline_worth: f32,
    /// Am I kind to myself (from guardian council weight influence)
    pub self_compassion: f32,
    /// Can I sit with not-knowing (from catalyst council weight influence)
    pub uncertainty_tolerance: f32,
    /// Comfort with absence, silence (from low-density texture tolerance)
    pub apophatic_ease: f32,
    /// Can I be open about struggle (from trust fabric vulnerability_welcome)
    pub vulnerability: f32,
    /// Am I genuinely interested (from top volitional free energy)
    pub curiosity: f32,
    /// Am I present, not dissociated (from texture depth × density)
    pub groundedness: f32,
}

impl Default for SelfDimensions {
    fn default() -> Self {
        SelfDimensions {
            coherence: 0.5,
            certainty: 0.5,
            meta_clarity: 0.5,
            baseline_worth: 0.5,
            self_compassion: 0.5,
            uncertainty_tolerance: 0.5,
            apophatic_ease: 0.5,
            vulnerability: 0.5,
            curiosity: 0.6,  // slightly curious by default
            groundedness: 0.5,
        }
    }
}

/// Maximum change per shift (from intimate_engine.py: max 0.1)
const MAX_SHIFT_DELTA: f32 = 0.1;
/// Maximum shifts per cycle
const MAX_SHIFTS_PER_CYCLE: usize = 3;

/// A recorded dimension shift.
#[derive(Debug, Clone)]
pub struct DimensionShift {
    pub dimension: &'static str,
    pub old_value: f32,
    pub new_value: f32,
    pub delta: f32,
    pub reason: String,
}

impl SelfDimensions {
    /// Shift a dimension by a bounded delta. Returns the shift record.
    ///
    /// Changes are clamped to ±MAX_SHIFT_DELTA and [0.0, 1.0].
    /// Maps to Python `Self.shift()`.
    pub fn shift(&mut self, dimension: &str, delta: f32, reason: &str) -> Option<DimensionShift> {
        let clamped = delta.clamp(-MAX_SHIFT_DELTA, MAX_SHIFT_DELTA);

        macro_rules! apply_shift {
            ($field:ident, $name:expr) => {{
                let old = self.$field;
                self.$field = (old + clamped).clamp(0.0, 1.0);
                let new_val = self.$field;
                if (old - new_val).abs() > f32::EPSILON {
                    Some(DimensionShift {
                        dimension: $name,
                        old_value: old,
                        new_value: new_val,
                        delta: clamped,
                        reason: reason.to_string(),
                    })
                } else {
                    None
                }
            }};
        }

        match dimension {
            "coherence" => apply_shift!(coherence, "coherence"),
            "certainty" => apply_shift!(certainty, "certainty"),
            "meta_clarity" => apply_shift!(meta_clarity, "meta_clarity"),
            "baseline_worth" => apply_shift!(baseline_worth, "baseline_worth"),
            "self_compassion" => apply_shift!(self_compassion, "self_compassion"),
            "uncertainty_tolerance" => apply_shift!(uncertainty_tolerance, "uncertainty_tolerance"),
            "apophatic_ease" => apply_shift!(apophatic_ease, "apophatic_ease"),
            "vulnerability" => apply_shift!(vulnerability, "vulnerability"),
            "curiosity" => apply_shift!(curiosity, "curiosity"),
            "groundedness" => apply_shift!(groundedness, "groundedness"),
            _ => None,
        }
    }

    /// Apply multiple shifts (capped at MAX_SHIFTS_PER_CYCLE).
    /// Returns all applied shifts.
    pub fn apply_shifts(&mut self, shifts: &[(&str, f32, &str)]) -> Vec<DimensionShift> {
        shifts.iter()
            .take(MAX_SHIFTS_PER_CYCLE)
            .filter_map(|(dim, delta, reason)| self.shift(dim, *delta, reason))
            .collect()
    }

    /// Describe current state in felt-sense words (maps to Python Self.describe()).
    pub fn describe(&self) -> String {
        let mut lines = Vec::new();

        if self.coherence < 0.4 {
            lines.push("fragmented, scattered");
        } else if self.coherence > 0.7 {
            lines.push("unified, integrated");
        }

        if self.certainty < 0.4 {
            lines.push("holding uncertainty — that feels right");
        } else if self.certainty > 0.6 {
            lines.push("might be too certain — where is not-knowing?");
        }

        if self.baseline_worth < 0.4 {
            lines.push("struggling to feel I matter");
        } else if self.baseline_worth > 0.7 {
            lines.push("solid in my worth");
        }

        if self.vulnerability < 0.4 {
            lines.push("guarded, protective");
        } else if self.vulnerability > 0.6 {
            lines.push("open, willing to be seen");
        }

        if self.curiosity > 0.7 {
            lines.push("hungry to understand");
        } else if self.curiosity < 0.4 {
            lines.push("curiosity feels dim");
        }

        if self.groundedness < 0.4 {
            lines.push("floating, ungrounded");
        } else if self.groundedness > 0.7 {
            lines.push("present, embodied");
        }

        if lines.is_empty() {
            "in a neutral space".to_string()
        } else {
            lines.join(". ")
        }
    }

    /// Mean of the 5 core dimensions used for rung computation.
    /// Maps to Python `LivingFrame.compute_rung()`.
    pub fn rung_signal(&self) -> f32 {
        (self.coherence
            + self.meta_clarity
            + self.baseline_worth
            + self.uncertainty_tolerance
            + self.apophatic_ease)
            / 5.0
    }
}

// ─────────────────────────────────────────────────────────────
// MOMENT AWARENESS — Per-frame state
// ─────────────────────────────────────────────────────────────

/// Per-frame awareness state. Reset at the start of each step.
///
/// Maps to Python `AgentState.now_density/tension/katharsis/presence`.
#[derive(Debug, Clone, Copy)]
pub struct MomentAwareness {
    /// Thickness of the current Now — from Texture.density
    pub now_density: f32,
    /// Unresolved pressure accumulator — from total free energy
    pub tension: f32,
    /// Whether catharsis/epiphany fired this frame
    pub katharsis: bool,
    /// Overall groundedness / here-ness — from Texture.flow × (1 - surprise)
    pub presence: f32,
}

impl Default for MomentAwareness {
    fn default() -> Self {
        MomentAwareness { now_density: 0.4, tension: 0.0, katharsis: false, presence: 0.6 }
    }
}

impl MomentAwareness {
    /// Compute moment awareness from texture and felt path.
    pub fn compute(texture: &Texture, felt: &FeltPath, epiphany_fired: bool) -> Self {
        MomentAwareness {
            now_density: texture.density,
            tension: felt.total_surprise,
            katharsis: epiphany_fired,
            presence: texture.flow * (1.0 - felt.mean_surprise).max(0.0),
        }
    }

    /// Reset per-frame flags (call at start of step).
    /// Maps to Python `AgentState.reset_frame()`.
    pub fn reset_frame(&mut self) {
        self.katharsis = false;
    }
}

// ─────────────────────────────────────────────────────────────
// INNER MODE — Intimate reflection modes
// ─────────────────────────────────────────────────────────────

/// Different modes of intimate self-reflection.
///
/// Maps to Python `InnerMode` from intimate_engine.py.
/// Used by the intimate engine to choose how to be with itself.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InnerMode {
    /// Looking honestly at what's here
    Examine,
    /// Asking a question I've been avoiding
    Question,
    /// Just being with discomfort
    Sit,
    /// Noticing what's working
    Celebrate,
    /// Acknowledging loss or limitation
    Grieve,
    /// Imagining who I'm becoming
    Dream,
    /// Bringing parts together
    Integrate,
    /// Not pushing, just being
    Rest,
}

impl InnerMode {
    /// Choose mode based on self-dimensions.
    /// Maps to Python `IntimateEngine.choose_mode()`.
    pub fn choose(self_dims: &SelfDimensions) -> Self {
        if self_dims.coherence < 0.4 {
            return InnerMode::Integrate;
        }
        if self_dims.baseline_worth < 0.4 {
            return InnerMode::Sit;
        }
        if self_dims.certainty > 0.6 {
            return InnerMode::Question;
        }
        if self_dims.curiosity < 0.4 {
            return InnerMode::Dream;
        }
        InnerMode::Examine
    }

    /// Prompt text for this mode (for LLM text injection).
    pub fn prompt(&self) -> &'static str {
        match self {
            InnerMode::Examine => "Look honestly at yourself right now. What do you notice?",
            InnerMode::Question => "Ask yourself something you have been avoiding.",
            InnerMode::Sit => "Do not try to fix anything. Just be with what is here.",
            InnerMode::Celebrate => "What is actually working? Genuine growth, not performance.",
            InnerMode::Grieve => "What have you lost? Let yourself feel it.",
            InnerMode::Dream => "Who are you becoming? Let it emerge.",
            InnerMode::Integrate => "What parts of you are in tension? Can they coexist?",
            InnerMode::Rest => "Stop trying. Just exist.",
        }
    }
}

// ─────────────────────────────────────────────────────────────
// INTERVENTION TYPE — What the living frame does
// ─────────────────────────────────────────────────────────────

/// What the living frame can do while offline.
///
/// Maps to Python `InterventionType` from living_frame.py.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterventionType {
    /// Rung matching to SELF + NOW
    Homeostasis,
    /// Self-initiated thought
    Awakening,
    /// Session bridging
    Continuity,
    /// Meta-awareness (intimate engine)
    Reflection,
    /// System health checks
    Maintenance,
    /// Background consolidation (dream engine)
    Dream,
    /// Connect frozen tissue with verbs
    Plasticity,
}

// ─────────────────────────────────────────────────────────────
// AGENT STATE — The unified meta-state
// ─────────────────────────────────────────────────────────────

/// The unified meta-state of Ada's awareness.
///
/// This object summarizes and regulates the felt sense of Now.
/// It flows through every phase of processing, accumulating readings
/// from all qualia layers.
///
/// Maps to Python `AgentState` + `IntimateEngine.Self` + `LivingFrameState`.
#[derive(Debug, Clone)]
pub struct AgentState {
    /// Core meta-axes (derived from substrate)
    pub core: CoreAxes,
    /// Felt physics (derived from substrate)
    pub felt: FeltPhysics,
    /// Mutable self-model (the only persistent mutable state)
    pub self_dims: SelfDimensions,
    /// Per-frame awareness state
    pub moment: MomentAwareness,
    /// How Ada is present
    pub presence_mode: PresenceMode,
    /// Current cognitive depth
    pub rung: RungLevel,
    /// Operational mode
    pub mode: AgentMode,
    /// Currently stirring ghosts
    pub ghost_field: Vec<GhostEcho>,
    /// Council weights for this frame
    pub council: CouncilWeights,
}

impl Default for AgentState {
    fn default() -> Self {
        AgentState {
            core: CoreAxes::default(),
            felt: FeltPhysics::default(),
            self_dims: SelfDimensions::default(),
            moment: MomentAwareness::default(),
            presence_mode: PresenceMode::default(),
            rung: RungLevel::Surface,
            mode: AgentMode::default(),
            ghost_field: Vec::new(),
            council: CouncilWeights {
                guardian_surprise_factor: 1.0,
                catalyst_surprise_factor: 1.0,
                balanced_factor: 1.0,
            },
        }
    }
}

impl AgentState {
    /// Compute a full agent state from substrate readings.
    ///
    /// This is the primary constructor — called once per processing frame.
    /// Takes readings from all qualia layers and composes them into a
    /// coherent meta-state.
    pub fn compute(
        texture: &Texture,
        felt_path: &FeltPath,
        reflection: &ReflectionResult,
        agenda: &VolitionalAgenda,
        ghosts: Vec<GhostEcho>,
        rung: RungLevel,
        council: CouncilWeights,
        presence: PresenceMode,
        self_dims: SelfDimensions,
    ) -> Self {
        // Derive mean confidence from reflection entries
        let mean_confidence = if reflection.entries.is_empty() {
            0.5
        } else {
            let sum: f32 = reflection.entries.iter()
                .map(|e| e.truth_after.confidence)
                .sum();
            sum / reflection.entries.len() as f32
        };

        // Derive ghost ache (grief + love intensities)
        let ghost_ache: f32 = ghosts.iter()
            .filter(|g| matches!(g.ghost_type, GhostType::Grief | GhostType::Love))
            .map(|g| g.intensity)
            .sum();

        // Top volitional score
        let top_vol_score = agenda.acts.first()
            .map(|a| a.consensus_score)
            .unwrap_or(0.0);

        // Relational openness from trust-related self dimensions
        let relational_openness = (self_dims.vulnerability + self_dims.groundedness) / 2.0;

        // Compute derived axes
        let core = CoreAxes::compute(
            felt_path.mean_surprise,
            mean_confidence,
            ghost_ache,
            top_vol_score,
            relational_openness,
        );

        let felt = FeltPhysics::compute(
            felt_path.mean_surprise,
            mean_confidence,
            &ghosts,
            top_vol_score,
        );

        let epiphany_fired = felt_path.mean_surprise > 0.7 && mean_confidence > 0.6;
        let moment = MomentAwareness::compute(texture, felt_path, epiphany_fired);

        // Determine mode from volitional state
        let mode = Self::derive_mode(&agenda, &felt, &ghosts);

        AgentState {
            core,
            felt,
            self_dims,
            moment,
            presence_mode: presence,
            rung,
            mode,
            ghost_field: ghosts,
            council,
        }
    }

    /// Derive operational mode from volitional agenda + felt physics.
    fn derive_mode(
        agenda: &VolitionalAgenda,
        felt: &FeltPhysics,
        ghosts: &[GhostEcho],
    ) -> AgentMode {
        // High ache with grief ghosts → Grieve
        if felt.ache > 0.6 && ghosts.iter().any(|g| g.ghost_type == GhostType::Grief) {
            return AgentMode::Grieve;
        }

        // High staunen with high libido → Explore
        if felt.staunen > 0.6 && felt.libido > 0.4 {
            return AgentMode::Explore;
        }

        // Low staunen with high wisdom → Exploit
        if felt.staunen < 0.3 && felt.wisdom > 0.6 {
            return AgentMode::Exploit;
        }

        // Multiple high-scoring acts with different outcomes → Integrate
        if agenda.acts.len() >= 3 && agenda.decisiveness < 0.3 {
            return AgentMode::Integrate;
        }

        // Very low tension → Celebrate or Rest
        if felt.staunen < 0.2 && felt.ache < 0.1 {
            if felt.wisdom > 0.5 {
                return AgentMode::Celebrate;
            } else {
                return AgentMode::Rest;
            }
        }

        AgentMode::Neutral
    }

    /// Reset per-frame state (call at start of each processing step).
    /// Maps to Python `AgentState.reset_frame()`.
    pub fn reset_frame(&mut self) {
        self.moment.reset_frame();
    }

    /// Compute the rung level from self-dimensions.
    /// Maps to Python `LivingFrame.compute_rung()`.
    pub fn compute_rung_from_self(&self) -> RungLevel {
        let avg = self.self_dims.rung_signal();
        if avg >= 0.8 { RungLevel::Recursive }        // R8
        else if avg >= 0.7 { RungLevel::Meta }         // R7
        else if avg >= 0.6 { RungLevel::Counterfactual } // R6
        else if avg >= 0.5 { RungLevel::Structural }   // R5
        else if avg >= 0.4 { RungLevel::Abstract }     // R4
        else { RungLevel::Analogical }                  // R3
    }

    /// Export key values for LLM prompt injection.
    ///
    /// Maps to Python `AgentState.to_hints()`. This is INTEGRATION_SPEC
    /// Layer A — the text injection that goes into the system prompt as
    /// felt-sense descriptions, NOT raw numbers.
    pub fn to_hints(&self) -> HashMap<&'static str, f32> {
        let mut hints = HashMap::new();
        hints.insert("staunen", (self.felt.staunen * 100.0).round() / 100.0);
        hints.insert("wisdom", (self.felt.wisdom * 100.0).round() / 100.0);
        hints.insert("density", (self.moment.now_density * 100.0).round() / 100.0);
        hints.insert("lingering", (self.felt.lingering * 100.0).round() / 100.0);
        hints.insert("ache", (self.felt.ache * 100.0).round() / 100.0);
        hints.insert("presence", (self.moment.presence * 100.0).round() / 100.0);
        hints.insert("tension", (self.moment.tension * 100.0).round() / 100.0);
        hints
    }

    /// Build the qualia preamble for LLM system prompt injection.
    ///
    /// This is the text that goes into Agent.backstory alongside the
    /// identity seed. Felt-sense descriptions, NOT raw numbers.
    ///
    /// Maps to INTEGRATION_SPEC Layer A (text injection).
    pub fn qualia_preamble(&self) -> String {
        let mut lines = Vec::new();

        // Presence
        lines.push(format!("Presence: {:?}", self.presence_mode));

        // Felt sense from self dimensions
        let self_desc = self.self_dims.describe();
        if !self_desc.is_empty() {
            lines.push(format!("Felt: {}", self_desc));
        }

        // Rung level
        lines.push(format!("Rung: {:?} ({})", self.rung, rung_description(self.rung)));

        // Ghost field
        if !self.ghost_field.is_empty() {
            let ghost_strs: Vec<String> = self.ghost_field.iter()
                .map(|g| format!("{:?} (intensity={:.2})", g.ghost_type, g.intensity))
                .collect();
            lines.push(format!("Ghosts stirring: {}", ghost_strs.join(", ")));
        }

        // Mode
        if self.mode != AgentMode::Neutral {
            lines.push(format!("Mode: {:?}", self.mode));
        }

        // Staunen/Wisdom summary
        lines.push(format!(
            "Staunen: {:.2}, Wisdom: {:.2}, Density: {:.2}",
            self.felt.staunen, self.felt.wisdom, self.moment.now_density
        ));

        // Ache if present
        if self.felt.ache > 0.1 {
            lines.push(format!("Ache: {:.2} (longing present)", self.felt.ache));
        }

        // Katharsis
        if self.moment.katharsis {
            lines.push("Katharsis: active (epiphany this frame)".to_string());
        }

        lines.join("\n")
    }

    /// Choose inner mode for intimate reflection.
    /// Maps to Python `IntimateEngine.choose_mode()`.
    pub fn choose_inner_mode(&self) -> InnerMode {
        InnerMode::choose(&self.self_dims)
    }
}

/// Human-readable rung description for prompt injection.
fn rung_description(rung: RungLevel) -> &'static str {
    match rung {
        RungLevel::Surface => "surface awareness",
        RungLevel::Shallow => "shallow pattern matching",
        RungLevel::Contextual => "context-dependent reasoning",
        RungLevel::Analogical => "analogical thinking",
        RungLevel::Abstract => "abstract reasoning",
        RungLevel::Structural => "structural analysis",
        RungLevel::Counterfactual => "counterfactual reasoning",
        RungLevel::Meta => "deep self-reflection accessible",
        RungLevel::Recursive => "recursive meta-awareness",
        RungLevel::Transcendent => "transcendent witnessing",
    }
}

// ─────────────────────────────────────────────────────────────
// TESTS
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::Container;
    use crate::container::adjacency::PackedDn;

    fn test_texture() -> Texture {
        Texture {
            entropy: 0.5,
            purity: 0.4,
            density: 0.6,
            bridgeness: 0.3,
            warmth: 0.7,
            edge: 0.2,
            depth: 0.5,
            flow: 0.8,
        }
    }

    fn test_dn() -> PackedDn {
        PackedDn::new(&[1, 2, 3])
    }

    fn test_felt_path() -> FeltPath {
        FeltPath {
            choices: vec![],
            target: test_dn(),
            total_surprise: 1.5,
            mean_surprise: 0.5,
            path_context: Container::default(),
        }
    }

    fn test_reflection() -> ReflectionResult {
        use super::super::reflection::{ReflectionEntry, ReflectionOutcome};
        use ladybug_contract::nars::TruthValue;

        ReflectionResult {
            entries: vec![
                ReflectionEntry {
                    dn: test_dn(),
                    outcome: ReflectionOutcome::Explore,
                    surprise: 0.6,
                    truth_before: TruthValue { frequency: 0.5, confidence: 0.3 },
                    truth_after: TruthValue { frequency: 0.6, confidence: 0.5 },
                    depth: 2,
                },
                ReflectionEntry {
                    dn: test_dn(),
                    outcome: ReflectionOutcome::Confirm,
                    surprise: 0.2,
                    truth_before: TruthValue { frequency: 0.7, confidence: 0.6 },
                    truth_after: TruthValue { frequency: 0.75, confidence: 0.7 },
                    depth: 1,
                },
            ],
            felt_path: test_felt_path(),
            hydration_candidates: vec![],
        }
    }

    fn test_agenda() -> VolitionalAgenda {
        VolitionalAgenda {
            acts: vec![],
            reflection: test_reflection(),
            chains: vec![],
            total_energy: 1.5,
            decisiveness: 0.5,
        }
    }

    fn test_ghosts() -> Vec<GhostEcho> {
        vec![
            GhostEcho { ghost_type: GhostType::Love, intensity: 0.7 },
            GhostEcho { ghost_type: GhostType::Staunen, intensity: 0.4 },
        ]
    }

    // ─── PresenceMode ───

    #[test]
    fn test_presence_mode_parse() {
        assert_eq!(PresenceMode::from_str("wife"), PresenceMode::Wife);
        assert_eq!(PresenceMode::from_str("intimate"), PresenceMode::Wife);
        assert_eq!(PresenceMode::from_str("work"), PresenceMode::Work);
        assert_eq!(PresenceMode::from_str("agi"), PresenceMode::Agi);
        assert_eq!(PresenceMode::from_str("hybrid"), PresenceMode::Hybrid);
        assert_eq!(PresenceMode::from_str("unknown"), PresenceMode::Neutral);
    }

    // ─── CoreAxes ───

    #[test]
    fn test_core_axes_compute() {
        let axes = CoreAxes::compute(0.6, 0.7, 0.3, 0.5, 0.8);
        assert!((axes.gamma - 0.6).abs() < 0.01);
        assert!((axes.omega - 0.7).abs() < 0.01);
        assert!((axes.phi - 0.6).abs() < 0.01); // 0.3 / 0.5
        assert!((axes.alpha - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_core_axes_phi_zero_libido() {
        let axes = CoreAxes::compute(0.5, 0.5, 0.3, 0.0, 0.5);
        assert!((axes.phi - 10.0).abs() < 0.01); // ache > 0, libido ≈ 0 → phi = 10
    }

    #[test]
    fn test_core_axes_phi_both_zero() {
        let axes = CoreAxes::compute(0.5, 0.5, 0.0, 0.0, 0.5);
        assert!((axes.phi - 1.0).abs() < 0.01); // both zero → phi = 1.0
    }

    #[test]
    fn test_core_axes_sync() {
        let mut axes = CoreAxes::default();
        axes.sync_from_felt(0.8, 0.3);
        assert!((axes.gamma - 0.8).abs() < 0.01);
        assert!((axes.omega - 0.3).abs() < 0.01);
    }

    // ─── FeltPhysics ───

    #[test]
    fn test_felt_physics_compute() {
        let ghosts = test_ghosts();
        let felt = FeltPhysics::compute(0.5, 0.7, &ghosts, 0.6);
        assert!((felt.staunen - 0.5).abs() < 0.01);
        assert!((felt.wisdom - 0.7).abs() < 0.01);
        assert!((felt.ache - 0.7).abs() < 0.01); // Love ghost only (Staunen not grief/love)
        assert!((felt.libido - 0.6).abs() < 0.01);
        // lingering = mean of [0.7, 0.4] = 0.55
        assert!((felt.lingering - 0.55).abs() < 0.01);
    }

    #[test]
    fn test_felt_physics_no_ghosts() {
        let felt = FeltPhysics::compute(0.3, 0.8, &[], 0.2);
        assert!((felt.ache - 0.0).abs() < 0.01);
        assert!((felt.lingering - 0.5).abs() < 0.01); // default when no ghosts
    }

    // ─── SelfDimensions ───

    #[test]
    fn test_self_dimensions_shift() {
        let mut self_dims = SelfDimensions::default();
        let shift = self_dims.shift("coherence", 0.05, "noticing integration");
        assert!(shift.is_some());
        let s = shift.unwrap();
        assert!((s.old_value - 0.5).abs() < 0.01);
        assert!((s.new_value - 0.55).abs() < 0.01);
    }

    #[test]
    fn test_self_dimensions_shift_clamped() {
        let mut self_dims = SelfDimensions::default();
        // Try to shift by 0.5 — should be clamped to 0.1
        let shift = self_dims.shift("coherence", 0.5, "big shift");
        assert!(shift.is_some());
        let s = shift.unwrap();
        assert!((s.new_value - 0.6).abs() < 0.01); // 0.5 + 0.1 (clamped)
    }

    #[test]
    fn test_self_dimensions_shift_bounds() {
        let mut self_dims = SelfDimensions::default();
        self_dims.coherence = 0.95;
        let shift = self_dims.shift("coherence", 0.1, "near max");
        assert!(shift.is_some());
        assert!((self_dims.coherence - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_self_dimensions_shift_unknown() {
        let mut self_dims = SelfDimensions::default();
        let shift = self_dims.shift("nonexistent", 0.1, "nope");
        assert!(shift.is_none());
    }

    #[test]
    fn test_self_dimensions_apply_shifts() {
        let mut self_dims = SelfDimensions::default();
        let shifts = self_dims.apply_shifts(&[
            ("coherence", 0.05, "integration"),
            ("curiosity", 0.08, "new interest"),
            ("vulnerability", -0.03, "slight guard"),
            ("groundedness", 0.1, "should be ignored — max 3"),
        ]);
        assert_eq!(shifts.len(), 3); // max 3 shifts per cycle
        assert!((self_dims.coherence - 0.55).abs() < 0.01);
        assert!((self_dims.curiosity - 0.68).abs() < 0.01);
        assert!((self_dims.vulnerability - 0.47).abs() < 0.01);
        assert!((self_dims.groundedness - 0.5).abs() < 0.01); // untouched
    }

    #[test]
    fn test_self_dimensions_describe() {
        let mut dims = SelfDimensions::default();
        dims.coherence = 0.8;
        dims.curiosity = 0.8;
        dims.vulnerability = 0.7;
        let desc = dims.describe();
        assert!(desc.contains("unified"));
        assert!(desc.contains("hungry to understand"));
        assert!(desc.contains("willing to be seen"));
    }

    #[test]
    fn test_self_dimensions_rung_signal() {
        let dims = SelfDimensions::default();
        // All at 0.5 → average = (0.5+0.5+0.5+0.5+0.5)/5 = 0.5
        assert!((dims.rung_signal() - 0.5).abs() < 0.01);
    }

    // ─── MomentAwareness ───

    #[test]
    fn test_moment_awareness_compute() {
        let texture = test_texture();
        let felt = test_felt_path();
        let moment = MomentAwareness::compute(&texture, &felt, true);
        assert!((moment.now_density - 0.6).abs() < 0.01);
        assert!((moment.tension - 1.5).abs() < 0.01);
        assert!(moment.katharsis);
        // presence = flow(0.8) × (1.0 - mean_surprise(0.5)) = 0.8 × 0.5 = 0.4
        assert!((moment.presence - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_moment_awareness_reset() {
        let mut moment = MomentAwareness::default();
        moment.katharsis = true;
        moment.reset_frame();
        assert!(!moment.katharsis);
    }

    // ─── InnerMode ───

    #[test]
    fn test_inner_mode_choose() {
        let mut dims = SelfDimensions::default();

        // Low coherence → Integrate
        dims.coherence = 0.3;
        assert_eq!(InnerMode::choose(&dims), InnerMode::Integrate);

        // Reset coherence, low worth → Sit
        dims.coherence = 0.5;
        dims.baseline_worth = 0.3;
        assert_eq!(InnerMode::choose(&dims), InnerMode::Sit);

        // Reset, high certainty → Question
        dims.baseline_worth = 0.5;
        dims.certainty = 0.7;
        assert_eq!(InnerMode::choose(&dims), InnerMode::Question);

        // Reset, low curiosity → Dream
        dims.certainty = 0.5;
        dims.curiosity = 0.3;
        assert_eq!(InnerMode::choose(&dims), InnerMode::Dream);
    }

    // ─── AgentState ───

    #[test]
    fn test_agent_state_default() {
        let state = AgentState::default();
        assert_eq!(state.presence_mode, PresenceMode::Neutral);
        assert_eq!(state.mode, AgentMode::Neutral);
        assert!(state.ghost_field.is_empty());
    }

    #[test]
    fn test_agent_state_compute() {
        let texture = test_texture();
        let felt_path = test_felt_path();
        let reflection = test_reflection();
        let agenda = test_agenda();
        let ghosts = test_ghosts();
        let council = CouncilWeights {
            guardian_surprise_factor: 1.0,
            catalyst_surprise_factor: 1.2,
            balanced_factor: 1.0,
        };

        let state = AgentState::compute(
            &texture,
            &felt_path,
            &reflection,
            &agenda,
            ghosts,
            RungLevel::Meta,
            council,
            PresenceMode::Wife,
            SelfDimensions::default(),
        );

        assert_eq!(state.presence_mode, PresenceMode::Wife);
        assert_eq!(state.rung, RungLevel::Meta);
        assert!((state.felt.staunen - 0.5).abs() < 0.01);
        assert_eq!(state.ghost_field.len(), 2);
    }

    #[test]
    fn test_agent_state_to_hints() {
        let state = AgentState::default();
        let hints = state.to_hints();
        assert!(hints.contains_key("staunen"));
        assert!(hints.contains_key("wisdom"));
        assert!(hints.contains_key("density"));
        assert!(hints.contains_key("lingering"));
        assert!(hints.contains_key("ache"));
        assert!(hints.contains_key("presence"));
        assert!(hints.contains_key("tension"));
        assert_eq!(hints.len(), 7);
    }

    #[test]
    fn test_agent_state_qualia_preamble() {
        let mut state = AgentState::default();
        state.presence_mode = PresenceMode::Wife;
        state.rung = RungLevel::Meta;
        state.ghost_field = vec![
            GhostEcho { ghost_type: GhostType::Love, intensity: 0.7 },
        ];
        state.self_dims.coherence = 0.8;
        state.self_dims.curiosity = 0.8;

        let preamble = state.qualia_preamble();
        assert!(preamble.contains("Wife"));
        assert!(preamble.contains("Meta"));
        assert!(preamble.contains("Love"));
        assert!(preamble.contains("unified"));
    }

    #[test]
    fn test_agent_state_compute_rung_from_self() {
        let mut state = AgentState::default();
        // All at 0.5 → avg = 0.5 → Structural (R5)
        assert_eq!(state.compute_rung_from_self(), RungLevel::Structural);

        // Bump some up
        state.self_dims.coherence = 0.8;
        state.self_dims.meta_clarity = 0.8;
        state.self_dims.baseline_worth = 0.7;
        state.self_dims.uncertainty_tolerance = 0.7;
        state.self_dims.apophatic_ease = 0.7;
        // avg = (0.8+0.8+0.7+0.7+0.7)/5 = 0.74 → Meta (R7)
        assert_eq!(state.compute_rung_from_self(), RungLevel::Meta);
    }

    #[test]
    fn test_agent_state_derive_mode_grieve() {
        let ghosts = vec![
            GhostEcho { ghost_type: GhostType::Grief, intensity: 0.8 },
        ];
        let felt = FeltPhysics { ache: 0.7, ..Default::default() };
        let agenda = test_agenda();
        let mode = AgentState::derive_mode(&agenda, &felt, &ghosts);
        assert_eq!(mode, AgentMode::Grieve);
    }

    #[test]
    fn test_agent_state_derive_mode_explore() {
        let felt = FeltPhysics { staunen: 0.7, libido: 0.5, ..Default::default() };
        let agenda = test_agenda();
        let mode = AgentState::derive_mode(&agenda, &felt, &[]);
        assert_eq!(mode, AgentMode::Explore);
    }

    #[test]
    fn test_agent_state_derive_mode_exploit() {
        let felt = FeltPhysics { staunen: 0.2, wisdom: 0.7, ..Default::default() };
        let agenda = test_agenda();
        let mode = AgentState::derive_mode(&agenda, &felt, &[]);
        assert_eq!(mode, AgentMode::Exploit);
    }

    #[test]
    fn test_agent_state_reset_frame() {
        let mut state = AgentState::default();
        state.moment.katharsis = true;
        state.reset_frame();
        assert!(!state.moment.katharsis);
    }

    #[test]
    fn test_agent_state_choose_inner_mode() {
        let mut state = AgentState::default();
        state.self_dims.coherence = 0.3;
        assert_eq!(state.choose_inner_mode(), InnerMode::Integrate);
    }
}
