//! 10-Layer cognitive stack markers.
//!
//! Each layer has a 3-byte marker stored in Container 0 metadata at W12-W15:
//! (activation: u8, stability: u8, flags: u8)
//!
//! Read/write via [`MetaView::layer_marker(layer_index)`].
//!
//! ## 10-Layer Cognitive Stack
//!
//! ```text
//! L1  Recognition     — pattern matching, fingerprint encoding
//! L2  Resonance       — field binding, similarity search
//! L3  Appraisal       — gestalt, hypothesis, evaluation
//! L4  Routing         — branch selection, template dispatch
//! L5  Execution       — active manipulation, synthesis
//!     ─── single agent boundary ───
//! L6  Delegation      — cognitive fan-out, multi-agent
//! L7  Contingency     — cross-branch, could-be-otherwise
//! L8  Integration     — evidence merge, meta-awareness
//! L9  Validation      — NARS, Brier, Socratic sieve
//! L10 Crystallization — what survives becomes system
//! ```

/// Type ID constants for the 10 cognitive layers.
pub const LAYER_RECOGNITION: u16 = 0x0200;     // L1
pub const LAYER_RESONANCE: u16 = 0x0201;       // L2
pub const LAYER_APPRAISAL: u16 = 0x0202;       // L3
pub const LAYER_ROUTING: u16 = 0x0203;         // L4
pub const LAYER_EXECUTION: u16 = 0x0204;       // L5
pub const LAYER_DELEGATION: u16 = 0x0205;      // L6
pub const LAYER_CONTINGENCY: u16 = 0x0206;     // L7
pub const LAYER_INTEGRATION: u16 = 0x0207;     // L8
pub const LAYER_VALIDATION: u16 = 0x0208;      // L9
pub const LAYER_CRYSTALLIZATION: u16 = 0x0209; // L10

/// Number of cognitive layers.
pub const NUM_LAYERS: usize = 10;

/// Layer names, indexed 0-9.
pub const LAYER_NAMES: [&str; NUM_LAYERS] = [
    "Recognition",
    "Resonance",
    "Appraisal",
    "Routing",
    "Execution",
    "Delegation",
    "Contingency",
    "Integration",
    "Validation",
    "Crystallization",
];

// Deprecation aliases for old 7-layer names.
// These keep the same type ID values (0x0200-0x0206) so existing
// on-disk records remain readable.
#[deprecated(note = "use LAYER_RECOGNITION")]
pub const LAYER_SUBSTRATE: u16 = LAYER_RECOGNITION;
#[deprecated(note = "use LAYER_RESONANCE")]
pub const LAYER_FELT_CORE: u16 = LAYER_RESONANCE;
#[deprecated(note = "use LAYER_APPRAISAL")]
pub const LAYER_BODY: u16 = LAYER_APPRAISAL;
#[deprecated(note = "use LAYER_ROUTING")]
pub const LAYER_QUALIA: u16 = LAYER_ROUTING;
#[deprecated(note = "use LAYER_EXECUTION")]
pub const LAYER_VOLITION: u16 = LAYER_EXECUTION;
#[deprecated(note = "use LAYER_DELEGATION")]
pub const LAYER_GESTALT: u16 = LAYER_DELEGATION;
#[deprecated(note = "use LAYER_CONTINGENCY")]
pub const LAYER_META: u16 = LAYER_CONTINGENCY;
