//! 7-Layer consciousness markers.
//!
//! Each layer has a 5-byte marker stored in Container 0 metadata at W12-W15:
//! (activation: u8, stability: u8, flags: u16, tag: u8)
//!
//! Read/write via [`MetaView::layer_marker(layer_index)`].

/// Type ID constants for the 7 consciousness layers.
pub const LAYER_SUBSTRATE: u16 = 0x0200;
pub const LAYER_FELT_CORE: u16 = 0x0201;
pub const LAYER_BODY: u16 = 0x0202;
pub const LAYER_QUALIA: u16 = 0x0203;
pub const LAYER_VOLITION: u16 = 0x0204;
pub const LAYER_GESTALT: u16 = 0x0205;
pub const LAYER_META: u16 = 0x0206;

/// Number of consciousness layers.
pub const NUM_LAYERS: usize = 7;

/// Layer names, indexed 0-6.
pub const LAYER_NAMES: [&str; NUM_LAYERS] = [
    "Substrate",
    "Felt Core",
    "Body",
    "Qualia",
    "Volition",
    "Gestalt",
    "Meta",
];
