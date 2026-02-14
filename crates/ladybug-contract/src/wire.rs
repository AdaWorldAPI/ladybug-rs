//! Binary Wire Protocol — CogPacket replaces JSON serialization.
//!
//! Internal communication between ladybug-rs, crewai-rust, and n8n-rs uses
//! bitpacked binary packets built on the Container primitive. No JSON.
//!
//! ```text
//! ┌──────────────────── CogPacket (1088 bytes) ────────────────────────┐
//! │                                                                     │
//! │  HEADER (64 bytes = 8 × u64)                                       │
//! │  ════════════════════════════                                       │
//! │  W0: magic(32) | version(8) | flags(8) | opcode(16)               │
//! │  W1: source_addr(16) | target_addr(16) | layer(4) | rung(4) | rsv │
//! │  W2: cycle(64) — monotonic processing cycle                        │
//! │  W3: nars_frequency(f32) | nars_confidence(f32)                    │
//! │  W4: satisfaction_packed(64) — 10 layers × 6 bits each             │
//! │  W5: field_modulation(64) — resonance_threshold(16) | fan_out(8)  │
//! │      | depth_bias(8) | noise_tolerance(8) | speed_bias(8) |        │
//! │      | exploration(8) | reserved(8)                                │
//! │  W6: timestamp_ns(64)                                              │
//! │  W7: checksum(32) | payload_containers(8) | reserved(24)          │
//! │                                                                     │
//! │  PAYLOAD (1024 bytes = 1 Container, or 2048 bytes = 2 Containers) │
//! │  ═════════════════════════════════════════════════════════════════  │
//! │  Container 0: Content / query fingerprint / execution result       │
//! │  Container 1: (Optional) Context / delta / modulation fingerprint  │
//! │                                                                     │
//! └─────────────────────────────────────────────────────────────────────┘
//!
//! Minimum packet: 64 (header) + 1024 (1 container) = 1088 bytes
//! Maximum packet: 64 (header) + 2048 (2 containers) = 2112 bytes
//!
//! This replaces V1StepDelegationRequest/Response with pure binary.
//! The 8+8 addressing (source_addr, target_addr) uses BindSpace's
//! prefix:slot scheme directly — no string parsing, no JSON.
//! ```

use crate::codebook::OpCategory;
use crate::container::{Container, CONTAINER_BYTES, CONTAINER_WORDS};
use crate::nars::TruthValue;

/// Magic bytes identifying a CogPacket.
pub const COG_MAGIC: u32 = 0x434F_4750; // "COGP"

/// Current wire protocol version.
pub const WIRE_VERSION: u8 = 1;

/// Minimum packet size: header + 1 container.
pub const MIN_PACKET_BYTES: usize = HEADER_BYTES + CONTAINER_BYTES;

/// Maximum packet size: header + 2 containers.
pub const MAX_PACKET_BYTES: usize = HEADER_BYTES + CONTAINER_BYTES * 2;

/// Header size in bytes.
pub const HEADER_BYTES: usize = 64;

/// Header size in u64 words.
pub const HEADER_WORDS: usize = HEADER_BYTES / 8;

// =============================================================================
// FLAGS
// =============================================================================

/// Packet is a response (otherwise request).
pub const FLAG_RESPONSE: u8 = 0x01;
/// Packet carries 2 containers (otherwise 1).
pub const FLAG_DUAL_CONTAINER: u8 = 0x02;
/// Request acknowledgment.
pub const FLAG_ACK_REQUESTED: u8 = 0x04;
/// Payload is SIMD-aligned (64-byte boundary).
pub const FLAG_SIMD_ALIGNED: u8 = 0x08;
/// Packet carries validation result (Pass/Hold/Reject in W7).
pub const FLAG_VALIDATED: u8 = 0x10;
/// Packet carries crystallization event.
pub const FLAG_CRYSTALLIZED: u8 = 0x20;
/// Error response.
pub const FLAG_ERROR: u8 = 0x40;
/// Packet is a delegation (cross-runtime).
pub const FLAG_DELEGATION: u8 = 0x80;

// =============================================================================
// OPCODES (12-bit, maps to CAM codebook categories)
// =============================================================================

/// Well-known wire opcodes for inter-runtime operations.
pub mod wire_ops {
    /// Resonance search (L1/L2) — query against BindSpace.
    pub const RESONATE: u16 = 0x300;
    /// Write to working memory (L5).
    pub const EXECUTE: u16 = 0x301;
    /// Delegate to another agent (L6).
    pub const DELEGATE: u16 = 0x302;
    /// Counterfactual query (L7).
    pub const COUNTERFACTUAL: u16 = 0x303;
    /// Evidence merge (L8).
    pub const INTEGRATE: u16 = 0x304;
    /// Validation request (L9).
    pub const VALIDATE: u16 = 0x305;
    /// Crystallization event (L10).
    pub const CRYSTALLIZE: u16 = 0x306;
    /// Style selection (L4).
    pub const ROUTE: u16 = 0x307;
    /// Collapse gate decision.
    pub const COLLAPSE: u16 = 0x308;
    /// Ping/health.
    pub const PING: u16 = 0xFFC;
    /// Pong/response.
    pub const PONG: u16 = 0xFFD;
}

// =============================================================================
// COGPACKET
// =============================================================================

/// Binary wire packet for inter-runtime communication.
///
/// Replaces JSON serialization with bitpacked binary.
/// Uses Container as the payload unit — same type used throughout the stack.
#[derive(Clone)]
pub struct CogPacket {
    /// Header words (8 × u64).
    pub header: [u64; HEADER_WORDS],
    /// Payload container(s). Always at least one; optionally two.
    pub payload: Vec<Container>,
}

impl CogPacket {
    // =========================================================================
    // CONSTRUCTORS
    // =========================================================================

    /// Create a request packet with a single container payload.
    pub fn request(
        opcode: u16,
        source_addr: u16,
        target_addr: u16,
        payload: Container,
    ) -> Self {
        let mut pkt = Self {
            header: [0u64; HEADER_WORDS],
            payload: vec![payload],
        };
        pkt.set_magic_version();
        pkt.set_opcode(opcode);
        pkt.set_source_addr(source_addr);
        pkt.set_target_addr(target_addr);
        pkt.set_payload_count(1);
        pkt.update_checksum();
        pkt
    }

    /// Create a response packet.
    pub fn response(
        opcode: u16,
        source_addr: u16,
        target_addr: u16,
        payload: Container,
    ) -> Self {
        let mut pkt = Self::request(opcode, source_addr, target_addr, payload);
        pkt.set_flags(pkt.flags() | FLAG_RESPONSE);
        pkt.update_checksum();
        pkt
    }

    /// Create a delegation packet (cross-runtime).
    pub fn delegation(
        opcode: u16,
        source_addr: u16,
        target_addr: u16,
        content: Container,
        context: Container,
    ) -> Self {
        let mut pkt = Self {
            header: [0u64; HEADER_WORDS],
            payload: vec![content, context],
        };
        pkt.set_magic_version();
        pkt.set_opcode(opcode);
        pkt.set_source_addr(source_addr);
        pkt.set_target_addr(target_addr);
        pkt.set_flags(FLAG_DELEGATION | FLAG_DUAL_CONTAINER);
        pkt.set_payload_count(2);
        pkt.update_checksum();
        pkt
    }

    // =========================================================================
    // HEADER ACCESSORS — W0: magic(32) | version(8) | flags(8) | opcode(16)
    // =========================================================================

    fn set_magic_version(&mut self) {
        self.header[0] = (COG_MAGIC as u64) << 32
            | (WIRE_VERSION as u64) << 24
            | (self.flags() as u64) << 16
            | (self.opcode() as u64);
    }

    /// Verify magic bytes.
    pub fn verify_magic(&self) -> bool {
        ((self.header[0] >> 32) as u32) == COG_MAGIC
    }

    /// Protocol version.
    pub fn version(&self) -> u8 {
        ((self.header[0] >> 24) & 0xFF) as u8
    }

    /// Flags byte.
    pub fn flags(&self) -> u8 {
        ((self.header[0] >> 16) & 0xFF) as u8
    }

    /// Set flags byte.
    pub fn set_flags(&mut self, flags: u8) {
        self.header[0] = (self.header[0] & !0x00FF_0000) | ((flags as u64) << 16);
    }

    /// 12-bit opcode (CAM codebook).
    pub fn opcode(&self) -> u16 {
        (self.header[0] & 0xFFFF) as u16
    }

    /// Set opcode.
    pub fn set_opcode(&mut self, opcode: u16) {
        self.header[0] = (self.header[0] & !0xFFFF) | (opcode as u64);
    }

    /// Opcode category.
    pub fn op_category(&self) -> OpCategory {
        OpCategory::from_id(self.opcode())
    }

    /// Is this a response packet?
    pub fn is_response(&self) -> bool {
        self.flags() & FLAG_RESPONSE != 0
    }

    /// Is this a delegation (cross-runtime)?
    pub fn is_delegation(&self) -> bool {
        self.flags() & FLAG_DELEGATION != 0
    }

    /// Is this an error response?
    pub fn is_error(&self) -> bool {
        self.flags() & FLAG_ERROR != 0
    }

    // =========================================================================
    // HEADER ACCESSORS — W1: source(16) | target(16) | layer(4) | rung(4) | rsv
    // =========================================================================

    /// Source address (8+8 BindSpace addressing).
    pub fn source_addr(&self) -> u16 {
        ((self.header[1] >> 48) & 0xFFFF) as u16
    }

    pub fn set_source_addr(&mut self, addr: u16) {
        self.header[1] = (self.header[1] & 0x0000_FFFF_FFFF_FFFF) | ((addr as u64) << 48);
    }

    /// Source prefix (high byte of 8+8 address).
    pub fn source_prefix(&self) -> u8 {
        (self.source_addr() >> 8) as u8
    }

    /// Source slot (low byte of 8+8 address).
    pub fn source_slot(&self) -> u8 {
        (self.source_addr() & 0xFF) as u8
    }

    /// Target address (8+8 BindSpace addressing).
    pub fn target_addr(&self) -> u16 {
        ((self.header[1] >> 32) & 0xFFFF) as u16
    }

    pub fn set_target_addr(&mut self, addr: u16) {
        self.header[1] = (self.header[1] & 0xFFFF_0000_FFFF_FFFF)
            | ((addr as u64) << 32);
    }

    /// Target prefix.
    pub fn target_prefix(&self) -> u8 {
        (self.target_addr() >> 8) as u8
    }

    /// Target slot.
    pub fn target_slot(&self) -> u8 {
        (self.target_addr() & 0xFF) as u8
    }

    /// Dominant cognitive layer (0-9 → L1-L10).
    pub fn layer(&self) -> u8 {
        ((self.header[1] >> 28) & 0xF) as u8
    }

    pub fn set_layer(&mut self, layer: u8) {
        self.header[1] = (self.header[1] & !(0xF << 28)) | (((layer & 0xF) as u64) << 28);
    }

    /// Pearl's causal rung (0-3: none/see/do/imagine).
    pub fn rung(&self) -> u8 {
        ((self.header[1] >> 24) & 0xF) as u8
    }

    pub fn set_rung(&mut self, rung: u8) {
        self.header[1] = (self.header[1] & !(0xF << 24)) | (((rung & 0xF) as u64) << 24);
    }

    // =========================================================================
    // HEADER ACCESSORS — W2: cycle(64)
    // =========================================================================

    /// Processing cycle (monotonic).
    pub fn cycle(&self) -> u64 {
        self.header[2]
    }

    pub fn set_cycle(&mut self, cycle: u64) {
        self.header[2] = cycle;
    }

    // =========================================================================
    // HEADER ACCESSORS — W3: nars_frequency(f32) | nars_confidence(f32)
    // =========================================================================

    /// NARS truth value encoded in W3.
    pub fn truth_value(&self) -> TruthValue {
        let freq = f32::from_bits((self.header[3] >> 32) as u32);
        let conf = f32::from_bits((self.header[3] & 0xFFFF_FFFF) as u32);
        TruthValue::new(freq, conf)
    }

    pub fn set_truth_value(&mut self, tv: &TruthValue) {
        self.header[3] = ((tv.frequency.to_bits() as u64) << 32)
            | (tv.confidence.to_bits() as u64);
    }

    // =========================================================================
    // HEADER ACCESSORS — W4: satisfaction_packed (10 × 6 bits = 60 bits)
    // =========================================================================

    /// Get satisfaction score for a layer (0-9). Returns 0.0-1.0.
    pub fn satisfaction(&self, layer: u8) -> f32 {
        let shift = (layer as u32) * 6;
        let raw = ((self.header[4] >> shift) & 0x3F) as f32;
        raw / 63.0
    }

    /// Set satisfaction score for a layer (0-9).
    pub fn set_satisfaction(&mut self, layer: u8, score: f32) {
        let quantized = ((score.clamp(0.0, 1.0) * 63.0) as u64) & 0x3F;
        let shift = (layer as u32) * 6;
        self.header[4] = (self.header[4] & !(0x3F << shift)) | (quantized << shift);
    }

    /// Get all 10 satisfaction scores as an array.
    pub fn satisfaction_array(&self) -> [f32; 10] {
        let mut scores = [0.0f32; 10];
        for i in 0..10 {
            scores[i] = self.satisfaction(i as u8);
        }
        scores
    }

    /// Set all 10 satisfaction scores.
    pub fn set_satisfaction_array(&mut self, scores: &[f32; 10]) {
        self.header[4] = 0;
        for (i, &s) in scores.iter().enumerate() {
            self.set_satisfaction(i as u8, s);
        }
    }

    // =========================================================================
    // HEADER ACCESSORS — W5: field_modulation
    // =========================================================================

    /// Field modulation: resonance threshold (0.0-1.0, 16-bit precision).
    pub fn resonance_threshold(&self) -> f32 {
        let raw = ((self.header[5] >> 48) & 0xFFFF) as f32;
        raw / 65535.0
    }

    pub fn set_resonance_threshold(&mut self, threshold: f32) {
        let quantized = ((threshold.clamp(0.0, 1.0) * 65535.0) as u64) & 0xFFFF;
        self.header[5] = (self.header[5] & 0x0000_FFFF_FFFF_FFFF)
            | (quantized << 48);
    }

    /// Fan-out degree (0-255).
    pub fn fan_out(&self) -> u8 {
        ((self.header[5] >> 40) & 0xFF) as u8
    }

    pub fn set_fan_out(&mut self, fan_out: u8) {
        self.header[5] = (self.header[5] & !(0xFF << 40)) | ((fan_out as u64) << 40);
    }

    /// Depth bias (0.0-1.0, 8-bit).
    pub fn depth_bias(&self) -> f32 {
        ((self.header[5] >> 32) & 0xFF) as f32 / 255.0
    }

    pub fn set_depth_bias(&mut self, bias: f32) {
        let q = ((bias.clamp(0.0, 1.0) * 255.0) as u64) & 0xFF;
        self.header[5] = (self.header[5] & !(0xFF << 32)) | (q << 32);
    }

    /// Noise tolerance (0.0-1.0, 8-bit).
    pub fn noise_tolerance(&self) -> f32 {
        ((self.header[5] >> 24) & 0xFF) as f32 / 255.0
    }

    pub fn set_noise_tolerance(&mut self, tol: f32) {
        let q = ((tol.clamp(0.0, 1.0) * 255.0) as u64) & 0xFF;
        self.header[5] = (self.header[5] & !(0xFF << 24)) | (q << 24);
    }

    /// Speed bias (0.0-1.0, 8-bit).
    pub fn speed_bias(&self) -> f32 {
        ((self.header[5] >> 16) & 0xFF) as f32 / 255.0
    }

    pub fn set_speed_bias(&mut self, bias: f32) {
        let q = ((bias.clamp(0.0, 1.0) * 255.0) as u64) & 0xFF;
        self.header[5] = (self.header[5] & !(0xFF << 16)) | (q << 16);
    }

    /// Exploration factor (0.0-1.0, 8-bit).
    pub fn exploration(&self) -> f32 {
        ((self.header[5] >> 8) & 0xFF) as f32 / 255.0
    }

    pub fn set_exploration(&mut self, expl: f32) {
        let q = ((expl.clamp(0.0, 1.0) * 255.0) as u64) & 0xFF;
        self.header[5] = (self.header[5] & !(0xFF << 8)) | (q << 8);
    }

    // =========================================================================
    // HEADER ACCESSORS — W6: timestamp_ns
    // =========================================================================

    /// Timestamp in nanoseconds (monotonic clock).
    pub fn timestamp_ns(&self) -> u64 {
        self.header[6]
    }

    pub fn set_timestamp_ns(&mut self, ts: u64) {
        self.header[6] = ts;
    }

    // =========================================================================
    // HEADER ACCESSORS — W7: checksum(32) | payload_count(8) | reserved
    // =========================================================================

    /// CRC32-like checksum of header (words 0-6).
    pub fn checksum(&self) -> u32 {
        ((self.header[7] >> 32) & 0xFFFF_FFFF) as u32
    }

    /// Number of payload containers (1 or 2).
    pub fn payload_count(&self) -> u8 {
        ((self.header[7] >> 24) & 0xFF) as u8
    }

    fn set_payload_count(&mut self, count: u8) {
        self.header[7] = (self.header[7] & !(0xFF << 24)) | ((count as u64) << 24);
    }

    /// Compute and store checksum.
    pub fn update_checksum(&mut self) {
        let sum = self.compute_checksum();
        self.header[7] = (self.header[7] & 0x0000_0000_FFFF_FFFF) | ((sum as u64) << 32);
    }

    /// Verify checksum.
    pub fn verify_checksum(&self) -> bool {
        self.checksum() == self.compute_checksum()
    }

    fn compute_checksum(&self) -> u32 {
        // Simple XOR-fold of header words 0-6
        let mut h = 0u64;
        for i in 0..7 {
            h ^= self.header[i];
        }
        // Mix the halves
        let lo = h as u32;
        let hi = (h >> 32) as u32;
        lo ^ hi ^ lo.rotate_left(7) ^ hi.rotate_right(13)
    }

    // =========================================================================
    // PAYLOAD ACCESS
    // =========================================================================

    /// Primary payload container (always present).
    pub fn content(&self) -> &Container {
        &self.payload[0]
    }

    /// Mutable primary payload.
    pub fn content_mut(&mut self) -> &mut Container {
        &mut self.payload[0]
    }

    /// Context container (present when FLAG_DUAL_CONTAINER is set).
    pub fn context(&self) -> Option<&Container> {
        if self.payload.len() > 1 {
            Some(&self.payload[1])
        } else {
            None
        }
    }

    /// Mutable context container.
    pub fn context_mut(&mut self) -> Option<&mut Container> {
        if self.payload.len() > 1 {
            Some(&mut self.payload[1])
        } else {
            None
        }
    }

    // =========================================================================
    // SERIALIZATION — ZERO-COPY BINARY
    // =========================================================================

    /// Encode to raw bytes (header + payload containers).
    /// No JSON. No base64. Just u64 words in little-endian.
    pub fn encode(&self) -> Vec<u8> {
        let n_containers = self.payload.len();
        let total_bytes = HEADER_BYTES + n_containers * CONTAINER_BYTES;
        let mut buf = Vec::with_capacity(total_bytes);

        // Header: 8 × u64, little-endian
        for &w in &self.header {
            buf.extend_from_slice(&w.to_le_bytes());
        }

        // Payload containers: 128 × u64 each, little-endian
        for container in &self.payload {
            for &w in &container.words {
                buf.extend_from_slice(&w.to_le_bytes());
            }
        }

        buf
    }

    /// Decode from raw bytes.
    pub fn decode(data: &[u8]) -> Result<Self, WireError> {
        if data.len() < MIN_PACKET_BYTES {
            return Err(WireError::TooShort {
                expected: MIN_PACKET_BYTES,
                got: data.len(),
            });
        }

        // Decode header
        let mut header = [0u64; HEADER_WORDS];
        for (i, chunk) in data[..HEADER_BYTES].chunks_exact(8).enumerate() {
            header[i] = u64::from_le_bytes(chunk.try_into().unwrap());
        }

        // Verify magic
        let magic = ((header[0] >> 32) & 0xFFFF_FFFF) as u32;
        if magic != COG_MAGIC {
            return Err(WireError::BadMagic { got: magic });
        }

        // Verify version
        let version = ((header[0] >> 24) & 0xFF) as u8;
        if version != WIRE_VERSION {
            return Err(WireError::VersionMismatch {
                expected: WIRE_VERSION,
                got: version,
            });
        }

        // Determine number of containers from header
        let payload_count = ((header[7] >> 24) & 0xFF) as usize;
        let payload_count = payload_count.max(1).min(2);
        let expected_bytes = HEADER_BYTES + payload_count * CONTAINER_BYTES;

        if data.len() < expected_bytes {
            return Err(WireError::TooShort {
                expected: expected_bytes,
                got: data.len(),
            });
        }

        // Decode payload containers
        let mut payload = Vec::with_capacity(payload_count);
        for c_idx in 0..payload_count {
            let offset = HEADER_BYTES + c_idx * CONTAINER_BYTES;
            let mut words = [0u64; CONTAINER_WORDS];
            for (i, chunk) in data[offset..offset + CONTAINER_BYTES]
                .chunks_exact(8)
                .enumerate()
            {
                words[i] = u64::from_le_bytes(chunk.try_into().unwrap());
            }
            payload.push(Container { words });
        }

        let pkt = Self { header, payload };

        // Verify checksum
        if !pkt.verify_checksum() {
            return Err(WireError::ChecksumMismatch);
        }

        Ok(pkt)
    }

    /// Total encoded size in bytes.
    pub fn encoded_size(&self) -> usize {
        HEADER_BYTES + self.payload.len() * CONTAINER_BYTES
    }
}

impl std::fmt::Debug for CogPacket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CogPacket(op={:#05x}, src={:#06x}, tgt={:#06x}, layer={}, cycle={}, containers={})",
            self.opcode(),
            self.source_addr(),
            self.target_addr(),
            self.layer(),
            self.cycle(),
            self.payload.len(),
        )
    }
}

// =============================================================================
// ERRORS
// =============================================================================

/// Wire protocol errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WireError {
    /// Packet too short.
    TooShort { expected: usize, got: usize },
    /// Invalid magic bytes.
    BadMagic { got: u32 },
    /// Version mismatch.
    VersionMismatch { expected: u8, got: u8 },
    /// Checksum verification failed.
    ChecksumMismatch,
}

impl std::fmt::Display for WireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooShort { expected, got } => {
                write!(f, "packet too short: expected {expected}, got {got}")
            }
            Self::BadMagic { got } => write!(f, "bad magic: {got:#010x}"),
            Self::VersionMismatch { expected, got } => {
                write!(f, "version mismatch: expected {expected}, got {got}")
            }
            Self::ChecksumMismatch => write!(f, "checksum mismatch"),
        }
    }
}

impl std::error::Error for WireError {}

// =============================================================================
// CONVERSION FROM LEGACY V1 TYPES
// =============================================================================

/// Convert a legacy V1 delegation request to a CogPacket.
///
/// This is the migration path: existing crewai-rust/n8n-rs code can convert
/// V1 JSON envelopes to CogPackets at the boundary, then use pure binary
/// internally.
impl CogPacket {
    /// Create from V1 delegation data (migration helper).
    ///
    /// The step_type string is converted to an opcode and 8+8 address:
    /// - "crew.*" → PREFIX_AGENTS(0x0C) + opcode DELEGATE
    /// - "lb.*" → PREFIX_CAUSAL(0x05) + opcode RESONATE
    /// - "n8n.*" → PREFIX_A2A(0x0F) + opcode EXECUTE
    pub fn from_step_type(step_type: &str, content_hash: u64) -> Self {
        let (prefix, opcode) = match step_type.split('.').next() {
            Some("crew") => (0x0Cu8, wire_ops::DELEGATE),
            Some("lb") => (0x05u8, wire_ops::RESONATE),
            Some("n8n") => (0x0Fu8, wire_ops::EXECUTE),
            _ => (0x0Fu8, wire_ops::EXECUTE),
        };

        let source_addr = ((prefix as u16) << 8) | 0x00;
        let target_addr = ((prefix as u16) << 8) | 0x01;

        // Content hash expanded to container via SplitMix64
        let container = Container::random(content_hash);

        Self::request(opcode, source_addr, target_addr, container)
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packet_roundtrip() {
        let content = Container::random(42);
        let pkt = CogPacket::request(wire_ops::RESONATE, 0x0500, 0x8042, content.clone());

        let bytes = pkt.encode();
        assert_eq!(bytes.len(), MIN_PACKET_BYTES);

        let decoded = CogPacket::decode(&bytes).unwrap();
        assert!(decoded.verify_magic());
        assert_eq!(decoded.opcode(), wire_ops::RESONATE);
        assert_eq!(decoded.source_addr(), 0x0500);
        assert_eq!(decoded.target_addr(), 0x8042);
        assert_eq!(decoded.source_prefix(), 0x05);
        assert_eq!(decoded.target_slot(), 0x42);
        assert_eq!(decoded.content(), &content);
    }

    #[test]
    fn test_dual_container_delegation() {
        let content = Container::random(1);
        let context = Container::random(2);
        let pkt = CogPacket::delegation(
            wire_ops::DELEGATE,
            0x0C00,
            0x0C01,
            content.clone(),
            context.clone(),
        );

        let bytes = pkt.encode();
        assert_eq!(bytes.len(), MAX_PACKET_BYTES);

        let decoded = CogPacket::decode(&bytes).unwrap();
        assert!(decoded.is_delegation());
        assert_eq!(decoded.payload.len(), 2);
        assert_eq!(decoded.content(), &content);
        assert_eq!(decoded.context().unwrap(), &context);
    }

    #[test]
    fn test_satisfaction_packing() {
        let mut pkt = CogPacket::request(wire_ops::RESONATE, 0, 0, Container::zero());
        let scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        pkt.set_satisfaction_array(&scores);

        for i in 0..10 {
            let recovered = pkt.satisfaction(i as u8);
            assert!(
                (recovered - scores[i]).abs() < 0.02,
                "layer {i}: expected {}, got {recovered}",
                scores[i]
            );
        }
    }

    #[test]
    fn test_truth_value_roundtrip() {
        let mut pkt = CogPacket::request(wire_ops::VALIDATE, 0, 0, Container::zero());
        let tv = TruthValue::new(0.85, 0.92);
        pkt.set_truth_value(&tv);

        let recovered = pkt.truth_value();
        assert!((recovered.frequency - 0.85).abs() < 1e-6);
        assert!((recovered.confidence - 0.92).abs() < 1e-6);
    }

    #[test]
    fn test_field_modulation_packing() {
        let mut pkt = CogPacket::request(wire_ops::ROUTE, 0, 0, Container::zero());
        pkt.set_resonance_threshold(0.85);
        pkt.set_fan_out(12);
        pkt.set_depth_bias(0.9);
        pkt.set_noise_tolerance(0.1);
        pkt.set_speed_bias(0.5);
        pkt.set_exploration(0.35);

        assert!((pkt.resonance_threshold() - 0.85).abs() < 0.001);
        assert_eq!(pkt.fan_out(), 12);
        assert!((pkt.depth_bias() - 0.9).abs() < 0.01);
        assert!((pkt.noise_tolerance() - 0.1).abs() < 0.01);
        assert!((pkt.speed_bias() - 0.5).abs() < 0.01);
        assert!((pkt.exploration() - 0.35).abs() < 0.01);
    }

    #[test]
    fn test_layer_and_rung() {
        let mut pkt = CogPacket::request(wire_ops::COUNTERFACTUAL, 0, 0, Container::zero());
        pkt.set_layer(7); // L7 Contingency
        pkt.set_rung(3); // Imagine

        assert_eq!(pkt.layer(), 7);
        assert_eq!(pkt.rung(), 3);
    }

    #[test]
    fn test_bad_magic_rejected() {
        let mut bytes = CogPacket::request(wire_ops::PING, 0, 0, Container::zero()).encode();
        // Corrupt magic
        bytes[4] = 0xFF;
        assert!(matches!(
            CogPacket::decode(&bytes),
            Err(WireError::BadMagic { .. })
        ));
    }

    #[test]
    fn test_too_short_rejected() {
        let bytes = vec![0u8; 10];
        assert!(matches!(
            CogPacket::decode(&bytes),
            Err(WireError::TooShort { .. })
        ));
    }

    #[test]
    fn test_checksum_verification() {
        let pkt = CogPacket::request(wire_ops::EXECUTE, 0x0C00, 0x8001, Container::random(99));
        assert!(pkt.verify_checksum());

        let mut bytes = pkt.encode();
        // Corrupt a header word
        bytes[16] ^= 0xFF;
        assert!(matches!(
            CogPacket::decode(&bytes),
            Err(WireError::ChecksumMismatch)
        ));
    }

    #[test]
    fn test_response_flag() {
        let pkt = CogPacket::response(wire_ops::PONG, 0, 0, Container::zero());
        assert!(pkt.is_response());
        assert!(!pkt.is_delegation());
        assert!(!pkt.is_error());
    }

    #[test]
    fn test_from_step_type() {
        let pkt = CogPacket::from_step_type("crew.agent", 12345);
        assert_eq!(pkt.opcode(), wire_ops::DELEGATE);
        assert_eq!(pkt.source_prefix(), 0x0C);

        let pkt = CogPacket::from_step_type("lb.resonate", 67890);
        assert_eq!(pkt.opcode(), wire_ops::RESONATE);
        assert_eq!(pkt.source_prefix(), 0x05);
    }

    #[test]
    fn test_op_category() {
        let pkt = CogPacket::request(wire_ops::RESONATE, 0, 0, Container::zero());
        assert_eq!(pkt.op_category(), OpCategory::Hamming);
    }
}
