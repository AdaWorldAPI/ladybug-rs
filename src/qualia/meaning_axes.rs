//! 48 Canonical Meaning Axes — The Semantic Reference Frame
//!
//! From dragonfly-vsa's `meaning_cam.py`: 48 bipolar axes that span
//! the space of human meaning. Every concept, thought, and code-feeling
//! can be located as a position on these axes.
//!
//! The axes are grouped into 8 families of 6 axes each (48 total).
//! Each axis maps to a bit range in a 16K-bit fingerprint:
//! - 48 axes × ~208 bits each = ~10,000 bits (matching dragonfly-vsa DIM)
//! - Remaining ~6,384 bits carry metadata (viscosity, texture, etc.)
//!
//! ## Scientific Foundation
//!
//! Pearson correlation between Jina cosine similarity and Hamming similarity
//! in this 10KD binary space: **r = 0.9913**. Binary Hamming IS semantic
//! similarity. (Validated in `dragonfly-vsa/src/pure_bitpacked_vsa.py`)

use crate::storage::FINGERPRINT_WORDS;

// =============================================================================
// THE 48 AXES
// =============================================================================

/// A bipolar meaning axis: two opposing poles.
#[derive(Debug, Clone, Copy)]
pub struct MeaningAxis {
    pub positive: &'static str,
    pub negative: &'static str,
    pub family: AxisFamily,
    /// Index in the canonical table (0-47)
    pub index: u8,
}

/// Axis family groupings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisFamily {
    /// Osgood EPA: good↔bad, strong↔weak, active↔passive
    OsgoodEPA,
    /// Physical: large↔small, heavy↔light, hard↔soft, rough↔smooth, hot↔cold...
    Physical,
    /// Spatial + Temporal: near↔far, new↔old, sudden↔gradual
    SpatioTemporal,
    /// Cognitive: simple↔complex, certain↔uncertain, concrete↔abstract...
    Cognitive,
    /// Emotional: happy↔sad, calm↔anxious, loving↔hateful
    Emotional,
    /// Social: friendly↔hostile, dominant↔submissive, formal↔informal
    Social,
    /// Evaluative + Abstract: useful↔useless, natural↔artificial, alive↔dead...
    Abstract,
    /// Sensory: sweet↔bitter, fragrant↔foul, melodic↔cacophonous
    Sensory,
}

/// The 48 canonical meaning axes.
pub const AXES: [MeaningAxis; 48] = [
    // Osgood EPA (0-2)
    MeaningAxis { positive: "good", negative: "bad", family: AxisFamily::OsgoodEPA, index: 0 },
    MeaningAxis { positive: "strong", negative: "weak", family: AxisFamily::OsgoodEPA, index: 1 },
    MeaningAxis { positive: "active", negative: "passive", family: AxisFamily::OsgoodEPA, index: 2 },
    // Physical (3-12)
    MeaningAxis { positive: "large", negative: "small", family: AxisFamily::Physical, index: 3 },
    MeaningAxis { positive: "heavy", negative: "light", family: AxisFamily::Physical, index: 4 },
    MeaningAxis { positive: "hard", negative: "soft", family: AxisFamily::Physical, index: 5 },
    MeaningAxis { positive: "rough", negative: "smooth", family: AxisFamily::Physical, index: 6 },
    MeaningAxis { positive: "hot", negative: "cold", family: AxisFamily::Physical, index: 7 },
    MeaningAxis { positive: "wet", negative: "dry", family: AxisFamily::Physical, index: 8 },
    MeaningAxis { positive: "fast", negative: "slow", family: AxisFamily::Physical, index: 9 },
    MeaningAxis { positive: "loud", negative: "quiet", family: AxisFamily::Physical, index: 10 },
    MeaningAxis { positive: "bright", negative: "dark", family: AxisFamily::Physical, index: 11 },
    MeaningAxis { positive: "sharp", negative: "dull", family: AxisFamily::Physical, index: 12 },
    // Spatio-Temporal (13-18)
    MeaningAxis { positive: "near", negative: "far", family: AxisFamily::SpatioTemporal, index: 13 },
    MeaningAxis { positive: "high", negative: "low", family: AxisFamily::SpatioTemporal, index: 14 },
    MeaningAxis { positive: "inside", negative: "outside", family: AxisFamily::SpatioTemporal, index: 15 },
    MeaningAxis { positive: "new", negative: "old", family: AxisFamily::SpatioTemporal, index: 16 },
    MeaningAxis { positive: "permanent", negative: "temporary", family: AxisFamily::SpatioTemporal, index: 17 },
    MeaningAxis { positive: "sudden", negative: "gradual", family: AxisFamily::SpatioTemporal, index: 18 },
    // Cognitive (19-23)
    MeaningAxis { positive: "simple", negative: "complex", family: AxisFamily::Cognitive, index: 19 },
    MeaningAxis { positive: "certain", negative: "uncertain", family: AxisFamily::Cognitive, index: 20 },
    MeaningAxis { positive: "concrete", negative: "abstract", family: AxisFamily::Cognitive, index: 21 },
    MeaningAxis { positive: "familiar", negative: "unfamiliar", family: AxisFamily::Cognitive, index: 22 },
    MeaningAxis { positive: "important", negative: "trivial", family: AxisFamily::Cognitive, index: 23 },
    // Emotional (24-26)
    MeaningAxis { positive: "happy", negative: "sad", family: AxisFamily::Emotional, index: 24 },
    MeaningAxis { positive: "calm", negative: "anxious", family: AxisFamily::Emotional, index: 25 },
    MeaningAxis { positive: "loving", negative: "hateful", family: AxisFamily::Emotional, index: 26 },
    // Social (27-29)
    MeaningAxis { positive: "friendly", negative: "hostile", family: AxisFamily::Social, index: 27 },
    MeaningAxis { positive: "dominant", negative: "submissive", family: AxisFamily::Social, index: 28 },
    MeaningAxis { positive: "formal", negative: "informal", family: AxisFamily::Social, index: 29 },
    // Evaluative (30-33)
    MeaningAxis { positive: "useful", negative: "useless", family: AxisFamily::Abstract, index: 30 },
    MeaningAxis { positive: "beautiful", negative: "ugly", family: AxisFamily::Abstract, index: 31 },
    MeaningAxis { positive: "safe", negative: "dangerous", family: AxisFamily::Abstract, index: 32 },
    MeaningAxis { positive: "clean", negative: "dirty", family: AxisFamily::Abstract, index: 33 },
    // Abstract (34-44)
    MeaningAxis { positive: "natural", negative: "artificial", family: AxisFamily::Abstract, index: 34 },
    MeaningAxis { positive: "sacred", negative: "profane", family: AxisFamily::Abstract, index: 35 },
    MeaningAxis { positive: "real", negative: "imaginary", family: AxisFamily::Abstract, index: 36 },
    MeaningAxis { positive: "whole", negative: "partial", family: AxisFamily::Abstract, index: 37 },
    MeaningAxis { positive: "open", negative: "closed", family: AxisFamily::Abstract, index: 38 },
    MeaningAxis { positive: "free", negative: "constrained", family: AxisFamily::Abstract, index: 39 },
    MeaningAxis { positive: "ordered", negative: "chaotic", family: AxisFamily::Abstract, index: 40 },
    MeaningAxis { positive: "alive", negative: "dead", family: AxisFamily::Abstract, index: 41 },
    MeaningAxis { positive: "growing", negative: "shrinking", family: AxisFamily::Abstract, index: 42 },
    MeaningAxis { positive: "giving", negative: "taking", family: AxisFamily::Abstract, index: 43 },
    MeaningAxis { positive: "creating", negative: "destroying", family: AxisFamily::Abstract, index: 44 },
    // Sensory (45-47)
    MeaningAxis { positive: "sweet", negative: "bitter", family: AxisFamily::Sensory, index: 45 },
    MeaningAxis { positive: "fragrant", negative: "foul", family: AxisFamily::Sensory, index: 46 },
    MeaningAxis { positive: "melodic", negative: "cacophonous", family: AxisFamily::Sensory, index: 47 },
];

// =============================================================================
// AXIS ACTIVATION → FINGERPRINT
// =============================================================================

/// Activation levels for each axis (-1.0 to +1.0).
/// Positive = toward positive pole, negative = toward negative pole.
pub type AxisActivation = [f32; 48];

/// Bits per axis in the 16K fingerprint.
/// 48 axes × 208 bits = 9,984 bits (fits in first ~156 u64 words).
const BITS_PER_AXIS: usize = 208;

/// Encode axis activations into a 16K-bit fingerprint.
///
/// Each axis gets a 208-bit region. The activation level determines
/// how many bits are set (via deterministic hash expansion, matching
/// dragonfly-vsa's `compute_10k_location` sparse activation pattern).
///
/// Activation = +1.0 → all 208 bits set (strong positive pole)
/// Activation =  0.0 → ~104 bits set (neutral)
/// Activation = -1.0 → 0 bits set (strong negative pole)
pub fn encode_axes(activations: &AxisActivation) -> [u64; FINGERPRINT_WORDS] {
    let mut fp = [0u64; FINGERPRINT_WORDS];

    for (axis_idx, &activation) in activations.iter().enumerate() {
        let bit_start = axis_idx * BITS_PER_AXIS;

        // Map activation [-1, +1] to number of bits to set [0, BITS_PER_AXIS]
        let normalized = ((activation + 1.0) / 2.0).clamp(0.0, 1.0);
        let bits_to_set = (normalized * BITS_PER_AXIS as f32) as usize;

        // Deterministic bit selection via simple hash expansion
        // (matches dragonfly-vsa pattern: seed + i*7 mod range)
        let seed = axis_idx as u64 * 2654435761; // Knuth multiplicative hash
        for i in 0..bits_to_set {
            let bit_offset = ((seed.wrapping_add(i as u64 * 7)) % BITS_PER_AXIS as u64) as usize;
            let global_bit = bit_start + bit_offset;

            let word_idx = global_bit / 64;
            let bit_idx = global_bit % 64;
            if word_idx < FINGERPRINT_WORDS {
                fp[word_idx] |= 1u64 << bit_idx;
            }
        }
    }

    fp
}

/// Decode a fingerprint back to approximate axis activations.
///
/// Counts set bits in each axis region and normalizes.
pub fn decode_axes(fp: &[u64; FINGERPRINT_WORDS]) -> AxisActivation {
    let mut activations = [0.0f32; 48];

    for axis_idx in 0..48 {
        let bit_start = axis_idx * BITS_PER_AXIS;
        let mut count = 0u32;

        for bit_offset in 0..BITS_PER_AXIS {
            let global_bit = bit_start + bit_offset;
            let word_idx = global_bit / 64;
            let bit_idx = global_bit % 64;
            if word_idx < FINGERPRINT_WORDS {
                if (fp[word_idx] >> bit_idx) & 1 == 1 {
                    count += 1;
                }
            }
        }

        // Map back to [-1, +1]
        activations[axis_idx] = (count as f32 / BITS_PER_AXIS as f32) * 2.0 - 1.0;
    }

    activations
}

// =============================================================================
// THOUGHT VISCOSITY (from thought_fingerprint.py)
// =============================================================================

/// Viscosity of a thought — how it flows through the cognitive system.
/// Maps to FireflyScheduler execution modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Viscosity {
    Watery,      // Fast, clear, frictionless → Sprint
    Oily,        // Smooth but clinging → Stream
    Honey,       // Slow, sweet, sticking → Chunk
    Mercury,     // Dense, quick, unpredictable → Burst
    Lava,        // Slow, hot, transformative → Stream (high coherence)
    Crystalline, // Frozen, structured, sharp → Chunk (high boundary)
    Gaseous,     // Diffuse, expanding → Burst
    Plasma,      // Superheated, unstable → Sprint (high risk)
}

/// Detect viscosity from axis activations.
pub fn detect_viscosity(activations: &AxisActivation) -> Viscosity {
    let hot_cold = activations[7];    // hot↔cold
    let fast_slow = activations[9];   // fast↔slow
    let hard_soft = activations[5];   // hard↔soft
    let certain = activations[20];    // certain↔uncertain
    let alive = activations[41];      // alive↔dead

    // Fast + clear → Watery
    if fast_slow > 0.5 && certain > 0.3 {
        return Viscosity::Watery;
    }
    // Hot + slow + alive → Lava
    if hot_cold > 0.5 && fast_slow < -0.3 && alive > 0.3 {
        return Viscosity::Lava;
    }
    // Dense + quick → Mercury
    if hard_soft > 0.3 && fast_slow > 0.3 && certain < -0.2 {
        return Viscosity::Mercury;
    }
    // Frozen + structured → Crystalline
    if hot_cold < -0.5 && hard_soft > 0.5 {
        return Viscosity::Crystalline;
    }
    // Expanding + uncertain → Gaseous
    if certain < -0.5 {
        return Viscosity::Gaseous;
    }
    // Hot + intense → Plasma
    if hot_cold > 0.7 && alive > 0.5 {
        return Viscosity::Plasma;
    }
    // Slow + sticky → Honey
    if fast_slow < -0.3 {
        return Viscosity::Honey;
    }
    // Default
    Viscosity::Oily
}

// =============================================================================
// CODE-AS-FEELING ENCODER
// =============================================================================

/// Felt dimensions extracted from source code text.
/// Maps the 10 phenomenological regions from `code_as_feeling.py`
/// onto the 48 canonical axes.
pub struct CodeFeeling {
    /// Raw activations on all 48 axes
    pub activations: AxisActivation,
    /// Detected viscosity
    pub viscosity: Viscosity,
}

impl CodeFeeling {
    /// Analyze text for felt dimensions and encode to axis activations.
    ///
    /// Keyword triggers (from `code_as_feeling.py`) map to axes:
    /// - arousal keywords → active↔passive (2), hot↔cold (7)
    /// - warmth keywords → loving↔hateful (26), friendly↔hostile (27)
    /// - presence keywords → real↔imaginary (36), near↔far (13)
    /// - depth keywords → heavy↔light (4), inside↔outside (15)
    /// - trust keywords → safe↔dangerous (32), certain↔uncertain (20)
    /// - integration keywords → whole↔partial (37), ordered↔chaotic (40)
    pub fn from_text(text: &str) -> Self {
        let lower = text.to_lowercase();
        let mut activations = [0.0f32; 48];

        // Arousal triggers → active, hot
        let arousal = Self::count_triggers(&lower, &[
            "arousal", "stirring", "burning", "pulse", "heat", "fire", "ignite",
        ]);
        activations[2] += arousal; // active↔passive
        activations[7] += arousal; // hot↔cold

        // Warmth triggers → loving, friendly
        let warmth = Self::count_triggers(&lower, &[
            "warmth", "warm", "glow", "soft", "tender", "gentle",
        ]);
        activations[26] += warmth; // loving↔hateful
        activations[27] += warmth; // friendly↔hostile

        // Presence triggers → real, near
        let presence = Self::count_triggers(&lower, &[
            "presence", "here", "now", "moment", "awareness", "conscious",
        ]);
        activations[36] += presence; // real↔imaginary
        activations[13] += presence; // near↔far

        // Depth triggers → heavy, inside
        let depth = Self::count_triggers(&lower, &[
            "depth", "deep", "profound", "abyss", "ocean", "core",
        ]);
        activations[4] += depth;  // heavy↔light
        activations[15] += depth; // inside↔outside

        // Trust triggers → safe, certain
        let trust = Self::count_triggers(&lower, &[
            "trust", "safe", "held", "secure", "believe",
        ]);
        activations[32] += trust; // safe↔dangerous
        activations[20] += trust; // certain↔uncertain

        // Integration triggers → whole, ordered
        let integration = Self::count_triggers(&lower, &[
            "integration", "whole", "complete", "unified", "together",
        ]);
        activations[37] += integration; // whole↔partial
        activations[40] += integration; // ordered↔chaotic

        // Surrender triggers → open, free
        let surrender = Self::count_triggers(&lower, &[
            "surrender", "yield", "release", "let go",
        ]);
        activations[38] += surrender; // open↔closed
        activations[39] += surrender; // free↔constrained

        // Analytical/code triggers → concrete, ordered
        let analytical = Self::count_triggers(&lower, &[
            "fn ", "struct ", "impl ", "let ", "match ", "pub ", "mod ",
            "return", "async", "trait",
        ]);
        activations[21] += analytical; // concrete↔abstract
        activations[40] += analytical; // ordered↔chaotic
        activations[29] += analytical; // formal↔informal

        // Creativity triggers → creating, alive, new
        let creative = Self::count_triggers(&lower, &[
            "create", "build", "design", "invent", "imagine", "dream",
        ]);
        activations[44] += creative; // creating↔destroying
        activations[41] += creative; // alive↔dead
        activations[16] += creative; // new↔old

        // Clamp all to [-1, 1]
        for a in activations.iter_mut() {
            *a = a.clamp(-1.0, 1.0);
        }

        let viscosity = detect_viscosity(&activations);
        Self { activations, viscosity }
    }

    /// Encode this feeling into a 16K-bit fingerprint.
    pub fn to_fingerprint(&self) -> [u64; FINGERPRINT_WORDS] {
        encode_axes(&self.activations)
    }

    fn count_triggers(text: &str, triggers: &[&str]) -> f32 {
        let count: usize = triggers.iter().filter(|t| text.contains(**t)).count();
        (count as f32 * 0.2).min(1.0)
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_axis_count() {
        assert_eq!(AXES.len(), 48);
        for (i, axis) in AXES.iter().enumerate() {
            assert_eq!(axis.index as usize, i);
        }
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let mut activations = [0.0f32; 48];
        activations[0] = 0.8;   // good
        activations[7] = -0.5;  // cold
        activations[41] = 1.0;  // alive

        let fp = encode_axes(&activations);
        let decoded = decode_axes(&fp);

        // Round-trip should preserve direction (sign) and rough magnitude
        assert!(decoded[0] > 0.3, "good axis should be positive: {}", decoded[0]);
        assert!(decoded[7] < -0.1, "cold axis should be negative: {}", decoded[7]);
        assert!(decoded[41] > 0.5, "alive axis should be strongly positive: {}", decoded[41]);
    }

    #[test]
    fn test_neutral_is_half_set() {
        let activations = [0.0f32; 48]; // all neutral
        let fp = encode_axes(&activations);

        let total_bits: u32 = fp.iter().map(|w| w.count_ones()).sum();
        // Neutral = ~50% of 48×208 = ~4992 bits set
        assert!(total_bits > 3000, "neutral should set ~half the bits: {}", total_bits);
        assert!(total_bits < 7000, "neutral should set ~half the bits: {}", total_bits);
    }

    #[test]
    fn test_code_feeling_rust_code() {
        let rust_code = r#"
            pub fn create_handler(ctx: &Context) -> impl Trait {
                let result = match input {
                    Some(v) => build_response(v),
                    None => return Err("not found"),
                };
                Ok(result)
            }
        "#;

        let feeling = CodeFeeling::from_text(rust_code);

        // Rust code should activate concrete + ordered + formal
        assert!(feeling.activations[21] > 0.0, "concrete should be positive");
        assert!(feeling.activations[40] > 0.0, "ordered should be positive");
        assert!(feeling.activations[29] > 0.0, "formal should be positive");
    }

    #[test]
    fn test_viscosity_detection() {
        let mut hot_fast = [0.0f32; 48];
        hot_fast[7] = 0.8;  // hot
        hot_fast[9] = 0.6;  // fast
        hot_fast[20] = 0.5; // certain
        assert_eq!(detect_viscosity(&hot_fast), Viscosity::Watery);

        let mut frozen_hard = [0.0f32; 48];
        frozen_hard[7] = -0.7;  // cold
        frozen_hard[5] = 0.7;   // hard
        assert_eq!(detect_viscosity(&frozen_hard), Viscosity::Crystalline);
    }

    #[test]
    fn test_similar_feelings_close_hamming() {
        let feeling_a = CodeFeeling::from_text("warm gentle trust safe love");
        let feeling_b = CodeFeeling::from_text("warmth tender trust secure loving");
        let feeling_c = CodeFeeling::from_text("cold hostile dangerous chaotic");

        let fp_a = feeling_a.to_fingerprint();
        let fp_b = feeling_b.to_fingerprint();
        let fp_c = feeling_c.to_fingerprint();

        // Hamming distance between similar feelings should be less than dissimilar
        let dist_ab = hamming(&fp_a, &fp_b);
        let dist_ac = hamming(&fp_a, &fp_c);

        assert!(dist_ab < dist_ac,
            "similar feelings should be closer: ab={} ac={}", dist_ab, dist_ac);
    }

    fn hamming(a: &[u64; FINGERPRINT_WORDS], b: &[u64; FINGERPRINT_WORDS]) -> u32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x ^ y).count_ones()).sum()
    }
}
