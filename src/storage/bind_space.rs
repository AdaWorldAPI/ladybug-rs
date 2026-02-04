//! Universal Bind Space - Updated with Grammar Layer
//!
//! # 8-bit Prefix : 8-bit Address Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                      PREFIX (8-bit) : ADDRESS (8-bit)                       │
//! ├─────────────────┬───────────────────────────────────────────────────────────┤
//! │  0x00-0x0F:XX   │  SURFACE (16 prefixes × 256 = 4,096)                      │
//! │                 │  0x00: Lance/Kuzu    0x08: Concepts                       │
//! │                 │  0x01: SQL/CQL       0x09: Qualia ops                     │
//! │                 │  0x02: Cypher/GQL    0x0A: Memory ops                     │
//! │                 │  0x03: GraphQL       0x0B: Learning ops                   │
//! │                 │  0x04: NARS/ACT-R    0x0C: NSM Primitives (65+191)        │
//! │                 │  0x05: Causal        0x0D: Grammar Templates (256)        │
//! │                 │  0x06: Meta          0x0E: Speech Acts (256)              │
//! │                 │  0x07: Verbs         0x0F: User Calibration (256)         │
//! ├─────────────────┼───────────────────────────────────────────────────────────┤
//! │  0x10-0x14:XX   │  FLUID/CRYSTAL (5 prefixes × 256 = 1,280)                 │
//! │                 │  Context Crystal: 5×5×5 temporal SPO grid                 │
//! │                 │  Mexican hat weighting: [0.3, 0.7, 1.0, 0.7, 0.3]         │
//! ├─────────────────┼───────────────────────────────────────────────────────────┤
//! │  0x15-0x7F:XX   │  FLUID/EDGES (107 prefixes × 256 = 27,392)                │
//! │                 │  TTL governed working memory                              │
//! ├─────────────────┼───────────────────────────────────────────────────────────┤
//! │  0x80-0xFF:XX   │  NODES (128 prefixes × 256 = 32,768)                      │
//! │                 │  THE UNIVERSAL BIND SPACE                                 │
//! │                 │  All languages hit this. Any syntax. Same addresses.      │
//! └─────────────────┴───────────────────────────────────────────────────────────┘
//! ```

use std::time::Instant;

// =============================================================================
// ADDRESS CONSTANTS (8-bit prefix : 8-bit slot)
// =============================================================================

/// Fingerprint words (10K bits = 156 × 64-bit words)
pub const FINGERPRINT_WORDS: usize = 156;

/// Slots per chunk (2^8 = 256)
pub const CHUNK_SIZE: usize = 256;

// -----------------------------------------------------------------------------
// SURFACE: 16 prefixes (0x00-0x0F) × 256 = 4,096 addresses
// -----------------------------------------------------------------------------

pub const PREFIX_SURFACE_START: u8 = 0x00;
pub const PREFIX_SURFACE_END: u8 = 0x0F;
pub const SURFACE_PREFIXES: usize = 16;
pub const SURFACE_SIZE: usize = 4096;

/// Surface compartments - Query Languages (0x00-0x07)
/// Some prefixes support slot subdivision for related languages (0x00-0x7F: primary, 0x80-0xFF: secondary)
pub const PREFIX_LANCE: u8 = 0x00;     // Lance/Kuzu - vector ops
pub const PREFIX_SQL: u8 = 0x01;       // SQL/CQL (columnar languages)
pub const PREFIX_CQL: u8 = 0x01;       // CQL shares prefix with SQL (slot 0x80+)
pub const PREFIX_CYPHER: u8 = 0x02;    // Cypher/GQL (property graph languages)
pub const PREFIX_GQL: u8 = 0x02;       // GQL shares prefix with Cypher (slot 0x80+)
pub const PREFIX_GRAPHQL: u8 = 0x03;   // GraphQL ops
pub const PREFIX_NARS: u8 = 0x04;      // NARS/ACT-R (cognitive architectures)
pub const PREFIX_ACTR: u8 = 0x04;      // ACT-R shares prefix with NARS (slot 0x80+)
pub const PREFIX_CAUSAL: u8 = 0x05;    // Causal reasoning (Pearl)
pub const PREFIX_META: u8 = 0x06;      // Meta-cognition
pub const PREFIX_VERBS: u8 = 0x07;     // Verbs (CAUSES, BECOMES...)

/// Surface compartments - Cognitive Ops (0x08-0x0B)
pub const PREFIX_CONCEPTS: u8 = 0x08;  // Core concepts/types
pub const PREFIX_QUALIA: u8 = 0x09;    // Qualia operations (18D)
pub const PREFIX_MEMORY: u8 = 0x0A;    // Memory operations
pub const PREFIX_LEARNING: u8 = 0x0B;  // Learning operations

/// Surface compartments - Grammar Layer (0x0C-0x0F) [NEW]
pub const PREFIX_NSM: u8 = 0x0C;           // NSM Primitives (65 Wierzbicka + 191 extensions)
pub const PREFIX_TEMPLATES: u8 = 0x0D;     // Grammar Templates (construction grammar)
pub const PREFIX_SPEECH_ACTS: u8 = 0x0E;   // Speech Acts (pragmatics)
pub const PREFIX_CALIBRATION: u8 = 0x0F;   // User Calibration (per-user quirks)

// Slot subdivision boundary for shared prefixes
// Slots 0x00-0x7F: primary language (SQL, Cypher, NARS)
// Slots 0x80-0xFF: secondary language (CQL, GQL, ACT-R)
pub const SLOT_SUBDIVISION: u8 = 0x80;

// -----------------------------------------------------------------------------
// NSM Slots (0x0C:00-0xFF)
// Wierzbicka's 65 semantic primitives + 191 learned extensions
// -----------------------------------------------------------------------------

/// NSM primitive slot assignments (0x00-0x40 = 65 primes)
pub mod nsm_slots {
    // Substantives (0x00-0x05)
    pub const I: u8 = 0x00;
    pub const YOU: u8 = 0x01;
    pub const SOMEONE: u8 = 0x02;
    pub const SOMETHING: u8 = 0x03;
    pub const PEOPLE: u8 = 0x04;
    pub const BODY: u8 = 0x05;
    
    // Determiners (0x06-0x08)
    pub const THIS: u8 = 0x06;
    pub const THE_SAME: u8 = 0x07;
    pub const OTHER: u8 = 0x08;
    
    // Quantifiers (0x09-0x0E)
    pub const ONE: u8 = 0x09;
    pub const TWO: u8 = 0x0A;
    pub const SOME: u8 = 0x0B;
    pub const ALL: u8 = 0x0C;
    pub const MUCH: u8 = 0x0D;
    pub const MANY: u8 = 0x0E;
    
    // Evaluators (0x0F-0x10)
    pub const GOOD: u8 = 0x0F;
    pub const BAD: u8 = 0x10;
    
    // Descriptors (0x11-0x12)
    pub const BIG: u8 = 0x11;
    pub const SMALL: u8 = 0x12;
    
    // Mental predicates (0x13-0x18)
    pub const THINK: u8 = 0x13;
    pub const KNOW: u8 = 0x14;
    pub const WANT: u8 = 0x15;
    pub const FEEL: u8 = 0x16;
    pub const SEE: u8 = 0x17;
    pub const HEAR: u8 = 0x18;
    
    // Speech (0x19-0x1B)
    pub const SAY: u8 = 0x19;
    pub const WORDS: u8 = 0x1A;
    pub const TRUE: u8 = 0x1B;
    
    // Actions/Events (0x1C-0x1F)
    pub const DO: u8 = 0x1C;
    pub const HAPPEN: u8 = 0x1D;
    pub const MOVE: u8 = 0x1E;
    pub const TOUCH: u8 = 0x1F;
    
    // Existence/Possession (0x20-0x21)
    pub const THERE_IS: u8 = 0x20;
    pub const HAVE: u8 = 0x21;
    
    // Life/Death (0x22-0x23)
    pub const LIVE: u8 = 0x22;
    pub const DIE: u8 = 0x23;
    
    // Time (0x24-0x2B)
    pub const WHEN: u8 = 0x24;
    pub const NOW: u8 = 0x25;
    pub const BEFORE: u8 = 0x26;
    pub const AFTER: u8 = 0x27;
    pub const A_LONG_TIME: u8 = 0x28;
    pub const A_SHORT_TIME: u8 = 0x29;
    pub const FOR_SOME_TIME: u8 = 0x2A;
    pub const MOMENT: u8 = 0x2B;
    
    // Space (0x2C-0x33)
    pub const WHERE: u8 = 0x2C;
    pub const HERE: u8 = 0x2D;
    pub const ABOVE: u8 = 0x2E;
    pub const BELOW: u8 = 0x2F;
    pub const FAR: u8 = 0x30;
    pub const NEAR: u8 = 0x31;
    pub const SIDE: u8 = 0x32;
    pub const INSIDE: u8 = 0x33;
    
    // Logical (0x34-0x38)
    pub const NOT: u8 = 0x34;
    pub const MAYBE: u8 = 0x35;
    pub const CAN: u8 = 0x36;
    pub const BECAUSE: u8 = 0x37;
    pub const IF: u8 = 0x38;
    
    // Intensifier (0x39)
    pub const VERY: u8 = 0x39;
    
    // Similarity (0x3A)
    pub const LIKE: u8 = 0x3A;
    
    // Augmentatives (0x3B-0x40)
    pub const MORE: u8 = 0x3B;
    pub const PART: u8 = 0x3C;
    pub const KIND: u8 = 0x3D;
    pub const WORD: u8 = 0x3E;
    pub const THING: u8 = 0x3F;
    pub const WAY: u8 = 0x40;
    
    // === LEARNED EXTENSIONS (0x41-0xFF) ===
    // 191 slots for semantic molecules learned from corpus
    pub const EXTENSION_START: u8 = 0x41;
    pub const EXTENSION_END: u8 = 0xFF;
    
    /// Number of core NSM primitives
    pub const NSM_PRIMITIVE_COUNT: usize = 65;
    
    /// Number of extension slots
    pub const NSM_EXTENSION_COUNT: usize = 191;
}

// -----------------------------------------------------------------------------
// Template Slots (0x0D:00-0xFF)
// Construction Grammar templates from agi-chat
// -----------------------------------------------------------------------------

pub mod template_slots {
    // Core clauses (0x00-0x1F)
    pub const TRANSITIVE_DECLARATIVE: u8 = 0x00;
    pub const INTRANSITIVE_DECLARATIVE: u8 = 0x01;
    pub const COPULAR_STATE: u8 = 0x02;
    pub const EXISTENTIAL: u8 = 0x03;
    pub const POSSESSION: u8 = 0x04;
    pub const DITRANSITIVE: u8 = 0x05;
    pub const PASSIVE: u8 = 0x06;
    pub const CAUSATIVE: u8 = 0x07;
    pub const RESULTATIVE: u8 = 0x08;
    pub const BENEFACTIVE: u8 = 0x09;
    pub const EXPERIENCER: u8 = 0x0A;
    pub const PERCEPTION: u8 = 0x0B;
    pub const COMMUNICATION: u8 = 0x0C;
    pub const MOTION_INTRANSITIVE: u8 = 0x0D;
    pub const MOTION_TRANSITIVE: u8 = 0x0E;
    pub const CHANGE_OF_STATE: u8 = 0x0F;
    
    // Questions (0x20-0x2F)
    pub const WH_QUESTION: u8 = 0x20;
    pub const YESNO_QUESTION: u8 = 0x21;
    pub const TAG_QUESTION: u8 = 0x22;
    pub const RHETORICAL_QUESTION: u8 = 0x23;
    pub const EMBEDDED_QUESTION: u8 = 0x24;
    
    // Directives (0x30-0x3F)
    pub const IMPERATIVE_COMMAND: u8 = 0x30;
    pub const POLITE_REQUEST: u8 = 0x31;
    pub const SUGGESTION: u8 = 0x32;
    pub const PERMISSION: u8 = 0x33;
    pub const PROHIBITION: u8 = 0x34;
    pub const INVITATION: u8 = 0x35;
    
    // Mental state (0x40-0x4F)
    pub const MENTAL_STATE: u8 = 0x40;
    pub const BELIEF_REPORT: u8 = 0x41;
    pub const PREFERENCE_EXPRESSION: u8 = 0x42;
    pub const INTENTION_EXPRESSION: u8 = 0x43;
    pub const DESIRE_EXPRESSION: u8 = 0x44;
    pub const KNOWLEDGE_CLAIM: u8 = 0x45;
    
    // Modals (0x50-0x5F)
    pub const MODAL_ABILITY: u8 = 0x50;
    pub const MODAL_OBLIGATION: u8 = 0x51;
    pub const MODAL_PERMISSION: u8 = 0x52;
    pub const MODAL_POSSIBILITY: u8 = 0x53;
    pub const MODAL_NECESSITY: u8 = 0x54;
    pub const MODAL_VOLITION: u8 = 0x55;
    
    // Complex (0x60-0x7F)
    pub const CONDITIONAL: u8 = 0x60;
    pub const RELATIVE_CLAUSE: u8 = 0x61;
    pub const COMPLEMENT_CLAUSE: u8 = 0x62;
    pub const ADVERBIAL_CLAUSE: u8 = 0x63;
    pub const CLEFT: u8 = 0x64;
    pub const PSEUDO_CLEFT: u8 = 0x65;
    pub const EXTRAPOSITION: u8 = 0x66;
    pub const COORDINATION: u8 = 0x67;
    pub const SUBORDINATION: u8 = 0x68;
    
    // Language-specific (0x80-0xBF)
    // German
    pub const DE_TECAMOLO: u8 = 0x80;           // Temporal-Causal-Modal-Local
    pub const DE_VERB_SECOND: u8 = 0x81;        // V2 word order
    pub const DE_BRACKET: u8 = 0x82;            // Satzklammer
    pub const DE_SUBJUNCTIVE_II: u8 = 0x83;     // Konjunktiv II
    pub const DE_RELATIVE_FINAL: u8 = 0x84;     // Relativsatz verb-final
    
    // French
    pub const FR_NEGATION: u8 = 0x90;
    pub const FR_CLITIC_CLUSTER: u8 = 0x91;
    pub const FR_SUBJUNCTIVE: u8 = 0x92;
    
    // Spanish
    pub const ES_SER_ESTAR: u8 = 0xA0;
    pub const ES_SUBJUNCTIVE: u8 = 0xA1;
    pub const ES_CLITIC_DOUBLING: u8 = 0xA2;
    
    // User-calibrated (0xC0-0xFF)
    // Reserved for templates learned from user corrections
    pub const USER_TEMPLATE_START: u8 = 0xC0;
    pub const USER_TEMPLATE_END: u8 = 0xFF;
}

// -----------------------------------------------------------------------------
// Speech Act Slots (0x0E:00-0xFF)
// Pragmatic patterns mapping to NSM mental predicates
// -----------------------------------------------------------------------------

pub mod speech_act_slots {
    // Assertives (0x00-0x1F) → THINK commitment
    pub const ASSERT: u8 = 0x00;
    pub const CLAIM: u8 = 0x01;
    pub const REPORT: u8 = 0x02;
    pub const PREDICT: u8 = 0x03;
    pub const DESCRIBE: u8 = 0x04;
    pub const INFORM: u8 = 0x05;
    pub const EXPLAIN: u8 = 0x06;
    pub const CONFIRM: u8 = 0x07;
    pub const DENY: u8 = 0x08;
    pub const CONCEDE: u8 = 0x09;
    pub const HYPOTHESIZE: u8 = 0x0A;
    pub const SPECULATE: u8 = 0x0B;
    pub const INSIST: u8 = 0x0C;
    pub const GUARANTEE: u8 = 0x0D;
    
    // Directives (0x20-0x3F) → WANT commitment
    pub const REQUEST: u8 = 0x20;
    pub const COMMAND: u8 = 0x21;
    pub const SUGGEST: u8 = 0x22;
    pub const INVITE: u8 = 0x23;
    pub const ADVISE: u8 = 0x24;
    pub const WARN: u8 = 0x25;
    pub const URGE: u8 = 0x26;
    pub const BEG: u8 = 0x27;
    pub const FORBID: u8 = 0x28;
    pub const PERMIT: u8 = 0x29;
    pub const RECOMMEND: u8 = 0x2A;
    pub const DEMAND: u8 = 0x2B;
    pub const INSTRUCT: u8 = 0x2C;
    
    // Commissives (0x40-0x5F) → intention commitment
    pub const PROMISE: u8 = 0x40;
    pub const COMMIT: u8 = 0x41;
    pub const OFFER: u8 = 0x42;
    pub const PLEDGE: u8 = 0x43;
    pub const THREATEN: u8 = 0x44;
    pub const REFUSE: u8 = 0x45;
    pub const ACCEPT: u8 = 0x46;
    pub const AGREE: u8 = 0x47;
    pub const VOLUNTEER: u8 = 0x48;
    pub const GUARANTEE_ACT: u8 = 0x49;
    
    // Expressives (0x60-0x7F) → FEEL (retractable)
    pub const THANK: u8 = 0x60;
    pub const APOLOGIZE: u8 = 0x61;
    pub const COMPLAIN: u8 = 0x62;
    pub const PRAISE: u8 = 0x63;
    pub const CRITICIZE: u8 = 0x64;
    pub const CONGRATULATE: u8 = 0x65;
    pub const SYMPATHIZE: u8 = 0x66;
    pub const GREET: u8 = 0x67;
    pub const FAREWELL: u8 = 0x68;
    pub const WELCOME: u8 = 0x69;
    pub const CELEBRATE: u8 = 0x6A;
    pub const LAMENT: u8 = 0x6B;
    
    // Declarations (0x80-0x9F) → world-change
    pub const DECLARE: u8 = 0x80;
    pub const DEFINE: u8 = 0x81;
    pub const NAME: u8 = 0x82;
    pub const APPOINT: u8 = 0x83;
    pub const RESIGN: u8 = 0x84;
    pub const BAPTIZE: u8 = 0x85;
    pub const PRONOUNCE: u8 = 0x86;
    pub const SENTENCE: u8 = 0x87;
    pub const BLESS: u8 = 0x88;
    pub const FIRE: u8 = 0x89;
    pub const HIRE: u8 = 0x8A;
    
    // Meta-speech (0xA0-0xBF) → meta-communication
    pub const CLARIFY: u8 = 0xA0;
    pub const ELABORATE: u8 = 0xA1;
    pub const REPHRASE: u8 = 0xA2;
    pub const SUMMARIZE: u8 = 0xA3;
    pub const CORRECT: u8 = 0xA4;
    pub const RETRACT: u8 = 0xA5;
    pub const QUALIFY: u8 = 0xA6;
    pub const DIGRESS: u8 = 0xA7;
    pub const RETURN_TOPIC: u8 = 0xA8;
    pub const INTERRUPT: u8 = 0xA9;
    pub const YIELD_FLOOR: u8 = 0xAA;
    
    // User-defined (0xC0-0xFF)
    pub const USER_ACT_START: u8 = 0xC0;
    pub const USER_ACT_END: u8 = 0xFF;
}

// -----------------------------------------------------------------------------
// FLUID ZONE: 112 prefixes (0x10-0x7F) × 256 = 28,672 addresses
// Split into Crystal (0x10-0x14) and Working Memory (0x15-0x7F)
// -----------------------------------------------------------------------------

pub const PREFIX_FLUID_START: u8 = 0x10;
pub const PREFIX_FLUID_END: u8 = 0x7F;
pub const FLUID_PREFIXES: usize = 112;
pub const FLUID_SIZE: usize = 28672;

/// Context Crystal prefixes (5 temporal slices)
pub const PREFIX_CRYSTAL_S_MINUS_2: u8 = 0x10;  // S-2: 2 sentences before
pub const PREFIX_CRYSTAL_S_MINUS_1: u8 = 0x11;  // S-1: 1 sentence before
pub const PREFIX_CRYSTAL_S_CURRENT: u8 = 0x12;  // S0:  Current sentence
pub const PREFIX_CRYSTAL_S_PLUS_1: u8 = 0x13;   // S+1: 1 sentence after
pub const PREFIX_CRYSTAL_S_PLUS_2: u8 = 0x14;   // S+2: 2 sentences after

/// Mexican hat temporal weights for crystal
pub const CRYSTAL_WEIGHTS: [f32; 5] = [0.3, 0.7, 1.0, 0.7, 0.3];

/// Working memory starts after crystal
pub const PREFIX_WORKING_START: u8 = 0x15;

// -----------------------------------------------------------------------------
// NODES: 128 prefixes (0x80-0xFF) × 256 = 32,768 addresses
// -----------------------------------------------------------------------------

pub const PREFIX_NODE_START: u8 = 0x80;
pub const PREFIX_NODE_END: u8 = 0xFF;
pub const NODE_PREFIXES: usize = 128;
pub const NODE_SIZE: usize = 32768;

pub const TOTAL_ADDRESSES: usize = 65536;

// =============================================================================
// ADDRESS TYPE
// =============================================================================

/// 16-bit address as prefix:slot
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Addr(pub u16);

impl Addr {
    #[inline(always)]
    pub fn new(prefix: u8, slot: u8) -> Self {
        Self(((prefix as u16) << 8) | (slot as u16))
    }
    
    #[inline(always)]
    pub fn prefix(self) -> u8 {
        (self.0 >> 8) as u8
    }
    
    #[inline(always)]
    pub fn slot(self) -> u8 {
        (self.0 & 0xFF) as u8
    }
    
    #[inline(always)]
    pub fn is_surface(self) -> bool {
        self.prefix() <= PREFIX_SURFACE_END
    }
    
    #[inline(always)]
    pub fn is_fluid(self) -> bool {
        let p = self.prefix();
        p >= PREFIX_FLUID_START && p <= PREFIX_FLUID_END
    }
    
    #[inline(always)]
    pub fn is_node(self) -> bool {
        self.prefix() >= PREFIX_NODE_START
    }
    
    // === Grammar-specific checks ===
    
    /// Is this an NSM primitive address?
    #[inline(always)]
    pub fn is_nsm(self) -> bool {
        self.prefix() == PREFIX_NSM
    }
    
    /// Is this an NSM core primitive (0-64)?
    #[inline(always)]
    pub fn is_nsm_core(self) -> bool {
        self.prefix() == PREFIX_NSM && self.slot() <= nsm_slots::WAY
    }
    
    /// Is this an NSM extension (learned)?
    #[inline(always)]
    pub fn is_nsm_extension(self) -> bool {
        self.prefix() == PREFIX_NSM && self.slot() >= nsm_slots::EXTENSION_START
    }
    
    /// Is this a grammar template address?
    #[inline(always)]
    pub fn is_template(self) -> bool {
        self.prefix() == PREFIX_TEMPLATES
    }
    
    /// Is this a speech act address?
    #[inline(always)]
    pub fn is_speech_act(self) -> bool {
        self.prefix() == PREFIX_SPEECH_ACTS
    }
    
    /// Is this a user calibration address?
    #[inline(always)]
    pub fn is_calibration(self) -> bool {
        self.prefix() == PREFIX_CALIBRATION
    }
    
    /// Is this in the context crystal?
    #[inline(always)]
    pub fn is_crystal(self) -> bool {
        let p = self.prefix();
        p >= PREFIX_CRYSTAL_S_MINUS_2 && p <= PREFIX_CRYSTAL_S_PLUS_2
    }
    
    /// Get crystal temporal position (-2 to +2)
    #[inline(always)]
    pub fn crystal_position(self) -> Option<i32> {
        if self.is_crystal() {
            Some((self.prefix() as i32) - (PREFIX_CRYSTAL_S_CURRENT as i32))
        } else {
            None
        }
    }
    
    /// Get Mexican hat weight for crystal cell
    #[inline]
    pub fn crystal_weight(self) -> f32 {
        if let Some(pos) = self.crystal_position() {
            CRYSTAL_WEIGHTS[(pos + 2) as usize]
        } else {
            0.0
        }
    }
    
    // === Convenience constructors ===
    
    /// Create NSM primitive address
    #[inline]
    pub fn nsm(slot: u8) -> Self {
        Self::new(PREFIX_NSM, slot)
    }
    
    /// Create template address
    #[inline]
    pub fn template(slot: u8) -> Self {
        Self::new(PREFIX_TEMPLATES, slot)
    }
    
    /// Create speech act address
    #[inline]
    pub fn speech_act(slot: u8) -> Self {
        Self::new(PREFIX_SPEECH_ACTS, slot)
    }
    
    /// Create calibration address
    #[inline]
    pub fn calibration(slot: u8) -> Self {
        Self::new(PREFIX_CALIBRATION, slot)
    }
    
    /// Create crystal address from temporal position and cell index
    #[inline]
    pub fn crystal(position: i32, cell: u8) -> Self {
        let prefix = (PREFIX_CRYSTAL_S_CURRENT as i32 + position).clamp(
            PREFIX_CRYSTAL_S_MINUS_2 as i32,
            PREFIX_CRYSTAL_S_PLUS_2 as i32,
        ) as u8;
        Self::new(prefix, cell)
    }
}

impl std::fmt::Display for Addr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0x{:02X}:{:02X}", self.prefix(), self.slot())
    }
}

// =============================================================================
// ZONE HELPER FUNCTIONS
// =============================================================================

/// Categorize a prefix into its zone
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Zone {
    Surface(SurfaceCompartment),
    Fluid(FluidType),
    Node,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceCompartment {
    Lance,
    Sql,
    Cypher,
    GraphQL,
    Nars,
    Causal,
    Meta,
    Verbs,
    Concepts,
    Qualia,
    Memory,
    Learning,
    Nsm,
    Templates,
    SpeechActs,
    Calibration,
}

/// Sublanguage within a shared prefix (slot-based discrimination)
/// Slots 0x00-0x7F map to primary language, 0x80-0xFF to secondary
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sublanguage {
    // Columnar (prefix 0x01)
    Sql,
    Cql,  // Cassandra Query Language
    // Graph (prefix 0x02)
    Cypher,
    Gql,  // ISO Graph Query Language
    // Cognitive (prefix 0x04)
    Nars,
    ActR, // ACT-R cognitive architecture
}

impl Sublanguage {
    /// Determine sublanguage from address (for shared prefixes)
    /// Returns None for non-subdivided compartments
    pub fn from_addr(addr: Addr) -> Option<Self> {
        let prefix = addr.prefix();
        let slot = addr.slot();
        let is_secondary = slot >= SLOT_SUBDIVISION;

        match prefix {
            PREFIX_SQL => Some(if is_secondary { Sublanguage::Cql } else { Sublanguage::Sql }),
            PREFIX_CYPHER => Some(if is_secondary { Sublanguage::Gql } else { Sublanguage::Cypher }),
            PREFIX_NARS => Some(if is_secondary { Sublanguage::ActR } else { Sublanguage::Nars }),
            _ => None,
        }
    }

    /// Check if a compartment supports sublanguages
    pub fn compartment_has_sublanguages(comp: SurfaceCompartment) -> bool {
        matches!(comp, SurfaceCompartment::Sql | SurfaceCompartment::Cypher | SurfaceCompartment::Nars)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FluidType {
    Crystal { position: i32 },
    Working,
}

impl Addr {
    pub fn zone(self) -> Zone {
        let p = self.prefix();
        
        if p <= PREFIX_SURFACE_END {
            let comp = match p {
                PREFIX_LANCE => SurfaceCompartment::Lance,
                PREFIX_SQL => SurfaceCompartment::Sql,
                PREFIX_CYPHER => SurfaceCompartment::Cypher,
                PREFIX_GRAPHQL => SurfaceCompartment::GraphQL,
                PREFIX_NARS => SurfaceCompartment::Nars,
                PREFIX_CAUSAL => SurfaceCompartment::Causal,
                PREFIX_META => SurfaceCompartment::Meta,
                PREFIX_VERBS => SurfaceCompartment::Verbs,
                PREFIX_CONCEPTS => SurfaceCompartment::Concepts,
                PREFIX_QUALIA => SurfaceCompartment::Qualia,
                PREFIX_MEMORY => SurfaceCompartment::Memory,
                PREFIX_LEARNING => SurfaceCompartment::Learning,
                PREFIX_NSM => SurfaceCompartment::Nsm,
                PREFIX_TEMPLATES => SurfaceCompartment::Templates,
                PREFIX_SPEECH_ACTS => SurfaceCompartment::SpeechActs,
                PREFIX_CALIBRATION => SurfaceCompartment::Calibration,
                _ => unreachable!(),
            };
            Zone::Surface(comp)
        } else if p >= PREFIX_NODE_START {
            Zone::Node
        } else {
            // Fluid zone
            if p <= PREFIX_CRYSTAL_S_PLUS_2 {
                let pos = (p as i32) - (PREFIX_CRYSTAL_S_CURRENT as i32);
                Zone::Fluid(FluidType::Crystal { position: pos })
            } else {
                Zone::Fluid(FluidType::Working)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_nsm_addresses() {
        let addr = Addr::nsm(nsm_slots::WANT);
        assert!(addr.is_nsm());
        assert!(addr.is_nsm_core());
        assert!(!addr.is_nsm_extension());
        assert_eq!(addr.prefix(), PREFIX_NSM);
        assert_eq!(addr.slot(), nsm_slots::WANT);
    }
    
    #[test]
    fn test_crystal_addresses() {
        let addr = Addr::crystal(-2, 10);
        assert!(addr.is_crystal());
        assert_eq!(addr.crystal_position(), Some(-2));
        assert_eq!(addr.crystal_weight(), 0.3);
        
        let addr_current = Addr::crystal(0, 10);
        assert_eq!(addr_current.crystal_weight(), 1.0);
    }
    
    #[test]
    fn test_zone_classification() {
        assert!(matches!(Addr::nsm(0).zone(), Zone::Surface(SurfaceCompartment::Nsm)));
        assert!(matches!(Addr::template(0).zone(), Zone::Surface(SurfaceCompartment::Templates)));
        assert!(matches!(Addr::crystal(0, 0).zone(), Zone::Fluid(FluidType::Crystal { position: 0 })));
        assert!(matches!(Addr::new(0x80, 0).zone(), Zone::Node));
    }
}
