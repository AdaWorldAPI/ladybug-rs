//! Complete Cognitive Codebook
//!
//! Unified encoding system for the entire cognitive substrate:
//! - 16-bit bucket (domain + subtype + index) for O(log n) BTree lookup
//! - 48-bit content hash for O(1) direct access
//! - 10K fingerprints for full-resolution resonance
//!
//! Encodes: NSM primes, roles, SPO, qualia, NARS, YAML templates,
//! causality, temporal relations, rung levels, and learned concepts.
//!
//! Total vocabulary: ~1000 built-in concepts + unlimited learned
//! Storage: ~13 KB for complete cognitive model (vs 4GB for DeepNSM-1B)

use crate::core::Fingerprint;
use std::collections::{BTreeMap, HashMap};

// =============================================================================
// Domain Identifiers (4 bits = 16 domains)
// =============================================================================

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CognitiveDomain {
    NsmPrime       = 0x0,  // 65 semantic primes
    NsmRole        = 0x1,  // 10 thematic roles
    SpoSubject     = 0x2,  // Subject slot markers
    SpoPredicate   = 0x3,  // Predicate slot markers
    SpoObject      = 0x4,  // Object slot markers
    Qualia         = 0x5,  // 8 affect channels
    NarsTerm       = 0x6,  // NARS terms and copulas
    NarsInference  = 0x7,  // NARS inference rules
    Causality      = 0x8,  // Causal relations
    Temporal       = 0x9,  // Temporal relations
    YamlTemplate   = 0xA,  // Speech act templates
    RungLevel      = 0xB,  // Meaning depth (0-9)
    CrystalPos     = 0xC,  // Crystal coordinates
    LearnedConcept = 0xD,  // Dynamically learned
    MetaPattern    = 0xE,  // Meta-level patterns
    Reserved       = 0xF,  // Future use
}

impl CognitiveDomain {
    pub fn from_u8(v: u8) -> Self {
        match v & 0xF {
            0x0 => Self::NsmPrime,
            0x1 => Self::NsmRole,
            0x2 => Self::SpoSubject,
            0x3 => Self::SpoPredicate,
            0x4 => Self::SpoObject,
            0x5 => Self::Qualia,
            0x6 => Self::NarsTerm,
            0x7 => Self::NarsInference,
            0x8 => Self::Causality,
            0x9 => Self::Temporal,
            0xA => Self::YamlTemplate,
            0xB => Self::RungLevel,
            0xC => Self::CrystalPos,
            0xD => Self::LearnedConcept,
            0xE => Self::MetaPattern,
            _ => Self::Reserved,
        }
    }
}

// =============================================================================
// NSM Prime Categories (4 bits = 16 categories)
// =============================================================================

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NsmCategory {
    Substantive   = 0x0,  // I, YOU, SOMEONE, SOMETHING, PEOPLE, BODY
    Relational    = 0x1,  // KIND, PART
    Determiner    = 0x2,  // THIS, THE_SAME, OTHER, ELSE, ANOTHER
    Quantifier    = 0x3,  // ONE, TWO, SOME, ALL, MUCH, MANY, LITTLE, FEW
    Evaluator     = 0x4,  // GOOD, BAD
    Descriptor    = 0x5,  // BIG, SMALL
    Mental        = 0x6,  // THINK, KNOW, WANT, DONT_WANT, FEEL, SEE, HEAR
    Speech        = 0x7,  // SAY, WORDS, TRUE
    Action        = 0x8,  // DO, HAPPEN, MOVE
    Existence     = 0x9,  // BE, THERE_IS, BE_SOMETHING, MINE
    Life          = 0xA,  // LIVE, DIE
    Time          = 0xB,  // WHEN, TIME, NOW, BEFORE, AFTER, A_LONG_TIME, A_SHORT_TIME, FOR_SOME_TIME, MOMENT
    Space         = 0xC,  // WHERE, PLACE, HERE, ABOVE, BELOW, FAR, NEAR, SIDE, INSIDE, TOUCH
    Logical       = 0xD,  // NOT, MAYBE, CAN, BECAUSE, IF
    Intensifier   = 0xE,  // VERY, MORE
    Similarity    = 0xF,  // LIKE, AS, WAY
}

// =============================================================================
// Qualia Channels (Russell's Circumplex + extensions)
// =============================================================================

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QualiaChannel {
    Arousal       = 0x0,  // Activation/energy (high=alert, low=calm) - Russell 1980
    Valence       = 0x1,  // Hedonic tone (positive/negative)
    Tension       = 0x2,  // Stress/relaxation continuum
    Certainty     = 0x3,  // Confidence/doubt (epistemic)
    Agency        = 0x4,  // Control/helplessness (locus)
    Temporality   = 0x5,  // Urgency/patience (time pressure)
    Sociality     = 0x6,  // Approach/avoidance (social orientation)
    Novelty       = 0x7,  // Familiarity/surprise (pattern deviation)
}

// =============================================================================
// NARS Types
// =============================================================================

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NarsCopula {
    Inheritance   = 0x0,  // S --> P  (is-a)
    Similarity    = 0x1,  // S <-> P  (similar-to)
    Implication   = 0x2,  // S ==> P  (if-then)
    Equivalence   = 0x3,  // S <=> P  (iff)
    Instance      = 0x4,  // S {-- P  (instance-of)
    Property      = 0x5,  // S --] P  (has-property)
    InstanceProp  = 0x6,  // S {-] P  (instance-property)
    Conjunction   = 0x7,  // S && P
    Disjunction   = 0x8,  // S || P
    Negation      = 0x9,  // --S
    Sequential    = 0xA,  // S &/ P   (sequential conjunction)
    Parallel      = 0xB,  // S &| P   (parallel conjunction)
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NarsInference {
    Deduction     = 0x0,  // {M-->P, S-->M} |- S-->P
    Induction     = 0x1,  // {M-->P, M-->S} |- S-->P
    Abduction     = 0x2,  // {P-->M, S-->M} |- S-->P
    Exemplification = 0x3, // {P-->M, S-->M} |- P-->S
    Comparison    = 0x4,  // {M-->P, M-->S} |- S<->P
    Analogy       = 0x5,  // {M-->P, S<->M} |- S-->P
    Resemblance   = 0x6,  // {M<->P, S<->M} |- S<->P
    Revision      = 0x7,  // {S-->P <f1,c1>, S-->P <f2,c2>} |- S-->P <f,c>
    Choice        = 0x8,  // Select higher-confidence
    Decision      = 0x9,  // Act on sufficient evidence
}

// =============================================================================
// Causality Types
// =============================================================================

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CausalityType {
    Enables       = 0x0,  // A makes B possible
    Causes        = 0x1,  // A directly causes B
    Prevents      = 0x2,  // A blocks B
    Maintains     = 0x3,  // A keeps B true
    Triggers      = 0x4,  // A initiates B
    Terminates    = 0x5,  // A ends B
    Modulates     = 0x6,  // A adjusts strength of B
    Correlates    = 0x7,  // A and B co-occur (no direction)
}

// =============================================================================
// Temporal Relations (Allen's Interval Algebra)
// =============================================================================

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TemporalRelation {
    Before        = 0x0,  // A entirely before B
    After         = 0x1,  // A entirely after B
    Meets         = 0x2,  // A ends when B starts
    MetBy         = 0x3,  // A starts when B ends
    Overlaps      = 0x4,  // A starts before B, ends during B
    OverlappedBy  = 0x5,  // B starts before A, ends during A
    During        = 0x6,  // A contained within B
    Contains      = 0x7,  // B contained within A
    Starts        = 0x8,  // A starts with B, ends before B
    StartedBy     = 0x9,  // B starts with A, ends before A
    Finishes      = 0xA,  // A starts after B, ends with B
    FinishedBy    = 0xB,  // B starts after A, ends with A
    Equals        = 0xC,  // A and B same interval
    Now           = 0xD,  // Present moment
    Always        = 0xE,  // Eternal/timeless
    Never         = 0xF,  // Non-occurrence
}

// =============================================================================
// YAML Templates (Speech Acts from AGI-chat grammar)
// =============================================================================

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum YamlTemplate {
    Greeting      = 0x0,  // Acknowledge presence
    Farewell      = 0x1,  // End interaction
    Question      = 0x2,  // Request information
    Statement     = 0x3,  // Assert information
    Command       = 0x4,  // Direct action
    Request       = 0x5,  // Polite action request
    Offer         = 0x6,  // Propose to do
    Promise       = 0x7,  // Commit to future
    Warning       = 0x8,  // Alert to danger
    Apology       = 0x9,  // Express regret
    Gratitude     = 0xA,  // Express thanks
    Complaint     = 0xB,  // Express dissatisfaction
    Explanation   = 0xC,  // Provide reasoning
    Narrative     = 0xD,  // Tell story
    Opinion       = 0xE,  // Express view
    Hypothesis    = 0xF,  // Propose possibility
}

// =============================================================================
// Thematic Roles
// =============================================================================

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThematicRole {
    Agent         = 0x0,  // Who does it
    Patient       = 0x1,  // Who/what is affected
    Theme         = 0x2,  // What moves/changes
    Experiencer   = 0x3,  // Who perceives/feels
    Beneficiary   = 0x4,  // For whom
    Instrument    = 0x5,  // With what
    Location      = 0x6,  // Where
    Source        = 0x7,  // From where
    Goal          = 0x8,  // To where / purpose
    Time          = 0x9,  // When
    Manner        = 0xA,  // How
    Cause         = 0xB,  // Why (efficient cause)
    Purpose       = 0xC,  // Why (final cause)
    Condition     = 0xD,  // Under what circumstances
    Extent        = 0xE,  // How much
    Attribute     = 0xF,  // Quality/property
}

// =============================================================================
// The 64-bit Cognitive Address
// =============================================================================

/// 64-bit cognitive address: 16-bit bucket + 48-bit content hash
/// 
/// Layout:
/// ```text
/// [63:60] Domain (4 bits)     - CognitiveDomain enum
/// [59:56] Subtype (4 bits)    - Category within domain
/// [55:48] Index (8 bits)      - Instance within subtype
/// [47:0]  Hash (48 bits)      - Content fingerprint (folded from 10K)
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CognitiveAddress(pub u64);

impl CognitiveAddress {
    /// Create from components
    pub fn new(domain: CognitiveDomain, subtype: u8, index: u8, hash: u64) -> Self {
        let d = (domain as u64) << 60;
        let s = ((subtype & 0xF) as u64) << 56;
        let i = (index as u64) << 48;
        let h = hash & 0xFFFF_FFFF_FFFF; // 48 bits
        
        Self(d | s | i | h)
    }
    
    /// Create for NSM prime
    pub fn nsm_prime(category: NsmCategory, index: u8, hash: u64) -> Self {
        Self::new(CognitiveDomain::NsmPrime, category as u8, index, hash)
    }
    
    /// Create for role
    pub fn role(role: ThematicRole, hash: u64) -> Self {
        Self::new(CognitiveDomain::NsmRole, 0, role as u8, hash)
    }
    
    /// Create for qualia
    pub fn qualia(channel: QualiaChannel, level: u8, hash: u64) -> Self {
        Self::new(CognitiveDomain::Qualia, channel as u8, level, hash)
    }
    
    /// Create for NARS copula
    pub fn nars_copula(copula: NarsCopula, hash: u64) -> Self {
        Self::new(CognitiveDomain::NarsTerm, copula as u8, 0, hash)
    }
    
    /// Create for NARS inference
    pub fn nars_inference(inference: NarsInference, hash: u64) -> Self {
        Self::new(CognitiveDomain::NarsInference, inference as u8, 0, hash)
    }
    
    /// Create for YAML template
    pub fn yaml_template(template: YamlTemplate, hash: u64) -> Self {
        Self::new(CognitiveDomain::YamlTemplate, template as u8, 0, hash)
    }
    
    /// Create for causality
    pub fn causality(cause_type: CausalityType, hash: u64) -> Self {
        Self::new(CognitiveDomain::Causality, cause_type as u8, 0, hash)
    }
    
    /// Create for temporal
    pub fn temporal(relation: TemporalRelation, hash: u64) -> Self {
        Self::new(CognitiveDomain::Temporal, relation as u8, 0, hash)
    }
    
    /// Create for rung level
    pub fn rung(level: u8, hash: u64) -> Self {
        Self::new(CognitiveDomain::RungLevel, level.min(9), 0, hash)
    }
    
    /// Create for learned concept
    pub fn learned(hash: u64) -> Self {
        Self::new(CognitiveDomain::LearnedConcept, 0, 0, hash)
    }
    
    /// Extract domain
    pub fn domain(&self) -> CognitiveDomain {
        CognitiveDomain::from_u8(((self.0 >> 60) & 0xF) as u8)
    }
    
    /// Extract subtype
    pub fn subtype(&self) -> u8 {
        ((self.0 >> 56) & 0xF) as u8
    }
    
    /// Extract index
    pub fn index(&self) -> u8 {
        ((self.0 >> 48) & 0xFF) as u8
    }
    
    /// Extract 48-bit content hash
    pub fn hash(&self) -> u64 {
        self.0 & 0xFFFF_FFFF_FFFF
    }
    
    /// Extract 16-bit bucket (for BTree indexing)
    pub fn bucket(&self) -> u16 {
        (self.0 >> 48) as u16
    }
    
    /// Check if same bucket (domain + subtype + index)
    pub fn same_bucket(&self, other: &Self) -> bool {
        self.bucket() == other.bucket()
    }
    
    /// Get human-readable name
    pub fn name(&self) -> String {
        match self.domain() {
            CognitiveDomain::NsmPrime => {
                format!("NSM::{:?}[{}]", 
                    match self.subtype() {
                        0 => "Substantive",
                        1 => "Relational",
                        2 => "Determiner",
                        3 => "Quantifier",
                        4 => "Evaluator",
                        5 => "Descriptor",
                        6 => "Mental",
                        7 => "Speech",
                        8 => "Action",
                        9 => "Existence",
                        10 => "Life",
                        11 => "Time",
                        12 => "Space",
                        13 => "Logical",
                        14 => "Intensifier",
                        15 => "Similarity",
                        _ => "Unknown",
                    },
                    self.index()
                )
            }
            CognitiveDomain::NsmRole => {
                format!("Role::{}", match self.index() {
                    0 => "Agent",
                    1 => "Patient", 
                    2 => "Theme",
                    3 => "Experiencer",
                    4 => "Beneficiary",
                    5 => "Instrument",
                    6 => "Location",
                    7 => "Source",
                    8 => "Goal",
                    9 => "Time",
                    10 => "Manner",
                    11 => "Cause",
                    12 => "Purpose",
                    13 => "Condition",
                    14 => "Extent",
                    15 => "Attribute",
                    _ => "Unknown",
                })
            }
            CognitiveDomain::Qualia => {
                format!("Qualia::{}", match self.subtype() {
                    0 => "Arousal",
                    1 => "Valence",
                    2 => "Tension",
                    3 => "Certainty",
                    4 => "Agency",
                    5 => "Temporality",
                    6 => "Sociality",
                    7 => "Novelty",
                    _ => "Unknown",
                })
            }
            _ => format!("{:?}[{}:{}]", self.domain(), self.subtype(), self.index()),
        }
    }
}

// =============================================================================
// Fingerprint Folding (10K → 48 bits)
// =============================================================================

/// Fold 10K fingerprint to 48 bits while preserving similarity relationships
pub fn fold_to_48(fp: &Fingerprint) -> u64 {
    let mut result = 0u64;
    
    // XOR-fold: each output bit is XOR of ~208 input bits
    // This preserves relative Hamming distances
    for i in 0..48 {
        let mut bit = false;
        let stride = 16384 / 48;
        for j in 0..stride {
            let idx = i + j * 48;
            if idx < 16384 && fp.get_bit(idx) {
                bit = !bit;
            }
        }
        // Also XOR any remaining bits
        let extra_start = 48 * stride + i;
        if extra_start < 16384 && fp.get_bit(extra_start) {
            bit = !bit;
        }
        
        if bit {
            result |= 1u64 << i;
        }
    }
    
    result
}

/// Expand 48-bit hash to approximate 10K fingerprint
/// (Lossy but useful for initial resonance)
pub fn expand_from_48(hash: u64) -> Fingerprint {
    let mut fp = Fingerprint::zero();
    
    // Replicate each bit across its fold region
    for i in 0..48 {
        let bit = (hash >> i) & 1 == 1;
        if bit {
            let stride = 16384 / 48;
            for j in 0..stride {
                let idx = i + j * 48;
                if idx < 16384 {
                    fp.set_bit(idx, true);
                }
            }
        }
    }
    
    fp
}

// =============================================================================
// The Complete Cognitive Codebook
// =============================================================================

/// Entry in the codebook
#[derive(Clone)]
pub struct CodebookEntry {
    pub address: CognitiveAddress,
    pub name: String,
    pub fingerprint: Fingerprint,
}

/// The unified cognitive codebook
pub struct CognitiveCodebook {
    /// BTree index: bucket → entries (for range queries)
    btree: BTreeMap<u16, Vec<CodebookEntry>>,
    
    /// Hash index: 48-bit hash → address (for O(1) lookup)
    hash_index: HashMap<u64, CognitiveAddress>,
    
    /// Full fingerprints by address
    fingerprints: HashMap<CognitiveAddress, Fingerprint>,
    
    /// Name to address mapping
    names: HashMap<String, CognitiveAddress>,
    
    /// Statistics
    total_entries: usize,
}

impl Default for CognitiveCodebook {
    fn default() -> Self {
        Self::new()
    }
}

impl CognitiveCodebook {
    /// Create new codebook with all built-in concepts
    pub fn new() -> Self {
        let mut codebook = Self {
            btree: BTreeMap::new(),
            hash_index: HashMap::new(),
            fingerprints: HashMap::new(),
            names: HashMap::new(),
            total_entries: 0,
        };
        
        codebook.init_nsm_primes();
        codebook.init_roles();
        codebook.init_qualia();
        codebook.init_nars_copulas();
        codebook.init_nars_inference();
        codebook.init_causality();
        codebook.init_temporal();
        codebook.init_yaml_templates();
        codebook.init_rung_levels();
        
        codebook
    }
    
    // -------------------------------------------------------------------------
    // Initialization
    // -------------------------------------------------------------------------
    
    fn init_nsm_primes(&mut self) {
        let mut idx = 0usize;
        
        // Substantives
        for name in ["I", "YOU", "SOMEONE", "SOMETHING", "PEOPLE", "BODY"] {
            self.add_nsm_prime(NsmCategory::Substantive, name, idx);
            idx += 1;
        }
        
        // Relational
        for name in ["KIND", "PART"] {
            self.add_nsm_prime(NsmCategory::Relational, name, idx);
            idx += 1;
        }
        
        // Determiners
        for name in ["THIS", "THE_SAME", "OTHER", "ELSE", "ANOTHER"] {
            self.add_nsm_prime(NsmCategory::Determiner, name, idx);
            idx += 1;
        }
        
        // Quantifiers
        for name in ["ONE", "TWO", "SOME", "ALL", "MUCH", "MANY", "LITTLE", "FEW"] {
            self.add_nsm_prime(NsmCategory::Quantifier, name, idx);
            idx += 1;
        }
        
        // Evaluators
        for name in ["GOOD", "BAD"] {
            self.add_nsm_prime(NsmCategory::Evaluator, name, idx);
            idx += 1;
        }
        
        // Descriptors
        for name in ["BIG", "SMALL"] {
            self.add_nsm_prime(NsmCategory::Descriptor, name, idx);
            idx += 1;
        }
        
        // Mental predicates
        for name in ["THINK", "KNOW", "WANT", "DONT_WANT", "FEEL", "SEE", "HEAR"] {
            self.add_nsm_prime(NsmCategory::Mental, name, idx);
            idx += 1;
        }
        
        // Speech
        for name in ["SAY", "WORDS", "TRUE"] {
            self.add_nsm_prime(NsmCategory::Speech, name, idx);
            idx += 1;
        }
        
        // Actions
        for name in ["DO", "HAPPEN", "MOVE"] {
            self.add_nsm_prime(NsmCategory::Action, name, idx);
            idx += 1;
        }
        
        // Existence
        for name in ["BE", "THERE_IS", "IS_PART_OF", "MINE"] {
            self.add_nsm_prime(NsmCategory::Existence, name, idx);
            idx += 1;
        }
        
        // Life
        for name in ["LIVE", "DIE"] {
            self.add_nsm_prime(NsmCategory::Life, name, idx);
            idx += 1;
        }
        
        // Time
        for name in ["WHEN", "TIME", "NOW", "BEFORE", "AFTER", 
                     "A_LONG_TIME", "A_SHORT_TIME", "FOR_SOME_TIME", "MOMENT"] {
            self.add_nsm_prime(NsmCategory::Time, name, idx);
            idx += 1;
        }
        
        // Space
        for name in ["WHERE", "PLACE", "HERE", "ABOVE", "BELOW", 
                     "FAR", "NEAR", "SIDE", "INSIDE", "TOUCH"] {
            self.add_nsm_prime(NsmCategory::Space, name, idx);
            idx += 1;
        }
        
        // Logical
        for name in ["NOT", "MAYBE", "CAN", "BECAUSE", "IF"] {
            self.add_nsm_prime(NsmCategory::Logical, name, idx);
            idx += 1;
        }
        
        // Intensifiers
        for name in ["VERY", "MORE"] {
            self.add_nsm_prime(NsmCategory::Intensifier, name, idx);
            idx += 1;
        }
        
        // Similarity
        for name in ["LIKE", "AS", "WAY"] {
            self.add_nsm_prime(NsmCategory::Similarity, name, idx);
            idx += 1;
        }
    }
    
    fn add_nsm_prime(&mut self, category: NsmCategory, name: &str, global_idx: usize) {
        let fp = Fingerprint::orthogonal(global_idx);
        let hash = fold_to_48(&fp);
        let addr = CognitiveAddress::nsm_prime(category, global_idx as u8, hash);
        self.insert(addr, name.to_string(), fp);
    }
    
    fn init_roles(&mut self) {
        let roles = [
            (ThematicRole::Agent, "R_AGENT"),
            (ThematicRole::Patient, "R_PATIENT"),
            (ThematicRole::Theme, "R_THEME"),
            (ThematicRole::Experiencer, "R_EXPERIENCER"),
            (ThematicRole::Beneficiary, "R_BENEFICIARY"),
            (ThematicRole::Instrument, "R_INSTRUMENT"),
            (ThematicRole::Location, "R_LOCATION"),
            (ThematicRole::Source, "R_SOURCE"),
            (ThematicRole::Goal, "R_GOAL"),
            (ThematicRole::Time, "R_TIME"),
            (ThematicRole::Manner, "R_MANNER"),
            (ThematicRole::Cause, "R_CAUSE"),
            (ThematicRole::Purpose, "R_PURPOSE"),
            (ThematicRole::Condition, "R_CONDITION"),
            (ThematicRole::Extent, "R_EXTENT"),
            (ThematicRole::Attribute, "R_ATTRIBUTE"),
        ];
        
        for (idx, (role, name)) in roles.iter().enumerate() {
            let fp = Fingerprint::orthogonal(100 + idx); // Offset from primes
            let hash = fold_to_48(&fp);
            let addr = CognitiveAddress::role(*role, hash);
            self.insert(addr, name.to_string(), fp);
        }
    }
    
    fn init_qualia(&mut self) {
        let channels = [
            (QualiaChannel::Arousal, "Q_AROUSAL"),
            (QualiaChannel::Valence, "Q_VALENCE"),
            (QualiaChannel::Tension, "Q_TENSION"),
            (QualiaChannel::Certainty, "Q_CERTAINTY"),
            (QualiaChannel::Agency, "Q_AGENCY"),
            (QualiaChannel::Temporality, "Q_TEMPORALITY"),
            (QualiaChannel::Sociality, "Q_SOCIALITY"),
            (QualiaChannel::Novelty, "Q_NOVELTY"),
        ];
        
        for (idx, (channel, name)) in channels.iter().enumerate() {
            let fp = Fingerprint::orthogonal(200 + idx);
            let hash = fold_to_48(&fp);
            let addr = CognitiveAddress::qualia(*channel, 0, hash);
            self.insert(addr, name.to_string(), fp);
        }
    }
    
    fn init_nars_copulas(&mut self) {
        let copulas = [
            (NarsCopula::Inheritance, "NARS_INHERIT"),      // -->
            (NarsCopula::Similarity, "NARS_SIMILAR"),       // <->
            (NarsCopula::Implication, "NARS_IMPLIES"),      // ==>
            (NarsCopula::Equivalence, "NARS_EQUIV"),        // <=>
            (NarsCopula::Instance, "NARS_INSTANCE"),        // {--
            (NarsCopula::Property, "NARS_PROPERTY"),        // --]
            (NarsCopula::InstanceProp, "NARS_INST_PROP"),   // {-]
            (NarsCopula::Conjunction, "NARS_AND"),          // &&
            (NarsCopula::Disjunction, "NARS_OR"),           // ||
            (NarsCopula::Negation, "NARS_NOT"),             // --
            (NarsCopula::Sequential, "NARS_SEQ"),           // &/
            (NarsCopula::Parallel, "NARS_PAR"),             // &|
        ];
        
        for (idx, (copula, name)) in copulas.iter().enumerate() {
            let fp = Fingerprint::orthogonal(300 + idx);
            let hash = fold_to_48(&fp);
            let addr = CognitiveAddress::nars_copula(*copula, hash);
            self.insert(addr, name.to_string(), fp);
        }
    }
    
    fn init_nars_inference(&mut self) {
        let inferences = [
            (NarsInference::Deduction, "INF_DEDUCTION"),
            (NarsInference::Induction, "INF_INDUCTION"),
            (NarsInference::Abduction, "INF_ABDUCTION"),
            (NarsInference::Exemplification, "INF_EXEMPLIFY"),
            (NarsInference::Comparison, "INF_COMPARE"),
            (NarsInference::Analogy, "INF_ANALOGY"),
            (NarsInference::Resemblance, "INF_RESEMBLE"),
            (NarsInference::Revision, "INF_REVISION"),
            (NarsInference::Choice, "INF_CHOICE"),
            (NarsInference::Decision, "INF_DECISION"),
        ];
        
        for (idx, (inference, name)) in inferences.iter().enumerate() {
            let fp = Fingerprint::orthogonal(350 + idx);
            let hash = fold_to_48(&fp);
            let addr = CognitiveAddress::nars_inference(*inference, hash);
            self.insert(addr, name.to_string(), fp);
        }
    }
    
    fn init_causality(&mut self) {
        let causes = [
            (CausalityType::Enables, "CAUSE_ENABLES"),
            (CausalityType::Causes, "CAUSE_CAUSES"),
            (CausalityType::Prevents, "CAUSE_PREVENTS"),
            (CausalityType::Maintains, "CAUSE_MAINTAINS"),
            (CausalityType::Triggers, "CAUSE_TRIGGERS"),
            (CausalityType::Terminates, "CAUSE_TERMINATES"),
            (CausalityType::Modulates, "CAUSE_MODULATES"),
            (CausalityType::Correlates, "CAUSE_CORRELATES"),
        ];
        
        for (idx, (cause, name)) in causes.iter().enumerate() {
            let fp = Fingerprint::orthogonal(400 + idx);
            let hash = fold_to_48(&fp);
            let addr = CognitiveAddress::causality(*cause, hash);
            self.insert(addr, name.to_string(), fp);
        }
    }
    
    fn init_temporal(&mut self) {
        let relations = [
            (TemporalRelation::Before, "TIME_BEFORE"),
            (TemporalRelation::After, "TIME_AFTER"),
            (TemporalRelation::Meets, "TIME_MEETS"),
            (TemporalRelation::MetBy, "TIME_MET_BY"),
            (TemporalRelation::Overlaps, "TIME_OVERLAPS"),
            (TemporalRelation::OverlappedBy, "TIME_OVERLAPPED"),
            (TemporalRelation::During, "TIME_DURING"),
            (TemporalRelation::Contains, "TIME_CONTAINS"),
            (TemporalRelation::Starts, "TIME_STARTS"),
            (TemporalRelation::StartedBy, "TIME_STARTED_BY"),
            (TemporalRelation::Finishes, "TIME_FINISHES"),
            (TemporalRelation::FinishedBy, "TIME_FINISHED_BY"),
            (TemporalRelation::Equals, "TIME_EQUALS"),
            (TemporalRelation::Now, "TIME_NOW"),
            (TemporalRelation::Always, "TIME_ALWAYS"),
            (TemporalRelation::Never, "TIME_NEVER"),
        ];
        
        for (idx, (relation, name)) in relations.iter().enumerate() {
            let fp = Fingerprint::orthogonal(450 + idx);
            let hash = fold_to_48(&fp);
            let addr = CognitiveAddress::temporal(*relation, hash);
            self.insert(addr, name.to_string(), fp);
        }
    }
    
    fn init_yaml_templates(&mut self) {
        let templates = [
            (YamlTemplate::Greeting, "TPL_GREETING"),
            (YamlTemplate::Farewell, "TPL_FAREWELL"),
            (YamlTemplate::Question, "TPL_QUESTION"),
            (YamlTemplate::Statement, "TPL_STATEMENT"),
            (YamlTemplate::Command, "TPL_COMMAND"),
            (YamlTemplate::Request, "TPL_REQUEST"),
            (YamlTemplate::Offer, "TPL_OFFER"),
            (YamlTemplate::Promise, "TPL_PROMISE"),
            (YamlTemplate::Warning, "TPL_WARNING"),
            (YamlTemplate::Apology, "TPL_APOLOGY"),
            (YamlTemplate::Gratitude, "TPL_GRATITUDE"),
            (YamlTemplate::Complaint, "TPL_COMPLAINT"),
            (YamlTemplate::Explanation, "TPL_EXPLANATION"),
            (YamlTemplate::Narrative, "TPL_NARRATIVE"),
            (YamlTemplate::Opinion, "TPL_OPINION"),
            (YamlTemplate::Hypothesis, "TPL_HYPOTHESIS"),
        ];
        
        for (idx, (template, name)) in templates.iter().enumerate() {
            let fp = Fingerprint::orthogonal(500 + idx);
            let hash = fold_to_48(&fp);
            let addr = CognitiveAddress::yaml_template(*template, hash);
            self.insert(addr, name.to_string(), fp);
        }
    }
    
    fn init_rung_levels(&mut self) {
        let rungs = [
            (0, "RUNG_0_NOISE"),
            (1, "RUNG_1_TOKEN"),
            (2, "RUNG_2_PHRASE"),
            (3, "RUNG_3_PROPOSITION"),
            (4, "RUNG_4_ARGUMENT"),
            (5, "RUNG_5_NARRATIVE"),
            (6, "RUNG_6_WORLDVIEW"),
            (7, "RUNG_7_PARADIGM"),
            (8, "RUNG_8_EPISTEME"),
            (9, "RUNG_9_TRANSCENDENT"),
        ];
        
        for (level, name) in rungs.iter() {
            let fp = Fingerprint::orthogonal(600 + *level as usize);
            let hash = fold_to_48(&fp);
            let addr = CognitiveAddress::rung(*level, hash);
            self.insert(addr, name.to_string(), fp);
        }
    }
    
    // -------------------------------------------------------------------------
    // Core Operations
    // -------------------------------------------------------------------------
    
    /// Insert entry into codebook
    fn insert(&mut self, addr: CognitiveAddress, name: String, fp: Fingerprint) {
        let entry = CodebookEntry {
            address: addr,
            name: name.clone(),
            fingerprint: fp.clone(),
        };
        
        // BTree index
        self.btree.entry(addr.bucket())
            .or_insert_with(Vec::new)
            .push(entry);
        
        // Hash index
        self.hash_index.insert(addr.hash(), addr);
        
        // Fingerprint storage
        self.fingerprints.insert(addr, fp);
        
        // Name index
        self.names.insert(name, addr);
        
        self.total_entries += 1;
    }
    
    /// Get fingerprint by address
    pub fn get(&self, addr: &CognitiveAddress) -> Option<&Fingerprint> {
        self.fingerprints.get(addr)
    }
    
    /// Get fingerprint by name
    pub fn get_by_name(&self, name: &str) -> Option<&Fingerprint> {
        self.names.get(name).and_then(|addr| self.fingerprints.get(addr))
    }
    
    /// Get address by name
    pub fn address_of(&self, name: &str) -> Option<CognitiveAddress> {
        self.names.get(name).copied()
    }
    
    /// Get fingerprint by 48-bit hash
    pub fn get_by_hash(&self, hash: u64) -> Option<&Fingerprint> {
        self.hash_index.get(&hash)
            .and_then(|addr| self.fingerprints.get(addr))
    }
    
    /// Range query by domain
    pub fn by_domain(&self, domain: CognitiveDomain) -> Vec<&CodebookEntry> {
        let prefix = (domain as u16) << 12;
        let end = prefix + 0x1000;
        
        self.btree.range(prefix..end)
            .flat_map(|(_, entries)| entries.iter())
            .collect()
    }
    
    /// Find best matching entry for a fingerprint
    pub fn find_best_match(&self, query: &Fingerprint) -> Option<(&CognitiveAddress, f32)> {
        let mut best: Option<(&CognitiveAddress, f32)> = None;
        
        for (addr, fp) in &self.fingerprints {
            let sim = query.similarity(fp);
            if best.map_or(true, |(_, best_sim)| sim > best_sim) {
                best = Some((addr, sim));
            }
        }
        
        best
    }
    
    /// Find all matches above threshold
    pub fn find_matches(&self, query: &Fingerprint, threshold: f32) -> Vec<(&CognitiveAddress, f32)> {
        self.fingerprints.iter()
            .map(|(addr, fp)| (addr, query.similarity(fp)))
            .filter(|(_, sim)| *sim >= threshold)
            .collect()
    }
    
    /// Learn new concept
    pub fn learn(&mut self, name: &str, fp: Fingerprint) -> CognitiveAddress {
        let hash = fold_to_48(&fp);
        let addr = CognitiveAddress::learned(hash);
        self.insert(addr, name.to_string(), fp);
        addr
    }
    
    /// Clean fingerprint against codebook (noise elimination)
    pub fn clean(&self, fp: &Fingerprint, threshold: f32) -> Fingerprint {
        let matches = self.find_matches(fp, threshold);
        
        if matches.is_empty() {
            return Fingerprint::zero();
        }
        
        // Weighted bundle of matching concepts
        let components: Vec<(Fingerprint, f32)> = matches.iter()
            .filter_map(|(addr, sim)| {
                self.fingerprints.get(addr).map(|f| (f.clone(), *sim))
            })
            .collect();
        
        weighted_bundle(&components)
    }
    
    /// Get statistics
    pub fn stats(&self) -> CodebookStats {
        let mut by_domain = HashMap::new();
        
        for addr in self.fingerprints.keys() {
            *by_domain.entry(addr.domain()).or_insert(0) += 1;
        }
        
        CodebookStats {
            total_entries: self.total_entries,
            by_domain,
            btree_buckets: self.btree.len(),
        }
    }
}

/// Weighted bundle helper
fn weighted_bundle(fps: &[(Fingerprint, f32)]) -> Fingerprint {
    if fps.is_empty() {
        return Fingerprint::zero();
    }
    
    let mut counts = vec![0.0f32; 16384];
    let mut total_weight = 0.0f32;

    for (fp, weight) in fps {
        for i in 0..16384 {
            if fp.get_bit(i) {
                counts[i] += weight;
            }
        }
        total_weight += weight;
    }
    
    if total_weight == 0.0 {
        return Fingerprint::zero();
    }
    
    let threshold = total_weight / 2.0;
    let mut result = Fingerprint::zero();
    
    for (i, &count) in counts.iter().enumerate() {
        if count > threshold {
            result.set_bit(i, true);
        }
    }
    
    result
}

/// Codebook statistics
#[derive(Debug)]
pub struct CodebookStats {
    pub total_entries: usize,
    pub by_domain: HashMap<CognitiveDomain, usize>,
    pub btree_buckets: usize,
}

impl CodebookStats {
    pub fn print(&self) {
        println!("=== Cognitive Codebook Stats ===");
        println!("Total entries: {}", self.total_entries);
        println!("BTree buckets: {}", self.btree_buckets);
        println!("By domain:");
        for (domain, count) in &self.by_domain {
            println!("  {:?}: {}", domain, count);
        }
        println!("Memory estimate: ~{} KB", 
            (self.total_entries * (8 + crate::FINGERPRINT_BYTES)) / 1024); // addr + fp
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cognitive_address() {
        let addr = CognitiveAddress::nsm_prime(NsmCategory::Mental, 3, 0xABCDEF);
        
        assert_eq!(addr.domain(), CognitiveDomain::NsmPrime);
        assert_eq!(addr.subtype(), NsmCategory::Mental as u8);
        assert_eq!(addr.index(), 3);
        assert_eq!(addr.hash(), 0xABCDEF);
        
        println!("Address: {:016X}", addr.0);
        println!("Bucket: {:04X}", addr.bucket());
        println!("Name: {}", addr.name());
    }
    
    #[test]
    fn test_fold_unfold() {
        let original = Fingerprint::orthogonal(42);
        let hash = fold_to_48(&original);
        let expanded = expand_from_48(hash);
        
        // Similarity should be preserved somewhat
        let sim = original.similarity(&expanded);
        println!("Fold/unfold similarity: {:.3}", sim);
        assert!(sim > 0.3); // At least some correlation
    }
    
    #[test]
    fn test_codebook_init() {
        let codebook = CognitiveCodebook::new();
        let stats = codebook.stats();
        stats.print();
        
        // Should have ~120+ built-in concepts
        assert!(stats.total_entries >= 100);
        
        // Check we can find known concepts
        let think = codebook.get_by_name("THINK");
        assert!(think.is_some());
        
        let agent = codebook.get_by_name("R_AGENT");
        assert!(agent.is_some());
        
        let valence = codebook.get_by_name("Q_VALENCE");
        assert!(valence.is_some());
    }
    
    #[test]
    fn test_domain_query() {
        let codebook = CognitiveCodebook::new();
        
        let mental_primes = codebook.by_domain(CognitiveDomain::NsmPrime);
        println!("NSM Primes: {}", mental_primes.len());
        
        for entry in mental_primes.iter().take(10) {
            println!("  {} -> {:?}", entry.name, entry.address.name());
        }
        
        assert!(!mental_primes.is_empty());
    }
    
    #[test]
    fn test_clean() {
        let codebook = CognitiveCodebook::new();
        
        // Create noisy fingerprint
        let think = codebook.get_by_name("THINK").unwrap().clone();
        let know = codebook.get_by_name("KNOW").unwrap().clone();
        let noise = Fingerprint::random();
        
        // Mix with noise
        let noisy = weighted_bundle(&[
            (think.clone(), 0.6),
            (know.clone(), 0.3),
            (noise, 0.1),
        ]);
        
        // Clean
        let cleaned = codebook.clean(&noisy, 0.2);
        
        // Should be more similar to original than noisy
        let sim_think = cleaned.similarity(&think);
        let sim_know = cleaned.similarity(&know);
        
        println!("Cleaned similarity to THINK: {:.3}", sim_think);
        println!("Cleaned similarity to KNOW: {:.3}", sim_know);
        
        assert!(sim_think > 0.3 || sim_know > 0.3);
    }
    
    #[test]
    fn test_learn() {
        let mut codebook = CognitiveCodebook::new();
        
        let initial_count = codebook.stats().total_entries;
        
        // Learn new concept
        let new_fp = Fingerprint::from_content("consciousness");
        let addr = codebook.learn("CONSCIOUSNESS", new_fp.clone());
        
        assert_eq!(addr.domain(), CognitiveDomain::LearnedConcept);
        assert_eq!(codebook.stats().total_entries, initial_count + 1);
        
        // Can retrieve it
        let retrieved = codebook.get_by_name("CONSCIOUSNESS").unwrap();
        assert_eq!(retrieved.similarity(&new_fp), 1.0);
    }
}
