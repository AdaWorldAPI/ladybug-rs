//! CognitiveAddress — 64-bit address for the cognitive codebook.
//!
//! Layout: `[Domain:4][Subtype:4][Index:8][Hash:48]`

/// 64-bit cognitive address.
///
/// Encodes domain, subtype, index, and a 48-bit content hash.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CognitiveAddress(pub u64);

/// Primary domain classifier (4 bits, 16 domains).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CognitiveDomain {
    NsmPrime = 0x0,
    NsmRole = 0x1,
    SpoSubject = 0x2,
    SpoPredicate = 0x3,
    SpoObject = 0x4,
    Qualia = 0x5,
    NarsTerm = 0x6,
    NarsInference = 0x7,
    Causality = 0x8,
    Temporal = 0x9,
    YamlTemplate = 0xA,
    RungLevel = 0xB,
    CrystalPos = 0xC,
    LearnedConcept = 0xD,
    MetaPattern = 0xE,
    Reserved = 0xF,
}

impl CognitiveDomain {
    pub fn from_u8(v: u8) -> Self {
        match v & 0xF {
            0x0 => CognitiveDomain::NsmPrime,
            0x1 => CognitiveDomain::NsmRole,
            0x2 => CognitiveDomain::SpoSubject,
            0x3 => CognitiveDomain::SpoPredicate,
            0x4 => CognitiveDomain::SpoObject,
            0x5 => CognitiveDomain::Qualia,
            0x6 => CognitiveDomain::NarsTerm,
            0x7 => CognitiveDomain::NarsInference,
            0x8 => CognitiveDomain::Causality,
            0x9 => CognitiveDomain::Temporal,
            0xA => CognitiveDomain::YamlTemplate,
            0xB => CognitiveDomain::RungLevel,
            0xC => CognitiveDomain::CrystalPos,
            0xD => CognitiveDomain::LearnedConcept,
            0xE => CognitiveDomain::MetaPattern,
            _ => CognitiveDomain::Reserved,
        }
    }
}

/// Wierzbicka's Natural Semantic Metalanguage categories.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NsmCategory {
    Substantive = 0x0,
    Relational = 0x1,
    Determiner = 0x2,
    Quantifier = 0x3,
    Evaluator = 0x4,
    Descriptor = 0x5,
    Mental = 0x6,
    Speech = 0x7,
    Action = 0x8,
    Existence = 0x9,
    Life = 0xA,
    Time = 0xB,
    Space = 0xC,
    Logical = 0xD,
    Intensifier = 0xE,
    Similarity = 0xF,
}

/// Thematic roles (semantic case roles).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThematicRole {
    Agent = 0x0,
    Patient = 0x1,
    Theme = 0x2,
    Experiencer = 0x3,
    Beneficiary = 0x4,
    Instrument = 0x5,
    Location = 0x6,
    Source = 0x7,
    Goal = 0x8,
    Time = 0x9,
    Manner = 0xA,
    Cause = 0xB,
    Purpose = 0xC,
    Condition = 0xD,
    Extent = 0xE,
    Attribute = 0xF,
}

/// NARS copulas (term relations).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NarsCopula {
    Inheritance = 0x0,
    Similarity = 0x1,
    Implication = 0x2,
    Equivalence = 0x3,
    Instance = 0x4,
    Property = 0x5,
    InstanceProp = 0x6,
    Conjunction = 0x7,
    Disjunction = 0x8,
    Negation = 0x9,
    Sequential = 0xA,
    Parallel = 0xB,
}

/// NARS inference rules.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NarsInference {
    Deduction = 0x0,
    Induction = 0x1,
    Abduction = 0x2,
    Exemplification = 0x3,
    Comparison = 0x4,
    Analogy = 0x5,
    Resemblance = 0x6,
    Revision = 0x7,
    Choice = 0x8,
    Decision = 0x9,
}

/// Pearl's causal relation types.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CausalityType {
    Enables = 0x0,
    Causes = 0x1,
    Prevents = 0x2,
    Maintains = 0x3,
    Triggers = 0x4,
    Terminates = 0x5,
    Modulates = 0x6,
    Correlates = 0x7,
}

/// Allen's interval algebra relations.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TemporalRelation {
    Before = 0x0,
    After = 0x1,
    Meets = 0x2,
    MetBy = 0x3,
    Overlaps = 0x4,
    OverlappedBy = 0x5,
    During = 0x6,
    Contains = 0x7,
    Starts = 0x8,
    StartedBy = 0x9,
    Finishes = 0xA,
    FinishedBy = 0xB,
    Equals = 0xC,
    Now = 0xD,
    Always = 0xE,
    Never = 0xF,
}

/// Speech act templates.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum YamlTemplate {
    Greeting = 0x0,
    Farewell = 0x1,
    Question = 0x2,
    Statement = 0x3,
    Command = 0x4,
    Request = 0x5,
    Offer = 0x6,
    Promise = 0x7,
    Warning = 0x8,
    Apology = 0x9,
    Gratitude = 0xA,
    Complaint = 0xB,
    Explanation = 0xC,
    Narrative = 0xD,
    Opinion = 0xE,
    Hypothesis = 0xF,
}

/// Russell Circumplex affect channels.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QualiaChannel {
    Arousal = 0x0,
    Valence = 0x1,
    Tension = 0x2,
    Certainty = 0x3,
    Agency = 0x4,
    Temporality = 0x5,
    Sociality = 0x6,
    Novelty = 0x7,
}

// ============================================================================
// CognitiveAddress constructors and accessors
// ============================================================================

impl CognitiveAddress {
    /// Create from raw components.
    pub fn new(domain: CognitiveDomain, subtype: u8, index: u8, hash: u64) -> Self {
        let addr = ((domain as u64) << 60)
            | (((subtype & 0xF) as u64) << 56)
            | ((index as u64) << 48)
            | (hash & 0x0000_FFFF_FFFF_FFFF);
        CognitiveAddress(addr)
    }

    pub fn nsm_prime(category: NsmCategory, index: u8, hash: u64) -> Self {
        Self::new(CognitiveDomain::NsmPrime, category as u8, index, hash)
    }

    pub fn role(role: ThematicRole, hash: u64) -> Self {
        Self::new(CognitiveDomain::NsmRole, role as u8, 0, hash)
    }

    pub fn qualia(channel: QualiaChannel, level: u8, hash: u64) -> Self {
        Self::new(CognitiveDomain::Qualia, channel as u8, level, hash)
    }

    pub fn nars_copula(copula: NarsCopula, hash: u64) -> Self {
        Self::new(CognitiveDomain::NarsTerm, copula as u8, 0, hash)
    }

    pub fn nars_inference(inference: NarsInference, hash: u64) -> Self {
        Self::new(CognitiveDomain::NarsInference, inference as u8, 0, hash)
    }

    pub fn yaml_template(template: YamlTemplate, hash: u64) -> Self {
        Self::new(CognitiveDomain::YamlTemplate, template as u8, 0, hash)
    }

    pub fn causality(cause_type: CausalityType, hash: u64) -> Self {
        Self::new(CognitiveDomain::Causality, cause_type as u8, 0, hash)
    }

    pub fn temporal(relation: TemporalRelation, hash: u64) -> Self {
        Self::new(CognitiveDomain::Temporal, relation as u8, 0, hash)
    }

    pub fn rung(level: u8, hash: u64) -> Self {
        Self::new(CognitiveDomain::RungLevel, 0, level, hash)
    }

    pub fn learned(hash: u64) -> Self {
        Self::new(CognitiveDomain::LearnedConcept, 0, 0, hash)
    }

    // -- Accessors --

    /// Extract domain from bits [63:60].
    pub fn domain(&self) -> CognitiveDomain {
        CognitiveDomain::from_u8((self.0 >> 60) as u8)
    }

    /// Extract subtype from bits [59:56].
    pub fn subtype(&self) -> u8 {
        ((self.0 >> 56) & 0xF) as u8
    }

    /// Extract index from bits [55:48].
    pub fn index(&self) -> u8 {
        ((self.0 >> 48) & 0xFF) as u8
    }

    /// Extract 48-bit content hash from bits [47:0].
    pub fn hash(&self) -> u64 {
        self.0 & 0x0000_FFFF_FFFF_FFFF
    }

    /// 16-bit bucket (domain + subtype + index).
    pub fn bucket(&self) -> u16 {
        (self.0 >> 48) as u16
    }

    /// Check if same bucket.
    pub fn same_bucket(&self, other: &Self) -> bool {
        self.bucket() == other.bucket()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cognitive_address_encoding() {
        let addr = CognitiveAddress::qualia(QualiaChannel::Arousal, 128, 0xABCDEF);
        assert_eq!(addr.domain(), CognitiveDomain::Qualia);
        assert_eq!(addr.subtype(), 0);
        assert_eq!(addr.index(), 128);
        assert_eq!(addr.hash() & 0xFFFFFF, 0xABCDEF);
    }

    #[test]
    fn test_domain_round_trip() {
        for d in 0..=15u8 {
            let domain = CognitiveDomain::from_u8(d);
            let addr = CognitiveAddress::new(domain, 0, 0, 0);
            assert_eq!(addr.domain(), domain);
        }
    }

    #[test]
    fn test_bucket_equality() {
        let a = CognitiveAddress::nsm_prime(NsmCategory::Mental, 5, 0x111);
        let b = CognitiveAddress::nsm_prime(NsmCategory::Mental, 5, 0x222);
        assert!(a.same_bucket(&b));
    }
}
