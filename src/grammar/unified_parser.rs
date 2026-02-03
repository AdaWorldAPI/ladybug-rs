//! Unified Grammar Parser - LangExtract Without LLM
//!
//! Replaces external LLM calls with pure NSM + Template + NARS stack.
//! The key insight: we don't need LLMs because we have:
//! 1. NSM primitives as semantic atoms
//! 2. Construction grammar templates for syntactic patterns
//! 3. CollapseGate for confidence-based decisions
//! 4. NARS for inference and calibration
//!
//! ## Architecture
//!
//! ```text
//!                      INPUT TEXT
//!                          │
//!                          ▼
//!               ┌──────────────────────┐
//!               │   TOKENIZE + TAG     │
//!               │   (tree-sitter/POS)  │
//!               └──────────┬───────────┘
//!                          │
//!          ┌───────────────┼───────────────┐
//!          ▼               ▼               ▼
//!    ┌──────────┐   ┌──────────┐   ┌──────────┐
//!    │   NSM    │   │ TEMPLATE │   │  QUALIA  │
//!    │ Activate │   │  Match   │   │ Extract  │
//!    └────┬─────┘   └────┬─────┘   └────┬─────┘
//!         │              │              │
//!         └──────────────┼──────────────┘
//!                        ▼
//!               ┌──────────────────────┐
//!               │   GRAMMAR TRIANGLE   │
//!               │ NSM × Template × Q   │
//!               └──────────┬───────────┘
//!                          │
//!                          ▼
//!               ┌──────────────────────┐
//!               │    COLLAPSE GATE     │
//!               │  SD < 0.15 → FLOW    │
//!               │  0.15-0.35 → HOLD    │
//!               │  SD > 0.35 → BLOCK   │
//!               └──────────┬───────────┘
//!                          │
//!          ┌───────────────┼───────────────┐
//!          ▼               ▼               ▼
//!       FLOW:          HOLD:           BLOCK:
//!    Commit to       Store in       Check User
//!    Node Zone      Crystal TTL    Calibration
//! ```

use std::collections::HashMap;

use crate::core::Fingerprint;
use crate::storage::bind_space::{
    Addr, PREFIX_NSM, PREFIX_TEMPLATES, PREFIX_SPEECH_ACTS, PREFIX_CALIBRATION,
    nsm_slots, template_slots, speech_act_slots,
};

// =============================================================================
// GRAMMAR TRIANGLE (Unified Output)
// =============================================================================

/// The Grammar Triangle integrates NSM + Template + Qualia into one structure
#[derive(Clone, Debug)]
pub struct GrammarTriangle {
    /// NSM primitive activations (65 dimensions)
    pub nsm: NsmActivation,
    
    /// Matched template candidates (top 3)
    pub templates: TemplateCandidates,
    
    /// Speech act classification
    pub speech_act: SpeechActClassification,
    
    /// Qualia dimensions (18D felt-sense)
    pub qualia: QualiaExtraction,
    
    /// Causality flow
    pub causality: CausalityFlow,
    
    /// SPO extraction
    pub spo: SpoTriple,
    
    /// Source text
    pub source: String,
    
    /// Collapse state
    pub collapse_state: CollapseState,
}

impl GrammarTriangle {
    /// Parse text into grammar triangle (main entry point)
    pub fn parse(text: &str) -> Self {
        let tokens = tokenize(text);
        
        // Parallel extraction
        let nsm = NsmActivation::from_tokens(&tokens);
        let templates = TemplateCandidates::match_templates(&tokens, &nsm);
        let speech_act = SpeechActClassification::classify(&tokens, &nsm);
        let qualia = QualiaExtraction::extract(&tokens, &nsm);
        let causality = CausalityFlow::analyze(&tokens);
        let spo = SpoTriple::extract(&tokens, &templates);
        
        // Compute collapse state
        let collapse_state = CollapseState::compute(&templates);
        
        Self {
            nsm,
            templates,
            speech_act,
            qualia,
            causality,
            spo,
            source: text.to_string(),
            collapse_state,
        }
    }
    
    /// Convert to addresses in bind space
    pub fn to_addresses(&self) -> Vec<Addr> {
        let mut addrs = Vec::new();
        
        // Top NSM activations → 0x0C:XX
        for (slot, weight) in self.nsm.top_activations(5) {
            if weight > 0.3 {
                addrs.push(Addr::new(PREFIX_NSM, slot));
            }
        }
        
        // Best template match → 0x0D:XX
        if let Some(template) = self.templates.best() {
            addrs.push(Addr::new(PREFIX_TEMPLATES, template.slot));
        }
        
        // Speech act → 0x0E:XX
        addrs.push(Addr::new(PREFIX_SPEECH_ACTS, self.speech_act.slot));
        
        addrs
    }
    
    /// Convert to 10K-bit fingerprint
    pub fn to_fingerprint(&self) -> Fingerprint {
        // Allocate fingerprint regions:
        // bits 0-3999:    NSM contribution (65 primes → ~60 bits each)
        // bits 4000-5999: Template contribution  
        // bits 6000-7999: Qualia contribution
        // bits 8000-9999: Causality contribution
        
        let nsm_fp = self.nsm.to_fingerprint();
        let template_fp = self.templates.to_fingerprint();
        let qualia_fp = self.qualia.to_fingerprint();
        let causality_fp = self.causality.to_fingerprint();
        
        // Bundle all contributions
        bundle(&[nsm_fp, template_fp, qualia_fp, causality_fp])
    }
}

// =============================================================================
// NSM ACTIVATION
// =============================================================================

/// Activation weights for NSM primitives
#[derive(Clone, Debug)]
pub struct NsmActivation {
    /// Weights for each of 65 NSM primitives
    weights: [f32; 65],
}

impl NsmActivation {
    pub fn from_tokens(tokens: &[Token]) -> Self {
        let mut weights = [0.0f32; 65];
        
        for token in tokens {
            // Match token against NSM keyword patterns
            for (slot, (keywords, base_weight)) in NSM_KEYWORDS.iter().enumerate() {
                if slot >= 65 { break; }
                
                let token_lower = token.text.to_lowercase();
                for &keyword in *keywords {
                    if token_lower == keyword || token_lower.contains(keyword) {
                        // Accumulate with soft saturation
                        weights[slot] = (weights[slot] + base_weight).min(1.0);
                    }
                }
            }
        }
        
        Self { weights }
    }
    
    /// Get top N activated primitives with their slots
    pub fn top_activations(&self, n: usize) -> Vec<(u8, f32)> {
        let mut sorted: Vec<_> = self.weights.iter()
            .enumerate()
            .map(|(i, &w)| (i as u8, w))
            .filter(|(_, w)| *w > 0.0)
            .collect();
        
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(n);
        sorted
    }
    
    /// Get weight for specific primitive
    pub fn weight(&self, slot: u8) -> f32 {
        self.weights.get(slot as usize).copied().unwrap_or(0.0)
    }
    
    /// Convert to fingerprint contribution (bits 0-3999)
    pub fn to_fingerprint(&self) -> Fingerprint {
        let mut fp = Fingerprint::zero();
        
        for (i, &weight) in self.weights.iter().enumerate() {
            if weight > 0.3 {
                // Each primitive gets ~60 bits spread across its region
                let base = i * 60;
                let num_bits = (weight * 60.0) as usize;
                
                // Deterministic bit pattern based on primitive index
                let seed = (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
                for j in 0..num_bits.min(60) {
                    let bit_pos = base + ((seed.wrapping_mul((j + 1) as u64) % 60) as usize);
                    if bit_pos < 4000 {
                        fp.set_bit(bit_pos, true);
                    }
                }
            }
        }
        
        fp
    }
}

// NSM keyword mappings (slot → (keywords, base_weight))
static NSM_KEYWORDS: &[(&[&str], f32)] = &[
    // Substantives
    (&["i", "me", "my", "myself", "mine"], 0.9),           // I
    (&["you", "your", "yourself", "yours"], 0.9),          // YOU
    (&["someone", "person", "one", "who", "individual"], 0.8), // SOMEONE
    (&["something", "thing", "it", "what", "object"], 0.8),    // SOMETHING
    (&["people", "they", "them", "everyone", "folks"], 0.8),   // PEOPLE
    (&["body", "physical", "flesh", "corporal"], 0.8),         // BODY
    
    // Determiners
    (&["this", "these", "here"], 0.8),                     // THIS
    (&["same", "identical", "equal"], 0.8),                // THE_SAME
    (&["other", "another", "else", "different"], 0.8),     // OTHER
    
    // Quantifiers
    (&["one", "single", "a", "an"], 0.7),                  // ONE
    (&["two", "both", "pair", "couple"], 0.8),             // TWO
    (&["some", "few", "several"], 0.7),                    // SOME
    (&["all", "every", "each", "entire", "whole"], 0.8),   // ALL
    (&["much", "lot", "plenty"], 0.7),                     // MUCH
    (&["many", "numerous", "multiple"], 0.7),              // MANY
    
    // Evaluators
    (&["good", "great", "wonderful", "excellent"], 0.8),   // GOOD
    (&["bad", "wrong", "terrible", "awful", "evil"], 0.8), // BAD
    
    // Descriptors
    (&["big", "large", "huge", "enormous"], 0.8),          // BIG
    (&["small", "little", "tiny", "minute"], 0.8),         // SMALL
    
    // Mental predicates
    (&["think", "consider", "suppose", "ponder", "believe"], 0.9), // THINK
    (&["know", "understand", "realize", "aware", "comprehend"], 0.9), // KNOW
    (&["want", "desire", "wish", "need", "yearn", "crave"], 0.9),     // WANT
    (&["feel", "emotion", "sense", "experience"], 0.9),               // FEEL
    (&["see", "look", "watch", "observe", "view"], 0.8),              // SEE
    (&["hear", "listen", "sound"], 0.8),                              // HEAR
    
    // Speech
    (&["say", "tell", "speak", "mention", "state"], 0.8),  // SAY
    (&["words", "language", "speech", "verbal"], 0.8),     // WORDS
    (&["true", "truth", "real", "actual", "fact"], 0.8),   // TRUE
    
    // Actions
    (&["do", "make", "create", "perform", "act"], 0.8),    // DO
    (&["happen", "occur", "event", "transpire"], 0.8),     // HAPPEN
    (&["move", "motion", "go", "travel"], 0.8),            // MOVE
    (&["touch", "contact", "feel", "handle"], 0.8),        // TOUCH
    
    // Existence
    (&["exist", "presence", "being", "there is"], 0.8),    // THERE_IS
    (&["have", "possess", "own", "hold", "contain"], 0.9), // HAVE
    
    // Life/Death
    (&["live", "alive", "life", "living"], 0.8),           // LIVE
    (&["die", "death", "dead", "dying", "perish"], 0.8),   // DIE
    
    // Time
    (&["when", "time", "moment"], 0.7),                    // WHEN
    (&["now", "present", "current", "today"], 0.8),        // NOW
    (&["before", "past", "once", "ago", "earlier"], 0.8),  // BEFORE
    (&["after", "then", "future", "next", "later"], 0.8),  // AFTER
    (&["long time", "ages", "forever", "extended"], 0.7),  // A_LONG_TIME
    (&["short time", "brief", "instant", "quick"], 0.7),   // A_SHORT_TIME
    (&["for some time", "while", "period"], 0.7),          // FOR_SOME_TIME
    (&["moment", "instant", "second", "flash"], 0.7),      // MOMENT
    
    // Space
    (&["where", "place", "location"], 0.7),                // WHERE
    (&["here", "this place"], 0.8),                        // HERE
    (&["above", "over", "up", "higher", "top"], 0.7),      // ABOVE
    (&["below", "under", "down", "lower", "beneath"], 0.7), // BELOW
    (&["far", "distant", "remote", "away"], 0.7),          // FAR
    (&["near", "close", "nearby"], 0.7),                   // NEAR
    (&["side", "beside", "next to"], 0.7),                 // SIDE
    (&["inside", "within", "interior", "inner"], 0.7),     // INSIDE
    
    // Logical
    (&["not", "no", "never", "none", "without"], 0.9),     // NOT
    (&["maybe", "perhaps", "possibly", "might"], 0.8),     // MAYBE
    (&["can", "able", "capable", "possible"], 0.8),        // CAN
    (&["because", "since", "reason", "cause"], 0.9),       // BECAUSE
    (&["if", "whether", "condition", "suppose"], 0.8),     // IF
    
    // Intensifier
    (&["very", "extremely", "highly", "really", "quite"], 0.7), // VERY
    
    // Similarity
    (&["like", "similar", "as", "resemble"], 0.7),         // LIKE
    
    // Augmentatives
    (&["more", "additional", "further"], 0.7),             // MORE
    (&["part", "portion", "piece", "section"], 0.7),       // PART
    (&["kind", "type", "sort", "variety"], 0.7),           // KIND
    (&["word", "term", "expression"], 0.7),                // WORD
    (&["thing", "object", "item"], 0.7),                   // THING
    (&["way", "manner", "method", "fashion"], 0.7),        // WAY
];

// =============================================================================
// TEMPLATE CANDIDATES
// =============================================================================

/// Top 3 template matches (Triangle superposition)
#[derive(Clone, Debug)]
pub struct TemplateCandidates {
    /// Candidates sorted by confidence
    candidates: Vec<TemplateMatch>,
}

#[derive(Clone, Debug)]
pub struct TemplateMatch {
    pub slot: u8,
    pub name: &'static str,
    pub confidence: f32,
    pub matched_roles: Vec<GrammarRole>,
}

#[derive(Clone, Debug)]
pub enum GrammarRole {
    Subject,
    Verb,
    Object,
    IndirectObject,
    Complement,
    Adjunct,
    Modal,
    Auxiliary,
}

impl TemplateCandidates {
    pub fn match_templates(tokens: &[Token], nsm: &NsmActivation) -> Self {
        let mut candidates = Vec::new();
        
        // Score each template based on token patterns and NSM activations
        for &(slot, name, pattern, nsm_hints) in TEMPLATE_PATTERNS.iter() {
            let score = score_template(tokens, nsm, pattern, nsm_hints);
            if score > 0.2 {
                let matched_roles = extract_roles(tokens, pattern);
                candidates.push(TemplateMatch {
                    slot,
                    name,
                    confidence: score,
                    matched_roles,
                });
            }
        }
        
        // Sort by confidence and keep top 3
        candidates.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(3);
        
        Self { candidates }
    }
    
    /// Get best match
    pub fn best(&self) -> Option<&TemplateMatch> {
        self.candidates.first()
    }
    
    /// Get dispersion (SD of confidence scores)
    pub fn dispersion(&self) -> f32 {
        if self.candidates.len() < 2 {
            return 0.0;
        }
        
        let scores: Vec<f32> = self.candidates.iter().map(|c| c.confidence).collect();
        let mean = scores.iter().sum::<f32>() / scores.len() as f32;
        let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / scores.len() as f32;
        variance.sqrt()
    }
    
    /// Is the triangle homogeneous (all same family)?
    pub fn is_homogeneous(&self) -> bool {
        if self.candidates.len() < 2 {
            return true;
        }
        
        // Check if all templates are in the same family (same high nibble)
        let family = self.candidates[0].slot >> 4;
        self.candidates.iter().all(|c| (c.slot >> 4) == family)
    }
    
    /// Convert to fingerprint contribution (bits 4000-5999)
    pub fn to_fingerprint(&self) -> Fingerprint {
        let mut fp = Fingerprint::zero();
        
        for (i, candidate) in self.candidates.iter().enumerate() {
            let weight = candidate.confidence * (1.0 - i as f32 * 0.2);
            let base = 4000 + (candidate.slot as usize) * 8;
            let num_bits = (weight * 8.0) as usize;
            
            for j in 0..num_bits.min(8) {
                if base + j < 6000 {
                    fp.set_bit(base + j, true);
                }
            }
        }
        
        fp
    }
}

// Template patterns: (slot, name, token_pattern, nsm_hints)
static TEMPLATE_PATTERNS: &[(u8, &str, &str, &[u8])] = &[
    // Core clauses
    (template_slots::TRANSITIVE_DECLARATIVE, "transitive.declarative", "NP VP NP", &[nsm_slots::DO]),
    (template_slots::INTRANSITIVE_DECLARATIVE, "intransitive.declarative", "NP VP", &[nsm_slots::DO]),
    (template_slots::COPULAR_STATE, "copular.state", "NP be ADJ", &[nsm_slots::THERE_IS]),
    (template_slots::EXISTENTIAL, "existential", "there be NP", &[nsm_slots::THERE_IS]),
    (template_slots::POSSESSION, "possession", "NP have NP", &[nsm_slots::HAVE]),
    (template_slots::DITRANSITIVE, "ditransitive", "NP VP NP NP", &[nsm_slots::DO]),
    
    // Mental state
    (template_slots::MENTAL_STATE, "mental.state", "NP think/feel VP", &[nsm_slots::THINK, nsm_slots::FEEL]),
    (template_slots::BELIEF_REPORT, "belief.report", "NP believe/think that", &[nsm_slots::THINK]),
    (template_slots::DESIRE_EXPRESSION, "desire.expression", "NP want/desire VP", &[nsm_slots::WANT]),
    (template_slots::KNOWLEDGE_CLAIM, "knowledge.claim", "NP know that", &[nsm_slots::KNOW]),
    
    // Questions
    (template_slots::WH_QUESTION, "wh.question", "WH VP NP", &[]),
    (template_slots::YESNO_QUESTION, "yesno.question", "AUX NP VP", &[]),
    
    // Directives
    (template_slots::IMPERATIVE_COMMAND, "imperative.command", "VP NP", &[nsm_slots::WANT]),
    (template_slots::POLITE_REQUEST, "polite.request", "could/would you VP", &[nsm_slots::WANT, nsm_slots::CAN]),
    
    // Modals
    (template_slots::MODAL_ABILITY, "modal.ability", "NP can/able VP", &[nsm_slots::CAN]),
    (template_slots::MODAL_OBLIGATION, "modal.obligation", "NP must/should VP", &[]),
    
    // Complex
    (template_slots::CONDITIONAL, "conditional", "if CLAUSE then CLAUSE", &[nsm_slots::IF]),
    (template_slots::RELATIVE_CLAUSE, "relative.clause", "NP who/which VP", &[nsm_slots::SOMEONE, nsm_slots::SOMETHING]),
    
    // German-specific
    (template_slots::DE_TECAMOLO, "de.tecamolo", "temporal causal modal local", &[]),
    (template_slots::DE_VERB_SECOND, "de.verb_second", "X V2 ...", &[]),
];

fn score_template(tokens: &[Token], nsm: &NsmActivation, _pattern: &str, nsm_hints: &[u8]) -> f32 {
    let mut score = 0.0;
    
    // Check NSM hint activations
    for &hint in nsm_hints {
        score += nsm.weight(hint) * 0.3;
    }
    
    // Check token pattern (simplified)
    let has_subject = tokens.iter().any(|t| t.pos == POS::Noun || t.pos == POS::Pronoun);
    let has_verb = tokens.iter().any(|t| t.pos == POS::Verb);
    
    if has_subject { score += 0.2; }
    if has_verb { score += 0.2; }
    
    score.min(1.0)
}

fn extract_roles(_tokens: &[Token], _pattern: &str) -> Vec<GrammarRole> {
    // Simplified role extraction
    vec![GrammarRole::Subject, GrammarRole::Verb]
}

// =============================================================================
// SPEECH ACT CLASSIFICATION
// =============================================================================

#[derive(Clone, Debug)]
pub struct SpeechActClassification {
    pub slot: u8,
    pub name: &'static str,
    pub commitment_type: CommitmentType,
    pub confidence: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommitmentType {
    Belief,     // THINK
    Desire,     // WANT
    Intention,  // Commissive
    Feeling,    // FEEL (retractable)
    WorldChange,
    Meta,
}

impl SpeechActClassification {
    pub fn classify(tokens: &[Token], nsm: &NsmActivation) -> Self {
        // Determine speech act from NSM activations and token patterns
        
        // Check for questions
        if tokens.iter().any(|t| t.text.ends_with('?')) || 
           tokens.iter().any(|t| ["what", "who", "where", "when", "why", "how"].contains(&t.text.to_lowercase().as_str())) {
            return Self {
                slot: speech_act_slots::REQUEST,
                name: "REQUEST",
                commitment_type: CommitmentType::Desire,
                confidence: 0.8,
            };
        }
        
        // Check for directives (imperatives)
        if tokens.first().map(|t| t.pos == POS::Verb).unwrap_or(false) {
            return Self {
                slot: speech_act_slots::COMMAND,
                name: "COMMAND",
                commitment_type: CommitmentType::Desire,
                confidence: 0.7,
            };
        }
        
        // Check for expressives (thank, sorry, etc.)
        if tokens.iter().any(|t| ["thank", "thanks", "sorry", "apologize"].contains(&t.text.to_lowercase().as_str())) {
            return Self {
                slot: if tokens.iter().any(|t| t.text.to_lowercase().contains("thank")) {
                    speech_act_slots::THANK
                } else {
                    speech_act_slots::APOLOGIZE
                },
                name: "EXPRESSIVE",
                commitment_type: CommitmentType::Feeling,
                confidence: 0.9,
            };
        }
        
        // Check for commissives (promise, will, etc.)
        if tokens.iter().any(|t| ["promise", "will", "commit", "swear"].contains(&t.text.to_lowercase().as_str())) {
            return Self {
                slot: speech_act_slots::PROMISE,
                name: "PROMISE",
                commitment_type: CommitmentType::Intention,
                confidence: 0.8,
            };
        }
        
        // Default: assertion
        Self {
            slot: speech_act_slots::ASSERT,
            name: "ASSERT",
            commitment_type: CommitmentType::Belief,
            confidence: 0.6 + nsm.weight(nsm_slots::THINK) * 0.2,
        }
    }
}

// =============================================================================
// QUALIA EXTRACTION (18D → simplified 8D)
// =============================================================================

#[derive(Clone, Debug, Default)]
pub struct QualiaExtraction {
    pub arousal: f32,      // Calm ↔ Excited
    pub valence: f32,      // Negative ↔ Positive
    pub tension: f32,      // Relaxed ↔ Tense
    pub depth: f32,        // Surface ↔ Profound
    pub certainty: f32,    // Doubtful ↔ Certain
    pub intimacy: f32,     // Distant ↔ Intimate
    pub urgency: f32,      // Relaxed ↔ Urgent
    pub novelty: f32,      // Familiar ↔ Novel
}

impl QualiaExtraction {
    pub fn extract(tokens: &[Token], nsm: &NsmActivation) -> Self {
        let mut q = Self::default();
        
        // Valence from GOOD/BAD
        q.valence = 0.5 + (nsm.weight(nsm_slots::GOOD) - nsm.weight(nsm_slots::BAD)) * 0.4;
        
        // Certainty from mental predicates
        q.certainty = 0.3 + nsm.weight(nsm_slots::KNOW) * 0.4 - nsm.weight(nsm_slots::MAYBE) * 0.3;
        
        // Urgency from temporal markers
        q.urgency = 0.3 + nsm.weight(nsm_slots::NOW) * 0.3;
        
        // Depth from complexity (token count)
        q.depth = (tokens.len() as f32 / 20.0).min(1.0);
        
        // Arousal from punctuation
        if tokens.iter().any(|t| t.text.contains('!')) {
            q.arousal = 0.8;
        } else {
            q.arousal = 0.4;
        }
        
        // Intimacy from pronouns
        if nsm.weight(nsm_slots::I) > 0.5 || nsm.weight(nsm_slots::YOU) > 0.5 {
            q.intimacy = 0.7;
        } else {
            q.intimacy = 0.3;
        }
        
        q.tension = 0.5;
        q.novelty = 0.5;
        
        q
    }
    
    /// Convert to fingerprint contribution (bits 6000-7999)
    pub fn to_fingerprint(&self) -> Fingerprint {
        let mut fp = Fingerprint::zero();
        
        let dims = [
            self.arousal, self.valence, self.tension, self.depth,
            self.certainty, self.intimacy, self.urgency, self.novelty,
        ];
        
        for (i, &val) in dims.iter().enumerate() {
            let base = 6000 + i * 250;  // 250 bits per dimension
            let num_bits = (val * 250.0) as usize;
            
            for j in 0..num_bits.min(250) {
                if base + j < 8000 {
                    fp.set_bit(base + j, true);
                }
            }
        }
        
        fp
    }
}

// =============================================================================
// CAUSALITY FLOW
// =============================================================================

#[derive(Clone, Debug)]
pub struct CausalityFlow {
    pub agency: Agency,
    pub temporality: Temporality,
    pub dependency: DependencyType,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Agency {
    Active,
    Passive,
    Middle,
    Stative,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Temporality {
    Past,
    Present,
    Future,
    Habitual,
    Perfective,
    Imperfective,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DependencyType {
    Causes,
    Enables,
    Prevents,
    Allows,
    Triggers,
    None,
}

impl CausalityFlow {
    pub fn analyze(tokens: &[Token]) -> Self {
        // Determine temporality from verb tense markers
        let temporality = if tokens.iter().any(|t| ["was", "were", "did", "had"].contains(&t.text.to_lowercase().as_str())) {
            Temporality::Past
        } else if tokens.iter().any(|t| ["will", "shall", "going to"].contains(&t.text.to_lowercase().as_str())) {
            Temporality::Future
        } else {
            Temporality::Present
        };
        
        // Determine agency
        let agency = if tokens.iter().any(|t| ["by", "was", "were", "been"].contains(&t.text.to_lowercase().as_str())) {
            Agency::Passive
        } else if tokens.iter().any(|t| t.pos == POS::Verb) {
            Agency::Active
        } else {
            Agency::Stative
        };
        
        // Determine dependency from causal markers
        let dependency = if tokens.iter().any(|t| ["because", "since", "therefore", "so"].contains(&t.text.to_lowercase().as_str())) {
            DependencyType::Causes
        } else if tokens.iter().any(|t| ["if", "when", "unless"].contains(&t.text.to_lowercase().as_str())) {
            DependencyType::Enables
        } else {
            DependencyType::None
        };
        
        Self { agency, temporality, dependency }
    }
    
    /// Convert to fingerprint contribution (bits 8000-9999)
    pub fn to_fingerprint(&self) -> Fingerprint {
        let mut fp = Fingerprint::zero();
        
        // Agency (bits 8000-8499)
        let agency_base = 8000 + (self.agency as usize) * 125;
        for i in 0..100 {
            fp.set_bit(agency_base + i, true);
        }
        
        // Temporality (bits 8500-8999)
        let temp_base = 8500 + (self.temporality as usize) * 83;
        for i in 0..70 {
            fp.set_bit(temp_base + i, true);
        }
        
        // Dependency (bits 9000-9499)
        let dep_base = 9000 + (self.dependency as usize) * 83;
        for i in 0..70 {
            fp.set_bit(dep_base + i, true);
        }
        
        fp
    }
}

// =============================================================================
// SPO TRIPLE EXTRACTION
// =============================================================================

#[derive(Clone, Debug, Default)]
pub struct SpoTriple {
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub object: Option<String>,
}

impl SpoTriple {
    pub fn extract(tokens: &[Token], templates: &TemplateCandidates) -> Self {
        let mut spo = Self::default();
        
        // Find first noun phrase as subject
        for token in tokens {
            if spo.subject.is_none() && (token.pos == POS::Noun || token.pos == POS::Pronoun) {
                spo.subject = Some(token.text.clone());
            } else if spo.predicate.is_none() && token.pos == POS::Verb {
                spo.predicate = Some(token.text.clone());
            } else if spo.object.is_none() && spo.predicate.is_some() && 
                      (token.pos == POS::Noun || token.pos == POS::Pronoun) {
                spo.object = Some(token.text.clone());
            }
        }
        
        // Refine based on template
        if let Some(template) = templates.best() {
            // Template-specific refinements could go here
            let _ = template;
        }
        
        spo
    }
}

// =============================================================================
// COLLAPSE STATE
// =============================================================================

#[derive(Clone, Debug)]
pub struct CollapseState {
    pub gate: GateState,
    pub sd: f32,
    pub can_collapse: bool,
    pub action: CollapseAction,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateState {
    Flow,   // SD < 0.15: Commit
    Hold,   // 0.15 ≤ SD ≤ 0.35: Superposition
    Block,  // SD > 0.35: Need clarification
}

#[derive(Clone, Debug)]
pub enum CollapseAction {
    CommitToNode { template_slot: u8 },
    HoldInCrystal { ttl_seconds: u64 },
    CheckCalibration { missing_info: Vec<String> },
}

impl CollapseState {
    pub fn compute(templates: &TemplateCandidates) -> Self {
        let sd = templates.dispersion();
        let is_homogeneous = templates.is_homogeneous();
        
        let (gate, can_collapse, action) = if sd < 0.15 && is_homogeneous {
            let slot = templates.best().map(|t| t.slot).unwrap_or(0);
            (GateState::Flow, true, CollapseAction::CommitToNode { template_slot: slot })
        } else if sd <= 0.35 {
            (GateState::Hold, false, CollapseAction::HoldInCrystal { ttl_seconds: 60 })
        } else {
            let missing = vec!["role clarification".to_string()];
            (GateState::Block, false, CollapseAction::CheckCalibration { missing_info: missing })
        };
        
        Self { gate, sd, can_collapse, action }
    }
}

// =============================================================================
// TOKENIZER (Simplified)
// =============================================================================

#[derive(Clone, Debug)]
pub struct Token {
    pub text: String,
    pub pos: POS,
    pub index: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum POS {
    Noun,
    Verb,
    Adjective,
    Adverb,
    Pronoun,
    Determiner,
    Preposition,
    Conjunction,
    Interjection,
    Punctuation,
    Unknown,
}

fn tokenize(text: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    
    for (i, word) in text.split_whitespace().enumerate() {
        let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
        let pos = classify_pos(clean);
        
        tokens.push(Token {
            text: clean.to_string(),
            pos,
            index: i,
        });
        
        // Handle trailing punctuation
        if let Some(punct) = word.chars().last() {
            if !punct.is_alphanumeric() {
                tokens.push(Token {
                    text: punct.to_string(),
                    pos: POS::Punctuation,
                    index: i,
                });
            }
        }
    }
    
    tokens
}

fn classify_pos(word: &str) -> POS {
    let lower = word.to_lowercase();
    
    // Pronouns
    if ["i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
        "my", "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs",
        "myself", "yourself", "himself", "herself", "itself", "ourselves", "themselves",
        "who", "whom", "whose", "which", "what", "this", "that", "these", "those"].contains(&lower.as_str()) {
        return POS::Pronoun;
    }
    
    // Determiners
    if ["a", "an", "the", "some", "any", "no", "every", "each", "all", "both", "few", "many", "much",
        "several", "other", "another"].contains(&lower.as_str()) {
        return POS::Determiner;
    }
    
    // Common verbs
    if ["is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
        "do", "does", "did", "doing", "done", "will", "would", "shall", "should", "may", "might",
        "can", "could", "must", "go", "goes", "went", "gone", "going", "get", "gets", "got",
        "know", "knows", "knew", "known", "knowing", "think", "thinks", "thought", "thinking",
        "want", "wants", "wanted", "wanting", "feel", "feels", "felt", "feeling",
        "see", "sees", "saw", "seen", "seeing", "hear", "hears", "heard", "hearing",
        "say", "says", "said", "saying", "make", "makes", "made", "making",
        "take", "takes", "took", "taken", "taking", "come", "comes", "came", "coming",
        "give", "gives", "gave", "given", "giving", "find", "finds", "found", "finding",
        "tell", "tells", "told", "telling", "ask", "asks", "asked", "asking",
        "use", "uses", "used", "using", "put", "puts", "putting", "try", "tries", "tried", "trying",
        "leave", "leaves", "left", "leaving", "call", "calls", "called", "calling",
        "need", "needs", "needed", "needing", "keep", "keeps", "kept", "keeping",
        "let", "lets", "letting", "begin", "begins", "began", "begun", "beginning",
        "seem", "seems", "seemed", "seeming", "help", "helps", "helped", "helping",
        "show", "shows", "showed", "shown", "showing", "believe", "believes", "believed",
        "understand", "understands", "understood", "understanding", "love", "loves", "loved"].contains(&lower.as_str()) {
        return POS::Verb;
    }
    
    // Prepositions
    if ["in", "on", "at", "to", "for", "with", "by", "from", "of", "about", "into", "through",
        "during", "before", "after", "above", "below", "between", "under", "over", "against",
        "among", "around", "behind", "beside", "beyond", "within", "without", "upon", "toward"].contains(&lower.as_str()) {
        return POS::Preposition;
    }
    
    // Conjunctions
    if ["and", "or", "but", "nor", "for", "yet", "so", "because", "although", "though",
        "while", "if", "unless", "until", "when", "where", "whether", "that", "than"].contains(&lower.as_str()) {
        return POS::Conjunction;
    }
    
    // Common adjectives
    if ["good", "bad", "big", "small", "large", "little", "great", "new", "old", "young",
        "long", "short", "high", "low", "right", "wrong", "true", "false", "beautiful",
        "important", "different", "same", "other", "first", "last", "next", "best", "worst"].contains(&lower.as_str()) {
        return POS::Adjective;
    }
    
    // Common adverbs
    if ["very", "really", "also", "just", "still", "only", "even", "now", "then", "here", "there",
        "always", "never", "often", "sometimes", "soon", "already", "almost", "enough", "quite",
        "too", "well", "much", "more", "most", "less", "least", "ever", "ago"].contains(&lower.as_str()) {
        return POS::Adverb;
    }
    
    // Default: assume noun (most open-class words are nouns)
    POS::Noun
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

fn bundle(fps: &[Fingerprint]) -> Fingerprint {
    if fps.is_empty() {
        return Fingerprint::zero();
    }
    
    // Majority voting for bundle
    let mut result = Fingerprint::zero();
    let threshold = fps.len() / 2;
    
    for bit in 0..10000 {
        let count: usize = fps.iter()
            .filter(|fp| fp.get_bit(bit))
            .count();
        
        if count > threshold {
            result.set_bit(bit, true);
        }
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_simple() {
        let triangle = GrammarTriangle::parse("I want to know what you think about this");
        
        // Check NSM activations
        assert!(triangle.nsm.weight(nsm_slots::I) > 0.0);
        assert!(triangle.nsm.weight(nsm_slots::WANT) > 0.0);
        assert!(triangle.nsm.weight(nsm_slots::KNOW) > 0.0);
        assert!(triangle.nsm.weight(nsm_slots::YOU) > 0.0);
        assert!(triangle.nsm.weight(nsm_slots::THINK) > 0.0);
    }
    
    #[test]
    fn test_collapse_state() {
        let triangle = GrammarTriangle::parse("The cat sat on the mat");
        
        // Should have clear template match (transitive or intransitive)
        assert!(triangle.templates.best().is_some());
        
        // Check gate state
        println!("Gate: {:?}, SD: {}", triangle.collapse_state.gate, triangle.collapse_state.sd);
    }
    
    #[test]
    fn test_speech_act() {
        let question = GrammarTriangle::parse("What do you want?");
        assert_eq!(question.speech_act.commitment_type, CommitmentType::Desire);
        
        let statement = GrammarTriangle::parse("I know the answer");
        assert_eq!(statement.speech_act.commitment_type, CommitmentType::Belief);
    }
}
