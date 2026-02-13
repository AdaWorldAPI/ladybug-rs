//! Cognitive Graph Substrate
//!
//! Nodes, Edges, and Verbs - all as fingerprints.
//! The graph IS the cognitive architecture.
//!
//! Node = Fingerprint (concept, entity, state)
//! Edge = Fingerprint ⊗ Verb ⊗ Fingerprint (relationship)
//! Verb = Fingerprint (operation, relation type)
//!
//! 144 verbs at Go board intersections (Sigma topology)

use crate::core::Fingerprint;
use std::collections::HashMap;

// =============================================================================
// THE 144 VERBS (Go Board Topology)
// =============================================================================

/// Verb categories - each gets a region on the 12×12 Go board
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VerbCategory {
    // Structural (row 0-1, 24 verbs)
    Structural = 0,

    // Causal (row 2-3, 24 verbs)
    Causal = 1,

    // Temporal (row 4-5, 24 verbs)
    Temporal = 2,

    // Epistemic (row 6-7, 24 verbs)
    Epistemic = 3,

    // Agentive (row 8-9, 24 verbs)
    Agentive = 4,

    // Experiential (row 10-11, 24 verbs)
    Experiential = 5,
}

/// The 144 core verbs
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Verb {
    // =========================================================================
    // STRUCTURAL VERBS (0-23) - How things relate structurally
    // =========================================================================
    IsA = 0,           // Inheritance: cat IS_A animal
    HasA = 1,          // Composition: car HAS_A engine
    PartOf = 2,        // Mereology: wheel PART_OF car
    InstanceOf = 3,    // Instantiation: fluffy INSTANCE_OF cat
    KindOf = 4,        // Taxonomy: tabby KIND_OF cat
    SameAs = 5,        // Identity: morning_star SAME_AS evening_star
    SimilarTo = 6,     // Similarity: cat SIMILAR_TO tiger
    DifferentFrom = 7, // Distinction: cat DIFFERENT_FROM dog
    OppositeOf = 8,    // Antonymy: hot OPPOSITE_OF cold
    Contains = 9,      // Containment: box CONTAINS items
    Overlaps = 10,     // Overlap: set_a OVERLAPS set_b
    Disjoint = 11,     // Disjunction: mammals DISJOINT reptiles
    SubsetOf = 12,     // Subset: cats SUBSET_OF mammals
    SupersetOf = 13,   // Superset: mammals SUPERSET_OF cats
    AdjacentTo = 14,   // Adjacency: room_a ADJACENT_TO room_b
    ConnectedTo = 15,  // Connection: node_a CONNECTED_TO node_b
    DerivedFrom = 16,  // Derivation: theorem DERIVED_FROM axioms
    ComposedOf = 17,   // Composition: water COMPOSED_OF h2o
    AttributeOf = 18,  // Attribution: red ATTRIBUTE_OF apple
    ValueOf = 19,      // Value: 5 VALUE_OF count
    RoleOf = 20,       // Role: agent ROLE_OF actor
    ContextOf = 21,    // Context: meeting CONTEXT_OF discussion
    DomainOf = 22,     // Domain: medicine DOMAIN_OF diagnosis
    RangeOf = 23,      // Range: positive RANGE_OF sqrt

    // =========================================================================
    // CAUSAL VERBS (24-47) - How things cause/affect each other
    // =========================================================================
    Causes = 24,      // Direct causation: fire CAUSES heat
    CausedBy = 25,    // Inverse: heat CAUSED_BY fire
    Enables = 26,     // Enablement: key ENABLES unlock
    Prevents = 27,    // Prevention: vaccine PREVENTS disease
    Maintains = 28,   // Maintenance: thermostat MAINTAINS temperature
    Triggers = 29,    // Triggering: spark TRIGGERS explosion
    Terminates = 30,  // Termination: cure TERMINATES illness
    Modulates = 31,   // Modulation: diet MODULATES weight
    Amplifies = 32,   // Amplification: feedback AMPLIFIES signal
    Dampens = 33,     // Dampening: resistance DAMPENS current
    Correlates = 34,  // Correlation: ice_cream CORRELATES sunburn
    Influences = 35,  // Influence: weather INFLUENCES mood
    Determines = 36,  // Determination: genes DETERMINE traits
    Constrains = 37,  // Constraint: budget CONSTRAINS options
    Requires = 38,    // Requirement: fire REQUIRES oxygen
    Produces = 39,    // Production: factory PRODUCES goods
    Consumes = 40,    // Consumption: fire CONSUMES fuel
    Transforms = 41,  // Transformation: caterpillar TRANSFORMS butterfly
    Initiates = 42,   // Initiation: signal INITIATES process
    Sustains = 43,    // Sustenance: food SUSTAINS life
    Accelerates = 44, // Acceleration: catalyst ACCELERATES reaction
    Decelerates = 45, // Deceleration: friction DECELERATES motion
    Blocks = 46,      // Blocking: wall BLOCKS path
    Facilitates = 47, // Facilitation: tool FACILITATES task

    // =========================================================================
    // TEMPORAL VERBS (48-71) - How things relate in time
    // =========================================================================
    Before = 48,           // Allen: a BEFORE b
    After = 49,            // Allen: a AFTER b
    Meets = 50,            // Allen: a MEETS b (end touches start)
    MetBy = 51,            // Allen: a MET_BY b
    TemporalOverlaps = 52, // Allen: a OVERLAPS b (partial)
    OverlappedBy = 53,     // Allen: a OVERLAPPED_BY b
    During = 54,           // Allen: a DURING b (contained)
    TemporalContains = 55, // Allen: a CONTAINS b
    Starts = 56,           // Allen: a STARTS b (same start)
    StartedBy = 57,        // Allen: a STARTED_BY b
    Finishes = 58,         // Allen: a FINISHES b (same end)
    FinishedBy = 59,       // Allen: a FINISHED_BY b
    Equals = 60,           // Allen: a EQUALS b (same interval)
    Precedes = 61,         // Precedence: a PRECEDES b
    Follows = 62,          // Following: a FOLLOWS b
    Concurrent = 63,       // Concurrency: a CONCURRENT b
    Interrupted = 64,      // Interruption: a INTERRUPTED b
    Resumed = 65,          // Resumption: a RESUMED after b
    Periodic = 66,         // Periodicity: a PERIODIC every b
    Eventual = 67,         // Eventuality: a EVENTUAL in b
    Immediate = 68,        // Immediacy: a IMMEDIATE after b
    Gradual = 69,          // Graduality: a GRADUAL over b
    Sudden = 70,           // Suddenness: a SUDDEN at b
    Persistent = 71,       // Persistence: a PERSISTENT through b

    // =========================================================================
    // EPISTEMIC VERBS (72-95) - How things relate to knowledge/belief
    // =========================================================================
    Knows = 72,          // Knowledge: agent KNOWS fact
    Believes = 73,       // Belief: agent BELIEVES proposition
    Thinks = 74,         // Thought: agent THINKS idea
    Doubts = 75,         // Doubt: agent DOUBTS claim
    Assumes = 76,        // Assumption: agent ASSUMES premise
    Infers = 77,         // Inference: agent INFERS conclusion
    Deduces = 78,        // Deduction: agent DEDUCES theorem
    Induces = 79,        // Induction: agent INDUCES pattern
    Abduces = 80,        // Abduction: agent ABDUCES explanation
    Remembers = 81,      // Memory: agent REMEMBERS event
    Forgets = 82,        // Forgetting: agent FORGETS detail
    Expects = 83,        // Expectation: agent EXPECTS outcome
    Surprises = 84,      // Surprise: event SURPRISES agent
    Confirms = 85,       // Confirmation: evidence CONFIRMS theory
    Refutes = 86,        // Refutation: evidence REFUTES theory
    Justifies = 87,      // Justification: reason JUSTIFIES belief
    Questions = 88,      // Questioning: agent QUESTIONS claim
    Understands = 89,    // Understanding: agent UNDERSTANDS concept
    Misunderstands = 90, // Misunderstanding: agent MISUNDERSTANDS meaning
    Clarifies = 91,      // Clarification: explanation CLARIFIES concept
    Confuses = 92,       // Confusion: ambiguity CONFUSES agent
    Learns = 93,         // Learning: agent LEARNS skill
    Teaches = 94,        // Teaching: agent TEACHES student
    Realizes = 95,       // Realization: agent REALIZES truth

    // =========================================================================
    // AGENTIVE VERBS (96-119) - How agents act and interact
    // =========================================================================
    Does = 96,          // Action: agent DOES action
    Makes = 97,         // Creation: agent MAKES artifact
    Uses = 98,          // Usage: agent USES tool
    Gives = 99,         // Giving: agent GIVES object to recipient
    Takes = 100,        // Taking: agent TAKES object
    Wants = 101,        // Desire: agent WANTS goal
    Needs = 102,        // Need: agent NEEDS resource
    Intends = 103,      // Intention: agent INTENDS action
    Plans = 104,        // Planning: agent PLANS sequence
    Decides = 105,      // Decision: agent DECIDES choice
    Chooses = 106,      // Choice: agent CHOOSES option
    Tries = 107,        // Attempt: agent TRIES task
    Succeeds = 108,     // Success: agent SUCCEEDS at goal
    Fails = 109,        // Failure: agent FAILS at task
    Helps = 110,        // Helping: agent HELPS other
    Hinders = 111,      // Hindering: agent HINDERS other
    Cooperates = 112,   // Cooperation: agent COOPERATES with other
    Competes = 113,     // Competition: agent COMPETES with other
    Communicates = 114, // Communication: agent COMMUNICATES message
    Promises = 115,     // Promise: agent PROMISES commitment
    Requests = 116,     // Request: agent REQUESTS action
    Commands = 117,     // Command: agent COMMANDS action
    Permits = 118,      // Permission: agent PERMITS action
    Forbids = 119,      // Prohibition: agent FORBIDS action

    // =========================================================================
    // EXPERIENTIAL VERBS (120-143) - How things are experienced/felt
    // =========================================================================
    Sees = 120,        // Vision: agent SEES object
    Hears = 121,       // Audition: agent HEARS sound
    Feels = 122,       // Touch/emotion: agent FEELS sensation
    Tastes = 123,      // Gustation: agent TASTES flavor
    Smells = 124,      // Olfaction: agent SMELLS odor
    Perceives = 125,   // Perception: agent PERCEIVES stimulus
    Attends = 126,     // Attention: agent ATTENDS to focus
    Ignores = 127,     // Ignoring: agent IGNORES distraction
    Enjoys = 128,      // Enjoyment: agent ENJOYS experience
    Dislikes = 129,    // Dislike: agent DISLIKES experience
    Fears = 130,       // Fear: agent FEARS threat
    Hopes = 131,       // Hope: agent HOPES for outcome
    Regrets = 132,     // Regret: agent REGRETS action
    Appreciates = 133, // Appreciation: agent APPRECIATES value
    Suffers = 134,     // Suffering: agent SUFFERS from pain
    Delights = 135,    // Delight: experience DELIGHTS agent
    Bores = 136,       // Boredom: experience BORES agent
    Excites = 137,     // Excitement: stimulus EXCITES agent
    Calms = 138,       // Calming: experience CALMS agent
    Angers = 139,      // Anger: event ANGERS agent
    Saddens = 140,     // Sadness: event SADDENS agent
    Satisfies = 141,   // Satisfaction: outcome SATISFIES agent
    Frustrates = 142,  // Frustration: obstacle FRUSTRATES agent
    Inspires = 143,    // Inspiration: idea INSPIRES agent
}

impl Verb {
    /// Get verb category
    pub fn category(&self) -> VerbCategory {
        let id = *self as u8;
        match id / 24 {
            0 => VerbCategory::Structural,
            1 => VerbCategory::Causal,
            2 => VerbCategory::Temporal,
            3 => VerbCategory::Epistemic,
            4 => VerbCategory::Agentive,
            _ => VerbCategory::Experiential,
        }
    }

    /// Get Go board position (12×12 grid)
    pub fn board_position(&self) -> (u8, u8) {
        let id = *self as u8;
        let row = (id / 12) as u8;
        let col = (id % 12) as u8;
        (row, col)
    }

    /// Get fingerprint for this verb
    pub fn fingerprint(&self) -> Fingerprint {
        Fingerprint::from_content(&format!("VERB::{:?}", self))
    }

    /// Get verb name
    pub fn name(&self) -> &'static str {
        match self {
            // Structural
            Verb::IsA => "IS_A",
            Verb::HasA => "HAS_A",
            Verb::PartOf => "PART_OF",
            Verb::InstanceOf => "INSTANCE_OF",
            Verb::KindOf => "KIND_OF",
            Verb::SameAs => "SAME_AS",
            Verb::SimilarTo => "SIMILAR_TO",
            Verb::DifferentFrom => "DIFFERENT_FROM",
            Verb::OppositeOf => "OPPOSITE_OF",
            Verb::Contains => "CONTAINS",
            Verb::Overlaps => "OVERLAPS",
            Verb::Disjoint => "DISJOINT",
            Verb::SubsetOf => "SUBSET_OF",
            Verb::SupersetOf => "SUPERSET_OF",
            Verb::AdjacentTo => "ADJACENT_TO",
            Verb::ConnectedTo => "CONNECTED_TO",
            Verb::DerivedFrom => "DERIVED_FROM",
            Verb::ComposedOf => "COMPOSED_OF",
            Verb::AttributeOf => "ATTRIBUTE_OF",
            Verb::ValueOf => "VALUE_OF",
            Verb::RoleOf => "ROLE_OF",
            Verb::ContextOf => "CONTEXT_OF",
            Verb::DomainOf => "DOMAIN_OF",
            Verb::RangeOf => "RANGE_OF",

            // Causal
            Verb::Causes => "CAUSES",
            Verb::CausedBy => "CAUSED_BY",
            Verb::Enables => "ENABLES",
            Verb::Prevents => "PREVENTS",
            Verb::Maintains => "MAINTAINS",
            Verb::Triggers => "TRIGGERS",
            Verb::Terminates => "TERMINATES",
            Verb::Modulates => "MODULATES",
            Verb::Amplifies => "AMPLIFIES",
            Verb::Dampens => "DAMPENS",
            Verb::Correlates => "CORRELATES",
            Verb::Influences => "INFLUENCES",
            Verb::Determines => "DETERMINES",
            Verb::Constrains => "CONSTRAINS",
            Verb::Requires => "REQUIRES",
            Verb::Produces => "PRODUCES",
            Verb::Consumes => "CONSUMES",
            Verb::Transforms => "TRANSFORMS",
            Verb::Initiates => "INITIATES",
            Verb::Sustains => "SUSTAINS",
            Verb::Accelerates => "ACCELERATES",
            Verb::Decelerates => "DECELERATES",
            Verb::Blocks => "BLOCKS",
            Verb::Facilitates => "FACILITATES",

            // Temporal (using pattern for brevity)
            Verb::Before => "BEFORE",
            Verb::After => "AFTER",
            Verb::Meets => "MEETS",
            Verb::MetBy => "MET_BY",
            Verb::During => "DURING",
            Verb::Starts => "STARTS",
            Verb::StartedBy => "STARTED_BY",
            Verb::Finishes => "FINISHES",
            Verb::FinishedBy => "FINISHED_BY",
            Verb::Equals => "EQUALS",
            Verb::Precedes => "PRECEDES",
            Verb::Follows => "FOLLOWS",
            Verb::Concurrent => "CONCURRENT",
            Verb::Interrupted => "INTERRUPTED",
            Verb::Resumed => "RESUMED",
            Verb::Periodic => "PERIODIC",
            Verb::Eventual => "EVENTUAL",
            Verb::Immediate => "IMMEDIATE",
            Verb::Gradual => "GRADUAL",
            Verb::Sudden => "SUDDEN",
            Verb::Persistent => "PERSISTENT",

            // Epistemic
            Verb::Knows => "KNOWS",
            Verb::Believes => "BELIEVES",
            Verb::Thinks => "THINKS",
            Verb::Doubts => "DOUBTS",
            Verb::Assumes => "ASSUMES",
            Verb::Infers => "INFERS",
            Verb::Deduces => "DEDUCES",
            Verb::Induces => "INDUCES",
            Verb::Abduces => "ABDUCES",
            Verb::Remembers => "REMEMBERS",
            Verb::Forgets => "FORGETS",
            Verb::Expects => "EXPECTS",
            Verb::Surprises => "SURPRISES",
            Verb::Confirms => "CONFIRMS",
            Verb::Refutes => "REFUTES",
            Verb::Justifies => "JUSTIFIES",
            Verb::Questions => "QUESTIONS",
            Verb::Understands => "UNDERSTANDS",
            Verb::Misunderstands => "MISUNDERSTANDS",
            Verb::Clarifies => "CLARIFIES",
            Verb::Confuses => "CONFUSES",
            Verb::Learns => "LEARNS",
            Verb::Teaches => "TEACHES",
            Verb::Realizes => "REALIZES",

            // Agentive
            Verb::Does => "DOES",
            Verb::Makes => "MAKES",
            Verb::Uses => "USES",
            Verb::Gives => "GIVES",
            Verb::Takes => "TAKES",
            Verb::Wants => "WANTS",
            Verb::Needs => "NEEDS",
            Verb::Intends => "INTENDS",
            Verb::Plans => "PLANS",
            Verb::Decides => "DECIDES",
            Verb::Chooses => "CHOOSES",
            Verb::Tries => "TRIES",
            Verb::Succeeds => "SUCCEEDS",
            Verb::Fails => "FAILS",
            Verb::Helps => "HELPS",
            Verb::Hinders => "HINDERS",
            Verb::Cooperates => "COOPERATES",
            Verb::Competes => "COMPETES",
            Verb::Communicates => "COMMUNICATES",
            Verb::Promises => "PROMISES",
            Verb::Requests => "REQUESTS",
            Verb::Commands => "COMMANDS",
            Verb::Permits => "PERMITS",
            Verb::Forbids => "FORBIDS",

            // Experiential
            Verb::Sees => "SEES",
            Verb::Hears => "HEARS",
            Verb::Feels => "FEELS",
            Verb::Tastes => "TASTES",
            Verb::Smells => "SMELLS",
            Verb::Perceives => "PERCEIVES",
            Verb::Attends => "ATTENDS",
            Verb::Ignores => "IGNORES",
            Verb::Enjoys => "ENJOYS",
            Verb::Dislikes => "DISLIKES",
            Verb::Fears => "FEARS",
            Verb::Hopes => "HOPES",
            Verb::Regrets => "REGRETS",
            Verb::Appreciates => "APPRECIATES",
            Verb::Suffers => "SUFFERS",
            Verb::Delights => "DELIGHTS",
            Verb::Bores => "BORES",
            Verb::Excites => "EXCITES",
            Verb::Calms => "CALMS",
            Verb::Angers => "ANGERS",
            Verb::Saddens => "SADDENS",
            Verb::Satisfies => "SATISFIES",
            Verb::Frustrates => "FRUSTRATES",
            Verb::Inspires => "INSPIRES",

            // Temporal duplicates (different enum variant) - Allen interval relations
            Verb::OverlappedBy => "OVERLAPPED_BY",
            Verb::TemporalOverlaps => "TEMPORAL_OVERLAPS",
            Verb::TemporalContains => "TEMPORAL_CONTAINS",
        }
    }

    /// Get inverse verb if exists
    pub fn inverse(&self) -> Option<Verb> {
        match self {
            Verb::IsA => Some(Verb::HasA), // approximate
            Verb::Causes => Some(Verb::CausedBy),
            Verb::CausedBy => Some(Verb::Causes),
            Verb::Before => Some(Verb::After),
            Verb::After => Some(Verb::Before),
            Verb::Contains => Some(Verb::PartOf),
            Verb::PartOf => Some(Verb::Contains),
            Verb::Gives => Some(Verb::Takes),
            Verb::Takes => Some(Verb::Gives),
            Verb::Helps => Some(Verb::Hinders),
            Verb::Hinders => Some(Verb::Helps),
            Verb::Enables => Some(Verb::Prevents),
            Verb::Prevents => Some(Verb::Enables),
            _ => None,
        }
    }
}

// =============================================================================
// COGNITIVE NODE
// =============================================================================

/// A node in the cognitive graph - everything is a fingerprint
#[derive(Clone, Debug)]
pub struct CogNode {
    /// The fingerprint IS the identity
    pub fingerprint: Fingerprint,

    /// Optional human-readable name
    pub name: Option<String>,

    /// Node type (NSM prime, concept, entity, etc)
    pub node_type: NodeType,

    /// Rung level (abstraction depth)
    pub rung: u8,

    /// Qualia signature (emotional coloring)
    pub qualia: [f32; 8],

    /// NARS truth value (frequency, confidence)
    pub truth: Option<(f32, f32)>,

    /// Activation level (for spreading activation)
    pub activation: f32,

    /// Creation timestamp
    pub created_at: u64,

    /// Last access timestamp
    pub accessed_at: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NodeType {
    // Semantic types
    NsmPrime, // One of 65 NSM primes
    Concept,  // Abstract concept
    Entity,   // Concrete entity
    Event,    // Temporal event
    State,    // State of affairs
    Action,   // Action/process
    Property, // Property/attribute
    Relation, // Relation type

    // Cognitive types
    Goal,       // ACT-R goal
    Chunk,      // ACT-R chunk
    Production, // ACT-R production rule
    Belief,     // NARS belief
    Query,      // NARS query
    Memory,     // Episodic memory

    // Meta types
    Verb,     // One of 144 verbs
    Role,     // Thematic role
    Template, // YAML template
    Crystal,  // Crystal projection
}

impl CogNode {
    pub fn new(fingerprint: Fingerprint, node_type: NodeType) -> Self {
        Self {
            fingerprint,
            name: None,
            node_type,
            rung: 3, // Default: proposition level
            qualia: [0.0; 8],
            truth: None,
            activation: 0.0,
            created_at: now_millis(),
            accessed_at: now_millis(),
        }
    }

    pub fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    pub fn with_rung(mut self, rung: u8) -> Self {
        self.rung = rung.min(9);
        self
    }

    pub fn with_qualia(mut self, qualia: [f32; 8]) -> Self {
        self.qualia = qualia;
        self
    }

    pub fn with_truth(mut self, frequency: f32, confidence: f32) -> Self {
        self.truth = Some((frequency, confidence));
        self
    }

    /// Compute similarity to another node
    pub fn similarity(&self, other: &CogNode) -> f32 {
        self.fingerprint.similarity(&other.fingerprint)
    }

    /// Update activation with decay
    pub fn update_activation(&mut self, boost: f32, decay: f32) {
        self.activation = (self.activation * decay) + boost;
        self.accessed_at = now_millis();
    }
}

// =============================================================================
// COGNITIVE EDGE
// =============================================================================

/// An edge in the cognitive graph
/// Encoded as: from ⊗ verb ⊗ to
#[derive(Clone, Debug)]
pub struct CogEdge {
    /// Source node fingerprint
    pub from: Fingerprint,

    /// Target node fingerprint
    pub to: Fingerprint,

    /// The verb (relation type)
    pub verb: Verb,

    /// Edge fingerprint: from ⊗ verb ⊗ to
    pub fingerprint: Fingerprint,

    /// Edge weight (strength)
    pub weight: f32,

    /// NARS truth value
    pub truth: Option<(f32, f32)>,

    /// Temporal bounds (if temporal relation)
    pub temporal: Option<(u64, u64)>,

    /// Creation timestamp
    pub created_at: u64,
}

impl CogEdge {
    pub fn new(from: Fingerprint, verb: Verb, to: Fingerprint) -> Self {
        let verb_fp = verb.fingerprint();

        // Edge fingerprint = from ⊗ verb ⊗ to
        let edge_fp = from.bind(&verb_fp).bind(&to);

        Self {
            from,
            to,
            verb,
            fingerprint: edge_fp,
            weight: 1.0,
            truth: None,
            temporal: None,
            created_at: now_millis(),
        }
    }

    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    pub fn with_truth(mut self, frequency: f32, confidence: f32) -> Self {
        self.truth = Some((frequency, confidence));
        self
    }

    pub fn with_temporal(mut self, start: u64, end: u64) -> Self {
        self.temporal = Some((start, end));
        self
    }

    /// Check if edge matches a pattern (any component can be wildcard/None)
    pub fn matches(
        &self,
        from: Option<&Fingerprint>,
        verb: Option<Verb>,
        to: Option<&Fingerprint>,
    ) -> bool {
        let from_match = from.map_or(true, |f| self.from.similarity(f) > 0.9);
        let verb_match = verb.map_or(true, |v| self.verb == v);
        let to_match = to.map_or(true, |t| self.to.similarity(t) > 0.9);

        from_match && verb_match && to_match
    }

    /// Recover 'to' from edge fingerprint and 'from'
    /// edge = from ⊗ verb ⊗ to
    /// to = edge ⊗ from ⊗ verb (XOR is self-inverse)
    pub fn recover_to(&self) -> Fingerprint {
        let verb_fp = self.verb.fingerprint();
        self.fingerprint.bind(&self.from).bind(&verb_fp)
    }

    /// Recover 'from' from edge fingerprint and 'to'
    pub fn recover_from(&self) -> Fingerprint {
        let verb_fp = self.verb.fingerprint();
        self.fingerprint.bind(&self.to).bind(&verb_fp)
    }
}

// =============================================================================
// COGNITIVE GRAPH
// =============================================================================

/// The cognitive graph - nodes and edges as fingerprints
pub struct CogGraph {
    /// Nodes indexed by fingerprint hash
    nodes: HashMap<u64, CogNode>,

    /// Edges indexed by fingerprint hash
    edges: HashMap<u64, CogEdge>,

    /// Adjacency: from_hash -> [(verb, to_hash)]
    adjacency: HashMap<u64, Vec<(Verb, u64)>>,

    /// Reverse adjacency: to_hash -> [(verb, from_hash)]
    reverse_adj: HashMap<u64, Vec<(Verb, u64)>>,

    /// Verb fingerprints (cached)
    verb_fps: HashMap<Verb, Fingerprint>,
}

impl CogGraph {
    pub fn new() -> Self {
        let mut verb_fps = HashMap::new();

        // Pre-compute all 144 verb fingerprints
        for i in 0..144 {
            let verb: Verb = unsafe { std::mem::transmute(i as u8) };
            verb_fps.insert(verb, verb.fingerprint());
        }

        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            adjacency: HashMap::new(),
            reverse_adj: HashMap::new(),
            verb_fps,
        }
    }

    /// Add a node
    pub fn add_node(&mut self, node: CogNode) {
        let hash = fp_hash(&node.fingerprint);
        self.nodes.insert(hash, node);
    }

    /// Add an edge
    pub fn add_edge(&mut self, edge: CogEdge) {
        let from_hash = fp_hash(&edge.from);
        let to_hash = fp_hash(&edge.to);
        let edge_hash = fp_hash(&edge.fingerprint);

        // Update adjacency lists
        self.adjacency
            .entry(from_hash)
            .or_default()
            .push((edge.verb, to_hash));

        self.reverse_adj
            .entry(to_hash)
            .or_default()
            .push((edge.verb, from_hash));

        self.edges.insert(edge_hash, edge);
    }

    /// Get node by fingerprint
    pub fn get_node(&self, fp: &Fingerprint) -> Option<&CogNode> {
        let hash = fp_hash(fp);
        self.nodes.get(&hash)
    }

    /// Get outgoing edges from a node
    pub fn outgoing(&self, fp: &Fingerprint) -> Vec<&CogEdge> {
        let hash = fp_hash(fp);
        if let Some(adj) = self.adjacency.get(&hash) {
            adj.iter()
                .filter_map(|(verb, to_hash)| {
                    // Reconstruct edge fingerprint
                    let to_node = self.nodes.get(to_hash)?;
                    let edge_fp = fp.bind(&verb.fingerprint()).bind(&to_node.fingerprint);
                    let edge_hash = fp_hash(&edge_fp);
                    self.edges.get(&edge_hash)
                })
                .collect()
        } else {
            vec![]
        }
    }

    /// Get incoming edges to a node
    pub fn incoming(&self, fp: &Fingerprint) -> Vec<&CogEdge> {
        let hash = fp_hash(fp);
        if let Some(adj) = self.reverse_adj.get(&hash) {
            adj.iter()
                .filter_map(|(verb, from_hash)| {
                    let from_node = self.nodes.get(from_hash)?;
                    let edge_fp = from_node.fingerprint.bind(&verb.fingerprint()).bind(fp);
                    let edge_hash = fp_hash(&edge_fp);
                    self.edges.get(&edge_hash)
                })
                .collect()
        } else {
            vec![]
        }
    }

    /// Find edges by verb
    pub fn edges_by_verb(&self, verb: Verb) -> Vec<&CogEdge> {
        self.edges.values().filter(|e| e.verb == verb).collect()
    }

    /// Traverse from a node following verbs
    pub fn traverse(&self, start: &Fingerprint, verbs: &[Verb], max_depth: usize) -> Vec<&CogNode> {
        let mut visited = std::collections::HashSet::new();
        let mut result = Vec::new();
        let mut frontier = vec![(start.clone(), 0usize)];

        while let Some((current_fp, depth)) = frontier.pop() {
            let hash = fp_hash(&current_fp);

            if visited.contains(&hash) || depth > max_depth {
                continue;
            }
            visited.insert(hash);

            if let Some(node) = self.nodes.get(&hash) {
                result.push(node);

                if let Some(adj) = self.adjacency.get(&hash) {
                    for (verb, to_hash) in adj {
                        if verbs.is_empty() || verbs.contains(verb) {
                            if let Some(to_node) = self.nodes.get(to_hash) {
                                frontier.push((to_node.fingerprint.clone(), depth + 1));
                            }
                        }
                    }
                }
            }
        }

        result
    }

    /// Find similar nodes by fingerprint
    pub fn find_similar(
        &self,
        query: &Fingerprint,
        threshold: f32,
        limit: usize,
    ) -> Vec<(&CogNode, f32)> {
        let mut matches: Vec<_> = self
            .nodes
            .values()
            .map(|node| {
                let sim = query.similarity(&node.fingerprint);
                (node, sim)
            })
            .filter(|(_, sim)| *sim > threshold)
            .collect();

        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        matches.truncate(limit);
        matches
    }

    /// Spreading activation from a node
    pub fn spread_activation(
        &mut self,
        start: &Fingerprint,
        initial: f32,
        decay: f32,
        threshold: f32,
    ) {
        let mut activations: HashMap<u64, f32> = HashMap::new();
        let start_hash = fp_hash(start);
        activations.insert(start_hash, initial);

        let mut frontier = vec![start_hash];

        while let Some(current_hash) = frontier.pop() {
            let current_act = *activations.get(&current_hash).unwrap_or(&0.0);

            if current_act < threshold {
                continue;
            }

            // Update node activation
            if let Some(node) = self.nodes.get_mut(&current_hash) {
                node.update_activation(current_act, 0.0);
            }

            // Spread to neighbors
            if let Some(adj) = self.adjacency.get(&current_hash) {
                for (_, to_hash) in adj {
                    let new_act = current_act * decay;
                    let entry = activations.entry(*to_hash).or_insert(0.0);
                    if new_act > *entry {
                        *entry = new_act;
                        frontier.push(*to_hash);
                    }
                }
            }
        }
    }

    /// Node count
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Edge count
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

impl Default for CogGraph {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Hash fingerprint to u64 for indexing
fn fp_hash(fp: &Fingerprint) -> u64 {
    let raw = fp.as_raw();
    let mut hash = 0u64;
    for &word in raw.iter() {
        hash ^= word;
    }
    hash
}

/// Current time in milliseconds
fn now_millis() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verb_categories() {
        assert_eq!(Verb::IsA.category(), VerbCategory::Structural);
        assert_eq!(Verb::Causes.category(), VerbCategory::Causal);
        assert_eq!(Verb::Before.category(), VerbCategory::Temporal);
        assert_eq!(Verb::Knows.category(), VerbCategory::Epistemic);
        assert_eq!(Verb::Does.category(), VerbCategory::Agentive);
        assert_eq!(Verb::Sees.category(), VerbCategory::Experiential);
    }

    #[test]
    fn test_verb_board_position() {
        let (row, col) = Verb::IsA.board_position();
        assert_eq!(row, 0);
        assert_eq!(col, 0);

        let (row, col) = Verb::Causes.board_position();
        assert_eq!(row, 2);
        assert_eq!(col, 0);
    }

    #[test]
    fn test_verb_count() {
        // We should have exactly 144 verbs
        assert_eq!(Verb::Inspires as u8, 143);
    }

    #[test]
    fn test_edge_creation() {
        let cat = Fingerprint::from_content("cat");
        let animal = Fingerprint::from_content("animal");

        let edge = CogEdge::new(cat.clone(), Verb::IsA, animal.clone());

        // Verify binding
        let recovered_to = edge.recover_to();
        assert!(recovered_to.similarity(&animal) > 0.99);
    }

    #[test]
    fn test_graph_basic() {
        let mut graph = CogGraph::new();

        // Create nodes
        let cat_fp = Fingerprint::from_content("cat");
        let animal_fp = Fingerprint::from_content("animal");

        let cat = CogNode::new(cat_fp.clone(), NodeType::Concept).with_name("cat");
        let animal = CogNode::new(animal_fp.clone(), NodeType::Concept).with_name("animal");

        graph.add_node(cat);
        graph.add_node(animal);

        // Create edge: cat IS_A animal
        let edge = CogEdge::new(cat_fp.clone(), Verb::IsA, animal_fp.clone());
        graph.add_edge(edge);

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);

        // Query
        let outgoing = graph.outgoing(&cat_fp);
        assert_eq!(outgoing.len(), 1);
        assert_eq!(outgoing[0].verb, Verb::IsA);
    }

    #[test]
    fn test_find_similar() {
        let mut graph = CogGraph::new();

        // Add some concepts
        for name in ["cat", "dog", "tiger", "lion", "car", "truck"] {
            let fp = Fingerprint::from_content(name);
            let node = CogNode::new(fp, NodeType::Concept).with_name(name);
            graph.add_node(node);
        }

        // Find similar to "cat"
        let query = Fingerprint::from_content("cat");
        let similar = graph.find_similar(&query, 0.0, 3);

        assert!(!similar.is_empty());
        // First should be exact match
        assert!(similar[0].1 > 0.99);
    }
}
