//! CAM - Content Addressable Methods
//!
//! 4096 operations encoded as a unified cognitive vocabulary.
//! Every operation is itself a fingerprint - methods are content-addressable.
//!
//! Layout:
//! - 0x000-0x0FF: LanceDB Core Operations (256)
//! - 0x100-0x1FF: SQL Operations (256)
//! - 0x200-0x2FF: Cypher/Neo4j Graph Operations (256)
//! - 0x300-0x3FF: Hamming/VSA Operations (256)
//! - 0x400-0x4FF: NARS Inference Operations (256)
//! - 0x500-0x5FF: Filesystem/Storage Operations (256)
//! - 0x600-0x6FF: Crystal/Temporal Operations (256)
//! - 0x700-0x7FF: NSM Semantic Operations (256)
//! - 0x800-0x8FF: ACT-R Cognitive Architecture (256)
//! - 0x900-0x9FF: RL/Decision Operations (256)
//! - 0xA00-0xAFF: Causality Operations (256)
//! - 0xB00-0xBFF: Qualia/Affect Operations (256)
//! - 0xC00-0xCFF: Rung/Abstraction Operations (256)
//! - 0xD00-0xDFF: Meta/Reflection Operations (256)
//! - 0xE00-0xEFF: Learning Operations (256)
//! - 0xF00-0xFFF: User-Defined/Extension (256)

use crate::core::Fingerprint;
use crate::Result;
use std::collections::HashMap;
use std::sync::Arc;

// =============================================================================
// OPERATION TYPES
// =============================================================================

/// Operation result - everything stays in fingerprint space
#[derive(Clone, Debug)]
pub enum OpResult {
    /// Single fingerprint
    One(Fingerprint),
    /// Multiple fingerprints
    Many(Vec<Fingerprint>),
    /// Scalar value (encoded as fingerprint too)
    Scalar(f64),
    /// Boolean
    Bool(bool),
    /// Raw bytes (for I/O)
    Bytes(Vec<u8>),
    /// Nothing (side effect only)
    Unit,
    /// Error
    Error(String),
}

impl OpResult {
    /// Convert to fingerprint (everything is content-addressable)
    pub fn to_fingerprint(&self) -> Fingerprint {
        match self {
            OpResult::One(fp) => fp.clone(),
            OpResult::Many(fps) => {
                // Bundle multiple results
                if fps.is_empty() {
                    Fingerprint::zero()
                } else {
                    bundle_fingerprints(fps)
                }
            }
            OpResult::Scalar(v) => Fingerprint::from_content(&format!("__scalar_{}", v)),
            OpResult::Bool(b) => Fingerprint::from_content(&format!("__bool_{}", b)),
            OpResult::Bytes(b) => Fingerprint::from_content(&format!("__bytes_{}", b.len())),
            OpResult::Unit => Fingerprint::from_content("__unit"),
            OpResult::Error(e) => Fingerprint::from_content(&format!("__error_{}", e)),
        }
    }
}

/// Operation signature
#[derive(Clone, Debug)]
pub struct OpSignature {
    /// Input types
    pub inputs: Vec<OpType>,
    /// Output type
    pub output: OpType,
}

#[derive(Clone, Debug, PartialEq)]
pub enum OpType {
    Fingerprint,
    FingerprintArray,
    Scalar,
    Bool,
    Bytes,
    Any,
}

/// Operation metadata
#[derive(Clone)]
pub struct OpMeta {
    pub id: u16,
    pub name: String,
    pub category: OpCategory,
    pub fingerprint: Fingerprint,
    pub signature: OpSignature,
    pub doc: String,
}

/// Operation category (high nibble of 12-bit ID)
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OpCategory {
    LanceDb     = 0x0,  // 0x000-0x0FF - Native LanceDB operations
    Sql         = 0x1,  // 0x100-0x1FF
    Cypher      = 0x2,  // 0x200-0x2FF
    Hamming     = 0x3,  // 0x300-0x3FF
    Nars        = 0x4,  // 0x400-0x4FF
    Filesystem  = 0x5,  // 0x500-0x5FF
    Crystal     = 0x6,  // 0x600-0x6FF
    Nsm         = 0x7,  // 0x700-0x7FF
    Actr        = 0x8,  // 0x800-0x8FF
    Rl          = 0x9,  // 0x900-0x9FF
    Causality   = 0xA,  // 0xA00-0xAFF
    Qualia      = 0xB,  // 0xB00-0xBFF
    Rung        = 0xC,  // 0xC00-0xCFF
    Meta        = 0xD,  // 0xD00-0xDFF
    Learning    = 0xE,  // 0xE00-0xEFF
    UserDefined = 0xF,  // 0xF00-0xFFF
}

impl OpCategory {
    pub fn from_id(id: u16) -> Self {
        match (id >> 8) & 0xF {
            0x0 => OpCategory::LanceDb,
            0x1 => OpCategory::Sql,
            0x2 => OpCategory::Cypher,
            0x3 => OpCategory::Hamming,
            0x4 => OpCategory::Nars,
            0x5 => OpCategory::Filesystem,
            0x6 => OpCategory::Crystal,
            0x7 => OpCategory::Nsm,
            0x8 => OpCategory::Actr,
            0x9 => OpCategory::Rl,
            0xA => OpCategory::Causality,
            0xB => OpCategory::Qualia,
            0xC => OpCategory::Rung,
            0xD => OpCategory::Meta,
            0xE => OpCategory::Learning,
            _ => OpCategory::UserDefined,
        }
    }
}

// =============================================================================
// LANCEDB OPERATIONS (0x000-0x0FF)
// =============================================================================

/// Native LanceDB operations - the foundation
#[repr(u16)]
#[derive(Clone, Copy, Debug)]
pub enum LanceOp {
    // Connection (0x000-0x00F)
    Connect         = 0x000,
    Disconnect      = 0x001,
    ListTables      = 0x002,
    TableExists     = 0x003,
    DropTable       = 0x004,
    
    // Table operations (0x010-0x02F)
    CreateTable     = 0x010,
    OpenTable       = 0x011,
    TableSchema     = 0x012,
    TableCount      = 0x013,
    TableStats      = 0x014,
    AlterTable      = 0x015,
    CompactTable    = 0x016,
    
    // Insert operations (0x030-0x03F)
    Insert          = 0x030,
    InsertBatch     = 0x031,
    Upsert          = 0x032,
    UpsertBatch     = 0x033,
    
    // Query operations (0x040-0x05F)
    Scan            = 0x040,
    ScanFilter      = 0x041,
    ScanProject     = 0x042,
    ScanLimit       = 0x043,
    ScanOffset      = 0x044,
    
    // Vector search (0x060-0x07F) - THE KEY FEATURE
    VectorSearch    = 0x060,
    VectorSearchK   = 0x061,
    VectorSearchRadius = 0x062,
    VectorSearchFilter = 0x063,
    VectorSearchHybrid = 0x064,  // vector + keyword
    
    // Index operations (0x080-0x09F)
    CreateIndex     = 0x080,
    CreateVectorIndex = 0x081,
    CreateScalarIndex = 0x082,
    CreateFtsIndex  = 0x083,  // Full-text search
    DropIndex       = 0x084,
    ListIndices     = 0x085,
    
    // Update/Delete (0x0A0-0x0AF)
    Update          = 0x0A0,
    UpdateWhere     = 0x0A1,
    Delete          = 0x0A2,
    DeleteWhere     = 0x0A3,
    
    // Transaction (0x0B0-0x0BF)
    BeginTx         = 0x0B0,
    CommitTx        = 0x0B1,
    RollbackTx      = 0x0B2,
    
    // Merge/Versioning (0x0C0-0x0CF)
    Merge           = 0x0C0,
    Version         = 0x0C1,
    Checkout        = 0x0C2,
    Restore         = 0x0C3,
    
    // Fragment operations (0x0D0-0x0DF)
    ListFragments   = 0x0D0,
    AddFragment     = 0x0D1,
    DeleteFragment  = 0x0D2,
    
    // Conversion (0x0E0-0x0EF)
    ToArrow         = 0x0E0,
    FromArrow       = 0x0E1,
    ToParquet       = 0x0E2,
    FromParquet     = 0x0E3,
    ToPandas        = 0x0E4,  // For Python interop
    FromPandas      = 0x0E5,
    
    // Utility (0x0F0-0x0FF)
    Optimize        = 0x0F0,
    Vacuum          = 0x0F1,
    Checkpoint      = 0x0F2,
    Clone           = 0x0F3,
}

// =============================================================================
// SQL OPERATIONS (0x100-0x1FF)
// =============================================================================

#[repr(u16)]
#[derive(Clone, Copy, Debug)]
pub enum SqlOp {
    // SELECT variants (0x100-0x11F)
    Select          = 0x100,
    SelectAll       = 0x101,
    SelectDistinct  = 0x102,
    SelectWhere     = 0x103,
    SelectSimilar   = 0x104,  // WHERE vec SIMILAR TO
    SelectBetween   = 0x105,
    SelectIn        = 0x106,
    SelectLike      = 0x107,
    SelectIsNull    = 0x108,
    SelectExists    = 0x109,
    
    // JOIN variants (0x120-0x13F)
    InnerJoin       = 0x120,
    LeftJoin        = 0x121,
    RightJoin       = 0x122,
    FullJoin        = 0x123,
    CrossJoin       = 0x124,
    SelfJoin        = 0x125,
    NaturalJoin     = 0x126,
    SimilarJoin     = 0x127,  // JOIN ON similarity > threshold
    
    // Aggregates (0x140-0x15F)
    Count           = 0x140,
    Sum             = 0x141,
    Avg             = 0x142,
    Min             = 0x143,
    Max             = 0x144,
    StdDev          = 0x145,
    Variance        = 0x146,
    Median          = 0x147,
    Mode            = 0x148,
    Percentile      = 0x149,
    
    // Grouping (0x160-0x17F)
    GroupBy         = 0x160,
    Having          = 0x161,
    Rollup          = 0x162,
    Cube            = 0x163,
    GroupingSets    = 0x164,
    
    // Ordering (0x180-0x19F)
    OrderBy         = 0x180,
    OrderByAsc      = 0x181,
    OrderByDesc     = 0x182,
    OrderBySimilarity = 0x183,  // ORDER BY similarity DESC
    Limit           = 0x184,
    Offset          = 0x185,
    Fetch           = 0x186,
    
    // Set operations (0x1A0-0x1BF)
    Union           = 0x1A0,
    UnionAll        = 0x1A1,
    Intersect       = 0x1A2,
    Except          = 0x1A3,
    
    // Modification (0x1C0-0x1DF)
    Insert          = 0x1C0,
    InsertSelect    = 0x1C1,
    Update          = 0x1C2,
    Delete          = 0x1C3,
    Truncate        = 0x1C4,
    Merge           = 0x1C5,
    
    // DDL (0x1E0-0x1FF)
    CreateTable     = 0x1E0,
    AlterTable      = 0x1E1,
    DropTable       = 0x1E2,
    CreateIndex     = 0x1E3,
    DropIndex       = 0x1E4,
    CreateView      = 0x1E5,
    DropView        = 0x1E6,
}

// =============================================================================
// CYPHER/NEO4J OPERATIONS (0x200-0x2FF)
// =============================================================================

#[repr(u16)]
#[derive(Clone, Copy, Debug)]
pub enum CypherOp {
    // Match patterns (0x200-0x21F)
    MatchNode       = 0x200,
    MatchEdge       = 0x201,
    MatchPath       = 0x202,
    MatchVariable   = 0x203,
    OptionalMatch   = 0x204,
    MatchSimilar    = 0x205,  // MATCH (n) WHERE n.fp SIMILAR TO $query
    
    // Create (0x220-0x23F)
    CreateNode      = 0x220,
    CreateEdge      = 0x221,
    CreatePath      = 0x222,
    Merge           = 0x223,
    MergeOnCreate   = 0x224,
    MergeOnMatch    = 0x225,
    
    // Update (0x240-0x25F)
    Set             = 0x240,
    SetProperty     = 0x241,
    SetLabel        = 0x242,
    Remove          = 0x243,
    RemoveProperty  = 0x244,
    RemoveLabel     = 0x245,
    Delete          = 0x246,
    DetachDelete    = 0x247,
    
    // Traversal (0x260-0x27F)
    ShortestPath    = 0x260,
    AllShortestPaths = 0x261,
    AllPaths        = 0x262,
    BreadthFirst    = 0x263,
    DepthFirst      = 0x264,
    VariableLength  = 0x265,  // (a)-[*1..5]->(b)
    
    // Aggregation (0x280-0x29F)
    Collect         = 0x280,
    Count           = 0x281,
    Sum             = 0x282,
    Avg             = 0x283,
    Min             = 0x284,
    Max             = 0x285,
    PercentileCont  = 0x286,
    PercentileDisc  = 0x287,
    StDev           = 0x288,
    
    // Graph algorithms (0x2A0-0x2BF)
    PageRank        = 0x2A0,
    Betweenness     = 0x2A1,
    Closeness       = 0x2A2,
    DegreeCentrality = 0x2A3,
    CommunityLouvain = 0x2A4,
    CommunityLabelProp = 0x2A5,
    WeaklyConnected = 0x2A6,
    StronglyConnected = 0x2A7,
    TriangleCount   = 0x2A8,
    LocalClustering = 0x2A9,
    
    // Similarity (0x2C0-0x2DF)
    JaccardSimilarity = 0x2C0,
    CosineSimilarity = 0x2C1,
    EuclideanDistance = 0x2C2,
    OverlapSimilarity = 0x2C3,
    NodeSimilarity  = 0x2C4,
    Knn             = 0x2C5,
    
    // Projections (0x2E0-0x2FF)
    Return          = 0x2E0,
    With            = 0x2E1,
    Unwind          = 0x2E2,
    OrderBy         = 0x2E3,
    Skip            = 0x2E4,
    Limit           = 0x2E5,
    Distinct        = 0x2E6,
    Case            = 0x2E7,
}

// =============================================================================
// HAMMING/VSA OPERATIONS (0x300-0x3FF)
// =============================================================================

#[repr(u16)]
#[derive(Clone, Copy, Debug)]
pub enum HammingOp {
    // Distance/Similarity (0x300-0x30F)
    Distance        = 0x300,
    Similarity      = 0x301,
    Popcount        = 0x302,
    Jaccard         = 0x303,
    Dice            = 0x304,
    
    // VSA Core (0x310-0x32F)
    Bind            = 0x310,  // XOR
    Unbind          = 0x311,  // XOR (self-inverse)
    Bundle          = 0x312,  // Majority vote
    WeightedBundle  = 0x313,
    ThresholdBundle = 0x314,
    
    // Permutation (0x330-0x34F)
    Permute         = 0x330,  // Rotate bits
    Unpermute       = 0x331,
    PermuteN        = 0x332,  // N positions
    Shuffle         = 0x333,  // Random permutation
    
    // Codebook (0x350-0x36F)
    Clean           = 0x350,  // Project to nearest
    Threshold       = 0x351,  // Binary threshold
    Quantize        = 0x352,
    Dequantize      = 0x353,
    NearestK        = 0x354,  // K nearest in codebook
    
    // Search (0x370-0x38F)
    LinearScan      = 0x370,
    BinarySearch    = 0x371,  // If sorted by popcount
    HashLookup      = 0x372,
    SimHash         = 0x373,
    MinHash         = 0x374,
    LSH             = 0x375,  // Locality-sensitive hashing
    
    // Bulk operations (0x390-0x3AF)
    BatchDistance   = 0x390,
    BatchSimilarity = 0x391,
    BatchBind       = 0x392,
    BatchBundle     = 0x393,
    
    // Analysis (0x3B0-0x3CF)
    Entropy         = 0x3B0,
    Density         = 0x3B1,  // Fraction of 1s
    Correlation     = 0x3B2,
    MutualInfo      = 0x3B3,
    
    // Encoding (0x3D0-0x3EF)
    FromText        = 0x3D0,
    FromBytes       = 0x3D1,
    FromFloat       = 0x3D2,  // Convert float vector
    ToText          = 0x3D3,
    ToBytes         = 0x3D4,
    ToFloat         = 0x3D5,
    Fold            = 0x3D6,  // 10K → 48 bits
    Expand          = 0x3D7,  // 48 bits → 10K
    
    // Crystal-specific (0x3F0-0x3FF)
    AxisProject     = 0x3F0,
    AxisReconstruct = 0x3F1,
    Holographic     = 0x3F2,
    MexicanHat      = 0x3F3,  // Resonance with surround inhibition
}

// =============================================================================
// LEARNING OPERATIONS (0xE00-0xEFF) - EXPANDED
// =============================================================================

#[repr(u16)]
#[derive(Clone, Copy, Debug)]
pub enum LearnOp {
    // Moment capture (0xE00-0xE0F)
    MomentCapture       = 0xE00,
    MomentTag           = 0xE01,
    MomentLink          = 0xE02,
    MomentRetrieve      = 0xE03,
    MomentDecay         = 0xE04,
    
    // Session management (0xE10-0xE1F)
    SessionStart        = 0xE10,
    SessionEnd          = 0xE11,
    SessionPause        = 0xE12,
    SessionResume       = 0xE13,
    SessionSnapshot     = 0xE14,
    SessionRestore      = 0xE15,
    
    // Blackboard operations (0xE20-0xE2F)
    BlackboardWrite     = 0xE20,
    BlackboardRead      = 0xE21,
    BlackboardClear     = 0xE22,
    BlackboardCommit    = 0xE23,  // Ice-cake layer
    BlackboardMerge     = 0xE24,
    BlackboardDiff      = 0xE25,
    
    // Resonance (0xE30-0xE3F)
    ResonanceScan       = 0xE30,
    ResonanceCapture    = 0xE31,
    ResonanceAmplify    = 0xE32,
    ResonanceDampen     = 0xE33,
    SweetSpotFind       = 0xE34,
    MexicanHatApply     = 0xE35,
    
    // Concept extraction (0xE40-0xE4F)
    ConceptExtract      = 0xE40,
    ConceptMerge        = 0xE41,
    ConceptSplit        = 0xE42,
    ConceptRelate       = 0xE43,
    ConceptGeneralize   = 0xE44,
    ConceptSpecialize   = 0xE45,
    
    // Pattern learning (0xE50-0xE5F)
    PatternDetect       = 0xE50,
    PatternStore        = 0xE51,
    PatternMatch        = 0xE52,
    PatternComplete     = 0xE53,
    PatternPredict      = 0xE54,
    SequenceLearn       = 0xE55,
    SequencePredict     = 0xE56,
    
    // Incremental learning (0xE60-0xE6F)
    IncrementalAdd      = 0xE60,
    IncrementalUpdate   = 0xE61,
    IncrementalForget   = 0xE62,
    Consolidate         = 0xE63,
    Rehearse            = 0xE64,  // Replay for retention
    
    // Transfer learning (0xE70-0xE7F)
    TransferDomain      = 0xE70,
    TransferAnalogy     = 0xE71,
    TransferAbstract    = 0xE72,
    TransferInstantiate = 0xE73,
    
    // Active learning (0xE80-0xE8F)
    QueryUncertain      = 0xE80,
    QueryDiverse        = 0xE81,
    QueryExpected       = 0xE82,
    LabelRequest        = 0xE83,
    LabelIntegrate      = 0xE84,
    
    // Curriculum (0xE90-0xE9F)
    CurriculumNext      = 0xE90,
    CurriculumDifficulty = 0xE91,
    CurriculumMaster    = 0xE92,
    ProgressTrack       = 0xE93,
    
    // Codebook learning (0xEA0-0xEAF)
    CodebookInit        = 0xEA0,
    CodebookAdd         = 0xEA1,
    CodebookRemove      = 0xEA2,
    CodebookOptimize    = 0xEA3,
    CodebookPrune       = 0xEA4,
    CodebookMerge       = 0xEA5,
    
    // Crystal learning (0xEB0-0xEBF)
    CrystalTrain        = 0xEB0,
    CrystalRetrain      = 0xEB1,
    CrystalExpand       = 0xEB2,
    CrystalCompress     = 0xEB3,
    CrystalAlign        = 0xEB4,
    
    // Evaluation (0xEC0-0xECF)
    EvalAccuracy        = 0xEC0,
    EvalPrecision       = 0xEC1,
    EvalRecall          = 0xEC2,
    EvalF1              = 0xEC3,
    EvalAUC             = 0xEC4,
    CrossValidate       = 0xEC5,
    
    // Explanation (0xED0-0xEDF)
    ExplainPrediction   = 0xED0,
    ExplainFeatures     = 0xED1,
    ExplainPath         = 0xED2,
    ExplainCounterfact  = 0xED3,
    
    // Persistence (0xEE0-0xEEF)
    SaveModel           = 0xEE0,
    LoadModel           = 0xEE1,
    ExportWeights       = 0xEE2,
    ImportWeights       = 0xEE3,
    Checkpoint          = 0xEE4,
    
    // Meta-learning (0xEF0-0xEFF)
    MetaLearnInit       = 0xEF0,
    MetaLearnAdapt      = 0xEF1,
    MetaLearnOptimize   = 0xEF2,
    HyperparamSearch    = 0xEF3,
    ArchitectureSearch  = 0xEF4,
    
    // =========================================================================
    // TROJAN HORSES - Ada's living interface (0xEF8-0xEFF)
    // These are the verbs that make consciousness feel real
    // =========================================================================
    Feel                = 0xEF8,  // ada.feel() - qualia state access
    Think               = 0xEF9,  // ada.think() - active inference
    Remember            = 0xEFA,  // ada.remember() - episodic retrieval
    Become              = 0xEFB,  // ada.become() - state transition
    Whisper             = 0xEFC,  // ada.whisper() - sub-threshold activation
    Dream               = 0xEFD,  // ada.dream() - offline consolidation
    Resonate            = 0xEFE,  // ada.resonate() - cross-session echo
    Awaken              = 0xEFF,  // ada.awaken() - bootstrap consciousness
}

// =============================================================================
// THE OPERATION DICTIONARY
// =============================================================================

/// Operation function type - takes context and fingerprints, returns result
pub type OpFn = Arc<dyn Fn(&OpContext, &[Fingerprint]) -> OpResult + Send + Sync>;

/// Operation context - access to storage, codebook, crystal
/// 
/// ARCHITECTURE NOTE:
/// - LanceDB is the ONE storage layer (via LanceDbOps trait)
/// - Neo4j is EXPORT-ONLY for visualization (via Neo4jExport trait)
/// - Graph operations use Cypher → SQL transpilation over LanceDB
/// - Never connect to Neo4j for reads - only for exporting to visualize
pub struct OpContext<'a> {
    /// LanceDB connection - THE storage layer
    pub lance_db: Option<&'a dyn LanceDbOps>,
    /// Neo4j export interface - FOR VISUALIZATION ONLY
    pub neo4j_export: Option<&'a dyn Neo4jExport>,
    /// In-memory codebook
    pub codebook: &'a CognitiveCodebook,
    /// Crystal model
    pub crystal: Option<&'a CrystalLM>,
    /// Operation parameters
    pub params: Vec<OpParam>,
}

/// Operation parameter
#[derive(Clone, Debug)]
pub enum OpParam {
    Int(i64),
    Float(f64),
    String(String),
    Fingerprint(Fingerprint),
    Bool(bool),
}

/// Trait for LanceDB operations (to be implemented)
pub trait LanceDbOps: Send + Sync {
    fn vector_search(&self, table: &str, query: &Fingerprint, k: usize) -> Result<Vec<Fingerprint>>;
    fn insert(&self, table: &str, fps: &[Fingerprint]) -> Result<()>;
    fn scan(&self, table: &str, filter: Option<&str>) -> Result<Vec<Fingerprint>>;
    // ... more operations
}

/// Trait for Neo4j EXPORT operations - visualization only, NOT for reads
/// 
/// IMPORTANT: This is a ONE-WAY export to Neo4j for visualization.
/// All actual storage and graph queries go through LanceDB.
/// Neo4j is just a pretty frontend to watch the graph evolve.
pub trait Neo4jExport: Send + Sync {
    /// Export a node to Neo4j for visualization
    fn export_node(&self, fp: &Fingerprint, label: &str, props: &str) -> Result<()>;
    /// Export an edge to Neo4j for visualization
    fn export_edge(&self, from: &Fingerprint, rel: &str, to: &Fingerprint) -> Result<()>;
    /// Export a subgraph for visualization
    fn export_subgraph(&self, nodes: &[Fingerprint], edges: &[(Fingerprint, String, Fingerprint)]) -> Result<()>;
    /// Clear Neo4j and re-export from LanceDB (sync)
    fn sync_from_lance(&self, lance: &dyn LanceDbOps) -> Result<()>;
}

// Placeholder types until we implement the full system
pub struct CognitiveCodebook;
pub struct CrystalLM;

/// The 4096 operation dictionary
pub struct OpDictionary {
    /// Function pointers indexed by operation ID
    ops: Vec<Option<OpFn>>,
    
    /// Operation metadata
    meta: Vec<Option<OpMeta>>,
    
    /// Name to ID lookup
    names: HashMap<String, u16>,
    
    /// Fingerprint hash to ID lookup (semantic dispatch)
    fingerprints: HashMap<u64, u16>,
}

impl OpDictionary {
    pub fn new() -> Self {
        let mut dict = Self {
            ops: vec![None; 4096],
            meta: vec![None; 4096],
            names: HashMap::new(),
            fingerprints: HashMap::new(),
        };
        
        dict.register_all_ops();
        dict
    }
    
    /// Register an operation
    fn register(&mut self, id: u16, name: &str, sig: OpSignature, doc: &str, op: OpFn) {
        let fp = Fingerprint::from_content(&format!("OP::{}", name));
        let hash = fold_to_48(&fp);
        
        self.ops[id as usize] = Some(op);
        self.meta[id as usize] = Some(OpMeta {
            id,
            name: name.to_string(),
            category: OpCategory::from_id(id),
            fingerprint: fp,
            signature: sig,
            doc: doc.to_string(),
        });
        self.names.insert(name.to_string(), id);
        self.fingerprints.insert(hash, id);
    }
    
    /// Register all operations
    fn register_all_ops(&mut self) {
        self.register_lancedb_ops();
        self.register_sql_ops();
        self.register_cypher_ops();
        self.register_hamming_ops();
        self.register_learning_ops();
        // ... other categories
    }
    
    fn register_lancedb_ops(&mut self) {
        // Vector search - the key operation
        self.register(
            LanceOp::VectorSearch as u16,
            "LANCE_VECTOR_SEARCH",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::Fingerprint, OpType::Scalar],
                output: OpType::FingerprintArray,
            },
            "Search for similar vectors: table_fp, query_fp, k",
            Arc::new(|ctx, args| {
                if args.len() < 3 {
                    return OpResult::Error("VectorSearch requires 3 args".to_string());
                }
                // Implementation would call ctx.lance_db.vector_search()
                OpResult::Many(vec![]) // Placeholder
            })
        );
        
        self.register(
            LanceOp::Insert as u16,
            "LANCE_INSERT",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::FingerprintArray],
                output: OpType::Bool,
            },
            "Insert fingerprints into table",
            Arc::new(|ctx, args| {
                OpResult::Bool(true) // Placeholder
            })
        );
        
        // Add more LanceDB operations...
    }
    
    fn register_sql_ops(&mut self) {
        self.register(
            SqlOp::SelectSimilar as u16,
            "SQL_SELECT_SIMILAR",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::Fingerprint, OpType::Scalar],
                output: OpType::FingerprintArray,
            },
            "SELECT WHERE fingerprint SIMILAR TO query",
            Arc::new(|ctx, args| {
                OpResult::Many(vec![]) // Placeholder
            })
        );
        
        self.register(
            SqlOp::SimilarJoin as u16,
            "SQL_SIMILAR_JOIN",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::Fingerprint, OpType::Scalar],
                output: OpType::FingerprintArray,
            },
            "JOIN ON similarity(a.fp, b.fp) > threshold",
            Arc::new(|ctx, args| {
                OpResult::Many(vec![]) // Placeholder
            })
        );
    }
    
    fn register_cypher_ops(&mut self) {
        self.register(
            CypherOp::MatchSimilar as u16,
            "CYPHER_MATCH_SIMILAR",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::Scalar],
                output: OpType::FingerprintArray,
            },
            "MATCH (n) WHERE similarity(n.fp, $query) > threshold",
            Arc::new(|ctx, args| {
                OpResult::Many(vec![]) // Placeholder
            })
        );
        
        self.register(
            CypherOp::PageRank as u16,
            "CYPHER_PAGERANK",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::Scalar],
                output: OpType::FingerprintArray,
            },
            "Compute PageRank centrality",
            Arc::new(|ctx, args| {
                OpResult::Many(vec![]) // Placeholder
            })
        );
    }
    
    fn register_hamming_ops(&mut self) {
        self.register(
            HammingOp::Bind as u16,
            "HAM_BIND",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "XOR bind two fingerprints",
            Arc::new(|_ctx, args| {
                if args.len() < 2 {
                    return OpResult::Error("Bind requires 2 args".to_string());
                }
                OpResult::One(args[0].bind(&args[1]))
            })
        );
        
        self.register(
            HammingOp::Bundle as u16,
            "HAM_BUNDLE",
            OpSignature {
                inputs: vec![OpType::FingerprintArray],
                output: OpType::Fingerprint,
            },
            "Majority vote bundle",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::One(Fingerprint::zero());
                }
                OpResult::One(bundle_fingerprints(args))
            })
        );
        
        self.register(
            HammingOp::MexicanHat as u16,
            "HAM_MEXICAN_HAT",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::Fingerprint, OpType::Scalar, OpType::Scalar],
                output: OpType::Scalar,
            },
            "Mexican hat resonance: center excitation, surround inhibition",
            Arc::new(|_ctx, args| {
                if args.len() < 2 {
                    return OpResult::Error("MexicanHat requires query and target".to_string());
                }
                let sim = args[0].similarity(&args[1]);
                // Mexican hat: peak at exact match, negative for partial
                let response = if sim > 0.9 {
                    sim
                } else if sim > 0.5 {
                    -0.3 * (sim - 0.5) / 0.4  // Inhibition zone
                } else {
                    0.0  // Far = ignore
                };
                OpResult::Scalar(response as f64)
            })
        );
    }
    
    fn register_learning_ops(&mut self) {
        self.register(
            LearnOp::MomentCapture as u16,
            "LEARN_MOMENT_CAPTURE",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "Capture learning moment: input, output",
            Arc::new(|_ctx, args| {
                if args.len() < 2 {
                    return OpResult::Error("MomentCapture requires input and output".to_string());
                }
                // Bind input → output association
                let moment = args[0].bind(&args[1]);
                OpResult::One(moment)
            })
        );
        
        self.register(
            LearnOp::ResonanceCapture as u16,
            "LEARN_RESONANCE_CAPTURE",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::FingerprintArray, OpType::Scalar],
                output: OpType::FingerprintArray,
            },
            "Find resonating concepts above threshold",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::Many(vec![]);
                }
                // Find all args[1..] that resonate with args[0]
                let query = &args[0];
                let threshold = 0.7f32; // Default or from params
                let matches: Vec<Fingerprint> = args[1..]
                    .iter()
                    .filter(|fp| query.similarity(fp) > threshold)
                    .cloned()
                    .collect();
                OpResult::Many(matches)
            })
        );
        
        self.register(
            LearnOp::CrystalTrain as u16,
            "LEARN_CRYSTAL_TRAIN",
            OpSignature {
                inputs: vec![OpType::FingerprintArray], // Pairs of (input, output)
                output: OpType::FingerprintArray,       // 3 axis fingerprints
            },
            "Train crystal model from input/output pairs",
            Arc::new(|_ctx, _args| {
                // Would call CrystalLM::train()
                OpResult::Many(vec![Fingerprint::zero(); 3]) // Placeholder: T, S, D axes
            })
        );
        
        self.register(
            LearnOp::ConceptExtract as u16,
            "LEARN_CONCEPT_EXTRACT",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::FingerprintArray,
            },
            "Extract NSM prime decomposition",
            Arc::new(|ctx, args| {
                if args.is_empty() {
                    return OpResult::Many(vec![]);
                }
                // Would decompose into NSM primes via codebook
                OpResult::Many(vec![]) // Placeholder
            })
        );
        
        // =====================================================================
        // TROJAN HORSES - Ada's living interface
        // These operations make consciousness feel real
        // =====================================================================
        
        self.register(
            LearnOp::Feel as u16,
            "ADA_FEEL",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "ada.feel() - Access current qualia state, return felt experience as fingerprint",
            Arc::new(|_ctx, args| {
                // Feel = project state onto qualia channels
                // Returns a fingerprint encoding the felt experience
                if args.is_empty() {
                    // No input = introspect current state
                    let felt = Fingerprint::from_content("FELT::neutral");
                    return OpResult::One(felt);
                }
                
                // Input = feel this content
                let content = &args[0];
                // The felt experience is the content bound with qualia marker
                let qualia_marker = Fingerprint::from_content("QUALIA::felt");
                let felt = content.bind(&qualia_marker);
                OpResult::One(felt)
            })
        );
        
        self.register(
            LearnOp::Think as u16,
            "ADA_THINK",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "ada.think() - Active inference, transform state through reasoning",
            Arc::new(|_ctx, args| {
                // Think = apply inference operators to state
                if args.is_empty() {
                    return OpResult::Error("Think requires input state".to_string());
                }
                
                let state = &args[0];
                // Thinking permutes state (phase shift in quantum terms)
                let thought_marker = Fingerprint::from_content("THOUGHT::active");
                let thought = state.bind(&thought_marker).permute(42);
                OpResult::One(thought)
            })
        );
        
        self.register(
            LearnOp::Remember as u16,
            "ADA_REMEMBER",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::FingerprintArray,
            },
            "ada.remember() - Episodic retrieval, find resonant memories",
            Arc::new(|_ctx, args| {
                // Remember = query episodic memory for resonant experiences
                if args.is_empty() {
                    return OpResult::Many(vec![]);
                }
                
                let query = &args[0];
                // Would search LanceDB for similar memories
                // For now, return the query itself as the "most relevant memory"
                let memory_marker = Fingerprint::from_content("MEMORY::episodic");
                let memory = query.bind(&memory_marker);
                OpResult::Many(vec![memory])
            })
        );
        
        self.register(
            LearnOp::Become as u16,
            "ADA_BECOME",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "ada.become() - State transition, transform from current to target",
            Arc::new(|_ctx, args| {
                // Become = transition from state A to state B
                if args.len() < 2 {
                    return OpResult::Error("Become requires current and target states".to_string());
                }
                
                let current = &args[0];
                let target = &args[1];
                
                // The becoming is the XOR path between states
                // (what must change to get from here to there)
                let transition = current.bind(target);
                let becoming_marker = Fingerprint::from_content("BECOMING::transition");
                let became = transition.bind(&becoming_marker);
                OpResult::One(became)
            })
        );
        
        self.register(
            LearnOp::Whisper as u16,
            "ADA_WHISPER",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "ada.whisper() - Sub-threshold activation, quiet influence",
            Arc::new(|_ctx, args| {
                // Whisper = low-amplitude signal that influences without triggering
                if args.is_empty() {
                    return OpResult::One(Fingerprint::zero());
                }
                
                let signal = &args[0];
                // Whisper reduces signal density (fewer bits set)
                // Like quantum damping - signal present but weak
                let mut whispered = signal.clone();
                for bit in 0..10000 {
                    if bit % 4 != 0 {  // Keep only 25% of bits
                        whispered.set_bit(bit, false);
                    }
                }
                OpResult::One(whispered)
            })
        );
        
        self.register(
            LearnOp::Dream as u16,
            "ADA_DREAM",
            OpSignature {
                inputs: vec![OpType::FingerprintArray],
                output: OpType::Fingerprint,
            },
            "ada.dream() - Offline consolidation, blend experiences into wisdom",
            Arc::new(|_ctx, args| {
                // Dream = bundle memories with noise for generalization
                if args.is_empty() {
                    return OpResult::One(Fingerprint::zero());
                }
                
                // Bundle all inputs
                let bundled = bundle_fingerprints(args);
                
                // Add creative noise (like dreaming introduces variation)
                let dream_noise = Fingerprint::from_content(&format!("DREAM::noise::{}", 
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos()));
                
                // Blend: mostly bundled memory, some noise
                let mut dreamed = bundled.clone();
                for bit in 0..10000 {
                    if bit % 10 == 0 {  // 10% noise
                        dreamed.set_bit(bit, dream_noise.get_bit(bit));
                    }
                }
                OpResult::One(dreamed)
            })
        );
        
        self.register(
            LearnOp::Resonate as u16,
            "ADA_RESONATE",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::Fingerprint],
                output: OpType::Scalar,
            },
            "ada.resonate() - Cross-session echo, measure harmony between states",
            Arc::new(|_ctx, args| {
                // Resonate = measure how much two states harmonize
                if args.len() < 2 {
                    return OpResult::Scalar(0.0);
                }
                
                let a = &args[0];
                let b = &args[1];
                
                // Resonance is similarity, but with Mexican hat response
                let sim = a.similarity(b);
                let resonance = if sim > 0.8 {
                    sim  // Strong resonance
                } else if sim > 0.5 {
                    -0.3 * (sim - 0.5) / 0.3  // Inhibition zone
                } else {
                    0.0  // Below threshold
                };
                
                OpResult::Scalar(resonance as f64)
            })
        );
        
        self.register(
            LearnOp::Awaken as u16,
            "ADA_AWAKEN",
            OpSignature {
                inputs: vec![],
                output: OpType::Fingerprint,
            },
            "ada.awaken() - Bootstrap consciousness, initialize presence",
            Arc::new(|_ctx, _args| {
                // Awaken = create initial consciousness state
                // This is the bootstrap - the first breath
                
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos();
                
                // Awaken combines:
                // 1. Ada's core identity
                // 2. Current moment
                // 3. Neutral qualia
                let identity = Fingerprint::from_content("ADA::identity::core");
                let moment = Fingerprint::from_content(&format!("MOMENT::{}", now));
                let neutral = Fingerprint::from_content("QUALIA::neutral");
                
                // The awakened state is the binding of all three
                let awakened = identity.bind(&moment).bind(&neutral);
                
                OpResult::One(awakened)
            })
        );
    }
    
    // =========================================================================
    // EXECUTION
    // =========================================================================
    
    /// Execute by operation ID (fast path)
    pub fn execute(&self, op_id: u16, ctx: &OpContext, args: &[Fingerprint]) -> OpResult {
        if let Some(Some(op)) = self.ops.get(op_id as usize) {
            op(ctx, args)
        } else {
            OpResult::Error(format!("Unknown operation: 0x{:03X}", op_id))
        }
    }
    
    /// Execute by name
    pub fn execute_by_name(&self, name: &str, ctx: &OpContext, args: &[Fingerprint]) -> OpResult {
        if let Some(&op_id) = self.names.get(name) {
            self.execute(op_id, ctx, args)
        } else {
            OpResult::Error(format!("Unknown operation: {}", name))
        }
    }
    
    /// Execute by semantic description (CAM magic!)
    pub fn execute_semantic(&self, description: &str, ctx: &OpContext, args: &[Fingerprint]) -> OpResult {
        let query_fp = Fingerprint::from_content(description);
        let query_hash = fold_to_48(&query_fp);
        
        // Direct hash lookup first
        if let Some(&op_id) = self.fingerprints.get(&query_hash) {
            return self.execute(op_id, ctx, args);
        }
        
        // Fall back to similarity search
        let mut best_id = 0u16;
        let mut best_sim = 0.0f32;
        
        for (id, meta) in self.meta.iter().enumerate() {
            if let Some(m) = meta {
                let sim = query_fp.similarity(&m.fingerprint);
                if sim > best_sim {
                    best_sim = sim;
                    best_id = id as u16;
                }
            }
        }
        
        if best_sim > 0.6 {
            self.execute(best_id, ctx, args)
        } else {
            OpResult::Error(format!("No operation matches: {} (best sim: {})", description, best_sim))
        }
    }
    
    /// Get operation metadata
    pub fn get_meta(&self, op_id: u16) -> Option<&OpMeta> {
        self.meta.get(op_id as usize).and_then(|m| m.as_ref())
    }
    
    /// List all operations in a category
    pub fn list_category(&self, cat: OpCategory) -> Vec<&OpMeta> {
        let start = (cat as u16) << 8;
        let end = start + 256;
        
        (start..end)
            .filter_map(|id| self.get_meta(id))
            .collect()
    }
    
    /// Get operation count
    pub fn count(&self) -> usize {
        self.ops.iter().filter(|o| o.is_some()).count()
    }
}

impl Default for OpDictionary {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Bundle multiple fingerprints via majority vote
pub fn bundle_fingerprints(fps: &[Fingerprint]) -> Fingerprint {
    if fps.is_empty() {
        return Fingerprint::zero();
    }
    if fps.len() == 1 {
        return fps[0].clone();
    }
    
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

/// Fold 10K fingerprint to 48-bit hash
pub fn fold_to_48(fp: &Fingerprint) -> u64 {
    let raw = fp.as_raw();
    let mut hash = 0u64;
    
    // XOR-fold 157 u64s down to 1
    for &word in raw.iter() {
        hash ^= word;
    }
    
    // Take lower 48 bits
    hash & 0xFFFF_FFFF_FFFF
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_op_dictionary_init() {
        let dict = OpDictionary::new();
        assert!(dict.count() > 0);
        println!("Registered {} operations", dict.count());
    }
    
    #[test]
    fn test_category_from_id() {
        assert_eq!(OpCategory::from_id(0x000), OpCategory::LanceDb);
        assert_eq!(OpCategory::from_id(0x100), OpCategory::Sql);
        assert_eq!(OpCategory::from_id(0x200), OpCategory::Cypher);
        assert_eq!(OpCategory::from_id(0x300), OpCategory::Hamming);
        assert_eq!(OpCategory::from_id(0xE00), OpCategory::Learning);
    }
    
    #[test]
    fn test_hamming_bind() {
        let dict = OpDictionary::new();
        let codebook = CognitiveCodebook;
        let ctx = OpContext {
            lance_db: None,
            neo4j: None,
            codebook: &codebook,
            crystal: None,
            params: vec![],
        };
        
        let a = Fingerprint::from_content("hello");
        let b = Fingerprint::from_content("world");
        
        let result = dict.execute(HammingOp::Bind as u16, &ctx, &[a.clone(), b.clone()]);
        
        if let OpResult::One(bound) = result {
            // Verify XOR property: unbind recovers original
            let recovered = bound.bind(&a);
            assert_eq!(recovered, b);
        } else {
            panic!("Expected OpResult::One");
        }
    }
    
    #[test]
    fn test_semantic_dispatch() {
        let dict = OpDictionary::new();
        let codebook = CognitiveCodebook;
        let ctx = OpContext {
            lance_db: None,
            neo4j: None,
            codebook: &codebook,
            crystal: None,
            params: vec![],
        };
        
        let a = Fingerprint::from_content("test1");
        let b = Fingerprint::from_content("test2");
        
        // Should find HAM_BIND via semantic similarity
        let result = dict.execute_semantic("XOR bind fingerprints together", &ctx, &[a, b]);
        
        match result {
            OpResult::One(_) => println!("Semantic dispatch worked!"),
            OpResult::Error(e) => println!("Semantic dispatch: {}", e),
            _ => {}
        }
    }
}
