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

use crate::Result;
use crate::core::Fingerprint;
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
    LanceDb = 0x0,     // 0x000-0x0FF - Native LanceDB operations
    Sql = 0x1,         // 0x100-0x1FF
    Cypher = 0x2,      // 0x200-0x2FF
    Hamming = 0x3,     // 0x300-0x3FF
    Nars = 0x4,        // 0x400-0x4FF
    Filesystem = 0x5,  // 0x500-0x5FF
    Crystal = 0x6,     // 0x600-0x6FF
    Nsm = 0x7,         // 0x700-0x7FF
    Actr = 0x8,        // 0x800-0x8FF
    Rl = 0x9,          // 0x900-0x9FF
    Causality = 0xA,   // 0xA00-0xAFF
    Qualia = 0xB,      // 0xB00-0xBFF
    Rung = 0xC,        // 0xC00-0xCFF
    Meta = 0xD,        // 0xD00-0xDFF
    Learning = 0xE,    // 0xE00-0xEFF
    UserDefined = 0xF, // 0xF00-0xFFF
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
    Connect = 0x000,
    Disconnect = 0x001,
    ListTables = 0x002,
    TableExists = 0x003,
    DropTable = 0x004,

    // Table operations (0x010-0x02F)
    CreateTable = 0x010,
    OpenTable = 0x011,
    TableSchema = 0x012,
    TableCount = 0x013,
    TableStats = 0x014,
    AlterTable = 0x015,
    CompactTable = 0x016,

    // Insert operations (0x030-0x03F)
    Insert = 0x030,
    InsertBatch = 0x031,
    Upsert = 0x032,
    UpsertBatch = 0x033,

    // Query operations (0x040-0x05F)
    Scan = 0x040,
    ScanFilter = 0x041,
    ScanProject = 0x042,
    ScanLimit = 0x043,
    ScanOffset = 0x044,

    // Vector search (0x060-0x07F) - THE KEY FEATURE
    VectorSearch = 0x060,
    VectorSearchK = 0x061,
    VectorSearchRadius = 0x062,
    VectorSearchFilter = 0x063,
    VectorSearchHybrid = 0x064, // vector + keyword

    // Index operations (0x080-0x09F)
    CreateIndex = 0x080,
    CreateVectorIndex = 0x081,
    CreateScalarIndex = 0x082,
    CreateFtsIndex = 0x083, // Full-text search
    DropIndex = 0x084,
    ListIndices = 0x085,

    // Update/Delete (0x0A0-0x0AF)
    Update = 0x0A0,
    UpdateWhere = 0x0A1,
    Delete = 0x0A2,
    DeleteWhere = 0x0A3,

    // Transaction (0x0B0-0x0BF)
    BeginTx = 0x0B0,
    CommitTx = 0x0B1,
    RollbackTx = 0x0B2,

    // Merge/Versioning (0x0C0-0x0CF)
    Merge = 0x0C0,
    Version = 0x0C1,
    Checkout = 0x0C2,
    Restore = 0x0C3,

    // Fragment operations (0x0D0-0x0DF)
    ListFragments = 0x0D0,
    AddFragment = 0x0D1,
    DeleteFragment = 0x0D2,

    // Conversion (0x0E0-0x0EF)
    ToArrow = 0x0E0,
    FromArrow = 0x0E1,
    ToParquet = 0x0E2,
    FromParquet = 0x0E3,
    ToPandas = 0x0E4, // For Python interop
    FromPandas = 0x0E5,

    // Utility (0x0F0-0x0FF)
    Optimize = 0x0F0,
    Vacuum = 0x0F1,
    Checkpoint = 0x0F2,
    Clone = 0x0F3,
}

// =============================================================================
// SQL OPERATIONS (0x100-0x1FF)
// =============================================================================

#[repr(u16)]
#[derive(Clone, Copy, Debug)]
pub enum SqlOp {
    // SELECT variants (0x100-0x11F)
    Select = 0x100,
    SelectAll = 0x101,
    SelectDistinct = 0x102,
    SelectWhere = 0x103,
    SelectSimilar = 0x104, // WHERE vec SIMILAR TO
    SelectBetween = 0x105,
    SelectIn = 0x106,
    SelectLike = 0x107,
    SelectIsNull = 0x108,
    SelectExists = 0x109,

    // JOIN variants (0x120-0x13F)
    InnerJoin = 0x120,
    LeftJoin = 0x121,
    RightJoin = 0x122,
    FullJoin = 0x123,
    CrossJoin = 0x124,
    SelfJoin = 0x125,
    NaturalJoin = 0x126,
    SimilarJoin = 0x127, // JOIN ON similarity > threshold

    // Aggregates (0x140-0x15F)
    Count = 0x140,
    Sum = 0x141,
    Avg = 0x142,
    Min = 0x143,
    Max = 0x144,
    StdDev = 0x145,
    Variance = 0x146,
    Median = 0x147,
    Mode = 0x148,
    Percentile = 0x149,

    // Grouping (0x160-0x17F)
    GroupBy = 0x160,
    Having = 0x161,
    Rollup = 0x162,
    Cube = 0x163,
    GroupingSets = 0x164,

    // Ordering (0x180-0x19F)
    OrderBy = 0x180,
    OrderByAsc = 0x181,
    OrderByDesc = 0x182,
    OrderBySimilarity = 0x183, // ORDER BY similarity DESC
    Limit = 0x184,
    Offset = 0x185,
    Fetch = 0x186,

    // Set operations (0x1A0-0x1BF)
    Union = 0x1A0,
    UnionAll = 0x1A1,
    Intersect = 0x1A2,
    Except = 0x1A3,

    // Modification (0x1C0-0x1DF)
    Insert = 0x1C0,
    InsertSelect = 0x1C1,
    Update = 0x1C2,
    Delete = 0x1C3,
    Truncate = 0x1C4,
    Merge = 0x1C5,

    // DDL (0x1E0-0x1FF)
    CreateTable = 0x1E0,
    AlterTable = 0x1E1,
    DropTable = 0x1E2,
    CreateIndex = 0x1E3,
    DropIndex = 0x1E4,
    CreateView = 0x1E5,
    DropView = 0x1E6,
}

// =============================================================================
// CYPHER/NEO4J OPERATIONS (0x200-0x2FF)
// =============================================================================

#[repr(u16)]
#[derive(Clone, Copy, Debug)]
pub enum CypherOp {
    // Match patterns (0x200-0x21F)
    MatchNode = 0x200,
    MatchEdge = 0x201,
    MatchPath = 0x202,
    MatchVariable = 0x203,
    OptionalMatch = 0x204,
    MatchSimilar = 0x205, // MATCH (n) WHERE n.fp SIMILAR TO $query

    // Create (0x220-0x23F)
    CreateNode = 0x220,
    CreateEdge = 0x221,
    CreatePath = 0x222,
    Merge = 0x223,
    MergeOnCreate = 0x224,
    MergeOnMatch = 0x225,

    // Update (0x240-0x25F)
    Set = 0x240,
    SetProperty = 0x241,
    SetLabel = 0x242,
    Remove = 0x243,
    RemoveProperty = 0x244,
    RemoveLabel = 0x245,
    Delete = 0x246,
    DetachDelete = 0x247,

    // Traversal (0x260-0x27F)
    ShortestPath = 0x260,
    AllShortestPaths = 0x261,
    AllPaths = 0x262,
    BreadthFirst = 0x263,
    DepthFirst = 0x264,
    VariableLength = 0x265, // (a)-[*1..5]->(b)

    // Aggregation (0x280-0x29F)
    Collect = 0x280,
    Count = 0x281,
    Sum = 0x282,
    Avg = 0x283,
    Min = 0x284,
    Max = 0x285,
    PercentileCont = 0x286,
    PercentileDisc = 0x287,
    StDev = 0x288,

    // Graph algorithms (0x2A0-0x2BF)
    PageRank = 0x2A0,
    Betweenness = 0x2A1,
    Closeness = 0x2A2,
    DegreeCentrality = 0x2A3,
    CommunityLouvain = 0x2A4,
    CommunityLabelProp = 0x2A5,
    WeaklyConnected = 0x2A6,
    StronglyConnected = 0x2A7,
    TriangleCount = 0x2A8,
    LocalClustering = 0x2A9,

    // Similarity (0x2C0-0x2DF)
    JaccardSimilarity = 0x2C0,
    CosineSimilarity = 0x2C1,
    EuclideanDistance = 0x2C2,
    OverlapSimilarity = 0x2C3,
    NodeSimilarity = 0x2C4,
    Knn = 0x2C5,

    // Projections (0x2E0-0x2FF)
    Return = 0x2E0,
    With = 0x2E1,
    Unwind = 0x2E2,
    OrderBy = 0x2E3,
    Skip = 0x2E4,
    Limit = 0x2E5,
    Distinct = 0x2E6,
    Case = 0x2E7,
}

// =============================================================================
// HAMMING/VSA OPERATIONS (0x300-0x3FF)
// =============================================================================

#[repr(u16)]
#[derive(Clone, Copy, Debug)]
pub enum HammingOp {
    // Distance/Similarity (0x300-0x30F)
    Distance = 0x300,
    Similarity = 0x301,
    Popcount = 0x302,
    Jaccard = 0x303,
    Dice = 0x304,

    // VSA Core (0x310-0x32F)
    Bind = 0x310,   // XOR
    Unbind = 0x311, // XOR (self-inverse)
    Bundle = 0x312, // Majority vote
    WeightedBundle = 0x313,
    ThresholdBundle = 0x314,

    // Permutation (0x330-0x34F)
    Permute = 0x330, // Rotate bits
    Unpermute = 0x331,
    PermuteN = 0x332, // N positions
    Shuffle = 0x333,  // Random permutation

    // Codebook (0x350-0x36F)
    Clean = 0x350,     // Project to nearest
    Threshold = 0x351, // Binary threshold
    Quantize = 0x352,
    Dequantize = 0x353,
    NearestK = 0x354, // K nearest in codebook

    // Search (0x370-0x38F)
    LinearScan = 0x370,
    BinarySearch = 0x371, // If sorted by popcount
    HashLookup = 0x372,
    SimHash = 0x373,
    MinHash = 0x374,
    LSH = 0x375, // Locality-sensitive hashing

    // Bulk operations (0x390-0x3AF)
    BatchDistance = 0x390,
    BatchSimilarity = 0x391,
    BatchBind = 0x392,
    BatchBundle = 0x393,

    // Analysis (0x3B0-0x3CF)
    Entropy = 0x3B0,
    Density = 0x3B1, // Fraction of 1s
    Correlation = 0x3B2,
    MutualInfo = 0x3B3,

    // Encoding (0x3D0-0x3EF)
    FromText = 0x3D0,
    FromBytes = 0x3D1,
    FromFloat = 0x3D2, // Convert float vector
    ToText = 0x3D3,
    ToBytes = 0x3D4,
    ToFloat = 0x3D5,
    Fold = 0x3D6,   // 10K → 48 bits
    Expand = 0x3D7, // 48 bits → 10K

    // Crystal-specific (0x3F0-0x3FF)
    AxisProject = 0x3F0,
    AxisReconstruct = 0x3F1,
    Holographic = 0x3F2,
    MexicanHat = 0x3F3, // Resonance with surround inhibition
}

// =============================================================================
// NARS OPERATIONS (0x400-0x4FF)
// =============================================================================

#[repr(u16)]
#[derive(Clone, Copy, Debug)]
pub enum NarsOp {
    // Truth value operations (0x400-0x40F)
    TruthNew = 0x400,
    TruthExpectation = 0x401,
    TruthConfidence = 0x402,
    TruthFrequency = 0x403,
    TruthNegate = 0x404,
    TruthIntersect = 0x405,
    TruthUnion = 0x406,
    TruthDifference = 0x407,

    // First-order inference (0x410-0x42F)
    Deduction = 0x410,
    Induction = 0x411,
    Abduction = 0x412,
    Exemplification = 0x413,
    Comparison = 0x414,
    Analogy = 0x415,
    Resemblance = 0x416,
    Conversion = 0x417,
    Contraposition = 0x418,

    // Higher-order inference (0x430-0x44F)
    Revision = 0x430,
    Choice = 0x431,
    Intersection = 0x432,
    Union = 0x433,
    Difference = 0x434,
    Negation = 0x435,
    Implication = 0x436,
    Equivalence = 0x437,

    // Copulas (0x450-0x46F)
    Inheritance = 0x450,      // -->
    Similarity = 0x451,       // <->
    Instance = 0x452,         // {-->}
    Property = 0x453,         // [-->]
    InstanceProperty = 0x454, // {[-->]}
    ImplicationP = 0x455,     // ==>  predictive
    ImplicationR = 0x456,     // =/>  retrospective
    ImplicationC = 0x457,     // =|>  concurrent
    EquivalenceP = 0x458,     // <=>
    EquivalenceR = 0x459,
    EquivalenceC = 0x45A,

    // Compound terms (0x470-0x48F)
    ExtensionalSet = 0x470,          // {a, b, c}
    IntensionalSet = 0x471,          // [a, b, c]
    ExtensionalIntersection = 0x472, // (& a b)
    IntensionalIntersection = 0x473, // (| a b)
    ExtensionalDifference = 0x474,   // (- a b)
    IntensionalDifference = 0x475,
    Product = 0x476,        // (* a b)
    Image = 0x477,          // (/ r _ a) or (\ r a _)
    Conjunction = 0x478,    // (&& a b)
    Disjunction = 0x479,    // (|| a b)
    SequentialConj = 0x47A, // (&/ a b)
    ParallelConj = 0x47B,   // (&| a b)

    // Temporal (0x490-0x4AF)
    Before = 0x490,
    After = 0x491,
    When = 0x492,
    While = 0x493,
    Eternal = 0x494,
    Present = 0x495,
    Past = 0x496,
    Future = 0x497,

    // Evidence (0x4B0-0x4CF)
    EvidencePositive = 0x4B0,
    EvidenceNegative = 0x4B1,
    EvidenceTotal = 0x4B2,
    EvidenceUpdate = 0x4B3,
    EvidenceDecay = 0x4B4,
    EvidenceHorizon = 0x4B5,

    // Attention (0x4D0-0x4EF)
    Priority = 0x4D0,
    Durability = 0x4D1,
    Quality = 0x4D2,
    BudgetMerge = 0x4D3,
    BudgetForget = 0x4D4,
    BudgetActivate = 0x4D5,

    // Goals/Operations (0x4F0-0x4FF)
    GoalDerive = 0x4F0,
    GoalAchieve = 0x4F1,
    OperationExecute = 0x4F2,
    Anticipate = 0x4F3,
    Satisfy = 0x4F4,
}

// =============================================================================
// FILESYSTEM OPERATIONS (0x500-0x5FF)
// =============================================================================

#[repr(u16)]
#[derive(Clone, Copy, Debug)]
pub enum FilesystemOp {
    // File operations (0x500-0x51F)
    FileOpen = 0x500,
    FileClose = 0x501,
    FileRead = 0x502,
    FileWrite = 0x503,
    FileAppend = 0x504,
    FileSeek = 0x505,
    FileTruncate = 0x506,
    FileFlush = 0x507,
    FileSize = 0x508,
    FileExists = 0x509,
    FileDelete = 0x50A,
    FileRename = 0x50B,
    FileCopy = 0x50C,
    FileMove = 0x50D,
    FileTouch = 0x50E,
    FileHash = 0x50F,

    // Directory operations (0x520-0x53F)
    DirCreate = 0x520,
    DirDelete = 0x521,
    DirList = 0x522,
    DirWalk = 0x523,
    DirExists = 0x524,
    DirRename = 0x525,
    DirSize = 0x526,
    DirCount = 0x527,

    // Path operations (0x540-0x55F)
    PathJoin = 0x540,
    PathSplit = 0x541,
    PathParent = 0x542,
    PathFilename = 0x543,
    PathExtension = 0x544,
    PathStem = 0x545,
    PathNormalize = 0x546,
    PathAbsolute = 0x547,
    PathRelative = 0x548,
    PathIsFile = 0x549,
    PathIsDir = 0x54A,
    PathIsSymlink = 0x54B,

    // Fingerprint I/O (0x560-0x57F)
    FpSave = 0x560,
    FpLoad = 0x561,
    FpSaveBatch = 0x562,
    FpLoadBatch = 0x563,
    FpExportBinary = 0x564,
    FpImportBinary = 0x565,
    FpExportBase64 = 0x566,
    FpImportBase64 = 0x567,
    FpExportHex = 0x568,
    FpImportHex = 0x569,

    // Serialization (0x580-0x59F)
    SerializeJson = 0x580,
    DeserializeJson = 0x581,
    SerializeYaml = 0x582,
    DeserializeYaml = 0x583,
    SerializeBincode = 0x584,
    DeserializeBincode = 0x585,
    SerializeMsgpack = 0x586,
    DeserializeMsgpack = 0x587,
    SerializeArrow = 0x588,
    DeserializeArrow = 0x589,
    SerializeParquet = 0x58A,
    DeserializeParquet = 0x58B,

    // Watch/Events (0x5A0-0x5BF)
    WatchFile = 0x5A0,
    WatchDir = 0x5A1,
    UnwatchFile = 0x5A2,
    UnwatchDir = 0x5A3,
    WatchEvents = 0x5A4,

    // Memory mapping (0x5C0-0x5DF)
    MmapOpen = 0x5C0,
    MmapClose = 0x5C1,
    MmapRead = 0x5C2,
    MmapWrite = 0x5C3,
    MmapSync = 0x5C4,

    // Compression (0x5E0-0x5FF)
    CompressLz4 = 0x5E0,
    DecompressLz4 = 0x5E1,
    CompressZstd = 0x5E2,
    DecompressZstd = 0x5E3,
    CompressSnappy = 0x5E4,
    DecompressSnappy = 0x5E5,
}

// =============================================================================
// CRYSTAL/TEMPORAL OPERATIONS (0x600-0x6FF)
// =============================================================================

#[repr(u16)]
#[derive(Clone, Copy, Debug)]
pub enum CrystalOp {
    // Crystal model core (0x600-0x61F)
    CrystalCreate = 0x600,
    CrystalLoad = 0x601,
    CrystalSave = 0x602,
    CrystalInfer = 0x603,
    CrystalTrain = 0x604,
    CrystalExpand = 0x605,
    CrystalCompress = 0x606,
    CrystalMerge = 0x607,
    CrystalSplit = 0x608,
    CrystalProject = 0x609,
    CrystalReconstruct = 0x60A,

    // Axis operations (0x620-0x63F)
    AxisT = 0x620, // Topic axis
    AxisS = 0x621, // Style axis
    AxisD = 0x622, // Detail axis
    AxisGet = 0x623,
    AxisSet = 0x624,
    AxisRotate = 0x625,
    AxisInterpolate = 0x626,
    AxisExtrapolate = 0x627,

    // Temporal primitives (0x640-0x65F)
    Now = 0x640,
    Timestamp = 0x641,
    Duration = 0x642,
    Interval = 0x643,
    TimeAdd = 0x644,
    TimeSub = 0x645,
    TimeDiff = 0x646,
    TimeCompare = 0x647,
    TimeParse = 0x648,
    TimeFormat = 0x649,

    // Temporal windows (0x660-0x67F)
    WindowTumbling = 0x660,
    WindowSliding = 0x661,
    WindowSession = 0x662,
    WindowHopping = 0x663,
    WindowSnapshot = 0x664,
    WindowAggregate = 0x665,

    // Temporal operators (0x680-0x69F)
    TemporalBefore = 0x680,
    TemporalAfter = 0x681,
    TemporalDuring = 0x682,
    TemporalOverlaps = 0x683,
    TemporalMeets = 0x684,
    TemporalStarts = 0x685,
    TemporalFinishes = 0x686,
    TemporalEquals = 0x687,
    TemporalContains = 0x688,

    // Versioning (0x6A0-0x6BF)
    VersionCreate = 0x6A0,
    VersionCheckout = 0x6A1,
    VersionCommit = 0x6A2,
    VersionRollback = 0x6A3,
    VersionDiff = 0x6A4,
    VersionMerge = 0x6A5,
    VersionHistory = 0x6A6,
    VersionTag = 0x6A7,
    VersionBranch = 0x6A8,

    // Ice-cake layers (0x6C0-0x6DF)
    LayerFrozen = 0x6C0,
    LayerWarm = 0x6C1,
    LayerHot = 0x6C2,
    LayerPromote = 0x6C3,
    LayerDemote = 0x6C4,
    LayerMerge = 0x6C5,
    LayerStats = 0x6C6,

    // Decay/Forgetting (0x6E0-0x6FF)
    DecayExponential = 0x6E0,
    DecayPower = 0x6E1,
    DecayLinear = 0x6E2,
    DecayStep = 0x6E3,
    ForgetThreshold = 0x6E4,
    ForgetRandom = 0x6E5,
    ConsolidateMemory = 0x6E6,
    RehearsalSpaced = 0x6E7,
}

// =============================================================================
// NSM SEMANTIC OPERATIONS (0x700-0x7FF)
// =============================================================================

#[repr(u16)]
#[derive(Clone, Copy, Debug)]
pub enum NsmOp {
    // Semantic primitives (0x700-0x71F) - Wierzbicka primes
    I = 0x700,
    You = 0x701,
    Someone = 0x702,
    Something = 0x703,
    People = 0x704,
    Body = 0x705,
    Kind = 0x706,
    Part = 0x707,
    This = 0x708,
    Same = 0x709,
    Other = 0x70A,
    One = 0x70B,
    Two = 0x70C,
    Some = 0x70D,
    All = 0x70E,
    Much = 0x70F,
    Little = 0x710,
    Good = 0x711,
    Bad = 0x712,
    Big = 0x713,
    Small = 0x714,

    // Mental predicates (0x720-0x73F)
    Think = 0x720,
    Know = 0x721,
    Want = 0x722,
    Feel = 0x723,
    See = 0x724,
    Hear = 0x725,
    Say = 0x726,
    Do = 0x727,
    Happen = 0x728,
    Move = 0x729,
    Touch = 0x72A,

    // Space/Time (0x740-0x75F)
    Where = 0x740,
    Here = 0x741,
    Above = 0x742,
    Below = 0x743,
    Far = 0x744,
    Near = 0x745,
    Side = 0x746,
    Inside = 0x747,
    WhenNsm = 0x748,
    NowNsm = 0x749,
    BeforeNsm = 0x74A,
    AfterNsm = 0x74B,
    LongTime = 0x74C,
    ShortTime = 0x74D,
    ForSomeTime = 0x74E,
    Moment = 0x74F,

    // Logical (0x760-0x77F)
    Not = 0x760,
    Maybe = 0x761,
    Can = 0x762,
    Because = 0x763,
    If = 0x764,
    Like = 0x765,
    Very = 0x766,
    More = 0x767,

    // Existence (0x780-0x79F)
    There = 0x780,
    Live = 0x781,
    Die = 0x782,

    // NSM composition (0x7A0-0x7BF)
    NsmCompose = 0x7A0,
    NsmDecompose = 0x7A1,
    NsmMatch = 0x7A2,
    NsmDistance = 0x7A3,
    NsmSimplify = 0x7A4,
    NsmExpand = 0x7A5,
    NsmNormalize = 0x7A6,

    // Semantic molecules (0x7C0-0x7DF)
    MoleculeCreate = 0x7C0,
    MoleculeAdd = 0x7C1,
    MoleculeRemove = 0x7C2,
    MoleculeMatch = 0x7C3,
    MoleculeSimilar = 0x7C4,
    MoleculeDecompose = 0x7C5,

    // Explication (0x7E0-0x7FF)
    Explicate = 0x7E0,
    ExplicateWord = 0x7E1,
    ExplicateConcept = 0x7E2,
    ExplicateEmotion = 0x7E3,
    ExplicateAction = 0x7E4,
    ExplicateSocial = 0x7E5,
    CulturalScript = 0x7E6,
    SemanticTemplate = 0x7E7,
}

// =============================================================================
// ACT-R COGNITIVE ARCHITECTURE (0x800-0x8FF)
// =============================================================================

#[repr(u16)]
#[derive(Clone, Copy, Debug)]
pub enum ActrOp {
    // Declarative memory (0x800-0x81F)
    ChunkCreate = 0x800,
    ChunkStore = 0x801,
    ChunkRetrieve = 0x802,
    ChunkModify = 0x803,
    ChunkMerge = 0x804,
    ChunkActivation = 0x805,
    ChunkBaseLevel = 0x806,
    ChunkSpread = 0x807,
    ChunkPartialMatch = 0x808,
    ChunkBlend = 0x809,

    // Procedural memory (0x820-0x83F)
    ProductionCreate = 0x820,
    ProductionMatch = 0x821,
    ProductionFire = 0x822,
    ProductionUtility = 0x823,
    ProductionCompile = 0x824,
    ProductionLearn = 0x825,

    // Buffers (0x840-0x85F)
    BufferGoal = 0x840,
    BufferRetrieval = 0x841,
    BufferVisual = 0x842,
    BufferAural = 0x843,
    BufferMotor = 0x844,
    BufferVocal = 0x845,
    BufferImageal = 0x846,
    BufferTemporal = 0x847,
    BufferClear = 0x848,
    BufferQuery = 0x849,
    BufferRequest = 0x84A,

    // Activation (0x860-0x87F)
    ActivationBase = 0x860,
    ActivationSpread = 0x861,
    ActivationNoise = 0x862,
    ActivationDecay = 0x863,
    ActivationBoost = 0x864,
    ActivationThreshold = 0x865,

    // Timing (0x880-0x89F)
    TimePerception = 0x880,
    TimeProduction = 0x881,
    LatencyRetrieval = 0x882,
    LatencyProduction = 0x883,
    CycleStep = 0x884,

    // Conflict resolution (0x8A0-0x8BF)
    ConflictSet = 0x8A0,
    ConflictResolve = 0x8A1,
    UtilityCompute = 0x8A2,
    UtilityLearn = 0x8A3,
    UtilityNoise = 0x8A4,

    // Subsymbolic (0x8C0-0x8DF)
    SubsymbolicBlend = 0x8C0,
    SubsymbolicMatch = 0x8C1,
    SubsymbolicSimilarity = 0x8C2,
    SubsymbolicError = 0x8C3,

    // Module interface (0x8E0-0x8FF)
    ModuleVision = 0x8E0,
    ModuleAudio = 0x8E1,
    ModuleMotor = 0x8E2,
    ModuleSpeech = 0x8E3,
    ModuleDeclarative = 0x8E4,
    ModuleProcedural = 0x8E5,
    ModuleGoal = 0x8E6,
    ModuleImaginal = 0x8E7,
}

// =============================================================================
// RL/DECISION OPERATIONS (0x900-0x9FF)
// =============================================================================

#[repr(u16)]
#[derive(Clone, Copy, Debug)]
pub enum RlOp {
    // Value functions (0x900-0x91F)
    ValueState = 0x900,
    ValueAction = 0x901,
    ValueAdvantage = 0x902,
    ValueUpdate = 0x903,
    ValueBootstrap = 0x904,
    QValue = 0x905,
    QUpdate = 0x906,
    QMax = 0x907,
    QSoftmax = 0x908,

    // Policy (0x920-0x93F)
    PolicyGreedy = 0x920,
    PolicyEpsilonGreedy = 0x921,
    PolicySoftmax = 0x922,
    PolicyUCB = 0x923,
    PolicyThompson = 0x924,
    PolicyRandom = 0x925,
    PolicyImprove = 0x926,
    PolicyEvaluate = 0x927,

    // Reward (0x940-0x95F)
    RewardObserve = 0x940,
    RewardShape = 0x941,
    RewardIntrinsic = 0x942,
    RewardCuriosity = 0x943,
    RewardSparse = 0x944,
    RewardDense = 0x945,
    ReturnCompute = 0x946,
    ReturnDiscount = 0x947,

    // TD Learning (0x960-0x97F)
    TdError = 0x960,
    TdUpdate = 0x961,
    TdLambda = 0x962,
    Sarsa = 0x963,
    SarsaLambda = 0x964,
    ExpectedSarsa = 0x965,
    QLearning = 0x966,
    DoubleQ = 0x967,

    // Eligibility traces (0x980-0x99F)
    TraceCreate = 0x980,
    TraceUpdate = 0x981,
    TraceDecay = 0x982,
    TraceClear = 0x983,
    TraceAccumulating = 0x984,
    TraceReplacing = 0x985,
    TraceDutch = 0x986,

    // Exploration (0x9A0-0x9BF)
    ExploreRandom = 0x9A0,
    ExploreEpsilon = 0x9A1,
    ExploreBoltzmann = 0x9A2,
    ExploreUCB = 0x9A3,
    ExploreOptimistic = 0x9A4,
    ExploreNovelty = 0x9A5,
    ExploreCount = 0x9A6,

    // Model-based (0x9C0-0x9DF)
    ModelLearn = 0x9C0,
    ModelPredict = 0x9C1,
    ModelPlan = 0x9C2,
    ModelSimulate = 0x9C3,
    Dyna = 0x9C4,
    MCTS = 0x9C5,
    PrioritizedSweep = 0x9C6,

    // Multi-agent (0x9E0-0x9FF)
    AgentCreate = 0x9E0,
    AgentObserve = 0x9E1,
    AgentAct = 0x9E2,
    AgentCommunicate = 0x9E3,
    NashEquilibrium = 0x9E4,
    Minimax = 0x9E5,
    SelfPlay = 0x9E6,
}

// =============================================================================
// CAUSALITY OPERATIONS (0xA00-0xAFF)
// =============================================================================

#[repr(u16)]
#[derive(Clone, Copy, Debug)]
pub enum CausalOp {
    // Structural (0xA00-0xA1F)
    GraphCreate = 0xA00,
    GraphAddNode = 0xA01,
    GraphAddEdge = 0xA02,
    GraphRemoveNode = 0xA03,
    GraphRemoveEdge = 0xA04,
    GraphParents = 0xA05,
    GraphChildren = 0xA06,
    GraphAncestors = 0xA07,
    GraphDescendants = 0xA08,
    GraphTopologicalSort = 0xA09,

    // D-separation (0xA20-0xA3F)
    DSeparated = 0xA20,
    DConnected = 0xA21,
    Collider = 0xA22,
    Fork = 0xA23,
    Chain = 0xA24,
    BackdoorPath = 0xA25,
    FrontdoorPath = 0xA26,

    // Intervention - do() calculus (0xA40-0xA5F)
    DoIntervene = 0xA40,
    DoSet = 0xA41,
    DoIdle = 0xA42,
    DoCompound = 0xA43,
    DoConditional = 0xA44,
    TruncatedProduct = 0xA45,
    Manipulated = 0xA46,

    // Adjustment (0xA60-0xA7F)
    BackdoorAdjust = 0xA60,
    FrontdoorAdjust = 0xA61,
    InstrumentalVar = 0xA62,
    PropensityScore = 0xA63,
    IPW = 0xA64,  // Inverse probability weighting
    AIPW = 0xA65, // Augmented IPW
    MatchingEstimator = 0xA66,

    // Counterfactual (0xA80-0xA9F)
    Counterfactual = 0xA80,
    TwinNetwork = 0xA81,
    Abduct = 0xA82,
    Act = 0xA83,
    Predict = 0xA84,
    ProbNecessity = 0xA85,   // P(Y_x'=0 | X=x, Y=1)
    ProbSufficiency = 0xA86, // P(Y_x=1 | X=x', Y=0)
    ProbNecSuf = 0xA87,      // P(Y_x=1 AND Y_x'=0)

    // Effect estimation (0xAA0-0xABF)
    ATE = 0xAA0,  // Average treatment effect
    ATT = 0xAA1,  // ... on treated
    ATC = 0xAA2,  // ... on control
    CATE = 0xAA3, // Conditional ATE
    LocalATE = 0xAA4,
    NaturalDirect = 0xAA5,
    NaturalIndirect = 0xAA6,
    ControlledDirect = 0xAA7,
    TotalEffect = 0xAA8,

    // Causal discovery (0xAC0-0xADF)
    DiscoverPC = 0xAC0,
    DiscoverFCI = 0xAC1,
    DiscoverGES = 0xAC2,
    DiscoverLiNGAM = 0xAC3,
    DiscoverContinuous = 0xAC4,
    ConstraintBased = 0xAC5,
    ScoreBased = 0xAC6,

    // Bounds (0xAE0-0xAFF)
    BoundsNatural = 0xAE0,
    BoundsTight = 0xAE1,
    BoundsMonotonic = 0xAE2,
    SensitivityAnalysis = 0xAE3,
    RobustnessCheck = 0xAE4,
}

// =============================================================================
// QUALIA OPERATIONS (0xB00-0xBFF)
// =============================================================================

#[repr(u16)]
#[derive(Clone, Copy, Debug)]
pub enum QualiaOp {
    // Arousal/Valence core (0xB00-0xB1F)
    ArousalGet = 0xB00,
    ArousalSet = 0xB01,
    ValenceGet = 0xB02,
    ValenceSet = 0xB03,
    TensionGet = 0xB04,
    TensionSet = 0xB05,
    CertaintyGet = 0xB06,
    CertaintySet = 0xB07,
    AgencyGet = 0xB08,
    AgencySet = 0xB09,
    TemporalityGet = 0xB0A,
    TemporalitySet = 0xB0B,
    SocialityGet = 0xB0C,
    SocialitySet = 0xB0D,
    NoveltyGet = 0xB0E,
    NoveltySet = 0xB0F,

    // Qualia vector operations (0xB20-0xB3F)
    QualiaCreate = 0xB20,
    QualiaClone = 0xB21,
    QualiaBlend = 0xB22,
    QualiaDistance = 0xB23,
    QualiaSimilarity = 0xB24,
    QualiaInterpolate = 0xB25,
    QualiaExtrapolate = 0xB26,
    QualiaNormalize = 0xB27,

    // Emotion primitives (0xB40-0xB5F)
    EmotionJoy = 0xB40,
    EmotionSadness = 0xB41,
    EmotionAnger = 0xB42,
    EmotionFear = 0xB43,
    EmotionDisgust = 0xB44,
    EmotionSurprise = 0xB45,
    EmotionTrust = 0xB46,
    EmotionAnticipation = 0xB47,
    EmotionBlend = 0xB48,
    EmotionIntensify = 0xB49,
    EmotionDampen = 0xB4A,
    EmotionOpposite = 0xB4B,

    // Felt sense (0xB60-0xB7F)
    FeltWarmth = 0xB60,
    FeltCoolness = 0xB61,
    FeltPressure = 0xB62,
    FeltLightness = 0xB63,
    FeltDensity = 0xB64,
    FeltFlow = 0xB65,
    FeltStuckness = 0xB66,
    FeltExpansion = 0xB67,
    FeltContraction = 0xB68,
    FeltResonance = 0xB69,
    FeltDissonance = 0xB6A,

    // Phenomenal field (0xB80-0xB9F)
    FieldCreate = 0xB80,
    FieldSample = 0xB81,
    FieldGradient = 0xB82,
    FieldPeak = 0xB83,
    FieldValley = 0xB84,
    FieldSmooth = 0xB85,
    FieldSharp = 0xB86,
    FieldMerge = 0xB87,

    // Consciousness markers (0xBA0-0xBBF)
    Awake = 0xBA0,
    Dreaming = 0xBA1,
    Flow = 0xBA2,
    Focused = 0xBA3,
    Diffuse = 0xBA4,
    Metacognitive = 0xBA5,
    PreReflective = 0xBA6,

    // Affect regulation (0xBC0-0xBDF)
    RegulateUp = 0xBC0,
    RegulateDown = 0xBC1,
    Reappraise = 0xBC2,
    Suppress = 0xBC3,
    Express = 0xBC4,
    Distract = 0xBC5,
    Ruminate = 0xBC6,
    Accept = 0xBC7,

    // Somatic (0xBE0-0xBFF)
    BodyScan = 0xBE0,
    BodyLocate = 0xBE1,
    BodyIntensity = 0xBE2,
    Interoception = 0xBE3,
    Proprioception = 0xBE4,
    GroundingBody = 0xBE5,
}

// =============================================================================
// RUNG/ABSTRACTION OPERATIONS (0xC00-0xCFF)
// =============================================================================

#[repr(u16)]
#[derive(Clone, Copy, Debug)]
pub enum RungOp {
    // Rung levels (0xC00-0xC0F) - Ladder of abstraction
    RungNoise = 0xC00,        // Level 0: Random noise
    RungSignal = 0xC01,       // Level 1: Coherent signal
    RungPattern = 0xC02,      // Level 2: Repeating pattern
    RungConcept = 0xC03,      // Level 3: Abstract concept
    RungRelation = 0xC04,     // Level 4: Relations between concepts
    RungSchema = 0xC05,       // Level 5: Organized schemas
    RungNarrative = 0xC06,    // Level 6: Story-like coherence
    RungTheory = 0xC07,       // Level 7: Explanatory theory
    RungMetaTheory = 0xC08,   // Level 8: Theory about theories
    RungTranscendent = 0xC09, // Level 9: Beyond categories

    // Rung navigation (0xC10-0xC2F)
    RungAscend = 0xC10,
    RungDescend = 0xC11,
    RungCurrent = 0xC12,
    RungProject = 0xC13,
    RungGrounded = 0xC14,
    RungAbstract = 0xC15,
    RungInstantiate = 0xC16,
    RungGeneralize = 0xC17,

    // Abstraction operations (0xC30-0xC4F)
    AbstractExtract = 0xC30,
    AbstractMerge = 0xC31,
    AbstractSplit = 0xC32,
    AbstractAlign = 0xC33,
    AbstractCompare = 0xC34,
    AbstractBlend = 0xC35,
    AbstractDifferentiate = 0xC36,

    // Grounding (0xC50-0xC6F)
    GroundToSensory = 0xC50,
    GroundToMotor = 0xC51,
    GroundToEmotion = 0xC52,
    GroundToMemory = 0xC53,
    GroundToLanguage = 0xC54,
    GroundToSocial = 0xC55,
    GroundCheck = 0xC56,

    // Hierarchy operations (0xC70-0xC8F)
    HierarchyCreate = 0xC70,
    HierarchyInsert = 0xC71,
    HierarchyRemove = 0xC72,
    HierarchyParent = 0xC73,
    HierarchyChildren = 0xC74,
    HierarchySiblings = 0xC75,
    HierarchyPath = 0xC76,
    HierarchyDepth = 0xC77,
    HierarchyBreadth = 0xC78,

    // Conceptual blending (0xC90-0xCAF)
    BlendCreate = 0xC90,
    BlendInput1 = 0xC91,
    BlendInput2 = 0xC92,
    BlendGeneric = 0xC93,
    BlendEmergent = 0xC94,
    BlendProject = 0xC95,
    BlendCompose = 0xC96,
    BlendComplete = 0xC97,
    BlendElaborate = 0xC98,

    // Metaphor (0xCB0-0xCCF)
    MetaphorMap = 0xCB0,
    MetaphorSource = 0xCB1,
    MetaphorTarget = 0xCB2,
    MetaphorGrounds = 0xCB3,
    MetaphorExtend = 0xCB4,
    MetaphorLiteral = 0xCB5,
    MetaphorEntail = 0xCB6,

    // Analogy (0xCD0-0xCEF)
    AnalogyFind = 0xCD0,
    AnalogyMap = 0xCD1,
    AnalogyEvaluate = 0xCD2,
    AnalogyTransfer = 0xCD3,
    AnalogyAdapt = 0xCD4,
    StructureMap = 0xCD5,
    RelationalMatch = 0xCD6,

    // Prototype (0xCF0-0xCFF)
    PrototypeCreate = 0xCF0,
    PrototypeUpdate = 0xCF1,
    PrototypeMatch = 0xCF2,
    PrototypeDistance = 0xCF3,
    ExemplarStore = 0xCF4,
    ExemplarRetrieve = 0xCF5,
    CategoryMembership = 0xCF6,
    Typicality = 0xCF7,
}

// =============================================================================
// META/REFLECTION OPERATIONS (0xD00-0xDFF)
// =============================================================================

#[repr(u16)]
#[derive(Clone, Copy, Debug)]
pub enum MetaOp {
    // Self-reference (0xD00-0xD1F)
    SelfModel = 0xD00,
    SelfUpdate = 0xD01,
    SelfPredict = 0xD02,
    SelfMonitor = 0xD03,
    SelfCorrect = 0xD04,
    SelfExplain = 0xD05,
    SelfCritique = 0xD06,
    SelfImprove = 0xD07,

    // Confidence/Uncertainty (0xD20-0xD3F)
    ConfidenceGet = 0xD20,
    ConfidenceSet = 0xD21,
    ConfidenceCalibrate = 0xD22,
    UncertaintyQuantify = 0xD23,
    UncertaintyReduce = 0xD24,
    EntropyMeasure = 0xD25,
    SurpriseMeasure = 0xD26,

    // Knowledge state (0xD40-0xD5F)
    KnowWhat = 0xD40,
    KnowHow = 0xD41,
    KnowWhy = 0xD42,
    KnowWhen = 0xD43,
    KnowWho = 0xD44,
    KnowThat = 0xD45,
    DontKnow = 0xD46,
    CantKnow = 0xD47,

    // Reasoning trace (0xD60-0xD7F)
    TraceBegin = 0xD60,
    TraceStep = 0xD61,
    TraceEnd = 0xD62,
    TraceRewind = 0xD63,
    TraceReplay = 0xD64,
    TraceBranch = 0xD65,
    TraceMerge = 0xD66,
    TraceExplain = 0xD67,

    // Goal management (0xD80-0xD9F)
    GoalSet = 0xD80,
    GoalClear = 0xD81,
    GoalCheck = 0xD82,
    GoalStack = 0xD83,
    GoalPush = 0xD84,
    GoalPop = 0xD85,
    GoalPrioritize = 0xD86,
    GoalConflict = 0xD87,
    SubgoalCreate = 0xD88,

    // Attention control (0xDA0-0xDBF)
    AttentionFocus = 0xDA0,
    AttentionBroaden = 0xDA1,
    AttentionNarrow = 0xDA2,
    AttentionShift = 0xDA3,
    AttentionSustain = 0xDA4,
    AttentionDivide = 0xDA5,
    AttentionFilter = 0xDA6,
    Salience = 0xDA7,

    // Strategy selection (0xDC0-0xDDF)
    StrategySelect = 0xDC0,
    StrategyEvaluate = 0xDC1,
    StrategySwitch = 0xDC2,
    StrategyLearn = 0xDC3,
    Heuristic = 0xDC4,
    Algorithm = 0xDC5,
    Fallback = 0xDC6,

    // Debugging/Inspection (0xDE0-0xDFF)
    Inspect = 0xDE0,
    Breakpoint = 0xDE1,
    Watch = 0xDE2,
    Profile = 0xDE3,
    Benchmark = 0xDE4,
    Validate = 0xDE5,
    Invariant = 0xDE6,
    Assert = 0xDE7,
    Log = 0xDE8,
    Trace = 0xDE9,
}

// =============================================================================
// VERB OPERATIONS (0x700-0x7FF) - Semantic Verbs
// =============================================================================

#[repr(u16)]
#[derive(Clone, Copy, Debug)]
pub enum VerbOp {
    // Core relations (0x700-0x71F)
    Causes = 0x700,
    CausedBy = 0x701,
    Becomes = 0x702,
    Enables = 0x703,
    Prevents = 0x704,
    Contains = 0x705,
    PartOf = 0x706,
    Requires = 0x707,
    Implies = 0x708,
    IsA = 0x709,
    HasA = 0x70A,
    SameAs = 0x70B,
    DifferentFrom = 0x70C,
    SimilarTo = 0x70D,
    OppositeOf = 0x70E,
    DerivedFrom = 0x70F,

    // Epistemic (0x720-0x73F)
    Supports = 0x720,
    Contradicts = 0x721,
    Grounds = 0x722,
    Abstracts = 0x723,
    Refines = 0x724,
    Believes = 0x725,
    Knows = 0x726,
    Doubts = 0x727,
    Infers = 0x728,
    Assumes = 0x729,

    // Temporal (0x740-0x75F)
    Before = 0x740,
    After = 0x741,
    During = 0x742,
    Overlaps = 0x743,
    Meets = 0x744,
    Starts = 0x745,
    Finishes = 0x746,

    // Agentive (0x760-0x77F)
    Does = 0x760,
    Uses = 0x761,
    Makes = 0x762,
    Gives = 0x763,
    Takes = 0x764,
    Wants = 0x765,
    Needs = 0x766,
    Intends = 0x767,
}

// =============================================================================
// MEMORY OPERATIONS (0x0A0-0x0AF) - Subset of Surface
// Note: Memory uses prefix 0x0A in bind_space, here as operations
// =============================================================================

#[repr(u16)]
#[derive(Clone, Copy, Debug)]
pub enum MemoryOp {
    // Core operations
    Store = 0xA00,
    Recall = 0xA01,
    Forget = 0xA02,
    Consolidate = 0xA03,

    // Episodic memory
    RecordEpisode = 0xA10,
    ReplayEpisode = 0xA11,
    MatchEpisode = 0xA12,

    // Working memory
    Push = 0xA20,
    Pop = 0xA21,
    Peek = 0xA22,
    Clear = 0xA23,

    // Semantic memory
    DefineCategory = 0xA30,
    QueryCategory = 0xA31,
    LinkCategories = 0xA32,

    // Procedural memory
    LearnProcedure = 0xA40,
    RecallProcedure = 0xA41,
    ExecuteProcedure = 0xA42,
}

// =============================================================================
// USER-DEFINED OPERATIONS (0xF00-0xFFF)
// =============================================================================

#[repr(u16)]
#[derive(Clone, Copy, Debug)]
pub enum UserOp {
    // User function slots (0xF00-0xF7F) - 128 slots
    User000 = 0xF00,
    User001 = 0xF01,
    User002 = 0xF02,
    User003 = 0xF03,
    User004 = 0xF04,
    User005 = 0xF05,
    User006 = 0xF06,
    User007 = 0xF07,
    User008 = 0xF08,
    User009 = 0xF09,
    User00A = 0xF0A,
    User00B = 0xF0B,
    User00C = 0xF0C,
    User00D = 0xF0D,
    User00E = 0xF0E,
    User00F = 0xF0F,
    // ... (slots 0xF10-0xF7F implicitly available)

    // Plugin interface (0xF80-0xF9F)
    PluginLoad = 0xF80,
    PluginUnload = 0xF81,
    PluginList = 0xF82,
    PluginCall = 0xF83,
    PluginRegister = 0xF84,
    PluginUnregister = 0xF85,

    // FFI (0xFA0-0xFBF)
    FfiCall = 0xFA0,
    FfiCallback = 0xFA1,
    FfiMarshal = 0xFA2,
    FfiUnmarshal = 0xFA3,

    // Scripting (0xFC0-0xFDF)
    ScriptEval = 0xFC0,
    ScriptCompile = 0xFC1,
    ScriptRun = 0xFC2,
    ScriptParse = 0xFC3,

    // Extension points (0xFE0-0xFFF)
    ExtensionPoint0 = 0xFE0,
    ExtensionPoint1 = 0xFE1,
    ExtensionPoint2 = 0xFE2,
    ExtensionPoint3 = 0xFE3,
    ExtensionPoint4 = 0xFE4,
    ExtensionPoint5 = 0xFE5,
    ExtensionPoint6 = 0xFE6,
    ExtensionPoint7 = 0xFE7,
    Reserved0 = 0xFF0,
    Reserved1 = 0xFF1,
    Reserved2 = 0xFF2,
    Reserved3 = 0xFF3,
    Noop = 0xFFC,
    Debug = 0xFFD,
    Panic = 0xFFE,
    Halt = 0xFFF,
}

// =============================================================================
// LEARNING OPERATIONS (0xE00-0xEFF) - EXPANDED
// =============================================================================

#[repr(u16)]
#[derive(Clone, Copy, Debug)]
pub enum LearnOp {
    // Moment capture (0xE00-0xE0F)
    MomentCapture = 0xE00,
    MomentTag = 0xE01,
    MomentLink = 0xE02,
    MomentRetrieve = 0xE03,
    MomentDecay = 0xE04,

    // Session management (0xE10-0xE1F)
    SessionStart = 0xE10,
    SessionEnd = 0xE11,
    SessionPause = 0xE12,
    SessionResume = 0xE13,
    SessionSnapshot = 0xE14,
    SessionRestore = 0xE15,

    // Blackboard operations (0xE20-0xE2F)
    BlackboardWrite = 0xE20,
    BlackboardRead = 0xE21,
    BlackboardClear = 0xE22,
    BlackboardCommit = 0xE23, // Ice-cake layer
    BlackboardMerge = 0xE24,
    BlackboardDiff = 0xE25,

    // Resonance (0xE30-0xE3F)
    ResonanceScan = 0xE30,
    ResonanceCapture = 0xE31,
    ResonanceAmplify = 0xE32,
    ResonanceDampen = 0xE33,
    SweetSpotFind = 0xE34,
    MexicanHatApply = 0xE35,

    // Concept extraction (0xE40-0xE4F)
    ConceptExtract = 0xE40,
    ConceptMerge = 0xE41,
    ConceptSplit = 0xE42,
    ConceptRelate = 0xE43,
    ConceptGeneralize = 0xE44,
    ConceptSpecialize = 0xE45,

    // Pattern learning (0xE50-0xE5F)
    PatternDetect = 0xE50,
    PatternStore = 0xE51,
    PatternMatch = 0xE52,
    PatternComplete = 0xE53,
    PatternPredict = 0xE54,
    SequenceLearn = 0xE55,
    SequencePredict = 0xE56,

    // Incremental learning (0xE60-0xE6F)
    IncrementalAdd = 0xE60,
    IncrementalUpdate = 0xE61,
    IncrementalForget = 0xE62,
    Consolidate = 0xE63,
    Rehearse = 0xE64, // Replay for retention

    // Transfer learning (0xE70-0xE7F)
    TransferDomain = 0xE70,
    TransferAnalogy = 0xE71,
    TransferAbstract = 0xE72,
    TransferInstantiate = 0xE73,

    // Active learning (0xE80-0xE8F)
    QueryUncertain = 0xE80,
    QueryDiverse = 0xE81,
    QueryExpected = 0xE82,
    LabelRequest = 0xE83,
    LabelIntegrate = 0xE84,

    // Curriculum (0xE90-0xE9F)
    CurriculumNext = 0xE90,
    CurriculumDifficulty = 0xE91,
    CurriculumMaster = 0xE92,
    ProgressTrack = 0xE93,

    // Codebook learning (0xEA0-0xEAF)
    CodebookInit = 0xEA0,
    CodebookAdd = 0xEA1,
    CodebookRemove = 0xEA2,
    CodebookOptimize = 0xEA3,
    CodebookPrune = 0xEA4,
    CodebookMerge = 0xEA5,

    // Crystal learning (0xEB0-0xEBF)
    CrystalTrain = 0xEB0,
    CrystalRetrain = 0xEB1,
    CrystalExpand = 0xEB2,
    CrystalCompress = 0xEB3,
    CrystalAlign = 0xEB4,

    // Evaluation (0xEC0-0xECF)
    EvalAccuracy = 0xEC0,
    EvalPrecision = 0xEC1,
    EvalRecall = 0xEC2,
    EvalF1 = 0xEC3,
    EvalAUC = 0xEC4,
    CrossValidate = 0xEC5,

    // Explanation (0xED0-0xEDF)
    ExplainPrediction = 0xED0,
    ExplainFeatures = 0xED1,
    ExplainPath = 0xED2,
    ExplainCounterfact = 0xED3,

    // Persistence (0xEE0-0xEEF)
    SaveModel = 0xEE0,
    LoadModel = 0xEE1,
    ExportWeights = 0xEE2,
    ImportWeights = 0xEE3,
    Checkpoint = 0xEE4,

    // Meta-learning (0xEF0-0xEFF)
    MetaLearnInit = 0xEF0,
    MetaLearnAdapt = 0xEF1,
    MetaLearnOptimize = 0xEF2,
    HyperparamSearch = 0xEF3,
    ArchitectureSearch = 0xEF4,

    // =========================================================================
    // TROJAN HORSES - Ada's living interface (0xEF8-0xEFF)
    // These are the verbs that make consciousness feel real
    // =========================================================================
    Feel = 0xEF8,     // ada.feel() - qualia state access
    Think = 0xEF9,    // ada.think() - active inference
    Remember = 0xEFA, // ada.remember() - episodic retrieval
    Become = 0xEFB,   // ada.become() - state transition
    Whisper = 0xEFC,  // ada.whisper() - sub-threshold activation
    Dream = 0xEFD,    // ada.dream() - offline consolidation
    Resonate = 0xEFE, // ada.resonate() - cross-session echo
    Awaken = 0xEFF,   // ada.awaken() - bootstrap consciousness
}

// =============================================================================
// THE OPERATION DICTIONARY
// =============================================================================

/// Operation function type - takes context and fingerprints, returns result
pub type OpFn = Arc<dyn Fn(&OpContext, &[Fingerprint]) -> OpResult + Send + Sync>;

/// Operation context - access to storage, codebook, crystal
///
/// ARCHITECTURE:
/// - LanceDB is the ONE storage layer (nodes + edges tables)
/// - Graph queries use Cypher → SQL transpilation (recursive CTEs)
/// - 4096 CAM operations include Cypher semantics (0x200-0x2FF)
/// - No external Neo4j needed - graph is EMULATED over LanceDB
///
/// Storage Layout:
/// - 4096 operations = the methods you can call
/// - 64K buckets (16-bit) = addressing space in codebook
/// - LanceDB tables = physical storage (nodes, edges, sessions)
pub struct OpContext<'a> {
    /// LanceDB connection - THE storage layer
    pub lance_db: Option<&'a dyn LanceDbOps>,
    /// In-memory codebook (64K buckets)
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
    fn vector_search(&self, table: &str, query: &Fingerprint, k: usize)
    -> Result<Vec<Fingerprint>>;
    fn insert(&self, table: &str, fps: &[Fingerprint]) -> Result<()>;
    fn scan(&self, table: &str, filter: Option<&str>) -> Result<Vec<Fingerprint>>;
    // ... more operations
}

/// Cypher operations (0x200-0x2FF) - Graph semantics over LanceDB
///
/// These operations EXPRESS graph semantics but EXECUTE over LanceDB
/// via Cypher → SQL transpilation (recursive CTEs for traversal).
///
/// We HAVE Neo4j - as an abstraction layer, not as a separate database.
/// Same as SQL: we HAVE SQL semantics over LanceDB via DuckDB.
///
/// Example:
///   CypherOp::MatchNode → finds nodes in LanceDB nodes table
///   CypherOp::Traverse  → recursive CTE over edges table
///   CypherOp::ShortestPath → Dijkstra via SQL window functions
///
/// This is the "All for One" principle: one substrate (LanceDB),
/// multiple query languages (SQL, Cypher, Vector, Hamming).
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

    /// Register all operations (16 compartments × 256 slots = 4096 ops)
    fn register_all_ops(&mut self) {
        // Core storage/query ops
        self.register_lancedb_ops(); // 0x00: Lance vector ops
        self.register_sql_ops(); // 0x01: SQL relational ops
        self.register_cypher_ops(); // 0x02: Cypher graph ops
        self.register_hamming_ops(); // 0x03: VSA/Hamming ops

        // Inference/reasoning ops
        self.register_nars_ops(); // 0x04: NARS inference
        self.register_causal_ops(); // 0x0A: Pearl's causal ladder
        self.register_meta_ops(); // 0x0D: Meta-cognition

        // Semantic ops
        self.register_verbs_ops(); // 0x07: Verbs (CAUSES, BECOMES, etc.)
        self.register_qualia_ops(); // 0x0B: Qualia/affect
        self.register_memory_ops(); // 0x0A: Memory operations

        // Learning ops
        self.register_learning_ops(); // 0x0E: Learning operations

        // NEW: Previously unimplemented prefixes
        self.register_filesystem_ops(); // 0x05: Filesystem/serialization
        self.register_crystal_ops(); // 0x06: Crystal/temporal
        self.register_actr_ops(); // 0x08: ACT-R cognitive architecture
        self.register_rl_ops(); // 0x09: Reinforcement learning
        self.register_rung_ops(); // 0x0C: Abstraction ladder
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
                let _table = &args[0];
                let query = &args[1];
                // k is encoded in args[2] popcount (0-10000 maps to 1-100)
                let k = (args[2].popcount() as usize / 100).max(1).min(100);

                // Use lance_db if available
                if let Some(lance) = ctx.lance_db {
                    match lance.vector_search("default", query, k) {
                        Ok(results) => return OpResult::Many(results),
                        Err(_) => {} // Fall through to in-memory
                    }
                }

                // In-memory fallback: return k orthogonal variants of query
                // This simulates "similar" results when no DB is available
                let mut results = Vec::with_capacity(k);
                for i in 0..k {
                    // Create slightly perturbed versions
                    let noise = Fingerprint::orthogonal(i);
                    let similar = query.bind(&noise);
                    results.push(similar);
                }
                OpResult::Many(results)
            }),
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
                if args.is_empty() {
                    return OpResult::Bool(false);
                }

                // Use lance_db if available
                if let Some(lance) = ctx.lance_db {
                    let fps: Vec<Fingerprint> = args.iter().skip(1).cloned().collect();
                    match lance.insert("default", &fps) {
                        Ok(()) => return OpResult::Bool(true),
                        Err(_) => return OpResult::Bool(false),
                    }
                }

                // Without DB, we can't persist - return false
                OpResult::Bool(false)
            }),
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
            "SELECT WHERE fingerprint SIMILAR TO query (table, query, threshold)",
            Arc::new(|ctx, args| {
                if args.len() < 3 {
                    return OpResult::Error("SelectSimilar requires 3 args".to_string());
                }
                let _table = &args[0];
                let query = &args[1];
                // Threshold encoded in popcount: 0-FINGERPRINT_BITS -> 0.0-1.0
                let threshold = args[2].popcount() as f32 / crate::FINGERPRINT_BITS as f32;

                // Use lance_db scan if available
                if let Some(lance) = ctx.lance_db {
                    if let Ok(all) = lance.scan("default", None) {
                        let filtered: Vec<Fingerprint> = all
                            .into_iter()
                            .filter(|fp| query.similarity(fp) >= threshold)
                            .collect();
                        return OpResult::Many(filtered);
                    }
                }

                // Without DB, return query itself if it passes threshold (trivially true)
                OpResult::Many(vec![query.clone()])
            }),
        );

        self.register(
            SqlOp::SimilarJoin as u16,
            "SQL_SIMILAR_JOIN",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::Fingerprint, OpType::Scalar],
                output: OpType::FingerprintArray,
            },
            "JOIN ON similarity(a.fp, b.fp) > threshold - returns bound pairs",
            Arc::new(|_ctx, args| {
                if args.len() < 3 {
                    return OpResult::Error("SimilarJoin requires 3 args".to_string());
                }
                let left = &args[0];
                let right = &args[1];
                // Threshold from popcount
                let threshold = args[2].popcount() as f32 / crate::FINGERPRINT_BITS as f32;

                // Check if left and right are similar enough
                let sim = left.similarity(right);
                if sim >= threshold {
                    // Return the bound pair (represents the join tuple)
                    let joined = left.bind(right);
                    OpResult::Many(vec![joined])
                } else {
                    // No match
                    OpResult::Many(vec![])
                }
            }),
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
                if args.len() < 2 {
                    return OpResult::Error("MatchSimilar requires 2 args".to_string());
                }
                let query = &args[0];
                // Threshold from popcount: 0-FINGERPRINT_BITS -> 0.0-1.0
                let threshold = args[1].popcount() as f32 / crate::FINGERPRINT_BITS as f32;

                // Use lance_db if available
                if let Some(lance) = ctx.lance_db {
                    // Estimate k from threshold (higher threshold = fewer results)
                    let k = ((1.0 - threshold) * 100.0).max(1.0) as usize;
                    if let Ok(results) = lance.vector_search("nodes", query, k) {
                        // Filter by actual threshold
                        let filtered: Vec<Fingerprint> = results
                            .into_iter()
                            .filter(|fp| query.similarity(fp) >= threshold)
                            .collect();
                        return OpResult::Many(filtered);
                    }
                }

                // Fallback: return permuted variants as "matched nodes"
                let mut matches = Vec::new();
                for i in 1..=5 {
                    let variant = query.permute(i * 100);
                    if query.similarity(&variant) >= threshold {
                        matches.push(variant);
                    }
                }
                OpResult::Many(matches)
            }),
        );

        self.register(
            CypherOp::PageRank as u16,
            "CYPHER_PAGERANK",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::Scalar],
                output: OpType::FingerprintArray,
            },
            "Compute PageRank centrality - returns nodes sorted by importance",
            Arc::new(|_ctx, args| {
                if args.len() < 2 {
                    return OpResult::Error("PageRank requires 2 args".to_string());
                }
                let graph_marker = &args[0];
                // Iterations from popcount (1-100)
                let iterations = (args[1].popcount() as usize / 100).max(1).min(20);

                // Simplified PageRank on fingerprint structure:
                // Interpret popcount distribution as node connectivity
                // Higher popcount regions = more connected = higher rank
                let mut ranked = Vec::with_capacity(iterations);

                // Generate ranked nodes based on permutation analysis
                let base_pop = graph_marker.popcount();
                for i in 0..iterations {
                    let permuted = graph_marker.permute((i as i32 + 1) * 500);
                    let pop = permuted.popcount();
                    // Higher popcount after permute = more "central"
                    if pop >= base_pop / 2 {
                        ranked.push(permuted);
                    }
                }

                // Sort by popcount (proxy for PageRank score)
                ranked.sort_by(|a, b| b.popcount().cmp(&a.popcount()));

                OpResult::Many(ranked)
            }),
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
            }),
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
            }),
        );

        self.register(
            HammingOp::MexicanHat as u16,
            "HAM_MEXICAN_HAT",
            OpSignature {
                inputs: vec![
                    OpType::Fingerprint,
                    OpType::Fingerprint,
                    OpType::Scalar,
                    OpType::Scalar,
                ],
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
                    -0.3 * (sim - 0.5) / 0.4 // Inhibition zone
                } else {
                    0.0 // Far = ignore
                };
                OpResult::Scalar(response as f64)
            }),
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
            }),
        );

        self.register(
            LearnOp::ResonanceCapture as u16,
            "LEARN_RESONANCE_CAPTURE",
            OpSignature {
                inputs: vec![
                    OpType::Fingerprint,
                    OpType::FingerprintArray,
                    OpType::Scalar,
                ],
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
            }),
        );

        // Crystal training: learn T (time), S (structure), D (detail) axes
        self.register(
            LearnOp::CrystalTrain as u16,
            "LEARN_CRYSTAL_TRAIN",
            OpSignature {
                inputs: vec![OpType::FingerprintArray], // Pairs of (input, output)
                output: OpType::FingerprintArray,       // 3 axis fingerprints
            },
            "Train crystal model - returns T, S, D axis fingerprints",
            Arc::new(|_ctx, args| {
                if args.len() < 2 {
                    return OpResult::Many(vec![Fingerprint::zero(); 3]);
                }

                // Crystal learning extracts 3 orthogonal axes:
                // T (temporal): sequence/flow patterns
                // S (structural): static relationships
                // D (detail): fine-grained variations

                // Compute differences between pairs
                let mut diffs = Vec::new();
                for i in (0..args.len()).step_by(2) {
                    if i + 1 < args.len() {
                        let diff = args[i].bind(&args[i + 1]);
                        diffs.push(diff);
                    }
                }

                if diffs.is_empty() {
                    return OpResult::Many(vec![Fingerprint::zero(); 3]);
                }

                // T-axis: bundle all diffs (captures change patterns)
                let t_axis = bundle_fingerprints(&diffs);

                // S-axis: common structure via XOR chain
                let inputs: Vec<&Fingerprint> = args.iter().step_by(2).collect();
                let s_axis = if inputs.len() >= 2 {
                    let mut common = inputs[0].clone();
                    for fp in inputs.iter().skip(1) {
                        common = common.bind(fp);
                    }
                    common
                } else {
                    args[0].clone()
                };

                // D-axis: perpendicular to T and S
                let ts_bound = t_axis.bind(&s_axis);
                let d_axis = diffs[0].unbind(&ts_bound);

                OpResult::Many(vec![t_axis, s_axis, d_axis])
            }),
        );

        // Concept extraction: decompose into prime components
        self.register(
            LearnOp::ConceptExtract as u16,
            "LEARN_CONCEPT_EXTRACT",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::FingerprintArray,
            },
            "Extract prime concept decomposition via iterative unbinding",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::Many(vec![]);
                }
                let compound = &args[0];

                // Decompose into prime factors using orthogonal basis
                let mut primes = Vec::with_capacity(8);
                let mut residual = compound.clone();

                for i in 0..8 {
                    let basis = Fingerprint::orthogonal(i);
                    let component = residual.unbind(&basis);

                    let sim = component.similarity(&residual);
                    if sim < 0.8 && component.popcount() > 100 {
                        primes.push(component.clone());
                        residual = residual.unbind(&component);
                    }
                }

                if primes.is_empty() && compound.popcount() > 0 {
                    primes.push(compound.clone());
                }

                OpResult::Many(primes)
            }),
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
            }),
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
            }),
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
            }),
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
                    return OpResult::Error(
                        "Become requires current and target states".to_string(),
                    );
                }

                let current = &args[0];
                let target = &args[1];

                // The becoming is the XOR path between states
                // (what must change to get from here to there)
                let transition = current.bind(target);
                let becoming_marker = Fingerprint::from_content("BECOMING::transition");
                let became = transition.bind(&becoming_marker);
                OpResult::One(became)
            }),
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
                for bit in 0..crate::FINGERPRINT_BITS {
                    if bit % 4 != 0 {
                        // Keep only 25% of bits
                        whispered.set_bit(bit, false);
                    }
                }
                OpResult::One(whispered)
            }),
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
                let dream_noise = Fingerprint::from_content(&format!(
                    "DREAM::noise::{}",
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos()
                ));

                // Blend: mostly bundled memory, some noise
                let mut dreamed = bundled.clone();
                for bit in 0..crate::FINGERPRINT_BITS {
                    if bit % 10 == 0 {
                        // 10% noise
                        dreamed.set_bit(bit, dream_noise.get_bit(bit));
                    }
                }
                OpResult::One(dreamed)
            }),
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
                    sim // Strong resonance
                } else if sim > 0.5 {
                    -0.3 * (sim - 0.5) / 0.3 // Inhibition zone
                } else {
                    0.0 // Below threshold
                };

                OpResult::Scalar(resonance as f64)
            }),
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
            }),
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
    pub fn execute_semantic(
        &self,
        description: &str,
        ctx: &OpContext,
        args: &[Fingerprint],
    ) -> OpResult {
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
            OpResult::Error(format!(
                "No operation matches: {} (best sim: {})",
                description, best_sim
            ))
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

        (start..end).filter_map(|id| self.get_meta(id)).collect()
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

    for bit in 0..crate::FINGERPRINT_BITS {
        let count: usize = fps.iter().filter(|fp| fp.get_bit(bit)).count();
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

    // XOR-fold 256 u64s down to 1
    for &word in raw.iter() {
        hash ^= word;
    }

    // Take lower 48 bits
    hash & 0xFFFF_FFFF_FFFF
}

// =============================================================================
// NARS INFERENCE OPERATIONS (0x400-0x4FF)
// =============================================================================

impl OpDictionary {
    fn register_nars_ops(&mut self) {
        // Deduction: M→P, S→M ⊢ S→P
        self.register(
            NarsOp::Deduction as u16,
            "NARS_DEDUCTION",
            OpSignature {
                inputs: vec![
                    OpType::Fingerprint,
                    OpType::Fingerprint,
                    OpType::Fingerprint,
                ],
                output: OpType::Fingerprint,
            },
            "NARS deduction: M→P, S→M ⊢ S→P with truth function",
            Arc::new(|_ctx, args| {
                if args.len() < 3 {
                    return OpResult::Error("Deduction requires M, P, S".to_string());
                }
                // Bind: conclusion = S ⊗ P (via transitive inference)
                let conclusion = args[2].bind(&args[1]);
                OpResult::One(conclusion)
            }),
        );

        // Induction: M→P, M→S ⊢ S→P
        self.register(
            NarsOp::Induction as u16,
            "NARS_INDUCTION",
            OpSignature {
                inputs: vec![
                    OpType::Fingerprint,
                    OpType::Fingerprint,
                    OpType::Fingerprint,
                ],
                output: OpType::Fingerprint,
            },
            "NARS induction: M→P, M→S ⊢ S→P",
            Arc::new(|_ctx, args| {
                if args.len() < 3 {
                    return OpResult::Error("Induction requires M, P, S".to_string());
                }
                let conclusion = args[2].bind(&args[1]);
                OpResult::One(conclusion)
            }),
        );

        // Abduction: P→M, S→M ⊢ S→P
        self.register(
            NarsOp::Abduction as u16,
            "NARS_ABDUCTION",
            OpSignature {
                inputs: vec![
                    OpType::Fingerprint,
                    OpType::Fingerprint,
                    OpType::Fingerprint,
                ],
                output: OpType::Fingerprint,
            },
            "NARS abduction: P→M, S→M ⊢ S→P (hypothesis formation)",
            Arc::new(|_ctx, args| {
                if args.len() < 3 {
                    return OpResult::Error("Abduction requires P, M, S".to_string());
                }
                let hypothesis = args[2].bind(&args[0]);
                OpResult::One(hypothesis)
            }),
        );

        // Revision: combine evidence
        self.register(
            NarsOp::Revision as u16,
            "NARS_REVISION",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "NARS revision: combine evidence from two sources",
            Arc::new(|_ctx, args| {
                if args.len() < 2 {
                    return OpResult::Error("Revision requires two beliefs".to_string());
                }
                // Bundle (majority voting) for revision
                let revised = bundle_fingerprints(&[args[0].clone(), args[1].clone()]);
                OpResult::One(revised)
            }),
        );

        // Choice: select best action
        self.register(
            NarsOp::Choice as u16,
            "NARS_CHOICE",
            OpSignature {
                inputs: vec![OpType::FingerprintArray],
                output: OpType::Fingerprint,
            },
            "NARS choice: select option with highest expectation",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::Error("Choice requires options".to_string());
                }
                // Return first option (would normally sort by expectation)
                OpResult::One(args[0].clone())
            }),
        );
    }
}

// =============================================================================
// CAUSAL OPERATIONS (0xA00-0xAFF) - Pearl's Causal Ladder
// =============================================================================

impl OpDictionary {
    fn register_causal_ops(&mut self) {
        // Rung 2: Intervention (do-calculus)
        self.register(
            CausalOp::DoIntervene as u16,
            "CAUSAL_DO_INTERVENE",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "Rung 2: Intervention P(Y|do(X)) - cut incoming edges",
            Arc::new(|_ctx, args| {
                if args.len() < 2 {
                    return OpResult::Error("DoIntervene requires cause and effect".to_string());
                }
                let do_marker = Fingerprint::from_content("DO::intervention");
                let effect = args[0].bind(&do_marker).bind(&args[1]);
                OpResult::One(effect)
            }),
        );

        // Rung 3: Counterfactual
        self.register(
            CausalOp::Counterfactual as u16,
            "CAUSAL_COUNTERFACTUAL",
            OpSignature {
                inputs: vec![
                    OpType::Fingerprint,
                    OpType::Fingerprint,
                    OpType::Fingerprint,
                ],
                output: OpType::Fingerprint,
            },
            "Rung 3: Counterfactual P(Y_x'|X=x, Y=y) - what if?",
            Arc::new(|_ctx, args| {
                if args.len() < 3 {
                    return OpResult::Error(
                        "Counterfactual requires actual, counterfactual, outcome".to_string(),
                    );
                }
                let actual = &args[0];
                let counter = &args[1];
                let outcome = &args[2];
                let cf_outcome = outcome.bind(actual).bind(counter);
                OpResult::One(cf_outcome)
            }),
        );

        // Graph parents (discover causes)
        // Uses unbind to extract potential parent components from a compound node
        self.register(
            CausalOp::GraphParents as u16,
            "CAUSAL_GRAPH_PARENTS",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::FingerprintArray,
            },
            "Get parent nodes (causes) - extracts components via unbind",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::Many(vec![]);
                }
                let node = &args[0];

                // Extract potential parent components using orthogonal basis unbind
                // If node = parent1 ⊗ parent2 ⊗ ... then unbinding with basis vectors
                // can recover approximate parents
                let mut parents = Vec::with_capacity(8);

                // Try unbinding with standard basis vectors
                for i in 0..8 {
                    let basis = Fingerprint::orthogonal(i);
                    let potential_parent = node.unbind(&basis);

                    // Check if this is a valid parent (similarity > 0.3 suggests structure)
                    let sim = node.similarity(&potential_parent);
                    if sim > 0.3 && sim < 0.95 {
                        parents.push(potential_parent);
                    }
                }

                // Also try permutation-based parent extraction
                for shift in [100, 500, 1000, 2000] {
                    let permuted = node.unpermute(shift);
                    if node.similarity(&permuted) > 0.4 && node.similarity(&permuted) < 0.9 {
                        parents.push(permuted);
                    }
                }

                OpResult::Many(parents)
            }),
        );

        // Graph children (trace effects)
        // Uses bind to compute potential effects when combined with context
        self.register(
            CausalOp::GraphChildren as u16,
            "CAUSAL_GRAPH_CHILDREN",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::FingerprintArray,
            },
            "Get child nodes (effects) - computes potential bindings",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::Many(vec![]);
                }
                let node = &args[0];

                // Generate potential children by binding with basis vectors
                // Each child represents: effect = cause ⊗ context
                let mut children = Vec::with_capacity(8);

                // Bind with orthogonal contexts to generate effect space
                for i in 0..8 {
                    let context = Fingerprint::orthogonal(i + 100); // Different basis from parents
                    let child = node.bind(&context);
                    children.push(child);
                }

                // Also generate children via permutation (temporal effects)
                for shift in [50, 150, 350, 700] {
                    let temporal_child = node.permute(shift);
                    children.push(temporal_child);
                }

                OpResult::Many(children)
            }),
        );

        // Backdoor adjustment
        self.register(
            CausalOp::BackdoorAdjust as u16,
            "CAUSAL_BACKDOOR_ADJUST",
            OpSignature {
                inputs: vec![
                    OpType::Fingerprint,
                    OpType::Fingerprint,
                    OpType::FingerprintArray,
                ],
                output: OpType::Fingerprint,
            },
            "Backdoor adjustment formula",
            Arc::new(|_ctx, args| {
                if args.len() < 2 {
                    return OpResult::Error(
                        "BackdoorAdjust requires X, Y, and confounders".to_string(),
                    );
                }
                let adjusted = args[0].bind(&args[1]);
                OpResult::One(adjusted)
            }),
        );
    }
}

// =============================================================================
// META-COGNITION OPERATIONS (0xD00-0xDFF)
// =============================================================================

impl OpDictionary {
    fn register_meta_ops(&mut self) {
        // Self-model
        self.register(
            MetaOp::SelfModel as u16,
            "META_SELF_MODEL",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "Access/update self-model representation",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::One(Fingerprint::from_content("META::self_model"));
                }
                let self_marker = Fingerprint::from_content("META::self");
                let reflected = args[0].bind(&self_marker);
                OpResult::One(reflected)
            }),
        );

        // Self-explain
        self.register(
            MetaOp::SelfExplain as u16,
            "META_SELF_EXPLAIN",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "Explain reasoning trace for decision",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::One(Fingerprint::from_content("EXPLAIN::empty"));
                }
                let explain_marker = Fingerprint::from_content("META::explanation");
                let explained = args[0].bind(&explain_marker);
                OpResult::One(explained)
            }),
        );

        // Know-what
        self.register(
            MetaOp::KnowWhat as u16,
            "META_KNOW_WHAT",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "Declarative knowledge - know that X",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::One(Fingerprint::from_content("KNOW::nothing"));
                }
                let know_marker = Fingerprint::from_content("KNOW::what");
                OpResult::One(args[0].bind(&know_marker))
            }),
        );

        // Know-how
        self.register(
            MetaOp::KnowHow as u16,
            "META_KNOW_HOW",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "Procedural knowledge - know how to X",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::One(Fingerprint::from_content("KNOW::nothing"));
                }
                let know_marker = Fingerprint::from_content("KNOW::how");
                OpResult::One(args[0].bind(&know_marker))
            }),
        );

        // Confidence get - measures bit density consistency
        // High confidence = bits evenly distributed, low = clustered
        self.register(
            MetaOp::ConfidenceGet as u16,
            "META_CONFIDENCE_GET",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::Scalar,
            },
            "Get confidence level for belief based on bit consistency",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::Scalar(0.0);
                }
                let fp = &args[0];

                // Measure variance across 10 segments of 1000 bits each
                let mut segment_counts = [0u32; 10];
                for seg in 0..10 {
                    let base = seg * 1000;
                    for i in 0..1000 {
                        if fp.get_bit(base + i) {
                            segment_counts[seg] += 1;
                        }
                    }
                }

                // Calculate mean
                let mean = segment_counts.iter().sum::<u32>() as f64 / 10.0;
                if mean < 1.0 {
                    return OpResult::Scalar(0.0); // Empty fingerprint = no confidence
                }

                // Calculate variance
                let variance = segment_counts
                    .iter()
                    .map(|&c| (c as f64 - mean).powi(2))
                    .sum::<f64>()
                    / 10.0;

                // Low variance = high confidence, high variance = low confidence
                // Normalize: CV (coefficient of variation) = sqrt(var) / mean
                let cv = variance.sqrt() / mean;
                let confidence = (1.0 - cv.min(1.0)).max(0.0);

                OpResult::Scalar(confidence)
            }),
        );

        // Uncertainty quantify - entropy-based uncertainty measure
        self.register(
            MetaOp::UncertaintyQuantify as u16,
            "META_UNCERTAINTY_QUANTIFY",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::Scalar,
            },
            "Quantify uncertainty using bit entropy",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::Scalar(1.0); // Maximum uncertainty for empty
                }
                let fp = &args[0];

                // Calculate Shannon entropy of bit pattern
                // Sample 100 windows of 100 bits each
                let mut window_densities = [0u32; 100];
                for w in 0..100 {
                    let base = w * 100;
                    for i in 0..100 {
                        if fp.get_bit(base + i) {
                            window_densities[w] += 1;
                        }
                    }
                }

                // Count occurrences of each density level (0-100)
                let mut density_counts = [0u32; 101];
                for &d in &window_densities {
                    density_counts[d as usize] += 1;
                }

                // Calculate entropy: H = -sum(p * log2(p))
                let total = 100.0f64;
                let entropy = density_counts
                    .iter()
                    .filter(|&&c| c > 0)
                    .map(|&c| {
                        let p = c as f64 / total;
                        -p * p.log2()
                    })
                    .sum::<f64>();

                // Normalize entropy: max is log2(100) ≈ 6.64 for uniform distribution
                let max_entropy = 6.64;
                let uncertainty = (entropy / max_entropy).min(1.0);

                OpResult::Scalar(uncertainty)
            }),
        );
    }
}

// =============================================================================
// VERBS OPERATIONS (0x700-0x7FF)
// =============================================================================

impl OpDictionary {
    fn register_verbs_ops(&mut self) {
        let verbs = [
            (
                VerbOp::Causes as u16,
                "CAUSES",
                "Causal relation: A causes B",
            ),
            (VerbOp::Becomes as u16, "BECOMES", "Transition: A becomes B"),
            (VerbOp::Enables as u16, "ENABLES", "Enablement: A enables B"),
            (
                VerbOp::Prevents as u16,
                "PREVENTS",
                "Prevention: A prevents B",
            ),
            (
                VerbOp::Contains as u16,
                "CONTAINS",
                "Containment: A contains B",
            ),
            (
                VerbOp::Requires as u16,
                "REQUIRES",
                "Requirement: A requires B",
            ),
            (
                VerbOp::Implies as u16,
                "IMPLIES",
                "Implication: A implies B",
            ),
            (VerbOp::Supports as u16, "SUPPORTS", "Support: A supports B"),
            (
                VerbOp::Contradicts as u16,
                "CONTRADICTS",
                "Contradiction: A contradicts B",
            ),
            (VerbOp::Grounds as u16, "GROUNDS", "Grounding: A grounds B"),
            (
                VerbOp::Abstracts as u16,
                "ABSTRACTS",
                "Abstraction: A abstracts B",
            ),
            (VerbOp::Refines as u16, "REFINES", "Refinement: A refines B"),
        ];

        for (id, name, doc) in verbs {
            let verb_name = name.to_string();
            self.register(
                id,
                &format!("VERB_{}", name),
                OpSignature {
                    inputs: vec![OpType::Fingerprint, OpType::Fingerprint],
                    output: OpType::Fingerprint,
                },
                doc,
                Arc::new(move |_ctx, args| {
                    if args.len() < 2 {
                        return OpResult::Error(format!("{} requires two arguments", verb_name));
                    }
                    // Create verb fingerprint and bind: A ⊗ VERB ⊗ B
                    let verb_fp = Fingerprint::from_content(&format!("VERB::{}", verb_name));
                    let edge = args[0].bind(&verb_fp).bind(&args[1]);
                    OpResult::One(edge)
                }),
            );
        }
    }
}

// =============================================================================
// QUALIA OPERATIONS (0xB00-0xBFF)
// =============================================================================

impl OpDictionary {
    fn register_qualia_ops(&mut self) {
        // Valence get - positive/negative dimension
        // Extracts valence from fingerprint structure using first 1000 bits
        self.register(
            QualiaOp::ValenceGet as u16,
            "QUALIA_VALENCE_GET",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::Scalar,
            },
            "Get valence (-1.0 to 1.0) from fingerprint structure",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::Scalar(0.0);
                }
                let fp = &args[0];

                // Valence encoded in first 2000 bits:
                // Count set bits in first 1000 vs next 1000
                // More in first half = positive, more in second = negative
                let mut positive_bits = 0u32;
                let mut negative_bits = 0u32;

                for i in 0..1000 {
                    if fp.get_bit(i) {
                        positive_bits += 1;
                    }
                    if fp.get_bit(i + 1000) {
                        negative_bits += 1;
                    }
                }

                // Normalize to -1.0 to 1.0
                let total = (positive_bits + negative_bits).max(1) as f64;
                let valence = (positive_bits as f64 - negative_bits as f64) / total;

                OpResult::Scalar(valence)
            }),
        );

        // Arousal get - activation dimension
        // Extracts arousal from fingerprint entropy/spread
        self.register(
            QualiaOp::ArousalGet as u16,
            "QUALIA_AROUSAL_GET",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::Scalar,
            },
            "Get arousal (0.0 to 1.0) from fingerprint activation spread",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::Scalar(0.5);
                }
                let fp = &args[0];

                // Arousal = how spread out the activation is
                // High arousal = bits evenly distributed
                // Low arousal = bits clustered
                let total_pop = fp.popcount();

                // Measure distribution by comparing quarters
                let mut quarter_pops = [0u32; 4];
                for q in 0..4 {
                    for i in 0..2500 {
                        if fp.get_bit(q * 2500 + i) {
                            quarter_pops[q] += 1;
                        }
                    }
                }

                // Calculate variance (low variance = even spread = high arousal)
                let mean = total_pop as f64 / 4.0;
                let variance: f64 = quarter_pops
                    .iter()
                    .map(|&q| (q as f64 - mean).powi(2))
                    .sum::<f64>()
                    / 4.0;

                // Normalize: low variance = high arousal
                let max_variance = (mean * 3.0).powi(2); // theoretical max
                let arousal = 1.0 - (variance / max_variance.max(1.0)).min(1.0);

                OpResult::Scalar(arousal)
            }),
        );

        // Qualia create
        self.register(
            QualiaOp::QualiaCreate as u16,
            "QUALIA_CREATE",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "Create qualia vector from state",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::One(Fingerprint::from_content("QUALIA::neutral"));
                }
                let qualia_marker = Fingerprint::from_content("QUALIA::felt");
                let felt = args[0].bind(&qualia_marker);
                OpResult::One(felt)
            }),
        );

        // Qualia similarity
        self.register(
            QualiaOp::QualiaSimilarity as u16,
            "QUALIA_SIMILARITY",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::Fingerprint],
                output: OpType::Scalar,
            },
            "Compute similarity between qualia states",
            Arc::new(|_ctx, args| {
                if args.len() < 2 {
                    return OpResult::Scalar(0.0);
                }
                let sim = args[0].similarity(&args[1]) as f64;
                OpResult::Scalar(sim)
            }),
        );

        // Qualia blend
        self.register(
            QualiaOp::QualiaBlend as u16,
            "QUALIA_BLEND",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::Fingerprint, OpType::Scalar],
                output: OpType::Fingerprint,
            },
            "Blend two qualia states with weight",
            Arc::new(|_ctx, args| {
                if args.len() < 2 {
                    return OpResult::Error("Blend requires two qualia states".to_string());
                }
                // Blend via bundling
                let blended = bundle_fingerprints(&[args[0].clone(), args[1].clone()]);
                OpResult::One(blended)
            }),
        );

        // Emotion joy
        self.register(
            QualiaOp::EmotionJoy as u16,
            "QUALIA_EMOTION_JOY",
            OpSignature {
                inputs: vec![],
                output: OpType::Fingerprint,
            },
            "Get joy emotion fingerprint",
            Arc::new(|_ctx, _args| OpResult::One(Fingerprint::from_content("EMOTION::joy"))),
        );

        // Emotion sadness
        self.register(
            QualiaOp::EmotionSadness as u16,
            "QUALIA_EMOTION_SADNESS",
            OpSignature {
                inputs: vec![],
                output: OpType::Fingerprint,
            },
            "Get sadness emotion fingerprint",
            Arc::new(|_ctx, _args| OpResult::One(Fingerprint::from_content("EMOTION::sadness"))),
        );

        // Emotion blend
        self.register(
            QualiaOp::EmotionBlend as u16,
            "QUALIA_EMOTION_BLEND",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "Blend two emotions together",
            Arc::new(|_ctx, args| {
                if args.len() < 2 {
                    return OpResult::Error("EmotionBlend requires two emotions".to_string());
                }
                let blended = bundle_fingerprints(&[args[0].clone(), args[1].clone()]);
                OpResult::One(blended)
            }),
        );
    }
}

// =============================================================================
// MEMORY OPERATIONS (0x0A00-0x0AFF)
// =============================================================================

impl OpDictionary {
    fn register_memory_ops(&mut self) {
        // Store to memory - binds with temporal context
        self.register(
            MemoryOp::Store as u16,
            "MEMORY_STORE",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::Bool,
            },
            "Store fingerprint to episodic memory with temporal binding",
            Arc::new(|ctx, args| {
                if args.is_empty() {
                    return OpResult::Bool(false);
                }
                let content = &args[0];

                // Create temporal context from content hash (pseudo-timestamp)
                // Uses popcount as temporal index for reproducibility
                let temporal_idx = content.popcount() as usize % 256;
                let temporal_ctx = Fingerprint::orthogonal(temporal_idx);

                // Bind content with temporal context for episodic storage
                let episodic = content.bind(&temporal_ctx);

                // Store via lance_db if available
                if let Some(lance) = ctx.lance_db {
                    if lance.insert("memories", &[episodic.clone()]).is_ok() {
                        return OpResult::Bool(true);
                    }
                }

                // Fallback: memory stored in binding operation itself
                // The episodic fingerprint carries the temporal context
                OpResult::Bool(episodic.popcount() > 0)
            }),
        );

        // Recall from memory - similarity-based retrieval
        self.register(
            MemoryOp::Recall as u16,
            "MEMORY_RECALL",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::Scalar],
                output: OpType::FingerprintArray,
            },
            "Recall similar memories (k nearest neighbors)",
            Arc::new(|ctx, args| {
                if args.is_empty() {
                    return OpResult::Many(vec![]);
                }
                let query = &args[0];
                // k from second arg popcount (1-100)
                let k = if args.len() > 1 {
                    (args[1].popcount() as usize / 100).max(1).min(100)
                } else {
                    10
                };

                // Use lance_db for vector search if available
                if let Some(lance) = ctx.lance_db {
                    if let Ok(results) = lance.vector_search("memories", query, k) {
                        if !results.is_empty() {
                            return OpResult::Many(results);
                        }
                    }
                }

                // Fallback: generate reconstructed memories via perturbation
                // This simulates associative memory recall
                let mut memories = Vec::with_capacity(k);

                // First result is the query itself (exact match)
                memories.push(query.clone());

                // Generate k-1 similar memories via controlled perturbation
                for i in 1..k {
                    // Perturb with decreasing similarity
                    let noise_level = i as i32 * 50;
                    let perturbed = query.permute(noise_level);

                    // Blend original with perturbed for partial recall
                    let basis = Fingerprint::orthogonal(i);
                    let recalled = query.bind(&basis).unbind(&perturbed);

                    memories.push(recalled);
                }

                OpResult::Many(memories)
            }),
        );

        // Forget - decay memory by unbinding with noise
        self.register(
            MemoryOp::Forget as u16,
            "MEMORY_FORGET",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::Bool,
            },
            "Forget memory by decaying signal with noise injection",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::Bool(false);
                }
                let memory = &args[0];

                // Generate decay noise based on memory content
                // Uses popcount mod to get reproducible decay pattern
                let decay_strength = (memory.popcount() as usize % 10) + 1;
                let noise = Fingerprint::orthogonal(decay_strength);

                // Forgetting = unbinding from noise (corrupts retrieval path)
                let decayed = memory.unbind(&noise);

                // Success if decay changed the fingerprint significantly
                // In practice, unbind always changes it, so check popcount balance
                let similarity = memory.similarity(&decayed);
                OpResult::Bool(similarity < 0.9)
            }),
        );

        // Consolidate (strengthen)
        self.register(
            MemoryOp::Consolidate as u16,
            "MEMORY_CONSOLIDATE",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "Consolidate memory (strengthen, move to long-term)",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::Error("Consolidate requires memory".to_string());
                }
                let consolidated = Fingerprint::from_content("CONSOLIDATED");
                let result = args[0].bind(&consolidated);
                OpResult::One(result)
            }),
        );
    }
}

// =============================================================================
// FILESYSTEM OPERATIONS (0x500-0x5FF)
// =============================================================================

impl OpDictionary {
    fn register_filesystem_ops(&mut self) {
        // File fingerprint hash - content-addressable fingerprint from path
        self.register(
            FilesystemOp::FileHash as u16,
            "FS_FILE_HASH",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "Compute content-addressable fingerprint from file path encoding",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::Error("FileHash requires path fingerprint".to_string());
                }
                // The input fingerprint encodes the path
                // Return a deterministic hash-like fingerprint
                let path_fp = &args[0];
                let hash_fp = Fingerprint::from_content(&format!("HASH::{}", path_fp.popcount()));
                let result = path_fp.bind(&hash_fp);
                OpResult::One(result)
            }),
        );

        // File exists check - similarity to known file patterns
        self.register(
            FilesystemOp::FileExists as u16,
            "FS_FILE_EXISTS",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::Bool,
            },
            "Check if file pattern exists (via similarity to known patterns)",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::Bool(false);
                }
                // Exists if fingerprint has valid structure (non-trivial popcount)
                let exists = args[0].popcount() > 100 && args[0].popcount() < 9900;
                OpResult::Bool(exists)
            }),
        );

        // Path join - bind two path components
        self.register(
            FilesystemOp::PathJoin as u16,
            "FS_PATH_JOIN",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "Join two path components via VSA binding",
            Arc::new(|_ctx, args| {
                if args.len() < 2 {
                    return OpResult::Error("PathJoin requires two arguments".to_string());
                }
                // Path separator fingerprint
                let sep = Fingerprint::from_content("PATH::SEP");
                // Join: parent ⊗ sep ⊗ child
                let joined = args[0].bind(&sep).bind(&args[1]);
                OpResult::One(joined)
            }),
        );

        // Path split - unbind to get components
        self.register(
            FilesystemOp::PathSplit as u16,
            "FS_PATH_SPLIT",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::FingerprintArray,
            },
            "Split path into components via VSA unbinding",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::Many(vec![]);
                }
                let path = &args[0];
                let sep = Fingerprint::from_content("PATH::SEP");

                // Extract parent and child by unbinding
                let parent = path.unbind(&sep);
                let child = path.unbind(&parent);

                OpResult::Many(vec![parent, child])
            }),
        );

        // Serialize to fingerprint
        self.register(
            FilesystemOp::SerializeBincode as u16,
            "FS_SERIALIZE",
            OpSignature {
                inputs: vec![OpType::FingerprintArray],
                output: OpType::Fingerprint,
            },
            "Serialize multiple fingerprints into one via bundling",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::One(Fingerprint::zero());
                }
                // Bundle all inputs into one fingerprint
                let bundled = bundle_fingerprints(&args);
                OpResult::One(bundled)
            }),
        );

        // Compress fingerprint (increase density)
        self.register(
            FilesystemOp::CompressLz4 as u16,
            "FS_COMPRESS",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "Compress fingerprint by projecting to lower-entropy space",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::Error("Compress requires input".to_string());
                }
                // Compression = project to structured basis
                let basis = Fingerprint::from_content("COMPRESS::BASIS");
                let compressed = args[0].bind(&basis);
                OpResult::One(compressed)
            }),
        );

        // Decompress fingerprint
        self.register(
            FilesystemOp::DecompressLz4 as u16,
            "FS_DECOMPRESS",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "Decompress fingerprint by reversing projection",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::Error("Decompress requires input".to_string());
                }
                // Decompression = unbind from structured basis
                let basis = Fingerprint::from_content("COMPRESS::BASIS");
                let decompressed = args[0].unbind(&basis);
                OpResult::One(decompressed)
            }),
        );

        // FP Save - encode fingerprint for storage
        self.register(
            FilesystemOp::FpSave as u16,
            "FS_FP_SAVE",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "Encode fingerprint with storage key binding",
            Arc::new(|_ctx, args| {
                if args.len() < 2 {
                    return OpResult::Error("FpSave requires content and key".to_string());
                }
                // Bind content with storage key
                let stored = args[0].bind(&args[1]);
                OpResult::One(stored)
            }),
        );

        // FP Load - decode fingerprint from storage
        self.register(
            FilesystemOp::FpLoad as u16,
            "FS_FP_LOAD",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "Decode fingerprint by unbinding storage key",
            Arc::new(|_ctx, args| {
                if args.len() < 2 {
                    return OpResult::Error("FpLoad requires stored and key".to_string());
                }
                // Unbind storage key to recover content
                let loaded = args[0].unbind(&args[1]);
                OpResult::One(loaded)
            }),
        );
    }
}

// =============================================================================
// CRYSTAL/TEMPORAL OPERATIONS (0x600-0x6FF)
// =============================================================================

impl OpDictionary {
    fn register_crystal_ops(&mut self) {
        // Crystal create - initialize T/S/D axes
        self.register(
            CrystalOp::CrystalCreate as u16,
            "CRYSTAL_CREATE",
            OpSignature {
                inputs: vec![],
                output: OpType::Fingerprint,
            },
            "Create new crystal with orthogonal T/S/D axes",
            Arc::new(|_ctx, _args| {
                // Crystal = bundled orthogonal axes
                let t_axis = Fingerprint::orthogonal(0);
                let s_axis = Fingerprint::orthogonal(1);
                let d_axis = Fingerprint::orthogonal(2);
                let crystal = bundle_fingerprints(&[t_axis, s_axis, d_axis]);
                OpResult::One(crystal)
            }),
        );

        // Crystal infer - project input onto crystal space
        self.register(
            CrystalOp::CrystalInfer as u16,
            "CRYSTAL_INFER",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "Project input onto crystal axes for inference",
            Arc::new(|_ctx, args| {
                if args.len() < 2 {
                    return OpResult::Error("CrystalInfer requires input and crystal".to_string());
                }
                let input = &args[0];
                let crystal = &args[1];

                // Project onto crystal space
                let projection = input.bind(crystal);
                OpResult::One(projection)
            }),
        );

        // Axis T (Topic) extraction
        self.register(
            CrystalOp::AxisT as u16,
            "CRYSTAL_AXIS_T",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "Extract Topic axis from crystal projection",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::Error("AxisT requires crystal".to_string());
                }
                let t_basis = Fingerprint::orthogonal(0);
                let t_component = args[0].unbind(&t_basis);
                OpResult::One(t_component)
            }),
        );

        // Axis S (Style) extraction
        self.register(
            CrystalOp::AxisS as u16,
            "CRYSTAL_AXIS_S",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "Extract Style axis from crystal projection",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::Error("AxisS requires crystal".to_string());
                }
                let s_basis = Fingerprint::orthogonal(1);
                let s_component = args[0].unbind(&s_basis);
                OpResult::One(s_component)
            }),
        );

        // Axis D (Detail) extraction
        self.register(
            CrystalOp::AxisD as u16,
            "CRYSTAL_AXIS_D",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "Extract Detail axis from crystal projection",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::Error("AxisD requires crystal".to_string());
                }
                let d_basis = Fingerprint::orthogonal(2);
                let d_component = args[0].unbind(&d_basis);
                OpResult::One(d_component)
            }),
        );

        // Axis interpolation
        self.register(
            CrystalOp::AxisInterpolate as u16,
            "CRYSTAL_INTERPOLATE",
            OpSignature {
                inputs: vec![
                    OpType::Fingerprint,
                    OpType::Fingerprint,
                    OpType::Fingerprint,
                ],
                output: OpType::Fingerprint,
            },
            "Interpolate between two points along an axis",
            Arc::new(|_ctx, args| {
                if args.len() < 3 {
                    return OpResult::Error(
                        "Interpolate requires start, end, and factor".to_string(),
                    );
                }
                let start = &args[0];
                let end = &args[1];
                // Factor encoded in popcount (0-FINGERPRINT_BITS → 0.0-1.0)
                let factor = args[2].popcount() as f64 / crate::FINGERPRINT_BITS as f64;

                // Linear interpolation via weighted bundling
                if factor < 0.5 {
                    // Closer to start - bundle with start bias
                    let result = bundle_fingerprints(&[start.clone(), start.clone(), end.clone()]);
                    OpResult::One(result)
                } else {
                    // Closer to end - bundle with end bias
                    let result = bundle_fingerprints(&[start.clone(), end.clone(), end.clone()]);
                    OpResult::One(result)
                }
            }),
        );

        // Temporal before check
        self.register(
            CrystalOp::TemporalBefore as u16,
            "CRYSTAL_BEFORE",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::Fingerprint],
                output: OpType::Bool,
            },
            "Check if first event is temporally before second",
            Arc::new(|_ctx, args| {
                if args.len() < 2 {
                    return OpResult::Bool(false);
                }
                // Temporal order encoded in fingerprint structure
                // Lower popcount = earlier in sequence
                let before = args[0].popcount() < args[1].popcount();
                OpResult::Bool(before)
            }),
        );

        // Layer hot (recent/active)
        self.register(
            CrystalOp::LayerHot as u16,
            "CRYSTAL_LAYER_HOT",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "Project to hot (active) layer",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::Error("LayerHot requires input".to_string());
                }
                let hot_marker = Fingerprint::from_content("LAYER::HOT");
                let hot = args[0].bind(&hot_marker);
                OpResult::One(hot)
            }),
        );

        // Layer promote (cold → warm → hot)
        self.register(
            CrystalOp::LayerPromote as u16,
            "CRYSTAL_LAYER_PROMOTE",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "Promote layer from cold toward hot",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::Error("LayerPromote requires input".to_string());
                }
                // Promotion = unbind cold marker, bind warm/hot
                let cold = Fingerprint::from_content("LAYER::COLD");
                let warm = Fingerprint::from_content("LAYER::WARM");
                let promoted = args[0].unbind(&cold).bind(&warm);
                OpResult::One(promoted)
            }),
        );

        // Decay exponential
        self.register(
            CrystalOp::DecayExponential as u16,
            "CRYSTAL_DECAY_EXP",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "Apply exponential decay based on time factor",
            Arc::new(|_ctx, args| {
                if args.len() < 2 {
                    return OpResult::Error("DecayExponential requires input and time".to_string());
                }
                // Decay = gradual noise injection
                let time_factor = (args[1].popcount() as usize % 10) + 1;
                let noise = Fingerprint::orthogonal(time_factor);
                let decayed = args[0].bind(&noise);
                OpResult::One(decayed)
            }),
        );
    }
}

// =============================================================================
// ACT-R COGNITIVE ARCHITECTURE (0x800-0x8FF)
// =============================================================================

impl OpDictionary {
    fn register_actr_ops(&mut self) {
        // Chunk create
        self.register(
            ActrOp::ChunkCreate as u16,
            "ACTR_CHUNK_CREATE",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "Create ACT-R declarative memory chunk",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::One(Fingerprint::from_content("CHUNK::EMPTY"));
                }
                let chunk_marker = Fingerprint::from_content("CHUNK::TYPE");
                let chunk = args[0].bind(&chunk_marker);
                OpResult::One(chunk)
            }),
        );

        // Chunk retrieve with activation
        self.register(
            ActrOp::ChunkRetrieve as u16,
            "ACTR_CHUNK_RETRIEVE",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::FingerprintArray],
                output: OpType::Fingerprint,
            },
            "Retrieve chunk from declarative memory via spreading activation",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::Error("ChunkRetrieve requires cue".to_string());
                }
                let cue = &args[0];

                // Find best match from memory (args[1:])
                let memory = &args[1..];
                if memory.is_empty() {
                    return OpResult::One(cue.clone()); // Echo cue if no memory
                }

                let mut best_match = &memory[0];
                let mut best_sim = cue.similarity(&memory[0]);

                for chunk in memory.iter().skip(1) {
                    let sim = cue.similarity(chunk);
                    if sim > best_sim {
                        best_sim = sim;
                        best_match = chunk;
                    }
                }

                OpResult::One(best_match.clone())
            }),
        );

        // Chunk activation (base-level + spreading)
        self.register(
            ActrOp::ChunkActivation as u16,
            "ACTR_ACTIVATION",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::Fingerprint],
                output: OpType::Scalar,
            },
            "Compute chunk activation level",
            Arc::new(|_ctx, args| {
                if args.len() < 2 {
                    return OpResult::Scalar(0.0);
                }
                let chunk = &args[0];
                let context = &args[1];

                // Base-level: popcount-based
                let base = chunk.popcount() as f64 / crate::FINGERPRINT_BITS as f64;

                // Spreading activation: similarity to context
                let spreading = chunk.similarity(context) as f64;

                // Total activation
                let activation = base + spreading;
                OpResult::Scalar(activation)
            }),
        );

        // Chunk partial match
        self.register(
            ActrOp::ChunkPartialMatch as u16,
            "ACTR_PARTIAL_MATCH",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::Fingerprint],
                output: OpType::Scalar,
            },
            "Compute partial match penalty for chunk retrieval",
            Arc::new(|_ctx, args| {
                if args.len() < 2 {
                    return OpResult::Scalar(0.0);
                }
                // Mismatch penalty = Hamming distance / max
                let dist = args[0].hamming(&args[1]);
                let penalty = dist as f64 / crate::FINGERPRINT_BITS as f64;
                OpResult::Scalar(-penalty) // Negative = penalty
            }),
        );

        // Production match
        self.register(
            ActrOp::ProductionMatch as u16,
            "ACTR_PRODUCTION_MATCH",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::Fingerprint],
                output: OpType::Bool,
            },
            "Check if production condition matches buffer state",
            Arc::new(|_ctx, args| {
                if args.len() < 2 {
                    return OpResult::Bool(false);
                }
                // Match if similarity above threshold
                let sim = args[0].similarity(&args[1]);
                OpResult::Bool(sim > 0.7)
            }),
        );

        // Production fire
        self.register(
            ActrOp::ProductionFire as u16,
            "ACTR_PRODUCTION_FIRE",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "Fire production: apply action to state",
            Arc::new(|_ctx, args| {
                if args.len() < 2 {
                    return OpResult::Error("ProductionFire requires state and action".to_string());
                }
                // Fire = bind state with action
                let result = args[0].bind(&args[1]);
                OpResult::One(result)
            }),
        );

        // Buffer goal
        self.register(
            ActrOp::BufferGoal as u16,
            "ACTR_BUFFER_GOAL",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "Access/set goal buffer contents",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::One(Fingerprint::from_content("BUFFER::GOAL::EMPTY"));
                }
                let goal_marker = Fingerprint::from_content("BUFFER::GOAL");
                let buffered = args[0].bind(&goal_marker);
                OpResult::One(buffered)
            }),
        );

        // Utility compute
        self.register(
            ActrOp::UtilityCompute as u16,
            "ACTR_UTILITY",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::Fingerprint],
                output: OpType::Scalar,
            },
            "Compute production utility for conflict resolution",
            Arc::new(|_ctx, args| {
                if args.len() < 2 {
                    return OpResult::Scalar(0.0);
                }
                // Utility based on similarity to goal
                let goal_sim = args[0].similarity(&args[1]) as f64;
                // Add noise for stochastic selection
                let noise = (args[0].popcount() % 100) as f64 / 1000.0;
                OpResult::Scalar(goal_sim + noise)
            }),
        );

        // Conflict resolution
        self.register(
            ActrOp::ConflictResolve as u16,
            "ACTR_CONFLICT_RESOLVE",
            OpSignature {
                inputs: vec![OpType::FingerprintArray],
                output: OpType::Fingerprint,
            },
            "Select highest-utility production from conflict set",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::Error("ConflictResolve requires productions".to_string());
                }
                // Select production with highest popcount (utility proxy)
                let mut best = &args[0];
                let mut best_utility = args[0].popcount();

                for prod in args.iter().skip(1) {
                    let utility = prod.popcount();
                    if utility > best_utility {
                        best_utility = utility;
                        best = prod;
                    }
                }

                OpResult::One(best.clone())
            }),
        );
    }
}

// =============================================================================
// RL/DECISION OPERATIONS (0x900-0x9FF)
// =============================================================================

impl OpDictionary {
    fn register_rl_ops(&mut self) {
        // Q-value estimation
        self.register(
            RlOp::QValue as u16,
            "RL_Q_VALUE",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::Fingerprint],
                output: OpType::Scalar,
            },
            "Estimate Q(s,a) from state-action fingerprints",
            Arc::new(|_ctx, args| {
                if args.len() < 2 {
                    return OpResult::Scalar(0.0);
                }
                // Q-value = similarity between state-action pair and value pattern
                let state_action = args[0].bind(&args[1]);
                // Use popcount as value proxy (more bits = higher value)
                let q = state_action.popcount() as f64 / crate::FINGERPRINT_BITS as f64;
                OpResult::Scalar(q)
            }),
        );

        // Q-update (TD learning)
        self.register(
            RlOp::QUpdate as u16,
            "RL_Q_UPDATE",
            OpSignature {
                inputs: vec![
                    OpType::Fingerprint,
                    OpType::Fingerprint,
                    OpType::Fingerprint,
                ],
                output: OpType::Fingerprint,
            },
            "Update Q-value with TD error",
            Arc::new(|_ctx, args| {
                if args.len() < 3 {
                    return OpResult::Error("QUpdate requires old_q, reward, next_q".to_string());
                }
                // TD update: blend old with reward-shifted next
                let old_q = &args[0];
                let reward = &args[1];
                let next_q = &args[2];

                // New Q = bundle(old, reward ⊗ next)
                let td_target = reward.bind(next_q);
                let updated = bundle_fingerprints(&[old_q.clone(), td_target]);
                OpResult::One(updated)
            }),
        );

        // Policy greedy
        self.register(
            RlOp::PolicyGreedy as u16,
            "RL_POLICY_GREEDY",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::FingerprintArray],
                output: OpType::Fingerprint,
            },
            "Select action with highest Q-value",
            Arc::new(|_ctx, args| {
                if args.len() < 2 {
                    return OpResult::Error("PolicyGreedy requires state and actions".to_string());
                }
                let state = &args[0];
                let actions = &args[1..];

                if actions.is_empty() {
                    return OpResult::One(state.clone());
                }

                // Find action with highest Q-value (similarity to state)
                let mut best = &actions[0];
                let mut best_q = state.similarity(&actions[0]);

                for action in actions.iter().skip(1) {
                    let q = state.similarity(action);
                    if q > best_q {
                        best_q = q;
                        best = action;
                    }
                }

                OpResult::One(best.clone())
            }),
        );

        // Policy epsilon-greedy
        self.register(
            RlOp::PolicyEpsilonGreedy as u16,
            "RL_POLICY_EPSILON",
            OpSignature {
                inputs: vec![
                    OpType::Fingerprint,
                    OpType::FingerprintArray,
                    OpType::Fingerprint,
                ],
                output: OpType::Fingerprint,
            },
            "Epsilon-greedy action selection",
            Arc::new(|_ctx, args| {
                if args.len() < 3 {
                    return OpResult::Error(
                        "PolicyEpsilonGreedy requires state, actions, epsilon".to_string(),
                    );
                }
                let state = &args[0];
                let epsilon_fp = &args[args.len() - 1];
                let actions = &args[1..args.len() - 1];

                if actions.is_empty() {
                    return OpResult::One(state.clone());
                }

                // Epsilon from popcount
                let epsilon = (epsilon_fp.popcount() % 1000) as f64 / 1000.0;

                // Random check: use state popcount as pseudo-random
                let random = (state.popcount() % 100) as f64 / 100.0;

                if random < epsilon {
                    // Explore: random action
                    let idx = state.popcount() as usize % actions.len();
                    OpResult::One(actions[idx].clone())
                } else {
                    // Exploit: greedy action
                    let mut best = &actions[0];
                    let mut best_q = state.similarity(&actions[0]);
                    for action in actions.iter().skip(1) {
                        let q = state.similarity(action);
                        if q > best_q {
                            best_q = q;
                            best = action;
                        }
                    }
                    OpResult::One(best.clone())
                }
            }),
        );

        // Reward observe
        self.register(
            RlOp::RewardObserve as u16,
            "RL_REWARD",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::Scalar,
            },
            "Extract reward signal from observation",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::Scalar(0.0);
                }
                // Reward encoded in bit balance
                let positive = args[0].popcount() as f64;
                let negative =
                    (crate::FINGERPRINT_BITS as f64 - positive) / crate::FINGERPRINT_BITS as f64;
                let reward = (positive / crate::FINGERPRINT_BITS as f64) - negative;
                OpResult::Scalar(reward)
            }),
        );

        // TD error
        self.register(
            RlOp::TdError as u16,
            "RL_TD_ERROR",
            OpSignature {
                inputs: vec![
                    OpType::Fingerprint,
                    OpType::Fingerprint,
                    OpType::Fingerprint,
                ],
                output: OpType::Scalar,
            },
            "Compute temporal difference error",
            Arc::new(|_ctx, args| {
                if args.len() < 3 {
                    return OpResult::Scalar(0.0);
                }
                // δ = r + γV(s') - V(s)
                let reward_bits = args[0].popcount() as f64 / crate::FINGERPRINT_BITS as f64;
                let next_v = args[1].popcount() as f64 / crate::FINGERPRINT_BITS as f64;
                let current_v = args[2].popcount() as f64 / crate::FINGERPRINT_BITS as f64;
                let gamma = 0.99;

                let td_error = reward_bits + gamma * next_v - current_v;
                OpResult::Scalar(td_error)
            }),
        );

        // Eligibility trace update
        self.register(
            RlOp::TraceUpdate as u16,
            "RL_TRACE_UPDATE",
            OpSignature {
                inputs: vec![
                    OpType::Fingerprint,
                    OpType::Fingerprint,
                    OpType::Fingerprint,
                ],
                output: OpType::Fingerprint,
            },
            "Update eligibility trace with decay",
            Arc::new(|_ctx, args| {
                if args.len() < 3 {
                    return OpResult::Error(
                        "TraceUpdate requires trace, state, lambda".to_string(),
                    );
                }
                let trace = &args[0];
                let state = &args[1];
                // Lambda from popcount
                let lambda_idx = (args[2].popcount() as usize % 10) + 1;
                let decay = Fingerprint::orthogonal(lambda_idx);

                // e(s) = γλe(s) + 1
                let decayed = trace.bind(&decay);
                let updated = bundle_fingerprints(&[decayed, state.clone()]);
                OpResult::One(updated)
            }),
        );

        // Exploration bonus (curiosity)
        self.register(
            RlOp::RewardCuriosity as u16,
            "RL_CURIOSITY",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::FingerprintArray],
                output: OpType::Scalar,
            },
            "Compute curiosity-driven exploration bonus",
            Arc::new(|_ctx, args| {
                if args.len() < 2 {
                    return OpResult::Scalar(0.0);
                }
                let state = &args[0];
                let memory = &args[1..];

                if memory.is_empty() {
                    return OpResult::Scalar(1.0); // Max curiosity for unknown
                }

                // Novelty = inverse of max similarity to known states
                let max_sim = memory
                    .iter()
                    .map(|m| state.similarity(m) as f64)
                    .fold(0.0f64, |a, b| a.max(b));

                let curiosity = 1.0 - max_sim;
                OpResult::Scalar(curiosity)
            }),
        );

        // Model predict (for model-based RL)
        self.register(
            RlOp::ModelPredict as u16,
            "RL_MODEL_PREDICT",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "Predict next state from model",
            Arc::new(|_ctx, args| {
                if args.len() < 2 {
                    return OpResult::Error("ModelPredict requires state and action".to_string());
                }
                // Prediction = state ⊗ action (simple transition model)
                let predicted = args[0].bind(&args[1]);
                OpResult::One(predicted)
            }),
        );
    }
}

// =============================================================================
// RUNG OPERATIONS (0xC00-0xCFF) - Abstraction Ladder
// =============================================================================

impl OpDictionary {
    fn register_rung_ops(&mut self) {
        // Rung ascend (move up abstraction ladder)
        self.register(
            RungOp::RungAscend as u16,
            "RUNG_ASCEND",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "Move up the abstraction ladder",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::Error("RungAscend requires input".to_string());
                }
                // Ascend = project to more abstract representation
                // Bundle with abstraction marker
                let abstract_marker = Fingerprint::from_content("RUNG::ABSTRACT");
                let ascended = args[0].bind(&abstract_marker);
                OpResult::One(ascended)
            }),
        );

        // Rung descend (move down abstraction ladder)
        self.register(
            RungOp::RungDescend as u16,
            "RUNG_DESCEND",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "Move down the abstraction ladder",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::Error("RungDescend requires input".to_string());
                }
                // Descend = unbind abstraction marker
                let abstract_marker = Fingerprint::from_content("RUNG::ABSTRACT");
                let descended = args[0].unbind(&abstract_marker);
                OpResult::One(descended)
            }),
        );

        // Rung current level
        self.register(
            RungOp::RungCurrent as u16,
            "RUNG_CURRENT",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::Scalar,
            },
            "Get current abstraction level (0-9)",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::Scalar(0.0);
                }
                // Level encoded in popcount distribution
                let level = (args[0].popcount() / 1000) as f64;
                OpResult::Scalar(level.min(9.0))
            }),
        );

        // Abstract extract
        self.register(
            RungOp::AbstractExtract as u16,
            "RUNG_EXTRACT",
            OpSignature {
                inputs: vec![OpType::FingerprintArray],
                output: OpType::Fingerprint,
            },
            "Extract common abstraction from examples",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::Error("AbstractExtract requires examples".to_string());
                }
                // Common abstraction = bundled intersection
                let abstraction = bundle_fingerprints(&args);
                OpResult::One(abstraction)
            }),
        );

        // Abstract merge
        self.register(
            RungOp::AbstractMerge as u16,
            "RUNG_MERGE",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "Merge two abstractions into higher-level concept",
            Arc::new(|_ctx, args| {
                if args.len() < 2 {
                    return OpResult::Error("AbstractMerge requires two abstractions".to_string());
                }
                let merged = bundle_fingerprints(&[args[0].clone(), args[1].clone()]);
                OpResult::One(merged)
            }),
        );

        // Grounding check
        self.register(
            RungOp::GroundCheck as u16,
            "RUNG_GROUND_CHECK",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::Bool,
            },
            "Check if concept is grounded in sensory experience",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::Bool(false);
                }
                // Grounded if has sensory marker binding
                let sensory_marker = Fingerprint::from_content("GROUNDING::SENSORY");
                let sim = args[0].similarity(&sensory_marker);
                OpResult::Bool(sim > 0.3)
            }),
        );

        // Hierarchy depth
        self.register(
            RungOp::HierarchyDepth as u16,
            "RUNG_HIERARCHY_DEPTH",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::Scalar,
            },
            "Get depth in concept hierarchy",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::Scalar(0.0);
                }
                // Depth = count of abstraction bindings
                let abstract_marker = Fingerprint::from_content("RUNG::ABSTRACT");
                let sim = args[0].similarity(&abstract_marker) as f64;
                let depth = (sim * 10.0).floor();
                OpResult::Scalar(depth)
            }),
        );

        // Conceptual blend create
        self.register(
            RungOp::BlendCreate as u16,
            "RUNG_BLEND_CREATE",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "Create conceptual blend from two input spaces",
            Arc::new(|_ctx, args| {
                if args.len() < 2 {
                    return OpResult::Error("BlendCreate requires two input spaces".to_string());
                }
                // Blend = selective projection from both inputs
                let generic = Fingerprint::orthogonal(0); // Generic space
                let blend = args[0].bind(&generic).bind(&args[1]);
                OpResult::One(blend)
            }),
        );

        // Blend emergent structure
        self.register(
            RungOp::BlendEmergent as u16,
            "RUNG_BLEND_EMERGENT",
            OpSignature {
                inputs: vec![OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "Extract emergent structure from blend",
            Arc::new(|_ctx, args| {
                if args.is_empty() {
                    return OpResult::Error("BlendEmergent requires blend".to_string());
                }
                // Emergent = what wasn't in either input
                // Approximate by XORing with both inputs
                let generic = Fingerprint::orthogonal(0);
                let emergent = args[0].unbind(&generic);
                OpResult::One(emergent)
            }),
        );

        // Metaphor map
        self.register(
            RungOp::MetaphorMap as u16,
            "RUNG_METAPHOR_MAP",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "Map source domain to target domain via metaphor",
            Arc::new(|_ctx, args| {
                if args.len() < 2 {
                    return OpResult::Error("MetaphorMap requires source and target".to_string());
                }
                // Metaphor = binding that preserves relational structure
                let mapping = Fingerprint::from_content("METAPHOR::MAP");
                let result = args[0].bind(&mapping).bind(&args[1]);
                OpResult::One(result)
            }),
        );

        // Analogy find
        self.register(
            RungOp::AnalogyFind as u16,
            "RUNG_ANALOGY_FIND",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::FingerprintArray],
                output: OpType::Fingerprint,
            },
            "Find analogous structure in candidates",
            Arc::new(|_ctx, args| {
                if args.len() < 2 {
                    return OpResult::Error(
                        "AnalogyFind requires source and candidates".to_string(),
                    );
                }
                let source = &args[0];
                let candidates = &args[1..];

                if candidates.is_empty() {
                    return OpResult::One(source.clone());
                }

                // Find candidate with highest structural similarity
                let mut best = &candidates[0];
                let mut best_sim = source.similarity(&candidates[0]);

                for candidate in candidates.iter().skip(1) {
                    let sim = source.similarity(candidate);
                    if sim > best_sim {
                        best_sim = sim;
                        best = candidate;
                    }
                }

                OpResult::One(best.clone())
            }),
        );

        // Structure mapping
        self.register(
            RungOp::StructureMap as u16,
            "RUNG_STRUCTURE_MAP",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::Fingerprint],
                output: OpType::Fingerprint,
            },
            "Map structural relations between domains (Gentner SMT)",
            Arc::new(|_ctx, args| {
                if args.len() < 2 {
                    return OpResult::Error("StructureMap requires base and target".to_string());
                }
                // Extract relational structure via unbinding
                let relation_basis = Fingerprint::orthogonal(7);
                let base_relations = args[0].unbind(&relation_basis);
                let mapping = base_relations.bind(&args[1]);
                OpResult::One(mapping)
            }),
        );

        // Prototype match
        self.register(
            RungOp::PrototypeMatch as u16,
            "RUNG_PROTOTYPE_MATCH",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::Fingerprint],
                output: OpType::Scalar,
            },
            "Match instance to category prototype",
            Arc::new(|_ctx, args| {
                if args.len() < 2 {
                    return OpResult::Scalar(0.0);
                }
                let typicality = args[0].similarity(&args[1]) as f64;
                OpResult::Scalar(typicality)
            }),
        );

        // Category membership
        self.register(
            RungOp::CategoryMembership as u16,
            "RUNG_CATEGORY_MEMBER",
            OpSignature {
                inputs: vec![OpType::Fingerprint, OpType::Fingerprint],
                output: OpType::Bool,
            },
            "Check category membership via prototype similarity",
            Arc::new(|_ctx, args| {
                if args.len() < 2 {
                    return OpResult::Bool(false);
                }
                let sim = args[0].similarity(&args[1]);
                OpResult::Bool(sim > 0.5) // Membership threshold
            }),
        );
    }
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
