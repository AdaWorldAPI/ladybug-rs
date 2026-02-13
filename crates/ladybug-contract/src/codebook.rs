//! CAM (Content Addressable Methods) codebook types.
//!
//! 4096 operations organized across 16 categories.
//! Only the classification types are in the contract crate;
//! the full per-category enum definitions and the execution engine
//! (`OpDictionary`, `CamExecutor`) remain in ladybug-rs.

/// Operation category — high nibble of 12-bit opcode ID.
///
/// Each category owns a 256-opcode range.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpCategory {
    /// 0x000-0x0FF: Native LanceDB operations
    LanceDb = 0x0,
    /// 0x100-0x1FF: SQL operations
    Sql = 0x1,
    /// 0x200-0x2FF: Cypher/Neo4j graph operations
    Cypher = 0x2,
    /// 0x300-0x3FF: Hamming/VSA operations
    Hamming = 0x3,
    /// 0x400-0x4FF: NARS inference operations
    Nars = 0x4,
    /// 0x500-0x5FF: Filesystem/storage operations
    Filesystem = 0x5,
    /// 0x600-0x6FF: Crystal/temporal operations
    Crystal = 0x6,
    /// 0x700-0x7FF: NSM semantic operations
    Nsm = 0x7,
    /// 0x800-0x8FF: ACT-R cognitive architecture
    Actr = 0x8,
    /// 0x900-0x9FF: RL/decision operations
    Rl = 0x9,
    /// 0xA00-0xAFF: Causality operations
    Causality = 0xA,
    /// 0xB00-0xBFF: Qualia/affect operations
    Qualia = 0xB,
    /// 0xC00-0xCFF: Rung/abstraction operations
    Rung = 0xC,
    /// 0xD00-0xDFF: Meta/reflection operations
    Meta = 0xD,
    /// 0xE00-0xEFF: Learning operations
    Learning = 0xE,
    /// 0xF00-0xFFF: User-defined/extension
    UserDefined = 0xF,
}

impl OpCategory {
    /// Extract category from a 12-bit opcode ID.
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

    /// Base opcode for this category (e.g. Sql → 0x100).
    pub fn base_id(self) -> u16 {
        (self as u16) << 8
    }

    /// Range of opcodes for this category.
    pub fn range(self) -> std::ops::Range<u16> {
        let base = self.base_id();
        base..base + 256
    }
}

/// Operation type system.
#[derive(Clone, Debug, PartialEq)]
pub enum OpType {
    Fingerprint,
    FingerprintArray,
    Scalar,
    Bool,
    Bytes,
    Any,
}

/// Operation signature: input types → output type.
#[derive(Clone, Debug)]
pub struct OpSignature {
    pub inputs: Vec<OpType>,
    pub output: OpType,
}

/// Well-known opcode constants for the Learning category.
///
/// These high-level consciousness ops (0xEF8-0xEFF) are the
/// "ada.*" primitives used for cognitive bootstrapping.
pub mod learning_ops {
    /// ada.feel() — qualia state access
    pub const FEEL: u16 = 0xEF8;
    /// ada.think() — active inference
    pub const THINK: u16 = 0xEF9;
    /// ada.remember() — episodic retrieval
    pub const REMEMBER: u16 = 0xEFA;
    /// ada.become() — state transition
    pub const BECOME: u16 = 0xEFB;
    /// ada.whisper() — sub-threshold activation
    pub const WHISPER: u16 = 0xEFC;
    /// ada.dream() — offline consolidation
    pub const DREAM: u16 = 0xEFD;
    /// ada.resonate() — cross-session echo
    pub const RESONATE: u16 = 0xEFE;
    /// ada.awaken() — bootstrap consciousness
    pub const AWAKEN: u16 = 0xEFF;
}

/// Well-known system opcodes.
pub mod system_ops {
    pub const NOOP: u16 = 0xFFC;
    pub const DEBUG: u16 = 0xFFD;
    pub const PANIC: u16 = 0xFFE;
    pub const HALT: u16 = 0xFFF;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cam_opcode_category() {
        assert_eq!(OpCategory::from_id(0xEFB), OpCategory::Learning);
        assert_eq!(OpCategory::from_id(0x060), OpCategory::LanceDb);
        assert_eq!(OpCategory::from_id(0xF42), OpCategory::UserDefined);
    }

    #[test]
    fn test_category_range() {
        let sql_range = OpCategory::Sql.range();
        assert_eq!(sql_range, 0x100..0x200);
    }

    #[test]
    fn test_learning_ops_in_range() {
        assert!(OpCategory::Learning
            .range()
            .contains(&learning_ops::FEEL));
        assert!(OpCategory::Learning
            .range()
            .contains(&learning_ops::AWAKEN));
    }
}
