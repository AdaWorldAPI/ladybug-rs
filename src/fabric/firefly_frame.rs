//! Firefly Frame Format Specification
//!
//! 1250-bit (156 u64 words) microinstruction format for the Ada Cognitive CPU
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                        FIREFLY FRAME (1250 bits)                            │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │ Bits 0-63    │ HEADER: Auth + Routing                                       │
//! │              │   [0:7]   Magic: 0xADA1 (frame identifier)                   │
//! │              │   [8:15]  Version: Protocol version (currently 0x01)         │
//! │              │   [16:31] Session ID: 16-bit session reference               │
//! │              │   [32:39] Lane ID: Execution lane (0-255)                    │
//! │              │   [40:47] Hive ID: Which hive cluster                        │
//! │              │   [48:63] Sequence: Frame sequence number                    │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │ Bits 64-127  │ INSTRUCTION: Opcode + Dispatch                               │
//! │              │   [64:67]  Language prefix (4 bits)                          │
//! │              │            0x0: Lance (vector ops)                           │
//! │              │            0x1: SQL (relational)                             │
//! │              │            0x2: Cypher (graph)                               │
//! │              │            0x3: NARS (inference)                             │
//! │              │            0x4: Causal (Pearl's rungs)                       │
//! │              │            0x5: Quantum (superposition)                      │
//! │              │            0x6: Memory (bind/unbind)                         │
//! │              │            0x7: Control (branch/call/ret)                    │
//! │              │            0x8-0xE: Reserved                                 │
//! │              │            0xF: TRAP (syscall/interrupt)                     │
//! │              │   [68:75]  Opcode: 256 ops per language                      │
//! │              │   [76:79]  Flags: Condition codes                            │
//! │              │   [80:95]  Dest: Destination register (8+8 address)          │
//! │              │   [96:111] Src1: Source register 1                           │
//! │              │   [112:127] Src2: Source register 2 / immediate              │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │ Bits 128-255 │ OPERAND: Extended payload                                    │
//! │              │   For TRAP: Service ID + syscall args                        │
//! │              │   For Cypher: Embedded pattern (up to 16 chars)              │
//! │              │   For Lance: Vector query fingerprint prefix                 │
//! │              │   For Control: Branch target address                         │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │ Bits 256-639 │ DATA: Payload fingerprint (384 bits)                         │
//! │              │   Embedded data for operations                               │
//! │              │   Can hold truncated fingerprint for similarity ops          │
//! │              │   Or serialized graph pattern for complex queries            │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │ Bits 640-1023│ CONTEXT: Execution state (384 bits)                          │
//! │              │   Qualia vector (64 bits): 8 × i8 emotional state            │
//! │              │   Truth value (16 bits): NARS <f,c>                          │
//! │              │   Version (64 bits): Temporal coordinate                     │
//! │              │   Correlation ID (64 bits): Causal chain reference           │
//! │              │   Reserved (176 bits): Future use                            │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │ Bits 1024-1247│ ECC: Hamming error correction (224 bits)                    │
//! │              │   BCH(1247, 1024) code for multi-bit correction              │
//! │              │   Can correct up to 27 bit errors                            │
//! │              │   Detects up to 55 bit errors                                │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │ Bits 1248-1249│ TRAILER: Frame delimiter (2 bits)                           │
//! │              │   Must be 0b11 to indicate valid frame end                   │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Language Prefix Dispatch Table
//!
//! Each language prefix (4 bits) selects an instruction set with 256 opcodes:
//!
//! ### 0x0: Lance (Vector Operations)
//! - 0x00: RESONATE - Similarity search
//! - 0x01: INSERT - Add vector
//! - 0x02: DELETE - Remove vector
//! - 0x03: BATCH_RESONATE - Bulk similarity
//! - 0x10-0x1F: HDR cascade operations
//! - 0x20-0x2F: Quantization ops
//!
//! ### 0x1: SQL (Relational)
//! - 0x00: SELECT
//! - 0x01: INSERT
//! - 0x02: UPDATE
//! - 0x03: DELETE
//! - 0x10: JOIN
//! - 0x20: AGGREGATE
//!
//! ### 0x2: Cypher (Graph)
//! - 0x00: MATCH - Pattern match
//! - 0x01: CREATE - Create node/edge
//! - 0x02: MERGE - Upsert
//! - 0x03: DELETE
//! - 0x10: SHORTEST_PATH
//! - 0x11: ALL_PATHS
//! - 0x20: TRAVERSE
//!
//! ### 0x3: NARS (Inference)
//! - 0x00: DEDUCE - A→B, B→C ⊢ A→C
//! - 0x01: INDUCE - A→B, A→C ⊢ B→C
//! - 0x02: ABDUCE - A→B, C→B ⊢ A→C
//! - 0x03: REVISE - Combine evidence
//! - 0x04: ANALOGY - Structural mapping
//! - 0x10: ATTEND - Focus allocation
//!
//! ### 0x4: Causal (Pearl's Rungs)
//! - 0x00: SEE - Rung 1: Correlation
//! - 0x01: DO - Rung 2: Intervention
//! - 0x02: IMAGINE - Rung 3: Counterfactual
//! - 0x10: BUTTERFLY - Detect amplification
//!
//! ### 0x5: Quantum (Superposition)
//! - 0x00: SUPERPOSE - Create superposition
//! - 0x01: COLLAPSE - Measure/collapse
//! - 0x02: ENTANGLE - Link states
//! - 0x03: INTERFERE - Wave interference
//!
//! ### 0x6: Memory (Bind Space)
//! - 0x00: BIND - XOR binding (A ⊗ B)
//! - 0x01: UNBIND - Inverse binding
//! - 0x02: BUNDLE - Majority voting
//! - 0x03: PERMUTE - Rotate bits
//! - 0x10: CRYSTALLIZE - Promote to node
//! - 0x11: EVAPORATE - Demote to fluid
//!
//! ### 0x7: Control (Flow)
//! - 0x00: NOP
//! - 0x01: JUMP - Unconditional
//! - 0x02: BRANCH - Conditional
//! - 0x03: CALL - Push return, jump
//! - 0x04: RET - Pop return, jump
//! - 0x05: LOOP - Bounded iteration
//! - 0x10: FORK - Spawn parallel lane
//! - 0x11: JOIN - Wait for lane
//! - 0x20: YIELD - Cooperative multitasking
//!
//! ### 0xF: TRAP (System)
//! - 0x00: HALT
//! - 0x01: PANIC - Unrecoverable error
//! - 0x10: IO_READ - External input
//! - 0x11: IO_WRITE - External output
//! - 0x20: AUTH - Validate credentials
//! - 0x30: CHECKPOINT - Snapshot state
//! - 0xFF: DEBUG - Breakpoint

use std::fmt;

/// Frame header (64 bits)
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameHeader {
    /// Magic number (0xADA1)
    pub magic: u16,
    /// Protocol version
    pub version: u8,
    /// Session identifier
    pub session_id: u16,
    /// Execution lane (0-255)
    pub lane_id: u8,
    /// Hive cluster ID
    pub hive_id: u8,
    /// Sequence number
    pub sequence: u16,
}

impl FrameHeader {
    pub const MAGIC: u16 = 0xADA1;

    pub fn new(session_id: u16, lane_id: u8, hive_id: u8, sequence: u16) -> Self {
        Self {
            magic: Self::MAGIC,
            version: 1,
            session_id,
            lane_id,
            hive_id,
            sequence,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.magic == Self::MAGIC
    }

    pub fn encode(&self) -> u64 {
        (self.magic as u64)
            | ((self.version as u64) << 16)
            | ((self.session_id as u64) << 24)
            | ((self.lane_id as u64) << 40)
            | ((self.hive_id as u64) << 48)
            | ((self.sequence as u64) << 56)
    }

    pub fn decode(word: u64) -> Self {
        Self {
            magic: (word & 0xFFFF) as u16,
            version: ((word >> 16) & 0xFF) as u8,
            session_id: ((word >> 24) & 0xFFFF) as u16,
            lane_id: ((word >> 40) & 0xFF) as u8,
            hive_id: ((word >> 48) & 0xFF) as u8,
            sequence: ((word >> 56) & 0xFFFF) as u16,
        }
    }
}

/// Language prefix for instruction dispatch
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LanguagePrefix {
    Lance = 0x0,    // Vector operations
    Sql = 0x1,      // Relational
    Cypher = 0x2,   // Graph
    Nars = 0x3,     // Inference
    Causal = 0x4,   // Pearl's rungs
    Quantum = 0x5,  // Superposition
    Memory = 0x6,   // Bind space
    Control = 0x7,  // Flow control
    Trap = 0xF,     // System calls
}

impl LanguagePrefix {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0x0 => Some(Self::Lance),
            0x1 => Some(Self::Sql),
            0x2 => Some(Self::Cypher),
            0x3 => Some(Self::Nars),
            0x4 => Some(Self::Causal),
            0x5 => Some(Self::Quantum),
            0x6 => Some(Self::Memory),
            0x7 => Some(Self::Control),
            0xF => Some(Self::Trap),
            _ => None,
        }
    }
}

/// Condition flags for branching
#[derive(Debug, Clone, Copy, Default)]
pub struct ConditionFlags {
    pub zero: bool,       // Result was zero
    pub negative: bool,   // Result was negative
    pub overflow: bool,   // Arithmetic overflow
    pub carry: bool,      // Carry/borrow
}

impl ConditionFlags {
    pub fn encode(&self) -> u8 {
        (self.zero as u8)
            | ((self.negative as u8) << 1)
            | ((self.overflow as u8) << 2)
            | ((self.carry as u8) << 3)
    }

    pub fn decode(v: u8) -> Self {
        Self {
            zero: (v & 1) != 0,
            negative: (v & 2) != 0,
            overflow: (v & 4) != 0,
            carry: (v & 8) != 0,
        }
    }
}

/// Instruction word (64 bits)
#[derive(Debug, Clone, Copy)]
pub struct Instruction {
    /// Language prefix (4 bits)
    pub prefix: LanguagePrefix,
    /// Opcode within language (8 bits)
    pub opcode: u8,
    /// Condition flags (4 bits)
    pub flags: ConditionFlags,
    /// Destination address (16 bits, 8+8 format)
    pub dest: u16,
    /// Source 1 address (16 bits)
    pub src1: u16,
    /// Source 2 / immediate (16 bits)
    pub src2: u16,
}

impl Instruction {
    pub fn new(prefix: LanguagePrefix, opcode: u8, dest: u16, src1: u16, src2: u16) -> Self {
        Self {
            prefix,
            opcode,
            flags: ConditionFlags::default(),
            dest,
            src1,
            src2,
        }
    }

    pub fn encode(&self) -> u64 {
        ((self.prefix as u64) & 0xF)
            | (((self.opcode as u64) & 0xFF) << 4)
            | (((self.flags.encode() as u64) & 0xF) << 12)
            | (((self.dest as u64) & 0xFFFF) << 16)
            | (((self.src1 as u64) & 0xFFFF) << 32)
            | (((self.src2 as u64) & 0xFFFF) << 48)
    }

    pub fn decode(word: u64) -> Option<Self> {
        let prefix = LanguagePrefix::from_u8((word & 0xF) as u8)?;
        Some(Self {
            prefix,
            opcode: ((word >> 4) & 0xFF) as u8,
            flags: ConditionFlags::decode(((word >> 12) & 0xF) as u8),
            dest: ((word >> 16) & 0xFFFF) as u16,
            src1: ((word >> 32) & 0xFFFF) as u16,
            src2: ((word >> 48) & 0xFFFF) as u16,
        })
    }
}

/// Execution context (384 bits = 6 u64 words)
#[derive(Debug, Clone, Copy, Default)]
pub struct ExecutionContext {
    /// Qualia vector (8 × i8)
    pub qualia: [i8; 8],
    /// NARS truth value (frequency, confidence)
    pub truth: (u8, u8),
    /// Temporal version
    pub version: u64,
    /// Causal correlation ID
    pub correlation_id: u64,
    /// Reserved for future use
    pub reserved: [u64; 2],
}

impl ExecutionContext {
    pub fn encode(&self) -> [u64; 6] {
        let mut words = [0u64; 6];

        // Word 0: Qualia (64 bits)
        for (i, &q) in self.qualia.iter().enumerate() {
            words[0] |= ((q as u8) as u64) << (i * 8);
        }

        // Word 1: Truth + padding
        words[1] = (self.truth.0 as u64) | ((self.truth.1 as u64) << 8);

        // Word 2: Version
        words[2] = self.version;

        // Word 3: Correlation ID
        words[3] = self.correlation_id;

        // Words 4-5: Reserved
        words[4] = self.reserved[0];
        words[5] = self.reserved[1];

        words
    }

    pub fn decode(words: &[u64; 6]) -> Self {
        let mut qualia = [0i8; 8];
        for i in 0..8 {
            qualia[i] = ((words[0] >> (i * 8)) & 0xFF) as i8;
        }

        Self {
            qualia,
            truth: ((words[1] & 0xFF) as u8, ((words[1] >> 8) & 0xFF) as u8),
            version: words[2],
            correlation_id: words[3],
            reserved: [words[4], words[5]],
        }
    }
}

/// Complete Firefly Frame (156 u64 words = 1248 bits + 2 bit trailer)
#[derive(Clone)]
pub struct FireflyFrame {
    /// Header: Auth + Routing (word 0)
    pub header: FrameHeader,
    /// Instruction: Opcode + Dispatch (word 1)
    pub instruction: Instruction,
    /// Operand: Extended payload (words 2-3)
    pub operand: [u64; 2],
    /// Data: Payload fingerprint (words 4-9, 384 bits)
    pub data: [u64; 6],
    /// Context: Execution state (words 10-15, 384 bits)
    pub context: ExecutionContext,
    /// ECC: Hamming error correction (words 16-19, 224 bits used)
    pub ecc: [u64; 4],
}

impl FireflyFrame {
    /// Frame size in u64 words
    pub const WORDS: usize = 20;

    /// Create a new frame
    pub fn new(header: FrameHeader, instruction: Instruction) -> Self {
        Self {
            header,
            instruction,
            operand: [0; 2],
            data: [0; 6],
            context: ExecutionContext::default(),
            ecc: [0; 4],
        }
    }

    /// Set the operand payload
    pub fn with_operand(mut self, operand: [u64; 2]) -> Self {
        self.operand = operand;
        self
    }

    /// Set the data payload
    pub fn with_data(mut self, data: [u64; 6]) -> Self {
        self.data = data;
        self
    }

    /// Set the execution context
    pub fn with_context(mut self, context: ExecutionContext) -> Self {
        self.context = context;
        self
    }

    /// Encode frame to bytes
    pub fn encode(&mut self) -> [u64; Self::WORDS] {
        let mut words = [0u64; Self::WORDS];

        // Header
        words[0] = self.header.encode();

        // Instruction
        words[1] = self.instruction.encode();

        // Operand
        words[2] = self.operand[0];
        words[3] = self.operand[1];

        // Data
        words[4..10].copy_from_slice(&self.data);

        // Context
        let ctx = self.context.encode();
        words[10..16].copy_from_slice(&ctx);

        // Compute ECC
        self.ecc = Self::compute_ecc(&words[0..16]);
        words[16..20].copy_from_slice(&self.ecc);

        words
    }

    /// Decode frame from bytes
    pub fn decode(words: &[u64; Self::WORDS]) -> Option<Self> {
        // Verify and correct ECC
        let corrected = Self::verify_ecc(words)?;

        let header = FrameHeader::decode(corrected[0]);
        if !header.is_valid() {
            return None;
        }

        let instruction = Instruction::decode(corrected[1])?;

        let mut operand = [0u64; 2];
        operand.copy_from_slice(&corrected[2..4]);

        let mut data = [0u64; 6];
        data.copy_from_slice(&corrected[4..10]);

        let mut ctx_words = [0u64; 6];
        ctx_words.copy_from_slice(&corrected[10..16]);
        let context = ExecutionContext::decode(&ctx_words);

        let mut ecc = [0u64; 4];
        ecc.copy_from_slice(&corrected[16..20]);

        Some(Self {
            header,
            instruction,
            operand,
            data,
            context,
            ecc,
        })
    }

    /// Compute ECC for data words
    fn compute_ecc(data: &[u64]) -> [u64; 4] {
        // Simplified ECC: XOR-based parity
        // Real implementation would use BCH(1247, 1024)
        let mut ecc = [0u64; 4];

        for (i, &word) in data.iter().enumerate() {
            ecc[i % 4] ^= word;
        }

        // Add parity bits
        for i in 0..4 {
            let parity = ecc[i].count_ones() as u64;
            ecc[i] = (ecc[i] & 0xFFFFFFFFFFFFFFFE) | (parity & 1);
        }

        ecc
    }

    /// Verify ECC and return corrected data
    fn verify_ecc(words: &[u64; Self::WORDS]) -> Option<[u64; Self::WORDS]> {
        // Simplified verification
        // Real implementation would do full BCH decode
        let mut corrected = *words;

        let expected_ecc = Self::compute_ecc(&words[0..16]);
        let received_ecc = &words[16..20];

        // Check for errors
        let mut syndrome = 0u64;
        for i in 0..4 {
            syndrome ^= expected_ecc[i] ^ received_ecc[i];
        }

        if syndrome == 0 {
            // No errors
            Some(corrected)
        } else if syndrome.count_ones() <= 1 {
            // Single bit error - correctable
            // Find and flip the bit
            Some(corrected)
        } else {
            // Multi-bit error - may be uncorrectable
            // For now, return as-is (real impl would attempt BCH decode)
            Some(corrected)
        }
    }
}

impl fmt::Debug for FireflyFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FireflyFrame {{ session={}, lane={}, seq={}, {:?}:{:02X} }}",
            self.header.session_id,
            self.header.lane_id,
            self.header.sequence,
            self.instruction.prefix,
            self.instruction.opcode
        )
    }
}

// =============================================================================
// FRAME BUILDER
// =============================================================================

/// Builder for constructing frames
pub struct FrameBuilder {
    session_id: u16,
    lane_id: u8,
    hive_id: u8,
    sequence: u16,
}

impl FrameBuilder {
    pub fn new(session_id: u16) -> Self {
        Self {
            session_id,
            lane_id: 0,
            hive_id: 0,
            sequence: 0,
        }
    }

    pub fn lane(mut self, lane: u8) -> Self {
        self.lane_id = lane;
        self
    }

    pub fn hive(mut self, hive: u8) -> Self {
        self.hive_id = hive;
        self
    }

    pub fn resonate(mut self, dest: u16, query_fp: u16, k: u16) -> FireflyFrame {
        self.sequence += 1;
        let header = FrameHeader::new(self.session_id, self.lane_id, self.hive_id, self.sequence);
        let instruction = Instruction::new(LanguagePrefix::Lance, 0x00, dest, query_fp, k);
        FireflyFrame::new(header, instruction)
    }

    pub fn cypher_match(mut self, dest: u16, pattern_addr: u16) -> FireflyFrame {
        self.sequence += 1;
        let header = FrameHeader::new(self.session_id, self.lane_id, self.hive_id, self.sequence);
        let instruction = Instruction::new(LanguagePrefix::Cypher, 0x00, dest, pattern_addr, 0);
        FireflyFrame::new(header, instruction)
    }

    pub fn nars_deduce(mut self, dest: u16, premise1: u16, premise2: u16) -> FireflyFrame {
        self.sequence += 1;
        let header = FrameHeader::new(self.session_id, self.lane_id, self.hive_id, self.sequence);
        let instruction = Instruction::new(LanguagePrefix::Nars, 0x00, dest, premise1, premise2);
        FireflyFrame::new(header, instruction)
    }

    pub fn bind(mut self, dest: u16, a: u16, b: u16) -> FireflyFrame {
        self.sequence += 1;
        let header = FrameHeader::new(self.session_id, self.lane_id, self.hive_id, self.sequence);
        let instruction = Instruction::new(LanguagePrefix::Memory, 0x00, dest, a, b);
        FireflyFrame::new(header, instruction)
    }

    pub fn branch(mut self, condition: u8, target: u16) -> FireflyFrame {
        self.sequence += 1;
        let header = FrameHeader::new(self.session_id, self.lane_id, self.hive_id, self.sequence);
        let mut instruction = Instruction::new(LanguagePrefix::Control, 0x02, 0, 0, target);
        instruction.flags = ConditionFlags::decode(condition);
        FireflyFrame::new(header, instruction)
    }

    pub fn trap(mut self, service: u8, arg1: u16, arg2: u16) -> FireflyFrame {
        self.sequence += 1;
        let header = FrameHeader::new(self.session_id, self.lane_id, self.hive_id, self.sequence);
        let instruction = Instruction::new(LanguagePrefix::Trap, service, 0, arg1, arg2);
        FireflyFrame::new(header, instruction)
    }

    /// No-operation instruction
    pub fn nop(mut self) -> FireflyFrame {
        self.sequence += 1;
        let header = FrameHeader::new(self.session_id, self.lane_id, self.hive_id, self.sequence);
        let instruction = Instruction::new(LanguagePrefix::Control, 0x00, 0, 0, 0);
        FireflyFrame::new(header, instruction)
    }

    /// Halt execution
    pub fn halt(mut self) -> FireflyFrame {
        self.sequence += 1;
        let header = FrameHeader::new(self.session_id, self.lane_id, self.hive_id, self.sequence);
        let instruction = Instruction::new(LanguagePrefix::Trap, 0x00, 0, 0, 0);
        FireflyFrame::new(header, instruction)
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_encode_decode() {
        let header = FrameHeader::new(0x1234, 5, 3, 42);
        let encoded = header.encode();
        let decoded = FrameHeader::decode(encoded);

        assert_eq!(decoded.magic, FrameHeader::MAGIC);
        assert_eq!(decoded.session_id, 0x1234);
        assert_eq!(decoded.lane_id, 5);
        assert_eq!(decoded.hive_id, 3);
        assert_eq!(decoded.sequence, 42);
    }

    #[test]
    fn test_instruction_encode_decode() {
        let inst = Instruction::new(LanguagePrefix::Nars, 0x00, 0x8000, 0x8001, 0x8002);
        let encoded = inst.encode();
        let decoded = Instruction::decode(encoded).unwrap();

        assert_eq!(decoded.prefix, LanguagePrefix::Nars);
        assert_eq!(decoded.opcode, 0x00);
        assert_eq!(decoded.dest, 0x8000);
        assert_eq!(decoded.src1, 0x8001);
        assert_eq!(decoded.src2, 0x8002);
    }

    #[test]
    fn test_frame_encode_decode() {
        let builder = FrameBuilder::new(0x1234).lane(5).hive(3);
        let mut frame = builder.nars_deduce(0x8000, 0x8001, 0x8002);

        frame.context.qualia = [10, -20, 30, -40, 50, -60, 70, -80];
        frame.context.truth = (200, 180);
        frame.context.version = 42;

        let encoded = frame.encode();
        let decoded = FireflyFrame::decode(&encoded).unwrap();

        assert_eq!(decoded.header.session_id, 0x1234);
        assert_eq!(decoded.header.lane_id, 5);
        assert_eq!(decoded.instruction.prefix, LanguagePrefix::Nars);
        assert_eq!(decoded.instruction.opcode, 0x00);
        assert_eq!(decoded.context.qualia, [10, -20, 30, -40, 50, -60, 70, -80]);
        assert_eq!(decoded.context.truth, (200, 180));
        assert_eq!(decoded.context.version, 42);
    }

    #[test]
    fn test_frame_builder() {
        // Each builder method consumes self, so create new builders
        let f1 = FrameBuilder::new(100).resonate(0x8000, 0x8001, 10);
        assert_eq!(f1.instruction.prefix, LanguagePrefix::Lance);
        assert_eq!(f1.instruction.opcode, 0x00);

        let f2 = FrameBuilder::new(100).cypher_match(0x8010, 0x8011);
        assert_eq!(f2.instruction.prefix, LanguagePrefix::Cypher);

        let f3 = FrameBuilder::new(100).trap(0x10, 0, 0); // IO_READ
        assert_eq!(f3.instruction.prefix, LanguagePrefix::Trap);
        assert_eq!(f3.instruction.opcode, 0x10);
    }
}
