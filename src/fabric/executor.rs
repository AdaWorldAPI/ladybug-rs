//! Firefly Executor: The ALU of the Cognitive CPU
//!
//! Executes FireflyFrames locally against BindSpace.
//! No network transport - just pure instruction execution.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                         EXECUTION PIPELINE                                  │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │  1. FETCH   │ Read frame from instruction queue                             │
//! │  2. DECODE  │ Extract prefix, opcode, operands                              │
//! │  3. EXECUTE │ Dispatch to language runtime                                  │
//! │  4. WRITE   │ Store result in register file                                 │
//! │  5. COMMIT  │ Update program counter, check interrupts                      │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::storage::{BindSpace, Substrate, SubstrateConfig, Addr, FINGERPRINT_WORDS};
use crate::nars::TruthValue;
use super::firefly_frame::{
    FireflyFrame, Instruction, LanguagePrefix,
    ExecutionContext, ConditionFlags,
};

// =============================================================================
// REGISTER FILE
// =============================================================================

/// Register file: 65536 addressable locations (8+8 addressing)
///
/// Surface (0x00-0x0F): System registers, constants
/// Fluid (0x10-0x7F): Working memory, temporaries
/// Nodes (0x80-0xFF): Persistent storage
pub struct RegisterFile {
    /// The underlying bind space IS the register file
    bind_space: BindSpace,
    /// Substrate for vector operations
    substrate: Substrate,
    /// Access counters for LRU
    access_count: AtomicU64,
}

impl RegisterFile {
    pub fn new() -> Self {
        Self {
            bind_space: BindSpace::new(),
            substrate: Substrate::new(SubstrateConfig::default()),
            access_count: AtomicU64::new(0),
        }
    }

    /// Read fingerprint from address
    pub fn read(&self, addr: u16) -> Option<[u64; FINGERPRINT_WORDS]> {
        self.access_count.fetch_add(1, Ordering::Relaxed);
        self.bind_space.read(Addr(addr)).map(|n| n.fingerprint)
    }

    /// Write fingerprint to address
    pub fn write(&mut self, addr: u16, fingerprint: [u64; FINGERPRINT_WORDS]) {
        self.bind_space.write_at(Addr(addr), fingerprint);
    }

    /// Vector similarity search
    pub fn resonate(&self, query: &[u64; FINGERPRINT_WORDS], k: usize) -> Vec<(u16, f32)> {
        self.substrate.resonate(query, k)
            .into_iter()
            .map(|(addr, sim)| (addr.0, sim))
            .collect()
    }
}

impl Default for RegisterFile {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// EXECUTION RESULT
// =============================================================================

/// Result of executing a single instruction
#[derive(Debug, Clone)]
pub enum ExecResult {
    /// Success with optional return value
    Ok(Option<[u64; FINGERPRINT_WORDS]>),
    /// Similarity search results
    Resonate(Vec<(u16, f32)>),
    /// NARS inference result
    Truth(TruthValue),
    /// Branch taken (new PC)
    Branch(u16),
    /// Halt execution
    Halt,
    /// Error
    Error(String),
}

// =============================================================================
// EXECUTOR
// =============================================================================

/// The cognitive CPU executor
pub struct Executor {
    /// Register file (BindSpace + Substrate)
    pub registers: RegisterFile,
    /// Program counter
    pc: u16,
    /// Condition flags
    flags: ConditionFlags,
    /// Execution statistics
    stats: ExecutorStats,
    /// Interrupt pending
    interrupt_pending: Option<u8>,
}

#[derive(Debug, Clone, Default)]
pub struct ExecutorStats {
    pub instructions_executed: u64,
    pub cycles: u64,
    pub lance_ops: u64,
    pub sql_ops: u64,
    pub cypher_ops: u64,
    pub nars_ops: u64,
    pub causal_ops: u64,
    pub quantum_ops: u64,
    pub memory_ops: u64,
    pub control_ops: u64,
    pub trap_ops: u64,
    pub total_time: Duration,
}

impl Executor {
    pub fn new() -> Self {
        Self {
            registers: RegisterFile::new(),
            pc: 0x8000, // Start in node space
            flags: ConditionFlags::default(),
            stats: ExecutorStats::default(),
            interrupt_pending: None,
        }
    }

    /// Execute a single frame
    pub fn execute(&mut self, frame: &FireflyFrame) -> ExecResult {
        let start = Instant::now();
        self.stats.instructions_executed += 1;

        // Decode
        let inst = &frame.instruction;

        // Dispatch based on language prefix
        let result = match inst.prefix {
            LanguagePrefix::Lance => self.exec_lance(inst, &frame.data),
            LanguagePrefix::Sql => self.exec_sql(inst, &frame.operand),
            LanguagePrefix::Cypher => self.exec_cypher(inst, &frame.operand),
            LanguagePrefix::Nars => self.exec_nars(inst, &frame.context),
            LanguagePrefix::Causal => self.exec_causal(inst, &frame.context),
            LanguagePrefix::Quantum => self.exec_quantum(inst, &frame.data),
            LanguagePrefix::Memory => self.exec_memory(inst),
            LanguagePrefix::Control => self.exec_control(inst),
            LanguagePrefix::Trap => self.exec_trap(inst, &frame.operand),
        };

        self.stats.total_time += start.elapsed();
        self.stats.cycles += 1;

        result
    }

    /// Execute a program (sequence of frames)
    pub fn run(&mut self, program: &[FireflyFrame], max_cycles: u64) -> Vec<ExecResult> {
        let mut results = Vec::new();
        let mut cycle = 0;

        while cycle < max_cycles {
            // Check for halt or end of program
            let pc_idx = (self.pc.saturating_sub(0x8000)) as usize;
            if pc_idx >= program.len() {
                break;
            }

            // Check for pending interrupt
            if let Some(int) = self.interrupt_pending.take() {
                // Handle interrupt (simplified: just record it)
                results.push(ExecResult::Error(format!("Interrupt {}", int)));
                continue;
            }

            // Fetch and execute
            let frame = &program[pc_idx];
            let result = self.execute(frame);

            // Handle result
            match &result {
                ExecResult::Branch(target) => {
                    self.pc = *target;
                }
                ExecResult::Halt => {
                    results.push(result);
                    break;
                }
                ExecResult::Error(_) => {
                    results.push(result);
                    break;
                }
                _ => {
                    self.pc = self.pc.wrapping_add(1);
                }
            }

            results.push(result);
            cycle += 1;
        }

        results
    }

    // =========================================================================
    // LANCE: Vector Operations
    // =========================================================================

    fn exec_lance(&mut self, inst: &Instruction, _data: &[u64; 6]) -> ExecResult {
        self.stats.lance_ops += 1;

        match inst.opcode {
            // RESONATE: Similarity search
            0x00 => {
                // src1 = query address, src2 = k
                if let Some(query) = self.registers.read(inst.src1) {
                    let k = inst.src2 as usize;
                    let results = self.registers.resonate(&query, k.max(1));
                    ExecResult::Resonate(results)
                } else {
                    ExecResult::Error("RESONATE: query not found".into())
                }
            }

            // INSERT: Add vector to index
            0x01 => {
                if let Some(fp) = self.registers.read(inst.src1) {
                    self.registers.substrate.write(fp);
                    ExecResult::Ok(None)
                } else {
                    ExecResult::Error("INSERT: source not found".into())
                }
            }

            // HAMMING: Compute Hamming distance
            0x10 => {
                let a = self.registers.read(inst.src1);
                let b = self.registers.read(inst.src2);
                match (a, b) {
                    (Some(a), Some(b)) => {
                        let dist: u32 = a.iter()
                            .zip(b.iter())
                            .map(|(x, y)| (x ^ y).count_ones())
                            .sum();
                        // Store distance in dest (as fingerprint with popcount)
                        let mut result = [0u64; FINGERPRINT_WORDS];
                        result[0] = dist as u64;
                        self.registers.write(inst.dest, result);
                        self.flags.zero = dist == 0;
                        ExecResult::Ok(Some(result))
                    }
                    _ => ExecResult::Error("HAMMING: operand not found".into())
                }
            }

            _ => ExecResult::Error(format!("Lance opcode {:02X} not implemented", inst.opcode))
        }
    }

    // =========================================================================
    // SQL: Relational Operations (stub)
    // =========================================================================

    fn exec_sql(&mut self, inst: &Instruction, _operand: &[u64; 2]) -> ExecResult {
        self.stats.sql_ops += 1;
        // SQL would integrate with DataFusion
        ExecResult::Error(format!("SQL opcode {:02X} not yet implemented", inst.opcode))
    }

    // =========================================================================
    // CYPHER: Graph Operations (stub)
    // =========================================================================

    fn exec_cypher(&mut self, inst: &Instruction, _operand: &[u64; 2]) -> ExecResult {
        self.stats.cypher_ops += 1;
        // Cypher would integrate with graph traversal
        ExecResult::Error(format!("Cypher opcode {:02X} not yet implemented", inst.opcode))
    }

    // =========================================================================
    // NARS: Inference Operations
    // =========================================================================

    fn exec_nars(&mut self, inst: &Instruction, ctx: &ExecutionContext) -> ExecResult {
        self.stats.nars_ops += 1;

        // Extract truth values from context or registers
        let t1 = TruthValue::new(
            ctx.truth.0 as f32 / 255.0,
            ctx.truth.1 as f32 / 255.0,
        );

        match inst.opcode {
            // DEDUCE: A→B, B→C ⊢ A→C
            0x00 => {
                // Second truth value from src2 register (simplified)
                let t2 = TruthValue::new(0.8, 0.7); // Would read from register
                let result = t1.deduction(&t2);
                ExecResult::Truth(result)
            }

            // INDUCE: A→B, A→C ⊢ B→C
            0x01 => {
                let t2 = TruthValue::new(0.8, 0.7);
                let result = t1.induction(&t2);
                ExecResult::Truth(result)
            }

            // ABDUCE: A→B, C→B ⊢ A→C
            0x02 => {
                let t2 = TruthValue::new(0.8, 0.7);
                let result = t1.abduction(&t2);
                ExecResult::Truth(result)
            }

            // REVISE: Combine evidence
            0x03 => {
                let t2 = TruthValue::new(0.8, 0.7);
                let result = t1.revision(&t2);
                ExecResult::Truth(result)
            }

            // NEGATE
            0x04 => {
                let result = t1.negation();
                ExecResult::Truth(result)
            }

            _ => ExecResult::Error(format!("NARS opcode {:02X} not implemented", inst.opcode))
        }
    }

    // =========================================================================
    // CAUSAL: Pearl's Three Rungs
    // =========================================================================

    fn exec_causal(&mut self, inst: &Instruction, _ctx: &ExecutionContext) -> ExecResult {
        self.stats.causal_ops += 1;

        match inst.opcode {
            // SEE: Rung 1 - Correlation
            0x00 => {
                // Would query correlation store
                ExecResult::Ok(None)
            }

            // DO: Rung 2 - Intervention
            0x01 => {
                // Would perform intervention
                ExecResult::Ok(None)
            }

            // IMAGINE: Rung 3 - Counterfactual
            0x02 => {
                // Would fork world and simulate
                ExecResult::Ok(None)
            }

            _ => ExecResult::Error(format!("Causal opcode {:02X} not implemented", inst.opcode))
        }
    }

    // =========================================================================
    // QUANTUM: Superposition Operations
    // =========================================================================

    fn exec_quantum(&mut self, inst: &Instruction, _data: &[u64; 6]) -> ExecResult {
        self.stats.quantum_ops += 1;

        match inst.opcode {
            // SUPERPOSE: Create superposition of vectors
            0x00 => {
                let a = self.registers.read(inst.src1);
                let b = self.registers.read(inst.src2);
                match (a, b) {
                    (Some(a), Some(b)) => {
                        // Superposition via bundling (majority vote)
                        let mut result = [0u64; FINGERPRINT_WORDS];
                        for i in 0..FINGERPRINT_WORDS {
                            // Simple: OR for superposition
                            result[i] = a[i] | b[i];
                        }
                        self.registers.write(inst.dest, result);
                        ExecResult::Ok(Some(result))
                    }
                    _ => ExecResult::Error("SUPERPOSE: operand not found".into())
                }
            }

            // COLLAPSE: Measure/select from superposition
            0x01 => {
                if let Some(fp) = self.registers.read(inst.src1) {
                    // Collapse is identity for now (would use probability)
                    self.registers.write(inst.dest, fp);
                    ExecResult::Ok(Some(fp))
                } else {
                    ExecResult::Error("COLLAPSE: source not found".into())
                }
            }

            // INTERFERE: Wave interference (XOR)
            0x03 => {
                let a = self.registers.read(inst.src1);
                let b = self.registers.read(inst.src2);
                match (a, b) {
                    (Some(a), Some(b)) => {
                        let mut result = [0u64; FINGERPRINT_WORDS];
                        for i in 0..FINGERPRINT_WORDS {
                            result[i] = a[i] ^ b[i];
                        }
                        self.registers.write(inst.dest, result);
                        ExecResult::Ok(Some(result))
                    }
                    _ => ExecResult::Error("INTERFERE: operand not found".into())
                }
            }

            _ => ExecResult::Error(format!("Quantum opcode {:02X} not implemented", inst.opcode))
        }
    }

    // =========================================================================
    // MEMORY: Bind Space Operations
    // =========================================================================

    fn exec_memory(&mut self, inst: &Instruction) -> ExecResult {
        self.stats.memory_ops += 1;

        match inst.opcode {
            // BIND: XOR binding (A ⊗ B)
            0x00 => {
                let a = self.registers.read(inst.src1);
                let b = self.registers.read(inst.src2);
                match (a, b) {
                    (Some(a), Some(b)) => {
                        let mut result = [0u64; FINGERPRINT_WORDS];
                        for i in 0..FINGERPRINT_WORDS {
                            result[i] = a[i] ^ b[i];
                        }
                        self.registers.write(inst.dest, result);
                        ExecResult::Ok(Some(result))
                    }
                    _ => ExecResult::Error("BIND: operand not found".into())
                }
            }

            // UNBIND: Same as BIND (XOR is self-inverse)
            0x01 => {
                // Delegate to BIND
                let mut inst_bind = *inst;
                inst_bind.opcode = 0x00;
                self.exec_memory(&inst_bind)
            }

            // BUNDLE: Majority vote
            0x02 => {
                // Would need more than 2 operands
                ExecResult::Error("BUNDLE: requires array input".into())
            }

            // PERMUTE: Rotate bits
            0x03 => {
                if let Some(fp) = self.registers.read(inst.src1) {
                    let shift = inst.src2 as u32;
                    let mut result = [0u64; FINGERPRINT_WORDS];
                    for i in 0..FINGERPRINT_WORDS {
                        result[i] = fp[i].rotate_left(shift);
                    }
                    self.registers.write(inst.dest, result);
                    ExecResult::Ok(Some(result))
                } else {
                    ExecResult::Error("PERMUTE: source not found".into())
                }
            }

            // LOAD: Copy from address
            0x10 => {
                if let Some(fp) = self.registers.read(inst.src1) {
                    self.registers.write(inst.dest, fp);
                    ExecResult::Ok(Some(fp))
                } else {
                    ExecResult::Error("LOAD: source not found".into())
                }
            }

            // STORE: Copy to address
            0x11 => {
                if let Some(fp) = self.registers.read(inst.src1) {
                    self.registers.write(inst.dest, fp);
                    ExecResult::Ok(None)
                } else {
                    ExecResult::Error("STORE: source not found".into())
                }
            }

            _ => ExecResult::Error(format!("Memory opcode {:02X} not implemented", inst.opcode))
        }
    }

    // =========================================================================
    // CONTROL: Flow Control
    // =========================================================================

    fn exec_control(&mut self, inst: &Instruction) -> ExecResult {
        self.stats.control_ops += 1;

        match inst.opcode {
            // NOP
            0x00 => ExecResult::Ok(None),

            // JUMP: Unconditional branch
            0x01 => ExecResult::Branch(inst.src2),

            // BRANCH: Conditional branch
            0x02 => {
                let take_branch = match inst.flags.encode() & 0x0F {
                    0x00 => true,                    // Always
                    0x01 => self.flags.zero,         // BEQ (zero set)
                    0x02 => !self.flags.zero,        // BNE (zero clear)
                    0x03 => self.flags.negative,     // BMI (negative)
                    0x04 => !self.flags.negative,    // BPL (positive)
                    _ => false,
                };
                if take_branch {
                    ExecResult::Branch(inst.src2)
                } else {
                    ExecResult::Ok(None)
                }
            }

            // CALL: Push return, jump (simplified - no stack yet)
            0x03 => {
                // Would push PC to stack
                ExecResult::Branch(inst.src2)
            }

            // RET: Pop return, jump
            0x04 => {
                // Would pop from stack
                ExecResult::Halt // For now, RET halts
            }

            // CMP: Compare and set flags
            0x10 => {
                let a = self.registers.read(inst.src1);
                let b = self.registers.read(inst.src2);
                match (a, b) {
                    (Some(a), Some(b)) => {
                        // Compare by Hamming distance
                        let dist: u32 = a.iter()
                            .zip(b.iter())
                            .map(|(x, y)| (x ^ y).count_ones())
                            .sum();
                        self.flags.zero = dist == 0;
                        self.flags.negative = false; // Hamming is always positive
                        ExecResult::Ok(None)
                    }
                    _ => ExecResult::Error("CMP: operand not found".into())
                }
            }

            _ => ExecResult::Error(format!("Control opcode {:02X} not implemented", inst.opcode))
        }
    }

    // =========================================================================
    // TRAP: System Calls
    // =========================================================================

    fn exec_trap(&mut self, inst: &Instruction, _operand: &[u64; 2]) -> ExecResult {
        self.stats.trap_ops += 1;

        match inst.opcode {
            // HALT
            0x00 => ExecResult::Halt,

            // PANIC
            0x01 => ExecResult::Error("PANIC".into()),

            // DEBUG: Print state
            0xFF => {
                println!("DEBUG: PC={:04X} flags={:?}", self.pc, self.flags);
                println!("       stats={:?}", self.stats);
                ExecResult::Ok(None)
            }

            _ => ExecResult::Error(format!("Trap {:02X} not implemented", inst.opcode))
        }
    }

    /// Get execution statistics
    pub fn stats(&self) -> &ExecutorStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = ExecutorStats::default();
    }
}

impl Default for Executor {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fabric::firefly_frame::{FrameBuilder, FrameHeader};

    fn random_fp(seed: u64) -> [u64; FINGERPRINT_WORDS] {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut fp = [0u64; FINGERPRINT_WORDS];
        for i in 0..FINGERPRINT_WORDS {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            i.hash(&mut hasher);
            fp[i] = hasher.finish();
        }
        fp
    }

    #[test]
    fn test_executor_bind() {
        let mut exec = Executor::new();

        // Write two fingerprints
        let a = random_fp(1);
        let b = random_fp(2);
        exec.registers.write(0x8000, a);
        exec.registers.write(0x8001, b);

        // Create BIND instruction
        let builder = FrameBuilder::new(1);
        let frame = builder.bind(0x8002, 0x8000, 0x8001);

        // Execute
        let result = exec.execute(&frame);
        assert!(matches!(result, ExecResult::Ok(Some(_))));

        // Verify result is XOR of inputs
        let bound = exec.registers.read(0x8002).unwrap();
        for i in 0..FINGERPRINT_WORDS {
            assert_eq!(bound[i], a[i] ^ b[i]);
        }
    }

    #[test]
    fn test_executor_nars_deduce() {
        let mut exec = Executor::new();

        let builder = FrameBuilder::new(1);
        let mut frame = builder.nars_deduce(0x8000, 0x8001, 0x8002);

        // Set truth value in context
        frame.context.truth = (200, 180); // 0.78, 0.71

        let result = exec.execute(&frame);
        assert!(matches!(result, ExecResult::Truth(_)));

        if let ExecResult::Truth(t) = result {
            assert!(t.frequency > 0.0 && t.frequency < 1.0);
            assert!(t.confidence > 0.0 && t.confidence < 1.0);
        }
    }

    #[test]
    fn test_executor_control_flow() {
        let mut exec = Executor::new();

        // BRANCH to address 0x8010
        let builder = FrameBuilder::new(1);
        let frame = builder.branch(0, 0x8010); // Always branch

        let result = exec.execute(&frame);
        assert!(matches!(result, ExecResult::Branch(0x8010)));
    }

    #[test]
    fn test_executor_program() {
        let mut exec = Executor::new();

        // Write a test fingerprint
        let fp = random_fp(42);
        exec.registers.write(0x8000, fp);

        let builder = FrameBuilder::new(1);

        // Simple program: LOAD, PERMUTE, HALT
        let program = vec![
            // Load from 0x8000 to 0x8001
            {
                let header = FrameHeader::new(1, 0, 0, 1);
                let inst = Instruction::new(LanguagePrefix::Memory, 0x10, 0x8001, 0x8000, 0);
                FireflyFrame::new(header, inst)
            },
            // Permute by 7 bits
            {
                let header = FrameHeader::new(1, 0, 0, 2);
                let inst = Instruction::new(LanguagePrefix::Memory, 0x03, 0x8002, 0x8001, 7);
                FireflyFrame::new(header, inst)
            },
            // Halt
            {
                let header = FrameHeader::new(1, 0, 0, 3);
                let inst = Instruction::new(LanguagePrefix::Trap, 0x00, 0, 0, 0);
                FireflyFrame::new(header, inst)
            },
        ];

        let results = exec.run(&program, 100);

        assert_eq!(results.len(), 3);
        assert!(matches!(results[0], ExecResult::Ok(Some(_))));
        assert!(matches!(results[1], ExecResult::Ok(Some(_))));
        assert!(matches!(results[2], ExecResult::Halt));

        // Verify permutation worked
        let original = exec.registers.read(0x8001).unwrap();
        let permuted = exec.registers.read(0x8002).unwrap();
        for i in 0..FINGERPRINT_WORDS {
            assert_eq!(permuted[i], original[i].rotate_left(7));
        }
    }
}
