//! GEL: Graph Executable Language
//!
//! A human-readable assembly language for the cognitive CPU.
//! Compiles to FireflyFrames for execution.
//!
//! ## Syntax
//!
//! ```gel
//! ; Comments start with ; or #
//! .session 100         ; Set session ID
//! .lane 0              ; Execution lane
//! .hive 0              ; Hive ID
//!
//! ; Labels end with :
//! main:
//!     load r0, [0x8000]        ; Load fingerprint
//!     resonate r1, r0, 10      ; Find 10 nearest
//!     bind r2, r0, r1          ; XOR binding
//!     halt                     ; Stop execution
//!
//! ; Registers: r0-r255 (fluid zone)
//! ; Addresses: [0x8000] (node zone)
//! ; Immediates: #42, #0xFF
//! ```
//!
//! ## Instruction Set
//!
//! | Prefix  | Instructions                                    |
//! |---------|------------------------------------------------|
//! | Lance   | resonate, insert, hamming                       |
//! | SQL     | select, join, filter                            |
//! | Cypher  | match, traverse, path                           |
//! | NARS    | deduce, induce, abduce, revise, negate         |
//! | Causal  | see, do, imagine                                |
//! | Quantum | superpose, collapse, interfere                  |
//! | Memory  | bind, unbind, bundle, permute, load, store     |
//! | Control | nop, jump, branch, call, ret, cmp, halt        |
//! | Trap    | syscall, debug                                  |

use super::firefly_frame::{FireflyFrame, FrameHeader, Instruction, LanguagePrefix};
use std::collections::HashMap;

// =============================================================================
// AST
// =============================================================================

/// GEL source location for error messages
#[derive(Debug, Clone, Copy)]
pub struct Location {
    pub line: usize,
    pub column: usize,
}

/// GEL operand
#[derive(Debug, Clone)]
pub enum Operand {
    /// Register: r0-r255 (maps to fluid zone 0x10XX)
    Register(u8),
    /// Node address: [0x8000] (maps to node zone)
    Address(u16),
    /// Immediate value: #42
    Immediate(u16),
    /// Label reference
    Label(String),
}

/// GEL instruction
#[derive(Debug, Clone)]
pub struct GelInstruction {
    pub mnemonic: String,
    pub operands: Vec<Operand>,
    pub location: Location,
}

/// GEL directive
#[derive(Debug, Clone)]
pub enum Directive {
    Session(u16),
    Lane(u8),
    Hive(u8),
    Origin(u16),
    Data(Vec<u8>),
}

/// GEL statement
#[derive(Debug, Clone)]
pub enum Statement {
    Label(String),
    Directive(Directive),
    Instruction(GelInstruction),
}

/// GEL program
#[derive(Debug, Clone)]
pub struct GelProgram {
    pub statements: Vec<Statement>,
    pub labels: HashMap<String, u16>,
}

// =============================================================================
// PARSER
// =============================================================================

/// GEL parser
pub struct GelParser {
    input: String,
    pos: usize,
    line: usize,
    column: usize,
}

#[derive(Debug)]
pub struct ParseError {
    pub message: String,
    pub location: Location,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{}: {}",
            self.location.line, self.location.column, self.message
        )
    }
}

impl std::error::Error for ParseError {}

impl GelParser {
    pub fn new(input: &str) -> Self {
        Self {
            input: input.to_string(),
            pos: 0,
            line: 1,
            column: 1,
        }
    }

    /// Parse the entire program
    pub fn parse(mut self) -> Result<GelProgram, ParseError> {
        let mut statements = Vec::new();
        let mut labels = HashMap::new();
        let mut addr = 0x8000u16; // Start in node zone

        while !self.at_end() {
            self.skip_whitespace_and_comments();
            if self.at_end() {
                break;
            }

            // Parse statement
            if self.peek() == '.' {
                // Directive
                statements.push(self.parse_directive()?);
            } else if self.peek_is_label() {
                // Label
                let name = self.parse_identifier()?;
                self.expect(':')?;
                labels.insert(name.clone(), addr);
                statements.push(Statement::Label(name));
            } else if self.peek().is_alphabetic() {
                // Instruction
                let inst = self.parse_instruction()?;
                statements.push(Statement::Instruction(inst));
                addr = addr.wrapping_add(1);
            } else {
                return Err(self.error("unexpected character"));
            }
        }

        Ok(GelProgram { statements, labels })
    }

    fn parse_directive(&mut self) -> Result<Statement, ParseError> {
        self.expect('.')?;
        let name = self.parse_identifier()?;
        self.skip_horizontal_ws();

        let directive = match name.as_str() {
            "session" => Directive::Session(self.parse_number()? as u16),
            "lane" => Directive::Lane(self.parse_number()? as u8),
            "hive" => Directive::Hive(self.parse_number()? as u8),
            "origin" | "org" => Directive::Origin(self.parse_number()? as u16),
            _ => return Err(self.error(&format!("unknown directive: .{}", name))),
        };

        Ok(Statement::Directive(directive))
    }

    fn parse_instruction(&mut self) -> Result<GelInstruction, ParseError> {
        let location = self.location();
        let mnemonic = self.parse_identifier()?.to_lowercase();
        self.skip_horizontal_ws();

        let mut operands = Vec::new();

        // Parse operands separated by commas
        if !self.at_eol() && self.peek() != ';' && self.peek() != '#' {
            operands.push(self.parse_operand()?);

            while self.try_consume(',') {
                self.skip_horizontal_ws();
                operands.push(self.parse_operand()?);
            }
        }

        Ok(GelInstruction {
            mnemonic,
            operands,
            location,
        })
    }

    fn parse_operand(&mut self) -> Result<Operand, ParseError> {
        self.skip_horizontal_ws();

        if self.peek() == 'r' || self.peek() == 'R' {
            // Register: r0-r255
            self.advance();
            let num = self.parse_number()?;
            Ok(Operand::Register(num as u8))
        } else if self.peek() == '[' {
            // Address: [0x8000]
            self.advance();
            self.skip_horizontal_ws();
            let addr = self.parse_number()?;
            self.skip_horizontal_ws();
            self.expect(']')?;
            Ok(Operand::Address(addr as u16))
        } else if self.peek() == '#' || self.peek().is_ascii_digit() {
            // Immediate: #42 or just 42
            if self.peek() == '#' {
                self.advance();
            }
            let num = self.parse_number()?;
            Ok(Operand::Immediate(num as u16))
        } else if self.peek().is_alphabetic() || self.peek() == '_' {
            // Label reference
            let name = self.parse_identifier()?;
            Ok(Operand::Label(name))
        } else {
            Err(self.error("expected operand"))
        }
    }

    fn parse_identifier(&mut self) -> Result<String, ParseError> {
        let mut id = String::new();
        while !self.at_end() && (self.peek().is_alphanumeric() || self.peek() == '_') {
            id.push(self.advance());
        }
        if id.is_empty() {
            return Err(self.error("expected identifier"));
        }
        Ok(id)
    }

    fn parse_number(&mut self) -> Result<i64, ParseError> {
        let start = self.pos;

        // Handle hex: 0x...
        if self.peek() == '0' && self.pos + 1 < self.input.len() {
            let next = self.input.chars().nth(self.pos + 1);
            if next == Some('x') || next == Some('X') {
                self.advance(); // 0
                self.advance(); // x
                let hex_start = self.pos;
                while !self.at_end() && self.peek().is_ascii_hexdigit() {
                    self.advance();
                }
                let hex = &self.input[hex_start..self.pos];
                return i64::from_str_radix(hex, 16).map_err(|_| self.error("invalid hex number"));
            }
        }

        // Decimal
        if self.peek() == '-' {
            self.advance();
        }
        while !self.at_end() && self.peek().is_ascii_digit() {
            self.advance();
        }

        let num_str = &self.input[start..self.pos];
        num_str.parse().map_err(|_| self.error("invalid number"))
    }

    fn peek_is_label(&self) -> bool {
        // Look ahead for ':'
        let mut pos = self.pos;
        while pos < self.input.len() {
            let c = self.input.chars().nth(pos).unwrap_or('\0');
            if c == ':' {
                return true;
            }
            if c.is_whitespace() || c == ';' || c == '#' || (!c.is_alphanumeric() && c != '_') {
                return false;
            }
            pos += 1;
        }
        false
    }

    fn at_end(&self) -> bool {
        self.pos >= self.input.len()
    }

    fn at_eol(&self) -> bool {
        self.at_end() || self.peek() == '\n'
    }

    fn peek(&self) -> char {
        self.input.chars().nth(self.pos).unwrap_or('\0')
    }

    fn advance(&mut self) -> char {
        let c = self.peek();
        self.pos += 1;
        if c == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        c
    }

    fn try_consume(&mut self, c: char) -> bool {
        if self.peek() == c {
            self.advance();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, c: char) -> Result<(), ParseError> {
        if self.try_consume(c) {
            Ok(())
        } else {
            Err(self.error(&format!("expected '{}'", c)))
        }
    }

    fn skip_horizontal_ws(&mut self) {
        while !self.at_end() && (self.peek() == ' ' || self.peek() == '\t') {
            self.advance();
        }
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            // Skip whitespace
            while !self.at_end() && self.peek().is_whitespace() {
                self.advance();
            }
            // Skip comments
            if self.peek() == ';' || self.peek() == '#' {
                while !self.at_end() && self.peek() != '\n' {
                    self.advance();
                }
            } else {
                break;
            }
        }
    }

    fn location(&self) -> Location {
        Location {
            line: self.line,
            column: self.column,
        }
    }

    fn error(&self, message: &str) -> ParseError {
        ParseError {
            message: message.to_string(),
            location: self.location(),
        }
    }
}

// =============================================================================
// COMPILER
// =============================================================================

/// Compile GEL program to FireflyFrames
pub struct GelCompiler {
    session_id: u16,
    lane_id: u8,
    hive_id: u8,
    origin: u16,
    sequence: u16,
}

impl GelCompiler {
    pub fn new() -> Self {
        Self {
            session_id: 1,
            lane_id: 0,
            hive_id: 0,
            origin: 0x8000,
            sequence: 0,
        }
    }

    /// Compile GEL program to frames
    pub fn compile(&mut self, program: &GelProgram) -> Result<Vec<FireflyFrame>, String> {
        let mut frames = Vec::new();

        for stmt in &program.statements {
            match stmt {
                Statement::Directive(d) => self.apply_directive(d),
                Statement::Label(_) => {} // Labels resolved in first pass
                Statement::Instruction(inst) => {
                    let frame = self.compile_instruction(inst, &program.labels)?;
                    frames.push(frame);
                }
            }
        }

        Ok(frames)
    }

    fn apply_directive(&mut self, directive: &Directive) {
        match directive {
            Directive::Session(id) => self.session_id = *id,
            Directive::Lane(id) => self.lane_id = *id,
            Directive::Hive(id) => self.hive_id = *id,
            Directive::Origin(addr) => self.origin = *addr,
            Directive::Data(_) => {} // TODO: data embedding
        }
    }

    fn compile_instruction(
        &mut self,
        inst: &GelInstruction,
        labels: &HashMap<String, u16>,
    ) -> Result<FireflyFrame, String> {
        let header = FrameHeader::new(self.session_id, self.hive_id, self.lane_id, self.sequence);
        self.sequence = self.sequence.wrapping_add(1);

        // Resolve operands
        let ops: Vec<u16> = inst
            .operands
            .iter()
            .map(|op| self.resolve_operand(op, labels))
            .collect::<Result<Vec<_>, _>>()?;

        // Compile based on mnemonic
        let instruction = match inst.mnemonic.as_str() {
            // === LANCE: Vector Operations ===
            "resonate" => {
                let (dest, query, k) = self.get_3_ops(&ops, &inst.mnemonic)?;
                Instruction::new(LanguagePrefix::Lance, 0x00, dest, query, k)
            }
            "insert" => {
                let src = self.get_1_op(&ops, &inst.mnemonic)?;
                Instruction::new(LanguagePrefix::Lance, 0x01, 0, src, 0)
            }
            "hamming" => {
                let (dest, a, b) = self.get_3_ops(&ops, &inst.mnemonic)?;
                Instruction::new(LanguagePrefix::Lance, 0x10, dest, a, b)
            }

            // === NARS: Inference ===
            "deduce" => {
                let (dest, a, b) = self.get_3_ops(&ops, &inst.mnemonic)?;
                Instruction::new(LanguagePrefix::Nars, 0x00, dest, a, b)
            }
            "induce" => {
                let (dest, a, b) = self.get_3_ops(&ops, &inst.mnemonic)?;
                Instruction::new(LanguagePrefix::Nars, 0x01, dest, a, b)
            }
            "abduce" => {
                let (dest, a, b) = self.get_3_ops(&ops, &inst.mnemonic)?;
                Instruction::new(LanguagePrefix::Nars, 0x02, dest, a, b)
            }
            "revise" => {
                let (dest, a, b) = self.get_3_ops(&ops, &inst.mnemonic)?;
                Instruction::new(LanguagePrefix::Nars, 0x03, dest, a, b)
            }
            "negate" => {
                let (dest, src) = self.get_2_ops(&ops, &inst.mnemonic)?;
                Instruction::new(LanguagePrefix::Nars, 0x04, dest, src, 0)
            }

            // === CAUSAL: Pearl's Three Rungs ===
            "see" => {
                let (dest, query) = self.get_2_ops(&ops, &inst.mnemonic)?;
                Instruction::new(LanguagePrefix::Causal, 0x00, dest, query, 0)
            }
            "do" | "intervene" => {
                let (dest, target, value) = self.get_3_ops(&ops, &inst.mnemonic)?;
                Instruction::new(LanguagePrefix::Causal, 0x01, dest, target, value)
            }
            "imagine" => {
                let (dest, query) = self.get_2_ops(&ops, &inst.mnemonic)?;
                Instruction::new(LanguagePrefix::Causal, 0x02, dest, query, 0)
            }

            // === QUANTUM: Superposition ===
            "superpose" => {
                let (dest, a, b) = self.get_3_ops(&ops, &inst.mnemonic)?;
                Instruction::new(LanguagePrefix::Quantum, 0x00, dest, a, b)
            }
            "collapse" => {
                let (dest, src) = self.get_2_ops(&ops, &inst.mnemonic)?;
                Instruction::new(LanguagePrefix::Quantum, 0x01, dest, src, 0)
            }
            "interfere" => {
                let (dest, a, b) = self.get_3_ops(&ops, &inst.mnemonic)?;
                Instruction::new(LanguagePrefix::Quantum, 0x03, dest, a, b)
            }

            // === MEMORY: Bind Space Operations ===
            "bind" => {
                let (dest, a, b) = self.get_3_ops(&ops, &inst.mnemonic)?;
                Instruction::new(LanguagePrefix::Memory, 0x00, dest, a, b)
            }
            "unbind" => {
                let (dest, a, b) = self.get_3_ops(&ops, &inst.mnemonic)?;
                Instruction::new(LanguagePrefix::Memory, 0x01, dest, a, b)
            }
            "bundle" => {
                // Bundle needs special handling for multiple inputs
                let dest = self.get_1_op(&ops, &inst.mnemonic)?;
                Instruction::new(LanguagePrefix::Memory, 0x02, dest, 0, 0)
            }
            "permute" => {
                let (dest, src, shift) = self.get_3_ops(&ops, &inst.mnemonic)?;
                Instruction::new(LanguagePrefix::Memory, 0x03, dest, src, shift)
            }
            "load" => {
                let (dest, src) = self.get_2_ops(&ops, &inst.mnemonic)?;
                Instruction::new(LanguagePrefix::Memory, 0x10, dest, src, 0)
            }
            "store" => {
                let (dest, src) = self.get_2_ops(&ops, &inst.mnemonic)?;
                Instruction::new(LanguagePrefix::Memory, 0x11, dest, src, 0)
            }

            // === CONTROL: Flow Control ===
            "nop" => Instruction::new(LanguagePrefix::Control, 0x00, 0, 0, 0),
            "jump" | "jmp" => {
                let target = self.get_1_op(&ops, &inst.mnemonic)?;
                Instruction::new(LanguagePrefix::Control, 0x01, 0, 0, target)
            }
            "branch" | "br" => {
                let target = self.get_1_op(&ops, &inst.mnemonic)?;
                Instruction::new(LanguagePrefix::Control, 0x02, 0, 0, target)
            }
            "beq" => {
                let target = self.get_1_op(&ops, &inst.mnemonic)?;
                let mut inst = Instruction::new(LanguagePrefix::Control, 0x02, 0, 0, target);
                inst.flags.zero = true; // Branch if zero
                inst
            }
            "bne" => {
                let target = self.get_1_op(&ops, &inst.mnemonic)?;
                let mut inst = Instruction::new(LanguagePrefix::Control, 0x02, 0, 0, target);
                inst.flags.negative = true; // Use negative as "not equal" marker
                inst
            }
            "call" => {
                let target = self.get_1_op(&ops, &inst.mnemonic)?;
                Instruction::new(LanguagePrefix::Control, 0x03, 0, 0, target)
            }
            "ret" | "return" => Instruction::new(LanguagePrefix::Control, 0x04, 0, 0, 0),
            "cmp" => {
                let (a, b) = self.get_2_ops(&ops, &inst.mnemonic)?;
                Instruction::new(LanguagePrefix::Control, 0x10, 0, a, b)
            }
            "halt" => Instruction::new(LanguagePrefix::Trap, 0x00, 0, 0, 0),

            // === TRAP: System Calls ===
            "syscall" => {
                let num = ops.first().copied().unwrap_or(0);
                Instruction::new(LanguagePrefix::Trap, num as u8, 0, 0, 0)
            }
            "debug" => Instruction::new(LanguagePrefix::Trap, 0xFF, 0, 0, 0),
            "panic" => Instruction::new(LanguagePrefix::Trap, 0x01, 0, 0, 0),

            // === CYPHER: Graph (stubs) ===
            "match" => {
                let (dest, pattern) = self.get_2_ops(&ops, &inst.mnemonic)?;
                Instruction::new(LanguagePrefix::Cypher, 0x00, dest, pattern, 0)
            }
            "traverse" => {
                let (dest, start, hops) = self.get_3_ops(&ops, &inst.mnemonic)?;
                Instruction::new(LanguagePrefix::Cypher, 0x01, dest, start, hops)
            }
            "path" => {
                let (dest, src, target) = self.get_3_ops(&ops, &inst.mnemonic)?;
                Instruction::new(LanguagePrefix::Cypher, 0x02, dest, src, target)
            }

            // === SQL: Relational (stubs) ===
            "select" => {
                let (dest, table) = self.get_2_ops(&ops, &inst.mnemonic)?;
                Instruction::new(LanguagePrefix::Sql, 0x00, dest, table, 0)
            }
            "filter" => {
                let (dest, src, pred) = self.get_3_ops(&ops, &inst.mnemonic)?;
                Instruction::new(LanguagePrefix::Sql, 0x01, dest, src, pred)
            }
            "join" => {
                let (dest, left, right) = self.get_3_ops(&ops, &inst.mnemonic)?;
                Instruction::new(LanguagePrefix::Sql, 0x02, dest, left, right)
            }

            _ => return Err(format!("unknown instruction: {}", inst.mnemonic)),
        };

        Ok(FireflyFrame::new(header, instruction))
    }

    fn resolve_operand(&self, op: &Operand, labels: &HashMap<String, u16>) -> Result<u16, String> {
        match op {
            Operand::Register(r) => {
                // Registers map to fluid zone: 0x10XX
                Ok(0x1000 | (*r as u16))
            }
            Operand::Address(addr) => Ok(*addr),
            Operand::Immediate(val) => Ok(*val),
            Operand::Label(name) => labels
                .get(name)
                .copied()
                .ok_or_else(|| format!("undefined label: {}", name)),
        }
    }

    fn get_1_op(&self, ops: &[u16], mnemonic: &str) -> Result<u16, String> {
        ops.first()
            .copied()
            .ok_or_else(|| format!("{} requires 1 operand", mnemonic))
    }

    fn get_2_ops(&self, ops: &[u16], mnemonic: &str) -> Result<(u16, u16), String> {
        if ops.len() < 2 {
            return Err(format!("{} requires 2 operands", mnemonic));
        }
        Ok((ops[0], ops[1]))
    }

    fn get_3_ops(&self, ops: &[u16], mnemonic: &str) -> Result<(u16, u16, u16), String> {
        if ops.len() < 3 {
            return Err(format!("{} requires 3 operands", mnemonic));
        }
        Ok((ops[0], ops[1], ops[2]))
    }
}

impl Default for GelCompiler {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// CONVENIENCE FUNCTIONS
// =============================================================================

/// Parse and compile GEL source code to frames
pub fn compile(source: &str) -> Result<Vec<FireflyFrame>, String> {
    let parser = GelParser::new(source);
    let program = parser.parse().map_err(|e| e.to_string())?;
    let mut compiler = GelCompiler::new();
    compiler.compile(&program)
}

/// Disassemble frames back to GEL source
pub fn disassemble(frames: &[FireflyFrame]) -> String {
    let mut output = String::new();

    for (i, frame) in frames.iter().enumerate() {
        let inst = &frame.instruction;
        let prefix = match inst.prefix {
            LanguagePrefix::Lance => "lance",
            LanguagePrefix::Sql => "sql",
            LanguagePrefix::Cypher => "cypher",
            LanguagePrefix::Nars => "nars",
            LanguagePrefix::Causal => "causal",
            LanguagePrefix::Quantum => "quantum",
            LanguagePrefix::Memory => "mem",
            LanguagePrefix::Control => "ctrl",
            LanguagePrefix::Trap => "trap",
        };

        let mnemonic = get_mnemonic(inst.prefix, inst.opcode);

        output.push_str(&format!(
            "{:04X}: {} {} r{}, r{}, #{}\n",
            0x8000 + i,
            mnemonic,
            prefix,
            inst.dest,
            inst.src1,
            inst.src2
        ));
    }

    output
}

fn get_mnemonic(prefix: LanguagePrefix, opcode: u8) -> &'static str {
    match prefix {
        LanguagePrefix::Lance => match opcode {
            0x00 => "resonate",
            0x01 => "insert",
            0x10 => "hamming",
            _ => "?lance",
        },
        LanguagePrefix::Nars => match opcode {
            0x00 => "deduce",
            0x01 => "induce",
            0x02 => "abduce",
            0x03 => "revise",
            0x04 => "negate",
            _ => "?nars",
        },
        LanguagePrefix::Memory => match opcode {
            0x00 => "bind",
            0x01 => "unbind",
            0x02 => "bundle",
            0x03 => "permute",
            0x10 => "load",
            0x11 => "store",
            _ => "?mem",
        },
        LanguagePrefix::Control => match opcode {
            0x00 => "nop",
            0x01 => "jump",
            0x02 => "branch",
            0x03 => "call",
            0x04 => "ret",
            0x10 => "cmp",
            _ => "?ctrl",
        },
        LanguagePrefix::Trap => match opcode {
            0x00 => "halt",
            0x01 => "panic",
            0xFF => "debug",
            _ => "?trap",
        },
        _ => "?",
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fabric::executor::Executor;
    use crate::storage::FINGERPRINT_WORDS;

    #[test]
    fn test_parse_simple() {
        let source = r#"
            ; Simple test program
            .session 100

            main:
                load r0, [0x8000]
                bind r1, r0, r0
                halt
        "#;

        let parser = GelParser::new(source);
        let program = parser.parse().unwrap();

        assert!(program.labels.contains_key("main"));
        assert_eq!(program.statements.len(), 5); // directive, label, 3 instructions
    }

    #[test]
    fn test_compile_and_run() {
        let source = r#"
            .session 42

            ; Bind two fingerprints
            load r0, [0x8000]   ; Load from node zone
            load r1, [0x8001]
            bind r2, r0, r1     ; XOR bind
            halt
        "#;

        // Compile
        let frames = compile(source).unwrap();
        assert_eq!(frames.len(), 4); // load, load, bind, halt

        // Verify first frame
        assert_eq!(frames[0].instruction.prefix, LanguagePrefix::Memory);
        assert_eq!(frames[0].instruction.opcode, 0x10); // load

        // Verify session ID propagated
        assert_eq!(frames[0].header.session_id, 42);
    }

    #[test]
    fn test_control_flow() {
        let source = r#"
            start:
                cmp r0, r1
                beq done
                nop
            done:
                halt
        "#;

        let frames = compile(source).unwrap();
        assert_eq!(frames.len(), 4);

        // Branch should target 'done' label
        assert_eq!(frames[1].instruction.prefix, LanguagePrefix::Control);
        assert_eq!(frames[1].instruction.opcode, 0x02); // branch
    }

    #[test]
    fn test_nars_inference() {
        let source = r#"
            ; NARS inference chain
            deduce r2, r0, r1
            revise r3, r2, r2
            halt
        "#;

        let frames = compile(source).unwrap();
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0].instruction.prefix, LanguagePrefix::Nars);
        assert_eq!(frames[0].instruction.opcode, 0x00); // deduce
    }

    #[test]
    fn test_disassemble() {
        let source = "bind r0, r1, r2\nhalt";
        let frames = compile(source).unwrap();
        let disasm = disassemble(&frames);
        assert!(disasm.contains("bind"));
        assert!(disasm.contains("halt"));
    }

    #[test]
    fn test_full_execution() {
        // Write fingerprints, bind them, verify result
        let source = r#"
            .session 1

            ; Program: Bind and permute
            load r0, [0x8000]
            load r1, [0x8001]
            bind r2, r0, r1
            permute r3, r2, 7
            halt
        "#;

        let frames = compile(source).unwrap();

        // Create executor and preload data
        let mut exec = Executor::new();

        // Write test fingerprints
        let fp1 = [0xAAAAAAAAAAAAAAAAu64; FINGERPRINT_WORDS];
        let fp2 = [0x5555555555555555u64; FINGERPRINT_WORDS];
        exec.registers.write(0x8000, fp1);
        exec.registers.write(0x8001, fp2);

        // Run program
        let results = exec.run(&frames, 100);

        assert_eq!(results.len(), 5);
        assert!(matches!(
            results.last(),
            Some(crate::fabric::executor::ExecResult::Halt)
        ));
    }
}
