//! Cognitive Stack - The Unified Integration Layer
//!
//! Wires together all components into a coherent cognitive substrate.

use std::collections::HashMap;
use std::time::{Duration, Instant};

// Re-export for external use
pub use crate::storage::bind_space::{Addr, nsm_slots, template_slots, speech_act_slots};

// =============================================================================
// THINKING STYLE (12 styles)
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ThinkingStyle {
    #[default]
    Analytical,
    Convergent,
    Systematic,
    Creative,
    Divergent,
    Exploratory,
    Focused,
    Diffuse,
    Peripheral,
    Intuitive,
    Deliberate,
    Metacognitive,
}

#[derive(Clone, Copy, Debug)]
pub struct StyleParams {
    pub resonance_threshold: f32,
    pub fan_out: usize,
    pub exploration: f32,
    pub speed: f32,
    pub collapse_bias: f32,
}

impl ThinkingStyle {
    pub fn params(&self) -> StyleParams {
        match self {
            Self::Analytical => StyleParams {
                resonance_threshold: 0.85, fan_out: 3, exploration: 0.05, speed: 0.1, collapse_bias: -0.1,
            },
            Self::Convergent => StyleParams {
                resonance_threshold: 0.75, fan_out: 4, exploration: 0.1, speed: 0.3, collapse_bias: -0.05,
            },
            Self::Systematic => StyleParams {
                resonance_threshold: 0.70, fan_out: 5, exploration: 0.1, speed: 0.2, collapse_bias: 0.0,
            },
            Self::Creative => StyleParams {
                resonance_threshold: 0.35, fan_out: 12, exploration: 0.8, speed: 0.5, collapse_bias: 0.15,
            },
            Self::Divergent => StyleParams {
                resonance_threshold: 0.40, fan_out: 10, exploration: 0.7, speed: 0.4, collapse_bias: 0.1,
            },
            Self::Exploratory => StyleParams {
                resonance_threshold: 0.30, fan_out: 15, exploration: 0.9, speed: 0.6, collapse_bias: 0.2,
            },
            Self::Focused => StyleParams {
                resonance_threshold: 0.90, fan_out: 1, exploration: 0.0, speed: 0.2, collapse_bias: -0.15,
            },
            Self::Diffuse => StyleParams {
                resonance_threshold: 0.45, fan_out: 8, exploration: 0.4, speed: 0.5, collapse_bias: 0.05,
            },
            Self::Peripheral => StyleParams {
                resonance_threshold: 0.20, fan_out: 20, exploration: 0.6, speed: 0.7, collapse_bias: 0.25,
            },
            Self::Intuitive => StyleParams {
                resonance_threshold: 0.50, fan_out: 3, exploration: 0.3, speed: 0.9, collapse_bias: 0.0,
            },
            Self::Deliberate => StyleParams {
                resonance_threshold: 0.70, fan_out: 7, exploration: 0.2, speed: 0.1, collapse_bias: -0.05,
            },
            Self::Metacognitive => StyleParams {
                resonance_threshold: 0.50, fan_out: 5, exploration: 0.3, speed: 0.3, collapse_bias: 0.0,
            },
        }
    }
}

// =============================================================================
// CONTEXT CRYSTAL (5×5×5 in Fluid Zone)
// =============================================================================

pub const CRYSTAL_GRID: usize = 5;
pub const CRYSTAL_CELLS: usize = 125; // 5×5×5

/// Mexican hat weights for temporal axis
pub const MEXICAN_HAT: [f32; 5] = [0.3, 0.7, 1.0, 0.7, 0.3];

#[derive(Clone)]
pub struct ContextCrystal {
    /// 5×5×5 grid: cells[temporal][subject][object]
    cells: Box<[[[CrystalCell; CRYSTAL_GRID]; CRYSTAL_GRID]; CRYSTAL_GRID]>,
    
    /// Current sentence position in temporal axis
    current_position: usize,
    
    /// Total sentences processed
    total_processed: usize,
}

#[derive(Clone, Default)]
pub struct CrystalCell {
    /// Superposed fingerprint
    pub fingerprint: Option<Vec<u64>>,
    
    /// Contribution count
    pub count: u32,
    
    /// TTL expiration
    pub expires_at: Option<Instant>,
    
    /// Qualia summary
    pub qualia_sum: [f32; 8],
}

impl Default for ContextCrystal {
    fn default() -> Self {
        Self::new()
    }
}

impl ContextCrystal {
    pub fn new() -> Self {
        Self {
            cells: Box::new([[[CrystalCell::default(); CRYSTAL_GRID]; CRYSTAL_GRID]; CRYSTAL_GRID]),
            current_position: 2, // Start at center (S0)
            total_processed: 0,
        }
    }
    
    /// Insert parsed triangle into crystal
    pub fn insert(&mut self, triangle: &ParsedTriangle, ttl: Duration) {
        // Compute cell indices
        let t_idx = self.current_position;
        let s_idx = hash_to_grid(&triangle.spo.subject);
        let o_idx = hash_to_grid(&triangle.spo.object);
        
        // Get cell
        let cell = &mut self.cells[t_idx][s_idx][o_idx];
        
        // Bundle fingerprint
        if let Some(ref mut existing) = cell.fingerprint {
            // Majority vote bundle
            for (i, word) in triangle.fingerprint.iter().enumerate() {
                if i < existing.len() {
                    existing[i] |= word; // Simple OR for now
                }
            }
        } else {
            cell.fingerprint = Some(triangle.fingerprint.clone());
        }
        
        // Update qualia
        for (i, &q) in triangle.qualia.iter().enumerate() {
            if i < 8 {
                cell.qualia_sum[i] += q;
            }
        }
        
        cell.count += 1;
        cell.expires_at = Some(Instant::now() + ttl);
        
        self.total_processed += 1;
    }
    
    /// Advance temporal window
    pub fn advance(&mut self) {
        // Shift temporal axis
        // This is like a sliding window
        self.current_position = (self.current_position + 1) % CRYSTAL_GRID;
    }
    
    /// Query crystal with weighted resonance
    pub fn query(&self, query_fp: &[u64], threshold: f32) -> Vec<CrystalMatch> {
        let mut matches = Vec::new();
        
        for t in 0..CRYSTAL_GRID {
            let temporal_weight = MEXICAN_HAT[t];
            
            for s in 0..CRYSTAL_GRID {
                for o in 0..CRYSTAL_GRID {
                    let cell = &self.cells[t][s][o];
                    
                    if let Some(ref cell_fp) = cell.fingerprint {
                        let similarity = hamming_similarity(query_fp, cell_fp);
                        let weighted_sim = similarity * temporal_weight;
                        
                        if weighted_sim >= threshold {
                            matches.push(CrystalMatch {
                                position: (t, s, o),
                                similarity: weighted_sim,
                                raw_similarity: similarity,
                                temporal_weight,
                            });
                        }
                    }
                }
            }
        }
        
        matches.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        matches
    }
    
    /// Evict expired cells
    pub fn tick(&mut self) {
        let now = Instant::now();
        
        for t in 0..CRYSTAL_GRID {
            for s in 0..CRYSTAL_GRID {
                for o in 0..CRYSTAL_GRID {
                    let cell = &mut self.cells[t][s][o];
                    
                    if let Some(expires) = cell.expires_at {
                        if now > expires {
                            // Evaporate
                            *cell = CrystalCell::default();
                        }
                    }
                }
            }
        }
    }
    
    /// Crystallize high-confidence cell to Node zone
    pub fn crystallize(&mut self, position: (usize, usize, usize)) -> Option<CrystalCell> {
        let (t, s, o) = position;
        
        if t < CRYSTAL_GRID && s < CRYSTAL_GRID && o < CRYSTAL_GRID {
            let cell = std::mem::take(&mut self.cells[t][s][o]);
            if cell.count > 0 {
                return Some(cell);
            }
        }
        None
    }
}

fn hash_to_grid(value: &Option<String>) -> usize {
    match value {
        Some(s) => {
            let hash: u64 = s.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
            (hash % CRYSTAL_GRID as u64) as usize
        }
        None => 2, // Center
    }
}

#[derive(Clone, Debug)]
pub struct CrystalMatch {
    pub position: (usize, usize, usize),
    pub similarity: f32,
    pub raw_similarity: f32,
    pub temporal_weight: f32,
}

// =============================================================================
// PARSED TRIANGLE (Unified output from Grammar Parser)
// =============================================================================

#[derive(Clone, Debug)]
pub struct ParsedTriangle {
    /// Source text
    pub source: String,
    
    /// 10K-bit fingerprint as u64 array
    pub fingerprint: Vec<u64>,
    
    /// Activated addresses
    pub addresses: Vec<Addr>,
    
    /// SPO extraction
    pub spo: SpoExtraction,
    
    /// Qualia vector (8D)
    pub qualia: [f32; 8],
    
    /// Collapse state
    pub collapse: CollapseResult,
    
    /// Top NSM activations
    pub nsm_top: Vec<(u8, f32)>,
    
    /// Best template match
    pub template: Option<TemplateInfo>,
    
    /// Speech act
    pub speech_act: SpeechActInfo,
}

#[derive(Clone, Debug, Default)]
pub struct SpoExtraction {
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub object: Option<String>,
}

#[derive(Clone, Debug)]
pub struct CollapseResult {
    pub gate: GateState,
    pub sd: f32,
    pub action: CollapseAction,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateState {
    Flow,
    Hold,
    Block,
}

#[derive(Clone, Debug)]
pub enum CollapseAction {
    CommitToNode { addr: Addr },
    HoldInCrystal { ttl_seconds: u64 },
    CheckCalibration { slot: u8 },
    AskClarification { reason: String },
}

#[derive(Clone, Debug)]
pub struct TemplateInfo {
    pub slot: u8,
    pub name: String,
    pub confidence: f32,
}

#[derive(Clone, Debug)]
pub struct SpeechActInfo {
    pub slot: u8,
    pub name: String,
    pub commitment_type: String,
}

// =============================================================================
// USER CALIBRATION (Prefix 0x0F)
// =============================================================================

#[derive(Clone, Debug)]
pub struct UserCalibration {
    /// Calibration entries by slot
    entries: HashMap<u8, CalibrationEntry>,
    
    /// Next available slot
    next_slot: u8,
}

#[derive(Clone, Debug)]
pub struct CalibrationEntry {
    /// Trigger pattern (simplified fingerprint)
    pub trigger: Vec<u64>,
    
    /// Correction type
    pub correction: CorrectionType,
    
    /// Confidence from usage
    pub confidence: f32,
    
    /// Usage count
    pub usage_count: u32,
}

#[derive(Clone, Debug)]
pub enum CorrectionType {
    /// Override template selection
    TemplateOverride { from: u8, to: u8 },
    
    /// NSM weight adjustment
    NsmWeightAdjust { primitive: u8, weight: f32 },
    
    /// Speech act bias
    SpeechActBias { formal: u8, informal: u8 },
    
    /// Language variant
    LanguageVariant { standard: u8, variant: u8 },
}

impl Default for UserCalibration {
    fn default() -> Self {
        Self::new()
    }
}

impl UserCalibration {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            next_slot: 0,
        }
    }
    
    /// Check for matching calibration
    pub fn check(&self, fingerprint: &[u64]) -> Option<&CalibrationEntry> {
        for entry in self.entries.values() {
            if hamming_similarity(fingerprint, &entry.trigger) > 0.8 {
                return Some(entry);
            }
        }
        None
    }
    
    /// Add new calibration
    pub fn add(&mut self, trigger: Vec<u64>, correction: CorrectionType) -> u8 {
        let slot = self.next_slot;
        self.next_slot = self.next_slot.wrapping_add(1);
        
        self.entries.insert(slot, CalibrationEntry {
            trigger,
            correction,
            confidence: 0.5,
            usage_count: 0,
        });
        
        slot
    }
    
    /// Update calibration confidence on usage
    pub fn used(&mut self, slot: u8) {
        if let Some(entry) = self.entries.get_mut(&slot) {
            entry.usage_count += 1;
            entry.confidence = (entry.confidence * 0.9 + 0.1).min(1.0);
        }
    }
}

// =============================================================================
// CODEBOOK (Learned NSM Extensions)
// =============================================================================

#[derive(Clone, Debug, Default)]
pub struct Codebook {
    /// Learned concept entries (slots 0x41-0xFF in prefix 0x0C)
    entries: HashMap<u8, CodebookEntry>,
    
    /// Next available extension slot
    next_slot: u8,
}

#[derive(Clone, Debug)]
pub struct CodebookEntry {
    /// Concept fingerprint
    pub fingerprint: Vec<u64>,
    
    /// NSM decomposition (which primitives compose this concept)
    pub nsm_composition: Vec<(u8, f32)>,
    
    /// Label (human-readable)
    pub label: String,
    
    /// Confidence from training
    pub confidence: f32,
    
    /// Usage count
    pub usage_count: u32,
}

impl Codebook {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            next_slot: nsm_slots::EXTENSION_START,
        }
    }
    
    /// Lookup by fingerprint resonance
    pub fn lookup(&self, fingerprint: &[u64], threshold: f32) -> Option<(u8, &CodebookEntry)> {
        let mut best: Option<(u8, f32, &CodebookEntry)> = None;
        
        for (&slot, entry) in &self.entries {
            let sim = hamming_similarity(fingerprint, &entry.fingerprint);
            if sim >= threshold {
                if best.map(|(_, s, _)| sim > s).unwrap_or(true) {
                    best = Some((slot, sim, entry));
                }
            }
        }
        
        best.map(|(slot, _, entry)| (slot, entry))
    }
    
    /// Mint new concept from observed pattern
    pub fn mint(&mut self, fingerprint: Vec<u64>, nsm_composition: Vec<(u8, f32)>, label: String) -> Option<u8> {
        if self.next_slot == 0xFF {
            return None; // Full
        }
        
        let slot = self.next_slot;
        self.next_slot = self.next_slot.wrapping_add(1);
        if self.next_slot == 0 {
            self.next_slot = 0xFF; // Prevent wrap
        }
        
        self.entries.insert(slot, CodebookEntry {
            fingerprint,
            nsm_composition,
            label,
            confidence: 0.5,
            usage_count: 0,
        });
        
        Some(slot)
    }
}

// =============================================================================
// COGNITIVE STACK (Main Integration)
// =============================================================================

#[derive(Clone)]
pub struct CognitiveStack {
    /// Current thinking style
    pub style: ThinkingStyle,
    
    /// Context crystal (Fluid zone 0x10-0x14)
    pub crystal: ContextCrystal,
    
    /// User calibrations (Surface zone 0x0F)
    pub calibration: UserCalibration,
    
    /// Learned codebook (Surface zone 0x0C extensions)
    pub codebook: Codebook,
    
    /// Node zone (committed concepts)
    pub nodes: HashMap<Addr, NodeEntry>,
    
    /// Processing statistics
    pub stats: StackStats,
}

#[derive(Clone, Debug)]
pub struct NodeEntry {
    pub fingerprint: Vec<u64>,
    pub created_at: Instant,
    pub source: String,
}

#[derive(Clone, Debug, Default)]
pub struct StackStats {
    pub total_parsed: u64,
    pub flow_count: u64,
    pub hold_count: u64,
    pub block_count: u64,
    pub calibration_applied: u64,
}

impl Default for CognitiveStack {
    fn default() -> Self {
        Self::new(ThinkingStyle::Analytical)
    }
}

impl CognitiveStack {
    pub fn new(style: ThinkingStyle) -> Self {
        Self {
            style,
            crystal: ContextCrystal::new(),
            calibration: UserCalibration::new(),
            codebook: Codebook::new(),
            nodes: HashMap::new(),
            stats: StackStats::default(),
        }
    }
    
    /// Process text through full stack
    pub fn process(&mut self, text: &str) -> ProcessResult {
        self.stats.total_parsed += 1;
        
        // 1. Parse through Grammar Triangle
        let triangle = self.parse(text);
        
        // 2. Apply thinking style modulation
        let adjusted_sd = triangle.collapse.sd + self.style.params().collapse_bias;
        
        // 3. Determine collapse action
        let gate = if adjusted_sd < 0.15 {
            GateState::Flow
        } else if adjusted_sd <= 0.35 {
            GateState::Hold
        } else {
            GateState::Block
        };
        
        // 4. Execute action
        let action_result = match gate {
            GateState::Flow => {
                self.stats.flow_count += 1;
                self.commit_to_node(&triangle)
            }
            GateState::Hold => {
                self.stats.hold_count += 1;
                self.hold_in_crystal(&triangle)
            }
            GateState::Block => {
                self.stats.block_count += 1;
                self.handle_block(&triangle)
            }
        };
        
        // 5. Check for learning opportunities
        self.maybe_learn(&triangle);
        
        ProcessResult {
            triangle,
            gate,
            action_result,
            style: self.style,
        }
    }
    
    /// Parse text to triangle (without committing)
    fn parse(&self, text: &str) -> ParsedTriangle {
        // Tokenize and extract
        let tokens = tokenize(text);
        
        // NSM activation
        let nsm_weights = compute_nsm_weights(&tokens);
        let nsm_top: Vec<(u8, f32)> = nsm_weights.iter()
            .enumerate()
            .filter(|(_, &w)| w > 0.3)
            .map(|(i, &w)| (i as u8, w))
            .collect();
        
        // Template matching
        let template = match_best_template(&tokens, &nsm_weights);
        
        // Speech act
        let speech_act = classify_speech_act(&tokens, &nsm_weights);
        
        // SPO extraction
        let spo = extract_spo(&tokens);
        
        // Qualia
        let qualia = extract_qualia(&tokens, &nsm_weights);
        
        // Generate fingerprint
        let fingerprint = generate_fingerprint(&nsm_weights, &template, &qualia);
        
        // Compute addresses
        let mut addresses = Vec::new();
        for (slot, _) in &nsm_top {
            addresses.push(Addr::new(0x0C, *slot));
        }
        if let Some(ref t) = template {
            addresses.push(Addr::new(0x0D, t.slot));
        }
        addresses.push(Addr::new(0x0E, speech_act.slot));
        
        // Compute collapse state
        let sd = compute_dispersion(&nsm_weights);
        let collapse = CollapseResult {
            gate: if sd < 0.15 { GateState::Flow } 
                  else if sd <= 0.35 { GateState::Hold } 
                  else { GateState::Block },
            sd,
            action: CollapseAction::HoldInCrystal { ttl_seconds: 60 },
        };
        
        ParsedTriangle {
            source: text.to_string(),
            fingerprint,
            addresses,
            spo,
            qualia,
            collapse,
            nsm_top,
            template,
            speech_act,
        }
    }
    
    /// Commit to Node zone (FLOW)
    fn commit_to_node(&mut self, triangle: &ParsedTriangle) -> ActionResult {
        // Allocate address in Node zone
        let slot = (self.nodes.len() % 256) as u8;
        let prefix = 0x80 + ((self.nodes.len() / 256) % 128) as u8;
        let addr = Addr::new(prefix, slot);
        
        self.nodes.insert(addr, NodeEntry {
            fingerprint: triangle.fingerprint.clone(),
            created_at: Instant::now(),
            source: triangle.source.clone(),
        });
        
        ActionResult::Committed { addr }
    }
    
    /// Hold in Crystal (HOLD)
    fn hold_in_crystal(&mut self, triangle: &ParsedTriangle) -> ActionResult {
        self.crystal.insert(triangle, Duration::from_secs(60));
        ActionResult::Held { ttl_seconds: 60 }
    }
    
    /// Handle BLOCK state
    fn handle_block(&mut self, triangle: &ParsedTriangle) -> ActionResult {
        // Check user calibration first
        if let Some(entry) = self.calibration.check(&triangle.fingerprint) {
            self.stats.calibration_applied += 1;
            return ActionResult::CalibrationApplied {
                correction: format!("{:?}", entry.correction),
            };
        }
        
        // Need clarification
        ActionResult::NeedClarification {
            reason: "High dispersion - template match uncertain".to_string(),
        }
    }
    
    /// Check for learning opportunities
    fn maybe_learn(&mut self, triangle: &ParsedTriangle) {
        // If we have a strong NSM pattern not in codebook, consider minting
        if triangle.nsm_top.len() >= 2 {
            let threshold = self.style.params().resonance_threshold;
            if self.codebook.lookup(&triangle.fingerprint, threshold).is_none() {
                // Strong new pattern - could mint
                // (For now, just note it)
            }
        }
    }
    
    /// Query the stack
    pub fn query(&self, text: &str, threshold: f32) -> QueryResult {
        let triangle = self.parse(text);
        
        // Query crystal
        let crystal_matches = self.crystal.query(&triangle.fingerprint, threshold);
        
        // Query nodes
        let mut node_matches = Vec::new();
        for (addr, entry) in &self.nodes {
            let sim = hamming_similarity(&triangle.fingerprint, &entry.fingerprint);
            if sim >= threshold {
                node_matches.push((*addr, sim));
            }
        }
        node_matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Query codebook
        let codebook_match = self.codebook.lookup(&triangle.fingerprint, threshold);
        
        QueryResult {
            query_triangle: triangle,
            crystal_matches,
            node_matches,
            codebook_match: codebook_match.map(|(slot, entry)| (slot, entry.label.clone())),
        }
    }
    
    /// Set thinking style
    pub fn set_style(&mut self, style: ThinkingStyle) {
        self.style = style;
    }
    
    /// Tick (maintenance)
    pub fn tick(&mut self) {
        self.crystal.tick();
    }
}

#[derive(Clone, Debug)]
pub enum ActionResult {
    Committed { addr: Addr },
    Held { ttl_seconds: u64 },
    CalibrationApplied { correction: String },
    NeedClarification { reason: String },
}

#[derive(Clone, Debug)]
pub struct ProcessResult {
    pub triangle: ParsedTriangle,
    pub gate: GateState,
    pub action_result: ActionResult,
    pub style: ThinkingStyle,
}

#[derive(Clone, Debug)]
pub struct QueryResult {
    pub query_triangle: ParsedTriangle,
    pub crystal_matches: Vec<CrystalMatch>,
    pub node_matches: Vec<(Addr, f32)>,
    pub codebook_match: Option<(u8, String)>,
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

fn tokenize(text: &str) -> Vec<Token> {
    text.split_whitespace()
        .enumerate()
        .map(|(i, word)| Token {
            text: word.to_string(),
            index: i,
        })
        .collect()
}

struct Token {
    text: String,
    index: usize,
}

fn compute_nsm_weights(tokens: &[Token]) -> [f32; 65] {
    let mut weights = [0.0f32; 65];
    
    // Simple keyword matching (would use proper NLP in production)
    for token in tokens {
        let lower = token.text.to_lowercase();
        
        // Mental predicates
        if ["want", "desire", "wish", "need"].contains(&lower.as_str()) {
            weights[nsm_slots::WANT as usize] += 0.3;
        }
        if ["know", "understand", "realize"].contains(&lower.as_str()) {
            weights[nsm_slots::KNOW as usize] += 0.3;
        }
        if ["think", "believe", "suppose"].contains(&lower.as_str()) {
            weights[nsm_slots::THINK as usize] += 0.3;
        }
        if ["feel", "sense"].contains(&lower.as_str()) {
            weights[nsm_slots::FEEL as usize] += 0.3;
        }
        
        // Pronouns
        if ["i", "me", "my"].contains(&lower.as_str()) {
            weights[nsm_slots::I as usize] += 0.3;
        }
        if ["you", "your"].contains(&lower.as_str()) {
            weights[nsm_slots::YOU as usize] += 0.3;
        }
        
        // Evaluators
        if ["good", "great", "wonderful"].contains(&lower.as_str()) {
            weights[nsm_slots::GOOD as usize] += 0.3;
        }
        if ["bad", "terrible", "awful"].contains(&lower.as_str()) {
            weights[nsm_slots::BAD as usize] += 0.3;
        }
        
        // Logical
        if lower == "not" || lower == "no" {
            weights[nsm_slots::NOT as usize] += 0.3;
        }
        if lower == "because" {
            weights[nsm_slots::BECAUSE as usize] += 0.3;
        }
        if lower == "if" {
            weights[nsm_slots::IF as usize] += 0.3;
        }
    }
    
    // Clamp to [0, 1]
    for w in &mut weights {
        *w = w.min(1.0);
    }
    
    weights
}

fn match_best_template(tokens: &[Token], nsm_weights: &[f32; 65]) -> Option<TemplateInfo> {
    // Simplified template matching
    let has_verb = tokens.iter().any(|t| {
        ["is", "are", "was", "want", "know", "think", "have", "do", "go", "see"]
            .contains(&t.text.to_lowercase().as_str())
    });
    
    if has_verb {
        if nsm_weights[nsm_slots::WANT as usize] > 0.2 {
            return Some(TemplateInfo {
                slot: template_slots::DESIRE_EXPRESSION,
                name: "desire.expression".to_string(),
                confidence: 0.7,
            });
        }
        Some(TemplateInfo {
            slot: template_slots::TRANSITIVE_DECLARATIVE,
            name: "transitive.declarative".to_string(),
            confidence: 0.5,
        })
    } else {
        None
    }
}

fn classify_speech_act(tokens: &[Token], nsm_weights: &[f32; 65]) -> SpeechActInfo {
    // Check for question
    if tokens.iter().any(|t| t.text.ends_with('?')) {
        return SpeechActInfo {
            slot: speech_act_slots::REQUEST,
            name: "REQUEST".to_string(),
            commitment_type: "desire".to_string(),
        };
    }
    
    // Default assertion
    SpeechActInfo {
        slot: speech_act_slots::ASSERT,
        name: "ASSERT".to_string(),
        commitment_type: "belief".to_string(),
    }
}

fn extract_spo(tokens: &[Token]) -> SpoExtraction {
    let mut spo = SpoExtraction::default();
    
    for (i, token) in tokens.iter().enumerate() {
        let lower = token.text.to_lowercase();
        
        // Simple heuristic: first pronoun/noun is subject
        if spo.subject.is_none() {
            if ["i", "you", "he", "she", "it", "we", "they"].contains(&lower.as_str()) {
                spo.subject = Some(token.text.clone());
                continue;
            }
        }
        
        // First verb after subject is predicate
        if spo.subject.is_some() && spo.predicate.is_none() {
            if ["is", "are", "was", "want", "know", "think", "have", "do", "go", "see", "feel"]
                .contains(&lower.as_str()) {
                spo.predicate = Some(token.text.clone());
                continue;
            }
        }
        
        // Anything after predicate could be object
        if spo.predicate.is_some() && spo.object.is_none() && i > 1 {
            spo.object = Some(token.text.clone());
        }
    }
    
    spo
}

fn extract_qualia(tokens: &[Token], nsm_weights: &[f32; 65]) -> [f32; 8] {
    let mut qualia = [0.5f32; 8]; // Default neutral
    
    // arousal, valence, tension, depth, certainty, intimacy, urgency, novelty
    
    // Valence from GOOD/BAD
    qualia[1] = 0.5 + (nsm_weights[nsm_slots::GOOD as usize] - nsm_weights[nsm_slots::BAD as usize]) * 0.4;
    
    // Certainty from KNOW vs MAYBE
    qualia[4] = 0.3 + nsm_weights[nsm_slots::KNOW as usize] * 0.4;
    
    // Intimacy from I/YOU
    if nsm_weights[nsm_slots::I as usize] > 0.2 || nsm_weights[nsm_slots::YOU as usize] > 0.2 {
        qualia[5] = 0.7;
    }
    
    // Arousal from punctuation
    if tokens.iter().any(|t| t.text.contains('!')) {
        qualia[0] = 0.8;
    }
    
    qualia
}

fn generate_fingerprint(nsm_weights: &[f32; 65], template: &Option<TemplateInfo>, qualia: &[f32; 8]) -> Vec<u64> {
    // 10K bits = 157 u64 words
    let mut fp = vec![0u64; 157];
    
    // NSM contribution (bits 0-3999)
    for (i, &weight) in nsm_weights.iter().enumerate() {
        if weight > 0.3 {
            let base_bit = i * 60;
            let num_bits = (weight * 60.0) as usize;
            
            for j in 0..num_bits.min(60) {
                let bit = base_bit + j;
                if bit < 4000 {
                    let word = bit / 64;
                    let offset = bit % 64;
                    fp[word] |= 1u64 << offset;
                }
            }
        }
    }
    
    // Template contribution (bits 4000-5999)
    if let Some(ref t) = template {
        let base_bit = 4000 + (t.slot as usize) * 8;
        let num_bits = (t.confidence * 8.0) as usize;
        
        for j in 0..num_bits.min(8) {
            let bit = base_bit + j;
            if bit < 6000 {
                let word = bit / 64;
                let offset = bit % 64;
                fp[word] |= 1u64 << offset;
            }
        }
    }
    
    // Qualia contribution (bits 6000-7999)
    for (i, &q) in qualia.iter().enumerate() {
        let base_bit = 6000 + i * 250;
        let num_bits = (q * 250.0) as usize;
        
        for j in 0..num_bits.min(250) {
            let bit = base_bit + j;
            if bit < 8000 {
                let word = bit / 64;
                let offset = bit % 64;
                fp[word] |= 1u64 << offset;
            }
        }
    }
    
    fp
}

fn compute_dispersion(nsm_weights: &[f32; 65]) -> f32 {
    let active: Vec<f32> = nsm_weights.iter().filter(|&&w| w > 0.1).copied().collect();
    
    if active.len() < 2 {
        return 0.0;
    }
    
    let mean = active.iter().sum::<f32>() / active.len() as f32;
    let variance = active.iter().map(|w| (w - mean).powi(2)).sum::<f32>() / active.len() as f32;
    variance.sqrt()
}

fn hamming_similarity(a: &[u64], b: &[u64]) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    
    let total_bits = a.len() * 64;
    let differing_bits: u32 = a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum();
    
    1.0 - (differing_bits as f32 / total_bits as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_stack_process() {
        let mut stack = CognitiveStack::new(ThinkingStyle::Analytical);
        
        let result = stack.process("I want to understand this");
        
        println!("Gate: {:?}", result.gate);
        println!("Action: {:?}", result.action_result);
        println!("NSM top: {:?}", result.triangle.nsm_top);
        
        assert!(result.triangle.nsm_top.iter().any(|(slot, _)| *slot == nsm_slots::WANT));
        assert!(result.triangle.nsm_top.iter().any(|(slot, _)| *slot == nsm_slots::I));
    }
    
    #[test]
    fn test_style_modulation() {
        let mut stack_analytical = CognitiveStack::new(ThinkingStyle::Analytical);
        let mut stack_creative = CognitiveStack::new(ThinkingStyle::Creative);
        
        let text = "Maybe I think something";
        
        let result_a = stack_analytical.process(text);
        let result_c = stack_creative.process(text);
        
        // Creative should have higher collapse_bias, so might have different gate state
        println!("Analytical gate: {:?}", result_a.gate);
        println!("Creative gate: {:?}", result_c.gate);
    }
    
    #[test]
    fn test_crystal_insert_query() {
        let mut stack = CognitiveStack::new(ThinkingStyle::Analytical);
        
        stack.process("I love this beautiful day");
        stack.process("The weather is wonderful");
        
        let result = stack.query("I feel good today", 0.3);
        
        println!("Crystal matches: {}", result.crystal_matches.len());
        println!("Node matches: {}", result.node_matches.len());
    }
}
