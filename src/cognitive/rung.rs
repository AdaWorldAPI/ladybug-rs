//! Rung System — Meaning Depth Levels
//!
//! Rungs 0-9 represent semantic abstraction depth.
//! SD alone does NOT trigger rung elevation.
//!
//! Rung shift occurs ONLY upon:
//! 1. Sustained non-collapse (BLOCK state persists)
//! 2. Predictive failure (P metric drops)
//! 3. Structural mismatch (no legal grammar parse)
//!
//! Uses coarse rung bands (0-2, 3-5, 6-9) for bucket key addressing.

use std::collections::VecDeque;
use std::time::Instant;

use super::collapse_gate::GateState;

// =============================================================================
// TYPES
// =============================================================================

/// Rung level (0-9)
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(u8)]
pub enum RungLevel {
    #[default]
    Surface = 0,        // Literal, immediate meaning
    Shallow = 1,        // Simple inference, common implicature
    Contextual = 2,     // Situation-dependent interpretation
    Analogical = 3,     // Metaphor, similarity-based reasoning
    Abstract = 4,       // Generalized patterns, principles
    Structural = 5,     // Schema-level understanding
    Counterfactual = 6, // What-if reasoning, alternatives
    Meta = 7,           // Reasoning about reasoning
    Recursive = 8,      // Self-referential, strange loops
    Transcendent = 9,   // Beyond normal semantic bounds
}

impl RungLevel {
    /// Get numeric value
    pub fn as_u8(&self) -> u8 {
        *self as u8
    }
    
    /// Create from numeric value
    pub fn from_u8(n: u8) -> Self {
        match n {
            0 => Self::Surface,
            1 => Self::Shallow,
            2 => Self::Contextual,
            3 => Self::Analogical,
            4 => Self::Abstract,
            5 => Self::Structural,
            6 => Self::Counterfactual,
            7 => Self::Meta,
            8 => Self::Recursive,
            _ => Self::Transcendent, // 9+
        }
    }
    
    /// Get the coarse band for bucket addressing
    pub fn band(&self) -> RungBand {
        match self.as_u8() {
            0..=2 => RungBand::Low,
            3..=5 => RungBand::Mid,
            _ => RungBand::High,
        }
    }
    
    /// Get semantic description
    pub fn description(&self) -> &'static str {
        match self {
            Self::Surface => "Literal, immediate meaning",
            Self::Shallow => "Simple inference, common implicature",
            Self::Contextual => "Situation-dependent interpretation",
            Self::Analogical => "Metaphor, similarity-based reasoning",
            Self::Abstract => "Generalized patterns, principles",
            Self::Structural => "Schema-level understanding",
            Self::Counterfactual => "What-if reasoning, alternatives",
            Self::Meta => "Reasoning about reasoning",
            Self::Recursive => "Self-referential, strange loops",
            Self::Transcendent => "Beyond normal semantic bounds",
        }
    }
    
    /// Can elevate to next rung?
    pub fn can_elevate(&self) -> bool {
        self.as_u8() < 9
    }
    
    /// Get next rung (capped at Transcendent)
    pub fn next(&self) -> Self {
        Self::from_u8(self.as_u8().saturating_add(1).min(9))
    }
    
    /// Get previous rung (capped at Surface)
    pub fn prev(&self) -> Self {
        Self::from_u8(self.as_u8().saturating_sub(1))
    }
    
    /// All rungs for iteration
    pub const ALL: [RungLevel; 10] = [
        Self::Surface, Self::Shallow, Self::Contextual,
        Self::Analogical, Self::Abstract, Self::Structural,
        Self::Counterfactual, Self::Meta, Self::Recursive,
        Self::Transcendent,
    ];
}

impl std::fmt::Display for RungLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "R{}: {}", self.as_u8(), self.description())
    }
}

/// Coarse rung band for bucket key
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RungBand {
    Low,  // 0-2: Surface processing
    Mid,  // 3-5: Pattern processing
    High, // 6-9: Meta processing
}

impl RungBand {
    /// Get band string for key generation
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Low => "0-2",
            Self::Mid => "3-5",
            Self::High => "6-9",
        }
    }
}

impl std::fmt::Display for RungBand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// What triggered a rung shift
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RungTrigger {
    /// BLOCK state persisted for N consecutive turns
    SustainedBlock { consecutive_blocks: u32 },
    /// P metric dropped below threshold
    PredictiveFailure { p_metric: f32 },
    /// No legal grammar parse available
    StructuralMismatch,
    /// Explicit elevation request
    Manual,
}

impl std::fmt::Display for RungTrigger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SustainedBlock { consecutive_blocks } => 
                write!(f, "BLOCK persisted for {} turns", consecutive_blocks),
            Self::PredictiveFailure { p_metric } => 
                write!(f, "P metric dropped to {:.3}", p_metric),
            Self::StructuralMismatch => 
                write!(f, "No legal grammar parse"),
            Self::Manual => 
                write!(f, "Manual elevation"),
        }
    }
}

/// Rung shift decision
#[derive(Clone, Debug)]
pub struct RungShiftDecision {
    /// Whether to shift
    pub should_shift: bool,
    /// Current rung
    pub current: RungLevel,
    /// Target rung (if shifting)
    pub target: Option<RungLevel>,
    /// Trigger that caused shift
    pub trigger: Option<RungTrigger>,
    /// Human-readable reason
    pub reason: String,
}

/// Record of a rung shift event
#[derive(Clone, Debug)]
pub struct RungShiftEvent {
    pub from: RungLevel,
    pub to: RungLevel,
    pub trigger: RungTrigger,
    pub timestamp: Instant,
}

/// Thresholds for rung elevation
#[derive(Clone, Debug)]
pub struct RungThresholds {
    /// Consecutive BLOCKs before elevation
    pub sustained_block_turns: u32,
    /// P metric threshold for failure
    pub p_metric_threshold: f32,
    /// Window size for P metric averaging
    pub p_metric_window: usize,
    /// Cooldown between shifts
    pub shift_cooldown: std::time::Duration,
}

impl Default for RungThresholds {
    fn default() -> Self {
        Self {
            sustained_block_turns: 3,
            p_metric_threshold: 0.3,
            p_metric_window: 5,
            shift_cooldown: std::time::Duration::from_secs(10),
        }
    }
}

// =============================================================================
// RUNG STATE
// =============================================================================

/// State tracking for rung system
#[derive(Clone, Debug)]
pub struct RungState {
    /// Current rung level
    pub current: RungLevel,
    /// Consecutive BLOCK turns
    pub consecutive_blocks: u32,
    /// Recent P metric values (sliding window)
    pub recent_p_metrics: VecDeque<f32>,
    /// Whether structural mismatch is active
    pub structural_mismatch: bool,
    /// Last shift time
    pub last_shift_at: Option<Instant>,
    /// History of shifts (last 10)
    pub shift_history: VecDeque<RungShiftEvent>,
    /// Thresholds
    pub thresholds: RungThresholds,
}

impl Default for RungState {
    fn default() -> Self {
        Self::new()
    }
}

impl RungState {
    /// Create initial rung state
    pub fn new() -> Self {
        Self {
            current: RungLevel::Surface,
            consecutive_blocks: 0,
            recent_p_metrics: VecDeque::with_capacity(10),
            structural_mismatch: false,
            last_shift_at: None,
            shift_history: VecDeque::with_capacity(10),
            thresholds: RungThresholds::default(),
        }
    }
    
    /// Create with custom thresholds
    pub fn with_thresholds(thresholds: RungThresholds) -> Self {
        Self {
            thresholds,
            ..Self::new()
        }
    }
    
    /// Update state with new gate result
    pub fn update(&mut self, gate_state: GateState, p_metric: f32, has_legal_parse: bool) {
        // Update consecutive BLOCKs
        if gate_state == GateState::Block {
            self.consecutive_blocks += 1;
        } else {
            self.consecutive_blocks = 0;
        }
        
        // Update P metrics (sliding window)
        if self.recent_p_metrics.len() >= self.thresholds.p_metric_window {
            self.recent_p_metrics.pop_front();
        }
        self.recent_p_metrics.push_back(p_metric);
        
        // Update structural mismatch
        self.structural_mismatch = !has_legal_parse;
    }
    
    /// Evaluate whether rung shift should occur
    pub fn evaluate_shift(&self) -> RungShiftDecision {
        let now = Instant::now();
        
        // Check cooldown
        if let Some(last) = self.last_shift_at {
            if now.duration_since(last) < self.thresholds.shift_cooldown {
                return RungShiftDecision {
                    should_shift: false,
                    current: self.current,
                    target: None,
                    trigger: None,
                    reason: "Shift cooldown active".to_string(),
                };
            }
        }
        
        // Check if at max rung
        if !self.current.can_elevate() {
            return RungShiftDecision {
                should_shift: false,
                current: self.current,
                target: None,
                trigger: None,
                reason: "Already at maximum rung".to_string(),
            };
        }
        
        // Trigger 1: Sustained non-collapse (BLOCK persists)
        if self.consecutive_blocks >= self.thresholds.sustained_block_turns {
            return RungShiftDecision {
                should_shift: true,
                current: self.current,
                target: Some(self.current.next()),
                trigger: Some(RungTrigger::SustainedBlock {
                    consecutive_blocks: self.consecutive_blocks,
                }),
                reason: format!("BLOCK persisted for {} consecutive turns", 
                               self.consecutive_blocks),
            };
        }
        
        // Trigger 2: Predictive failure (P metric drops)
        if self.recent_p_metrics.len() >= self.thresholds.p_metric_window {
            let avg_p: f32 = self.recent_p_metrics.iter().sum::<f32>() 
                / self.recent_p_metrics.len() as f32;
            
            if avg_p < self.thresholds.p_metric_threshold {
                return RungShiftDecision {
                    should_shift: true,
                    current: self.current,
                    target: Some(self.current.next()),
                    trigger: Some(RungTrigger::PredictiveFailure { p_metric: avg_p }),
                    reason: format!("Average P metric ({:.3}) below threshold ({:.3})",
                                   avg_p, self.thresholds.p_metric_threshold),
                };
            }
        }
        
        // Trigger 3: Structural mismatch (no legal parse)
        if self.structural_mismatch {
            return RungShiftDecision {
                should_shift: true,
                current: self.current,
                target: Some(self.current.next()),
                trigger: Some(RungTrigger::StructuralMismatch),
                reason: "No legal grammar parse available".to_string(),
            };
        }
        
        // No shift needed
        RungShiftDecision {
            should_shift: false,
            current: self.current,
            target: None,
            trigger: None,
            reason: "No elevation trigger active".to_string(),
        }
    }
    
    /// Apply a shift decision
    pub fn apply_shift(&mut self, decision: &RungShiftDecision) {
        if !decision.should_shift {
            return;
        }
        
        if let (Some(target), Some(trigger)) = (decision.target, decision.trigger) {
            let event = RungShiftEvent {
                from: self.current,
                to: target,
                trigger,
                timestamp: Instant::now(),
            };
            
            // Update state
            self.current = target;
            self.consecutive_blocks = 0;
            self.structural_mismatch = false;
            self.last_shift_at = Some(Instant::now());
            
            // Record in history (keep last 10)
            if self.shift_history.len() >= 10 {
                self.shift_history.pop_front();
            }
            self.shift_history.push_back(event);
        }
    }
    
    /// Manually shift to a specific rung
    pub fn manual_shift(&mut self, target: RungLevel) {
        if target == self.current {
            return;
        }
        
        let event = RungShiftEvent {
            from: self.current,
            to: target,
            trigger: RungTrigger::Manual,
            timestamp: Instant::now(),
        };
        
        self.current = target;
        self.last_shift_at = Some(Instant::now());
        
        if self.shift_history.len() >= 10 {
            self.shift_history.pop_front();
        }
        self.shift_history.push_back(event);
    }
    
    /// Reset to surface rung
    pub fn reset(&mut self) {
        self.current = RungLevel::Surface;
        self.consecutive_blocks = 0;
        self.recent_p_metrics.clear();
        self.structural_mismatch = false;
        // Keep history
    }
    
    /// Get current band
    pub fn band(&self) -> RungBand {
        self.current.band()
    }
    
    /// Generate bucket key suffix based on rung
    pub fn bucket_key_suffix(&self) -> String {
        format!("_r{}", self.band())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rung_levels() {
        assert_eq!(RungLevel::Surface.as_u8(), 0);
        assert_eq!(RungLevel::Transcendent.as_u8(), 9);
        assert_eq!(RungLevel::from_u8(5), RungLevel::Structural);
        assert_eq!(RungLevel::from_u8(100), RungLevel::Transcendent);
    }
    
    #[test]
    fn test_rung_bands() {
        assert_eq!(RungLevel::Surface.band(), RungBand::Low);
        assert_eq!(RungLevel::Contextual.band(), RungBand::Low);
        assert_eq!(RungLevel::Analogical.band(), RungBand::Mid);
        assert_eq!(RungLevel::Structural.band(), RungBand::Mid);
        assert_eq!(RungLevel::Counterfactual.band(), RungBand::High);
        assert_eq!(RungLevel::Transcendent.band(), RungBand::High);
    }
    
    #[test]
    fn test_sustained_block_trigger() {
        let mut state = RungState::new();
        state.thresholds.sustained_block_turns = 3;
        
        // Not enough blocks
        state.update(GateState::Block, 0.5, true);
        state.update(GateState::Block, 0.5, true);
        let decision = state.evaluate_shift();
        assert!(!decision.should_shift);
        
        // Third block triggers
        state.update(GateState::Block, 0.5, true);
        let decision = state.evaluate_shift();
        assert!(decision.should_shift);
        assert_eq!(decision.target, Some(RungLevel::Shallow));
    }
    
    #[test]
    fn test_predictive_failure_trigger() {
        let mut state = RungState::new();
        state.thresholds.p_metric_window = 3;
        state.thresholds.p_metric_threshold = 0.3;
        
        // Low P metrics
        state.update(GateState::Flow, 0.1, true);
        state.update(GateState::Flow, 0.2, true);
        state.update(GateState::Flow, 0.1, true);
        
        let decision = state.evaluate_shift();
        assert!(decision.should_shift);
        assert!(matches!(decision.trigger, Some(RungTrigger::PredictiveFailure { .. })));
    }
    
    #[test]
    fn test_structural_mismatch_trigger() {
        let mut state = RungState::new();
        
        state.update(GateState::Flow, 0.8, false); // No legal parse
        
        let decision = state.evaluate_shift();
        assert!(decision.should_shift);
        assert_eq!(decision.trigger, Some(RungTrigger::StructuralMismatch));
    }
    
    #[test]
    fn test_manual_shift() {
        let mut state = RungState::new();
        
        state.manual_shift(RungLevel::Meta);
        assert_eq!(state.current, RungLevel::Meta);
        assert_eq!(state.shift_history.len(), 1);
    }
    
    #[test]
    fn test_cooldown() {
        let mut state = RungState::new();
        state.thresholds.shift_cooldown = std::time::Duration::from_secs(1000); // Long cooldown
        
        // Trigger a shift
        state.consecutive_blocks = 10;
        let decision = state.evaluate_shift();
        assert!(decision.should_shift);
        state.apply_shift(&decision);
        
        // Try to shift again immediately
        state.consecutive_blocks = 10;
        let decision = state.evaluate_shift();
        assert!(!decision.should_shift);
        assert!(decision.reason.contains("cooldown"));
    }
}
