//! Layer 5: False Flow Detector — Coherence Without Progress
//!
//! Detects when the system is spinning its wheels: high coherence (things seem
//! consistent) but low novelty (nothing new is being learned). The "busy work" trap.

use std::collections::VecDeque;

/// False flow severity escalation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FalseFlowSeverity {
    None = 0,
    Caution = 1,
    Warning = 2,
    Severe = 3,
}

impl FalseFlowSeverity {
    pub fn from_bits(bits: u8) -> Self {
        match bits {
            0 => Self::None,
            1 => Self::Caution,
            2 => Self::Warning,
            _ => Self::Severe,
        }
    }
}

/// False flow detector — identifies stagnation disguised as productivity.
#[derive(Debug, Clone)]
pub struct FalseFlowDetector {
    /// Rolling coherence window
    coherence_window: VecDeque<f32>,
    /// Rolling novelty window
    novelty_window: VecDeque<f32>,
    /// Window capacity
    window_size: usize,
    /// Current severity
    pub severity: FalseFlowSeverity,
    /// Ticks in current severity state
    ticks_in_state: u32,
}

impl FalseFlowDetector {
    pub fn new(window_size: usize) -> Self {
        let window_size = window_size.max(5);
        Self {
            coherence_window: VecDeque::with_capacity(window_size),
            novelty_window: VecDeque::with_capacity(window_size),
            window_size,
            severity: FalseFlowSeverity::None,
            ticks_in_state: 0,
        }
    }

    /// Update with new coherence and novelty measurements.
    pub fn tick(&mut self, coherence: f32, novelty: f32) {
        if self.coherence_window.len() >= self.window_size {
            self.coherence_window.pop_front();
            self.novelty_window.pop_front();
        }
        self.coherence_window.push_back(coherence);
        self.novelty_window.push_back(novelty);

        // Need at least 5 samples
        if self.coherence_window.len() < 5 {
            return;
        }

        let n = self.coherence_window.len() as f32;
        let avg_coherence: f32 = self.coherence_window.iter().sum::<f32>() / n;
        let avg_novelty: f32 = self.novelty_window.iter().sum::<f32>() / n;

        // Coherence delta (rate of change)
        let len = self.coherence_window.len();
        let coherence_delta =
            (self.coherence_window[len - 1] - self.coherence_window[len - 2]).abs();

        let is_false_flow =
            avg_coherence > 0.7 && avg_novelty < 0.2 && coherence_delta < 0.05;

        if is_false_flow {
            self.ticks_in_state += 1;
            self.severity = match self.ticks_in_state {
                0..=5 => FalseFlowSeverity::Caution,
                6..=15 => FalseFlowSeverity::Warning,
                _ => FalseFlowSeverity::Severe,
            };
        } else {
            self.ticks_in_state = 0;
            self.severity = FalseFlowSeverity::None;
        }
    }

    /// Whether to force a disruption (style change, topic switch).
    pub fn should_disrupt(&self) -> bool {
        self.severity == FalseFlowSeverity::Severe
    }
}

impl Default for FalseFlowDetector {
    fn default() -> Self {
        Self::new(20)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_false_flow_initially() {
        let ff = FalseFlowDetector::new(20);
        assert_eq!(ff.severity, FalseFlowSeverity::None);
        assert!(!ff.should_disrupt());
    }

    #[test]
    fn test_false_flow_detection() {
        let mut ff = FalseFlowDetector::new(20);
        // Simulate coherent but non-novel activity
        for _ in 0..20 {
            ff.tick(0.85, 0.05); // high coherence, low novelty
        }
        assert_ne!(ff.severity, FalseFlowSeverity::None);
    }

    #[test]
    fn test_severe_triggers_disrupt() {
        let mut ff = FalseFlowDetector::new(20);
        for _ in 0..25 {
            ff.tick(0.9, 0.01);
        }
        assert!(ff.should_disrupt());
    }

    #[test]
    fn test_novelty_prevents_false_flow() {
        let mut ff = FalseFlowDetector::new(20);
        for _ in 0..20 {
            ff.tick(0.85, 0.5); // high coherence AND high novelty = real flow
        }
        assert_eq!(ff.severity, FalseFlowSeverity::None);
    }
}
