//! Layer 3: Temporal Hysteresis — State Persistence
//!
//! Prevents thrashing between states. Each state has a minimum dwell time
//! before transition is allowed. This avoids oscillation from brief perturbations.

/// Temporal hysteresis prevents thrashing between states.
#[derive(Debug, Clone)]
pub struct TemporalHysteresis {
    /// Current state entry timestamp (monotonic ticks)
    pub entered_at: u64,
    /// Minimum ticks before state can change (per-state)
    pub dwell_times: DwellConfig,
}

/// Dwell time configuration (in ticks).
#[derive(Debug, Clone)]
pub struct DwellConfig {
    /// Minimum time in high-trust states before degrading
    pub trust_high_min: u64,
    /// Minimum time in low-trust states before improving
    pub trust_low_min: u64,
    /// Minimum time in DK position before changing
    pub dk_position_min: u64,
    /// Minimum time in homeostasis state before changing
    pub homeostasis_min: u64,
}

impl Default for DwellConfig {
    fn default() -> Self {
        Self {
            trust_high_min: 120,   // ~2 min at 1Hz
            trust_low_min: 180,    // ~3 min
            dk_position_min: 300,  // ~5 min
            homeostasis_min: 600,  // ~10 min
        }
    }
}

impl TemporalHysteresis {
    pub fn new() -> Self {
        Self {
            entered_at: 0,
            dwell_times: DwellConfig::default(),
        }
    }

    /// Whether enough time has passed to allow a state transition.
    pub fn can_transition(&self, current_tick: u64, min_dwell: u64) -> bool {
        current_tick.saturating_sub(self.entered_at) >= min_dwell
    }

    /// Record state entry.
    pub fn enter_state(&mut self, current_tick: u64) {
        self.entered_at = current_tick;
    }

    /// Check if trust state can transition.
    pub fn can_trust_transition(&self, current_tick: u64, currently_high: bool) -> bool {
        let min = if currently_high {
            self.dwell_times.trust_high_min
        } else {
            self.dwell_times.trust_low_min
        };
        self.can_transition(current_tick, min)
    }

    /// Check if DK position can transition.
    pub fn can_dk_transition(&self, current_tick: u64) -> bool {
        self.can_transition(current_tick, self.dwell_times.dk_position_min)
    }

    /// Check if homeostasis state can transition.
    pub fn can_homeostasis_transition(&self, current_tick: u64) -> bool {
        self.can_transition(current_tick, self.dwell_times.homeostasis_min)
    }
}

impl Default for TemporalHysteresis {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dwell_blocks_transition() {
        let mut h = TemporalHysteresis::new();
        h.enter_state(100);
        assert!(!h.can_transition(150, 120)); // Only 50 ticks, need 120
        assert!(h.can_transition(250, 120));  // 150 ticks, need 120
    }

    #[test]
    fn test_trust_transitions() {
        let mut h = TemporalHysteresis::new();
        h.enter_state(0);
        assert!(!h.can_trust_transition(100, true)); // Need 120
        assert!(h.can_trust_transition(200, true));   // 200 > 120
        assert!(!h.can_trust_transition(100, false)); // Need 180
        assert!(h.can_trust_transition(200, false));   // 200 > 180
    }
}
