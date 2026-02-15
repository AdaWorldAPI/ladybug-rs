//! Firefly Scheduler — MUL-Driven Parallel Execution with SIMD Fan-In
//!
//! The scheduler consumes a `MulSnapshot` and selects an execution mode:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │  MUL State        │  Mode      │  Behavior                         │
//! ├───────────────────┼────────────┼───────────────────────────────────┤
//! │  Flow + high mod  │  Sprint    │  Full pipeline, N-lane fan-out    │
//! │  Flow + moderate  │  Stream    │  Steady single-lane, sequential   │
//! │  Boredom          │  Burst     │  Inject novelty, random walks     │
//! │  Anxiety          │  Chunk     │  Small batches, verify each       │
//! │  Apathy / blocked │  Idle      │  Heartbeat only, conservation     │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Fan-Out / Fan-In
//!
//! **FORK**: Dispatch N copies of a frame across N lanes (parallel executors).
//! **JOIN**: Collect results via SIMD majority-vote bundle (VSA fan-in).
//!
//! The bundled result is the consensus fingerprint — bits that appear in
//! more than half the lane results survive. This is content-addressable
//! consensus: no coordinator, no voting protocol, just XOR + popcount.

use super::executor::{ExecResult, Executor};
use super::firefly_frame::FireflyFrame;
use crate::mul::{HomeostasisState, MulSnapshot};
use crate::storage::FINGERPRINT_WORDS;

// =============================================================================
// EXECUTION MODE
// =============================================================================

/// Execution mode derived from MUL state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Full pipeline, parallel N-lane fan-out. Max throughput.
    Sprint,
    /// Steady sequential frames, normal cadence. Default productive mode.
    Stream,
    /// Inject novelty — exploratory frames, random walks, perturbation.
    Burst,
    /// Small batches with verification checkpoints. Cautious processing.
    Chunk,
    /// Minimal heartbeat. Conservation mode when gate blocked or apathetic.
    Idle,
}

impl ExecutionMode {
    /// Number of parallel lanes to use.
    pub fn lane_count(&self) -> usize {
        match self {
            Self::Sprint => 8,  // full fan-out
            Self::Stream => 1,  // single lane
            Self::Burst => 4,   // moderate exploration
            Self::Chunk => 2,   // small batches, verify after each
            Self::Idle => 0,    // no execution
        }
    }

    /// Max frames per dispatch cycle.
    pub fn batch_size(&self) -> usize {
        match self {
            Self::Sprint => 64,
            Self::Stream => 8,
            Self::Burst => 16,
            Self::Chunk => 4,
            Self::Idle => 0,
        }
    }

    /// Whether results should be bundled (SIMD majority vote).
    pub fn bundle_results(&self) -> bool {
        matches!(self, Self::Sprint | Self::Burst)
    }

    /// Whether to inject noise/perturbation frames.
    pub fn inject_novelty(&self) -> bool {
        matches!(self, Self::Burst)
    }

    /// Whether to verify each chunk before proceeding.
    pub fn verify_chunks(&self) -> bool {
        matches!(self, Self::Chunk)
    }
}

// =============================================================================
// DISPATCH PLAN
// =============================================================================

/// A dispatch plan produced by the scheduler for one cycle.
#[derive(Debug, Clone)]
pub struct DispatchPlan {
    /// Selected execution mode
    pub mode: ExecutionMode,
    /// Frames to execute (distributed across lanes)
    pub frames: Vec<FireflyFrame>,
    /// Lane assignments (frame index → lane id)
    pub lane_assignments: Vec<u8>,
    /// Free will modifier at time of planning
    pub modifier: f32,
    /// Whether this plan requires fan-in bundling
    pub needs_bundle: bool,
}

impl DispatchPlan {
    /// Frames assigned to a specific lane.
    pub fn frames_for_lane(&self, lane_id: u8) -> Vec<&FireflyFrame> {
        self.frames
            .iter()
            .enumerate()
            .filter(|(i, _)| self.lane_assignments.get(*i) == Some(&lane_id))
            .map(|(_, f)| f)
            .collect()
    }

    /// Number of active lanes.
    pub fn active_lanes(&self) -> usize {
        if self.lane_assignments.is_empty() {
            return 0;
        }
        let max = self.lane_assignments.iter().copied().max().unwrap_or(0);
        (max as usize) + 1
    }
}

// =============================================================================
// SIMD BUNDLE COLLECTOR (FAN-IN)
// =============================================================================

/// Collects parallel lane results and bundles via majority vote.
///
/// This is VSA fan-in: each lane produces a fingerprint, and the bundle
/// is the consensus. Bits present in >50% of results survive.
/// Uses word-level popcount for SIMD-friendly operation.
#[derive(Debug)]
pub struct BundleCollector {
    /// Accumulated fingerprints from lanes
    results: Vec<[u64; FINGERPRINT_WORDS]>,
    /// Expected number of results
    expected: usize,
}

impl BundleCollector {
    pub fn new(expected_lanes: usize) -> Self {
        Self {
            results: Vec::with_capacity(expected_lanes),
            expected: expected_lanes,
        }
    }

    /// Add a lane result.
    pub fn push(&mut self, fingerprint: [u64; FINGERPRINT_WORDS]) {
        self.results.push(fingerprint);
    }

    /// Whether all expected results have arrived.
    pub fn is_complete(&self) -> bool {
        self.results.len() >= self.expected
    }

    /// Number of results collected so far.
    pub fn collected(&self) -> usize {
        self.results.len()
    }

    /// SIMD-friendly majority vote bundle.
    ///
    /// For each bit position across all 256 words (16,384 bits):
    /// count how many results have that bit set. If > N/2, set it.
    ///
    /// Uses word-level operations: for each word position, count
    /// set bits across all results using XOR chains and popcount.
    pub fn bundle(&self) -> [u64; FINGERPRINT_WORDS] {
        let n = self.results.len();
        if n == 0 {
            return [0u64; FINGERPRINT_WORDS];
        }
        if n == 1 {
            return self.results[0];
        }

        let threshold = n / 2;
        let mut output = [0u64; FINGERPRINT_WORDS];

        // For each word position (256 words = 16,384 bits)
        for w in 0..FINGERPRINT_WORDS {
            let mut result_word: u64 = 0;

            // For each bit in this word
            for bit in 0..64u32 {
                let mask = 1u64 << bit;
                let mut count = 0usize;

                // Count across all results — this is the hot loop
                // On AVX-512: 256 results × 256 words = ~65K iterations
                // On scalar: same but popcount is per-word
                for result in &self.results {
                    if result[w] & mask != 0 {
                        count += 1;
                    }
                }

                if count > threshold {
                    result_word |= mask;
                }
            }

            output[w] = result_word;
        }

        output
    }

    /// Fast bundle using columnar popcount.
    ///
    /// Instead of bit-by-bit, accumulate per-bit counts in u16 arrays.
    /// Then threshold. ~4x faster than naive for large N.
    pub fn bundle_fast(&self) -> [u64; FINGERPRINT_WORDS] {
        let n = self.results.len();
        if n == 0 {
            return [0u64; FINGERPRINT_WORDS];
        }
        if n == 1 {
            return self.results[0];
        }

        let threshold = (n / 2) as u16;
        let mut output = [0u64; FINGERPRINT_WORDS];

        // Process 64 bits at a time using per-bit counters
        // 64 counters per word position
        for w in 0..FINGERPRINT_WORDS {
            let mut counts = [0u16; 64];

            // Accumulate: for each result, splat its word bits into counters
            for result in &self.results {
                let word = result[w];
                // Unrolled bit extraction — compiler vectorizes this
                for bit in 0..64 {
                    counts[bit] += ((word >> bit) & 1) as u16;
                }
            }

            // Threshold: rebuild output word from counts
            let mut word: u64 = 0;
            for bit in 0..64 {
                if counts[bit] > threshold {
                    word |= 1u64 << bit;
                }
            }
            output[w] = word;
        }

        output
    }
}

// =============================================================================
// FIREFLY SCHEDULER
// =============================================================================

/// MUL-driven execution scheduler.
///
/// Reads the metacognitive state and decides HOW to execute:
/// - Sprint (parallel fan-out + SIMD bundle)
/// - Stream (steady sequential)
/// - Burst (novelty injection)
/// - Chunk (cautious batches)
/// - Idle (conservation)
pub struct FireflyScheduler {
    /// Current execution mode
    mode: ExecutionMode,
    /// Lane executors
    executors: Vec<Executor>,
    /// Number of lanes
    num_lanes: usize,
    /// Cycle counter
    cycle: u64,
    /// Total frames dispatched
    frames_dispatched: u64,
    /// Total bundles completed
    bundles_completed: u64,
}

impl FireflyScheduler {
    /// Create with a maximum lane count.
    pub fn new(max_lanes: usize) -> Self {
        let max_lanes = max_lanes.max(1).min(16);
        let executors = (0..max_lanes).map(|_| Executor::new()).collect();
        Self {
            mode: ExecutionMode::Stream,
            executors,
            num_lanes: max_lanes,
            cycle: 0,
            frames_dispatched: 0,
            bundles_completed: 0,
        }
    }

    /// Select execution mode from MUL snapshot.
    pub fn select_mode(&mut self, snapshot: &MulSnapshot) -> ExecutionMode {
        let mode = if !snapshot.gate_open {
            // Gate blocked → idle
            ExecutionMode::Idle
        } else {
            let modifier = snapshot.free_will_modifier.value();
            match snapshot.homeostasis_state {
                HomeostasisState::Flow if modifier > 0.7 => ExecutionMode::Sprint,
                HomeostasisState::Flow => ExecutionMode::Stream,
                HomeostasisState::Boredom => ExecutionMode::Burst,
                HomeostasisState::Anxiety => ExecutionMode::Chunk,
                HomeostasisState::Apathy => ExecutionMode::Idle,
            }
        };
        self.mode = mode;
        mode
    }

    /// Current mode.
    pub fn mode(&self) -> ExecutionMode {
        self.mode
    }

    /// Plan a dispatch cycle for the given frames.
    ///
    /// Assigns frames to lanes based on current mode and returns
    /// a DispatchPlan that the caller can execute.
    pub fn plan(&self, frames: Vec<FireflyFrame>, modifier: f32) -> DispatchPlan {
        let lane_count = self.mode.lane_count().min(self.num_lanes);

        if lane_count == 0 || frames.is_empty() {
            return DispatchPlan {
                mode: self.mode,
                frames: vec![],
                lane_assignments: vec![],
                modifier,
                needs_bundle: false,
            };
        }

        // Distribute frames across lanes (round-robin)
        let lane_assignments: Vec<u8> = frames
            .iter()
            .enumerate()
            .map(|(i, _)| (i % lane_count) as u8)
            .collect();

        DispatchPlan {
            mode: self.mode,
            frames,
            lane_assignments,
            modifier,
            needs_bundle: self.mode.bundle_results(),
        }
    }

    /// Execute a dispatch plan and return results.
    ///
    /// For Sprint/Burst modes, results are bundled via SIMD majority vote.
    /// For Stream/Chunk modes, results are returned sequentially.
    pub fn execute(&mut self, plan: &DispatchPlan) -> SchedulerResult {
        self.cycle += 1;
        let lane_count = plan.active_lanes().min(self.executors.len());

        if lane_count == 0 || plan.frames.is_empty() {
            return SchedulerResult {
                mode: plan.mode,
                cycle: self.cycle,
                lane_results: vec![],
                bundled: None,
                frames_executed: 0,
            };
        }

        // Execute frames on their assigned lanes
        let mut lane_results: Vec<Vec<ExecResult>> = vec![Vec::new(); lane_count];
        let mut frames_executed = 0u64;

        for (i, frame) in plan.frames.iter().enumerate() {
            let lane_id = plan.lane_assignments[i] as usize;
            if lane_id < self.executors.len() {
                let result = self.executors[lane_id].execute(frame);
                lane_results[lane_id].push(result);
                frames_executed += 1;
            }
        }

        self.frames_dispatched += frames_executed;

        // Bundle if needed (fan-in via SIMD majority vote)
        let bundled = if plan.needs_bundle {
            let mut collector = BundleCollector::new(lane_count);

            // Collect fingerprint results from each lane
            for lane in &lane_results {
                for result in lane {
                    if let ExecResult::Ok(Some(fp)) = result {
                        collector.push(*fp);
                    }
                }
            }

            if collector.collected() > 1 {
                self.bundles_completed += 1;
                Some(collector.bundle_fast())
            } else {
                None
            }
        } else {
            None
        };

        SchedulerResult {
            mode: plan.mode,
            cycle: self.cycle,
            lane_results,
            bundled,
            frames_executed,
        }
    }

    /// Convenience: select mode + plan + execute in one call.
    pub fn dispatch(
        &mut self,
        snapshot: &MulSnapshot,
        frames: Vec<FireflyFrame>,
    ) -> SchedulerResult {
        let _mode = self.select_mode(snapshot);
        let plan = self.plan(frames, snapshot.confidence());
        self.execute(&plan)
    }

    /// Stats.
    pub fn stats(&self) -> SchedulerStats {
        SchedulerStats {
            cycles: self.cycle,
            frames_dispatched: self.frames_dispatched,
            bundles_completed: self.bundles_completed,
            current_mode: self.mode,
            num_lanes: self.num_lanes,
        }
    }
}

// =============================================================================
// RESULT TYPES
// =============================================================================

/// Result of a scheduler dispatch cycle.
#[derive(Debug)]
pub struct SchedulerResult {
    /// Mode that was used
    pub mode: ExecutionMode,
    /// Cycle number
    pub cycle: u64,
    /// Per-lane results
    pub lane_results: Vec<Vec<ExecResult>>,
    /// Bundled consensus fingerprint (Sprint/Burst modes only)
    pub bundled: Option<[u64; FINGERPRINT_WORDS]>,
    /// Total frames executed this cycle
    pub frames_executed: u64,
}

impl SchedulerResult {
    /// Whether any lane produced a result.
    pub fn has_results(&self) -> bool {
        self.lane_results.iter().any(|lane| !lane.is_empty())
    }

    /// Total results across all lanes.
    pub fn total_results(&self) -> usize {
        self.lane_results.iter().map(|lane| lane.len()).sum()
    }
}

/// Scheduler statistics.
#[derive(Debug, Clone)]
pub struct SchedulerStats {
    pub cycles: u64,
    pub frames_dispatched: u64,
    pub bundles_completed: u64,
    pub current_mode: ExecutionMode,
    pub num_lanes: usize,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mul::{
        DKPosition, FalseFlowSeverity, FreeWillModifier, GateBlockReason, MetaUncertaintyLayer,
        MulSnapshot, RiskVector, TrustLevel, TrustQualia,
    };

    fn make_snapshot(
        homeostasis: HomeostasisState,
        modifier_val: f32,
        gate_open: bool,
    ) -> MulSnapshot {
        MulSnapshot {
            trust_level: TrustLevel::Solid,
            dk_position: DKPosition::SlopeOfEnlightenment,
            homeostasis_state: homeostasis,
            false_flow_severity: FalseFlowSeverity::None,
            free_will_modifier: FreeWillModifier::from_value(modifier_val),
            gate_open,
            gate_block_reason: if gate_open {
                None
            } else {
                Some(GateBlockReason::MountStupid)
            },
            allostatic_load: 0.1,
        }
    }

    fn make_nop_frames(n: usize) -> Vec<FireflyFrame> {
        use crate::fabric::firefly_frame::FrameBuilder;
        (0..n)
            .map(|_| FrameBuilder::new(0).lane(0).nop())
            .collect()
    }

    // --- Mode selection tests ---

    #[test]
    fn test_sprint_mode() {
        let mut sched = FireflyScheduler::new(8);
        let snap = make_snapshot(HomeostasisState::Flow, 0.9, true);
        assert_eq!(sched.select_mode(&snap), ExecutionMode::Sprint);
    }

    #[test]
    fn test_stream_mode() {
        let mut sched = FireflyScheduler::new(8);
        let snap = make_snapshot(HomeostasisState::Flow, 0.5, true);
        assert_eq!(sched.select_mode(&snap), ExecutionMode::Stream);
    }

    #[test]
    fn test_burst_mode() {
        let mut sched = FireflyScheduler::new(8);
        let snap = make_snapshot(HomeostasisState::Boredom, 0.8, true);
        assert_eq!(sched.select_mode(&snap), ExecutionMode::Burst);
    }

    #[test]
    fn test_chunk_mode() {
        let mut sched = FireflyScheduler::new(8);
        let snap = make_snapshot(HomeostasisState::Anxiety, 0.5, true);
        assert_eq!(sched.select_mode(&snap), ExecutionMode::Chunk);
    }

    #[test]
    fn test_idle_on_gate_blocked() {
        let mut sched = FireflyScheduler::new(8);
        let snap = make_snapshot(HomeostasisState::Flow, 0.9, false);
        assert_eq!(sched.select_mode(&snap), ExecutionMode::Idle);
    }

    #[test]
    fn test_idle_on_apathy() {
        let mut sched = FireflyScheduler::new(8);
        let snap = make_snapshot(HomeostasisState::Apathy, 0.5, true);
        assert_eq!(sched.select_mode(&snap), ExecutionMode::Idle);
    }

    // --- Lane count tests ---

    #[test]
    fn test_lane_counts() {
        assert_eq!(ExecutionMode::Sprint.lane_count(), 8);
        assert_eq!(ExecutionMode::Stream.lane_count(), 1);
        assert_eq!(ExecutionMode::Burst.lane_count(), 4);
        assert_eq!(ExecutionMode::Chunk.lane_count(), 2);
        assert_eq!(ExecutionMode::Idle.lane_count(), 0);
    }

    // --- Dispatch plan tests ---

    #[test]
    fn test_plan_distributes_frames() {
        let sched = FireflyScheduler::new(8);
        let frames = make_nop_frames(16);
        let plan = DispatchPlan {
            mode: ExecutionMode::Sprint,
            frames: frames.clone(),
            lane_assignments: (0..16).map(|i| (i % 8) as u8).collect(),
            modifier: 0.9,
            needs_bundle: true,
        };
        assert_eq!(plan.active_lanes(), 8);
        assert_eq!(plan.frames_for_lane(0).len(), 2);
        assert_eq!(plan.frames_for_lane(7).len(), 2);
    }

    #[test]
    fn test_idle_plan_empty() {
        let mut sched = FireflyScheduler::new(8);
        sched.mode = ExecutionMode::Idle;
        let plan = sched.plan(make_nop_frames(10), 0.0);
        assert!(plan.frames.is_empty());
        assert!(!plan.needs_bundle);
    }

    // --- Bundle collector tests ---

    #[test]
    fn test_bundle_majority_vote() {
        let mut collector = BundleCollector::new(3);

        // Three results where word 0 has bits 0,1 set in majority
        let mut fp1 = [0u64; FINGERPRINT_WORDS];
        fp1[0] = 0b111; // bits 0,1,2
        let mut fp2 = [0u64; FINGERPRINT_WORDS];
        fp2[0] = 0b011; // bits 0,1
        let mut fp3 = [0u64; FINGERPRINT_WORDS];
        fp3[0] = 0b101; // bits 0,2

        collector.push(fp1);
        collector.push(fp2);
        collector.push(fp3);

        let bundled = collector.bundle_fast();
        // Bit 0: 3/3 → set
        // Bit 1: 2/3 → set (>1)
        // Bit 2: 2/3 → set (>1)
        assert_eq!(bundled[0], 0b111);
    }

    #[test]
    fn test_bundle_minority_dropped() {
        let mut collector = BundleCollector::new(3);

        let mut fp1 = [0u64; FINGERPRINT_WORDS];
        fp1[0] = 0b1000; // bit 3 only in one result
        let fp2 = [0u64; FINGERPRINT_WORDS]; // all zeros
        let fp3 = [0u64; FINGERPRINT_WORDS]; // all zeros

        collector.push(fp1);
        collector.push(fp2);
        collector.push(fp3);

        let bundled = collector.bundle_fast();
        // Bit 3: 1/3 → NOT set (1 is not > 1)
        assert_eq!(bundled[0], 0);
    }

    #[test]
    fn test_bundle_single_passthrough() {
        let mut collector = BundleCollector::new(1);
        let mut fp = [0u64; FINGERPRINT_WORDS];
        fp[0] = 0xDEADBEEF;
        fp[255] = 0xCAFEBABE;
        collector.push(fp);

        let bundled = collector.bundle_fast();
        assert_eq!(bundled[0], 0xDEADBEEF);
        assert_eq!(bundled[255], 0xCAFEBABE);
    }

    #[test]
    fn test_bundle_consistency() {
        // bundle() and bundle_fast() should produce identical results
        let mut collector = BundleCollector::new(5);
        for i in 0..5u64 {
            let mut fp = [0u64; FINGERPRINT_WORDS];
            fp[0] = i * 0x1111_1111_1111_1111;
            fp[1] = !i * 0x2222_2222_2222_2222;
            collector.push(fp);
        }

        let slow = collector.bundle();
        let fast = collector.bundle_fast();
        assert_eq!(slow, fast);
    }

    // --- Full dispatch tests ---

    #[test]
    fn test_dispatch_sprint() {
        let mut sched = FireflyScheduler::new(8);
        let snap = make_snapshot(HomeostasisState::Flow, 0.9, true);
        let frames = make_nop_frames(8);
        let result = sched.dispatch(&snap, frames);

        assert_eq!(result.mode, ExecutionMode::Sprint);
        assert_eq!(result.frames_executed, 8);
        assert!(result.has_results());
    }

    #[test]
    fn test_dispatch_idle_no_execution() {
        let mut sched = FireflyScheduler::new(8);
        let snap = make_snapshot(HomeostasisState::Flow, 0.9, false); // gate blocked
        let frames = make_nop_frames(8);
        let result = sched.dispatch(&snap, frames);

        assert_eq!(result.mode, ExecutionMode::Idle);
        assert_eq!(result.frames_executed, 0);
    }

    #[test]
    fn test_stats_accumulate() {
        let mut sched = FireflyScheduler::new(4);
        let snap = make_snapshot(HomeostasisState::Flow, 0.5, true);

        sched.dispatch(&snap, make_nop_frames(4));
        sched.dispatch(&snap, make_nop_frames(4));

        let stats = sched.stats();
        assert_eq!(stats.cycles, 2);
        assert_eq!(stats.frames_dispatched, 8);
        assert_eq!(stats.current_mode, ExecutionMode::Stream);
    }

    // --- MUL integration test ---

    #[test]
    fn test_mul_to_scheduler_flow() {
        let mut mul = MetaUncertaintyLayer::new();
        mul.trust_qualia = TrustQualia::uniform(0.9);
        mul.risk_vector = RiskVector::low();

        // Flow + high trust → Sprint
        let snapshot = mul.evaluate(true);
        let mut sched = FireflyScheduler::new(8);
        let mode = sched.select_mode(&snapshot);
        assert_eq!(mode, ExecutionMode::Sprint);

        // Anxiety → Chunk
        for _ in 0..20 {
            mul.tick(0.8, 0.3, 0.95, 0.2); // high challenge, low skill
        }
        let snapshot2 = mul.evaluate(true);
        assert_eq!(snapshot2.homeostasis_state, HomeostasisState::Anxiety);
        let mode2 = sched.select_mode(&snapshot2);
        assert_eq!(mode2, ExecutionMode::Chunk);
    }
}
