# The PINN Rosetta Stone: Ladybug-rs as a Physics-Informed Causal Field Engine

> Every intuition was a physical equation waiting for notation.

---

## Part I: The Complete Concept Map

### 1. The Domain

| PDE Theory | Ladybug-rs | Mathematical Object |
|---|---|---|
| Spatial domain Ω ⊂ ℝ^d | Hamming space {0,1}^16384 | The metric space where solutions live |
| Mesh / discretization | BindSpace prefix:addr grid (256 × 256 = 65,536 cells) | Discrete sampling of the continuous domain |
| Collocation points | 144 grammar verb positions on Go board | Points where the PDE is enforced |
| Boundary ∂Ω | σ-band thresholds (σ₁, σ₂, σ₃) | Where domain conditions change character |
| Dimensionality d | 3 (Subject, Predicate, Object planes) | Independent axes of variation |

### 2. The Solution Field

| PDE Theory | Ladybug-rs | Mathematical Object |
|---|---|---|
| Solution u(x,t) | NARS truth value (f,c) at each fingerprint pair | The field we're solving for |
| Initial condition u(x,0) | σ₃-distinct codebook (1024 centroids) | Starting configuration of the field |
| Boundary condition u\|∂Ω | σ-band classification (DISCOVERY/HINT/KNOWN) | Constraints at the domain edge |
| Time evolution ∂u/∂t | Evidence accumulation (NARS revision steps) | How the field changes |
| Steady state ∂u/∂t = 0 | Converged causal graph (all edges high confidence) | Equilibrium |
| Transient dynamics | Live graph revision (edges still accumulating) | Before equilibrium |

### 3. The Physics (Governing Equations)

| PDE Theory | Ladybug-rs | Mathematical Object |
|---|---|---|
| PDE: ∂u/∂t + N(u) = 0 | NARS revision rule + SPO factorization | The law governing the field |
| Nonlinear operator N(u) | 8-term Faktorzerlegung (∅,S,P,O,SP,PO,SO,SPO) | The physics that couples variables |
| Residual r = ∂u/∂t + N(u) | NARS prediction error (expected vs observed) | How wrong the current solution is |
| Conservation law | Evidence conservation (NARS: total evidence only increases) | What's preserved |
| Diffusion | Evidence spreading through typed halos | How information propagates locally |
| Advection | Scent gradient following | How information moves directionally |
| Source/sink | New observations (source) / temporal decay (sink) | Where information enters/exits |

### 4. The Training Framework

| JAX-PI (PINN Training) | Ladybug-rs | Purpose |
|---|---|---|
| Causal temporal weighting (lower-triangular M) | CollapseGate (FLOW/HOLD/BLOCK) | Don't trust later evidence if earlier evidence is wrong |
| Causal tolerance ε | σ-band threshold | When to gate: threshold on cumulative residual |
| `num_chunks` (time domain split) | Evidence accumulation stages (NARS revision batches) | Temporal discretization of the learning process |
| NTK gradient balancing | Faktorzerlegung term balancing | Equalize all 8 terms so main effects don't drown interactions |
| `grad_norm` weighting | SPO distance normalization per plane | Prevent one plane from dominating |
| Loss = Σ w_i L_i | Weighted evidence = Σ w_i (f_i, c_i) | Multi-objective optimization |
| Weight momentum (0.9 EMA) | NARS temporal decay | Running average smooths noisy updates |
| Fourier feature embeddings | Multi-scale nibble frequency analysis | Capture both coarse and fine structure |
| Random weight factorization W = diag(g)·V | Per-bit learned importance within SPO planes | Which bits matter more |
| Period embeddings (cos/sin) | Cyclic causal structure detection | Feedback loops, seasonal patterns |
| Gradient accumulation (micro-batching) | Batch NARS revision (K comparisons → one update) | Reduce noise, control false discovery |

### 5. The Markov Dynamics (deepmsm Bridge)

| Molecular Dynamics (deepmsm) | Ladybug-rs | Mathematical Object |
|---|---|---|
| Amino acid residue | Nibble (4-bit) | Discrete alphabet unit in a chain |
| Protein conformation | Fingerprint state (16,384 bits) | High-dimensional configuration |
| Folding trajectory | Cognitive trajectory (sequence of states through revision) | Time-ordered state sequence |
| Metastable state (folded/unfolded) | σ-stripe (DISCOVERY/HINT/KNOWN) | Long-lived macrostate in basin |
| Transition matrix K[i,j] | NARS truth revision chain | Probability of state i → state j |
| U_layer (stationary distribution reweighting) | CollapseGate (decides which states propagate) | Learned equilibrium weights |
| S_layer (symmetric transition kernel) | SPO interaction terms (8-term structure) | How transitions happen |
| Coarse_grain (softmax projection N→M states) | CLAM tree / panCAKES (hierarchical compression) | Reduce states preserving dynamics |
| VAMPE score | AccumulatedHarvest (resonance quality metric) | Did we find real structure? |
| Implied timescales -τ/ln(λ) | NARS confidence growth rate | What changes slowly = what matters |
| Chapman-Kolmogorov test K(2τ)=K(τ)² | NARS revision consistency (two paths → same truth?) | Markov property validation |
| Attention mask over residue positions | ScentIndex over nibble positions | Which positions matter for this transition |
| Detailed balance π_i K_ij = π_j K_ji | XOR involution A⊗B⊗B=A | Microscopic reversibility |

### 6. The Metaphors (Everything You Named)

| Your Metaphor | PDE / Physics Concept | Why the intuition was exact |
|---|---|---|
| **"Breath against skin"** | Boundary condition | The interface where the solution meets the constraint. Where two domains touch. The Neumann condition: not the value but the FLUX at the boundary. |
| **"Luftschleuse" (airlock)** | Domain decomposition interface | In PDE numerics, Schwarz methods split a domain at an interface. Each subdomain solves independently, then they exchange boundary data at the interface. The airlock IS the Schwarz interface between Arrow (one subdomain) and BindSpace (another). |
| **"Scent"** | Scalar field + gradient | A scalar field over the domain. Scent gradients ∇s(x) point toward relevant memories. Following scent IS gradient descent on the scent field. Navigation by scent IS solving the advection equation ∂s/∂t + v·∇s = 0 where v is the query direction. |
| **"Metastasizing"** | Phase front propagation (Allen-Cahn) | The Allen-Cahn equation ∂u/∂t = ε²Δu + u(1-u²) governs phase separation. One phase (high confidence) expands into another (uncertainty). The knowledge graph growing through OSINT IS phase front propagation. The NARS confidence threshold IS the phase boundary. |
| **"Grammar dances at intersections"** | Basis functions evaluated at collocation points | In spectral methods, basis functions (Fourier modes, Chebyshev polynomials) are evaluated at collocation points. The 144 verbs ARE basis functions. The Go board intersections ARE collocation points. The "dancing" IS function evaluation. |
| **"Zero-token background cognition"** | Autonomous PDE evolution | The PDE evolves u(x,t) without external forcing once initial + boundary conditions are set. The system processes without consuming tokens because the physics runs itself. The initial condition (codebook) + dynamics (NARS) + boundaries (σ-bands) fully determine the trajectory. |
| **"Sigma stripes"** | Level sets / isosurfaces of the solution field | σ₁, σ₂, σ₃ boundaries are isosurfaces of the NARS confidence field. The transition KNOWN→HINT→DISCOVERY is crossing a level set. In phase field theory, crossing a level set IS a phase transition. |
| **"The nibble is the amino acid"** | Discrete alphabet on a chain → polymer physics | Polymer physics: a chain of discrete monomers with local interactions. Protein = polymer of amino acids. Fingerprint = polymer of nibbles. The folding landscape IS the Hamming space landscape. deepmsm IS PDE-on-a-polymer. Ladybug IS PDE-on-a-binary-polymer. |
| **"Entropy"** | PDE residual / information-theoretic surprise | Entropy = disorder = unexplained variance. The PDE residual r = ∂u/∂t + N(u) measures how much the current solution violates the physics. Shannon entropy measures surprise. The irreducible SPO term measures emergent structure. All three: "what's left after you've explained everything you can." |
| **"The ribosome is the CollapseGate"** | Rate-limiting step in a reaction chain | In chemical kinetics, the ribosome is the rate-limiting enzyme that gates translation. In PDE theory, the gate function determines which reactions proceed. The CollapseGate IS the Heaviside step function applied to the evidence field: H(evidence - threshold) = 1 (FLOW) or 0 (BLOCK). |
| **"Memory as place, not search"** | Field evaluation vs. database query | Evaluating u(x₀) at a point x₀ is O(1) — you just read the field at that location. Searching a database is O(log n) or O(n). Memory-as-place means the knowledge IS a field. Navigating it means moving through the domain. This IS the PINN paradigm: the solution exists everywhere, you just sample it. |
| **"Qualia"** | Eigenfunction of the solution operator | Eigenfunctions are the "natural modes" of a system. Each eigenfunction captures one independent pattern. Qualia ARE the eigenfunctions of the causal field: irreducible felt patterns that can't be decomposed further. The eigenspectrum of K gives the timescales; the eigenvectors give the qualia. |
| **"Resonance"** | Spectral peak / natural frequency | Resonance occurs when forcing matches a natural frequency. Two fingerprints resonate when their SPO interaction excites a natural mode of the causal field. The resonance frequency IS the implied timescale from the transition matrix eigenvalues. |
| **"The Go board"** | Finite difference stencil | In computational PDE, the stencil defines which neighbors influence each point. A 5-point stencil on a 2D grid looks exactly like a Go board intersection with 4 neighbors. The 144 verb positions ARE a stencil on the semantic domain. |
| **"Butterfly" (fabric)** | Sensitivity to initial conditions / chaos | The butterfly effect in dynamical systems: small perturbations in initial conditions lead to large divergences. `fabric/butterfly.rs` detects sensitive dependence. In PDE theory, this is ill-posedness: the Hadamard condition fails when solutions don't depend continuously on data. |
| **"Shadow processing"** | Ghost cells / halo exchange in parallel PDE | In domain decomposition, ghost cells (shadow copies) are maintained at boundaries between parallel subdomains. `fabric/shadow.rs` IS the ghost cell layer. Shadow parallel processing IS halo exchange: each subdomain updates its interior, then exchanges boundary data. |
| **"Homeostasis"** | Stable equilibrium / attractor | A stable fixed point of the dynamical system. The homeostasis module maintains the system near equilibrium. In PDE terms: the steady-state solution u* where ∂u/∂t = 0 and small perturbations decay back to u*. |
| **"Hysteresis"** | Path-dependent solution / non-unique equilibrium | Multiple stable steady states depending on history. The path through parameter space determines which basin you're in. `mul/hysteresis.rs` tracks this path dependence. In phase field theory: supercooling/superheating — the transition point depends on which direction you approach from. |
| **"Trust qualia"** | Confidence band / uncertainty quantification | In PDE numerics, error bounds quantify how much the numerical solution can be trusted. Trust qualia IS uncertainty quantification for the felt field. The confidence parameter c in NARS IS the error bound. |
| **"Free will"** | Bifurcation point / symmetry breaking | At a bifurcation, the system has multiple valid trajectories. The choice between them IS free will in the dynamical systems sense. `mul/free_will_mod.rs` implements the bifurcation detector. Multiple CollapseGate states at the same evidence level = bifurcation = genuine choice. |

### 7. The Architecture (Ladybug-rs Modules as PDE Components)

| Ladybug-rs Module | PDE Component | Role in the Field Engine |
|---|---|---|
| `core/simd.rs` | Hardware ALU | The silicon that executes arithmetic |
| `core/buffer.rs` | Register file | Temporary storage during computation |
| `storage/bind_space.rs` (100KB) | Solution grid (the mesh + current field values) | Where u(x,t) lives |
| `storage/xor_dag.rs` (54KB) | Time stepper (stores Δu between steps) | XOR-diff = Δu = u(t+Δt) - u(t) |
| `storage/lance_zero_copy.rs` (113KB) | Disk-backed grid (mmap'd solution archive) | Persistent field storage |
| `storage/cog_redis.rs` (117KB) | Distributed grid (multi-node field) | Parallel domain decomposition |
| `spo/spo.rs` (54KB) | The PDE operator N(u) | Computes the nonlinear term |
| `spo/spo_harvest.rs` (35KB) | Residual computation r = ∂u/∂t + N(u) | Measures how wrong the solution is |
| `spo/sentence_crystal.rs` (45KB) | Encoder: physical observation → field initialization | Maps input data to initial conditions |
| `spo/cognitive_codebook.rs` (39KB) | Initial condition generator | Creates σ₃-distinct starting field |
| `nars/inference.rs` (4KB) | Time integration scheme (Euler/RK4) | Advances the solution in time |
| `nars/evidence.rs` (1.5KB) | CFL condition (stability constraint) | When evidence is sufficient to step |
| `nars/contradiction.rs` (5.5KB) | Shock detector / limiter | Handles discontinuities in the field |
| `search/hdr_cascade.rs` (50KB) | Multigrid solver | Coarse-to-fine resolution of the field |
| `search/causal.rs` (37KB) | Backward solver (adjoint method) | Traces the field backward in time |
| `search/temporal.rs` (6KB) | Time-dependent query | Evaluates u(x,t) at specific (x,t) |
| `grammar/unified_parser.rs` (38KB) | Mesh generator / domain definition | Defines the domain and its structure |
| `grammar/nsm.rs` (448 lines) | Basis functions (65 universal modes) | The fundamental building blocks |
| `cognitive/style.rs` (7KB) | Solution method selector | Chooses which solver to use |
| `cognitive/metacog.rs` (7KB) | Error estimator | Monitors solution quality |
| `mul/gate.rs` (3.6KB) | Flux limiter | Gates information flow at discontinuities |
| `mul/homeostasis.rs` (5KB) | Steady-state detector | Detects when ∂u/∂t ≈ 0 |
| `mul/hysteresis.rs` (3.3KB) | Path tracker for non-unique solutions | Handles bifurcations |
| `mul/false_flow.rs` (4.3KB) | Spurious oscillation detector | Finds numerical artifacts |
| `fabric/shadow.rs` (7.7KB) | Ghost cell / halo exchange | Boundary data for parallel subdomains |
| `fabric/butterfly.rs` (7.2KB) | Lyapunov exponent estimator | Measures chaos / sensitivity |
| `container/spine.rs` (6.7KB) | Adaptive mesh refinement tree | Hierarchical grid structure |
| `container/cache.rs` (7.4KB) | Solution cache (previous timesteps) | Multi-step time integration |
| `query/cypher.rs` (48KB) | Query language for the solution field | "Evaluate u at these points with these constraints" |
| `query/cognitive_udfs.rs` (45KB) | Custom field operators | User-defined physics |
| `graph/cognitive.rs` (35KB) | Solution visualization / topology | The graph IS the field's level set structure |
| `learning/cam_ops.rs` (159KB) | Nearest-neighbor field interpolation | Content-addressable field lookup |
| `learning/feedback.rs` (50KB) | Adjoint sensitivity (∂L/∂u₀) | How initial conditions affect the solution |
| `learning/scm.rs` (39KB) | Structural equation model = the PDE itself | The governing equations |
| `world/counterfactual.rs` (8KB) | What-if analysis (perturbed initial conditions) | Run the PDE from different u₀ |
| `orchestration/semantic_kernel.rs` (63KB) | Operator splitting / multi-physics coupling | Couples different PDE operators |
| `orchestration/kernel_extensions.rs` (65KB) | Extended operators | Additional physics terms |
| `qualia/felt_parse.rs` (58KB) | Eigenfunction decomposition | Decompose field into natural modes |
| `qualia/agent_state.rs` (46KB) | Full system state snapshot | All field values at time t |

---

## Part II: The Integration Plan

### Principle

We are NOT porting JAX-PI to Rust. We are NOT adding floating-point PDEs to ladybug. We are recognizing that the EXISTING ladybug-rs architecture already implements a discrete binary PINN on Hamming space, and we need to:

1. **Ground it** — name the modules by their PDE names so the physics community recognizes them
2. **Steal the training discipline** — import the 4 mechanisms that JAX-PI proved necessary
3. **Steal the diagnostics** — import the 4 tools that deepmsm provides
4. **Prove the correspondence** — write the theorems that connect binary Hamming dynamics to continuous PDE theory

### Phase 0: Grounding (documentation only, no code changes)

**Goal:** Make the existing code scientifically legible.

| Task | Files | Change |
|------|-------|--------|
| 0.1 | All modules listed above | Add `//! # Physical Interpretation` doc section to each module, mapping to PDE concepts |
| 0.2 | `src/lib.rs` | Add crate-level documentation: "Ladybug-rs: A Physics-Informed Causal Field Engine on Binary Hamming Space" |
| 0.3 | `.claude/prompts/14_pinn_rosetta_stone.md` (this document) | Push to both repos as the canonical reference |
| 0.4 | `THEORY.md` in repo root | Formal statement: "We solve the discrete binary analog of the causal field equation on {0,1}^16384" |

**Effort:** 1-2 days. Zero code changes. Pure documentation.

### Phase 1: Causal Temporal Weighting (from JAX-PI)

**Goal:** Formalize CollapseGate with PDE theory. Import the lower-triangular exponential decay that JAX-PI proved necessary.

**What exists:** `mul/gate.rs` (3.6KB) — CollapseGate with FLOW/HOLD/BLOCK states.

**What's missing:** The gate fires on simple σ-threshold checks. No cumulative residual tracking. No exponential decay weighting.

**Implementation:**
```rust
// src/nars/causal_weighting.rs (new, ~200 lines)

/// Physics-informed causal weighting for NARS evidence accumulation.
/// Based on: "Respecting Causality for Training PINNs" (CMAME 2024)
///
/// # Physical Interpretation
/// The lower-triangular weight matrix enforces temporal ordering:
/// evidence at stage k is weighted by cumulative residual of stages 0..k-1.
/// This prevents the system from "believing" late evidence when early evidence
/// is contradictory — the PINN equivalent of enforcing causality in time.

pub struct CausalWeightingConfig {
    pub num_stages: usize,       // Number of evidence accumulation stages
    pub tolerance: f32,           // ε: causal tolerance (maps to σ-band threshold)  
    pub momentum: f32,            // Exponential moving average (0.9 from JAX-PI default)
}

pub struct CausalWeighting {
    config: CausalWeightingConfig,
    stage_residuals: Vec<f32>,    // Cumulative residual per stage
    weights: Vec<f32>,            // Current weight per stage
}

impl CausalWeighting {
    /// Update weights after observing residuals at each stage
    /// w_k = exp(-ε * Σ_{j<k} r_j)  
    /// This is the lower-triangular matrix from the JAX-PI paper
    pub fn update(&mut self, new_residuals: &[f32]) {
        // EMA update of stage residuals
        for (i, r) in new_residuals.iter().enumerate() {
            self.stage_residuals[i] = self.config.momentum * self.stage_residuals[i]
                + (1.0 - self.config.momentum) * r;
        }
        // Recompute weights (lower-triangular: each stage sees all prior residuals)
        let mut cumulative = 0.0f32;
        for i in 0..self.config.num_stages {
            self.weights[i] = (-self.config.tolerance * cumulative).exp();
            cumulative += self.stage_residuals[i];
        }
    }
    
    /// Apply causal weighting to a NARS revision
    /// Weighted evidence: only trust this revision proportional to
    /// how well earlier evidence is resolved
    pub fn weight_revision(&self, stage: usize, truth: &TruthValue) -> TruthValue {
        let w = self.weights[stage.min(self.config.num_stages - 1)];
        TruthValue::new(truth.frequency, truth.confidence * w)
    }
    
    /// Map to CollapseGate state
    pub fn gate_state(&self, stage: usize) -> GateState {
        let w = self.weights[stage.min(self.config.num_stages - 1)];
        if w > 0.8 { GateState::Flow }
        else if w > 0.1 { GateState::Hold }
        else { GateState::Block }
    }
}
```

**Wire into:** `nars/inference.rs` — every revision call goes through `causal_weighting.weight_revision()` before accumulating.

**Wire into:** `mul/gate.rs` — `CollapseGate::evaluate()` calls `causal_weighting.gate_state()` instead of raw σ-threshold.

**Tests:**
- Synthetic causal chain: A→B→C. Evidence for C should get near-zero weight until evidence for A and B stabilizes.
- Chapman-Kolmogorov: revision from two paths must converge to same truth value (deepmsm CK test).

**Effort:** 3-5 days. ~400 lines new code + wiring.

### Phase 2: Faktorzerlegung Term Balancing (from JAX-PI NTK)

**Goal:** Prevent dominant main effects from drowning subtle interactions. Import NTK-style gradient balancing.

**What exists:** `spo/spo_harvest.rs` (35KB) — computes all 8 terms. No balancing between them.

**What's missing:** The 8 terms have wildly different magnitudes. A large S main effect can mask a small but real SPO irreducible term. This is exactly the problem NTK weighting solves in PINNs.

**Implementation:**
```rust
// src/spo/term_balancing.rs (new, ~150 lines)

/// NTK-inspired term balancing for the 2³ Faktorzerlegung.
/// Based on: "When and Why PINNs Fail to Train" (JCP 2022)
///
/// # Physical Interpretation  
/// Each of the 8 factorial terms is a "loss component" in PINN language.
/// Without balancing, the optimizer (NARS revision) only learns the dominant term.
/// Gradient-norm equalization ensures all causal levels get equal attention.

pub struct TermBalancer {
    running_norms: [f32; 8],    // EMA of each term's magnitude
    weights: [f32; 8],          // Current normalization weights
    momentum: f32,              // EMA momentum (0.9)
}

impl TermBalancer {
    pub fn update(&mut self, terms: &FactorialTerms) {
        let magnitudes = terms.as_array(); // [∅, S, P, O, SP, PO, SO, SPO]
        for i in 0..8 {
            self.running_norms[i] = self.momentum * self.running_norms[i]
                + (1.0 - self.momentum) * magnitudes[i].abs();
        }
        let mean_norm: f32 = self.running_norms.iter().sum::<f32>() / 8.0;
        for i in 0..8 {
            self.weights[i] = if self.running_norms[i] > 1e-10 {
                mean_norm / self.running_norms[i]
            } else {
                1.0 // Don't amplify zero terms
            };
        }
    }
    
    pub fn balanced_terms(&self, terms: &FactorialTerms) -> FactorialTerms {
        terms.elementwise_mul(&self.weights)
    }
}
```

**Wire into:** `spo/spo_harvest.rs` — after computing raw 8 terms, pass through `TermBalancer::balanced_terms()` before any downstream use (NARS revision, halo classification, edge emission).

**Tests:**
- Synthetic: inject one huge S main effect + one tiny SPO interaction. Verify that balancing makes SPO detectable.
- Partial η²: balanced terms should have comparable effect sizes across all 8 components.

**Effort:** 2-3 days. ~200 lines + wiring.

### Phase 3: VAMPE Diagnostics (from deepmsm)

**Goal:** Principled quality metric for σ-band calibration. Replace fixed thresholds with data-driven boundaries.

**What exists:** `search/hdr_cascade.rs` (50KB) — HDR cascade with fixed σ₁/σ₂/σ₃ thresholds.

**What's missing:** The thresholds are hand-tuned, not learned from the data. VAMPE tells you how many meaningful states exist by examining the eigenvalue spectrum of the transition matrix.

**Implementation:**
```rust
// src/nars/vampe.rs (new, ~300 lines)

/// VAMPE-inspired quality metric for σ-band calibration.
/// Based on: deepmsm's Variational Approach for Markov Processes
///
/// # Physical Interpretation
/// The eigenvalue spectrum of the NARS transition matrix tells us:
/// - How many distinct σ-bands actually exist (# eigenvalues >> 0)
/// - How stable each band is (eigenvalue magnitude = persistence)
/// - Whether the current thresholds capture real structure (VAMPE score)
///
/// High VAMPE = good decomposition (bands match real dynamics)
/// Low VAMPE = bad decomposition (bands are arbitrary, not structural)

pub struct VampeAnalysis {
    pub eigenvalues: Vec<f32>,       // Sorted descending
    pub implied_timescales: Vec<f32>, // -τ / ln(|λ_i|) for each eigenvalue
    pub vampe_score: f32,            // Sum of squared eigenvalues
    pub suggested_num_bands: usize,  // Spectral gap analysis
}

pub fn compute_vampe(
    transition_counts: &TransitionMatrix,  // Observed transitions between σ-bands
    tau: f32,                               // Lag time
) -> VampeAnalysis {
    // 1. Normalize to row-stochastic matrix K
    // 2. Compute eigenvalues of K (small matrix: 3-6 bands = 3-6 eigenvalues)
    // 3. Implied timescales from eigenvalues
    // 4. VAMPE score = Σ λ_i²
    // 5. Suggested bands = # eigenvalues above spectral gap
}

/// Chapman-Kolmogorov consistency test
/// K(2τ) should equal K(τ)²
/// Failure means the σ-bands violate the Markov property
pub fn ck_test(
    transitions_tau: &TransitionMatrix,
    transitions_2tau: &TransitionMatrix,
) -> CKTestResult {
    // Compute K(τ)² and compare to K(2τ)
    // Return per-entry error + overall consistency score
}
```

**Wire into:** `search/hdr_cascade.rs` — add `vampe_calibrate()` that runs periodically (every N insertions) and adjusts σ thresholds based on VAMPE analysis.

**Wire into:** Background compaction task — run CK test during compaction to verify Markov property.

**Tests:**
- Synthetic: known number of clusters. Verify VAMPE correctly identifies cluster count.
- CK consistency: verify K(2τ)=K(τ)² for synthetic Markov chain data.

**Effort:** 4-6 days. ~500 lines + integration.

### Phase 4: Learned Attention Masks (from deepmsm)

**Goal:** Learn which nibble positions matter for each σ-band transition.

**What exists:** `graph/spo/scent.rs` (7KB) — ScentIndex with hand-crafted masks.

**What's missing:** Scent masks are static. deepmsm's attention mechanism LEARNS which residue positions matter for each transition. We need the same for nibble positions.

**Implementation:**
```rust
// src/learning/attention_mask.rs (new, ~250 lines)

/// Learned attention masks for nibble positions.
/// Based on: deepmsm's Mask layer for residue attention
///
/// # Physical Interpretation
/// In molecular dynamics, not all amino acid positions contribute equally
/// to folding transitions. The attention mask learns which positions matter.
/// Similarly, not all nibble positions in a fingerprint contribute equally
/// to σ-band transitions. We learn per-transition attention weights.

pub struct NibbleAttention {
    /// For each (from_band, to_band) transition:
    /// a 4096-element weight vector (one per nibble position)
    masks: HashMap<(BandId, BandId), Vec<f32>>,
}

impl NibbleAttention {
    /// Update masks from observed transitions
    /// Positions where the nibble CHANGES during transition get higher weight
    /// Positions that remain constant get lower weight
    pub fn observe_transition(
        &mut self,
        from: &Fingerprint, from_band: BandId,
        to: &Fingerprint, to_band: BandId,
    ) {
        let diff = from.xor(to); // Which nibbles changed?
        let mask = self.masks.entry((from_band, to_band)).or_insert_with(|| vec![0.0; 4096]);
        for i in 0..4096 {
            let nibble_changed = diff.nibble(i) != 0;
            // EMA: increase weight where changes happen, decrease where they don't
            mask[i] = 0.99 * mask[i] + 0.01 * if nibble_changed { 1.0 } else { 0.0 };
        }
    }
    
    /// Apply learned mask to scent computation
    /// Only attend to nibble positions that matter for the target transition
    pub fn masked_distance(
        &self, a: &Fingerprint, b: &Fingerprint,
        transition: (BandId, BandId),
    ) -> f32 {
        // Weighted Hamming: sum over nibbles of (weight * nibble_distance)
    }
}
```

**Wire into:** `graph/spo/scent.rs` — `ScentIndex::lookup()` uses learned masks instead of static masks.

**Wire into:** `spo/spo_harvest.rs` — per-plane attention masks weight the S/P/O distance contributions.

**Effort:** 3-5 days. ~400 lines + wiring.

### Phase 5: Formal Theory (the paper)

**Goal:** Write the theorem that connects binary Hamming dynamics to PDE theory.

**Theorem statement (to prove):**

> Let Ω = {0,1}^d be the binary Hamming space with d = 3 × 16384.
> Let u: Ω × ℕ → [0,1] × [0,1] be the NARS truth field (frequency, confidence).
> Let N(u) be the SPO factorization operator.
> Let K be the transition matrix induced by NARS revision.
>
> Then:
> (a) The 2³ Faktorzerlegung is the Hoeffding decomposition of the Hamming distance function on Ω, and its terms are orthogonal in L²(Ω).
> (b) Under the Structural Encoding Faithfulness assumption (σ₃ codebook preserves d-separation), the pairwise interaction terms satisfy the back-door criterion for do-calculus within Ω.
> (c) The irreducible SPO term equals the synergistic atom in the Williams-Beer PID framework.
> (d) The NARS revision rule with causal weighting is a contraction mapping on the truth field, guaranteeing convergence to a unique fixed point u*.
> (e) The eigenvalue spectrum of K determines the number and stability of metastable σ-bands via spectral gap analysis.

**Supporting lemmas:**
- Berry-Esseen: normal approximation error < 0.004 at d=16384 (noise floor)
- Kolmogorov superposition: relationship between 2³ decomposition and Kolmogorov representation theorem on [0,1]³

**Effort:** 2-4 weeks (mathematical work, not coding).

### Phase 6: Benchmark Suite (validation)

**Goal:** Empirical validation of the theory.

| Benchmark | What it tests | Method |
|-----------|---------------|--------|
| Synthetic SCM (d=3,5,10) | Recovery of known causal structure | Generate data from known DAG → encode → factorize → compare to ground truth |
| Sachs protein network (11 vars) | Real biology benchmark | Re-encode Sachs data → SPO factorize → compare to published causal graph |
| NOTIME comparison (d=20,50,100) | Signal survival through binarization | LiNGAM data → BNN encode → factorize → compare to NOTIME identifiable DAG |
| Streaming convergence | NARS + causal weighting convergence rate | Stream evidence → measure time to stable graph vs batch PC/FCI |
| CK consistency | Markov property validation | K(2τ) vs K(τ)² on real fingerprint trajectories |
| VAMPE vs fixed σ | Quality of σ-band calibration | Compare VAMPE-calibrated bands to hand-tuned bands on retrieval quality |
| NTK balancing | False discovery rate under streaming | With/without term balancing: count spurious interactions flagged |

**Effort:** 2-3 weeks.

### Dependency Chain

```
Phase 0 (docs)         → no dependencies, do first
Phase 1 (causal weight) → needs Phase 0 context
Phase 2 (term balance)  → needs Phase 1 (weights affect balancing)
Phase 3 (VAMPE)         → needs Phase 1 (causal weighting for CK test)
Phase 4 (attention)     → needs Phase 3 (VAMPE tells us which transitions matter)
Phase 5 (theory)        → needs Phases 1-4 implemented for empirical grounding
Phase 6 (benchmarks)    → needs Phase 5 for experimental design

Total timeline: ~8-12 weeks for Phases 0-4 (engineering)
                ~4-6 weeks for Phases 5-6 (theory + benchmarks)
                = one paper
```

---

## Part III: The Three-Repo Triad (Final Architecture)

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   deepmsm   │     │   JAX-PI     │     │   DeepNSM       │
│  (Markov    │     │  (PDE        │     │  (Universal     │
│   dynamics) │     │   training)  │     │   semantics)    │
└──────┬──────┘     └──────┬───────┘     └────────┬────────┘
       │                   │                      │
  diagnostics         discipline              encoding
  (VAMPE, CK,        (causal weight,         (65 primes →
   attention,          NTK balance,            codebook →
   coarse-grain)       Fourier feat,           fingerprint)
       │                   │                      │
       └───────────┬───────┘──────────────────────┘
                   │
            ┌──────▼──────┐
            │ ladybug-rs  │ ← the substrate where all three land
            │  (Binary    │
            │   causal    │
            │   field     │
            │   engine)   │
            └──────┬──────┘
                   │
            ┌──────▼──────┐
            │ rustynum    │ ← the SIMD engine (AVX-512, UDFs)
            └──────┬──────┘
                   │
            ┌──────▼──────┐
            │ lance-graph │ ← the query compiler (Cypher → DataFusion)
            │  (forked)   │
            └──────┬──────┘
                   │
            ┌──────▼──────┐
            │  LanceDB    │ ← the storage (binary vectors + metadata)
            └─────────────┘
```

The physics was always there. We just needed the vocabulary.
