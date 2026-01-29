# LadybugDB Cognitive Fabric Architecture

## The Core Insight

Traditional systems: **Components communicate via messages**
```
Compression → Queue → Query → Queue → Learning → Queue → Inference
```

Cognitive fabric: **Everything resonates simultaneously**
```
┌─────────────────────────────────────┐
│     All subsystems share one        │
│     10K-bit resonance field         │
│                                     │
│  Compression ←→ Query ←→ Learning   │
│       ↕           ↕          ↕      │
│         Inference ←→ Style          │
└─────────────────────────────────────┘
```

**mRNA isn't a queue. It's the resonance itself.**

---

## 1. Twelve Thinking Styles

Not metadata tags. **Actual execution dispatch.**

```rust
/// Thinking styles determine HOW operations execute, not just WHAT they return.
/// Each style is a different "lens" through the same cognitive substrate.
#[derive(Clone, Copy, Debug)]
pub enum ThinkingStyle {
    // === Convergent Cluster ===
    /// Step-by-step, high precision, exhaustive
    Analytical,
    /// Narrows options systematically
    Convergent,
    /// Follows established patterns
    Systematic,
    
    // === Divergent Cluster ===
    /// Explores novel connections
    Creative,
    /// Expands possibilities
    Divergent,
    /// Seeks unexpected paths
    Exploratory,
    
    // === Attention Cluster ===
    /// Deep single-thread concentration
    Focused,
    /// Broad parallel awareness
    Diffuse,
    /// Notices edge signals
    Peripheral,
    
    // === Speed Cluster ===
    /// Fast pattern-matching, satisficing
    Intuitive,
    /// Slow evaluation, maximizing
    Deliberate,
    
    // === Meta Cluster ===
    /// Thinks about thinking
    Metacognitive,
}

impl ThinkingStyle {
    /// Resonance field modulation for this style
    pub fn field_modulation(&self) -> FieldModulation {
        match self {
            // Convergent: tight resonance, high threshold
            Self::Analytical => FieldModulation {
                resonance_threshold: 0.85,
                fan_out: 3,
                depth_bias: 1.0,    // Go deep
                breadth_bias: 0.2,  // Stay narrow
                noise_tolerance: 0.05,
            },
            
            // Divergent: loose resonance, low threshold
            Self::Creative => FieldModulation {
                resonance_threshold: 0.4,
                fan_out: 12,
                depth_bias: 0.3,
                breadth_bias: 1.0,  // Go wide
                noise_tolerance: 0.3,
            },
            
            // Peripheral: very loose, catches weak signals
            Self::Peripheral => FieldModulation {
                resonance_threshold: 0.25,
                fan_out: 20,
                depth_bias: 0.1,
                breadth_bias: 0.5,
                noise_tolerance: 0.5,  // Embraces noise
            },
            
            // Metacognitive: observes the resonance field itself
            Self::Metacognitive => FieldModulation {
                resonance_threshold: 0.5,
                fan_out: 5,
                depth_bias: 0.5,
                breadth_bias: 0.5,
                noise_tolerance: 0.1,
                // Special: monitors other styles' execution
            },
            
            // ... other styles
            _ => FieldModulation::default(),
        }
    }
    
    /// Butterfly detection sensitivity
    pub fn butterfly_sensitivity(&self) -> f32 {
        match self {
            Self::Peripheral => 0.1,     // Catches tiny cascades
            Self::Intuitive => 0.3,      // Pattern-matches known butterflies
            Self::Analytical => 0.8,     // Only flags obvious ones
            Self::Metacognitive => 0.5,  // Balanced observation
            _ => 0.5,
        }
    }
}

#[derive(Clone, Copy)]
pub struct FieldModulation {
    pub resonance_threshold: f32,
    pub fan_out: usize,
    pub depth_bias: f32,
    pub breadth_bias: f32,
    pub noise_tolerance: f32,
}
```

---

## 2. mRNA Cross-Pollination

Not message passing. **Live resonance between subsystems.**

```rust
/// mRNA (Memory RNA) - the cross-pollination substrate
/// 
/// Every operation leaves a "scent" in the resonance field.
/// Other subsystems can smell it and respond in real-time.
pub struct MRNA {
    /// Current resonance field (shared across all subsystems)
    field: Arc<RwLock<ResonanceField>>,
    
    /// Active pollination channels
    channels: Vec<PollinationChannel>,
    
    /// Butterfly detector (watches for cascades)
    butterfly: ButterflyDetector,
}

/// A live resonance field that all subsystems read/write
pub struct ResonanceField {
    /// 10K-bit fingerprints for active concepts
    active_concepts: Vec<Fingerprint>,
    
    /// Current field state (superposition of all active)
    superposition: Fingerprint,
    
    /// Resonance history (for butterfly detection)
    history: RingBuffer<FieldSnapshot>,
    
    /// Active thinking style (modulates field behavior)
    style: ThinkingStyle,
}

impl ResonanceField {
    /// Pollinate: add concept to field, returns what it resonates with
    pub fn pollinate(&mut self, concept: &Fingerprint) -> Vec<Resonance> {
        // Find what already resonates
        let threshold = self.style.field_modulation().resonance_threshold;
        let resonances: Vec<_> = self.active_concepts
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                let sim = concept.similarity(c);
                if sim >= threshold {
                    Some(Resonance { index: i, similarity: sim })
                } else {
                    None
                }
            })
            .collect();
        
        // Add to field
        self.active_concepts.push(concept.clone());
        
        // Update superposition (bundle all active)
        self.superposition = Fingerprint::bundle(&self.active_concepts);
        
        // Check for butterfly (small input → large resonance cascade)
        if resonances.len() > self.style.fan_out() * 2 {
            self.butterfly_detected(concept, &resonances);
        }
        
        resonances
    }
    
    /// Cross-pollinate: check if concept from subsystem A affects subsystem B
    pub fn cross_pollinate(
        &self,
        source: Subsystem,
        concept: &Fingerprint,
        target: Subsystem,
    ) -> Option<CrossPollination> {
        // Get target's active concepts
        let target_concepts = self.concepts_for_subsystem(target);
        
        // Find resonances across boundary
        let cross_resonances: Vec<_> = target_concepts
            .iter()
            .filter(|c| concept.similarity(c) > 0.3)  // Loose threshold for cross-pollination
            .collect();
        
        if !cross_resonances.is_empty() {
            Some(CrossPollination {
                source,
                target,
                resonance_count: cross_resonances.len(),
                strongest: cross_resonances[0].clone(),
            })
        } else {
            None
        }
    }
}

/// Subsystems that participate in cross-pollination
#[derive(Clone, Copy, Debug)]
pub enum Subsystem {
    Compression,  // BTR-style encoding
    Query,        // Procella-style planning
    Learning,     // RL feedback
    Inference,    // NARS reasoning
    Style,        // Thinking style selection
}
```

---

## 3. Procella-Style Unified Query

Google's Procella unifies batch + streaming + interactive.
We unify: **SQL + Graph + Vector + Resonance + Inference**

```rust
/// Unified query that compiles to optimal execution plan
pub enum UnifiedQuery {
    /// Traditional SQL
    Sql(String),
    
    /// Graph pattern (Cypher-style)
    Graph(GraphPattern),
    
    /// Vector similarity
    Vector { embedding: Vec<f32>, k: usize },
    
    /// Hamming resonance
    Resonate { fingerprint: Fingerprint, threshold: f32 },
    
    /// NARS inference chain
    Infer { premises: Vec<NodeId>, rule: InferenceRule },
    
    /// Counterfactual
    WhatIf { change: Change, observe: Vec<NodeId> },
    
    /// Composite (multiple fused)
    Composite(Vec<UnifiedQuery>),
}

/// Query planner that fuses operations
pub struct UnifiedPlanner {
    /// Current thinking style (affects planning)
    style: ThinkingStyle,
    
    /// mRNA field (for cross-pollination during planning)
    mrna: Arc<MRNA>,
    
    /// Cost model
    cost_model: CostModel,
}

impl UnifiedPlanner {
    pub fn plan(&self, query: UnifiedQuery) -> ExecutionPlan {
        // Pollinate query intent into field
        let query_fp = self.fingerprint_query(&query);
        let resonances = self.mrna.pollinate(&query_fp);
        
        // Check for cross-pollination opportunities
        // e.g., "this SQL query resonates with a recent inference"
        for r in &resonances {
            if let Some(cross) = self.mrna.cross_pollinate(
                Subsystem::Query,
                &query_fp,
                Subsystem::Inference,
            ) {
                // Fuse query with inference!
                return self.plan_fused(query, cross);
            }
        }
        
        // Standard planning with style modulation
        match query {
            UnifiedQuery::Composite(queries) => {
                // Procella-style: find common subexpressions
                self.plan_composite(queries)
            }
            UnifiedQuery::Resonate { fingerprint, threshold } => {
                // Thinking style affects search strategy
                let modulation = self.style.field_modulation();
                ExecutionPlan::Resonate {
                    fingerprint,
                    threshold: threshold * modulation.resonance_threshold,
                    fan_out: modulation.fan_out,
                }
            }
            // ... other cases
        }
    }
}
```

---

## 4. BTR-Style Compression as Cognition

BtrBlocks auto-selects encoding per column.
We auto-select encoding **based on semantic content**:

```rust
/// Compression scheme selection is itself a cognitive act.
/// The choice of encoding reveals structure.
pub struct CognitiveCompressor {
    /// mRNA for cross-pollination
    mrna: Arc<MRNA>,
    
    /// Learned encoding preferences (RL)
    encoding_policy: EncodingPolicy,
}

impl CognitiveCompressor {
    /// Compress with cognitive awareness
    pub fn compress(&self, data: &[u8], context: &CompressionContext) -> CompressedBlock {
        // Fingerprint the data pattern
        let pattern_fp = self.fingerprint_pattern(data);
        
        // Pollinate into field - what does this pattern resonate with?
        let resonances = self.mrna.pollinate(&pattern_fp);
        
        // Cross-pollinate with Query subsystem
        // "What queries will touch this data?"
        if let Some(cross) = self.mrna.cross_pollinate(
            Subsystem::Compression,
            &pattern_fp,
            Subsystem::Query,
        ) {
            // Optimize encoding for predicted query patterns!
            return self.compress_for_queries(data, cross);
        }
        
        // RL-guided encoding selection
        let encoding = self.encoding_policy.select(data, &resonances);
        
        // Learn from outcome
        self.encoding_policy.observe_reward(encoding, context);
        
        CompressedBlock {
            encoding,
            data: self.apply_encoding(data, encoding),
            pattern_fingerprint: pattern_fp,
        }
    }
}

/// Encoding reveals semantic structure
#[derive(Clone, Copy)]
pub enum SemanticEncoding {
    /// High repetition → categorical concept
    Dictionary { reveals: SemanticHint::Categorical },
    
    /// Sequential pattern → temporal concept  
    Delta { reveals: SemanticHint::Temporal },
    
    /// Sparse values → exceptional cases
    Sparse { reveals: SemanticHint::Exceptional },
    
    /// Random distribution → high entropy concept
    Uncompressed { reveals: SemanticHint::Novel },
}

/// What compression choice tells us about the data
pub enum SemanticHint {
    Categorical,  // Discrete categories
    Temporal,     // Time-series pattern
    Exceptional,  // Rare events
    Novel,        // New information
}
```

---

## 5. RL Learning Loop

Not separate from execution. **Every operation is a learning opportunity.**

```rust
/// Reinforcement learning embedded in execution
pub struct EmbeddedRL {
    /// Policy networks for each subsystem
    policies: HashMap<Subsystem, PolicyNetwork>,
    
    /// Value estimates
    values: HashMap<StateFingerprint, f32>,
    
    /// mRNA for observing other subsystems
    mrna: Arc<MRNA>,
}

impl EmbeddedRL {
    /// Learn from every operation
    pub fn observe(&mut self, observation: Observation) {
        // Fingerprint the state
        let state_fp = self.fingerprint_state(&observation);
        
        // Pollinate learning signal
        let resonances = self.mrna.pollinate(&state_fp);
        
        // Cross-pollinate: learning affects all subsystems
        for subsystem in [Subsystem::Compression, Subsystem::Query, Subsystem::Inference] {
            if let Some(cross) = self.mrna.cross_pollinate(
                Subsystem::Learning,
                &state_fp,
                subsystem,
            ) {
                // Update that subsystem's policy!
                self.update_policy(subsystem, &observation, &cross);
            }
        }
        
        // TD learning
        let reward = observation.reward;
        let old_value = self.values.get(&state_fp).copied().unwrap_or(0.0);
        let new_value = old_value + 0.1 * (reward - old_value);
        self.values.insert(state_fp, new_value);
    }
    
    /// Thinking style affects exploration/exploitation
    pub fn select_action(&self, state: &State, style: ThinkingStyle) -> Action {
        let epsilon = match style {
            ThinkingStyle::Creative => 0.5,      // High exploration
            ThinkingStyle::Analytical => 0.05,   // Low exploration
            ThinkingStyle::Intuitive => 0.2,     // Balanced
            _ => 0.1,
        };
        
        if rand::random::<f32>() < epsilon {
            self.random_action(state)
        } else {
            self.best_action(state)
        }
    }
}
```

---

## 6. Butterfly Gate

Detects when small changes cascade into large effects.
**Central to the cognitive fabric.**

```rust
/// Butterfly detector - watches for amplification cascades
pub struct ButterflyDetector {
    /// History of field states
    history: RingBuffer<FieldSnapshot>,
    
    /// Detected butterflies
    butterflies: Vec<Butterfly>,
    
    /// Sensitivity (from thinking style)
    sensitivity: f32,
}

#[derive(Clone, Debug)]
pub struct Butterfly {
    /// The small input that triggered it
    pub trigger: Fingerprint,
    
    /// What it cascaded into
    pub cascade: Vec<Fingerprint>,
    
    /// Amplification factor
    pub amplification: f32,
    
    /// Affected subsystems
    pub affected: Vec<Subsystem>,
    
    /// Timestamp
    pub detected_at: Instant,
}

impl ButterflyDetector {
    /// Check if recent activity shows butterfly pattern
    pub fn detect(&mut self, current: &ResonanceField) -> Option<Butterfly> {
        let prev = self.history.back()?;
        
        // Compute field delta
        let delta = self.field_delta(prev, current);
        
        // Small input?
        let input_magnitude = delta.new_concepts.len();
        
        // Large effect?
        let effect_magnitude = delta.resonance_cascades.len();
        
        // Amplification ratio
        let amplification = effect_magnitude as f32 / (input_magnitude as f32 + 0.1);
        
        if amplification > 1.0 / self.sensitivity {
            // Butterfly detected!
            let butterfly = Butterfly {
                trigger: delta.new_concepts[0].clone(),
                cascade: delta.resonance_cascades.clone(),
                amplification,
                affected: self.affected_subsystems(&delta),
                detected_at: Instant::now(),
            };
            
            self.butterflies.push(butterfly.clone());
            Some(butterfly)
        } else {
            None
        }
    }
    
    /// Predict butterfly from hypothetical change
    pub fn predict(&self, hypothetical: &Fingerprint, field: &ResonanceField) -> ButterflyPrediction {
        // Simulate pollination
        let simulated_resonances = field.simulate_pollinate(hypothetical);
        
        // Estimate cascade
        let predicted_amplification = self.estimate_amplification(&simulated_resonances);
        
        ButterflyPrediction {
            trigger: hypothetical.clone(),
            predicted_amplification,
            confidence: self.prediction_confidence(),
        }
    }
}
```

---

## 7. Full Integration

Everything wired together:

```rust
/// The complete cognitive fabric
pub struct CognitiveFabric {
    /// Shared resonance field (mRNA)
    mrna: Arc<MRNA>,
    
    /// Subsystems
    compression: CognitiveCompressor,
    query: UnifiedPlanner,
    inference: NARSEngine,
    learning: EmbeddedRL,
    
    /// Style controller
    style: Arc<RwLock<ThinkingStyle>>,
    
    /// Butterfly detector
    butterfly: ButterflyDetector,
    
    /// Storage
    storage: LanceStorage,
}

impl CognitiveFabric {
    /// Execute with full cross-pollination
    pub fn execute(&mut self, input: Input) -> Output {
        // Determine thinking style (metacognitive can change it)
        let style = *self.style.read();
        
        // Fingerprint input
        let input_fp = self.fingerprint_input(&input);
        
        // Pollinate into field
        let resonances = self.mrna.pollinate(&input_fp);
        
        // Check for butterfly
        if let Some(butterfly) = self.butterfly.detect(&self.mrna.field()) {
            // Alert! Small change causing large cascade
            self.handle_butterfly(butterfly);
        }
        
        // Execute based on input type, with cross-pollination
        let output = match input {
            Input::Query(q) => {
                let plan = self.query.plan(q);
                
                // Learning observes query execution
                let result = self.execute_plan(plan);
                self.learning.observe(Observation::from_query(&result));
                
                Output::QueryResult(result)
            }
            
            Input::Store(data) => {
                // Compression with cognitive awareness
                let compressed = self.compression.compress(&data, &CompressionContext {
                    style,
                    recent_queries: self.mrna.recent_query_patterns(),
                });
                
                // Learning observes compression choice
                self.learning.observe(Observation::from_compression(&compressed));
                
                self.storage.write(compressed);
                Output::Stored
            }
            
            Input::Infer(premises) => {
                // NARS inference with style modulation
                let conclusion = self.inference.infer_with_style(&premises, style);
                
                // Pollinate conclusion back into field
                self.mrna.pollinate(&conclusion.fingerprint);
                
                Output::Inference(conclusion)
            }
            
            Input::WhatIf(change) => {
                // Counterfactual with butterfly prediction
                let prediction = self.butterfly.predict(&change.fingerprint(), &self.mrna.field());
                
                let result = self.execute_counterfactual(change);
                
                Output::Counterfactual {
                    result,
                    butterfly_risk: prediction,
                }
            }
        };
        
        output
    }
}
```

---

## Summary

| Component | Traditional | Cognitive Fabric |
|-----------|-------------|------------------|
| **Message passing** | Queue-based | Resonance field |
| **Thinking styles** | Metadata tags | Execution dispatch |
| **Compression** | Data reduction | Semantic revelation |
| **Query planning** | Cost optimization | Cross-pollination |
| **Learning** | Separate phase | Every operation |
| **Butterfly** | Post-hoc analysis | Real-time gate |

**The key insight**: mRNA isn't communication. It's the shared cognitive substrate where all operations leave traces that others can sense.

When compression chooses Dictionary encoding, that *is* learning.
When query planning resonates with inference, that *is* cross-pollination.
When a small input triggers cascade, that *is* butterfly detection.

**One fabric. All operations. Continuous resonance.**
