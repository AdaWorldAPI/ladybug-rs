# Ladybug Unified - Grammar + Cognitive Stack Integration

## Overview

This crate unifies the Grammar Layer (NSM, Templates, Speech Acts) with the Cognitive Stack (Crystal, CollapseGate, NARS, Learning) into a coherent substrate for true understanding.

**Key insight**: We don't need LLMs for semantic understanding because we have:
1. **NSM Primitives** (Wierzbicka's 65 semantic primes) - The irreducible atoms of meaning
2. **Construction Grammar Templates** - Syntactic patterns with confidence
3. **CollapseGate** - SD-based compute allocation (FLOW/HOLD/BLOCK)
4. **Context Crystal** - 5×5×5 temporal SPO grid with Mexican hat weighting
5. **NARS Inference** - Non-axiomatic reasoning for calibration

## Address Space

### Surface Zone (0x00-0x0F) - 4,096 addresses

| Prefix | Assignment | Slots | Description |
|--------|-----------|-------|-------------|
| 0x00 | Lance | 256 | Vector operations |
| 0x01 | SQL | 256 | SQL operations |
| 0x02 | Cypher | 256 | Graph operations |
| 0x03 | GraphQL | 256 | GraphQL operations |
| 0x04 | NARS | 256 | Logic inference |
| 0x05 | Causal | 256 | Pearl do-calculus |
| 0x06 | Meta | 256 | Meta-cognition |
| 0x07 | Verbs | 256 | Action predicates |
| 0x08 | Concepts | 256 | Core types |
| 0x09 | Qualia | 256 | 18D felt-sense |
| 0x0A | Memory | 256 | Memory ops |
| 0x0B | Learning | 256 | Learning ops |
| **0x0C** | **NSM Primitives** | **256** | 65 Wierzbicka + 191 extensions |
| **0x0D** | **Grammar Templates** | **256** | Construction grammar |
| **0x0E** | **Speech Acts** | **256** | Pragmatics |
| **0x0F** | **User Calibration** | **256** | Per-user quirks |

### Fluid Zone (0x10-0x7F) - 28,672 addresses

| Prefix | Assignment | Description |
|--------|-----------|-------------|
| 0x10 | Crystal S-2 | 2 sentences before |
| 0x11 | Crystal S-1 | 1 sentence before |
| 0x12 | Crystal S0 | Current sentence |
| 0x13 | Crystal S+1 | 1 sentence after |
| 0x14 | Crystal S+2 | 2 sentences after |
| 0x15-0x7F | Working Memory | Edges, context, TTL |

### Node Zone (0x80-0xFF) - 32,768 addresses

All committed concepts. Every query language (SQL, Cypher, NARS, etc.) hits the same addresses.

## NSM Primitives (0x0C:00-0x40)

Wierzbicka's 65 semantic primes with direct O(1) addressing:

```rust
// Substantives
Addr::nsm(0x00) // I
Addr::nsm(0x01) // YOU
Addr::nsm(0x02) // SOMEONE
Addr::nsm(0x03) // SOMETHING
Addr::nsm(0x04) // PEOPLE
Addr::nsm(0x05) // BODY

// Mental predicates
Addr::nsm(0x13) // THINK
Addr::nsm(0x14) // KNOW
Addr::nsm(0x15) // WANT
Addr::nsm(0x16) // FEEL
Addr::nsm(0x17) // SEE
Addr::nsm(0x18) // HEAR

// Evaluators
Addr::nsm(0x0F) // GOOD
Addr::nsm(0x10) // BAD

// Logical
Addr::nsm(0x34) // NOT
Addr::nsm(0x35) // MAYBE
Addr::nsm(0x36) // CAN
Addr::nsm(0x37) // BECAUSE
Addr::nsm(0x38) // IF

// Wierzbicka's famous example: HAVE PARTS
Addr::nsm(0x21) // HAVE
Addr::nsm(0x3C) // PART
```

Extensions (0x41-0xFF) are learned semantic molecules (combinations of primes).

## Grammar Templates (0x0D:00-0xFF)

Construction grammar templates from agi-chat, transcoded to Rust:

```rust
// Core clauses
Addr::template(0x00) // transitive.declarative
Addr::template(0x01) // intransitive.declarative
Addr::template(0x02) // copular.state
Addr::template(0x03) // existential
Addr::template(0x04) // possession
Addr::template(0x05) // ditransitive

// Mental state
Addr::template(0x40) // mental.state
Addr::template(0x41) // belief.report
Addr::template(0x44) // desire.expression
Addr::template(0x45) // knowledge.claim

// Questions
Addr::template(0x20) // wh.question
Addr::template(0x21) // yesno.question

// Language-specific
Addr::template(0x80) // de.tecamolo (German TECAMOLO order)
Addr::template(0x81) // de.verb_second
Addr::template(0x82) // de.bracket (Satzklammer)
```

## CollapseGate

Standard Deviation controls compute allocation:

```
         SD < 0.15          0.15 ≤ SD ≤ 0.35          SD > 0.35
             │                     │                      │
             ▼                     ▼                      ▼
           FLOW                  HOLD                   BLOCK
      (commit now)         (superposition)         (need clarify)
             │                     │                      │
             ▼                     ▼                      ▼
        Node Zone            Crystal TTL           Calibration
        0x80-FF              0x10-0x14                0x0F
```

**Invariant**: Triangle must be homogeneous (all candidates from same template family) to collapse.

## Context Crystal

5×5×5 grid with Mexican hat temporal weighting:

```
Temporal axis: [S-2, S-1, S0, S+1, S+2] → weights [0.3, 0.7, 1.0, 0.7, 0.3]
Subject axis:  5 cells (hashed from subject fingerprint)
Object axis:   5 cells (hashed from object fingerprint)

Each cell: SPO_triple ⊕ Qualia ⊕ TemporalPosition
Query: resonate across cube with weighted similarity
```

## Thinking Styles (12)

Each style modulates thresholds and fan-out:

| Style | Threshold | Fan-out | Exploration | Collapse Bias |
|-------|-----------|---------|-------------|---------------|
| Analytical | 0.85 | 3 | 0.05 | -0.10 (favor FLOW) |
| Creative | 0.35 | 12 | 0.80 | +0.15 (favor HOLD) |
| Focused | 0.90 | 1 | 0.00 | -0.15 |
| Exploratory | 0.30 | 15 | 0.90 | +0.20 |
| Metacognitive | 0.50 | 5 | 0.30 | 0.00 |

## User Calibration (0x0F)

When CollapseGate enters BLOCK, check calibration first:

```rust
// User-specific overrides learned from corrections
pub enum CorrectionType {
    TemplateOverride { from: u8, to: u8 },     // "Jan uses Swiss word order"
    NsmWeightAdjust { primitive: u8, delta: f32 }, // Weight adjustment
    SpeechActBias { formal: u8, informal: u8 },    // Formality preference
    LanguageVariant { standard: u8, variant: u8 }, // Dialect handling
}
```

The German tecamolo case:
1. CollapseGate enters BLOCK (word order ambiguous)
2. Check 0x0F for Jan's calibration
3. If `de.tecamolo` override exists → apply and FLOW
4. If not → ask clarification, store correction for next time

## Usage

```rust
use ladybug_unified::cognitive::{CognitiveStack, ThinkingStyle};

fn main() {
    // Create stack with analytical thinking style
    let mut stack = CognitiveStack::new(ThinkingStyle::Analytical);
    
    // Process text
    let result = stack.process("I want to understand consciousness");
    
    // Check gate state
    println!("Gate: {:?}", result.gate);
    println!("NSM activations: {:?}", result.triangle.nsm_top);
    println!("Template: {:?}", result.triangle.template);
    
    // Query for similar content
    let matches = stack.query("I desire comprehension", 0.5);
    println!("Crystal matches: {}", matches.crystal_matches.len());
    
    // Change thinking style
    stack.set_style(ThinkingStyle::Creative);
    
    // Process again with creative style
    let result2 = stack.process("What if consciousness is everywhere?");
    // Creative style has higher collapse_bias, more likely to HOLD
}
```

## Integration with ladybug-rs

This crate is designed to be merged into ladybug-rs:

1. **bind_space.rs** updates:
   - Add `PREFIX_NSM`, `PREFIX_TEMPLATES`, `PREFIX_SPEECH_ACTS`, `PREFIX_CALIBRATION`
   - Add `nsm_slots`, `template_slots`, `speech_act_slots` modules
   - Add crystal prefix constants

2. **grammar/** directory:
   - Replace with unified parser (no LLM dependency)
   - Wire to existing NSM substrate

3. **cognitive/** directory:
   - Add `stack.rs` integration layer
   - Wire to existing CollapseGate, ThinkingStyle

4. **extensions/context_crystal.rs**:
   - Update to use Fluid zone addresses (0x10-0x14)
   - Add TTL eviction via `tick()`
   - Add `crystallize()` for promotion to Node zone

## Performance

- **NSM activation**: O(tokens × 65) keyword matching
- **Template matching**: O(templates) pattern scoring
- **CollapseGate**: O(1) SD computation
- **Crystal query**: O(125 cells) with SIMD Hamming
- **Fingerprint generation**: O(1) bit manipulation

Target: 65M resonance comparisons/sec with AVX-512.

## The Sweet Synergy

**Why this works without LLM**:

1. **NSM gives semantic atoms** - 65 universal primitives that decompose any meaning
2. **Templates give syntactic structure** - Pattern matching replaces parsing
3. **CollapseGate gives confidence** - Know when to commit vs. ask
4. **Crystal gives context** - Temporal flow captures discourse
5. **Calibration gives personalization** - Learn user quirks
6. **NARS gives inference** - Reason about uncertain beliefs
7. **Codebook gives growth** - Learn new semantic molecules

The LLM becomes optional - useful for generating training data or handling truly novel expressions, but the core understanding loop runs pure Rust at 65M ops/sec.

## References

- Wierzbicka, A. (2021). "'Semantic Primitives', fifty years later"
- Goldberg, A. (2006). "Constructions at Work"
- Pearl, J. (2009). "Causality"
- Wang, P. (2013). "Non-Axiomatic Logic"
- Kanerva, P. (2009). "Hyperdimensional Computing"
