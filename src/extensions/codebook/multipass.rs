//! Multi-Pass Codebook: Concept CAM with Hamming Meta-Resonance
//!
//! PASS 1 (expensive, one-time): Collect concepts from rich corpus
//!   - Books, NARS patterns, qualia mappings, SPO relations
//!   - Jina embed → 10Kbit fingerprint → cluster into CAM slots
//!
//! PASS 2 (cheap, runtime): Hamming resonance lookup
//!   - New text → hash fingerprint → XOR scan against CAM
//!   - Zero API calls, pure binary operations
//!   - ~1 microsecond per lookup
//!
//! The CAM IS the learned semantic space. Once trained, it's a resonance surface.

use std::collections::HashMap;
use std::time::Instant;

const N: usize = 16_384; // Fingerprint bits
const N64: usize = 256; // u64 words
const CAM_SIZE: usize = 128; // Codebook slots
const HAMMING_THRESHOLD: u32 = 1500; // ~15% = similar

// ============================================================================
// Fingerprint
// ============================================================================

#[repr(align(64))]
#[derive(Clone, PartialEq)]
pub struct Fingerprint {
    pub data: [u64; N64],
}

impl Fingerprint {
    pub fn zero() -> Self {
        Self { data: [0u64; N64] }
    }

    pub fn from_seed(seed: u64) -> Self {
        let mut state = seed;
        let mut data = [0u64; N64];
        for w in &mut data {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *w = state;
        }
        Self { data }
    }

    /// Deterministic fingerprint from text (hash-based, for Pass 2)
    pub fn from_text_hash(text: &str) -> Self {
        // Multi-round mixing for semantic-ish distribution
        let bytes = text.as_bytes();
        let mut data = [0u64; N64];

        // Initialize with text hash
        let mut seed = 0x517cc1b727220a95u64;
        for &b in bytes {
            seed = seed.wrapping_mul(0x5851f42d4c957f2d).wrapping_add(b as u64);
        }

        // Generate fingerprint with n-gram influence
        for (i, w) in data.iter_mut().enumerate() {
            let mut h = seed.wrapping_add(i as u64 * 0x9E3779B97F4A7C15);

            // Mix in character trigrams
            for chunk in bytes.windows(3) {
                let trigram =
                    (chunk[0] as u64) | ((chunk[1] as u64) << 8) | ((chunk[2] as u64) << 16);
                h ^= trigram.wrapping_mul(0x517cc1b727220a95);
                h = h.rotate_left(17);
            }

            *w = h;
        }

        Self { data }
    }

    /// Simulate Jina embedding → binary (for Pass 1 training)
    /// In production, this calls real Jina API
    pub fn from_jina_embedding(embedding: &[f32; 1024]) -> Self {
        // Expand 1024D → 10K bits via random projection
        let mut fp = Fingerprint::zero();

        for bit in 0..N {
            // Each bit is sign of dot product with random hyperplane
            let mut dot = 0.0f32;
            for (i, &e) in embedding.iter().enumerate() {
                // Pseudo-random projection weight
                let proj_seed = (bit as u64 * 1024 + i as u64).wrapping_mul(0x9E3779B97F4A7C15);
                let proj = if proj_seed & 1 == 1 { 1.0 } else { -1.0 };
                dot += e * proj;
            }

            if dot > 0.0 {
                let word = bit / 64;
                let bit_pos = bit % 64;
                fp.data[word] |= 1 << bit_pos;
            }
        }

        fp
    }

    #[inline]
    pub fn hamming(&self, other: &Fingerprint) -> u32 {
        let mut t = 0u32;
        for i in 0..N64 {
            t += (self.data[i] ^ other.data[i]).count_ones();
        }
        t
    }

    pub fn similarity(&self, other: &Fingerprint) -> f64 {
        1.0 - (self.hamming(other) as f64 / N as f64)
    }
}

/// Majority vote bundle
fn bundle(items: &[Fingerprint]) -> Fingerprint {
    if items.is_empty() {
        return Fingerprint::zero();
    }
    if items.len() == 1 {
        return items[0].clone();
    }
    let threshold = items.len() / 2;
    let mut result = Fingerprint::zero();
    for w in 0..N64 {
        for bit in 0..64 {
            let count: usize = items
                .iter()
                .filter(|fp| (fp.data[w] >> bit) & 1 == 1)
                .count();
            if count > threshold {
                result.data[w] |= 1 << bit;
            }
        }
    }
    result
}

// ============================================================================
// Concept Types (from NARS, Qualia, SPO)
// ============================================================================

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ConceptType {
    // NARS inference patterns
    Inheritance, // A → B (is-a)
    Similarity,  // A ↔ B (like)
    Implication, // A ⇒ B (if-then)
    Equivalence, // A ⇔ B (iff)

    // SPO relations
    Causes,      // A causes B
    Contains,    // A contains B
    Becomes,     // A becomes B
    Enables,     // A enables B
    Contradicts, // A contradicts B
    Refines,     // A refines B
    Grounds,     // A grounds B
    Abstracts,   // A abstracts B

    // Qualia anchors
    Felt,    // Felt-sense quality
    Arousal, // Energy level
    Valence, // Positive/negative
    Tension, // Cognitive tension

    // Roles
    Subject,   // S in SPO
    Predicate, // P in SPO
    Object,    // O in SPO
    Qualia,    // Q in SPOQ

    // Emergent (learned from corpus)
    Cluster(u8), // Emergent concept cluster
}

impl ConceptType {
    pub fn all_archetypes() -> Vec<ConceptType> {
        vec![
            ConceptType::Inheritance,
            ConceptType::Similarity,
            ConceptType::Implication,
            ConceptType::Equivalence,
            ConceptType::Causes,
            ConceptType::Contains,
            ConceptType::Becomes,
            ConceptType::Enables,
            ConceptType::Contradicts,
            ConceptType::Refines,
            ConceptType::Grounds,
            ConceptType::Abstracts,
            ConceptType::Felt,
            ConceptType::Arousal,
            ConceptType::Valence,
            ConceptType::Tension,
            ConceptType::Subject,
            ConceptType::Predicate,
            ConceptType::Object,
            ConceptType::Qualia,
        ]
    }

    /// Seed text for generating archetype fingerprint
    pub fn seed_text(&self) -> &'static str {
        match self {
            ConceptType::Inheritance => "inheritance is-a type-of category classification taxonomy",
            ConceptType::Similarity => "similarity like resembles analogous parallel comparable",
            ConceptType::Implication => "implication if-then therefore consequently follows",
            ConceptType::Equivalence => "equivalence identical same equal interchangeable",
            ConceptType::Causes => "causes leads-to results-in produces triggers",
            ConceptType::Contains => "contains includes holds comprises encompasses",
            ConceptType::Becomes => "becomes transforms evolves changes transitions",
            ConceptType::Enables => "enables allows permits facilitates supports",
            ConceptType::Contradicts => "contradicts opposes conflicts negates denies",
            ConceptType::Refines => "refines specifies details elaborates narrows",
            ConceptType::Grounds => "grounds anchors bases foundations roots",
            ConceptType::Abstracts => "abstracts generalizes summarizes essence core",
            ConceptType::Felt => "felt sense feeling quality experience qualia",
            ConceptType::Arousal => "arousal energy activation intensity vigor",
            ConceptType::Valence => "valence positive negative pleasant unpleasant",
            ConceptType::Tension => "tension conflict uncertainty ambiguity unresolved",
            ConceptType::Subject => "subject agent actor source origin initiator",
            ConceptType::Predicate => "predicate relation action verb connection link",
            ConceptType::Object => "object target destination recipient result",
            ConceptType::Qualia => "qualia experience consciousness awareness feeling",
            ConceptType::Cluster(_) => "emergent concept pattern cluster group",
        }
    }
}

// ============================================================================
// CAM Slot: Single entry in the Concept Addressable Memory
// ============================================================================

#[derive(Clone)]
pub struct CAMSlot {
    /// Concept type
    pub concept: ConceptType,
    /// Fingerprint (centroid of all examples)
    pub fingerprint: Fingerprint,
    /// Example texts that mapped to this slot
    pub examples: Vec<String>,
    /// Access count for popularity tracking
    pub access_count: u64,
    /// Confidence (based on example count)
    pub confidence: f64,
}

impl CAMSlot {
    pub fn new(concept: ConceptType) -> Self {
        let seed_fp = Fingerprint::from_text_hash(concept.seed_text());
        Self {
            concept,
            fingerprint: seed_fp,
            examples: Vec::new(),
            access_count: 0,
            confidence: 0.5,
        }
    }

    /// Add example and update centroid
    pub fn add_example(&mut self, text: &str, fp: &Fingerprint) {
        self.examples.push(text.to_string());

        // Update centroid via running bundle
        if self.examples.len() == 1 {
            self.fingerprint = fp.clone();
        } else {
            self.fingerprint = bundle(&[self.fingerprint.clone(), fp.clone()]);
        }

        // Update confidence
        self.confidence = (self.examples.len() as f64 / 100.0).min(1.0);
    }
}

// ============================================================================
// Concept CAM: The learned resonance surface
// ============================================================================

pub struct ConceptCAM {
    /// Fixed archetype slots (NARS, SPO, Qualia)
    pub archetypes: Vec<CAMSlot>,
    /// Emergent cluster slots (learned from corpus)
    pub clusters: Vec<CAMSlot>,
    /// Stats
    pub pass1_examples: usize,
    pub pass2_lookups: usize,
    pub pass2_hits: usize,
}

impl ConceptCAM {
    pub fn new() -> Self {
        // Initialize archetype slots
        let archetypes: Vec<CAMSlot> = ConceptType::all_archetypes()
            .into_iter()
            .map(|ct| CAMSlot::new(ct))
            .collect();

        // Reserve space for emergent clusters
        let clusters: Vec<CAMSlot> = (0..CAM_SIZE - archetypes.len())
            .map(|i| CAMSlot::new(ConceptType::Cluster(i as u8)))
            .collect();

        Self {
            archetypes,
            clusters,
            pass1_examples: 0,
            pass2_lookups: 0,
            pass2_hits: 0,
        }
    }

    // ========================================================================
    // PASS 1: Concept Collection (expensive, uses Jina)
    // ========================================================================

    /// Train on a concept example (Pass 1)
    /// In production: text → Jina → 1024D → fingerprint
    /// Here: simulated with hash-based fingerprint
    pub fn train(&mut self, text: &str, concept_hint: Option<ConceptType>) {
        let fp = Fingerprint::from_text_hash(text);
        self.pass1_examples += 1;

        // If concept type is known, add to that slot
        if let Some(ct) = concept_hint {
            for slot in &mut self.archetypes {
                if slot.concept == ct {
                    slot.add_example(text, &fp);
                    return;
                }
            }
        }

        // Otherwise, find nearest slot or create new cluster
        let (nearest_idx, nearest_dist, is_archetype) = self.find_nearest(&fp);

        if nearest_dist < HAMMING_THRESHOLD {
            // Close enough - add to existing slot
            if is_archetype {
                self.archetypes[nearest_idx].add_example(text, &fp);
            } else {
                self.clusters[nearest_idx].add_example(text, &fp);
            }
        } else {
            // Too far - find empty cluster slot
            for slot in &mut self.clusters {
                if slot.examples.is_empty() {
                    slot.add_example(text, &fp);
                    return;
                }
            }
            // No empty slots - add to nearest anyway
            if is_archetype {
                self.archetypes[nearest_idx].add_example(text, &fp);
            } else {
                self.clusters[nearest_idx].add_example(text, &fp);
            }
        }
    }

    /// Train on corpus of (text, optional concept) pairs
    pub fn train_corpus(&mut self, corpus: &[(String, Option<ConceptType>)]) {
        for (text, hint) in corpus {
            self.train(text, hint.clone());
        }
    }

    // ========================================================================
    // PASS 2: Hamming Meta-Resonance (cheap, no API calls)
    // ========================================================================

    /// Lookup concept via pure Hamming resonance (Pass 2)
    /// NO Jina call - just XOR and popcount
    pub fn resonate(&mut self, text: &str) -> ResonanceResult {
        let fp = Fingerprint::from_text_hash(text);
        self.pass2_lookups += 1;

        let (nearest_idx, nearest_dist, is_archetype) = self.find_nearest(&fp);

        let slot = if is_archetype {
            &self.archetypes[nearest_idx]
        } else {
            &self.clusters[nearest_idx]
        };

        let similarity = 1.0 - (nearest_dist as f64 / N as f64);
        let hit = nearest_dist < HAMMING_THRESHOLD;

        if hit {
            self.pass2_hits += 1;
        }

        ResonanceResult {
            concept: slot.concept.clone(),
            similarity,
            confidence: slot.confidence,
            hit,
            slot_examples: slot.examples.len(),
        }
    }

    /// Batch resonance for efficiency
    pub fn resonate_batch(&mut self, texts: &[&str]) -> Vec<ResonanceResult> {
        texts.iter().map(|t| self.resonate(t)).collect()
    }

    /// Find nearest CAM slot (used by both Pass 1 and Pass 2)
    fn find_nearest(&self, fp: &Fingerprint) -> (usize, u32, bool) {
        let mut best_idx = 0;
        let mut best_dist = u32::MAX;
        let mut is_archetype = true;

        // Check archetypes
        for (i, slot) in self.archetypes.iter().enumerate() {
            let dist = fp.hamming(&slot.fingerprint);
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
                is_archetype = true;
            }
        }

        // Check clusters
        for (i, slot) in self.clusters.iter().enumerate() {
            if slot.examples.is_empty() {
                continue;
            }
            let dist = fp.hamming(&slot.fingerprint);
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
                is_archetype = false;
            }
        }

        (best_idx, best_dist, is_archetype)
    }

    /// Get stats
    pub fn stats(&self) -> CAMStats {
        let archetype_populated = self
            .archetypes
            .iter()
            .filter(|s| !s.examples.is_empty())
            .count();
        let clusters_populated = self
            .clusters
            .iter()
            .filter(|s| !s.examples.is_empty())
            .count();

        CAMStats {
            total_slots: self.archetypes.len() + self.clusters.len(),
            archetype_slots: self.archetypes.len(),
            cluster_slots: self.clusters.len(),
            archetype_populated,
            clusters_populated,
            pass1_examples: self.pass1_examples,
            pass2_lookups: self.pass2_lookups,
            pass2_hit_rate: if self.pass2_lookups > 0 {
                self.pass2_hits as f64 / self.pass2_lookups as f64
            } else {
                0.0
            },
            memory_bytes: (self.archetypes.len() + self.clusters.len()) * N64 * 8,
        }
    }
}

#[derive(Debug)]
pub struct ResonanceResult {
    pub concept: ConceptType,
    pub similarity: f64,
    pub confidence: f64,
    pub hit: bool,
    pub slot_examples: usize,
}

#[derive(Debug)]
pub struct CAMStats {
    pub total_slots: usize,
    pub archetype_slots: usize,
    pub cluster_slots: usize,
    pub archetype_populated: usize,
    pub clusters_populated: usize,
    pub pass1_examples: usize,
    pub pass2_lookups: usize,
    pub pass2_hit_rate: f64,
    pub memory_bytes: usize,
}

// ============================================================================
// Demo
// ============================================================================

fn main() {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║      MULTI-PASS CODEBOOK: Concept CAM with Hamming Meta-Resonance     ║");
    println!("╠═══════════════════════════════════════════════════════════════════════╣");
    println!("║  Pass 1: Collect concepts (expensive Jina, one-time)                  ║");
    println!("║  Pass 2: Hamming resonance (cheap XOR+popcount, runtime)              ║");
    println!("╚═══════════════════════════════════════════════════════════════════════╝");
    println!();

    let mut cam = ConceptCAM::new();

    // =========================================================================
    // PASS 1: Train on concept corpus
    // =========================================================================

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PASS 1: Concept Collection (Training)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // NARS patterns
    let nars_corpus = vec![
        (
            "bird is-a animal".to_string(),
            Some(ConceptType::Inheritance),
        ),
        (
            "penguin is-a bird".to_string(),
            Some(ConceptType::Inheritance),
        ),
        (
            "mammal is-a animal".to_string(),
            Some(ConceptType::Inheritance),
        ),
        (
            "dog similar-to wolf".to_string(),
            Some(ConceptType::Similarity),
        ),
        (
            "cat resembles tiger".to_string(),
            Some(ConceptType::Similarity),
        ),
        (
            "rain implies wet".to_string(),
            Some(ConceptType::Implication),
        ),
        ("fire causes heat".to_string(), Some(ConceptType::Causes)),
        ("water enables life".to_string(), Some(ConceptType::Enables)),
    ];

    // SPO relations
    let spo_corpus = vec![
        (
            "function contains loop".to_string(),
            Some(ConceptType::Contains),
        ),
        (
            "class becomes instance".to_string(),
            Some(ConceptType::Becomes),
        ),
        (
            "theory refines hypothesis".to_string(),
            Some(ConceptType::Refines),
        ),
        (
            "evidence grounds belief".to_string(),
            Some(ConceptType::Grounds),
        ),
        (
            "model abstracts reality".to_string(),
            Some(ConceptType::Abstracts),
        ),
        (
            "claim contradicts evidence".to_string(),
            Some(ConceptType::Contradicts),
        ),
    ];

    // Qualia patterns
    let qualia_corpus = vec![
        (
            "warm fuzzy feeling comfort".to_string(),
            Some(ConceptType::Felt),
        ),
        (
            "excitement energy enthusiasm".to_string(),
            Some(ConceptType::Arousal),
        ),
        (
            "pleasant positive good".to_string(),
            Some(ConceptType::Valence),
        ),
        (
            "uncertain conflicted torn".to_string(),
            Some(ConceptType::Tension),
        ),
    ];

    // Unlabeled corpus (will cluster automatically)
    let unlabeled_corpus = vec![
        ("user authenticates with password".to_string(), None),
        ("login requires credentials".to_string(), None),
        ("session token expires".to_string(), None),
        ("database query returns results".to_string(), None),
        ("cache invalidation strategy".to_string(), None),
        ("network request timeout".to_string(), None),
        ("memory allocation failure".to_string(), None),
        ("thread synchronization lock".to_string(), None),
    ];

    let t0 = Instant::now();

    cam.train_corpus(&nars_corpus);
    cam.train_corpus(&spo_corpus);
    cam.train_corpus(&qualia_corpus);
    cam.train_corpus(&unlabeled_corpus);

    let train_time = t0.elapsed();

    let stats = cam.stats();
    println!("  Training corpus: {} examples", stats.pass1_examples);
    println!(
        "  Archetype slots populated: {} / {}",
        stats.archetype_populated, stats.archetype_slots
    );
    println!(
        "  Cluster slots populated: {} / {}",
        stats.clusters_populated, stats.cluster_slots
    );
    println!(
        "  Training time: {:.2}ms",
        train_time.as_secs_f64() * 1000.0
    );
    println!("  CAM memory: {} KB", stats.memory_bytes / 1024);
    println!();

    // =========================================================================
    // PASS 2: Hamming Meta-Resonance (Runtime)
    // =========================================================================

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PASS 2: Hamming Meta-Resonance (Runtime Lookup)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let test_queries = vec![
        // Should match NARS patterns
        "whale is-a mammal",
        "elephant similar to mammoth",
        "smoke implies fire",
        // Should match SPO relations
        "array contains elements",
        "caterpillar becomes butterfly",
        "assumption contradicts fact",
        // Should match qualia
        "warm nostalgic memory",
        "high energy vibrant",
        // Should match emergent clusters
        "user login session",
        "database connection pool",
        // Novel concepts
        "quantum entanglement superposition",
        "blockchain consensus mechanism",
    ];

    let t0 = Instant::now();

    for query in &test_queries {
        let result = cam.resonate(query);
        let hit_marker = if result.hit { "✓" } else { "○" };
        println!("  {} \"{}\"", hit_marker, query);
        println!(
            "     → {:?} (sim={:.3}, conf={:.2}, examples={})",
            result.concept, result.similarity, result.confidence, result.slot_examples
        );
    }

    let resonate_time = t0.elapsed();

    println!();
    println!(
        "  {} lookups in {:.3}ms = {:.1}µs per lookup",
        test_queries.len(),
        resonate_time.as_secs_f64() * 1000.0,
        resonate_time.as_secs_f64() * 1_000_000.0 / test_queries.len() as f64
    );

    // =========================================================================
    // Benchmark: Throughput
    // =========================================================================

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Throughput Benchmark");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Generate 10K random queries
    let bench_queries: Vec<String> = (0..10_000)
        .map(|i| format!("concept query test number {} with variation {}", i, i % 100))
        .collect();

    let t0 = Instant::now();
    for query in &bench_queries {
        let _ = cam.resonate(query);
    }
    let bench_time = t0.elapsed();

    let stats = cam.stats();
    println!(
        "  10K lookups in {:.2}ms",
        bench_time.as_secs_f64() * 1000.0
    );
    println!(
        "  Throughput: {:.0} lookups/sec",
        10_000.0 / bench_time.as_secs_f64()
    );
    println!("  Hit rate: {:.1}%", stats.pass2_hit_rate * 100.0);
    println!();

    // =========================================================================
    // Summary
    // =========================================================================

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("SUMMARY: Multi-Pass Codebook Architecture");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("  ┌─────────────────────────────────────────────────────────────┐");
    println!("  │  PASS 1: CONCEPT COLLECTION                                 │");
    println!("  │    • Rich corpus: books, NARS, qualia, SPO                  │");
    println!("  │    • Jina embed → 10Kbit fingerprint                        │");
    println!("  │    • Cluster into CAM slots (archetypes + emergent)         │");
    println!("  │    • ONE-TIME cost, amortized across all future lookups     │");
    println!("  ├─────────────────────────────────────────────────────────────┤");
    println!("  │  PASS 2: HAMMING META-RESONANCE                             │");
    println!("  │    • New text → hash fingerprint (NO API CALL)              │");
    println!("  │    • XOR scan against CAM slots                             │");
    println!(
        "  │    • ~{} µs per lookup ({} slots × 256 XOR+popcnt)       │",
        bench_time.as_secs_f64() * 1_000_000.0 / 10_000.0,
        stats.total_slots
    );
    println!(
        "  │    • Memory: {} KB (fits in L2 cache)                      │",
        stats.memory_bytes / 1024
    );
    println!("  ├─────────────────────────────────────────────────────────────┤");
    println!("  │  THE INSIGHT:                                               │");
    println!("  │    The CAM IS the learned semantic space.                   │");
    println!("  │    Once trained, it's a pure resonance surface.             │");
    println!("  │    All lookups are binary operations—no embedding calls.    │");
    println!("  └─────────────────────────────────────────────────────────────┘");
    println!();
}
