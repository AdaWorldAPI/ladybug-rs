//! Cognitive Search Engine
//!
//! Unified search across three dimensions:
//! - **NARS**: Inference operations (deduction, induction, abduction, analogy)
//! - **Qualia**: Felt resonance (arousal, valence, tension, certainty)
//! - **SPO**: Graph structure (subject-predicate-object)
//!
//! This is "human-like" search: not just similarity, but meaning, feeling, and connection.
//!
//! # The Three Dimensions
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    COGNITIVE SEARCH SPACE                               │
//! │                                                                         │
//! │   NARS (Logic)              Qualia (Feel)           SPO (Structure)    │
//! │   ━━━━━━━━━━━━              ━━━━━━━━━━━━━           ━━━━━━━━━━━━━━     │
//! │   deduction                 arousal                 subject            │
//! │   induction                 valence                 predicate          │
//! │   abduction                 tension                 object             │
//! │   analogy                   certainty                                  │
//! │   revision                  agency                                     │
//! │   comparison                temporality                                │
//! │   contraposition            sociality                                  │
//! │   conversion                novelty                                    │
//! │                                                                         │
//! │   "What follows?"           "How does it feel?"     "How connected?"   │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Human Cognitive Operations
//!
//! ```text
//! FANOUT        → What connects to this? (SPO expansion)
//! EXTRAPOLATE   → What comes next? (NARS deduction + temporal)
//! EXPLORE       → What's nearby? (HDR cascade + qualia curiosity)
//! ABDUCT        → What explains this? (NARS abduction)
//! CONTRADICT    → What conflicts? (NARS negation + comparison)
//! INDUCE        → What pattern emerges? (NARS induction)
//! DEDUCE        → What must follow? (NARS deduction)
//! SYNTHESIZE    → How do these combine? (NARS + qualia bundle)
//! INFER         → What's implied? (NARS inheritance chain)
//! ASSOCIATE     → What's related? (qualia similarity)
//! INTUIT        → What feels right? (qualia resonance)
//! JUDGE         → Is this true? (NARS truth value)
//! ```

use crate::learning::cognitive_frameworks::{NarsInference, TruthValue};
use crate::search::causal::CausalSearch;
use crate::search::hdr_cascade::{HdrIndex, MexicanHat, RollingWindow, hamming_distance};

// =============================================================================
// CONSTANTS
// =============================================================================

const WORDS: usize = 256;

// =============================================================================
// QUALIA DIMENSIONS
// =============================================================================

/// 8-dimensional qualia vector (Russell's circumplex + extensions)
#[derive(Clone, Copy, Debug, Default)]
pub struct QualiaVector {
    /// Activation level (0.0 = calm, 1.0 = excited)
    pub arousal: f32,
    /// Hedonic tone (-1.0 = negative, 1.0 = positive)
    pub valence: f32,
    /// Stress level (0.0 = relaxed, 1.0 = tense)
    pub tension: f32,
    /// Epistemic certainty (0.0 = doubt, 1.0 = confident)
    pub certainty: f32,
    /// Sense of control (0.0 = helpless, 1.0 = empowered)
    pub agency: f32,
    /// Time pressure (0.0 = patient, 1.0 = urgent)
    pub temporality: f32,
    /// Social orientation (0.0 = avoidant, 1.0 = approach)
    pub sociality: f32,
    /// Pattern deviation (0.0 = familiar, 1.0 = surprising)
    pub novelty: f32,
}

impl QualiaVector {
    /// Create from array
    pub fn from_array(arr: [f32; 8]) -> Self {
        Self {
            arousal: arr[0],
            valence: arr[1],
            tension: arr[2],
            certainty: arr[3],
            agency: arr[4],
            temporality: arr[5],
            sociality: arr[6],
            novelty: arr[7],
        }
    }

    /// Convert to array
    pub fn to_array(&self) -> [f32; 8] {
        [
            self.arousal,
            self.valence,
            self.tension,
            self.certainty,
            self.agency,
            self.temporality,
            self.sociality,
            self.novelty,
        ]
    }

    /// Euclidean distance
    pub fn distance(&self, other: &QualiaVector) -> f32 {
        let a = self.to_array();
        let b = other.to_array();
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Cosine similarity
    pub fn similarity(&self, other: &QualiaVector) -> f32 {
        let a = self.to_array();
        let b = other.to_array();

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f32 = a.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();

        if mag_a < 1e-6 || mag_b < 1e-6 {
            return 0.0;
        }

        dot / (mag_a * mag_b)
    }

    /// Blend two qualia vectors
    pub fn blend(&self, other: &QualiaVector, weight: f32) -> QualiaVector {
        let a = self.to_array();
        let b = other.to_array();
        let blended: [f32; 8] = std::array::from_fn(|i| a[i] * (1.0 - weight) + b[i] * weight);
        QualiaVector::from_array(blended)
    }

    /// Mexican hat response in qualia space
    pub fn resonance(&self, other: &QualiaVector, excite: f32, inhibit: f32) -> f32 {
        let dist = self.distance(other);
        if dist < excite {
            1.0 - (dist / excite)
        } else if dist < inhibit {
            let t = (dist - excite) / (inhibit - excite);
            -0.5 * (1.0 - t)
        } else {
            0.0
        }
    }
}

// =============================================================================
// SPO TRIPLE
// =============================================================================

/// Subject-Predicate-Object triple
#[derive(Clone, Debug)]
pub struct SpoTriple {
    pub subject: [u64; WORDS],
    pub predicate: [u64; WORDS],
    pub object: [u64; WORDS],
    /// Bound fingerprint: S ⊗ P ⊗ O
    pub fingerprint: [u64; WORDS],
}

impl SpoTriple {
    /// Create from components
    pub fn new(subject: [u64; WORDS], predicate: [u64; WORDS], object: [u64; WORDS]) -> Self {
        let mut fingerprint = [0u64; WORDS];
        for i in 0..WORDS {
            fingerprint[i] = subject[i] ^ predicate[i] ^ object[i];
        }
        Self {
            subject,
            predicate,
            object,
            fingerprint,
        }
    }

    /// Unbind to get subject: fp ⊗ P ⊗ O = S
    pub fn unbind_subject(&self) -> [u64; WORDS] {
        let mut result = [0u64; WORDS];
        for i in 0..WORDS {
            result[i] = self.fingerprint[i] ^ self.predicate[i] ^ self.object[i];
        }
        result
    }

    /// Unbind to get predicate: fp ⊗ S ⊗ O = P
    pub fn unbind_predicate(&self) -> [u64; WORDS] {
        let mut result = [0u64; WORDS];
        for i in 0..WORDS {
            result[i] = self.fingerprint[i] ^ self.subject[i] ^ self.object[i];
        }
        result
    }

    /// Unbind to get object: fp ⊗ S ⊗ P = O
    pub fn unbind_object(&self) -> [u64; WORDS] {
        let mut result = [0u64; WORDS];
        for i in 0..WORDS {
            result[i] = self.fingerprint[i] ^ self.subject[i] ^ self.predicate[i];
        }
        result
    }
}

// =============================================================================
// COGNITIVE ATOM
// =============================================================================

/// An atom in cognitive space: fingerprint + qualia + truth value
#[derive(Clone, Debug)]
pub struct CognitiveAtom {
    /// Content fingerprint
    pub fingerprint: [u64; WORDS],
    /// Felt quality
    pub qualia: QualiaVector,
    /// NARS truth value
    pub truth: TruthValue,
    /// Optional label
    pub label: Option<String>,
    /// Timestamp
    pub timestamp: u64,
}

impl CognitiveAtom {
    pub fn new(fingerprint: [u64; WORDS]) -> Self {
        Self {
            fingerprint,
            qualia: QualiaVector::default(),
            truth: TruthValue::new(1.0, 0.5),
            label: None,
            timestamp: 0,
        }
    }

    pub fn with_qualia(mut self, qualia: QualiaVector) -> Self {
        self.qualia = qualia;
        self
    }

    pub fn with_truth(mut self, truth: TruthValue) -> Self {
        self.truth = truth;
        self
    }

    pub fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }
}

// =============================================================================
// COGNITIVE SEARCH RESULT
// =============================================================================

/// Result from cognitive search
#[derive(Clone, Debug)]
pub struct CognitiveResult {
    /// The found atom
    pub atom: CognitiveAtom,
    /// How it was found
    pub via: SearchVia,
    /// Relevance scores
    pub scores: RelevanceScores,
}

/// How the result was found
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchVia {
    // NARS inference
    Deduction,
    Induction,
    Abduction,
    Analogy,
    Revision,
    Comparison,

    // Qualia resonance
    QualiaMatch,
    ArousalMatch,
    ValenceMatch,

    // SPO structure
    SubjectMatch,
    PredicateMatch,
    ObjectMatch,

    // Hybrid
    Fanout,
    Extrapolate,
    Explore,
    Synthesize,
    Intuit,
}

/// Relevance scores across dimensions
#[derive(Clone, Copy, Debug, Default)]
pub struct RelevanceScores {
    /// Fingerprint similarity (Hamming)
    pub content: f32,
    /// Qualia resonance
    pub qualia: f32,
    /// NARS truth expectation
    pub truth: f32,
    /// SPO structural match
    pub structure: f32,
    /// Combined score
    pub combined: f32,
}

// =============================================================================
// COGNITIVE SEARCH ENGINE
// =============================================================================

/// Unified cognitive search across NARS, Qualia, and SPO
pub struct CognitiveSearch {
    /// Stored atoms
    atoms: Vec<CognitiveAtom>,
    /// SPO triples
    triples: Vec<SpoTriple>,
    /// HDR index for content search
    content_index: HdrIndex,
    /// Causal search for inference chains
    causal: CausalSearch,
    /// Mexican hat for qualia resonance
    qualia_hat: MexicanHat,
    /// Rolling window for coherence
    window: RollingWindow,
    /// NARS confidence horizon
    k: f32,
}

impl CognitiveSearch {
    /// Create new cognitive search engine
    pub fn new() -> Self {
        Self {
            atoms: Vec::new(),
            triples: Vec::new(),
            content_index: HdrIndex::new(),
            causal: CausalSearch::new(),
            qualia_hat: MexicanHat::new(500, 2000), // Tighter for qualia
            window: RollingWindow::new(100),
            k: 1.0,
        }
    }

    /// Add an atom
    pub fn add_atom(&mut self, atom: CognitiveAtom) {
        self.content_index.add(&atom.fingerprint);
        self.atoms.push(atom);
    }

    /// Add an SPO triple
    pub fn add_triple(&mut self, triple: SpoTriple) {
        self.content_index.add(&triple.fingerprint);
        self.triples.push(triple);
    }

    /// Add an atom with qualia
    pub fn add_with_qualia(
        &mut self,
        fingerprint: [u64; WORDS],
        qualia: QualiaVector,
        truth: TruthValue,
    ) {
        let atom = CognitiveAtom::new(fingerprint)
            .with_qualia(qualia)
            .with_truth(truth);
        self.add_atom(atom);
    }

    // =========================================================================
    // NARS INFERENCE OPERATIONS
    // =========================================================================

    /// DEDUCE: What must follow from these premises?
    /// {M → P, S → M} ⊢ S → P
    pub fn deduce(
        &self,
        premise1: &CognitiveAtom, // M → P
        premise2: &CognitiveAtom, // S → M
    ) -> Option<CognitiveResult> {
        // Deduction truth function
        let truth = NarsInference::deduction(premise1.truth, premise2.truth);

        // Bind conclusions: S → P
        // The conclusion fingerprint emerges from binding
        let mut conclusion_fp = [0u64; WORDS];
        for i in 0..WORDS {
            // S from premise2, P from premise1
            conclusion_fp[i] = premise2.fingerprint[i] ^ premise1.fingerprint[i];
        }

        // Blend qualia from premises
        let qualia = premise1.qualia.blend(&premise2.qualia, 0.5);

        let atom = CognitiveAtom::new(conclusion_fp)
            .with_qualia(qualia)
            .with_truth(truth);

        Some(CognitiveResult {
            atom,
            via: SearchVia::Deduction,
            scores: RelevanceScores {
                truth: truth.expectation(),
                combined: truth.expectation(),
                ..Default::default()
            },
        })
    }

    /// INDUCE: What pattern emerges from examples?
    /// {M → P, M → S} ⊢ S → P (with weak confidence)
    pub fn induce(
        &self,
        premise1: &CognitiveAtom, // M → P
        premise2: &CognitiveAtom, // M → S
    ) -> Option<CognitiveResult> {
        let truth = NarsInference::induction(premise1.truth, premise2.truth);

        let mut conclusion_fp = [0u64; WORDS];
        for i in 0..WORDS {
            conclusion_fp[i] = premise1.fingerprint[i] ^ premise2.fingerprint[i];
        }

        let qualia = premise1.qualia.blend(&premise2.qualia, 0.5);

        let atom = CognitiveAtom::new(conclusion_fp)
            .with_qualia(qualia)
            .with_truth(truth);

        Some(CognitiveResult {
            atom,
            via: SearchVia::Induction,
            scores: RelevanceScores {
                truth: truth.expectation(),
                combined: truth.expectation(),
                ..Default::default()
            },
        })
    }

    /// ABDUCT: What explains this observation?
    /// {P → M, S → M} ⊢ S → P (hypothesis)
    pub fn abduct(
        &self,
        premise1: &CognitiveAtom, // P → M
        premise2: &CognitiveAtom, // S → M
    ) -> Option<CognitiveResult> {
        let truth = NarsInference::abduction(premise1.truth, premise2.truth);

        let mut conclusion_fp = [0u64; WORDS];
        for i in 0..WORDS {
            conclusion_fp[i] = premise1.fingerprint[i] ^ premise2.fingerprint[i];
        }

        // Abduction increases novelty (it's a hypothesis)
        let mut qualia = premise1.qualia.blend(&premise2.qualia, 0.5);
        qualia.novelty = (qualia.novelty + 0.3).min(1.0);
        qualia.certainty *= 0.7; // Reduce certainty (it's a guess)

        let atom = CognitiveAtom::new(conclusion_fp)
            .with_qualia(qualia)
            .with_truth(truth);

        Some(CognitiveResult {
            atom,
            via: SearchVia::Abduction,
            scores: RelevanceScores {
                truth: truth.expectation(),
                combined: truth.expectation(),
                ..Default::default()
            },
        })
    }

    /// CONTRADICT: Find atoms that conflict with this one
    pub fn contradict(&self, query: &CognitiveAtom, k: usize) -> Vec<CognitiveResult> {
        // Look for atoms with opposite valence or high tension
        self.atoms
            .iter()
            .filter_map(|atom| {
                // Opposite valence = contradiction
                let valence_opposite = (query.qualia.valence - atom.qualia.valence).abs() > 1.5;
                // Low truth when query has high truth = contradiction
                let truth_conflict = query.truth.f > 0.7 && atom.truth.f < 0.3;

                if valence_opposite || truth_conflict {
                    let qualia_score = 1.0 - query.qualia.similarity(&atom.qualia);
                    Some(CognitiveResult {
                        atom: atom.clone(),
                        via: SearchVia::Comparison,
                        scores: RelevanceScores {
                            qualia: qualia_score,
                            truth: 1.0 - atom.truth.expectation(),
                            combined: qualia_score,
                            ..Default::default()
                        },
                    })
                } else {
                    None
                }
            })
            .take(k)
            .collect()
    }

    // =========================================================================
    // QUALIA RESONANCE OPERATIONS
    // =========================================================================

    /// INTUIT: Find atoms that feel similar
    pub fn intuit(&self, query_qualia: &QualiaVector, k: usize) -> Vec<CognitiveResult> {
        let mut results: Vec<_> = self
            .atoms
            .iter()
            .map(|atom| {
                let resonance = query_qualia.resonance(
                    &atom.qualia,
                    0.3, // Excite threshold
                    0.8, // Inhibit threshold
                );
                (atom, resonance)
            })
            .filter(|(_, r)| *r > 0.0)
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        results
            .into_iter()
            .take(k)
            .map(|(atom, resonance)| CognitiveResult {
                atom: atom.clone(),
                via: SearchVia::Intuit,
                scores: RelevanceScores {
                    qualia: resonance,
                    combined: resonance,
                    ..Default::default()
                },
            })
            .collect()
    }

    /// ASSOCIATE: Find atoms by qualia similarity (without Mexican hat)
    pub fn associate(&self, query_qualia: &QualiaVector, k: usize) -> Vec<CognitiveResult> {
        let mut results: Vec<_> = self
            .atoms
            .iter()
            .map(|atom| {
                let similarity = query_qualia.similarity(&atom.qualia);
                (atom, similarity)
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        results
            .into_iter()
            .take(k)
            .map(|(atom, similarity)| CognitiveResult {
                atom: atom.clone(),
                via: SearchVia::QualiaMatch,
                scores: RelevanceScores {
                    qualia: similarity,
                    combined: similarity,
                    ..Default::default()
                },
            })
            .collect()
    }

    /// Find by specific qualia dimension
    pub fn find_by_arousal(&self, min: f32, max: f32, k: usize) -> Vec<CognitiveResult> {
        self.atoms
            .iter()
            .filter(|atom| atom.qualia.arousal >= min && atom.qualia.arousal <= max)
            .take(k)
            .map(|atom| CognitiveResult {
                atom: atom.clone(),
                via: SearchVia::ArousalMatch,
                scores: RelevanceScores {
                    qualia: atom.qualia.arousal,
                    combined: atom.qualia.arousal,
                    ..Default::default()
                },
            })
            .collect()
    }

    pub fn find_by_valence(&self, min: f32, max: f32, k: usize) -> Vec<CognitiveResult> {
        self.atoms
            .iter()
            .filter(|atom| atom.qualia.valence >= min && atom.qualia.valence <= max)
            .take(k)
            .map(|atom| CognitiveResult {
                atom: atom.clone(),
                via: SearchVia::ValenceMatch,
                scores: RelevanceScores {
                    qualia: atom.qualia.valence,
                    combined: atom.qualia.valence,
                    ..Default::default()
                },
            })
            .collect()
    }

    // =========================================================================
    // SPO GRAPH OPERATIONS
    // =========================================================================

    /// FANOUT: Find all triples connected to a subject
    pub fn fanout_subject(&self, subject: &[u64; WORDS], k: usize) -> Vec<CognitiveResult> {
        self.triples
            .iter()
            .filter(|t| hamming_distance(&t.subject, subject) < 500)
            .take(k)
            .map(|triple| {
                let atom = CognitiveAtom::new(triple.fingerprint);
                CognitiveResult {
                    atom,
                    via: SearchVia::SubjectMatch,
                    scores: RelevanceScores {
                        structure: 1.0,
                        combined: 1.0,
                        ..Default::default()
                    },
                }
            })
            .collect()
    }

    /// Find triples by predicate
    pub fn find_by_predicate(&self, predicate: &[u64; WORDS], k: usize) -> Vec<CognitiveResult> {
        self.triples
            .iter()
            .filter(|t| hamming_distance(&t.predicate, predicate) < 500)
            .take(k)
            .map(|triple| {
                let atom = CognitiveAtom::new(triple.fingerprint);
                CognitiveResult {
                    atom,
                    via: SearchVia::PredicateMatch,
                    scores: RelevanceScores {
                        structure: 1.0,
                        combined: 1.0,
                        ..Default::default()
                    },
                }
            })
            .collect()
    }

    /// Find triples by object
    pub fn find_by_object(&self, object: &[u64; WORDS], k: usize) -> Vec<CognitiveResult> {
        self.triples
            .iter()
            .filter(|t| hamming_distance(&t.object, object) < 500)
            .take(k)
            .map(|triple| {
                let atom = CognitiveAtom::new(triple.fingerprint);
                CognitiveResult {
                    atom,
                    via: SearchVia::ObjectMatch,
                    scores: RelevanceScores {
                        structure: 1.0,
                        combined: 1.0,
                        ..Default::default()
                    },
                }
            })
            .collect()
    }

    /// ABBA unbind: given triple and two components, find the third
    pub fn unbind_spo(
        &self,
        triple_fp: &[u64; WORDS],
        known1: &[u64; WORDS],
        known2: &[u64; WORDS],
    ) -> [u64; WORDS] {
        let mut result = [0u64; WORDS];
        for i in 0..WORDS {
            result[i] = triple_fp[i] ^ known1[i] ^ known2[i];
        }
        result
    }

    // =========================================================================
    // HYBRID OPERATIONS
    // =========================================================================

    /// EXPLORE: Find nearby in content + qualia space
    pub fn explore(
        &self,
        query_fp: &[u64; WORDS],
        query_qualia: &QualiaVector,
        k: usize,
    ) -> Vec<CognitiveResult> {
        // Search by content
        let content_results = self.content_index.search(query_fp, k * 2);

        // Re-score by combining content and qualia
        let mut results: Vec<_> = content_results
            .into_iter()
            .filter_map(|(idx, dist)| {
                if idx >= self.atoms.len() {
                    return None;
                }
                let atom = &self.atoms[idx];

                let content_score = 1.0 - (dist as f32 / crate::FINGERPRINT_BITS as f32);
                let qualia_score = query_qualia.similarity(&atom.qualia);
                let combined = 0.6 * content_score + 0.4 * qualia_score;

                Some(CognitiveResult {
                    atom: atom.clone(),
                    via: SearchVia::Explore,
                    scores: RelevanceScores {
                        content: content_score,
                        qualia: qualia_score,
                        combined,
                        ..Default::default()
                    },
                })
            })
            .collect();

        results.sort_by(|a, b| {
            b.scores
                .combined
                .partial_cmp(&a.scores.combined)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);
        results
    }

    /// EXTRAPOLATE: Given a sequence, what comes next?
    pub fn extrapolate(&self, sequence: &[[u64; WORDS]], k: usize) -> Vec<CognitiveResult> {
        if sequence.len() < 2 {
            return Vec::new();
        }

        // Compute "direction" as XOR of successive elements
        let mut direction = [0u64; WORDS];
        for i in 0..sequence.len() - 1 {
            for j in 0..WORDS {
                direction[j] ^= sequence[i][j] ^ sequence[i + 1][j];
            }
        }

        // Extrapolate: last element XOR direction
        let last = sequence.last().unwrap();
        let mut predicted = [0u64; WORDS];
        for i in 0..WORDS {
            predicted[i] = last[i] ^ direction[i];
        }

        // Find atoms near the predicted position
        let matches = self.content_index.search(&predicted, k);

        matches
            .into_iter()
            .filter_map(|(idx, dist)| {
                if idx >= self.atoms.len() {
                    return None;
                }
                let atom = &self.atoms[idx];
                let score = 1.0 - (dist as f32 / crate::FINGERPRINT_BITS as f32);

                Some(CognitiveResult {
                    atom: atom.clone(),
                    via: SearchVia::Extrapolate,
                    scores: RelevanceScores {
                        content: score,
                        combined: score,
                        ..Default::default()
                    },
                })
            })
            .collect()
    }

    /// SYNTHESIZE: Bundle multiple atoms into one
    pub fn synthesize(&self, atoms: &[CognitiveAtom]) -> CognitiveResult {
        if atoms.is_empty() {
            return CognitiveResult {
                atom: CognitiveAtom::new([0u64; WORDS]),
                via: SearchVia::Synthesize,
                scores: RelevanceScores::default(),
            };
        }

        // Bundle fingerprints (majority vote per bit)
        let mut counts = [0i32; WORDS * 64];
        for atom in atoms {
            for (i, &word) in atom.fingerprint.iter().enumerate() {
                for bit in 0..64 {
                    if (word >> bit) & 1 == 1 {
                        counts[i * 64 + bit] += 1;
                    } else {
                        counts[i * 64 + bit] -= 1;
                    }
                }
            }
        }

        let mut bundled = [0u64; WORDS];
        for i in 0..WORDS {
            for bit in 0..64 {
                if counts[i * 64 + bit] > 0 {
                    bundled[i] |= 1 << bit;
                }
            }
        }

        // Blend qualia
        let mut blended_qualia = QualiaVector::default();
        let n = atoms.len() as f32;
        for atom in atoms {
            let q = atom.qualia.to_array();
            let mut b = blended_qualia.to_array();
            for i in 0..8 {
                b[i] += q[i] / n;
            }
            blended_qualia = QualiaVector::from_array(b);
        }

        // Combine truth values (NARS revision)
        let mut combined_truth = atoms[0].truth;
        for atom in atoms.iter().skip(1) {
            combined_truth = NarsInference::revision(combined_truth, atom.truth);
        }

        let atom = CognitiveAtom::new(bundled)
            .with_qualia(blended_qualia)
            .with_truth(combined_truth);

        CognitiveResult {
            atom,
            via: SearchVia::Synthesize,
            scores: RelevanceScores {
                truth: combined_truth.expectation(),
                combined: combined_truth.expectation(),
                ..Default::default()
            },
        }
    }

    /// JUDGE: Evaluate truth of a statement
    pub fn judge(&self, statement: &[u64; WORDS]) -> TruthValue {
        // Find similar atoms and use their truth values
        let matches = self.content_index.search(statement, 5);

        if matches.is_empty() {
            return TruthValue::new(0.5, 0.1); // Unknown
        }

        // Weight by similarity
        let mut weighted_f = 0.0;
        let mut weighted_c = 0.0;
        let mut total_weight = 0.0;

        for (idx, dist) in matches {
            if idx >= self.atoms.len() {
                continue;
            }
            let atom = &self.atoms[idx];
            let weight = 1.0 - (dist as f32 / crate::FINGERPRINT_BITS as f32);

            weighted_f += atom.truth.f * weight;
            weighted_c += atom.truth.c * weight;
            total_weight += weight;
        }

        if total_weight < 1e-6 {
            return TruthValue::new(0.5, 0.1);
        }

        TruthValue::new(weighted_f / total_weight, weighted_c / total_weight)
    }

    // =========================================================================
    // COHERENCE
    // =========================================================================

    /// Get coherence stats
    pub fn coherence(&self) -> (f32, f32) {
        self.window.stats()
    }

    /// Is search pattern coherent?
    pub fn is_coherent(&self) -> bool {
        self.window.is_coherent(0.3)
    }
}

impl Default for CognitiveSearch {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn random_fp() -> [u64; WORDS] {
        let mut fp = [0u64; WORDS];
        for i in 0..WORDS {
            fp[i] = rand::random();
        }
        fp
    }

    #[test]
    fn test_qualia_vector() {
        let q1 = QualiaVector {
            arousal: 0.8,
            valence: 0.6,
            ..Default::default()
        };

        let q2 = QualiaVector {
            arousal: 0.7,
            valence: 0.5,
            ..Default::default()
        };

        let sim = q1.similarity(&q2);
        assert!(sim > 0.9); // Should be very similar

        let dist = q1.distance(&q2);
        assert!(dist < 0.5); // Should be close
    }

    #[test]
    fn test_spo_unbind() {
        let s = random_fp();
        let p = random_fp();
        let obj = random_fp();

        let triple = SpoTriple::new(s, p, obj);

        // Unbind should recover original components
        let recovered_s = triple.unbind_subject();
        let recovered_p = triple.unbind_predicate();
        let recovered_o = triple.unbind_object();

        assert_eq!(hamming_distance(&recovered_s, &s), 0);
        assert_eq!(hamming_distance(&recovered_p, &p), 0);
        assert_eq!(hamming_distance(&recovered_o, &obj), 0);
    }

    #[test]
    fn test_cognitive_search() {
        let mut search = CognitiveSearch::new();

        // Add atoms with qualia
        let fp1 = random_fp();
        let q1 = QualiaVector {
            arousal: 0.8,
            valence: 0.9,
            ..Default::default()
        };
        search.add_with_qualia(fp1, q1, TruthValue::new(0.9, 0.8));

        let fp2 = random_fp();
        let q2 = QualiaVector {
            arousal: 0.7,
            valence: 0.8,
            ..Default::default()
        };
        search.add_with_qualia(fp2, q2, TruthValue::new(0.8, 0.7));

        // Intuit should find similar qualia
        let query_q = QualiaVector {
            arousal: 0.75,
            valence: 0.85,
            ..Default::default()
        };
        let results = search.intuit(&query_q, 5);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_deduction() {
        let search = CognitiveSearch::new();

        let premise1 = CognitiveAtom::new(random_fp()).with_truth(TruthValue::new(0.9, 0.9));
        let premise2 = CognitiveAtom::new(random_fp()).with_truth(TruthValue::new(0.8, 0.8));

        let result = search.deduce(&premise1, &premise2);
        assert!(result.is_some());

        let r = result.unwrap();
        assert_eq!(r.via, SearchVia::Deduction);
        // Deduction should have lower confidence than premises
        assert!(r.atom.truth.c < premise1.truth.c);
    }

    #[test]
    fn test_synthesize() {
        let search = CognitiveSearch::new();

        let atoms: Vec<_> = (0..5)
            .map(|i| {
                CognitiveAtom::new(random_fp())
                    .with_truth(TruthValue::new(0.8, 0.7))
                    .with_qualia(QualiaVector {
                        arousal: 0.5 + i as f32 * 0.1,
                        ..Default::default()
                    })
            })
            .collect();

        let result = search.synthesize(&atoms);
        assert_eq!(result.via, SearchVia::Synthesize);
        // Synthesis should have combined truth
        assert!(result.atom.truth.c > 0.0);
    }
}
