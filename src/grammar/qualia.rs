//! Qualia Field — 18D Phenomenal Coordinates
//!
//! The third vertex of the Grammar Triangle.
//! Captures the felt-sense dimensions of meaning.
//!
//! Based on experiential qualities relevant to meaning:
//! valence, arousal, dominance, intimacy, etc.

use crate::core::Fingerprint;

/// The 18 qualia dimensions
pub const QUALIA_DIMENSIONS: [&str; 18] = [
    "valence",        // Positive/negative feeling
    "arousal",        // Activation level (calm to excited)
    "dominance",      // Control/agency (submissive to dominant)
    "intimacy",       // Closeness (distant to intimate)
    "certainty",      // Epistemic confidence (uncertain to certain)
    "urgency",        // Temporal pressure (relaxed to urgent)
    "depth",          // Abstraction level (surface to deep)
    "novelty",        // Familiarity (routine to novel)
    "complexity",     // Cognitive load (simple to complex)
    "coherence",      // Internal consistency (fragmented to unified)
    "salience",       // Attention capture (background to salient)
    "aesthetic",      // Beauty/elegance (plain to beautiful)
    "moral",          // Ethical valence (wrong to right)
    "social",         // Social relevance (private to public)
    "temporal_span",  // Time scale (momentary to eternal)
    "spatial_span",   // Space scale (local to universal)
    "concreteness",   // Abstraction (abstract to concrete)
    "intentionality", // Directedness (aimless to purposeful)
];

/// 18D qualia field representing phenomenal experience
#[derive(Clone, Debug)]
pub struct QualiaField {
    /// Coordinates in 18D qualia space (all in 0.0-1.0 range)
    coordinates: [f32; 18],
}

impl Default for QualiaField {
    fn default() -> Self {
        // Default to neutral (0.5) on all dimensions
        Self {
            coordinates: [0.5; 18],
        }
    }
}

impl QualiaField {
    /// Create with neutral values
    pub fn new() -> Self {
        Self::default()
    }

    /// Create from explicit coordinates
    pub fn from_coordinates(coords: [f32; 18]) -> Self {
        let mut coordinates = coords;
        for c in coordinates.iter_mut() {
            *c = c.clamp(0.0, 1.0);
        }
        Self { coordinates }
    }

    /// Compute qualia from text using keyword activation
    pub fn from_text(text: &str) -> Self {
        let text_lower = text.to_lowercase();
        let mut coordinates = [0.5f32; 18];

        // Valence: positive vs negative
        coordinates[0] = Self::compute_valence(&text_lower);

        // Arousal: intensity markers
        coordinates[1] = Self::compute_arousal(&text_lower);

        // Dominance: control/agency
        coordinates[2] = Self::compute_dominance(&text_lower);

        // Intimacy: personal markers
        coordinates[3] = Self::compute_intimacy(&text_lower);

        // Certainty: epistemic confidence
        coordinates[4] = Self::compute_certainty(&text_lower);

        // Urgency: temporal pressure
        coordinates[5] = Self::compute_urgency(&text_lower);

        // Depth: abstraction level
        coordinates[6] = Self::compute_depth(&text_lower);

        // Novelty: familiarity
        coordinates[7] = Self::compute_novelty(&text_lower);

        // Complexity: cognitive load
        coordinates[8] = Self::compute_complexity(&text_lower);

        // Coherence: internal consistency
        coordinates[9] = Self::compute_coherence(&text_lower);

        // Salience: attention capture
        coordinates[10] = Self::compute_salience(&text_lower);

        // Aesthetic: beauty/elegance
        coordinates[11] = Self::compute_aesthetic(&text_lower);

        // Moral: ethical valence
        coordinates[12] = Self::compute_moral(&text_lower);

        // Social: social relevance
        coordinates[13] = Self::compute_social(&text_lower);

        // Temporal span
        coordinates[14] = Self::compute_temporal_span(&text_lower);

        // Spatial span
        coordinates[15] = Self::compute_spatial_span(&text_lower);

        // Concreteness
        coordinates[16] = Self::compute_concreteness(&text_lower);

        // Intentionality
        coordinates[17] = Self::compute_intentionality(&text_lower);

        Self { coordinates }
    }

    // === Dimension computation functions ===

    fn compute_valence(text: &str) -> f32 {
        let positive = [
            "love",
            "joy",
            "bright",
            "beautiful",
            "wonder",
            "delight",
            "happy",
            "good",
            "great",
            "excellent",
            "amazing",
            "wonderful",
        ];
        let negative = [
            "hate", "sorrow", "dark", "terrible", "cruel", "pain", "death", "bad", "awful",
            "horrible", "sad", "angry",
        ];

        let pos_count: usize = positive.iter().map(|w| text.matches(w).count()).sum();
        let neg_count: usize = negative.iter().map(|w| text.matches(w).count()).sum();

        let diff = (pos_count as f32 - neg_count as f32) / 5.0;
        (diff + 1.0) / 2.0 // Normalize to 0-1
    }

    fn compute_arousal(text: &str) -> f32 {
        let high = [
            "!",
            "suddenly",
            "burst",
            "cry",
            "passion",
            "fire",
            "explode",
            "intense",
            "extreme",
            "overwhelming",
            "ecstatic",
            "furious",
        ];
        let low = [
            "calm", "peaceful", "quiet", "gentle", "soft", "serene", "tranquil", "relaxed",
            "still", "silent",
        ];

        let high_count: usize = high.iter().map(|w| text.matches(w).count()).sum();
        let low_count: usize = low.iter().map(|w| text.matches(w).count()).sum();

        let arousal = (high_count as f32 - low_count as f32 * 0.5) * 0.2;
        (arousal + 0.5).clamp(0.0, 1.0)
    }

    fn compute_dominance(text: &str) -> f32 {
        let dominant = [
            "control",
            "power",
            "command",
            "lead",
            "decide",
            "force",
            "dominate",
            "rule",
            "authority",
            "master",
        ];
        let submissive = [
            "submit",
            "obey",
            "follow",
            "serve",
            "yield",
            "surrender",
            "helpless",
            "weak",
            "dependent",
        ];

        let dom_count: usize = dominant.iter().map(|w| text.matches(w).count()).sum();
        let sub_count: usize = submissive.iter().map(|w| text.matches(w).count()).sum();

        let score = (dom_count as f32 - sub_count as f32) * 0.2;
        (score + 0.5).clamp(0.0, 1.0)
    }

    fn compute_intimacy(text: &str) -> f32 {
        let intimate = [
            "heart", "soul", "love", "dear", "soft", "gentle", "whisper", "touch", "embrace",
            "kiss", "close", "together", "us", "we",
        ];

        let count: usize = intimate.iter().map(|w| text.matches(w).count()).sum();
        (count as f32 * 0.15).min(1.0)
    }

    fn compute_certainty(text: &str) -> f32 {
        let certain = [
            "definitely",
            "certainly",
            "absolutely",
            "clearly",
            "obviously",
            "undoubtedly",
            "surely",
            "must",
            "always",
            "never",
        ];
        let uncertain = [
            "maybe",
            "perhaps",
            "possibly",
            "might",
            "could",
            "seems",
            "appears",
            "uncertain",
            "unclear",
            "sometimes",
        ];

        let cert_count: usize = certain.iter().map(|w| text.matches(w).count()).sum();
        let uncert_count: usize = uncertain.iter().map(|w| text.matches(w).count()).sum();

        let score = (cert_count as f32 - uncert_count as f32) * 0.2;
        (score + 0.5).clamp(0.0, 1.0)
    }

    fn compute_urgency(text: &str) -> f32 {
        let urgent = [
            "now",
            "immediately",
            "urgent",
            "hurry",
            "quick",
            "fast",
            "asap",
            "deadline",
            "emergency",
            "critical",
            "rush",
        ];

        let count: usize = urgent.iter().map(|w| text.matches(w).count()).sum();
        (count as f32 * 0.2).min(1.0)
    }

    fn compute_depth(text: &str) -> f32 {
        let deep = [
            "fundamental",
            "essence",
            "core",
            "underlying",
            "profound",
            "deep",
            "meaning",
            "philosophy",
            "metaphysical",
            "existential",
        ];
        let surface = [
            "surface",
            "obvious",
            "simple",
            "basic",
            "straightforward",
            "literal",
            "plain",
            "clear",
        ];

        let deep_count: usize = deep.iter().map(|w| text.matches(w).count()).sum();
        let surf_count: usize = surface.iter().map(|w| text.matches(w).count()).sum();

        let score = (deep_count as f32 - surf_count as f32) * 0.2;
        (score + 0.5).clamp(0.0, 1.0)
    }

    fn compute_novelty(text: &str) -> f32 {
        let novel = [
            "new",
            "novel",
            "unique",
            "original",
            "innovative",
            "fresh",
            "unprecedented",
            "surprising",
            "unexpected",
            "discovery",
        ];
        let familiar = [
            "usual", "typical", "common", "normal", "regular", "routine", "standard", "ordinary",
            "familiar",
        ];

        let novel_count: usize = novel.iter().map(|w| text.matches(w).count()).sum();
        let fam_count: usize = familiar.iter().map(|w| text.matches(w).count()).sum();

        let score = (novel_count as f32 - fam_count as f32) * 0.2;
        (score + 0.5).clamp(0.0, 1.0)
    }

    fn compute_complexity(text: &str) -> f32 {
        let complex = [
            "complex",
            "complicated",
            "intricate",
            "sophisticated",
            "nuanced",
            "multifaceted",
            "layered",
            "elaborate",
        ];
        let simple = [
            "simple",
            "easy",
            "basic",
            "straightforward",
            "clear",
            "plain",
            "obvious",
            "direct",
        ];

        let comp_count: usize = complex.iter().map(|w| text.matches(w).count()).sum();
        let simp_count: usize = simple.iter().map(|w| text.matches(w).count()).sum();

        let score = (comp_count as f32 - simp_count as f32) * 0.2;
        (score + 0.5).clamp(0.0, 1.0)
    }

    fn compute_coherence(text: &str) -> f32 {
        let coherent = [
            "therefore",
            "thus",
            "because",
            "so",
            "hence",
            "consequently",
            "connected",
            "unified",
            "consistent",
            "follows",
        ];
        let fragmented = [
            "but",
            "however",
            "although",
            "despite",
            "yet",
            "random",
            "scattered",
            "disconnected",
        ];

        let coh_count: usize = coherent.iter().map(|w| text.matches(w).count()).sum();
        let frag_count: usize = fragmented.iter().map(|w| text.matches(w).count()).sum();

        let score = (coh_count as f32 - frag_count as f32 * 0.5) * 0.15;
        (score + 0.5).clamp(0.0, 1.0)
    }

    fn compute_salience(text: &str) -> f32 {
        let salient = [
            "important",
            "crucial",
            "critical",
            "essential",
            "key",
            "vital",
            "significant",
            "notable",
            "remarkable",
            "!",
        ];

        let count: usize = salient.iter().map(|w| text.matches(w).count()).sum();
        (count as f32 * 0.15).min(1.0)
    }

    fn compute_aesthetic(text: &str) -> f32 {
        let beautiful = [
            "beautiful",
            "elegant",
            "graceful",
            "lovely",
            "stunning",
            "gorgeous",
            "artistic",
            "aesthetic",
            "refined",
            "exquisite",
        ];
        let ugly = ["ugly", "crude", "harsh", "rough", "plain", "dull"];

        let beau_count: usize = beautiful.iter().map(|w| text.matches(w).count()).sum();
        let ugly_count: usize = ugly.iter().map(|w| text.matches(w).count()).sum();

        let score = (beau_count as f32 - ugly_count as f32) * 0.2;
        (score + 0.5).clamp(0.0, 1.0)
    }

    fn compute_moral(text: &str) -> f32 {
        let good_moral = [
            "right",
            "ethical",
            "moral",
            "just",
            "fair",
            "honest",
            "virtuous",
            "noble",
            "honorable",
            "good",
        ];
        let bad_moral = [
            "wrong",
            "immoral",
            "unjust",
            "unfair",
            "dishonest",
            "evil",
            "corrupt",
            "wicked",
        ];

        let good_count: usize = good_moral.iter().map(|w| text.matches(w).count()).sum();
        let bad_count: usize = bad_moral.iter().map(|w| text.matches(w).count()).sum();

        let score = (good_count as f32 - bad_count as f32) * 0.2;
        (score + 0.5).clamp(0.0, 1.0)
    }

    fn compute_social(text: &str) -> f32 {
        let social = [
            "people",
            "society",
            "community",
            "public",
            "social",
            "everyone",
            "we",
            "us",
            "together",
            "shared",
        ];
        let private = [
            "private",
            "personal",
            "individual",
            "alone",
            "solitary",
            "secret",
            "hidden",
        ];

        let soc_count: usize = social.iter().map(|w| text.matches(w).count()).sum();
        let priv_count: usize = private.iter().map(|w| text.matches(w).count()).sum();

        let score = (soc_count as f32 - priv_count as f32) * 0.15;
        (score + 0.5).clamp(0.0, 1.0)
    }

    fn compute_temporal_span(text: &str) -> f32 {
        let eternal = [
            "forever",
            "eternal",
            "always",
            "never-ending",
            "timeless",
            "permanent",
            "everlasting",
            "infinite",
        ];
        let momentary = [
            "moment",
            "instant",
            "brief",
            "temporary",
            "fleeting",
            "now",
            "today",
            "currently",
        ];

        let eter_count: usize = eternal.iter().map(|w| text.matches(w).count()).sum();
        let mom_count: usize = momentary.iter().map(|w| text.matches(w).count()).sum();

        let score = (eter_count as f32 - mom_count as f32) * 0.2;
        (score + 0.5).clamp(0.0, 1.0)
    }

    fn compute_spatial_span(text: &str) -> f32 {
        let universal = [
            "universe",
            "world",
            "global",
            "everywhere",
            "all",
            "universal",
            "cosmic",
            "infinite",
            "vast",
        ];
        let local = [
            "here",
            "local",
            "nearby",
            "this place",
            "small",
            "particular",
            "specific",
        ];

        let univ_count: usize = universal.iter().map(|w| text.matches(w).count()).sum();
        let local_count: usize = local.iter().map(|w| text.matches(w).count()).sum();

        let score = (univ_count as f32 - local_count as f32) * 0.2;
        (score + 0.5).clamp(0.0, 1.0)
    }

    fn compute_concreteness(text: &str) -> f32 {
        let concrete = [
            "physical", "tangible", "material", "solid", "real", "object", "thing", "body", "hand",
            "see", "touch",
        ];
        let abstract_words = [
            "concept",
            "idea",
            "abstract",
            "theoretical",
            "principle",
            "notion",
            "thought",
            "mental",
        ];

        let conc_count: usize = concrete.iter().map(|w| text.matches(w).count()).sum();
        let abs_count: usize = abstract_words.iter().map(|w| text.matches(w).count()).sum();

        let score = (conc_count as f32 - abs_count as f32) * 0.2;
        (score + 0.5).clamp(0.0, 1.0)
    }

    fn compute_intentionality(text: &str) -> f32 {
        let purposeful = [
            "intend",
            "purpose",
            "goal",
            "aim",
            "want",
            "plan",
            "decide",
            "choose",
            "will",
            "deliberately",
        ];
        let aimless = [
            "random",
            "accidental",
            "happen",
            "chance",
            "aimless",
            "wandering",
            "drift",
        ];

        let purp_count: usize = purposeful.iter().map(|w| text.matches(w).count()).sum();
        let aim_count: usize = aimless.iter().map(|w| text.matches(w).count()).sum();

        let score = (purp_count as f32 - aim_count as f32) * 0.2;
        (score + 0.5).clamp(0.0, 1.0)
    }

    // === Accessor methods ===

    /// Get coordinate value by dimension name
    pub fn get(&self, dimension: &str) -> Option<f32> {
        QUALIA_DIMENSIONS
            .iter()
            .position(|d| *d == dimension)
            .map(|i| self.coordinates[i])
    }

    /// Set coordinate value by dimension name
    pub fn set(&mut self, dimension: &str, value: f32) {
        if let Some(i) = QUALIA_DIMENSIONS.iter().position(|d| *d == dimension) {
            self.coordinates[i] = value.clamp(0.0, 1.0);
        }
    }

    /// Get all coordinates
    pub fn coordinates(&self) -> &[f32; 18] {
        &self.coordinates
    }

    /// Euclidean distance to another qualia field
    pub fn distance(&self, other: &Self) -> f32 {
        self.coordinates
            .iter()
            .zip(other.coordinates.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Cosine similarity to another qualia field
    pub fn similarity(&self, other: &Self) -> f32 {
        let dot: f32 = self
            .coordinates
            .iter()
            .zip(other.coordinates.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm1: f32 = self.coordinates.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = other.coordinates.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm1 > 0.0 && norm2 > 0.0 {
            dot / (norm1 * norm2)
        } else {
            0.0
        }
    }

    /// Convert to fingerprint contribution
    pub fn to_fingerprint_contribution(&self) -> Fingerprint {
        let mut fp = Fingerprint::zero();

        // Each dimension gets ~555 bits (FINGERPRINT_BITS / 18)
        // We use threshold-based encoding
        for (i, coord) in self.coordinates.iter().enumerate() {
            let base_bit = i * 555;
            let num_bits = (*coord * 555.0) as usize;

            for j in 0..num_bits.min(555) {
                let bit_pos = base_bit + j;
                if bit_pos < crate::FINGERPRINT_BITS {
                    fp.set_bit(bit_pos, true);
                }
            }
        }

        fp
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valence_detection() {
        let positive = QualiaField::from_text("I love this beautiful wonderful day");
        let negative = QualiaField::from_text("I hate this terrible awful situation");

        assert!(positive.get("valence").unwrap() > 0.5);
        assert!(negative.get("valence").unwrap() < 0.5);
    }

    #[test]
    fn test_arousal_detection() {
        let high = QualiaField::from_text("Suddenly there was an intense explosion!");
        let low = QualiaField::from_text("The calm peaceful lake was serene and quiet");

        assert!(high.get("arousal").unwrap() > low.get("arousal").unwrap());
    }

    #[test]
    fn test_intimacy_detection() {
        let intimate = QualiaField::from_text("Our hearts touched as we embraced together");
        let distant = QualiaField::from_text("The quarterly report shows a 5% increase");

        assert!(intimate.get("intimacy").unwrap() > distant.get("intimacy").unwrap());
    }

    #[test]
    fn test_similarity() {
        let q1 = QualiaField::from_text("I feel happy and joyful");
        let q2 = QualiaField::from_text("I am delighted and pleased");
        let q3 = QualiaField::from_text("I am angry and furious");

        let sim_12 = q1.similarity(&q2);
        let sim_13 = q1.similarity(&q3);

        // Similar emotions should be closer
        assert!(sim_12 > sim_13);
    }
}
