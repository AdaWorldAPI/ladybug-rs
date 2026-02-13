//! Adversarial Self-Critique — structured challenges to beliefs.
//!
//! Five challenge types from Advocatus Diaboli:
//! 1. NEGATION:     What if the opposite is true?
//! 2. SUBSTITUTION: What if this is actually something else?
//! 3. DEPENDENCY:   What breaks if this is false?
//! 4. WEATHER:      Is this emerging from pressure, not truth?
//! 5. COMFORT:      What does believing this protect?
//!
//! # Science
//! - Wang (2006): NARS negation: ¬<f,c> = <1-f, c>
//! - Mercier & Sperber (2011): Argumentative theory of reasoning
//! - Kahneman (2011): Premortem technique

use super::TruthValue;

/// Kind of adversarial challenge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChallengeKind {
    /// What if the opposite is true?
    Negation,
    /// What if this is actually something else?
    Substitution,
    /// What breaks if this is false?
    Dependency,
    /// Is this emerging from pressure, not truth?
    Weather,
    /// What does believing this protect?
    Comfort,
}

/// Result of one adversarial challenge.
#[derive(Debug, Clone)]
pub struct Challenge {
    /// Type of challenge applied
    pub kind: ChallengeKind,
    /// Alternative truth value proposed by the challenge
    pub alternative_truth: TruthValue,
    /// Whether the original belief survives this challenge
    pub survives: bool,
    /// Strength of the challenge (0.0 = weak, 1.0 = devastating)
    pub strength: f32,
}

/// Skepticism schedule: increases doubt after consecutive high-confidence outputs.
///
/// # Science
/// - Descartes (1641): Methodological doubt
/// - Wang (2006): NARS confidence erosion for unvalidated beliefs
/// - Tetlock (2005): Superforecasters exhibit calibrated self-doubt
#[derive(Debug, Clone)]
pub struct SkepticismSchedule {
    consecutive_confident: u32,
    base_skepticism: f32,
}

impl SkepticismSchedule {
    pub fn new(base_skepticism: f32) -> Self {
        Self {
            consecutive_confident: 0,
            base_skepticism,
        }
    }

    /// Update schedule and return current skepticism level.
    /// Skepticism increases logarithmically with consecutive high-confidence outputs.
    pub fn update(&mut self, truth: &TruthValue) -> f32 {
        if truth.confidence > 0.8 {
            self.consecutive_confident += 1;
        } else {
            self.consecutive_confident = 0;
        }
        self.base_skepticism + (self.consecutive_confident as f32 + 1.0).ln() * 0.1
    }

    /// Reset consecutive count.
    pub fn reset(&mut self) {
        self.consecutive_confident = 0;
    }
}

/// Apply all five challenge types to a belief and return results.
///
/// Each challenge evaluates whether the truth value survives a specific
/// form of adversarial scrutiny.
pub fn critique(truth: &TruthValue) -> Vec<Challenge> {
    vec![
        negation(truth),
        substitution(truth),
        dependency(truth),
        weather(truth),
        comfort(truth),
    ]
}

/// Challenge 1: NEGATION — What if the opposite is true?
///
/// NARS negation: ¬<f,c> = <1-f, c>.
/// Survives if original expectation exceeds negated expectation.
pub fn negation(truth: &TruthValue) -> Challenge {
    let negated = truth.negation();
    Challenge {
        kind: ChallengeKind::Negation,
        alternative_truth: negated,
        survives: truth.expectation() > negated.expectation(),
        strength: negated.expectation(),
    }
}

/// Challenge 2: SUBSTITUTION — What if this is actually something else?
///
/// Reduces confidence by 30% (substitution introduces alternative).
/// Survives if confidence remains > 0.5.
pub fn substitution(truth: &TruthValue) -> Challenge {
    let alt = TruthValue::new(1.0 - truth.frequency, truth.confidence * 0.7);
    Challenge {
        kind: ChallengeKind::Substitution,
        alternative_truth: alt,
        survives: truth.confidence > 0.5,
        strength: 1.0 - truth.confidence,
    }
}

/// Challenge 3: DEPENDENCY — What breaks if this is false?
///
/// Tests: if confidence were halved, would we still believe?
/// Survives if expectation at halved confidence still exceeds 0.5.
pub fn dependency(truth: &TruthValue) -> Challenge {
    let degraded = TruthValue::new(truth.frequency, truth.confidence * 0.5);
    Challenge {
        kind: ChallengeKind::Dependency,
        alternative_truth: degraded,
        survives: degraded.expectation() > 0.5,
        strength: (truth.expectation() - degraded.expectation()).abs(),
    }
}

/// Challenge 4: WEATHER — Is this emerging from pressure, not truth?
///
/// "Weather" challenges test if the belief is inflated by external pressure
/// rather than genuine evidence. Modeled as: high frequency + low confidence
/// suggests pressure inflation.
pub fn weather(truth: &TruthValue) -> Challenge {
    // Pressure indicator: high frequency with low evidence (low confidence)
    let pressure = truth.frequency * (1.0 - truth.confidence);
    let genuine = truth.frequency * truth.confidence;
    Challenge {
        kind: ChallengeKind::Weather,
        alternative_truth: TruthValue::new(genuine, truth.confidence),
        survives: genuine > pressure,
        strength: pressure,
    }
}

/// Challenge 5: COMFORT — What does believing this protect?
///
/// Comfort beliefs resist updating (high confidence, mediocre frequency).
/// Tests whether the belief is held for protection rather than truth.
pub fn comfort(truth: &TruthValue) -> Challenge {
    // Comfort indicator: strong confidence despite ambiguous frequency
    let ambiguity = 1.0 - (2.0 * truth.frequency - 1.0).abs(); // max at f=0.5
    let comfort_score = ambiguity * truth.confidence;
    Challenge {
        kind: ChallengeKind::Comfort,
        alternative_truth: TruthValue::new(truth.frequency, truth.confidence * (1.0 - ambiguity)),
        survives: comfort_score < 0.3,
        strength: comfort_score,
    }
}

/// Determine if a belief is robust: survives at least 4 of 5 challenges.
pub fn is_robust(truth: &TruthValue) -> bool {
    let challenges = critique(truth);
    let survived = challenges.iter().filter(|c| c.survives).count();
    survived >= 4
}

/// Compute aggregate robustness score: proportion of challenges survived,
/// weighted by challenge strength.
pub fn robustness_score(truth: &TruthValue) -> f32 {
    let challenges = critique(truth);
    let total_weight: f32 = challenges.iter().map(|c| c.strength).sum();
    if total_weight <= 0.0 {
        return 1.0;
    }
    let survived_weight: f32 = challenges
        .iter()
        .filter(|c| c.survives)
        .map(|c| c.strength)
        .sum();
    survived_weight / total_weight
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strong_belief_survives() {
        let strong = TruthValue::new(0.9, 0.9);
        let challenges = critique(&strong);
        let survived = challenges.iter().filter(|c| c.survives).count();
        assert!(
            survived >= 3,
            "Strong belief should survive most challenges, got {survived}"
        );
    }

    #[test]
    fn test_weak_belief_challenged() {
        let weak = TruthValue::new(0.5, 0.2);
        let challenges = critique(&weak);
        let survived = challenges.iter().filter(|c| c.survives).count();
        assert!(
            survived <= 3,
            "Weak belief should be challenged, survived {survived}"
        );
    }

    #[test]
    fn test_negation_challenge() {
        let positive = TruthValue::new(0.8, 0.8);
        let c = negation(&positive);
        assert_eq!(c.kind, ChallengeKind::Negation);
        assert!(c.survives, "Positive belief should survive negation");

        let neutral = TruthValue::new(0.5, 0.5);
        let c2 = negation(&neutral);
        assert!(!c2.survives, "Neutral belief should not survive negation");
    }

    #[test]
    fn test_weather_challenge() {
        let inflated = TruthValue::new(0.9, 0.1); // High freq, low confidence → pressure
        let c = weather(&inflated);
        assert!(
            !c.survives,
            "Pressure-inflated belief should fail weather challenge"
        );

        let genuine = TruthValue::new(0.9, 0.9); // High freq, high confidence → genuine
        let c2 = weather(&genuine);
        assert!(
            c2.survives,
            "Genuine belief should survive weather challenge"
        );
    }

    #[test]
    fn test_skepticism_schedule() {
        let mut schedule = SkepticismSchedule::new(0.1);
        let confident = TruthValue::new(0.9, 0.9);

        let s1 = schedule.update(&confident);
        let s2 = schedule.update(&confident);
        let s3 = schedule.update(&confident);
        assert!(
            s3 > s2 && s2 > s1,
            "Skepticism should increase with consecutive confidence"
        );

        let uncertain = TruthValue::new(0.5, 0.3);
        let s4 = schedule.update(&uncertain);
        assert!(s4 < s3, "Skepticism should reset after uncertainty");
    }

    #[test]
    fn test_robustness_score() {
        let strong = TruthValue::new(0.95, 0.95);
        assert!(robustness_score(&strong) > 0.5);

        let weak = TruthValue::new(0.5, 0.1);
        assert!(robustness_score(&weak) < robustness_score(&strong));
    }

    #[test]
    fn test_is_robust() {
        let strong = TruthValue::new(0.95, 0.95);
        assert!(is_robust(&strong));
    }
}
