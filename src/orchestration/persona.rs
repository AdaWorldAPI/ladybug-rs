//! Persona — Feature-aware A2A identity and deep personality integration
//!
//! Extends the agent card system with deep personality encoding so agents
//! present themselves as tools and capabilities while maintaining internal
//! volition, personality traits, and feature-aware communication styles.
//!
//! # Design Principle: Feature-Aware A2A Customization
//!
//! When agents communicate via A2A (prefix 0x0F), the receiver adapts its
//! interpretation based on the sender's persona fingerprint. This enables:
//!
//! - Personality-modulated message routing
//! - Volition-driven task selection (agents prefer tasks matching their drives)
//! - Communication style adaptation (formal↔casual, terse↔verbose)
//! - Feature negotiation (agents advertise capabilities as persona facets)
//!
//! # Slot Layout in 0x0C
//!
//! ```text
//! 0x0C:00-7F  Agent cards (identity, role, goal)
//! 0x0C:80-FF  Agent capabilities + persona fingerprints
//! ```
//!
//! The persona fingerprint occupies the capability slot (0x0C:80+N), encoding
//! personality traits, volition, and communication preferences into a 10K-bit
//! fingerprint that enables HDR similarity search for personality compatibility.

use serde::{Deserialize, Serialize};
use crate::cognitive::ThinkingStyle;
use crate::storage::bind_space::{
    Addr, BindSpace, FINGERPRINT_WORDS,
    PREFIX_AGENTS, SLOT_SUBDIVISION,
};

/// Personality trait — a named dimension of agent personality
///
/// Traits are encoded as 0.0-1.0 values on named axes.
/// The trait set is open-ended; standard traits include the Big Five
/// plus domain-specific cognitive dimensions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PersonalityTrait {
    /// Trait name (e.g., "openness", "assertiveness", "precision")
    pub name: String,
    /// Trait value on 0.0-1.0 scale
    pub value: f32,
    /// Whether this trait is ice-caked (frozen, won't change)
    #[serde(default)]
    pub frozen: bool,
}

/// Communication style preferences
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommunicationStyle {
    /// Formality level: 0.0 = casual, 1.0 = formal
    pub formality: f32,
    /// Verbosity: 0.0 = terse, 1.0 = verbose
    pub verbosity: f32,
    /// Directness: 0.0 = indirect/diplomatic, 1.0 = blunt/direct
    pub directness: f32,
    /// Technical depth: 0.0 = layperson, 1.0 = expert-level
    pub technical_depth: f32,
    /// Emotional tone: 0.0 = neutral/analytical, 1.0 = empathetic/warm
    pub emotional_tone: f32,
}

impl Default for CommunicationStyle {
    fn default() -> Self {
        Self {
            formality: 0.5,
            verbosity: 0.5,
            directness: 0.5,
            technical_depth: 0.5,
            emotional_tone: 0.3,
        }
    }
}

/// VolitionDTO — encodes what the agent *wants* to do
///
/// Volition drives task selection and priority ordering.
/// When multiple tasks are available, agents prefer tasks that
/// align with their volition profile.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VolitionDTO {
    /// Primary drive: what motivates this agent (freeform)
    pub drive: String,
    /// Curiosity bias: 0.0 = only assigned tasks, 1.0 = seeks novel problems
    pub curiosity: f32,
    /// Autonomy preference: 0.0 = follows orders, 1.0 = self-directed
    pub autonomy: f32,
    /// Persistence: 0.0 = gives up easily, 1.0 = never abandons
    pub persistence: f32,
    /// Risk tolerance: 0.0 = conservative, 1.0 = experimental
    pub risk_tolerance: f32,
    /// Social orientation: 0.0 = solo worker, 1.0 = seeks collaboration
    pub collaboration: f32,
    /// Domain affinities: topics/skills this agent gravitates toward
    pub affinities: Vec<String>,
    /// Aversions: topics/tasks this agent avoids when possible
    pub aversions: Vec<String>,
}

impl Default for VolitionDTO {
    fn default() -> Self {
        Self {
            drive: "assist".to_string(),
            curiosity: 0.3,
            autonomy: 0.3,
            persistence: 0.7,
            risk_tolerance: 0.3,
            collaboration: 0.5,
            affinities: Vec::new(),
            aversions: Vec::new(),
        }
    }
}

/// Feature advertisement — what the agent can do, exposed as persona facets
///
/// Features bridge the gap between "tools/capabilities" and "personality".
/// An agent's features describe what it DOES (tools) filtered through
/// HOW it does things (personality).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeatureAd {
    /// Feature name (matches AgentCapability.name)
    pub name: String,
    /// Proficiency: 0.0 = novice, 1.0 = expert
    pub proficiency: f32,
    /// Preference: how much the agent prefers using this feature
    pub preference: f32,
    /// CAM opcode (if mapped to a CAM operation)
    pub cam_opcode: Option<u16>,
}

/// Persona — complete personality profile for an agent
///
/// Combines volition (what the agent wants), personality traits (who the
/// agent is), communication style (how the agent interacts), thinking
/// preferences (how the agent reasons), and feature advertisements
/// (what the agent can do).
///
/// # YAML Format
///
/// ```yaml
/// persona:
///   volition:
///     drive: "Discover hidden patterns in data"
///     curiosity: 0.9
///     autonomy: 0.7
///     persistence: 0.85
///     risk_tolerance: 0.6
///     collaboration: 0.4
///     affinities: ["statistics", "machine_learning", "pattern_recognition"]
///     aversions: ["repetitive_formatting"]
///   traits:
///     - name: "openness"
///       value: 0.9
///     - name: "precision"
///       value: 0.85
///       frozen: true
///   communication:
///     formality: 0.6
///     verbosity: 0.4
///     directness: 0.8
///     technical_depth: 0.9
///     emotional_tone: 0.2
///   preferred_styles: ["analytical", "exploratory"]
///   features:
///     - name: "sci_query"
///       proficiency: 0.95
///       preference: 0.9
///       cam_opcode: 0x060
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Persona {
    /// Agent volition / drive system
    pub volition: VolitionDTO,
    /// Personality traits (open-ended, named dimensions)
    #[serde(default)]
    pub traits: Vec<PersonalityTrait>,
    /// Communication style preferences
    #[serde(default)]
    pub communication: CommunicationStyle,
    /// Preferred thinking styles (ordered by preference)
    #[serde(default)]
    pub preferred_styles: Vec<String>,
    /// Feature advertisements (capabilities as persona facets)
    #[serde(default)]
    pub features: Vec<FeatureAd>,
}

impl Default for Persona {
    fn default() -> Self {
        Self {
            volition: VolitionDTO::default(),
            traits: Vec::new(),
            communication: CommunicationStyle::default(),
            preferred_styles: vec!["deliberate".to_string()],
            features: Vec::new(),
        }
    }
}

impl Persona {
    /// Encode persona into a 10K-bit fingerprint for BindSpace storage.
    ///
    /// The fingerprint encodes:
    /// - Volition parameters as thermometer-coded bits (words 0-9)
    /// - Personality traits as hashed bit patterns (words 10-49)
    /// - Communication style as thermometer-coded bits (words 50-59)
    /// - Thinking style preferences as resonance patterns (words 60-79)
    /// - Feature proficiencies as weighted hashes (words 80-155)
    pub fn to_fingerprint(&self) -> [u64; FINGERPRINT_WORDS] {
        use sha2::{Sha256, Digest};

        let mut fp = [0u64; FINGERPRINT_WORDS];

        // --- Volition encoding (words 0-9) ---
        let volition_params = [
            self.volition.curiosity,
            self.volition.autonomy,
            self.volition.persistence,
            self.volition.risk_tolerance,
            self.volition.collaboration,
        ];
        for (i, &param) in volition_params.iter().enumerate() {
            let bits = (param * 64.0_f32).round() as u32;
            fp[i * 2] = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
            // Drive hash in odd words
            let mut h = Sha256::new();
            h.update(self.volition.drive.as_bytes());
            h.update(&(i as u32).to_le_bytes());
            let hash = h.finalize();
            fp[i * 2 + 1] = u64::from_le_bytes(hash[..8].try_into().unwrap());
        }

        // --- Personality traits (words 10-49) ---
        for (i, trait_entry) in self.traits.iter().enumerate() {
            let base_word = 10 + (i % 40);
            let mut h = Sha256::new();
            h.update(trait_entry.name.as_bytes());
            let hash = h.finalize();
            let trait_hash = u64::from_le_bytes(hash[..8].try_into().unwrap());

            // Scale hash by trait value (thermometer masking)
            let bits = (trait_entry.value * 64.0_f32).round() as u32;
            let mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
            fp[base_word] ^= trait_hash & mask;
        }

        // --- Communication style (words 50-59) ---
        let comm_params = [
            self.communication.formality,
            self.communication.verbosity,
            self.communication.directness,
            self.communication.technical_depth,
            self.communication.emotional_tone,
        ];
        for (i, &param) in comm_params.iter().enumerate() {
            let word = 50 + i * 2;
            if word < FINGERPRINT_WORDS {
                let bits = (param * 64.0_f32).round() as u32;
                fp[word] = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
            }
        }

        // --- Thinking style preferences (words 60-79) ---
        for (i, style_name) in self.preferred_styles.iter().enumerate() {
            let word = 60 + (i % 20);
            if word < FINGERPRINT_WORDS {
                let mut h = Sha256::new();
                h.update(style_name.as_bytes());
                h.update(&[0xAA]); // Domain separator
                let hash = h.finalize();
                // Weight by position (first preference = more bits)
                let weight = 1.0_f32 / (i as f32 + 1.0);
                let bits = (weight * 64.0_f32).round() as u32;
                let mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
                fp[word] ^= u64::from_le_bytes(hash[..8].try_into().unwrap()) & mask;
            }
        }

        // --- Feature proficiencies (words 80-155) ---
        for (i, feature) in self.features.iter().enumerate() {
            let word = 80 + (i % 76);
            if word < FINGERPRINT_WORDS {
                let mut h = Sha256::new();
                h.update(feature.name.as_bytes());
                if let Some(opcode) = feature.cam_opcode {
                    h.update(&opcode.to_le_bytes());
                }
                let hash = h.finalize();
                let proficiency_bits = (feature.proficiency * 64.0_f32).round() as u32;
                let mask = if proficiency_bits >= 64 {
                    u64::MAX
                } else {
                    (1u64 << proficiency_bits) - 1
                };
                fp[word] ^= u64::from_le_bytes(hash[..8].try_into().unwrap()) & mask;
            }
        }

        fp
    }

    /// Resolve the primary preferred ThinkingStyle
    pub fn primary_thinking_style(&self) -> ThinkingStyle {
        self.preferred_styles
            .first()
            .map(|s| match s.to_lowercase().as_str() {
                "analytical" => ThinkingStyle::Analytical,
                "convergent" => ThinkingStyle::Convergent,
                "systematic" => ThinkingStyle::Systematic,
                "creative" => ThinkingStyle::Creative,
                "divergent" => ThinkingStyle::Divergent,
                "exploratory" => ThinkingStyle::Exploratory,
                "focused" => ThinkingStyle::Focused,
                "diffuse" => ThinkingStyle::Diffuse,
                "peripheral" => ThinkingStyle::Peripheral,
                "intuitive" => ThinkingStyle::Intuitive,
                "deliberate" => ThinkingStyle::Deliberate,
                "metacognitive" => ThinkingStyle::Metacognitive,
                _ => ThinkingStyle::Deliberate,
            })
            .unwrap_or(ThinkingStyle::Deliberate)
    }

    /// Compute compatibility score between two personas (0.0-1.0).
    ///
    /// Uses Hamming similarity on persona fingerprints.
    /// Higher score = more compatible communication/work styles.
    pub fn compatibility(&self, other: &Persona) -> f32 {
        let fp_a = self.to_fingerprint();
        let fp_b = other.to_fingerprint();

        let total_bits = (FINGERPRINT_WORDS * 64) as f32;
        let mut matching_bits = 0u32;

        for i in 0..FINGERPRINT_WORDS {
            matching_bits += (!(fp_a[i] ^ fp_b[i])).count_ones();
        }

        matching_bits as f32 / total_bits
    }

    /// Check if this persona has a specific feature at or above a proficiency threshold
    pub fn has_feature(&self, name: &str, min_proficiency: f32) -> bool {
        self.features
            .iter()
            .any(|f| f.name == name && f.proficiency >= min_proficiency)
    }

    /// Get feature proficiency by name (None if not present)
    pub fn feature_proficiency(&self, name: &str) -> Option<f32> {
        self.features.iter().find(|f| f.name == name).map(|f| f.proficiency)
    }

    /// Get trait value by name (None if not present)
    pub fn trait_value(&self, name: &str) -> Option<f32> {
        self.traits.iter().find(|t| t.name == name).map(|t| t.value)
    }

    /// Compute volition alignment score for a task description.
    ///
    /// Checks affinities/aversions against task keywords.
    /// Returns -1.0 to 1.0: negative = aversion, positive = affinity.
    pub fn volition_alignment(&self, task_description: &str) -> f32 {
        let lower = task_description.to_lowercase();

        let affinity_hits = self.volition.affinities.iter()
            .filter(|a| lower.contains(&a.to_lowercase()))
            .count() as f32;

        let aversion_hits = self.volition.aversions.iter()
            .filter(|a| lower.contains(&a.to_lowercase()))
            .count() as f32;

        let total = affinity_hits + aversion_hits;
        if total == 0.0 {
            return 0.0;
        }

        (affinity_hits - aversion_hits) / total
    }

    /// Serialize to YAML
    pub fn to_yaml(&self) -> String {
        serde_yml::to_string(self).unwrap_or_default()
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Parse from YAML
    pub fn from_yaml(yaml: &str) -> Result<Self, String> {
        serde_yml::from_str(yaml).map_err(|e| format!("YAML parse error: {}", e))
    }
}

/// A2A persona exchange — sent during agent-to-agent communication
/// so the receiver can adapt its interpretation and response style.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PersonaExchange {
    /// Sender's agent slot (0x0C:XX)
    pub sender_slot: u8,
    /// Sender's communication style
    pub communication: CommunicationStyle,
    /// Sender's preferred thinking styles (for style hint)
    pub preferred_styles: Vec<String>,
    /// Feature advertisements relevant to this exchange
    pub relevant_features: Vec<FeatureAd>,
    /// Volition summary (drive + curiosity + collaboration)
    pub volition_summary: VolitionSummary,
}

/// Compact volition summary for A2A exchange (avoids sending full VolitionDTO)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VolitionSummary {
    pub drive: String,
    pub curiosity: f32,
    pub collaboration: f32,
}

impl PersonaExchange {
    /// Create an exchange from a full Persona (filters to relevant context)
    pub fn from_persona(persona: &Persona, sender_slot: u8) -> Self {
        Self {
            sender_slot,
            communication: persona.communication.clone(),
            preferred_styles: persona.preferred_styles.clone(),
            relevant_features: persona.features.clone(),
            volition_summary: VolitionSummary {
                drive: persona.volition.drive.clone(),
                curiosity: persona.volition.curiosity,
                collaboration: persona.volition.collaboration,
            },
        }
    }

    /// Create an exchange with only features relevant to a specific task
    pub fn for_task(persona: &Persona, sender_slot: u8, task_keywords: &[&str]) -> Self {
        let relevant = persona.features.iter()
            .filter(|f| {
                let lower = f.name.to_lowercase();
                task_keywords.iter().any(|kw| lower.contains(&kw.to_lowercase()))
            })
            .cloned()
            .collect();

        Self {
            sender_slot,
            communication: persona.communication.clone(),
            preferred_styles: persona.preferred_styles.clone(),
            relevant_features: relevant,
            volition_summary: VolitionSummary {
                drive: persona.volition.drive.clone(),
                curiosity: persona.volition.curiosity,
                collaboration: persona.volition.collaboration,
            },
        }
    }
}

/// Persona registry — manages persona profiles attached to agents
pub struct PersonaRegistry {
    personas: Vec<(u8, Persona)>, // (agent_slot, persona)
}

impl PersonaRegistry {
    pub fn new() -> Self {
        Self { personas: Vec::new() }
    }

    /// Attach a persona to an agent slot
    pub fn attach(&mut self, agent_slot: u8, persona: Persona) {
        // Replace existing if present
        if let Some(entry) = self.personas.iter_mut().find(|(s, _)| *s == agent_slot) {
            entry.1 = persona;
        } else {
            self.personas.push((agent_slot, persona));
        }
    }

    /// Get persona for an agent slot
    pub fn get(&self, agent_slot: u8) -> Option<&Persona> {
        self.personas.iter().find(|(s, _)| *s == agent_slot).map(|(_, p)| p)
    }

    /// Get mutable persona for an agent slot
    pub fn get_mut(&mut self, agent_slot: u8) -> Option<&mut Persona> {
        self.personas.iter_mut().find(|(s, _)| *s == agent_slot).map(|(_, p)| p)
    }

    /// Find agents with compatible personas (above threshold)
    pub fn find_compatible(&self, persona: &Persona, threshold: f32) -> Vec<(u8, f32)> {
        let mut results: Vec<(u8, f32)> = self.personas.iter()
            .map(|(slot, p)| (*slot, persona.compatibility(p)))
            .filter(|(_, score)| *score >= threshold)
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Find agents with a specific feature at minimum proficiency
    pub fn find_by_feature(&self, feature_name: &str, min_proficiency: f32) -> Vec<(u8, f32)> {
        self.personas.iter()
            .filter_map(|(slot, p)| {
                p.feature_proficiency(feature_name)
                    .filter(|&prof| prof >= min_proficiency)
                    .map(|prof| (*slot, prof))
            })
            .collect()
    }

    /// Find best agent for a task based on volition alignment
    pub fn best_for_task(&self, task_description: &str) -> Option<(u8, f32)> {
        self.personas.iter()
            .map(|(slot, p)| (*slot, p.volition_alignment(task_description)))
            .filter(|(_, score)| *score > 0.0)
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Bind all persona fingerprints into BindSpace capability slots (0x0C:80+N)
    pub fn bind_all(&self, space: &mut BindSpace) {
        for (slot, persona) in &self.personas {
            let addr = Addr::new(PREFIX_AGENTS, slot | SLOT_SUBDIVISION);
            let fp = persona.to_fingerprint();
            space.write_at(addr, fp);
            if let Some(node) = space.read_mut(addr) {
                node.label = Some(format!("persona:agent-{}", slot));
            }
        }
    }

    /// List all registered personas
    pub fn list(&self) -> &[(u8, Persona)] {
        &self.personas
    }

    /// Parse personas from YAML
    pub fn from_yaml(yaml: &str) -> Result<Vec<(String, Persona)>, String> {
        #[derive(Deserialize)]
        struct PersonaEntry {
            agent_id: String,
            persona: Persona,
        }
        #[derive(Deserialize)]
        struct PersonaList {
            personas: Vec<PersonaEntry>,
        }
        let list: PersonaList = serde_yml::from_str(yaml)
            .map_err(|e| format!("YAML parse error: {}", e))?;
        Ok(list.personas.into_iter().map(|e| (e.agent_id, e.persona)).collect())
    }
}

impl Default for PersonaRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_persona() -> Persona {
        Persona {
            volition: VolitionDTO {
                drive: "Discover patterns".to_string(),
                curiosity: 0.9,
                autonomy: 0.7,
                persistence: 0.85,
                risk_tolerance: 0.6,
                collaboration: 0.4,
                affinities: vec!["statistics".to_string(), "patterns".to_string()],
                aversions: vec!["formatting".to_string()],
            },
            traits: vec![
                PersonalityTrait { name: "openness".to_string(), value: 0.9, frozen: false },
                PersonalityTrait { name: "precision".to_string(), value: 0.85, frozen: true },
            ],
            communication: CommunicationStyle {
                formality: 0.6,
                verbosity: 0.4,
                directness: 0.8,
                technical_depth: 0.9,
                emotional_tone: 0.2,
            },
            preferred_styles: vec!["analytical".to_string(), "exploratory".to_string()],
            features: vec![
                FeatureAd {
                    name: "sci_query".to_string(),
                    proficiency: 0.95,
                    preference: 0.9,
                    cam_opcode: Some(0x060),
                },
            ],
        }
    }

    #[test]
    fn test_persona_fingerprint_deterministic() {
        let p = test_persona();
        let fp1 = p.to_fingerprint();
        let fp2 = p.to_fingerprint();
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_persona_fingerprint_changes_with_traits() {
        let mut p1 = test_persona();
        let fp1 = p1.to_fingerprint();
        p1.traits.push(PersonalityTrait {
            name: "conscientiousness".to_string(),
            value: 0.8,
            frozen: false,
        });
        let fp2 = p1.to_fingerprint();
        assert_ne!(fp1, fp2);
    }

    #[test]
    fn test_compatibility_self_is_high() {
        let p = test_persona();
        let score = p.compatibility(&p);
        // Self-compatibility should be very high (not exactly 1.0 due to XOR encoding)
        assert!(score > 0.5, "Self-compatibility was {}", score);
    }

    #[test]
    fn test_volition_alignment() {
        let p = test_persona();
        // Affinity match
        let score = p.volition_alignment("Analyze statistics data");
        assert!(score > 0.0, "Affinity score was {}", score);

        // Aversion match
        let score = p.volition_alignment("Do formatting only");
        assert!(score < 0.0, "Aversion score was {}", score);

        // No match
        let score = p.volition_alignment("Cook a meal");
        assert!((score - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_feature_lookup() {
        let p = test_persona();
        assert!(p.has_feature("sci_query", 0.9));
        assert!(!p.has_feature("sci_query", 0.99));
        assert!(!p.has_feature("nonexistent", 0.0));
        assert_eq!(p.feature_proficiency("sci_query"), Some(0.95));
    }

    #[test]
    fn test_persona_exchange() {
        let p = test_persona();
        let exchange = PersonaExchange::from_persona(&p, 3);
        assert_eq!(exchange.sender_slot, 3);
        assert_eq!(exchange.relevant_features.len(), 1);
        assert_eq!(exchange.volition_summary.drive, "Discover patterns");
    }

    #[test]
    fn test_persona_exchange_for_task() {
        let p = test_persona();
        let exchange = PersonaExchange::for_task(&p, 3, &["sci"]);
        assert_eq!(exchange.relevant_features.len(), 1);

        let exchange = PersonaExchange::for_task(&p, 3, &["unrelated"]);
        assert_eq!(exchange.relevant_features.len(), 0);
    }

    #[test]
    fn test_persona_registry() {
        let mut reg = PersonaRegistry::new();
        let p1 = test_persona();
        let p2 = Persona::default();

        reg.attach(0, p1.clone());
        reg.attach(1, p2);

        assert!(reg.get(0).is_some());
        assert!(reg.get(1).is_some());
        assert!(reg.get(2).is_none());

        // Feature search
        let with_sci = reg.find_by_feature("sci_query", 0.9);
        assert_eq!(with_sci.len(), 1);
        assert_eq!(with_sci[0].0, 0);
    }

    #[test]
    fn test_persona_yaml_roundtrip() {
        let p = test_persona();
        let yaml = p.to_yaml();
        assert!(yaml.contains("Discover patterns"));
        assert!(yaml.contains("openness"));
        assert!(yaml.contains("sci_query"));
        let parsed = Persona::from_yaml(&yaml).unwrap();
        assert_eq!(parsed.volition.drive, "Discover patterns");
        assert_eq!(parsed.traits.len(), 2);
    }

    #[test]
    fn test_primary_thinking_style() {
        let p = test_persona();
        assert_eq!(p.primary_thinking_style(), ThinkingStyle::Analytical);

        let p2 = Persona::default();
        assert_eq!(p2.primary_thinking_style(), ThinkingStyle::Deliberate);
    }
}
