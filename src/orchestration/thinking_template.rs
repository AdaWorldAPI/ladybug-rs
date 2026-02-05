//! Thinking Templates — YAML thinking styles mapped to BindSpace 0x0D
//!
//! Bridges crewAI's YAML-based agent configuration with ladybug-rs's 12 thinking
//! styles and their FieldModulation parameters.
//!
//! # Prefix 0x0D Layout
//!
//! ```text
//! Slot 0x00-0x0B: 12 base ThinkingStyle fingerprints (one per style)
//! Slot 0x0C-0xFF: Custom variants / overrides (up to 244 templates)
//! ```
//!
//! Each slot stores a fingerprint encoding the style's FieldModulation parameters
//! so agents can find compatible thinking styles via HDR resonance search.

use serde::{Deserialize, Serialize};
use crate::cognitive::{ThinkingStyle, FieldModulation};
use crate::storage::bind_space::{
    Addr, BindSpace, FINGERPRINT_WORDS, PREFIX_THINKING,
};

/// Style override — allows YAML templates to fine-tune FieldModulation
/// Any field left as None inherits from the base ThinkingStyle.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StyleOverride {
    pub resonance_threshold: Option<f32>,
    pub fan_out: Option<usize>,
    pub depth_bias: Option<f32>,
    pub breadth_bias: Option<f32>,
    pub noise_tolerance: Option<f32>,
    pub speed_bias: Option<f32>,
    pub exploration: Option<f32>,
}

impl StyleOverride {
    /// Apply overrides on top of a base FieldModulation
    pub fn apply(&self, base: &FieldModulation) -> FieldModulation {
        FieldModulation {
            resonance_threshold: self.resonance_threshold.unwrap_or(base.resonance_threshold),
            fan_out: self.fan_out.unwrap_or(base.fan_out),
            depth_bias: self.depth_bias.unwrap_or(base.depth_bias),
            breadth_bias: self.breadth_bias.unwrap_or(base.breadth_bias),
            noise_tolerance: self.noise_tolerance.unwrap_or(base.noise_tolerance),
            speed_bias: self.speed_bias.unwrap_or(base.speed_bias),
            exploration: self.exploration.unwrap_or(base.exploration),
        }
    }
}

/// Thinking template — a named thinking style with optional overrides.
///
/// This is what gets defined in YAML:
/// ```yaml
/// templates:
///   - name: "deep_research"
///     base_style: "analytical"
///     overrides:
///       resonance_threshold: 0.90
///       depth_bias: 1.0
///       noise_tolerance: 0.02
///   - name: "brainstorm"
///     base_style: "creative"
///     overrides:
///       fan_out: 20
///       exploration: 0.95
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThinkingTemplate {
    pub name: String,
    pub base_style: String,
    #[serde(default)]
    pub overrides: StyleOverride,
    #[serde(default)]
    pub description: String,
    /// Assigned slot in 0x0D prefix (set by registry)
    #[serde(skip)]
    pub slot: Option<u8>,
}

impl ThinkingTemplate {
    /// Resolve the base ThinkingStyle enum variant
    pub fn resolve_base(&self) -> ThinkingStyle {
        match self.base_style.to_lowercase().as_str() {
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
        }
    }

    /// Compute the effective FieldModulation (base + overrides)
    pub fn effective_modulation(&self) -> FieldModulation {
        let base = self.resolve_base().field_modulation();
        self.overrides.apply(&base)
    }

    /// Encode FieldModulation as a fingerprint for BindSpace storage.
    ///
    /// The 7 modulation parameters are encoded into the first 7 words
    /// of the fingerprint as f32-expanded bit patterns, enabling
    /// Hamming-based similarity search between thinking styles.
    pub fn modulation_fingerprint(&self) -> [u64; FINGERPRINT_WORDS] {
        let m = self.effective_modulation();
        let mut fp = [0u64; FINGERPRINT_WORDS];

        // Encode each parameter into bit patterns across fingerprint words
        // Using thermometer coding: set bits proportional to parameter value
        let params = [
            m.resonance_threshold,
            m.fan_out as f32 / 20.0, // Normalize fan_out to 0..1 range
            m.depth_bias,
            m.breadth_bias,
            m.noise_tolerance,
            m.speed_bias,
            m.exploration,
        ];

        for (i, &param) in params.iter().enumerate() {
            let bits_to_set = (param * 64.0_f32).round() as u32;
            // Set `bits_to_set` bits in word group starting at i*22
            let base_word = i * 22;
            for w in 0..22 {
                if base_word + w >= FINGERPRINT_WORDS {
                    break;
                }
                let bits_for_word = bits_to_set.min(64);
                fp[base_word + w] = if bits_for_word >= 64 {
                    u64::MAX
                } else {
                    (1u64 << bits_for_word) - 1
                };
            }
        }
        fp
    }

    /// Get BindSpace address for this template
    pub fn addr(&self) -> Option<Addr> {
        self.slot.map(|s| Addr::new(PREFIX_THINKING, s))
    }
}

/// Registry of thinking templates
pub struct ThinkingTemplateRegistry {
    templates: Vec<ThinkingTemplate>,
    next_custom_slot: u8,
}

impl ThinkingTemplateRegistry {
    pub fn new() -> Self {
        Self {
            templates: Vec::new(),
            // First 12 slots (0x00-0x0B) reserved for base styles
            next_custom_slot: 0x0C,
        }
    }

    /// Seed the 12 base ThinkingStyle templates into slots 0x00-0x0B
    pub fn seed_base_styles(&mut self) {
        for (i, style) in ThinkingStyle::ALL.iter().enumerate() {
            let template = ThinkingTemplate {
                name: format!("{:?}", style).to_lowercase(),
                base_style: format!("{:?}", style).to_lowercase(),
                overrides: StyleOverride::default(),
                description: format!("Base {} thinking style", style),
                slot: Some(i as u8),
            };
            self.templates.push(template);
        }
    }

    /// Register a custom template in the next available slot
    pub fn register(&mut self, mut template: ThinkingTemplate) -> Result<Addr, String> {
        if self.next_custom_slot == 0xFF {
            return Err("Template registry full (244 custom max)".to_string());
        }

        template.slot = Some(self.next_custom_slot);
        let addr = Addr::new(PREFIX_THINKING, self.next_custom_slot);
        self.templates.push(template);
        self.next_custom_slot += 1;
        Ok(addr)
    }

    /// Bind all templates into BindSpace
    pub fn bind_all(&self, space: &mut BindSpace) {
        for template in &self.templates {
            if let Some(slot) = template.slot {
                let addr = Addr::new(PREFIX_THINKING, slot);
                let fp = template.modulation_fingerprint();
                space.write_at(addr, fp);
                if let Some(node) = space.read_mut(addr) {
                    node.label = Some(format!("style:{}", template.name));
                }
            }
        }
    }

    /// Parse templates from YAML
    pub fn from_yaml(yaml: &str) -> Result<Vec<ThinkingTemplate>, String> {
        #[derive(Deserialize)]
        struct TemplateList {
            templates: Vec<ThinkingTemplate>,
        }
        let list: TemplateList = serde_yml::from_str(yaml)
            .map_err(|e| format!("YAML parse error: {}", e))?;
        Ok(list.templates)
    }

    /// Find template by name
    pub fn get(&self, name: &str) -> Option<&ThinkingTemplate> {
        self.templates.iter().find(|t| t.name == name)
    }

    /// Get template by slot
    pub fn get_by_slot(&self, slot: u8) -> Option<&ThinkingTemplate> {
        self.templates.iter().find(|t| t.slot == Some(slot))
    }

    /// List all registered templates
    pub fn list(&self) -> &[ThinkingTemplate] {
        &self.templates
    }
}

impl Default for ThinkingTemplateRegistry {
    fn default() -> Self {
        let mut reg = Self::new();
        reg.seed_base_styles();
        reg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_style_override_apply() {
        let base = ThinkingStyle::Analytical.field_modulation();
        let over = StyleOverride {
            resonance_threshold: Some(0.95),
            fan_out: Some(1),
            ..Default::default()
        };
        let effective = over.apply(&base);
        assert!((effective.resonance_threshold - 0.95).abs() < f32::EPSILON);
        assert_eq!(effective.fan_out, 1);
        // Unchanged fields inherit from base
        assert!((effective.depth_bias - base.depth_bias).abs() < f32::EPSILON);
    }

    #[test]
    fn test_yaml_template_parse() {
        let yaml = r#"
templates:
  - name: "deep_research"
    base_style: "analytical"
    description: "Deep statistical research"
    overrides:
      resonance_threshold: 0.95
      depth_bias: 1.0
  - name: "brainstorm"
    base_style: "creative"
    overrides:
      fan_out: 20
      exploration: 0.95
"#;
        let templates = ThinkingTemplateRegistry::from_yaml(yaml).unwrap();
        assert_eq!(templates.len(), 2);
        assert_eq!(templates[0].name, "deep_research");
        assert_eq!(templates[0].resolve_base(), ThinkingStyle::Analytical);
        assert_eq!(templates[1].resolve_base(), ThinkingStyle::Creative);
    }

    #[test]
    fn test_base_styles_seeded() {
        let reg = ThinkingTemplateRegistry::default();
        assert_eq!(reg.list().len(), 12);
        // First slot should be the first style
        let first = reg.get_by_slot(0).unwrap();
        assert_eq!(first.resolve_base(), ThinkingStyle::Analytical);
    }

    #[test]
    fn test_modulation_fingerprint_deterministic() {
        let t = ThinkingTemplate {
            name: "test".to_string(),
            base_style: "creative".to_string(),
            overrides: StyleOverride::default(),
            description: String::new(),
            slot: None,
        };
        let fp1 = t.modulation_fingerprint();
        let fp2 = t.modulation_fingerprint();
        assert_eq!(fp1, fp2);
    }
}
