//! Agent Card — crewAI agent definition mapped to BindSpace 0x0C
//!
//! Each agent card occupies a slot in prefix 0x0C.
//! Slots 0x00-0x7F: agent definitions (128 agents max)
//! Slots 0x80-0xFF: capability fingerprints (1:1 with agent slot)
//!
//! Compatible with crewAI's agents.yaml format.

use serde::{Deserialize, Serialize};
use crate::cognitive::ThinkingStyle;
use crate::storage::bind_space::{
    Addr, BindSpace, FINGERPRINT_WORDS,
    PREFIX_AGENTS, SLOT_SUBDIVISION,
};

/// Agent role mirrors crewAI's role field
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentRole {
    pub name: String,
    pub description: String,
}

/// Agent goal — what the agent is trying to achieve
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentGoal {
    pub objective: String,
    pub success_criteria: Vec<String>,
    pub constraints: Vec<String>,
}

/// Agent capability — tool or skill the agent can use
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentCapability {
    pub name: String,
    pub description: String,
    /// CAM opcode this capability maps to (0x000-0xFFF)
    pub cam_opcode: Option<u16>,
    /// Whether this capability requires sci/v1 validation
    pub requires_validation: bool,
}

/// Agent card — full agent definition compatible with crewAI agents.yaml
///
/// Maps to BindSpace prefix 0x0C.
/// The agent's fingerprint encodes its role + goal + backstory
/// for similarity-based agent selection via HDR cascade.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentCard {
    /// Unique agent identifier (maps to slot in 0x0C:XX)
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Agent role (crewAI compatible)
    pub role: AgentRole,
    /// Agent goal
    pub goal: AgentGoal,
    /// Backstory / system prompt context
    pub backstory: String,
    /// Thinking style for this agent
    pub thinking_style: String,
    /// Available capabilities/tools
    pub capabilities: Vec<AgentCapability>,
    /// Whether the agent can delegate to others
    pub allow_delegation: bool,
    /// Memory enabled (maps to blackboard at 0x0E)
    pub memory: bool,
    /// Verbose output
    pub verbose: bool,
    /// Maximum iterations before stopping
    pub max_iter: u32,
    /// BindSpace slot assigned to this agent (0x00-0x7F)
    pub slot: Option<u8>,
}

impl AgentCard {
    /// Parse thinking_style field to ThinkingStyle enum
    pub fn resolve_thinking_style(&self) -> ThinkingStyle {
        match self.thinking_style.to_lowercase().as_str() {
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

    /// Get the BindSpace address for this agent's card
    pub fn card_addr(&self) -> Option<Addr> {
        self.slot.map(|s| Addr::new(PREFIX_AGENTS, s))
    }

    /// Get the BindSpace address for this agent's capability fingerprint
    pub fn capability_addr(&self) -> Option<Addr> {
        self.slot.map(|s| Addr::new(PREFIX_AGENTS, s | SLOT_SUBDIVISION))
    }

    /// Generate a fingerprint from the agent's identity
    /// (role + goal + backstory → SHA256 → expanded to 10K bits)
    pub fn identity_fingerprint(&self) -> [u64; FINGERPRINT_WORDS] {
        use sha2::{Sha256, Digest};

        let identity = format!(
            "{}:{}:{}:{}",
            self.role.name, self.role.description,
            self.goal.objective, self.backstory
        );

        let mut hasher = Sha256::new();
        hasher.update(identity.as_bytes());
        let hash = hasher.finalize();

        let mut fp = [0u64; FINGERPRINT_WORDS];
        for (i, word) in fp.iter_mut().enumerate() {
            let mut h = Sha256::new();
            h.update(&hash);
            h.update(&(i as u32).to_le_bytes());
            let block = h.finalize();
            *word = u64::from_le_bytes(block[..8].try_into().unwrap());
        }
        fp
    }
}

impl Default for AgentCard {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: String::new(),
            role: AgentRole {
                name: "assistant".to_string(),
                description: "General-purpose assistant".to_string(),
            },
            goal: AgentGoal {
                objective: String::new(),
                success_criteria: Vec::new(),
                constraints: Vec::new(),
            },
            backstory: String::new(),
            thinking_style: "deliberate".to_string(),
            capabilities: Vec::new(),
            allow_delegation: false,
            memory: true,
            verbose: false,
            max_iter: 25,
            slot: None,
        }
    }
}

/// Agent registry — manages all agents in prefix 0x0C
pub struct AgentRegistry {
    agents: Vec<AgentCard>,
    next_slot: u8,
}

impl AgentRegistry {
    pub fn new() -> Self {
        Self {
            agents: Vec::new(),
            next_slot: 0,
        }
    }

    /// Register an agent, assigning it the next available slot
    pub fn register(&mut self, mut card: AgentCard) -> Result<Addr, String> {
        if self.next_slot >= SLOT_SUBDIVISION {
            return Err("Agent registry full (128 max)".to_string());
        }

        card.slot = Some(self.next_slot);
        let addr = Addr::new(PREFIX_AGENTS, self.next_slot);
        self.agents.push(card);
        self.next_slot += 1;
        Ok(addr)
    }

    /// Bind all registered agents into BindSpace
    pub fn bind_all(&self, space: &mut BindSpace) {
        for card in &self.agents {
            if let Some(slot) = card.slot {
                let addr = Addr::new(PREFIX_AGENTS, slot);
                let fp = card.identity_fingerprint();
                space.write_at(addr, fp);
                if let Some(node) = space.read_mut(addr) {
                    node.label = Some(format!("agent:{}", card.id));
                }
            }
        }
    }

    /// Parse agents from YAML (crewAI agents.yaml format)
    pub fn from_yaml(yaml: &str) -> Result<Vec<AgentCard>, String> {
        serde_yml::from_str(yaml).map_err(|e| format!("YAML parse error: {}", e))
    }

    /// Get agent by slot
    pub fn get(&self, slot: u8) -> Option<&AgentCard> {
        self.agents.iter().find(|a| a.slot == Some(slot))
    }

    /// Get agent by id
    pub fn get_by_id(&self, id: &str) -> Option<&AgentCard> {
        self.agents.iter().find(|a| a.id == id)
    }

    /// List all registered agents
    pub fn list(&self) -> &[AgentCard] {
        &self.agents
    }

    /// Count of registered agents
    pub fn count(&self) -> usize {
        self.agents.len()
    }
}

impl Default for AgentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_card_yaml_roundtrip() {
        let card = AgentCard {
            id: "researcher".to_string(),
            name: "Research Agent".to_string(),
            role: AgentRole {
                name: "researcher".to_string(),
                description: "Performs deep research on scientific topics".to_string(),
            },
            goal: AgentGoal {
                objective: "Find and validate scientific claims".to_string(),
                success_criteria: vec!["p < 0.05".to_string()],
                constraints: vec!["Use sci/v1 validation".to_string()],
            },
            backstory: "Expert in statistical analysis".to_string(),
            thinking_style: "analytical".to_string(),
            capabilities: vec![AgentCapability {
                name: "sci_query".to_string(),
                description: "Query sci/v1 endpoints".to_string(),
                cam_opcode: Some(0x060),
                requires_validation: true,
            }],
            allow_delegation: true,
            memory: true,
            verbose: false,
            max_iter: 25,
            slot: None,
        };

        let yaml = serde_yml::to_string(&card).unwrap();
        let parsed: AgentCard = serde_yml::from_str(&yaml).unwrap();
        assert_eq!(parsed.id, "researcher");
        assert_eq!(parsed.resolve_thinking_style(), ThinkingStyle::Analytical);
    }

    #[test]
    fn test_agent_registry() {
        let mut registry = AgentRegistry::new();

        let card = AgentCard {
            id: "agent-0".to_string(),
            ..Default::default()
        };

        let addr = registry.register(card).unwrap();
        assert_eq!(addr.prefix(), PREFIX_AGENTS);
        assert_eq!(addr.slot(), 0);
        assert_eq!(registry.count(), 1);
    }

    #[test]
    fn test_identity_fingerprint_deterministic() {
        let card = AgentCard {
            id: "test".to_string(),
            role: AgentRole {
                name: "tester".to_string(),
                description: "Tests things".to_string(),
            },
            ..Default::default()
        };

        let fp1 = card.identity_fingerprint();
        let fp2 = card.identity_fingerprint();
        assert_eq!(fp1, fp2);
    }
}
