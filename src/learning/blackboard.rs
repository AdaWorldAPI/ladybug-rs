//! Blackboard — Persistent session state for agent handoffs

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::cognitive::GateState;
use crate::learning::session::{SessionState, IceCakedDecision};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IceCakedLayer {
    pub layer_id: u32,
    pub decision_id: String,
    pub content: String,
    pub rationale: String,
    pub gate_state: String,
    pub ice_caked_at_cycle: u64,
}

impl From<&IceCakedDecision> for IceCakedLayer {
    fn from(d: &IceCakedDecision) -> Self {
        let gate_state = match d.gate_state {
            GateState::Flow => "FLOW",
            GateState::Hold => "HOLD",
            GateState::Block => "BLOCK",
        };
        Self {
            layer_id: 0,
            decision_id: d.moment_id.clone(),
            content: d.content.clone(),
            rationale: d.rationale.clone(),
            gate_state: gate_state.to_string(),
            ice_caked_at_cycle: d.ice_caked_at_cycle,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Decision {
    pub id: String,
    pub task: String,
    pub choice: String,
    pub rationale: String,
    pub gate_state: String,
    pub ice_caked: bool,
    pub cycle: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaskState {
    pub id: String,
    pub description: String,
    pub phase: String,
    pub progress: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConsciousnessState {
    pub thinking_style: String,
    pub coherence: f32,
    pub dominant_layer: String,
    pub emergence: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Blackboard {
    pub session_id: String,
    pub current_task: TaskState,
    pub consciousness: ConsciousnessState,
    pub decisions: Vec<Decision>,
    pub ice_cake_layers: Vec<IceCakedLayer>,
    pub files_modified: Vec<String>,
    pub blockers: Vec<String>,
    pub next_steps: Vec<String>,
    pub resonance_captures: u64,
    pub concepts_extracted: u64,
    pub cycle: u64,
}

impl Blackboard {
    pub fn new(session_id: &str, task_id: &str, task_description: &str) -> Self {
        Self {
            session_id: session_id.to_string(),
            current_task: TaskState {
                id: task_id.to_string(),
                description: task_description.to_string(),
                phase: "Initialize".to_string(),
                progress: 0.0,
            },
            consciousness: ConsciousnessState {
                thinking_style: "analytical".to_string(),
                coherence: 0.0,
                dominant_layer: "L1".to_string(),
                emergence: 0.0,
            },
            decisions: Vec::new(),
            ice_cake_layers: Vec::new(),
            files_modified: Vec::new(),
            blockers: Vec::new(),
            next_steps: Vec::new(),
            resonance_captures: 0,
            concepts_extracted: 0,
            cycle: 0,
        }
    }
    
    pub fn update_from_session(&mut self, state: &SessionState) {
        self.current_task.phase = format!("{:?}", state.phase);
        self.current_task.progress = state.progress;
        self.consciousness.coherence = state.coherence;
        self.resonance_captures = state.moment_count as u64;
        self.cycle = state.cycle;
    }
    
    pub fn record_decision(&mut self, task: &str, choice: &str, rationale: &str, gate: GateState) {
        let decision = Decision {
            id: uuid::Uuid::new_v4().to_string(),
            task: task.to_string(),
            choice: choice.to_string(),
            rationale: rationale.to_string(),
            gate_state: format!("{:?}", gate),
            ice_caked: false,
            cycle: self.cycle,
        };
        self.decisions.push(decision);
    }
    
    pub fn add_ice_cake(&mut self, decision: &IceCakedDecision) {
        let mut layer = IceCakedLayer::from(decision);
        layer.layer_id = self.ice_cake_layers.len() as u32 + 1;
        self.ice_cake_layers.push(layer);
    }
    
    pub fn record_file_modified(&mut self, path: &str) {
        if !self.files_modified.contains(&path.to_string()) {
            self.files_modified.push(path.to_string());
        }
    }
    
    pub fn add_next_step(&mut self, step: &str) {
        self.next_steps.push(step.to_string());
    }
    
    pub fn to_yaml(&self) -> String {
        serde_yaml::to_string(self).unwrap_or_default()
    }
    
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
    
    pub fn handover_summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("# Session Handover: {}\n\n", self.session_id));
        s.push_str("## Current Task\n");
        s.push_str(&format!("- **ID**: {}\n", self.current_task.id));
        s.push_str(&format!("- **Phase**: {}\n", self.current_task.phase));
        s.push_str(&format!("- **Progress**: {:.0}%\n\n", self.current_task.progress * 100.0));
        
        if !self.ice_cake_layers.is_empty() {
            s.push_str("## Ice-Caked (Frozen Commitments) ❄️\n");
            for layer in &self.ice_cake_layers {
                s.push_str(&format!("{}. {}\n", layer.layer_id, layer.content));
                s.push_str(&format!("   Rationale: {}\n", layer.rationale));
            }
            s.push_str("\n");
        }
        
        if !self.next_steps.is_empty() {
            s.push_str("## Next Steps\n");
            for (i, step) in self.next_steps.iter().enumerate() {
                s.push_str(&format!("{}. {}\n", i + 1, step));
            }
        }
        
        s.push_str(&format!("\n## Stats\n- Resonance Captures: {}\n- Concepts Extracted: {}\n", 
            self.resonance_captures, self.concepts_extracted));
        s
    }
}
