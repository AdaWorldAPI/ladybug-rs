//! LearningSession — 6-phase learning loop lifecycle

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::cognitive::{GateState, ThinkingStyle, evaluate_gate};
use crate::core::Fingerprint;
use crate::learning::moment::{Moment, MomentBuilder, Qualia};

#[derive(Clone, Debug, PartialEq)]
pub enum SessionPhase {
    Initialize,
    Encounter,
    Struggle,
    Breakthrough,
    Consolidate,
    Apply,
    MetaLearn,
    Complete,
}

impl SessionPhase {
    pub fn next(&self) -> Option<SessionPhase> {
        match self {
            Self::Initialize => Some(Self::Encounter),
            Self::Encounter => Some(Self::Struggle),
            Self::Struggle => Some(Self::Breakthrough),
            Self::Breakthrough => Some(Self::Consolidate),
            Self::Consolidate => Some(Self::Apply),
            Self::Apply => Some(Self::MetaLearn),
            Self::MetaLearn => Some(Self::Complete),
            Self::Complete => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SessionState {
    pub session_id: String,
    pub task_id: String,
    pub phase: SessionPhase,
    pub progress: f32,
    pub thinking_style: ThinkingStyle,
    pub coherence: f32,
    pub ice_cake_layers: u32,
    pub moment_count: usize,
    pub breakthrough_count: usize,
    pub cycle: u64,
}

#[derive(Clone, Debug)]
pub struct IceCakedDecision {
    pub moment_id: String,
    pub content: String,
    pub rationale: String,
    pub gate_state: GateState,
    pub ice_caked_at_cycle: u64,
}

pub struct LearningSession {
    pub id: String,
    pub task_id: String,
    pub phase: SessionPhase,
    pub progress: f32,
    pub moments: Vec<Moment>,
    moment_index: HashMap<String, usize>,
    pub ice_caked: Vec<IceCakedDecision>,
    pub cycle: u64,
    pub started_at: Instant,
    pub last_activity: Instant,
}

impl LearningSession {
    pub fn new(task_id: &str) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            task_id: task_id.to_string(),
            phase: SessionPhase::Initialize,
            progress: 0.0,
            moments: Vec::new(),
            moment_index: HashMap::new(),
            ice_caked: Vec::new(),
            cycle: 0,
            started_at: Instant::now(),
            last_activity: Instant::now(),
        }
    }

    pub fn state(&self) -> SessionState {
        SessionState {
            session_id: self.id.clone(),
            task_id: self.task_id.clone(),
            phase: self.phase.clone(),
            progress: self.progress,
            thinking_style: ThinkingStyle::default(),
            coherence: 0.5,
            ice_cake_layers: self.ice_caked.len() as u32,
            moment_count: self.moments.len(),
            breakthrough_count: self.moments.iter().filter(|m| m.is_breakthrough()).count(),
            cycle: self.cycle,
        }
    }

    pub fn encounter(&mut self, content: &str) -> &Moment {
        self.transition_to(SessionPhase::Encounter);
        let moment = MomentBuilder::new(&self.id, content)
            .encounter()
            .qualia(0.5, 0.2, 0.5)
            .build();
        self.add_moment(moment)
    }

    pub fn struggle(&mut self, content: &str, effort: f32, confusion: f32) -> &Moment {
        self.transition_to(SessionPhase::Struggle);
        let mut qualia = Qualia::from_metrics(0.3, effort, 0.3);
        qualia.confusion = confusion;
        let moment = MomentBuilder::new(&self.id, content)
            .struggle()
            .build()
            .with_qualia(qualia);
        self.add_moment(moment)
    }

    pub fn fail(&mut self, content: &str, lesson: &str) -> &Moment {
        let mut qualia = Qualia::from_metrics(0.4, 0.8, 0.2);
        qualia.surprise = 0.6;
        let moment = MomentBuilder::new(&self.id, &format!("{} | Lesson: {}", content, lesson))
            .failure()
            .build()
            .with_qualia(qualia);
        self.add_moment(moment)
    }

    pub fn breakthrough(&mut self, content: &str, satisfaction: f32) -> &Moment {
        self.transition_to(SessionPhase::Breakthrough);
        let qualia = Qualia::from_metrics(0.8, 0.6, satisfaction);
        let moment = MomentBuilder::new(&self.id, content)
            .breakthrough()
            .build()
            .with_qualia(qualia);
        self.add_moment(moment)
    }

    pub fn ice_cake(&mut self, moment_id: &str, rationale: &str) -> Option<&IceCakedDecision> {
        self.transition_to(SessionPhase::Consolidate);
        let moment = self.get_moment(moment_id)?;
        let scores = vec![moment.qualia.satisfaction, 1.0 - moment.qualia.confusion];
        let decision = evaluate_gate(&scores, false);

        let ice_caked = IceCakedDecision {
            moment_id: moment_id.to_string(),
            content: moment.content.clone(),
            rationale: rationale.to_string(),
            gate_state: decision.state,
            ice_caked_at_cycle: self.cycle,
        };

        self.ice_caked.push(ice_caked);
        self.ice_caked.last()
    }

    pub fn apply(&mut self, content: &str, success: bool) -> &Moment {
        self.transition_to(SessionPhase::Apply);
        let satisfaction = if success { 0.9 } else { 0.4 };
        let qualia = Qualia::from_metrics(0.2, 0.3, satisfaction);
        let moment = MomentBuilder::new(&self.id, content)
            .build()
            .with_qualia(qualia);
        self.add_moment(moment)
    }

    pub fn meta_reflect(&mut self, reflection: &str) -> &Moment {
        self.transition_to(SessionPhase::MetaLearn);
        let breakthrough_count = self.moments.iter().filter(|m| m.is_breakthrough()).count();
        let novelty = if breakthrough_count > 0 { 0.7 } else { 0.3 };
        let qualia = Qualia::from_metrics(novelty, 0.4, 0.8);
        let moment = MomentBuilder::new(&self.id, reflection)
            .build()
            .with_qualia(qualia);
        self.add_moment(moment)
    }

    fn add_moment(&mut self, moment: Moment) -> &Moment {
        let idx = self.moments.len();
        self.moment_index.insert(moment.id.clone(), idx);
        self.cycle += 1;
        self.moments.push(moment);
        self.last_activity = Instant::now();
        &self.moments[idx]
    }

    pub fn get_moment(&self, id: &str) -> Option<&Moment> {
        self.moment_index.get(id).map(|&idx| &self.moments[idx])
    }

    fn transition_to(&mut self, new_phase: SessionPhase) {
        if self.phase != new_phase {
            self.phase = new_phase;
            self.progress = 0.0;
        }
    }

    pub fn find_similar(&self, query: &Fingerprint, threshold: f32) -> Vec<(&Moment, f32)> {
        let mut results: Vec<_> = self
            .moments
            .iter()
            .map(|m| (m, query.similarity(&m.resonance_vector)))
            .filter(|(_, sim)| *sim >= threshold)
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    pub fn breakthroughs(&self) -> Vec<&Moment> {
        self.moments
            .iter()
            .filter(|m| m.is_breakthrough())
            .collect()
    }

    pub fn duration(&self) -> Duration {
        self.started_at.elapsed()
    }

    pub fn complete(&mut self) {
        self.phase = SessionPhase::Complete;
        self.progress = 1.0;
    }
}
