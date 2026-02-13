//! Moment — Atomic unit of learning capture

use crate::cognitive::ThinkingStyle;
use crate::core::Fingerprint;
use crate::nars::TruthValue;
use std::time::{SystemTime, UNIX_EPOCH};

/// Qualia — The felt quality of a learning moment
#[derive(Clone, Debug, Default)]
pub struct Qualia {
    pub novelty: f32,
    pub effort: f32,
    pub satisfaction: f32,
    pub confusion: f32,
    pub surprise: f32,
    pub qidx: u8,
}

impl Qualia {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_metrics(novelty: f32, effort: f32, satisfaction: f32) -> Self {
        let mut q = Self {
            novelty: novelty.clamp(0.0, 1.0),
            effort: effort.clamp(0.0, 1.0),
            satisfaction: satisfaction.clamp(0.0, 1.0),
            confusion: 0.0,
            surprise: 0.0,
            qidx: 0,
        };
        q.compute_qidx();
        q
    }

    pub fn compute_qidx(&mut self) {
        let breakthrough = (self.novelty * self.satisfaction * 15.0) as u8;
        let clean_effort = (self.effort * (1.0 - self.confusion) * 15.0) as u8;
        self.qidx = (breakthrough << 4) | clean_effort;
    }

    pub fn is_breakthrough(&self) -> bool {
        self.novelty > 0.6 && self.satisfaction > 0.7
    }

    pub fn is_struggle(&self) -> bool {
        self.effort > 0.5 && self.confusion > 0.4
    }

    pub fn weight_fingerprint(&self, fp: &Fingerprint) -> Fingerprint {
        let qualia_sig = Fingerprint::from_content(&format!(
            "qualia:{}:{}:{}:{}:{}",
            (self.novelty * 100.0) as u32,
            (self.effort * 100.0) as u32,
            (self.satisfaction * 100.0) as u32,
            (self.confusion * 100.0) as u32,
            (self.surprise * 100.0) as u32,
        ));
        fp.bind(&qualia_sig)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum MomentType {
    Encounter,
    Struggle,
    Breakthrough,
    Failure,
    Application,
    MetaReflection,
}

#[derive(Clone, Debug)]
pub struct Moment {
    pub id: String,
    pub session_id: String,
    pub timestamp_ms: u64,
    pub moment_type: MomentType,
    pub content: String,
    pub fingerprint: Fingerprint,
    pub resonance_vector: Fingerprint,
    pub qualia: Qualia,
    pub thinking_style: ThinkingStyle,
    pub truth: TruthValue,
    pub tags: Vec<String>,
    pub parent_id: Option<String>,
    pub related_files: Vec<String>,
}

impl Moment {
    pub fn new(session_id: &str, content: &str, moment_type: MomentType) -> Self {
        let fingerprint = Fingerprint::from_content(content);
        let qualia = Qualia::default();
        let resonance_vector = qualia.weight_fingerprint(&fingerprint);

        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: session_id.to_string(),
            timestamp_ms,
            moment_type,
            content: content.to_string(),
            fingerprint,
            resonance_vector,
            qualia,
            thinking_style: ThinkingStyle::default(),
            truth: TruthValue::unknown(),
            tags: Vec::new(),
            parent_id: None,
            related_files: Vec::new(),
        }
    }

    pub fn with_qualia(mut self, qualia: Qualia) -> Self {
        self.qualia = qualia;
        self.resonance_vector = self.qualia.weight_fingerprint(&self.fingerprint);
        self
    }

    pub fn with_style(mut self, style: ThinkingStyle) -> Self {
        self.thinking_style = style;
        self
    }

    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self
    }

    pub fn is_breakthrough(&self) -> bool {
        self.moment_type == MomentType::Breakthrough || self.qualia.is_breakthrough()
    }

    pub fn resonance(&self, other: &Moment) -> f32 {
        self.resonance_vector.similarity(&other.resonance_vector)
    }
}

pub struct MomentBuilder {
    session_id: String,
    content: String,
    moment_type: MomentType,
    qualia: Option<Qualia>,
    style: Option<ThinkingStyle>,
    tags: Vec<String>,
    parent_id: Option<String>,
    files: Vec<String>,
}

impl MomentBuilder {
    pub fn new(session_id: &str, content: &str) -> Self {
        Self {
            session_id: session_id.to_string(),
            content: content.to_string(),
            moment_type: MomentType::Encounter,
            qualia: None,
            style: None,
            tags: Vec::new(),
            parent_id: None,
            files: Vec::new(),
        }
    }

    pub fn encounter(mut self) -> Self {
        self.moment_type = MomentType::Encounter;
        self
    }
    pub fn struggle(mut self) -> Self {
        self.moment_type = MomentType::Struggle;
        self
    }
    pub fn breakthrough(mut self) -> Self {
        self.moment_type = MomentType::Breakthrough;
        self
    }
    pub fn failure(mut self) -> Self {
        self.moment_type = MomentType::Failure;
        self
    }

    pub fn qualia(mut self, novelty: f32, effort: f32, satisfaction: f32) -> Self {
        self.qualia = Some(Qualia::from_metrics(novelty, effort, satisfaction));
        self
    }

    pub fn tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self
    }

    pub fn build(self) -> Moment {
        let mut moment = Moment::new(&self.session_id, &self.content, self.moment_type);
        if let Some(q) = self.qualia {
            moment = moment.with_qualia(q);
        }
        if let Some(s) = self.style {
            moment = moment.with_style(s);
        }
        moment.tags = self.tags;
        moment.parent_id = self.parent_id;
        moment.related_files = self.files;
        moment
    }
}
