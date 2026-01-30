//! ConceptExtractor — Extract reusable concepts from breakthroughs

use std::collections::HashMap;
use crate::core::Fingerprint;
use crate::nars::TruthValue;
use crate::learning::moment::Moment;

#[derive(Clone, Debug)]
pub struct ExtractedConcept {
    pub id: String,
    pub name: String,
    pub description: String,
    pub cam_fingerprint: u64,
    pub full_fingerprint: Fingerprint,
    pub abstraction_level: u8,
    pub source_moment_id: String,
    pub truth: TruthValue,
    pub relations: Vec<ConceptRelation>,
    pub tags: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct ConceptRelation {
    pub target_id: String,
    pub relation_type: RelationType,
    pub strength: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RelationType {
    Enables, Causes, Supports, Contradicts, Refines, Grounds, Abstracts, SimilarTo, PartOf, Requires,
}

impl RelationType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Enables => "ENABLES",
            Self::Causes => "CAUSES",
            Self::Supports => "SUPPORTS",
            Self::Contradicts => "CONTRADICTS",
            Self::Refines => "REFINES",
            Self::Grounds => "GROUNDS",
            Self::Abstracts => "ABSTRACTS",
            Self::SimilarTo => "SIMILAR_TO",
            Self::PartOf => "PART_OF",
            Self::Requires => "REQUIRES",
        }
    }
}

pub struct ConceptExtractor {
    concepts: HashMap<String, ExtractedConcept>,
    cam_index: HashMap<u64, String>,
    pub total_extractions: u64,
    pub duplicate_hits: u64,
}

impl ConceptExtractor {
    pub fn new() -> Self {
        Self {
            concepts: HashMap::new(),
            cam_index: HashMap::new(),
            total_extractions: 0,
            duplicate_hits: 0,
        }
    }
    
    pub fn extract(&mut self, moment: &Moment) -> Option<ExtractedConcept> {
        if !moment.is_breakthrough() { return None; }
        
        self.total_extractions += 1;
        let cam = self.content_addressable_fingerprint(&moment.content);
        
        if let Some(existing_id) = self.cam_index.get(&cam) {
            self.duplicate_hits += 1;
            return self.concepts.get(existing_id).cloned();
        }
        
        let concept = ExtractedConcept {
            id: uuid::Uuid::new_v4().to_string(),
            name: self.extract_name(&moment.content),
            description: moment.content.clone(),
            cam_fingerprint: cam,
            full_fingerprint: moment.fingerprint.clone(),
            abstraction_level: self.estimate_abstraction(&moment.content),
            source_moment_id: moment.id.clone(),
            truth: TruthValue::new(moment.qualia.satisfaction, 0.5 + moment.qualia.satisfaction * 0.4),
            relations: Vec::new(),
            tags: moment.tags.clone(),
        };
        
        self.cam_index.insert(cam, concept.id.clone());
        self.concepts.insert(concept.id.clone(), concept.clone());
        Some(concept)
    }
    
    fn content_addressable_fingerprint(&self, content: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        let normalized = content.to_lowercase().split_whitespace().collect::<Vec<_>>().join(" ");
        normalized.hash(&mut hasher);
        hasher.finish() & 0xFFFF_FFFF_FFFF
    }
    
    fn extract_name(&self, content: &str) -> String {
        let name = content.split('.').next().unwrap_or(content);
        if name.len() > 50 { format!("{}...", &name[..47]) } else { name.to_string() }
    }
    
    fn estimate_abstraction(&self, content: &str) -> u8 {
        let lower = content.to_lowercase();
        let abstract_kw = ["principle", "pattern", "generally", "always", "strategy", "architecture"];
        let concrete_kw = ["file", "function", "line", "error", "bug", "code", "method"];
        
        let abs = abstract_kw.iter().filter(|&k| lower.contains(k)).count() as i32;
        let con = concrete_kw.iter().filter(|&k| lower.contains(k)).count() as i32;
        (abs - con + 5).clamp(0, 10) as u8
    }
    
    pub fn get(&self, id: &str) -> Option<&ExtractedConcept> { self.concepts.get(id) }
    
    pub fn all(&self) -> impl Iterator<Item = &ExtractedConcept> { self.concepts.values() }
    
    pub fn to_cypher(&self) -> String {
        let mut cypher = String::new();
        for c in self.concepts.values() {
            cypher.push_str(&format!(
                "CREATE (c:Concept {{id: '{}', name: '{}', cam: {}, abstraction: {}}})\n",
                c.id, c.name.replace('\'', "\\'"), c.cam_fingerprint, c.abstraction_level
            ));
        }
        cypher
    }
}

impl Default for ConceptExtractor {
    fn default() -> Self { Self::new() }
}
