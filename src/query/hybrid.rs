//! Hybrid Query Engine - Unified Cypher + Vector + Temporal
//!
//! Enables queries like:
//! ```text
//! HYBRID {
//!   MATCH (a:Person)-[:KNOWS*1..3]->(b:Person)
//!   WHERE SIMILAR(a.embedding, $query, 0.8)
//!   AT VERSION 42
//!   RETURN b
//! }
//! ```
//!
//! Execution Pipeline:
//! 1. Vector search → narrow candidate set
//! 2. Cypher filter → apply graph constraints
//! 3. Temporal resolve → select correct version

use std::time::{Duration, Instant};

use crate::storage::{
    BindSpace, Addr, Substrate, SubstrateConfig, FINGERPRINT_WORDS,
};
use crate::query::cypher::{CypherParser, CypherQuery, PatternElement};

// =============================================================================
// HYBRID QUERY TYPES
// =============================================================================

/// Causal query mode (Pearl's 3 rungs)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalMode {
    /// Rung 1: What correlates?
    See,
    /// Rung 2: What happens if we intervene?
    Do,
    /// Rung 3: What would have happened?
    Imagine,
}

/// Temporal constraint for queries
#[derive(Debug, Clone)]
pub enum TemporalConstraint {
    /// Query at specific version
    AtVersion(u64),
    /// Query in version range
    VersionRange { from: u64, to: u64 },
    /// Query at specific timestamp
    AtTime(u64),
    /// Latest version (default)
    Latest,
}

impl Default for TemporalConstraint {
    fn default() -> Self {
        TemporalConstraint::Latest
    }
}

/// Vector similarity constraint
#[derive(Debug, Clone)]
pub struct VectorConstraint {
    /// Query fingerprint
    pub query: [u64; FINGERPRINT_WORDS],
    /// Similarity threshold (0.0 - 1.0)
    pub threshold: f32,
    /// Maximum Hamming distance (alternative to threshold)
    pub max_distance: Option<u32>,
    /// Number of results to retrieve
    pub k: usize,
}

impl VectorConstraint {
    pub fn new(query: [u64; FINGERPRINT_WORDS], threshold: f32, k: usize) -> Self {
        Self {
            query,
            threshold,
            max_distance: None,
            k,
        }
    }

    pub fn with_max_distance(query: [u64; FINGERPRINT_WORDS], max_dist: u32, k: usize) -> Self {
        Self {
            query,
            threshold: 1.0 - (max_dist as f32 / (FINGERPRINT_WORDS * 64) as f32),
            max_distance: Some(max_dist),
            k,
        }
    }
}

/// Qualia filter (simplified - qidx range)
#[derive(Debug, Clone, Default)]
pub struct QualiaFilter {
    /// Minimum qidx value
    pub min_qidx: Option<u8>,
    /// Maximum qidx value
    pub max_qidx: Option<u8>,
}

impl QualiaFilter {
    /// Check if a qidx passes the filter
    pub fn matches(&self, qidx: u8) -> bool {
        if let Some(min) = self.min_qidx {
            if qidx < min {
                return false;
            }
        }
        if let Some(max) = self.max_qidx {
            if qidx > max {
                return false;
            }
        }
        true
    }
}

/// NARS truth value filter
#[derive(Debug, Clone)]
pub struct TruthFilter {
    /// Minimum frequency (0.0 - 1.0)
    pub min_frequency: f32,
    /// Minimum confidence (0.0 - 1.0)
    pub min_confidence: f32,
}

impl Default for TruthFilter {
    fn default() -> Self {
        Self {
            min_frequency: 0.5,
            min_confidence: 0.5,
        }
    }
}

/// Graph constraint
#[derive(Debug, Clone)]
pub struct GraphConstraint {
    /// Cypher pattern to match
    pub pattern: String,
    /// Parsed query (cached)
    pub parsed: Option<CypherQuery>,
}

impl GraphConstraint {
    pub fn new(pattern: &str) -> Self {
        let parsed = CypherParser::parse(pattern).ok();
        Self {
            pattern: pattern.to_string(),
            parsed,
        }
    }
}

/// Causal constraint
#[derive(Debug, Clone)]
pub struct CausalConstraint {
    /// Which rung of Pearl's ladder
    pub mode: CausalMode,
    /// Action/intervention to check
    pub action: Option<String>,
    /// Outcome to verify
    pub outcome: Option<String>,
}

// =============================================================================
// HYBRID QUERY
// =============================================================================

/// Unified hybrid query combining all constraint types
#[derive(Debug, Clone, Default)]
pub struct HybridQuery {
    /// Vector similarity constraint
    pub vector: Option<VectorConstraint>,
    /// Graph pattern constraint
    pub graph: Option<GraphConstraint>,
    /// Causal reasoning constraint
    pub causal: Option<CausalConstraint>,
    /// Temporal versioning constraint
    pub temporal: TemporalConstraint,
    /// Qualia filter
    pub qualia: Option<QualiaFilter>,
    /// Label filter
    pub label_filter: Option<String>,
    /// Maximum results
    pub limit: usize,
    /// Offset for pagination
    pub offset: usize,
}

impl HybridQuery {
    pub fn new() -> Self {
        Self {
            limit: 100,
            offset: 0,
            ..Default::default()
        }
    }

    /// Add vector similarity constraint
    pub fn with_vector(mut self, query: [u64; FINGERPRINT_WORDS], threshold: f32, k: usize) -> Self {
        self.vector = Some(VectorConstraint::new(query, threshold, k));
        self
    }

    /// Add graph pattern constraint
    pub fn with_graph(mut self, pattern: &str) -> Self {
        self.graph = Some(GraphConstraint::new(pattern));
        self
    }

    /// Add causal constraint
    pub fn with_causal(mut self, mode: CausalMode) -> Self {
        self.causal = Some(CausalConstraint {
            mode,
            action: None,
            outcome: None,
        });
        self
    }

    /// Add temporal constraint
    pub fn at_version(mut self, version: u64) -> Self {
        self.temporal = TemporalConstraint::AtVersion(version);
        self
    }

    /// Add qualia filter
    pub fn with_qualia(mut self, min: Option<u8>, max: Option<u8>) -> Self {
        self.qualia = Some(QualiaFilter {
            min_qidx: min,
            max_qidx: max,
        });
        self
    }

    /// Add label filter
    pub fn with_label(mut self, label: &str) -> Self {
        self.label_filter = Some(label.to_string());
        self
    }

    /// Set limit
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = n;
        self
    }

    /// Set offset
    pub fn offset(mut self, n: usize) -> Self {
        self.offset = n;
        self
    }
}

// =============================================================================
// HYBRID RESULT
// =============================================================================

/// Single result from hybrid query
#[derive(Debug, Clone)]
pub struct HybridResult {
    /// Address of the matched node
    pub addr: Addr,
    /// Fingerprint
    pub fingerprint: [u64; FINGERPRINT_WORDS],
    /// Hamming distance (if vector search was used)
    pub distance: Option<u32>,
    /// Similarity score (0.0 - 1.0)
    pub similarity: f32,
    /// Qualia index
    pub qidx: u8,
    /// Node label
    pub label: Option<String>,
    /// Version at which this was found
    pub version: u64,
    /// Which constraints were satisfied
    pub satisfied: Vec<String>,
}

impl HybridResult {
    /// Compute composite score
    pub fn score(&self) -> f32 {
        self.similarity
    }
}

/// Query execution statistics
#[derive(Debug, Clone, Default)]
pub struct HybridStats {
    /// Total execution time
    pub total_time: Duration,
    /// Time spent in vector search
    pub vector_time: Duration,
    /// Time spent in graph filtering
    pub graph_time: Duration,
    /// Time spent in causal filtering
    pub causal_time: Duration,
    /// Time spent in temporal resolution
    pub temporal_time: Duration,
    /// Number of candidates after vector search
    pub vector_candidates: usize,
    /// Number after graph filtering
    pub graph_candidates: usize,
    /// Number after causal filtering
    pub causal_candidates: usize,
    /// Final result count
    pub final_count: usize,
}

// =============================================================================
// HYBRID QUERY ENGINE
// =============================================================================

/// Engine for executing hybrid queries
pub struct HybridEngine {
    /// Substrate for vector search
    substrate: Substrate,
    /// Bind space for node storage
    bind_space: BindSpace,
    /// Current version counter
    current_version: u64,
}

impl HybridEngine {
    pub fn new() -> Self {
        Self {
            substrate: Substrate::new(SubstrateConfig::default()),
            bind_space: BindSpace::new(),
            current_version: 1,
        }
    }

    /// Create from existing storage
    pub fn with_storage(substrate: Substrate, bind_space: BindSpace) -> Self {
        Self {
            substrate,
            bind_space,
            current_version: 1,
        }
    }

    /// Execute a hybrid query
    pub fn execute(&self, query: &HybridQuery) -> Result<(Vec<HybridResult>, HybridStats), String> {
        let start = Instant::now();
        let mut stats = HybridStats::default();

        // Stage 1: Vector search (narrows candidate set)
        let vector_start = Instant::now();
        let candidates = self.execute_vector(query)?;
        stats.vector_time = vector_start.elapsed();
        stats.vector_candidates = candidates.len();

        // Stage 2: Graph filter
        let graph_start = Instant::now();
        let graph_filtered = self.execute_graph(query, candidates)?;
        stats.graph_time = graph_start.elapsed();
        stats.graph_candidates = graph_filtered.len();

        // Stage 3: Causal filter
        let causal_start = Instant::now();
        let causal_filtered = self.execute_causal(query, graph_filtered)?;
        stats.causal_time = causal_start.elapsed();
        stats.causal_candidates = causal_filtered.len();

        // Stage 4: Temporal resolution + final filtering
        let temporal_start = Instant::now();
        let results = self.execute_temporal(query, causal_filtered)?;
        stats.temporal_time = temporal_start.elapsed();
        stats.final_count = results.len();

        stats.total_time = start.elapsed();

        Ok((results, stats))
    }

    /// Stage 1: Vector similarity search
    fn execute_vector(&self, query: &HybridQuery) -> Result<Vec<(Addr, f32)>, String> {
        match &query.vector {
            Some(vc) => {
                // Use substrate resonate for vector search
                let results = self.substrate.resonate(&vc.query, vc.k);

                // Convert CogAddr to Addr and filter by threshold
                Ok(results
                    .into_iter()
                    .filter(|(_, sim)| *sim >= vc.threshold)
                    .map(|(cog_addr, sim)| (Addr(cog_addr.0), sim))
                    .collect())
            }
            None => {
                // No vector constraint - return all nodes up to limit
                let mut results = Vec::new();
                for prefix in 0x80..=0xFF_u8 {
                    for slot in 0..=255_u8 {
                        let addr = Addr::new(prefix, slot);
                        if self.bind_space.read(addr).is_some() {
                            results.push((addr, 1.0f32));
                            if results.len() >= query.limit * 10 {
                                return Ok(results);
                            }
                        }
                    }
                }
                Ok(results)
            }
        }
    }

    /// Stage 2: Graph pattern matching
    fn execute_graph(&self, query: &HybridQuery, candidates: Vec<(Addr, f32)>) -> Result<Vec<(Addr, f32)>, String> {
        match &query.graph {
            Some(gc) => {
                if let Some(ref parsed) = gc.parsed {
                    // Apply pattern constraints (simplified - check labels)
                    Ok(candidates
                        .into_iter()
                        .filter(|(addr, _)| {
                            self.matches_pattern(*addr, parsed)
                        })
                        .collect())
                } else {
                    Ok(candidates)
                }
            }
            None => Ok(candidates),
        }
    }

    /// Check if a node matches a Cypher pattern
    fn matches_pattern(&self, addr: Addr, query: &CypherQuery) -> bool {
        if let Some(node) = self.bind_space.read(addr) {
            // Check label matches if specified in pattern
            if let Some(ref match_clause) = query.match_clause {
                for pattern in &match_clause.patterns {
                    for element in &pattern.elements {
                        if let PatternElement::Node(node_pat) = element {
                            for label in &node_pat.labels {
                                // Check if node has matching label
                                if let Some(ref node_label) = node.label {
                                    if !node_label.to_lowercase().contains(&label.to_lowercase()) {
                                        return false;
                                    }
                                } else {
                                    return false;
                                }
                            }
                        }
                    }
                }
            }
            true
        } else {
            false
        }
    }

    /// Stage 3: Causal reasoning filter
    fn execute_causal(&self, query: &HybridQuery, candidates: Vec<(Addr, f32)>) -> Result<Vec<(Addr, f32)>, String> {
        match &query.causal {
            Some(_cc) => {
                // Placeholder - would check causal graph
                // For now, pass all candidates
                Ok(candidates)
            }
            None => Ok(candidates),
        }
    }

    /// Stage 4: Temporal resolution and final filtering
    fn execute_temporal(&self, query: &HybridQuery, candidates: Vec<(Addr, f32)>) -> Result<Vec<HybridResult>, String> {
        let version = match &query.temporal {
            TemporalConstraint::AtVersion(v) => *v,
            TemporalConstraint::Latest => self.current_version,
            TemporalConstraint::VersionRange { to, .. } => *to,
            TemporalConstraint::AtTime(_) => self.current_version,
        };

        let mut results = Vec::new();

        for (addr, similarity) in candidates {
            if let Some(node) = self.bind_space.read(addr) {
                // Apply qualia filter
                if let Some(ref qf) = query.qualia {
                    if !qf.matches(node.qidx) {
                        continue;
                    }
                }

                // Apply label filter
                if let Some(ref label_filter) = query.label_filter {
                    match &node.label {
                        Some(label) if label.contains(label_filter) => {}
                        _ => continue,
                    }
                }

                // Compute distance from similarity
                let max_bits = (FINGERPRINT_WORDS * 64) as f32;
                let distance = ((1.0 - similarity) * max_bits) as u32;

                let mut satisfied = Vec::new();
                if query.vector.is_some() {
                    satisfied.push("vector".to_string());
                }
                if query.graph.is_some() {
                    satisfied.push("graph".to_string());
                }
                if query.causal.is_some() {
                    satisfied.push("causal".to_string());
                }

                results.push(HybridResult {
                    addr,
                    fingerprint: node.fingerprint,
                    distance: Some(distance),
                    similarity,
                    qidx: node.qidx,
                    label: node.label.clone(),
                    version,
                    satisfied,
                });
            }
        }

        // Sort by score (similarity)
        results.sort_by(|a, b| b.score().partial_cmp(&a.score()).unwrap());

        // Apply offset and limit
        Ok(results
            .into_iter()
            .skip(query.offset)
            .take(query.limit)
            .collect())
    }

    /// Add a node to the engine
    pub fn insert(&mut self, fingerprint: [u64; FINGERPRINT_WORDS], label: Option<&str>) -> Addr {
        self.substrate.write(fingerprint);
        match label {
            Some(l) => self.bind_space.write_labeled(fingerprint, l),
            None => self.bind_space.write(fingerprint),
        }
    }

    /// Get current version
    pub fn version(&self) -> u64 {
        self.current_version
    }

    /// Advance version
    pub fn advance_version(&mut self) -> u64 {
        self.current_version += 1;
        self.current_version
    }
}

impl Default for HybridEngine {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// HYBRID QUERY PARSER
// =============================================================================

/// Parse hybrid query syntax
pub fn parse_hybrid(input: &str) -> Result<HybridQuery, String> {
    let mut query = HybridQuery::new();
    let input = input.trim();

    // Check for HYBRID { ... } wrapper
    let content = if input.starts_with("HYBRID") {
        let start = input.find('{').ok_or("Expected '{' after HYBRID")?;
        let end = input.rfind('}').ok_or("Expected '}' to close HYBRID")?;
        &input[start + 1..end]
    } else {
        input
    };

    // Parse individual clauses
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if line.starts_with("MATCH") {
            // Graph pattern
            query.graph = Some(GraphConstraint::new(line));
        } else if line.starts_with("AT VERSION") {
            if let Some(v) = line.strip_prefix("AT VERSION ") {
                if let Ok(version) = v.trim().parse::<u64>() {
                    query.temporal = TemporalConstraint::AtVersion(version);
                }
            }
        } else if line.starts_with("LIMIT") {
            if let Some(l) = line.strip_prefix("LIMIT ") {
                if let Ok(limit) = l.trim().parse::<usize>() {
                    query.limit = limit;
                }
            }
        } else if line.starts_with("OFFSET") || line.starts_with("SKIP") {
            let l = line.strip_prefix("OFFSET ").or_else(|| line.strip_prefix("SKIP "));
            if let Some(l) = l {
                if let Ok(offset) = l.trim().parse::<usize>() {
                    query.offset = offset;
                }
            }
        }
    }

    Ok(query)
}

// =============================================================================
// REDIS COMMAND INTEGRATION
// =============================================================================

/// Execute hybrid query from Redis-like command
pub fn execute_hybrid_command(
    engine: &HybridEngine,
    args: &[&str],
) -> Result<String, String> {
    if args.is_empty() {
        return Err("HYBRID requires query string".to_string());
    }

    let query_str = args.join(" ");
    let query = parse_hybrid(&query_str)?;
    let (results, stats) = engine.execute(&query)?;

    // Format response
    let mut response = String::new();
    response.push_str(&format!("Results: {} ({})\n", results.len(), format_duration(stats.total_time)));
    response.push_str(&format!(
        "Pipeline: vector({}ms, {}) -> graph({}ms, {}) -> causal({}ms, {})\n",
        stats.vector_time.as_millis(),
        stats.vector_candidates,
        stats.graph_time.as_millis(),
        stats.graph_candidates,
        stats.causal_time.as_millis(),
        stats.causal_candidates,
    ));

    for (i, result) in results.iter().enumerate() {
        response.push_str(&format!(
            "{}. {:04X} | sim={:.3} dist={} | label={:?}\n",
            i + 1,
            result.addr.0,
            result.similarity,
            result.distance.unwrap_or(0),
            result.label,
        ));
    }

    Ok(response)
}

fn format_duration(d: Duration) -> String {
    let nanos = d.as_nanos();
    if nanos >= 1_000_000 {
        format!("{:.2}ms", nanos as f64 / 1_000_000.0)
    } else if nanos >= 1_000 {
        format!("{:.2}us", nanos as f64 / 1_000.0)
    } else {
        format!("{}ns", nanos)
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn random_fingerprint(seed: u64) -> [u64; FINGERPRINT_WORDS] {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut fp = [0u64; FINGERPRINT_WORDS];
        for i in 0..FINGERPRINT_WORDS {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            i.hash(&mut hasher);
            fp[i] = hasher.finish();
        }
        fp
    }

    #[test]
    fn test_hybrid_query_builder() {
        let query = HybridQuery::new()
            .with_vector(random_fingerprint(42), 0.8, 10)
            .with_graph("MATCH (a:Person)-[:KNOWS]->(b)")
            .with_causal(CausalMode::See)
            .at_version(5)
            .limit(20);

        assert!(query.vector.is_some());
        assert!(query.graph.is_some());
        assert!(query.causal.is_some());
        assert_eq!(query.limit, 20);
        match query.temporal {
            TemporalConstraint::AtVersion(v) => assert_eq!(v, 5),
            _ => panic!("Expected AtVersion"),
        }
    }

    #[test]
    fn test_hybrid_engine_execute() {
        let mut engine = HybridEngine::new();

        // Insert some nodes
        for i in 0..100 {
            let fp = random_fingerprint(i);
            engine.insert(fp, Some(&format!("Node{}", i)));
        }

        // Query with vector constraint
        let query_fp = random_fingerprint(42);
        let query = HybridQuery::new()
            .with_vector(query_fp, 0.0, 10)  // Low threshold to get results
            .limit(5);

        let (results, stats) = engine.execute(&query).unwrap();

        assert!(results.len() <= 5);
        assert!(stats.total_time.as_nanos() > 0);
    }

    #[test]
    fn test_qualia_filter() {
        let filter = QualiaFilter {
            min_qidx: Some(50),
            max_qidx: Some(200),
        };

        assert!(filter.matches(100));
        assert!(filter.matches(50));
        assert!(filter.matches(200));
        assert!(!filter.matches(49));
        assert!(!filter.matches(201));
    }

    #[test]
    fn test_parse_hybrid() {
        let query_str = r#"
            HYBRID {
                MATCH (a:Person)-[:KNOWS]->(b:Person)
                AT VERSION 42
                LIMIT 10
            }
        "#;

        let query = parse_hybrid(query_str).unwrap();
        assert!(query.graph.is_some());
        assert_eq!(query.limit, 10);
        match query.temporal {
            TemporalConstraint::AtVersion(v) => assert_eq!(v, 42),
            _ => panic!("Expected AtVersion(42)"),
        }
    }
}
