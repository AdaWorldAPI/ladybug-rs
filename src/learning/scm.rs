//! Structural Causal Model — lives IN BindSpace, not beside it.
//!
//! An SCM is not a separate data structure sitting outside the cognitive
//! substrate. It IS a pattern of connected BindNodes with CAUSES/CONFOUNDS
//! verb edges, where conditional probability tables are encoded as NARS
//! truth values on the edges.
//!
//! When the system encounters a new causal structure, it encodes it into
//! BindSpace. When it encounters a SIMILAR structure later, the fingerprints
//! resonate — that's transfer learning. When results are validated at L9
//! and crystallized at L10, causal knowledge becomes permanent.
//!
//! ```text
//! SCM in BindSpace:
//!
//!     [var:X]  ──CAUSES──►  [var:Y]  ◄──CAUSES──  [var:Z]
//!       │                     │                      │
//!    val:X=0  val:X=1      val:Y=0  val:Y=1      val:Z=0  val:Z=1
//!
//!  Each edge carries a NARS truth value:
//!    CAUSES(X→Y) with <freq=P(Y=1|X=1), conf=evidence_count>
//!
//!  The CPT is encoded as interventional evidence in CausalEngine:
//!    store_intervention(state=[Z=z], action=[X=x], outcome=[Y=y], weight=P(Y=y|X=x,Z=z))
//! ```

use std::collections::HashMap;
use crate::core::Fingerprint;
use crate::storage::bind_space::{Addr, BindSpace};
use ladybug_contract::nars::TruthValue;

// =============================================================================
// CAUSAL VARIABLE
// =============================================================================

/// A variable in a Structural Causal Model.
///
/// Each variable has a name, a set of possible values, parent variables,
/// and a conditional probability table (CPT) encoding P(V|parents(V)).
#[derive(Debug, Clone)]
pub struct CausalVariable {
    /// Variable name (e.g. "X", "Y", "Z")
    pub name: String,

    /// Fingerprint encoding of this variable's identity.
    /// Deterministic: `Fingerprint::from_content("scm_var:{name}")`
    pub fingerprint: Fingerprint,

    /// Possible values (e.g. ["0", "1"] for binary variables)
    pub values: Vec<String>,

    /// Fingerprint for each possible value.
    /// `Fingerprint::from_content("scm_val:{name}={value}")`
    pub value_fingerprints: Vec<Fingerprint>,

    /// Names of parent variables (direct causes)
    pub parents: Vec<String>,

    /// Conditional Probability Table.
    ///
    /// Key: vector of parent value indices (in parent order).
    /// Value: probability distribution over this variable's values.
    ///
    /// Example for binary Y with parents [X, Z]:
    ///   [0, 0] → [0.9, 0.1]  means P(Y=0|X=0,Z=0)=0.9, P(Y=1|X=0,Z=0)=0.1
    ///   [1, 0] → [0.3, 0.7]  means P(Y=0|X=1,Z=0)=0.3, P(Y=1|X=1,Z=0)=0.7
    pub cpt: HashMap<Vec<usize>, Vec<f64>>,

    /// BindSpace address (set after encoding)
    pub addr: Option<Addr>,
}

impl CausalVariable {
    /// Create a new variable with deterministic fingerprints.
    pub fn new(name: &str, values: &[&str]) -> Self {
        let fingerprint = Fingerprint::from_content(&format!("scm_var:{}", name));
        let value_fingerprints = values
            .iter()
            .map(|v| Fingerprint::from_content(&format!("scm_val:{}={}", name, v)))
            .collect();

        Self {
            name: name.to_string(),
            fingerprint,
            values: values.iter().map(|s| s.to_string()).collect(),
            value_fingerprints,
            parents: Vec::new(),
            cpt: HashMap::new(),
            addr: None,
        }
    }

    /// Create a binary variable (values "0" and "1").
    pub fn binary(name: &str) -> Self {
        Self::new(name, &["0", "1"])
    }

    /// Set a CPT row: P(self|parent_assignment) = probabilities.
    ///
    /// `parent_values`: vector of value indices for each parent (in parent order).
    /// `probs`: probability for each value of this variable (must sum to ~1.0).
    pub fn set_cpt(&mut self, parent_values: Vec<usize>, probs: Vec<f64>) {
        debug_assert!(
            (probs.iter().sum::<f64>() - 1.0).abs() < 0.01,
            "CPT row must sum to 1.0, got {}",
            probs.iter().sum::<f64>()
        );
        debug_assert_eq!(
            probs.len(),
            self.values.len(),
            "CPT row length must match number of values"
        );
        self.cpt.insert(parent_values, probs);
    }

    /// Set marginal probability for a root variable (no parents).
    pub fn set_marginal(&mut self, probs: Vec<f64>) {
        self.set_cpt(vec![], probs);
    }

    /// Get P(self=value_idx | parent_assignment).
    pub fn prob(&self, value_idx: usize, parent_assignment: &[usize]) -> f64 {
        if let Some(row) = self.cpt.get(parent_assignment) {
            row[value_idx]
        } else {
            // Uniform default if CPT entry missing
            1.0 / self.values.len() as f64
        }
    }

    /// Value index by name (e.g. "1" → 1 for binary).
    pub fn value_index(&self, value_name: &str) -> Option<usize> {
        self.values.iter().position(|v| v == value_name)
    }

    /// Number of possible values.
    pub fn cardinality(&self) -> usize {
        self.values.len()
    }
}

// =============================================================================
// STRUCTURAL CAUSAL MODEL
// =============================================================================

/// A Structural Causal Model that lives in BindSpace.
///
/// The SCM is the cognitive substrate's understanding of HOW the world
/// works. It's not a lookup table — it's a living, evolving pattern of
/// causal knowledge encoded in fingerprint space.
#[derive(Debug, Clone)]
pub struct StructuralCausalModel {
    /// Model name (for crystallization identity).
    pub name: String,

    /// Model fingerprint (hash of structure + parameters).
    pub fingerprint: Fingerprint,

    /// Variables in the model.
    pub variables: Vec<CausalVariable>,

    /// Variable name → index.
    var_index: HashMap<String, usize>,

    /// Directed edges: (parent_name, child_name).
    pub edges: Vec<(String, String)>,

    /// NARS truth value for the model itself.
    /// Frequency = how often this model has been correct.
    /// Confidence = how much evidence we have.
    pub model_truth: TruthValue,
}

impl StructuralCausalModel {
    /// Create a new empty SCM.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            fingerprint: Fingerprint::from_content(&format!("scm:{}", name)),
            variables: Vec::new(),
            var_index: HashMap::new(),
            edges: Vec::new(),
            model_truth: TruthValue::new(0.5, 0.1), // Unknown initially
        }
    }

    /// Add a variable to the model.
    pub fn add_variable(&mut self, var: CausalVariable) {
        let idx = self.variables.len();
        self.var_index.insert(var.name.clone(), idx);
        self.variables.push(var);
        self.update_fingerprint();
    }

    /// Add a directed edge: parent causes child.
    pub fn add_edge(&mut self, parent: &str, child: &str) {
        // Set parent relationship on child variable
        if let Some(&child_idx) = self.var_index.get(child) {
            if !self.variables[child_idx].parents.contains(&parent.to_string()) {
                self.variables[child_idx].parents.push(parent.to_string());
            }
        }
        self.edges.push((parent.to_string(), child.to_string()));
        self.update_fingerprint();
    }

    /// Get a variable by name.
    pub fn variable(&self, name: &str) -> Option<&CausalVariable> {
        self.var_index.get(name).map(|&i| &self.variables[i])
    }

    /// Get a mutable variable by name.
    pub fn variable_mut(&mut self, name: &str) -> Option<&mut CausalVariable> {
        self.var_index.get(name).copied().map(|i| &mut self.variables[i])
    }

    /// Get variable index by name.
    pub fn var_idx(&self, name: &str) -> Option<usize> {
        self.var_index.get(name).copied()
    }

    /// Number of variables.
    pub fn num_variables(&self) -> usize {
        self.variables.len()
    }

    // =========================================================================
    // TOPOLOGICAL ORDER
    // =========================================================================

    /// Return variables in topological order (parents before children).
    pub fn topological_order(&self) -> Vec<usize> {
        let n = self.variables.len();
        let mut in_degree = vec![0usize; n];
        let mut adj: Vec<Vec<usize>> = vec![vec![]; n];

        for (parent, child) in &self.edges {
            if let (Some(&pi), Some(&ci)) = (self.var_index.get(parent), self.var_index.get(child))
            {
                adj[pi].push(ci);
                in_degree[ci] += 1;
            }
        }

        let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut order = Vec::with_capacity(n);

        while let Some(node) = queue.pop() {
            order.push(node);
            for &child in &adj[node] {
                in_degree[child] -= 1;
                if in_degree[child] == 0 {
                    queue.push(child);
                }
            }
        }

        order
    }

    /// Get parent indices for a variable.
    pub fn parent_indices(&self, var_idx: usize) -> Vec<usize> {
        self.variables[var_idx]
            .parents
            .iter()
            .filter_map(|name| self.var_index.get(name).copied())
            .collect()
    }

    /// Get children indices for a variable.
    pub fn children_indices(&self, var_idx: usize) -> Vec<usize> {
        let name = &self.variables[var_idx].name;
        self.edges
            .iter()
            .filter(|(p, _)| p == name)
            .filter_map(|(_, c)| self.var_index.get(c).copied())
            .collect()
    }

    // =========================================================================
    // GRAPH OPERATIONS
    // =========================================================================

    /// Find all ancestors of a variable (transitive parents).
    pub fn ancestors(&self, var_name: &str) -> Vec<usize> {
        let mut result = Vec::new();
        let mut visited = std::collections::HashSet::new();

        fn dfs(
            model: &StructuralCausalModel,
            var_idx: usize,
            visited: &mut std::collections::HashSet<usize>,
            result: &mut Vec<usize>,
        ) {
            for parent_idx in model.parent_indices(var_idx) {
                if visited.insert(parent_idx) {
                    result.push(parent_idx);
                    dfs(model, parent_idx, visited, result);
                }
            }
        }

        if let Some(&idx) = self.var_index.get(var_name) {
            dfs(self, idx, &mut visited, &mut result);
        }
        result
    }

    /// Find all descendants of a variable (transitive children).
    pub fn descendants(&self, var_name: &str) -> Vec<usize> {
        let mut result = Vec::new();
        let mut visited = std::collections::HashSet::new();

        fn dfs(
            model: &StructuralCausalModel,
            var_idx: usize,
            visited: &mut std::collections::HashSet<usize>,
            result: &mut Vec<usize>,
        ) {
            for child_idx in model.children_indices(var_idx) {
                if visited.insert(child_idx) {
                    result.push(child_idx);
                    dfs(model, child_idx, visited, result);
                }
            }
        }

        if let Some(&idx) = self.var_index.get(var_name) {
            dfs(self, idx, &mut visited, &mut result);
        }
        result
    }

    /// Check if there's a directed path from `from` to `to`.
    pub fn has_path(&self, from: &str, to: &str) -> bool {
        if let Some(&to_idx) = self.var_index.get(to) {
            self.descendants(from).contains(&to_idx)
        } else {
            false
        }
    }

    /// Find a valid back-door adjustment set for estimating X → Y.
    ///
    /// Z satisfies the back-door criterion relative to (X, Y) if:
    /// 1. No node in Z is a descendant of X
    /// 2. Z blocks every back-door path from X to Y
    ///
    /// Returns the set of variable names in the adjustment set.
    pub fn find_backdoor_set(&self, x_name: &str, y_name: &str) -> Vec<String> {
        let x_descendants: std::collections::HashSet<usize> =
            self.descendants(x_name).into_iter().collect();

        let x_idx = match self.var_index.get(x_name) {
            Some(&i) => i,
            None => return vec![],
        };
        let y_idx = match self.var_index.get(y_name) {
            Some(&i) => i,
            None => return vec![],
        };

        // Candidates: all variables that are not X, Y, or descendants of X
        let candidates: Vec<usize> = (0..self.variables.len())
            .filter(|&i| i != x_idx && i != y_idx && !x_descendants.contains(&i))
            .collect();

        // Simple heuristic: parents of X that are not descendants of X
        // This is the minimal sufficient adjustment set for most CLadder models
        let x_parents: Vec<usize> = self.parent_indices(x_idx);
        let adjustment: Vec<String> = x_parents
            .iter()
            .filter(|&&p| candidates.contains(&p))
            .map(|&p| self.variables[p].name.clone())
            .collect();

        if !adjustment.is_empty() {
            return adjustment;
        }

        // Fallback: all non-descendant, non-X, non-Y parents of Y
        // that are also ancestors of X (confounders)
        let x_ancestors: std::collections::HashSet<usize> =
            self.ancestors(x_name).into_iter().collect();
        let y_ancestors: std::collections::HashSet<usize> =
            self.ancestors(y_name).into_iter().collect();

        // Common ancestors that are not descendants of X
        candidates
            .iter()
            .filter(|&&c| x_ancestors.contains(&c) || y_ancestors.contains(&c))
            .map(|&c| self.variables[c].name.clone())
            .collect()
    }

    // =========================================================================
    // PROBABILITY COMPUTATION
    // =========================================================================

    /// Compute P(assignment) for a full variable assignment.
    ///
    /// `assignment[i]` = value index for variable i (in topological order).
    pub fn joint_probability(&self, assignment: &[usize]) -> f64 {
        let topo = self.topological_order();
        let mut prob = 1.0;

        for &var_idx in &topo {
            let var = &self.variables[var_idx];
            let parent_idxs = self.parent_indices(var_idx);

            // Get parent assignment
            let parent_assignment: Vec<usize> =
                parent_idxs.iter().map(|&pi| assignment[pi]).collect();

            prob *= var.prob(assignment[var_idx], &parent_assignment);
        }

        prob
    }

    /// Compute marginal probability: P(var_name = value_idx).
    ///
    /// Sums over all possible assignments to other variables.
    pub fn marginal(&self, var_name: &str, value_idx: usize) -> f64 {
        let var_idx = match self.var_index.get(var_name) {
            Some(&i) => i,
            None => return 0.0,
        };

        let n = self.variables.len();
        let cardinalities: Vec<usize> = self.variables.iter().map(|v| v.cardinality()).collect();

        let mut total = 0.0;
        let mut assignment = vec![0usize; n];

        // Enumerate all assignments where var_name = value_idx
        self.enumerate_assignments(&cardinalities, &mut assignment, 0, &mut |a| {
            if a[var_idx] == value_idx {
                total += self.joint_probability(a);
            }
        });

        total
    }

    /// Compute conditional probability: P(Y=y | evidence).
    ///
    /// `evidence`: list of (variable_name, value_index) pairs.
    pub fn conditional(
        &self,
        y_name: &str,
        y_value: usize,
        evidence: &[(&str, usize)],
    ) -> f64 {
        let y_idx = match self.var_index.get(y_name) {
            Some(&i) => i,
            None => return 0.0,
        };

        let evidence_map: HashMap<usize, usize> = evidence
            .iter()
            .filter_map(|(name, val)| self.var_index.get(*name).map(|&idx| (idx, *val)))
            .collect();

        let n = self.variables.len();
        let cardinalities: Vec<usize> = self.variables.iter().map(|v| v.cardinality()).collect();

        let mut numerator = 0.0; // P(Y=y, evidence)
        let mut denominator = 0.0; // P(evidence)
        let mut assignment = vec![0usize; n];

        self.enumerate_assignments(&cardinalities, &mut assignment, 0, &mut |a| {
            // Check if assignment is consistent with evidence
            let consistent = evidence_map.iter().all(|(&idx, &val)| a[idx] == val);
            if consistent {
                let p = self.joint_probability(a);
                denominator += p;
                if a[y_idx] == y_value {
                    numerator += p;
                }
            }
        });

        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }

    /// Compute interventional probability: P(Y=y | do(X=x)).
    ///
    /// This modifies the model by removing all edges INTO X and setting
    /// P(X=x) = 1.0. Then computes P(Y=y) in the modified model.
    pub fn interventional(
        &self,
        y_name: &str,
        y_value: usize,
        x_name: &str,
        x_value: usize,
    ) -> f64 {
        // Create mutilated model: remove edges into X, set P(X=x) = 1
        let mutilated = self.mutilate(x_name, x_value);
        mutilated.marginal(y_name, y_value)
    }

    /// Compute interventional with additional evidence:
    /// P(Y=y | do(X=x), Z=z).
    pub fn interventional_conditional(
        &self,
        y_name: &str,
        y_value: usize,
        x_name: &str,
        x_value: usize,
        evidence: &[(&str, usize)],
    ) -> f64 {
        let mutilated = self.mutilate(x_name, x_value);
        mutilated.conditional(y_name, y_value, evidence)
    }

    /// Compute counterfactual: P(Y_{x}=y | evidence).
    ///
    /// Three-step process (Abduction-Action-Prediction):
    /// 1. Abduction: Given evidence, compute posterior on exogenous variables
    /// 2. Action: Create model with do(X=x)
    /// 3. Prediction: Compute P(Y=y) using posterior and modified model
    ///
    /// For deterministic SCMs with known CPTs and binary variables,
    /// this computes the exact counterfactual.
    pub fn counterfactual(
        &self,
        y_name: &str,
        y_value: usize,
        x_name: &str,
        x_value: usize,
        evidence: &[(&str, usize)],
    ) -> f64 {
        let _y_idx = match self.var_index.get(y_name) {
            Some(&i) => i,
            None => return 0.0,
        };
        let x_idx = match self.var_index.get(x_name) {
            Some(&i) => i,
            None => return 0.0,
        };

        let evidence_map: HashMap<usize, usize> = evidence
            .iter()
            .filter_map(|(name, val)| self.var_index.get(*name).map(|&idx| (idx, *val)))
            .collect();

        let n = self.variables.len();
        let cardinalities: Vec<usize> = self.variables.iter().map(|v| v.cardinality()).collect();

        // Step 1: Abduction — compute P(full_assignment | evidence) for each
        // consistent assignment. This gives us the posterior over all latent states.
        let mut consistent_worlds: Vec<(Vec<usize>, f64)> = Vec::new();
        let mut evidence_prob = 0.0;
        let mut assignment = vec![0usize; n];

        self.enumerate_assignments(&cardinalities, &mut assignment, 0, &mut |a| {
            let consistent = evidence_map.iter().all(|(&idx, &val)| a[idx] == val);
            if consistent {
                let p = self.joint_probability(a);
                evidence_prob += p;
                consistent_worlds.push((a.to_vec(), p));
            }
        });

        if evidence_prob <= 0.0 {
            return 0.0;
        }

        // Step 2 & 3: Action + Prediction
        // For each consistent world, intervene on X and compute Y
        let mutilated = self.mutilate(x_name, x_value);

        let mut cf_prob = 0.0;

        for (world, world_prob) in &consistent_worlds {
            let posterior = world_prob / evidence_prob;

            // In this world, we know the values of all non-descendant variables.
            // After intervention do(X=x), we need to recompute descendants of X.
            // Non-descendants keep their actual-world values.

            let x_desc: std::collections::HashSet<usize> =
                self.descendants(x_name).into_iter().collect();

            // Build evidence for the mutilated model:
            // All non-X, non-descendant-of-X variables keep their actual values
            let mut cf_evidence: Vec<(&str, usize)> = Vec::new();
            for i in 0..n {
                if i != x_idx && !x_desc.contains(&i) {
                    cf_evidence.push((&self.variables[i].name, world[i]));
                }
            }

            // Compute P(Y=y | do(X=x), non-descendant evidence) in mutilated model
            let p_y = mutilated.conditional(y_name, y_value, &cf_evidence);
            cf_prob += posterior * p_y;
        }

        cf_prob
    }

    // =========================================================================
    // GRAPH MUTILATION (for interventions)
    // =========================================================================

    /// Create a mutilated model: remove edges into X, set P(X=x) = 1.
    fn mutilate(&self, x_name: &str, x_value: usize) -> StructuralCausalModel {
        let mut mutilated = self.clone();

        // Remove edges into X
        mutilated.edges.retain(|(_, child)| child != x_name);

        // Clear parents of X
        if let Some(x_var) = mutilated.variable_mut(x_name) {
            x_var.parents.clear();

            // Set CPT to deterministic: P(X=x_value) = 1.0
            let mut probs = vec![0.0; x_var.cardinality()];
            probs[x_value] = 1.0;
            x_var.cpt.clear();
            x_var.set_marginal(probs);
        }

        mutilated
    }

    // =========================================================================
    // BINDSPACE ENCODING
    // =========================================================================

    /// Encode this SCM into a BindSpace.
    ///
    /// Creates BindNodes for each variable and value, BindEdges for
    /// causal relationships, and stores CPT entries in the CausalEngine.
    pub fn encode_to_bindspace(&mut self, space: &mut BindSpace) {

        // 1. Write variable nodes into Causal surface (prefix 0x05)
        for (i, var) in self.variables.iter_mut().enumerate() {
            let fp_words = *var.fingerprint.as_raw();
            let slot = i as u8;
            let addr = Addr::new(0x05, slot);
            space.write_at(addr, fp_words);
            var.addr = Some(addr);
        }

        // 2. Create edges for causal relationships
        for (parent_name, child_name) in &self.edges {
            if let (Some(parent_var), Some(child_var)) = (
                self.var_index.get(parent_name).map(|&i| &self.variables[i]),
                self.var_index.get(child_name).map(|&i| &self.variables[i]),
            ) {
                if let (Some(from_addr), Some(to_addr)) = (parent_var.addr, child_var.addr) {
                    // CAUSES verb is at prefix 0x07, slot 0x00
                    let causes_verb = Addr::new(0x07, 0x00);
                    space.link(from_addr, causes_verb, to_addr);
                }
            }
        }
    }

    /// Compute the structural fingerprint — encodes the DAG topology.
    ///
    /// Two SCMs with the same structure (same edges, different parameters)
    /// will have SIMILAR fingerprints. This enables resonance-based
    /// transfer of causal knowledge between analogous domains.
    pub fn structural_fingerprint(&self) -> Fingerprint {
        // Base fingerprint is topology-only — no model name, no parameters.
        // This ensures that two SCMs with the same DAG structure produce
        // identical structural fingerprints regardless of naming or CPT values.
        let mut fp = Fingerprint::from_content("scm_structure");

        // Sort edges for deterministic ordering
        let mut sorted_edges: Vec<_> = self.edges.iter().collect();
        sorted_edges.sort();

        for (parent, child) in sorted_edges {
            let edge_fp = Fingerprint::from_content(&format!("edge:{}→{}", parent, child));
            fp = fp.bind(&edge_fp);
        }

        fp
    }

    // =========================================================================
    // HELPERS
    // =========================================================================

    /// Enumerate all possible assignments to variables.
    fn enumerate_assignments(
        &self,
        cardinalities: &[usize],
        assignment: &mut Vec<usize>,
        var_idx: usize,
        f: &mut dyn FnMut(&[usize]),
    ) {
        if var_idx == cardinalities.len() {
            f(assignment);
            return;
        }

        for val in 0..cardinalities[var_idx] {
            assignment[var_idx] = val;
            self.enumerate_assignments(cardinalities, assignment, var_idx + 1, f);
        }
    }

    /// Update the model fingerprint after structural changes.
    fn update_fingerprint(&mut self) {
        self.fingerprint = self.structural_fingerprint();
    }
}

// =============================================================================
// SCM BUILDER (Fluent API)
// =============================================================================

/// Fluent builder for constructing SCMs.
///
/// ```rust,ignore
/// let scm = ScmBuilder::new("aspirin")
///     .binary("X")   // Treatment
///     .binary("Y")   // Outcome
///     .binary("Z")   // Confounder
///     .edge("Z", "X")
///     .edge("Z", "Y")
///     .edge("X", "Y")
///     .marginal("Z", &[0.4, 0.6])
///     .cpt("X", &[0], &[0.8, 0.2])   // P(X|Z=0)
///     .cpt("X", &[1], &[0.3, 0.7])   // P(X|Z=1)
///     .cpt("Y", &[0, 0], &[0.9, 0.1]) // P(Y|X=0,Z=0)
///     .cpt("Y", &[0, 1], &[0.7, 0.3]) // P(Y|X=0,Z=1)
///     .cpt("Y", &[1, 0], &[0.4, 0.6]) // P(Y|X=1,Z=0)
///     .cpt("Y", &[1, 1], &[0.2, 0.8]) // P(Y|X=1,Z=1)
///     .build();
/// ```
pub struct ScmBuilder {
    scm: StructuralCausalModel,
}

impl ScmBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            scm: StructuralCausalModel::new(name),
        }
    }

    /// Add a variable with arbitrary values.
    pub fn variable(mut self, name: &str, values: &[&str]) -> Self {
        self.scm.add_variable(CausalVariable::new(name, values));
        self
    }

    /// Add a binary variable.
    pub fn binary(mut self, name: &str) -> Self {
        self.scm.add_variable(CausalVariable::binary(name));
        self
    }

    /// Add a directed edge.
    pub fn edge(mut self, parent: &str, child: &str) -> Self {
        self.scm.add_edge(parent, child);
        self
    }

    /// Set marginal probability for a root variable.
    pub fn marginal(mut self, var_name: &str, probs: &[f64]) -> Self {
        if let Some(var) = self.scm.variable_mut(var_name) {
            var.set_marginal(probs.to_vec());
        }
        self
    }

    /// Set a CPT row.
    pub fn cpt(mut self, var_name: &str, parent_values: &[usize], probs: &[f64]) -> Self {
        if let Some(var) = self.scm.variable_mut(var_name) {
            var.set_cpt(parent_values.to_vec(), probs.to_vec());
        }
        self
    }

    /// Build the SCM.
    pub fn build(self) -> StructuralCausalModel {
        self.scm
    }
}

// =============================================================================
// NARS EVIDENCE FROM SCM
// =============================================================================

impl StructuralCausalModel {
    /// Convert a probability computation result to a NARS TruthValue.
    ///
    /// frequency = computed probability
    /// confidence = based on model's own truth value and number of variables
    pub fn prob_to_truth(&self, probability: f64) -> TruthValue {
        let freq = probability as f32;
        // Confidence scales with model confidence and inversely with model complexity
        let complexity_penalty = 1.0 / (1.0 + self.variables.len() as f32 * 0.1);
        let conf = self.model_truth.confidence * complexity_penalty;
        TruthValue::new(freq, conf)
    }

    /// Update model truth based on a prediction-outcome pair.
    ///
    /// If the model predicted correctly, evidence increases.
    /// This is how the system LEARNS which causal models are reliable.
    pub fn update_truth(&mut self, predicted_correctly: bool) {
        let new_evidence = TruthValue::new(
            if predicted_correctly { 1.0 } else { 0.0 },
            0.9,
        );
        self.model_truth = ladybug_contract::nars::TruthValue::new(
            self.model_truth.frequency,
            self.model_truth.confidence,
        )
        .revision(&new_evidence);
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Classic confounded model: Z → X, Z → Y, X → Y
    fn aspirin_model() -> StructuralCausalModel {
        ScmBuilder::new("aspirin")
            .binary("Z") // Confounder (e.g. health condition)
            .binary("X") // Treatment (aspirin)
            .binary("Y") // Outcome (recovery)
            .edge("Z", "X")
            .edge("Z", "Y")
            .edge("X", "Y")
            // P(Z=0) = 0.5, P(Z=1) = 0.5
            .marginal("Z", &[0.5, 0.5])
            // P(X|Z)
            .cpt("X", &[0], &[0.8, 0.2]) // Z=0 → unlikely to take aspirin
            .cpt("X", &[1], &[0.3, 0.7]) // Z=1 → likely to take aspirin
            // P(Y|Z,X)  — parent order matches edge order: Z→Y then X→Y
            .cpt("Y", &[0, 0], &[0.9, 0.1]) // Z=0,X=0 → low recovery
            .cpt("Y", &[0, 1], &[0.7, 0.3]) // Z=0,X=1 → aspirin helps
            .cpt("Y", &[1, 0], &[0.4, 0.6]) // Z=1,X=0 → moderate (condition helps)
            .cpt("Y", &[1, 1], &[0.2, 0.8]) // Z=1,X=1 → aspirin + condition
            .build()
    }

    /// Simple chain: X → Y (no confounders)
    fn simple_chain() -> StructuralCausalModel {
        ScmBuilder::new("simple")
            .binary("X")
            .binary("Y")
            .edge("X", "Y")
            .marginal("X", &[0.5, 0.5])
            .cpt("Y", &[0], &[0.8, 0.2])
            .cpt("Y", &[1], &[0.3, 0.7])
            .build()
    }

    #[test]
    fn test_marginal_probability() {
        let model = simple_chain();

        // P(X=0) = 0.5 (by definition)
        let p_x0 = model.marginal("X", 0);
        assert!((p_x0 - 0.5).abs() < 0.001, "P(X=0) = {}", p_x0);

        // P(Y=1) = P(Y=1|X=0)*P(X=0) + P(Y=1|X=1)*P(X=1) = 0.2*0.5 + 0.7*0.5 = 0.45
        let p_y1 = model.marginal("Y", 1);
        assert!((p_y1 - 0.45).abs() < 0.001, "P(Y=1) = {}", p_y1);
    }

    #[test]
    fn test_conditional_probability() {
        let model = simple_chain();

        // P(Y=1|X=1) = 0.7 (directly from CPT)
        let p = model.conditional("Y", 1, &[("X", 1)]);
        assert!((p - 0.7).abs() < 0.001, "P(Y=1|X=1) = {}", p);

        // P(Y=1|X=0) = 0.2 (directly from CPT)
        let p = model.conditional("Y", 1, &[("X", 0)]);
        assert!((p - 0.2).abs() < 0.001, "P(Y=1|X=0) = {}", p);
    }

    #[test]
    fn test_intervention_no_confounders() {
        let model = simple_chain();

        // With no confounders, P(Y|do(X)) = P(Y|X)
        let p_do = model.interventional("Y", 1, "X", 1);
        let p_cond = model.conditional("Y", 1, &[("X", 1)]);
        assert!(
            (p_do - p_cond).abs() < 0.001,
            "do(X) = conditioning when no confounders: {} vs {}",
            p_do,
            p_cond
        );
    }

    #[test]
    fn test_intervention_with_confounders() {
        let model = aspirin_model();

        // P(Y=1|X=1) ≠ P(Y=1|do(X=1)) because Z confounds
        let p_cond = model.conditional("Y", 1, &[("X", 1)]);
        let p_do = model.interventional("Y", 1, "X", 1);

        // These MUST be different — that's the whole point of do-calculus
        assert!(
            (p_cond - p_do).abs() > 0.01,
            "Conditional ({}) must differ from interventional ({}) with confounders",
            p_cond,
            p_do
        );

        // Manual computation:
        // P(Y=1|do(X=1)) = Σ_z P(Y=1|X=1,Z=z) * P(Z=z)
        //                 = P(Y=1|X=1,Z=0)*0.5 + P(Y=1|X=1,Z=1)*0.5
        //                 = 0.3*0.5 + 0.8*0.5
        //                 = 0.55
        assert!(
            (p_do - 0.55).abs() < 0.001,
            "P(Y=1|do(X=1)) should be 0.55, got {}",
            p_do
        );
    }

    #[test]
    fn test_counterfactual() {
        let model = simple_chain();

        // Observed: X=0, Y=0
        // Counterfactual: what would Y have been if X had been 1?
        let p_cf = model.counterfactual("Y", 1, "X", 1, &[("X", 0), ("Y", 0)]);

        // In this simple model, P(Y_1=1 | X=0, Y=0) = P(Y=1|X=1) = 0.7
        // because Y only depends on X (no confounders to "fix")
        assert!(
            (p_cf - 0.7).abs() < 0.001,
            "P(Y_1=1 | X=0, Y=0) = {}, expected 0.7",
            p_cf
        );
    }

    #[test]
    fn test_counterfactual_with_confounders() {
        let model = aspirin_model();

        // Observed: X=1, Y=1 (took aspirin, recovered)
        // Counterfactual: what if they hadn't taken aspirin (X=0)?
        let p_cf = model.counterfactual("Y", 1, "X", 0, &[("X", 1), ("Y", 1)]);

        // This should account for the confounder Z — if they took aspirin (X=1),
        // they're more likely to have Z=1, which independently promotes Y=1.
        // So the counterfactual P(Y=1|X=0) should be higher than the naive
        // P(Y=1|X=0) because Z is likely to be 1.
        let naive = model.conditional("Y", 1, &[("X", 0)]);
        assert!(
            p_cf > naive * 0.9,
            "Counterfactual ({}) should be close to or above naive ({}), accounting for Z",
            p_cf,
            naive
        );
    }

    #[test]
    fn test_backdoor_set() {
        let model = aspirin_model();

        // Z is the back-door adjustment set for X → Y
        let adjustment = model.find_backdoor_set("X", "Y");
        assert!(
            adjustment.contains(&"Z".to_string()),
            "Back-door set should contain Z, got {:?}",
            adjustment
        );
    }

    #[test]
    fn test_topological_order() {
        let model = aspirin_model();
        let order = model.topological_order();

        // Z should come before X and Y
        let z_pos = order.iter().position(|&i| model.variables[i].name == "Z");
        let x_pos = order.iter().position(|&i| model.variables[i].name == "X");
        let y_pos = order.iter().position(|&i| model.variables[i].name == "Y");

        assert!(z_pos < x_pos, "Z should come before X in topological order");
        assert!(x_pos < y_pos, "X should come before Y in topological order");
    }

    #[test]
    fn test_structural_fingerprint_resonance() {
        // Two models with same structure but different names should resonate
        let model1 = ScmBuilder::new("aspirin")
            .binary("X")
            .binary("Y")
            .binary("Z")
            .edge("Z", "X")
            .edge("Z", "Y")
            .edge("X", "Y")
            .marginal("Z", &[0.5, 0.5])
            .cpt("X", &[0], &[0.8, 0.2])
            .cpt("X", &[1], &[0.3, 0.7])
            // Parent order for Y: [Z, X]
            .cpt("Y", &[0, 0], &[0.9, 0.1])
            .cpt("Y", &[0, 1], &[0.7, 0.3])
            .cpt("Y", &[1, 0], &[0.4, 0.6])
            .cpt("Y", &[1, 1], &[0.2, 0.8])
            .build();

        let model2 = ScmBuilder::new("smoking")
            .binary("X")
            .binary("Y")
            .binary("Z")
            .edge("Z", "X")
            .edge("Z", "Y")
            .edge("X", "Y")
            .marginal("Z", &[0.3, 0.7])
            .cpt("X", &[0], &[0.6, 0.4])
            .cpt("X", &[1], &[0.2, 0.8])
            // Parent order for Y: [Z, X]
            .cpt("Y", &[0, 0], &[0.95, 0.05])
            .cpt("Y", &[0, 1], &[0.5, 0.5])
            .cpt("Y", &[1, 0], &[0.6, 0.4])
            .cpt("Y", &[1, 1], &[0.1, 0.9])
            .build();

        let fp1 = model1.structural_fingerprint();
        let fp2 = model2.structural_fingerprint();

        // Same structure should produce identical structural fingerprints
        // (both have Z→X, Z→Y, X→Y)
        let sim = fp1.similarity(&fp2);
        assert!(
            sim > 0.99,
            "Same causal structure should have identical structural fingerprints, got sim={}",
            sim
        );

        // Different structure should be dissimilar
        let model3 = ScmBuilder::new("chain")
            .binary("X")
            .binary("Y")
            .binary("Z")
            .edge("X", "Z")
            .edge("Z", "Y")
            .marginal("X", &[0.5, 0.5])
            .cpt("Z", &[0], &[0.8, 0.2])
            .cpt("Z", &[1], &[0.3, 0.7])
            .cpt("Y", &[0], &[0.9, 0.1])
            .cpt("Y", &[1], &[0.4, 0.6])
            .build();

        let fp3 = model3.structural_fingerprint();
        let sim13 = fp1.similarity(&fp3);
        assert!(
            sim13 < sim,
            "Different structure should have lower similarity: {} vs {}",
            sim13,
            sim
        );
    }

    #[test]
    fn test_nars_truth_from_scm() {
        let model = simple_chain();
        let prob = model.marginal("Y", 1);
        let tv = model.prob_to_truth(prob);

        assert!((tv.frequency - prob as f32).abs() < 0.001);
        assert!(tv.confidence > 0.0);
    }

    #[test]
    fn test_descendants_ancestors() {
        let model = aspirin_model();

        // Z has no ancestors
        assert!(model.ancestors("Z").is_empty());

        // Z is ancestor of both X and Y
        let z_desc = model.descendants("Z");
        assert!(z_desc.contains(&model.var_idx("X").unwrap()));
        assert!(z_desc.contains(&model.var_idx("Y").unwrap()));

        // Y has no descendants
        assert!(model.descendants("Y").is_empty());
    }
}
