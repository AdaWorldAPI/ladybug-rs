//! Python bindings via PyO3
//!
//! ```python
//! import ladybug
//!
//! # Core operations
//! fp1 = ladybug.Fingerprint("hello world")
//! fp2 = ladybug.Fingerprint("hello there")
//! distance = fp1.hamming(fp2)
//! similarity = fp1.similarity(fp2)
//!
//! # NARS inference
//! truth = ladybug.TruthValue(frequency=0.9, confidence=0.8)
//! conclusion = truth.deduction(other_truth)
//!
//! # Hybrid queries (unified Graph + Vector + Temporal)
//! engine = ladybug.HybridEngine()
//! engine.insert(fp1, label="greeting")
//! results, stats = engine.execute(
//!     ladybug.HybridQuery()
//!         .with_vector(fp2, threshold=0.5)
//!         .limit(10)
//! )
//! ```

#![cfg(feature = "python")]

use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::core::Fingerprint;
use crate::nars::TruthValue;
use crate::storage::{FINGERPRINT_WORDS};
use crate::FINGERPRINT_U64;
use crate::query::hybrid::{
    HybridQuery, HybridEngine, CausalMode,
    parse_hybrid,
};

// =============================================================================
// FINGERPRINT BINDINGS
// =============================================================================

/// Python wrapper for Fingerprint
#[pyclass(name = "Fingerprint")]
#[derive(Clone)]
pub struct PyFingerprint {
    inner: Fingerprint,
}

#[pymethods]
impl PyFingerprint {
    /// Create from content string
    #[new]
    fn new(content: &str) -> Self {
        Self {
            inner: Fingerprint::from_content(content),
        }
    }

    /// Create random fingerprint
    #[staticmethod]
    fn random() -> Self {
        Self {
            inner: Fingerprint::random(),
        }
    }

    /// Hamming distance to another fingerprint
    fn hamming(&self, other: &PyFingerprint) -> u32 {
        self.inner.hamming(&other.inner)
    }

    /// Similarity (0.0 - 1.0)
    fn similarity(&self, other: &PyFingerprint) -> f32 {
        self.inner.similarity(&other.inner)
    }

    /// Bind (XOR) with another fingerprint
    fn bind(&self, other: &PyFingerprint) -> PyFingerprint {
        PyFingerprint {
            inner: self.inner.bind(&other.inner),
        }
    }

    /// Count set bits
    fn popcount(&self) -> u32 {
        self.inner.popcount()
    }

    fn __repr__(&self) -> String {
        format!("Fingerprint({} bits set)", self.inner.popcount())
    }
}

impl PyFingerprint {
    /// Get the raw fingerprint data for internal use
    fn as_raw(&self) -> [u64; FINGERPRINT_U64] {
        *self.inner.as_raw()
    }
}

// =============================================================================
// TRUTH VALUE BINDINGS (NARS)
// =============================================================================

/// Python wrapper for TruthValue
#[pyclass(name = "TruthValue")]
#[derive(Clone)]
pub struct PyTruthValue {
    inner: TruthValue,
}

#[pymethods]
impl PyTruthValue {
    #[new]
    fn new(frequency: f32, confidence: f32) -> Self {
        Self {
            inner: TruthValue::new(frequency, confidence),
        }
    }

    /// Create from evidence counts
    #[staticmethod]
    fn from_evidence(positive: f32, negative: f32) -> Self {
        Self {
            inner: TruthValue::from_evidence(positive, negative),
        }
    }

    /// Frequency component
    #[getter]
    fn frequency(&self) -> f32 {
        self.inner.frequency
    }

    /// Confidence component
    #[getter]
    fn confidence(&self) -> f32 {
        self.inner.confidence
    }

    /// Expected value for decision making
    fn expectation(&self) -> f32 {
        self.inner.expectation()
    }

    /// Revision: combine with independent evidence
    fn revision(&self, other: &PyTruthValue) -> PyTruthValue {
        PyTruthValue {
            inner: self.inner.revision(&other.inner),
        }
    }

    /// Deduction: A→B, B→C ⊢ A→C
    fn deduction(&self, other: &PyTruthValue) -> PyTruthValue {
        PyTruthValue {
            inner: self.inner.deduction(&other.inner),
        }
    }

    /// Induction: A→B, A→C ⊢ B→C
    fn induction(&self, other: &PyTruthValue) -> PyTruthValue {
        PyTruthValue {
            inner: self.inner.induction(&other.inner),
        }
    }

    /// Abduction: A→B, C→B ⊢ A→C
    fn abduction(&self, other: &PyTruthValue) -> PyTruthValue {
        PyTruthValue {
            inner: self.inner.abduction(&other.inner),
        }
    }

    /// Negation
    fn negation(&self) -> PyTruthValue {
        PyTruthValue {
            inner: self.inner.negation(),
        }
    }

    fn __repr__(&self) -> String {
        format!("<{:.2}, {:.2}>", self.inner.frequency, self.inner.confidence)
    }

    fn __str__(&self) -> String {
        format!("⟨{:.0}%, {:.0}%⟩",
            self.inner.frequency * 100.0,
            self.inner.confidence * 100.0
        )
    }
}

// =============================================================================
// HYBRID QUERY BINDINGS
// =============================================================================

/// Python wrapper for HybridQuery
#[pyclass(name = "HybridQuery")]
#[derive(Clone)]
pub struct PyHybridQuery {
    inner: HybridQuery,
}

#[pymethods]
impl PyHybridQuery {
    /// Create a new hybrid query
    #[new]
    fn new() -> Self {
        Self {
            inner: HybridQuery::new(),
        }
    }

    /// Add vector similarity constraint
    #[pyo3(signature = (query, threshold=0.8, k=10))]
    fn with_vector(&self, query: &PyFingerprint, threshold: f32, k: usize) -> Self {
        // Convert from 157-word Fingerprint format to 156-word storage format
        let raw = query.as_raw();
        let mut storage_fp = [0u64; FINGERPRINT_WORDS];
        storage_fp.copy_from_slice(&raw[..FINGERPRINT_WORDS]);
        Self {
            inner: self.inner.clone().with_vector(storage_fp, threshold, k),
        }
    }

    /// Add graph pattern constraint (Cypher MATCH clause)
    fn with_graph(&self, pattern: &str) -> Self {
        Self {
            inner: self.inner.clone().with_graph(pattern),
        }
    }

    /// Add causal constraint (see, do, or imagine)
    fn with_causal(&self, mode: &str) -> PyResult<Self> {
        let causal_mode = match mode.to_lowercase().as_str() {
            "see" | "correlate" => CausalMode::See,
            "do" | "intervene" => CausalMode::Do,
            "imagine" | "counterfactual" => CausalMode::Imagine,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Invalid causal mode: {}. Use 'see', 'do', or 'imagine'", mode)
            )),
        };
        Ok(Self {
            inner: self.inner.clone().with_causal(causal_mode),
        })
    }

    /// Add temporal version constraint
    fn at_version(&self, version: u64) -> Self {
        Self {
            inner: self.inner.clone().at_version(version),
        }
    }

    /// Add qualia filter (qidx range)
    #[pyo3(signature = (min_qidx=None, max_qidx=None))]
    fn with_qualia(&self, min_qidx: Option<u8>, max_qidx: Option<u8>) -> Self {
        Self {
            inner: self.inner.clone().with_qualia(min_qidx, max_qidx),
        }
    }

    /// Add label filter
    fn with_label(&self, label: &str) -> Self {
        Self {
            inner: self.inner.clone().with_label(label),
        }
    }

    /// Set result limit
    fn limit(&self, n: usize) -> Self {
        Self {
            inner: self.inner.clone().limit(n),
        }
    }

    /// Set result offset (for pagination)
    fn offset(&self, n: usize) -> Self {
        Self {
            inner: self.inner.clone().offset(n),
        }
    }

    fn __repr__(&self) -> String {
        format!("HybridQuery(limit={}, offset={})", self.inner.limit, self.inner.offset)
    }
}

/// Python wrapper for HybridResult
#[pyclass(name = "HybridResult")]
#[derive(Clone)]
pub struct PyHybridResult {
    /// Node address (16-bit)
    #[pyo3(get)]
    addr: u16,
    /// Similarity score (0.0 - 1.0)
    #[pyo3(get)]
    similarity: f32,
    /// Hamming distance
    #[pyo3(get)]
    distance: u32,
    /// Qualia index
    #[pyo3(get)]
    qidx: u8,
    /// Node label
    #[pyo3(get)]
    label: Option<String>,
    /// Version
    #[pyo3(get)]
    version: u64,
    /// Fingerprint data (156 words from storage, padded to 157 for Fingerprint)
    fingerprint_data: [u64; FINGERPRINT_WORDS],
}

#[pymethods]
impl PyHybridResult {
    /// Get the fingerprint
    fn fingerprint(&self) -> PyFingerprint {
        // Convert from 156-word storage format to 157-word Fingerprint format
        let mut fp_data = [0u64; FINGERPRINT_U64];
        fp_data[..FINGERPRINT_WORDS].copy_from_slice(&self.fingerprint_data);
        PyFingerprint {
            inner: Fingerprint::from_raw(fp_data),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "HybridResult(addr=0x{:04X}, sim={:.3}, label={:?})",
            self.addr, self.similarity, self.label
        )
    }
}

/// Python wrapper for HybridStats
#[pyclass(name = "HybridStats")]
#[derive(Clone)]
pub struct PyHybridStats {
    /// Total execution time in microseconds
    #[pyo3(get)]
    total_time_us: u64,
    /// Vector search time in microseconds
    #[pyo3(get)]
    vector_time_us: u64,
    /// Graph filter time in microseconds
    #[pyo3(get)]
    graph_time_us: u64,
    /// Candidates after vector search
    #[pyo3(get)]
    vector_candidates: usize,
    /// Candidates after graph filter
    #[pyo3(get)]
    graph_candidates: usize,
    /// Final result count
    #[pyo3(get)]
    final_count: usize,
}

#[pymethods]
impl PyHybridStats {
    fn __repr__(&self) -> String {
        format!(
            "HybridStats(total={}us, results={})",
            self.total_time_us, self.final_count
        )
    }
}

/// Python wrapper for HybridEngine
#[pyclass(name = "HybridEngine")]
pub struct PyHybridEngine {
    inner: HybridEngine,
}

#[pymethods]
impl PyHybridEngine {
    /// Create a new hybrid engine
    #[new]
    fn new() -> Self {
        Self {
            inner: HybridEngine::new(),
        }
    }

    /// Insert a node with optional label
    #[pyo3(signature = (fingerprint, label=None))]
    fn insert(&mut self, fingerprint: &PyFingerprint, label: Option<&str>) -> u16 {
        // Convert from 157-word Fingerprint format to 156-word storage format
        let raw = fingerprint.as_raw();
        let mut storage_fp = [0u64; FINGERPRINT_WORDS];
        storage_fp.copy_from_slice(&raw[..FINGERPRINT_WORDS]);
        self.inner.insert(storage_fp, label).0
    }

    /// Execute a hybrid query
    fn execute(&self, query: &PyHybridQuery) -> PyResult<(Vec<PyHybridResult>, PyHybridStats)> {
        let (results, stats) = self.inner.execute(&query.inner)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

        let py_results: Vec<PyHybridResult> = results
            .into_iter()
            .map(|r| PyHybridResult {
                addr: r.addr.0,
                similarity: r.similarity,
                distance: r.distance.unwrap_or(0),
                qidx: r.qidx,
                label: r.label,
                version: r.version,
                fingerprint_data: r.fingerprint,
            })
            .collect();

        let py_stats = PyHybridStats {
            total_time_us: stats.total_time.as_micros() as u64,
            vector_time_us: stats.vector_time.as_micros() as u64,
            graph_time_us: stats.graph_time.as_micros() as u64,
            vector_candidates: stats.vector_candidates,
            graph_candidates: stats.graph_candidates,
            final_count: stats.final_count,
        };

        Ok((py_results, py_stats))
    }

    /// Execute a hybrid query from a query string
    fn query(&self, query_str: &str) -> PyResult<(Vec<PyHybridResult>, PyHybridStats)> {
        let query = parse_hybrid(query_str)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

        let (results, stats) = self.inner.execute(&query)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

        let py_results: Vec<PyHybridResult> = results
            .into_iter()
            .map(|r| PyHybridResult {
                addr: r.addr.0,
                similarity: r.similarity,
                distance: r.distance.unwrap_or(0),
                qidx: r.qidx,
                label: r.label,
                version: r.version,
                fingerprint_data: r.fingerprint,
            })
            .collect();

        let py_stats = PyHybridStats {
            total_time_us: stats.total_time.as_micros() as u64,
            vector_time_us: stats.vector_time.as_micros() as u64,
            graph_time_us: stats.graph_time.as_micros() as u64,
            vector_candidates: stats.vector_candidates,
            graph_candidates: stats.graph_candidates,
            final_count: stats.final_count,
        };

        Ok((py_results, py_stats))
    }

    /// Get current version
    #[getter]
    fn version(&self) -> u64 {
        self.inner.version()
    }

    /// Advance to next version
    fn advance_version(&mut self) -> u64 {
        self.inner.advance_version()
    }

    fn __repr__(&self) -> String {
        format!("HybridEngine(version={})", self.inner.version())
    }
}

// =============================================================================
// MODULE DEFINITION
// =============================================================================

/// Module definition
#[pymodule]
fn ladybug(_py: Python, m: &PyModule) -> PyResult<()> {
    // Core types
    m.add_class::<PyFingerprint>()?;
    m.add_class::<PyTruthValue>()?;

    // Hybrid query types
    m.add_class::<PyHybridQuery>()?;
    m.add_class::<PyHybridResult>()?;
    m.add_class::<PyHybridStats>()?;
    m.add_class::<PyHybridEngine>()?;

    // Constants
    m.add("VERSION", env!("CARGO_PKG_VERSION"))?;
    m.add("FINGERPRINT_BITS", crate::FINGERPRINT_BITS)?;
    m.add("FINGERPRINT_WORDS", FINGERPRINT_WORDS)?;

    Ok(())
}
