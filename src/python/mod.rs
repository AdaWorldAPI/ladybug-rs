//! Python bindings via PyO3
//!
//! ```python
//! import ladybug
//!
//! # Open database
//! db = ladybug.open("./mydb")
//!
//! # SQL
//! results = db.sql("SELECT * FROM thoughts WHERE confidence > 0.7")
//!
//! # Cypher
//! results = db.cypher("MATCH (a)-[:CAUSES]->(b) RETURN b")
//!
//! # Resonance (Hamming similarity)
//! similar = db.resonate(fingerprint, threshold=0.7, limit=10)
//!
//! # Counterfactual
//! what_if = db.fork().remove("feature_flag").propagate().diff()
//!
//! # NARS inference
//! truth = ladybug.TruthValue(frequency=0.9, confidence=0.8)
//! conclusion = truth.deduction(other_truth)
//!
//! # Batch Hamming distance (for bighorn integration)
//! distances = ladybug.batch_hamming(query_bytes, [fp1_bytes, fp2_bytes])
//!
//! # Direct byte operations
//! fp = ladybug.Fingerprint.from_bytes(my_2048_bytes)
//! raw_bytes = fp.to_bytes()
//! ```

#![cfg(feature = "python")]

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};

use crate::core::Fingerprint;
use crate::core::simd::{batch_hamming as rust_batch_hamming, hamming_distance};
use crate::nars::TruthValue;
use crate::storage::Database;
use crate::{FINGERPRINT_BITS, FINGERPRINT_BYTES};

/// Python wrapper for Fingerprint
#[pyclass(name = "Fingerprint")]
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

    /// Create from raw bytes (2048 bytes = 256 u64 words)
    #[staticmethod]
    fn from_bytes(bytes: &[u8]) -> PyResult<Self> {
        Fingerprint::from_bytes(bytes)
            .map(|inner| Self { inner })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    /// Create random fingerprint
    #[staticmethod]
    fn random() -> Self {
        Self {
            inner: Fingerprint::random(),
        }
    }

    /// Create zero fingerprint
    #[staticmethod]
    fn zero() -> Self {
        Self {
            inner: Fingerprint::zero(),
        }
    }

    /// Create orthogonal basis fingerprint for index
    #[staticmethod]
    fn orthogonal(index: usize) -> Self {
        Self {
            inner: Fingerprint::orthogonal(index),
        }
    }

    /// Get raw bytes (2048 bytes)
    fn to_bytes<'py>(&self, py: Python<'py>) -> &'py PyBytes {
        PyBytes::new(py, self.inner.as_bytes())
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

    /// Unbind (same as bind, XOR is self-inverse)
    fn unbind(&self, other: &PyFingerprint) -> PyFingerprint {
        PyFingerprint {
            inner: self.inner.unbind(&other.inner),
        }
    }

    /// Permute bits (for sequence encoding)
    fn permute(&self, positions: i32) -> PyFingerprint {
        PyFingerprint {
            inner: self.inner.permute(positions),
        }
    }

    /// Inverse permute
    fn unpermute(&self, positions: i32) -> PyFingerprint {
        PyFingerprint {
            inner: self.inner.unpermute(positions),
        }
    }

    /// Bitwise AND
    fn and_(&self, other: &PyFingerprint) -> PyFingerprint {
        PyFingerprint {
            inner: self.inner.and(&other.inner),
        }
    }

    /// Bitwise OR
    fn or_(&self, other: &PyFingerprint) -> PyFingerprint {
        PyFingerprint {
            inner: self.inner.or(&other.inner),
        }
    }

    /// Bitwise NOT
    fn not_(&self) -> PyFingerprint {
        PyFingerprint {
            inner: self.inner.not(),
        }
    }

    /// Count set bits
    fn popcount(&self) -> u32 {
        self.inner.popcount()
    }

    /// Bit density (0.0 - 1.0)
    fn density(&self) -> f32 {
        self.inner.density()
    }

    /// Get bit at position
    fn get_bit(&self, pos: usize) -> bool {
        self.inner.get_bit(pos)
    }

    /// Set bit at position
    fn set_bit(&mut self, pos: usize, value: bool) {
        self.inner.set_bit(pos, value);
    }

    fn __repr__(&self) -> String {
        format!(
            "Fingerprint({} bits set, density={:.2})",
            self.inner.popcount(),
            self.inner.density()
        )
    }
}

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
        format!(
            "<{:.2}, {:.2}>",
            self.inner.frequency, self.inner.confidence
        )
    }

    fn __str__(&self) -> String {
        format!(
            "⟨{:.0}%, {:.0}%⟩",
            self.inner.frequency * 100.0,
            self.inner.confidence * 100.0
        )
    }
}

/// Python wrapper for Database
#[pyclass(name = "Database")]
pub struct PyDatabase {
    inner: Database,
}

#[pymethods]
impl PyDatabase {
    /// Open database at path
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let db = Database::open(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(Self { inner: db })
    }

    /// Create in-memory database
    #[staticmethod]
    fn memory() -> Self {
        Self {
            inner: Database::memory(),
        }
    }

    /// Execute SQL query
    fn sql(&self, query: &str) -> PyResult<Vec<Vec<String>>> {
        let result = self
            .inner
            .sql(query)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result.rows)
    }

    /// Execute Cypher query
    fn cypher(&self, query: &str) -> PyResult<Vec<Vec<String>>> {
        let result = self
            .inner
            .cypher(query)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result.rows)
    }

    /// Resonance search by content
    fn resonate(&self, content: &str, threshold: f32, limit: usize) -> Vec<(usize, f32)> {
        self.inner.resonate_content(content, threshold, limit)
    }

    /// Resonance search by fingerprint
    fn resonate_fp(&self, fp: &PyFingerprint, threshold: f32, limit: usize) -> Vec<(usize, f32)> {
        self.inner.resonate(&fp.inner, threshold, limit)
    }

    /// Index fingerprints for search
    fn index(&self, py: Python, fingerprints: &PyList) -> PyResult<()> {
        let fps: Vec<Fingerprint> = fingerprints
            .iter()
            .map(|item| {
                let py_fp: PyRef<PyFingerprint> = item.extract()?;
                Ok(py_fp.inner.clone())
            })
            .collect::<PyResult<Vec<_>>>()?;

        self.inner.index_fingerprints(fps);
        Ok(())
    }

    /// Fork for counterfactual reasoning
    fn fork(&self) -> PyDatabase {
        PyDatabase {
            inner: self.inner.fork(),
        }
    }

    /// Database path
    #[getter]
    fn path(&self) -> &str {
        self.inner.path()
    }

    /// Current version
    #[getter]
    fn version(&self) -> u64 {
        self.inner.version()
    }

    /// Number of indexed fingerprints
    fn fingerprint_count(&self) -> usize {
        self.inner.fingerprint_count()
    }

    fn __repr__(&self) -> String {
        format!(
            "Database('{}', version={})",
            self.inner.path(),
            self.inner.version()
        )
    }
}

/// Open database (convenience function)
#[pyfunction]
fn open(path: &str) -> PyResult<PyDatabase> {
    PyDatabase::new(path)
}

/// Get SIMD level
#[pyfunction]
fn simd_level() -> &'static str {
    crate::core::simd::simd_level()
}

// =============================================================================
// BATCH OPERATIONS (for bighorn integration)
// =============================================================================

/// Batch Hamming distance from query to all candidates
/// Returns list of distances (u32)
#[pyfunction]
fn batch_hamming(py: Python, query: &PyFingerprint, candidates: &PyList) -> PyResult<Vec<u32>> {
    let fps: Vec<Fingerprint> = candidates
        .iter()
        .map(|item| {
            let py_fp: PyRef<PyFingerprint> = item.extract()?;
            Ok(py_fp.inner.clone())
        })
        .collect::<PyResult<Vec<_>>>()?;

    Ok(rust_batch_hamming(&query.inner, &fps))
}

/// Batch Hamming from raw bytes
/// Query: 2048 bytes, Candidates: list of 2048-byte arrays
/// Returns list of distances
#[pyfunction]
fn batch_hamming_bytes(
    py: Python,
    query_bytes: &[u8],
    candidate_bytes: Vec<&[u8]>,
) -> PyResult<Vec<u32>> {
    let query = Fingerprint::from_bytes(query_bytes)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let candidates: Result<Vec<Fingerprint>, _> = candidate_bytes
        .iter()
        .map(|b| Fingerprint::from_bytes(b))
        .collect();

    let candidates =
        candidates.map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(rust_batch_hamming(&query, &candidates))
}

/// Raw Hamming distance between two byte arrays
#[pyfunction]
fn hamming_bytes(a: &[u8], b: &[u8]) -> PyResult<u32> {
    let fp_a = Fingerprint::from_bytes(a)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let fp_b = Fingerprint::from_bytes(b)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(hamming_distance(&fp_a, &fp_b))
}

/// Majority vote bundling of multiple fingerprints
/// Returns a new fingerprint where each bit is the majority vote
#[pyfunction]
fn bundle(py: Python, fingerprints: &PyList) -> PyResult<PyFingerprint> {
    if fingerprints.is_empty() {
        return Ok(PyFingerprint {
            inner: Fingerprint::zero(),
        });
    }

    let fps: Vec<Fingerprint> = fingerprints
        .iter()
        .map(|item| {
            let py_fp: PyRef<PyFingerprint> = item.extract()?;
            Ok(py_fp.inner.clone())
        })
        .collect::<PyResult<Vec<_>>>()?;

    // Majority vote per bit
    let threshold = fps.len() / 2;
    let mut result = Fingerprint::zero();

    for bit in 0..FINGERPRINT_BITS {
        let count: usize = fps.iter().filter(|fp| fp.get_bit(bit)).count();
        if count > threshold {
            result.set_bit(bit, true);
        }
    }

    Ok(PyFingerprint { inner: result })
}

/// Top-k search by Hamming distance
/// Returns list of (index, distance, similarity) tuples
#[pyfunction]
fn topk_hamming(
    py: Python,
    query: &PyFingerprint,
    candidates: &PyList,
    k: usize,
) -> PyResult<Vec<(usize, u32, f32)>> {
    let fps: Vec<Fingerprint> = candidates
        .iter()
        .map(|item| {
            let py_fp: PyRef<PyFingerprint> = item.extract()?;
            Ok(py_fp.inner.clone())
        })
        .collect::<PyResult<Vec<_>>>()?;

    let distances = rust_batch_hamming(&query.inner, &fps);

    // Get top-k
    let mut indexed: Vec<(usize, u32)> = distances.into_iter().enumerate().collect();

    let k = k.min(indexed.len());
    indexed.select_nth_unstable_by_key(k.saturating_sub(1), |&(_, d)| d);
    indexed.truncate(k);
    indexed.sort_by_key(|&(_, d)| d);

    Ok(indexed
        .into_iter()
        .map(|(idx, dist)| {
            let similarity = 1.0 - (dist as f32 / FINGERPRINT_BITS as f32);
            (idx, dist, similarity)
        })
        .collect())
}

/// Threshold search - find all within Hamming distance threshold
#[pyfunction]
fn threshold_hamming(
    py: Python,
    query: &PyFingerprint,
    candidates: &PyList,
    max_distance: u32,
) -> PyResult<Vec<(usize, u32, f32)>> {
    let fps: Vec<Fingerprint> = candidates
        .iter()
        .map(|item| {
            let py_fp: PyRef<PyFingerprint> = item.extract()?;
            Ok(py_fp.inner.clone())
        })
        .collect::<PyResult<Vec<_>>>()?;

    let distances = rust_batch_hamming(&query.inner, &fps);

    Ok(distances
        .into_iter()
        .enumerate()
        .filter(|&(_, d)| d <= max_distance)
        .map(|(idx, dist)| {
            let similarity = 1.0 - (dist as f32 / FINGERPRINT_BITS as f32);
            (idx, dist, similarity)
        })
        .collect())
}

// =============================================================================
// MODULE DEFINITION
// =============================================================================

/// Module definition
#[pymodule]
fn ladybug(_py: Python, m: &PyModule) -> PyResult<()> {
    // Classes
    m.add_class::<PyFingerprint>()?;
    m.add_class::<PyTruthValue>()?;
    m.add_class::<PyDatabase>()?;

    // Database functions
    m.add_function(wrap_pyfunction!(open, m)?)?;

    // Info functions
    m.add_function(wrap_pyfunction!(simd_level, m)?)?;

    // Batch operations (for bighorn integration)
    m.add_function(wrap_pyfunction!(batch_hamming, m)?)?;
    m.add_function(wrap_pyfunction!(batch_hamming_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(bundle, m)?)?;
    m.add_function(wrap_pyfunction!(topk_hamming, m)?)?;
    m.add_function(wrap_pyfunction!(threshold_hamming, m)?)?;

    // Constants
    m.add("VERSION", env!("CARGO_PKG_VERSION"))?;
    m.add("FINGERPRINT_BITS", FINGERPRINT_BITS)?;
    m.add("FINGERPRINT_BYTES", FINGERPRINT_BYTES)?;

    Ok(())
}
