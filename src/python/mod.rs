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
//! ```

#![cfg(feature = "python")]

use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::core::Fingerprint;
use crate::nars::TruthValue;
use crate::storage::Database;

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
        let result = self.inner.sql(query)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result.rows)
    }
    
    /// Execute Cypher query
    fn cypher(&self, query: &str) -> PyResult<Vec<Vec<String>>> {
        let result = self.inner.cypher(query)
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
        format!("Database('{}', version={})", self.inner.path(), self.inner.version())
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

/// Module definition
#[pymodule]
fn ladybug(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyFingerprint>()?;
    m.add_class::<PyTruthValue>()?;
    m.add_class::<PyDatabase>()?;
    m.add_function(wrap_pyfunction!(open, m)?)?;
    m.add_function(wrap_pyfunction!(simd_level, m)?)?;
    
    // Constants
    m.add("VERSION", env!("CARGO_PKG_VERSION"))?;
    m.add("FINGERPRINT_BITS", crate::FINGERPRINT_BITS)?;
    
    Ok(())
}
