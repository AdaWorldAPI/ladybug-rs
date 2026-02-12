//! Container-native UDFs for DataFusion.
//!
//! These UDFs delegate to existing Container methods — NO reimplementation.
//! They expose the Container search/semiring API via SQL alongside
//! the existing direct Rust API.
//!
//! # HARD CONTRACT 3: UDFs are ADDITIVE
//!
//! New UDFs call existing Container methods. They do not reimplement them.
//! One implementation (Container), multiple expositions (Rust + SQL).
//!
//! # New UDFs
//!
//! Container operations:
//! - `belichtung(a, b)` → UInt32 — Belichtungsmesser 7-point estimate
//! - `bundle(a, b, ...)` → FixedSizeBinary(1024) — Majority-vote bundle
//! - `cascade_filter(query, candidate, threshold)` → UInt32 — Full 5-level cascade
//! - `word_diff(a, b)` → UInt32 — Count differing words (0-128)
//! - `mexican_hat(distance)` → Float32 — Mexican hat wavelet response
//! - `container_hamming(a, b)` → UInt32 — Container Hamming (8K bits)
//! - `container_similarity(a, b)` → Float32 — Container similarity
//! - `container_popcount(x)` → UInt32 — Container popcount (8K bits)
//! - `container_xor(a, b)` → FixedSizeBinary(1024) — Container XOR bind

use std::any::Any;

use arrow::datatypes::DataType;
use datafusion::common::Result;
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignature, Volatility,
};

use crate::container::{Container, CONTAINER_BYTES, CONTAINER_BITS, CONTAINER_WORDS};
use crate::container::search::{belichtungsmesser, word_diff_count, hamming_early_exit, MexicanHat};

/// Container fingerprint size in bytes (128 × 8 = 1024)
const C_BYTES: i32 = CONTAINER_BYTES as i32;

// =============================================================================
// HELPERS
// =============================================================================

/// Extract a Container from a ScalarValue.
fn scalar_to_container(s: &datafusion::scalar::ScalarValue) -> Result<Container> {
    let bytes = match s {
        datafusion::scalar::ScalarValue::Binary(Some(b)) => b.as_slice(),
        datafusion::scalar::ScalarValue::LargeBinary(Some(b)) => b.as_slice(),
        datafusion::scalar::ScalarValue::FixedSizeBinary(_, Some(b)) => b.as_slice(),
        _ => return Err(datafusion::error::DataFusionError::Execution(
            "Expected binary scalar for Container".into(),
        )),
    };
    if bytes.len() < CONTAINER_BYTES {
        return Err(datafusion::error::DataFusionError::Execution(
            format!("Container requires {} bytes, got {}", CONTAINER_BYTES, bytes.len()),
        ));
    }
    let arr: [u8; CONTAINER_BYTES] = bytes[..CONTAINER_BYTES].try_into().unwrap();
    Ok(Container::from_bytes(&arr))
}

/// Convert a Container to ColumnarValue scalar.
fn container_to_scalar(c: &Container) -> ColumnarValue {
    ColumnarValue::Scalar(datafusion::scalar::ScalarValue::FixedSizeBinary(
        C_BYTES,
        Some(c.as_bytes().to_vec()),
    ))
}

// =============================================================================
// 1. BELICHTUNG UDF
// =============================================================================

/// Belichtungsmesser: 7-point exposure meter estimate.
/// Delegates to `container::search::belichtungsmesser(a, b)`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct BelichtungUdf {
    signature: Signature,
}

impl BelichtungUdf {
    pub fn new() -> Self {
        Self {
            signature: Signature::one_of(
                vec![
                    TypeSignature::Exact(vec![
                        DataType::FixedSizeBinary(C_BYTES),
                        DataType::FixedSizeBinary(C_BYTES),
                    ]),
                    TypeSignature::Exact(vec![DataType::Binary, DataType::Binary]),
                    TypeSignature::Any(2),
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl Default for BelichtungUdf {
    fn default() -> Self { Self::new() }
}

impl ScalarUDFImpl for BelichtungUdf {
    fn as_any(&self) -> &dyn Any { self }
    fn name(&self) -> &str { "belichtung" }
    fn signature(&self) -> &Signature { &self.signature }
    fn return_type(&self, _: &[DataType]) -> Result<DataType> { Ok(DataType::UInt32) }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let args = &args.args;
        let a = match &args[0] {
            ColumnarValue::Scalar(s) => scalar_to_container(s)?,
            _ => return Err(datafusion::error::DataFusionError::Execution(
                "belichtung currently supports scalar arguments".into(),
            )),
        };
        let b = match &args[1] {
            ColumnarValue::Scalar(s) => scalar_to_container(s)?,
            _ => return Err(datafusion::error::DataFusionError::Execution(
                "belichtung currently supports scalar arguments".into(),
            )),
        };

        let dist = belichtungsmesser(&a, &b);
        Ok(ColumnarValue::Scalar(datafusion::scalar::ScalarValue::UInt32(Some(dist))))
    }
}

// =============================================================================
// 2. BUNDLE UDF
// =============================================================================

/// Majority-vote bundle of two containers.
/// Delegates to `Container::bundle()`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct BundleUdf {
    signature: Signature,
}

impl BundleUdf {
    pub fn new() -> Self {
        Self {
            signature: Signature::one_of(
                vec![
                    TypeSignature::Exact(vec![
                        DataType::FixedSizeBinary(C_BYTES),
                        DataType::FixedSizeBinary(C_BYTES),
                    ]),
                    TypeSignature::Exact(vec![DataType::Binary, DataType::Binary]),
                    TypeSignature::Any(2),
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl Default for BundleUdf {
    fn default() -> Self { Self::new() }
}

impl ScalarUDFImpl for BundleUdf {
    fn as_any(&self) -> &dyn Any { self }
    fn name(&self) -> &str { "bundle" }
    fn signature(&self) -> &Signature { &self.signature }
    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::FixedSizeBinary(C_BYTES))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let args = &args.args;
        let a = match &args[0] {
            ColumnarValue::Scalar(s) => scalar_to_container(s)?,
            _ => return Err(datafusion::error::DataFusionError::Execution(
                "bundle currently supports scalar arguments".into(),
            )),
        };
        let b = match &args[1] {
            ColumnarValue::Scalar(s) => scalar_to_container(s)?,
            _ => return Err(datafusion::error::DataFusionError::Execution(
                "bundle currently supports scalar arguments".into(),
            )),
        };

        let result = Container::bundle(&[&a, &b]);
        Ok(container_to_scalar(&result))
    }
}

// =============================================================================
// 3. CASCADE FILTER UDF
// =============================================================================

/// Full 5-level cascade filter.
/// Delegates to `belichtungsmesser` → `word_diff_count` → `hamming_early_exit`
/// → `MexicanHat::response()`. Same code path as `cascade_search()`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct CascadeFilterUdf {
    signature: Signature,
}

impl CascadeFilterUdf {
    pub fn new() -> Self {
        Self {
            signature: Signature::one_of(
                vec![
                    TypeSignature::Exact(vec![
                        DataType::FixedSizeBinary(C_BYTES),
                        DataType::FixedSizeBinary(C_BYTES),
                        DataType::UInt32,
                    ]),
                    TypeSignature::Any(3),
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl Default for CascadeFilterUdf {
    fn default() -> Self { Self::new() }
}

impl ScalarUDFImpl for CascadeFilterUdf {
    fn as_any(&self) -> &dyn Any { self }
    fn name(&self) -> &str { "cascade_filter" }
    fn signature(&self) -> &Signature { &self.signature }
    fn return_type(&self, _: &[DataType]) -> Result<DataType> { Ok(DataType::UInt32) }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let args = &args.args;
        let query = match &args[0] {
            ColumnarValue::Scalar(s) => scalar_to_container(s)?,
            _ => return Err(datafusion::error::DataFusionError::Execution(
                "cascade_filter: expected scalar query".into(),
            )),
        };
        let candidate = match &args[1] {
            ColumnarValue::Scalar(s) => scalar_to_container(s)?,
            _ => return Err(datafusion::error::DataFusionError::Execution(
                "cascade_filter: expected scalar candidate".into(),
            )),
        };
        let threshold = match &args[2] {
            ColumnarValue::Scalar(datafusion::scalar::ScalarValue::UInt32(Some(v))) => *v,
            _ => return Err(datafusion::error::DataFusionError::Execution(
                "cascade_filter: expected UInt32 threshold".into(),
            )),
        };

        // L0: Belichtungsmesser
        let l0_max = threshold * CONTAINER_BITS as u32 / 448 + 200;
        let estimate = belichtungsmesser(&query, &candidate);
        if estimate > l0_max {
            // Rejected at L0 — return MAX
            return Ok(ColumnarValue::Scalar(datafusion::scalar::ScalarValue::UInt32(Some(u32::MAX))));
        }

        // L1: Word-diff count
        let l1_max = (threshold / 32).max(4) + CONTAINER_WORDS as u32 / 4;
        let wd = word_diff_count(&query, &candidate);
        if wd > l1_max {
            return Ok(ColumnarValue::Scalar(datafusion::scalar::ScalarValue::UInt32(Some(u32::MAX))));
        }

        // L2: Stacked popcount with early exit
        match hamming_early_exit(&query, &candidate, threshold) {
            Some(dist) => Ok(ColumnarValue::Scalar(datafusion::scalar::ScalarValue::UInt32(Some(dist)))),
            None => Ok(ColumnarValue::Scalar(datafusion::scalar::ScalarValue::UInt32(Some(u32::MAX)))),
        }
    }
}

// =============================================================================
// 4. WORD DIFF UDF
// =============================================================================

/// Count how many of 128 words differ.
/// Delegates to `container::search::word_diff_count()`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct WordDiffUdf {
    signature: Signature,
}

impl WordDiffUdf {
    pub fn new() -> Self {
        Self {
            signature: Signature::one_of(
                vec![
                    TypeSignature::Exact(vec![
                        DataType::FixedSizeBinary(C_BYTES),
                        DataType::FixedSizeBinary(C_BYTES),
                    ]),
                    TypeSignature::Any(2),
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl Default for WordDiffUdf {
    fn default() -> Self { Self::new() }
}

impl ScalarUDFImpl for WordDiffUdf {
    fn as_any(&self) -> &dyn Any { self }
    fn name(&self) -> &str { "word_diff" }
    fn signature(&self) -> &Signature { &self.signature }
    fn return_type(&self, _: &[DataType]) -> Result<DataType> { Ok(DataType::UInt32) }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let args = &args.args;
        let a = match &args[0] {
            ColumnarValue::Scalar(s) => scalar_to_container(s)?,
            _ => return Err(datafusion::error::DataFusionError::Execution(
                "word_diff: expected scalar".into(),
            )),
        };
        let b = match &args[1] {
            ColumnarValue::Scalar(s) => scalar_to_container(s)?,
            _ => return Err(datafusion::error::DataFusionError::Execution(
                "word_diff: expected scalar".into(),
            )),
        };
        let count = word_diff_count(&a, &b);
        Ok(ColumnarValue::Scalar(datafusion::scalar::ScalarValue::UInt32(Some(count))))
    }
}

// =============================================================================
// 5. MEXICAN HAT UDF
// =============================================================================

/// Mexican hat wavelet response for a Hamming distance.
/// Delegates to `MexicanHat::default_8k().response()`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct MexicanHatUdf {
    signature: Signature,
}

impl MexicanHatUdf {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(vec![DataType::UInt32], Volatility::Immutable),
        }
    }
}

impl Default for MexicanHatUdf {
    fn default() -> Self { Self::new() }
}

impl ScalarUDFImpl for MexicanHatUdf {
    fn as_any(&self) -> &dyn Any { self }
    fn name(&self) -> &str { "mexican_hat" }
    fn signature(&self) -> &Signature { &self.signature }
    fn return_type(&self, _: &[DataType]) -> Result<DataType> { Ok(DataType::Float32) }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let args = &args.args;
        let distance = match &args[0] {
            ColumnarValue::Scalar(datafusion::scalar::ScalarValue::UInt32(Some(v))) => *v,
            _ => return Err(datafusion::error::DataFusionError::Execution(
                "mexican_hat: expected UInt32 distance".into(),
            )),
        };
        let hat = MexicanHat::default_8k();
        let response = hat.response(distance);
        Ok(ColumnarValue::Scalar(datafusion::scalar::ScalarValue::Float32(Some(response))))
    }
}

// =============================================================================
// 6. CONTAINER HAMMING UDF
// =============================================================================

/// Container Hamming distance (8K bits, as opposed to 16K Fingerprint Hamming).
/// Delegates to `Container::hamming()`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct ContainerHammingUdf {
    signature: Signature,
}

impl ContainerHammingUdf {
    pub fn new() -> Self {
        Self {
            signature: Signature::one_of(
                vec![
                    TypeSignature::Exact(vec![
                        DataType::FixedSizeBinary(C_BYTES),
                        DataType::FixedSizeBinary(C_BYTES),
                    ]),
                    TypeSignature::Exact(vec![DataType::Binary, DataType::Binary]),
                    TypeSignature::Any(2),
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl Default for ContainerHammingUdf {
    fn default() -> Self { Self::new() }
}

impl ScalarUDFImpl for ContainerHammingUdf {
    fn as_any(&self) -> &dyn Any { self }
    fn name(&self) -> &str { "container_hamming" }
    fn signature(&self) -> &Signature { &self.signature }
    fn return_type(&self, _: &[DataType]) -> Result<DataType> { Ok(DataType::UInt32) }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let args = &args.args;
        let a = match &args[0] {
            ColumnarValue::Scalar(s) => scalar_to_container(s)?,
            _ => return Err(datafusion::error::DataFusionError::Execution(
                "container_hamming: expected scalar".into(),
            )),
        };
        let b = match &args[1] {
            ColumnarValue::Scalar(s) => scalar_to_container(s)?,
            _ => return Err(datafusion::error::DataFusionError::Execution(
                "container_hamming: expected scalar".into(),
            )),
        };
        let dist = a.hamming(&b);
        Ok(ColumnarValue::Scalar(datafusion::scalar::ScalarValue::UInt32(Some(dist))))
    }
}

// =============================================================================
// 7. CONTAINER SIMILARITY UDF
// =============================================================================

/// Container similarity (1.0 - hamming/8192).
/// Delegates to `Container::similarity()`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct ContainerSimilarityUdf {
    signature: Signature,
}

impl ContainerSimilarityUdf {
    pub fn new() -> Self {
        Self {
            signature: Signature::one_of(
                vec![
                    TypeSignature::Exact(vec![
                        DataType::FixedSizeBinary(C_BYTES),
                        DataType::FixedSizeBinary(C_BYTES),
                    ]),
                    TypeSignature::Any(2),
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl Default for ContainerSimilarityUdf {
    fn default() -> Self { Self::new() }
}

impl ScalarUDFImpl for ContainerSimilarityUdf {
    fn as_any(&self) -> &dyn Any { self }
    fn name(&self) -> &str { "container_similarity" }
    fn signature(&self) -> &Signature { &self.signature }
    fn return_type(&self, _: &[DataType]) -> Result<DataType> { Ok(DataType::Float32) }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let args = &args.args;
        let a = match &args[0] {
            ColumnarValue::Scalar(s) => scalar_to_container(s)?,
            _ => return Err(datafusion::error::DataFusionError::Execution(
                "container_similarity: expected scalar".into(),
            )),
        };
        let b = match &args[1] {
            ColumnarValue::Scalar(s) => scalar_to_container(s)?,
            _ => return Err(datafusion::error::DataFusionError::Execution(
                "container_similarity: expected scalar".into(),
            )),
        };
        let sim = a.similarity(&b);
        Ok(ColumnarValue::Scalar(datafusion::scalar::ScalarValue::Float32(Some(sim))))
    }
}

// =============================================================================
// 8. CONTAINER POPCOUNT UDF
// =============================================================================

/// Container popcount (8K bits).
/// Delegates to `Container::popcount()`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct ContainerPopcountUdf {
    signature: Signature,
}

impl ContainerPopcountUdf {
    pub fn new() -> Self {
        Self {
            signature: Signature::one_of(
                vec![
                    TypeSignature::Exact(vec![DataType::FixedSizeBinary(C_BYTES)]),
                    TypeSignature::Exact(vec![DataType::Binary]),
                    TypeSignature::Any(1),
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl Default for ContainerPopcountUdf {
    fn default() -> Self { Self::new() }
}

impl ScalarUDFImpl for ContainerPopcountUdf {
    fn as_any(&self) -> &dyn Any { self }
    fn name(&self) -> &str { "container_popcount" }
    fn signature(&self) -> &Signature { &self.signature }
    fn return_type(&self, _: &[DataType]) -> Result<DataType> { Ok(DataType::UInt32) }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let args = &args.args;
        let c = match &args[0] {
            ColumnarValue::Scalar(s) => scalar_to_container(s)?,
            _ => return Err(datafusion::error::DataFusionError::Execution(
                "container_popcount: expected scalar".into(),
            )),
        };
        let count = c.popcount();
        Ok(ColumnarValue::Scalar(datafusion::scalar::ScalarValue::UInt32(Some(count))))
    }
}

// =============================================================================
// 9. CONTAINER XOR UDF
// =============================================================================

/// Container XOR bind (8K bits).
/// Delegates to `Container::xor()`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct ContainerXorUdf {
    signature: Signature,
}

impl ContainerXorUdf {
    pub fn new() -> Self {
        Self {
            signature: Signature::one_of(
                vec![
                    TypeSignature::Exact(vec![
                        DataType::FixedSizeBinary(C_BYTES),
                        DataType::FixedSizeBinary(C_BYTES),
                    ]),
                    TypeSignature::Any(2),
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl Default for ContainerXorUdf {
    fn default() -> Self { Self::new() }
}

impl ScalarUDFImpl for ContainerXorUdf {
    fn as_any(&self) -> &dyn Any { self }
    fn name(&self) -> &str { "container_xor" }
    fn signature(&self) -> &Signature { &self.signature }
    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::FixedSizeBinary(C_BYTES))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let args = &args.args;
        let a = match &args[0] {
            ColumnarValue::Scalar(s) => scalar_to_container(s)?,
            _ => return Err(datafusion::error::DataFusionError::Execution(
                "container_xor: expected scalar".into(),
            )),
        };
        let b = match &args[1] {
            ColumnarValue::Scalar(s) => scalar_to_container(s)?,
            _ => return Err(datafusion::error::DataFusionError::Execution(
                "container_xor: expected scalar".into(),
            )),
        };
        let result = a.xor(&b);
        Ok(container_to_scalar(&result))
    }
}

// =============================================================================
// REGISTRATION
// =============================================================================

use datafusion::execution::context::SessionContext;
use datafusion::logical_expr::ScalarUDF;

/// Register all 9 container UDFs with a DataFusion context.
pub fn register_container_udfs(ctx: &SessionContext) {
    ctx.register_udf(ScalarUDF::from(BelichtungUdf::new()));
    ctx.register_udf(ScalarUDF::from(BundleUdf::new()));
    ctx.register_udf(ScalarUDF::from(CascadeFilterUdf::new()));
    ctx.register_udf(ScalarUDF::from(WordDiffUdf::new()));
    ctx.register_udf(ScalarUDF::from(MexicanHatUdf::new()));
    ctx.register_udf(ScalarUDF::from(ContainerHammingUdf::new()));
    ctx.register_udf(ScalarUDF::from(ContainerSimilarityUdf::new()));
    ctx.register_udf(ScalarUDF::from(ContainerPopcountUdf::new()));
    ctx.register_udf(ScalarUDF::from(ContainerXorUdf::new()));
}

/// All container UDFs as a vector.
pub fn all_container_udfs() -> Vec<ScalarUDF> {
    vec![
        ScalarUDF::from(BelichtungUdf::new()),
        ScalarUDF::from(BundleUdf::new()),
        ScalarUDF::from(CascadeFilterUdf::new()),
        ScalarUDF::from(WordDiffUdf::new()),
        ScalarUDF::from(MexicanHatUdf::new()),
        ScalarUDF::from(ContainerHammingUdf::new()),
        ScalarUDF::from(ContainerSimilarityUdf::new()),
        ScalarUDF::from(ContainerPopcountUdf::new()),
        ScalarUDF::from(ContainerXorUdf::new()),
    ]
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use arrow::datatypes::Field;
    use datafusion::config::ConfigOptions;

    /// Helper to build ScalarFunctionArgs with correct DF 51 fields.
    fn make_args(args: Vec<ColumnarValue>, return_type: DataType) -> ScalarFunctionArgs {
        let arg_fields: Vec<Arc<Field>> = args.iter().enumerate().map(|(i, _)| {
            Arc::new(Field::new(format!("arg{i}"), DataType::Binary, true))
        }).collect();
        ScalarFunctionArgs {
            args,
            arg_fields,
            number_rows: 1,
            return_field: Arc::new(Field::new("result", return_type, true)),
            config_options: Arc::new(ConfigOptions::default()),
        }
    }

    #[test]
    fn test_container_udf_registration() {
        let ctx = SessionContext::new();
        register_container_udfs(&ctx);
        // Verify all 9 UDFs are registered
        let names = [
            "belichtung", "bundle", "cascade_filter", "word_diff",
            "mexican_hat", "container_hamming", "container_similarity",
            "container_popcount", "container_xor",
        ];
        let state = ctx.state();
        let udfs = state.scalar_functions();
        for name in &names {
            assert!(
                udfs.contains_key(*name),
                "UDF '{}' not registered", name,
            );
        }
    }

    #[test]
    fn test_belichtung_scalar() {
        let a = Container::random(42);
        let b = Container::random(43);
        let expected = belichtungsmesser(&a, &b);

        let udf = BelichtungUdf::new();
        let a_sv = datafusion::scalar::ScalarValue::FixedSizeBinary(
            C_BYTES, Some(a.as_bytes().to_vec()),
        );
        let b_sv = datafusion::scalar::ScalarValue::FixedSizeBinary(
            C_BYTES, Some(b.as_bytes().to_vec()),
        );
        let result = udf.invoke_with_args(make_args(
            vec![
                ColumnarValue::Scalar(a_sv),
                ColumnarValue::Scalar(b_sv),
            ],
            DataType::UInt32,
        )).unwrap();

        match result {
            ColumnarValue::Scalar(datafusion::scalar::ScalarValue::UInt32(Some(v))) => {
                assert_eq!(v, expected);
            }
            _ => panic!("Expected UInt32 scalar"),
        }
    }

    #[test]
    fn test_container_hamming_scalar() {
        let a = Container::random(10);
        let b = Container::random(20);
        let expected = a.hamming(&b);

        let udf = ContainerHammingUdf::new();
        let a_sv = datafusion::scalar::ScalarValue::FixedSizeBinary(
            C_BYTES, Some(a.as_bytes().to_vec()),
        );
        let b_sv = datafusion::scalar::ScalarValue::FixedSizeBinary(
            C_BYTES, Some(b.as_bytes().to_vec()),
        );
        let result = udf.invoke_with_args(make_args(
            vec![
                ColumnarValue::Scalar(a_sv),
                ColumnarValue::Scalar(b_sv),
            ],
            DataType::UInt32,
        )).unwrap();

        match result {
            ColumnarValue::Scalar(datafusion::scalar::ScalarValue::UInt32(Some(v))) => {
                assert_eq!(v, expected);
            }
            _ => panic!("Expected UInt32 scalar"),
        }
    }

    #[test]
    fn test_mexican_hat_scalar() {
        let hat = MexicanHat::default_8k();
        let udf = MexicanHatUdf::new();

        // Test excitation zone (distance < 45)
        let result = udf.invoke_with_args(make_args(
            vec![ColumnarValue::Scalar(datafusion::scalar::ScalarValue::UInt32(Some(20)))],
            DataType::Float32,
        )).unwrap();

        match result {
            ColumnarValue::Scalar(datafusion::scalar::ScalarValue::Float32(Some(v))) => {
                let expected = hat.response(20);
                assert!((v - expected).abs() < 0.001);
            }
            _ => panic!("Expected Float32 scalar"),
        }
    }

    #[test]
    fn test_bundle_scalar() {
        let a = Container::random(100);
        let b = Container::random(200);
        let expected = Container::bundle(&[&a, &b]);

        let udf = BundleUdf::new();
        let a_sv = datafusion::scalar::ScalarValue::FixedSizeBinary(
            C_BYTES, Some(a.as_bytes().to_vec()),
        );
        let b_sv = datafusion::scalar::ScalarValue::FixedSizeBinary(
            C_BYTES, Some(b.as_bytes().to_vec()),
        );
        let result = udf.invoke_with_args(make_args(
            vec![
                ColumnarValue::Scalar(a_sv),
                ColumnarValue::Scalar(b_sv),
            ],
            DataType::FixedSizeBinary(C_BYTES),
        )).unwrap();

        match result {
            ColumnarValue::Scalar(datafusion::scalar::ScalarValue::FixedSizeBinary(_, Some(bytes))) => {
                let result_c = Container::from_bytes(&bytes[..CONTAINER_BYTES].try_into().unwrap());
                assert_eq!(result_c, expected);
            }
            _ => panic!("Expected FixedSizeBinary scalar"),
        }
    }

    #[test]
    fn test_cascade_filter_scalar() {
        let query = Container::random(1);
        let candidate = Container::random(1); // same seed = identical = distance 0

        let udf = CascadeFilterUdf::new();
        let q_sv = datafusion::scalar::ScalarValue::FixedSizeBinary(
            C_BYTES, Some(query.as_bytes().to_vec()),
        );
        let c_sv = datafusion::scalar::ScalarValue::FixedSizeBinary(
            C_BYTES, Some(candidate.as_bytes().to_vec()),
        );
        let result = udf.invoke_with_args(make_args(
            vec![
                ColumnarValue::Scalar(q_sv),
                ColumnarValue::Scalar(c_sv),
                ColumnarValue::Scalar(datafusion::scalar::ScalarValue::UInt32(Some(100))),
            ],
            DataType::UInt32,
        )).unwrap();

        match result {
            ColumnarValue::Scalar(datafusion::scalar::ScalarValue::UInt32(Some(v))) => {
                assert_eq!(v, 0, "Identical containers should have distance 0");
            }
            _ => panic!("Expected UInt32 scalar"),
        }
    }
}
