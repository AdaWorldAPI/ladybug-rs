//! Cognitive UDFs for DataFusion
//!
//! Modern ScalarUDFImpl implementations for LadybugDB cognitive operations.
//!
//! UDFs:
//! - `hamming(a, b)` -> Hamming distance (0-16384)
//! - `similarity(a, b)` -> Similarity (0.0-1.0)
//! - `popcount(x)` -> Bit count
//! - `xor_bind(a, b)` -> XOR binding (VSA)
//! - `extract_scent(fp)` -> 5-byte scent extraction
//! - `scent_distance(a, b)` -> Scent Hamming distance (0-40)
//!
//! NARS UDFs:
//! - `nars_deduction(f1, c1, f2, c2)` -> Deduction truth function
//! - `nars_induction(f1, c1, f2, c2)` -> Induction truth function
//! - `nars_abduction(f1, c1, f2, c2)` -> Abduction truth function
//! - `nars_revision(f1, c1, f2, c2)` -> Revision truth function

use std::any::Any;
use std::sync::Arc;

use arrow::array::*;
use arrow::datatypes::{DataType, Field};
use datafusion::common::Result;
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignature, Volatility,
};

use crate::core::{DIM, scent_distance as core_scent_distance, extract_scent, SCENT_BYTES};

// =============================================================================
// CONSTANTS
// =============================================================================

/// Fingerprint size in bytes (2048 for 16K bits)
pub const FP_BYTES: usize = crate::FINGERPRINT_BYTES;

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Compute Hamming distance between two byte slices
#[inline]
pub fn hamming_bytes(a: &[u8], b: &[u8]) -> u32 {
    let min_len = a.len().min(b.len());
    let mut dist: u32 = 0;

    // Process 8 bytes at a time
    let chunks = min_len / 8;
    for i in 0..chunks {
        let offset = i * 8;
        let a_u64 = u64::from_le_bytes(a[offset..offset + 8].try_into().unwrap());
        let b_u64 = u64::from_le_bytes(b[offset..offset + 8].try_into().unwrap());
        dist += (a_u64 ^ b_u64).count_ones();
    }

    // Remaining bytes
    for i in (chunks * 8)..min_len {
        dist += (a[i] ^ b[i]).count_ones();
    }

    dist
}

/// Compute similarity from Hamming distance
#[inline]
fn similarity_from_distance(dist: u32) -> f32 {
    1.0 - (dist as f32 / DIM as f32)
}

/// Compute popcount of a byte slice
#[inline]
fn popcount_bytes(data: &[u8]) -> u32 {
    let mut count = 0u32;
    for chunk in data.chunks(8) {
        if chunk.len() == 8 {
            let val = u64::from_le_bytes(chunk.try_into().unwrap());
            count += val.count_ones();
        } else {
            for &b in chunk {
                count += b.count_ones();
            }
        }
    }
    count
}

/// XOR two byte slices
#[inline]
fn xor_bytes(a: &[u8], b: &[u8]) -> Vec<u8> {
    a.iter().zip(b.iter()).map(|(x, y)| x ^ y).collect()
}

fn scalar_to_bytes(s: &datafusion::scalar::ScalarValue) -> Result<Vec<u8>> {
    match s {
        datafusion::scalar::ScalarValue::Binary(Some(b)) => Ok(b.clone()),
        datafusion::scalar::ScalarValue::LargeBinary(Some(b)) => Ok(b.clone()),
        datafusion::scalar::ScalarValue::FixedSizeBinary(_, Some(b)) => Ok(b.clone()),
        _ => Err(datafusion::error::DataFusionError::Execution(
            "Expected binary scalar".into(),
        )),
    }
}

fn expand_to_arrays(a: &ColumnarValue, b: &ColumnarValue) -> Result<(ArrayRef, ArrayRef)> {
    match (a, b) {
        (ColumnarValue::Array(arr), ColumnarValue::Scalar(s)) => {
            let len = arr.len();
            let expanded = s.to_array_of_size(len)?;
            Ok((arr.clone(), expanded))
        }
        (ColumnarValue::Scalar(s), ColumnarValue::Array(arr)) => {
            let len = arr.len();
            let expanded = s.to_array_of_size(len)?;
            Ok((expanded, arr.clone()))
        }
        (ColumnarValue::Array(a), ColumnarValue::Array(b)) => Ok((a.clone(), b.clone())),
        _ => Err(datafusion::error::DataFusionError::Execution(
            "Unexpected argument combination".into(),
        )),
    }
}

// =============================================================================
// HAMMING DISTANCE UDF
// =============================================================================

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct HammingUdf {
    signature: Signature,
}

impl HammingUdf {
    pub fn new() -> Self {
        Self {
            signature: Signature::one_of(
                vec![
                    TypeSignature::Exact(vec![DataType::Binary, DataType::Binary]),
                    TypeSignature::Exact(vec![DataType::LargeBinary, DataType::LargeBinary]),
                    TypeSignature::Exact(vec![
                        DataType::FixedSizeBinary(FP_BYTES as i32),
                        DataType::FixedSizeBinary(FP_BYTES as i32),
                    ]),
                    TypeSignature::Any(2),
                ],
                Volatility::Immutable,
            ),
        }
    }

    fn hamming_array(&self, a: ArrayRef, b: ArrayRef) -> Result<ArrayRef> {
        let len = a.len();
        let mut builder = UInt32Builder::with_capacity(len);

        if let (Some(a_fixed), Some(b_fixed)) = (
            a.as_any().downcast_ref::<FixedSizeBinaryArray>(),
            b.as_any().downcast_ref::<FixedSizeBinaryArray>(),
        ) {
            for i in 0..len {
                if a_fixed.is_null(i) || b_fixed.is_null(i) {
                    builder.append_null();
                } else {
                    let dist = hamming_bytes(a_fixed.value(i), b_fixed.value(i));
                    builder.append_value(dist);
                }
            }
            return Ok(Arc::new(builder.finish()));
        }

        if let (Some(a_bin), Some(b_bin)) = (
            a.as_any().downcast_ref::<BinaryArray>(),
            b.as_any().downcast_ref::<BinaryArray>(),
        ) {
            for i in 0..len {
                if a_bin.is_null(i) || b_bin.is_null(i) {
                    builder.append_null();
                } else {
                    let dist = hamming_bytes(a_bin.value(i), b_bin.value(i));
                    builder.append_value(dist);
                }
            }
            return Ok(Arc::new(builder.finish()));
        }

        if let (Some(a_large), Some(b_large)) = (
            a.as_any().downcast_ref::<LargeBinaryArray>(),
            b.as_any().downcast_ref::<LargeBinaryArray>(),
        ) {
            for i in 0..len {
                if a_large.is_null(i) || b_large.is_null(i) {
                    builder.append_null();
                } else {
                    let dist = hamming_bytes(a_large.value(i), b_large.value(i));
                    builder.append_value(dist);
                }
            }
            return Ok(Arc::new(builder.finish()));
        }

        Err(datafusion::error::DataFusionError::Execution(
            "hamming requires binary arrays".into(),
        ))
    }
}

impl Default for HammingUdf {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for HammingUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "hamming"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::UInt32)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let args = &args.args;
        match (&args[0], &args[1]) {
            (ColumnarValue::Array(a), ColumnarValue::Array(b)) => {
                let result = self.hamming_array(a.clone(), b.clone())?;
                Ok(ColumnarValue::Array(result))
            }
            (ColumnarValue::Scalar(a), ColumnarValue::Scalar(b)) => {
                let a_bytes = scalar_to_bytes(a)?;
                let b_bytes = scalar_to_bytes(b)?;
                let dist = hamming_bytes(&a_bytes, &b_bytes);
                Ok(ColumnarValue::Scalar(
                    datafusion::scalar::ScalarValue::UInt32(Some(dist)),
                ))
            }
            _ => {
                let (a_arr, b_arr) = expand_to_arrays(&args[0], &args[1])?;
                let result = self.hamming_array(a_arr, b_arr)?;
                Ok(ColumnarValue::Array(result))
            }
        }
    }
}

// =============================================================================
// SIMILARITY UDF
// =============================================================================

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SimilarityUdf {
    signature: Signature,
}

impl SimilarityUdf {
    pub fn new() -> Self {
        Self {
            signature: Signature::one_of(
                vec![
                    TypeSignature::Exact(vec![DataType::Binary, DataType::Binary]),
                    TypeSignature::Exact(vec![DataType::LargeBinary, DataType::LargeBinary]),
                    TypeSignature::Exact(vec![
                        DataType::FixedSizeBinary(FP_BYTES as i32),
                        DataType::FixedSizeBinary(FP_BYTES as i32),
                    ]),
                    TypeSignature::Any(2),
                ],
                Volatility::Immutable,
            ),
        }
    }

    fn similarity_array(&self, a: ArrayRef, b: ArrayRef) -> Result<ArrayRef> {
        let len = a.len();
        let mut builder = Float32Builder::with_capacity(len);

        if let (Some(a_fixed), Some(b_fixed)) = (
            a.as_any().downcast_ref::<FixedSizeBinaryArray>(),
            b.as_any().downcast_ref::<FixedSizeBinaryArray>(),
        ) {
            for i in 0..len {
                if a_fixed.is_null(i) || b_fixed.is_null(i) {
                    builder.append_null();
                } else {
                    let dist = hamming_bytes(a_fixed.value(i), b_fixed.value(i));
                    builder.append_value(similarity_from_distance(dist));
                }
            }
            return Ok(Arc::new(builder.finish()));
        }

        if let (Some(a_bin), Some(b_bin)) = (
            a.as_any().downcast_ref::<BinaryArray>(),
            b.as_any().downcast_ref::<BinaryArray>(),
        ) {
            for i in 0..len {
                if a_bin.is_null(i) || b_bin.is_null(i) {
                    builder.append_null();
                } else {
                    let dist = hamming_bytes(a_bin.value(i), b_bin.value(i));
                    builder.append_value(similarity_from_distance(dist));
                }
            }
            return Ok(Arc::new(builder.finish()));
        }

        Err(datafusion::error::DataFusionError::Execution(
            "similarity requires binary arrays".into(),
        ))
    }
}

impl Default for SimilarityUdf {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for SimilarityUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "similarity"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float32)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let args = &args.args;
        match (&args[0], &args[1]) {
            (ColumnarValue::Array(a), ColumnarValue::Array(b)) => {
                let result = self.similarity_array(a.clone(), b.clone())?;
                Ok(ColumnarValue::Array(result))
            }
            (ColumnarValue::Scalar(a), ColumnarValue::Scalar(b)) => {
                let a_bytes = scalar_to_bytes(a)?;
                let b_bytes = scalar_to_bytes(b)?;
                let dist = hamming_bytes(&a_bytes, &b_bytes);
                let sim = similarity_from_distance(dist);
                Ok(ColumnarValue::Scalar(
                    datafusion::scalar::ScalarValue::Float32(Some(sim)),
                ))
            }
            _ => {
                let (a_arr, b_arr) = expand_to_arrays(&args[0], &args[1])?;
                let result = self.similarity_array(a_arr, b_arr)?;
                Ok(ColumnarValue::Array(result))
            }
        }
    }
}

// =============================================================================
// POPCOUNT UDF
// =============================================================================

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct PopcountUdf {
    signature: Signature,
}

impl PopcountUdf {
    pub fn new() -> Self {
        Self {
            signature: Signature::one_of(
                vec![
                    TypeSignature::Exact(vec![DataType::Binary]),
                    TypeSignature::Exact(vec![DataType::LargeBinary]),
                    TypeSignature::Exact(vec![DataType::UInt64]),
                    TypeSignature::Any(1),
                ],
                Volatility::Immutable,
            ),
        }
    }

    fn popcount_array(&self, arr: ArrayRef) -> Result<ArrayRef> {
        let len = arr.len();
        let mut builder = UInt32Builder::with_capacity(len);

        if let Some(u64_arr) = arr.as_any().downcast_ref::<UInt64Array>() {
            for i in 0..len {
                if u64_arr.is_null(i) {
                    builder.append_null();
                } else {
                    builder.append_value(u64_arr.value(i).count_ones());
                }
            }
            return Ok(Arc::new(builder.finish()));
        }

        if let Some(bin_arr) = arr.as_any().downcast_ref::<BinaryArray>() {
            for i in 0..len {
                if bin_arr.is_null(i) {
                    builder.append_null();
                } else {
                    builder.append_value(popcount_bytes(bin_arr.value(i)));
                }
            }
            return Ok(Arc::new(builder.finish()));
        }

        if let Some(fixed_arr) = arr.as_any().downcast_ref::<FixedSizeBinaryArray>() {
            for i in 0..len {
                if fixed_arr.is_null(i) {
                    builder.append_null();
                } else {
                    builder.append_value(popcount_bytes(fixed_arr.value(i)));
                }
            }
            return Ok(Arc::new(builder.finish()));
        }

        Err(datafusion::error::DataFusionError::Execution(
            "popcount requires binary or uint64 array".into(),
        ))
    }
}

impl Default for PopcountUdf {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for PopcountUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "popcount"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::UInt32)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let args = &args.args;
        match &args[0] {
            ColumnarValue::Array(arr) => {
                let result = self.popcount_array(arr.clone())?;
                Ok(ColumnarValue::Array(result))
            }
            ColumnarValue::Scalar(s) => {
                let count = match s {
                    datafusion::scalar::ScalarValue::UInt64(Some(n)) => n.count_ones(),
                    datafusion::scalar::ScalarValue::Binary(Some(b)) => popcount_bytes(b),
                    datafusion::scalar::ScalarValue::LargeBinary(Some(b)) => popcount_bytes(b),
                    datafusion::scalar::ScalarValue::FixedSizeBinary(_, Some(b)) => popcount_bytes(b),
                    _ => {
                        return Err(datafusion::error::DataFusionError::Execution(
                            "popcount requires binary or uint64".into(),
                        ))
                    }
                };
                Ok(ColumnarValue::Scalar(
                    datafusion::scalar::ScalarValue::UInt32(Some(count)),
                ))
            }
        }
    }
}

// =============================================================================
// XOR_BIND UDF
// =============================================================================

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct XorBindUdf {
    signature: Signature,
}

impl XorBindUdf {
    pub fn new() -> Self {
        Self {
            signature: Signature::one_of(
                vec![
                    TypeSignature::Exact(vec![DataType::Binary, DataType::Binary]),
                    TypeSignature::Exact(vec![DataType::LargeBinary, DataType::LargeBinary]),
                    TypeSignature::Any(2),
                ],
                Volatility::Immutable,
            ),
        }
    }

    fn xor_array(&self, a: ArrayRef, b: ArrayRef) -> Result<ArrayRef> {
        let len = a.len();

        if let (Some(a_fixed), Some(b_fixed)) = (
            a.as_any().downcast_ref::<FixedSizeBinaryArray>(),
            b.as_any().downcast_ref::<FixedSizeBinaryArray>(),
        ) {
            let size = a_fixed.value_length() as usize;
            let mut builder = FixedSizeBinaryBuilder::with_capacity(len, size as i32);
            for i in 0..len {
                if a_fixed.is_null(i) || b_fixed.is_null(i) {
                    builder.append_null();
                } else {
                    let result = xor_bytes(a_fixed.value(i), b_fixed.value(i));
                    builder.append_value(&result)?;
                }
            }
            return Ok(Arc::new(builder.finish()));
        }

        if let (Some(a_bin), Some(b_bin)) = (
            a.as_any().downcast_ref::<BinaryArray>(),
            b.as_any().downcast_ref::<BinaryArray>(),
        ) {
            let mut builder = BinaryBuilder::with_capacity(len, len * FP_BYTES);
            for i in 0..len {
                if a_bin.is_null(i) || b_bin.is_null(i) {
                    builder.append_null();
                } else {
                    let result = xor_bytes(a_bin.value(i), b_bin.value(i));
                    builder.append_value(&result);
                }
            }
            return Ok(Arc::new(builder.finish()));
        }

        Err(datafusion::error::DataFusionError::Execution(
            "xor_bind requires binary arrays".into(),
        ))
    }
}

impl Default for XorBindUdf {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for XorBindUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "xor_bind"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        Ok(arg_types[0].clone())
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let args = &args.args;
        match (&args[0], &args[1]) {
            (ColumnarValue::Array(a), ColumnarValue::Array(b)) => {
                let result = self.xor_array(a.clone(), b.clone())?;
                Ok(ColumnarValue::Array(result))
            }
            (ColumnarValue::Scalar(a), ColumnarValue::Scalar(b)) => {
                let a_bytes = scalar_to_bytes(a)?;
                let b_bytes = scalar_to_bytes(b)?;
                let result = xor_bytes(&a_bytes, &b_bytes);
                Ok(ColumnarValue::Scalar(
                    datafusion::scalar::ScalarValue::Binary(Some(result)),
                ))
            }
            _ => {
                let (a_arr, b_arr) = expand_to_arrays(&args[0], &args[1])?;
                let result = self.xor_array(a_arr, b_arr)?;
                Ok(ColumnarValue::Array(result))
            }
        }
    }
}

// =============================================================================
// SCENT UDFs
// =============================================================================

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct ExtractScentUdf {
    signature: Signature,
}

impl ExtractScentUdf {
    pub fn new() -> Self {
        Self {
            signature: Signature::one_of(
                vec![
                    TypeSignature::Exact(vec![DataType::Binary]),
                    TypeSignature::Exact(vec![DataType::FixedSizeBinary(FP_BYTES as i32)]),
                    TypeSignature::Any(1),
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl Default for ExtractScentUdf {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for ExtractScentUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "extract_scent"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::FixedSizeBinary(SCENT_BYTES as i32))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let args = &args.args;
        match &args[0] {
            ColumnarValue::Array(arr) => {
                let len = arr.len();
                let mut builder = FixedSizeBinaryBuilder::with_capacity(len, SCENT_BYTES as i32);

                if let Some(fixed) = arr.as_any().downcast_ref::<FixedSizeBinaryArray>() {
                    for i in 0..len {
                        if fixed.is_null(i) {
                            builder.append_null();
                        } else {
                            let scent = extract_scent(fixed.value(i));
                            builder.append_value(&scent)?;
                        }
                    }
                } else if let Some(bin) = arr.as_any().downcast_ref::<BinaryArray>() {
                    for i in 0..len {
                        if bin.is_null(i) {
                            builder.append_null();
                        } else {
                            let scent = extract_scent(bin.value(i));
                            builder.append_value(&scent)?;
                        }
                    }
                } else {
                    return Err(datafusion::error::DataFusionError::Execution(
                        "extract_scent requires binary array".into(),
                    ));
                }

                Ok(ColumnarValue::Array(Arc::new(builder.finish())))
            }
            ColumnarValue::Scalar(s) => {
                let bytes = scalar_to_bytes(s)?;
                let scent = extract_scent(&bytes);
                Ok(ColumnarValue::Scalar(
                    datafusion::scalar::ScalarValue::FixedSizeBinary(
                        SCENT_BYTES as i32,
                        Some(scent.to_vec()),
                    ),
                ))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct ScentDistanceUdf {
    signature: Signature,
}

impl ScentDistanceUdf {
    pub fn new() -> Self {
        Self {
            signature: Signature::one_of(
                vec![
                    TypeSignature::Exact(vec![
                        DataType::FixedSizeBinary(SCENT_BYTES as i32),
                        DataType::FixedSizeBinary(SCENT_BYTES as i32),
                    ]),
                    TypeSignature::Any(2),
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl Default for ScentDistanceUdf {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for ScentDistanceUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "scent_distance"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::UInt32)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let args = &args.args;
        match (&args[0], &args[1]) {
            (ColumnarValue::Array(a), ColumnarValue::Array(b)) => {
                let len = a.len();
                let mut builder = UInt32Builder::with_capacity(len);

                let a_fixed = a.as_any().downcast_ref::<FixedSizeBinaryArray>();
                let b_fixed = b.as_any().downcast_ref::<FixedSizeBinaryArray>();

                if let (Some(a), Some(b)) = (a_fixed, b_fixed) {
                    for i in 0..len {
                        if a.is_null(i) || b.is_null(i) {
                            builder.append_null();
                        } else {
                            let a_scent: [u8; SCENT_BYTES] =
                                a.value(i).try_into().unwrap_or([0; SCENT_BYTES]);
                            let b_scent: [u8; SCENT_BYTES] =
                                b.value(i).try_into().unwrap_or([0; SCENT_BYTES]);
                            builder.append_value(core_scent_distance(&a_scent, &b_scent));
                        }
                    }
                    return Ok(ColumnarValue::Array(Arc::new(builder.finish())));
                }

                Err(datafusion::error::DataFusionError::Execution(
                    "scent_distance requires fixed size binary arrays".into(),
                ))
            }
            (ColumnarValue::Scalar(a), ColumnarValue::Scalar(b)) => {
                let a_bytes = scalar_to_bytes(a)?;
                let b_bytes = scalar_to_bytes(b)?;
                let a_scent: [u8; SCENT_BYTES] =
                    a_bytes.as_slice().try_into().unwrap_or([0; SCENT_BYTES]);
                let b_scent: [u8; SCENT_BYTES] =
                    b_bytes.as_slice().try_into().unwrap_or([0; SCENT_BYTES]);
                let dist = core_scent_distance(&a_scent, &b_scent);
                Ok(ColumnarValue::Scalar(
                    datafusion::scalar::ScalarValue::UInt32(Some(dist)),
                ))
            }
            _ => Err(datafusion::error::DataFusionError::Execution(
                "scent_distance requires matching argument types".into(),
            )),
        }
    }
}

// =============================================================================
// NARS TRUTH FUNCTION UDFs
// =============================================================================

/// NARS Deduction: A->B, B->C |- A->C
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct NarsDeductionUdf {
    signature: Signature,
}

impl NarsDeductionUdf {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![
                    DataType::Float64,
                    DataType::Float64,
                    DataType::Float64,
                    DataType::Float64,
                ],
                Volatility::Immutable,
            ),
        }
    }

    #[inline]
    fn deduction(f1: f64, c1: f64, f2: f64, c2: f64) -> (f64, f64) {
        let f = f1 * f2;
        let c = f1 * f2 * c1 * c2;
        (f, c)
    }
}

impl Default for NarsDeductionUdf {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for NarsDeductionUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "nars_deduction"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Struct(
            vec![
                Field::new("f", DataType::Float64, false),
                Field::new("c", DataType::Float64, false),
            ]
            .into(),
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let (f1, c1, f2, c2) = extract_nars_args(&args.args)?;
        let (f, c) = Self::deduction(f1, c1, f2, c2);
        Ok(nars_result_scalar(f, c))
    }
}

/// NARS Induction: A->B, A->C |- B->C
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct NarsInductionUdf {
    signature: Signature,
}

impl NarsInductionUdf {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![
                    DataType::Float64,
                    DataType::Float64,
                    DataType::Float64,
                    DataType::Float64,
                ],
                Volatility::Immutable,
            ),
        }
    }

    #[inline]
    fn induction(f1: f64, c1: f64, f2: f64, c2: f64) -> (f64, f64) {
        let f = f2;
        let w_plus = f1 * c1 * f2 * c2;
        let w = f1 * c1 * c2;
        let c = if w > 0.0 { w_plus / w } else { 0.0 };
        (f, c.min(0.99))
    }
}

impl Default for NarsInductionUdf {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for NarsInductionUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "nars_induction"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Struct(
            vec![
                Field::new("f", DataType::Float64, false),
                Field::new("c", DataType::Float64, false),
            ]
            .into(),
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let (f1, c1, f2, c2) = extract_nars_args(&args.args)?;
        let (f, c) = Self::induction(f1, c1, f2, c2);
        Ok(nars_result_scalar(f, c))
    }
}

/// NARS Abduction: A->B, C->B |- A->C
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct NarsAbductionUdf {
    signature: Signature,
}

impl NarsAbductionUdf {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![
                    DataType::Float64,
                    DataType::Float64,
                    DataType::Float64,
                    DataType::Float64,
                ],
                Volatility::Immutable,
            ),
        }
    }

    #[inline]
    fn abduction(f1: f64, c1: f64, f2: f64, c2: f64) -> (f64, f64) {
        let f = f1;
        let w_plus = f1 * c1 * f2 * c2;
        let w = f2 * c1 * c2;
        let c = if w > 0.0 { w_plus / w } else { 0.0 };
        (f, c.min(0.99))
    }
}

impl Default for NarsAbductionUdf {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for NarsAbductionUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "nars_abduction"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Struct(
            vec![
                Field::new("f", DataType::Float64, false),
                Field::new("c", DataType::Float64, false),
            ]
            .into(),
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let (f1, c1, f2, c2) = extract_nars_args(&args.args)?;
        let (f, c) = Self::abduction(f1, c1, f2, c2);
        Ok(nars_result_scalar(f, c))
    }
}

/// NARS Revision: Combine evidence
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct NarsRevisionUdf {
    signature: Signature,
}

impl NarsRevisionUdf {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![
                    DataType::Float64,
                    DataType::Float64,
                    DataType::Float64,
                    DataType::Float64,
                ],
                Volatility::Immutable,
            ),
        }
    }

    #[inline]
    fn revision(f1: f64, c1: f64, f2: f64, c2: f64) -> (f64, f64) {
        let k = 1.0;
        let w1 = c1 / (1.0 - c1);
        let w2 = c2 / (1.0 - c2);
        let w = w1 + w2;
        let f = if w > 0.0 {
            (w1 * f1 + w2 * f2) / w
        } else {
            (f1 + f2) / 2.0
        };
        let c = w / (w + k);
        (f, c.min(0.99))
    }
}

impl Default for NarsRevisionUdf {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for NarsRevisionUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "nars_revision"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Struct(
            vec![
                Field::new("f", DataType::Float64, false),
                Field::new("c", DataType::Float64, false),
            ]
            .into(),
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let (f1, c1, f2, c2) = extract_nars_args(&args.args)?;
        let (f, c) = Self::revision(f1, c1, f2, c2);
        Ok(nars_result_scalar(f, c))
    }
}

// =============================================================================
// HELPER FUNCTIONS FOR NARS
// =============================================================================

fn extract_nars_args(args: &[ColumnarValue]) -> Result<(f64, f64, f64, f64)> {
    let f1 = match &args[0] {
        ColumnarValue::Scalar(datafusion::scalar::ScalarValue::Float64(Some(v))) => *v,
        _ => {
            return Err(datafusion::error::DataFusionError::Execution(
                "NARS functions require Float64 scalars".into(),
            ))
        }
    };
    let c1 = match &args[1] {
        ColumnarValue::Scalar(datafusion::scalar::ScalarValue::Float64(Some(v))) => *v,
        _ => {
            return Err(datafusion::error::DataFusionError::Execution(
                "NARS functions require Float64 scalars".into(),
            ))
        }
    };
    let f2 = match &args[2] {
        ColumnarValue::Scalar(datafusion::scalar::ScalarValue::Float64(Some(v))) => *v,
        _ => {
            return Err(datafusion::error::DataFusionError::Execution(
                "NARS functions require Float64 scalars".into(),
            ))
        }
    };
    let c2 = match &args[3] {
        ColumnarValue::Scalar(datafusion::scalar::ScalarValue::Float64(Some(v))) => *v,
        _ => {
            return Err(datafusion::error::DataFusionError::Execution(
                "NARS functions require Float64 scalars".into(),
            ))
        }
    };
    Ok((f1, c1, f2, c2))
}

fn nars_result_scalar(f: f64, c: f64) -> ColumnarValue {
    use datafusion::scalar::ScalarValue;

    let f_field = Field::new("f", DataType::Float64, false);
    let c_field = Field::new("c", DataType::Float64, false);

    ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(StructArray::from(vec![
        (
            Arc::new(f_field),
            Arc::new(Float64Array::from(vec![f])) as ArrayRef,
        ),
        (
            Arc::new(c_field),
            Arc::new(Float64Array::from(vec![c])) as ArrayRef,
        ),
    ]))))
}

// =============================================================================
// MEMBRANE UDFs (Sigma-10 Consciousness Encoding)
// =============================================================================

use crate::cognitive::membrane::{Membrane, ConsciousnessParams};

/// Membrane Encode: tau, sigma, qualia -> 10K-bit fingerprint
///
/// Encodes consciousness parameters into a searchable fingerprint:
/// - tau (temporal) -> bits 0-3333
/// - sigma (signal) -> bits 3334-6666
/// - qualia (semantic) -> bits 6667-9999
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct MembraneEncodeUdf {
    signature: Signature,
}

impl MembraneEncodeUdf {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![DataType::Float32, DataType::Float32, DataType::Utf8],
                Volatility::Immutable,
            ),
        }
    }
}

impl Default for MembraneEncodeUdf {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for MembraneEncodeUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "membrane_encode"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::FixedSizeBinary(FP_BYTES as i32))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let args = &args.args;
        // Extract scalar arguments
        let tau = match &args[0] {
            ColumnarValue::Scalar(datafusion::scalar::ScalarValue::Float32(Some(v))) => *v,
            _ => return Err(datafusion::error::DataFusionError::Execution(
                "membrane_encode requires Float32 for tau".into(),
            )),
        };
        let sigma = match &args[1] {
            ColumnarValue::Scalar(datafusion::scalar::ScalarValue::Float32(Some(v))) => *v,
            _ => return Err(datafusion::error::DataFusionError::Execution(
                "membrane_encode requires Float32 for sigma".into(),
            )),
        };
        let qualia = match &args[2] {
            ColumnarValue::Scalar(datafusion::scalar::ScalarValue::Utf8(Some(s))) => s.clone(),
            _ => return Err(datafusion::error::DataFusionError::Execution(
                "membrane_encode requires Utf8 for qualia".into(),
            )),
        };

        // Encode using membrane
        let mut membrane = Membrane::default();
        let params = ConsciousnessParams::new(tau, sigma, qualia);
        let fp = membrane.encode(&params);
        let bytes = fp.to_bytes();

        Ok(ColumnarValue::Scalar(
            datafusion::scalar::ScalarValue::FixedSizeBinary(
                FP_BYTES as i32,
                Some(bytes),
            ),
        ))
    }
}

/// Membrane Decode: 10K-bit fingerprint -> tau, sigma, qualia_placeholder
///
/// Decodes a consciousness fingerprint back to approximate parameters.
/// Note: qualia cannot be recovered (hash is one-way).
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct MembraneDecodeUdf {
    signature: Signature,
}

impl MembraneDecodeUdf {
    pub fn new() -> Self {
        Self {
            signature: Signature::one_of(
                vec![
                    TypeSignature::Exact(vec![DataType::FixedSizeBinary(FP_BYTES as i32)]),
                    TypeSignature::Exact(vec![DataType::Binary]),
                    TypeSignature::Any(1),
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl Default for MembraneDecodeUdf {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for MembraneDecodeUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "membrane_decode"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Struct(
            vec![
                Field::new("tau", DataType::Float32, false),
                Field::new("sigma", DataType::Float32, false),
                Field::new("qualia", DataType::Utf8, false),
            ]
            .into(),
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let args = &args.args;
        let bytes = match &args[0] {
            ColumnarValue::Scalar(s) => scalar_to_bytes(s)?,
            _ => return Err(datafusion::error::DataFusionError::Execution(
                "membrane_decode requires scalar binary".into(),
            )),
        };

        // Decode using membrane
        let fp = crate::core::Fingerprint::from_bytes(&bytes).map_err(|e| {
            datafusion::error::DataFusionError::Execution(format!("Invalid fingerprint: {}", e))
        })?;
        let mut membrane = Membrane::default();
        let params = membrane.decode(&fp);

        // Build result struct
        use datafusion::scalar::ScalarValue;
        let tau_field = Field::new("tau", DataType::Float32, false);
        let sigma_field = Field::new("sigma", DataType::Float32, false);
        let qualia_field = Field::new("qualia", DataType::Utf8, false);

        Ok(ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(
            StructArray::from(vec![
                (
                    Arc::new(tau_field),
                    Arc::new(Float32Array::from(vec![params.tau])) as ArrayRef,
                ),
                (
                    Arc::new(sigma_field),
                    Arc::new(Float32Array::from(vec![params.sigma])) as ArrayRef,
                ),
                (
                    Arc::new(qualia_field),
                    Arc::new(StringArray::from(vec![params.qualia.as_str()])) as ArrayRef,
                ),
            ]),
        ))))
    }
}

// =============================================================================
// UDF REGISTRATION
// =============================================================================

use datafusion::execution::context::SessionContext;
use datafusion::logical_expr::ScalarUDF;

/// Register all cognitive UDFs with a DataFusion context
pub fn register_cognitive_udfs(ctx: &SessionContext) {
    // VSA operations
    ctx.register_udf(ScalarUDF::from(HammingUdf::new()));
    ctx.register_udf(ScalarUDF::from(SimilarityUdf::new()));
    ctx.register_udf(ScalarUDF::from(PopcountUdf::new()));
    ctx.register_udf(ScalarUDF::from(XorBindUdf::new()));

    // Scent Index support
    ctx.register_udf(ScalarUDF::from(ExtractScentUdf::new()));
    ctx.register_udf(ScalarUDF::from(ScentDistanceUdf::new()));

    // NARS inference
    ctx.register_udf(ScalarUDF::from(NarsDeductionUdf::new()));
    ctx.register_udf(ScalarUDF::from(NarsInductionUdf::new()));
    ctx.register_udf(ScalarUDF::from(NarsAbductionUdf::new()));
    ctx.register_udf(ScalarUDF::from(NarsRevisionUdf::new()));

    // Consciousness membrane
    ctx.register_udf(ScalarUDF::from(MembraneEncodeUdf::new()));
    ctx.register_udf(ScalarUDF::from(MembraneDecodeUdf::new()));
}

/// Create all UDFs as a vector
pub fn all_cognitive_udfs() -> Vec<ScalarUDF> {
    vec![
        ScalarUDF::from(HammingUdf::new()),
        ScalarUDF::from(SimilarityUdf::new()),
        ScalarUDF::from(PopcountUdf::new()),
        ScalarUDF::from(XorBindUdf::new()),
        ScalarUDF::from(ExtractScentUdf::new()),
        ScalarUDF::from(ScentDistanceUdf::new()),
        ScalarUDF::from(NarsDeductionUdf::new()),
        ScalarUDF::from(NarsInductionUdf::new()),
        ScalarUDF::from(NarsAbductionUdf::new()),
        ScalarUDF::from(NarsRevisionUdf::new()),
        ScalarUDF::from(MembraneEncodeUdf::new()),
        ScalarUDF::from(MembraneDecodeUdf::new()),
    ]
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::prelude::*;

    #[test]
    fn test_hamming_bytes() {
        let a = vec![0xFF, 0x00, 0xFF, 0x00];
        let b = vec![0x00, 0xFF, 0x00, 0xFF];
        assert_eq!(hamming_bytes(&a, &b), 32);

        let c = vec![0xFF; 8];
        let d = vec![0xFF; 8];
        assert_eq!(hamming_bytes(&c, &d), 0);
    }

    #[test]
    fn test_similarity() {
        assert_eq!(similarity_from_distance(0), 1.0);
        assert_eq!(similarity_from_distance(DIM as u32), 0.0);
        let mid = similarity_from_distance(DIM as u32 / 2);
        assert!((mid - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_popcount() {
        let data = vec![0xFF, 0xFF, 0xFF, 0xFF];
        assert_eq!(popcount_bytes(&data), 32);

        let zeros = vec![0x00; 8];
        assert_eq!(popcount_bytes(&zeros), 0);
    }

    #[test]
    fn test_xor() {
        let a = vec![0xFF, 0x00];
        let b = vec![0x00, 0xFF];
        let result = xor_bytes(&a, &b);
        assert_eq!(result, vec![0xFF, 0xFF]);
    }

    #[test]
    fn test_nars_deduction() {
        let (f, c) = NarsDeductionUdf::deduction(0.9, 0.8, 0.85, 0.75);
        assert!((f - 0.765).abs() < 0.01);
        assert!(c > 0.0 && c < 1.0);
    }

    #[test]
    fn test_nars_revision() {
        let (f, c) = NarsRevisionUdf::revision(0.8, 0.6, 0.9, 0.7);
        assert!(f > 0.8 && f < 0.9);
        assert!(c > 0.6 && c < 0.99);
    }

    #[tokio::test]
    async fn test_udf_registration() {
        let ctx = SessionContext::new();
        register_cognitive_udfs(&ctx);

        let state = ctx.state();
        let udfs = state.scalar_functions();
        assert!(udfs.contains_key("hamming"));
        assert!(udfs.contains_key("similarity"));
        assert!(udfs.contains_key("popcount"));
        assert!(udfs.contains_key("xor_bind"));
        assert!(udfs.contains_key("extract_scent"));
        assert!(udfs.contains_key("scent_distance"));
        assert!(udfs.contains_key("nars_deduction"));
    }
}
