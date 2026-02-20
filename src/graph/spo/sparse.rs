//! Sparse Container — bitmap + non-zero words encoding of a Container.
//!
//! At 30% density (typical), stores 320 bytes instead of 1024.
//! Three sparse containers fit in one content Container (960 bytes < 1024).

use ladybug_contract::container::{Container, CONTAINER_WORDS};

/// Error types for SPO operations.
#[derive(Clone, Debug)]
pub enum SpoError {
    BitmapWordMismatch { bitmap_ones: u32, word_count: usize },
    AxesOverflow { total: usize, max: usize },
    WrongGeometry,
    DuplicateDn { dn: u64 },
    DnNotFound { dn: u64 },
    EmptyChain,
}

impl std::fmt::Display for SpoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BitmapWordMismatch { bitmap_ones, word_count } =>
                write!(f, "Bitmap has {} ones but {} words supplied", bitmap_ones, word_count),
            Self::AxesOverflow { total, max } =>
                write!(f, "Axes too dense: {} words needed, max {}", total, max),
            Self::WrongGeometry =>
                write!(f, "Record geometry is not Spo"),
            Self::DuplicateDn { dn } =>
                write!(f, "DN {:#x} already exists", dn),
            Self::DnNotFound { dn } =>
                write!(f, "DN {:#x} not found", dn),
            Self::EmptyChain =>
                write!(f, "Chain is empty"),
        }
    }
}

impl std::error::Error for SpoError {}

// ============================================================================
// SPARSE CONTAINER
// ============================================================================

/// Sparse encoding of a Container: bitmap (which words are non-zero) + only those words.
///
/// # Invariants
/// - `bitmap[0].count_ones() + bitmap[1].count_ones() == words.len()`
/// - `to_dense()` produces exact original Container (lossless)
/// - `hamming_sparse(a, b) == a.to_dense().hamming(&b.to_dense())`
#[derive(Clone, Debug)]
pub struct SparseContainer {
    /// 128 bits: bit i set ↔ Container word i is non-zero and stored.
    pub bitmap: [u64; 2],
    /// Only the non-zero words, ordered by bit position.
    pub words: Vec<u64>,
}

impl SparseContainer {
    /// Construct with validation.
    pub fn new(bitmap: [u64; 2], words: Vec<u64>) -> Result<Self, SpoError> {
        let expected = bitmap[0].count_ones() + bitmap[1].count_ones();
        if words.len() != expected as usize {
            return Err(SpoError::BitmapWordMismatch {
                bitmap_ones: expected,
                word_count: words.len(),
            });
        }
        Ok(Self { bitmap, words })
    }

    /// Empty sparse container (all zeros).
    pub fn zero() -> Self {
        Self { bitmap: [0; 2], words: Vec::new() }
    }

    /// Number of stored (non-zero) words.
    #[inline]
    pub fn word_count(&self) -> usize {
        self.words.len()
    }

    /// Density: fraction of non-zero words (0.0 to 1.0).
    #[inline]
    pub fn density(&self) -> f32 {
        self.words.len() as f32 / CONTAINER_WORDS as f32
    }

    /// Is a specific Container word stored (non-zero)?
    #[inline]
    pub fn has_word(&self, index: usize) -> bool {
        debug_assert!(index < CONTAINER_WORDS);
        let half = index / 64;
        let bit = index % 64;
        self.bitmap[half] & (1u64 << bit) != 0
    }

    /// Get the value of Container word `index`. Returns 0 if not stored.
    pub fn get_word(&self, index: usize) -> u64 {
        debug_assert!(index < CONTAINER_WORDS);
        if !self.has_word(index) {
            return 0;
        }
        // Count how many bits are set before this position = index into words vec.
        let half = index / 64;
        let bit = index % 64;
        let mut rank = 0u32;
        if half > 0 {
            rank += self.bitmap[0].count_ones();
        }
        // Count bits set before `bit` in bitmap[half]
        let mask = if bit == 0 { 0 } else { (1u64 << bit) - 1 };
        rank += (self.bitmap[half] & mask).count_ones();
        self.words[rank as usize]
    }

    /// Lossless conversion FROM dense Container.
    pub fn from_dense(container: &Container) -> Self {
        let mut bitmap = [0u64; 2];
        let mut words = Vec::new();

        for i in 0..CONTAINER_WORDS {
            if container.words[i] != 0 {
                let half = i / 64;
                let bit = i % 64;
                bitmap[half] |= 1u64 << bit;
                words.push(container.words[i]);
            }
        }
        Self { bitmap, words }
    }

    /// Lossless conversion TO dense Container.
    pub fn to_dense(&self) -> Container {
        let mut container = Container::zero();
        let mut word_idx = 0;

        for i in 0..CONTAINER_WORDS {
            let half = i / 64;
            let bit = i % 64;
            if self.bitmap[half] & (1u64 << bit) != 0 {
                container.words[i] = self.words[word_idx];
                word_idx += 1;
            }
        }
        container
    }

    /// Hamming distance between two sparse containers WITHOUT densification.
    ///
    /// For words present in both: XOR and popcount.
    /// For words present in only one: full popcount of that word.
    /// For words present in neither: 0 contribution.
    pub fn hamming_sparse(a: &SparseContainer, b: &SparseContainer) -> u32 {
        let mut dist = 0u32;

        // Words in both
        let both_0 = a.bitmap[0] & b.bitmap[0];
        let both_1 = a.bitmap[1] & b.bitmap[1];

        // Words in only A
        let only_a_0 = a.bitmap[0] & !b.bitmap[0];
        let only_a_1 = a.bitmap[1] & !b.bitmap[1];

        // Words in only B
        let only_b_0 = b.bitmap[0] & !a.bitmap[0];
        let only_b_1 = b.bitmap[1] & !a.bitmap[1];

        // Process shared words: XOR and count
        for i in 0..128 {
            let half = i / 64;
            let bit = i % 64;
            let in_both = if half == 0 { both_0 } else { both_1 };
            if in_both & (1u64 << bit) != 0 {
                let wa = a.get_word(i);
                let wb = b.get_word(i);
                dist += (wa ^ wb).count_ones();
            }

            // Words only in A: all bits differ from B's zero
            let only_a = if half == 0 { only_a_0 } else { only_a_1 };
            if only_a & (1u64 << bit) != 0 {
                dist += a.get_word(i).count_ones();
            }

            // Words only in B
            let only_b = if half == 0 { only_b_0 } else { only_b_1 };
            if only_b & (1u64 << bit) != 0 {
                dist += b.get_word(i).count_ones();
            }
        }

        dist
    }

    /// XOR bind in sparse domain.
    pub fn bind_sparse(a: &SparseContainer, b: &SparseContainer) -> SparseContainer {
        // Union of bitmaps
        let bitmap = [a.bitmap[0] | b.bitmap[0], a.bitmap[1] | b.bitmap[1]];
        let mut words = Vec::new();

        for i in 0..128 {
            let half = i / 64;
            let bit = i % 64;
            if bitmap[half] & (1u64 << bit) != 0 {
                let wa = if a.has_word(i) { a.get_word(i) } else { 0 };
                let wb = if b.has_word(i) { b.get_word(i) } else { 0 };
                let xor = wa ^ wb;
                if xor != 0 {
                    words.push(xor);
                } else {
                    // XOR cancelled out — clear bitmap bit
                    // (handled below by rebuilding)
                }
            }
        }

        // Rebuild bitmap to reflect actual non-zero words after XOR
        SparseContainer::from_dense(&{
            let mut c = Container::zero();
            let da = a.to_dense();
            let db = b.to_dense();
            for i in 0..CONTAINER_WORDS {
                c.words[i] = da.words[i] ^ db.words[i];
            }
            c
        })
    }
}

impl Default for SparseContainer {
    fn default() -> Self {
        Self::zero()
    }
}

impl PartialEq for SparseContainer {
    fn eq(&self, other: &Self) -> bool {
        self.bitmap == other.bitmap && self.words == other.words
    }
}

impl Eq for SparseContainer {}

// ============================================================================
// PACKED AXES: Three sparse containers in one content Container
// ============================================================================

/// Maximum content words available for sparse axis data.
/// 128 total words - 6 bitmap words (2 per axis) = 122.
pub const MAX_AXIS_CONTENT_WORDS: usize = 122;

/// Pack three sparse axes into one Container.
///
/// Layout: [X_bmp(2)] [X_words(Nx)] [Y_bmp(2)] [Y_words(Ny)] [Z_bmp(2)] [Z_words(Nz)] [pad]
pub fn pack_axes(
    x: &SparseContainer,
    y: &SparseContainer,
    z: &SparseContainer,
) -> Result<(Container, AxisDescriptors), SpoError> {
    let total = x.word_count() + y.word_count() + z.word_count() + 6;
    if total > CONTAINER_WORDS {
        return Err(SpoError::AxesOverflow {
            total,
            max: CONTAINER_WORDS,
        });
    }

    let mut container = Container::zero();
    let mut offset = 0usize;

    // X axis
    let x_offset = offset;
    container.words[offset] = x.bitmap[0];
    container.words[offset + 1] = x.bitmap[1];
    offset += 2;
    for &w in &x.words {
        container.words[offset] = w;
        offset += 1;
    }

    // Y axis
    let y_offset = offset;
    container.words[offset] = y.bitmap[0];
    container.words[offset + 1] = y.bitmap[1];
    offset += 2;
    for &w in &y.words {
        container.words[offset] = w;
        offset += 1;
    }

    // Z axis
    let z_offset = offset;
    container.words[offset] = z.bitmap[0];
    container.words[offset + 1] = z.bitmap[1];
    offset += 2;
    for &w in &z.words {
        container.words[offset] = w;
        offset += 1;
    }

    let desc = AxisDescriptors {
        x_offset: x_offset as u16,
        x_count: x.word_count() as u16,
        y_offset: y_offset as u16,
        y_count: y.word_count() as u16,
        z_offset: z_offset as u16,
        z_count: z.word_count() as u16,
        total_words: offset as u16,
        flags: 0,
    };

    Ok((container, desc))
}

/// Unpack content Container into three sparse axes using descriptors.
pub fn unpack_axes(
    content: &Container,
    desc: &AxisDescriptors,
) -> Result<(SparseContainer, SparseContainer, SparseContainer), SpoError> {
    let x = unpack_one_axis(content, desc.x_offset as usize, desc.x_count as usize)?;
    let y = unpack_one_axis(content, desc.y_offset as usize, desc.y_count as usize)?;
    let z = unpack_one_axis(content, desc.z_offset as usize, desc.z_count as usize)?;
    Ok((x, y, z))
}

fn unpack_one_axis(
    content: &Container,
    offset: usize,
    count: usize,
) -> Result<SparseContainer, SpoError> {
    let bitmap = [content.words[offset], content.words[offset + 1]];
    let words: Vec<u64> = content.words[offset + 2..offset + 2 + count].to_vec();
    SparseContainer::new(bitmap, words)
}

/// Axis layout descriptors (stored in meta W34-W35).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AxisDescriptors {
    pub x_offset: u16,
    pub x_count: u16,
    pub y_offset: u16,
    pub y_count: u16,
    pub z_offset: u16,
    pub z_count: u16,
    pub total_words: u16,
    pub flags: u16,
}

impl AxisDescriptors {
    /// Pack into two u64 words (meta W34, W35).
    pub fn to_words(&self) -> [u64; 2] {
        let w34 = (self.x_offset as u64)
            | ((self.x_count as u64) << 16)
            | ((self.y_offset as u64) << 32)
            | ((self.y_count as u64) << 48);
        let w35 = (self.z_offset as u64)
            | ((self.z_count as u64) << 16)
            | ((self.total_words as u64) << 32)
            | ((self.flags as u64) << 48);
        [w34, w35]
    }

    /// Unpack from two u64 words.
    pub fn from_words(words: &[u64; 2]) -> Self {
        Self {
            x_offset: (words[0] & 0xFFFF) as u16,
            x_count: ((words[0] >> 16) & 0xFFFF) as u16,
            y_offset: ((words[0] >> 32) & 0xFFFF) as u16,
            y_count: ((words[0] >> 48) & 0xFFFF) as u16,
            z_offset: (words[1] & 0xFFFF) as u16,
            z_count: ((words[1] >> 16) & 0xFFFF) as u16,
            total_words: ((words[1] >> 32) & 0xFFFF) as u16,
            flags: ((words[1] >> 48) & 0xFFFF) as u16,
        }
    }

    /// Is overflow flag set? (axes stored in linked records instead)
    pub fn is_overflow(&self) -> bool {
        self.flags & 1 != 0
    }

    /// Is this a meta-awareness record?
    pub fn is_meta_awareness(&self) -> bool {
        self.flags & 2 != 0
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_roundtrip_lossless() {
        let dense = Container::random(42);
        let sparse = SparseContainer::from_dense(&dense);
        assert_eq!(sparse.to_dense(), dense);
    }

    #[test]
    fn test_sparse_zero_is_empty() {
        let sparse = SparseContainer::from_dense(&Container::zero());
        assert_eq!(sparse.word_count(), 0);
        assert_eq!(sparse.bitmap, [0, 0]);
    }

    #[test]
    fn test_sparse_bitmap_consistency() {
        let sparse = SparseContainer::from_dense(&Container::random(99));
        let ones = sparse.bitmap[0].count_ones() + sparse.bitmap[1].count_ones();
        assert_eq!(ones as usize, sparse.words.len());
    }

    #[test]
    fn test_sparse_hamming_equivalence() {
        let a = Container::random(1);
        let b = Container::random(2);
        let sa = SparseContainer::from_dense(&a);
        let sb = SparseContainer::from_dense(&b);
        assert_eq!(
            SparseContainer::hamming_sparse(&sa, &sb),
            a.hamming(&b)
        );
    }

    #[test]
    fn test_sparse_hamming_zero() {
        let a = Container::random(42);
        let sa = SparseContainer::from_dense(&a);
        assert_eq!(SparseContainer::hamming_sparse(&sa, &sa), 0);
    }

    #[test]
    fn test_sparse_density() {
        // Random container should have ~100% density (all words non-zero)
        let sparse = SparseContainer::from_dense(&Container::random(42));
        assert!(sparse.density() > 0.9);

        // Zero container should have 0% density
        let zero = SparseContainer::from_dense(&Container::zero());
        assert_eq!(zero.density(), 0.0);
    }

    #[test]
    fn test_pack_unpack_roundtrip() {
        let x = SparseContainer::from_dense(&Container::random(1));
        let y = SparseContainer::from_dense(&Container::random(2));
        let z = SparseContainer::from_dense(&Container::random(3));

        // Full random containers are too dense to pack (128 words each)
        // Use sparse ones instead
        let mut sx = SparseContainer::zero();
        let mut sy = SparseContainer::zero();
        let mut sz = SparseContainer::zero();

        // Create ~30% density containers
        for i in 0..38 {
            sx.bitmap[0] |= 1u64 << i;
            sx.words.push(0xDEAD_0000 + i as u64);
            sy.bitmap[0] |= 1u64 << (i + 1);
            sy.words.push(0xBEEF_0000 + i as u64);
            sz.bitmap[1] |= 1u64 << i;
            sz.words.push(0xCAFE_0000 + i as u64);
        }

        let (packed, desc) = pack_axes(&sx, &sy, &sz).unwrap();
        let (ux, uy, uz) = unpack_axes(&packed, &desc).unwrap();

        assert_eq!(sx.bitmap, ux.bitmap);
        assert_eq!(sx.words, ux.words);
        assert_eq!(sy.bitmap, uy.bitmap);
        assert_eq!(sy.words, uy.words);
        assert_eq!(sz.bitmap, uz.bitmap);
        assert_eq!(sz.words, uz.words);
    }

    #[test]
    fn test_pack_overflow_detection() {
        // Three fully-dense containers won't fit
        let full = SparseContainer::from_dense(&Container::random(1));
        let result = pack_axes(&full, &full, &full);
        assert!(result.is_err());
    }

    #[test]
    fn test_axis_descriptor_roundtrip() {
        let desc = AxisDescriptors {
            x_offset: 0,
            x_count: 38,
            y_offset: 40,
            y_count: 38,
            z_offset: 80,
            z_count: 38,
            total_words: 120,
            flags: 0b10, // is_meta_awareness
        };
        let words = desc.to_words();
        let restored = AxisDescriptors::from_words(&words);
        assert_eq!(desc, restored);
        assert!(restored.is_meta_awareness());
        assert!(!restored.is_overflow());
    }
}
