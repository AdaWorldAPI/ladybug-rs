# Delta Encoding Formats

> **Document Version**: 1.0.0
> **Last Updated**: 2026-02-04
> **Applies To**: XOR diff versioning, Redis backup, S3 archival

---

## Prefix Envelope Architecture

The magic bytes are embedded in the **prefix envelope**, extending the 8+8 addressing model:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PREFIX ENVELOPE FORMAT                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Standard Address:   [PREFIX:8][SLOT:8]   = 16-bit (2^16 = 65,536)         │
│                       └── 0x00-0xFE valid prefixes                          │
│                                                                             │
│  Escape to Extended: [FF:FF]              = Magic byte, read more          │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [FF:FF]                    → Sparse Bitpacked follows (16-bit space)      │
│  [FF:FF][FF:FF]             → 32-bit format (Hamming/Float32)              │
│  [FF:FF][FF:FF][FF:FF]      → 48-bit format (10000D XOR compressed)        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                     PREFIX ENVELOPE IN BINARY                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Normal:       [PP][SS]                                                    │
│                 │   │                                                       │
│                 │   └── Slot (0x00-0xFF)                                   │
│                 └────── Prefix (0x00-0xFE), NOT 0xFF                       │
│                                                                             │
│  Format 1:     [FF][FF] [entry_count:16] [sparse_entries...]               │
│                ════════                                                     │
│                Envelope                                                     │
│                                                                             │
│  Format 2:     [FF][FF] [FF][FF] [variant:8] [payload...]                  │
│                ════════════════                                             │
│                Extended Envelope                                            │
│                                                                             │
│  Format 3:     [FF][FF] [FF][FF] [FF][FF] [variant:8] [payload...]         │
│                ══════════════════════════                                   │
│                Full Extended Envelope (48-bit space)                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why FF:FF as Escape?

- `0xFF` prefix is **reserved** in the 8+8 model (not a valid tier)
- `0xFF:0xFF` = 65535 = highest possible 16-bit value
- Self-synchronizing: scanner can detect format by checking first 2 bytes
- Backward compatible: old readers see invalid address, skip

```rust
/// Check if data starts with prefix envelope escape
pub fn detect_format(data: &[u8]) -> DeltaFormat {
    if data.len() < 2 {
        return DeltaFormat::Invalid;
    }

    // Check first envelope
    if data[0] != 0xFF || data[1] != 0xFF {
        // Normal 16-bit address, raw data follows
        return DeltaFormat::Raw16Bit;
    }

    // FF:FF detected, check for extended envelope
    if data.len() < 4 {
        return DeltaFormat::SparseBitpacked;  // FF:FF only
    }

    if data[2] != 0xFF || data[3] != 0xFF {
        return DeltaFormat::SparseBitpacked;  // FF:FF, then data
    }

    // FF:FF:FF:FF detected, check for full extended
    if data.len() < 6 {
        return DeltaFormat::HammingFloat32;  // FF:FF:FF:FF only
    }

    if data[4] != 0xFF || data[5] != 0xFF {
        return DeltaFormat::HammingFloat32;  // FF:FF:FF:FF, then data
    }

    // FF:FF:FF:FF:FF:FF = full 48-bit / 10000D
    DeltaFormat::Xor10000D
}
```

---

## Format 1: Sparse Bitpacked (`FFFF`)

**Magic**: `0xFFFF` (2 bytes)
**Address Space**: 2^16 = 65,536 addresses
**Use Case**: Most delta backups (typically <5% addresses change)

### Structure

```
┌──────────┬───────────────┬────────────────────────────────────────────────┐
│  FFFF    │  Entry Count  │  Sparse Entries...                             │
│  2 bytes │  2 bytes      │  Variable                                      │
└──────────┴───────────────┴────────────────────────────────────────────────┘

Each Sparse Entry:
┌──────────┬──────────┬────────────────────────────────────────────────────┐
│  Addr    │  Bitmap  │  Non-Zero Words (only where bitmap bit = 1)        │
│  2 bytes │  20 bytes│  Variable (0-156 × 8 bytes)                        │
└──────────┴──────────┴────────────────────────────────────────────────────┘
```

### Bitmap Encoding

The 20-byte bitmap indicates which of the 156 fingerprint words are non-zero:
- Bit N set → word N is stored
- Bit N clear → word N is zero (not stored)

```rust
/// Format 1: Sparse Bitpacked (FFFF header)
pub struct SparseBitpacked {
    pub magic: u16,           // Always 0xFFFF
    pub entry_count: u16,     // Number of changed addresses
    pub entries: Vec<SparseEntry>,
}

pub struct SparseEntry {
    pub addr: u16,                      // 16-bit address
    pub bitmap: [u8; 20],               // 156 bits = which words present
    pub non_zero_words: Vec<u64>,       // Only non-zero words
}

impl SparseEntry {
    /// Decode to full fingerprint
    pub fn to_fingerprint(&self) -> [u64; 156] {
        let mut fp = [0u64; 156];
        let mut word_idx = 0;

        for i in 0..156 {
            if self.bitmap[i / 8] & (1 << (i % 8)) != 0 {
                fp[i] = self.non_zero_words[word_idx];
                word_idx += 1;
            }
        }

        fp
    }

    /// Encode from XOR delta (only store non-zero)
    pub fn from_delta(addr: u16, delta: &[u64; 156]) -> Self {
        let mut bitmap = [0u8; 20];
        let mut non_zero_words = Vec::new();

        for (i, &word) in delta.iter().enumerate() {
            if word != 0 {
                bitmap[i / 8] |= 1 << (i % 8);
                non_zero_words.push(word);
            }
        }

        Self { addr, bitmap, non_zero_words }
    }
}
```

### Compression Ratio

For typical cognitive data (sparse XOR deltas):
- Average non-zero words per delta: ~15-20 (of 156)
- Compression: **80-90%** vs full fingerprint

---

## Format 2: Float32 / 32-bit Hamming Delta (`FFFF FFFF`)

**Magic**: `0xFFFF_FFFF` (4 bytes)
**Address Space**: Extended or compressed representation
**Use Case**: Hamming distance deltas, float32 quantized vectors

### Variant A: 32-bit Hamming Detail Delta

```
┌──────────────┬───────────┬────────────────────────────────────────────────┐
│  FFFF FFFF   │  Variant  │  Payload                                       │
│  4 bytes     │  1 byte   │  Variable                                      │
└──────────────┴───────────┴────────────────────────────────────────────────┘

Variant 0x01: Hamming Delta
┌──────────┬──────────┬──────────┬────────────────────────────────────────┐
│  Addr    │  Hamming │  Bit Pos │  (only changed bit positions stored)   │
│  2 bytes │  2 bytes │  Variable│                                        │
└──────────┴──────────┴──────────┴────────────────────────────────────────┘
```

For fingerprints where only a few bits changed:
- Store Hamming distance (number of changed bits)
- Store list of bit positions that flipped

```rust
/// Format 2A: Hamming Detail Delta
pub struct HammingDelta {
    pub magic: u32,             // Always 0xFFFF_FFFF
    pub variant: u8,            // 0x01 for Hamming
    pub addr: u16,
    pub hamming_distance: u16,  // Number of bits changed
    pub bit_positions: Vec<u16>, // Which bits flipped (0..10000)
}

impl HammingDelta {
    /// Apply to previous fingerprint
    pub fn apply(&self, prev: &mut [u64; 156]) {
        for &bit_pos in &self.bit_positions {
            let word_idx = (bit_pos / 64) as usize;
            let bit_idx = bit_pos % 64;
            prev[word_idx] ^= 1u64 << bit_idx;
        }
    }

    /// Create from two fingerprints
    pub fn compute(addr: u16, old: &[u64; 156], new: &[u64; 156]) -> Self {
        let mut bit_positions = Vec::new();

        for (i, (&o, &n)) in old.iter().zip(new.iter()).enumerate() {
            let diff = o ^ n;
            for bit in 0..64 {
                if diff & (1u64 << bit) != 0 {
                    bit_positions.push((i * 64 + bit) as u16);
                }
            }
        }

        Self {
            magic: 0xFFFF_FFFF,
            variant: 0x01,
            addr,
            hamming_distance: bit_positions.len() as u16,
            bit_positions,
        }
    }
}
```

### Variant B: Float32 Compressed

For vector embeddings that need floating-point precision:

```
Variant 0x02: Float32 Quantized
┌──────────┬──────────┬──────────┬────────────────────────────────────────┐
│  Addr    │  Scale   │  Offset  │  Quantized Values (f32 × dims)         │
│  2 bytes │  4 bytes │  4 bytes │  4 × D bytes                           │
└──────────┴──────────┴──────────┴────────────────────────────────────────┘
```

---

## Format 3: Non-Sparse Full (`FFFF FFFF FFFF`)

**Magic**: `0xFFFF_FFFF_FFFF` (6 bytes)
**Address Space**: 2^48 or full 10000D vectors
**Use Case**: Full snapshots, non-sparse data, high-dimensional vectors

### Variant A: 48-bit Extended Address Space

For future expansion beyond 65,536 addresses:

```
┌────────────────────┬───────────┬─────────────────────────────────────────┐
│  FFFF FFFF FFFF    │  Variant  │  Payload                                │
│  6 bytes           │  1 byte   │  Variable                               │
└────────────────────┴───────────┴─────────────────────────────────────────┘

Variant 0x01: Extended Address Space
┌──────────┬──────────────────────────────────────────────────────────────┐
│  Addr    │  Full Fingerprint (no compression)                           │
│  6 bytes │  156 × 8 = 1248 bytes                                        │
└──────────┴──────────────────────────────────────────────────────────────┘
```

### Variant B: 10000D XOR Compressed

For full 10,000-bit fingerprints with XOR delta compression:

```
Variant 0x02: 10000D XOR Block
┌──────────┬──────────┬────────────────────────────────────────────────────┐
│  Version │  Count   │  XOR Blocks...                                     │
│  8 bytes │  4 bytes │  Variable                                          │
└──────────┴──────────┴────────────────────────────────────────────────────┘

Each XOR Block:
┌──────────┬──────────┬────────────────────────────────────────────────────┐
│  Addr    │  Length  │  XOR Mask (variable, run-length encoded)           │
│  6 bytes │  2 bytes │  Variable                                          │
└──────────┴──────────┴────────────────────────────────────────────────────┘
```

```rust
/// Format 3B: 10000D XOR Compressed
pub struct XorCompressed10000D {
    pub magic: [u8; 6],         // Always 0xFF × 6
    pub variant: u8,            // 0x02 for 10000D XOR
    pub version: u64,           // Target version
    pub block_count: u32,
    pub blocks: Vec<XorBlock10000D>,
}

pub struct XorBlock10000D {
    pub addr: u64,              // 48-bit address (padded to 64)
    pub rle_xor: Vec<u8>,       // Run-length encoded XOR mask
}

impl XorBlock10000D {
    /// RLE encode XOR mask (good for sparse changes)
    pub fn rle_encode(xor: &[u64; 156]) -> Vec<u8> {
        let mut result = Vec::new();
        let bytes: &[u8] = bytemuck::cast_slice(xor);

        let mut i = 0;
        while i < bytes.len() {
            let byte = bytes[i];
            let mut count = 1u8;

            while i + count as usize < bytes.len()
                && bytes[i + count as usize] == byte
                && count < 255
            {
                count += 1;
            }

            if count >= 3 || byte == 0xFF {
                // RLE: marker + count + byte
                result.push(0xFF);
                result.push(count);
                result.push(byte);
            } else {
                // Literal
                for _ in 0..count {
                    result.push(byte);
                }
            }

            i += count as usize;
        }

        result
    }
}
```

---

## Format Selection Logic

```rust
/// Auto-select best format for delta
pub fn select_format(
    old: &BindSpace,
    new: &BindSpace,
) -> DeltaFormat {
    let mut changed_addrs = 0;
    let mut total_hamming = 0u64;
    let mut max_hamming = 0u32;

    // Analyze changes
    for addr in 0..65536u16 {
        let old_fp = old.read(Addr(addr));
        let new_fp = new.read(Addr(addr));

        match (old_fp, new_fp) {
            (Some(o), Some(n)) if o.fingerprint != n.fingerprint => {
                changed_addrs += 1;
                let h = hamming_distance(&o.fingerprint, &n.fingerprint);
                total_hamming += h as u64;
                max_hamming = max_hamming.max(h);
            }
            (None, Some(_)) | (Some(_), None) => {
                changed_addrs += 1;
                max_hamming = 10000; // Full fingerprint change
            }
            _ => {}
        }
    }

    // Decision tree
    if changed_addrs == 0 {
        DeltaFormat::None // No changes
    } else if changed_addrs < 1000 && max_hamming < 100 {
        // Few addresses with small Hamming changes
        DeltaFormat::HammingDelta  // FFFF FFFF
    } else if changed_addrs < 5000 {
        // Moderate changes, use sparse
        DeltaFormat::SparseBitpacked  // FFFF
    } else {
        // Many changes, use full XOR compressed
        DeltaFormat::Xor10000D  // FFFF FFFF FFFF
    }
}

pub enum DeltaFormat {
    None,
    SparseBitpacked,    // FFFF
    HammingDelta,       // FFFF FFFF (variant 0x01)
    Float32Quantized,   // FFFF FFFF (variant 0x02)
    ExtendedAddress,    // FFFF FFFF FFFF (variant 0x01)
    Xor10000D,          // FFFF FFFF FFFF (variant 0x02)
}
```

---

## Redis Key Schema with Format Tags

```
# Key includes format indicator
ladybug:delta:sparse:{from}:{to}      → Format 1 (FFFF)
ladybug:delta:hamming:{from}:{to}     → Format 2A (FFFF FFFF, 0x01)
ladybug:delta:float32:{from}:{to}     → Format 2B (FFFF FFFF, 0x02)
ladybug:delta:full:{from}:{to}        → Format 3 (FFFF FFFF FFFF)
ladybug:snapshot:{version}            → Full snapshot (Parquet)

# Or self-describing with magic in payload
ladybug:delta:{from}:{to}             → Magic bytes determine format
```

---

## Encoding/Decoding Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ENCODING PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  BindSpace₁ ─┬─► Diff ─► Analyze ─► Select Format ─► Encode ─► Store       │
│  BindSpace₂ ─┘                                                              │
│                                                                             │
│  Analysis:                                                                  │
│  ├─ changed_addrs < 1000 AND max_hamming < 100 → Hamming Delta (FFFF FFFF) │
│  ├─ changed_addrs < 5000 → Sparse Bitpacked (FFFF)                         │
│  └─ else → Full XOR Compressed (FFFF FFFF FFFF)                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         DECODING PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Load ─► Read Magic ─► Dispatch:                                           │
│                        ├─ FFFF → SparseBitpacked::decode()                 │
│                        ├─ FFFF FFFF → match variant { 0x01, 0x02 }         │
│                        └─ FFFF FFFF FFFF → match variant { 0x01, 0x02 }    │
│                                                                             │
│  Apply to previous snapshot → Reconstructed BindSpace                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Size Comparison

| Format | Typical Delta | Full Snapshot | Compression |
|--------|---------------|---------------|-------------|
| Sparse (FFFF) | 5-50 KB | 80 MB | 99%+ |
| Hamming (FFFF FFFF) | 1-10 KB | 80 MB | 99.9%+ |
| Full XOR (FFFF FFFF FFFF) | 20-200 KB | 80 MB | 95%+ |
| Uncompressed | N/A | 80 MB | 0% |

**Note**: 80 MB = 65,536 addresses × 1,248 bytes/fingerprint
