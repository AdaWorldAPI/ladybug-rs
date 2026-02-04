# Delta Encoding Formats

> **Document Version**: 3.0.0
> **Last Updated**: 2026-02-04
> **Applies To**: XOR diff versioning, Redis backup, S3 archival

---

## Prefix Reservation and Blocking

**Critical**: Once the `0xFF` prefix is used for escape sequences, it is **permanently blocked** for all other uses in the address space.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PREFIX BLOCKING                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  8+8 Address Model Tiers:                                                   │
│    0x00-0x0F:xx  SURFACE (16 prefixes × 256 slots = 4,096 addresses)       │
│    0x10-0x7F:xx  FLUID   (112 prefixes × 256 slots = 28,672 addresses)     │
│    0x80-0xFE:xx  NODES   (127 prefixes × 256 slots = 32,512 addresses)     │
│    0xFF:xx       BLOCKED (256 slots reserved for escape formats)           │
│                  ═══════                                                    │
│                                                                             │
│  Total usable addresses: 65,280 (not 65,536)                               │
│  Reserved for escapes:   256                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why This Works

The 8+8 model already reserves `0xFF` as an invalid tier. No valid cognitive address uses prefix `0xFF`. This is by design:

```rust
/// Validate address - 0xFF prefix is NEVER valid for data
pub fn is_valid_address(addr: u16) -> bool {
    (addr >> 8) != 0xFF
}

/// Address ranges
pub const SURFACE_START: u16 = 0x0000;
pub const SURFACE_END: u16   = 0x0FFF;  // 0x00-0x0F prefixes
pub const FLUID_START: u16   = 0x1000;
pub const FLUID_END: u16     = 0x7FFF;  // 0x10-0x7F prefixes
pub const NODES_START: u16   = 0x8000;
pub const NODES_END: u16     = 0xFEFF;  // 0x80-0xFE prefixes
pub const ESCAPE_START: u16  = 0xFF00;  // 0xFF prefix = escape
pub const ESCAPE_END: u16    = 0xFFFF;
```

### Handling Literal 0xFF in Data

If raw data (not addresses) contains byte 0xFF, it must be escaped within the payload:

```rust
/// Escape 0xFF bytes in payload data
pub fn escape_payload(data: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(data.len() * 2);
    for &byte in data {
        if byte == 0xFF {
            // FF in payload → FF:00 (escape + null = literal FF)
            result.push(0xFF);
            result.push(0x00);
        } else {
            result.push(byte);
        }
    }
    result
}

/// Unescape payload
pub fn unescape_payload(data: &[u8]) -> Vec<u8> {
    let mut result = Vec::new();
    let mut i = 0;
    while i < data.len() {
        if data[i] == 0xFF && i + 1 < data.len() && data[i + 1] == 0x00 {
            // FF:00 = literal 0xFF
            result.push(0xFF);
            i += 2;
        } else {
            result.push(data[i]);
            i += 1;
        }
    }
    result
}
```

### Alternative: Escape Space as Feature Enable Bits

Instead of using FF:xx purely for data format encoding, the escape space can function as **feature enable bits** / **mode flags**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ESCAPE AS ENABLE BITS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Each FF:xx address = enable/disable flag for feature xx                   │
│                                                                             │
│  Write to FF:00 → updates feature 0x00 state                               │
│  Write to FF:01 → updates feature 0x01 state                               │
│  ...                                                                        │
│  Write to FF:FF → updates feature 0xFF state                               │
│                                                                             │
│  Backend writes to MULTIPLE prefixes:                                       │
│  Every update is an implicit mode update to ONE feature prefix only        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Feature Prefix Mapping:
┌────────┬────────────────────────────────────────────────────────────────────┐
│  FF:0x │  Core system modes (0=disable all, 1=enable, 2=reset, etc.)       │
│  FF:1x │  Hamming mode flags (distance thresholds, search params)          │
│  FF:2x │  CAM operation modes (4096 ops control)                           │
│  FF:Bx │  Float backend modes (precision, quantization level)              │
│  FF:Dx │  Dimensional sparse modes (bitmap compression settings)           │
│  FF:Ex │  Extended address modes (48-bit expansion control)                │
│  FF:Fx │  Full 10000D modes (XOR/RLE compression flags)                    │
└────────┴────────────────────────────────────────────────────────────────────┘
```

### Multi-Backend Write Pattern

```rust
/// Backend writes to escape address to set feature mode
pub fn set_feature_mode(space: &mut BindSpace, feature: u8, mode: &[u8]) {
    let escape_addr = 0xFF00 | (feature as u16);
    space.write_raw(Addr(escape_addr), mode);
}

/// Read current mode for a feature
pub fn get_feature_mode(space: &BindSpace, feature: u8) -> Option<&[u8]> {
    let escape_addr = 0xFF00 | (feature as u16);
    space.read_raw(Addr(escape_addr))
}

/// Example: Enable dimensional sparse mode with specific settings
pub fn enable_sparse_dim(space: &mut BindSpace, max_entries: u16, bitmap_size: u8) {
    let mode = [
        0x01,                           // Enabled
        (max_entries >> 8) as u8,       // Max entries high
        max_entries as u8,              // Max entries low
        bitmap_size,                    // Bitmap size (20 default)
    ];
    set_feature_mode(space, 0xD0, &mode);  // FF:D0 = sparse dim mode
}
```

### Implicit Mode Updates

When writing data to a regular prefix (00:xx to FE:xx), the corresponding feature mode in FF:xx can be implicitly updated:

```rust
/// Write data and implicitly update feature mode
pub fn write_with_mode_update(
    space: &mut BindSpace,
    addr: Addr,
    data: &[u8],
    feature_flags: u8,
) {
    // Write main data
    space.write(addr, data);

    // Implicitly update the feature prefix's mode
    let prefix = (addr.0 >> 8) as u8;
    let mode_addr = 0xFF00 | (prefix as u16);

    // Merge with existing mode or set new
    let current = space.read_raw(Addr(mode_addr)).unwrap_or(&[0]);
    let new_mode = current[0] | feature_flags;
    space.write_raw(Addr(mode_addr), &[new_mode]);
}
```

---

## Bit-Efficient Prefix Envelope

Every bit counts. The format indicator is encoded in the **upper nibble** of the escape byte, with data beginning immediately after.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     BIT-PACKED FORMAT ENCODING                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Normal Address (16-bit):                                                   │
│  [PPPP PPPP][SSSS SSSS]    where PP ≠ 0xFF                                 │
│                                                                             │
│  Extended Format (FF escape):                                               │
│  [1111 1111][TTTT DDDD][...]                                               │
│   └─ FF ─┘  │    │                                                         │
│             │    └── First 4 data bits (not wasted)                        │
│             └─────── Type nibble (0-F format selector)                     │
│                                                                             │
│  Type nibble values:                                                        │
│     0x0_ = ESCAPE (FF:00 = literal 0xFF byte)                              │
│     0x1_ = Hamming bit positions (16-bit addr space)                       │
│     0x2_ - 0x9_ = Reserved for future                                      │
│     0xA_ = Archive reference (Parquet pointer)                             │
│     0xB_ = Binary float (f32 quantized)                                    │
│     0xC_ = Codebook (learned quantization)                                 │
│     0xD_ = Dimensional sparse (16-bit addr space)                          │
│     0xE_ = Extended (48-bit addr space)                                    │
│     0xF_ = Full (10000D XOR, RLE compressed)                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The Lower Nibble is DATA

The lower nibble of byte[1] is NOT wasted - it's the first 4 bits of the payload:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     NIBBLE PACKING                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [FF][Dn][data...]  where n = first data nibble                            │
│                                                                             │
│  Example: FF:D3 means:                                                      │
│    - Format D (Dimensional sparse)                                          │
│    - First data nibble = 0x3 (e.g., entry count high bits)                 │
│                                                                             │
│  Example: FF:FF means:                                                      │
│    - Format F (Full 10000D)                                                 │
│    - First data nibble = 0xF                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Detection Code

```rust
/// Detect format - type is in upper nibble
pub fn detect_format(data: &[u8]) -> DeltaFormat {
    if data.is_empty() {
        return DeltaFormat::Invalid;
    }

    // Not escape? Normal 16-bit data
    if data[0] != 0xFF {
        return DeltaFormat::Raw16Bit;
    }

    if data.len() < 2 {
        return DeltaFormat::Invalid;
    }

    // Upper nibble = format type
    let type_nibble = data[1] >> 4;

    match type_nibble {
        0x1 => DeltaFormat::Hamming,      // 1x = Hamming bit positions
        0xB => DeltaFormat::Float32,      // Bx = Binary float
        0xD => DeltaFormat::SparseDim,    // Dx = Dimensional sparse
        0xE => DeltaFormat::Extended48,   // Ex = Extended 48-bit
        0xF => DeltaFormat::Full10000D,   // Fx = Full 10000D XOR
        _ => DeltaFormat::Unknown,
    }
}

/// Extract first data nibble (lower nibble of byte[1])
pub fn first_data_nibble(data: &[u8]) -> u8 {
    data[1] & 0x0F
}

pub enum DeltaFormat {
    Invalid,
    Raw16Bit,       // No escape, normal address
    Hamming,        // FF:1x - bit position deltas
    Float32,        // FF:Bx - binary float
    SparseDim,      // FF:Dx - dimensional sparse
    Extended48,     // FF:Ex - extended 48-bit
    Full10000D,     // FF:Fx - full 10000D XOR
    Unknown,
}
```

---

## Format FF:Dx - Sparse Dimensional

**Escape**: `FF` + upper nibble `D`
**Lower nibble**: Entry count high 4 bits
**Total overhead**: 12 bits (1.5 bytes), not 16

### Structure

```
┌────────────┬───────────────────┬────────────────────────────────────────────┐
│  FF:Dn     │  Entry Count Low  │  Sparse Entries...                         │
│  12+4 bits │  8 bits           │  Variable                                  │
└────────────┴───────────────────┴────────────────────────────────────────────┘

Entry count = (n << 8) | low_byte = 12-bit count (0-4095 entries)

Each Sparse Entry:
┌──────────┬──────────┬────────────────────────────────────────────────────┐
│  Addr    │  Bitmap  │  Non-Zero Words (only where bitmap bit = 1)        │
│  2 bytes │  20 bytes│  Variable (0-156 × 8 bytes)                        │
└──────────┴──────────┴────────────────────────────────────────────────────┘
```

```rust
/// Format FF:Dx - Sparse Dimensional
/// Entry count encoded: upper nibble in byte[1], lower 8 bits in byte[2]
pub fn encode_sparse_dim(entries: &[SparseEntry]) -> Vec<u8> {
    let count = entries.len() as u16;
    assert!(count <= 4095, "Max 4095 entries in sparse format");

    let mut buf = Vec::new();

    // FF escape
    buf.push(0xFF);

    // Upper nibble D + high 4 bits of count
    buf.push(0xD0 | ((count >> 8) as u8 & 0x0F));

    // Low 8 bits of count
    buf.push(count as u8);

    // Entries
    for entry in entries {
        buf.extend_from_slice(&entry.addr.to_le_bytes());
        buf.extend_from_slice(&entry.bitmap);
        for word in &entry.non_zero_words {
            buf.extend_from_slice(&word.to_le_bytes());
        }
    }

    buf
}

pub fn decode_sparse_dim_header(data: &[u8]) -> u16 {
    let high = (data[1] & 0x0F) as u16;
    let low = data[2] as u16;
    (high << 8) | low
}
```

---

## Format FF:1x - Hamming Delta

**Escape**: `FF` + upper nibble `1`
**Lower nibble**: Entry count high 4 bits

### Structure

```
┌────────────┬───────────────────┬────────────────────────────────────────────┐
│  FF:1n     │  Entry Count Low  │  Hamming Entries...                        │
│  12+4 bits │  8 bits           │  Variable                                  │
└────────────┴───────────────────┴────────────────────────────────────────────┘

Each Hamming Entry:
┌──────────┬──────────┬────────────────────────────────────────────────────┐
│  Addr    │  BitCnt  │  Bit Positions (14-bit each, packed)               │
│  2 bytes │  7 bits  │  Variable                                          │
└──────────┴──────────┴────────────────────────────────────────────────────┘
```

Bit positions are 14-bit (max 9999), packed without byte boundaries.

```rust
/// Bit-pack 14-bit positions (0-9999 fits in 14 bits)
pub fn pack_bit_positions(positions: &[u16]) -> Vec<u8> {
    let mut bits = Vec::new();

    for &pos in positions {
        assert!(pos < 10000);
        // Pack 14 bits
        for i in (0..14).rev() {
            bits.push((pos >> i) & 1 == 1);
        }
    }

    // Convert bits to bytes
    let mut bytes = Vec::new();
    for chunk in bits.chunks(8) {
        let mut byte = 0u8;
        for (i, &bit) in chunk.iter().enumerate() {
            if bit {
                byte |= 1 << (7 - i);
            }
        }
        bytes.push(byte);
    }

    bytes
}
```

---

## Format FF:Bx - Float32 (Binary float)

**Escape**: `FF` + upper nibble `B`
**Lower nibble**: Dimensions high 4 bits

```
┌────────────┬───────────────────┬────────────────────────────────────────────┐
│  FF:Bn     │  Dims Low + Count │  Float Entries...                          │
│  12+4 bits │  16 bits          │  Variable                                  │
└────────────┴───────────────────┴────────────────────────────────────────────┘
```

---

## Format FF:Ex - Extended 48-bit

**Escape**: `FF` + upper nibble `E`
**Lower nibble**: Reserved (0)

48-bit addresses are atomic - 6 bytes each, no splitting.

```
┌────────────┬───────────────────┬────────────────────────────────────────────┐
│  FF:E0     │  Entry Count      │  Extended Entries...                       │
│  12+4 bits │  32 bits          │  Variable                                  │
└────────────┴───────────────────┴────────────────────────────────────────────┘

Each Extended Entry:
┌──────────────┬────────────────────────────────────────────────────────────┐
│  Addr        │  Full Fingerprint (no compression)                         │
│  48 bits     │  156 × 64 = 9984 bits                                      │
└──────────────┴────────────────────────────────────────────────────────────┘
```

The 48-bit address is stored as a contiguous 6-byte value.

---

## Format FF:Fx - Full 10000D XOR

**Escape**: `FF` + upper nibble `F`
**Lower nibble**: Compression flags

```
┌────────────┬───────────────────┬────────────────────────────────────────────┐
│  FF:Fn     │  Version + Count  │  XOR Blocks...                             │
│  12+4 bits │  96 bits          │  Variable                                  │
└────────────┴───────────────────┴────────────────────────────────────────────┘

Flags (lower nibble n):
  bit 0: RLE enabled
  bit 1: word-aligned
  bit 2: reserved
  bit 3: reserved

Each XOR Block:
┌──────────┬──────────┬────────────────────────────────────────────────────┐
│  Addr    │  Length  │  XOR Mask (optionally RLE encoded)                 │
│  16 bits │  12 bits │  Variable                                          │
└──────────┴──────────┴────────────────────────────────────────────────────┘
```

---

## Format Selection

```rust
pub fn select_format(changed: usize, max_hamming: u32) -> DeltaFormat {
    if changed == 0 {
        DeltaFormat::Raw16Bit
    } else if changed < 1000 && max_hamming < 100 {
        DeltaFormat::Hamming      // FF:1x
    } else if changed < 4096 {
        DeltaFormat::SparseDim    // FF:Dx
    } else {
        DeltaFormat::Full10000D   // FF:Fx
    }
}
```

---

## Redis Key Schema

```
# Self-describing (format in first 12 bits after FF)
ladybug:delta:{from}:{to}             → [FF][Tx][payload...]

# Snapshots (no escape prefix)
ladybug:snapshot:{version}            → Parquet blob
```

---

## Bit Efficiency Summary

| Format | Header | Type | Savings vs 16-bit escape |
|--------|--------|------|--------------------------|
| Hamming | `FF:1x` | 12 bits + 4 data | 4 bits saved |
| Float32 | `FF:Bx` | 12 bits + 4 data | 4 bits saved |
| Sparse | `FF:Dx` | 12 bits + 4 data | 4 bits saved |
| Extended | `FF:Ex` | 12 bits + 4 reserved | 4 bits saved |
| Full | `FF:Fx` | 12 bits + 4 flags | 4 bits used for flags |

The lower nibble is **never wasted** - it's either data or flags.
