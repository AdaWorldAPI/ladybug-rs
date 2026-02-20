# SPO 3D Schema — Byte-Level Layout

---

## 1. SPO COGRECORD LAYOUT (2048 bytes)

```text
OFFSET  SIZE    FIELD
─────────────────────────────────────────────────────────
 META CONTAINER (1024 bytes = 128 × u64)
─────────────────────────────────────────────────────────
0x000   8B      W0   DN address (PackedDn u64)
0x008   8B      W1   node_kind:u8 | count:u8 | geometry:u8(=6) | flags:u8
                      | schema_version:u16 | provenance_hash:u16
0x010   8B      W2   created_ms:u32 | modified_ms:u32
0x018   8B      W3   label_hash:u32 | tree_depth:u8 | branch:u8 | reserved:u16
0x020   8B      W4   NARS frequency (f32 LE) | padding:u32
0x028   8B      W5   NARS confidence (f32 LE) | padding:u32
0x030   8B      W6   NARS positive evidence (f32 LE) | padding:u32
0x038   8B      W7   NARS negative evidence (f32 LE) | padding:u32
0x040   8B      W8   DN parent (u64)
0x048   8B      W9   DN first_child (u64)
0x050   8B      W10  DN next_sibling (u64)
0x058   8B      W11  DN prev_sibling (u64)
0x060   8B      W12  scent_x_hist[0..8]    ← nibble bins 0x0-0x7 for X axis
0x068   8B      W13  scent_x_hist[8..16]   ← nibble bins 0x8-0xF for X axis
0x070   8B      W14  scent_y_hist[0..8]    ← nibble bins 0x0-0x7 for Y axis
0x078   8B      W15  scent_y_hist[8..16]   ← nibble bins 0x8-0xF for Y axis
0x080   8B      W16  scent_z_hist[0..8]    ← nibble bins 0x0-0x7 for Z axis
0x088   8B      W17  scent_z_hist[8..16]   ← nibble bins 0x8-0xF for Z axis
0x090  128B     W18-33 Inline edge index (64 edges: verb:u8 | target_hint:u8)
0x110   8B      W34  axis_descriptors_0:
                      x_offset:u16 | x_count:u16 | y_offset:u16 | y_count:u16
0x118   8B      W35  axis_descriptors_1:
                      z_offset:u16 | z_count:u16 | total_words:u16 | flags:u16
0x120  32B      W36-39 Reserved (axis overflow / future use)
0x140  64B      W40-47 Bloom filter (512 bits)
0x180  64B      W48-55 Graph metrics (full precision f64)
0x1C0  64B      W56-63 Qualia (18 channels × f16 + 8 slots)
0x200 128B      W64-79 Rung history + collapse gate history
0x280 128B      W80-95 Representation language descriptor
0x300 128B      W96-111 DN-Sparse adjacency (compact inline CSR)
0x380 112B      W112-125 Reserved
0x3F0   8B      W126 Checksum (CRC32:u32 | parity:u32)
0x3F8   8B      W127 Schema version (u32) | geometry_version:u16 | reserved:u16

─────────────────────────────────────────────────────────
 CONTENT CONTAINER (1024 bytes = 128 × u64) — PACKED SPARSE AXES
─────────────────────────────────────────────────────────
0x400  16B      W0-1   X axis bitmap (128 bits)
0x410  Nx×8B    W2..   X axis non-zero words (Nx words)
        16B            Y axis bitmap (128 bits)
        Ny×8B          Y axis non-zero words (Ny words)
        16B            Z axis bitmap (128 bits)
        Nz×8B          Z axis non-zero words (Nz words)
        ...            Padding to 128 words

Total: 6 + Nx + Ny + Nz ≤ 128 words
       6 + Nx + Ny + Nz ≤ 122 non-zero content words
─────────────────────────────────────────────────────────
 TOTAL RECORD: 2048 bytes
─────────────────────────────────────────────────────────
```

## 2. SPARSE CONTAINER WIRE FORMAT

```text
┌────────────────┬──────────────────────────────┐
│ bitmap[0] u64  │ bits 0-63: which words exist │
│ bitmap[1] u64  │ bits 64-127: which words exist│
├────────────────┴──────────────────────────────┤
│ word[0]  u64   │ first non-zero word          │
│ word[1]  u64   │ second non-zero word         │
│ ...            │                              │
│ word[N-1] u64  │ last non-zero word           │
└────────────────────────────────────────────────┘

N = popcount(bitmap[0]) + popcount(bitmap[1])

To find the value of Container word `i`:
  if bitmap[i/64] & (1 << (i%64)) == 0:
    return 0  (word is zero)
  else:
    idx = popcount(bitmap[0..i/64]) + popcount(bitmap[i/64] & ((1 << (i%64)) - 1))
    return words[idx]
```

## 3. AXIS DESCRIPTOR FORMAT (meta W34-W35)

```text
W34 (8 bytes):
  bits  0-15:  x_offset  — byte offset of X bitmap within content container
  bits 16-31:  x_count   — number of non-zero words in X axis
  bits 32-47:  y_offset  — byte offset of Y bitmap within content container
  bits 48-63:  y_count   — number of non-zero words in Y axis

W35 (8 bytes):
  bits  0-15:  z_offset  — byte offset of Z bitmap within content container
  bits 16-31:  z_count   — number of non-zero words in Z axis
  bits 32-47:  total_words — total packed words (6 + Nx + Ny + Nz)
  bits 48-63:  flags     — bit 0: overflow (axes in linked records)
                            bit 1: has_meta_level (this is a meta-awareness record)
                            bits 2-15: reserved
```

## 4. NIBBLE SCENT FORMAT (meta W12-W17, 48 bytes)

```text
For each axis (X, Y, Z), 2 words = 16 bytes = 16 bins:

W12: X_hist[0x0] X_hist[0x1] X_hist[0x2] X_hist[0x3]
     X_hist[0x4] X_hist[0x5] X_hist[0x6] X_hist[0x7]
W13: X_hist[0x8] X_hist[0x9] X_hist[0xA] X_hist[0xB]
     X_hist[0xC] X_hist[0xD] X_hist[0xE] X_hist[0xF]

W14-W15: Y axis histogram (same layout)
W16-W17: Z axis histogram (same layout)

Each bin is u8: saturating count of nibble occurrences.
For a 38-word axis (304 bytes = 608 nibbles), max bin ≈ 38.
u8 range (0-255) is sufficient.

Computation:
  for each word in sparse_axis.words:
    for each nibble in word (16 nibbles per u64):
      hist[nibble] += 1  (saturating)
```

## 5. INLINE EDGE INDEX FORMAT (meta W18-W33, 128 bytes)

```text
64 edge slots, 2 bytes each = 128 bytes.
4 edges per word (4 × 16 bits = 64 bits per word).

Each edge slot (16 bits):
  bits 0-7:   verb_codebook_id (u8) — codebook entry 0-255
  bits 8-15:  target_dn_hint (u8)  — low 8 bits of target DN

Layout per word:
  W18: edge[0]:u16 | edge[1]:u16 | edge[2]:u16 | edge[3]:u16
  W19: edge[4]:u16 | edge[5]:u16 | edge[6]:u16 | edge[7]:u16
  ...
  W33: edge[60]:u16 | edge[61]:u16 | edge[62]:u16 | edge[63]:u16
```

## 6. LANCE COLUMNAR MAPPING

```text
CogRecord field         → Lance column           → Lance type
─────────────────────────────────────────────────────────────
meta.W0 (DN)           → dn                     → UInt64
meta (full)            → meta                   → FixedSizeBinary(1024)
content W0-1 (X bmp)   → x_bitmap               → FixedSizeBinary(16)
content W2..2+Nx       → x_words                → Binary (variable)
content (Y bmp)        → y_bitmap               → FixedSizeBinary(16)
content (Y words)      → y_words                → Binary
content (Z bmp)        → z_bitmap               → FixedSizeBinary(16)
content (Z words)      → z_words                → Binary
meta W12-17            → scent                  → FixedSizeBinary(48)
meta W2 low 32         → created                → Int64
meta W4 low 32         → nars_freq              → Float32
meta W5 low 32         → nars_conf              → Float32
meta W35 bit 49        → is_meta_awareness      → Boolean

Primary sort:    dn >> 48 (DN tier prefix)
Secondary sort:  scent[0..4] (first 4 bytes of X histogram)
Tertiary sort:   dn (exact)

Partition:       dn >> 56 (top byte = ~256 partitions)
```

## 7. REDIS WIRE FORMAT

```text
Key:    dn:<hex_dn>
Value:  2048 bytes (raw CogRecord: meta || content)
TTL:    none (persistent)

Scent index key:  scent:<hex_x_hist_0_3>
Value:  sorted set of DNs with scent distance as score
```

## 8. INVARIANT CHECKSUMS (meta W126-W127)

```text
W126:
  bits 0-31:   CRC32 of content container (1024 bytes)
  bits 32-63:  XOR parity of all meta words W0-W125

W127:
  bits 0-31:   schema_version (currently 1)
  bits 32-47:  geometry_version (0 = initial Spo)
  bits 48-63:  reserved (0)

Verification on read:
  1. Check W127 schema_version matches expected
  2. Compute CRC32 of content, compare to W126[0..31]
  3. XOR-fold W0-W125, compare to W126[32..63]
  4. If mismatch: record is corrupt, do not use
```
