//! Container geometry — how content containers are interpreted.

/// How the content containers of a [`CogRecord`] are arranged and interpreted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ContainerGeometry {
    /// 1 content container: flat 8K CAM fingerprint (default, most common).
    /// Total record: 2 × 1 KB = 2 KB.
    Cam = 0,

    /// 3 content containers: X (what) + Y (where) + Z (how), holographic.
    /// Store: trace = X ⊕ Y ⊕ Z.  Probe: given any 2, recover 3rd.
    /// Total record: 4 × 1 KB = 4 KB.
    Xyz = 1,

    /// 1 content container (CAM proxy) + pointer to external float vector.
    /// External: 1024-D / 1536-D / 4096-D f32 in Lance/Redis.
    /// Total record: 2 × 1 KB = 2 KB (+ external).
    Bridge = 2,

    /// 2 content containers: primary + secondary (stacked planes or 16K compat).
    /// Total record: 3 × 1 KB = 3 KB.
    Extended = 3,

    /// N content containers: first is summary bundle, rest are chunks.
    /// Supports multimodal: text chapters, image fingerprints, audio segments.
    /// Container index IS sequence position.
    /// Total record: (N+1) × 1 KB.
    Chunked = 4,

    /// N content containers in BFS heap layout: serialized DN subtree.
    /// Children of node i at indices k*i+1..k*(i+1).
    /// Adjacency is implicit in position. Spine = XOR of children.
    /// Total record: (N+1) × 1 KB.
    Tree = 5,
}

impl ContainerGeometry {
    /// Default content container count for this geometry.
    pub fn default_content_count(self) -> usize {
        match self {
            ContainerGeometry::Cam => 1,
            ContainerGeometry::Xyz => 3,
            ContainerGeometry::Bridge => 1,
            ContainerGeometry::Extended => 2,
            ContainerGeometry::Chunked => 1, // just summary; chunks added later
            ContainerGeometry::Tree => 1,    // root; children added later
        }
    }

    /// Decode from u8.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(ContainerGeometry::Cam),
            1 => Some(ContainerGeometry::Xyz),
            2 => Some(ContainerGeometry::Bridge),
            3 => Some(ContainerGeometry::Extended),
            4 => Some(ContainerGeometry::Chunked),
            5 => Some(ContainerGeometry::Tree),
            _ => None,
        }
    }
}

impl Default for ContainerGeometry {
    fn default() -> Self {
        ContainerGeometry::Cam
    }
}
