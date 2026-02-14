//! DN-keyed Redis adapter — key mapping and pipeline patterns.
//!
//! Maps PackedDn addresses to Redis key patterns for CogRecord storage.
//! No actual Redis dependency — this defines the protocol and key layout.
//!
//! ```text
//! DnNodeStore    → KEYS ada:dn:*
//!   add_node     → SET ada:dn:{hex} {container_bytes}
//!   get_node     → GET ada:dn:{hex}
//!   children     → KEYS ada:dn:{prefix}??     (1 level deeper)
//!   subtree      → KEYS ada:dn:{prefix}*      (prefix scan)
//!   walk_to_root → MGET ada:dn:{ancestor1} ada:dn:{ancestor2} ...
//! ```

use super::Container;
use super::adjacency::PackedDn;
use super::geometry::ContainerGeometry;
use super::meta::MetaViewMut;
use super::record::CogRecord;

// ============================================================================
// KEY NAMESPACE
// ============================================================================

/// Default namespace prefix for DN-keyed records.
pub const DN_PREFIX: &str = "ada:dn:";

/// Default namespace prefix for spine records.
pub const SPINE_PREFIX: &str = "ada:spine:";

// ============================================================================
// KEY GENERATION
// ============================================================================

/// Generate a Redis key for a DN address.
#[inline]
pub fn dn_key(dn: PackedDn) -> String {
    format!("{}{}", DN_PREFIX, dn.hex())
}

/// Generate a Redis key for a spine address.
#[inline]
pub fn spine_key(dn: PackedDn) -> String {
    format!("{}{}", SPINE_PREFIX, dn.hex())
}

/// Generate keys for walk-to-root (pipelined MGET).
pub fn walk_to_root_keys(dn: PackedDn) -> Vec<String> {
    dn.ancestors().into_iter().map(dn_key).collect()
}

/// Generate a KEYS pattern for direct children of a DN.
/// Returns a glob pattern like "ada:dn:0102??????000000".
pub fn children_pattern(dn: PackedDn) -> String {
    let d = dn.depth() as usize;
    let hex = dn.hex();
    // Each level = 2 hex chars. Children have d+1 levels.
    // Known prefix = d*2 chars, next 2 chars = wildcard, rest = zeros
    let prefix = &hex[..d * 2];
    let suffix_len = 16 - (d + 1) * 2;
    format!("{}{}??{}", DN_PREFIX, prefix, "0".repeat(suffix_len))
}

/// Generate a KEYS pattern for all descendants (subtree scan).
pub fn subtree_pattern(dn: PackedDn) -> String {
    let d = dn.depth() as usize;
    let hex = dn.hex();
    let prefix = &hex[..d * 2];
    format!("{}{}*", DN_PREFIX, prefix)
}

// ============================================================================
// PIPELINE BUILDER
// ============================================================================

/// A batch of Redis operations to execute as a pipeline.
/// No actual Redis I/O — just builds the command list.
#[derive(Default)]
pub struct RedisPipeline {
    /// Sequence of commands to pipeline.
    pub commands: Vec<RedisCommand>,
}

/// A single Redis command in a pipeline.
#[derive(Debug, Clone)]
pub enum RedisCommand {
    /// GET key
    Get(String),
    /// SET key value_bytes
    Set(String, Vec<u8>),
    /// MGET key1 key2 ...
    MGet(Vec<String>),
    /// DEL key
    Del(String),
    /// KEYS pattern
    Keys(String),
}

impl RedisPipeline {
    pub fn new() -> Self {
        Self {
            commands: Vec::new(),
        }
    }

    /// Add a GET for a DN.
    pub fn get_dn(&mut self, dn: PackedDn) -> &mut Self {
        self.commands.push(RedisCommand::Get(dn_key(dn)));
        self
    }

    /// Add a SET for a DN with its CogRecord serialized as bytes.
    pub fn set_dn(&mut self, dn: PackedDn, record: &CogRecord) -> &mut Self {
        let bytes = cog_record_to_bytes(record);
        self.commands.push(RedisCommand::Set(dn_key(dn), bytes));
        self
    }

    /// Add an MGET for multiple DNs (frontier expansion).
    pub fn mget_dns(&mut self, dns: &[PackedDn]) -> &mut Self {
        let keys: Vec<String> = dns.iter().map(|dn| dn_key(*dn)).collect();
        self.commands.push(RedisCommand::MGet(keys));
        self
    }

    /// Add walk-to-root MGET (all ancestors of a DN in one round-trip).
    pub fn walk_to_root(&mut self, dn: PackedDn) -> &mut Self {
        let keys = walk_to_root_keys(dn);
        if !keys.is_empty() {
            self.commands.push(RedisCommand::MGet(keys));
        }
        self
    }

    /// Add a spine propagation: marks all spines along the DN path as dirty.
    /// In practice this would be RPUSH to a dirty queue.
    pub fn propagate_dirty(&mut self, dn: PackedDn) -> &mut Self {
        let mut current = dn;
        while let Some(parent) = current.parent() {
            self.commands.push(RedisCommand::Set(
                spine_key(parent),
                vec![], // Empty = "dirty, recompute on next read"
            ));
            current = parent;
        }
        self
    }

    /// Number of commands in the pipeline.
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// Check if pipeline is empty.
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }
}

// ============================================================================
// SERIALIZATION: CogRecord <-> bytes for Redis storage
// ============================================================================

/// Serialize a CogRecord to bytes: meta (1024 bytes) + content (1024 bytes).
pub fn cog_record_to_bytes(record: &CogRecord) -> Vec<u8> {
    let total = 2 * super::CONTAINER_BYTES;
    let mut buf = Vec::with_capacity(total);

    // Meta container first
    for &word in &record.meta.words {
        buf.extend_from_slice(&word.to_le_bytes());
    }

    // Content container
    for &word in &record.content.words {
        buf.extend_from_slice(&word.to_le_bytes());
    }

    buf
}

/// Deserialize a CogRecord from bytes.
pub fn cog_record_from_bytes(data: &[u8]) -> Option<CogRecord> {
    if data.len() < 2 * super::CONTAINER_BYTES {
        return None;
    }

    // Parse meta
    let meta = parse_container(&data[..super::CONTAINER_BYTES])?;

    // Parse content
    let content = parse_container(
        &data[super::CONTAINER_BYTES..2 * super::CONTAINER_BYTES],
    )?;

    Some(CogRecord { meta, content })
}

/// Extension trait to add serialization methods to CogRecord.
pub trait CogRecordSerde {
    /// Serialize to bytes: meta (1024 bytes) + content (1024 bytes).
    fn to_bytes(&self) -> Vec<u8>;

    /// Deserialize from a byte slice (variable-length, for Redis storage).
    fn from_bytes_slice(data: &[u8]) -> Option<CogRecord>;
}

impl CogRecordSerde for CogRecord {
    fn to_bytes(&self) -> Vec<u8> {
        cog_record_to_bytes(self)
    }

    fn from_bytes_slice(data: &[u8]) -> Option<CogRecord> {
        cog_record_from_bytes(data)
    }
}

fn parse_container(data: &[u8]) -> Option<Container> {
    if data.len() < super::CONTAINER_BYTES {
        return None;
    }
    let mut words = [0u64; super::CONTAINER_WORDS];
    for (i, chunk) in data
        .chunks_exact(8)
        .enumerate()
        .take(super::CONTAINER_WORDS)
    {
        words[i] = u64::from_le_bytes(chunk.try_into().ok()?);
    }
    Some(Container { words })
}

// ============================================================================
// RECORD BUILDER: Convenience for creating DN-stamped records
// ============================================================================

/// Build a CogRecord with a DN address and fingerprint content.
pub fn build_record(
    dn: PackedDn,
    fingerprint: &Container,
    geometry: ContainerGeometry,
) -> CogRecord {
    let mut record = CogRecord::new(geometry);
    record.content = fingerprint.clone();

    {
        let mut meta = MetaViewMut::new(&mut record.meta.words);
        meta.set_dn_addr(dn.raw());
        meta.update_checksum();
    }

    record
}
