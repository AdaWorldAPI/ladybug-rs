//! Addr ↔ PackedDn bridge.
//!
//! `Addr` (u16, hash-based, flat) and `PackedDn` (u64, path-aware, hierarchical)
//! are fundamentally different addressing models:
//!
//! - `Addr`: O(1) array-indexed, loses hierarchy, used by BindSpace/CogRedis
//! - `PackedDn`: 7-level sortable hierarchy, used by Container/DnSpineCache
//!
//! This bridge provides bidirectional conversion with a lookup table.
//! The mapping is NOT a pure function — it requires a registry because
//! multiple PackedDns can hash to the same Addr (collision domain).

use std::collections::HashMap;

use super::adjacency::PackedDn;

/// Legacy 16-bit address (prefix:slot), matching `crate::storage::bind_space::Addr`.
/// Duplicated here to avoid circular dependency. The `From` conversions
/// in integration code bridge between this and `bind_space::Addr`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LegacyAddr(pub u16);

impl LegacyAddr {
    /// Construct from prefix:slot.
    #[inline]
    pub fn new(prefix: u8, slot: u8) -> Self {
        Self(((prefix as u16) << 8) | slot as u16)
    }

    /// Extract prefix byte.
    #[inline]
    pub fn prefix(self) -> u8 {
        (self.0 >> 8) as u8
    }

    /// Extract slot byte.
    #[inline]
    pub fn slot(self) -> u8 {
        (self.0 & 0xFF) as u8
    }

    /// Is this a node-tier address? (prefix >= 0x80)
    #[inline]
    pub fn is_node(self) -> bool {
        self.prefix() >= 0x80
    }
}

impl From<u16> for LegacyAddr {
    fn from(v: u16) -> Self { Self(v) }
}

impl From<LegacyAddr> for u16 {
    fn from(a: LegacyAddr) -> u16 { a.0 }
}

/// Bidirectional Addr ↔ PackedDn registry.
///
/// Not a pure function — maintains explicit mapping tables because:
/// 1. PackedDn → Addr is a lossy hash (multiple DNs can collide)
/// 2. Addr → PackedDn requires the reverse lookup table
pub struct AddrBridge {
    /// PackedDn → LegacyAddr mapping.
    dn_to_addr: HashMap<PackedDn, LegacyAddr>,

    /// LegacyAddr → PackedDn mapping.
    /// If multiple DNs hash to the same Addr, only the most recently
    /// registered one is stored (last-writer-wins).
    addr_to_dn: HashMap<LegacyAddr, PackedDn>,
}

impl AddrBridge {
    /// Create a new empty bridge.
    pub fn new() -> Self {
        Self {
            dn_to_addr: HashMap::new(),
            addr_to_dn: HashMap::new(),
        }
    }

    /// Create with estimated capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            dn_to_addr: HashMap::with_capacity(cap),
            addr_to_dn: HashMap::with_capacity(cap),
        }
    }

    /// Register a bidirectional mapping.
    pub fn register(&mut self, dn: PackedDn, addr: LegacyAddr) {
        self.dn_to_addr.insert(dn, addr);
        self.addr_to_dn.insert(addr, dn);
    }

    /// Resolve PackedDn → LegacyAddr.
    /// If not registered, computes a deterministic hash.
    pub fn addr_for(&self, dn: PackedDn) -> LegacyAddr {
        if let Some(&addr) = self.dn_to_addr.get(&dn) {
            return addr;
        }
        // Fallback: deterministic hash into node tier (0x80-0xFF)
        dn_to_addr_hash(dn)
    }

    /// Resolve LegacyAddr → PackedDn.
    /// Returns None if no mapping is registered.
    pub fn dn_for(&self, addr: LegacyAddr) -> Option<PackedDn> {
        self.addr_to_dn.get(&addr).copied()
    }

    /// Number of registered mappings.
    pub fn len(&self) -> usize {
        self.dn_to_addr.len()
    }

    /// Is the bridge empty?
    pub fn is_empty(&self) -> bool {
        self.dn_to_addr.is_empty()
    }

    /// Register a DN, auto-computing the Addr from hash.
    /// Returns the assigned Addr.
    pub fn register_auto(&mut self, dn: PackedDn) -> LegacyAddr {
        let addr = dn_to_addr_hash(dn);
        self.register(dn, addr);
        addr
    }

    /// All registered DNs.
    pub fn all_dns(&self) -> impl Iterator<Item = &PackedDn> {
        self.dn_to_addr.keys()
    }
}

impl Default for AddrBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Deterministic hash of a PackedDn to a LegacyAddr in the node tier.
///
/// Uses the same SplitMix64 hash that bind_space::dn_path_to_addr uses,
/// but operates on the packed u64 directly instead of a string path.
pub fn dn_to_addr_hash(dn: PackedDn) -> LegacyAddr {
    let mut z = dn.raw().wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z = z ^ (z >> 31);

    // Map into node tier: prefix 0x80-0xFF, slot 0x00-0xFF
    let prefix = 0x80 | ((z >> 8) as u8 & 0x7F);
    let slot = (z & 0xFF) as u8;
    LegacyAddr::new(prefix, slot)
}

/// Compute a "leaf hint" byte for a PackedDn.
/// Used for InlineEdge target_hint field.
/// This is the low byte of the DN hash, giving 256-way discrimination.
pub fn dn_leaf_hint(dn: PackedDn) -> u8 {
    let mut z = dn.raw().wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    (z & 0xFF) as u8
}
