//! BitpackedCsr ↔ DnSpineCache bridge.
//!
//! Provides methods for BitpackedCsr to read adjacency data from
//! Container 0 inline edges in the shared DnSpineCache.
//!
//! # HARD CONTRACT: BitpackedCsr API is UNCHANGED
//!
//! BitpackedCsr keeps its existing `children()`, `children_via()`, `out_degree()`
//! methods. This bridge provides *additional* functions that can read from
//! DnSpineCache, giving BindSpace/UnifiedEngine a secondary path.

use super::adjacency::PackedDn;
use super::dn_spine_cache::DnSpineCache;
use super::addr_bridge::{AddrBridge, LegacyAddr};

/// Read children of a DN from Container 0 inline edges.
/// Returns (verb, target_hint) pairs matching the given verb filter.
pub fn children_via_container(
    dn_cache: &DnSpineCache,
    dn: PackedDn,
    verb_filter: Option<u8>,
) -> Vec<(u8, u8)> {
    let view = match dn_cache.inline_edges(dn) {
        Some(v) => v,
        None => return Vec::new(),
    };

    view.iter()
        .filter(|(_, e)| match verb_filter {
            Some(v) => e.verb == v,
            None => true,
        })
        .map(|(_, e)| (e.verb, e.target_hint))
        .collect()
}

/// Read children of a legacy Addr from Container 0 inline edges,
/// using the AddrBridge to resolve Addr → PackedDn.
/// Returns target hints as u16 (matching BitpackedCsr's u16 edge format).
pub fn children_from_addr(
    dn_cache: &DnSpineCache,
    bridge: &AddrBridge,
    addr: LegacyAddr,
) -> Vec<u16> {
    let dn = match bridge.dn_for(addr) {
        Some(d) => d,
        None => return Vec::new(),
    };

    let view = match dn_cache.inline_edges(dn) {
        Some(v) => v,
        None => return Vec::new(),
    };

    // Convert inline edges to u16 target addresses via bridge
    view.iter()
        .map(|(_, e)| {
            // The target_hint is a compressed 8-bit value.
            // We return it as u16 for compatibility with BitpackedCsr's edge format.
            e.target_hint as u16
        })
        .collect()
}

/// Out-degree of a legacy Addr via Container 0.
pub fn out_degree_from_addr(
    dn_cache: &DnSpineCache,
    bridge: &AddrBridge,
    addr: LegacyAddr,
) -> usize {
    let dn = match bridge.dn_for(addr) {
        Some(d) => d,
        None => return 0,
    };

    dn_cache.inline_edges(dn)
        .map(|v| v.count())
        .unwrap_or(0)
}

/// Filtered children of a legacy Addr, by verb, via Container 0.
pub fn children_via_from_addr(
    dn_cache: &DnSpineCache,
    bridge: &AddrBridge,
    addr: LegacyAddr,
    verb: u8,
) -> Vec<u16> {
    let dn = match bridge.dn_for(addr) {
        Some(d) => d,
        None => return Vec::new(),
    };

    let view = match dn_cache.inline_edges(dn) {
        Some(v) => v,
        None => return Vec::new(),
    };

    view.iter()
        .filter(|(_, e)| e.verb == verb)
        .map(|(_, e)| e.target_hint as u16)
        .collect()
}
