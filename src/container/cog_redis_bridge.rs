//! CogRedis ↔ DnSpineCache bridge.
//!
//! Provides delegation methods that CogRedis can call to read/write
//! graph edges through the shared DnSpineCache, while keeping the
//! CogRedis API signatures unchanged.
//!
//! # HARD CONTRACT 1: CogRedis struct and API are UNCHANGED
//!
//! This module provides *helper functions* that CogRedis can optionally
//! delegate to. CogRedis's own Vec<CogEdge>, fanout_cache, fanin_cache,
//! and hot_cache remain in place. The bridge adds a *secondary read path*
//! from Container 0 inline edges.
//!
//! # Integration pattern
//!
//! ```text
//! CogRedis.fanout(addr)
//!   ├─ primary: check fanout_cache → scan edges Vec  (existing path)
//!   └─ secondary: if DnSpineCache attached,
//!                  also check Container 0 inline edges
//! ```

use super::adjacency::{PackedDn, EdgeDescriptor};
use super::dn_spine_cache::DnSpineCache;
use super::addr_bridge::{AddrBridge, LegacyAddr, dn_leaf_hint};

/// Edge data extracted from Container 0 inline edges, in CogRedis-compatible form.
/// This is NOT CogEdge — it's a lightweight struct that CogRedis can convert
/// to CogEdge using its own fingerprint data.
#[derive(Debug, Clone)]
pub struct InlineEdgeInfo {
    /// Source DN.
    pub from_dn: PackedDn,
    /// Target hint (low byte of target DN hash).
    pub target_hint: u8,
    /// Verb ID.
    pub verb_id: u8,
    /// Source legacy address (if bridge is available).
    pub from_addr: Option<LegacyAddr>,
}

/// Read outgoing inline edges from a DN's Container 0.
pub fn fanout_from_container(
    dn_cache: &DnSpineCache,
    dn: PackedDn,
) -> Vec<InlineEdgeInfo> {
    let view = match dn_cache.inline_edges(dn) {
        Some(v) => v,
        None => return Vec::new(),
    };

    view.iter()
        .map(|(_, edge)| InlineEdgeInfo {
            from_dn: dn,
            target_hint: edge.target_hint,
            verb_id: edge.verb,
            from_addr: None,
        })
        .collect()
}

/// Read outgoing inline edges with address resolution.
pub fn fanout_resolved(
    dn_cache: &DnSpineCache,
    bridge: &AddrBridge,
    dn: PackedDn,
) -> Vec<InlineEdgeInfo> {
    let from_addr = Some(bridge.addr_for(dn));

    let view = match dn_cache.inline_edges(dn) {
        Some(v) => v,
        None => return Vec::new(),
    };

    view.iter()
        .map(|(_, edge)| InlineEdgeInfo {
            from_dn: dn,
            target_hint: edge.target_hint,
            verb_id: edge.verb,
            from_addr,
        })
        .collect()
}

/// Write an edge into Container 0 inline edges.
/// Returns true if the edge was added (space available).
pub fn bind_inline(
    dn_cache: &mut DnSpineCache,
    from_dn: PackedDn,
    verb_id: u8,
    to_dn: PackedDn,
) -> bool {
    let target_hint = dn_leaf_hint(to_dn);
    dn_cache.add_inline_edge(from_dn, verb_id, target_hint).is_some()
}

/// Remove an edge from Container 0 inline edges.
pub fn unbind_inline(
    dn_cache: &mut DnSpineCache,
    from_dn: PackedDn,
    verb_id: u8,
    to_dn: PackedDn,
) -> bool {
    let target_hint = dn_leaf_hint(to_dn);
    dn_cache.remove_inline_edge(from_dn, verb_id, target_hint)
}

/// Count outgoing edges from a DN's Container 0.
pub fn out_degree_container(dn_cache: &DnSpineCache, dn: PackedDn) -> usize {
    dn_cache.inline_edges(dn)
        .map(|v| v.count())
        .unwrap_or(0)
}

/// Read overflow (CSR) edges from Container 0.
pub fn overflow_edges(
    dn_cache: &DnSpineCache,
    dn: PackedDn,
) -> Vec<EdgeDescriptor> {
    dn_cache.overflow_edges(dn)
        .map(|v| v.edges().filter(|e| !e.is_empty()).collect())
        .unwrap_or_default()
}

/// Total edge count for a DN (inline + overflow).
pub fn total_edges(dn_cache: &DnSpineCache, dn: PackedDn) -> usize {
    let inline = dn_cache.inline_edges(dn)
        .map(|v| v.count())
        .unwrap_or(0);
    let overflow = dn_cache.overflow_edges(dn)
        .map(|v| v.edge_count() as usize)
        .unwrap_or(0);
    inline + overflow
}
