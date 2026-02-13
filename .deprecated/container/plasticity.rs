//! Post-insert plasticity: prune, consolidate, rename.
//!
//! Runs entirely in-memory on DnSpineCache. No Lance I/O until flush.
//!
//! After a leaf is inserted and a spine recomputed, plasticity checks:
//! 1. **Prune**: Is any child now too far from the recomputed spine?
//!    If so, find a better spine and reparent.
//! 2. **Consolidate**: Is any sibling spine now too close?
//!    If so, merge them (bundle + reparent children).
//! 3. **Rename**: Does the merged/changed spine match a codebook entry?
//!    If so, label it.

use super::Container;
use super::adjacency::PackedDn;
use super::dn_spine_cache::{DnSpineCache, PRUNE_THRESHOLD, CONSOLIDATION_THRESHOLD};
use super::search::belichtungsmesser;

/// Results of a plasticity pass.
#[derive(Debug, Default, Clone)]
pub struct PlasticityOps {
    /// Children that were moved to a better spine: (child_dn, new_spine_dn).
    pub pruned: Vec<(PackedDn, PackedDn)>,

    /// Spines that were merged: (survivor_dn, absorbed_dn).
    pub merged: Vec<(PackedDn, PackedDn)>,

    /// Spines that were renamed from codebook: (spine_dn, new_label).
    pub renamed: Vec<(PackedDn, String)>,
}

impl PlasticityOps {
    /// True if any plasticity operations were performed.
    pub fn any_changes(&self) -> bool {
        !self.pruned.is_empty() || !self.merged.is_empty() || !self.renamed.is_empty()
    }

    /// Total number of operations.
    pub fn total_ops(&self) -> usize {
        self.pruned.len() + self.merged.len() + self.renamed.len()
    }
}

/// Run plasticity on a changed spine.
///
/// # Arguments
/// - `cache`: The DnSpineCache
/// - `changed_spine_dn`: The spine that was just recomputed
/// - `codebook`: Reference containers for labeling. Each entry is (label, container).
/// - `prune_threshold`: Hamming distance above which a child is reparented
/// - `consolidation_threshold`: Hamming distance below which sibling spines merge
pub fn post_insert_plasticity(
    cache: &mut DnSpineCache,
    changed_spine_dn: PackedDn,
    codebook: &[(&str, &Container)],
    prune_threshold: u32,
    consolidation_threshold: u32,
) -> PlasticityOps {
    let mut ops = PlasticityOps::default();

    // Read the current spine content
    let spine = match cache.read_content(changed_spine_dn) {
        Some(s) => s.clone(),
        None => return ops,
    };

    // ========================================================================
    // PRUNE: child too far from recomputed spine?
    // ========================================================================
    let children = cache.children_dns(changed_spine_dn);
    for child_dn in children {
        let child = match cache.read_content(child_dn) {
            Some(c) => c.clone(),
            None => continue,
        };
        let dist = spine.hamming(&child);
        if dist <= prune_threshold {
            continue;
        }
        // Find a better spine
        if let Some((better_dn, _)) = cache
            .find_closest_spine(&child)
            .filter(|(dn, d)| *dn != changed_spine_dn && *d < dist)
        {
            // Reparent to better spine
            let slots = (
                cache.slot(changed_spine_dn),
                cache.slot(better_dn),
                cache.slot(child_dn),
            );
            if let (Some(old_slot), Some(new_slot), Some(child_slot)) = slots {
                cache.cache.reparent(
                    child_slot.content_idx,
                    old_slot.content_idx,
                    new_slot.content_idx,
                );
                ops.pruned.push((child_dn, better_dn));
            }
        }
    }

    // ========================================================================
    // CONSOLIDATE: sibling spine too close?
    // ========================================================================
    if let Some(parent_dn) = changed_spine_dn.parent() {
        let siblings = cache.children_dns(parent_dn);
        for sib_dn in siblings {
            if sib_dn == changed_spine_dn {
                continue;
            }

            let sib_slot = match cache.slot(sib_dn) {
                Some(s) if s.is_spine => s,
                _ => continue,
            };
            let _ = sib_slot;

            let sib = match cache.read_content(sib_dn) {
                Some(s) => s.clone(),
                None => continue,
            };

            let delta = spine.hamming(&sib);
            if delta < consolidation_threshold {
                // Merge: bundle the two spines
                let merged = Container::bundle(&[&spine, &sib]);

                // Merge spines in cache
                cache.merge_spines(changed_spine_dn, sib_dn, &merged);

                // Codebook probe
                if let Some(name) = codebook_probe(codebook, &merged) {
                    cache.set_label(changed_spine_dn, &name);
                    ops.renamed.push((changed_spine_dn, name));
                }

                ops.merged.push((changed_spine_dn, sib_dn));

                // Only merge one sibling per plasticity pass to avoid cascading
                break;
            }
        }
    }

    // ========================================================================
    // RENAME: codebook probe on the (possibly merged) spine
    // ========================================================================
    if ops.merged.is_empty() && !codebook.is_empty() {
        // Even without merging, check if spine now matches a codebook entry
        let spine_now = match cache.read_content(changed_spine_dn) {
            Some(s) => s.clone(),
            None => return ops,
        };
        if let Some(name) = codebook_probe(codebook, &spine_now) {
            cache.set_label(changed_spine_dn, &name);
            ops.renamed.push((changed_spine_dn, name));
        }
    }

    ops
}

/// Probe codebook for the closest match using Belichtungsmesser.
/// Returns the label if within 2σ of the codebook entry.
pub fn codebook_probe(codebook: &[(&str, &Container)], spine: &Container) -> Option<String> {
    let threshold = super::EXPECTED_DISTANCE - 2 * super::SIGMA_APPROX; // ~4006

    let mut best: Option<(&str, u32)> = None;
    for &(label, entry) in codebook {
        let dist = belichtungsmesser(spine, entry);
        match &best {
            None if dist < threshold => best = Some((label, dist)),
            Some((_, best_dist)) if dist < *best_dist => best = Some((label, dist)),
            _ => {}
        }
    }

    best.map(|(label, _)| label.to_string())
}

/// Run plasticity with default thresholds.
pub fn plasticity_default(
    cache: &mut DnSpineCache,
    changed_spine_dn: PackedDn,
    codebook: &[(&str, &Container)],
) -> PlasticityOps {
    post_insert_plasticity(
        cache,
        changed_spine_dn,
        codebook,
        PRUNE_THRESHOLD,
        CONSOLIDATION_THRESHOLD,
    )
}
