//! Spine-guided leaf insertion with automatic branch splitting.
//!
//! The insert algorithm has three paths, all O(1) in terms of containers touched:
//!
//! 1. **Sibling leaf**: leaf is close to an existing child under the best spine.
//!    1 container write, 1 spine dirty. The cheapest case.
//!
//! 2. **New sub-branch**: leaf is related to a spine's child but divergent enough
//!    to warrant a new sub-spine. 2 container writes, 2 spines dirty.
//!    This is automatic taxonomy refinement — the popcount of the delta IS the
//!    signal that a split is needed.
//!
//! 3. **New top-level branch**: leaf doesn't resonate with any existing spine.
//!    1 container write, new spine allocated, summary dirty. Rare for mature trees.
//!
//! None of these paths touches uninvolved branches. The total write amplification
//! is bounded by O(depth) = O(log n), with a constant of 1 KB XOR per level.

use super::Container;
use super::cache::CacheError;
use super::search::belichtungsmesser;
use super::spine::SpineCache;

/// Default split threshold: if delta popcount between a leaf and its nearest
/// sibling exceeds this, create a new sub-spine instead of inserting as sibling.
///
/// Tuned for 8K containers: ~2σ of random Hamming distance.
/// Siblings closer than this share enough signal to live under the same spine.
/// Beyond this, the divergence justifies structural separation.
pub const SPLIT_THRESHOLD: u32 = 2000;

/// Threshold for "doesn't resonate with any spine at all".
/// If the Belichtungsmesser estimate to the best spine exceeds this,
/// the leaf gets its own top-level branch.
///
/// Set to ~EXPECTED_DISTANCE - 1σ. Anything beyond random noise.
pub const RESONANCE_THRESHOLD: u32 = 4050;

/// Result of a leaf insertion, describing which path was taken.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InsertResult {
    /// Path 1: leaf added as sibling under an existing spine.
    Sibling {
        /// Index of the newly allocated leaf slot.
        leaf_idx: usize,
        /// Index of the parent spine that absorbed the leaf.
        spine_idx: usize,
    },

    /// Path 2: leaf triggered creation of a new sub-spine.
    /// The closest existing child was reparented under the new sub-spine.
    SubBranch {
        /// Index of the newly allocated leaf slot.
        leaf_idx: usize,
        /// Index of the newly created sub-spine.
        new_spine_idx: usize,
        /// Index of the parent spine the sub-spine was attached to.
        parent_spine_idx: usize,
        /// The existing child that was reparented under the new sub-spine.
        reparented_child: usize,
    },

    /// Path 3: leaf doesn't resonate with any existing spine.
    /// A new top-level branch was created.
    NewBranch {
        /// Index of the newly allocated leaf slot.
        leaf_idx: usize,
        /// Index of the newly created spine (top-level).
        spine_idx: usize,
    },
}

impl InsertResult {
    /// The index of the inserted leaf, regardless of which path was taken.
    pub fn leaf_idx(&self) -> usize {
        match self {
            InsertResult::Sibling { leaf_idx, .. } => *leaf_idx,
            InsertResult::SubBranch { leaf_idx, .. } => *leaf_idx,
            InsertResult::NewBranch { leaf_idx, .. } => *leaf_idx,
        }
    }

    /// Number of containers written (excluding dirty spine recomputations).
    pub fn containers_written(&self) -> usize {
        match self {
            InsertResult::Sibling { .. } => 1,
            InsertResult::SubBranch { .. } => 2,
            InsertResult::NewBranch { .. } => 1,
        }
    }

    /// Number of spines marked dirty.
    pub fn spines_dirtied(&self) -> usize {
        match self {
            InsertResult::Sibling { .. } => 1,
            InsertResult::SubBranch { .. } => 2,
            InsertResult::NewBranch { .. } => 1,
        }
    }
}

/// Insert a leaf into a spine-managed tree.
///
/// Uses Belichtungsmesser (7-point estimator, ~14 cycles per spine) to find the
/// closest spine, then checks delta popcount against `split_threshold` to decide
/// between sibling insertion and sub-branch creation.
///
/// # Arguments
/// - `cache`: The spine cache managing the tree
/// - `leaf`: The container to insert (must be non-zero)
/// - `summary_idx`: Index of the tree's summary/root spine (if any)
/// - `split_threshold`: Delta popcount above which a sub-branch is created.
///   Pass `SPLIT_THRESHOLD` for the default.
///
/// # Returns
/// - `Ok(InsertResult)` describing which path was taken
/// - `Err(CacheError)` if the leaf is zero or allocation fails
pub fn insert_leaf(
    cache: &mut SpineCache,
    leaf: &Container,
    summary_idx: Option<usize>,
    split_threshold: u32,
) -> Result<InsertResult, CacheError> {
    if leaf.is_zero() {
        return Err(CacheError::ZeroContainer { idx: usize::MAX });
    }

    let spine_indices = cache.spine_indices();

    // No spines exist yet — create a fresh top-level branch
    if spine_indices.is_empty() {
        return create_new_branch(cache, leaf, summary_idx);
    }

    // Step 1: Belichtungsmesser scan against all spines.
    // ~14 cycles per spine. For 100 spines = ~1400 cycles ≈ 0.5 µs.
    let mut best_spine: Option<(usize, u32)> = None;
    for &spine_idx in &spine_indices {
        let spine_data = cache.read(spine_idx);
        let estimate = belichtungsmesser(spine_data, leaf);
        match &best_spine {
            None => best_spine = Some((spine_idx, estimate)),
            Some((_, best_dist)) if estimate < *best_dist => {
                best_spine = Some((spine_idx, estimate));
            }
            _ => {}
        }
    }

    let (best_spine_idx, best_dist) = best_spine.unwrap();

    // Step 2: Does the leaf resonate with any spine at all?
    if best_dist > RESONANCE_THRESHOLD {
        return create_new_branch(cache, leaf, summary_idx);
    }

    // Step 3: Check delta against children of the winning spine.
    let children: Vec<usize> = cache.spine_children(best_spine_idx).to_vec();

    if children.is_empty() {
        // Spine declared but no children — leaf becomes the first child
        let leaf_idx = cache.push_leaf(leaf)?;
        cache.add_child_to_spine(best_spine_idx, leaf_idx);
        mark_summary_dirty(cache, summary_idx);
        return Ok(InsertResult::Sibling {
            leaf_idx,
            spine_idx: best_spine_idx,
        });
    }

    // Find closest child by exact XOR popcount (not Belichtungsmesser —
    // children are few, so we can afford exact distance)
    let mut closest_child: usize = children[0];
    let mut min_delta: u32 = u32::MAX;
    for &child_idx in &children {
        let child_data = cache.read(child_idx);
        let delta_pc = child_data.xor(leaf).popcount();
        if delta_pc < min_delta {
            min_delta = delta_pc;
            closest_child = child_idx;
        }
    }

    // Step 4: Sibling or sub-branch?
    if min_delta < split_threshold {
        // Path 1: Close enough — append as sibling leaf
        let leaf_idx = cache.push_leaf(leaf)?;
        cache.add_child_to_spine(best_spine_idx, leaf_idx);
        mark_summary_dirty(cache, summary_idx);
        Ok(InsertResult::Sibling {
            leaf_idx,
            spine_idx: best_spine_idx,
        })
    } else {
        // Path 2: Too divergent — create new sub-spine
        //
        // The existing closest_child gets reparented under a new sub-spine,
        // together with the new leaf. The sub-spine then replaces
        // closest_child under best_spine_idx.
        let leaf_idx = cache.push_leaf(leaf)?;
        let new_spine_idx = cache.push_spine(vec![closest_child, leaf_idx]);

        // Reparent: closest_child moves from best_spine to new_spine.
        // (push_spine already declared new_spine over [closest_child, leaf_idx],
        //  so we only need to remove closest_child from best_spine and add
        //  new_spine_idx under best_spine instead.)
        if let Some(children) = cache.spines().get(&best_spine_idx).cloned() {
            // Remove closest_child from old spine, add new_spine_idx
            let mut updated: Vec<usize> = children
                .into_iter()
                .filter(|&c| c != closest_child)
                .collect();
            updated.push(new_spine_idx);

            // Re-declare the parent spine with updated children
            // (We need to clear old mapping and re-declare)
            cache.redeclare_spine(best_spine_idx, updated);
        }

        mark_summary_dirty(cache, summary_idx);

        Ok(InsertResult::SubBranch {
            leaf_idx,
            new_spine_idx,
            parent_spine_idx: best_spine_idx,
            reparented_child: closest_child,
        })
    }
}

/// Path 3: Create a new top-level branch (spine + leaf).
fn create_new_branch(
    cache: &mut SpineCache,
    leaf: &Container,
    summary_idx: Option<usize>,
) -> Result<InsertResult, CacheError> {
    let leaf_idx = cache.push_leaf(leaf)?;
    let spine_idx = cache.push_spine(vec![leaf_idx]);

    mark_summary_dirty(cache, summary_idx);

    Ok(InsertResult::NewBranch {
        leaf_idx,
        spine_idx,
    })
}

/// Mark the summary spine dirty if one is designated.
fn mark_summary_dirty(cache: &mut SpineCache, summary_idx: Option<usize>) {
    if let Some(idx) = summary_idx {
        cache.cache.mark_dirty(idx);
    }
}
