# Deprecated Code -- Algorithm Debt Consolidation

These files were part of the Algorithm Debt Consolidation (commit b704a68).
They built a parallel storage universe (DnSpineCache) instead of extending
the canonical BindSpace model.

The BindSpace Unification (see docs/BINDSPACE_UNIFICATION.md) replaces
these with:
- DnIndex (pure addressing, no data storage) replaces DnSpineCache + AddrBridge
- Container::view() (zero-copy lens) replaces Container conversions
- Merged UDFs in cognitive_udfs.rs replace container_udfs.rs
- Rewritten DnTreeTableProvider reads BindSpace directly

These files are kept for reference. Ideas worth harvesting:
- plasticity.rs: prune/consolidate/rename algorithm
- container_udfs.rs: belichtung, cascade_filter, word_diff, mexican_hat formulas
- dn_spine_cache.rs: SpineCache arena allocator concept
