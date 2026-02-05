# Technical Debt & Loose Ends

> **Last Updated**: 2026-02-05
> **Severity Scale**: CRITICAL > HIGH > MEDIUM > LOW
> **Reference**: `docs/STORAGE_CONTRACTS.md` for detailed race condition analysis

---

## Executive Summary

The ladybug-rs storage stack contains **9 known race conditions** (2 CRITICAL, 3 HIGH, 3 MEDIUM, 1 LOW), plus additional loose ends in the orchestration layer and Lance integration. None of these affect the core BindSpace addressing or the crewAI orchestration layer, but they block production deployment of the persistence/durability features.

---

## Critical Race Conditions (Data Loss Under Normal Operation)

### 1. WAL is Write-Behind, Not Write-Ahead

**Location**: `src/storage/hardening.rs:WriteAheadLog`
**Severity**: CRITICAL
**Impact**: Data loss on crash

The WAL writes to memory FIRST, then logs to disk. If the process crashes between these steps, the write is lost but the client believes it succeeded.

```rust
// CURRENT (broken):
self.bind_space.write_at(addr, fp);  // Memory first
self.wal.append(entry)?;              // Disk second - CRASH HERE = LOST

// REQUIRED FIX:
self.wal.append(entry)?;              // Disk first
self.wal.sync()?;                     // fsync!
self.bind_space.write_at(addr, fp);  // Memory second
```

**Status**: Not fixed. Workaround: disable WAL in production (accept data loss risk).

### 2. Temporal Store Serializable Conflict Detection Gap

**Location**: `src/storage/temporal.rs:check_conflicts`
**Severity**: HIGH (promoted from STORAGE_CONTRACTS)
**Impact**: Lost updates under serializable isolation

Window between `check_conflicts()` and `commit()` allows concurrent commits to violate serializable isolation. A transaction can commit based on stale reads.

```rust
// CURRENT (broken):
self.check_conflicts(&txn)?;  // Check under READ lock
// GAP: another thread commits here
let version = self.versions.advance(); // Commit

// REQUIRED FIX:
let mut entries = self.entries.write()?;  // WRITE lock for entire commit
// Check conflicts under write lock
// Advance version under write lock
// Apply writes under write lock
```

**Status**: Not fixed. Only affects `IsolationLevel::Serializable` transactions.

---

## High Severity Race Conditions (Data Loss Under Concurrent Load)

### 3. LRU Tracker Duplicate Entries

**Location**: `src/storage/hardening.rs:LruTracker`
**Severity**: HIGH
**Impact**: Wrong address evicted or panic on missing entry

The `touch()` method releases the `access_times` lock before acquiring the `order` lock. Concurrent touches of the same address create duplicate entries in the order queue, causing `maybe_evict()` to evict wrong addresses.

**Fix**: Hold both locks atomically. Dedup before push.

### 4. WriteBuffer ID Allocation Gap

**Location**: `src/storage/resilient.rs:WriteBuffer`
**Severity**: HIGH
**Impact**: Orphaned writes if flusher checks between ID allocation and buffer insertion

The write ID is allocated via `fetch_add` before the entry is inserted into the buffer. A flusher checking `pending_count()` in this window may miss the write.

**Fix**: Hold buffer lock across ID allocation and insertion.

### 5. XorDag Parity TOCTOU

**Location**: `src/storage/xor_dag.rs:update_parity_blocks`
**Severity**: HIGH
**Impact**: Inconsistent parity → wrong data on recovery

Parity is computed after releasing the bind_space lock. A concurrent write can change the data between the write and the parity computation, making the parity inconsistent.

**Fix**: Hold bind_space lock through entire commit (writes + parity update).

---

## Medium Severity Issues (Inconsistent State, Recoverable)

### 6. DependencyGraph Partial Write

**Location**: `src/storage/resilient.rs:DependencyGraph`
**Severity**: MEDIUM
**Impact**: Out-of-order writes on crash between map updates

`record()` writes to `addr_writes` and `dependencies` in separate lock scopes. A crash between them leaves a dependency-free entry that can be flushed out of order.

**Fix**: Single lock scope for both map updates.

### 7. EpochGuard Steal Race

**Location**: `src/storage/xor_dag.rs:EpochGuard`
**Severity**: MEDIUM
**Impact**: Work items orphaned if submitted during epoch advance

`submit_work()` reads epoch, but the epoch can advance before the lock is acquired, causing the item to be pushed to a slot that has already been drained.

**Fix**: Retry loop with epoch re-check under lock.

### 8. TieredStorage Eviction Race

**Location**: `src/storage/snapshots.rs:TieredStorage`
**Severity**: MEDIUM
**Impact**: Fresh data evicted if written during eviction scan

Eviction reads candidates under read lock, then removes under write lock. A write between these steps can be lost when the eviction removes the "stale" entry that now holds fresh data.

**Fix**: Hold write lock for entire eviction. Re-check staleness under lock.

---

## Low Severity Issues

### 9. SnapshotChain Length Race

**Location**: `src/storage/snapshots.rs:SnapshotChain`
**Severity**: LOW
**Impact**: Chain exceeds max_chain_length by concurrent creator count

The length check and increment are non-atomic. Two concurrent `create_delta()` calls can both pass the check and both increment, exceeding the limit.

**Fix**: Compare-and-swap (CAS) loop for atomic check-and-increment.

---

## Lance / S3 Integration Debt

### 10. Lance API Mismatch

**Location**: `src/storage/lance.rs`, `Cargo.toml`
**Severity**: HIGH (blocks S3 integration)

`Cargo.toml` specifies `lance = "1.0"` but the vendored copy is `lance = "2.1.0-beta.0"`. All API calls in `lance.rs` target the 1.0 API.

**To fix**:
1. Add `[patch.crates-io]` to use vendored lance
2. Update `lance.rs` for 2.1 API (`Dataset::query()` changed, schema types moved)
3. Test with `--features lancedb`

### 11. No S3 Backup Implementation

**Location**: `docs/BACKUP_AND_SCHEMA.md` (pseudocode only)
**Severity**: MEDIUM (per user request: S3 should be primary)

The backup architecture document specifies S3 as primary storage tier, but no implementation exists. The `backup_to_s3()` and `restore_from_s3()` functions are pseudocode only.

**Required**:
- Implement S3 via Lance ObjectStore (`vendor/lance/rust/lance-io/src/object_store/providers/aws.rs`)
- Parquet-based full snapshots
- Lifecycle policy for tiering (Standard -> IA -> Glacier)

### 12. No Redis Backup Functions

**Location**: `src/storage/redis_adapter.rs`
**Severity**: MEDIUM

The Redis adapter has type definitions but no actual `backup_to_redis()` or `restore_from_redis()` implementations. XOR delta compression is documented but not coded.

### 13. No PostgreSQL Integration

**Location**: N/A (not implemented)
**Severity**: LOW (deferred)

The backup strategy document mentions PostgreSQL for schema metadata storage on Railway, but no driver code exists.

---

## Orchestration Layer Debt

### 14. Hierarchical Dispatch Not Differentiated

**Location**: `src/orchestration/crew_bridge.rs:dispatch_crew()`
**Severity**: LOW

The `"hierarchical"` dispatch mode currently behaves identically to `"sequential"`. In a proper hierarchical crew, the first task should go to a manager agent who delegates the rest via A2A.

### 15. Task Dependency Resolution Not Implemented

**Location**: `src/orchestration/crew_bridge.rs`
**Severity**: LOW

`CrewTask.depends_on` field exists but is not enforced during dispatch. Tasks with dependencies are submitted regardless of whether their prerequisites are complete.

### 16. A2A Message Delivery Not Reliable

**Location**: `src/orchestration/a2a.rs:receive()`
**Severity**: LOW

The `receive()` method only finds messages with `status == Pending`, but `send()` marks messages as `Delivered` immediately. This means `receive()` returns no messages unless the status handling is corrected.

**Fix**: Either keep messages as `Pending` until `receive()` is called, or filter by `Delivered` status.

### 17. sci/v1 Not Wired to crewAI

**Location**: `src/flight/crew_actions.rs`
**Severity**: LOW

The `sci.query` action type is listed in the documentation but not implemented in crew_actions.rs. Research agents cannot currently route sci/v1 queries through the Flight bridge.

---

## Build/Test Debt

### 18. 10 Test Failures with Experimental Features

**Status**: Known, documented in CLAUDE.md

| Test | Error | Root Cause |
|------|-------|------------|
| `collapse_gate::test_sd_calculation` | threshold | Algorithm logic |
| `quantum_ops::test_permute_adjoint` | not inverse | Permute logic |
| `cypher::test_variable_length` | ParseFloatError | Tokenizer |
| `causal_ops::test_store_query_correlation` | empty | Query issue |
| `causal::test_correlation_store` | empty | Query issue |
| `context_crystal::test_temporal_flow` | popcount=0 | Insert not persisting |
| `nsm_substrate::test_codebook_initialization` | primes<60 | Init issue |
| `nsm_substrate::test_learning` | vocab<65 | Learning issue |
| `jina_cache::test_cache_hit_rate` | off-by-one | Trivial fix |
| `crystal_lm::test_serialize` | None unwrap | Validation strict |

These are in experimental feature modules, not in the core or orchestration layers.

---

## Fix Priority

| Priority | Issues | Rationale |
|----------|--------|-----------|
| **P0** (fix before production) | #1 (WAL), #2 (Temporal), #5 (Parity) | Data corruption/loss |
| **P1** (fix before scale) | #3 (LRU), #4 (WriteBuffer), #8 (Eviction) | Concurrent access bugs |
| **P2** (fix when needed) | #6 (DepGraph), #7 (Epoch), #10 (Lance) | Correctness/integration |
| **P3** (nice to have) | #9 (Chain), #14-#17 (Orchestration), #18 (Tests) | Polish/completeness |

---

## Lock Ordering Convention

To prevent deadlocks when fixing these races, acquire locks in this order:

```
1.  BindSpace (top-level)
2.  XorDag.parity_blocks
3.  XorDag.active_txns
4.  XorDag.delta_chain
5.  TemporalStore.entries
6.  TemporalStore.addr_index
7.  TieredStorage.hot
8.  TieredStorage.warm_index
9.  TieredStorage.cold_index
10. WriteBuffer.pending
11. DependencyGraph (both maps together)
12. LruTracker (both fields together)
```

**NEVER acquire a higher-numbered lock while holding a lower-numbered lock.**
