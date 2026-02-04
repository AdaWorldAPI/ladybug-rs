# Storage Layer Contracts & Race Condition Analysis

> **Document Version**: 1.0.0
> **Last Updated**: 2026-02-04
> **Status**: CRITICAL - Contains known vulnerabilities requiring hardening

---

## Executive Summary

This document catalogs **9 critical race conditions and data loss vulnerabilities** in the storage acceleration stack (Parquet/Arrow, BTree/Snapshots, Redis acceleration). Each issue includes:
- Root cause analysis
- Data loss scenario
- Contract violation
- Recommended fix

**Severity Legend**:
- `CRITICAL` - Data loss under normal operation
- `HIGH` - Data loss under concurrent load
- `MEDIUM` - Inconsistent state, recoverable
- `LOW` - Performance degradation

---

## Table of Contents

1. [LRU Tracker Race Condition](#1-lru-tracker-race-condition)
2. [Write-Ahead Log Non-Atomicity](#2-write-ahead-log-non-atomicity)
3. [WriteBuffer Ordering Violation](#3-writebuffer-ordering-violation)
4. [DependencyGraph Partial Write](#4-dependencygraph-partial-write)
5. [XorDag Parity Update TOCTOU](#5-xordag-parity-update-toctou)
6. [EpochGuard Steal Race](#6-epochguard-steal-race)
7. [TieredStorage Eviction Race](#7-tieredstorage-eviction-race)
8. [SnapshotChain Length Race](#8-snapshotchain-length-race)
9. [TemporalStore Conflict Detection Gap](#9-temporalstore-conflict-detection-gap)

---

## 1. LRU Tracker Race Condition

**Location**: `src/storage/hardening.rs:LruTracker`
**Severity**: `HIGH`
**Component**: Redis acceleration layer

### The Problem

```rust
// src/storage/hardening.rs:66-76
pub fn touch(&self, addr: u16) {
    if let Ok(mut times) = self.access_times.write() {
        times.insert(addr, Instant::now());
    }
    // GAP: Lock released between operations!
    if let Ok(mut order) = self.order.write() {
        order.push_back(addr);  // Duplicate entries possible
        // ...
    }
}
```

### Race Scenario

```
Thread A                          Thread B
─────────────────────────────────────────────────
touch(0x8001)
  write access_times OK
  [lock released]
                                  touch(0x8001)
                                    write access_times OK
                                    [lock released]
                                    write order: push 0x8001
                                    [lock released]
  write order: push 0x8001
  [DUPLICATE in order queue!]
```

### Data Loss Scenario

When `maybe_evict()` runs:
1. Same address appears twice in `order`
2. First eviction succeeds, removes from `access_times`
3. Second eviction attempt reads stale position
4. Evicts WRONG address or panics on missing entry

### Contract Violation

**Expected**: `order.len() == access_times.len()` always
**Actual**: `order.len() >= access_times.len()` after race

### Fix Required

```rust
// Atomic compound operation
pub fn touch(&self, addr: u16) {
    // Hold BOTH locks atomically
    let mut times = self.access_times.write().unwrap();
    let mut order = self.order.write().unwrap();

    times.insert(addr, Instant::now());

    // Remove existing entry before push (dedup)
    order.retain(|&a| a != addr);
    order.push_back(addr);
}
```

---

## 2. Write-Ahead Log Non-Atomicity

**Location**: `src/storage/hardening.rs:WriteAheadLog`
**Severity**: `CRITICAL`
**Component**: Durability layer

### The Problem

```rust
// src/storage/hardening.rs:WalEntry applied BEFORE logged
pub fn log_and_write(&mut self, addr: Addr, fp: [...]) -> Result<()> {
    // Step 1: Write to bind_space (in memory)
    self.bind_space.write_at(addr, fp);

    // Step 2: Log to WAL (disk) -- CRASH HERE = DATA LOSS
    self.wal.append(WalEntry::Write { addr, fp, ... })?;

    // Step 3: Acknowledge
    Ok(())
}
```

### Race Scenario

```
Time    Action                    Disk State           Memory State
────────────────────────────────────────────────────────────────────
T1      write_at(0x8001, fp)      -                    fp @ 0x8001
T2      [POWER FAILURE]           -                    [LOST]
T3      Recovery                  WAL empty            BindSpace empty
                                  DATA LOST!
```

### Data Loss Scenario

1. Client sends write request
2. `write_at()` succeeds in RAM
3. System crash before `wal.append()` completes
4. On recovery, WAL has no record
5. Write is permanently lost

### Contract Violation

**Expected**: If `write()` returns `Ok`, data is durable
**Actual**: Data durable only after WAL sync (not guaranteed)

### Fix Required

```rust
// Write-ahead, not write-behind!
pub fn log_and_write(&mut self, addr: Addr, fp: [...]) -> Result<()> {
    // Step 1: Log to WAL FIRST
    self.wal.append(WalEntry::Write { addr, fp, ... })?;
    self.wal.sync()?;  // fsync!

    // Step 2: Apply to memory
    self.bind_space.write_at(addr, fp);

    Ok(())
}
```

---

## 3. WriteBuffer Ordering Violation

**Location**: `src/storage/resilient.rs:WriteBuffer`
**Severity**: `HIGH`
**Component**: Buffered write layer

### The Problem

```rust
// src/storage/resilient.rs:57-65
pub fn buffer_write(&self, addr: Addr, fp: [...]) -> WriteId {
    let id = self.next_id.fetch_add(1, Ordering::SeqCst);

    // Counter incremented but write not yet buffered
    // WINDOW: pending_count() sees id but write missing

    if let Ok(mut buffer) = self.pending.write() {
        buffer.insert(id, BufferedWrite { ... });
    }
    // ...
}
```

### Race Scenario

```
Thread A (writer)                Thread B (flusher)
─────────────────────────────────────────────────────
next_id.fetch_add(1) -> 42
                                  pending_count() -> "41 pending"
                                  flush_pending() -> flushes 1-41
                                  "all caught up"
pending.insert(42, ...)
                                  [Write 42 ORPHANED until next flush]
```

### Data Loss Scenario

If flush uses `pending_count()` to decide "all synced":
1. Write 42 allocated ID but not in buffer
2. Flusher sees count, flushes, marks "durable"
3. Client thinks write 42 is safe
4. Crash before next flush cycle
5. Write 42 lost

### Contract Violation

**Expected**: `next_id - 1 == max(pending.keys())` always
**Actual**: Gap exists between ID allocation and buffer insertion

### Fix Required

```rust
pub fn buffer_write(&self, addr: Addr, fp: [...]) -> WriteId {
    // Hold lock across ID allocation and insertion
    let mut buffer = self.pending.write().unwrap();
    let id = self.next_id.fetch_add(1, Ordering::SeqCst);
    buffer.insert(id, BufferedWrite { ... });
    id
}
```

---

## 4. DependencyGraph Partial Write

**Location**: `src/storage/resilient.rs:DependencyGraph`
**Severity**: `MEDIUM`
**Component**: Write ordering

### The Problem

```rust
// src/storage/resilient.rs:118-135
pub fn record(&self, id: WriteId, addr: Addr, prev_id: Option<WriteId>) {
    // Write 1: addr_writes
    if let Ok(mut aw) = self.addr_writes.write() {
        aw.entry(addr).or_default().push(id);
    }
    // CRASH HERE = PARTIAL STATE

    // Write 2: dependencies
    if let Ok(mut deps) = self.dependencies.write() {
        if let Some(prev) = prev_id {
            deps.entry(id).or_default().insert(prev);
        }
    }
}
```

### Race Scenario

```
Thread A (record)                 Thread B (can_flush)
───────────────────────────────────────────────────────
addr_writes.insert(id=42)
                                   can_flush(42)?
                                   check dependencies[42]
                                   -> MISSING (not yet written)
                                   -> returns TRUE (no deps)
[crash before deps write]
                                   FLUSHES id=42 out of order!
```

### Data Loss Scenario

1. Write 42 depends on write 41 (same address)
2. `record()` crashes between map updates
3. `can_flush(42)` sees no dependencies
4. Write 42 flushed before 41
5. On recovery, data corruption (wrong order)

### Contract Violation

**Expected**: Atomicity of dependency recording
**Actual**: Two separate map updates, non-atomic

### Fix Required

```rust
pub fn record(&self, id: WriteId, addr: Addr, prev_id: Option<WriteId>) {
    // Single lock scope for both operations
    let mut aw = self.addr_writes.write().unwrap();
    let mut deps = self.dependencies.write().unwrap();

    aw.entry(addr).or_default().push(id);
    if let Some(prev) = prev_id {
        deps.entry(id).or_default().insert(prev);
    }
}
```

---

## 5. XorDag Parity Update TOCTOU

**Location**: `src/storage/xor_dag.rs:update_parity_blocks`
**Severity**: `HIGH`
**Component**: ACID parity protection

### The Problem

```rust
// src/storage/xor_dag.rs:1192-1209
fn update_parity_blocks(&self, parity_ids: &HashSet<u64>, bind_space: &BindSpace) {
    let mut parity_blocks = self.parity_blocks.write()?;

    for &id in parity_ids {
        if let Some(block) = parity_blocks.get_mut(&id) {
            // READ from bind_space (passed in, already released lock)
            let fps: Vec<_> = block.covered_addrs.iter()
                .filter_map(|&a| bind_space.read(a).map(|n| n.fingerprint))
                .collect();

            // WRITE parity based on stale reads
            block.parity_fingerprint = xor_fingerprints_multi(&fps);
        }
    }
}
```

### Race Scenario

```
Thread A (commit)                 Thread B (concurrent write)
────────────────────────────────────────────────────────────
bind_space.write(addr, fp_A)
[release bind_space lock]
                                   bind_space.write(addr, fp_B)
                                   [commit B succeeds]
update_parity_blocks()
  read bind_space[addr] -> fp_B   (not fp_A!)
  compute parity with fp_B

  PARITY IS NOW INCONSISTENT!
  P was computed for fp_A write but includes fp_B
```

### Data Loss Scenario

1. Parity block covers addresses 0x8000-0x80FF
2. Thread A writes 0x8001, releases bind_space lock
3. Thread B writes 0x8001 with different value, commits
4. Thread A computes parity seeing Thread B's value
5. Parity P = A XOR B XOR C... but A is actually B now
6. On recovery, `recover_addr()` returns WRONG data

### Contract Violation

**Expected**: Parity computed atomically with writes
**Actual**: TOCTOU between write and parity computation

### Fix Required

```rust
pub fn commit(&self, txn_id: u64) -> Result<Version, DagError> {
    // Hold bind_space lock through ENTIRE commit
    let mut bind_space = self.bind_space.write()?;

    // Apply writes
    for write in txn.writes.values() {
        bind_space.write_at(write.addr, write.fingerprint);
    }

    // Update parity WHILE STILL HOLDING LOCK
    self.update_parity_blocks_locked(&txn.parity_updates, &bind_space)?;

    Ok(version)
}
```

---

## 6. EpochGuard Steal Race

**Location**: `src/storage/xor_dag.rs:EpochGuard`
**Severity**: `MEDIUM`
**Component**: Work stealing protection

### The Problem

```rust
// src/storage/xor_dag.rs:319-327
pub fn submit_work(&self, item: WorkItem) {
    let epoch = self.epoch.load(Ordering::SeqCst);
    let slot = (epoch % 2) as usize;
    // WINDOW: epoch could advance here
    if let Ok(mut pending) = self.pending[slot].lock() {
        pending.push_back(item);  // Pushed to OLD slot!
    }
}

// src/storage/xor_dag.rs:330-360
pub fn try_steal(&self) -> Option<Vec<WorkItem>> {
    let current_epoch = self.epoch.load(Ordering::SeqCst);
    let old_slot = ((current_epoch + 1) % 2) as usize;
    // Steals from old_slot while submit_work might be adding to it
    ...
}
```

### Race Scenario

```
Thread A (submit)                 Thread B (steal)
────────────────────────────────────────────────────
epoch.load() -> 5
slot = 5 % 2 = 1
                                   epoch.load() -> 5
                                   advance_epoch() -> 6
                                   old_slot = (6+1) % 2 = 1
                                   try_steal(slot=1)
                                   drain pending[1]
pending[1].push(item)
                                   [item MISSED - added after drain]
```

### Data Loss Scenario

1. Work item submitted to slot 1
2. Concurrent epoch advance makes slot 1 the "old" slot
3. Stealer drains slot 1 (misses item being added)
4. Item stuck in slot 1
5. Next epoch advance makes slot 0 "old"
6. Item in slot 1 never processed (orphaned)

### Contract Violation

**Expected**: All submitted work eventually processed
**Actual**: Work can be orphaned if submitted during steal

### Fix Required

```rust
pub fn submit_work(&self, item: WorkItem) {
    // Retry loop with epoch check
    loop {
        let epoch = self.epoch.load(Ordering::SeqCst);
        let slot = (epoch % 2) as usize;

        if let Ok(mut pending) = self.pending[slot].lock() {
            // Re-check epoch under lock
            if self.epoch.load(Ordering::SeqCst) == epoch {
                pending.push_back(item);
                return;
            }
            // Epoch changed, retry with new slot
        }
    }
}
```

---

## 7. TieredStorage Eviction Race

**Location**: `src/storage/snapshots.rs:TieredStorage`
**Severity**: `MEDIUM`
**Component**: Cold storage tier

### The Problem

```rust
// src/storage/snapshots.rs:770-806
fn evict_from_hot(&self, needed: usize) {
    // Step 1: READ candidates (under read lock)
    let candidates: Vec<_> = {
        let hot = self.hot.read().unwrap();
        hot.iter().map(...).collect()
    };
    // READ LOCK RELEASED

    // Step 2: Process candidates one by one
    for (addr, _, size) in candidates {
        // Another thread could write to addr here!
        let data = {
            let hot = self.hot.read().unwrap();
            hot.get(&addr).map(|e| e.data.clone())
        };

        if let Some(data) = data {
            self.migrate_to_warm(addr, &data)?;
            // Step 3: Remove under WRITE lock
            let mut hot = self.hot.write().unwrap();
            // Entry might have been updated since we read it!
            hot.remove(&addr);
        }
    }
}
```

### Race Scenario

```
Thread A (eviction)               Thread B (write)
───────────────────────────────────────────────────────
read candidates: [0x8001]
[release read lock]
                                   write(0x8001, new_data)
                                   [entry updated with fresh data]
read hot[0x8001] -> old_data
migrate_to_warm(old_data)
remove hot[0x8001]
                                   [FRESH DATA LOST!]
```

### Data Loss Scenario

1. Entry 0x8001 marked for eviction (old, stale)
2. Client writes NEW data to 0x8001 (hot, important)
3. Eviction migrates OLD data to warm tier
4. Eviction removes entry from hot
5. New data NEVER written to warm/cold
6. On hot tier capacity limit, data lost

### Contract Violation

**Expected**: Only truly stale data evicted
**Actual**: Fresh writes can be evicted if timing overlaps

### Fix Required

```rust
fn evict_from_hot(&self, needed: usize) {
    let mut hot = self.hot.write().unwrap();  // Hold write lock entire time

    let mut candidates: Vec<_> = hot.iter()
        .filter(|(_, e)| now.duration_since(e.last_access) > threshold)
        .map(|(addr, _)| *addr)
        .collect();

    candidates.sort_by_key(|addr| hot.get(addr).map(|e| e.last_access));

    for addr in candidates {
        if freed >= needed { break; }
        if let Some(entry) = hot.get(&addr) {
            // Re-check staleness under lock
            if now.duration_since(entry.last_access) > threshold {
                self.migrate_to_warm(addr, &entry.data)?;
                hot.remove(&addr);
                freed += entry.data.len();
            }
        }
    }
}
```

---

## 8. SnapshotChain Length Race

**Location**: `src/storage/snapshots.rs:SnapshotChain`
**Severity**: `LOW`
**Component**: Delta snapshot management

### The Problem

```rust
// src/storage/snapshots.rs:231-306
pub fn create_delta(...) -> Result<SnapshotId, SnapshotError> {
    // Step 1: Check chain length
    let chain_len = self.chain_length.load(Ordering::SeqCst) as usize;
    if chain_len >= self.max_chain_length {
        return Err(SnapshotError::ChainTooLong { ... });
    }
    // WINDOW: Another thread could increment here

    // Step 2: Create snapshot (assumes check passed)
    let id = self.next_id.fetch_add(1, Ordering::SeqCst);
    // ...

    // Step 3: Insert and increment counter
    self.snapshots.insert(id, snapshot);
    self.chain_length.fetch_add(1, Ordering::SeqCst);
    // Could now exceed max_chain_length!
}
```

### Race Scenario

```
Thread A                          Thread B
────────────────────────────────────────────────────
chain_length.load() -> 15
max = 16, OK to proceed
                                   chain_length.load() -> 15
                                   max = 16, OK to proceed
                                   create snapshot
                                   chain_length++ -> 16
create snapshot
chain_length++ -> 17
                                   [Chain length 17 > max 16!]
```

### Data Loss Scenario

Not data loss, but:
1. Chain exceeds max length
2. Reconstruction traverses longer chain than expected
3. Memory usage exceeds bounds
4. Potential OOM on constrained systems

### Contract Violation

**Expected**: `chain_length <= max_chain_length` always
**Actual**: Can exceed by number of concurrent creators

### Fix Required

```rust
pub fn create_delta(...) -> Result<SnapshotId, SnapshotError> {
    // Atomic check-and-increment
    loop {
        let current = self.chain_length.load(Ordering::SeqCst);
        if current >= self.max_chain_length {
            return Err(SnapshotError::ChainTooLong { ... });
        }

        // CAS to claim a slot
        if self.chain_length.compare_exchange(
            current, current + 1,
            Ordering::SeqCst, Ordering::SeqCst
        ).is_ok() {
            break;  // Slot claimed
        }
        // CAS failed, retry
    }

    // Now safe to create snapshot
    // ...
}
```

---

## 9. TemporalStore Conflict Detection Gap

**Location**: `src/storage/temporal.rs:check_conflicts`
**Severity**: `HIGH`
**Component**: ACID transaction layer

### The Problem

```rust
// src/storage/temporal.rs:441-462
fn check_conflicts(&self, txn: &Transaction) -> Result<(), TemporalError> {
    let entries = self.entries.read()?;  // Snapshot of entries

    for &addr in &txn.read_set {
        if let Some(indices) = self.addr_index.read()?.get(&addr).cloned() {
            for idx in indices {
                if let Some(entry) = entries.get(idx) {
                    // Check if written after our snapshot
                    if entry.created_version > txn.start_version {
                        return Err(TemporalError::Conflict { ... });
                    }
                }
            }
        }
    }
    // GAP: Another commit could happen between check and our commit
    Ok(())
}

// src/storage/temporal.rs:379-429
pub fn commit(&self, txn_id: TxnId) -> Result<Version, TemporalError> {
    // ...
    if txn.isolation == IsolationLevel::Serializable {
        self.check_conflicts(&txn)?;  // Conflict check
    }
    // GAP: Another thread commits here, modifying addresses we read!

    let commit_version = self.versions.advance();
    // Apply writes (might conflict with concurrent commit)
    // ...
}
```

### Race Scenario

```
Thread A (T1, Serializable)       Thread B (T2)
──────────────────────────────────────────────────────
read(0x8001), record in read_set
                                   write(0x8001, new_value)
                                   commit() -> version 5
check_conflicts()
  entries[0x8001].version = 5
  5 > start_version(4)? NO
  [T2's commit created entry at version 5]
  [But T1 started at version 4]
  CONFLICT MISSED!

advance() -> version 6
apply writes
SERIALIZABLE VIOLATION!
```

### Data Loss Scenario

1. T1 reads X=100 at version 4
2. T2 writes X=200, commits at version 5
3. T1 checks conflicts: sees X written at v5 > v4
4. BUT: `check_conflicts` reads old entry, not new one
5. T1 commits based on stale X=100
6. Lost update: T2's write effectively lost

### Contract Violation

**Expected**: Serializable isolation prevents lost updates
**Actual**: Window between check and commit allows lost updates

### Fix Required

```rust
pub fn commit(&self, txn_id: TxnId) -> Result<Version, TemporalError> {
    // Hold ALL locks through entire commit
    let mut entries = self.entries.write()?;
    let mut addr_index = self.addr_index.write()?;
    let mut edges = self.edges.write()?;

    if txn.isolation == IsolationLevel::Serializable {
        // Check conflicts under WRITE lock
        for &addr in &txn.read_set {
            if let Some(indices) = addr_index.get(&addr) {
                for &idx in indices {
                    if let Some(entry) = entries.get(idx) {
                        if entry.created_version > txn.start_version {
                            return Err(TemporalError::Conflict { ... });
                        }
                    }
                }
            }
        }
    }

    // Advance version and apply writes atomically
    let commit_version = self.versions.advance();
    // ... apply writes under held locks

    Ok(commit_version)
}
```

---

## Interface Contracts Summary

### BindSpace Contract

```rust
/// INVARIANTS:
/// 1. All reads return consistent snapshots
/// 2. All writes are atomic at address granularity
/// 3. No partial fingerprints visible
///
/// THREAD SAFETY:
/// - RwLock protects entire structure
/// - read() requires read lock
/// - write_at() requires write lock
/// - NEVER release lock between read-modify-write
```

### CogRedis Contract

```rust
/// INVARIANTS:
/// 1. GET always returns last successful SET
/// 2. BIND creates edge atomically
/// 3. RESONATE returns consistent similarity scores
///
/// THREAD SAFETY:
/// - Delegates to BindSpace locks
/// - Commands are individually atomic
/// - Multi-command sequences NOT atomic (use transactions)
```

### XorDag Contract

```rust
/// INVARIANTS:
/// 1. Parity P = XOR of all covered fingerprints
/// 2. Any single failure recoverable: missing = P XOR others
/// 3. ACID transactions: all-or-nothing commits
///
/// THREAD SAFETY:
/// - Transactions provide isolation
/// - Parity updates atomic with writes
/// - Work stealing respects epochs
```

### TemporalStore Contract

```rust
/// INVARIANTS:
/// 1. Versions monotonically increase
/// 2. read_at(v) returns state AS OF version v
/// 3. Committed writes never lost (durability)
/// 4. Serializable txns prevent lost updates
///
/// THREAD SAFETY:
/// - MVCC via versioning
/// - Conflict detection for Serializable
/// - Checkpoints are consistent snapshots
```

### TieredStorage Contract

```rust
/// INVARIANTS:
/// 1. Data exists in exactly ONE tier
/// 2. Hot tier bounded by max_hot_size
/// 3. Eviction only affects truly stale data
/// 4. Transparent read across tiers
///
/// THREAD SAFETY:
/// - Per-tier RwLocks
/// - Eviction holds write lock
/// - Migration atomic per-entry
```

---

## Testing Recommendations

### Stress Tests Required

1. **LRU Stress**: 1000 threads touching same address
2. **WAL Crash**: Kill process during write, verify recovery
3. **Parity Consistency**: Concurrent writes, verify `verify_parity()`
4. **Epoch Boundary**: Submit work during epoch advance
5. **Eviction Storm**: Write/read during heavy eviction
6. **Delta Chain**: Concurrent delta creation at max length
7. **Serializable**: Classic lost update scenario with timing

### Fuzz Testing Targets

```
cargo +nightly fuzz run lru_tracker
cargo +nightly fuzz run wal_recovery
cargo +nightly fuzz run parity_update
cargo +nightly fuzz run epoch_guard
cargo +nightly fuzz run tiered_eviction
cargo +nightly fuzz run temporal_conflict
```

---

## Appendix: Lock Ordering Convention

To prevent deadlocks, acquire locks in this order:

```
1. BindSpace (top-level)
2. XorDag.parity_blocks
3. XorDag.active_txns
4. XorDag.delta_chain
5. TemporalStore.entries
6. TemporalStore.addr_index
7. TieredStorage.hot
8. TieredStorage.warm_index
9. TieredStorage.cold_index
10. WriteBuffer.pending
11. DependencyGraph (both maps together)
12. LruTracker (both fields together)
```

**NEVER acquire a higher-numbered lock while holding a lower-numbered lock.**
