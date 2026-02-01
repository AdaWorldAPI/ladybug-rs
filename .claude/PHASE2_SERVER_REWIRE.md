# Phase 2: Rewire server.rs to Use RedisAdapter

**Date**: 2026-02-01
**Repo**: `https://github.com/AdaWorldAPI/ladybug-rs`
**Branch**: Create `phase2/server-rewire` from `main` (commit `98bcaec9`)
**Binary**: `src/bin/server.rs` → `ladybug-server`

---

## The Problem

`src/bin/server.rs` (724 lines) imports and uses `CogRedis` directly:

```rust
// line 25
use ladybug::storage::cog_redis::{CogRedis, SetOptions, RedisResult, CamResult};

// line 64
struct DatabaseState {
    cog_redis: CogRedis,
}

// line 70
cog_redis: CogRedis::new(),
```

Every endpoint calls `db.cog_redis.execute_command()` or `db.cog_redis.*` methods.

Meanwhile, `RedisAdapter` (981 lines in `src/storage/redis_adapter.rs`) wraps `Substrate` (1,171 lines in `src/storage/substrate.rs`) which owns `BindSpace` as the single source of truth, runs `RubiconSearch` for similarity, tracks fluid TTL, and does crystallize/evaporate lifecycle.

**All of that is dead code from the running server's perspective.** The entire Substrate stack — 66K+ lines of architecture — is bypassed.

---

## The Fix

Replace `CogRedis` with `RedisAdapter` in `server.rs`. The API surface is nearly identical — both have `execute()` / `execute_command()` methods. The difference is the return types and command enums.

### What Changes

| Current (CogRedis) | Target (RedisAdapter) |
|--------------------|-----------------------|
| `use ladybug::storage::cog_redis::{CogRedis, SetOptions, RedisResult, CamResult}` | `use ladybug::storage::redis_adapter::{RedisAdapter, RedisResult, RedisCommand, SetOptions, CamResult, NodeResult, SearchHit, EdgeResult}` |
| `cog_redis: CogRedis` | `adapter: RedisAdapter` |
| `CogRedis::new()` | `RedisAdapter::default_new()` |
| `db.cog_redis.execute_command(command_string)` | `db.adapter.execute(command_string)` |
| `db.cog_redis.set(truncated, opts)` | `db.adapter.execute_command(RedisCommand::Set { ... })` |
| `db.cog_redis.query_pattern(&pattern, threshold)` | `db.adapter.execute_command(RedisCommand::Search { ... })` |
| `db.cog_redis.execute_cam_named(op_name, &args)` | `db.adapter.execute_command(RedisCommand::Cam { ... })` |

### RedisAdapter API Reference

```rust
// src/storage/redis_adapter.rs

pub struct RedisAdapter {
    substrate: Substrate,
    key_map: HashMap<String, CogAddr>,
}

impl RedisAdapter {
    pub fn new(config: SubstrateConfig) -> Self;
    pub fn default_new() -> Self;
    pub fn substrate(&self) -> &Substrate;
    pub fn substrate_mut(&mut self) -> &mut Substrate;

    // String command → parse → execute
    pub fn execute(&mut self, command: &str) -> RedisResult;

    // Typed command → execute
    pub fn execute_command(&mut self, cmd: RedisCommand) -> RedisResult;
}
```

### RedisResult Variants (redis_adapter)

```rust
pub enum RedisResult {
    Ok,
    String(String),
    Integer(i64),
    Float(f64),
    Addr(CogAddr),
    Array(Vec<RedisResult>),
    Node(NodeResult),
    Search(Vec<SearchHit>),
    Edge(EdgeResult),
    Error(String),
    Nil,
}

pub struct NodeResult {
    pub addr: CogAddr,
    pub fingerprint: [u64; FINGERPRINT_WORDS],
    pub label: Option<String>,
    pub qidx: u8,
    pub popcount: u32,
    pub tier: Tier,
}

pub struct SearchHit {
    pub addr: CogAddr,
    pub distance: u32,
    pub similarity: f32,
    pub label: Option<String>,
}

pub struct EdgeResult {
    pub from: CogAddr,
    pub verb: CogAddr,
    pub to: CogAddr,
    pub fingerprint: [u64; FINGERPRINT_WORDS],
    pub weight: f32,
}
```

### RedisCommand Enum

```rust
pub enum RedisCommand {
    Get { key: String },
    Set { key: String, value: String, options: SetOptions },
    Del { key: String, mode: DeleteMode },
    Bind { from: String, verb: String, to: String },
    Unbind { edge: String, known: String },
    Resonate { query: String, k: usize },
    Search { query: String, k: usize, threshold: f32 },
    Traverse { start: String, verb: String, hops: usize },
    Fanout { addr: String },
    Fanin { addr: String },
    Crystallize { addr: String },
    Evaporate { addr: String },
    Tick,
    Cam { operation: String, args: Vec<String> },
    Info,
    Stats,
    Ping,
    Unknown(String),
}
```

### CamResult (redis_adapter)

```rust
pub enum CamResult {
    Fingerprint([u64; FINGERPRINT_WORDS]),
    Addr(CogAddr),
    Addresses(Vec<CogAddr>),
    Value(f64),
    String(String),
    Error(String),
}
```

---

## Endpoint-by-Endpoint Mapping

### 1. `/redis` (handle_redis_command, line 277)

**Current**: Parses JSON `{"command": "..."}`, calls `db.cog_redis.execute_command(command_string)`.

**New**: Same parse, call `db.adapter.execute(command_string)` — the string parser is built into RedisAdapter.

**Handle new result variants**: `RedisResult::Node(n)` → serialize addr, label, popcount, qidx. `RedisResult::Search(hits)` → serialize as array of {addr, distance, similarity}. `RedisResult::Edge(e)` → serialize from, verb, to, weight. `RedisResult::Float(f)` → serialize as number.

### 2. `/sql` (handle_sql_query, line 337)

**Current**: Maps SQL to a Redis command string, calls `db.cog_redis.execute_command()`. This is a stub.

**New**: Same approach, call `db.adapter.execute()`. The SQL endpoint remains a stub — real SQL goes through `Database` (async, LanceDB-backed). Note this for future Phase 3.

### 3. `/cypher` (handle_cypher_query, line 386)

**Current**: Returns stub JSON. Does not call CogRedis at all.

**New**: No change needed. The `_state` parameter is already unused.

### 4. `/vectors/search` (handle_vector_search, line 439)

**Current**: Creates `Fingerprint`, calls `db.cog_redis.query_pattern(&pattern, threshold)`, manually computes Hamming distance for each result.

**New**: Use `RedisCommand::Search` or `RedisCommand::Resonate`:
```rust
let result = db.adapter.execute_command(RedisCommand::Search {
    query: content_or_hex,
    k,
    threshold,
});
// result is RedisResult::Search(Vec<SearchHit>)
// SearchHit already has .distance and .similarity computed
```

The manual Hamming distance computation and 157→156 truncation hack go away — RedisAdapter handles fingerprint sizing internally.

### 5. `/vectors/insert` (handle_vector_insert, line 509)

**Current**: Creates `Fingerprint`, truncates to 156 u64s, calls `db.cog_redis.set(truncated, opts)`.

**New**: Use `RedisCommand::Set`:
```rust
let result = db.adapter.execute_command(RedisCommand::Set {
    key: label_or_auto,
    value: content,
    options: RedisSetOptions { promote: true, ..Default::default() },
});
// result is RedisResult::Addr(addr) on success
```

### 6. `/cam/:operation` (handle_cam_operation, line 542)

**Current**: Calls `db.cog_redis.execute_cam_named(op_name, &args)`, returns `CamResult`.

**New**: Use `RedisCommand::Cam`:
```rust
let result = db.adapter.execute_command(RedisCommand::Cam {
    operation: op_name.to_string(),
    args: args.iter().map(|a| a.to_string()).collect(),
});
```

**Note**: Check whether RedisAdapter's `cmd_cam` returns a `RedisResult` wrapping `CamResult`, or if you need to add `CamResult` handling. If `cmd_cam` currently returns `RedisResult::String(...)` or similar, the JSON serialization in the handler may need adjustment. Read `cmd_cam` implementation at line 557 of `redis_adapter.rs`.

### 7. `/fingerprint` (handle_fingerprint_create, line 618)

**Current**: Uses `Fingerprint::from_content()` directly. No CogRedis call.

**New**: No change needed.

### 8. `/health`, `/`, `/info`

**Current**: Static JSON responses.

**New**: `/stats` and `/info` can optionally pull from `db.adapter.execute_command(RedisCommand::Stats)` for live data. But this is optional — the rewire itself doesn't require it.

---

## Implementation Steps

### Step 1: Create Branch

```bash
git checkout main
git pull
git checkout -b phase2/server-rewire
```

### Step 2: Update Imports (line 25)

Replace:
```rust
use ladybug::storage::cog_redis::{CogRedis, SetOptions, RedisResult, CamResult};
```

With:
```rust
use ladybug::storage::{
    // Redis Adapter (routes to Substrate → BindSpace)
    RedisAdapter, RedisResult, RedisCommand, DeleteMode,
    RedisSetOptions,  // ← aliased in mod.rs to avoid collision with cog_redis::SetOptions
    NodeResult, SearchHit, EdgeResult, CamResult,
    // Address types (still exported from cog_redis, needed for JSON serialization)
    CogAddr, Tier,
};
```

**Import note**: `mod.rs` re-exports `redis_adapter::SetOptions` as `RedisSetOptions` to avoid collision with `cog_redis::SetOptions`. Use `RedisSetOptions` throughout server.rs. All types above are re-exported from `ladybug::storage` — no need to reach into submodules directly.

### Step 3: Update DatabaseState (line 62-71)

```rust
struct DatabaseState {
    adapter: RedisAdapter,
}

impl DatabaseState {
    fn new() -> Self {
        Self {
            adapter: RedisAdapter::default_new(),
        }
    }
}
```

### Step 4: Update Each Handler

Work through handlers 1-6 above. For each:
1. Change `db.cog_redis.*` to `db.adapter.*`
2. Handle new `RedisResult` variants in JSON serialization
3. Remove the 157→156 truncation hacks
4. Remove manual Hamming distance computations where `SearchHit` already provides them

### Step 5: Build and Test

```bash
# Must compile with the features Railway uses
FEATURES="simd,parallel,codebook,hologram,quantum"
cargo build --bin ladybug-server --features "$FEATURES"

# Run tests
cargo test --features "spo,quantum,codebook"
```

Compilation is the primary gate. If `server.rs` compiles against `RedisAdapter`, the wiring is correct.

### Step 6: Smoke Test

```bash
# Start server locally
cargo run --bin ladybug-server --features "simd,parallel,codebook,hologram,quantum" &

# Health
curl http://127.0.0.1:5000/health

# Insert
curl -X POST http://127.0.0.1:5000/vectors/insert \
  -H "Content-Type: application/json" \
  -d '{"vectors": [{"content": "hello world"}]}'

# Search
curl -X POST http://127.0.0.1:5000/vectors/search \
  -H "Content-Type: application/json" \
  -d '{"query": "hello", "k": 5, "threshold": 0.3}'

# Redis command
curl -X POST http://127.0.0.1:5000/redis \
  -H "Content-Type: application/json" \
  -d '{"command": "PING"}'

# Stats
curl -X POST http://127.0.0.1:5000/redis \
  -H "Content-Type: application/json" \
  -d '{"command": "STATS"}'
```

### Step 7: Commit and PR

```bash
git add src/bin/server.rs
git commit -m "Phase 2: rewire server.rs to use RedisAdapter → Substrate

Replace CogRedis with RedisAdapter in HTTP server. This activates
the full Substrate stack: BindSpace as single source of truth,
RubiconSearch for similarity, fluid lifecycle (crystallize/evaporate),
and write buffering for future Lance persistence.

Breaking changes: None (HTTP API unchanged)
Internal: server now routes through RedisAdapter → Substrate → BindSpace
instead of CogRedis → raw HashMap"

git push origin phase2/server-rewire
```

---

## What NOT To Do

1. **Do NOT rewrite the HTTP layer.** Keep the hand-rolled TcpListener. Axum migration is Phase 3 — separate concern.
2. **Do NOT touch RedisAdapter or Substrate.** Those are correct. This PR only changes `server.rs`.
3. **Do NOT add Lance connection logic.** Lance persistence is Phase 4. The WAL and write buffer exist but don't flush yet. That's fine.
4. **Do NOT add async.** `RedisAdapter` is sync. The current server is sync (thread-per-connection). Async migration comes with axum.
5. **Do NOT update CLAUDE.md in this PR.** One file changed, one concern.
6. **Do NOT change the HTTP API contract.** Same endpoints, same JSON shapes. Callers see no difference.

---

## Verification Checklist

After the PR, verify:

- [ ] `src/bin/server.rs` has ZERO references to `cog_redis` module
- [ ] `grep -r "cog_redis" src/bin/` returns nothing
- [ ] `grep "RedisAdapter" src/bin/server.rs` returns the import and struct field
- [ ] `cargo build --bin ladybug-server --features "simd,parallel,codebook,hologram,quantum"` succeeds
- [ ] `cargo test --features "spo,quantum,codebook"` passes same count as before (no regressions)
- [ ] `/health` returns 200
- [ ] `/redis` with `{"command": "PING"}` returns `{"success": true, "result": "PONG"}`
- [ ] `/vectors/insert` returns an address in the response
- [ ] `/vectors/search` returns results with distance and similarity fields
- [ ] No 157→156 truncation hacks remain in server.rs
- [ ] No manual Hamming distance computation in server.rs (SearchHit has it)

---

## Context for the Agent

**Why this matters**: Railway currently runs a server that uses `CogRedis` — a flat HashMap with no lifecycle, no search cascade, no fluid zone. The Substrate/RedisAdapter stack has all of this, but it's dead code because `server.rs` never calls it. This single file change activates 66K lines of architecture.

**Difficulty**: Low-medium. The API shapes are nearly identical. The main work is mapping `RedisResult` variants to JSON serialization, which is straightforward pattern matching.

**Risk**: Low. The HTTP API contract doesn't change. If RedisAdapter's execute() handles the same command strings as CogRedis's execute_command(), the switch is transparent to callers.

**Time estimate**: 30-60 minutes for an agent that reads the types carefully.
