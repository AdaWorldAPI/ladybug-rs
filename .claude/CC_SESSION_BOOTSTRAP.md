# CC_SESSION_BOOTSTRAP.md

## Clone. Check. Start.

```bash
mkdir adaworld && cd adaworld

# REQUIRED (ladybug-rs won't compile without these):
git clone https://github.com/AdaWorldAPI/ladybug-rs
git clone https://github.com/AdaWorldAPI/rustynum
git clone https://github.com/AdaWorldAPI/crewai-rust
git clone https://github.com/AdaWorldAPI/n8n-rs

# REFERENCE (read during harvest, never modify):
git clone https://github.com/AdaWorldAPI/lance-graph
git clone https://github.com/AdaWorldAPI/holograph

cd ladybug-rs
cargo check --no-default-features --features "simd"
```

## First command in every session:

```bash
cat CLAUDE.md
cat .claude/prompts/26_entry_point.md
```

## What you write to:

```
ladybug-rs     ONLY THIS
```

## What you read from:

```
rustynum       Path dep. DO NOT MODIFY.
crewai-rust    Path dep. DO NOT MODIFY.
n8n-rs         Path dep. DO NOT MODIFY.
lance-graph    Harvest source. DO NOT MODIFY.
holograph      Harvest source. DO NOT MODIFY.
```

## GitHub PAT:

See userPreferences in Claude context. Not stored in repo.

```
```

## If cargo check fails on path deps:

```bash
# Minimal check (skips sibling deps):
cargo check --no-default-features --features "simd"

# This compiles most of src/ except orchestration/ and flight/
```
