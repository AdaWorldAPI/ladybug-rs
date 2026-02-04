# Backup, Restore, Schema Safety & Storage Strategy

> **Document Version**: 1.0.0
> **Last Updated**: 2026-02-04
> **Applies To**: Railway deployments, S3 archival, Redis acceleration

---

## Executive Summary

This document covers:
1. **XOR Diff Versioning to Redis** - Efficient incremental backups
2. **Storage Backend Options** - S3 vs Redis/PostgreSQL vs GitHub
3. **Schema Safety** - Graceful migrations without data loss
4. **Backup/Restore Procedures** - Production-ready strategies

**Recommended Architecture**:
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RECOMMENDED STORAGE TOPOLOGY                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │
│  │  HOT TIER   │     │  WARM TIER  │     │  COLD TIER  │                   │
│  │  (Railway)  │────▶│   (Redis)   │────▶│    (S3)     │                   │
│  │ BindSpace   │     │ XOR Deltas  │     │ Full Snaps  │                   │
│  │ In-Memory   │     │ Incremental │     │ Parquet     │                   │
│  └─────────────┘     └─────────────┘     └─────────────┘                   │
│        ▲                    │                   │                           │
│        │                    │                   │                           │
│        └────── RESTORE ─────┴────── RESTORE ────┘                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Table of Contents

1. [XOR Diff Versioning to Redis](#1-xor-diff-versioning-to-redis)
2. [Storage Backend Comparison](#2-storage-backend-comparison)
3. [S3 Integration via LanceDB](#3-s3-integration-via-lancedb)
4. [Schema Safety & Migration](#4-schema-safety--migration)
5. [Backup Procedures](#5-backup-procedures)
6. [Restore Procedures](#6-restore-procedures)
7. [Disaster Recovery](#7-disaster-recovery)

---

## 1. XOR Diff Versioning to Redis

### Concept

XOR diff versioning exploits the property that `A ⊕ B ⊕ B = A`:
- Store full snapshot S₀
- Store deltas D₁ = S₁ ⊕ S₀, D₂ = S₂ ⊕ S₁, etc.
- Reconstruct any version by chaining XORs
- **Redis benefit**: Deltas are sparse (typically <5% of full snapshot)

### Implementation

```rust
/// XOR Delta stored in Redis
pub struct RedisDelta {
    /// Version this delta transitions TO
    pub to_version: u64,
    /// Version this delta transitions FROM
    pub from_version: u64,
    /// Sparse delta: only changed addresses stored
    /// Format: addr (u16) + xor_mask (variable length)
    pub sparse_data: Vec<u8>,
    /// Timestamp
    pub timestamp: u64,
}

impl RedisDelta {
    /// Create delta from two BindSpace snapshots
    pub fn compute(old: &BindSpace, new: &BindSpace) -> Self {
        let mut sparse_data = Vec::new();

        for addr in 0..65536u16 {
            let old_fp = old.read(Addr(addr)).map(|n| n.fingerprint);
            let new_fp = new.read(Addr(addr)).map(|n| n.fingerprint);

            match (old_fp, new_fp) {
                (Some(old), Some(new)) if old != new => {
                    // Compute XOR delta
                    let mut xor = [0u64; FINGERPRINT_WORDS];
                    for i in 0..FINGERPRINT_WORDS {
                        xor[i] = old[i] ^ new[i];
                    }

                    // Encode sparse: addr + non-zero words only
                    sparse_data.extend_from_slice(&addr.to_le_bytes());
                    encode_sparse_fingerprint(&xor, &mut sparse_data);
                }
                (None, Some(new)) => {
                    // Creation: full fingerprint as "delta from zero"
                    sparse_data.extend_from_slice(&addr.to_le_bytes());
                    sparse_data.push(0xFF); // Marker: full fingerprint
                    for word in new.iter() {
                        sparse_data.extend_from_slice(&word.to_le_bytes());
                    }
                }
                (Some(_), None) => {
                    // Deletion: marker
                    sparse_data.extend_from_slice(&addr.to_le_bytes());
                    sparse_data.push(0x00); // Marker: deleted
                }
                _ => {} // No change
            }
        }

        Self {
            to_version: new.version(),
            from_version: old.version(),
            sparse_data,
            timestamp: now_micros(),
        }
    }
}

/// Encode fingerprint sparsely (only non-zero words)
fn encode_sparse_fingerprint(fp: &[u64; FINGERPRINT_WORDS], out: &mut Vec<u8>) {
    // Bitmap: which words are non-zero (156 bits = 20 bytes)
    let mut bitmap = [0u8; 20];
    for (i, &word) in fp.iter().enumerate() {
        if word != 0 {
            bitmap[i / 8] |= 1 << (i % 8);
        }
    }

    out.extend_from_slice(&bitmap);

    // Only non-zero words
    for (i, &word) in fp.iter().enumerate() {
        if word != 0 {
            out.extend_from_slice(&word.to_le_bytes());
        }
    }
}
```

### Redis Schema

```
# Key naming convention
ladybug:snapshot:{version}        -> Full snapshot (Parquet bytes)
ladybug:delta:{from}:{to}         -> XOR delta (sparse format)
ladybug:latest                    -> Current version number
ladybug:schema:version            -> Schema version for migrations
ladybug:checkpoint:{name}         -> Named checkpoint -> version mapping

# Example Redis commands
SET ladybug:latest 42
SET ladybug:delta:41:42 <sparse_delta_bytes>
SET ladybug:snapshot:40 <parquet_bytes>
SET ladybug:checkpoint:pre_migration 41
```

### Backup to Redis (Incremental)

```rust
/// Backup current state to Redis incrementally
pub async fn backup_to_redis(
    bind_space: &BindSpace,
    redis: &mut redis::aio::Connection,
    force_full: bool,
) -> Result<u64, BackupError> {
    let current_version = bind_space.version();

    // Get last backed up version
    let last_version: Option<u64> = redis::cmd("GET")
        .arg("ladybug:latest")
        .query_async(redis).await?;

    match last_version {
        None | Some(_) if force_full => {
            // First backup or forced full: write full snapshot
            let snapshot = bind_space.to_parquet()?;

            redis::pipe()
                .set(format!("ladybug:snapshot:{}", current_version), snapshot)
                .set("ladybug:latest", current_version)
                .query_async(redis).await?;

            Ok(current_version)
        }
        Some(last) if last < current_version => {
            // Incremental: compute and store delta
            let old_snapshot = restore_version(redis, last).await?;
            let delta = RedisDelta::compute(&old_snapshot, bind_space);

            redis::pipe()
                .set(format!("ladybug:delta:{}:{}", last, current_version),
                     delta.sparse_data)
                .set("ladybug:latest", current_version)
                .query_async(redis).await?;

            Ok(current_version)
        }
        Some(last) => {
            // Already up to date
            Ok(last)
        }
    }
}
```

### Restore from Redis

```rust
/// Restore to specific version from Redis
pub async fn restore_from_redis(
    redis: &mut redis::aio::Connection,
    target_version: u64,
) -> Result<BindSpace, RestoreError> {
    // Find nearest full snapshot <= target
    let mut version = target_version;
    let mut deltas_needed = Vec::new();

    loop {
        // Check if full snapshot exists
        let snapshot: Option<Vec<u8>> = redis::cmd("GET")
            .arg(format!("ladybug:snapshot:{}", version))
            .query_async(redis).await?;

        if snapshot.is_some() {
            break;
        }

        // Find delta leading to this version
        let delta_key = find_delta_to_version(redis, version).await?;
        deltas_needed.push(delta_key);
        version = extract_from_version(&delta_key);
    }

    // Load base snapshot
    let snapshot_bytes: Vec<u8> = redis::cmd("GET")
        .arg(format!("ladybug:snapshot:{}", version))
        .query_async(redis).await?
        .ok_or(RestoreError::SnapshotNotFound)?;

    let mut bind_space = BindSpace::from_parquet(&snapshot_bytes)?;

    // Apply deltas in order
    deltas_needed.reverse();
    for delta_key in deltas_needed {
        let delta_bytes: Vec<u8> = redis::cmd("GET")
            .arg(&delta_key)
            .query_async(redis).await?
            .ok_or(RestoreError::DeltaNotFound)?;

        apply_delta(&mut bind_space, &delta_bytes)?;
    }

    Ok(bind_space)
}
```

### Compaction Strategy

```rust
/// Compact delta chain: merge consecutive deltas, create new full snapshot
pub async fn compact_redis_backups(
    redis: &mut redis::aio::Connection,
    max_chain_length: usize,
) -> Result<CompactionStats, CompactionError> {
    let mut stats = CompactionStats::default();

    // Find all delta chains
    let chains = find_delta_chains(redis).await?;

    for chain in chains {
        if chain.len() > max_chain_length {
            // Reconstruct state at chain end
            let end_version = chain.last().unwrap().to_version;
            let state = restore_from_redis(redis, end_version).await?;

            // Create new full snapshot
            let snapshot = state.to_parquet()?;
            redis::cmd("SET")
                .arg(format!("ladybug:snapshot:{}", end_version))
                .arg(&snapshot)
                .query_async(redis).await?;

            // Delete old deltas (keep last N for point-in-time recovery)
            let keep_count = 3;
            for delta in chain.iter().take(chain.len().saturating_sub(keep_count)) {
                redis::cmd("DEL")
                    .arg(format!("ladybug:delta:{}:{}", delta.from_version, delta.to_version))
                    .query_async(redis).await?;
                stats.deltas_removed += 1;
            }

            stats.snapshots_created += 1;
        }
    }

    Ok(stats)
}
```

---

## 2. Storage Backend Comparison

### Railway-Native Options

| Feature | Redis (Railway) | PostgreSQL (Railway) | S3 (External) |
|---------|-----------------|---------------------|---------------|
| **Latency** | <1ms | 2-5ms | 50-200ms |
| **Durability** | AOF/RDB | WAL + Checkpoints | 11 9s |
| **Cost** | $$/GB | $/GB | ¢¢/GB |
| **Max Size** | ~8GB practical | TB+ | Unlimited |
| **Backup** | BGSAVE | pg_dump | Versioning |
| **Point-in-Time** | AOF replay | PITR | Object versioning |
| **Zero-Copy** | No (network) | No (network) | mmap via Lance |
| **Railway Native** | Yes | Yes | No (external) |

### Recommendation Matrix

| Use Case | Primary | Secondary | Archive |
|----------|---------|-----------|---------|
| **Development** | Redis | - | GitHub (git push) |
| **Production <1GB** | Redis | PostgreSQL | S3 |
| **Production >1GB** | PostgreSQL | Redis (hot cache) | S3 |
| **ML/Vector Heavy** | Lance/S3 | Redis (index) | Glacier |

### Hybrid Architecture (Recommended)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        HYBRID STORAGE ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CLIENT REQUEST                                                             │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     LADYBUG (Railway Container)                      │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │                  BindSpace (In-Memory, 8+8)                    │  │   │
│  │  │  • 65,536 addresses                                           │  │   │
│  │  │  • 3-5 cycle lookup                                           │  │   │
│  │  │  • ~80MB for full population                                  │  │   │
│  │  └─────────────────────┬─────────────────────────────────────────┘  │   │
│  │                        │                                            │   │
│  │        ┌───────────────┼───────────────┐                           │   │
│  │        ▼               ▼               ▼                           │   │
│  │   WRITE PATH      READ PATH      BACKGROUND                        │   │
│  │   ──────────      ─────────      ──────────                        │   │
│  │   1. WAL append   1. BindSpace   • Compact deltas                  │   │
│  │   2. BindSpace    2. Redis LRU   • Sync to S3                      │   │
│  │   3. Async Redis  3. S3 cold     • Verify parity                   │   │
│  │                                                                     │   │
│  └─────────────────────┬───────────────────────────────────────────────┘   │
│                        │                                                    │
│          ┌─────────────┼─────────────┐                                     │
│          ▼             ▼             ▼                                     │
│    ┌───────────┐ ┌───────────┐ ┌───────────┐                              │
│    │   Redis   │ │ PostgreSQL│ │    S3     │                              │
│    │ (Railway) │ │ (Railway) │ │ (AWS/R2)  │                              │
│    ├───────────┤ ├───────────┤ ├───────────┤                              │
│    │ XOR Deltas│ │ Schema    │ │ Parquet   │                              │
│    │ LRU Cache │ │ Metadata  │ │ Snapshots │                              │
│    │ Checkpts  │ │ Versions  │ │ Archive   │                              │
│    └───────────┘ └───────────┘ └───────────┘                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. S3 Integration via LanceDB

### Lance ObjectStore Configuration

The vendored Lance (2.1.0-beta) supports S3 natively via `object_store` crate:

```rust
// From vendor/lance/rust/lance-io/src/object_store/providers/aws.rs

use object_store::aws::{AmazonS3Builder, AwsCredentialProvider};

/// Create S3-backed Lance storage
pub async fn create_s3_store(
    bucket: &str,
    prefix: &str,
    region: &str,
) -> Result<Arc<dyn ObjectStore>> {
    let storage_options = StorageOptions::default()
        .with_env_s3();  // Reads AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

    let builder = AmazonS3Builder::new()
        .with_bucket_name(bucket)
        .with_region(region)
        .with_retry(RetryConfig {
            max_retries: 5,
            retry_timeout: Duration::from_secs(30),
            ..Default::default()
        });

    Ok(Arc::new(builder.build()?))
}
```

### Ladybug S3 Backup Integration

```rust
/// Backup BindSpace to S3 via Lance format
pub async fn backup_to_s3(
    bind_space: &BindSpace,
    s3_uri: &str,  // e.g., "s3://ladybug-backups/prod/"
) -> Result<BackupManifest, BackupError> {
    // Convert BindSpace to Arrow RecordBatch
    let batch = bind_space.to_record_batch()?;

    // Create Lance Dataset on S3
    let dataset = Dataset::write(
        batch,
        s3_uri,
        WriteParams::default()
            .with_mode(WriteMode::Create)  // Or Append for incremental
            .with_commit_lock_timeout(Duration::from_secs(30)),
    ).await?;

    let manifest = BackupManifest {
        version: bind_space.version(),
        timestamp: now_micros(),
        s3_uri: s3_uri.to_string(),
        lance_version: dataset.version(),
        row_count: dataset.count_rows().await?,
        size_bytes: dataset.size_bytes().await?,
    };

    Ok(manifest)
}

/// Restore BindSpace from S3
pub async fn restore_from_s3(
    s3_uri: &str,
    version: Option<u64>,  // None = latest
) -> Result<BindSpace, RestoreError> {
    // Open Lance Dataset
    let dataset = if let Some(v) = version {
        Dataset::open_at(s3_uri, v).await?
    } else {
        Dataset::open(s3_uri).await?
    };

    // Stream RecordBatches (zero-copy where possible)
    let mut bind_space = BindSpace::new();
    let mut stream = dataset.scan().await?;

    while let Some(batch) = stream.try_next().await? {
        bind_space.load_from_record_batch(&batch)?;
    }

    Ok(bind_space)
}
```

### S3 Cost Optimization

```rust
/// Tiered S3 storage with lifecycle policies
pub struct S3TierConfig {
    /// Hot: S3 Standard (recent snapshots)
    pub hot_bucket: String,
    /// Warm: S3 Infrequent Access (older snapshots)
    pub warm_bucket: String,
    /// Cold: S3 Glacier (archive)
    pub cold_bucket: String,
    /// Days before hot -> warm
    pub hot_to_warm_days: u32,
    /// Days before warm -> cold
    pub warm_to_cold_days: u32,
}

impl Default for S3TierConfig {
    fn default() -> Self {
        Self {
            hot_bucket: "ladybug-hot".into(),
            warm_bucket: "ladybug-warm".into(),
            cold_bucket: "ladybug-archive".into(),
            hot_to_warm_days: 7,
            warm_to_cold_days: 30,
        }
    }
}
```

---

## 4. Schema Safety & Migration

### Schema Versioning

```rust
/// Schema version embedded in every backup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaVersion {
    /// Major version: breaking changes
    pub major: u32,
    /// Minor version: backward-compatible additions
    pub minor: u32,
    /// Patch version: bug fixes
    pub patch: u32,
    /// Fingerprint dimensions (currently 156 words = 10,000 bits)
    pub fingerprint_words: usize,
    /// Address space size (currently 65,536)
    pub address_space: usize,
    /// Feature flags enabled at backup time
    pub features: Vec<String>,
}

impl SchemaVersion {
    pub const CURRENT: SchemaVersion = SchemaVersion {
        major: 1,
        minor: 0,
        patch: 0,
        fingerprint_words: 156,
        address_space: 65536,
        features: vec![],
    };

    /// Check if this schema can be loaded by current code
    pub fn is_compatible(&self) -> Compatibility {
        if self.major != Self::CURRENT.major {
            Compatibility::Incompatible {
                reason: format!("Major version mismatch: {} vs {}",
                    self.major, Self::CURRENT.major),
            }
        } else if self.fingerprint_words != Self::CURRENT.fingerprint_words {
            Compatibility::NeedsMigration {
                migration: Migration::FingerprintResize {
                    from: self.fingerprint_words,
                    to: Self::CURRENT.fingerprint_words,
                },
            }
        } else {
            Compatibility::Compatible
        }
    }
}

pub enum Compatibility {
    Compatible,
    NeedsMigration { migration: Migration },
    Incompatible { reason: String },
}

pub enum Migration {
    FingerprintResize { from: usize, to: usize },
    AddressSpaceExpand { from: usize, to: usize },
    FeatureUpgrade { features: Vec<String> },
}
```

### Graceful Schema Upgrade

```rust
/// Migrate BindSpace to new schema version
pub fn migrate_schema(
    old_data: &[u8],
    old_schema: &SchemaVersion,
    new_schema: &SchemaVersion,
) -> Result<Vec<u8>, MigrationError> {
    match old_schema.is_compatible() {
        Compatibility::Compatible => {
            // No migration needed
            Ok(old_data.to_vec())
        }
        Compatibility::NeedsMigration { migration } => {
            match migration {
                Migration::FingerprintResize { from, to } => {
                    migrate_fingerprint_size(old_data, from, to)
                }
                Migration::AddressSpaceExpand { from, to } => {
                    migrate_address_space(old_data, from, to)
                }
                Migration::FeatureUpgrade { features } => {
                    migrate_features(old_data, &features)
                }
            }
        }
        Compatibility::Incompatible { reason } => {
            Err(MigrationError::Incompatible(reason))
        }
    }
}

/// Resize fingerprints (e.g., 156 -> 200 words)
fn migrate_fingerprint_size(
    old_data: &[u8],
    from_words: usize,
    to_words: usize,
) -> Result<Vec<u8>, MigrationError> {
    let old_bind_space = BindSpace::from_bytes_with_word_count(old_data, from_words)?;
    let mut new_bind_space = BindSpace::new_with_word_count(to_words);

    // Copy and resize each fingerprint
    for addr in 0..65536u16 {
        if let Some(node) = old_bind_space.read(Addr(addr)) {
            let mut new_fp = vec![0u64; to_words];

            // Copy existing words
            let copy_len = from_words.min(to_words);
            new_fp[..copy_len].copy_from_slice(&node.fingerprint[..copy_len]);

            // If expanding, new bits are zero (neutral for XOR)
            // If shrinking, truncate (may lose information!)

            new_bind_space.write_at_dynamic(Addr(addr), &new_fp);
        }
    }

    Ok(new_bind_space.to_bytes())
}
```

### Pre-Migration Validation

```rust
/// Validate migration before executing
pub async fn validate_migration(
    backup_uri: &str,
    target_schema: &SchemaVersion,
) -> Result<MigrationPlan, ValidationError> {
    // Load backup metadata only (not full data)
    let manifest = load_backup_manifest(backup_uri).await?;

    // Check compatibility
    let compat = manifest.schema.is_compatible_with(target_schema);

    // Estimate migration cost
    let estimated_time = estimate_migration_time(&manifest, target_schema);
    let estimated_size = estimate_migrated_size(&manifest, target_schema);

    // Check for potential data loss
    let data_loss_warnings = check_data_loss_potential(&manifest, target_schema);

    Ok(MigrationPlan {
        source_schema: manifest.schema,
        target_schema: target_schema.clone(),
        compatibility: compat,
        estimated_duration: estimated_time,
        estimated_output_size: estimated_size,
        warnings: data_loss_warnings,
        requires_downtime: compat.requires_downtime(),
    })
}
```

### Online Schema Migration

```rust
/// Migrate schema without downtime
pub async fn online_migrate(
    source: &str,
    target: &str,
    new_schema: &SchemaVersion,
    progress_callback: impl Fn(MigrationProgress),
) -> Result<(), MigrationError> {
    // Phase 1: Create new dataset with new schema (parallel to live)
    let new_dataset = create_dataset_with_schema(target, new_schema).await?;

    // Phase 2: Copy existing data with transformation
    let source_dataset = Dataset::open(source).await?;
    let mut total_rows = source_dataset.count_rows().await?;
    let mut migrated = 0;

    let mut stream = source_dataset.scan().await?;
    while let Some(batch) = stream.try_next().await? {
        let transformed = transform_batch(&batch, new_schema)?;
        new_dataset.append(transformed).await?;

        migrated += batch.num_rows();
        progress_callback(MigrationProgress {
            phase: Phase::DataCopy,
            rows_processed: migrated,
            total_rows,
            percent: (migrated * 100 / total_rows) as u8,
        });
    }

    // Phase 3: Catch up on writes that happened during migration
    let catchup_version = source_dataset.version();
    // ... apply delta from original version to catchup_version

    // Phase 4: Atomic switch (rename datasets)
    atomic_swap_datasets(source, target).await?;

    progress_callback(MigrationProgress {
        phase: Phase::Complete,
        rows_processed: migrated,
        total_rows,
        percent: 100,
    });

    Ok(())
}
```

---

## 5. Backup Procedures

### Automated Backup Schedule

```rust
/// Backup configuration
pub struct BackupConfig {
    /// Full snapshot interval (e.g., daily)
    pub full_snapshot_interval: Duration,
    /// Incremental delta interval (e.g., hourly)
    pub delta_interval: Duration,
    /// Maximum delta chain length before forced full snapshot
    pub max_delta_chain: usize,
    /// Retention policy
    pub retention: RetentionPolicy,
    /// Destinations
    pub destinations: Vec<BackupDestination>,
}

pub struct RetentionPolicy {
    /// Keep all backups for this duration
    pub keep_all: Duration,
    /// After keep_all, keep daily for this duration
    pub keep_daily: Duration,
    /// After keep_daily, keep weekly for this duration
    pub keep_weekly: Duration,
    /// After keep_weekly, keep monthly forever
    pub keep_monthly: bool,
}

impl Default for BackupConfig {
    fn default() -> Self {
        Self {
            full_snapshot_interval: Duration::from_secs(86400),  // Daily
            delta_interval: Duration::from_secs(3600),           // Hourly
            max_delta_chain: 24,                                  // Force full after 24 deltas
            retention: RetentionPolicy {
                keep_all: Duration::from_secs(86400 * 7),        // 7 days
                keep_daily: Duration::from_secs(86400 * 30),     // 30 days
                keep_weekly: Duration::from_secs(86400 * 90),    // 90 days
                keep_monthly: true,
            },
            destinations: vec![
                BackupDestination::Redis { url: "redis://localhost:6379".into() },
                BackupDestination::S3 { bucket: "ladybug-backups".into(), region: "us-east-1".into() },
            ],
        }
    }
}
```

### Backup Verification

```rust
/// Verify backup integrity
pub async fn verify_backup(uri: &str) -> Result<VerificationReport, VerifyError> {
    let mut report = VerificationReport::default();

    // 1. Check manifest exists and is valid
    let manifest = load_manifest(uri).await?;
    report.manifest_valid = true;

    // 2. Verify all referenced files exist
    for file in &manifest.files {
        if !file_exists(&file.path).await? {
            report.missing_files.push(file.path.clone());
        }
    }

    // 3. Verify checksums
    for file in &manifest.files {
        let actual_checksum = compute_checksum(&file.path).await?;
        if actual_checksum != file.checksum {
            report.checksum_mismatches.push(ChecksumMismatch {
                path: file.path.clone(),
                expected: file.checksum.clone(),
                actual: actual_checksum,
            });
        }
    }

    // 4. Verify schema compatibility
    report.schema_compatible = manifest.schema.is_compatible().is_compatible();

    // 5. Verify parity (if XOR-protected)
    if manifest.has_parity {
        report.parity_valid = verify_parity(uri).await?;
    }

    // 6. Test restore of random sample
    let sample_addresses: Vec<u16> = (0..100).map(|_| rand::random()).collect();
    for addr in sample_addresses {
        if let Err(e) = restore_single_address(uri, Addr(addr)).await {
            report.restore_failures.push((addr, e.to_string()));
        }
    }

    Ok(report)
}
```

---

## 6. Restore Procedures

### Point-in-Time Recovery

```rust
/// Restore to specific point in time
pub async fn restore_point_in_time(
    destinations: &[BackupDestination],
    target_time: Timestamp,
) -> Result<BindSpace, RestoreError> {
    // Find backup closest to but not after target_time
    let mut best_backup: Option<(BackupDestination, BackupManifest)> = None;

    for dest in destinations {
        let manifests = list_backups(dest).await?;
        for manifest in manifests {
            if manifest.timestamp <= target_time {
                match &best_backup {
                    None => best_backup = Some((dest.clone(), manifest)),
                    Some((_, best)) if manifest.timestamp > best.timestamp => {
                        best_backup = Some((dest.clone(), manifest));
                    }
                    _ => {}
                }
            }
        }
    }

    let (dest, manifest) = best_backup.ok_or(RestoreError::NoBackupFound)?;

    // Restore from best backup
    let bind_space = restore_from_destination(&dest, Some(manifest.version)).await?;

    // Apply deltas up to target_time
    let deltas = find_deltas_in_range(&dest, manifest.timestamp, target_time).await?;
    for delta in deltas {
        apply_delta(&mut bind_space, &delta)?;
    }

    Ok(bind_space)
}
```

### Disaster Recovery Runbook

```markdown
## Disaster Recovery Runbook

### Scenario 1: Railway Container Crash

1. Railway auto-restarts container
2. On startup, Ladybug loads from:
   a. Redis (fast, incremental)
   b. S3 (fallback, full snapshot)
3. Verify data integrity with `verify_backup()`
4. Resume operations

### Scenario 2: Redis Data Loss

1. Stop writes (set read-only mode)
2. Load latest S3 snapshot:
   ```bash
   curl -X POST http://localhost:8080/api/restore?source=s3
   ```
3. Apply deltas from PostgreSQL (if any)
4. Rebuild Redis cache incrementally
5. Resume writes

### Scenario 3: Complete Data Loss

1. Identify most recent valid backup:
   ```bash
   curl http://localhost:8080/api/backups/list
   ```
2. Verify backup integrity:
   ```bash
   curl -X POST http://localhost:8080/api/backups/verify?uri=s3://...
   ```
3. Restore from backup:
   ```bash
   curl -X POST http://localhost:8080/api/restore?source=s3&version=<version>
   ```
4. Verify restored data:
   ```bash
   curl http://localhost:8080/api/health/deep
   ```
5. Re-enable writes and monitor

### Scenario 4: Schema Corruption

1. Export current data (if readable):
   ```bash
   curl http://localhost:8080/api/export?format=json > emergency_export.json
   ```
2. Restore from last known good schema version:
   ```bash
   curl -X POST http://localhost:8080/api/restore?source=s3&schema_version=1.0.0
   ```
3. Migrate if necessary:
   ```bash
   curl -X POST http://localhost:8080/api/migrate?target_schema=1.1.0
   ```
```

---

## 7. Disaster Recovery

### RPO/RTO Targets

| Scenario | RPO (Max Data Loss) | RTO (Recovery Time) |
|----------|---------------------|---------------------|
| Container restart | 0 (in-memory intact) | <10s |
| Redis failure | 1 hour (last delta) | <5 min |
| S3 failure | 24 hours (last full) | <30 min |
| Complete disaster | 24 hours | <2 hours |

### Cross-Region Replication

```rust
/// Configure cross-region backup for disaster recovery
pub struct CrossRegionConfig {
    pub primary_region: String,
    pub secondary_regions: Vec<String>,
    pub replication_lag_max: Duration,
    pub failover_threshold: Duration,
}

impl Default for CrossRegionConfig {
    fn default() -> Self {
        Self {
            primary_region: "us-east-1".into(),
            secondary_regions: vec!["eu-west-1".into(), "ap-southeast-1".into()],
            replication_lag_max: Duration::from_secs(300),  // 5 minutes
            failover_threshold: Duration::from_secs(60),    // 1 minute
        }
    }
}
```

---

## Appendix: Configuration Examples

### Railway Environment Variables

```bash
# Redis (Railway native)
REDIS_URL=redis://default:xxx@containers-us-west-xxx.railway.app:6379

# PostgreSQL (Railway native)
DATABASE_URL=postgresql://postgres:xxx@containers-us-west-xxx.railway.app:5432/railway

# S3 (external)
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=xxx
AWS_REGION=us-east-1
LADYBUG_S3_BUCKET=ladybug-backups

# Backup configuration
LADYBUG_BACKUP_INTERVAL=3600
LADYBUG_FULL_SNAPSHOT_INTERVAL=86400
LADYBUG_MAX_DELTA_CHAIN=24
```

### Docker Compose (Local Development)

```yaml
version: '3.8'
services:
  ladybug:
    build: .
    ports:
      - "8080:8080"
      - "50051:50051"
    environment:
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/ladybug
    depends_on:
      - redis
      - postgres
      - minio

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: ladybug
      POSTGRES_PASSWORD: postgres
    volumes:
      - postgres_data:/var/lib/postgresql/data

  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio_data:/data

volumes:
  redis_data:
  postgres_data:
  minio_data:
```
