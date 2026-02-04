# =============================================================================
# LadybugDB — Docker Multi-Stage Build (v3)
# =============================================================================
# Triple-binary: AVX-512, AVX-2, and generic x86-64
# Runtime auto-selects best binary via /proc/cpuinfo
#
# BUILD:
#   docker build -t ladybugdb .
#
# RUN:
#   docker run -p 8080:8080 ladybugdb
#   docker run -p 8080:8080 -e LADYBUG_DATA_DIR=/data -v ./data:/data ladybugdb
#
# RAILWAY:
#   Auto-detects via RAILWAY_* env vars → binds 0.0.0.0:$PORT
#
# CLAUDE CODE:
#   Auto-detects via CLAUDE_* env vars → binds 127.0.0.1:5432
# =============================================================================

# =============================================================================
# STAGE 1: Builder — compile three binaries
# Rust 1.93 stable (edition 2024 support, full rmp-serde/time compat)
# =============================================================================
FROM rust:1.93-slim-bookworm AS builder

RUN apt-get update && apt-get install -y \
    pkg-config libssl-dev cmake protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# --- Dependency caching ---
COPY Cargo.toml Cargo.lock* ./
RUN mkdir -p src/bin && \
    echo "fn main() {}" > src/bin/server.rs && \
    echo "pub fn dummy() {}" > src/lib.rs && \
    cargo fetch 2>/dev/null || true

# --- Full source ---
COPY . .

# Features to enable (flight enables Arrow Flight gRPC)
# NOTE: lancedb feature has API compatibility issues - enable after fixing lance module
ARG FEATURES="simd,parallel,flight"

# --- AVX-512 binary (Railway, cloud servers) ---
RUN RUSTFLAGS="-C target-cpu=x86-64-v4 -C link-arg=-s" \
    cargo build --release --bin ladybug-server --features "$FEATURES" && \
    cp target/release/ladybug-server /build/ladybug-avx512 && \
    cargo clean -p ladybug

# --- AVX-2 binary (most modern x86-64) ---
RUN RUSTFLAGS="-C target-cpu=x86-64-v3 -C link-arg=-s" \
    cargo build --release --bin ladybug-server --features "$FEATURES" && \
    cp target/release/ladybug-server /build/ladybug-avx2 && \
    cargo clean -p ladybug

# --- Generic binary (any x86-64) ---
RUN RUSTFLAGS="-C link-arg=-s" \
    cargo build --release --bin ladybug-server --features "$FEATURES" && \
    cp target/release/ladybug-server /build/ladybug-generic

# --- Flight gRPC binary (single build, runtime selects SIMD) ---
RUN RUSTFLAGS="-C target-cpu=x86-64-v3 -C link-arg=-s" \
    cargo build --release --bin ladybug-flight --features "$FEATURES" && \
    cp target/release/ladybug-flight /build/ladybug-flight

# =============================================================================
# STAGE 2: Runtime (minimal ~50MB)
# =============================================================================
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl procps \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -s /bin/bash ladybug \
    && mkdir -p /data && chown ladybug:ladybug /data

# Copy all server binaries (HTTP)
COPY --from=builder /build/ladybug-avx512 /usr/local/bin/
COPY --from=builder /build/ladybug-avx2   /usr/local/bin/
COPY --from=builder /build/ladybug-generic /usr/local/bin/

# Copy Flight gRPC binary
COPY --from=builder /build/ladybug-flight /usr/local/bin/

# Copy docs
COPY --from=builder /build/README.md /opt/ladybug/
COPY --from=builder /build/docs/ /opt/ladybug/docs/

# Auto-select entrypoint
COPY <<'ENTRY' /usr/local/bin/ladybug-start
#!/bin/sh
set -e
if grep -q "avx512f" /proc/cpuinfo 2>/dev/null; then
  BIN=ladybug-avx512; LVL=AVX-512
elif grep -q "avx2" /proc/cpuinfo 2>/dev/null; then
  BIN=ladybug-avx2; LVL=AVX-2
else
  BIN=ladybug-generic; LVL=Generic
fi
echo "[ladybugdb] SIMD: ${LVL} → ${BIN}"
exec "/usr/local/bin/${BIN}" "$@"
ENTRY
RUN chmod +x /usr/local/bin/ladybug-start

USER ladybug
WORKDIR /home/ladybug

ENV LADYBUG_HOST=0.0.0.0
ENV LADYBUG_PORT=8080
ENV LADYBUG_FLIGHT_PORT=50051
ENV LADYBUG_DATA_DIR=/data

# HTTP REST API
EXPOSE 8080
# Arrow Flight gRPC
EXPOSE 50051

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${LADYBUG_PORT}/health || exit 1

ENTRYPOINT ["ladybug-start"]
