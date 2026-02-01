# =============================================================================
# Ladybug-RS Multi-Stage Dockerfile
# =============================================================================
# Build optimized for Railway (AVX-512) or fallback (AVX2/generic)
#
# Build arguments:
#   --build-arg TARGET_CPU=native    (auto-detect, default)
#   --build-arg TARGET_CPU=skylake   (AVX-512)
#   --build-arg TARGET_CPU=haswell   (AVX2)
#   --build-arg TARGET_CPU=x86-64    (generic fallback)
#
# Usage:
#   docker build -t ladybug-rs .
#   docker build -t ladybug-rs --build-arg TARGET_CPU=skylake .
#
# Railway deployment:
#   docker build -t ladybug-rs --target runtime-avx512 .
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM rust:1.76-bookworm AS builder

# Build arguments
ARG TARGET_CPU=native
ARG FEATURES="simd,parallel,codebook,hologram,quantum,spo"

# Install build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    pkg-config \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up workspace
WORKDIR /app

# Copy Cargo files first for dependency caching
COPY Cargo.toml Cargo.lock* ./

# Create dummy main for dependency caching
RUN mkdir -p src/bin && \
    echo 'fn main() { println!("dummy"); }' > src/bin/server.rs && \
    echo 'pub fn dummy() {}' > src/lib.rs

# Pre-build dependencies (cached layer)
RUN RUSTFLAGS="-C target-cpu=${TARGET_CPU}" \
    cargo build --release --bin ladybug-server --features "${FEATURES}" 2>/dev/null || true

# Remove dummy files
RUN rm -rf src target/release/.fingerprint/ladybug* target/release/deps/ladybug*

# Copy actual source
COPY . .

# Build with optimizations
RUN RUSTFLAGS="-C target-cpu=${TARGET_CPU} -C opt-level=3 -C lto=thin" \
    cargo build --release --bin ladybug-server --features "${FEATURES}"

# Strip binary for smaller size
RUN strip target/release/ladybug-server

# -----------------------------------------------------------------------------
# Stage 2: Runtime (minimal)
# -----------------------------------------------------------------------------
FROM debian:bookworm-slim AS runtime

# Install minimal runtime dependencies (curl for healthcheck)
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -r -u 1000 -s /bin/false ladybug

# Create data directory
RUN mkdir -p /data && chown ladybug:ladybug /data

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/ladybug-server /app/ladybug-server

# Set ownership
RUN chown -R ladybug:ladybug /app

USER ladybug

# Default environment for Railway/Claude backend
ENV HOST=0.0.0.0
ENV PORT=8080
ENV DATA_DIR=/data
ENV RUST_LOG=info

# Expose port
EXPOSE 8080

# Health check (uses /health endpoint)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:8080/health || exit 1

# Run server
CMD ["./ladybug-server"]

# =============================================================================
# Stage 3: AVX-512 optimized build for Railway/Modern servers
# =============================================================================
FROM builder AS builder-avx512

RUN RUSTFLAGS="-C target-cpu=skylake-avx512 -C opt-level=3 -C lto=thin -C target-feature=+avx512f,+avx512vl,+avx512vpopcntdq" \
    cargo build --release --bin ladybug-server --features "simd,parallel,codebook,hologram,quantum,spo" && \
    strip target/release/ladybug-server

FROM debian:bookworm-slim AS runtime-avx512

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create user and directories
RUN useradd -r -u 1000 -s /bin/false ladybug && \
    mkdir -p /data && \
    chown ladybug:ladybug /data

WORKDIR /app
COPY --from=builder-avx512 /app/target/release/ladybug-server /app/ladybug-server
RUN chown -R ladybug:ladybug /app

USER ladybug

# Railway-optimized environment
ENV HOST=0.0.0.0
ENV PORT=8080
ENV DATA_DIR=/data
ENV RUST_LOG=info
ENV RAILWAY_ENVIRONMENT=production

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:8080/health || exit 1

CMD ["./ladybug-server"]

# =============================================================================
# Stage 4: Claude Backend (localhost-bound)
# =============================================================================
FROM runtime AS runtime-claude

# Override for Claude backend (localhost only)
ENV HOST=127.0.0.1
ENV PORT=5000

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sf http://127.0.0.1:5000/health || exit 1
