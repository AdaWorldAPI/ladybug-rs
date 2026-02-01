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
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM rust:1.75-bookworm AS builder

# Build arguments
ARG TARGET_CPU=native
ARG FEATURES="simd,parallel,codebook,hologram,quantum"

# Install build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    pkg-config \
    libssl-dev \
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
RUN RUSTFLAGS="-C target-cpu=${TARGET_CPU} -C opt-level=3 -C lto=fat" \
    cargo build --release --bin ladybug-server --features "${FEATURES}"

# Strip binary for smaller size
RUN strip target/release/ladybug-server

# -----------------------------------------------------------------------------
# Stage 2: Runtime (minimal)
# -----------------------------------------------------------------------------
FROM debian:bookworm-slim AS runtime

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -r -u 1000 -s /bin/false ladybug

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/ladybug-server /app/ladybug-server

# Set ownership
RUN chown -R ladybug:ladybug /app

USER ladybug

# Default environment
ENV HOST=0.0.0.0
ENV PORT=8080
ENV RUST_LOG=info

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run server
CMD ["./ladybug-server"]

# =============================================================================
# Alternative: AVX-512 optimized build for Railway/Modern servers
# =============================================================================
FROM builder AS builder-avx512

RUN RUSTFLAGS="-C target-cpu=skylake-avx512 -C opt-level=3 -C lto=fat -C target-feature=+avx512f,+avx512vl" \
    cargo build --release --bin ladybug-server --features "simd,parallel,codebook,hologram,quantum" && \
    strip target/release/ladybug-server

FROM debian:bookworm-slim AS runtime-avx512

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
RUN useradd -r -u 1000 -s /bin/false ladybug

WORKDIR /app
COPY --from=builder-avx512 /app/target/release/ladybug-server /app/ladybug-server
RUN chown -R ladybug:ladybug /app

USER ladybug
ENV HOST=0.0.0.0
ENV PORT=8080
ENV RUST_LOG=info
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1
CMD ["./ladybug-server"]
