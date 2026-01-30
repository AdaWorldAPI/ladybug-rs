//! Hamming distance benchmark
use criterion::{criterion_group, criterion_main, Criterion};

fn hamming_benchmark(_c: &mut Criterion) {
    // Placeholder
}

criterion_group!(benches, hamming_benchmark);
criterion_main!(benches);
