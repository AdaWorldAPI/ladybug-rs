//! Hamming distance benchmark
use criterion::{Criterion, criterion_group, criterion_main};

fn hamming_benchmark(_c: &mut Criterion) {
    // Placeholder
}

criterion_group!(benches, hamming_benchmark);
criterion_main!(benches);
