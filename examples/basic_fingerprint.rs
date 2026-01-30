//! Basic Fingerprint Example
//!
//! Demonstrates 10K-bit VSA fingerprints and their operations.

use ladybug::core::Fingerprint;

fn main() {
    // Create fingerprints from content
    let apple = Fingerprint::from_content("apple");
    let red = Fingerprint::from_content("red");
    let color = Fingerprint::from_content("color");

    // XOR bind: creates compound representation
    // red_apple = red ⊗ apple
    let red_apple = red.bind(&apple);

    // Unbind to recover (ABBA retrieval)
    // recovered = red_apple ⊗ red ≈ apple
    let recovered = red_apple.unbind(&red);
    assert_eq!(recovered, apple, "XOR is exact inverse");

    // Similarity check
    let similarity = apple.similarity(&recovered);
    println!("Apple recovered with {:.1}% similarity", similarity * 100.0);

    // Bundle multiple concepts (majority voting)
    let fruits = [
        Fingerprint::from_content("apple"),
        Fingerprint::from_content("banana"),
        Fingerprint::from_content("orange"),
    ];

    // Check orthogonality (random concepts should be ~50% similar)
    let sim_ab = fruits[0].similarity(&fruits[1]);
    let sim_ac = fruits[0].similarity(&fruits[2]);
    println!("apple-banana similarity: {:.1}%", sim_ab * 100.0);
    println!("apple-orange similarity: {:.1}%", sim_ac * 100.0);

    // Hamming distance
    let distance = apple.hamming(&red);
    println!("Hamming distance apple-red: {} bits", distance);

    // Density (fraction of 1s)
    println!("Apple density: {:.1}%", apple.density() * 100.0);

    println!("\n✓ All fingerprint operations working!");
}
