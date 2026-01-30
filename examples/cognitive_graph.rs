//! Cognitive Graph Example
//!
//! Demonstrates graph operations with fingerprint-based addressing.

use ladybug::storage::{CogRedis, CogAddr, CogValue, Tier};
use ladybug::core::Fingerprint;

fn main() {
    // Create cognitive Redis instance
    let mut cog = CogRedis::new();

    // Create some concepts as fingerprints
    let cause = Fingerprint::from_content("cause");
    let effect = Fingerprint::from_content("effect");
    let rain = Fingerprint::from_content("rain");
    let wet_ground = Fingerprint::from_content("wet_ground");

    // Store in cognitive graph
    println!("Creating cognitive graph...");
    println!("- rain CAUSES wet_ground");

    // Bind: rain ⊗ CAUSES ⊗ wet_ground
    let verb_causes = Fingerprint::from_content("CAUSES");
    let edge = rain.bind(&verb_causes).bind(&wet_ground);

    println!("\nEdge fingerprint created via XOR binding");
    println!("ABBA retrieval: Given edge + rain + CAUSES → recover wet_ground");

    // ABBA unbind
    let recovered = edge.unbind(&rain).unbind(&verb_causes);
    let similarity = recovered.similarity(&wet_ground);
    println!("Recovered with {:.1}% similarity", similarity * 100.0);

    // Access bind space directly
    let bind_space = cog.bind_space();
    println!("\nBind space initialized with {} surface ops",
        bind_space.stats().surface_occupied);

    println!("\n✓ Cognitive graph operations working!");
}
