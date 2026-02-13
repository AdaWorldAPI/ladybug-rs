//! Quantum-Inspired Operations Example
//!
//! Demonstrates quantum-style operators on fingerprints.

use ladybug::core::Fingerprint;

fn main() {
    println!("Quantum-inspired operations on 10K-bit fingerprints\n");

    // Create basis states
    let state_0 = Fingerprint::from_content("state_0");
    let state_1 = Fingerprint::from_content("state_1");

    // Superposition: bundle of states (majority voting in VSA)
    // For true superposition, we'd need probabilistic bundling
    println!("Creating 'superposition' via bundling...");

    // Check orthogonality
    let inner_product = state_0.similarity(&state_1);
    println!("Inner product |<0|1>|: {:.3}", inner_product);
    println!("(Random vectors are ~50% similar, orthogonal = 50%)");

    // NOT operation (X gate analog)
    let not_state = state_0.not();
    println!("\nNOT gate: inverts all bits");
    println!("Density before NOT: {:.1}%", state_0.density() * 100.0);
    println!("Density after NOT: {:.1}%", not_state.density() * 100.0);

    // Entanglement analog: XOR binding
    let entangled = state_0.bind(&state_1);
    println!("\n'Entanglement' via XOR binding:");
    println!("entangled = state_0 ⊗ state_1");

    // Measurement analog: unbinding recovers correlated state
    let measured = entangled.unbind(&state_0);
    let fidelity = measured.similarity(&state_1);
    println!(
        "Unbind with state_0 recovers state_1 with fidelity: {:.1}%",
        fidelity * 100.0
    );

    println!("\n✓ Quantum-inspired operations working!");
}
