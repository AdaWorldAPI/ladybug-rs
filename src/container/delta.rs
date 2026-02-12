//! XOR delta compression + ECC parity + recovery.
//!
//! Adjacent containers that are semantically similar can be delta-encoded.
//! The parity container enables single-container recovery (RAID-5 style).

use super::Container;

/// Compute XOR parity across a set of containers (RAID-5 style).
/// parity = c[0] ⊕ c[1] ⊕ c[2] ⊕ ... ⊕ c[n]
pub fn xor_parity(containers: &[&Container]) -> Container {
    let mut parity = Container::zero();
    for c in containers {
        parity = parity.xor(c);
    }
    parity
}

/// Recover a lost container given the survivors + parity.
///
/// Because parity = c[0] ⊕ c[1] ⊕ ... ⊕ c[n],
/// the missing container = parity ⊕ (all survivors).
pub fn recover(survivors: &[&Container], parity: &Container) -> Container {
    let mut result = parity.clone();
    for s in survivors {
        result = result.xor(s);
    }
    result
}

/// Delta-encode `target` relative to `base`.
/// Returns (delta, information_content) where information_content = popcount(delta).
pub fn delta_encode(base: &Container, target: &Container) -> (Container, u32) {
    let delta = base.xor(target);
    let info = delta.popcount();
    (delta, info)
}

/// Recover target from base + delta. XOR is self-inverse.
pub fn delta_decode(base: &Container, delta: &Container) -> Container {
    base.xor(delta)
}

/// Chain-encode: encode a sequence of containers as deltas from previous.
/// Returns (first_container, deltas).
/// Total storage ≈ 1 full container + N-1 sparse deltas.
pub fn chain_encode(containers: &[Container]) -> (Container, Vec<(Container, u32)>) {
    if containers.is_empty() {
        return (Container::zero(), Vec::new());
    }

    let first = containers[0].clone();
    let deltas: Vec<(Container, u32)> = containers.windows(2)
        .map(|pair| delta_encode(&pair[0], &pair[1]))
        .collect();

    (first, deltas)
}

/// Decode a chain-encoded sequence back to full containers.
pub fn chain_decode(first: &Container, deltas: &[(Container, u32)]) -> Vec<Container> {
    let mut result = Vec::with_capacity(deltas.len() + 1);
    result.push(first.clone());

    for (delta, _info) in deltas {
        let prev = result.last().unwrap();
        result.push(delta_decode(prev, delta));
    }

    result
}
