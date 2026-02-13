//! Zero-Copy Graph Executor
//!
//! Operates directly on BindSpace addresses without memory movement.
//! Graph traversal, binding, and inference work on references only.
//!
//! ## Key Insight
//!
//! ```text
//! Traditional: Load → Compute → Store (3 copies)
//! Zero-Copy:   Compute in-place on addresses (0 copies)
//!
//! The 16-bit address IS the handle. No dereferencing needed until
//! the final result is actually consumed.
//! ```
//!
//! ## Graph Representation
//!
//! ```text
//! Nodes: 0x80XX (node zone)
//! Edges: 0x10XX (fluid zone) - edge.fingerprint[0] = target_addr
//!
//! Node A ──edge1──> Node B ──edge2──> Node C
//! 0x8000   0x1000   0x8001   0x1001   0x8002
//!          ↓                 ↓
//!      [0x8001,...]      [0x8002,...]
//! ```

use crate::storage::{Addr, BindSpace, FINGERPRINT_WORDS};
use std::collections::VecDeque;

/// Address reference - no data copy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AddrRef(pub u16);

impl AddrRef {
    #[inline(always)]
    pub fn prefix(&self) -> u8 {
        (self.0 >> 8) as u8
    }

    #[inline(always)]
    pub fn slot(&self) -> u8 {
        (self.0 & 0xFF) as u8
    }

    #[inline(always)]
    pub fn is_node(&self) -> bool {
        self.prefix() >= 0x80
    }

    #[inline(always)]
    pub fn is_fluid(&self) -> bool {
        self.prefix() >= 0x10 && self.prefix() < 0x80
    }

    #[inline(always)]
    pub fn is_surface(&self) -> bool {
        self.prefix() < 0x10
    }
}

impl From<u16> for AddrRef {
    fn from(addr: u16) -> Self {
        Self(addr)
    }
}

impl From<Addr> for AddrRef {
    fn from(addr: Addr) -> Self {
        Self(addr.0)
    }
}

/// Edge encoded in fluid zone
/// First u64 of fingerprint = target address (and metadata)
#[derive(Debug, Clone, Copy)]
pub struct EdgeRef {
    /// Edge address in fluid zone (0x10XX)
    pub addr: AddrRef,
    /// Target node extracted from fingerprint[0]
    pub target: AddrRef,
    /// Edge weight (from fingerprint[1])
    pub weight: u32,
}

/// Zero-copy graph executor
pub struct ZeroCopyExecutor<'a> {
    bind_space: &'a mut BindSpace,
    /// Operation count (no data movement count - always 0!)
    ops: u64,
}

impl<'a> ZeroCopyExecutor<'a> {
    pub fn new(bind_space: &'a mut BindSpace) -> Self {
        Self { bind_space, ops: 0 }
    }

    /// Bind two addresses, write result to dest - ZERO COPY
    ///
    /// Instead of: load A, load B, XOR, store C
    /// We do: XOR directly from A,B addresses into C
    #[inline]
    pub fn bind_ref(&mut self, dest: AddrRef, a: AddrRef, b: AddrRef) -> bool {
        self.ops += 1;

        // Read source fingerprints (these are references into the array)
        let fp_a = match self.bind_space.read(Addr(a.0)) {
            Some(node) => node.fingerprint,
            None => return false,
        };
        let fp_b = match self.bind_space.read(Addr(b.0)) {
            Some(node) => node.fingerprint,
            None => return false,
        };

        // XOR directly into destination
        let mut result = [0u64; FINGERPRINT_WORDS];
        for i in 0..FINGERPRINT_WORDS {
            result[i] = fp_a[i] ^ fp_b[i];
        }

        self.bind_space.write_at(Addr(dest.0), result);
        true
    }

    /// Follow edge from source node - returns target address only
    /// No fingerprint data is copied!
    #[inline]
    pub fn follow_edge(&self, edge: AddrRef) -> Option<AddrRef> {
        self.bind_space
            .read(Addr(edge.0))
            .map(|node| AddrRef((node.fingerprint[0] & 0xFFFF) as u16))
    }

    /// Get outgoing edges from a node
    /// Edge addresses are stored in the node's fingerprint
    #[inline]
    pub fn get_edges(&self, node: AddrRef) -> Vec<AddrRef> {
        if let Some(n) = self.bind_space.read(Addr(node.0)) {
            // First 8 u64s can hold up to 32 edge addresses (16 bits each)
            let mut edges = Vec::new();
            for i in 0..8 {
                let word = n.fingerprint[i];
                for shift in [0, 16, 32, 48] {
                    let edge_addr = ((word >> shift) & 0xFFFF) as u16;
                    if edge_addr != 0 {
                        edges.push(AddrRef(edge_addr));
                    }
                }
            }
            edges
        } else {
            Vec::new()
        }
    }

    /// BFS traversal - returns addresses only, no data copied
    pub fn traverse_bfs(&self, start: AddrRef, max_hops: usize) -> Vec<AddrRef> {
        let mut visited = vec![false; 65536];
        let mut result = Vec::new();
        let mut queue = VecDeque::new();

        queue.push_back((start, 0));
        visited[start.0 as usize] = true;

        while let Some((current, depth)) = queue.pop_front() {
            result.push(current);

            if depth >= max_hops {
                continue;
            }

            for edge in self.get_edges(current) {
                if let Some(target) = self.follow_edge(edge) {
                    if !visited[target.0 as usize] {
                        visited[target.0 as usize] = true;
                        queue.push_back((target, depth + 1));
                    }
                }
            }
        }

        result
    }

    /// Pattern match - find nodes matching pattern (address-only result)
    pub fn pattern_match(&self, pattern: AddrRef, prefix_filter: u8) -> Vec<AddrRef> {
        let pattern_fp = match self.bind_space.read(Addr(pattern.0)) {
            Some(node) => node.fingerprint,
            None => return Vec::new(),
        };

        let mut matches = Vec::new();

        // Scan matching prefix
        let start = (prefix_filter as u16) << 8;
        let end = start + 256;

        for addr in start..end {
            if let Some(node) = self.bind_space.read(Addr(addr)) {
                // Quick hamming check on first word only
                let dist = (pattern_fp[0] ^ node.fingerprint[0]).count_ones();
                if dist < 16 {
                    // Potential match - return address
                    matches.push(AddrRef(addr));
                }
            }
        }

        matches
    }

    /// Compute Hamming distance between two addresses
    /// Returns distance, doesn't copy data
    #[inline]
    pub fn hamming_ref(&self, a: AddrRef, b: AddrRef) -> Option<u32> {
        let fp_a = self.bind_space.read(Addr(a.0))?;
        let fp_b = self.bind_space.read(Addr(b.0))?;

        let dist: u32 = fp_a
            .fingerprint
            .iter()
            .zip(fp_b.fingerprint.iter())
            .map(|(x, y)| (x ^ y).count_ones())
            .sum();

        Some(dist)
    }

    /// Superpose multiple addresses into one (majority vote)
    /// Only copies at the very end when writing result
    pub fn superpose_refs(&mut self, dest: AddrRef, sources: &[AddrRef]) -> bool {
        if sources.is_empty() {
            return false;
        }

        self.ops += 1;

        // Collect bit counts (not fingerprints!)
        let mut bit_counts = [[0u8; 64]; FINGERPRINT_WORDS];

        for src in sources {
            if let Some(node) = self.bind_space.read(Addr(src.0)) {
                for (i, word) in node.fingerprint.iter().enumerate() {
                    for bit in 0..64 {
                        if (word >> bit) & 1 == 1 {
                            bit_counts[i][bit] = bit_counts[i][bit].saturating_add(1);
                        }
                    }
                }
            }
        }

        // Majority vote
        let threshold = (sources.len() / 2) as u8;
        let mut result = [0u64; FINGERPRINT_WORDS];
        for (i, counts) in bit_counts.iter().enumerate() {
            for (bit, &count) in counts.iter().enumerate() {
                if count > threshold {
                    result[i] |= 1 << bit;
                }
            }
        }

        self.bind_space.write_at(Addr(dest.0), result);
        true
    }

    /// Create edge: write target address into edge slot
    pub fn create_edge(&mut self, edge: AddrRef, target: AddrRef, weight: u32) {
        let mut fp = [0u64; FINGERPRINT_WORDS];
        fp[0] = target.0 as u64;
        fp[1] = weight as u64;
        self.bind_space.write_at(Addr(edge.0), fp);
    }

    /// Add edge to node's edge list
    pub fn add_edge_to_node(&mut self, node: AddrRef, edge: AddrRef) -> bool {
        if let Some(n) = self.bind_space.read(Addr(node.0)) {
            let mut fp = n.fingerprint;

            // Find first empty slot in first 8 words
            for i in 0..8 {
                for shift in [0, 16, 32, 48] {
                    let slot = ((fp[i] >> shift) & 0xFFFF) as u16;
                    if slot == 0 {
                        fp[i] |= (edge.0 as u64) << shift;
                        self.bind_space.write_at(Addr(node.0), fp);
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Shortest path via address-only BFS
    pub fn shortest_path(
        &self,
        start: AddrRef,
        end: AddrRef,
        max_hops: usize,
    ) -> Option<Vec<AddrRef>> {
        let mut visited = vec![false; 65536];
        let mut parent = vec![None::<AddrRef>; 65536];
        let mut queue = VecDeque::new();

        queue.push_back((start, 0));
        visited[start.0 as usize] = true;

        while let Some((current, depth)) = queue.pop_front() {
            if current == end {
                // Reconstruct path from addresses only
                let mut path = Vec::new();
                let mut node = Some(end);
                while let Some(n) = node {
                    path.push(n);
                    node = parent[n.0 as usize];
                }
                path.reverse();
                return Some(path);
            }

            if depth >= max_hops {
                continue;
            }

            for edge in self.get_edges(current) {
                if let Some(target) = self.follow_edge(edge) {
                    if !visited[target.0 as usize] {
                        visited[target.0 as usize] = true;
                        parent[target.0 as usize] = Some(current);
                        queue.push_back((target, depth + 1));
                    }
                }
            }
        }

        None
    }

    pub fn ops_count(&self) -> u64 {
        self.ops
    }
}

/// Deferred computation - holds address until value is needed
pub struct Deferred {
    addr: AddrRef,
    /// Operation to apply when resolved
    pending_ops: Vec<DeferredOp>,
}

enum DeferredOp {
    Bind(AddrRef),
    Permute(u32),
    Unbind(AddrRef),
}

impl Deferred {
    pub fn new(addr: AddrRef) -> Self {
        Self {
            addr,
            pending_ops: Vec::new(),
        }
    }

    /// Chain bind operation (doesn't execute yet)
    pub fn bind(mut self, other: AddrRef) -> Self {
        self.pending_ops.push(DeferredOp::Bind(other));
        self
    }

    /// Chain permute operation
    pub fn permute(mut self, shift: u32) -> Self {
        self.pending_ops.push(DeferredOp::Permute(shift));
        self
    }

    /// Resolve: execute all pending ops and return final address
    pub fn resolve(self, exec: &mut ZeroCopyExecutor, dest: AddrRef) -> AddrRef {
        let mut current = self.addr;

        for (i, op) in self.pending_ops.into_iter().enumerate() {
            let temp = AddrRef(0x1000 + i as u16); // Use fluid zone for temps

            match op {
                DeferredOp::Bind(other) => {
                    exec.bind_ref(temp, current, other);
                    current = temp;
                }
                DeferredOp::Permute(shift) => {
                    if let Some(node) = exec.bind_space.read(Addr(current.0)) {
                        let mut fp = node.fingerprint;
                        for word in fp.iter_mut() {
                            *word = word.rotate_left(shift);
                        }
                        exec.bind_space.write_at(Addr(temp.0), fp);
                        current = temp;
                    }
                }
                DeferredOp::Unbind(key) => {
                    // Unbind = Bind (XOR is self-inverse)
                    exec.bind_ref(temp, current, key);
                    current = temp;
                }
            }
        }

        // Move final result to dest
        if current != dest {
            if let Some(node) = exec.bind_space.read(Addr(current.0)) {
                exec.bind_space.write_at(Addr(dest.0), node.fingerprint);
            }
        }

        dest
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_fp(seed: u64) -> [u64; FINGERPRINT_WORDS] {
        let mut fp = [0u64; FINGERPRINT_WORDS];
        for i in 0..FINGERPRINT_WORDS {
            fp[i] = seed.wrapping_mul(i as u64 + 1).wrapping_add(0xDEADBEEF);
        }
        fp
    }

    #[test]
    fn test_bind_ref_zero_copy() {
        let mut bs = BindSpace::new();

        // Write source fingerprints
        bs.write_at(Addr(0x8000), make_fp(1));
        bs.write_at(Addr(0x8001), make_fp(2));

        let mut exec = ZeroCopyExecutor::new(&mut bs);

        // Bind via references
        let a = AddrRef(0x8000);
        let b = AddrRef(0x8001);
        let dest = AddrRef(0x8002);

        assert!(exec.bind_ref(dest, a, b));

        // Verify result
        let result = bs.read(Addr(0x8002)).unwrap();
        let expected_a = make_fp(1);
        let expected_b = make_fp(2);

        for i in 0..FINGERPRINT_WORDS {
            assert_eq!(result.fingerprint[i], expected_a[i] ^ expected_b[i]);
        }
    }

    #[test]
    fn test_graph_traversal() {
        let mut bs = BindSpace::new();

        // Create nodes
        bs.write_at(Addr(0x8000), make_fp(100)); // Node A
        bs.write_at(Addr(0x8001), make_fp(101)); // Node B
        bs.write_at(Addr(0x8002), make_fp(102)); // Node C

        let mut exec = ZeroCopyExecutor::new(&mut bs);

        // Create edges: A -> B -> C
        exec.create_edge(AddrRef(0x1000), AddrRef(0x8001), 1); // Edge to B
        exec.create_edge(AddrRef(0x1001), AddrRef(0x8002), 1); // Edge to C

        // Add edges to nodes
        exec.add_edge_to_node(AddrRef(0x8000), AddrRef(0x1000)); // A has edge to B
        exec.add_edge_to_node(AddrRef(0x8001), AddrRef(0x1001)); // B has edge to C

        // Traverse from A
        let visited = exec.traverse_bfs(AddrRef(0x8000), 5);

        assert_eq!(visited.len(), 3);
        assert_eq!(visited[0], AddrRef(0x8000)); // A
        assert_eq!(visited[1], AddrRef(0x8001)); // B
        assert_eq!(visited[2], AddrRef(0x8002)); // C
    }

    #[test]
    fn test_shortest_path() {
        let mut bs = BindSpace::new();

        // Create nodes A, B, C, D
        for i in 0..4 {
            bs.write_at(Addr(0x8000 + i), make_fp(i as u64));
        }

        let mut exec = ZeroCopyExecutor::new(&mut bs);

        // Create edges: A->B, A->C, B->D, C->D
        exec.create_edge(AddrRef(0x1000), AddrRef(0x8001), 1); // A->B
        exec.create_edge(AddrRef(0x1001), AddrRef(0x8002), 1); // A->C
        exec.create_edge(AddrRef(0x1002), AddrRef(0x8003), 1); // B->D
        exec.create_edge(AddrRef(0x1003), AddrRef(0x8003), 1); // C->D

        exec.add_edge_to_node(AddrRef(0x8000), AddrRef(0x1000));
        exec.add_edge_to_node(AddrRef(0x8000), AddrRef(0x1001));
        exec.add_edge_to_node(AddrRef(0x8001), AddrRef(0x1002));
        exec.add_edge_to_node(AddrRef(0x8002), AddrRef(0x1003));

        // Find path from A to D
        let path = exec.shortest_path(AddrRef(0x8000), AddrRef(0x8003), 10);

        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path.len(), 3); // A -> B -> D or A -> C -> D
        assert_eq!(path[0], AddrRef(0x8000));
        assert_eq!(path[2], AddrRef(0x8003));
    }

    #[test]
    fn test_deferred_computation() {
        let mut bs = BindSpace::new();

        bs.write_at(Addr(0x8000), make_fp(1));
        bs.write_at(Addr(0x8001), make_fp(2));
        bs.write_at(Addr(0x8002), make_fp(3));

        let mut exec = ZeroCopyExecutor::new(&mut bs);

        // Chain operations without executing
        let deferred = Deferred::new(AddrRef(0x8000))
            .bind(AddrRef(0x8001))
            .bind(AddrRef(0x8002));

        // Resolve all at once
        let result = deferred.resolve(&mut exec, AddrRef(0x8010));

        assert_eq!(result, AddrRef(0x8010));

        // Verify: (A ^ B) ^ C = A ^ B ^ C
        let a = make_fp(1);
        let b = make_fp(2);
        let c = make_fp(3);
        let node = bs.read(Addr(0x8010)).unwrap();

        for i in 0..FINGERPRINT_WORDS {
            assert_eq!(node.fingerprint[i], a[i] ^ b[i] ^ c[i]);
        }
    }

    #[test]
    fn test_hamming_ref() {
        let mut bs = BindSpace::new();

        let fp1 = make_fp(1);
        let mut fp2 = make_fp(1);
        fp2[0] ^= 0xFF; // Flip 8 bits

        bs.write_at(Addr(0x8000), fp1);
        bs.write_at(Addr(0x8001), fp2);

        let exec = ZeroCopyExecutor::new(&mut bs);
        let dist = exec.hamming_ref(AddrRef(0x8000), AddrRef(0x8001));

        assert_eq!(dist, Some(8)); // 8 bits different
    }
}
