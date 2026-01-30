//! Crystal Memory: Inference engine over 43K crystal attractors.
//!
//! 170MB budget → 43,000 × 4KB crystals
//! Each crystal encodes an attractor basin in 2^1,250,000 configuration space.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    CRYSTAL MEMORY                            │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                             │
//! │   43K crystals × 4KB = 170MB                                │
//! │                                                             │
//! │   INFERENCE:                                                │
//! │   Query → Route (Hamming) → Expand → Settle → Compress     │
//! │                                                             │
//! │   LEARNING:                                                 │
//! │   (Input, Target) → Sculpt attractor landscape             │
//! │                                                             │
//! │   The knowledge isn't STORED. It's SHAPED.                 │
//! │                                                             │
//! └─────────────────────────────────────────────────────────────┘
//! ```

use crate::core::Fingerprint;
use crate::FINGERPRINT_U64;
use super::field::QuorumField;
use super::crystal4k::Crystal4K;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Default capacity: 170MB / 4KB ≈ 43,000 crystals
pub const DEFAULT_CAPACITY: usize = 43_000;

/// Maximum settle steps during inference
pub const DEFAULT_SETTLE_STEPS: usize = 100;

/// Crystal Memory: 170MB of sculpted attractor landscape.
pub struct CrystalMemory {
    /// The crystals (attractor basins)
    crystals: Vec<Crystal4K>,
    
    /// Routing index: signature of each crystal for fast lookup
    signatures: Vec<Fingerprint>,
    
    /// Reusable workspace (156KB)
    workspace: QuorumField,
    
    /// Settle steps for inference
    settle_steps: usize,
}

impl CrystalMemory {
    /// Create empty memory with given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            crystals: Vec::with_capacity(capacity),
            signatures: Vec::with_capacity(capacity),
            workspace: QuorumField::default_threshold(),
            settle_steps: DEFAULT_SETTLE_STEPS,
        }
    }
    
    /// Create with default capacity (43K crystals, 170MB)
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }
    
    /// Number of crystals
    pub fn len(&self) -> usize {
        self.crystals.len()
    }
    
    /// Is empty?
    pub fn is_empty(&self) -> bool {
        self.crystals.is_empty()
    }
    
    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.crystals.len() * Crystal4K::size_bytes() +
        self.signatures.len() * FINGERPRINT_U64 * 8 +
        QuorumField::size_bytes()
    }
    
    /// Set settle steps for inference
    pub fn set_settle_steps(&mut self, steps: usize) {
        self.settle_steps = steps;
    }
    
    /// Add a crystal (returns index)
    pub fn add(&mut self, crystal: Crystal4K) -> usize {
        let idx = self.crystals.len();
        self.signatures.push(crystal.signature());
        self.crystals.push(crystal);
        idx
    }
    
    /// Add from a trained field
    pub fn add_field(&mut self, field: &QuorumField) -> usize {
        self.add(Crystal4K::from_field(field))
    }
    
    /// Get crystal by index
    pub fn get(&self, idx: usize) -> Option<&Crystal4K> {
        self.crystals.get(idx)
    }
    
    /// Find nearest crystal by signature (routing)
    ///
    /// Returns (index, distance)
    pub fn route(&self, query: &Fingerprint) -> Option<(usize, u32)> {
        if self.signatures.is_empty() {
            return None;
        }
        
        let mut best_idx = 0;
        let mut best_dist = u32::MAX;
        
        for (i, sig) in self.signatures.iter().enumerate() {
            let dist = query.hamming(sig);
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }
        
        Some((best_idx, best_dist))
    }
    
    /// Find k nearest crystals
    pub fn route_k(&self, query: &Fingerprint, k: usize) -> Vec<(usize, u32)> {
        let mut distances: Vec<(usize, u32)> = self.signatures
            .iter()
            .enumerate()
            .map(|(i, sig)| (i, query.hamming(sig)))
            .collect();
        
        // Partial sort for top-k
        let k = k.min(distances.len());
        distances.select_nth_unstable_by_key(k.saturating_sub(1), |&(_, d)| d);
        distances.truncate(k);
        distances.sort_by_key(|&(_, d)| d);
        
        distances
    }
    
    /// Inference: query → settled attractor
    ///
    /// 1. Route to nearest crystal
    /// 2. Expand crystal to workspace
    /// 3. Inject query
    /// 4. Let quorum dynamics settle
    /// 5. Compress back to 4KB
    pub fn infer(&mut self, query: &Crystal4K) -> Option<Crystal4K> {
        // Route by signature
        let (idx, _dist) = self.route(&query.signature())?;
        
        // Expand crystal to workspace
        let crystal = &self.crystals[idx];
        self.workspace = crystal.expand();
        
        // Inject query pattern at center
        let center = 2; // 5/2 = 2
        let query_expanded = query.expand();
        let query_sig = query_expanded.get(center, center, center);
        self.workspace.inject_at(center, center, center, &query_sig);
        
        // Settle into attractor
        self.workspace.settle(self.settle_steps);
        
        // Compress result
        Some(Crystal4K::from_field(&self.workspace))
    }
    
    /// Inference from raw fingerprint
    pub fn infer_fp(&mut self, query: &Fingerprint) -> Option<Crystal4K> {
        // Create crystal from single fingerprint
        let input_crystal = Crystal4K::new(
            query.clone(),
            query.permute(1),
            query.permute(2),
        );
        self.infer(&input_crystal)
    }
    
    /// Learn: sculpt attractor toward target
    ///
    /// Hebbian-style: cells matching target get reinforced.
    pub fn learn(&mut self, input: &Crystal4K, target: &Crystal4K, learning_rate: f32) {
        // Find or create nearest crystal
        let (idx, dist) = match self.route(&input.signature()) {
            Some((idx, dist)) if dist < 3000 => (idx, dist), // Close enough to modify
            _ => {
                // Create new crystal
                let idx = self.add(input.clone());
                (idx, 0)
            }
        };
        
        // Expand current crystal
        let crystal = &self.crystals[idx];
        let mut field = crystal.expand();
        
        // Expand target
        let target_field = target.expand();
        
        // Sculpt: move cells toward target
        // This is a simplified Hebbian update
        let lr = (learning_rate * 64.0) as u32; // Scale for bit operations
        
        for x in 0..5 {
            for y in 0..5 {
                for z in 0..5 {
                    let current = field.get(x, y, z);
                    let target_cell = target_field.get(x, y, z);
                    
                    // Interpolate: some bits from current, some from target
                    let mut new_data = [0u64; FINGERPRINT_U64];
                    for i in 0..FINGERPRINT_U64 {
                        // Random bits select source (crude interpolation)
                        let mask = Fingerprint::random().as_raw()[i];
                        let threshold_mask = if lr > 32 { !0u64 } else { mask };
                        
                        new_data[i] = (current.as_raw()[i] & !threshold_mask) |
                                     (target_cell.as_raw()[i] & threshold_mask);
                    }
                    
                    field.set(x, y, z, &Fingerprint::from_raw(new_data));
                }
            }
        }
        
        // Compress back
        let new_crystal = Crystal4K::from_field(&field);
        self.crystals[idx] = new_crystal.clone();
        self.signatures[idx] = new_crystal.signature();
    }
    
    /// Batch learn from (input, target) pairs
    pub fn batch_learn(&mut self, pairs: &[(Crystal4K, Crystal4K)], learning_rate: f32) {
        for (input, target) in pairs {
            self.learn(input, target, learning_rate);
        }
    }
    
    /// Save to file
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        
        let mut file = std::fs::File::create(path)?;
        
        // Header: version, count
        file.write_all(&[1u8])?; // Version
        file.write_all(&(self.crystals.len() as u64).to_le_bytes())?;
        
        // Crystals
        for crystal in &self.crystals {
            file.write_all(&crystal.to_bytes())?;
        }
        
        Ok(())
    }
    
    /// Load from file
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        use std::io::Read;
        
        let mut file = std::fs::File::open(path)?;
        
        // Header
        let mut version = [0u8; 1];
        file.read_exact(&mut version)?;
        
        let mut count_bytes = [0u8; 8];
        file.read_exact(&mut count_bytes)?;
        let count = u64::from_le_bytes(count_bytes) as usize;
        
        // Crystals
        let mut memory = Self::with_capacity(count);
        let crystal_size = Crystal4K::size_bytes();
        let mut buffer = vec![0u8; crystal_size];
        
        for _ in 0..count {
            file.read_exact(&mut buffer)?;
            if let Some(crystal) = Crystal4K::from_bytes(&buffer) {
                memory.add(crystal);
            }
        }
        
        Ok(memory)
    }
    
    /// Create memory seeded with random crystals
    pub fn random(count: usize) -> Self {
        let mut memory = Self::with_capacity(count);
        
        for _ in 0..count {
            let crystal = Crystal4K::new(
                Fingerprint::random(),
                Fingerprint::random(),
                Fingerprint::random(),
            );
            memory.add(crystal);
        }
        
        memory
    }
}

impl Default for CrystalMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for CrystalMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CrystalMemory {{ crystals: {}, memory: {:.1}MB }}",
            self.crystals.len(),
            self.memory_bytes() as f64 / 1_000_000.0
        )
    }
}

/// Statistics about crystal memory
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub crystal_count: usize,
    pub memory_bytes: usize,
    pub avg_popcount: f32,
    pub signature_entropy: f32,
}

impl CrystalMemory {
    /// Compute statistics
    pub fn stats(&self) -> MemoryStats {
        let crystal_count = self.crystals.len();
        let memory_bytes = self.memory_bytes();
        
        let avg_popcount = if crystal_count > 0 {
            self.crystals.iter().map(|c| c.popcount()).sum::<u32>() as f32 / crystal_count as f32
        } else {
            0.0
        };
        
        // Estimate signature entropy (variance in Hamming distances)
        let signature_entropy = if crystal_count > 1 {
            let mut distances = Vec::new();
            for i in 0..crystal_count.min(100) {
                for j in (i+1)..crystal_count.min(100) {
                    distances.push(self.signatures[i].hamming(&self.signatures[j]) as f32);
                }
            }
            
            if distances.is_empty() {
                0.0
            } else {
                let mean = distances.iter().sum::<f32>() / distances.len() as f32;
                let variance = distances.iter()
                    .map(|&d| (d - mean).powi(2))
                    .sum::<f32>() / distances.len() as f32;
                variance.sqrt() / 5000.0 // Normalize
            }
        } else {
            0.0
        };
        
        MemoryStats {
            crystal_count,
            memory_bytes,
            avg_popcount,
            signature_entropy,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_creation() {
        let memory = CrystalMemory::new();
        assert_eq!(memory.len(), 0);
    }
    
    #[test]
    fn test_add_and_route() {
        let mut memory = CrystalMemory::new();
        
        let c1 = Crystal4K::new(
            Fingerprint::from_content("a"),
            Fingerprint::from_content("b"),
            Fingerprint::from_content("c"),
        );
        let c2 = Crystal4K::new(
            Fingerprint::from_content("x"),
            Fingerprint::from_content("y"),
            Fingerprint::from_content("z"),
        );
        
        memory.add(c1.clone());
        memory.add(c2.clone());
        
        // Query similar to c1 should route there
        let query = c1.signature();
        let (idx, _dist) = memory.route(&query).unwrap();
        assert_eq!(idx, 0);
    }
    
    #[test]
    fn test_inference() {
        let mut memory = CrystalMemory::new();
        
        // Add a crystal
        let crystal = Crystal4K::new(
            Fingerprint::from_content("base_x"),
            Fingerprint::from_content("base_y"),
            Fingerprint::from_content("base_z"),
        );
        memory.add(crystal.clone());
        
        // Inference should return something
        let result = memory.infer(&crystal);
        assert!(result.is_some());
    }
    
    #[test]
    fn test_memory_size() {
        let mut memory = CrystalMemory::new();
        
        // Add 1000 crystals
        for i in 0..1000 {
            let crystal = Crystal4K::new(
                Fingerprint::from_content(&format!("x_{}", i)),
                Fingerprint::from_content(&format!("y_{}", i)),
                Fingerprint::from_content(&format!("z_{}", i)),
            );
            memory.add(crystal);
        }
        
        let bytes = memory.memory_bytes();
        // ~4KB per crystal + signature overhead
        assert!(bytes > 4_000_000); // > 4MB
        assert!(bytes < 10_000_000); // < 10MB for 1K crystals
    }
}
