//! Core buffer pool for zero-allocation hot paths

/// Pre-allocated buffer pool for SIMD operations
pub struct BufferPool {
    // Distances buffer
    distances: Vec<u32>,
    // Indices buffer
    indices: Vec<usize>,
    // Capacity
    capacity: usize,
}

impl BufferPool {
    pub fn new(capacity: usize) -> Self {
        Self {
            distances: vec![0; capacity],
            indices: vec![0; capacity],
            capacity,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn distances_mut(&mut self) -> &mut [u32] {
        &mut self.distances
    }

    pub fn indices_mut(&mut self) -> &mut [usize] {
        &mut self.indices
    }
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new(10_000)
    }
}
