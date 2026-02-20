//! SpoBuilder — construct SPO CogRecords for nodes, edges, and meta-awareness.
//!
//! Nodes:  X = entity identity, Y = near-zero, Z = mirror of X
//! Edges:  X = BIND(src, permute(verb,1)), Y = verb, Z = BIND(tgt, permute(verb,2))
//! Meta:   X = BUNDLE(chain X axes), Y = CHAIN_DISCOVERED, Z = BUNDLE(chain Z axes)

use ladybug_contract::container::Container;
use ladybug_contract::nars::TruthValue;
use ladybug_contract::record::CogRecord;

use super::scent::NibbleScent;
use super::sparse::{pack_axes, unpack_axes, AxisDescriptors, SparseContainer, SpoError};

/// SPO geometry ID (matches ContainerGeometry::Spo = 6).
const GEOMETRY_SPO: u8 = 6;

/// Meta word offsets (from ladybug_contract::meta).
const W_DN_ADDR: usize = 0;
const W_TYPE: usize = 1;
const W_NARS_BASE: usize = 4;
const W_SCENT_BASE: usize = 12;
const W_AXIS_DESC: usize = 34;

pub struct SpoBuilder;

impl SpoBuilder {
    /// Build a node record (entity with labels and properties).
    ///
    /// X axis = BUNDLE(label_fps ++ property_bind_fps)
    /// Y axis = near-zero (nodes have no predicate)
    /// Z axis = clone of X (self-referential for content lookup)
    pub fn build_node(
        dn: u64,
        label_fps: &[&Container],
        property_fps: &[(&Container, &Container)], // (key_fp, val_fp) pairs
        nars: TruthValue,
    ) -> Result<CogRecord, SpoError> {
        // Build X axis: bundle of labels + property bindings
        let mut x_components: Vec<&Container> = label_fps.to_vec();
        let prop_bindings: Vec<Container> = property_fps
            .iter()
            .map(|(key, val)| key.xor(val)) // BIND(key, val)
            .collect();
        let prop_refs: Vec<&Container> = prop_bindings.iter().collect();
        x_components.extend(prop_refs.iter());

        let x_dense = if x_components.is_empty() {
            Container::random(dn) // deterministic from DN if no properties
        } else {
            Container::bundle(&x_components)
        };

        let x = SparseContainer::from_dense(&x_dense);
        let y = SparseContainer::zero(); // nodes have no verb
        let z = x.clone(); // self-referential for CAM content lookup

        Self::assemble_record(dn, &x, &y, &z, nars, 0)
    }

    /// Build an edge record (relationship between two entities).
    ///
    /// X axis = BIND(src_fp, PERMUTE(verb_fp, 1))  — forward-query optimized
    /// Y axis = verb_fp                              — pure predicate
    /// Z axis = BIND(tgt_fp, PERMUTE(verb_fp, 2))   — reverse-query optimized
    pub fn build_edge(
        dn: u64,
        src_fp: &Container,
        verb_fp: &Container,
        tgt_fp: &Container,
        nars: TruthValue,
    ) -> Result<CogRecord, SpoError> {
        let verb_role1 = verb_fp.permute(1); // role marker for subject slot
        let verb_role2 = verb_fp.permute(2); // role marker for object slot

        let x_dense = src_fp.xor(&verb_role1);  // BIND(src, permute(verb,1))
        let y_dense = verb_fp.clone();            // pure verb
        let z_dense = tgt_fp.xor(&verb_role2);   // BIND(tgt, permute(verb,2))

        let x = SparseContainer::from_dense(&x_dense);
        let y = SparseContainer::from_dense(&y_dense);
        let z = SparseContainer::from_dense(&z_dense);

        Self::assemble_record(dn, &x, &y, &z, nars, 0)
    }

    /// Build a meta-awareness record from a causal chain.
    ///
    /// X axis = BUNDLE(chain node X axes)   — entities involved in the chain
    /// Y axis = deterministic CHAIN marker  — "I noticed a causal chain"
    /// Z axis = BUNDLE(chain node Z axes)   — what the chain implies
    ///
    /// The meta-record's Z can become another chain's X (Piaget recursion).
    pub fn build_meta_awareness(
        dn: u64,
        chain_records: &[&CogRecord],
        chain_coherence: f32,
    ) -> Result<CogRecord, SpoError> {
        if chain_records.is_empty() {
            return Err(SpoError::EmptyChain);
        }

        // Collect X and Z axes from chain records
        let mut x_denses = Vec::new();
        let mut z_denses = Vec::new();

        for record in chain_records {
            let desc = AxisDescriptors::from_words(&[
                record.meta.words[W_AXIS_DESC],
                record.meta.words[W_AXIS_DESC + 1],
            ]);
            if let Ok((x, _y, z)) = unpack_axes(&record.content, &desc) {
                x_denses.push(x.to_dense());
                z_denses.push(z.to_dense());
            }
        }

        let x_refs: Vec<&Container> = x_denses.iter().collect();
        let z_refs: Vec<&Container> = z_denses.iter().collect();

        let x_dense = Container::bundle(&x_refs);
        // Y axis: deterministic "chain discovered" marker
        let y_dense = Container::random(0xCHA1_D15C); // CHAIN_DISCOVERED seed
        let z_dense = Container::bundle(&z_refs);

        let x = SparseContainer::from_dense(&x_dense);
        let y = SparseContainer::from_dense(&y_dense);
        let z = SparseContainer::from_dense(&z_dense);

        let nars = TruthValue::new(
            chain_coherence.clamp(0.0, 1.0),
            (chain_coherence * 0.8).clamp(0.0, 1.0),
        );

        Self::assemble_record(dn, &x, &y, &z, nars, 0b10) // flag: is_meta_awareness
    }

    // ========================================================================
    // INTERNAL
    // ========================================================================

    fn assemble_record(
        dn: u64,
        x: &SparseContainer,
        y: &SparseContainer,
        z: &SparseContainer,
        nars: TruthValue,
        extra_flags: u16,
    ) -> Result<CogRecord, SpoError> {
        // Pack sparse axes into content container
        let (content, mut desc) = pack_axes(x, y, z)?;
        desc.flags |= extra_flags;

        // Compute scent
        let scent = NibbleScent::from_axes(x, y, z);

        // Build meta container
        let mut meta = Container::zero();

        // W0: DN address
        meta.words[W_DN_ADDR] = dn;

        // W1: type info with geometry=Spo(6)
        meta.words[W_TYPE] = (GEOMETRY_SPO as u64) << 16
            | 1u64 << 8  // container count = 1 content
            | 0u64;      // node_kind = 0 (generic)

        // W4-W7: NARS truth
        meta.words[W_NARS_BASE] = nars.frequency.to_bits() as u64;
        meta.words[W_NARS_BASE + 1] = nars.confidence.to_bits() as u64;

        // W12-W17: Scent
        let scent_words = scent.to_words();
        for (i, &w) in scent_words.iter().enumerate() {
            meta.words[W_SCENT_BASE + i] = w;
        }

        // W34-W35: Axis descriptors
        let desc_words = desc.to_words();
        meta.words[W_AXIS_DESC] = desc_words[0];
        meta.words[W_AXIS_DESC + 1] = desc_words[1];

        Ok(CogRecord { meta, content })
    }
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

/// Create a deterministic fingerprint from a string label (for testing/seeding).
pub fn label_fp(label: &str) -> Container {
    // Simple hash: sum bytes with mixing
    let mut seed = 0u64;
    for (i, b) in label.bytes().enumerate() {
        seed ^= (b as u64).wrapping_mul(0x9e3779b97f4a7c15);
        seed = seed.rotate_left((i as u32) % 64);
    }
    Container::random(seed)
}

/// Create a deterministic DN hash from a string (for testing).
pub fn dn_hash(name: &str) -> u64 {
    let mut h = 0xcbf29ce484222325u64; // FNV-1a offset basis
    for b in name.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3); // FNV prime
    }
    h
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_node_roundtrip() {
        let lbl_person = label_fp("Person");
        let key_name = label_fp("name");
        let val_jan = label_fp("Jan");

        let record = SpoBuilder::build_node(
            dn_hash("jan"),
            &[&lbl_person],
            &[(&key_name, &val_jan)],
            TruthValue::new(1.0, 0.9),
        )
        .unwrap();

        // Verify DN is stamped
        assert_eq!(record.meta.words[0], dn_hash("jan"));

        // Verify geometry is Spo
        let geom = (record.meta.words[1] >> 16) & 0xFF;
        assert_eq!(geom, GEOMETRY_SPO as u64);

        // Verify axes can be unpacked
        let desc = AxisDescriptors::from_words(&[record.meta.words[34], record.meta.words[35]]);
        let (x, y, z) = unpack_axes(&record.content, &desc).unwrap();

        // Y should be zero (node has no verb)
        assert_eq!(y.word_count(), 0);

        // X and Z should be identical (self-referential)
        assert_eq!(x.bitmap, z.bitmap);
        assert_eq!(x.words, z.words);

        // X should be non-empty
        assert!(x.word_count() > 0);
    }

    #[test]
    fn test_build_edge_three_axes() {
        let src = Container::random(1); // Jan
        let verb = Container::random(2); // KNOWS
        let tgt = Container::random(3); // Ada

        let record = SpoBuilder::build_edge(
            dn_hash("jan_knows_ada"),
            &src,
            &verb,
            &tgt,
            TruthValue::new(0.8, 0.9),
        )
        .unwrap();

        let desc = AxisDescriptors::from_words(&[record.meta.words[34], record.meta.words[35]]);
        let (x, y, z) = unpack_axes(&record.content, &desc).unwrap();

        // All three axes should be populated for edges
        assert!(x.word_count() > 0, "X axis should be populated");
        assert!(y.word_count() > 0, "Y axis should be populated");
        assert!(z.word_count() > 0, "Z axis should be populated");

        // Y should be close to the verb (it IS the verb)
        let y_dense = y.to_dense();
        let dist_to_verb = y_dense.hamming(&verb);
        assert_eq!(dist_to_verb, 0, "Y axis should equal verb exactly");
    }

    #[test]
    fn test_build_meta_awareness() {
        let src = Container::random(10);
        let verb1 = Container::random(20);
        let tgt1 = Container::random(30);
        let verb2 = Container::random(40);
        let tgt2 = Container::random(50);

        let edge1 = SpoBuilder::build_edge(
            dn_hash("e1"), &src, &verb1, &tgt1,
            TruthValue::new(0.8, 0.9),
        ).unwrap();

        let edge2 = SpoBuilder::build_edge(
            dn_hash("e2"), &tgt1, &verb2, &tgt2,
            TruthValue::new(0.7, 0.8),
        ).unwrap();

        let meta = SpoBuilder::build_meta_awareness(
            dn_hash("meta_e1_e2"),
            &[&edge1, &edge2],
            0.85,
        ).unwrap();

        // Verify meta-awareness flag is set
        let desc = AxisDescriptors::from_words(&[meta.meta.words[34], meta.meta.words[35]]);
        assert!(desc.is_meta_awareness());

        // Verify scent is populated
        let scent = NibbleScent::from_words(&[
            meta.meta.words[12], meta.meta.words[13],
            meta.meta.words[14], meta.meta.words[15],
            meta.meta.words[16], meta.meta.words[17],
        ]);
        assert_ne!(scent, NibbleScent::zero());
    }

    #[test]
    fn test_label_fp_deterministic() {
        let a = label_fp("Person");
        let b = label_fp("Person");
        assert_eq!(a, b);

        let c = label_fp("Concept");
        assert_ne!(a, c);
    }

    #[test]
    fn test_dn_hash_deterministic() {
        let a = dn_hash("jan");
        let b = dn_hash("jan");
        assert_eq!(a, b);

        let c = dn_hash("ada");
        assert_ne!(a, c);
    }
}
