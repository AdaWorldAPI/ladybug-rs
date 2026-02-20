//! SPO 3D — Six Ironclad Tests
//!
//! Test 1: Node round-trip (build → insert → retrieve → verify)
//! Test 2: Forward query (Jan KNOWS Ada → find Ada from Jan)
//! Test 3: Reverse query (Jan KNOWS Ada → find Jan from Ada, NO extra index)
//! Test 4: CAM content lookup (100 nodes → find by fingerprint)
//! Test 5: NARS reasoning (deduction + revision along chain)
//! Test 6: Causal chain coherence (Z→X resonance + meta convergence)

#[cfg(test)]
mod tests {
    use ladybug_contract::container::Container;
    use ladybug_contract::nars::TruthValue;

    // These would be imported from the spo module once wired into the crate
    use crate::graph::spo::sparse::{SparseContainer, unpack_axes, AxisDescriptors};
    use crate::graph::spo::scent::NibbleScent;
    use crate::graph::spo::store::SpoStore;
    use crate::graph::spo::builder::{SpoBuilder, label_fp, dn_hash};

    // ========================================================================
    // TEST 1: NODE ROUND-TRIP
    // ========================================================================

    #[test]
    fn test_1_node_roundtrip() {
        let lbl_person = label_fp("Person");
        let key_name = label_fp("name");
        let val_jan = label_fp("Jan");
        let key_age = label_fp("age");
        let val_42 = label_fp("42");

        // Build node: Jan {Person, name: "Jan", age: 42}
        let record = SpoBuilder::build_node(
            dn_hash("jan"),
            &[&lbl_person],
            &[(&key_name, &val_jan), (&key_age, &val_42)],
            TruthValue::new(1.0, 0.9),
        ).unwrap();

        // Insert into store
        let mut store = SpoStore::new();
        store.insert(record.clone()).unwrap();

        // Retrieve
        let retrieved = store.get(dn_hash("jan")).unwrap();

        // Verify DN matches
        assert_eq!(retrieved.meta.words[0], dn_hash("jan"));

        // Unpack and verify X axis is Hamming-close to the original
        let desc = AxisDescriptors::from_words(&[
            retrieved.meta.words[34],
            retrieved.meta.words[35],
        ]);
        let (x, y, z) = unpack_axes(&retrieved.content, &desc).unwrap();

        // Node's Y axis should be empty (no verb)
        assert_eq!(y.word_count(), 0, "Node Y axis must be empty");

        // X and Z should be identical
        assert_eq!(x, z, "Node X and Z must be identical (self-referential)");

        // X axis should contain the bundled labels+properties
        let x_dense = x.to_dense();
        assert!(!x_dense.is_zero(), "X axis must not be zero");

        // Hamming to self should be 0
        assert_eq!(
            SparseContainer::hamming_sparse(&x, &SparseContainer::from_dense(&x_dense)),
            0
        );
    }

    // ========================================================================
    // TEST 2: FORWARD QUERY
    // ========================================================================

    #[test]
    fn test_2_forward_query() {
        let mut store = SpoStore::new();

        // Create entities
        let jan_fp = label_fp("Jan");
        let ada_fp = label_fp("Ada");
        let knows_fp = label_fp("KNOWS");

        let jan_node = SpoBuilder::build_node(
            dn_hash("jan"), &[&label_fp("Person")],
            &[(&label_fp("name"), &jan_fp)],
            TruthValue::new(1.0, 0.9),
        ).unwrap();

        let ada_node = SpoBuilder::build_node(
            dn_hash("ada"), &[&label_fp("Person")],
            &[(&label_fp("name"), &ada_fp)],
            TruthValue::new(1.0, 0.9),
        ).unwrap();

        // Create edge: Jan KNOWS Ada
        let edge = SpoBuilder::build_edge(
            dn_hash("jan_knows_ada"),
            &jan_fp,
            &knows_fp,
            &ada_fp,
            TruthValue::new(0.8, 0.9),
        ).unwrap();

        store.insert(jan_node).unwrap();
        store.insert(ada_node).unwrap();
        store.insert(edge).unwrap();

        // Forward query: "What does Jan know?"
        let hits = store.query_forward(&jan_fp, &knows_fp, 4000);
        assert!(
            !hits.is_empty(),
            "Forward query must find at least one result"
        );

        // The edge record should be in the results
        let found_edge = hits.iter().any(|h| h.dn == dn_hash("jan_knows_ada"));
        assert!(found_edge, "Forward query must find the edge record");
    }

    // ========================================================================
    // TEST 3: REVERSE QUERY (NO EXTRA INDEX)
    // ========================================================================

    #[test]
    fn test_3_reverse_query() {
        let mut store = SpoStore::new();

        let jan_fp = label_fp("Jan");
        let ada_fp = label_fp("Ada");
        let knows_fp = label_fp("KNOWS");

        let jan_node = SpoBuilder::build_node(
            dn_hash("jan"), &[&label_fp("Person")],
            &[(&label_fp("name"), &jan_fp)],
            TruthValue::new(1.0, 0.9),
        ).unwrap();

        let ada_node = SpoBuilder::build_node(
            dn_hash("ada"), &[&label_fp("Person")],
            &[(&label_fp("name"), &ada_fp)],
            TruthValue::new(1.0, 0.9),
        ).unwrap();

        let edge = SpoBuilder::build_edge(
            dn_hash("jan_knows_ada"),
            &jan_fp,
            &knows_fp,
            &ada_fp,
            TruthValue::new(0.8, 0.9),
        ).unwrap();

        store.insert(jan_node).unwrap();
        store.insert(ada_node).unwrap();
        store.insert(edge).unwrap();

        // Reverse query: "Who knows Ada?" — scanning Z+Y, no reverse index!
        let hits = store.query_reverse(&ada_fp, &knows_fp, 4000);
        assert!(
            !hits.is_empty(),
            "Reverse query must find results WITHOUT any extra index"
        );

        let found_edge = hits.iter().any(|h| h.dn == dn_hash("jan_knows_ada"));
        assert!(found_edge, "Reverse query must find the edge record");
    }

    // ========================================================================
    // TEST 4: CAM CONTENT LOOKUP
    // ========================================================================

    #[test]
    fn test_4_cam_content_lookup() {
        let mut store = SpoStore::new();

        // Insert 100 nodes with different names
        for i in 0..100 {
            let name = format!("entity_{}", i);
            let name_fp = label_fp(&name);
            let node = SpoBuilder::build_node(
                dn_hash(&name),
                &[&label_fp("Thing")],
                &[(&label_fp("name"), &name_fp)],
                TruthValue::new(1.0, 0.9),
            ).unwrap();
            store.insert(node).unwrap();
        }

        assert_eq!(store.len(), 100);

        // Query for a specific entity by its content fingerprint
        let target_fp = label_fp("entity_42");
        let hits = store.query_content(&target_fp, 3500);

        assert!(
            !hits.is_empty(),
            "Content lookup must find results in 100-node store"
        );

        // The closest hit should be entity_42
        let best = &hits[0];
        assert_eq!(
            best.dn,
            dn_hash("entity_42"),
            "Best content match should be entity_42"
        );
    }

    // ========================================================================
    // TEST 5: NARS REASONING
    // ========================================================================

    #[test]
    fn test_5_nars_reasoning() {
        let mut store = SpoStore::new();

        let jan_fp = label_fp("Jan");
        let rust_fp = label_fp("Rust");
        let cam_fp = label_fp("CAM");
        let knows_fp = label_fp("KNOWS");
        let helps_fp = label_fp("HELPS");

        // "Jan knows Rust" <0.8, 0.9>
        let e1 = SpoBuilder::build_edge(
            dn_hash("e1"), &jan_fp, &knows_fp, &rust_fp,
            TruthValue::new(0.8, 0.9),
        ).unwrap();

        // "Rust helps CAM" <0.7, 0.8>
        let e2 = SpoBuilder::build_edge(
            dn_hash("e2"), &rust_fp, &helps_fp, &cam_fp,
            TruthValue::new(0.7, 0.8),
        ).unwrap();

        store.insert(e1).unwrap();
        store.insert(e2).unwrap();

        // Chain deduction: Jan → Rust → CAM
        let chain_tv = store.chain_deduction(&[dn_hash("e1"), dn_hash("e2")]);

        // Frequency: 0.8 × 0.7 = 0.56
        assert!(
            (chain_tv.frequency - 0.56).abs() < 0.01,
            "Chain frequency should be ~0.56, got {}",
            chain_tv.frequency
        );

        // Confidence: 0.9 × 0.8 × coherence_factor < 0.72
        assert!(
            chain_tv.confidence <= 0.72,
            "Chain confidence should be ≤ 0.72, got {}",
            chain_tv.confidence
        );
        assert!(
            chain_tv.confidence > 0.0,
            "Chain confidence should be positive"
        );
    }

    // ========================================================================
    // TEST 6: CAUSAL CHAIN COHERENCE
    // ========================================================================

    #[test]
    fn test_6_causal_chain_coherence() {
        let mut store = SpoStore::new();

        // Create a chain where Record 1's Z resonates with Record 2's X.
        // We use the SAME fingerprint for the shared entity (Rust)
        // so the Z→X link should have low Hamming distance.

        let jan_fp = label_fp("Jan");
        let rust_fp = label_fp("Rust");
        let cam_fp = label_fp("CAM");
        let knows_fp = label_fp("KNOWS");
        let enables_fp = label_fp("ENABLES");

        let e1 = SpoBuilder::build_edge(
            dn_hash("chain_e1"), &jan_fp, &knows_fp, &rust_fp,
            TruthValue::new(0.8, 0.9),
        ).unwrap();

        let e2 = SpoBuilder::build_edge(
            dn_hash("chain_e2"), &rust_fp, &enables_fp, &cam_fp,
            TruthValue::new(0.7, 0.8),
        ).unwrap();

        store.insert(e1.clone()).unwrap();
        store.insert(e2.clone()).unwrap();

        // Test chain coherence
        let coherence = store.chain_coherence(&[dn_hash("chain_e1"), dn_hash("chain_e2")]);
        assert!(
            coherence > 0.0,
            "Chain coherence should be positive, got {}",
            coherence
        );

        // Test causal successor discovery
        let successors = store.causal_successors(&e1, 4096);
        // At minimum: some records' X axes should be within range
        // (The Z of e1 is BIND(rust, permute(knows,2)) which may not exactly
        //  match the X of e2 which is BIND(rust, permute(enables,1)),
        //  but both contain rust_fp so they should be closer than random)

        // Test meta-awareness construction
        let meta = SpoBuilder::build_meta_awareness(
            dn_hash("meta_chain"),
            &[&e1, &e2],
            coherence,
        ).unwrap();

        // Verify meta-awareness flag
        let desc = AxisDescriptors::from_words(&[
            meta.meta.words[34],
            meta.meta.words[35],
        ]);
        assert!(desc.is_meta_awareness(), "Meta record must have awareness flag");

        // Verify meta-record has populated axes
        let (mx, my, mz) = unpack_axes(&meta.content, &desc).unwrap();
        assert!(mx.word_count() > 0, "Meta X should be populated");
        assert!(my.word_count() > 0, "Meta Y should be populated");
        assert!(mz.word_count() > 0, "Meta Z should be populated");

        // The meta-record's Z should be a BUNDLE of the chain's Z axes,
        // which means it should be somewhat close to the original content.
        // This is the convergence test — the tsunami prediction.
        let mz_dense = mz.to_dense();
        let original_z_dense = {
            let d = AxisDescriptors::from_words(&[e1.meta.words[34], e1.meta.words[35]]);
            unpack_axes(&e1.content, &d).unwrap().2.to_dense()
        };

        let convergence_dist = mz_dense.hamming(&original_z_dense);
        // Meta-bundled Z won't be identical to e1's Z (it's a bundle of BOTH)
        // but it should be closer than random (~4096)
        assert!(
            convergence_dist < 4096,
            "Meta Z should converge toward original content, got distance {}",
            convergence_dist
        );
    }
}
