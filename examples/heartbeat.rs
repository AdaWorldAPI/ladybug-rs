//! Heartbeat Demo — Proof of life for the ladybug-rs cognitive substrate.
//!
//! Exercises the core subsystems in a single pass to verify the system is
//! alive and functioning:
//!
//!   1. BindSpace — 8+8 addressing, O(1) read/write
//!   2. CogRedis  — DN.SET/DN.GET command execution
//!   3. NARS      — Truth value revision
//!   4. Collapse  — Gate evaluation (FLOW/HOLD/BLOCK)
//!   5. SIMD      — Hamming distance computation
//!
//! Usage: `cargo run --example heartbeat`

use std::time::Instant;

use ladybug::cognitive::{GateState, get_gate_state};
use ladybug::core::Fingerprint;
use ladybug::core::simd::hamming_distance;
use ladybug::nars::TruthValue;
use ladybug::storage::{Addr, BindSpace, FINGERPRINT_WORDS};

fn main() {
    println!();
    println!("==========================================================");
    println!("  LADYBUG-RS HEARTBEAT");
    println!("==========================================================");
    println!();

    let t0 = Instant::now();
    let mut checks_passed = 0u32;
    let total_checks = 5u32;

    // ── 1. BindSpace ────────────────────────────────────────────────────
    print!("  [1/5] BindSpace 8+8 addressing ... ");
    {
        let mut bs = BindSpace::new();
        let addr = Addr::new(0x80, 0x01); // Node zone
        let fp = [42u64; FINGERPRINT_WORDS];
        bs.write_at(addr, fp);
        let read_back = bs.read(addr);
        assert!(read_back.is_some(), "BindSpace read after write failed");
        let node = read_back.unwrap();
        assert_eq!(node.fingerprint[0], 42, "Fingerprint data mismatch");
        checks_passed += 1;
        println!("OK  (addr={:04x})", addr.0);
    }

    // ── 2. CogRedis ────────────────────────────────────────────────────
    print!("  [2/5] CogRedis DN.SET/DN.GET ... ");
    {
        use ladybug::storage::{CogRedis, RedisResult};
        let mut redis = CogRedis::new();

        // DN.SET returns the hex address of the new node
        let result = redis.execute_command("DN.SET Ada:A:heartbeat:test hello");
        let addr_hex = match &result {
            RedisResult::String(s) => {
                assert!(!s.is_empty(), "DN.SET should return address");
                s.clone()
            }
            _ => panic!("DN.SET failed: {:?}", result),
        };

        // DN.GET returns an array with node info
        let get_result = redis.execute_command("DN.GET Ada:A:heartbeat:test");
        match &get_result {
            RedisResult::Array(arr) => {
                assert!(!arr.is_empty(), "DN.GET should return node info");
            }
            _ => panic!("DN.GET failed: {:?}", get_result),
        }
        checks_passed += 1;
        println!("OK  (addr={})", addr_hex);
    }

    // ── 3. NARS Truth Value Revision ────────────────────────────────────
    print!("  [3/5] NARS revision ............. ");
    {
        let tv1 = TruthValue::new(0.9, 0.8);
        let tv2 = TruthValue::new(0.85, 0.7);
        let revised = tv1.revision(&tv2);

        // Revised confidence must be higher than either input
        assert!(
            revised.confidence > tv1.confidence && revised.confidence > tv2.confidence,
            "Revision must increase confidence: {:.3} should be > {:.3} and {:.3}",
            revised.confidence,
            tv1.confidence,
            tv2.confidence
        );
        // Frequency should be between inputs (weighted average)
        assert!(
            revised.frequency >= 0.0 && revised.frequency <= 1.0,
            "Frequency out of range"
        );
        checks_passed += 1;
        println!(
            "OK  (f={:.3}, c={:.3})",
            revised.frequency, revised.confidence
        );
    }

    // ── 4. Collapse Gate ────────────────────────────────────────────────
    print!("  [4/5] Collapse gate ............. ");
    {
        let flow = get_gate_state(0.05);
        let hold = get_gate_state(0.25);
        let block = get_gate_state(0.45);
        assert_eq!(flow, GateState::Flow, "SD=0.05 should be Flow");
        assert_eq!(hold, GateState::Hold, "SD=0.25 should be Hold");
        assert_eq!(block, GateState::Block, "SD=0.45 should be Block");
        checks_passed += 1;
        println!("OK  (Flow/Hold/Block)");
    }

    // ── 5. SIMD Hamming Distance ────────────────────────────────────────
    print!("  [5/5] SIMD hamming distance ..... ");
    {
        let a = Fingerprint::from_content("heartbeat_a");
        let b = Fingerprint::from_content("heartbeat_b");
        let d = hamming_distance(&a, &b);
        assert!(d > 0, "Different fingerprints should have distance > 0");
        // Self-distance must be 0
        assert_eq!(hamming_distance(&a, &a), 0, "Self-distance must be 0");
        checks_passed += 1;
        println!("OK  (d={})", d);
    }

    // ── Summary ─────────────────────────────────────────────────────────
    let elapsed = t0.elapsed();
    println!();
    println!("----------------------------------------------------------");
    println!(
        "  Result: {}/{} checks passed in {:.1}ms",
        checks_passed,
        total_checks,
        elapsed.as_secs_f64() * 1000.0
    );

    if checks_passed == total_checks {
        println!("  HEARTBEAT: ALIVE");
    } else {
        println!(
            "  HEARTBEAT: DEGRADED ({} failures)",
            total_checks - checks_passed
        );
    }
    println!("==========================================================");
    println!();
}
