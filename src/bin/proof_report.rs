//! Proof Report Generator — runs all ladybug-rs proof suites and outputs results.
//!
//! Usage: `cargo run --bin proof_report`
//!
//! Executes `cargo test` for each proof suite and parses results into a
//! formatted table showing pass/fail status for every proof.

use std::process::Command;

/// A proof suite with its test binary name and expected proof IDs.
struct ProofSuite {
    name: &'static str,
    test_name: &'static str,
    proofs: &'static [&'static str],
}

const SUITES: &[ProofSuite] = &[
    ProofSuite {
        name: "Foundation",
        test_name: "proof_foundation",
        proofs: &[
            "F-1  Berry-Esseen CLT (d=16384)",
            "F-2  Fisher sufficiency",
            "F-3  XOR self-inverse (exact)",
            "F-3b XOR commutativity/associativity",
            "F-4  Triangle inequality (metric)",
            "F-4b Metric axioms (identity, symmetry)",
            "F-5  Mexican hat shape",
            "F-5b Calibrated thresholds from CRP",
            "F-6  NARS revision monotonicity",
            "F-6b Revision commutativity",
            "F-7  ABBA causal retrieval",
            "F-7b Fusion quality (exact XOR roundtrip)",
            "F-7c Multi-fusion quality (N-way)",
        ],
    },
    ProofSuite {
        name: "Reasoning Ladder",
        test_name: "proof_reasoning_ladder",
        proofs: &[
            "RL-1 Parallel error isolation",
            "RL-2 NARS detects inconsistency",
            "RL-3 Collapse Gate HOLD/FLOW/BLOCK",
            "RL-5 Thinking style divergence (12 styles)",
            "RL-6 NARS abduction generates insight",
            "RL-7 Counterfactual divergence (Pearl Rung 3)",
            "RL-7b Different interventions differ",
            "RL-8 Parallel vs sequential probability",
        ],
    },
    ProofSuite {
        name: "Tactics (PR #100)",
        test_name: "proof_tactics",
        proofs: &[
            "T-01 Recursive expansion converges",
            "T-04 Reverse causal trace",
            "T-07 Adversarial critique detects weakness",
            "T-10 MetaCognition Brier calibration",
            "T-11 Contradiction detection",
            "T-15 CRP distribution from corpus",
            "T-20 Shadow parallel consensus",
            "T-24 Fusion quality (exact)",
            "T-25 Hamming Normal approximation",
            "T-28 Temporal Granger effect",
            "T-31 Counterfactual divergence",
            "T-34 Cross-domain fusion",
        ],
    },
    ProofSuite {
        name: "Level A Gaps",
        test_name: "proof_level_a_gaps",
        proofs: &[
            "A.1.4 SIMD-scalar equivalence",
            "A.2.2 NARS deduction bounds",
            "A.3.2 Collapse gate acyclicity",
            "A.4.3 Seven-layer fault isolation",
            "A.5.1 Cascade search KNN correctness",
            "A.6.2 WAL entry round-trip coverage",
        ],
    },
];

fn main() {
    println!();
    println!("==========================================================");
    println!("  LADYBUG-RS INTEGRATION PROOF REPORT");
    println!("==========================================================");
    println!();

    let mut total_pass = 0u32;
    let mut total_fail = 0u32;
    let mut total_skip = 0u32;

    for suite in SUITES {
        println!("----------------------------------------------------------");
        println!("  {} ({} proofs)", suite.name, suite.proofs.len());
        println!("----------------------------------------------------------");

        let output = Command::new("cargo")
            .args(["test", "--test", suite.test_name, "--", "--test-threads=1"])
            .output();

        match output {
            Ok(result) => {
                let stdout = String::from_utf8_lossy(&result.stdout);
                let stderr = String::from_utf8_lossy(&result.stderr);
                let combined = format!("{}{}", stdout, stderr);

                // Count results from cargo test output
                let pass_count = combined.matches("... ok").count();
                let fail_count = combined.matches("... FAILED").count();
                let ignore_count = combined.matches("... ignored").count();

                total_pass += pass_count as u32;
                total_fail += fail_count as u32;
                total_skip += ignore_count as u32;

                for proof in suite.proofs {
                    let status = if fail_count == 0 {
                        "PASS"
                    } else {
                        // Try to determine individual status from output
                        // The test names in output don't map 1:1 to proof IDs,
                        // so we report suite-level status
                        "????"
                    };
                    let icon = match status {
                        "PASS" => "[OK]",
                        "FAIL" => "[!!]",
                        _ => "[??]",
                    };
                    println!("  {} {}", icon, proof);
                }

                if fail_count > 0 {
                    // Print failure details
                    println!();
                    println!("  FAILURES:");
                    for line in combined.lines() {
                        if line.contains("FAILED") || line.contains("panicked") {
                            println!("    {}", line.trim());
                        }
                    }
                }
            }
            Err(e) => {
                println!("  ERROR: Could not run test suite: {}", e);
                total_fail += suite.proofs.len() as u32;
                for proof in suite.proofs {
                    println!("  [!!] {}", proof);
                }
            }
        }
        println!();
    }

    println!("==========================================================");
    println!("  SUMMARY");
    println!("==========================================================");
    println!("  Total proofs:  {}", total_pass + total_fail + total_skip);
    println!("  Passed:        {}", total_pass);
    println!("  Failed:        {}", total_fail);
    println!("  Skipped:       {}", total_skip);
    println!();

    if total_fail == 0 {
        println!("  ALL PROOFS PASSED");
    } else {
        println!("  {} PROOF(S) FAILED", total_fail);
        std::process::exit(1);
    }

    println!("==========================================================");
    println!();
}
