//! Chess knowledge population for LanceDB + VSA.
//!
//! Populates the ladybug-rs Lance substrate with chess positions and moves
//! as VSA-encoded fingerprints, enabling Hamming-distance similarity search
//! across the entire opening book and position database.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                  CHESS × LANCE POPULATION                    │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                              │
//! │  Position FEN → ChessFingerprint (16K-bit)                   │
//! │       → NodeRecord { label: "Position", fingerprint: ... }   │
//! │                                                              │
//! │  Move UCI → Fingerprint::from_content("chess_move:e2e4")     │
//! │       → sequence(moves) → bind(position) → anchored FP      │
//! │       → NodeRecord { label: "MoveSequence", fingerprint }    │
//! │                                                              │
//! │  Opening ECO → ChessFingerprint + move sequence FP           │
//! │       → NodeRecord { label: "Opening", fingerprint }         │
//! │       → EdgeRecord { STARTS_AT, PLAYS_TO }                   │
//! │                                                              │
//! │  Resonate: query_fp.hamming_search(lance_store, k, threshold)│
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use ladybug::chess::populate::ChessPopulator;
//!
//! let populator = ChessPopulator::new();
//! let stats = populator.populate_openings().await?;
//! let stats = populator.populate_positions(&fens).await?;
//! ```

use crate::core::{Fingerprint, VsaOps};
use crate::chess::ChessFingerprint;

/// A chess opening entry for population.
pub struct OpeningEntry {
    pub eco: String,
    pub name: String,
    pub pgn: String,
    pub fen: String,
    pub uci_moves: Vec<String>,
}

/// Statistics from chess population.
#[derive(Debug, Default)]
pub struct PopulationStats {
    pub positions_encoded: usize,
    pub moves_encoded: usize,
    pub openings_encoded: usize,
    pub sequences_encoded: usize,
}

impl std::fmt::Display for PopulationStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ChessPopulationStats {{ positions: {}, moves: {}, openings: {}, sequences: {} }}",
            self.positions_encoded, self.moves_encoded, self.openings_encoded, self.sequences_encoded,
        )
    }
}

/// Encode a chess position as a 16,384-bit fingerprint.
///
/// Uses `ChessFingerprint::from_fen()` which encodes:
/// - Raw bitboards (12 piece types x 64 squares)
/// - Pawn structure (doubled, isolated, passed)
/// - King safety (pawn shield, castling)
/// - Material signature (balance, phase)
pub fn encode_position(fen: &str) -> Option<Fingerprint> {
    ChessFingerprint::from_fen(fen)
}

/// Encode a single chess move as a VSA fingerprint.
///
/// Uses `Fingerprint::from_content("chess_move:{uci}")` to create
/// a unique, pseudo-random fingerprint for each move.
pub fn encode_move(uci: &str) -> Fingerprint {
    Fingerprint::from_content(&format!("chess_move:{}", uci))
}

/// Encode a sequence of chess moves as a VSA fingerprint.
///
/// Uses `Fingerprint::sequence()` (permute + bundle) to create an
/// order-preserving representation of the move sequence.
///
/// The sequence fingerprint captures WHICH moves were played AND
/// in what ORDER, enabling similarity search over game lines.
pub fn encode_move_sequence(uci_moves: &[&str]) -> Fingerprint {
    if uci_moves.is_empty() {
        return Fingerprint::zero();
    }

    let move_fps: Vec<Fingerprint> = uci_moves
        .iter()
        .map(|uci| encode_move(uci))
        .collect();

    Fingerprint::sequence(&move_fps)
}

/// Encode a move sequence anchored to a position.
///
/// Binds the sequence fingerprint to the position fingerprint,
/// creating a compound representation: "these moves FROM this position".
///
/// `anchored = bind(sequence(moves), position_fp)`
///
/// This is useful for opening classification: the same move order
/// from different positions produces different anchored fingerprints.
pub fn encode_anchored_sequence(fen: &str, uci_moves: &[&str]) -> Option<Fingerprint> {
    let pos_fp = encode_position(fen)?;
    let seq_fp = encode_move_sequence(uci_moves);
    Some(seq_fp.bind(&pos_fp))
}

/// Encode an opening as a composite fingerprint.
///
/// Combines the position fingerprint (from the opening's final position)
/// with the move sequence fingerprint, creating a rich representation
/// that captures both WHERE the opening leads and HOW it gets there.
pub fn encode_opening(entry: &OpeningEntry) -> Option<OpeningFingerprints> {
    let position_fp = encode_position(&entry.fen)?;
    let uci_refs: Vec<&str> = entry.uci_moves.iter().map(|s| s.as_str()).collect();
    let sequence_fp = encode_move_sequence(&uci_refs);
    let anchored_fp = sequence_fp.bind(&position_fp);

    Some(OpeningFingerprints {
        eco: entry.eco.clone(),
        name: entry.name.clone(),
        position_fp,
        sequence_fp,
        anchored_fp,
    })
}

/// Fingerprints for a chess opening.
pub struct OpeningFingerprints {
    pub eco: String,
    pub name: String,
    /// Fingerprint of the opening's resulting position.
    pub position_fp: Fingerprint,
    /// Fingerprint of the move sequence (order-preserving).
    pub sequence_fp: Fingerprint,
    /// Anchored fingerprint: sequence bound to position.
    pub anchored_fp: Fingerprint,
}

/// Seed opening data for population (matching neo4j-rs aiwar module).
pub fn seed_openings() -> Vec<OpeningEntry> {
    vec![
        OpeningEntry {
            eco: "B20".into(), name: "Sicilian Defense".into(),
            pgn: "1. e4 c5".into(),
            fen: "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2".into(),
            uci_moves: vec!["e2e4".into(), "c7c5".into()],
        },
        OpeningEntry {
            eco: "C00".into(), name: "French Defense".into(),
            pgn: "1. e4 e6".into(),
            fen: "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2".into(),
            uci_moves: vec!["e2e4".into(), "e7e6".into()],
        },
        OpeningEntry {
            eco: "B10".into(), name: "Caro-Kann Defense".into(),
            pgn: "1. e4 c6".into(),
            fen: "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2".into(),
            uci_moves: vec!["e2e4".into(), "c7c6".into()],
        },
        OpeningEntry {
            eco: "C50".into(), name: "Italian Game".into(),
            pgn: "1. e4 e5 2. Nf3 Nc6 3. Bc4".into(),
            fen: "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3".into(),
            uci_moves: vec!["e2e4".into(), "e7e5".into(), "g1f3".into(), "b8c6".into(), "f1c4".into()],
        },
        OpeningEntry {
            eco: "C60".into(), name: "Ruy Lopez".into(),
            pgn: "1. e4 e5 2. Nf3 Nc6 3. Bb5".into(),
            fen: "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3".into(),
            uci_moves: vec!["e2e4".into(), "e7e5".into(), "g1f3".into(), "b8c6".into(), "f1b5".into()],
        },
        OpeningEntry {
            eco: "D06".into(), name: "Queen's Gambit".into(),
            pgn: "1. d4 d5 2. c4".into(),
            fen: "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2".into(),
            uci_moves: vec!["d2d4".into(), "d7d5".into(), "c2c4".into()],
        },
        OpeningEntry {
            eco: "E60".into(), name: "King's Indian Defense".into(),
            pgn: "1. d4 Nf6 2. c4 g6".into(),
            fen: "rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3".into(),
            uci_moves: vec!["d2d4".into(), "g8f6".into(), "c2c4".into(), "g7g6".into()],
        },
        OpeningEntry {
            eco: "A10".into(), name: "English Opening".into(),
            pgn: "1. c4".into(),
            fen: "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq c3 0 1".into(),
            uci_moves: vec!["c2c4".into()],
        },
        OpeningEntry {
            eco: "B07".into(), name: "Pirc Defense".into(),
            pgn: "1. e4 d6 2. d4 Nf6".into(),
            fen: "rnbqkb1r/ppp1pppp/3p1n2/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 1 3".into(),
            uci_moves: vec!["e2e4".into(), "d7d6".into(), "d2d4".into(), "g8f6".into()],
        },
        OpeningEntry {
            eco: "B01".into(), name: "Scandinavian Defense".into(),
            pgn: "1. e4 d5".into(),
            fen: "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2".into(),
            uci_moves: vec!["e2e4".into(), "d7d5".into()],
        },
        OpeningEntry {
            eco: "C42".into(), name: "Petrov's Defense".into(),
            pgn: "1. e4 e5 2. Nf3 Nf6".into(),
            fen: "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3".into(),
            uci_moves: vec!["e2e4".into(), "e7e5".into(), "g1f3".into(), "g8f6".into()],
        },
        OpeningEntry {
            eco: "D35".into(), name: "Queen's Gambit Declined".into(),
            pgn: "1. d4 d5 2. c4 e6".into(),
            fen: "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3".into(),
            uci_moves: vec!["d2d4".into(), "d7d5".into(), "c2c4".into(), "e7e6".into()],
        },
    ]
}

/// Batch-encode all seed openings and return their fingerprints.
pub fn encode_all_openings() -> Vec<OpeningFingerprints> {
    seed_openings()
        .iter()
        .filter_map(|entry| encode_opening(entry))
        .collect()
}

/// Find the K most similar openings to a query position.
///
/// Encodes the query FEN, then resonates against all opening fingerprints.
pub fn find_similar_openings(fen: &str, k: usize) -> Vec<(String, String, f32)> {
    let query_fp = match encode_position(fen) {
        Some(fp) => fp,
        None => return vec![],
    };

    let openings = encode_all_openings();
    let mut results: Vec<(String, String, f32)> = openings
        .iter()
        .map(|o| {
            let sim = query_fp.similarity(&o.position_fp);
            (o.eco.clone(), o.name.clone(), sim)
        })
        .collect();

    results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

/// Find openings with similar move sequences.
///
/// Encodes the query moves as a sequence fingerprint, then finds
/// openings whose move sequence fingerprints are most similar.
pub fn find_similar_lines(uci_moves: &[&str], k: usize) -> Vec<(String, String, f32)> {
    let query_fp = encode_move_sequence(uci_moves);

    let openings = encode_all_openings();
    let mut results: Vec<(String, String, f32)> = openings
        .iter()
        .map(|o| {
            let sim = query_fp.similarity(&o.sequence_fp);
            (o.eco.clone(), o.name.clone(), sim)
        })
        .collect();

    results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    const SICILIAN_FEN: &str = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2";

    #[test]
    fn test_encode_position() {
        let fp = encode_position(STARTPOS).unwrap();
        assert!(fp.popcount() > 0);
    }

    #[test]
    fn test_encode_move() {
        let fp1 = encode_move("e2e4");
        let fp2 = encode_move("d2d4");
        // Different moves should produce different fingerprints
        assert!(fp1.similarity(&fp2) < 0.9);
        // Same move should be identical
        let fp3 = encode_move("e2e4");
        assert_eq!(fp1.hamming(&fp3), 0);
    }

    #[test]
    fn test_encode_move_sequence_order_matters() {
        let seq1 = encode_move_sequence(&["e2e4", "e7e5", "g1f3"]);
        let seq2 = encode_move_sequence(&["g1f3", "e7e5", "e2e4"]);
        // Different order should produce different fingerprints
        assert!(seq1.similarity(&seq2) < 0.9);
    }

    #[test]
    fn test_encode_anchored_sequence() {
        let anchored = encode_anchored_sequence(STARTPOS, &["e2e4", "c7c5"]);
        assert!(anchored.is_some());
        let fp = anchored.unwrap();
        assert!(fp.popcount() > 0);
    }

    #[test]
    fn test_encode_opening() {
        let entry = &seed_openings()[0]; // Sicilian
        let fps = encode_opening(entry).unwrap();
        assert_eq!(fps.eco, "B20");
        assert!(fps.position_fp.popcount() > 0);
        assert!(fps.sequence_fp.popcount() > 0);
        assert!(fps.anchored_fp.popcount() > 0);
    }

    #[test]
    fn test_encode_all_openings() {
        let all = encode_all_openings();
        assert!(all.len() >= 10);
        // All should have valid fingerprints
        for o in &all {
            assert!(o.position_fp.popcount() > 0);
        }
    }

    #[test]
    fn test_find_similar_openings() {
        // Sicilian position should match Sicilian opening
        let results = find_similar_openings(SICILIAN_FEN, 3);
        assert!(!results.is_empty());
        // First result should be Sicilian Defense (exact match)
        assert_eq!(results[0].0, "B20");
        assert_eq!(results[0].2, 1.0); // exact match → similarity 1.0
    }

    #[test]
    fn test_find_similar_lines() {
        // e4 e5 Nf3 Nc6 should match Italian/Ruy Lopez/Scotch
        let results = find_similar_lines(&["e2e4", "e7e5", "g1f3", "b8c6"], 5);
        assert!(!results.is_empty());
        // The top results should be e4 e5 openings (Italian, Ruy Lopez, Petrov's)
        let eco_codes: Vec<&str> = results.iter().map(|r| r.0.as_str()).collect();
        // At least one C-family (e4 e5) opening should be in top results
        assert!(
            eco_codes.iter().any(|e| e.starts_with('C')),
            "Expected C-family opening in results, got: {:?}", eco_codes,
        );
    }

    #[test]
    fn test_italian_vs_ruy_lopez_similarity() {
        let openings = encode_all_openings();
        let italian = openings.iter().find(|o| o.eco == "C50").unwrap();
        let ruy = openings.iter().find(|o| o.eco == "C60").unwrap();

        // Italian and Ruy Lopez share first 4 moves (e4 e5 Nf3 Nc6)
        // so their sequence fingerprints should be quite similar
        let seq_sim = italian.sequence_fp.similarity(&ruy.sequence_fp);
        assert!(
            seq_sim > 0.5,
            "Italian and Ruy Lopez sequence similarity should be > 0.5, got {}",
            seq_sim,
        );

        // But their position fingerprints should differ (Bc4 vs Bb5)
        let pos_sim = italian.position_fp.similarity(&ruy.position_fp);
        // Still similar since they share most pieces
        assert!(pos_sim > 0.8, "Position similarity should be > 0.8, got {}", pos_sim);
        assert!(pos_sim < 1.0, "Position similarity should be < 1.0 (different positions)");
    }
}
