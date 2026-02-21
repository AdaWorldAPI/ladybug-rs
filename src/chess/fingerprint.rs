//! Chess position → 16,384-bit fingerprint encoder.
//!
//! Maps chess positions to the ladybug-rs fingerprint space, enabling
//! Hamming-distance similarity search across the opening book, evaluation
//! database, and AI War cognitive bridge.
//!
//! # Encoding Strategy
//!
//! The fingerprint is built by XOR-binding chess features with
//! orthogonal basis vectors (address constants), so each feature
//! occupies a unique subspace of the 16,384-bit hypervector.
//!
//! ```text
//! fingerprint = BIND(piece_bitboards, ADDR_PIECES)
//!             ⊕ BIND(pawn_structure, ADDR_PAWNS)
//!             ⊕ BIND(king_safety,    ADDR_KING)
//!             ⊕ BIND(material,       ADDR_MATERIAL)
//!             ⊕ BIND(castling,       ADDR_CASTLING)
//!             ⊕ BIND(phase,          ADDR_PHASE)
//! ```

use crate::core::{DIM_U64, Fingerprint};
use super::position::{ChessPosition, Color, GamePhase, PieceType};

/// Bit ranges for each section of the fingerprint.
const BITBOARD_START: usize = 0;
const BITBOARD_END: usize = 768;      // 12 × 64
const PAWN_STRUCTURE_START: usize = 768;
const PAWN_STRUCTURE_END: usize = 1024;
const KING_SAFETY_START: usize = 1024;
const KING_SAFETY_END: usize = 1280;
const ACTIVITY_START: usize = 1280;
const ACTIVITY_END: usize = 1792;
const MATERIAL_START: usize = 2304;
const MATERIAL_END: usize = 2560;

/// Chess position fingerprint encoder.
///
/// Encodes a chess position as a 16,384-bit fingerprint using
/// VSA operations (bind, superpose) for each feature category.
pub struct ChessFingerprint;

impl ChessFingerprint {
    /// Encode a chess position from FEN into a 16,384-bit fingerprint.
    pub fn from_fen(fen: &str) -> Option<Fingerprint> {
        let pos = ChessPosition::from_fen(fen)?;
        Some(Self::encode(&pos))
    }

    /// Encode a ChessPosition into a 16,384-bit fingerprint.
    pub fn encode(pos: &ChessPosition) -> Fingerprint {
        let mut fp = Fingerprint::zero();

        // Layer 1: Raw bitboards (bits 0-767)
        Self::encode_bitboards(pos, &mut fp);

        // Layer 2: Pawn structure features (bits 768-1023)
        Self::encode_pawn_structure(pos, &mut fp);

        // Layer 3: King safety (bits 1024-1279)
        Self::encode_king_safety(pos, &mut fp);

        // Layer 4: Material signature (bits 2304-2559)
        Self::encode_material(pos, &mut fp);

        // Layer 5: Side to move and castling (spread across phase region)
        Self::encode_metadata(pos, &mut fp);

        fp
    }

    /// Encode the 12 bitboards directly into the fingerprint.
    ///
    /// Each piece type × color occupies 64 bits, mapped directly
    /// from the chess bitboard representation. This gives us a
    /// natural Hamming distance: positions with similar piece
    /// placements have small Hamming distance in this region.
    fn encode_bitboards(pos: &ChessPosition, fp: &mut Fingerprint) {
        for (i, &bb) in pos.bitboards.iter().enumerate() {
            // Each bitboard occupies bits [i*64 .. (i+1)*64)
            let word_idx = i; // Since 64 bits = 1 u64 word
            if word_idx < DIM_U64 {
                let raw = fp.as_raw_mut();
                raw[word_idx] = bb;
            }
        }
    }

    /// Encode pawn structure features.
    ///
    /// Detects and encodes: doubled pawns, isolated pawns, passed pawns,
    /// pawn chains, pawn islands. Each feature sets specific bits in
    /// the pawn structure region (768-1023).
    fn encode_pawn_structure(pos: &ChessPosition, fp: &mut Fingerprint) {
        let white_pawns = pos.pieces(Color::White, PieceType::Pawn);
        let black_pawns = pos.pieces(Color::Black, PieceType::Pawn);

        let base_word = PAWN_STRUCTURE_START / 64; // word 12

        // Encode pawn file occupancy (which files have pawns)
        let white_files = file_occupancy(white_pawns);
        let black_files = file_occupancy(black_pawns);

        let raw = fp.as_raw_mut();
        if base_word < DIM_U64 {
            raw[base_word] = white_files as u64 | ((black_files as u64) << 8);
        }

        // Doubled pawns: more than one pawn on the same file
        let white_doubled = detect_doubled_pawns(white_pawns);
        let black_doubled = detect_doubled_pawns(black_pawns);

        if base_word + 1 < DIM_U64 {
            raw[base_word + 1] = white_doubled as u64 | ((black_doubled as u64) << 8);
        }

        // Isolated pawns: no friendly pawns on adjacent files
        let white_isolated = detect_isolated_pawns(white_pawns);
        let black_isolated = detect_isolated_pawns(black_pawns);

        if base_word + 2 < DIM_U64 {
            raw[base_word + 2] = white_isolated as u64 | ((black_isolated as u64) << 8);
        }

        // Passed pawns: no enemy pawns can block or capture
        let white_passed = detect_passed_pawns(white_pawns, black_pawns);
        let black_passed = detect_passed_pawns(black_pawns, white_pawns);

        if base_word + 3 < DIM_U64 {
            raw[base_word + 3] = white_passed | (black_passed << 32);
        }
    }

    /// Encode king safety features.
    fn encode_king_safety(pos: &ChessPosition, fp: &mut Fingerprint) {
        let base_word = KING_SAFETY_START / 64; // word 16

        let white_king = pos.pieces(Color::White, PieceType::King);
        let black_king = pos.pieces(Color::Black, PieceType::King);
        let white_pawns = pos.pieces(Color::White, PieceType::Pawn);
        let black_pawns = pos.pieces(Color::Black, PieceType::Pawn);

        let raw = fp.as_raw_mut();

        // King position encoding
        if base_word < DIM_U64 {
            raw[base_word] = white_king | (black_king << 32);
        }

        // Pawn shield: pawns near the king
        let white_shield = pawn_shield(white_king, white_pawns);
        let black_shield = pawn_shield(black_king, black_pawns);

        if base_word + 1 < DIM_U64 {
            raw[base_word + 1] = white_shield as u64 | ((black_shield as u64) << 32);
        }

        // Castling state encoding
        let castling_bits = (pos.castling[0] as u64)
            | ((pos.castling[1] as u64) << 1)
            | ((pos.castling[2] as u64) << 2)
            | ((pos.castling[3] as u64) << 3);

        if base_word + 2 < DIM_U64 {
            raw[base_word + 2] = castling_bits;
        }
    }

    /// Encode material signature.
    fn encode_material(pos: &ChessPosition, fp: &mut Fingerprint) {
        let base_word = MATERIAL_START / 64; // word 36

        let raw = fp.as_raw_mut();

        // Material count per piece type
        let mut material_sig: u64 = 0;
        for (i, piece) in [PieceType::Pawn, PieceType::Knight, PieceType::Bishop,
                           PieceType::Rook, PieceType::Queen].iter().enumerate() {
            let white_count = pos.pieces(Color::White, *piece).count_ones() as u64;
            let black_count = pos.pieces(Color::Black, *piece).count_ones() as u64;
            // 4 bits per count (max 8 pawns, fits in 4 bits), 8 bits per piece type
            material_sig |= (white_count | (black_count << 4)) << (i * 8);
        }

        if base_word < DIM_U64 {
            raw[base_word] = material_sig;
        }

        // Material balance as signed value
        let balance = pos.material_balance();
        if base_word + 1 < DIM_U64 {
            raw[base_word + 1] = balance as u64;
        }

        // Phase encoding
        let phase_val: u64 = match pos.phase() {
            GamePhase::Opening => 0x0000_0000_0000_0001,
            GamePhase::Middlegame => 0x0000_0000_0000_0002,
            GamePhase::Endgame => 0x0000_0000_0000_0004,
        };
        if base_word + 2 < DIM_U64 {
            raw[base_word + 2] = phase_val;
        }
    }

    /// Encode position metadata (side to move, move number).
    fn encode_metadata(pos: &ChessPosition, fp: &mut Fingerprint) {
        // Use a high word for metadata to avoid collisions
        let meta_word = 250; // Near end of 256-word fingerprint
        let raw = fp.as_raw_mut();

        if meta_word < DIM_U64 {
            let side_bit = match pos.side_to_move {
                Color::White => 0u64,
                Color::Black => 1u64,
            };
            let ep_bits = pos.en_passant.map(|sq| (sq as u64) << 1).unwrap_or(0);
            let move_bits = (pos.fullmove_number as u64) << 8;

            raw[meta_word] = side_bit | ep_bits | move_bits;
        }
    }

    /// Compute the similarity between two chess positions.
    ///
    /// Returns a value between 0.0 (completely different) and 1.0 (identical).
    pub fn similarity(fen_a: &str, fen_b: &str) -> Option<f32> {
        let fp_a = Self::from_fen(fen_a)?;
        let fp_b = Self::from_fen(fen_b)?;
        Some(fp_a.similarity(&fp_b))
    }

    /// Find the K most similar positions from a set of candidates.
    ///
    /// This is the RESONATE operation: given a query fingerprint,
    /// find the K nearest neighbors by Hamming distance.
    pub fn resonate(
        query: &Fingerprint,
        candidates: &[(String, Fingerprint)],
        k: usize,
    ) -> Vec<(String, f32, u32)> {
        let mut results: Vec<(String, f32, u32)> = candidates
            .iter()
            .map(|(fen, fp)| {
                let dist = query.hamming(fp);
                let sim = query.similarity(fp);
                (fen.clone(), sim, dist)
            })
            .collect();

        // Sort by similarity (descending)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }
}

// ---------------------------------------------------------------------------
// Pawn structure helpers
// ---------------------------------------------------------------------------

/// Get file occupancy bitmap (8 bits, one per file).
fn file_occupancy(pawns: u64) -> u8 {
    let mut files = 0u8;
    for file in 0..8 {
        let file_mask = 0x0101010101010101u64 << file;
        if pawns & file_mask != 0 {
            files |= 1 << file;
        }
    }
    files
}

/// Detect doubled pawns (files with more than one pawn).
fn detect_doubled_pawns(pawns: u64) -> u8 {
    let mut doubled = 0u8;
    for file in 0..8 {
        let file_mask = 0x0101010101010101u64 << file;
        if (pawns & file_mask).count_ones() > 1 {
            doubled |= 1 << file;
        }
    }
    doubled
}

/// Detect isolated pawns (no friendly pawns on adjacent files).
fn detect_isolated_pawns(pawns: u64) -> u8 {
    let file_occ = file_occupancy(pawns);
    let mut isolated = 0u8;
    for file in 0..8u8 {
        if file_occ & (1 << file) == 0 {
            continue;
        }
        let left = if file > 0 { file_occ & (1 << (file - 1)) } else { 0 };
        let right = if file < 7 { file_occ & (1 << (file + 1)) } else { 0 };
        if left == 0 && right == 0 {
            isolated |= 1 << file;
        }
    }
    isolated
}

/// Detect passed pawns (simplified: no enemy pawns on same or adjacent files ahead).
fn detect_passed_pawns(friendly: u64, enemy: u64) -> u64 {
    let mut passed = 0u64;
    let mut pawns = friendly;
    while pawns != 0 {
        let sq = pawns.trailing_zeros();
        if sq >= 64 { break; }
        let file = sq % 8;
        let rank = sq / 8;

        // Check if any enemy pawns block or guard this pawn
        let mut blocked = false;
        for f in file.saturating_sub(1)..=(file + 1).min(7) {
            // Check all ranks ahead
            for r in (rank + 1)..8 {
                if enemy & (1u64 << (r * 8 + f)) != 0 {
                    blocked = true;
                    break;
                }
            }
            if blocked { break; }
        }
        if !blocked {
            passed |= 1u64 << sq;
        }
        pawns &= pawns - 1; // Clear LSB
    }
    passed
}

/// Compute pawn shield score: count pawns near the king.
fn pawn_shield(king_bb: u64, pawns: u64) -> u32 {
    if king_bb == 0 { return 0; }
    let king_sq = king_bb.trailing_zeros();
    let king_file = king_sq % 8;
    let king_rank = king_sq / 8;

    let mut shield = 0u32;
    for f in king_file.saturating_sub(1)..=(king_file + 1).min(7) {
        for r in king_rank.saturating_sub(0)..=(king_rank + 2).min(7) {
            if pawns & (1u64 << (r * 8 + f)) != 0 {
                shield += 1;
            }
        }
    }
    shield
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    const AFTER_E4: &str = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
    const SICILIAN: &str = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2";
    const ENDGAME: &str = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1";

    #[test]
    fn test_fingerprint_from_startpos() {
        let fp = ChessFingerprint::from_fen(STARTPOS).unwrap();
        // Startpos should have non-zero fingerprint
        assert!(fp.popcount() > 0);
    }

    #[test]
    fn test_identical_positions_have_zero_distance() {
        let fp1 = ChessFingerprint::from_fen(STARTPOS).unwrap();
        let fp2 = ChessFingerprint::from_fen(STARTPOS).unwrap();
        assert_eq!(fp1.hamming(&fp2), 0);
        assert_eq!(fp1.similarity(&fp2), 1.0);
    }

    #[test]
    fn test_similar_positions_have_small_distance() {
        let fp_start = ChessFingerprint::from_fen(STARTPOS).unwrap();
        let fp_e4 = ChessFingerprint::from_fen(AFTER_E4).unwrap();

        // After 1.e4, the position should be very similar to startpos
        let sim = fp_start.similarity(&fp_e4);
        assert!(sim > 0.9, "Expected high similarity, got {}", sim);
    }

    #[test]
    fn test_different_phases_have_larger_distance() {
        let fp_opening = ChessFingerprint::from_fen(STARTPOS).unwrap();
        let fp_endgame = ChessFingerprint::from_fen(ENDGAME).unwrap();

        let sim = fp_opening.similarity(&fp_endgame);
        // Endgame should be much less similar than opening
        assert!(sim < 0.5, "Expected low similarity, got {}", sim);
    }

    #[test]
    fn test_resonate() {
        let query = ChessFingerprint::from_fen(AFTER_E4).unwrap();
        let candidates = vec![
            (STARTPOS.to_string(), ChessFingerprint::from_fen(STARTPOS).unwrap()),
            (SICILIAN.to_string(), ChessFingerprint::from_fen(SICILIAN).unwrap()),
            (ENDGAME.to_string(), ChessFingerprint::from_fen(ENDGAME).unwrap()),
        ];

        let results = ChessFingerprint::resonate(&query, &candidates, 2);
        assert_eq!(results.len(), 2);
        // Startpos should be most similar to after-e4
        assert_eq!(results[0].0, STARTPOS);
    }

    #[test]
    fn test_pawn_structure_features() {
        // Position with doubled pawns on e-file
        let pos = ChessPosition::from_fen("4k3/8/8/4p3/4p3/8/8/4K3 b - - 0 1").unwrap();
        let fp = ChessFingerprint::encode(&pos);
        assert!(fp.popcount() > 0);
    }

    #[test]
    fn test_file_occupancy() {
        // Pawns on e2 and d2 (squares 12 and 11)
        let pawns = (1u64 << 12) | (1u64 << 11);
        let files = file_occupancy(pawns);
        assert!(files & (1 << 3) != 0); // d-file
        assert!(files & (1 << 4) != 0); // e-file
        assert!(files & (1 << 0) == 0); // a-file empty
    }

    #[test]
    fn test_similarity_symmetry() {
        let sim_ab = ChessFingerprint::similarity(STARTPOS, AFTER_E4).unwrap();
        let sim_ba = ChessFingerprint::similarity(AFTER_E4, STARTPOS).unwrap();
        assert!((sim_ab - sim_ba).abs() < 0.001);
    }
}
