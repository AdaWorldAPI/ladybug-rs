//! Chess position representation for fingerprint encoding.
//!
//! Provides a minimal board representation sufficient for fingerprint
//! generation. This is NOT a full chess engine — for move generation
//! and search, use stonksfish.

use std::fmt;

/// Piece color.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Color {
    White,
    Black,
}

impl Color {
    pub fn opposite(self) -> Self {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }

    pub fn index(self) -> usize {
        match self {
            Color::White => 0,
            Color::Black => 1,
        }
    }
}

/// Chess piece type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PieceType {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

impl PieceType {
    pub fn index(self) -> usize {
        match self {
            PieceType::Pawn => 0,
            PieceType::Knight => 1,
            PieceType::Bishop => 2,
            PieceType::Rook => 3,
            PieceType::Queen => 4,
            PieceType::King => 5,
        }
    }

    /// Material value in centipawns.
    pub fn value(self) -> i32 {
        match self {
            PieceType::Pawn => 100,
            PieceType::Knight => 320,
            PieceType::Bishop => 330,
            PieceType::Rook => 500,
            PieceType::Queen => 900,
            PieceType::King => 0,
        }
    }
}

/// Game phase classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GamePhase {
    Opening,
    Middlegame,
    Endgame,
}

impl fmt::Display for GamePhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GamePhase::Opening => write!(f, "opening"),
            GamePhase::Middlegame => write!(f, "middlegame"),
            GamePhase::Endgame => write!(f, "endgame"),
        }
    }
}

/// A chess position parsed from FEN.
///
/// Contains the 12 bitboards (6 piece types × 2 colors) and metadata
/// needed for fingerprint generation.
#[derive(Debug, Clone)]
pub struct ChessPosition {
    /// Bitboards: [white_pawns, white_knights, ..., black_pawns, ...]
    /// Index: color * 6 + piece_type
    pub bitboards: [u64; 12],
    /// Side to move.
    pub side_to_move: Color,
    /// Castling rights: [white_kingside, white_queenside, black_kingside, black_queenside]
    pub castling: [bool; 4],
    /// En passant target square (0-63), or None.
    pub en_passant: Option<u8>,
    /// Halfmove clock (for 50-move rule).
    pub halfmove_clock: u32,
    /// Fullmove number.
    pub fullmove_number: u32,
}

impl ChessPosition {
    /// Parse a FEN string into a ChessPosition.
    pub fn from_fen(fen: &str) -> Option<Self> {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.is_empty() {
            return None;
        }

        let mut pos = ChessPosition {
            bitboards: [0u64; 12],
            side_to_move: Color::White,
            castling: [false; 4],
            en_passant: None,
            halfmove_clock: 0,
            fullmove_number: 1,
        };

        // Parse piece placement (rank 8 to rank 1)
        let mut rank: i32 = 7;
        let mut file: i32 = 0;

        for ch in parts[0].chars() {
            match ch {
                '/' => {
                    rank -= 1;
                    file = 0;
                }
                '1'..='8' => {
                    file += ch.to_digit(10).unwrap() as i32;
                }
                _ => {
                    let square = (rank * 8 + file) as u8;
                    let (color, piece) = match ch {
                        'P' => (Color::White, PieceType::Pawn),
                        'N' => (Color::White, PieceType::Knight),
                        'B' => (Color::White, PieceType::Bishop),
                        'R' => (Color::White, PieceType::Rook),
                        'Q' => (Color::White, PieceType::Queen),
                        'K' => (Color::White, PieceType::King),
                        'p' => (Color::Black, PieceType::Pawn),
                        'n' => (Color::Black, PieceType::Knight),
                        'b' => (Color::Black, PieceType::Bishop),
                        'r' => (Color::Black, PieceType::Rook),
                        'q' => (Color::Black, PieceType::Queen),
                        'k' => (Color::Black, PieceType::King),
                        _ => return None,
                    };
                    let idx = color.index() * 6 + piece.index();
                    pos.bitboards[idx] |= 1u64 << square;
                    file += 1;
                }
            }
        }

        // Parse side to move
        if parts.len() > 1 {
            pos.side_to_move = match parts[1] {
                "w" => Color::White,
                "b" => Color::Black,
                _ => Color::White,
            };
        }

        // Parse castling
        if parts.len() > 2 {
            for ch in parts[2].chars() {
                match ch {
                    'K' => pos.castling[0] = true,
                    'Q' => pos.castling[1] = true,
                    'k' => pos.castling[2] = true,
                    'q' => pos.castling[3] = true,
                    _ => {}
                }
            }
        }

        // Parse en passant
        if parts.len() > 3 && parts[3] != "-" {
            let ep = parts[3];
            if ep.len() == 2 {
                let file = ep.as_bytes()[0].wrapping_sub(b'a');
                let rank = ep.as_bytes()[1].wrapping_sub(b'1');
                if file < 8 && rank < 8 {
                    pos.en_passant = Some(rank * 8 + file);
                }
            }
        }

        // Parse halfmove clock
        if parts.len() > 4 {
            pos.halfmove_clock = parts[4].parse().unwrap_or(0);
        }

        // Parse fullmove number
        if parts.len() > 5 {
            pos.fullmove_number = parts[5].parse().unwrap_or(1);
        }

        Some(pos)
    }

    /// Total piece count (excluding kings, they're always 2).
    pub fn piece_count(&self) -> u32 {
        self.bitboards.iter().map(|bb| bb.count_ones()).sum()
    }

    /// Classify the game phase based on piece count.
    pub fn phase(&self) -> GamePhase {
        let count = self.piece_count();
        if count <= 10 {
            GamePhase::Endgame
        } else if count <= 24 {
            GamePhase::Middlegame
        } else {
            GamePhase::Opening
        }
    }

    /// Get the bitboard for a specific piece type and color.
    pub fn pieces(&self, color: Color, piece: PieceType) -> u64 {
        self.bitboards[color.index() * 6 + piece.index()]
    }

    /// Get all pieces of a color combined.
    pub fn color_combined(&self, color: Color) -> u64 {
        let base = color.index() * 6;
        self.bitboards[base] | self.bitboards[base + 1] |
        self.bitboards[base + 2] | self.bitboards[base + 3] |
        self.bitboards[base + 4] | self.bitboards[base + 5]
    }

    /// Get all occupied squares.
    pub fn occupied(&self) -> u64 {
        self.color_combined(Color::White) | self.color_combined(Color::Black)
    }

    /// Material balance in centipawns (positive = white advantage).
    pub fn material_balance(&self) -> i32 {
        let mut balance = 0i32;
        for piece in [PieceType::Pawn, PieceType::Knight, PieceType::Bishop,
                      PieceType::Rook, PieceType::Queen] {
            let white_count = self.pieces(Color::White, piece).count_ones() as i32;
            let black_count = self.pieces(Color::Black, piece).count_ones() as i32;
            balance += (white_count - black_count) * piece.value();
        }
        balance
    }
}

impl fmt::Display for ChessPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for rank in (0..8).rev() {
            for file in 0..8 {
                let sq = rank * 8 + file;
                let bit = 1u64 << sq;
                let mut found = false;
                let pieces = [
                    ('P', Color::White, PieceType::Pawn),
                    ('N', Color::White, PieceType::Knight),
                    ('B', Color::White, PieceType::Bishop),
                    ('R', Color::White, PieceType::Rook),
                    ('Q', Color::White, PieceType::Queen),
                    ('K', Color::White, PieceType::King),
                    ('p', Color::Black, PieceType::Pawn),
                    ('n', Color::Black, PieceType::Knight),
                    ('b', Color::Black, PieceType::Bishop),
                    ('r', Color::Black, PieceType::Rook),
                    ('q', Color::Black, PieceType::Queen),
                    ('k', Color::Black, PieceType::King),
                ];
                for (ch, color, piece) in &pieces {
                    if self.pieces(*color, *piece) & bit != 0 {
                        write!(f, "{}", ch)?;
                        found = true;
                        break;
                    }
                }
                if !found {
                    write!(f, ".")?;
                }
            }
            if rank > 0 {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    const AFTER_E4: &str = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";

    #[test]
    fn test_parse_startpos() {
        let pos = ChessPosition::from_fen(STARTPOS).unwrap();
        assert_eq!(pos.piece_count(), 32);
        assert_eq!(pos.side_to_move, Color::White);
        assert_eq!(pos.phase(), GamePhase::Opening);
        assert!(pos.castling[0]); // White kingside
        assert!(pos.castling[1]); // White queenside
        assert!(pos.castling[2]); // Black kingside
        assert!(pos.castling[3]); // Black queenside
        assert_eq!(pos.material_balance(), 0);
    }

    #[test]
    fn test_parse_after_e4() {
        let pos = ChessPosition::from_fen(AFTER_E4).unwrap();
        assert_eq!(pos.side_to_move, Color::Black);
        assert!(pos.en_passant.is_some());
        assert_eq!(pos.material_balance(), 0);
    }

    #[test]
    fn test_endgame_detection() {
        // King + Rook vs King
        let pos = ChessPosition::from_fen("4k3/8/8/8/8/8/8/4K2R w K - 0 1").unwrap();
        assert_eq!(pos.phase(), GamePhase::Endgame);
    }

    #[test]
    fn test_piece_counts() {
        let pos = ChessPosition::from_fen(STARTPOS).unwrap();
        assert_eq!(pos.pieces(Color::White, PieceType::Pawn).count_ones(), 8);
        assert_eq!(pos.pieces(Color::Black, PieceType::Pawn).count_ones(), 8);
        assert_eq!(pos.pieces(Color::White, PieceType::Knight).count_ones(), 2);
        assert_eq!(pos.pieces(Color::White, PieceType::King).count_ones(), 1);
    }

    #[test]
    fn test_display() {
        let pos = ChessPosition::from_fen(STARTPOS).unwrap();
        let display = format!("{}", pos);
        assert!(display.contains("rnbqkbnr"));
        assert!(display.contains("RNBQKBNR"));
    }
}
