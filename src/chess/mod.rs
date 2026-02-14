//! Chess position fingerprinting for ladybug-rs.
//!
//! Encodes chess positions as 16,384-bit fingerprints enabling
//! Hamming-distance-based similarity search via HDR cascade.
//!
//! # Fingerprint Layout (16,384 bits)
//!
//! ```text
//! Bits 0-767:       Raw Bitboards (12 piece types × 64 squares)
//! Bits 768-1023:    Pawn Structure (doubled, isolated, passed, chains)
//! Bits 1024-1279:   King Safety (pawn shield, open files, tropism)
//! Bits 1280-1791:   Piece Activity (mobility, centralization, coordination)
//! Bits 1792-2303:   Tactical Motifs (pins, forks, skewers, discovered attacks)
//! Bits 2304-2559:   Material Signature (balance, phase, endgame type)
//! Bits 2560-4095:   Strategic Themes (space, pawn majorities, color complex)
//! Bits 4096-8191:   Opening/Plan Context (ECO, move history, agent notes)
//! Bits 8192-16383:  AI War Cognitive Bridge (cross-domain resonance)
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use ladybug::chess::ChessFingerprint;
//!
//! let fp = ChessFingerprint::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
//! let similar = fp.resonate(&database, 10);
//! ```

pub mod fingerprint;
pub mod position;

pub use fingerprint::ChessFingerprint;
pub use position::{ChessPosition, GamePhase, PieceType, Color};
