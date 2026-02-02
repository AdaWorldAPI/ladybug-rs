//! MCP Actions for Arrow Flight
//!
//! Implements the MCP tool interface via Flight DoAction.

use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::storage::BindSpace;
use crate::search::HdrIndex;
use crate::storage::bind_space::{Addr, FINGERPRINT_WORDS};

/// MCP Action types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum McpAction {
    /// Encode text/data to fingerprint
    Encode {
        text: Option<String>,
        data: Option<Vec<u8>>,
        style: Option<String>, // creative, balanced, precise
    },
    /// Bind fingerprint to address
    Bind {
        address: u16,
        fingerprint: Vec<u8>,
        label: Option<String>,
    },
    /// Read from address
    Read {
        address: u16,
    },
    /// Find similar fingerprints
    Resonate {
        query: Vec<u8>,
        k: usize,
        threshold: Option<u32>,
    },
    /// Compute Hamming distance
    Hamming {
        a: Vec<u8>,
        b: Vec<u8>,
    },
    /// XOR bind two fingerprints
    XorBind {
        a: Vec<u8>,
        b: Vec<u8>,
    },
    /// Get BindSpace statistics
    Stats,
}

/// MCP Action result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum McpResult {
    /// Encoded fingerprint
    Fingerprint {
        fingerprint: Vec<u8>,
        bits_set: u32,
        encoding_style: String,
    },
    /// Bind confirmation
    Bound {
        address: u16,
        success: bool,
    },
    /// Read result
    Node {
        address: u16,
        fingerprint: Vec<u8>,
        label: Option<String>,
        zone: String,
    },
    /// Search results
    Matches {
        results: Vec<MatchResult>,
        query_time_ns: u64,
        cascade_stats: CascadeStats,
    },
    /// Hamming distance
    Distance {
        distance: u32,
        similarity: f32,
        max_bits: u32,
    },
    /// XOR result
    Combined {
        fingerprint: Vec<u8>,
        bits_set: u32,
    },
    /// Statistics
    Stats {
        total_nodes: usize,
        surface_nodes: usize,
        fluid_nodes: usize,
        node_space_nodes: usize,
    },
    /// Error
    Error {
        message: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchResult {
    pub address: u16,
    pub fingerprint: Vec<u8>,
    pub label: Option<String>,
    pub distance: u32,
    pub similarity: f32,
    pub cascade_level: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadeStats {
    pub l0_candidates: usize,
    pub l1_candidates: usize,
    pub l2_candidates: usize,
    pub final_candidates: usize,
}

/// Execute an MCP action
pub async fn execute_action(
    action_type: &str,
    body: &[u8],
    bind_space: Arc<RwLock<BindSpace>>,
    hdr_cascade: Arc<RwLock<HdrIndex>>,
) -> Result<Vec<u8>, String> {
    // Parse JSON body into McpAction based on action_type
    let action: McpAction = match action_type {
        "encode" => {
            let params: serde_json::Value = serde_json::from_slice(body)
                .map_err(|e| e.to_string())?;
            McpAction::Encode {
                text: params.get("text").and_then(|v| v.as_str()).map(String::from),
                data: params.get("data").and_then(|v| {
                    v.as_str().and_then(|s| hex::decode(s).ok())
                }),
                style: params.get("style").and_then(|v| v.as_str()).map(String::from),
            }
        }
        "bind" => {
            let params: serde_json::Value = serde_json::from_slice(body)
                .map_err(|e| e.to_string())?;
            McpAction::Bind {
                address: params.get("address")
                    .and_then(|v| v.as_u64())
                    .ok_or("missing address")? as u16,
                fingerprint: params.get("fingerprint")
                    .and_then(|v| v.as_str())
                    .and_then(|s| hex::decode(s).ok())
                    .ok_or("missing fingerprint")?,
                label: params.get("label").and_then(|v| v.as_str()).map(String::from),
            }
        }
        "read" => {
            let params: serde_json::Value = serde_json::from_slice(body)
                .map_err(|e| e.to_string())?;
            McpAction::Read {
                address: params.get("address")
                    .and_then(|v| v.as_u64())
                    .ok_or("missing address")? as u16,
            }
        }
        "resonate" => {
            let params: serde_json::Value = serde_json::from_slice(body)
                .map_err(|e| e.to_string())?;
            McpAction::Resonate {
                query: params.get("query")
                    .and_then(|v| v.as_str())
                    .and_then(|s| hex::decode(s).ok())
                    .ok_or("missing query")?,
                k: params.get("k")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(10) as usize,
                threshold: params.get("threshold")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32),
            }
        }
        "hamming" => {
            let params: serde_json::Value = serde_json::from_slice(body)
                .map_err(|e| e.to_string())?;
            McpAction::Hamming {
                a: params.get("a")
                    .and_then(|v| v.as_str())
                    .and_then(|s| hex::decode(s).ok())
                    .ok_or("missing a")?,
                b: params.get("b")
                    .and_then(|v| v.as_str())
                    .and_then(|s| hex::decode(s).ok())
                    .ok_or("missing b")?,
            }
        }
        "xor_bind" => {
            let params: serde_json::Value = serde_json::from_slice(body)
                .map_err(|e| e.to_string())?;
            McpAction::XorBind {
                a: params.get("a")
                    .and_then(|v| v.as_str())
                    .and_then(|s| hex::decode(s).ok())
                    .ok_or("missing a")?,
                b: params.get("b")
                    .and_then(|v| v.as_str())
                    .and_then(|s| hex::decode(s).ok())
                    .ok_or("missing b")?,
            }
        }
        "stats" => McpAction::Stats,
        _ => return Err(format!("Unknown action: {}", action_type)),
    };

    // Execute the action
    let result = execute_mcp_action(action, bind_space, hdr_cascade).await?;

    // Serialize result to JSON
    serde_json::to_vec(&result).map_err(|e| e.to_string())
}

async fn execute_mcp_action(
    action: McpAction,
    bind_space: Arc<RwLock<BindSpace>>,
    _hdr_cascade: Arc<RwLock<HdrIndex>>,
) -> Result<McpResult, String> {
    match action {
        McpAction::Encode { text, data, style } => {
            // Use Sigma-10 membrane encoding
            let input = if let Some(t) = text {
                t.into_bytes()
            } else if let Some(d) = data {
                d
            } else {
                return Err("Either text or data required".to_string());
            };

            // TODO: Call actual Sigma-10 membrane encoder
            // For now, simple hash-based fingerprint
            use sha2::{Sha256, Digest};
            let mut hasher = Sha256::new();
            hasher.update(&input);
            let hash = hasher.finalize();

            // Expand to full fingerprint size (1248 bytes = 156 * 8)
            let mut fingerprint = vec![0u8; 1248];
            for (i, chunk) in fingerprint.chunks_mut(32).enumerate() {
                let mut h = Sha256::new();
                h.update(&hash);
                h.update(&[i as u8]);
                chunk.copy_from_slice(&h.finalize()[..chunk.len().min(32)]);
            }

            let bits_set: u32 = fingerprint.iter()
                .map(|b| b.count_ones())
                .sum();

            Ok(McpResult::Fingerprint {
                fingerprint,
                bits_set,
                encoding_style: style.unwrap_or_else(|| "balanced".to_string()),
            })
        }

        McpAction::Bind { address, fingerprint, label } => {
            let addr = Addr(address);

            // Convert bytes to [u64; FINGERPRINT_WORDS] (156 words = 1248 bytes)
            let mut fp_array = [0u64; FINGERPRINT_WORDS];
            for (i, chunk) in fingerprint.chunks(8).enumerate() {
                if i >= FINGERPRINT_WORDS {
                    break;
                }
                if chunk.len() == 8 {
                    fp_array[i] = u64::from_le_bytes(chunk.try_into().unwrap());
                } else {
                    // Partial chunk - pad with zeros
                    let mut buf = [0u8; 8];
                    buf[..chunk.len()].copy_from_slice(chunk);
                    fp_array[i] = u64::from_le_bytes(buf);
                }
            }

            let mut space = bind_space.write();
            let success = space.write_at(addr, fp_array);

            // Set label if provided and write succeeded
            if success {
                if let Some(lbl) = label {
                    if let Some(node) = space.read_mut(addr) {
                        node.label = Some(lbl);
                    }
                }
            }

            Ok(McpResult::Bound { address, success })
        }

        McpAction::Read { address } => {
            let addr = Addr(address);
            let space = bind_space.read();

            if let Some(node) = space.read(addr) {
                let fingerprint: Vec<u8> = node.fingerprint
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect();

                let zone = match addr.prefix() {
                    0x00..=0x0F => "surface",
                    0x10..=0x7F => "fluid",
                    0x80..=0xFF => "node",
                }.to_string();

                Ok(McpResult::Node {
                    address,
                    fingerprint,
                    label: node.label.clone(),
                    zone,
                })
            } else {
                Ok(McpResult::Error {
                    message: format!("No node at address {:#06x}", address),
                })
            }
        }

        McpAction::Resonate { query, k, threshold } => {
            // TODO: Implement HDR cascade search
            let start = std::time::Instant::now();

            // Placeholder results
            let results = vec![];
            let cascade_stats = CascadeStats {
                l0_candidates: 0,
                l1_candidates: 0,
                l2_candidates: 0,
                final_candidates: 0,
            };

            Ok(McpResult::Matches {
                results,
                query_time_ns: start.elapsed().as_nanos() as u64,
                cascade_stats,
            })
        }

        McpAction::Hamming { a, b } => {
            let max_len = a.len().min(b.len());
            let distance: u32 = a.iter().zip(b.iter())
                .map(|(x, y)| (x ^ y).count_ones())
                .sum();

            let max_bits = (max_len * 8) as u32;
            let similarity = if max_bits > 0 {
                1.0 - (distance as f32 / max_bits as f32)
            } else {
                0.0
            };

            Ok(McpResult::Distance {
                distance,
                similarity,
                max_bits,
            })
        }

        McpAction::XorBind { a, b } => {
            let fingerprint: Vec<u8> = a.iter().zip(b.iter())
                .map(|(x, y)| x ^ y)
                .collect();

            let bits_set: u32 = fingerprint.iter()
                .map(|b| b.count_ones())
                .sum();

            Ok(McpResult::Combined {
                fingerprint,
                bits_set,
            })
        }

        McpAction::Stats => {
            let space = bind_space.read();
            let stats = space.stats();

            Ok(McpResult::Stats {
                total_nodes: stats.surface_count + stats.fluid_count + stats.node_count,
                surface_nodes: stats.surface_count,
                fluid_nodes: stats.fluid_count,
                node_space_nodes: stats.node_count,
            })
        }
    }
}
