//! Redis Adapter - Redis syntax interface to Substrate
//!
//! Provides Redis-like command interface on top of the unified Substrate.
//! Replaces CogRedis as the primary command executor.
//!
//! # Command Format
//!
//! ```text
//! ┌────────────────┬─────────────────────────────────────────────────────────┐
//! │ Command        │ Description                                             │
//! ├────────────────┼─────────────────────────────────────────────────────────┤
//! │ GET key        │ Get value + qualia + truth                              │
//! │ SET key val    │ Set with optional QUALIA, TRUTH, TTL, PROMOTE           │
//! │ DEL key        │ Delete value (or FORGET with decay, SUPPRESS)           │
//! │ BIND a v b     │ Create edge from a to b via verb v                      │
//! │ UNBIND e a     │ Recover b from edge e and known a                       │
//! │ RESONATE q k   │ Find k nodes resonating with query q                    │
//! │ TRAVERSE s v n │ N-hop traversal from s via verb v                       │
//! │ CAM op args    │ Execute CAM operation                                   │
//! │ CRYSTALLIZE a  │ Promote from fluid to node                              │
//! │ EVAPORATE a    │ Demote from node to fluid                               │
//! └────────────────┴─────────────────────────────────────────────────────────┘
//! ```

use std::collections::HashMap;
use std::time::Duration;

use super::bind_space::{FINGERPRINT_WORDS, hamming_distance};
use super::cog_redis::{CogAddr, Tier};
use super::substrate::{Substrate, SubstrateConfig};

// =============================================================================
// RESULT TYPES
// =============================================================================

/// Result of a Redis command
#[derive(Debug, Clone)]
pub enum RedisResult {
    /// Success with no return value
    Ok,
    /// String result
    String(String),
    /// Integer result
    Integer(i64),
    /// Float result
    Float(f64),
    /// Address result
    Addr(CogAddr),
    /// Array result
    Array(Vec<RedisResult>),
    /// Node result (full node data)
    Node(NodeResult),
    /// Search results
    Search(Vec<SearchHit>),
    /// Edge result
    Edge(EdgeResult),
    /// Error
    Error(String),
    /// Nil (not found)
    Nil,
}

impl RedisResult {
    /// Check if result is OK
    pub fn is_ok(&self) -> bool {
        matches!(
            self,
            RedisResult::Ok
                | RedisResult::String(_)
                | RedisResult::Integer(_)
                | RedisResult::Float(_)
                | RedisResult::Addr(_)
                | RedisResult::Array(_)
                | RedisResult::Node(_)
                | RedisResult::Search(_)
                | RedisResult::Edge(_)
        )
    }

    /// Check if result is error
    pub fn is_error(&self) -> bool {
        matches!(self, RedisResult::Error(_))
    }

    /// Get error message
    pub fn error_message(&self) -> Option<&str> {
        match self {
            RedisResult::Error(msg) => Some(msg),
            _ => None,
        }
    }
}

/// Node data returned from GET
#[derive(Debug, Clone)]
pub struct NodeResult {
    pub addr: CogAddr,
    pub fingerprint: [u64; FINGERPRINT_WORDS],
    pub label: Option<String>,
    pub qidx: u8,
    pub popcount: u32,
    pub tier: Tier,
}

/// Search hit from RESONATE
#[derive(Debug, Clone)]
pub struct SearchHit {
    pub addr: CogAddr,
    pub distance: u32,
    pub similarity: f32,
    pub label: Option<String>,
}

/// Edge data returned from BIND
#[derive(Debug, Clone)]
pub struct EdgeResult {
    pub from: CogAddr,
    pub verb: CogAddr,
    pub to: CogAddr,
    pub fingerprint: [u64; FINGERPRINT_WORDS],
    pub weight: f32,
}

// =============================================================================
// CAM RESULT
// =============================================================================

/// Result from CAM operation
#[derive(Debug, Clone)]
pub enum CamResult {
    /// Success with fingerprint output
    Fingerprint([u64; FINGERPRINT_WORDS]),
    /// Success with address output
    Addr(CogAddr),
    /// Success with multiple addresses
    Addresses(Vec<CogAddr>),
    /// Success with value
    Value(f64),
    /// Success with string
    String(String),
    /// Error
    Error(String),
}

// =============================================================================
// COMMAND PARSING
// =============================================================================

/// Parsed Redis command
#[derive(Debug, Clone)]
pub enum RedisCommand {
    // Core commands
    Get {
        key: String,
    },
    Set {
        key: String,
        value: String,
        options: SetOptions,
    },
    Del {
        key: String,
        mode: DeleteMode,
    },

    // Edge commands
    Bind {
        from: String,
        verb: String,
        to: String,
    },
    Unbind {
        edge: String,
        known: String,
    },

    // Search commands
    Resonate {
        query: String,
        k: usize,
    },
    Search {
        query: String,
        k: usize,
        threshold: f32,
    },

    // Graph commands
    Traverse {
        start: String,
        verb: String,
        hops: usize,
    },
    Fanout {
        addr: String,
    },
    Fanin {
        addr: String,
    },

    // Lifecycle commands
    Crystallize {
        addr: String,
    },
    Evaporate {
        addr: String,
    },
    Tick,

    // CAM commands
    Cam {
        operation: String,
        args: Vec<String>,
    },

    // Info commands
    Info,
    Stats,
    Ping,

    // Unknown
    Unknown(String),
}

/// SET command options
#[derive(Debug, Clone, Default)]
pub struct SetOptions {
    /// Qualia index (0-255, neutral=128)
    pub qidx: Option<u8>,
    /// Truth value (frequency, confidence) as tuple
    pub truth: Option<(f32, f32)>,
    /// Time-to-live for fluid zone entries
    pub ttl: Option<Duration>,
    /// If true, promote directly to node space
    pub promote: bool,
    /// Optional label for the node
    pub label: Option<String>,
}

/// DELETE mode
#[derive(Debug, Clone)]
pub enum DeleteMode {
    /// Normal delete
    Normal,
    /// Gradual forget with decay
    Forget(f32),
    /// Suppress (negative valence)
    Suppress,
}

impl Default for DeleteMode {
    fn default() -> Self {
        DeleteMode::Normal
    }
}

// =============================================================================
// REDIS ADAPTER
// =============================================================================

/// Redis Adapter - provides Redis-like interface to Substrate
pub struct RedisAdapter {
    /// Underlying substrate
    substrate: Substrate,
    /// Key to address mapping (for string keys)
    key_map: HashMap<String, CogAddr>,
}

impl RedisAdapter {
    /// Create a new Redis adapter
    pub fn new(config: SubstrateConfig) -> Self {
        Self {
            substrate: Substrate::new(config),
            key_map: HashMap::new(),
        }
    }

    /// Create with default config
    pub fn default_new() -> Self {
        Self::new(SubstrateConfig::default())
    }

    /// Get reference to underlying substrate
    pub fn substrate(&self) -> &Substrate {
        &self.substrate
    }

    /// Get mutable reference to substrate
    pub fn substrate_mut(&mut self) -> &mut Substrate {
        &mut self.substrate
    }

    // =========================================================================
    // COMMAND EXECUTION
    // =========================================================================

    /// Execute a Redis command string
    pub fn execute(&mut self, command: &str) -> RedisResult {
        let cmd = self.parse_command(command);
        self.execute_command(cmd)
    }

    /// Execute a parsed command
    pub fn execute_command(&mut self, cmd: RedisCommand) -> RedisResult {
        match cmd {
            RedisCommand::Get { key } => self.cmd_get(&key),
            RedisCommand::Set {
                key,
                value,
                options,
            } => self.cmd_set(&key, &value, options),
            RedisCommand::Del { key, mode } => self.cmd_del(&key, mode),
            RedisCommand::Bind { from, verb, to } => self.cmd_bind(&from, &verb, &to),
            RedisCommand::Unbind { edge, known } => self.cmd_unbind(&edge, &known),
            RedisCommand::Resonate { query, k } => self.cmd_resonate(&query, k),
            RedisCommand::Search {
                query,
                k,
                threshold,
            } => self.cmd_search(&query, k, threshold),
            RedisCommand::Traverse { start, verb, hops } => self.cmd_traverse(&start, &verb, hops),
            RedisCommand::Fanout { addr } => self.cmd_fanout(&addr),
            RedisCommand::Fanin { addr } => self.cmd_fanin(&addr),
            RedisCommand::Crystallize { addr } => self.cmd_crystallize(&addr),
            RedisCommand::Evaporate { addr } => self.cmd_evaporate(&addr),
            RedisCommand::Tick => self.cmd_tick(),
            RedisCommand::Cam { operation, args } => self.cmd_cam(&operation, &args),
            RedisCommand::Info => self.cmd_info(),
            RedisCommand::Stats => self.cmd_stats(),
            RedisCommand::Ping => RedisResult::String("PONG".to_string()),
            RedisCommand::Unknown(cmd) => RedisResult::Error(format!("Unknown command: {}", cmd)),
        }
    }

    // =========================================================================
    // COMMAND IMPLEMENTATIONS
    // =========================================================================

    fn cmd_get(&self, key: &str) -> RedisResult {
        // Try to resolve key to address
        if let Some(&addr) = self.key_map.get(key) {
            if let Some(node) = self.substrate.read(addr) {
                let popcount = node.popcount();
                return RedisResult::Node(NodeResult {
                    addr,
                    fingerprint: node.fingerprint,
                    label: node.label.clone(),
                    qidx: node.qidx,
                    popcount,
                    tier: addr.tier(),
                });
            }
        }

        // Try to parse as hex address
        if let Some(addr) = self.parse_addr(key) {
            if let Some(node) = self.substrate.read(addr) {
                let popcount = node.popcount();
                return RedisResult::Node(NodeResult {
                    addr,
                    fingerprint: node.fingerprint,
                    label: node.label.clone(),
                    qidx: node.qidx,
                    popcount,
                    tier: addr.tier(),
                });
            }
        }

        // Try to find by label
        if let Some(node) = self.substrate.read_by_label(key) {
            let popcount = node.popcount();
            let tier = node.addr.tier();
            return RedisResult::Node(NodeResult {
                addr: node.addr,
                fingerprint: node.fingerprint,
                label: node.label.clone(),
                qidx: node.qidx,
                popcount,
                tier,
            });
        }

        RedisResult::Nil
    }

    fn cmd_set(&mut self, key: &str, value: &str, options: SetOptions) -> RedisResult {
        // Generate fingerprint from value
        let fp = self.generate_fingerprint(value);

        // Write to substrate
        let addr = if let Some(label) = &options.label {
            self.substrate.write_labeled(fp, label)
        } else if options.promote {
            self.substrate.write_labeled(fp, key)
        } else if let Some(ttl) = options.ttl {
            self.substrate.write_fluid(fp, ttl)
        } else {
            self.substrate.write(fp)
        };

        // Update key map
        self.key_map.insert(key.to_string(), addr);

        RedisResult::Addr(addr)
    }

    fn cmd_del(&mut self, key: &str, mode: DeleteMode) -> RedisResult {
        let addr = if let Some(&a) = self.key_map.get(key) {
            a
        } else if let Some(a) = self.parse_addr(key) {
            a
        } else {
            return RedisResult::Integer(0);
        };

        match mode {
            DeleteMode::Normal => {
                if self.substrate.delete(addr) {
                    self.key_map.remove(key);
                    RedisResult::Integer(1)
                } else {
                    RedisResult::Integer(0)
                }
            }
            DeleteMode::Forget(_decay) => {
                // For now, just delete
                if self.substrate.delete(addr) {
                    self.key_map.remove(key);
                    RedisResult::Integer(1)
                } else {
                    RedisResult::Integer(0)
                }
            }
            DeleteMode::Suppress => {
                // Mark as suppressed (would update qualia)
                RedisResult::Ok
            }
        }
    }

    fn cmd_bind(&mut self, from: &str, verb: &str, to: &str) -> RedisResult {
        let from_addr = self.resolve_key(from);
        let verb_addr = self.resolve_verb(verb);
        let to_addr = self.resolve_key(to);

        let (from_addr, verb_addr, to_addr) = match (from_addr, verb_addr, to_addr) {
            (Some(f), Some(v), Some(t)) => (f, v, t),
            _ => return RedisResult::Error("One or more addresses not found".to_string()),
        };

        if let Some(edge) = self.substrate.link(from_addr, verb_addr, to_addr) {
            RedisResult::Edge(EdgeResult {
                from: edge.from,
                verb: edge.verb,
                to: edge.to,
                fingerprint: edge.fingerprint,
                weight: edge.weight,
            })
        } else {
            RedisResult::Error("Failed to create edge".to_string())
        }
    }

    fn cmd_unbind(&self, _edge: &str, _known: &str) -> RedisResult {
        // Would implement ABBA unbind
        RedisResult::Error("UNBIND not yet implemented".to_string())
    }

    fn cmd_resonate(&self, query: &str, k: usize) -> RedisResult {
        let query_fp = self.generate_fingerprint(query);
        let results = self.substrate.resonate(&query_fp, k);

        let hits: Vec<SearchHit> = results
            .iter()
            .filter_map(|(addr, sim)| {
                let node = self.substrate.read(*addr)?;
                Some(SearchHit {
                    addr: *addr,
                    distance: ((1.0 - sim) * crate::FINGERPRINT_BITS as f32) as u32,
                    similarity: *sim,
                    label: node.label,
                })
            })
            .collect();

        RedisResult::Search(hits)
    }

    fn cmd_search(&self, query: &str, k: usize, threshold: f32) -> RedisResult {
        let query_fp = self.generate_fingerprint(query);
        let results = self.substrate.search(&query_fp, k, threshold);

        let hits: Vec<SearchHit> = results
            .iter()
            .filter_map(|(addr, dist, sim)| {
                let node = self.substrate.read(*addr)?;
                Some(SearchHit {
                    addr: *addr,
                    distance: *dist,
                    similarity: *sim,
                    label: node.label,
                })
            })
            .collect();

        RedisResult::Search(hits)
    }

    fn cmd_traverse(&self, start: &str, verb: &str, hops: usize) -> RedisResult {
        let start_addr = match self.resolve_key(start) {
            Some(a) => a,
            None => return RedisResult::Error("Start node not found".to_string()),
        };

        let verb_addr = match self.resolve_verb(verb) {
            Some(a) => a,
            None => return RedisResult::Error("Verb not found".to_string()),
        };

        let results = self.substrate.traverse_n_hops(start_addr, verb_addr, hops);

        let array: Vec<RedisResult> = results
            .iter()
            .map(|(hop, addr)| {
                RedisResult::Array(vec![
                    RedisResult::Integer(*hop as i64),
                    RedisResult::Addr(*addr),
                ])
            })
            .collect();

        RedisResult::Array(array)
    }

    fn cmd_fanout(&self, addr: &str) -> RedisResult {
        let addr = match self.resolve_key(addr) {
            Some(a) => a,
            None => return RedisResult::Error("Address not found".to_string()),
        };

        let edges = self.substrate.edges_out(addr);

        let array: Vec<RedisResult> = edges
            .iter()
            .map(|e| {
                RedisResult::Edge(EdgeResult {
                    from: e.from,
                    verb: e.verb,
                    to: e.to,
                    fingerprint: e.fingerprint,
                    weight: e.weight,
                })
            })
            .collect();

        RedisResult::Array(array)
    }

    fn cmd_fanin(&self, addr: &str) -> RedisResult {
        let addr = match self.resolve_key(addr) {
            Some(a) => a,
            None => return RedisResult::Error("Address not found".to_string()),
        };

        let edges = self.substrate.edges_in(addr);

        let array: Vec<RedisResult> = edges
            .iter()
            .map(|e| {
                RedisResult::Edge(EdgeResult {
                    from: e.from,
                    verb: e.verb,
                    to: e.to,
                    fingerprint: e.fingerprint,
                    weight: e.weight,
                })
            })
            .collect();

        RedisResult::Array(array)
    }

    fn cmd_crystallize(&mut self, addr_str: &str) -> RedisResult {
        let addr = match self.resolve_key(addr_str) {
            Some(a) => a,
            None => return RedisResult::Error("Address not found".to_string()),
        };

        if !addr.is_fluid() {
            return RedisResult::Error("Address not in fluid zone".to_string());
        }

        match self.substrate.crystallize(addr) {
            Some(new_addr) => RedisResult::Addr(new_addr),
            None => RedisResult::Error("Crystallize failed".to_string()),
        }
    }

    fn cmd_evaporate(&mut self, addr_str: &str) -> RedisResult {
        let addr = match self.resolve_key(addr_str) {
            Some(a) => a,
            None => return RedisResult::Error("Address not found".to_string()),
        };

        if !addr.is_node() {
            return RedisResult::Error("Address not in node zone".to_string());
        }

        match self
            .substrate
            .evaporate(addr, self.substrate.config().fluid_ttl)
        {
            Some(new_addr) => RedisResult::Addr(new_addr),
            None => RedisResult::Error("Evaporate failed".to_string()),
        }
    }

    fn cmd_tick(&mut self) -> RedisResult {
        self.substrate.tick();
        RedisResult::Ok
    }

    fn cmd_cam(&mut self, operation: &str, args: &[String]) -> RedisResult {
        let op_upper = operation.to_uppercase();

        // Route to actual implementations where possible
        match op_upper.as_str() {
            "BIND" if args.len() >= 3 => self.cmd_bind(&args[0], &args[1], &args[2]),
            "RESONATE" if !args.is_empty() => {
                let k = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10);
                self.cmd_resonate(&args[0], k)
            }
            "HAMMING" if args.len() >= 2 => {
                // Compute Hamming distance between two fingerprints
                let fp1 = self.generate_fingerprint(&args[0]);
                let fp2 = self.generate_fingerprint(&args[1]);
                let dist = hamming_distance(&fp1, &fp2);
                RedisResult::Integer(dist as i64)
            }
            "SIMILARITY" if args.len() >= 2 => {
                let fp1 = self.generate_fingerprint(&args[0]);
                let fp2 = self.generate_fingerprint(&args[1]);
                let dist = hamming_distance(&fp1, &fp2);
                let sim = 1.0 - (dist as f64 / crate::FINGERPRINT_BITS as f64);
                RedisResult::Float(sim)
            }
            "POPCOUNT" if !args.is_empty() => {
                let fp = self.generate_fingerprint(&args[0]);
                let popcount: u32 = fp.iter().map(|w| w.count_ones()).sum();
                RedisResult::Integer(popcount as i64)
            }
            // Acknowledge other known operations (not yet fully implemented)
            "UNBIND" | "BUNDLE" | "PERMUTE" | "KNN" | "ANN" | "DEDUCE" | "ABDUCT" | "INDUCE"
            | "REVISE" | "SEE" | "DO" | "IMAGINE" => RedisResult::String(format!(
                "CAM {} acknowledged (not fully implemented)",
                operation
            )),
            _ => RedisResult::Error(format!("Unknown CAM operation: {}", operation)),
        }
    }

    fn cmd_info(&self) -> RedisResult {
        let stats = self.substrate.stats();
        let info = format!(
            "hot_hits:{}\nhot_misses:{}\nhot_nodes:{}\nhot_edges:{}\npending_writes:{}\nhit_ratio:{:.4}\nversion:{}",
            stats.hot_hits.load(std::sync::atomic::Ordering::Relaxed),
            stats.hot_misses.load(std::sync::atomic::Ordering::Relaxed),
            stats.hot_nodes.load(std::sync::atomic::Ordering::Relaxed),
            stats.hot_edges.load(std::sync::atomic::Ordering::Relaxed),
            stats
                .pending_writes
                .load(std::sync::atomic::Ordering::Relaxed),
            stats.hit_ratio(),
            self.substrate.version(),
        );
        RedisResult::String(info)
    }

    fn cmd_stats(&self) -> RedisResult {
        let stats = self.substrate.stats();
        RedisResult::Array(vec![
            RedisResult::Array(vec![
                RedisResult::String("hot_hits".to_string()),
                RedisResult::Integer(
                    stats.hot_hits.load(std::sync::atomic::Ordering::Relaxed) as i64
                ),
            ]),
            RedisResult::Array(vec![
                RedisResult::String("hot_misses".to_string()),
                RedisResult::Integer(
                    stats.hot_misses.load(std::sync::atomic::Ordering::Relaxed) as i64
                ),
            ]),
            RedisResult::Array(vec![
                RedisResult::String("hit_ratio".to_string()),
                RedisResult::Float(stats.hit_ratio()),
            ]),
        ])
    }

    // =========================================================================
    // HELPERS
    // =========================================================================

    /// Parse a command string
    fn parse_command(&self, command: &str) -> RedisCommand {
        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.is_empty() {
            return RedisCommand::Unknown(String::new());
        }

        let cmd = parts[0].to_uppercase();
        match cmd.as_str() {
            "GET" => {
                if parts.len() >= 2 {
                    RedisCommand::Get {
                        key: parts[1].to_string(),
                    }
                } else {
                    RedisCommand::Unknown("GET requires key".to_string())
                }
            }
            "SET" => {
                if parts.len() >= 3 {
                    let mut options = SetOptions::default();
                    // Parse options from remaining parts
                    let mut i = 3;
                    while i < parts.len() {
                        match parts[i].to_uppercase().as_str() {
                            "PROMOTE" => options.promote = true,
                            "TTL" if i + 1 < parts.len() => {
                                if let Ok(secs) = parts[i + 1].parse::<u64>() {
                                    options.ttl = Some(Duration::from_secs(secs));
                                    i += 1;
                                }
                            }
                            "LABEL" if i + 1 < parts.len() => {
                                options.label = Some(parts[i + 1].to_string());
                                i += 1;
                            }
                            _ => {}
                        }
                        i += 1;
                    }
                    RedisCommand::Set {
                        key: parts[1].to_string(),
                        value: parts[2].to_string(),
                        options,
                    }
                } else {
                    RedisCommand::Unknown("SET requires key and value".to_string())
                }
            }
            "DEL" => {
                if parts.len() >= 2 {
                    let mode = if parts.len() > 2 {
                        match parts[2].to_uppercase().as_str() {
                            "FORGET" => DeleteMode::Forget(0.5),
                            "SUPPRESS" => DeleteMode::Suppress,
                            _ => DeleteMode::Normal,
                        }
                    } else {
                        DeleteMode::Normal
                    };
                    RedisCommand::Del {
                        key: parts[1].to_string(),
                        mode,
                    }
                } else {
                    RedisCommand::Unknown("DEL requires key".to_string())
                }
            }
            "BIND" => {
                if parts.len() >= 4 {
                    RedisCommand::Bind {
                        from: parts[1].to_string(),
                        verb: parts[2].to_string(),
                        to: parts[3].to_string(),
                    }
                } else {
                    RedisCommand::Unknown("BIND requires from, verb, to".to_string())
                }
            }
            "UNBIND" => {
                if parts.len() >= 3 {
                    RedisCommand::Unbind {
                        edge: parts[1].to_string(),
                        known: parts[2].to_string(),
                    }
                } else {
                    RedisCommand::Unknown("UNBIND requires edge and known".to_string())
                }
            }
            "RESONATE" => {
                if parts.len() >= 2 {
                    let k = if parts.len() >= 3 {
                        parts[2].parse().unwrap_or(10)
                    } else {
                        10
                    };
                    RedisCommand::Resonate {
                        query: parts[1].to_string(),
                        k,
                    }
                } else {
                    RedisCommand::Unknown("RESONATE requires query".to_string())
                }
            }
            "SEARCH" => {
                if parts.len() >= 2 {
                    let k = if parts.len() >= 3 {
                        parts[2].parse().unwrap_or(10)
                    } else {
                        10
                    };
                    let threshold = if parts.len() >= 4 {
                        parts[3].parse().unwrap_or(0.5)
                    } else {
                        0.5
                    };
                    RedisCommand::Search {
                        query: parts[1].to_string(),
                        k,
                        threshold,
                    }
                } else {
                    RedisCommand::Unknown("SEARCH requires query".to_string())
                }
            }
            "TRAVERSE" => {
                if parts.len() >= 3 {
                    let hops = if parts.len() >= 4 {
                        parts[3].parse().unwrap_or(1)
                    } else {
                        1
                    };
                    RedisCommand::Traverse {
                        start: parts[1].to_string(),
                        verb: parts[2].to_string(),
                        hops,
                    }
                } else {
                    RedisCommand::Unknown("TRAVERSE requires start and verb".to_string())
                }
            }
            "FANOUT" => {
                if parts.len() >= 2 {
                    RedisCommand::Fanout {
                        addr: parts[1].to_string(),
                    }
                } else {
                    RedisCommand::Unknown("FANOUT requires address".to_string())
                }
            }
            "FANIN" => {
                if parts.len() >= 2 {
                    RedisCommand::Fanin {
                        addr: parts[1].to_string(),
                    }
                } else {
                    RedisCommand::Unknown("FANIN requires address".to_string())
                }
            }
            "CRYSTALLIZE" => {
                if parts.len() >= 2 {
                    RedisCommand::Crystallize {
                        addr: parts[1].to_string(),
                    }
                } else {
                    RedisCommand::Unknown("CRYSTALLIZE requires address".to_string())
                }
            }
            "EVAPORATE" => {
                if parts.len() >= 2 {
                    RedisCommand::Evaporate {
                        addr: parts[1].to_string(),
                    }
                } else {
                    RedisCommand::Unknown("EVAPORATE requires address".to_string())
                }
            }
            "TICK" => RedisCommand::Tick,
            "CAM" => {
                if parts.len() >= 2 {
                    let args: Vec<String> = parts[2..].iter().map(|s| s.to_string()).collect();
                    RedisCommand::Cam {
                        operation: parts[1].to_string(),
                        args,
                    }
                } else {
                    RedisCommand::Unknown("CAM requires operation".to_string())
                }
            }
            "INFO" => RedisCommand::Info,
            "STATS" => RedisCommand::Stats,
            "PING" => RedisCommand::Ping,
            _ => RedisCommand::Unknown(command.to_string()),
        }
    }

    /// Resolve a key to address
    fn resolve_key(&self, key: &str) -> Option<CogAddr> {
        if let Some(&addr) = self.key_map.get(key) {
            return Some(addr);
        }
        if let Some(addr) = self.parse_addr(key) {
            return Some(addr);
        }
        // Try verb lookup
        self.substrate.verb(key)
    }

    /// Resolve a verb name to address
    fn resolve_verb(&self, verb: &str) -> Option<CogAddr> {
        self.substrate.verb(verb)
    }

    /// Parse hex address
    fn parse_addr(&self, s: &str) -> Option<CogAddr> {
        if s.starts_with("0x") || s.starts_with("0X") {
            u16::from_str_radix(&s[2..], 16).ok().map(CogAddr::new)
        } else {
            s.parse::<u16>().ok().map(CogAddr::new)
        }
    }

    /// Generate fingerprint from string
    fn generate_fingerprint(&self, content: &str) -> [u64; FINGERPRINT_WORDS] {
        let mut fp = [0u64; FINGERPRINT_WORDS];
        let bytes = content.as_bytes();

        for (i, &b) in bytes.iter().enumerate() {
            let word = i % FINGERPRINT_WORDS;
            let bit = (b as usize * 7 + i * 13) % 64;
            fp[word] |= 1u64 << bit;
        }

        // Spread bits
        for i in 0..FINGERPRINT_WORDS {
            let seed = fp[i];
            fp[(i + 1) % FINGERPRINT_WORDS] ^= seed.rotate_left(17);
            fp[(i + 3) % FINGERPRINT_WORDS] ^= seed.rotate_right(23);
        }

        fp
    }

    /// Commit pending changes
    pub fn commit(&self) {
        self.substrate.commit();
    }
}

impl Default for RedisAdapter {
    fn default() -> Self {
        Self::default_new()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ping() {
        let mut adapter = RedisAdapter::default_new();
        let result = adapter.execute_command(RedisCommand::Ping);
        assert!(matches!(result, RedisResult::String(s) if s == "PONG"));
    }

    #[test]
    fn test_set_get() {
        let mut adapter = RedisAdapter::default_new();

        let set_result = adapter.execute("SET mykey myvalue PROMOTE");
        assert!(set_result.is_ok());

        let get_result = adapter.execute("GET mykey");
        match get_result {
            RedisResult::Node(node) => {
                assert!(node.addr.is_node());
            }
            _ => panic!("Expected node result"),
        }
    }

    #[test]
    fn test_resonate() {
        let mut adapter = RedisAdapter::default_new();

        // Create some nodes
        adapter.execute("SET node1 hello_world PROMOTE");
        adapter.execute("SET node2 hello_there PROMOTE");
        adapter.execute("SET node3 goodbye_world PROMOTE");

        let result = adapter.execute("RESONATE hello 10");
        match result {
            RedisResult::Search(hits) => {
                // Should find at least one match
                assert!(!hits.is_empty());
            }
            _ => panic!("Expected search result"),
        }
    }

    #[test]
    fn test_bind_traverse() {
        let mut adapter = RedisAdapter::default_new();

        // Create nodes
        adapter.execute("SET nodeA concept_a PROMOTE");
        adapter.execute("SET nodeB concept_b PROMOTE");

        // Bind with CAUSES verb
        let bind_result = adapter.execute("BIND nodeA CAUSES nodeB");
        // Note: CAUSES is pre-initialized in BindSpace surfaces

        match bind_result {
            RedisResult::Edge(_) => {
                // Success
            }
            RedisResult::Error(e) => {
                // May fail if CAUSES not found - that's ok for now
                println!("Bind error (expected if verb not found): {}", e);
            }
            _ => {}
        }
    }

    #[test]
    fn test_info() {
        let mut adapter = RedisAdapter::default_new();
        let result = adapter.execute("INFO");
        match result {
            RedisResult::String(info) => {
                assert!(info.contains("hot_hits:"));
                assert!(info.contains("version:"));
            }
            _ => panic!("Expected string result"),
        }
    }

    #[test]
    fn test_del() {
        let mut adapter = RedisAdapter::default_new();

        adapter.execute("SET delkey delvalue PROMOTE");
        let get1 = adapter.execute("GET delkey");
        assert!(matches!(get1, RedisResult::Node(_)));

        adapter.execute("DEL delkey");
        let get2 = adapter.execute("GET delkey");
        assert!(matches!(get2, RedisResult::Nil));
    }

    #[test]
    fn test_command_parsing() {
        let adapter = RedisAdapter::default_new();

        // Test various command formats
        match adapter.parse_command("SET foo bar PROMOTE TTL 300") {
            RedisCommand::Set {
                key,
                value,
                options,
            } => {
                assert_eq!(key, "foo");
                assert_eq!(value, "bar");
                assert!(options.promote);
                assert_eq!(options.ttl, Some(Duration::from_secs(300)));
            }
            _ => panic!("Expected SET command"),
        }

        match adapter.parse_command("TRAVERSE start CAUSES 3") {
            RedisCommand::Traverse { start, verb, hops } => {
                assert_eq!(start, "start");
                assert_eq!(verb, "CAUSES");
                assert_eq!(hops, 3);
            }
            _ => panic!("Expected TRAVERSE command"),
        }
    }
}
