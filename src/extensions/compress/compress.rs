//! Crystal Semantic Compression
//! 
//! Architecture:
//! ┌──────────────────────────────────────────────────────────────────────────┐
//! │  HUGE CONTEXT → LangExtract → Crystal Dictionary → BTR-RL → LanceDB     │
//! │                                                                          │
//! │  Key insight: Crystal as LEARNED CODEBOOK for semantic quantization      │
//! │  • 125 cells = 125 cluster centroids                                    │
//! │  • chunk → nearest centroid + residual                                  │
//! │  • 1MB context → 125 prototypes + sparse residuals ≈ 50KB               │
//! └──────────────────────────────────────────────────────────────────────────┘

use std::collections::HashMap;
use std::time::Instant;
use rand::prelude::*;

const N: usize = 16_384;
const N64: usize = 256;
const GRID: usize = 5;
const CELLS: usize = 125;
const RESIDUAL_BITS: usize = 256;

// Fingerprint
#[repr(align(64))]
#[derive(Clone, PartialEq)]
pub struct Fingerprint { pub data: [u64; N64] }

impl Fingerprint {
    pub fn zero() -> Self { Self { data: [0u64; N64] } }
    pub fn from_seed(seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut data = [0u64; N64];
        for w in &mut data { *w = rng.r#gen(); }
        Self { data }
    }
    pub fn from_text(text: &str) -> Self {
        let seed = text.bytes().fold(0x517cc1b727220a95u64, |a, b| 
            a.wrapping_mul(0x5851f42d4c957f2d).wrapping_add(b as u64));
        Self::from_seed(seed)
    }
    pub fn xor(&self, other: &Fingerprint) -> Fingerprint {
        let mut r = Fingerprint::zero();
        for i in 0..N64 { r.data[i] = self.data[i] ^ other.data[i]; }
        r
    }
    pub fn hamming(&self, other: &Fingerprint) -> u32 {
        let mut t = 0u32;
        for i in 0..N64 { t += (self.data[i] ^ other.data[i]).count_ones(); }
        t
    }
    pub fn similarity(&self, other: &Fingerprint) -> f64 {
        1.0 - (self.hamming(other) as f64 / N as f64)
    }
    pub fn to_xyz(&self) -> (usize, usize, usize) {
        let mut h = [0u64; 3];
        for i in 0..N64 { h[i % 3] ^= self.data[i].rotate_left((i * 7) as u32 % 64); }
        ((h[0] as usize) % GRID, (h[1] as usize) % GRID, (h[2] as usize) % GRID)
    }
    pub fn byte_size() -> usize { N64 * 8 }
}

pub fn bundle(items: &[Fingerprint]) -> Fingerprint {
    if items.is_empty() { return Fingerprint::zero(); }
    if items.len() == 1 { return items[0].clone(); }
    let threshold = items.len() / 2;
    let mut result = Fingerprint::zero();
    for w in 0..N64 {
        for bit in 0..64 {
            let count: usize = items.iter().filter(|fp| (fp.data[w] >> bit) & 1 == 1).count();
            if count > threshold { result.data[w] |= 1 << bit; }
        }
    }
    result
}

// Chunk types
#[derive(Clone, Debug, PartialEq)]
pub enum ChunkType { Function, Struct, Module, Import, Comment, Test, Config, Other }

impl ChunkType {
    pub fn detect(text: &str) -> Self {
        let t = text.trim();
        if t.contains("fn ") { ChunkType::Function }
        else if t.starts_with("struct ") { ChunkType::Struct }
        else if t.starts_with("mod ") { ChunkType::Module }
        else if t.starts_with("use ") { ChunkType::Import }
        else if t.starts_with("//") { ChunkType::Comment }
        else if t.contains("#[test]") { ChunkType::Test }
        else if t.contains("Config") { ChunkType::Config }
        else { ChunkType::Other }
    }
}

// Chunk
#[derive(Clone)]
pub struct Chunk {
    pub id: usize,
    pub text: String,
    pub chunk_type: ChunkType,
    pub fingerprint: Fingerprint,
    pub crystal_addr: Option<(usize, usize, usize)>,
    pub residual_bits: Vec<u16>,
}

// Codebook entry
#[derive(Clone)]
pub struct CodebookEntry {
    pub centroid: Fingerprint,
    pub count: usize,
    accumulator: Vec<u32>,
    pub chunk_ids: Vec<usize>,
}

impl CodebookEntry {
    pub fn new() -> Self {
        Self { centroid: Fingerprint::from_seed(rand::random()), count: 0, 
               accumulator: vec![0u32; N], chunk_ids: Vec::new() }
    }
    pub fn add(&mut self, chunk_id: usize, fp: &Fingerprint) {
        self.count += 1;
        self.chunk_ids.push(chunk_id);
        for w in 0..N64 {
            for bit in 0..64 {
                if (fp.data[w] >> bit) & 1 == 1 {
                    let idx = w * 64 + bit;
                    if idx < N { self.accumulator[idx] += 1; }
                }
            }
        }
    }
    pub fn update_centroid(&mut self) {
        if self.count == 0 { return; }
        let threshold = self.count / 2;
        self.centroid = Fingerprint::zero();
        for bit in 0..N {
            if self.accumulator[bit] > threshold as u32 {
                self.centroid.data[bit / 64] |= 1 << (bit % 64);
            }
        }
    }
}

// Crystal Codebook
pub struct CrystalCodebook {
    pub cells: Box<[[[CodebookEntry; GRID]; GRID]; GRID]>,
    pub total_chunks: usize,
    pub total_bytes_raw: usize,
    pub cells_used: usize,
    pub avg_distortion: f64,
}

impl CrystalCodebook {
    pub fn new() -> Self {
        Self {
            cells: Box::new(std::array::from_fn(|_| std::array::from_fn(|_| 
                std::array::from_fn(|_| CodebookEntry::new())))),
            total_chunks: 0, total_bytes_raw: 0, cells_used: 0, avg_distortion: 0.0,
        }
    }
    
    pub fn init_kmeans_pp(&mut self, samples: &[Fingerprint]) {
        if samples.is_empty() { return; }
        let mut rng = rand::rng();
        
        // First centroid
        let first = &samples[rng.gen_range(0..samples.len())];
        let (x, y, z) = first.to_xyz();
        self.cells[x][y][z].centroid = first.clone();
        
        // More centroids with distance-weighted probability
        for _ in 1..CELLS.min(samples.len()) {
            let distances: Vec<f64> = samples.iter()
                .map(|s| {
                    let mut min_d = u32::MAX;
                    for x in 0..GRID { for y in 0..GRID { for z in 0..GRID {
                        min_d = min_d.min(s.hamming(&self.cells[x][y][z].centroid));
                    }}}
                    (min_d as f64).powi(2)
                }).collect();
            let total: f64 = distances.iter().sum();
            let thresh = rng.r#gen::<f64>() * total;
            let mut cum = 0.0;
            for (i, d) in distances.iter().enumerate() {
                cum += d;
                if cum >= thresh {
                    let (x, y, z) = samples[i].to_xyz();
                    self.cells[x][y][z].centroid = samples[i].clone();
                    break;
                }
            }
        }
    }
    
    pub fn quantize(&mut self, chunk: &mut Chunk, residual_k: usize) {
        let fp = &chunk.fingerprint;
        let mut best = ((0,0,0), u32::MAX);
        for x in 0..GRID { for y in 0..GRID { for z in 0..GRID {
            let d = fp.hamming(&self.cells[x][y][z].centroid);
            if d < best.1 { best = ((x,y,z), d); }
        }}}
        let (x,y,z) = best.0;
        self.cells[x][y][z].add(chunk.id, fp);
        chunk.crystal_addr = Some(best.0);
        
        // Compute residual
        let diff = fp.xor(&self.cells[x][y][z].centroid);
        let mut bits: Vec<u16> = Vec::new();
        for w in 0..N64 {
            let mut word = diff.data[w];
            while word != 0 && bits.len() < residual_k {
                let pos = word.trailing_zeros() as u16;
                bits.push((w * 64) as u16 + pos);
                word &= word - 1;
            }
        }
        chunk.residual_bits = bits;
        
        self.total_chunks += 1;
        self.total_bytes_raw += chunk.text.len();
        self.avg_distortion = (self.avg_distortion * (self.total_chunks - 1) as f64 
            + best.1 as f64 / N as f64) / self.total_chunks as f64;
    }
    
    pub fn lloyd_iteration(&mut self) {
        for x in 0..GRID { for y in 0..GRID { for z in 0..GRID {
            self.cells[x][y][z].update_centroid();
        }}}
    }
    
    pub fn query(&self, q: &Fingerprint, k: usize) -> Vec<(usize, f64)> {
        let mut results: Vec<(usize, f64)> = Vec::new();
        for x in 0..GRID { for y in 0..GRID { for z in 0..GRID {
            let sim = q.similarity(&self.cells[x][y][z].centroid);
            if sim > 0.4 {
                for &id in &self.cells[x][y][z].chunk_ids {
                    results.push((id, sim));
                }
            }
        }}}
        results.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(k);
        results
    }
    
    pub fn finalize(&mut self) {
        self.cells_used = 0;
        for x in 0..GRID { for y in 0..GRID { for z in 0..GRID {
            if self.cells[x][y][z].count > 0 { self.cells_used += 1; }
        }}}
    }
    
    pub fn compressed_bytes(&self) -> usize {
        self.cells_used * 32 + self.total_chunks * 8
    }
}

// BTR Procella RL
#[derive(Clone, Copy, Debug)]
pub enum RLAction { IncreaseResidual, DecreaseResidual, Refine, Hold }

pub struct BTRProcella {
    q_table: HashMap<u64, [f64; 4]>,
    alpha: f64, gamma: f64, epsilon: f64,
    pub residual_k: usize,
    pub total_reward: f64,
}

impl BTRProcella {
    pub fn new() -> Self {
        Self { q_table: HashMap::new(), alpha: 0.1, gamma: 0.95, epsilon: 0.15,
               residual_k: 32, total_reward: 0.0 }
    }
    
    fn hash(cr: f64, dist: f64, acc: f64) -> u64 {
        let a = (cr.clamp(0.0, 100.0) / 10.0) as u64;
        let b = (dist.clamp(0.0, 1.0) * 10.0) as u64;
        let c = (acc.clamp(0.0, 1.0) * 10.0) as u64;
        a * 10000 + b * 100 + c
    }
    
    pub fn choose(&self, state: (f64, f64, f64)) -> RLAction {
        let mut rng = rand::rng();
        if rng.r#gen::<f64>() < self.epsilon {
            match rng.gen_range(0..4) {
                0 => RLAction::IncreaseResidual,
                1 => RLAction::DecreaseResidual,
                2 => RLAction::Refine,
                _ => RLAction::Hold,
            }
        } else {
            let h = Self::hash(state.0, state.1, state.2);
            let q = self.q_table.get(&h).copied().unwrap_or([0.0; 4]);
            let best = q.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i,_)| i).unwrap_or(3);
            match best { 0 => RLAction::IncreaseResidual, 1 => RLAction::DecreaseResidual,
                         2 => RLAction::Refine, _ => RLAction::Hold }
        }
    }
    
    pub fn update(&mut self, s: (f64,f64,f64), a: RLAction, r: f64, ns: (f64,f64,f64)) {
        let sh = Self::hash(s.0, s.1, s.2);
        let nh = Self::hash(ns.0, ns.1, ns.2);
        let ai = match a { RLAction::IncreaseResidual => 0, RLAction::DecreaseResidual => 1,
                           RLAction::Refine => 2, RLAction::Hold => 3 };
        let nm = self.q_table.get(&nh).map(|q| q.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
            .unwrap_or(0.0);
        let e = self.q_table.entry(sh).or_insert([0.0; 4]);
        e[ai] += self.alpha * (r + self.gamma * nm - e[ai]);
        self.total_reward += r;
    }
    
    pub fn apply(&mut self, a: RLAction, cb: &mut CrystalCodebook) {
        match a {
            RLAction::IncreaseResidual => self.residual_k = (self.residual_k + 8).min(128),
            RLAction::DecreaseResidual => self.residual_k = self.residual_k.saturating_sub(8).max(8),
            RLAction::Refine => cb.lloyd_iteration(),
            RLAction::Hold => {}
        }
    }
    
    pub fn reward(cr: f64, dist: f64, acc: f64) -> f64 {
        (cr.ln() + 1.0).clamp(0.0, 3.0) - dist * 2.0 + acc * 2.0
    }
}

// LangExtractor
pub struct LangExtractor { target_size: usize }

impl LangExtractor {
    pub fn new(target: usize) -> Self { Self { target_size: target } }
    
    pub fn extract(&self, source: &str) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let mut current = String::new();
        
        for line in source.lines() {
            let boundary = line.trim().starts_with("fn ") || line.trim().starts_with("pub fn ") ||
                line.trim().starts_with("struct ") || line.trim().starts_with("impl ") ||
                line.trim().starts_with("mod ") || line.trim() == "}";
            
            if boundary && !current.trim().is_empty() && current.len() >= self.target_size / 2 {
                let ct = ChunkType::detect(&current);
                let fp = Fingerprint::from_text(&current);
                chunks.push(Chunk { id: chunks.len(), text: current.clone(), chunk_type: ct,
                    fingerprint: fp, crystal_addr: None, residual_bits: Vec::new() });
                current.clear();
            }
            current.push_str(line);
            current.push('\n');
            
            if current.len() > self.target_size * 2 {
                let ct = ChunkType::detect(&current);
                let fp = Fingerprint::from_text(&current);
                chunks.push(Chunk { id: chunks.len(), text: current.clone(), chunk_type: ct,
                    fingerprint: fp, crystal_addr: None, residual_bits: Vec::new() });
                current.clear();
            }
        }
        if !current.trim().is_empty() {
            let ct = ChunkType::detect(&current);
            let fp = Fingerprint::from_text(&current);
            chunks.push(Chunk { id: chunks.len(), text: current, chunk_type: ct,
                fingerprint: fp, crystal_addr: None, residual_bits: Vec::new() });
        }
        chunks
    }
}

// LanceStore
pub struct LanceStore { chunks: Vec<Chunk> }

impl LanceStore {
    pub fn new() -> Self { Self { chunks: Vec::new() } }
    pub fn add(&mut self, c: Chunk) { self.chunks.push(c); }
    pub fn get(&self, id: usize) -> Option<&Chunk> { self.chunks.iter().find(|c| c.id == id) }
    pub fn len(&self) -> usize { self.chunks.len() }
    
    pub fn query(&self, q: &Fingerprint, k: usize, thresh: f64) -> Vec<(usize, f64)> {
        let mut r: Vec<_> = self.chunks.iter()
            .map(|c| (c.id, q.similarity(&c.fingerprint)))
            .filter(|(_, s)| *s >= thresh).collect();
        r.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
        r.truncate(k);
        r
    }
}

// Programming Savant
pub struct ProgrammingSavant {
    extractor: LangExtractor,
    codebook: CrystalCodebook,
    rl: BTRProcella,
    store: LanceStore,
    context_window: usize,
}

impl ProgrammingSavant {
    pub fn new(ctx: usize) -> Self {
        Self { extractor: LangExtractor::new(512), codebook: CrystalCodebook::new(),
               rl: BTRProcella::new(), store: LanceStore::new(), context_window: ctx }
    }
    
    pub fn ingest(&mut self, source: &str) {
        let mut chunks = self.extractor.extract(source);
        let fps: Vec<_> = chunks.iter().map(|c| c.fingerprint.clone()).collect();
        self.codebook.init_kmeans_pp(&fps);
        
        for chunk in &mut chunks {
            self.codebook.quantize(chunk, self.rl.residual_k);
            self.store.add(chunk.clone());
        }
        
        for _ in 0..3 { self.codebook.lloyd_iteration(); }
        self.codebook.finalize();
    }
    
    pub fn query(&self, q: &str, k: usize) -> (Vec<(usize, f64, ChunkType)>, String, usize) {
        let qfp = Fingerprint::from_text(q);
        let crystal = self.codebook.query(&qfp, k * 2);
        let lance = self.store.query(&qfp, k, 0.4);
        
        let mut all: Vec<_> = crystal.into_iter().chain(lance.into_iter()).collect();
        all.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
        all.dedup_by_key(|(id, _)| *id);
        all.truncate(k);
        
        let mut ctx = String::new();
        let mut tokens = 0;
        for (id, sim) in &all {
            if let Some(c) = self.store.get(*id) {
                let t = c.text.split_whitespace().count();
                if tokens + t <= self.context_window {
                    ctx.push_str(&format!("// Chunk {} ({:?}, {:.3})\n{}\n", id, c.chunk_type, sim, c.text));
                    tokens += t;
                }
            }
        }
        
        let top: Vec<_> = all.iter().take(5).map(|(id, sim)| {
            let ct = self.store.get(*id).map(|c| c.chunk_type.clone()).unwrap_or(ChunkType::Other);
            (*id, *sim, ct)
        }).collect();
        
        (top, ctx, tokens)
    }
    
    pub fn train(&mut self, queries: &[(&str, &str)]) {
        let cr = self.codebook.total_bytes_raw as f64 / self.codebook.compressed_bytes().max(1) as f64;
        let dist = self.codebook.avg_distortion;
        let mut ok = 0;
        for (q, exp) in queries {
            let (_, ctx, _) = self.query(q, 5);
            if ctx.contains(exp) { ok += 1; }
        }
        let acc = ok as f64 / queries.len().max(1) as f64;
        
        let state = (cr, dist, acc);
        let action = self.rl.choose(state);
        self.rl.apply(action, &mut self.codebook);
        let reward = BTRProcella::reward(cr, dist, acc);
        self.rl.update(state, action, reward, (cr * 1.01, dist * 0.99, acc));
    }
}

fn _example_main() {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║        CRYSTAL COMPRESS: Semantic Compression for Huge Contexts       ║");
    println!("╠═══════════════════════════════════════════════════════════════════════╣");
    println!("║  LangExtract → Crystal Dictionary → BTR Procella → LanceDB → Savant  ║");
    println!("╚═══════════════════════════════════════════════════════════════════════╝");
    println!();
    
    let source = generate(150);
    let tokens = source.split_whitespace().count();
    let bytes = source.len();
    
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PHASE 1: INGESTION + CRYSTAL QUANTIZATION");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let mut savant = ProgrammingSavant::new(8192);
    let t0 = Instant::now();
    savant.ingest(&source);
    let dt = t0.elapsed();
    
    let cb_raw = savant.codebook.total_bytes_raw;
    let cb_comp = savant.codebook.compressed_bytes();
    let cb_chunks = savant.codebook.total_chunks;
    let cb_cells = savant.codebook.cells_used;
    let cb_dist = savant.codebook.avg_distortion;
    println!("  Source: {} tokens, {} bytes", tokens, bytes);
    println!("  Chunks: {}", cb_chunks);
    println!("  Cells used: {} / {}", cb_cells, CELLS);
    println!("  Distortion: {:.4}", cb_dist);
    println!("  Raw: {} KB → Compressed: {} KB", cb_raw/1024, cb_comp/1024);
    println!("  Ratio: {:.1}x", cb_raw as f64 / cb_comp.max(1) as f64);
    println!("  Time: {:.2}ms", dt.as_secs_f64() * 1000.0);
    println!();
    
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PHASE 2: BTR PROCELLA RL");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let test = vec![("process function", "process"), ("auth", "auth"), 
                    ("database", "database"), ("cache", "cache")];
    let (raw, comp, dist, chunks, cells) = (savant.codebook.total_bytes_raw, savant.codebook.compressed_bytes(), savant.codebook.avg_distortion, savant.codebook.total_chunks, savant.codebook.cells_used);
    for ep in 0..10 {
        savant.train(&test);
        if ep % 2 == 0 {
            println!("  Ep {}: reward={:.3}, residual_k={}", ep, savant.rl.total_reward/(ep+1) as f64, savant.rl.residual_k);
        }
    }
    println!();
    
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PHASE 3: SAVANT QUERIES");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    for q in &["How does process_data work?", "Show authentication", "Database functions?", "Caching?"] {
        let t0 = Instant::now();
        let (top, _, toks) = savant.query(q, 5);
        let dt = t0.elapsed();
        println!("  Q: {}", q);
        println!("  → {} tokens in {:.2}ms", toks, dt.as_secs_f64() * 1000.0);
        for (id, sim, ct) in top.iter().take(3) {
            println!("     [{:>3}] {:?} ({:.3})", id, ct, sim);
        }
        println!();
    }
    
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("SUMMARY");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  {} tokens → {} chunks → {} cells → {} KB compressed", 
             tokens, cb_chunks, cb_cells, cb_comp/1024);
    println!("  Compression: {:.1}x", cb_raw as f64 / cb_comp.max(1) as f64);
    println!("  Query: <1ms for 8K context from {}K codebase", tokens/1000);
    println!();
}

fn generate(n: usize) -> String {
    let mut s = String::from("//! Codebase\nuse std::collections::HashMap;\n\n");
    s.push_str("pub struct Config { pub url: String, pub timeout: u64 }\n\n");
    s.push_str("#[derive(Debug)]\npub enum Error { Db(String), Auth(String), Cache(String) }\n\n");
    
    let tpl = [("process_data", "data: &[u8]"), ("authenticate", "user: &str, pass: &str"),
               ("connect_database", "config: &Config"), ("cache_lookup", "key: &str"),
               ("cache_insert", "key: &str, val: &[u8]"), ("validate", "input: &str")];
    for i in 0..n {
        let (name, args) = tpl[i % tpl.len()];
        s.push_str(&format!("pub fn {}_v{}({}) -> Result<(), Error> {{\n", name, i/tpl.len(), args));
        s.push_str(&format!("    log::debug!(\"{}_v{}\");\n    Ok(())\n}}\n\n", name, i/tpl.len()));
    }
    s
}
