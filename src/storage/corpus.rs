//! Corpus module - Scent-indexed Arrow filesystem for training data
//!
//! # Scented Arrow FS
//!
//! Training corpora (books, documents, codebases) are stored as Arrow files
//! with fingerprint indexing for fast semantic search.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                         SCENTED ARROW FILESYSTEM                            │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │  Document → Chunks → Fingerprints → Arrow Table                            │
//! │                                                                             │
//! │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
//! │  │  Gutenberg  │ →  │   Chunker   │ →  │  Scent Gen  │ → Arrow             │
//! │  │   Book      │    │  (semantic) │    │  (384-bit)  │                     │
//! │  └─────────────┘    └─────────────┘    └─────────────┘                     │
//! │                                                                             │
//! │  Columns:                                                                   │
//! │  - chunk_id: u64                                                           │
//! │  - doc_id: string                                                          │
//! │  - text: string (the chunk content)                                        │
//! │  - fingerprint: binary[48] (384-bit scent)                                 │
//! │  - position: u32 (chunk position in document)                              │
//! │  - metadata: json (title, author, etc)                                     │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Chunking Strategies
//!
//! - **Sentence**: Split on sentence boundaries
//! - **Paragraph**: Split on paragraph boundaries
//! - **Sliding**: Overlapping windows for context
//! - **Semantic**: Split on topic changes (requires embeddings)

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::path::{Path, PathBuf};

// ============================================================================
// Fingerprint Generation (Scent)
// ============================================================================

/// 384-bit fingerprint for semantic similarity
pub type Fingerprint = [u8; 48];

/// Generate a fingerprint from text using character n-gram hashing
pub fn generate_fingerprint(text: &str) -> Fingerprint {
    let mut fp = [0u8; 48];

    // Use 3-grams and 4-grams for robust fingerprinting
    let text_lower = text.to_lowercase();
    let chars: Vec<char> = text_lower.chars().collect();

    // 3-grams
    for window in chars.windows(3) {
        let hash = hash_ngram(&window.iter().collect::<String>());
        set_bit(&mut fp, hash as usize % 384);
    }

    // 4-grams
    for window in chars.windows(4) {
        let hash = hash_ngram(&window.iter().collect::<String>());
        set_bit(&mut fp, hash as usize % 384);
    }

    // Word-level features
    for word in text_lower.split_whitespace() {
        if word.len() >= 4 {
            let hash = hash_ngram(word);
            set_bit(&mut fp, hash as usize % 384);
        }
    }

    fp
}

/// Simple hash function for n-grams
fn hash_ngram(s: &str) -> u64 {
    let mut hash: u64 = 5381;
    for c in s.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(c as u64);
    }
    hash
}

/// Set a bit in the fingerprint
fn set_bit(fp: &mut Fingerprint, bit: usize) {
    let byte_idx = bit / 8;
    let bit_idx = bit % 8;
    if byte_idx < fp.len() {
        fp[byte_idx] |= 1 << bit_idx;
    }
}

/// Compute Hamming distance between two fingerprints
pub fn hamming_distance(a: &Fingerprint, b: &Fingerprint) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Compute similarity (0.0 to 1.0) from Hamming distance
pub fn fingerprint_similarity(a: &Fingerprint, b: &Fingerprint) -> f32 {
    let dist = hamming_distance(a, b);
    1.0 - (dist as f32 / 384.0)
}

// ============================================================================
// Chunking
// ============================================================================

/// Chunking strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkStrategy {
    /// Split on sentence boundaries
    Sentence { max_sentences: usize },
    /// Split on paragraph boundaries
    Paragraph { max_paragraphs: usize },
    /// Fixed-size sliding windows with overlap
    Sliding { window_size: usize, overlap: usize },
    /// Split by character count with word boundary respect
    FixedSize { target_size: usize },
}

impl Default for ChunkStrategy {
    fn default() -> Self {
        Self::Paragraph { max_paragraphs: 3 }
    }
}

/// A single chunk of text
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Unique chunk ID
    pub id: u64,
    /// Source document ID
    pub doc_id: String,
    /// The actual text content
    pub text: String,
    /// 384-bit fingerprint
    pub fingerprint: Fingerprint,
    /// Position in document (0-indexed)
    pub position: u32,
    /// Character offset in original document
    pub char_offset: usize,
    /// Character length
    pub char_length: usize,
}

/// Chunk a document into pieces
pub fn chunk_document(
    doc_id: &str,
    text: &str,
    strategy: ChunkStrategy,
    start_id: u64,
) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut next_id = start_id;

    match strategy {
        ChunkStrategy::Sentence { max_sentences } => {
            chunks = chunk_by_sentences(doc_id, text, max_sentences, &mut next_id);
        }
        ChunkStrategy::Paragraph { max_paragraphs } => {
            chunks = chunk_by_paragraphs(doc_id, text, max_paragraphs, &mut next_id);
        }
        ChunkStrategy::Sliding { window_size, overlap } => {
            chunks = chunk_sliding_window(doc_id, text, window_size, overlap, &mut next_id);
        }
        ChunkStrategy::FixedSize { target_size } => {
            chunks = chunk_fixed_size(doc_id, text, target_size, &mut next_id);
        }
    }

    chunks
}

/// Split by sentence boundaries
fn chunk_by_sentences(doc_id: &str, text: &str, max_sentences: usize, next_id: &mut u64) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut position = 0u32;

    // Simple sentence detection
    let sentence_ends = ['.', '!', '?'];
    let mut sentences = Vec::new();
    let mut current_start = 0;

    for (i, c) in text.char_indices() {
        if sentence_ends.contains(&c) {
            // Check for common abbreviations
            let before = &text[current_start..=i];
            if !is_abbreviation(before) {
                sentences.push((current_start, i + c.len_utf8()));
                current_start = i + c.len_utf8();
            }
        }
    }

    // Handle remaining text
    if current_start < text.len() {
        sentences.push((current_start, text.len()));
    }

    // Group sentences into chunks
    for group in sentences.chunks(max_sentences) {
        if group.is_empty() {
            continue;
        }

        let start = group[0].0;
        let end = group.last().unwrap().1;
        let chunk_text = text[start..end].trim().to_string();

        if !chunk_text.is_empty() {
            chunks.push(Chunk {
                id: *next_id,
                doc_id: doc_id.to_string(),
                text: chunk_text.clone(),
                fingerprint: generate_fingerprint(&chunk_text),
                position,
                char_offset: start,
                char_length: end - start,
            });
            *next_id += 1;
            position += 1;
        }
    }

    chunks
}

/// Check if text ends with a common abbreviation
fn is_abbreviation(text: &str) -> bool {
    let abbrevs = ["Mr.", "Mrs.", "Ms.", "Dr.", "Jr.", "Sr.", "vs.", "etc.", "i.e.", "e.g."];
    abbrevs.iter().any(|a| text.ends_with(a))
}

/// Split by paragraph boundaries
fn chunk_by_paragraphs(doc_id: &str, text: &str, max_paragraphs: usize, next_id: &mut u64) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut position = 0u32;

    // Split on double newlines
    let paragraphs: Vec<(usize, &str)> = text
        .split("\n\n")
        .scan(0usize, |offset, para| {
            let start = *offset;
            *offset += para.len() + 2; // +2 for the \n\n
            Some((start, para))
        })
        .filter(|(_, p)| !p.trim().is_empty())
        .collect();

    // Group paragraphs into chunks
    for group in paragraphs.chunks(max_paragraphs) {
        if group.is_empty() {
            continue;
        }

        let start = group[0].0;
        let chunk_text: String = group
            .iter()
            .map(|(_, p)| *p)
            .collect::<Vec<_>>()
            .join("\n\n");

        if !chunk_text.trim().is_empty() {
            let trimmed = chunk_text.trim().to_string();
            chunks.push(Chunk {
                id: *next_id,
                doc_id: doc_id.to_string(),
                text: trimmed.clone(),
                fingerprint: generate_fingerprint(&trimmed),
                position,
                char_offset: start,
                char_length: chunk_text.len(),
            });
            *next_id += 1;
            position += 1;
        }
    }

    chunks
}

/// Sliding window chunking with overlap
fn chunk_sliding_window(
    doc_id: &str,
    text: &str,
    window_size: usize,
    overlap: usize,
    next_id: &mut u64,
) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut position = 0u32;
    let chars: Vec<char> = text.chars().collect();

    let step = window_size.saturating_sub(overlap).max(1);
    let mut start = 0;

    while start < chars.len() {
        let end = (start + window_size).min(chars.len());
        let chunk_text: String = chars[start..end].iter().collect();

        // Trim to word boundaries if possible
        let trimmed = trim_to_word_boundary(&chunk_text);

        if !trimmed.is_empty() {
            chunks.push(Chunk {
                id: *next_id,
                doc_id: doc_id.to_string(),
                text: trimmed.clone(),
                fingerprint: generate_fingerprint(&trimmed),
                position,
                char_offset: start,
                char_length: end - start,
            });
            *next_id += 1;
            position += 1;
        }

        if end >= chars.len() {
            break;
        }

        start += step;
    }

    chunks
}

/// Fixed size chunking respecting word boundaries
fn chunk_fixed_size(doc_id: &str, text: &str, target_size: usize, next_id: &mut u64) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut position = 0u32;
    let mut current_start = 0;

    while current_start < text.len() {
        let mut end = (current_start + target_size).min(text.len());

        // Try to break at word boundary
        if end < text.len() {
            if let Some(space_pos) = text[current_start..end].rfind(char::is_whitespace) {
                end = current_start + space_pos + 1;
            }
        }

        let chunk_text = text[current_start..end].trim().to_string();

        if !chunk_text.is_empty() {
            chunks.push(Chunk {
                id: *next_id,
                doc_id: doc_id.to_string(),
                text: chunk_text.clone(),
                fingerprint: generate_fingerprint(&chunk_text),
                position,
                char_offset: current_start,
                char_length: end - current_start,
            });
            *next_id += 1;
            position += 1;
        }

        current_start = end;
    }

    chunks
}

/// Trim to word boundaries
fn trim_to_word_boundary(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    // Find last complete word
    if let Some(last_space) = trimmed.rfind(char::is_whitespace) {
        if last_space > trimmed.len() / 2 {
            return trimmed[..last_space].trim().to_string();
        }
    }

    trimmed.to_string()
}

// ============================================================================
// Document Types
// ============================================================================

/// Document metadata
#[derive(Debug, Clone)]
pub struct DocumentMeta {
    pub id: String,
    pub title: Option<String>,
    pub author: Option<String>,
    pub source: Option<String>,
    pub language: Option<String>,
    pub year: Option<i32>,
    pub tags: Vec<String>,
    pub custom: HashMap<String, String>,
}

impl Default for DocumentMeta {
    fn default() -> Self {
        Self {
            id: String::new(),
            title: None,
            author: None,
            source: None,
            language: None,
            year: None,
            tags: Vec::new(),
            custom: HashMap::new(),
        }
    }
}

/// A complete document with metadata and chunks
#[derive(Debug, Clone)]
pub struct Document {
    pub meta: DocumentMeta,
    pub raw_text: String,
    pub chunks: Vec<Chunk>,
    pub total_chars: usize,
    pub total_words: usize,
}

impl Document {
    /// Create a new document and chunk it
    pub fn new(id: &str, text: String, strategy: ChunkStrategy) -> Self {
        let meta = DocumentMeta {
            id: id.to_string(),
            ..Default::default()
        };

        let total_chars = text.chars().count();
        let total_words = text.split_whitespace().count();
        let chunks = chunk_document(id, &text, strategy, 0);

        Self {
            meta,
            raw_text: text,
            chunks,
            total_chars,
            total_words,
        }
    }

    /// Set title
    pub fn with_title(mut self, title: &str) -> Self {
        self.meta.title = Some(title.to_string());
        self
    }

    /// Set author
    pub fn with_author(mut self, author: &str) -> Self {
        self.meta.author = Some(author.to_string());
        self
    }

    /// Add a tag
    pub fn with_tag(mut self, tag: &str) -> Self {
        self.meta.tags.push(tag.to_string());
        self
    }
}

// ============================================================================
// Corpus Store
// ============================================================================

/// Configuration for the corpus store
#[derive(Debug, Clone)]
pub struct CorpusConfig {
    /// Default chunking strategy
    pub default_strategy: ChunkStrategy,
    /// Maximum chunks to keep in memory index
    pub max_index_size: usize,
    /// Path for persistent storage
    pub storage_path: PathBuf,
    /// Enable fingerprint indexing
    pub enable_fingerprint_index: bool,
}

impl Default for CorpusConfig {
    fn default() -> Self {
        Self {
            default_strategy: ChunkStrategy::Paragraph { max_paragraphs: 3 },
            max_index_size: 100_000,
            storage_path: PathBuf::from("./data/corpus"),
            enable_fingerprint_index: true,
        }
    }
}

/// The main corpus store
#[derive(Debug)]
pub struct CorpusStore {
    /// Configuration
    config: CorpusConfig,
    /// Documents by ID
    documents: RwLock<HashMap<String, Document>>,
    /// Chunk index (chunk_id -> chunk)
    chunks: RwLock<HashMap<u64, Chunk>>,
    /// Fingerprint index for fast similarity search
    fingerprint_index: RwLock<Vec<(u64, Fingerprint)>>,
    /// Next chunk ID
    next_chunk_id: AtomicU64,
    /// Statistics
    stats: RwLock<CorpusStats>,
}

/// Corpus statistics
#[derive(Debug, Clone, Default)]
pub struct CorpusStats {
    pub total_documents: usize,
    pub total_chunks: usize,
    pub total_chars: usize,
    pub total_words: usize,
    pub index_size: usize,
}

impl CorpusStore {
    /// Create a new corpus store
    pub fn new(config: CorpusConfig) -> Self {
        Self {
            config,
            documents: RwLock::new(HashMap::new()),
            chunks: RwLock::new(HashMap::new()),
            fingerprint_index: RwLock::new(Vec::new()),
            next_chunk_id: AtomicU64::new(1),
            stats: RwLock::new(CorpusStats::default()),
        }
    }

    /// Add a document to the corpus
    pub fn add_document(&self, doc: Document) -> Result<(), CorpusError> {
        let doc_id = doc.meta.id.clone();

        // Check if already exists
        if self.documents.read().unwrap().contains_key(&doc_id) {
            return Err(CorpusError::DocumentExists(doc_id));
        }

        // Index chunks
        {
            let mut chunks = self.chunks.write().unwrap();
            let mut fp_index = self.fingerprint_index.write().unwrap();

            for chunk in &doc.chunks {
                chunks.insert(chunk.id, chunk.clone());

                if self.config.enable_fingerprint_index {
                    fp_index.push((chunk.id, chunk.fingerprint));
                }
            }
        }

        // Update stats
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_documents += 1;
            stats.total_chunks += doc.chunks.len();
            stats.total_chars += doc.total_chars;
            stats.total_words += doc.total_words;
            stats.index_size = self.fingerprint_index.read().unwrap().len();
        }

        // Store document
        self.documents.write().unwrap().insert(doc_id, doc);

        Ok(())
    }

    /// Add raw text as a document
    pub fn add_text(&self, id: &str, text: String) -> Result<(), CorpusError> {
        let start_id = self.next_chunk_id.fetch_add(10000, Ordering::SeqCst);
        let mut doc = Document::new(id, text, self.config.default_strategy);

        // Reassign chunk IDs
        for chunk in &mut doc.chunks {
            chunk.id = self.next_chunk_id.fetch_add(1, Ordering::SeqCst);
        }

        self.add_document(doc)
    }

    /// Get a document by ID
    pub fn get_document(&self, doc_id: &str) -> Option<Document> {
        self.documents.read().unwrap().get(doc_id).cloned()
    }

    /// Get a chunk by ID
    pub fn get_chunk(&self, chunk_id: u64) -> Option<Chunk> {
        self.chunks.read().unwrap().get(&chunk_id).cloned()
    }

    /// Search for similar chunks by fingerprint
    pub fn search_similar(&self, query: &Fingerprint, k: usize) -> Vec<(Chunk, f32)> {
        let fp_index = self.fingerprint_index.read().unwrap();

        // Calculate similarities
        let mut scored: Vec<(u64, f32)> = fp_index
            .iter()
            .map(|(id, fp)| (*id, fingerprint_similarity(query, fp)))
            .collect();

        // Sort by similarity descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Get top k chunks
        let chunks = self.chunks.read().unwrap();
        scored
            .into_iter()
            .take(k)
            .filter_map(|(id, sim)| chunks.get(&id).map(|c| (c.clone(), sim)))
            .collect()
    }

    /// Search for chunks similar to text
    pub fn search_text(&self, query: &str, k: usize) -> Vec<(Chunk, f32)> {
        let fp = generate_fingerprint(query);
        self.search_similar(&fp, k)
    }

    /// Remove a document
    pub fn remove_document(&self, doc_id: &str) -> Option<Document> {
        let doc = self.documents.write().unwrap().remove(doc_id)?;

        // Remove chunks
        {
            let mut chunks = self.chunks.write().unwrap();
            let mut fp_index = self.fingerprint_index.write().unwrap();

            for chunk in &doc.chunks {
                chunks.remove(&chunk.id);
            }

            // Rebuild fingerprint index without removed chunks
            let chunk_ids: HashSet<u64> = doc.chunks.iter().map(|c| c.id).collect();
            fp_index.retain(|(id, _)| !chunk_ids.contains(id));
        }

        // Update stats
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_documents = stats.total_documents.saturating_sub(1);
            stats.total_chunks = stats.total_chunks.saturating_sub(doc.chunks.len());
            stats.total_chars = stats.total_chars.saturating_sub(doc.total_chars);
            stats.total_words = stats.total_words.saturating_sub(doc.total_words);
            stats.index_size = self.fingerprint_index.read().unwrap().len();
        }

        Some(doc)
    }

    /// List all document IDs
    pub fn list_documents(&self) -> Vec<String> {
        self.documents.read().unwrap().keys().cloned().collect()
    }

    /// Get statistics
    pub fn stats(&self) -> CorpusStats {
        self.stats.read().unwrap().clone()
    }

    /// Iterate over all chunks
    pub fn iter_chunks(&self) -> Vec<Chunk> {
        self.chunks.read().unwrap().values().cloned().collect()
    }
}

// ============================================================================
// Gutenberg Support
// ============================================================================

/// Parse a Project Gutenberg text file
pub fn parse_gutenberg(text: &str) -> Result<(DocumentMeta, String), CorpusError> {
    // Gutenberg files have standard headers and footers
    let start_markers = [
        "*** START OF THIS PROJECT GUTENBERG",
        "***START OF THIS PROJECT GUTENBERG",
        "*** START OF THE PROJECT GUTENBERG",
    ];
    let end_markers = [
        "*** END OF THIS PROJECT GUTENBERG",
        "***END OF THIS PROJECT GUTENBERG",
        "*** END OF THE PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ];

    // Find content start
    let content_start = start_markers
        .iter()
        .filter_map(|marker| text.find(marker))
        .min()
        .map(|pos| text[pos..].find('\n').map(|i| pos + i + 1).unwrap_or(pos))
        .unwrap_or(0);

    // Find content end
    let content_end = end_markers
        .iter()
        .filter_map(|marker| text.find(marker))
        .min()
        .unwrap_or(text.len());

    let content = text[content_start..content_end].trim().to_string();

    // Parse metadata from header
    let header = &text[..content_start];
    let mut meta = DocumentMeta::default();

    // Extract title
    if let Some(title_line) = header.lines().find(|l| l.starts_with("Title:")) {
        meta.title = Some(title_line[6..].trim().to_string());
    }

    // Extract author
    if let Some(author_line) = header.lines().find(|l| l.starts_with("Author:")) {
        meta.author = Some(author_line[7..].trim().to_string());
    }

    // Extract language
    if let Some(lang_line) = header.lines().find(|l| l.starts_with("Language:")) {
        meta.language = Some(lang_line[9..].trim().to_string());
    }

    // Extract release date/year
    if let Some(date_line) = header.lines().find(|l| l.starts_with("Release Date:") || l.starts_with("Posting Date:")) {
        // Try to extract year
        let date_text = if date_line.starts_with("Release Date:") {
            &date_line[13..]
        } else {
            &date_line[13..]
        };

        // Look for 4-digit year
        for word in date_text.split_whitespace() {
            if let Ok(year) = word.trim_matches(|c: char| !c.is_numeric()).parse::<i32>() {
                if (1000..=2100).contains(&year) {
                    meta.year = Some(year);
                    break;
                }
            }
        }
    }

    meta.source = Some("Project Gutenberg".to_string());
    meta.tags.push("gutenberg".to_string());

    if content.is_empty() {
        return Err(CorpusError::ParseError("No content found".to_string()));
    }

    Ok((meta, content))
}

/// Load a Gutenberg book into the corpus
pub fn load_gutenberg_book(
    store: &CorpusStore,
    book_id: &str,
    text: &str,
    strategy: ChunkStrategy,
) -> Result<String, CorpusError> {
    let (meta, content) = parse_gutenberg(text)?;

    let start_id = store.next_chunk_id.fetch_add(10000, Ordering::SeqCst);
    let chunks = chunk_document(book_id, &content, strategy, start_id);

    // Reassign chunk IDs to be unique
    let mut unique_chunks = Vec::new();
    for mut chunk in chunks {
        chunk.id = store.next_chunk_id.fetch_add(1, Ordering::SeqCst);
        unique_chunks.push(chunk);
    }

    let total_chars = content.chars().count();
    let total_words = content.split_whitespace().count();

    let doc = Document {
        meta: DocumentMeta {
            id: book_id.to_string(),
            ..meta
        },
        raw_text: content,
        chunks: unique_chunks,
        total_chars,
        total_words,
    };

    store.add_document(doc)?;
    Ok(book_id.to_string())
}

// ============================================================================
// Errors
// ============================================================================

/// Corpus-related errors
#[derive(Debug, Clone)]
pub enum CorpusError {
    DocumentExists(String),
    DocumentNotFound(String),
    ChunkNotFound(u64),
    ParseError(String),
    IoError(String),
}

impl std::fmt::Display for CorpusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DocumentExists(id) => write!(f, "Document already exists: {}", id),
            Self::DocumentNotFound(id) => write!(f, "Document not found: {}", id),
            Self::ChunkNotFound(id) => write!(f, "Chunk not found: {}", id),
            Self::ParseError(msg) => write!(f, "Parse error: {}", msg),
            Self::IoError(msg) => write!(f, "I/O error: {}", msg),
        }
    }
}

impl std::error::Error for CorpusError {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fingerprint_generation() {
        let fp1 = generate_fingerprint("The quick brown fox jumps over the lazy dog");
        let fp2 = generate_fingerprint("The quick brown fox jumps over the lazy dog");
        let fp3 = generate_fingerprint("A completely different sentence about cats");

        // Same text should produce same fingerprint
        assert_eq!(fp1, fp2);

        // Different text should produce different fingerprint
        assert_ne!(fp1, fp3);

        // Similar text should have lower distance than different text
        let fp4 = generate_fingerprint("The quick brown fox leaps over the lazy dog");
        let dist_similar = hamming_distance(&fp1, &fp4);
        let dist_different = hamming_distance(&fp1, &fp3);
        assert!(dist_similar < dist_different);
    }

    #[test]
    fn test_fingerprint_similarity() {
        let fp1 = generate_fingerprint("Hello world");
        let fp2 = generate_fingerprint("Hello world");
        let fp3 = generate_fingerprint("Goodbye universe");

        let sim_same = fingerprint_similarity(&fp1, &fp2);
        let sim_diff = fingerprint_similarity(&fp1, &fp3);

        assert!((sim_same - 1.0).abs() < 0.001); // Should be 1.0
        assert!(sim_diff < sim_same);
    }

    #[test]
    fn test_chunk_by_paragraphs() {
        let text = "First paragraph with some text.\n\nSecond paragraph here.\n\nThird paragraph too.";
        let chunks = chunk_document("doc1", text, ChunkStrategy::Paragraph { max_paragraphs: 1 }, 0);

        assert_eq!(chunks.len(), 3);
        assert!(chunks[0].text.contains("First"));
        assert!(chunks[1].text.contains("Second"));
        assert!(chunks[2].text.contains("Third"));
    }

    #[test]
    fn test_chunk_by_sentences() {
        let text = "First sentence. Second sentence. Third sentence. Fourth sentence.";
        let chunks = chunk_document("doc1", text, ChunkStrategy::Sentence { max_sentences: 2 }, 0);

        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].text.contains("First"));
        assert!(chunks[0].text.contains("Second"));
    }

    #[test]
    fn test_sliding_window() {
        let text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let chunks = chunk_document(
            "doc1",
            text,
            ChunkStrategy::Sliding { window_size: 10, overlap: 5 },
            0,
        );

        assert!(chunks.len() >= 3);
        // Windows should overlap
        assert!(chunks[0].text.ends_with(&chunks[1].text[..5]));
    }

    #[test]
    fn test_fixed_size_chunks() {
        let text = "This is a test document with multiple words that should be split into fixed size chunks";
        let chunks = chunk_document(
            "doc1",
            text,
            ChunkStrategy::FixedSize { target_size: 20 },
            0,
        );

        assert!(chunks.len() > 1);
        for chunk in &chunks {
            // Should be close to target size (may be shorter due to word boundaries)
            assert!(chunk.text.len() <= 25);
        }
    }

    #[test]
    fn test_document_creation() {
        let text = "Para 1.\n\nPara 2.\n\nPara 3.".to_string();
        let doc = Document::new("test", text, ChunkStrategy::Paragraph { max_paragraphs: 1 })
            .with_title("Test Doc")
            .with_author("Test Author")
            .with_tag("test");

        assert_eq!(doc.meta.title, Some("Test Doc".to_string()));
        assert_eq!(doc.meta.author, Some("Test Author".to_string()));
        assert!(doc.meta.tags.contains(&"test".to_string()));
        assert_eq!(doc.chunks.len(), 3);
    }

    #[test]
    fn test_corpus_store() {
        let store = CorpusStore::new(CorpusConfig::default());

        // Add a document
        let doc = Document::new(
            "doc1",
            "First paragraph.\n\nSecond paragraph.".to_string(),
            ChunkStrategy::Paragraph { max_paragraphs: 1 },
        );
        store.add_document(doc).unwrap();

        // Verify it's stored
        assert!(store.get_document("doc1").is_some());

        let stats = store.stats();
        assert_eq!(stats.total_documents, 1);
        assert_eq!(stats.total_chunks, 2);
    }

    #[test]
    fn test_similarity_search() {
        let store = CorpusStore::new(CorpusConfig::default());

        // Add documents
        let doc1 = Document::new(
            "doc1",
            "The cat sat on the mat.".to_string(),
            ChunkStrategy::Paragraph { max_paragraphs: 1 },
        );
        let doc2 = Document::new(
            "doc2",
            "A dog ran in the park.".to_string(),
            ChunkStrategy::Paragraph { max_paragraphs: 1 },
        );
        let doc3 = Document::new(
            "doc3",
            "The cat sat on the rug.".to_string(),
            ChunkStrategy::Paragraph { max_paragraphs: 1 },
        );

        store.add_document(doc1).unwrap();
        store.add_document(doc2).unwrap();
        store.add_document(doc3).unwrap();

        // Search for similar to "cat on mat"
        let results = store.search_text("The cat sat on", 2);

        assert_eq!(results.len(), 2);
        // Should find cat-related chunks first
        assert!(results[0].0.text.contains("cat") || results[1].0.text.contains("cat"));
    }

    #[test]
    fn test_remove_document() {
        let store = CorpusStore::new(CorpusConfig::default());

        let doc = Document::new(
            "doc1",
            "Test content.".to_string(),
            ChunkStrategy::Paragraph { max_paragraphs: 1 },
        );
        store.add_document(doc).unwrap();

        assert_eq!(store.stats().total_documents, 1);

        store.remove_document("doc1");
        assert_eq!(store.stats().total_documents, 0);
        assert!(store.get_document("doc1").is_none());
    }

    #[test]
    fn test_parse_gutenberg() {
        let gutenberg_text = r#"
The Project Gutenberg EBook of Test Book, by Test Author

Title: Test Book
Author: Test Author
Language: English
Release Date: January 1, 2020

*** START OF THIS PROJECT GUTENBERG EBOOK TEST BOOK ***

This is the actual content of the book.

It has multiple paragraphs.

And goes on for a while.

*** END OF THIS PROJECT GUTENBERG EBOOK TEST BOOK ***

End of the Project Gutenberg EBook of Test Book
"#;

        let (meta, content) = parse_gutenberg(gutenberg_text).unwrap();

        assert_eq!(meta.title, Some("Test Book".to_string()));
        assert_eq!(meta.author, Some("Test Author".to_string()));
        assert_eq!(meta.language, Some("English".to_string()));
        assert!(content.contains("actual content"));
        assert!(!content.contains("*** START"));
        assert!(!content.contains("*** END"));
    }

    #[test]
    fn test_load_gutenberg_book() {
        let store = CorpusStore::new(CorpusConfig::default());

        let gutenberg_text = r#"
Title: Test
Author: Author

*** START OF THIS PROJECT GUTENBERG EBOOK ***

Chapter 1.

Chapter 2.

*** END OF THIS PROJECT GUTENBERG EBOOK ***
"#;

        let id = load_gutenberg_book(
            &store,
            "pg12345",
            gutenberg_text,
            ChunkStrategy::Paragraph { max_paragraphs: 1 },
        )
        .unwrap();

        let doc = store.get_document(&id).unwrap();
        assert_eq!(doc.meta.title, Some("Test".to_string()));
        assert!(doc.chunks.len() >= 2);
    }
}
