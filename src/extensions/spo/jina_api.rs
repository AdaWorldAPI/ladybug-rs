//! Jina AI Embedding API Client
//!
//! Actual API integration for jina-embeddings-v3

use std::io::{Read, Write};
use std::net::TcpStream;

const JINA_API_URL: &str = "api.jina.ai";
const JINA_EMBED_ENDPOINT: &str = "/v1/embeddings";

pub struct JinaClient {
    api_key: String,
}

impl JinaClient {
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
        }
    }

    /// Get embedding for single text
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        let embeddings = self.embed_batch(&[text])?;
        embeddings
            .into_iter()
            .next()
            .ok_or("No embedding returned".to_string())
    }

    /// Get embeddings for batch of texts (more efficient)
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        // Build JSON request body
        let input_json: String = texts
            .iter()
            .map(|t| format!("\"{}\"", t.replace("\"", "\\\"")))
            .collect::<Vec<_>>()
            .join(",");

        let body = format!(
            r#"{{"model":"jina-embeddings-v3","input":[{}]}}"#,
            input_json
        );

        // HTTP request (simplified - in production use reqwest or similar)
        let request = format!(
            "POST {} HTTP/1.1\r\n\
             Host: {}\r\n\
             Authorization: Bearer {}\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\
             \r\n\
             {}",
            JINA_EMBED_ENDPOINT,
            JINA_API_URL,
            self.api_key,
            body.len(),
            body
        );

        // Connect via TLS would require rustls/native-tls
        // For now, return placeholder that matches API structure
        // In production, use: reqwest::blocking::Client

        // Placeholder: generate deterministic embeddings from text
        Ok(texts.iter().map(|t| generate_pseudo_embedding(t)).collect())
    }
}

/// Generate deterministic pseudo-embedding for testing
/// Replace with actual API call in production
fn generate_pseudo_embedding(text: &str) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut embedding = vec![0.0f32; 1024];

    // Create deterministic values based on text content
    let bytes = text.as_bytes();

    for (i, window) in bytes.windows(3.min(bytes.len())).enumerate() {
        let mut hasher = DefaultHasher::new();
        window.hash(&mut hasher);
        (i as u64).hash(&mut hasher);
        let h = hasher.finish();

        // Spread across embedding dimensions
        for j in 0..16 {
            let idx = ((h >> (j * 4)) as usize + i * 17) % 1024;
            let sign = if (h >> (j + 48)) & 1 == 0 { 1.0 } else { -1.0 };
            embedding[idx] += sign * 0.1;
        }
    }

    // Add character-level features
    for (i, &byte) in bytes.iter().enumerate() {
        let idx = (byte as usize * 4 + i) % 1024;
        embedding[idx] += 0.05;
    }

    // L2 normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut embedding {
            *x /= norm;
        }
    }

    embedding
}

/// Real Jina API call using curl (shell out)
/// This works in environments where we can't use TLS directly
pub fn jina_embed_curl(api_key: &str, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
    use std::process::Command;

    // Build JSON
    let input_json: String = texts
        .iter()
        .map(|t| format!("\"{}\"", t.replace("\"", "\\\"").replace("\n", "\\n")))
        .collect::<Vec<_>>()
        .join(",");

    let body = format!(
        r#"{{"model":"jina-embeddings-v3","input":[{}],"dimensions":1024}}"#,
        input_json
    );

    let output = Command::new("curl")
        .args(&[
            "-s",
            "-X",
            "POST",
            "https://api.jina.ai/v1/embeddings",
            "-H",
            &format!("Authorization: Bearer {}", api_key),
            "-H",
            "Content-Type: application/json",
            "-d",
            &body,
        ])
        .output()
        .map_err(|e| format!("curl failed: {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "API error: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let response = String::from_utf8_lossy(&output.stdout);

    // Parse embeddings from JSON response
    // Response format: {"data":[{"embedding":[...]},...],...}
    parse_jina_response(&response)
}

fn parse_jina_response(json: &str) -> Result<Vec<Vec<f32>>, String> {
    let mut embeddings = Vec::new();

    // Find "data" array
    let data_start = json.find("\"data\"").ok_or("No data field")?;
    let array_start = json[data_start..].find('[').ok_or("No data array")? + data_start;

    // Find each embedding array
    let mut pos = array_start;
    while let Some(emb_start) = json[pos..].find("\"embedding\"") {
        let emb_pos = pos + emb_start;
        let arr_start = json[emb_pos..].find('[').ok_or("No embedding array")? + emb_pos;
        let arr_end = json[arr_start..].find(']').ok_or("No embedding end")? + arr_start;

        let arr_str = &json[arr_start + 1..arr_end];
        let values: Vec<f32> = arr_str
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();

        if values.len() >= 1024 {
            embeddings.push(values[..1024].to_vec());
        }

        pos = arr_end + 1;
    }

    if embeddings.is_empty() {
        // Try to extract error message
        if let Some(err_start) = json.find("\"error\"") {
            let msg_start = json[err_start..].find("\"message\"").unwrap_or(0) + err_start;
            let quote1 = json[msg_start..].find(':').unwrap_or(0) + msg_start + 1;
            let quote2 = json[quote1..].find('"').unwrap_or(0) + quote1 + 1;
            let quote3 = json[quote2..].find('"').unwrap_or(100) + quote2;
            return Err(format!("Jina API error: {}", &json[quote2..quote3]));
        }
        return Err(format!(
            "Failed to parse embeddings from: {}...",
            &json[..200.min(json.len())]
        ));
    }

    Ok(embeddings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pseudo_embedding() {
        let e1 = generate_pseudo_embedding("Ada");
        let e2 = generate_pseudo_embedding("Ada");
        let e3 = generate_pseudo_embedding("Jan");

        // Same text → same embedding
        assert_eq!(e1, e2);

        // Different text → different embedding
        assert_ne!(e1, e3);

        // Correct dimension
        assert_eq!(e1.len(), 1024);

        // Normalized (L2 norm ≈ 1)
        let norm: f32 = e1.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }
}
