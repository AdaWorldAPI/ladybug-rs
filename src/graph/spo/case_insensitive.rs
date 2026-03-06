//! CaseInsensitiveLookup — forgiving label/verb matching.
//!
//! Stolen from lance-graph's `case_insensitive.rs`. The trait gives you
//! `map.get_ci("causes")` on existing HashMaps — exact match first (free),
//! case-insensitive fallback only when needed.

use std::collections::HashMap;

/// Extension trait for case-insensitive lookup on `HashMap<String, V>`.
///
/// Fast path: exact match (normal HashMap lookup, O(1)).
/// Slow path: case-insensitive linear scan (only if exact fails).
pub trait CaseInsensitiveLookup<V> {
    /// Get a value by key, falling back to case-insensitive match.
    fn get_ci(&self, key: &str) -> Option<&V>;

    /// Check if a key exists (case-insensitive).
    fn contains_key_ci(&self, key: &str) -> bool;
}

impl<V> CaseInsensitiveLookup<V> for HashMap<String, V> {
    fn get_ci(&self, key: &str) -> Option<&V> {
        // Fast path: exact match
        if let Some(v) = self.get(key) {
            return Some(v);
        }
        // Slow path: case-insensitive scan
        let key_lower = key.to_lowercase();
        self.iter()
            .find(|(k, _)| k.to_lowercase() == key_lower)
            .map(|(_, v)| v)
    }

    fn contains_key_ci(&self, key: &str) -> bool {
        self.get_ci(key).is_some()
    }
}

/// Same trait for `BTreeMap<String, V>` (used by SpoStore).
impl<V> CaseInsensitiveLookup<V> for std::collections::BTreeMap<String, V> {
    fn get_ci(&self, key: &str) -> Option<&V> {
        if let Some(v) = self.get(key) {
            return Some(v);
        }
        let key_lower = key.to_lowercase();
        self.iter()
            .find(|(k, _)| k.to_lowercase() == key_lower)
            .map(|(_, v)| v)
    }

    fn contains_key_ci(&self, key: &str) -> bool {
        self.get_ci(key).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_match() {
        let mut map = HashMap::new();
        map.insert("CAUSES".to_string(), 42);
        assert_eq!(map.get_ci("CAUSES"), Some(&42));
    }

    #[test]
    fn test_case_insensitive_match() {
        let mut map = HashMap::new();
        map.insert("CAUSES".to_string(), 42);
        assert_eq!(map.get_ci("causes"), Some(&42));
        assert_eq!(map.get_ci("Causes"), Some(&42));
        assert_eq!(map.get_ci("cAuSeS"), Some(&42));
    }

    #[test]
    fn test_no_match() {
        let mut map = HashMap::new();
        map.insert("CAUSES".to_string(), 42);
        assert_eq!(map.get_ci("knows"), None);
    }

    #[test]
    fn test_contains_key_ci() {
        let mut map = HashMap::new();
        map.insert("CAUSES".to_string(), 42);
        assert!(map.contains_key_ci("causes"));
        assert!(!map.contains_key_ci("knows"));
    }

    #[test]
    fn test_btree_ci() {
        let mut map = std::collections::BTreeMap::new();
        map.insert("KNOWS".to_string(), "alice");
        assert_eq!(map.get_ci("knows"), Some(&"alice"));
        assert_eq!(map.get_ci("KNOWS"), Some(&"alice"));
    }
}
