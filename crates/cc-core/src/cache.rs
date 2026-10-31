//! Pass cache keys (spec §3.3): blake3 over the input artifact hashes, the pass's config section, the
//! model set, the versions of the tools it calls and the pass version. A pass is a pure function of
//! these, so equal keys mean equal outputs.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::canon;

#[derive(Debug, Clone, Serialize)]
pub struct CacheKeyInput<'a> {
    pub pass: &'a str,
    pub pass_version: u32,
    /// `(kind, hash)`; sorted before hashing.
    pub inputs: Vec<(String, String)>,
    /// The section without `resources`: threads and memory caps never change outputs.
    pub config: &'a serde_json::Value,
    /// Empty for passes that do not depend on the model set.
    pub models: &'a [String],
    pub tools: &'a BTreeMap<String, String>,
}

pub fn cache_key(input: &CacheKeyInput<'_>) -> String {
    let mut inputs = input.inputs.clone();
    inputs.sort();
    let mut models = input.models.to_vec();
    models.sort();
    let doc = serde_json::json!({
        "pass": input.pass,
        "pass_version": input.pass_version,
        "inputs": inputs,
        "config": input.config,
        "models": models,
        "tools": input.tools,
    });
    // Compact canonical JSON sorts every object key, so TOML key order cannot change the key.
    let bytes = canon::to_canonical_compact(&doc).unwrap_or_default();
    canon::hash_bytes(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(config: &str, inputs: Vec<(String, String)>) -> String {
        let t: toml::Table = config.parse().unwrap();
        let v = canon::to_canonical_value(&t).unwrap();
        cache_key(&CacheKeyInput {
            pass: "p",
            pass_version: 1,
            inputs,
            config: &v,
            models: &[],
            tools: &BTreeMap::new(),
        })
    }

    #[test]
    fn stable_under_key_and_input_reordering() {
        let a = key(
            "x = 1\ny = { b = 2, a = 3 }",
            vec![("k1".into(), "h1".into()), ("k2".into(), "h2".into())],
        );
        let b = key(
            "y = { a = 3, b = 2 }\nx = 1",
            vec![("k2".into(), "h2".into()), ("k1".into(), "h1".into())],
        );
        assert_eq!(a, b);
    }

    #[test]
    fn changes_with_value_or_input() {
        let base = key("x = 1", vec![("k".into(), "h".into())]);
        assert_ne!(base, key("x = 2", vec![("k".into(), "h".into())]));
        assert_ne!(base, key("x = 1", vec![("k".into(), "h2".into())]));
    }
}
