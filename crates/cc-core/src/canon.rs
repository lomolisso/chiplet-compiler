//! Canonical JSON and content hashes. Determinism rule (conventions): same inputs and config give
//! byte-identical artifacts, so object keys are always sorted, independent of serde_json features.

use std::fs::File;
use std::io::Write;
use std::path::Path;

use serde::Serialize;
use serde_json::{Map, Value};

use crate::error::{CoreError, IoContext, Result};

/// Recursively sort object keys. Works whether or not serde_json's `preserve_order` is enabled
/// somewhere in the dependency graph (feature unification).
pub fn sort_keys(v: Value) -> Value {
    match v {
        Value::Object(m) => {
            let mut entries: Vec<(String, Value)> = m.into_iter().collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            let mut out = Map::new();
            for (k, v) in entries {
                out.insert(k, sort_keys(v));
            }
            Value::Object(out)
        }
        Value::Array(a) => Value::Array(a.into_iter().map(sort_keys).collect()),
        other => other,
    }
}

/// Serialize to a sorted `serde_json::Value`.
pub fn to_canonical_value<T: Serialize + ?Sized>(v: &T) -> Result<Value, serde_json::Error> {
    Ok(sort_keys(serde_json::to_value(v)?))
}

/// Pretty, sorted JSON with a trailing newline: the on-disk format of every JSON artifact.
pub fn to_canonical_pretty<T: Serialize + ?Sized>(v: &T) -> Result<Vec<u8>, serde_json::Error> {
    let mut bytes = serde_json::to_vec_pretty(&to_canonical_value(v)?)?;
    bytes.push(b'\n');
    Ok(bytes)
}

/// Compact, sorted JSON: the hashing format (cache keys).
pub fn to_canonical_compact<T: Serialize + ?Sized>(v: &T) -> Result<Vec<u8>, serde_json::Error> {
    serde_json::to_vec(&to_canonical_value(v)?)
}

/// Current time, RFC 3339 UTC with milliseconds.
pub fn now_utc() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

/// blake3 of a byte string, lowercase hex.
pub fn hash_bytes(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

/// blake3 of a file's bytes, streamed.
pub fn hash_file(path: &Path) -> Result<String> {
    let file = File::open(path).at(path)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update_reader(file).at(path)?;
    Ok(hasher.finalize().to_hex().to_string())
}

/// Content hash of a source tree: blake3 over the sorted `(relative path, file hash)` pairs of every
/// regular file under `root` whose name ends with one of `suffixes` (all files if empty). Directories
/// named `__pycache__` or starting with `.` are skipped. A missing `root` hashes as empty.
pub fn hash_tree(root: &Path, suffixes: &[&str]) -> Result<String> {
    fn walk(dir: &Path, base: &Path, suffixes: &[&str], out: &mut Vec<(String, String)>) -> Result<()> {
        for entry in std::fs::read_dir(dir).at(dir)? {
            let path = entry.at(dir)?.path();
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            if path.is_dir() {
                if name != "__pycache__" && !name.starts_with('.') {
                    walk(&path, base, suffixes, out)?;
                }
            } else if path.is_file() && (suffixes.is_empty() || suffixes.iter().any(|s| name.ends_with(s))) {
                let rel = path
                    .strip_prefix(base)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .into_owned();
                out.push((rel, hash_file(&path)?));
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    if root.is_dir() {
        walk(root, root, suffixes, &mut files)?;
    } else if root.is_file() {
        files.push((String::new(), hash_file(root)?));
    }
    files.sort();
    let mut h = blake3::Hasher::new();
    for (rel, hash) in &files {
        h.update(rel.as_bytes());
        h.update(&[0]);
        h.update(hash.as_bytes());
        h.update(&[0]);
    }
    Ok(h.finalize().to_hex().to_string())
}

/// Write via a temporary sibling and rename, so readers never see a half-written file. No fsync: the
/// rename gives readers atomicity, and an fsync per artifact costs ~100 ms on the NFS home directory.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let dir = path.parent().ok_or_else(|| CoreError::Io {
        path: path.to_path_buf(),
        source: std::io::Error::other("path has no parent directory"),
    })?;
    std::fs::create_dir_all(dir).at(dir)?;
    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let tmp = dir.join(format!(".{file_name}.tmp-{}", std::process::id()));
    {
        let mut f = File::create(&tmp).at(&tmp)?;
        f.write_all(bytes).at(&tmp)?;
    }
    std::fs::rename(&tmp, path).at(path)
}

/// Write `v` as canonical pretty JSON, atomically.
pub fn write_json_file<T: Serialize + ?Sized>(path: &Path, v: &T) -> Result<()> {
    let bytes = to_canonical_pretty(v).map_err(|source| CoreError::Json {
        path: path.to_path_buf(),
        source,
    })?;
    write_atomic(path, &bytes)
}

/// Read and deserialize a JSON file.
pub fn read_json_file<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let bytes = std::fs::read(path).at(path)?;
    serde_json::from_slice(&bytes).map_err(|source| CoreError::Json {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn keys_sorted_recursively() {
        let v = json!({"b": 1, "a": {"z": [ {"y": 1, "x": 2} ], "c": null}});
        let s = String::from_utf8(to_canonical_compact(&v).unwrap()).unwrap();
        assert_eq!(s, r#"{"a":{"c":null,"z":[{"x":2,"y":1}]},"b":1}"#);
    }

    #[test]
    fn pretty_has_trailing_newline() {
        let bytes = to_canonical_pretty(&json!({"a": 1})).unwrap();
        assert_eq!(bytes.last(), Some(&b'\n'));
    }

    #[test]
    fn atomic_write_and_hash() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("sub/x.json");
        write_atomic(&p, b"abc").unwrap();
        assert_eq!(hash_file(&p).unwrap(), hash_bytes(b"abc"));
    }
}
