//! The artifact repository (design §3.4). Layout per pass:
//!
//! ```text
//! build/.<pass>/manifest.json  verify.json  out/<kind>/…  views/  logs/
//! ```
//!
//! Each artifact kind owns the directory `out/<kind>/`, so every file a pass writes belongs to exactly
//! one kind and a kind's content hash covers all of it.

use std::path::{Path, PathBuf};

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::canon;
use crate::error::{CoreError, IoContext, Result};
use crate::manifest::{FileRecord, Manifest, OutputRecord};
use crate::verify::VerifyReport;

/// The name of an artifact kind, e.g. `"hello.a"`. Also its directory name under `out/`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArtifactKind(pub &'static str);

impl ArtifactKind {
    pub fn name(&self) -> &'static str {
        self.0
    }

    /// Kinds are path components: `[a-z0-9_.]`, not starting with `.`.
    pub fn is_valid(&self) -> bool {
        !self.0.is_empty()
            && !self.0.starts_with('.')
            && self
                .0
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '.')
    }
}

impl std::fmt::Display for ArtifactKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

/// The root of all pass directories (`paths.build_dir`).
#[derive(Debug, Clone)]
pub struct ArtifactStore {
    root: PathBuf,
}

impl ArtifactStore {
    pub fn new(build_dir: impl Into<PathBuf>) -> Self {
        Self {
            root: build_dir.into(),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn pass_dir(&self, pass: &str) -> PassDir {
        PassDir {
            pass: pass.to_string(),
            root: self.root.join(format!(".{pass}")),
        }
    }

    /// `build/.pipeline/`: run summaries and driver logs.
    pub fn pipeline_dir(&self) -> PathBuf {
        self.root.join(".pipeline")
    }
}

/// `build/.<pass>/`.
#[derive(Debug, Clone)]
pub struct PassDir {
    pass: String,
    root: PathBuf,
}

impl PassDir {
    pub fn pass(&self) -> &str {
        &self.pass
    }
    pub fn path(&self) -> &Path {
        &self.root
    }
    pub fn out_dir(&self) -> PathBuf {
        self.root.join("out")
    }
    pub fn views_dir(&self) -> PathBuf {
        self.root.join("views")
    }
    pub fn logs_dir(&self) -> PathBuf {
        self.root.join("logs")
    }
    pub fn manifest_path(&self) -> PathBuf {
        self.root.join("manifest.json")
    }
    pub fn verify_path(&self) -> PathBuf {
        self.root.join("verify.json")
    }

    pub fn kind(&self, kind: ArtifactKind) -> KindDir {
        KindDir {
            kind,
            dir: self.out_dir().join(kind.name()),
        }
    }

    /// Remove the previous attempt's outputs, views, logs and verify report, then recreate the
    /// directories. The manifest is kept until it is overwritten.
    pub fn reset(&self) -> Result<()> {
        for d in [self.out_dir(), self.views_dir(), self.logs_dir()] {
            if d.exists() {
                std::fs::remove_dir_all(&d).at(&d)?;
            }
            std::fs::create_dir_all(&d).at(&d)?;
        }
        let v = self.verify_path();
        if v.exists() {
            std::fs::remove_file(&v).at(&v)?;
        }
        Ok(())
    }

    /// Delete the whole pass directory.
    pub fn remove(&self) -> Result<()> {
        if self.root.exists() {
            std::fs::remove_dir_all(&self.root).at(&self.root)?;
        }
        Ok(())
    }

    pub fn read_manifest(&self) -> Result<Option<Manifest>> {
        let p = self.manifest_path();
        if !p.exists() {
            return Ok(None);
        }
        canon::read_json_file(&p).map(Some)
    }

    pub fn write_manifest(&self, m: &Manifest) -> Result<()> {
        canon::write_json_file(&self.manifest_path(), m)
    }

    pub fn read_verify(&self) -> Result<Option<VerifyReport>> {
        let p = self.verify_path();
        if !p.exists() {
            return Ok(None);
        }
        canon::read_json_file(&p).map(Some)
    }

    pub fn write_verify(&self, r: &VerifyReport) -> Result<()> {
        canon::write_json_file(&self.verify_path(), r)
    }

    /// Write a human view (DOT, Markdown, …) under `views/`.
    pub fn write_view(&self, rel: &str, contents: &str) -> Result<PathBuf> {
        let p = safe_join(&self.views_dir(), rel)?;
        canon::write_atomic(&p, contents.as_bytes())?;
        Ok(p)
    }
}

/// `build/.<pass>/out/<kind>/`: typed read/write of one artifact kind.
#[derive(Debug, Clone)]
pub struct KindDir {
    kind: ArtifactKind,
    dir: PathBuf,
}

impl KindDir {
    pub fn kind(&self) -> ArtifactKind {
        self.kind
    }
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn path(&self, rel: &str) -> Result<PathBuf> {
        safe_join(&self.dir, rel)
    }

    pub fn exists(&self) -> bool {
        self.dir.is_dir()
    }

    pub fn create(&self) -> Result<()> {
        std::fs::create_dir_all(&self.dir).at(&self.dir)
    }

    /// Canonical pretty JSON (sorted keys).
    pub fn write_json<T: Serialize + ?Sized>(&self, rel: &str, v: &T) -> Result<PathBuf> {
        let p = self.path(rel)?;
        canon::write_json_file(&p, v)?;
        Ok(p)
    }

    pub fn read_json<T: DeserializeOwned>(&self, rel: &str) -> Result<T> {
        canon::read_json_file(&self.path(rel)?)
    }

    /// Postcard for large binary data, always with a JSON summary beside it at `<rel>.summary.json`
    /// (conventions: Artifacts / Formats).
    pub fn write_postcard<T: Serialize + ?Sized, S: Serialize + ?Sized>(
        &self,
        rel: &str,
        v: &T,
        summary: &S,
    ) -> Result<PathBuf> {
        let p = self.path(rel)?;
        let bytes = postcard::to_stdvec(v).map_err(|source| CoreError::Postcard {
            path: p.clone(),
            source,
        })?;
        canon::write_atomic(&p, &bytes)?;
        canon::write_json_file(&self.path(&format!("{rel}.summary.json"))?, summary)?;
        Ok(p)
    }

    pub fn read_postcard<T: DeserializeOwned>(&self, rel: &str) -> Result<T> {
        let p = self.path(rel)?;
        let bytes = std::fs::read(&p).at(&p)?;
        postcard::from_bytes(&bytes).map_err(|source| CoreError::Postcard { path: p, source })
    }

    /// Every file of this kind, sorted by relative path, with its blake3 hash and size.
    pub fn files(&self) -> Result<Vec<FileRecord>> {
        let mut rels = Vec::new();
        if self.dir.is_dir() {
            walk(&self.dir, &self.dir, &mut rels)?;
        }
        rels.sort();
        rels.into_iter()
            .map(|rel| {
                let p = self.dir.join(&rel);
                let bytes = std::fs::metadata(&p).at(&p)?.len();
                Ok(FileRecord {
                    path: rel,
                    hash: canon::hash_file(&p)?,
                    bytes,
                })
            })
            .collect()
    }

    /// The kind's content hash: blake3 over its sorted `(path, file hash)` pairs.
    pub fn record(&self) -> Result<OutputRecord> {
        let files = self.files()?;
        Ok(OutputRecord {
            kind: self.kind.name().to_string(),
            hash: hash_records(&files),
            files,
        })
    }
}

pub fn hash_records(files: &[FileRecord]) -> String {
    let mut h = blake3::Hasher::new();
    for f in files {
        h.update(f.path.as_bytes());
        h.update(b"\0");
        h.update(f.hash.as_bytes());
        h.update(b"\n");
    }
    h.finalize().to_hex().to_string()
}

fn walk(base: &Path, dir: &Path, out: &mut Vec<String>) -> Result<()> {
    for entry in std::fs::read_dir(dir).at(dir)? {
        let entry = entry.at(dir)?;
        let p = entry.path();
        let ft = entry.file_type().at(&p)?;
        if ft.is_dir() {
            walk(base, &p, out)?;
        } else if ft.is_file() {
            let name = entry.file_name();
            // Temporaries of an interrupted atomic write are not artifacts.
            if name.to_string_lossy().starts_with('.') {
                continue;
            }
            let rel = p.strip_prefix(base).map_err(|_| CoreError::Io {
                path: p.clone(),
                source: std::io::Error::other("not under the kind directory"),
            })?;
            out.push(rel.to_string_lossy().replace('\\', "/"));
        }
    }
    Ok(())
}

/// Join a relative artifact path, rejecting absolute paths and `..` (a pass writes only its own dir).
fn safe_join(base: &Path, rel: &str) -> Result<PathBuf> {
    let p = Path::new(rel);
    let ok = !rel.is_empty()
        && p.components()
            .all(|c| matches!(c, std::path::Component::Normal(_)));
    if !ok {
        return Err(CoreError::Artifact {
            kind: base.display().to_string(),
            message: format!("invalid relative path `{rel}`"),
        });
    }
    Ok(base.join(p))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct Big {
        xs: Vec<u32>,
    }

    #[test]
    fn json_and_postcard_round_trip() {
        let d = tempfile::tempdir().unwrap();
        let store = ArtifactStore::new(d.path());
        let k = store.pass_dir("p").kind(ArtifactKind("k.one"));
        k.write_json("a/b.json", &serde_json::json!({"z": 1, "a": 2}))
            .unwrap();
        let text = std::fs::read_to_string(k.path("a/b.json").unwrap()).unwrap();
        assert!(text.find("\"a\"").unwrap() < text.find("\"z\"").unwrap());
        let big = Big {
            xs: (0..100).collect(),
        };
        k.write_postcard("g.bin", &big, &serde_json::json!({"n": 100}))
            .unwrap();
        assert_eq!(k.read_postcard::<Big>("g.bin").unwrap(), big);
        let rels: Vec<_> = k.files().unwrap().into_iter().map(|f| f.path).collect();
        assert_eq!(rels, ["a/b.json", "g.bin", "g.bin.summary.json"]);
    }

    #[test]
    fn hash_changes_with_content() {
        let d = tempfile::tempdir().unwrap();
        let k = ArtifactStore::new(d.path()).pass_dir("p").kind(ArtifactKind("k"));
        k.write_json("x.json", &1).unwrap();
        let h1 = k.record().unwrap().hash;
        k.write_json("x.json", &2).unwrap();
        assert_ne!(h1, k.record().unwrap().hash);
    }

    #[test]
    fn rejects_escaping_paths() {
        let d = tempfile::tempdir().unwrap();
        let k = ArtifactStore::new(d.path()).pass_dir("p").kind(ArtifactKind("k"));
        assert!(k.write_json("../x.json", &1).is_err());
        assert!(k.write_json("/tmp/x.json", &1).is_err());
    }
}
