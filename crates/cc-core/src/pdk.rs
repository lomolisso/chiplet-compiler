//! PDKs as data (design §3.7; docs/tools/pdks.md): one manifest per cell library, `<name>.toml` in
//! `paths.pdk_manifests` (the repository's `pdks/`). A manifest names the library's files relative to its
//! root, which is `paths.pdks.<name>` when set and `paths.pdk_root` joined with the manifest's `root`
//! otherwise, and gives what every flow needs about the library: its corner, the cell that drives the
//! inputs, a default clock and the XLS delay model. Code names no library.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config::LoadedConfig;
use crate::error::{CoreError, IoContext, Result};

/// `pdks/<name>.toml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PdkManifest {
    /// Equals the file stem; the kit's name in `config.toml` and in every artifact.
    pub name: String,
    pub description: String,
    /// The kit's directory under `paths.pdk_root` (absolute: used as it is).
    pub root: PathBuf,
    /// The Liberty files, relative to the root: `{group}`, `{vt}` and `{corner}` are substituted and `*`
    /// matches any run of characters within the file name. With `groups`, exactly one file per group;
    /// without, exactly one file.
    pub liberty: String,
    /// Cell groups shipped as separate Liberty files (ASAP7: AO, OA, INVBUF, SEQ, SIMPLE).
    #[serde(default)]
    pub groups: Vec<String>,
    /// Threshold-voltage flavour and process corner of the files (as their names spell them).
    #[serde(default)]
    pub vt: String,
    #[serde(default)]
    pub corner: String,
    /// With `groups`: the one library `chipc doctor` merges them into (`{vt}`, `{corner}` substituted).
    #[serde(default)]
    pub merged: Option<String>,
    /// LEF files, relative to the root, as `liberty` (`*` in the file name).
    #[serde(default)]
    pub lef: Vec<String>,
    /// The cell that drives every input of a characterized design.
    pub driving_cell: String,
    /// The default clock target, ps.
    pub clock_ps: f64,
    /// XLS's delay model for this library (`codegen_main --delay_model`), if XLS has one.
    #[serde(default)]
    pub xls_delay_model: Option<String>,
}

/// A manifest with its root resolved.
#[derive(Debug, Clone, PartialEq)]
pub struct Pdk {
    pub manifest: PdkManifest,
    /// The manifest file.
    pub file: PathBuf,
    pub root: PathBuf,
}

fn err(kit: &str, message: impl Into<String>) -> CoreError {
    CoreError::Pdk {
        kit: kit.to_string(),
        message: message.into(),
    }
}

/// The manifest directory (`paths.pdk_manifests`).
pub fn manifest_dir(cfg: &LoadedConfig) -> PathBuf {
    cfg.resolve(&cfg.config.paths.pdk_manifests)
}

/// The names of every manifest, sorted.
pub fn names(cfg: &LoadedConfig) -> Result<Vec<String>> {
    let dir = manifest_dir(cfg);
    let mut out = Vec::new();
    for e in std::fs::read_dir(&dir).at(&dir)? {
        let p = e.at(&dir)?.path();
        if p.extension().is_some_and(|x| x == "toml")
            && let Some(s) = p.file_stem()
        {
            out.push(s.to_string_lossy().into_owned());
        }
    }
    out.sort();
    Ok(out)
}

/// Kit `name`'s manifest with its root.
pub fn load(cfg: &LoadedConfig, name: &str) -> Result<Pdk> {
    let file = manifest_dir(cfg).join(format!("{name}.toml"));
    if !file.is_file() {
        return Err(err(
            name,
            format!(
                "no manifest {} (kits: {})",
                file.display(),
                names(cfg).unwrap_or_default().join(", ")
            ),
        ));
    }
    let text = std::fs::read_to_string(&file).at(&file)?;
    let manifest: PdkManifest = toml::from_str(&text).map_err(|e| err(name, format!("{}: {e}", file.display())))?;
    if manifest.name != name {
        return Err(err(
            name,
            format!("{}: `name` is {:?}, the file stem {name:?}", file.display(), manifest.name),
        ));
    }
    if !manifest.groups.is_empty() && manifest.merged.is_none() {
        return Err(err(name, "a kit with `groups` needs `merged`"));
    }
    let root = match cfg.config.paths.pdks.get(name) {
        Some(p) => cfg.resolve(p),
        None if manifest.root.is_absolute() => manifest.root.clone(),
        None => cfg
            .config
            .paths
            .pdk_root
            .as_ref()
            .map(|r| cfg.resolve(r).join(&manifest.root))
            .ok_or_else(|| {
                err(
                    name,
                    format!("paths.pdk_root (or paths.pdks.{name}) is unset (config.local.toml)"),
                )
            })?,
    };
    Ok(Pdk { manifest, file, root })
}

impl Pdk {
    pub fn name(&self) -> &str {
        &self.manifest.name
    }

    fn expand(&self, pattern: &str, group: &str) -> String {
        pattern
            .replace("{group}", group)
            .replace("{vt}", &self.manifest.vt)
            .replace("{corner}", &self.manifest.corner)
    }

    /// The Liberty files: one per group in `groups` order, or the one file.
    pub fn liberty_files(&self) -> Result<Vec<PathBuf>> {
        let groups: Vec<&str> = if self.manifest.groups.is_empty() {
            vec![""]
        } else {
            self.manifest.groups.iter().map(String::as_str).collect()
        };
        groups
            .iter()
            .map(|g| self.one(&self.expand(&self.manifest.liberty, g)))
            .collect()
    }

    /// The LEF files, one per pattern.
    pub fn lef_files(&self) -> Result<Vec<PathBuf>> {
        self.manifest
            .lef
            .iter()
            .map(|p| self.one(&self.expand(p, "")))
            .collect()
    }

    /// The merged library's name (kits with groups).
    pub fn merged_name(&self) -> Option<String> {
        self.manifest.merged.as_ref().map(|m| self.expand(m, ""))
    }

    /// The single file matching `pattern` (relative to the root, `*` within the file name).
    fn one(&self, pattern: &str) -> Result<PathBuf> {
        let rel = Path::new(pattern);
        let dir = self.root.join(rel.parent().unwrap_or(Path::new("")));
        let file = rel
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_default();
        if !file.contains('*') {
            let p = dir.join(&file);
            return if p.is_file() {
                Ok(p)
            } else {
                Err(err(self.name(), format!("{} is missing", p.display())))
            };
        }
        let mut hits: Vec<String> = std::fs::read_dir(&dir)
            .at(&dir)?
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .filter(|n| glob(&file, n))
            .collect();
        hits.sort();
        match hits.as_slice() {
            [one] => Ok(dir.join(one)),
            _ => Err(err(
                self.name(),
                format!(
                    "{}: expected one file matching {file}, found {}",
                    dir.display(),
                    hits.len()
                ),
            )),
        }
    }
}

/// Does `name` match `pattern`, where `*` matches any run of characters?
pub fn glob(pattern: &str, name: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    let [first, mid @ .., last] = parts.as_slice() else {
        return pattern == name;
    };
    let Some(mut rest) = name.strip_prefix(first) else {
        return false;
    };
    for m in mid {
        match rest.find(m) {
            Some(k) => rest = &rest[k + m.len()..],
            None => return false,
        }
    }
    rest.len() >= last.len() && rest.ends_with(last)
}

/// Every kit's manifest, by name (an unreadable one is an error).
pub fn load_all(cfg: &LoadedConfig) -> Result<BTreeMap<String, Pdk>> {
    names(cfg)?
        .into_iter()
        .map(|n| load(cfg, &n).map(|p| (n, p)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_matches_within_a_name() {
        assert!(glob("a_*.lib", "a_211120.lib"));
        assert!(!glob("a_*.lib", "a_211120.lib.7z"));
        assert!(!glob("a_RVT_*.lib", "a_LVT_1.lib"));
        assert!(glob("stdcells.lib", "stdcells.lib"));
        assert!(glob("x*y*z", "x1y2z"));
        assert!(!glob("ab*ba", "aba"));
    }

    fn cfg(dir: &Path, extra: &str) -> LoadedConfig {
        LoadedConfig::from_toml_str(
            dir.to_path_buf(),
            &format!("[paths]\npdk_manifests = \"m\"\npdk_root = \"r\"\n{extra}"),
        )
        .unwrap()
    }

    #[test]
    fn resolves_groups_and_roots() {
        let d = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(d.path().join("m")).unwrap();
        std::fs::write(
            d.path().join("m/k.toml"),
            "name = \"k\"\ndescription = \"x\"\nroot = \"kit\"\nliberty = \"L/k_{group}_{vt}_*.lib\"\n\
             groups = [\"A\", \"B\"]\nvt = \"R\"\nmerged = \"k_{vt}\"\ndriving_cell = \"BUF\"\nclock_ps = 1000\n",
        )
        .unwrap();
        let lib = d.path().join("r/kit/L");
        std::fs::create_dir_all(&lib).unwrap();
        for f in ["k_A_R_1.lib", "k_B_R_2.lib", "k_B_L_2.lib", "k_A_R_1.lib.7z"] {
            std::fs::write(lib.join(f), "").unwrap();
        }
        let c = cfg(d.path(), "");
        assert_eq!(names(&c).unwrap(), ["k"]);
        let p = load(&c, "k").unwrap();
        assert_eq!(p.merged_name().as_deref(), Some("k_R"));
        let files: Vec<String> = p
            .liberty_files()
            .unwrap()
            .iter()
            .map(|f| f.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(files, ["k_A_R_1.lib", "k_B_R_2.lib"]);
        std::fs::write(lib.join("k_A_R_3.lib"), "").unwrap();
        assert!(p.liberty_files().unwrap_err().to_string().contains("found 2"));
        // paths.pdks.<kit> overrides the root.
        let c = cfg(d.path(), "pdks = { k = \"elsewhere\" }\n");
        assert_eq!(load(&c, "k").unwrap().root, d.path().join("elsewhere"));
        assert!(load(&c, "nope").unwrap_err().to_string().contains("kits: k"));
    }
}
