//! Layered configuration (spec §3.5): built-in defaults < `config.toml` < `config.local.toml` <
//! `--set key=value`. Fixed sections are strict (`deny_unknown_fields`); `[passes.<name>]` sections are
//! opaque here and parsed strictly by their pass, so `cc-core` knows no pass types.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::error::{CoreError, IoContext, Result};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Config {
    pub pipeline: PipelineConfig,
    pub resources: ResourcesConfig,
    pub paths: PathsConfig,
    pub models: ModelsConfig,
    pub testing: TestingConfig,
    pub tools: ToolsConfig,
    pub passes: BTreeMap<String, toml::Table>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct PipelineConfig {
    pub max_parallel_passes: usize,
    pub targets: Vec<String>,
    pub stop_on_failure: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_parallel_passes: 2,
            targets: Vec::new(),
            stop_on_failure: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ResourcesConfig {
    pub memory_budget_gb: f64,
    pub threads_total: usize,
}

impl Default for ResourcesConfig {
    fn default() -> Self {
        Self {
            memory_budget_gb: 32.0,
            threads_total: 16,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct PathsConfig {
    pub build_dir: PathBuf,
    pub models_dir: PathBuf,
    pub third_party_dir: PathBuf,
    // Machine-specific (config.local.toml).
    pub eda_toggle: Option<PathBuf>,
    pub scratch_dir: Option<PathBuf>,
    pub pdk_root: Option<PathBuf>,
    pub nangate45_lib: Option<PathBuf>,
    pub nangate45_lef: Option<PathBuf>,
    pub asap7_root: Option<PathBuf>,
    pub asap7_nldm_dir: Option<PathBuf>,
    pub asap7_lef_dir: Option<PathBuf>,
    pub synopsys_root: Option<PathBuf>,
    pub cadence_root: Option<PathBuf>,
}

impl Default for PathsConfig {
    fn default() -> Self {
        Self {
            build_dir: "build".into(),
            models_dir: "models".into(),
            third_party_dir: "third_party".into(),
            eda_toggle: None,
            scratch_dir: None,
            pdk_root: None,
            nangate45_lib: None,
            nangate45_lef: None,
            asap7_root: None,
            asap7_nldm_dir: None,
            asap7_lef_dir: None,
            synopsys_root: None,
            cadence_root: None,
        }
    }
}

/// `models.keys`: `"all"` or an explicit list of catalog keys.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ModelKeys {
    #[default]
    All,
    List(Vec<String>),
}

impl Serialize for ModelKeys {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        match self {
            ModelKeys::All => s.serialize_str("all"),
            ModelKeys::List(v) => v.serialize(s),
        }
    }
}

impl<'de> Deserialize<'de> for ModelKeys {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Str(String),
            List(Vec<String>),
        }
        match Raw::deserialize(d)? {
            Raw::Str(s) if s == "all" => Ok(ModelKeys::All),
            Raw::Str(s) => Err(serde::de::Error::custom(format!(
                "expected \"all\" or a list of keys, got \"{s}\""
            ))),
            Raw::List(v) => Ok(ModelKeys::List(v)),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelConfigKind {
    #[default]
    Full,
    Reduced,
}

impl ModelConfigKind {
    /// The key of the ports' `CONFIGS` dict.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Reduced => "reduced",
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ModelsConfig {
    pub keys: ModelKeys,
    pub config: ModelConfigKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct TestingConfig {
    pub deep_models: Vec<String>,
    pub tiny_dim: u32,
    pub seed: u64,
}

impl Default for TestingConfig {
    fn default() -> Self {
        Self {
            deep_models: Vec::new(),
            tiny_dim: 4,
            seed: 0,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ToolsConfig {
    pub host: HostToolConfig,
    pub docker: DockerConfig,
    pub eda: EdaConfig,
    pub smt: SmtConfig,
    pub xls: XlsPins,
    pub allo: AlloPins,
    pub grami: GramiPins,
    pub pdk: PdkConfig,
}

/// XLS: one commit for the release tarball and the Bazel-built tools (`docs/context/tooling/xls.md` §1).
/// Pins have no built-in value; `config.toml` holds them and `chipc doctor` checks them.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct XlsPins {
    pub tag: String,
    pub commit: String,
    pub release_sha256: String,
}

/// Allo and the LLVM commit its submodule records (`docs/context/tooling/allo.md` §1).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct AlloPins {
    pub commit: String,
    pub llvm_commit: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct GramiPins {
    pub commit: String,
}

/// Which ASAP7 Liberty set is compiled to `.db` (`docs/context/tooling/pdks.md` §3).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct PdkConfig {
    pub asap7_vt: String,
    pub asap7_corner: String,
    pub asap7_groups: Vec<String>,
}

impl Default for PdkConfig {
    fn default() -> Self {
        Self {
            asap7_vt: "RVT".into(),
            asap7_corner: "TT".into(),
            asap7_groups: ["AO", "OA", "INVBUF", "SEQ", "SIMPLE"].map(String::from).to_vec(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct HostToolConfig {
    pub watchdog_interval_s: f64,
}

impl Default for HostToolConfig {
    fn default() -> Self {
        Self {
            watchdog_interval_s: 0.5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct DockerConfig {
    pub image: String,
    pub tag: String,
    pub memory_gb: f64,
    pub cpus: f64,
    pub pids_limit: i64,
    pub build: DockerBuildConfig,
}

impl Default for DockerConfig {
    fn default() -> Self {
        Self {
            image: "chiplet-compiler/toolchain".into(),
            tag: "dev".into(),
            memory_gb: 8.0,
            cpus: 4.0,
            pids_limit: 4096,
            build: DockerBuildConfig::default(),
        }
    }
}

/// `[tools.docker.build]`: inputs of `docker/build.sh` that only the image build uses. Pins have
/// no built-in value (config.toml holds them); the caps bound the one-time build (spec §8).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct DockerBuildConfig {
    pub ubuntu_2404: String,
    pub ubuntu_2204: String,
    pub apt_snapshot: String,
    pub pypi_exclude_newer: String,
    pub uv_version: String,
    pub uv_sha256: String,
    pub bazelisk_version: String,
    pub bazelisk_sha256: String,
    pub z3_version: String,
    pub z3_sha256: String,
    pub bitwuzla_version: String,
    pub bitwuzla_sha256: String,
    pub memory_gb: u32,
    pub cpus: u32,
    pub link_jobs: u32,
    pub bazel_ram_mb: u32,
}

impl Default for DockerBuildConfig {
    fn default() -> Self {
        Self {
            ubuntu_2404: String::new(),
            ubuntu_2204: String::new(),
            apt_snapshot: String::new(),
            pypi_exclude_newer: String::new(),
            uv_version: String::new(),
            uv_sha256: String::new(),
            bazelisk_version: String::new(),
            bazelisk_sha256: String::new(),
            z3_version: String::new(),
            z3_sha256: String::new(),
            bitwuzla_version: String::new(),
            bitwuzla_sha256: String::new(),
            memory_gb: 64,
            cpus: 16,
            link_jobs: 2,
            bazel_ram_mb: 49152,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct EdaConfig {
    pub max_licenses: usize,
    pub watchdog_interval_s: f64,
    /// Extra attempts after a license failure; the job never counts as a design failure.
    pub license_retries: u32,
    /// Wait before the first retry; doubles on each further attempt.
    pub license_backoff_s: f64,
    /// Substrings of tool output that mean "no license" (docs/context/tooling/eda_environment.md §4).
    pub license_error_patterns: Vec<String>,
}

impl Default for EdaConfig {
    fn default() -> Self {
        Self {
            max_licenses: 2,
            watchdog_interval_s: 2.0,
            license_retries: 3,
            license_backoff_s: 30.0,
            license_error_patterns: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct SmtConfig {
    pub solver: String,
    pub timeout_s: u64,
}

impl Default for SmtConfig {
    fn default() -> Self {
        Self {
            solver: "bitwuzla".into(),
            timeout_s: 60,
        }
    }
}

/// `[passes.<name>.resources]`: what the scheduler admits against the budget (spec §3.3, §8).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, default)]
pub struct PassResources {
    pub threads: usize,
    pub memory_gb: f64,
    pub timeout_min: f64,
}

impl Default for PassResources {
    fn default() -> Self {
        Self {
            threads: 4,
            memory_gb: 4.0,
            timeout_min: 60.0,
        }
    }
}

/// One pass's section with `resources` split off. The pass parses `options` into its own strict type.
#[derive(Debug, Clone)]
pub struct PassSection {
    pub pass: String,
    pub options: toml::Table,
    pub resources: PassResources,
}

impl PassSection {
    pub fn parse<T: DeserializeOwned>(&self) -> Result<T> {
        serde_path_to_error::deserialize(toml::Value::Table(self.options.clone())).map_err(|e| {
            CoreError::PassConfig {
                pass: self.pass.clone(),
                message: describe(&e),
            }
        })
    }

    /// The options as canonical JSON (recorded in the manifest, hashed into the cache key).
    pub fn options_json(&self) -> serde_json::Value {
        crate::canon::to_canonical_value(&self.options).unwrap_or(serde_json::Value::Null)
    }
}

/// One source of configuration, in increasing priority.
#[derive(Debug, Clone)]
pub struct Layer {
    pub name: String,
    pub table: toml::Table,
}

impl Layer {
    pub fn parse(name: impl Into<String>, text: &str) -> Result<Self> {
        let name = name.into();
        let table = text.parse::<toml::Table>().map_err(|e| CoreError::Config {
            layer: name.clone(),
            message: e.to_string(),
        })?;
        Ok(Self { name, table })
    }

    /// `--set a.b.c=value` overrides; the value is TOML when it parses as TOML, else a string.
    pub fn from_overrides(sets: &[String]) -> Result<Self> {
        let mut table = toml::Table::new();
        for s in sets {
            let (key, raw) = s.split_once('=').ok_or_else(|| CoreError::Config {
                layer: "--set".into(),
                message: format!("`{s}` is not key=value"),
            })?;
            let path: Vec<&str> = key.trim().split('.').collect();
            if path.iter().any(|p| p.is_empty()) {
                return Err(CoreError::Config {
                    layer: "--set".into(),
                    message: format!("bad key `{key}`"),
                });
            }
            let value = format!("v = {raw}")
                .parse::<toml::Table>()
                .ok()
                .and_then(|mut t| t.remove("v"))
                .unwrap_or_else(|| toml::Value::String(raw.to_string()));
            insert_path(&mut table, &path, value).map_err(|message| CoreError::Config {
                layer: "--set".into(),
                message,
            })?;
        }
        Ok(Self {
            name: "--set".into(),
            table,
        })
    }
}

fn insert_path(
    table: &mut toml::Table,
    path: &[&str],
    value: toml::Value,
) -> std::result::Result<(), String> {
    match path {
        [] => Err("empty key".into()),
        [last] => {
            table.insert((*last).to_string(), value);
            Ok(())
        }
        [head, rest @ ..] => {
            let entry = table
                .entry((*head).to_string())
                .or_insert_with(|| toml::Value::Table(toml::Table::new()));
            match entry {
                toml::Value::Table(t) => insert_path(t, rest, value),
                _ => Err(format!("`{head}` is not a table")),
            }
        }
    }
}

/// Deep merge: tables merge key by key, everything else (scalars, arrays) is replaced.
pub fn merge_into(base: &mut toml::Table, over: toml::Table) {
    for (k, v) in over {
        match (base.get_mut(&k), v) {
            (Some(toml::Value::Table(b)), toml::Value::Table(o)) => merge_into(b, o),
            (_, v) => {
                base.insert(k, v);
            }
        }
    }
}

fn describe(e: &serde_path_to_error::Error<toml::de::Error>) -> String {
    let path = e.path().to_string();
    let msg = e.inner().to_string();
    let msg = msg.trim();
    if path == "." || path.is_empty() {
        msg.to_string()
    } else {
        format!("at `{path}`: {msg}")
    }
}

fn deserialize_config(layer: &str, table: toml::Table) -> Result<Config> {
    serde_path_to_error::deserialize(toml::Value::Table(table)).map_err(|e| CoreError::Config {
        layer: layer.to_string(),
        message: describe(&e),
    })
}

/// The merged configuration plus where it came from. Relative paths resolve against `root`
/// (the directory of `config.toml`).
#[derive(Debug, Clone)]
pub struct LoadedConfig {
    pub config: Config,
    pub root: PathBuf,
    pub layers: Vec<String>,
}

impl LoadedConfig {
    /// Load `path`, then `<stem>.local.toml` beside it if present, then `--set` overrides.
    pub fn load(path: &Path, overrides: &[String]) -> Result<Self> {
        let path = std::path::absolute(path).at(path)?;
        let root = path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let mut layers = Vec::new();
        let text = std::fs::read_to_string(&path).at(&path)?;
        layers.push(Layer::parse(path.display().to_string(), &text)?);
        let local = local_path(&path);
        if local.exists() {
            let text = std::fs::read_to_string(&local).at(&local)?;
            layers.push(Layer::parse(local.display().to_string(), &text)?);
        }
        if !overrides.is_empty() {
            layers.push(Layer::from_overrides(overrides)?);
        }
        Self::from_layers(root, layers)
    }

    /// Merge layers in order. Each layer is first checked alone (all fields have defaults), so an
    /// unknown key or bad type is reported against the layer that contains it.
    pub fn from_layers(root: PathBuf, layers: Vec<Layer>) -> Result<Self> {
        let mut merged = toml::Table::new();
        let mut names = Vec::new();
        for layer in layers {
            deserialize_config(&layer.name, layer.table.clone())?;
            names.push(layer.name);
            merge_into(&mut merged, layer.table);
        }
        let config = deserialize_config("merged config", merged)?;
        validate(&config)?;
        Ok(Self {
            config,
            root,
            layers: names,
        })
    }

    /// A single in-memory layer (tests and fixtures).
    pub fn from_toml_str(root: PathBuf, text: &str) -> Result<Self> {
        Self::from_layers(root, vec![Layer::parse("<string>", text)?])
    }

    pub fn resolve(&self, p: &Path) -> PathBuf {
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.root.join(p)
        }
    }

    pub fn build_dir(&self) -> PathBuf {
        self.resolve(&self.config.paths.build_dir)
    }

    pub fn models_dir(&self) -> PathBuf {
        self.resolve(&self.config.paths.models_dir)
    }

    /// The section of pass `name`; a missing section means all defaults.
    pub fn pass_section(&self, name: &str) -> Result<PassSection> {
        let mut options = self.config.passes.get(name).cloned().unwrap_or_default();
        let resources = match options.remove("resources") {
            None => PassResources::default(),
            Some(v) => serde_path_to_error::deserialize::<_, PassResources>(v).map_err(|e| {
                CoreError::PassConfig {
                    pass: name.to_string(),
                    message: format!("resources: {}", describe(&e)),
                }
            })?,
        };
        if resources.threads == 0 || !positive(resources.memory_gb) || !positive(resources.timeout_min) {
            return Err(CoreError::PassConfig {
                pass: name.to_string(),
                message: "resources: threads, memory_gb and timeout_min must be positive".into(),
            });
        }
        Ok(PassSection {
            pass: name.to_string(),
            options,
            resources,
        })
    }

    /// Catalog keys: the subdirectories of `models_dir` that hold a `recipe.py` (spec §2), sorted.
    pub fn model_catalog(&self) -> Result<Vec<String>> {
        let dir = self.models_dir();
        if !dir.is_dir() {
            return Ok(Vec::new());
        }
        let mut keys = Vec::new();
        for entry in std::fs::read_dir(&dir).at(&dir)? {
            let entry = entry.at(&dir)?;
            if entry.path().join("recipe.py").is_file() {
                keys.push(entry.file_name().to_string_lossy().into_owned());
            }
        }
        keys.sort();
        Ok(keys)
    }

    /// The model set of this run: `--models` if given, else `models.keys`. Unknown keys are an error.
    pub fn resolve_models(&self, cli: Option<&[String]>) -> Result<Vec<String>> {
        let catalog = self.model_catalog()?;
        let wanted: Vec<String> = match (cli, &self.config.models.keys) {
            (Some(list), _) => list.to_vec(),
            (None, ModelKeys::List(list)) => list.clone(),
            (None, ModelKeys::All) => return Ok(catalog),
        };
        let mut out = Vec::new();
        for k in wanted {
            if !catalog.contains(&k) {
                return Err(CoreError::Config {
                    layer: "models".into(),
                    message: format!(
                        "unknown model key `{k}` (catalog: {})",
                        self.models_dir().display()
                    ),
                });
            }
            out.push(k);
        }
        out.sort();
        out.dedup();
        Ok(out)
    }
}

/// `config.toml` → `config.local.toml`.
pub fn local_path(path: &Path) -> PathBuf {
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    path.with_file_name(format!("{stem}.local.toml"))
}

/// `x > 0`, false for NaN.
fn positive(x: f64) -> bool {
    x > 0.0
}

fn validate(c: &Config) -> Result<()> {
    let bad = |message: &str| {
        Err(CoreError::Config {
            layer: "merged config".into(),
            message: message.into(),
        })
    };
    if c.pipeline.max_parallel_passes == 0 {
        return bad("pipeline.max_parallel_passes must be >= 1");
    }
    if !positive(c.resources.memory_budget_gb) {
        return bad("resources.memory_budget_gb must be > 0");
    }
    if c.resources.threads_total == 0 {
        return bad("resources.threads_total must be >= 1");
    }
    if !positive(c.tools.host.watchdog_interval_s) || !positive(c.tools.eda.watchdog_interval_s) {
        return bad("watchdog intervals must be > 0");
    }
    if c.tools.eda.max_licenses == 0 {
        return bad("tools.eda.max_licenses must be >= 1");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layers(texts: &[(&str, &str)]) -> Result<LoadedConfig> {
        let ls = texts
            .iter()
            .map(|(n, t)| Layer::parse(*n, t))
            .collect::<Result<Vec<_>>>()?;
        LoadedConfig::from_layers(PathBuf::from("/repo"), ls)
    }

    #[test]
    fn defaults_apply() {
        let c = layers(&[("a", "")]).unwrap();
        assert_eq!(c.config.pipeline.max_parallel_passes, 2);
        assert_eq!(c.build_dir(), PathBuf::from("/repo/build"));
    }

    #[test]
    fn merge_order_local_then_set() {
        let mut ls = vec![
            Layer::parse("base", "[resources]\nmemory_budget_gb = 32\nthreads_total = 16").unwrap(),
            Layer::parse("local", "[resources]\nmemory_budget_gb = 8").unwrap(),
        ];
        ls.push(Layer::from_overrides(&["resources.threads_total=3".into()]).unwrap());
        let c = LoadedConfig::from_layers("/r".into(), ls).unwrap();
        assert_eq!(c.config.resources.memory_budget_gb, 8.0);
        assert_eq!(c.config.resources.threads_total, 3);
    }

    #[test]
    fn unknown_key_names_layer_and_path() {
        let err = layers(&[("base", ""), ("local", "[pipeline]\nmax_paralel = 3")]).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("local"), "{msg}");
        assert!(msg.contains("max_paralel"), "{msg}");
    }

    #[test]
    fn set_value_falls_back_to_string() {
        let l = Layer::from_overrides(&["passes.x.greeting=hi there".into(), "passes.x.n=3".into()]).unwrap();
        let x = l.table["passes"]["x"].as_table().unwrap();
        assert_eq!(x["greeting"].as_str(), Some("hi there"));
        assert_eq!(x["n"].as_integer(), Some(3));
    }

    #[test]
    fn pass_section_splits_resources_strictly() {
        let c = layers(&[(
            "a",
            "[passes.p]\nk = 1\n[passes.p.resources]\nthreads = 2\nmemory_gb = 1",
        )])
        .unwrap();
        let s = c.pass_section("p").unwrap();
        assert_eq!(s.resources.threads, 2);
        assert!(s.options.contains_key("k") && !s.options.contains_key("resources"));
        let c = layers(&[("a", "[passes.p.resources]\nthreds = 2")]).unwrap();
        assert!(c.pass_section("p").unwrap_err().to_string().contains("threds"));
    }

    #[test]
    fn model_keys_forms() {
        let c = layers(&[("a", "[models]\nkeys = [\"b\", \"a\"]")]).unwrap();
        assert_eq!(
            c.config.models.keys,
            ModelKeys::List(vec!["b".into(), "a".into()])
        );
        assert!(layers(&[("a", "[models]\nkeys = \"some\"")]).is_err());
    }
}
