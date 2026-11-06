//! `chipc doctor` (design §3.6, §8): prove the environment works by making every tool do one
//! small real job, build the liberty `.db` files, and record the evidence in `build/.env/`.
//!
//! Every external call goes through a `ToolRunner` (host: git/docker CLI/df; Docker: the image;
//! EDA: `eda-run`). `--dry` prints each command instead of running it and writes nothing.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use cc_core::canon;
use cc_core::config::Config;
use cc_core::manifest::{InputRecord, MANIFEST_SCHEMA_VERSION};
use cc_core::runner::{
    Caps, DockerRunner, EdaRunner, EdaSettings, EdaTool, Invocation, Mount, Runners, ToolLog, ToolOutcome,
    mem_available_bytes,
};
use cc_core::{
    ArtifactKind, ArtifactStore, CancelToken, Check, CheckStatus, CoreError, LoadedConfig, Manifest, PassDir,
    PassResources, PassStatus, VerifyReport,
};

use crate::env::pdk_db::{self, LibertyLib};
use crate::error::PipelineError;

/// `build/.env/`.
pub const ENV_DIR: &str = "env";
pub const PDK_DB: ArtifactKind = ArtifactKind("pdk_db");
pub const SELFTEST: ArtifactKind = ArtifactKind("selftest");
pub const DC_TINY: ArtifactKind = ArtifactKind("dc_tiny");
/// `out/pdk_lib/<library>.lib`: the ASAP7 group libraries of one VT/corner merged into one (`liberty.rs`).
pub const PDK_LIB: ArtifactKind = ArtifactKind("pdk_lib");

const DOCTOR_VERSION: u32 = 1;
/// The steps `cc-selftest` reports (docker/selftest.sh).
pub const SELFTEST_STEPS: [&str; 10] = [
    "xls",
    "xls_equiv",
    "rtl_tools",
    "smt",
    "grami",
    "allo",
    "xls_extern",
    "xls_float_lib",
    "flopoco_ghdl",
    "hardfloat",
];
/// Version query per licensed tool (flags observed on this host, docs/tools/{synopsys,cadence}.md).
const EDA_VERSION_PROBES: [(&str, &[&str]); 6] = [
    ("dc_shell", &["-version"]),
    ("lc_shell", &["-version"]),
    ("fm_shell", &["-version"]),
    ("pt_shell", &["-version"]),
    ("vcs", &["-full64", "-ID"]),
    ("xrun", &["-version"]),
];
const GIB: f64 = (1u64 << 30) as f64;

// Doctor's own caps: tiny jobs only (design §8).
const HOST_CAPS: Caps = Caps {
    memory_gb: 1.0,
    cpus: 1.0,
    timeout_s: 120.0,
    pids_limit: 256,
};
const DOCKER_CAPS: Caps = Caps {
    memory_gb: 4.0,
    cpus: 2.0,
    timeout_s: 900.0,
    pids_limit: 1024,
};
const EDA_PROBE_CAPS: Caps = Caps {
    memory_gb: 4.0,
    cpus: 1.0,
    timeout_s: 600.0,
    pids_limit: 1024,
};
const LC_CAPS: Caps = Caps {
    memory_gb: 8.0,
    cpus: 1.0,
    timeout_s: 3600.0,
    pids_limit: 1024,
};
const DC_CAPS: Caps = Caps {
    memory_gb: 8.0,
    cpus: 2.0,
    timeout_s: 1800.0,
    pids_limit: 1024,
};

#[derive(Debug, Clone, Copy, Default)]
pub struct DoctorOptions {
    pub dry: bool,
    pub skip_licenses: bool,
}

#[derive(Debug, Clone, Copy)]
enum Via {
    Host,
    Docker,
    Eda,
}

impl Via {
    fn word(self) -> &'static str {
        match self {
            Via::Host => "host",
            Via::Docker => "docker",
            Via::Eda => "eda",
        }
    }
}

/// One kit as `chipc doctor` found it: its manifest, the library the flows link (the merged one for a
/// kit with groups) and every library compiled to `.db`.
struct KitLibs {
    pdk: cc_core::pdk::Pdk,
    link: LibertyLib,
    all: Vec<LibertyLib>,
    /// A shipped Liberty file with the units (a merged library keeps its first group's header).
    units: PathBuf,
}

/// Run every check; write `build/.env/{verify,manifest}.json` unless `dry`. The report's `passed()`
/// is the exit status.
pub fn doctor(
    cfg: &LoadedConfig,
    runners: &Runners,
    opts: DoctorOptions,
    out: &mut dyn Write,
) -> Result<VerifyReport, PipelineError> {
    let dir = ArtifactStore::new(cfg.build_dir()).pass_dir(ENV_DIR);
    if !opts.dry {
        let logs = dir.logs_dir();
        if logs.exists() {
            std::fs::remove_dir_all(&logs).map_err(|source| CoreError::Io {
                path: logs.clone(),
                source,
            })?;
        }
    }
    let mut d = Doctor {
        cfg,
        runners,
        opts,
        out,
        log: ToolLog::new(ENV_DIR, dir.logs_dir()),
        dir,
        cancel: CancelToken::new(),
        docker_fmt: DockerRunner::from_config(cfg.config.tools.docker.clone()),
        eda_fmt: EdaRunner::new(EdaSettings::from_config(cfg)),
        report: VerifyReport::new(ENV_DIR),
        versions: BTreeMap::new(),
        inputs: Vec::new(),
    };
    let started_at = canon::now_utc();
    let t0 = Instant::now();
    d.host();
    d.submodules();
    d.docker();
    d.eda_versions();
    let kits = d.pdk();
    if opts.skip_licenses {
        for k in &kits {
            d.push(Check::skip(format!("dc.{}", k.pdk.name()), "--skip-licenses"));
        }
    } else {
        d.dc_tiny(&kits);
    }
    d.finish(&started_at, t0)?;
    Ok(d.report)
}

struct Doctor<'a> {
    cfg: &'a LoadedConfig,
    runners: &'a Runners,
    opts: DoctorOptions,
    out: &'a mut dyn Write,
    dir: PassDir,
    log: ToolLog,
    cancel: CancelToken,
    docker_fmt: DockerRunner,
    eda_fmt: EdaRunner,
    report: VerifyReport,
    versions: BTreeMap<String, String>,
    inputs: Vec<InputRecord>,
}

impl Doctor<'_> {
    fn config(&self) -> &Config {
        &self.cfg.config
    }

    fn push(&mut self, c: Check) {
        let word = match c.status {
            CheckStatus::Pass => "pass",
            CheckStatus::Fail => "FAIL",
            CheckStatus::Skip => "skip",
        };
        let note = c.evidence.notes.first().map(String::as_str).unwrap_or("");
        let _ = writeln!(self.out, "{word:4}  {:<28} {note}", c.name);
        self.report.push(c);
    }

    /// Run `inv` through the runner (or print it under `--dry`). `None` after pushing a skip/fail
    /// check named `name`.
    fn exec(&mut self, name: &str, via: Via, inv: &Invocation, caps: Caps) -> Option<ToolOutcome> {
        if self.opts.dry {
            let argv = match via {
                Via::Host => std::iter::once(inv.program.clone())
                    .chain(inv.args.iter().cloned())
                    .collect(),
                Via::Docker => {
                    let mut a = vec!["docker".to_string()];
                    a.extend(self.docker_fmt.docker_args("chipc-env-dry", None, inv, &caps));
                    a
                }
                Via::Eda => match self.eda_fmt.eda_command(inv) {
                    Ok(c) => std::iter::once(c.program).chain(c.args).collect(),
                    Err(e) => vec![format!("<{e}>")],
                },
            };
            let cwd = inv
                .cwd
                .as_ref()
                .map(|p| format!("  (cwd {})", p.display()))
                .unwrap_or_default();
            let _ = writeln!(self.out, "[{}] {}{cwd}", via.word(), shell_join(&argv));
            self.push(Check::skip(name, "dry run"));
            return None;
        }
        let runner = match via {
            Via::Host => &self.runners.host,
            Via::Docker => &self.runners.docker,
            Via::Eda => &self.runners.eda,
        };
        match runner.run(inv, &caps, &self.log, &self.cancel) {
            Ok(o) => Some(o),
            Err(e) => {
                self.push(Check::new(name, false).note(e.to_string()));
                None
            }
        }
    }

    /// Write a generated input file (scripts, tiny RTL); only printed under `--dry`.
    fn write(&mut self, path: &Path, contents: &str) -> bool {
        if self.opts.dry {
            let _ = writeln!(self.out, "[write] {}", path.display());
            return true;
        }
        canon::write_atomic(path, contents.as_bytes()).is_ok()
    }

    fn rel(&self, p: &Path) -> String {
        p.strip_prefix(self.dir.path())
            .map(|r| r.display().to_string())
            .unwrap_or_else(|_| p.display().to_string())
    }

    // -- host ---------------------------------------------------------------------------------

    fn host(&mut self) {
        let budget = self.config().resources.memory_budget_gb;
        let c = match mem_available_bytes() {
            Some(b) => {
                let gb = b as f64 / GIB;
                let mut c = Check::new("host.memory", true)
                    .note(format!("MemAvailable {gb:.0} GB, memory_budget_gb {budget}"))
                    .count("mem_available_gb", gb as u64);
                if gb < budget {
                    c = c.note("warning: free memory is below resources.memory_budget_gb");
                }
                c
            }
            None => Check::new("host.memory", false).note("cannot read /proc/meminfo"),
        };
        self.push(c);

        let Some(scratch) = self
            .config()
            .paths
            .scratch_dir
            .as_ref()
            .map(|p| self.cfg.resolve(p))
        else {
            self.push(
                Check::new("host.scratch_dir", false)
                    .note("paths.scratch_dir is not set (config.local.toml)"),
            );
            return;
        };
        if !self.opts.dry {
            let probe = scratch.join("tmp").join(".doctor-probe");
            let writable = canon::write_atomic(&probe, b"ok").is_ok();
            let _ = std::fs::remove_file(&probe);
            if !writable {
                self.push(
                    Check::new("host.scratch_dir", false)
                        .note(format!("{} is not writable", scratch.display())),
                );
                return;
            }
        }
        let inv = Invocation::new("df", "df").args(["-Pk".to_string(), scratch.display().to_string()]);
        let Some(o) = self.exec("host.scratch_dir", Via::Host, &inv, HOST_CAPS) else {
            return;
        };
        let free_gb = o
            .stdout_string()
            .ok()
            .and_then(|s| s.lines().nth(1)?.split_whitespace().nth(3)?.parse::<u64>().ok())
            .map(|kb| kb / (1 << 20));
        let mut c = Check::new("host.scratch_dir", o.success() && free_gb.is_some())
            .note(format!(
                "{} ({} GB free)",
                scratch.display(),
                free_gb.unwrap_or(0)
            ))
            .count("free_gb", free_gb.unwrap_or(0));
        if free_gb.is_some_and(|g| g < 50) {
            c = c.note("warning: less than 50 GB free on the scratch disk");
        }
        self.push(c);
    }

    // -- submodules -----------------------------------------------------------------------------

    fn submodules(&mut self) {
        let root = self.cfg.root.clone();
        let tp = self.cfg.resolve(&self.config().paths.third_party_dir);
        let rel_tp = tp
            .strip_prefix(&root)
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| tp.display().to_string());
        let gitmodules = std::fs::read_to_string(root.join(".gitmodules")).unwrap_or_default();
        let t = &self.config().tools;
        let pins = [
            ("xls", t.xls.commit.clone()),
            ("allo", t.allo.commit.clone()),
            ("grami", t.grami.commit.clone()),
            ("hardfloat", t.hardfloat.commit.clone()),
        ];
        for (name, pin) in pins {
            let check = format!("submodule.{name}");
            let path = format!("{rel_tp}/{name}");
            let shallow = gitmodules_shallow(&gitmodules, &path);
            let inv = git(&tp.join(name), &["rev-parse", "HEAD"]);
            let Some(o) = self.exec(&check, Via::Host, &inv, HOST_CAPS) else {
                continue;
            };
            let head = o.stdout_string().unwrap_or_default().trim().to_string();
            let ok = o.success() && !pin.is_empty() && head == pin && shallow;
            let mut c = Check::new(check, ok).note(format!("HEAD {head}, pin {pin}"));
            if !shallow {
                c = c.note(format!(".gitmodules: {path} is not `shallow = true`"));
            }
            self.versions.insert(format!("{name}_commit"), head);
            self.push(c);
        }

        let llvm_pin = self.config().tools.allo.llvm_commit.clone();
        let inv = git(&tp.join("allo"), &["ls-tree", "HEAD", "externals/llvm-project"]);
        if let Some(o) = self.exec("submodule.allo.llvm", Via::Host, &inv, HOST_CAPS) {
            let rec = o.stdout_string().unwrap_or_default();
            let sha = rec.split_whitespace().nth(2).unwrap_or("").to_string();
            self.push(
                Check::new("submodule.allo.llvm", o.success() && sha == llvm_pin)
                    .note(format!("Allo records LLVM {sha}, pin {llvm_pin}")),
            );
        }

        // Every patch set `third_party/patches/<submodule>/` must apply to its submodule as the image build
        // applies it: in file order, each patch on top of the ones before. `git apply --check` checks every
        // patch against the unpatched tree, so the series is applied to a copy of the files it touches
        // (`patches/<submodule>/` in doctor's directory; the submodule is never written).
        let patch_root = tp.join("patches");
        let sets = patch_sets(&patch_root);
        if sets.is_empty() {
            self.push(
                Check::new("submodule.patches", false)
                    .note(format!("no patch sets in {}", patch_root.display())),
            );
        }
        for (name, patches) in sets {
            let check = format!("submodule.{name}.patches");
            if patches.is_empty() {
                self.push(
                    Check::new(check, false).note(format!("no patches in {}/{name}", patch_root.display())),
                );
                continue;
            }
            let copy = self.dir.path().join("patches").join(&name);
            if !self.opts.dry
                && let Err(e) = copy_patched_files(&tp.join(&name), &patches, &copy)
            {
                self.push(Check::new(check, false).note(format!("cannot copy the patched files: {e}")));
                continue;
            }
            for p in &patches {
                // GIT_CEILING_DIRECTORIES: the copy is a plain directory, even inside a checkout.
                let inv = Invocation::new("git", "git")
                    .args(["apply".to_string(), p.display().to_string()])
                    .cwd(&copy)
                    .env(
                        "GIT_CEILING_DIRECTORIES",
                        copy.parent().unwrap_or(&copy).display().to_string(),
                    );
                let Some(o) = self.exec(&check, Via::Host, &inv, HOST_CAPS) else {
                    continue;
                };
                let file = p
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let ok = o.success();
                self.push(Check::new(check.clone(), ok).note(format!("{file} applies")));
                if !ok {
                    break; // the later patches build on this one
                }
            }
        }
    }

    // -- docker -----------------------------------------------------------------------------------

    fn docker(&mut self) {
        let image = self.docker_fmt.default_image();
        let inv = Invocation::new("docker", "docker").args(["version", "--format", "{{.Server.Version}}"]);
        let daemon = self.exec("docker.daemon", Via::Host, &inv, HOST_CAPS);
        if let Some(o) = &daemon {
            let v = o.stdout_string().unwrap_or_default().trim().to_string();
            self.push(Check::new("docker.daemon", o.success()).note(format!("server {v}")));
            self.versions.insert("docker".into(), v);
        }
        let inv =
            Invocation::new("docker", "docker").args(["image", "inspect", "--format", "{{.Id}}", &image]);
        let present = match self.exec("docker.image", Via::Host, &inv, HOST_CAPS) {
            Some(o) => {
                let id = o.stdout_string().unwrap_or_default().trim().to_string();
                let ok = o.success() && !id.is_empty();
                let note = if ok {
                    format!("{image} {id}")
                } else {
                    format!("{image} is missing: build it with docker/build.sh")
                };
                self.push(Check::new("docker.image", ok).note(note));
                if ok {
                    self.versions.insert("image".into(), format!("{image}@{id}"));
                }
                ok
            }
            None => self.opts.dry,
        };
        if !present {
            for n in ["image.pins", "image.tools"] {
                self.push(Check::skip(n, "no image"));
            }
            for s in SELFTEST_STEPS {
                self.push(Check::skip(format!("image.selftest.{s}"), "no image"));
            }
            return;
        }

        // Pins recorded at build time == config.
        let inv = Invocation::new("versions", "cat").arg("/opt/cc/versions.json");
        let recorded = self
            .exec("image.pins", Via::Docker, &inv, DOCKER_CAPS)
            .and_then(|o| serde_json::from_str::<serde_json::Value>(&o.stdout_string().ok()?).ok());
        if let Some(rec) = &recorded {
            let mut c = Check::new("image.pins", true);
            let mut bad = 0;
            for (k, want) in expected_pins(self.config()) {
                let have = rec["pins"][k].as_str().unwrap_or("");
                if have != want {
                    bad += 1;
                    c = c.note(format!("{k}: image {have:?}, config {want:?}"));
                }
            }
            c = c.count("mismatches", bad);
            if bad > 0 {
                c = c.with_status(CheckStatus::Fail);
            } else {
                c = c.note("all pins match config.toml");
            }
            self.push(c);
        } else if !self.opts.dry {
            self.push(Check::new("image.pins", false).note("cannot read /opt/cc/versions.json"));
        }

        // Every tool still answers, with the version recorded at build time.
        let inv = Invocation::new("cc-probe", "cc-probe");
        if let Some(o) = self.exec("image.tools", Via::Docker, &inv, DOCKER_CAPS) {
            let now: serde_json::Value =
                serde_json::from_str(&o.stdout_string().unwrap_or_default()).unwrap_or_default();
            let mut c = Check::new("image.tools", o.success());
            let mut bad = 0;
            let tools = recorded.as_ref().map(|r| r["tools"].clone()).unwrap_or_default();
            let names: BTreeSet<String> = tools
                .as_object()
                .map(|m| m.keys().cloned().collect())
                .unwrap_or_default();
            for name in &names {
                let want = tools[name]["version"].as_str().unwrap_or("");
                let have = now["tools"][name]["version"].as_str().unwrap_or("");
                if want.is_empty() || have != want {
                    bad += 1;
                    c = c.note(format!("{name}: now {have:?}, recorded {want:?}"));
                } else {
                    self.versions.insert(name.clone(), have.to_string());
                }
            }
            c = c.count("tools", names.len() as u64).count("mismatches", bad);
            if bad > 0 || names.is_empty() {
                c = c.with_status(CheckStatus::Fail);
            } else {
                c = c.note(format!(
                    "{} tools answer with their recorded versions",
                    names.len()
                ));
            }
            self.push(c);
        }

        // One real job per tool, outputs kept as evidence in out/selftest/.
        let kd = self.dir.kind(SELFTEST);
        if !self.opts.dry {
            let _ = std::fs::remove_dir_all(kd.dir());
            if kd.create().is_err() {
                self.push(Check::new("image.selftest", false).note("cannot create out/selftest"));
                return;
            }
        }
        let inv = Invocation::new("cc-selftest", "cc-selftest")
            .arg("/work/out")
            .cwd("/work/out")
            .mount(Mount {
                src: kd.dir().to_path_buf(),
                dst: "/work/out".into(),
                readonly: false,
            });
        let Some(o) = self.exec("image.selftest", Via::Docker, &inv, DOCKER_CAPS) else {
            return;
        };
        let stdout = o.stdout_string().unwrap_or_default();
        let passed: BTreeSet<&str> = stdout
            .lines()
            .filter_map(|l| l.strip_prefix("cc-selftest: pass "))
            .collect();
        let evidence: BTreeMap<&str, &[&str]> = BTreeMap::from([
            ("xls", &["add8.cc", "add8.ir", "add8.opt.ir", "add8.v"][..]),
            ("xls_equiv", &["add8.ir", "add8.opt.ir"][..]),
            ("rtl_tools", &["add8.v"][..]),
            ("smt", &["q.smt2"][..]),
            ("grami", &["two_paths.lg", "Output.txt"][..]),
            ("allo", &["allo_add.py"][..]),
            (
                "xls_extern",
                &["allo_extern.py", "allo_extern/test_block.cpp", "allo_extern.v"][..],
            ),
            ("xls_float_lib", &["sincos.v", "sincos.stat"][..]),
            ("flopoco_ghdl", &["fpexp.vhdl", "fpexp.stat"][..]),
            ("hardfloat", &["hf_add_f16.v", "hf_add_f16.stat"][..]),
        ]);
        for step in SELFTEST_STEPS {
            let mut c = Check::new(format!("image.selftest.{step}"), passed.contains(step));
            for f in evidence.get(step).copied().unwrap_or_default() {
                c = c.path(format!("out/selftest/{f}"));
            }
            if step == "grami"
                && let Some(n) = grami_pattern_count(&kd.dir().join("Output.txt"))
            {
                c = c
                    .count("frequent_patterns", n)
                    .note(format!("{n} frequent patterns (freq 2)"));
            }
            if !passed.contains(step) {
                c = c.note(format!("see {}", o.stderr.display()));
            }
            self.push(c);
        }
    }

    // -- EDA versions ------------------------------------------------------------------------------

    fn eda_versions(&mut self) {
        for (prog, args) in EDA_VERSION_PROBES {
            let name = format!("eda.version.{prog}");
            let inv = Invocation::new(prog, prog)
                .args(args.iter().copied())
                .cwd(self.dir.path().join("eda").join("version").join(prog));
            let Some(o) = self.exec(&name, Via::Eda, &inv, EDA_PROBE_CAPS) else {
                continue;
            };
            let text = outcome_text(&o);
            match version_line(&text) {
                Some(v) => {
                    self.versions.insert(prog.to_string(), v.clone());
                    self.push(Check::new(name, true).note(v));
                }
                None => self.push(
                    Check::new(name, false).note(format!("no version line; see {}", o.stdout.display())),
                ),
            }
        }
    }

    // -- PDKs and .db ------------------------------------------------------------------------------

    /// Every kit's manifest (`pdks/<kit>.toml`) names files that exist; group libraries are merged; every
    /// Liberty file compiles to `.db` (cached by `.lib` hash + LC version).
    fn pdk(&mut self) -> Vec<KitLibs> {
        let pdks = match cc_core::pdk::load_all(self.cfg) {
            Ok(p) if !p.is_empty() => p,
            Ok(_) => {
                self.push(Check::new("pdk.manifests", false).note(format!(
                    "no manifests in {}",
                    cc_core::pdk::manifest_dir(self.cfg).display()
                )));
                return Vec::new();
            }
            Err(e) => {
                self.push(Check::new("pdk.manifests", false).note(e.to_string()));
                return Vec::new();
            }
        };
        let mut kits = Vec::new();
        for (name, pdk) in pdks {
            let check = format!("pdk.{name}");
            let found = pdk.liberty_files().and_then(|ps| {
                ps.iter()
                    .map(|p| pdk_db::read_library(p))
                    .collect::<Result<Vec<_>, _>>()
            });
            let (found, lefs) = match (found, pdk.lef_files()) {
                (Ok(f), Ok(l)) => (f, l),
                (Err(e), _) | (_, Err(e)) => {
                    self.push(Check::new(check, false).note(e.to_string()));
                    continue;
                }
            };
            let note = if pdk.manifest.groups.is_empty() {
                format!(
                    "library {}",
                    found
                        .iter()
                        .map(|l| l.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            } else {
                format!(
                    "{} {}: {} libraries",
                    pdk.manifest.vt,
                    pdk.manifest.corner,
                    found.len()
                )
            };
            let mut c = Check::new(check, true)
                .count("libraries", found.len() as u64)
                .note(note)
                .path(pdk.file.display().to_string());
            for p in found.iter().map(|l| &l.path).chain(&lefs) {
                c = c.path(p.display().to_string());
            }
            self.push(c);
            let link = match pdk.merged_name() {
                Some(merged) => match self.merge_groups(&name, &merged, &found) {
                    Some(m) => m,
                    None => continue,
                },
                None => found[0].clone(),
            };
            let units = found[0].path.clone();
            let mut all = found;
            if !all.contains(&link) {
                all.push(link.clone());
            }
            kits.push(KitLibs {
                pdk,
                link,
                all,
                units,
            });
        }
        let libs: Vec<LibertyLib> = kits.iter().flat_map(|k| k.all.iter().cloned()).collect();
        if libs.is_empty() {
            self.push(Check::new("pdk.db", false).note("no Liberty files"));
            return kits;
        }

        for l in &libs {
            if let Ok(hash) = canon::hash_file(&l.path) {
                self.inputs.push(InputRecord {
                    kind: format!("liberty:{}", l.name),
                    producer: l.path.display().to_string(),
                    hash,
                });
            }
        }
        let kd = self.dir.kind(PDK_DB);
        let lc_version = self.versions.get("lc_shell").cloned();
        let cached = !self.opts.dry
            && self.dir.read_manifest().ok().flatten().is_some_and(|m| {
                m.tool_versions.get("lc_shell") == lc_version.as_ref()
                    && self.inputs.iter().all(|i| m.inputs.contains(i))
            })
            && libs.iter().all(|l| non_empty(&kd.dir().join(l.db_file())));
        if cached {
            self.push(
                Check::new("pdk.db", true)
                    .count("db_files", libs.len() as u64)
                    .note(format!(
                        "{} .db files up to date (same .lib hashes and lc_shell)",
                        libs.len()
                    )),
            );
            return kits;
        }

        let job = self.dir.path().join("eda").join("lc");
        let script = job.join("make_db.tcl");
        if !self.write(&script, &pdk_db::make_db_tcl(&libs)) {
            self.push(Check::new("pdk.db", false).note(format!("cannot write {}", script.display())));
            return kits;
        }
        let inv = Invocation::new("lc_shell", EdaTool::Lc.program())
            .args(EdaTool::Lc.batch_args("make_db.tcl"))
            .cwd(&job);
        let Some(o) = self.exec("pdk.db", Via::Eda, &inv, LC_CAPS) else {
            return kits;
        };
        let text = outcome_text(&o);
        let errors = text
            .lines()
            .filter(|l| l.trim_start().starts_with("Error"))
            .count() as u64;
        let warnings = text
            .lines()
            .filter(|l| l.trim_start().starts_with("Warning"))
            .count() as u64;
        let _ = kd.create();
        let mut copied = 0u64;
        for l in &libs {
            let src = job.join(l.db_file());
            if non_empty(&src) && std::fs::copy(&src, kd.dir().join(l.db_file())).is_ok() {
                copied += 1;
            }
        }
        let ok = o.success() && errors == 0 && copied == libs.len() as u64;
        let mut c = Check::new("pdk.db", ok)
            .count("db_files", copied)
            .count("lc_errors", errors)
            .count("lc_warnings", warnings)
            .note(format!("{copied}/{} .db files written to out/pdk_db", libs.len()))
            .path(self.rel(&job.join("make_db.tcl")));
        for l in &libs {
            c = c.path(format!("out/pdk_db/{}", l.db_file()));
        }
        self.push(c);
        kits
    }

    /// A kit's group libraries merged into one (`out/pdk_lib/<merged>.lib`), which LC then compiles with
    /// the rest: DC's library analysis needs a library with inverters and flops, Yosys reads one file.
    fn merge_groups(&mut self, kit: &str, name: &str, found: &[LibertyLib]) -> Option<LibertyLib> {
        let check = format!("pdk.{kit}_merged");
        let texts: Result<Vec<String>, String> = found
            .iter()
            .map(|l| std::fs::read_to_string(&l.path).map_err(|e| format!("{}: {e}", l.path.display())))
            .collect();
        let merged = texts.and_then(|t| {
            let refs: Vec<&str> = t.iter().map(String::as_str).collect();
            super::liberty::merge(name, &refs)
        });
        let m = match merged {
            Ok(m) => m,
            Err(e) => {
                self.push(Check::new(check, false).note(e));
                return None;
            }
        };
        let kd = self.dir.kind(PDK_LIB);
        let path = kd.dir().join(format!("{name}.lib"));
        if !self.opts.dry && (kd.create().is_err() || !self.write(&path, &m.text)) {
            self.push(Check::new(check, false).note(format!("cannot write {}", path.display())));
            return None;
        }
        let mut c = Check::new(check, true)
            .count("cells", m.cells as u64)
            .count("differing_groups", m.differing.len() as u64)
            .path(format!("out/pdk_lib/{name}.lib"))
            .note(format!(
                "{name}: {} cells from {} libraries",
                m.cells,
                found.len()
            ));
        if !m.differing.is_empty() {
            c = c.note(format!("kept the first library's {}", m.differing.join(", ")));
        }
        self.push(c);
        Some(LibertyLib {
            path,
            name: name.to_string(),
        })
    }

    // -- licensed synthesis -----------------------------------------------------------------------

    /// An 8-bit adder through `compile_ultra` on each kit's linked library at its default clock: proves the
    /// Synopsys license and that the `.db` files link.
    fn dc_tiny(&mut self, kits: &[KitLibs]) {
        let db_dir = self.dir.kind(PDK_DB).dir().to_path_buf();
        for k in kits {
            let label = k.pdk.name().to_string();
            let name = format!("dc.{label}");
            let dbs = vec![k.link.db_file()];
            if !self.opts.dry && !dbs.iter().all(|f| non_empty(&db_dir.join(f))) {
                self.push(Check::new(name, false).note("no .db files (see pdk.db)"));
                continue;
            }
            // The clock in the library's time unit (ns for Nangate45, ps for ASAP7: pdks.md §5).
            let period = match std::fs::read_to_string(&k.units)
                .map_err(|e| e.to_string())
                .and_then(|t| super::liberty::header(&t))
                .and_then(|h| super::liberty::time_unit_ps(h.get("time_unit").map_or("", String::as_str)))
            {
                Ok(unit) => format!("{}", k.pdk.manifest.clock_ps / unit),
                Err(e) => {
                    self.push(Check::new(name, false).note(format!("{}: time_unit: {e}", k.units.display())));
                    continue;
                }
            };
            let job = self.dir.path().join("eda").join(format!("dc_{label}"));
            let script = dc_script(&EdaTool::Dc.cap_preamble(&DC_CAPS), &db_dir, &dbs, &period);
            if !self.write(&job.join("add8.v"), ADD8_V) || !self.write(&job.join("syn.tcl"), &script) {
                self.push(Check::new(name, false).note("cannot write the DC job files"));
                continue;
            }
            let inv = Invocation::new("dc_shell", EdaTool::Dc.program())
                .args(EdaTool::Dc.batch_args("syn.tcl"))
                .cwd(&job);
            let Some(o) = self.exec(&name, Via::Eda, &inv, DC_CAPS) else {
                continue;
            };
            let text = outcome_text(&o);
            let errors = text
                .lines()
                .filter(|l| l.trim_start().starts_with("Error"))
                .count() as u64;
            let area = std::fs::read_to_string(job.join("area.rpt"))
                .ok()
                .and_then(|r| total_cell_area(&r));
            let netlist = job.join("add8_netlist.v");
            let ok = errors == 0 && area.is_some_and(|a| a > 0.0) && non_empty(&netlist);
            let kd = self.dir.kind(DC_TINY);
            let _ = kd.create();
            for (src, dst) in [
                ("area.rpt", format!("{label}_area.rpt")),
                ("add8_netlist.v", format!("{label}_netlist.v")),
            ] {
                let _ = std::fs::copy(job.join(src), kd.dir().join(dst));
            }
            self.push(
                Check::new(name, ok)
                    .note(format!(
                        "compile_ultra: total cell area {} um^2",
                        area.map_or("?".into(), |a| format!("{a:.4}"))
                    ))
                    .count("dc_errors", errors)
                    .path(format!("out/dc_tiny/{label}_area.rpt"))
                    .path(format!("out/dc_tiny/{label}_netlist.v")),
            );
        }
    }

    // -- results ---------------------------------------------------------------------------------

    fn finish(&mut self, started_at: &str, t0: Instant) -> Result<(), PipelineError> {
        let failed = self.report.failed_checks().len();
        let total = self.report.checks.len();
        if self.opts.dry {
            let _ = writeln!(
                self.out,
                "dry run: {total} checks planned, nothing executed or written"
            );
            return Ok(());
        }
        let mut outputs = Vec::new();
        for k in [PDK_DB, PDK_LIB, SELFTEST, DC_TINY] {
            let kd = self.dir.kind(k);
            if kd.exists() {
                outputs.push(kd.record()?);
            }
        }
        let config = canon::to_canonical_value(&self.config().tools).unwrap_or_default();
        let key_src =
            serde_json::json!({ "inputs": self.inputs, "tool_versions": self.versions, "config": config });
        let manifest = Manifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            pass: ENV_DIR.into(),
            pass_version: DOCTOR_VERSION,
            cache_key: canon::hash_bytes(&canon::to_canonical_compact(&key_src).unwrap_or_default()),
            status: if failed == 0 {
                PassStatus::Ok
            } else {
                PassStatus::Failed
            },
            inputs: self.inputs.clone(),
            outputs,
            config,
            resources: PassResources::default(),
            tool_versions: self.versions.clone(),
            models: Vec::new(),
            started_at: started_at.to_string(),
            finished_at: Some(canon::now_utc()),
            duration_ms: Some(t0.elapsed().as_millis() as u64),
            verify_ms: None,
            peak_rss_bytes: None,
            error: (failed > 0).then(|| format!("failed checks: {}", self.report.failed_checks().join(", "))),
        };
        self.dir.write_verify(&self.report)?;
        self.dir.write_manifest(&manifest)?;
        let _ = writeln!(
            self.out,
            "{}: {}/{total} checks pass; wrote {}",
            if failed == 0 {
                "doctor: green"
            } else {
                "doctor: NOT green"
            },
            total - failed,
            self.dir.verify_path().display()
        );
        Ok(())
    }
}

const ADD8_V: &str =
    "module add8 (input [7:0] a, input [7:0] b, output [8:0] y);\n  assign y = a + b;\nendmodule\n";

/// The tiny DC job (synopsys.md §4): combinational, so a virtual clock constrains I/O paths.
pub fn dc_script(preamble: &[String], db_dir: &Path, dbs: &[String], period: &str) -> String {
    let mut s = String::from("# Generated by chipc doctor: license + .db link check.\n");
    for l in preamble {
        s.push_str(l);
        s.push('\n');
    }
    s.push_str(&format!(
        "set search_path [list . {}]\nset target_library [list {}]\nset link_library [concat * $target_library]\n\
         analyze -format verilog add8.v\nelaborate add8\ncurrent_design add8\nlink\n\
         create_clock -name vclk -period {period}\nset_input_delay 0 -clock vclk [all_inputs]\n\
         set_output_delay 0 -clock vclk [all_outputs]\ncompile_ultra\n\
         report_area -nosplit > area.rpt\nreport_timing -nosplit > timing.rpt\n\
         write_file -format verilog -hierarchy -output add8_netlist.v\nquit\n",
        pdk_db::tcl_word(&db_dir.display().to_string()),
        dbs.join(" ")
    ));
    s
}

/// `Total cell area:  123.4` from `report_area`.
pub fn total_cell_area(report: &str) -> Option<f64> {
    report
        .lines()
        .find(|l| l.trim_start().starts_with("Total cell area:"))?
        .split_whitespace()
        .last()?
        .parse()
        .ok()
}

/// The first line that names a version: `… version …` (Synopsys, VCS) or `Tool: <name> <ver>` (xrun).
/// Cadence's `//` comment lines are skipped.
pub fn version_line(text: &str) -> Option<String> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.starts_with("//"))
        .find(|l| {
            let lower = l.to_ascii_lowercase();
            lower.contains("version") || lower.starts_with("tool:")
        })
        .map(|l| l.split_whitespace().collect::<Vec<_>>().join(" "))
}

/// Whether `.gitmodules` declares `path` with `shallow = true`.
pub fn gitmodules_shallow(text: &str, path: &str) -> bool {
    let mut in_section = false;
    let mut found_path = false;
    for line in text.lines().map(str::trim) {
        if line.starts_with('[') {
            if in_section && found_path {
                return false;
            }
            in_section = true;
            found_path = false;
            continue;
        }
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        match (k.trim(), v.trim()) {
            ("path", p) if p == path => found_path = true,
            ("shallow", "true") if in_section && found_path => return true,
            _ => {}
        }
    }
    false
}

/// Copy into `dst` (emptied first) every file of `src` that the patches modify, as `diff --git a/<path>`
/// names it; files the patches create are left to them.
fn copy_patched_files(src: &Path, patches: &[PathBuf], dst: &Path) -> std::io::Result<()> {
    if dst.exists() {
        std::fs::remove_dir_all(dst)?;
    }
    std::fs::create_dir_all(dst)?;
    for p in patches {
        for path in patched_paths(&std::fs::read_to_string(p)?) {
            let from = src.join(&path);
            if from.is_file() {
                let to = dst.join(&path);
                if let Some(parent) = to.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::copy(from, to)?;
            }
        }
    }
    Ok(())
}

/// The paths a patch reads (`--- a/<path>` lines; `/dev/null` for created files is skipped).
fn patched_paths(patch: &str) -> BTreeSet<String> {
    patch
        .lines()
        .filter_map(|l| l.strip_prefix("--- a/"))
        .map(|p| p.split('\t').next().unwrap_or(p).trim_end().to_string())
        .collect()
}

/// The patch sets under `third_party/patches/`: one directory per submodule, its `*.patch` files in
/// name order (the order the image build applies them). Sorted by submodule name.
fn patch_sets(root: &Path) -> Vec<(String, Vec<PathBuf>)> {
    let Ok(dirs) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut sets: Vec<(String, Vec<PathBuf>)> = dirs
        .flatten()
        .filter(|e| e.path().is_dir())
        .map(|e| {
            let mut patches: Vec<PathBuf> = std::fs::read_dir(e.path())
                .map(|rd| {
                    rd.flatten()
                        .map(|f| f.path())
                        .filter(|p| p.extension().is_some_and(|x| x == "patch"))
                        .collect()
                })
                .unwrap_or_default();
            patches.sort();
            (e.file_name().to_string_lossy().into_owned(), patches)
        })
        .collect();
    sets.sort();
    sets
}

/// GraMi `Output.txt`: line 1 elapsed seconds, line 2 the number of frequent subgraphs (grami.md §5).
fn grami_pattern_count(path: &Path) -> Option<u64> {
    std::fs::read_to_string(path)
        .ok()?
        .lines()
        .nth(1)?
        .trim()
        .parse()
        .ok()
}

fn git(dir: &Path, args: &[&str]) -> Invocation {
    Invocation::new("git", "git")
        .args(["-C".to_string(), dir.display().to_string()])
        .args(args.iter().copied())
}

fn outcome_text(o: &ToolOutcome) -> String {
    let mut s = std::fs::read_to_string(&o.stdout).unwrap_or_default();
    s.push('\n');
    s.push_str(&std::fs::read_to_string(&o.stderr).unwrap_or_default());
    s
}

fn non_empty(p: &Path) -> bool {
    std::fs::metadata(p).is_ok_and(|m| m.is_file() && m.len() > 0)
}

/// The pins `docker/build.sh` passed, as `cc-probe --pin` recorded them in `/opt/cc/versions.json`.
pub fn expected_pins(c: &Config) -> Vec<(&'static str, String)> {
    let t = &c.tools;
    let b = &t.docker.build;
    vec![
        ("image_tag", t.docker.tag.clone()),
        ("xls_tag", t.xls.tag.clone()),
        ("xls_commit", t.xls.commit.clone()),
        ("allo_commit", t.allo.commit.clone()),
        ("llvm_commit", t.allo.llvm_commit.clone()),
        ("grami_commit", t.grami.commit.clone()),
        ("z3_version", b.z3_version.clone()),
        ("bitwuzla_version", b.bitwuzla_version.clone()),
        ("flopoco_commit", b.flopoco_commit.clone()),
        ("hardfloat_commit", t.hardfloat.commit.clone()),
        ("chisel_version", b.chisel_version.clone()),
        ("firtool_version", b.firtool_version.clone()),
        ("sbt_version", b.sbt_version.clone()),
        ("uv_version", b.uv_version.clone()),
        ("bazelisk_version", b.bazelisk_version.clone()),
        ("apt_snapshot", b.apt_snapshot.clone()),
        ("pypi_exclude_newer", b.pypi_exclude_newer.clone()),
    ]
}

/// Copy-pasteable rendering of an argv.
fn shell_join(argv: &[String]) -> String {
    argv.iter()
        .map(|a| {
            if !a.is_empty()
                && a.chars()
                    .all(|c| c.is_ascii_alphanumeric() || "/._-=:,+@%".contains(c))
            {
                a.clone()
            } else {
                format!("'{}'", a.replace('\'', r"'\''"))
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_lines_seen_on_this_host() {
        // Outputs observed 2026-10-30 (docs/tools/synopsys.md, cadence.md).
        let dc = "Info: [VCS_SAVE_RESTORE_INFO] ASLR ...\n\ndc_shell version    -  W-2024.09\n";
        assert_eq!(version_line(dc).as_deref(), Some("dc_shell version - W-2024.09"));
        assert_eq!(
            version_line("TOOL:\txrun\t24.03-s005\n").as_deref(),
            Some("TOOL: xrun 24.03-s005")
        );
        assert_eq!(version_line("// Warning: This version is 883 days old.\n"), None);
        assert_eq!(version_line("nothing here"), None);
    }

    #[test]
    fn area_and_gitmodules_parsers() {
        assert_eq!(
            total_cell_area("Number of ports: 3\nTotal cell area:          41.230001\n"),
            Some(41.230001)
        );
        assert_eq!(total_cell_area("no area"), None);
        let gm = "[submodule \"third_party/a\"]\n\tpath = third_party/a\n\turl = u\n\tshallow = true\n\
                  [submodule \"third_party/b\"]\n\tpath = third_party/b\n\turl = u\n";
        assert!(gitmodules_shallow(gm, "third_party/a"));
        assert!(!gitmodules_shallow(gm, "third_party/b"));
        assert!(!gitmodules_shallow(gm, "third_party/c"));
    }

    #[test]
    fn patched_paths_are_the_old_sides() {
        let patch = "diff --git a/x/y.cc b/x/y.cc\n--- a/x/y.cc\n+++ b/x/y.cc\n@@ -1 +1 @@\n-a\n+b\n\
                     diff --git a/new.py b/new.py\nnew file mode 100644\n--- /dev/null\n+++ b/new.py\n\
                     --- a/z.h\t2026-01-01\n";
        let got: Vec<String> = patched_paths(patch).into_iter().collect();
        assert_eq!(got, vec!["x/y.cc".to_string(), "z.h".to_string()]);
    }

    #[test]
    fn patch_sets_are_per_submodule_in_name_order() {
        let root = tempfile::tempdir().unwrap();
        for f in [
            "xls/0002-b.patch",
            "xls/0001-a.patch",
            "xls/README",
            "allo/0001-c.patch",
        ] {
            let p = root.path().join(f);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, "").unwrap();
        }
        std::fs::create_dir_all(root.path().join("grami")).unwrap();
        let sets: Vec<(String, Vec<String>)> = patch_sets(root.path())
            .into_iter()
            .map(|(n, ps)| {
                let names = ps
                    .iter()
                    .map(|p| p.file_name().unwrap().to_string_lossy().into_owned());
                (n, names.collect())
            })
            .collect();
        assert_eq!(
            sets,
            vec![
                ("allo".to_string(), vec!["0001-c.patch".to_string()]),
                ("grami".to_string(), vec![]),
                (
                    "xls".to_string(),
                    vec!["0001-a.patch".to_string(), "0002-b.patch".to_string()]
                ),
            ]
        );
        assert!(patch_sets(&root.path().join("missing")).is_empty());
    }

    #[test]
    fn dc_script_caps_and_libraries() {
        let s = dc_script(
            &["set_host_options -max_cores 2".into()],
            Path::new("/b/.env/out/pdk_db"),
            &["X.db".into()],
            "2.0",
        );
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines[1], "set_host_options -max_cores 2");
        assert!(lines.contains(&"set target_library [list X.db]"));
        assert!(lines.contains(&"set search_path [list . /b/.env/out/pdk_db]"));
        assert!(lines.contains(&"create_clock -name vclk -period 2.0"));
        assert_eq!(lines.last(), Some(&"quit"));
    }
}
