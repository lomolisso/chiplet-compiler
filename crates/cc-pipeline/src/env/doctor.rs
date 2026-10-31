//! `chipc doctor` (spec §3.6, §8): prove the environment works by making every tool do one
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
const DOCTOR_VERSION: u32 = 1;
/// The steps `cc-selftest` reports (docker/selftest.sh).
pub const SELFTEST_STEPS: [&str; 6] = ["xls", "xls_equiv", "rtl_tools", "smt", "grami", "allo"];
/// Version query per licensed tool (flags observed on this host, docs/context/tooling/{synopsys,cadence}.md).
const EDA_VERSION_PROBES: [(&str, &[&str]); 8] = [
    ("dc_shell", &["-version"]),
    ("lc_shell", &["-version"]),
    ("fm_shell", &["-version"]),
    ("pt_shell", &["-version"]),
    ("vcs", &["-full64", "-ID"]),
    ("genus", &["-version"]),
    ("lec", &["-version"]),
    ("xrun", &["-version"]),
];
const GIB: f64 = (1u64 << 30) as f64;

// Doctor's own caps: tiny jobs only (spec §8).
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
    let libs = d.pdk();
    if opts.skip_licenses {
        d.push(Check::skip("dc.nangate45", "--skip-licenses"));
        d.push(Check::skip("dc.asap7", "--skip-licenses"));
        d.push(Check::skip("license.genus", "--skip-licenses"));
        d.push(Check::skip("license.lec", "--skip-licenses"));
    } else {
        d.dc_tiny(&libs);
        d.cadence_licenses();
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
            let probe = scratch.join("chipc").join(".doctor-probe");
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

        let patch_dir = tp.join("patches").join("grami");
        let mut patches: Vec<PathBuf> = std::fs::read_dir(&patch_dir)
            .map(|rd| {
                rd.flatten()
                    .map(|e| e.path())
                    .filter(|p| p.extension().is_some_and(|x| x == "patch"))
                    .collect()
            })
            .unwrap_or_default();
        patches.sort();
        if patches.is_empty() {
            self.push(
                Check::new("submodule.grami.patches", false)
                    .note(format!("no patches in {}", patch_dir.display())),
            );
        }
        for p in patches {
            let inv = git(&tp.join("grami"), &["apply", "--check", &p.display().to_string()]);
            if let Some(o) = self.exec("submodule.grami.patches", Via::Host, &inv, HOST_CAPS) {
                let name = p
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                self.push(Check::new("submodule.grami.patches", o.success()).note(format!("{name} applies")));
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

    /// PDK files exist; the Liberty files compile to `.db` (cached by `.lib` hash + LC version).
    fn pdk(&mut self) -> Vec<LibertyLib> {
        let p = self.config().paths.clone();
        let mut libs = Vec::new();

        let nangate = p.nangate45_lib.as_ref().map(|x| self.cfg.resolve(x));
        let lef = p.nangate45_lef.as_ref().map(|x| self.cfg.resolve(x));
        match (&nangate, &lef) {
            (Some(lib), Some(lef)) if lib.is_file() && lef.is_file() => match pdk_db::read_library(lib) {
                Ok(l) => {
                    self.push(
                        Check::new("pdk.nangate45", true)
                            .path(lib.display().to_string())
                            .path(lef.display().to_string())
                            .note(format!("library {}", l.name)),
                    );
                    libs.push(l);
                }
                Err(e) => self.push(Check::new("pdk.nangate45", false).note(e.to_string())),
            },
            _ => self.push(
                Check::new("pdk.nangate45", false)
                    .note("paths.nangate45_lib / nangate45_lef missing or unset (config.local.toml)"),
            ),
        }

        let pdk = self.config().tools.pdk.clone();
        let label = format!("{} {}", pdk.asap7_vt, pdk.asap7_corner);
        match p.asap7_nldm_dir.as_ref().map(|x| self.cfg.resolve(x)) {
            None => self.push(
                Check::new("pdk.asap7", false).note("paths.asap7_nldm_dir is unset (config.local.toml)"),
            ),
            Some(dir) => match pdk_db::asap7_libs(&dir, &pdk).and_then(|ps| {
                ps.iter()
                    .map(|p| pdk_db::read_library(p))
                    .collect::<Result<Vec<_>, _>>()
            }) {
                Ok(found) => {
                    let mut c = Check::new("pdk.asap7", true)
                        .count("libraries", found.len() as u64)
                        .note(format!("{label}: {} libraries", found.len()));
                    for l in &found {
                        c = c.path(l.path.display().to_string());
                    }
                    self.push(c);
                    libs.extend(found);
                }
                Err(e) => self.push(Check::new("pdk.asap7", false).note(e.to_string())),
            },
        }
        if libs.is_empty() {
            self.push(Check::new("pdk.db", false).note("no Liberty files"));
            return libs;
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
            return libs;
        }

        let job = self.dir.path().join("eda").join("lc");
        let script = job.join("make_db.tcl");
        if !self.write(&script, &pdk_db::make_db_tcl(&libs)) {
            self.push(Check::new("pdk.db", false).note(format!("cannot write {}", script.display())));
            return libs;
        }
        let inv = Invocation::new("lc_shell", EdaTool::Lc.program())
            .args(EdaTool::Lc.batch_args("make_db.tcl"))
            .cwd(&job);
        let Some(o) = self.exec("pdk.db", Via::Eda, &inv, LC_CAPS) else {
            return libs;
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
        libs
    }

    // -- licensed synthesis -----------------------------------------------------------------------

    /// An 8-bit adder through `compile_ultra` on each kit: proves the Synopsys license and that the
    /// `.db` files link.
    fn dc_tiny(&mut self, libs: &[LibertyLib]) {
        let db_dir = self.dir.kind(PDK_DB).dir().to_path_buf();
        let nangate: Vec<String> = libs
            .iter()
            .filter(|l| !l.name.starts_with("asap7"))
            .map(LibertyLib::db_file)
            .collect();
        let asap7: Vec<String> = libs
            .iter()
            .filter(|l| l.name.starts_with("asap7"))
            .map(LibertyLib::db_file)
            .collect();
        // Clock period in each library's time unit: ns for Nangate45, ps for ASAP7 (pdks.md §5).
        for (label, dbs, period) in [("nangate45", nangate, "2.0"), ("asap7", asap7, "1000")] {
            let name = format!("dc.{label}");
            if dbs.is_empty() || (!self.opts.dry && !dbs.iter().all(|f| non_empty(&db_dir.join(f)))) {
                self.push(Check::new(name, false).note("no .db files (see pdk.db)"));
                continue;
            }
            let job = self.dir.path().join("eda").join(format!("dc_{label}"));
            let script = dc_script(&EdaTool::Dc.cap_preamble(&DC_CAPS), &db_dir, &dbs, period);
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

    /// Cadence licenses can fail intermittently (cadence.md §5): check both tools we use.
    fn cadence_licenses(&mut self) {
        let job = self.dir.path().join("eda").join("genus_license");
        let inv = Invocation::new("genus", EdaTool::Genus.program())
            .args([
                "-batch",
                "-no_gui",
                "-disable_user_startup",
                "-execute",
                "set_db / .max_cpus_per_server 1; exit",
            ])
            .cwd(&job);
        if let Some(o) = self.exec("license.genus", Via::Eda, &inv, EDA_PROBE_CAPS) {
            let text = outcome_text(&o);
            let checked_out = text
                .lines()
                .find(|l| l.contains("Checking out license"))
                .map(str::trim);
            let mut c = Check::new("license.genus", o.success() && checked_out.is_some()).note(
                checked_out
                    .unwrap_or("no `Checking out license` line")
                    .to_string(),
            );
            // Genus nests its temp dir under TMPDIR, which EdaRunner points at the scratch disk.
            if let Some(t) = text
                .lines()
                .find_map(|l| l.strip_prefix("TMPDIR is being set to "))
            {
                let on_scratch = self
                    .config()
                    .paths
                    .scratch_dir
                    .as_ref()
                    .is_some_and(|s| Path::new(t.trim()).starts_with(self.cfg.resolve(s)));
                c = c.note(format!(
                    "Genus temp dir {} ({})",
                    t.trim(),
                    if on_scratch {
                        "scratch disk"
                    } else {
                        "NOT on scratch"
                    }
                ));
                if !on_scratch {
                    c = c.with_status(CheckStatus::Fail);
                }
            }
            self.push(c);
        }

        let job = self.dir.path().join("eda").join("lec_license");
        if !self.write(&job.join("exit.do"), "exit -force\n") {
            self.push(Check::new("license.lec", false).note("cannot write exit.do"));
            return;
        }
        let inv = Invocation::new("lec", EdaTool::Lec.program())
            .args(EdaTool::Lec.batch_args("exit.do"))
            .cwd(&job);
        // Exit status is nonzero when no comparison ran (conformal.hlp: EXIT), so judge by the log.
        if let Some(o) = self.exec("license.lec", Via::Eda, &inv, EDA_PROBE_CAPS) {
            let text = outcome_text(&o);
            let lines: Vec<&str> = text
                .lines()
                .filter(|l| l.contains("Check out Conformal_"))
                .map(str::trim)
                .collect();
            self.push(
                Check::new("license.lec", !lines.is_empty() && o.killed.is_none()).note(
                    if lines.is_empty() {
                        "no license checkout line".into()
                    } else {
                        lines.join("; ")
                    },
                ),
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
        for k in [PDK_DB, SELFTEST, DC_TINY] {
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

/// The first line that names a version: `… version …` (Synopsys, Genus, VCS) or `Tool: <name> <ver>`
/// (LEC, xrun). Conformal `//` comment lines are skipped.
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
        // Outputs observed 2026-10-30 (docs/context/tooling/synopsys.md, cadence.md).
        let dc = "Info: [VCS_SAVE_RESTORE_INFO] ASLR ...\n\ndc_shell version    -  W-2024.09\n";
        assert_eq!(version_line(dc).as_deref(), Some("dc_shell version - W-2024.09"));
        assert_eq!(
            version_line("Tool:\tlec\t24.10-p100\n").as_deref(),
            Some("Tool: lec 24.10-p100")
        );
        let genus = "No DISPLAY environment variable set.\nProgram Name: Genus(TM) Synthesis Solution, Version: 21.19-s055_1\n";
        assert!(version_line(genus).unwrap().ends_with("21.19-s055_1"));
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
