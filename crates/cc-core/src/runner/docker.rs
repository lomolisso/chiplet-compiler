//! `DockerRunner`: one capped container per call (docs/context/tooling/docker_resources.md §2, §3, §6).
//!
//! `--memory-swap` equals `--memory` (no swap: host swap is full); `--init` reaps and forwards signals;
//! the job runs as the invoking user because the daemon has no userns remapping. The container is named
//! and not `--rm`, so `OOMKilled` can be read after exit; it is removed afterwards.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use crate::cancel::CancelToken;
use crate::config::DockerConfig;
use crate::error::{CoreError, IoContext, Result};
use crate::runner::process::{self, Supervisor};
use crate::runner::{CallLog, Caps, Invocation, KillReason, ToolLog, ToolOutcome, ToolRunner, sanitize};

#[derive(Debug, Clone)]
pub struct DockerRunner {
    pub cfg: DockerConfig,
    pub docker_bin: String,
    pub uid: u32,
    pub gid: u32,
    pub sample_interval: Duration,
}

impl DockerRunner {
    pub fn from_config(cfg: DockerConfig) -> Self {
        let (uid, gid) = process::current_uid_gid();
        Self {
            cfg,
            docker_bin: "docker".into(),
            uid,
            gid,
            sample_interval: Duration::from_millis(500),
        }
    }

    pub fn default_image(&self) -> String {
        format!("{}:{}", self.cfg.image, self.cfg.tag)
    }

    /// The full `docker run` argument vector (without the `docker` binary). Pure: unit-tested.
    pub fn docker_args(
        &self,
        name: &str,
        cidfile: Option<&Path>,
        inv: &Invocation,
        caps: &Caps,
    ) -> Vec<String> {
        let mem = format!("{}m", (caps.memory_gb * 1024.0).round() as u64);
        let mut a: Vec<String> = vec!["run".into(), "--init".into(), "--name".into(), name.into()];
        if let Some(c) = cidfile {
            a.extend(["--cidfile".into(), c.display().to_string()]);
        }
        a.extend([
            "--user".into(),
            format!("{}:{}", self.uid, self.gid),
            "-e".into(),
            "HOME=/tmp/home".into(),
        ]);
        for (k, v) in &inv.env {
            a.extend(["-e".into(), format!("{k}={v}")]);
        }
        a.extend([
            "--memory".into(),
            mem.clone(),
            "--memory-swap".into(),
            mem,
            "--memory-swappiness".into(),
            "0".into(),
            "--cpus".into(),
            format!("{}", caps.cpus),
            "--pids-limit".into(),
            format!("{}", caps.pids_limit),
            "--ulimit".into(),
            "core=0".into(),
            "--network".into(),
            "none".into(),
        ]);
        for m in &inv.mounts {
            let ro = if m.readonly { ",readonly" } else { "" };
            a.extend([
                "--mount".into(),
                format!("type=bind,src={},dst={}{ro}", m.src.display(), m.dst),
            ]);
        }
        if let Some(cwd) = &inv.cwd {
            a.extend(["--workdir".into(), cwd.display().to_string()]);
        }
        a.push(inv.image.clone().unwrap_or_else(|| self.default_image()));
        a.push(inv.program.clone());
        a.extend(inv.args.iter().cloned());
        a
    }

    fn docker(&self, args: &[&str]) -> Option<String> {
        let out = Command::new(&self.docker_bin)
            .args(args)
            .stdin(Stdio::null())
            .output()
            .ok()?;
        out.status
            .success()
            .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
    }
}

/// Peak memory of the container's cgroup: v1 `memory.max_usage_in_bytes` (this host) or v2 `memory.peak`.
fn cgroup_peak(cid: &str) -> Option<u64> {
    // Docker creates the cidfile before writing the id: an empty id would name the parent `docker`
    // cgroup (every container's peak, verified 2026-10-30).
    if cid.is_empty() || !cid.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let candidates = [
        format!("/sys/fs/cgroup/memory/docker/{cid}/memory.max_usage_in_bytes"),
        format!("/sys/fs/cgroup/memory/system.slice/docker-{cid}.scope/memory.max_usage_in_bytes"),
        format!("/sys/fs/cgroup/system.slice/docker-{cid}.scope/memory.peak"),
    ];
    candidates
        .iter()
        .find_map(|p| std::fs::read_to_string(p).ok()?.trim().parse().ok())
}

impl ToolRunner for DockerRunner {
    fn name(&self) -> &'static str {
        "docker"
    }

    fn run(&self, inv: &Invocation, caps: &Caps, log: &ToolLog, cancel: &CancelToken) -> Result<ToolOutcome> {
        cancel.check()?;
        // `--mount` errors on a missing source (unlike `-v`, which creates it as root): create writable ones.
        for m in &inv.mounts {
            if !m.readonly {
                std::fs::create_dir_all(&m.src).at(&m.src)?;
            }
        }
        let call = CallLog::start(log, &inv.tool)?;
        let name = format!(
            "chipc-{}-{}-{}-{}",
            sanitize(log.pass()),
            sanitize(&inv.tool),
            call.files.n,
            std::process::id()
        );
        let cidfile: PathBuf = call.files.stdout.with_extension("cid");
        if cidfile.exists() {
            std::fs::remove_file(&cidfile).at(&cidfile)?;
        }
        let args = self.docker_args(&name, Some(&cidfile), inv, caps);
        let mut cmd = Command::new(&self.docker_bin);
        cmd.args(&args).stdin(Stdio::null());
        cmd.stdout(File::create(&call.files.stdout).at(&call.files.stdout)?);
        cmd.stderr(File::create(&call.files.stderr).at(&call.files.stderr)?);
        let mut child = cmd.spawn().map_err(|e| CoreError::Tool {
            tool: inv.tool.clone(),
            message: format!("spawn `{}`: {e}", self.docker_bin),
        })?;
        let sample = || {
            std::fs::read_to_string(&cidfile)
                .ok()
                .and_then(|c| cgroup_peak(c.trim()))
        };
        let kill = || {
            let _ = self.docker(&["kill", &name]);
        };
        let sup = Supervisor {
            timeout: caps.timeout(),
            cancel,
            sample_interval: self.sample_interval,
            cap_bytes: None,
            sample: &sample,
            kill: &kill,
        };
        let res = process::supervise(&mut child, &sup).at(&call.files.stderr)?;
        let mut killed = res.killed;
        if killed.is_none()
            && self
                .docker(&["inspect", "-f", "{{.State.OOMKilled}}", &name])
                .as_deref()
                == Some("true")
        {
            killed = Some(KillReason::OomKilled);
        }
        let _ = self.docker(&["rm", "-f", &name]);
        let (exit_code, signal) = process::exit_parts(&res.status);
        let mut argv = vec![self.docker_bin.clone()];
        argv.extend(args);
        let outcome = ToolOutcome {
            runner: self.name().into(),
            tool: inv.tool.clone(),
            argv,
            exit_code,
            signal,
            killed,
            duration_ms: call.t0.elapsed().as_millis() as u64,
            peak_rss_bytes: res.peak_bytes,
            stdout: call.files.stdout.clone(),
            stderr: call.files.stderr.clone(),
        };
        call.finish(caps, &outcome)?;
        Ok(outcome)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::Mount;

    #[test]
    fn peak_ignores_an_unwritten_cidfile() {
        assert_eq!(cgroup_peak(""), None);
        assert_eq!(cgroup_peak("../x"), None);
    }

    #[test]
    fn args_carry_caps_mounts_and_user() {
        let r = DockerRunner {
            uid: 1000,
            gid: 100,
            ..DockerRunner::from_config(DockerConfig::default())
        };
        let inv = Invocation::new("t", "codegen_main")
            .arg("--x")
            .env("A", "1")
            .cwd("/work/out")
            .mount(Mount {
                src: "/repo".into(),
                dst: "/work/repo".into(),
                readonly: true,
            });
        let caps = Caps {
            memory_gb: 16.0,
            cpus: 8.0,
            timeout_s: 60.0,
            pids_limit: 4096,
        };
        let a = r.docker_args("n", None, &inv, &caps).join(" ");
        assert_eq!(
            a,
            "run --init --name n --user 1000:100 -e HOME=/tmp/home -e A=1 --memory 16384m --memory-swap 16384m \
             --memory-swappiness 0 --cpus 8 --pids-limit 4096 --ulimit core=0 --network none \
             --mount type=bind,src=/repo,dst=/work/repo,readonly --workdir /work/out \
             chiplet-compiler/toolchain:dev codegen_main --x"
        );
    }
}
