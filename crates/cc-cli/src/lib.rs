//! `chipc`: the command line. The logic lives here (writing to any `Write`) so tests can drive it
//! in-process; `main.rs` only parses arguments and sets up logging. `anyhow` is used only in this crate.

use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use cc_core::LoadedConfig;
use cc_pipeline::{Pass, Pipeline, RunOptions, RunStatus};
use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "chipc", version, about = "chiplet-compiler driver")]
pub struct Cli {
    /// Configuration file; `<stem>.local.toml` beside it is merged on top.
    #[arg(long, global = true, default_value = "config.toml")]
    pub config: PathBuf,

    /// Override a config key, e.g. `--set resources.memory_budget_gb=16` (repeatable; highest priority).
    #[arg(long = "set", global = true, value_name = "KEY=VALUE")]
    pub set: Vec<String>,

    /// More log output on stderr (-v debug, -vv trace).
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    pub verbose: u8,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Run the pipeline up to the targets, reusing cached passes.
    Run {
        /// Force this pass and everything depending on it to rerun.
        #[arg(long)]
        from: Option<String>,
        /// Run only this pass; its inputs must already exist.
        #[arg(long, conflicts_with = "from")]
        only: Option<String>,
        /// Restrict the model set (comma-separated catalog keys).
        #[arg(long, value_delimiter = ',')]
        models: Option<Vec<String>>,
        /// Pass to reach (repeatable); default `pipeline.targets`.
        #[arg(long = "target")]
        targets: Vec<String>,
    },
    /// Re-run a pass's checks on its existing artifacts.
    Verify { pass: String },
    /// Pass states from the manifests.
    Status,
    /// Remove one pass's build directory, or (with --yes) every pass's and the run summaries.
    Clean {
        pass: Option<String>,
        #[arg(long)]
        yes: bool,
    },
    /// Print the pass DAG as Graphviz DOT.
    Graph,
    /// Run the test suites through cargo-nextest (smoke by default).
    Test {
        #[arg(long)]
        deep: bool,
        #[arg(long, default_value_t = 2)]
        test_threads: usize,
        /// Extra arguments for `cargo nextest run`.
        #[arg(last = true)]
        extra: Vec<String>,
    },
    /// Check the environment (tools, images, licenses, PDKs). Implemented in another change.
    Doctor,
}

fn load(cli: &Cli) -> Result<LoadedConfig> {
    LoadedConfig::load(&cli.config, &cli.set).with_context(|| format!("loading {}", cli.config.display()))
}

fn pipeline(cli: &Cli, passes: Vec<Box<dyn Pass>>) -> Result<Pipeline> {
    Ok(Pipeline::new(passes, load(cli)?)?)
}

/// Where `main` should write the JSON log of a `run` (`build/.pipeline/logs/`), if the config loads.
pub fn run_log_path(cli: &Cli) -> Option<PathBuf> {
    if !matches!(cli.command, Command::Run { .. }) {
        return None;
    }
    let cfg = load(cli).ok()?;
    let stamp = chrono::Utc::now().format("%Y%m%dT%H%M%S%.3fZ");
    Some(
        cfg.build_dir()
            .join(".pipeline")
            .join("logs")
            .join(format!("chipc-{stamp}.jsonl")),
    )
}

fn status_word(s: RunStatus) -> &'static str {
    match s {
        RunStatus::Ok => "ok",
        RunStatus::CacheHit => "cache hit",
        RunStatus::Failed => "FAILED",
        RunStatus::Cancelled => "cancelled",
        RunStatus::Skipped => "skipped",
    }
}

/// Execute a parsed command with the given registry. Returns the process exit code.
pub fn execute(cli: &Cli, passes: Vec<Box<dyn Pass>>, out: &mut dyn Write) -> Result<i32> {
    match &cli.command {
        Command::Run {
            from,
            only,
            models,
            targets,
        } => {
            let p = pipeline(cli, passes)?;
            let opts = RunOptions {
                targets: targets.clone(),
                from: from.clone(),
                only: only.clone(),
                models: models.clone(),
            };
            let s = p.run(&opts)?;
            for r in &s.passes {
                let ms = r.duration_ms.map(|d| format!("{d} ms")).unwrap_or_default();
                let msg = r
                    .message
                    .as_deref()
                    .map(|m| format!("  ({m})"))
                    .unwrap_or_default();
                writeln!(
                    out,
                    "{:<24} {:<10} {:>10}{msg}",
                    r.pass,
                    status_word(r.status),
                    ms
                )?;
            }
            let t = &s.totals;
            writeln!(
                out,
                "ran {}, cache hits {}, failed {}, cancelled {}, skipped {} in {} ms",
                t.ran, t.cache_hits, t.failed, t.cancelled, t.skipped, s.duration_ms
            )?;
            if let Some(path) = &s.path {
                writeln!(out, "summary: {}", path.display())?;
            }
            Ok(if s.ok { 0 } else { 1 })
        }
        Command::Verify { pass } => {
            let p = pipeline(cli, passes)?;
            let r = p.verify_pass(pass)?;
            for c in &r.checks {
                let st = serde_json_word(&c.status);
                writeln!(out, "{:<40} {st}", c.name)?;
            }
            writeln!(
                out,
                "{}: {}",
                pass,
                if r.passed() { "verified" } else { "FAILED" }
            )?;
            Ok(if r.passed() { 0 } else { 1 })
        }
        Command::Status => {
            let p = pipeline(cli, passes)?;
            writeln!(
                out,
                "{:<24} {:<10} {:<6} {:<26} {:>10}  key",
                "pass", "status", "fresh", "finished", "duration"
            )?;
            for r in p.status() {
                let fresh = match r.fresh {
                    Some(true) => "yes",
                    Some(false) => "no",
                    None => "-",
                };
                let key = r
                    .cache_key
                    .as_deref()
                    .map(|k| &k[..k.len().min(12)])
                    .unwrap_or("-");
                writeln!(
                    out,
                    "{:<24} {:<10} {:<6} {:<26} {:>10}  {key}",
                    r.pass,
                    r.status,
                    fresh,
                    r.finished_at.as_deref().unwrap_or("-"),
                    r.duration_ms
                        .map(|d| format!("{d} ms"))
                        .unwrap_or_else(|| "-".into()),
                )?;
            }
            Ok(0)
        }
        Command::Clean { pass, yes } => {
            let p = pipeline(cli, passes)?;
            if pass.is_none() && !yes {
                bail!("`chipc clean` without a pass removes every pass directory; add --yes to confirm");
            }
            for d in p.clean(pass.as_deref())? {
                writeln!(out, "removed {}", d.display())?;
            }
            Ok(0)
        }
        Command::Graph => {
            let p = pipeline(cli, passes)?;
            write!(out, "{}", p.graph().to_dot())?;
            Ok(0)
        }
        Command::Test {
            deep,
            test_threads,
            extra,
        } => {
            let cfg = load(cli)?;
            let profile = if *deep { "deep" } else { "smoke" };
            let status = std::process::Command::new("cargo")
                .args([
                    "nextest",
                    "run",
                    "--workspace",
                    "--profile",
                    profile,
                    "--test-threads",
                ])
                .arg(test_threads.to_string())
                .args(extra)
                .current_dir(&cfg.root)
                .status()
                .context("running cargo nextest (install: https://get.nexte.st)")?;
            Ok(status.code().unwrap_or(1))
        }
        Command::Doctor => {
            writeln!(
                out,
                "chipc doctor: environment checks (tools, image, licenses, PDKs) are implemented in another change"
            )?;
            Ok(0)
        }
    }
}

fn serde_json_word(s: &cc_core::CheckStatus) -> &'static str {
    match s {
        cc_core::CheckStatus::Pass => "pass",
        cc_core::CheckStatus::Fail => "FAIL",
        cc_core::CheckStatus::Skip => "skip",
    }
}
