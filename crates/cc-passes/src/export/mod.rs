//! `export` (design §4.1): every phase of every port → StableHLO with component composites and
//! symbolic input sizes, plus the JSON the Rust passes read. The work is done by the Python exporter
//! `python/cc_export` (the only Python pass), one capped process per model; this module drives it, and
//! `verify` checks its outputs independently (see docs/passes/export.md).

mod analysis;
mod audit;
mod config;
pub mod json;
mod verify;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use cc_core::runner::{BoundRunner, Caps, Invocation, Mount};
use cc_core::{ArtifactKind, CoreError, LoadedConfig, PassSection, VerifyReport, canon};
use cc_pipeline::{Pass, PassContext, PassError};
use rayon::prelude::*;

pub use config::{ExportConfig, ExportRunner};

/// `out/export/<model>/{export.json, <phase>/{module.mlir, exported.bin, graph.json, symbols.json,
/// census.json, census_concrete.json}}`.
pub const EXPORT: ArtifactKind = ArtifactKind("export");
pub const MODEL_FILE: &str = "export.json";
pub const PHASE_FILES: [&str; 6] = [
    "module.mlir",
    "exported.bin",
    "graph.json",
    "symbols.json",
    "census.json",
    "census_concrete.json",
];

/// Exit status of `python -m cc_export` when a phase failed to export (recorded in `export.json`).
const EXIT_PHASE_FAILED: i32 = 3;

const REPO_IN_CONTAINER: &str = "/work/repo";
const OUT_IN_CONTAINER: &str = "/work/out";

#[derive(Debug)]
pub struct Export;

impl Pass for Export {
    fn name(&self) -> &'static str {
        "export"
    }

    fn version(&self) -> u32 {
        1
    }

    fn requires(&self) -> &[ArtifactKind] {
        &[]
    }

    fn provides(&self) -> &[ArtifactKind] {
        &[EXPORT]
    }

    /// The ports, the exporter and the Python environment are inputs too (they are not artifacts).
    fn fingerprints(
        &self,
        config: &LoadedConfig,
        section: &PassSection,
    ) -> Result<BTreeMap<String, String>, CoreError> {
        let cfg: ExportConfig = section.parse()?;
        let root = &config.root;
        let mut m = BTreeMap::new();
        m.insert("models".into(), canon::hash_tree(&config.models_dir(), &[])?);
        m.insert(
            "cc_export".into(),
            canon::hash_tree(&root.join("python/cc_export"), &[".py"])?,
        );
        m.insert("verify".into(), canon::hash_tree(&root.join("verify"), &[".py"])?);
        m.insert("uv_lock".into(), canon::hash_tree(&root.join("uv.lock"), &[])?);
        if cfg.runner == ExportRunner::Docker {
            let d = &config.config.tools.docker;
            m.insert("image".into(), format!("{}:{}", d.image, d.tag));
        }
        Ok(m)
    }

    fn check_config(&self, section: &PassSection) -> Result<(), CoreError> {
        section.parse::<ExportConfig>().map(drop)
    }

    fn run(&self, ctx: &mut PassContext) -> Result<(), PassError> {
        let cfg: ExportConfig = ctx.options()?;
        let ctx: &PassContext = ctx;
        let out = ctx.output(EXPORT)?.dir().to_path_buf();
        let runner = model_runner(ctx, &cfg);
        let failures: Vec<String> = ctx
            .models()
            .par_iter()
            .with_max_len(1)
            .map(|m| -> Result<Option<String>, PassError> {
                ctx.check_cancelled()?;
                let inv = invocation(ctx, &cfg, &["-m", "cc_export"], &model_args(ctx, &cfg, m, &out));
                let o = runner.run(&inv)?;
                if o.exit_code == Some(EXIT_PHASE_FAILED) && o.killed.is_none() {
                    return Ok(Some(m.clone()));
                }
                o.check()?;
                Ok(None)
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect();
        if !failures.is_empty() {
            tracing::warn!(models = ?failures, "export: some phases failed; see export.json and verify.json");
        }
        verify::write_coverage_view(ctx, &out)?;
        Ok(())
    }

    fn verify(&self, ctx: &PassContext) -> Result<VerifyReport, PassError> {
        verify::verify(ctx)
    }
}

/// The per-model runner: each model's process gets `memory_gb / threads` and one CPU (tracing is
/// single-threaded), so `threads` models run side by side within the pass's declaration.
fn model_runner(ctx: &PassContext, cfg: &ExportConfig) -> BoundRunner {
    let r = match cfg.runner {
        ExportRunner::Docker => ctx.docker(),
        ExportRunner::Host => ctx.host(),
    };
    let res = ctx.resources();
    let per = Caps {
        memory_gb: res.memory_gb / res.threads.max(1) as f64,
        cpus: 1.0,
        ..*r.caps()
    };
    let caps = r.caps().min(&per);
    r.with_caps(caps)
}

fn model_args(ctx: &PassContext, cfg: &ExportConfig, model: &str, out: &Path) -> Vec<String> {
    let out_dir = match cfg.runner {
        ExportRunner::Docker => format!("{OUT_IN_CONTAINER}/{model}"),
        ExportRunner::Host => out.join(model).display().to_string(),
    };
    let mut a = vec![
        "--model".to_string(),
        model.to_string(),
        "--config".to_string(),
        ctx.model_config().as_str().to_string(),
        "--out".to_string(),
        out_dir,
    ];
    for c in &cfg.exclude_classes {
        a.extend(["--exclude".to_string(), c.clone()]);
    }
    a
}

/// `python <module args> <args>` in the configured environment, with the repository on `PYTHONPATH`.
fn invocation(ctx: &PassContext, cfg: &ExportConfig, module: &[&str], args: &[String]) -> Invocation {
    let repo = ctx.repo_root();
    let out = ctx
        .output(EXPORT)
        .map(|k| k.dir().to_path_buf())
        .unwrap_or_default();
    let common = |inv: Invocation, root: &str| {
        inv.env("PYTHONPATH", format!("{root}/python:{root}"))
            .env("JAX_PLATFORMS", "cpu")
            .env("KERAS_BACKEND", "jax")
            .args(module.iter().copied())
            .args(args.iter().cloned())
    };
    match cfg.runner {
        ExportRunner::Docker => common(
            Invocation::new("cc_export", cfg.docker_python.clone()),
            REPO_IN_CONTAINER,
        )
        .cwd(REPO_IN_CONTAINER)
        .mount(Mount {
            src: repo.to_path_buf(),
            dst: REPO_IN_CONTAINER.into(),
            readonly: true,
        })
        .mount(Mount {
            src: out,
            dst: OUT_IN_CONTAINER.into(),
            readonly: false,
        }),
        ExportRunner::Host => {
            let (prog, pre) = cfg
                .host_python
                .split_first()
                .map_or(("python", &[][..]), |(p, r)| (p.as_str(), r));
            let root = repo.display().to_string();
            common(
                Invocation::new("cc_export", prog).args(pre.iter().cloned()),
                &root,
            )
            .cwd(PathBuf::from(repo))
        }
    }
}
