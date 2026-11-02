//! `characterize` (design §4.9): the area, delay and leakage of every hardware primitive the merged units
//! are built from, per kit and timing point, each netlist proven equal to its RTL
//! (docs/passes/characterize.md).

pub mod catalog;
pub mod config;
pub mod flow;
pub mod json;
pub mod sanity;
mod verify;
mod view;

use std::collections::{BTreeMap, BTreeSet};

use cc_core::{ArtifactKind, CoreError, LoadedConfig, PassSection, VerifyReport, canon};
use cc_pipeline::{Pass, PassContext, PassError, RunnerKind, ToolSpec};

pub use config::{CharacterizeConfig, Tool};
use json::{LIBRARY_SCHEMA, Library, Row, SUMMARY_SCHEMA, Summary, TargetInfo};

/// `out/library/{library,catalog,coverage,summary}.json`, `rtl/`, `netlist/`, `reports/`.
pub const LIBRARY: ArtifactKind = ArtifactKind("library");
pub const VIEW: &str = "library.md";

const TOOLS: &[ToolSpec] = &[
    ToolSpec {
        name: "ir_converter_main",
        runner: RunnerKind::Docker,
        program: "ir_converter_main",
        version_args: &["--version"],
    },
    ToolSpec {
        name: "yosys",
        runner: RunnerKind::Docker,
        program: "yosys",
        version_args: &["-V"],
    },
];

#[derive(Debug)]
pub struct Characterize;

impl Pass for Characterize {
    fn name(&self) -> &'static str {
        "characterize"
    }

    fn version(&self) -> u32 {
        1
    }

    fn requires(&self) -> &[ArtifactKind] {
        &[]
    }

    fn provides(&self) -> &[ArtifactKind] {
        &[LIBRARY]
    }

    fn tools(&self) -> &[ToolSpec] {
        TOOLS
    }

    fn model_dependent(&self) -> bool {
        false
    }

    /// The licensed tools' versions and the libraries come from `chipc doctor`'s manifest
    /// (`build/.env/`): a new `.db` or tool version reruns the pass.
    fn fingerprints(
        &self,
        config: &LoadedConfig,
        _section: &PassSection,
    ) -> Result<BTreeMap<String, String>, CoreError> {
        let env = cc_core::ArtifactStore::new(config.build_dir()).pass_dir(cc_pipeline::env::doctor::ENV_DIR);
        let mut m = BTreeMap::new();
        m.insert("catalog".into(), catalog::CATALOG_VERSION.to_string());
        let d = &config.config.tools.docker;
        m.insert("image".into(), format!("{}:{}", d.image, d.tag));
        if let Some(man) = env.read_manifest()? {
            for (k, v) in &man.tool_versions {
                if matches!(k.as_str(), "dc_shell" | "fm_shell" | "genus" | "lec" | "lc_shell") {
                    m.insert(format!("tool:{k}"), v.clone());
                }
            }
            for i in &man.inputs {
                m.insert(i.kind.clone(), i.hash.clone());
            }
            for o in &man.outputs {
                for f in &o.files {
                    if o.kind == "pdk_db" || o.kind == "pdk_lib" {
                        m.insert(format!("{}:{}", o.kind, f.path), f.hash.clone());
                    }
                }
            }
        }
        Ok(m)
    }

    fn check_config(&self, section: &PassSection) -> Result<(), CoreError> {
        let c: CharacterizeConfig = section.parse()?;
        c.check().map_err(|message| CoreError::PassConfig {
            pass: "characterize".into(),
            message,
        })
    }

    fn run(&self, ctx: &mut PassContext) -> Result<(), PassError> {
        let cfg: CharacterizeConfig = ctx.options()?;
        let ctx: &PassContext = ctx;
        let fail = |e: String| PassError::failed(e);
        cfg.check().map_err(fail)?;
        let cat = catalog::catalog(&cfg).map_err(fail)?;
        let kits = flow::kits(ctx, &cfg).map_err(fail)?;
        let out = ctx.output(LIBRARY)?;
        out.write_json("catalog.json", &cat)?;

        let rtl_dir = flow::rtl(ctx, &cat)?;
        let rtl_out = out.dir().join("rtl");
        std::fs::create_dir_all(&rtl_out).map_err(|e| fail(format!("{}: {e}", rtl_out.display())))?;
        for p in cat.iter().filter(|p| p.source.is_some()) {
            for ext in ["x", "v"] {
                let src = rtl_dir.join(format!("{}.{ext}", p.id));
                if src.is_file() {
                    let dst = out.path(&format!("rtl/{}.{ext}", p.id))?;
                    std::fs::copy(&src, &dst).map_err(|e| fail(format!("{}: {e}", dst.display())))?;
                }
            }
        }
        let designs: Vec<_> = cat
            .iter()
            .filter(|p| p.source.is_some())
            .map(flow::design)
            .collect();
        let jobs = flow::jobs(ctx, &cfg, &kits, &designs, &rtl_dir)?;
        let (ran, reused, secs) = flow::run_jobs(ctx, &cfg, &kits, &jobs, &rtl_dir)?;

        let by_id: BTreeMap<&str, &catalog::Primitive> = cat.iter().map(|p| (p.id.as_str(), p)).collect();
        let mut rows: Vec<Row> = Vec::new();
        let mut bad: Vec<String> = Vec::new();
        for j in &jobs {
            for (row, problems) in flow::rows(&cfg, &kits[j.kit].0, j, &by_id, out.dir())? {
                for p in problems {
                    bad.push(format!("{} {} {}: {p}", row.key, row.kit, row.target));
                }
                rows.push(row);
            }
        }
        let order: BTreeMap<&str, usize> = cat.iter().enumerate().map(|(i, p)| (p.id.as_str(), i)).collect();
        let kit_order: BTreeMap<&str, usize> = cfg
            .pdks
            .iter()
            .enumerate()
            .map(|(i, k)| (k.as_str(), i))
            .collect();
        let tgt_order: BTreeMap<&str, usize> = cfg
            .targets
            .iter()
            .enumerate()
            .map(|(i, t)| (t.as_str(), i))
            .collect();
        rows.sort_by_key(|r| {
            (
                order[r.id.as_str()],
                kit_order[r.kit.as_str()],
                tgt_order[r.target.as_str()],
            )
        });

        let mut tool_versions = flow::eda_versions(ctx);
        tool_versions.retain(|k, _| matches!(k.as_str(), "dc_shell" | "fm_shell" | "genus" | "lec"));
        let mut targets = Vec::new();
        for (k, _) in &kits {
            for t in &cfg.targets {
                targets.push(TargetInfo {
                    kit: k.name.clone(),
                    target: t.clone(),
                    period_ps: cfg.period_ps(&k.name, t).map_err(fail)?,
                });
            }
        }
        let library = Library {
            schema: LIBRARY_SCHEMA.into(),
            tool: cfg.tool.name().into(),
            lec_tool: cfg.tool.lec().into(),
            tool_versions,
            kits: kits.iter().map(|(_, i)| i.clone()).collect(),
            targets,
            rows,
        };
        let coverage = json::coverage(&cat, &library.rows);
        out.write_json("library.json", &library)?;
        out.write_json("coverage.json", &coverage)?;
        let summary = summarize(&cfg, &cat, &library.rows, ran, reused, secs);
        out.write_json("summary.json", &summary)?;
        if cfg.views {
            ctx.dir()
                .write_view(VIEW, &view::view(&library, &coverage, &summary))?;
        }
        if !bad.is_empty() {
            bad.sort();
            let n = bad.len();
            bad.truncate(60);
            return Err(fail(format!("{n} problems:\n{}", bad.join("\n"))));
        }
        Ok(())
    }

    fn verify(&self, ctx: &PassContext) -> Result<VerifyReport, PassError> {
        verify::verify(ctx)
    }
}

fn summarize(
    cfg: &CharacterizeConfig,
    cat: &[catalog::Primitive],
    rows: &[Row],
    ran: u64,
    reused: u64,
    secs: f64,
) -> Summary {
    let mut by_kind = BTreeMap::new();
    for p in cat {
        *by_kind.entry(format!("{:?}", p.kind).to_lowercase()).or_default() += 1;
    }
    let pending = cat.iter().filter(|p| p.pending.is_some()).count() as u64;
    let provisional = cat.iter().filter(|p| p.provisional).count() as u64;
    Summary {
        schema: SUMMARY_SCHEMA.into(),
        tool: cfg.tool.name().into(),
        primitives: cat.len() as u64,
        by_kind,
        measured: cat.len() as u64 - pending - provisional,
        provisional,
        pending,
        rows: rows.len() as u64,
        lec_equivalent: rows.iter().filter(|r| r.lec.equivalent).count() as u64,
        designware_rows: rows.iter().filter(|r| r.designware).count() as u64,
        jobs_run: ran,
        jobs_reused: reused,
        tool_seconds: (secs * 10.0).round() / 10.0,
    }
}

/// The datapath element type names (`f8E4M3FN`, `i32`, `ui8`, `i1`) in the catalog's spelling.
pub fn type_name(mlir: &str) -> String {
    mlir.to_ascii_lowercase()
}

/// Keys of the catalog, for the coverage check.
pub fn keys(cat: &[catalog::Primitive]) -> BTreeSet<String> {
    cat.iter().map(|p| p.key.clone()).collect()
}

pub fn hash_text(s: &str) -> String {
    canon::hash_bytes(s.as_bytes())
}
