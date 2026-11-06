//! The two synthesis flows (design §3.7): the industry flow's Design Compiler (licensed, through
//! `eda-run`; Formality for the optional equivalence check) and the open flow's Yosys + ABC (the toolchain
//! image; Yosys' equivalence flow). `[passes.characterize] tool` selects one by name.

use std::path::Path;

use cc_core::adapters::{
    FlowRunners, Lec, LecOptions, LibraryFormat, Measured, Named, Registry, ReportTexts, SynthCapabilities,
    SynthFlow, SynthJob,
};
use cc_core::runner::{Caps, EdaTool, Invocation, Mount, ToolOutcome};
use cc_core::{CoreError, Result};

use super::{Kit, parse, scripts};

/// Every synthesis flow, by `[passes.characterize] tool`.
pub static FLOWS: Registry<dyn SynthFlow> = Registry::new("synthesis flow", &[&Dc, &Yosys]);

/// Design Compiler (`compile_ultra`) with Formality.
#[derive(Debug, Clone, Copy)]
pub struct Dc;

/// Yosys + ABC with Yosys' equivalence flow.
#[derive(Debug, Clone, Copy)]
pub struct Yosys;

fn killed(dir: &Path, what: &str, o: &ToolOutcome) -> Result<()> {
    match &o.killed {
        Some(k) => Err(CoreError::Tool {
            tool: what.to_string(),
            message: format!("{}: killed ({k:?})", dir.display()),
        }),
        None => Ok(()),
    }
}

fn write(path: &Path, text: &str) -> Result<()> {
    std::fs::write(path, text).map_err(|source| CoreError::Io {
        path: path.to_path_buf(),
        source,
    })
}

/// Copy a call's stdout into the job directory as `name`.
fn keep(dir: &Path, name: &str, o: &ToolOutcome) -> Result<()> {
    std::fs::copy(&o.stdout, dir.join(name)).map_err(|source| CoreError::Io {
        path: dir.join(name),
        source,
    })?;
    Ok(())
}

impl Named for Dc {
    fn name(&self) -> &'static str {
        "dc"
    }
}

impl SynthFlow for Dc {
    fn capabilities(&self) -> SynthCapabilities {
        SynthCapabilities {
            licensed: true,
            library: LibraryFormat::Db,
            equivalence: "formality",
            hierarchy: true,
            ip_blocks: true,
            flop_only_paths: true,
            leakage: true,
        }
    }

    /// One session per design: a session that synthesized one design gives the next a different result
    /// (a 32-bit divider after a 16-bit one: 15 614 against 9 340 µm²), so every primitive starts from a
    /// fresh tool.
    fn synth_scripts(&self, job: &SynthJob) -> Vec<(String, String)> {
        let preamble = EdaTool::Dc.cap_preamble(&job.caps).join("\n");
        job.designs
            .iter()
            .map(|d| {
                (
                    format!("syn_{}.tcl", d.id),
                    scripts::dc(
                        &preamble,
                        job.kit,
                        std::slice::from_ref(d),
                        job.period_ps,
                        job.cache_dir,
                    ),
                )
            })
            .collect()
    }

    fn lec_scripts(&self, job: &SynthJob) -> Vec<(String, String)> {
        let Some(LecOptions {
            timeout_min,
            strategies,
        }) = job.lec
        else {
            return Vec::new();
        };
        vec![(
            "lec.tcl".into(),
            scripts::formality(
                &EdaTool::Fm.cap_preamble(&job.caps).join("\n"),
                job.kit,
                job.designs,
                *timeout_min,
                strategies,
            ),
        )]
    }

    /// Each design's session log, appended in order (the markers split it again).
    fn synthesize(&self, runners: &FlowRunners, job: &SynthJob, dir: &Path) -> Result<()> {
        let mut log = String::new();
        for d in job.designs {
            let inv = Invocation::new("dc_shell", EdaTool::Dc.program())
                .args(EdaTool::Dc.batch_args(&format!("syn_{}.tcl", d.id)))
                .cwd(dir);
            let o = runners.eda.run(&inv)?;
            killed(dir, &d.id, &o)?;
            log.push_str(&std::fs::read_to_string(&o.stdout).unwrap_or_default());
        }
        write(&dir.join("synth.log"), &log)
    }

    fn check_equivalence(&self, runners: &FlowRunners, _job: &SynthJob, dir: &Path) -> Result<()> {
        let o = runners.eda.run(
            &Invocation::new("fm_shell", EdaTool::Fm.program())
                .args(["-work_path", "fm"])
                .args(EdaTool::Fm.batch_args("lec.tcl"))
                .cwd(dir),
        )?;
        killed(dir, "lec.log", &o)?;
        keep(dir, "lec.log", &o)
    }

    fn measure(&self, r: &ReportTexts, _id: &str, _period_ps: f64, time_unit_ps: f64) -> Measured {
        parse::dc(r.area, r.qor, r.power, time_unit_ps)
    }

    fn equivalence(&self, report: &str) -> Lec {
        parse::formality(report)
    }

    fn used_ip_blocks(&self, log: &str) -> bool {
        parse::designware(log)
    }
}

/// The Liberty file as the container sees it (its directory is mounted at `/kit`).
fn lib_in_container(kit: &Kit) -> String {
    let f = kit
        .liberty
        .file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_default();
    format!("/kit/{f}")
}

impl Named for Yosys {
    fn name(&self) -> &'static str {
        "yosys"
    }
}

impl SynthFlow for Yosys {
    fn capabilities(&self) -> SynthCapabilities {
        SynthCapabilities {
            licensed: false,
            library: LibraryFormat::Liberty,
            equivalence: "yosys-equiv",
            hierarchy: false,
            ip_blocks: false,
            flop_only_paths: false,
            leakage: false,
        }
    }

    /// One script for the batch; the equivalence check runs inside it, after each design's synthesis.
    fn synth_scripts(&self, job: &SynthJob) -> Vec<(String, String)> {
        vec![(
            "run.sh".into(),
            scripts::yosys(
                &lib_in_container(job.kit),
                job.designs,
                job.period_ps,
                job.lec.map(|l| u64::from(l.timeout_min) * 60),
            ),
        )]
    }

    fn lec_scripts(&self, _job: &SynthJob) -> Vec<(String, String)> {
        Vec::new()
    }

    fn synthesize(&self, runners: &FlowRunners, job: &SynthJob, dir: &Path) -> Result<()> {
        let lib_dir = job.kit.liberty.parent().unwrap_or(Path::new("/"));
        let inv = Invocation::new("yosys", "bash")
            .arg("run.sh")
            .cwd("/work")
            .mount(Mount {
                src: dir.to_path_buf(),
                dst: "/work".into(),
                readonly: false,
            })
            .mount(Mount {
                src: lib_dir.to_path_buf(),
                dst: "/kit".into(),
                readonly: true,
            });
        let base = *runners.docker.caps();
        let o = runners
            .docker
            .clone()
            .with_caps(Caps { cpus: 1.0, ..base })
            .run(&inv)?;
        killed(dir, "synth.log", &o)?;
        keep(dir, "synth.log", &o)
    }

    fn check_equivalence(&self, _runners: &FlowRunners, _job: &SynthJob, _dir: &Path) -> Result<()> {
        Ok(())
    }

    fn measure(&self, r: &ReportTexts, id: &str, period_ps: f64, _time_unit_ps: f64) -> Measured {
        parse::yosys(r.area, r.log, id, period_ps)
    }

    fn equivalence(&self, report: &str) -> Lec {
        parse::yosys_equiv(report)
    }

    fn used_ip_blocks(&self, _log: &str) -> bool {
        false
    }
}
