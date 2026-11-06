//! One GraMi run (docs/tools/grami.md §4): the directed subgraph build of our image, exact
//! (`limitedtime=false`), in its own job directory, under the container's memory and time caps.
//! `maxNodes` is `max_vertices` (our patch `third_party/patches/grami/0002`): GraMi treats a larger
//! pattern as infrequent, so it is neither reported nor extended; the node count only grows along an
//! extension, so every pattern within the bound is still found with its support. Results
//! are cached in `cache/grami/<key>/tau<τ>/` (outside `out/`): `Output.txt` holds GraMi's timing, which
//! must not reach the artifacts.

use std::path::{Path, PathBuf};
use std::time::Instant;

use cc_core::adapters::{MineJob, Miner, MinerCapabilities, Named, Registry};
use cc_core::runner::{BoundRunner, Caps, Invocation, KillReason, Mount};
use cc_core::{CoreError, Result};
use cc_pipeline::{PassContext, PassError};

/// What a run gave.
pub use cc_core::adapters::MineRun as Run;

/// The miners, by `[passes.mine] miner`.
pub static MINERS: Registry<dyn Miner> = Registry::new("miner", &[&Grami]);

/// GraMi (Elseidy, Abdelhamid, Skiadopoulos and Kalnis, PVLDB 7(7), 2014), its directed build in the
/// toolchain image.
#[derive(Debug, Clone, Copy)]
pub struct Grami;

/// The cache directory of a family graph and a threshold.
pub fn job_dir(cache: &Path, key: &str, tau: u64) -> PathBuf {
    cache.join("grami").join(key).join(format!("tau{tau}"))
}

/// Run `miner` on `lg` at support `tau` within `timeout_s` in its cache directory under the pass's
/// `cache/`, or reuse a cached result.
#[allow(clippy::too_many_arguments)]
pub fn run(
    ctx: &PassContext,
    miner: &dyn Miner,
    key: &str,
    lg: &str,
    tau: u64,
    heap_gb: u64,
    max_vertices: usize,
    timeout_s: f64,
) -> Result<Run, PassError> {
    let dir = job_dir(&ctx.dir().path().join("cache"), key, tau);
    Ok(miner.mine(
        &ctx.docker(),
        &MineJob {
            dir: &dir,
            graph: lg,
            support: tau,
            max_vertices,
            memory_gb: heap_gb,
            timeout_s,
        },
    )?)
}

fn failed(message: String) -> CoreError {
    CoreError::Tool {
        tool: "grami".into(),
        message,
    }
}

fn io(dir: &Path) -> impl Fn(std::io::Error) -> CoreError + '_ {
    move |e| failed(format!("{}: {e}", dir.display()))
}

impl Named for Grami {
    fn name(&self) -> &'static str {
        "grami"
    }
}

impl Miner for Grami {
    fn capabilities(&self) -> MinerCapabilities {
        MinerCapabilities {
            directed: true,
            vertex_bound: true,
            mni_support: true,
        }
    }

    /// Exact (`limitedtime=false`), in a container of `memory_gb + 2` GB (the Java heap and the JVM) and
    /// one CPU.
    fn mine(&self, runner: &BoundRunner, job: &MineJob) -> Result<Run> {
        let dir = job.dir;
        let done = dir.join("done");
        let output = dir.join("Output.txt");
        if done.is_file()
            && let Ok(t) = std::fs::read_to_string(&output)
        {
            return Ok(Run::Done(t));
        }
        std::fs::create_dir_all(dir).map_err(io(dir))?;
        let _ = std::fs::remove_file(&output);
        std::fs::write(dir.join("graph.lg"), job.graph).map_err(io(dir))?;
        let heap_gb = job.memory_gb;
        let mem = (heap_gb + 2) as f64;
        if runner.caps().memory_gb < mem {
            return Err(failed(format!(
                "GraMi needs a {mem} GB container (heap_gb + 2) but the cap is {} GB",
                runner.caps().memory_gb
            )));
        }
        let base = *runner.caps();
        let r = runner.clone().with_caps(Caps {
            memory_gb: mem,
            cpus: 1.0,
            timeout_s: job.timeout_s,
            ..base
        });
        let cmd = format!(
            "exec java -Xmx{heap_gb}g -XX:ActiveProcessorCount=1 -XX:+UseSerialGC -cp \"$GRAMI_DIRECTED_CP\" \
             Dijkstra.main filename=graph.lg datasetFolder=/work/ freq={tau} limitedtime=false \
             maxLabelAppearance=-1 maxNodes={max_vertices} approximate=1",
            tau = job.support,
            max_vertices = job.max_vertices
        );
        let inv = Invocation::new("grami", "sh")
            .args(["-c", &cmd])
            .cwd("/work")
            .mount(Mount {
                src: dir.to_path_buf(),
                dst: "/work".into(),
                readonly: false,
            });
        let t0 = Instant::now();
        let o = r.run(&inv)?;
        if o.killed == Some(KillReason::Timeout) {
            return Ok(Run::Timeout);
        }
        let o = o.check()?;
        let text = std::fs::read_to_string(&output).map_err(|e| {
            failed(format!(
                "GraMi wrote no {} ({e}); see {}",
                output.display(),
                o.stderr.display()
            ))
        })?;
        let err = std::fs::read_to_string(&o.stderr).unwrap_or_default();
        if err.contains("Exception") || err.contains("Error") {
            return Err(failed(format!(
                "GraMi reported an error; see {}",
                o.stderr.display()
            )));
        }
        std::fs::write(&done, format!("{:.3}\n", t0.elapsed().as_secs_f64())).map_err(io(dir))?;
        Ok(Run::Done(text))
    }
}
