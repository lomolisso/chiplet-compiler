//! One GraMi run (docs/tools/grami.md §4): the directed subgraph build of our image, exact
//! (`limitedtime=false`), in its own job directory, under the container's memory and time caps.
//! `maxLabelAppearance` is `max_vertices`: a pattern within `max_vertices` repeats no label more often,
//! and the bound is anti-monotone (GraMi's pruner treats a pattern over it as infrequent, so it is not
//! extended), so nothing the pass keeps is lost. Results
//! are cached in `cache/grami/<key>/tau<τ>/` (outside `out/`): `Output.txt` holds GraMi's timing, which
//! must not reach the artifacts.

use std::path::{Path, PathBuf};
use std::time::Instant;

use cc_core::runner::{Caps, Invocation, KillReason, Mount};
use cc_pipeline::{PassContext, PassError};

/// What a run gave.
#[derive(Debug, Clone)]
pub enum Run {
    /// GraMi's `Output.txt`.
    Done(String),
    /// Killed at the time limit.
    Timeout,
}

/// The cache directory of a family graph and a threshold.
pub fn job_dir(cache: &Path, key: &str, tau: u64) -> PathBuf {
    cache.join("grami").join(key).join(format!("tau{tau}"))
}

/// Run GraMi on `lg` at support `tau` within `timeout_s`, or reuse a cached result.
pub fn run(
    ctx: &PassContext,
    key: &str,
    lg: &str,
    tau: u64,
    heap_gb: u64,
    max_vertices: usize,
    timeout_s: f64,
) -> Result<Run, PassError> {
    let dir = job_dir(&ctx.dir().path().join("cache"), key, tau);
    let done = dir.join("done");
    let output = dir.join("Output.txt");
    if done.is_file()
        && let Ok(t) = std::fs::read_to_string(&output)
    {
        return Ok(Run::Done(t));
    }
    std::fs::create_dir_all(&dir).map_err(|e| PassError::failed(format!("{}: {e}", dir.display())))?;
    let _ = std::fs::remove_file(&output);
    std::fs::write(dir.join("graph.lg"), lg).map_err(|e| PassError::failed(format!("{}: {e}", dir.display())))?;
    let r = ctx.docker();
    let mem = (heap_gb + 2) as f64;
    if r.caps().memory_gb < mem {
        return Err(PassError::failed(format!(
            "GraMi needs a {mem} GB container (heap_gb + 2) but the cap is {} GB",
            r.caps().memory_gb
        )));
    }
    let base = *r.caps();
    let r = r.with_caps(Caps {
        memory_gb: mem,
        cpus: 1.0,
        timeout_s,
        ..base
    });
    let cmd = format!(
        "exec java -Xmx{heap_gb}g -XX:ActiveProcessorCount=1 -XX:+UseSerialGC -cp \"$GRAMI_DIRECTED_CP\" \
         Dijkstra.main filename=graph.lg datasetFolder=/work/ freq={tau} limitedtime=false \
         maxLabelAppearance={max_vertices} approximate=1"
    );
    let inv = Invocation::new("grami", "sh")
        .args(["-c", &cmd])
        .cwd("/work")
        .mount(Mount {
            src: dir.clone(),
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
        PassError::failed(format!(
            "GraMi wrote no {} ({e}); see {}",
            output.display(),
            o.stderr.display()
        ))
    })?;
    let err = std::fs::read_to_string(&o.stderr).unwrap_or_default();
    if err.contains("Exception") || err.contains("Error") {
        return Err(PassError::failed(format!(
            "GraMi reported an error; see {}",
            o.stderr.display()
        )));
    }
    std::fs::write(&done, format!("{:.3}\n", t0.elapsed().as_secs_f64()))
        .map_err(|e| PassError::failed(format!("{}: {e}", done.display())))?;
    Ok(Run::Done(text))
}
