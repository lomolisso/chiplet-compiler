//! Parallel batches, the solver version and a verdict cache (docs/framework/smt.md, "Running").

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use cc_core::CoreError;
use cc_core::runner::{BoundRunner, Caps, Invocation};
use rayon::prelude::*;

use crate::{Batch, Location, Solver, Verdict, run_batch};

/// The content hash of a query (blake3, hex).
pub fn query_hash(text: &str) -> String {
    blake3::hash(text.as_bytes()).to_hex().to_string()
}

/// Run `files` in `shards` batches at once (a deterministic round-robin split, each one process) and
/// merge the answers. Each batch gets an equal share of the runner's memory and CPU caps, so together
/// they stay within them. Every batch must report the same solver version.
pub fn run_sharded(
    runner: &BoundRunner,
    location: &Location,
    solver: Solver,
    dir: &Path,
    files: &[String],
    timeout_s: u64,
    shards: usize,
) -> Result<Batch, CoreError> {
    let shards = shards.clamp(1, files.len().max(1));
    let c = runner.caps();
    let share = Caps {
        memory_gb: c.memory_gb / shards as f64,
        cpus: (c.cpus / shards as f64).max(1.0),
        timeout_s: c.timeout_s,
        pids_limit: c.pids_limit,
    };
    let runner = &runner.clone().with_caps(share);
    let parts: Vec<Vec<String>> = (0..shards)
        .map(|k| files.iter().skip(k).step_by(shards).cloned().collect())
        .collect();
    let batches: Vec<Batch> = parts
        .par_iter()
        .filter(|p| !p.is_empty())
        .map(|p| run_batch(runner, location, solver, dir, p, timeout_s))
        .collect::<Result<_, _>>()?;
    let mut out = Batch {
        solver,
        version: String::new(),
        verdicts: BTreeMap::new(),
    };
    for b in batches {
        if out.version.is_empty() {
            out.version = b.version.clone();
        } else if b.version != out.version {
            return Err(CoreError::Tool {
                tool: format!("smt-{}", solver.name()),
                message: format!("two versions in one run: {} and {}", out.version, b.version),
            });
        }
        out.verdicts.extend(b.verdicts);
    }
    Ok(out)
}

/// The first line of `<solver> --version`.
pub fn solver_version(runner: &BoundRunner, location: &Location, solver: Solver) -> Result<String, CoreError> {
    let program = match location {
        Location::Image => solver.name().to_string(),
        Location::Host(p) => p.get(&solver).cloned().unwrap_or_else(|| solver.name().to_string()),
    };
    let inv = Invocation::new(format!("smt-{}", solver.name()), program).arg("--version");
    let out = runner.run(&inv)?;
    let text = out.stdout_string()?;
    out.check()?;
    Ok(text.lines().next().unwrap_or_default().trim().to_string())
}

/// Verdicts by (solver, version, query hash): a directory of one small file per answered query. Only
/// `unsat` and `sat` are stored; anything else is asked again.
#[derive(Debug, Clone)]
pub struct Cache {
    dir: PathBuf,
}

impl Cache {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Cache { dir: dir.into() }
    }

    fn path(&self, solver: Solver, version: &str, hash: &str) -> PathBuf {
        let v = query_hash(version);
        self.dir.join(format!("{}-{}", solver.name(), &v[..16])).join(hash)
    }

    pub fn get(&self, solver: Solver, version: &str, hash: &str) -> Option<Verdict> {
        let text = std::fs::read_to_string(self.path(solver, version, hash)).ok()?;
        match text.lines().next()? {
            "unsat" => Some(Verdict::Unsat),
            "sat" => Some(Verdict::Sat {
                model: text.lines().skip(1).collect::<Vec<_>>().join("\n"),
            }),
            _ => None,
        }
    }

    pub fn put(&self, solver: Solver, version: &str, hash: &str, v: &Verdict) -> std::io::Result<()> {
        let text = match v {
            Verdict::Unsat => "unsat\n".to_string(),
            Verdict::Sat { model } => format!("sat\n{model}\n"),
            _ => return Ok(()),
        };
        let p = self.path(solver, version, hash);
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d)?;
        }
        let tmp = p.with_extension("tmp");
        std::fs::write(&tmp, text)?;
        std::fs::rename(tmp, p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caches_only_answers() {
        let d = std::env::temp_dir().join(format!("cc-smt-cache-{}", std::process::id()));
        let c = Cache::new(&d);
        c.put(Solver::Z3, "Z3 5.3.3", "h1", &Verdict::Unsat).unwrap();
        c.put(Solver::Z3, "Z3 5.3.3", "h2", &Verdict::Timeout).unwrap();
        assert_eq!(c.get(Solver::Z3, "Z3 5.3.3", "h1"), Some(Verdict::Unsat));
        assert_eq!(c.get(Solver::Z3, "Z3 5.3.3", "h2"), None);
        assert_eq!(c.get(Solver::Z3, "Z3 5.3.4", "h1"), None);
        assert_eq!(c.get(Solver::Bitwuzla, "Z3 5.3.3", "h1"), None);
        let _ = std::fs::remove_dir_all(d);
    }
}
