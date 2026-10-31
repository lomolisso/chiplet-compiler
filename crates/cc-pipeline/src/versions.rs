//! Tool versions for cache keys: `program version_args…` through the pass's runner, first non-empty
//! output line, memoised per (runner, tool) for the lifetime of the pipeline.

use std::collections::BTreeMap;
use std::sync::Mutex;

use cc_core::runner::{Caps, Invocation, Runners, ToolLog};
use cc_core::{CancelToken, CoreError};

use crate::pass::{RunnerKind, ToolSpec};

#[derive(Debug, Default)]
pub struct ToolVersions {
    cache: Mutex<BTreeMap<(RunnerKind, &'static str), String>>,
}

const PROBE_CAPS: Caps = Caps {
    memory_gb: 1.0,
    cpus: 1.0,
    timeout_s: 120.0,
    pids_limit: 256,
};

impl ToolVersions {
    pub fn resolve(
        &self,
        specs: &[ToolSpec],
        runners: &Runners,
        log: &ToolLog,
    ) -> Result<BTreeMap<String, String>, CoreError> {
        let mut out = BTreeMap::new();
        for spec in specs {
            let key = (spec.runner, spec.name);
            let cached = self
                .cache
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .get(&key)
                .cloned();
            let version = match cached {
                Some(v) => v,
                None => {
                    let v = probe(spec, runners, log)?;
                    self.cache
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .insert(key, v.clone());
                    v
                }
            };
            out.insert(spec.name.to_string(), version);
        }
        Ok(out)
    }
}

fn probe(spec: &ToolSpec, runners: &Runners, log: &ToolLog) -> Result<String, CoreError> {
    let runner = match spec.runner {
        RunnerKind::Host => &runners.host,
        RunnerKind::Docker => &runners.docker,
        RunnerKind::Eda => &runners.eda,
    };
    let inv = Invocation::new(format!("{}-version", spec.name), spec.program)
        .args(spec.version_args.iter().copied());
    let out = runner.run(&inv, &PROBE_CAPS, log, &CancelToken::new())?.check()?;
    let stdout = out.stdout_string()?;
    let stderr = std::fs::read_to_string(&out.stderr).unwrap_or_default();
    stdout
        .lines()
        .chain(stderr.lines())
        .map(str::trim)
        .find(|l| !l.is_empty())
        .map(str::to_string)
        .ok_or_else(|| CoreError::Tool {
            tool: spec.name.into(),
            message: "version probe printed nothing".into(),
        })
}
