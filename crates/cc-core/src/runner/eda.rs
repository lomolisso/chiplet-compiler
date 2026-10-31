//! `EdaRunner` (stub; completed in another change). Host EDA tools run only through `eda-run` in an interactive
//! bash, inheriting the session environment, never by sourcing `eda-on` in the driver
//! (docs/context/tooling/eda_environment.md §2).

use std::path::PathBuf;
use std::time::Duration;

use crate::cancel::CancelToken;
use crate::error::{CoreError, Result};
use crate::runner::{Caps, Invocation, ToolLog, ToolOutcome, ToolRunner};

#[derive(Debug, Clone)]
pub struct EdaRunner {
    pub eda_toggle: Option<PathBuf>,
    pub watchdog_interval: Duration,
}

impl EdaRunner {
    pub fn new(eda_toggle: Option<PathBuf>, watchdog_interval: Duration) -> Self {
        Self {
            eda_toggle,
            watchdog_interval,
        }
    }

    /// The verified form: `bash --norc -i -c '. "$EDA_TOGGLE"; eda-run "$@"' eda-run <tool> … </dev/null`
    /// with `EDA_TOGGLE` in the environment.
    pub fn eda_command(&self, inv: &Invocation) -> Result<EdaCommand> {
        let toggle = self.eda_toggle.as_ref().ok_or_else(|| CoreError::Tool {
            tool: inv.tool.clone(),
            message: "paths.eda_toggle is not set (config.local.toml)".into(),
        })?;
        let mut args: Vec<String> = vec![
            "--norc".into(),
            "-i".into(),
            "-c".into(),
            r#". "$EDA_TOGGLE"; eda-run "$@""#.into(),
            "eda-run".into(),
            inv.program.clone(),
        ];
        args.extend(inv.args.iter().cloned());
        Ok(EdaCommand {
            program: "bash".into(),
            args,
            env: vec![("EDA_TOGGLE".into(), toggle.display().to_string())],
        })
    }
}

/// The host command line `EdaRunner` executes for one invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdaCommand {
    pub program: String,
    pub args: Vec<String>,
    /// Added to the inherited environment (which must keep Lmod's `MODULEPATH` / `module`).
    pub env: Vec<(String, String)>,
}

impl ToolRunner for EdaRunner {
    fn name(&self) -> &'static str {
        "eda"
    }

    fn run(
        &self,
        _inv: &Invocation,
        _caps: &Caps,
        _log: &ToolLog,
        _cancel: &CancelToken,
    ) -> Result<ToolOutcome> {
        Err(CoreError::Unimplemented {
            runner: "eda",
            task: "a later release",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_matches_verified_form() {
        let r = EdaRunner::new(Some("/home/u/bin/eda-toggle.sh".into()), Duration::from_secs(2));
        let c = r
            .eda_command(&Invocation::new("dc", "dc_shell").args(["-f", "x.tcl"]))
            .unwrap();
        assert_eq!(c.program, "bash");
        assert_eq!(
            c.args.join(" "),
            r#"--norc -i -c . "$EDA_TOGGLE"; eda-run "$@" eda-run dc_shell -f x.tcl"#
        );
        assert_eq!(
            c.env,
            [("EDA_TOGGLE".to_string(), "/home/u/bin/eda-toggle.sh".to_string())]
        );
    }
}
