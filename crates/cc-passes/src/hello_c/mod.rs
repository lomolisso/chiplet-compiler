//! `hello_c`: independent of `hello_a → hello_b`; calls `/bin/echo` through the `HostRunner`. Exercises a
//! parallel branch, tool logging and tool versions in the cache key.

mod config;
mod verify;

use std::time::Duration;

use cc_core::runner::Invocation;
use cc_core::{ArtifactKind, CoreError, PassSection, VerifyReport};
use cc_pipeline::{Pass, PassContext, PassError, RunnerKind, ToolSpec};
use serde::{Deserialize, Serialize};

pub use config::HelloCConfig;

pub const HELLO_C: ArtifactKind = ArtifactKind("hello.c");
pub const ECHO_FILE: &str = "echo.json";

const TOOLS: &[ToolSpec] = &[ToolSpec {
    name: "echo",
    runner: RunnerKind::Host,
    program: "/bin/echo",
    version_args: &["--version"],
}];

/// `out/hello.c/echo.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Echo {
    pub text: String,
    pub stdout: String,
    /// The tool call's stdout log, relative to the pass directory.
    pub log: String,
}

#[derive(Debug)]
pub struct HelloC;

impl Pass for HelloC {
    fn name(&self) -> &'static str {
        "hello_c"
    }

    fn version(&self) -> u32 {
        1
    }

    fn requires(&self) -> &[ArtifactKind] {
        &[]
    }

    fn provides(&self) -> &[ArtifactKind] {
        &[HELLO_C]
    }

    fn tools(&self) -> &[ToolSpec] {
        TOOLS
    }

    fn model_dependent(&self) -> bool {
        false
    }

    fn check_config(&self, section: &PassSection) -> Result<(), CoreError> {
        section.parse::<HelloCConfig>().map(drop)
    }

    fn run(&self, ctx: &mut PassContext) -> Result<(), PassError> {
        let cfg: HelloCConfig = ctx.options()?;
        ctx.cancel_token().sleep(Duration::from_millis(cfg.sleep_ms))?;
        let out = ctx
            .host()
            .run(&Invocation::new("echo", "/bin/echo").arg(&cfg.text))?
            .check()?;
        let log = out
            .stdout
            .strip_prefix(ctx.dir().path())
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|_| out.stdout.display().to_string());
        let e = Echo {
            text: cfg.text,
            stdout: out.stdout_string()?,
            log,
        };
        ctx.output(HELLO_C)?.write_json(ECHO_FILE, &e)?;
        Ok(())
    }

    fn verify(&self, ctx: &PassContext) -> Result<VerifyReport, PassError> {
        verify::verify(ctx)
    }
}
