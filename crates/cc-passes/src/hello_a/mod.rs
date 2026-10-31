//! `hello_a`: writes a greeting. Root of the toy chain `hello_a → hello_b`.

mod config;
mod verify;

use std::time::Duration;

use cc_core::{ArtifactKind, CoreError, PassSection, VerifyReport};
use cc_pipeline::{Pass, PassContext, PassError};
use serde::{Deserialize, Serialize};

pub use config::HelloAConfig;

pub const HELLO_A: ArtifactKind = ArtifactKind("hello.a");
pub const GREETING_FILE: &str = "greeting.json";

/// `out/hello.a/greeting.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Greeting {
    pub greeting: String,
    /// Unicode scalar values in `greeting`.
    pub chars: u64,
}

#[derive(Debug)]
pub struct HelloA;

impl Pass for HelloA {
    fn name(&self) -> &'static str {
        "hello_a"
    }

    fn version(&self) -> u32 {
        1
    }

    fn requires(&self) -> &[ArtifactKind] {
        &[]
    }

    fn provides(&self) -> &[ArtifactKind] {
        &[HELLO_A]
    }

    fn model_dependent(&self) -> bool {
        false
    }

    fn check_config(&self, section: &PassSection) -> Result<(), CoreError> {
        section.parse::<HelloAConfig>().map(drop)
    }

    fn run(&self, ctx: &mut PassContext) -> Result<(), PassError> {
        let cfg: HelloAConfig = ctx.options()?;
        ctx.cancel_token().sleep(Duration::from_millis(cfg.sleep_ms))?;
        let g = Greeting {
            chars: cfg.greeting.chars().count() as u64,
            greeting: cfg.greeting,
        };
        ctx.output(HELLO_A)?.write_json(GREETING_FILE, &g)?;
        Ok(())
    }

    fn verify(&self, ctx: &PassContext) -> Result<VerifyReport, PassError> {
        verify::verify(ctx)
    }
}
