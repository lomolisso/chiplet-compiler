//! `hello_b`: reads `hello.a` and appends a suffix. Exercises a dependency edge, input reads and a
//! certificate (the hash of the input it consumed).

mod config;
mod verify;

use std::time::Duration;

use cc_core::{ArtifactKind, CoreError, PassSection, VerifyReport};
use cc_pipeline::{Pass, PassContext, PassError};
use serde::{Deserialize, Serialize};

use crate::hello_a::{GREETING_FILE, Greeting, HELLO_A};

pub use config::HelloBConfig;

pub const HELLO_B: ArtifactKind = ArtifactKind("hello.b");
pub const MESSAGE_FILE: &str = "message.json";

/// `out/hello.b/message.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Message {
    pub message: String,
    /// Certificate: content hash of the `hello.a` artifact this message was derived from.
    pub input_hash: String,
}

#[derive(Debug)]
pub struct HelloB;

impl Pass for HelloB {
    fn name(&self) -> &'static str {
        "hello_b"
    }

    fn version(&self) -> u32 {
        1
    }

    fn requires(&self) -> &[ArtifactKind] {
        &[HELLO_A]
    }

    fn provides(&self) -> &[ArtifactKind] {
        &[HELLO_B]
    }

    fn model_dependent(&self) -> bool {
        false
    }

    fn check_config(&self, section: &PassSection) -> Result<(), CoreError> {
        section.parse::<HelloBConfig>().map(drop)
    }

    fn run(&self, ctx: &mut PassContext) -> Result<(), PassError> {
        let cfg: HelloBConfig = ctx.options()?;
        if cfg.fail {
            return Err(PassError::failed("induced failure (passes.hello_b.fail = true)"));
        }
        ctx.cancel_token().sleep(Duration::from_millis(cfg.sleep_ms))?;
        let input = ctx.input(HELLO_A)?;
        let g: Greeting = input.read_json(GREETING_FILE)?;
        let m = Message {
            message: format!("{}{}", g.greeting, cfg.suffix),
            input_hash: input.record()?.hash,
        };
        ctx.output(HELLO_B)?.write_json(MESSAGE_FILE, &m)?;
        Ok(())
    }

    fn verify(&self, ctx: &PassContext) -> Result<VerifyReport, PassError> {
        verify::verify(ctx)
    }
}
