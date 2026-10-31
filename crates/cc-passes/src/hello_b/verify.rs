//! Recompute the message from the input and config, and re-check the input-hash certificate.

use cc_core::{Check, VerifyReport};
use cc_pipeline::{PassContext, PassError};

use super::{HELLO_B, HelloBConfig, MESSAGE_FILE, Message};
use crate::hello_a::{GREETING_FILE, Greeting, HELLO_A};

pub(super) fn verify(ctx: &PassContext) -> Result<VerifyReport, PassError> {
    let cfg: HelloBConfig = ctx.options()?;
    let input = ctx.input(HELLO_A)?;
    let mut r = VerifyReport::new(ctx.name());
    let path = format!("out/{}/{MESSAGE_FILE}", HELLO_B.name());
    let m = match ctx.output(HELLO_B)?.read_json::<Message>(MESSAGE_FILE) {
        Ok(m) => m,
        Err(e) => {
            r.push(
                Check::new("message_parses", false)
                    .path(&path)
                    .note(e.to_string()),
            );
            return Ok(r);
        }
    };
    r.push(Check::new("message_parses", true).path(&path));
    let g: Greeting = input.read_json(GREETING_FILE)?;
    r.push(
        Check::new(
            "message_recomputed",
            m.message == format!("{}{}", g.greeting, cfg.suffix),
        )
        .path(&path),
    );
    let current = input.record()?.hash;
    r.push(
        Check::new("certificate_input_hash", m.input_hash == current)
            .path(&path)
            .note(format!("input hash {current}")),
    );
    Ok(r)
}
