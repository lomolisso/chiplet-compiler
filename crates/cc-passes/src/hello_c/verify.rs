//! The recorded stdout equals the configured text, and the tool call is in the log.

use cc_core::{Check, VerifyReport};
use cc_pipeline::{PassContext, PassError};

use super::{ECHO_FILE, Echo, HELLO_C, HelloCConfig};

pub(super) fn verify(ctx: &PassContext) -> Result<VerifyReport, PassError> {
    let cfg: HelloCConfig = ctx.options()?;
    let mut r = VerifyReport::new(ctx.name());
    let path = format!("out/{}/{ECHO_FILE}", HELLO_C.name());
    let e = match ctx.output(HELLO_C)?.read_json::<Echo>(ECHO_FILE) {
        Ok(e) => e,
        Err(err) => {
            r.push(Check::new("echo_parses", false).path(&path).note(err.to_string()));
            return Ok(r);
        }
    };
    r.push(Check::new("echo_parses", true).path(&path));
    r.push(
        Check::new(
            "stdout_is_text",
            e.stdout == format!("{}\n", cfg.text) && e.text == cfg.text,
        )
        .path(&path),
    );
    let logged = ctx.dir().logs_dir().join("tools.jsonl");
    let call_logged = std::fs::read_to_string(&logged).is_ok_and(|s| s.contains("\"tool\":\"echo\""));
    r.push(
        Check::new("tool_call_logged", call_logged)
            .path("logs/tools.jsonl")
            .path(&e.log),
    );
    Ok(r)
}
