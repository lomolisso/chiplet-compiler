//! Checks on `greeting.json`, recomputed from the config, not from `run`.

use cc_core::{Check, VerifyReport};
use cc_pipeline::{PassContext, PassError};

use super::{GREETING_FILE, Greeting, HELLO_A, HelloAConfig};

pub(super) fn verify(ctx: &PassContext) -> Result<VerifyReport, PassError> {
    let cfg: HelloAConfig = ctx.options()?;
    let out = ctx.output(HELLO_A)?;
    let mut r = VerifyReport::new(ctx.name());
    let path = format!("out/{}/{GREETING_FILE}", HELLO_A.name());
    match out.read_json::<Greeting>(GREETING_FILE) {
        Err(e) => {
            r.push(
                Check::new("greeting_parses", false)
                    .path(&path)
                    .note(e.to_string()),
            );
        }
        Ok(g) => {
            r.push(Check::new("greeting_parses", true).path(&path));
            r.push(Check::new("greeting_matches_config", g.greeting == cfg.greeting).path(&path));
            let n = g.greeting.chars().count() as u64;
            r.push(
                Check::new("char_count", g.chars == n)
                    .path(&path)
                    .count("expected", n)
                    .count("recorded", g.chars),
            );
        }
    }
    Ok(r)
}
