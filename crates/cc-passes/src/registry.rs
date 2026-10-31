//! Every pass, in one place (spec §3.2). Adding a pass is one line here; the pipeline derives the order
//! from `requires`/`provides`.

use cc_pipeline::Pass;

pub fn registry() -> Vec<Box<dyn Pass>> {
    vec![
        // framework self-test.
        Box::new(crate::hello_a::HelloA),
        Box::new(crate::hello_b::HelloB),
        Box::new(crate::hello_c::HelloC),
        // another change
        Box::new(crate::export::Export),
    ]
}
