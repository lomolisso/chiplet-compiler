//! Every pass, in one place (design §3.2). Adding a pass is one line here; the pipeline derives the order
//! from `requires`/`provides`.

use cc_pipeline::Pass;

pub fn registry() -> Vec<Box<dyn Pass>> {
    vec![
        // Framework self-test.
        Box::new(crate::hello_a::HelloA),
        Box::new(crate::hello_b::HelloB),
        Box::new(crate::hello_c::HelloC),
        // Model passes.
        Box::new(crate::export::Export),
        Box::new(crate::import::Import),
        Box::new(crate::components::Components),
        Box::new(crate::normalize::Normalize),
        Box::new(crate::kernels::Kernels),
    ]
}
