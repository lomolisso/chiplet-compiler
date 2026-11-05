//! Every external resource family and its implementations (design §3.7; docs/framework/adapters.md), in
//! one place. Each registry selects an implementation by the name its config key holds; the traits are
//! `cc_core::adapters`, the implementations stay next to the code that uses them.
//!
//! | Family | Registry | Config key |
//! |---|---|---|
//! | synthesis flow | [`SYNTH_FLOWS`] | `[passes.characterize] tool` |
//! | SMT solver | [`SMT_SOLVERS`] | `[tools.smt] solver` (the lemmas: the solver declaring them) |
//! | clique | [`CLIQUE_SOLVERS`] | `[passes.merge.clique] algorithm` |
//! | maximum independent set | [`MIS_SOLVERS`] | `[passes.mine.mis] algorithm` |
//! | frequent subgraph miner | [`MINERS`] | `[passes.mine] miner` |
//! | HLS front end | [`HLS_FRONTENDS`] | `[passes.allo_emit] frontend` (planned pass) |
//! | place and route | [`PNR_FLOWS`] | none yet (later versions) |

use std::path::Path;

use cc_core::adapters::{HlsCapabilities, HlsFrontend, Named, PnrFlow, Registry};
use cc_core::runner::BoundRunner;
use cc_core::{CoreError, Result};

pub use crate::eda::FLOWS as SYNTH_FLOWS;
pub use crate::merge::algo::clique::SOLVERS as CLIQUE_SOLVERS;
pub use crate::mine::algo::grami::MINERS;
pub use crate::mine::algo::mis::SOLVERS as MIS_SOLVERS;
pub use cc_smt::SOLVERS as SMT_SOLVERS;

/// The HLS front ends.
pub static HLS_FRONTENDS: Registry<dyn HlsFrontend> = Registry::new("HLS front end", &[&Allo]);

/// The place-and-route flows: Innovus (industry flow) and OpenROAD (open flow) in later versions.
pub static PNR_FLOWS: Registry<dyn PnrFlow> = Registry::new("place-and-route flow", &[]);

/// Allo (Chen et al., PLDI 2024) with its XLS backend: declared, not implemented until the `allo_emit`
/// and `xls` passes exist.
#[derive(Debug, Clone, Copy)]
pub struct Allo;

impl Named for Allo {
    fn name(&self) -> &'static str {
        "allo"
    }
}

impl HlsFrontend for Allo {
    fn capabilities(&self) -> HlsCapabilities {
        HlsCapabilities {
            implemented: false,
            external_modules: false,
            targets: &["xls"],
        }
    }

    fn generate(&self, _runner: &BoundRunner, _src: &Path, top: &str, _out: &Path) -> Result<Vec<String>> {
        Err(CoreError::Tool {
            tool: "allo".into(),
            message: format!("{top}: the Allo front end is not implemented yet (the allo_emit and xls passes)"),
        })
    }
}
