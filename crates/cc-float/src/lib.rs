//! Float glue around third-party IP cores (design §4.12, §6.4; docs/framework/float.md): the wrappers'
//! Verilog ([`glue`], [`rtl`]), their SMT proofs through Yosys ([`proof`]), and compositions of cores
//! sized to an error budget ([`compose`]).

pub mod compose;
pub mod glue;
pub mod proof;
pub mod rtl;

pub use glue::{Built, CoreNeed, Dir, Glue, Role, SignOp};
pub use rtl::{Mutation, Rtl};

use cc_smt::float::{Flavour, Format};

/// What went wrong while building or proving a wrapper.
#[derive(Debug, thiserror::Error)]
pub enum FloatError {
    #[error("glue: {0}")]
    Glue(String),
    #[error("core instance: {0}")]
    Core(String),
    #[error("proof: {0}")]
    Proof(String),
    #[error("composition: {0}")]
    Compose(String),
    #[error("{0}")]
    Io(String),
    #[error("SMT term: {0}")]
    Sort(#[from] cc_smt::term::SortError),
    #[error(transparent)]
    Tool(#[from] cc_core::CoreError),
}

/// A format by its `cc_smt` name (`f32`, `bf16`, `f8e4m3fn`, `f<e>m<m>`, `f<e>m<m>fn`).
pub fn format_by_name(n: &str) -> Option<Format> {
    let known = [
        Format::F16,
        Format::BF16,
        Format::F32,
        Format::F64,
        Format::E4M3FN,
        Format::E5M2,
    ];
    if let Some(f) = known.into_iter().find(|f| f.name() == n) {
        return Some(f);
    }
    let rest = n.strip_prefix('f')?;
    let (rest, flavour) = match rest.strip_suffix("fn") {
        Some(r) => (r, Flavour::FiniteNan),
        None => (rest, Flavour::Ieee),
    };
    let (e, m) = rest.split_once('m')?;
    Some(Format {
        e: e.parse().ok()?,
        m: m.parse().ok()?,
        flavour,
    })
}
