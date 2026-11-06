//! SMT solvers (design §3.7, §5; docs/framework/smt.md): a solver answers SMT-LIB 2 files out of process.
//! Implementations: Bitwuzla and Z3 (`cc_smt`). `[tools.smt] solver` picks the one that proves; the other
//! re-checks, and the lemmas go to the solver that declares [`SmtCapabilities::lemmas`].

use super::Named;

/// What a solver accepts, as observed on the pinned releases (docs/framework/smt.md, "Scripts").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmtCapabilities {
    /// QF_BV.
    pub bit_vectors: bool,
    /// Uninterpreted functions over bit-vectors and the interchange float sorts.
    pub uninterpreted_functions: bool,
    /// The IEEE interchange float sorts (Float16, Float32, Float64, Float128).
    pub interchange_floats: bool,
    /// Float sorts of any exponent and significand width (`(_ FloatingPoint 8 8)`).
    pub any_float_sort: bool,
    /// Proves the lemmas: the float encodings against native sorts of any width (needs
    /// `any_float_sort`) and the composed multiplier steps, which bit-blasting alone does not finish.
    pub lemmas: bool,
}

/// An SMT solver binary, in the toolchain image or on the host. It is called as `<program> <file>` on a
/// query that ends with `(check-sat)` and `(get-model)`, and as `<program> --version`.
pub trait SmtSolver: Named {
    fn capabilities(&self) -> SmtCapabilities;

    /// The program name (on the image's `PATH`; the host path comes from `[tools.smt]`).
    fn program(&self) -> &'static str;
}
