//! High-level synthesis front ends (design §3.7, §4.13, §4.14): a unit description to RTL. Implementation:
//! Allo with XLS, a stub until the `allo_emit` and `xls` passes exist (`cc_passes::adapters`).

use std::path::Path;

use super::Named;
use crate::error::Result;
use crate::runner::BoundRunner;

/// What a front end supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HlsCapabilities {
    /// Generates RTL now (false: declared, not implemented yet).
    pub implemented: bool,
    /// Instantiates external Verilog modules (IP cores) from the description.
    pub external_modules: bool,
    /// Its backends, by name.
    pub targets: &'static [&'static str],
}

/// A front end: the unit description in `src` (its own language), top function `top`, to Verilog in
/// `out`; returns the Verilog files written.
pub trait HlsFrontend: Named {
    fn capabilities(&self) -> HlsCapabilities;
    fn generate(&self, runner: &BoundRunner, src: &Path, top: &str, out: &Path) -> Result<Vec<String>>;
}
