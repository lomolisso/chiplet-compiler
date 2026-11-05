//! Place and route (design §1, §3.7): declared for later versions (Innovus in the industry flow, OpenROAD
//! in the open flow); no implementation is registered yet.

use std::path::Path;

use super::Named;
use super::synth::Kit;
use crate::error::Result;
use crate::runner::BoundRunner;

/// What a place-and-route flow supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PnrCapabilities {
    pub licensed: bool,
    /// The synthesis flow whose netlists it takes (`dc`, `yosys`).
    pub synth_flow: &'static str,
}

/// A place-and-route flow: a gate netlist on a kit (with its LEF) to a placed and routed design in `dir`.
pub trait PnrFlow: Named {
    fn capabilities(&self) -> PnrCapabilities;
    fn place_and_route(&self, runner: &BoundRunner, kit: &Kit, netlist: &Path, top: &str, dir: &Path) -> Result<()>;
}
