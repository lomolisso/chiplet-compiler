//! Synthesis flows (design §3.7, §4.9): a flow turns a batch of Verilog designs into gate netlists on one
//! cell library at one clock period, reports area, delay and leakage, and optionally checks each netlist
//! against its RTL. Implementations: Design Compiler (with Formality) and Yosys + ABC
//! (`cc_passes::eda`).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::Named;
use crate::error::Result;
use crate::runner::{BoundRunner, Caps};

/// One cell library at one corner, with the constraints every design on it gets.
#[derive(Debug, Clone, PartialEq)]
pub struct Kit {
    /// The PDK manifest's name (`pdks/<name>.toml`).
    pub name: String,
    /// The corner as the library states it (`typical`, `PVT_0P7V_25C`).
    pub corner: String,
    /// The Liberty `library (NAME)`.
    pub lib_name: String,
    /// The Liberty file (flows that read Liberty).
    pub liberty: PathBuf,
    /// The compiled `.db` (flows that read `.db`).
    pub db: PathBuf,
    /// The library's time unit in ps (1000 for `1ns`, 1 for `1ps`).
    pub time_unit_ps: f64,
    /// Cell driving every input.
    pub driving_cell: String,
    /// Load on every output, in the library's capacitance unit.
    pub load: f64,
    /// Every cell of the library.
    pub cells: BTreeSet<String>,
    /// Cells with a `latch` group.
    pub latches: BTreeSet<String>,
    /// Cells with an `ff` group (flip-flops).
    pub flops: BTreeSet<String>,
}

impl Kit {
    /// A period in ps, in the library's time unit.
    pub fn period(&self, ps: f64) -> f64 {
        ps / self.time_unit_ps
    }
}

/// One design of a job: its top module and its RTL files (relative to the job's `rtl/`; `.v` Verilog,
/// `.vhdl` VHDL), dependencies first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Design {
    pub id: String,
    pub files: Vec<String>,
    /// Has a `clk` port.
    pub clocked: bool,
    /// Vendor libraries its RTL instantiates cells of (`designware`); only a flow that links them can
    /// synthesize it.
    pub libraries: Vec<String>,
}

impl Design {
    /// Its Verilog files.
    pub fn verilog(&self) -> impl Iterator<Item = &String> {
        self.files.iter().filter(|f| !f.ends_with(".vhdl"))
    }

    /// Its VHDL files.
    pub fn vhdl(&self) -> impl Iterator<Item = &String> {
        self.files.iter().filter(|f| f.ends_with(".vhdl"))
    }
}

/// Area, delay and leakage of one design as the synthesis reports state them.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Measured {
    /// Total cell area, µm².
    pub area_um2: Option<f64>,
    /// Sequential (non-combinational) cell area, µm².
    pub seq_area_um2: Option<f64>,
    /// Longest path: arrival at the endpoint, ps.
    pub delay_ps: Option<f64>,
    /// Worst slack against the period, ps.
    pub slack_ps: Option<f64>,
    /// Cell leakage, nW.
    pub leakage_nw: Option<f64>,
    /// Area per direct child instance (hierarchy kept: split pieces, IP blocks), µm².
    pub children: BTreeMap<String, f64>,
}

/// An equivalence check's outcome.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Lec {
    /// Every compare point proven equivalent and none failing, aborted or unmatched.
    pub equivalent: bool,
    pub passing: u64,
    pub failing: u64,
    /// Aborted or inconclusive points (never a pass).
    pub aborted: u64,
    pub unmatched: u64,
    pub verdict: String,
}

impl Lec {
    /// The check ran out of time: nothing failed, some points stayed unverified.
    pub fn inconclusive(&self) -> bool {
        !self.equivalent && self.failing == 0 && self.aborted > 0
    }
}

/// The library format a flow links against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LibraryFormat {
    /// Synopsys `.db`, compiled from the Liberty files by `chipc doctor`.
    Db,
    /// The Liberty text.
    Liberty,
}

/// What a flow supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SynthCapabilities {
    /// Runs licensed tools (through `eda-run`, one license per session) rather than the toolchain image.
    pub licensed: bool,
    pub library: LibraryFormat,
    /// The equivalence checker of its optional check (`Row.lec_tool`).
    pub equivalence: &'static str,
    /// Keeps the hierarchy and reports the area of each child instance (split multipliers' pieces).
    pub hierarchy: bool,
    /// May map arithmetic to vendor IP blocks (DesignWare).
    pub ip_blocks: bool,
    /// Reports a timing path for a design that is only flip-flops.
    pub flop_only_paths: bool,
    /// Reports leakage power.
    pub leakage: bool,
}

/// Options of the equivalence check (absent: no check).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LecOptions {
    /// Wall-clock limit per design and strategy, in minutes.
    pub timeout_min: u32,
    /// Alternate strategies tried in order while a design is not proven (Formality).
    pub strategies: Vec<String>,
}

/// One tool session's work: a batch of designs on one kit at one period.
#[derive(Debug, Clone)]
pub struct SynthJob<'a> {
    pub kit: &'a Kit,
    pub designs: &'a [Design],
    pub period_ps: f64,
    /// The caps of one session: its scripts start with the tool's own caps for them.
    pub caps: Caps,
    /// A directory the tool may keep library analysis in, shared by every job (DC's `alib`).
    pub cache_dir: &'a Path,
    pub lec: Option<&'a LecOptions>,
}

/// The runners a flow may use: `eda` for licensed host tools, `docker` for the toolchain image.
#[derive(Debug, Clone)]
pub struct FlowRunners {
    pub eda: BoundRunner,
    pub docker: BoundRunner,
}

/// The report texts of one design (empty when the flow writes no such report).
#[derive(Debug, Clone, Copy, Default)]
pub struct ReportTexts<'a> {
    pub area: &'a str,
    pub qor: &'a str,
    pub timing: &'a str,
    pub power: &'a str,
    /// The design's part of the tool log.
    pub log: &'a str,
}

/// A synthesis flow. Every script works in its job directory: the RTL is under `rtl/`, each design writes
/// `<id>.net.v`, `<id>.area.rpt` and the flow's other `<id>.<what>.rpt`, and the log brackets each design
/// with `@@begin <id>` / `@@end <id>`.
pub trait SynthFlow: Named {
    fn capabilities(&self) -> SynthCapabilities;

    /// File name → text of the synthesis scripts of `job`. The text is part of the job's key, so it must
    /// be a pure function of the job.
    fn synth_scripts(&self, job: &SynthJob) -> Vec<(String, String)>;

    /// File name → text of the separate equivalence scripts (none without `job.lec`, or when the flow
    /// checks inside its synthesis scripts).
    fn lec_scripts(&self, job: &SynthJob) -> Vec<(String, String)>;

    /// Run the synthesis scripts, already written into `dir` with the RTL; the log goes to
    /// `dir/synth.log`.
    fn synthesize(&self, runners: &FlowRunners, job: &SynthJob, dir: &Path) -> Result<()>;

    /// Run the equivalence scripts, already written into `dir`; the log goes to `dir/lec.log`.
    fn check_equivalence(&self, runners: &FlowRunners, job: &SynthJob, dir: &Path) -> Result<()>;

    /// Area, delay and leakage of design `id` from its reports, in µm², ps and nW.
    fn measure(&self, reports: &ReportTexts, id: &str, period_ps: f64, time_unit_ps: f64) -> Measured;

    /// The outcome of design `id`'s equivalence check from its report (`<id>.lec.rpt`).
    fn equivalence(&self, report: &str) -> Lec;

    /// Did the session map arithmetic to vendor IP blocks (from the design's log)?
    fn used_ip_blocks(&self, log: &str) -> bool;
}
