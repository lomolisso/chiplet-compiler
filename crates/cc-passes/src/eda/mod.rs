//! The synthesis and equivalence flow shared by the passes that run the licensed tools (and their
//! open-source fallback): kits, script generators and report parsers (docs/passes/characterize.md §4,
//! docs/tools/synopsys.md, cadence.md). Everything here is pure: the passes run the scripts.

pub mod netlist;
pub mod parse;
pub mod scripts;

use std::collections::BTreeSet;
use std::path::PathBuf;

use cc_pipeline::env::liberty;

/// One cell library at one corner, with the constraints every design on it gets.
#[derive(Debug, Clone, PartialEq)]
pub struct Kit {
    /// `nangate45` or `asap7`.
    pub name: String,
    /// The corner as the library states it (`typical`, `PVT_0P7V_25C`).
    pub corner: String,
    /// The Liberty `library (NAME)`.
    pub lib_name: String,
    /// The Liberty file (Genus, Conformal, Yosys).
    pub liberty: PathBuf,
    /// The compiled `.db` (Design Compiler, Formality).
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
    /// A kit from its Liberty text: units, corner and cell sets from the file, the load as
    /// `load_fanout` times the first input pin capacitance of `driving_cell`.
    pub fn from_liberty(
        name: &str,
        text: &str,
        liberty: PathBuf,
        db: PathBuf,
        driving_cell: &str,
        load_fanout: f64,
    ) -> Result<Kit, String> {
        let attrs = liberty::header(text)?;
        let lib_name = cc_pipeline::env::pdk_db::library_name(text).ok_or("no library name")?;
        let time_unit_ps = unit_ps(attrs.get("time_unit").ok_or("no time_unit")?)?;
        let corner = attrs
            .get("default_operating_conditions")
            .cloned()
            .unwrap_or_else(|| "default".into());
        let cells = liberty::cell_names(text)?;
        if !cells.contains(driving_cell) {
            return Err(format!(
                "{name}: driving cell {driving_cell} is not in {lib_name}"
            ));
        }
        let cap = liberty::input_capacitance(text, driving_cell)?
            .ok_or_else(|| format!("{name}: {driving_cell} has no input pin capacitance"))?;
        let (latches, flops) = liberty::sequential_cells(text)?;
        Ok(Kit {
            name: name.into(),
            corner,
            lib_name,
            liberty,
            db,
            time_unit_ps,
            driving_cell: driving_cell.into(),
            load: load_fanout * cap,
            cells,
            latches,
            flops,
        })
    }

    /// A period in ps, in the library's time unit.
    pub fn period(&self, ps: f64) -> f64 {
        ps / self.time_unit_ps
    }
}

/// `"1ns"` → 1000, `"1ps"` → 1, `"10ps"` → 10.
pub fn unit_ps(s: &str) -> Result<f64, String> {
    let s = s.trim().trim_matches('"');
    let split = s
        .find(|c: char| c.is_ascii_alphabetic())
        .ok_or_else(|| format!("time unit {s:?}"))?;
    let (n, u) = s.split_at(split);
    let n: f64 = if n.trim().is_empty() {
        1.0
    } else {
        n.trim().parse().map_err(|_| format!("time unit {s:?}"))?
    };
    let scale = match u.trim() {
        "ps" => 1.0,
        "ns" => 1e3,
        "us" => 1e6,
        _ => return Err(format!("time unit {s:?}")),
    };
    Ok(n * scale)
}

/// One design of a job: its top module and its Verilog files (relative to the job's `rtl/`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Design {
    pub id: String,
    pub files: Vec<String>,
    /// Has a `clk` port.
    pub clocked: bool,
}

/// The per-design files every flow writes in its job directory.
pub fn outputs(id: &str) -> [String; 2] {
    [format!("{id}.net.v"), format!("{id}.area.rpt")]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn units() {
        assert_eq!(unit_ps("\"1ns\"").unwrap(), 1000.0);
        assert_eq!(unit_ps("1ps").unwrap(), 1.0);
        assert_eq!(unit_ps("10ps").unwrap(), 10.0);
        assert!(unit_ps("1fs").is_err());
    }
}
