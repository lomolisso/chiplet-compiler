//! The synthesis flows (`SynthFlow`, design §3.7): Design Compiler with Formality, and Yosys + ABC with its
//! equivalence flow (docs/passes/characterize.md §4, docs/tools/synopsys.md). Kits, script generators and
//! report parsers are pure functions; `flows` runs the scripts through the tool runners.

pub mod flows;
pub mod netlist;
pub mod parse;
pub mod scripts;

use std::path::PathBuf;

pub use cc_core::adapters::{Design, Kit};
use cc_pipeline::env::liberty;
pub use flows::{Dc, FLOWS, Yosys};

/// A kit from its Liberty text: units, corner and cell sets from the file, the load as `load_fanout` times
/// the first input pin capacitance of `driving_cell`.
pub fn kit(
    name: &str,
    text: &str,
    liberty: PathBuf,
    db: PathBuf,
    driving_cell: &str,
    load_fanout: f64,
) -> Result<Kit, String> {
    let attrs = liberty::header(text)?;
    let lib_name = cc_pipeline::env::pdk_db::library_name(text).ok_or("no library name")?;
    let time_unit_ps = liberty::time_unit_ps(attrs.get("time_unit").ok_or("no time_unit")?)?;
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
