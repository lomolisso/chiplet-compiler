//! `import` (design §4.2): each exported phase (`graph.json`) → a `cc_ir::Module` with inlined calls,
//! symbolic dimensions on every value, size computation tagged and every op classified by the op table
//! (docs/passes/import.md).

pub mod algo;
mod config;
pub mod json;
mod verify;
mod view;

use std::collections::{BTreeMap, BTreeSet};

use cc_core::{ArtifactKind, CoreError, PassSection, VerifyReport};
use cc_ir::Module;
use cc_pipeline::{Pass, PassContext, PassError};
use rayon::prelude::*;

pub use config::ImportConfig;

use crate::export::json::{GraphFile, ModelFile, SymbolsFile};
use crate::export::{EXPORT, MODEL_FILE};
use json::{INVENTORY_SCHEMA, Inventory, InventoryRow, Ledger, SUMMARY_SCHEMA, Summary};

/// `out/import/<model>/<phase>/ir.postcard` (+ `.summary.json`) and `out/import/op_inventory.json`.
pub const IMPORT: ArtifactKind = ArtifactKind("import");
pub const IR_FILE: &str = "ir.postcard";
pub const INVENTORY_FILE: &str = "op_inventory.json";

#[derive(Debug)]
pub struct Import;

impl Pass for Import {
    fn name(&self) -> &'static str {
        "import"
    }

    fn version(&self) -> u32 {
        2
    }

    fn requires(&self) -> &[ArtifactKind] {
        &[EXPORT]
    }

    fn provides(&self) -> &[ArtifactKind] {
        &[IMPORT]
    }

    fn check_config(&self, section: &PassSection) -> Result<(), CoreError> {
        section.parse::<ImportConfig>().map(drop)
    }

    fn run(&self, ctx: &mut PassContext) -> Result<(), PassError> {
        let cfg: ImportConfig = ctx.options()?;
        let ctx: &PassContext = ctx;
        let units = phases(ctx)?;
        let results: Vec<Unit> = units
            .par_iter()
            .with_max_len(1)
            .map(|(m, p)| import_phase(ctx, &cfg, m, p))
            .collect::<Result<_, _>>()?;
        write_inventory(ctx, &results)?;
        let failed: Vec<String> = results.iter().filter_map(|u| u.error.clone()).collect();
        if failed.is_empty() {
            Ok(())
        } else {
            Err(PassError::failed(format!(
                "{} phase(s) failed to import:\n{}",
                failed.len(),
                failed.join("\n")
            )))
        }
    }

    fn verify(&self, ctx: &PassContext) -> Result<VerifyReport, PassError> {
        verify::verify(ctx)
    }
}

/// Every exported `(model, phase)`, sorted.
pub(crate) fn phases(ctx: &PassContext) -> Result<Vec<(String, String)>, PassError> {
    let input = ctx.input(EXPORT)?;
    let mut out = Vec::new();
    for m in ctx.models() {
        let mf: ModelFile = input.read_json(&format!("{m}/{MODEL_FILE}"))?;
        for (p, rec) in &mf.phases {
            if rec.error.is_none() {
                out.push((m.clone(), p.clone()));
            }
        }
    }
    out.sort();
    Ok(out)
}

/// What one phase contributes to the inventory.
struct Unit {
    model: String,
    /// Op keys of `graph.json` (every function, nested regions included) with counts.
    ops: BTreeMap<String, u64>,
    error: Option<String>,
}

/// The table key of a `graph.json` op.
pub(crate) fn op_key(op: &crate::export::json::Op) -> String {
    match op
        .attributes
        .get("call_target_name")
        .and_then(crate::export::json::Attr::as_str)
    {
        Some(t) if op.name == "stablehlo.custom_call" => format!("{}@{t}", op.name),
        _ => op.name.clone(),
    }
}

fn graph_ops(g: &GraphFile) -> BTreeMap<String, u64> {
    fn region(r: &crate::export::json::Region, out: &mut BTreeMap<String, u64>) {
        for b in &r.blocks {
            for op in &b.ops {
                *out.entry(op_key(op)).or_default() += 1;
                for rr in &op.regions {
                    region(rr, out);
                }
            }
        }
    }
    let mut out = BTreeMap::new();
    for f in &g.module.funcs {
        region(&f.body, &mut out);
    }
    out
}

fn import_phase(ctx: &PassContext, cfg: &ImportConfig, model: &str, phase: &str) -> Result<Unit, PassError> {
    ctx.check_cancelled()?;
    let input = ctx.input(EXPORT)?;
    let g: GraphFile = input.read_json(&format!("{model}/{phase}/graph.json"))?;
    let syms: SymbolsFile = input.read_json(&format!("{model}/{phase}/symbols.json"))?;
    let ops = graph_ops(&g);
    let built = match algo::build::build(&g, &syms, cfg) {
        Ok(b) => b,
        Err(e) => {
            return Ok(Unit {
                model: model.into(),
                ops,
                error: Some(format!("{model}/{phase}: {e}")),
            });
        }
    };
    drop(g);
    let mut module = built.module;
    algo::size_tag::tag(&mut module);
    let summary = summarize(&module, built.ledger);
    let out = ctx.output(IMPORT)?;
    out.write_postcard(&format!("{model}/{phase}/{IR_FILE}"), &module, &summary)?;
    if cfg.views {
        ctx.dir()
            .write_view(&format!("{model}/{phase}.dot"), &view::dot(&module))?;
    }
    Ok(Unit {
        model: model.into(),
        ops,
        error: None,
    })
}

const STRUCTURAL: [&str; 3] = ["func.return", "stablehlo.return", "stablehlo.composite"];

pub(crate) fn summarize(m: &Module, ledger: Ledger) -> Summary {
    let mut by_class = BTreeMap::new();
    let mut by_op = BTreeMap::new();
    let mut dim_as_data = BTreeMap::new();
    let mut composites = BTreeMap::new();
    let (mut size_ops, mut compute) = (0u64, 0u64);
    for op in &m.ops {
        let key = match &op.target {
            Some(t) => format!("{}@{t}", op.name),
            None => op.name.clone(),
        };
        *by_op.entry(key).or_insert(0u64) += 1;
        let class = cc_ir::lookup(&op.name, op.target.as_deref())
            .map(|s| format!("{:?}", s.class).to_lowercase())
            .unwrap_or_else(|| "unknown".into());
        *by_class.entry(class).or_insert(0u64) += 1;
        if !op.dim_as_data.is_empty() {
            *dim_as_data.entry(op.name.clone()).or_insert(0u64) += op.dim_as_data.len() as u64;
        }
        if op.name == "stablehlo.composite"
            && let Some(cc_ir::Attr::Str(n)) = op.attrs.get("name")
        {
            *composites.entry(n.clone()).or_insert(0u64) += 1;
        }
        if !STRUCTURAL.contains(&op.name.as_str()) {
            compute += 1;
            size_ops += u64::from(op.size);
        }
    }
    let symbolic_values = m
        .values
        .iter()
        .filter(|v| v.ty.tensor().is_some_and(|t| !t.is_static()))
        .count() as u64;
    Summary {
        schema: SUMMARY_SCHEMA.into(),
        model: m.model.clone(),
        phase: m.phase.clone(),
        config: m.config.clone(),
        ir_hash: cc_ir::hash::module_hash(m),
        symbols: m.symbols.clone(),
        constraints: m.constraints.iter().map(|c| c.text.clone()).collect(),
        funcs: m.funcs.len() as u64,
        ops: m.ops.len() as u64,
        values: m.values.len() as u64,
        by_class,
        by_op,
        size_ops,
        size_share: if compute == 0 {
            0.0
        } else {
            (size_ops as f64 / compute as f64 * 1e4).round() / 1e4
        },
        dim_as_data,
        composites,
        symbolic_values,
        ledger,
    }
}

fn write_inventory(ctx: &PassContext, units: &[Unit]) -> Result<(), PassError> {
    let mut rows: BTreeMap<String, (u64, BTreeSet<String>)> = BTreeMap::new();
    for u in units {
        for (k, n) in &u.ops {
            let e = rows.entry(k.clone()).or_default();
            e.0 += n;
            e.1.insert(u.model.clone());
        }
    }
    let mut unknown = Vec::new();
    let ops = rows
        .into_iter()
        .map(|(k, (count, models))| {
            let (name, target) = match k.split_once('@') {
                Some((n, t)) => (n, Some(t)),
                None => (k.as_str(), None),
            };
            let entry = cc_ir::lookup(name, target).map(json::entry_json);
            if entry.is_none() {
                unknown.push(k.clone());
            }
            (
                k,
                InventoryRow {
                    entry,
                    count,
                    models: models.into_iter().collect(),
                },
            )
        })
        .collect();
    ctx.output(IMPORT)?.write_json(
        INVENTORY_FILE,
        &Inventory {
            schema: INVENTORY_SCHEMA.into(),
            ops,
            unknown,
        },
    )?;
    Ok(())
}
