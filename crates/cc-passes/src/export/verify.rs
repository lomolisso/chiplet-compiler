//! Independent checks of the export (design §4.1, §5). Per model (check names `<model>:<check>`):
//!
//! - `exported`: `export.json` parses and every phase exported.
//! - `schema`: every `graph.json` / `symbols.json` / `census.json` parses into the strict types.
//! - `mlir_verifier`: a fresh process (`python -m cc_export.facts`) re-parses each `module.mlir` and runs
//!   the MLIR verifier.
//! - `roundtrip`: the op-name multiset of `graph.json` (counted here) equals the one of `module.mlir`
//!   (counted by that fresh process).
//! - `census`: the GEMM/conv census recomputed here from `graph.json` (calls and composites expanded)
//!   equals `census.json`.
//! - `recipe_audit`: the census meets the recipe's audit (`audit.rs`, the rules of `verify/dtype_audit.py`).
//! - `decompositions`: every composite decomposition and call target is a function of the module.
//! - `components`: no interception failure; the coverage recomputed here equals `graph.json`'s; every op
//!   is inside a composite or counted `anon`.
//! - `symbols`: the model declares input-dependent sizes; every symbol of every phase's shapes is declared,
//!   appears in some input, and every declared symbol is used.
//! - `invariance`: re-exported with every symbol's config field perturbed, every phase gave the identical
//!   graph and no input shape changed (the exporter's certificate: both hashes and the perturbation).
//! - `size_variants`: phases declared as size variants of another have its graph.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use cc_core::runner::BoundRunner;
use cc_core::{Check, VerifyReport};
use cc_pipeline::{PassContext, PassError};
use rayon::prelude::*;
use serde::Deserialize;

use super::json::{CensusFile, GraphFile, ModelFile, SymbolsFile};
use super::{EXPORT, ExportConfig, MODEL_FILE, analysis, audit, invocation, model_runner};

#[derive(Debug, Deserialize)]
struct Facts {
    verified: bool,
    op_multiset: BTreeMap<String, u64>,
    #[serde(default)]
    error: Option<String>,
}

pub(super) fn verify(ctx: &PassContext) -> Result<VerifyReport, PassError> {
    let cfg: ExportConfig = ctx.options()?;
    let out = ctx.output(EXPORT)?.dir().to_path_buf();
    let runner = model_runner(ctx, &cfg);
    let per_model: Vec<Vec<Check>> = ctx
        .models()
        .par_iter()
        .with_max_len(1)
        .map(|m| verify_model(ctx, &cfg, &runner, &out, m))
        .collect::<Result<_, _>>()?;
    let mut r = VerifyReport::new(ctx.name());
    for c in per_model.into_iter().flatten() {
        r.push(c);
    }
    Ok(r)
}

/// Failing items of one check, with the check's evidence.
struct Acc {
    name: String,
    notes: Vec<String>,
    paths: Vec<String>,
    counts: BTreeMap<String, u64>,
}

impl Acc {
    fn new(model: &str, check: &str) -> Self {
        Self {
            name: format!("{model}:{check}"),
            notes: Vec::new(),
            paths: Vec::new(),
            counts: BTreeMap::new(),
        }
    }
    fn fail(&mut self, note: impl Into<String>) {
        self.notes.push(note.into());
    }
    fn done(self) -> Check {
        let mut c = Check::new(self.name, self.notes.is_empty());
        for n in self.notes.into_iter().take(50) {
            c = c.note(n);
        }
        for p in self.paths {
            c = c.path(p);
        }
        for (k, v) in self.counts {
            c = c.count(k, v);
        }
        c
    }
}

fn read<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, String> {
    cc_core::canon::read_json_file(path).map_err(|e| e.to_string())
}

/// Identifiers of a JAX shape spec (symbols), without its functions and the `_` placeholder.
fn spec_symbols(v: &serde_json::Value, out: &mut BTreeSet<String>) {
    match v {
        serde_json::Value::String(s) => {
            let mut cur = String::new();
            for ch in s.chars().chain(std::iter::once(' ')) {
                if ch.is_ascii_alphanumeric() || ch == '_' {
                    cur.push(ch);
                } else {
                    let starts_alpha = cur
                        .chars()
                        .next()
                        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_');
                    if starts_alpha
                        && !["_", "mod", "floordiv", "max", "min", "non_negative"].contains(&cur.as_str())
                    {
                        out.insert(cur.clone());
                    }
                    cur.clear();
                }
            }
        }
        serde_json::Value::Array(a) => a.iter().for_each(|x| spec_symbols(x, out)),
        serde_json::Value::Object(o) => o.values().for_each(|x| spec_symbols(x, out)),
        _ => {}
    }
}

fn verify_model(
    ctx: &PassContext,
    cfg: &ExportConfig,
    runner: &BoundRunner,
    out: &Path,
    model: &str,
) -> Result<Vec<Check>, PassError> {
    let dir = out.join(model);
    let rel = |p: &str| format!("out/{}/{model}/{p}", EXPORT.name());
    let mut checks = Vec::new();

    let mut exported = Acc::new(model, "exported");
    exported.paths.push(rel(MODEL_FILE));
    let mf: ModelFile = match read(&dir.join(MODEL_FILE)) {
        Ok(m) => m,
        Err(e) => {
            exported.fail(format!("{MODEL_FILE}: {e}"));
            return Ok(vec![exported.done()]);
        }
    };
    for (ph, rec) in &mf.phases {
        if let Some(e) = &rec.error {
            exported.fail(format!("{ph}: {}", e.lines().next().unwrap_or_default()));
        }
    }
    if mf.config != ctx.model_config().as_str() {
        exported.fail(format!(
            "exported at `{}`, the run is at `{}`",
            mf.config,
            ctx.model_config().as_str()
        ));
    }
    exported.counts.insert("phases".into(), mf.phases.len() as u64);

    // MLIR-side facts from a fresh process.
    let mut verifier = Acc::new(model, "mlir_verifier");
    let facts: BTreeMap<String, Facts> = {
        let inv = invocation(
            ctx,
            cfg,
            &["-m", "cc_export.facts"],
            &[facts_dir(cfg, model, &dir)],
        );
        let o = runner.run(&inv)?.check()?;
        verifier.paths.push(o.stdout.display().to_string());
        let text = o.stdout_string()?;
        serde_json::from_str(text.lines().last().unwrap_or("{}")).unwrap_or_else(|e| {
            verifier.fail(format!("cc_export.facts output: {e}"));
            BTreeMap::new()
        })
    };

    let mut schema = Acc::new(model, "schema");
    let mut roundtrip = Acc::new(model, "roundtrip");
    let mut census = Acc::new(model, "census");
    let mut decomp = Acc::new(model, "decompositions");
    let mut comps = Acc::new(model, "components");
    let mut syms = Acc::new(model, "symbols");
    let mut inv = Acc::new(model, "invariance");
    let mut per_phase_census = BTreeMap::new();
    let mut used_symbols = BTreeSet::new();
    if mf.symbols.is_empty() {
        syms.fail("the model declares no input-dependent sizes (ModelSpec.symbols)");
    }
    for (ph, rec) in mf.phases.iter().filter(|(_, r)| r.error.is_none()) {
        let pdir = dir.join(ph);
        let g: GraphFile = match read(&pdir.join("graph.json")) {
            Ok(g) => g,
            Err(e) => {
                schema.fail(format!("{ph}/graph.json: {e}"));
                continue;
            }
        };
        let sy: Result<SymbolsFile, _> = read(&pdir.join("symbols.json"));
        let ce: Result<CensusFile, _> = read(&pdir.join("census.json"));

        match facts.get(ph) {
            Some(f) if f.verified => {
                let ours = analysis::multiset(&g.module);
                if ours != f.op_multiset {
                    let diff: Vec<String> = ours
                        .keys()
                        .chain(f.op_multiset.keys())
                        .collect::<BTreeSet<_>>()
                        .into_iter()
                        .filter(|k| ours.get(*k) != f.op_multiset.get(*k))
                        .take(5)
                        .map(|k| format!("{k}: graph {:?} mlir {:?}", ours.get(k), f.op_multiset.get(k)))
                        .collect();
                    roundtrip.fail(format!("{ph}: {}", diff.join("; ")));
                }
            }
            Some(f) => verifier.fail(format!(
                "{ph}: {}",
                f.error.as_deref().unwrap_or("MLIR verifier failed")
            )),
            None => verifier.fail(format!("{ph}: module.mlir not checked")),
        }

        let x = analysis::expand(&g);
        if !x.missing_callees.is_empty() {
            decomp.fail(format!("{ph}: missing {:?}", x.missing_callees));
        }
        match &ce {
            Ok(c) => {
                if c.summary != x.census {
                    census.fail(format!(
                        "{ph}: census.json {:?} != graph.json {:?}",
                        c.summary, x.census
                    ));
                }
                per_phase_census.insert(ph.clone(), c.summary.clone());
            }
            Err(e) => schema.fail(format!("{ph}/census.json: {e}")),
        }
        let cov = &g.coverage;
        let mine = (
            x.ops,
            x.in_composite,
            x.in_leaf_composite,
            x.anon,
            x.instances,
            x.leaf_instances,
        );
        let theirs = (
            cov.ops,
            cov.in_composite,
            cov.in_leaf_composite,
            cov.anon,
            cov.instances,
            cov.leaf_instances,
        );
        if mine != theirs || g.composites.len() as u64 != x.instances {
            comps.fail(format!(
                "{ph}: coverage recomputed {mine:?}, graph.json {theirs:?}"
            ));
        }
        if !g.interception.failures.is_empty() {
            comps.fail(format!(
                "{ph}: {} interception failures, first: {:?}",
                g.interception.failures.len(),
                g.interception.failures[0]
            ));
        }
        for (k, v) in [
            ("ops", x.ops),
            ("in_composite", x.in_composite),
            ("in_leaf_composite", x.in_leaf_composite),
            ("anon", x.anon),
            ("instances", x.instances),
        ] {
            *comps.counts.entry(k.into()).or_default() += v;
        }

        match &sy {
            Ok(s) => {
                let mut in_shapes = BTreeSet::new();
                spec_symbols(&s.shapes, &mut in_shapes);
                for n in &in_shapes {
                    if !mf.symbols.contains_key(n) {
                        syms.fail(format!("{ph}: symbol `{n}` of the shapes is not declared"));
                    }
                }
                for (n, d) in &s.symbols {
                    if d.inputs.as_ref().is_none_or(Vec::is_empty) {
                        syms.fail(format!("{ph}: symbol `{n}` appears in no input"));
                    }
                }
                used_symbols.extend(in_shapes);
            }
            Err(e) => schema.fail(format!("{ph}/symbols.json: {e}")),
        }

        let ic = rec.checks.get("invariance");
        let ok = ic.and_then(|c| c.get("ok")).and_then(serde_json::Value::as_bool) == Some(true);
        let h = ic.and_then(|c| c.get("hash"));
        let hp = ic.and_then(|c| c.get("perturbed_hash"));
        let changed = ic
            .and_then(|c| c.get("changed_inputs"))
            .and_then(serde_json::Value::as_array)
            .map_or(0, Vec::len);
        if !(ok && h.is_some() && h == hp && changed == 0) {
            inv.fail(format!(
                "{ph}: {}",
                ic.map(|c| c.to_string())
                    .unwrap_or_else(|| "no certificate".into())
                    .chars()
                    .take(600)
                    .collect::<String>()
            ));
        }
    }
    for n in mf.symbols.keys() {
        if !used_symbols.contains(n) {
            syms.fail(format!("symbol `{n}` is declared but no phase uses it"));
        }
    }

    let mut variants = Acc::new(model, "size_variants");
    for (ph, rec) in &mf.phases {
        if let Some(base) = &rec.size_variant_of {
            let other = mf.phases.get(base).and_then(|b| b.hash.as_ref());
            if rec.hash.is_none() || rec.hash.as_ref() != other {
                variants.fail(format!(
                    "{ph} is declared a size variant of {base} but its graph differs"
                ));
            }
        }
    }

    let mut recipe = Acc::new(model, "recipe_audit");
    match &mf.recipe.audit {
        Some(a) => {
            let dtypes: BTreeSet<String> = mf.recipe.param_dtypes.iter().cloned().collect();
            for e in audit::check(a, &per_phase_census, &mf.config, &dtypes) {
                recipe.fail(e);
            }
        }
        None => recipe.fail("no recipe audit for this model (models/<key>/recipe.py)"),
    }

    checks.extend([
        exported.done(),
        schema.done(),
        verifier.done(),
        roundtrip.done(),
        census.done(),
        recipe.done(),
        decomp.done(),
        comps.done(),
        syms.done(),
        inv.done(),
        variants.done(),
    ]);
    Ok(checks)
}

fn facts_dir(cfg: &ExportConfig, model: &str, host_dir: &Path) -> String {
    match cfg.runner {
        super::ExportRunner::Docker => format!("{}/{model}", super::OUT_IN_CONTAINER),
        super::ExportRunner::Host => host_dir.display().to_string(),
    }
}

/// `views/coverage.md`: per model, the share of executed ops inside composites.
pub(super) fn write_coverage_view(ctx: &PassContext, out: &Path) -> Result<(), PassError> {
    let mut s = String::from(
        "# Export: component coverage\n\nShare of executed non-structural ops (calls and composites expanded) \
         inside some composite (`any`) and inside a leaf composite (`leaf`); `anon` ops are outside every \
         composite.\n\n| model | phases | ops | any | leaf | anon | instances |\n|---|---|---|---|---|---|---|\n",
    );
    for m in ctx.models() {
        let Ok(mf) = read::<ModelFile>(&out.join(m).join(MODEL_FILE)) else {
            s.push_str(&format!("| {m} | export failed | | | | | |\n"));
            continue;
        };
        let (mut ops, mut any, mut leaf, mut anon, mut inst) = (0u64, 0u64, 0u64, 0u64, 0u64);
        for c in mf.phases.values().filter_map(|p| p.coverage.as_ref()) {
            ops += c.ops;
            any += c.in_composite;
            leaf += c.in_leaf_composite;
            anon += c.anon;
            inst += c.instances;
        }
        let f = |n: u64| if ops == 0 { 0.0 } else { n as f64 / ops as f64 };
        s.push_str(&format!(
            "| {m} | {} | {ops} | {:.3} | {:.3} | {anon} | {inst} |\n",
            mf.phases.len(),
            f(any),
            f(leaf)
        ));
    }
    ctx.dir().write_view("coverage.md", &s)?;
    Ok(())
}
