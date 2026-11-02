//! `verify` for characterize (docs/passes/characterize.md §6). It reads only `out/library/` (the report
//! and netlist copies), the configuration, the Liberty files and, when present, `build/.datapaths/out`;
//! it never reuses the run's parse results.

use std::collections::{BTreeMap, BTreeSet};

use cc_core::{ArtifactStore, Check, CheckStatus, VerifyReport, canon};
use cc_pipeline::{PassContext, PassError};

use super::catalog::{self, Primitive};
use super::config::{CharacterizeConfig, Tool};
use super::json::{Coverage, Library, Row, Status};
use super::{LIBRARY, flow, sanity};
use crate::datapaths::json::{Datapath, Src};
use crate::eda::{Kit, parse};

pub fn verify(ctx: &PassContext) -> Result<VerifyReport, PassError> {
    let cfg: CharacterizeConfig = ctx.options()?;
    let out = ctx.output(LIBRARY)?;
    let mut rep = VerifyReport::new("characterize");
    let lib: Library = out.read_json("library.json")?;
    let cov: Coverage = out.read_json("coverage.json")?;
    let stored: Vec<Primitive> = out.read_json("catalog.json")?;

    // The catalog regenerates from the configuration, byte for byte.
    let cat = match catalog::catalog(&cfg) {
        Ok(c) => c,
        Err(e) => {
            rep.push(Check::new("catalog", false).note(e));
            return Ok(rep);
        }
    };
    let same = canon::to_canonical_compact(&cat).ok() == canon::to_canonical_compact(&stored).ok();
    rep.push(
        Check::new("catalog", same)
            .count("primitives", cat.len() as u64)
            .path("out/library/catalog.json")
            .note(if same {
                "regenerated catalog equals catalog.json"
            } else {
                "catalog.json differs from the regenerated catalog"
            }),
    );

    // RTL copies equal the sources.
    let mut rtl_bad = Vec::new();
    for p in &cat {
        if let Some(catalog::Source::Dslx(t) | catalog::Source::Verilog(t)) = &p.source {
            let ext = if matches!(p.source, Some(catalog::Source::Dslx(_))) {
                "x"
            } else {
                "v"
            };
            let f = out.dir().join(format!("rtl/{}.{ext}", p.id));
            if std::fs::read_to_string(&f).ok().as_deref() != Some(t.as_str()) {
                rtl_bad.push(p.id.clone());
            }
            if !out.dir().join(format!("rtl/{}.v", p.id)).is_file() {
                rtl_bad.push(format!("{} (no Verilog)", p.id));
            }
        }
    }
    rep.push(note_list(
        Check::new("rtl", rtl_bad.is_empty()).count("mismatched", rtl_bad.len() as u64),
        &rtl_bad,
    ));

    complete(&mut rep, &cfg, &cat, &lib, &cov);

    let kits = match flow::kits(ctx, &cfg) {
        Ok(k) => k,
        Err(e) => {
            rep.push(Check::new("kits", false).note(e));
            return Ok(rep);
        }
    };
    for (kit, info) in &kits {
        let stored = lib.kits.iter().find(|k| k.name == kit.name);
        let ok = stored.is_some_and(|s| s.liberty_hash == info.liberty_hash && s.db_hash == info.db_hash);
        rep.push(Check::new(format!("{}:libraries", kit.name), ok).note(format!(
            "{} {} (.lib and .db hashes as recorded)",
            kit.lib_name, kit.corner
        )));
        for t in &cfg.targets {
            let rows: Vec<&Row> = lib
                .rows
                .iter()
                .filter(|r| r.kit == kit.name && &r.target == t)
                .collect();
            per_target(&mut rep, &cfg, kit, t, &rows, out.dir());
        }
    }

    // Area against width: notes only.
    let mut flags = Vec::new();
    let series = sanity::series(&lib.rows);
    for s in &series {
        flags.extend(sanity::flags(s));
    }
    let mut c = Check::new("sanity", true)
        .with_status(CheckStatus::Pass)
        .count("series", series.len() as u64)
        .count("flags", flags.len() as u64);
    for f in flags.iter().take(200) {
        c = c.note(f.clone());
    }
    rep.push(c);

    datapaths_coverage(ctx, &mut rep, &cat);
    Ok(rep)
}

fn note_list(mut c: Check, items: &[String]) -> Check {
    for i in items.iter().take(40) {
        c = c.note(i.clone());
    }
    c
}

/// Every primitive with RTL has one row per kit and target, nothing else has rows, and coverage states
/// each primitive as the catalog does.
fn complete(
    rep: &mut VerifyReport,
    cfg: &CharacterizeConfig,
    cat: &[Primitive],
    lib: &Library,
    cov: &Coverage,
) {
    let mut want = BTreeSet::new();
    for p in cat.iter().filter(|p| p.source.is_some()) {
        for k in &cfg.pdks {
            for t in &cfg.targets {
                want.insert((p.key.clone(), k.clone(), t.clone()));
            }
        }
    }
    let mut have = BTreeMap::new();
    for r in &lib.rows {
        *have
            .entry((r.key.clone(), r.kit.clone(), r.target.clone()))
            .or_insert(0u32) += 1;
    }
    let mut bad = Vec::new();
    for w in &want {
        match have.get(w) {
            Some(1) => {}
            Some(n) => bad.push(format!("{} {} {}: {n} rows", w.0, w.1, w.2)),
            None => bad.push(format!("{} {} {}: no row", w.0, w.1, w.2)),
        }
    }
    for h in have.keys() {
        if !want.contains(h) {
            bad.push(format!("{} {} {}: row without a primitive", h.0, h.1, h.2));
        }
    }
    let by_key: BTreeMap<&str, &Primitive> = cat.iter().map(|p| (p.key.as_str(), p)).collect();
    if cov.primitives.len() != cat.len() {
        bad.push(format!(
            "coverage lists {} primitives, the catalog {}",
            cov.primitives.len(),
            cat.len()
        ));
    }
    for e in &cov.primitives {
        let Some(p) = by_key.get(e.key.as_str()) else {
            bad.push(format!("coverage: {} is not in the catalog", e.key));
            continue;
        };
        let status = if p.pending.is_some() {
            Status::Pending
        } else if p.provisional {
            Status::Provisional
        } else {
            Status::Measured
        };
        if e.status != status || (status == Status::Pending && e.reason.as_deref().is_none_or(str::is_empty))
        {
            bad.push(format!("coverage: {} is {:?}", e.key, e.status));
        }
    }
    let pending = cat.iter().filter(|p| p.pending.is_some()).count() as u64;
    rep.push(note_list(
        Check::new("complete", bad.is_empty())
            .count("rows", lib.rows.len() as u64)
            .count("expected_rows", want.len() as u64)
            .count("pending", pending)
            .path("out/library/library.json")
            .path("out/library/coverage.json"),
        &bad,
    ));
}

/// The equivalence, log and value checks of one kit at one timing point, from the copies.
fn per_target(
    rep: &mut VerifyReport,
    cfg: &CharacterizeConfig,
    kit: &Kit,
    target: &str,
    rows: &[&Row],
    root: &std::path::Path,
) {
    let read = |rel: &str| std::fs::read_to_string(root.join(rel)).unwrap_or_default();
    let (mut lec_bad, mut log_bad, mut val_bad) = (Vec::new(), Vec::new(), Vec::new());
    let mut proven = 0u64;
    for r in rows {
        let base = format!("reports/{}/{}/{}", kit.name, target, r.id);
        let log = read(&format!("{base}.log"));
        let (seg, lseg) = log.split_once("@@lec\n").unwrap_or((log.as_str(), ""));
        let net = read(&r.netlist);
        let lec_text = read(&format!("{base}.lec.rpt"));
        let lec = match cfg.tool {
            Tool::Dc => parse::formality(&lec_text),
            Tool::Genus => parse::conformal(&lec_text),
            Tool::Yosys => parse::yosys_equiv(&lec_text),
        };
        if lec.equivalent {
            proven += 1;
        } else {
            lec_bad.push(format!(
                "{}: {} ({} failing, {} aborted)",
                r.key, lec.verdict, lec.failing, lec.aborted
            ));
        }
        let measured = match cfg.tool {
            Tool::Dc => parse::dc(
                &read(&format!("{base}.area.rpt")),
                &read(&format!("{base}.qor.rpt")),
                &read(&format!("{base}.power.rpt")),
                kit.time_unit_ps,
            ),
            Tool::Genus => parse::genus(
                &read(&format!("{base}.area.rpt")),
                &read(&format!("{base}.timing.rpt")),
                &read(&format!("{base}.power.rpt")),
                &r.id,
            ),
            Tool::Yosys => parse::yosys(&read(&format!("{base}.area.rpt")), seg, &r.id, r.period_ps),
        };
        let problems = flow::problems_of(
            kit,
            seg,
            lseg,
            &read(&format!("{base}.check.rpt")),
            &net,
            &measured,
            &lec,
            !(cfg.tool == Tool::Yosys
                && matches!(r.kind, catalog::Kind::Register | catalog::Kind::Accumulator)),
        );
        for p in problems.into_iter().filter(|p| !p.starts_with("equivalence")) {
            log_bad.push(format!("{}: {p}", r.key));
        }
        if measured != r.measured || lec != r.lec {
            val_bad.push(format!("{}: library.json differs from its reports", r.key));
        }
        if canon::hash_bytes(net.as_bytes()) != r.netlist_hash {
            val_bad.push(format!("{}: netlist hash", r.key));
        }
    }
    let n = rows.len() as u64;
    let pre = format!("{}:{target}", kit.name);
    rep.push(note_list(
        Check::new(format!("{pre}:lec"), lec_bad.is_empty() && n > 0)
            .count("rows", n)
            .count("equivalent", proven)
            .note(format!("{} on every netlist against its RTL", cfg.tool.lec())),
        &lec_bad,
    ));
    rep.push(note_list(
        Check::new(format!("{pre}:log"), log_bad.is_empty())
            .count("rows", n)
            .count("problems", log_bad.len() as u64)
            .note("no tool errors, unresolved references, latches or cells outside the library"),
        &log_bad,
    ));
    rep.push(note_list(
        Check::new(format!("{pre}:rows"), val_bad.is_empty())
            .count("rows", n)
            .note("every value re-read from the report copies; netlist hashes"),
        &val_bad,
    ));
}

/// Every vertex of every datapath (template stages included) and every single-op accumulator maps to a
/// primitive of the catalog, measured or pending.
fn datapaths_coverage(ctx: &PassContext, rep: &mut VerifyReport, cat: &[Primitive]) {
    let dp_dir = ArtifactStore::new(ctx.resolve(&ctx.paths().build_dir)).pass_dir("datapaths");
    let files = dp_dir.out_dir().join("datapaths").join("datapaths");
    if !files.is_dir() {
        rep.push(Check::skip(
            "coverage.datapaths",
            "build/.datapaths/out is not there (run the analysis chain to check)",
        ));
        return;
    }
    let manifest_hash = canon::hash_file(&dp_dir.manifest_path()).unwrap_or_default();
    let keys: BTreeMap<&str, &Primitive> = cat.iter().map(|p| (p.key.as_str(), p)).collect();
    let mut used: BTreeMap<String, u64> = BTreeMap::new();
    let mut gaps: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut classes = 0u64;
    let mut entries: Vec<_> = std::fs::read_dir(&files)
        .map(|d| d.filter_map(|e| e.ok()).collect())
        .unwrap_or_else(|_| Vec::new());
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let Ok(text) = std::fs::read_to_string(e.path()) else {
            continue;
        };
        let Ok(dp) = serde_json::from_str::<Datapath>(&text) else {
            gaps.entry("unreadable datapath".into())
                .or_default()
                .insert(e.file_name().to_string_lossy().into_owned());
            continue;
        };
        classes += 1;
        for k in datapath_keys(&dp) {
            match k {
                Ok(key) => {
                    if keys.contains_key(key.as_str()) {
                        *used.entry(key).or_default() += 1;
                    } else {
                        gaps.entry(key).or_default().insert(dp.id.clone());
                    }
                }
                Err(why) => {
                    gaps.entry(why).or_default().insert(dp.id.clone());
                }
            }
        }
    }
    let pending: Vec<&String> = used
        .keys()
        .filter(|k| keys[k.as_str()].pending.is_some())
        .collect();
    let provisional = used.keys().filter(|k| keys[k.as_str()].provisional).count() as u64;
    let mut c = Check::new("coverage.datapaths", gaps.is_empty())
        .count("classes", classes)
        .count("keys_used", used.len() as u64)
        .count("keys_provisional", provisional)
        .count("keys_pending", pending.len() as u64)
        .count("gaps", gaps.len() as u64)
        .path(dp_dir.manifest_path().display().to_string())
        .note(format!("build/.datapaths/manifest.json blake3 {manifest_hash}"));
    if !pending.is_empty() {
        c = c.note(format!(
            "pending (softfloat): {}",
            pending.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
        ));
    }
    for (k, zs) in gaps.iter().take(40) {
        c = c.note(format!(
            "gap {k}: {} classes, e.g. {}",
            zs.len(),
            zs.iter().next().map_or("", |s| s.as_str())
        ));
    }
    rep.push(c);
}

/// The library keys one datapath needs, per tuple (an error names what cannot be mapped).
pub fn datapath_keys(dp: &Datapath) -> Vec<Result<String, String>> {
    let mut out = Vec::new();
    let commutative: BTreeSet<&str> = cc_ir::ops::TABLE
        .iter()
        .filter(|s| s.elementwise && s.commutative)
        .map(|s| s.name.trim_start_matches("stablehlo."))
        .collect();
    for t in &dp.tuples {
        // An element class `w<e>`, or a folded constant's value column (its type is in the tuple).
        let ty = |w: &str| -> Option<String> {
            if let Some(c) = t.consts.get(w) {
                return Some(super::type_name(c));
            }
            let e: usize = w.strip_prefix('w')?.parse().ok()?;
            t.elems.get(e).map(|s| super::type_name(s))
        };
        let src_ty = |s: &Src| -> Option<String> {
            match s {
                Src::Vertex { v, .. } => dp.vertices.iter().find(|x| x.id == *v).and_then(|x| ty(&x.width)),
                Src::Input { name } => dp
                    .inputs
                    .iter()
                    .find(|i| &i.name == name)
                    .and_then(|i| ty(&i.width)),
                Src::Acc { index } => dp
                    .accumulator
                    .as_ref()?
                    .entries
                    .get(*index as usize)
                    .and_then(|a| ty(&a.width)),
                // A template's core result is an input; a datapath class's is its accumulator's final value.
                Src::Result { index } => dp
                    .inputs
                    .iter()
                    .find(|i| i.name == format!("core_result{index}"))
                    .map(|i| i.width.clone())
                    .or_else(|| {
                        dp.accumulator
                            .as_ref()?
                            .entries
                            .get(*index as usize)
                            .map(|a| a.width.clone())
                    })
                    .and_then(|w| ty(&w)),
            }
        };
        for v in &dp.vertices {
            if v.op == "const" || matches!(v.op.as_str(), "complex" | "real" | "imag") {
                continue;
            }
            let Some(rt) = ty(&v.width) else {
                out.push(Err(format!("{}: no type for {}", v.label, v.width)));
                continue;
            };
            if let Some(inner) = rt.strip_prefix("complex<").and_then(|r| r.strip_suffix('>')) {
                let ops: &[&str] = match v.op.as_str() {
                    "multiply" => &["multiply", "add", "subtract"],
                    "convert" => &[],
                    o => &[o][..],
                };
                for o in ops {
                    out.push(Ok(format!("{o}@{inner}")));
                }
                if v.op == "convert"
                    && let Some(s) = v.args.first().and_then(|a| src_ty(&a.src))
                {
                    let s = s
                        .strip_prefix("complex<")
                        .and_then(|r| r.strip_suffix('>'))
                        .unwrap_or(&s)
                        .to_string();
                    if s != inner {
                        out.push(Ok(format!("convert@{s}>{inner}")));
                    }
                }
                continue;
            }
            match v.op.as_str() {
                "convert" => match v.args.first().and_then(|a| src_ty(&a.src)) {
                    Some(s) if s == rt => {}
                    Some(s) => out.push(Ok(format!("convert@{s}>{rt}"))),
                    None => out.push(Err(format!("{}: convert source type", v.label))),
                },
                "compare" => {
                    let Some(ot) = v.args.first().and_then(|a| src_ty(&a.src)) else {
                        out.push(Err("compare operand type".into()));
                        continue;
                    };
                    let dir = v.attrs.get("comparison_direction").cloned().unwrap_or_default();
                    let ot = match v.attrs.get("compare_type").map(String::as_str) {
                        Some("UNSIGNED") if ot.starts_with('i') && ot != "i1" => format!("u{ot}"),
                        Some("SIGNED") if ot.starts_with("ui") => ot[1..].to_string(),
                        Some("TOTALORDER") => format!("{ot} (TOTALORDER)"),
                        _ => ot,
                    };
                    out.push(Ok(format!("compare.{dir}@{ot}")));
                }
                "select" if rt == "i1" => out.push(Ok("mux2@w1".into())),
                // A predicate on its operand's type.
                "is_finite" => match v.args.first().and_then(|a| src_ty(&a.src)) {
                    Some(ot) => out.push(Ok(format!("is_finite@{ot}"))),
                    None => out.push(Err("is_finite operand type".into())),
                },
                op => out.push(Ok(format!("{op}@{rt}"))),
            }
        }
        if let Some(a) = &dp.accumulator
            && let [one] = a.combiner.as_slice()
            && let Some(v) = dp.vertices.iter().find(|x| x.id == *one)
            && commutative.contains(v.op.as_str())
            && let Some(rt) = ty(&v.width)
            && !rt.starts_with("complex<")
        {
            out.push(Ok(format!("accumulator.{}@{rt}", v.op)));
        }
    }
    out
}
