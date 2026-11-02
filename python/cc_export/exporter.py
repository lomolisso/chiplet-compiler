"""Export every phase of one model: StableHLO with composites and symbolic input sizes, plus its JSON.

Nothing executes: `jax.export` traces and lowers on `ShapeDtypeStruct`s only (no `init`, no `apply` on
data, no calls). Per phase, `<out>/<phase>/` receives `module.mlir`, `exported.bin`, `graph.json`,
`symbols.json`, `census.json` and `census_concrete.json`; `<out>/export.json` holds the model-level summary
and self-checks.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import jax
from jax import export

from . import analysis, composite, flax_adapter, graph, haiku_adapter, keras_adapter, symbols

SCHEMA_EXPORT = "cc.export.model/1"
SCHEMA_SYMBOLS = "cc.export.symbols/1"
SCHEMA_CENSUS = "cc.export.census/1"
SCHEMA_CENSUS_CONCRETE = "cc.export.census_concrete/1"
PLATFORMS = ("cpu",)


@contextlib.contextmanager
def interception(**session_kw):
    with composite.session(**session_kw) as s:
        with flax_adapter.intercept(), haiku_adapter.intercept(), keras_adapter.intercept():
            yield s


def _phase_args(spec, ph) -> tuple:
    params = spec.param_specs
    if ph.param_keys is not None:
        params = {k: params[k] for k in ph.param_keys}
    return (params, *ph.inputs)


def _fresh(fn):
    # A new function object per trace: jit's trace cache is keyed by the function, and interception
    # changes what a trace produces.
    def f(*a):
        return fn(*a)

    return f


def trace_phase(spec, ph, exclude, sym_fields=None) -> dict[str, Any]:
    """Export one phase; return the Exported plus everything derived from it (no files written)."""
    args = _phase_args(spec, ph)
    shapes = None if ph.shapes is None else (None, *ph.shapes)
    sym_args, cons = symbols.symbolic_args(args, shapes, spec.constraints)

    # Ports trace internally too (alphafold2 / graphcast `dce` run `jax.make_jaxpr` on a fixed function):
    # JAX's trace caches would hand a later trace the jaxpr of an earlier one (traced with different
    # interception), so every trace starts from empty caches.
    jax.clear_caches()
    with interception(discover=True) as disc:
        jax.eval_shape(_fresh(ph.fn), *sym_args)
    roots = tuple(disc.top_calls) if len(disc.top_calls) == 1 else ()

    jax.clear_caches()
    with interception(exclude=exclude, roots=roots, symbolic_fields=sym_fields) as sess:
        exp = export.export(jax.jit(_fresh(ph.fn)), platforms=PLATFORMS)(*sym_args)
    jax.clear_caches()
    text = exp.mlir_module()
    module, facts = graph.walk(text, loc_strip=[_repo_root() + os.sep])

    flat = jax.tree_util.tree_flatten_with_path(sym_args)[0]
    paths = [jax.tree_util.keystr(p) for p, _ in flat]
    concrete = {jax.tree_util.keystr(p): tuple(x.shape) for p, x in jax.tree_util.tree_flatten_with_path(args)[0]}
    kinds = ["param" if p[0] == jax.tree_util.SequenceKey(0) else "input" for p, _ in flat]
    kept = list(exp.module_kept_var_idx)
    inputs = [
        {
            "flat_index": i,
            "path": paths[i],
            "kind": kinds[i],
            "main_arg": kept.index(i) if i in kept else None,
            "shape": [d if isinstance(d, int) else str(d) for d in exp.in_avals[i].shape],
            "dtype": str(exp.in_avals[i].dtype),
        }
        for i in range(len(flat))
    ]
    instances, coverage = analysis.composites_and_coverage(module)
    return {
        "exp": exp,
        "text": text,
        "module": module,
        "facts": facts,
        "hash": analysis.canonical_hash(module),
        "inputs": inputs,
        "paths": paths,
        "concrete": concrete,
        "kinds": kinds,
        "constraints": cons,
        "roots": list(roots),
        "interception": sess.summary(),
        "instances": instances,
        "coverage": coverage,
    }


def _repo_root() -> str:
    return str(Path(__file__).resolve().parents[2])


def _dump(path: Path, obj: Any, compact: bool = False) -> None:
    if compact:  # graph/census JSON can be tens of MB at full size
        path.write_text(json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n")
    else:
        path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")


def _census_crosscheck(text: str, summary: dict) -> dict:
    from verify import hlo

    theirs = hlo.summarize(hlo.census(text))
    return {"ok": theirs == summary, "verify_hlo": theirs}


def census_concrete(spec, ph) -> dict:
    """`verify/hlo.py`'s census of the phase lowered on its concrete input shapes (the scenario sizes):
    integer FLOPs (2 * MACs) of every executed dot_general / convolution, for one execution of the phase.
    Lowering only, without component composites; nothing runs."""
    from verify import hlo

    jax.clear_caches()
    text = jax.jit(_fresh(ph.fn)).lower(*_phase_args(spec, ph)).as_text()
    jax.clear_caches()
    ops = hlo.census(text)
    summary = hlo.summarize(ops)
    unknown = sum(1 for o in ops if o.flops is None)
    return {
        "summary": summary,
        "flops": None if unknown else sum(o.flops for o in ops),
        "ops": len(ops),
        "unknown": unknown,
    }


def _input_diff(a: list[dict], b: list[dict]) -> list[dict]:
    out = []
    for x, y in zip(a, b):
        if x["shape"] != y["shape"] or x["dtype"] != y["dtype"]:
            out.append({"path": x["path"], "base": x["shape"], "perturbed": y["shape"]})
    if len(a) != len(b):
        out.append({"path": "<pytree>", "base": len(a), "perturbed": len(b)})
    return out


def export_model(
    key: str,
    cfg_name: str,
    out: Path,
    phases: list[str] | None = None,
    exclude: tuple[str, ...] = (),
    invariance: bool = True,
    port: Any = None,
) -> dict:
    """Export `models.<key>.model` (or `port`, any object with `CONFIGS` and `build`)."""
    mod = port if port is not None else importlib.import_module(f"models.{key}.model")
    cfg = mod.CONFIGS[cfg_name]
    spec = mod.build(cfg)
    alt_spec = None
    if invariance and spec.symbols:
        alt_spec = mod.build(symbols.perturbed(cfg, spec.symbols))
    out.mkdir(parents=True, exist_ok=True)
    sym_fields = symbols.symbolic_fields(cfg, spec.symbols)

    declared = set(spec.symbols)
    used_anywhere: set[str] = set()
    summary: dict[str, Any] = {
        "schema": SCHEMA_EXPORT,
        "model": key,
        "config": cfg_name,
        "versions": versions(),
        "symbols": {
            n: {"field": s.field, "value": symbols.field_value(cfg, s.field), "alt": s.alt, "doc": s.doc}
            for n, s in sorted(spec.symbols.items())
        },
        "constraints": list(spec.constraints),
        "exclude": sorted(exclude),
        "phases": {},
    }
    for ph in spec.phases:
        if phases is not None and ph.name not in phases:
            continue
        t0 = time.monotonic()
        rec: dict[str, Any] = {"trip_count": ph.trip_count, "size_variant_of": ph.size_variant_of}
        summary["phases"][ph.name] = rec
        used = symbols.phase_symbols(ph.shapes)
        used_anywhere |= used
        try:
            r = trace_phase(spec, ph, exclude, sym_fields)
        except Exception as e:  # noqa: BLE001 - recorded; the phase fails verify
            rec["error"] = f"{type(e).__name__}: {e}"[:4000]
            rec["traceback"] = traceback.format_exc()[-8000:]
            continue
        pdir = out / ph.name
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "module.mlir").write_text(r["text"])
        (pdir / "exported.bin").write_bytes(r["exp"].serialize())

        cen = analysis.census(r["module"])
        where, concrete = symbols.input_axes(r["paths"], list(r["exp"].in_avals), r["kinds"])
        sizes, size_errors = symbols.solve_sizes(where, r["concrete"])
        _dump(
            pdir / "graph.json",
            {
                "schema": graph.SCHEMA,
                "model": key,
                "phase": ph.name,
                "config": cfg_name,
                "entry": "main",
                "module": r["module"],
                "inputs": r["inputs"],
                "composites": r["instances"],
                "coverage": r["coverage"],
                "interception": r["interception"],
            },
            compact=True,
        )
        _dump(
            pdir / "symbols.json",
            {
                "schema": SCHEMA_SYMBOLS,
                "model": key,
                "phase": ph.name,
                "shapes": ph.shapes,
                "symbols": {
                    n: summary["symbols"][n] | {"inputs": where.get(n, []), "size": sizes.get(n)}
                    for n in sorted(used & declared)
                },
                "constraints": list(r["constraints"]),
                "concrete_input_axes": concrete,
                "trip_count": ph.trip_count,
                "trip_count_expr": ph.trip_count_expr,
            },
        )
        _dump(pdir / "census.json", {"schema": SCHEMA_CENSUS, "model": key, "phase": ph.name, **cen}, compact=True)
        try:
            cc = census_concrete(spec, ph)
        except Exception as e:  # noqa: BLE001 - recorded; the phase fails verify
            cc = {"summary": {}, "flops": None, "ops": 0, "unknown": 0, "error": f"{type(e).__name__}: {e}"[:4000]}
        _dump(
            pdir / "census_concrete.json",
            {"schema": SCHEMA_CENSUS_CONCRETE, "model": key, "phase": ph.name, "config": cfg_name, **cc},
        )

        checks: dict[str, Any] = {}
        reparsed = graph.walk(r["text"])[1]
        checks["mlir_verifier"] = {"ok": reparsed["verified"]}
        gm = graph.graph_multiset(r["module"])
        checks["roundtrip"] = {"ok": gm == r["facts"]["op_multiset"]}
        checks["census_crosscheck"] = _census_crosscheck(r["text"], cen["summary"])
        funcs = {f["name"] for f in r["module"]["funcs"]}
        missing = sorted({i["decomposition"] for i in r["instances"]} - funcs)
        checks["decompositions"] = {"ok": not missing, "missing": missing}
        checks["interception"] = {"ok": not r["interception"]["failures"], "failures": r["interception"]["failures"]}
        undeclared = sorted(used - declared)
        checks["symbols_declared"] = {"ok": not undeclared, "undeclared": undeclared}
        absent = sorted(n for n in used if n not in where)
        checks["symbols_in_inputs"] = {"ok": not absent, "absent": absent}
        checks["symbol_sizes"] = {"ok": not size_errors, "errors": size_errors}
        checks["census_concrete"] = {
            "ok": cc["flops"] is not None and "error" not in cc,
            **({"error": cc["error"]} if "error" in cc else {}),
        }
        if alt_spec is not None:
            alt_ph = next((p for p in alt_spec.phases if p.name == ph.name), None)
            if alt_ph is None:
                checks["invariance"] = {"ok": False, "error": "phase missing in the perturbed build"}
            else:
                try:
                    ra = trace_phase(alt_spec, alt_ph, exclude, sym_fields)
                    diff = _input_diff(r["inputs"], ra["inputs"])
                    checks["invariance"] = {
                        "ok": ra["hash"] == r["hash"] and not diff,
                        "hash": r["hash"],
                        "perturbed_hash": ra["hash"],
                        "changed_inputs": diff,
                    }
                    if ra["hash"] != r["hash"]:
                        d = analysis.first_difference(r["module"], ra["module"])
                        for x in d:
                            x["where"] = analysis.describe_path(r["module"], x["path"])
                        checks["invariance"]["differences"] = d
                except Exception as e:  # noqa: BLE001
                    checks["invariance"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:4000]}
        else:
            checks["invariance"] = {"ok": not spec.symbols, "hash": r["hash"], "note": "model declares no symbols"}
        rec.update(
            {
                "hash": r["hash"],
                "roots": r["roots"],
                "coverage": r["coverage"],
                "census": cen["summary"],
                "checks": checks,
                "seconds": round(time.monotonic() - t0, 2),
                "files": {
                    f: (pdir / f).stat().st_size
                    for f in (
                        "module.mlir",
                        "exported.bin",
                        "graph.json",
                        "symbols.json",
                        "census.json",
                        "census_concrete.json",
                    )
                },
            }
        )

    model_checks: dict[str, Any] = {}
    unused = sorted(declared - used_anywhere) if phases is None else []
    model_checks["symbols_used"] = {"ok": not unused, "unused": unused}
    variants = []
    for name, rec in summary["phases"].items():
        base = rec.get("size_variant_of")
        if base is None:
            continue
        b = summary["phases"].get(base, {})
        variants.append({"phase": name, "of": base, "ok": "hash" in rec and rec.get("hash") == b.get("hash")})
    model_checks["size_variants"] = {"ok": all(v["ok"] for v in variants), "pairs": variants}
    summary["recipe"] = recipe_facts(key, spec)
    if summary["recipe"]["audit"] is not None and phases is None:
        from models.quant_recipes import RECIPES
        from verify.dtype_audit import check

        per_phase = {p: r["census"] for p, r in summary["phases"].items() if "census" in r}
        errs = check(RECIPES[key], per_phase, cfg_name, set(summary["recipe"]["param_dtypes"]))
        model_checks["recipe_audit"] = {"ok": not errs, "errors": errs}
    summary["checks"] = model_checks
    _dump(out / "export.json", summary)
    return summary


def recipe_facts(key: str, spec) -> dict:
    """The recipe's audit (as data, for the Rust-side check) and the parameter storage dtypes."""
    from models.quant_recipes import RECIPES

    rec = RECIPES.get(key)  # None for toy ports (tests)
    dtypes = sorted({str(leaf.dtype) for leaf in jax.tree_util.tree_leaves(spec.param_specs)})
    return {"audit": None if rec is None else rec["audit"], "param_dtypes": dtypes}


def versions() -> dict[str, str]:
    import importlib.metadata as md

    out = {}
    for pkg in ("jax", "jaxlib", "flax", "dm-haiku", "keras", "numpy"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = "absent"
    return out
