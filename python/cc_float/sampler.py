"""Composition-error sampler: the error of each composition of cores, per format, against the budget
(docs/framework/float.md, "Compositions").

Cores are modelled at their contracts (`fpmodel.core`): a correctly rounded core rounds the reference; an
`e`-ulp core is moved `e - 1/2` ulps to one side first (both sides tried, every combination for two
cores), the worst case of a core that errs systematically. Inputs: every value (pairs for power) of the
16-bit and 8-bit formats; for f32, every binade with seeded random significands plus the hard cases of
each recipe. Errors are measured before the flush of a flushing format (the flush is glue, proven
separately).

    python -m cc_float.sampler --out compositions.json [--formats f32,bf16,...] [--jobs 12]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import compositions as C
from .fpmodel import F32, FORMATS, Fmt, all_values, round_result, ulp_error

#: Core error classes sampled (ulps of the inner precision): correctly rounded, faithful, and the 2-ulp
#: DesignWare exponentials and binary logarithm (the largest guarantee among the providers).
CLASSES = {"correctly_rounded": 0.5, "faithful": 1.0, "ulp2": 2.0}

#: The recipes per operator, and the core roles whose errors size them.
RECIPES = {
    "power": ("power", ["log", "exp"]),
    "tanh": ("tanh", ["exp"]),
    "log_plus_one": ("log_plus_one", ["log"]),
    "rsqrt": ("rsqrt", ["sqrt"]),
}

#: Operators computed by one f32 core for the narrower formats.
WIDER_OPS = ["exponential", "log", "log_plus_one", "tanh", "rsqrt", "sine", "cosine"]

SEED = 20261105


def modes(n: int, e: float) -> list[tuple[str, ...]]:
    if e <= 0.5:
        return [("cr",) * n]
    return [("cr",) * n] + list(itertools.product("+-", repeat=n))


def domain(op: str, x: np.ndarray) -> np.ndarray:
    """Ordinary operands (the special values are glue)."""
    if op in ("log", "rsqrt"):
        return x[(x > 0) & np.isfinite(x)]
    if op == "log_plus_one":
        return x[(x > -1) & (x != 0) & np.isfinite(x)]
    return x[(x != 0) & np.isfinite(x)]


def f32_unary(op: str, n: int = 256) -> np.ndarray:
    """Every f32 binade (seeded random significands, both ends), and the recipe's hard cases."""
    rng = np.random.default_rng(SEED)
    out = []
    for e in range(F32.emin, F32.emax + 1):
        sig = np.concatenate([[1.0, 2.0 - 2.0**-23], 1.0 + rng.integers(0, 1 << 23, n) / 2.0**23])
        v = np.ldexp(sig, e)
        out += [v, -v]
    if op == "tanh":
        for c in (C.tanh_cubic_limit(F32), C.tanh_clamp(F32)):
            out.append(c * (1.0 + np.linspace(-2.0**-10, 2.0**-10, 4097)))
    if op == "log_plus_one":
        out.append(-1.0 + np.ldexp(1.0, -np.arange(1, 25)))
        out.append(np.ldexp(1.0, -np.arange(1, 126)) * (1 + rng.random(125)))
    x = np.concatenate(out).astype(np.float32).astype(np.float64)
    return domain(op, x)


def f32_pairs(n: int = 400_000) -> tuple[np.ndarray, np.ndarray]:
    """Pairs for power: log-uniform x and y, x near 1 with large y (|y log2 x| near tau), integer y."""
    rng = np.random.default_rng(SEED + 1)
    x = np.exp2(rng.uniform(-126, 128, n))
    y = np.sign(rng.uniform(-1, 1, n)) * np.exp2(rng.uniform(-30, 10, n))
    k = rng.integers(1, 24, n // 4)
    xn = 1.0 + np.sign(rng.uniform(-1, 1, n // 4)) * np.ldexp(1.0, -k) * (1 + rng.random(n // 4))
    yn = rng.uniform(-1, 1, n // 4) * 128.0 / np.abs(np.log2(xn))
    yi = rng.integers(-64, 64, n // 4).astype(np.float64)
    xi = np.exp2(rng.uniform(-2, 2, n // 4))
    xs = np.concatenate([x, xn, xi]).astype(np.float32).astype(np.float64)
    ys = np.concatenate([y, yn, yi]).astype(np.float32).astype(np.float64)
    ok = (xs > 0) & np.isfinite(xs) & np.isfinite(ys) & (ys != 0) & (xs != 1)
    return xs[ok], ys[ok]


def reference(op: str, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    with np.errstate(all="ignore"):
        if op == "power":
            return np.power(x, y)
        return C.REFERENCE[op](x)


def evaluate(f: Fmt, op: str, recipe: str, q: int, err: dict[str, float], x, y=None):
    """Max error (ulps of `f`) over every mode of the cores, and the worst input."""
    v = reference(op, x, y)
    keep = np.isfinite(v)
    if op == "power":
        # Results beyond the format's range: overflow and underflow are exact decisions of the glue.
        keep &= (np.abs(v) <= 2.0 * f.max) & (np.abs(v) >= 2.0 ** (f.emin - f.m))
    x, v = x[keep], v[keep]
    y = y[keep] if y is not None else None
    roles = RECIPES[recipe][1] if recipe != "wider" else ["core"]
    worst, at = 0.0, None
    for ms in modes(len(roles), max(err[r] for r in roles)):
        if recipe == "power":
            w = C.power(x, y, q, err, ms)
        elif recipe == "tanh":
            w = C.tanh(x, q, err, ms, f)
        elif recipe == "log_plus_one":
            w = C.log_plus_one(x, q, err, ms)
        elif recipe == "rsqrt":
            w = C.rsqrt(x, q, err, ms)
        else:
            w = C.wider(op, x, q, err, ms)
        r = round_result(f, w, flush=False)
        e = ulp_error(f, r, v)
        e = np.where(np.isnan(e), np.inf, e)
        i = int(np.argmax(e)) if len(e) else 0
        if len(e) and e[i] > worst:
            worst = float(e[i])
            at = [float(x[i])] + ([float(y[i])] if y is not None else [])
    return worst, at, int(len(x))


def _pow_chunk(args):
    fname, q, err, xs = args
    f = FORMATS[fname]
    ys = all_values(f)
    ys = ys[ys != 0]
    xx = np.repeat(xs, len(ys))
    yy = np.tile(ys, len(xs))
    return evaluate(f, "power", "power", q, err, xx, yy)


def pow_exhaustive(f: Fmt, q: int, err: dict[str, float], jobs: int):
    """Every positive x against every non-zero y, in chunks over x."""
    xs = all_values(f)
    xs = xs[(xs > 0) & (xs != 1)]
    chunk = max(1, (1 << 20) // len(all_values(f)))
    parts = [(f.name, q, err, xs[i : i + chunk]) for i in range(0, len(xs), chunk)]
    worst, at, n = 0.0, None, 0
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for w, a, k in ex.map(_pow_chunk, parts):
            n += k
            if w > worst:
                worst, at = w, a
    return worst, at, n


def row(f: Fmt, op: str, recipe: str, cls: str, jobs: int, k: int | None = None) -> dict:
    roles = RECIPES[recipe][1] if recipe != "wider" else ["core"]
    e = CLASSES[cls]
    err = dict.fromkeys(roles, e)
    t = C.budget(op, f)
    if recipe == "wider":
        q = F32.p
        k = q - f.p
        c = e
    else:
        c = C.constant(recipe, f, err)
        if k is None:
            k = C.extra_bits(c, t)
        q = f.p + k
    if f.name == "f32":
        if recipe == "power":
            x, y = f32_pairs()
            worst, at, n = evaluate(f, op, recipe, q, err, x, y)
        else:
            worst, at, n = evaluate(f, op, recipe, q, err, f32_unary(op))
        method = "stratified"
    elif recipe == "power":
        worst, at, n = pow_exhaustive(f, q, err, jobs)
        method = "exhaustive"
    else:
        worst, at, n = evaluate(f, op, recipe, q, err, domain(op, all_values(f)))
        method = "exhaustive"
    return {
        "op": op,
        "format": f.name,
        "recipe": recipe,
        "core_class": cls,
        "core_errors": err,
        "constant": c,
        "k": k,
        "q": q,
        "bound_ulp": 0.5 + c * 2.0**-k,
        "budget_ulp": t,
        "inputs": n,
        "method": method,
        "max_ulp": worst,
        "worst_input": at,
        "within_budget": bool(worst <= t),
    }


def fp8_arith(f: Fmt) -> list[dict]:
    """+, -, *, /, sqrt of an 8-bit format on f16 cores: every pair, against one rounding of the exact
    value (Figueroa's double-rounding condition q >= 2p + 2 holds for f16)."""
    from .fpmodel import F16, rnd

    v = all_values(f)
    x = np.repeat(v, len(v))
    y = np.tile(v, len(v))
    out = []
    ops = {
        "add": np.add,
        "subtract": np.subtract,
        "multiply": np.multiply,
        "divide": np.divide,
    }
    with np.errstate(all="ignore"):
        for name, fn in ops.items():
            exact = fn(x, y)
            via = round_result(f, rnd(exact, F16.p, F16.emin))
            direct = round_result(f, exact)
            same = (via == direct) | (np.isnan(via) & np.isnan(direct))
            out.append(
                {
                    "op": name,
                    "format": f.name,
                    "recipe": "wider_f16",
                    "inputs": int(len(x)),
                    "method": "exhaustive",
                    "not_correctly_rounded": int((~same).sum()),
                }
            )
        s = v[v >= 0]
        exact = np.sqrt(s)
        via = round_result(f, rnd(exact, F16.p, F16.emin))
        direct = round_result(f, exact)
        out.append(
            {
                "op": "sqrt",
                "format": f.name,
                "recipe": "wider_f16",
                "inputs": int(len(s)),
                "method": "exhaustive",
                "not_correctly_rounded": int((via != direct).sum()),
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--formats", default="f32,bf16,f16,f8e4m3fn")
    ap.add_argument("--ops", default=",".join(RECIPES))
    ap.add_argument("--classes", default=",".join(CLASSES))
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--no-pow-exhaustive", action="store_true", help="skip power on the 16-bit formats")
    a = ap.parse_args(argv)
    rows = []
    for fname in a.formats.split(","):
        f = FORMATS[fname]
        for op in a.ops.split(","):
            if op == "power" and f.m in (7, 10) and a.no_pow_exhaustive:
                continue
            recipe = RECIPES[op][0]
            for cls in a.classes.split(","):
                r = row(f, op, recipe, cls, a.jobs)
                print(json.dumps({k: r[k] for k in ("op", "format", "core_class", "k", "max_ulp", "budget_ulp")}), flush=True)
                rows.append(r)
        if f.name != "f32":
            for op in WIDER_OPS:
                for cls in a.classes.split(","):
                    r = row(f, op, "wider", cls, a.jobs)
                    print(json.dumps({k: r[k] for k in ("op", "format", "core_class", "k", "max_ulp", "budget_ulp")}), flush=True)
                    rows.append(r)
        if f.e <= 5 and f.m <= 3:
            rows += fp8_arith(f)
    doc = {
        "budget": "CUDA math library table (f32), 1 ulp (16-bit and 8-bit formats)",
        "classes": CLASSES,
        "seed": SEED,
        "rows": rows,
    }
    with open(a.out, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    bad = [r for r in rows if not r.get("within_budget", r.get("not_correctly_rounded", 0) == 0)]
    for r in bad:
        print("over budget:", r["op"], r["format"], r.get("core_class"), r.get("max_ulp"), file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

