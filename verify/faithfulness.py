"""Static faithfulness check of a port against its upstream model (no execution).

    python -m verify.faithfulness <key> [--cfg full] [--no-flops]

Each model needs a spec module `verify/upstream/<key>.py` (none ship with the repo; write
one from primary sources before running) exporting:

  REFERENCE: str                     upstream code + checkpoint the port claims to be
  config_pairs(cfg) -> list[(field, port value, upstream value)]
        port Config values next to the upstream config.json / yaml / source constants
  upstream_params() -> {name: shape}  the upstream checkpoint's tensors (safetensors
        header via verify.fetch, or derived from the upstream module definitions)
  rename(port_path) -> upstream name | None
        maps a port parameter path ("a/b/kernel") to the upstream tensor; None = the
        port has no upstream counterpart (must be listed in PORT_ONLY with a reason)
  PORT_ONLY: dict[pattern, reason]    regexes of port params with no upstream tensor
  UPSTREAM_ONLY: dict[pattern, reason] regexes of upstream tensors the port omits
                                       (buffers, training-only heads, unused encoders)
  FOLDED: dict[pattern, reason]        upstream tensors folded into other port tensors by
                                       the quantization recipe (e.g. BN into conv weights)
  PUBLISHED_PARAMS: int | None         published parameter count of the ported part
  PUBLISHED_EXCLUDE: regex | None      upstream tensors not in the published count (buffers)
  expected_flops(cfg) -> {phase: (flops, rel_tol)}  analytic FLOPs (2 * MACs) per call

Checks:
  1. config     every pair equal
  2. params     every port tensor maps to an upstream tensor with the same shape up to a
                permutation of axes (or, for MERGE groups, a reshape of the same numel);
                every upstream tensor is used or explained; totals agree
  3. structure  FLOPs of every phase (abstract lowering) within tolerance of the analytic
                count from the upstream architecture
A source-level review of the port against the upstream forward pass complements these checks.
"""

from __future__ import annotations

import argparse
import collections
import importlib
import json
import math
import os
import re
import sys

import jax

from verify import hlo

REPORTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def port_params(key: str, cfg_name: str = "full") -> dict[str, tuple[tuple[int, ...], str]]:
    mod = hlo.load(key)
    spec = mod.build(mod.CONFIGS[cfg_name])
    out = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(spec.param_specs)[0]:
        name = "/".join(str(getattr(k, "key", getattr(k, "name", getattr(k, "idx", k)))) for k in path)
        out[name] = (tuple(leaf.shape), str(leaf.dtype))
    return out


def _matches(patterns: dict, name: str) -> str | None:
    for pat, reason in patterns.items():
        if re.search(pat, name):
            return reason
    return None


def check_params(up_mod, port: dict) -> dict:
    up = {k: tuple(v[0] if isinstance(v, tuple) and v and isinstance(v[0], tuple) else v)
          for k, v in up_mod.upstream_params().items()}
    used: dict[str, list[str]] = collections.defaultdict(list)
    bad, port_only, unexplained = [], [], []
    for path, (shape, _) in port.items():
        name = up_mod.rename(path)
        if name is None:
            reason = _matches(getattr(up_mod, "PORT_ONLY", {}), path)
            (port_only if reason else unexplained).append((path, reason))
            continue
        if name not in up:
            unexplained.append((path, f"-> {name} not in upstream"))
            continue
        used[name].append(path)
    # shape check per upstream tensor (several port tensors may split one upstream tensor)
    for name, paths in used.items():
        ushape = up[name]
        pshapes = [port[p][0] for p in paths]
        if len(paths) == 1:
            ps = pshapes[0]
            ok = ps == ushape or sorted(ps) == sorted(ushape) or (
                math.prod(ps) == math.prod(ushape) and _matches(getattr(up_mod, "RESHAPE_OK", {}), paths[0]))
        else:
            ok = sum(math.prod(s) for s in pshapes) == math.prod(ushape)
        if not ok:
            bad.append((name, ushape, list(zip(paths, pshapes))))
    folded = {n for n in up if n not in used and _matches(getattr(up_mod, "FOLDED", {}), n)
              and not _matches(getattr(up_mod, "UPSTREAM_ONLY", {}), n)}
    upstream_left = []
    for name, shape in up.items():
        if name in used or name in folded:
            continue
        reason = _matches(getattr(up_mod, "UPSTREAM_ONLY", {}), name)
        if reason is None:
            upstream_left.append((name, shape))
    n_port = sum(math.prod(s) for s, _ in port.values())
    n_port_mapped = sum(math.prod(port[p][0]) for ps in used.values() for p in ps)
    consumed = set(used) | folded
    n_up_used = sum(math.prod(up[n]) for n in consumed)
    excl = getattr(up_mod, "PUBLISHED_EXCLUDE", None)
    n_up_learnable = sum(math.prod(up[n]) for n in consumed if not (excl and re.search(excl, n)))
    return {"port_tensors": len(port), "upstream_tensors": len(up), "mapped": sum(map(len, used.values())), "folded": len(folded),
            "shape_mismatch": bad, "port_unexplained": unexplained,
            "port_only": dict(collections.Counter(r for _, r in port_only)),
            "upstream_unexplained": upstream_left, "port_params": n_port,
            "port_params_mapped": n_port_mapped, "upstream_params_used": n_up_used, "upstream_learnable_used": n_up_learnable,
            "published": getattr(up_mod, "PUBLISHED_PARAMS", None)}


def check_flops(key: str, up_mod, cfg_name: str) -> dict:
    mod = hlo.load(key)
    cfg = mod.CONFIGS[cfg_name]
    exp = up_mod.expected_flops(cfg)
    texts = hlo.lower_phases(key, cfg_name, phases=set(exp))
    out = {}
    for ph, (want, tol) in exp.items():
        got = sum(op.flops for op in hlo.census(texts[ph]))
        out[ph] = {"port": got, "expected": want, "ratio": got / want if want else None,
                   "ok": abs(got - want) <= tol * want}
    return out


def run(key: str, cfg_name: str = "full", flops: bool = True) -> dict:
    up_mod = importlib.import_module(f"verify.upstream.{key}")
    mod = hlo.load(key)
    cfg = mod.CONFIGS[cfg_name]
    pairs = up_mod.config_pairs(cfg)
    res = {"key": key, "reference": up_mod.REFERENCE,
           "config": {"checked": len(pairs),
                      "mismatch": [(f, p, u) for f, p, u in pairs if p != u]}}
    res["params"] = check_params(up_mod, port_params(key, cfg_name))
    if flops:
        res["flops"] = check_flops(key, up_mod, cfg_name)
    p = res["params"]
    res["ok"] = (not res["config"]["mismatch"] and not p["shape_mismatch"]
                 and not p["port_unexplained"] and not p["upstream_unexplained"]
                 and (p["published"] is None or p["published"] == p["upstream_learnable_used"])
                 and all(v["ok"] for v in res.get("flops", {}).values()))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("key")
    ap.add_argument("--cfg", default="full")
    ap.add_argument("--no-flops", action="store_true")
    a = ap.parse_args(argv)
    res = run(a.key, a.cfg, not a.no_flops)
    os.makedirs(os.path.join(REPORTS, "faithfulness"), exist_ok=True)
    with open(os.path.join(REPORTS, "faithfulness", f"{a.key}.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    p = res["params"]
    print(f"{a.key}: {'OK' if res['ok'] else 'FAIL'}  ({res['reference']})")
    print(f"  config: {res['config']['checked']} fields, mismatches {res['config']['mismatch']}")
    print(f"  params: {p['mapped']}/{p['port_tensors']} port tensors mapped ({p['folded']} upstream folded) onto "
          f"{p['upstream_tensors']} upstream; port {p['port_params']:,} "
          f"(mapped {p['port_params_mapped']:,}), upstream used {p['upstream_params_used']:,} "
          f"(learnable {p['upstream_learnable_used']:,}), published {p['published']}")
    for k in ("shape_mismatch", "port_unexplained", "upstream_unexplained"):
        if p[k]:
            print(f"  {k} ({len(p[k])}): {p[k][:8]}")
    if p["port_only"]:
        print(f"  port_only ({sum(p['port_only'].values())}): {p['port_only']}")
    for ph, v in res.get("flops", {}).items():
        print(f"  flops {ph}: port {v['port']:.4e} expected {v['expected']:.4e} "
              f"ratio {v['ratio']:.4f} {'ok' if v['ok'] else 'MISMATCH'}")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
