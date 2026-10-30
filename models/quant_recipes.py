"""The quantization recipe of every port: which datatypes each model computes in.

Each model declares its recipe as `RECIPE` in models/<key>/recipe.py; RECIPES collects
them by catalog key. Each port implements exactly its entry, and verify/dtype_audit.py
checks the lowered graphs against `audit`.

Selection rule: the model owner's official quantized checkpoint where one exists;
otherwise the most-deployed MLPerf / production recipe for the model's platform.
Models that are not served quantized in practice get the platform fallback: FP8 E4M3
W8A8 per-tensor (GPU datacenter) or INT8 W8A8, per-channel weights + per-tensor static
activations (edge / automotive / mobile / recsys). In every recipe, norms, softmax,
scans, FFTs and decode / NMS math stay in floating point unless stated.

Fields
  recipe        one-line name
  basis         "official" | "mlperf" | "production" | "fallback"
  sources       primary sources the recipe was confirmed against
  weights       weight dtype + scale granularity
  activations   activation dtype + scale granularity (static / dynamic)
  accumulator   GEMM accumulator dtype
  storage       quantized storage that is not a GEMM operand (KV cache, embedding /
                feature tables), or None
  float_ops     ops kept in floating point, with their dtype
  audit         {"allow": {signature: why}, "require": [signature, ...]}: every
                dot_general / convolution in every phase must have an allowed signature
                ("<kind> <lhs>x<rhs>-><acc>", StableHLO element types), and each required
                signature must occur. Optional: audit["phases"][phase] = {"allow", "require"}
                (per-phase lists), audit["counts"][cfg][phase] = {signature: n} (exact op
                counts) and audit["params"] = {dtype: why} (parameter storage dtypes).
  notes         deviations from the research summary, caveats
"""

import importlib

from models.catalog import MODELS


def _load():
    out = {}
    for m in MODELS:
        try:
            out[m["key"]] = importlib.import_module(f"models.{m['key']}.recipe").RECIPE
        except ModuleNotFoundError:
            pass
    return out


RECIPES = _load()
