"""AlphaFold 2 monomer (model_1_ptm): one recycling iteration, static + unrolled,
FP8 W8A8 on a bf16 trunk.

The network is DeepMind's own JAX/Haiku code (google-deepmind/alphafold
c77e5d2a), vendored at upstream/alphafold; only `alphafold/model/*` is
imported (the HMMER/HHblits/TF data pipeline is excluded).

Quantization (GPU-datacenter fallback: AF2 is served bf16/fp32, never
quantized; see recipe.py) is written into the vendored files, each edit
marked `# quantized: FP8 W8A8` (pristine copy: git 7b50334):
  * bf16 trunk exactly as AF-Multimer runs it: config.py gains multimer's
    `global_config.bfloat16` / `bfloat16_output`; EmbeddingsAndEvoformer
    casts its features to bf16, runs under `utils.bfloat16_context()` (fp32
    params cast on read) and returns fp32. LayerNorm and softmax compute in
    fp32; the structure module and all heads stay fp32.
  * `common_modules.get_weight` serves every weight GEMM (Linear and the
    get_parameter + einsum projections of Attention, GlobalAttention, the
    pair biases and OuterProductMean): a weight requested in bf16 is stored
    float8_e4m3fn with per-tensor `<name>_scale` / `<name>_input_scale`; the
    input is quantized, f8 x f8 -> f32, rescaled to bf16. Attention BMMs,
    the triangle-multiplication einsum and the OPM outer product stay bf16.

What this file adds on top of the library -- config, scoped
monkeypatches for the static graph (none of them for quantization):

  * `RecyclingIteration` -- the body of `modules.AlphaFold.__call__`'s
    recycling `hk.while_loop` (`do_call` + `get_prev`) as a stand-alone module
    named "alphafold", so the parameter tree is exactly the canonical one
    (`alphafold/alphafold_iteration/...`). prev_pos / prev_msa_first_row /
    prev_pair are explicit inputs; the new prev_* are outputs. The outer loop
    is not traced (Phase.trip_count = num_recycle + 1).
  * `patched_config` -- config-level patches: no inference subbatching
    (`global_config.subbatch_size`, `outer_product_mean.chunk_size`,
    `template.subbatch_size` = None, which `mapping.sharded_apply` documents as
    "no sharding"), `global_config.deterministic = True` (eval, no dropout;
    with is_training=False dropout is already off), and the recipe's
    `bfloat16 = True`, `bfloat16_output = False`.
  * `unrolled()` -- while active, `hk.scan` is called with `unroll=True`, so
    every `layer_stack` (Evoformer 48, extra-MSA 4, template pair stack 2)
    and the per-template `sharded_map` (shard size 1) lower to straight-line
    code with the canonical stacked parameters sliced per layer. The structure
    module's 8 iterations are already a Python loop in folding.py. A scan whose
    length is symbolic (the template map in the symbolic export, T templates)
    stays a loop.
  * API / dtype shims applied to patched model AND reference alike:
    `_JnpCompat` (jnp.clip a_min/a_max keywords removed in JAX 0.12),
    `_stable_softmax` (fp64 dispatch for an fp64 config), `_x32_tables`
    (residue_constants tables to 32-bit; a no-op under x32).
"""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass

AF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upstream")
if AF_DIR not in sys.path:
    sys.path.insert(0, AF_DIR)

import haiku as hk  # noqa: E402
import jax  # noqa: E402

import numpy as np  # noqa: E402
from alphafold.common import residue_constants  # noqa: E402
from alphafold.model import config as af_config  # noqa: E402
from alphafold.model import modules  # noqa: E402
from alphafold.model import utils as af_utils  # noqa: E402

MODEL_NAME = "model_1_ptm"


class _JnpCompat:
    """jax.numpy for modules.py with the pre-0.6 `jnp.clip(a_min=, a_max=)`
    keywords (removed in our JAX 0.12; modules.py's relpos uses them). Pure API
    shim, same op (stablehlo.clamp); applies to patched and reference alike."""

    def __getattr__(self, name):
        return getattr(jax.numpy, name)

    @staticmethod
    def clip(x, min=None, max=None, *, a_min=None, a_max=None):  # noqa: A002
        return jax.numpy.clip(x, a_min if min is None else min,
                              a_max if max is None else max)


modules.jnp = _JnpCompat()

_orig_stable_softmax = af_utils.stable_softmax


def _stable_softmax(logits):
    """utils.stable_softmax only dispatches fp32 / bf16 and raises otherwise;
    add fp64 (same jax.nn.softmax) for an fp64 config. fp32
    graphs take the library function unchanged."""
    if logits.dtype == jax.numpy.float64:
        return jax.nn.softmax(logits)
    return _orig_stable_softmax(logits)


af_utils.stable_softmax = _stable_softmax


def _x32_tables():
    """residue_constants' int64 / float64 numpy tables -> int32 / float32.

    A no-op for every fp32 graph (JAX canonicalizes them to 32 bits under
    x32 anyway). It keeps an fp64 run consistent: such a run switches
    jax_enable_x64 on mid-process, and JAX then lowered a 64-bit numpy table
    as i32 into an i64 call (MLIR verifier error) while x32 lowerings were
    still alive."""
    for k, v in list(vars(residue_constants).items()):
        if isinstance(v, np.ndarray) and v.dtype in (np.int64, np.float64):
            setattr(residue_constants, k, v.astype(np.int32 if v.dtype == np.int64
                                                   else np.float32))


_x32_tables()


@dataclass(frozen=True)
class Config:
    num_res: int = 256
    # max_msa_clusters = 512; model_1 sets reduce_msa_clusters_by_max_templates,
    # so the data pipeline emits 512 - 4 = 508 cluster rows and the model appends
    # the 4 template torsion-angle rows -> 512 rows in the Evoformer.
    num_msa: int = 508
    num_extra_msa: int = 5120          # model_1: data.common.max_extra_msa = 5120
    num_templates: int = 4
    num_ensemble: int = 1
    num_recycle: int = 3
    # (dotted key, value) overrides of the model_1_ptm ConfigDict (architecture
    # shrink for the reduced config, and the reference's chunk sizes).
    overrides: tuple = ()
    # Activation recording mode (model.record_module_outputs); not network.
    eager_activations: bool = False
    # Synthetic-input style (model.synthetic_features); not network.
    coherent_features: bool = False
    # Input-feature / fp32-region dtype (the trunk is bf16 + FP8 regardless).
    dtype: str = "float32"


def model_config(cfg: Config):
    """Canonical model_1_ptm `config.model` (+ cfg.overrides): the reference."""
    c = af_config.model_config(MODEL_NAME).model
    c.num_recycle = cfg.num_recycle
    c.embeddings_and_evoformer.template.max_templates = cfg.num_templates
    for key, value in cfg.overrides:
        node = c
        *path, leaf = key.split(".")
        for p in path:
            node = node[p]
        node[leaf] = value
    return c


def patched_config(cfg: Config):
    """Static-graph config: subbatching off, deterministic eval; FP8 recipe on: bf16
    trunk (whose weight GEMMs are FP8 W8A8), fp32 structure module and heads."""
    c = model_config(cfg)
    c.global_config.subbatch_size = None
    c.global_config.deterministic = True
    c.global_config.bfloat16 = True
    c.global_config.bfloat16_output = False
    c.embeddings_and_evoformer.evoformer.outer_product_mean.chunk_size = None
    c.embeddings_and_evoformer.template.subbatch_size = None
    return c


class RecyclingIteration(hk.Module):
    """One iteration of `modules.AlphaFold`'s recycling loop (do_call + get_prev)."""

    def __init__(self, config, name="alphafold"):
        super().__init__(name=name)
        self.config = config
        self.global_config = config.global_config

    def __call__(self, batch, prev):
        impl = modules.AlphaFoldIteration(self.config, self.global_config)
        # num_ensemble == 1, so ensemble_representations=True (RunModel's choice)
        # would only add a zero-trip ensembling while_loop and a divide by 1.
        ret = impl(ensembled_batch=batch, non_ensembled_batch=dict(prev),
                   is_training=False, compute_loss=False,
                   ensemble_representations=False)
        new_prev = {
            "prev_pos": ret["structure_module"]["final_atom_positions"],
            "prev_msa_first_row": ret["representations"]["msa_first_row"],
            "prev_pair": ret["representations"]["pair"],
        }
        return ret, jax.tree.map(jax.lax.stop_gradient, new_prev)


def make_fn(model_cfg):
    def fwd(batch, prev):
        return RecyclingIteration(model_cfg)(batch, prev)
    return hk.transform(fwd)


_orig_scan = hk.scan


def _scan_unrolled(f, init, xs, length=None, reverse=False, unroll=1):
    """Unroll fully when the trip count is a constant (layer stacks); a scan whose length
    is an input size (the per-template sharded_map over T templates) keeps its unroll,
    so the graph does not depend on that size."""
    n = length if length is not None else jax.tree.leaves(xs)[0].shape[0]
    if jax.export.is_symbolic_dim(n):
        return _scan_symbolic(f, init, xs, n, reverse)
    return _orig_scan(f, init, xs, length=length, reverse=reverse, unroll=True)


def _scan_symbolic(f, init, xs, n, reverse):
    """hk.scan over a symbolic length n (lax.scan needs a constant length): the same
    sequential loop as an hk.fori_loop over i in [0, n), x_i = xs[i]. Only scans without
    stacked outputs occur here (sharded_apply's scan_iteration returns ys = ())."""
    def body(i, carry):
        j = n - 1 - i if reverse else i
        x = jax.tree.map(lambda v: jax.lax.dynamic_index_in_dim(v, j, keepdims=False), xs)
        carry, y = f(carry, x)
        if jax.tree.leaves(y):
            raise NotImplementedError("symbolic-length hk.scan with stacked outputs")
        return carry

    return hk.fori_loop(0, n, body, init), ()


@contextlib.contextmanager
def unrolled():
    """Fully unroll every hk.scan (layer_stack, sharded_map) while active."""
    hk.scan = _scan_unrolled
    try:
        yield
    finally:
        hk.scan = _orig_scan
