"""AlphaFold 2 (model_1_ptm) capture spec: one recycling iteration.

Phase `recycle_iteration` = one call of `AlphaFoldIteration` (embeddings,
recycling embedder, template stack, extra-MSA stack, 48 Evoformer blocks,
structure module, all heads) with prev_* as inputs and the new prev_* as
outputs. modules.AlphaFold runs it num_recycle + 1 = 4 times.

Recipe (recipe.py, net.py): FP8 E4M3 W8A8 per-tensor static on every weight
GEMM of the bf16 trunk (EmbeddingsAndEvoformer); attention / triangle / OPM
products bf16; LayerNorm, softmax, structure module and heads fp32. Inputs,
prev_* and outputs stay fp32 (the trunk casts on entry and exit). Parameters:
trunk weights float8_e4m3fn plus fp32 `*_scale` / `*_input_scale` scalars.
"""

from __future__ import annotations

import re
from collections import Counter

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax._src import core as jax_core
from jax._src import source_info_util
from jax._src.interpreters import partial_eval as pe
from jax._src.lax.control_flow.loops import _stack as _scan_stack

from models.spec import ModelSpec, Phase, Sym, as_spec
from models.alphafold2 import net
from models.alphafold2.net import Config

from alphafold.common import residue_constants as rc  # noqa: E402  (path set by net)

_E = "embeddings_and_evoformer"
_T = _E + ".template"
_TPS = _T + ".template_pair_stack"
_EVO = _E + ".evoformer"
_SM = "heads.structure_module"

# Reduced: every block type kept (templates + torsion rows, extra-MSA stack with
# global column attention, Evoformer, structure module + sidechain, all heads),
# 2 layers per stack and small widths. The library chunk sizes are shrunk too
# (3 rows / 5 cols / 7 pairs) so the canonical sharded_apply really chunks,
# including its remainder path.
REDUCED_OVERRIDES = (
    (_E + ".evoformer_num_block", 2),
    (_E + ".extra_msa_stack_num_block", 2),
    (_E + ".msa_channel", 32),
    (_E + ".pair_channel", 16),
    (_E + ".seq_channel", 24),
    (_E + ".extra_msa_channel", 16),
    (_EVO + ".outer_product_mean.num_outer_channel", 8),
    (_EVO + ".outer_product_mean.chunk_size", 5),
    (_EVO + ".triangle_multiplication_outgoing.num_intermediate_channel", 16),
    (_EVO + ".triangle_multiplication_incoming.num_intermediate_channel", 16),
    (_T + ".subbatch_size", 7),
    (_T + ".attention.key_dim", 16),
    (_T + ".attention.value_dim", 16),
    (_TPS + ".num_block", 2),
    (_TPS + ".triangle_attention_starting_node.key_dim", 8),
    (_TPS + ".triangle_attention_starting_node.value_dim", 8),
    (_TPS + ".triangle_attention_ending_node.key_dim", 8),
    (_TPS + ".triangle_attention_ending_node.value_dim", 8),
    (_TPS + ".triangle_multiplication_outgoing.num_intermediate_channel", 8),
    (_TPS + ".triangle_multiplication_incoming.num_intermediate_channel", 8),
    ("global_config.subbatch_size", 3),
    (_SM + ".num_layer", 2),
    (_SM + ".num_channel", 24),
    (_SM + ".num_head", 4),
    (_SM + ".num_scalar_qk", 4),
    (_SM + ".num_scalar_v", 4),
    (_SM + ".sidechain.num_channel", 16),
    ("heads.predicted_lddt.num_channels", 16),
)

CONFIGS = {
    "full": Config(),
    "reduced": Config(num_res=16, num_msa=8, num_extra_msa=12, num_templates=2,
                      overrides=REDUCED_OVERRIDES, eager_activations=True),
}

PHASE = "recycle_iteration"


# ----------------------------------------------------------------------------
# Inputs (synthetic features with the data pipeline's names / shapes / dtypes)
# ----------------------------------------------------------------------------

def _atom_tables():
    a37_to_14 = []
    for rt in rc.restypes:
        names = rc.restype_name_to_atom14_names[rc.restype_1to3[rt]]
        idx14 = {n: i for i, n in enumerate(names)}
        a37_to_14.append([idx14.get(n, 0) for n in rc.atom_types])
    a37_to_14.append([0] * 37)
    a14_mask = [[1.0 if n else 0.0 for n in rc.restype_name_to_atom14_names[rc.restype_1to3[rt]]]
                for rt in rc.restypes] + [[0.0] * 14]
    a37_mask = np.zeros([21, 37], np.float32)
    for i, rt in enumerate(rc.restypes):
        for name in rc.residue_atoms[rc.restype_1to3[rt]]:
            a37_mask[i, rc.atom_order[name]] = 1
    return np.array(a37_to_14, np.int32), a37_mask, np.array(a14_mask, np.float32)


def float_dtype(cfg: Config):
    """fp32, or fp64 (enables jax_enable_x64 on first use)."""
    if cfg.dtype == "float64" and not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)
        jax.clear_caches()      # functions traced under x32 must not be reused
    return jnp.dtype(cfg.dtype)


def input_spec(cfg: Config):
    N, S, E, T, B = cfg.num_res, cfg.num_msa, cfg.num_extra_msa, cfg.num_templates, cfg.num_ensemble
    f32, i32 = float_dtype(cfg), jnp.int32
    mc = net.model_config(cfg)[_E]
    batch = {
        "aatype": ((B, N), i32),
        "target_feat": ((B, N, 22), f32),
        "residue_index": ((B, N), i32),
        "seq_mask": ((B, N), f32),
        "msa_feat": ((B, S, N, 49), f32),
        "msa_mask": ((B, S, N), f32),
        "extra_msa": ((B, E, N), i32),
        "extra_has_deletion": ((B, E, N), f32),
        "extra_deletion_value": ((B, E, N), f32),
        "extra_msa_mask": ((B, E, N), f32),
        "template_aatype": ((B, T, N), i32),
        "template_all_atom_positions": ((B, T, N, 37, 3), f32),
        "template_all_atom_masks": ((B, T, N, 37), f32),
        "template_mask": ((B, T), f32),
        "template_pseudo_beta": ((B, T, N, 3), f32),
        "template_pseudo_beta_mask": ((B, T, N), f32),
        "residx_atom37_to_atom14": ((B, N, 37), i32),
        "atom37_atom_exists": ((B, N, 37), f32),
    }
    prev = {
        "prev_pos": ((N, 37, 3), f32),
        "prev_msa_first_row": ((N, mc.msa_channel), f32),
        "prev_pair": ((N, N, mc.pair_channel), f32),
    }
    to_spec = lambda d: {k: jax.ShapeDtypeStruct(s, t) for k, (s, t) in d.items()}
    return to_spec(batch), to_spec(prev)


# Symbolic input shapes: N residues, S MSA cluster rows, E extra-MSA rows. The template
# slots stay concrete: AlphaFold always feeds a fixed number of padded slots with a template_mask
# (a padded capacity, decided with the user). The leading ensemble dim is 1 (AlphaFoldIteration
# asserts it without ensembling).
INPUT_SHAPES = (
    {
        "aatype": "_, N", "target_feat": "_, N, _", "residue_index": "_, N", "seq_mask": "_, N",
        "msa_feat": "_, S, N, _", "msa_mask": "_, S, N",
        "extra_msa": "_, E, N", "extra_has_deletion": "_, E, N",
        "extra_deletion_value": "_, E, N", "extra_msa_mask": "_, E, N",
        "template_aatype": "_, _, N", "template_all_atom_positions": "_, _, N, _, _",
        "template_all_atom_masks": "_, _, N, _", "template_mask": None,
        "template_pseudo_beta": "_, _, N, _", "template_pseudo_beta_mask": "_, _, N",
        "residx_atom37_to_atom14": "_, N, _", "atom37_atom_exists": "_, N, _",
    },
    {"prev_pos": "N, _, _", "prev_msa_first_row": "N, _", "prev_pair": "N, N, _"},
)


def _backbone(rng: np.random.Generator, n: int, ideal: bool = False) -> np.ndarray:
    """Chain-like atom37 coordinates [n, 37, 3]: a CA walk with 3.8 A steps.

    ideal=False: other atoms scattered +-1.2 A around CA. ideal=True: a
    persistent walk, and N / C / CB / O placed at their literature positions
    (residue_constants.rigid_group_atom_positions, ALA) in a local frame,
    side-chain atoms around CB."""
    steps = rng.normal(size=(n, 3))
    if ideal:
        for i in range(1, n):
            steps[i] = 0.8 * steps[i - 1] / np.linalg.norm(steps[i - 1]) + 0.6 * steps[i]
    steps *= 3.8 / np.linalg.norm(steps, axis=-1, keepdims=True)
    ca = np.cumsum(steps, axis=0)
    if not ideal:
        pos = ca[:, None, :] + rng.normal(scale=1.2, size=(n, 37, 3))
        pos[:, rc.atom_order["CA"]] = ca
        return pos.astype(np.float32)
    nxt = np.concatenate([ca[1:], ca[-1:] + steps[-1:]])
    prv = np.concatenate([ca[:1] - steps[:1], ca[:-1]])
    e1 = nxt - prv
    e1 /= np.linalg.norm(e1, axis=-1, keepdims=True)
    u = np.cross(nxt - ca, ca - prv) + 1e-3 * rng.normal(size=(n, 3))
    u -= (u * e1).sum(-1, keepdims=True) * e1
    e2 = u / np.linalg.norm(u, axis=-1, keepdims=True)
    rot = np.stack([e1, e2, np.cross(e1, e2)], axis=-1)          # local -> global
    local = {a: np.array(x) for a, _, x in rc.rigid_group_atom_positions["ALA"]}
    local["O"] = local["C"] + local["O"]
    pos = (ca + np.einsum("nij,j->ni", rot, local["CB"]))[:, None, :] + \
        rng.normal(scale=1.0, size=(n, 37, 3))
    for a, x in local.items():
        pos[:, rc.atom_order[a]] = ca + np.einsum("nij,j->ni", rot, x)
    return pos.astype(np.float32)


def _mutate(rng, seq, n, rate, vocab):
    rows = np.repeat(seq[None], n, 0)
    hit = rng.uniform(size=rows.shape) < rate
    rows[hit] = rng.integers(0, vocab, size=hit.sum())
    return rows


def synthetic_features(cfg: Config, seed: int):
    """Synthetic features with the pipeline's names / shapes / dtypes.

    cfg.coherent_features=False: independent random MSA / templates / prev_*
    (exercises every path with non-trivial values).
    cfg.coherent_features=True: the inputs of a first recycling iteration
    of a family -- MSA / extra-MSA rows are 15%-mutated copies of the query,
    the templates are noisy copies of one chain, prev_* = 0, no padded
    residues (monomer inference runs at the true length). Garbage inputs put
    the trained structure module in an ill-conditioned regime where the
    canonical model itself moves its atoms by ~1e-3 relative under a 1e-6
    input perturbation.
    """
    rng = np.random.default_rng(seed)
    bspec, pspec = input_spec(cfg)
    N, S, E, T = cfg.num_res, cfg.num_msa, cfg.num_extra_msa, cfg.num_templates
    coh = cfg.coherent_features
    a37_to_14, a37_mask, _ = _atom_tables()
    aatype = rng.integers(0, 20, size=N).astype(np.int32)
    msa = _mutate(rng, aatype, S, 0.15, 21) if coh else rng.integers(0, 22, size=(S, N))
    msa[0] = aatype
    if coh:
        profile = np.eye(23)[msa].mean(0, keepdims=True).repeat(S, 0) * 0.9 + 0.1 / 23
        dele = np.zeros((S, N, 1))
        msa_feat = np.concatenate([np.eye(23)[msa], dele, dele, profile, dele], axis=-1)
    else:
        profile = rng.dirichlet(np.ones(23), size=(S, N))
        msa_feat = np.concatenate([
            np.eye(23)[msa], rng.integers(0, 2, (S, N, 1)), rng.uniform(0, 1, (S, N, 1)),
            profile, rng.uniform(0, 1, (S, N, 1))], axis=-1)
    if coh:
        chain = _backbone(rng, N, ideal=True)
        t_aatype = _mutate(rng, aatype, T, 0.2, 20).astype(np.int32)
        t_pos = np.stack([chain + rng.normal(scale=0.3, size=chain.shape) for _ in range(T)])
    else:
        t_aatype = rng.integers(0, 21, size=(T, N)).astype(np.int32)
        t_pos = np.stack([_backbone(rng, N) for _ in range(T)]).astype(np.float32)
    t_mask = a37_mask[t_aatype] * (rng.uniform(size=(T, N, 1)) > 0.1)
    is_gly = t_aatype == rc.restype_order["G"]
    cb = np.where(is_gly, rc.atom_order["CA"], rc.atom_order["CB"])
    t_pb = np.take_along_axis(t_pos, cb[..., None, None].repeat(3, -1), axis=2)[:, :, 0]
    t_pb_mask = np.take_along_axis(t_mask, cb[..., None], axis=2)[..., 0]
    tmpl_mask = np.ones(T, np.float32)
    if T > 1:
        tmpl_mask[-1] = 0.0                     # one padded template slot
    seq_mask = np.ones(N, np.float32)
    if not coh:                                 # (monomer inference does not pad)
        seq_mask[-max(1, N // 16):] = 0.0       # padded tail residues
    extra = (_mutate(rng, aatype, E, 0.15, 21) if coh else rng.integers(0, 23, size=(E, N)))
    batch = {
        "aatype": aatype,
        "target_feat": np.concatenate([np.zeros((N, 1)), np.eye(21)[aatype]], -1),
        "residue_index": np.arange(N),
        "seq_mask": seq_mask,
        "msa_feat": msa_feat,
        "msa_mask": (rng.uniform(size=(S, N)) > 0.05) * seq_mask,
        "extra_msa": extra,
        "extra_has_deletion": (0 if coh else 1) * rng.integers(0, 2, size=(E, N)),
        "extra_deletion_value": (0 if coh else 1) * rng.uniform(0, 1, size=(E, N)),
        "extra_msa_mask": (rng.uniform(size=(E, N)) > 0.05) * seq_mask,
        "template_aatype": t_aatype,
        "template_all_atom_positions": t_pos * t_mask[..., None],
        "template_all_atom_masks": t_mask,
        "template_mask": tmpl_mask,
        "template_pseudo_beta": t_pb * t_pb_mask[..., None],
        "template_pseudo_beta_mask": t_pb_mask,
        "residx_atom37_to_atom14": a37_to_14[aatype],
        "atom37_atom_exists": a37_mask[aatype],
    }
    batch = {k: jnp.asarray(np.asarray(v)[None], bspec[k].dtype) for k, v in batch.items()}
    prev_pos = _backbone(rng, N) * a37_mask[aatype][..., None]
    prev = {
        "prev_pos": prev_pos,
        "prev_msa_first_row": rng.normal(size=pspec["prev_msa_first_row"].shape),
        "prev_pair": rng.normal(size=pspec["prev_pair"].shape),
    }
    if coh:
        prev = {k: np.zeros_like(v) for k, v in prev.items()}
    prev = {k: jnp.asarray(v, float_dtype(cfg)) for k, v in prev.items()}
    return batch, prev


def example_inputs(cfg: Config, rng):
    seed = int(jax.random.randint(rng, (), 0, 2 ** 30))
    return {PHASE: synthetic_features(cfg, seed)}


# ----------------------------------------------------------------------------
# Forward functions
# ----------------------------------------------------------------------------

def with_seq_length(batch, cfg: Config):
    """Rebuild the two features the network reads but that never reach an output,
    so they are not dead graph inputs:
      * `seq_length`: only its shape (the ensemble dim) is read;
      * `atom14_atom_exists`: only fills StructureModule's `final_atom14_mask`,
        which is dropped when compute_loss=False. Derived from aatype exactly as
        the data pipeline's make_atom14_masks does."""
    b = dict(batch)
    ensemble, num_res = batch["aatype"].shape          # (from the inputs: N is a symbol)
    b["seq_length"] = jnp.full((ensemble,), num_res, jnp.int32)
    b["atom14_atom_exists"] = jnp.asarray(_atom_tables()[2])[batch["aatype"]]
    return b


_RNG = lambda: jax.random.PRNGKey(0)   # unused: eval mode, no dropout


def _module_scope(next_f, args, kwargs, context):
    """hk interceptor: run every module method under a name stack equal to the
    module's full haiku name (== its parameter path). Haiku's own named_call
    only pushes the *local* name onto the *call* nesting, which differs from
    module ownership (e.g. evoformer_iteration is created outside layer_stack
    but called inside it; Transition's linears are called inside
    hk.Sequential). Metadata only."""
    name = getattr(context.module, "module_name", None)
    if name is None:                    # __init__ runs before the name exists
        return next_f(*args, **kwargs)
    with source_info_util.reset_name_stack(), jax.named_scope(name):
        return next_f(*args, **kwargs)


def _clean_name_stack(ns):
    """Keep plain scopes; drop transform wrappers (vmap(..)) and `eval_jaxpr`."""
    return source_info_util.NameStack(tuple(
        e for e in ns.stack
        if isinstance(e, source_info_util.Scope) and e.name != "eval_jaxpr"))


def _is_absolute(ns) -> bool:
    return bool(ns.stack) and ns.stack[0].name.split("/")[0] == ROOT_MODULE


def _clean_param(v):
    if isinstance(v, jax_core.Jaxpr):          # (JAX 0.12 merged ClosedJaxpr into Jaxpr)
        return _clean_jaxpr(v)
    if isinstance(v, tuple) and any(isinstance(x, jax_core.Jaxpr) for x in v):
        return tuple(_clean_param(x) for x in v)
    return v


def _clean_jaxpr(jaxpr):
    eqns = [e.replace(params={k: _clean_param(v) for k, v in e.params.items()},
                      source_info=e.source_info.replace(
                          name_stack=_clean_name_stack(e.source_info.name_stack)))
            for e in jaxpr.eqns]
    return jaxpr.replace(eqns=eqns)


def _eval(jaxpr, consts, args):
    """core.eval_jaxpr, plus: inline `eval_jaxpr` call eqns and expand fully
    unrolled scans (unroll >= length) here, the same way JAX's own lowering of
    such a scan does (slice xs, run the body, stack ys), so neither leaves a
    haiku-irrelevant `eval_jaxpr` scope on the ops."""
    env = {}
    read = lambda v: v.val if isinstance(v, jax_core.Literal) else env[v]
    for v, x in zip(jaxpr.all_invars, [*consts, *args]):
        env[v] = x
    for eqn in jaxpr.eqns:
        invals = [read(v) for v in eqn.invars]
        ns = _clean_name_stack(eqn.source_info.name_stack)
        if not _is_absolute(ns):
            ns = source_info_util.current_name_stack() + ns
        p = eqn.params
        with source_info_util.user_context(eqn.source_info.traceback, name_stack=ns), \
                eqn.ctx.manager:
            if eqn.primitive is jax_core.eval_jaxpr_p:
                outs = _eval(p["call_jaxpr"], p["call_jaxpr"].consts, invals)
            elif eqn.primitive.name == "scan" and not jax.export.is_symbolic_dim(
                    p["length"]) and (p["unroll"] is True or p["unroll"] >= p["length"]):
                # == jax's _scan_impl with unroll == length (straight-line).
                cs, carry, xs = map(list, p["ft_in"].update(invals).unpack())
                body, ys = p["jaxpr"], []
                for i_ in range(p["length"]):
                    i = p["length"] - i_ - 1 if p["reverse"] else i_
                    x = [jax.lax.index_in_dim(v, i, keepdims=False) for v in xs]
                    carry_y = _eval(body, body.consts, [*cs, *carry, *x])
                    carry, y = map(list, p["ft_out"].update(carry_y).unpack())
                    ys.append(y)
                ys = list(reversed(ys)) if p["reverse"] else ys
                outs = [*carry, *map(_scan_stack, zip(*ys))]
            else:
                outs = eqn.primitive.bind(
                    *invals, **eqn.primitive.get_bind_params(
                        {k: _clean_param(v) for k, v in p.items()}))
        if not eqn.primitive.multiple_results:
            outs = [outs]
        for v, x in zip(eqn.outvars, outs):
            env[v] = x
    return [read(v) for v in jaxpr.outvars]


def dce(fn):
    """Trace `fn` to a jaxpr, drop dead equations, re-emit the rest.

    capture() lowers with keep_unused=True, which also skips JAX's equation
    DCE, so code whose results reach no output would stay in the graph (XLA
    deletes it at compile time anyway). In AlphaFold that is (a) RNG key
    splitting (haiku rng sequence, SafeKey.split, layer_stack's per-layer
    split; threefry lowers to a while loop on CPU) -- only consumed by dropout,
    which is off -- and (b) the template frames' quaternion `rot_to_quat` (an
    `eigh` LAPACK custom_call): the QuatAffine is built with an explicit
    rotation, so the quaternion is never read. Name stacks are set to haiku
    module paths (see _module_scope, _clean_name_stack, _eval).
    """
    def wrapped(*args):
        closed, out_shape = jax.make_jaxpr(fn, return_shape=True)(*args)
        jaxpr, _ = pe.dce_jaxpr(closed.jaxpr, [True] * len(closed.jaxpr.outvars),
                                instantiate=True)
        out = _eval(jaxpr, closed.consts, jax.tree.leaves(args))
        return jax.tree.unflatten(jax.tree.structure(out_shape), out)
    return wrapped


ROOT_MODULE = "alphafold"


def make_forward(cfg: Config, make_transformed, ctx=None):
    """Phase fn. The (patched) config and haiku transform are built inside the
    traced function."""
    def forward(params, batch, prev):
        with (ctx() if ctx else _null()), hk.intercept_methods(_module_scope):
            return make_transformed(cfg).apply(params, _RNG(), with_seq_length(batch, cfg), prev)
    return dce(forward)


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def patched_transformed(cfg: Config):
    return net.make_fn(net.patched_config(cfg))


def init_variables(cfg: Config, rng):
    batch, prev = synthetic_features(cfg, 0)
    with net.unrolled():
        return patched_transformed(cfg).init(rng, with_seq_length(batch, cfg), prev)


# Block-internal modules, not recorded in the (full-size) jit mode: the
# canonical model runs them inside lax.scan, so they have no outer-level value,
# and the unrolled patched model would return ~100 GB of them.
_BLOCK_INTERNAL = re.compile(r"/(evoformer_iteration|extra_msa_stack|single_template_embedding)(/|$)")


def record_module_outputs(cfg: Config, transformed, params, batch, prev, ctx=None) -> dict:
    """{"<haiku module>#<call k><leaf path>:<shape>": array} for module __call__
    outputs, in execution order.

    cfg.eager_activations (reduced): run op-by-op under `jax.disable_jit()`, so
    `lax.scan` bodies (layer stacks, sharded_apply chunks) execute as Python
    loops and EVERY call is concrete and recorded -- per Evoformer block,
    per chunk. Calls under hk.vmap / hk.eval_shape (tracers) are skipped.

    Otherwise (full size): one jitted run returning the outputs of every
    module call made at the outermost trace level (not inside a scan / vmap),
    except block internals (_BLOCK_INTERNAL).
    """
    counts, order = Counter(), []

    def keep(x, outer):
        if not (isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jnp.floating)):
            return False
        if outer is None:
            return not isinstance(x, jax_core.Tracer)
        return isinstance(x, jax_core.Tracer) and x._trace is outer

    def run(params, batch, prev, outer=None):
        acts = {}

        def interceptor(next_f, args, kwargs, context):
            out = next_f(*args, **kwargs)
            if context.method_name != "__call__":
                return out
            name = context.module.module_name
            if outer is not None and _BLOCK_INTERNAL.search(name):
                return out
            leaves = [(p, x) for p, x in jax.tree_util.tree_flatten_with_path(out)[0]
                      if keep(x, outer)]
            if not leaves:
                return out
            k = counts[name]
            counts[name] += 1
            for p, x in leaves:
                shape = "x".join(map(str, x.shape))
                key = f"{name}#{k}{jax.tree_util.keystr(p)}:{shape}"
                acts[key] = x
                order.append(key)
            return out

        with hk.intercept_methods(interceptor), (ctx() if ctx else _null()):
            transformed.apply(params, _RNG(), with_seq_length(batch, cfg), prev)
        return acts

    if cfg.eager_activations:
        with jax.disable_jit():
            acts = run(params, batch, prev)
    else:
        acts = jax.jit(lambda *a: run(*a, outer=jax_core.trace_ctx.trace))(params, batch, prev)
    return {k: np.asarray(acts[k]) for k in order}


def intermediates(cfg: Config, variables, batch, prev) -> dict:
    return record_module_outputs(cfg, patched_transformed(cfg), variables, batch, prev,
                                 ctx=net.unrolled)


# ----------------------------------------------------------------------------
# Spec
# ----------------------------------------------------------------------------

def build(cfg: Config) -> ModelSpec:
    float_dtype(cfg)
    forward = make_forward(cfg, patched_transformed, ctx=net.unrolled)
    batch, prev = input_spec(cfg)
    params = jax.eval_shape(lambda: init_variables(cfg, jax.random.PRNGKey(0)))
    e = net.model_config(cfg)[_E]
    return ModelSpec(
        name="alphafold2",
        variant=(f"AlphaFold 2 monomer {net.MODEL_NAME}, FP8 E4M3 W8A8 per-tensor on a bf16 "
                 f"trunk (GPU fallback), fp32 structure module + heads, N_res={cfg.num_res}, "
                 f"N_msa={cfg.num_msa}+{cfg.num_templates} template rows, "
                 f"N_extra={cfg.num_extra_msa}, N_templ={cfg.num_templates}, "
                 f"num_ensemble={cfg.num_ensemble}, Evoformer x{e.evoformer_num_block}, "
                 f"extra-MSA x{e.extra_msa_stack_num_block}"),
        phases=[Phase(PHASE, forward, (batch, prev), trip_count=cfg.num_recycle + 1,
                      shapes=INPUT_SHAPES,
                      note="one recycling iteration (AlphaFoldIteration); modules.AlphaFold "
                           "runs it num_recycle + 1 times, feeding prev_* back")],
        param_specs=as_spec(params),
        symbols={"N": Sym("num_res", 24, "residues (sequence length)"),
                 "S": Sym("num_msa", 10, "MSA cluster rows (without the template rows)"),
                 "E": Sym("num_extra_msa", 14, "extra-MSA rows")},
        excluded=[
            "data pipeline: HMMER/HHblits/template search, TF feature processing "
            "(MSA clustering, masking, cropping, make_atom14_masks)",
            "recycling while_loop of modules.AlphaFold (trip count = num_recycle + 1) and "
            "its per-recycle slicing of the resampled MSA features",
            "losses / metrics (compute_loss=False)",
            "post-processing: pLDDT / PAE / pTM from logits, relaxation (Amber)",
        ],
    )
