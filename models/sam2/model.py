"""SAM 2.1 Hiera-L, semi-supervised video object segmentation with the SAM2 video
predictor: capture spec. Networks live in net.py.

Scenario: a 1024^2 video of `num_frames` frames, one tracked object (batch 1), one
positive click on frame 0 (SAM2VideoPredictor: init_state, add_new_points_or_box on
frame 0, propagate_in_video). Each network invocation is a phase:

  image_encoder        SAM2Base.forward_image, every frame                       x N
  sam_heads_cond       frame 0: + no_mem_embed, SAM heads with the click          x 1
  memory_encoder_cond  frame 0 (propagate_in_video_preflight): mask -> memory     x 1
  memory_attention_fNN frames 1..15: memory bank still filling (one phase per
                       distinct bank size)                                        x 1 each
  memory_attention     frames 16..N-1: full bank, 7 spatial memories + 16 ptrs   x N-16
  sam_heads            frames 1..N-1: SAM heads, no prompt                       x N-1
  memory_encoder       frames 1..N-1: predicted mask -> memory                   x N-1

Parameters are one tree with a top-level key per consumer group (`Phase.param_keys`):
image_encoder, cond_frame, sam_heads, memory_conditioning, memory_encoding.

Quantization: INT8 W8A8 edge fallback (models/sam2/recipe.py, net.py docstring): every conv
and Linear int8 (per-channel weights, per-tensor static activations), attention cores and
norms fp16, position encodings fp32. Tensors passed between phases are fp16: vision
features / positional encodings, and the memory bank (maskmem_features, maskmem_pos_enc,
object pointers), which the host stores fp16 (the predictor's bf16 storage on an fp16
target). The frame, click coordinates, mask logits and scores stay fp32.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from models.spec import ModelSpec, Phase, Sym, as_spec
from models.sam2 import net
from models.sam2.net import Config


def _with_bank(cfg: Config) -> Config:
    """`cfg` with the steady-state bank sizes (the `NM` / `NP` symbols' fields) filled in."""
    max_ptrs = min(cfg.num_frames, cfg.max_obj_ptrs_in_encoder)
    return dataclasses.replace(cfg, bank_mems=min(cfg.num_frames, cfg.num_maskmem),
                               bank_ptrs=max_ptrs)


CONFIGS = {
    "full": _with_bank(Config()),
    # Same structure (4 Hiera stages with q-pooling, windowed + global blocks, bicubic
    # window pos-embed 7 -> 64, FPN with top-down levels, 4-token object pointers,
    # 7 spatial memories, 16 pointers), small sizes.
    "reduced": _with_bank(Config(embed_dim=16, num_heads=1, stages=(1, 2, 4, 2),
                                 global_att_blocks=(5, 6), window_spec=(8, 4, 8, 4), d_model=64,
                                 mem_attn_layers=2, mem_attn_ffn=128, mem_dim=16, image_size=256)),
}

GROUPS = {"image_encoder": net.SAM2ImageEncoder, "cond_frame": net.SAM2CondFrame,
          "sam_heads": net.SAM2Heads, "memory_conditioning": net.SAM2MemoryConditioning,
          "memory_encoding": net.SAM2MemoryEncoding}


# ----------------------------------------------------------------------------
# scenario: memory bank sizes per frame
# ----------------------------------------------------------------------------

def bank_size(cfg: Config, frame_idx: int) -> tuple[int, int]:
    """(spatial memories, object pointers) attended on tracked frame `frame_idx` >= 1:
    the conditioning frame 0 plus the previous non-conditioning frames 1..frame_idx-1,
    up to num_maskmem - 1 and min(num_frames, max_obj_ptrs_in_encoder) - 1 of them."""
    max_ptrs = min(cfg.num_frames, cfg.max_obj_ptrs_in_encoder)
    return min(frame_idx, cfg.num_maskmem), min(frame_idx, max_ptrs)


def memory_phases(cfg: Config) -> list[tuple[str, int, int, int, int]]:
    """[(phase, first frame, trip count, n_mem, n_ptr)]: one phase per warm-up bank size,
    then the steady state."""
    full = bank_size(cfg, cfg.num_frames)
    out, t = [], 1
    while bank_size(cfg, t) != full:
        out.append((f"memory_attention_f{t:02d}", t, 1) + bank_size(cfg, t))
        t += 1
    steady = (full[0] if cfg.bank_mems is None else cfg.bank_mems,
              full[1] if cfg.bank_ptrs is None else cfg.bank_ptrs)
    out.append(("memory_attention", t, cfg.num_frames - t) + steady)
    return out


# ----------------------------------------------------------------------------
# shapes / params
# ----------------------------------------------------------------------------

def input_specs(cfg: Config) -> dict:
    S, f32, f16 = jax.ShapeDtypeStruct, jnp.float32, jnp.float16
    B, C, M, f, I = cfg.batch, cfg.d_model, cfg.mem_dim, cfg.feat_size, cfg.image_size
    feat = S((B, C, f, f), f16)                     # image-encoder outputs are fp16
    s0, s1 = S((B, C // 8, 4 * f, 4 * f), f16), S((B, C // 4, 2 * f, 2 * f), f16)
    score = S((B, 1), f32)
    out = {
        "image_encoder": (S((B, 3, I, I), f32),),
        "sam_heads_cond": (feat, s0, s1, S((B, 1, 2), f32), S((B, 1), jnp.int32)),
        "memory_encoder_cond": (feat, S((B, 1, 4 * f, 4 * f), f32), score),
        "sam_heads": (feat, s0, s1),
        "memory_encoder": (feat, S((B, 1, I, I), f32), score),
    }
    for name, _, _, n_mem, n_ptr in memory_phases(cfg):
        # memory bank stored fp16: maskmem_features, maskmem_pos_enc, object pointers
        out[name] = (feat, feat, S((n_mem, B, M, f, f), f16), S((B, M, f, f), f16),
                     S((n_ptr, B, C), f16), S((), jnp.int32))
    return out


def _module(cfg: Config, group: str, phase: str | None = None):
    if group == "sam_heads":
        return net.SAM2Heads(cfg, num_points=1 if phase == "sam_heads_cond" else 0)
    if group == "memory_encoding":
        return net.SAM2MemoryEncoding(cfg, cond=phase == "memory_encoder_cond")
    return GROUPS[group](cfg)


PHASE_GROUPS = {"image_encoder": ("image_encoder",), "sam_heads_cond": ("cond_frame", "sam_heads"),
                "memory_encoder_cond": ("memory_encoding",), "sam_heads": ("sam_heads",),
                "memory_encoder": ("memory_encoding",)}


def phase_groups(phase: str) -> tuple[str, ...]:
    return PHASE_GROUPS.get(phase, ("memory_conditioning",))


def init_variables(cfg: Config, rng):
    specs = input_specs(cfg)
    zeros = lambda ph: [jnp.zeros(s.shape, s.dtype) for s in specs[ph]]
    k = jax.random.split(rng, 5)
    steady = memory_phases(cfg)[-1][0]
    feat = zeros("sam_heads")[0]
    return {
        "image_encoder": _module(cfg, "image_encoder").init(k[0], *zeros("image_encoder"))["params"],
        "cond_frame": _module(cfg, "cond_frame").init(k[1], feat)["params"],
        "sam_heads": _module(cfg, "sam_heads", "sam_heads").init(k[2], *zeros("sam_heads"))["params"],
        "memory_conditioning": _module(cfg, "memory_conditioning", steady).init(
            k[3], *zeros(steady))["params"],
        "memory_encoding": _module(cfg, "memory_encoding", "memory_encoder").init(
            k[4], *zeros("memory_encoder"))["params"],
    }


# ----------------------------------------------------------------------------
# phases
# ----------------------------------------------------------------------------

def _run(cfg: Config, phase: str, params: dict, *inputs, capture: bool = False):
    """Returns (outputs, {"<group>/<module path>": activation} or None)."""
    kw = dict(capture_intermediates=True, mutable=["intermediates"]) if capture else {}
    acts = {}

    def apply(group, *args):
        out = _module(cfg, group, phase).apply({"params": params[group]}, *args, **kw)
        if capture:
            out, st = out
            acts.update(_flatten(group, st["intermediates"]))
        return out

    if phase == "sam_heads_cond":
        feat, s0, s1, coords, labels = inputs
        out = apply("sam_heads", apply("cond_frame", feat), s0, s1, coords, labels)
    else:
        (group,) = phase_groups(phase)
        out = apply(group, *inputs)
    return out, (acts if capture else None)


def _flatten(group: str, inter) -> dict:
    flat = {}
    for path, v in jax.tree_util.tree_flatten_with_path(
            inter, is_leaf=lambda t: isinstance(t, tuple))[0]:
        keys = [k.key for k in path if hasattr(k, "key")]
        if not keys or keys[-1] != "__call__":
            continue
        out = v[0]                               # first call (see PromptEncoder.pe_layer)
        while isinstance(out, (tuple, list)):    # (hidden, ...) -> hidden
            out = out[0]
        flat["/".join([group] + keys[:-1])] = out
    return flat


def phase_fn(cfg: Config, phase: str):
    def fn(params, *inputs):
        return _run(cfg, phase, params, *inputs)[0]
    fn.__name__ = phase
    return fn


def example_inputs(cfg: Config, rng):
    specs = input_specs(cfg)
    keys = iter(jax.random.split(rng, 256))
    rnd = lambda s, scale=1.0: (scale * jax.random.normal(next(keys), s.shape, jnp.float32)).astype(s.dtype)
    out = {}
    for ph, sp in specs.items():
        if ph == "sam_heads_cond":
            click = jax.random.uniform(next(keys), sp[3].shape, jnp.float32, 0, cfg.image_size)
            out[ph] = (rnd(sp[0]), rnd(sp[1]), rnd(sp[2]), click, jnp.ones(sp[4].shape, jnp.int32))
        elif ph in ("memory_encoder", "memory_encoder_cond"):
            out[ph] = (rnd(sp[0]), rnd(sp[1], 4.0), rnd(sp[2], 4.0))     # mask / score logits
        elif ph.startswith("memory_attention"):
            first, trips = next((m[1], m[2]) for m in memory_phases(cfg) if m[0] == ph)
            frame_idx = first + trips // 2                               # a frame of this phase
            out[ph] = tuple(rnd(s) for s in sp[:5]) + (jnp.asarray(frame_idx, jnp.int32),)
        else:
            out[ph] = tuple(rnd(s) for s in sp)
    return out


def intermediates(cfg: Config, variables, *inputs, phase: str) -> dict:
    return _run(cfg, phase, variables, *inputs, capture=True)[1]


# ----------------------------------------------------------------------------
# spec
# ----------------------------------------------------------------------------

def build(cfg: Config) -> ModelSpec:
    variables = jax.eval_shape(lambda: init_variables(cfg, jax.random.key(0)))
    specs = input_specs(cfg)
    N = cfg.num_frames
    mph = memory_phases(cfg)
    # Symbolic sizes: B tracked objects (every phase), NM / NP the memory-bank sizes. The frame
    # size is SAM2Base.image_size (frames are resized to it; _forward_sam_heads asserts the
    # stride-16 map is sam_image_embedding_size), the click count is 1 (_use_multimask: the
    # multimask head serves 0..1 clicks), so those axes are architecture constants.
    feat = "B, _, _, _"
    shapes = {
        "image_encoder": (feat,),
        "sam_heads_cond": (feat, feat, feat, "B, _, _", "B, _"),
        "memory_encoder_cond": (feat, feat, "B, _"),
        "sam_heads": (feat, feat, feat),
        "memory_encoder": (feat, feat, "B, _"),
    }
    for name, *_ in mph:
        shapes[name] = (feat, feat, "NM, B, _, _, _", feat, "NP, B, _", "")
    steady = mph[-1][0]
    ph = lambda name, trips, note: Phase(name, phase_fn(cfg, name), specs[name], trip_count=trips,
                                         param_keys=phase_groups(name), note=note,
                                         shapes=shapes[name],
                                         size_variant_of=steady if name.startswith("memory_attention_f")
                                         else None)
    phases = [
        ph("image_encoder", N, "SAM2Base.forward_image on one normalized frame: Hiera-L trunk, "
           "FPN neck (scalp 1), conv_s0/conv_s1 -> (backbone_fpn[0..2], vision_pos_enc[0..2]); "
           "every frame once (frame 0 in init_state, frames 1.. in propagate_in_video)"),
        ph("sam_heads_cond", 1, "frame 0 (add_new_points_or_box): vision_feats[-1] + no_mem_embed, "
           "then _forward_sam_heads with the click (+ padding point), multimask output; returns "
           "(low_res_multimasks, high_res_multimasks, ious, low_res_masks, high_res_masks, obj_ptr, "
           "object_score_logits)"),
        ph("memory_encoder_cond", 1, "frame 0 (propagate_in_video_preflight): pred_masks bilinear to "
           "image size, binarized (is_mask_from_pts), memory encoder -> (maskmem_features, "
           "maskmem_pos_enc)"),
    ]
    for name, first, trips, n_mem, n_ptr in mph:
        frames = f"frame {first}" if trips == 1 else f"frames {first}..{first + trips - 1}"
        phases.append(ph(name, trips,
                         f"{frames}: _prepare_memory_conditioned_features with {n_mem} spatial memories "
                         f"({n_mem} x {cfg.feat_size}^2 tokens) + {n_ptr} object pointers "
                         f"({n_ptr * cfg.d_model // cfg.mem_dim} tokens); inputs: fp16 vision feat/pos, "
                         "fp16 memory bank: maskmem_features [cond, oldest..newest], maskmem_pos_enc, obj_ptrs "
                         "[cond, 1 back, 2 back, ...], frame_idx (int32, the cond pointer's temporal "
                         "offset) -> pix_feat_with_mem"))
    phases += [
        ph("sam_heads", N - 1, "frames 1..N-1: _forward_sam_heads(pix_feat_with_mem, no point "
           "(one label -1 point + padding), multimask_output_for_tracking) -> 7-tuple as sam_heads_cond"),
        ph("memory_encoder", N - 1, "frames 1..N-1: _encode_new_memory(high_res_masks, sigmoid, "
           "x20 - 10, no-object embedding) -> (maskmem_features, maskmem_pos_enc)"),
    ]
    return ModelSpec(
        name="sam2",
        variant=(f"SAM 2.1 Hiera-L video predictor, {cfg.image_size}^2, {N} frames, 1 object, "
                 f"1 click on frame 0, memory bank {cfg.num_maskmem} frames + "
                 f"{min(N, cfg.max_obj_ptrs_in_encoder)} object pointers; INT8 W8A8 edge fallback "
                 "(int8 per-channel weights, per-tensor static activations on every conv / Linear; "
                 "attention cores and norms fp16; fp16 memory bank)"),
        phases=phases,
        param_specs=as_spec(variables),
        symbols={"B": Sym("batch", 2, "tracked objects (batched prompts) per call"),
                 "NM": Sym("bank_mems", 4, "spatial memories in the bank (conditioning frame + "
                                           "previous frames)"),
                 "NP": Sym("bank_ptrs", 9, "object pointers in the bank (conditioning frame + "
                                           "previous frames)")},
        # maskmem_tpos_enc has num_maskmem rows; at most max_obj_ptrs_in_encoder pointers
        constraints=(f"NM <= {cfg.num_maskmem}", f"NP <= {cfg.max_obj_ptrs_in_encoder}"),
        excluded=[
            "video decoding, resize to 1024^2 and ImageNet normalization (load_video_frames; host)",
            "click coordinate normalization to the 1024 grid (add_new_points_or_box; host)",
            "memory bank bookkeeping: frame selection, fp16 storage of maskmem_features / "
            "maskmem_pos_enc / obj_ptrs, stacking past obj_ptrs (output_dict / inference_state; "
            "host tensor glue)",
            "fill_holes_in_mask_scores (fill_hole_area=8: connected components via the sam2._C "
            "CUDA extension, data-dependent) on pred_masks, before storage and before frame 0's "
            "memory encoding",
            "resize of pred_masks to the original video resolution and thresholding (postprocess)",
        ],
    )

