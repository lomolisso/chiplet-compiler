"""FLUX.1 [schnell] text-to-image (1024x1024, 4 steps) capture spec; networks live in net.py.

Quantized as BFL's official FP8 TensorRT export (FLUX.1-schnell-onnx transformer.opt/fp8):
FP8 E4M3 per-tensor static W8A8 on the 494 block linears and FP8 attention in all 57
blocks of transformer_step; CLIP, T5 and the VAE run bf16 (see net.py, recipe.py).

Parameters are one tree with a top-level key per FluxPipeline component
({"text_encoder", "text_encoder_2", "transformer", "vae"}); each phase consumes
one key via `Phase.param_keys`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from models.spec import ModelSpec, Phase, Sym, as_spec
from models.flux import net
from models.flux.net import (AutoencoderKL, CLIPTextConfig, CLIPTextModel, FluxConfig,
                             FluxTransformer2DModel, T5Config, T5EncoderModel, VAEConfig)

COMPONENTS = ("text_encoder", "text_encoder_2", "transformer", "vae")
PHASE_COMPONENT = {"clip_text_encoder": "text_encoder", "t5_text_encoder": "text_encoder_2",
                   "transformer_step": "transformer", "vae_decode": "vae"}


@dataclass(frozen=True)
class Config:
    clip: CLIPTextConfig = CLIPTextConfig()
    t5: T5Config = T5Config()
    transformer: FluxConfig = FluxConfig()
    vae: VAEConfig = VAEConfig()
    batch: int = 1                   # num_images_per_prompt = 1, one prompt
    latent: int = 128                # 1024 / vae_scale_factor 8; packed to (128/2)^2 = 4096 tokens
    max_sequence_length: int = 256   # model card / BFL cli for schnell (FluxPipeline default 512)
    steps: int = 4                   # num_inference_steps (schnell default)
    dtype: str = "bfloat16"          # model card: FluxPipeline.from_pretrained(..., torch_dtype=bfloat16)
    # Per-module activations or block-level only (keeps memory bounded at full size).
    coarse_activations: bool = False
    # Packed image-token count of transformer_step, if not (latent/2)^2: lets the denoiser
    # be exported with the token count as its own symbol (a square is not a linear shape).
    n_image_tokens: int | None = None
    # Packed token grid (rows, cols) of vae_decode, if not (latent/2, latent/2): their own
    # symbols (FluxPipeline takes any height / width, multiples of 16).
    grid_h: int | None = None
    grid_w: int | None = None
    # CLIP prompt tokens: the tokenizer pads to model_max_length 77 (= max_position_embeddings)
    clip_tokens: int = 77

    @property
    def clip_seq(self) -> int:
        return self.clip_tokens

    @property
    def image_tokens(self) -> int:
        return (self.latent // 2) ** 2 if self.n_image_tokens is None else self.n_image_tokens

    @property
    def grid(self) -> tuple[int, int]:
        return (self.latent // 2 if self.grid_h is None else self.grid_h,
                self.latent // 2 if self.grid_w is None else self.grid_w)


def _with_sizes(cfg: Config) -> Config:
    """`cfg` with the token counts of the square scenario (the `N` / `H` / `W` symbols'
    fields) filled in."""
    g = cfg.latent // 2
    return dataclasses.replace(cfg, n_image_tokens=g * g, grid_h=g, grid_w=g)


REDUCED = Config(
    clip=CLIPTextConfig(vocab_size=1000, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                        num_attention_heads=2),
    t5=T5Config(vocab_size=100, d_model=32, d_kv=8, d_ff=64, num_layers=3, num_heads=4,
                relative_attention_num_buckets=8, relative_attention_max_distance=12),
    transformer=FluxConfig(in_channels=16, num_layers=2, num_single_layers=2, attention_head_dim=16,
                           num_attention_heads=2, joint_attention_dim=32, pooled_projection_dim=32,
                           axes_dims_rope=(4, 6, 6)),
    vae=VAEConfig(latent_channels=4, block_out_channels=(32, 32, 64, 64)),
    latent=8, max_sequence_length=16,
)

CONFIGS = {
    "full": _with_sizes(Config()),
    "reduced": _with_sizes(REDUCED),
}


def _modules(cfg: Config) -> dict:
    dt = jnp.dtype(cfg.dtype)
    return {
        "text_encoder": CLIPTextModel(cfg.clip, dtype=dt),
        "text_encoder_2": T5EncoderModel(cfg.t5, dtype=dt),
        "transformer": FluxTransformer2DModel(cfg.transformer, dtype=dt),
        "vae": AutoencoderKL(cfg.vae, dtype=dt),
    }


def input_specs(cfg: Config) -> dict:
    dt = jnp.dtype(cfg.dtype)
    S = jax.ShapeDtypeStruct
    B, N, T = cfg.batch, cfg.image_tokens, cfg.max_sequence_length
    tr = cfg.transformer
    return {
        "clip_text_encoder": (S((B, cfg.clip_seq), jnp.int32),),
        "t5_text_encoder": (S((B, T), jnp.int32),),
        "transformer_step": (S((B, N, tr.in_channels), dt),            # packed latents
                             S((B, T, tr.joint_attention_dim), dt),     # prompt_embeds (T5)
                             S((B, tr.pooled_projection_dim), dt),      # pooled_prompt_embeds (CLIP)
                             S((B,), dt),                               # timestep / 1000
                             S((N, 3), dt),                             # latent_image_ids
                             S((T, 3), dt)),                            # text_ids
        # packed latents after step 4, as their (latent/2, latent/2) token grid (a contiguous
        # host-side reshape of [B, N, 4C], so that every axis is a linear size)
        "vae_decode": (S((B, *cfg.grid, tr.in_channels), dt),),
    }


def init_variables(cfg: Config, rng):
    mods, specs = _modules(cfg), input_specs(cfg)
    keys = jax.random.split(rng, len(COMPONENTS))
    out = {}
    for k, (ph, comp) in zip(keys, PHASE_COMPONENT.items()):
        zeros = [jnp.zeros(s.shape, s.dtype) for s in specs[ph]]
        out[comp] = mods[comp].init(k, *zeros)["params"]
    return out


def latent_image_ids(cfg: Config) -> np.ndarray:
    """FluxPipeline._prepare_latent_image_ids: (0, row, col) per packed latent token."""
    h = w = cfg.latent // 2
    ids = np.zeros((h, w, 3), np.float32)
    ids[..., 1] += np.arange(h)[:, None]
    ids[..., 2] += np.arange(w)[None, :]
    return ids.reshape(h * w, 3)


def example_inputs(cfg: Config, rng):
    specs = input_specs(cfg)
    r = jax.random.split(rng, 8)
    dt = jnp.dtype(cfg.dtype)
    n = 12                                                      # prompt tokens
    # CLIPTokenizer: [BOS, prompt..., EOS] padded with EOS ("<|endoftext|>") to 77
    cv = cfg.clip.vocab_size
    bos, eos = (49406, 49407) if cv == 49408 else (cv - 2, cv - 1)
    clip = np.full((cfg.batch, cfg.clip_seq), eos, np.int32)
    clip[:, 0] = bos
    clip[:, 1:1 + n] = np.asarray(jax.random.randint(r[0], (cfg.batch, n), 1, bos), np.int32)
    # T5TokenizerFast: [prompt..., EOS(1)] padded with PAD(0) to max_sequence_length
    t5 = np.zeros((cfg.batch, cfg.max_sequence_length), np.int32)
    t5[:, :n] = np.asarray(jax.random.randint(r[1], (cfg.batch, n), 2, cfg.t5.vocab_size), np.int32)
    t5[:, n] = 1
    lat, ehs, pooled, _, img_ids, txt_ids = specs["transformer_step"]
    return {
        "clip_text_encoder": (jnp.asarray(clip),),
        "t5_text_encoder": (jnp.asarray(t5),),
        "transformer_step": (jax.random.normal(r[2], lat.shape, dt),
                             jax.random.normal(r[3], ehs.shape, dt),
                             jax.random.normal(r[4], pooled.shape, dt),
                             jnp.full((cfg.batch,), 1.0, dt),       # first step: sigma 1.0 (t 1000)
                             jnp.asarray(latent_image_ids(cfg), dt),
                             jnp.zeros(txt_ids.shape, dt)),         # text_ids are all zero
        "vae_decode": (jax.random.normal(r[5], specs["vae_decode"][0].shape, dt),),
    }


def _phase_fn(cfg: Config, comp: str):
    module = _modules(cfg)[comp]

    def fn(params, *inputs):
        return module.apply({"params": params[comp]}, *inputs)
    fn.__name__ = comp
    return fn


COARSE_TYPES = (net.CLIPEncoderLayer, net.T5Block, net.FluxTransformerBlock,
                net.FluxSingleTransformerBlock, net.CombinedTimestepTextProjEmbeddings,
                net.ResnetBlock2D, net.Attention)


def intermediates(cfg: Config, variables, *inputs, phase: str) -> dict:
    """Per-module outputs of the phase's network, keyed "<component>/<flax module path>"."""
    comp = PHASE_COMPONENT[phase]
    filt = (lambda mdl, _name: isinstance(mdl, COARSE_TYPES)) if cfg.coarse_activations else True
    _, state = _modules(cfg)[comp].apply({"params": variables[comp]}, *inputs,
                                         capture_intermediates=filt, mutable=["intermediates"])
    flat = {}
    for path, v in jax.tree_util.tree_flatten_with_path(
            state["intermediates"], is_leaf=lambda t: isinstance(t, tuple))[0]:
        keys = [k.key for k in path if hasattr(k, "key")]
        if not keys or keys[-1] != "__call__":
            continue
        out = v[0]
        while isinstance(out, tuple):        # (hidden, ...) -> hidden
            out = out[0]
        flat["/".join([comp] + keys[:-1])] = out
    return flat


def build(cfg: Config) -> ModelSpec:
    variables = jax.eval_shape(lambda: init_variables(cfg, jax.random.key(0)))
    specs = input_specs(cfg)
    T, N = cfg.max_sequence_length, cfg.image_tokens
    phases = [
        Phase("clip_text_encoder", _phase_fn(cfg, "text_encoder"), specs["clip_text_encoder"],
              trip_count=1, param_keys=("text_encoder",), shapes=("B, L",),
              note="CLIPTextModel(input_ids).pooler_output -> pooled_prompt_embeds [1,768]"),
        Phase("t5_text_encoder", _phase_fn(cfg, "text_encoder_2"), specs["t5_text_encoder"],
              trip_count=1, param_keys=("text_encoder_2",), shapes=("B, T",),
              note=f"T5EncoderModel(input_ids)[0] -> prompt_embeds [1,{T},{cfg.t5.d_model}] "
                   "(no attention mask, as FluxPipeline)"),
        Phase("transformer_step", _phase_fn(cfg, "transformer"), specs["transformer_step"],
              trip_count=cfg.steps, param_keys=("transformer",),
              shapes=("B, N, _", "B, T, _", "B, _", "B", "N, _", "T, _"),
              note=f"FluxTransformer2DModel on {N} packed image + {T} text tokens (incl. RoPE from "
                   "ids, timestep/text embedding); returns the velocity. No CFG (guidance_scale 0, "
                   "guidance_embeds=False). The flow-matching Euler update is host scheduler math "
                   "(excluded)."),
        Phase("vae_decode", _phase_fn(cfg, "vae"), specs["vae_decode"], trip_count=1,
              param_keys=("vae",), shapes=("B, H, W, _",),
              note="_unpack_latents, latents / 0.3611 + 0.1159, AutoencoderKL.decode(...).sample "
                   f"-> image [1,3,{8 * cfg.latent},{8 * cfg.latent}]"),
    ]
    return ModelSpec(
        name="flux",
        variant=f"FLUX.1 [schnell], {8 * cfg.latent}^2 (latent {cfg.vae.latent_channels}x{cfg.latent}^2, "
                f"{N} image tokens), T5 max_sequence_length {T}, {cfg.steps} steps, no CFG, "
                f"batch {cfg.batch}, {cfg.dtype}; FP8 E4M3 W8A8 transformer + FP8 MHA (BFL FP8 TensorRT "
                f"export), CLIP / T5 / VAE {cfg.dtype}",
        phases=phases,
        param_specs=as_spec(variables),
        symbols={"B": Sym("batch", 2, "images per call (prompts x num_images_per_prompt)"),
                 "L": Sym("clip_tokens", 40, "CLIP prompt tokens (the tokenizer pads to 77)"),
                 "T": Sym("max_sequence_length", 12, "T5 prompt tokens (max_sequence_length)"),
                 "N": Sym("n_image_tokens", 20, "packed image tokens: (height/16) x (width/16)"),
                 "H": Sym("grid_h", 3, "packed latent token rows (height / 16)"),
                 "W": Sym("grid_w", 5, "packed latent token columns (width / 16)")},
        # CLIP's learned position table; FluxPipeline.check_inputs: max_sequence_length <= 512
        constraints=(f"L <= {cfg.clip.max_position_embeddings}", "T <= 512"),
        excluded=[
            "CLIPTokenizer (77, padded with EOS) and T5TokenizerFast "
            f"(max_sequence_length {T}, padded with PAD) tokenization (host)",
            "text_ids = zeros(T, 3) and latent_image_ids (0,row,col) construction (host; static "
            "inputs of transformer_step)",
            "initial latent noise randn [1,16,128,128] and FluxPipeline._pack_latents (once, before "
            "the loop)",
            "FlowMatchEulerDiscreteScheduler (shift 1.0, no dynamic shifting): sigmas "
            "linspace(1, 1/4, 4) -> timesteps 1000,750,500,250, and the Euler step "
            "latents + (sigma_next - sigma) * v in fp32 (host scheduler math; 4x)",
            "image postprocess (denormalize to [0,1], PIL conversion)",
        ],
    )

