"""SDXL base 1.0 (MLPerf Inference text-to-image) capture spec; networks live in net.py.

Quantized as NVIDIA's MLPerf Hopper TensorRT engines (net.py docstring, recipe.py):
FP16 CLIP-L / CLIP-bigG, FP8 UNet (W8A8 + FP8 MHA, fp16 elsewhere, fp16 I/O), INT8 VAE
convs in an fp32 / fp16 decoder (fp32 I/O).

Parameters are one tree with a top-level key per component
({"text_encoder_l", "text_encoder_g", "unet", "vae"}); each phase consumes one
key via `Phase.param_keys`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from models.spec import ModelSpec, Phase, Sym, as_spec
from models.sdxl import net
from models.sdxl.net import (CLIP_G, CLIP_L, CLIPTextConfig, FlaxAutoencoderKL,
                             FlaxCLIPTextModelWithProjectionModule, FlaxCLIPTextModule,
                             FlaxUNet2DConditionModel, UNetConfig, VAEConfig)

COMPONENTS = ("text_encoder_l", "text_encoder_g", "unet", "vae")


@dataclass(frozen=True)
class Config:
    clip_l: CLIPTextConfig = CLIP_L
    clip_g: CLIPTextConfig = CLIP_G
    unet: UNetConfig = UNetConfig()
    vae: VAEConfig = VAEConfig()
    batch: int = 1               # images per prompt
    latent: int = 128            # 1024 / 8
    steps: int = 20              # EulerDiscrete steps (MLPerf)
    dtype: str = "float16"       # CLIP / UNet engine precision (the VAE engine is fp32-based)
    # Per-module activation comparison granularity: every module (reduced) or
    # block-level modules only (full size; keeps memory bounded).
    coarse_activations: bool = False
    # UNet batch, if not the CFG batch 2 x batch: its own symbol (unet_step has no input
    # whose batch is `batch` itself).
    unet_batch: int | None = None
    # CLIP prompt tokens: both tokenizers pad to model_max_length 77 (= max_position_embeddings)
    clip_tokens: int = 77
    # unet_step's latent side in units of the UNet's total downsampling factor
    # (unet_factor), if not latent / unet_factor: its own symbols, so that the stride-2
    # down path and the 2x up path provably meet at the skip concatenations.
    unet_grid: int | None = None

    @property
    def cfg_batch(self) -> int:  # classifier-free guidance doubles the UNet batch
        return 2 * self.batch if self.unet_batch is None else self.unet_batch

    @property
    def seq(self) -> int:
        return self.clip_tokens

    @property
    def unet_factor(self) -> int:  # 2x downsample between consecutive down blocks
        return 2 ** (len(self.unet.block_out_channels) - 1)

    @property
    def unet_latent(self) -> int:
        return self.latent if self.unet_grid is None else self.unet_factor * self.unet_grid


REDUCED = Config(
    clip_l=CLIPTextConfig(vocab_size=1000, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                          num_attention_heads=2, hidden_act="quick_gelu", projection_dim=32),
    clip_g=CLIPTextConfig(vocab_size=1000, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                          num_attention_heads=4, hidden_act="gelu", projection_dim=48),
    unet=UNetConfig(sample_size=16, block_out_channels=(32, 64, 64), attention_head_dim=(1, 2, 4),
                    transformer_layers_per_block=(1, 1, 2), cross_attention_dim=96,
                    addition_time_embed_dim=8, projection_class_embeddings_input_dim=48 + 6 * 8),
    vae=VAEConfig(block_out_channels=(32, 32, 64, 64)),
    latent=16,
)

def _with_sizes(cfg: Config) -> Config:
    """`cfg` with the UNet (CFG) batch and grid (the `BU` / `HU`, `WU` symbols' fields) filled in."""
    assert cfg.latent % cfg.unet_factor == 0, (cfg.latent, cfg.unet_factor)
    return dataclasses.replace(cfg, unet_batch=2 * cfg.batch, unet_grid=cfg.latent // cfg.unet_factor)


CONFIGS = {"full": _with_sizes(Config(coarse_activations=True)), "reduced": _with_sizes(REDUCED)}


def _modules(cfg: Config) -> dict:
    dt = jnp.dtype(cfg.dtype)
    return {
        "text_encoder_l": FlaxCLIPTextModule(cfg.clip_l, dtype=dt),
        "text_encoder_g": FlaxCLIPTextModelWithProjectionModule(cfg.clip_g, dtype=dt),
        "unet": FlaxUNet2DConditionModel(cfg.unet, dtype=dt),
        "vae": FlaxAutoencoderKL(cfg.vae, dtype=jnp.float32),
    }


def input_specs(cfg: Config) -> dict:
    dt = jnp.dtype(cfg.dtype)
    S = jax.ShapeDtypeStruct
    B, L = cfg.cfg_batch, cfg.latent
    text_embeds_dim = cfg.unet.projection_class_embeddings_input_dim - 6 * cfg.unet.addition_time_embed_dim
    ids = (S((cfg.batch, cfg.seq), jnp.int32),)
    return {
        "text_encoder_l": ids,
        "text_encoder_g": ids,
        "unet_step": (S((B, cfg.unet.in_channels, cfg.unet_latent, cfg.unet_latent), dt),  # latent_model_input
                      S((), dt),                                        # timestep t
                      S((B, cfg.seq, cfg.unet.cross_attention_dim), dt),  # prompt_embeds
                      S((B, text_embeds_dim), dt),                      # added_cond text_embeds
                      S((B, 6), dt)),                                   # added_cond time_ids
        "vae_decode": (S((cfg.batch, cfg.vae.latent_channels, L, L), jnp.float32),),
    }


PHASE_COMPONENT = {"text_encoder_l": "text_encoder_l", "text_encoder_g": "text_encoder_g",
                   "unet_step": "unet", "vae_decode": "vae"}


def init_variables(cfg: Config, rng):
    mods, specs = _modules(cfg), input_specs(cfg)
    keys = jax.random.split(rng, len(COMPONENTS))
    out = {}
    for k, (ph, comp) in zip(keys, PHASE_COMPONENT.items()):
        zeros = [jnp.zeros(s.shape, s.dtype) for s in specs[ph]]
        out[comp] = mods[comp].init(k, *zeros)["params"]
    return out


def _token_ids(cfg: Config, clip: CLIPTextConfig, rng, pad_with_eos: bool):
    """[BOS, prompt..., EOS, pad...] as the SDXL CLIP tokenizers produce (77 tokens)."""
    bos, eos = clip.vocab_size - 2, clip.vocab_size - 1
    n = 12
    toks = np.asarray(jax.random.randint(rng, (cfg.batch, n), 1, clip.vocab_size - 2), np.int32)
    ids = np.full((cfg.batch, cfg.seq), eos if pad_with_eos else 0, np.int32)
    ids[:, 0], ids[:, 1:1 + n], ids[:, 1 + n] = bos, toks, eos
    return jnp.asarray(ids)


def example_inputs(cfg: Config, rng):
    specs = input_specs(cfg)
    r = jax.random.split(rng, 8)
    lat, _, ehs, temb, _ = specs["unet_step"]
    size = 8 * cfg.latent
    # micro-conditioning: (orig_h, orig_w, crop_top, crop_left, target_h, target_w)
    time_ids = jnp.tile(jnp.asarray([[size, size, 0, 0, size, size]], lat.dtype), (cfg.cfg_batch, 1))
    return {
        # tokenizer (CLIP-L) pads with EOS, tokenizer_2 (bigG) pads with "!" (id 0)
        "text_encoder_l": (_token_ids(cfg, cfg.clip_l, r[0], pad_with_eos=True),),
        "text_encoder_g": (_token_ids(cfg, cfg.clip_g, r[1], pad_with_eos=False),),
        "unet_step": (jax.random.normal(r[2], lat.shape, lat.dtype),
                      jnp.asarray(999.0, lat.dtype),          # first EulerDiscrete timestep
                      jax.random.normal(r[3], ehs.shape, ehs.dtype),
                      jax.random.normal(r[4], temb.shape, temb.dtype),
                      time_ids),
        "vae_decode": (jax.random.normal(r[5], specs["vae_decode"][0].shape,
                                         specs["vae_decode"][0].dtype),),
    }


def _phase_fn(cfg: Config, comp: str):
    module = _modules(cfg)[comp]

    def fn(params, *inputs):
        return module.apply({"params": params[comp]}, *inputs)
    fn.__name__ = comp
    return fn


# Block-level modules compared at full size (coarse_activations).
COARSE_TYPES = (net.FlaxCLIPEncoderLayer, net.FlaxResnetBlock2D, net.FlaxTransformer2DModel,
                net.FlaxBasicTransformerBlock, net.FlaxVAEResnetBlock2D, net.FlaxAttentionBlock,
                net.FlaxTimestepEmbedding)


def intermediates(cfg: Config, variables, *inputs, phase: str) -> dict:
    """Per-module outputs of the phase's network, keyed "<component>/<flax module path>"."""
    mods, flat = _modules(cfg), {}
    if cfg.coarse_activations:
        filt = lambda mdl, _name: isinstance(mdl, COARSE_TYPES)
    else:
        filt = True
    for comp in (PHASE_COMPONENT[phase],):
        _, state = mods[comp].apply({"params": variables[comp]}, *inputs,
                                    capture_intermediates=filt, mutable=["intermediates"])
        for path, v in jax.tree_util.tree_flatten_with_path(
                state["intermediates"], is_leaf=lambda t: isinstance(t, tuple))[0]:
            keys = [k.key for k in path if hasattr(k, "key")]
            if not keys or keys[-1] != "__call__":
                continue
            name = "/".join([comp] + keys[:-1])
            out = v[0]
            while isinstance(out, tuple):        # (hidden, ...) -> hidden
                out = out[0]
            flat[name] = out
    return flat


def build(cfg: Config) -> ModelSpec:
    variables = jax.eval_shape(lambda: init_variables(cfg, jax.random.key(0)))
    specs = input_specs(cfg)
    phases = [
        Phase("text_encoder_l", _phase_fn(cfg, "text_encoder_l"), specs["text_encoder_l"],
              trip_count=2, param_keys=("text_encoder_l",), shapes=("B, S",),
              note="CLIPTextModel; prompt + negative prompt = 2 calls/image. "
                   "Outputs (hidden_states[-2], pooler_output)."),
        Phase("text_encoder_g", _phase_fn(cfg, "text_encoder_g"), specs["text_encoder_g"],
              trip_count=2, param_keys=("text_encoder_g",), shapes=("B, S",),
              note="CLIPTextModelWithProjection; prompt + negative prompt = 2 calls/image. "
                   "Outputs (hidden_states[-2], text_embeds)."),
        Phase("unet_step", _phase_fn(cfg, "unet"), specs["unet_step"], trip_count=cfg.steps,
              param_keys=("unet",),
              shapes=(f"BU, _, {cfg.unet_factor}*HU, {cfg.unet_factor}*WU", "", "BU, S, _", "BU, _", "BU, _"),
              note="One UNet call on the CFG batch (uncond, cond); returns noise_pred. "
                   "CFG combine + Euler update are host scheduler math (excluded)."),
        Phase("vae_decode", _phase_fn(cfg, "vae"), specs["vae_decode"], trip_count=1,
              param_keys=("vae",), shapes=("B, _, H, W",), note="AutoencoderKL.decode(latents / scaling_factor).sample"),
    ]
    return ModelSpec(
        name="sdxl",
        variant=f"SDXL base 1.0, {8 * cfg.latent}^2 (latent {cfg.latent}^2), {cfg.steps} Euler steps, "
                f"CFG batch {cfg.cfg_batch}; MLPerf NVIDIA Hopper TensorRT: FP16 CLIP, FP8 UNet "
                f"(W8A8 + FP8 MHA), INT8 VAE convs (fp32/fp16 decoder)",
        phases=phases,
        param_specs=as_spec(variables),
        symbols={"B": Sym("batch", 2, "images per call (prompts x images per prompt)"),
                 "BU": Sym("unet_batch", 6, "UNet batch: the CFG batch (uncond, cond) x images"),
                 "S": Sym("clip_tokens", 40, "CLIP prompt tokens (the tokenizers pad to 77)"),
                 "H": Sym("latent", 12, "latent height (image height / 8)"),
                 "W": Sym("latent", 12, "latent width (image width / 8)"),
                 "HU": Sym("unet_grid", 3, f"unet_step latent height / {cfg.unet_factor} (the UNet's "
                                           "downsampling factor)"),
                 "WU": Sym("unet_grid", 3, f"unet_step latent width / {cfg.unet_factor}")},
        # CLIP's learned position tables; the CFG batch is (uncond, cond) pairs
        constraints=(f"S <= {cfg.clip_l.max_position_embeddings}", "mod(BU, 2) == 0"),
        excluded=[
            "CLIP tokenization (both tokenizers; host)",
            "prompt-embedding concat [L(768) | G(1280)] -> 2048, negative/positive stacking, "
            "time_ids construction (host tensor glue in StableDiffusionXLPipeline)",
            "EulerDiscreteScheduler: sigma schedule, scale_model_input (latents / sqrt(sigma^2+1)), "
            "CFG combine uncond + 8.0*(cond-uncond), Euler step (host scheduler math; 20x)",
            "latent init noise; latents / vae.scaling_factor (0.13025) before decode",
            "image postprocess (denormalize to [0,1]); no refiner, no watermark, no safety checker",
        ],
    )
