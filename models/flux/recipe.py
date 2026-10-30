"""FLUX.1 [schnell] quantization recipe (see models/quant_recipes.py for the fields)."""

F8 = "f8E4M3FNxf8E4M3FN->f32"

RECIPE = {
    "recipe": "BFL/NVIDIA FP8 TensorRT export (FLUX.1-schnell-onnx transformer.opt/fp8, ModelOpt FP8 + FP8 MHA)",
    "basis": "official",
    "sources": [
        "huggingface.co/black-forest-labs/FLUX.1-schnell-onnx @ 7ad1eace "
        "transformer.opt/fp8/model.onnx (graph read from the byte-identical mirror "
        "Futuremark/FLUX.1-schnell-onnx @ 747a18d1, sha256 9c2a08f6...): 1216 "
        "TRT_FP8QuantizeLinear/DequantizeLinear pairs, every scale a bf16 scalar; "
        "input_quantizer + weight_quantizer on all 14 linears of each double block "
        "(incl. norm1.linear, norm1_context.linear) and all 6 of each single block "
        "(incl. norm.linear); Q, K (each pre-scaled by d^-1/4), softmax output and V "
        "quantized in all 19 + 38 attentions; x_embedder, context_embedder, "
        "time_text_embed.*, norm_out.linear, proj_out and pos_embed unquantized",
        "... t5.opt/model.onnx (bf16 T5, wo bf16, softmax fp32); t5-fp8.opt/model.onnx "
        "(optional FP8 T5: blocks 1-22 linears FP8, blocks 0 and 23 and attention BMMs bf16)",
        "black-forest-labs/flux main src/flux/trt/trt_manager.py (CLIP, VAE fixed bf16; "
        "transformer bf16|fp8|fp4; T5 bf16|fp8) and src/flux/cli.py "
        "(trt_t5_precision = os.getenv('TRT_T5_PRECISION', 'bf16'))",
        "NVIDIA/TensorRT release/10.13 and release/11.0 demo/Diffusion/demo_diffusion/"
        "pipeline/flux_pipeline.py (download_onnx_models: transformer.opt/fp8 with --fp8, "
        "always t5.opt; CLIP/T5/VAE bf16 in the fp8 pipeline) and utils_modelopt.py "
        "(quantize_lvl level 4: every Linear except name == 'proj_out', FP8 MHA when "
        "head_size % 16 == 0; filter_func_no_proj_out disables time_text_embed, "
        "context_embedder, norm_out, x_embedder, pos_embed; SD_FP8_BF16 config: "
        "num_bits (4,3), axis None, max calibration, trt_high_precision_dtype BFloat16)",
        "NVIDIA/TensorRT-Model-Optimizer main examples/diffusers/quantization/utils.py "
        "filter_func_default (same disabled modules)",
    ],
    "weights": "fp8 e4m3, per-tensor (one scale per linear)",
    "activations": "fp8 e4m3, per-tensor static (max calibration), one input scale per linear; "
                   "attention Q, K, softmax probabilities and V each one static scale",
    "accumulator": "fp32 (bf16 epilogue: scale, bias)",
    "storage": None,
    "float_ops": {
        "transformer x_embedder, context_embedder, time_text_embed (4 linears), norm_out.linear, proj_out": "bf16",
        "RoPE (fp32 cos/sin), timestep sinusoid": "fp32",
        "LayerNorm / QK RMSNorm statistics": "fp32",
        "attention softmax": "fp32 (inside the FP8 FMHA; probabilities re-quantized to fp8)",
        "AdaLN modulation, gates, residual adds, GELU": "bf16",
        "CLIP text encoder (all GEMMs; softmax fp32)": "bf16",
        "T5 encoder (all GEMMs incl. wo; softmax fp32)": "bf16",
        "VAE decoder (convs, mid-block attention; GroupNorm statistics fp32)": "bf16",
    },
    "audit": {"allow": {f"dot_general {F8}": "transformer block linears (19x14 + 38x6) and "
                                              "attention QK^T / PV (57x2)",
                        "dot_general bf16xbf16->bf16": "transformer embedders / norm_out / "
                                                       "proj_out; CLIP, T5 linears and PV; VAE linears",
                        "dot_general bf16xbf16->f32": "CLIP / T5 / VAE attention logits",
                        "convolution bf16xbf16->bf16": "VAE decoder convolutions"},
              "require": [f"dot_general {F8}", "convolution bf16xbf16->bf16"],
              "phases": {
                  "clip_text_encoder": {"allow": {"dot_general bf16xbf16->bf16": "linears, PV",
                                                  "dot_general bf16xbf16->f32": "attention logits"},
                                        "require": ["dot_general bf16xbf16->bf16"]},
                  "t5_text_encoder": {"allow": {"dot_general bf16xbf16->bf16": "linears incl. wo, PV",
                                                "dot_general bf16xbf16->f32": "attention scores"},
                                      "require": ["dot_general bf16xbf16->bf16"]},
                  "transformer_step": {"allow": {f"dot_general {F8}": "block linears + FP8 MHA",
                                                 "dot_general bf16xbf16->bf16":
                                                     "x/context embedder, time_text_embed, norm_out, proj_out"},
                                       "require": [f"dot_general {F8}", "dot_general bf16xbf16->bf16"]},
                  "vae_decode": {"allow": {"convolution bf16xbf16->bf16": "decoder convs",
                                           "dot_general bf16xbf16->bf16": "mid-block attention q/k/v/o, PV",
                                           "dot_general bf16xbf16->f32": "mid-block attention logits"},
                                 "require": ["convolution bf16xbf16->bf16"]},
              },
              # full: 19x14 + 38x6 = 494 FP8 linears + 57x2 FP8 MHA BMMs = 608 per step;
              # 8 bf16 = x_embedder, context_embedder, 4 time_text_embed, norm_out.linear, proj_out
              "counts": {
                  "full": {"clip_text_encoder": {"dot_general bf16xbf16->bf16": 84,
                                                 "dot_general bf16xbf16->f32": 12},
                           "t5_text_encoder": {"dot_general bf16xbf16->bf16": 192,
                                               "dot_general bf16xbf16->f32": 24},
                           "transformer_step": {f"dot_general {F8}": 608,
                                                "dot_general bf16xbf16->bf16": 8},
                           "vae_decode": {"convolution bf16xbf16->bf16": 35,
                                          "dot_general bf16xbf16->bf16": 5,
                                          "dot_general bf16xbf16->f32": 1}},
                  "reduced": {"clip_text_encoder": {"dot_general bf16xbf16->bf16": 21,
                                                    "dot_general bf16xbf16->f32": 3},
                              "t5_text_encoder": {"dot_general bf16xbf16->bf16": 24,
                                                  "dot_general bf16xbf16->f32": 3},
                              "transformer_step": {f"dot_general {F8}": 48,      # 2x14 + 2x6 + 4x2
                                                   "dot_general bf16xbf16->bf16": 8},
                              "vae_decode": {"convolution bf16xbf16->bf16": 34,
                                             "dot_general bf16xbf16->bf16": 5,
                                             "dot_general bf16xbf16->f32": 1}},
              },
              "params": {"float8_e4m3fn": "block linear weights stored FP8 E4M3 (one kernel_scale each)"}},
    "notes": (
        "Departures from the research summary: (1) FP8 attention runs in all 57 blocks "
        "(double and single stream), not only the double-stream blocks; (2) the AdaLN "
        "modulation linears inside the blocks (norm1.linear, norm1_context.linear, "
        "norm.linear) are FP8 (1-row GEMMs); only the top-level embedders, time_text_embed, "
        "norm_out.linear and proj_out stay bf16; (3) T5 runs bf16 including wo: the "
        "deployed TRT T5 (t5.opt) has no fp32 wo, so the port's transformers-style fp32 "
        "wo (_keep_in_fp32_modules) becomes bf16 in Stage 2. The optional FP8 T5 "
        "(t5-fp8.opt, TRT_T5_PRECISION=fp8) is not the default of either the BFL CLI or "
        "the NVIDIA demo and is not modelled. NVFP4 (transformer.opt/fp4) is the "
        "Blackwell variant; FP8 is the Hopper/Ada export. FP8 MHA follows ModelOpt's "
        "sdpa decomposition: q and k are each multiplied by head_dim^-1/4 before their "
        "quantizers. The double-block attention output is quantized by the to_out / "
        "to_add_out input quantizers (no separate bmm2_output quantizer in the BFL "
        "export). Weights are stored bf16 in the ONNX and converted at engine build; the "
        "port stores them fp8 with a scalar kernel_scale (bias fp32, added in the epilogue). "
        "Each linear owns its input scale even where several share one input (q/k/v), as "
        "each ModelOpt linear has its own input_quantizer (494 x 2 + 57 x 4 = 1216 Q/DQ pairs, "
        "as in the ONNX). The reduced config is bf16 too, so the reduced audit sees the "
        "deployed signatures."
    ),
}
