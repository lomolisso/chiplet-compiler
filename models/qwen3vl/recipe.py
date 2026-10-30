"""Qwen3-VL-235B-A22B quantization recipe (see models/quant_recipes.py for the fields)."""

F8 = "f8E4M3FNxf8E4M3FN->f32"
# Float GEMMs: bf16 (both configs).
BF = "bf16xbf16->bf16"

RECIPE = {
    "recipe": "Qwen official FP8 (fine-grained E4M3, 128x128 weight blocks, dynamic 1x128 activations)",
    "basis": "official",
    "sources": [
        "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 @ 7fbcd8c9 config.json quantization_config: "
        "quant_method fp8, fmt e4m3, weight_block_size [128, 128], activation_scheme dynamic, "
        "ignored_layers = lm_head, every model.visual.* linear / norm / patch_embed / pos_embed "
        "(27 blocks, merger, 3 deepstack mergers), all 94 language_model.layers.*.mlp.gate",
        "... safetensors headers (1952 tensors): F8_E4M3 q/k/v/o_proj.weight with F32 "
        "weight_scale_inv [N/128, K/128] (q [64, 32], k/v [4, 32], o [32, 64]); fused experts "
        "gate_up_proj [128, 4096, 3072] F8_E4M3 + gate_up_proj_scale_inv [128, 32, 24], "
        "down_proj [128, 1536, 4096] + down_proj_scale_inv [128, 12, 32] (scales [E, K/128, N/128]); "
        "BF16: embed_tokens, lm_head, mlp.gate, every norm, the whole visual tower",
        "... README.md: 'fine-grained fp8 quantization with block size of 128'; serve with vLLM / SGLang",
        "vllm-project/vllm v0.11.0 model_executor/layers/quantization/fp8.py (Fp8Config: block "
        "quantization requires activation_scheme dynamic; ignored_layers -> UnquantizedLinearMethod) "
        "and quantization/utils/fp8_utils.py apply_w8a8_block_fp8_linear "
        "(per_token_group_quant_fp8(input, block_size[1]=128): one fp32 scale per token per 128 K)",
        "vllm v0.11.0 layers/fused_moe/fused_moe.py (fp8 w8a8 block_shape [128, 128] experts, "
        "activations per-token-group quantized for both expert GEMMs)",
    ],
    "weights": "fp8 E4M3, one fp32 scale (weight_scale_inv, a dequant multiplier) per 128x128 "
               "block of every text attention projection and every expert gate_up / down matrix",
    "activations": "fp8 E4M3, dynamic, one fp32 scale per token per 128-wide K group (computed "
                   "in the graph) at the input of each FP8 GEMM",
    "accumulator": "fp32 per 128-K block, scaled by act-group x weight-block scale and summed in fp32 "
                   "(models.quant.block_scaled_dot); bf16 output",
    "storage": None,
    "float_ops": {"vision tower (patch-embed conv, 27 ViT blocks, 3 DeepStack mergers, merger)": "bf16",
                  "token embedding, lm_head": "bf16",
                  "MoE router (mlp.gate) + softmax / top-k": "bf16 GEMM, fp32 softmax",
                  "attention QK^T / PV (KV cache bf16, kv_cache_dtype auto)": "bf16, fp32 softmax",
                  "RMSNorm / LayerNorm, M-RoPE, SwiGLU, MoE combine, DeepStack adds": "fp32 / bf16 as float port"},
    "audit": {"allow": {f"dot_general {F8}": "text q/k/v/o and expert gate_up / down "
                                             "(block-scaled, one partial product per 128-K block)",
                        f"dot_general {BF}": "vision linears + attention, router, text QK^T / PV, lm_head",
                        f"convolution {BF}": "vision patch-embed Conv3d (ignored by the FP8 config)"},
              "require": [f"dot_general {F8}", f"dot_general {BF}", f"convolution {BF}"],
              "params": {"float8_e4m3fn": "q/k/v/o kernels and expert wi / wo",
                         "float32": "128x128 block scales (weight_scale_inv)",
                         "bfloat16": "vision tower, router, embedding, lm_head, norms"}},
    "notes": "Confirms the research summary. Departures / precisions: (1) the official config "
             "lists the float carve-outs under `ignored_layers` (vLLM Fp8Config key), not "
             "`modules_to_not_convert`; (2) the entire vision tower is float (all 27 ViT blocks, "
             "not only the mergers), and the embedding is bf16 (not a linear, never converted); "
             "(3) MLPerf v6.1 submitters' recipes (NVFP4, MXFP4, per-channel FP8) are not used: "
             "the model owner's official checkpoint takes precedence; (4) no FP8 KV cache or FP8 "
             "attention in the official recipe (no kv_cache_scheme; vLLM default kv dtype auto = "
             "bf16). Expert scales are stored [E, K/128, N/128] (weights [E, K, N]); attention "
             "scales [N/128, K/128] (torch Linear [N, K]) -- transpose to the port's [K, N] kernels. "
             "gate_up (2F = 3072) has 24 N-blocks: gate and up never share a block (F = 1536 = 12 x 128). "
             "Port: attention scales stored [K/128, N/128] (the checkpoint's [N/128, K/128] "
             "transposed, mapped by verify/upstream/qwen3vl.py), expert scales as stored; experts "
             "run block_scaled_dot(batch_dims=1) over the static-dispatch buffers [E, C, D]. The "
             "reduced config is bf16 with every FP8 GEMM K and N a multiple of 128 (D 128, "
             "head_dim 64, F 128), so it keeps the 128 block.",
}
