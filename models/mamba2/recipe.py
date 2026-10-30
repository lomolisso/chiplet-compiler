"""Mamba-2 2.7B quantization recipe (see models/quant_recipes.py for the fields)."""

F8 = "f8E4M3FNxf8E4M3FN->f32"

RECIPE = {
    "recipe": "FP8 E4M3 W8A8 per-tensor static on the Mamba-2 mixer linears (ModelOpt FP8 / vLLM)",
    "basis": "fallback",
    "sources": [
        "HF hub search 2026-10-30: no quantized checkpoint from state-spaces; quantized "
        "mamba2-2.7b uploads are third-party with <= 30 downloads (ut-enyac/quamba2-2.7b-w8a8 / "
        "-w4a8 / -w4a16, GGUF Q8_0 / Q5_K_M, mlx 4/8-bit) vs 11k for AntonV/mamba2-2.7b-hf",
        "NVIDIA/TensorRT-LLM v0.20.0 examples/models/core/mamba/README.md: Mamba2 support matrix "
        "FP16 / BF16 only (no quantization)",
        "NVIDIA/TensorRT-Model-Optimizer main modelopt_recipes/configs/ptq/presets/model/fp8.yaml "
        "(FP8_DEFAULT_CFG: w8a8_fp8_fp8 per-tensor weight + input quantizers, max calibration) and "
        "units/default_disabled_quantizers.yaml ('*mixer.conv1d*', '*lm_head*', nn.Embedding disabled)",
        "nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8 hf_quant_config.json + safetensors header (ModelOpt "
        "FP8 on Mamba-2 mixers): mixer.in_proj / out_proj weight F8_E4M3 with scalar F32 weight_scale "
        "and (1,) F32 input_scale; conv1d, A_log, D, dt_bias, norms, embeddings, lm_head BF16",
        "vllm-project/vllm v0.11.0 model_executor/layers/mamba/mamba_mixer2.py: in_proj / out_proj "
        "built with quant_config, conv1d with quant_config=None",
    ],
    "weights": "fp8 e4m3, per-tensor scale (in_proj, out_proj)",
    "activations": "fp8 e4m3, per-tensor static scale (max calibration) at the in_proj and "
                   "out_proj inputs",
    "accumulator": "fp32 (bf16 output)",
    "storage": None,
    "float_ops": {"embedding gather (bf16) and tied lm_head (bf16 GEMM)": "bf16",
                  "depthwise causal conv1d + SiLU": "bf16",
                  "dt softplus / dt_bias": "bf16",
                  "SSD chunk scan, A = -exp(A_log), D skip, decode state update": "fp32",
                  "decode y = state . C (bmm)": "bf16",
                  "RMSNorm, gated RMSNorm, residual stream": "fp32",
                  "conv_state cache": "bf16", "ssm_state cache": "fp32"},
    "audit": {"allow": {f"dot_general {F8}": "mixer in_proj / out_proj (2 per layer per phase)",
                        "dot_general f32xf32->f32": "SSD chunk-scan einsums (prefill only; fp32 "
                                                    "as in HF / mamba_ssm)",
                        "dot_general bf16xbf16->bf16": "tied lm_head (last position); decode "
                                                       "bmm(state, C)",
                        "convolution bf16xbf16->bf16": "depthwise conv1d (not quantized)"},
              "require": [f"dot_general {F8}", "dot_general f32xf32->f32",
                          "dot_general bf16xbf16->bf16", "convolution bf16xbf16->bf16"],
              "params": {"float8_e4m3fn": "in_proj / out_proj kernels"}},
    "notes": "No official or widely deployed quantized Mamba-2 2.7B exists and TensorRT-LLM runs "
             "Mamba-2 only in FP16/BF16, so the GPU-datacenter fallback applies. Its form is "
             "pinned to NVIDIA's own FP8 recipe for Mamba-2 mixers (ModelOpt FP8_DEFAULT_CFG as "
             "shipped in nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8 and served by vLLM): exactly "
             "in_proj and out_proj are FP8; conv1d, A_log / D / dt_bias, norms, embeddings and the "
             "(tied) lm_head stay bf16. Matches the research summary. Pure SSM: no KV cache, and "
             "no FP8 state cache in any recipe (HF keeps ssm_state fp32; vLLM's default "
             "mamba_ssm_cache_dtype 'auto' would be bf16). Quamba2 W8A8 (int8, incl. SSM inputs) "
             "is a research checkpoint and was not taken. The reduced config runs in bf16 (was "
             "fp32) so both configs lower with the deployed dtypes. Prefill computes the logits "
             "of the last position only (generate() / vLLM serving; HF model(input_ids) would "
             "project all positions).",
}
