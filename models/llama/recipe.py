"""Llama 3.1-8B quantization recipe (see models/quant_recipes.py for the fields)."""

F8 = "f8E4M3FNxf8E4M3FN->f32"

RECIPE = {
    "recipe": "ModelOpt FP8 E4M3 W8A8 per-tensor static + FP8 KV cache + FP8 attention "
              "(TensorRT-LLM, MLPerf Inference Hopper)",
    "basis": "mlperf",
    "sources": [
        "mlcommons/inference_results_v5.1 closed/HPE/configs/HPE_Cray_XD670_H200_SXM_141GBx8/"
        "Offline/llama3_1-8b.py and closed/Dell/configs/R7715-H100-NVL-94GBx3/Offline/llama3_1-8b.py "
        "(NVIDIA code on Hopper): precision 'fp8', trtllm_checkpoint_flags kv_cache_dtype 'fp8', "
        "build flags use_fp8_context_fmha / use_paged_context_fmha / use_fused_mlp enable",
        "mlcommons/inference_results_v5.1 closed/NVIDIA/code/llama3_1-8b/tensorrt/builder.py "
        "(LLAMA3_1_8BQuantizerConfig: Precision.FP8 -> fp8-quantized-modelopt, calibration on "
        "mlperf_llama3.1-8b_calibration_1k) and code/llmlib/builder.py (TensorRT-LLM "
        "examples/quantization/quantize.py --qformat fp8 --kv_cache_dtype fp8)",
        "NVIDIA/TensorRT-Model-Optimizer 0.27.0 modelopt/torch/quantization/config.py: "
        "FP8_DEFAULT_CFG (weight and input quantizers num_bits (4,3), axis None, algorithm 'max'), "
        "_default_disabled_quantizer_cfg ('*lm_head*' disabled), FP8_KV_CFG ('*[kv]_bmm_quantizer' "
        "(4,3), axis None)",
        "HF nvidia/Llama-3.1-8B-Instruct-FP8 @ 42d9515 (ModelOpt 0.27.0 export of the same recipe): "
        "hf_quant_config.json quant_algo FP8, kv_cache_quant_algo FP8, exclude_modules ['lm_head']; "
        "safetensors header: q/k/v/o/gate/up/down weight F8_E4M3 with scalar F32 weight_scale and "
        "input_scale, k_proj.k_scale / v_proj.v_scale scalar F32 per layer; embed_tokens, lm_head, "
        "norms BF16",
        "NVIDIA/TensorRT-LLM v1.0.0 cpp/kernels/xqa/mha_sm90.cu (generation on Hopper with an FP8 "
        "cache: MathElem = fp8, Q cast to fp8, QK^T and PV FP8 GMMA with fp32 accumulation, P "
        "scaled by 448 before the fp8 conversion; one kvCacheScale shared by K and V), "
        "cpp/tensorrt_llm/kernels/unfusedAttentionKernels/unfusedAttentionKernels_2_template.h "
        "(FP8 context FMHA: Q, K, V quantized with the same kv scale) and "
        "cpp/tensorrt_llm/common/attentionOp.cpp (FP8 attention output quantized with o_proj's "
        "input scale)",
    ],
    "weights": "fp8 e4m3, one fp32 scale per tensor (q, k, v, o, gate, up, down)",
    "activations": "fp8 e4m3, per-tensor static (max calibration) GEMM inputs; attention "
                   "Q/K/V in fp8 with the per-layer KV scale, softmax probabilities in fp8 "
                   "(fixed scale 1/448), attention output quantized with o_proj's input scale",
    "accumulator": "fp32 (bf16 epilogue)",
    "storage": "KV cache fp8 e4m3, one static per-tensor scale per layer (max(k_scale, v_scale) "
               "in TensorRT-LLM)",
    "float_ops": {"embedding gather": "bf16", "lm_head GEMM (ModelOpt excludes *lm_head*)": "bf16",
                  "RMSNorm": "fp32 math, bf16 out", "RoPE": "fp32 angles, bf16 rotate",
                  "softmax": "fp32", "SiLU * up, residual adds": "bf16"},
    "audit": {"allow": {f"dot_general {F8}": "q/k/v/o/gate/up/down GEMMs; QK^T and PV "
                                             "(prefill: FP8 context FMHA, decode: XQA sm90)",
                        "dot_general bf16xbf16->bf16": "lm_head (last position) only"},
              "require": [f"dot_general {F8}", "dot_general bf16xbf16->bf16"],
              "params": {"float8_e4m3fn": "q/k/v/o/gate/up/down weights (the FP8 KV cache is a "
                                          "phase input, float8_e4m3fn in model.cache_spec)"}},
    "notes": "Departures from the research summary: none in the arithmetic (FP8 E4M3 per-tensor "
             "static W8A8, FP8 KV, FP8 context attention; lm_head, embeddings, norms bf16). Added: "
             "generation-phase attention is FP8 too on Hopper (XQA sm90 runs QK^T / PV as FP8 "
             "GMMA when the cache is FP8), so decode_step also gets FP8 attention BMMs. MLPerf "
             "v5.1 NVIDIA's own submission is Blackwell NVFP4 (B200 config precision 'fp4'); the "
             "Hopper FP8 configs come from partner submissions (HPE, Dell) running NVIDIA's code. "
             "Checkpoint naming: the nvidia/Llama-3.1-8B-Instruct-FP8 export keeps separate "
             "k_scale / v_scale and separate q/k/v and gate/up input scales; TensorRT-LLM fuses "
             "QKV and gate/up (max of the input scales) and uses one KV scale -- same dtypes. "
             "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8 (vLLM, compressed-tensors) has the same "
             "per-tensor static W8A8 GEMMs but no FP8 KV. The reduced config must also run in "
             "bf16 for the audit signatures to apply.",
}
