"""Gemma 4 26B-A4B quantization recipe (see models/quant_recipes.py for the fields)."""

I8 = "i8xi8->i32"
BF = "bf16xbf16->bf16"
F32 = "f32xf32->f32"
_GGUF = "google/gemma-4-26B-A4B-it-qat-q4_0-gguf @ d1c082be"
_LCPP = "ggml-org/llama.cpp @ 5047748f"

RECIPE = {
    "recipe": "Google official QAT Q4_0 GGUF on llama.cpp: Q4_0 weights x Q8_0 activations "
              "(int8 dots per 32-K block), Q6_K x Q8_K tied lm_head",
    "basis": "official",
    "sources": [
        f"{_GGUF} gemma-4-26B_q4_0-it.gguf header (658 tensors, general.file_type 2 = MOSTLY_Q4_0): "
        "Q4_0 = blk.*.attn_q / attn_k / attn_v / attn_output, ffn_gate / ffn_up / ffn_down (dense "
        "shared MLP), ffn_gate_up_exps [2816, 1408, 128] and ffn_down_exps [704, 2816, 128] (experts); "
        "Q6_K = token_embd [2816, 262144] (tied lm_head); F32 = ffn_gate_inp [2816, 128] (router), "
        "ffn_gate_inp.scale, ffn_down_exps.scale (per-expert scale), every norm, layer_output_scale, "
        "rope_freqs",
        f"{_GGUF} gemma-4-26B-it-mmproj.gguf header (general.file_type 32 = MOSTLY_BF16): every vision "
        "linear and mm.input_projection BF16, patch_embd / position_embd / norms / std_* F32",
        f"{_GGUF} README.md: 'GGUF (Q4_0): Ready-to-deploy ... llama.cpp'; QAT bf16 weights in "
        "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized @ f1e06dc5 (same 1013 tensors / config as "
        "google/gemma-4-26B-A4B-it)",
        f"{_LCPP} ggml/src/ggml-common.h block_q4_0 {{ggml_half d; uint8 qs[16]}} (QK4_0 32), "
        "block_q8_0 {ggml_half d; int8 qs[32]}, block_q6_K {uint8 ql[128], qh[64]; int8 scales[16]; "
        "ggml_half d} (QK_K 256), block_q8_K {float d; int8 qs[256]; int16 bsums[16]}",
        f"{_LCPP} ggml/src/ggml-cpu/ggml-cpu.c type_traits_cpu: Q4_0 vec_dot_type = Q8_0, Q6_K "
        "vec_dot_type = Q8_K; ggml-cpu/quants.c ggml_vec_dot_q4_0_q8_0_generic (per 32-block int "
        "sumi, sumf += sumi * d_w * d_x) and ggml_vec_dot_q6_K_q8_K_generic (int8 x int8 per 16 "
        "sub-block, times the int8 sub-scale in int32, summed over the 256 super-block, then "
        "d_w * d_x); ggml-quants.c quantize_row_q8_0_ref / quantize_row_q8_K_ref (d = amax / 127), "
        "dequantize_row_q6_K (get_rows of the embedding)",
    ],
    "weights": "Q4_0: int4 in [-8, 7] (ggml q - 8), one fp16 scale d per 32 consecutive weights "
               "along K: text q / k / v / o, dense shared MLP gate / up / down, all 128 experts' "
               "gate_up / down. Token embedding (tied lm_head): Q6_K, int8 values in [-32, 31] "
               "(6-bit q - 32), an int8 scale per 16 and an fp16 scale per 256 along K",
    "activations": "dynamic int8 per block along K, quantized in the graph: Q8_0 (32, d = amax / "
                   "127) for every Q4_0 GEMM, Q8_K (256) for the Q6_K lm_head",
    "accumulator": "int32 per 32-block (Q4_0) / per 16-sub-block times int8 sub-scale over the "
                   "256 super-block (Q6_K); fp32 d_w * d_x scaling and block sum; bf16 out",
    "storage": "token embedding table Q6_K (gathered rows dequantized to bf16); KV cache stays "
               "16-bit (llama.cpp default f16; port bf16)",
    "float_ops": {"MoE router (ffn_gate_inp, F32 in the GGUF) + softmax / top-k": "bf16 GEMM "
                  "(library dtype), fp32 softmax",
                  "attention QK^T / PV, softmax": "bf16 (library dtype)",
                  "RMSNorms, p-RoPE, GeGLU, per-expert scale, MoE combine, logit soft-cap":
                      "bf16 / fp32 as the float port",
                  "vision tower (27 blocks, patch / pos embedding, pooling, standardize) and "
                  "mm_input_projection (mmproj GGUF is BF16 / F32)": "fp32 (library keeps vision fp32)"},
    "audit": {"allow": {f"dot_general {I8}": "Q4_0 x Q8_0 (q/k/v/o, dense MLP, experts) and "
                                             "Q6_K x Q8_K (tied lm_head)",
                        f"dot_general {BF}": "router (float in the GGUF), attention QK^T / PV",
                        f"dot_general {F32}": "vision_encoder phase (vision tower + projector, "
                                              "not quantized, fp32 in the library)"},
              "require": [f"dot_general {I8}", f"dot_general {BF}", f"dot_general {F32}"],
              "params": {"int4": "Q4_0 values of every text attention / dense-MLP / expert kernel",
                         "float16": "Q4_0 block scales; Q6_K super-block scale",
                         "int8": "Q6_K embedding values and sub-block scales"}},
    "notes": (
        "Confirms the research summary's official Google QAT Q4_0 release (int4, block 32, fp16 "
        "scale) with these precisions from the GGUF header: router (ffn_gate_inp), per-expert scale "
        "and router scale stay float; vision tower and mm projection are not quantized (separate "
        "BF16 mmproj); the token embedding / tied lm_head is Q6_K, not Q4_0; Q4_0 blocks run along "
        "K. Departure from the research summary: not weight-only. The artifact is a GGUF consumed "
        "by llama.cpp, whose CPU (vec_dot_type Q8_0 / Q8_K) and CUDA MMQ / MMVQ kernels quantize the "
        "activations on the fly and run integer dots; the port models the CPU reference kernels "
        "(Q4_0 x Q8_0 per 32-K block; Q6_K x Q8_K). Metal and the dequant + cuBLAS path would be "
        "weight-only (not modelled). Implementation: quantization is written into the vendored "
        "library (models/gemma4/upstream/gemma, every edit marked '# quantized'): "
        "_config.TransformerConfig.weight_quant / embed_quant, set to 'q4_0' / 'q6_k' in "
        "_gemma4.Gemma4_26B_A4B.config; _layers.Einsum(quant='q4_0') holds int4 `w` + float16 "
        "`<weight_name>_scale` (w's shape with the innermost contracted axis / 32) and runs "
        "_layers.q4_0_einsum (models.quant.block_scaled_dot kind='int8', weight_block (32, 1)); "
        "_moe._Weight(quant='q4_0', k_axis) returns (values, scales) for the experts, which "
        "net.MoECapacityDispatch runs with q4_0_einsum batched over experts (the ragged library "
        "path raises for Q4_0); _modules.Embedder(quant='q6_k') holds input_embedding int8 [V, D] "
        "+ input_embedding_scale int8 [V, D/16] + input_embedding_d float16 [V, D/256], dequantizes "
        "gathered rows (_layers.q6_k_dequant) and runs the lm_head with _layers.q6_k_dot; "
        "utils/_dtype_params keeps int4 / int8 / float16 parameters as declared (the library "
        "otherwise casts every text param to bf16 at init). The router keeps the library's bf16 "
        "(GGUF stores it F32); the KV cache stays bf16 (llama.cpp f16). Reduced config resized so "
        "every K is a multiple of 32 and D = 256 holds one Q6_K super-block (embed 256, head_dim "
        "32, global key 64) and runs in bf16 like the full model."
    ),
}
