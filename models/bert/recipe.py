"""BERT-Large quantization recipe (see models/quant_recipes.py for the fields)."""

I8 = "i8xi8->i32"

_NV = "mlcommons/inference_results_v4.1 closed/NVIDIA/code/bert/tensorrt"

RECIPE = {
    "recipe": "TensorRT INT8 W8A8 incl. attention BMMs (MLPerf Inference NVIDIA bert-99)",
    "basis": "mlperf",
    "sources": [
        f"{_NV}/network.py BERTVarSeqLenINT8 (unchanged in v5.0): QKV / attention-out / FC1(+GELU) / "
        "FC2 / squad FC as INT8 convs or SmallTileGEMM; every tensor gets a per-tensor dynamic "
        "range from the QAT model's amax (`*_input_amax`, `*_input_quantizer_amax`)",
        f"{_NV}/network.py: CustomQKVToContextPluginDynamic type_id int8, dr_qkv = max(qv_a, "
        "qv_b, av_b amax) for Q, K, V; dq_probs = av_a amax / 127 for the softmax probabilities",
        f"{_NV}/network.py: CustomSkipLayerNormPluginDynamic type int8 (int8 in / out, fp32 "
        "gamma / beta), CustomEmbLayerNormPluginDynamic (fp32 tables, output_fp16), "
        "squad logits fp16, add_gelu tanh approximation with precision FLOAT",
        f"{_NV}/README.md: INT8 for bert-99 from bert_large_v1_1_fake_quant.onnx (MLCommons "
        "QAT model); fp16 / FP8 for bert-99.9",
        "mlcommons/inference_results_v4.1 closed/NVIDIA/documentation/calibration.md: symmetric; "
        "weights per-channel (per-row for matrix multiply) max-abs; activations per-tensor",
    ],
    "weights": "int8, per-output-channel (per-row) symmetric max-abs, all GEMMs incl. qa_outputs",
    "activations": ("int8, per-tensor static symmetric (QAT amax): GEMM inputs, Q, K, V, softmax "
                    "probabilities (scale amax/127), residual stream between layers"),
    "accumulator": "int32 (fp32 epilogue: scale, bias)",
    "storage": "hidden states between ops / layers stay int8 (skip-LayerNorm int8 in and out)",
    "float_ops": {"embedding gathers + sum + LayerNorm (fp32 tables)": "fp32, output int8",
                  "softmax (inside the fused MHA, int8 in / int8 out)": "fp32",
                  "skip-LayerNorm (residual add + LN, int8 in / out)": "fp32",
                  "GELU (tanh approximation, FC1 epilogue, int8 out)": "fp32",
                  "attention mask": "fp32",
                  "qa_outputs logits": "fp16"},
    "audit": {"allow": {f"dot_general {I8}": "Q/K/V, attention out, FC1, FC2, QK^T, PV, qa_outputs"},
              "require": [f"dot_general {I8}"],
              "params": {"int8": "GEMM kernels"}},
    "notes": ("Confirms the research summary (all GEMMs and both attention BMMs INT8, softmax "
              "output int8, GELU / LN fp32 with int8 I/O). Additions: the residual stream is int8 "
              "between layers (skip-LN plugins are int8 in and out); Q, K and V share one scale "
              "(max of the three amaxes); the builder computes GELU with the tanh approximation "
              "(HF uses exact erf), which Stage 2 adopts as part of the deployed arithmetic; "
              "qa_outputs is an INT8 conv with fp16 logits. NVIDIA packs variable-length "
              "sequences (no padding), so the port's additive mask only removes padded keys."),
}
