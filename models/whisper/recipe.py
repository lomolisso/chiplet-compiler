"""Whisper large-v3 quantization recipe (see models/quant_recipes.py for the fields)."""

I8 = "i8xi8->i32"
F16 = "f16xf16->f16"

_CT2 = "OpenNMT/CTranslate2 v4.5.0"

RECIPE = {
    "recipe": "CTranslate2 int8_float16 (faster-whisper): W8A8 int8, dynamic per-row activations",
    "basis": "production",
    "sources": [
        f"{_CT2} src/layers/common.cc Dense::operator(): int8 weight -> _quantize_op(input) "
        "(per-row scale), int8 GEMM into an INT32 output, _dequantize_op(input scale, weight "
        "scale, bias) in the float type",
        f"{_CT2} src/ops/quantize.cc: INT8 scale.resize({{input.size() / depth}}), one scale per "
        "row of the flattened [tokens, depth] input",
        f"{_CT2} python/ctranslate2/specs/model_spec.py _quantize: every `weight` with a "
        "`weight_scale` slot (LinearSpec, EmbeddingsSpec, Conv1DSpec), scale = 127 / "
        "amax(|W[i, :]|) per output row; other float variables -> float16 for int8_float16",
        f"{_CT2} src/models/model.cc: on CUDA (and DNNL) quantized conv is unsupported -> conv "
        "weights converted to the float type; aliases also alias `<name>_scale`",
        f"{_CT2} src/models/whisper.cc is_quantizable / python/ctranslate2/converters/"
        "transformers.py WhisperLoader (proj_out -> decoder/projection, tied to embed_tokens)",
        f"{_CT2} src/layers/attention.cc: QK^T / softmax / PV via ops::MatMul / SoftMax in the "
        "float type; K/V cache kept in the float type",
        f"{_CT2} docs/quantization.md: int8_float16 on GPU (CC >= 7.0); per-row symmetric weights",
        "mlcommons/inference_results_v5.1 closed/Intel/code/whisper/pytorch-cpu/quantize_model.py "
        "(AutoRound RTN W8 per-channel sym, act_bits 8, incl. proj_out) and SUT.py (vLLM bf16, "
        "kv_cache_dtype fp8): the MLPerf int8 submission, same GEMM arithmetic",
        "mlcommons/inference_results_v5.1 closed/NVIDIA/code/whisper/tensorrt/builder.py: FP16 "
        "only (no quantized NVIDIA recipe)",
    ],
    "weights": ("int8, per-output-row symmetric (127 / row amax): every Linear incl. the tied LM "
                "head, and the token embedding table"),
    "activations": "int8, dynamic per-row (per-token) symmetric amax, computed in the graph",
    "accumulator": "int32 (dequantized with input scale x weight scale + bias to fp16)",
    "storage": ("token embedding table int8 with per-row scales (gathered rows dequantized to "
                "fp16); self and cross K/V caches fp16"),
    "float_ops": {"conv1 / conv2 stem (+ GELU)": "fp16 (no quantized conv on CUDA)",
                  "QK^T, softmax, PV": "fp16",
                  "LayerNorm, GELU, residual adds": "fp16",
                  "sinusoidal / learned position embeddings": "fp16"},
    "audit": {"allow": {f"dot_general {I8}": "q/k/v/out, cross q/k/v/out, fc1, fc2, LM head",
                        f"dot_general {F16}": "attention QK^T and PV",
                        f"convolution {F16}": "conv1, conv2"},
              "require": [f"dot_general {I8}", f"dot_general {F16}", f"convolution {F16}"],
              "params": {"int8": "token embedding table (gathered, dequantized per row) and kernels",
                         "float16": "conv stem, LayerNorm, position embeddings"}},
    "notes": ("Corrects the research summary: CTranslate2 int8 is NOT weight-only. Dense with "
              "int8 weights quantizes its input on the fly (one scale per row) and runs an "
              "int8 x int8 -> int32 GEMM. The token embeddings (and so the tied LM head, whose "
              "scale is aliased) are int8 per-row; the conv stem is fp16 on GPU; attention "
              "BMMs, softmax, LayerNorm and the KV cache are fp16. Choice: CTranslate2 / "
              "faster-whisper is the most-deployed quantized Whisper runtime; MLPerf's NVIDIA "
              "submissions are FP16 (not quantized) and the only MLPerf int8 submission (Intel "
              "CPU, v5.1) has the same W8A8 per-channel / dynamic per-token GEMMs (it adds an FP8 "
              "KV cache and keeps embeddings bf16). Stage 2 switches the float dtype from bf16 "
              "(MLPerf) to fp16 (int8_float16), also in the reduced config. CTranslate2 fuses "
              "self-attention QKV and cross-attention KV into one GEMM each (same per-row "
              "scales, same arithmetic); the port keeps separate projections. Needs a per-row "
              "symmetric int8 embedding: `QEmbed(per_row=True)`. Biases are stored fp32 "
              "(QDense) where CTranslate2 stores them fp16; they are added in the fp32 "
              "epilogue before the fp16 output either way."),
}
