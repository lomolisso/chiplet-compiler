"""Wide & Deep quantization recipe (see models/quant_recipes.py for the fields)."""

I8 = "i8xi8->i32"
F16 = "f16xf16->f16"

RECIPE = {
    "recipe": ("Recsys fallback: INT8 W8A8 deep MLP + FBGEMM rowwise uint8 embedding tables "
               "(fp16 scale + bias per row)"),
    "basis": "fallback",
    "sources": [
        "NVIDIA DeepLearningExamples@729963dd TensorFlow2/Recommendation/WideAndDeep/triton/"
        "README.md: deployed as TF SavedModel with automatic FP16 or TensorRT FP16; no INT8 "
        "path (so no quantized production recipe for this model)",
        "... /trainer/utils/arguments.py: --amp opt-in (training mixed precision only)",
        "FBGEMM fbgemm_gpu IntNBitTableBatchedEmbeddingBagsCodegen (inference TBE, "
        "pytorch.org/FBGEMM tbe_ops_inference): SparseType.INT8 rows = dim bytes + "
        "scale_bias_size_in_bytes 4 (fp16 scale, fp16 bias), output dtype fp16 by default; "
        "FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf",
        "models/quant_recipes.py fallback rule: recsys -> INT8 W8A8 (per-output-channel "
        "symmetric weights, per-tensor static symmetric activations)",
    ],
    "weights": "deep Dense layers (6, incl. the 1024 -> 1 output): int8 per-output-channel symmetric",
    "activations": "int8, per-tensor static symmetric (calibrated) at every deep Dense input",
    "accumulator": "int32 (fp32 epilogue: scale, bias, ReLU)",
    "storage": ("16 deep embedding tables: uint8 rowwise asymmetric, x = q * scale + bias with "
                "fp16 scale and bias per row (FBGEMM INT8 TBE); rows dequantized to fp16"),
    "float_ops": {"multi-hot masked sum pooling, deep-input concat": "fp16",
                  "wide part ([card, 1] tables, numerics @ W[13, 1], bias)": "fp16",
                  "sigmoid(wide + deep)": "fp32"},
    "audit": {"allow": {f"dot_general {I8}": "deep Dense layers",
                        f"dot_general {F16}": "wide numeric [13] x [13, 1]"},
              "require": [f"dot_general {I8}"],
              "params": {"uint8": "deep embedding tables (row-wise codes)",
                         "float16": "row-wise scale_bias, wide tables / numeric kernel / bias",
                         "int8": "deep Dense kernels"}},
    "notes": ("Confirms the research summary: NVIDIA ships FP16 only, so the recsys fallback "
              "applies (INT8 W8A8 MLP) with the industry embedding format (FBGEMM / TorchRec "
              "inference rowwise INT8, fp16 scale + bias per row). Departures: the wide part's "
              "[card, 1] tables stay fp16 (rowwise uint8 would store 1 byte of data plus 4 bytes "
              "of scale / bias per row) and its 13 x 1 numeric product stays fp16 (a GEMV of "
              "13 MACs); uint4 for the largest tables is not used (no evidence for this model). "
              "The deep input (fp16 embeddings + numerics) is quantized at dense's input."),
}
