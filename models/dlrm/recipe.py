"""DLRMv2 quantization recipe (see models/quant_recipes.py for the fields)."""

I8 = "i8xi8->i32"

_NV = "mlcommons/inference_results_v4.1 closed/NVIDIA"

RECIPE = {
    "recipe": "TensorRT INT8 W8A8 + INT8 embedding tables (MLPerf Inference NVIDIA dlrm-v2-99)",
    "basis": "mlperf",
    "sources": [
        f"{_NV}/code/dlrm-v2/tensorrt/builder.py DLRMv2TRTNetwork defaults: bot_mlp, embeddings, "
        "interaction_op, top_mlp, final_linear precision 'int8'; sigmoid fp32 output",
        f"{_NV}/code/dlrm-v2/tensorrt/builder.py dump_embedding_weights.int8_quantize: one scale "
        "per table = max|table| / 127, rint, clip [-127, 127]; DLRMv2EmbeddingLookupPlugin reads "
        "the int8 mega table + scales and writes the int8 interaction input",
        f"{_NV}/code/dlrm-v2/tensorrt/builder.py _build_linear (add_matrix_multiply with constant "
        "weights) and _build_interaction_op (use_conv=True: V / W as 1x1 convs, bias in the conv)",
        f"{_NV}/code/dlrm-v2/tensorrt/calibrator.cache (TRT-100100-EntropyCalibration2: per-tensor "
        "ranges for every MLP weight `*.mm.w.output`, GEMM / bias / ReLU output, embedding "
        "lookup output, crossnet vx / wvx / left_term / out)",
        f"{_NV}/configs/dlrm-v2/Offline/__init__.py: *_HighAccuracy (99.9) only sets "
        "interaction_op_precision = 'fp16'",
        f"{_NV}/documentation/calibration.md (DLRMv2 embedding tables mapped by max-abs per table)",
        "mlcommons/inference_results_v4.1 closed/Intel/code/dlrm-v2-99/pytorch-cpu/python/"
        "calibration.py: IPEX qint8, per-tensor affine histogram activations, per-channel "
        "symmetric weights",
    ],
    "weights": ("int8 symmetric: bottom / top MLP and final linear per-tensor (calibrated range "
                "of the constant weight of add_matrix_multiply); DCN V and W per-output-channel "
                "(1x1 convs)"),
    "activations": ("int8, per-tensor static symmetric (entropy calibration): dense input, every "
                    "GEMM input, interaction input, crossnet intermediates"),
    "accumulator": "int32 (fp32 epilogue: scale, bias, ReLU)",
    "storage": ("26 embedding tables int8 with one symmetric fp32 scale per table (max|t| / 127); "
                "gathered rows dequantized, sum-pooled in fp32, written with the bottom-MLP "
                "output as one int8 [B, 3456] tensor"),
    "float_ops": {"embedding bag sum pooling": "fp32 (int8 out)",
                  "crossnet x0 * (W V x + b) + x (int8 I/O)": "fp32",
                  "sigmoid": "fp32"},
    "audit": {"allow": {f"dot_general {I8}": "bottom MLP, DCN V / W, top MLP, final linear"},
              "require": [f"dot_general {I8}"],
              "params": {"int8": "embedding tables and GEMM kernels"}},
    "notes": ("Confirms the research summary for the 99% target (everything INT8; 99.9% moves "
              "only the interaction to fp16). Refinements: the MLP weights are per-tensor (TRT "
              "matrix-multiply layers with calibrated constant ranges), the DCN weights "
              "per-channel; the table scale is max|t| / 127 in the builder code (calibration.md "
              "says 127.5). Intel (asymmetric per-tensor activations, per-channel weights) is "
              "not chosen. Implementation (user decision: no runtime substitution): copy keras-rs "
              "FeatureCross and EmbedReduce / DistributedEmbedding (default-device path) into "
              "models/dlrm/upstream/ with the Apache-2.0 header and edit those copies and "
              "upstream/model.py: int8 tables + `embedding_scale` (per table), Keras int8 dense "
              "layers with `kernel_scale` / `input_scale` written with keras.ops (int8 x int8 "
              "matmul -> int32 on the JAX backend), activation quantize ops on the interaction "
              "input; net.named_variables gains the scale paths. As built: every INT8 layer owns "
              "its calibrated `input_scale` and quantizes its fp32 input (TensorRT keeps the "
              "tensor int8 between layers with the same per-tensor scale; same GEMM "
              "arithmetic), int8 kernels / tables and scales are non-trainable Keras "
              "variables passed to stateless_call."),
}
