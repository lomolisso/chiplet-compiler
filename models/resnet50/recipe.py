"""ResNet-50 quantization recipe (see models/quant_recipes.py for the fields)."""

I8 = "i8xi8->i32"

RECIPE = {
    "recipe": "TensorRT INT8 W8A8 (MLPerf Inference NVIDIA)",
    "basis": "mlperf",
    "sources": [
        "mlcommons/inference_results_v5.0 closed/NVIDIA/code/resnet50/tensorrt/README.md "
        "(INT8, softmax removed, TensorRT symmetric calibration)",
        "... /calibrator.cache (TRT-100100-EntropyCalibration2, one scale per tensor, "
        "input_tensor_0 quantized)",
        "... /rn50_graphsurgeon.py (fc replaced by conv, TopK subnetwork)",
    ],
    "weights": "int8, per-output-channel symmetric (BN folded)",
    "activations": "int8, per-tensor static symmetric (entropy calibration); INT8 network input",
    "accumulator": "int32 (fp32 epilogue: scale, bias, residual add, ReLU)",
    "storage": None,
    "float_ops": {"conv epilogue (scale, bias, residual add, ReLU)": "fp32"},
    "audit": {"allow": {f"convolution {I8}": "all 53 convs",
                        f"dot_general {I8}": "fc"},
              "require": [f"convolution {I8}", f"dot_general {I8}"]},
    "notes": "Max pool and global average pool run on INT8 tensors (int32 sum). TopK host-side.",
}
