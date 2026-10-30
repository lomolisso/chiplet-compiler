"""YOLO11-L quantization recipe (see models/quant_recipes.py for the fields)."""

I8 = "i8xi8->i32"

RECIPE = {
    "recipe": "Ultralytics TensorRT INT8 export (format=engine, quantize=8; TensorRT 8-10 implicit INT8)",
    "basis": "production",
    "sources": [
        "ultralytics/ultralytics @84694ebf (v8.4.183) ultralytics/utils/export/engine.py::onnx2engine: "
        "BuilderFlag.INT8 + BuilderFlag.FP16 ('unquantized layers take the fastest precision allowed'); "
        "IInt8Calibrator with CalibrationAlgoType.MINMAX_CALIBRATION (ENTROPY_CALIBRATION_2 only for DLA) "
        "on images / 255; per-layer constraints in the Detect head (the highest /model.N/ prefix): "
        "ACTIVATION layers named *sigmoid* -> float32, CONVOLUTION layers ending '.2/Conv' "
        "(cv2.i.2, cv3.i.2) or under '/dfl/' -> float16, with OBEY_PRECISION_CONSTRAINTS "
        "(Ultralytics issue #24668)",
        "same file, modelopt_quantize_onnx (TensorRT 11, strongly typed): ModelOpt ONNX "
        "quantize_mode='int8', calibration_method='max', nodes_to_exclude=[r'.*\\.2/Conv$', r'.*/dfl/'] "
        "(the same carve-outs as Q/DQ)",
        "ultralytics/engine/exporter.py::export_engine (ONNX -> onnx2engine, calibration dataloader "
        "from get_int8_calibration_dataloader)",
        "TensorRT developer guide, implicit quantization: weights per-output-channel symmetric, "
        "activations per-tensor symmetric from the calibrator's dynamic ranges",
    ],
    "weights": "int8 per-output-channel symmetric (BN folded) for every conv except the Detect "
               "head's six final 1x1 convs and the DFL conv",
    "activations": "int8 per-tensor static symmetric (min-max calibration); fp32 network input "
                   "quantized by the first conv",
    "accumulator": "int32 (fp32 epilogue: scale, folded-BN bias, SiLU, residual add)",
    "storage": None,
    "float_ops": {"Detect cv2.i.2 / cv3.i.2 1x1 convs (box / class logits)": "fp16",
                  "DFL softmax and its fixed 1x1 conv (arange(16) weights)": "fp16",
                  "anchor / stride box decode (dist2bbox, * stride)": "fp16",
                  "class-score sigmoid": "fp32",
                  "PSA attention softmax": "fp16",
                  "max pool (SPPF), nearest upsample, concat": "int8 data movement"},
    "audit": {"allow": {f"convolution {I8}": "backbone, neck, PSA qkv/pe/proj/ffn, head convs "
                                              "except the final ones",
                        "convolution f16xf16->f16": "Detect cv2.i.2 / cv3.i.2 and the DFL conv",
                        f"dot_general {I8}": "C2PSA attention q^T k and v attn^T"},
              "require": [f"convolution {I8}", "convolution f16xf16->f16", f"dot_general {I8}"],
              # 174 convs: exactly the 6 final head convs + DFL fp16, all others int8;
              # 2 PSABlocks x (q^T k, v attn^T)
              "counts": {"full": {"forward": {f"convolution {I8}": 167, "convolution f16xf16->f16": 7,
                                              f"dot_general {I8}": 4}}},
              "params": {"int8": "conv weights", "float16": "final head convs + DFL weights"}},
    "notes": "Confirms the research summary (head sigmoid fp32; final head convs + DFL fp16) and "
             "pins the source to engine.py (not exporter.py) at the provenance commit. The NMS-free "
             "one2one head does not exist for YOLO11 (end2end unset). The PSA attention matmuls "
             "(2 PSABlocks, ~0.25 GFLOP of 87) are taken as INT8: the TRT 11 / ModelOpt path "
             "quantizes MatMul, and in implicit mode both operands carry calibrated ranges; TensorRT "
             "may still pick an fp16 tactic for them (open doubt; the audit requires int8). MLPerf "
             "v6.0 edge YOLO submissions' precision could not be checked (GitHub API rate limit; "
             "NVIDIA's v6.0 closed code has no yolo directory). Network output fp32 (concat of fp16 "
             "boxes and fp32 scores); NMS host-side.",
}
