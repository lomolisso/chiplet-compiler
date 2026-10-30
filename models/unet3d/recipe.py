"""MLPerf 3D U-Net (KiTS19) quantization recipe (see models/quant_recipes.py for the fields)."""

I8 = "i8xi8->i32"
NV = "mlcommons/inference_results_v5.0 closed/NVIDIA/code"

RECIPE = {
    "recipe": "TensorRT INT8 W8A8 (MLPerf Inference NVIDIA, 99% and 99.9%)",
    "basis": "mlperf",
    "sources": [
        f"{NV}/3d-unet/tensorrt/README.md (INT8 meets both the 99% and 99.9% targets; "
        "ConvTranspose 2x2x2/2 -> 1x1x1 conv with 8x output channels + PixelShuffle3D; plugins "
        "INSTNORM3D (InstanceNorm + ReLU fused), CONV3D3X3X3C1K32 (first conv), CONV3D1X1X1K4 (last conv))",
        f"{NV}/3d-unet/tensorrt/builder.py: input tensor INT8 LINEAR with the calibrated range; output "
        "FP16 LINEAR; BuilderFlag.FP16 fallback; UNet3DKiTS19MinMaxCalibrator (MINMAX_CALIBRATION); "
        "conv_for_deconv weight transpose (k,2,2,2 -> 8k) and bias repeat",
        f"{NV}/plugin/conv3D3X3X3C1K32Plugin/src/conv3d3x3x3C1K32Plugin.cpp: INT8 LINEAR -> INT8 CDHW32, "
        "weights int8 with ONE scale (127 / max element)",
        f"{NV}/plugin/conv3D1X1X1K4Plugin/src/conv3d1x1x1K4Plugin.cu: INT8 CDHW32 -> FP16 LINEAR, weights "
        "int8 with one scale, fp32 bias, scale = s_in * s_w",
        f"{NV}/3d-unet/tensorrt/calibrator.py (IInt8MinMaxCalibrator)",
    ],
    "weights": "int8 symmetric: per-output-channel for the 21 TensorRT convs and the 5 "
               "deconv-as-1x1x1 convs (8*Cout channels); per-tensor for the first (1->32, 3^3) "
               "and the output (32->3, 1^3) conv plugins",
    "activations": "int8 per-tensor static symmetric (min-max calibration); INT8 network input",
    "accumulator": "int32 (fp32 epilogue: scale, bias); the output conv dequantizes to fp16",
    "storage": None,
    "float_ops": {"InstanceNorm3d + ReLU (INSTNORM3D_TRT plugin)": "int8 in / int8 out, fp32 statistics",
                  "network output (logits)": "fp16",
                  "PixelShuffle3D, concat": "int8 data movement"},
    "audit": {"allow": {f"convolution {I8}": "all 23 convs + 5 transposed convs rewritten as "
                                              "1x1x1 convs (+ pixel shuffle)"},
              "require": [f"convolution {I8}"],
              # every conv int8: 23 regular (incl. the first / output plugin convs) + 5 deconvs
              "counts": {cfg: {"forward": {f"convolution {I8}": 28}} for cfg in ("full", "reduced")},
              "params": {"int8": "conv weights"}},
    "notes": "Departures from the research summary: (1) the output layer is NOT fp16 compute: it "
             "is an int8 x int8 conv (CONV3D1X1X1K4 plugin, per-tensor weight scale) whose epilogue "
             "writes fp16; only the logits are fp16. (2) The first and last convs use per-tensor "
             "weight scales (plugins), the rest per-channel. (3) NVIDIA rewrites every "
             "ConvTranspose3d(k2, s2) into a 1x1x1 conv to 8*Cout channels + PixelShuffle3D (exact "
             "same math; the port keeps torch's [Cout] bias and repeats it 8x in the epilogue, NVIDIA "
             "stores it repeated); the port adopts this rewrite (models/unet3d/qops.py), which also removes the 8x zero-dilated-input "
             "FLOP inflation of the current lax.conv_transpose lowering (dense lowering counts "
             "1065.3 GFLOP vs 954.6 GFLOP of useful work per 128^3 ROI). The InstanceNorm plugin "
             "fuses ReLU (attrs relu=1, alpha=0), int8 in and out. Sliding-window Gaussian "
             "blending stays host-side (excluded), as in NVIDIA's harness (fp16 kernel patches). "
             "Quirk (scale value only, no dtype effect): both conv plugins compute the weight scale "
             "from the max element, not max |w|.",
}
