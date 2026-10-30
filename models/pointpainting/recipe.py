"""PointPainting (MLPerf Automotive) quantization recipe (see models/quant_recipes.py for the fields)."""

I8 = "i8xi8->i32"
MA = "mlcommons/mlperf_automotive@58100ef automotive/3d-object-detection"

RECIPE = {
    "recipe": "INT8 W8A8 platform fallback (automotive, TensorRT INT8) with the PointPillars "
              "mixed-precision carve-outs (PFN, conv_reg, one stage-0 conv in fp16)",
    "basis": "fallback",
    "sources": [
        f"{MA}/README.md: PointPainting reference 0.5425 mAP (Waymo), precision fp32; "
        "model/painter.py deeplabv3plus_resnet50(19, OS16); model/pointpillars_core.py",
        "mlcommons/automotive_results_v0.5 summary_results.json: no PointPainting submission "
        "(NVIDIA submitted only bevformer); inference_results_v5.0 / v5.1: the only pointpainting "
        "rows are fp32 (GATEOverflow, RTX 4090); v6.0: none",
        "NVIDIA-AI-IOT/CUDA-PointPillars main tool/build_trt_engine.sh: trtexec --fp16 (no INT8 "
        "path); src/pointpillar/pointpillar-scatter.cpp: PPScatterPlugin kFLOAT / kHALF only",
        "NVIDIA TAO Toolkit PointPillars trtexec page: 'PointPillars currently does not support "
        "INT8 calibration'",
        "arXiv 2601.12638 (Fuengfusin et al., Mixed Precision PointPillars with TensorRT), "
        "Table III: KITTI mAP fp32 64.64, full INT8 PTQ 13.12, PTQ with fp16 on "
        "voxel_encoder.pfn_layers.0.linear, bbox_head.conv_reg, backbone.blocks.0.3: 63.84; "
        "symmetric min-max calibration (ModelOpt + trtexec)",
        "TensorRT operator docs (Resize, GridSample, SoftMax: float16 / float32 / bfloat16 only): "
        "the bilinear upsamples stay fp16",
    ],
    "weights": "int8 per-output-channel symmetric (BN folded) on every segmenter conv and every "
               "PointPillars conv / deconv except the three fp16 carve-outs",
    "activations": "int8 per-tensor static symmetric (TensorRT calibration); fp32 normalized "
                   "image quantized by the segmenter's first conv; pillar features fp16 into the PFN",
    "accumulator": "int32 (fp32 epilogue: scale, folded-BN bias, ReLU, residual add)",
    "storage": None,
    "float_ops": {"PFN Conv1d(16, 64) (pillar_encoder/conv) + BN + ReLU + max over points": "fp16",
                  "pillar scatter (PPScatterPlugin) onto the BEV canvas": "fp16",
                  "backbone multi_blocks_0_3 (second 3x3 conv of stage 0)": "fp16",
                  "head conv_reg (box regression 1x1)": "fp16",
                  "bilinear upsampling (ASPP pooling broadcast, decoder x4, logits to input size)": "fp16",
                  "ASPP branch outputs, decoder concat, neck outputs (quantized after concat)": "fp16",
                  "segmenter logits (network output)": "fp32",
                  "PFN feature construction (offsets, mean, mask)": "fp16",
                  "max pool, concat": "int8 data movement"},
    "audit": {"allow": {f"convolution {I8}": "segmenter: all 62 convs; pointpillars: backbone, "
                                              "neck deconvs (as 1x1 convs), conv_cls, conv_dir_cls",
                        "convolution f16xf16->f16": "PFN conv, multi_blocks_0_3, conv_reg"},
              "require": [f"convolution {I8}", "convolution f16xf16->f16"],
              "phases": {"segmenter": {"allow": [f"convolution {I8}"],
                                       "require": [f"convolution {I8}"]},
                         "segmenter_side": {"allow": [f"convolution {I8}"],
                                            "require": [f"convolution {I8}"]},
                         "pointpillars": {"allow": [f"convolution {I8}", "convolution f16xf16->f16"],
                                          "require": [f"convolution {I8}",
                                                      "convolution f16xf16->f16"]}},
              # Segmenter: ResNet-50 53 (conv1 + 16 x 3 + 4 downsample) + head 9 (low-level
              # project, ASPP 1x1 + 3 atrous + pooling 1x1 + project, classifier 3x3 + 1x1).
              # PointPillars: fp16 = PFN conv + multi_blocks_0_3 + conv_reg; int8 = backbone
              # (4 + 6 + 6) - 1 + neck 3 + head cls / dir 2 = 20. Same structure in both configs.
              "counts": {cfg: {"segmenter": {f"convolution {I8}": 62},
                               "segmenter_side": {f"convolution {I8}": 62},
                               "pointpillars": {f"convolution {I8}": 20,
                                                "convolution f16xf16->f16": 3}}
                         for cfg in ("full", "reduced")}},
    "notes": "No MLPerf or production INT8 PointPainting exists (MLPerf: fp32 only; NVIDIA "
             "CUDA-PointPillars and TAO PointPillars are fp16, TAO states INT8 calibration is "
             "unsupported), so the automotive INT8 fallback applies. Departure from the research "
             "summary: the documented carve-out is not 'the first block' but one conv, "
             "backbone.blocks.0.3 in mmdet3d naming = the second 3x3 conv of stage 0 "
             "(multi_blocks_0_3 here, same Sequential indexing); the head's cls / dir convs stay "
             "INT8. That carve-out set comes from an academic TensorRT study on KITTI (mmdet3d "
             "PointPillars, unpainted), mapped onto the MLPerf module names; it is the only "
             "published layer list. The DeepLabV3+ segmenter has no standard INT8 deployment "
             "either; it follows the generic TensorRT INT8 fallback (ResNet-50 recipe: per-channel "
             "weights, BN folded, per-tensor activations), with TensorRT's Resize in fp16. "
             "The segmenter's final 1x1 conv writes fp16 logits, upsampled in fp16 (Resize) and "
             "returned as fp32 for the host softmax / painting. The neck ConvTranspose2d (k = s, "
             "non-overlapping) is computed as a 1x1 INT8 conv to s^2 * Cout channels + "
             "depth-to-space (NVIDIA's 3D U-Net rewrite; same math), which removes the zero work "
             "of the dilated-input lowering (pointpillars 517.6 -> 280.8 GFLOP). Concatenated "
             "branches (ASPP, decoder, neck) are produced in fp16 and quantized once after the "
             "concat (TensorRT gives concat inputs one common scale). ASPP pooling and max pool "
             "run on INT8 tensors.",
}
