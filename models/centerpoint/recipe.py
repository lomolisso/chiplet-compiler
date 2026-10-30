"""CenterPoint (nuScenes, voxel 0.075) quantization recipe (see models/quant_recipes.py for the fields)."""

I8 = "i8xi8->i32"
CP = "NVIDIA-AI-IOT/Lidar_AI_Solution master CUDA-CenterPoint"

RECIPE = {
    "recipe": "NVIDIA CUDA-CenterPoint INT8 (libspconv INT8 sparse backbone from the PTQ/QAT "
              "recipe + TensorRT INT8 RPN / head)",
    "basis": "production",
    "sources": [
        f"{CP}/qat/tools/sparseconv_quantization.py: SparseConvolutionQunat weights "
        "QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL (axis 0 = Cout of spconv KRSC), inputs "
        "QuantDescriptor(num_bits=8, calib_method='histogram'), compute_amax(method='mse'); "
        "QuantAdd with ONE per-tensor _input_quantizer shared by both add inputs, inserted in "
        "every SparseBasicBlock",
        f"{CP}/qat/onnx_export/export-scn.py lines 44-68: BN folded (funcs.layer_fusion_bn), ReLU "
        "fused; every layer precision int8 / output int8, except conv_input, conv1[0].conv1, "
        "conv1[0].conv2 precision + output fp16, conv1[0].quant_add precision fp16 output int8, "
        "extra_conv output fp16",
        f"{CP}/README.md: det3d nusc_centerpoint_voxelnet_0075voxel_fix_bn_z; timing table "
        "Voxelization FP32, 3D Backbone FP16, RPN + Head FP16/INT8 (A30 3.8/2.5 ms, Orin 11.3/7.0), "
        "Decode + NMS FP16",
        f"{CP}/tool/build.trt.sh: RPN + head engine trtexec --fp16, fp16 I/O (the default build)",
        "NVIDIA-AI-IOT/Lidar_AI_Solution libraries/3DSparseConvolution/README.md: "
        "centerpoint.scn.PTQ.onnx INT8 59.15 mAP / 66.45 NDS vs FP16 59.5 / 66.71",
    ],
    "weights": "int8 per-output-channel symmetric (BN folded) on every sparse conv except the "
               "first three (fp16) and on every RPN / head conv",
    "activations": "int8 per-tensor static symmetric (histogram calibration, MSE amax); "
                   "residual adds quantized with one shared per-tensor scale for both inputs",
    "accumulator": "int32 (fp32 epilogue: scale, folded-BN bias, residual add, ReLU)",
    "storage": None,
    "float_ops": {"MeanVFE (voxel point mean)": "fp32",
                  "conv_input (SubMConv3d 5 -> 16) + BN + ReLU": "fp16",
                  "conv1_0 (first SparseBasicBlock: conv1, conv2, add): compute fp16, add output int8": "fp16",
                  "conv_out_0 (extra_conv) output and HeightCompression BEV canvas": "fp16",
                  "rulebook gathers / scatters (int8 rows); strided-spconv scatter-add of the "
                  "int32 partial sums": "data movement",
                  "head outputs (center, center_z, dim, rot, vel, hm logits)": "fp16"},
    "audit": {"allow": {f"dot_general {I8}": "sparse conv offset GEMMs from conv1_1 on",
                        "dot_general f16xf16->f16": "conv_input and conv1_0.conv1/conv2 offset GEMMs",
                        f"convolution {I8}": "BEV backbone, deblocks (1x1 + depth-to-space), "
                                             "shared conv, separate heads"},
              "require": [f"dot_general {I8}", "dot_general f16xf16->f16", f"convolution {I8}"],
              "phases": {"forward": {"allow": [f"dot_general {I8}", "dot_general f16xf16->f16",
                                               f"convolution {I8}"],
                                     "require": [f"dot_general {I8}", "dot_general f16xf16->f16",
                                                 f"convolution {I8}"]}},
              # Exact op counts (same architecture in both configs; only extents shrink).
              # A SubMConv3d lowers to 27 GEMMs (centre + 26 offsets), a strided SparseConv3d to
              # kv GEMMs (27, or 3 for conv_out's (3, 1, 1) kernel).
              #   fp16: conv_input 27 + conv1_0 (2 SubM) 54                           = 81
              #   int8: conv1_1 (2 SubM) 54 + levels 2..4 x (strided 27 + 2 blocks x 2 SubM x 27
              #         = 135) 405 + conv_out 3                                        = 462
              #   int8 convs: BEV 2 levels x (1 + 5) 12 + 2 deblocks + shared conv 1
              #         + 6 heads x 6 sub-heads x 2 convs 72                          = 87
              "counts": {cfg: {"forward": {f"dot_general {I8}": 462,
                                           "dot_general f16xf16->f16": 81,
                                           f"convolution {I8}": 87}}
                         for cfg in ("full", "reduced")}},
    "notes": "Confirms the research summary for the sparse backbone (per-channel weights, "
             "histogram/MSE activations, quantized residual adds) and pins the carve-outs: the "
             "first conv AND the whole first SparseBasicBlock run fp16 (its add writes int8 for "
             "conv1_1); the last strided conv computes int8 but writes fp16 into the dense BEV map. "
             "CUDA-CenterPoint's model is det3d SpMiddleResNetFHD + RPN + CenterHead, layer for "
             "layer the same as this OpenPCDet VoxelResBackBone8x / BaseBEVBackbone / CenterHead "
             "(conv_input, conv1..4, extra_conv = conv_out); the recipe is applied by layer "
             "position. Port form: BN folded everywhere; the two BEV deblocks (ConvTranspose2d "
             "k = s = 1, 2, non-overlapping) are INT8 1x1 convs to s^2 * Cout channels + "
             "depth-to-space (the rewrite NVIDIA uses for MLPerf 3D U-Net; no zero-dilated "
             "input); the BEV concat is requantized with one scale; QuantAdd requantizes conv2's "
             "output and the identity to its one scale before the add. Open doubts: (1) RPN + "
             "head: the README times an INT8 RPN + head engine but ships no calibration recipe "
             "and the default build is --fp16; INT8 (per-channel weights, per-tensor activations, "
             "fp16 head outputs) is the coordinator's decision (every model quantized); fp16 RPN "
             "would be the alternative. (2) The shipped default pipeline loads the non-PTQ SCN ONNX (fp16); "
             "the INT8 SCN is the PTQ / QAT export. (3) libspconv is closed source; its INT8 "
             "accumulator is assumed int32. Decode / NMS fp16 on the host (excluded).",
}
