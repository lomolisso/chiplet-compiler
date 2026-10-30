"""RT-DETRv2-R50 quantization recipe (see models/quant_recipes.py for the fields)."""

I8 = "i8xi8->i32"
TRT = "docs.nvidia.com/deeplearning/tensorrt/latest/_static/operators"

RECIPE = {
    "recipe": "INT8 W8A8 platform fallback (edge, TensorRT INT8): int8 convs and Linears; "
              "deformable sampling, attention core, softmax, top-k, resize fp16",
    "basis": "fallback",
    "sources": [
        "lyuwenyu/RT-DETR rtdetrv2_pytorch/tools/export_trt.py: BuilderFlag.FP16 only; README: "
        "speed measured with fp16, TensorRT >= 8.5.1 (no official INT8)",
        "huggingface.co/PekingU/rtdetr_v2_r50vd: model.safetensors F32 only",
        f"{TRT}/GridSample.html, SoftMax.html, Resize.html: float16 / float32 / bfloat16 only; "
        "TopK.html: int32 / int64 / float16 / float32 / bfloat16; MatrixMultiply.html: int8 / "
        "fp8 'only supported through explicit quantization'",
        "NVIDIA/TensorRT plugin/multiscaleDeformableAttnPlugin: FP32 / FP16 only",
        "NVIDIA TAO RT-DETR (tao_deploy): INT8 'only supported for the ResNet series of "
        "backbones', implicit IInt8EntropyCalibrator2 (nvidia_tao_deploy/engine/calibrator.py), "
        "no per-layer carve-outs",
    ],
    "weights": "int8 per-output-channel symmetric on every conv (frozen BN / BN folded) and every "
               "Linear (input projections, AIFI q/k/v/o + FFN, enc_output, score / box heads, "
               "decoder self-attn q/k/v/o, value_proj, sampling_offsets, attention_weights, "
               "output_proj, FFN, query_pos_head, bbox_embed, class_embed)",
    "activations": "int8 per-tensor static symmetric (entropy calibration); fp32 image input "
                   "quantized by the first conv",
    "accumulator": "int32 (fp32 epilogue: scale, bias, SiLU / ReLU / GELU, residual add)",
    "storage": None,
    "float_ops": {"self-attention QK^T and AV (AIFI, decoder)": "fp16 operands, fp32 accumulation",
                  "softmax (attention, deformable attention weights)": "fp32",
                  "deformable bilinear sampling (grid_sample) and its weighted sum": "fp16",
                  "LayerNorm": "fp16",
                  "top-k query selection and gathers": "fp16",
                  "nearest 2x upsample (Resize)": "fp16",
                  "sine position embedding, anchors, sigmoid / inverse_sigmoid box refinement": "fp32",
                  "max / avg pool, concat": "int8 data movement"},
    "audit": {"allow": {f"convolution {I8}": "ResNet-50-vd, input projections, CCFM convs",
                        f"dot_general {I8}": "all Linear layers",
                        "dot_general f16xf16->f32": "AIFI and decoder self-attention QK^T / AV"},
              "require": [f"convolution {I8}", f"dot_general {I8}", "dot_general f16xf16->f32"],
              "phases": {"forward": {"allow": {f"convolution {I8}": "", f"dot_general {I8}": "",
                                               "dot_general f16xf16->f32": ""},
                                     "require": [f"convolution {I8}", f"dot_general {I8}",
                                                 "dot_general f16xf16->f32"]}},
              # Exact op counts. Convs: stem 3 + 3 per bottleneck (sum(depths)) + one shortcut
              # per stage + 3 encoder_input_proj + CCFM 2 lateral + 2 downsample + 4 CSP x
              # (conv1, conv2, 3 RepVGG x 2) + 3 decoder_input_proj (full: 3 + 48 + 4 + 3 + 36
              # + 3 = 97). Linears: AIFI 6 per layer (q, k, v, o, fc1, fc2) + query selection
              # 5 (enc_output, enc_score_head, enc_bbox_head x3) + 16 per decoder layer
              # (query_pos_head x2, q/k/v/o, value_proj, sampling_offsets, attention_weights,
              # output_proj, fc1, fc2, bbox_embed x3, class_embed). Attention core: QK^T + AV
              # per self-attention (AIFI layers + decoder layers).
              "counts": {"full": {"forward": {f"convolution {I8}": 97,
                                              f"dot_general {I8}": 6 * 1 + 5 + 16 * 6,
                                              "dot_general f16xf16->f32": 2 * (1 + 6)}},
                         "reduced": {"forward": {f"convolution {I8}": 3 + 3 * 7 + 4 + 3 + 36 + 3,
                                                 f"dot_general {I8}": 6 * 2 + 5 + 16 * 3,
                                                 "dot_general f16xf16->f32": 2 * (2 + 3)}}},
              "params": {"int8": "conv / Linear kernels (per-channel fp32 kernel_scale)"}},
    "notes": "No official or production INT8 RT-DETRv2: upstream and HF ship fp32 weights and an "
             "fp16 TensorRT export; TAO's RT-DETR allows implicit INT8 calibration (ResNet "
             "backbones) without carve-outs; Ultralytics' RT-DETR checkpoints are rtdetr-l/x, not "
             "v2-R50. So the edge INT8 fallback applies, and the float set is what TensorRT "
             "cannot run in INT8: GridSample (the deformable sampling) and the deformable-attention "
             "plugin are fp32/fp16, SoftMax / Resize have no int8 type, TopK has no int8 type, and "
             "activation-activation MatMuls are INT8 only with explicit Q/DQ, which the fallback "
             "does not insert. Confirms the research summary. Open doubts: sampling_offsets and "
             "attention_weights (sub-pixel offsets) and the box / class heads are INT8 by the "
             "fallback rule; no source documents keeping them fp16. TensorRT 11 is strongly "
             "typed (explicit Q/DQ only, Ultralytics engine.py), so a current build would carry "
             "the same int8 layers as ModelOpt Q/DQ nodes.",
}
