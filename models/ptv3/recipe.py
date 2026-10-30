"""Point Transformer V3 (nuScenes semseg) quantization recipe (see models/quant_recipes.py for the fields)."""

I8 = "i8xi8->i32"
# Exact op counts (both configs: same architecture, only capacities differ):
#   int8: stem 125 (centre + 124 offsets) + 22 blocks x (27 CPE offsets + CPE Linear + qkv +
#         proj + fc1 + fc2 = 32) + 4 pooling proj + 8 unpooling proj / proj_skip + seg head = 842
#   fp16: 22 blocks x (QK^T + AV) = 44
COUNTS = {f"dot_general {I8}": 125 + 22 * 32 + 4 + 8 + 1, "dot_general f16xf16->f32": 2 * 22}

RECIPE = {
    "recipe": "INT8 W8A8 platform fallback (automotive): int8 Linears and sparse convs, "
              "fp16 attention core (flash-attention numerics)",
    "basis": "fallback",
    "sources": [
        "autowarefoundation/autoware_universe perception/autoware_ptv3/config/ptv3.param.yaml: "
        "trt_precision fp16 (schema enum fp16 / fp32 / strongly-typed, no int8); "
        "autoware_tensorrt_plugins implicit_gemm_plugin.cpp (sparse conv): kFLOAT / kHALF only",
        "huggingface.co/AutowareFoundation/ptv3 model card: TensorRT, FP16 by default",
        "Pointcept/Pointcept README and point_transformer_v3m1_base.py SerializedAttention: "
        "enable_flash runs attention in fp16 / bf16 (flash_dtype), upcast_attention / "
        "upcast_softmax options; no quantization code",
        "huggingface.co/Pointcept/PointTransformerV3 nuscenes-semseg-pt-v3m1-0-base/config.py: "
        "enable_flash=True (the released model's attention numerics)",
    ],
    "weights": "int8 per-output-channel symmetric on every Linear (qkv, proj, fc1, fc2, CPE "
               "Linear, pooling / unpooling proj, seg head) and every SubMConv3d (stem k5, CPE k3); "
               "BN folded into the preceding Linear / conv",
    "activations": "int8 per-tensor static symmetric (calibrated)",
    "accumulator": "int32 (fp32 epilogue: scale, bias, residual add)",
    "storage": None,
    "float_ops": {"attention QK^T, AV (fp16 operands, fp32 accumulation)": "fp16",
                  "softmax": "fp32",
                  "LayerNorm, GELU, residual adds": "fp16",
                  "segment max (pooling), patch / rulebook gathers and scatters": "data movement",
                  "seg_logits": "fp16"},
    "audit": {"allow": {f"dot_general {I8}": "Linears and sparse-conv offset GEMMs",
                        "dot_general f16xf16->f32": "attention QK^T and AV per patch"},
              "require": [f"dot_general {I8}", "dot_general f16xf16->f32"],
              "phases": {"forward": {"allow": {f"dot_general {I8}": "Linears, SubMConv3d",
                                               "dot_general f16xf16->f32": "attention core"},
                                     "require": list(COUNTS)}},
              "counts": {"full": {"forward": COUNTS}, "reduced": {"forward": COUNTS}},
              "params": {"int8": "Linear and SubMConv3d kernels"}},
    "notes": "No quantized PTv3 exists: Pointcept ships none, Autoware's TensorRT deployment is "
             "fp16 only (no int8 precision option; its sparse-conv plugin is fp32/fp16), and no "
             "NVIDIA INT8/FP8 PTv3 was found, so the automotive INT8 fallback applies. Departure "
             "from the research summary: the CPE SubMConv3d of every block is INT8 too, not only "
             "the embedding stem (they are the bulk of the GEMM work, 27 C^2 per point vs 12 C^2 "
             "for the Linears, and INT8 SubM conv is the deployed sparse-conv practice in the same "
             "domain, NVIDIA libspconv / CUDA-CenterPoint). Attention stays fp16 with fp32 "
             "accumulation and softmax, matching the released model's flash-attention numerics. "
             "Open doubt: the stem's input features are raw coordinates (m) + strength; a "
             "per-tensor int8 scale over the +-54 m range is coarse (CenterPoint's recipe keeps "
             "its first sparse conv fp16), but the fallback has no carve-out evidence for PTv3 "
             "(coordinator decision: stem int8). BatchNorm folding: the stem BN and the four "
             "unpooling proj / proj_skip BNs fold into the Linear / conv before them; the four "
             "pooling BNs follow a segment max and stay fp16 elementwise ops (a negative BN "
             "scale does not commute with max). Sparse-conv offset partial sums accumulate in "
             "int32 (scatter-add in int32), one fp32 epilogue per conv. Full config per call: "
             "int8 GEMMs 3200.1 GFLOP (84.8%), fp16 attention 575.0 GFLOP (15.2%).",
}
