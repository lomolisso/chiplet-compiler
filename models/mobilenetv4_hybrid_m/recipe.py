"""MobileNetV4-Hybrid-Medium quantization recipe (see models/quant_recipes.py for the fields)."""

from models.mobilenetv4_conv_m.recipe import RECIPE as CONV_M

I8 = "i8xi8->i32"

RECIPE = {
    "recipe": "TFLite full-integer INT8 PTQ, incl. Mobile-MQA (mobile fallback, same converter "
              "recipe as mobilenetv4_conv_m)",
    "basis": "fallback",
    "sources": CONV_M["sources"] + [
        "tensorflow/tensorflow master tensorflow/compiler/mlir/lite/ir/tfl_ops.td: TFL_BatchMatMulOp is "
        "QuantizableResult and accepts int8 x int8 operands (so full-integer PTQ emits an int8 "
        "BATCH_MATMUL for q k^T and attn v); TFL_SoftmaxOp is QuantizableResult with a fixed output "
        "range",
    ],
    "weights": CONV_M["weights"] + "; MQA query / key / value / output 1x1 projections and K/V "
               "down-sampling depthwise convs per-output-channel like every conv; layer scale "
               "folded into the preceding conv",
    "activations": CONV_M["activations"] + "; both MQA matmul operands (q, k, v and the int8 "
                   "softmax probabilities) per-tensor static asymmetric",
    "accumulator": CONV_M["accumulator"],
    "storage": None,
    "float_ops": {},
    "audit": {"allow": {f"convolution {I8}": "every conv / depthwise conv incl. MQA projections",
                        f"dot_general {I8}": "MQA q k^T and attn v (BATCH_MATMUL), classifier"},
              "require": [f"convolution {I8}", f"dot_general {I8}"],
              # 8 MQA blocks x (q k^T, attn v) + classifier
              "counts": {"full": {"forward": {f"convolution {I8}": 120, f"dot_general {I8}": 17}}},
              "params": {"int8": "conv / FC weights, pre-norm MUL / ADD constants",
                         "int32": "conv / FC bias, zero points"}},
    "notes": "No quantized deployment of Hybrid-M exists (not in MLPerf Mobile, no owner checkpoint), "
             "so the mobile fallback applies. On the mobile platform the INT8 W8A8 fallback is "
             "realised by the TFLite converter, i.e. the conv_m scheme: per-channel symmetric weights "
             "but ASYMMETRIC per-tensor activations (the brief's generic fallback text says "
             "symmetric; asymmetric is what TFLite emits, and it keeps the two MNv4 variants "
             "consistent). Attention is integer end to end, as the converter does it: int8 "
             "BATCH_MATMUL for q k^T (the scalar 1/sqrt(d) is a constant multiply; in the integer graph "
             "it only changes the query tensor's quantization scale) and for attn v, int8 SOFTMAX (output scale 1/256, zp -128). The research "
             "summary's alternative (softmax and norms float, or w8a16 attention) is not what a "
             "full-integer TFLite build produces, so it is not used. Graph rewrites: BN after a conv "
             "and LayerScale (a per-channel constant MUL after pw_proj / output.proj, no activation "
             "in between) fold into that conv; the pre-attention `norm` BN follows a residual add, "
             "so it stays an int8 per-channel MUL + ADD (elementwise, not a GEMM). Softmax / uint8 "
             "I/O as in conv_m.",
}
