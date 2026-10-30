"""MobileNetV4-Conv-Medium quantization recipe (see models/quant_recipes.py for the fields)."""

I8 = "i8xi8->i32"

TFLITE_MODEL = ("https://mobile.mlcommons-storage.org/app-resources/models/v4_0/tflite/"
                "MobileNetV4-Conv-Large-int8-ptq.tflite")

RECIPE = {
    "recipe": "TFLite full-integer INT8 PTQ (MLPerf Mobile image_classification_v2)",
    "basis": "mlperf",
    "sources": [
        "mlcommons/mobile_app_open master mobile_back_tflite/cpp/backend_tflite/backend_settings/"
        "tflite_settings_android.pbtxt: image_classification_v2 NPU model = " + TFLITE_MODEL +
        " (md5 590a7a88640a18d28b16b6f571cdfc93, MNv4-Conv-L 384^2, TF Model Garden)",
        "that .tflite's flatbuffer, read 2026-10-30: 57 CONV_2D + 45 DEPTHWISE_CONV_2D with int8 per-tensor "
        "asymmetric inputs (nonzero zero points), int8 per-channel weights (zp 0), int32 per-channel bias, "
        "int8 outputs; 23 int8 ADD; 2 int8 MEAN; FULLY_CONNECTED with a per-tensor int8 weight and "
        "int32 bias; int8 SOFTMAX (out scale 1/256, zp -128); uint8 input (QUANTIZE u8->i8) and uint8 "
        "output (QUANTIZE i8->u8); no float op",
        "LiteRT quantization spec, developers.google.com/edge/litert/models/quantization_spec "
        "(activations asymmetric int8, weights symmetric zp 0 in [-127, 127], int32 bias with scale "
        "s_in * s_w, SOFTMAX output fixed (1/256, -128))",
    ],
    "weights": "int8 symmetric (zp 0, [-127, 127]): per-output-channel for every conv and depthwise "
               "conv (BN folded); per-tensor for the classifier (FULLY_CONNECTED)",
    "activations": "int8 per-tensor static asymmetric (calibrated scale + zero point) everywhere; "
                   "uint8 network input and output (one QUANTIZE each)",
    "accumulator": "int32 (int32 bias with zero-point correction folded in; requantized to int8 in "
                   "the epilogue, ReLU as the clamp of the requantization)",
    "storage": None,
    "float_ops": {},
    "audit": {"allow": {f"convolution {I8}": "every conv / depthwise conv incl. conv_head",
                        f"dot_general {I8}": "classifier (per-tensor weight)"},
              "require": [f"convolution {I8}", f"dot_general {I8}"],
              "counts": {"full": {"forward": {f"convolution {I8}": 77, f"dot_general {I8}": 1}}},
              "params": {"int8": "conv / FC weights", "int32": "conv / FC bias, zero points"}},
    "notes": "Matches the research summary. The deployed MLPerf Mobile model is MNv4-Conv-Large; the "
             "same converter recipe is applied to Conv-Medium (no Conv-M int8 artifact exists from the "
             "model owner). Residual adds, global average pool (MEAN) and softmax are int8 ops with "
             "their own output scale / zero point (TFLite computes them in fixed point). The TFLite "
             "graph includes the softmax and a final int8->uint8 QUANTIZE, so Stage 2 moves the "
             "softmax from `excluded` into the forward phase (int8 in / int8 out, scale 1/256, zp "
             "-128) and returns uint8 probabilities; top-k stays host-side. Input: uint8 image, "
             "requantized to int8 (zp shift) before conv_stem.",
}
