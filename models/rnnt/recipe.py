"""RNN-T quantization recipe (see models/quant_recipes.py for the fields)."""

I8 = "i8xi8->i32"
F16 = "f16xf16->f16"
F32 = "f32xf32->f32"

_NV = "mlcommons/inference_results_v4.0 closed/NVIDIA/code/rnnt/tensorrt"
_IN = "mlcommons/inference_results_v4.0 closed/Intel/code/rnnt/pytorch-cpu"

RECIPE = {
    "recipe": "MLPerf v4.0 NVIDIA: INT8 encoder LSTMs, FP16 prediction network and joint",
    "basis": "mlperf",
    "sources": [
        f"{_NV}/rnnt_network.py RNNTEncoder.add_rolled_rnns: RNNTEncoderPlugin dataType "
        "trt.DataType.INT8, weightsI / weightsH / bias passed as fp16, scaleFactors = 1 / "
        "calibrated range of the stack input (`input` for pre_rnn, `encoder_reshape` for post_rnn)",
        f"{_NV}/calibrator.py (IInt8MinMaxCalibrator) + calibrator.cache "
        "(TRT-7200-MinMaxCalibration: input, per-layer output / hidden / cell ranges)",
        f"{_NV}/rnnt_network.py RNNTDecoder: RNNTDecoderPlugin dataType trt.DataType.HALF (fp16 "
        "weights); RNNTJointFC1 / RNNTJointBackend: plain FC layers, fc2 weights dumped fp16 "
        "for fc2_top1_ker",
        "mlcommons/inference_results_v4.0 closed/NVIDIA/configs/rnnt/__init__.py GPUBaseConfig: "
        "precision fp16, input_dtype fp16 (the INT8 part is the encoder plugin only)",
        "mlcommons/inference_results_v4.0 closed/NVIDIA/documentation/calibration.md: plugins "
        "quantize weights per-channel (per-row) max-abs; activations per-tensor",
        f"{_IN}/models/quant_lstm.py iLSTMLayer / modeling_rnnt.py: Intel's int8 encoder LSTM "
        "(one weight scale per layer, [x, h] share one static scale, c fp16, pre_rnn output int8) "
        "with bf16 prediction network and joint",
    ],
    "weights": ("encoder LSTMs: int8, per-output-row symmetric max-abs (W_ih and W_hh); "
                "prediction LSTM, embedding and joint: fp16"),
    "activations": ("encoder LSTM GEMM inputs int8, per-tensor static symmetric: the layer input "
                    "x (min-max calibrated; stack input `input` / `encoder_reshape`) and the "
                    "hidden state h (|h| <= 1, calibrated range ~1)"),
    "accumulator": "int32 (fp32 epilogue: scale, b_ih + b_hh)",
    "storage": "encoder hidden state h int8 between steps and layers; cell state c fp16",
    "float_ops": {"preprocessing (STFT, mel matmul, log, normalization; DALI)": "fp32",
                  "encoder gate nonlinearities and cell update": "fp16",
                  "prediction network (embedding + 2-layer LSTM)": "fp16",
                  "joint (FC1, ReLU, FC2)": "fp16"},
    "audit": {"allow": {f"dot_general {I8}": "encoder LSTM input projections and recurrent steps",
                        f"dot_general {F16}": "prediction LSTM, joint",
                        f"dot_general {F32}": "preprocess mel filterbank matmul"},
              "require": [f"dot_general {I8}", f"dot_general {F16}"],
              "params": {"int8": "encoder W_ih / W_hh",
                         "float16": "prediction LSTM / embedding, joint"}},
    "notes": ("Confirms the split (encoder INT8, prediction / joint 16-bit float) of both "
              "submitters; picks NVIDIA (GPU, the larger deployment). Correction to the research "
              "summary: NVIDIA's plugin receives fp16 weights plus one input scale per LSTM stack "
              "and dataType INT8, so it quantizes the weights itself (NVIDIA calibration.md: "
              "per-row max-abs) and runs int8 GEMMs; it is not 'fp16 weights with int8 "
              "activations', and no cell-state scale is passed (c stays float). The plugin source "
              "is not published, so the h scale (taken from the calibrated hidden ranges, ~1/127) "
              "and the fp16 gate math are inferred; Intel's published kernel matches this shape "
              "(int8 x and h, float c) but uses one per-layer weight scale and bf16 elsewhere. "
              "Stage 2: the in_proj phase quantizes the layer input (pre_l0: calibrated feature "
              "scale; others: the previous layer's int8 h, StackTime on int8); the step phase "
              "takes int8 h and fp16 c and returns them; prediction / joint params and "
              "activations fp16; preprocessing unchanged (fp32). Encoder biases are stored "
              "fp32 (the plugin receives fp16) and added in the fp32 epilogue; the last "
              "encoder layer's step also returns h in fp16 (the encoder output)."),
}
