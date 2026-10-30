"""SDXL quantization recipe (see models/quant_recipes.py for the fields)."""

F8 = "f8E4M3FNxf8E4M3FN->f32"
_NV = "mlcommons/inference_results_v5.0 closed/NVIDIA/code/stable-diffusion-xl"

RECIPE = {
    "recipe": "MLPerf NVIDIA Hopper: FP8 UNet (W8A8 + FP8 MHA), INT8 VAE convs, FP16 CLIP (TensorRT)",
    "basis": "mlperf",
    "sources": [
        f"{_NV}/modelopt/utils.py SDXL_FP8_CFG (E4M3 per-tensor W / A / q,k,v_bmm / softmax, "
        "algorithm max, trt_high_precision_dtype Half), filter_func_unet (fp16 carve-outs), "
        "get_int8_config (VAE: W axis 0, Conv A per-tensor), filter_func (VAE carve-outs), quantize_lvl",
        f"{_NV}/modelopt/quantize_unet.py (set_fmha -> AttnProcessor, mtq.quantize, 500 captions x 20 "
        "steps), export_unet.py; quantize_vae.py (int8, 64 captions, get_int8_config(pipe.vae)), "
        "export_vae.py (quant_level 3.0, filter_func, QDQ in fp32)",
        f"{_NV}/tensorrt/builder.py SDXLCLIPBuilder (fp16 if sm < 100), SDXLUNetXLBuilder (fp8, "
        "strongly typed), SDXLVAEBuilder (int8 strongly typed unless SoC or sm >= 100)",
        f"{_NV}/tensorrt/sdxl_graphsurgeon.py insert_fp8_mha_cast (TRT FP8 MHA pattern), "
        "insert_vae_fp16_conv_cast / insert_vae_fp16_linear_casts (VAE fp16 islands), convert_fp16_io",
        f"{_NV}/tensorrt/download_model.sh ('Datacenter: Use FP8 Unet and INT8 VAE engine'), "
        "tensorrt/README.md ('UNet FP8-FP16, VAE INT8-FP32 on H200'), configs/stable-diffusion-xl "
        "(H100/H200/GH200 precision fp8). Same modelopt/ code and VAE builder in v4.1 and v5.1.",
        "NVIDIA/Model-Optimizer modelopt/torch/quantization/plugins/diffusion/diffusers.py "
        "_QuantAttention (_quantized_baddbmm: q,k quantizers; _quantized_bmm: softmax, v quantizers)",
    ],
    "weights": "UNet: fp8 E4M3 per-tensor; VAE: int8 per-output-channel symmetric; CLIP: fp16",
    "activations": "UNet: fp8 E4M3 per-tensor static (max calibration); VAE: int8 per-tensor static "
                   "symmetric (max); FP8 attention: Q, K, V and softmax probabilities per-tensor static",
    "accumulator": "fp32 (FP8 GEMM / conv / MHA BMMs), int32 (VAE int8 convs)",
    "storage": None,
    "float_ops": {
        "UNet carve-outs (fp16): time_embedding, add_embedding, every time_emb_proj, conv_in, conv_out, "
        "every conv_shortcut, up_blocks.2.resnets.2.conv1/conv2": "fp16",
        "UNet GroupNorm / LayerNorm / SiLU / GEGLU gelu / residuals, FP8 GEMM epilogues": "fp16",
        "UNet attention softmax (fused FP8 MHA; output quantized to fp8)": "fp32",
        "UNet timestep sinusoids": "fp32 (cast to fp16)",
        "CLIP-L and CLIP-bigG text encoders (whole phases)": "fp16",
        "VAE fp16 islands: conv_in, mid_block resnet convs, up_blocks.2/3 resnets.1-2 convs, mid-block "
        "attention to_q/k/v/to_out.0 and both BMMs, GroupNorm affine of up_blocks / conv_norm_out": "fp16",
        "VAE post_quant_conv, up_blocks.3.resnets.0 conv1/conv2/conv_shortcut, conv_out, GroupNorm "
        "statistics, softmax, SiLU, residual adds, int8 conv epilogues": "fp32",
    },
    "audit": {
        "allow": {},     # every phase has its own list below
        "phases": {
            "text_encoder_l": {"allow": {"dot_general f16xf16->f16": "fp16 engine (linears + MHA BMMs)"},
                               "require": ["dot_general f16xf16->f16"]},
            "text_encoder_g": {"allow": {"dot_general f16xf16->f16": "fp16 engine (linears + MHA BMMs)"},
                               "require": ["dot_general f16xf16->f16"]},
            "unet_step": {
                "allow": {f"dot_general {F8}": "transformer linears (proj_in/out, to_q/k/v/out, ff) "
                                               "and FP8 MHA BMMs (QK^T, PV)",
                          f"convolution {F8}": "resnet conv1/conv2, down/upsampler convs",
                          "dot_general f16xf16->f16": "time_embedding, add_embedding, time_emb_proj",
                          "convolution f16xf16->f16": "conv_in, conv_out, conv_shortcut, "
                                                      "up_blocks.2.resnets.2.conv1/conv2"},
                "require": [f"dot_general {F8}", f"convolution {F8}", "dot_general f16xf16->f16",
                            "convolution f16xf16->f16"]},
            "vae_decode": {
                "allow": {"convolution i8xi8->i32": "up_blocks.0-1 convs, up_blocks.2.resnets.0, upsamplers",
                          "convolution f16xf16->f16": "conv_in, mid_block resnets, up_blocks.2/3 resnets.1-2",
                          "convolution f32xf32->f32": "post_quant_conv, up_blocks.3.resnets.0, conv_out",
                          "dot_general f16xf16->f16": "mid-block attention linears and BMMs"},
                "require": ["convolution i8xi8->i32", "convolution f16xf16->f16",
                            "convolution f32xf32->f32", "dot_general f16xf16->f16"]},
        },
        "require": [],
        # full: 758 FP8 UNet layers (722 linears + 36 convs) + 140 attentions x 2 FP8 BMMs;
        # 36 fp16 carve-outs (21 linears + 15 convs); VAE 18 INT8 / 13 fp16 / 5 fp32 convs.
        "counts": {
            "full": {"text_encoder_l": {"dot_general f16xf16->f16": 96},
                     "text_encoder_g": {"dot_general f16xf16->f16": 257},
                     "unet_step": {f"dot_general {F8}": 1002, f"convolution {F8}": 36,
                                   "dot_general f16xf16->f16": 21, "convolution f16xf16->f16": 15},
                     "vae_decode": {"convolution i8xi8->i32": 18, "convolution f16xf16->f16": 13,
                                    "convolution f32xf32->f32": 5, "dot_general f16xf16->f16": 6}},
            "reduced": {"text_encoder_l": {"dot_general f16xf16->f16": 24},
                        "text_encoder_g": {"dot_general f16xf16->f16": 25},
                        "unet_step": {f"dot_general {F8}": 260, f"convolution {F8}": 36,
                                      "dot_general f16xf16->f16": 21, "convolution f16xf16->f16": 14},
                        "vae_decode": {"convolution i8xi8->i32": 18, "convolution f16xf16->f16": 13,
                                       "convolution f32xf32->f32": 4, "dot_general f16xf16->f16": 6}},
        },
        "params": {"float8_e4m3fn": "UNet FP8 weights", "int8": "VAE INT8 conv weights",
                   "float16": "fp16 engine weights (CLIP, UNet carve-outs and norms, VAE islands)"},
    },
    "notes": (
        "Platform: Hopper datacenter (H100/H200/GH200), NVIDIA MLPerf v4.1-v5.1 (same modelopt code). "
        "B200 runs FP8 UNet with fp32 CLIP and fp32 VAE; Orin/L40S run an INT8 UNet. "
        "Departures from the research summary: (1) the VAE INT8 path is real on Hopper (doubt "
        "resolved by builder.py / download_model.sh), but it is not SmoothQuant in effect: "
        "quantize_vae.py calls get_int8_config(pipe.vae) without --alpha (alpha 0.8, not the 1.0 "
        "passed), SmoothQuant only touches Linear layers, and every VAE Linear is in mid_block, which "
        "filter_func disables -- so the deployed VAE is plain INT8 W8A8 on 18 convs (per-output-channel "
        "weights, per-tensor max-calibrated activations); (2) the VAE is a mixed fp32/fp16/int8 graph: "
        "the ONNX is fp32 (export_vae.py 'QDQ needs to be in FP32'), graphsurgeon casts 13 convs and the "
        "mid-block attention to fp16, 5 convs stay fp32; (3) UNet I/O is fp16 (convert_fp16_io), VAE "
        "I/O fp32, CLIP takes int32 ids and returns fp16; (4) UNet float dtype is fp16 "
        "(trt_high_precision_dtype Half), not bf16; (5) FP8 attention covers both attn1 and attn2 "
        "(cross-attention) of every transformer block. ONNX inserts a fp16 cast around the softmax; "
        "the TRT fused FP8 MHA kernel computes it in fp32 -- modelled as fp32. "
        "Calibration: UNet 500 MLPerf calibration captions x 20 steps, VAE 64 captions. "
        "Port form: FP8 / INT8 biases are added in the fp32 epilogue (models/quant.py) where TensorRT adds them after the DQ in fp16; CLIP MHA softmax is fp32 (fused kernel)."
    ),
}
