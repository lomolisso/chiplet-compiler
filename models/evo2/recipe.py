"""Evo 2 7B quantization recipe (see models/quant_recipes.py for the fields)."""

F8 = "f8E4M3FNxf8E4M3FN->f32"

RECIPE = {
    "recipe": "Vortex + Transformer Engine FP8 Hyena input projections (official evo2_7b config)",
    "basis": "official",
    "sources": [
        "ArcInstitute/evo2 @ 53f1959 evo2/configs/evo2-7b-1m.yml (CONFIG_MAP['evo2_7b']): "
        "use_fp8_input_projections: True",
        "ArcInstitute/evo2 @ 53f1959 evo2/models.py load_evo2_model (keeps the flag when "
        "transformer_engine imports; warns and falls back to bf16 for 7B models without TE) "
        "and README.md ('Full install': TE + flash-attn; 7B 'can run in bfloat16 without TE')",
        "Zymrael/vortex @ 8b00afe vortex/model/model.py ParallelGatedConvBlock: `projections = "
        "TELinear(d, 3d, bias=qkv_proj_bias, use_fp8=use_fp8_input_projections)`, "
        "pad_to_multiple(16) of the projection input when FP8 is on; out_filter_dense, MLP "
        "l1/l2/l3, Wqkv, out_proj are torch nn.Linear (bf16)",
        "Zymrael/vortex @ 8b00afe vortex/model/layers.py set_format_recipe: "
        "DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo='max') "
        "(margin 0 default); TELinear.forward under te.fp8_autocast; params_dtype bf16",
        "arcinstitute/evo2_7b @ bda0089 evo2_7b.pt (data.pkl read via range requests): "
        "blocks.<i>.projections._extra_state for the 27 Hyena blocks = TE fp8 meta with "
        "scale_fwd / scale_inv_fwd [3] (input, weight, output), amax_history_fwd [16, 3], "
        "fp8_max_fwd 448 (E4M3), fp8_max_bwd 57344 (E5M2); e.g. block 0 input amax 15.56, "
        "weight amax 0.2295",
    ],
    "weights": "fp8 E4M3, one per-tensor scale per projection (TE casts the bf16 master weight "
               "each forward with its delayed scale = 448 / max(amax history); the port "
               "stores the FP8 weight + `kernel_scale`, same GEMM)",
    "activations": "fp8 E4M3, per-tensor, delayed scaling (scale from the 16-step amax history "
                   "of earlier calls, not from the current tensor): a per-tensor scale input "
                   "to the GEMM",
    "accumulator": "fp32 (cuBLASLt FP8 GEMM, scale_inv epilogue, bf16 output)",
    "storage": None,
    "float_ops": {
        "out_filter_dense, MLP l1/l2/l3, MHA Wqkv / out_proj, attention BMMs, tied unembed": "bf16",
        "softmax": "fp32 (flash-attn / vortex CrossAttention)",
        "featurizer + HCS depthwise FIR, HCM / HCL FFT convolutions, modal filter, "
        "step_fir / step_iir, IIR state": "fp32",
        "log_poles, residues": "fp32",
        "RMSNorm, gelu, gating, residuals, RoPE": "bf16 (fp32 internal math as in torch)",
        "KV cache, FIR states from prefill": "bf16",
    },
    "audit": {
        "allow": {
            f"dot_general {F8}": "27 Hyena `projections` (TE FP8 GEMMs)",
            "dot_general bf16xbf16->bf16": "out_filter_dense (prefill), MLP, Wqkv, QK^T / PV, "
                                           "unembed",
            "dot_general bf16xbf16->f32": "addmm-style bias Linears: out_proj, decode "
                                          "out_filter_dense",
            "convolution f32xf32->f32": "prefill featurizer (27) + HCS (9) depthwise FIRs",
        },
        "require": [f"dot_general {F8}", "convolution f32xf32->f32"],
        "params": {"float8_e4m3fn": "27 FP8 projection weights (projections/kernel)"},
    },
    "notes": (
        "Departures from the research summary: (1) the TE format is Format.HYBRID, not E4M3; "
        "HYBRID means E4M3 for forward tensors and E5M2 for gradients, so inference GEMMs are "
        "E4M3 x E4M3 (the checkpoint's fp8_max_fwd = 448 confirms). amax_history_len 16 and "
        "amax_compute_algo 'max' confirmed; margin is TE's default 0. (2) 'Default path': FP8 "
        "is on whenever transformer_engine is importable (the README's recommended full "
        "install, Hopper); the light install (no TE) silently runs bf16 projections, which "
        "is what the float port was. (3) The 27 projections are the only TE layers in vortex; "
        "the checkpoint also has savanna FP8 meta for the attention dense "
        "(blocks.<attn>.mixer.dense._extra_state) but vortex drops that key and runs "
        "out_proj as bf16 nn.Linear. (4) The 'NLL 0.9951 vs 0.9953' figure was not found in "
        "the evo2 sources (test_evo2.py expects loss 0.3477 / acc 86.35% for evo2_7b without "
        "saying which path). Graph details: vortex pads the projection input's "
        "sequence axis to a multiple of 16 when FP8 is on (decode: 1 -> 16 rows, then "
        "slices row 0); this is FP8-kernel alignment and is not modelled (coordinator "
        "decision), so the decode FP8 GEMM is [1, 4096] x [4096, 12288]. The delayed-scaling "
        "bookkeeping (amax of the input / weight, "
        "history roll, scale update after the call) is outside the GEMM path and the scale "
        "the GEMM sees is fixed per call, so it is modelled as a static per-tensor scale "
        "(FP8_STATIC, `input_scale` / `kernel_scale`; their checkpoint counterparts are "
        "the scale_inv_fwd entries in projections._extra_state). The reduced config runs in "
        "bf16 (it was fp32 for the old numerics checks), so the audit sees the deployed "
        "signatures at both sizes."
    ),
}
