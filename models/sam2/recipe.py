"""SAM 2.1 Hiera-L video predictor quantization recipe (see models/quant_recipes.py for the fields)."""

from models.sam2.model import CONFIGS, memory_phases

I8 = "i8xi8->i32"
CONV, DOT, ATT, PE = f"convolution {I8}", f"dot_general {I8}", "dot_general f16xf16->f32", \
    "dot_general f32xf32->f32"


def _phase_counts(cfg) -> dict:
    """Exact GEMM / conv counts per phase, from the architecture (a layer left float or
    quantized twice changes them)."""
    blocks = sum(cfg.stages)
    q_pool = min(cfg.q_pool, len(cfg.stages) - 1)       # stage starts with a dim change + q-pool
    image = {
        CONV: 1 + len(cfg.channel_list) + 2,             # patch_embed + FPN laterals + conv_s0/s1
        DOT: 4 * blocks + q_pool,                         # qkv, proj, 2 MLP per block + shortcut proj
        ATT: 2 * blocks,                                  # QK^T, AV per block
    }
    two_way = 2 * (4 + 4 + 2 + 4) + 4                     # 2 blocks x (self, t2i, MLP, i2t) + final
    heads = {
        CONV: 2,                                          # 2 upscaling ConvTranspose -> 1x1 convs
        DOT: two_way + 4 * 3 + 3 + 3 + 3,                 # + 4 hypernetwork, IoU, obj-score, obj_ptr MLPs
        ATT: 2 * (3 * 2) + 2 + 1,                         # 3 attentions / block, final; hyper_in @ up
        PE: 2,                                            # random-Fourier PE: dense grid, points
    }
    mem_enc = {CONV: 4 + 1 + 1 + cfg.fuser_layers + 1,   # 4 strided + 1x1 downsampler, pix_feat_proj,
               DOT: 2 * cfg.fuser_layers}                 # dwconvs, out_proj; pwconv1/2 per CXBlock
    mem_att = {DOT: 1 + cfg.mem_attn_layers * (4 + 4 + 2),   # obj_ptr_tpos_proj; self, cross, FFN
               ATT: 4 * cfg.mem_attn_layers}
    out = {"image_encoder": image, "sam_heads_cond": heads, "sam_heads": heads,
           "memory_encoder_cond": mem_enc, "memory_encoder": mem_enc}
    out.update({name: mem_att for name, *_ in memory_phases(cfg)})
    return out


_FULL = _phase_counts(CONFIGS["full"])
_WHY = {CONV: "convs (ConvTranspose as 1x1 conv + depth-to-space)", DOT: "Linear layers",
        ATT: "attention QK^T / AV; hyper_in @ upscaled", PE: "random-Fourier PE coords @ gaussian"}

RECIPE = {
    "recipe": "INT8 W8A8 platform fallback (edge): int8 convs and Linears in every component; "
              "attention core, softmax, norms fp16; memory bank stored fp16",
    "basis": "fallback",
    "sources": [
        "facebookresearch/sam2 README: reduced precision only via torch.autocast(bfloat16); "
        "SAM2VideoPredictor stores maskmem_features in bfloat16; no quantized checkpoint",
        "quic/ai-hub-models qai_hub_models/models/sam2/manifest.yaml: checkpoint sam2.1_hiera_t "
        "(Tiny only), supported_precisions [float, w8a8]; SAM2 exports only the encoder and the "
        "decoder components (no memory attention / memory encoder)",
        "Torch-TensorRT SAM2 tutorial (Hiera-L): fp16 with use_fp32_acc, no INT8",
    ],
    "weights": "int8 per-output-channel symmetric on every conv (patch embed, FPN laterals, "
               "conv_s0 / s1, mask-decoder ConvTranspose2d k2 s2 (as a 1x1 conv to 4 Cout channels + "
               "depth-to-space, per-channel over the 4 Cout rows), mask downsampler, pix_feat_proj, CXBlock "
               "depthwise 7x7, out_proj) and every Linear (Hiera qkv / proj / MLP / shortcut proj, "
               "two-way transformer, hypernetwork / IoU / object-score / obj_ptr MLPs, memory "
               "attention q/k/v/out + FFN, obj_ptr_tpos_proj, CXBlock pwconv)",
    "activations": "int8 per-tensor static symmetric (calibrated); fp32 normalized frame "
                   "quantized by the patch-embed conv; a tensor read by several INT8 layers (Hiera "
                   "norm1 -> qkv + shortcut proj, shared q/k/v inputs) is quantized once",
    "accumulator": "int32 (fp32 epilogue: scale, bias; written fp16)",
    "storage": "memory bank: maskmem_features, maskmem_pos_enc and object pointers stored fp16 "
               "(the predictor's bf16 storage on an fp16 edge target)",
    "float_ops": {"attention QK^T and AV (Hiera, two-way transformer, memory attention)":
                  "fp16 operands, fp32 accumulation",
                  "softmax": "fp32",
                  "LayerNorm / LayerNorm2d (fp32 statistics, fp16 out), GELU, ReLU, residual adds": "fp16",
                  "RoPE, sine / random-Fourier position encodings (incl. the 2 x 128 coords @ "
                  "gaussian GEMM), pos-embed bicubic": "fp32",
                  "mask product hyper_in @ upscaled embedding": "fp16 operands, fp32 accumulation",
                  "nearest FPN upsample, q max-pool, window partition, depth-to-space": "data movement (fp16 / int8)",
                  "mask logits, bilinear mask upsampling, mask selection, IoU / object-score heads' "
                  "outputs and sigmoids": "fp32",
                  "tensors between phases (vision features / pos, memory bank, obj_ptr)": "fp16"},
    "audit": {"allow": dict(_WHY),
              "require": [CONV, DOT, ATT, PE],
              # per phase: exactly the signatures its layers use, all required
              "phases": {ph: {"allow": {sig: _WHY[sig] for sig in c}, "require": list(c)}
                         for ph, c in _FULL.items()},
              "counts": {name: _phase_counts(cfg) for name, cfg in CONFIGS.items()},
              "params": {"int8": "conv / Linear weights"}},
    "notes": "No official or production INT8 SAM 2.1 Hiera-L: Meta ships fp32 weights with bf16 "
             "autocast; Qualcomm AI Hub's w8a8 release is the Tiny model, encoder + decoder only "
             "(range scheme, per-channel / per-tensor and symmetric / asymmetric undocumented); "
             "NVIDIA's documented path is fp16. So the edge INT8 fallback applies, as in the "
             "research summary. Note: in the steady state the memory attention's attention cores "
             "(cross: 4096 queries x 28,736 keys, 482 GFLOP; self: 69 GFLOP) are ~91% of its "
             "605.6 GFLOP per frame and stay fp16 by this rule (memory attention is ~24% of the "
             "video's GEMM work); a Qualcomm-style full w8a8 would also quantize the attention "
             "matmuls. "
             "The random-Fourier PE matmul is positional-encoding math (fp32). The memory-bank "
             "storage dtype is a phase input / output dtype change in model.py (bf16 -> fp16); "
             "the image-encoder outputs (vision features, positional encodings, s0 / s1) are "
             "fp16 phase tensors too. The two mask-decoder ConvTranspose2d (k = stride = 2) are "
             "rewritten as 1x1 convs to 4 Cout channels + depth-to-space (NVIDIA's deconv "
             "rewrite, the same math), which removes the 4x zero-input over-count: sam_heads "
             "3.64 instead of 6.06 GFLOP.",
}
