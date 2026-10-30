"""AlphaFold 2 (model_1_ptm) quantization recipe (see models/quant_recipes.py for the fields)."""

F8 = "f8E4M3FNxf8E4M3FN->f32"
BF16 = "bf16xbf16->bf16"
F32 = "f32xf32->f32"
PH = "recycle_iteration"

RECIPE = {
    "recipe": "FP8 E4M3 W8A8 per-tensor static on the bf16 trunk's weight GEMMs "
              "(GPU-datacenter fallback; ModelOpt FP8 form)",
    "basis": "fallback",
    "sources": [
        "google-deepmind/alphafold c77e5d2a alphafold/model/config.py: monomer global_config has "
        "no bf16 mode (fp32); multimer CONFIG_MULTIMER global_config 'bfloat16': True, "
        "'bfloat16_output': False",
        "... alphafold/model/modules_multimer.py EmbeddingsAndEvoformer.__call__: the whole "
        "embedding + template + extra-MSA + Evoformer + single projection under "
        "utils.bfloat16_context() (fp32 params cast to bf16 on read), outputs cast back to fp32 "
        "(structure module and heads fp32); common_modules.LayerNorm computes bf16 inputs in "
        "fp32; utils.stable_softmax runs bf16 logits in fp32",
        "NVIDIA NIM for OpenFold2 docs (docs.nvidia.com/nim/bionemo/openfold2/2.5.0/"
        "support-matrix.html: precision BF16 on every listed GPU, A100..GB300; 1.1.0/"
        "performance.html: 'precision for deepspeed evoformer kernel bf16', 'precision for the "
        "rest of the model fp32') -- the most deployed accelerated AF2, not quantized",
        "Web search 2026-10-30 (AlphaFold2 / Evoformer INT8 / FP8 post-training quantization): "
        "no official or deployed quantized AF2 (ColabFold, OpenFold, BioNeMo/NIM all bf16/fp32)",
        "NVIDIA/TensorRT-Model-Optimizer FP8_DEFAULT_CFG (per-tensor FP8 weight + input "
        "quantizers on every Linear, max calibration; BMMs not quantized) -- form of the fallback",
    ],
    "weights": "fp8 e4m3, per-tensor scale (all 125 weight GEMMs of EmbeddingsAndEvoformer: "
               "common_modules.Linear sites and the get_parameter projection weights of "
               "Attention / GlobalAttention / pair-bias / OuterProductMean)",
    "activations": "fp8 e4m3, per-tensor static scale at each weight-GEMM input; bf16 "
                   "activations elsewhere in the trunk",
    "accumulator": "fp32 (x input_scale * weight_scale, cast to bf16; bias added in bf16)",
    "storage": None,
    "float_ops": {
        "attention QK^T / PV (MSA row / column, extra-MSA global, triangle start / end, "
        "template pointwise), triangle-multiplication einsum, OPM outer product and mask "
        "normalization": "bf16",
        "LayerNorm statistics, softmax (stable_softmax)": "fp32 (bf16 in / out)",
        "residual stream, biases, sigmoid gates, ReLU in the trunk": "bf16",
        "relpos one-hot, prev_pos / template distograms, template frames and torsion angles "
        "(feature construction), prev_* LayerNorms": "fp32, cast to bf16 at the trunk input",
        "structure module (IPA incl. point attention, transitions, sidechain ResNet, frames) "
        "and all heads (masked MSA, distogram, experimentally resolved, pLDDT, PAE)": "fp32",
    },
    "audit": {"allow": {f"dot_general {F8}": "trunk weight GEMMs (Linear, attention q/k/v/"
                                             "gating/output, pair-bias, OPM output)",
                        f"dot_general {BF16}": "trunk activation x activation products "
                                               "(attention BMMs, triangle einsum, OPM)",
                        f"dot_general {F32}": "structure module and heads (fp32, as the "
                                              "multimer bf16 config keeps them)"},
              "require": [f"dot_general {F8}", f"dot_general {BF16}", f"dot_general {F32}"],
              # Every trunk weight GEMM FP8; trunk BMMs bf16 = 12 per Evoformer / extra-MSA
              # block, 6 per template pair block per template, 2 pointwise; the rest fp32.
              "counts": {"reduced": {PH: {f"dot_general {F8}": 288, f"dot_general {BF16}": 74,
                                          f"dot_general {F32}": 48}},
                         "full": {PH: {f"dot_general {F8}": 2410, f"dot_general {BF16}": 674,
                                       f"dot_general {F32}": 168}}},
              "params": {"float8_e4m3fn": "trunk weight-GEMM weights"}},
    "notes": (
        "AF2 is not served quantized anywhere: DeepMind ships fp32 (monomer) / bf16-trunk "
        "(multimer) params, and NVIDIA's OpenFold2 NIM (TensorRT-BioNeMo) is BF16. So the GPU "
        "fallback applies, on the bf16 trunk the library itself defines for multimer: "
        "EmbeddingsAndEvoformer in bf16 with fp32 LayerNorm / softmax, structure module and "
        "heads fp32 (bfloat16_output=False). Departures from the research summary: (1) the "
        "pair-bias projections (feat_2d_weights) and the OuterProductMean output_w GEMM are "
        "weight GEMMs and are FP8 too (ModelOpt quantizes every Linear); (2) heads are fp32 "
        "with the structure module, outside the bf16 region. "
        "Implementation (edits in the vendored library, each marked '# quantized: FP8 W8A8'; "
        "pristine copy at git 7b50334): config.py CONFIG.global_config gains multimer's "
        "'bfloat16' / 'bfloat16_output' (False; net.patched_config sets True / False); "
        "modules.py EmbeddingsAndEvoformer.__call__ wraps the original body (now the "
        "hk.transparent _call) as modules_multimer does -- target_feat / msa_feat / seq_mask / "
        "msa_mask / extra_msa_mask cast to bf16, body under utils.bfloat16_context(), outputs "
        "back to fp32 -- and _call casts the fp32-built features (prev_pos dgram, relpos "
        "one-hot, extra-MSA feature, template torsion features, prev_* LayerNorm outputs) to "
        "bf16; common_modules.get_weight replaces hk.get_parameter + einsum for every weight "
        "GEMM (Linear.__call__, Attention / GlobalAttention query_w / key_w / value_w / "
        "gating_w / output_w, MSARowAttentionWithPairBias and TriangleAttention "
        "feat_2d_weights, OuterProductMean output_w): a weight requested in bf16 (the trunk) "
        "is stored float8_e4m3fn with '<name>_scale' and '<name>_input_scale' (fp32 scalars), "
        "any other dtype is the original fp32 path (structure module, heads, folding.py "
        "untouched). The switch is the requested weight dtype: bf16 exists only inside the "
        "trunk, so no call-site flag was needed. One input scale per weight GEMM (ModelOpt: "
        "one input quantizer per Linear), so q_data is quantized separately for query_w and "
        "gating_w. No hard-coded fp32 spot of the monomer code broke under bf16 (the template "
        "path already follows query_embedding.dtype)."),
}
