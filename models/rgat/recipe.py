"""R-GAT quantization recipe (see models/quant_recipes.py for the fields)."""

F8 = "f8E4M3FNxf8E4M3FN->f32"
NV = "mlcommons/inference_results_v5.0 closed/NVIDIA/code/rgat/pytorch/"

RECIPE = {
    "recipe": "MLPerf NVIDIA FP8 E4M3 node-feature storage + FP8 E4M3 W8A8 GAT / classifier "
              "linears (GPU-datacenter fallback), fp16 elsewhere",
    "basis": "fallback",
    "sources": [
        NV + "config.py:46 (fp8_embedding=True, gatconv_backend='cugraph')",
        NV + "preprocess_data.py:531-580 fp8_conversion (every node table cast with one "
        "TE FP8Helper; config['fp8'] = {scale, format}), :635-636 (--fp8_format default "
        "e4m3, --scale default 1.0); preprocess_data.sh:33-38 passes neither, so E4M3, scale 1.0",
        NV + "common/helper.py:47-73 FP8Helper (TE FP8TensorMeta scale / scale_inv, "
        "cast_from_fp8 -> float16)",
        NV + "dataset.py:222-235, 343-358 (WholeGraph gather of the uint8-stored FP8 rows, "
        "then fp8_to_fp16 per node type)",
        NV + "rgat.py:143 model.half(); model/dgl_model.py cuGraphGATConvCustom (shared "
        "lin on src and dst, pylibcugraphops mha_gat_n2n fused attention)",
        "mlcommons/inference_results_v5.0 closed/NVIDIA/configs/rgat/__init__.py:27 "
        "precision = 'fp16'",
        "same code in inference_results_v5.1 and v6.0 closed/NVIDIA/code/rgat (v6.0 differs "
        "from v5.1 only in a preprocess_data.sh path; rgat.py:159 model.half())",
        "comparison: inference_results_v5.0 closed/Intel/code/rgat/pytorch-cpu run_offline.sh "
        "(--use-bf16 --use-qint8-gemm, QINT8_BLOCK_SIZE=128), backend.py:62-69 "
        "(remap_and_quantize_qint8 on every GATConv fc.weight), scripts/download_dataset.sh:59 "
        "(int8 features, block 128), measurements.json weight_data_types INT8",
    ],
    "weights": "fp8 E4M3, per-tensor (every live GATConv fc and the classifier Linear)",
    "activations": "fp8 E4M3, per-tensor static; the layer-0 fc consume the stored FP8 "
                   "features directly with their storage scale (1.0)",
    "accumulator": "fp32 (epilogue: input_scale * kernel_scale, + classifier bias)",
    "storage": "node features fp8 E4M3, one global scale 1.0 for all six node types "
               "(gathered host-side / WholeGraph, so FP8 rows are the phase input)",
    "float_ops": {
        "attention logits (feat * attn_l / attn_r).sum, u_add_v, leaky_relu(0.2)": "fp32 (fp16 in)",
        "edge softmax (segment max / exp / segment sum / div)": "fp32",
        "u_mul_e + segment-sum aggregation": "fp32 accumulation, fp16 out",
        "GATConv bias, HeteroGraphConv relation sum, leaky_relu(0.01)": "fp16",
        "FP8 GEMM epilogues (scales, classifier bias)": "fp32, output fp16",
        "classifier logits": "fp16",
    },
    "audit": {"allow": {f"dot_general {F8}": "52 GATConv fc GEMMs (fc on src and dst of the "
                                                "26 live relations x layers) + classifier"},
              "require": [f"dot_general {F8}"],
              "phases": {"forward": {"allow": {f"dot_general {F8}": "every GEMM of the phase"},
                                     "require": [f"dot_general {F8}"]}},
              # 2 fc GEMMs per live GATConv (26 live) + the classifier, no other GEMM
              "counts": {"reduced": {"forward": {f"dot_general {F8}": 53}},
                         "full": {"forward": {f"dot_general {F8}": 53}}},
              "params": {"float8_e4m3fn": "fc / classifier weights (FP8 E4M3 per-tensor)",
                         "float16": "attn_l / attn_r / GATConv bias (model.half())"}},
    "notes": (
        "NVIDIA's deployed R-GAT (v5.0-v6.0, the only GPU submitter code) computes every GEMM "
        "and the fused cugraph-ops GAT attention in fp16; its only low-precision element is FP8 "
        "E4M3 node-feature storage with a single fixed scale 1.0 (a plain cast, not a "
        "calibrated amax scale; not per node type), upcast to fp16 right after the gather. "
        "No FP8 GEMMs exist in the code, so the GEMMs take the GPU-datacenter fallback (FP8 "
        "W8A8 per-tensor static); the classifier Linear(512, 2983) is included as a linear. "
        "Basis: storage = MLPerf NVIDIA (deployed), GEMMs = GPU-datacenter fallback (not deployed "
        "anywhere in FP8). Departures from the research summary: (1) the feature 'per-tensor scale' is one "
        "global constant 1.0, not calibrated; (2) non-quantized compute is fp16 (NVIDIA "
        "model.half(), config precision fp16), not the generic bf16 of FP8 recipes; "
        "(3) softmax / aggregation accumulate in fp32 (fp16 I/O) -- cugraph-ops internals are "
        "assumed, the kernel is not in the submission. Intel (CPU) differs: bf16 compute, "
        "int8 fc weights, int8 features in blocks of 128 with per-block scales."),
}
