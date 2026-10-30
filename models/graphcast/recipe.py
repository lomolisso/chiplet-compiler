"""GraphCast_operational quantization recipe (see models/quant_recipes.py for the fields)."""

F8 = "f8E4M3FNxf8E4M3FN->f32"

RECIPE = {
    "recipe": "FP8 E4M3 W8A8 per-tensor static on every MLP linear, bf16 activations "
              "(GPU-datacenter fallback; ModelOpt FP8 form)",
    "basis": "fallback",
    "sources": [
        "google-deepmind/graphcast 08cf7362 graphcast_demo.ipynb construct_wrapped_graphcast: "
        "Bfloat16Cast(GraphCast) inside InputsAndResiduals -- the released inference setup is "
        "bf16, no quantized checkpoint or path exists",
        "... graphcast/casting.py Bfloat16Cast / bfloat16_variable_view: inputs, targets "
        "template and forcings cast to bf16 after normalization; params requested as bf16 are "
        "created fp32 and cast to bf16 on read; predictions cast back to the targets dtype (fp32)",
        "... graphcast/graphcast.py GraphCast.__init__: grid2mesh_gnn f32_aggregation=True, "
        "mesh_gnn / mesh2grid_gnn f32_aggregation=False; deep_typed_graph_net.py "
        "_networks_builder: hk.nets.MLP (1 hidden layer, swish) + hk.LayerNorm per update",
        "NVIDIA/TensorRT-Model-Optimizer FP8_DEFAULT_CFG (per-tensor FP8 weight + input "
        "quantizers on every Linear, max calibration) -- form of the fallback",
    ],
    "weights": "fp8 e4m3, per-tensor scale (all 86 MLP linears: node / edge embedders, 18 edge "
               "and 18 node updates, decoder output MLP)",
    "activations": "fp8 e4m3, per-tensor static scale at each linear input (the concatenated "
                   "[edge, sender, receiver] / [node, aggregated] features)",
    "accumulator": "fp32 (scale + bias epilogue, bf16 output)",
    "storage": None,
    "float_ops": {
        "swish, LayerNorm (statistics accumulated in fp32 by jnp.mean / jnp.var), residual "
        "adds, gathers, mesh / mesh2grid segment_sum": "bf16",
        "grid2mesh segment_sum (f32_aggregation=True)": "fp32",
        "input / forcing normalization, residual add + unnormalization of the predictions "
        "(normalization.InputsAndResiduals, outside Bfloat16Cast)": "fp32",
        "structural node / edge features (host numpy)": "fp32 -> bf16",
    },
    "audit": {"allow": {f"dot_general {F8}": "the MLP linears (no other GEMMs in the step)"},
              "require": [f"dot_general {F8}"],
              "phases": {"forecast_step": {"allow": {f"dot_general {F8}": "MLP linears"},
                                           "require": [f"dot_general {F8}"]}},
              # 2 linears per MLP: 5 embedders + grid2mesh (3) + processor (2 per step)
              # + mesh2grid (2) + output MLP
              "counts": {"full": {"forecast_step": {f"dot_general {F8}": 86}},
                         "reduced": {"forecast_step": {f"dot_general {F8}": 34}}},
              "params": {"float8_e4m3fn": "MLP weights `w`"}},
    "notes": (
        "GraphCast is only released and run in bf16 (Bfloat16Cast); no quantized deployment "
        "exists, so the GPU fallback applies: FP8 on every MLP linear (there are no other GEMMs "
        "in the step: aggregation is segment_sum, message gathers are indexing). Matches the "
        "research summary. Small-K edge embedders (K=4) and the 512->83 output layer are "
        "included (ModelOpt quantizes every Linear). "
        "IMPLEMENTATION (vendored-file edits only, each marked '# quantized: FP8 W8A8'; no "
        "interceptors / monkeypatches / custom getters): the linears are haiku's own hk.Linear "
        "inside hk.nets.MLP (site-packages, not editable), built in one place, "
        "deep_typed_graph_net.py DeepTypedGraphNet._networks_builder.build_mlp. Edit: added to "
        "graphcast/mlp.py an FP8 copy of haiku's MLP + Linear (haiku is Apache-2.0 like "
        "graphcast; keep its header note) that preserves the module / param names "
        "'<name>_mlp/~/linear_<i>/{w,b}' (Fp8Linear / Fp8MLP); build_mlp uses it. Per linear: 'w' requested "
        "as float8_e4m3fn (casting._bfloat16_creator only rewrites bf16 requests, so it stays "
        "fp8), new fp32 scalar params 'w_scale' and 'input_scale'; x quantized with "
        "input_scale, dot_general(..., preferred_element_type=f32), * (input_scale * w_scale) "
        "+ b (b requested and added in fp32), cast to the input dtype (bf16). jraph.concatenated_args "
        "stays, so the concat is the quantized input. LinearNormConditioning (GenCast only) "
        "untouched. verify/upstream/graphcast.py marks r'(w_scale|input_scale)$' PORT_ONLY "
        "and reads the upstream manifest from graphcast_manifest.json, snapshotted from the "
        "pristine library (7b50334:models/graphcast/upstream)."),
}
