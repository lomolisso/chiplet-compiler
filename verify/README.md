# verify/

Checks that the JAX ports in `models/` describe the upstream models and compute in the
datatypes of their quantization recipe (`models/<key>/recipe.py`, collected by
`models/quant_recipes.py`). **Nothing here runs a model**: the checks read configs and
checkpoint *headers*, and lower phases abstractly (`jax.jit(...).lower` on
`ShapeDtypeStruct`s). Run from the repo root with the project venv.

| Command | What it checks |
|---|---|
| `python -m verify.faithfulness <key>` | Config fields vs upstream; every port parameter mapped onto the upstream checkpoint's tensors (shape up to axis permutation; folded / scale tensors explained); parameter count vs published; FLOPs per phase vs an analytic count from the upstream architecture. Needs a spec module `upstream/<key>.py` per model (none ship with the repo: write one from primary sources first); writes `reports/faithfulness/<key>.json`. |
| `python -m verify.dtype_audit [keys] [--cfg full] [--report]` | Every `dot_general` / `convolution` has a signature (`<lhs>x<rhs>-><accumulator>`) allowed by the recipe, per phase, with exact op counts where pinned, and the declared parameter storage dtypes exist. `--report` writes `reports/dtype_histogram_<cfg>.md`. |
| `python -m verify.hlo <key> [reduced\|full]` | Prints the GEMM / conv census of every phase. |
| `python -m pytest verify/test_quant.py` | Unit tests of `models/quant.py` on tiny arrays. |

`fetch.py` reads safetensors headers by HTTP range request and caches responses in
`verify/.cache/` (gitignored), so reruns are offline. Gated checkpoints are compared via
byte-identical ungated mirrors (named in each spec module).

## Arithmetic mix (full config, per model invocation)

Share of GEMM / conv FLOPs by operand type, measured on 2026-10-30 with
`python -m verify.dtype_audit --cfg full --report` (phase trip counts applied). `int8` = i8 x i8 -> i32; `fp8` = f8E4M3FN x f8E4M3FN -> f32;
the rest are the float carve-outs of each recipe. gemma4's int8 is GGML Q4_0 x Q8_0:
int4 weights unpacked to int8.

| Model | TFLOP | Mix |
|---|---|---|
| resnet50 | 0.008 | int8 100% |
| mobilenetv4_conv_m | 0.002 | int8 100% |
| mobilenetv4_hybrid_m | 0.003 | int8 100% |
| yolo11 | 0.087 | int8 99.5%, f16 0.5% |
| unet3d | 0.955 | int8 100% |
| bert | 0.246 | int8 100% |
| whisper | 2.978 | int8 84.9%, f16 15.2% |
| llama | 35.18 | fp8 99.6%, bf16 0.4% |
| gemma4 | 113.0 | int8 94.4%, f32 3.2%, bf16 2.3% |
| qwen3vl | 554.3 | fp8 99.0%, bf16 1.0% |
| rtdetr_v2 | 0.138 | int8 99.5%, f16 0.5% |
| sam2 | 161.6 | int8 70.0%, f16 30.0% |
| alphafold2 | 131.8 | fp8 61.1%, bf16 38.8% |
| mamba2 | 12.62 | fp8 88.7%, f32 10.9%, bf16 0.4% |
| evo2 | 120.2 | bf16 80.3%, fp8 19.6% |
| rnnt | 0.031 | int8 91.8%, f16 8.0%, f32 0.2% |
| rgat | 0.070 | fp8 100% |
| graphcast | 1152.6 | fp8 100% |
| dlrm | 0.004 | int8 100% |
| wide_deep | 1.435 | int8 100% |
| sdxl | 281.1 | fp8 93.9%, f16 3.5%, int8 2.2%, f32 0.4% |
| flux | 290.7 | fp8 95.6%, bf16 4.4% |
| pointpainting | 5.189 | int8 99.5%, f16 0.5% |
| centerpoint | 2.456 | int8 99.8%, f16 0.2% |
| ptv3 | 151.0 | int8 84.8%, f16 15.2% |
