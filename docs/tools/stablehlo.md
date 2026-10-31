# StableHLO semantics that matter for kernel analysis

Our passes read the StableHLO that `jax.export` produces (`jax_export.md`) and derive kernels from op semantics
(iteration spaces, reductions, data movement). This file lists the facts from the spec that the analysis must
respect. Spec: https://openxla.org/stablehlo/spec (section anchors `#<op_name>`); passes:
`stablehlo/transforms/Passes.td` in https://github.com/openxla/stablehlo.

## 1. General rules

- Values are tensors with static or dynamic (`?`) dimension sizes; **unranked** tensors are not allowed. Shapes
  must agree at run time, else behaviour is undefined (spec "Dynamism", "Shape mismatches").
- Ops are SSA, in `func.func` bodies; regions (`reduce` bodies, `while`, `case`, `composite` decompositions via
  symbol) hold sub-computations. JAX emits `func.call` to private functions (e.g. composite decompositions,
  nested named computations); the call graph must be traversed or inlined (`func.call` has no special semantics).
- Element types: `i1…i64`, `ui*`, `f8E4M3FN`, `f8E5M2`, … `bf16`, `f16`, `f32`, `f64`, complex, and quantized
  `!quant.uniform<…>` (§8).

## 2. Elementwise ops (`add`, `multiply`, `maximum`, `convert`, `select`, `clamp`, …)

- For non-quantized tensors `add` requires `type(lhs) = type(rhs) = type(result)` (spec `#add`, C1): identical
  shapes, **no implicit broadcasting**. Broadcasts are explicit `broadcast_in_dim` ops upstream. (`chlo.broadcast_*`
  ops do broadcast, but JAX's StableHLO export does not use CHLO for these; they appear only after passes such as
  quant-to-math, §8.)
- Iteration space of an elementwise op = result shape; all operands indexed identically.
- `convert` changes element type only (int↔float rounding follows the spec's `convert` rules); treat it as a
  datapath op (it becomes hardware for FP8/int casts).
- `clamp` operands `min`/`max` may be rank-0 (scalar), the one built-in scalar broadcast (spec `#clamp`).

## 3. `broadcast_in_dim` (spec `#broadcast_in_dim`)

- `broadcast_dimensions` maps operand dim `d` to result dim `broadcast_dimensions[d]`;
  `size(broadcast_dimensions) = rank(operand)` (C2), values in `[0, rank(result))` (C3); operand dim of size 1 is
  read at index 0 (`operand_index[d] = 0 if dim(operand, d) = 1`).
- It is pure data movement (affine index map with zero strides); fold into consumers' access functions rather than
  treating as a kernel.

## 4. `dot_general` (spec `#dot_general`)

- Attributes: `lhs_batching_dimensions`, `rhs_batching_dimensions`, `lhs_contracting_dimensions`,
  `rhs_contracting_dimensions`, `precision_config`, optional `algorithm` (DotAlgorithm; if set, `precision_config`
  must be `DEFAULT`).
- **Result layout**: `shape(result) = dim(lhs, lhs_batching_dimensions) + dim(lhs, lhs_result_dimensions) +
  dim(rhs, rhs_result_dimensions)` (C12): batch dims first, then lhs free dims (in order), then rhs free dims.
- Element types: non-quantized requires `element_type(lhs) = element_type(rhs)` (C13); the **result element type
  is not constrained**: this is how JAX's `preferred_element_type` appears: e.g. `i8 × i8 → i32`,
  `f8E4M3FN × f8E4M3FN → f32` are encoded purely in the result type. Mixed operand dtypes (int4 weight × bf16
  activation) therefore show up as a `convert` before the dot (QDQ form in `models/quant.py`).
- `precision_config` "controls the tradeoff between speed and accuracy for computations on accelerator
  backends"; record it but do not derive hardware from it.
- Kernel analysis: iteration space = batch × lhs-free × rhs-free × contracting; contracting dims are the
  reduction dims (accumulator width = result element type).

## 5. `convolution` (spec `#convolution`)

- `dimension_numbers`: `input_batch_dimension`, `input_feature_dimension`, `input_spatial_dimensions`,
  `kernel_input_feature_dimension`, `kernel_output_feature_dimension`, `kernel_spatial_dimensions`,
  `output_batch_dimension`, `output_feature_dimension`, `output_spatial_dimensions` (printed as e.g.
  `[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]`).
- Window attrs: `window_strides`, `padding` (lo/hi per spatial dim), `lhs_dilation` (input/base dilation,
  transposed conv), `rhs_dilation` (kernel/window dilation), `window_reversal` ("appears to be unused … planning to
  remove it").
- Grouping: `feature_group_count` splits lhs along `input_feature_dimension`
  (`dim(lhs, input_feature_dimension) % feature_group_count = 0`, C11); depthwise conv has
  `feature_group_count = C_in`; `batch_group_count` splits along batch (C10) and appears in weight-gradient convs,
  should not appear in inference.
- Result element type is again free (int8 conv → i32 accumulators).

## 6. `reduce`, `reduce_window` (spec `#reduce`, `#reduce_window`)

- `reduce`: "The order of reductions is implementation-defined, which means that `body` and `init_values` must
  form a monoid"; `schedule` is "an implementation-defined full binary tree". So reassociation (tree adders,
  partial sums) is semantics-preserving **by definition**, but for floating-point bodies results may differ bit
  for bit across orders; our FP verification must compare against a fixed order or use tolerance, and say which.
- `reduce_window` applies `body` to each window (attrs `window_dimensions`, `window_strides`, `base_dilations`,
  `window_dilations`, `padding`); each window is reduced like `reduce`, so the same order freedom applies
  (inference from the spec's definition in terms of `reduce`, (unverified) wording). Max-pool = `reduce_window`
  with `maximum` body; avg-pool = `reduce_window` add + divide.
- Multi-operand reduces (variadic, e.g. argmax as (value, index) pairs) exist; the body is not a single binary op.

## 7. `gather` / `scatter` (spec `#gather`, `#scatter`)

- `gather` dimension numbers: `offset_dims`, `collapsed_slice_dims`, `operand_batching_dims`,
  `start_indices_batching_dims`, `start_index_map`, `index_vector_dim`, plus `slice_sizes`, `indices_are_sorted`.
  `slice_sizes[collapsed_slice_dims…] <= 1` (C9); zeros are inserted at collapsed and operand-batching dims to form
  `full_offset_index`. Embedding lookups, `take_along_axis`, and some reshapes/transposes lower to gather.
- `scatter`: `update_computation` combines; "`schedule` is an implementation-defined permutation of
  `index_space(updates[0])`"; with duplicate indices and a non-commutative/non-associative combiner the result
  is order dependent. `unique_indices = true` with non-unique indices → undefined behaviour;
  `indices_are_sorted = true` lets the implementation assume sorted indices.
- Treat gather/scatter as data-movement kernels with data-dependent addresses; they are not affine.

## 8. Quantized types and `stablehlo-legalize-quant-to-math`

- Type: `tensor<2x!quant.uniform<i8:f32:0, {0.1:-30,0.5:-20}>>` = storage `i8`, expressed `f32`, per-axis on dim 0
  with (scale:zero_point) per channel; per-tensor form `!quant.uniform<i8:f32, 1.0:0>`. Ops `uniform_quantize`,
  `uniform_dequantize`; quantized `dot_general` is defined as `dequantize_op_quantize(...)` (hybrid:
  `hybrid_dequantize_then_op`), with `expressed_type(lhs) = expressed_type(rhs) = expressed_type(result)` (C18).
- **Our models do not use these types**: `models/quant.py` writes explicit QDQ arithmetic with plain `i8/i4/f8`
  tensors, so integer GEMMs appear as `dot_general` on integer operands with i32 results plus f32 epilogues.
- `stablehlo-legalize-quant-to-math` (func pass) converts UniformQuantized programs to integer/float math; its
  documented output rescales through **f32**: `convert` i8→f32, `chlo.broadcast_multiply` by the scale ratio,
  `chlo.broadcast_add` zero-point term, `round_nearest_even`, `convert` →i32, add, `clamp` to [-128,127], `convert`
  →i8. So quant-to-math does not yield integer-only requantization (fixed-point multiplier + shift); if we ever
  ingest quantized-type graphs, requant hardware must be derived from the f32 math explicitly.
- Related passes: `stablehlo-legalize-qdq-to-quantized-op` (fuse dequantize→float op→quantize into a quantized
  op; "does not delete any preexisting op"), `stablehlo-legalize-quantized-op-to-qdq` (inverse).

## 9. Dynamic shapes (from symbolic export)

Spec definitions: `dynamic_broadcast_in_dim` (result shape from `output_dimensions`), `dynamic_reshape`,
`dynamic_iota` (`output_shape`), `dynamic_pad` (padding as values), `dynamic_gather` (`slice_sizes` as a value),
`dynamic_conv` (padding as a value), `get_dimension_size` ("size of the given dimension"), each "functionally
identical" to its static counterpart. JAX additionally emits `real_dynamic_slice`, `dynamic_reduce_window`,
`dynamic_top_k`, `dynamic_rng_bit_generator` (`jax_export.md` §3).
- Analysis approach: treat dim variables symbolically (they come from `get_dimension_size` of `main` args), and
  map each dynamic op to its static twin with symbolic sizes.
- Useful passes: `stablehlo-refine-shapes` (specialise to static shapes after setting argument types; removes
  `custom_call @shape_refinement_operand_wrapper`), `stablehlo-canonicalize-dynamism` (dynamic → static op when
  the dynamic operands are constants), `stablehlo-check-shape-assertions` (evaluates `@shape_assertion`).
  Running them needs a `stablehlo-opt` build (Docker); availability of prebuilt binaries (unverified).

## 10. `composite` (spec `#composite`)

- "The `composite` op can be replaced with its decomposition without changing program semantics"; attributes
  `name`, `composite_attributes`, `decomposition` (symbol), `version`.
- Our kernel identification uses composites as *hints* (names/attrs from Flax modules), but must verify the
  decomposition, never trust the name (design §1, ground rules). `stablehlo-legalize-composite-to-call` replaces composites
  by `func.call` to the decomposition; `stablehlo-wrap-in-composite` does the reverse for chosen ops.

## 11. Other ops to expect

`transpose`, `reshape`, `slice`, `concatenate`, `pad`, `iota`, `reverse` (data movement); `while`, `case`, `if`
(control; scan-based models such as Mamba2/RNN-T use `while`); `custom_call` (opaque; only `shape_assertion` is
expected; anything else must be reported); `rng_bit_generator` (should not appear in inference); `sort`, `top_k`
lowering via `chlo`/custom_call (unverified per model).

## Sources

- StableHLO spec: https://openxla.org/stablehlo/spec, sections `add`, `broadcast_in_dim`, `clamp`,
  `composite`, `convolution`, `dot_general`, `dynamic_broadcast_in_dim`, `dynamic_conv`, `dynamic_gather`,
  `dynamic_iota`, `dynamic_pad`, `dynamic_reshape`, `gather`, `get_dimension_size`, `reduce`, `reduce_window`,
  `scatter`, `uniform_quantize`, `uniform_dequantize`, "Dynamism", quantization types.
- `stablehlo/transforms/Passes.td` (main, 2026-10-30): `StablehloLegalizeQuantToMathPass`,
  `StablehloLegalizeQDQToQuantizedOpPass`, `StablehloRefineShapesPass`, `StablehloCanonicalizeDynamismPass`,
  `StablehloCheckShapeAssertionsPass`, `StablehloLegalizeCompositeToCallPass`, `StablehloWrapInCompositePass`.
- Repo: `models/quant.py` (QDQ form).
