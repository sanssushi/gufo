# GSQ-RCO support in gufo — state of the solution

Branch: `feat/gsq-rco` (based on `main`).

This reference documents the fork's GSQ-RCO support: the quantization types,
the loader rules, the GPU paths, the exactness contracts those paths
preserve, how the result is verified, its measured performance, and the
files involved.

The support is fork-only. `feat/gsq-rco` is a long-lived series of patches
rebased onto upstream `main`.

## Summary

gufo runs the ISTA-DASLab GSQ-RCO GGUFs of Qwen3.8-Flash-Next: IQ3_XXS
("xxs", 3.00 bits per weight) and IQ3_S ("s", 3.50 bits per weight). The
two variants, called mixes below, support the same features as the unsloth
UD-Q4_K_XL target: AR and MTP generation (with the shared Q8_0 MTP sidecar),
image input (BF16 mmproj sidecar), the chat template, graph capture, batched
sessions and serving modes. All in-repo quality checks pass on both mixes,
and the UD-Q4_K_XL control shows no regression relative to `main`.

Speed (details in "Performance state"):

- AR token generation is within about 7% of the control on both mixes.
- MTP token generation at a 32768-token context measures the multi-row
  verification forward pass. For greedy runs, upstream's prompt-lookup
  draft copies bypass the neural draft head (the two mixes and the control
  all take the same 17 cycles and accept 111 draft tokens), so this
  measurement is dominated by verifying eight rows at a time. The remaining
  gap (about -2 to -3% for xxs and about -6% for s) is the per-element
  decode cost of the low-bit weight formats at those eight verification
  rows; the shared expert's decoded weight fragments are reused across the
  tokens in a group. It is not a draft, attention or dispatch problem.
- MTP token generation with no preceding context follows the draft
  acceptance rate rather than kernel speed, so the result varies with the
  draft trajectory.
- Prefill is about 15-21% slower and is compute-bound. The gap is the
  arithmetic cost of the F16 WMMA path used for these formats, not a byte,
  launch or dispatch problem.

On the implementation side, the IQ2/IQ3 prefill decoder builds each group of
signed bytes directly from the quantization grid words, using packed sign
masks. Mixed-type SSM layers fuse the four-tap convolution into their
standalone QKV projection; the gate keeps its own projection type.

## Targets and artifacts

| Item | Description |
| --- | --- |
| GSQ-RCO GGUFs | [ISTA-DASLab GSQ-RCO releases](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) of Qwen3.8-Flash-Next: IQ3_XXS ("xxs", 3.00 bits per weight) and IQ3_S ("s", 3.50 bits per weight), each with the shared MTP Q8_0 sidecar and the BF16 mmproj sidecar |
| Control | unsloth UD-Q4_K_XL (4 shards): the speed and quality baseline for this feature |
| Reference implementation | GSQHalo.cpp (a llama.cpp fork): read-only reference for the formats and kernel ideas |

Facts that shape the implementation:

- Shard 2 of a GSQ mix contains a single tensor, the per-layer token-embedding
  (PLE) table (`per_layer_token_embd.weight`, IQ4_NL); shard discovery
  handles this unbalanced layout.
- Images and MTP use the shared Q8_0 and BF16 sidecars. These attach by
  identity keys, not by weight type, so they work with GSQ targets.
- Each mix stores these types per role:

| Role | Types |
| --- | --- |
| dense attn, SSM and shared-expert projections | IQ4_XS, IQ3_S, Q6_K, Q5_K, Q4_K, IQ3_XXS, IQ2_S |
| routed experts | Q2_0, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, Q4_K, Q5_K, Q6_K |
| hyper-connection mixers (`hc_*`) | BF16 |
| `token_embd` | IQ3_S or IQ4_XS |
| `output` | Q5_K (xxs) or Q6_K (s) |
| `ple_conv1d` | F16 |
| router and GDN alpha/beta | BF16 |

- Only 4 of 12 attention layers in xxs and 5 of 12 in s, plus 9 of 36 SSM
  layers in xxs and 12 of 36 in s, have all their projections in one
  quantization type. Attention and SSM stacking therefore merges only those
  groups (see "Stacked attention and SSM projections").

## Type layer (`src/core`)

- `gguf_reader.hpp` `GgmlType` includes `kIQ2_XS = 17`, `kIQ3_XXS = 18`,
  `kIQ2_S = 22` and `kQ2_0 = 42`. The numbers are GGUF type ids; all other
  ids keep their values (never renumber). Q2_0 has no upstream id and follows
  GSQHalo's `block_q2_0` layout: an fp16 scale, 16 code bytes and 64 weights
  per block. Keep it isolated in the quant layer.
- `quant/ggml_dequant.{hpp,cpp}` defines the block structs, row sizes, CPU
  dequantization and f32 dot products for IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
  IQ3_S, IQ4_NL, IQ4_XS and Q2_0. Its grid and sign tables match GSQHalo's
  `ggml-common.h`. The CPU entry points (`quant::Dequantize`/`quant::Dot`,
  `cpu::DequantizeRow`) handle all of these types; unknown types abort
  instead of silently filling with zeroes.
- `DescribeQwenGemmFormat` lists these types and sets `block_elements`
  for the quantized-only group, so enabling a format directly cannot hit
  `kMisalignedQuantK` through a missing block size.
- `tests/core/ggml_dequant_test.cpp` covers hand-built blocks, row-size
  rules and test vectors that distinguish correct decoding from plausible
  mistakes (IQ2_S high grid bits in `qh`, the two IQ3_XXS grid halves, sign
  indices beyond 0 and 1). With `GUFO_GSQ_MODEL` set, it also checks rows
  from the real shards against frozen per-type samples; the results are
  bit-exact against an independent port of GSQHalo's `ggml-quants.c`.

## Loader and upload checks

`src/models/qwen38_flash_next/weights.cpp`:

- `FormatOf` maps each type to its block geometry (elements and bytes):
  Q2_0 {64,18}, IQ2_XXS {256,66}, IQ2_XS {256,74}, IQ2_S {256,82},
  IQ3_XXS {256,98}, IQ3_S {256,110}, IQ4_XS {256,136}. The values duplicate
  `quant::QuantizedBlockElements`/`QuantizedRowBytes`; the two definitions
  agree.
- Per-role allowed-type lists. Each list is exactly the union of the types
  used in the two mixes' per-tensor allocation maps plus the types accepted
  for UD-Q4_K_XL, and no wider. The GSQ types by role are:

| Role | Types |
| --- | --- |
| dense (attn_*, ssm_qkv/gate/out, indexer, PLE key/value, nextn.eh_proj) | Q4_K, Q5_K, Q6_K, IQ4_XS, IQ3_S |
| shared expert (ffn_*_shexp) | Q4_K, Q5_K, Q6_K, IQ4_XS, IQ3_S, IQ4_NL, Q2_0 |
| experts (ffn_*_exps) | IQ4_NL, Q2_0, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS |
| embd (token_embd) | IQ3_S, IQ4_XS |
| output | Q5_K |
| router, ffn_gate_inp_shexp, GDN alpha/beta | BF16 |
| ple_conv1d | F16 |
| hc_*, ssm_alpha/beta | BF16 |

`device_model.cpp`:

- `Stack` accepts BF16 stacked weights; the router and GDN alpha/beta
  weights run through the generic BF16 BLAS path in their stored layout.
- The Q6_K type check allows the LM head (Q6_K head kernels exist) and
  deliberately rejects Q6_K `token_embd` and routed experts: no supported
  mix stores Q6_K there, and the expert paths are untested. Keep this check
  strict.
- `ple_conv1d` is F16 in GSQ. Both consumers read it according to its
  encoding; reading it as `float*` would use half the bytes, read out of
  bounds and produce NaN logits.
- Weight uploads include a zeroed `kTailMargin` tail. The wide MMQ path
  reads whole 256-element k-iterations past the end of short rows; an
  uninitialized tail can poison the accumulator (0 × inf) and make
  `qfn_mmq_sanitize_f32` zero whole rows. Test uploads must include the
  zeroed tail too.
- `qwen38_flash_next.weights` checks each role's accepted types and that
  disallowed types are rejected.

## GPU execution paths

Kernel sources are under `src/models/qwen38_flash_next/kernels/rocm/`:
`mmq/` holds the vendored llama.cpp MMQ code and the gufo-owned adapter;
`kernels.hip.cpp` holds shared GEMV/GEMM kernels. Vendored policy:
`mmq.hpp`, `ggml-common.h` and the other vendored files stay at or near
zero diff; gufo owns `qfn_mmq.h`/`qfn_mmq.hip.cpp`, `executor.cpp`,
`weights.cpp` and `core/quant`. Every local addition to a vendored file is
recorded in `kernels/rocm/mmq/VENDOR.md` as something to re-apply when the
vendored code is re-imported (see "Vendored-code obligations").

### Routed experts (MoE): the main compute path

- Gate/up projections use IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS
  or Q2_0 (plus Q4_K, Q5_K, Q5_1 and Q8_0). Down projections
  use IQ4_NL or Q2_0 (plus Q5_K, Q5_1 and Q8_0).
- Small batches (up to 8 rows) use MMVQ. `moe_vector_projection` covers every
  routed format; its supported set is the set defined by
  `qfn_mmq_moe_gated_type`, plus Q5_1. The fused SwiGLU gated vector path in
  `mmvq.hip.cpp` supports every routed format except Q5_1, and the vector
  paths handle up to 8 rows except Q4_K. For a group's shared expert, the
  grouped gated kernel decodes a weight fragment once per row and K block,
  then reuses it for every token in the group. `Q4MoeFragment` handles Q4_K;
  `IqMoeFragment` handles IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S
  (`vecdotq.hpp`). `qfn_mmq_moe_gated_type`
  is the single definition of the gated-format set (in the adapter, exported
  through `qfn_mmq.h`); the executor's `GatedMoeType` delegates to it, so the
  two sets cannot differ.
- Wide batches also use MMQ: `mul_mat_q_case` instantiations plus the
  type-generic `qfn_mmq_moe_raw` and `qfn_mmq_moe_pair_unique` in
  `qfn_mmq.hip.cpp`. The routed tile width comes from
  `qfn_mmq_routed_tile_cols_for_counts`; `kPanel 32` is the measured
  optimum. A smaller width costs more than a larger one, because each column
  tile streams and re-decodes a full slice of a weight row.
- For the low-bit expert formats, the main prefill path is the F16-tile WMMA
  route. `RoutedF16GEMMKernel` uses `kLowBit` sub-block staging to dequantize
  rows of the eight low-bit expert formats into F16 tiles with the shared
  `DecodeQuantSub16` decoders (`src/models/qwen/hip/quant_ops.hpp`). The
  paired gate/up SwiGLU fusion runs in the same pass. The IQ2/IQ3 sub-block
  decoders build four signed-byte words from the grid entries one group at
  a time (two 64-bit IQ2 words or four 32-bit IQ3 words per 16-element
  sub-block). They keep
  the sign bytes in registers and negate per byte; Q2_0 expands a packed
  code word. Magnitudes and signs are decoded per group, not per element.
  Q5_1 rows stage their 24-byte blocks as stored in shared memory (LDS);
  the scale, high word and codes are read per K block. K-quant and IQ rows
  stage codes plus their per-block affine. The paired gate/up path uses one
  design throughout: cached headers, `cndmask`-based code-group selection,
  the F16 LDS pair epilogue and its `HeaderByte` header decoder.
- Within the MMQ path, wide batches use the 64-token down-projection map and
  the 128-token paired gate/up map (`routed_64_tiles_` / `routed_pair_*`).
  `WmmaExperts` selects the path; its condition includes the `RoutedF16Type`
  check. (`WmmaExperts` is upstream's helper shape.)
- The id-map bounds scan reads the per-expert counts without modifying them
  and writes each expert's block base next to its count. The MoE epilogue
  folds into the combine through the `moe_pending_` deferral: the residual
  (`kPrior`) and the fused inject (`kInject`) are carried until
  `HcApplyDeferred` applies them. The routed SwiGLU is computed during the
  down-projection activation quantization (`quantize_mmq_q8_1` with `x_up`),
  which avoids an extra F32 round trip. The combine's square-sum uses one
  pinned `__fmaf_rn` chain in both the deferred and separate paths.
- Q2_0 is wired end to end inside the vendored layer (recorded in
  `VENDOR.md`): `block_q2_0` and `QK2_0`/`QI2_0`/`QR2_0` in
  `ggml-common.h`; `vec_dot_q2_0_q8_1`, which decodes 2-bit groups per lane
  with `__builtin_amdgcn_perm`; `load_tiles_q2_0`, which unpacks blocks into
  a Q8_0 tile so the Q8_0 vector dot products apply; and the
  `mmq_type_traits` and tile/DS-layout cases in
  `mmq.hpp`/`common.hpp`/`vecdotq.hpp`.
- `moe_wave64.hip.cpp` needs no changes: it is a Q4_K-only batch kernel with
  no type check.

### Dense projections, heads and embeddings

- `qfn_mmq_dense(weight_type, …)` handles the dense and shared-expert set:
  Q8_0, Q2_0, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S,
  IQ4_NL and IQ4_XS. `qfn_mmq_dense_vec(M, N, K)` (same argument order as
  its siblings) routes through `moe_vector_projection` with
  `n_experts = n_expert_used = 1` and a pool-allocated zeroed ids buffer. It
  accepts the same formats as the gated path; Q5_1 is rejected by both, so
  decode and prefill cannot disagree.
- `Executor::Dense` keeps the Q8_0 fast path and sends every other
  non-F32/BF16/F16 matrix through the adapter (MMVQ below `MatrixRows`,
  MMQ above). The generic fallback rejects other types with an error, and the
  error path distinguishes an unsupported format (probed through
  `qfn_mmq_moe_gated_type`) from a launch failure. `DenseBatch` sends
  quantized rows through one `qfn_mmq_dense` call.
- Decode projections build the shared Q8_1 activation rows only when a
  consumer is Q8_0. The attention Q/K/V, SSM QKV/gate and PLE key/value
  groups quantize once for their Q8_0 consumers; every other format
  quantizes the F32 rows in its own entry, so the GSQ dense groups run no
  unused shared pass.
- Q6_K is fully supported by the adapter (`WeightType::kQ6_K = 14`; the
  MMVQ, MMQ and gated cases exist).
- Wide dense GEMMs (prefill): `DenseF16GEMMKernel` decodes the k-quant/IQ
  weights into F16 and feeds the shared LDS/WMMA path used by Q8_0
  (`DenseQuantF16Gemm`; the executor stages activations to F16 through
  `PrepareHalfInput` at wide `MatrixRows`). The int8 WMMA route
  (`KQuantWmmaGemm`/`Q8ActFillSums`) is not used. Rebuilding it over the
  per-lane decoder family at the 256-row stack geometry measures slower:
  the per-accumulator-element fixups cost more than the 2x int8 WMMA rate
  gains (see "Performance state").
- Wide prefill SSM projections fuse the four-tap convolution into the
  quantized projection (`DenseQuantF16SsmGemm`, with `kSsmConv` epilogue
  instantiations for the K-quant/IQ `ssm_in` types) behind the executor
  check `KQuantWmmaType(ssm_in.type) && cols <= max_half_cols`, alongside
  `DenseF16Route`. Mixed-type layers keep QKV and gate separate; their
  standalone [10240][2560] QKV projection takes the same fused epilogue
  behind the check on `ssm_qkv.type`, and the independently typed gate
  keeps its own projection. The epilogue reads only the accumulated tile,
  `conv_w` and the history, so it does not depend on the weight format. The
  Q8_0 route keeps each wave's decoded weight rows in registers; the GSQ
  stacks stage their decoded fragments in shared memory (`LDS`), because
  the register variant is slower for them (see "Performance state").
- `UnquantizedF16Gemm` handles F16 weights or BF16 weights treated as F16,
  with F16 activation rows, in 64-row × 128-token tiles (the narrow shape
  also used by W8A8Gemm). The accumulation order along K is fixed per
  output, so results stay bit-exact across every prefill chunk width. It
  serves the HC down fusion, the BF16 indexer/alpha-beta projections and the
  tuned F16 blaslt shapes.
- Embeddings: `EmbeddingType`/`SmallType` and the gather (`RowElement`/
  `RowBytes` via `gufo::hip::QuantBlockElement`) accept the GSQ `token_embd`
  types. Unmapped types trap on the device (`__builtin_trap()`) and throw on
  the host. The LM head (Q5_K/Q6_K) runs through the same dense dispatch.
- Decode with few rows: `mul_mat_vec_q_dense_multi` (`mmvq.hip.cpp`) shares
  weight rows across the token batch. For dense projections at 2-8 tokens,
  one warp computes all tokens for its two weight rows; the per-row and
  per-token dot products match the one-token waves. Routed calls never use
  this path; it is enabled only when `n_experts == n_expert_used == 1`.
- 16-bit few-row long-K GEMVs (the BF16 `hc` down projection at [320,
  10240]): `SmallGemmWideKernel` computes two weight rows per wave (m ≤ 512,
  k ≥ 8192, 1-8 tokens). With four rows per wave, the 320-row shape leaves
  too few resident waves; two rows raise cold-cache throughput from 82 to
  116 GB/s. The control's Q8_0 `hc` down projection keeps its specialized
  `qfn_q8_hc_down` path.

### Stacked attention and SSM projections

The `stackable` check in `device_model.cpp` accepts the same quantized dense
formats as the executor's dense dispatch, so the two cannot disagree. `Stack`
concatenates those formats the same way as Q8_0: the same zeroed
`kTailMargin` tail; `max_q8_cols` stays unchanged for Q8_0; and the K-quant
stacks also register `max_half_cols`, from which the F16-tile route sizes
activation rows. Same-type `attn_q`/`attn_k`/`attn_v` triplets merge into one
[13312][2560] GEMM, whose output the stacked path fuses (`Dense` +
`PrepareAttention`). Same-type `ssm_qkv`/`ssm_gate` pairs merge over the
`ssm_in` layout. Mixed-type groups stay separate: the exactness
requirement that operator outputs match a CPU reference forbids requantizing
weights, so only groups whose projections share one type are merged.

### Hyper-connection (BF16 mixers)

Three fused paths accept BF16 weights, where UD-Q4_K_XL stores Q8_0:

- Inject fusion: the mix epilogues (`HcMixEpilogue`/`Vec4`/`Vec4F16` and
  `HcMixF16Gemm`'s inject launch) and the deferred wide combines
  (`HcCombineF16`/`HcCombineMoeF16`) take `WeightType inject_type`
  (kF32/kBF16) and read BF16 through the exact `Bf16ToF32` expansion
  (`InjectElement`/`InjectLoad4`). `HcInjectPart` writes the deferred
  combine's next-mixer partials for both encodings. `fused_inject` accepts
  kBF16 in `HcMix` and `HcMixBatch`.
- Wide route and gate-mix: `wide_mixer_` accepts a BF16 down projection; the
  normalization output is staged as F16 (W8A8 uses the tiled-Q8 norm path).
  `fused_projection` accepts BF16 down and up projections. The down
  projection reads the F16 norm through `UnquantizedF16Gemm` with
  `kStageBf16` on `DenseF16GEMMKernel`; BF16 rows are rounded to F16 when
  written to shared memory, which is exact for every BF16 value in F16's
  range, and the route keeps the same two-chain K order as the F16 path.
  `HcMixF16Gemm` runs the up projection as `kHalfWeights + kStageBf16` on
  the 128-row `kHcMix` tile (32 hidden channels per tile across the four
  streams, with the same two K16 chains and 128-token width as the Q8_0
  256-row specialization). The SiLU'd rank rows reach the gate buffer as
  F16 through SiluScale + NarrowActivations. F32 down projections
  re-normalize into the F32 scratch buffer on the wide route.
- Two pinned caveats: the mixed row's tiled-Q8 copy is taken only when a
  following Q8_0 dense projection consumes it and `hidden <= max_q8_cols`.
  A GSQ upload without the Q8_0 MTP sidecar has `max_q8_cols == 0`, which
  is how the probe runs; chat attaches the sidecar automatically. The
  epilogue's mix and inject FMAs are pinned with `__fmaf_rn`, because the
  build uses `-ffast-math` and reassociation would break the
  byte-exactness contracts.
- The wide combine writes the next mixer's F16 norm for every consumer and
  its tiled Q8 copy only when that mixer's down projection is Q8_0; the GSQ
  BF16 down projections read the F16 norm. The mix epilogue likewise writes
  the mixed row's tiled Q8 copy only when a following dense projection is
  Q8_0 (an attention or SSM stack, the router, or the shared expert). The
  fused MoE combine accepts an omitted Q8 norm and skips only its
  quantization and store, so its residual, normalization, deferred-update
  and inject paths run unchanged.
- The BF16 hyper-connection weights are within |w| ≤ 8.56, so F16 staging
  cannot overflow. 823 of 640 million values are below 6.1e-8 and flush to
  zero in F16 staging, contributing less than 5e-5; this is accepted.

### MTP catch-up (Q8_0 catch-up branch)

The shared-Q8_0 MTP sidecar makes every mix use `Moe`'s Q8_0 catch-up
branch. Its pinned behaviors:

- The branch sets both routed id maps to -1 first; otherwise stale
  prompt-batch histograms scatter garbage rows into live output slots.
- The one-token down-projection GEMM writes into the caller's F16-stride
  view. `RowScratch` advances by whole floats (the adapter's F32 layout),
  which is twice the F16 row stride, and it cannot know the route.
- `moe_pending_` and the deferred-epilogue flags are restored when
  `UseScratch` is restored.

### Multi-session batched decode (more than 8 rows)

Non-Q8_0 dense projections are split into chunks of 8 rows on the serial
MMVQ path (`MMVQ_MAX_BATCH_SIZE = kDecodeRows = 8`), so each weight matrix
is swept ceil(rows/8) times. Verification must match single-session decode
row by row, and the wide MMQ path only approximates that result
(|MMVQ − MMQ| < 2e-2 · scale, pinned). Single-user AR and MTP stay at 8
rows or fewer. If batch throughput is ever pursued, the workable design is
a single serial-path launch for all rows (`qfn_mmq_dense_vec` tiles
arbitrary token counts through `grid.z`; per-row equality across token
counts is pinned), not a fixed-width MMQ path.

## Exactness and determinism contracts

These are the contracts that keep results reproducible. *Pinned* means
enforced by a test or by an arithmetic form that must not change.
Breaking any of them silently breaks session or audit replay.

- Operator tests compare each type and role (dense, routed, paired, gated,
  head, embedding) exactly against F64 references computed by CPU
  dequantization. The 19 `qwen38_flash_next.*` ctest suites are 12 HIP
  operator suites (`mtp_ops`, `blaslt_ops`, `hc_mix_ops`, `gdn_ops`,
  `attention_ops`, `moe_epilogue_ops`, `moe_ids_ops`, `ple_ops`,
  `projection_ops`, `routed_wmma_ops`, `select_ops`, `snapshot_copy`) plus
  the CPU suites (`config`, `continuation_layout`, `draft_vocab`, `ngram`,
  `mtp_sampling`, `prompt_lookup`, `weights`).
- FMA and accumulation: every shared MMQ vector-dot accumulation is pinned
  with `__fadd_rn`/`__fmaf_rn` (`mmq.hpp` typed cases and `vecdotq.hpp`).
  Tests pin chunk-width exactness and the dense MMQ tile width. The reason
  is that `-ffast-math` reassociation (one to eight units in the last place)
  breaks downstream byte-exactness assumptions (HC epilogues, batch
  invariance).
- Invariance checks: routed batch-width exactness (a 224-row batch vs a 96-row
  final chunk), MMQ tile-width invariance (`CheckMoeWidthExact`),
  dense-vector rows bit-identical across 1-8 tokens (`CheckDenseVecTokens`),
  stacked results bit-exact against separate projections per row for every
  consuming path (`CheckDenseStackExact`), paired gate/up bit-exact against
  separate projections per type, and decode 16-bit GEMVs bit-exact against
  the CPU-expanded reference (`CheckDecodeGemv`).
- Quantized fused SSM convolution: `CheckSsmProjectionQuant` pins the fused
  `DenseQuantF16SsmGemm` pass bit-exact against the separate
  `DenseQuantF16Gemm` projection for each stacked K-quant/IQ format (same
  decode, same K order per output), through its GatedDeltaNet consumer, at
  a full and a partial final token tile. The standalone QKV case uses an
  independently typed gate and covers 1,024-, 1,025- and 2,049-token
  batches, a nonzero rolling history and mid-tile checkpoint prefixes;
  projection, convolution, recurrent state, history and checkpoint bytes
  match the separate chain. Fixtures verify rejection of unsupported
  shapes.
- The `UnquantizedF16Gemm` route keeps one K order per output:
  `CheckBf16Gemm` pins the BF16 route bit-exact against the F16-staged route
  across a prefill chunk split, and `blaslt_ops_test` pins the F16 route
  across chunk boundaries.
- HC BF16 byte-exactness: `CheckBf16Down`, `CheckFusedProjectionBf16`,
  `CheckBf16Inject` and `CheckCombineBf16Inject` compare against
  CPU-expanded F16/F32 references; the double-precision CPU mix reference
  stays within 5e-5/1e-4. The deferred combine's fused BF16 inject partials
  are byte-exact against the separate BF16 epilogue over the same F16 norm,
  and the fused residual and norm bytes are unchanged.
- Optional HC side outputs: the norm's tiled Q8 copy and the mixed row's
  tiled Q8 copy are written only for consumers that read them. A requested
  copy keeps its exact bytes; an omitted copy is left untouched while the
  residual, F16 norm and inject partials stay bit-identical. The fused MoE
  combine covers the gamma-null case and the deferred and separate paths at
  one-row, 96-row and 2,049-row batches.
- MoE: the fused epilogue residual is byte-exact against
  `MoeEpilogueVec4 + HcCombine`. The combine's square-sum and gamma-scale
  multiplies use the same pinned `__fmaf_rn`/`__fmul_rn` form in the
  deferred (`kPrior`) and separate paths; a split form diverges at Q8
  rounding boundaries in this compilation context. The
  routed SwiGLU-in-down-quantize codes are bit-identical to `SwigluKernel`
  followed by the plain entry for all thirteen routed formats
  (`CheckSiluDownMmq`). The grouped gated vector's IQ2/IQ3 fragment reuse
  keeps each format's integer scale formula, `dp4a` order and float scale
  product. The float scale product uses the same multiplication order as the
  scalar routed path, enforced by an inline-assembly barrier (`IqMoeScale`),
  so the output is byte-exact against the separate projections plus SwiGLU
  at every width, including duplicate and invalid slots and nonfinite scales
  (`routed_wmma_ops`).
- Type-id checks: every `WeightType` value is pinned against both
  `core::GgmlType` and the adapter's `ggml_type` in enum order (the raw-cast
  check in `routed_wmma_ops_test`); GGUF type ids are pinned in
  `ggml_dequant_test`. A moved id fails to compile instead of silently
  misrouting.
- Whole-block rules: MMQ entry points require whole weight blocks
  (`K % block_elements == 0`); Q2_0's 64-weight block with K ≡ 32 mod 64
  must fail, not truncate. Q2_0 tail cases at k=64 and k=320, and dense
  tight-K cases, carry zeroed weight tails. Tests verify that Q5_1 and Q4_0
  are rejected at the executor probe and at both dense entry points.
- MTP audit: `--mtp-audit` pins the reference check's down-projection inputs
  to the GPU's SwiGLU outputs (`stage_inputs`), with loose stage comparisons
  (2e-3) and a tight `ffn-experts` comparison (5e-5). Without this,
  ulp-level GPU-vs-F64 noise flips one Q8_1 code and amplifies to 5e-4
  divergence. This is a property of the probe only; production runs on the
  GPU alone and is deterministic.
- Draft-vocabulary checks compare the small-head greedy continuation and
  final logits byte-exactly with full-head AR. The MTP audit compares real
  table-row scores and remapped rare-token IDs with an independent full-head
  projection, including empty-slot masking, table replacement, scalar heads,
  all-small-head batches and mixed full/small-head batches. Its one-row
  prefix guarantees rare-token candidate coverage independently of generated
  language or neural-draft acceptance rate.
- Determinism: greedy traces are deterministic per binary. The depth-32768
  128-token benchmark completion is bit-identical (sha `d856f78e…`) across
  all three models, AR and MTP. Depth-0 traces diverge per model and binary
  in the known borderline-argmax case (not a correctness or speed issue).
  MTP cycle counts are deterministic per binary.
- Quality checks: `gpu_probe --reference` argmax matches the F32 reference on
  both mixes; KL divergence is 4e-4 to 2e-2 per mix, with a few
  deterministic borderline argmax flips in the known case.

## Verification

From a clone with the GSQ-RCO GGUFs (and, for control comparisons, the
unsloth UD-Q4_K_XL GGUFs):

```sh
# Tests (RelWithDebInfo + assertions)
cmake --preset gpu-test && cmake --build --preset gpu-test
ctest --preset gpu-full -R '^qwen38_flash_next\.' --output-on-failure   # 19 suites
ctest --preset gpu-full -R 'ggml_dequant_test|ggml_gemm_parity_test|qwen_gemm_route_test'

# Measurement binary
cmake --preset release && cmake --build --preset release   # -> build/release/gufo
```

- Session replay: `qwen38_flash_next_session_test` runs with a model path:
  `session_test --model FIRST.gguf --mtp-model MTP.gguf` on xxs/s/q4. It
  covers prefill chunking, image prefill, graph capture/replay, batched
  sessions and serving modes; focused flags exist for each area.
- MTP audit: `gpu_probe --model FIRST_SHARD.gguf --prompt TEXT --mtp-model
  MTP.gguf --mtp-audit` on xxs/s/q4.
- Reference probe: `gpu_probe --model … --prompt … --reference`.
- Functional API baselines: `tests/functional/run.py --record-baseline`
  then compare; suites `discovery responses conversation batch
  sampling-defaults state-edges`, modes `off` and `mtp` (`off` = zero
  drafts, `mtp` = real draft execution).

## Performance state

Measured with the release binary on an AMD Strix Halo (`gfx1151`).

Benchmark setup: `gufo bench`, greedy, `-p 2048 -n 128 -d 0,32768 -r 1 -v`,
single user. Depth (`d`) is the number of context tokens before generation;
`tg128` is 128 generated tokens and `pp2048` is a 2048-token prompt.
The shared Q8_0 MTP sidecar is attached in MTP mode for the MTP measurements
and detached (`--speculative off`) for the AR measurements. One binary runs
all three models. Each entry is the difference from the unsloth UD-Q4_K_XL
control measured in alternating rounds on that binary (two rounds, r1/r2).
Negative values mean the GSQ mix is slower than the control.

Δ vs UD-Q4_K_XL control (alternating rounds r1/r2 unless marked steady or
focused depth 0):

| Measurement | xxs | s |
| --- | ---: | ---: |
| AR tg128, steady state | ≈ -1% | ≈ -6% |
| AR tg128 at d32768 | -1.6/-2.1% | -6.5/-6.7% |
| MTP tg128 at d0 (acceptance-rate dependent, focused run) | -10.2/-11.5% | -19.9/-20.3% |
| MTP tg128 at d32768 | -2.6/-2.8% | -6.5/-5.8% |
| pp2048 at d32768, AR (speculation off) | -16.7/-15.8% | -17.3/-16.3% |
| pp2048 at d32768, MTP | -15.4/-15.4% | -16.2/-15.8% |
| pp2048 at d0, steady state | ≈ -19..-21% | ≈ -19..-21% |

### Decode

- Decode results: AR token generation is within 7% of the control. Steady
  depth-0 measurements are about -1% (xxs) and -6% (s);
  depth-32768 measurements are -1.6/-2.1% (xxs) and -6.5/-6.7% (s).
  Depth-0 AR
  measurements carry launch and run noise: individual runs from the
  multi-depth benchmark read as low as 22.7 t/s (xxs) and 21.8 t/s (s),
  focused reruns that alternate models read about 26.7 t/s (xxs) and
  25.2-25.5 t/s (s), and the control itself reads 23.5-27 t/s across
  focused runs. MTP depth-0
  measurements follow the draft acceptance rate rather than kernel speed
  (the control reads 37-44 t/s across rounds; a depth-0 run can take an
  unusual draft trajectory at the same cycle count, which moves the result
  with the acceptance rate). At depth 32768, upstream's prompt-lookup draft
  copies bypass the neural draft head for greedy runs (the two mixes and the
  control take the same 17 cycles and accept 111 draft tokens), so this
  measurement reflects the multi-row verification forward pass. The
  roughly -2 to -3% (xxs) and -6% (s) gap is the low-bit formats'
  per-element decode at eight verification rows, with shared-expert
  weight-fragment reuse. The routed gate/up stage is 434 ms per 17
  verification cycles of the copied 32K prompt on xxs, at that stage's
  decode-bound limit. The gap is not a draft, attention or dispatch
  problem.
- Decode limits: standalone saturation measurements per format, using the
  true GGUF block layouts and a CPU dot-product check for each measurement,
  give a practical DRAM read peak of 216.2 GB/s. The Q6_K head MMVQ reaches
  98-100% of it, so it has no headroom (the control's Q8_0 head moves more
  bytes per token at the same peak). The low-bit expert MMVQs reach 85-95%
  of the maximum measured bandwidth of that format (512 experts in
  use, paired gate/up: Q2_0 214, IQ4_NL 211, Q4_K 211, Q5_K 211, Q6_K 213,
  IQ4_XS 208, IQ3_S 153, IQ3_XXS 138, IQ2_S 118, IQ2_XS 110, IQ2_XXS
  102 GB/s; down: IQ4_NL 212, Q2_0 185). Q2_0 and IQ4_NL reach the DRAM
  peak. The IQ2/IQ3 family stays below it because its vector dot products
  decode through a per-lane grid lookup table; the gap is larger than wider
  loads or better wave scheduling can recover, so the lookup cost is that
  family's limit. The one clearly slow decode shape is the BF16 `hc`-down
  16-bit GEMV (629 MB at 84 GB/s, against the control's Q8_0 320 MB at
  182 GB/s). Its two-rows-per-wave form reaches 116 GB/s with a cold cache;
  the faster double-tile variant is not worth the extra code.

### Prefill

- Prefill: the about -16..-23% gap is compute-bound, not byte- or
  launch-bound. The dequant-to-F16 route runs at the F16-WMMA limit at these
  shapes (the wide HC down projection moves about 25% fewer bytes for about
  3% less time). Prefill execution has 5-8% idle time on both mixes, so it
  is not launch-bound. The mixes issue 21% more dispatches than the
  control, which follows from the mixed-type stacks. The unquantized route
  keeps its 64×128 tile; upstream's <128,128> wide tile is not used for it
  (64×128 covers the same shapes, and the control's prefill shows no
  regression). In a warmed-up xxs pp2048 forward pass, HC-up takes 70.5 ms
  (control 51.6 ms; the BF16 128-row tile keeps both K16 chains
  spill-free), the 16-bit-weight HC-down family costs 157 ms more than the
  control (the BF16 down projection alone is about 158 ms, against the
  control's 48 ms int8 WMMA specialization), mixed-type dense stacks cost
  166 ms more, and routed gate/up costs 20 ms more. Depth-32768 prefill
  measurements are about -15..-17%. Depth-0 measurements fall in
  about -19..-21%, and their spread follows the control's own depth-0 rate
  (a lower control rate widens the gap and a higher one narrows it; the
  control's prefill rate varies more between rounds than the mixes'). That
  spread is measurement variance, not a difference in the execution path.

### Rejected options

- Options that measure slower (not built, and why): the HC up conversion is
  not limited by conversion throughput (`v_cvt` is 2.8% of the kernel body;
  an F16-weights variant of the same shape runs 3.2x faster on 1.9x the
  bytes, so the remaining cost is the staging structure at k=320 plus the
  format's bytes). There is no bit-field conversion path. The dense
  compact-Q8 int8 WMMA route (per-lane decoders, 256-row stack geometry)
  loses its first A/B comparison: IQ4_XS stacks run at 26.0 TF against the
  F16 route's 30.0, Q6_K at 20.1 against 28.3, and Q4_K at 21.4 against 28.3
  (a BM=128 variant measured 856 against 1108 tokens/s). The
  per-accumulator-element fixups cost more than the 2x int8 WMMA rate gains.
  The routed compact-Q8 path depends on this route and is not built. The
  BF16 hyper-connection down projection [320, 10240] keeps the 64×128 tile
  with 64-element K stages. A 32-token/256-element-stage variant measures
  about 180 ms and a 128-token/128-element-stage variant about 160 ms,
  against 158 ms in the warmed pp2048 forward pass. The smaller token tile
  multiplies BF16-to-F16 weight staging per token, and at 128 tokens the
  halved barrier count does not offset the larger stage. The register-held
  weight path for SSM projections is Q8_0-only. The register variant
  measures 53.9-54.3 ms (xxs) and 70.2 ms (s) on the warmed pp2048 forward
  pass, against 48.7-49.2 and 66.2 ms with LDS staging. The half-wave
  exchange costs sixteen lane permutes per K32 block, where the staged path
  costs twelve LDS instructions, and the low-bit decode itself requires the
  LDS round trip.
- The 64-row narrow tile and the BK=2/WM=8/WN=1 plan are slower for the
  K-quant/IQ dense route. At 2048 tokens the narrow tile reads 4.6-4.7 ms
  against 3.8 ms for the 512-row K/V and indexer projections and is flat at
  640 rows (24.5 against 24.7 ms). The BK=2 plan is flat at the 13312-row
  attention stacks (18.3 against 18.1 ms) and 2-13.6% slower on the separate
  6144-, 10240- and 12288-row projections, worst for IQ4_XS at 12288 rows
  (16.9 against 14.9 ms). The 256×128 tile with one K block per stage and
  four-by-two wave groups keeps every quantized dense shape.

### Control

- Control health: UD-Q4_K_XL shows no regression relative to `main`.

## Vendored-code obligations

`kernels/rocm/mmq/VENDOR.md` records everything that must be re-applied
when the vendored code is updated:

- Q2_0 additions to `ggml-common.h`, `mmq.hpp`, `vecdotq.hpp` and
  `common.hpp`.
- The `__fadd_rn`/`__fmaf_rn` pinning of every shared MMQ vector-dot
  accumulation, for chunk-width exactness.
- `mmq_args::mmq_x_request` with the host-side
  `dense_tile_cols`/requested-width selection; the routed tile-width path
  uses it.
- `quantize_mmq_q8_1`'s `x_up`, which forms the routed SwiGLU during
  down-projection quantization.

Policy: keep edits to vendored `mmq.hpp`/`ggml-common.h` at or near zero;
re-imports must re-apply the recorded blocks.

## Known quirks and accepted limitations

- `weights.cpp` `FormatOf` block geometry duplicates
  `quant::QuantizedBlockElements`/`QuantizedRowBytes`; this is an intentional
  duplicate and the values agree.
- Q2_0 is a non-upstream type id (42); keep it isolated in the quant layer.
- Multi-session batched decode sweeps non-Q8_0 dense projections again for
  every 8 rows (by design; see above).
- `GdnRowSplitKernel`'s 112 B of scratch belongs to upstream code, not this
  feature.
- `DenseBatch`/`GatedDenseBatch` executor-level coverage comes from session
  replay, not direct operator tests: these are private `Executor` members
  behind a full `DeviceModel`, and a synthetic test hook would add
  scaffolding that the fork's minimal-diff policy rules out.
- MTP depth-0 greedy traces are deterministic per binary but not stable
  across binaries (the borderline-argmax case; accepted).
- The s-mix full test suites have a known seeded-sampling borderline case: a
  session-replay cache check and one `--mtp-audit` candidate check can flip
  between process runs. Both pass on rerun. Logits are pinned bit-identical
  across the affected paths, so this is sampling-boundary noise, not a
  numerics problem.

## File map

| Area | Files |
| --- | --- |
| Type layer | `src/core/gguf_reader.hpp`, `src/core/quant/ggml_dequant.{hpp,cpp}` |
| Loader | `src/models/qwen38_flash_next/weights.cpp` (`FormatOf`, role allowlists), `device_model.cpp` (upload checks, `Stack`, `kTailMargin`) |
| Adapter (gufo-owned) | `kernels/rocm/mmq/qfn_mmq.h`, `qfn_mmq.hip.cpp`, `mmvq.hip.cpp`, `quantize.hip.cpp` |
| Vendored MMQ | `kernels/rocm/mmq/{mmq.hpp, mmvq.hpp, vecdotq.hpp, common.hpp, ggml-common.h, …}`, `VENDOR.md` |
| Shared decoders | `src/models/qwen/hip/quant_ops.hpp` (`DecodeQuantSub16`, packed grid-group decode) |
| Executor / dispatch | `src/models/qwen38_flash_next/executor.cpp` |
| Other kernels | `kernels.hip.cpp` (`DenseQuantF16SsmGemm`/`SsmConvBoundaryKernel`, `SmallGemmWideKernel`), `moe_wave64.hip.cpp` (Q4_K-only, no type check), small-batch/prefill waves under `src/models/qwen/hip/kernels/` |
| Operator tests | `tests/models/qwen38_flash_next/*_test.cpp`, `mtp_audit.cpp`, `gpu_probe.cpp`, `reference_probe.cpp` |
| Core quant tests | `tests/core/ggml_dequant_test.cpp` |
| Benchmark and profiling tools | `tools/bench/`, `tools/prof/prof.py`, `tools/prof/isa_mix.py` |
