# GSQ-RCO support in gufo — state of the solution

Branch `feat/gsq-rco` (off `main`).

This is the state-of-the-solution reference for the fork's GSQ-RCO feature.
It describes what the solution is now — types, loader, kernel dispatch,
exactness contracts, verification, performance, file map — not how it got
there; the branch history carries that. The support is fork-only:
`feat/gsq-rco` is a long-lived rebased patch series on this fork, kept
current over upstream by rebase.

## Summary

gufo runs the ISTA-DASLab GSQ-RCO GGUFs of Qwen3.8-Flash-Next — IQ3_XXS ("xxs",
3.00 bpw) and IQ3_S ("s", 3.50 bpw) — with the same feature set as the
unsloth UD-Q4_K_XL target: AR and MTP generation (shared-Q8_0 MTP sidecar),
image input (BF16 mmproj sidecar), chat template, graph capture, batched
sessions, serving modes. All in-repo quality gates are green on both mixes, and
the UD-Q4_K_XL control is regression-free vs `main`. Speed (see
"Performance"): AR token generation is at parity or within about 7% on both
mixes; MTP token generation at depth 32768 times the multi-row verification
forward, because upstream's prompt-lookup draft copies bypass the neural draft
head for greedy runs (identical 17 cycles / 111 accepted across all three
models): the residual (about -3..-3.5% xxs, -6..-6.5% s) is the low-bit
formats' per-element decode at eight verification rows with the shared-expert
weight-fragment reuse in place, not a draft, attention or dispatch deficit.
MTP tg at depth 0 rides the
draft-trajectory measurement class — accept-rate driven, not kernel speed.
Prefill sits at about -19..-24%, measured compute-bound — the residual is the
formats' F16-WMMA arithmetic cost, not a byte, launch or dispatch deficit.

## Targets and artifacts

| What | What it is |
| --- | --- |
| GSQ-RCO GGUFs | [ISTA-DASLab GSQ-RCO releases](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) of Qwen3.8-Flash-Next: IQ3_XXS ("xxs", 3.00 bpw) and IQ3_S ("s", 3.50 bpw), each with the shared MTP Q8_0 and mmproj BF16 sidecars |
| Control | unsloth UD-Q4_K_XL (4 shards) — the speed and quality baseline this feature is held against |
| Reference implementation | GSQHalo.cpp (llama.cpp fork) — read-only reference for the formats and kernel ideas |

Facts that shape the code:

- Shard 2 of a GSQ mix is a single tensor: `per_layer_token_embd.weight`
  (the PLE table), IQ4_NL — shard discovery tolerates unbalanced shards.
- Images/MTP use the shared Q8_0/BF16 sidecars; they attach by identity keys,
  not weight types, and work on GSQ targets.
- A mix stores per tensor: dense attn/SSM/shexp projections in IQ4_XS, IQ3_S,
  Q6_K, Q5_K, Q4_K, IQ3_XXS or IQ2_S; routed experts across Q2_0, IQ2_XXS,
  IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, Q4_K, Q5_K, Q6_K; hc_*
  hyper-connection mixers in BF16; token_embd IQ3_S/IQ4_XS; output Q5_K (xxs)
  or Q6_K (s); ple_conv1d F16; router and GDN alpha/beta BF16.
- Only 4/12 (xxs) and 5/12 (s) attention layers and 8/24 SSM layers per mix
  are same-type, so attention/SSM stacking merges only those groups
  (see "Stacked attention and SSM projections").

## Type layer (`src/core`)

- `gguf_reader.hpp` `GgmlType` gained `kIQ2_XS = 17`, `kIQ3_XXS = 18`,
  `kIQ2_S = 22`, `kQ2_0 = 42` — the values are GGUF ids, existing ids are
  untouched (never renumber). Q2_0 is a non-upstream id (GSQHalo's
  `block_q2_0`: fp16 scale + 16 code bytes, 64 weights per block); keep it
  isolated in the quant layer.
- `quant/ggml_dequant.{hpp,cpp}`: block structs, row sizes, CPU dequant and
  f32 dot for IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS and Q2_0,
  with grid/sign tables verbatim from GSQHalo's `ggml-common.h`. The CPU
  dispatchers (`quant::Dequantize`/`quant::Dot`, `cpu::DequantizeRow`) route
  all of them; unknown types abort rather than silently zero-fill.
- `DescribeQwenGemmFormat` lists the new types and sets `block_elements` for
  the quantized-only group (so a direct-flag flip cannot hit
  `kMisalignedQuantK` by omission).
- Tests: `tests/core/ggml_dequant_test.cpp` — hand-built blocks, row-size
  contracts, discriminating vectors (IQ2_S qh high grid bits, IQ3_XXS
  grid1/grid2 halves, sign indices beyond 0/1), and a `GUFO_GSQ_MODEL`-gated
  row oracle over the real shards (frozen samples per type; verified bit-exact
  against an independent port of GSQHalo's `ggml-quants.c` arithmetic).

## Loader and upload gates

`src/models/qwen38_flash_next/weights.cpp`:

- `FormatOf` block geometries (block elements/bytes): Q2_0 {64,18}, IQ2_XXS
  {256,66}, IQ2_XS {256,74}, IQ2_S {256,82}, IQ3_XXS {256,98}, IQ3_S {256,110},
  IQ4_XS {256,136}. (Duplicates `quant::QuantizedBlockElements`/
  `QuantizedRowBytes` — pre-existing pattern, values agree.)
- Per-role allowed-type lists, exactly the maps-union plus the previous
  Q4_K_XL set, nothing wider:

| Role | Types beyond the pre-existing set |
| --- | --- |
| dense (attn_*, ssm_qkv/gate/out, indexer, PLE key/value, nextn.eh_proj) | Q4_K, Q5_K, Q6_K, IQ4_XS, IQ3_S |
| shexp (ffn_*_shexp) | dense set + IQ4_NL, Q2_0 |
| experts (ffn_*_exps) | IQ4_NL, Q2_0, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS |
| embd (token_embd) | IQ3_S, IQ4_XS |
| output | Q5_K |
| router, ffn_gate_inp_shexp, GDN alpha/beta | BF16 |
| ple_conv1d | F16 |
| hc_*, ssm_alpha/beta | BF16 (already accepted) |

`device_model.cpp`:

- `Stack` accepts BF16 stacks (router and GDN alpha/beta run through generic
  BF16 blas as stored).
- The Q6_K capability gate allows the LM head (Q6_K head kernels exist) and
  deliberately rejects Q6_K token_embd and routed experts — no supported mix
  stores Q6_K there and the expert paths are untested; keep it strict.
- `ple_conv1d` is F16 in GSQ; both consumers decode by the tensor's encoding
  (a past bug read it as `float*` — half the bytes, OOB reads, NaN logits).
- Weight uploads carry a zeroed `kTailMargin` tail: the wide MMQ tier reads
  whole 256-element k-iterations past short rows, and an unzeroed tail
  poisons accumulators (0 × inf) so `qfn_mmq_sanitize_f32` zeroes whole rows.
  Test uploads must carry a zeroed tail too.
- Whitelist CI: `qwen38_flash_next.weights` asserts each role's accepted types
  plus negative fixtures per role class.

## GPU execution paths

Kernel sources live under `src/models/qwen38_flash_next/kernels/rocm/`
(`mmq/` for the vendored llama.cpp MMQ plus the gufo-owned adapter,
`kernels.hip.cpp` for shared GEMV/GEMM kernels). The vendored policy:
`mmq.hpp`, `ggml-common.h` etc. stay near-zero-diff; gufo owns `qfn_mmq.h` /
`qfn_mmq.hip.cpp`, `executor.cpp`, `weights.cpp`, `core/quant`. All local
additions to vendored files are recorded as re-import obligations in
`kernels/rocm/mmq/VENDOR.md` (see "Vendored-code obligations").

### Routed experts (MoE) — the hot path

- Formats: gate/up IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, Q2_0
  (plus the pre-existing Q4_K/Q5_K/Q5_1/Q8_0); down IQ4_NL, Q2_0 (plus
  Q5_K/Q5_1/Q8_0).
- Decode (≤ 8 rows): MMVQ — `moe_vector_projection` covers every routed
  format (its supported set is the gated set plus Q5_1); the fused SwiGLU
  gated vector path in `mmvq.hip.cpp` runs for every routed format except
  Q5_1; the vec paths take rows ≤ 8 except Q4_K. The grouped gated kernel
  walking a group's shared expert decodes its weight fragment once per row
  and K block and reuses it across the group's tokens: `Q4MoeFragment` for
  Q4_K, `IqMoeFragment` for IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S
  (`vecdotq.hpp`). `qfn_mmq_moe_gated_type` is
  the single gated-format definition (adapter, exported via `qfn_mmq.h`);
  the executor's `GatedMoeType` delegates to it, so the sets cannot drift.
- Prefill (wide): MMQ — `mul_mat_q_case` instantiations plus the type-generic
  `qfn_mmq_moe_raw` / `qfn_mmq_moe_pair_unique` in `qfn_mmq.hip.cpp`. Routed
  tile width comes from `qfn_mmq_routed_tile_cols_for_counts` (kPanel 32, the
  measured optimum; under-shooting width costs more than over-shooting
  because every column tile re-streams and re-decodes one full weight row
  slice).
- The big prefill tier is the F16-tile WMMA route: `RoutedF16GEMMKernel`'s
  `kLowBit` sub-block staging dequantizes the eight low-bit expert formats
  into F16 tiles using the shared `DecodeQuantSub16` decoders
  (`src/models/qwen/hip/quant_ops.hpp`), with the paired gate/up SwiGLU fusion
  riding along. Q5_1 rows stage their 24-byte blocks as stored in LDS (scale,
  high word and codes read per K block); K-quant/IQ rows stage codes plus
  their per-block affine. The series' pair design is kept wholesale — cached
  headers, cndmask code-group selection, the F16 LDS pair epilogue — with its
  `HeaderByte` header decoder.
- Wide batches take the 64-token down map and the 128-token paired gate/up
  map (`routed_64_tiles_` / `routed_pair_*`); `WmmaExperts` (upstream's helper
  shape, its condition carrying our `RoutedF16Type` gate) selects the tier.
- Secondary-cost structure: the id-map bounds scan keeps counts read-only and
  lands block bases beside them; the MoE epilogue folds into the combine via
  the `moe_pending_` deferral seam (the adopted `kPrior`/`kInject`
  deferred-residual and fused-inject fold, settled by `HcApplyDeferred`);
  the routed SwiGLU is formed inside the down quantize (`quantize_mmq_q8_1`
  with `x_up`), skipping the F32 round trip. The combine's square-sum uses one
  pinned `__fmaf_rn` chain shared by the deferred and separate paths.
- Q2_0 is wired end to end inside the vendored layer (recorded in VENDOR.md):
  `block_q2_0` + QK2_0/QI2_0/QR2_0 in `ggml-common.h`,
  `vec_dot_q2_0_q8_1` (chunked per-lane crumb decode via
  `__builtin_amdgcn_perm`), `load_tiles_q2_0` (unpacks into the Q8_0 tile so
  the Q8_0 vec dots apply), `mmq_type_traits` and tile/DS-layout cases in
  `mmq.hpp`/`common.hpp`/`vecdotq.hpp`.
- `moe_wave64.hip.cpp` is untouched (Q4_K-only batch kernel, no type gate).

### Dense projections, heads, embeddings

- `qfn_mmq_dense(weight_type, …)` dispatches the dense/shexp set: Q8_0, Q2_0,
  Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS.
  `qfn_mmq_dense_vec(M, N, K)` (argument order like its siblings) routes
  through `moe_vector_projection` with `n_experts = n_expert_used = 1` and a
  pool-allocated zeroed ids buffer; its accepted set equals the gated set
  (Q5_1 rejected on both — decode and prefill cannot diverge).
- `Executor::Dense` keeps the Q8_0 fast tier and routes every other
  non-F32/BF16/F16 matrix through the adapter (MMVQ below `MatrixRows`, MMQ
  above). The generic tail rejects non-F32/BF16/F16 loudly, and the error
  path distinguishes unsupported format (probed via `qfn_mmq_moe_gated_type`)
  from launch failure. `DenseBatch` runs quantized rows through one
  `qfn_mmq_dense` call.
- Q6_K is a first-class adapter format (`WeightType::kQ6_K = 14`; MMVQ/MMQ/
  gated cases all present).
- Wide dense GEMMs (prefill): `DenseF16GEMMKernel`'s `kQType` decode feeds
  the shared Q8_0 LDS/WMMA path (`DenseQuantF16Gemm` entry; executor
  `PrepareHalfInput` staging at wide `MatrixRows`). This superseded and
  deleted the earlier `KQuantWmmaGemm`/`Q8ActFillSums` int8 route; a rebuild
  of that route over the per-lane decoder family at the 256-row stack
  geometry measured negative (the int8 fixups eat the 2x WMMA rate; see
  "Performance") and was dropped again.
- Wide prefill SSM projections: the four-tap convolution fuses into the
  quantized projection (`DenseQuantF16SsmGemm`, `kSsmConv` epilogue
  instantiations for the K-quant/IQ ssm_in types) behind the widened
  executor gate (`KQuantWmmaType(ssm_in.type) && cols <= max_half_cols`,
  alongside `DenseF16Route`); the epilogue reads only the accumulated tile,
  conv_w and history, so the weight format is transparent to it. The Q8_0
  route retains each wave's decoded weight rows in registers; the GSQ
  stacks stage their decoded fragments through LDS (the register variant is
  slower for them; see "Performance").
- `UnquantizedF16Gemm` (F16 or BF16-as-F16 weights, F16 activation rows) runs
  64-row x 128-token tiles (the W8A8Gemm narrow geometry; the K order per
  output is fixed, so results stay bit-exact across every prefill chunk
  width). It serves the HC down fusion, the BF16 indexer/alpha-beta
  projections and the calibrated F16 blaslt shapes.
- Embeddings: `EmbeddingType`/`SmallType` and the gather (`RowElement`/
  `RowBytes` via `gufo::hip::QuantBlockElement`) accept the GSQ token_embd
  types; unmapped types `__builtin_trap()` on device (host mappers throw).
  The LM head (Q5_K/Q6_K) runs through the same dense dispatch.
- Decode few rows: `mul_mat_vec_q_dense_multi` (`mmvq.hip.cpp`) shares
  weight rows across the token sweep — one warp computes all tokens of its
  two weight rows for dense projections at 2..8 tokens (per-row and per-token
  dot products identical to the one-token waves). Routed calls never take it
  (gated on `n_experts == n_expert_used == 1`).
- 16-bit few-row long-K GEMVs (the BF16 hc down [320, 10240]):
  `SmallGemmWideKernel` computes two weight rows per wave (m ≤ 512,
  k ≥ 8192, 1..8 tokens); four rows starved the 320-row shape of resident
  waves (82 → 116 GB/s cold, measured). The control's Q8_0 hc down
  keeps its specialized `qfn_q8_hc_down` path.

### Stacked attention and SSM projections

`device_model.cpp`'s `stackable` gate accepts the adapter's quantized dense
set (single-sourced with the executor's dense probe), and `Stack` raw-stacks
those formats like Q8_0 — same zeroed `kTailMargin` tail, `max_q8_cols`
unchanged for Q8_0, plus `max_half_cols` registration for the K-quant stacks
(the F16-tile route sizes activation rows from it). Same-type attn_q/k/v
triplets merge into one [13312][2560] GEMM whose output the existing stacked
path fuses (`Dense` + `PrepareAttention`); same-type ssm_qkv/ssm_gate pairs
merge over the existing `ssm_in` layout. Mixed-type groups stay separate by
design: requantizing weights is forbidden by the exact-vs-CPU-reference gate,
so the merge is per same-type group only.

### Hyper-connection (BF16 mixers)

Three fused paths accept BF16 where Q4_K_XL used Q8_0:

- Inject fusion: the mix epilogues (`HcMixEpilogue`/`Vec4`/`Vec4F16` and
  `HcMixF16Gemm`'s inject launch) take `WeightType inject_type`
  (kF32/kBF16) and read BF16 through the exact `Bf16ToF32` expansion
  (`InjectElement`/`InjectLoad4`); `fused_inject` accepts kBF16 in `HcMix`
  and `HcMixBatch`.
- Wide route + gate-mix: `wide_mixer_` accepts a BF16 down (F16 norm
  staging; the tiled-Q8 norm remains for W8A8); `fused_projection` accepts
  BF16 down/up; the down reads the F16 norm through `UnquantizedF16Gemm`
  with `kStageBf16` on `DenseF16GEMMKernel` (BF16 rows rounded to F16 at the
  LDS commit — exact for every BF16 value in F16's range, same two-chain K
  order as the F16 path); `HcMixF16Gemm` runs the up as
  `kHalfWeights + kStageBf16`, the BF16 up on the 128-row `kHcMix` tile
  (32 hidden channels per tile across the four streams, the same two K16
  chains and 128-token width as the Q8_0 256-row specialization); the
  SiLU'd rank rows reach the gate buffer as
  F16 via SiluScale + NarrowActivations. F32 downs re-norm into the F32
  scratch on the wide route.
- Two pinned hazards: the mixed row's tiled-Q8 copy is gated on
  `hidden <= max_q8_cols` (a GSQ upload without the Q8_0 MTP sidecar has
  `max_q8_cols == 0` — the probe path; chat auto-attaches the sidecar), and
  the epilogue's mix/inject FMAs are pinned with `__fmaf_rn` (the build is
  `-ffast-math`; reassociation breaks the byte-exactness contracts).
- Real hc BF16 weights were scanned: |w| ≤ 8.56 (no F16 overflow);
  823/640M values below 6.1e-8 flush in the F16 staging (contribution
  < 5e-5) — accepted.

### MTP catch-up (Q8_0 catch-up branch)

The shared-Q8_0 MTP sidecar makes every mix take `Moe`'s Q8_0 catch-up
branch. Pinned behaviors: the branch memsets both routed id maps to -1 first
(stale prompt-batch histograms otherwise scatter garbage rows into live
output slots); the one-token down-projection GEMM's output lands on the
caller's F16-stride view (`RowScratch` advances by whole floats — the
adapter's F32 layout — which is twice the F16 row stride, and it cannot know
the route); `moe_pending_` and the deferred-epilogue flags are restored
across the `UseScratch` restore.

### Multi-session batched decode (> 8 rows) — by design, do not "fix"

Non-Q8_0 dense is chunked on the serial MMVQ tier at 8 rows
(`MMVQ_MAX_BATCH_SIZE = kDecodeRows = 8`), re-sweeping each weight matrix
ceil(rows/8) times. Verification must match single-session decode per row,
and the wide MMQ tier only approximates it (|MMVQ − MMQ| < 2e-2 · scale,
pinned). Single-user AR/MTP stays at ≤ 8 rows. If batch throughput is ever
pursued, the feasible shape is one serial-tier launch for all rows
(`qfn_mmq_dense_vec` tiles arbitrary token counts via grid.z; per-row
equality across token counts is pinned) — not a new pinned-width MMQ path.

## Exactness and determinism contracts

What is pinned, where, and why. Breaking any of these silently breaks
session/audit replay.

- Operator tests are exact vs CPU-dequant F64 references per type × role
  (dense, routed, paired, gated, head, embedding): the 19
  `qwen38_flash_next.*` ctest suites — 12 HIP operator suites (`mtp_ops`,
  `blaslt_ops`, `hc_mix_ops`, `gdn_ops`, `attention_ops`, `moe_epilogue_ops`,
  `moe_ids_ops`, `ple_ops`, `projection_ops`, `routed_wmma_ops`, `select_ops`,
  `snapshot_copy`) plus the CPU suites (`config`, `continuation_layout`,
  `draft_vocab`, `ngram`, `mtp_sampling`, `prompt_lookup`, `weights`).
- FMA/accumulation pinning: every shared MMQ vec-dot accumulation is pinned
  with `__fadd_rn`/`__fmaf_rn` (`mmq.hpp` typed cases and `vecdotq.hpp`);
  chunk-width exactness and the dense MMQ tile width are pinned by tests.
  Reason: `-ffast-math` reassociation (1–8 ulps) breaks downstream
  byte-exactness assumptions (HC epilogues, batch invariance).
- Invariance pins: routed batch-width exactness (224-row vs 96-row tail),
  MMQ tile-width invariance (`CheckMoeWidthExact`), dense-vec rows
  bit-identical across 1..8 tokens (`CheckDenseVecTokens`), stacked ==
  separate per-row bit-exact for every consuming tier
  (`CheckDenseStackExact`), pair == separate per type, decode 16-bit GEMVs
  byte-exact vs the CPU-expanded reference (`CheckDecodeGemv`).
- Quantized fused SSM convolution: `CheckSsmProjectionQuant` pins the fused
  `DenseQuantF16SsmGemm` pass bit-exact against the separate
  `DenseQuantF16Gemm` projection per stacked K-quant/IQ format (same decode,
  same K order per output) through its GatedDeltaNet consumer, at a full and
  a partial final token tile, plus rejection fixtures.
- The widened `UnquantizedF16Gemm` route keeps one K order per output:
  `CheckBf16Gemm` pins the BF16 route bit-exact against the F16-staged route
  across a prefill chunk split, and `blaslt_ops_test` pins the F16 route
  across chunk boundaries.
- HC BF16 byte-exactness: `CheckBf16Down`, `CheckFusedProjectionBf16`,
  `CheckBf16Inject` against CPU-expanded F16/F32 references; double CPU mix
  reference within 5e-5/1e-4.
- MoE: the fused epilogue residual is byte-exact against
  `MoeEpilogueVec4 + HcCombine`, and the combine's square-sum and gamma-scale
  multiplies use one pinned `__fmaf_rn`/`__fmul_rn` spelling for the deferred
  (`kPrior`) and separate paths alike (a split spelling diverged at Q8
  rounding boundaries in this compilation context); the routed
  SwiGLU-in-down-quantize codes are bit-identical to
  SwigluKernel + the plain entry for all thirteen routed formats
  (`CheckSiluDownMmq`). The grouped gated vector's IQ2/IQ3 fragment reuse
  keeps each format's integer scale formula, dp4a order and float scale
  product — the product pinned with an asm barrier (`IqMoeScale`) to the
  scalar routed path's association — so its output is byte-exact against the
  separate projections plus SwiGLU at every width, including duplicate and
  invalid slots and nonfinite scales (`routed_wmma_ops`).
- Type-id bridges: every `WeightType` value is pinned against both
  `core::GgmlType` and the adapter's `ggml_type` in enum order (raw-cast
  bridge in `routed_wmma_ops_test`); GGUF type ids are pinned in
  `ggml_dequant_test`. A moved id fails to compile, not silently misroute.
- Whole-block contracts: MMQ entry points require whole weight blocks
  (`K % block_elements == 0` — Q2_0's 64-block with K ≡ 32 mod 64 must fail,
  not truncate); Q2_0 tails k=64/320 and dense tight-K cases carry zeroed
  weight tails; negative tests pin Q5_1/Q4_0 rejection at the executor probe
  and both dense entry points.
- MTP oracle: `--mtp-audit` pins the oracle's down-projection inputs to the
  GPU's SwiGLU outputs (`stage_inputs`) with loose stage compares (2e-3) and
  a tight `ffn-experts` compare (5e-5) — ulp-level GPU-vs-F64 noise otherwise
  flips one Q8_1 code and amplifies to 5e-4 divergence (a probe-only class;
  production is GPU-only and deterministic).
- Draft-vocabulary checks compare the small-head greedy continuation and
  final logits byte-exactly with full-head AR. The MTP audit compares real
  table-row scores and remapped rare-token IDs with an independent full-head
  projection, including empty-slot masking, table replacement, scalar heads,
  all-small-head batches and mixed full/small-head batches. Its one-row prefix
  guarantees rare-token candidate coverage independently of generated language
  or neural-draft acceptance rate.
- Determinism profile: greedy traces are deterministic per binary; the bench
  d32768 128-token completion is bit-identical (sha `d856f78e…`) across all
  three models, AR and MTP. d0 traces diverge per model/binary in the
  documented borderline-argmax class (not a correctness or speed item). MTP
  cycle counts are deterministic per binary.
- Quality oracles: `gpu_probe --reference` argmax matches the F32 oracle on
  both mixes; KL 4e-4..2e-2 per mix with a couple of deterministic borderline
  argmax flips — the documented class.

## Verification

From a clone with the GSQ-RCO (and, for the control comparisons, the
unsloth UD-Q4_K_XL) GGUFs:

```sh
# Tests (RelWithDebInfo + assertions)
cmake --preset gpu-test && cmake --build --preset gpu-test
ctest --preset gpu-full -R '^qwen38_flash_next\.' --output-on-failure   # 19 suites
ctest --preset gpu-full -R 'ggml_dequant_test|ggml_gemm_parity_test|qwen_gemm_route_test'

# Measurement binary
cmake --preset release && cmake --build --preset release   # -> build/release/gufo
```

- Session replay: the model-gated `qwen38_flash_next_session_test` binary —
  `session_test --model FIRST.gguf --mtp-model MTP.gguf` on xxs/s/q4
  (covers prefill chunking, image prefill, graph capture/replay, batched
  sessions, serving modes; focused flags exist for each area).
- MTP audit: `gpu_probe --model FIRST_SHARD.gguf --prompt TEXT --mtp-model
  MTP.gguf --mtp-audit` on xxs/s/q4.
- Reference probe: `gpu_probe --model … --prompt … --reference`.
- Functional API baselines: `tests/functional/run.py --record-baseline`
  then compare; suites `discovery responses conversation batch
  sampling-defaults state-edges`, modes `off` and `mtp` (`off` = zero
  drafts, `mtp` = real draft execution).

## Performance state

Measured on the release binary, AMD Strix Halo (`gfx1151`).
Measurement shape: `gufo bench`, greedy, `-p 2048 -n 128 -d 0,32768 -r 1 -v`,
single-user, shared-Q8_0 MTP sidecar attached in MTP mode for the MTP cells
and detached (`--speculative off`) for the AR cells, one binary for
all three models; every cell is the delta against the unsloth UD-Q4_K_XL
control measured in interleaved rounds on that binary (two rounds, r1/r2).

Δ vs UD-Q4_K_XL control (canonical alternating rounds, r1/r2 unless marked
steady):

| cell | xxs | s |
| --- | ---: | ---: |
| AR tg128 (steady) | ≈ -1% | ≈ -6% |
| AR tg128 @ d32768 | -2.0/-2.0% | -6.5/-6.6% |
| MTP tg128 (draft-trajectory) | -11.8/-10.3% | -19.1/-18.8% |
| MTP tg128 @ d32768 | -2.9/-3.3% | -6.2/-5.7% |
| pp2048 @ d32768 (off) | -20.8/-19.7% | -21.1/-19.5% |
| pp2048 @ d32768 (mtp) | -19.2/-19.0% | -19.0/-18.7% |
| pp2048 d0 (steady) | ≈ -23% | ≈ -22..-23% |

- Decode reading: AR tg is at parity to about -7%; steady depth-0 cells
  measure about -1% (xxs) and -6% (s), and the depth-32768 cells read
  -2.0/-2.0% (xxs) and -6.5/-6.6% (s). Depth-0 AR cells carry launch
  run-noise: individual multi-depth-round cells read as low as 22.7 t/s
  (xxs) and 21.8 t/s (s) while focused alternating reruns read 26.7-26.9
  and 25.4-25.5 t/s against the control's ~27. MTP tg depth-0 cells
  sit in the draft-trajectory class: speed tracks draft accept-rate, not
  kernel speed (the control itself reads 37-43 t/s across rounds with
  identical cycles and accepted counts; each model's counters are
  deterministic per binary). At depth 32768 upstream's prompt-lookup draft
  copies bypass the neural draft head for greedy runs (identical 17 cycles /
  111 accepted across all three models), so that cell times the multi-row
  verification forward: the -3..-3.5% (xxs) / -6..-6.5% (s) residual is the
  low-bit formats' per-element decode at 8 verification rows with the
  shared-expert weight-fragment reuse in place (the routed gate/up stage is
  434 ms per 17 copied-32K verification cycles on xxs, at that stage's
  decode-bound ceiling), not a draft, attention or dispatch deficit.
- Decode ceilings (standalone per-format saturation cells measured with
  canonical block accounting, real GGUF weight blocks and per-cell CPU-dot
  validation): the practical DRAM read peak is 216.2 GB/s;
  the Q6_K head MMVQ runs at 98-100% of it — no headroom (the control's
  Q8_0 head moves more bytes per token at the same peak). The low-bit
  expert MMVQs run at 85-95% of their own formats' standalone saturation
  ceilings (512 used experts, pair gate/up: Q2_0 214, IQ4_NL 211, Q4_K
  211, Q5_K 211, Q6_K 213, IQ4_XS 208, IQ3_S 153, IQ3_XXS 138, IQ2_S 118,
  IQ2_XS 110, IQ2_XXS 102 GB/s; down: IQ4_NL 212, Q2_0 185). Q2_0 and
  IQ4_NL reach DRAM class; the remaining gap for the IQ2/IQ3 family is
  the per-lane grid-LUT decode of the IQ vec-dot family, and it exceeds
  what the known techniques (wider loads, wave scheduling) could recover
  — the LUT decode cost is that family's ceiling. The one genuinely slow
  decode shape is the BF16 hc-down 16-bit GEMV (629 MB at 84 GB/s vs the
  control's Q8_0 320 MB at 182); its two-rows-per-wave form runs at
  116 GB/s cold, and the faster double-tile variant does not pay for its
  code.
- Prefill: the residual about -19..-24% is measured compute-bound, not byte-
  or launch-bound. The dequant-to-F16 route runs at 13-14 TFLOPS ≈ the
  F16-WMMA ceiling at these shapes (the wide HC down moves ~25% fewer bytes
  for ~3% less time); the pp window idles 5-8% on both mixes (not
  launch-bound); +21% dispatches vs the control is the mixed-type-stack
  class. The unquantized narrow plan stays 64x128; upstream's <128,128>
  wide plan is not used for that route (the 64x128 plan serves the same
  shapes, and the control pp showed no regression). The warmed xxs pp2048
  forward carries HC-up at 71.6 ms (control 51.4; the BF16 128-row tile
  keeps both K16 chains spill-free), the half-weights HC-down family at
  +161 ms (the BF16 down projection alone is about 158 ms, against the
  control's 48 ms int8 WMMA specialization), the mixed-type dense stacks at
  +178 ms and routed gate/up at +62 ms. Depth-32768 pp cells are tightly
  -19..-21%; depth-0 cells read about -22..-23.5% in steady alternating
  cells and read wider when the control's d0 rate runs low (the control's
  pp rate varies more between rounds than the mixes'), so that spread is
  measurement variance, not a path difference.
- Measured-negative routes (not built, and why): the HC up conversion is
  not conversion-bound (v_cvt 2.8% of the kernel body; an F16-weights
  variant of the same shape runs 3.2x faster on 1.9x bytes — the
  residual is stage structure at k=320 plus format bytes), so no
  bit-field conversion path exists. The dense compact-Q8 int8 WMMA route
  (per-lane decoder family, 256-row stack geometry) loses its first
  shape-class A/B (IQ4_XS stacks 26.0 TF vs the F16 route's 30.0, Q6_K
  20.1 vs 28.3, Q4_K 21.4 vs 28.3; an earlier BM=128 route measured 856
  vs 1108 t/s) — the per-accumulator-element fixups eat the 2x int8
  WMMA rate; the routed compact-Q8 tier was gated on it and not built.
  The BF16 hyper-connection down projection [320, 10240] keeps the
  64 x 128 tile with 64-element K stages: the 32-token/256-element-stage
  and 128-token/128-element-stage plans measure about 180 ms and 160 ms
  against 158 ms in the warmed pp2048 forward; the smaller token tile
  multiplies BF16-to-F16 weight staging per token, and at 128 tokens the
  halved barrier count does not pay for the larger stage.
  The SSM register-owned weight path is Q8_0-only: the register variant
  measures 53.9-54.3 ms (xxs) and 70.2 ms (s) on the warmed pp2048 forward
  against 48.7-49.2 and 66.2 ms with LDS staging. The half-wave exchange
  costs sixteen lane permutes per K32 block where the staged path costs
  twelve LDS instructions, and the low-bit decode already covers the LDS
  round trip.
- Control health: UD-Q4_K_XL is regression-free vs `main`.

## Vendored-code obligations

`kernels/rocm/mmq/VENDOR.md` records every keep-on-re-import obligation:

- Q2_0 additions to `ggml-common.h`, `mmq.hpp`, `vecdotq.hpp`, `common.hpp`.
- The `__fadd_rn`/`__fmaf_rn` pinning of every shared MMQ vec-dot
  accumulation (chunk-width exactness is the reason).
- `mmq_args::mmq_x_request` with the host `dense_tile_cols`/requested-width
  selection (the routed tile-width path is the consumer).
- `quantize_mmq_q8_1`'s `x_up` (the routed SwiGLU formed in the down
  quantize).

Policy: keep vendored `mmq.hpp`/`ggml-common.h` edits to zero or near-zero;
re-imports must re-apply the recorded blocks.

## Known quirks and accepted debt

- `weights.cpp` `FormatOf` block geometry duplicates
  `quant::QuantizedBlockElements`/`QuantizedRowBytes` (pre-existing pattern,
  values verified agreeing).
- Q2_0 is a non-upstream type id (42); keep it isolated in the quant layer.
- Multi-session batched decode re-sweeps non-Q8_0 dense per 8 rows (by
  design; see above).
- `GdnRowSplitKernel` 112 B scratch is main-side drift (from #445/#485),
  left for a main-side follow-up.
- `DenseBatch`/`GatedDenseBatch` executor-level coverage is via session
  replay, not direct operator tests (private `Executor` members behind a
  full `DeviceModel`; a synthetic seam would be new scaffolding the fork's
  minimal-diff bar rules out).
- MTP d0 greedy traces are per-binary deterministic but not cross-binary
  stable (borderline argmax class; documented, accepted).
- The s-mix full gate suites can flip a seeded-sampling borderline case
  across process runs (a session-replay cache check and one `--mtp-audit`
  candidate check have each flipped once; both pass on rerun; logits are
  pinned bit-identical across the affected paths, so the class is
  sampling-boundary noise, not numerics).

## File map

| Area | Files |
| --- | --- |
| Type layer | `src/core/gguf_reader.hpp`, `src/core/quant/ggml_dequant.{hpp,cpp}` |
| Loader | `src/models/qwen38_flash_next/weights.cpp` (`FormatOf`, role whitelists), `device_model.cpp` (upload gates, `Stack`, `kTailMargin`) |
| Adapter (gufo-owned) | `kernels/rocm/mmq/qfn_mmq.h`, `qfn_mmq.hip.cpp`, `mmvq.hip.cpp`, `quantize.hip.cpp` |
| Vendored MMQ | `kernels/rocm/mmq/{mmq.hpp, mmvq.hpp, vecdotq.hpp, common.hpp, ggml-common.h, …}`, `VENDOR.md` |
| Shared decoders | `src/models/qwen/hip/quant_ops.hpp` (`DecodeQuantSub16`) |
| Executor / dispatch | `src/models/qwen38_flash_next/executor.cpp` |
| Other kernels touched | `kernels.hip.cpp` (`SmallGemmWideKernel`), `moe_wave64.hip.cpp` (untouched), small-batch/prefill waves under `src/models/qwen/hip/kernels/` |
| Operator tests | `tests/models/qwen38_flash_next/*_test.cpp`, `mtp_audit.cpp`, `gpu_probe.cpp`, `reference_probe.cpp` |
| Core quant tests | `tests/core/ggml_dequant_test.cpp` |
| Bench/profile tools | `tools/bench/`, `tools/prof/prof.py`, `tools/prof/isa_mix.py` |
