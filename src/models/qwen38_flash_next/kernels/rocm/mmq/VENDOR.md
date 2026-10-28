# Quantized HIP kernels

Adapted from `ggml-org/llama.cpp` at
`5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a` (MIT; see `LICENSE`).
The upstream source directory is `ggml/src/ggml-cuda`; Gufo builds these
sources with HIP for AMD Strix Halo (`gfx1151`).

`qfn_mmq.h` is the public adapter. It supplies Q8_0 dense projections and
Q4_K, Q5_K, Q5_1 and Q8_0 routed expert projections, including the paired
Q4_K gate/up path. `qfn_ggml_stubs.*` supplies the minimal context and scratch
pool without the ggml graph runtime. Gufo owns scheduling and HIP graphs.

Local additions beyond the import (GSQ-RCO support) live beside their
upstream neighbors: `block_q2_0` plus `QK2_0`/`QI2_0`/`QR2_0` in
`ggml-common.h`, `load_tiles_q2_0` and the `mmq_type_traits` case in
`mmq.hpp`, `vec_dot_q2_0_q8_1` with `VDR_Q2_0_Q8_1_MMVQ`/`VDR_Q2_0_Q8_1_MMQ`
in `vecdotq.hpp`, and the `ggml_hip_type_traits` case in `common.hpp`. Q2_0
(GGUF type id 42) is not an upstream llama.cpp id; `QI2_0` counts vec-dot
chunks, not packed int32s. The shared MMQ vec-dot accumulation is pinned
with `__fadd_rn`/`__fmaf_rn` (every typed vec-dot case in `mmq.hpp` from
~line 520, plus the `vecdotq.hpp` impls) so `-ffast-math` cannot reassociate
it; chunk-width exactness depends on this (`test(qwen38-flash-next): cover
chunk-width exactness` pins it). `mmq_args::mmq_x_request` and the host
requested-width selection (`mmq.hpp:4085`, `4324-4329`) let the caller pick
the column-tile width; `qfn_mmq_set_routed_tile_cols` feeds it for the routed
tile-width path. `quantize_mmq_q8_1` takes an optional
`x_up` and forms `silu(x) * x_up` in the gather-quantize (`silu_mul`, matching
gufo's `SiluF`/`SwigluKernel` exactly); keep that when re-importing, and keep
the arithmetic aligned with `kernels.hip.cpp`. Keep these blocks when
re-importing.

The kernels use native HIP runtime, hipBLAS and HIP numeric types. The
upstream runtime aliases, NVIDIA capability dispatch and PTX branches have
been removed; retained matrix fragments use AMD WMMA/MFMA. Headers use the
`.hpp` suffix. Gufo owns scheduling and HIP graphs.

When importing an upstream fix, preserve the Gufo namespace, adapter and
HIP-only dispatch, update the pin, and run the Flash-Next operator tests
under `tests/models/qwen38_flash_next/` plus the AR/MTP session replay.
The routed GEMM test compares supported formats against a floating-point
reference and checks the paired adapter against two separate projections.
Build and validate through Nix as described in `AGENTS.md`.
