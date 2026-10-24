// SPDX-License-Identifier: MIT
// HIP adapters for the llama.cpp-derived quantized kernels. See VENDOR.md.
#include "qfn_mmq_prelude.h"
namespace qfn_mmq {
#include "qfn_mmq.h"
#include "common.hpp"
#include "mmq.hpp"
#include "mmvq.hpp"
#include "quantize.hpp"
#include "mmid.hpp"
static int g_routed_max_expert_rows = 0;
static int g_routed_tile_cols = 0;
static int prepare_decode_kernels();
extern "C" int qfn_mmq_init(int device) {
    if (device < 0) {
        fprintf(stderr, "qfn_mmq_init: invalid device %d\n", device);
        return -1;
    }
    ggml_hip_set_device(device);

    const auto & info = ggml_hip_info();
    if (info.device_count == 0) {
        fprintf(stderr, "qfn_mmq_init: no HIP devices found\n");
        return -1;
    }
    if (device >= info.device_count) {
        fprintf(stderr, "qfn_mmq_init: device %d out of range (have %d)\n",
                device, info.device_count);
        return -1;
    }

    return prepare_decode_kernels();
}

extern "C" void qfn_mmq_set_routed_max_expert_rows(int rows) {
    g_routed_max_expert_rows = rows > 0 ? rows : 0;
}

extern "C" void qfn_mmq_set_routed_tile_cols(int cols) {
    g_routed_tile_cols = cols > 0 ? cols : 0;
}

extern "C" int qfn_mmq_routed_tile_cols_for_counts(
        const unsigned int *counts, int n_experts) {

    if (!counts || n_experts <= 0) return 0;
    /* Per tile the kernel re-streams and re-decodes one weight row slice for
     * every column tile, while the pad columns a wider tile leaves are cheap
     * fragment work over data already in flight. kPanel is that fixed tile
     * cost in units of one pad column: forced-width sweeps of the routed
     * prefill GEMMs put the per-launch optimum at 48 columns at 2048-4096
     * token chunks, and under-shooting the width costs more than over-shooting
     * (0.9 vs 0.25 ms per launch), so 32 reproduces the measured optima. */
    constexpr int kPanel = 32;
    int best_cols = 0;
    long long best_cost = 0;
    for (int w = 16; w <= 80; w += 16) {
        long long cost = 0;
        for (int e = 0; e < n_experts; e++) {
            const unsigned int c = counts[e];
            if (c == 0u) continue;
            cost += (long long)((c + (unsigned int)w - 1u) / (unsigned int)w) *
                    (long long)(w + kPanel);
        }
        if (best_cols == 0 || cost < best_cost) {
            best_cols = w;
            best_cost = cost;
        }
    }
    return best_cols;
}

__global__ static void qfn_mmq_sanitize_f32_kernel(float *p, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = p[i];
    if (!isfinite(v)) p[i] = 0.0f;
}

static void qfn_mmq_sanitize_f32(float *p, uint64_t n, hipStream_t stream) {
    if (!p || n == 0) return;
    qfn_mmq_sanitize_f32_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, stream>>>(p, n);
}

ggml_backend_hip_context * get_ctx_for_device(int device) {
    static std::unique_ptr<ggml_backend_hip_context> cached[GGML_HIP_MAX_DEVICES];
    if (device < 0 || device >= GGML_HIP_MAX_DEVICES) return nullptr;
    if (!cached[device]) {
        cached[device] = std::make_unique<ggml_backend_hip_context>(device);
    }
    return cached[device].get();
}

// Prompt arithmetic must stay independent of the chunk width: the tile-width
// rule in mul_mat_q_case fits ncols_max, so a bulk prefill and its chunks pick
// different template instantiations, which accumulate in different orders.
// Dense callers request the widest usable tile instead, like the routed pin.
template <ggml_type type>
static int dense_tile_cols(const int cc, const size_t smpbo,
                           const int warp_size, const int nwarps,
                           const int mmq_y) {
    const int mmq_x_max = get_mmq_x_max_host(cc);
    const int ncw = (amd_wmma_available(cc) && !amd_mfma_available(cc))
            ? QFN_WMMA_MMQ_NCW : 1;
    int best = 0;
    for (int mmq_x = 8; mmq_x <= mmq_x_max; mmq_x += 8) {
        const int granularity = mmq_get_granularity_host(mmq_x, cc);
        if (mmq_x % (granularity * ncw) != 0) continue;
        if (mmq_get_nbytes_shared<type>(mmq_x, mmq_y, cc, warp_size, nwarps)
                > smpbo) continue;
        best = mmq_x;
    }
    return best;
}

template <ggml_type type>
int qfn_mmq_dense_impl(
        const char  * tag,
        const void  * W,
        const float * X_f32,
        float       * out_f32,
        int           M,
        int           N,
        int           K,
        hipStream_t  stream) {

    if (!W || !X_f32 || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (K <= 0 || M <= 0 || N <= 0) {
        fprintf(stderr, "%s: bad shape M=%d N=%d K=%d\n", tag, M, N, K);
        return -1;
    }
    if (K % ggml_blck_size(type) != 0) {

        fprintf(stderr, "%s: K=%d is not a whole weight block\n", tag, K);
        return -1;
    }

    const int dev = ggml_hip_get_device();
    const int cc  = ggml_hip_info().devices[dev].cc;
    const size_t smpbo = ggml_hip_info().devices[dev].smpbo;
    const int warp_size = ggml_hip_info().devices[dev].warp_size;

    ggml_backend_hip_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get HIP context for device %d\n", tag, dev);
        return -1;
    }

    const int64_t ne00         = K;
    const int64_t ne10_padded  = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const int64_t ne11         = N;
    const int64_t ne12         = 1;
    const int64_t ne13         = 1;

    const size_t y_block_size = sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = 4 * QK8_1;
    const size_t nbytes_src1_q8_1 =
        ne13 * ne12 * ne11 * ne10_padded * y_block_size /
            y_values_per_block +
        get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);

    ggml_hip_pool_alloc<char> src1_q8_1_pool(ctx->pool(), nbytes_src1_q8_1);
    char *src1_q8_1 = src1_q8_1_pool.get();

        quantize_mmq_q8_1_hip(
            X_f32, nullptr, nullptr, (void *)src1_q8_1,
            type, K, (int64_t)K, 0, 0,
            ne10_padded, ne11, ne12, ne13,
            stream);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "%s: quantize failed: %s\n", tag, hipGetErrorString(err));
        return -2;
    }

    const int64_t blck   = ggml_blck_size(type);
    const int64_t s01    = (int64_t)K / blck;
    const int64_t s1     = (int64_t)M;
    const int64_t s12    = ne11 * ne10_padded * y_block_size /
                           (y_values_per_block * sizeof(int));
    const int64_t s13    = ne12 * s12;

    const bool use_stream_k = false;

    mmq_args args = {
        (const char *)W,
        type,
        (const int *)src1_q8_1,
        nullptr,
        nullptr,
        out_f32,
        ne00,    (int64_t)M,    ne11,
        s01,ne11,          s1,
        1,   1,
        0, s12, 0,
        1,    1,
        0, s13, 0,
        use_stream_k,
        ne11,
    };
    const bool inline_sanitize =
        type == GGML_TYPE_Q8_0 && !use_stream_k;
    args.sanitize_output = inline_sanitize;

    args.mmq_x_request = dense_tile_cols<type>(
        cc, smpbo, warp_size, mmq_get_nwarps_host(cc, warp_size),
        get_mmq_y_host(cc));
    if (args.mmq_x_request <= 0) {
        fprintf(stderr, "%s: no usable tile width for type %d\n", tag,
                (int)type);
        return -1;
    }

    mul_mat_q_case<type>(*ctx, args, stream);

    err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "%s: mul_mat_q_case launch failed: %s\n", tag, hipGetErrorString(err));
        return -3;
    }
    if (!inline_sanitize) {
        qfn_mmq_sanitize_f32(out_f32, (uint64_t)M * (uint64_t)N, stream);
    }
    return 0;
}

extern "C" int qfn_mmq_dense(
        int weight_type, const void * W, const float * X, float * out,
        int M, int N, int K, hipStream_t stream) {
    switch (static_cast<ggml_type>(weight_type)) {
        case GGML_TYPE_Q8_0:
            return qfn_mmq_dense_impl<GGML_TYPE_Q8_0>(
                "qfn_mmq_dense", W, X, out, M, N, K, stream);
        case GGML_TYPE_Q2_0:
            return qfn_mmq_dense_impl<GGML_TYPE_Q2_0>(
                "qfn_mmq_dense", W, X, out, M, N, K, stream);
        case GGML_TYPE_Q4_K:
            return qfn_mmq_dense_impl<GGML_TYPE_Q4_K>(
                "qfn_mmq_dense", W, X, out, M, N, K, stream);
        case GGML_TYPE_Q5_K:
            return qfn_mmq_dense_impl<GGML_TYPE_Q5_K>(
                "qfn_mmq_dense", W, X, out, M, N, K, stream);
        case GGML_TYPE_Q6_K:
            return qfn_mmq_dense_impl<GGML_TYPE_Q6_K>(
                "qfn_mmq_dense", W, X, out, M, N, K, stream);
        case GGML_TYPE_IQ2_XXS:
            return qfn_mmq_dense_impl<GGML_TYPE_IQ2_XXS>(
                "qfn_mmq_dense", W, X, out, M, N, K, stream);
        case GGML_TYPE_IQ2_XS:
            return qfn_mmq_dense_impl<GGML_TYPE_IQ2_XS>(
                "qfn_mmq_dense", W, X, out, M, N, K, stream);
        case GGML_TYPE_IQ2_S:
            return qfn_mmq_dense_impl<GGML_TYPE_IQ2_S>(
                "qfn_mmq_dense", W, X, out, M, N, K, stream);
        case GGML_TYPE_IQ3_XXS:
            return qfn_mmq_dense_impl<GGML_TYPE_IQ3_XXS>(
                "qfn_mmq_dense", W, X, out, M, N, K, stream);
        case GGML_TYPE_IQ3_S:
            return qfn_mmq_dense_impl<GGML_TYPE_IQ3_S>(
                "qfn_mmq_dense", W, X, out, M, N, K, stream);
        case GGML_TYPE_IQ4_NL:
            return qfn_mmq_dense_impl<GGML_TYPE_IQ4_NL>(
                "qfn_mmq_dense", W, X, out, M, N, K, stream);
        case GGML_TYPE_IQ4_XS:
            return qfn_mmq_dense_impl<GGML_TYPE_IQ4_XS>(
                "qfn_mmq_dense", W, X, out, M, N, K, stream);
        default:
            fprintf(stderr, "qfn_mmq_dense: unsupported weight type %d\n",
                    weight_type);
            return -1;
    }
}

template <ggml_type type>
int qfn_mmq_moe_impl(
        const char    * tag,
        const void    * W,
        const float   * X_f32,
        const int32_t * ids,
        float         * out_f32,
        int             M,
        int             K,
        int             n_tokens,
        int             n_experts,
        int             n_expert_used,
        hipStream_t    stream,
        const void * W_b = nullptr, float *out_b = nullptr,
        const float * X_up = nullptr) {

    if (!W || !X_f32 || !ids || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || K <= 0 || n_tokens <= 0 || n_experts <= 0 || n_expert_used <= 0) {
        fprintf(stderr, "%s: bad shape M=%d K=%d ntok=%d nexp=%d nused=%d\n",
                tag, M, K, n_tokens, n_experts, n_expert_used);
        return -1;
    }
    if (K % ggml_blck_size(type) != 0) {
        fprintf(stderr, "%s: K=%d is not a whole weight block\n", tag, K);
        return -1;
    }
    if (n_expert_used > n_experts) {
        fprintf(stderr, "%s: n_expert_used=%d > n_experts=%d\n", tag, n_expert_used, n_experts);
        return -1;
    }

    const int dev = ggml_hip_get_device();
    const int cc  = ggml_hip_info().devices[dev].cc;

    ggml_backend_hip_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get HIP context for device %d\n", tag, dev);
        return -1;
    }

    const int64_t ne_get_rows  = (int64_t)n_tokens * n_expert_used;
    const int64_t ne00         = K;
    const int64_t ne10_padded  = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const int64_t ne11         = 1;
    const int64_t ne12         = n_tokens;
    const int64_t blck         = ggml_blck_size(type);
    const int64_t s01          = (int64_t)K / blck;
    const int64_t s02          = (int64_t)M * s01;

    ggml_hip_pool_alloc<int32_t> ids_src1(ctx->pool(), ne_get_rows);
    ggml_hip_pool_alloc<int32_t> ids_dst(ctx->pool(), ne_get_rows);
    ggml_hip_pool_alloc<int32_t> expert_bounds(ctx->pool(), n_experts + 1);

    hipMemsetAsync(ids_src1.get(), 0, ne_get_rows * sizeof(int32_t), stream);
    hipMemsetAsync(ids_dst.get(),  0, ne_get_rows * sizeof(int32_t), stream);

    const int si1  = n_expert_used;
    const int sis1 = 1;

    ggml_hip_launch_mm_ids_helper(
        ids, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
        n_experts, n_tokens, n_expert_used, (int)ne11, si1, sis1, stream);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "%s: mm_ids_helper failed: %s\n", tag, hipGetErrorString(err));
        return -2;
    }

    const size_t y_block_size = sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = 4 * QK8_1;
    const size_t nbytes_src1_q8_1 =
        ne_get_rows * ne10_padded * y_block_size / y_values_per_block +
        get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
    ggml_hip_pool_alloc<char> src1_q8_1(ctx->pool(), nbytes_src1_q8_1);

    const int64_t s11_src = (int64_t)K;
    const int64_t s12_src = (int64_t)K * ne11;
    const int64_t s13_src = (int64_t)K * ne11 * ne12;

        quantize_mmq_q8_1_hip(
            X_f32, X_up, ids_src1.get(), (void *)src1_q8_1.get(),
            type, K, s11_src, s12_src, s13_src,
            ne10_padded, ne_get_rows, 1, 1,
            stream);

    err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "%s: MMQ activation quantize failed: %s\n",
                tag, hipGetErrorString(err));
        return -3;
    }

    const int64_t s1            = (int64_t)M;

    const int64_t s12_mmq = ne11 * ne10_padded * y_block_size /
                            (y_values_per_block * sizeof(int));
    const int64_t s13_mmq = ne12 * s12_mmq;

    const bool use_stream_k = false;

    mmq_args args = {
        (const char *)W,
        type,
        (const int *)src1_q8_1.get(),
        ids_dst.get(),
        expert_bounds.get(),
        out_f32,
        ne00,
        (int64_t)M,
        ne_get_rows,
        s01,
        ne_get_rows,
        s1,
        (int64_t)n_experts,
        (int64_t)n_experts,
        s02,
        s12_mmq,
        (int64_t)0,
        1,
        1,
        0,
        s13_mmq,
        0,
        use_stream_k,
        W_b ? n_tokens : ne_get_rows,
        nullptr,
        0,
    };

    args.ncols_grid_max =
        g_routed_max_expert_rows > 0 &&
        (int64_t)g_routed_max_expert_rows < ne_get_rows
            ? (int64_t)g_routed_max_expert_rows
            : 0;
    args.mmq_x_request = g_routed_tile_cols;

    mul_mat_q_case<type>(*ctx, args, stream);
    if (W_b) {
        args.x = static_cast<const char *>(W_b);
        args.dst = out_b;
        mul_mat_q_case<type>(*ctx, args, stream);
    }

    err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "%s: mul_mat_q_case (moe) launch failed: %s\n", tag, hipGetErrorString(err));
        return -4;
    }

    return 0;
}

// The fused gate/up SwiGLU vector path covers every routed expert format
// except Q5_1 (down projections only).
extern "C" int qfn_mmq_moe_gated_type(int weight_type) {
  switch (static_cast<ggml_type>(weight_type)) {
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K:
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_Q2_0:
    case GGML_TYPE_IQ2_XXS:
    case GGML_TYPE_IQ2_XS:
    case GGML_TYPE_IQ2_S:
    case GGML_TYPE_IQ3_XXS:
    case GGML_TYPE_IQ3_S:
    case GGML_TYPE_IQ4_NL:
    case GGML_TYPE_IQ4_XS:
      return 1;
    default:
      return 0;
  }
}

static int moe_vector_projection(const char* tag, int weight_type, const void* W,
                                 const float* X_f32, const int32_t* ids,
                                 float* out_f32, int M, int K, int n_tokens,
                                 int n_experts, int n_expert_used,
                                 hipStream_t stream, const void* W_b,
                                 float* out_b, bool gated,
                                 const void* X_q8 = nullptr) {
  const auto type = static_cast<ggml_type>(weight_type);
  if (!qfn_mmq_moe_gated_type(weight_type) && type != GGML_TYPE_Q5_1) {
    fprintf(stderr, "%s: unsupported weight type %d\n", tag, weight_type);
    return -1;
  }

  if (!W || !(X_f32 || X_q8) || !ids || !out_f32 ||
      (gated ? !W_b : bool(W_b) != bool(out_b))) {
    fprintf(stderr, "%s: null pointer\n", tag);
    return -1;
  }
  if (M <= 0 || K <= 0 || n_tokens <= 0 || n_experts <= 0 ||
      n_expert_used <= 0) {
    fprintf(stderr, "%s: bad shape M=%d K=%d ntok=%d nexp=%d nused=%d\n", tag,
            M, K, n_tokens, n_experts, n_expert_used);
    return -1;
  }
  if (K % ggml_blck_size(type) != 0) {
    fprintf(stderr, "%s: K=%d is not a whole weight block\n", tag, K);
    return -1;
  }
  if (n_expert_used > n_experts) {
    fprintf(stderr, "%s: n_expert_used=%d > n_experts=%d\n", tag, n_expert_used,
            n_experts);
    return -1;
  }
  const int max_gated_rows = type == GGML_TYPE_Q4_K
                                 ? MMVQ_MAX_ROUTED_BATCH
                                 : MMVQ_MAX_BATCH_SIZE;
  if (gated && (n_tokens > max_gated_rows || n_expert_used > 32 ||
                !qfn_mmq_moe_gated_type(weight_type))) {
    fprintf(stderr,
            "%s: gated vector exceeds its format's row or expert limit\n",
            tag);
    return -1;
  }

  const int dev = ggml_hip_get_device();
  ggml_backend_hip_context* ctx = get_ctx_for_device(dev);
  if (!ctx) {
    fprintf(stderr, "%s: failed to get HIP context for device %d\n", tag, dev);
    return -1;
  }

  const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
  const size_t nbytes_q8_1 =
      (size_t)n_tokens * ne10_padded * sizeof(block_q8_1) / QK8_1;
  // The down projection's slot rows are quantized in expert order, so the
  // slots that share an expert read its weights in the same block.
  const bool by_expert =
      !gated && !W_b && !X_q8 &&
      (type == GGML_TYPE_Q5_1 || type == GGML_TYPE_Q8_0) && K == 640 &&
      n_expert_used == 1 && n_tokens > 1;
  const size_t group_bytes =
      by_expert ? 2 * size_t(n_tokens) * sizeof(int32_t)
      : gated && n_tokens > 1
          ? (n_tokens > MMVQ_MAX_BATCH_SIZE
                 ? sizeof(int32_t) +
                       size_t(n_tokens) * n_expert_used * sizeof(MoeBatchGroup)
                 : size_t(mmvq_moe_group_ints(n_tokens, n_expert_used)) *
                       sizeof(int32_t))
          : 0;
  // A caller-quantized input (the same rows quantize_row_q8_1 writes) only
  // needs the pool for the gated route's expert groups.
  const size_t input_bytes = X_q8 ? 0 : nbytes_q8_1;
  ggml_hip_pool_alloc<char> src1_q8_1_pool;
  char* src1_q8_1_ptr = nullptr;
  if (input_bytes + group_bytes > 0) {
    src1_q8_1_pool.alloc(ctx->pool(), input_bytes + group_bytes);
    src1_q8_1_ptr = src1_q8_1_pool.get();
  }
  hipError_t err = hipSuccess;
  int32_t* const sorted_ids =
      by_expert ? reinterpret_cast<int32_t*>(src1_q8_1_ptr + input_bytes)
                : nullptr;
  int32_t* const slot_of_rank = by_expert ? sorted_ids + n_tokens : nullptr;
  if (by_expert) {
    quantize_row_q8_1_by_expert_hip(X_f32, ids, (void*)src1_q8_1_ptr,
                                    sorted_ids, slot_of_rank, K, K,
                                    ne10_padded, n_tokens, stream);
    err = hipGetLastError();
    if (err != hipSuccess) {
      fprintf(stderr, "%s: quantize_row_q8_1_by_expert_hip failed: %s\n", tag,
              hipGetErrorString(err));
      return -2;
    }
  } else if (!X_q8) {
    quantize_row_q8_1_hip(X_f32, nullptr, (void*)src1_q8_1_ptr, type, K,
                          (int64_t)K, (int64_t)K, (int64_t)K * n_tokens,
                          ne10_padded, 1, n_tokens, 1, stream);

    err = hipGetLastError();
    if (err != hipSuccess) {
      fprintf(stderr, "%s: quantize_row_q8_1_hip failed: %s\n", tag,
              hipGetErrorString(err));
      return -2;
    }
  }
  const auto* input = reinterpret_cast<const block_q8_1*>(
      X_q8 ? X_q8 : static_cast<const void*>(src1_q8_1_ptr));
  int32_t* groups = reinterpret_cast<int32_t*>(src1_q8_1_ptr + input_bytes);

  const int input_stride = ne10_padded / QK8_1;
  if (gated) {
    if (n_tokens > MMVQ_MAX_BATCH_SIZE &&
        hipMemsetAsync(groups, 0, sizeof(int32_t), stream) != hipSuccess)
      return -2;
    mul_mat_vec_moe_gated(W, W_b, type, input, ids, groups, out_f32, K, M,
                          n_tokens, n_expert_used, input_stride, stream);
    err = hipGetLastError();
    if (err != hipSuccess) {
      fprintf(stderr, "%s: gated vector launch failed: %s\n", tag,
              hipGetErrorString(err));
      return -3;
    }
    return 0;
  }
  // Gate and up share one Q8 input. Each projection covers every slot in
  // one grid; the kernel retains the per-format number of waves per block.
  for (int projection = 0; projection < (W_b ? 2 : 1); ++projection) {
    const void* weights = projection == 0 ? W : W_b;
    float* output = projection == 0 ? out_f32 : out_b;
    if (n_experts == 1 && n_expert_used == 1 && n_tokens > 1 &&
        n_tokens <= MMVQ_MAX_BATCH_SIZE) {
      // The single-expert contract (dense rows): every token shares the
      // expert, so the multi-token waves compute them per weight-row pair.
      mul_mat_vec_dense_multi(weights, type, input, ids, output, K, M,
                              n_tokens, input_stride, stream);
    } else {
      mul_mat_vec_moe_dispatch(weights, type, input,
                               by_expert ? sorted_ids : ids, output, K, M,
                               n_tokens, n_expert_used, input_stride, stream,
                               slot_of_rank);
    }

    err = hipGetLastError();
    if (err != hipSuccess) {
      fprintf(stderr, "%s: mul_mat_vec_moe_dispatch launch failed: %s\n", tag,
              hipGetErrorString(err));
      return -3;
    }
  }

  return 0;
}

extern "C" int qfn_mmq_moe_vec(int weight_type, const void* W,
                               const float* X_f32, const int32_t* ids,
                               float* out_f32, int M, int K, int n_tokens,
                               int n_experts, int n_expert_used,
                               hipStream_t stream, const void* W_b,
                               float* out_b) {
  return moe_vector_projection("qfn_mmq_moe_vec", weight_type, W, X_f32, ids,
                               out_f32, M, K, n_tokens, n_experts, n_expert_used,
                               stream, W_b, out_b, false);
}

extern "C" int qfn_mmq_moe_gated_vec(int weight_type, const void* gate,
                                     const void* up, const float* x,
                                     const int32_t* ids, float* out, int m,
                                     int k, int tokens, int experts,
                                     int experts_used, hipStream_t stream) {
  return moe_vector_projection("qfn_mmq_moe_gated_vec", weight_type, gate, x,
                               ids, out, m, k, tokens, experts, experts_used,
                               stream, up, nullptr, true);
}

extern "C" int qfn_mmq_moe_gated_vec_preq(int weight_type, const void* gate,
                                          const void* up, const void* x_q8,
                                          const int32_t* ids, float* out,
                                          int m, int k, int tokens, int experts,
                                          int experts_used,
                                          hipStream_t stream) {
  return moe_vector_projection("qfn_mmq_moe_gated_vec_preq", weight_type, gate,
                               nullptr, ids, out, m, k, tokens, experts,
                               experts_used, stream, up, nullptr, true, x_q8);
}

extern "C" int qfn_mmq_q8_0_moe_raw(const void* W, const float* X,
                                    const int32_t* ids, float* out, int M,
                                    int K, int n_tokens, int n_experts,
                                    int n_expert_used, hipStream_t stream) {
  return qfn_mmq_moe_impl<GGML_TYPE_Q8_0>(
      "qfn_mmq_q8_0_moe_raw", W, X, ids, out, M, K, n_tokens, n_experts,
      n_expert_used, stream, nullptr, nullptr);
}

// The dense product is the routed one with a single expert slot per token and
// every token routed to it; the [N][1][M] output is the dense [N][M] layout.
static int dense_vector_projection(const char* tag, int weight_type,
                                   const void* W, const float* X_f32,
                                   const void* X_q8, float* out_f32, int M,
                                   int N, int K, hipStream_t stream) {
  // Same formats as qfn_mmq_dense: the routed set without Q5_1 (down
  // projections only), so decode and prefill accept one dense set.
  if (!qfn_mmq_moe_gated_type(weight_type)) {
    fprintf(stderr, "%s: unsupported weight type %d\n", tag, weight_type);
    return -1;
  }
  if (!W || !(X_f32 || X_q8) || !out_f32 || M <= 0 || K <= 0 || N <= 0) {
    fprintf(stderr, "%s: bad arguments M=%d N=%d K=%d\n", tag, M, N, K);
    return -1;
  }
  ggml_backend_hip_context* ctx = get_ctx_for_device(ggml_hip_get_device());
  if (!ctx) {
    fprintf(stderr, "%s: failed to get HIP context\n", tag);
    return -1;
  }
  ggml_hip_pool_alloc<int32_t> ids(ctx->pool(), (size_t)N);
  if (hipMemsetAsync(ids.get(), 0, (size_t)N * sizeof(int32_t), stream) !=
      hipSuccess) {
    fprintf(stderr, "%s: failed to zero expert ids\n", tag);
    return -2;
  }
  return moe_vector_projection(tag, weight_type, W, X_f32, ids.get(), out_f32,
                               M, K, N, 1, 1, stream, nullptr, nullptr, false,
                               X_q8);
}

extern "C" int qfn_mmq_dense_vec(int weight_type, const void* W,
                                 const float* X_f32, float* out_f32, int M,
                                 int N, int K, hipStream_t stream) {
  return dense_vector_projection("qfn_mmq_dense_vec", weight_type, W, X_f32,
                                 nullptr, out_f32, M, N, K, stream);
}

extern "C" int qfn_mmq_dense_vec_preq(int weight_type, const void* W,
                                      const void* X_q8, float* out_f32, int M,
                                      int N, int K, hipStream_t stream) {
  return dense_vector_projection("qfn_mmq_dense_vec_preq", weight_type, W,
                                 nullptr, X_q8, out_f32, M, N, K, stream);
}

static int moe_mmq_projection(int weight_type, const char* tag, const void* W,
                              const float* X_f32, const int32_t* ids,
                              float* out_f32, int M, int K, int n_tokens,
                              int n_experts, int n_expert_used,
                              hipStream_t stream, const void* W_b,
                              float* out_b, const float* X_up = nullptr) {
  switch (static_cast<ggml_type>(weight_type)) {
    case GGML_TYPE_Q4_K:
      return qfn_mmq_moe_impl<GGML_TYPE_Q4_K>(
          tag, W, X_f32, ids, out_f32, M, K, n_tokens, n_experts, n_expert_used,
          stream, W_b, out_b, X_up);
    case GGML_TYPE_Q5_K:
      return qfn_mmq_moe_impl<GGML_TYPE_Q5_K>(
          tag, W, X_f32, ids, out_f32, M, K, n_tokens, n_experts, n_expert_used,
          stream, W_b, out_b, X_up);
    case GGML_TYPE_Q6_K:
      return qfn_mmq_moe_impl<GGML_TYPE_Q6_K>(
          tag, W, X_f32, ids, out_f32, M, K, n_tokens, n_experts, n_expert_used,
          stream, W_b, out_b, X_up);
    case GGML_TYPE_Q5_1:
      return qfn_mmq_moe_impl<GGML_TYPE_Q5_1>(
          tag, W, X_f32, ids, out_f32, M, K, n_tokens, n_experts, n_expert_used,
          stream, W_b, out_b, X_up);
    case GGML_TYPE_Q8_0:
      return qfn_mmq_moe_impl<GGML_TYPE_Q8_0>(
          tag, W, X_f32, ids, out_f32, M, K, n_tokens, n_experts, n_expert_used,
          stream, W_b, out_b, X_up);
    case GGML_TYPE_Q2_0:
      return qfn_mmq_moe_impl<GGML_TYPE_Q2_0>(
          tag, W, X_f32, ids, out_f32, M, K, n_tokens, n_experts, n_expert_used,
          stream, W_b, out_b, X_up);
    case GGML_TYPE_IQ2_XXS:
      return qfn_mmq_moe_impl<GGML_TYPE_IQ2_XXS>(
          tag, W, X_f32, ids, out_f32, M, K, n_tokens, n_experts, n_expert_used,
          stream, W_b, out_b, X_up);
    case GGML_TYPE_IQ2_XS:
      return qfn_mmq_moe_impl<GGML_TYPE_IQ2_XS>(
          tag, W, X_f32, ids, out_f32, M, K, n_tokens, n_experts, n_expert_used,
          stream, W_b, out_b, X_up);
    case GGML_TYPE_IQ2_S:
      return qfn_mmq_moe_impl<GGML_TYPE_IQ2_S>(
          tag, W, X_f32, ids, out_f32, M, K, n_tokens, n_experts, n_expert_used,
          stream, W_b, out_b, X_up);
    case GGML_TYPE_IQ3_XXS:
      return qfn_mmq_moe_impl<GGML_TYPE_IQ3_XXS>(
          tag, W, X_f32, ids, out_f32, M, K, n_tokens, n_experts, n_expert_used,
          stream, W_b, out_b, X_up);
    case GGML_TYPE_IQ3_S:
      return qfn_mmq_moe_impl<GGML_TYPE_IQ3_S>(
          tag, W, X_f32, ids, out_f32, M, K, n_tokens, n_experts, n_expert_used,
          stream, W_b, out_b, X_up);
    case GGML_TYPE_IQ4_NL:
      return qfn_mmq_moe_impl<GGML_TYPE_IQ4_NL>(
          tag, W, X_f32, ids, out_f32, M, K, n_tokens, n_experts, n_expert_used,
          stream, W_b, out_b, X_up);
    case GGML_TYPE_IQ4_XS:
      return qfn_mmq_moe_impl<GGML_TYPE_IQ4_XS>(
          tag, W, X_f32, ids, out_f32, M, K, n_tokens, n_experts, n_expert_used,
          stream, W_b, out_b, X_up);
    default:
      fprintf(stderr, "%s: unsupported weight type %d\n", tag, weight_type);
      return -1;
  }
}

extern "C" int qfn_mmq_moe_raw(
        int weight_type, const void * W, const float * X, const int32_t * ids,
        float * out, int M, int K, int n_tokens, int n_experts, int n_expert_used,
        hipStream_t stream) {
    return moe_mmq_projection(weight_type, "qfn_mmq_moe_raw", W, X, ids, out, M,
                              K, n_tokens, n_experts, n_expert_used, stream,
                              nullptr, nullptr);
}

// The routed down projection over the gate/up pair's SwiGLU: silu(gate)*up is
// formed in the activation quantize, so the wide pair path skips the separate
// SwigluKernel pass and its F32 round trip. Codes are bit-identical to
// SwigluKernel followed by qfn_mmq_moe_raw.
extern "C" int qfn_mmq_moe_raw_silu(
        int weight_type, const void * W, const float * gate, const float * up,
        const int32_t * ids, float * out, int M, int K, int n_tokens,
        int n_experts, int n_expert_used, hipStream_t stream) {
    if (!gate || !up) return -1;
    return moe_mmq_projection(weight_type, "qfn_mmq_moe_raw_silu", W, gate,
                              ids, out, M, K, n_tokens, n_experts,
                              n_expert_used, stream, nullptr, nullptr, up);
}

extern "C" int qfn_mmq_moe_pair_unique(
    int weight_type, const void * W_a, const void * W_b, const float * X,
    const int32_t * ids, float * out_a, float * out_b, int M, int K,
    int n_tokens, int n_experts, int n_expert_used, hipStream_t stream) {
    if (!W_b || !out_b) return -1;
    return moe_mmq_projection(weight_type, "qfn_mmq_moe_pair_unique", W_a, X,
                              ids, out_a, M, K, n_tokens, n_experts,
                              n_expert_used, stream, W_b, out_b);
}

extern "C" size_t qfn_mmq_q8_1_bytes(int N, int K) {
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    return (size_t)N * ne10_padded * sizeof(block_q8_1) / QK8_1;
}

extern "C" int qfn_mmq_quantize_q8_1(
        const float * X_f32, void * X_q8, int N, int K, hipStream_t stream) {
    if (!X_f32 || !X_q8 || N <= 0 || K <= 0 || K % 32 != 0) {
        fprintf(stderr, "qfn_mmq_quantize_q8_1: bad arguments N=%d K=%d\n", N, K);
        return -1;
    }
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    quantize_row_q8_1_hip(
        X_f32, nullptr, X_q8, GGML_TYPE_Q8_0, K,
        (int64_t)K, (int64_t)K * N, (int64_t)K * N,
        ne10_padded, N, 1, 1, stream);
    return hipGetLastError() == hipSuccess ? 0 : -2;
}

// The 320-row HC down projection has too few waves to hide its long K
// loads. Fetch two iterations together, retaining the MMVQ sum order.
template<int tokens, bool grouped = false>
__launch_bounds__(32) __global__ static void qfn_q8_hc_down_kernel(
        const block_q8_0* __restrict__ weights,
        const block_q8_1* __restrict__ input, float* __restrict__ output) {
    constexpr int blocks = 10240 / QK8_0;
    constexpr int prefetch = 2;
    if constexpr (grouped) {
        input += size_t(blockIdx.y) * tokens * blocks;
        output += size_t(blockIdx.y) * tokens * 320;
    }
    const int lane = threadIdx.x;
    const int part = (lane & 3) * 2;
    float sum[tokens] = {};
    for (int first = lane / 4; first < blocks; first += 8 * prefetch) {
        int v[prefetch][2], u[prefetch][tokens][2];
        half dw[prefetch], dx[prefetch][tokens];
#pragma unroll
        for (int j = 0; j < prefetch; ++j) {
            const int kb = first + 8 * j;
            const auto& w = weights[blockIdx.x * blocks + kb];
            dw[j] = w.d;
#pragma unroll
            for (int i = 0; i < 2; ++i)
                v[j][i] = get_int_b2(w.qs, part + i);
#pragma unroll
            for (int t = 0; t < tokens; ++t) {
                const auto& x = input[t * blocks + kb];
                dx[j][t] = __low2half(x.ds);
#pragma unroll
                for (int i = 0; i < 2; ++i)
                    u[j][t][i] = get_int_b4(x.qs, part + i);
            }
        }
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (int j = 0; j < prefetch; ++j) {
#pragma unroll
            for (int t = 0; t < tokens; ++t) {
                int dot = ggml_hip_dp4a(v[j][0], u[j][t][0], 0);
                dot = ggml_hip_dp4a(v[j][1], u[j][t][1], dot);
                if constexpr (tokens == 1) {
                    const float scale =
                        __fmul_rn(__half2float(dw[j]), __half2float(dx[j][t]));
                    const float product = __fmul_rn(scale, static_cast<float>(dot));
                    sum[t] = __fadd_rn(sum[t], product);
                } else {
                    // Preserve the batched MMVQ weight-dot rounding and
                    // input-scale FMA; reassociating the scales changes logits.
                    const float product =
                        __fmul_rn(__half2float(dw[j]), static_cast<float>(dot));
                    sum[t] = __fmaf_rn(__half2float(dx[j][t]), product, sum[t]);
                }
            }
        }
    }
#pragma unroll
    for (int t = 0; t < tokens; ++t) {
        sum[t] = warp_reduce_sum<32>(sum[t]);
        if (lane == 0) output[t * 320 + blockIdx.x] = sum[t];
    }
}

template<int tokens = 1>
static int prepare_hc_down() {
    hipFuncAttributes attributes{};
    if (hipFuncGetAttributes(
            &attributes,
            reinterpret_cast<const void*>(qfn_q8_hc_down_kernel<tokens, false>)) != hipSuccess)
        return -2;
    if constexpr (tokens < MMVQ_MAX_BATCH_SIZE)
        return prepare_hc_down<tokens + 1>();
    return 0;
}

static int prepare_decode_kernels() {
    // Resolving the first scalar HC function can leave a ~30 ms GPU idle
    // gap after prefill. Load the AR/MTP functions before serving, without
    // trial launches, temporary activations, or changes to session state.
    const auto status = prepare_hc_down();
    return status == 0 ? prepare_q8_decode() : status;
}

template<int tokens = 1>
static void launch_q8_hc_down(const void* weights, const void* input,
                              float* output, int rows, hipStream_t stream) {
    if (rows == tokens) {
        qfn_q8_hc_down_kernel<tokens><<<320, 32, 0, stream>>>(
            static_cast<const block_q8_0*>(weights),
            static_cast<const block_q8_1*>(input), output);
    } else if constexpr (tokens < MMVQ_MAX_BATCH_SIZE) {
        launch_q8_hc_down<tokens + 1>(weights, input, output, rows, stream);
    }
}

extern "C" int qfn_mmq_q8_0_dense_vec_preq(const void* W, const void* W_gate,
                                           const void* X_q8, float* out_f32,
                                           int M, int N, int K,
                                           hipStream_t stream) {
  const bool wide_matrix = K == 2560 && M >= 8192 && N <= 48;
  const bool grouped_matrix =
      N > 48 && N % 32 == 0 &&
      ((K == 2560 && M >= 1024) || (K == 320 && M == 10240));
  if (!W || !X_q8 || !out_f32 || M <= 0 || N <= 0 || K <= 0 || K % 32 != 0 ||
      (N > 32 && !wide_matrix && !grouped_matrix &&
       (M != 320 || K != 10240 || N % 8 != 0)) ||
      (N > MMVQ_MAX_BATCH_SIZE && W_gate)) {
    fprintf(stderr,
            "qfn_mmq_q8_0_dense_vec_preq: bad arguments M=%d N=%d K=%d\n", M, N,
            K);
    return -1;
  }
  if (N <= MMVQ_MAX_BATCH_SIZE && M == 320 && K == 10240 && W_gate == nullptr) {
    launch_q8_hc_down(W, X_q8, out_f32, N, stream);
    return hipGetLastError() == hipSuccess ? 0 : -3;
  }
  if (N > MMVQ_MAX_BATCH_SIZE && N % 8 == 0 && M == 320 && K == 10240) {
    qfn_q8_hc_down_kernel<8, true><<<dim3(320, N / 8), 32, 0, stream>>>(
        static_cast<const block_q8_0*>(W),
        static_cast<const block_q8_1*>(X_q8), out_f32);
    return hipGetLastError() == hipSuccess ? 0 : -3;
  }
  const int input_stride = GGML_PAD(K, MATRIX_ROW_PADDING) / QK8_1;
  mul_mat_vec_q8_dispatch(W, W_gate, static_cast<const block_q8_1*>(X_q8),
                          out_f32, K, M, N, input_stride, stream);
  return hipGetLastError() == hipSuccess ? 0 : -3;
}
template void mul_mat_q_case<GGML_TYPE_Q8_0>(
    ggml_backend_hip_context&, const mmq_args&, hipStream_t);
template void mul_mat_q_case<GGML_TYPE_Q4_K>(
    ggml_backend_hip_context&, const mmq_args&, hipStream_t);
template void mul_mat_q_case<GGML_TYPE_Q5_1>(
    ggml_backend_hip_context&, const mmq_args&, hipStream_t);
template void mul_mat_q_case<GGML_TYPE_Q5_K>(
    ggml_backend_hip_context&, const mmq_args&, hipStream_t);
template void mul_mat_q_case<GGML_TYPE_Q6_K>(
    ggml_backend_hip_context&, const mmq_args&, hipStream_t);
template void mul_mat_q_case<GGML_TYPE_Q2_0>(
    ggml_backend_hip_context&, const mmq_args&, hipStream_t);
template void mul_mat_q_case<GGML_TYPE_IQ2_XXS>(
    ggml_backend_hip_context&, const mmq_args&, hipStream_t);
template void mul_mat_q_case<GGML_TYPE_IQ2_XS>(
    ggml_backend_hip_context&, const mmq_args&, hipStream_t);
template void mul_mat_q_case<GGML_TYPE_IQ2_S>(
    ggml_backend_hip_context&, const mmq_args&, hipStream_t);
template void mul_mat_q_case<GGML_TYPE_IQ3_XXS>(
    ggml_backend_hip_context&, const mmq_args&, hipStream_t);
template void mul_mat_q_case<GGML_TYPE_IQ3_S>(
    ggml_backend_hip_context&, const mmq_args&, hipStream_t);
template void mul_mat_q_case<GGML_TYPE_IQ4_NL>(
    ggml_backend_hip_context&, const mmq_args&, hipStream_t);
template void mul_mat_q_case<GGML_TYPE_IQ4_XS>(
    ggml_backend_hip_context&, const mmq_args&, hipStream_t);
} // namespace qfn_mmq
