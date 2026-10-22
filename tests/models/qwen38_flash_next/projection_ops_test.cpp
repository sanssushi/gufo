#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/quant/ggml_gemm.hpp"
#include "src/models/qwen/vision/device_input.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"
#include "src/models/qwen38_flash_next/rope_scaling.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

float Uniform(std::uint32_t* state, float scale) {
  return scale *
         static_cast<float>(static_cast<int>(NextRandom(state) & 0xFFFFU) -
                            32768) /
         32768.0F;
}

/// Row-major Q8_0 weights [m][k] and their dequantized values.
struct Q8Weights {
  std::vector<std::uint8_t> blocks;
  std::vector<float> values;
};

Q8Weights MakeWeights(std::size_t m, std::size_t k, std::uint32_t seed,
                      bool reference_values = true) {
  Q8Weights w;
  w.blocks.resize(m * (k / 32) * 34);
  if (reference_values)
    w.values.resize(m * k);
  for (std::size_t r = 0; r < m; ++r) {
    for (std::size_t b = 0; b < k / 32; ++b) {
      float raw[32];
      float max_abs = 0.0F;
      for (float& v : raw) {
        v = Uniform(&seed, 1.0F);
        max_abs = std::max(max_abs, std::abs(v));
      }
      const __half d = __float2half(max_abs / 127.0F);
      const float df = __half2float(d);
      std::uint8_t* block = w.blocks.data() + (r * (k / 32) + b) * 34;
      std::memcpy(block, &d, 2);
      for (std::size_t i = 0; i < 32; ++i) {
        const auto qv = static_cast<std::int8_t>(
            std::lround(df != 0.0F ? raw[i] / df : 0.0F));
        block[2 + i] = static_cast<std::uint8_t>(qv);
        if (reference_values)
          w.values[r * k + b * 32 + i] = df * static_cast<float>(qv);
      }
    }
  }
  return w;
}

// Reuse the wide projection fixture to check the fused convolution through
// its actual consumer, including the state carried into the next chunk.
// `quant_type` picks the fused entry: kQ8_0 (the Q8 control stack) runs
// DenseF16SsmGemm; every GSQ-RCO K-quant/IQ stack runs DenseQuantF16SsmGemm.
void CheckSsmProjection(
    const void* w, const __half* x, float* projected,
    const std::vector<float>& reference, std::uint32_t batch,
    gufo::core::GgmlType quant_type = gufo::core::GgmlType::kQ8_0) {
  constexpr std::uint32_t m = 16384, k = 2560, channels = 10240;
  constexpr std::uint32_t kh = 16, vh = 48, d = 128, value_dim = vh * d;
  const std::size_t conv_count = std::size_t(batch) * channels;
  const std::size_t state_count = std::size_t(vh) * d * d;
  const std::size_t out_count = std::size_t(batch) * value_dim;
  struct Buffers {
    std::vector<float*> pointers;
    ~Buffers() {
      for (float* p : pointers) {
        (void)hipFree(p);
      }
    }
    float* Make(std::size_t count) {
      float* p = nullptr;
      CheckHip(hipMalloc(&p, count * sizeof(float)), "SSM allocation");
      pointers.push_back(p);
      return p;
    }
  } buffers;
  std::uint32_t seed = 0x349B71U;
  auto values = [&](std::size_t count, float scale, float offset = 0.0F) {
    std::vector<float> v(count);
    for (float& x : v) {
      x = offset + Uniform(&seed, scale);
    }
    return v;
  };
  auto upload = [](float* p, const std::vector<float>& v) {
    CheckHip(
        hipMemcpy(p, v.data(), v.size() * sizeof(float), hipMemcpyHostToDevice),
        "SSM upload");
  };
  auto input = [&](std::size_t count, float scale, float offset = 0.0F) {
    float* p = buffers.Make(count);
    upload(p, values(count, scale, offset));
    return p;
  };
  auto download = [](const float* p, std::size_t count) {
    std::vector<float> v(count);
    CheckHip(
        hipMemcpy(v.data(), p, count * sizeof(float), hipMemcpyDeviceToHost),
        "SSM download");
    return v;
  };
  auto exact = [&](const std::vector<float>& expected, const float* actual,
                   const char* name) {
    const auto result = download(actual, expected.size());
    if (std::memcmp(expected.data(), result.data(),
                    expected.size() * sizeof(float)) != 0) {
      throw std::runtime_error(std::string("fused SSM changed ") + name);
    }
    for (float value : result) {
      if (!std::isfinite(value)) {
        throw std::runtime_error(std::string("nonfinite SSM ") + name);
      }
    }
  };
  float* conv_w = input(channels * 4, 0.05F);
  float* ab = input(std::size_t(batch) * 2 * vh, 1.0F);
  float* a = input(vh, 0.5F, -1.5F);
  float* dt = input(vh, 1.0F);
  float* norm = input(d, 0.5F, 1.0F);
  const auto history = values(channels * 3, 0.5F);
  const auto initial_state = values(state_count, 0.05F);
  float* conv_state = buffers.Make(history.size());
  float* state = buffers.Make(state_count);
  float* scratch = buffers.Make(conv_count + channels * 4);
  float* qn = buffers.Make(std::size_t(batch) * kh * d);
  float* kn = buffers.Make(std::size_t(batch) * kh * d);
  float* raw = buffers.Make(out_count);
  float* out = buffers.Make(out_count);
  float* fused = buffers.Make(reference.size() + 8);
  auto consume = [&](float* qkvz, bool convolved) {
    q::GatedDeltaNet(qkvz, m, qkvz + channels, m, ab, conv_w, a, dt, norm,
                     conv_state, scratch, qn, kn, raw, state, out, nullptr, {},
                     {}, batch, kh, vh, d, 4, true, convolved, 1e-6F, nullptr);
  };
  upload(conv_state, history);
  upload(state, initial_state);
  consume(projected, false);
  const auto expected_conv = download(scratch, conv_count);
  const auto expected_history = download(conv_state, history.size());
  const auto expected_state = download(state, state_count);
  const auto expected_out = download(out, out_count);
  for (int replay = 0; replay < 2; ++replay) {
    upload(conv_state, history);
    upload(state, initial_state);
    CheckHip(hipMemset(fused, 0xFF, (reference.size() + 8) * sizeof(float)),
             "poison fused projection");
    CheckHip(hipMemset(scratch, 0xFF, conv_count * sizeof(float)),
             "poison fused convolution");
    const bool launched =
        quant_type == gufo::core::GgmlType::kQ8_0
            ? q::DenseF16SsmGemm(w, x, conv_w, conv_state, fused, scratch,
                                 batch, m, k, channels, 4, nullptr)
            : q::DenseQuantF16SsmGemm(quant_type, w, x, conv_w, conv_state,
                                      fused, scratch, batch, m, k, channels, 4,
                                      nullptr);
    if (!launched) {
      throw std::runtime_error("SSM projection rejected the shape");
    }
    exact(expected_conv, scratch, "convolution");
    const auto projected = download(fused, reference.size() + 8);
    for (std::size_t t = 0; t < batch; ++t) {
      const std::size_t first = t + 3 >= batch ? 0 : channels;
      if (std::memcmp(projected.data() + t * m + first,
                      reference.data() + t * m + first,
                      (m - first) * sizeof(float)) != 0) {
        throw std::runtime_error("SSM projection changed Z or final QKV");
      }
    }
    for (std::size_t i = reference.size(); i < projected.size(); ++i) {
      if (std::bit_cast<std::uint32_t>(projected[i]) != 0xFFFFFFFFU) {
        throw std::runtime_error("SSM projection overwrote its guard");
      }
    }
    consume(fused, true);
    exact(expected_history, conv_state, "history");
    exact(expected_state, state, "recurrent state");
    exact(expected_out, out, "output");
  }
  // A rejected geometry must not touch any pointer.
  if (q::DenseF16SsmGemm(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                         0, m, k, channels, 4, nullptr) ||
      q::DenseF16SsmGemm(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                         batch, m, k, channels, 3, nullptr)) {
    throw std::runtime_error("SSM projection accepted an unsupported shape");
  }
  std::cout << "fused SSM convolution, output, state and replay are exact\n";
}

void CheckAttentionProjection(
    const void* weights, const __half* input, const float* projected,
    std::uint32_t batch,
    const gufo::models::qwen38_flash_next::YarnRope& yarn = {}) {
  constexpr std::uint32_t start = 131069;
  const std::size_t query_bytes = (std::size_t(batch) * 6144 + 16) * 4;
  const std::size_t cache_bytes =
      (std::size_t(start + batch) * 512 + 16) * sizeof(__half);
  struct Buffers {
    std::vector<void*> pointers;
    ~Buffers() {
      for (void* p : pointers)
        (void)hipFree(p);
    }
    void* Make(std::size_t bytes) {
      void* p = nullptr;
      CheckHip(hipMalloc(&p, bytes), "attention allocation");
      pointers.push_back(p);
      CheckHip(hipMemset(p, 0xA5, bytes), "attention guards");
      return p;
    }
  } buffers;
  void* expected[4];
  void* actual[4];
  for (unsigned i = 0; i < 4; ++i) {
    const auto bytes = i < 2 ? query_bytes : cache_bytes;
    expected[i] = buffers.Make(bytes);
    actual[i] = buffers.Make(bytes);
  }
  auto* position = static_cast<std::uint32_t*>(buffers.Make(4));
  CheckHip(hipMemcpy(position, &start, 4, hipMemcpyHostToDevice),
           "attention position");
  std::vector<float> gamma(512);
  std::uint32_t seed = 0x34185A9U;
  for (float& value : gamma)
    value = 1.0F + Uniform(&seed, 0.5F);
  auto* device_gamma = static_cast<float*>(buffers.Make(gamma.size() * 4));
  CheckHip(hipMemcpy(device_gamma, gamma.data(), gamma.size() * 4,
                     hipMemcpyHostToDevice),
           "attention norm weights");
  if (!q::PrepareAttention(projected, 13312, device_gamma, device_gamma + 256,
                           static_cast<float*>(expected[0]) + 8,
                           static_cast<float*>(expected[1]) + 8,
                           static_cast<__half*>(expected[2]) + 8,
                           static_cast<__half*>(expected[3]) + 8, batch, 24, 2,
                           256, 64, position, 1e7F, 1e-6F, nullptr, nullptr,
                           false, yarn))
    throw std::runtime_error("attention preparation rejected the model shape");
  hipStream_t stream = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t replay = nullptr;
  CheckHip(hipStreamCreate(&stream), "attention stream");
  gufo::models::qwen::vision::DeviceInput vision;
  vision.PrepareRope(stream);
  const auto* descriptor = vision.rope();
  vision.PrepareRope(stream);
  if (descriptor == nullptr || vision.rope() != descriptor)
    throw std::runtime_error("prepared image descriptor is not stable");
  CheckHip(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal),
           "attention capture");
  if (!q::AttentionF16Gemm(weights, input, device_gamma, device_gamma + 256,
                           static_cast<float*>(actual[0]) + 8,
                           static_cast<float*>(actual[1]) + 8,
                           static_cast<__half*>(actual[2]) + 8,
                           static_cast<__half*>(actual[3]) + 8, batch, position,
                           1e7F, 1e-6F, stream, vision.rope(), yarn))
    throw std::runtime_error("attention fusion rejected the model shape");
  CheckHip(hipStreamEndCapture(stream, &graph), "attention capture end");
  CheckHip(hipGraphInstantiate(&replay, graph, nullptr, nullptr, 0),
           "attention graph");
  for (unsigned repetition = 0; repetition < 2; ++repetition) {
    vision.RestoreLayout({}, stream);
    if (vision.rope() != descriptor)
      throw std::runtime_error("text restore replaced the image descriptor");
    for (unsigned i = 0; i < 4; ++i)
      CheckHip(hipMemsetAsync(actual[i], 0xA5,
                              i < 2 ? query_bytes : cache_bytes, stream),
               "attention replay guards");
    CheckHip(hipGraphLaunch(replay, stream), "attention replay");
    CheckHip(hipStreamSynchronize(stream), "attention wait");
    for (unsigned i = 0; i < 4; ++i) {
      const auto bytes = i < 2 ? query_bytes : cache_bytes;
      std::vector<std::byte> a(bytes), b(bytes);
      CheckHip(hipMemcpy(a.data(), expected[i], bytes, hipMemcpyDeviceToHost),
               "attention reference download");
      CheckHip(hipMemcpy(b.data(), actual[i], bytes, hipMemcpyDeviceToHost),
               "attention output download");
      if (a != b)
        throw std::runtime_error(
            "attention fusion changed rounding, cache boundaries or replay");
    }
  }
  if (q::AttentionF16Gemm(weights, input, device_gamma, device_gamma + 256,
                          static_cast<float*>(actual[0]) + 8,
                          static_cast<float*>(actual[1]) + 8,
                          static_cast<__half*>(actual[2]) + 8,
                          static_cast<__half*>(actual[3]) + 8, 0, position,
                          1e7F, 1e-6F, nullptr, nullptr, yarn))
    throw std::runtime_error("attention fusion accepted an empty batch");
  CheckHip(hipGraphExecDestroy(replay), "attention graph free");
  CheckHip(hipGraphDestroy(graph), "attention graph definition free");
  CheckHip(hipStreamDestroy(stream), "attention stream free");
  std::cout << "fused attention projection yarn=" << yarn.enabled
            << ": exact against PrepareAttention, including replay\n";
}

void CheckHcDownProjection(const void* w, const void* tiled,
                           const float* projected, std::uint32_t batch) {
  const std::size_t count = std::size_t(batch) * 320;
  float* activated = nullptr;
  __half* reference = nullptr;
  __half* output = nullptr;
  CheckHip(hipMalloc(&activated, count * sizeof(float)), "HC allocation");
  CheckHip(hipMalloc(&reference, (count + 16) * sizeof(__half)),
           "HC reference");
  CheckHip(hipMalloc(&output, (count + 16) * sizeof(__half)), "HC output");
  CheckHip(hipMemset(reference, 0xA5, (count + 16) * sizeof(__half)),
           "HC guards");
  CheckHip(hipMemcpy(activated, projected, count * sizeof(float),
                     hipMemcpyDeviceToDevice),
           "HC projection copy");
  q::SiluScale(activated, 0.25F, count, nullptr);
  q::NarrowActivations(activated, reference + 8, false, count, nullptr);
  std::vector<__half> expected(count + 16), actual(count + 16);
  CheckHip(hipMemcpy(expected.data(), reference,
                     expected.size() * sizeof(__half), hipMemcpyDeviceToHost),
           "HC reference download");
  hipStream_t stream = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t replay = nullptr;
  CheckHip(hipStreamCreate(&stream), "HC stream");
  CheckHip(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal),
           "HC capture");
  if (!q::HcDownF16Gemm(w, tiled, output + 8, batch, stream))
    throw std::runtime_error("fused HC down rejected the model shape");
  CheckHip(hipStreamEndCapture(stream, &graph), "HC capture end");
  CheckHip(hipGraphInstantiate(&replay, graph, nullptr, nullptr, 0),
           "HC graph");
  for (unsigned repetition = 0; repetition < 2; ++repetition) {
    CheckHip(
        hipMemsetAsync(output, 0xA5, actual.size() * sizeof(__half), stream),
        "HC output guards");
    CheckHip(hipGraphLaunch(replay, stream), "HC replay");
    CheckHip(
        hipMemcpyAsync(actual.data(), output, actual.size() * sizeof(__half),
                       hipMemcpyDeviceToHost, stream),
        "HC download");
    CheckHip(hipStreamSynchronize(stream), "HC wait");
    if (std::memcmp(actual.data(), expected.data(),
                    actual.size() * sizeof(__half)) != 0)
      throw std::runtime_error(
          "fused HC down changed rounding, guards or replay");
  }
  if (q::HcDownF16Gemm(w, tiled, output + 8, 95, nullptr))
    throw std::runtime_error("fused HC down accepted a short batch");
  CheckHip(hipGraphExecDestroy(replay), "HC graph free");
  CheckHip(hipGraphDestroy(graph), "HC graph definition free");
  CheckHip(hipStreamDestroy(stream), "HC stream free");
  CheckHip(hipFree(output), "HC output free");
  CheckHip(hipFree(reference), "HC reference free");
  CheckHip(hipFree(activated), "HC activation free");
}

/// A batch a few tokens past a full 128-token tile computes its tail with
/// W8A8TailKernel; every row must equal the tiled kernel's row from a batch
/// whose tail is too wide to split.
void CheckW8A8Tail(std::size_t m, std::size_t k, std::uint32_t seed) {
  constexpr std::size_t kShort = 2051;  // three tokens past 16 tiles
  constexpr std::size_t kLong = 2091;   // a 43-token tail stays tiled
  const Q8Weights w = MakeWeights(m, k, seed, false);
  std::vector<float> x(kLong * k);
  std::uint32_t state = seed ^ 0x7A11U;
  for (float& v : x) {
    v = Uniform(&state, 2.0F);
  }
  void* d_w = nullptr;
  float* d_x = nullptr;
  void* d_tiled = nullptr;
  float* d_long = nullptr;
  float* d_short = nullptr;
  CheckHip(hipMalloc(&d_w, w.blocks.size() + 4096), "tail weights");
  CheckHip(hipMalloc(&d_x, x.size() * 4), "tail input");
  CheckHip(hipMalloc(&d_tiled, q::Q8TiledBytes(kLong, k)), "tail tiles");
  CheckHip(hipMalloc(&d_long, kLong * m * 4), "tail long output");
  CheckHip(hipMalloc(&d_short, kLong * m * 4), "tail short output");
  CheckHip(
      hipMemcpy(d_w, w.blocks.data(), w.blocks.size(), hipMemcpyHostToDevice),
      "tail weight upload");
  CheckHip(hipMemcpy(d_x, x.data(), x.size() * 4, hipMemcpyHostToDevice),
           "tail input upload");
  CheckHip(hipMemset(d_short, 0x7F, kLong * m * 4), "tail guards");
  q::QuantizeQ8Tiled(d_x, d_tiled, kLong, k, nullptr);
  if (!q::W8A8Gemm(d_w, d_tiled, d_long, kLong, m, k, nullptr) ||
      !q::W8A8Gemm(d_w, d_tiled, d_short, kShort, m, k, nullptr))
    throw std::runtime_error("W8A8 tail shapes rejected");
  std::vector<float> a(kLong * m);
  std::vector<float> b(kLong * m);
  CheckHip(hipMemcpy(a.data(), d_long, a.size() * 4, hipMemcpyDeviceToHost),
           "tail long download");
  CheckHip(hipMemcpy(b.data(), d_short, b.size() * 4, hipMemcpyDeviceToHost),
           "tail short download");
  if (std::memcmp(a.data(), b.data(), kShort * m * 4) != 0)
    throw std::runtime_error("W8A8 tail rows differ from the tiled rows");
  for (std::size_t i = kShort * m; i < b.size(); ++i) {
    std::uint32_t bits;
    std::memcpy(&bits, &b[i], sizeof(bits));
    if (bits != 0x7F7F7F7FU)
      throw std::runtime_error("W8A8 tail wrote past its batch");
  }
  if (m == 320 && k == 10240) {
    __half* h_long = nullptr;
    __half* h_short = nullptr;
    CheckHip(hipMalloc(&h_long, kLong * m * 2), "HC tail long");
    CheckHip(hipMalloc(&h_short, kLong * m * 2), "HC tail short");
    if (!q::HcDownF16Gemm(d_w, d_tiled, h_long, kLong, nullptr) ||
        !q::HcDownF16Gemm(d_w, d_tiled, h_short, kShort, nullptr))
      throw std::runtime_error("HC down tail shapes rejected");
    std::vector<__half> ha(kShort * m);
    std::vector<__half> hb(kShort * m);
    CheckHip(hipMemcpy(ha.data(), h_long, ha.size() * 2, hipMemcpyDeviceToHost),
             "HC tail long download");
    CheckHip(
        hipMemcpy(hb.data(), h_short, hb.size() * 2, hipMemcpyDeviceToHost),
        "HC tail short download");
    if (std::memcmp(ha.data(), hb.data(), ha.size() * 2) != 0)
      throw std::runtime_error("HC down tail rows differ from the tiled rows");
    (void)hipFree(h_long);
    (void)hipFree(h_short);
  }
  std::cout << "W8A8 tail m=" << m << " k=" << k << ": exact\n";
  (void)hipFree(d_w);
  (void)hipFree(d_x);
  (void)hipFree(d_tiled);
  (void)hipFree(d_long);
  (void)hipFree(d_short);
}

/// The shared-expert producer hands its down projection F16 rows directly.
/// Those bytes must equal the F32 SwiGLU followed by the activation narrow.
void CheckSwigluHalf() {
  constexpr std::size_t kCount = 4099;  // odd count with a partial float4 tail
  std::vector<float> gate(kCount), up(kCount);
  std::uint32_t seed = 0x5A17C0DEU;
  for (std::size_t i = 0; i < kCount; ++i) {
    gate[i] = Uniform(&seed, 4.0F);
    up[i] = Uniform(&seed, 4.0F);
  }
  // F16 ties: the exact midpoints between adjacent half values must round
  // nearest-even. Cover normals, subnormals and both signs; the large
  // products stay below F16 saturation.
  const float mid = (1.0F + (1.0F + std::ldexp(1.0F, -10))) * 0.5F;
  const float subnormal_mid = std::ldexp(1.5F, -24);
  const float ties[][2] = {{1.0F, mid},
                           {mid, 1.0F},
                           {-1.0F, -mid},
                           {-mid, -1.0F},
                           {1.0F, subnormal_mid},
                           {subnormal_mid, 1.0F},
                           {-1.0F, -subnormal_mid},
                           {300.0F, 200.0F}};
  for (std::size_t i = 0; i < sizeof(ties) / sizeof(ties[0]); ++i) {
    gate[i * 7] = ties[i][0];
    up[i * 7] = ties[i][1];
  }
  void* d_gate = nullptr;
  void* d_up = nullptr;
  void* d_ref_gate = nullptr;
  void* d_ref = nullptr;
  void* d_actual = nullptr;
  CheckHip(hipMalloc(&d_gate, kCount * sizeof(float)), "SwiGLU gate");
  CheckHip(hipMalloc(&d_up, kCount * sizeof(float)), "SwiGLU up");
  CheckHip(hipMalloc(&d_ref_gate, kCount * sizeof(float)), "SwiGLU reference");
  CheckHip(hipMalloc(&d_ref, kCount * sizeof(__half)), "SwiGLU half reference");
  CheckHip(hipMalloc(&d_actual, kCount * sizeof(__half)), "SwiGLU half output");
  CheckHip(hipMemcpy(d_gate, gate.data(), kCount * sizeof(float),
                     hipMemcpyHostToDevice),
           "SwiGLU gate upload");
  CheckHip(
      hipMemcpy(d_up, up.data(), kCount * sizeof(float), hipMemcpyHostToDevice),
      "SwiGLU up upload");
  CheckHip(hipMemcpy(d_ref_gate, d_gate, kCount * sizeof(float),
                     hipMemcpyDeviceToDevice),
           "SwiGLU reference copy");
  q::Swiglu(static_cast<float*>(d_ref_gate), static_cast<const float*>(d_up),
            kCount, nullptr);
  q::NarrowActivations(static_cast<const float*>(d_ref_gate), d_ref, false,
                       kCount, nullptr);
  q::SwigluHalf(static_cast<const float*>(d_gate),
                static_cast<const float*>(d_up), static_cast<__half*>(d_actual),
                kCount, nullptr);
  std::vector<__half> reference(kCount), actual(kCount);
  CheckHip(hipMemcpy(reference.data(), d_ref, kCount * sizeof(__half),
                     hipMemcpyDeviceToHost),
           "SwiGLU half reference download");
  CheckHip(hipMemcpy(actual.data(), d_actual, kCount * sizeof(__half),
                     hipMemcpyDeviceToHost),
           "SwiGLU half output download");
  if (std::memcmp(reference.data(), actual.data(), kCount * sizeof(__half)) !=
      0)
    throw std::runtime_error("SwiGLU F16 producer changed rounding");
  std::cout << "SwiGLU F16 producer: exact\n";
  (void)hipFree(d_gate);
  (void)hipFree(d_up);
  (void)hipFree(d_ref_gate);
  (void)hipFree(d_ref);
  (void)hipFree(d_actual);
}

void CheckRoutedQ8Placement() {
  constexpr int rows = 64, cols = 512, tokens = 65, experts = 2;
  const auto weights = MakeWeights(rows * experts, cols, 31415, false);
  std::vector<float> input(tokens * cols), output(tokens * experts * rows);
  std::vector<std::int32_t> ids(tokens * experts);
  std::uint32_t seed = 271828;
  // Isolate the dot product and row layout from quantizer tie semantics.
  // Non-power-of-two scales still exercise FP32 product contraction.
  for (int b = 0; b < cols / 32; ++b) {
    const float scale = 0.0012345F * (1 + b % 7);
    for (int j = 0; j < 32; ++j)
      input[b * 32 + j] =
          (j == 0 ? 127 : int(NextRandom(&seed) % 255) - 127) * scale;
  }
  for (int t = 0; t < tokens; ++t) {
    if (t)
      std::copy_n(input.data(), cols, input.data() + t * cols);
    for (int e = 0; e < experts; ++e)
      ids[t * experts + e] = (t + e) % experts;
  }
  void* dw = nullptr;
  float *dx = nullptr, *dy = nullptr;
  std::int32_t* di = nullptr;
  CheckHip(hipMalloc(&dw, weights.blocks.size()), "routed Q8 weights");
  CheckHip(hipMalloc(&dx, input.size() * sizeof(float)), "routed Q8 input");
  CheckHip(hipMalloc(&dy, output.size() * sizeof(float)), "routed Q8 output");
  CheckHip(hipMalloc(&di, ids.size() * sizeof(std::int32_t)), "routed Q8 IDs");
  CheckHip(hipMemcpy(dw, weights.blocks.data(), weights.blocks.size(),
                     hipMemcpyHostToDevice),
           "routed Q8 upload");
  CheckHip(hipMemcpy(dx, input.data(), input.size() * sizeof(float),
                     hipMemcpyHostToDevice),
           "routed Q8 upload");
  CheckHip(hipMemcpy(di, ids.data(), ids.size() * sizeof(std::int32_t),
                     hipMemcpyHostToDevice),
           "routed Q8 upload");
  qfn_mmq_set_routed_max_expert_rows(tokens);
  qfn_mmq_set_routed_tile_cols(32);
  if (qfn_mmq_moe_raw(static_cast<int>(q::WeightType::kQ8_0), dw, dx, di, dy,
                      rows, cols, tokens, experts, experts, nullptr) != 0)
    throw std::runtime_error("routed Q8 placement projection failed");
  CheckHip(hipMemcpy(output.data(), dy, output.size() * sizeof(float),
                     hipMemcpyDeviceToHost),
           "routed Q8 download");
  for (int row = 0; row < experts * rows; ++row) {
    double expected = 0;
    for (int b = 0; b < cols / 32; ++b) {
      float maximum = 0;
      for (int j = 0; j < 32; ++j)
        maximum = std::max(maximum, std::abs(input[b * 32 + j]));
      const float inverse = 127.0F / maximum;
      const float scale = 1.0F / inverse;
      const auto* block = weights.blocks.data() + (row * (cols / 32) + b) * 34;
      __half weight_scale;
      std::memcpy(&weight_scale, block, sizeof(weight_scale));
      int dot = 0;
      for (int j = 0; j < 32; ++j)
        dot += static_cast<std::int8_t>(block[2 + j]) *
               static_cast<int>(std::round(input[b * 32 + j] * inverse));
      expected += double(__half2float(weight_scale)) * scale * dot;
    }
    if (!std::isfinite(output[row]) ||
        std::abs(double(output[row]) - expected) > 1e-4)
      throw std::runtime_error(
          "routed Q8 differs from FP64 dot oracle at row " +
          std::to_string(row) + ": actual=" + std::to_string(output[row]) +
          " expected=" + std::to_string(expected));
  }
  // Identical activations must not depend on expert packing, column minitile
  // or the ragged final tile. Fast-math formerly contracted these differently.
  for (std::size_t slot = 0; slot < ids.size(); ++slot) {
    if (std::memcmp(output.data() + slot * rows,
                    output.data() + ids[slot] * rows,
                    rows * sizeof(float)) != 0)
      throw std::runtime_error("routed Q8 output depends on row placement");
  }
  qfn_mmq_set_routed_max_expert_rows(0);
  qfn_mmq_set_routed_tile_cols(0);
  for (void* p : {dw, static_cast<void*>(dx), static_cast<void*>(dy),
                  static_cast<void*>(di)})
    CheckHip(hipFree(p), "routed Q8 free");
}

double Run(std::size_t batch, std::size_t m, std::size_t k, std::uint32_t seed,
           std::size_t reference_tokens = 0) {
  const Q8Weights w = MakeWeights(m, k, seed);
  std::vector<float> x(batch * k);
  std::uint32_t state = seed ^ 0xABCDEF01U;
  for (float& v : x) {
    v = Uniform(&state, 2.0F);
  }
  void* d_w = nullptr;
  float* d_x = nullptr;
  float* d_mmq = nullptr;
  float* d_w8 = nullptr;
  float* d_f16 = nullptr;
  __half* d_x_half = nullptr;
  void* d_tiled = nullptr;
  CheckHip(hipMalloc(&d_w, w.blocks.size() + 4096), "hipMalloc");
  CheckHip(hipMalloc(&d_x, x.size() * 4), "hipMalloc");
  CheckHip(hipMalloc(&d_mmq, batch * m * 4), "hipMalloc");
  CheckHip(hipMalloc(&d_w8, batch * m * 4), "hipMalloc");
  CheckHip(hipMalloc(&d_tiled, q::Q8TiledBytes(batch, k)), "hipMalloc");
  CheckHip(hipMalloc(&d_f16, batch * m * 4), "hipMalloc");
  CheckHip(hipMalloc(&d_x_half, x.size() * 2), "hipMalloc");
  CheckHip(
      hipMemcpy(d_w, w.blocks.data(), w.blocks.size(), hipMemcpyHostToDevice),
      "upload");
  CheckHip(hipMemcpy(d_x, x.data(), x.size() * 4, hipMemcpyHostToDevice),
           "upload");
  CheckHip(hipMemset(d_w8, 0, batch * m * 4), "memset");
  CheckHip(hipMemset(d_f16, 0, batch * m * 4), "memset");
  if (qfn_mmq_dense(static_cast<int>(q::WeightType::kQ8_0), d_w, d_x, d_mmq,
                    static_cast<int>(m), static_cast<int>(batch),
                    static_cast<int>(k), nullptr) != 0) {
    throw std::runtime_error("MMQ dense failed");
  }
  q::QuantizeQ8Tiled(d_x, d_tiled, batch, k, nullptr);
  if (!q::W8A8Gemm(d_w, d_tiled, d_w8, batch, m, k, nullptr)) {
    throw std::runtime_error("W8A8 GEMM rejected the shape");
  }
  q::NarrowActivations(d_x, d_x_half, false, x.size(), nullptr);
  if (!q::DenseF16Gemm(d_w, d_x_half, d_f16, batch, m, k, nullptr)) {
    throw std::runtime_error("dense F16 GEMM rejected the shape");
  }
  CheckHip(hipDeviceSynchronize(), "GEMMs");
  std::vector<float> mmq(batch * m);
  std::vector<float> w8(batch * m);
  std::vector<float> f16(batch * m);
  CheckHip(hipMemcpy(f16.data(), d_f16, f16.size() * 4, hipMemcpyDeviceToHost),
           "download");
  CheckHip(hipMemcpy(mmq.data(), d_mmq, mmq.size() * 4, hipMemcpyDeviceToHost),
           "download");
  CheckHip(hipMemcpy(w8.data(), d_w8, w8.size() * 4, hipMemcpyDeviceToHost),
           "download");
  // Both routes quantize the activations per 32-wide block, so they agree
  // to accumulation order; the F64 reference over the dequantized weights
  // bounds the activation quantization itself.
  double worst_vs_mmq = 0.0;
  double worst_f16_vs_mmq = 0.0;
  double worst_vs_ref = 0.0;
  double worst_f16 = 0.0;
  double ref_scale = 0.0;
  for (std::size_t i = 0; i < w8.size(); ++i) {
    if (!std::isfinite(w8[i]) || !std::isfinite(f16[i])) {
      throw std::runtime_error("projection output is not finite");
    }
    worst_vs_mmq =
        std::max(worst_vs_mmq, std::abs(static_cast<double>(mmq[i] - w8[i])));
    worst_f16_vs_mmq = std::max(worst_f16_vs_mmq,
                                std::abs(static_cast<double>(mmq[i] - f16[i])));
  }
  if (!q::W8A8Gemm(d_w, d_tiled, d_w8, batch, m, k, nullptr)) {
    throw std::runtime_error("W8A8 replay rejected the shape");
  }
  std::vector<float> replay(w8.size());
  CheckHip(hipMemcpy(replay.data(), d_w8, replay.size() * sizeof(float),
                     hipMemcpyDeviceToHost),
           "replay download");
  if (replay != w8) {
    throw std::runtime_error("W8A8 replay changed the output");
  }
  if (!q::DenseF16Gemm(d_w, d_x_half, d_f16, batch, m, k, nullptr)) {
    throw std::runtime_error("F16 replay rejected the shape");
  }
  CheckHip(hipMemcpy(replay.data(), d_f16, replay.size() * sizeof(float),
                     hipMemcpyDeviceToHost),
           "F16 replay download");
  if (std::memcmp(replay.data(), f16.data(), replay.size() * sizeof(float)) !=
      0) {
    throw std::runtime_error("F16 replay changed the output");
  }
  if (batch >= 1024 && m == 2560 && k == 6144) {
    const std::size_t stride = k + 32;
    __half* padded = nullptr;
    // The ragged final row ends at the allocation boundary. NaN padding
    // exposes accidental reads without masking them with extra valid rows.
    const auto elements = (batch - 1) * stride + k;
    CheckHip(hipMalloc(&padded, elements * sizeof(__half)), "padded input");
    CheckHip(hipMemset(padded, 0xFF, elements * sizeof(__half)), "padding");
    CheckHip(hipMemcpy2D(padded, stride * sizeof(__half), d_x_half,
                         k * sizeof(__half), k * sizeof(__half), batch,
                         hipMemcpyDeviceToDevice),
             "padded rows");
    if (!q::DenseF16Gemm(d_w, padded, d_f16, batch, m, k, nullptr, stride))
      throw std::runtime_error("strided F16 projection rejected the shape");
    CheckHip(hipMemcpy(replay.data(), d_f16, replay.size() * sizeof(float),
                       hipMemcpyDeviceToHost),
             "strided output");
    CheckHip(hipFree(padded), "free padded input");
    if (std::memcmp(replay.data(), f16.data(), replay.size() * sizeof(float)))
      throw std::runtime_error("strided F16 projection changed the output");
  }

  // Large production shapes still compare every output against MMQ. Sample
  // evenly spaced tokens for the more expensive independent F64 reference.
  const std::size_t samples =
      reference_tokens == 0 ? batch : std::min(batch, reference_tokens);
  for (std::size_t sample = 0; sample < samples; ++sample) {
    const std::size_t t =
        samples > 1 ? sample * (batch - 1) / (samples - 1) : 0;
    for (std::size_t r = 0; r < m; ++r) {
      double ref = 0.0;
      for (std::size_t i = 0; i < k; ++i) {
        ref += static_cast<double>(w.values[r * k + i]) * x[t * k + i];
      }
      const std::size_t idx = t * m + r;
      worst_vs_ref = std::max(worst_vs_ref, std::abs(ref - w8[idx]));
      worst_f16 = std::max(worst_f16, std::abs(ref - f16[idx]));
      ref_scale = std::max(ref_scale, std::abs(ref));
    }
  }
  std::cout << "W8A8 batch=" << batch << " m=" << m << " k=" << k
            << ": worst |W8A8 - MMQ| " << worst_vs_mmq << ", worst |F16 - MMQ| "
            << worst_f16_vs_mmq << ", worst |W8A8 - F64| " << worst_vs_ref
            << ", worst |F16 - F64| " << worst_f16 << " (reference scale "
            << ref_scale << ")\n";
  if (m == 16384 && k == 2560 && batch > 8) {
    CheckSsmProjection(d_w, d_x_half, d_f16, f16,
                       static_cast<std::uint32_t>(batch));
  }
  if (m == 320 && k == 10240 && batch >= 96) {
    CheckHcDownProjection(d_w, d_tiled, d_w8,
                          static_cast<std::uint32_t>(batch));
  }
  if (m == 13312 && k == 2560 && batch > 8) {
    CheckAttentionProjection(d_w, d_x_half, d_f16,
                             static_cast<std::uint32_t>(batch));
    CheckAttentionProjection(
        d_w, d_x_half, d_f16, static_cast<std::uint32_t>(batch),
        gufo::models::qwen38_flash_next::MakeYarnRope(
            {.factor = 2.5F, .original_context = 262144}, 64, 1e7F));
  }
  (void)hipFree(d_w);
  (void)hipFree(d_x);
  (void)hipFree(d_mmq);
  (void)hipFree(d_w8);
  (void)hipFree(d_tiled);
  (void)hipFree(d_f16);
  (void)hipFree(d_x_half);
  // The MMQ agreement is accumulation order; the F64 gap is the shared
  // 8-bit activation quantization, well under 1% of the output scale. The
  // F16 route's gap is its F16 activation rounding, a few ulps smaller.
  return worst_vs_mmq < 1e-3
             ? std::max({worst_vs_ref, worst_f16, worst_f16_vs_mmq}) / ref_scale
             : 1.0;
}

// Unquantized router, alpha/beta and indexer projections must keep the same
// result when a token moves between decode and any verification batch width.
void CheckSmallProjection(q::WeightType type, unsigned rows, unsigned cols) {
  constexpr unsigned tokens = 64;
  const unsigned element_bytes = type == q::WeightType::kF32 ? 4 : 2;
  std::vector<std::uint8_t> weights(std::size_t(rows) * cols * element_bytes);
  std::vector<float> reference_weights(std::size_t(rows) * cols);
  std::vector<float> input(std::size_t(tokens) * cols);
  std::uint32_t seed = 0x319F42U;
  for (std::size_t i = 0; i < reference_weights.size(); ++i) {
    float value = Uniform(&seed, 0.125F);
    if (type == q::WeightType::kF32) {
      std::memcpy(weights.data() + i * 4, &value, 4);
    } else if (type == q::WeightType::kBF16) {
      const auto packed =
          static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value) >> 16);
      std::memcpy(weights.data() + i * 2, &packed, 2);
      value = std::bit_cast<float>(std::uint32_t(packed) << 16);
    } else {
      const __half packed = __float2half(value);
      std::memcpy(weights.data() + i * 2, &packed, 2);
      value = __half2float(packed);
    }
    reference_weights[i] = value;
  }
  for (float& value : input)
    value = Uniform(&seed, 1.7F);
  void* dw = nullptr;
  float* dx = nullptr;
  float* dy = nullptr;
  const std::size_t count = std::size_t(tokens) * rows;
  CheckHip(hipMalloc(&dw, weights.size()), "small weights allocation");
  CheckHip(hipMalloc(&dx, input.size() * sizeof(float)),
           "small input allocation");
  CheckHip(hipMalloc(&dy, (count + 8) * sizeof(float)),
           "small output allocation");
  CheckHip(hipMemcpy(dw, weights.data(), weights.size(), hipMemcpyHostToDevice),
           "small weights upload");
  CheckHip(hipMemcpy(dx, input.data(), input.size() * sizeof(float),
                     hipMemcpyHostToDevice),
           "small input upload");
  for (unsigned token = 0; token < tokens; ++token)
    q::SmallGemm(dw, type, dx + token * cols, dy + token * rows, 1, rows, cols,
                 nullptr);
  std::vector<float> scalar(count), batch(count + 8), replay(count);
  CheckHip(hipMemcpy(scalar.data(), dy, count * sizeof(float),
                     hipMemcpyDeviceToHost),
           "small scalar output");
  for (unsigned token = 0; token < tokens; ++token) {
    for (unsigned row = 0; row < rows; ++row) {
      double expected = 0.0;
      for (unsigned col = 0; col < cols; ++col)
        expected += double(reference_weights[std::size_t(row) * cols + col]) *
                    input[std::size_t(token) * cols + col];
      const float actual = scalar[std::size_t(token) * rows + row];
      if (!std::isfinite(actual) || std::abs(double(actual) - expected) >
                                        2e-5 * (1.0 + std::abs(expected)))
        throw std::runtime_error(
            "small projection differs from FP64 reference");
    }
  }
  for (const unsigned n :
       {1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 16U, 31U, 32U, 33U, tokens}) {
    CheckHip(hipMemset(dy, 0xA5, batch.size() * sizeof(float)),
             "small output poison");
    q::SmallGemm(dw, type, dx, dy, n, rows, cols, nullptr);
    CheckHip(hipMemcpy(batch.data(), dy, batch.size() * sizeof(float),
                       hipMemcpyDeviceToHost),
             "small batch output");
    if (std::memcmp(scalar.data(), batch.data(),
                    std::size_t(n) * rows * sizeof(float)))
      throw std::runtime_error("small projection changes with batch width");
    for (std::size_t i = std::size_t(n) * rows; i < batch.size(); ++i)
      if (std::bit_cast<std::uint32_t>(batch[i]) != 0xA5A5A5A5U)
        throw std::runtime_error("small projection overwrote its output guard");
  }
  q::SmallGemm(dw, type, dx, dy, tokens, rows, cols, nullptr);
  CheckHip(hipMemcpy(replay.data(), dy, count * sizeof(float),
                     hipMemcpyDeviceToHost),
           "small replay output");
  if (std::memcmp(scalar.data(), replay.data(), count * sizeof(float)))
    throw std::runtime_error("small projection replay differs");
  CheckHip(hipFree(dy), "small output free");
  CheckHip(hipFree(dx), "small input free");
  CheckHip(hipFree(dw), "small weights free");
}

void CheckDecodeGrouping(int rows, int cols, int grouped_tokens = 0) {
  const int tokens = grouped_tokens != 0 ? grouped_tokens
                     : rows == 320 && cols == 10240
                         ? 64
                         : (rows >= 8192 && cols == 2560 ? 48 : 32);
  const auto w = MakeWeights(rows, cols, 11, false);
  // The large fixtures exercise matrix dispatch. Gated vectors already have
  // independent-weight coverage in the small fixtures.
  const auto gate =
      rows <= 320 ? MakeWeights(rows, cols, 17, false) : Q8Weights{};
  std::vector<float> x(tokens * cols);
  std::uint32_t seed = 37;
  for (auto& v : x)
    v = Uniform(&seed, 2.0F);
  void* dw = nullptr;
  void* dg = nullptr;
  float* dx = nullptr;
  void* dq = nullptr;
  float* out = nullptr;
  CheckHip(hipMalloc(&dw, w.blocks.size() + 4096), "decode weights");
  if (!gate.blocks.empty())
    CheckHip(hipMalloc(&dg, gate.blocks.size() + 4096), "decode gate");
  CheckHip(hipMalloc(&dx, x.size() * sizeof(float)), "decode inputs");
  CheckHip(hipMalloc(&dq, qfn_mmq_q8_1_bytes(tokens, cols)),
           "decode quantized inputs");
  CheckHip(hipMalloc(&out, (rows * tokens + 4) * sizeof(float)),
           "decode output");
  CheckHip(
      hipMemcpy(dw, w.blocks.data(), w.blocks.size(), hipMemcpyHostToDevice),
      "weights upload");
  if (dg != nullptr)
    CheckHip(hipMemcpy(dg, gate.blocks.data(), gate.blocks.size(),
                       hipMemcpyHostToDevice),
             "gate upload");
  CheckHip(
      hipMemcpy(dx, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice),
      "input upload");
  if (qfn_mmq_quantize_q8_1(dx, dq, tokens, cols, nullptr))
    throw std::runtime_error("decode input quantization failed");
  for (const bool gated : {false, true}) {
    if (gated && dg == nullptr)
      continue;
    for (int t = 0; t < tokens; ++t) {
      const auto* qrow = static_cast<const std::uint8_t*>(dq) +
                         t * qfn_mmq_q8_1_bytes(1, cols);
      if (qfn_mmq_q8_0_dense_vec_preq(dw, gated ? dg : nullptr, qrow,
                                      out + t * rows, rows, 1, cols, nullptr))
        throw std::runtime_error("scalar dense projection failed");
    }
    std::vector<float> scalar(tokens * rows), batch(tokens * rows + 4);
    CheckHip(hipMemcpy(scalar.data(), out, scalar.size() * sizeof(float),
                       hipMemcpyDeviceToHost),
             "scalar output");
    for (int n = 2; n <= (gated ? 8 : tokens); ++n) {
      if (grouped_tokens != 0 && n > 32 && n % 32 != 0)
        continue;
      if (rows == 320 && cols == 10240 && n > 32 && n % 8 != 0)
        continue;
      CheckHip(hipMemset(out, 0xA5, batch.size() * sizeof(float)),
               "decode output guard");
      // End every shape at the allocation boundary. Using a full shared
      // input buffer can hide overreads by a partially populated final wave.
      const auto* input = static_cast<const std::uint8_t*>(dq) +
                          qfn_mmq_q8_1_bytes(tokens - n, cols);
      if (qfn_mmq_q8_0_dense_vec_preq(dw, gated ? dg : nullptr, input, out,
                                      rows, n, cols, nullptr))
        throw std::runtime_error("batched dense projection failed");
      CheckHip(hipMemcpy(batch.data(), out, batch.size() * sizeof(float),
                         hipMemcpyDeviceToHost),
               "batch output");
      if (std::memcmp(scalar.data() + (tokens - n) * rows, batch.data(),
                      n * rows * sizeof(float)) != 0)
        throw std::runtime_error(
            "Q8 dense grouping differs: M=" + std::to_string(rows) +
            " K=" + std::to_string(cols) + " N=" + std::to_string(n) +
            " gated=" + std::to_string(gated));
      for (std::size_t i = std::size_t(n) * rows; i < batch.size(); ++i) {
        if (std::bit_cast<std::uint32_t>(batch[i]) != 0xA5A5A5A5U)
          throw std::runtime_error("Q8 batch projection overwrote its guard");
      }
    }
  }
  for (void* ptr :
       {dw, dg, static_cast<void*>(dx), dq, static_cast<void*>(out)})
    CheckHip(hipFree(ptr), "decode test free");
}

void CheckMtpOutputHead(float input_scale) {
  constexpr int rows = 65, cols = 2560, blocks = cols / 32;
  auto weights = MakeWeights(rows, cols, 73);
  std::vector<float> input(cols), output(rows + 8, -1234567.0F);
  std::uint32_t seed = 31;
  for (auto& value : input)
    value = Uniform(&seed, 2.0F * input_scale);
  std::vector<std::uint8_t> quantized(qfn_mmq_q8_1_bytes(1, cols));
  void *dw = nullptr, *dq = nullptr;
  float *dx = nullptr, *dy = nullptr;
  CheckHip(hipMalloc(&dw, weights.blocks.size()), "MTP Q8 weights");
  CheckHip(hipMalloc(&dq, quantized.size()), "MTP Q8 input");
  CheckHip(hipMalloc(&dx, input.size() * 4), "MTP input");
  CheckHip(hipMalloc(&dy, output.size() * 4), "MTP output");
  CheckHip(hipMemcpy(dw, weights.blocks.data(), weights.blocks.size(),
                     hipMemcpyHostToDevice),
           "weights");
  CheckHip(hipMemcpy(dx, input.data(), input.size() * 4, hipMemcpyHostToDevice),
           "input");
  CheckHip(
      hipMemcpy(dy, output.data(), output.size() * 4, hipMemcpyHostToDevice),
      "guards");
  if (qfn_mmq_quantize_q8_1(dx, dq, 1, cols, nullptr))
    throw std::runtime_error("MTP input quantization failed");
  hipStream_t stream;
  hipGraph_t graph;
  hipGraphExec_t replay;
  CheckHip(hipStreamCreate(&stream), "MTP stream");
  CheckHip(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal),
           "MTP capture");
  if (qfn_mmq_q8_0_dense_vec_preq(dw, nullptr, dq, dy + 4, rows, 1, cols,
                                  stream))
    throw std::runtime_error("MTP Q8 head launch failed");
  CheckHip(hipStreamEndCapture(stream, &graph), "MTP capture end");
  CheckHip(hipGraphInstantiate(&replay, graph, nullptr, nullptr, 0),
           "MTP graph");
  std::vector<float> first;
  for (unsigned repetition = 0; repetition < 2; ++repetition) {
    CheckHip(hipGraphLaunch(replay, stream), "MTP replay");
    CheckHip(hipStreamSynchronize(stream), "MTP wait");
    CheckHip(
        hipMemcpy(output.data(), dy, output.size() * 4, hipMemcpyDeviceToHost),
        "MTP output");
    if (repetition == 0)
      first = output;
    else if (output != first)
      throw std::runtime_error("MTP Q8 replay changed");
  }
  CheckHip(
      hipMemcpy(quantized.data(), dq, quantized.size(), hipMemcpyDeviceToHost),
      "MTP quantized input");
  for (int i = 0; i < rows + 8; ++i)
    if ((i < 4 || i >= rows + 4) && output[i] != -1234567.0F)
      throw std::runtime_error("MTP Q8 output guard changed");
  for (int row = 0; row < rows; ++row) {
    double expected = 0;
    for (int b = 0; b < blocks; ++b) {
      const auto* w = weights.blocks.data() + (row * blocks + b) * 34;
      const auto* x = quantized.data() + b * 36;
      __half wd, xd;
      std::memcpy(&wd, w, 2);
      std::memcpy(&xd, x, 2);
      int dot = 0;
      for (int i = 0; i < 32; ++i)
        dot += static_cast<std::int8_t>(w[2 + i]) *
               static_cast<std::int8_t>(x[4 + i]);
      expected += double(__half2float(wd)) * __half2float(xd) * dot;
    }
    if (!std::isfinite(output[row + 4]) ||
        std::abs(output[row + 4] - expected) > 0.001 * input_scale)
      throw std::runtime_error("MTP Q8 projection differs from F64 dot oracle");
  }
  CheckHip(hipGraphExecDestroy(replay), "MTP graph free");
  CheckHip(hipGraphDestroy(graph), "MTP source graph free");
  CheckHip(hipStreamDestroy(stream), "MTP stream free");
  for (void* p : {dw, dq, static_cast<void*>(dx), static_cast<void*>(dy)})
    CheckHip(hipFree(p), "MTP free");
}

template<typename T>
std::vector<T> Download(const T* device, std::size_t count) {
  std::vector<T> host(count);
  CheckHip(
      hipMemcpy(host.data(), device, count * sizeof(T), hipMemcpyDeviceToHost),
      "download");
  return host;
}

std::size_t QuantBlockElements(q::WeightType type) {
  switch (type) {
    case q::WeightType::kQ8_0:
    case q::WeightType::kIQ4_NL:
      return 32;
    case q::WeightType::kQ2_0:
      return 64;
    default:
      return 256;
  }
}

std::size_t QuantBlockBytes(q::WeightType type) {
  switch (type) {
    case q::WeightType::kQ8_0:
      return 34;
    case q::WeightType::kQ2_0:
    case q::WeightType::kIQ4_NL:
      return 18;
    case q::WeightType::kQ4_K:
      return 144;
    case q::WeightType::kQ5_K:
      return 176;
    case q::WeightType::kQ6_K:
      return 210;
    case q::WeightType::kIQ2_XXS:
      return 66;
    case q::WeightType::kIQ2_XS:
      return 74;
    case q::WeightType::kIQ2_S:
      return 82;
    case q::WeightType::kIQ3_XXS:
      return 98;
    case q::WeightType::kIQ3_S:
      return 110;
    case q::WeightType::kIQ4_XS:
      return 136;
    default:
      return 0;
  }
}

/// Random packed blocks with safe scales; the values come from the CPU
/// dequantization oracle, independently of the GPU kernels.
struct DenseWeights {
  std::vector<std::uint8_t> packed;
  std::vector<float> values;
};

DenseWeights MakeQuant(q::WeightType type, std::size_t m, std::size_t k,
                       std::uint32_t seed) {
  const std::size_t elements = QuantBlockElements(type);
  const std::size_t bytes = QuantBlockBytes(type);
  const std::size_t blocks = m * (k / elements);
  DenseWeights w;
  w.packed.resize(blocks * bytes);
  w.values.resize(m * k);
  for (std::size_t b = 0; b < blocks; ++b) {
    std::uint8_t* blk = w.packed.data() + b * bytes;
    for (std::size_t i = 0; i < bytes; ++i)
      blk[i] = static_cast<std::uint8_t>(NextRandom(&seed) & 0xFF);
    const __half d = __float2half(Uniform(&seed, 0.02F) + 0.03F);
    if (type == q::WeightType::kQ6_K) {
      std::memcpy(blk + bytes - 2, &d, 2);
    } else {
      std::memcpy(blk, &d, 2);
      if (type == q::WeightType::kQ4_K || type == q::WeightType::kQ5_K) {
        const __half dmin = __float2half(Uniform(&seed, 0.01F) + 0.015F);
        std::memcpy(blk + 2, &dmin, 2);
      }
    }
  }
  const auto ggml =
      static_cast<gufo::core::GgmlType>(static_cast<std::uint32_t>(type));
  for (std::size_t r = 0; r < m; ++r) {
    gufo::quant::Dequantize(ggml, w.packed.data() + r * (k / elements) * bytes,
                            w.values.data() + r * k, k);
  }
  return w;
}

/// The GSQ-RCO ssm_in stacks store the K-quant/IQ dense formats and fuse the
/// four-tap convolution into the quantized F16-tile projection. The fused
/// pass must stay bit-identical to the separate dense F16-tile projection
/// (same decode, same K order per output) through its consumer.
void CheckSsmProjectionQuant(gufo::core::GgmlType type, std::uint32_t batch) {
  constexpr std::uint32_t m = 16384, k = 2560;
  const auto wtype =
      static_cast<q::WeightType>(static_cast<std::uint32_t>(type));
  const DenseWeights w = MakeQuant(wtype, m, k, 0x51D6C0DEU);
  std::vector<float> x(static_cast<std::size_t>(batch) * k);
  std::uint32_t state = 0xC0FFEEU;
  for (float& v : x) {
    v = Uniform(&state, 2.0F);
  }
  // The MMQ tier reads whole 256-element k-iterations past short rows;
  // production uploads carry a zeroed tail (device_model.cpp kTailMargin).
  constexpr std::size_t tail = 4096;
  std::uint8_t* d_w = nullptr;
  float* d_x = nullptr;
  __half* d_x_half = nullptr;
  float* d_ref = nullptr;
  CheckHip(hipMalloc(&d_w, w.packed.size() + tail), "SSM quant weights");
  CheckHip(hipMalloc(&d_x, x.size() * 4), "SSM quant input");
  CheckHip(hipMalloc(&d_x_half, x.size() * sizeof(__half)), "SSM quant halves");
  CheckHip(hipMalloc(&d_ref, x.size() / k * m * 4), "SSM quant projection");
  CheckHip(
      hipMemcpy(d_w, w.packed.data(), w.packed.size(), hipMemcpyHostToDevice),
      "SSM quant weights upload");
  CheckHip(hipMemset(d_w + w.packed.size(), 0, tail), "SSM quant weights tail");
  CheckHip(hipMemcpy(d_x, x.data(), x.size() * 4, hipMemcpyHostToDevice),
           "SSM quant input upload");
  q::NarrowActivations(d_x, d_x_half, false, x.size(), nullptr);
  if (!q::DenseQuantF16Gemm(type, d_w, d_x_half, d_ref, batch, m, k, nullptr)) {
    throw std::runtime_error("SSM quant projection rejected the shape");
  }
  const auto reference = Download(d_ref, x.size() / k * m);
  CheckSsmProjection(d_w, d_x_half, d_ref, reference, batch, type);
  CheckHip(hipFree(d_w), "SSM quant free");
  CheckHip(hipFree(d_x), "SSM quant free");
  CheckHip(hipFree(d_x_half), "SSM quant free");
  CheckHip(hipFree(d_ref), "SSM quant free");
}

/// Mixed-type SSM layers keep QKV and gate as separate projections. The
/// standalone QKV matrix can still fuse the four-tap convolution while the
/// independently typed gate runs unchanged. The fused pass must stay
/// bit-identical to the separate QKV projection through GatedDeltaNet,
/// including the boundary raw rows the rolling history and the checkpoint
/// prefix read.
void CheckSsmSeparateProjectionQuant(gufo::core::GgmlType type,
                                     gufo::core::GgmlType gate_type,
                                     std::uint32_t batch,
                                     std::uint32_t checkpoint_tokens) {
  constexpr std::uint32_t channels = 10240, gate_m = 6144, k = 2560;
  constexpr std::uint32_t kh = 16, vh = 48, d = 128, value_dim = vh * d;
  constexpr std::size_t tail = 4096;
  const auto wtype =
      static_cast<q::WeightType>(static_cast<std::uint32_t>(type));
  const auto gtype =
      static_cast<q::WeightType>(static_cast<std::uint32_t>(gate_type));
  const DenseWeights qkv = MakeQuant(wtype, channels, k, 0x6B1D1024U);
  const DenseWeights gate = MakeQuant(gtype, gate_m, k, 0x9A7E6144U);
  std::vector<float> x(static_cast<std::size_t>(batch) * k);
  std::uint32_t seed = 0xC0FFEEU;
  for (float& v : x)
    v = Uniform(&seed, 2.0F);

  struct Buffers {
    std::vector<float*> pointers;
    ~Buffers() {
      for (float* p : pointers) {
        (void)hipFree(p);
      }
    }
    float* Make(std::size_t count) {
      float* p = nullptr;
      CheckHip(hipMalloc(&p, count * sizeof(float)), "separate SSM allocation");
      pointers.push_back(p);
      return p;
    }
  } buffers;
  auto values = [&](std::size_t count, float scale, float offset = 0.0F) {
    std::vector<float> v(count);
    for (float& value : v) {
      value = offset + Uniform(&seed, scale);
    }
    return v;
  };
  auto upload = [](float* p, const std::vector<float>& v) {
    CheckHip(
        hipMemcpy(p, v.data(), v.size() * sizeof(float), hipMemcpyHostToDevice),
        "separate SSM upload");
  };
  auto constant = [&](float* p, std::size_t count, float value) {
    std::vector<float> v(count, value);
    upload(p, v);
  };

  std::uint8_t* d_qkv_w = nullptr;
  std::uint8_t* d_gate_w = nullptr;
  float* d_x = nullptr;
  __half* d_x_half = nullptr;
  CheckHip(hipMalloc(&d_qkv_w, qkv.packed.size() + tail),
           "separate SSM QKV weights");
  CheckHip(hipMalloc(&d_gate_w, gate.packed.size() + tail),
           "separate SSM gate weights");
  CheckHip(hipMalloc(&d_x, x.size() * sizeof(float)), "separate SSM input");
  CheckHip(hipMalloc(&d_x_half, x.size() * sizeof(__half)),
           "separate SSM halves");
  CheckHip(hipMemcpy(d_qkv_w, qkv.packed.data(), qkv.packed.size(),
                     hipMemcpyHostToDevice),
           "separate SSM QKV weights upload");
  CheckHip(hipMemcpy(d_gate_w, gate.packed.data(), gate.packed.size(),
                     hipMemcpyHostToDevice),
           "separate SSM gate weights upload");
  CheckHip(hipMemset(d_qkv_w + qkv.packed.size(), 0, tail),
           "separate SSM QKV weights tail");
  CheckHip(hipMemset(d_gate_w + gate.packed.size(), 0, tail),
           "separate SSM gate weights tail");
  CheckHip(
      hipMemcpy(d_x, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice),
      "separate SSM input upload");
  q::NarrowActivations(d_x, d_x_half, false, x.size(), nullptr);

  const std::size_t conv_count = std::size_t(batch) * channels;
  const std::size_t state_count = std::size_t(vh) * d * d;
  const std::size_t out_count = std::size_t(batch) * value_dim;
  const std::size_t hist_count = std::size_t(3) * channels;
  const float poison = 1.0e30F;
  const auto history = values(hist_count, 0.5F);
  const auto initial_state = values(state_count, 0.05F);
  float* qkv_ref = buffers.Make(conv_count);
  float* z_ref = buffers.Make(out_count);
  float* scratch_ref = buffers.Make(conv_count + 4 * channels);
  float* qkv_fused = buffers.Make(conv_count + 8);
  float* z_fused = buffers.Make(out_count);
  float* scratch_fused = buffers.Make(conv_count + 4 * channels + 8);
  float* conv_state = buffers.Make(hist_count);
  float* state = buffers.Make(state_count);
  float* qn = buffers.Make(std::size_t(batch) * kh * d);
  float* kn = buffers.Make(std::size_t(batch) * kh * d);
  float* raw = buffers.Make(out_count);
  float* out = buffers.Make(out_count);
  float* chk_state = buffers.Make(state_count);
  float* chk_history = buffers.Make(hist_count);
  float* conv_w = buffers.Make(std::size_t(channels) * 4);
  float* ab = buffers.Make(std::size_t(batch) * 2 * vh);
  float* a = buffers.Make(vh);
  float* dt = buffers.Make(vh);
  float* norm = buffers.Make(d);

  const auto conv = values(channels * 4, 0.05F);
  const auto alpha_beta = values(std::size_t(batch) * 2 * vh, 1.0F);
  const auto decay = values(vh, 0.5F, -1.5F);
  const auto delta = values(vh, 1.0F);
  const auto norm_weight = values(d, 0.5F, 1.0F);
  upload(conv_state, history);
  upload(state, initial_state);
  upload(conv_w, conv);
  upload(ab, alpha_beta);
  upload(a, decay);
  upload(dt, delta);
  upload(norm, norm_weight);
  const auto consume = [&](const float* qkv_rows, const float* gate_rows,
                           bool convolved, float* conv_state_rows,
                           float* scratch, float* state_rows, float* out_rows,
                           float* checkpoint_state, float* checkpoint_history) {
    q::GatedDeltaNet(qkv_rows, channels, gate_rows, gate_m, ab, conv_w, a, dt,
                     norm, conv_state_rows, scratch, qn, kn, raw, state_rows,
                     out_rows, nullptr, {}, {}, batch, kh, vh, d, 4, true,
                     convolved, 1e-6F, nullptr, nullptr,
                     q::GdnCheckpoint{checkpoint_state, checkpoint_history,
                                      checkpoint_tokens});
  };
  auto download = [](const float* p, std::size_t count) {
    return Download(p, count);
  };
  auto same = [](const std::vector<float>& expected,
                 const std::vector<float>& actual, const char* name) {
    if (expected.size() != actual.size() ||
        std::memcmp(expected.data(), actual.data(),
                    expected.size() * sizeof(float)) != 0) {
      throw std::runtime_error(std::string("separate SSM changed ") + name);
    }
    for (float value : actual) {
      if (!std::isfinite(value)) {
        throw std::runtime_error(std::string("nonfinite separate SSM ") + name);
      }
    }
  };

  if (!q::DenseQuantF16Gemm(type, d_qkv_w, d_x_half, qkv_ref, batch, channels,
                            k, nullptr) ||
      !q::DenseQuantF16Gemm(gate_type, d_gate_w, d_x_half, z_ref, batch, gate_m,
                            k, nullptr)) {
    throw std::runtime_error("separate SSM projection rejected the shape");
  }
  const auto qkv_reference = download(qkv_ref, conv_count);
  const auto gate_reference = download(z_ref, out_count);
  constant(chk_state, state_count, poison);
  constant(chk_history, hist_count, poison);
  consume(qkv_ref, z_ref, false, conv_state, scratch_ref, state, out, chk_state,
          chk_history);
  const auto expected_conv = download(scratch_ref, conv_count);
  const auto expected_state = download(state, state_count);
  const auto expected_history = download(conv_state, hist_count);
  const auto expected_out = download(out, out_count);
  const auto expected_checkpoint_state = download(chk_state, state_count);
  const auto expected_checkpoint_history = download(chk_history, hist_count);

  for (int replay = 0; replay < 2; ++replay) {
    upload(conv_state, history);
    upload(state, initial_state);
    constant(chk_state, state_count, poison);
    constant(chk_history, hist_count, poison);
    CheckHip(hipMemset(qkv_fused, 0xFF, (conv_count + 8) * sizeof(float)),
             "poison fused QKV");
    CheckHip(hipMemset(scratch_fused, 0xFF,
                       (conv_count + 4 * channels + 8) * sizeof(float)),
             "poison fused convolution");
    if (!q::DenseQuantF16SsmGemm(type, d_qkv_w, d_x_half, conv_w,
                                 history.data(), qkv_fused, scratch_fused,
                                 batch, channels, k, channels, 4, nullptr,
                                 checkpoint_tokens)) {
      throw std::runtime_error("standalone SSM projection rejected the shape");
    }
    if (!q::DenseQuantF16Gemm(gate_type, d_gate_w, d_x_half, z_fused, batch,
                              gate_m, k, nullptr)) {
      throw std::runtime_error("separate SSM gate rejected the shape");
    }
    same(expected_conv, download(scratch_fused, conv_count), "convolution");
    same(gate_reference, download(z_fused, out_count), "gate projection");
    const auto projected = download(qkv_fused, conv_count + 8);
    for (std::uint32_t t = 0; t < batch; ++t) {
      const std::uint32_t tile = t % 32;
      const bool raw = tile < 3 || tile >= 29 || t + 3 >= batch ||
                       (checkpoint_tokens != 0 && t < checkpoint_tokens &&
                        t + 3 >= checkpoint_tokens);
      if (raw && std::memcmp(projected.data() + std::size_t(t) * channels,
                             qkv_reference.data() + std::size_t(t) * channels,
                             std::size_t(channels) * sizeof(float)) != 0) {
        throw std::runtime_error("standalone SSM changed a retained QKV row");
      }
    }
    for (std::size_t i = conv_count; i < projected.size(); ++i) {
      if (std::bit_cast<std::uint32_t>(projected[i]) != 0xFFFFFFFFU) {
        throw std::runtime_error("standalone SSM overwrote its QKV guard");
      }
    }
    consume(qkv_fused, z_fused, true, conv_state, scratch_fused, state, out,
            chk_state, chk_history);
    same(expected_history, download(conv_state, hist_count), "history");
    same(expected_state, download(state, state_count), "recurrent state");
    same(expected_out, download(out, out_count), "output");
    same(expected_checkpoint_state, download(chk_state, state_count),
         "checkpoint state");
    same(expected_checkpoint_history, download(chk_history, hist_count),
         "checkpoint history");
  }
  if (q::DenseQuantF16SsmGemm(type, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, 0, channels, k, channels, 4, nullptr) ||
      q::DenseQuantF16SsmGemm(type, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, batch, gate_m, k, channels, 4,
                              nullptr)) {
    throw std::runtime_error(
        "standalone SSM projection accepted an unsupported shape");
  }
  CheckHip(hipFree(d_qkv_w), "separate SSM free");
  CheckHip(hipFree(d_gate_w), "separate SSM free");
  CheckHip(hipFree(d_x), "separate SSM free");
  CheckHip(hipFree(d_x_half), "separate SSM free");
  std::cout << "standalone fused SSM convolution, gate, state, history and "
               "checkpoint are exact\n";
}

/// Formats outside the adapter set (Q5_1 is routed down-projection only,
/// Q4_0 has no GPU kernels) must fail the executor's format probe and both
/// dense entry points, so a widened whitelist cannot make decode and prefill
/// disagree. The F16-tile dense set is the K-quant/IQ family: the Q8_0 route
/// has its own GEMM and the low-bit formats stay on the adapter's MMQ.
void CheckDenseRejectsNonDense() {
  for (int type : {static_cast<int>(q::WeightType::kQ5_1),
                   static_cast<int>(gufo::core::GgmlType::kQ4_0)}) {
    if (qfn_mmq_moe_gated_type(type) != 0)
      throw std::runtime_error("dense format probe accepted a non-dense type");
    if (qfn_mmq_dense(type, nullptr, nullptr, nullptr, 1, 1, 32, nullptr) == 0)
      throw std::runtime_error("qfn_mmq_dense accepted a non-dense type");
    if (qfn_mmq_dense_vec(type, nullptr, nullptr, nullptr, 1, 1, 32, nullptr) ==
        0)
      throw std::runtime_error("qfn_mmq_dense_vec accepted a non-dense type");
    if (q::KQuantWmmaType(static_cast<gufo::core::GgmlType>(type)))
      throw std::runtime_error("KQuantWmmaType accepted a non-dense type");
    if (q::DenseQuantF16Gemm(static_cast<gufo::core::GgmlType>(type), nullptr,
                             nullptr, nullptr, 1, 1, 32, nullptr))
      throw std::runtime_error("DenseQuantF16Gemm accepted a non-dense type");
  }
  for (q::WeightType type : {q::WeightType::kQ4_K, q::WeightType::kQ5_K,
                             q::WeightType::kQ6_K, q::WeightType::kIQ4_NL,
                             q::WeightType::kIQ3_S, q::WeightType::kIQ4_XS}) {
    if (!q::KQuantWmmaType(static_cast<gufo::core::GgmlType>(
            static_cast<std::uint32_t>(type))))
      throw std::runtime_error("KQuantWmmaType rejected a K-quant format");
  }
  for (q::WeightType type : {q::WeightType::kQ8_0, q::WeightType::kQ2_0,
                             q::WeightType::kIQ2_XXS, q::WeightType::kIQ2_XS,
                             q::WeightType::kIQ2_S, q::WeightType::kIQ3_XXS}) {
    if (q::KQuantWmmaType(static_cast<gufo::core::GgmlType>(
            static_cast<std::uint32_t>(type))))
      throw std::runtime_error("KQuantWmmaType accepted a non-K-quant format");
    if (q::DenseQuantF16Gemm(
            static_cast<gufo::core::GgmlType>(static_cast<std::uint32_t>(type)),
            nullptr, nullptr, nullptr, 1, 1, 32, nullptr))
      throw std::runtime_error(
          "DenseQuantF16Gemm accepted a non-K-quant format");
    if (q::DenseQuantF16SsmGemm(
            static_cast<gufo::core::GgmlType>(static_cast<std::uint32_t>(type)),
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1024, 16384,
            2560, 10240, 4, nullptr))
      throw std::runtime_error(
          "DenseQuantF16SsmGemm accepted a non-K-quant format");
  }
  if (q::DenseQuantF16SsmGemm(gufo::core::GgmlType::kQ4_K, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, 0, 16384,
                              2560, 10240, 4, nullptr))
    throw std::runtime_error("DenseQuantF16SsmGemm accepted a short batch");
}

/// MMQ, MMVQ and (for the K-quant/IQ formats production routes there) WMMA
/// dense projections for a GSQ-RCO format over random packed blocks, against
/// an F64 reference over the CPU-dequantized weights.
void CheckDenseQuant(q::WeightType type, std::size_t m, std::size_t k,
                     std::size_t n_tokens, std::uint32_t seed) {
  const DenseWeights w = MakeQuant(type, m, k, seed);
  std::vector<float> x(n_tokens * k);
  std::uint32_t state = seed ^ 0x5A5A5A5AU;
  for (float& v : x) {
    v = Uniform(&state, 2.0F);
  }
  constexpr std::size_t guard = 8;
  constexpr float poison = -1234567.0F;
  const std::vector<float> zeros(n_tokens * m + guard, poison);
  std::uint8_t* d_w = nullptr;
  float* d_x = nullptr;
  float* d_mmq = nullptr;
  float* d_vec = nullptr;
  float* d_wmma = nullptr;
  std::uint16_t* d_half = nullptr;
  const bool wmma = q::KQuantWmmaType(
      static_cast<gufo::core::GgmlType>(static_cast<std::uint32_t>(type)));
  // The MMQ tier reads whole 256-element k-iterations, so short rows
  // over-read past the tensor; production uploads carry a zeroed tail for
  // this (device_model.cpp kTailMargin).
  constexpr std::size_t tail = 4096;
  CheckHip(hipMalloc(&d_w, w.packed.size() + tail), "dense quant weights");
  CheckHip(hipMalloc(&d_x, x.size() * 4), "dense quant input");
  CheckHip(hipMalloc(&d_mmq, zeros.size() * 4), "dense quant MMQ output");
  CheckHip(hipMalloc(&d_vec, zeros.size() * 4), "dense quant MMVQ output");
  if (wmma) {
    CheckHip(hipMalloc(&d_wmma, zeros.size() * 4), "dense quant WMMA output");
    CheckHip(hipMalloc(&d_half, n_tokens * k * sizeof(__half)),
             "dense quant F16 input");
  }
  CheckHip(
      hipMemcpy(d_w, w.packed.data(), w.packed.size(), hipMemcpyHostToDevice),
      "dense quant weights upload");
  CheckHip(
      hipMemset(static_cast<std::uint8_t*>(d_w) + w.packed.size(), 0, tail),
      "dense quant weights tail");
  CheckHip(hipMemcpy(d_x, x.data(), x.size() * 4, hipMemcpyHostToDevice),
           "dense quant input upload");
  CheckHip(
      hipMemcpy(d_mmq, zeros.data(), zeros.size() * 4, hipMemcpyHostToDevice),
      "dense quant MMQ guards");
  CheckHip(
      hipMemcpy(d_vec, zeros.data(), zeros.size() * 4, hipMemcpyHostToDevice),
      "dense quant MMVQ guards");
  if (wmma) {
    CheckHip(hipMemcpy(d_wmma, zeros.data(), zeros.size() * 4,
                       hipMemcpyHostToDevice),
             "dense quant WMMA guards");
  }
  if (qfn_mmq_dense(static_cast<int>(type), d_w, d_x, d_mmq,
                    static_cast<int>(m), static_cast<int>(n_tokens),
                    static_cast<int>(k), nullptr) != 0)
    throw std::runtime_error("dense MMQ failed");
  if (qfn_mmq_dense_vec(static_cast<int>(type), d_w, d_x, d_vec,
                        static_cast<int>(m), static_cast<int>(n_tokens),
                        static_cast<int>(k), nullptr) != 0)
    throw std::runtime_error("dense MMVQ failed");
  if (wmma) {
    q::NarrowActivations(d_x, d_half, false, n_tokens * k, nullptr);
    if (!q::DenseQuantF16Gemm(
            static_cast<gufo::core::GgmlType>(static_cast<std::uint32_t>(type)),
            d_w, reinterpret_cast<const __half*>(d_half), d_wmma, n_tokens, m,
            k, nullptr))
      throw std::runtime_error("dense F16-tile WMMA failed");
  }
  CheckHip(hipDeviceSynchronize(), "dense quant GEMMs");
  const auto mmq = Download(d_mmq, zeros.size());
  const auto vec = Download(d_vec, zeros.size());
  const auto wmma_out = wmma ? Download(d_wmma, zeros.size())
                             : std::vector<float>(zeros.size(), 0.0F);
  for (std::size_t i = n_tokens * m; i < zeros.size(); ++i) {
    if (mmq[i] != poison || vec[i] != poison || (wmma && wmma_out[i] != poison))
      throw std::runtime_error("dense quant projection overwrote its guard");
  }
  double vs_mmq = 0.0, vs_ref = 0.0, mmq_vs_ref = 0.0, scale = 0.0;
  double wmma_vs_ref = 0.0, wmma_vs_mmq = 0.0;
  for (std::size_t t = 0; t < n_tokens; ++t) {
    for (std::size_t row = 0; row < m; ++row) {
      double ref = 0.0;
      const float* wr = w.values.data() + row * k;
      const float* xr = x.data() + t * k;
      for (std::size_t i = 0; i < k; ++i) {
        ref += static_cast<double>(wr[i]) * xr[i];
      }
      const float v = mmq[t * m + row];
      const float u = vec[t * m + row];
      if (!std::isfinite(v) || !std::isfinite(u)) {
        throw std::runtime_error("dense quant output is not finite");
      }
      vs_mmq = std::max(vs_mmq, std::abs(static_cast<double>(u - v)));
      vs_ref = std::max(vs_ref, std::abs(ref - static_cast<double>(u)));
      mmq_vs_ref = std::max(mmq_vs_ref, std::abs(ref - static_cast<double>(v)));
      scale = std::max(scale, std::abs(ref));
      if (wmma) {
        const float q = wmma_out[t * m + row];
        if (!std::isfinite(q)) {
          throw std::runtime_error("dense quant WMMA output is not finite");
        }
        wmma_vs_ref =
            std::max(wmma_vs_ref, std::abs(ref - static_cast<double>(q)));
        wmma_vs_mmq =
            std::max(wmma_vs_mmq, std::abs(static_cast<double>(v - q)));
      }
    }
  }
  // The vector/MMQ tiers approximate the F64 reference through the same
  // activation quantization, so each may sit a full quantization step from
  // it and two apart from each other. The F16-tile tier rounds its weight
  // tiles and activation rows to F16 instead, a finer step on the same
  // contract.
  if (!(vs_ref < 1e-2 * scale && mmq_vs_ref < 1e-2 * scale &&
        vs_mmq < 2e-2 * scale && wmma_vs_ref < 1e-2 * scale &&
        wmma_vs_mmq < 2e-2 * scale)) {
    throw std::runtime_error(
        "dense quant projection differs from the F64 oracle: m=" +
        std::to_string(m) + " k=" + std::to_string(k) +
        " n=" + std::to_string(n_tokens) + " vs_mmq=" + std::to_string(vs_mmq) +
        " vs_ref=" + std::to_string(vs_ref) +
        " mmq_vs_ref=" + std::to_string(mmq_vs_ref) +
        " wmma_vs_ref=" + std::to_string(wmma_vs_ref) + " wmma_vs_mmq=" +
        std::to_string(wmma_vs_mmq) + " scale=" + std::to_string(scale));
  }
  std::cout << "dense " << std::to_string(static_cast<unsigned>(type))
            << " m=" << m << " k=" << k << " n=" << n_tokens
            << ": worst |MMVQ - MMQ| " << vs_mmq << ", worst |MMVQ - F64| "
            << vs_ref << ", worst |MMQ - F64| " << mmq_vs_ref
            << ", worst |WMMA - F64| " << wmma_vs_ref << ", worst |WMMA - MMQ| "
            << wmma_vs_mmq << " (scale " << scale << ")\n";
  for (void* p : {static_cast<void*>(d_w), static_cast<void*>(d_x),
                  static_cast<void*>(d_mmq), static_cast<void*>(d_vec),
                  static_cast<void*>(d_wmma), static_cast<void*>(d_half)})
    if (p != nullptr)
      CheckHip(hipFree(p), "dense quant free");
}

/// The dense vector projection computes every token of a decode or
/// verification batch in one call: its rows must stay bit-identical to the
/// one-token waves at any batch width (the batched-decode contract).
void CheckDenseVecTokens(q::WeightType type, std::size_t m, std::size_t k,
                         std::size_t tokens, std::uint32_t seed) {
  const DenseWeights w = MakeQuant(type, m, k, seed);
  std::vector<float> x(tokens * k);
  std::uint32_t state = seed ^ 0xA5A5A5A5U;
  for (float& v : x) {
    v = Uniform(&state, 2.0F);
  }
  constexpr std::size_t guard = 8;
  constexpr float poison = -1234567.0F;
  const std::vector<float> zeros(tokens * m + guard, poison);
  std::uint8_t* d_w = nullptr;
  float* d_x = nullptr;
  float* d_out = nullptr;
  constexpr std::size_t tail = 4096;
  CheckHip(hipMalloc(&d_w, w.packed.size() + tail), "dense vec weights");
  CheckHip(hipMalloc(&d_x, x.size() * 4), "dense vec input");
  CheckHip(hipMalloc(&d_out, zeros.size() * 4), "dense vec output");
  CheckHip(
      hipMemcpy(d_w, w.packed.data(), w.packed.size(), hipMemcpyHostToDevice),
      "dense vec weights upload");
  CheckHip(
      hipMemset(static_cast<std::uint8_t*>(d_w) + w.packed.size(), 0, tail),
      "dense vec weights tail");
  CheckHip(hipMemcpy(d_x, x.data(), x.size() * 4, hipMemcpyHostToDevice),
           "dense vec input upload");
  std::vector<float> scalar(tokens * m, poison);
  for (std::size_t t = 0; t < tokens; ++t) {
    CheckHip(
        hipMemcpy(d_out, zeros.data(), zeros.size() * 4, hipMemcpyHostToDevice),
        "dense vec guards");
    if (qfn_mmq_dense_vec(
            static_cast<int>(type), d_w, d_x + static_cast<std::size_t>(t) * k,
            d_out, static_cast<int>(m), 1, static_cast<int>(k), nullptr) != 0)
      throw std::runtime_error("dense vec one-token projection failed");
    CheckHip(hipMemcpy(scalar.data() + static_cast<std::size_t>(t) * m, d_out,
                       m * sizeof(float), hipMemcpyDeviceToHost),
             "dense vec scalar output");
  }
  for (std::size_t n = 2; n <= tokens; ++n) {
    CheckHip(
        hipMemcpy(d_out, zeros.data(), zeros.size() * 4, hipMemcpyHostToDevice),
        "dense vec guards");
    if (qfn_mmq_dense_vec(static_cast<int>(type), d_w, d_x + (tokens - n) * k,
                          d_out, static_cast<int>(m), static_cast<int>(n),
                          static_cast<int>(k), nullptr) != 0)
      throw std::runtime_error("dense vec batched projection failed");
    const auto batch = Download(d_out, zeros.size());
    if (std::memcmp(scalar.data() + (tokens - n) * m, batch.data(),
                    n * m * sizeof(float)) != 0)
      throw std::runtime_error(
          "dense vec rows depend on the token count: m=" + std::to_string(m) +
          " k=" + std::to_string(k) + " n=" + std::to_string(n));
    for (std::size_t i = n * m; i < zeros.size(); ++i) {
      if (batch[i] != poison)
        throw std::runtime_error("dense vec projection overwrote its guard");
    }
  }
  std::cout << "dense vec " << std::to_string(static_cast<unsigned>(type))
            << " m=" << m << " k=" << k << ": rows exact at 1.." << tokens
            << " tokens\n";
  for (void* p : {static_cast<void*>(d_w), static_cast<void*>(d_x),
                  static_cast<void*>(d_out)})
    CheckHip(hipFree(p), "dense vec free");
}

/// One-shot and chunked prefill must write identical logits per output
/// column (session_test CheckPrefillChunks): the projection result for a
/// token may not depend on how many other tokens share the call. Runs the
/// adapter MMQ tier, or the F16-tile WMMA tier production uses at wide batch.
void CheckDenseChunkExact(q::WeightType type, std::size_t m, std::size_t k,
                          std::size_t length, std::size_t boundary,
                          std::uint32_t seed, bool wmma = false) {
  const DenseWeights w = MakeQuant(type, m, k, seed);
  std::vector<float> x(length * k);
  std::uint32_t state = seed ^ 0x5A5A5A5AU;
  for (float& v : x) {
    v = Uniform(&state, 2.0F);
  }
  constexpr std::size_t tail = 4096;
  std::uint8_t* d_w = nullptr;
  float* d_x = nullptr;
  float* d_bulk = nullptr;
  float* d_head = nullptr;
  float* d_tail = nullptr;
  std::uint16_t* d_half = nullptr;
  CheckHip(hipMalloc(&d_w, w.packed.size() + tail), "chunk exact weights");
  CheckHip(hipMalloc(&d_x, x.size() * 4), "chunk exact input");
  CheckHip(hipMalloc(&d_bulk, length * m * 4), "chunk exact bulk output");
  CheckHip(hipMalloc(&d_head, boundary * m * 4), "chunk exact head output");
  CheckHip(hipMalloc(&d_tail, (length - boundary) * m * 4),
           "chunk exact tail output");
  if (wmma) {
    CheckHip(hipMalloc(&d_half, length * k * sizeof(__half)),
             "chunk exact F16 input");
  }
  CheckHip(
      hipMemcpy(d_w, w.packed.data(), w.packed.size(), hipMemcpyHostToDevice),
      "chunk exact weights upload");
  CheckHip(
      hipMemset(static_cast<std::uint8_t*>(d_w) + w.packed.size(), 0, tail),
      "chunk exact weights tail");
  CheckHip(hipMemcpy(d_x, x.data(), x.size() * 4, hipMemcpyHostToDevice),
           "chunk exact input upload");
  const auto run_tier = [&](float* out, std::size_t rows,
                            std::size_t row_offset) {
    if (!wmma) {
      return qfn_mmq_dense(static_cast<int>(type), d_w, d_x + row_offset * k,
                           out, static_cast<int>(m), static_cast<int>(rows),
                           static_cast<int>(k), nullptr) == 0;
    }
    q::NarrowActivations(d_x + row_offset * k, d_half, false, rows * k,
                         nullptr);
    return q::DenseQuantF16Gemm(
        static_cast<gufo::core::GgmlType>(static_cast<std::uint32_t>(type)),
        d_w, reinterpret_cast<const __half*>(d_half), out, rows, m, k, nullptr);
  };
  if (!run_tier(d_bulk, length, 0))
    throw std::runtime_error("chunk exact bulk projection failed");
  if (!run_tier(d_head, boundary, 0))
    throw std::runtime_error("chunk exact head projection failed");
  if (!run_tier(d_tail, length - boundary, boundary))
    throw std::runtime_error("chunk exact tail projection failed");
  CheckHip(hipDeviceSynchronize(), "chunk exact GEMMs");
  const auto bulk = Download(d_bulk, length * m);
  const auto head = Download(d_head, boundary * m);
  const auto tail_out = Download(d_tail, (length - boundary) * m);
  for (std::size_t t = 0; t < length; ++t) {
    const float* chunk = t < boundary ? head.data() + t * m
                                      : tail_out.data() + (t - boundary) * m;
    if (std::memcmp(bulk.data() + t * m, chunk, m * sizeof(float)) != 0) {
      double worst = 0.0;
      std::size_t rows = 0;
      std::size_t first_row = m;
      for (std::size_t r = 0; r < m; ++r) {
        const double d = std::abs(static_cast<double>(bulk[t * m + r]) -
                                  static_cast<double>(chunk[r]));
        if (d != 0.0) {
          ++rows;
          first_row = std::min(first_row, r);
          worst = std::max(worst, d);
        }
      }
      std::cerr << "MISMATCH type=" << static_cast<unsigned>(type) << " m=" << m
                << " k=" << k << " length=" << length
                << " boundary=" << boundary << " token=" << t
                << " rows=" << rows << "/" << m << " first_row=" << first_row
                << " worst=" << worst << " bulk=" << bulk[t * m + first_row]
                << " chunk=" << chunk[first_row] << "\n";
      throw std::runtime_error(
          "dense projection depends on the chunk width: type=" +
          std::to_string(static_cast<unsigned>(type)) +
          " m=" + std::to_string(m) + " k=" + std::to_string(k) + " length=" +
          std::to_string(length) + " boundary=" + std::to_string(boundary) +
          " token=" + std::to_string(t));
    }
  }
  std::cout << "dense chunk exact " << static_cast<unsigned>(type) << " m=" << m
            << " k=" << k << " length=" << length << " boundary=" << boundary
            << (wmma ? " wmma" : " mmq") << "\n";
  for (void* p : {static_cast<void*>(d_w), static_cast<void*>(d_x),
                  static_cast<void*>(d_bulk), static_cast<void*>(d_head),
                  static_cast<void*>(d_tail), static_cast<void*>(d_half)})
    if (p != nullptr)
      CheckHip(hipFree(p), "chunk exact free");
}

/// A stacked upload (device_model Stack) merges same-type projections of
/// one input into one weight matrix whose rows are the concatenated parts:
/// the attention stack is [q|gate; k; v] and the SSM stack [qkv; gate].
/// Every tier must produce each output row from its own weight row alone --
/// the K order is per row -- so the stacked GEMM writes the same rows as
/// the separate projections it merges, and a sliced call over one part of
/// the stack (the MTP catch-up route) matches both.
void CheckDenseStackExact(q::WeightType type,
                          const std::vector<std::size_t>& parts, std::size_t k,
                          std::size_t n_tokens, std::uint32_t seed) {
  std::size_t stacked = 0;
  std::vector<DenseWeights> w;
  for (std::size_t m : parts) {
    stacked += m;
    w.push_back(MakeQuant(type, m, k, seed++));
  }
  std::vector<float> x(n_tokens * k);
  std::uint32_t state = seed ^ 0x5A5A5A5AU;
  for (float& v : x) {
    v = Uniform(&state, 2.0F);
  }
  const std::size_t bytes = QuantBlockBytes(type);
  const std::size_t elements = QuantBlockElements(type);
  std::vector<std::uint8_t> packed;
  for (const DenseWeights& part : w) {
    packed.insert(packed.end(), part.packed.begin(), part.packed.end());
  }
  constexpr std::size_t tail = 4096;
  packed.resize(packed.size() + tail, 0);
  const bool wmma = q::KQuantWmmaType(
      static_cast<gufo::core::GgmlType>(static_cast<std::uint32_t>(type)));
  std::uint8_t* d_w = nullptr;
  float* d_x = nullptr;
  float* d_stacked = nullptr;
  float* d_part = nullptr;
  std::uint16_t* d_half = nullptr;
  CheckHip(hipMalloc(&d_w, packed.size()), "stack exact weights");
  CheckHip(hipMalloc(&d_x, x.size() * 4), "stack exact input");
  CheckHip(hipMalloc(&d_stacked, n_tokens * stacked * 4),
           "stack exact stacked output");
  CheckHip(hipMalloc(&d_part, n_tokens * stacked * 4), "stack exact part");
  if (wmma) {
    CheckHip(hipMalloc(&d_half, n_tokens * k * sizeof(__half)),
             "stack exact F16 input");
  }
  CheckHip(hipMemcpy(d_w, packed.data(), packed.size(), hipMemcpyHostToDevice),
           "stack exact weights upload");
  CheckHip(hipMemcpy(d_x, x.data(), x.size() * 4, hipMemcpyHostToDevice),
           "stack exact input upload");
  if (wmma) {
    q::NarrowActivations(d_x, d_half, false, n_tokens * k, nullptr);
  }
  const char* tier = "";
  const auto check_tier = [&](const char* name, auto&& gemm) {
    tier = name;
    gemm(d_w, stacked, d_stacked);
    const auto full = Download(d_stacked, n_tokens * stacked);
    std::size_t base = 0;
    for (std::size_t i = 0; i < parts.size(); ++i) {
      const std::size_t m = parts[i];
      gemm(d_w + base * (k / elements) * bytes, m, d_part);
      CheckHip(hipDeviceSynchronize(), "stack exact GEMMs");
      const auto part = Download(d_part, n_tokens * m);
      for (std::size_t t = 0; t < n_tokens; ++t) {
        if (std::memcmp(full.data() + t * stacked + base, part.data() + t * m,
                        m * sizeof(float)) != 0) {
          for (std::size_t r = 0; r < m; ++r) {
            if (full[t * stacked + base + r] != part[t * m + r]) {
              std::cerr << "MISMATCH type=" << static_cast<unsigned>(type)
                        << " part=" << i << " m=" << m << " k=" << k
                        << " n=" << n_tokens << " token=" << t << " row=" << r
                        << " stacked=" << full[t * stacked + base + r]
                        << " sliced=" << part[t * m + r] << " tier=" << tier
                        << "\n";
              throw std::runtime_error(
                  "stacked projection row differs from its separate GEMM");
            }
          }
        }
      }
      base += m;
    }
  };
  check_tier("mmq", [&](const void* weights, std::size_t m, float* out) {
    if (qfn_mmq_dense(static_cast<int>(type), weights, d_x, out,
                      static_cast<int>(m), static_cast<int>(n_tokens),
                      static_cast<int>(k), nullptr) != 0)
      throw std::runtime_error(std::string("stack exact MMQ failed: ") + tier);
  });
  check_tier("mmvq", [&](const void* weights, std::size_t m, float* out) {
    if (qfn_mmq_dense_vec(static_cast<int>(type), weights, d_x, out,
                          static_cast<int>(m), static_cast<int>(n_tokens),
                          static_cast<int>(k), nullptr) != 0)
      throw std::runtime_error(std::string("stack exact MMVQ failed: ") + tier);
  });
  if (wmma) {
    check_tier("wmma", [&](const void* weights, std::size_t m, float* out) {
      if (!q::DenseQuantF16Gemm(static_cast<gufo::core::GgmlType>(
                                    static_cast<std::uint32_t>(type)),
                                weights,
                                reinterpret_cast<const __half*>(d_half), out,
                                n_tokens, m, k, nullptr))
        throw std::runtime_error(std::string("stack exact WMMA failed: ") +
                                 tier);
    });
  }
  std::cout << "dense stack exact " << static_cast<unsigned>(type)
            << " m=" << stacked << " k=" << k << " n=" << n_tokens
            << " parts=" << parts.size()
            << (wmma ? " wmma+mmq+mmvq" : " mmq+mmvq") << "\n";
  for (void* p : {static_cast<void*>(d_w), static_cast<void*>(d_x),
                  static_cast<void*>(d_stacked), static_cast<void*>(d_part),
                  static_cast<void*>(d_half)})
    if (p != nullptr)
      CheckHip(hipFree(p), "stack exact free");
}

/// Embedding gather for a GSQ-RCO token table against the CPU oracle.
void CheckEmbedQuant(q::WeightType type, std::size_t rows, std::size_t hidden,
                     std::uint32_t seed) {
  const DenseWeights w = MakeQuant(type, rows, hidden, seed);
  std::vector<std::int32_t> tokens(rows);
  std::uint32_t state = seed ^ 0x5A5A5A5AU;
  for (std::size_t t = 0; t < rows; ++t) {
    tokens[t] = static_cast<std::int32_t>(NextRandom(&state) % rows);
  }
  constexpr unsigned streams = 4;
  std::uint8_t* d_w = nullptr;
  std::int32_t* d_tokens = nullptr;
  float* d_res = nullptr;
  CheckHip(hipMalloc(&d_w, w.packed.size()), "embed weights");
  CheckHip(hipMalloc(&d_tokens, tokens.size() * 4), "embed tokens");
  CheckHip(hipMalloc(&d_res, rows * streams * hidden * 4), "embed output");
  CheckHip(
      hipMemcpy(d_w, w.packed.data(), w.packed.size(), hipMemcpyHostToDevice),
      "embed weights upload");
  CheckHip(hipMemcpy(d_tokens, tokens.data(), tokens.size() * 4,
                     hipMemcpyHostToDevice),
           "embed tokens upload");
  q::EmbedTokens(d_w, type, d_tokens, d_res, static_cast<std::uint32_t>(rows),
                 static_cast<std::uint32_t>(hidden), streams, nullptr);
  CheckHip(hipDeviceSynchronize(), "embed");
  const auto res = Download(d_res, rows * streams * hidden);
  double worst = 0.0, scale = 0.0;
  for (std::size_t t = 0; t < rows; ++t) {
    for (std::size_t i = 0; i < hidden; ++i) {
      const float expected =
          w.values[static_cast<std::size_t>(tokens[t]) * hidden + i];
      for (std::size_t s = 0; s < streams; ++s) {
        const float v = res[(t * streams + s) * hidden + i];
        worst = std::max(worst, std::abs(static_cast<double>(v - expected)));
        scale = std::max(scale, std::abs(static_cast<double>(expected)));
      }
    }
  }
  if (!(worst <= 1e-6 * (scale + 1.0))) {
    throw std::runtime_error("embedding gather differs from the CPU oracle: " +
                             std::to_string(worst) + " scale " +
                             std::to_string(scale));
  }
  std::cout << "embed " << std::to_string(static_cast<unsigned>(type))
            << " rows=" << rows << " hidden=" << hidden
            << ": worst |GPU - CPU| " << worst << " (scale " << scale << ")\n";
  for (void* p : {static_cast<void*>(d_w), static_cast<void*>(d_tokens),
                  static_cast<void*>(d_res)})
    CheckHip(hipFree(p), "embed free");
}

}  // namespace

int main() {
  try {
    CheckHip(q::PrepareW8A8GemmWave64(), "prepare wide Q8 projection");
    if (qfn_mmq_init(0) != 0)
      throw std::runtime_error("prepare quantized decode projections");
    CheckRoutedQ8Placement();
    CheckSmallProjection(q::WeightType::kF32, 513, 2560);
    CheckSmallProjection(q::WeightType::kF32, 96, 2560);
    CheckSmallProjection(q::WeightType::kBF16, 129, 2560);
    CheckSmallProjection(q::WeightType::kF16, 513, 2560);
    CheckSmallProjection(q::WeightType::kF16, 96, 2560);
    CheckSmallProjection(q::WeightType::kBF16, 128, 2560);
    CheckSmallProjection(q::WeightType::kBF16, 512, 2560);
    CheckSmallProjection(q::WeightType::kF16, 7, 131);
    CheckDecodeGrouping(64, 2560);
    CheckDecodeGrouping(320, 10240);
    CheckDecodeGrouping(1024, 2560, 256);
    CheckDecodeGrouping(10240, 320, 256);
    CheckDecodeGrouping(2561, 2560, 64);
    CheckDecodeGrouping(12289, 2560);
    CheckDecodeGrouping(65537, 2560);
    CheckMtpOutputHead(1.0F);
    // Small activations must retain products below F16's normal range.
    CheckMtpOutputHead(0.0001F);
    CheckW8A8Tail(320, 10240, 0x7A1320U);
    CheckW8A8Tail(640, 2560, 0x7A1640U);
    CheckSwigluHalf();
    CheckDenseRejectsNonDense();
    // Per-type dense projections against the CPU-dequantized oracle: the
    // decode-width vector path, the tiled MMQ path, and a wide batch.
    std::uint32_t seed = 0x1D1D1D1DU;
    for (q::WeightType type :
         {q::WeightType::kQ8_0, q::WeightType::kQ2_0, q::WeightType::kQ4_K,
          q::WeightType::kQ5_K, q::WeightType::kQ6_K, q::WeightType::kIQ2_XXS,
          q::WeightType::kIQ2_XS, q::WeightType::kIQ2_S,
          q::WeightType::kIQ3_XXS, q::WeightType::kIQ3_S,
          q::WeightType::kIQ4_NL, q::WeightType::kIQ4_XS}) {
      CheckDenseQuant(type, 64, 512, 1, seed++);
      CheckDenseQuant(type, 64, 512, 7, seed++);
      CheckDenseQuant(type, 129, 2560, 24, seed++);
    }
    // Production shapes: attn_qkv / shexp gate-up / shexp down.
    CheckDenseQuant(q::WeightType::kQ6_K, 10240, 2560, 1, seed++);
    CheckDenseQuant(q::WeightType::kQ6_K, 640, 2560, 7, seed++);
    CheckDenseQuant(q::WeightType::kIQ4_XS, 10240, 2560, 1, seed++);
    CheckDenseQuant(q::WeightType::kIQ4_XS, 640, 2560, 7, seed++);
    CheckDenseQuant(q::WeightType::kQ5_K, 10240, 2560, 1, seed++);
    CheckDenseQuant(q::WeightType::kIQ4_NL, 2560, 640, 7, seed++);
    CheckDenseQuant(q::WeightType::kQ2_0, 2560, 640, 7, seed++);
    // Tight K tails: a single weight block and a one-block trailing tile.
    CheckDenseQuant(q::WeightType::kQ2_0, 64, 64, 1, seed++);
    CheckDenseQuant(q::WeightType::kQ2_0, 64, 64, 7, seed++);
    CheckDenseQuant(q::WeightType::kQ2_0, 129, 320, 24, seed++);
    // Chunk-width exactness at the session test's boundary pairs: the MMQ
    // tile width may not depend on the token count. (160,80) keeps every
    // token's tile slot identical across the three calls, so it is the
    // slot-control pair; (96,32) shifts slots at the boundary.
    for (q::WeightType type :
         {q::WeightType::kQ8_0, q::WeightType::kQ2_0, q::WeightType::kQ4_K,
          q::WeightType::kQ5_K, q::WeightType::kQ6_K, q::WeightType::kIQ2_XXS,
          q::WeightType::kIQ2_XS, q::WeightType::kIQ2_S,
          q::WeightType::kIQ3_XXS, q::WeightType::kIQ3_S,
          q::WeightType::kIQ4_NL, q::WeightType::kIQ4_XS}) {
      CheckDenseChunkExact(type, 64, 512, 160, 80, seed++);
      CheckDenseChunkExact(type, 64, 512, 96, 32, seed++);
      CheckDenseChunkExact(type, 64, 512, 96, 48, seed++);
      CheckDenseChunkExact(type, 64, 512, 128, 64, seed++);
      CheckDenseChunkExact(type, 64, 512, 136, 94, seed++);
      CheckDenseChunkExact(type, 64, 512, 136, 127, seed++);
      CheckDenseChunkExact(type, 64, 512, 136, 135, seed++);
      CheckDenseChunkExact(type, 64, 512, 2048, 1025, seed++);
    }
    CheckDenseChunkExact(q::WeightType::kIQ4_XS, 10240, 2560, 136, 94, seed++);
    CheckDenseChunkExact(q::WeightType::kIQ4_XS, 2560, 6144, 17, 9, seed++);
    CheckDenseChunkExact(q::WeightType::kIQ4_XS, 2560, 6144, 32, 16, seed++);
    CheckDenseChunkExact(q::WeightType::kQ6_K, 10240, 2560, 4096, 2048, seed++);
    CheckDenseChunkExact(q::WeightType::kQ2_0, 2560, 640, 136, 94, seed++);
    // The F16-tile WMMA tier production runs at wide batch must keep the
    // same contract, including across its 96-token launch-config boundary
    // and the row-tile groups its production shapes draw.
    for (q::WeightType type : {q::WeightType::kQ4_K, q::WeightType::kQ5_K,
                               q::WeightType::kQ6_K, q::WeightType::kIQ4_NL,
                               q::WeightType::kIQ3_S, q::WeightType::kIQ4_XS}) {
      CheckDenseChunkExact(type, 64, 512, 160, 80, seed++, true);
      CheckDenseChunkExact(type, 64, 512, 96, 32, seed++, true);
      CheckDenseChunkExact(type, 64, 512, 136, 94, seed++, true);
      CheckDenseChunkExact(type, 64, 512, 136, 135, seed++, true);
    }
    CheckDenseChunkExact(q::WeightType::kQ6_K, 10240, 2560, 136, 94, seed++,
                         true);
    CheckDenseChunkExact(q::WeightType::kQ6_K, 10240, 2560, 130, 97, seed++,
                         true);
    CheckDenseChunkExact(q::WeightType::kQ4_K, 2560, 6144, 136, 94, seed++,
                         true);
    CheckDenseChunkExact(q::WeightType::kIQ4_NL, 2560, 640, 136, 94, seed++,
                         true);
    // Stacked-vs-separate exactness for the formats the GSQ-RCO mixes
    // actually stack (same-type attention triples and SSM pairs), at the
    // production shapes: [12288; 512; 512] attention and [10240; 6144]
    // SSM projections over k=2560.
    for (q::WeightType type : {q::WeightType::kQ4_K, q::WeightType::kQ5_K,
                               q::WeightType::kQ6_K, q::WeightType::kIQ4_XS}) {
      CheckDenseStackExact(type, {12288, 512, 512}, 2560, 7, seed++);
      CheckDenseStackExact(type, {12288, 512, 512}, 2560, 96, seed++);
      CheckDenseStackExact(type, {10240, 6144}, 2560, 7, seed++);
      CheckDenseStackExact(type, {10240, 6144}, 2560, 96, seed++);
    }
    // Token-table gathers for the GSQ embedding formats.
    CheckEmbedQuant(q::WeightType::kIQ3_S, 17, 2560, seed++);
    CheckEmbedQuant(q::WeightType::kIQ4_XS, 17, 2560, seed++);
    CheckEmbedQuant(q::WeightType::kIQ3_S, 5, 512, seed++);
    CheckEmbedQuant(q::WeightType::kIQ4_XS, 5, 512, seed++);
    // Decode and verification batch widths share the dense vector entry: the
    // rows must be bit-identical across the batch width.
    for (q::WeightType type :
         {q::WeightType::kQ8_0, q::WeightType::kQ2_0, q::WeightType::kQ4_K,
          q::WeightType::kQ5_K, q::WeightType::kQ6_K, q::WeightType::kIQ2_XXS,
          q::WeightType::kIQ2_XS, q::WeightType::kIQ2_S,
          q::WeightType::kIQ3_XXS, q::WeightType::kIQ3_S,
          q::WeightType::kIQ4_NL, q::WeightType::kIQ4_XS}) {
      CheckDenseVecTokens(type, 64, 512, 8, seed++);
      CheckDenseVecTokens(type, 129, 2560, 8, seed++);
    }
    CheckDenseVecTokens(q::WeightType::kQ6_K, 2560, 6144, 8, seed++);
    CheckDenseVecTokens(q::WeightType::kIQ4_XS, 2560, 6144, 8, seed++);
    CheckDenseVecTokens(q::WeightType::kIQ4_NL, 2560, 640, 8, seed++);
    CheckDenseVecTokens(q::WeightType::kQ2_0, 2560, 640, 8, seed++);
    CheckDenseVecTokens(q::WeightType::kQ2_0, 129, 320, 8, seed++);
    bool ok = true;
    // Ragged batch and rows against the 128-wide macro tiles, the 64-token
    // tile below 96, and the model's ssm_out / shexp_down widths.
    ok = Run(100, 320, 2560, 0x1234ABCDU) < 1e-2 && ok;
    ok = Run(37, 640, 2560, 0x0BADF00DU) < 1e-2 && ok;
    ok = Run(200, 200, 6144, 0xDEADBEEFU) < 1e-2 && ok;
    ok = Run(1025, 2560, 6144, 0x51A17U, 2) < 1e-2 && ok;
    ok = Run(2049, 2560, 6144, 0x25606144U, 2) < 1e-2 && ok;
    // Wide SSM projection: two K blocks per stage, with a partial token tile.
    ok = Run(2049, 16384, 2560, 0x16384256U, 2) < 1e-2 && ok;
    ok = Run(2049, 13312, 2560, 0x13312256U, 2) < 1e-2 && ok;
    // Every prefill pass wider than a vector batch fuses these projections:
    // one partial 32-token tile, a ragged 128-token tile and a mid-size pass.
    for (const std::size_t batch : {9U, 33U, 470U}) {
      const auto seed = static_cast<std::uint32_t>(batch);
      ok = Run(batch, 16384, 2560, 0x16384000U + seed, 2) < 1e-2 && ok;
      ok = Run(batch, 13312, 2560, 0x13312000U + seed, 2) < 1e-2 && ok;
    }
    // The GSQ-RCO ssm stacks fuse the same convolution over every dense
    // F16-tile weight format, at a full and a partial final token tile.
    for (auto type :
         {gufo::core::GgmlType::kQ4_K, gufo::core::GgmlType::kQ5_K,
          gufo::core::GgmlType::kQ6_K, gufo::core::GgmlType::kIQ4_NL,
          gufo::core::GgmlType::kIQ4_XS, gufo::core::GgmlType::kIQ3_S}) {
      CheckSsmProjectionQuant(type, 1024);
    }
    CheckSsmProjectionQuant(gufo::core::GgmlType::kIQ4_XS, 1025);
    // Mixed-type SSM layers: the standalone QKV matrix fuses the same
    // convolution while the gate keeps its own type, with the checkpoint
    // prefix and rolling history reading the retained raw rows. Every
    // prefill pass wider than a vector batch takes the fused route.
    CheckSsmSeparateProjectionQuant(gufo::core::GgmlType::kQ6_K,
                                    gufo::core::GgmlType::kIQ3_S, 9, 6);
    CheckSsmSeparateProjectionQuant(gufo::core::GgmlType::kIQ4_XS,
                                    gufo::core::GgmlType::kQ4_K, 33, 30);
    CheckSsmSeparateProjectionQuant(gufo::core::GgmlType::kQ5_K,
                                    gufo::core::GgmlType::kQ6_K, 470, 468);
    CheckSsmSeparateProjectionQuant(gufo::core::GgmlType::kQ6_K,
                                    gufo::core::GgmlType::kIQ3_S, 1024, 1000);
    CheckSsmSeparateProjectionQuant(gufo::core::GgmlType::kIQ4_XS,
                                    gufo::core::GgmlType::kQ4_K, 1025, 1025);
    CheckSsmSeparateProjectionQuant(gufo::core::GgmlType::kQ5_K,
                                    gufo::core::GgmlType::kQ6_K, 2049, 2046);
    // HC up: the grouped grid, including the last partial token tile.
    ok = Run(2049, 10240, 320, 0x8A8A320U, 2) < 1e-2 && ok;
    // HC down fusion: its first supported batch and a partial final tile.
    ok = Run(96, 320, 10240, 0x320A96U, 2) < 1e-2 && ok;
    ok = Run(2049, 320, 10240, 0x320A2049U, 2) < 1e-2 && ok;
    return ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
