#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {

constexpr std::uint32_t kTokens = 19;
constexpr std::uint32_t kHidden = 2560;
constexpr std::uint32_t kStreams = 4;
constexpr std::uint32_t kNormPad = 32;

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

template<typename T>
class HipBuffer {
public:
  explicit HipBuffer(std::size_t count) : count_(count) {
    void* allocation = nullptr;
    CheckHip(hipMalloc(&allocation, bytes()), "hipMalloc");
    data_ = static_cast<T*>(allocation);
  }
  ~HipBuffer() {
    if (data_ != nullptr) {
      (void)hipFree(data_);
    }
  }

  HipBuffer(const HipBuffer&) = delete;
  HipBuffer& operator=(const HipBuffer&) = delete;
  HipBuffer(HipBuffer&&) = delete;
  HipBuffer& operator=(HipBuffer&&) = delete;

  [[nodiscard]] T* get() noexcept { return data_; }
  [[nodiscard]] std::size_t bytes() const noexcept {
    return count_ * sizeof(T);
  }

private:
  T* data_{nullptr};
  std::size_t count_{0};
};

std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

std::vector<float> MakeValues(std::size_t count, std::uint32_t seed,
                              float scale) {
  std::vector<float> values(count);
  for (float& value : values) {
    value = scale *
            static_cast<float>(static_cast<int>(NextRandom(&seed) & 0xFFFFU) -
                               32768) /
            32768.0F;
  }
  return values;
}

template<typename T>
void Upload(HipBuffer<T>* destination, const std::vector<T>& source) {
  CheckHip(hipMemcpy(destination->get(), source.data(), destination->bytes(),
                     hipMemcpyHostToDevice),
           "upload");
}

std::vector<float> Download(HipBuffer<float>* source, std::size_t count) {
  std::vector<float> values(count);
  CheckHip(hipMemcpy(values.data(), source->get(), source->bytes(),
                     hipMemcpyDeviceToHost),
           "download");
  return values;
}

std::vector<__half> Download(HipBuffer<__half>* source, std::size_t count) {
  std::vector<__half> values(count);
  CheckHip(hipMemcpy(values.data(), source->get(),
                     values.size() * sizeof(__half), hipMemcpyDeviceToHost),
           "download");
  return values;
}

std::uint16_t FloatToBf16(float v) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &v, sizeof(bits));
  // Round to nearest even over the seven BF16 mantissa bits.
  bits += 0x7FFFu + ((bits >> 16) & 1u);
  return static_cast<std::uint16_t>(bits >> 16);
}

float Bf16ToFloat(std::uint16_t h) {
  const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
  float v = 0.0F;
  std::memcpy(&v, &bits, sizeof(v));
  return v;
}

// Every fused or BF16 path must preserve the reference path's bytes down to
// the last bit: the CPU expansion of a BF16 weight is the F32 or F16 value
// both routes compute with.
template<typename T>
void Exact(HipBuffer<T>& reference, HipBuffer<T>& actual, const char* name) {
  std::vector<std::uint8_t> a(reference.bytes()), b(actual.bytes());
  CheckHip(
      hipMemcpy(a.data(), reference.get(), a.size(), hipMemcpyDeviceToHost),
      "reference download");
  CheckHip(hipMemcpy(b.data(), actual.get(), b.size(), hipMemcpyDeviceToHost),
           "fused download");
  if (a != b) {
    throw std::runtime_error(std::string("HC fusion changed ") + name);
  }
}

// A few rows and tokens against an independent double dot: the shared
// F16 GEMM's math on the mixer shapes the BLAS contract checks do not cover.
void SpotOracle(const std::vector<__half>& w, const std::vector<__half>& x,
                const std::vector<float>& out, std::uint32_t m, std::uint32_t k,
                std::uint32_t batch, const char* name) {
  double scale = 0.0;
  double worst = 0.0;
  const std::uint32_t rows[4] = {0, m / 3, 2 * m / 3, m - 1};
  for (std::uint32_t t = 0; t < batch; t += std::max(1U, batch / 4)) {
    for (std::uint32_t r : rows) {
      double expected = 0.0;
      for (std::uint32_t i = 0; i < k; ++i) {
        expected += static_cast<double>(
                        __half2float(w[static_cast<std::size_t>(r) * k + i])) *
                    __half2float(x[static_cast<std::size_t>(t) * k + i]);
      }
      const double actual = out[static_cast<std::size_t>(t) * m + r];
      scale = std::max(scale, std::abs(expected));
      worst = std::max(worst, std::abs(expected - actual));
    }
  }
  std::cout << name << " worst |F16 - F64| " << worst << " (reference scale "
            << scale << ")\n";
  if (!std::isfinite(worst) || worst > 1e-4 * std::max(1.0, scale)) {
    throw std::runtime_error(std::string(name) +
                             " differs from the F64 oracle");
  }
}

// Fusing the projection must preserve every downstream representation,
// including the inject partials' reduction order and quantization ties.
void CheckFusedProjection(std::uint32_t tokens) {
  constexpr std::size_t kRank = 320;
  constexpr std::size_t kDim = kStreams * kHidden;
  const std::size_t mixed_count = static_cast<std::size_t>(tokens) * kHidden;
  const std::size_t inject_count = static_cast<std::size_t>(tokens) * kStreams *
                                   q::HcInjectPartsVec4(kHidden);
  const std::size_t q8_bytes = q::Q8TiledBytes(tokens, kHidden);
  const auto low =
      MakeValues(static_cast<std::size_t>(tokens) * kRank, 0x12345678U, 1.0F);
  const auto norm =
      MakeValues(static_cast<std::size_t>(tokens) * kDim, 0x87654321U, 0.5F);
  std::vector<__half> low_half(low.size()), norm_half(norm.size());
  std::transform(low.begin(), low.end(), low_half.begin(),
                 [](float v) { return __float2half_rn(v); });
  std::transform(norm.begin(), norm.end(), norm_half.begin(),
                 [](float v) { return __float2half_rn(v); });
  std::vector<std::uint8_t> weights(kDim * (kRank / 32) * 34);
  std::uint32_t seed = 0xABCD0123U;
  for (std::size_t block = 0; block < weights.size() / 34; ++block) {
    const __half scale = __float2half_rn(
        static_cast<float>(1 + NextRandom(&seed) % 13) / 2048.0F);
    std::memcpy(weights.data() + block * 34, &scale, sizeof(scale));
    for (std::size_t j = 0; j < 32; ++j) {
      weights[block * 34 + 2 + j] = static_cast<std::uint8_t>(
          static_cast<int>(NextRandom(&seed) % 255) - 127);
    }
  }
  HipBuffer<std::uint8_t> d_up(weights.size());
  HipBuffer<__half> d_low(low.size()), d_norm(norm.size());
  HipBuffer<float> d_inject_w(kStreams * kDim);
  HipBuffer<float> d_gate(norm.size()), d_ref(mixed_count), d_out(mixed_count);
  HipBuffer<__half> d_ref_half(mixed_count), d_out_half(mixed_count);
  HipBuffer<std::uint8_t> d_ref_q8(q8_bytes), d_out_q8(q8_bytes);
  HipBuffer<float> d_ref_inject(inject_count), d_out_inject(inject_count);
  CheckHip(hipMemcpy(d_up.get(), weights.data(), d_up.bytes(),
                     hipMemcpyHostToDevice),
           "up weights");
  CheckHip(hipMemcpy(d_low.get(), low_half.data(), d_low.bytes(),
                     hipMemcpyHostToDevice),
           "low rank input");
  CheckHip(hipMemcpy(d_norm.get(), norm_half.data(), d_norm.bytes(),
                     hipMemcpyHostToDevice),
           "normalized streams");
  Upload(&d_inject_w, MakeValues(kStreams * kDim, 0xDEADBEEFU, 0.1F));
  CheckHip(hipMemset(d_ref_q8.get(), 0, q8_bytes), "clear reference Q8");
  CheckHip(hipMemset(d_out_q8.get(), 0, q8_bytes), "clear fused Q8");
  if (!q::DenseF16Gemm(d_up.get(), d_low.get(), d_gate.get(), tokens, kDim,
                       kRank, nullptr)) {
    throw std::runtime_error("HC reference projection rejected");
  }
  q::HcMixEpilogueVec4F16(d_norm.get(), d_gate.get(), d_inject_w.get(),
                          d_ref.get(), d_ref_half.get(), d_ref_q8.get(),
                          d_ref_inject.get(), tokens, kHidden, nullptr);
  // Small case covers all optional outputs; the large ragged case also
  // replays the full path to catch cross-block races and tail writes.
  const unsigned first = tokens == 96 ? 0 : 7;
  const unsigned end = tokens == 96 ? 8 : 9;
  for (unsigned iteration = first; iteration < end; ++iteration) {
    const unsigned outputs = std::min(iteration, 7U);
    if (!q::HcMixF16Gemm(d_up.get(), d_low.get(), d_norm.get(),
                         outputs & 4 ? d_inject_w.get() : nullptr, d_out.get(),
                         outputs & 1 ? d_out_half.get() : nullptr,
                         outputs & 2 ? d_out_q8.get() : nullptr,
                         outputs & 4 ? d_out_inject.get() : nullptr, tokens,
                         kHidden, kRank, nullptr)) {
      throw std::runtime_error("HC fused projection rejected");
    }
    Exact(d_ref, d_out, "F32 mixed row");
    if (outputs & 1)
      Exact(d_ref_half, d_out_half, "F16 mixed row");
    if (outputs & 2)
      Exact(d_ref_q8, d_out_q8, "Q8 mixed row");
    if (outputs & 4)
      Exact(d_ref_inject, d_out_inject, "inject partials");
  }
  if (!q::HcMixF16Gemm(d_up.get(), d_low.get(), d_norm.get(), d_inject_w.get(),
                       nullptr, d_out_half.get(), nullptr, d_out_inject.get(),
                       tokens, kHidden, kRank, nullptr))
    throw std::runtime_error("HC F16-only projection rejected");
  Exact(d_ref_half, d_out_half, "F16-only mixed row");
  Exact(d_ref_inject, d_out_inject, "F16-only inject partials");
  std::cerr << "HC projection fusion exact at " << tokens << " tokens\n";
}

// A wide prefill pass gets the FFN mixer's inject partials from the
// attention combine (HcCombineF16 with inject_w); a short pass over the same
// rows computes them in the mixer epilogue (HcMixEpilogueVec4F16, and
// HcMixF16Gemm from 96 rows). Each token's partials must not depend on which
// pass computes them, bit for bit: a 1-ulp difference changes the stream
// weights of the next combine and with them the greedy tokens.
void CheckFusedInjectPartials(std::uint32_t tokens) {
  constexpr std::size_t kRank = 320;
  constexpr std::size_t kDim = kStreams * kHidden;
  constexpr float kEps = 1e-6F;
  const std::uint32_t parts = q::HcInjectPartsVec4(kHidden);
  const std::size_t rows = std::size_t{tokens} * kDim;
  const std::size_t inject_count = std::size_t{tokens} * kStreams * parts;
  HipBuffer<float> d_res(rows), d_block(std::size_t{tokens} * kHidden);
  HipBuffer<float> d_inject(inject_count), d_gamma(kDim);
  HipBuffer<float> d_inject_w(kStreams * kDim), d_defer(tokens * kStreams);
  HipBuffer<float> d_fused(inject_count), d_epilogue(inject_count);
  HipBuffer<float> d_gemm(inject_count), d_gate(rows);
  HipBuffer<float> d_mixed(std::size_t{tokens} * kHidden);
  HipBuffer<__half> d_xn(rows);
  Upload(&d_res, MakeValues(rows, 0x51CE0001U + tokens, 0.8F));
  Upload(&d_block,
         MakeValues(std::size_t{tokens} * kHidden, 0x51CE0002U, 0.3F));
  Upload(&d_inject, MakeValues(inject_count, 0x51CE0003U, 2.0F));
  auto gamma = MakeValues(kDim, 0x51CE0004U, 0.5F);
  for (float& g : gamma)
    g += 1.0F;
  Upload(&d_gamma, gamma);
  Upload(&d_inject_w, MakeValues(kStreams * kDim, 0x51CE0005U, 0.05F));
  Upload(&d_gate, MakeValues(rows, 0x51CE0006U, 2.0F));
  if (!q::HcCombineF16(d_res.get(), d_block.get(), d_inject.get(), parts,
                       d_gamma.get(), d_xn.get(), nullptr, tokens, kHidden,
                       kStreams, kEps, nullptr, d_defer.get(), d_fused.get(),
                       d_inject_w.get()))
    throw std::runtime_error("deferring HC combine rejected");
  q::HcMixEpilogueVec4F16(d_xn.get(), d_gate.get(), d_inject_w.get(),
                          d_mixed.get(), nullptr, nullptr, d_epilogue.get(),
                          tokens, kHidden, nullptr);
  const auto fused = Download(&d_fused, inject_count);
  const auto epilogue = Download(&d_epilogue, inject_count);
  std::size_t diffs = 0;
  for (std::size_t i = 0; i < inject_count; ++i)
    diffs += std::memcmp(&fused[i], &epilogue[i], sizeof(float)) != 0;
  std::cout << "HC inject partials, fused combine vs mixer epilogue at "
            << tokens << " tokens: " << diffs << " of " << inject_count
            << " differ\n";
  if (diffs != 0)
    throw std::runtime_error("HC inject partials depend on the pass route");
  if (tokens >= 96) {
    // The fused projection's partials (HcMixF16Gemm) over the same norm.
    std::vector<std::uint8_t> weights(kDim * (kRank / 32) * 34, 0);
    HipBuffer<std::uint8_t> d_up(weights.size());
    HipBuffer<__half> d_low(std::size_t{tokens} * kRank);
    CheckHip(hipMemcpy(d_up.get(), weights.data(), d_up.bytes(),
                       hipMemcpyHostToDevice),
             "zero up weights");
    CheckHip(hipMemset(d_low.get(), 0, d_low.bytes()), "zero low rank");
    if (!q::HcMixF16Gemm(d_up.get(), d_low.get(), d_xn.get(), d_inject_w.get(),
                         d_mixed.get(), nullptr, nullptr, d_gemm.get(), tokens,
                         kHidden, kRank, nullptr))
      throw std::runtime_error("HC fused projection rejected");
    const auto gemm = Download(&d_gemm, inject_count);
    if (std::memcmp(gemm.data(), fused.data(), inject_count * sizeof(float)))
      throw std::runtime_error("HC projection inject partials differ");
    std::cout << "HC inject partials, fused projection equal at " << tokens
              << " tokens\n";
  }
}

// The BF16 GEMM shapes of the GSQ mixes at every width the mixer chains
// run (decode included) must reach the same numbers as the F16 staging of
// their exact expansion: the fetch-side rounding is the CPU's.
void CheckBf16Gemm(std::uint32_t tokens, std::size_t rows, std::size_t cols,
                   const char* label) {
  const std::size_t x_count = static_cast<std::size_t>(tokens) * cols;
  const std::size_t out_count = static_cast<std::size_t>(tokens) * rows;
  const auto norm = MakeValues(x_count, 0x87654321U, 0.5F);
  std::vector<__half> norm_half(norm.size());
  std::transform(norm.begin(), norm.end(), norm_half.begin(),
                 [](float v) { return __float2half_rn(v); });
  const auto down = MakeValues(rows * cols, 0x13572468U, 0.02F);
  std::vector<std::uint16_t> down_bf16(down.size());
  std::vector<__half> down_half(down.size());
  for (std::size_t i = 0; i < down.size(); ++i) {
    down_bf16[i] = FloatToBf16(down[i]);
    down_half[i] = __float2half_rn(Bf16ToFloat(down_bf16[i]));
  }
  constexpr std::size_t kGuard = 32;
  constexpr float kPoison = 1234.5F;
  HipBuffer<__half> d_norm(norm_half.size()), d_down_half(down_half.size());
  HipBuffer<std::uint16_t> d_down_bf16(down_bf16.size());
  HipBuffer<float> d_ref(out_count), d_out(out_count + 2 * kGuard);
  std::vector<float> poison(d_out.bytes() / sizeof(float), kPoison);
  CheckHip(hipMemcpy(d_out.get(), poison.data(), d_out.bytes(),
                     hipMemcpyHostToDevice),
           "poison BF16 projection output");
  Upload(&d_norm, norm_half);
  Upload(&d_down_half, down_half);
  Upload(&d_down_bf16, down_bf16);
  if (!q::UnquantizedF16Gemm(d_down_half.get(), d_norm.get(), d_ref.get(),
                             tokens, rows, cols, nullptr))
    throw std::runtime_error("HC BF16 reference projection rejected");
  if (!q::UnquantizedF16Gemm(d_down_bf16.get(), d_norm.get(),
                             d_out.get() + kGuard, tokens, rows, cols, nullptr,
                             q::WeightType::kBF16))
    throw std::runtime_error("HC BF16 projection rejected");
  const auto reference = Download(&d_ref, out_count);
  const auto actual = Download(&d_out, poison.size());
  if (std::memcmp(reference.data(), actual.data() + kGuard,
                  out_count * sizeof(float)) != 0) {
    throw std::runtime_error(std::string(label) + " projection changed");
  }
  for (std::size_t i = 0; i < kGuard; ++i) {
    if (actual[i] != kPoison || actual[kGuard + out_count + i] != kPoison) {
      throw std::runtime_error(std::string(label) +
                               " projection wrote outside the live rows");
    }
  }
  SpotOracle(down_half, norm_half, reference, rows, cols, tokens, label);
  // Every plan keeps one K order per output, so a token's row must stay
  // bit-identical across any prefill chunk split.
  if (tokens >= 129) {
    const std::uint32_t head = tokens - 33;
    HipBuffer<float> d_split(out_count);
    for (const auto [off, n] :
         {std::pair<std::uint32_t, std::uint32_t>{0, head},
          {head, tokens - head}}) {
      if (!q::UnquantizedF16Gemm(
              d_down_bf16.get(),
              d_norm.get() + static_cast<std::size_t>(off) * cols,
              d_split.get() + static_cast<std::size_t>(off) * rows, n, rows,
              cols, nullptr, q::WeightType::kBF16))
        throw std::runtime_error("HC BF16 split projection rejected");
    }
    const auto split = Download(&d_split, out_count);
    if (std::memcmp(split.data(), reference.data(),
                    out_count * sizeof(float)) != 0) {
      throw std::runtime_error("HC BF16 projection changed across a chunk");
    }
  }
  // The wide HC-down producer writes rows with a 32-half pad. Reading the
  // same values from the padded stride must preserve every output byte, the
  // padding must stay untouched, and the route exists only from 1,024 rows.
  if (rows < cols && cols == static_cast<std::size_t>(kStreams) * kHidden) {
    const std::size_t pitch = cols + kNormPad;
    const __half pad_poison = __float2half_rn(-4321.0F);
    std::vector<__half> padded(static_cast<std::size_t>(tokens) * pitch,
                               pad_poison);
    for (std::uint32_t t = 0; t < tokens; ++t) {
      std::copy_n(norm_half.begin() + static_cast<std::size_t>(t) * cols, cols,
                  padded.begin() + static_cast<std::size_t>(t) * pitch);
    }
    HipBuffer<__half> d_padded(padded.size());
    Upload(&d_padded, padded);
    HipBuffer<float> d_padded_out(out_count);
    const bool accepted = q::UnquantizedF16Gemm(
        d_down_bf16.get(), d_padded.get(), d_padded_out.get(), tokens, rows,
        cols, nullptr, q::WeightType::kBF16, pitch);
    if (tokens < 1024) {
      if (accepted) {
        throw std::runtime_error(
            "padded BF16 projection accepted below 1024 tokens");
      }
    } else {
      if (!accepted) {
        throw std::runtime_error("padded BF16 projection rejected");
      }
      const auto padded_out = Download(&d_padded_out, out_count);
      if (std::memcmp(padded_out.data(), reference.data(),
                      out_count * sizeof(float)) != 0) {
        throw std::runtime_error(std::string(label) +
                                 " projection changed with a padded norm");
      }
      const auto after = Download(&d_padded, padded.size());
      for (std::uint32_t t = 0; t < tokens; ++t) {
        for (std::uint32_t i = 0; i < kNormPad; ++i) {
          const __half value =
              after[static_cast<std::size_t>(t) * pitch + cols + i];
          if (std::memcmp(&value, &pad_poison, sizeof(value)) != 0) {
            throw std::runtime_error(std::string(label) +
                                     " projection wrote norm padding");
          }
        }
      }
      if (tokens >= 2048) {
        const std::uint32_t head = tokens - 1024;
        HipBuffer<float> d_padded_split(out_count);
        for (const auto [off, n] :
             {std::pair<std::uint32_t, std::uint32_t>{0, head},
              {head, 1024U}}) {
          if (!q::UnquantizedF16Gemm(
                  d_down_bf16.get(),
                  d_padded.get() + static_cast<std::size_t>(off) * pitch,
                  d_padded_split.get() + static_cast<std::size_t>(off) * rows,
                  n, rows, cols, nullptr, q::WeightType::kBF16, pitch))
            throw std::runtime_error("padded BF16 split projection rejected");
        }
        const auto padded_split = Download(&d_padded_split, out_count);
        if (std::memcmp(padded_split.data(), reference.data(),
                        out_count * sizeof(float)) != 0) {
          throw std::runtime_error(std::string(label) +
                                   " projection changed across a padded chunk");
        }
      }
    }
  }
  std::cerr << label << " projection exact at " << tokens << " tokens\n";
}

// The fused HC-down epilogue must reproduce half(silu(down * scale)) byte for
// byte: the separate projection, SiLU and narrow chain is the reference, the
// independent double formula bounds the values, and the live rows and guards
// are checked. Padding must stay untouched.
void CheckHcDownSilu(std::uint32_t tokens, bool padded, bool bf16 = true) {
  constexpr std::size_t kRows = 320;
  constexpr std::size_t kCols = static_cast<std::size_t>(kStreams) * kHidden;
  constexpr std::size_t kGuard = 32;
  constexpr float kScale = 0.25F;
  const __half kPoison = __float2half_rn(-4321.0F);
  const std::size_t pitch = padded ? kCols + kNormPad : kCols;
  const std::size_t live = static_cast<std::size_t>(tokens) * kRows;
  const auto norm =
      MakeValues(static_cast<std::size_t>(tokens) * kCols, 0x87654321U, 0.5F);
  std::vector<__half> norm_half(norm.size());
  std::transform(norm.begin(), norm.end(), norm_half.begin(),
                 [](float v) { return __float2half_rn(v); });
  const auto down = MakeValues(kRows * kCols, 0x13572468U, 0.02F);
  std::vector<std::uint16_t> down_bf16(down.size());
  std::vector<__half> down_half(down.size());
  for (std::size_t i = 0; i < down.size(); ++i) {
    down_bf16[i] = FloatToBf16(down[i]);
    down_half[i] = __float2half_rn(down[i]);
  }

  HipBuffer<__half> d_norm(static_cast<std::size_t>(tokens) * pitch);
  {
    std::vector<__half> padded_values(d_norm.bytes() / sizeof(__half), kPoison);
    for (std::uint32_t t = 0; t < tokens; ++t) {
      std::copy_n(norm_half.begin() + static_cast<std::size_t>(t) * kCols,
                  kCols,
                  padded_values.begin() + static_cast<std::size_t>(t) * pitch);
    }
    Upload(&d_norm, padded_values);
  }
  HipBuffer<std::uint16_t> d_down_bf16(down_bf16.size());
  HipBuffer<__half> d_down_half(down_half.size());
  HipBuffer<float> d_lo(live);
  HipBuffer<__half> d_ref(live + 2 * kGuard), d_fused(live + 2 * kGuard);
  std::vector<__half> poison(d_ref.bytes() / sizeof(__half), kPoison);
  CheckHip(hipMemcpy(d_ref.get(), poison.data(), d_ref.bytes(),
                     hipMemcpyHostToDevice),
           "poison HC down reference");
  CheckHip(hipMemcpy(d_fused.get(), poison.data(), d_fused.bytes(),
                     hipMemcpyHostToDevice),
           "poison HC down fused");
  Upload(&d_down_bf16, down_bf16);
  Upload(&d_down_half, down_half);
  const void* down_ptr =
      bf16 ? static_cast<const void*>(d_down_bf16.get()) : d_down_half.get();
  const auto weight_type = bf16 ? q::WeightType::kBF16 : q::WeightType::kF16;

  const std::size_t stride = padded ? pitch : 0;
  if (!q::UnquantizedF16Gemm(down_ptr, d_norm.get(), d_lo.get(), tokens, kRows,
                             kCols, nullptr, weight_type, stride)) {
    throw std::runtime_error("HC down reference rejected");
  }
  const auto down_f32 = Download(&d_lo, live);
  q::SiluScale(d_lo.get(), kScale, live, nullptr);
  q::NarrowActivations(d_lo.get(), d_ref.get() + kGuard, false, live, nullptr);
  if (!q::HcDownF16SiluGemm(down_ptr, d_norm.get(), d_fused.get() + kGuard,
                            tokens, kRows, kCols, nullptr, weight_type, stride,
                            kScale)) {
    throw std::runtime_error("fused HC down rejected");
  }
  Exact(d_ref, d_fused, "HC down SiLU");
  const auto fused = Download(&d_fused, poison.size());
  for (std::size_t i = 0; i < kGuard; ++i) {
    if (std::memcmp(&fused[i], &kPoison, sizeof(kPoison)) != 0 ||
        std::memcmp(&fused[kGuard + live + i], &kPoison, sizeof(kPoison)) !=
            0) {
      throw std::runtime_error("fused HC down wrote outside the live rows");
    }
  }
  if (padded) {
    std::vector<__half> pad(kNormPad);
    for (const std::uint32_t t : {0U, tokens - 1}) {
      CheckHip(
          hipMemcpy(pad.data(),
                    d_norm.get() + static_cast<std::size_t>(t) * pitch + kCols,
                    kNormPad * sizeof(__half), hipMemcpyDeviceToHost),
          "download HC norm padding");
      for (const __half& value : pad) {
        if (std::memcmp(&value, &kPoison, sizeof(kPoison)) != 0) {
          throw std::runtime_error("fused HC down wrote norm padding");
        }
      }
    }
  }
  // Independent double SiLU over the reference's F32 down values. The GPU's
  // exp/reciprocal are approximate, so this bounds the values rather than
  // pinning them; Exact above carries the byte-level pin.
  double worst = 0.0;
  for (std::size_t i = 0; i < live; i += std::max<std::size_t>(1, live / 997)) {
    const double scaled = static_cast<double>(down_f32[i]) * kScale;
    const double exact = scaled / (1.0 + std::exp(-scaled));
    const double actual = static_cast<double>(__half2float(fused[kGuard + i]));
    worst = std::max(worst, std::abs(actual - exact));
  }
  if (!std::isfinite(worst) || worst > 2e-3) {
    throw std::runtime_error("fused HC down SiLU differs from the double form");
  }
  std::cerr << "HC down SiLU exact at " << tokens << " tokens"
            << (padded ? " padded" : "") << "\n";
}

// The fused entry must reject exactly the shapes the separate projection
// rejects: below 96 tokens, a padded stride below 1,024 tokens, and other
// geometries.
void CheckHcDownSiluRejects() {
  constexpr std::size_t kRows = 320;
  constexpr std::size_t kCols = static_cast<std::size_t>(kStreams) * kHidden;
  HipBuffer<__half> norm(96 * kCols);
  HipBuffer<std::uint16_t> down(kRows * kCols);
  HipBuffer<__half> out(96 * kRows);
  constexpr float kScale = 0.25F;
  const auto rejected = [&](std::uint32_t tokens, std::size_t rows,
                            std::size_t cols, std::size_t stride) {
    return !q::HcDownF16SiluGemm(down.get(), norm.get(), out.get(), tokens,
                                 rows, cols, nullptr, q::WeightType::kBF16,
                                 stride, kScale);
  };
  if (!rejected(95, kRows, kCols, 0) ||
      !rejected(96, kRows, kCols, kCols + kNormPad) ||
      !rejected(96, kRows - 1, kCols, 0) ||
      !rejected(96, kRows, kCols - 32, 0)) {
    throw std::runtime_error("fused HC down accepted an unsupported shape");
  }
  std::cerr << "HC down SiLU rejects unsupported shapes\n";
}

// The decode mixer-up route must keep the wide plan's exact bytes at every
// token width and leave the bytes beyond the live rows untouched. BF16 lanes
// across F16's subnormal range and its round-to-nearest ties are included.
void CheckBf16UpNarrow() {
  constexpr std::size_t kRows = static_cast<std::size_t>(kHidden) * kStreams;
  constexpr std::size_t kCols = 320;
  constexpr std::size_t kGuard = 32;
  constexpr float kPoison = 1234.5F;
  const auto x = MakeValues(kCols * 8, 0x24681357U, 0.5F);
  std::vector<__half> x_half(x.size());
  std::transform(x.begin(), x.end(), x_half.begin(),
                 [](float v) { return __float2half_rn(v); });
  const auto base = MakeValues(kRows * kCols, 0x13572468U, 0.02F);
  std::vector<std::uint16_t> weights(base.size());
  for (std::size_t i = 0; i < weights.size(); ++i) {
    weights[i] = FloatToBf16(base[i]);
  }
  const std::uint16_t specials[] = {0x0001, 0x007F, 0x0080, 0x3380,
                                    0x33C0, 0x3C00, 0x4100, 0xC080};
  constexpr std::size_t kSpecialCount = sizeof(specials) / sizeof(specials[0]);
  for (std::size_t i = 0; i < weights.size(); i += 977) {
    weights[i] = specials[(i / 977) % kSpecialCount];
  }
  std::vector<__half> weights_half(weights.size());
  for (std::size_t i = 0; i < weights.size(); ++i) {
    weights_half[i] = __float2half_rn(Bf16ToFloat(weights[i]));
  }
  HipBuffer<std::uint16_t> d_weights_bf16(weights.size());
  HipBuffer<__half> d_weights_half(weights.size());
  HipBuffer<__half> d_x(x_half.size());
  Upload(&d_weights_bf16, weights);
  Upload(&d_weights_half, weights_half);
  Upload(&d_x, x_half);
  for (std::uint32_t tokens = 1; tokens <= 8; ++tokens) {
    const std::size_t out_count = static_cast<std::size_t>(tokens) * kRows;
    HipBuffer<float> d_ref(out_count);
    HipBuffer<float> d_out(out_count + 2 * kGuard);
    std::vector<float> poison(d_out.bytes() / sizeof(float), kPoison);
    CheckHip(hipMemcpy(d_out.get(), poison.data(), d_out.bytes(),
                       hipMemcpyHostToDevice),
             "poison narrow up output");
    if (!q::UnquantizedF16Gemm(d_weights_half.get(), d_x.get(), d_ref.get(),
                               tokens, kRows, kCols, nullptr))
      throw std::runtime_error("HC BF16 up reference rejected");
    if (!q::UnquantizedF16Gemm(d_weights_bf16.get(), d_x.get(),
                               d_out.get() + kGuard, tokens, kRows, kCols,
                               nullptr, q::WeightType::kBF16))
      throw std::runtime_error("HC BF16 up projection rejected");
    const auto reference = Download(&d_ref, out_count);
    const auto actual = Download(&d_out, poison.size());
    if (std::memcmp(reference.data(), actual.data() + kGuard,
                    out_count * sizeof(float)) != 0) {
      throw std::runtime_error("HC BF16 up changed at a narrow width");
    }
    for (std::size_t i = 0; i < kGuard; ++i) {
      if (actual[i] != kPoison || actual[kGuard + out_count + i] != kPoison) {
        throw std::runtime_error("HC BF16 up wrote outside the live rows");
      }
    }
    SpotOracle(weights_half, x_half, reference, kRows, kCols, tokens,
               "HC BF16 up narrow");
  }
  std::cerr << "HC BF16 up exact with guards at widths 1-8\n";
}

// Fusing the BF16 up and inject of the GSQ mixes must match the CPU
// expansion of those weights through the separate GEMM and F32-inject
// epilogue bit for bit.
void CheckFusedProjectionBf16(std::uint32_t tokens) {
  constexpr std::size_t kRank = 320;
  constexpr std::size_t kDim = kStreams * kHidden;
  const std::size_t mixed_count = static_cast<std::size_t>(tokens) * kHidden;
  const std::size_t inject_count = static_cast<std::size_t>(tokens) * kStreams *
                                   q::HcInjectPartsVec4(kHidden);
  const std::size_t q8_bytes = q::Q8TiledBytes(tokens, kHidden);
  const auto low =
      MakeValues(static_cast<std::size_t>(tokens) * kRank, 0x12345678U, 1.0F);
  const auto norm =
      MakeValues(static_cast<std::size_t>(tokens) * kDim, 0x87654321U, 0.5F);
  std::vector<__half> low_half(low.size()), norm_half(norm.size());
  std::transform(low.begin(), low.end(), low_half.begin(),
                 [](float v) { return __float2half_rn(v); });
  std::transform(norm.begin(), norm.end(), norm_half.begin(),
                 [](float v) { return __float2half_rn(v); });
  const auto up = MakeValues(kDim * kRank, 0x23456789U, 0.02F);
  const auto inject_w = MakeValues(kStreams * kDim, 0xDEADBEEFU, 0.1F);
  std::vector<std::uint16_t> up_bf16(up.size()), inj_bf16(inject_w.size());
  std::vector<__half> up_half(up.size());
  std::vector<float> inj_f32(inject_w.size());
  for (std::size_t i = 0; i < up.size(); ++i) {
    up_bf16[i] = FloatToBf16(up[i]);
    up_half[i] = __float2half_rn(Bf16ToFloat(up_bf16[i]));
  }
  for (std::size_t i = 0; i < inject_w.size(); ++i) {
    inj_bf16[i] = FloatToBf16(inject_w[i]);
    inj_f32[i] = Bf16ToFloat(inj_bf16[i]);
  }
  HipBuffer<__half> d_low(low_half.size()), d_norm(norm_half.size()),
      d_up_half(up_half.size());
  HipBuffer<std::uint16_t> d_up_bf16(up_bf16.size()),
      d_inj_bf16(inj_bf16.size());
  HipBuffer<float> d_inj_f32(inj_f32.size());
  HipBuffer<float> d_gate(norm.size()), d_ref(mixed_count), d_out(mixed_count);
  HipBuffer<__half> d_ref_half(mixed_count), d_out_half(mixed_count);
  HipBuffer<std::uint8_t> d_ref_q8(q8_bytes), d_out_q8(q8_bytes);
  HipBuffer<float> d_ref_inject(inject_count), d_out_inject(inject_count);
  Upload(&d_low, low_half);
  Upload(&d_norm, norm_half);
  Upload(&d_up_half, up_half);
  Upload(&d_up_bf16, up_bf16);
  Upload(&d_inj_bf16, inj_bf16);
  Upload(&d_inj_f32, inj_f32);
  CheckHip(hipMemset(d_ref_q8.get(), 0, q8_bytes), "clear reference Q8");
  CheckHip(hipMemset(d_out_q8.get(), 0, q8_bytes), "clear fused Q8");
  if (!q::UnquantizedF16Gemm(d_up_half.get(), d_low.get(), d_gate.get(), tokens,
                             kDim, kRank, nullptr))
    throw std::runtime_error("HC BF16 reference projection rejected");
  q::HcMixEpilogueVec4F16(d_norm.get(), d_gate.get(), d_inj_f32.get(),
                          d_ref.get(), d_ref_half.get(), d_ref_q8.get(),
                          d_ref_inject.get(), tokens, kHidden, nullptr);
  const unsigned first = tokens == 96 ? 0 : 7;
  const unsigned end = tokens == 96 ? 8 : 9;
  for (unsigned iteration = first; iteration < end; ++iteration) {
    const unsigned outputs = std::min(iteration, 7U);
    if (!q::HcMixF16Gemm(d_up_bf16.get(), d_low.get(), d_norm.get(),
                         outputs & 4 ? d_inj_bf16.get() : nullptr, d_out.get(),
                         outputs & 1 ? d_out_half.get() : nullptr,
                         outputs & 2 ? d_out_q8.get() : nullptr,
                         outputs & 4 ? d_out_inject.get() : nullptr, tokens,
                         kHidden, kRank, nullptr, q::WeightType::kBF16,
                         q::WeightType::kBF16))
      throw std::runtime_error("HC BF16 fused projection rejected");
    Exact(d_ref, d_out, "BF16 F32 mixed row");
    if (outputs & 1)
      Exact(d_ref_half, d_out_half, "BF16 F16 mixed row");
    if (outputs & 2)
      Exact(d_ref_q8, d_out_q8, "BF16 Q8 mixed row");
    if (outputs & 4)
      Exact(d_ref_inject, d_out_inject, "BF16 inject partials");
  }
  const auto gate = Download(&d_gate, norm.size());
  const auto mixed = Download(&d_ref, mixed_count);
  const auto inject_ref = Download(&d_ref_inject, inject_count);
  SpotOracle(up_half, low_half, gate, kDim, kRank, tokens, "HC BF16 up");
  // The CPU mix reference over the reference chain's gate and the expanded
  // weights: the mix arithmetic itself, on the tolerances an exp() allows.
  const std::uint32_t vec_parts = q::HcInjectPartsVec4(kHidden);
  double worst_mixed = 0.0;
  double worst_inject = 0.0;
  for (std::uint32_t t = 0; t < tokens; ++t) {
    for (std::uint32_t h = 0; h < kHidden; ++h) {
      double acc = 0.0;
      for (std::uint32_t s = 0; s < kStreams; ++s) {
        const std::size_t idx =
            (static_cast<std::size_t>(t) * kStreams + s) * kHidden + h;
        acc += static_cast<double>(__half2float(norm_half[idx])) /
               (1.0 + std::exp(-static_cast<double>(gate[idx])));
      }
      acc /= kStreams;
      worst_mixed = std::max(
          worst_mixed,
          std::abs(acc - mixed[static_cast<std::size_t>(t) * kHidden + h]));
    }
    for (std::uint32_t s = 0; s < kStreams; ++s) {
      double expected = 0.0;
      for (std::uint32_t i = 0; i < kDim; ++i) {
        expected +=
            static_cast<double>(
                inj_f32[static_cast<std::size_t>(s) * kDim + i]) *
            __half2float(norm_half[static_cast<std::size_t>(t) * kDim + i]);
      }
      double candidate = 0.0;
      for (std::uint32_t p = 0; p < vec_parts; ++p) {
        candidate += inject_ref[(static_cast<std::size_t>(t) * kStreams + s) *
                                    vec_parts +
                                p];
      }
      const double scale = std::max(1.0, std::abs(expected));
      worst_inject =
          std::max(worst_inject, std::abs(expected - candidate) / scale);
    }
  }
  std::cout << "HC BF16 mix vs CPU: mixed absolute error " << worst_mixed
            << ", inject relative error " << worst_inject << '\n';
  if (worst_mixed > 5e-5 || worst_inject > 1e-4) {
    throw std::runtime_error("HC BF16 mix differs from the CPU mix");
  }
  std::cerr << "HC BF16 projection fusion exact at " << tokens << " tokens\n";
}

// The BF16 inject weights of the GSQ mixes take every epilogue path, against
// the CPU expansion of those weights.
void CheckBf16Inject() {
  constexpr std::size_t kHcDim = static_cast<std::size_t>(kStreams) * kHidden;
  const std::size_t rows = static_cast<std::size_t>(kTokens) * kHcDim;
  const std::size_t mixed = static_cast<std::size_t>(kTokens) * kHidden;
  const std::uint32_t scalar_parts = q::HcInjectParts(kHidden);
  const std::uint32_t vec_parts = q::HcInjectPartsVec4(kHidden);
  const std::size_t scalar_inject =
      static_cast<std::size_t>(kTokens) * kStreams * scalar_parts;
  const std::size_t vec_inject =
      static_cast<std::size_t>(kTokens) * kStreams * vec_parts;
  const auto xn = MakeValues(rows, 0x1234ABCDU, 0.5F);
  const auto gate = MakeValues(rows, 0xBADC0FFEU, 2.0F);
  const auto inject_w = MakeValues(kStreams * kHcDim, 0xDEADBEEFU, 0.1F);
  std::vector<std::uint16_t> inj_bf16(inject_w.size());
  std::vector<float> inj_f32(inject_w.size());
  std::vector<__half> xn_half(rows);
  for (std::size_t i = 0; i < inject_w.size(); ++i) {
    inj_bf16[i] = FloatToBf16(inject_w[i]);
    inj_f32[i] = Bf16ToFloat(inj_bf16[i]);
  }
  std::transform(xn.begin(), xn.end(), xn_half.begin(),
                 [](float v) { return __float2half_rn(v); });
  HipBuffer<float> d_xn(rows), d_gate(rows), d_inj_f32(inj_f32.size());
  HipBuffer<std::uint16_t> d_inj_bf16(inj_bf16.size());
  HipBuffer<__half> d_xn_half(xn_half.size());
  HipBuffer<float> d_ref_mixed(mixed), d_mixed(mixed);
  HipBuffer<float> d_ref_scalar(scalar_inject), d_scalar(scalar_inject);
  HipBuffer<float> d_ref_vec(vec_inject), d_vec(vec_inject);
  Upload(&d_xn, xn);
  Upload(&d_gate, gate);
  Upload(&d_inj_f32, inj_f32);
  Upload(&d_inj_bf16, inj_bf16);
  Upload(&d_xn_half, xn_half);
  q::HcMixEpilogue(d_xn.get(), d_gate.get(), d_inj_f32.get(), d_ref_mixed.get(),
                   d_ref_scalar.get(), kTokens, kHidden, kStreams, nullptr);
  q::HcMixEpilogue(d_xn.get(), d_gate.get(), d_inj_bf16.get(), d_mixed.get(),
                   d_scalar.get(), kTokens, kHidden, kStreams, nullptr, nullptr,
                   q::WeightType::kBF16);
  Exact(d_ref_mixed, d_mixed, "BF16 scalar mixed row");
  Exact(d_ref_scalar, d_scalar, "BF16 scalar inject partials");
  q::HcMixEpilogueVec4(d_xn.get(), d_gate.get(), d_inj_f32.get(),
                       d_ref_mixed.get(), d_ref_vec.get(), kTokens, kHidden,
                       kStreams, nullptr);
  q::HcMixEpilogueVec4(d_xn.get(), d_gate.get(), d_inj_bf16.get(),
                       d_mixed.get(), d_vec.get(), kTokens, kHidden, kStreams,
                       nullptr, q::WeightType::kBF16);
  Exact(d_ref_mixed, d_mixed, "BF16 vec4 mixed row");
  Exact(d_ref_vec, d_vec, "BF16 vec4 inject partials");
  q::HcMixEpilogueVec4F16(d_xn_half.get(), d_gate.get(), d_inj_f32.get(),
                          d_ref_mixed.get(), nullptr, nullptr, d_ref_vec.get(),
                          kTokens, kHidden, nullptr);
  q::HcMixEpilogueVec4F16(d_xn_half.get(), d_gate.get(), d_inj_bf16.get(),
                          d_mixed.get(), nullptr, nullptr, d_vec.get(), kTokens,
                          kHidden, nullptr, q::WeightType::kBF16);
  Exact(d_ref_mixed, d_mixed, "BF16 F16 mixed row");
  Exact(d_ref_vec, d_vec, "BF16 F16 inject partials");
  std::cerr << "HC BF16 inject weights exact on every epilogue\n";
}

// A deferred wide combine given BF16 inject weights writes the next mixer's
// inject partials from its own F16-rounded norm. The fused partials must
// match the separate BF16 epilogue over that norm byte for byte, and the
// fusion must leave the deferred update, residual and norm unchanged.
void CheckCombineBf16Inject(std::uint32_t tokens) {
  constexpr std::size_t kDim = kStreams * kHidden;
  constexpr float kEps = 1e-6F;
  const std::size_t rows = static_cast<std::size_t>(tokens) * kDim;
  const std::size_t mixed = static_cast<std::size_t>(tokens) * kHidden;
  const std::uint32_t parts = q::HcInjectPartsVec4(kHidden);
  const std::size_t inject_in_count =
      static_cast<std::size_t>(tokens) * kStreams * parts;
  const auto res = MakeValues(rows, 0x0F0F0F0FU, 1.0F);
  const auto block_out = MakeValues(mixed, 0xF0F0F0F0U, 1.0F);
  const auto gamma = MakeValues(kDim, 0x600DCAFEU, 0.5F);
  const auto gate = MakeValues(rows, 0xBADC0FFEU, 2.0F);
  const auto inject_in = MakeValues(inject_in_count, 0x24680U, 0.5F);
  HipBuffer<float> d_res_plain(rows), d_res_fused(rows);
  HipBuffer<float> d_block(block_out.size()), d_gamma(gamma.size());
  HipBuffer<float> d_gate(gate.size()), d_inject_in(inject_in.size());
  HipBuffer<__half> d_norm_plain(rows), d_norm_fused(rows);
  const std::size_t pitch = kDim + kNormPad;
  const std::size_t padded_count = static_cast<std::size_t>(tokens) * pitch;
  const __half pad_poison = __float2half_rn(-4321.0F);
  const std::vector<__half> pad_fill(padded_count, pad_poison);
  HipBuffer<__half> d_norm_padded(padded_count);
  HipBuffer<float> d_res_padded(rows);
  HipBuffer<float> d_defer_padded(tokens * kStreams);
  HipBuffer<float> d_mixed_padded(mixed), d_inject_padded(inject_in_count);
  HipBuffer<float> d_defer_plain(tokens * kStreams),
      d_defer_fused(tokens * kStreams);
  HipBuffer<float> d_mixed(mixed);
  HipBuffer<float> d_inject_ref(inject_in_count),
      d_inject_fused(inject_in_count);
  Upload(&d_block, block_out);
  Upload(&d_gamma, gamma);
  Upload(&d_gate, gate);
  Upload(&d_inject_in, inject_in);
  // Two replacement weight buffers: the combine seam must use whichever
  // matrix the caller passes, with no residue of the previous one.
  for (std::uint32_t pattern = 0; pattern < 2; ++pattern) {
    const auto inject_w =
        MakeValues(kStreams * kDim, 0xDEADBEEFU + pattern, 0.1F);
    std::vector<std::uint16_t> inj_bf16(inject_w.size());
    std::vector<float> inj_f32(inject_w.size());
    for (std::size_t i = 0; i < inject_w.size(); ++i) {
      inj_bf16[i] = FloatToBf16(inject_w[i]);
      inj_f32[i] = Bf16ToFloat(inj_bf16[i]);
    }
    HipBuffer<std::uint16_t> d_inj_bf16(inj_bf16.size());
    HipBuffer<float> d_inj_f32(inj_f32.size());
    Upload(&d_inj_bf16, inj_bf16);
    Upload(&d_inj_f32, inj_f32);
    Upload(&d_res_plain, res);
    Upload(&d_res_fused, res);
    if (!q::HcCombineF16(d_res_plain.get(), d_block.get(), d_inject_in.get(),
                         parts, d_gamma.get(), d_norm_plain.get(), nullptr,
                         tokens, kHidden, kStreams, kEps, nullptr,
                         d_defer_plain.get(), nullptr, nullptr)) {
      throw std::runtime_error("plain deferred HC combine rejected");
    }
    if (!q::HcCombineF16(d_res_fused.get(), d_block.get(), d_inject_in.get(),
                         parts, d_gamma.get(), d_norm_fused.get(), nullptr,
                         tokens, kHidden, kStreams, kEps, nullptr,
                         d_defer_fused.get(), d_inject_fused.get(),
                         d_inj_bf16.get(), q::WeightType::kBF16)) {
      throw std::runtime_error("BF16 fused HC combine rejected");
    }
    Upload(&d_res_padded, res);
    Upload(&d_norm_padded, pad_fill);
    if (!q::HcCombineF16(d_res_padded.get(), d_block.get(), d_inject_in.get(),
                         parts, d_gamma.get(), d_norm_padded.get(), nullptr,
                         tokens, kHidden, kStreams, kEps, nullptr,
                         d_defer_padded.get(), nullptr, nullptr,
                         q::WeightType::kF32, pitch)) {
      throw std::runtime_error("padded HC combine rejected");
    }
    Exact(d_norm_plain, d_norm_fused, "F16 norm");
    {
      const auto padded = Download(&d_norm_padded, padded_count);
      const auto packed = Download(&d_norm_plain, rows);
      for (std::uint32_t t = 0; t < tokens; ++t) {
        if (std::memcmp(padded.data() + static_cast<std::size_t>(t) * pitch,
                        packed.data() + static_cast<std::size_t>(t) * kDim,
                        kDim * sizeof(__half)) != 0) {
          throw std::runtime_error("padded combine changed the F16 norm");
        }
        for (std::uint32_t i = 0; i < kNormPad; ++i) {
          const __half value =
              padded[static_cast<std::size_t>(t) * pitch + kDim + i];
          if (std::memcmp(&value, &pad_poison, sizeof(value)) != 0) {
            throw std::runtime_error("padded combine wrote norm padding");
          }
        }
      }
    }
    q::HcMixEpilogueVec4F16(d_norm_plain.get(), d_gate.get(), d_inj_f32.get(),
                            d_mixed.get(), nullptr, nullptr, d_inject_ref.get(),
                            tokens, kHidden, nullptr);
    Exact(d_inject_fused, d_inject_ref, "BF16 inject partials");
    q::HcMixEpilogueVec4F16(d_norm_padded.get(), d_gate.get(), d_inj_f32.get(),
                            d_mixed_padded.get(), nullptr, nullptr,
                            d_inject_padded.get(), tokens, kHidden, nullptr,
                            q::WeightType::kF32, pitch);
    Exact(d_mixed, d_mixed_padded, "padded F16 mixed row");
    Exact(d_inject_ref, d_inject_padded, "padded inject partials");
    q::HcApplyDeferred(d_res_plain.get(), d_block.get(), d_defer_plain.get(),
                       tokens, kHidden, nullptr);
    q::HcApplyDeferred(d_res_fused.get(), d_block.get(), d_defer_fused.get(),
                       tokens, kHidden, nullptr);
    q::HcApplyDeferred(d_res_padded.get(), d_block.get(), d_defer_padded.get(),
                       tokens, kHidden, nullptr);
    Exact(d_res_plain, d_res_fused, "residual");
    Exact(d_res_plain, d_res_padded, "padded residual");
    std::vector<__half> norm(rows);
    CheckHip(hipMemcpy(norm.data(), d_norm_fused.get(), d_norm_fused.bytes(),
                       hipMemcpyDeviceToHost),
             "download norm");
    const auto actual = Download(&d_inject_fused, inject_in_count);
    double worst = 0.0;
    for (std::uint32_t t = 0; t < tokens; ++t) {
      for (std::uint32_t s = 0; s < kStreams; ++s) {
        double expected = 0.0;
        for (std::uint32_t i = 0; i < kDim; ++i) {
          expected +=
              static_cast<double>(
                  inj_f32[static_cast<std::size_t>(s) * kDim + i]) *
              __half2float(norm[static_cast<std::size_t>(t) * kDim + i]);
        }
        double candidate = 0.0;
        const std::size_t base =
            (static_cast<std::size_t>(t) * kStreams + s) * parts;
        for (std::uint32_t p = 0; p < parts; ++p) {
          candidate += actual[base + p];
        }
        worst = std::max(worst, std::abs(expected - candidate) /
                                    std::max(1.0, std::abs(expected)));
      }
    }
    if (worst > 1e-4) {
      throw std::runtime_error("BF16 fused inject differs from the CPU dot");
    }
  }
  std::cerr << "HC BF16 fused combine inject exact at " << tokens
            << " tokens\n";
}

// The decode combine emits the next mixer's Q8_1 rows only when that mixer's
// down projection reads them. The plain and Q8-producing combines must agree
// bit for bit on the residual and F32 norm at every decode width, the Q8 rows
// must equal the adapter's own quantization of that norm, and a following
// BF16 down projection must produce identical bytes from either norm. With a
// null gamma the Q8 entry must reject without launching and leave every
// output untouched.
void CheckCombineQ8Demand() {
  constexpr float kEps = 1e-6F;
  constexpr std::size_t kHcDim = static_cast<std::size_t>(kHidden) * kStreams;
  constexpr std::size_t kDownRows = 320;
  const auto gamma = MakeValues(kHcDim, 0x600DCAFEU, 0.5F);
  const auto down = MakeValues(kDownRows * kHcDim, 0x13572468U, 0.02F);
  std::vector<std::uint16_t> down_bf16(down.size());
  for (std::size_t i = 0; i < down.size(); ++i)
    down_bf16[i] = FloatToBf16(down[i]);
  HipBuffer<std::uint16_t> d_down(down_bf16.size());
  HipBuffer<float> d_gamma(gamma.size());
  Upload(&d_down, down_bf16);
  Upload(&d_gamma, gamma);
  for (const std::uint32_t parts : {1U, q::HcInjectParts(kHidden)}) {
    for (std::uint32_t width = 1; width <= 8; ++width) {
      const std::size_t rows = static_cast<std::size_t>(width) * kHcDim;
      const std::size_t mixed = static_cast<std::size_t>(width) * kHidden;
      const std::size_t inject_count =
          static_cast<std::size_t>(width) * kStreams * parts;
      const auto q8_bytes =
          qfn_mmq_q8_1_bytes(static_cast<int>(width), static_cast<int>(kHcDim));
      const auto res_in = MakeValues(rows, 0x0F0F0F0FU + width, 1.0F);
      const auto block = MakeValues(mixed, 0xF0F0F0F0U + width, 1.0F);
      const auto inject = MakeValues(inject_count, 0x24680U + width, 0.5F);
      HipBuffer<float> d_block(mixed), d_inject(inject_count);
      HipBuffer<float> d_res_plain(rows), d_res_q8(rows);
      HipBuffer<float> d_xn_plain(rows), d_xn_q8(rows);
      HipBuffer<std::uint8_t> d_q8(q8_bytes + 64);
      Upload(&d_block, block);
      Upload(&d_inject, inject);
      Upload(&d_res_plain, res_in);
      Upload(&d_res_q8, res_in);
      CheckHip(hipMemset(d_q8.get(), 0xA5, d_q8.bytes()), "poison Q8 rows");
      q::HcCombine(d_res_plain.get(), d_block.get(), d_inject.get(), parts,
                   d_gamma.get(), d_xn_plain.get(), width, kHidden, kStreams,
                   kEps, nullptr);
      if (!q::HcCombineQ8_1(d_res_q8.get(), d_block.get(), d_inject.get(),
                            parts, d_gamma.get(), d_xn_q8.get(), d_q8.get(),
                            width, kHidden, kStreams, kEps, nullptr)) {
        throw std::runtime_error("Q8 combine rejected a valid width");
      }
      CheckHip(hipDeviceSynchronize(), "HC combine synchronization");
      const auto res_plain = Download(&d_res_plain, rows);
      const auto res_q8 = Download(&d_res_q8, rows);
      const auto xn_plain = Download(&d_xn_plain, rows);
      const auto xn_q8 = Download(&d_xn_q8, rows);
      if (std::memcmp(res_plain.data(), res_q8.data(), rows * sizeof(float)) !=
          0) {
        throw std::runtime_error("Q8 combine changed the residual");
      }
      if (std::memcmp(xn_plain.data(), xn_q8.data(), rows * sizeof(float)) !=
          0) {
        throw std::runtime_error("Q8 combine changed the F32 norm");
      }
      // Independent F64 reference of the residual update and grouped norm.
      double worst_res = 0.0;
      double worst_xn = 0.0;
      for (std::uint32_t t = 0; t < width; ++t) {
        for (std::uint32_t s = 0; s < kStreams; ++s) {
          double logit = 0.0;
          for (std::uint32_t p = 0; p < parts; ++p) {
            logit +=
                inject[(static_cast<std::size_t>(t) * kStreams + s) * parts +
                       p];
          }
          const double w = 2.0 / (1.0 + std::exp(-logit / kStreams));
          const std::size_t base =
              (static_cast<std::size_t>(t) * kStreams + s) * kHidden;
          std::vector<double> v(kHidden);
          double ss = 0.0;
          for (std::uint32_t i = 0; i < kHidden; ++i) {
            v[i] = static_cast<double>(res_in[base + i]) +
                   static_cast<double>(
                       block[static_cast<std::size_t>(t) * kHidden + i]) *
                       w;
            ss += v[i] * v[i];
          }
          const double scale = 1.0 / std::sqrt(ss / kHidden + kEps);
          for (std::uint32_t i = 0; i < kHidden; ++i) {
            worst_res =
                std::max(worst_res, std::abs(v[i] - res_plain[base + i]));
            const double n = v[i] * scale * gamma[s * kHidden + i];
            worst_xn = std::max(worst_xn, std::abs(n - xn_plain[base + i]) /
                                              std::max(1e-2, std::abs(n)));
          }
        }
      }
      if (worst_res > 1e-4 || worst_xn > 1e-3) {
        throw std::runtime_error("HC combine differs from the F64 reference");
      }
      // The fused Q8 rows must be the adapter's quantization of the stored
      // norm, and nothing may be written past the width's rows.
      HipBuffer<std::uint8_t> d_q8_ref(q8_bytes);
      if (qfn_mmq_quantize_q8_1(d_xn_plain.get(), d_q8_ref.get(),
                                static_cast<int>(width),
                                static_cast<int>(kHcDim), nullptr) != 0) {
        throw std::runtime_error("Q8_1 reference quantization failed");
      }
      CheckHip(hipDeviceSynchronize(), "Q8_1 reference synchronization");
      std::vector<std::uint8_t> q8(q8_bytes + 64), q8_ref(q8_bytes);
      CheckHip(
          hipMemcpy(q8.data(), d_q8.get(), q8.size(), hipMemcpyDeviceToHost),
          "download Q8 rows");
      CheckHip(hipMemcpy(q8_ref.data(), d_q8_ref.get(), q8_ref.size(),
                         hipMemcpyDeviceToHost),
               "download reference Q8 rows");
      if (std::memcmp(q8.data(), q8_ref.data(), q8_bytes) != 0) {
        throw std::runtime_error("Q8 combine differs from the adapter rows");
      }
      for (std::size_t i = q8_bytes; i < q8.size(); ++i) {
        if (q8[i] != 0xA5) {
          throw std::runtime_error("Q8 combine wrote past its rows");
        }
      }
      // The BF16 down projection at the mixer shape reads the stored norm:
      // identical norm bytes must give identical projection bytes.
      const std::size_t down_count =
          static_cast<std::size_t>(width) * kDownRows;
      HipBuffer<float> d_down_plain(down_count), d_down_q8(down_count);
      q::SmallGemm(d_down.get(), q::WeightType::kBF16, d_xn_plain.get(),
                   d_down_plain.get(), width, kDownRows, kHcDim, nullptr);
      q::SmallGemm(d_down.get(), q::WeightType::kBF16, d_xn_q8.get(),
                   d_down_q8.get(), width, kDownRows, kHcDim, nullptr);
      CheckHip(hipDeviceSynchronize(), "BF16 down synchronization");
      Exact(d_down_plain, d_down_q8, "BF16 down projection");
    }
  }
  // Null gamma: the Q8 entry launches nothing; the plain entry still updates
  // the residual and leaves xn untouched.
  {
    constexpr std::uint32_t width = 8;
    const std::size_t rows = static_cast<std::size_t>(width) * kHcDim;
    const std::size_t mixed = static_cast<std::size_t>(width) * kHidden;
    const std::uint32_t parts = q::HcInjectParts(kHidden);
    const std::size_t inject_count =
        static_cast<std::size_t>(width) * kStreams * parts;
    const auto q8_bytes =
        qfn_mmq_q8_1_bytes(static_cast<int>(width), static_cast<int>(kHcDim));
    const auto res_in = MakeValues(rows, 0xABCDEF01U, 1.0F);
    const auto block = MakeValues(mixed, 0x10203040U, 1.0F);
    const auto inject = MakeValues(inject_count, 0x55667788U, 0.5F);
    HipBuffer<float> d_block(mixed), d_inject(inject_count);
    HipBuffer<float> d_res_plain(rows), d_res_q8(rows);
    HipBuffer<float> d_xn_plain(rows), d_xn_q8(rows);
    HipBuffer<std::uint8_t> d_q8(q8_bytes);
    Upload(&d_block, block);
    Upload(&d_inject, inject);
    Upload(&d_res_plain, res_in);
    CheckHip(hipMemset(d_res_q8.get(), 0xA5, d_res_q8.bytes()),
             "poison null-gamma residual");
    CheckHip(hipMemset(d_xn_plain.get(), 0xA5, d_xn_plain.bytes()),
             "poison null-gamma plain norm");
    CheckHip(hipMemset(d_xn_q8.get(), 0xA5, d_xn_q8.bytes()),
             "poison null-gamma Q8 norm");
    CheckHip(hipMemset(d_q8.get(), 0xA5, d_q8.bytes()),
             "poison null-gamma Q8 rows");
    if (q::HcCombineQ8_1(d_res_q8.get(), d_block.get(), d_inject.get(), parts,
                         nullptr, d_xn_q8.get(), d_q8.get(), width, kHidden,
                         kStreams, kEps, nullptr)) {
      throw std::runtime_error("Q8 combine accepted a null gamma");
    }
    q::HcCombine(d_res_plain.get(), d_block.get(), d_inject.get(), parts,
                 nullptr, d_xn_plain.get(), width, kHidden, kStreams, kEps,
                 nullptr);
    CheckHip(hipDeviceSynchronize(), "null-gamma combine synchronization");
    const auto untouched = [&](const void* data, std::size_t bytes,
                               const char* label) {
      std::vector<std::uint8_t> actual(bytes, 0);
      CheckHip(hipMemcpy(actual.data(), data, bytes, hipMemcpyDeviceToHost),
               label);
      for (const auto byte : actual) {
        if (byte != 0xA5) {
          throw std::runtime_error(std::string(label) + " changed");
        }
      }
    };
    untouched(d_res_q8.get(), d_res_q8.bytes(), "null-gamma residual");
    untouched(d_xn_q8.get(), d_xn_q8.bytes(), "null-gamma Q8 norm");
    untouched(d_q8.get(), d_q8.bytes(), "null-gamma Q8 rows");
    untouched(d_xn_plain.get(), d_xn_plain.bytes(), "null-gamma plain norm");
    const auto res_plain = Download(&d_res_plain, rows);
    double worst = 0.0;
    for (std::uint32_t t = 0; t < width; ++t) {
      for (std::uint32_t s = 0; s < kStreams; ++s) {
        double logit = 0.0;
        for (std::uint32_t p = 0; p < parts; ++p) {
          logit +=
              inject[(static_cast<std::size_t>(t) * kStreams + s) * parts + p];
        }
        const double w = 2.0 / (1.0 + std::exp(-logit / kStreams));
        const std::size_t base =
            (static_cast<std::size_t>(t) * kStreams + s) * kHidden;
        for (std::uint32_t i = 0; i < kHidden; ++i) {
          const double expected =
              static_cast<double>(res_in[base + i]) +
              static_cast<double>(
                  block[static_cast<std::size_t>(t) * kHidden + i]) *
                  w;
          worst = std::max(worst, std::abs(expected - res_plain[base + i]));
        }
      }
    }
    if (worst > 1e-4) {
      throw std::runtime_error("null-gamma combine changed the residual");
    }
  }
  std::cerr << "HC combine Q8 demand exact at widths 1-8\n";
}

// The wide BF16 HC chain: combine writes the next mixer's F16 norm, the down
// projection reads it, and the up projection's fused epilogue rereads it.
// A padded norm row must reproduce the packed chain byte for byte without
// writing the padding; the deferred residual and stream weights must match.
void CheckPaddedChain(std::uint32_t tokens, bool deferred) {
  constexpr std::size_t kRank = 320;
  constexpr std::size_t kDim = kStreams * kHidden;
  const std::size_t pitch = kDim + kNormPad;
  const std::size_t rows = static_cast<std::size_t>(tokens) * kDim;
  const std::size_t low_count = static_cast<std::size_t>(tokens) * kRank;
  const std::size_t mixed_count = static_cast<std::size_t>(tokens) * kHidden;
  const std::uint32_t parts = q::HcInjectPartsVec4(kHidden);
  const std::size_t inject_count =
      static_cast<std::size_t>(tokens) * kStreams * parts;
  const std::size_t q8_bytes = q::Q8TiledBytes(tokens, kHidden);
  const auto down = MakeValues(kRank * kDim, 0x13572468U, 0.02F);
  const auto up = MakeValues(kDim * kRank, 0x23456789U, 0.02F);
  const auto inject_w = MakeValues(kStreams * kDim, 0xDEADBEEFU, 0.1F);
  std::vector<std::uint16_t> down_bf16(down.size()), up_bf16(up.size()),
      inj_bf16(inject_w.size());
  for (std::size_t i = 0; i < down.size(); ++i)
    down_bf16[i] = FloatToBf16(down[i]);
  for (std::size_t i = 0; i < up.size(); ++i)
    up_bf16[i] = FloatToBf16(up[i]);
  for (std::size_t i = 0; i < inject_w.size(); ++i)
    inj_bf16[i] = FloatToBf16(inject_w[i]);
  const auto residual = MakeValues(rows, 0x0F0F0F0FU, 1.0F);
  const auto block_out = MakeValues(mixed_count, 0xF0F0F0F0U, 1.0F);
  const auto gamma = MakeValues(kDim, 0x600DCAFEU, 0.5F);
  const auto inject_in = MakeValues(inject_count, 0x24680U, 0.5F);
  const __half pad_poison = __float2half_rn(-4321.0F);
  const std::vector<__half> pad_fill(static_cast<std::size_t>(tokens) * pitch,
                                     pad_poison);
  HipBuffer<std::uint16_t> d_down(down_bf16.size()), d_up(up_bf16.size()),
      d_inj(inj_bf16.size());
  HipBuffer<float> d_block(block_out.size()), d_gamma(gamma.size()),
      d_inject_in(inject_in.size());
  HipBuffer<float> d_res_packed(rows), d_res_padded(rows);
  HipBuffer<__half> d_norm_packed(rows), d_norm_padded(pad_fill.size());
  HipBuffer<float> d_defer_packed(tokens * kStreams),
      d_defer_padded(tokens * kStreams);
  HipBuffer<float> d_lo_packed(low_count), d_lo_padded(low_count);
  HipBuffer<__half> d_gate_packed(low_count), d_gate_padded(low_count);
  HipBuffer<float> d_mixed_packed(mixed_count), d_mixed_padded(mixed_count);
  HipBuffer<__half> d_half_packed(mixed_count), d_half_padded(mixed_count);
  HipBuffer<std::uint8_t> d_q8_packed(q8_bytes), d_q8_padded(q8_bytes);
  HipBuffer<float> d_inject_packed(inject_count), d_inject_padded(inject_count);
  Upload(&d_down, down_bf16);
  Upload(&d_up, up_bf16);
  Upload(&d_inj, inj_bf16);
  Upload(&d_block, block_out);
  Upload(&d_gamma, gamma);
  Upload(&d_inject_in, inject_in);
  Upload(&d_norm_padded, pad_fill);
  auto combine = [&](HipBuffer<float>* res, HipBuffer<__half>* norm,
                     HipBuffer<float>* defer, std::uint32_t stride) {
    const bool deferred_result = q::HcCombineF16(
        res->get(), d_block.get(), d_inject_in.get(), parts, d_gamma.get(),
        norm->get(), nullptr, tokens, kHidden, kStreams, 1e-6F, nullptr,
        deferred ? defer->get() : nullptr, nullptr, nullptr,
        q::WeightType::kF32, stride);
    return !deferred || deferred_result;
  };
  auto down_projection = [&](HipBuffer<__half>* norm, HipBuffer<float>* lo,
                             std::uint32_t stride) {
    return q::UnquantizedF16Gemm(d_down.get(), norm->get(), lo->get(), tokens,
                                 kRank, kDim, nullptr, q::WeightType::kBF16,
                                 stride);
  };
  auto silu_narrow = [&](HipBuffer<float>* lo, HipBuffer<__half>* gate) {
    q::SiluScale(lo->get(), 1.0F / static_cast<float>(kStreams),
                 static_cast<std::size_t>(tokens) * kRank, nullptr);
    q::NarrowActivations(lo->get(), gate->get(), false,
                         static_cast<std::size_t>(tokens) * kRank, nullptr);
  };
  auto up_projection = [&](HipBuffer<__half>* gate, HipBuffer<__half>* norm,
                           HipBuffer<float>* mixed, HipBuffer<__half>* half,
                           HipBuffer<std::uint8_t>* q8,
                           HipBuffer<float>* inject, std::uint32_t stride) {
    return q::HcMixF16Gemm(d_up.get(), gate->get(), norm->get(), d_inj.get(),
                           mixed->get(), half->get(), q8->get(), inject->get(),
                           tokens, kHidden, kRank, nullptr,
                           q::WeightType::kBF16, q::WeightType::kBF16, stride);
  };
  Upload(&d_res_packed, residual);
  if (!combine(&d_res_packed, &d_norm_packed, &d_defer_packed, 0))
    throw std::runtime_error("packed HC combine rejected");
  if (!down_projection(&d_norm_packed, &d_lo_packed, 0))
    throw std::runtime_error("packed HC down rejected");
  silu_narrow(&d_lo_packed, &d_gate_packed);
  if (!up_projection(&d_gate_packed, &d_norm_packed, &d_mixed_packed,
                     &d_half_packed, &d_q8_packed, &d_inject_packed, 0))
    throw std::runtime_error("packed HC up rejected");
  Upload(&d_res_padded, residual);
  if (!combine(&d_res_padded, &d_norm_padded, &d_defer_padded, pitch))
    throw std::runtime_error("padded HC combine rejected");
  if (!down_projection(&d_norm_padded, &d_lo_padded, pitch))
    throw std::runtime_error("padded HC down rejected");
  silu_narrow(&d_lo_padded, &d_gate_padded);
  if (!up_projection(&d_gate_padded, &d_norm_padded, &d_mixed_padded,
                     &d_half_padded, &d_q8_padded, &d_inject_padded, pitch))
    throw std::runtime_error("padded HC up rejected");
  if (deferred) {
    q::HcApplyDeferred(d_res_packed.get(), d_block.get(), d_defer_packed.get(),
                       tokens, kHidden, nullptr);
    q::HcApplyDeferred(d_res_padded.get(), d_block.get(), d_defer_padded.get(),
                       tokens, kHidden, nullptr);
  }
  const auto norm_packed = Download(&d_norm_packed, rows);
  const auto norm_padded = Download(&d_norm_padded, pad_fill.size());
  for (std::uint32_t t = 0; t < tokens; ++t) {
    if (std::memcmp(norm_padded.data() + static_cast<std::size_t>(t) * pitch,
                    norm_packed.data() + static_cast<std::size_t>(t) * kDim,
                    kDim * sizeof(__half)) != 0) {
      throw std::runtime_error("padded chain changed the norm");
    }
    for (std::uint32_t i = 0; i < kNormPad; ++i) {
      const __half value =
          norm_padded[static_cast<std::size_t>(t) * pitch + kDim + i];
      if (std::memcmp(&value, &pad_poison, sizeof(value)) != 0) {
        throw std::runtime_error("padded chain wrote norm padding");
      }
    }
  }
  Exact(d_lo_packed, d_lo_padded, "padded HC down output");
  Exact(d_gate_packed, d_gate_padded, "padded HC SiLU/narrow");
  Exact(d_mixed_packed, d_mixed_padded, "padded HC mixed row");
  Exact(d_half_packed, d_half_padded, "padded HC mixed F16 row");
  Exact(d_q8_packed, d_q8_padded, "padded HC mixed Q8 row");
  Exact(d_inject_packed, d_inject_padded, "padded HC inject partials");
  Exact(d_res_packed, d_res_padded, "padded HC residual");
  if (deferred)
    Exact(d_defer_packed, d_defer_padded, "padded HC stream weights");
  std::cerr << "HC padded chain exact at " << tokens << " tokens ("
            << (deferred ? "deferred" : "immediate") << ")\n";
}

// The decode 16-bit GEMV tier (vectorized row waves over F32 activations):
// every encoding follows the F64 oracle, and an output row must not depend
// on the token count, which is the batched-decode contract.
void CheckDecodeGemv(std::uint32_t tokens, std::size_t rows, std::size_t cols,
                     const char* label) {
  const std::size_t x_count = static_cast<std::size_t>(tokens) * cols;
  const std::size_t out_count = static_cast<std::size_t>(tokens) * rows;
  const auto x = MakeValues(x_count, 0x12345678U, 1.0F);
  const auto w = MakeValues(rows * cols, 0x13572468U, 0.02F);
  std::vector<std::uint16_t> w_bf16(w.size());
  std::vector<__half> w_half(w.size());
  std::vector<float> w_bf16_exact(w.size()), w_half_exact(w.size());
  for (std::size_t i = 0; i < w.size(); ++i) {
    w_bf16[i] = FloatToBf16(w[i]);
    w_half[i] = __float2half_rn(w[i]);
    w_bf16_exact[i] = Bf16ToFloat(w_bf16[i]);
    w_half_exact[i] = __half2float(w_half[i]);
  }
  HipBuffer<std::uint16_t> d_bf16(w_bf16.size());
  HipBuffer<__half> d_half(w_half.size());
  HipBuffer<float> d_x(x.size()), d_out(out_count), d_one(rows);
  Upload(&d_bf16, w_bf16);
  Upload(&d_half, w_half);
  Upload(&d_x, x);
  for (int encoding = 0; encoding < 2; ++encoding) {
    const auto& exact = encoding == 0 ? w_bf16_exact : w_half_exact;
    const void* w_ptr = encoding == 0 ? static_cast<const void*>(d_bf16.get())
                                      : static_cast<const void*>(d_half.get());
    const q::WeightType type =
        encoding == 0 ? q::WeightType::kBF16 : q::WeightType::kF16;
    q::SmallGemm(w_ptr, type, d_x.get(), d_out.get(), tokens, rows, cols,
                 nullptr);
    const auto out = Download(&d_out, out_count);
    double scale = 0.0;
    double worst = 0.0;
    const std::size_t sample_rows[4] = {0, rows / 3, 2 * rows / 3, rows - 1};
    for (std::uint32_t t = 0; t < tokens; t += std::max(1U, tokens / 4)) {
      for (std::size_t r : sample_rows) {
        double expected = 0.0;
        for (std::size_t i = 0; i < cols; ++i) {
          expected += static_cast<double>(exact[r * cols + i]) *
                      x[static_cast<std::size_t>(t) * cols + i];
        }
        const double actual = out[static_cast<std::size_t>(t) * rows + r];
        scale = std::max(scale, std::abs(expected));
        worst = std::max(worst, std::abs(expected - actual));
      }
    }
    if (!std::isfinite(worst) || worst > 1e-4 * std::max(1.0, scale)) {
      throw std::runtime_error(std::string(label) +
                               " differs from the F64 oracle");
    }
    // The same row at one token and at a full decode batch: separate
    // templates, one reduction.
    q::SmallGemm(w_ptr, type, d_x.get(), d_one.get(), 1, rows, cols, nullptr);
    const auto one = Download(&d_one, rows);
    if (!std::equal(one.begin(), one.end(), out.begin())) {
      throw std::runtime_error(std::string(label) +
                               " row depends on the token count");
    }
  }
  std::cerr << label << " GEMV exact at " << tokens << " tokens\n";
}

}  // namespace

int main() {
  try {
    CheckFusedProjection(96);
    CheckFusedProjection(2049);
    for (std::uint32_t n : {7U, 96U, 1619U})
      CheckFusedInjectPartials(n);
    constexpr std::size_t kRank = 320;
    constexpr std::size_t kDim = kStreams * kHidden;
    for (const std::uint32_t tokens : {1U, 8U, 96U, 127U, 128U, 129U, 1023U,
                                       1024U, 1025U, 2047U, 2048U, 2049U})
      CheckBf16Gemm(tokens, kRank, kDim, "HC BF16 down");
    for (const std::uint32_t tokens : {96U, 1023U, 1024U, 1025U, 2049U}) {
      CheckHcDownSilu(tokens, false);
      if (tokens >= 1024) {
        CheckHcDownSilu(tokens, true);
      }
    }
    CheckHcDownSilu(96, false, false);
    CheckHcDownSilu(1024, false, false);
    CheckHcDownSiluRejects();
    for (std::uint32_t tokens = 1; tokens <= 8; ++tokens)
      CheckBf16Gemm(tokens, kDim, kRank, "HC BF16 up");
    CheckBf16Gemm(96, kDim, kRank, "HC BF16 up");
    CheckBf16Gemm(2049, kDim, kRank, "HC BF16 up");
    CheckBf16UpNarrow();
    CheckDecodeGemv(1, kRank, kDim, "decode HC down");
    CheckDecodeGemv(8, kRank, kDim, "decode HC down");
    CheckDecodeGemv(8, kDim, kRank, "decode HC up");
    CheckDecodeGemv(8, 512, kHidden, "decode router");
    CheckDecodeGemv(3, 96, 344, "decode tail-K");
    CheckFusedProjectionBf16(96);
    CheckFusedProjectionBf16(2049);
    CheckBf16Inject();
    CheckCombineBf16Inject(96);
    CheckCombineBf16Inject(2049);
    CheckCombineQ8Demand();
    CheckPaddedChain(1024, false);
    CheckPaddedChain(1025, true);
    CheckPaddedChain(2049, true);
    constexpr std::size_t kHcDim = static_cast<std::size_t>(kHidden) * kStreams;
    constexpr std::size_t kRows = static_cast<std::size_t>(kTokens) * kHcDim;
    constexpr std::size_t kMixed = static_cast<std::size_t>(kTokens) * kHidden;
    const std::uint32_t scalar_parts = q::HcInjectParts(kHidden);
    const std::uint32_t vec_parts = q::HcInjectPartsVec4(kHidden);
    const std::size_t scalar_inject =
        static_cast<std::size_t>(kTokens) * kStreams * scalar_parts;
    const std::size_t vec_inject =
        static_cast<std::size_t>(kTokens) * kStreams * vec_parts;

    const auto xn = MakeValues(kRows, 0x1234ABCDU, 0.5F);
    const auto gate = MakeValues(kRows, 0xBADC0FFEU, 2.0F);
    const auto inject_w = MakeValues(kStreams * kHcDim, 0xDEADBEEFU, 0.1F);
    HipBuffer<float> d_xn(kRows);
    HipBuffer<float> d_gate(kRows);
    HipBuffer<float> d_weight(kStreams * kHcDim);
    HipBuffer<float> d_mixed_scalar(kMixed);
    HipBuffer<float> d_mixed_vec(kMixed);
    HipBuffer<float> d_inject_scalar(scalar_inject);
    HipBuffer<float> d_inject_vec(vec_inject);
    Upload(&d_xn, xn);
    Upload(&d_gate, gate);
    Upload(&d_weight, inject_w);

    q::HcMixEpilogue(d_xn.get(), d_gate.get(), d_weight.get(),
                     d_mixed_scalar.get(), d_inject_scalar.get(), kTokens,
                     kHidden, kStreams, nullptr);
    q::HcMixEpilogueVec4(d_xn.get(), d_gate.get(), d_weight.get(),
                         d_mixed_vec.get(), d_inject_vec.get(), kTokens,
                         kHidden, kStreams, nullptr);
    CheckHip(hipDeviceSynchronize(), "HC mix synchronization");

    const auto mixed_scalar = Download(&d_mixed_scalar, kMixed);
    const auto mixed_vec = Download(&d_mixed_vec, kMixed);
    const auto inject_scalar = Download(&d_inject_scalar, scalar_inject);
    const auto inject_vec = Download(&d_inject_vec, vec_inject);
    double worst_mixed = 0.0;
    for (std::size_t i = 0; i < kMixed; ++i) {
      worst_mixed = std::max(
          worst_mixed,
          std::abs(static_cast<double>(mixed_scalar[i] - mixed_vec[i])));
    }
    double worst_inject = 0.0;
    for (std::uint32_t t = 0; t < kTokens; ++t) {
      for (std::uint32_t s = 0; s < kStreams; ++s) {
        double scalar = 0.0;
        double vectorized = 0.0;
        const std::size_t scalar_base =
            (static_cast<std::size_t>(t) * kStreams + s) * scalar_parts;
        const std::size_t vec_base =
            (static_cast<std::size_t>(t) * kStreams + s) * vec_parts;
        for (std::uint32_t p = 0; p < scalar_parts; ++p) {
          scalar += inject_scalar[scalar_base + p];
        }
        for (std::uint32_t p = 0; p < vec_parts; ++p) {
          vectorized += inject_vec[vec_base + p];
        }
        const double scale = std::max(1.0, std::abs(scalar));
        worst_inject =
            std::max(worst_inject, std::abs(scalar - vectorized) / scale);
      }
    }
    std::cerr << "HC mix vec4 worst mixed absolute error " << worst_mixed
              << ", inject relative error " << worst_inject << std::endl;
    if (worst_mixed > 1e-6 || worst_inject > 5e-5) {
      return 1;
    }

    // F16 mixer input route: the combine writes the next mixer's norm as
    // F16 and the epilogue reads it back. The residual update must stay
    // bit-identical; the norm and the mixed output carry F16 rounding.
    constexpr float kEps = 1e-6F;
    const auto res = MakeValues(kRows, 0x0F0F0F0FU, 1.0F);
    const auto block_out = MakeValues(kMixed, 0xF0F0F0F0U, 1.0F);
    const auto gamma = MakeValues(kHcDim, 0x600DCAFEU, 0.5F);
    HipBuffer<float> d_res_ref(kRows);
    HipBuffer<float> d_res_f16(kRows);
    HipBuffer<float> d_block_out(kMixed);
    HipBuffer<float> d_gamma(kHcDim);
    HipBuffer<float> d_xn_ref(kRows);
    HipBuffer<__half> d_xn_f16(kRows);
    HipBuffer<float> d_inject_ref(vec_inject);
    Upload(&d_res_ref, res);
    Upload(&d_res_f16, res);
    Upload(&d_block_out, block_out);
    Upload(&d_gamma, gamma);
    q::HcCombine(d_res_ref.get(), d_block_out.get(), d_inject_vec.get(),
                 vec_parts, d_gamma.get(), d_xn_ref.get(), kTokens, kHidden,
                 kStreams, kEps, nullptr);
    const std::size_t q8_bytes = q::Q8TiledBytes(kTokens, kHcDim);
    HipBuffer<std::uint8_t> d_xn_q8(q8_bytes);
    q::HcCombineF16(d_res_f16.get(), d_block_out.get(), d_inject_vec.get(),
                    vec_parts, d_gamma.get(), d_xn_f16.get(), d_xn_q8.get(),
                    kTokens, kHidden, kStreams, kEps, nullptr);
    q::HcMixEpilogueVec4(d_xn_ref.get(), d_gate.get(), d_weight.get(),
                         d_mixed_scalar.get(), d_inject_ref.get(), kTokens,
                         kHidden, kStreams, nullptr);
    q::HcMixEpilogueVec4F16(d_xn_f16.get(), d_gate.get(), d_weight.get(),
                            d_mixed_vec.get(), nullptr, nullptr,
                            d_inject_vec.get(), kTokens, kHidden, nullptr);
    CheckHip(hipDeviceSynchronize(), "HC F16 synchronization");
    const auto res_ref = Download(&d_res_ref, kRows);
    const auto res_f16 = Download(&d_res_f16, kRows);
    const auto xn_ref = Download(&d_xn_ref, kRows);
    // CPU reference of the combine (residual update and grouped norm) over
    // the same inject partials.
    {
      double worst_cpu_res = 0.0;
      double worst_cpu_xn = 0.0;
      for (std::uint32_t t = 0; t < kTokens; ++t) {
        for (std::uint32_t s = 0; s < kStreams; ++s) {
          double logit = 0.0;
          for (std::uint32_t p = 0; p < vec_parts; ++p) {
            logit += inject_vec[(static_cast<std::size_t>(t) * kStreams + s) *
                                    vec_parts +
                                p];
          }
          const double w =
              2.0 / (1.0 + std::exp(-logit / static_cast<double>(kStreams)));
          const std::size_t base =
              (static_cast<std::size_t>(t) * kStreams + s) * kHidden;
          std::vector<double> v(kHidden);
          double ss = 0.0;
          for (std::uint32_t i = 0; i < kHidden; ++i) {
            v[i] = res[base + i] +
                   block_out[static_cast<std::size_t>(t) * kHidden + i] * w;
            ss += v[i] * v[i];
          }
          const double scale = 1.0 / std::sqrt(ss / kHidden + kEps);
          for (std::uint32_t i = 0; i < kHidden; ++i) {
            worst_cpu_res =
                std::max(worst_cpu_res, std::abs(v[i] - res_ref[base + i]));
            const double n = v[i] * scale * gamma[s * kHidden + i];
            worst_cpu_xn =
                std::max(worst_cpu_xn, std::abs(n - xn_ref[base + i]) /
                                           std::max(1e-2, std::abs(n)));
          }
        }
      }
      std::cerr << "HC combine vs CPU: residual " << worst_cpu_res << ", norm "
                << worst_cpu_xn << std::endl;
      if (worst_cpu_res > 1e-4 || worst_cpu_xn > 1e-3) {
        return 1;
      }
    }
    std::vector<__half> xn_f16(kRows);
    CheckHip(hipMemcpy(xn_f16.data(), d_xn_f16.get(), kRows * sizeof(__half),
                       hipMemcpyDeviceToHost),
             "download");
    const auto mixed_ref = Download(&d_mixed_scalar, kMixed);
    const auto mixed_f16 = Download(&d_mixed_vec, kMixed);
    const auto inject_ref = Download(&d_inject_ref, vec_inject);
    const auto inject_f16 = Download(&d_inject_vec, vec_inject);
    double worst_res = 0.0;
    double worst_xn = 0.0;
    for (std::size_t i = 0; i < kRows; ++i) {
      worst_res = std::max(
          worst_res, std::abs(static_cast<double>(res_ref[i] - res_f16[i])));
      const double scale =
          std::max(1e-2, std::abs(static_cast<double>(xn_ref[i])));
      worst_xn = std::max(
          worst_xn,
          std::abs(static_cast<double>(xn_ref[i] - __half2float(xn_f16[i]))) /
              scale);
    }
    // The mix sums four streams of either sign, so it is scored absolutely
    // on values of order one.
    double worst_mixed_f16 = 0.0;
    for (std::size_t i = 0; i < kMixed; ++i) {
      worst_mixed_f16 =
          std::max(worst_mixed_f16,
                   std::abs(static_cast<double>(mixed_ref[i] - mixed_f16[i])));
    }
    double worst_inject_f16 = 0.0;
    for (std::uint32_t t = 0; t < kTokens; ++t) {
      for (std::uint32_t s = 0; s < kStreams; ++s) {
        double reference = 0.0;
        double candidate = 0.0;
        const std::size_t base =
            (static_cast<std::size_t>(t) * kStreams + s) * vec_parts;
        for (std::uint32_t p = 0; p < vec_parts; ++p) {
          reference += inject_ref[base + p];
          candidate += inject_f16[base + p];
        }
        const double scale = std::max(1.0, std::abs(reference));
        worst_inject_f16 =
            std::max(worst_inject_f16, std::abs(reference - candidate) / scale);
      }
    }
    // The fused tiled-Q8 norm: dequantize each 32-block back and compare
    // with the F32 norm (8-bit rounding of the block's absmax).
    std::vector<std::uint8_t> xn_q8(q8_bytes);
    CheckHip(
        hipMemcpy(xn_q8.data(), d_xn_q8.get(), q8_bytes, hipMemcpyDeviceToHost),
        "download");
    double worst_q8 = 0.0;
    for (std::uint32_t t = 0; t < kTokens; ++t) {
      for (std::size_t kb = 0; kb < kHcDim / 32; ++kb) {
        const std::uint8_t* tile =
            xn_q8.data() + (((t / 16) * (kHcDim / 32)) + kb) * 576;
        float d = 0.0F;
        std::memcpy(&d, tile + 512 + (t % 16) * 4, 4);
        for (std::size_t j = 0; j < 32; ++j) {
          const auto code = static_cast<std::int8_t>(
              tile[((j / 16) * 256) + ((t % 16) * 16) + (j % 16)]);
          const double ref =
              xn_ref[static_cast<std::size_t>(t) * kHcDim + kb * 32 + j];
          const double err = std::abs(ref - d * static_cast<double>(code));
          // One rounding step of the block's absmax / 127 grid.
          worst_q8 = std::max(
              worst_q8, err / std::max(1e-3, static_cast<double>(std::abs(d))));
        }
      }
    }
    std::cout << "HC F16 mixer input: residual absolute error " << worst_res
              << ", tiled Q8 norm worst error " << worst_q8
              << " quantization steps\n";
    std::cout << "HC F16 mixer input: residual absolute error " << worst_res
              << ", norm relative error " << worst_xn
              << ", mixed absolute error " << worst_mixed_f16
              << ", inject relative error " << worst_inject_f16 << '\n';
    if (worst_res != 0.0 || worst_xn > 2e-3 || worst_mixed_f16 > 1e-3 ||
        worst_inject_f16 > 4e-3 || worst_q8 > 0.51) {
      return 1;
    }

    // Side outputs are independent: Q8 must also work without an F16
    // output or inject weights, including a one-token batch.
    for (const auto count : {1U, kTokens}) {
      const std::size_t mixed_count = static_cast<std::size_t>(count) * kHidden;
      const auto bytes = q::Q8TiledBytes(count, kHidden);
      HipBuffer<__half> d_half(mixed_count);
      HipBuffer<std::uint8_t> d_q8(bytes);
      for (unsigned outputs = 0; outputs < 8; ++outputs) {
        const bool half_output = (outputs & 1U) != 0;
        const bool q8_output = (outputs & 2U) != 0;
        const bool inject_output = (outputs & 4U) != 0;
        CheckHip(hipMemset(d_q8.get(), 0xA5, bytes), "poison Q8 output");
        q::HcMixEpilogueVec4F16(d_xn_f16.get(), d_gate.get(),
                                inject_output ? d_weight.get() : nullptr,
                                d_mixed_vec.get(),
                                half_output ? d_half.get() : nullptr,
                                q8_output ? d_q8.get() : nullptr,
                                inject_output ? d_inject_vec.get() : nullptr,
                                count, kHidden, nullptr);
        const auto actual = Download(&d_mixed_vec, kMixed);
        if (std::memcmp(actual.data(), mixed_f16.data(),
                        mixed_count * sizeof(float)) != 0) {
          throw std::runtime_error("optional outputs changed the mixed row");
        }
        if (inject_output) {
          const auto actual_inject = Download(&d_inject_vec, vec_inject);
          if (std::memcmp(actual_inject.data(), inject_f16.data(),
                          static_cast<std::size_t>(count) * kStreams *
                              vec_parts * sizeof(float)) != 0) {
            throw std::runtime_error("mixer replay changed inject partials");
          }
        }
        if (half_output) {
          std::vector<__half> half(mixed_count);
          CheckHip(hipMemcpy(half.data(), d_half.get(), d_half.bytes(),
                             hipMemcpyDeviceToHost),
                   "download mixed F16");
          for (std::size_t i = 0; i < mixed_count; ++i) {
            if (__half2float(half[i]) !=
                __half2float(__float2half(mixed_f16[i]))) {
              throw std::runtime_error("mixed F16 rounding differs");
            }
          }
        }
        if (q8_output) {
          std::vector<std::uint8_t> packed(bytes);
          CheckHip(hipMemcpy(packed.data(), d_q8.get(), bytes,
                             hipMemcpyDeviceToHost),
                   "download mixed Q8");
          for (std::uint32_t t = 0; t < count; ++t) {
            for (std::uint32_t kb = 0; kb < kHidden / 32; ++kb) {
              const auto* tile =
                  packed.data() + ((t / 16) * (kHidden / 32) + kb) * 576;
              float scale;
              std::memcpy(&scale, tile + 512 + (t % 16) * 4, 4);
              if (!std::isfinite(scale) || scale < 0.0F) {
                throw std::runtime_error("invalid mixed Q8 scale");
              }
              for (std::uint32_t j = 0; j < 32; ++j) {
                const auto code = static_cast<std::int8_t>(
                    tile[(j / 16) * 256 + (t % 16) * 16 + j % 16]);
                const float expected =
                    mixed_f16[static_cast<std::size_t>(t) * kHidden + kb * 32 +
                              j];
                if (std::abs(expected - scale * code) >
                    0.501F * scale + 1e-7F) {
                  throw std::runtime_error("mixed Q8 rounding differs");
                }
              }
            }
          }
        }
      }
    }
    for (const std::uint32_t chunk : {1U, 8U, 9U, 16U}) {
      Upload(&d_res_f16, res);
      Upload(&d_inject_vec, inject_vec);
      for (std::uint32_t off = 0; off < kTokens; off += chunk) {
        q::HcCombineF16(
            d_res_f16.get() + std::size_t(off) * kHcDim,
            d_block_out.get() + std::size_t(off) * kHidden,
            d_inject_vec.get() + std::size_t(off) * kStreams * vec_parts,
            vec_parts, d_gamma.get(),
            d_xn_f16.get() + std::size_t(off) * kHcDim, nullptr,
            std::min(chunk, kTokens - off), kHidden, kStreams, kEps, nullptr);
      }
      CheckHip(hipDeviceSynchronize(), "short HC prefill");
      const auto actual_res = Download(&d_res_f16, kRows);
      std::vector<__half> actual_norm(kRows);
      CheckHip(hipMemcpy(actual_norm.data(), d_xn_f16.get(),
                         kRows * sizeof(__half), hipMemcpyDeviceToHost),
               "short HC norm");
      if (actual_res != res_f16 ||
          std::memcmp(actual_norm.data(), xn_f16.data(),
                      kRows * sizeof(__half)) != 0)
        throw std::runtime_error("HC prefill depends on chunk width");
    }
    std::cout << "HC mixed outputs and short prefill chunks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
