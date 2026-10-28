#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {

// The model's MoE combine geometry: top-10 slots over a 2,560-wide hidden
// state, router row stride num_experts + 1.
constexpr std::uint32_t kTokens = 37;
constexpr std::uint32_t kSlots = 10;
constexpr std::uint32_t kHidden = 2560;
constexpr std::uint32_t kGateStride = 513;
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
                              float scale, float offset = 0.0F) {
  std::vector<float> values(count);
  for (float& value : values) {
    value = offset +
            scale *
                static_cast<float>(
                    static_cast<int>(NextRandom(&seed) & 0xFFFFU) - 32768) /
                32768.0F;
  }
  return values;
}

std::uint16_t FloatToBf16(float v) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &v, sizeof(bits));
  bits += 0x7FFFu + ((bits >> 16) & 1u);
  return static_cast<std::uint16_t>(bits >> 16);
}

float Bf16ToFloat(std::uint16_t h) {
  const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
  float v = 0.0F;
  std::memcpy(&v, &bits, sizeof(v));
  return v;
}

template<typename T>
void Upload(HipBuffer<T>* destination, const std::vector<T>& source) {
  CheckHip(hipMemcpy(destination->get(), source.data(), destination->bytes(),
                     hipMemcpyHostToDevice),
           "upload");
}

template<typename T>
std::vector<T> Download(HipBuffer<T>* source, std::size_t count) {
  std::vector<T> values(count);
  CheckHip(hipMemcpy(values.data(), source->get(), source->bytes(),
                     hipMemcpyDeviceToHost),
           "download");
  return values;
}

// Independent host evaluation of the inject contract: one rounded product,
// the remaining stream/component FMAs, then descending wave sums and the
// ordered eight-wave total. The ordinary mixed epilogue reassociates its
// first products under fast-math and is not an exact F32 reference.
std::vector<float> InjectReference(const std::vector<__half>& norm,
                                   const std::vector<float>& weights,
                                   std::uint32_t parts) {
  constexpr std::size_t kThreads = 256;
  constexpr std::size_t kStreams = 4;
  constexpr std::size_t kDim = kStreams * kHidden;
  std::vector<float> reference(kTokens * kStreams * parts);
  for (std::size_t t = 0; t < kTokens; ++t) {
    for (std::size_t o = 0; o < kStreams; ++o) {
      for (std::size_t part = 0; part < parts; ++part) {
        std::array<float, kThreads> dots{};
        for (std::size_t lane = 0; lane < kThreads; ++lane) {
          const std::size_t i = (part * kThreads + lane) * 4;
          if (i >= kHidden)
            continue;
          const std::size_t first = o * kDim + i;
          volatile float first_product =
              weights[first] * __half2float(norm[t * kDim + i]);
          float dot = first_product;
          for (std::size_t s = 0; s < kStreams; ++s) {
            for (std::size_t c = s == 0 ? 1 : 0; c < 4; ++c) {
              dot = std::fma(weights[o * kDim + s * kHidden + i + c],
                             __half2float(norm[t * kDim + s * kHidden + i + c]),
                             dot);
            }
          }
          dots[lane] = dot;
        }
        for (std::size_t off = 16; off > 0; off /= 2) {
          const auto previous = dots;
          for (std::size_t lane = 0; lane < kThreads; ++lane) {
            volatile float sum = previous[lane] + previous[lane ^ off];
            dots[lane] = sum;
          }
        }
        float total = 0.0F;
        for (std::size_t wave = 0; wave < kThreads / 32; ++wave) {
          volatile float sum = total + dots[wave * 32];
          total = sum;
        }
        reference[(t * kStreams + o) * parts + part] = total;
      }
    }
  }
  return reference;
}

}  // namespace

int main() {
  try {
    constexpr std::size_t kExpert =
        static_cast<std::size_t>(kTokens) * kSlots * kHidden;
    constexpr std::size_t kOut = static_cast<std::size_t>(kTokens) * kHidden;
    const auto expert_out = MakeValues(kExpert, 0x1234ABCDU, 1.0F);
    const auto weights = MakeValues(static_cast<std::size_t>(kTokens) * kSlots,
                                    0xBADC0FFEU, 0.5F, 0.5F);
    const auto shared = MakeValues(kOut, 0xDEADBEEFU, 1.0F);
    const auto gate = MakeValues(
        static_cast<std::size_t>(kTokens) * kGateStride, 0xC0FFEE11U, 3.0F);
    HipBuffer<float> d_expert(kExpert);
    HipBuffer<float> d_weights(weights.size());
    HipBuffer<float> d_shared(kOut);
    HipBuffer<float> d_gate(gate.size());
    HipBuffer<float> d_ref(kOut);
    HipBuffer<float> d_vec(kOut);
    Upload(&d_expert, expert_out);
    Upload(&d_weights, weights);
    Upload(&d_shared, shared);
    Upload(&d_gate, gate);
    q::MoeEpilogue(d_expert.get(), d_weights.get(), d_shared.get(),
                   d_gate.get(), kGateStride, d_ref.get(), kTokens, kSlots,
                   kHidden, nullptr);
    q::MoeEpilogueVec4(d_expert.get(), d_weights.get(), d_shared.get(),
                       d_gate.get(), kGateStride, d_vec.get(), kTokens, kSlots,
                       kHidden, nullptr);
    CheckHip(hipDeviceSynchronize(), "MoE epilogue synchronization");
    const auto ref = Download(&d_ref, kOut);
    const auto vec = Download(&d_vec, kOut);
    double worst = 0.0;
    for (std::size_t i = 0; i < kOut; ++i) {
      worst = std::max(worst, std::abs(static_cast<double>(ref[i] - vec[i])));
    }
    std::cout << "MoE epilogue vec4 worst absolute error " << worst << '\n';
    // Omitting an unused norm must preserve the fused residual exactly.
    // This also covers ragged token counts and all ten routed experts.
    constexpr std::size_t kResidual = kOut * 4;
    const auto residual = MakeValues(kResidual, 0x13579U, 1.0F);
    const auto parts = q::HcInjectPartsVec4(kHidden);
    HipBuffer<float> d_res_norm(kResidual);
    HipBuffer<float> d_res_only(kResidual);
    HipBuffer<float> d_res_separate(kResidual);
    HipBuffer<float> d_inject(kTokens * 4 * parts);
    HipBuffer<float> d_gamma(4 * kHidden);
    HipBuffer<__half> d_half(kExpert);
    HipBuffer<__half> d_norm(kResidual);
    HipBuffer<std::uint8_t> d_q8(q::Q8TiledBytes(kTokens, 4 * kHidden));
    Upload(&d_res_norm, residual);
    Upload(&d_res_only, residual);
    Upload(&d_res_separate, residual);
    Upload(&d_inject, MakeValues(kTokens * 4 * parts, 0x24680U, 0.5F));
    Upload(&d_gamma, MakeValues(4 * kHidden, 0x98765U, 0.5F, 1.0F));
    q::NarrowActivations(d_expert.get(), d_half.get(), false, kExpert, nullptr);
    for (bool normalize : {true, false}) {
      if (!q::HcCombineMoeF16(normalize ? d_res_norm.get() : d_res_only.get(),
                              d_half.get(), d_weights.get(), d_shared.get(),
                              d_gate.get(), kGateStride, kSlots, d_inject.get(),
                              parts, normalize ? d_gamma.get() : nullptr,
                              normalize ? d_norm.get() : nullptr,
                              normalize ? d_q8.get() : nullptr, kTokens,
                              kHidden, 4, 1e-6F, nullptr)) {
        throw std::runtime_error("fused MoE residual rejected");
      }
    }
    if (Download(&d_res_norm, kResidual) != Download(&d_res_only, kResidual)) {
      throw std::runtime_error("skipping unused norm changed the residual");
    }
    // The tiled Q8 norm is optional when the next mixer's down projection
    // does not read it. A null output must leave the buffer untouched and
    // produce the same residual and F16 norm; a requested one keeps its
    // exact bytes.
    {
      HipBuffer<float> d_res_noq8(kResidual);
      HipBuffer<__half> d_norm_noq8(kResidual);
      Upload(&d_res_norm, residual);
      CheckHip(hipMemset(d_q8.get(), 0, d_q8.bytes()), "clear requested Q8");
      if (!q::HcCombineMoeF16(d_res_norm.get(), d_half.get(), d_weights.get(),
                              d_shared.get(), d_gate.get(), kGateStride, kSlots,
                              d_inject.get(), parts, d_gamma.get(),
                              d_norm.get(), d_q8.get(), kTokens, kHidden, 4,
                              1e-6F, nullptr)) {
        throw std::runtime_error("fused MoE norm rejected with Q8");
      }
      const auto requested_res = Download(&d_res_norm, kResidual);
      const auto requested_norm = Download(&d_norm, kResidual);
      Upload(&d_res_noq8, residual);
      CheckHip(hipMemset(d_q8.get(), 0xA5, d_q8.bytes()),
               "poison unused Q8 norm");
      if (!q::HcCombineMoeF16(d_res_noq8.get(), d_half.get(), d_weights.get(),
                              d_shared.get(), d_gate.get(), kGateStride, kSlots,
                              d_inject.get(), parts, d_gamma.get(),
                              d_norm_noq8.get(), nullptr, kTokens, kHidden, 4,
                              1e-6F, nullptr)) {
        throw std::runtime_error("fused MoE norm rejected without Q8");
      }
      if (Download(&d_res_noq8, kResidual) != requested_res) {
        throw std::runtime_error("skipping the Q8 norm changed the residual");
      }
      const auto noq8_norm = Download(&d_norm_noq8, kResidual);
      if (std::memcmp(noq8_norm.data(), requested_norm.data(),
                      d_norm.bytes()) != 0) {
        throw std::runtime_error("skipping the Q8 norm changed the F16 norm");
      }
      const std::vector<std::uint8_t> poison(d_q8.bytes(), 0xA5);
      if (Download(&d_q8, d_q8.bytes()) != poison) {
        throw std::runtime_error("unused Q8 norm output was written");
      }
    }
    q::MoeEpilogueVec4F16(d_half.get(), d_weights.get(), d_shared.get(),
                          d_gate.get(), kGateStride, d_vec.get(), kTokens,
                          kSlots, kHidden, nullptr);
    q::HcCombine(d_res_separate.get(), d_vec.get(), d_inject.get(), parts,
                 nullptr, nullptr, kTokens, kHidden, 4, 1e-6F, nullptr);
    if (Download(&d_res_separate, kResidual) !=
        Download(&d_res_only, kResidual)) {
      throw std::runtime_error("fusing the residual-only combine changed it");
    }
    // A deferred update must keep the separate F32 store's rounding before
    // the next combine and preserve the downstream F16/Q8 normalization.
    HipBuffer<float> d_prior(kOut);
    HipBuffer<float> d_prior_w(kTokens * 4);
    HipBuffer<__half> d_norm_separate(kResidual);
    HipBuffer<std::uint8_t> d_q8_separate(d_q8.bytes());
    Upload(&d_prior, MakeValues(kOut, 0xFEED123U, 0.23F));
    Upload(&d_prior_w, MakeValues(kTokens * 4, 0xABC789U, 0.75F, 1.0F));
    HipBuffer<float> d_next_inject(kTokens * 4 * parts);
    HipBuffer<float> d_next_inject_w(kHidden * 4 * 4);
    const auto next_inject_weights =
        MakeValues(kHidden * 4 * 4, 0xDED123U, 0.1F);
    Upload(&d_next_inject_w, next_inject_weights);
    std::vector<std::uint16_t> next_inject_bf16(next_inject_weights.size());
    std::vector<float> next_inject_f32(next_inject_weights.size());
    for (std::size_t i = 0; i < next_inject_weights.size(); ++i) {
      next_inject_bf16[i] = FloatToBf16(next_inject_weights[i]);
      next_inject_f32[i] = Bf16ToFloat(next_inject_bf16[i]);
    }
    HipBuffer<std::uint16_t> d_next_inject_bf16(next_inject_bf16.size());
    CheckHip(hipMemcpy(d_next_inject_bf16.get(), next_inject_bf16.data(),
                       d_next_inject_bf16.bytes(), hipMemcpyHostToDevice),
             "upload BF16 inject weights");
    for (bool bf16 : {false, true}) {
      const void* next_w =
          bf16 ? static_cast<const void*>(d_next_inject_bf16.get())
               : static_cast<const void*>(d_next_inject_w.get());
      const q::WeightType next_type =
          bf16 ? q::WeightType::kBF16 : q::WeightType::kF32;
      const auto& next_reference = bf16 ? next_inject_f32 : next_inject_weights;
      for (unsigned mode = 0; mode < 3; ++mode) {
        const bool normalize = mode != 0;
        const bool inject_next = mode == 2;
        Upload(&d_res_norm, residual);
        Upload(&d_res_separate, residual);
        CheckHip(hipMemset(d_q8.get(), 0, d_q8.bytes()), "clear fused Q8");
        CheckHip(hipMemset(d_q8_separate.get(), 0, d_q8_separate.bytes()),
                 "clear reference Q8");
        q::HcApplyDeferred(d_res_separate.get(), d_prior.get(), d_prior_w.get(),
                           kTokens, kHidden, nullptr);
        for (bool deferred : {false, true}) {
          if (!q::HcCombineMoeF16(
                  deferred ? d_res_norm.get() : d_res_separate.get(),
                  d_half.get(), d_weights.get(), d_shared.get(), d_gate.get(),
                  kGateStride, kSlots, d_inject.get(), parts,
                  normalize ? d_gamma.get() : nullptr,
                  normalize ? (deferred ? d_norm.get() : d_norm_separate.get())
                            : nullptr,
                  normalize ? (deferred ? d_q8.get() : d_q8_separate.get())
                            : nullptr,
                  kTokens, kHidden, 4, 1e-6F, nullptr,
                  deferred ? d_prior.get() : nullptr,
                  deferred ? d_prior_w.get() : nullptr,
                  deferred && inject_next ? d_next_inject.get() : nullptr,
                  deferred && inject_next ? next_w : nullptr,
                  deferred && inject_next ? next_type : q::WeightType::kF32)) {
            throw std::runtime_error("deferred fused combine rejected");
          }
        }
        const auto fused = Download(&d_res_norm, kResidual);
        const auto separate = Download(&d_res_separate, kResidual);
        if (std::memcmp(fused.data(), separate.data(), d_res_norm.bytes()) !=
            0) {
          throw std::runtime_error("deferred combine changed residual bytes");
        }
        if (normalize) {
          const auto norm = Download(&d_norm, kResidual);
          const auto reference_norm = Download(&d_norm_separate, kResidual);
          if (std::memcmp(norm.data(), reference_norm.data(), d_norm.bytes()) !=
              0) {
            std::size_t changed = 0;
            for (std::size_t i = 0; i < norm.size(); ++i) {
              if (std::memcmp(&norm[i], &reference_norm[i], sizeof(__half)) !=
                  0) {
                if (changed < 8)
                  std::cerr << "norm mismatch " << i << ": "
                            << __half2float(norm[i]) << " vs "
                            << __half2float(reference_norm[i]) << '\n';
                ++changed;
              }
            }
            throw std::runtime_error("deferred combine changed " +
                                     std::to_string(changed) +
                                     " F16 norm values");
          }
          if (Download(&d_q8, d_q8.bytes()) !=
              Download(&d_q8_separate, d_q8_separate.bytes())) {
            throw std::runtime_error("deferred combine changed Q8 norm bytes");
          }
          if (inject_next) {
            const auto actual_inject =
                Download(&d_next_inject, kTokens * 4 * parts);
            const auto reference_inject =
                InjectReference(reference_norm, next_reference, parts);
            if (std::memcmp(actual_inject.data(), reference_inject.data(),
                            d_next_inject.bytes()) != 0) {
              throw std::runtime_error("deferred combine changed inject bytes");
            }
          }
        }
      }
    }
    // The MMQ down projection writes F32 expert rows; its fused combine
    // must reproduce MoeEpilogueVec4 + HcCombine byte for byte as well.
    HipBuffer<float> d_res_f32(kResidual);
    HipBuffer<float> d_res_f32_norm(kResidual);
    Upload(&d_res_f32, residual);
    Upload(&d_res_f32_norm, residual);
    for (bool normalize : {true, false}) {
      if (!q::HcCombineMoeF32(
              normalize ? d_res_f32_norm.get() : d_res_f32.get(),
              d_expert.get(), d_weights.get(), d_shared.get(), d_gate.get(),
              kGateStride, kSlots, d_inject.get(), parts,
              normalize ? d_gamma.get() : nullptr,
              normalize ? d_norm.get() : nullptr,
              normalize ? d_q8.get() : nullptr, kTokens, kHidden, 4, 1e-6F,
              nullptr)) {
        throw std::runtime_error("fused F32 MoE residual rejected");
      }
    }
    if (Download(&d_res_f32_norm, kResidual) !=
        Download(&d_res_f32, kResidual)) {
      throw std::runtime_error("norm pass changed the F32 fused residual");
    }
    Upload(&d_res_separate, residual);
    q::MoeEpilogueVec4(d_expert.get(), d_weights.get(), d_shared.get(),
                       d_gate.get(), kGateStride, d_vec.get(), kTokens, kSlots,
                       kHidden, nullptr);
    q::HcCombine(d_res_separate.get(), d_vec.get(), d_inject.get(), parts,
                 nullptr, nullptr, kTokens, kHidden, 4, 1e-6F, nullptr);
    if (Download(&d_res_separate, kResidual) !=
        Download(&d_res_f32, kResidual)) {
      throw std::runtime_error("fusing the F32 combine changed it");
    }
    // Row boundaries: single-row, the wide-mixer threshold and a large
    // prefill batch. The optional Q8 norm must not alter the residual or
    // F16 norm, must stay unwritten when null, and the residual-only
    // combine must stay fused.
    for (const std::uint32_t rows : {1U, 96U, 2049U}) {
      const std::size_t out = static_cast<std::size_t>(rows) * kHidden;
      const std::size_t residual_count = out * 4;
      const std::size_t expert_count =
          static_cast<std::size_t>(rows) * kSlots * kHidden;
      HipBuffer<float> d_expert_rows(expert_count);
      HipBuffer<__half> d_half_rows(expert_count);
      HipBuffer<float> d_weights_rows(static_cast<std::size_t>(rows) * kSlots);
      HipBuffer<float> d_shared_rows(out);
      HipBuffer<float> d_gate_rows(static_cast<std::size_t>(rows) *
                                   kGateStride);
      HipBuffer<float> d_inject_rows(static_cast<std::size_t>(rows) * 4 *
                                     parts);
      HipBuffer<float> d_gamma_rows(4 * kHidden);
      HipBuffer<float> d_res_q8(residual_count);
      HipBuffer<float> d_res_noq8(residual_count);
      HipBuffer<float> d_res_only(residual_count);
      HipBuffer<float> d_res_plain(residual_count);
      HipBuffer<__half> d_norm_q8(residual_count);
      HipBuffer<__half> d_norm_noq8(residual_count);
      HipBuffer<std::uint8_t> d_q8_rows(q::Q8TiledBytes(rows, 4 * kHidden));
      HipBuffer<float> d_block_rows(out);
      Upload(&d_expert_rows, MakeValues(expert_count, 0x51A2U + rows, 1.0F));
      Upload(&d_weights_rows,
             MakeValues(static_cast<std::size_t>(rows) * kSlots, 0x9E37U, 0.5F,
                        0.5F));
      Upload(&d_shared_rows, MakeValues(out, 0x7C15U, 1.0F));
      Upload(&d_gate_rows,
             MakeValues(static_cast<std::size_t>(rows) * kGateStride, 0x3D2BU,
                        3.0F));
      Upload(&d_inject_rows,
             MakeValues(static_cast<std::size_t>(rows) * 4 * parts, 0x6A09U,
                        0.5F));
      Upload(&d_gamma_rows, MakeValues(4 * kHidden, 0xBB67U, 0.5F, 1.0F));
      const auto row_residual =
          MakeValues(residual_count, 0x85EBU + rows, 1.0F);
      Upload(&d_res_q8, row_residual);
      Upload(&d_res_noq8, row_residual);
      Upload(&d_res_only, row_residual);
      q::NarrowActivations(d_expert_rows.get(), d_half_rows.get(), false,
                           expert_count, nullptr);
      if (!q::HcCombineMoeF16(
              d_res_q8.get(), d_half_rows.get(), d_weights_rows.get(),
              d_shared_rows.get(), d_gate_rows.get(), kGateStride, kSlots,
              d_inject_rows.get(), parts, d_gamma_rows.get(), d_norm_q8.get(),
              d_q8_rows.get(), rows, kHidden, 4, 1e-6F, nullptr)) {
        throw std::runtime_error("row-size fused combine rejected");
      }
      CheckHip(hipMemset(d_q8_rows.get(), 0xA5, d_q8_rows.bytes()),
               "poison row Q8 norm");
      if (!q::HcCombineMoeF16(
              d_res_noq8.get(), d_half_rows.get(), d_weights_rows.get(),
              d_shared_rows.get(), d_gate_rows.get(), kGateStride, kSlots,
              d_inject_rows.get(), parts, d_gamma_rows.get(), d_norm_noq8.get(),
              nullptr, rows, kHidden, 4, 1e-6F, nullptr)) {
        throw std::runtime_error("row-size fused combine rejected without Q8");
      }
      if (Download(&d_res_noq8, residual_count) !=
          Download(&d_res_q8, residual_count)) {
        throw std::runtime_error("row-size Q8 output changed the residual");
      }
      const auto norm_q8 = Download(&d_norm_q8, residual_count);
      const auto norm_noq8 = Download(&d_norm_noq8, residual_count);
      if (std::memcmp(norm_q8.data(), norm_noq8.data(), d_norm_q8.bytes()) !=
          0) {
        throw std::runtime_error("row-size Q8 output changed the F16 norm");
      }
      const std::vector<std::uint8_t> poison(d_q8_rows.bytes(), 0xA5);
      if (Download(&d_q8_rows, d_q8_rows.bytes()) != poison) {
        throw std::runtime_error("row-size unused Q8 output was written");
      }
      if (!q::HcCombineMoeF16(d_res_only.get(), d_half_rows.get(),
                              d_weights_rows.get(), d_shared_rows.get(),
                              d_gate_rows.get(), kGateStride, kSlots,
                              d_inject_rows.get(), parts, nullptr, nullptr,
                              nullptr, rows, kHidden, 4, 1e-6F, nullptr)) {
        throw std::runtime_error("row-size residual-only combine rejected");
      }
      Upload(&d_res_plain, row_residual);
      q::MoeEpilogueVec4F16(d_half_rows.get(), d_weights_rows.get(),
                            d_shared_rows.get(), d_gate_rows.get(), kGateStride,
                            d_block_rows.get(), rows, kSlots, kHidden, nullptr);
      q::HcCombine(d_res_plain.get(), d_block_rows.get(), d_inject_rows.get(),
                   parts, nullptr, nullptr, rows, kHidden, 4, 1e-6F, nullptr);
      if (Download(&d_res_only, residual_count) !=
          Download(&d_res_plain, residual_count)) {
        throw std::runtime_error("row-size residual-only combine changed");
      }
      if (rows >= 1024) {
        const std::size_t pitch = 4 * kHidden + kNormPad;
        const std::size_t padded_count = static_cast<std::size_t>(rows) * pitch;
        const __half pad_poison = __float2half_rn(-4321.0F);
        const std::vector<__half> pad_fill(padded_count, pad_poison);
        HipBuffer<__half> d_norm_padded(padded_count);
        HipBuffer<float> d_res_padded(residual_count);
        Upload(&d_res_padded, row_residual);
        Upload(&d_norm_padded, pad_fill);
        if (!q::HcCombineMoeF16(d_res_padded.get(), d_half_rows.get(),
                                d_weights_rows.get(), d_shared_rows.get(),
                                d_gate_rows.get(), kGateStride, kSlots,
                                d_inject_rows.get(), parts, d_gamma_rows.get(),
                                d_norm_padded.get(), nullptr, rows, kHidden, 4,
                                1e-6F, nullptr, nullptr, nullptr, nullptr,
                                nullptr, q::WeightType::kF32, pitch)) {
          throw std::runtime_error("row-size padded F16 combine rejected");
        }
        if (Download(&d_res_padded, residual_count) !=
            Download(&d_res_noq8, residual_count)) {
          throw std::runtime_error("padded F16 combine changed the residual");
        }
        const auto padded = Download(&d_norm_padded, padded_count);
        const auto packed = Download(&d_norm_q8, residual_count);
        for (std::uint32_t t = 0; t < rows; ++t) {
          if (std::memcmp(
                  padded.data() + static_cast<std::size_t>(t) * pitch,
                  packed.data() + static_cast<std::size_t>(t) * 4 * kHidden,
                  4 * kHidden * sizeof(__half)) != 0) {
            throw std::runtime_error("padded F16 combine changed the norm");
          }
          for (std::uint32_t i = 0; i < kNormPad; ++i) {
            const __half value =
                padded[static_cast<std::size_t>(t) * pitch + 4 * kHidden + i];
            if (std::memcmp(&value, &pad_poison, sizeof(value)) != 0) {
              throw std::runtime_error("padded F16 combine wrote norm padding");
            }
          }
        }
        HipBuffer<float> d_res_f32_packed(residual_count);
        HipBuffer<__half> d_norm_f32_packed(residual_count);
        Upload(&d_res_f32_packed, row_residual);
        if (!q::HcCombineMoeF32(d_res_f32_packed.get(), d_expert_rows.get(),
                                d_weights_rows.get(), d_shared_rows.get(),
                                d_gate_rows.get(), kGateStride, kSlots,
                                d_inject_rows.get(), parts, d_gamma_rows.get(),
                                d_norm_f32_packed.get(), nullptr, rows, kHidden,
                                4, 1e-6F, nullptr)) {
          throw std::runtime_error("row-size packed F32 combine rejected");
        }
        const auto packed_f32 = Download(&d_norm_f32_packed, residual_count);
        Upload(&d_res_padded, row_residual);
        Upload(&d_norm_padded, pad_fill);
        if (!q::HcCombineMoeF32(d_res_padded.get(), d_expert_rows.get(),
                                d_weights_rows.get(), d_shared_rows.get(),
                                d_gate_rows.get(), kGateStride, kSlots,
                                d_inject_rows.get(), parts, d_gamma_rows.get(),
                                d_norm_padded.get(), nullptr, rows, kHidden, 4,
                                1e-6F, nullptr, pitch)) {
          throw std::runtime_error("row-size padded F32 combine rejected");
        }
        if (Download(&d_res_padded, residual_count) !=
            Download(&d_res_f32_packed, residual_count)) {
          throw std::runtime_error("padded F32 combine changed the residual");
        }
        const auto padded_f32 = Download(&d_norm_padded, padded_count);
        for (std::uint32_t t = 0; t < rows; ++t) {
          if (std::memcmp(
                  padded_f32.data() + static_cast<std::size_t>(t) * pitch,
                  packed_f32.data() + static_cast<std::size_t>(t) * 4 * kHidden,
                  4 * kHidden * sizeof(__half)) != 0) {
            throw std::runtime_error("padded F32 combine changed the norm");
          }
          for (std::uint32_t i = 0; i < kNormPad; ++i) {
            const __half value =
                padded_f32[static_cast<std::size_t>(t) * pitch + 4 * kHidden +
                           i];
            if (std::memcmp(&value, &pad_poison, sizeof(value)) != 0) {
              throw std::runtime_error("padded F32 combine wrote norm padding");
            }
          }
        }
      }
    }
    return worst == 0.0 ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
