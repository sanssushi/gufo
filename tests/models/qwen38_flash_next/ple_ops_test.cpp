// The PLE conv weights arrive in their tensor's GGUF encoding: F32 in the
// UD-Q4_K_XL mix, F16 in GSQ-RCO. Both encodings of the same values must
// convolve identically; the F16 tensor is half the byte length of an F32 one,
// so a raw float read also walks off its allocation.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {

constexpr std::uint32_t kChannels = 64;
constexpr std::uint32_t kKernel = 4;
constexpr std::uint32_t kDilation = 3;
constexpr std::uint32_t kTokens = 5;
constexpr std::uint32_t kHistory = (kKernel - 1) * kDilation;

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

void* Upload(const void* host, std::size_t bytes) {
  void* device = nullptr;
  CheckHip(hipMalloc(&device, bytes), "hipMalloc");
  CheckHip(hipMemcpy(device, host, bytes, hipMemcpyHostToDevice), "upload");
  return device;
}

std::vector<float> Download(const float* device, std::size_t count) {
  std::vector<float> host(count);
  CheckHip(hipMemcpy(host.data(), device, count * sizeof(float),
                     hipMemcpyDeviceToHost),
           "download");
  return host;
}

/// Runs the conv with one weight encoding and returns its output and the
/// advanced history.
std::vector<float> RunConv(const std::vector<std::uint8_t>& weights_raw,
                           q::WeightType wtype, const std::vector<float>& in,
                           const std::vector<float>& history,
                           std::vector<float>* history_out) {
  const std::size_t hist_count = static_cast<std::size_t>(kHistory) * kChannels;
  const std::size_t out_count = static_cast<std::size_t>(kTokens) * kChannels;
  auto* d_w = static_cast<const std::uint8_t*>(
      Upload(weights_raw.data(), weights_raw.size()));
  auto* d_in = static_cast<const float*>(Upload(in.data(), in.size() * 4));
  auto* d_history =
      static_cast<float*>(Upload(history.data(), history.size() * 4));
  const std::vector<float> zeros((kTokens + kHistory) * kChannels + 64, 0.0F);
  auto* d_scratch = static_cast<float*>(Upload(zeros.data(), zeros.size() * 4));
  auto* d_out = static_cast<float*>(
      Upload(zeros.data(), (out_count + 64) * sizeof(float)));
  q::PleConv(d_in, d_w, wtype, d_history, d_scratch, d_out, q::RollbackRows{},
             kTokens, kChannels, kKernel, kDilation, nullptr);
  CheckHip(hipGetLastError(), "PleConv launch");
  CheckHip(hipDeviceSynchronize(), "PleConv");
  const auto all = Download(d_out, out_count + 64);
  for (std::size_t i = out_count; i < out_count + 64; ++i) {
    if (all[i] != 0.0F) {
      throw std::runtime_error("PleConv overwrote its output guard");
    }
  }
  *history_out = Download(d_history, hist_count);
  for (void* p : {const_cast<void*>(static_cast<const void*>(d_w)),
                  const_cast<void*>(static_cast<const void*>(d_in)),
                  static_cast<void*>(d_history), static_cast<void*>(d_scratch),
                  static_cast<void*>(d_out)}) {
    CheckHip(hipFree(p), "free");
  }
  return std::vector<float>(all.begin(), all.begin() + out_count);
}

void CheckConvFormats() {
  std::uint32_t state = 0x2B1D2B1DU;
  const std::size_t wcount = static_cast<std::size_t>(kChannels) * kKernel;
  std::vector<float> weights32(wcount);
  std::vector<std::uint8_t> weights16(wcount * 2);
  for (std::size_t i = 0; i < wcount; ++i) {
    const __half h = __float2half(Uniform(&state, 0.5F));
    weights32[i] = __half2float(h);
    std::memcpy(weights16.data() + i * 2, &h, 2);
  }
  std::vector<std::uint8_t> weights32_raw(wcount * sizeof(float));
  std::memcpy(weights32_raw.data(), weights32.data(), weights32_raw.size());
  std::vector<float> in(static_cast<std::size_t>(kTokens) * kChannels);
  std::vector<float> history(static_cast<std::size_t>(kHistory) * kChannels);
  for (float& v : in) {
    v = Uniform(&state, 2.0F);
  }
  for (float& v : history) {
    v = Uniform(&state, 2.0F);
  }
  std::vector<float> history32, history16;
  const auto out32 =
      RunConv(weights32_raw, q::WeightType::kF32, in, history, &history32);
  const auto out16 =
      RunConv(weights16, q::WeightType::kF16, in, history, &history16);
  for (std::size_t i = 0; i < out32.size(); ++i) {
    if (std::memcmp(&out32[i], &out16[i], sizeof(float)) != 0) {
      throw std::runtime_error(
          "F16 conv weights do not match the F32 encodings at element " +
          std::to_string(i));
    }
  }
  if (history32 != history16) {
    throw std::runtime_error("conv history advanced differently per encoding");
  }
  std::cout << "PLE conv F16 matches F32 over " << out32.size() << " outputs\n";
}

}  // namespace

int main() {
  try {
    CheckConvFormats();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
