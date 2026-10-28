// A wide copy verification (Executor::ForwardWide) must give every row the
// logits of consecutive verification passes of at most eight rows, and the
// logits of one-row verification passes, and a rollback to any kept prefix
// must continue exactly as those passes cut back at the same token. Two
// synthetic models keep the production layer geometry: a control model with
// Q8_0 projections and Q4_K/Q5_1 experts, and a GSQ-RCO model with BF16
// hyper-connections, independently typed dense QKV/gate projections,
// IQ2_S/IQ3_XXS routed gate/up and Q2_0/IQ4_NL routed down. Four trunk
// layers and reduced vocabulary and expert counts keep it cheap, so no model
// file is loaded.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/platform/fs.hpp"
#include "src/core/quant/ggml_gemm.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/ngram.hpp"
#include "tests/support/temporary_directory.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

using core::GgmlType;

/// Which quantization types the synthetic trunk stores. The MTP block always
/// uses the shared Q8_0 sidecar's types, as the artifact's draft sidecar does.
enum class TrunkStyle { kQ8Control, kGsqRco };

struct DeviceModelTestAccess {
  DeviceModel& model;
  std::uint64_t state;

  std::uint64_t Next() {
    // splitmix64
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  float Uniform() {  // [-1, 1)
    return static_cast<float>(Next() >> 40) / 8388608.0F - 1.0F;
  }

  void* Upload(const std::vector<std::uint8_t>& bytes) {
    void* p = nullptr;
    constexpr std::size_t kTail = 4096;
    if (hipMalloc(&p, bytes.size() + kTail) != hipSuccess ||
        hipMemcpy(p, bytes.data(), bytes.size(), hipMemcpyHostToDevice) !=
            hipSuccess ||
        hipMemset(static_cast<std::uint8_t*>(p) + bytes.size(), 0, kTail) !=
            hipSuccess)
      throw std::runtime_error("synthetic weight upload failed");
    model.allocations_.push_back(p);
    return p;
  }
  static void PutHalf(std::uint8_t* dst, float value) {
    const auto h = static_cast<_Float16>(value);
    std::memcpy(dst, &h, 2);
  }

  DeviceTensor Make(GgmlType type, std::uint32_t rows, std::uint32_t cols,
                    std::vector<std::uint8_t> bytes, std::uint32_t experts) {
    DeviceTensor t;
    t.data = Upload(bytes);
    t.type = type;
    t.rows = rows;
    t.cols = cols;
    t.experts = experts;
    // Mirror DeviceModel::Upload's activation-staging sizing: 16-bit weights
    // and the K-quant/IQ dense F16-tile route both read s_.x_half.
    if (type == GgmlType::kF16 || type == GgmlType::kBF16 ||
        (experts == 1 && KQuantWmmaType(type)))
      model.max_half_cols_ = std::max<std::size_t>(model.max_half_cols_, cols);
    if (type == GgmlType::kQ8_0 && experts == 1)
      model.max_q8_cols_ = std::max<std::size_t>(model.max_q8_cols_, cols);
    return t;
  }
  DeviceTensor F32(std::uint32_t rows, std::uint32_t cols, float center,
                   float spread) {
    std::vector<std::uint8_t> bytes(std::size_t{rows} * cols * 4);
    for (std::size_t i = 0; i < bytes.size() / 4; ++i) {
      const float v = center + spread * Uniform();
      std::memcpy(bytes.data() + 4 * i, &v, 4);
    }
    return Make(GgmlType::kF32, rows, cols, std::move(bytes), 1);
  }
  DeviceTensor Half(GgmlType type, std::uint32_t rows, std::uint32_t cols,
                    float spread) {
    std::vector<std::uint8_t> bytes(std::size_t{rows} * cols * 2);
    for (std::size_t i = 0; i < bytes.size() / 2; ++i) {
      const float v = spread * Uniform();
      if (type == GgmlType::kF16) {
        PutHalf(bytes.data() + 2 * i, v);
      } else {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &v, 4);
        const auto bf = static_cast<std::uint16_t>(bits >> 16);
        std::memcpy(bytes.data() + 2 * i, &bf, 2);
      }
    }
    return Make(type, rows, cols, std::move(bytes), 1);
  }
  /// Uniform int8 quants have a standard deviation of about 73.
  DeviceTensor Q8(std::uint32_t rows, std::uint32_t cols,
                  std::uint32_t experts = 1, float spread = 0.0F) {
    if (spread == 0.0F)
      spread = 1.0F / std::sqrt(static_cast<float>(cols));
    const std::size_t blocks = std::size_t{experts} * rows * (cols / 32);
    std::vector<std::uint8_t> bytes(blocks * 34);
    for (std::size_t b = 0; b < blocks; ++b) {
      auto* block = bytes.data() + b * 34;
      PutHalf(block, spread / 73.0F * (1.0F + 0.25F * Uniform()));
      for (int j = 0; j < 32; j += 8) {
        auto bits = Next();
        for (int k = 0; k < 8; ++k, bits >>= 8)
          block[2 + j + k] = static_cast<std::uint8_t>(
              std::max<std::int8_t>(static_cast<std::int8_t>(bits), -127));
      }
    }
    return Make(GgmlType::kQ8_0, rows, cols, std::move(bytes), experts);
  }
  /// Random 6-bit scales and minimums; dmin = 7.5 d centres the weights.
  DeviceTensor Q4K(std::uint32_t rows, std::uint32_t cols,
                   std::uint32_t experts) {
    const float d = 1.0F / std::sqrt(static_cast<float>(cols)) / 260.0F;
    const std::size_t blocks = std::size_t{experts} * rows * (cols / 256);
    std::vector<std::uint8_t> bytes(blocks * 144);
    for (std::size_t b = 0; b < blocks; ++b) {
      auto* block = bytes.data() + b * 144;
      PutHalf(block, d);
      PutHalf(block + 2, 7.5F * d);
      for (int j = 4; j < 144; j += 8) {
        auto bits = Next();
        for (int k = 0; k < 8 && j + k < 144; ++k, bits >>= 8)
          block[j + k] = static_cast<std::uint8_t>(bits);
      }
    }
    return Make(GgmlType::kQ4_K, rows, cols, std::move(bytes), experts);
  }
  DeviceTensor Q51(std::uint32_t rows, std::uint32_t cols,
                   std::uint32_t experts) {
    const float d = 1.0F / std::sqrt(static_cast<float>(cols)) / 9.2F;
    const std::size_t blocks = std::size_t{experts} * rows * (cols / 32);
    std::vector<std::uint8_t> bytes(blocks * 24);
    for (std::size_t b = 0; b < blocks; ++b) {
      auto* block = bytes.data() + b * 24;
      PutHalf(block, d);
      PutHalf(block + 2, -15.5F * d);
      for (int j = 4; j < 24; j += 8) {
        auto bits = Next();
        for (int k = 0; k < 8 && j + k < 24; ++k, bits >>= 8)
          block[j + k] = static_cast<std::uint8_t>(bits);
      }
    }
    return Make(GgmlType::kQ5_1, rows, cols, std::move(bytes), experts);
  }
  static std::size_t QuantBlockBytes(GgmlType type) {
    switch (type) {
      case GgmlType::kQ2_0:
      case GgmlType::kIQ4_NL:
        return 18;
      case GgmlType::kIQ2_S:
        return 82;
      case GgmlType::kIQ3_XXS:
        return 98;
      case GgmlType::kIQ3_S:
        return 110;
      case GgmlType::kIQ4_XS:
        return 136;
      case GgmlType::kQ4_K:
        return 144;
      case GgmlType::kQ5_K:
        return 176;
      case GgmlType::kQ6_K:
        return 210;
      default:
        throw std::logic_error("unsupported synthetic quantization type");
    }
  }
  static std::size_t QuantBlockElements(GgmlType type) {
    switch (type) {
      case GgmlType::kQ2_0:
        return 64;
      case GgmlType::kIQ4_NL:
        return 32;
      case GgmlType::kIQ2_S:
      case GgmlType::kIQ3_XXS:
      case GgmlType::kIQ3_S:
      case GgmlType::kIQ4_XS:
      case GgmlType::kQ4_K:
      case GgmlType::kQ5_K:
      case GgmlType::kQ6_K:
        return 256;
      default:
        throw std::logic_error("unsupported synthetic quantization type");
    }
  }
  /// Puts the block scale where the canonical layout keeps it (Q6_K last) and
  /// centres K-quant minimums on zero.
  static void PutBlockScale(GgmlType type, std::uint8_t* block, float d) {
    if (type == GgmlType::kQ6_K) {
      PutHalf(block + QuantBlockBytes(type) - 2, d);
      return;
    }
    PutHalf(block, d);
    if (type == GgmlType::kQ4_K || type == GgmlType::kQ5_K)
      PutHalf(block + 2, 7.5F * d);
  }
  /// Keeps the packed sub-block scale fields in a mid range, so the random
  /// codes decode to a model-like magnitude instead of a scale-field extreme.
  static void FixScales(GgmlType type, std::uint8_t* data, std::size_t size) {
    const std::size_t bytes = QuantBlockBytes(type);
    for (std::size_t b = 0; b < size / bytes; ++b) {
      std::uint8_t* block = data + b * bytes;
      switch (type) {
        case GgmlType::kQ6_K:
          for (int i = 0; i < 16; ++i)
            block[192 + i] = 8;
          break;
        case GgmlType::kIQ4_XS:
          block[2] = 0xAA;  // scales_h: 2 high bits per sub-block
          block[3] = 0xAA;
          for (int i = 0; i < 4; ++i)
            block[4 + i] = 0x88;  // scales_l: 8 low bits per sub-block
          break;
        case GgmlType::kIQ3_S:
          for (int i = 0; i < 4; ++i)
            block[106 + i] = 0x44;
          break;
        case GgmlType::kIQ2_S:
          for (int i = 0; i < 8; ++i)
            block[74 + i] = 0x44;
          break;
        case GgmlType::kIQ3_XXS:
          for (int i = 0; i < 8; ++i) {
            std::uint32_t word = 0;
            std::memcpy(&word, block + 66 + 4 * i, 4);
            word = (word & 0x0FFFFFFFU) | (4U << 28);
            std::memcpy(block + 66 + 4 * i, &word, 4);
          }
          break;
        default:
          break;
      }
    }
  }
  /// Canonical GSQ-RCO blocks with random codes and a block scale from the
  /// CPU dequantization oracle (never from a GPU kernel), so the synthetic
  /// weights carry the shards' magnitude.
  DeviceTensor Quant(GgmlType type, std::uint32_t rows, std::uint32_t cols,
                     std::uint32_t experts = 1) {
    const std::size_t elements = QuantBlockElements(type);
    const std::size_t bytes = QuantBlockBytes(type);
    const std::size_t blocks = std::size_t{experts} * rows * (cols / elements);
    std::vector<std::uint8_t> data(blocks * bytes);
    for (std::size_t i = 0; i < data.size(); i += 8) {
      auto bits = Next();
      const auto n = std::min<std::size_t>(8, data.size() - i);
      std::memcpy(data.data() + i, &bits, n);
    }
    FixScales(type, data.data(), data.size());
    std::vector<float> values(elements);
    PutBlockScale(type, data.data(), 1.0F);
    quant::Dequantize(type, data.data(), values.data(), elements);
    double sum = 0.0;
    for (float v : values)
      sum += static_cast<double>(v) * v;
    const float rms =
        static_cast<float>(std::sqrt(sum / static_cast<double>(elements)));
    const float d =
        1.0F / std::sqrt(static_cast<float>(cols)) / (rms > 0.0F ? rms : 1.0F);
    for (std::size_t b = 0; b < blocks; ++b)
      PutBlockScale(type, data.data() + b * bytes, d);
    return Make(type, rows, cols, std::move(data), experts);
  }

  DeviceMixer Mixer(const Config& c, bool inject, bool q8_inject,
                    TrunkStyle style) {
    DeviceMixer m;
    m.norm = F32(1, c.HcDim(), 1.0F, 0.1F);
    if (style == TrunkStyle::kGsqRco) {
      m.down = Half(GgmlType::kBF16, c.hc_low_rank, c.HcDim(),
                    1.0F / std::sqrt(static_cast<float>(c.HcDim())));
      m.up = Half(GgmlType::kBF16, c.HcDim(), c.hc_low_rank,
                  1.0F / std::sqrt(static_cast<float>(c.hc_low_rank)));
      if (inject)
        m.inject = Half(GgmlType::kBF16, c.hc_count, c.HcDim(), 0.01F);
    } else {
      m.down = Q8(c.hc_low_rank, c.HcDim());
      m.up = Q8(c.HcDim(), c.hc_low_rank);
      if (inject)
        m.inject = q8_inject ? Q8(c.hc_count, c.HcDim())
                             : F32(c.hc_count, c.HcDim(), 0.0F, 0.01F);
    }
    return m;
  }
  void Ffn(const Config& c, DeviceLayer& l, TrunkStyle style,
           std::uint32_t layer) {
    const auto e = c.num_experts;
    const bool gsq = style == TrunkStyle::kGsqRco;
    l.router = Half(gsq ? GgmlType::kBF16 : GgmlType::kF16, e + 1,
                    c.hidden_size, 0.05F);
    if (gsq) {
      // Even layers carry the xxs routed formats, odd layers the s formats.
      // The IQ4_XS shared pair fuses its gated vector launch; the Q6_K pair
      // keeps its separate dense waves at verification width.
      const bool xxs = layer % 2 == 0;
      l.ffn_gate_exps = Quant(xxs ? GgmlType::kIQ2_S : GgmlType::kIQ3_XXS,
                              c.expert_ff, c.hidden_size, e);
      l.ffn_up_exps = Quant(xxs ? GgmlType::kIQ2_S : GgmlType::kIQ3_XXS,
                            c.expert_ff, c.hidden_size, e);
      l.ffn_down_exps = Quant(xxs ? GgmlType::kQ2_0 : GgmlType::kIQ4_NL,
                              c.hidden_size, c.expert_ff, e);
      l.shexp_gate = Quant(xxs ? GgmlType::kIQ4_XS : GgmlType::kQ6_K,
                           c.shared_expert_ff, c.hidden_size);
      l.shexp_up = Quant(xxs ? GgmlType::kIQ4_XS : GgmlType::kQ6_K,
                         c.shared_expert_ff, c.hidden_size);
      l.shexp_down = Quant(xxs ? GgmlType::kQ2_0 : GgmlType::kIQ4_NL,
                           c.hidden_size, c.shared_expert_ff);
      return;
    }
    if (layer == 2) {  // one Q8_0 expert layer, as the artifact has
      l.ffn_gate_exps = Q8(c.expert_ff, c.hidden_size, e);
      l.ffn_up_exps = Q8(c.expert_ff, c.hidden_size, e);
      l.ffn_down_exps = Q8(c.hidden_size, c.expert_ff, e);
    } else {
      l.ffn_gate_exps = Q4K(c.expert_ff, c.hidden_size, e);
      l.ffn_up_exps = Q4K(c.expert_ff, c.hidden_size, e);
      l.ffn_down_exps = Q51(c.hidden_size, c.expert_ff, e);
    }
    l.shexp_gate = Q8(c.shared_expert_ff, c.hidden_size);
    l.shexp_up = Q8(c.shared_expert_ff, c.hidden_size);
    l.shexp_down = Q8(c.hidden_size, c.shared_expert_ff);
  }
  void Attention(const Config& c, DeviceLayer& l, TrunkStyle style) {
    if (style == TrunkStyle::kGsqRco) {
      // Mixed types keep Q, K and V separate, as the mixes' layers do.
      l.attn_q = Quant(GgmlType::kIQ4_XS, 2 * c.AttentionQDim(), c.hidden_size);
      l.attn_k = Quant(GgmlType::kQ6_K, c.AttentionKvDim(), c.hidden_size);
      l.attn_v = Quant(GgmlType::kIQ3_S, c.AttentionKvDim(), c.hidden_size);
      l.attn_out = Quant(GgmlType::kQ5_K, c.hidden_size, c.AttentionQDim());
    } else {
      l.attn_qkv =
          Q8(2 * c.AttentionQDim() + 2 * c.AttentionKvDim(), c.hidden_size);
      l.attn_out = Q8(c.hidden_size, c.AttentionQDim());
    }
    l.attn_q_norm = F32(1, c.head_dim, 1.0F, 0.1F);
    l.attn_k_norm = F32(1, c.head_dim, 1.0F, 0.1F);
    l.indexer_q = Half(GgmlType::kBF16, c.indexer_heads * c.indexer_head_dim,
                       c.hidden_size, 0.05F);
    l.indexer_k =
        Half(GgmlType::kBF16, c.indexer_head_dim, c.hidden_size, 0.05F);
    l.indexer_q_norm = F32(1, c.indexer_head_dim, 1.0F, 0.1F);
    l.indexer_k_norm = F32(1, c.indexer_head_dim, 1.0F, 0.1F);
  }
  void Linear(const Config& c, DeviceLayer& l, TrunkStyle style,
              std::uint32_t layer) {
    if (style == TrunkStyle::kGsqRco) {
      // Independently typed QKV and gate matrices: no stack, each in its own
      // quantization type.
      static constexpr GgmlType kQkv[3] = {GgmlType::kIQ3_S, GgmlType::kQ5_K,
                                           GgmlType::kIQ4_XS};
      static constexpr GgmlType kGate[3] = {GgmlType::kIQ4_XS, GgmlType::kIQ3_S,
                                            GgmlType::kQ4_K};
      static constexpr GgmlType kOut[3] = {GgmlType::kQ6_K, GgmlType::kIQ4_XS,
                                           GgmlType::kIQ3_S};
      l.ssm_qkv = Quant(kQkv[layer % 3], c.SsmConvChannels(), c.hidden_size);
      l.ssm_gate = Quant(kGate[layer % 3], c.SsmValueDim(), c.hidden_size);
      l.ssm_out = Quant(kOut[layer % 3], c.hidden_size, c.SsmValueDim());
      l.ssm_alpha_beta =
          Half(GgmlType::kBF16, 2 * c.ssm_num_v_heads, c.hidden_size, 0.05F);
    } else {
      l.ssm_in = Q8(c.SsmConvChannels() + c.SsmValueDim(), c.hidden_size);
      l.ssm_alpha_beta =
          Half(GgmlType::kF16, 2 * c.ssm_num_v_heads, c.hidden_size, 0.05F);
      l.ssm_out = Q8(c.hidden_size, c.SsmValueDim());
    }
    l.ssm_conv1d = F32(c.SsmConvChannels(), c.ssm_conv_kernel, 0.0F, 0.5F);
    l.ssm_dt = F32(1, c.ssm_num_v_heads, 0.5F, 0.5F);
    l.ssm_a = F32(1, c.ssm_num_v_heads, -1.25F, 0.75F);
    l.ssm_norm = F32(1, c.ssm_head_dim, 1.0F, 0.1F);
  }

  static std::unique_ptr<DeviceModel> Build(const Config& c, TrunkStyle style) {
    std::unique_ptr<DeviceModel> m(new DeviceModel());
    DeviceModelTestAccess b{*m, 0x5EED};
    m->config_ = c;
    const bool gsq = style == TrunkStyle::kGsqRco;
    m->token_embd_ =
        gsq ? b.Quant(GgmlType::kIQ3_S, c.vocab_size, c.hidden_size)
            : b.Q8(c.vocab_size, c.hidden_size, 1, 0.05F);
    m->output_ = gsq ? b.Quant(GgmlType::kQ6_K, c.vocab_size, c.hidden_size)
                     : b.Q8(c.vocab_size, c.hidden_size);
    m->output_bytes_ =
        std::size_t{c.vocab_size} *
        (gsq ? c.hidden_size / 256 * 210 : c.hidden_size / 32 * 34);
    m->hc_head_ = b.Mixer(c, false, false, style);
    for (std::uint32_t il = 0; il < c.num_layers; ++il) {
      DeviceLayer l;
      l.linear = c.IsLinearLayer(il);
      l.hc_attn = b.Mixer(c, true, false, style);
      l.hc_ffn = b.Mixer(c, true, false, style);
      if (l.linear) {
        b.Linear(c, l, style, il);
      } else {
        b.Attention(c, l, style);
      }
      if (c.IsPleLayer(il)) {
        l.ple_key =
            gsq ? b.Half(
                      GgmlType::kBF16, c.HcDim(), c.PleEmbeddingDim(),
                      1.0F / std::sqrt(static_cast<float>(c.PleEmbeddingDim())))
                : b.Q8(c.HcDim(), c.PleEmbeddingDim());
        l.ple_value = gsq ? b.Half(GgmlType::kBF16, c.hidden_size,
                                   c.PleEmbeddingDim(), 0.02F)
                          : b.Q8(c.hidden_size, c.PleEmbeddingDim());
        l.ple_norm_key = b.F32(1, c.HcDim(), 1.0F, 0.1F);
        l.ple_norm_query = b.F32(1, c.HcDim(), 1.0F, 0.1F);
        l.ple_norm_conv = b.F32(1, c.HcDim(), 1.0F, 0.1F);
        l.ple_conv1d =
            gsq ? b.Half(GgmlType::kF16, c.HcDim(), c.ple_conv_kernel, 0.5F)
                : b.F32(c.HcDim(), c.ple_conv_kernel, 0.0F, 0.5F);
      }
      b.Ffn(c, l, style, il);
      m->layers_.push_back(l);
    }
    // The draft sidecar is shared Q8_0 for both mixes.
    DeviceLayer& mtp = m->mtp_;
    mtp.hc_attn = b.Mixer(c, true, true, TrunkStyle::kQ8Control);
    mtp.hc_ffn = b.Mixer(c, true, true, TrunkStyle::kQ8Control);
    b.Attention(c, mtp, TrunkStyle::kQ8Control);
    b.Ffn(c, mtp, TrunkStyle::kQ8Control, 2);
    mtp.nextn_enorm = b.F32(1, c.hidden_size, 1.0F, 0.1F);
    mtp.nextn_hnorm = b.F32(1, c.HcDim(), 1.0F, 0.1F);
    mtp.nextn_fc_embedding = b.Q8(c.hidden_size, c.hidden_size);
    mtp.nextn_fc_hidden = b.Q8(c.hidden_size, c.hidden_size);
    mtp.nextn_head = b.Mixer(c, false, false, TrunkStyle::kQ8Control);
    m->has_mtp_ = true;
    if (hipDeviceSynchronize() != hipSuccess)
      throw std::runtime_error("synthetic model upload failed");
    return m;
  }
};

// Test-only access: production has no switch for disabling graphs or overlap.
struct ExecutorTestAccess {
  static void EagerNext(Session& session) { session.warmed_.clear(); }
  static std::size_t Graphs(const Session& session) {
    return session.graphs_.size();
  }
  static bool Capturing(const Executor& exec) { return exec.capturing_; }
  /// Forces a decode layout; returns the one Create calibrated.
  static bool SetDecodeOverlap(const Executor& exec, bool overlap) {
    return std::exchange(const_cast<Executor&>(exec).decode_overlap_, overlap);
  }
  static void RequireIdle(const Executor& exec) {
    for (const auto stream : {exec.stream_, exec.side_stream_}) {
      hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
      if (hipStreamIsCapturing(stream, &status) != hipSuccess ||
          status != hipStreamCaptureStatusNone)
        throw std::runtime_error("failed pass left a HIP capture active");
    }
    if (exec.capturing_ || exec.side_pending_)
      throw std::runtime_error("failed pass left capture or side work pending");
  }
  static void PoisonShared(const Executor& exec, std::uint32_t rows) {
    if (hipMemsetAsync(exec.s_.shexp_out, 0xff,
                       std::size_t{rows} * exec.config().hidden_size * 4,
                       exec.stream_) != hipSuccess)
      throw std::runtime_error("shared-expert poison failed");
  }
  static void CheckNormAllocation(const Executor& exec,
                                  std::uint32_t expected_pad) {
    void* base = nullptr;
    std::size_t bytes = 0;
    if (exec.hc_norm_pad_ != expected_pad ||
        hipMemGetAddressRange(&base, &bytes, exec.s_.xn_half) != hipSuccess ||
        bytes < std::size_t{exec.options_.max_batch} *
                    (exec.config().HcDim() + exec.hc_norm_pad_) *
                    sizeof(__half))
      throw std::runtime_error(
          "HC norm allocation does not include row padding");
  }
  static void CheckPaddedNormLayerHook(const Executor& exec) {
    std::string error;
    auto session =
        exec.CreateSession(core::SessionMode::kAutoregressive, 128, &error);
    if (!session)
      throw std::runtime_error(error);
    const auto pitch = exec.config().HcDim() + exec.hc_norm_pad_;
    const auto bytes =
        std::size_t{Executor::kHookParkedRows} * pitch * sizeof(__half);
    if (hipMemsetAsync(exec.s_.xn_half, 0x2a, bytes, exec.stream_) !=
        hipSuccess)
      throw std::runtime_error("padded norm initialization failed");
    exec.xn_half_ = true;
    exec.xn_half_pitch_ = pitch;
    exec.SetLayerHook({}, [&] {
      if (hipMemsetAsync(exec.s_.xn_half, 0x7b, bytes, exec.stream_) !=
          hipSuccess)
        throw std::runtime_error("peer norm overwrite failed");
      exec.xn_half_pitch_ = 0;
    });
    if (!exec.RunLayerHook(*session, &error))
      throw std::runtime_error(error);
    std::vector<std::uint8_t> actual(bytes);
    if (hipStreamSynchronize(exec.stream_) != hipSuccess ||
        hipMemcpy(actual.data(), exec.s_.xn_half, bytes,
                  hipMemcpyDeviceToHost) != hipSuccess)
      throw std::runtime_error("padded norm download failed");
    if (exec.xn_half_pitch_ != pitch ||
        !std::all_of(actual.begin(), actual.end(),
                     [](auto byte) { return byte == 0x2a; }))
      throw std::runtime_error("layer hook changed padded norm rows or pitch");
    exec.SetLayerHook({}, {});
    exec.xn_half_ = false;
    exec.xn_half_pitch_ = 0;
  }
  static void CheckMoeCapture(const Executor& exec, std::uint32_t rows,
                              bool throw_in_capture = false) {
    std::string error;
    auto session =
        exec.CreateSession(core::SessionMode::kAutoregressive, 128, &error);
    if (!session)
      throw std::runtime_error(error);
    const std::size_t count = std::size_t{rows} * exec.config().hidden_size;
    std::vector<float> input(count);
    for (std::size_t i = 0; i < count; ++i)
      input[i] = 0.1F * std::sin(static_cast<float>(i) * 0.13F);
    if (hipMemcpy(exec.s_.mixed, input.data(), count * 4,
                  hipMemcpyHostToDevice) != hipSuccess)
      throw std::runtime_error("MoE input upload failed");
    bool checked = false;
    const auto body = [&] {
      exec.DropReadyQ8();
      exec.q8t_src_ = nullptr;
      exec.half_src_ = nullptr;
      if (!exec.Moe(exec.model_->layers().front(), exec.s_.mixed,
                    exec.s_.block_out, rows, &error))
        throw std::runtime_error(error);
      if (exec.capturing_) {
        hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
        if (hipStreamIsCapturing(exec.side_stream_, &status) != hipSuccess ||
            status != (rows <= 8 && exec.decode_overlap_
                           ? hipStreamCaptureStatusActive
                           : hipStreamCaptureStatusNone))
          throw std::runtime_error("unexpected MoE overlap shape");
        checked = true;
        if (throw_in_capture)
          throw std::runtime_error("injected capture failure");
      }
      return true;
    };
    const auto read = [&](std::vector<float>& out) {
      if (hipStreamSynchronize(exec.stream_) != hipSuccess ||
          hipMemcpy(out.data(), exec.s_.block_out, count * 4,
                    hipMemcpyDeviceToHost) != hipSuccess)
        throw std::runtime_error("MoE output download failed");
    };
    // The first pass warms the shape eagerly; the second captures and
    // launches; the third replays. Comparing the eager and replayed block
    // output catches a graph that dropped the shared expert, which poisoning
    // its scratch makes visible.
    constexpr std::uint64_t key = std::uint64_t{1} << 63;
    std::vector<float> expected(count), actual(count);
    if (!exec.Run(*session, key, true, body, &error, false))
      throw std::runtime_error(error);
    read(expected);
    const auto main = exec.stream_;
    bool threw = false;
    try {
      if (!exec.Run(*session, key, true, body, &error, false))
        throw std::runtime_error(error);
    } catch (const std::runtime_error& e) {
      if (!throw_in_capture ||
          std::string_view(e.what()) != "injected capture failure")
        throw;
      threw = true;
    }
    if (!checked || threw != throw_in_capture || exec.stream_ != main)
      throw std::runtime_error("capture failure was not exercised or restored");
    RequireIdle(exec);
    if (throw_in_capture) {
      throw_in_capture = false;
      if (Graphs(*session) != 0 ||
          !exec.Run(*session, key, true, body, &error, false))
        throw std::runtime_error("capture did not recover after exception");
    }
    if (Graphs(*session) != 1)
      throw std::runtime_error("MoE capture did not install a usable graph");
    PoisonShared(exec, rows);
    if (!exec.Run(*session, key, true, body, &error, false))
      throw std::runtime_error(error);
    read(actual);
    if (std::memcmp(expected.data(), actual.data(), count * 4) != 0)
      throw std::runtime_error("MoE graph replay changed the block output");
    RequireIdle(exec);
    // A direct MoE probe does not run the combine that consumes this state.
    exec.moe_pending_ = false;
    exec.moe_f32_rows_ = false;
  }
};

}  // namespace gufo::models::qwen38_flash_next::rocm

namespace {

namespace qfn = gufo::models::qwen38_flash_next;
namespace rocm = qfn::rocm;
using gufo::core::GgmlType;
using rocm::Executor;
using rocm::Session;
using rocm::TrunkStyle;

qfn::Config SyntheticConfig() {
  qfn::Config c;
  c.num_layers = 4;  // three DeltaNet layers, then attention
  c.num_layers_all = 5;
  c.nextn_layers = 1;
  c.hidden_size = 2560;
  c.vocab_size = 8192;
  c.context_length = 4096;
  c.rms_eps = 1e-6F;
  c.hc_count = 4;
  c.hc_low_rank = 320;
  c.full_attention_interval = 4;
  c.num_heads = 24;
  c.num_kv_heads = 2;
  c.head_dim = 256;
  c.rotary_dim = 64;
  c.rope_theta = 1e7F;
  c.rope_sections = {11, 11, 10, 0};
  c.indexer_heads = 4;
  c.indexer_head_dim = 128;
  c.indexer_top_k = 2048;
  c.compress_ratio = 4;
  c.ssm_conv_kernel = 4;
  c.ssm_head_dim = 128;
  c.ssm_num_k_heads = 16;
  c.ssm_num_v_heads = 48;
  c.ssm_inner_size = 6144;
  c.num_experts = 16;
  c.num_experts_used = 10;
  c.expert_ff = 640;
  c.shared_expert_ff = 640;
  c.ple_layer = 1;  // after layer 0: graphs split into prefix and suffix
  c.ple_ngram_size = 3;
  c.ple_heads_per_ngram = 8;
  c.ple_heads = 16;
  c.ple_head_dim = 160;
  c.ple_conv_kernel = 4;
  c.ple_eos_token = 8191;
  c.ple_multipliers = {1000003ULL, 998244353ULL, 1000000007ULL};
  for (std::uint32_t h = 0; h < c.ple_heads; ++h) {
    c.ple_head_offsets[h] = h * 1024;
    c.ple_head_vocab[h] = 1021;
  }
  c.ple_rows = 16 * 1024;
  return c;
}

[[noreturn]] void Throw(const std::string& what, const std::string& error) {
  throw std::runtime_error(what + ": " + error);
}

std::int32_t Argmax(std::span<const float> row) {
  return static_cast<std::int32_t>(std::max_element(row.begin(), row.end()) -
                                   row.begin());
}

void RequireExact(std::span<const float> expected,
                  std::span<const float> actual, const std::string& what) {
  const auto finite = [](float v) { return std::isfinite(v); };
  if (expected.size() != actual.size() ||
      !std::all_of(expected.begin(), expected.end(), finite) ||
      !std::all_of(actual.begin(), actual.end(), finite) ||
      std::memcmp(expected.data(), actual.data(), expected.size_bytes()) != 0)
    throw std::runtime_error(what);
}

// Each width warms, captures and replays twice with different token histories.
// Poisoning the shared output prevents an earlier eager/reference pass from
// concealing a graph that omits the shared expert. The oracle is forced eager
// without adding a production dispatch switch.
void DecodeGraphChecks(const Executor& exec) {
  using Access = rocm::ExecutorTestAccess;
  const auto& c = exec.config();
  std::string error;
  // Both decode layouts must replay exactly, whichever one Create picked.
  const bool calibrated = Access::SetDecodeOverlap(exec, false);
  std::cout << "calibrated decode overlap=" << calibrated << '\n';
  for (const bool overlap : {false, true}) {
    Access::SetDecodeOverlap(exec, overlap);
    for (const auto prompt_size : {4U, c.indexer_top_k + 8}) {
      for (std::uint32_t width = 1; width <= 8; ++width) {
        auto eager = exec.CreateSession(gufo::core::SessionMode::kSpeculative,
                                        128, &error);
        auto replay = exec.CreateSession(gufo::core::SessionMode::kSpeculative,
                                         128, &error);
        if (!eager || !replay)
          Throw("graph sessions", error);
        std::vector<std::int32_t> prompt(prompt_size);
        for (std::uint32_t i = 0; i < prompt_size; ++i)
          prompt[i] = static_cast<std::int32_t>((i * 17 + 3) % c.vocab_size);
        for (auto* session : {eager.get(), replay.get()}) {
          if (!exec.Forward(*session, prompt, 0, nullptr,
                            Executor::ForwardMode::kPrefill, &error))
            Throw("graph prefill", error);
        }
        std::size_t graphs = 0;
        for (std::uint32_t pass = 0; pass < 4; ++pass) {
          std::vector<std::int32_t> tokens(width);
          for (std::uint32_t i = 0; i < width; ++i)
            tokens[i] = static_cast<std::int32_t>(
                (pass * 53 + i * 23 + prompt_size) % c.vocab_size);
          std::vector<float> expected(std::size_t{width} * c.vocab_size);
          std::vector<float> actual(expected.size());
          const auto mode = width == 1 ? Executor::ForwardMode::kDecode
                                       : Executor::ForwardMode::kVerify;
          Access::EagerNext(*eager);
          if (!exec.Forward(*eager, tokens, width, expected.data(), mode,
                            &error))
            Throw("eager graph oracle", error);
          Access::PoisonShared(exec, width);
          if (!exec.Forward(*replay, tokens, width, actual.data(), mode,
                            &error))
            Throw("graph candidate", error);
          RequireExact(
              expected, actual,
              "eager/replay logits differ: width=" + std::to_string(width) +
                  " pass=" + std::to_string(pass));
          const auto count = Access::Graphs(*replay);
          if ((pass == 0 && count != 0) || (pass == 1 && count == 0) ||
              (pass > 1 && count != graphs))
            throw std::runtime_error("fixture did not warm/capture/replay");
          graphs = count;
        }
        std::cout << "decode graph width=" << width
                  << " sparse=" << (prompt_size > c.indexer_top_k)
                  << " overlap=" << overlap << " eager/replay logits exact\n";
      }
    }
    for (const auto rows : {1U, 8U, 9U, 32U})
      Access::CheckMoeCapture(exec, rows);
    Access::CheckMoeCapture(exec, 1, true);
  }

  // Cancel the suffix capture after at least one layer has recorded MoE side
  // work (the overlapped layout is still forced). Prefix capture and
  // eager-before-capture execution stay enabled.
  auto failed =
      exec.CreateSession(gufo::core::SessionMode::kAutoregressive, 128, &error);
  auto peer =
      exec.CreateSession(gufo::core::SessionMode::kAutoregressive, 128, &error);
  if (!failed || !peer)
    Throw("capture cancellation sessions", error);
  const std::array<std::int32_t, 4> prompt{3, 7, 11, 19};
  std::vector<float> expected(c.vocab_size), actual(c.vocab_size);
  const std::int32_t token = 23;
  if (!exec.Forward(*failed, prompt, 0, nullptr,
                    Executor::ForwardMode::kPrefill, &error) ||
      !exec.Forward(*failed, std::span(&token, 1), 1, actual.data(),
                    Executor::ForwardMode::kDecode, &error))
    Throw("capture cancellation warmup", error);
  unsigned captured_checks = 0;
  failed->SetCancellationCheck(
      [&] { return Access::Capturing(exec) && ++captured_checks == 3; });
  if (exec.Forward(*failed, std::span(&token, 1), 1, actual.data(),
                   Executor::ForwardMode::kDecode, &error) ||
      captured_checks != 3)
    throw std::runtime_error("fixture did not cancel during capture");
  Access::RequireIdle(exec);
  failed->SetCancellationCheck({});
  failed->Reset();
  for (auto* session : {peer.get(), failed.get()}) {
    if (!exec.Forward(*session, prompt, 0, nullptr,
                      Executor::ForwardMode::kPrefill, &error))
      Throw("post-cancellation prefill", error);
  }
  if (!exec.Forward(*peer, std::span(&token, 1), 1, expected.data(),
                    Executor::ForwardMode::kDecode, &error) ||
      !exec.Forward(*failed, std::span(&token, 1), 1, actual.data(),
                    Executor::ForwardMode::kDecode, &error))
    Throw("post-cancellation decode", error);
  RequireExact(expected, actual, "cancelled capture changed reused executor");
  Access::RequireIdle(exec);
  Access::SetDecodeOverlap(exec, calibrated);
  std::cout << "capture exception/cancellation recovery exact\n";
}

const char* StyleName(TrunkStyle style) {
  return style == TrunkStyle::kGsqRco ? "gsq" : "q8";
}

/// Rollback positions around every eight-row segment boundary, the tail and
/// the full chain. For the 2,030-token prompt, 17 keeps a prefix before the
/// sparse indexer frontier and 18 keeps one past it.
std::vector<std::uint32_t> BoundaryKeeps(std::uint32_t n) {
  std::vector<std::uint32_t> keeps{1, 7, 8, 9, 16, 17, 18, 23};
  for (std::uint32_t k = 24; k < n; k += 8)
    keeps.push_back(k);
  keeps.push_back(n > 1 ? n - 1 : n);
  keeps.push_back(n);
  std::ranges::sort(keeps);
  keeps.erase(std::unique(keeps.begin(), keeps.end()), keeps.end());
  while (!keeps.empty() && keeps.back() > n)
    keeps.pop_back();
  return keeps;
}

/// A wide copy verification (ForwardWide) must give every row the logits of
/// consecutive verification passes of at most eight rows and of one-row
/// verification passes, and a rollback to any kept prefix must continue
/// exactly as those passes cut back at the same token. Chains cross segment
/// boundaries, end mid-segment and, from the longer prompt, cross the sparse
/// attention budget. Poison beyond the selected rows stays untouched.
int WideChecks(const Executor& exec, TrunkStyle style) {
  const auto vocab = exec.config().vocab_size;
  constexpr float kPoison = -12345.0F;
  const char* const name = StyleName(style);
  std::string error;
  int failures = 0;
  const auto prefill = [&](std::uint32_t length) {
    auto session =
        exec.CreateSession(gufo::core::SessionMode::kSpeculative, 2304, &error);
    if (!session)
      Throw("wide session", error);
    std::vector<std::int32_t> prompt(length);
    for (std::uint32_t t = 0; t < length; ++t)
      prompt[t] =
          static_cast<std::int32_t>((t * 2246822519U + 29U) % (vocab - 2));
    std::vector<float> logits(vocab);
    if (!exec.Forward(*session, prompt, 1, logits.data(),
                      Executor::ForwardMode::kPrefill, &error))
      Throw("wide prefill", error);
    return session;
  };
  // Three decode steps after the kept prefix, greedy from its last row.
  const auto continue_from = [&](Session& session, std::int32_t token) {
    std::vector<float> out;
    std::vector<float> logits(vocab);
    for (int step = 0; step < 3; ++step) {
      if (!exec.Forward(session, std::span<const std::int32_t>(&token, 1), 1,
                        logits.data(), Executor::ForwardMode::kDecode, &error))
        Throw("wide continuation", error);
      out.insert(out.end(), logits.begin(), logits.end());
      token = Argmax(logits);
    }
    return out;
  };
  // One-row verification passes over the same chain.
  const auto scalar_rows = [&](std::uint32_t prompt,
                               std::span<const std::int32_t> chain) {
    auto session = prefill(prompt);
    std::vector<float> rows;
    std::vector<float> logits(vocab);
    for (const auto token : chain) {
      if (!exec.Forward(*session, std::span<const std::int32_t>(&token, 1), 1,
                        logits.data(), Executor::ForwardMode::kVerify, &error))
        Throw("scalar verification", error);
      rows.insert(rows.end(), logits.begin(), logits.end());
    }
    return rows;
  };
  struct Case {
    std::uint32_t prompt;
    std::uint32_t rows;
    bool every_prefix;
  };
  // The short prompt rolls back to every kept prefix at every width; the
  // long one places the sparse-indexer frontier inside the chain and rolls
  // back around it at the two widths that cross it in different segments.
  constexpr Case kCases[] = {
      {320U, 64U, true}, {320U, 33U, true},   {320U, 32U, true},
      {320U, 20U, true}, {320U, 17U, true},   {320U, 16U, true},
      {320U, 9U, true},  {2030U, 64U, false}, {2030U, 33U, false}};
  for (const Case& test : kCases) {
    const auto prompt = test.prompt;
    const auto n = test.rows;
    {
      std::vector<std::int32_t> chain(n);
      for (std::uint32_t t = 0; t < n; ++t)
        chain[t] =
            static_cast<std::int32_t>((t * 40503U + prompt) % (vocab - 2));
      const auto scalar = scalar_rows(prompt, chain);
      // A degenerate fixture (zero weights or a collapsed residual) would
      // make every comparison below trivially true.
      bool varied = false;
      for (std::uint32_t t = 1; t < n; ++t)
        varied |= std::memcmp(scalar.data() + std::size_t{(t - 1)} * vocab,
                              scalar.data() + std::size_t{t} * vocab,
                              std::size_t{vocab} * sizeof(float)) != 0;
      if (!varied || !std::all_of(scalar.begin(), scalar.end(),
                                  [](float v) { return std::isfinite(v); }))
        Throw("degenerate scalar reference", name);
      std::vector<std::uint32_t> keeps;
      if (test.every_prefix) {
        for (std::uint32_t keep = 1; keep <= n; ++keep)
          keeps.push_back(keep);
      } else {
        keeps = BoundaryKeeps(n);
      }
      for (const std::uint32_t keep : keeps) {
        auto wide = prefill(prompt);
        std::vector<float> wide_rows(std::size_t{n} * vocab + 8, kPoison);
        if (!exec.ForwardWide(*wide, chain, &error) ||
            !exec.SelectBatchLogits(0, n, wide_rows.data(), &error))
          Throw("wide verification", error);
        const bool tail_ok =
            std::all_of(wide_rows.begin() + std::size_t{n} * vocab,
                        wide_rows.end(), [](float v) { return v == kPoison; });
        if (keep < n && !exec.Rollback(*wide, keep, &error))
          Throw("wide rollback", error);
        // Consecutive passes of at most eight rows, the one holding the
        // last kept token cut back to it.
        auto reference = prefill(prompt);
        std::vector<float> rows;
        for (std::uint32_t first = 0; first < keep; first += 8) {
          const auto count = std::min(8U, n - first);
          std::vector<float> logits(std::size_t{count} * vocab);
          if (!exec.Forward(
                  *reference,
                  std::span<const std::int32_t>(chain).subspan(first, count),
                  count, logits.data(), Executor::ForwardMode::kVerify,
                  &error) ||
              !exec.Rollback(*reference, std::min(count, keep - first), &error))
            Throw("reference verification", error);
          rows.insert(rows.end(), logits.begin(), logits.end());
        }
        const auto compared = std::min(rows.size(), std::size_t{n} * vocab);
        const bool same_rows = std::memcmp(rows.data(), wide_rows.data(),
                                           compared * sizeof(float)) == 0;
        const bool same_scalar = std::memcmp(scalar.data(), wide_rows.data(),
                                             compared * sizeof(float)) == 0;
        const auto next = Argmax(std::span<const float>(wide_rows).subspan(
            std::size_t{keep - 1} * vocab, vocab));
        const bool same_state =
            wide->position() == reference->position() &&
            continue_from(*wide, next) == continue_from(*reference, next);
        std::cout << name << " prompt=" << prompt << " rows=" << n
                  << " keep=" << keep << ": logits "
                  << (same_rows ? "exact" : "DIFFER") << ", scalar "
                  << (same_scalar ? "exact" : "DIFFERS") << ", continuation "
                  << (same_state ? "exact" : "DIFFERS") << ", tail "
                  << (tail_ok ? "clean" : "POISONED") << '\n';
        failures += !same_rows + !same_scalar + !same_state + !tail_ok;
      }
    }
  }
  return failures == 0 ? 0 : 1;
}

struct LedgerCall {
  std::string operation;
  gufo::core::GgmlType type;
  std::uint32_t rows;
  std::uint32_t matrix_rows;
  std::uint32_t matrix_cols;
};

void RecordLedgerCall(void* context, const Executor::BatchCall& call) {
  static_cast<std::vector<LedgerCall>*>(context)->push_back(
      {call.operation, call.type, call.rows, call.matrix_rows,
       call.matrix_cols});
}

/// The wide pass's own call ledger: dense, shared and routed work separately,
/// with the outer width and weight format of every call. The tiny replay
/// makes the shape accounting explicit for the optimization backlog.
int LedgerChecks(Executor& exec, TrunkStyle style) {
  const auto vocab = exec.config().vocab_size;
  const auto layers = exec.config().num_layers;
  const char* const name = StyleName(style);
  const bool gsq = style == TrunkStyle::kGsqRco;
  std::string error;
  auto session =
      exec.CreateSession(gufo::core::SessionMode::kSpeculative, 2304, &error);
  if (!session)
    Throw("ledger session", error);
  std::vector<std::int32_t> prompt(40);
  for (std::uint32_t t = 0; t < prompt.size(); ++t)
    prompt[t] =
        static_cast<std::int32_t>((t * 2654435761U + 17U) % (vocab - 2));
  std::vector<float> logits(vocab);
  if (!exec.Forward(*session, prompt, 1, logits.data(),
                    Executor::ForwardMode::kPrefill, &error))
    Throw("ledger prefill", error);
  const std::uint32_t rows = 33;
  std::vector<std::int32_t> chain(rows);
  for (std::uint32_t t = 0; t < rows; ++t)
    chain[t] = static_cast<std::int32_t>((t * 40503U + 11U) % (vocab - 2));
  std::vector<LedgerCall> calls;
  exec.SetBatchCallHook(RecordLedgerCall, &calls);
  const bool ok = exec.ForwardWide(*session, chain, &error);
  exec.SetBatchCallHook(nullptr, nullptr);
  if (!ok)
    Throw("ledger wide pass", error);

  struct Summary {
    std::string operation;
    gufo::core::GgmlType type;
    std::uint32_t rows;
    std::uint32_t matrix_rows;
    std::uint32_t matrix_cols;
    std::uint32_t count;
  };
  std::vector<Summary> summary;
  for (const auto& call : calls) {
    const auto same = [&](const Summary& s) {
      return s.operation == call.operation && s.type == call.type &&
             s.rows == call.rows && s.matrix_rows == call.matrix_rows &&
             s.matrix_cols == call.matrix_cols;
    };
    const auto it = std::find_if(summary.begin(), summary.end(), same);
    if (it == summary.end())
      summary.push_back({call.operation, call.type, call.rows, call.matrix_rows,
                         call.matrix_cols, 1});
    else
      ++it->count;
  }
  for (const auto& s : summary)
    std::cout << name << " ledger " << s.operation
              << " type=" << gufo::core::ToString(s.type) << " rows=" << s.rows
              << " m=" << s.matrix_rows << " k=" << s.matrix_cols
              << " calls=" << s.count << '\n';

  const auto matching = [&](const char* operation) {
    std::vector<const Summary*> out;
    for (const auto& s : summary)
      if (s.operation == operation)
        out.push_back(&s);
    return out;
  };
  const auto total_rows = [](const std::vector<const Summary*>& list) {
    std::uint64_t total = 0;
    for (const auto* s : list)
      total += std::uint64_t{s->count} * s->rows;
    return total;
  };
  const auto total_calls = [](const std::vector<const Summary*>& list) {
    std::uint32_t calls = 0;
    for (const auto* s : list)
      calls += s->count;
    return calls;
  };
  const auto has_type = [](const std::vector<const Summary*>& list,
                           gufo::core::GgmlType type) {
    return std::ranges::any_of(
        list, [&](const Summary* s) { return s->type == type; });
  };
  int failures = 0;
  const auto expect = [&](bool condition, const std::string& what) {
    if (!condition) {
      std::cerr << name << " ledger FAIL: " << what << '\n';
      ++failures;
    }
  };
  const auto wide = matching("wide");
  expect(wide.size() == 1 && wide[0]->rows == rows,
         "one wide entry at the requested width");
  const auto dense = matching("dense");
  expect(!dense.empty(), "dense work is recorded");
  expect(std::ranges::all_of(dense,
                             [&](const Summary* s) { return s->rows <= rows; }),
         "no dense projection exceeds the outer width");
  expect(std::any_of(dense.begin(), dense.end(),
                     [&](const Summary* s) { return s->rows == rows; }),
         "dense projections run at the outer width");
  const auto router = matching("router");
  expect(total_calls(router) == layers &&
             has_type(router, gsq ? GgmlType::kBF16 : GgmlType::kF16),
         "the router runs once per layer in its stored format");
  const auto shared_gate = matching("shared-gate-up");
  const auto shared_up = matching("shared-up");
  const auto shared_down = matching("shared-down");
  expect(total_calls(shared_gate) == layers &&
             total_calls(shared_up) == layers &&
             total_calls(shared_down) == layers,
         "shared-expert gate, up and down run once per layer");
  const auto routed_gate = matching("routed-gate-up");
  const auto routed_down = matching("routed-down");
  expect(!routed_gate.empty() && !routed_down.empty(),
         "routed work is recorded");
  expect(total_rows(routed_gate) == std::uint64_t{layers} * rows,
         "routed gate/up covers every layer's rows once");
  expect(total_rows(routed_down) ==
             std::uint64_t{layers} * rows * exec.config().num_experts_used,
         "routed down covers every layer's slot rows once");
  if (gsq) {
    expect(has_type(dense, GgmlType::kBF16), "BF16 hyper-connections record");
    expect(has_type(dense, GgmlType::kIQ3_S) &&
               has_type(dense, GgmlType::kIQ4_XS) &&
               has_type(dense, GgmlType::kQ6_K) &&
               has_type(dense, GgmlType::kQ5_K),
           "GSQ dense formats record");
    expect(has_type(routed_gate, GgmlType::kIQ2_S) &&
               has_type(routed_gate, GgmlType::kIQ3_XXS),
           "IQ2_S and IQ3_XXS routed gate/up record");
    expect(has_type(routed_down, GgmlType::kQ2_0) &&
               has_type(routed_down, GgmlType::kIQ4_NL),
           "Q2_0 and IQ4_NL routed down record");
    expect(std::ranges::all_of(
               routed_gate, [&](const Summary* s) { return s->rows <= rows; }),
           "GSQ routed calls do not exceed the outer width");
    expect(total_calls(routed_gate) >= layers,
           "every GSQ layer records routed work");
    expect(std::ranges::any_of(
               routed_gate, [&](const Summary* s) { return s->rows == rows; }),
           "the GSQ routed gate/up group spans the outer width");
    expect(std::ranges::any_of(routed_down,
                               [&](const Summary* s) {
                                 return s->rows ==
                                        rows * exec.config().num_experts_used;
                               }),
           "the GSQ routed down spans the outer slot rows");
  } else {
    expect(has_type(routed_gate, GgmlType::kQ4_K) &&
               has_type(routed_gate, GgmlType::kQ8_0),
           "the control's grouped and generic routed branches record");
    expect(std::ranges::any_of(routed_gate,
                               [&](const Summary* s) {
                                 return s->type == GgmlType::kQ4_K &&
                                        s->rows == rows;
                               }),
           "the control's Q4_K group spans the outer width");
  }
  std::cout << name << " ledger summary: dense calls=" << total_calls(dense)
            << " (" << dense.size()
            << " shapes), shared gate/up=" << total_calls(shared_gate)
            << ", router=" << total_calls(router)
            << ", routed gate/up calls=" << total_calls(routed_gate)
            << " rows=" << total_rows(routed_gate)
            << ", routed down calls=" << total_calls(routed_down)
            << " rows=" << total_rows(routed_down) << '\n';
  return failures == 0 ? 0 : 1;
}

void WriteNgramTable(const std::filesystem::path& path,
                     const qfn::Config& config) {
  // The n-gram rows come from a BF16 table file, as from the artifact.
  std::uint64_t state = 0xB16;
  const auto next = [&]() {
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  };
  std::vector<std::uint16_t> rows(config.ple_rows * config.ple_head_dim);
  for (auto& v : rows) {
    const float f =
        0.1F * (static_cast<float>(next() >> 40) / 8388608.0F - 1.0F);
    std::uint32_t bits = 0;
    std::memcpy(&bits, &f, 4);
    v = static_cast<std::uint16_t>(bits >> 16);
  }
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(rows.data()),
            static_cast<std::streamsize>(rows.size() * 2));
  if (!out)
    throw std::runtime_error("n-gram table write");
}

}  // namespace

int main(int argc, char** argv) {
  const bool graph_only =
      argc == 2 && std::string_view(argv[1]) == "--decode-graph-only";
  if (argc != 1 && !graph_only)
    return 2;
  try {
    auto config = SyntheticConfig();
    if (graph_only) {
      config.vocab_size = 256;
      config.ple_eos_token = 255;
      config.indexer_top_k = 64;
    }
    TemporaryDirectory temp("qfn-wide-verification");
    const auto ngram_path = temp.path() / "ngram.bin";
    WriteNgramTable(ngram_path, config);
    for (const auto style :
         {rocm::TrunkStyle::kQ8Control, rocm::TrunkStyle::kGsqRco}) {
      if (graph_only && style != rocm::TrunkStyle::kQ8Control)
        continue;
      std::cout << "== " << StyleName(style) << " fixture ==\n";
      auto model = rocm::DeviceModelTestAccess::Build(config, style);
      std::string error;
      if (!graph_only && !model->SetDraftVocab(2048, {8000, 8001}, &error))
        Throw("draft vocabulary", error);
      const auto file = gufo::platform::OpenReadFile(ngram_path);
      auto ngram =
          qfn::NgramTable::Open(file, 0, config.ple_rows, config.ple_head_dim,
                                gufo::core::GgmlType::kBF16, &error);
      gufo::platform::CloseFile(file);
      if (!ngram)
        Throw("n-gram table", error);
      // A whole 2,030-token prompt in one pass needs the wide scratch.
      auto exec = Executor::Create(*model, ngram.get(),
                                   {.max_batch = graph_only ? 128U : 2048U,
                                    .max_logit_rows = 8,
                                    .max_speculative = 8},
                                   &error);
      if (!exec)
        Throw("executor", error);
      rocm::ExecutorTestAccess::CheckNormAllocation(
          *exec, style == rocm::TrunkStyle::kGsqRco ? 32 : 0);
      if (style == rocm::TrunkStyle::kGsqRco)
        rocm::ExecutorTestAccess::CheckPaddedNormLayerHook(*exec);
      if (graph_only) {
        DecodeGraphChecks(*exec);
        return 0;
      }
      if (WideChecks(*exec, style) != 0)
        return 1;
      if (LedgerChecks(*exec, style) != 0)
        return 1;
    }
    std::cout << "wide verification matches eight-row and scalar passes\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
