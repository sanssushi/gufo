#include "src/models/qwen38_flash_next/weights.hpp"

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"

namespace qfn = gufo::models::qwen38_flash_next;

namespace {

using gufo::core::GgmlType;

/// Serializes a complete `qwen4exp` trunk GGUF image in memory. Payload
/// bytes are never written: the image is stored in a sparse file sized for
/// the reader's extent checks, and the loader only reads headers.
class Image {
public:
  void MetaString(const std::string& key, const std::string& value) {
    Text(key);
    Pod(std::uint32_t{8});
    Text(value);
    ++metadata_count_;
  }

  void MetaU32(const std::string& key, std::uint32_t value) {
    Text(key);
    Pod(std::uint32_t{4});
    Pod(value);
    ++metadata_count_;
  }

  void MetaF32(const std::string& key, float value) {
    Text(key);
    Pod(std::uint32_t{6});
    Pod(value);
    ++metadata_count_;
  }

  void MetaU64Array(const std::string& key,
                    const std::vector<std::uint64_t>& values) {
    Text(key);
    Pod(std::uint32_t{9});
    Pod(std::uint32_t{10});
    Pod(std::uint64_t{values.size()});
    for (const auto value : values)
      Pod(value);
    ++metadata_count_;
  }

  void Tensor(const std::string& name, std::vector<std::uint64_t> dims,
              GgmlType type) {
    tensors_.push_back({name, std::move(dims), type});
  }

  /// Retypes one tensor so a single role can be probed per fixture.
  void TensorType(const std::string& name, GgmlType type) {
    for (auto& tensor : tensors_) {
      if (tensor.name == name) {
        tensor.type = type;
        return;
      }
    }
    throw std::runtime_error("unknown fixture tensor " + name);
  }

  /// Payload bytes the largest tensor needs for the current type assignment.
  [[nodiscard]] std::size_t PayloadBytes() const {
    std::size_t total = 0;
    for (const auto& tensor : tensors_) {
      std::size_t elements = 1;
      for (const auto dim : tensor.dims)
        elements *= dim;
      total =
          std::max(total, gufo::quant::EncodedSizeBytes(tensor.type, elements));
    }
    return total;
  }

  /// Header bytes up to the 32-aligned payload base.
  [[nodiscard]] std::vector<std::uint8_t> Header() const {
    std::vector<std::uint8_t> out;
    const auto pod = [&out]<class T>(T value) {
      const auto begin = out.size();
      out.resize(begin + sizeof(value));
      std::memcpy(out.data() + begin, &value, sizeof(value));
    };
    const auto text = [&out, &pod](const std::string& value) {
      pod(std::uint64_t{value.size()});
      out.insert(out.end(), value.begin(), value.end());
    };
    pod(std::uint32_t{0x46554747});
    pod(std::uint32_t{3});
    pod(std::uint64_t{tensors_.size()});
    pod(std::uint64_t{metadata_count_});
    out.insert(out.end(), metadata_.begin(), metadata_.end());
    for (const auto& tensor : tensors_) {
      text(tensor.name);
      pod(std::uint32_t{static_cast<std::uint32_t>(tensor.dims.size())});
      for (const auto dim : tensor.dims)
        pod(dim);
      pod(std::uint32_t{static_cast<std::uint32_t>(tensor.type)});
      pod(std::uint64_t{0});
    }
    out.resize((out.size() + 31) / 32 * 32);
    return out;
  }

private:
  struct TensorInfo {
    std::string name;
    std::vector<std::uint64_t> dims;
    GgmlType type;
  };

  template<class T>
  void Pod(T value) {
    const auto begin = metadata_.size();
    metadata_.resize(begin + sizeof(value));
    std::memcpy(metadata_.data() + begin, &value, sizeof(value));
  }

  void Text(const std::string& value) {
    Pod(std::uint64_t{value.size()});
    metadata_.insert(metadata_.end(), value.begin(), value.end());
  }

  std::vector<std::uint8_t> metadata_;
  std::uint64_t metadata_count_{0};
  std::vector<TensorInfo> tensors_;
};

constexpr std::uint64_t kHidden = 2560;
constexpr std::uint64_t kHcDim = 10240;
constexpr std::uint64_t kLowRank = 320;
constexpr std::uint64_t kHcCount = 4;
constexpr std::uint64_t kSsmConvChannels = 10240;
constexpr std::uint64_t kSsmValueDim = 6144;
constexpr std::uint64_t kSsmHeads = 48;
constexpr std::uint64_t kExpertFf = 640;
constexpr std::uint64_t kExperts = 512;
constexpr std::uint64_t kVocab = 32;
constexpr std::uint64_t kPleDim = 2560;
constexpr std::uint64_t kPleHeadDim = 160;
constexpr std::uint64_t kPleRows = 16;

void AddMetadata(Image& image) {
  image.MetaString("general.architecture", "qwen4exp");
  for (const auto& [key, value] :
       std::initializer_list<std::pair<const char*, std::uint32_t>>{
           {"block_count", 48},
           {"embedding_length", 2560},
           {"context_length", 262144},
           {"hyper_connection.count", 4},
           {"hyper_connection.low_rank", 320},
           {"full_attention_interval", 4},
           {"attention.head_count", 24},
           {"attention.head_count_kv", 2},
           {"attention.key_length", 256},
           {"rope.dimension_count", 64},
           {"attention.indexer.head_count", 4},
           {"attention.indexer.key_length", 128},
           {"attention.indexer.top_k", 2048},
           {"ssm.conv_kernel", 4},
           {"ssm.state_size", 128},
           {"ssm.group_count", 16},
           {"ssm.time_step_rank", 48},
           {"ssm.inner_size", 6144},
           {"expert_count", 512},
           {"expert_used_count", 10},
           {"expert_feed_forward_length", 640},
           {"expert_shared_feed_forward_length", 640},
           {"ple.ngram_size", 3},
           {"ple.heads_per_ngram", 8},
           {"ple.conv_kernel", 4},
           {"ple.eos_token_id", 0},
           {"embedding_length_per_layer_input", 160}}) {
    image.MetaU32(std::string{"qwen4exp."} + key, value);
  }
  image.MetaF32("qwen4exp.attention.layer_norm_rms_epsilon", 1e-6F);
  image.MetaF32("qwen4exp.rope.freq_base", 1e7F);
  image.MetaU64Array("qwen4exp.rope.dimension_sections", {11, 11, 10, 0});
  std::vector<std::uint64_t> ratios(48);
  for (std::size_t i = 3; i < ratios.size(); i += 4)
    ratios[i] = 4;
  image.MetaU64Array("qwen4exp.attention.compress_ratios", ratios);
  image.MetaU64Array("qwen4exp.ple.layers", {1});
  image.MetaU64Array("qwen4exp.ple.layer_multipliers", {17, 29, 43});
  std::vector<std::uint64_t> offsets(16), vocab(16, 1);
  for (std::uint64_t i = 0; i < offsets.size(); ++i)
    offsets[i] = i;
  image.MetaU64Array("qwen4exp.ple.head_offsets", offsets);
  image.MetaU64Array("qwen4exp.ple.head_vocab_sizes", vocab);
}

void AddMixer(Image& image, const std::string& prefix, bool with_inject) {
  image.Tensor(prefix + "_norm.weight", {kHcDim, 1}, GgmlType::kF32);
  image.Tensor(prefix + "_down.weight", {kHcDim, kLowRank}, GgmlType::kQ8_0);
  image.Tensor(prefix + "_up.weight", {kLowRank, kHcDim}, GgmlType::kQ8_0);
  if (with_inject)
    image.Tensor(prefix + "_inject.weight", {kHcDim, kHcCount}, GgmlType::kF32);
}

/// Complete trunk tensor inventory with baseline types the loader accepts.
Image BaseFixture() {
  Image image;
  AddMetadata(image);
  image.Tensor("token_embd.weight", {kHidden, kVocab}, GgmlType::kQ8_0);
  image.Tensor("output.weight", {kHidden, kVocab}, GgmlType::kQ8_0);
  AddMixer(image, "output_hc", false);
  image.Tensor("per_layer_token_embd.weight", {kPleHeadDim, kPleRows},
               GgmlType::kIQ4_NL);
  for (std::uint32_t il = 0; il < 48; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    AddMixer(image, p + "hc_attn", true);
    AddMixer(image, p + "hc_ffn", true);
    if (((il + 1) % 4) != 0) {
      image.Tensor(p + "attn_qkv.weight", {kHidden, kSsmConvChannels},
                   GgmlType::kQ8_0);
      image.Tensor(p + "attn_gate.weight", {kHidden, kSsmValueDim},
                   GgmlType::kQ8_0);
      image.Tensor(p + "ssm_conv1d.weight", {4, kSsmConvChannels},
                   GgmlType::kF32);
      image.Tensor(p + "ssm_alpha.weight", {kHidden, kSsmHeads},
                   GgmlType::kF32);
      image.Tensor(p + "ssm_beta.weight", {kHidden, kSsmHeads}, GgmlType::kF32);
      image.Tensor(p + "ssm_dt.bias", {kSsmHeads, 1}, GgmlType::kF32);
      image.Tensor(p + "ssm_a", {kSsmHeads, 1}, GgmlType::kF32);
      image.Tensor(p + "ssm_norm.weight", {128, 1}, GgmlType::kF32);
      image.Tensor(p + "ssm_out.weight", {kSsmValueDim, kHidden},
                   GgmlType::kQ8_0);
    } else {
      image.Tensor(p + "attn_q.weight", {kHidden, 2 * 24 * 256},
                   GgmlType::kQ8_0);
      image.Tensor(p + "attn_k.weight", {kHidden, 512}, GgmlType::kQ8_0);
      image.Tensor(p + "attn_v.weight", {kHidden, 512}, GgmlType::kQ8_0);
      image.Tensor(p + "attn_output.weight", {24 * 256, kHidden},
                   GgmlType::kQ8_0);
      image.Tensor(p + "attn_q_norm.weight", {256, 1}, GgmlType::kF32);
      image.Tensor(p + "attn_k_norm.weight", {256, 1}, GgmlType::kF32);
      image.Tensor(p + "indexer.q_proj.weight", {kHidden, 512},
                   GgmlType::kQ8_0);
      image.Tensor(p + "indexer.k_proj.weight", {kHidden, 128},
                   GgmlType::kQ8_0);
      image.Tensor(p + "indexer.q_norm.weight", {128, 1}, GgmlType::kF32);
      image.Tensor(p + "indexer.k_norm.weight", {128, 1}, GgmlType::kF32);
    }
    if (il == 1) {
      image.Tensor(p + "ple_key.weight", {kPleDim, kHcDim}, GgmlType::kQ8_0);
      image.Tensor(p + "ple_value.weight", {kPleDim, kHidden}, GgmlType::kQ8_0);
      image.Tensor(p + "ple_norm_key.weight", {kHcDim, 1}, GgmlType::kF32);
      image.Tensor(p + "ple_norm_query.weight", {kHcDim, 1}, GgmlType::kF32);
      image.Tensor(p + "ple_norm_conv.weight", {kHcDim, 1}, GgmlType::kF32);
      image.Tensor(p + "ple_conv1d.weight", {4, kHcDim}, GgmlType::kF32);
    }
    image.Tensor(p + "ffn_gate_inp.weight", {kHidden, kExperts},
                 GgmlType::kF32);
    image.Tensor(p + "ffn_gate_exps.weight", {kHidden, kExpertFf, kExperts},
                 GgmlType::kIQ2_XXS);
    image.Tensor(p + "ffn_up_exps.weight", {kHidden, kExpertFf, kExperts},
                 GgmlType::kIQ2_XXS);
    image.Tensor(p + "ffn_down_exps.weight", {kExpertFf, kHidden, kExperts},
                 GgmlType::kQ2_0);
    image.Tensor(p + "ffn_gate_inp_shexp.weight", {kHidden, 1}, GgmlType::kF32);
    image.Tensor(p + "ffn_gate_shexp.weight", {kHidden, kExpertFf},
                 GgmlType::kQ8_0);
    image.Tensor(p + "ffn_up_shexp.weight", {kHidden, kExpertFf},
                 GgmlType::kQ8_0);
    image.Tensor(p + "ffn_down_shexp.weight", {kExpertFf, kHidden},
                 GgmlType::kQ2_0);
  }
  return image;
}

struct Case {
  bool bound{false};
  std::string error;
};

/// Writes the image as a sparse file and binds it. Every tensor payload lies
/// at offset zero, so one region sized for the largest tensor covers the
/// whole inventory.
Case Bind(const Image& image, const std::filesystem::path& path) {
  const auto header = image.Header();
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char*>(header.data()),
            static_cast<std::streamsize>(header.size()));
  out.close();
  if (truncate(path.c_str(),
               static_cast<off_t>(header.size() + image.PayloadBytes())) != 0)
    throw std::runtime_error("failed to size the fixture file");
  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  if (reader == nullptr)
    return {false, error};
  const auto weights = qfn::ModelWeights::Bind(*reader, &error);
  return {weights.has_value(), error};
}

std::string TypeName(GgmlType type) {
  return std::string(gufo::core::ToString(type));
}

/// Every storage type the parser accepts, so each role table is asserted
/// both for what it binds and for everything it refuses.
const std::vector<GgmlType>& Universe() {
  static const std::vector<GgmlType> types = {
      GgmlType::kF32,    GgmlType::kF16,     GgmlType::kQ4_0,
      GgmlType::kQ4_1,   GgmlType::kQ5_0,    GgmlType::kQ5_1,
      GgmlType::kQ8_0,   GgmlType::kQ8_1,    GgmlType::kQ2_K,
      GgmlType::kQ3_K,   GgmlType::kQ4_K,    GgmlType::kQ5_K,
      GgmlType::kQ6_K,   GgmlType::kQ8_K,    GgmlType::kIQ2_XXS,
      GgmlType::kIQ2_XS, GgmlType::kIQ3_XXS, GgmlType::kIQ4_NL,
      GgmlType::kIQ3_S,  GgmlType::kIQ2_S,   GgmlType::kIQ4_XS,
      GgmlType::kI32,    GgmlType::kBF16,    GgmlType::kQ2_0};
  return types;
}

struct Role {
  const char* tensor;
  std::vector<GgmlType> accepted;
};

/// Effective accepted types per tensor role: the union of the two GSQ-RCO
/// allocation maps with today's UD-Q4_K_XL set. Tensors whose `cols` are
/// not divisible by a quant block cannot bind that type, so the expert
/// rows list only the types their geometry allows, exactly as the released
/// maps use them.
const std::vector<Role>& Roles() {
  static const std::vector<Role> roles = {
      {"blk.0.attn_qkv.weight",
       {GgmlType::kQ8_0, GgmlType::kQ6_K, GgmlType::kQ5_K, GgmlType::kQ4_K,
        GgmlType::kIQ4_XS, GgmlType::kIQ3_S, GgmlType::kBF16, GgmlType::kF16,
        GgmlType::kF32}},
      {"blk.3.attn_q.weight",
       {GgmlType::kQ8_0, GgmlType::kQ6_K, GgmlType::kQ5_K, GgmlType::kQ4_K,
        GgmlType::kIQ4_XS, GgmlType::kIQ3_S, GgmlType::kBF16, GgmlType::kF16,
        GgmlType::kF32}},
      {"blk.0.ssm_out.weight",
       {GgmlType::kQ8_0, GgmlType::kQ6_K, GgmlType::kQ5_K, GgmlType::kQ4_K,
        GgmlType::kIQ4_XS, GgmlType::kIQ3_S, GgmlType::kBF16, GgmlType::kF16,
        GgmlType::kF32}},
      {"blk.1.ple_key.weight",
       {GgmlType::kQ8_0, GgmlType::kQ6_K, GgmlType::kQ5_K, GgmlType::kQ4_K,
        GgmlType::kIQ4_XS, GgmlType::kIQ3_S, GgmlType::kBF16, GgmlType::kF16,
        GgmlType::kF32}},
      {"blk.0.ffn_gate_exps.weight",
       {GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ6_K, GgmlType::kQ5_1,
        GgmlType::kQ8_0, GgmlType::kIQ4_NL, GgmlType::kQ2_0, GgmlType::kIQ2_XXS,
        GgmlType::kIQ2_XS, GgmlType::kIQ2_S, GgmlType::kIQ3_XXS,
        GgmlType::kIQ3_S, GgmlType::kIQ4_XS}},
      {"blk.0.ffn_down_exps.weight",
       {GgmlType::kQ5_1, GgmlType::kQ8_0, GgmlType::kIQ4_NL, GgmlType::kQ2_0}},
      {"blk.0.ffn_gate_shexp.weight",
       {GgmlType::kQ8_0, GgmlType::kQ6_K, GgmlType::kQ5_K, GgmlType::kQ4_K,
        GgmlType::kIQ4_XS, GgmlType::kIQ3_S, GgmlType::kIQ4_NL, GgmlType::kQ2_0,
        GgmlType::kBF16, GgmlType::kF16, GgmlType::kF32}},
      {"blk.0.ffn_down_shexp.weight",
       {GgmlType::kQ8_0, GgmlType::kIQ4_NL, GgmlType::kQ2_0, GgmlType::kBF16,
        GgmlType::kF16, GgmlType::kF32}},
      {"token_embd.weight",
       {GgmlType::kQ8_0, GgmlType::kIQ3_S, GgmlType::kIQ4_XS, GgmlType::kQ6_K,
        GgmlType::kBF16, GgmlType::kF16, GgmlType::kF32}},
      {"output.weight",
       {GgmlType::kQ8_0, GgmlType::kQ6_K, GgmlType::kQ5_K, GgmlType::kBF16,
        GgmlType::kF16, GgmlType::kF32}},
      {"blk.0.ffn_gate_inp.weight", {GgmlType::kF32, GgmlType::kBF16}},
      {"blk.0.ffn_gate_inp_shexp.weight", {GgmlType::kF32, GgmlType::kBF16}},
      {"blk.0.hc_attn_down.weight",
       {GgmlType::kQ8_0, GgmlType::kBF16, GgmlType::kF16, GgmlType::kF32}},
      {"blk.0.hc_attn_up.weight",
       {GgmlType::kQ8_0, GgmlType::kBF16, GgmlType::kF16, GgmlType::kF32}},
      {"blk.0.hc_attn_inject.weight",
       {GgmlType::kF32, GgmlType::kQ8_0, GgmlType::kBF16}},
      {"blk.0.ssm_alpha.weight",
       {GgmlType::kF32, GgmlType::kBF16, GgmlType::kQ8_0}},
      {"blk.1.ple_conv1d.weight", {GgmlType::kF32, GgmlType::kF16}},
      {"per_layer_token_embd.weight", {GgmlType::kIQ4_NL, GgmlType::kBF16}},
      {"blk.0.hc_attn_norm.weight", {GgmlType::kF32}},
      {"blk.3.attn_q_norm.weight", {GgmlType::kF32}},
  };
  return roles;
}

void CheckRoleTables(const Image& base, const std::filesystem::path& path) {
  for (const auto& role : Roles()) {
    for (const auto probe : Universe()) {
      Image image = base;
      image.TensorType(role.tensor, probe);
      const Case result = Bind(image, path);
      const bool expected =
          std::find(role.accepted.begin(), role.accepted.end(), probe) !=
          role.accepted.end();
      if (expected) {
        if (!result.bound) {
          throw std::runtime_error(std::string(role.tensor) + " rejected " +
                                   TypeName(probe) + ": " + result.error);
        }
        continue;
      }
      if (result.bound) {
        throw std::runtime_error(std::string(role.tensor) +
                                 " accepted out-of-role type " +
                                 TypeName(probe));
      }
      if (result.error.find(role.tensor) == std::string::npos) {
        throw std::runtime_error(std::string(role.tensor) + " " +
                                 TypeName(probe) + " failed without naming " +
                                 "the tensor: " + result.error);
      }
    }
    std::cout << "role " << role.tensor << ": " << role.accepted.size()
              << " accepted types pinned\n";
  }
}

/// One negative fixture per role class, matching the manual step-2 checks:
/// the loader must fail loudly and name the offending tensor.
void CheckRoleClassNegatives(const Image& base,
                             const std::filesystem::path& path) {
  for (const auto& [tensor, type] :
       std::initializer_list<std::pair<const char*, GgmlType>>{
           {"token_embd.weight", GgmlType::kQ2_0},
           {"output.weight", GgmlType::kIQ3_S},
           {"blk.0.attn_qkv.weight", GgmlType::kIQ2_S},
           {"blk.0.ffn_gate_shexp.weight", GgmlType::kIQ2_S}}) {
    Image image = base;
    image.TensorType(tensor, type);
    const Case result = Bind(image, path);
    if (result.bound) {
      throw std::runtime_error(std::string(tensor) + " accepted " +
                               TypeName(type));
    }
    if (result.error.find("has unsupported format") == std::string::npos ||
        result.error.find(tensor) == std::string::npos) {
      throw std::runtime_error(std::string(tensor) + " " + TypeName(type) +
                               " failed with: " + result.error);
    }
    std::cout << "negative " << tensor << " = " << TypeName(type)
              << " rejected: " << result.error << '\n';
  }
}

}  // namespace

int main() {
  try {
    char dir[] = "/tmp/gufo-weights-whitelist-XXXXXX";
    if (mkdtemp(dir) == nullptr)
      throw std::runtime_error("failed to create the fixture directory");
    const std::filesystem::path root(dir);
    const auto path = root / "model.gguf";
    const Image base = BaseFixture();
    CheckRoleTables(base, path);
    CheckRoleClassNegatives(base, path);
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::cout << "weights whitelist tables ok\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
