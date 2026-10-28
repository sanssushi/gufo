#include "src/core/quant/ggml_dequant.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace {

constexpr std::size_t kBlockElements = 256;

void ExpectNear(float actual, float expected, std::string_view message,
                float tolerance = 1.0e-5F) {
  if (std::abs(actual - expected) > tolerance) {
    std::cerr << "Assertion failed: " << message << " actual=" << actual
              << " expected=" << expected << '\n';
    std::exit(1);
  }
}

std::array<std::uint8_t, 144> MakeQ4Block(
    const std::array<std::uint8_t, 8>& scales,
    const std::array<std::uint8_t, 8>& minima) {
  std::array<std::uint8_t, 144> block{};
  block[0] = 0x00;
  block[1] = 0x3C;
  block[2] = 0x00;
  block[3] = 0x38;
  for (std::size_t index = 0; index < 4; ++index) {
    block[4 + index] = static_cast<std::uint8_t>(
        (scales[index] & 0x3FU) | ((scales[index + 4] >> 4U) << 6U));
    block[8 + index] = static_cast<std::uint8_t>(
        (minima[index] & 0x3FU) | ((minima[index + 4] >> 4U) << 6U));
    block[12 + index] = static_cast<std::uint8_t>(
        (scales[index + 4] & 0x0FU) | ((minima[index + 4] & 0x0FU) << 4U));
  }
  for (std::size_t pair = 0; pair < 4; ++pair) {
    for (std::size_t lane = 0; lane < 32; ++lane) {
      const auto low = static_cast<std::uint8_t>((lane + pair) & 0x0FU);
      const auto high = static_cast<std::uint8_t>((15U - lane + pair) & 0x0FU);
      block[16 + (pair * 32) + lane] =
          static_cast<std::uint8_t>(low | (high << 4U));
    }
  }
  return block;
}

std::array<std::uint8_t, 210> MakeQ6Block(
    const std::array<std::int8_t, 16>& scales) {
  std::array<std::uint8_t, 210> block{};
  const auto logical_quant = [](std::size_t index) {
    return static_cast<std::int8_t>(static_cast<int>((index * 7U) % 64U) - 32);
  };
  for (std::size_t half = 0; half < 2; ++half) {
    for (std::size_t lane = 0; lane < 32; ++lane) {
      std::array<std::uint8_t, 4> code{};
      for (std::size_t segment = 0; segment < code.size(); ++segment) {
        code[segment] = static_cast<std::uint8_t>(
            logical_quant((half * 128) + (segment * 32) + lane) + 32);
      }
      block[(half * 64) + lane] = static_cast<std::uint8_t>(
          (code[0] & 0x0FU) | ((code[2] & 0x0FU) << 4U));
      block[(half * 64) + 32 + lane] = static_cast<std::uint8_t>(
          (code[1] & 0x0FU) | ((code[3] & 0x0FU) << 4U));
      block[128 + (half * 32) + lane] = static_cast<std::uint8_t>(
          ((code[0] >> 4U) & 0x03U) | (((code[1] >> 4U) & 0x03U) << 2U) |
          (((code[2] >> 4U) & 0x03U) << 4U) |
          (((code[3] >> 4U) & 0x03U) << 6U));
    }
  }
  std::ranges::copy(scales, block.begin() + 192);
  block[208] = 0x00;
  block[209] = 0x3C;
  return block;
}

std::array<std::uint8_t, 110> MakeQ3Block(
    const std::array<std::int8_t, 16>& scales) {
  std::array<std::uint8_t, 110> block{};
  for (std::size_t index = 0; index < scales.size(); ++index) {
    const auto encoded = static_cast<std::uint8_t>(scales[index] + 32);
    if (index < 8) {
      block[96 + index] |= encoded & 0x0FU;
    } else {
      block[96 + index - 8] |=
          static_cast<std::uint8_t>((encoded & 0x0FU) << 4U);
    }
    block[104 + (index % 4)] |= static_cast<std::uint8_t>(
        ((encoded >> 4U) & 0x03U) << (2U * (index / 4)));
  }

  for (std::size_t index = 0; index < kBlockElements; ++index) {
    const std::size_t half = index / 128;
    const std::size_t within_half = index % 128;
    const std::size_t pair = within_half / 32;
    const std::size_t lane = within_half % 32;
    const auto quant =
        static_cast<std::int8_t>(static_cast<int>((index * 3U) % 8U) - 4);
    const auto low = static_cast<std::uint8_t>(quant < 0 ? quant + 4 : quant);
    block[32 + (half * 32) + lane] |=
        static_cast<std::uint8_t>(low << (2U * pair));
    if (quant >= 0) {
      block[lane] |= static_cast<std::uint8_t>(1U << ((half * 4) + pair));
    }
  }
  block[108] = 0x00;
  block[109] = 0x3C;
  return block;
}

void TestQ4() {
  const std::array<std::uint8_t, 8> scales{1, 7, 15, 31, 32, 47, 55, 63};
  const std::array<std::uint8_t, 8> minima{0, 1, 2, 3, 4, 5, 6, 7};
  const auto block = MakeQ4Block(scales, minima);
  std::array<float, kBlockElements> actual{};
  gufo::quant::DequantizeQ4_K(block.data(), actual.data(), actual.size());

  std::array<float, kBlockElements> expected{};
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const std::size_t group = index / 32;
    const std::size_t pair = group / 2;
    const std::size_t lane = index % 32;
    const auto quant =
        group % 2 == 0 ? static_cast<std::uint8_t>((lane + pair) & 0x0FU)
                       : static_cast<std::uint8_t>((15U - lane + pair) & 0x0FU);
    expected[index] = static_cast<float>(scales[group] * quant) -
                      (0.5F * static_cast<float>(minima[group]));
    ExpectNear(actual[index], expected[index], "Q4_K dequant value");
  }

  std::array<float, kBlockElements> vector{};
  std::iota(vector.begin(), vector.end(), -0.5F);
  const float expected_dot = std::inner_product(
      expected.begin(), expected.end(), vector.begin(), 0.0F);
  ExpectNear(gufo::quant::DotProductQ4_K(block.data(), vector, vector.size()),
             expected_dot, "Q4_K dot product");
}

void TestQ6() {
  const std::array<std::int8_t, 16> scales{-8, -7, -6, -5, -4, -3, -2, -1,
                                           1,  2,  3,  4,  5,  6,  7,  8};
  const auto block = MakeQ6Block(scales);
  std::array<float, kBlockElements> actual{};
  gufo::quant::DequantizeQ6_K(block.data(), actual.data(), actual.size());

  std::array<float, kBlockElements> expected{};
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const std::size_t half = index / 128;
    const std::size_t within_half = index % 128;
    const std::size_t segment = within_half / 32;
    const std::size_t lane = within_half % 32;
    const std::size_t scale_index = (half * 8) + (lane / 16) + (segment * 2);
    const auto quant =
        static_cast<std::int8_t>(static_cast<int>((index * 7U) % 64U) - 32);
    expected[index] =
        static_cast<float>(scales[scale_index]) * static_cast<float>(quant);
    ExpectNear(actual[index], expected[index], "Q6_K dequant value");
  }

  std::array<float, kBlockElements> vector{};
  std::iota(vector.begin(), vector.end(), -0.25F);
  const float expected_dot = std::inner_product(
      expected.begin(), expected.end(), vector.begin(), 0.0F);
  ExpectNear(gufo::quant::DotProductQ6_K(block.data(), vector, vector.size()),
             expected_dot, "Q6_K dot product");
}

void TestQ3() {
  const std::array<std::int8_t, 16> scales{-32, -27, -18, -9, -1, 0,  1,  7,
                                           12,  18,  23,  27, 29, 30, 31, -16};
  const auto block = MakeQ3Block(scales);
  std::array<float, kBlockElements> actual{};
  gufo::quant::DequantizeQ3_K(block.data(), actual.data(), actual.size());

  std::array<float, kBlockElements> expected{};
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const std::size_t half = index / 128;
    const std::size_t within_half = index % 128;
    const std::size_t scale_index =
        (half * 8) + ((within_half / 32) * 2) + ((within_half % 32) / 16);
    const auto quant =
        static_cast<std::int8_t>(static_cast<int>((index * 3U) % 8U) - 4);
    expected[index] =
        static_cast<float>(scales[scale_index]) * static_cast<float>(quant);
    ExpectNear(actual[index], expected[index], "Q3_K dequant value");
  }

  std::array<float, kBlockElements> vector{};
  std::iota(vector.begin(), vector.end(), 0.125F);
  const float expected_dot = std::inner_product(
      expected.begin(), expected.end(), vector.begin(), 0.0F);
  ExpectNear(gufo::quant::DotProductQ3_K(block.data(), vector, vector.size()),
             expected_dot, "Q3_K dot product");
}

void TestQ8() {
  struct Q8Block {
    float d;
    std::int8_t qs[256];
    std::int16_t bsums[16];
  };
  Q8Block block{};
  block.d = 0.00390625F;
  for (std::size_t i = 0; i < kBlockElements; ++i) {
    block.qs[i] = static_cast<std::int8_t>(static_cast<int>(i % 65) - 32);
    block.bsums[i / 16] += block.qs[i];
  }

  std::array<float, kBlockElements> actual{};
  gufo::quant::DequantizeQ8_K(&block, actual.data(), actual.size());

  std::array<float, kBlockElements> expected{};
  for (std::size_t index = 0; index < kBlockElements; ++index) {
    expected[index] = block.d * static_cast<float>(block.qs[index]);
    ExpectNear(actual[index], expected[index], "Q8_K dequant value");
  }

  std::array<float, kBlockElements> vector{};
  std::iota(vector.begin(), vector.end(), -0.75F);
  const float expected_dot = std::inner_product(
      expected.begin(), expected.end(), vector.begin(), 0.0F);
  ExpectNear(gufo::quant::DotProductQ8_K(&block, vector, vector.size()),
             expected_dot, "Q8_K dot product");
}

void TestQ8_0() {
  constexpr std::size_t kQ8_0Elements = 32;
  struct Q8_0Block {
    std::uint16_t d;
    std::int8_t qs[kQ8_0Elements];
  };
  static_assert(sizeof(Q8_0Block) == 34);
  Q8_0Block block{};
  block.d = 0x3800;  // fp16 0.5
  for (std::size_t i = 0; i < kQ8_0Elements; ++i) {
    block.qs[i] = static_cast<std::int8_t>(static_cast<int>(i % 65) - 32);
  }

  std::array<float, kQ8_0Elements> actual{};
  gufo::quant::DequantizeQ8_0(&block, actual.data(), actual.size());

  std::array<float, kQ8_0Elements> expected{};
  for (std::size_t index = 0; index < kQ8_0Elements; ++index) {
    expected[index] =
        gufo::quant::Fp16ToFloat(block.d) * static_cast<float>(block.qs[index]);
    ExpectNear(actual[index], expected[index], "Q8_0 dequant value");
  }

  std::array<float, kQ8_0Elements> vector{};
  std::iota(vector.begin(), vector.end(), -0.75F);
  const float expected_dot = std::inner_product(
      expected.begin(), expected.end(), vector.begin(), 0.0F);
  ExpectNear(gufo::quant::DotProductQ8_0(&block, vector, vector.size()),
             expected_dot, "Q8_0 dot product");
}

void TestQ5_K() {
  constexpr std::size_t kQ5Elements = 256;
  struct Q5Block {
    std::uint16_t d;
    std::uint16_t dmin;
    std::uint8_t scales[12];
    std::uint8_t qh[32];
    std::uint8_t qs[128];
  };
  static_assert(sizeof(Q5Block) == 176);

  // QuantizedRowBytes covers every packed projection/embedding format used by
  // Qwen, including both Q8 layouts.
  if (gufo::quant::QuantizedRowBytes(gufo::core::GgmlType::kQ5_K,
                                     kQ5Elements) != 176 ||
      gufo::quant::QuantizedRowBytes(gufo::core::GgmlType::kQ8_K, 256) != 292 ||
      gufo::quant::QuantizedRowBytes(gufo::core::GgmlType::kQ8_0, 32) != 34 ||
      gufo::quant::EncodedSizeBytes(gufo::core::GgmlType::kF32, 8) != 32 ||
      gufo::quant::EncodedSizeBytes(gufo::core::GgmlType::kBF16, 8) != 16 ||
      gufo::quant::EncodedSizeBytes(gufo::core::GgmlType::kQ8_K, 256) != 292) {
    std::cerr << "Assertion failed: QuantizedRowBytes contract\n";
    std::exit(1);
  }

  // Build a handcrafted block. scales[12] is chosen so GetQ4ScaleMin(0..7)
  // yields (scale=4, min=1) for every index: packed[0..3]=4, packed[4..7]=1,
  // packed[8..11]=0x14 (high-half scale/min encodings) -> sc=4, m=1 all 8.
  const auto make_block = [](bool high_bits) {
    Q5Block block{};
    block.d = 0x3400;     // fp16 0.25
    block.dmin = 0x3000;  // fp16 0.125
    for (std::size_t index = 0; index < 4; ++index) {
      block.scales[index] = 4;
      block.scales[index + 4] = 1;
      block.scales[index + 8] = 0x14;
    }
    // qs[gg*32+lane] nibbles: group 0 low nibble=1, group 1 -> 2, etc.
    for (std::size_t gg = 0; gg < 4; ++gg) {
      const auto nibble = static_cast<std::uint8_t>(gg + 1);
      const auto val = static_cast<std::uint8_t>(nibble | (nibble << 4U));
      for (std::size_t lane = 0; lane < 32; ++lane) {
        block.qs[(gg * 32) + lane] = val;
      }
    }
    if (high_bits) {
      for (std::size_t lane = 0; lane < 32; ++lane) {
        block.qh[lane] = 0xFF;
      }
    }
    return block;
  };

  const auto check_block = [](const Q5Block& block, bool high_bits) {
    std::array<float, kQ5Elements> actual{};
    gufo::quant::DequantizeQ5_K(&block, actual.data(), actual.size());

    std::array<float, kQ5Elements> expected{};
    for (std::size_t index = 0; index < kQ5Elements; ++index) {
      const std::size_t gg = index / 64;
      const auto quant = static_cast<float>(gg + 1 + (high_bits ? 16 : 0));
      // Fp16ToFloat(0x3400)=0.25, Fp16ToFloat(0x3000)=0.125, sc=4, m=1.
      expected[index] = quant - 0.125F;
      ExpectNear(actual[index], expected[index],
                 high_bits ? "Q5_K dequant (high bits)" : "Q5_K dequant value");
    }

    std::array<float, kQ5Elements> unit{};
    unit.fill(1.0F);
    const float expected_dot =
        std::accumulate(expected.begin(), expected.end(), 0.0F);
    ExpectNear(gufo::quant::DotProductQ5_K(&block, unit, unit.size()),
               expected_dot, "Q5_K dot product");
  };

  check_block(make_block(false), false);  // no high bits
  check_block(make_block(true), true);    // all high bits set
}

void TestQ2_0() {
  constexpr std::size_t kQ2Elements = 64;
  struct Q2Block {
    std::uint16_t d;
    std::uint8_t qs[16];
  };
  static_assert(sizeof(Q2Block) == 18);

  if (gufo::quant::QuantizedBlockElements(gufo::core::GgmlType::kQ2_0) !=
          kQ2Elements ||
      gufo::quant::QuantizedRowBytes(gufo::core::GgmlType::kQ2_0,
                                     kQ2Elements) != 18) {
    std::cerr << "Assertion failed: Q2_0 row-size contract\n";
    std::exit(1);
  }

  Q2Block block{};
  block.d = 0x3800;  // fp16 0.5
  for (std::size_t byte = 0; byte < 16; ++byte) {
    // Four codes per byte: 00=-1, 01=0, 10=+1, 11=+2.
    block.qs[byte] = static_cast<std::uint8_t>(
        ((byte * 4 + 0) % 4) | (((byte * 4 + 1) % 4) << 2U) |
        (((byte * 4 + 2) % 4) << 4U) | (((byte * 4 + 3) % 4) << 6U));
  }

  std::array<float, kQ2Elements> actual{};
  gufo::quant::DequantizeQ2_0(&block, actual.data(), actual.size());

  std::array<float, kQ2Elements> expected{};
  for (std::size_t index = 0; index < kQ2Elements; ++index) {
    const auto quant = static_cast<int>(index % 4);
    expected[index] = 0.5F * static_cast<float>(quant - 1);
    ExpectNear(actual[index], expected[index], "Q2_0 dequant value");
  }

  std::array<float, kQ2Elements> vector{};
  std::iota(vector.begin(), vector.end(), -0.5F);
  const float expected_dot = std::inner_product(
      expected.begin(), expected.end(), vector.begin(), 0.0F);
  ExpectNear(gufo::quant::DotProductQ2_0(&block, vector, vector.size()),
             expected_dot, "Q2_0 dot product");
}

void TestIQ2_XXS() {
  // Grid 0 packs the magnitude 8 eight times; grid 1 starts with 43. Sign
  // index 1 (ksigns[1] = 0x81) negates elements 0 and 7.
  std::array<std::uint8_t, 66> block{};
  block[0] = 0x00;
  block[1] = 0x3C;  // fp16 1.0
  for (std::size_t ib32 = 0; ib32 < 8; ++ib32) {
    for (std::size_t l = 0; l < 4; ++l) {
      block[2 + (ib32 * 8) + l] = l == 3 ? 1 : 0;
    }
    std::uint32_t signs_word = static_cast<std::uint32_t>(3 + ib32)
                               << 28U;  // db = 1*(0.5+s)*0.25
    for (std::size_t l = 0; l < 4; ++l) {
      if (ib32 % 2 == 1) {
        signs_word |= 1U << (7U * l);  // sign index 1
      }
    }
    std::memcpy(block.data() + 2 + (ib32 * 8) + 4, &signs_word,
                sizeof(signs_word));
  }

  std::array<float, kBlockElements> actual{};
  gufo::quant::DequantizeIQ2_XXS(block.data(), actual.data(), actual.size());

  std::array<float, kBlockElements> expected{};
  for (std::size_t index = 0; index < kBlockElements; ++index) {
    const std::size_t ib32 = index / 32;
    const std::size_t l = (index % 32) / 8;
    const std::size_t j = index % 8;
    const float db = (0.5F + static_cast<float>(3 + ib32)) * 0.25F;
    const float magnitude = l == 3 && j == 0 ? 43.0F : 8.0F;
    const bool negate = (ib32 % 2 == 1) && (j == 0 || j == 7);
    expected[index] = negate ? -db * magnitude : db * magnitude;
    ExpectNear(actual[index], expected[index], "IQ2_XXS dequant value");
  }

  std::array<float, kBlockElements> vector{};
  std::iota(vector.begin(), vector.end(), -0.25F);
  const float expected_dot = std::inner_product(
      expected.begin(), expected.end(), vector.begin(), 0.0F);
  ExpectNear(
      gufo::quant::DotProductIQ2_XXS(block.data(), vector, vector.size()),
      expected_dot, "IQ2_XXS dot product");

  // Discriminating block: grid index 2 and sign indices 42/16/5/0 exercise
  // the j >= 4 sign-mask bits that sign index 0/1 never reach. Magnitudes
  // come from ggml-common.h iq2xxs_grid.
  std::array<std::uint8_t, 66> rich{};
  rich[1] = 0x3C;  // fp16 1.0
  const std::array<std::uint8_t, 4> kGridIndex = {2, 1, 2, 1};
  const std::array<std::uint8_t, 4> kSignBytes = {0xAA, 0x90, 0x05, 0x00};
  // ksigns_iq2xs[42] = 0xAA, [16] = 0x90, [5] = 0x05, [0] = 0x00.
  const std::array<std::array<std::uint8_t, 8>, 4> kMagnitudes = {{
      {25, 25, 8, 8, 8, 8, 8, 8},  // iq2xxs_grid[2]
      {43, 8, 8, 8, 8, 8, 8, 8},   // iq2xxs_grid[1]
      {25, 25, 8, 8, 8, 8, 8, 8},  // iq2xxs_grid[2]
      {43, 8, 8, 8, 8, 8, 8, 8},   // iq2xxs_grid[1]
  }};
  for (std::size_t ib32 = 0; ib32 < 8; ++ib32) {
    for (std::size_t l = 0; l < 4; ++l) {
      rich[2 + (ib32 * 8) + l] = kGridIndex[l];
    }
    std::uint32_t signs_word = 2U << 28U;  // db = 1*(0.5+2)*0.25
    signs_word |= 42U | (16U << 7U) | (5U << 14U) | (0U << 21U);
    std::memcpy(rich.data() + 2 + (ib32 * 8) + 4, &signs_word,
                sizeof(signs_word));
  }

  gufo::quant::DequantizeIQ2_XXS(rich.data(), actual.data(), actual.size());
  for (std::size_t index = 0; index < kBlockElements; ++index) {
    const std::size_t l = (index % 32) / 8;
    const std::size_t j = index % 8;
    const float db = (0.5F + 2.0F) * 0.25F;
    const float magnitude = static_cast<float>(kMagnitudes[l][j]);
    const bool negate = (kSignBytes[l] & (1U << j)) != 0;
    expected[index] = negate ? -db * magnitude : db * magnitude;
    ExpectNear(actual[index], expected[index],
               "IQ2_XXS sign-index dequant value");
  }
  const float expected_rich_dot = std::inner_product(
      expected.begin(), expected.end(), vector.begin(), 0.0F);
  ExpectNear(gufo::quant::DotProductIQ2_XXS(rich.data(), vector, vector.size()),
             expected_rich_dot, "IQ2_XXS sign-index dot product");
}

void TestIQ2_XS() {
  // Code = grid index | sign index << 9; grid 0 magnitudes are all 8.
  std::array<std::uint8_t, 74> block{};
  block[0] = 0x00;
  block[1] = 0x3C;  // fp16 1.0
  for (std::size_t ib32 = 0; ib32 < 8; ++ib32) {
    for (std::size_t l = 0; l < 4; ++l) {
      const std::uint16_t code =
          static_cast<std::uint16_t>((ib32 + l) % 2 == 0 ? 0 : (1U << 9U));
      std::memcpy(block.data() + 2 + ((ib32 * 4) + l) * 2, &code, sizeof(code));
    }
    block[66 + ib32] = static_cast<std::uint8_t>(((1 + ib32) & 0x0FU) |
                                                 (((2 + ib32) & 0x0FU) << 4U));
  }

  std::array<float, kBlockElements> actual{};
  gufo::quant::DequantizeIQ2_XS(block.data(), actual.data(), actual.size());

  std::array<float, kBlockElements> expected{};
  for (std::size_t index = 0; index < kBlockElements; ++index) {
    const std::size_t ib32 = index / 32;
    const std::size_t l = (index % 32) / 8;
    const std::size_t j = index % 8;
    const int scale =
        l / 2 == 0 ? 1 + static_cast<int>(ib32) : 2 + static_cast<int>(ib32);
    const float db = (0.5F + static_cast<float>(scale)) * 0.25F;
    const bool negate = ((ib32 + l) % 2 == 1) && (j == 0 || j == 7);
    expected[index] = negate ? -db * 8.0F : db * 8.0F;
    ExpectNear(actual[index], expected[index], "IQ2_XS dequant value");
  }

  std::array<float, kBlockElements> vector{};
  std::iota(vector.begin(), vector.end(), 0.125F);
  const float expected_dot = std::inner_product(
      expected.begin(), expected.end(), vector.begin(), 0.0F);
  ExpectNear(gufo::quant::DotProductIQ2_XS(block.data(), vector, vector.size()),
             expected_dot, "IQ2_XS dot product");

  // Discriminating block: codes pair grid index 2 with sign indices 42/16/5/0
  // so the j >= 4 sign-mask bits are exercised. Magnitudes come from
  // ggml-common.h iq2xs_grid.
  std::array<std::uint8_t, 74> rich{};
  rich[1] = 0x3C;  // fp16 1.0
  const std::array<std::uint8_t, 4> kSignBytes = {0xAA, 0x90, 0x05, 0x00};
  // ksigns_iq2xs[42] = 0xAA, [16] = 0x90, [5] = 0x05, [0] = 0x00.
  const std::array<std::array<std::uint8_t, 8>, 4> kMagnitudes = {{
      {25, 25, 8, 8, 8, 8, 8, 8},  // iq2xs_grid[2], sign 42
      {25, 25, 8, 8, 8, 8, 8, 8},  // iq2xs_grid[2], sign 16
      {43, 8, 8, 8, 8, 8, 8, 8},   // iq2xs_grid[1], sign 5
      {25, 25, 8, 8, 8, 8, 8, 8},  // iq2xs_grid[2], sign 0
  }};
  const std::array<std::uint16_t, 4> kCodes = {
      static_cast<std::uint16_t>(2U | (42U << 9U)),
      static_cast<std::uint16_t>(2U | (16U << 9U)),
      static_cast<std::uint16_t>(1U | (5U << 9U)),
      static_cast<std::uint16_t>(2U | (0U << 9U)),
  };
  for (std::size_t ib32 = 0; ib32 < 8; ++ib32) {
    for (std::size_t l = 0; l < 4; ++l) {
      std::memcpy(rich.data() + 2 + ((ib32 * 4) + l) * 2, &kCodes[l],
                  sizeof(kCodes[l]));
    }
    rich[66 + ib32] = static_cast<std::uint8_t>(((1 + ib32) & 0x0FU) |
                                                (((2 + ib32) & 0x0FU) << 4U));
  }

  gufo::quant::DequantizeIQ2_XS(rich.data(), actual.data(), actual.size());
  for (std::size_t index = 0; index < kBlockElements; ++index) {
    const std::size_t ib32 = index / 32;
    const std::size_t l = (index % 32) / 8;
    const std::size_t j = index % 8;
    const int scale =
        l / 2 == 0 ? 1 + static_cast<int>(ib32) : 2 + static_cast<int>(ib32);
    const float db = (0.5F + static_cast<float>(scale)) * 0.25F;
    const float magnitude = static_cast<float>(kMagnitudes[l][j]);
    const bool negate = (kSignBytes[l] & (1U << j)) != 0;
    expected[index] = negate ? -db * magnitude : db * magnitude;
    ExpectNear(actual[index], expected[index],
               "IQ2_XS sign-index dequant value");
  }
  const float expected_rich_dot = std::inner_product(
      expected.begin(), expected.end(), vector.begin(), 0.0F);
  ExpectNear(gufo::quant::DotProductIQ2_XS(rich.data(), vector, vector.size()),
             expected_rich_dot, "IQ2_XS sign-index dot product");
}

void TestIQ2_S() {
  // Low grid byte 0 plus high bits 0 select grid 0; sign bytes live past the
  // 32 low grid bytes.
  std::array<std::uint8_t, 82> block{};
  block[0] = 0x00;
  block[1] = 0x3C;  // fp16 1.0
  for (std::size_t ib32 = 0; ib32 < 8; ++ib32) {
    for (std::size_t l = 0; l < 4; ++l) {
      block[2 + (ib32 * 4) + l] = 0;
      block[2 + 32 + (ib32 * 4) + l] =
          (ib32 + l) % 2 == 0 ? 0 : static_cast<std::uint8_t>(0x81U);
    }
    block[2 + 64 + ib32] = 0;  // no high grid bits
    block[2 + 72 + ib32] = static_cast<std::uint8_t>(
        ((1 + ib32) & 0x0FU) | (((2 + ib32) & 0x0FU) << 4U));
  }

  std::array<float, kBlockElements> actual{};
  gufo::quant::DequantizeIQ2_S(block.data(), actual.data(), actual.size());

  std::array<float, kBlockElements> expected{};
  for (std::size_t index = 0; index < kBlockElements; ++index) {
    const std::size_t ib32 = index / 32;
    const std::size_t l = (index % 32) / 8;
    const std::size_t j = index % 8;
    const int scale =
        l / 2 == 0 ? 1 + static_cast<int>(ib32) : 2 + static_cast<int>(ib32);
    const float db = (0.5F + static_cast<float>(scale)) * 0.25F;
    const bool negate = ((ib32 + l) % 2 == 1) && (j == 0 || j == 7);
    expected[index] = negate ? -db * 8.0F : db * 8.0F;
    ExpectNear(actual[index], expected[index], "IQ2_S dequant value");
  }

  std::array<float, kBlockElements> vector{};
  std::iota(vector.begin(), vector.end(), -0.75F);
  const float expected_dot = std::inner_product(
      expected.begin(), expected.end(), vector.begin(), 0.0F);
  ExpectNear(gufo::quant::DotProductIQ2_S(block.data(), vector, vector.size()),
             expected_dot, "IQ2_S dot product");

  // Discriminating block: qh supplies the 0x100/0x200/0x300 grid-index high
  // bits and sign bytes cover the j >= 4 mask bits. Magnitudes come from
  // ggml-common.h iq2s_grid.
  std::array<std::uint8_t, 82> rich{};
  rich[1] = 0x3C;  // fp16 1.0
  // qh = 0xE4 selects high bits 0,1,2,3 for l = 0..3; low bytes pick
  // iq2s_grid[0x00a], [0x100], [0x205], [0x303].
  const std::array<std::uint8_t, 4> kLowGrid = {0x0a, 0x00, 0x05, 0x03};
  const std::array<std::uint8_t, 4> kSignBytes = {0xAA, 0x90, 0x05, 0x00};
  // ksigns_iq2xs[42] = 0xAA, [16] = 0x90, [5] = 0x05, [0] = 0x00.
  const std::array<std::array<std::uint8_t, 8>, 4> kMagnitudes = {{
      {43, 8, 43, 8, 8, 8, 8, 8},      // iq2s_grid[0x00a]
      {25, 25, 25, 25, 25, 8, 25, 8},  // iq2s_grid[0x100]
      {8, 8, 25, 25, 25, 8, 8, 25},    // iq2s_grid[0x205]
      {8, 8, 43, 25, 8, 8, 43, 25},    // iq2s_grid[0x303]
  }};
  for (std::size_t ib32 = 0; ib32 < 8; ++ib32) {
    for (std::size_t l = 0; l < 4; ++l) {
      rich[2 + (ib32 * 4) + l] = kLowGrid[l];
      rich[2 + 32 + (ib32 * 4) + l] = kSignBytes[l];
    }
    rich[2 + 64 + ib32] = 0xE4;
    rich[2 + 72 + ib32] = static_cast<std::uint8_t>(
        ((1 + ib32) & 0x0FU) | (((2 + ib32) & 0x0FU) << 4U));
  }

  gufo::quant::DequantizeIQ2_S(rich.data(), actual.data(), actual.size());
  for (std::size_t index = 0; index < kBlockElements; ++index) {
    const std::size_t ib32 = index / 32;
    const std::size_t l = (index % 32) / 8;
    const std::size_t j = index % 8;
    const int scale =
        l / 2 == 0 ? 1 + static_cast<int>(ib32) : 2 + static_cast<int>(ib32);
    const float db = (0.5F + static_cast<float>(scale)) * 0.25F;
    const float magnitude = static_cast<float>(kMagnitudes[l][j]);
    const bool negate = (kSignBytes[l] & (1U << j)) != 0;
    expected[index] = negate ? -db * magnitude : db * magnitude;
    ExpectNear(actual[index], expected[index],
               "IQ2_S high-grid-bit dequant value");
  }
  const float expected_rich_dot = std::inner_product(
      expected.begin(), expected.end(), vector.begin(), 0.0F);
  ExpectNear(gufo::quant::DotProductIQ2_S(rich.data(), vector, vector.size()),
             expected_rich_dot, "IQ2_S high-grid-bit dot product");
}

void TestIQ3_XXS() {
  // Grid index 0 magnitudes are all 4; trailing words hold scale and signs.
  std::array<std::uint8_t, 98> block{};
  block[0] = 0x00;
  block[1] = 0x3C;  // fp16 1.0
  for (std::size_t ib32 = 0; ib32 < 8; ++ib32) {
    for (std::size_t l = 0; l < 8; ++l) {
      block[2 + (ib32 * 8) + l] = 0;
    }
    const std::uint32_t signs_word = static_cast<std::uint32_t>(1 + ib32)
                                     << 28U;  // db = 1*(0.5+s)*0.5
    std::memcpy(block.data() + 2 + 64 + (ib32 * 4), &signs_word,
                sizeof(signs_word));
  }

  std::array<float, kBlockElements> actual{};
  gufo::quant::DequantizeIQ3_XXS(block.data(), actual.data(), actual.size());

  std::array<float, kBlockElements> expected{};
  for (std::size_t index = 0; index < kBlockElements; ++index) {
    const std::size_t ib32 = index / 32;
    expected[index] = (0.5F + static_cast<float>(1 + ib32)) * 0.5F * 4.0F;
    ExpectNear(actual[index], expected[index], "IQ3_XXS dequant value");
  }

  std::array<float, kBlockElements> vector{};
  std::iota(vector.begin(), vector.end(), -0.125F);
  const float expected_dot = std::inner_product(
      expected.begin(), expected.end(), vector.begin(), 0.0F);
  ExpectNear(
      gufo::quant::DotProductIQ3_XXS(block.data(), vector, vector.size()),
      expected_dot, "IQ3_XXS dot product");

  // Discriminating block: distinct grid1/grid2 indices per group (j < 4 reads
  // grid1, j >= 4 grid2) and sign indices covering the j >= 4 mask bits.
  // Magnitudes come from ggml-common.h iq3xxs_grid.
  std::array<std::uint8_t, 98> rich{};
  rich[1] = 0x3C;  // fp16 1.0
  const std::array<std::uint8_t, 4> kSignBytes = {0xAA, 0x90, 0x05, 0x00};
  // ksigns_iq2xs[42] = 0xAA, [16] = 0x90, [5] = 0x05, [0] = 0x00.
  const std::array<std::array<std::uint8_t, 4>, 2> kMagnitudes = {{
      {62, 12, 4, 4},  // iq3xxs_grid[5], read for j < 4
      {20, 36, 4, 4},  // iq3xxs_grid[9], read for j >= 4
  }};
  for (std::size_t ib32 = 0; ib32 < 8; ++ib32) {
    for (std::size_t l = 0; l < 4; ++l) {
      rich[2 + (ib32 * 8) + (2 * l) + 0] = 5;  // grid1 index
      rich[2 + (ib32 * 8) + (2 * l) + 1] = 9;  // grid2 index
    }
    std::uint32_t signs_word = 2U << 28U;  // db = 1*(0.5+2)*0.5
    signs_word |= 42U | (16U << 7U) | (5U << 14U) | (0U << 21U);
    std::memcpy(rich.data() + 2 + 64 + (ib32 * 4), &signs_word,
                sizeof(signs_word));
  }

  gufo::quant::DequantizeIQ3_XXS(rich.data(), actual.data(), actual.size());
  for (std::size_t index = 0; index < kBlockElements; ++index) {
    const std::size_t l = (index % 32) / 8;
    const std::size_t j = index % 8;
    const float db = (0.5F + 2.0F) * 0.5F;
    const float magnitude = static_cast<float>(kMagnitudes[j / 4][j % 4]);
    const bool negate = (kSignBytes[l] & (1U << j)) != 0;
    expected[index] = negate ? -db * magnitude : db * magnitude;
    ExpectNear(actual[index], expected[index],
               "IQ3_XXS grid-split dequant value");
  }
  const float expected_rich_dot = std::inner_product(
      expected.begin(), expected.end(), vector.begin(), 0.0F);
  ExpectNear(gufo::quant::DotProductIQ3_XXS(rich.data(), vector, vector.size()),
             expected_rich_dot, "IQ3_XXS grid-split dot product");
}

void TestNewTypeRowSizes() {
  if (gufo::quant::QuantizedRowBytes(gufo::core::GgmlType::kIQ2_XXS, 256) !=
          66 ||
      gufo::quant::QuantizedRowBytes(gufo::core::GgmlType::kIQ2_XS, 256) !=
          74 ||
      gufo::quant::QuantizedRowBytes(gufo::core::GgmlType::kIQ2_S, 256) != 82 ||
      gufo::quant::QuantizedRowBytes(gufo::core::GgmlType::kIQ3_XXS, 256) !=
          98 ||
      gufo::quant::QuantizedRowBytes(gufo::core::GgmlType::kQ2_0, 64) != 18 ||
      gufo::quant::QuantizedBlockElements(gufo::core::GgmlType::kIQ2_XS) !=
          256 ||
      gufo::quant::QuantizedBlockElements(gufo::core::GgmlType::kIQ2_S) !=
          256 ||
      gufo::quant::QuantizedBlockElements(gufo::core::GgmlType::kIQ3_XXS) !=
          256) {
    std::cerr << "Assertion failed: new-type row-size contract\n";
    std::exit(1);
  }
}

struct ModelSample {
  std::size_t index;
  float value;
};
void TestModelRow(const gufo::core::GgufReader& reader, std::string_view name,
                  gufo::core::GgmlType expected_type, std::size_t expected_k,
                  std::span<const ModelSample> samples, double expected_dot) {
  const auto* tensor = reader.FindTensor(name);
  if (tensor == nullptr || tensor->data == nullptr ||
      tensor->dimensions.empty()) {
    std::cerr << "Assertion failed: missing model tensor " << name << '\n';
    std::exit(1);
  }
  if (tensor->type != expected_type ||
      tensor->dimensions.front() != expected_k) {
    std::cerr << "Assertion failed: model tensor contract " << name << '\n';
    std::exit(1);
  }

  std::vector<float> row(expected_k);
  switch (tensor->type) {
    case gufo::core::GgmlType::kQ4_K:
      gufo::quant::DequantizeQ4_K(tensor->data, row.data(), row.size());
      break;
    case gufo::core::GgmlType::kQ6_K:
      gufo::quant::DequantizeQ6_K(tensor->data, row.data(), row.size());
      break;
    case gufo::core::GgmlType::kQ2_0:
      gufo::quant::DequantizeQ2_0(tensor->data, row.data(), row.size());
      break;
    case gufo::core::GgmlType::kIQ2_XXS:
      gufo::quant::DequantizeIQ2_XXS(tensor->data, row.data(), row.size());
      break;
    case gufo::core::GgmlType::kIQ2_XS:
      gufo::quant::DequantizeIQ2_XS(tensor->data, row.data(), row.size());
      break;
    case gufo::core::GgmlType::kIQ2_S:
      gufo::quant::DequantizeIQ2_S(tensor->data, row.data(), row.size());
      break;
    case gufo::core::GgmlType::kIQ3_XXS:
      gufo::quant::DequantizeIQ3_XXS(tensor->data, row.data(), row.size());
      break;
    default:
      std::cerr << "Assertion failed: unsupported model tensor type " << name
                << '\n';
      std::exit(1);
  }

  for (const auto& sample : samples) {
    ExpectNear(row[sample.index], sample.value, "real GGUF dequant value",
               1.0e-8F);
  }

  std::vector<float> input(expected_k);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] =
        static_cast<float>(static_cast<int>(index % 257) - 128) / 64.0F;
  }
  float actual_dot = 0.0F;
  switch (tensor->type) {
    case gufo::core::GgmlType::kQ4_K:
      actual_dot =
          gufo::quant::DotProductQ4_K(tensor->data, input, input.size());
      break;
    case gufo::core::GgmlType::kQ6_K:
      actual_dot =
          gufo::quant::DotProductQ6_K(tensor->data, input, input.size());
      break;
    case gufo::core::GgmlType::kQ2_0:
      actual_dot =
          gufo::quant::DotProductQ2_0(tensor->data, input, input.size());
      break;
    case gufo::core::GgmlType::kIQ2_XXS:
      actual_dot =
          gufo::quant::DotProductIQ2_XXS(tensor->data, input, input.size());
      break;
    case gufo::core::GgmlType::kIQ2_XS:
      actual_dot =
          gufo::quant::DotProductIQ2_XS(tensor->data, input, input.size());
      break;
    case gufo::core::GgmlType::kIQ2_S:
      actual_dot =
          gufo::quant::DotProductIQ2_S(tensor->data, input, input.size());
      break;
    case gufo::core::GgmlType::kIQ3_XXS:
      actual_dot =
          gufo::quant::DotProductIQ3_XXS(tensor->data, input, input.size());
      break;
    default:
      std::cerr << "Assertion failed: unsupported model tensor type " << name
                << '\n';
      std::exit(1);
  }
  ExpectNear(actual_dot, static_cast<float>(expected_dot),
             "real GGUF dot product", 5.0e-4F);
}

void TestModelRows(const std::filesystem::path& model_path) {
  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
  if (reader == nullptr) {
    std::cerr << "Assertion failed: MTP GGUF opens: " << error << '\n';
    std::exit(1);
  }

  constexpr std::array<ModelSample, 16> kQ4Samples{{
      {0, 0.008269786834716797F},
      {1, -0.019530773162841797F},
      {15, 0.023714542388916016F},
      {31, -0.01335287094116211F},
      {32, 0.009994983673095703F},
      {63, 0.005248546600341797F},
      {64, -0.002658843994140625F},
      {127, 0.013089179992675781F},
      {128, -0.006126880645751953F},
      {255, -0.0026617050170898438F},
      {256, -0.002628028392791748F},
      {511, 0.013987362384796143F},
      {1023, -0.0033451318740844727F},
      {2047, 0.0010213851928710938F},
      {4095, -0.00015413761138916016F},
      {5119, -0.0033974647521972656F},
  }};
  TestModelRow(*reader, "blk.64.nextn.eh_proj.weight",
               gufo::core::GgmlType::kQ4_K, 10240, kQ4Samples,
               1.1511284159496427);

  constexpr std::array<ModelSample, 16> kQ6Samples{{
      {0, -0.0027894973754882812F},
      {1, -0.013947486877441406F},
      {15, -0.024175643920898438F},
      {31, 0.009633064270019531F},
      {32, -0.003979682922363281F},
      {63, 0.008864402770996094F},
      {64, 0.0030994415283203125F},
      {127, -0.006855964660644531F},
      {128, -0.008070945739746094F},
      {255, 0.011157989501953125F},
      {256, -0.03636360168457031F},
      {511, 0.024633407592773438F},
      {1023, -0.0008414983749389648F},
      {2047, -0.012941122055053711F},
      {4095, 0.02767181396484375F},
      {5119, -0.012220144271850586F},
  }};
  TestModelRow(*reader, "blk.64.attn_q.weight", gufo::core::GgmlType::kQ6_K,
               5120, kQ6Samples, 0.5780322588980198);
}

// Frozen rows for the GSQ-RCO mixes, computed once from an independent port
// of GSQHalo's ggml-quants.c dequantizers and pinned here so the oracle is
// reproducible in-repo. GUFO_GSQ_MODEL names shard 1 of either mix; the mix
// is identified by the type of blk.0.ffn_gate_exps.weight (IQ2_XS in the
// IQ3_XXS mix, IQ3_XXS in the IQ3_S mix) and only its rows are checked.
void TestGsqModelRows(const std::filesystem::path& model_path) {
  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
  if (reader == nullptr) {
    std::cerr << "Assertion failed: GSQ GGUF opens: " << error << '\n';
    std::exit(1);
  }
  const auto* sentinel = reader->FindTensor("blk.0.ffn_gate_exps.weight");
  if (sentinel == nullptr) {
    std::cerr << "Assertion failed: GSQ sentinel tensor missing\n";
    std::exit(1);
  }

  // blk.1.ffn_gate_exps.weight: IQ2_XXS in the IQ3_XXS mix.
  constexpr std::array<ModelSample, 16> kIq2xxsGate{{
      {0, 0.004555702209472656F},
      {1, -0.004555702209472656F},
      {15, -0.004555702209472656F},
      {31, -0.004555702209472656F},
      {32, -0.021606087684631348F},
      {63, 0.012561678886413574F},
      {64, 0.012561678886413574F},
      {127, -0.008307456970214844F},
      {128, 0.006163597106933594F},
      {255, 0.020936131477355957F},
      {256, 0.00545954704284668F},
      {511, 0.007386445999145508F},
      {512, 0.026938408613204956F},
      {1023, -0.0054132938385009766F},
      {1535, -0.018916279077529907F},
      {2047, 0.005580902099609375F},
  }};
  // blk.0.ffn_gate_exps.weight: IQ2_XS in the IQ3_XXS mix.
  constexpr std::array<ModelSample, 16> kIq2xsGate{{
      {0, -0.052041709423065186F},
      {1, 0.009682178497314453F},
      {15, -0.009682178497314453F},
      {31, 0.005309581756591797F},
      {32, -0.010736286640167236F},
      {63, -0.004060268402099609F},
      {64, 0.04532665014266968F},
      {127, -0.02049654722213745F},
      {128, 0.00905752182006836F},
      {255, 0.004060268402099609F},
      {256, 0.00861501693725586F},
      {511, 0.010211765766143799F},
      {512, 0.021450966596603394F},
      {1023, 0.0072479248046875F},
      {1535, -0.007086992263793945F},
      {2047, 0.005246400833129883F},
  }};
  // blk.35.ffn_gate_exps.weight: IQ3_XXS in the IQ3_XXS mix.
  constexpr std::array<ModelSample, 16> kIq3xxsGateXxs{{
      {0, 0.004320025444030762F},
      {1, 0.004320025444030762F},
      {15, -0.0669603943824768F},
      {31, 0.012960076332092285F},
      {32, 0.002926468849182129F},
      {63, -0.008779406547546387F},
      {64, -0.05400031805038452F},
      {127, 0.01658332347869873F},
      {128, 0.007943272590637207F},
      {255, -0.010451674461364746F},
      {256, 0.010491609573364258F},
      {511, 0.002743959426879883F},
      {512, -0.016475200653076172F},
      {1023, 0.012391388416290283F},
      {1535, -0.0329471230506897F},
      {2047, 0.030986666679382324F},
  }};
  // blk.12.ffn_gate_exps.weight: IQ2_S in the IQ3_XXS mix.
  constexpr std::array<ModelSample, 16> kIq2sGateXxs{{
      {0, 0.023066997528076172F},
      {1, -0.004291534423828125F},
      {15, 0.023066997528076172F},
      {31, -0.01519918441772461F},
      {32, 0.004291534423828125F},
      {63, -0.004291534423828125F},
      {64, -0.020563602447509766F},
      {127, -0.008869171142578125F},
      {128, 0.016987323760986328F},
      {255, 0.026142597198486328F},
      {256, 0.0062732696533203125F},
      {511, 0.0057277679443359375F},
      {512, 0.019203871488571167F},
      {1023, -0.010150671005249023F},
      {1535, -0.014325976371765137F},
      {2047, -0.005115032196044922F},
  }};
  // blk.1.ffn_down_exps.weight: Q2_0 in the IQ3_XXS mix.
  constexpr std::array<ModelSample, 16> kQ20DownXxs{{
      {0, -0.01092529296875F},
      {1, -0.0F},
      {3, -0.0F},
      {15, 0.01092529296875F},
      {31, -0.01092529296875F},
      {32, -0.0F},
      {63, -0.0F},
      {64, -0.0F},
      {127, 0.010467529296875F},
      {128, 0.0F},
      {255, -0.01103973388671875F},
      {256, -0.0F},
      {300, -0.0F},
      {511, 0.0F},
      {512, 0.0F},
      {639, -0.0F},
  }};
  // blk.0.ffn_gate_exps.weight: IQ3_XXS in the IQ3_S mix.
  constexpr std::array<ModelSample, 16> kIq3xxsGateS{{
      {0, -0.04188188910484314F},
      {1, 0.013510286808013916F},
      {15, -0.013510286808013916F},
      {31, 0.002702057361602783F},
      {32, -0.024718821048736572F},
      {63, -0.005704343318939209F},
      {64, 0.03877952694892883F},
      {127, -0.020715773105621338F},
      {128, 0.002702057361602783F},
      {255, 0.002101600170135498F},
      {256, 0.0027794837951660156F},
      {511, 0.008913516998291016F},
      {512, 0.007941663265228271F},
      {1023, 0.011259019374847412F},
      {1535, -0.0028843283653259277F},
      {2047, 0.002300083637237549F},
  }};
  // blk.1.ffn_gate_exps.weight: IQ2_S in the IQ3_S mix.
  constexpr std::array<ModelSample, 16> kIq2sGateS{{
      {0, 0.004652738571166992F},
      {1, -0.004652738571166992F},
      {15, -0.004652738571166992F},
      {31, -0.004652738571166992F},
      {32, -0.01260116696357727F},
      {63, 0.01453980803489685F},
      {64, 0.01066252589225769F},
      {127, -0.004652738571166992F},
      {128, 0.005273103713989258F},
      {255, 0.02423301339149475F},
      {256, 0.01467391848564148F},
      {511, 0.008804351091384888F},
      {512, 0.023837178945541382F},
      {1023, -0.004396319389343262F},
      {1535, -0.013690441846847534F},
      {2047, 0.005593299865722656F},
  }};
  // blk.1.ffn_down_exps.weight: Q2_0 in the IQ3_S mix.
  constexpr std::array<ModelSample, 16> kQ20DownS{{
      {0, -0.0109100341796875F},
      {1, -0.0F},
      {3, -0.0F},
      {15, 0.0109100341796875F},
      {31, -0.0109100341796875F},
      {32, -0.0109100341796875F},
      {63, -0.0F},
      {64, -0.0F},
      {127, 0.01036834716796875F},
      {128, 0.0F},
      {255, -0.009796142578125F},
      {256, -0.0F},
      {300, -0.0F},
      {511, 0.00896453857421875F},
      {512, 0.0F},
      {639, -0.0F},
  }};

  if (sentinel->type == gufo::core::GgmlType::kIQ2_XS) {
    TestModelRow(*reader, "blk.1.ffn_gate_exps.weight",
                 gufo::core::GgmlType::kIQ2_XXS, 2560, kIq2xxsGate,
                 1.7502189874649048);
    TestModelRow(*reader, "blk.0.ffn_gate_exps.weight",
                 gufo::core::GgmlType::kIQ2_XS, 2560, kIq2xsGate,
                 -0.019631277769804);
    TestModelRow(*reader, "blk.35.ffn_gate_exps.weight",
                 gufo::core::GgmlType::kIQ3_XXS, 2560, kIq3xxsGateXxs,
                 0.2210134118795395);
    TestModelRow(*reader, "blk.12.ffn_gate_exps.weight",
                 gufo::core::GgmlType::kIQ2_S, 2560, kIq2sGateXxs,
                 -0.031863581389188766);
    TestModelRow(*reader, "blk.1.ffn_down_exps.weight",
                 gufo::core::GgmlType::kQ2_0, 640, kQ20DownXxs,
                 0.7798882722854614);
  } else if (sentinel->type == gufo::core::GgmlType::kIQ3_XXS) {
    TestModelRow(*reader, "blk.0.ffn_gate_exps.weight",
                 gufo::core::GgmlType::kIQ3_XXS, 2560, kIq3xxsGateS,
                 -0.33292102813720703);
    TestModelRow(*reader, "blk.1.ffn_gate_exps.weight",
                 gufo::core::GgmlType::kIQ2_S, 2560, kIq2sGateS,
                 1.4763036966323853);
    TestModelRow(*reader, "blk.1.ffn_down_exps.weight",
                 gufo::core::GgmlType::kQ2_0, 640, kQ20DownS,
                 0.6908912658691406);
  } else {
    std::cerr << "Assertion failed: unknown GSQ mix (sentinel type "
              << static_cast<int>(sentinel->type) << ")\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  TestQ4();
  TestQ6();
  TestQ3();
  TestQ8();
  TestQ8_0();
  TestQ5_K();
  TestQ2_0();
  TestIQ2_XXS();
  TestIQ2_XS();
  TestIQ2_S();
  TestIQ3_XXS();
  TestNewTypeRowSizes();
  if (const char* model = std::getenv("GUFO_MTP_MODEL");
      model != nullptr && std::string_view(model).size() > 0) {
    TestModelRows(model);
  }
  if (const char* model = std::getenv("GUFO_GSQ_MODEL");
      model != nullptr && std::string_view(model).size() > 0) {
    TestGsqModelRows(model);
  }
  return 0;
}
