#pragma once

#include <format>
#include "weaver/dsp/code.h"
#include "weaver/types.h"
namespace weaver {
struct GPSCACode {
  static constexpr const f32 CHIP_RATE_HZ = 1.023e6;
  static constexpr const u32 CHIP_COUNT = 1023;
  static constexpr const u16 PRN_COUNT = 64;
  static constexpr const dsp::Modulation MODULATION = dsp::Modulation::BPSK;

  static constexpr const size_t CHIP_WORD_COUNT = 515;

  struct CAGenerator {
    static constexpr const std::array<std::pair<u8, u8>, 32> CA_CPSELS{
        {{1, 5}, {2, 6}, {3, 7}, {4, 8}, {0, 8}, {1, 9}, {0, 7}, {1, 8}, {2, 9}, {1, 2}, {2, 3},
         {4, 5}, {5, 6}, {6, 7}, {7, 8}, {8, 9}, {0, 3}, {1, 4}, {2, 5}, {3, 6}, {4, 7}, {5, 8},
         {0, 2}, {3, 5}, {4, 6}, {5, 7}, {6, 8}, {7, 9}, {0, 5}, {1, 6}, {2, 7}, {3, 8}}};
    static const u16 G1_MASK = 0b1000000100;  
    static const u16 G2_MASK = 0b1110100110;
    static const u16 REG_MASK = 0b1111111111;
    static const u16 G1_INIT = 0b1111111111;
    static const u16 G2_INIT = 0b1111111111;

    CAGenerator(u16 prn) {
      auto [psel_1, psel_2] = CA_CPSELS[prn - 1];
      psel_mask = (1 << psel_1) | (1 << psel_2);
      g1 = G1_INIT;
      g2 = G2_INIT;
    }

    bool operator*() const {
      bool g2i = __builtin_popcount(g2 & psel_mask) & 0x1;
      return g2i ^ (g1 >> 9) ^ 0x1;
    }

    CAGenerator& operator++() {
      g1 = ((g1 << 1) & REG_MASK) | (__builtin_popcount(g1 & G1_MASK) & 0x1);
      g2 = ((g2 << 1) & REG_MASK) | (__builtin_popcount(g2 & G2_MASK) & 0x1);
      return *this;
    }

    CAGenerator operator++(int) {
      CAGenerator gen = *this;
      ++(*this);
      return gen;
    }

    u16 g1, g2;
    u16 psel_mask;
  };

  static void gen_chips(u16 prn, std::span<u8> out) {
    CAGenerator gen(prn);

    u8 cur_word = 0;
    for (size_t bit_i = 0; bit_i < 8 * (out.size() + 1); bit_i++, gen++) {
      if ((bit_i % CHIP_COUNT) == 0)
        gen = CAGenerator(prn);
      if (((bit_i % 8) == 0) && (bit_i != 0)) {
        out[(bit_i / 8) - 1] = cur_word;
        cur_word = 0;
      }
      cur_word <<= 1;
      cur_word |= *gen;
    }
  }
};
}  // namespace weaver
