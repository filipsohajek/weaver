#pragma once

#include <bitset>
#include <format>

#include "weaver/dsp/code.h"
#include "weaver/signal.h"
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

struct LNAVDataDecoder : public NavDataDecoder {
  f64 symbol_period_s() const override { return 0.020; }

  void process_symbol(cp_f32 symbol) override {
    bool in_bit = symbol.real() > 0;
    //std::cout << std::format("lnav_decoder: symbol.real={}, bit={}\n", symbol.real(), u8(in_bit));
    cur_word <<= 1;
    if (flip_data) {
      in_bit = !in_bit;
    }
    cur_word.set(0, in_bit);

    if (!has_sync && !try_sync) {
      u32 preamble = cur_word.to_ulong() & 0x1ff;
      bool preamble_match = (preamble == 0x8b) | (preamble == 0x174);
      if (preamble_match) {
        std::cout << "found preamble, word=" << cur_word.to_string() << ", trying to sync\n";
        word_bit_count = 8;
        word_id = 0;
        try_sync = true;
        return;
      }
    } else {
      word_bit_count++;
      if (word_bit_count == 30) {
        process_word();
        word_bit_count = 0;
      }
    }
  }

  std::optional<TimeOfWeek> tow() const override {
    if (!has_sync || !sys_time.has_value())
      return std::nullopt;

    TimeOfWeek sys_time_out = {.sys = GNSSSystem::GPS, .tow = sys_time.value()};
    sys_time_out += (30 * word_id + word_bit_count) * symbol_period_s();

    return sys_time_out;
  }

  std::queue<NavMessage>& message_queue() override {
    return nav_msgs;
  }

private:
  void process_word() {
    //std::cout << "processing word=" << cur_word.to_string() << "\n";
    auto corr_word = check_word(cur_word);
    ecc_passed = corr_word.has_value();
    if (!ecc_passed) {
      subframe_ecc_passed = false;
      if (!has_sync && try_sync) {
        std::cout << "sync attempt failed, word=" << cur_word.to_string() << "\n";
        try_sync = false;
      }
      return;
    }

    switch (word_id) {
      case 0:
        process_tlm(corr_word.value());
        subframe_ecc_passed = true;
        break;
      case 1:
        process_how(corr_word.value());
        break;
      case 2:
      case 3:
      case 4:
      case 5:
      case 6:
      case 7:
      case 8:
      case 9:
        subframe_words[word_id - 2] = std::bitset<24>((corr_word->to_ulong() >> 6) & 0xffffff);
        break;
    }
    word_id++;
    if (word_id == 10) {
      if (subframe_ecc_passed) {
        sys_time = new_sys_time;
        process_subframe();
      }
      word_id = 0;
    }
  }

  void process_tlm(std::bitset<32> word) {
    u8 preamble = (word >> 22).to_ulong() & 0xff;
    if (preamble != 0x8b) {
      std::cout << "invalid TLM preamble - synchronization lost\n";
      has_sync = false;
      try_sync = false;
    }

    u16 message = static_cast<u16>((word >> 8).to_ulong() & 0x3fff);
    std::cout << std::format("TLM: preamble={}, message={}, integrity_status={}\n", preamble,
                             message, u32(word[7]));
  }

  void process_how(std::bitset<32> word) {
    u8 tail_zeros = static_cast<u16>(word.to_ulong() & 0x3);
    if (tail_zeros != 0) {
      std::cout << "HOW tail zeros flipped, possible phase inversion, flipping data\n";
      flip_data = !flip_data;
      cur_word.flip();
      // magic
    }

    try_sync = false;
    has_sync = true;

    new_sys_time = static_cast<f64>((word >> 13).to_ulong() & 0x1ffff) * 6;
    subframe_id = static_cast<u16>(((word >> 8).to_ulong()) & 0x7);
    std::cout << std::format(
        "HOW: tow_count={}, alert={}, anti_spoof={}, subframe_id={}, zeros={}\n", new_sys_time,
        u32(word[12]), u32(word[11]), subframe_id, tail_zeros);
  }

  void process_subframe() {
    switch (subframe_id) {
      case 1:
        process_sf1();
        break;
      case 2:
        process_sf2();
        break;
      case 3:
        process_sf3();
        break;
    }
  }

  void check_ephemeris() {
    if (((eph_issues[0] & 0xff) != eph_issues[1]) || (eph_issues[1] != eph_issues[2]))
      return;
    nav_msgs.emplace(ephemeris);
    eph_issues[0] = -1;
    eph_issues[1] = -1;
    eph_issues[2] = -1;
  }

  void process_sf1() {
    ephemeris.week_nr = (subframe_words[0] >> 14).to_ulong() & 0x3fff;
    ephemeris.clock_params.issue = subframe_words[0].to_ulong() & 0x3;
    ephemeris.clock_params.issue <<= 8;
    ephemeris.clock_params.issue |= (subframe_words[5] >> 16).to_ulong() & 0xff;
    ephemeris.clock_params.group_delay =
        std::bit_cast<i8>(static_cast<u8>(subframe_words[4].to_ulong() & 0xff)) * std::pow(2, -31);

    ephemeris.clock_params.ref_tow = {
        .sys = GNSSSystem::GPS,
        .tow = static_cast<f64>((subframe_words[5].to_ulong() & 0xffff) * 16)};
    ephemeris.clock_params.frequency_drift =
        std::bit_cast<i8>(static_cast<u8>((subframe_words[6] >> 16).to_ulong() & 0xff)) *
        std::pow(2, -55);
    ephemeris.clock_params.drift =
        std::bit_cast<i16>(static_cast<u16>(subframe_words[6].to_ulong() & 0xffff)) *
        std::pow(2, -43);
    ephemeris.clock_params.offset =
        static_cast<f64>(sign_extend<22>((subframe_words[7].to_ulong() >> 2) & 0x3fffff)) *
        std::pow(2, -31);

    eph_issues[0] = ephemeris.clock_params.issue;
    check_ephemeris();
    std::cout << std::format(
        "LNAV subframe 1: week_nr={}, issue={}, group_delay={}, ref_sys_time={}, freq_drift={}, "
        "drift={}, offset={}\n",
        ephemeris.week_nr, ephemeris.clock_params.issue, ephemeris.clock_params.group_delay,
        ephemeris.clock_params.ref_tow.tow, ephemeris.clock_params.frequency_drift,
        ephemeris.clock_params.drift, ephemeris.clock_params.offset);
  }

  void process_sf2() {
    u32 M_0_u32 = static_cast<u32>(subframe_words[1].to_ulong() & 0xff) << 24;
    M_0_u32 |= static_cast<u32>(subframe_words[2].to_ulong() & 0xffffff);
    ephemeris.mean_anomaly = std::numbers::pi * std::bit_cast<i32>(M_0_u32) * std::pow(2, -31);
    ephemeris.mean_motion_diff =
        std::numbers::pi *
        std::bit_cast<i16>(static_cast<u16>((subframe_words[1] >> 8).to_ulong() & 0xffff)) *
        std::pow(2, -43);
    u32 e_u32 = (static_cast<u32>(subframe_words[3].to_ulong() & 0xff) << 24) |
                static_cast<u32>(subframe_words[4].to_ulong() & 0xffffff);
    ephemeris.eccentricity = e_u32 * std::pow(2, -33);
    u32 sqrt_A_u32 = (static_cast<u32>(subframe_words[5].to_ulong() & 0xff) << 24) |
                     static_cast<u32>(subframe_words[6].to_ulong() & 0xffffff);
    ephemeris.semmaj_axis_sqrt = sqrt_A_u32 * std::pow(2, -19);
    ephemeris.ref_tow = {
      .sys = GNSSSystem::GPS,
      .tow = static_cast<f64>((subframe_words[7] >> 8).to_ulong() & 0xffff) * 16
    };
    ephemeris.orbit_r_sin_corr =
        std::bit_cast<i16>(static_cast<u16>(subframe_words[0].to_ulong() & 0xffff)) *
        std::pow(2, -5);
    ephemeris.arg_lat_sin_corr =
        std::bit_cast<i16>(static_cast<u16>((subframe_words[5] >> 8).to_ulong() & 0xffff)) *
        std::pow(2, -29);
    ephemeris.arg_lat_cos_corr =
        std::bit_cast<i16>(static_cast<u16>((subframe_words[3] >> 8).to_ulong() & 0xffff)) *
        std::pow(2, -29);

    u8 iode = (subframe_words[0] >> 16).to_ulong() & 0xff;
    eph_issues[1] = iode;
    check_ephemeris();
    std::cout << std::format(
        "LNAV subframe 2: mean_anomaly={}, mean_motion_diff={}, eccentricity={}, "
        "semmaj_axis_sqrt={}, ref_sys_time={}, orbit_r_sin_corr={}, arg_lat_sin_corr={}, "
        "arg_lat_sin_corr={}, iode={}\n",
        ephemeris.mean_anomaly, ephemeris.mean_motion_diff, ephemeris.eccentricity, ephemeris.semmaj_axis_sqrt, ephemeris.ref_tow.tow,
        ephemeris.orbit_r_sin_corr, ephemeris.arg_lat_sin_corr, ephemeris.arg_lat_cos_corr, iode);
  }

  void process_sf3() {
    ephemeris.inc_angle_cos_corr =
        std::bit_cast<i16>(static_cast<u16>((subframe_words[0] >> 8).to_ulong() & 0xffff)) *
        std::pow(2, -29);
    u32 Omega_0_u32 = (subframe_words[0].to_ulong() & 0xff);
    Omega_0_u32 <<= 24;
    Omega_0_u32 |= subframe_words[1].to_ulong() & 0xffffff;
    ephemeris.asc_node_lon = std::numbers::pi * std::bit_cast<i32>(Omega_0_u32) * std::pow(2, -31);
    ephemeris.inc_angle_sin_corr =
        std::bit_cast<i16>(static_cast<u16>((subframe_words[2] >> 8).to_ulong() & 0xffff)) *
        std::pow(2, -29);
    u32 i_0_u32 = subframe_words[2].to_ulong() & 0xff;
    i_0_u32 <<= 24;
    i_0_u32 |= subframe_words[3].to_ulong() & 0xffffff;
    ephemeris.inc_angle = std::numbers::pi * std::bit_cast<i32>(i_0_u32) * std::pow(2, -31);
    ephemeris.orbit_r_cos_corr =
        std::bit_cast<i16>(static_cast<u16>((subframe_words[4] >> 8).to_ulong() & 0xffff)) *
        std::pow(2, -5);

    u32 omega_u32 = subframe_words[4].to_ulong() & 0xff;
    omega_u32 <<= 24;
    omega_u32 |= subframe_words[5].to_ulong() & 0xffffff;
    ephemeris.arg_perigee = std::numbers::pi * std::bit_cast<i32>(omega_u32) * std::pow(2, -31);

    ephemeris.right_asc_rate = std::numbers::pi *
                         static_cast<f64>(sign_extend<24>(subframe_words[6].to_ulong())) *
                         std::pow(2, -43);
    ephemeris.inc_angle_rate =
        std::numbers::pi *
        static_cast<f64>(sign_extend<14>((subframe_words[7].to_ulong() >> 2) & 0x3fff)) *
        std::pow(2, -43);

    u8 iode = (subframe_words[7] >> 16).to_ulong() & 0xff;
    eph_issues[2] = iode;
    check_ephemeris();
    std::cout << std::format(
        "LNAV subframe 3: inc_angle_cos_corr={}, asc_node_lon={}, inc_angle_sin_corr={}, "
        "inc_angle={}, orbit_r_cos_corr={}, arg_perigee={}, right_asc_rate={}, inc_angle_rate={}\n",
        ephemeris.inc_angle_cos_corr, ephemeris.asc_node_lon, ephemeris.inc_angle_sin_corr, ephemeris.inc_angle, ephemeris.orbit_r_cos_corr,
        ephemeris.arg_perigee, ephemeris.right_asc_rate, ephemeris.inc_angle_rate);
  }

  std::optional<std::bitset<32>> check_word(std::bitset<32> word) const {
    if (word[30]) {
      for (int i = 6; i <= 29; i++) {
        word.flip(i);
      }
    }

    bool ecc_failed = false;
    for (int i = 0; i < 6; i++) {
      std::bitset<32> mask_bitset(ECC_MASKS[i]);
      bool parity = ((mask_bitset & word).count() % 2 == 1);

      if (parity != word[5 - i]) {
        std::cout << std::format("ECC fail @ {}, word={}, mask={}, masked={}, count={}\n", 25 + i,
                                 word.to_string(), mask_bitset.to_string(),
                                 (mask_bitset & word).to_string(), (mask_bitset & word).count());
        ecc_failed = true;
      }
    }

    if (ecc_failed) {
      return std::nullopt;
    }
    return std::make_optional<std::bitset<32>>(word);
  }

  constexpr static const std::array<u32, 6> ECC_MASKS{0xbb1f3480, 0x5d8f9a40, 0xaec7cd00,
                                                      0x5763e680, 0x6bb1f340, 0x8b7a89c0};

  bool has_sync = false;
  bool try_sync = false;
  bool ecc_passed = false;
  bool flip_data = false;
  bool subframe_ecc_passed = false;

  u16 word_bit_count = 0;
  u16 subframe_id = 0;
  u16 word_id = 0;

  std::array<i16, 3> eph_issues{-1};
  Ephemeris ephemeris{.sys = GNSSSystem::GPS};
  std::queue<NavMessage> nav_msgs;

  std::optional<f64> sys_time = std::nullopt;
  f64 new_sys_time;

  std::bitset<32> cur_word;  // |-prev word tail (2)-|------------cur word data
                             // (24)------------|---parity (6)---|
  std::array<std::bitset<24>, 8> subframe_words;
};
}  // namespace weaver
