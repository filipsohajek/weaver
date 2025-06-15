#pragma once
#include <immintrin.h>

#include <cassert>

#include "weaver/dsp/code.h"
#include "weaver/types.h"

namespace weaver::dsp {
inline __m256 mm256_mul_cpps(__m256 a, __m256 b) {
  __m256 b_shuf = _mm256_permute_ps(b, 0xb1);
  __m256 a_real = _mm256_moveldup_ps(a);
  __m256 a_imag = _mm256_movehdup_ps(a);
  __m256 a_imag_prod = _mm256_mul_ps(a_imag, b_shuf);
  return _mm256_fmaddsub_ps(a_real, b, a_imag_prod);
}

inline cp_f32 mm256_hsum_cpps(__m256 in_cpps) {
  __m256i in_epi32 = std::bit_cast<__m256i>(in_cpps);
  f32 real_sum = std::bit_cast<f32>(_mm256_extract_epi32(in_epi32, 0)) +
                 std::bit_cast<f32>(_mm256_extract_epi32(in_epi32, 2)) +
                 std::bit_cast<f32>(_mm256_extract_epi32(in_epi32, 4)) +
                 std::bit_cast<f32>(_mm256_extract_epi32(in_epi32, 6));
  f32 imag_sum = std::bit_cast<f32>(_mm256_extract_epi32(in_epi32, 1)) +
                 std::bit_cast<f32>(_mm256_extract_epi32(in_epi32, 3)) +
                 std::bit_cast<f32>(_mm256_extract_epi32(in_epi32, 5)) +
                 std::bit_cast<f32>(_mm256_extract_epi32(in_epi32, 7));

  return {real_sum, imag_sum};
}

template<size_t NCorrelations, Modulation Modulation>
void mmcorr_avx2(size_t n,
                 const cp_i16* samples,
                 const u8* __restrict__ chips,
                 cp_f32* __restrict__ out,
                 f32 mix_init_phase,
                 f32 mix_phase_step,
                 f32 corr_offset,
                 f64 code_init_phase,
                 f64 code_phase_step) {
  __m256 float_scale = _mm256_set1_ps(1.0f / 32768.0f);
  __m256d code_phase_offset = _mm256_set1_pd(corr_offset);
  array<__m256, NCorrelations> partial_sums;
  for (size_t i = 0; i < NCorrelations; i++) {
    partial_sums[i] = _mm256_set1_ps(0.0f);
  }

  __m256 step_vec, cur_vec;
  std::complex<float> step = std::polar(1.0f, 4.0f * mix_phase_step);
  step_vec = std::bit_cast<__m256>(_mm256_set1_epi64x(std::bit_cast<uint64_t>(step)));
  for (size_t i = 0; i < sizeof(__m256i) / sizeof(std::complex<float>); i++) {
    std::complex<float> init = std::polar(1.0f, mix_init_phase + i * mix_phase_step);
    *(reinterpret_cast<std::complex<float>*>(&cur_vec) + i) = init;
  }

  __m256d code_step_vec = _mm256_set1_pd(code_phase_step);
  __m256d code_idx_vec = _mm256_set_pd(3, 2, 1, 0);
  __m256d code_inc_vec = _mm256_set1_pd(4);

  __m256 sign_mask = std::bit_cast<__m256>(_mm256_set1_epi32(0x80000000));

  for (size_t i = 0; i < n; i += sizeof(__m256i) / sizeof(std::complex<short>)) {
    __m256i in = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(samples + i));
    __m128i in_low = _mm256_extractf128_si256(in, 0);
    __m256 in_low_float = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(in_low));
    in_low_float = mm256_mul_cpps(in_low_float, cur_vec);
    cur_vec = mm256_mul_cpps(cur_vec, step_vec);

    __m128i in_high = _mm256_extractf128_si256(in, 1);
    __m256 in_high_float = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(in_high));
    in_high_float = mm256_mul_cpps(in_high_float, cur_vec);
    cur_vec = mm256_mul_cpps(cur_vec, step_vec);

    double code_base = code_init_phase + i * code_phase_step;
    size_t code_base_int = static_cast<size_t>(code_base) / 4;
    uint64_t chip_word = __bswap_32(*reinterpret_cast<const uint32_t*>(chips + code_base_int));
    uint64_t chip_dword = (chip_word << 33) | chip_word;
    __m256i chips = _mm256_set1_epi64x(chip_dword);
    __m256d code_base_vec = _mm256_set1_pd(4 * code_base_int);
    __m256d code_phase_low = _mm256_fmsub_pd(code_step_vec, code_idx_vec, code_base_vec);
    code_idx_vec = _mm256_add_pd(code_idx_vec, code_inc_vec);
    __m256d code_phase_high = _mm256_fmsub_pd(code_step_vec, code_idx_vec, code_base_vec);
    code_idx_vec = _mm256_add_pd(code_idx_vec, code_inc_vec);

    for (size_t j = 0; j < NCorrelations; j++) {
      __m256i code_phase_lo_int = _mm256_cvtepi32_epi64(_mm256_cvttpd_epi32(code_phase_low));
      __m256i code_phase_hi_int = _mm256_cvtepi32_epi64(_mm256_cvttpd_epi32(code_phase_high));

      code_phase_lo_int = _mm256_slli_epi64(code_phase_lo_int, 1);
      code_phase_hi_int = _mm256_slli_epi64(code_phase_hi_int, 1);

      __m256 chips_low = _mm256_and_ps(
          std::bit_cast<__m256>(_mm256_sllv_epi64(chips, code_phase_lo_int)), sign_mask);
      __m256 chips_high = _mm256_and_ps(
          std::bit_cast<__m256>(_mm256_sllv_epi64(chips, code_phase_hi_int)), sign_mask);

      __m256 out_low_float = _mm256_xor_ps(in_low_float, chips_low);
      __m256 out_high_float = _mm256_xor_ps(in_high_float, chips_high);

      partial_sums[j] =
          _mm256_fmadd_ps(out_low_float, float_scale, partial_sums[j]);
      partial_sums[j] =
          _mm256_fmadd_ps(out_high_float, float_scale, partial_sums[j]);

      code_phase_low = _mm256_add_pd(code_phase_low, code_phase_offset);
      code_phase_high = _mm256_add_pd(code_phase_high, code_phase_offset);
    }
  }
  for (size_t i = 0; i < NCorrelations; i++) {
    out[i] += mm256_hsum_cpps(partial_sums[i]);
  }
}

template<size_t NCorrelations, Modulation Modulation>
void modulate_avx2(size_t n,
                   const u8* __restrict__ chips,
                   cp_i16* __restrict__ out,
                   f32 mix_init_phase,
                   f32 mix_phase_step,
                   f64 code_init_phase,
                   f64 code_phase_step) {
  __m256 float_scale = _mm256_set1_ps(32768.0f);

  __m256 step_vec, cur_vec;
  std::complex<float> step = std::polar(1.0f, 4.0f * mix_phase_step);
  step_vec = std::bit_cast<__m256>(_mm256_set1_epi64x(std::bit_cast<uint64_t>(step)));
  std::complex<float> init = std::polar(1.0f, mix_init_phase);
  for (size_t i = 0; i < sizeof(__m256i) / sizeof(std::complex<float>); i++) {
    *(reinterpret_cast<std::complex<float>*>(&cur_vec) + i) = init;
    init *= std::polar(1.0f, mix_phase_step);
  }

  __m256d code_step_vec = _mm256_set1_pd(4 * code_phase_step);
  __m256d code_cur_vec =
      _mm256_set_pd(3 * code_phase_step + code_init_phase, 2 * code_phase_step + code_init_phase,
                    code_phase_step + code_init_phase, code_init_phase);

  __m256 sign_mask = std::bit_cast<__m256>(_mm256_set1_epi32(0x80000000));

  for (size_t i = 0; i < n; i += sizeof(__m256i) / sizeof(std::complex<short>)) {
    double code_base = code_init_phase + i * code_phase_step;
    int code_base_int = static_cast<int>(code_base);
    code_base_int = 8 * (code_base_int / 8);
    uint8_t chip_word1 = chips[code_base_int];
    uint8_t chip_word2 = chips[code_base_int + 1];
    uint64_t chip_word = (chip_word1 << 16) | chip_word2;
    uint64_t chip_dword = (chip_word << 48) | (chip_word << 1);
    __m256i chips = _mm256_set1_epi64x(chip_dword);
    __m256d code_base_vec = _mm256_set1_pd(static_cast<int>(code_base));
    __m256d code_phase_low = _mm256_sub_pd(code_cur_vec, code_base_vec);
    code_cur_vec = _mm256_add_pd(code_cur_vec, code_step_vec);
    __m256d code_phase_high = _mm256_sub_pd(code_cur_vec, code_base_vec);
    code_cur_vec = _mm256_add_pd(code_cur_vec, code_step_vec);

    __m256i code_phase_lo_int = _mm256_cvtepi32_epi64(_mm256_cvtpd_epi32(code_phase_low));
    __m256i code_phase_hi_int = _mm256_cvtepi32_epi64(_mm256_cvtpd_epi32(code_phase_high));

    __m256 chips_low = _mm256_and_ps(
        std::bit_cast<__m256>(_mm256_sllv_epi64(chips, code_phase_lo_int)), sign_mask);
    __m256 chips_high = _mm256_and_ps(
        std::bit_cast<__m256>(_mm256_sllv_epi64(chips, code_phase_hi_int)), sign_mask);

    __m256 in_low = _mm256_mul_ps(cur_vec, float_scale);
    cur_vec = mm256_mul_cpps(cur_vec, step_vec);
    __m256 in_high = _mm256_mul_ps(cur_vec, float_scale);
    cur_vec = mm256_mul_cpps(cur_vec, step_vec);

    __m256 out_low = _mm256_xor_ps(in_low, chips_low);
    __m256 out_high = _mm256_xor_ps(in_high, chips_high);
    
    _mm256_storeu_ps(reinterpret_cast<f32*>(out) + 16*i, out_low);
    _mm256_storeu_ps(reinterpret_cast<f32*>(out) + 16*i + 8, out_high);
  }
}
}  // namespace weaver::dsp
