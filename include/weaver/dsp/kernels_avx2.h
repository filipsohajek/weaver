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

inline void mm256_print_cpps(__m256 in_cpps) {
  __m256i in_epi32 = std::bit_cast<__m256i>(in_cpps);
  std::cout << std::bit_cast<f32>(_mm256_extract_epi32(in_epi32, 0)) << ", ";
  std::cout << std::bit_cast<f32>(_mm256_extract_epi32(in_epi32, 1)) << ", ";
  std::cout << std::bit_cast<f32>(_mm256_extract_epi32(in_epi32, 2)) << ", ";
  std::cout << std::bit_cast<f32>(_mm256_extract_epi32(in_epi32, 3)) << ", ";
  std::cout << std::bit_cast<f32>(_mm256_extract_epi32(in_epi32, 4)) << ", ";
  std::cout << std::bit_cast<f32>(_mm256_extract_epi32(in_epi32, 5)) << ", ";
  std::cout << std::bit_cast<f32>(_mm256_extract_epi32(in_epi32, 6)) << ", ";
  std::cout << std::bit_cast<f32>(_mm256_extract_epi32(in_epi32, 7)) << ", ";
  std::cout << "\n";
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
  const uint8_t FRAC_BITS = 39;

  __m256 float_scale = _mm256_set1_ps(1.0f / 32768.0f);
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

  f64 frac_scale = 1ul << FRAC_BITS;
  f64 eps = 1 / frac_scale;
  uint64_t code_init_phase_int = static_cast<uint64_t>((code_init_phase + eps) * frac_scale);
  uint64_t code_step_int = static_cast<uint64_t>((code_phase_step + eps) * frac_scale);
  uint64_t corr_offset_int = static_cast<uint64_t>((corr_offset + eps) * frac_scale);
  __m256i corr_phase_offset = _mm256_set1_epi64x(corr_offset_int);
  __m256i code_phase_vec = _mm256_set_epi64x(
      code_init_phase_int + 3 * code_step_int, code_init_phase_int + 2 * code_step_int,
      code_init_phase_int + code_step_int, code_init_phase_int);
  __m256i code_step_vec = _mm256_set1_epi64x(4 * code_step_int);

  __m256 sign_mask = std::bit_cast<__m256>(_mm256_set1_epi32(0x80000000));
  __m128i base_mask = _mm_set1_epi64x(0xffffffffffffffff << (FRAC_BITS + 2));

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

    __m128i base_vec_128 = _mm256_extractf128_si256(code_phase_vec, 0);
    uint64_t chip_base =
        std::bit_cast<uint64_t>(_mm_cvtsd_f64(std::bit_cast<__m128d>(base_vec_128)));
    chip_base >>= FRAC_BITS + 2;
    base_vec_128 = _mm_and_si128(base_vec_128, base_mask);
    __m256i base_vec = _mm256_broadcastq_epi64(base_vec_128);
    uint64_t chip_word = __bswap_32(*reinterpret_cast<const uint32_t*>(chips + chip_base));
    uint64_t chip_dword = (chip_word << 33) | chip_word;
    __m256i chip_vec = _mm256_set1_epi64x(chip_dword);

    __m256i code_phase_low = _mm256_sub_epi64(code_phase_vec, base_vec);
    code_phase_vec = _mm256_add_epi64(code_phase_vec, code_step_vec);
    __m256i code_phase_high = _mm256_sub_epi64(code_phase_vec, base_vec);
    code_phase_vec = _mm256_add_epi64(code_phase_vec, code_step_vec);

    for (size_t j = 0; j < NCorrelations; j++) {
      __m256i code_phase_lo_rd = _mm256_srli_epi64(code_phase_low, FRAC_BITS);
      code_phase_lo_rd = _mm256_slli_epi64(code_phase_lo_rd, 1);
      __m256i code_phase_hi_rd = _mm256_srli_epi64(code_phase_high, FRAC_BITS);
      code_phase_hi_rd = _mm256_slli_epi64(code_phase_hi_rd, 1);

      __m256 chips_low = _mm256_and_ps(
          std::bit_cast<__m256>(_mm256_sllv_epi64(chip_vec, code_phase_lo_rd)), sign_mask);
      __m256 chips_high = _mm256_and_ps(
          std::bit_cast<__m256>(_mm256_sllv_epi64(chip_vec, code_phase_hi_rd)), sign_mask);

      __m256 out_low_float = _mm256_xor_ps(in_low_float, chips_low);
      __m256 out_high_float = _mm256_xor_ps(in_high_float, chips_high);

      partial_sums[j] = _mm256_fmadd_ps(out_low_float, float_scale, partial_sums[j]);
      partial_sums[j] = _mm256_fmadd_ps(out_high_float, float_scale, partial_sums[j]);

      code_phase_low = _mm256_add_epi64(code_phase_low, corr_phase_offset);
      code_phase_high = _mm256_add_epi64(code_phase_high, corr_phase_offset);
    }
  }
  for (size_t i = 0; i < NCorrelations; i++) {
    out[i] += mm256_hsum_cpps(partial_sums[i]);
  }
}

template<typename T, Modulation Modulation>
void modulate_avx2(size_t n,
                   const u8* __restrict__ chips,
                   T* __restrict__ out,
                   f32 mix_init_phase,
                   f32 mix_phase_step,
                   f64 code_init_phase,
                   f64 code_phase_step) {
  const uint8_t FRAC_BITS = 39;
  __m256 float_scale = _mm256_set1_ps(32768.0f);

  __m256 step_vec, cur_vec;
  std::complex<float> step = std::polar(1.0f, 4.0f * mix_phase_step);
  step_vec = std::bit_cast<__m256>(_mm256_set1_epi64x(std::bit_cast<uint64_t>(step)));
  for (size_t i = 0; i < sizeof(__m256i) / sizeof(std::complex<float>); i++) {
    std::complex<float> init = std::polar(1.0f, mix_init_phase + i * mix_phase_step);
    *(reinterpret_cast<std::complex<float>*>(&cur_vec) + i) = init;
  }

  f64 frac_scale = 1ul << FRAC_BITS;
  f64 eps = 1 / frac_scale;
  uint64_t code_init_phase_int = static_cast<uint64_t>((code_init_phase + eps) * frac_scale);
  uint64_t code_step_int = static_cast<uint64_t>((code_phase_step + eps) * frac_scale);
  __m256i code_phase_vec = _mm256_set_epi64x(
      code_init_phase_int + 3 * code_step_int, code_init_phase_int + 2 * code_step_int,
      code_init_phase_int + code_step_int, code_init_phase_int);
  __m256i code_step_vec = _mm256_set1_epi64x(4 * code_step_int);

  __m256 sign_mask = std::bit_cast<__m256>(_mm256_set1_epi32(0x80000000));
  __m128i base_mask = _mm_set1_epi64x(0xffffffffffffffff << (FRAC_BITS + 2));

  for (size_t i = 0; i < n; i += sizeof(__m256i) / sizeof(std::complex<short>)) {
    __m256 in_low_float = cur_vec;
    cur_vec = mm256_mul_cpps(cur_vec, step_vec);
    __m256 in_high_float = cur_vec;
    cur_vec = mm256_mul_cpps(cur_vec, step_vec);

    __m128i base_vec_128 = _mm256_extractf128_si256(code_phase_vec, 0);
    uint64_t chip_base =
        std::bit_cast<uint64_t>(_mm_cvtsd_f64(std::bit_cast<__m128d>(base_vec_128)));
    chip_base >>= FRAC_BITS + 2;
    base_vec_128 = _mm_and_si128(base_vec_128, base_mask);
    __m256i base_vec = _mm256_broadcastq_epi64(base_vec_128);
    uint64_t chip_word = __bswap_32(*reinterpret_cast<const uint32_t*>(chips + chip_base));
    uint64_t chip_dword = (chip_word << 33) | chip_word;
    __m256i chip_vec = _mm256_set1_epi64x(chip_dword);

    __m256i code_phase_low = _mm256_sub_epi64(code_phase_vec, base_vec);
    code_phase_vec = _mm256_add_epi64(code_phase_vec, code_step_vec);
    __m256i code_phase_high = _mm256_sub_epi64(code_phase_vec, base_vec);
    code_phase_vec = _mm256_add_epi64(code_phase_vec, code_step_vec);

    __m256i code_phase_lo_rd = _mm256_srli_epi64(code_phase_low, FRAC_BITS);
    code_phase_lo_rd = _mm256_slli_epi64(code_phase_lo_rd, 1);
    __m256i code_phase_hi_rd = _mm256_srli_epi64(code_phase_high, FRAC_BITS);
    code_phase_hi_rd = _mm256_slli_epi64(code_phase_hi_rd, 1);

    __m256 chips_low = _mm256_and_ps(
        std::bit_cast<__m256>(_mm256_sllv_epi64(chip_vec, code_phase_lo_rd)), sign_mask);
    __m256 chips_high = _mm256_and_ps(
        std::bit_cast<__m256>(_mm256_sllv_epi64(chip_vec, code_phase_hi_rd)), sign_mask);

    __m256 out_low_float = _mm256_xor_ps(in_low_float, chips_low);
    __m256 out_high_float = _mm256_xor_ps(in_high_float, chips_high);

    if constexpr (std::is_same_v<T, cp_i16>) {
      __m256i out_low = _mm256_cvtps_epi32(_mm256_mul_ps(out_low_float, float_scale));
      __m256i out_high = _mm256_cvtps_epi32(_mm256_mul_ps(out_high_float, float_scale));

      __m256i out_vec = _mm256_permute4x64_epi64(_mm256_packs_epi32(out_low, out_high), 0xd8);

      _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), out_vec);
    } else if constexpr (std::is_same_v<T, cp_f32>) {
      _mm256_storeu_ps(reinterpret_cast<float*>(out + i), out_low_float);
      _mm256_storeu_ps(reinterpret_cast<float*>(out + i + 4), out_high_float);
    } else {
      static_assert(false, "sample type not supported by the AVX2 kernel");
    }
  }
}
}  // namespace weaver::dsp
