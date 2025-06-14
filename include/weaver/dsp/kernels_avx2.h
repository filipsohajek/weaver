#pragma once
#include <immintrin.h>

#include <cassert>

#include "kernels.h"
#include "weaver/types.h"

namespace weaver::dsp::avx2 {
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

inline __m256 mix_kernel_cpps(__m256 in,
                              __m256& phase_vec,
                              __m256 phase_step_vec,
                              f32 init_phase,
                              f32 phase_step,
                              size_t sample_vec_i) {
  const size_t RELOAD_COUNT = 128;
  const size_t VEC_SAMPLE_COUNT = sizeof(__m256i) / sizeof(cp_f32);
  if ((sample_vec_i % RELOAD_COUNT) == 0) {
    f32 new_init_phase =
        init_phase + static_cast<f32>(VEC_SAMPLE_COUNT) * sample_vec_i * phase_step;
    for (size_t i = 0; i < VEC_SAMPLE_COUNT; i++) {
      cp_f32 phase_cp = std::polar(1.0f, new_init_phase + phase_step * static_cast<f32>(i));
      reinterpret_cast<cp_f32*>(&phase_vec)[i] = phase_cp;
    }
  }

  __m256 out = mm256_mul_cpps(in, phase_vec);
  phase_vec = mm256_mul_cpps(phase_vec, phase_step_vec);
  return out;
}

inline __m256 code_mult_bpsk_kernel_cpps(__m256 in, __m256d phase_vec, __m256 chip_vec) {
  __m128i index_vec_128 = _mm256_cvttpd_epi32(phase_vec);
  __m256i index_vec = std::bit_cast<__m256i>(
      _mm256_moveldup_ps(std::bit_cast<__m256>(_mm256_cvtepi32_epi64(index_vec_128))));
  __m256i sign_vec = _mm256_permutevar8x32_epi32(std::bit_cast<__m256i>(chip_vec), index_vec);

  __m256 out = _mm256_mul_ps(in, std::bit_cast<__m256>(sign_vec));
  return out;
}

inline void init_code_phase(__m256d& phase_step_vec,
                            __m256d& phase_vec,
                            f64 init_phase,
                            f64 phase_step) {
  phase_step_vec = _mm256_set1_pd(4.0 * phase_step);
  for (size_t j = 0; j < sizeof(__m256) / sizeof(f64); j++) {
    reinterpret_cast<f64*>(&phase_vec)[j] = init_phase + static_cast<f64>(j) * phase_step;
  }
}

inline void init_mix_phase_step(__m256& phase_step_vec, f32 phase_step) {
  cp_f32 mix_phase_step_cp = std::polar(1.0f, 4.0f * phase_step);
  phase_step_vec = std::bit_cast<__m256>(_mm256_set1_epi64x(std::bit_cast<i64>(mix_phase_step_cp)));
}

template<size_t NCorrelations, Modulation Modulation>
void mmcorr_avx2(size_t n,
                 const cp_f32* samples_in,
                 const cp_f32* __restrict__ chips,
                 cp_f32* __restrict__ out,
                 f32 mix_init_phase,
                 f32 mix_phase_step,
                 f32 corr_offset,
                 f64 code_init_phase,
                 f64 code_phase_step) {
  __m256d code_phase_offset = _mm256_set1_pd(corr_offset);
  array<__m256, NCorrelations> partial_sums;
  for (size_t i = 0; i < NCorrelations; i++) {
    partial_sums[i] = _mm256_set1_ps(0.0f);
  }
  __m256d code_phase_step_vec, code_phase_vec_pd;
  init_code_phase(code_phase_step_vec, code_phase_vec_pd, code_init_phase, code_phase_step);
  __m256 mix_phase_vec = _mm256_setzero_ps(), mix_phase_step_vec;
  init_mix_phase_step(mix_phase_step_vec, mix_phase_step);

  size_t vec_count = (n * sizeof(cp_f32)) / sizeof(__m256);
  const auto* __restrict__ in_base = reinterpret_cast<const __m256*>(samples_in);

  for (size_t vec_i = 0; vec_i < vec_count; vec_i++) {
    __m256 in_vec = _mm256_load_ps(reinterpret_cast<const f32*>(in_base + vec_i));
    __m256 mixed_vec = mix_kernel_cpps(in_vec, mix_phase_vec, mix_phase_step_vec, mix_init_phase,
                                       mix_phase_step, vec_i);

    f64 phase_base = code_init_phase + 4 * vec_i * code_phase_step;

    __m256d code_phase_vec = code_phase_vec_pd;
    __m256d base_vec = _mm256_set1_pd(static_cast<i32>(phase_base));
    code_phase_vec_pd = _mm256_add_pd(code_phase_vec_pd, code_phase_step_vec);
    code_phase_vec = _mm256_sub_pd(code_phase_vec, base_vec);

    __m256 chip_vec =
        _mm256_loadu_ps(reinterpret_cast<const f32*>(&chips[static_cast<ssize_t>(phase_base)]));
    for (size_t replica_i = 0; replica_i < NCorrelations; replica_i++) {
      __m256 out_vec;
      if constexpr (Modulation == Modulation::BPSK) {
        out_vec = code_mult_bpsk_kernel_cpps(mixed_vec, code_phase_vec, chip_vec);
      } else {
        static_assert(always_false<void>, "modulation type not supported by the AVX2 kernel");
      }
      partial_sums[replica_i] = _mm256_add_ps(partial_sums[replica_i], out_vec);
      code_phase_vec = _mm256_add_pd(code_phase_vec, code_phase_offset);
    }
  }
  for (size_t i = 0; i < NCorrelations; i++) {
    out[i] += mm256_hsum_cpps(partial_sums[i]);
  }
}

template<size_t NCorrelations, Modulation Modulation>
void mmcorr_avx2(size_t n,
                 const cp_i16* samples,
                 const cp_i16* __restrict__ chips,
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
  std::complex<float> init = std::polar(1.0f, mix_init_phase);
  for (size_t i = 0; i < sizeof(__m256i) / sizeof(std::complex<float>); i++) {
    *(reinterpret_cast<std::complex<float>*>(&cur_vec) + i) = init;
    init *= std::polar(1.0f, mix_phase_step);
  }

  __m256d code_step_vec = _mm256_set1_pd(4 * code_phase_step);
  __m256d code_cur_vec =
      _mm256_set_pd(3 * code_phase_step + code_init_phase, 2 * code_phase_step + code_init_phase,
                    code_phase_step + code_init_phase, code_init_phase);

  for (size_t i = 0; i < n; i += sizeof(__m256i) / sizeof(std::complex<short>)) {
    __m256i in = _mm256_load_si256(reinterpret_cast<const __m256i*>(samples + i));

    double code_base = i * code_phase_step;
    __m256i chip_vec =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(chips + static_cast<int>(code_base)));
    __m256d code_base_vec = _mm256_set1_pd(static_cast<int>(code_base));
    __m256d code_phase_low = _mm256_sub_pd(code_cur_vec, code_base_vec);
    code_cur_vec = _mm256_add_pd(code_cur_vec, code_step_vec);
    __m256d code_phase_high = code_cur_vec;
    code_cur_vec = _mm256_add_pd(code_cur_vec, code_step_vec);

    array<__m256i, NCorrelations> replica_phases;
    for (size_t j = 0; j < NCorrelations; j++) {
      __m128i code_phase_lo_int = _mm256_cvtpd_epi32(code_phase_low);
      __m128i code_phase_hi_int = _mm256_cvtpd_epi32(code_phase_high);
      __m256i code_phase = _mm256_setr_m128i(code_phase_lo_int, code_phase_hi_int);
      replica_phases[j] = code_phase;
      code_phase_low = _mm256_add_pd(code_phase_low, code_phase_offset);
      code_phase_high = _mm256_add_pd(code_phase_high, code_phase_offset);
    }

    __m256 cur_tmp = weaver::dsp::avx2::mm256_mul_cpps(cur_vec, step_vec);
    __m256 new_cur_vec = weaver::dsp::avx2::mm256_mul_cpps(cur_tmp, step_vec);

    for (size_t replica_i = 0; replica_i < NCorrelations; replica_i++) {
      __m256i replica_in =
          _mm256_sign_epi16(in, _mm256_permutevar8x32_epi32(chip_vec, replica_phases[replica_i]));

      __m128i in_low = _mm256_extractf128_si256(replica_in, 0);
      __m256 in_low_float = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(in_low));
      __m256 out_low_float = weaver::dsp::avx2::mm256_mul_cpps(in_low_float, cur_vec);
      partial_sums[replica_i] =
          _mm256_fmadd_ps(out_low_float, float_scale, partial_sums[replica_i]);

      __m128i in_high = _mm256_extractf128_si256(replica_in, 1);
      __m256 in_high_float = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(in_high));
      __m256 out_high_float = weaver::dsp::avx2::mm256_mul_cpps(in_high_float, cur_tmp);
      partial_sums[replica_i] =
          _mm256_fmadd_ps(out_high_float, float_scale, partial_sums[replica_i]);
    }
    cur_vec = new_cur_vec;
  }
  for (size_t i = 0; i < NCorrelations; i++) {
    out[i] += mm256_hsum_cpps(partial_sums[i]);
  }
}

template<size_t NCorrelations, Modulation Modulation>
void modulate_avx2(size_t n,
                   const cp_i16* __restrict__ chips,
                   cp_i16* __restrict__ out,
                   f32 mix_init_phase,
                   f32 mix_phase_step,
                   f64 code_init_phase,
                   f64 code_phase_step) {
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

  for (size_t i = 0; i < n; i += sizeof(__m256i) / sizeof(std::complex<short>)) {
    double code_base = i * code_phase_step;
    __m256d code_base_vec = _mm256_set1_pd(static_cast<int>(code_base));
    code_cur_vec = _mm256_add_pd(code_cur_vec, code_step_vec);
    __m256d code_phase_low = _mm256_sub_pd(code_cur_vec, code_base_vec);
    code_cur_vec = _mm256_add_pd(code_cur_vec, code_step_vec);
    __m256d code_phase_high = _mm256_sub_pd(code_cur_vec, code_base_vec);

    __m256i chip_vec =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(chips + static_cast<int>(code_base)));
    __m128i code_phase_lo_int = _mm256_cvtpd_epi32(code_phase_low);
    __m128i code_phase_hi_int = _mm256_cvtpd_epi32(code_phase_high);
    __m256i code_phase = _mm256_setr_m128i(code_phase_lo_int, code_phase_hi_int);

    __m256i in = _mm256_permutevar8x32_epi32(chip_vec, code_phase);

    __m128i in_low = _mm256_extractf128_si256(in, 0);
    __m256 in_low_float = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(in_low));
    __m256 out_low_float = weaver::dsp::avx2::mm256_mul_cpps(in_low_float, cur_vec);
    __m256i out_low = _mm256_cvtps_epi32(out_low_float);

    __m256 cur_tmp = weaver::dsp::avx2::mm256_mul_cpps(cur_vec, step_vec);
    __m128i in_high = _mm256_extractf128_si256(in, 1);
    __m256 in_high_float = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(in_high));
    __m256 out_high_float = weaver::dsp::avx2::mm256_mul_cpps(in_high_float, cur_tmp);
    __m256i out_high = _mm256_cvtps_epi32(out_high_float);

    cur_vec = weaver::dsp::avx2::mm256_mul_cpps(cur_tmp, step_vec);

    __m256i out_vec = _mm256_packs_epi32(out_low, out_high);
    out_vec = _mm256_permute4x64_epi64(out_vec, 0xd8);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), out_vec);
  }
}
}  // namespace weaver::dsp::avx2
