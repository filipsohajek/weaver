#pragma once
#include "kernels_avx2.h"
#include "weaver/dsp/kernels_generic.h"

namespace weaver::dsp {
template<size_t NCorrelations, Modulation Modulation>
void mmcorr(size_t n,
            const cp_i16* samples,
            const u8* chips,
            cp_f32* out,
            f32 mix_init_phase,
            f32 mix_phase_step,
            f32 corr_offset,
            f64 code_init_phase,
            f64 code_phase_step) {
#ifdef __AVX2__
  size_t n_avx2 = 8 * (n / 8);
  mmcorr_avx2<NCorrelations, Modulation>(n_avx2, samples, chips, out, mix_init_phase, mix_phase_step,
                                         corr_offset, code_init_phase, code_phase_step);
  size_t rem_n = n % 8;
  if (rem_n) {
    mmcorr_gen<NCorrelations, Modulation>(
        rem_n, samples + n_avx2, chips, out, mix_init_phase + n_avx2 * mix_phase_step,
        mix_phase_step, corr_offset, code_init_phase + n_avx2 * code_phase_step,
        code_phase_step);
  }
#else
  mmcorr_gen<NCorrelations, Modulation>(n, samples, chips, out, mix_init_phase, mix_phase_step,
                                        corr_offset, code_init_phase, code_phase_step);
#endif
}

template<typename T, Modulation Modulation>
void modulate(size_t n,
              const u8* __restrict__ chips,
              T* __restrict__ out,
              f32 mix_init_phase,
              f32 mix_phase_step,
              f64 code_init_phase,
              f64 code_phase_step) {
#ifdef __AVX2__
  size_t n_avx2 = 8 * (n / 8);
  modulate_avx2<T, Modulation>(n_avx2, chips, out, mix_init_phase, mix_phase_step,
                                           code_init_phase, code_phase_step);
  size_t rem_n = n % 8;
  if (rem_n) {
    size_t rem_offset = 8 * (n / 8);
    modulate_gen<T, Modulation>(
        rem_n, chips, out + rem_offset, mix_init_phase + n_avx2 * mix_phase_step,
        mix_phase_step, code_init_phase + n_avx2 * code_phase_step, code_phase_step);
  }
#else
  modulate_gen<NCorrelations, Modulation>(n, chips, out, mix_init_phase, mix_phase_step,
                                          code_init_phase, code_phase_step);
#endif
}

inline void cvt_cpi16_cpf32(size_t n, const cp_i16* __restrict__ in, cp_f32* __restrict__ out) {
  for (size_t i = 0; i < n; i++) {
    out[i] = cp_cast<f32>(in[i]) / 32767.0f;
  }
}

inline void cvt_cpf32_cpi16(size_t n, const cp_f32* __restrict__ in, cp_i16* __restrict__ out) {
  for (size_t i = 0; i < n; i++) {
    out[i] = cp_cast<i16>(in[i] * 32767.0f);
  }
}

inline void mul_shift_cpf32(size_t n, const cp_f32* __restrict__ in_a, const cp_f32* __restrict__ in_b, ssize_t b_shift, cp_f32* __restrict__ out) {
  for (size_t i = 0; i < n; i++) {
    out[i] = in_a[i] * in_b[(i + b_shift + n) % n];
  }
}

inline void conj_cpf32(size_t n, const cp_f32* __restrict__ in, cp_f32* __restrict__ out) {
  for (size_t i = 0; i < n; i++) {
    out[i] = std::conj(in[i]);
  }
}
}  // namespace weaver::dsp
