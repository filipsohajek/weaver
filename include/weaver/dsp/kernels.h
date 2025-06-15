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
  mmcorr_avx2<NCorrelations, Modulation>(n, samples, chips, out, mix_init_phase, mix_phase_step,
                                         corr_offset, code_init_phase, code_phase_step);
  size_t rem_n = n % 8;
  if (rem_n) {
    size_t rem_offset = 8 * (n / 8);
    size_t processed_n = n - rem_offset;
    mmcorr_gen<NCorrelations, Modulation>(
        rem_n, samples + rem_offset, chips, out, mix_init_phase + processed_n * mix_phase_step,
        mix_phase_step, corr_offset, code_init_phase + processed_n * code_phase_step,
        code_phase_step);
  }
#else
  mmcorr_gen<NCorrelations, Modulation>(n, samples, chips, out, mix_init_phase, mix_phase_step,
                                        corr_offset, code_init_phase, code_phase_step);
#endif
}

template<size_t NCorrelations, Modulation Modulation>
void modulate(size_t n,
              const u8* __restrict__ chips,
              cp_i16* __restrict__ out,
              f32 mix_init_phase,
              f32 mix_phase_step,
              f64 code_init_phase,
              f64 code_phase_step) {
#ifdef __AVX2__
  modulate_avx2<NCorrelations, Modulation>(n, chips, out, mix_init_phase, mix_phase_step,
                                           code_init_phase, code_phase_step);
  size_t rem_n = n % 8;
  if (rem_n) {
    size_t rem_offset = 8 * (n / 8);
    size_t processed_n = n - rem_offset;
    modulate_gen<NCorrelations, Modulation>(
        rem_n, chips, out + rem_offset, mix_init_phase + processed_n * mix_phase_step,
        mix_phase_step, code_init_phase + n * code_phase_step, code_phase_step);
  }
#else
  modulate_gen<NCorrelations, Modulation>(n, chips, out, mix_init_phase, mix_phase_step,
                                          code_init_phase, code_phase_step);
#endif
}
}  // namespace weaver::dsp
