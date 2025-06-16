#pragma once
#include "weaver/dsp/code.h"
#include "weaver/types.h"
namespace weaver::dsp {
template<size_t NCorrelations, Modulation Modulation>
void mmcorr_gen(size_t n,
                const cp_i16* samples,
                const u8* __restrict__ chips,
                cp_f32* __restrict__ out,
                f32 mix_init_phase,
                f32 mix_phase_step,
                f32 corr_offset,
                f64 code_init_phase,
                f64 code_phase_step) {
  cp_f32 mix_cp = std::polar(1.0f, mix_init_phase);
  cp_f32 mix_step_cp = std::polar(1.0f, mix_phase_step);
  for (size_t i = 0; i < n; i++) {
    cp_i16 in = samples[i];
    cp_f32 mixed = cp_cast<f32>(in) / 32767.0f;
    mixed *= mix_cp;

    for (size_t replica_i = 0; replica_i < NCorrelations; replica_i++) {
      size_t chip_idx = static_cast<size_t>(code_init_phase + i * code_phase_step + replica_i * corr_offset);
      size_t chip_base = chip_idx / 4, chip_off = chip_idx % 4;
      uint8_t chip = (chips[chip_base] << (2 * chip_off)) >> 6;

      f32 out_real = (chip & 0x2) ? -mixed.real() : mixed.real();
      f32 out_imag = (chip & 0x1) ? -mixed.imag() : mixed.imag();
      out[replica_i] += cp_f32{out_real, out_imag};
    }

    mix_cp *= mix_step_cp;
  }
}

template<typename T, Modulation Modulation>
void modulate_gen(size_t n,
                  const u8* __restrict__ chips,
                  T* __restrict__ out,
                  f32 mix_init_phase,
                  f32 mix_phase_step,
                  f64 code_init_phase,
                  f64 code_phase_step) {
  cp_f32 mix_cp = std::polar(1.0f, mix_init_phase);
  cp_f32 mix_step_cp = std::polar(1.0f, mix_phase_step);
  for (size_t i = 0; i < n; i++) {
    size_t chip_idx = static_cast<size_t>(code_init_phase + i * code_phase_step);
    size_t chip_base = chip_idx / 4, chip_off = chip_idx % 4;
    uint8_t chip = (chips[chip_base] << (2 * chip_off)) >> 6;

    cp_f32 out_float = {mix_cp.real() * ((chip & 0x2) ? -1.0f : 1.0f),
                        mix_cp.imag() * ((chip & 0x1) ? -1.0f : 1.0f)};
    if constexpr (std::is_same_v<T, cp_f32>) {
      out[i] = out_float;
    } else if constexpr (std::is_same_v<T, cp_i16>) {
      out[i] = cp_cast<i16>(out_float * 32767.0f);
    } else {
      static_assert(false, "sample type not supported by the generic kernel");
    }

    mix_cp *= mix_step_cp;
  }
}

}  // namespace weaver::dsp
