#pragma once

#include <numeric>
#include "weaver/dsp/code.h"
#include "weaver/dsp/kernels.h"
#include "weaver/types.h"
namespace weaver {
class NavDataDecoder {
public:
  virtual ~NavDataDecoder() {};
  virtual f64 symbol_period_s() const = 0;
  virtual void process_symbol(cp_f32) = 0;
};

class Signal {
public:
  virtual ~Signal() {};
  virtual void correlate(std::span<const cp_i16> samples_in,
                         std::span<cp_f32> out,
                         f32 sample_rate,
                         f32 mix_phase,
                         f32 mix_frequency,
                         f64 code_phase,
                         f64 code_frequency,
                         std::span<const f64> code_offsets) const = 0;

  virtual void generate(std::span<cp_f32> samples_out,
                        f32 sample_rate,
                        f32 mix_phase,
                        f32 mix_frequency,
                        f64 code_phase,
                        f64 code_frequency) const = 0;
  virtual void generate(std::span<cp_f32> samples_out,
                        f32 sample_rate,
                        f32 mix_phase,
                        f32 mix_frequency,
                        f64 code_phase) const = 0;
  virtual size_t chip_count() const = 0;
  virtual f64 code_period_s() const = 0;
  virtual f64 carrier_freq() const = 0;
  virtual u16 prn() const = 0;
  virtual std::unique_ptr<NavDataDecoder> data_decoder() const {
    return nullptr;
  }
};


template<dsp::IsCode Code>
class CodeSignal : public Signal {
public:
  CodeSignal(u16 prn) : _prn(prn) {
    chips.resize((2 * Code::CHIP_COUNT + 12 + 3) / 8);
    Code::gen_chips(prn, chips);
  }

  void correlate(std::span<const cp_i16> samples_in,
                 std::span<cp_f32> out,
                 f32 sample_rate,
                 f32 mix_phase,
                 f32 mix_frequency,
                 f64 code_phase,
                 f64 code_frequency,
                 std::span<const f64> code_offsets) const override {
    f32 mix_init_phase = -2.0f * std::numbers::pi * mix_phase;
    f32 mix_phase_step = -2.0f * std::numbers::pi * mix_frequency / sample_rate;

    f64 code_init_phase = std::fmod(code_phase, 1.0) * Code::CHIP_COUNT;
    f64 code_offset_sum = std::accumulate(code_offsets.begin(), code_offsets.end(), 0.0);
    code_init_phase -= code_offset_sum;
    if (code_init_phase < 0)
      code_init_phase += Code::CHIP_COUNT;

    f64 code_phase_step = (Code::CHIP_COUNT * code_frequency) / sample_rate;
    
    if (code_offsets.size() == 1)
      dsp::mmcorr<1, Code::MODULATION>(
          samples_in.size(), samples_in.data(), chips.data(), out.data(), mix_init_phase,
          mix_phase_step, code_offsets.data(), code_init_phase, code_phase_step);
    else if (code_offsets.size() == 2)
      dsp::mmcorr<2, Code::MODULATION>(
          samples_in.size(), samples_in.data(), chips.data(), out.data(), mix_init_phase,
          mix_phase_step, code_offsets.data(), code_init_phase, code_phase_step);
  }

  void generate(std::span<cp_f32> samples_out,
                f32 sample_rate,
                f32 mix_phase,
                f32 mix_frequency,
                f64 code_phase,
                f64 code_frequency) const override {
    f32 mix_phase_step = mix_frequency / sample_rate;
    f64 code_phase_step = (Code::CHIP_COUNT * code_frequency) / sample_rate;
    f64 code_init_phase = code_phase * Code::CHIP_COUNT;
    while (!samples_out.empty()) {
      size_t gen_size =
          std::min(samples_out.size(), size_t(Code::CHIP_COUNT / code_phase_step));
      dsp::modulate<cp_f32, Code::MODULATION>(gen_size, chips.data(),
                                              samples_out.data(), mix_phase, mix_phase_step,
                                              code_init_phase, code_phase_step);

      samples_out = samples_out.subspan(gen_size);
      mix_phase_step += mix_phase_step * gen_size;
    }
  }

  void generate(std::span<cp_f32> samples_out,
                f32 sample_rate,
                f32 mix_phase,
                f32 mix_frequency,
                f64 code_phase) const override {
    generate(samples_out, sample_rate, mix_phase, mix_frequency, code_phase,
             Code::CHIP_RATE_HZ / Code::CHIP_COUNT);
  }

  size_t chip_count() const override {
    return Code::CHIP_COUNT;
  }

  f64 code_period_s() const override { return f64(Code::CHIP_COUNT) / f64(Code::CHIP_RATE_HZ); }

  u16 prn() const override { return _prn; }

protected:
  u16 _prn;
  std::vector<u8> chips;
};
}  // namespace weaver
