#pragma once

#include "weaver/dsp/code.h"
#include "weaver/dsp/kernels.h"
#include "weaver/types.h"
namespace weaver {
class Signal {
public:
  virtual ~Signal() {};
  virtual cp_f32 correlate(std::span<const cp_i16> samples_in,
                           std::span<cp_f32> out,
                           f32 sample_rate,
                           f32& mix_phase,
                           f32 mix_frequency,
                           f64& code_phase) const = 0;
  virtual f32 discriminate(std::span<cp_f32> corr_in) const = 0;
  virtual f64 disc_error_var(f64 cn0) const = 0;

  virtual void generate(std::span<cp_f32> samples_out,
                        f32 sample_rate,
                        f32 mix_phase,
                        f32 mix_frequency,
                        f64 code_phase) const = 0;
  virtual size_t n_replicas() const = 0;
  virtual f64 code_period_s() const = 0;
  virtual f64 carrier_freq() const = 0;
  virtual u16 prn() const = 0;
};

template<typename T>
concept IsCodeDiscriminator = requires(T t) {
  { T::N_REPLICAS } -> std::convertible_to<size_t>;
  { T::CORR_OFFSET } -> std::convertible_to<f32>;

  { T::disc_code(std::span<cp_f32>()) } -> std::convertible_to<f32>;
  { T::disc_error_var(f64()) } -> std::convertible_to<f64>;
};

template<dsp::IsCode Code, IsCodeDiscriminator CodeDisc>
class CodeSignal : public Signal {
public:
  CodeSignal(u16 prn) : _prn(prn) {}

  cp_f32 correlate(std::span<const cp_i16> samples_in,
                   std::span<cp_f32> out,
                   f32 sample_rate,
                   f32& mix_phase,
                   f32 mix_frequency,
                   f64& code_phase) const override {
    // TODO!!!
    f32 mix_init_phase = mix_phase;
    f32 mix_phase_step = mix_frequency / sample_rate;
    f64 code_init_phase = code_phase / Code::CHIP_COUNT;
    f64 code_phase_step = Code::CHIP_RATE_HZ / (sample_rate * Code::CHIP_COUNT);
    size_t n = std::min(samples_in.size(), size_t(1.0 / code_phase_step));
    dsp::mmcorr<CodeDisc::N_REPLICAS, Code::MODULATION>(
        n, samples_in.data(), Code::CHIPS[_prn].data(), out.data(), mix_init_phase,
        mix_phase_step, CodeDisc::CORR_OFFSET, code_init_phase, code_phase_step);
    return 0;
  }

  f32 discriminate(std::span<cp_f32> corr_in) const override {
    return CodeDisc::disc_code(corr_in);
  }

  f64 disc_error_var(f64 cn0) const override { return CodeDisc::disc_error_var(cn0); }

  void generate(std::span<cp_f32> samples_out,
                f32 sample_rate,
                f32 mix_phase,
                f32 mix_frequency,
                f64 code_phase) const override {
    f32 mix_phase_step = mix_frequency / sample_rate;
    f64 code_phase_step = Code::CHIP_RATE_HZ / sample_rate;
    f64 code_init_phase = code_phase * Code::CHIP_COUNT;
    while (!samples_out.empty()) {
      size_t gen_size = std::min(samples_out.size(), size_t((Code::CHIP_COUNT + 1)/ code_phase_step));
      dsp::modulate<cp_f32, Code::MODULATION>(gen_size, Code::CHIPS[_prn].data(),
                                            samples_out.data(), mix_phase, mix_phase_step,
                                            code_init_phase, code_phase_step);

      samples_out = samples_out.subspan(gen_size);
      code_init_phase += code_phase_step * gen_size;
      code_init_phase = std::fmod(code_init_phase, Code::CHIP_COUNT);
      mix_phase_step += mix_phase_step * gen_size;
    }
  }

  size_t n_replicas() const override { return CodeDisc::N_REPLICAS; }

  f64 code_period_s() const override { return f64(Code::CHIP_COUNT) / f64(Code::CHIP_RATE_HZ); }

  u16 prn() const override { return _prn; }

protected:
  u16 _prn;
};
}  // namespace weaver
