#pragma once

#include "weaver/dsp/code.h"
#include "weaver/dsp/kernels.h"
#include "weaver/types.h"
namespace weaver {
class Signal {
public:
  virtual ~Signal() {};
  virtual void correlate(std::span<const cp_i16> samples_in,
                         std::span<cp_f32> out,
                         f32 sample_rate,
                         f32 mix_phase,
                         f32 mix_frequency,
                         f64 code_phase) const = 0;
  virtual f32 discriminate(std::span<cp_f32> corr_in, cp_f32& prompt) const = 0;
  virtual f64 disc_error_var(f64 cn0, f64 int_time_s) const = 0;

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
  virtual size_t n_replicas() const = 0;
  virtual f64 code_period_s() const = 0;
  virtual f64 carrier_freq() const = 0;
  virtual u16 prn() const = 0;
};

template<typename T>
concept IsCodeDiscriminator = requires(T t) {
  { T::SPREAD } -> std::convertible_to<size_t>;
  { T::CORR_OFFSET } -> std::convertible_to<f32>;

  { T::disc_code(std::span<cp_f32>()) } -> std::convertible_to<f32>;
  { T::disc_error_var(f64(), f64()) } -> std::convertible_to<f64>;
};

template<f64 CorrOffset>
struct NELPCodeDiscriminator {
  static const constexpr size_t SPREAD = 1;
  static const constexpr f64 CORR_OFFSET = CorrOffset;

  static f32 disc_code(std::span<cp_f32> corr_res) {
    f64 e_sq = std::pow(corr_res[0].real(), 2) + std::pow(corr_res[0].imag(), 2);
    f64 l_sq = std::pow(corr_res[2].real(), 2) + std::pow(corr_res[2].imag(), 2);
    return 0.5 * (e_sq - l_sq) / (e_sq + l_sq);
  }

  static f64 disc_error_var(f64 cn0, f64 int_time_s) {
    return ((CORR_OFFSET / (4 * cn0)) * (1 + 2 / ((2 - CORR_OFFSET) * int_time_s * cn0)));
  }
};

template<dsp::IsCode Code, IsCodeDiscriminator CodeDisc>
class CodeSignal : public Signal {
public:
  CodeSignal(u16 prn) : _prn(prn) {}

  void correlate(std::span<const cp_i16> samples_in,
                 std::span<cp_f32> out,
                 f32 sample_rate,
                 f32 mix_phase,
                 f32 mix_frequency,
                 f64 code_phase) const override {
    f32 mix_init_phase = 2.0f * std::numbers::pi * mix_phase;
    f32 mix_phase_step = -2.0f * std::numbers::pi * mix_frequency / sample_rate;
    f64 code_init_phase = std::fmod(code_phase, 1.0) / Code::CHIP_COUNT;
    code_init_phase -= CodeDisc::CORR_OFFSET * CodeDisc::SPREAD;
    if (code_init_phase < 0)
      code_init_phase += Code::CHIP_COUNT;
    f64 code_phase_step = Code::CHIP_RATE_HZ / (sample_rate * Code::CHIP_COUNT);
    dsp::mmcorr<2 * CodeDisc::SPREAD + 1, Code::MODULATION>(
        samples_in.size(), samples_in.data(), Code::CHIPS[_prn].data(), out.data(), mix_init_phase,
        mix_phase_step, CodeDisc::CORR_OFFSET, code_init_phase, code_phase_step);
  }

  f32 discriminate(std::span<cp_f32> corr_in, cp_f32& prompt) const override {
    prompt = corr_in[CodeDisc::SPREAD];
    return CodeDisc::disc_code(corr_in);
  }

  f64 disc_error_var(f64 cn0, f64 int_time_s) const override {
    return CodeDisc::disc_error_var(cn0, int_time_s);
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
          std::min(samples_out.size(), size_t((Code::CHIP_COUNT + 1) / code_phase_step));
      dsp::modulate<cp_f32, Code::MODULATION>(gen_size, Code::CHIPS[_prn].data(),
                                              samples_out.data(), mix_phase, mix_phase_step,
                                              code_init_phase, code_phase_step);

      samples_out = samples_out.subspan(gen_size);
      code_init_phase += code_phase_step * gen_size;
      code_init_phase = std::fmod(code_init_phase, Code::CHIP_COUNT);
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

  size_t n_replicas() const override { return 2 * CodeDisc::SPREAD + 1; }

  f64 code_period_s() const override { return f64(Code::CHIP_COUNT) / f64(Code::CHIP_RATE_HZ); }

  u16 prn() const override { return _prn; }

protected:
  u16 _prn;
};
}  // namespace weaver
