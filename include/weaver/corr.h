#pragma once
#include "weaver/types.h"
#include "weaver/signal.h"

namespace weaver {
class Correlator {
public:
  struct Result {
    f64 code_disc_out;
    f64 carrier_disc_out;
    cp_f32 prompt;
    f64 int_time_s;
  };

  explicit Correlator(std::shared_ptr<Signal> signal, f64 sample_rate_hz) : signal(std::move(signal)), sample_rate_hz(sample_rate_hz) {
    reset(1.0);
  }

  span<cp_i16> process_samples(span<cp_i16> samples) {
    size_t proc_n = std::min(samples.size(), samples_rem);
    span<cp_i16> proc_span = samples.subspan(0, proc_n);
    span<cp_i16> res_span = samples.subspan(proc_n);

    signal->correlate(proc_span, corr_res, sample_rate_hz, carr_phase, carr_freq, code_phase + code_phase_offset);
    samples_rem -= proc_n;
    
    code_phase += proc_n/(sample_rate_hz * signal->code_period_s());
    if (code_phase >= 1.0)
      code_phase -= 1.0;
    carr_phase -= (carr_freq * proc_n)/sample_rate_hz;
    carr_phase = std::fmod(carr_phase, 1.0);

    return res_span;
  }

  std::optional<Result> take_result() {
    if (samples_rem)
      return std::nullopt;

    cp_f32 prompt;     
    f64 disc_out = signal->discriminate(corr_res, prompt);
    f64 carr_disc_out = disc_carrier(prompt);
    Result res {
      .code_disc_out = disc_out,
      .carrier_disc_out = carr_disc_out,
      .prompt = prompt,
      .int_time_s = int_time_s()
    };
    reset(int_time_codeper);
    return res;
  }

  void set_params(f64 code_offset, f32 carr_phase, f32 carr_freq) {
    this->code_phase_offset = code_offset;
    this->carr_phase = carr_phase;
    this->carr_freq = carr_freq;
  }

  void reset(f64 int_time_codeper) {
    this->int_time_codeper = int_time_codeper;
    corr_res.resize(signal->n_replicas());
    std::ranges::fill(corr_res, 0);
    samples_rem = int_time_s() * sample_rate_hz;
  }

  f64 int_time_s() const {
    return int_time_codeper * signal->code_period_s();
  }
private:
  f64 disc_carrier(cp_f32 prompt) const {
    return std::atan(prompt.imag() / prompt.real()) / (2 * std::numbers::pi_v<f32>);
  }

  std::shared_ptr<Signal> signal;
  f64 sample_rate_hz;

  f64 int_time_codeper;
  size_t samples_rem;
  std::vector<cp_f32> corr_res;

  f64 code_phase;

  f64 code_phase_offset;
  f32 carr_phase;
  f32 carr_freq;
};
}
