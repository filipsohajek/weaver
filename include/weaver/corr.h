#pragma once
#include <format>
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

  explicit Correlator(std::shared_ptr<Signal> signal, f64 sample_rate_hz, std::vector<f64> corr_offsets) : signal(std::move(signal)), corr_offsets(corr_offsets), sample_rate_hz(sample_rate_hz) {
    update_params(0, 1/this->signal->code_period_s(), 0, 0);
    reset(1.0);
  }

  span<cp_i16> process_samples(span<cp_i16> samples) {
    size_t proc_n = std::min(samples.size(), samples_rem);
    span<cp_i16> proc_span = samples.subspan(0, proc_n);
    span<cp_i16> res_span = samples.subspan(proc_n);

    signal->correlate(proc_span, corr_res, sample_rate_hz, carr_phase, carr_freq, code_phase, code_freq, corr_offsets);
    samples_rem -= proc_n;
    
    code_phase += (code_freq * proc_n)/sample_rate_hz;
    carr_phase += (carr_freq * proc_n)/sample_rate_hz;

    return res_span;
  }

  bool has_result() const {
    return !samples_rem;
  }

  span<const cp_f32> result() const {
    return corr_res;
  }

  void update_params(f64 carr_freq, f64 code_freq, f64 carr_phase_adj, f64 code_phase_adj) {
    this->carr_freq = carr_freq;
    this->code_freq = code_freq;
    this->carr_phase = std::fmod(carr_phase + carr_phase_adj, 1.0);
    this->code_phase = std::fmod(code_phase + code_phase_adj, 1.0);
  }

  void reset(f64 int_time_codeper = -1, f64 max_align_codeper = 0.0) {
    if (int_time_codeper < 0)
      int_time_codeper = this->int_time_codeper;

    f64 align_time;
    if (code_phase < 0.5)
      align_time = -std::min(max_align_codeper, code_phase);
    else
      align_time = std::min(max_align_codeper, 1 - code_phase);
    std::cout << std::format("correlator({}): code_phase={}, int_time_codeper={}, max_align_codeper={}, align_time={}, ", signal->id().prn, code_phase, int_time_codeper, max_align_codeper, align_time);

    this->int_time_codeper = int_time_codeper + align_time;
    std::cout << std::format("res_int_time_codeper={}\n", this->int_time_codeper);
    corr_res.resize(2 * corr_offsets.size() + 1);
    std::ranges::fill(corr_res, 0);
    samples_rem = target_n_samples();
  }

  f64 int_time_s() const {
    return int_time_codeper/code_freq;
  }

  span<const f64> offsets() const {
    return corr_offsets;
  }

  f64 elapsed_time_s() const {
    return (target_n_samples() - samples_rem) / sample_rate_hz;
  }

  f64 code_phase = 0;
  f64 carr_phase = 0;
  f32 carr_freq = 0;
  f64 code_freq = 0;
private:
  f64 target_n_samples() const {
    return int_time_codeper * (sample_rate_hz / code_freq);
  }

  std::shared_ptr<Signal> signal;
  std::vector<f64> corr_offsets;
  f64 sample_rate_hz;

  f64 int_time_codeper = 0;
  size_t samples_rem = 0;
  std::vector<cp_f32> corr_res;
};
}
