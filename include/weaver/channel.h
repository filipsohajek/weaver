#pragma once
#include "weaver/acq.h"
#include "weaver/corr.h"
#include "weaver/kalman.h"
#include "weaver/loop_filter.h"
#include "weaver/signal.h"
#include "weaver/types.h"
#include "weaver/util.h"

namespace weaver {
class Channel {
public:
  enum class CodeDiscriminator { EMLEnvelope };

  enum class CarrierDiscriminator { ATan };

  struct Parameters {
    AcqEngine::Parameters acq_params;
    f64 sample_rate_hz;
    std::vector<f64> corr_offsets;
    f64 acq_p_thresh = 0.05;

    std::unique_ptr<LoopFilter> filter;
    size_t cn0_est_prompts = 20;

    CodeDiscriminator code_disc;
    CarrierDiscriminator carrier_disc;
  };

  enum class State {
    FAILED,
    ACQUISITION,
    TRACK_INIT,
  };

  explicit Channel(std::shared_ptr<Signal> signal, Parameters params)
      : trace_file("out_trace"),
        corr(signal, params.sample_rate_hz, params.corr_offsets),
        signal(signal),
        acq(signal, params.acq_params),
        params(std::move(params)) {
    state = State::ACQUISITION;
  }

  void process_samples(span<cp_i16> samples) {
    while (!samples.empty()) {
      switch (state) {
        case State::FAILED:
          return;
        case State::ACQUISITION: {
          samples = acq.process(samples);
          if (!acq.finished())
            continue;
          AcqEngine::Result result = acq.result().value();
          std::cout << std::format("acquisition: code_offset={}, doppler_freq={}, p={}\n",
                                   result.code_offset, result.doppler_freq, result.p);
          if (result.p >= params.acq_p_thresh) {
            std::cout << "acquisition failed, failing channel\n";
            state = State::FAILED;
            continue;
          }
          setup_tracking(result);
          state = State::TRACK_INIT;
          break;
        }
        case State::TRACK_INIT:
          samples = process_track(samples);
          continue;
      }
    }
  }

private:
  span<cp_i16> process_track(span<cp_i16> samples) {
    span<cp_i16> res_samples = corr.process_samples(samples);
    if (!corr.has_result())
      return res_samples;

    span<const cp_f32> corr_res = corr.result();
    cp_f32 prompt = corr_res[corr_res.size() / 2];
    f64 carrier_disc_out = disc_carrier(prompt);
    f64 code_disc_out = -disc_code(corr_res);

    f64 cn0_db = 10 * std::log10(cn0());
    trace_file.write(reinterpret_cast<char*>(&prompt), sizeof(cp_f32));
    trace_file.write(reinterpret_cast<char*>(&cn0_db), sizeof(f64));
    trace_file.write(reinterpret_cast<char*>(&code_disc_out), sizeof(f64));
    trace_file.write(reinterpret_cast<char*>(&carrier_disc_out), sizeof(f64));
    trace_file.write(reinterpret_cast<char*>(&corr.code_phase), sizeof(f64));
    trace_file.write(reinterpret_cast<char*>(&corr.carr_phase), sizeof(f64));
    trace_file.write(reinterpret_cast<char*>(&corr.carr_freq), sizeof(f32));
    trace_file.write(reinterpret_cast<char*>(&corr.code_freq), sizeof(f64));
    trace_file.flush();

    std::cout << std::format(
        "process_track: prompt.re={}, prompt.im={}, code_disc={}, carr_disc={}, cn0={}\n",
        prompt.real(), prompt.imag(), code_disc_out, carrier_disc_out, cn0_db);
    update_cn0(prompt);
    params.filter->update_disc_statistics(code_disc_var(), carrier_disc_var());
    update_filter(code_disc_out, carrier_disc_out);

    corr.reset();

    return res_samples;
  }

  void update_filter(f64 code_disc_out, f64 carrier_disc_out) {
    LoopFilter::Output filter_out = params.filter->update(code_disc_out, carrier_disc_out);

    corr.update_params(filter_out.carrier_freq.value_or(corr.carr_freq),
                       filter_out.code_freq.value_or(corr.code_freq),
                       filter_out.carr_phase_adj.value_or(0),
                       filter_out.code_phase_adj.value_or(0));
  }

  void update_cn0(cp_f32 prompt) {
    cn0_prompts.push_back(prompt);
    if (cn0_prompts.size() > params.cn0_est_prompts)
      cn0_prompts.pop_front();
    cn0_n_rem -= 1;
    if (cn0_n_rem == 0) {
      cn0_n_rem = params.cn0_est_prompts;
      f64 m2 = 0;
      f64 m4 = 0;
      for (cp_f32 prompt : cn0_prompts) {
        f64 prompt_mag = std::abs(prompt);
        m2 += std::pow(prompt_mag, 2);
        m4 += std::pow(prompt_mag, 4);
      }
      m2 /= cn0_prompts.size();
      m4 /= cn0_prompts.size();

      f64 pd = std::sqrt(2 * std::pow(m2, 2) - m4);
      if (std::isnan(pd)) {
        cur_cn0 = 1e3;
        return;
      }
      f64 pn = m2 - pd;
      cur_cn0 = (pd / pn) / corr.int_time_s();
    }
  }

  f64 cn0() const { return cur_cn0; }

  void set_int_time(f64 int_time_codeper) {
    corr.reset(int_time_codeper);
    f64 int_time_s = corr.int_time_s();
    params.filter->set_int_time(int_time_s);
  }

  void setup_tracking(AcqEngine::Result res) {
    params.filter->init(signal.get(), res);
    corr.update_params(res.doppler_freq, 1 / signal->code_period_s(), 0, res.code_offset);
    cn0_n_rem = params.cn0_est_prompts;
    set_int_time(1.0);
  }

  f64 carrier_disc_var() const {
    switch (params.carrier_disc) {
      case CarrierDiscriminator::ATan:
        return (1 / (2 * cn0() * corr.int_time_s())) * (1 + 1 / (2 * corr.int_time_s() * cn0())) /
               std::pow(2 * std::numbers::pi_v<f32>, 2);
      default:
        return nan("");
    }
  }

  f64 disc_carrier(cp_f32 prompt) const {
    switch (params.carrier_disc) {
      case CarrierDiscriminator::ATan:
        return std::atan(prompt.imag() / prompt.real()) / (2 * std::numbers::pi_v<f32>);
      default:
        return nan("");
    }
  }

  f64 code_disc_var() const {
    switch (params.code_disc) {
      case CodeDiscriminator::EMLEnvelope: {
        f64 d = 2 * corr.offsets()[corr.offsets().size() - 1];
        return ((d / (4 * cn0() * corr.int_time_s())) *
                (1 + 2 / ((2 - d) * corr.int_time_s() * cn0())));
      }
      default:
        return nan("");
    }
  }

  f64 disc_code(span<const cp_f32> corr_res) const {
    switch (params.code_disc) {
      case CodeDiscriminator::EMLEnvelope: {
        f64 e = std::abs(corr_res[corr.offsets().size() - 1]);
        f64 l = std::abs(corr_res[corr.offsets().size() + 1]);
        return (0.5 * (e - l) / (e + l)) / signal->chip_count();
      }
      default:
        return nan("");
    }
  }

  std::deque<cp_f32> cn0_prompts;
  size_t cn0_n_rem;
  f64 cur_cn0 = 1e3;

  std::ofstream trace_file;
  State state;
  Correlator corr;
  std::shared_ptr<Signal> signal;
  AcqEngine acq;
  Parameters params;
};
}  // namespace weaver
