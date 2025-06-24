#pragma once
#include "weaver/acq.h"
#include "weaver/corr.h"
#include "weaver/signal.h"
#include "weaver/types.h"
#include "weaver/kalman.h"
#include "weaver/util.h"

namespace weaver {
class Channel {
public:
  struct Parameters {
    f64 sample_rate_hz;
    AcqEngine::Parameters acq_params;
    f64 acq_p_thresh = 0.05;

    f32 init_code_disc_var = 1e-6;
    f32 init_carrier_disc_var = 1e-4;

    f64 code_noise_var = 0;
    f64 phase_noise_var = 1e-3;
    f64 freq_noise_var = 1e-2;
    f64 freq_rate_noise_var = 0;

    f64 code_init_var = 1e-3;
    f64 phase_init_var = 1e-3;
    f64 freq_init_var = 1e-3;
    f64 freq_rate_init_var = 1e-3;

    size_t cn0_est_prompts = 100;
  };

  enum class State {
    FAILED,
    ACQUISITION,
    TRACK_INIT,
  };

  explicit Channel(std::shared_ptr<Signal> signal, Parameters params) : cn0_m2(params.cn0_est_prompts), cn0_m4(params.cn0_est_prompts), trace_file("out_trace"), params(params), signal(signal), acq(signal, params.acq_params), corr(signal, params.sample_rate_hz) {
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
          std::cout << std::format("acquisition: code_offset={}, doppler_freq={}, p={}\n", result.code_offset, result.doppler_freq, result.p);
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
    auto corr_opt = corr.take_result();
    if (!corr_opt.has_value())
      return res_samples;
    Correlator::Result corr_res = corr_opt.value();

    f64 cn0_db = 10 * std::log10(cn0());
    trace_file.write(reinterpret_cast<char*>(&corr_res.prompt), sizeof(cp_f32));
    trace_file.write(reinterpret_cast<char*>(&cn0_db), sizeof(f64));
    trace_file.write(reinterpret_cast<char*>(&corr_res.code_disc_out), sizeof(f64));
    trace_file.write(reinterpret_cast<char*>(&corr_res.carrier_disc_out), sizeof(f64));
    trace_file.write(reinterpret_cast<char*>(trk_filter.state.data()), 4 * sizeof(f64));
    trace_file.write(reinterpret_cast<char*>(trk_filter.state_cov.data()), 16 * sizeof(f64));
    trace_file.write(reinterpret_cast<char*>(trk_filter.meas_noise_cov.data()), 4 * sizeof(f64));
    trace_file.flush();

    std::cout << std::format("process_track: prompt.re={}, prompt.im={}, code_disc={}, carr_disc={}, cn0={}\n", corr_res.prompt.real(), corr_res.prompt.imag(), corr_res.code_disc_out, corr_res.carrier_disc_out, cn0_db);
    update_cn0(corr_res);
    update_filter(corr_res);

    return res_samples;
  }

  void update_filter(Correlator::Result corr_res) {
    f64 cn0 = this->cn0();
    trk_filter.meas_noise_cov = Eigen::DiagonalMatrix<f64, 2> {signal->disc_error_var(cn0, corr.int_time_s()), corr.carrier_disc_var(cn0)};
    trk_filter.update({corr_res.code_disc_out, -corr_res.carrier_disc_out});  
    trk_filter.state[0] = std::fmod(trk_filter.state[0], 1.0);
    trk_filter.state[1] = std::fmod(trk_filter.state[1], 1.0);
    f64 code_offset = trk_filter.state[0];
    f64 carr_phase = trk_filter.state[1];
    f64 carr_freq = trk_filter.state[2];

    corr.set_params(code_offset, carr_phase, carr_freq);
    std::cout << std::format("update_filter: code_offset={}, carr_phase={}, carr_freq={}, carr_freq_rate={}\n", code_offset, carr_phase, carr_freq, trk_filter.state[3]);
  }

  void update_cn0(Correlator::Result corr_res) {
    f64 m2 = std::pow(corr_res.prompt.real(), 2) + std::pow(corr_res.prompt.imag(), 2);
    cn0_m2.add(m2);
    cn0_m4.add(std::pow(m2, 2));
  }

  f32 cn0() const {
    if (cn0_m2.cur_n() < params.cn0_est_prompts)
      return 1e3;
    f64 pd = std::sqrt(std::max(2 * std::pow(cn0_m2.cur_avg(), 2) - cn0_m4.cur_avg(), 0.0));
    if (pd == 0.0)
      return 1e1;
    f64 pn = cn0_m2.cur_avg() - pd;
    return (pd/pn) / corr.int_time_s();
  }

  void set_int_time(f64 int_time_codeper) {
    corr.reset(int_time_codeper);
    f64 int_time_s = corr.int_time_s();

    trk_filter.meas_matrix = Eigen::Matrix<f64, 2, 4> {
      {1, 0, 0, 0},
      {0, 1, 0, 0}
    };
    f64 aiding_factor = 1.0 / (signal->carrier_freq() * signal->code_period_s());
    trk_filter.state_step_matrix = Eigen::Matrix<f64, 4, 4> {
      {1, 0, aiding_factor * int_time_s, 0.5 * aiding_factor * int_time_s * int_time_s},
      {0, 1, int_time_s, 0.5 * int_time_s * int_time_s},
      {0, 0, 1, int_time_s},
      {0, 0, 0, 1}
    };
  }

  void setup_tracking(AcqEngine::Result res) {
    corr.set_params(res.code_offset, 0.0f, res.doppler_freq);
    trk_filter.state = Eigen::Vector<f64, 4> {res.code_offset, 0.0f, res.doppler_freq, 0};
    trk_filter.state_cov = Eigen::DiagonalMatrix<f64, 4> {params.code_init_var, params.phase_init_var, params.freq_init_var, params.freq_rate_init_var};
    trk_filter.process_noise_cov = Eigen::DiagonalMatrix<f64, 4> {params.code_noise_var, params.phase_noise_var, params.freq_noise_var, params.freq_rate_noise_var};
    trk_filter.meas_noise_cov = Eigen::DiagonalMatrix<f64, 2> {params.init_code_disc_var, params.init_carrier_disc_var};
    set_int_time(1.0);
  }

  MovingAverage<f64> cn0_m2;
  MovingAverage<f64> cn0_m4;

  std::ofstream trace_file;
  State state;
  Parameters params;
  std::shared_ptr<Signal> signal;
  AcqEngine acq;
  Correlator corr;
  KalmanFilter<4, 2> trk_filter;
};
}
