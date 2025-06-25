#pragma once

#include "weaver/kalman.h"
#include "weaver/loop_filter.h"

namespace weaver {
struct KalmanLoopFilter : public LoopFilter {
  struct Parameters {
    f32 init_code_disc_var = 1e-6;
    f32 init_carrier_disc_var = 1e-8;

    f64 code_noise_var = 1e-7;
    f64 phase_noise_var = 1e-6;
    f64 freq_noise_var = 1e-5;
    f64 freq_rate_noise_var = 0;

    f64 code_init_var = 1e-3;
    f64 phase_init_var = 1;
    f64 freq_init_var = 500 * 500;
    f64 freq_rate_init_var = 1;

    bool code_aiding = true;
  };

  explicit KalmanLoopFilter(Parameters params) : params(params) {}

  void init(const Signal* signal, const AcqEngine::Result& acq_result) override {
    code_freq = 1 / signal->code_period_s();
    code_carrier_freq = signal->carrier_freq();

    filter.state = {0, 0, 0, 0};
    doppler_freq = acq_result.doppler_freq;
    filter.state_cov =
        Eigen::DiagonalMatrix<f64, 4>{params.code_init_var, params.phase_init_var,
                                      params.freq_init_var, params.freq_rate_init_var};
    filter.process_noise_cov =
        Eigen::DiagonalMatrix<f64, 4>{params.code_noise_var, params.phase_noise_var,
                                      params.freq_noise_var, params.freq_rate_noise_var};
    filter.meas_noise_cov =
        Eigen::DiagonalMatrix<f64, 2>{params.init_code_disc_var, params.init_carrier_disc_var};
  }

  Output update(f64 code_disc_out, f64 carr_disc_out) override {
    filter.update({code_disc_out, carr_disc_out});
    doppler_freq += filter.state(2);
    f64 code_phase_err = filter.state(0);
    f64 carr_phase_err = filter.state(1);

    filter.state(0) = 0;
    filter.state(1) = 0;
    filter.state(2) = 0;

    f64 code_freq = this->code_freq;
    if (params.code_aiding)
      code_freq += aiding_factor() * doppler_freq;
    std::cout << std::format("kalman: update: carrier_freq={:.2f}, code_freq={:.2f}, code_phase_adj={:.6f}, carr_phase_adj={:.6f}\n", doppler_freq, code_freq, code_phase_err, carr_phase_err);

    return {.code_freq = code_freq,
            .carrier_freq = doppler_freq,
            .code_phase_adj = code_phase_err,
            .carr_phase_adj = carr_phase_err};
  }

  void update_disc_statistics(f64 code_disc_var, f64 carr_disc_var) override {
    filter.meas_noise_cov = Eigen::DiagonalMatrix<f64, 2>{code_disc_var, carr_disc_var};
  }

  void set_int_time(f64 int_time_s) override {
    filter.state_step_matrix = Eigen::Matrix4d{
        {1, 0, aiding_factor() * int_time_s, 0.5 * aiding_factor() * std::pow(int_time_s, 2)},
        {0, 1, int_time_s, 0.5 * std::pow(int_time_s, 2)},
        {0, 0, 1, int_time_s},
        {0, 0, 0, 1}};
    filter.meas_matrix = Eigen::Matrix<f64, 2, 4> {
      {1, 0, -0.5 * aiding_factor() * int_time_s, -(1/6.0) * aiding_factor() * std::pow(int_time_s, 2)},
      {0, 1, -int_time_s, -(1/6.0) * std::pow(int_time_s, 2)}
    };
  }

private:
  f64 aiding_factor() const { return code_freq / code_carrier_freq; }

  f64 code_carrier_freq;
  f64 code_freq;

  Parameters params;
  f64 doppler_freq;
  KalmanFilter<4, 2> filter;
};
}  // namespace weaver
