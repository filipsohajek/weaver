#pragma once
#include <format>
#include <fstream>
#include <list>
#include <queue>
#include <ranges>
#include <unsupported/Eigen/FFT>

#include "weaver/math.h"
#include "weaver/signal.h"
#include "weaver/types.h"

namespace weaver {
class AcqEngine {
public:
  struct Parameters {
    f64 sample_rate_hz = 0;

    size_t n_coherent = 1;
    size_t n_noncoherent = 1;

    f64 doppler_min = -7500;
    f64 doppler_max = 7500;
    size_t doppler_step = 1;
  };

  struct Result {
    f64 code_offset;
    f64 p;
    f32 doppler_freq;
    f32 doppler_freq_std;
  };

  AcqEngine(std::shared_ptr<Signal> signal, Parameters params) : params(params), signal(std::move(signal)), acq_grid(n_doppler_bins(), n_code_bins()), replica_fft(fft_len()), samples(fft_len()), scratch(fft_len()), scratch2(fft_len()) {
    reset();
  }
  AcqEngine(std::shared_ptr<Signal> signal) : weaver::AcqEngine(signal, Parameters()) {}

  void reset() {
    acq_grid.setZero();

    auto& replica = scratch;
    signal->generate({replica.data(), fft_len()}, params.sample_rate_hz, 0, 0, 0);
    fft.fwd(replica_fft, replica);

    samples_rem = signal_len();
    noncoh_rem = params.n_noncoherent;
  }

  std::span<cp_i16> process(std::span<cp_i16> samples_in) {
    while (noncoh_rem && !samples_in.empty()) {
      size_t process_n = std::min(samples_in.size(), samples_rem);
      dsp::cvt_cpi16_cpf32(process_n, samples_in.data(),
                           samples.data() + (signal_len() - samples_rem));
      samples_rem -= process_n;
      samples_in = samples_in.subspan(process_n);
      if (samples_rem == 0) {
        acq_single_coh();
        samples_rem = signal_len();
        noncoh_rem -= 1;
        if (noncoh_rem == 0)
          acq_result = acq_search();
      }
    }

    return samples_in;
  }

  bool finished() const {
    return noncoh_rem == 0;
  }

  std::optional<Result> result() const {
    if (!finished())
      return std::nullopt;
    return acq_result;
  }

private:
  size_t n_code_bins() const {
    return signal->code_period_s() * params.sample_rate_hz;
  }

  size_t signal_len() const {
    return params.n_coherent * signal->code_period_s() * params.sample_rate_hz;
  }

  size_t fft_len() const { return 2 * signal_len(); }

  f64 doppler_step() const {
    return params.doppler_step / (fft_len() / params.sample_rate_hz);
  }

  size_t n_doppler_bins() const {
    return (max_doppler_bin() - min_doppler_bin()) + 1;
  }

  ssize_t min_doppler_bin() const {
    return params.doppler_min / doppler_step();
  }

  ssize_t max_doppler_bin() const {
    return params.doppler_max / doppler_step();
  }

  void acq_single_bin(Eigen::VectorXcf samples_fft, ssize_t doppler_bin) {
    auto& mul_res = scratch;
    auto& corr_res = scratch2;
    dsp::mul_shift_cpf32(fft_len(), replica_fft.data(), samples_fft.data(), doppler_bin * params.doppler_step, mul_res.data());
    fft.inv(corr_res, mul_res);

    size_t doppler_grid_bin = doppler_bin - min_doppler_bin();
    acq_grid.row(doppler_grid_bin) += (corr_res.cwiseAbs2()(Eigen::seq(0, n_code_bins() - 1))).transpose();
  }

  void acq_single_coh() {
    f32 sample_scale = std::sqrt(0.5 * signal_len() * sample_variance(samples | std::views::take(signal_len())));

    std::fill(samples.begin() + signal_len(), samples.end(), 0);
    fft.fwd(scratch.data(), samples.data(), fft_len());
    std::copy(scratch.begin(), scratch.end(), samples.begin());

    auto& samples_fft = samples;
    samples_fft = samples_fft.conjugate() / sample_scale;

    for (ssize_t doppler_bin = min_doppler_bin(); doppler_bin <= max_doppler_bin(); doppler_bin++) {
      acq_single_bin(samples_fft, doppler_bin);
    }
  }

  [[nodiscard]] Result acq_search() const {
    size_t doppler_grid_bin, code_bin;
    acq_grid.maxCoeff(&doppler_grid_bin, &code_bin);
    f64 max_val = acq_grid(doppler_grid_bin, code_bin);

    f32 doppler = (ssize_t(doppler_grid_bin) + min_doppler_bin()) * doppler_step();
    f64 code_offset = f64(code_bin) / n_code_bins();

    size_t dof = 2 * params.n_coherent * params.n_noncoherent;
    size_t max_count = n_code_bins() * n_doppler_bins();
    f64 p = 1 - std::pow(chi2_cdf(dof, max_val), max_count);

    return {
      .code_offset = code_offset,
      .p = p,
      .doppler_freq = doppler,
      .doppler_freq_std = f32(doppler_step())
    };
  }

  Parameters params;
  std::shared_ptr<Signal> signal;
  Eigen::FFT<f32> fft;

  Eigen::Matrix<f32, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> acq_grid;
  Eigen::VectorXcf replica_fft;
  Eigen::VectorXcf samples;
  Eigen::VectorXcf scratch;
  Eigen::VectorXcf scratch2;

  Result acq_result;

  size_t samples_rem;
  size_t noncoh_rem;
};
}  // namespace weaver
