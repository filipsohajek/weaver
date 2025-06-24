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
    size_t n_noncoh_candidates = 20;

    f64 doppler_min = -7500;
    f64 doppler_max = 7500;
    size_t doppler_step = 1;
  };

  struct Candidate {
    f64 code_offset;
    size_t chi2_dof;
    size_t max_count;
    f32 doppler_freq;
    f32 val;

    bool operator>(const Candidate& other) const {
      return val > other.val;
    }
    bool operator<(const Candidate& other) const {
      return val < other.val;
    }

    f64 p_val() const { return 1 - std::pow(weaver::chi2_cdf(chi2_dof, val), max_count); }
  };

  AcqEngine(std::shared_ptr<Signal> signal) : signal(std::move(signal)) {}

  void reset(Parameters params) {
    this->params = params;
    candidates.clear();

    replica_fft.resize(fft_len());
    samples.resize(fft_len());
    scratch.resize(fft_len());
    scratch2.resize(fft_len());

    auto& replica = scratch;
    signal->generate(replica, params.sample_rate_hz, 0, 0, 0);
    fft.fwd(replica_fft.data(), replica.data(), fft_len());

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
      }
    }

    return samples_in;
  }

  bool finished() const { return noncoh_rem == 0; }

  const std::list<Candidate>& acq_candidates() const { return candidates; }

private:
  size_t signal_len() const {
    return params.n_coherent * signal->code_period_s() * params.sample_rate_hz;
  }

  size_t fft_len() const { return 2 * signal_len(); }

  void acq_single_bin(std::span<cp_f32> samples_fft,
                      ssize_t doppler_shift,
                      std::list<Candidate>& candidates) {
    auto& mul_res = scratch;
    auto& corr_res = scratch2;
    dsp::mul_shift_cpf32(fft_len(), replica_fft.data(), samples_fft.data(), doppler_shift,
                         mul_res.data());
    fft.inv(corr_res.data(), mul_res.data(), fft_len());

    size_t replica_n_samples = signal->code_period_s() * params.sample_rate_hz;
    f64 doppler_step = params.doppler_step / (fft_len() / params.sample_rate_hz);
    for (size_t code_i = 0; code_i < replica_n_samples; code_i++) {
      f32 cand_val = std::pow(std::abs(corr_res[code_i]), 2);

      f64 code_offset = code_i / (signal->code_period_s() * params.sample_rate_hz);
      f32 doppler_freq = doppler_shift / f64(fft_len() / params.sample_rate_hz);

      Candidate cand = {
          .code_offset = code_offset,
          .chi2_dof =
              2 * params.n_coherent * params.n_coherent,
          .max_count = size_t(replica_n_samples * ((params.doppler_max - params.doppler_min) / doppler_step + 1)),
          .doppler_freq = doppler_freq,
          .val = cand_val
      };

      bool inserted_cand = false;
      for (auto& exist_cand : candidates) {
        if ((exist_cand.code_offset == cand.code_offset) && (exist_cand.doppler_freq == cand.doppler_freq)) {
          exist_cand.chi2_dof += cand.chi2_dof;
          exist_cand.val += cand.val;
          candidates.sort();
          inserted_cand = true;
          break;
        }
      }

      if (inserted_cand) 
        continue;

      auto insert_it = std::ranges::upper_bound(
          candidates, cand.val, {}, [](const Candidate& proj_cand) { return proj_cand.val; });
      candidates.insert(insert_it, cand);
      if (candidates.size() > params.n_noncoh_candidates)
        candidates.pop_front();
    }
  }

  void acq_single_coh() {
    f32 sample_scale = std::sqrt(0.5 * signal_len() * sample_variance(samples | std::views::take(signal_len())));

    std::fill(samples.begin() + signal_len(), samples.end(), 0);
    fft.fwd(scratch.data(), samples.data(), fft_len());
    std::copy(scratch.begin(), scratch.end(), samples.begin());

    auto& samples_fft = samples;
    for (auto& sample_bin : samples_fft) {
      sample_bin = std::conj(sample_bin) / sample_scale;
    }

    f64 doppler_step = params.doppler_step / (fft_len() / params.sample_rate_hz);
    ssize_t doppler_shift_min = params.doppler_min / doppler_step;
    ssize_t doppler_shift_max = params.doppler_max / doppler_step;

    for (ssize_t doppler_shift = doppler_shift_min; doppler_shift <= doppler_shift_max;
         doppler_shift++) {
      acq_single_bin(samples_fft, params.doppler_step * doppler_shift, candidates);
    }
  }

  Parameters params;
  Eigen::FFT<f32> fft;

  std::shared_ptr<Signal> signal;
  aligned_vector<cp_f32> replica_fft;
  aligned_vector<cp_f32> samples;
  aligned_vector<cp_f32> scratch;
  aligned_vector<cp_f32> scratch2;

  std::list<Candidate> candidates;

  size_t samples_rem;
  size_t noncoh_rem;
};
}  // namespace weaver
