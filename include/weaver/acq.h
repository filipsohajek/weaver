#pragma once
#include <format>
#include <ranges>
#include <list>
#include <fstream>
#include <queue>
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
    size_t n_noncoh_candidates = 40;

    f64 merge_thresh_s = 0.5e-6;

    f64 doppler_min = -7000;
    f64 doppler_max = 7000;
  };

  struct Candidate {
    f64 code_offset;
    size_t n_noncoh;
    f32 doppler_freq;
    f32 val;

    bool has_same_pos(const Candidate& other) const {
      return (other.code_offset == code_offset) && (other.doppler_freq == doppler_freq);
    }

    void update(const Candidate& other) {
      n_noncoh += other.n_noncoh;
      val += other.val;
    }

    bool operator<(const Candidate& other) const {
      return val < other.val;
    }

    bool operator>(f32 other_val) const {
      return val > other_val;
    }

    f64 p_val() const {
      return weaver::chi2_cdf(2*n_noncoh, val);
    }
  };

  AcqEngine(std::unique_ptr<Signal> signal) : signal(std::move(signal)) {
//    fft.SetFlag(Eigen::FFT<f32>::Flag::Unscaled);
  }

  void reset(Parameters params) {
    this->params = params;
    candidates.clear();

    replica_fft.resize(fft_len());
    samples.resize(fft_len());
    scratch.resize(fft_len());
    scratch2.resize(fft_len());

    auto& replica = scratch;
    signal->generate(std::span<cp_f32>(replica).subspan(signal_len()), params.sample_rate_hz, 0, 0,
                     0);
    fft.fwd(replica_fft.data(), replica.data(), fft_len());
    dsp::conj_cpf32(fft_len(), replica_fft.data(), replica_fft.data());

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

  const std::list<Candidate>& acq_candidates() const {
    return candidates;
  }
private:
  size_t signal_len() const {
    return params.n_coherent * signal->code_period_s() * params.sample_rate_hz;
  }

  size_t fft_len() const { return 2 * signal_len(); }

  f64 residual_code_phase() const {
    f64 res_samples = (noncoh_rem - params.n_noncoherent) * (params.n_coherent * signal->code_period_s() * params.sample_rate_hz - signal_len());
    return res_samples / (params.sample_rate_hz * signal->code_period_s());
  }

  void acq_single_bin(std::span<cp_f32> samples_fft,
                      ssize_t doppler_shift,
                      std::list<Candidate>& candidates,
                      f64 real_var, f64 imag_var) {
    auto& mul_res = scratch;
    auto& corr_res = scratch2;
    dsp::mul_shift_cpf32(fft_len(), samples_fft.data(), replica_fft.data(), doppler_shift,
                         mul_res.data());
    fft.inv(corr_res.data(), mul_res.data(), fft_len());

    std::priority_queue<Candidate> cand_queue;
    for (size_t code_i = 0; code_i < fft_len(); code_i++) {
      f32 cand_real = (corr_res[code_i].real()) / (std::sqrt(fft_len() * real_var));
      f32 cand_imag = (corr_res[code_i].imag()) / (std::sqrt(fft_len() * imag_var));
      f32 cand_val = std::pow(cand_real, 2) + std::pow(cand_imag, 2);

      if (!cand_queue.empty() && (cand_queue.top() > cand_val))
        continue;

      f64 code_offset = (fft_len() - code_i) / (signal->code_period_s() * params.sample_rate_hz);
      code_offset -= residual_code_phase();
      code_offset = std::fmod(code_offset, 1.0);
      f32 doppler_freq = doppler_shift / (2 * signal->code_period_s() * params.n_coherent);

      Candidate new_cand = {
          .code_offset = code_offset, .n_noncoh = 1, .doppler_freq = doppler_freq, .val = cand_val};
      if (cand_queue.size() == params.n_noncoh_candidates)
        cand_queue.pop();
      cand_queue.emplace(new_cand);
    }

    
    while (!cand_queue.empty()) {
      Candidate cand = cand_queue.top();
      cand_queue.pop();
      auto insert_it = std::ranges::upper_bound(candidates, cand.val, {}, [](const Candidate& proj_cand) {
        return proj_cand.val;
      });
      if ((insert_it == candidates.begin()) && (candidates.size() >= params.n_noncoh_candidates))
        return;
      candidates.insert(insert_it, cand);
      if (candidates.size() > params.n_noncoh_candidates)
        candidates.pop_front();
    }
  }

  void merge_candidates() {
    candidates.sort([](const Candidate& lhs, const Candidate& rhs) {
      if (lhs.doppler_freq < rhs.doppler_freq)
        return true;
      if (lhs.doppler_freq > rhs.doppler_freq)
        return false;
      if (lhs.code_offset < rhs.code_offset)
        return true;
      if (lhs.code_offset > rhs.code_offset)
        return false;
      return true;
    }); 

    f64 merge_thresh = params.merge_thresh_s / signal->code_period_s();
    for (auto it = candidates.begin(); it != candidates.end(); it++) {
      Candidate& ref_cand = *it;
      auto next_it = it;
      next_it++;
      auto merge_end_it = std::ranges::find_if(next_it, candidates.end(), [&](const Candidate& search_cand) {
        return (ref_cand.doppler_freq != search_cand.doppler_freq) || (std::abs(ref_cand.code_offset - search_cand.code_offset) > merge_thresh);
      });
      for (auto merge_it = next_it; merge_it != merge_end_it; merge_it++) {
        const Candidate& merge_cand = *merge_it;
        ref_cand.code_offset = (ref_cand.code_offset * ref_cand.val + merge_cand.code_offset * merge_cand.val) / (ref_cand.val + merge_cand.val);
        ref_cand.val += merge_cand.val; 
        ref_cand.n_noncoh += 1;
      }
      candidates.erase(next_it, merge_end_it);
    }
    
    candidates.sort([](const Candidate& lhs, const Candidate& rhs) {
      return lhs.val < rhs.val;
    });
  }

  void acq_single_coh() {
    auto real_samples = samples | std::views::take(signal_len()) | std::views::transform([](cp_f32 cp) { return cp.real(); });
    f64 real_var = sample_variance(real_samples);
    auto imag_samples = samples | std::views::take(signal_len()) | std::views::transform([](cp_f32 cp) { return cp.imag(); });
    f64 imag_var = sample_variance(imag_samples);

    std::fill(samples.begin() + signal_len(), samples.end(), 0);
    fft.fwd(scratch.data(), samples.data(), fft_len());
    std::copy(scratch.begin(), scratch.end(), samples.begin());
    auto& samples_fft = samples;

    f64 doppler_step = 1.0 / (2 * signal->code_period_s() * params.n_coherent);
    ssize_t doppler_shift_min = params.doppler_min / doppler_step;
    ssize_t doppler_shift_max = params.doppler_max / doppler_step;

    for (ssize_t doppler_shift = doppler_shift_min; doppler_shift <= doppler_shift_max;
         doppler_shift++) {
      acq_single_bin(samples_fft, doppler_shift, candidates, real_var, imag_var);
    }
    merge_candidates();
  }

  Parameters params;
  Eigen::FFT<f32> fft;

  std::unique_ptr<Signal> signal;
  aligned_vector<cp_f32> replica_fft;
  aligned_vector<cp_f32> samples;
  aligned_vector<cp_f32> scratch;
  aligned_vector<cp_f32> scratch2;

  std::list<Candidate> candidates;

  size_t samples_rem;
  size_t noncoh_rem;
};
}  // namespace weaver
