#pragma once
#include <random>
#include <vector>
#include <eigen3/Eigen/Core>
#include "weaver/signal.h"

namespace weaver {
struct SignalSim {
public:
  struct SignalSettings {
    std::shared_ptr<Signal> signal = nullptr;

    Eigen::Vector3d pos;
    Eigen::Vector3d vel;
    f64 code_phase = 0;
    f32 carrier_phase = 0;

    f64 cn0 = 1e5;
  };
  SignalSim(f64 sample_rate) : sample_rate(sample_rate) {}

  void generate(std::span<cp_f32> samples) {
    std::fill(samples.begin(), samples.end(), 0);
    size_t n = samples.size();
    f64 t_step = 1.0 / sample_rate;
    scratch.resize(n);
    std::normal_distribution<f32> noise(0, std::sqrt(noise_pwr));
    for (SignalSettings& signal : signals) {
      f64 carrier_freq = signal.signal->carrier_freq();
      f64 eff_vel = -signal.vel.dot(signal.pos.normalized());
      f64 code_freq = (1.0/signal.signal->code_period_s()) * (1 + eff_vel / 299792458.0f);
      signal.signal->generate(scratch, sample_rate, signal.carrier_phase, 0, signal.code_phase, code_freq);
      signal.code_phase += n * code_freq / sample_rate;
      signal.code_phase = std::fmod(signal.code_phase, 1.0);

      f64 amplitude = std::sqrt(signal.cn0 * noise_pwr);
      for (size_t i = 0; i < n; i++) {
        f64 range = signal.pos.norm();
        f64 doppler_phase = 2.0f * std::numbers::pi_v<f32> * carrier_freq * range / 299792458.0f;
        cp_f32 doppler = cp_cast<f32>(std::polar(1.0, doppler_phase));

        scratch[i] *= doppler;
        scratch[i] *= amplitude;
        scratch[i] += cp_f32(noise(mt), noise(mt));
        samples[i] += scratch[i];

        signal.pos += t_step * signal.vel;
      }
    }
  }
  std::vector<SignalSettings> signals;
private:
  std::mt19937_64 mt {std::random_device()()};
  std::vector<cp_f32, Eigen::aligned_allocator<cp_f32>> scratch;
  f64 sample_rate;
  f64 noise_pwr = 1e-4;
};
}
