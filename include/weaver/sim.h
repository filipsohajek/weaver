#pragma once
#include <random>
#include <vector>
#include <eigen3/Eigen/Core>
#include "weaver/signal.h"

namespace weaver {
struct SignalSim {
public:
  struct SignalSettings {
    std::unique_ptr<Signal> signal = nullptr;

    Eigen::Vector3d pos;
    Eigen::Vector3d vel;
    f64 code_phase = 0;
    f32 carrier_phase = 0;

    f64 rx_power = 1.0, cn0 = 1e5;
  };
  SignalSim(f64 sample_rate) : sample_rate(sample_rate) {}

  void generate(std::span<cp_f32> samples) {
    std::fill(samples.begin(), samples.end(), 0);
    size_t n = samples.size();
    f64 t_step = 1.0 / sample_rate;
    scratch.resize(n);
    for (SignalSettings& signal : signals) {
      f64 carrier_freq = signal.signal->carrier_freq();
      signal.signal->generate(scratch, sample_rate, signal.carrier_phase, 0, signal.code_phase); 
      signal.code_phase += (n * t_step) / signal.signal->code_period_s();
      signal.code_phase = std::fmod(signal.code_phase, 1.0);

      f64 amplitude = std::sqrt(signal.rx_power);
      f64 noise_pwr = signal.rx_power / signal.cn0;
      std::normal_distribution<f32> noise(0, std::sqrt(noise_pwr));
      for (size_t i = 0; i < n; i++) {
        f64 range = signal.pos.norm();
        f64 doppler_phase = 4.0f * std::numbers::pi_v<f32> * carrier_freq * range / 299792458.0f;
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
};
}
