#include <catch2/catch_test_macros.hpp>
#include <fstream>

#include "weaver/sim.h"
#include "weaver/gps.h"
#include "weaver/acq.h"

struct NoCodeDisc {
  static constexpr const size_t N_REPLICAS = 3;
  static constexpr const float CORR_OFFSET = 0.5;

  static weaver::f32 disc_code(std::span<weaver::cp_f32>) { return 0.0f; }

  static weaver::f64 disc_error_var(weaver::f64) { return 0.0; }
};

struct GPSL1Signal : public weaver::CodeSignal<weaver::GPSCACode, NoCodeDisc> {
  using weaver::CodeSignal<weaver::GPSCACode, NoCodeDisc>::CodeSignal;
  weaver::f64 carrier_freq() const override { return 1575.42e6; }
};

TEST_CASE("simulation works", "[sim]") {
  weaver::SignalSim sim(4e6);
  sim.signals.emplace_back(weaver::SignalSim::SignalSettings{
      .signal = std::make_unique<GPSL1Signal>(0),
      .pos = {0, 0, 382e3},
      .vel = {0, 0, 500},
      .code_phase = 0.3,
      .cn0 = 1e0
  });
  weaver::AcqEngine acq(std::make_unique<GPSL1Signal>(0));
  acq.reset(weaver::AcqEngine::Parameters{.sample_rate_hz = 4e6, .n_coherent = 2, .n_noncoherent = 2});

  std::ofstream out("out");

  size_t n = 65536;
  weaver::aligned_vector<weaver::cp_f32> samples(n);
  weaver::aligned_vector<weaver::cp_i16> samples_i16(n);
  sim.generate(samples);
  weaver::dsp::cvt_cpf32_cpi16(n, samples.data(), samples_i16.data());
  auto acq_res_samples = acq.process(samples_i16);
  std::cout << std::format("number of residual samples: {}\n", acq_res_samples.size());
  std::cout << "Acquisition candidates:\n";
  for (const weaver::AcqEngine::Candidate& cand : acq.acq_candidates()) {
    std::cout << std::format("code_offset={}, doppler_freq={}, n_noncoh={}, val={}, p={}\n", cand.code_offset, cand.doppler_freq, cand.n_noncoh, cand.val, cand.p_val());
  }
  std::cout << "\n";
  out.write(reinterpret_cast<char*>(samples.data()), n * sizeof(std::complex<float>));
}
