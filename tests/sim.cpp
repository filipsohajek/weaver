#include "weaver/sim.h"

#include <catch2/catch_test_macros.hpp>
#include <fstream>

#include "weaver/gps.h"

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
      .vel = {0, 0, 7500},
  });
  std::ofstream out("out");

  size_t n = 65536;
  std::complex<float>* samples = reinterpret_cast<std::complex<float>*>(
      std::aligned_alloc(32, sizeof(std::complex<float>) * n));
  sim.generate({samples, n});
  out.write(reinterpret_cast<char*>(samples), n * sizeof(std::complex<float>));
  std::free(samples);
}
