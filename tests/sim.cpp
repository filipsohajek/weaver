#include "weaver/sim.h"

#include <catch2/catch_test_macros.hpp>
#include <fstream>

#include "weaver/acq.h"
#include "weaver/channel.h"
#include "weaver/gps.h"

struct GPSL1Signal : public weaver::CodeSignal<weaver::GPSCACode, weaver::NELPCodeDiscriminator<0.5>> {
  using weaver::CodeSignal<weaver::GPSCACode, weaver::NELPCodeDiscriminator<0.5>>::CodeSignal;
  weaver::f64 carrier_freq() const override { return 1575.42e6; }
};

TEST_CASE("simulation works", "[sim]") {
  weaver::SignalSim sim(4e6);
  sim.signals.emplace_back(
      weaver::SignalSim::SignalSettings{.signal = std::make_shared<GPSL1Signal>(0),
                                        .pos = {0, 0, 382e3},
                                        .vel = {500, 0, 0},
                                        .code_phase = 0.3,
                                        .cn0 = 1e3});
  weaver::Channel channel(
      std::make_shared<GPSL1Signal>(0),
      weaver::Channel::Parameters{
      .sample_rate_hz = 4e6,
      .acq_params = weaver::AcqEngine::Parameters{
                                      .sample_rate_hz = 4e6, .n_coherent = 2, .n_noncoherent = 2}});

  size_t n = 65536;
  weaver::aligned_vector<weaver::cp_f32> samples(n);
  weaver::aligned_vector<weaver::cp_i16> samples_i16(n);
  for (size_t i = 0; i < 128; i++) {
    sim.generate(samples);
    weaver::dsp::cvt_cpf32_cpi16(n, samples.data(), samples_i16.data());
    channel.process_samples(samples_i16);
  }
}
