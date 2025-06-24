#include "weaver/sim.h"

#include <catch2/catch_test_macros.hpp>
#include <fstream>

#include "weaver/acq.h"
#include "weaver/channel.h"
#include "weaver/gps.h"

struct GPSL1Signal : public weaver::CodeSignal<weaver::GPSCACode> {
  using weaver::CodeSignal<weaver::GPSCACode>::CodeSignal;
  weaver::f64 carrier_freq() const override { return 1575.42e6; }
};

TEST_CASE("simulation works", "[sim]") {
  weaver::f64 sample_rate_hz = 4e6;
  weaver::SignalSim sim(sample_rate_hz);
  sim.signals.emplace_back(
      weaver::SignalSim::SignalSettings{.signal = std::make_shared<GPSL1Signal>(1),
                                        .pos = {0, 0, 382e3},
                                        .vel = {0, 0, 1000},
                                        .cn0 = 1e5});
  weaver::Channel channel(
      std::make_shared<GPSL1Signal>(3),
      weaver::Channel::Parameters{
      .sample_rate_hz = sample_rate_hz,
      .acq_params = weaver::AcqEngine::Parameters{
                                      .sample_rate_hz = sample_rate_hz, .n_coherent = 1, .n_noncoherent = 6, .doppler_step = 1},
      .acq_p_thresh = 0.05,
    });

  size_t n = 4000;
  weaver::aligned_vector<weaver::cp_f32> samples(n);
  weaver::aligned_vector<weaver::cp_i16> samples_i16(n);
  //std::ofstream out("out");
  std::ifstream in("/home/filip/dev/sw/weaver/build/samples.i16");
  while (!in.eof()) {
    //sim.generate(samples);
    //weaver::dsp::cvt_cpf32_cpi16(n, samples.data(), samples_i16.data());
    //out.write(reinterpret_cast<char*>(samples_i16.data()), n * sizeof(weaver::cp_i16));
    in.read(reinterpret_cast<char*>(samples_i16.data()), n * sizeof(weaver::cp_i16));
    channel.process_samples(samples_i16);
  }
}
