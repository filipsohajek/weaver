#include "weaver/sim.h"

#include <catch2/catch_test_macros.hpp>
#include <fstream>

#include "weaver/acq.h"
#include "weaver/channel.h"
#include "weaver/gps.h"
#include "weaver/kalman_loop_filter.h"

struct GPSL1Signal : public weaver::CodeSignal<weaver::GPSCACode> {
  using weaver::CodeSignal<weaver::GPSCACode>::CodeSignal;
  weaver::f64 carrier_freq() const override { return 1575.42e6; }
};

TEST_CASE("simulation works", "[sim]") {
  weaver::f64 sample_rate_hz = 4e6;

  std::vector<double> corr_offsets;
  corr_offsets.push_back(0.5);
  auto channel_params = weaver::Channel::Parameters{
      .acq_params = weaver::AcqEngine::Parameters{.sample_rate_hz = sample_rate_hz,
                                                  .n_coherent = 1,
                                                  .n_noncoherent = 1,
                                                  .doppler_step = 1},
      .sample_rate_hz = sample_rate_hz,
      .corr_offsets = std::move(corr_offsets),
      .acq_p_thresh = 0.05,
      .filter = std::make_unique<weaver::KalmanLoopFilter>(weaver::KalmanLoopFilter::Parameters{}),
      .code_disc = weaver::Channel::CodeDiscriminator::EMLEnvelope,
      .carrier_disc = weaver::Channel::CarrierDiscriminator::ATan};
  weaver::Channel channel(std::make_shared<GPSL1Signal>(1), std::move(channel_params));

  size_t n = 4000;
  weaver::aligned_vector<weaver::cp_f32> samples(n);
  weaver::aligned_vector<weaver::cp_i16> samples_i16(n);
  // std::ofstream out("out");
  std::ifstream in("/home/filip/dev/sw/weaver/2013_04_04_GNSS_SIGNAL_at_CTTC_SPAIN/2013_04_04_GNSS_SIGNAL_at_CTTC_SPAIN.dat");
  while (!in.eof()) {
    // sim.generate(samples);
    // weaver::dsp::cvt_cpf32_cpi16(n, samples.data(), samples_i16.data());
    // out.write(reinterpret_cast<char*>(samples_i16.data()), n * sizeof(weaver::cp_i16));
    in.read(reinterpret_cast<char*>(samples_i16.data()), n * sizeof(weaver::cp_i16));
    size_t samples_read = in.gcount() / sizeof(weaver::cp_i16);
    channel.process_samples({samples_i16.data(), samples_read});
  }
}
