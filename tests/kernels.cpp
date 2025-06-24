#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <random>
#include <fstream>

#include "weaver/dsp/kernels_avx2.h"
#include "weaver/dsp/kernels_generic.h"

TEST_CASE("AVX2 & generic multicorrelation is correct", "[kernels]") {
  const size_t n = 65536;
  const size_t n_corrs = 3;

  std::mt19937_64 mt{};
  std::uniform_int_distribution<short> unif;
  std::uniform_int_distribution<unsigned char> unif_c;
  std::vector<unsigned char> chips(n);
  std::complex<short>* samples = reinterpret_cast<std::complex<short>*>(
      std::aligned_alloc(32, sizeof(std::complex<short>) * n));
  for (size_t i = 0; i < n; i++) {
    chips[i] = 0xaa;
    samples[i] = {unif(mt), unif(mt)};
  }

  float mix_init_phase = 0.0f;
  float mix_phase_step = 2 * std::numbers::pi_v<float> / 128.0f;
  double code_init_phase = 0.0;
  double code_phase_step = 0.325;
  double corr_offset = 0.3312312;
  std::complex<float> out_avx2[n_corrs] {};
  BENCHMARK("mcorr_avx2 (n=65536)") {
    std::fill(out_avx2, out_avx2 + n_corrs, 0);
    weaver::dsp::mmcorr_avx2<n_corrs, weaver::dsp::Modulation::BPSK>(n, samples, chips.data(), out_avx2, mix_init_phase, mix_phase_step, corr_offset, code_init_phase, code_phase_step);
  };
  std::complex<float> out_gen[n_corrs] {};
  BENCHMARK("mcorr_gen (n=65536)") {
   std::fill(out_gen, out_gen +n_corrs, 0);
    weaver::dsp::mmcorr_gen<n_corrs, weaver::dsp::Modulation::BPSK>(n, samples, chips.data(), out_gen, mix_init_phase, mix_phase_step, corr_offset, code_init_phase, code_phase_step);
    return out_gen;
 };

  std::complex<float> out_exact[n_corrs] {};
  for (size_t i = 0; i < n; i++) {
    double mix_phase = mix_init_phase + i * mix_phase_step;
    std::complex<float> in = weaver::cp_cast<float>(samples[i]) / 32767.0f;
    in *= weaver::cp_cast<float>(std::polar(1.0, mix_phase));

    for (size_t replica_i = 0; replica_i < n_corrs; replica_i++) {
      size_t chip_idx = code_init_phase + i * code_phase_step + replica_i * corr_offset;
      out_exact[replica_i] += (chip_idx % 2) ? in : -in;
    }
  }

  for (size_t i = 0; i < n_corrs; i++) {
    double avx2_error = std::pow(std::abs(out_avx2[i] - out_exact[i]), 2);
    double gen_error = std::pow(std::abs(out_gen[i] - out_exact[i]), 2);
    INFO ("avx2: " << avx2_error << ", per sample: " << avx2_error / n);
    REQUIRE(avx2_error / n <= 1e-6);
    INFO("generic: " << gen_error << ", per sample: " << gen_error / n);
    REQUIRE(gen_error / n <= 1e-6);
  }

  std::free(samples);
}

TEST_CASE("AVX2 & generic modulation is correct", "[kernels][mod]") {
  const size_t n = 65536;

  std::mt19937_64 mt{};
  std::uniform_int_distribution<short> unif;
  std::uniform_int_distribution<unsigned char> unif_c;
  std::vector<unsigned char> chips(n);
  std::fill(chips.begin(), chips.end(), 0xaa);
  std::complex<float>* samples_avx2 = reinterpret_cast<std::complex<float>*>(
      std::aligned_alloc(32, sizeof(std::complex<float>) * n));
  std::complex<float>* samples_generic = reinterpret_cast<std::complex<float>*>(
      std::aligned_alloc(32, sizeof(std::complex<float>) * n));

  float mix_init_phase = 0.0f;
  float mix_phase_step = 2 * std::numbers::pi_v<float> / 128.0f;
  double code_init_phase = 0.0;
  double code_phase_step = 0.325;
  BENCHMARK("modulate_avx2 (n=65536)") {
    weaver::dsp::modulate_avx2<std::complex<float>, weaver::dsp::Modulation::BPSK>(n, chips.data(), samples_avx2, mix_init_phase, mix_phase_step, code_init_phase, code_phase_step);
    return samples_avx2;
  };

  BENCHMARK("modulate_gen (n=65536)") {
    weaver::dsp::modulate_gen<std::complex<float>, weaver::dsp::Modulation::BPSK>(n, chips.data(), samples_generic, mix_init_phase, mix_phase_step, code_init_phase, code_phase_step);
    return samples_generic;
  };

  double err_avx2 = 0.0, err_gen = 0.0;

  for (size_t i = 0; i < n; i++) {
    double mix_phase = mix_init_phase + i * mix_phase_step;
    size_t chip_idx = code_init_phase + i * code_phase_step;
    std::complex<float> in = weaver::cp_cast<float>(std::polar(1.0, mix_phase));
    std::complex<float> out = (chip_idx % 2) ? in : -in; 
    err_avx2 += std::pow(std::abs(out - samples_avx2[i]), 2);
    err_gen += std::pow(std::abs(out - samples_generic[i]), 2);
  }
  err_avx2 = std::sqrt(err_avx2);
  err_gen = std::sqrt(err_gen);

  INFO("avx2: " << err_avx2 << ", per sample: " << err_avx2 / n);
  REQUIRE(err_avx2 / n <= 1e-5);
  INFO("gen: " << err_gen << ", per sample: " << err_gen / n);
  REQUIRE(err_gen / n <= 1e-5);

  std::free(samples_avx2);
  std::free(samples_generic);
}
