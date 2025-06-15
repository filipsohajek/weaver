#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <random>

#include "weaver/dsp/kernels_avx2.h"
#include "weaver/dsp/kernels_generic.h"

TEST_CASE("AVX2 & generic multicorrelation is correct", "[kernels]") {
  const size_t n = 2048;
  const size_t n_corrs = 3;

  std::mt19937_64 mt{};
  std::uniform_int_distribution<short> unif;
  std::uniform_int_distribution<unsigned char> unif_c;
  std::vector<unsigned char> chips(n);
  std::complex<short>* samples = reinterpret_cast<std::complex<short>*>(
      std::aligned_alloc(32, sizeof(std::complex<short>) * n));
  for (size_t i = 0; i < n; i++) {
    chips[i] = 0x99;
    samples[i] = {unif(mt), unif(mt)};
  }

  float mix_init_phase = 0.0f;
  float mix_phase_step = 2 * std::numbers::pi_v<float> / 128.0f;
  double code_init_phase = 0.0;
  double code_phase_step = 0.325;
  double corr_offset = 0.3;
  std::complex<float> out_avx2[n_corrs] {};
  BENCHMARK("mcorr_avx2") {
    std::fill(out_avx2, out_avx2 + n_corrs, 0);
    weaver::dsp::mmcorr_avx2<n_corrs, weaver::dsp::Modulation::BPSK>(n, samples, chips.data(), out_avx2, mix_init_phase, mix_phase_step, corr_offset, code_init_phase, code_phase_step);
  };
  std::complex<float> out_gen[n_corrs] {};
  BENCHMARK("mcorr_gen") {
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
      std::complex<float> out { in.real() * ((chip_idx%2) ? 1.0f : -1.0f), in.imag() * ((chip_idx%2) ? -1.0f : 1.0f) };
      out_exact[replica_i] += out;
    }
  }

  for (size_t i = 0; i < n_corrs; i++) {
    double avx2_error = std::pow(std::abs(out_avx2[i] - out_exact[i]), 2);
    double gen_error = std::pow(std::abs(out_gen[i] - out_exact[i]), 2);
    std::cout << "avx2: " << avx2_error << ", per sample: " << avx2_error / n << "\n";
    std::cout << "generic: " << gen_error << ", per sample: " << gen_error / n << "\n";
  }

  std::free(samples);
}
