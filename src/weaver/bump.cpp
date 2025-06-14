#include "weaver/dsp/kernels_avx2.h"
#include <time.h>
#include <vector>
#include <random>
#include <format>
#include <chrono>
#include <algorithm>
#include <fstream>


int main() {
  /*const size_t n = 4096;
  std::mt19937_64 mt {std::random_device()()};
  std::vector<std::complex<short>> data(n);
  std::vector<std::complex<short>> chips(n);
  std::uniform_int_distribution<short> unif;
  std::fill(data.begin(), data.end(), std::complex<short> {32767, 0});
  for (size_t i = 0; i < n; i++) {
    short val = 2*((i % 32) >= 16) - 1;
    chips[i] = {val, static_cast<short>(-val)};
  }

  __m256 step_vec, cur_vec;
  float phase_step = 2*std::numbers::pi_v<float> / 128.0f;
  std::complex<float> step = std::polar(1.0f, 4.0f * phase_step);
  step_vec = std::bit_cast<__m256>(_mm256_set1_epi64x(std::bit_cast<uint64_t>(step)));
  std::complex<float> init {1.0f, 0.0f};
  for (size_t i = 0; i < sizeof(__m256i) / sizeof(std::complex<float>); i++) {
    *(reinterpret_cast<std::complex<float>*>(&cur_vec) + i) = init;
    init *= std::polar(1.0f, phase_step);
  }

  double code_step = 0.325;
  __m256d code_step_vec = _mm256_set1_pd(4 * code_step);
  __m256d code_cur_vec = _mm256_set_pd(3*code_step, 2*code_step, code_step, 0);

  for (size_t i = 0; i < n; i += sizeof(__m256i) / sizeof(std::complex<short>)) {
    __m256i in = _mm256_loadu_si256(reinterpret_cast<__m256i*>(data.data() + i));

    double code_base = i * code_step;
    __m256d code_base_vec = _mm256_set1_pd(static_cast<int>(code_base));
    code_cur_vec = _mm256_add_pd(code_cur_vec, code_step_vec);
    __m256d code_phase_low = _mm256_sub_pd(code_cur_vec, code_base_vec);
    code_cur_vec = _mm256_add_pd(code_cur_vec, code_step_vec);
    __m256d code_phase_high = _mm256_sub_pd(code_cur_vec, code_base_vec);

    __m256i chip_vec = _mm256_loadu_si256(reinterpret_cast<__m256i*>(chips.data() + static_cast<int>(code_base)));
    __m128i code_phase_lo_int = _mm256_cvtpd_epi32(code_phase_low);
    __m128i code_phase_hi_int = _mm256_cvtpd_epi32(code_phase_high);
    __m256i code_phase = _mm256_setr_m128i(code_phase_lo_int, code_phase_hi_int);

    in = _mm256_sign_epi16(in, _mm256_permutevar8x32_epi32(chip_vec, code_phase));

    __m128i in_low = _mm256_extractf128_si256(in, 0);
    __m256 in_low_float = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(in_low));
    __m256 out_low_float = weaver::dsp::avx2::mm256_mul_cpps(in_low_float, cur_vec);
    __m256i out_low = _mm256_cvtps_epi32(out_low_float); 

    __m256 cur_tmp = weaver::dsp::avx2::mm256_mul_cpps(cur_vec, step_vec);
    __m128i in_high = _mm256_extractf128_si256(in, 1);
    __m256 in_high_float = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(in_high));
    __m256 out_high_float = weaver::dsp::avx2::mm256_mul_cpps(in_high_float, cur_tmp);
    __m256i out_high = _mm256_cvtps_epi32(out_high_float); 

    cur_vec = weaver::dsp::avx2::mm256_mul_cpps(cur_tmp, step_vec);

    __m256i out = _mm256_packs_epi32(out_low, out_high);
    out = _mm256_permute4x64_epi64(out, 0xd8);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(data.data() + i), out);
  }

  std::ofstream out("out");
  for (auto& val : data) {
    out.write(reinterpret_cast<const char*>(&val), sizeof(std::complex<short>));
  }
*/

  std::mt19937_64 mt {std::random_device()()};
  std::uniform_int_distribution<short> unif;
  std::uniform_int_distribution<unsigned char> unif_c;
  std::vector<std::complex<short>> chips(65536);
  std::complex<short>* samples_ptr = reinterpret_cast<std::complex<short>*>(std::aligned_alloc(32, 4*65536));
  std::span<std::complex<short>> samples {samples_ptr, 65536};
  for (std::complex<short>& sample : samples)
    sample = {unif(mt), unif(mt)};
  for (auto& chip : chips)
    chip = {unif(mt), unif(mt)};

  for (size_t n = 8; n <= 65536; n *= 2) {
  const size_t N_REPS = 5000;
  const size_t N_REPS_WARMUP = 500;
  const size_t N_CORRS = 3;
  const float CORR_OFFSET = 0.5;
  const float MIX_PHASE_STEP = 0.01;
  const float CODE_PHASE_STEP = 0.3;
  std::vector<std::complex<float>> out(N_CORRS);

  std::vector<double> rep_times;
  std::complex<float> o;

  for (size_t rep = 0; rep < N_REPS; rep++) {
    std::fill(out.begin(), out.end(), 0);

    auto start = std::chrono::high_resolution_clock::now();
    weaver::dsp::avx2::mmcorr_avx2<N_CORRS, weaver::dsp::Modulation::BPSK>(n, samples.data(), chips.data(), out.data(), 0, MIX_PHASE_STEP, CORR_OFFSET, 0, CODE_PHASE_STEP);
    auto stop = std::chrono::high_resolution_clock::now();

    double rep_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
    if (rep > N_REPS_WARMUP)
      rep_times.push_back(rep_time);
    o += out[0] + out[1] + out[2];
//    std::cout << std::format("total={}, per_sample={}, out[0]={}+{}i, out[1]={}+{}i, out[2]={}+{}i\n", rep_time, rep_time/n, out[0].real(), out[0].imag(), out[1].real(), out[1].imag(), out[2].real(), out[2].imag());
  }
    std::cout << "n=" << n << "\n";
  std::cout << std::format("total: min={}, max={}, avg={}\n", *std::ranges::min_element(rep_times), *std::ranges::max_element(rep_times), std::accumulate(rep_times.begin(), rep_times.end(), 0) / rep_times.size());
  std::cout << std::format("per_sample: min={}, max={}, avg={}\n", *std::ranges::min_element(rep_times) / n, *std::ranges::max_element(rep_times) / n, std::accumulate(rep_times.begin(), rep_times.end(), 0.0) / (rep_times.size() * n));
  std::cout << o.real() << o.imag() << "\n";
    std::cout << "\n";
  }
}
