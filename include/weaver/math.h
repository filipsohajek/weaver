#pragma once
#include <cmath>
#include <numeric>
#include <ranges>
#include "weaver/types.h"

namespace weaver {
inline f64 ligf(f64 s, f64 x, size_t n_iters = 20) {
  f64 val = 0;
  for (size_t i = 0; i < n_iters; i++) {
    val += std::pow(x, i) / std::tgamma(s + i + 1);
  }
  return std::pow(x, s) * std::exp(-x) * val;
}

inline f64 chi2_cdf(f64 k, f64 x, size_t n_iters = 20) {
  return ligf(k/2, x/2, n_iters);
}

template<std::ranges::range R>
inline f64 mean(R&& r) {
  f64 sum = std::accumulate(std::ranges::begin(r), std::ranges::end(r), 0.0);
  return sum / std::ranges::size(r);
}

template<std::ranges::range R>
inline f64 sample_variance(R&& r) {
  auto r_squared = r | std::views::transform([](auto val) { return std::pow(val, 2); });
  return mean(r_squared) - std::pow(mean(r), 2);
}

}
