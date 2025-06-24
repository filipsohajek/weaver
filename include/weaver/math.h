#pragma once
#include <cmath>
#include <numeric>
#include <ranges>
#include <unsupported/Eigen/SpecialFunctions>
#include "weaver/types.h"

namespace weaver {
inline f64 chi2_cdf(f64 k, f64 x) {
  return Eigen::internal::scalar_igamma_op<f64>()(k/2, x/2);
}

template<std::ranges::range R>
inline auto mean(R&& r) {
  auto sum = std::accumulate(std::ranges::begin(r), std::ranges::end(r), std::ranges::range_value_t<R>());
  return sum / f32(std::ranges::size(r));
}

template<std::ranges::range R>
inline f64 sample_variance(R&& r) {
  auto sample_mean = mean(r);
  return mean(r | std::views::transform([&](auto val) { return std::pow(std::abs(val - sample_mean), 2); }));
}
}
