#pragma once
#include <deque>

#include "weaver/types.h"

namespace weaver {
template<typename T>
struct MovingAverage {
  explicit MovingAverage(size_t n) : n(n), sum{} {}
  T add(T val) {
    if (vals.size() == n) {
      sum -= vals.back();
      vals.pop_back();
    }
    vals.emplace_front(val);
    sum += val;

    return cur_avg();
  }

  T cur_sum() const { return sum; }

  size_t cur_n() const { return vals.size(); }

  T cur_avg() const { return sum / T(vals.size()); }

private:
  size_t n;
  T sum;
  std::deque<T> vals;
};
template<typename T>
struct ExponentialSmoother {
  explicit ExponentialSmoother(T init, f64 decay = 0) : cur_val(init), decay(decay) {}
  T cur_val = 0;
  f64 decay;

  T update(T val) {
    cur_val += (1 - decay) * (val - cur_val);
    return cur_val;
  }
};
}  // namespace weaver
