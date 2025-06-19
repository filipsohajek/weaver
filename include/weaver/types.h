#pragma once
#include <complex>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <vector>
#include <Eigen/Core>

namespace weaver {
using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

using f32 = float;
using f64 = double;

using cp_i16 = std::complex<i16>;
using cp_i32 = std::complex<i32>;
using cp_f32 = std::complex<f32>;
using cp_f64 = std::complex<f64>;

template<typename T, size_t Size>
using array = std::array<T, Size>;

template<typename T>
using span = std::span<T>;

template<class T>
constexpr bool always_false = false;

template<typename Ti, typename Tf>
inline Ti round(Tf val) {
  return static_cast<Ti>(std::round(val));
}

template<typename Ti, typename Tf>
inline Ti floor(Tf val) {
  return static_cast<Ti>(std::floor(val));
}

template<typename To, typename Ti>
inline std::complex<To> cp_cast(std::complex<Ti> val) {
  return {static_cast<To>(val.real()), static_cast<To>(val.imag())};
}

template<typename To, typename Ti>
std::span<To> convert_span(std::span<Ti> in_span) {
  return {reinterpret_cast<To*>(in_span.data()), in_span.size_bytes() / sizeof(To)};
}

inline cp_i16 cp_i16_polar(i16 mag, f32 theta) {
  return cp_cast<i16>(std::polar(static_cast<f32>(mag), theta));
}

struct AlignedDeletor {
  void operator()(void* ptr) const { std::free(ptr); }
};

template<typename T>
using aligned_array = std::unique_ptr<T[], AlignedDeletor>;

template<typename T>
using aligned_vector = std::vector<T, Eigen::aligned_allocator<T>>;

template<typename T>
aligned_array<T> alloc_aligned_array(size_t alignment, size_t length) {
  void* buf = std::aligned_alloc(alignment, sizeof(T) * length);
  return aligned_array<T>(new (buf) T[length], AlignedDeletor());
}

template<typename T>
std::ostream& operator<<(std::ostream& out, std::optional<T> const& opt) {
  if (opt)
    out << opt.value();
  else
    out << "[No value]";
  return out;
}

}  // namespace weaver
