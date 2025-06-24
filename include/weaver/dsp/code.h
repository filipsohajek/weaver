#pragma once

#include <concepts>

#include "weaver/types.h"
namespace weaver::dsp {
enum class Modulation { BPSK };
template<class T>
concept IsCode = requires(T t) {
  { T::CHIP_RATE_HZ } -> std::convertible_to<f32>;
  { T::CHIP_COUNT } -> std::convertible_to<u32>;
  { T::PRN_COUNT } -> std::convertible_to<u16>;
  { T::gen_chips(u16(), std::span<u8>()) };

  { T::MODULATION } -> std::convertible_to<Modulation>;
};
}  // namespace weaver::dsp
