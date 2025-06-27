#pragma once

#include "weaver/types.h"
namespace weaver {
enum class GNSSSystem : u8{
  GPS
};

enum class GNSSBand : u8 {
  L1 = 1
};

struct SignalID {
  GNSSSystem sys;
  GNSSBand band;
  u16 prn;

  bool operator==(const SignalID&) const = default;
};
};

namespace std {
  template<>
  struct hash<weaver::SignalID> {
    size_t operator()(const weaver::SignalID& sid) const {
      return std::hash<weaver::u8>()(weaver::u8(sid.sys)) ^ std::hash<weaver::u8>()(weaver::u8(sid.band)) ^ std::hash<weaver::u16>()(weaver::u16(sid.prn));
    }
  };
}
