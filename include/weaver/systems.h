#pragma once

#include "weaver/types.h"
namespace weaver {
enum class GNSSSystem {
  GPS
};

enum class GNSSBand {
  L1 = 1
};

struct SignalID {
  GNSSSystem sys;
  GNSSBand band;
  u16 prn;
};
};
