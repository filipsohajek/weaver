#pragma once
#include "weaver/acq.h"
#include "weaver/types.h"

namespace weaver {
struct LoopFilter {
  struct Output {
    std::optional<f64> code_freq;
    std::optional<f64> carrier_freq;
    std::optional<f64> code_phase_adj;
    std::optional<f64> carr_phase_adj;
  };
  virtual ~LoopFilter() {};

  virtual void init(const Signal*, const AcqEngine::Result&) = 0;
  [[nodiscard]] virtual Output update(f64 code_disc_out, f64 carr_disc_out) = 0;
  virtual void update_disc_statistics(f64 code_disc_var, f64 carr_disc_var) = 0;
  virtual void set_int_time(f64 int_time_s) = 0;
};
}
