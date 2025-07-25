#pragma once
#include "weaver/acq.h"
#include "weaver/corr.h"
#include "weaver/loop_filter.h"
#include "weaver/pvt.h"
#include "weaver/signal.h"
#include "weaver/types.h"
#include "weaver/util.h"

namespace weaver {
class Channel {
public:
  enum class CodeDiscriminator { EMLEnvelope };

  enum class CarrierDiscriminator { ATan };

  struct Parameters {
    AcqEngine::Parameters acq_params;
    f64 sample_rate_hz;
    std::vector<f64> corr_offsets;
    f64 acq_p_thresh = 0.05;
    std::unique_ptr<std::ostream> trace_stream = nullptr;

    std::unique_ptr<LoopFilter> filter;
    size_t cn0_est_prompts = 20;
    f64 cn0_low_clamp = 1e2;
    f64 cn0_decay = 0.8;
    f64 cn0_init = 1e3;

    f64 phase_lock_decay = 0.999;
    size_t phase_lock_est_prompts = 50;

    f64 pullin_time = 2.0;
    f64 cn0_lock_thresh_db = 30;
    f64 phase_lock_thresh = 0.85;

    size_t data_est_symbols = 300;
    f64 data_est_p_thresh = 0.05;
    
    CodeDiscriminator code_disc;
    CarrierDiscriminator carrier_disc;
  };

  enum class State {
    FAILED = 0,
    ACQUISITION = 1,
    TRACK_INIT = 2,
    TRACK_LOCKED = 3,
    DATA_LOCKED = 4
  };

  explicit Channel(std::shared_ptr<Signal> signal, Parameters params)
      : elapsed_time(0),
        cur_cn0(params.cn0_init, params.cn0_decay),
        phase_lock_ind(0, params.phase_lock_decay),
        trace_file("out_trace"),
        corr(signal, params.sample_rate_hz, params.corr_offsets),
        signal(signal),
        data_decoder(signal->data_decoder()),
        acq(signal, params.sample_rate_hz, params.acq_params),
        params(std::move(params)) {
    state = State::ACQUISITION;
  }

  void process_samples(span<cp_i16> samples) {
    while (!samples.empty()) {
      //std::cout << std::format("process_samples: state={}, samples.size()={}\n", i32(state), samples.size());
      size_t init_sample_count = samples.size();
      switch (state) {
        case State::FAILED:
          return;
        case State::ACQUISITION: {
          samples = acq.process(samples);
          if (!acq.finished())
            break;
          AcqEngine::Result result = acq.result().value();
          std::cout << std::format("acquisition({}): code_offset={}, doppler_freq={}, p={}\n", signal->id().prn,
                                   result.code_offset, result.doppler_freq, result.p);
          if (result.p >= params.acq_p_thresh) {
            std::cout << "acquisition failed, failing channel\n";
            state = State::FAILED;
            break;
          }
          setup_tracking(result);
          state = State::TRACK_INIT;
          break;
        }
        case State::TRACK_INIT:
          if (elapsed_time >= params.pullin_time) {
            state = is_locked() ? State::TRACK_LOCKED : State::FAILED;
            break;
          }
          // fallthrough
        case State::TRACK_LOCKED:
        case State::DATA_LOCKED:
          samples = process_track(samples);
          break;
      }
      size_t end_sample_count = samples.size();
      elapsed_time += (init_sample_count - end_sample_count) / params.sample_rate_hz;
    }
  }
  
  std::optional<TimeOfWeek> tow() const {
    if ((state < State::DATA_LOCKED) || !data_decoder)
      return std::nullopt;
    
    auto data_tow = data_decoder->tow();
    if (!data_tow.has_value())
      return std::nullopt;
    TimeOfWeek res_tow = data_tow.value();
    
    res_tow += data_code_phase_acc * signal->code_period_s() + corr.elapsed_time_s();
    return res_tow;
  }

  std::queue<NavMessage>* message_queue() {
    if ((state < State::DATA_LOCKED) || !data_decoder)
      return nullptr;

    return &data_decoder->message_queue();
  }

  SignalID sid() const {
    return signal->id();
  }
private:
  span<cp_i16> process_track(span<cp_i16> samples) {
    span<cp_i16> res_samples = corr.process_samples(samples);
    if (!corr.has_result())
      return res_samples;

    span<const cp_f32> corr_res = corr.result();
    cp_f32 prompt = corr_res[corr_res.size() / 2];
    f64 carrier_disc_out = disc_carrier(prompt);
    f64 code_disc_out = -disc_code(corr_res);

    f64 cn0_db = 10 * std::log10(cn0());
    f64 lock_ind = phase_lock_ind.cur_val; 
    if (params.trace_stream.get()) {
      params.trace_stream->write(reinterpret_cast<char*>(&prompt), sizeof(cp_f32));
      params.trace_stream->write(reinterpret_cast<char*>(&data_prompt_acc), sizeof(cp_f32));
      params.trace_stream->write(reinterpret_cast<char*>(&cn0_db), sizeof(f64));
      params.trace_stream->write(reinterpret_cast<char*>(&lock_ind), sizeof(f64));
      params.trace_stream->write(reinterpret_cast<char*>(&code_disc_out), sizeof(f64));
      params.trace_stream->write(reinterpret_cast<char*>(&carrier_disc_out), sizeof(f64));
      params.trace_stream->write(reinterpret_cast<char*>(&corr.code_phase), sizeof(f64));
      params.trace_stream->write(reinterpret_cast<char*>(&corr.carr_phase), sizeof(f64));
      params.trace_stream->write(reinterpret_cast<char*>(&corr.carr_freq), sizeof(f32));
      params.trace_stream->write(reinterpret_cast<char*>(&corr.code_freq), sizeof(f64));
    }

//    std::cout << std::format(
//        "process_track({}): prompt.re={}, prompt.im={}, code_disc={}, carr_disc={}, cn0={}, lock_ind={}\n",
//      signal->id().prn,
//        prompt.real(), prompt.imag(), code_disc_out, carrier_disc_out, cn0_db, lock_ind);

    update_cn0(prompt);
    update_data(prompt);
    update_data_transitions(prompt);
    update_filter(code_disc_out, carrier_disc_out);

    last_prompt = prompt;
    corr.reset(1.0, 1.0);

    return res_samples;
  }

  bool is_locked() const {
    return (10*std::log10(cn0()) >= params.cn0_lock_thresh_db) && (phase_lock_ind.cur_val >= params.phase_lock_thresh);
  }

  void eval_lock() {
    if (!is_locked() && (state != State::TRACK_INIT))
      state = State::FAILED;
  }

  void update_data_transitions(cp_f32 prompt) {
    size_t n_trans_bins = data_decoder->symbol_period_s() / signal->code_period_s();
    if (data_decoder && (state >= State::TRACK_LOCKED)) {       
      data_trans_bins.resize(n_trans_bins); 
      bool is_trans = std::signbit(prompt.real() * last_prompt.real());
      if (is_trans) {
        data_trans_bins[data_prompt_offset] += 1;
      }
      data_est_n_prompts += 1;
      
      if (data_est_n_prompts == n_trans_bins * params.data_est_symbols) {
        size_t n_est_prompts = n_trans_bins * params.data_est_symbols;
        auto max_bin_it = std::ranges::max_element(data_trans_bins);
        size_t data_trans_bin = std::distance(data_trans_bins.begin(), max_bin_it);

        f64 h0_expected_counts = f64(n_est_prompts) / f64(params.data_est_symbols);
        f64 counts_norm = (2 * (*max_bin_it) - h0_expected_counts) / std::sqrt(h0_expected_counts);
        f64 p = 1 - std::pow(norm_cdf(counts_norm), n_trans_bins); 

        std::cout << std::format("update_data_transitions({}): data transition estimation: bin_counts=[ ", signal->id().prn);
        for (u32 bin_count : data_trans_bins) {
          std::cout << bin_count << " "; 
        }
        std::cout << std::format("], est_trans_bin={}, counts_norm={}, h0_expected_counts={}, p={}\n", data_trans_bin, counts_norm, h0_expected_counts, p);

        if (p < params.data_est_p_thresh) {
          if (data_trans_offset != data_trans_bin) {
            data_prompt_acc = 0;
            data_code_phase_acc = 0;
          }
          data_trans_offset = data_trans_bin;
          state = State::DATA_LOCKED;
        } else {
          state = State::FAILED;
        }

        std::ranges::fill(data_trans_bins, 0);
        data_est_n_prompts = 0;
      }
    }

    data_prompt_offset += 1;
    if (data_prompt_offset == n_trans_bins)
      data_prompt_offset = 0;
  }

  void update_filter(f64 code_disc_out, f64 carrier_disc_out) {
    params.filter->update_disc_statistics(code_disc_var(), carrier_disc_var());
    LoopFilter::Output filter_out = params.filter->update(code_disc_out, carrier_disc_out);

    corr.update_params(filter_out.carrier_freq.value_or(corr.carr_freq),
                       filter_out.code_freq.value_or(corr.code_freq),
                       filter_out.carr_phase_adj.value_or(0),
                       filter_out.code_phase_adj.value_or(0));
  }

  void update_data(cp_f32 prompt) {
    if ((state != State::DATA_LOCKED) || !data_decoder)
      return;

    if (data_prompt_offset == data_trans_offset) {
      data_decoder->process_symbol(data_prompt_acc);
      data_prompt_acc = 0;
      data_code_phase_acc = 0;
    }

    data_code_phase_acc += corr.code_freq * corr.int_time_s();
    data_prompt_acc += prompt;
  }

  void update_cn0(cp_f32 prompt) {
    cn0_prompts.push_back(prompt);
    if (cn0_prompts.size() > params.cn0_est_prompts)
      cn0_prompts.pop_front();
    cn0_n_rem -= 1;
    if (cn0_n_rem == 0) {
      cn0_n_rem = params.cn0_est_prompts;
      f64 m2 = 0;
      f64 m4 = 0;
      for (cp_f32 prompt : cn0_prompts) {
        f64 prompt_mag = std::abs(prompt);
        m2 += std::pow(prompt_mag, 2);
        m4 += std::pow(prompt_mag, 4);
      }
      m2 /= cn0_prompts.size();
      m4 /= cn0_prompts.size();


      f64 ad = 2 * std::pow(m2, 2) - m4;
      f64 pd = std::sqrt(ad);
      f64 pn = m2 - pd;
      f64 new_cn0 = (pd / pn) / corr.int_time_s();
      if ((ad < 0) || (pn <= 0)) {
        cur_cn0.update(params.cn0_low_clamp);
        return;
      }
      cur_cn0.update(new_cn0);
      eval_lock();
    }
  
    phase_lock_prompts.push_back(prompt);
    if (phase_lock_prompts.size() > params.phase_lock_est_prompts)
      phase_lock_prompts.pop_front();
    phase_lock_n_rem -= 1;
    if (phase_lock_n_rem == 0) {
      phase_lock_n_rem = params.phase_lock_est_prompts;
      cp_f32 prompt_sum = std::accumulate(phase_lock_prompts.begin(), phase_lock_prompts.end(), cp_f32());
      f64 i_sq = std::pow(prompt_sum.real(), 2);
      f64 q_sq = std::pow(prompt_sum.imag(), 2);
      f64 lock_ind = (i_sq - q_sq) / (i_sq + q_sq);

      if (phase_lock_ind.cur_val == 0)
        phase_lock_ind.cur_val = lock_ind;
      else
        phase_lock_ind.update(lock_ind);
      eval_lock();
    }
  }

  f64 cn0() const { return cur_cn0.cur_val; }

  void set_int_time(f64 int_time_codeper) {
    corr.reset(int_time_codeper);
    f64 int_time_s = corr.int_time_s();
    params.filter->set_int_time(int_time_s);
  }

  void setup_tracking(AcqEngine::Result res) {
    params.filter->init(signal.get(), res);
    corr.update_params(res.doppler_freq, 1 / signal->code_period_s(), 0, res.code_offset);
    cn0_n_rem = params.cn0_est_prompts;
    phase_lock_n_rem = params.phase_lock_est_prompts;
    cur_cn0.update(params.cn0_init);
    set_int_time(1.0);
  }

  f64 carrier_disc_var() const {
    switch (params.carrier_disc) {
      case CarrierDiscriminator::ATan:
        return (1 / (2 * cn0() * corr.int_time_s())) * (1 + 1 / (2 * corr.int_time_s() * cn0())) /
               std::pow(2 * std::numbers::pi_v<f32>, 2);
      default:
        return nan("");
    }
  }

  f64 disc_carrier(cp_f32 prompt) const {
    switch (params.carrier_disc) {
      case CarrierDiscriminator::ATan:
        return std::atan(prompt.imag() / prompt.real()) / (2 * std::numbers::pi_v<f32>);
      default:
        return nan("");
    }
  }

  f64 code_disc_var() const {
    switch (params.code_disc) {
      case CodeDiscriminator::EMLEnvelope: {
        f64 d = 2 * corr.offsets()[corr.offsets().size() - 1];
        return ((d / (4 * cn0() * corr.int_time_s())) *
                (1 + 2 / ((2 - d) * corr.int_time_s() * cn0())));
      }
      default:
        return nan("");
    }
  }

  f64 disc_code(span<const cp_f32> corr_res) const {
    switch (params.code_disc) {
      case CodeDiscriminator::EMLEnvelope: {
        f64 e = std::abs(corr_res[corr.offsets().size() - 1]);
        f64 l = std::abs(corr_res[corr.offsets().size() + 1]);
        return (0.5 * (e - l) / (e + l)) / signal->chip_count();
      }
      default:
        return nan("");
    }
  }

  f64 elapsed_time;

  std::deque<cp_f32> cn0_prompts;
  size_t cn0_n_rem;
  ExponentialSmoother<f64> cur_cn0;

  ExponentialSmoother<f64> phase_lock_ind;
  size_t phase_lock_n_rem;
  std::deque<cp_f32> phase_lock_prompts;

  std::ofstream trace_file;
  State state;
  Correlator corr;
  std::shared_ptr<Signal> signal;
  
  size_t data_prompt_offset = 0;
  size_t data_trans_offset;
  std::unique_ptr<NavDataDecoder> data_decoder;
  std::vector<u32> data_trans_bins;
  size_t data_est_n_prompts = 0;
  cp_f32 data_prompt_acc;
  f64 data_code_phase_acc;
  cp_f32 last_prompt;

  AcqEngine acq;
  Parameters params;
};
}  // namespace weaver
