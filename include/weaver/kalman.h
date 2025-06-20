#pragma once
#include <eigen3/Eigen/Dense>
#include "weaver/types.h"

namespace weaver { 
template<size_t StateDim,
         size_t ObsDim,
         typename ValueType = f64,
         class StateType = Eigen::Vector<ValueType, StateDim>>
struct KalmanFilter {
  void update(Eigen::Vector<ValueType, ObsDim> meas_vec, bool error_meas = true) {
    auto pred_state = state_step_matrix * state;
    auto pred_covariances =
        state_step_matrix * state_cov * state_step_matrix.transpose() + process_noise_cov;

    auto kalman_gain = pred_covariances * meas_matrix.transpose() *
                       (meas_matrix * pred_covariances * meas_matrix.transpose() +
                        meas_noise_cov)
                           .inverse();
    if (error_meas) {
      state = pred_state + kalman_gain * meas_vec;
    } else {
      state = pred_state + kalman_gain * (meas_vec - meas_matrix * pred_state);
    }
    state_cov = (Eigen::Matrix<ValueType, StateDim, StateDim>::Identity() -
                  kalman_gain * meas_matrix) *
                 pred_covariances;
  }

  StateType state;
  Eigen::Matrix<ValueType, StateDim, StateDim> state_cov;
  Eigen::Matrix<ValueType, StateDim, StateDim> state_step_matrix;
  Eigen::Matrix<ValueType, ObsDim, StateDim> meas_matrix;

  Eigen::Matrix<ValueType, StateDim, StateDim> process_noise_cov;
  Eigen::Matrix<ValueType, ObsDim, ObsDim> meas_noise_cov;
};
}
