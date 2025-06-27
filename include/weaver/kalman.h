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

    auto kalman_gain =
        pred_covariances * meas_matrix.transpose() *
        (meas_matrix * pred_covariances * meas_matrix.transpose() + meas_noise_cov).inverse();
    if (error_meas) {
      state = pred_state + kalman_gain * meas_vec;
    } else {
      state = pred_state + kalman_gain * (meas_vec - meas_matrix * pred_state);
    }
    state_cov =
        (Eigen::Matrix<ValueType, StateDim, StateDim>::Identity() - kalman_gain * meas_matrix) *
        pred_covariances;
  }

  StateType state;
  Eigen::Matrix<ValueType, StateDim, StateDim> state_cov;
  Eigen::Matrix<ValueType, StateDim, StateDim> state_step_matrix;
  Eigen::Matrix<ValueType, ObsDim, StateDim> meas_matrix;

  Eigen::Matrix<ValueType, StateDim, StateDim> process_noise_cov;
  Eigen::Matrix<ValueType, ObsDim, ObsDim> meas_noise_cov;
};

template<ssize_t StateDim,
         ssize_t ObsDim,
         typename ValueType = f64,
         class StateType = Eigen::Vector<ValueType, StateDim>>
struct ExtendedKalmanFilter {
  ExtendedKalmanFilter(size_t state_dim, size_t obs_dim) {
    state.resize(state_dim);
    state.setZero();
    state_cov.resize(state_dim, state_dim);
    state_cov.setIdentity();
    state_cov *= 1e12;
    process_noise_cov.resize(state_dim, state_dim);
    process_noise_cov.setIdentity();
    process_noise_cov *= 0;
    meas_noise_cov.resize(obs_dim, obs_dim);
    meas_noise_cov.setIdentity();
    meas_noise_cov *= 0;
  }

  size_t add_obs_dim()
    requires(ObsDim == Eigen::Dynamic)
  {
    size_t old_obs_dim = meas_noise_cov.cols();
    meas_noise_cov.conservativeResize(old_obs_dim + 1, old_obs_dim + 1);
    meas_noise_cov.row(old_obs_dim).setZero();
    meas_noise_cov.col(old_obs_dim).setZero();
    meas_noise_cov(old_obs_dim, old_obs_dim) = 0.1;
    return old_obs_dim;
  }

  void remove_obs_dim(std::vector<i32> keep_dims)
    requires(ObsDim == Eigen::Dynamic)
  {
    Eigen::VectorXi keep_dims_vec(keep_dims_vec.data(), keep_dims_vec.size());
    meas_noise_cov = meas_noise_cov(keep_dims_vec, keep_dims);
  }

  template<typename StepFn, typename MeasFn>
  void update(Eigen::Vector<ValueType, ObsDim> meas_vec,
              StepFn step_fn,
              MeasFn meas_fn) {
    Eigen::VectorXd pred_state;
    Eigen::MatrixXd state_step_jacobian;
    std::tie(pred_state, state_step_jacobian) = step_fn(state);
    Eigen::MatrixXd pred_covariances =
        state_step_jacobian * state_cov * state_step_jacobian.transpose() + process_noise_cov;

    Eigen::VectorXd state_iterate = pred_state.eval();
    for (size_t i = 0; i < n_iekf_iters; i++) {
      Eigen::VectorXd meas_iterate;
      Eigen::MatrixXd meas_jacobian;
      std::tie(meas_iterate, meas_jacobian) = meas_fn(state_iterate);
      Eigen::MatrixXd kalman_gain =
          pred_covariances * meas_jacobian.transpose() *
          (meas_jacobian * pred_covariances * meas_jacobian.transpose() + meas_noise_cov).inverse();
      std::cout << "kalman_gain:\n" << kalman_gain << "\n";
      std::cout << "meas_vec:\n" << meas_vec << "\n";
      std::cout << "meas_iterate:\n" << meas_iterate << "\n";
      std::cout << "meas_diff:\n" << meas_vec - meas_iterate << "\n";
      std::cout << "pred_state:\n" << pred_state << "\n";
      std::cout << "state_iterate:\n" << state_iterate << "\n";
      std::cout << "iterate_diff:\n" << kalman_gain * (meas_vec - meas_iterate) << "\n";

      state_iterate = pred_state + kalman_gain * (meas_vec - meas_iterate - meas_jacobian * (pred_state - state_iterate));
      state_cov =
          (Eigen::Matrix<ValueType, StateDim, StateDim>::Identity(state_cov.rows(), state_cov.cols()) - kalman_gain * meas_jacobian) *
          pred_covariances;
    }
    state = state_iterate;
  }

  size_t n_iekf_iters = 10;

  Eigen::Vector<ValueType, StateDim> state;
  Eigen::Matrix<ValueType, StateDim, StateDim> state_cov;
  Eigen::Matrix<ValueType, StateDim, StateDim> process_noise_cov;
  Eigen::Matrix<ValueType, ObsDim, ObsDim> meas_noise_cov;
};
}  // namespace weaver
