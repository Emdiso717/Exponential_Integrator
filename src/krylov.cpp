#include "krylov.hpp"
#include <chrono>

void krylov::arnoldi() {
  Eigen::VectorXd v1 = b / (b.norm());
  V.resize(n, m_max + 1);
  V.setZero();
  H.resize(m_max + 1, m_max);
  H.setZero();
  Eigen::VectorXd w;
  V.col(0) = v1;
  double beta = b.norm();
  for (int i = 0; i < m_max; i++) {
    w = A * (V.col(i));
    for (int j = 0; j <= i; j++) {
      H(j, i) = V.col(j).dot(w);
      w = w - H(j, i) * V.col(j);
    }
    double r = w.norm();
    H(i + 1, i) = r;
    V.col(i + 1) = w / r;
    if (i >= m_min - 1 && r > 0) {
      int current_m = i + 1;
      double h_m1_m = r;
      double phi_element = compute_phi_element(phi_index, current_m);
      double error_estimate = beta * std::abs(h_m1_m) * std::abs(phi_element);
      if (error_estimate <= error) {
        m = current_m;
        return;
      }
    }
  }

  m = m_max;
}

Eigen::MatrixXd krylov::cal_phi_i(int i) {
  Eigen::MatrixXd result =
      Eigen::MatrixXd::Identity(m, m) / std::tgamma(i + 1.0);
  Eigen::MatrixXd factor = result;
  Eigen::MatrixXd temp;
  Eigen::VectorXd e1 = Eigen::VectorXd::Zero(m);
  e1(0) = 1.0;
  int interation = 1;
  while (1) {
    temp = (factor * H) / (i + interation);
    result += temp;
    interation++;
    factor = temp;
    if (factor.norm() <= error) {
      break;
    }
  }
  return b.norm() * V * result * e1;
}
