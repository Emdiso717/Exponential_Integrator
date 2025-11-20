#ifndef _KRYLOV_HPP_
#define _KRYLOV_HPP_

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>

class krylov {
private:
  /* data */
  int m_max;
  int m_min;
  Eigen::VectorXd b;
  Eigen::MatrixXd A;
  Eigen::MatrixXd V;
  Eigen::MatrixXd H;
  double error = 1e-5;
  int m;
  int n;

  static int compute_initial_m_max(int n) {

    if (n <= 50)
      return std::min(20, n);
    else if (n <= 200)
      return std::min(30, n);
    else if (n <= 500)
      return std::min(40, n);
    else
      return std::min(50, n);
  }
  int phi_index = 0;

  void arnoldi();
  double compute_phi_element(int p, int m_size) {
    Eigen::MatrixXd Hm = H.topLeftCorner(m_size, m_size);
    double element_value = (m_size == 1) ? 1.0 / std::tgamma(p + 2.0) : 0.0;
    Eigen::VectorXd col_1 = Eigen::VectorXd::Zero(m_size);
    col_1(0) = 1.0;

    Eigen::VectorXd H_col1 = Hm * col_1;
    double term = H_col1(m_size - 1) / std::tgamma(p + 3.0);
    element_value += term;

    int iteration = 1;

    while (true) {
      H_col1 = Hm * H_col1;
      iteration++;
      term = H_col1(m_size - 1) / std::tgamma(p + 1.0 + iteration + 1.0);
      element_value += term;
      if (std::abs(term) < error * 1e-2 && iteration >= m_size - 1) {
        break;
      }
      if (iteration > 200) {
        break;
      }
    }

    return element_value;
  }

public:
  krylov(Eigen::VectorXd b, Eigen::MatrixXd A, int n)
      : b(b), A(A), n(n), phi_index(0) {
    m_min = 5;
    m_max = compute_initial_m_max(n);
    arnoldi();
    Eigen::MatrixXd Vm = V.leftCols(m);
    Eigen::MatrixXd Hm = H.topLeftCorner(m, m);
    V = Vm, H = Hm;
  }

  krylov(Eigen::VectorXd b, Eigen::MatrixXd A, int n, int phi_idx)
      : b(b), A(A), n(n), phi_index(phi_idx) {
    m_min = 5;
    m_max = compute_initial_m_max(n);
    arnoldi();
    Eigen::MatrixXd Vm = V.leftCols(m);
    Eigen::MatrixXd Hm = H.topLeftCorner(m, m);
    V = Vm, H = Hm;
  }

  krylov(Eigen::VectorXd b, Eigen::MatrixXd A, int n, int m_max_user,
         int phi_idx)
      : b(b), A(A), n(n), m_max(m_max_user), phi_index(phi_idx) {
    m_min = 5;
    m_max = std::min(m_max, n);
    arnoldi();
    Eigen::MatrixXd Vm = V.leftCols(m);
    Eigen::MatrixXd Hm = H.topLeftCorner(m, m);
    V = Vm, H = Hm;
  }

  krylov(Eigen::VectorXd b, Eigen::MatrixXd A, int n, int m_min_user,
         int m_max_user, double tol, int phi_idx)
      : b(b), A(A), n(n), m_min(m_min_user), m_max(std::min(m_max_user, n)),
        error(tol), phi_index(phi_idx) {
    arnoldi();
    Eigen::MatrixXd Vm = V.leftCols(m);
    Eigen::MatrixXd Hm = H.topLeftCorner(m, m);
    V = Vm, H = Hm;
  }

  Eigen::MatrixXd cal_phi_i(int i);

  void set_phi_index(int phi_idx) { phi_index = phi_idx; }

  int get_dimension() const { return m; }
  void print_H() const { std::cout << "H: " << H << std::endl; }
};

#endif