#include "EI.hpp"

class ETD1 : public EI {
private:
  Eigen::MatrixXd c; // Linear Part (system matrix A)
  std::function<Eigen::VectorXd(const Eigen::VectorXd &)>
      N; // Nonlinear part N(x) = f(x) - A x

public:
  ETD1(int n_,
       const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &Func,
       Eigen::VectorXd point, Eigen::MatrixXd _c, double _h)
      : EI(n_, Func, point, _h), c(_c) {
    N = [this](const Eigen::VectorXd &X) { return f(X) - c * X; };
  }
  void e_hc() {}
  Eigen::VectorXd step() {
    int dim = static_cast<int>(x.size());

    krylov krylov_phi0(x, c * h, dim, 0);
    Eigen::VectorXd ehA_x = krylov_phi0.cal_phi_i(0);

    Eigen::VectorXd Nx = N(x);
    krylov krylov_phi1(Nx, c * h, dim, 1);
    Eigen::VectorXd phi1_hA_Nx = krylov_phi1.cal_phi_i(1);

    // ETD1 update: x_{n+1} = e^{hA} x_n + h * phi_1(hA) N(x_n)
    x = ehA_x + h * phi1_hA_Nx;
    return x;
  }
};
