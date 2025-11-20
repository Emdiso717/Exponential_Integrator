#include "src/krylov.hpp"
#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <unsupported/Eigen/MatrixFunctions>

using namespace Eigen;
using namespace std;

// 测试 1: 基本功能测试 - 计算 exp(hA)v
void test_basic_expmv() {
  cout << "\n=== Test 1: Basic exp(hA)v computation ===" << endl;

  int n = 10000;
  double h = 0.1;

  // 创建一个简单的测试矩阵
  MatrixXd A = MatrixXd::Random(n, n);
  // A = (A + A.transpose()) / 2.0; // 对称化
  // A *= 0.1;                      // 缩放，使特征值较小

  VectorXd v = VectorXd::Random(n);
  v.normalize();

  // phi_i 的精确计算（通过级数展开验证）
  // MatrixXd H = h * A;
  // MatrixXd phi_exact = MatrixXd::Identity(n, n) / std::tgamma(0 + 1.0);
  // MatrixXd term = phi_exact;
  // int k = 1;
  // while (term.norm() > 1e-10) {
  //   term = (term * H) / (0 + k);
  //   phi_exact += term;
  //   k++;
  //   if (k > 100)
  //     break;
  // }
  // VectorXd exact = phi_exact * v;

  // Krylov 方法
  krylov kry(v, h * A, n, 0); // phi_index = 0 for exp
  VectorXd krylov_result = kry.cal_phi_i(0);

  // 计算误差
  // double error = (krylov_result - exact).norm();
  // double relative_error = error / exact.norm();
  cout << "Matrix size: " << n << "x" << n << endl;
  cout << "Krylov dimension used: " << kry.get_dimension() << endl;
  // cout << "Absolute error: " << scientific << error << endl;
  // cout << "Relative error: " << scientific << relative_error << endl;

  // if (relative_error < 1e-5) {
  //   cout << "✓ Test PASSED" << endl;
  // } else {
  //   cout << "✗ Test FAILED" << endl;
  // }
}

// // 测试 2: phi 函数计算
// void test_phi_functions() {
//   cout << "\n=== Test 2: Phi functions computation ===" << endl;

//   int n = 8;
//   double h = 0.05;

//   MatrixXd A = MatrixXd::Random(n, n);
//   A = (A + A.transpose()) / 2.0;
//   A *= 0.1;

//   VectorXd v = VectorXd::Random(n);
//   v.normalize();

//   // 测试 phi_0, phi_1, phi_2
//   for (int i = 0; i <= 2; i++) {
//     krylov kry(v, h * A, n, i);
//     VectorXd result = kry.cal_phi_i(i);

//     // phi_i 的精确计算（通过级数展开验证）
//     MatrixXd H = h * A;
//     MatrixXd phi_exact = MatrixXd::Identity(n, n) / std::tgamma(i + 1.0);
//     MatrixXd term = phi_exact;
//     int k = 1;
//     while (term.norm() > 1e-10) {
//       term = (term * H) / (i + k);
//       phi_exact += term;
//       k++;
//       if (k > 100)
//         break;
//     }
//     VectorXd exact = phi_exact * v;

//     double error = (result - exact).norm();
//     cout << "phi_" << i << " relative error: " << scientific
//          << error / exact.norm() << endl;
//   }
//   cout << "✓ Phi functions test completed" << endl;
// }

// // 测试 3: 不同矩阵大小的性能
// void test_performance() {
//   cout << "\n=== Test 3: Performance with different matrix sizes ===" <<
//   endl;

//   vector<int> sizes = {20, 50, 100, 200, 500, 1000, 3000};
//   double h = 0.1;

//   for (int n : sizes) {
//     MatrixXd A = MatrixXd::Random(n, n);
//     A = (A + A.transpose()) / 2.0;
//     // 不缩放矩阵，保持原始特征值范围
//     // A *= 0.1;  // 注释掉，让矩阵保持原始大小

//     VectorXd v = VectorXd::Random(n);
//     v.normalize();

//     auto start = chrono::high_resolution_clock::now();
//     krylov kry(v, h * A, n, 0);
//     VectorXd result = kry.cal_phi_i(0);
//     auto end = chrono::high_resolution_clock::now();

//     auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

//     // 计算矩阵的谱半径（最大特征值的绝对值）来理解收敛速度
//     Eigen::SelfAdjointEigenSolver<MatrixXd> solver(A);
//     double max_eigenvalue = solver.eigenvalues().cwiseAbs().maxCoeff();
//     double h_norm = h * max_eigenvalue;

//     cout << "Size: " << setw(5) << n << "x" << setw(5) << n
//          << " | Krylov dim: " << setw(3) << kry.get_dimension()
//          << " | h*||A||: " << scientific << setprecision(2) << h_norm
//          << " | Time: " << fixed << setw(8) << duration.count() << " μs"
//          << endl;
//   }
// }

// // 测试 4: 误差估计的准确性
// void test_error_estimation() {
//   cout << "\n=== Test 4: Error estimation accuracy ===" << endl;

//   int n = 15;
//   double h = 0.1;

//   MatrixXd A = MatrixXd::Random(n, n);
//   A = (A + A.transpose()) / 2.0;
//   A *= 0.1;

//   VectorXd v = VectorXd::Random(n);
//   v.normalize();

//   // 使用不同的容差
//   vector<double> tolerances = {1e-3, 1e-5, 1e-7, 1e-9};

//   MatrixXd exp_hA = (h * A).exp();
//   VectorXd exact = exp_hA * v;

//   for (double tol : tolerances) {
//     krylov kry(v, h * A, n, 5, 30, tol, 0); // m_min, m_max, tol, phi_idx
//     VectorXd result = kry.cal_phi_i(0);

//     double actual_error = (result - exact).norm() / exact.norm();
//     cout << "Tolerance: " << scientific << tol
//          << " | Actual error: " << actual_error
//          << " | Krylov dim: " << kry.get_dimension() << endl;
//   }
// }

// // 测试 5: 特殊矩阵结构（分块矩阵）
// void test_block_structure() {
//   cout << "\n=== Test 5: Block matrix structure ===" << endl;

//   int dof = 10;
//   int n = 2 * dof;

//   // 创建分块矩阵 A = [0, I; Kt, 0]
//   MatrixXd A = MatrixXd::Zero(n, n);
//   A.topRightCorner(dof, dof) = MatrixXd::Identity(dof, dof);

//   // 创建一个简单的 Kt（对称正定）
//   MatrixXd Kt = MatrixXd::Random(dof, dof);
//   Kt = (Kt + Kt.transpose()) / 2.0;
//   Kt += MatrixXd::Identity(dof, dof) * 2.0; // 确保正定
//   A.bottomLeftCorner(dof, dof) = Kt;

//   double h = 0.01;
//   VectorXd v = VectorXd::Random(n);
//   v.normalize();

//   // 精确解
//   MatrixXd exp_hA = (h * A).exp();
//   VectorXd exact = exp_hA * v;

//   // Krylov 方法
//   krylov kry(v, h * A, n, 0);
//   VectorXd result = kry.cal_phi_i(0);

//   double error = (result - exact).norm() / exact.norm();
//   cout << "Block matrix size: " << n << "x" << n << endl;
//   cout << "Krylov dimension: " << kry.get_dimension() << endl;
//   cout << "Relative error: " << scientific << error << endl;

//   if (error < 1e-5) {
//     cout << "✓ Block structure test PASSED" << endl;
//   } else {
//     cout << "✗ Block structure test FAILED" << endl;
//   }
// }

// // 测试 6: 不同 phi_index 的影响
// void test_phi_index() {
//   cout << "\n=== Test 6: Different phi_index values ===" << endl;

//   int n = 12;
//   double h = 0.1;

//   MatrixXd A = MatrixXd::Random(n, n);
//   A = (A + A.transpose()) / 2.0;
//   A *= 0.1;

//   VectorXd v = VectorXd::Random(n);
//   v.normalize();

//   // 测试不同的 phi_index
//   for (int phi_idx = 0; phi_idx <= 2; phi_idx++) {
//     krylov kry(v, h * A, n, phi_idx);
//     int dim = kry.get_dimension();
//     VectorXd result = kry.cal_phi_i(phi_idx);

//     cout << "phi_index: " << phi_idx << " | Krylov dim: " << dim
//          << " | Result norm: " << fixed << setprecision(6) << result.norm()
//          << endl;
//   }
// }

int main() {
  cout << "========================================" << endl;
  cout << "Krylov Subspace Method Test Suite" << endl;
  cout << "========================================" << endl;

  try {
    test_basic_expmv();
    // test_phi_functions();
    // test_performance();
    // test_error_estimation();
    // test_block_structure();
    // test_phi_index();

    cout << "\n========================================" << endl;
    cout << "All tests completed!" << endl;
    cout << "========================================" << endl;
  } catch (const exception &e) {
    cerr << "Error: " << e.what() << endl;
    return 1;
  }

  return 0;
}
