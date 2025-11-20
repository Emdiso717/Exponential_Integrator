#include "phy_system/phy_system.hpp"
#include "src/ETD1.hpp"
#include <Eigen/Dense>
#include <functional>
#include <igl/opengl/glfw/Viewer.h>
#include <iostream>
#include <vector>

using namespace std;

int main() {
  NonlinearChain2D sys = make_nonlinear_chain_2d_tangent(20, 100.0);
  const int numNodes = sys.numNodes;
  const int dof = sys.dof;

  // Integrator
  double dt = 1e-2;
  ETD1 ei(dof, sys.f, sys.z0, sys.A, dt);

  // Build edge list from springs
  Eigen::MatrixXi E(sys.springs.size(), 2);
  for (int e = 0; e < (int)sys.springs.size(); ++e) {
    E(e, 0) = sys.springs[e].i;
    E(e, 1) = sys.springs[e].j;
  }

  // Vertex positions V (n x 3)
  Eigen::MatrixXd V(numNodes, 3);
  auto fill_positions = [&](const Eigen::VectorXd &z) {
    for (int a = 0; a < numNodes; ++a) {
      V(a, 0) = z(2 * a + 0);
      V(a, 1) = z(2 * a + 1);
      V(a, 2) = 0.0;
    }
  };
  Eigen::VectorXd z = sys.z0;
  fill_positions(z);

  // Viewer
  igl::opengl::glfw::Viewer viewer;
  Eigen::RowVector3d color(0.2, 0.6, 0.9);
  viewer.data().set_edges(V, E, color);
  viewer.data().point_size = 8.0f;
  viewer.data().show_lines = true;
  viewer.core().is_animating = true;

  // Animate by stepping integrator each frame
  viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &v) {
    z = ei.step();
    fill_positions(z);
    v.data().clear_edges();
    v.data().set_edges(V, E, color);
    return false; // continue normal drawing
  };

  std::cout << "Initial energy: " << sys.energy(z) << std::endl;
  viewer.launch();
}
