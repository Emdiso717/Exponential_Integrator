#pragma once

#include <Eigen/Dense>
#include <functional>
#include <vector>

struct Spring2D
{
    int i;
    int j;
    double restLength;
    double stiffness;
};

struct NonlinearChain2D
{
    int numNodes;
    int dof; // = numNodes * 2

    // System definition z' = A z + N(z)
    Eigen::MatrixXd A; // Only q' = v block is non-zero for this system
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f; // full RHS f(z)

    // Initial state z0 = [q0; v0]
    Eigen::VectorXd z0;

    // Geometry
    std::vector<Eigen::Vector2d> restPositions;
    std::vector<Spring2D> springs;

    // Compute total mechanical energy for state z = [q; v]
    double energy(const Eigen::VectorXd &z) const;
};

NonlinearChain2D make_nonlinear_chain_2d(int numNodes, double springK);


// Variant: linear part uses the true Jacobian (tangent stiffness) of F at rest
// i.e., K_t = \sum k (e e^T) assembled per edge at rest, with e = d0/||d0||.
// Then A has bottom-left = (1/m) K_t, and N(z) carries the nonlinear remainder F(q) - K_t q.
NonlinearChain2D make_nonlinear_chain_2d_tangent(int numNodes, double springK);
