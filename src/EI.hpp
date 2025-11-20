#ifndef EI_HPP
#define EI_HPP

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <functional>
#include <iostream>
#include "krylov.hpp"

class EI
{
protected:
    int n;    // Number of nodes N(vec3tor degree) = n * 3
    double h; // time step
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f;
    Eigen::VectorXd x;

public:
    EI(int n_, const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &Func, Eigen::VectorXd point, double _h)
        : n(n_), f(Func), x(point), h(_h) {};
    virtual Eigen::VectorXd step() { return Eigen::VectorXd(); };
};

#endif // EI_HPP