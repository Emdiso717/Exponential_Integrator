#include "phy_system.hpp"
#include "iostream"


/**
 * Make a nonlinear chain in 2D with spring mass system
 * @param numNodes number of nodes
 * @param springK spring stiffness
 * @return NonlinearChain2D system structure
 * 
 *  x` = 0 * x + I x` +0
 *  x`` = 0 * x + 0 x`` + M^{-1} F(x)
 */
NonlinearChain2D make_nonlinear_chain_2d(int numNodes, double springK)
{
    NonlinearChain2D sys;
    sys.numNodes = numNodes;
    sys.dof = numNodes * 2;

    // Rest geometry: nodes along x-axis with unit spacing
    sys.restPositions.resize(numNodes);
    for (int a = 0; a < numNodes; ++a)
    {
        sys.restPositions[a] = Eigen::Vector2d(static_cast<double>(a), 0.0);
    }

    // Springs only between consecutive nodes
    auto add_spring = [&](int i, int j, double k) {
        double L0 = (sys.restPositions[j] - sys.restPositions[i]).norm();
        sys.springs.push_back(Spring2D{i, j, L0, k});
    };
    for (int i = 0; i + 1 < numNodes; ++i) add_spring(i, i + 1, springK);

    // Initial condition z0 = [q0; v0]
    Eigen::VectorXd q0(sys.dof);
    for (int a = 0; a < numNodes; ++a)
    {
        q0(2 * a + 0) = sys.restPositions[a](0);
        q0(2 * a + 1) = sys.restPositions[a](1);
    }
    // small perturbation at node 1 (y)
    if (numNodes >= 2) q0(3) += 0.2; 

    Eigen::VectorXd v0 = Eigen::VectorXd::Zero(sys.dof);
    sys.z0.resize(2 * sys.dof);
    sys.z0.head(sys.dof) = q0;
    sys.z0.tail(sys.dof) = v0;

    // Linear part: q' = v A = (0,I;0,0)
    sys.A = Eigen::MatrixXd::Zero(2 * sys.dof, 2 * sys.dof);
    sys.A.topRightCorner(sys.dof, sys.dof) = Eigen::MatrixXd::Identity(sys.dof, sys.dof);

    // Nonlinear RHS builder
    std::vector<Eigen::Vector2d> rest = sys.restPositions;
    std::vector<Spring2D> springs = sys.springs;
    double mass = 1.0;
    sys.f = [A = sys.A, rest, springs, mass, dof = sys.dof](const Eigen::VectorXd &z) -> Eigen::VectorXd {
        Eigen::VectorXd dz = Eigen::VectorXd::Zero(2 * dof);
        auto pos = [&](int a) { return Eigen::Vector2d(z(2 * a + 0), z(2 * a + 1)); };
        Eigen::VectorXd acc = Eigen::VectorXd::Zero(dof);
        for (const auto &sp : springs)
        {
            Eigen::Vector2d pi = pos(sp.i);
            Eigen::Vector2d pj = pos(sp.j);
            Eigen::Vector2d d = pj - pi;
            double L = d.norm();
            if (L > 1e-12)
            {
                Eigen::Vector2d F = sp.stiffness * (1.0 - sp.restLength / L) * d;
                acc(2 * sp.i + 0) += F(0);
                acc(2 * sp.i + 1) += F(1);
                acc(2 * sp.j + 0) -= F(0);
                acc(2 * sp.j + 1) -= F(1);
            }
        }
        dz.tail(dof) += (1.0 / mass) * acc;
        dz.head(dof) += z.tail(dof);
        return dz;
        
    };

    return sys;
}

NonlinearChain2D make_nonlinear_chain_2d_tangent(int numNodes, double springK)
{
    NonlinearChain2D sys;
    sys.numNodes = numNodes;
    sys.dof = numNodes * 2;

    // Initial position (0,0) (1,0) (2,0) ... (numNodes-1,0)
    sys.restPositions.resize(numNodes);
    for (int a = 0; a < numNodes; ++a)
    {
        sys.restPositions[a] = Eigen::Vector2d(static_cast<double>(a), 0.0);
    }

    // Springs between consecutive nodes
    auto add_spring = [&](int i, int j, double k) {
        double L0 = (sys.restPositions[j] - sys.restPositions[i]).norm();
        sys.springs.push_back(Spring2D{i, j, L0, k});
    };
    for (int i = 0; i + 1 < numNodes; ++i) add_spring(i, i + 1, springK);

    // Initial state
    Eigen::VectorXd q0(sys.dof);
    for (int a = 0; a < numNodes; ++a)
    {
        q0(2 * a + 0) = sys.restPositions[a](0);
        q0(2 * a + 1) = sys.restPositions[a](1);
    }
    if (numNodes >= 2) q0(3) += 0.2;
    Eigen::VectorXd v0 = Eigen::VectorXd::Zero(sys.dof);
    sys.z0.resize(2 * sys.dof);
    sys.z0.head(sys.dof) = q0;
    sys.z0.tail(sys.dof) = v0;

    // Tangent stiffness at rest: per edge add k * (e e^T) with e = d0 / ||d0||
    Eigen::MatrixXd Kt = Eigen::MatrixXd::Zero(sys.dof, sys.dof);
    for (const auto &sp : sys.springs)
    {
        Eigen::Vector2d d0 = sys.restPositions[sp.j] - sys.restPositions[sp.i];
        double L0 = d0.norm();
        if (L0 <= 1e-12) continue;
        Eigen::Vector2d e = d0 / L0;
        Eigen::Matrix2d P = e * e.transpose(); // projection along spring direction
        // compact assembly using indices
        for (int a = 0; a < 2; ++a)
        {
            for (int b = 0; b < 2; ++b)
            {
                double v = sp.stiffness * P(a, b);
                int ii = 2 * sp.i + a;
                int jj = 2 * sp.j + a;
                int kk = 2 * sp.i + b;
                int ll = 2 * sp.j + b;
                Kt(ii, kk) += v;
                Kt(jj, ll) += v;
                Kt(ii, ll) -= v;
                Kt(jj, kk) -= v;
            }
        }
    }
    double mass = 1.0;
    sys.A = Eigen::MatrixXd::Zero(2 * sys.dof, 2 * sys.dof);
    sys.A.topRightCorner(sys.dof, sys.dof) = Eigen::MatrixXd::Identity(sys.dof, sys.dof);
    sys.A.bottomLeftCorner(sys.dof, sys.dof) = (1.0 / mass) * Kt;
    std::cout<< "Kt:" << Kt<<std::endl;
    std::cout << sys.A << std::endl;

    // Nonlinear remainder N: (1/m)(F(q) - Kt q)
    std::vector<Eigen::Vector2d> rest = sys.restPositions;
    std::vector<Spring2D> springs = sys.springs;
    sys.f = [A = sys.A, rest, springs, mass, dof = sys.dof, Kt](const Eigen::VectorXd &z) -> Eigen::VectorXd {
        Eigen::VectorXd dz = Eigen::VectorXd::Zero(dof*2);
        auto pos = [&](int a) { return Eigen::Vector2d(z(2 * a + 0), z(2 * a + 1)); };
        Eigen::VectorXd Fnl = Eigen::VectorXd::Zero(dof);
        for (const auto &sp : springs)
        {
            Eigen::Vector2d pi = pos(sp.i);
            Eigen::Vector2d pj = pos(sp.j);
            Eigen::Vector2d d = pj - pi;
            double L = d.norm();
            if (L > 1e-12)
            {
                Eigen::Vector2d Fij = sp.stiffness * (1.0 - sp.restLength / L) * d;
                Fnl(2 * sp.i + 0) += Fij(0);
                Fnl(2 * sp.i + 1) += Fij(1);
                Fnl(2 * sp.j + 0) -= Fij(0);
                Fnl(2 * sp.j + 1) -= Fij(1);
            }
        }
        Eigen::VectorXd q = z.head(dof);
        dz.tail(dof) += (1.0 / mass) * (Fnl);
        dz.head(dof) += z.tail(dof);
        return dz;
    };

    return sys;
}

double NonlinearChain2D::energy(const Eigen::VectorXd &z) const
{
    // z = [q; v], dof entries each
    const int ndof = dof;
    const double mass = 1.0; // consistent with system construction

    // Kinetic: 1/2 m v^T v
    const Eigen::VectorXd v = z.tail(ndof);
    double kinetic = 0.5 * mass * v.squaredNorm();

    // Potential: sum_springs 1/2 k (|d| - L0)^2
    auto pos = [&](int a) { return Eigen::Vector2d(z(2 * a + 0), z(2 * a + 1)); };
    double potential = 0.0;
    for (const auto &sp : springs)
    {
        Eigen::Vector2d pi = pos(sp.i);
        Eigen::Vector2d pj = pos(sp.j);
        Eigen::Vector2d d = pj - pi;
        double L = d.norm();
        double stretch = L - sp.restLength;
        potential += 0.5 * sp.stiffness * (stretch * stretch);
    }

    return kinetic + potential;
}
