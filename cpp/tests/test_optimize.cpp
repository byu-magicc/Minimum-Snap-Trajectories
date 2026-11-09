#include <gtest/gtest.h>
#include <chrono>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "optimize.hpp"
#include "matrix_builder.hpp"
#include <iostream>

namespace trajectory_planner {
namespace tests {

class OptimizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup runs before every test. We can initialize standard parameters here if needed.
    }
};

TEST_F(OptimizerTest, Simple1DOptimization) {
    // ==========================================
    // 1. DUMMY PROBLEM SETUP
    // ==========================================
    // We will solve a trivial 1D B-Spline optimization:
    // Minimize Snap (P matrix) subject to boundary conditions (A matrix).
    
    int num_control_points = 5;

    // 1a. Objective Function (Minimize the sum of squares of control points)
    // P is a diagonal matrix of 2.0 (representing a highly simplified Hessian)
    std::vector<Eigen::Triplet<double>> p_triplets;
    for (int i = 0; i < num_control_points; ++i) {
        p_triplets.push_back(Eigen::Triplet<double>(i, i, 2.0));
    }
    Eigen::SparseMatrix<double> P(num_control_points, num_control_points);
    P.setFromTriplets(p_triplets.begin(), p_triplets.end());

    // q is a zero vector
    Eigen::VectorXd q = Eigen::VectorXd::Zero(num_control_points);

    // 1b. Constraints (Force the first CP to 0.0, and the last CP to 10.0)
    std::vector<Eigen::Triplet<double>> a_triplets;
    a_triplets.push_back(Eigen::Triplet<double>(0, 0, 1.0));                      // 1 * cp[0]
    a_triplets.push_back(Eigen::Triplet<double>(1, num_control_points - 1, 1.0)); // 1 * cp[4]
    
    Eigen::SparseMatrix<double> A(2, num_control_points);
    A.setFromTriplets(a_triplets.begin(), a_triplets.end());

    Eigen::VectorXd l(2);
    l << 0.0, 10.0; // Lower bounds
    
    Eigen::VectorXd u(2);
    u << 0.0, 10.0; // Upper bounds (Since l == u, these act as equality constraints!)

    // ==========================================
    // 2. EXECUTE OPTIMIZER
    // ==========================================
    std::cout << "[ C++ ENGINE ] Passing Dummy Matrix to OSQP..." << std::endl;
    
    Eigen::VectorXd optimized_cp;
    
    // Start the high-resolution clock
    auto start = std::chrono::high_resolution_clock::now();

    // Run the solver
    EXPECT_NO_THROW({
        optimized_cp = run_qp_solver(P, q, A, l, u);
    });

    // Stop the clock
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration in microseconds
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "\n[ BENCHMARK ] OSQP Execution Time: " << duration.count() << " microseconds\n" << std::endl;

    // ==========================================
    // 3. ASSERTIONS
    // ==========================================
    // Verify OSQP actually returned an array of the correct size
    ASSERT_EQ(optimized_cp.size(), num_control_points);

    std::cout << "\n--- Optimized Dummy Control Points ---" << std::endl;
    std::cout << optimized_cp.transpose() << "\n" << std::endl;

    // Verify the Equality Constraints were strictly enforced
    EXPECT_NEAR(optimized_cp(0), 0.0, 1e-3);
    EXPECT_NEAR(optimized_cp(num_control_points - 1), 10.0, 1e-3);
    
    // Verify the unconstrained inner points settled to 0.0 to minimize the P matrix objective
    EXPECT_NEAR(optimized_cp(1), 0.0, 1e-3);
    EXPECT_NEAR(optimized_cp(2), 0.0, 1e-3);
    EXPECT_NEAR(optimized_cp(3), 0.0, 1e-3);
}

} // namespace tests
} // namespace trajectory_planner