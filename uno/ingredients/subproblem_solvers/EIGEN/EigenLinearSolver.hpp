// Copyright (c) 2024 Charlie Vanaret
// Licensed under the MIT license. See LICENSE file in the project directory for details.

#ifndef UNO_EIGENLINEARSOLVER_H
#define UNO_EIGENLINEARSOLVER_H

#include "../DirectSymmetricIndefiniteLinearSolver.hpp"
#include "../COOEvaluationSpace.hpp"
#include "linear_algebra/Vector.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

namespace uno {
   class EigenLinearSolver : public DirectSymmetricIndefiniteLinearSolver<double> {
   public:
      EigenLinearSolver();
      ~EigenLinearSolver() override = default;

      void initialize_hessian(const Subproblem& subproblem) override;
      void initialize_augmented_system(const Subproblem& subproblem) override;

      void do_symbolic_analysis() override;
      void do_numerical_factorization(const double* matrix_values) override;
      void solve_indefinite_system(const Vector<double>& matrix_values, const Vector<double>& rhs, Vector<double>& result) override;
      void solve_indefinite_system(Statistics& statistics, const Subproblem& subproblem, Direction& direction,
         const WarmstartInformation& warmstart_information) override;

      [[nodiscard]] Inertia get_inertia() const override;
      [[nodiscard]] size_t number_negative_eigenvalues() const override;
      [[nodiscard]] bool matrix_is_singular() const override;
      [[nodiscard]] size_t rank() const override;

      [[nodiscard]] EvaluationSpace& get_evaluation_space() override;

   protected:
      COOEvaluationSpace evaluation_space{};

      // Matrix dimensions
      size_t n{0};  // Number of variables
      size_t m{0};  // Number of constraints
      size_t augmented_size{0};  // Total size of augmented system

      // Sparse matrix storage
      Eigen::SparseMatrix<double> sparse_matrix;
      Eigen::SparseMatrix<double> original_matrix;  // Store original for iterative refinement

      // Multiple solver strategies
      std::unique_ptr<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>> ldlt_solver;
      std::unique_ptr<Eigen::SparseLU<Eigen::SparseMatrix<double>>> lu_solver;
      std::unique_ptr<Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>>> qr_solver;

      // Solver state
      enum class SolverType {
         LDLT,
         LU,
         QR
      };
      SolverType current_solver_type{SolverType::LDLT};

      // Matrix properties
      size_t positive_eigenvalues{0};
      size_t negative_eigenvalues{0};
      size_t zero_eigenvalues{0};
      bool is_singular{false};
      size_t matrix_rank{0};
      double condition_number{1.0};

      // Numerical tolerances
      static constexpr double zero_tolerance{1e-12};
      static constexpr double pivot_tolerance{1e-10};
      static constexpr double regularization_initial{1e-8};
      static constexpr double regularization_max{1e-2};
      static constexpr double scaling_threshold{1e10};  // For matrix scaling

      // Regularization state
      double current_regularization{0.0};
      bool needs_regularization{false};

      // State flags
      bool analysis_performed{false};
      bool factorization_performed{false};

      // Scaling vectors
      Eigen::VectorXd row_scales;
      Eigen::VectorXd col_scales;
      bool scaling_applied{false};

      // Helper methods
      bool try_factorization(const Eigen::SparseMatrix<double>& matrix);
      void apply_regularization(Eigen::SparseMatrix<double>& matrix, double delta);
      bool verify_solution(const Eigen::VectorXd& solution, const Eigen::VectorXd& rhs);
      void iterative_refinement(const Eigen::VectorXd& rhs, Eigen::VectorXd& solution);
      double estimate_condition_number();
      void compute_scaling_factors(const Eigen::SparseMatrix<double>& matrix);
      void apply_scaling(Eigen::SparseMatrix<double>& matrix);
      void scale_rhs(Eigen::VectorXd& rhs);
      void unscale_solution(Eigen::VectorXd& solution);
   };
} // namespace

#endif // UNO_EIGENLINEARSOLVER_H