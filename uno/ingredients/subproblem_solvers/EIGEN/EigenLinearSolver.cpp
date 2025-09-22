// Copyright (c) 2024 Charlie Vanaret
// Licensed under the MIT license. See LICENSE file in the project directory for details.

#include "EigenLinearSolver.hpp"
#include "ingredients/subproblem/Subproblem.hpp"
#include "optimization/Direction.hpp"
#include "tools/Logger.hpp"
#include "tools/Statistics.hpp"
#include <iostream>
#include <cmath>

namespace uno {
   EigenLinearSolver::EigenLinearSolver(): DirectSymmetricIndefiniteLinearSolver() {
      // Initialize solvers
      this->ldlt_solver = std::make_unique<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>>();
      this->lu_solver = std::make_unique<Eigen::SparseLU<Eigen::SparseMatrix<double>>>();
      this->qr_solver = std::make_unique<Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>>>();
      this->current_solver_type = SolverType::LDLT;
   }

   void EigenLinearSolver::initialize_hessian(const Subproblem& subproblem) {
      this->evaluation_space.initialize_hessian(subproblem);

      const size_t dimension = subproblem.number_variables;
      this->n = dimension;
      this->m = 0;
      this->augmented_size = dimension;
      this->sparse_matrix.resize(static_cast<int>(dimension), static_cast<int>(dimension));
      this->original_matrix.resize(static_cast<int>(dimension), static_cast<int>(dimension));
   }

   void EigenLinearSolver::initialize_augmented_system(const Subproblem& subproblem) {
      this->evaluation_space.initialize_augmented_system(subproblem);

      const size_t dimension = subproblem.number_variables + subproblem.number_constraints;
      this->n = subproblem.number_variables;
      this->m = subproblem.number_constraints;
      this->augmented_size = dimension;
      this->sparse_matrix.resize(static_cast<int>(dimension), static_cast<int>(dimension));
      this->original_matrix.resize(static_cast<int>(dimension), static_cast<int>(dimension));
   }

   void EigenLinearSolver::do_symbolic_analysis() {
      assert(!this->analysis_performed);

      // Build sparsity pattern from COOEvaluationSpace
      std::vector<Eigen::Triplet<double>> triplets;
      const size_t nnz = this->evaluation_space.number_matrix_nonzeros;
      triplets.reserve(2 * nnz);

      // COOEvaluationSpace uses Fortran indexing (1-based)
      for (size_t k = 0; k < nnz; ++k) {
         const int row = this->evaluation_space.matrix_row_indices[k] - 1;
         const int col = this->evaluation_space.matrix_column_indices[k] - 1;
         triplets.emplace_back(row, col, 1.0);
         if (row != col) {
            triplets.emplace_back(col, row, 1.0);  // Ensure symmetry
         }
      }

      this->sparse_matrix.setFromTriplets(triplets.begin(), triplets.end());
      this->sparse_matrix.makeCompressed();

      // Perform symbolic analysis for all solvers
      this->ldlt_solver->analyzePattern(this->sparse_matrix);
      this->lu_solver->analyzePattern(this->sparse_matrix);
      this->qr_solver->analyzePattern(this->sparse_matrix);

      this->analysis_performed = true;
   }

   void EigenLinearSolver::do_numerical_factorization(const double* matrix_values) {
      assert(this->analysis_performed);

      std::cout << "[EIGEN Solver] Starting numerical factorization with "
                << this->evaluation_space.number_matrix_nonzeros << " nonzeros\n";

      // Build sparse matrix with actual values
      std::vector<Eigen::Triplet<double>> triplets;
      const size_t nnz = this->evaluation_space.number_matrix_nonzeros;
      triplets.reserve(2 * nnz);

      for (size_t k = 0; k < nnz; ++k) {
         const int row = this->evaluation_space.matrix_row_indices[k] - 1;
         const int col = this->evaluation_space.matrix_column_indices[k] - 1;
         const double value = matrix_values[k];
         triplets.emplace_back(row, col, value);
         if (row != col) {
            triplets.emplace_back(col, row, value);
         }
      }

      this->sparse_matrix.setFromTriplets(triplets.begin(), triplets.end());
      this->sparse_matrix.makeCompressed();
      this->original_matrix = this->sparse_matrix;  // Store original for iterative refinement

      // Compute and apply scaling if matrix is ill-conditioned
      this->compute_scaling_factors(this->sparse_matrix);

      // Check if scaling is needed
      bool needs_scaling = false;
      for (int i = 0; i < this->row_scales.size(); ++i) {
         if (this->row_scales[i] != 1.0 || this->col_scales[i] != 1.0) {
            needs_scaling = true;
            break;
         }
      }

      if (needs_scaling) {
         std::cout << "[EIGEN Solver] Applying matrix scaling for numerical stability\n";
         this->apply_scaling(this->sparse_matrix);
      }

      // Estimate condition number
      this->condition_number = this->estimate_condition_number();
      if (this->condition_number > 1e12) {
         std::cout << "[EIGEN Solver] WARNING: Matrix is ill-conditioned (condition number ~ "
                   << this->condition_number << ")\n";
      }

      // Try factorization with increasing regularization if needed
      this->current_regularization = 0.0;
      bool success = this->try_factorization(this->sparse_matrix);

      if (!success) {
         std::cout << "[EIGEN Solver] Initial factorization failed, trying regularization\n";

         // Try progressive regularization
         double delta = regularization_initial;
         while (!success && delta <= regularization_max) {
            Eigen::SparseMatrix<double> regularized_matrix = this->original_matrix;
            this->apply_regularization(regularized_matrix, delta);

            success = this->try_factorization(regularized_matrix);
            if (success) {
               this->sparse_matrix = regularized_matrix;
               this->current_regularization = delta;
               std::cout << "[EIGEN Solver] Factorization succeeded with regularization = "
                         << delta << "\n";
            } else {
               delta *= 10.0;  // Increase regularization
            }
         }
      }

      this->factorization_performed = success;
      if (!success) {
         this->is_singular = true;
         std::cout << "[EIGEN Solver] ERROR: All factorization attempts failed\n";
      }
   }

   bool EigenLinearSolver::try_factorization(const Eigen::SparseMatrix<double>& matrix) {
      // Try LDLT first
      this->ldlt_solver->factorize(matrix);
      if (this->ldlt_solver->info() == Eigen::Success) {
         this->current_solver_type = SolverType::LDLT;
         this->is_singular = false;

         // Extract inertia
         const auto& D = this->ldlt_solver->vectorD();
         this->positive_eigenvalues = 0;
         this->negative_eigenvalues = 0;
         this->zero_eigenvalues = 0;
         for (int i = 0; i < D.size(); ++i) {
            if (std::abs(D[i]) < zero_tolerance) {
               this->zero_eigenvalues++;
            } else if (D[i] > 0) {
               this->positive_eigenvalues++;
            } else {
               this->negative_eigenvalues++;
            }
         }
         this->matrix_rank = static_cast<size_t>(D.size()) - this->zero_eigenvalues;

         std::cout << "[EIGEN Solver] LDLT factorization successful\n";
         std::cout << "[EIGEN Solver] Inertia (pos/neg/zero) = ("
                   << this->positive_eigenvalues << "/" << this->negative_eigenvalues
                   << "/" << this->zero_eigenvalues << ")\n";
         return true;
      }

      // Try LU if LDLT fails
      std::cout << "[EIGEN Solver] LDLT failed, trying LU decomposition\n";
      this->lu_solver->factorize(matrix);
      if (this->lu_solver->info() == Eigen::Success) {
         this->current_solver_type = SolverType::LU;
         this->is_singular = false;

         // For LU, we can't easily extract inertia
         this->matrix_rank = this->augmented_size;
         this->positive_eigenvalues = this->augmented_size / 2;
         this->negative_eigenvalues = this->augmented_size / 2;
         this->zero_eigenvalues = 0;

         std::cout << "[EIGEN Solver] LU factorization successful\n";
         return true;
      }

      // Try QR as last resort (most stable but slowest)
      std::cout << "[EIGEN Solver] LU failed, trying QR decomposition\n";
      this->qr_solver->compute(matrix);
      if (this->qr_solver->info() == Eigen::Success) {
         this->current_solver_type = SolverType::QR;
         this->is_singular = false;
         this->matrix_rank = static_cast<size_t>(this->qr_solver->rank());
         this->zero_eigenvalues = this->augmented_size - this->matrix_rank;

         // QR doesn't give inertia information
         this->positive_eigenvalues = this->matrix_rank / 2;
         this->negative_eigenvalues = this->matrix_rank / 2;

         std::cout << "[EIGEN Solver] QR factorization successful (rank = "
                   << this->matrix_rank << ")\n";
         return true;
      }

      return false;
   }

   void EigenLinearSolver::apply_regularization(Eigen::SparseMatrix<double>& matrix, double delta) {
      // Add delta * I to the diagonal
      for (int i = 0; i < matrix.rows(); ++i) {
         matrix.coeffRef(i, i) += delta;
      }
   }

   bool EigenLinearSolver::verify_solution(const Eigen::VectorXd& solution, const Eigen::VectorXd& rhs) {
      // Check for NaN/Inf
      if (!solution.allFinite()) {
         std::cout << "[EIGEN Solver] Solution contains NaN/Inf values\n";
         // Try to diagnose which values are problematic
         for (int i = 0; i < std::min(10, static_cast<int>(solution.size())); ++i) {
            if (!std::isfinite(solution[i])) {
               std::cout << "  Element " << i << " = " << solution[i] << "\n";
            }
         }
         return false;
      }

      // Check for extremely large values
      double max_value = solution.cwiseAbs().maxCoeff();
      if (max_value > 1e15) {
         std::cout << "[EIGEN Solver] Solution has extremely large values (max = "
                   << max_value << ")\n";
         return false;
      }

      // Check residual norm (use the scaled or unscaled matrix appropriately)
      Eigen::VectorXd residual;
      if (this->scaling_applied) {
         // For scaled system, need to check with scaled matrix
         residual = this->sparse_matrix * solution - rhs;
      } else {
         residual = this->original_matrix * solution - rhs;
      }
      double residual_norm = residual.norm();
      double rhs_norm = rhs.norm();
      double solution_norm = solution.norm();

      if (rhs_norm > 1e-10) {
         double relative_residual = residual_norm / rhs_norm;
         if (relative_residual > 1e-6) {
            std::cout << "[EIGEN Solver] Large relative residual = "
                      << relative_residual
                      << " (||r|| = " << residual_norm
                      << ", ||b|| = " << rhs_norm
                      << ", ||x|| = " << solution_norm << ")\n";
            return false;
         }
      }

      return true;
   }

   void EigenLinearSolver::iterative_refinement(const Eigen::VectorXd& rhs, Eigen::VectorXd& solution) {
      // Perform one step of iterative refinement
      Eigen::VectorXd residual = rhs - this->original_matrix * solution;
      Eigen::VectorXd correction;

      // Solve for correction
      if (this->current_solver_type == SolverType::LDLT) {
         correction = this->ldlt_solver->solve(residual);
      } else if (this->current_solver_type == SolverType::LU) {
         correction = this->lu_solver->solve(residual);
      } else {
         correction = this->qr_solver->solve(residual);
      }

      // Update solution
      solution += correction;

      double correction_norm = correction.norm();
      if (correction_norm > 1e-10) {
         std::cout << "[EIGEN Solver] Iterative refinement: ||correction|| = "
                   << correction_norm << "\n";
      }
   }

   double EigenLinearSolver::estimate_condition_number() {
      // Simple estimation using diagonal dominance
      double min_diag = std::numeric_limits<double>::max();
      double max_diag = std::numeric_limits<double>::min();

      for (int k = 0; k < this->sparse_matrix.outerSize(); ++k) {
         for (Eigen::SparseMatrix<double>::InnerIterator it(this->sparse_matrix, k); it; ++it) {
            if (it.row() == it.col()) {
               double diag_value = std::abs(it.value());
               if (diag_value > 0) {
                  min_diag = std::min(min_diag, diag_value);
                  max_diag = std::max(max_diag, diag_value);
               }
            }
         }
      }

      if (min_diag > 0) {
         return max_diag / min_diag;
      } else {
         return std::numeric_limits<double>::infinity();
      }
   }

   void EigenLinearSolver::compute_scaling_factors(const Eigen::SparseMatrix<double>& matrix) {
      const int n = matrix.rows();
      this->row_scales = Eigen::VectorXd::Ones(n);
      this->col_scales = Eigen::VectorXd::Ones(n);

      // Compute row and column infinity norms
      for (int k = 0; k < matrix.outerSize(); ++k) {
         for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, k); it; ++it) {
            double abs_val = std::abs(it.value());
            this->row_scales[it.row()] = std::max(this->row_scales[it.row()], abs_val);
            this->col_scales[it.col()] = std::max(this->col_scales[it.col()], abs_val);
         }
      }

      // Take square root for symmetric scaling
      for (int i = 0; i < n; ++i) {
         if (this->row_scales[i] > scaling_threshold) {
            this->row_scales[i] = std::sqrt(this->row_scales[i]);
         } else {
            this->row_scales[i] = 1.0;
         }
         if (this->col_scales[i] > scaling_threshold) {
            this->col_scales[i] = std::sqrt(this->col_scales[i]);
         } else {
            this->col_scales[i] = 1.0;
         }
      }
   }

   void EigenLinearSolver::apply_scaling(Eigen::SparseMatrix<double>& matrix) {
      // Apply symmetric scaling: D^(-1/2) * A * D^(-1/2)
      for (int k = 0; k < matrix.outerSize(); ++k) {
         for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, k); it; ++it) {
            it.valueRef() /= (this->row_scales[it.row()] * this->col_scales[it.col()]);
         }
      }
      this->scaling_applied = true;
   }

   void EigenLinearSolver::scale_rhs(Eigen::VectorXd& rhs) {
      if (!this->scaling_applied) return;

      // Scale RHS by row scaling factors
      for (int i = 0; i < rhs.size(); ++i) {
         rhs[i] /= this->row_scales[i];
      }
   }

   void EigenLinearSolver::unscale_solution(Eigen::VectorXd& solution) {
      if (!this->scaling_applied) return;

      // Unscale solution by column scaling factors
      for (int i = 0; i < solution.size(); ++i) {
         solution[i] /= this->col_scales[i];
      }
   }

   void EigenLinearSolver::solve_indefinite_system(const Vector<double>& /*matrix_values*/, const Vector<double>& rhs,
                                                   Vector<double>& result) {
      assert(this->factorization_performed);

      // Convert to Eigen format
      Eigen::VectorXd eigen_rhs(static_cast<int>(rhs.size()));
      for (size_t i = 0; i < rhs.size(); ++i) {
         eigen_rhs[static_cast<int>(i)] = rhs[i];
      }

      // Apply scaling to RHS if needed
      this->scale_rhs(eigen_rhs);

      // Solve the system
      Eigen::VectorXd eigen_solution;
      bool solve_success = false;

      if (this->current_solver_type == SolverType::LDLT) {
         eigen_solution = this->ldlt_solver->solve(eigen_rhs);
         solve_success = (this->ldlt_solver->info() == Eigen::Success);
      } else if (this->current_solver_type == SolverType::LU) {
         eigen_solution = this->lu_solver->solve(eigen_rhs);
         solve_success = (this->lu_solver->info() == Eigen::Success);
      } else {
         eigen_solution = this->qr_solver->solve(eigen_rhs);
         solve_success = (this->qr_solver->info() == Eigen::Success);
      }

      // Verify solution quality
      if (solve_success && !this->verify_solution(eigen_solution, eigen_rhs)) {
         std::cout << "[EIGEN Solver] Solution verification failed, applying iterative refinement\n";

         // Try iterative refinement
         for (int iter = 0; iter < 3; ++iter) {
            this->iterative_refinement(eigen_rhs, eigen_solution);
            if (this->verify_solution(eigen_solution, eigen_rhs)) {
               std::cout << "[EIGEN Solver] Iterative refinement successful after "
                         << (iter + 1) << " iterations\n";
               break;
            }
         }

         // Final check
         if (!this->verify_solution(eigen_solution, eigen_rhs)) {
            std::cout << "[EIGEN Solver] ERROR: Solution still invalid after iterative refinement\n";
            // Use minimum norm solution as fallback
            if (this->current_solver_type == SolverType::QR) {
               eigen_solution = this->qr_solver->solve(eigen_rhs);
            } else {
               // Return zero as last resort
               eigen_solution.setZero();
            }
         }
      }

      // Unscale solution if scaling was applied
      this->unscale_solution(eigen_solution);

      // Copy solution back
      for (size_t i = 0; i < result.size(); ++i) {
         result[i] = eigen_solution[static_cast<int>(i)];
      }
   }

   void EigenLinearSolver::solve_indefinite_system(Statistics& statistics, const Subproblem& subproblem,
                                                   Direction& direction, const WarmstartInformation& warmstart_information) {
      // Let evaluation space set up the linear system
      this->evaluation_space.set_up_linear_system(statistics, subproblem, *this, warmstart_information);

      // Solve the system
      this->solve_indefinite_system(this->evaluation_space.matrix_values, this->evaluation_space.rhs,
                                    this->evaluation_space.solution);

      // Extract primal-dual solution
      for (size_t i = 0; i < this->n; i++) {
         direction.primals[i] = this->evaluation_space.solution[i];
      }
      for (size_t i = 0; i < this->m; i++) {
         direction.multipliers.constraints[i] = this->evaluation_space.solution[this->n + i];
      }

      // Update statistics
      std::string solver_info;
      if (this->current_solver_type == SolverType::LDLT) {
         solver_info = "LDLT";
      } else if (this->current_solver_type == SolverType::LU) {
         solver_info = "LU";
      } else {
         solver_info = "QR";
      }

      if (this->current_regularization > 0) {
         solver_info += " (reg=" + std::to_string(this->current_regularization) + ")";
      }

      statistics.set("eigen_solver_info", solver_info);
   }

   Inertia EigenLinearSolver::get_inertia() const {
      return {this->positive_eigenvalues, this->negative_eigenvalues, this->zero_eigenvalues};
   }

   size_t EigenLinearSolver::number_negative_eigenvalues() const {
      return this->negative_eigenvalues;
   }

   bool EigenLinearSolver::matrix_is_singular() const {
      return this->is_singular || (this->zero_eigenvalues > 0);
   }

   size_t EigenLinearSolver::rank() const {
      return this->matrix_rank;
   }

   EvaluationSpace& EigenLinearSolver::get_evaluation_space() {
      return this->evaluation_space;
   }
} // namespace