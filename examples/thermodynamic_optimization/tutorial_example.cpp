// tutorial_example.cpp
// 完整教程：如何在 C++ 中编写优化问题

#include <iostream>
#include <cmath>
#include <vector>
#include "model/Model.hpp"
#include "Uno.hpp"
#include "options/Options.hpp"
#include "options/DefaultOptions.hpp"
#include "options/Presets.hpp"
#include "optimization/Iterate.hpp"
#include "linear_algebra/Vector.hpp"
#include "linear_algebra/SparseVector.hpp"
#include "symbolic/Range.hpp"

// ========================================
// 示例问题：Rosenbrock 函数优化
// 
// 最小化: f(x,y) = (1-x)² + 100*(y-x²)²
// 约束:   x² + y² ≤ 2
//         x + y = 1
// ========================================

class RosenbrockProblem : public uno::Model {
private:
    // 存储固定变量和松弛变量（必需的数据成员）
    uno::Vector<size_t> fixed_variables;
    uno::SparseVector<size_t> slacks;
    
public:
    // ====== 构造函数 ======
    RosenbrockProblem() 
        : Model("Rosenbrock",  // 问题名称
                2,             // 2个变量 (x, y)
                2,             // 2个约束
                1.0),          // 1.0 表示最小化
          fixed_variables(0),  // 没有固定变量
          slacks(0)           // 没有松弛变量
    {
        std::cout << "创建 Rosenbrock 优化问题\n";
    }
    
    // ====== 1. 线性算子支持（通常返回 false）======
    bool has_jacobian_operator() const override { return false; }
    bool has_jacobian_transposed_operator() const override { return false; }
    bool has_hessian_operator() const override { return false; }
    bool has_hessian_matrix() const override { return true; }  // 我们提供 Hessian 矩阵
    
    // ====== 2. 目标函数 ======
    double evaluate_objective(const uno::Vector<double>& vars) const override {
        double x = vars[0];
        double y = vars[1];
        
        // Rosenbrock 函数: (1-x)² + 100*(y-x²)²
        double term1 = (1 - x) * (1 - x);
        double term2 = 100 * (y - x*x) * (y - x*x);
        
        return term1 + term2;
    }
    
    // ====== 3. 约束函数 ======
    void evaluate_constraints(const uno::Vector<double>& vars, 
                            std::vector<double>& constraints) const override {
        double x = vars[0];
        double y = vars[1];
        
        // 约束1: x² + y² - 2 ≤ 0 (转换为 x² + y² - 2 = 0 的形式)
        constraints[0] = x*x + y*y - 2.0;
        
        // 约束2: x + y - 1 = 0
        constraints[1] = x + y - 1.0;
    }
    
    // ====== 4. 目标函数梯度 ======
    void evaluate_objective_gradient(const uno::Vector<double>& vars, 
                                    uno::Vector<double>& gradient) const override {
        double x = vars[0];
        double y = vars[1];
        
        // ∂f/∂x = -2(1-x) + 200(y-x²)(-2x)
        gradient[0] = -2*(1-x) - 400*x*(y - x*x);
        
        // ∂f/∂y = 200(y-x²)
        gradient[1] = 200*(y - x*x);
    }
    
    // ====== 5. 变量界限 ======
    double variable_lower_bound(size_t index) const override {
        return -10.0;  // x, y ∈ [-10, 10]
    }
    
    double variable_upper_bound(size_t index) const override {
        return 10.0;
    }
    
    // ====== 6. 约束界限 ======
    double constraint_lower_bound(size_t index) const override {
        if (index == 0) return -2.0;  // 第一个约束是不等式 (≤ 0), 下界设为 -2
        return 0.0;  // 第二个约束是等式
    }
    
    double constraint_upper_bound(size_t index) const override {
        return 0.0;  // 两个约束的上界都是 0
    }
    
    // ====== 7. 初始点 ======
    void initial_primal_point(uno::Vector<double>& x0) const override {
        x0[0] = -1.0;  // x 初始值
        x0[1] = 2.0;   // y 初始值
    }
    
    void initial_dual_point(uno::Vector<double>& y0) const override {
        y0[0] = 0.0;  // 第一个约束的拉格朗日乘子
        y0[1] = 0.0;  // 第二个约束的拉格朗日乘子
    }
    
    // ====== 8. Jacobian（约束的梯度）======
    void compute_constraint_jacobian_sparsity(int* row_indices, int* column_indices, 
                                             int solver_indexing, uno::MatrixOrder order) const override {
        // 稀疏结构：每个约束对每个变量的偏导
        // 约束1对x, 约束1对y, 约束2对x, 约束2对y
        int idx = 0;
        for (int i = 0; i < 2; ++i) {      // 2个约束
            for (int j = 0; j < 2; ++j) {  // 2个变量
                row_indices[idx] = i + solver_indexing;
                column_indices[idx] = j + solver_indexing;
                idx++;
            }
        }
    }
    
    void evaluate_constraint_jacobian(const uno::Vector<double>& vars, 
                                     double* jacobian_values) const override {
        double x = vars[0];
        double y = vars[1];
        
        // Jacobian 矩阵（按行存储）：
        // | ∂c1/∂x  ∂c1/∂y |   | 2x   2y |
        // | ∂c2/∂x  ∂c2/∂y | = | 1    1  |
        
        jacobian_values[0] = 2*x;  // ∂c1/∂x
        jacobian_values[1] = 2*y;  // ∂c1/∂y
        jacobian_values[2] = 1.0;  // ∂c2/∂x
        jacobian_values[3] = 1.0;  // ∂c2/∂y
    }
    
    // ====== 9. Hessian ======
    void compute_hessian_sparsity(int* row_indices, int* column_indices, 
                                 int solver_indexing) const override {
        // Hessian 是对称的，只存储上三角
        row_indices[0] = 0 + solver_indexing; column_indices[0] = 0 + solver_indexing; // (0,0)
        row_indices[1] = 0 + solver_indexing; column_indices[1] = 1 + solver_indexing; // (0,1)
        row_indices[2] = 1 + solver_indexing; column_indices[2] = 1 + solver_indexing; // (1,1)
    }
    
    void evaluate_lagrangian_hessian(const uno::Vector<double>& vars, double objective_multiplier,
                                    const uno::Vector<double>& constraint_multipliers, 
                                    double* hessian_values) const override {
        double x = vars[0];
        double lambda1 = constraint_multipliers[0];
        
        // Lagrangian 的 Hessian（只存上三角）
        // H = ∇²f + Σ λᵢ∇²cᵢ
        
        // ∂²L/∂x² 
        hessian_values[0] = objective_multiplier * (2 + 800*x*x - 400*(vars[1]-x*x)) + lambda1 * 2;
        
        // ∂²L/∂x∂y
        hessian_values[1] = objective_multiplier * (-400*x);
        
        // ∂²L/∂y²
        hessian_values[2] = objective_multiplier * 200 + lambda1 * 2;
    }
    
    void compute_hessian_vector_product(const double* x, const double* v, double objective_multiplier,
                                       const uno::Vector<double>& multipliers, double* result) const override {
        // Hessian-向量乘积（用于某些求解器）
        // 这里简化实现
        result[0] = v[0];
        result[1] = v[1];
    }
    
    // ====== 10. 约束分类 ======
    const uno::Collection<size_t>& get_equality_constraints() const override {
        static uno::Range eq_constraints(1, 2);  // 第二个约束是等式 [1, 2)
        return eq_constraints;
    }
    
    const uno::Collection<size_t>& get_inequality_constraints() const override {
        static uno::Range ineq_constraints(0, 1);  // 第一个约束是不等式 [0, 1)
        return ineq_constraints;
    }
    
    const uno::Collection<size_t>& get_linear_constraints() const override {
        static uno::Range lin_constraints(1, 2);  // 第二个约束是线性的
        return lin_constraints;
    }
    
    // ====== 11. 其他必需接口 ======
    const uno::SparseVector<size_t>& get_slacks() const override { return slacks; }
    const uno::Vector<size_t>& get_fixed_variables() const override { return fixed_variables; }
    
    size_t number_jacobian_nonzeros() const override { return 4; }  // 2×2 矩阵
    size_t number_hessian_nonzeros() const override { return 3; }   // 上三角：3个元素
    
    void postprocess_solution(uno::Iterate& iterate, uno::IterateStatus status) const override {
        std::cout << "\n求解完成！\n";
    }
};

// ========================================
//              主程序
// ========================================
int main() {
    try {
        std::cout << "===== Rosenbrock 函数优化教程 =====\n\n";
        
        // 步骤 1：创建问题实例
        RosenbrockProblem problem;
        
        // 步骤 2：设置求解器选项
        uno::Options options;
        uno::DefaultOptions::load(options);           // 加载默认选项
        uno::Presets::set(options, "filtersqp");     // 使用 filtersqp 算法
        
        // 可以自定义选项
        options.set("print_level", "3");              // 详细输出
        options.set("tolerance", "1e-6");             // 收敛容差
        options.set("max_iterations", "100");         // 最大迭代次数
        
        // 步骤 3：创建初始迭代点
        uno::Iterate initial_iterate(problem.number_variables, problem.number_constraints);
        problem.initial_primal_point(initial_iterate.primals);
        problem.initial_dual_point(initial_iterate.multipliers.constraints);
        
        // 步骤 4：创建求解器并求解
        uno::Uno solver;
        auto result = solver.solve(problem, initial_iterate, options);
        
        // 步骤 5：输出结果
        std::cout << "\n========== 优化结果 ==========\n";
        std::cout << "状态: ";
        switch(result.optimization_status) {
            case uno::OptimizationStatus::SUCCESS:
                std::cout << "成功\n";
                break;
            case uno::OptimizationStatus::ITERATION_LIMIT:
                std::cout << "达到迭代限制\n";
                break;
            case uno::OptimizationStatus::TIME_LIMIT:
                std::cout << "达到时间限制\n";
                break;
            default:
                std::cout << "其他状态\n";
        }
        
        std::cout << "迭代次数: " << result.iteration << "\n";
        std::cout << "CPU时间: " << result.cpu_time << " 秒\n";
        
        // 获取最优解
        const auto& x_opt = result.solution.primals;
        std::cout << "\n最优解:\n";
        std::cout << "  x = " << x_opt[0] << "\n";
        std::cout << "  y = " << x_opt[1] << "\n";
        
        // 计算最优值
        double f_opt = problem.evaluate_objective(x_opt);
        std::cout << "\n目标函数值: f(x,y) = " << f_opt << "\n";
        
        // 验证约束
        std::vector<double> constraints(2);
        problem.evaluate_constraints(x_opt, constraints);
        std::cout << "\n约束值:\n";
        std::cout << "  c1: x² + y² - 2 = " << constraints[0] << " (应该 ≤ 0)\n";
        std::cout << "  c2: x + y - 1 = " << constraints[1] << " (应该 = 0)\n";
        
        // 输出对偶变量（拉格朗日乘子）
        const auto& multipliers = result.solution.multipliers.constraints;
        std::cout << "\n拉格朗日乘子:\n";
        std::cout << "  λ1 = " << multipliers[0] << "\n";
        std::cout << "  λ2 = " << multipliers[1] << "\n";
        
        std::cout << "\n===== 教程完成 =====\n";
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}