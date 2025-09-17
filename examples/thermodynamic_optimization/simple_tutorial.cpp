// simple_tutorial.cpp
// 简化版教程：如何在 C++ 中编写和求解优化问题

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
// 简单示例：二次规划问题
// 
// 最小化: f(x,y) = x² + y² - 2x - 4y
// 约束:   x + y = 2
//         x ≥ 0, y ≥ 0
// ========================================

class SimpleQPProblem : public uno::Model {
private:
    uno::Vector<size_t> fixed_vars;
    uno::SparseVector<size_t> slacks;
    
public:
    SimpleQPProblem() 
        : Model("SimpleQP", 2, 1, 1.0),  // 2变量, 1约束, 最小化
          fixed_vars(0),
          slacks(0)
    {
        std::cout << "创建简单二次规划问题\n";
    }
    
    // ====== 必需的接口 ======
    bool has_jacobian_operator() const override { return false; }
    bool has_jacobian_transposed_operator() const override { return false; }
    bool has_hessian_operator() const override { return false; }
    bool has_hessian_matrix() const override { return true; }
    
    // ====== 目标函数: f(x,y) = x² + y² - 2x - 4y ======
    double evaluate_objective(const uno::Vector<double>& v) const override {
        double x = v[0];
        double y = v[1];
        return x*x + y*y - 2*x - 4*y;
    }
    
    // ====== 约束: x + y - 2 = 0 ======
    void evaluate_constraints(const uno::Vector<double>& v, 
                            std::vector<double>& c) const override {
        c[0] = v[0] + v[1] - 2.0;
    }
    
    // ====== 目标函数梯度 ======
    void evaluate_objective_gradient(const uno::Vector<double>& v, 
                                    uno::Vector<double>& g) const override {
        g[0] = 2*v[0] - 2;  // ∂f/∂x = 2x - 2
        g[1] = 2*v[1] - 4;  // ∂f/∂y = 2y - 4
    }
    
    // ====== 变量界限 ======
    double variable_lower_bound(size_t i) const override { 
        return 0.0;  // x,y ≥ 0
    }
    double variable_upper_bound(size_t i) const override { 
        return 10.0;  // 上界
    }
    
    // ====== 约束界限（等式约束）======
    double constraint_lower_bound(size_t i) const override { return 0.0; }
    double constraint_upper_bound(size_t i) const override { return 0.0; }
    
    // ====== 初始点 ======
    void initial_primal_point(uno::Vector<double>& x) const override {
        x[0] = 0.5;
        x[1] = 0.5;
    }
    
    void initial_dual_point(uno::Vector<double>& y) const override {
        y[0] = 0.0;
    }
    
    // ====== Jacobian（约束梯度）======
    void compute_constraint_jacobian_sparsity(int* rows, int* cols, 
                                             int idx, uno::MatrixOrder order) const override {
        rows[0] = idx;
        cols[0] = idx;
        rows[1] = idx;
        cols[1] = 1 + idx;
    }
    
    void evaluate_constraint_jacobian(const uno::Vector<double>& v, 
                                     double* jac) const override {
        jac[0] = 1.0;  // ∂c/∂x = 1
        jac[1] = 1.0;  // ∂c/∂y = 1
    }
    
    // ====== Hessian ======
    void compute_hessian_sparsity(int* rows, int* cols, int idx) const override {
        // 对角矩阵
        rows[0] = idx;     cols[0] = idx;      // (0,0)
        rows[1] = idx;     cols[1] = 1 + idx;  // (0,1)
        rows[2] = 1 + idx; cols[2] = 1 + idx;  // (1,1)
    }
    
    void evaluate_lagrangian_hessian(const uno::Vector<double>& v, double obj_mult,
                                    const uno::Vector<double>& mult, 
                                    double* hess) const override {
        // Hessian是常数矩阵（二次函数）
        hess[0] = 2.0 * obj_mult;  // ∂²f/∂x²
        hess[1] = 0.0;              // ∂²f/∂x∂y
        hess[2] = 2.0 * obj_mult;  // ∂²f/∂y²
    }
    
    void compute_hessian_vector_product(const double* x, const double* v, double obj_mult,
                                       const uno::Vector<double>& mult, double* res) const override {
        res[0] = 2.0 * obj_mult * v[0];
        res[1] = 2.0 * obj_mult * v[1];
    }
    
    // ====== 约束分类 ======
    const uno::Collection<size_t>& get_equality_constraints() const override {
        static uno::Range eq(0, 1);
        return eq;
    }
    
    const uno::Collection<size_t>& get_inequality_constraints() const override {
        static uno::Range ineq(0, 0);
        return ineq;
    }
    
    const uno::Collection<size_t>& get_linear_constraints() const override {
        static uno::Range lin(0, 1);
        return lin;
    }
    
    // ====== 其他接口 ======
    const uno::SparseVector<size_t>& get_slacks() const override { return slacks; }
    const uno::Vector<size_t>& get_fixed_variables() const override { return fixed_vars; }
    size_t number_jacobian_nonzeros() const override { return 2; }
    size_t number_hessian_nonzeros() const override { return 3; }
    
    void postprocess_solution(uno::Iterate& it, uno::IterateStatus st) const override {
        std::cout << "后处理完成\n";
    }
};

// ========================================
//           使用外部函数的示例
// ========================================
class ExternalFunctionProblem : public uno::Model {
private:
    uno::Vector<size_t> fixed_vars;
    uno::SparseVector<size_t> slacks;
    
    // 外部函数示例（可以是任何库）
    double my_external_function(double x, double y) const {
        // 这里可以调用：
        // - CoolProp 热力学库
        // - Cantera 化学反应库
        // - 自定义 C/C++ 函数
        // - Python C API
        return std::sin(x) * std::cos(y);
    }
    
public:
    ExternalFunctionProblem() 
        : Model("ExternalFunc", 2, 0, 1.0),  // 2变量, 无约束
          fixed_vars(0),
          slacks(0)
    {
        std::cout << "创建包含外部函数的优化问题\n";
    }
    
    // 基本接口（同上）
    bool has_jacobian_operator() const override { return false; }
    bool has_jacobian_transposed_operator() const override { return false; }
    bool has_hessian_operator() const override { return false; }
    bool has_hessian_matrix() const override { return true; }
    
    // 目标函数调用外部函数
    double evaluate_objective(const uno::Vector<double>& v) const override {
        double x = v[0];
        double y = v[1];
        
        // 调用外部函数！
        double external_value = my_external_function(x, y);
        
        // 使用外部函数值构建目标函数
        return (x - 1)*(x - 1) + (y - 2)*(y - 2) + external_value;
    }
    
    // 无约束
    void evaluate_constraints(const uno::Vector<double>& v, 
                            std::vector<double>& c) const override {
        // 无约束
    }
    
    // 梯度（数值微分）
    void evaluate_objective_gradient(const uno::Vector<double>& v, 
                                    uno::Vector<double>& g) const override {
        const double h = 1e-8;
        for (size_t i = 0; i < 2; ++i) {
            uno::Vector<double> vp = v, vm = v;
            vp[i] += h;
            vm[i] -= h;
            g[i] = (evaluate_objective(vp) - evaluate_objective(vm)) / (2*h);
        }
    }
    
    // 其他必需接口（简化实现）
    double variable_lower_bound(size_t i) const override { return -5.0; }
    double variable_upper_bound(size_t i) const override { return 5.0; }
    double constraint_lower_bound(size_t i) const override { return 0.0; }
    double constraint_upper_bound(size_t i) const override { return 0.0; }
    
    void initial_primal_point(uno::Vector<double>& x) const override {
        x[0] = 0.0;
        x[1] = 0.0;
    }
    
    void initial_dual_point(uno::Vector<double>& y) const override {}
    
    void compute_constraint_jacobian_sparsity(int* rows, int* cols, 
                                             int idx, uno::MatrixOrder order) const override {}
    
    void evaluate_constraint_jacobian(const uno::Vector<double>& v, 
                                     double* jac) const override {}
    
    void compute_hessian_sparsity(int* rows, int* cols, int idx) const override {
        rows[0] = idx;     cols[0] = idx;
        rows[1] = idx;     cols[1] = 1 + idx;
        rows[2] = 1 + idx; cols[2] = 1 + idx;
    }
    
    void evaluate_lagrangian_hessian(const uno::Vector<double>& v, double obj_mult,
                                    const uno::Vector<double>& mult, 
                                    double* hess) const override {
        // 单位矩阵近似
        hess[0] = 1.0;
        hess[1] = 0.0;
        hess[2] = 1.0;
    }
    
    void compute_hessian_vector_product(const double* x, const double* v, double obj_mult,
                                       const uno::Vector<double>& mult, double* res) const override {
        res[0] = v[0];
        res[1] = v[1];
    }
    
    const uno::Collection<size_t>& get_equality_constraints() const override {
        static uno::Range eq(0, 0);
        return eq;
    }
    
    const uno::Collection<size_t>& get_inequality_constraints() const override {
        static uno::Range ineq(0, 0);
        return ineq;
    }
    
    const uno::Collection<size_t>& get_linear_constraints() const override {
        static uno::Range lin(0, 0);
        return lin;
    }
    
    const uno::SparseVector<size_t>& get_slacks() const override { return slacks; }
    const uno::Vector<size_t>& get_fixed_variables() const override { return fixed_vars; }
    size_t number_jacobian_nonzeros() const override { return 0; }
    size_t number_hessian_nonzeros() const override { return 3; }
    
    void postprocess_solution(uno::Iterate& it, uno::IterateStatus st) const override {}
};

// ========================================
//              主程序
// ========================================
int main(int argc, char* argv[]) {
    try {
        std::cout << "\n===== C++ 优化问题求解教程 =====\n\n";
        
        // 选择要运行的示例
        int example = 1;
        if (argc > 1) {
            example = std::atoi(argv[1]);
        }
        
        if (example == 1) {
            std::cout << "示例 1: 简单二次规划\n";
            std::cout << "最小化: f(x,y) = x² + y² - 2x - 4y\n";
            std::cout << "约束: x + y = 2, x≥0, y≥0\n\n";
            
            // 创建问题
            SimpleQPProblem problem;
            
            // 设置选项
            uno::Options options;
            uno::DefaultOptions::load(options);
            uno::Presets::set(options, "filtersqp");
            options.set("print_level", "2");
            
            // 初始点
            uno::Iterate iterate(2, 1);
            problem.initial_primal_point(iterate.primals);
            problem.initial_dual_point(iterate.multipliers.constraints);
            
            // 求解
            uno::Uno solver;
            auto result = solver.solve(problem, iterate, options);
            
            // 输出结果
            std::cout << "\n=== 结果 ===\n";
            const auto& x = result.solution.primals;
            std::cout << "最优解: x = " << x[0] << ", y = " << x[1] << "\n";
            std::cout << "目标值: " << problem.evaluate_objective(x) << "\n";
            std::cout << "迭代次数: " << result.iteration << "\n";
            
            // 验证 KKT 条件
            std::cout << "\n理论最优解: x = 0, y = 2\n";
            std::cout << "理论最优值: f = -4\n";
            
        } else if (example == 2) {
            std::cout << "示例 2: 调用外部函数\n\n";
            
            // 创建包含外部函数的问题
            ExternalFunctionProblem problem;
            
            // 设置选项
            uno::Options options;
            uno::DefaultOptions::load(options);
            uno::Presets::set(options, "filtersqp");
            options.set("print_level", "2");
            
            // 初始点
            uno::Iterate iterate(2, 0);
            problem.initial_primal_point(iterate.primals);
            
            // 求解
            uno::Uno solver;
            auto result = solver.solve(problem, iterate, options);
            
            // 输出结果
            std::cout << "\n=== 结果 ===\n";
            const auto& x = result.solution.primals;
            std::cout << "最优解: x = " << x[0] << ", y = " << x[1] << "\n";
            std::cout << "目标值: " << problem.evaluate_objective(x) << "\n";
            std::cout << "外部函数值: sin(" << x[0] << ")*cos(" << x[1] << ") = " 
                      << std::sin(x[0])*std::cos(x[1]) << "\n";
        }
        
        std::cout << "\n===== 教程完成 =====\n";
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}