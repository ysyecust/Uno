// minimal_test.cpp
// 最小化的测试程序 - 验证C++接口调用外部函数

#include <iostream>
#include <cmath>
#include "Uno.hpp"
#include "options/Options.hpp"
#include "options/DefaultOptions.hpp"
#include "options/Presets.hpp"
#include "model/Model.hpp"
#include "optimization/Iterate.hpp"
#include "linear_algebra/Vector.hpp"
#include "linear_algebra/SparseVector.hpp"
#include "symbolic/Range.hpp"

// 极简的测试模型
class MinimalModel : public uno::Model {
private:
    uno::Vector<size_t> fixed_vars;
    uno::SparseVector<size_t> slacks;
    
    // 外部函数示例
    double external_function(double x) const {
        // 这里可以调用任何外部库
        return std::exp(-x * x);  // 示例：高斯函数
    }
    
public:
    MinimalModel() 
        : Model("Minimal", 2, 1, 1.0),  // 2变量, 1约束
          fixed_vars(0),
          slacks(0) {
        std::cout << "模型初始化\n";
    }
    
    // 必需的接口
    bool has_jacobian_operator() const override { return false; }
    bool has_jacobian_transposed_operator() const override { return false; }
    bool has_hessian_operator() const override { return false; }
    bool has_hessian_matrix() const override { return true; }
    
    // 目标函数 - 调用外部函数
    double evaluate_objective(const uno::Vector<double>& x) const override {
        // 调用外部函数
        double f1 = external_function(x[0]);
        double f2 = external_function(x[1]);
        return -f1 * f2;  // 最大化乘积
    }
    
    // 约束: x[0] + x[1] = 1
    void evaluate_constraints(const uno::Vector<double>& x, std::vector<double>& c) const override {
        c[0] = x[0] + x[1] - 1.0;
    }
    
    // 梯度
    void evaluate_objective_gradient(const uno::Vector<double>& x, uno::Vector<double>& g) const override {
        const double h = 1e-8;
        for (size_t i = 0; i < 2; ++i) {
            uno::Vector<double> xp = x, xm = x;
            xp[i] += h;
            xm[i] -= h;
            g[i] = (evaluate_objective(xp) - evaluate_objective(xm)) / (2*h);
        }
    }
    
    // 边界
    double variable_lower_bound(size_t) const override { return -10.0; }
    double variable_upper_bound(size_t) const override { return 10.0; }
    double constraint_lower_bound(size_t) const override { return 0.0; }
    double constraint_upper_bound(size_t) const override { return 0.0; }
    
    // 初始点
    void initial_primal_point(uno::Vector<double>& x) const override {
        x[0] = 0.3;
        x[1] = 0.7;
    }
    
    void initial_dual_point(uno::Vector<double>& y) const override {
        y[0] = 0.0;
    }
    
    // 稀疏结构
    void compute_constraint_jacobian_sparsity(int* rows, int* cols, int idx, uno::MatrixOrder) const override {
        rows[0] = idx; cols[0] = idx;
        rows[1] = idx; cols[1] = 1 + idx;
    }
    
    void compute_hessian_sparsity(int* rows, int* cols, int idx) const override {
        rows[0] = idx; cols[0] = idx;
        rows[1] = idx; cols[1] = 1 + idx;
        rows[2] = 1 + idx; cols[2] = 1 + idx;
    }
    
    void evaluate_constraint_jacobian(const uno::Vector<double>&, double* jac) const override {
        jac[0] = 1.0;
        jac[1] = 1.0;
    }
    
    void evaluate_lagrangian_hessian(const uno::Vector<double>&, double, const uno::Vector<double>&, double* hess) const override {
        hess[0] = 1.0;
        hess[1] = 0.0;
        hess[2] = 1.0;
    }
    
    void compute_hessian_vector_product(const double*, const double* v, double, const uno::Vector<double>&, double* res) const override {
        res[0] = v[0];
        res[1] = v[1];
    }
    
    // 集合
    const uno::SparseVector<size_t>& get_slacks() const override { return slacks; }
    const uno::Vector<size_t>& get_fixed_variables() const override { return fixed_vars; }
    const uno::Collection<size_t>& get_equality_constraints() const override { 
        static uno::Range r(0, 1);
        return r;
    }
    const uno::Collection<size_t>& get_inequality_constraints() const override { 
        static uno::Range r(0, 0);
        return r;
    }
    const uno::Collection<size_t>& get_linear_constraints() const override { 
        static uno::Range r(0, 1);
        return r;
    }
    
    size_t number_jacobian_nonzeros() const override { return 2; }
    size_t number_hessian_nonzeros() const override { return 3; }
    
    void postprocess_solution(uno::Iterate&, uno::IterateStatus) const override {
        std::cout << "后处理完成\n";
    }
};

int main() {
    try {
        std::cout << "===== C++ 接口测试：调用外部函数 =====\n\n";
        
        // 1. 创建模型
        MinimalModel model;
        
        // 2. 设置选项
        uno::Options options;
        uno::DefaultOptions::load(options);  // 加载默认选项
        uno::Presets::set(options, "filtersqp");  // 使用 filtersqp 预设
        options.set("print_level", "2");
        
        // 3. 创建初始点
        uno::Iterate iterate(2, 1);
        model.initial_primal_point(iterate.primals);
        model.initial_dual_point(iterate.multipliers.constraints);
        
        // 4. 求解
        std::cout << "\n开始求解...\n";
        uno::Uno solver;
        auto result = solver.solve(model, iterate, options);
        
        // 5. 输出结果
        std::cout << "\n===== 求解完成 =====\n";
        std::cout << "优化状态: " << static_cast<int>(result.optimization_status) << "\n";
        std::cout << "迭代次数: " << result.iteration << "\n";
        
        const auto& x = result.solution.primals;
        std::cout << "\n最优解:\n";
        std::cout << "  x[0] = " << x[0] << "\n";
        std::cout << "  x[1] = " << x[1] << "\n";
        
        double obj = model.evaluate_objective(x);
        std::cout << "\n目标函数值: " << obj << "\n";
        
        // 验证外部函数
        std::cout << "\n外部函数值:\n";
        std::cout << "  exp(-x[0]^2) = " << std::exp(-x[0]*x[0]) << "\n";
        std::cout << "  exp(-x[1]^2) = " << std::exp(-x[1]*x[1]) << "\n";
        
        std::cout << "\n✅ 成功通过C++接口调用外部函数!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}