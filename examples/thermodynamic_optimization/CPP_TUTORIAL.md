# C++ 接口使用教程 - 编写和求解优化问题

## 快速开始

### 最小示例
```cpp
#include "model/Model.hpp"
#include "Uno.hpp"

class MyProblem : public uno::Model {
    // 实现必需的虚函数...
};

int main() {
    MyProblem problem;
    uno::Options options;
    uno::DefaultOptions::load(options);
    uno::Presets::set(options, "filtersqp");
    
    uno::Iterate iterate(n_vars, n_constraints);
    uno::Uno solver;
    auto result = solver.solve(problem, iterate, options);
}
```

## 步骤详解

### 1. 继承 Model 类

```cpp
class MyOptimizationProblem : public uno::Model {
private:
    uno::Vector<size_t> fixed_vars;
    uno::SparseVector<size_t> slacks;
    
public:
    MyOptimizationProblem() 
        : Model("名称", n_variables, n_constraints, 1.0),  // 1.0=最小化
          fixed_vars(0),
          slacks(0) 
    {}
};
```

### 2. 实现必需的虚函数

#### 2.1 基本接口
```cpp
bool has_jacobian_operator() const override { return false; }
bool has_jacobian_transposed_operator() const override { return false; }
bool has_hessian_operator() const override { return false; }
bool has_hessian_matrix() const override { return true; }
```

#### 2.2 目标函数
```cpp
double evaluate_objective(const uno::Vector<double>& x) const override {
    // 可以调用任何外部函数！
    double result = external_function(x[0], x[1]);
    return result;
}
```

#### 2.3 约束函数
```cpp
void evaluate_constraints(const uno::Vector<double>& x, 
                        std::vector<double>& c) const override {
    c[0] = g1(x);  // 第一个约束
    c[1] = g2(x);  // 第二个约束
}
```

#### 2.4 梯度
```cpp
void evaluate_objective_gradient(const uno::Vector<double>& x, 
                                uno::Vector<double>& grad) const override {
    // 解析梯度或数值微分
    grad[0] = ∂f/∂x1;
    grad[1] = ∂f/∂x2;
}
```

#### 2.5 变量和约束界限
```cpp
double variable_lower_bound(size_t i) const override { return lb[i]; }
double variable_upper_bound(size_t i) const override { return ub[i]; }
double constraint_lower_bound(size_t i) const override { return cl[i]; }
double constraint_upper_bound(size_t i) const override { return cu[i]; }
```

### 3. 实现稀疏结构

#### Jacobian 稀疏模式
```cpp
void compute_constraint_jacobian_sparsity(int* rows, int* cols, 
                                         int idx, uno::MatrixOrder order) const override {
    // 定义哪些 (i,j) 位置非零
}

void evaluate_constraint_jacobian(const uno::Vector<double>& x, 
                                 double* jac_values) const override {
    // 填充非零值
}
```

#### Hessian 稀疏模式
```cpp
void compute_hessian_sparsity(int* rows, int* cols, int idx) const override {
    // 只存上三角部分
}

void evaluate_lagrangian_hessian(const uno::Vector<double>& x, double obj_mult,
                                const uno::Vector<double>& mult, 
                                double* hess_values) const override {
    // Lagrangian 的 Hessian
}
```

### 4. 约束分类

```cpp
const uno::Collection<size_t>& get_equality_constraints() const override {
    static uno::Range eq(0, n_eq);  // [0, n_eq) 是等式约束
    return eq;
}

const uno::Collection<size_t>& get_inequality_constraints() const override {
    static uno::Range ineq(n_eq, n_total);  // [n_eq, n_total) 是不等式
    return ineq;
}
```

## 完整示例

### 示例1：二次规划
```cpp
// 最小化: f(x,y) = x² + y² - 2x - 4y
// 约束: x + y = 2, x≥0, y≥0

class SimpleQPProblem : public uno::Model {
    // 见 simple_tutorial.cpp
};
```

### 示例2：调用外部函数
```cpp
class ExternalFunctionProblem : public uno::Model {
private:
    // 外部函数（可以是任何库）
    double my_external_function(double x, double y) const {
        // CoolProp, Cantera, 自定义函数等
        return std::sin(x) * std::cos(y);
    }
    
    double evaluate_objective(const uno::Vector<double>& v) const override {
        return my_external_function(v[0], v[1]);
    }
};
```

## 求解器选项

### 基本配置
```cpp
uno::Options options;
uno::DefaultOptions::load(options);     // 必须！
uno::Presets::set(options, "filtersqp"); // 选择算法预设
```

### 可用预设
- `"filtersqp"` - Fletcher's filter SQP（推荐）
- `"ipopt"` - Interior point 方法
- `"byrd"` - Byrd's watchdog

### 自定义选项
```cpp
options.set("print_level", "3");        // 0-5，输出详细度
options.set("tolerance", "1e-6");       // 收敛容差
options.set("max_iterations", "100");   // 最大迭代
options.set("time_limit", "3600");      // 时间限制（秒）
```

## 运行和结果

```cpp
// 创建初始点
uno::Iterate iterate(n_variables, n_constraints);
problem.initial_primal_point(iterate.primals);
problem.initial_dual_point(iterate.multipliers.constraints);

// 求解
uno::Uno solver;
auto result = solver.solve(problem, iterate, options);

// 检查结果
if (result.optimization_status == uno::OptimizationStatus::SUCCESS) {
    const auto& x = result.solution.primals;
    std::cout << "最优解: " << x[0] << ", " << x[1] << "\n";
    std::cout << "目标值: " << problem.evaluate_objective(x) << "\n";
}
```

## 编译

### CMakeLists.txt
```cmake
add_executable(my_optimizer my_optimizer.cpp)
target_link_libraries(my_optimizer 
    ${UNO_BUILD_DIR}/libuno.a
    /opt/homebrew/Cellar/highs/1.11.0/lib/libhighs.dylib
)
```

### 命令行
```bash
cmake --build . --target my_optimizer
./my_optimizer
```

## 常见问题

### Q: 如何使用数值微分？
```cpp
void evaluate_objective_gradient(const uno::Vector<double>& x, 
                                uno::Vector<double>& g) const override {
    const double h = 1e-8;
    for (size_t i = 0; i < n; ++i) {
        uno::Vector<double> xp = x, xm = x;
        xp[i] += h; xm[i] -= h;
        g[i] = (evaluate_objective(xp) - evaluate_objective(xm)) / (2*h);
    }
}
```

### Q: 如何处理无约束问题？
```cpp
MyProblem() : Model("Unconstrained", n_vars, 0, 1.0) {}
// 不需要实现 evaluate_constraints
```

### Q: 如何集成实际的外部库？
```cpp
#include "CoolProp.h"  // 热力学库

double evaluate_objective(const uno::Vector<double>& x) const override {
    double T = x[0], P = x[1];
    double density = CoolProp::PropsSI("D", "T", T, "P", P, "Water");
    return calculate_energy(density, x);
}
```

## 可运行示例

1. **minimal_test.cpp** - 最简单的外部函数调用
2. **simple_tutorial.cpp** - 包含两个完整示例
3. **tutorial_example.cpp** - Rosenbrock 函数优化

编译并运行：
```bash
./simple_tutorial 1  # 二次规划
./simple_tutorial 2  # 外部函数
./minimal_test       # 最小示例
```

---
*更新：2025-09-15*