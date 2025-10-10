# UNO C++ API 完整指南 - 不使用 AMPL 直接编写优化问题

## 概述

本指南展示如何通过纯 C++ 代码定义和求解优化问题，无需 AMPL 或 `.nl` 文件。这种方式的优势：
- ✅ 可以调用任何外部库（CoolProp, Cantera, 自定义函数等）
- ✅ 完全控制问题定义
- ✅ 动态生成问题（问题维度可以是运行时决定）
- ✅ 更容易集成到现有 C++ 项目

## 快速开始

### 最小可运行示例

```cpp
#include "model/Model.hpp"
#include "Uno.hpp"
#include "options/DefaultOptions.hpp"
#include "options/Presets.hpp"
#include "optimization/Iterate.hpp"

// 定义问题：最小化 f(x) = x²
class SimpleProblem : public uno::Model {
    uno::Vector<size_t> fixed_vars;
    uno::SparseVector<size_t> slacks;

public:
    SimpleProblem()
        : Model("Simple", 1, 0, 1.0), // 1变量, 0约束, 最小化
          fixed_vars(0), slacks(0) {}

    // 实现必需的虚函数（见下文详解）
    bool has_jacobian_operator() const override { return false; }
    bool has_jacobian_transposed_operator() const override { return false; }
    bool has_hessian_operator() const override { return false; }
    bool has_hessian_matrix() const override { return true; }

    double evaluate_objective(const uno::Vector<double>& x) const override {
        return x[0] * x[0];  // x²
    }

    void evaluate_constraints(const uno::Vector<double>&,
                            std::vector<double>&) const override {}

    void evaluate_objective_gradient(const uno::Vector<double>& x,
                                    uno::Vector<double>& g) const override {
        g[0] = 2 * x[0];  // 2x
    }

    // ... 其他必需的函数（见完整示例）
};

int main() {
    SimpleProblem problem;

    // 配置选项
    uno::Options options;
    uno::DefaultOptions::load(options);
    uno::Presets::set(options, "filtersqp");

    // 创建初始迭代点
    uno::Iterate iterate(1, 0);  // 1变量, 0约束

    // 求解
    uno::Uno solver;
    auto result = solver.solve(problem, iterate, options);

    std::cout << "最优解: x = " << iterate.primals[0] << std::endl;
    std::cout << "最优值: " << result.objective << std::endl;
}
```

## 详细步骤

### 步骤 1: 理解 Model 抽象

UNO 通过 `Model` 基类定义优化问题。你需要继承这个类并实现所有纯虚函数。

#### Model 类的核心信息

```cpp
class Model {
public:
    // 构造函数参数：
    Model(name, n_vars, n_cons, obj_sign);

    // 成员变量：
    const size_t number_variables;      // 变量数量 n
    const size_t number_constraints;    // 约束数量 m
    const double objective_sign;        // 1.0=最小化, -1.0=最大化
};
```

### 步骤 2: 定义你的问题类

#### 2.1 基本结构

```cpp
class MyOptimizationProblem : public uno::Model {
private:
    // 必需的成员变量
    uno::Vector<size_t> fixed_vars;        // 固定变量索引（通常为空）
    uno::SparseVector<size_t> slacks;      // 松弛变量（通常为空）

    // 你的问题特定数据
    std::vector<double> problem_data;

public:
    MyOptimizationProblem(size_t n, size_t m)
        : Model("MyProblem", n, m, 1.0),   // 名字, n变量, m约束, 最小化
          fixed_vars(0),                    // 无固定变量
          slacks(0)                         // 无松弛变量
    {
        // 初始化你的问题数据
    }

    // 实现所有纯虚函数...
};
```

#### 2.2 必需的接口函数

```cpp
// 告诉 UNO 你的问题提供了哪些算子
bool has_jacobian_operator() const override {
    return false;  // 通常为 false（提供稀疏矩阵）
}

bool has_jacobian_transposed_operator() const override {
    return false;  // 通常为 false
}

bool has_hessian_operator() const override {
    return false;  // 通常为 false（提供稀疏矩阵）
}

bool has_hessian_matrix() const override {
    return true;   // 通常为 true（提供 Hessian 矩阵）
}
```

### 步骤 3: 实现函数求值

#### 3.1 目标函数

```cpp
double evaluate_objective(const uno::Vector<double>& x) const override {
    // x[0], x[1], ..., x[n-1] 是当前变量值

    // 示例：Rosenbrock 函数
    double sum = 0.0;
    for (size_t i = 0; i < number_variables - 1; ++i) {
        double term1 = x[i+1] - x[i]*x[i];
        double term2 = 1 - x[i];
        sum += 100*term1*term1 + term2*term2;
    }
    return sum;

    // 或调用外部函数：
    // return my_external_function(x[0], x[1], x[2]);
}
```

#### 3.2 约束函数

```cpp
void evaluate_constraints(const uno::Vector<double>& x,
                        std::vector<double>& constraints) const override {
    // 计算所有约束的值 c(x)

    // 示例：两个约束
    constraints[0] = x[0] + x[1] - 1.0;           // 线性约束
    constraints[1] = x[0]*x[0] + x[1]*x[1] - 4.0; // 非线性约束

    // 注意：UNO 会自动处理不等式 c_L ≤ c(x) ≤ c_U
    // 你只需要计算 c(x) 的值
}
```

#### 3.3 目标函数梯度

```cpp
void evaluate_objective_gradient(const uno::Vector<double>& x,
                                uno::Vector<double>& gradient) const override {
    // gradient[i] = ∂f/∂x_i

    // 方法 1: 解析梯度（推荐）
    gradient[0] = 2*x[0] - 2;
    gradient[1] = 2*x[1] - 4;

    // 方法 2: 数值微分（不推荐，但可用）
    double eps = 1e-8;
    for (size_t i = 0; i < number_variables; ++i) {
        uno::Vector<double> x_plus = x;
        x_plus[i] += eps;
        gradient[i] = (evaluate_objective(x_plus) - evaluate_objective(x)) / eps;
    }
}
```

### 步骤 4: 实现约束 Jacobian

Jacobian 是约束对变量的导数矩阵 J_ij = ∂c_i/∂x_j。

#### 4.1 定义稀疏结构

```cpp
void compute_constraint_jacobian_sparsity(int* row_indices, int* column_indices,
                                         int solver_indexing,
                                         uno::MatrixOrder order) const override {
    // 告诉 UNO 哪些 (i,j) 位置是非零的

    // 示例：2个约束，3个变量
    // c_0 = x_0 + 2*x_1        (稀疏: (0,0), (0,1))
    // c_1 = x_1*x_2            (稀疏: (1,1), (1,2))

    int idx = 0;
    // 约束 0
    row_indices[idx] = 0 + solver_indexing;
    column_indices[idx] = 0 + solver_indexing;
    idx++;

    row_indices[idx] = 0 + solver_indexing;
    column_indices[idx] = 1 + solver_indexing;
    idx++;

    // 约束 1
    row_indices[idx] = 1 + solver_indexing;
    column_indices[idx] = 1 + solver_indexing;
    idx++;

    row_indices[idx] = 1 + solver_indexing;
    column_indices[idx] = 2 + solver_indexing;
    idx++;
}

size_t number_jacobian_nonzeros() const override {
    return 4;  // 上面定义了4个非零元素
}
```

#### 4.2 计算 Jacobian 数值

```cpp
void evaluate_constraint_jacobian(const uno::Vector<double>& x,
                                 double* jacobian_values) const override {
    // 按照 compute_constraint_jacobian_sparsity 定义的顺序填充值

    // 示例：对应上面的稀疏结构
    jacobian_values[0] = 1.0;        // ∂c_0/∂x_0 = 1
    jacobian_values[1] = 2.0;        // ∂c_0/∂x_1 = 2
    jacobian_values[2] = x[2];       // ∂c_1/∂x_1 = x_2
    jacobian_values[3] = x[1];       // ∂c_1/∂x_2 = x_1
}
```

### 步骤 5: 实现 Lagrangian Hessian

Lagrangian 的 Hessian: ∇²L = σ∇²f + Σλ_i∇²c_i

#### 5.1 定义稀疏结构

```cpp
void compute_hessian_sparsity(int* row_indices, int* column_indices,
                             int solver_indexing) const override {
    // 只需要下三角部分（对称矩阵）

    // 示例：3x3 对称矩阵，所有元素非零
    int idx = 0;
    for (size_t i = 0; i < number_variables; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            row_indices[idx] = i + solver_indexing;
            column_indices[idx] = j + solver_indexing;
            idx++;
        }
    }
}

size_t number_hessian_nonzeros() const override {
    // 对于 n 维稠密对称矩阵：n*(n+1)/2
    return number_variables * (number_variables + 1) / 2;
}
```

#### 5.2 计算 Hessian 数值

```cpp
void evaluate_lagrangian_hessian(const uno::Vector<double>& x,
                                 double objective_multiplier,
                                 const uno::Vector<double>& multipliers,
                                 double* hessian_values) const override {
    // objective_multiplier (σ): 通常为 1.0
    // multipliers (λ): 约束的 Lagrange 乘子

    // ∇²L = σ∇²f + Σλ_i∇²c_i

    // 示例：f(x) = x₁² + x₂²，c(x) = x₁ + x₂ - 1
    // ∇²f = [2  0]
    //       [0  2]
    // ∇²c = [0  0]  (线性约束，Hessian为0)
    //       [0  0]

    hessian_values[0] = 2.0 * objective_multiplier;  // ∂²L/∂x₁²
    hessian_values[1] = 0.0;                          // ∂²L/∂x₁∂x₂
    hessian_values[2] = 2.0 * objective_multiplier;  // ∂²L/∂x₂²
}
```

#### 5.3 Hessian-向量乘积（可选）

```cpp
void compute_hessian_vector_product(const double* x, const double* direction,
                                   double objective_multiplier,
                                   const uno::Vector<double>& multipliers,
                                   double* result) const override {
    // result = ∇²L * direction

    // 示例：对角 Hessian
    result[0] = 2.0 * objective_multiplier * direction[0];
    result[1] = 2.0 * objective_multiplier * direction[1];
}
```

### 步骤 6: 定义变量和约束界限

```cpp
// 变量界限: x_L ≤ x ≤ x_U
double variable_lower_bound(size_t variable_index) const override {
    // 示例：所有变量 ≥ 0
    return 0.0;

    // 或不同变量不同界限
    if (variable_index == 0) return 0.0;
    else return -1e20;  // 无下界（使用大负数）
}

double variable_upper_bound(size_t variable_index) const override {
    // 示例：所有变量 ≤ 10
    return 10.0;

    // 无上界：return 1e20;
}

// 约束界限: c_L ≤ c(x) ≤ c_U
double constraint_lower_bound(size_t constraint_index) const override {
    // 等式约束 c(x) = 0: 返回 0.0
    // 不等式约束 c(x) ≥ 0: 返回 0.0
    // 不等式约束 c(x) ≤ 0: 返回 -1e20
    return 0.0;
}

double constraint_upper_bound(size_t constraint_index) const override {
    // 等式约束 c(x) = 0: 返回 0.0
    // 不等式约束 c(x) ≤ 5: 返回 5.0
    // 不等式约束 c(x) ≥ 0: 返回 1e20
    return 0.0;
}
```

### 步骤 7: 约束分类

```cpp
const uno::Collection<size_t>& get_equality_constraints() const override {
    // 返回等式约束的索引
    static uno::Range equality_range(0, 2);  // 约束 0,1 是等式
    return equality_range;
}

const uno::Collection<size_t>& get_inequality_constraints() const override {
    // 返回不等式约束的索引
    static uno::Range inequality_range(2, 5);  // 约束 2,3,4 是不等式
    return inequality_range;
}

const uno::Collection<size_t>& get_linear_constraints() const override {
    // 返回线性约束的索引
    static uno::Range linear_range(0, 1);  // 约束 0 是线性
    return linear_range;
}
```

### 步骤 8: 初始点和辅助函数

```cpp
void initial_primal_point(uno::Vector<double>& x) const override {
    // 设置初始点（可行或不可行都行）
    for (size_t i = 0; i < number_variables; ++i) {
        x[i] = 1.0;  // 或读取用户提供的初始点
    }
}

void initial_dual_point(uno::Vector<double>& multipliers) const override {
    // 设置初始 Lagrange 乘子（通常全0即可）
    for (size_t i = 0; i < number_constraints; ++i) {
        multipliers[i] = 0.0;
    }
}

const uno::SparseVector<size_t>& get_slacks() const override {
    return slacks;  // 通常为空
}

const uno::Vector<size_t>& get_fixed_variables() const override {
    return fixed_vars;  // 通常为空
}

void postprocess_solution(uno::Iterate& iterate,
                         uno::IterateStatus status) const override {
    // 求解完成后的后处理（可选）
    std::cout << "优化完成，状态: " << (int)status << std::endl;
}
```

### 步骤 9: 求解问题

```cpp
int main() {
    // 1. 创建问题实例
    MyOptimizationProblem problem(n_vars, n_constraints);

    // 2. 配置求解器选项
    uno::Options options;
    uno::DefaultOptions::load(options);

    // 选择预设
    uno::Presets::set(options, "filtersqp");  // 或 "ipopt"

    // 或手动设置选项
    options.set("tolerance", "1e-6");
    options.set("max_iterations", "1000");
    options.set("linear_solver", "MUMPS");  // 如果有 MUMPS

    // 3. 创建初始迭代点
    uno::Iterate iterate(n_vars, n_constraints);
    problem.initial_primal_point(iterate.primals);
    problem.initial_dual_point(iterate.multipliers.constraints);

    // 4. 求解
    uno::Uno solver;
    auto result = solver.solve(problem, iterate, options);

    // 5. 输出结果
    std::cout << "优化状态: " << (int)result.status << std::endl;
    std::cout << "目标值: " << result.objective << std::endl;
    std::cout << "迭代次数: " << result.iterations << std::endl;

    std::cout << "最优解:" << std::endl;
    for (size_t i = 0; i < n_vars; ++i) {
        std::cout << "  x[" << i << "] = " << iterate.primals[i] << std::endl;
    }

    return 0;
}
```

## 完整示例

### 示例 1: Rosenbrock 函数（无约束）

```cpp
// min f(x,y) = (1-x)² + 100(y-x²)²
class RosenbrockProblem : public uno::Model {
    uno::Vector<size_t> fixed_vars;
    uno::SparseVector<size_t> slacks;

public:
    RosenbrockProblem()
        : Model("Rosenbrock", 2, 0, 1.0),
          fixed_vars(0), slacks(0) {}

    // 接口函数
    bool has_jacobian_operator() const override { return false; }
    bool has_jacobian_transposed_operator() const override { return false; }
    bool has_hessian_operator() const override { return false; }
    bool has_hessian_matrix() const override { return true; }

    // 目标函数
    double evaluate_objective(const uno::Vector<double>& x) const override {
        double dx = 1 - x[0];
        double dy = x[1] - x[0]*x[0];
        return dx*dx + 100*dy*dy;
    }

    // 无约束
    void evaluate_constraints(const uno::Vector<double>&,
                            std::vector<double>&) const override {}

    // 梯度
    void evaluate_objective_gradient(const uno::Vector<double>& x,
                                    uno::Vector<double>& g) const override {
        g[0] = -2*(1-x[0]) + 400*x[0]*(x[0]*x[0] - x[1]);
        g[1] = 200*(x[1] - x[0]*x[0]);
    }

    // Hessian
    void compute_hessian_sparsity(int* rows, int* cols, int idx) const override {
        rows[0] = idx;     cols[0] = idx;      // (0,0)
        rows[1] = 1+idx;   cols[1] = idx;      // (1,0)
        rows[2] = 1+idx;   cols[2] = 1+idx;    // (1,1)
    }

    void evaluate_lagrangian_hessian(const uno::Vector<double>& x, double σ,
                                    const uno::Vector<double>&,
                                    double* H) const override {
        H[0] = σ * (2 - 400*x[1] + 1200*x[0]*x[0]);  // ∂²L/∂x²
        H[1] = σ * (-400*x[0]);                       // ∂²L/∂x∂y
        H[2] = σ * 200;                               // ∂²L/∂y²
    }

    void compute_hessian_vector_product(const double* x, const double* v, double σ,
                                       const uno::Vector<double>&, double* r) const override {
        r[0] = σ * ((2 - 400*x[1] + 1200*x[0]*x[0])*v[0] - 400*x[0]*v[1]);
        r[1] = σ * (-400*x[0]*v[0] + 200*v[1]);
    }

    // 界限
    double variable_lower_bound(size_t) const override { return -10.0; }
    double variable_upper_bound(size_t) const override { return 10.0; }
    double constraint_lower_bound(size_t) const override { return 0.0; }
    double constraint_upper_bound(size_t) const override { return 0.0; }

    // 初始点
    void initial_primal_point(uno::Vector<double>& x) const override {
        x[0] = -1.2; x[1] = 1.0;
    }
    void initial_dual_point(uno::Vector<double>&) const override {}

    // 约束分类（无约束）
    const uno::Collection<size_t>& get_equality_constraints() const override {
        static uno::Range r(0, 0); return r;
    }
    const uno::Collection<size_t>& get_inequality_constraints() const override {
        static uno::Range r(0, 0); return r;
    }
    const uno::Collection<size_t>& get_linear_constraints() const override {
        static uno::Range r(0, 0); return r;
    }

    const uno::SparseVector<size_t>& get_slacks() const override { return slacks; }
    const uno::Vector<size_t>& get_fixed_variables() const override { return fixed_vars; }
    size_t number_jacobian_nonzeros() const override { return 0; }
    size_t number_hessian_nonzeros() const override { return 3; }
    void postprocess_solution(uno::Iterate&, uno::IterateStatus) const override {}
};
```

### 示例 2: 调用外部热力学库

```cpp
// 假设你有 CoolProp 库
#include "CoolProp.h"

class ThermodynamicOptimization : public uno::Model {
    uno::Vector<size_t> fixed_vars;
    uno::SparseVector<size_t> slacks;

    // 调用外部库
    double fugacity(double T, double P, const std::vector<double>& z) const {
        // 使用 CoolProp 计算逸度
        return CoolProp::PropsSI("fugacity", "T", T, "P", P,
                                "mole_fractions", z, "mixture");
    }

public:
    ThermodynamicOptimization(size_t n_components)
        : Model("Thermo", n_components + 2, 1, 1.0),  // T, P, z[0..n-1]
          fixed_vars(0), slacks(0) {}

    double evaluate_objective(const uno::Vector<double>& x) const override {
        double T = x[0];
        double P = x[1];
        std::vector<double> z(number_variables - 2);
        for (size_t i = 0; i < z.size(); ++i) {
            z[i] = x[i + 2];
        }

        // 目标：最小化逸度差
        double fug = fugacity(T, P, z);
        return fug * fug;  // 或其他目标函数
    }

    void evaluate_constraints(const uno::Vector<double>& x,
                            std::vector<double>& c) const override {
        // 约束：摩尔分数和为1
        double sum = 0.0;
        for (size_t i = 2; i < number_variables; ++i) {
            sum += x[i];
        }
        c[0] = sum - 1.0;
    }

    // 数值微分梯度（如果解析导数难以计算）
    void evaluate_objective_gradient(const uno::Vector<double>& x,
                                    uno::Vector<double>& g) const override {
        double f0 = evaluate_objective(x);
        double eps = 1e-6;

        for (size_t i = 0; i < number_variables; ++i) {
            uno::Vector<double> x_plus = x;
            x_plus[i] += eps;
            g[i] = (evaluate_objective(x_plus) - f0) / eps;
        }
    }

    // ... 其他必需函数的实现
};
```

## 编译和运行

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.10)
project(MyOptimization)

set(CMAKE_CXX_STANDARD 17)

# UNO 路径
set(UNO_DIR /path/to/Uno)

# 包含头文件
include_directories(${UNO_DIR})

# 链接库
link_directories(${UNO_DIR}/build)

add_executable(my_optimization my_problem.cpp)

target_link_libraries(my_optimization
    ${UNO_DIR}/build/libuno.a
    /opt/homebrew/lib/libhighs.dylib  # 如果使用 HiGHS
    # /path/to/libdmumps.a            # 如果使用 MUMPS
)
```

### 编译命令

```bash
mkdir build && cd build
cmake ..
make
./my_optimization
```

## 高级技巧

### 1. 动态维度问题

```cpp
class DynamicProblem : public uno::Model {
    std::vector<double> data;

public:
    DynamicProblem(size_t n, const std::vector<double>& input_data)
        : Model("Dynamic", n, 0, 1.0),
          fixed_vars(0), slacks(0), data(input_data) {
        // 问题维度由运行时决定
    }

    // ... 实现函数
};

int main() {
    size_t n = read_problem_size();  // 运行时读取
    std::vector<double> data = load_data();
    DynamicProblem problem(n, data);
    // ... 求解
}
```

### 2. 稀疏问题优化

对于大规模稀疏问题，只返回非零元素的稀疏结构：

```cpp
size_t number_hessian_nonzeros() const override {
    return 1000;  // 而不是 n*(n+1)/2
}

void compute_hessian_sparsity(int* rows, int* cols, int idx) const override {
    // 只返回实际非零的元素位置
    int k = 0;
    for (auto [i, j] : nonzero_pattern) {
        rows[k] = i + idx;
        cols[k] = j + idx;
        k++;
    }
}
```

### 3. 并行函数求值

如果你的函数求值可以并行化，可以在 `evaluate_objective` 等函数内使用 OpenMP：

```cpp
double evaluate_objective(const uno::Vector<double>& x) const override {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < 10000; ++i) {
        sum += expensive_computation(x, i);
    }
    return sum;
}
```

## 常见问题

### Q: 我的梯度/Hessian计算很复杂，能用数值微分吗？

A: 可以，但不推荐。数值微分慢且不精确。如果实在无法解析计算，考虑使用自动微分库（如 CppAD, ADOL-C）。

### Q: 能否在运行时改变问题维度？

A: 不能。Model 的维度在构造时固定。如果需要动态维度，每次创建新的 Model 实例。

### Q: 如何调试我的问题？

A:
1. 设置 `options.set("logger", "DEBUG")` 查看详细输出
2. 检查梯度/Hessian 正确性（与数值微分对比）
3. 从简单问题开始，逐步增加复杂度

### Q: 性能优化建议？

A:
1. 使用稀疏格式（避免稠密矩阵）
2. 提前计算常数项
3. 使用解析导数而非数值微分
4. 选择合适的线性求解器（MUMPS for large, HiGHS for QP）

## 总结

通过继承 `uno::Model` 类并实现纯虚函数，你可以：
- ✅ 定义任意复杂的优化问题
- ✅ 调用任何外部库和函数
- ✅ 完全控制问题的每个细节
- ✅ 无需学习 AMPL 语言

关键是理解 Model 接口的每个函数的含义，并正确实现稀疏结构和数值求值。

参考代码：
- `/Users/shaoyiyang/Documents/Code/Uno/examples/thermodynamic_optimization/simple_tutorial.cpp`
- `/Users/shaoyiyang/Documents/Code/Uno/examples/thermodynamic_optimization/minimal_test.cpp`
