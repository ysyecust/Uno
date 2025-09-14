# Uno C++ 接口使用指南：集成外部函数

## 概述

由于 uno_ampl 不支持 AMPL 外部函数，推荐使用 C++ 接口直接调用外部函数。

## 核心步骤

### 1. 继承 Model 类

```cpp
#include "model/Model.hpp"

class MyModel : public uno::Model {
public:
    MyModel() : Model("ModelName", n_vars, n_constraints, 1.0) {}
    
    // 必须实现的纯虚函数
    double evaluate_objective(const Vector<double>& x) const override;
    void evaluate_constraints(const Vector<double>& x, std::vector<double>& c) const override;
    // ... 其他虚函数
};
```

### 2. 在成员函数中调用外部函数

```cpp
double evaluate_objective(const Vector<double>& x) const override {
    // 直接调用外部热力学函数
    double fL = external_fugacity_liquid(T, P, x);
    double fV = external_fugacity_vapor(T, P, x);
    return (fL - fV) * (fL - fV);
}
```

### 3. 关键接口说明

#### 必须实现的虚函数

- `evaluate_objective()` - 目标函数
- `evaluate_constraints()` - 约束函数
- `evaluate_objective_gradient()` - 目标梯度
- `evaluate_constraint_jacobian()` - 约束Jacobian
- `evaluate_lagrangian_hessian()` - Lagrangian Hessian
- `variable_lower_bound()` / `variable_upper_bound()` - 变量界限
- `constraint_lower_bound()` / `constraint_upper_bound()` - 约束界限
- `initial_primal_point()` - 初始原始变量
- `initial_dual_point()` - 初始对偶变量

#### 集合类型

使用 `Range` 而不是抽象的 `Collection`：

```cpp
uno::Range equality_constraints(0, n_eq);  // [0, n_eq)
uno::Range inequality_constraints(n_eq, n_total);  // [n_eq, n_total)
```

### 4. 求解流程

```cpp
// 创建模型
MyModel model;

// 设置选项
uno::Options options;
options.set("print_level", "2");
options.set("tolerance", "1e-6");

// 创建初始迭代点
uno::Iterate initial_iterate(model.number_variables, model.number_constraints);
model.initial_primal_point(initial_iterate.primals);
model.initial_dual_point(initial_iterate.multipliers.constraints);

// 求解
uno::Uno solver;
auto result = solver.solve(model, initial_iterate, options);
```

## 外部函数集成示例

### 方式1：函数指针

```cpp
class ThermodynamicModel : public uno::Model {
private:
    // 函数指针
    std::function<double(double, double, const std::vector<double>&)> fugacity_func;
    
public:
    ThermodynamicModel() {
        // 绑定外部函数
        fugacity_func = &external_fugacity_calculation;
    }
    
    double evaluate_objective(const Vector<double>& x) const override {
        std::vector<double> composition(x.begin(), x.begin() + n_comp);
        return fugacity_func(T, P, composition);
    }
};
```

### 方式2：直接调用库函数

```cpp
// 链接外部库（如 CoolProp）
#include "CoolProp.h"

double evaluate_objective(const Vector<double>& x) const override {
    // 直接调用 CoolProp
    double rho = CoolProp::PropsSI("D", "T", T, "P", P, fluid);
    return calculate_gibbs_energy(rho, x);
}
```

### 方式3：C函数封装

```cpp
// 封装 C 函数
extern "C" {
    double fugacity_liquid_c(double T, double P, double* x, int n);
}

class Model : public uno::Model {
    double evaluate_objective(const Vector<double>& x) const override {
        return fugacity_liquid_c(T, P, x.data(), x.size());
    }
};
```

## 编译配置

### CMakeLists.txt

```cmake
# 添加可执行文件
add_executable(my_optimizer main.cpp MyModel.cpp)

# 链接 Uno 库
target_link_libraries(my_optimizer uno)

# 链接外部库（如果需要）
target_link_libraries(my_optimizer thermodynamics_lib)
```

### 包含路径

```cmake
include_directories(
    ${UNO_SOURCE_DIR}
    ${UNO_SOURCE_DIR}/uno
    ${EXTERNAL_LIB_INCLUDE}  # 外部库头文件
)
```

## 常见问题

### Q: Collection 是抽象类？
A: 使用 `Range` 或其他具体实现：
```cpp
uno::Range constraints(0, n);  // 不是 Collection<size_t>
```

### Q: Iterate 构造函数？
A: 使用两参数构造：
```cpp
uno::Iterate iter(n_variables, n_constraints);
```

### Q: 如何调试外部函数？
A: 在 evaluate_objective 中添加日志：
```cpp
double evaluate_objective(const Vector<double>& x) const override {
    double result = external_func(x);
    std::cout << "External func returned: " << result << "\n";
    return result;
}
```

## 完整示例

见文件：
- `simple_test.cpp` - 最简单的二元闪蒸示例
- `SimpleThermodynamicModel.hpp` - 完整的热力学模型
- `test_cpp_interface.cpp` - 测试程序

## 优势

1. **直接调用**：无需通过 AMPL 中间层
2. **类型安全**：C++ 编译时检查
3. **高性能**：避免函数指针查找开销
4. **灵活性**：可以使用任何 C/C++ 库
5. **调试方便**：可以使用标准 C++ 调试工具

## 限制

1. 需要手动实现所有 Model 接口
2. 需要自己计算导数（或使用自动微分）
3. 不能直接使用 AMPL 模型文件