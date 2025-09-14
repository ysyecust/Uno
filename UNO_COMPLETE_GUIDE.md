# Uno 完整使用指南

## 1. 项目概述

Uno (Unifying Nonlinear Optimization) 是一个模块化的非线性约束优化求解器，实现了统一的优化框架，支持多种算法策略的灵活组合。

### 主要特点
- 高度模块化的设计架构
- 支持多种算法策略组合
- 可扩展的外部函数接口
- 支持 AMPL 模型格式

## 2. 编译和安装

### 2.1 编译 AMPL Solver Library
```bash
cd /Users/shaoyiyang/Documents/Code/amplsolver
make -f makefile.u
# 生成 amplsolver.a
```

### 2.2 编译 Uno
```bash
cd /Users/shaoyiyang/Documents/Code/Uno
mkdir build && cd build
cmake .. -DAMPLSOLVER_DIR=/Users/shaoyiyang/Documents/Code/amplsolver
make -j4
```

## 3. 基本使用

### 3.1 使用 AMPL 模型

#### 转换 .mod 文件为 .nl 格式
```bash
ampl -ogmodel_name model_name.mod
```

#### 求解 .nl 文件
```bash
./uno_ampl problem.nl
```

### 3.2 使用 C++ 接口
```cpp
#include "uno/Uno.hpp"

class MyModel : public uno::Model {
    // 实现模型接口
};

int main() {
    MyModel model;
    uno::Options options;
    uno::Uno solver(model, options);
    solver.solve();
}
```

## 4. 模块化架构

### 4.1 核心组件
- **约束松弛策略** (constraint_relaxation_strategies)
  - FeasibilityRestoration
  - UnconstrainedStrategy
  - RelaxedPenalty

- **全局化策略** (globalization_strategies)
  - l1MeritFunction
  - FletcherFilterMethod
  - WaechterFilterMethod

- **全局化机制** (globalization_mechanisms)
  - TrustRegionStrategy
  - BacktrackingLineSearch

- **Hessian 模型** (hessian_models)
  - HessianModel
  - ConvexifiedHessian

### 4.2 设计模式
- 策略模式：算法组件的基础
- 工厂模式：创建策略实例
- 模板方法：定义算法框架

## 5. 外部函数集成

### 5.1 AMPL 外部函数（受限）
```c
#include "funcadd.h"

real my_function(arglist *al) {
    // 实现函数逻辑
    return result;
}

void funcadd(AmplExports *ae) {
    addfunc("my_function", (ufunc*)my_function, 0, n_args, ae);
}
```

**注意**：uno_ampl 当前不支持外部函数回调

### 5.2 C++ 模型类（推荐）
```cpp
class CustomModel : public uno::Model {
    double evaluate_objective(const Vector<double>& x) override {
        return custom_function(x);
    }
};
```

## 6. 热力学优化示例

详见 `examples/thermodynamic_optimization/` 目录：
- 外部函数库实现（fugacity, activity coefficient）
- 闪蒸计算模型
- C++ 集成接口

### 关键文件
- `thermo_functions.c` - 外部函数实现
- `ThermodynamicModel.hpp` - C++ 模型类
- `flash_calculation.mod` - AMPL 模型

## 7. 常见问题

### Q: 如何处理外部函数？
A: 使用 C++ Model 类接口，uno_ampl 暂不支持外部函数。

### Q: 如何调试求解过程？
A: 设置选项：
```cpp
options.set("print_level", 5);  // 详细输出
options.set("tolerance", 1e-6);  // 收敛容差
```

### Q: 如何选择算法策略？
A: 通过选项配置：
```cpp
options.set("constraint_relaxation_strategy", "feasibility_restoration");
options.set("globalization_strategy", "fletcher_filter_method");
```

## 8. 性能优化建议

1. **提供解析导数**：避免数值微分
2. **利用稀疏性**：使用稀疏矩阵表示
3. **合理初始值**：改善收敛性
4. **调整算法参数**：根据问题特点优化

## 9. 扩展开发

### 添加新策略
1. 继承相应基类
2. 实现虚函数接口
3. 在工厂中注册
4. 通过选项使用

### 集成外部库
- CoolProp（热力学性质）
- Cantera（化学反应）
- Eigen（线性代数）

## 10. 参考资源

- 源代码：https://github.com/cvanaret/Uno
- AMPL：https://ampl.com/
- ASL文档：https://ampl.com/resources/the-ampl-book/

---
*文档更新日期：2025-09-12*