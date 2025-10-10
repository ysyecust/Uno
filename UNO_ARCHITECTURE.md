# UNO 项目架构文档

## 1. 项目概述

### 1.1 简介
UNO (Uno Numerical Optimization) 是一个模块化、可扩展的非线性优化求解器框架。它支持多种优化算法，包括内点法、SQP方法、信赖域方法等，能够处理约束和无约束优化问题。

### 1.2 主要特性
- **模块化设计**：各组件独立可替换
- **多算法支持**：内点法、滤波器方法、信赖域等
- **灵活的线性求解器**：支持 MUMPS、MA27/MA57、Eigen 等
- **高级特性**：正则化、惯性修正、线搜索、滤波器
- **AMPL 接口**：支持标准优化建模语言

### 1.3 设计理念
- **可组合性**：通过"ingredients"（配料）概念组合不同算法组件
- **可扩展性**：易于添加新的求解器和算法
- **数值鲁棒性**：多种数值稳定化技术

## 2. 系统架构

### 2.1 整体架构图

```plantuml
@startuml uno_architecture
!theme plain
skinparam packageStyle rectangle
skinparam componentStyle uml2

package "UNO Framework" {

    package "Interface Layer" {
        [AMPL Interface] as AMPL
        [C++ API] as API
    }

    package "Core Engine" {
        [Uno Solver] as Solver
        [Optimization Algorithm] as OptAlgo
        [Model Management] as Model
    }

    package "Ingredients" {
        [Globalization] as Glob
        [Regularization] as Reg
        [Subproblem Solvers] as SubSolvers
        [Hessian Models] as Hess
        [Constraint Relaxation] as Relax
        [Inequality Handling] as Ineq
    }

    package "Foundation" {
        [Linear Algebra] as LA
        [Symbolic Computation] as Symb
        [Options Management] as Opts
        [Tools & Utilities] as Tools
    }

    package "Linear Solvers" {
        [MUMPS] as MUMPS
        [MA27/MA57] as MA
        [Eigen] as Eigen
        [HiGHS] as HiGHS
    }
}

AMPL --> Solver
API --> Solver
Solver --> OptAlgo
Solver --> Model
OptAlgo --> Glob
OptAlgo --> Reg
OptAlgo --> SubSolvers
OptAlgo --> Hess
OptAlgo --> Relax
OptAlgo --> Ineq
SubSolvers --> MUMPS
SubSolvers --> MA
SubSolvers --> Eigen
SubSolvers --> HiGHS
Glob --> LA
Reg --> LA
Hess --> LA
Model --> Symb
Solver --> Opts
Solver --> Tools

@enduml
```

### 2.2 层次结构

```
┌─────────────────────────────────────────┐
│          Interface Layer                │
│     (AMPL, C++ API, Python绑定)        │
├─────────────────────────────────────────┤
│          Core Optimization              │
│    (Uno, OptimizationAlgorithm)        │
├─────────────────────────────────────────┤
│          Algorithm Components           │
│         (Ingredients层)                 │
├─────────────────────────────────────────┤
│          Mathematical Foundation        │
│    (线性代数, 符号计算, 工具)          │
├─────────────────────────────────────────┤
│          External Solvers              │
│     (MUMPS, HSL, Eigen, HiGHS)        │
└─────────────────────────────────────────┘
```

## 3. 核心组件

### 3.1 主要模块详解

#### 3.1.1 Model（模型管理）
负责优化问题的表示和管理：

```plantuml
@startuml model_hierarchy
!theme plain

class Model {
    + evaluate_objective()
    + evaluate_constraints()
    + evaluate_objective_gradient()
    + evaluate_constraint_jacobian()
    + evaluate_lagrangian_hessian()
    + number_variables
    + number_constraints
}

class AMPLModel {
    - asl: ASL*
    + read_from_file()
}

class BoundRelaxedModel {
    - original_model: Model&
    + relax_bounds()
}

class HomogeneousEqualityConstrainedModel {
    + reformulate_constraints()
}

Model <|-- AMPLModel
Model <|-- BoundRelaxedModel
Model <|-- HomogeneousEqualityConstrainedModel

@enduml
```

#### 3.1.2 Ingredients（算法组件）

```plantuml
@startuml ingredients_components
!theme plain

package "Ingredients" {

    package "Globalization Mechanisms" {
        class LineSearch {
            + compute_step_length()
        }
        class TrustRegion {
            + compute_trust_region_step()
            + update_radius()
        }
        class Filter {
            + is_acceptable()
            + add_to_filter()
        }
    }

    package "Subproblem Solvers" {
        abstract class LinearSolver {
            + symbolic_analysis()
            + numerical_factorization()
            + solve_system()
            + get_inertia()
        }

        class MUMPSSolver
        class MA57Solver
        class EigenLinearSolver

        LinearSolver <|-- MUMPSSolver
        LinearSolver <|-- MA57Solver
        LinearSolver <|-- EigenLinearSolver
    }

    package "Regularization Strategies" {
        class PrimalDualRegularization {
            + compute_delta_w()
            + compute_delta_c()
        }
    }

    package "Hessian Models" {
        class ExactHessian
        class LBFGSHessian
        class GaussNewtonHessian
    }
}

@enduml
```

### 3.2 求解器工厂模式

```plantuml
@startuml factory_pattern
!theme plain

interface SolverFactory {
    + create(name: string): Solver
}

class LinearSolverFactory {
    + create("MUMPS"): MUMPSSolver
    + create("MA57"): MA57Solver
    + create("EIGEN"): EigenLinearSolver
}

class QPSolverFactory {
    + create("BQPD"): BQPDSolver
    + create("HiGHS"): HiGHSSolver
}

SolverFactory <|.. LinearSolverFactory
SolverFactory <|.. QPSolverFactory

@enduml
```

## 4. 计算工作流

### 4.1 主要优化循环

```plantuml
@startuml optimization_workflow
!theme plain
start

:读取问题 (AMPL/API);

:初始化求解器;
note right
  - 选择算法
  - 配置选项
  - 分配内存
end note

:预处理;
note right
  - 缩放
  - 约束松弛
  - 变量界限处理
end note

repeat
    :计算搜索方向;
    note right
      构建并求解子问题:
      - KKT系统
      - QP子问题
      - 信赖域子问题
    end note

    :线搜索/信赖域更新;

    if (使用滤波器?) then (是)
        :滤波器接受性测试;
    else (否)
        :传统接受性测试;
    endif

    :更新迭代点;

    :计算收敛指标;
    note right
      - 原始可行性
      - 对偶可行性
      - 互补性
      - 最优性
    end note

repeat while (未收敛?) is (是)

:后处理;

:输出结果;

stop

@enduml
```

### 4.2 子问题求解流程

```plantuml
@startuml subproblem_solving
!theme plain

start

:构建线性系统;
note right
  KKT系统:
  [H    -Jᵀ] [Δx]   [∇f]
  [-J    0 ] [Δλ] = [c ]
end note

:符号分析;
note right
  - 分析稀疏结构
  - 优化存储格式
  - 预分配内存
end note

:数值分解;

if (分解成功?) then (否)
    :应用正则化;
    note right
      H → H + δI
      δ 从小到大尝试
    end note
    :重新分解;
endif

:检查惯性;

if (惯性正确?) then (否)
    :惯性修正;
    note right
      调整 δ 使得
      惯性满足要求
    end note
endif

:求解线性系统;

:验证解质量;

if (解质量差?) then (是)
    :迭代改进;
endif

:返回搜索方向;

stop

@enduml
```

## 5. 数据流与控制流

### 5.1 数据流图

```plantuml
@startuml data_flow
!theme plain

actor User
entity "AMPL File" as AMPL
database "Problem Data" as Data
control "Uno Solver" as Solver
entity "Linear System" as LinSys
control "Linear Solver" as LinSolver
entity "Solution" as Solution

User -> AMPL : 创建模型
AMPL -> Data : 解析存储
Data -> Solver : 加载问题
Solver -> LinSys : 构建KKT系统
LinSys -> LinSolver : 求解
LinSolver -> Solution : 计算方向
Solution -> Solver : 更新迭代
Solver -> User : 返回结果

@enduml
```

### 5.2 控制流状态机

```plantuml
@startuml control_flow_state
!theme plain

[*] --> 初始化

初始化 --> 优化阶段 : 初始点可行
初始化 --> 可行性恢复 : 初始点不可行

优化阶段 --> 检查收敛
可行性恢复 --> 检查收敛

检查收敛 --> 优化阶段 : 未收敛且可行
检查收敛 --> 可行性恢复 : 未收敛且不可行
检查收敛 --> 成功 : 收敛到KKT点
检查收敛 --> 失败 : 达到迭代限制

成功 --> [*]
失败 --> [*]

note right of 优化阶段
  最小化目标函数
  同时保持可行性
end note

note right of 可行性恢复
  最小化约束违反
  忽略目标函数
end note

@enduml
```

## 6. 关键算法实现

### 6.1 内点法 (Interior Point Method)

#### 算法流程
```python
# 伪代码
def interior_point_method(problem):
    x, s, λ, z = initialize_variables()
    μ = initial_barrier_parameter

    while not converged:
        # 构建 KKT 系统
        KKT = build_kkt_system(x, s, λ, z, μ)

        # 求解搜索方向
        Δx, Δs, Δλ, Δz = solve_kkt_system(KKT)

        # 线搜索
        α = line_search(x, s, Δx, Δs)

        # 更新变量
        x += α * Δx
        s += α * Δs
        λ += α * Δλ
        z += α * Δz

        # 更新障碍参数
        μ = update_barrier_parameter(μ)

    return x
```

#### KKT 系统结构
```
┌                                    ┐ ┌    ┐   ┌      ┐
│  H + Σ    0     -Jᵀ    -I        │ │ Δx │   │  -∇L  │
│   0       Z     0      S         │ │ Δs │ = │  -rc  │
│  -J       0     0      0         │ │ Δλ │   │   c   │
│  -I       I     0      0         │ │ Δz │   │  x-s  │
└                                    ┘ └    ┘   └      ┘
```

### 6.2 信赖域方法 (Trust Region Method)

```plantuml
@startuml trust_region_algorithm
!theme plain

start

:初始化信赖域半径 Δ;

repeat
    :求解信赖域子问题|
    note right
      min  m(p) = f + gᵀp + ½pᵀHp
      s.t. ||p|| ≤ Δ
    end note

    :计算实际下降/预测下降;
    note right
      ρ = (f(x) - f(x+p)) / (m(0) - m(p))
    end note

    if (ρ > η₁) then (接受步)
        :x = x + p;
        if (ρ > η₂) then (很好)
            :Δ = min(γ₂Δ, Δmax);
        endif
    else (拒绝步)
        :Δ = γ₁Δ;
    endif

repeat while (未收敛?)

stop

@enduml
```

### 6.3 滤波器机制 (Filter Mechanism)

```plantuml
@startuml filter_mechanism
!theme plain

class Filter {
    - filter_pairs: vector<pair<h,f>>
    + is_acceptable(h_trial, f_trial): bool
    + add_to_filter(h, f): void
    + clear_filter(): void
}

note right of Filter::is_acceptable
  接受条件:
  h_trial ≤ β*h_k 或
  f_trial ≤ f_k - γ*h_k
  对所有 (h_k, f_k) ∈ filter
end note

note right of Filter::filter_pairs
  h: 约束违反度
  f: 目标函数值
end note

@enduml
```

### 6.4 惯性修正算法

```python
# 伪代码
def inertia_correction(H, target_inertia):
    """
    修正 Hessian 矩阵的惯性
    目标：n_pos = n, n_neg = 0, n_zero = 0 (凸化)
    """
    δ = δ_min

    while δ < δ_max:
        # 尝试分解 H + δI
        factorize(H + δ * I)
        inertia = get_inertia()

        if inertia == target_inertia:
            return δ

        # 调整 δ
        if inertia.n_neg > 0:
            δ *= 10  # 需要更大的正则化
        else:
            break

    return δ
```

## 7. 性能优化策略

### 7.1 稀疏矩阵处理

- **COO 格式**：用于构建和修改
- **CSR/CSC 格式**：用于矩阵运算
- **符号分析**：预先分析稀疏结构，重用于多次分解

### 7.2 内存管理

- **预分配策略**：避免动态分配
- **工作空间重用**：迭代间重用临时空间
- **延迟计算**：只在需要时计算 Hessian

### 7.3 数值稳定性

- **缩放技术**：平衡矩阵条件数
- **主元选择**：1×1 和 2×2 主元块
- **迭代改进**：提高解的精度

## 8. 扩展机制

### 8.1 添加新的线性求解器

```cpp
class NewSolver : public DirectSymmetricIndefiniteLinearSolver {
public:
    void do_symbolic_analysis() override;
    void do_numerical_factorization(const double* values) override;
    void solve_system(const Vector& rhs, Vector& solution) override;
    Inertia get_inertia() const override;
};
```

### 8.2 添加新的优化算法

```cpp
class NewAlgorithm : public OptimizationAlgorithm {
public:
    void initialize(const Problem& problem) override;
    Direction compute_direction(const Iterate& current) override;
    void update_iterate(Iterate& current, const Direction& d) override;
};
```

### 8.3 插件系统

通过工厂模式注册新组件：
```cpp
LinearSolverFactory::register("NEW_SOLVER",
    []() { return std::make_unique<NewSolver>(); });
```

## 9. 配置系统

### 9.1 选项层次

```plantuml
@startuml options_hierarchy
!theme plain

class Options {
    - options_map: map<string, value>
    + get(key): value
    + set(key, value): void
}

class DefaultOptions {
    + get_default_options(): Options
}

class Presets {
    + get_preset("ipopt"): Options
    + get_preset("filterslp"): Options
    + get_preset("byrd"): Options
}

Options <-- DefaultOptions : 提供默认值
Options <-- Presets : 提供预设配置

@enduml
```

### 9.2 关键配置参数

| 参数类别 | 参数名 | 说明 |
|---------|--------|------|
| 线性求解器 | `linear_solver` | MUMPS, MA27, MA57, EIGEN |
| 全局化策略 | `globalization_strategy` | line_search, trust_region |
| 正则化策略 | `regularization_strategy` | none, primal, primal_dual |
| Hessian模型 | `hessian_model` | exact, LBFGS, gauss_newton |
| QP求解器 | `QP_solver` | BQPD, HiGHS |
| 收敛容差 | `tolerance` | 默认 1e-8 |

## 10. 总结

### 10.1 架构特点

1. **高度模块化**：各组件通过接口解耦，易于替换和扩展
2. **灵活组合**：通过 ingredients 机制组合不同算法组件
3. **数值鲁棒**：多层次的数值稳定化策略
4. **性能优化**：稀疏矩阵、内存重用、并行化支持

### 10.2 适用场景

- 大规模非线性优化问题
- 约束优化问题（等式和不等式约束）
- 需要高度定制化的优化应用
- 研究和教学用途

### 10.3 未来发展

- GPU 加速支持
- 分布式计算能力
- 更多的机器学习集成
- 自动微分支持