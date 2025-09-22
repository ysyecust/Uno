# Uno 线性求解器集成与分析总结

## 项目概述

本项目完成了 Uno 优化框架中多个线性求解器的集成、测试和深度分析工作，包括：
- Eigen 求解器的集成与改进
- HSL 求解器（MA27、MA57）的集成
- 各求解器性能对比与失败原因分析
- 数值稳定性和惯性计算的深入研究

## 编译命令汇总

### 1. 基础编译（含 MUMPS 和 Eigen）

```bash
# 在 build 目录下
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
make uno_ampl -j8
```

### 2. 集成 HSL 求解器

```bash
# 配置 HSL 库路径
cmake .. -DHSL="/Users/shaoyiyang/Downloads/libHSL_binaries.v2025.7.21 2/lib/libhsl.dylib" \
         -DMETIS="/Users/shaoyiyang/Downloads/libHSL_binaries.v2025.7.21 2/lib/libmetis.dylib"

# macOS 需要移除隔离属性
xattr -cr "/Users/shaoyiyang/Downloads/libHSL_binaries.v2025.7.21 2/"

# 设置 gfortran 库路径并编译
export LIBRARY_PATH="/opt/homebrew/lib/gcc/current:$LIBRARY_PATH"
make clean && make uno_ampl -j8
```

### 3. 测试命令

```bash
# 测试不同求解器
./uno_ampl ../examples/cosfunc.nl linear_solver=MUMPS
./uno_ampl ../examples/cosfunc.nl linear_solver=EIGEN
./uno_ampl ../examples/cosfunc.nl linear_solver=MA27
./uno_ampl ../examples/cosfunc.nl linear_solver=MA57

# 使用 ipopt preset（更严格的测试）
./uno_ampl ../examples/hs015.nl preset=ipopt linear_solver=EIGEN
./uno_ampl ../examples/aug2dc.nl preset=ipopt linear_solver=MA57
```

## 代码更改

### 1. Eigen 求解器实现
- **新增文件**：
  - `uno/ingredients/subproblem_solvers/EIGEN/EigenLinearSolver.hpp`
  - `uno/ingredients/subproblem_solvers/EIGEN/EigenLinearSolver.cpp`

- **主要功能**：
  - 多重分解策略（LDLT → LU → QR）
  - 矩阵缩放以改善条件数
  - 正则化机制（渐进式，1e-8 到 1e-2）
  - 解验证和迭代改进
  - 惯性计算（虽然存在问题）

### 2. 集成到框架
- **修改文件**：
  - `uno/ingredients/subproblem_solvers/SymmetricIndefiniteLinearSolverFactory.cpp`
  - `CMakeLists.txt`（添加 Eigen 相关配置）

## 性能对比结果

### 小规模问题 (hs015.nl, preset=ipopt)

| 求解器 | 收敛状态 | 迭代次数 | CPU时间 |
|--------|----------|---------|---------|
| MA27   | ✅ 成功  | 21      | 0.099s  |
| MA57   | ✅ 成功  | 21      | 0.243s  |
| MUMPS  | ✅ 成功  | 21      | 0.003s  |
| Eigen  | ❌ 失败  | 2       | 0.002s  |

### 大规模病态问题 (aug2dc.nl, preset=ipopt)

| 求解器 | 收敛状态 | 迭代次数 | CPU时间 |
|--------|----------|---------|---------|
| MA27   | ✅ 成功  | 14      | 2.217s  |
| MA57   | ✅ 成功  | 14      | 2.788s  |
| MUMPS  | ✅ 成功  | 14      | 2.916s  |
| Eigen  | ❌ 失败  | 2       | 35.678s |

## 关键发现

### 1. Eigen 失败的根本原因
- **惯性计算错误**：矩阵缩放后，惯性从 (4,2,0) 变为 (0,6,0)
- **缺少 2×2 主元块支持**：无法稳定处理对称不定矩阵
- **简单的数值策略**：缺少专业求解器的高级技术

### 2. 专业求解器的优势
- **精确的惯性跟踪**：在分解过程中记录，不受缩放影响
- **2×2 主元块**：Bunch-Kaufman 算法，处理强耦合和小主元
- **高级数值技术**：阈值主元选择、静态主元、平衡缩放

### 3. 惯性在优化中的重要性
- 指导搜索方向（下降/上升）
- 决定正则化策略
- 检测数值问题

## 深入分析文档

保留的核心文档：
1. **INERTIA_EXPLANATION.md** - 惯性值的详细解释
2. **2x2_PIVOT_BLOCKS_EXPLANATION.md** - 2×2 主元块的原理
3. **EIGEN_FAILURE_ROOT_CAUSE_ANALYSIS.md** - Eigen 失败原因分析
4. **HSL_SOLVERS_COMPARISON.md** - HSL 求解器性能对比

## 结论与建议

### 当前状态
- ✅ 成功集成 Eigen、MA27、MA57 求解器
- ✅ 完成全面的性能测试和对比
- ✅ 深入理解了不同求解器的数值特性
- ⚠️ Eigen 求解器在病态问题上仍有局限

### 求解器选择建议
- **默认选择**：MUMPS（综合性能最好）
- **大规模问题**：MA27（速度快，内存效率高）
- **需要诊断信息**：MA57（提供详细信息）
- **简单良态问题**：Eigen（轻量级，无需外部依赖）

### 未来改进方向
1. 考虑集成更多专业求解器（SuperLU、CHOLMOD、Pardiso）
2. 为 Eigen 实现 2×2 主元块支持（工作量大）
3. 改进惯性计算的鲁棒性

## 环境信息
- 操作系统：macOS (Darwin 25.0.0)
- 编译器：clang++、gfortran 15.1.0
- CMake：3.x
- 依赖库：Eigen 3.4.0、MUMPS 5.8.1、HSL 2025.7.21