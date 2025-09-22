# Eigen 求解器失败根本原因分析

## 核心问题：惯性值计算错误

### 发现的关键差异

在 hs015.nl 问题的第2次迭代中：
- **第1次迭代**：惯性 (4/2/0) - 正常
- **第2次迭代**：惯性 (0/6/0) - 异常！所有特征值都变成负数

这个异常的惯性值触发了 Uno 框架中的惯性修正机制，最终导致求解失败。

## 为什么会出现 (0/6/0) 的惯性？

### 1. 矩阵缩放的副作用

当 Eigen 求解器检测到病态矩阵时，会应用缩放：
```cpp
[EIGEN Solver] Applying matrix scaling for numerical stability
```

但是缩放改变了矩阵的数值特性，导致：
- **缩放前**：正常的混合惯性 (4/2/0)
- **缩放后**：异常的全负惯性 (0/6/0)

### 2. LDLT 分解的惯性计算问题

在 `EigenLinearSolver.cpp` 中：
```cpp
// Extract inertia from LDLT
const auto& D = this->ldlt_solver->vectorD();
for (int i = 0; i < D.size(); ++i) {
    if (std::abs(D[i]) < zero_tolerance) {
        this->zero_eigenvalues++;
    } else if (D[i] > 0) {
        this->positive_eigenvalues++;
    } else {
        this->negative_eigenvalues++;
    }
}
```

问题在于：
1. **缩放后的对角元素 D** 不再准确反映原始矩阵的特征值符号
2. **Eigen 的 SimplicialLDLT** 使用的分解策略可能产生错误的符号

## 为什么 MA27/MA57/MUMPS 没有这个问题？

### 1. 专业的惯性计算

这些求解器有专门的惯性跟踪机制：
- **MA57**：`info[23]` 直接给出负特征值数量
- **MUMPS**：`infog[11]` 和 `infog[27]` 分别给出负特征值和零特征值
- 它们在分解过程中精确跟踪主元的符号变化

### 2. 内置的数值稳定化

专业求解器的缩放是内置的，不会破坏惯性信息：
- 使用 **平衡缩放**（equilibrium scaling）
- 保持 **符号不变性**
- 在缩放后仍能准确计算惯性

### 3. 主元选择策略

- **MA27/MA57**：使用多波前（multifrontal）方法，动态选择主元
- **MUMPS**：使用复杂的主元选择策略（AMD、METIS等）
- **Eigen LDLT**：固定的对角主元，容易受数值扰动影响

## Uno 框架的惯性修正机制

### 正常的惯性修正流程

1. **目标惯性**：对于凸优化，希望 Hessian 矩阵正定
2. **检测**：如果惯性不满足要求，添加正则化
3. **修正**：通过添加 δI（δ > 0）来改善惯性

### Eigen 的失败场景

1. **错误的惯性 (0/6/0)**：所有特征值都是负的
2. **过度修正**：需要极大的 δ 来修正
3. **不稳定**：δ 超过阈值，触发 `UnstableRegularization` 异常

## 具体例子：aug2dc.nl

```
[EIGEN Solver] Inertia (pos/neg/zero) = (20190/10006/0)  # 第2次迭代
# 应该是 (20200/9996/0) 左右

The inertia correction got unstable (delta_w > threshold)
```

惯性值的微小偏差导致正则化策略失控。

## 解决方案

### 短期修复

1. **禁用缩放后的 LDLT**：
   - 缩放后直接使用 LU 或 QR
   - 避免依赖可能错误的惯性信息

2. **改进惯性计算**：
   - 使用 SVD 或其他方法验证惯性
   - 对缩放矩阵进行惯性补偿

### 长期方案

1. **实现专业的惯性跟踪**：
   - 像 MA57 那样在分解过程中跟踪符号变化
   - 使用 Sylvester 惯性定理

2. **集成更好的求解器**：
   - SuperLU
   - CHOLMOD
   - Pardiso

## 结论

Eigen 求解器失败的根本原因是：

1. **矩阵缩放破坏了惯性信息的准确性**
2. **SimplicialLDLT 的惯性计算不够鲁棒**
3. **错误的惯性导致 Uno 的正则化策略失控**

相比之下，专业求解器（MA27/MA57/MUMPS）：
- 有精确的惯性跟踪机制
- 缩放不影响惯性计算
- 能正确引导正则化策略

这解释了为什么在相同的优化问题上，MA系列和 MUMPS 能收敛，而 Eigen 无法收敛。