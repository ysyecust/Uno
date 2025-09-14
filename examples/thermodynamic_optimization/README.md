# 热力学优化示例 - C++ 接口调用外部函数

## ✅ 已成功实现

本示例成功演示了如何通过 Uno 的 C++ 接口调用外部函数进行优化。

## 核心文件

### 1. `minimal_test.cpp` - **可运行示例**
演示如何在 C++ 中调用外部函数：
```cpp
class MinimalModel : public uno::Model {
    // 外部函数 - 可以是任何库！
    double external_function(double x) const {
        return std::exp(-x * x);  // 可替换为 CoolProp、Cantera 等
    }
    
    double evaluate_objective(const Vector<double>& x) const override {
        // 直接调用外部函数
        double f1 = external_function(x[0]);
        double f2 = external_function(x[1]);
        return -f1 * f2;
    }
};
```

### 2. `CPP_INTERFACE_GUIDE.md` - 详细指南
包含完整的实现步骤和常见问题解决方案。

### 3. AMPL 模型文件（仅供参考）
- `flash_calculation.mod` - 完整闪蒸计算模型
- `simplified_flash.mod` - 简化版本
- `thermo_functions.c` - AMPL 外部函数实现

**注意**：uno_ampl 不支持外部函数，这些文件仅用于与其他求解器（如 IPOPT）对比。

## 编译和运行

### 编译
```bash
cd /Users/shaoyiyang/Documents/Code/Uno/build
cmake --build . --target minimal_test
```

### 运行
```bash
./examples/thermodynamic_optimization/minimal_test
```

### 输出示例
```
===== C++ 接口测试：调用外部函数 =====
开始求解...
Optimization status: Success
最优解: x[0] = 0.5, x[1] = 0.5
外部函数值: exp(-0.25) = 0.7788
✅ 成功通过C++接口调用外部函数!
```

## 关键发现

| 方法 | 支持外部函数 | 说明 |
|-----|------------|------|
| AMPL/.nl 文件 | ❌ | uno_ampl 不支持外部函数回调 |
| C++ 接口 | ✅ | 完全支持，可调用任何 C/C++ 库 |

## 如何集成实际的热力学库

### 1. CoolProp
```cpp
#include "CoolProp.h"

double evaluate_objective(const Vector<double>& x) const override {
    double density = CoolProp::PropsSI("D", "T", T, "P", P, "Water");
    return calculate_gibbs_energy(density, x);
}
```

### 2. Cantera
```cpp
#include "cantera/thermo.h"

double evaluate_objective(const Vector<double>& x) const override {
    auto phase = Cantera::newPhase("gri30.yaml");
    phase->setState_TP(T, P);
    return phase->gibbs_mole();
}
```

### 3. 自定义 C 函数
```cpp
extern "C" {
    double my_thermo_function(double T, double P, double* x, int n);
}

double evaluate_objective(const Vector<double>& x) const override {
    return my_thermo_function(T, P, x.data(), x.size());
}
```

## 技术要点

1. **必须设置选项**：
```cpp
uno::DefaultOptions::load(options);
uno::Presets::set(options, "filtersqp");
```

2. **使用 Range 而非 Collection**：
```cpp
static uno::Range eq_constraints(0, n);
return eq_constraints;
```

3. **正确链接库**：
```cmake
target_link_libraries(minimal_test 
    libuno.a
    libhighs.dylib
)
```

## 总结

- ✅ C++ 接口完全支持外部函数调用
- ✅ 可以集成任何 C/C++ 库
- ✅ 性能最优，无中间层开销
- ❌ AMPL/.nl 方式不支持（uno_ampl 限制）

---
*更新日期：2025-09-12*