# Uno 编译和配置完整指南

## 概述
本文档记录了在 macOS (ARM64) 上成功编译 Uno 优化求解器的完整过程，包括解决 MUMPS 线性求解器的 MPI_ALLREDUCE 错误。

## 系统环境
- macOS (ARM64/Apple Silicon)
- Homebrew 包管理器
- GCC 15 (提供 gfortran)

## 一、安装依赖库

### 通过 Homebrew 安装
```bash
# 线性代数库
brew install openblas
brew install lapack

# 图分区库
brew install metis
brew install scotch  # 注意：只使用串行版本

# OpenMP 支持
brew install libomp

# 优化求解器
brew install highs

# 编译工具
brew install gcc  # 提供 gfortran-15
brew install cmake
```

## 二、编译 AMPL Solver Library (ASL)

```bash
cd /Users/shaoyiyang/Documents/Code/amplsolver
make -f makefile.u
# 生成 amplsolver.a
```

## 三、编译 MUMPS（串行版本）

### 3.1 配置 MUMPS
编辑 `/Users/shaoyiyang/Downloads/MUMPS_5.8.1/Makefile.inc`：

```makefile
# 1. 设置 SCOTCH 路径
SCOTCHDIR  = /opt/homebrew
ISCOTCH    = -I$(SCOTCHDIR)/include

# 2. 使用串行版本的 SCOTCH（重要！）
LSCOTCH    = -L$(SCOTCHDIR)/lib -lesmumps -lscotch -lscotcherr
# 不要使用：-lptesmumps -lptscotch -lptscotcherr

# 3. 设置 PORD
LPORDDIR = $(topdir)/PORD/lib/
IPORD    = -I$(topdir)/PORD/include/
LPORD    = -L$(LPORDDIR) -lpord$(PLAT)

# 4. 设置 METIS
LMETISDIR = /opt/homebrew/lib
IMETIS    = -I/opt/homebrew/include
LMETIS    = -L$(LMETISDIR) -lmetis

# 5. 配置排序选项（移除 ptscotch）
ORDERINGSF  = -Dpord -Dscotch -Dmetis
ORDERINGSC  = $(ORDERINGSF)

# 6. 编译器设置
CC      = gcc-15
FC      = gfortran-15
FL      = gfortran-15
AR      = ar vr 
RANLIB  = ranlib

# 7. 库设置
LAPACK = -L/opt/homebrew/lib -llapack
LIBBLAS = -L/opt/homebrew/opt/openblas/lib -lopenblas

# 8. 选择串行版本（关键！）
INCS = $(INCSEQ)
LIBS = $(LIBSEQ)
LIBSEQNEEDED = libseqneeded

# 注释掉并行版本
#INCS = $(INCPAR)
#LIBS = $(LIBPAR)
#LIBSEQNEEDED = 

# 9. Fortran/C 兼容性
CDEFS = -DAdd_

# 10. 优化选项
OPTF    = -O3 -fallow-argument-mismatch
OPTC    = -O3 -I.
OPTL    = -O3
```

### 3.2 编译 MUMPS
```bash
cd /Users/shaoyiyang/Downloads/MUMPS_5.8.1

# 清理旧文件
make clean

# 编译主库
make -j4

# 编译串行 MPI 库（重要）
cd libseq
make
cd ..
```

验证生成的库文件：
- `lib/libdmumps.a`
- `lib/libmumps_common.a`
- `lib/libpord.a`
- `libseq/libmpiseq.a`

## 四、修改 Uno 的 CMakeLists.txt

编辑 `/Users/shaoyiyang/Documents/Code/Uno/CMakeLists.txt`，在 MUMPS 部分（约第 210-235 行）：

```cmake
# 查找 OpenMP（设为可选而非必需）
find_package(OpenMP)
if(OpenMP_FOUND)
   list(APPEND LIBRARIES OpenMP::OpenMP_CXX)
else()
   message(WARNING "OpenMP not found, MUMPS may have reduced performance")
   # macOS 上手动添加 OpenMP
   if(APPLE)
      list(APPEND LIBRARIES /opt/homebrew/opt/libomp/lib/libomp.dylib)
      include_directories(/opt/homebrew/opt/libomp/include)
   endif()
endif()

# 添加 Fortran 运行时库（MUMPS 需要）
list(APPEND LIBRARIES /opt/homebrew/lib/gcc/current/libgfortran.dylib)

# 添加 SCOTCH 库（只使用串行版本）
if(APPLE)
   list(APPEND LIBRARIES /opt/homebrew/lib/libscotch.dylib)
   list(APPEND LIBRARIES /opt/homebrew/lib/libscotcherr.dylib)
   list(APPEND LIBRARIES /opt/homebrew/lib/libesmumps.dylib)
   # 注意：不要包含 libptscotch 和 libptesmumps
endif()

add_definitions("-D HAS_MUMPS")
message(STATUS "Found MUMPS")
```

## 五、编译 Uno

```bash
cd /Users/shaoyiyang/Documents/Code/Uno
mkdir -p build
cd build

# 配置 CMake
cmake .. \
  -DAMPLSOLVER=/Users/shaoyiyang/Documents/Code/amplsolver/amplsolver.a \
  -DMUMPS_LIBRARY=/Users/shaoyiyang/Downloads/MUMPS_5.8.1/lib/libdmumps.a \
  -DMUMPS_COMMON_LIBRARY=/Users/shaoyiyang/Downloads/MUMPS_5.8.1/lib/libmumps_common.a \
  -DMUMPS_PORD_LIBRARY=/Users/shaoyiyang/Downloads/MUMPS_5.8.1/lib/libpord.a \
  -DMUMPS_MPISEQ_LIBRARY=/Users/shaoyiyang/Downloads/MUMPS_5.8.1/libseq/libmpiseq.a \
  -DMUMPS_INCLUDE_DIR=/Users/shaoyiyang/Downloads/MUMPS_5.8.1/include \
  -DHIGHS_DIR=/opt/homebrew

# 编译
make -j4

# 编译 AMPL 接口
make uno_ampl
```

## 六、验证安装

### 6.1 查看可用策略
```bash
./uno_ampl --strategies
```

应该显示：
```
Available Uno strategies:
- Linear solvers: MUMPS
- QP solvers: HiGHS
- LP solvers: HiGHS
- Presets: filtersqp, ipopt
```

### 6.2 测试求解
```bash
# 测试基本求解
./uno_ampl ../examples/hs015.nl

# 使用 IPOPT 预设（内点法 + MUMPS）
./uno_ampl ../examples/hs015.nl preset=ipopt

# 处理负曲率问题
./uno_ampl ../examples/hs015.nl hessian_model=identity
```

## 七、使用指南

### 7.1 基本命令格式
```bash
./uno_ampl problem.nl [option=value] [option=value] ...
```

### 7.2 常用参数选项

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `preset` | 算法预设 | `filtersqp`, `ipopt` |
| `linear_solver` | 线性求解器 | `MUMPS` |
| `hessian_model` | Hessian 模型 | `exact`, `identity`, `zero` |
| `regularization_strategy` | 正则化策略 | `none`, `primal`, `primal_dual` |
| `globalization_mechanism` | 全局化机制 | `TR`（信赖域）, `LS`（线搜索） |
| `globalization_strategy` | 全局化策略 | `l1_merit`, `fletcher_filter_method`, `waechter_filter_method` |
| `primal_tolerance` | 原始容差 | `1e-6` |
| `dual_tolerance` | 对偶容差 | `1e-6` |
| `max_iterations` | 最大迭代次数 | `2000` |
| `logger` | 日志级别 | `SILENT`, `WARNING`, `INFO`, `DEBUG` |
| `print_solution` | 打印解 | `yes`, `no` |

### 7.3 使用选项文件
创建 `options.txt`：
```
preset ipopt
linear_solver MUMPS
primal_tolerance 1e-6
dual_tolerance 1e-6
max_iterations 1000
logger INFO
```

运行：
```bash
./uno_ampl problem.nl option_file=options.txt
```

### 7.4 常见问题解决方案

#### 负曲率问题
```bash
# 使用单位矩阵 Hessian
./uno_ampl problem.nl hessian_model=identity

# 或增加正则化
./uno_ampl problem.nl regularization_strategy=primal
```

#### MPI_ALLREDUCE 错误
确保 MUMPS 编译为串行版本，不使用并行 SCOTCH (ptscotch)。

#### 收敛问题
```bash
# 放松容差
./uno_ampl problem.nl primal_tolerance=1e-5 dual_tolerance=1e-5

# 增加迭代次数
./uno_ampl problem.nl max_iterations=5000

# 使用不同预设
./uno_ampl problem.nl preset=ipopt  # 或 preset=filtersqp
```

## 八、故障排除

### 8.1 常见错误及解决方案

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| `ERROR in MPI_ALLREDUCE` | MUMPS 编译为并行版本 | 重新编译 MUMPS 为串行版本 |
| `linear_solver was not found` | 缺少线性求解器 | 安装并链接 MUMPS |
| `negative curvature` | Hessian 非正定 | 使用 `hessian_model=identity` |
| `undefined symbols` | 缺少库依赖 | 检查 gfortran 和 SCOTCH 库链接 |

### 8.2 验证库链接
```bash
# 查看 uno_ampl 的库依赖
otool -L uno_ampl

# 应该包含：
# - libhighs.dylib
# - libgfortran.dylib
# - libscotch.dylib
# - libomp.dylib
```

## 九、性能优化建议

1. **使用精确 Hessian**：当问题良态时，`hessian_model=exact` 性能最佳
2. **选择合适的预设**：
   - `filtersqp`：适合一般非线性问题
   - `ipopt`：适合大规模问题和需要内点法的场景
3. **调整正则化**：对于病态问题，使用 `regularization_strategy=primal_dual`
4. **并行计算**：虽然 MUMPS 是串行版本，但 Uno 其他部分仍可利用多核

## 十、参考资源

- Uno 官方仓库：https://github.com/cvanaret/Uno
- MUMPS 官网：https://mumps-solver.org/
- AMPL：https://ampl.com/
- HSL 数学软件库：https://www.hsl.rl.ac.uk/

---
*文档更新日期：2025-09-15*
*作者：基于实际编译经验整理*