# flash_calculation.mod
# 闪蒸计算优化问题：最小化 Gibbs 自由能
# 用于计算汽液平衡

# 加载外部热力学函数库（编译后的 .so 文件）
# load thermo_functions.so;

# 声明外部函数
function fugacity_liquid;
function fugacity_vapor;
function mixing_enthalpy;

# 参数
param n_components := 3;  # 组分数量
param T := 300;           # 温度 [K]
param P := 101325;        # 压力 [Pa]

# 总摩尔数和进料组成
param F := 100;           # 总进料摩尔数 [mol]
param z{i in 1..n_components};  # 进料组成

# 给定进料组成
data;
param z :=
  1  0.3   # 组分1
  2  0.5   # 组分2
  3  0.2;  # 组分3

# 变量
var V >= 0, <= F;  # 气相摩尔数
var L >= 0, <= F;  # 液相摩尔数
var x{i in 1..n_components} >= 1e-6, <= 1;  # 液相摩尔分率
var y{i in 1..n_components} >= 1e-6, <= 1;  # 气相摩尔分率
var K{i in 1..n_components} >= 0.01, <= 100, := 1;  # K值（平衡比）

# 目标函数：最小化 Gibbs 自由能
# 简化版本：最小化逸度差的平方和
minimize gibbs_energy:
    sum{i in 1..n_components} (
        # 液相和气相逸度应该相等
        (fugacity_liquid(T, P, x[1], x[2], x[3]) * x[i] - 
         fugacity_vapor(T, P, y[1], y[2], y[3]) * y[i])^2
    ) + 0.001 * mixing_enthalpy(T, P, x[1], x[2], x[3]);

# 约束条件

# 1. 物料平衡
subject to material_balance:
    L + V = F;

subject to component_balance{i in 1..n_components}:
    L * x[i] + V * y[i] = F * z[i];

# 2. 摩尔分率归一化
subject to liquid_sum:
    sum{i in 1..n_components} x[i] = 1;

subject to vapor_sum:
    sum{i in 1..n_components} y[i] = 1;

# 3. 平衡关系（Rachford-Rice方程的替代形式）
subject to equilibrium{i in 1..n_components}:
    y[i] = K[i] * x[i];

# 4. K值约束（基于热力学）
# 在实际应用中，K值是温度和压力的函数
subject to K_value_constraint{i in 1..n_components}:
    K[i] = exp((300 - T)/100 + (101325 - P)/50000 + 0.1*i);  # 简化模型

# 初始值
let V := F * 0.5;
let L := F * 0.5;
let {i in 1..n_components} x[i] := z[i];
let {i in 1..n_components} y[i] := z[i];

# 如果直接用 Uno 求解
# option solver uno_ampl;
# solve;