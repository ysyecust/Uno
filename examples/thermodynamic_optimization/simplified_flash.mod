# simplified_flash.mod
# 简化的闪蒸计算（不使用外部函数，用于测试）

# 参数
param n_components := 3;
param T := 300;
param P := 1.0;  # bar
param F := 100;  # 总进料

# 进料组成
param z{i in 1..n_components};

# 组分性质（简化）
param Tc{i in 1..n_components};  # 临界温度
param Pc{i in 1..n_components};  # 临界压力
param omega{i in 1..n_components};  # 偏心因子

data;
param z :=
  1  0.3
  2  0.5
  3  0.2;

param Tc :=
  1  305.3
  2  369.8
  3  425.1;

param Pc :=
  1  48.7
  2  42.5
  3  38.0;

param omega :=
  1  0.099
  2  0.152
  3  0.201;

# 变量
var V >= 0, <= F;
var L >= 0, <= F;
var x{i in 1..n_components} >= 1e-6, <= 1;
var y{i in 1..n_components} >= 1e-6, <= 1;

# 简化的 K 值计算（Wilson 关联式）
param K{i in 1..n_components} = 
    (Pc[i]/P) * exp(5.373 * (1 + omega[i]) * (1 - Tc[i]/T));

# 目标：最小化 Rachford-Rice 函数的平方
minimize rachford_rice:
    (sum{i in 1..n_components} (z[i] * (K[i] - 1) / (1 + V/F * (K[i] - 1))))^2;

# 约束
subject to material_balance:
    L + V = F;

subject to component_balance{i in 1..n_components}:
    F * z[i] = L * x[i] + V * y[i];

subject to liquid_sum:
    sum{i in 1..n_components} x[i] = 1;

subject to vapor_sum:
    sum{i in 1..n_components} y[i] = 1;

subject to equilibrium{i in 1..n_components}:
    y[i] = K[i] * x[i];

# 初始值
let V := F * 0.5;
let L := F * 0.5;
let {i in 1..n_components} x[i] := z[i];
let {i in 1..n_components} y[i] := z[i];