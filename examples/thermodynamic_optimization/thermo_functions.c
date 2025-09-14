/* thermo_functions.c - 纯 C 版本的热力学函数库 */

/* 重要：必须先包含 funcadd.h，然后取消宏定义，再包含标准库 */
#include "funcadd.h"

/* 取消会冲突的宏定义 */
#ifdef getenv
#undef getenv
#endif
#ifdef strtod
#undef strtod
#endif
#ifdef printf
#undef printf
#endif
#ifdef fprintf
#undef fprintf
#endif
#ifdef sprintf
#undef sprintf
#endif
#ifdef snprintf
#undef snprintf
#endif
#ifdef vfprintf
#undef vfprintf
#endif
#ifdef vsprintf
#undef vsprintf
#endif
#ifdef vsnprintf
#undef vsnprintf
#endif

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* 辅助函数：计算活度系数 */
static double activity_coefficient_calc(double T, double x1, double x2, double A12) {
    /* Wilson 方程（简化） */
    return exp(-log(x1 + x2 * exp(-A12/T)) + 
               x2 * (A12/T) / (x1 + x2 * exp(-A12/T)));
}

/* 模拟的逸度计算函数 */
static double calculate_fugacity_internal(double T, double P, const double* x, int n) {
    /* 气体常数 */
    const double R = 8.314;  /* J/(mol·K) */
    
    /* 示例：简化的逸度系数计算 */
    double activity = 0.0;
    int i;
    
    for (i = 0; i < n; ++i) {
        /* 模拟活度系数 */
        activity += x[i] * log(x[i] + 1e-10);  /* 避免 log(0) */
    }
    
    /* 简化的逸度计算 */
    double Z = 1.0 - 0.1 * P / (R * T);  /* 压缩因子 */
    double phi = exp((Z - 1) - log(Z));  /* 逸度系数 */
    
    return P * phi * exp(activity);
}

/* AMPL 外部函数：液相逸度 */
real fugacity_liquid(arglist *al) {
    int n = al->n;
    real *args = al->ra;
    double *x;
    int i;
    double T, P, fug;
    double h = 1e-7;
    
    if (n < 3) {
        al->Errmsg = "fugacity_liquid requires at least 3 arguments: T, P, x1, [x2, ...]";
        return 0;
    }
    
    T = args[0];  /* 温度 [K] */
    P = args[1];  /* 压力 [Pa] */
    
    /* 分配摩尔分率数组 */
    x = (double*)malloc((n - 2) * sizeof(double));
    if (!x) {
        al->Errmsg = "Memory allocation failed";
        return 0;
    }
    
    /* 提取摩尔分率 */
    for (i = 2; i < n; ++i) {
        x[i - 2] = args[i];
    }
    
    /* 计算逸度 */
    fug = calculate_fugacity_internal(T, P, x, n - 2);
    
    /* 计算导数（如果需要） */
    if (al->derivs) {
        /* ∂f/∂T */
        al->derivs[0] = (calculate_fugacity_internal(T + h, P, x, n - 2) - 
                        calculate_fugacity_internal(T - h, P, x, n - 2)) / (2 * h);
        
        /* ∂f/∂P */
        al->derivs[1] = (calculate_fugacity_internal(T, P + h, x, n - 2) - 
                        calculate_fugacity_internal(T, P - h, x, n - 2)) / (2 * h);
        
        /* ∂f/∂xi */
        for (i = 0; i < n - 2; ++i) {
            double xi_save = x[i];
            x[i] = xi_save + h;
            double f_plus = calculate_fugacity_internal(T, P, x, n - 2);
            x[i] = xi_save - h;
            double f_minus = calculate_fugacity_internal(T, P, x, n - 2);
            x[i] = xi_save;  /* 恢复原值 */
            al->derivs[2 + i] = (f_plus - f_minus) / (2 * h);
        }
    }
    
    free(x);
    return fug;
}

/* 气相逸度 */
real fugacity_vapor(arglist *al) {
    /* 简化实现：调用液相函数并乘以系数 */
    double result = fugacity_liquid(al);
    return result * 1.1;  /* 简化示例 */
}

/* 活度系数（用于液相） */
real activity_coefficient(arglist *al) {
    double T, x1, x2, A12, gamma1;
    double h = 1e-7;
    
    if (al->n != 4) {
        al->Errmsg = "activity_coefficient requires 4 arguments: T, x1, x2, interaction_param";
        return 0;
    }
    
    T = al->ra[0];
    x1 = al->ra[1];
    x2 = al->ra[2];
    A12 = al->ra[3];  /* 二元交互参数 */
    
    /* Wilson 方程 */
    gamma1 = activity_coefficient_calc(T, x1, x2, A12);
    
    if (al->derivs) {
        /* 数值微分 */
        al->derivs[0] = (activity_coefficient_calc(T+h, x1, x2, A12) - 
                        activity_coefficient_calc(T-h, x1, x2, A12)) / (2*h);
        al->derivs[1] = (activity_coefficient_calc(T, x1+h, x2, A12) - 
                        activity_coefficient_calc(T, x1-h, x2, A12)) / (2*h);
        al->derivs[2] = (activity_coefficient_calc(T, x1, x2+h, A12) - 
                        activity_coefficient_calc(T, x1, x2-h, A12)) / (2*h);
        al->derivs[3] = (activity_coefficient_calc(T, x1, x2, A12+h) - 
                        activity_coefficient_calc(T, x1, x2, A12-h)) / (2*h);
    }
    
    return gamma1;
}

/* 混合焓 */
real mixing_enthalpy(arglist *al) {
    int n_comp, i, j;
    double H_mix = 0.0;
    
    if (al->n < 3) {
        al->Errmsg = "mixing_enthalpy requires at least 3 arguments";
        return 0;
    }
    
    /* double T = al->ra[0];  未使用 */
    /* double P = al->ra[1];  未使用 */
    n_comp = al->n - 2;
    
    for (i = 0; i < n_comp; ++i) {
        for (j = i+1; j < n_comp; ++j) {
            /* 简化的混合焓模型 */
            H_mix += al->ra[2+i] * al->ra[2+j] * 1000.0 * (i-j) * (i-j);
        }
    }
    
    return H_mix;
}

/* 注册所有函数 */
void funcadd(AmplExports *ae) {
    /* -3 表示至少3个参数 */
    addfunc("fugacity_liquid", (ufunc*)fugacity_liquid, 0, -3, ae);
    addfunc("fugacity_vapor", (ufunc*)fugacity_vapor, 0, -3, ae);
    addfunc("activity_coefficient", (ufunc*)activity_coefficient, 0, 4, ae);
    addfunc("mixing_enthalpy", (ufunc*)mixing_enthalpy, 0, -3, ae);
}