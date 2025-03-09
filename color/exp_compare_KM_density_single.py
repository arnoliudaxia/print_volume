import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import pandas as pd
K_dict, S_dict = {}, {}


os.makedirs('../workspace/experiment/density_error_sheet', exist_ok=True)
    
K = 0.09
S = 5.4

a = (S + K) / S
b = np.sqrt(a**2 - 1)

# 定义T(λ)函数
def T(t):
    return b / (a * np.sinh(b * S  * t) + b * np.cosh(b * S * t))

def exp_func(t, density):
    return np.exp(-density * t)

# 定义t的范围
t_values = np.linspace(1e-5, 200, 400)

# 计算T(λ)的值
T_values = T(t_values)

popt, _ = curve_fit(exp_func, t_values, T_values)
fitted_density = popt[0]
# 绘制曲线
difference = T_values - exp_func(t_values, fitted_density)

# 将结果添加到列表中
print("fitted_density: ", fitted_density, "difference.max(): ", difference.max(), "K: ", K, "S: ", S)

    # 将结果转换为DataFrame并保存为CSV

# 绘制T(λ)和拟合的指数函数
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(t_values, T_values, label='T(λ) vs t')
plt.plot(t_values, exp_func(t_values, fitted_density), label=f'Fitted exp(-density*t), density={fitted_density:.4f}')
plt.xlabel('t')
plt.ylabel('Value')
plt.title('T(λ) vs t and Fitted Exponential')
plt.legend()
plt.grid(True)

# 绘制差值
plt.subplot(1, 2, 2)
plt.plot(t_values, difference, label='Difference (T - exp)', color='red')
plt.xlabel('t')
plt.ylabel('Difference')
plt.title('Difference between T(λ) and Fitted Exponential')
plt.legend()
plt.grid(True)

# 保存图像
plt.tight_layout()
plt.savefig('temp.png')