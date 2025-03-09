import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import pandas as pd
K_dict, S_dict = {}, {}


os.makedirs('../workspace/experiment/density_error_sheet', exist_ok=True)
for color in ['w', 'c', 'm', 'y', 'k']:
    data = pd.read_csv(f'./color/data/calib_data/{color}.csv')
    wavelength = data['Wavelength'].values
    K_dict[color], S_dict[color] = data['K'].values, data['S'].values

    results = []  # 用于存储结果
    
    for i, lambda_ in enumerate(wavelength):
        K, S = K_dict[color][i], S_dict[color][i]
        # 计算a和b
        a = (S + K) / S
        b = np.sqrt(a**2 - 1)

        # 定义T(λ)函数
        def T(t):
            return b / (a * np.sinh(b * S  * t) + b * np.cosh(b * S * t))

        def exp_func(t, density):
            return np.exp(-density * t)

        # 定义t的范围
        t_values = np.linspace(1e-5, 5, 400)

        # 计算T(λ)的值
        T_values = T(t_values)

        popt, _ = curve_fit(exp_func, t_values, T_values)
        fitted_density = popt[0]
        # 绘制曲线
        difference = T_values - exp_func(t_values, fitted_density)

        # 将结果添加到列表中
        results.append([color, lambda_, fitted_density, np.abs(difference).max(), np.abs(difference).mean(), K, S])

    # 将结果转换为DataFrame并保存为CSV
    df_results = pd.DataFrame(results, columns=['color', 'lambda_', 'fitted_density', 'difference_max', 'difference_mean', 'K', 'S'])
    df_results.to_csv(f'../workspace/experiment/density_error_sheet/{color}.csv', index=False)
    
    # 计算difference_mean的统计量
    difference_means = [result[4] for result in results]  # 提取difference_mean列
    mean_of_difference_mean = np.mean(difference_means)
    std_of_difference_mean = np.std(difference_means)
    max_of_difference_mean = np.max(difference_means)

    print(f"Color: {color}, Mean of difference_mean: {mean_of_difference_mean}, "
          f"Std of difference_mean: {std_of_difference_mean}, "
          f"Max of difference_mean: {max_of_difference_mean}")

    # # 绘制T(λ)和拟合的指数函数
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(t_values, T_values, label='T(λ) vs t')
    # plt.plot(t_values, exp_func(t_values, fitted_density), label=f'Fitted exp(-density*t), density={fitted_density:.4f}')
    # plt.xlabel('t')
    # plt.ylabel('Value')
    # plt.title('T(λ) vs t and Fitted Exponential')
    # plt.legend()
    # plt.grid(True)

    # # 绘制差值
    # plt.subplot(1, 2, 2)
    # plt.plot(t_values, difference, label='Difference (T - exp)', color='red')
    # plt.xlabel('t')
    # plt.ylabel('Difference')
    # plt.title('Difference between T(λ) and Fitted Exponential')
    # plt.legend()
    # plt.grid(True)

    # # 保存图像
    # plt.tight_layout()
    # plt.savefig('temp.png')