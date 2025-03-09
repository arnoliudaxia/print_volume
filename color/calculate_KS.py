import pandas as pd
import numpy as np
from scipy.optimize import fsolve
import os
from tqdm import tqdm

# Change working directory
os.chdir('/media/vrlab/rabbit/print3dingp/color')

# 读取Excel文件中的T页和R页
file_path = 'data/color_calib.xlsx'
sheet_names = ['T', 'D']

result = {'T':{}, 'D':{}}

# 遍历每个工作表
for sheet in sheet_names:
    # 读取工作表
    df = pd.read_excel(file_path, sheet_name=sheet)
    
    # 获取列数
    num_columns = df.shape[1]
    
    # 遍历每两列
    for i in range(0, num_columns, 2):
        # 获取数据组名称
        group_name = df.columns[i]  # 使用列名作为数据组名称
        if "基线" in group_name:
            continue
        
        color_name, depth, ray_type = group_name.split('-')
        depth = float(depth[:-2])
        print(group_name, depth)

        # 获取数据
        data = df.iloc[1:, i:i+2].values
        wave_length = data[:, 0]
        percentage = data[:, 1] / 100

        result[sheet][color_name+'_'+str(depth)] = {
            'Color': color_name,
            'Wavelength': wave_length,
            'percentage': percentage,
            'x': depth,
        }

# 创建保存结果的目录
output_dir = 'data/KS_result_test'
os.makedirs(output_dir, exist_ok=True)

# 计算并保存结果
for color_name in result['D']:
    if color_name not in result['T'].keys():
        print(color_name)
        continue

    wave_length = result['T'][color_name]['Wavelength']
    T_all = result['T'][color_name]['percentage']
    R_all = result['D'][color_name]['percentage']
    T_all = np.clip(T_all, 0, 1)
    R_all = np.clip(R_all, 0, 1)
    x = result['T'][color_name]['x']

    # 用于保存结果的列表
    results_list = []
    
    for i in tqdm(range(len(T_all))):
        R = R_all[i]
        T = T_all[i]
        initial_guesses = [(1e-5, 1e-5), (1e-4, 1e-4), (1e-3, 1e-3), (1e-2, 1e-2), (1e-1, 1e-1), (1e0, 1e0), (1e1, 1e1), (1e2, 1e2), (1e3, 1e3), (1e4, 1e4), (1e5, 1e5)]  # 初始化多组K0和S0
        best_solution = None
        best_error = float('inf')
        
        def equations(vars):
            K, S = vars
            a = (S + K) / S
            b = np.sqrt(a**2 - 1)
            sinh_term = np.sinh(b * S * x)
            cosh_term = np.cosh(b * S * x)
            
            eq1 = sinh_term / (a * sinh_term + b * cosh_term) - R
            eq2 = b / (a * sinh_term + b * cosh_term) - T
            return [eq1, eq2]
        
        for K0, S0 in initial_guesses:
            solution = fsolve(equations, (K0, S0))
            error = np.sum(np.abs(equations(solution)))
            if error < best_error:
                best_error = error
                best_solution = solution
        
        K, S = best_solution
        if best_error > 1e-3:
            print(color_name, x, R, T, K, S)
            print(equations(best_solution))

        # 将结果添加到列表中
        results_list.append([result['T'][color_name]['Color'], wave_length[i], R, T, x, K, S, best_error])
    
    # 将结果转换为DataFrame并保存为CSV
    df_results = pd.DataFrame(results_list, columns=['Color', 'Wavelength', 'R', 'T', 'x', 'K', 'S', 'error'])
    output_file = os.path.join(output_dir, f"{color_name}.csv")
    df_results.to_csv(output_file, index=False)