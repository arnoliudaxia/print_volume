import pandas as pd
import numpy as np
import colour
import cv2
from tqdm import tqdm
import os
from scipy.optimize import fsolve, least_squares
import json
import cProfile
import pstats
from pstats import SortKey
import multiprocessing as mp
os.chdir('/media/vrlab/rabbit/print3dingp/print_volume')
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['SCIPY_NUM_THREADS'] = '1'

K = {}
S = {}
x = 1
z = 0.014

for color in ['c', 'm', 'y', 'k', 'w']:
    data = pd.read_csv(f'./color/data/calib_data/{color}.csv')
    if color == 'c':
        wavelength = data['Wavelength'].values
    K[color], S[color] = data['K'].values, data['S'].values



def KMmodel(K, S, x):
    a = (S + K) / S
    b = np.sqrt(a**2 - 1)
    sinh_term = np.sinh(b * S * x)
    cosh_term = np.cosh(b * S * x)
    
    R = sinh_term / (a * sinh_term + b * cosh_term)
    T = b / (a * sinh_term + b * cosh_term)
    R[np.isnan(R)] = 0
    T[np.isnan(T)] = 0
    return R, T

def calculate_rgb(wavelengths, spectrum):
    """
    Calculate RGB values from spectrum data using the colour package
    """
    # Create a SpectralDistribution object
    spd = colour.SpectralDistribution(
        dict(zip(wavelengths, spectrum)), 
        name="Sample SPD"
    )
    
    # Get CIE 1931 2-degree standard observer
    cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    
    # Use D65 illuminant
    illuminant = colour.SDS_ILLUMINANTS['D65']
    
    # Convert to XYZ using explicit parameters and integration method
    XYZ = colour.sd_to_XYZ(
        spd,
        cmfs=cmfs,
        illuminant=illuminant,
        method='Integration',
        shape=colour.SpectralShape(300, 800, 1)  # Specify our wavelength range
    )
    
    # Convert XYZ to sRGB
    rgb = colour.XYZ_to_sRGB(XYZ / 100.0)  # Divide by 100 to convert from percentage
    
    # Ensure values are between 0 and 1
    rgb = np.clip(rgb, 0, 1)
    
    return rgb

def concentration_to_rgbd(concentration):
    '''
    concentration: [c, m, y, k, w, clear]
    '''
    concentration = concentration / concentration.sum()
    K_blend = sum(concentration[j] * K[color] for j, color in enumerate(['c', 'm', 'y', 'k', 'w']))
    S_blend = sum(concentration[j] * S[color] for j, color in enumerate(['c', 'm', 'y', 'k', 'w']))
    R_blend, T_blend = KMmodel(K_blend, S_blend, x)
    C_blend = R_blend + T_blend
    C_rgb = calculate_rgb(wavelength, C_blend)

    _, T_alpha = KMmodel(K_blend, S_blend, z)
    T_alpha_rgb = calculate_rgb(wavelength, T_alpha)
    T_alpha = T_alpha_rgb.mean()
    density = -np.log(T_alpha) / z

    density = density * (1 - concentration[5])

    return np.concatenate([C_rgb, [density]])

def rgb_to_concentration_least_square(target_rgb):
    def objective_function(concentration):
        # 计算当前浓度下的 RGB
        concentration = np.array([concentration[0], concentration[1], concentration[2], concentration[3], concentration[4], 0])
        concentration = concentration / concentration.sum()
        predicted_rgb = concentration_to_rgbd(concentration)[:3]  # 只取 RGB 部分
        # 返回目标 RGB 和预测 RGB 之间的差异
        return predicted_rgb - target_rgb

    # 初始猜测浓度值
    initial_concentration = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    # 使用最小二乘法进行优化
    result = least_squares(objective_function, initial_concentration, bounds=(0, 1))
    result.x = result.x / result.x.sum()
    # 返回优化后的浓度值
    concentration = np.array([result.x[0], result.x[1], result.x[2], result.x[3], result.x[4], 0])
    return concentration

def brightness_to_kw_concentration_least_square(target_brightness):
    def objective_function(concentration):

        concentration = np.array([0, 0, 0, concentration[0], concentration[1], 0])
        concentration = concentration / concentration.sum()
        predicted_rgb = concentration_to_rgbd(concentration)[:3]  # 只取 RGB 部分
        predicted_brightness = predicted_rgb.mean()
        # 返回目标 RGB 和预测 RGB 之间的差异
        return predicted_brightness - target_brightness
    
    initial_concentration = np.array([0.5, 0.5])
    result = least_squares(objective_function, initial_concentration, bounds=(0, 1))
    result.x = result.x / result.x.sum()
    # 返回优化后的浓度值
    concentration = np.array([result.x[0], result.x[1]])
    return concentration

def process_single_rgb(rgb_values):
    r, g, b = rgb_values
    rgb = np.array([r, g, b]) / 10
    concentration = rgb_to_concentration_least_square(rgb)
    pred_rgbd = concentration_to_rgbd(concentration)
    
    return {
        'r': rgb[0], 'g': rgb[1], 'b': rgb[2],
        'c_c': concentration[0], 'c_m': concentration[1],
        'c_y': concentration[2], 'c_k': concentration[3],
        'c_w': concentration[4],
        'pred_r': pred_rgbd[0], 'pred_g': pred_rgbd[1],
        'pred_b': pred_rgbd[2], 'pred_d': pred_rgbd[3]
    }

if __name__ == "__main__":
    # Force 'spawn' method instead of the default 'fork' on Linux
    mp.set_start_method('spawn')

    os.makedirs('color/data/color_map', exist_ok=True)

    results = []

    # 计算总的迭代次数
    total_iterations = 11 * 11 * 11

    # rgb = np.array([1, 1, 1])
    # concentration = rgb_to_concentration(rgb)
    # pred_rgbd = concentration_to_rgbd(concentration)
    # print(concentration)
    # print(pred_rgbd)
    # exit()

    # Create all RGB combinations
    rgb_combinations = [(r, g, b) 
                       for r in range(11) 
                       for g in range(11) 
                       for b in range(11)]

    # Process in parallel
    with mp.Pool() as pool:
        results = list(tqdm(
            pool.imap(process_single_rgb, rgb_combinations),
            total=len(rgb_combinations),
            desc="Processing RGB values"
        ))

    # Create DataFrame and save as before
    df = pd.DataFrame(results)
    df.to_csv('./color/data/color_map/10_test.csv', index=False)


    results = []

    for i in range(101):
        brightness = i / 100
        concentration = brightness_to_kw_concentration_least_square(brightness)
        k_rate = concentration[0]
        w_rate = concentration[1]
        results.append({
            'brightness': brightness,
            'k_rate': k_rate,
            'w_rate': w_rate
        })
        # print(brightness, k_rate, w_rate)

    # 创建 DataFrame
    df = pd.DataFrame(results)

    # 将 DataFrame 写入 CSV 文件
    df.to_csv('data/color_map/brightness_kw_rates.csv', index=False)


    # with tqdm(total=total_iterations, desc="Processing RGB values") as pbar:
    #     for r in range(33):
    #         for g in range(33):
    #             for b in range(33):
    #                 rgb = np.array([r, g, b]) / 32
    #                 concentration = rgb_to_concentration_least_square(rgb)
    #                 pred_rgbd = concentration_to_rgbd(concentration)
                    
    #                 # 将结果添加到列表中
    #                 results.append({
    #                     'r': rgb[0],
    #                     'g': rgb[1],
    #                     'b': rgb[2],
    #                     'c_c': concentration[0],
    #                     'c_m': concentration[1],
    #                     'c_y': concentration[2],
    #                     'c_k': concentration[3],
    #                     'c_w': concentration[4],
    #                     'pred_r': pred_rgbd[0],
    #                     'pred_g': pred_rgbd[1],
    #                     'pred_b': pred_rgbd[2],
    #                     'pred_d': pred_rgbd[3]
    #                 })

    #                 # 更新进度条
    #                 pbar.update(1)

    # # 创建 DataFrame
    # df = pd.DataFrame(results)

    # # 将 DataFrame 写入 CSV 文件
    # df.to_csv('data/color_map/32.csv', index=False)

