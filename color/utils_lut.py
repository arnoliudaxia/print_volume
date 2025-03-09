import pandas as pd
import numpy as np
import colour
import cv2
from tqdm import tqdm
import os
from scipy.optimize import fsolve, least_squares
import json
import torch

K = {}
S = {}
x = 1
z = 0.014

rgb_to_concentration_map_file = 'color/data/color_map/10.csv'
rgb_to_concentration_map = pd.read_csv(rgb_to_concentration_map_file)
rgb_to_concentration_step = 1 / (len(rgb_to_concentration_map)**(1/3) - 1)
rgb_to_concentration_interpolate = round(len(rgb_to_concentration_map)**(1/3) - 1)

brightness_to_kw_concentration_map_file = 'color/data/color_map/brightness_kw_rates.csv'
brightness_to_kw_concentration_map = pd.read_csv(brightness_to_kw_concentration_map_file)
brightness_to_kw_step = 1 / (len(brightness_to_kw_concentration_map) - 1)
brightness_to_kw_interpolate = round(len(brightness_to_kw_concentration_map) - 1)

cmykwe_density_rgb = np.array(json.load(open('color/data/color_calib.json'))['density_rgb'])


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


def concentration_to_rgbd(concentration): #! 造一点数据
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

def rgb_to_concentration(rgb_array):
    
    # 计算 floor 和 ceil
    r_floor = np.floor(rgb_array[:, 0] * 10) / 10
    r_ceil = np.ceil(rgb_array[:, 0] * 10) / 10
    g_floor = np.floor(rgb_array[:, 1] * 10) / 10
    g_ceil = np.ceil(rgb_array[:, 1] * 10) / 10
    b_floor = np.floor(rgb_array[:, 2] * 10) / 10
    b_ceil = np.ceil(rgb_array[:, 2] * 10) / 10

    # 生成所有可能的组合
    nearest_values = np.array([
        [r_floor, g_floor, b_floor],
        [r_floor, g_floor, b_ceil],
        [r_floor, g_ceil, b_floor],
        [r_floor, g_ceil, b_ceil],
        [r_ceil, g_floor, b_floor],
        [r_ceil, g_floor, b_ceil],
        [r_ceil, g_ceil, b_floor],
        [r_ceil, g_ceil, b_ceil]
    ]).transpose(2, 0, 1)

    def trilinear_interpolation(rgb, values):
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        r0, g0, b0 = values[:, 0, 0], values[:, 0, 1], values[:, 0, 2]

        def lerp(v0, v1, t):
            return v0 + t * (v1 - v0)

        def interpolate(c000, c001, c010, c011, c100, c101, c110, c111):
            c00 = lerp(c000, c001, (b - b0) / rgb_to_concentration_step)
            c01 = lerp(c010, c011, (b - b0) / rgb_to_concentration_step)
            c10 = lerp(c100, c101, (b - b0) / rgb_to_concentration_step)
            c11 = lerp(c110, c111, (b - b0) / rgb_to_concentration_step)

            c0 = lerp(c00, c01, (g - g0) / rgb_to_concentration_step)
            c1 = lerp(c10, c11, (g - g0) / rgb_to_concentration_step)
            return lerp(c0, c1, (r - r0) / rgb_to_concentration_step)

        def value_to_index(value):
            return (value[:, 0] // rgb_to_concentration_step + 1) * ((rgb_to_concentration_interpolate+1)**2) + \
                   (value[:, 1] // rgb_to_concentration_step + 1) * (rgb_to_concentration_interpolate+1) + \
                   (value[:, 2] // rgb_to_concentration_step + 1)

        indices = value_to_index(values.reshape(-1, 3)).astype(int)
        c_values = rgb_to_concentration_map.iloc[indices].values[:, 3:12].reshape(-1, 8, 9)


        c_c = interpolate(c_values[:, 0, 0], c_values[:, 1, 0], c_values[:, 2, 0], c_values[:, 3, 0],
                          c_values[:, 4, 0], c_values[:, 5, 0], c_values[:, 6, 0], c_values[:, 7, 0])
        c_m = interpolate(c_values[:, 0, 1], c_values[:, 1, 1], c_values[:, 2, 1], c_values[:, 3, 1],
                          c_values[:, 4, 1], c_values[:, 5, 1], c_values[:, 6, 1], c_values[:, 7, 1])
        c_y = interpolate(c_values[:, 0, 2], c_values[:, 1, 2], c_values[:, 2, 2], c_values[:, 3, 2],
                          c_values[:, 4, 2], c_values[:, 5, 2], c_values[:, 6, 2], c_values[:, 7, 2])
        c_k = interpolate(c_values[:, 0, 3], c_values[:, 1, 3], c_values[:, 2, 3], c_values[:, 3, 3],
                          c_values[:, 4, 3], c_values[:, 5, 3], c_values[:, 6, 3], c_values[:, 7, 3])
        c_w = interpolate(c_values[:, 0, 4], c_values[:, 1, 4], c_values[:, 2, 4], c_values[:, 3, 4],
                          c_values[:, 4, 4], c_values[:, 5, 4], c_values[:, 6, 4], c_values[:, 7, 4])
        p_r = interpolate(c_values[:, 0, 5], c_values[:, 1, 5], c_values[:, 2, 5], c_values[:, 3, 5],
                          c_values[:, 4, 5], c_values[:, 5, 5], c_values[:, 6, 5], c_values[:, 7, 5])
        p_g = interpolate(c_values[:, 0, 6], c_values[:, 1, 6], c_values[:, 2, 6], c_values[:, 3, 6],
                          c_values[:, 4, 6], c_values[:, 5, 6], c_values[:, 6, 6], c_values[:, 7, 6])
        p_b = interpolate(c_values[:, 0, 7], c_values[:, 1, 7], c_values[:, 2, 7], c_values[:, 3, 7],
                          c_values[:, 4, 7], c_values[:, 5, 7], c_values[:, 6, 7], c_values[:, 7, 7])
        p_d = interpolate(c_values[:, 0, 8], c_values[:, 1, 8], c_values[:, 2, 8], c_values[:, 3, 8],
                          c_values[:, 4, 8], c_values[:, 5, 8], c_values[:, 6, 8], c_values[:, 7, 8])

        return np.stack([c_c, c_m, c_y, c_k, c_w, np.zeros_like(c_c)], axis=1), np.stack([p_r, p_g, p_b, p_d], axis=1)

    concentration = trilinear_interpolation(rgb_array, nearest_values)
    return concentration


def rgb_to_concentration_torch(rgb_array):
    # 将输入转换为 PyTorch 张量并移动到 GPU

    # 计算 floor 和 ceil
    r_floor = torch.floor(rgb_array[:, 0] * 10) / 10
    r_ceil = torch.ceil(rgb_array[:, 0] * 10) / 10
    g_floor = torch.floor(rgb_array[:, 1] * 10) / 10
    g_ceil = torch.ceil(rgb_array[:, 1] * 10) / 10
    b_floor = torch.floor(rgb_array[:, 2] * 10) / 10
    b_ceil = torch.ceil(rgb_array[:, 2] * 10) / 10

    # 生成所有可能的组合
    nearest_values = torch.stack([
        torch.stack([r_floor, g_floor, b_floor], dim=1),
        torch.stack([r_floor, g_floor, b_ceil], dim=1),
        torch.stack([r_floor, g_ceil, b_floor], dim=1),
        torch.stack([r_floor, g_ceil, b_ceil], dim=1),
        torch.stack([r_ceil, g_floor, b_floor], dim=1),
        torch.stack([r_ceil, g_floor, b_ceil], dim=1),
        torch.stack([r_ceil, g_ceil, b_floor], dim=1),
        torch.stack([r_ceil, g_ceil, b_ceil], dim=1)
    ], dim=1)

    def trilinear_interpolation(rgb, values):
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        r0, g0, b0 = values[:, 0, 0], values[:, 0, 1], values[:, 0, 2]

        def lerp(v0, v1, t):
            return v0 + t * (v1 - v0)

        def interpolate(c000, c001, c010, c011, c100, c101, c110, c111):
            c00 = lerp(c000, c001, (b - b0) / rgb_to_concentration_step)
            c01 = lerp(c010, c011, (b - b0) / rgb_to_concentration_step)
            c10 = lerp(c100, c101, (b - b0) / rgb_to_concentration_step)
            c11 = lerp(c110, c111, (b - b0) / rgb_to_concentration_step)

            c0 = lerp(c00, c01, (g - g0) / rgb_to_concentration_step)
            c1 = lerp(c10, c11, (g - g0) / rgb_to_concentration_step)
            return lerp(c0, c1, (r - r0) / rgb_to_concentration_step)

        def value_to_index(value):
            return (torch.div(value[:, 0], rgb_to_concentration_step, rounding_mode='floor') + 1) * ((rgb_to_concentration_interpolate+1)**2) + \
                   (torch.div(value[:, 1], rgb_to_concentration_step, rounding_mode='floor') + 1) * (rgb_to_concentration_interpolate+1) + \
                   (torch.div(value[:, 2], rgb_to_concentration_step, rounding_mode='floor') + 1)

        indices = value_to_index(values.reshape(-1, 3)).long()
        c_values = torch.tensor(rgb_to_concentration_map.iloc[indices.cpu()].values[:, 3:12], device='cuda').reshape(-1, 8, 9)

        c_c = interpolate(c_values[:, 0, 0], c_values[:, 1, 0], c_values[:, 2, 0], c_values[:, 3, 0],
                          c_values[:, 4, 0], c_values[:, 5, 0], c_values[:, 6, 0], c_values[:, 7, 0])
        c_m = interpolate(c_values[:, 0, 1], c_values[:, 1, 1], c_values[:, 2, 1], c_values[:, 3, 1],
                          c_values[:, 4, 1], c_values[:, 5, 1], c_values[:, 6, 1], c_values[:, 7, 1])
        c_y = interpolate(c_values[:, 0, 2], c_values[:, 1, 2], c_values[:, 2, 2], c_values[:, 3, 2],
                          c_values[:, 4, 2], c_values[:, 5, 2], c_values[:, 6, 2], c_values[:, 7, 2])
        c_k = interpolate(c_values[:, 0, 3], c_values[:, 1, 3], c_values[:, 2, 3], c_values[:, 3, 3],
                          c_values[:, 4, 3], c_values[:, 5, 3], c_values[:, 6, 3], c_values[:, 7, 3])
        c_w = interpolate(c_values[:, 0, 4], c_values[:, 1, 4], c_values[:, 2, 4], c_values[:, 3, 4],
                          c_values[:, 4, 4], c_values[:, 5, 4], c_values[:, 6, 4], c_values[:, 7, 4])
        p_r = interpolate(c_values[:, 0, 5], c_values[:, 1, 5], c_values[:, 2, 5], c_values[:, 3, 5],
                          c_values[:, 4, 5], c_values[:, 5, 5], c_values[:, 6, 5], c_values[:, 7, 5])
        p_g = interpolate(c_values[:, 0, 6], c_values[:, 1, 6], c_values[:, 2, 6], c_values[:, 3, 6],
                          c_values[:, 4, 6], c_values[:, 5, 6], c_values[:, 6, 6], c_values[:, 7, 6])
        p_b = interpolate(c_values[:, 0, 7], c_values[:, 1, 7], c_values[:, 2, 7], c_values[:, 3, 7],
                          c_values[:, 4, 7], c_values[:, 5, 7], c_values[:, 6, 7], c_values[:, 7, 7])
        p_d = interpolate(c_values[:, 0, 8], c_values[:, 1, 8], c_values[:, 2, 8], c_values[:, 3, 8],
                          c_values[:, 4, 8], c_values[:, 5, 8], c_values[:, 6, 8], c_values[:, 7, 8])

        return torch.stack([c_c, c_m, c_y, c_k, c_w, torch.zeros_like(c_c)], dim=1), torch.stack([p_r, p_g, p_b, p_d], dim=1)

    concentration = trilinear_interpolation(rgb_array, nearest_values)
    return concentration

def adjust_density(concentration, target_rgbd):

    def calculate_kw_rate(brightness):
        brightness_indices = (brightness * brightness_to_kw_interpolate).astype(int)
        k_rate = brightness_to_kw_concentration_map.iloc[brightness_indices]['k_rate'].values
        w_rate = 1 - k_rate
        return k_rate, w_rate

    def concentration_to_density(concentration):
        density_rgb = concentration @ cmykwe_density_rgb
        alpha_rgb = 1 - np.exp(-density_rgb * z)
        alpha_max = np.max(alpha_rgb, axis=1)
        alpha_min = np.min(alpha_rgb, axis=1)
        alpha = alpha_rgb.mean(axis=1)
        density = -np.log(1 - alpha) / z
        return density

    density = concentration_to_density(concentration)



    target_density = target_rgbd[:, 3]
    brightness = target_rgbd[:, :3].mean(axis=1)

    k_rate, w_rate = calculate_kw_rate(brightness)

    kw_density = concentration_to_density(np.column_stack([np.zeros_like(k_rate), np.zeros_like(k_rate), np.zeros_like(k_rate), k_rate, w_rate, np.zeros_like(k_rate)]))

    add_rate = (target_density - density) / (kw_density - density)
    add_rate = np.clip(add_rate, 0, 0.3)  # 0.3 is the max rate, magicnum

    color_rate = target_density / density
    concentration = np.where(
        density[:, None] < target_density[:, None] - 1e-3,
        concentration * (1 - add_rate[:, None]) + np.column_stack([np.zeros_like(k_rate), np.zeros_like(k_rate), np.zeros_like(k_rate), k_rate, w_rate, np.zeros_like(k_rate)]) * add_rate[:, None],
        np.column_stack([concentration[:, :5] * (color_rate)[:, None], 1 - (color_rate)])
    )
    return concentration, concentration_to_density(concentration)



def adjust_density_ablation(concentration, target_rgbd):
    def calculate_kw_rate(brightness):
        brightness_indices = (brightness * brightness_to_kw_interpolate).long()
        k_rate = torch.tensor(brightness_to_kw_concentration_map.iloc[brightness_indices.cpu()]['k_rate'].values, device='cuda')
        w_rate = 1 - k_rate
        return k_rate, w_rate

    def concentration_to_density(concentration):
        density_rgb = torch.matmul(concentration, torch.tensor(cmykwe_density_rgb, device='cuda'))
        alpha_rgb = 1 - torch.exp(-density_rgb * z)
        alpha = alpha_rgb.mean(dim=1)
        density = -torch.log(1 - alpha) / z
        return density

    # 将输入转换为 PyTorch 张量并移动到 GPU

    density = concentration_to_density(concentration)

    target_density = target_rgbd[:, 3]
    brightness = target_rgbd[:, :3].mean(dim=1)

    k_rate, w_rate = calculate_kw_rate(brightness)

    kw_density = concentration_to_density(torch.stack([torch.zeros_like(k_rate), torch.zeros_like(k_rate), torch.zeros_like(k_rate), k_rate, w_rate, torch.zeros_like(k_rate)], dim=1))

    add_rate = (target_density - density) / (kw_density - density)
    # print(add_rate.max(), add_rate.min())
    add_rate = torch.clamp(add_rate, 0, 0)  # 0.3 is the max rate, magicnum

    color_rate = target_density / density
    concentration = torch.where(
        density[:, None] < target_density[:, None],
        concentration * (1 - add_rate[:, None]) + torch.stack([torch.zeros_like(k_rate), torch.zeros_like(k_rate), torch.zeros_like(k_rate), k_rate, w_rate, torch.zeros_like(k_rate)], dim=1) * add_rate[:, None],
        torch.cat([concentration[:, :5] * color_rate[:, None], 1 - color_rate[:, None]], dim=1)
    )
    concentration[:, 0:2] *= 10
    concentration = concentration / concentration.sum(dim=1, keepdim=True)
    return concentration, concentration_to_density(concentration)


def adjust_density_torch(concentration, target_rgbd):
    def calculate_kw_rate(brightness):
        brightness_indices = (brightness * brightness_to_kw_interpolate).long()
        k_rate = torch.tensor(brightness_to_kw_concentration_map.iloc[brightness_indices.cpu()]['k_rate'].values, device='cuda')
        w_rate = 1 - k_rate
        return k_rate, w_rate

    def concentration_to_density(concentration):
        density_rgb = torch.matmul(concentration, torch.tensor(cmykwe_density_rgb, device='cuda'))
        alpha_rgb = 1 - torch.exp(-density_rgb * z)
        alpha = alpha_rgb.mean(dim=1)
        density = -torch.log(1 - alpha) / z
        return density

    # 将输入转换为 PyTorch 张量并移动到 GPU

    density = concentration_to_density(concentration)

    target_density = target_rgbd[:, 3]
    brightness = target_rgbd[:, :3].mean(dim=1)

    k_rate, w_rate = calculate_kw_rate(brightness)

    kw_density = concentration_to_density(torch.stack([torch.zeros_like(k_rate), torch.zeros_like(k_rate), torch.zeros_like(k_rate), k_rate, w_rate, torch.zeros_like(k_rate)], dim=1))

    add_rate = (target_density - density) / (kw_density - density)
    add_rate = torch.clamp(add_rate, 0, 0.3)  # 0.3 is the max rate, magicnum

    color_rate = target_density / density
    concentration = torch.where(
        density[:, None] < target_density[:, None],
        concentration * (1 - add_rate[:, None]) + torch.stack([torch.zeros_like(k_rate), torch.zeros_like(k_rate), torch.zeros_like(k_rate), k_rate, w_rate, torch.zeros_like(k_rate)], dim=1) * add_rate[:, None],
        torch.cat([concentration[:, :5] * color_rate[:, None], 1 - color_rate[:, None]], dim=1)
    )
    return concentration, concentration_to_density(concentration)


def rgbd_to_concentration(rgbd):
    rgb = rgbd[:, :3]
    concentration, pred_rgbd = rgb_to_concentration(rgb)
    concentration, new_density = adjust_density(concentration, rgbd)
    pred_rgbd[:, 3] = new_density
    return concentration, pred_rgbd

def rgbd_to_concentration_torch_ablation(rgbd):
    rgb = rgbd[:, :3]
    concentration, pred_rgbd = rgb_to_concentration_torch(rgb)
    concentration, new_density = adjust_density_ablation(concentration, rgbd)
    pred_rgbd[:, 3] = new_density
    return concentration, pred_rgbd

def rgbd_to_concentration_figure(rgbd):
    rgb = rgbd[:, :3]
    concentration_step1, pred_rgbd = rgb_to_concentration_torch(rgb)
    concentration, new_density = adjust_density_torch(concentration_step1, rgbd)
    pred_rgbd[:, 3] = new_density
    return concentration, pred_rgbd, concentration_step1

def rgbd_to_concentration_torch(rgbd):
    rgb = rgbd[:, :3]
    concentration, pred_rgbd = rgb_to_concentration_torch(rgb)
    concentration, new_density = adjust_density_torch(concentration, rgbd)
    pred_rgbd[:, 3] = new_density
    return concentration, pred_rgbd




if __name__ == "__main__":

    
    # for Brightness in range(1,11):
    #     for i in range(3):
            from tqdm import tqdm  # 导入tqdm库用于进度条

            # rgbs=np.load(f"needPred/colors_{Brightness}_{i}.npy")
            rgbs=np.load(f"colors.npy")
            # rgbs/=255.0
            results_rgb = []
            results_predrgb = []
            with tqdm(total=len(rgbs)*len(rgbs[0]), desc="Processing RGBs") as pbar:
                for figrgs in rgbs:
                    for rgb in figrgs:
                        # rgb = np.array([1, 0, 0]) / 10
                        concentration = rgb_to_concentration_least_square(rgb)
                        pred_rgbd = concentration_to_rgbd(concentration)
        
                        results_rgb.append([
                        [rgb[0],
                            rgb[1],
                            rgb[2],],
                        ])
                        results_predrgb.append([pred_rgbd[0],
                        pred_rgbd[1],
                        pred_rgbd[2],
                        pred_rgbd[3]])
                        
                        pbar.update(1)
                
            # np.save(f"pred_rgbs/results_rgb_{Brightness}_{i}.npy", np.array(results_rgb))
            # np.save(f"pred_rgbs/results_predrgb_{Brightness}_{i}.npy", np.array(results_predrgb))
            
            # np.save(f"pred_rgbs/results_rgb_{Brightness}_{i}.npy", np.array(results_rgb))
            np.save(f"results_predrgb.npy", np.array(results_predrgb))
            np.save(f"results_rgb.npy", np.array(results_rgb))
    # 更新进度条
    # pbar.update(1)

    # # 创建 DataFrame
    # df = pd.DataFrame(results)

    # # 将 DataFrame 写入 CSV 文件
    # df.to_csv('color/data/color_map/10.csv', index=False)


    # results = []

    # for i in range(101):
    #     brightness = i / 100
    #     concentration = brightness_to_kw_concentration_least_square(brightness)
    #     k_rate = concentration[0]
    #     w_rate = concentration[1]
    #     results.append({
    #         'brightness': brightness,
    #         'k_rate': k_rate,
    #         'w_rate': w_rate
    #     })
    #     print(brightness, k_rate, w_rate)

    # # 创建 DataFrame
    # df = pd.DataFrame(results)

    # # 将 DataFrame 写入 CSV 文件
    # df.to_csv('color/data/color_map/brightness_kw_rates.csv', index=False)


# if __name__ == "__main__":
#     # rgbs=np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
#     # r=rgb_to_concentration(rgbs)
#     # breakpoint()
#     # exit(0)

#     # 计算总的迭代次数
#     total_iterations = 11**5

#     c, m, y, k, w = np.meshgrid(np.arange(11), np.arange(11), np.arange(11), np.arange(11), np.arange(11), indexing='ij')
#     combinations = np.stack([c, m, y, k, w], axis=-1).reshape(-1, 5)
#     combinations = combinations.astype(np.float32) * 0.1

#     print("采样表格构建完成")

#     results = []

#     with tqdm(total=total_iterations, desc="Processing CMYKW values") as pbar:
#         for c,m,y,k,w in combinations:
#             pred_rgbd = concentration_to_rgbd(np.array([c,m,y,k,w,0])) 
            
#             results.append({
#                 'pred_r': pred_rgbd[0],
#                 'pred_g': pred_rgbd[1],
#                 'pred_b': pred_rgbd[2],
#                 'c': c,
#                 'm': m,
#                 'y': y,
#                 'k': k,
#             })

#             # 更新进度条
#             pbar.update(1)

#     # 创建 DataFrame
#     df = pd.DataFrame(results)

#     # 将 DataFrame 写入 CSV 文件
#     df.to_csv('colorMapping.csv', index=False)

