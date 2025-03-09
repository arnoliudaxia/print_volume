import numpy as np
import pandas as pd
import colour
import cv2
from tqdm import tqdm

def KMmodel(K, S, x):
    a = (S + K) / S
    b = np.sqrt(a**2 - 1)
    sinh_term = np.sinh(b * S * x)
    cosh_term = np.cosh(b * S * x)
    
    R = sinh_term / (a * sinh_term + b * cosh_term)
    T = b / (a * sinh_term + b * cosh_term)
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


def visualize_color_pair(line):
    # 创建一个100x700的白色图像
    image = np.ones((100, 500, 3), dtype=np.uint8) * 255
    
    # 定义颜色
    colors = {
        'c': (255, 255, 0),  # Cyan
        'm': (255, 0, 255),  # Magenta
        'y': (0, 255, 255),  # Yellow
        'k': (0, 0, 0),      # Black
        'w': (255, 255, 255) # White
    }
    
    # 绘制左侧子图，尺寸为200x100
    left_image = np.ones((100, 200, 3), dtype=np.uint8) * 255
    current_x = 0
    for i, color in enumerate(['c', 'm', 'y', 'k', 'w']):
        width = int(line[i] * 200)  # 根据权重计算宽度
        cv2.rectangle(left_image, (current_x, 0), (current_x + width, 100), colors[color][::-1], -1)
        current_x += width
    
    # 将左侧子图放入主图像
    image[:, :200] = left_image
    
    # 绘制R_rgb和T_rgb子图
    for i, rgb in enumerate([line[5:8], line[8:11], line[11:14]]):
        # 计算子图的左上角坐标
        top_left = ((i + 2) * 100, 0)
        bottom_right = ((i + 3) * 100, 100)
        
        # 将RGB值转换为0-255范围
        rgb_color = tuple(int(c * 255) for c in rgb)
        
        # 绘制子图
        cv2.rectangle(image, top_left, bottom_right, rgb_color, -1)
    
    return image


def generate_weights():
    """
    生成一个5维度的权重数组，范围在0-1之间，总和为1。
    具有10%的概率不加入白色，10%的概率不加入黑色。
    另外，有30%的概率限制黑色和白色的权重最大值为0.1。
    """
    weights = np.random.dirichlet(np.ones(5), size=1)[0]
    
    # 10%的概率不加入白色
    # if np.random.rand() < 0.1:
    #     weights[4] = 0  # 将白色的权重设置为0
    #     weights /= weights.sum()  # 重新归一化权重
    
    # # 10%的概率不加入黑色
    # if np.random.rand() < 0.1:
    #     weights[3] = 0  # 将黑色的权重设置为0
    #     weights /= weights.sum()  # 重新归一化权重
    
    # # 30%的概率限制黑色和白色的权重最大值为0.1
    # if np.random.rand() < 0.3:
    #     weights[3] = min(weights[3], 0.1)  # 限制黑色权重
    #     weights[4] = min(weights[4], 0.1)  # 限制白色权重
    #     weights /= weights.sum()  # 重新归一化权重
    print(weights)
    
    return weights


if __name__ == '__main__':
    data_num = 10
    
    images = []
    K = {}
    S = {}
    x = 1
    z = 0.014
    
    for color in ['c', 'm', 'y', 'k', 'w']:
        data = pd.read_csv(f'./color/data/calib_data/{color}.csv')
        if color == 'c':
            wavelength = data['Wavelength'].values
        K[color], S[color] = data['K'].values, data['S'].values

    results = []

    for i in tqdm(range(data_num)):
        # 使用封装的函数生成权重
        weights = generate_weights()
        # weights = [0, 0, i / 10, 0, 1 - i / 10]
        # 计算K_blend和S_blend
        K_blend = sum(weights[j] * K[color] for j, color in enumerate(['c', 'm', 'y', 'k', 'w']))
        S_blend = sum(weights[j] * S[color] for j, color in enumerate(['c', 'm', 'y', 'k', 'w']))
        K_S = K_blend / S_blend

        R_blend, T_blend = KMmodel(K_blend, S_blend, x)

        C_blend = R_blend + T_blend

        R_rgb, T_rgb = calculate_rgb(wavelength, R_blend), calculate_rgb(wavelength, T_blend)
        C_rgb = calculate_rgb(wavelength, C_blend)


        _, T_alpha = KMmodel(K_blend, S_blend, z)
        T_alpha_rgb = calculate_rgb(wavelength, T_alpha)
        T_alpha = T_alpha_rgb.mean()
        density = -np.log(T_alpha) / z

        # 将weights和R_rgb, T_rgb组合成一个列表并添加到results中
        results.append(list(weights) + list(R_rgb) + list(T_rgb) + list(C_rgb) + [density])



        image = visualize_color_pair(results[-1])
        images.append(image)

    # 将结果保存为CSV文件
    # df = pd.DataFrame(results, columns=['W_c', 'W_m', 'W_y', 'W_k', 'W_w', 'R_r', 'R_g', 'R_b', 'T_r', 'T_g', 'T_b', 'C_r', 'C_g', 'C_b', 'density'])
    # df.to_csv(f'./data/color_pair_data/{data_num}.csv', index=False)

    image_all = np.concatenate(images, axis=0)
    cv2.imwrite(f'./temp.png', image_all[:, :, ::-1])
        