
import numpy as np
import pandas as pd
import colour
import cv2


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

def save_rgb_as_image(rgb_values, filename):
    # 创建一个20x20的图像
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    
    # 将RGB值从[0, 1]转换为[0, 255]
    rgb_255 = (rgb_values * 255).astype(np.uint8)
    
    # 用RGB颜色填充图像
    image[:] = rgb_255
    
    # 只在filename不为空时保存图像
    if filename:
        cv2.imwrite(filename, image[:, :, ::-1])
    return image

def create_combined_image(images, col_num=5):
    # Calculate the size of the combined image
    grid_size = (col_num, len(images)//col_num)
    single_height, single_width = images[0].shape[:2]
    combined_height = single_height * grid_size[0]
    combined_width = single_width * grid_size[1]
    
    # Create an empty image for the combined result
    combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
    
    # Place each image in the grid
    for idx, image in enumerate(images):
        row = idx // grid_size[1]
        col = idx % grid_size[1]
        start_y = row * single_height
        start_x = col * single_width
        combined_image[start_y:start_y + single_height, start_x:start_x + single_width] = image
    
    return combined_image

if __name__ == '__main__':
    images = []
    for color in ['c', 'm', 'y', 'k', 'w', 'e']:
        data = pd.read_csv(f'./color/data/calib_data/{color}.csv')
        wavelength = data['Wavelength'].values
        K, S = data['K'].values, data['S'].values
        x = 0.001
        R, T = KMmodel(K, S, x)

        rgb_R = calculate_rgb(wavelength, R)
        rgb_T = calculate_rgb(wavelength, T)


        # print(rgb_R, rgb_T)
        
        density = -np.log(rgb_T) / x
        print(color, density)
