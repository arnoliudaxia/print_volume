import argparse
import os
from glob import glob
import cv2
import numpy as np
from tqdm import tqdm
import random

import json
from scipy.ndimage import convolve

import torch
import torch.nn.functional as F

from color.utils import rgbd_to_concentration

kw = 3.93 # 测量值
kk = 3.24 # 测量值
dz = 0.014 # 层高

pure_cmykw_colors = [
    [1, 0, 0, 0, 0],  # 纯青色
    [0, 1, 0, 0, 0],  # 纯品红
    [0, 0, 1, 0, 0],  # 纯黄色
    [0, 0, 0, 1, 0],  # 纯黑色
    [0, 0, 0, 0, 1],  # 纯白色
]

cmykw_save_color = [
    [0, 1, 1, 1],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [0, 0, 0, 1],
    [1, 1, 1, 1],
]


def calculate_concentration(image):
    # 将图像从 (W, H, 4) 转换为 (-1, 4)
    reshaped_image = image.reshape(-1, 4)
    
    # 计算浓度
    concentration, pred_rgbd = rgbd_to_concentration(reshaped_image)
    
    # 将结果从 (WH, 6) 重新 reshape 为 (W, H, 6)
    concentration_image = concentration.reshape(image.shape[0], image.shape[1], 6)
    pred_rgbd_image = pred_rgbd.reshape(image.shape[0], image.shape[1], 4)

    return concentration_image, pred_rgbd_image

def concentration_to_color(concentration):
    # 根据浓度计算颜色

    x_size, y_size = concentration.shape[0], concentration.shape[1]
    print_images = np.zeros((x_size, y_size, 4), dtype=np.float32)
    index_images = np.zeros((x_size, y_size), dtype=np.float32)
    c_concentration, m_concentration, y_concentration, k_concentration, w_concentration = concentration[:, :, 0], concentration[:, :, 1], concentration[:, :, 2], concentration[:, :, 3], concentration[:, :, 4]
    color_random = np.random.rand(x_size, y_size)
    c_mask = (color_random < c_concentration) 
    m_mask = (c_concentration <= color_random) & (color_random < c_concentration + m_concentration) 
    y_mask = (c_concentration + m_concentration <= color_random) & (color_random < c_concentration + m_concentration + y_concentration) 
    k_mask = (c_concentration + m_concentration + y_concentration <= color_random) & (color_random < c_concentration + m_concentration + y_concentration + k_concentration) 
    w_mask = (c_concentration + m_concentration + y_concentration + k_concentration <= color_random) & (color_random < c_concentration + m_concentration + y_concentration + k_concentration + w_concentration) 
    print_images[c_mask] = np.array(cmykw_save_color[0])
    print_images[m_mask] = np.array(cmykw_save_color[1])
    print_images[y_mask] = np.array(cmykw_save_color[2])
    print_images[k_mask] = np.array(cmykw_save_color[3])
    print_images[w_mask] = np.array(cmykw_save_color[4])
    index_images[:] = 5
    index_images[c_mask] = 0
    index_images[m_mask] = 1
    index_images[y_mask] = 2
    index_images[k_mask] = 3
    index_images[w_mask] = 4

    return print_images, index_images
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default='workspace/lego/volume/ngp_50')
    parser.add_argument('--output_folder', type=str, default='workspace/lego/print_volume/ngp50')
    opt = parser.parse_args()


    os.makedirs(os.path.join(opt.output_folder, 'print'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_folder, 'index'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_folder, 'concentration'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_folder, 'pred_rgbd'), exist_ok=True)

    image_paths = glob(os.path.join(opt.input_folder, 'array/*.npy'))
    
    with open(os.path.join(opt.input_folder, 'params.json'), "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    
    delta = data['delta']
    
    
    # 排序确保按照文件名的顺序读取图片
    image_paths = sorted(image_paths)
    
    print("Loading Images...")

    volume_imgs = []

    for path in tqdm(image_paths):
        # 读取图片
        img = np.load(path)
        img[:, :, 0:3] = cv2.cvtColor(img[:, :, 0:3],  cv2.COLOR_RGB2BGR)

        volume_imgs.append(img)
        if img is None:
            print(f"Warning: {path} could not be loaded.")
            continue
        
    volume_imgs = np.array(volume_imgs)

    # 将 NumPy 数组转换为 PyTorch 张量，并移动到 GPU
    volume_imgs = torch.tensor(volume_imgs, device='cpu')

    threshold = 100
    k_z, k_x, k_y = 12, 2, 4
    edge_adjust = [0 if k % 2 == 1 else 1 for k in [k_z, k_x, k_y]]

    batch_size = 32
    num_images = volume_imgs.shape[0]
    padded_volume_imgs = []

    for start_idx in tqdm(range(0, num_images, batch_size)):
        end_idx = min(start_idx + batch_size, num_images)
        
        # 确定当前批次的范围
        batch_imgs = volume_imgs[max(0, start_idx - k_z//2):min(num_images, end_idx + k_z//2)]
        z_pad = [min(k_z//2, start_idx), min(k_z//2, num_images - end_idx)]
        
        # 将 NumPy 数组转换为 PyTorch 张量，并移动到 GPU
        batch_imgs = batch_imgs.to(device='cuda')

        # 处理当前批次
        volume_density = batch_imgs[:, :, :, 3]
        volume_color = batch_imgs[:, :, :, 0:3].clone()
        volume_density_cliped = torch.clamp(volume_density, max=threshold)
        volume_density_residual = volume_density - volume_density_cliped

        blend_volume_imgs = batch_imgs.clone()
        blend_volume_imgs[:, :, :, 3] = volume_density_residual
        blend_volume_imgs[:, :, :, 0:3] = batch_imgs[:, :, :, 0:3]
        blend_volume_imgs[:, :, :, 0:3] = blend_volume_imgs[:, :, :, 0:3] * blend_volume_imgs[:, :, :, [3]]

        # 定义三维卷积核
        kernel = torch.ones((1, 1, k_z, k_x, k_y), device='cuda') / (k_z * k_x * k_y)
        
        # 对 blend_volume_imgs 进行三维卷积
        blend_volume_imgs = blend_volume_imgs.permute(3, 0, 1, 2).unsqueeze(0)  # 调整维度以适应卷积操作
        blurred_channels = []

        for i in range(4):
            channel = blend_volume_imgs[:, i:i+1, :, :, :]  # 取出单个通道，保持维度
            blurred_channel = F.conv3d(channel, kernel, padding=(k_z//2, k_x//2, k_y//2))
            blurred_channels.append(blurred_channel)
        
        # 将所有通道连接回去
        blend_volume_imgs = torch.cat(blurred_channels, dim=1)
        blend_volume_imgs = blend_volume_imgs.squeeze(0).permute(1, 2, 3, 0)  # 调整回原来的维度
        blend_volume_imgs = blend_volume_imgs[z_pad[0]:blend_volume_imgs.shape[0]-z_pad[1], :, :, :]  # 去除padding部分
        blend_volume_imgs = blend_volume_imgs[:blend_volume_imgs.shape[0]-edge_adjust[0], :blend_volume_imgs.shape[1]-edge_adjust[1], :blend_volume_imgs.shape[2]-edge_adjust[2], :]
        volume_density_cliped = volume_density_cliped[z_pad[0]:volume_density_cliped.shape[0]-z_pad[1], :, :]
        volume_color = volume_color[z_pad[0]:volume_color.shape[0]-z_pad[1], :, :]
        

        # 将卷积结果与 volume_density_cliped 相加

        blend_volume_imgs[:, :, :, 0:3] = blend_volume_imgs[:, :, :, 0:3] / (blend_volume_imgs[:, :, :, [3]] + 1e-9)
        blend_volume_imgs[:, :, :, 0:3] = torch.clamp(blend_volume_imgs[:, :, :, 0:3], min=0, max=1)

        blend_volume_imgs[:, :, :, 0:3] = (blend_volume_imgs[:, :, :, 0:3] * blend_volume_imgs[:, :, :, [3]] + volume_color * volume_density_cliped.unsqueeze(-1)) / (blend_volume_imgs[:, :, :, [3]] + volume_density_cliped.unsqueeze(-1) + 1e-9)
        blend_volume_imgs[:, :, :, 3] += volume_density_cliped
        
        padded_volume_imgs.append(blend_volume_imgs.cpu().numpy())

    # 将所有批次的结果合并
    volume_imgs = np.concatenate(padded_volume_imgs, axis=0)


    print("Calculating Images...")
    for i, img in enumerate(tqdm(volume_imgs)):

        path = image_paths[i]


        concentration_img, pred_rgbd_img = calculate_concentration(img)

        

        print_image, index_image = concentration_to_color(concentration_img)

        # 保存图像


        base_name = os.path.basename(path).replace('npy', 'png')

        save_path = os.path.join(opt.output_folder, 'concentration', base_name)
        np.save(save_path, concentration_img)

        save_path = os.path.join(opt.output_folder, 'pred_rgbd', base_name)
        np.save(save_path, pred_rgbd_img[:, :, [2,1,0,3]])

        save_path = os.path.join(opt.output_folder, 'print', base_name)
        print_image[:, :, 0:3] = cv2.cvtColor(print_image[:, :, 0:3], cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(save_path, print_image * 255)

        save_path = os.path.join(opt.output_folder, 'index', base_name)
        cv2.imwrite(save_path, index_image)
