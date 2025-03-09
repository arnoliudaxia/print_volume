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

from color.utils import rgbd_to_concentration_figure

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


def calculate_concentration_torch(image):
    # 将图像从 (W, H, 4) 转换为 (-1, 4)
    reshaped_image = image.reshape(-1, 4)
    
    # 计算浓度
    concentration, pred_rgbd, concentration_step1 = rgbd_to_concentration_figure(reshaped_image)
    # 将结果从 (WH, 6) 重新 reshape 为 (W, H, 6)

    concentration_step1 = concentration_step1.reshape(image.shape[0], image.shape[1], 6)
    concentration_image = concentration.reshape(image.shape[0], image.shape[1], 6)
    pred_rgbd_image = pred_rgbd.reshape(image.shape[0], image.shape[1], 4)

    return concentration_image, pred_rgbd_image, concentration_step1

def concentration_to_color_torch(concentration):
    # 将输入转换为 PyTorch 张量并移动到 GPU

    x_size, y_size = concentration.shape[0], concentration.shape[1]
    print_images = torch.zeros((x_size, y_size, 4), dtype=torch.float32, device='cuda')
    index_images = torch.full((x_size, y_size), 5, dtype=torch.float32, device='cuda')

    c_concentration = concentration[:, :, 0]
    m_concentration = concentration[:, :, 1]
    y_concentration = concentration[:, :, 2]
    k_concentration = concentration[:, :, 3]
    w_concentration = concentration[:, :, 4]

    color_random = torch.rand((x_size, y_size), device='cuda')

    c_mask = (color_random < c_concentration)
    m_mask = (c_concentration <= color_random) & (color_random < c_concentration + m_concentration)
    y_mask = (c_concentration + m_concentration <= color_random) & (color_random < c_concentration + m_concentration + y_concentration)
    k_mask = (c_concentration + m_concentration + y_concentration <= color_random) & (color_random < c_concentration + m_concentration + y_concentration + k_concentration)
    w_mask = (c_concentration + m_concentration + y_concentration + k_concentration <= color_random) & (color_random < c_concentration + m_concentration + y_concentration + k_concentration + w_concentration)

    cmykw_save_color_tensor = torch.tensor(cmykw_save_color, device='cuda', dtype=torch.float32)

    print_images[c_mask] = cmykw_save_color_tensor[0]
    print_images[m_mask] = cmykw_save_color_tensor[1]
    print_images[y_mask] = cmykw_save_color_tensor[2]
    print_images[k_mask] = cmykw_save_color_tensor[3]
    print_images[w_mask] = cmykw_save_color_tensor[4]

    index_images[c_mask] = 0
    index_images[m_mask] = 1
    index_images[y_mask] = 2
    index_images[k_mask] = 3
    index_images[w_mask] = 4

    return print_images.cpu().numpy(), index_images.cpu().numpy()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default='/media/vrlab/rabbit/print3dingp/print_ngp_lyf/💾HDDSnake/🐶MASK/lego_diffuse_d-1/volume/ngp_21')
    parser.add_argument('--output_folder', type=str, default='/media/vrlab/rabbit/print3dingp/workspace/experiment/lego_figure')
    parser.add_argument('--maskProportion', type=float, default=0.0)
    opt = parser.parse_args()
    print(f"input_folder: {opt.input_folder}")
    # print(f"output_folder: {opt.output_folder}")
    print(f"maskProportion: {opt.maskProportion}")

    batch_size = 250

    os.makedirs(os.path.join(opt.output_folder, 'print'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_folder, 'rgbd'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_folder, 'index'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_folder, 'concentration'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_folder, 'pred_rgbd'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_folder, 'concentration_c'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_folder, 'concentration_m'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_folder, 'concentration_y'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_folder, 'concentration_k'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_folder, 'concentration_w'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_folder, 'concentration_clear'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_folder, 'concentration_step1_c'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_folder, 'concentration_step1_m'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_folder, 'concentration_step1_y'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_folder, 'concentration_step1_k'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_folder, 'concentration_step1_w'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_folder, 'concentration_step1_clear'), exist_ok=True)


    # imgs = np.load(os.path.join(opt.input_folder, 'cut/allData.npy'))
    imgs = np.load(os.path.join(opt.input_folder, 'array/allData.npy'))
    if opt.maskProportion > 1e-3:
        percentile_90 = np.percentile(imgs[:,:,:,3].flatten(), opt.maskProportion*100)
        imgs[imgs[:,:,:,3] < percentile_90] = 0
        
    print("Loaded volume")
    
    
    
    # 排序确保按照文件名的顺序读取图片
    num_images = imgs.shape[0]
    imgCounter=0
    threshold = 100
    k_z, k_x, k_y = 12, 2, 4
    edge_adjust = [0 if k % 2 == 1 else 1 for k in [k_z, k_x, k_y]]

    for i in range(imgs.shape[0]):
         tmpImg= cv2.cvtColor(imgs[i, :, :, 0:3].astype(np.float32), cv2.COLOR_RGB2BGR),
         imgs[i, :, :, 0:3]=tmpImg[0].astype(np.float16)
        


    pigment_channels = ['c', 'm', 'y', 'k', 'w', 'clear']
    pigment_colors = np.array([
        [0, 255, 255, 255],
        [255, 0, 255, 255],
        [0, 255, 255, 255],
        [0, 0, 0, 255],
        [255, 255, 255, 255],
        [0, 0, 0, 0],
    ])

    # 定义三维卷积核
    kernel = torch.ones((1, 1, k_z, k_x, k_y), device='cuda') / (k_z * k_x * k_y)
    
    AllPrintImages=[]
    Allpred_rgbd_img=[]
    
    for start_idx in tqdm(range(0, num_images, batch_size)):
        end_idx = min(start_idx + batch_size, num_images)


        volume_imgs = imgs[max(0, start_idx - k_z//2):min(num_images, end_idx + k_z//2)].astype(np.float32)

        # 将 NumPy 数组转换为 PyTorch 张量，并移动到 GPU
        batch_imgs = torch.tensor(volume_imgs, device='cuda')

        z_pad = [min(k_z//2, start_idx), min(k_z//2, num_images - end_idx)]


        # 处理当前批次
        volume_density = batch_imgs[:, :, :, 3]
        volume_color = batch_imgs[:, :, :, 0:3].clone()
        volume_density_cliped = torch.clamp(volume_density, max=threshold)
        volume_density_residual = volume_density - volume_density_cliped

        blend_volume_imgs = batch_imgs.clone()
        blend_volume_imgs[:, :, :, 3] = volume_density_residual
        blend_volume_imgs[:, :, :, 0:3] = batch_imgs[:, :, :, 0:3]
        blend_volume_imgs[:, :, :, 0:3] = blend_volume_imgs[:, :, :, 0:3] * blend_volume_imgs[:, :, :, [3]]
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
        
        volume_imgs = blend_volume_imgs

        # 将所有批次的结果合并

        # print("Calculating Images...")
        for  i, img in  enumerate(volume_imgs):

            # print(path, i, img.shape)
            concentration_img, pred_rgbd_img, concentration_step1 = calculate_concentration_torch(img)


            print_image, index_image = concentration_to_color_torch(concentration_img)

            # 保存图像

            # concentration_img = concentration_img.detach().cpu().numpy()
            pred_rgbd_img = pred_rgbd_img.detach().cpu().numpy()


            # save_path = os.path.join(opt.output_folder, 'concentration', base_name)
            # np.save(save_path, concentration_img)
            # print(concentration_img.shape);exit()

            # save_path = os.path.join(opt.output_folder, 'pred_rgbd', f"{str(imgCounter).zfill(8)}.npy")
            Allpred_rgbd_img.append(pred_rgbd_img[:, :, [2,1,0,3]].astype(np.float16))
            # np.save(save_path,pred_rgbd_img[:, :, [2,1,0,3]])

            save_path = os.path.join(opt.output_folder, 'print', f"{str(imgCounter).zfill(8)}.png")

            base_name = os.path.basename(save_path)
            print_image[:, :, 0:3] = cv2.cvtColor(print_image[:, :, 0:3], cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, print_image * 255)

            # save_path = os.path.join(opt.output_folder, 'index', base_name)
            # cv2.imwrite(save_path, index_image)
            img = img.detach().cpu().numpy()
            concentration_step1 = concentration_step1.detach().cpu().numpy()

            print(imgCounter)
            for pigment_channel in pigment_channels:
                pigment_img = concentration_step1[:, :, [pigment_channels.index(pigment_channel)]] * pigment_colors[pigment_channels.index(pigment_channel)]
                pigment_img[img[:, :, 3] < 0.5, 3] = 0
                save_path = os.path.join(opt.output_folder, f'concentration_step1_{pigment_channel}', base_name)
                cv2.imwrite(save_path, pigment_img[:, :, [2,1,0,3]])

            concentration_img = concentration_img.detach().cpu().numpy()
            for pigment_channel in pigment_channels:

                pigment_img = concentration_img[:, :, [pigment_channels.index(pigment_channel)]] * pigment_colors[pigment_channels.index(pigment_channel)]
                save_path = os.path.join(opt.output_folder, f'concentration_{pigment_channel}', base_name)
                cv2.imwrite(save_path, pigment_img[:, :, [2,1,0,3]])

            alpha = 1 - np.exp(-img[:, :, [3]] * 0.05)
            rgbd_img = np.concatenate([img[:, :, 0:3], alpha], axis=2)
            save_path = os.path.join(opt.output_folder, 'rgbd', base_name)
            cv2.imwrite(save_path, (rgbd_img[:, :, [2,1,0,3]]*255).astype(np.uint8))
            imgCounter+=1

    # Allpred_rgbd_img=np.array(Allpred_rgbd_img)
    # pahht=os.path.join(opt.output_folder, 'pred_rgbd', "allData.npy")
    # np.save(pahht,Allpred_rgbd_img)
    # print(f"Save to {pahht}")