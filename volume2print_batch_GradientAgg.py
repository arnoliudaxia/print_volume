import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from color.utils import rgbd_to_concentration_torch

kw = 3.93 # 测量值
kk = 3.24 # 测量值
dz = 0.014 # 层高

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
    concentration, pred_rgbd = rgbd_to_concentration_torch(reshaped_image)
    
    # 将结果从 (WH, 6) 重新 reshape 为 (W, H, 6)
    concentration_image = concentration.reshape(image.shape[0], image.shape[1], 6)
    pred_rgbd_image = pred_rgbd.reshape(image.shape[0], image.shape[1], 4)
    

    return concentration_image, pred_rgbd_image

def concentration_to_color_torch(concentration):
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

    return print_images.cpu().numpy(), index_images.cpu().numpy(), [c_mask, m_mask, y_mask, k_mask, w_mask]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str)
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--maskProportion', type=float, default=0.0)
    parser.add_argument('--densityMax', type=float, default=1.0)
    parser.add_argument('--densityKernelSize', type=int, default=10)
    parser.add_argument('--target_frame', type=int, default=None, 
                       help='指定要处理的帧号，如果不指定则处理所有帧')
    
    opt = parser.parse_args()
    if opt.output_folder is None:
        opt.output_folder = opt.input_folder
    
    print(f"input_folder: {opt.input_folder}")
    print(f"output_folder: {opt.output_folder}")
    print(f"maskProportion: {opt.maskProportion}")

    # batch_size = 250
    batch_size = 700

    os.makedirs(os.path.join(opt.output_folder, 'print'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_folder, 'index'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_folder, 'concentration'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_folder, 'pred_rgbd'), exist_ok=True)


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
        

    original_density =torch.tensor(imgs[..., 3], device='cuda', dtype=torch.float64)  # 原始图像的深度通道
    # clip 0-500
    original_density = torch.clamp(original_density, min=0, max=opt.densityMax) 
    # 使用softmax计算初始概率分布
    probs = torch.softmax(original_density.flatten(), dim=0).reshape(original_density.shape)
    
    # 调整概率使得10x10区域的平均概率和为1
    kernel_size = opt.densityKernelSize
    total_pixels = imgs.size
    probs = probs /kernel_size/kernel_size * total_pixels # 放缩概率使得10x10区域平均和为1
    probs = probs.cpu()
    del original_density
    
    # 定义三维卷积核
    kernel = torch.ones((1, 1, k_z, k_x, k_y), device='cuda') / (k_z * k_x * k_y)
    
    AllPrintImages=[]
    Allpred_rgbd_img=[]
    progress_bar = tqdm(total=num_images, desc="Slicing...")
    for start_idx in range(0, num_images, batch_size):
        end_idx = min(start_idx + batch_size, num_images)


        volume_imgs = imgs[max(0, start_idx - k_z//2):min(num_images, end_idx + k_z//2)].astype(np.float32)

        # 将 NumPy 数组转换为 PyTorch 张量，并移动到 GPU
        batch_imgs = torch.tensor(volume_imgs, device='cuda')

        z_pad = [min(k_z//2, start_idx), min(k_z//2, num_images - end_idx)]


        # 处理当前批次``
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
            if opt.target_frame is not None:
                if imgCounter != opt.target_frame:
                    imgCounter+=1
                    progress_bar.update(1)  # 更新进度条
                    continue
                
            concentration_img, pred_rgbd_img = calculate_concentration_torch(img)
            print_image, index_image, masks = concentration_to_color_torch(concentration_img)
            
            c_mask = masks[0].cpu() 
            m_mask = masks[1].cpu() 
            y_mask = masks[2].cpu() 
            k_mask = masks[3].cpu() 
            w_mask = masks[4].cpu() 
            originalColored=c_mask | m_mask | y_mask | k_mask | w_mask

            for current_mask in [w_mask, k_mask]:
                y_coords, x_coords = torch.where(current_mask)
                points_queue = [(y, x, 1) for y, x in zip(y_coords.cpu().numpy(), x_coords.cpu().numpy())]

                while points_queue:
                    y, x, count = points_queue.pop(0)
                    # 获取局部概率
                    y1, y2 = max(0, y-kernel_size//2), min(current_mask.shape[0], y+kernel_size//2)
                    x1, x2 = max(0, x-kernel_size//2), min(current_mask.shape[1], x+kernel_size//2)
                    local_probs = probs[imgCounter, y1:y2, x1:x2]
                    prob_sum = local_probs.sum()

                    # 根据概率决定是否保留该点
                    if torch.rand(1) > prob_sum:
                        current_mask[y, x] = False
                        continue

                    # 归一化概率并采样新位置
                    local_probs = local_probs / prob_sum
                    flat_probs = local_probs.flatten()
                    try:
                        new_idx = torch.multinomial(flat_probs, 1)[0]
                        new_local_y = new_idx // local_probs.shape[1]
                        new_local_x = new_idx % local_probs.shape[1]
                        new_y = y1 + new_local_y
                        new_x = x1 + new_local_x

                        if not (w_mask[new_y, new_x] or  k_mask[new_y, new_x]):
                            if c_mask[new_y, new_x]:
                                c_mask[new_y, new_x] = False
                            elif m_mask[new_y, new_x]:
                                m_mask[new_y, new_x] = False
                            elif y_mask[new_y, new_x]:
                                y_mask[new_y, new_x] = False
                                
                            current_mask[new_y, new_x] = True
                        else:
                            # 如果目标位置已被占用且重试次数小于3，则加入队列末尾
                            if count < 3:
                                points_queue.append((y, x, count + 1))
                    except:
                        continue

            # 重新合成print_image
            print_image=np.zeros_like(print_image)

            print_image[c_mask] = cmykw_save_color[0]
            print_image[m_mask] = cmykw_save_color[1]
            print_image[y_mask] = cmykw_save_color[2]
            print_image[k_mask] = cmykw_save_color[3]
            print_image[w_mask] = cmykw_save_color[4]
            # 找到原来有颜色但现在变成透明的像素
            current_colored = c_mask | m_mask | y_mask | k_mask | w_mask
            need_fill = originalColored & ~current_colored
            if torch.any(need_fill) :
                y_coords, x_coords = torch.where(need_fill)
                updates = []  # 存储所有需要更新的位置和颜色
                for y, x in zip(y_coords, x_coords):
                    # 获取周围区域
                    y1, y2 = max(0, y-kernel_size//2), min(print_image.shape[0], y+kernel_size//2)
                    x1, x2 = max(0, x-kernel_size//2), min(print_image.shape[1], x+kernel_size//2)
                    
                    # 计算局部区域的平均颜色
                    local_region = print_image[y1:y2, x1:x2]
                    valid_pixels = local_region[..., 3] > 0  # 找到非透明像素
                    if not np.any(valid_pixels):
                        continue
                        
                    avg_color = np.mean(local_region[valid_pixels], axis=0)
                    
                    # 计算与CMY标准色的距离
                    distances = [
                        np.sum((avg_color - cmykw_save_color[0])**2),  # C
                        np.sum((avg_color - cmykw_save_color[1])**2),  # M
                        np.sum((avg_color - cmykw_save_color[2])**2),  # Y
                    ]
                    
                    # 选择最接近的颜色
                    closest_color_idx = np.argmin(distances)
                    updates.append((y, x, closest_color_idx))
                # 一次性更新所有位置
                for y, x, color_idx in updates:
                    print_image[y, x] = cmykw_save_color[color_idx]
            # white=np.array([1., 1., 1., 1])
            # def isWhitepixel(pixel):
            #     return np.all(pixel == white)
            # black=np.array([0., 0., 0., 1])
            # def isBlackpixel(pixel):
            #     return np.all(pixel == black)
            # transparent=np.array([0., 0., 0., 0])

            # for color in [white]:
            # # for color in [white, black]:
            #     # 找到值为1的位置
            #     y_coords, x_coords  = np.where(np.all(print_image == color, axis=-1))

                
                # # 对每个值为1的位置进行处理
                # for y, x in zip(y_coords, x_coords):
                #     # 获取以该点为中心的10x10区域的概率和
                #     y1, y2 = max(0, y-kernel_size//2), min(print_image.shape[0], y+kernel_size//2)
                #     x1, x2 = max(0, x-kernel_size//2), min(print_image.shape[1], x+kernel_size//2)
                #     local_probs=probs[imgCounter, y1:y2, x1:x2]
                #     prob_sum = local_probs.sum()

                #     # 根据whiteBlackProbilityField[x1:x2, y1:y2]中的概率进行随机选择，如果超出则丢弃
                #     if torch.rand(1) > prob_sum:  # 可以调整这个阈值
                #         print_image[y, x] = transparent
                #         continue
                #     # 将概率归一化
                #     # breakpoint()
                #     local_probs = local_probs / prob_sum
                #     # 将概率展平并进行采样
                #     flat_probs = local_probs.flatten()
                #     try:
                #         # 根据概率分布随机选择新位置
                #         new_idx = torch.multinomial(flat_probs, 1)[0]
                #         # 计算新的y,x坐标
                #         new_local_y = new_idx // local_probs.shape[1]
                #         new_local_x = new_idx % local_probs.shape[1]
                #         new_y = y1 + new_local_y
                #         new_x = x1 + new_local_x
                        
                #         # 在原位置删除点，在新位置添加点
                #         if print_image[new_y, new_x] != color:
                #             print(f"y:{y}, x:{x}, new_y:{new_y}, new_x:{new_x}")
                #             print_image[y, x] = transparent
                #             print_image[new_y, new_x] = color
                #     except:
                #         # 如果采样失败（例如所有概率都为0），保持原位置不变
                #         continue
            # concentration_img = concentration_img.detach().cpu().numpy()
            pred_rgbd_img = pred_rgbd_img.detach().cpu().numpy()

            # base_name = os.path.basename(path).replace('npy', 'png')

            # save_path = os.path.join(opt.output_folder, 'concentration', base_name)
            # np.save(save_path, concentration_img)

            # save_path = os.path.join(opt.output_folder, 'pred_rgbd', f"{str(imgCounter).zfill(8)}.npy")
            Allpred_rgbd_img.append(pred_rgbd_img[:, :, [2,1,0,3]].astype(np.float16))
            # np.save(save_path,pred_rgbd_img[:, :, [2,1,0,3]])

            save_path = os.path.join(opt.output_folder, 'print', f"{str(imgCounter).zfill(8)}.png")
            imgCounter+=1
            progress_bar.update(1)  # 更新进度条
            print_image[:, :, 0:3] = cv2.cvtColor(print_image[:, :, 0:3], cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, print_image * 255)

            # save_path = os.path.join(opt.output_folder, 'index', base_name)
            # cv2.imwrite(save_path, index_image)
    progress_bar.close()  # 关闭进度条
    Allpred_rgbd_img=np.array(Allpred_rgbd_img)
    pahht=os.path.join(opt.output_folder, 'pred_rgbd', "allData.npy")
    np.save(pahht,Allpred_rgbd_img)
    print(f"Save to {pahht}")