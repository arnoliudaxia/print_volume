import numpy as np
import cv2
import os
from tqdm import tqdm

def process_image(image_path, Pu, N, Pd, print_resolution, x_y_pad):
    # 读取图片
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)/255
    pad_x, pad_y = int(print_resolution[0] * x_y_pad), int(print_resolution[1] * x_y_pad)
    image_resolution = [print_resolution[0] - 2*pad_x, print_resolution[1] - 2*pad_y]
    image = cv2.resize(image, (image_resolution[1], image_resolution[0]))
    
    # 获取图片的尺寸
    H, W, C = image.shape

    # 生成目标数组
    total_layers = Pu + N + Pd
    result_array = np.zeros((print_resolution[2], print_resolution[0], print_resolution[1], C), dtype=image.dtype)

    # 填充图片的RGB到指定层
    for i in tqdm(range(Pu, Pu + N)):
        result_array[i, pad_x:pad_x + image_resolution[0], pad_y:pad_y + image_resolution[1], :3] = image[:, :, :3]  # 复制RGB通道
        result_array[i, pad_x:pad_x + image_resolution[0], pad_y:pad_y + image_resolution[1], 3] = image[:, :, 3] * 500

    return result_array

image_path = "/media/vrlab/rabbit/print3dingp/workspace/siggraph_logo/logo.png"
Pu, N, Pd = 300, 200, 300
width, height = 20, 20
x_y_pad = 0.1
print_voxel_size = [0.0846666, 0.042333, 0.014]
print_resolution_z = Pu + N + Pd
print_resolution_x = np.round(width / print_voxel_size[0]).astype(np.int32)
print_resolution_y = np.round(height / print_voxel_size[1]).astype(np.int32)
print_resolution = [print_resolution_x, print_resolution_y, print_resolution_z]
result = process_image(image_path, Pu, N, Pd, print_resolution, x_y_pad)
print("resolution:", print_resolution)
print("size:", print_resolution_x*print_voxel_size[0], print_resolution_y*print_voxel_size[1], print_resolution_z*print_voxel_size[2])
os.makedirs("/media/vrlab/rabbit/print3dingp/workspace/siggraph_logo/array", exist_ok=True)
np.save("/media/vrlab/rabbit/print3dingp/workspace/siggraph_logo/array/allData.npy", result)
