import numpy as np
import os
from tqdm import tqdm
# 定义体积的尺寸

# 读取二进制文件


bbox_size = 0.14

volume_length = 512

delta = bbox_size / volume_length

grid_bbox = [0, 0, 0, volume_length, volume_length, volume_length]

printing_height = 20

print_voxel_size = [0.0846666, 0.042333, 0.014]

print_resolution_z = int(printing_height / print_voxel_size[2])
print_resolution_x = int((grid_bbox[3] - grid_bbox[0]) / (grid_bbox[5] - grid_bbox[2]) * print_resolution_z / print_voxel_size[0] * print_voxel_size[2])
print_resolution_y = int((grid_bbox[4] - grid_bbox[1]) / (grid_bbox[5] - grid_bbox[2]) * print_resolution_z / print_voxel_size[1] * print_voxel_size[2])
print_resolution = [print_resolution_x, print_resolution_y, print_resolution_z]

volume_shape = (volume_length, volume_length, volume_length, -1)
file_path = f'/media/vrlab/rabbit/print3dingp/print_ngp/data/artemis/panda/volume_raw/{volume_length}x{volume_length}x{volume_length}_0.bin'
output_path = '../workspace/panda/ignp_volume'

os.makedirs(output_path, exist_ok=True)


with open(file_path, 'rb') as f:
    ingp_volume_data = np.fromfile(f, dtype=np.float32)

# 将一维数组重塑为三维数组
ingp_volume_array = ingp_volume_data.reshape(volume_shape)

ingp_volume_array[:, :, :, 3] = ingp_volume_array[:, :, :, 3] * delta / print_voxel_size[2]

ingp_volume_array = ingp_volume_array[grid_bbox[0]:grid_bbox[3], grid_bbox[1]:grid_bbox[4], grid_bbox[2]:grid_bbox[5], :]

# 生成每个维度的线性插值坐标
x_coords = np.linspace(grid_bbox[0], grid_bbox[3], print_resolution[0])
y_coords = np.linspace(grid_bbox[1], grid_bbox[4], print_resolution[1])
z_coords = np.linspace(grid_bbox[2], grid_bbox[5], print_resolution[2])

# 使用 meshgrid 生成网格坐标
xv, yv, zv = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')

# 创建一个大小为 print_resolution 的数组，其中每个位置的坐标是线性插值的结果
interpolated_coords = np.stack((xv, yv, zv), axis=-1)


def trilinear_interpolation(volume, coords):
    x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
    x0, y0, z0 = np.floor([x, y, z]).astype(int)
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

    # Ensure indices are within bounds
    x0 = np.clip(x0, 0, volume.shape[0] - 1)
    x1 = np.clip(x1, 0, volume.shape[0] - 1)
    y0 = np.clip(y0, 0, volume.shape[1] - 1)
    y1 = np.clip(y1, 0, volume.shape[1] - 1)
    z0 = np.clip(z0, 0, volume.shape[2] - 1)
    z1 = np.clip(z1, 0, volume.shape[2] - 1)

    xd, yd, zd = x - x0, y - y0, z - z0

    # Get the values at the corner points
    c000 = volume[x0, y0, z0]
    c100 = volume[x1, y0, z0]
    c010 = volume[x0, y1, z0]
    c110 = volume[x1, y1, z0]
    c001 = volume[x0, y0, z1]
    c101 = volume[x1, y0, z1]
    c011 = volume[x0, y1, z1]
    c111 = volume[x1, y1, z1]

    # Perform trilinear interpolation
    c00 = c000 * (1 - xd).reshape(xd.shape + (1,)) + c100 * xd.reshape(xd.shape + (1,))
    c01 = c001 * (1 - xd).reshape(xd.shape + (1,)) + c101 * xd.reshape(xd.shape + (1,))
    c10 = c010 * (1 - xd).reshape(xd.shape + (1,)) + c110 * xd.reshape(xd.shape + (1,))
    c11 = c011 * (1 - xd).reshape(xd.shape + (1,)) + c111 * xd.reshape(xd.shape + (1,))

    c0 = c00 * (1 - yd).reshape(yd.shape + (1,)) + c10 * yd.reshape(yd.shape + (1,))
    c1 = c01 * (1 - yd).reshape(yd.shape + (1,)) + c11 * yd.reshape(yd.shape + (1,))

    c = c0 * (1 - zd).reshape(zd.shape + (1,)) + c1 * zd.reshape(zd.shape + (1,))

    return c

# 对 interpolated_coords 进行矢量化采样，使用一重for循环在第二个维度上
os.makedirs(os.path.join(output_path, 'array'), exist_ok=True)
for j in tqdm(range(print_resolution[2])):
    out_basename = str(j).zfill(8)


    output_array = trilinear_interpolation(ingp_volume_array, interpolated_coords[:, :, [j], :])
    
    np.save(os.path.join(output_path, 'array', out_basename + '.npy'), output_array[:, :, 0, [2,1,0,3]])
    
