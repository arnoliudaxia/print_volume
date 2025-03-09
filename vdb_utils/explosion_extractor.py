import h5py
import numpy as np
from tqdm import tqdm
import os
from matplotlib.colors import LinearSegmentedColormap


h5_file_path = '/media/vrlab/rabbit/print3dingp/workspace/cloud/explosion/AerialExplosion_0060_3vol.h5'
output_path = '/media/vrlab/rabbit/print3dingp/workspace/cloud/explosion'
printing_height = 8
print_voxel_size = [0.0846666, 0.042333, 0.014]

delta = 1.0

h5_file = h5py.File(h5_file_path, 'r')

h5_density = np.array(h5_file['density']).transpose(0, 2, 1)
h5_flames = np.array(h5_file['flames']).transpose(0, 2, 1)
h5_temperature = np.array(h5_file['temperature']).transpose(0, 2, 1)
print(h5_flames.min(), h5_flames.max(), h5_temperature.min(), h5_temperature.max())
# exit()

grid_bbox = [0, 0, 0, h5_density.shape[0], h5_density.shape[1], h5_density.shape[2]]


print_resolution_z = int(printing_height / print_voxel_size[2])
print_resolution_x = int((grid_bbox[3] - grid_bbox[0]) / (grid_bbox[5] - grid_bbox[2]) * print_resolution_z / print_voxel_size[0] * print_voxel_size[2])
print_resolution_y = int((grid_bbox[4] - grid_bbox[1]) / (grid_bbox[5] - grid_bbox[2]) * print_resolution_z / print_voxel_size[1] * print_voxel_size[2])
print_resolution = [print_resolution_x, print_resolution_y, print_resolution_z]

print_size = [print_resolution[0] * print_voxel_size[0], print_resolution[1] * print_voxel_size[1], print_resolution[2] * print_voxel_size[2]]
print("print_size: ", print_size)
print("print_resolution: ", print_resolution)
print("h5_density.shape: ", h5_density.shape)
print("h5_density_range: ", h5_density.min(), h5_density.max())

colors = [(0.1, 0.1, 0.1), (0.4, 0.4, 0.4), (1, 1, 1)]  # 黑、红、黄、白
cmap = LinearSegmentedColormap.from_list('explosion', colors, N=256)
T_min, T_max = np.min(h5_temperature), np.max(h5_temperature)
normalized_temperature = (h5_temperature - T_min) / (T_max - T_min)
gamma = 0.6
adjusted_temperature = normalized_temperature ** gamma
rgb_field = cmap(adjusted_temperature)[:, :, :, :3]
print("channel R: ", rgb_field[:, :, :, 0].max(), rgb_field[:, :, :, 0].min())
print("channel G: ", rgb_field[:, :, :, 1].max(), rgb_field[:, :, :, 1].min())
print("channel B: ", rgb_field[:, :, :, 2].max(), rgb_field[:, :, :, 2].min())

h5_volume_array = np.zeros((h5_density.shape[0], h5_density.shape[1], h5_density.shape[2], 4))
h5_volume_array[:, :, :, 3] = h5_density * delta / print_voxel_size[2] / 4
h5_volume_array[:, :, :, 0] = h5_flames[:, :, :]**0.5 * 0.4
h5_volume_array[:, :, :, 1] = h5_flames[:, :, :]**0.5 * 0.14
h5_volume_array[:, :, :, 2] = h5_flames[:, :, :]**0.5 * 0.05
print(h5_volume_array[:, :, :, 0].max(), h5_volume_array[:, :, :, 0].min())
print(h5_volume_array[:, :, :, 1].max(), h5_volume_array[:, :, :, 1].min())
print(h5_volume_array[:, :, :, 2].max(), h5_volume_array[:, :, :, 2].min())
h5_volume_array[:, :, :, 0:3] *= 7
h5_volume_array[:, :, :, 0:3] = np.clip(h5_volume_array[:, :, :, 0:3], 0, 1)
# h5_volume_array[:, :, :, 0:3] *= rgb_field



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

os.makedirs(output_path, exist_ok=True)
os.makedirs(os.path.join(output_path, 'array'), exist_ok=True)

x_coords = np.linspace(grid_bbox[0], grid_bbox[3], print_resolution[0])
y_coords = np.linspace(grid_bbox[1], grid_bbox[4], print_resolution[1])
z_coords = np.linspace(grid_bbox[2], grid_bbox[5], print_resolution[2])


output_array = []
for j in tqdm(range(print_resolution[2])):

    out_basename = str(j).zfill(8)

    xv, yv, zv = np.meshgrid(x_coords, y_coords, z_coords[[j]], indexing='ij')

    interpolated_coords = np.stack((xv, yv, zv), axis=-1)

    output_array.append(trilinear_interpolation(h5_volume_array, interpolated_coords[:, :, [0], :])[:, :, 0, :])
    # print(output_array[-1].shape)
    
output_array = np.array(output_array)
print(output_array.shape)
np.save(os.path.join(output_path, 'array', 'allData.npy'), output_array[:, :, :, [2,1,0,3]])
    
