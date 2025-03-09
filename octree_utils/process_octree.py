import torch
import math
from tqdm import tqdm 
import numpy as np
import cv2
import os
import argparse

softplus = torch.nn.Softplus()
delta = 1 / 64 * 0.5 / 8

def process_batch(x_coords, y_coords, z_coords, msk, trivec_grid, density_grid, color_grid, batch_size, density_shift, grid_bbox, grid_resolution):
    save_array = np.zeros((len(z_coords), len(x_coords), len(y_coords), 4), dtype=np.float16)

    for batch_start in tqdm(range(0, len(z_coords), batch_size)):
        
        batch_end = min(batch_start + batch_size, len(z_coords))
        batch_z_coords = z_coords[batch_start:batch_end]

        zv, yv, xv = torch.meshgrid(x_coords, y_coords, batch_z_coords, indexing='ij')
        # zv, xv, yv = torch.meshgrid(x_coords, y_coords, batch_z_coords, indexing='ij')

        sample_grid = torch.stack((xv, yv, zv), dim=-1)
        sample_grid = (sample_grid - grid_bbox[0:3]) * grid_bbox[3:6] * grid_resolution

        sample_density = torch.zeros(sample_grid.shape[0], sample_grid.shape[1], sample_grid.shape[2], device=sample_grid.device)
        sample_color = torch.zeros(sample_grid.shape[0], sample_grid.shape[1], sample_grid.shape[2], 3, device=sample_grid.device)

        sample_grid_floor = sample_grid.floor().to(torch.long)
        _p = sample_grid - sample_grid_floor

        trivec_sample = trivec_grid[sample_grid_floor[..., 0], sample_grid_floor[..., 1], sample_grid_floor[..., 2]]
        density_sample = density_grid[sample_grid_floor[..., 0], sample_grid_floor[..., 1], sample_grid_floor[..., 2]]
        color_sample = color_grid[sample_grid_floor[..., 0], sample_grid_floor[..., 1], sample_grid_floor[..., 2]]
        msk_sample = msk[sample_grid_floor[..., 0], sample_grid_floor[..., 1], sample_grid_floor[..., 2]]
        # p = position_grid[sample_grid_floor[..., 0], sample_grid_floor[..., 1], sample_grid_floor[..., 2]]

        # tree_depth = 6
        # scale_modifier = 1
        # nsize = math.pow(2, -1.*tree_depth) * scale_modifier
        # scale = grid_bbox[3:6] * nsize
        # voxel_min = p - 0.5 * scale
        # voxel_max = p + 0.5 * scale

        _p = _p * 7
        # print(_p.floor().min(), _p.floor().max())
        _ip = torch.clamp(_p.floor(), 0, 6).to(torch.long)
        w = _p - _ip
        # print(w.min(), w.max())
        # w = torch.where(w < 0, torch.ones_like(w), w)
        # w = torch.where(w > 1, torch.zeros_like(w), w)

        for r in range(trivec_sample.shape[3]):
            trivec = trivec_sample[..., r, :, :] # (H, W, B, 3, n_dim)
            trivec_dim = trivec.shape[-1]

            # _ip = _p * 7
            # _ip_index = _ip.floor().to(torch.long) # (H, W, B, 3)
            # w = _ip - _ip_index # (H, W, B, 3)
            
            # _p = p - voxel_min / (voxel_max - voxel_min) * trivec_dim - 0.5
            # print(f"_p\n{_p}")
            # print(f"_ip_index\n{_ip_index}")
            # _ip = torch.max(torch.zeros_like(_p), _p.floor())
            # _ip = torch.min((trivec_dim - 2)*torch.ones_like(_p), _ip).to(torch.long)
            _ip_index = _ip

            _ip_index_x0_flat = _ip_index[..., 0].reshape(-1, 1)   # 形状: (H*W*B, 1)
            _ip_index_x1_flat = _ip_index[..., 0].reshape(-1, 1)+1 # 形状: (H*W*B, 1)
            _ip_index_y0_flat = _ip_index[..., 1].reshape(-1, 1)   # 形状: (H*W*B, 1)
            _ip_index_y1_flat = _ip_index[..., 1].reshape(-1, 1)+1 # 形状: (H*W*B, 1)
            _ip_index_z0_flat = _ip_index[..., 2].reshape(-1, 1)   # 形状: (H*W*B, 1)
            _ip_index_z1_flat = _ip_index[..., 2].reshape(-1, 1)+1 # 形状: (H*W*B, 1)

            trivec_x_flat = trivec[..., 0, :].reshape(-1, trivec.shape[-1])  # 形状: (H*W*B, n_dim)
            trivec_y_flat = trivec[..., 1, :].reshape(-1, trivec.shape[-1])  # 形状: (H*W*B, n_dim)
            trivec_z_flat = trivec[..., 2, :].reshape(-1, trivec.shape[-1])  # 形状: (H*W*B, n_dim)

            x0_feature = torch.gather(trivec_x_flat, 1, _ip_index_x0_flat).reshape(trivec.shape[:-2])  # 形状: (H, W, B)
            x1_feature = torch.gather(trivec_x_flat, 1, _ip_index_x1_flat).reshape(trivec.shape[:-2])  # 形状: (H, W, B)
            y0_feature = torch.gather(trivec_y_flat, 1, _ip_index_y0_flat).reshape(trivec.shape[:-2])  # 形状: (H, W, B)
            y1_feature = torch.gather(trivec_y_flat, 1, _ip_index_y1_flat).reshape(trivec.shape[:-2])  # 形状: (H, W, B)
            z0_feature = torch.gather(trivec_z_flat, 1, _ip_index_z0_flat).reshape(trivec.shape[:-2])  # 形状: (H, W, B)
            z1_feature = torch.gather(trivec_z_flat, 1, _ip_index_z1_flat).reshape(trivec.shape[:-2])  # 形状: (H, W, B)

            x_feature = x0_feature * w[..., 0] + x1_feature * (1 - w[..., 0])
            y_feature = y0_feature * w[..., 1] + y1_feature * (1 - w[..., 1])
            z_feature = z0_feature * w[..., 2] + z1_feature * (1 - w[..., 2])

            _feature = x_feature * y_feature * z_feature

            # c000, c100, c010, c110, c001, c101, c011, c111 = trivec.unbind(-1)
            # c00 = c000 * (1 - _p[..., 0, None]) + c100 * _p[..., 0, None]
            # c01 = c001 * (1 - _p[..., 0, None]) + c101 * _p[..., 0, None]
            # c10 = c010 * (1 - _p[..., 0, None]) + c110 * _p[..., 0, None]
            # c11 = c011 * (1 - _p[..., 0, None]) + c111 * _p[..., 0, None]

            # c0 = c00 * (1 - _p[..., 1, None]) + c10 * _p[..., 1, None]
            # c1 = c01 * (1 - _p[..., 1, None]) + c11 * _p[..., 1, None]

            # c = c0 * (1 - _p[..., 2, None]) + c1 * _p[..., 2, None]

            # _feature = c.prod(dim=-1)

            sample_density += density_sample[..., r] * _feature
            sample_color += color_sample[..., r, 0, :] * _feature[..., None]

        # print(f"0 color_sample: {color_sample.min()}~{color_sample.max()}")
        # print(f"0 sample_color: {sample_color.min()}~{sample_color.max()}")
        # sample_color = torch.clamp_min(sample_color*C0 + 0.5, 0.0)
        # print(f"1 sample_color: {sample_color.min()}~{sample_color.max()}")
        sample_color = torch.sigmoid(sample_color)

        # print(f"2 sample_color: {sample_color.min()}~{sample_color.max()}")
        sample_density = softplus(sample_density - density_shift * 10) * min(1 / (1 - density_shift), 25.0)
        sample_density *= msk_sample

        # 输出图像
        for i in range(sample_color.shape[2]):
            cv2.imwrite(f"{output_folder}/color_{str(batch_start + i).zfill(8)}.png", sample_color[:, :, i, :].detach().cpu().numpy()[:, :, [2,1,0]] * 255)

        for i in range(sample_color.shape[2]):
            save_alpha = 1 - np.exp(-sample_density[:, :, i].detach().cpu().numpy() * 1 / 64)
            cv2.imwrite(f"{output_folder}/density_{str(batch_start + i).zfill(8)}.png", save_alpha * 255)

        for i in range(sample_color.shape[2]):
            save_array[batch_start + i, :, :, :3] = sample_color[:, :, i, :].detach().cpu().numpy()[:, :, [2,1,0]]
            save_array[batch_start + i, :, :, 3] = sample_density[:, :, i].detach().cpu().numpy() * delta / print_voxel_size[2]

        # 释放内存
        del sample_grid, sample_density, sample_color
        torch.cuda.empty_cache()

    np.save(f"{output_folder}/array/allData.npy", save_array)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process octree data.')
    parser.add_argument('--octree_path', type=str, default='../workspace/octree_fox/octree_full_state.pt',
                        help='Path to the octree file')
    parser.add_argument('--output_folder', type=str, default='../workspace/octree_volume/yibu',
                        help='Path to the output folder')

    args = parser.parse_args()
    octree_path = args.octree_path
    output_folder = args.output_folder



    # octree_path = '../workspace/octree_fox/octree_full_state.pt'
    # output_folder = '../workspace/octree_volume/yibu'
    
    print(f"Octree path: {octree_path}")
    print(f"Output folder: {output_folder}")
    # output_folder = 'temp'
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(f"{output_folder}/array", exist_ok=True)
    octree = torch.load(octree_path, map_location=torch.device('cuda'))

    printing_height = 20
    print_voxel_size = [0.0846666, 0.042333, 0.014]
    
    grid_bbox = [0, 0, 0, 64, 64, 64]

    print_resolution_z = int(printing_height / print_voxel_size[2])
    print_resolution_x = int((grid_bbox[3] - grid_bbox[0]) / (grid_bbox[5] - grid_bbox[2]) * print_resolution_z / print_voxel_size[0] * print_voxel_size[2])
    print_resolution_y = int((grid_bbox[4] - grid_bbox[1]) / (grid_bbox[5] - grid_bbox[2]) * print_resolution_z / print_voxel_size[1] * print_voxel_size[2])
    print_resolution = [print_resolution_x, print_resolution_y, print_resolution_z]

    print(print_resolution)

    grid_bbox = octree['aabb']
    # print_resolution = [1000, 1000, 1000]
    # print_resolution = [64, 64, 64]

    trivec = octree['trivec']
    densities = octree['densities']


    shs = octree['shs']
    C0 = 0.28209479177387814
    colors = shs*C0
    octree_position = octree['octree_position'] + octree['aabb'][0:3]
    tree_depths = octree['depth']
    density_shift = octree['density_shift']
    aabb = octree['aabb']
    scale_modifier = 1.0
    grid_resolution = int(round(1./(math.pow(2, -1.*tree_depths[0]) * scale_modifier)))

    index_to_grid = torch.floor((octree_position - octree['aabb'][0:3]) / octree['aabb'][3:6] * grid_resolution).to(torch.long)

    msk = torch.zeros(grid_resolution, grid_resolution, grid_resolution).cuda()
    color_grid = torch.zeros(grid_resolution, grid_resolution, grid_resolution, 16, 1, 3).cuda()
    density_grid = torch.zeros(grid_resolution, grid_resolution, grid_resolution, 16).cuda()
    trivec_grid = torch.zeros(grid_resolution, grid_resolution, grid_resolution, 16, 3, 8).cuda()
    position_grid = torch.zeros(grid_resolution, grid_resolution, grid_resolution, 3).cuda()

    msk[index_to_grid[:, 0], index_to_grid[:, 1], index_to_grid[:, 2]] = 1
    color_grid[index_to_grid[:, 0], index_to_grid[:, 1], index_to_grid[:, 2]] = colors
    density_grid[index_to_grid[:, 0], index_to_grid[:, 1], index_to_grid[:, 2]] = densities
    trivec_grid[index_to_grid[:, 0], index_to_grid[:, 1], index_to_grid[:, 2]] = trivec
    position_grid[index_to_grid[:, 0], index_to_grid[:, 1], index_to_grid[:, 2]] = octree_position

    x_coords = torch.linspace(grid_bbox[0], grid_bbox[0]+grid_bbox[3]-1e-6, print_resolution[0]).cuda()
    y_coords = torch.linspace(grid_bbox[1], grid_bbox[1]+grid_bbox[4]-1e-6, print_resolution[1]).cuda()     
    z_coords = torch.linspace(grid_bbox[2], grid_bbox[2]+grid_bbox[5]-1e-6, print_resolution[2]).cuda()

    batch_size = 16  # 根据显存大小调整
    process_batch(x_coords, y_coords, z_coords, msk, trivec_grid, density_grid, color_grid, batch_size, density_shift, grid_bbox, grid_resolution)
    