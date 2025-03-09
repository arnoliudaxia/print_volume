import torch
# import numpy as np
import math
from tqdm import tqdm 

softplus = torch.nn.Softplus()

'''
volume: (dim**3)
coords: (N, 3), supposed in [0, dim]
'''
def trilinear_interpolation(volume, coords, dim):
    x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
    x0, y0, z0 = torch.floor(torch.stack([x, y, z])).long()
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

    # Ensure indices are within bounds
    x0 = torch.clamp(x0, 0, dim - 1)
    x1 = torch.clamp(x1, 0, dim - 1)
    y0 = torch.clamp(y0, 0, dim - 1)
    y1 = torch.clamp(y1, 0, dim - 1)
    z0 = torch.clamp(z0, 0, dim - 1)
    z1 = torch.clamp(z1, 0, dim - 1)

    xd, yd, zd = x - x0.float(), y - y0.float(), z - z0.float()

    # Get the values at the corner points
    c000 = volume[x0*(dim**0) + y0*(dim**1) + z0*(dim**2)]
    c100 = volume[x1*(dim**0) + y0*(dim**1) + z0*(dim**2)]
    c010 = volume[x0*(dim**0) + y1*(dim**1) + z0*(dim**2)]
    c110 = volume[x1*(dim**0) + y1*(dim**1) + z0*(dim**2)]
    c001 = volume[x0*(dim**0) + y0*(dim**1) + z1*(dim**2)]
    c101 = volume[x1*(dim**0) + y0*(dim**1) + z1*(dim**2)]
    c011 = volume[x0*(dim**0) + y1*(dim**1) + z1*(dim**2)]
    c111 = volume[x1*(dim**0) + y1*(dim**1) + z1*(dim**2)]

    # Perform trilinear interpolation
    c00 = c000 * (1 - xd).unsqueeze(-1) + c100 * xd.unsqueeze(-1)
    c01 = c001 * (1 - xd).unsqueeze(-1) + c101 * xd.unsqueeze(-1)
    c10 = c010 * (1 - xd).unsqueeze(-1) + c110 * xd.unsqueeze(-1)
    c11 = c011 * (1 - xd).unsqueeze(-1) + c111 * xd.unsqueeze(-1)

    c0 = c00 * (1 - yd).unsqueeze(-1) + c10 * yd.unsqueeze(-1)
    c1 = c01 * (1 - yd).unsqueeze(-1) + c11 * yd.unsqueeze(-1)

    c = c0 * (1 - zd).unsqueeze(-1) + c1 * zd.unsqueeze(-1)

    return c



def sample_trivec(
    p: torch.Tensor,
    trivec: torch.Tensor,
    trivec_dim: int,
    densities: torch.Tensor,
    colors: torch.Tensor,
    density_shift: float,
    used_rank: int,
    voxel_min: torch.Tensor,
    voxel_max: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        p: shape [B, 3]
        trivec: torch.Size([B, RANK, 3, dim**3])
        densities: torch.Size([B, RANK])
        colors: torch.Size([B, RANK, 1, 3])
        voxel_min: shape [B, 3]
        voxel_max: shape [B, 3]
    """
    # 计算线性插值权重
    _p = (p - voxel_min) / (voxel_max - voxel_min) * trivec_dim 

    out_density = torch.zeros(p.size(0), device=p.device)
    out_color = torch.zeros(p.size(0), 3, device=p.device)

    # 检查是否在合法范围内
    valid_mask = ((_p >= 0) & (_p <= trivec_dim)).all(dim=1)

    for b in range(p.size(0)):
        if not valid_mask[b]:
            continue

        for r in range(used_rank):
            # 计算三线性插值
            _density = trilinear_interpolation(trivec[b, r, 0], _p[b], trivec_dim).item() *\
                       trilinear_interpolation(trivec[b, r, 1], _p[b], trivec_dim).item() *\
                       trilinear_interpolation(trivec[b, r, 2], _p[b], trivec_dim).item()
            out_density[b] += densities[b, r] * _density
            out_color[b] += colors[b, r].squeeze(0) * _density

    out_color = torch.sigmoid(out_color)
    out_density = softplus(out_density - density_shift * 10) * min(1 / (1 - density_shift), 25.0)

    return valid_mask, out_density, out_color


def assign_to_subspaces(position, aabb, tile_num, density, color, trivec, interpolated_coords):
    """
    将点分配到子空间并根据新坐标返回子空间内的点

    Args:
        position (torch.Tensor): (N, 3) 的张量，表示点的位置。
        aabb (torch.Tensor): (6,) 的张量，表示边界框 [xmin, ymin, zmin, xmax, ymax, zmax]。
        tile_num (int): 每个维度的子空间划分数量。

    Returns:
        dict: 一个字典，其中键是子空间索引，值是属于该子空间的点的索引和位置。
    """
    xmin, ymin, zmin, xlen, ylen, zlen = aabb
    xmax = xmin + xlen
    ymax = ymin + ylen
    zmax = zmin + zlen
    
    # 子空间大小

    tile_size = torch.tensor([(xmax - xmin) / tile_num, 
                              (ymax - ymin) / tile_num, 
                              (zmax - zmin) / tile_num])

    # 归一化到子空间索引
    normalized_pos = (position - torch.tensor([xmin, ymin, zmin])) / tile_size
    
    # 子空间索引
    indices = torch.floor(normalized_pos).long()
    
    # 确保索引在合法范围内
    indices = torch.clamp(indices, 0, tile_num - 1)

    # 计算子空间索引标识符
    flat_indices = (indices[:, 0] * tile_num**2 + 
                    indices[:, 1] * tile_num + 
                    indices[:, 2])

    # 将点分组到子空间
    subspace_dict = {}
    for i, flat_index in enumerate(tqdm(flat_indices)):
        if flat_index.item() not in subspace_dict:
            subspace_dict[flat_index.item()] = []
        subspace_dict[flat_index.item()].append(i)

    # 组织输出：包含索引和位置


    subspace_len = len(interpolated_coords)

    indices_tensor = torch.zeros(subspace_len, device=interpolated_coords.device, dtype=torch.long)
    valid_tensor = torch.zeros(subspace_len, device=interpolated_coords.device, dtype=torch.bool)




    result = {}
    for key, indices in subspace_dict.items():
        result[key] = {
            "indices": torch.tensor(indices),
            "position": position[indices], 
            "density": density[indices], 
            "color": color[indices], 
            "trivec": trivec[indices]
        }
        print(indices, key)
        indices_tensor[indices] = torch.tensor(int(key), device=interpolated_coords.device, dtype=torch.long)
        valid_tensor[indices] = torch.tensor(True, device=interpolated_coords.device, dtype=torch.bool)

    return result, indices_tensor, valid_tensor

def query_subspace(subspace_dict, query_points, aabb, tile_num):
    """
    查询指定点所落入的子空间，并返回该子空间内的所有点的索引和位置。

    Args:
        subspace_dict (dict): 由 assign_to_subspaces 函数生成的子空间字典。
        query_points (torch.Tensor): (N, 3) 的张量，表示查询点。
        aabb (torch.Tensor): (6,) 的张量，表示边界框 [xmin, ymin, zmin, xmax, ymax, zmax]。
        tile_num (int): 每个维度的子空间划分数量。

    Returns:
        list[dict]: 每个查询点对应的子空间内的点的字典列表。
    """
    device = query_points.device  # 获取当前设备
    xmin, ymin, zmin, xlen, ylen, zlen = aabb.to(device)
    xmax = xmin + xlen
    ymax = ymin + ylen
    zmax = zmin + zlen

    # 子空间大小
    tile_size = torch.tensor([(xmax - xmin) / tile_num, 
                              (ymax - ymin) / tile_num, 
                              (zmax - zmin) / tile_num], device=device)

    # 计算查询点的子空间索引
    normalized_points = (query_points - torch.tensor([xmin, ymin, zmin], device=device)) / tile_size
    subspace_indices = torch.floor(normalized_points).long()

    # 确保索引在合法范围内
    subspace_indices = torch.clamp(subspace_indices, 0, tile_num - 1)

    # 计算子空间索引标识符
    flat_indices = (subspace_indices[:, 0] * tile_num**2 + 
                    subspace_indices[:, 1] * tile_num + 
                    subspace_indices[:, 2])
    
    flat_indices = torch.round(flat_indices).long()

    # 返回每个查询点对应的子空间内的点

    return flat_indices



def test2(octree):
    grid_bbox = octree['aabb']
    print_resolution = [30, 10, 10]

    # 生成每个维度的线性插值坐标
    x_coords = torch.linspace(grid_bbox[0], grid_bbox[0]+grid_bbox[3], print_resolution[0])
    y_coords = torch.linspace(grid_bbox[1], grid_bbox[1]+grid_bbox[4], print_resolution[1])
    z_coords = torch.linspace(grid_bbox[2], grid_bbox[2]+grid_bbox[5], print_resolution[2])

    # 使用 meshgrid 生成网格坐标
    xv, yv, zv = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')

    # 创建一个大小为 print_resolution 的数组，其中每个位置的坐标是线性插值的结果
    interpolated_coords = torch.stack((xv, yv, zv), dim=-1).reshape(-1,3)
    print(f"interpolated_coords: {interpolated_coords.shape}")

    N = len(octree['trivec'])
    trivec = octree['trivec']
    densities = octree['densities']
    colors = octree['shs']
    octree_position = octree['octree_position'] + octree['aabb'][0:3]
    tree_depths = octree['depth']
    density_shift = octree['density_shift']
    aabb = octree['aabb']
    scale_modifier = 1.0

    # out_densities  = torch.zeros(len(interpolated_coords), device=trivec.device)
    # out_colors     = torch.zeros((len(interpolated_coords), 3), device=trivec.device)
    # out_positions  = torch.zeros((len(interpolated_coords), 3), device=trivec.device)
    out_densities, out_colors, out_positions = [], [], []

    # tile_num = 16
    grid_resolution = 1./(math.pow(2, -1.*tree_depths[0]) * scale_modifier)
    tile_num = grid_resolution
    data, indices_tensor, valid_tensor = assign_to_subspaces(octree_position, aabb, tile_num, densities, colors, trivec, interpolated_coords)

    nsize = math.pow(2, -1.*6) * scale_modifier
    scale = aabb[3:6] * nsize

    data_n = []
    data_pnt = []
    data_voxel_min = []
    data_voxel_max = []


    print("start query subspace")
    interpolated_coords = interpolated_coords.to('cuda')
    flat_indices = query_subspace(data, interpolated_coords, aabb, tile_num)

    
    p = octree_position[indices_tensor[flat_indices]]
    data_voxel_min = p - 0.5 * scale
    data_voxel_max = p + 0.5 * scale
    data_n = flat_indices
    data_pnt = interpolated_coords


    # 将数据整理成批量形式

    # 使用批处理的 sample_trivec 函数
    valid_mask, out_densities, out_colors = sample_trivec(
        data_pnt, 
        trivec[data_n], 
        2, 
        densities[data_n], 
        colors[data_n], 
        density_shift, 
        16, 
        data_voxel_min, 
        data_voxel_max
    )

    # 检查无效的点
    if not valid_mask.all():
        invalid_indices = torch.where(~valid_mask)[0]
        for idx in invalid_indices:
            print(f"pnt: {data_pnt[idx]}, voxel_min: {data_voxel_min[idx]}, voxel_max: {data_voxel_max[idx]}")
        assert(valid_mask.all())

    # 直接使用批处理结果
    out_positions = data_pnt


    out_densities = torch.stack(out_densities, dim=0) # (K, 3)
    out_colors = torch.stack(out_colors, dim=0) # (K, 3)
    out_positions = torch.stack(out_positions, dim=0) # (K, 3)

    # 保存到文件
    with open('temp.obj', 'w') as f:
        for pos, col in zip(out_positions, out_colors):
            pos_str = str(pos[0].item()) + ' ' + str(pos[1].item()) + ' ' + str(pos[2].item())
            col_str = str(col[0].item()) + ' ' + str(col[1].item()) + ' ' + str(col[2].item())
            f.write(f"v {pos_str} {col_str}\n")

octree_path = '../workspace/octree_fox/octree_full_state.pt'
octree = torch.load(octree_path, map_location=torch.device('cpu'))
# octree = torch.load(octree_path)

for key in octree.keys():
    try:
        print(key, octree[key].shape)
    except:
        print(key, octree[key])

test2(octree)
