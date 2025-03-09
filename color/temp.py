import cv2
import numpy as np
from utils import concentration_to_rgbd
# 读取四通道图像
image = cv2.imread('/media/vrlab/rabbit/print3dingp/print_ngp_lyf/💾HDDSnake/🐶MASK/lego_diffuse_d-1/volume20/ngp_21/print/00000404.png', cv2.IMREAD_UNCHANGED)
# array = np.load('temp809.npy')
# array_img = array*255
# cv2.imwrite('temp809.png', array_img)
# array = np.load('temp809.npy')
image = image[:,:,[2,1,0,3]]
# 检查图像是否成功读取
# 保存图像为新的文件
image = image[0:100, 300:400]

target_colors = [
        [0, 255, 255, 255],
        [255, 0, 255, 255],
        [255, 255, 0, 255],
        [0, 0, 0, 255],
        [255, 255, 255, 255],
        [0, 0, 0, 0]
    ]

# 统计每种颜色的出现次数
unique_colors, counts = np.unique(image.reshape(-1, image.shape[2]), axis=0, return_counts=True)

# 计算总像素数
total_pixels = image.shape[0] * image.shape[1]

# 计算每种目标颜色的比例
concentration = np.zeros(6)
for i, color in enumerate(['c','m','y','k','w','e']):
    target_color = target_colors[i]
    # 查找目标颜色在unique_colors中的索引
    index = np.where((unique_colors == target_color).all(axis=1))[0]
    if index.size > 0:
        count = counts[index[0]]
        ratio = count / total_pixels
        concentration[i] = ratio
        print(f"Color {color} appears {count} times, ratio: {ratio:.4f}")
    else:
        concentration[i] = 0
        print(f"Color {color} appears 0 times, ratio: 0.0000")
cv2.imwrite('temp.png', image[:,:,[2,1,0,3]])
print("Image saved as temp.png")

concentration[5] = 0
concentration = concentration / np.sum(concentration)
rgbd = concentration_to_rgbd(concentration)
print("pred_rgbd", (rgbd * 255).astype(np.int32))


# concentration = np.array([0.0, 0.75, 0., 0., 0.25, 0.])
# concentration = concentration / np.sum(concentration)
# rgbd = concentration_to_rgbd(concentration)
# print("pred_rgbd", (rgbd * 255).astype(np.int32))