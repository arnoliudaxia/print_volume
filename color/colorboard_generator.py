import numpy as np
import os
import cv2
import pandas as pd
import utils

bg_color = np.array([1, 1, 1])
light_intensity = 1
x = 3
# bg_color = np.array([0, 0, 0])

K, S = {}, {}
for color in ['e', 'c', 'm', 'y', 'k', 'w']:
    data = pd.read_csv(f'./color/data/calib_data/{color}.csv')
    wavelength = data['Wavelength'].values
    K[color], S[color] = data['K'].values, data['S'].values



def create_board_image(board, sub_image_size=(10, 10)):
    # 将平均颜色数组重塑为100x100的图像
    image_size = (board.shape[0] * sub_image_size[0], board.shape[1] * sub_image_size[1])
    board_image = np.zeros((image_size[0], image_size[1], 3))
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            board_image[i*sub_image_size[0]:(i+1)*sub_image_size[0], j*sub_image_size[1]:(j+1)*sub_image_size[1]] = board[i, j]
    return board_image


def color_board_4color(color_name, blend_num=10):
    weights = np.zeros((blend_num, blend_num, 4))
    R_board = np.zeros((blend_num, blend_num, 3))
    T_board = np.zeros((blend_num, blend_num, 3))
    K4 = np.vstack([K[color] for color in color_name])
    S4 = np.vstack([S[color] for color in color_name])
    for i in range(blend_num):
        for j in range(blend_num):
            total = blend_num - 1
            if i + j <= total:
                weights0 = (total - i - j) / total
                weights1 = (i) / total
                weights2 = (j) / total
                weights3 = 0 / total
            else:
                weights0 = 0 / total
                weights1 = (total - j) / total
                weights2 = (total - i) / total
                weights3 = (i + j - total) / total

            weights[i, j] = np.array([weights0, weights1, weights2, weights3])
            K_ij = (weights[i, j].reshape(-1, 1) * K4).sum(axis=0)
            S_ij = (weights[i, j].reshape(-1, 1) * S4).sum(axis=0)
            R_ij, T_ij = utils.KMmodel(K_ij, S_ij, x)
            R_board[i, j] = utils.calculate_rgb(wavelength, R_ij)
            T_board[i, j] = utils.calculate_rgb(wavelength, T_ij)
    C_board = R_board + T_board
    return R_board, T_board, C_board
    
def color_board_2color(color_name, blend_num=10):
    weights = np.zeros((1, blend_num, 2))
    R_board = np.zeros((1, blend_num, 3))
    T_board = np.zeros((1, blend_num, 3))
    K2 = np.vstack([K[color] for color in color_name])
    S2 = np.vstack([S[color] for color in color_name])
    for i in range(blend_num):
        total = blend_num - 1
        weights0 = i / total
        weights1 = 1 - weights0

        weights[0, i] = np.array([weights0, weights1])
        K_ij = (weights[0, i].reshape(-1, 1) * K2).sum(axis=0)
        S_ij = (weights[0, i].reshape(-1, 1) * S2).sum(axis=0)
        R_ij, T_ij = utils.KMmodel(K_ij, S_ij, x)
        R_board[0, i] = utils.calculate_rgb(wavelength, R_ij)
        T_board[0, i] = utils.calculate_rgb(wavelength, T_ij)
    C_board = R_board + T_board
    C_board = np.clip(C_board, 0, 1)
    return R_board, T_board, C_board
    


if __name__ == '__main__':
    color_borad_4color_set = ['mykw', 'cykw', 'cmyk', 'cmyw', 'mcyk', 'mcyw', 'cykw', 'cmkw', 'ymkw']
    os.makedirs('../workspace/colorboard/4color', exist_ok=True)
    for color_4color in color_borad_4color_set:
        R_board, T_board, C_board = color_board_4color(color_4color, blend_num=10)
        board_image = create_board_image(C_board)
        cv2.imwrite(f'../workspace/colorboard/4color/result_{color_4color}_board.png', board_image[:, :, [2,1,0]]*255)

    color_borad_2color_set = ['cm', 'cy', 'ck', 'cw', 'my', 'mk', 'mw', 'yk', 'yw', 'kw']
    os.makedirs('../workspace/colorboard/2color', exist_ok=True)
    for color_2color in color_borad_2color_set:
        R_board, T_board, C_board = color_board_2color(color_2color, blend_num=10)
        board_image = create_board_image(C_board)
        cv2.imwrite(f'../workspace/colorboard/2color/result_{color_2color}_board.png', board_image[:, :, [2,1,0]]*255)


