import os
import numpy as np
import cv2
from tqdm import tqdm


input_folder_list = [
    # "/media/vrlab/rabbit/print3dingp/workspace/cloud/bunny_cloud_white_0.25",
    # "/media/vrlab/rabbit/print3dingp/workspace/cloud/bunny_cloud_white_0.0625",
    "/media/vrlab/rabbit/print3dingp/workspace/cloud/tornado_0000_brown_0.5",
    "/media/vrlab/rabbit/print3dingp/workspace/cloud/tornado_0000_brown_1.0"
    # "/media/vrlab/rabbit/print3dingp/workspace/cloud/wdas_cloud_half_white_0.2",
    # "/media/vrlab/rabbit/print3dingp/workspace/cloud/wdas_cloud_half_white_0.04"
    ]
for input_folder in input_folder_list:
    # input_folder = '/media/vrlab/rabbit/print3dingp/workspace/cloud/wdas_cloud_white_40'
    output_folder = input_folder + '_padding'
    print(input_folder, output_folder)

    # pred_rgbd_files = sorted([os.path.join(input_folder, 'pred_rgbd', f) for f in os.listdir(os.path.join(input_folder, 'pred_rgbd')) if f.endswith('.npy')])
    print_files = sorted([os.path.join(input_folder, 'print', f) for f in os.listdir(os.path.join(input_folder, 'print')) if f.endswith('.png')])

    padding_ratio = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1] # z, z, x, x, y, y

    sample_image = cv2.imread(print_files[0], cv2.IMREAD_UNCHANGED)

    # num_pred_rgbd_padding = int(len(pred_rgbd_files) * 0.1)
    num_print_padding_up = int(len(print_files) * padding_ratio[0])
    num_print_padding_down = int(len(print_files) * padding_ratio[1])
    num_print_padding_x0 = int(sample_image.shape[0] * padding_ratio[2])
    num_print_padding_x1 = int(sample_image.shape[0] * padding_ratio[3])
    num_print_padding_y0 = int(sample_image.shape[1] * padding_ratio[4])
    num_print_padding_y1 = int(sample_image.shape[1] * padding_ratio[5])

    # sample_array = np.load(pred_rgbd_files[0])

    # save_array = np.zeros((int(1.2*(sample_array.shape[0])), int(1.2*(sample_array.shape[1])), sample_array.shape[2]))
    zero_image = np.zeros((int(num_print_padding_x0 + num_print_padding_x1 + sample_image.shape[0]), int(num_print_padding_y0 + num_print_padding_y1 + sample_image.shape[1]), sample_image.shape[2]))

    os.makedirs(os.path.join(output_folder, 'print'), exist_ok=True)
    # for i in tqdm(range(num_pred_rgbd_padding)):
    #     np.save(os.path.join(output_folder, 'pred_rgbd', f'{i:08d}.png.npy'), save_array)

    for i in tqdm(range(num_print_padding_up)):
        cv2.imwrite(os.path.join(output_folder, 'print', f'{i:08d}.png'), zero_image)

    # for i in tqdm(range(num_pred_rgbd_padding, len(pred_rgbd_files)+num_pred_rgbd_padding)):
    #     ori_array = np.load(pred_rgbd_files[i-num_pred_rgbd_padding])
    #     save_array = np.zeros((int(1.2*(sample_array.shape[0])), int(1.2*(sample_array.shape[1])), sample_array.shape[2]))
    #     save_array[int(0.1*(sample_array.shape[0])):int(0.1*(sample_array.shape[0]))+sample_array.shape[0], int(0.1*(sample_array.shape[1])):int(0.1*(sample_array.shape[1]))+sample_array.shape[1], :] = ori_array
    #     np.save(os.path.join(output_folder, 'pred_rgbd', f'{i:08d}.png.npy'), save_array)

    for i in tqdm(range(num_print_padding_up, len(print_files)+num_print_padding_up)):
        ori_image = cv2.imread(print_files[i-num_print_padding_up], cv2.IMREAD_UNCHANGED)
        save_image = np.zeros((int((sample_image.shape[0] + num_print_padding_x0 + num_print_padding_x1)), int((sample_image.shape[1] + num_print_padding_y0 + num_print_padding_y1)), sample_image.shape[2]))
        save_image[num_print_padding_x0:num_print_padding_x0+sample_image.shape[0], num_print_padding_y0:num_print_padding_y0+sample_image.shape[1], :] = ori_image
        cv2.imwrite(os.path.join(output_folder, 'print', f'{i:08d}.png'), save_image)

    # for i in tqdm(range(len(pred_rgbd_files)+num_pred_rgbd_padding, len(pred_rgbd_files)+num_pred_rgbd_padding+num_pred_rgbd_padding)):
    #     np.save(os.path.join(output_folder, 'pred_rgbd', f'{i:08d}.png.npy'), save_array)

    for i in tqdm(range(len(print_files)+num_print_padding_up, len(print_files)+num_print_padding_up+num_print_padding_down)):
        cv2.imwrite(os.path.join(output_folder, 'print', f'{i:08d}.png'), zero_image)
