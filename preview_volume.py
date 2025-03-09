import cv2
import numpy as np
import glob
import os
import json
from tqdm import tqdm
import argparse
import concurrent.futures
from pathlib import Path

def read_images_from_folder(folder_path):
    imgs = np.load(os.path.join(folder_path, 'allData.npy'))
    for i in range(imgs.shape[0]):
         tmpImg= cv2.cvtColor(imgs[i, :, :, 0:3].astype(np.float32), cv2.COLOR_RGB2BGR),
         imgs[i, :, :, 0:3]=tmpImg[0].astype(np.float16)
        
        
    return np.flip(imgs, axis=0)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str)
    parser.add_argument('--savePrefix', type=str, default="")
    parser.add_argument('--maskProportion', type=float, default=0.0)
    parser.add_argument('--onlyOneView', action='store_true')
    parser.add_argument('--pinkBackground', action='store_true')
    opt = parser.parse_args()
    print(opt)
    folder_path = opt.input_folder

    stackImgs = read_images_from_folder(os.path.join(folder_path)) #(num, H, W, 4)
    if opt.maskProportion > 1e-3:
        print(f"Cut voxel density below {opt.maskProportion}")
        percentile_90 = np.percentile(stackImgs[:,:,:,3].flatten(), opt.maskProportion*100)
        stackImgs[stackImgs[:,:,:,3] < percentile_90] = 0
    # 获取图片的高度、宽度和样本数
    num, H, W, _ = stackImgs.shape
    
    #region HardCode
    

    # Step 1: 创建1D高斯分布，生成 num 个采样值
    # mean = 0.0
    # std = 0.1  # Standard deviation
    # x_min, x_max = -3 * std, 3 * std  # Range of the axis (usually within ±3 standard deviations)

    # # Step 1: Generate an array of points along the Gaussian axis
    # x = np.linspace(x_min, x_max, num)

    # Step 2: Calculate the Gaussian PDF for each point
    # pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    # Step 4: Apply the PDF to the 4th channel (index 3) of each image in stackImgs
    
    # gate= np.percentile( stackImgs[:, :, :, 3].flatten(), 95)

    # stackImgs[:, :, :, 3][stackImgs[:, :, :, 3] < gate]=0
    # for i in range(num):
        # stackImgs[i, :, :, 3] *= pdf[i]  # Multiply the corresponding pdf value with the 4th channel

    # 打印结果
    # print("After applying 1D Gaussian weighting to each image, alpha channel is modified.")
    
    # breakpoint()
    
    # stackImgs=stackImgs[int(num/7):int(num/2)]
    
    # stackImgs[:ImgNum//2,:,:,3]*=10
    
    # stackImgs[:,:,:,3]
    # H, W = stackImgs.shape[1], stackImgs.shape[2]
    # # 计算中间50%的区域的切片范围
    # top = H // 4
    # bottom = 3 * H // 4
    # left = W // 4
    # right = 3 * W // 4

    # # 对每张图片进行切片，保留中间50%的区域
    # stackImgs = stackImgs[:, top:bottom, left:right, :]
    # print("Cropped stackImgs shape:", stackImgs.shape)
    #endregion
    
    def dealWith3DStackImgs(images, saveName: str, stackheight, zoom_scale = (2, 1)):
        # 初始化视频写入器
        if opt.savePrefix!="":
            save_path = os.path.join(folder_path, '..', f'preview-{opt.savePrefix}-{saveName}.mp4')
        else:
            save_path = os.path.join(folder_path, '..', f'preview-{saveName}.mp4')
        frame_height, frame_width = images[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 编码
        fps = 30  # 设置帧率
        video_writer = cv2.VideoWriter(save_path, fourcc, fps, (frame_width*zoom_scale[1], frame_height*zoom_scale[0]))

        result_image = np.zeros((frame_height, frame_width, 4), dtype=np.float32)
        backGround = result_image.copy()
        if opt.pinkBackground:
            backGround[:,:,0]=1
            backGround[:,:,2]=1
            backGround[:,:,3]=1
        for i in tqdm(range(len(images)), desc=f"Writing {save_path}"):
            rgbd = images[i]
            density = rgbd[:, :, [3]]
            alpha = 1 - np.exp(-1 * (density) * stackheight)
            rgb = rgbd[:, :, :3]

            result_image[:, :, :3] = result_image[:, :, :3] + (1 - result_image[:, :, [3]]) * alpha * rgb
            result_image[:, :, [3]] = result_image[:, :, [3]] + (1 - result_image[:, :, [3]]) * alpha

            # 将结果图像转换为适合视频写入的格式
            frame=result_image.copy()
            rgb = backGround[:, :, :3]
            frame[:, :, :3] = frame[:, :, :3] + (1 - frame[:, :, [3]])  * rgb
            frame = (frame[:, :, 0:3] * 255).astype(np.uint8)
            frame = cv2.resize(frame, (frame.shape[1]*zoom_scale[1], frame.shape[0]*zoom_scale[0]))
            
  
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 转为 BGR 格式，适配 OpenCV
            video_writer.write(frame)
            # 保存最后一帧
            if i == len(images) - 1:
                last_frame_path = save_path.rsplit('.', 1)[0] + '_last_frame.png'
                last_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转回RGB格式
                cv2.imwrite(last_frame_path, cv2.cvtColor(last_frame_rgb, cv2.COLOR_RGB2BGR))
                print("Last frame saved to", last_frame_path)

            

        video_writer.release()  # 释放视频写入器
        print("Video saved to", save_path)
        # handbrake 转码
        tmp_file_path = Path(save_path).with_suffix('.tmp')
        handbrake_command = f"HandBrakeCLI -i {save_path} -o {tmp_file_path} -e x264 -q 20 --optimize"
        os.system(handbrake_command)
        # 检查转码是否成功（确保临时文件生成）
        if tmp_file_path.exists():
            # 使用 mv 命令覆盖原始文件
            os.system(f"mv {tmp_file_path} {save_path}")
            print(f"Video transcoding complete: {save_path}")
        else:
            print("Transcoding failed. Temporary file not created.")
        
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Prepare all the method calls
        futures = [
            executor.submit(dealWith3DStackImgs, stackImgs, "view1", 0.014, (2, 1)),
        ]
        if not opt.onlyOneView :
            futures.extend([
                # executor.submit(dealWith3DStackImgs, np.flip(stackImgs, axis=0), "view2", 0.014, (2, 1)),
            executor.submit(dealWith3DStackImgs, np.transpose(stackImgs, (1, 0, 2, 3)), "view3", 0.014 * 6, (1, 3)),
            # executor.submit(dealWith3DStackImgs, np.flip(np.transpose(stackImgs, (1, 0, 2, 3)), axis=0), "view4", 0.014 * 6, (1, 3)),
            executor.submit(dealWith3DStackImgs, np.transpose(stackImgs, (2, 0, 1, 3)), "view5", 0.014 * 3, (1, 6)),
            executor.submit(dealWith3DStackImgs, np.flip(np.transpose(stackImgs, (2, 0, 1, 3)), axis=0), "view6", 0.014 * 3, (1, 6))
            ])
        
        # Wait for all futures to complete
        concurrent.futures.wait(futures)
        concurrent.futures.as_completed(futures)
