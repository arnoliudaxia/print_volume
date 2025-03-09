import pandas as pd
from utils import KMmodel
import numpy as np
# 读取CSV文件
df_y = pd.read_csv('/media/vrlab/rabbit/print3dingp/print_volume/color/data/calib_data/y.csv')
df_c = pd.read_csv('/media/vrlab/rabbit/print3dingp/print_volume/color/data/calib_data/c.csv')

df_y3c1 = pd.read_csv('/media/vrlab/rabbit/print3dingp/color/data/KS_result_all/Y3C1_1.0.csv')
df_y1c3 = pd.read_csv('/media/vrlab/rabbit/print3dingp/color/data/KS_result_all/Y1C3_1.0.csv')
df_y2c2 = pd.read_csv('/media/vrlab/rabbit/print3dingp/color/data/KS_result_all/Y2C2_1.0.csv')




# 提取K列和S列
y_k_column = df_y['K'].values
y_s_column = df_y['S'].values
c_k_column = df_c['K'].values
c_s_column = df_c['S'].values


y3c1_R_target = df_y3c1['R'].values
y3c1_T_target = df_y3c1['T'].values
y1c3_R_target = df_y1c3['R'].values
y1c3_T_target = df_y1c3['T'].values
y2c2_R_target = df_y2c2['R'].values
y2c2_T_target = df_y2c2['T'].values

min_loss = float('inf')

# for k in np.linspace(1, 10, 100):

y3c1_R, y3c1_T = KMmodel(y_k_column*0.75 + c_k_column*0.25, y_s_column*0.75 + c_s_column*0.25, 1)
y1c3_R, y1c3_T = KMmodel(c_k_column*0.75 + y_k_column*0.25, c_s_column*0.75 + y_s_column*0.25, 1)
y2c2_R, y2c2_T = KMmodel(y_k_column*0.50 + c_k_column*0.50, y_s_column*0.50 + c_s_column*0.50, 1)

# loss = np.mean(np.abs(y3c1_R - y3c1_R_target)) + np.mean(np.abs(y1c3_R - y1c3_R_target)) + np.mean(np.abs(y2c2_R - y2c2_R_target))
# if loss < min_loss:
#     min_loss = loss
#     min_k = k
# # 创建DataFrame并保存为CSV文件
result_df = pd.DataFrame({
    'y3c1_R': y3c1_R_target,
    'y3c1_T': y3c1_T_target,
    'y1c3_R': y1c3_R_target,
    'y1c3_T': y1c3_T_target,
    'y2c2_R': y2c2_R_target,
    'y2c2_T': y2c2_T_target
})

        # 保存为CSV文件
result_df.to_csv('tempyc_target.csv', index=False)

# print(min_k)