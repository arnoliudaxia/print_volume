import pandas as pd
from utils import KMmodel
import numpy as np
# 读取CSV文件
df_m = pd.read_csv('/media/vrlab/rabbit/print3dingp/print_volume/color/data/calib_data/m.csv')
df_w = pd.read_csv('/media/vrlab/rabbit/print3dingp/print_volume/color/data/calib_data/w.csv')

df_m3w1 = pd.read_csv('/media/vrlab/rabbit/print3dingp/color/data/KS_result_all/M3W1_1.0.csv')
df_w3m1 = pd.read_csv('/media/vrlab/rabbit/print3dingp/color/data/KS_result_all/M1W3_1.0.csv')
df_w2m2 = pd.read_csv('/media/vrlab/rabbit/print3dingp/color/data/KS_result_all/M2W2_1.0.csv')




# 提取K列和S列
m_k_column = df_m['K'].values
m_s_column = df_m['S'].values
w_k_column = df_w['K'].values
w_s_column = df_w['S'].values


m3w1_R_target = df_m3w1['R'].values
m3w1_T_target = df_m3w1['T'].values
w3m1_R_target = df_w3m1['R'].values
w3m1_T_target = df_w3m1['T'].values
w2m2_R_target = df_w2m2['R'].values
w2m2_T_target = df_w2m2['T'].values

min_loss = float('inf')

for k in np.linspace(1, 10, 100):

    m_k_column_new = m_k_column*k
    m_s_column_new = m_s_column*k
    m3w1_R, m3w1_T = KMmodel(m_k_column_new*0.75 + w_k_column*0.25, m_s_column_new*0.75 + w_s_column*0.25, 1)
    w3m1_R, w3m1_T = KMmodel(w_k_column*0.75 + m_k_column_new*0.25, w_s_column*0.75 + m_s_column_new*0.25, 1)
    w2m2_R, w2m2_T = KMmodel(w_k_column*0.50 + m_k_column_new*0.50, w_s_column*0.50 + m_s_column_new*0.50, 1)

    loss = np.mean(np.abs(m3w1_R - m3w1_R_target)) + np.mean(np.abs(w3m1_R - w3m1_R_target)) + np.mean(np.abs(w2m2_R - w2m2_R_target))
    if loss < min_loss:
        min_loss = loss
        min_k = k
    # 创建DataFrame并保存为CSV文件
        result_df = pd.DataFrame({
            'm3w1_R': m3w1_R,
            'm3w1_T': m3w1_T,
            'w3m1_R': w3m1_R,
            'w3m1_T': w3m1_T,
            'w2m2_R': w2m2_R,
            'w2m2_T': w2m2_T
        })

        # 保存为CSV文件
        result_df.to_csv('temp2.csv', index=False)

print(min_k)