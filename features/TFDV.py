import pandas as pd
import numpy as np

# region 位移向量(相隔2帧)  相对位置(局部)
# displacement vectors 当前帧为前一帧与后一帧的位移向量
def distance_vector_2frame(data):
    pre = data.iloc[:, :4]
    pro_data_1 = data.groupby(1)
    dis_data = pd.DataFrame()
    for name, group in pro_data_1:
        temp = pd.DataFrame()
        first_frame = pd.DataFrame(group.iloc[1, :].values.reshape(1, -1))
        temp = pd.concat([temp, first_frame / (0.03 * 2)], axis=0)  # 第一帧的前后相减 一帧为30毫秒
        for i in range(1, group.shape[0]-1):
            temp = pd.concat([temp, pd.DataFrame(((group.iloc[i-1, :]-group.iloc[i+1, :])/(0.03*2)).values.reshape(1, -1))], axis=0)
        temp = pd.concat([temp, pd.DataFrame(((-group.iloc[group.shape[0]-1, :])/(0.03*2)).values.reshape(1, -1))], axis=0)  # 最后一帧
        dis_data = pd.concat([dis_data, temp], axis=0)
    dis_data = dis_data.reset_index(drop=True)
    dis_data = pd.concat([pre, dis_data.iloc[:, 4:]], axis=1, ignore_index=True)
    return dis_data