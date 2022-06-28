import pandas as pd
import numpy as np

# region 前一帧与当前帧的位移
def time_distance_frame(data):
    other_data = data.iloc[:, :4]  # 截取非坐标段
    data = data.groupby(1)  # video
    last_distance = pd.DataFrame()  # 存放最终需要数据
    for name, group in data:
        tem_distance_2 = pd.DataFrame()  # 存放距离数据
        for i in range(group.shape[0]-1, 0, -1):  # 单组最后一帧开始计算
            group.iloc[i, :] = group.iloc[i, :] - group.iloc[i-1, :]
        one_frame = pd.DataFrame(np.zeros((1, group.shape[1]-4), dtype=np.float64))  # 0帧
        group.iloc[0, 4:] = one_frame  # 第一帧置零
        for i in range(0, group.shape[1]-4, 3):  # 关节数*3
            tem_distance_1 = pd.concat([group.iloc[:, i+4], group.iloc[:, i+1+4], group.iloc[:, i+2+4]], axis=1)  # 存放当前关节xyz
            ret = np.linalg.norm(tem_distance_1.values, ord=None, axis=1)  # 求距离
            ret = pd.DataFrame(ret)
            tem_distance_2 = pd.concat([tem_distance_2, ret], axis=1)  # 得到当前group的关节距离数据
        tem_distance_2 = tem_distance_2.T.reset_index(drop=True).T  # 更新索引
        last_distance = pd.concat([last_distance, tem_distance_2], axis=0)
    last_distance = last_distance.reset_index(drop=True)  # 更新索引
    last_distance = pd.concat([other_data, last_distance], axis=1, ignore_index=True)  # 得到最终数据
    return last_distance
# endregion