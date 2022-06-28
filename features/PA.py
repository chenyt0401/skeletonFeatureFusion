import numpy as np
import pandas as pd

# 投影到xy、xz、yz平面角度(弧度值)
def get_plant_angle(data):
    pre_data = data.iloc[:, :4]
    data = data.groupby(1)  # video
    groups = pd.DataFrame()
    yz = np.array([0, 1, 1])  # 投影的面
    xz = np.array([1, 0, 1])
    xy = np.array([1, 1, 0])
    for name, group in data:
        plant = pd.DataFrame()
        for i in range(0, group.shape[1]-4, 3):  # 得到平面向量 20*3关节点xyz
            temp_1 = group.iloc[:, 4+i:4+i+3]*yz  # yz平面
            temp_2 = group.iloc[:, 4+i:4+i+3]*xz  # yz平面
            temp_3 = group.iloc[:, 4+i:4+i+3]*xy  # yz平面
            temp_all = group.iloc[:, 4+i:4+i+3]
            temp_1_x = np.linalg.norm(temp_1, ord=None, axis=1).reshape(-1,1)  # 模长 np.linalg.norm(xxx..., keepdims=True)可保持维度
            temp_2_x = np.linalg.norm(temp_2, ord=None, axis=1).reshape(-1,1)  # 模长
            temp_3_x = np.linalg.norm(temp_3, ord=None, axis=1).reshape(-1,1)  # 模长
            temp_all_1 = np.linalg.norm(temp_all, ord=None, axis=1).reshape(-1,1)  # 模长
            temp_x = (temp_1*temp_all).sum(axis=1).values.reshape(-1, 1)  # 按行相加  两列向量的点积
            temp_y = (temp_2*temp_all).sum(axis=1).values.reshape(-1, 1)
            temp_z = (temp_3*temp_all).sum(axis=1).values.reshape(-1, 1)
            cos_yz = temp_x / (temp_1_x * temp_all_1)  # cos值
            cos_xz = temp_y / (temp_2_x * temp_all_1)  # cos值
            cos_xy = temp_z / (temp_3_x * temp_all_1)  # cos值
            cos_yz = pd.DataFrame(cos_yz).fillna(1)  # 0度是cos值为1
            cos_yz = np.arccos(cos_yz)  # 转化为弧度值
            cos_xz = pd.DataFrame(cos_xz).fillna(1)
            cos_xz = np.arccos(cos_xz)
            cos_xy = pd.DataFrame(cos_xy).fillna(1)
            cos_xy = np.arccos(cos_xy)
            plant = pd.concat([plant, cos_yz, cos_xz, cos_xy], axis=1, ignore_index=True)
        groups = pd.concat([groups, plant], axis=0)  # 忽略索引进行组合
    groups = groups.reset_index(drop=True)  # 行名重新排序
    groups = pd.concat([pre_data, groups], axis=1, ignore_index=True)
    return groups

