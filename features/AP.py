import pandas as pd
import numpy as np

# region 角度
def angle_Florence(data):  # 特定角度
    def get_cos(vector_x, vector_y):  # 求向量x与y的夹角
        l_x = np.linalg.norm(vector_x, ord=None, axis=1).reshape(-1,1)  # 模长
        l_y = np.linalg.norm(vector_y, ord=None, axis=1).reshape(-1,1)  # 模长

        vector_dot = (vector_x * vector_y).sum(axis=1).values.reshape(-1, 1)  # 点积 按列相加(必须都为np)

        cos_ = pd.DataFrame(vector_dot / (l_x * l_y))  # cos值
        cos_ = pd.DataFrame(cos_).fillna(1)  # nan值填充1  转角度为0
        angle_hu = np.arccos(cos_)  # 弧度值
        return angle_hu

    pre = data.iloc[:, :4]
    # joints
    spinemid = data.iloc[:,4:7].T.reset_index(drop=True).T  # 取出数据并进行重置列索引
    neck = data.iloc[:, 7:10].T.reset_index(drop=True).T
    head = data.iloc[:, 10:13].T.reset_index(drop=True).T
    shoulder_l = data.iloc[:, 13:16].T.reset_index(drop=True).T
    elbow_l = data.iloc[:, 16:19].T.reset_index(drop=True).T
    wrist_l = data.iloc[:, 19:22].T.reset_index(drop=True).T
    shoulder_r = data.iloc[:, 22:25].T.reset_index(drop=True).T
    elbow_r = data.iloc[:, 25:28].T.reset_index(drop=True).T
    wrist_r = data.iloc[:, 28:31].T.reset_index(drop=True).T
    hip_l = data.iloc[:, 31:34].T.reset_index(drop=True).T
    knee_l = data.iloc[:, 34:37].T.reset_index(drop=True).T
    ankle_l = data.iloc[:, 37:40].T.reset_index(drop=True).T
    hip_r = data.iloc[:, 40:43].T.reset_index(drop=True).T
    knee_r = data.iloc[:, 43:46].T.reset_index(drop=True).T
    ankle_r = data.iloc[:, 46:49].T.reset_index(drop=True).T

    angle_1 = get_cos(elbow_l-shoulder_l, wrist_l-elbow_l)  # elbow_l
    angle_2 = get_cos(elbow_r-shoulder_r, wrist_l-elbow_r)  # elbow_r
    angle_3 = get_cos(shoulder_l-neck, elbow_l-shoulder_l)  # shoulder_l
    angle_4 = get_cos(shoulder_r-neck, elbow_r-shoulder_r)  # shoulder_r
    angle_5 = get_cos(knee_l-hip_l, ankle_l-knee_l)  # knee_l
    angle_6 = get_cos(knee_r-hip_r, ankle_r-knee_r)  # knee_r
    angle_7 = get_cos(hip_l-spinemid, knee_l-hip_l)  # hip_l
    angle_8 = get_cos(hip_r-spinemid, knee_r-hip_r)  # hip_r
    angle_9 = get_cos(wrist_l-head, wrist_r-head)  # 双手相对于头
    angle_10 = get_cos(ankle_l-head, ankle_r-head)  # 双脚相对于头
    angle_11 = get_cos(ankle_l-head, wrist_l-head)  # 左边--头
    angle_12 = get_cos(ankle_r-head, wrist_r-head)  # 右边--头
    angle = pd.concat([angle_1, angle_2, angle_3, angle_4, angle_5, angle_6, angle_7, angle_8, angle_9, angle_10, angle_11, angle_12], axis=1, ignore_index=True)
    angle = pd.concat([pre, angle], axis=1, ignore_index=True)
    return angle