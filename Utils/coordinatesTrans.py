import numpy as np
import pandas as pd

# region 以每组动作第一帧的spine为原点
def G_relative_coordinates(data):
    # 每个动作以第一帧为基准
    joints_num = int(data.iloc[0, 4:].shape[0] / 3)
    center_numx = 4
    jointspoint_location = 4
    data = data.groupby(1)  # 每个视频分组
    groups = pd.DataFrame()
    for name, group in data:
        center_joints_x = group.iloc[0, center_numx]
        center_joints_y = group.iloc[0, center_numx+1]
        center_joints_z = group.iloc[0, center_numx+2]
        joints_num = joints_num  #
        for i in range(0, joints_num*3, 3):
            group.iloc[:, jointspoint_location+i] = group.iloc[:, jointspoint_location+i] - center_joints_x
            group.iloc[:, jointspoint_location+i+1] = group.iloc[:, jointspoint_location+i+1] - center_joints_y
            group.iloc[:, jointspoint_location+i+2] = group.iloc[:, jointspoint_location+i+2] - center_joints_z
        groups = pd.concat([groups, group], axis=0)
    return groups
# endregion

# region 以每帧spine为原点
def L_relative_coordinates(data):
    joints_num = int(data.iloc[0, 4:].shape[0] / 3)
    jointspoint_location = 4
    base_x = data.iloc[:, jointspoint_location].values
    base_y = data.iloc[:, jointspoint_location + 1].values
    base_z = data.iloc[:, jointspoint_location + 2].values
    for i in range(3, joints_num * 3, 3):  # 深浅拷贝
        data.iloc[:, jointspoint_location + i] = data.iloc[:, jointspoint_location + i] - base_x
        data.iloc[:, jointspoint_location + i + 1] = data.iloc[:, jointspoint_location + i + 1] - base_y
        data.iloc[:, jointspoint_location + i + 2] = data.iloc[:, jointspoint_location + i + 2] - base_z
    data.iloc[:, jointspoint_location] = data.iloc[:, jointspoint_location] - base_x
    data.iloc[:, jointspoint_location + 1] = data.iloc[:, jointspoint_location + 1] - base_y
    data.iloc[:, jointspoint_location + 2] = data.iloc[:, jointspoint_location + 2] - base_z
    return data
# endregion