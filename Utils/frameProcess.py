import numpy as np
import pandas as pd


# 补帧函数
def add_frame(data, frame_num):  # 数据， 补帧到多少
    add = frame_num - data.shape[0]
    data = data.T  # 插帧只能按列
    index = 0
    while index != add:
        len = data.shape[1]
        e = 1  # 插帧位置
        for i in range(len-1):
            data.insert(loc=e, column='i', value=np.nan, allow_duplicates=True)
            e += 2
            index += 1
            if index == add:
                break
    data = data.T
    data = data.reset_index(drop=True)
    return data

# 删帧函数
def drop_frame(group, frame_num):  # 数据，目标帧数，数据长度
    drop_frame_num = group.shape[0] - frame_num
    inedx = 0  # 记录删帧次数
    group = group.reset_index(drop=True)
    while inedx < drop_frame_num:
        temp_l = group.shape[0]
        for i in range(0, temp_l, 2):  # 每次隔帧删帧
            group.drop([i+1], axis=0, inplace=True)
            inedx += 1
            if inedx == drop_frame_num:
                group = group.reset_index(drop=True)
                return group
        group = group.reset_index(drop=True)
    group = group.reset_index(drop=True)
    return group


# 数据处理，补帧到需要的大小
def process_data_frame(process_data, frame_num):
    process_data = process_data.groupby(1)
    # data_col_num = process_data.size().tolist()  # 获取每个视频的帧数
    frame_num = frame_num  # 补帧到多少
    groups = pd.DataFrame()
    for name, group in process_data:
        # 计算补帧
        data_len = int(len(group))
        add_frame_num = int(frame_num - data_len)
        if add_frame_num >= 0:
            group = add_frame(group, frame_num)  # 补Nan
            groups = pd.concat([groups, group], axis=0, ignore_index=True)
        else:
            group = drop_frame(group, frame_num)  # 删帧
            groups = pd.concat([groups, group], axis=0)
    groups = groups.reset_index(drop=True)
    groups = groups.iloc[:, :].astype(float)
    groups.interpolate(method='linear', axis=0, limit=None, inplace=True)  # 使Nan进行线性变换
    return groups
# endregion