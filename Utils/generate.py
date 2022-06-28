import pandas as pd
import random
import numpy as np


# 扩大缩小
def data_gen_1(data, times):  # 数据  倍数
    data = data.groupby(1) # 按video进行group
    data_col_num = data.size().tolist()
    frame = data_col_num[0]
    groups = pd.DataFrame()
    for name, group in data:
        groups = pd.concat([groups, group], axis=0)
        pre_group = group.iloc[:, :4]
        pro_group = group.iloc[:, 4:]
        for i in range(times-1):
            temp = pro_group * round(random.uniform(0.8, 1.3), 2)  # 对其他数据进行增强
            temp = pd.concat([pre_group, temp], axis=1)
            groups = pd.concat([groups, temp], axis=0)
    groups = groups.reset_index(drop=True)
    videos = pd.DataFrame()
    for i in range(int(groups.shape[0] / frame)):
        video = pd.DataFrame([i+1 for j in range(frame)])
        videos = pd.concat([videos, video], axis=0)
    videos = videos.reset_index(drop=True)  # 按列连接时行索引必须一致
    groups = groups.drop(1, axis=1)
    groups.insert(loc=1, column=1, value=videos, allow_duplicates=True)
    groups = groups.reset_index(drop=True)
    return groups

# 随机插0帧
def data_gen_2(data, times):
    get_data = pd.DataFrame()
    data = data.groupby(1)
    data_col_num = data.size().tolist()
    frame = data_col_num[0]
    for name, group in data:
        temp_data = pd.DataFrame()
        group = group.reset_index(drop=True)
        length = group.shape[1]
        wide = group.shape[0]
        for j in range(times-1):  # 扩大多少倍（第一倍为源数据）
            temp_group = group.T
            list_num = int(np.random.randint(0, 10, size=1))  # 生成10以内的随机数(前闭后开)
            list = [i for i in range(list_num)]
            for i in range(wide-10+list_num, wide):  # 首尾插0帧
                list.append(i)
            for i in list:
                temp_group.drop([i], axis=1, inplace=True)
                temp_group.insert(loc=i, column='i', value=np.zeros((length, 1)), allow_duplicates=True)
                temp_group = temp_group.T.reset_index(drop=True).T
            temp_group = temp_group.T.reset_index(drop=True)
            temp_data = pd.concat([temp_data, temp_group], axis=0)
        type = pd.DataFrame()
        for j in range(times-1):
            type = pd.concat([type, group.iloc[:, 2]], axis=0)
        type = type.T.reset_index(drop=True).T
        temp_data.drop([2], axis=1, inplace=True)
        temp_data.insert(loc=2, column=2, value=type, allow_duplicates=True)
        get_data = pd.concat([get_data, group, temp_data], axis=0)
    get_data = get_data.reset_index(drop=True)
    videos = pd.DataFrame()
    for i in range(int(get_data.shape[0] / frame)):
        video = pd.DataFrame([i+1 for j in range(frame)])
        videos = pd.concat([videos, video], axis=0)
    videos = videos.reset_index(drop=True)  # 按列连接时行索引必须一致
    get_data.drop(1, axis=1, inplace=True)
    get_data.insert(loc=1, column=1, value=videos, allow_duplicates=True)
    return get_data

# 随机插相似帧
def data_gen_3(data, times):
    get_data = pd.DataFrame()
    data = data.groupby(1)
    data_col_num = data.size().tolist()
    frame = data_col_num[0]
    for name, group in data:
        temp_data = pd.DataFrame()
        group = group.reset_index(drop=True)
        length = group.shape[1]
        wide = group.shape[0]
        for j in range(times - 1):  # 扩大多少倍（第一倍为源数据）
            temp_group = group.T
            list = np.random.randint(1, wide - 2, size=10).tolist()  # 生成10个随机数
            for i in list:
                temp_group.drop([i], axis=1, inplace=True)
                insert_value = np.array(temp_group.iloc[:, i-1])
                temp_group.insert(loc=i, column='i', value=insert_value, allow_duplicates=True)
                temp_group = temp_group.T.reset_index(drop=True).T
            temp_group = temp_group.T.reset_index(drop=True)
            temp_data = pd.concat([temp_data, temp_group], axis=0)
        type = pd.DataFrame()
        for j in range(times - 1):
            type = pd.concat([type, group.iloc[:, 2]], axis=0)
        type = type.T.reset_index(drop=True).T
        temp_data.drop([2], axis=1, inplace=True)
        temp_data.insert(loc=2, column=2, value=type, allow_duplicates=True)
        get_data = pd.concat([get_data, group, temp_data], axis=0)
    get_data = get_data.reset_index(drop=True)
    videos = pd.DataFrame()
    for i in range(int(get_data.shape[0] / frame)):
        video = pd.DataFrame([i + 1 for j in range(frame)])
        videos = pd.concat([videos, video], axis=0)
    videos = videos.reset_index(drop=True)  # 按列连接时行索引必须一致
    get_data.drop(1, axis=1, inplace=True)
    get_data.insert(loc=1, column=1, value=videos, allow_duplicates=True)
    return get_data