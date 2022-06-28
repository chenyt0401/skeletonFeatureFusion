import numpy as np
import pandas as pd

#  划分数据集
def traingen_test_data(data, frame):  # 划分数据，一部分用于数据增强，一部分用于测试
    col_num = data.shape[1]
    split_rate = 0.7  # 取百分之70作为训练集
    groups = data.groupby(2)  # type类型区分
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    for name, group in groups:
        spilt_point = int(group.shape[0]/frame*split_rate)
        group = group.values.reshape(int(group.shape[0]/frame), -1)
        np.random.shuffle(group)  # 打乱顺序
        group = pd.DataFrame(group)
        train_data = pd.concat([train_data, group.iloc[:spilt_point, :]], axis=0)
        test_data = pd.concat([test_data, group.iloc[spilt_point:, :]], axis=0)
    # x_temp = train_data.shape[1] / 49
    # y_temp = test_data.shape[1] / 49
    train_data=np.array(train_data)  # shuffle必须数组形式才有用  pandas.values无效
    np.random.shuffle(train_data)  # 乱序
    test_data = np.array(test_data)
    np.random.shuffle(test_data)
    train_data = pd.DataFrame(train_data.reshape(-1, col_num))
    test_data = pd.DataFrame(test_data.reshape(-1, col_num))
    train_order = pd.DataFrame(np.arange(train_data.shape[0]).reshape(-1, 1))
    test_order = pd.DataFrame(np.arange(test_data.shape[0]).reshape(-1, 1))
    train_data = train_data.drop(0, axis=1)
    train_data.insert(loc=0, column=0, value=train_order, allow_duplicates=True)
    test_data = test_data.drop(0, axis=1)
    test_data.insert(loc=0, column=0, value=test_order, allow_duplicates=True)

    train_videos = pd.DataFrame()
    for i in range(int(train_data.shape[0] / frame)):
        video = pd.DataFrame([i + 1 for j in range(frame)])
        train_videos = pd.concat([train_videos, video], axis=0)
    train_videos = train_videos.reset_index(drop=True)  # 按列连接时行索引必须一致
    train_data = train_data.drop(1, axis=1)
    train_data.insert(loc=1, column=1, value=train_videos, allow_duplicates=True)
    train_data = train_data.reset_index(drop=True)

    test_videos = pd.DataFrame()
    for i in range(int(test_data.shape[0] / frame)):
        video = pd.DataFrame([i + 1 for j in range(frame)])
        test_videos = pd.concat([test_videos, video], axis=0)
    test_videos = test_videos.reset_index(drop=True)  # 按列连接时行索引必须一致
    test_data = test_data.drop(1, axis=1)
    test_data.insert(loc=1, column=1, value=test_videos, allow_duplicates=True)
    test_data = test_data.reset_index(drop=True)
    return train_data, test_data
