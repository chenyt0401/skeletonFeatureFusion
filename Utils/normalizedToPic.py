import pandas as pd


def normalized_to_pic(data, joints_num, jointspoint_location):  # 输入坐标点数据，关节点数，关节点坐标在数据中的初始列
    data_x = pd.DataFrame()
    data_y = pd.DataFrame()
    data_z = pd.DataFrame()
    for i in range(0, joints_num*3, 3):
        data_x = pd.concat([data_x, data.iloc[:, i+jointspoint_location]], axis=1)
        data_y = pd.concat([data_y, data.iloc[:, i+jointspoint_location+1]], axis=1)
        data_z = pd.concat([data_z, data.iloc[:, i+jointspoint_location+2]], axis=1)
    min_x = data_x.min().min()
    min_y = data_y.min().min()
    min_z = data_z.min().min()
    max_min_x = data_x.max().max() - min_x
    max_min_y = data_y.max().max() - min_y
    max_min_z = data_z.max().max() - min_z
    for i in range(joints_num):
        data_x.iloc[:, i] = (data_x.iloc[:, i]-min_x) / max_min_x * (255-0)
        data_y.iloc[:, i] = (data_y.iloc[:, i]-min_y) / max_min_y * (255-0)
        data_z.iloc[:, i] = (data_z.iloc[:, i]-min_z) / max_min_z * (255-0)
    data_xyz = pd.DataFrame()
    for i in range(joints_num):
        data_xyz = pd.concat([data_xyz, data_x.iloc[:, i], data_y.iloc[:, i], data_z.iloc[:, i]], axis=1)
    data = pd.concat([data.iloc[:, 0:jointspoint_location], data_xyz], axis=1)
    return data
# endregion