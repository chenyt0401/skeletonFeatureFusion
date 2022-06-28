import pandas as pd
from itertools import combinations
import numpy as np

# region 当前帧所有关节点（单位）方向
def JJO(new_data, joints):
    data_x = pd.DataFrame()
    data_y = pd.DataFrame()
    data_z = pd.DataFrame()
    for i in range(0, joints * 3, 3):  # 将关节点的xyz分开处理
        data_x = pd.concat([data_x, new_data.iloc[:, i + 4]], axis=1)  # 4为数据里坐标点开始位置  x坐标
        data_y = pd.concat([data_y, new_data.iloc[:, i + 4 + 1]], axis=1)  # y坐标
        data_z = pd.concat([data_z, new_data.iloc[:, i + 4 + 2]], axis=1)  # z坐标
    cc_x = list(combinations(data_x.columns, 2))  # 对x的所有的组合取列名
    cc_y = list(combinations(data_y.columns, 2))
    cc_z = list(combinations(data_z.columns, 2))
    print(cc_x)
    tmp_list_x = []
    tmp_list_y = []
    tmp_list_z = []
    for columns in cc_x:  # 取出x的所有列
        tmp_list_x.append(list(zip(data_x[columns[0]], data_x[columns[1]])))
    for columns in cc_y:
        tmp_list_y.append(list(zip(data_y[columns[0]], data_y[columns[1]])))
    for columns in cc_z:
        tmp_list_z.append(list(zip(data_z[columns[0]], data_z[columns[1]])))
    tmp_list_x = pd.DataFrame(tmp_list_x).T  # append按行排列，所以需要转置
    print(tmp_list_x.shape)
    tmp_list_y = pd.DataFrame(tmp_list_y).T
    tmp_list_z = pd.DataFrame(tmp_list_z).T
    for j in range(int(joints * (joints - 1) / 2)):  # 关节点的两两组合 （15个关节取2的所有排列）
        for i in range(new_data.shape[0]):  # 视频数*帧数
            tmp_list_x.iloc[:, j][i] = tmp_list_x.iloc[:, j][i][1] - tmp_list_x.iloc[:, j][i][0]  # 取x的距离
            tmp_list_y.iloc[:, j][i] = tmp_list_y.iloc[:, j][i][1] - tmp_list_y.iloc[:, j][i][0]
            tmp_list_z.iloc[:, j][i] = tmp_list_z.iloc[:, j][i][1] - tmp_list_z.iloc[:, j][i][0]
    get_data = pd.DataFrame()
    for i in range(int(joints * (joints - 1) / 2)):  # 计算
        tem_data = pd.concat([tmp_list_x.iloc[:, i], tmp_list_y.iloc[:, i], tmp_list_z.iloc[:, i]], axis=1, ignore_index=True)
        tem_data = tem_data.astype(float)
        ret = np.linalg.norm(tem_data.values, ord=None, axis=1)  # 对每一个组合好后的xyz进行开方并取根号（计算距离）
        ret = pd.DataFrame(ret)
        ret = pd.concat([ret, ret, ret], axis=1, ignore_index=True)
        tem_data = tem_data / ret
        get_data = pd.concat([get_data, tem_data], axis=1, ignore_index=True)
    mir_data = -get_data
    get_data = pd.concat([new_data.iloc[:, :4], get_data, mir_data], axis=1, ignore_index=True)
    return get_data
# endregion