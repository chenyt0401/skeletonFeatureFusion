import numpy as np
import pandas as pd

# region 每个video第一帧与其他帧的距离向量
def time_vector_frame_global(data):
    pre = data.iloc[:, :4]
    get_data = pd.DataFrame()
    data = data.groupby(1)
    for name, group in data:
        oneth_frame = pd.DataFrame(group.iloc[0, :].values.reshape(-1, group.shape[1]))  # 每个video的第一帧
        temp = pd.DataFrame()
        for i in range(group.shape[0]):
            temp = pd.concat([temp, group.iloc[i, :]-oneth_frame], axis=0)
        get_data = pd.concat([get_data, temp], axis=0)
    get_data = get_data.reset_index(drop=True)
    get_data = get_data.T.reset_index(drop=True).T
    get_data = pd.concat([pre, get_data.iloc[:, 4:]], axis=1, ignore_index=True)
    return get_data
# endregion