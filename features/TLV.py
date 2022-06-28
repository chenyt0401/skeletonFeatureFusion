import pandas as pd
import numpy as np

# region 前一帧与当前帧的向量（方向）(局部)
def time_vector_frame_location(data):
    other_data = data.iloc[:, :4]  # 截取非坐标段
    data = data.groupby(1)
    get_data = pd.DataFrame()
    for name, group in data:
        for i in range(group.shape[0]-1, 0, -1):
            group.iloc[i, :] = group.iloc[i, :] - group.iloc[i - 1, :]
        one_frame = pd.DataFrame(np.zeros((1, group.shape[1]-4), dtype=np.float64))
        group.iloc[0, 4:] = one_frame  # 第一帧置零
        get_data = pd.concat([get_data, group], axis=0)
    get_data = get_data.reset_index(drop=True)
    get_data = pd.concat([other_data, get_data.iloc[:, 4:]], axis=1, ignore_index=True)
    return get_data
# endregion