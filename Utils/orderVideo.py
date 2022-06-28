import pandas as pd
import numpy as np

def order_video(data, frame):
    data = data.reset_index(drop=True)
    order = pd.DataFrame(np.arange(data.shape[0]).reshape(-1, 1))
    data = data.drop(0, axis=1)
    data.insert(loc=0, column=0, value=order, allow_duplicates=True)

    videos = pd.DataFrame()
    for i in range(int(data.shape[0] / frame)):  # 帧数需修改
        video = pd.DataFrame([i + 1 for j in range(frame)])
        videos = pd.concat([videos, video], axis=0)
    videos = videos.reset_index(drop=True)  # 按列连接时行索引必须一致
    data = data.drop(1, axis=1)
    data.insert(loc=1, column=1, value=videos, allow_duplicates=True)
    data = data.reset_index(drop=True)
    return data