import pandas as pd

# relative position feature
def new_relative_position(data):
    pre = data.iloc[:, :4]
    spinemid = data.iloc[:, 4:7].T.reset_index(drop=True).T  # 取出数据并进行重置列索引
    head = data.iloc[:, 10:13].T.reset_index(drop=True).T
    # 左手
    shoulder_l = data.iloc[:, 13:16].T.reset_index(drop=True).T
    elbow_l = data.iloc[:, 16:19].T.reset_index(drop=True).T
    wrist_l = data.iloc[:, 19:22].T.reset_index(drop=True).T
    # 右手
    shoulder_r = data.iloc[:, 22:25].T.reset_index(drop=True).T
    elbow_r = data.iloc[:, 25:28].T.reset_index(drop=True).T
    wrist_r = data.iloc[:, 28:31].T.reset_index(drop=True).T
    # 左腿
    hip_l = data.iloc[:, 31:34].T.reset_index(drop=True).T
    knee_l = data.iloc[:, 34:37].T.reset_index(drop=True).T
    ankle_l = data.iloc[:, 37:40].T.reset_index(drop=True).T
    # 右腿
    hip_r = data.iloc[:, 40:43].T.reset_index(drop=True).T
    knee_r = data.iloc[:, 43:46].T.reset_index(drop=True).T
    ankle_r = data.iloc[:, 46:49].T.reset_index(drop=True).T
    # head
    head_feature = pd.concat([wrist_l.iloc[:, 0]-head.iloc[:, 0],
                              wrist_l.iloc[:, 1]-head.iloc[:, 1],
                              wrist_l.iloc[:, 2]-head.iloc[:, 2],
                              wrist_r.iloc[:, 0]-head.iloc[:, 0],
                              wrist_r.iloc[:, 1]-head.iloc[:, 1],
                              wrist_r.iloc[:, 2]-head.iloc[:, 2],
                              ankle_l.iloc[:, 0] - head.iloc[:, 0],
                              ankle_l.iloc[:, 1] - head.iloc[:, 1],
                              ankle_l.iloc[:, 2] - head.iloc[:, 2],
                              ankle_r.iloc[:, 0] - head.iloc[:, 0],
                              ankle_r.iloc[:, 1] - head.iloc[:, 1],
                              ankle_r.iloc[:, 2] - head.iloc[:, 2]], axis=1, ignore_index=True)
    # spine
    spine_feature = pd.concat([wrist_l.iloc[:, 0]-spinemid.iloc[:, 0],
                               wrist_l.iloc[:, 1]-spinemid.iloc[:, 1],
                               wrist_l.iloc[:, 2]-spinemid.iloc[:, 2],
                               wrist_r.iloc[:, 0] - spinemid.iloc[:, 0],
                               wrist_r.iloc[:, 1] - spinemid.iloc[:, 1],
                               wrist_r.iloc[:, 2] - spinemid.iloc[:, 2],
                               head.iloc[:, 0] - spinemid.iloc[:, 0],
                               head.iloc[:, 1] - spinemid.iloc[:, 1],
                               head.iloc[:, 2] - spinemid.iloc[:, 2]], axis=1, ignore_index=True)
    # shoulder_l
    shoulder_l_feature = pd.concat([elbow_l.iloc[:, 0]-shoulder_l.iloc[:, 0],
                                    elbow_l.iloc[:, 1]-shoulder_l.iloc[:, 1],
                                    elbow_l.iloc[:, 2]-shoulder_l.iloc[:, 2],
                                    ankle_l.iloc[:, 0] - shoulder_l.iloc[:, 0],
                                    ankle_l.iloc[:, 1] - shoulder_l.iloc[:, 1],
                                    ankle_l.iloc[:, 2] - shoulder_l.iloc[:, 2]], axis=1, ignore_index=True)
    # shoulder_r
    shoulder_r_feature = pd.concat([elbow_r.iloc[:, 0]-shoulder_r.iloc[:, 0],
                                    elbow_r.iloc[:, 1]-shoulder_r.iloc[:, 1],
                                    elbow_r.iloc[:, 2]-shoulder_r.iloc[:, 2],
                                    ankle_r.iloc[:, 0] - shoulder_r.iloc[:, 0],
                                    ankle_r.iloc[:, 1] - shoulder_r.iloc[:, 1],
                                    ankle_r.iloc[:, 2] - shoulder_r.iloc[:, 2]], axis=1, ignore_index=True)
    # hip_l
    hip_l_feature = pd.concat([knee_l.iloc[:, 0]-hip_l.iloc[:, 0],
                               knee_l.iloc[:, 1]-hip_l.iloc[:, 1],
                               knee_l.iloc[:, 2]-hip_l.iloc[:, 2],
                               ankle_l.iloc[:, 0] - hip_l.iloc[:, 0],
                               ankle_l.iloc[:, 1] - hip_l.iloc[:, 1],
                               ankle_l.iloc[:, 2] - hip_l.iloc[:, 2]], axis=1, ignore_index=True)
    # hip_r
    hip_r_feature = pd.concat([knee_r.iloc[:, 0]-hip_r.iloc[:, 0],
                               knee_r.iloc[:, 1]-hip_r.iloc[:, 1],
                               knee_r.iloc[:, 2]-hip_r.iloc[:, 2],
                               ankle_r.iloc[:, 0] - hip_r.iloc[:, 0],
                               ankle_r.iloc[:, 1] - hip_r.iloc[:, 1],
                               ankle_r.iloc[:, 2] - hip_r.iloc[:, 2]], axis=1, ignore_index=True)
    feature = pd.concat([pre, head_feature, spine_feature,
                         shoulder_l_feature, shoulder_r_feature,
                         hip_l_feature, hip_r_feature], axis=1, ignore_index=True)
    return feature