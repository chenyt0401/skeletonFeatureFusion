def joints_reduce(data):
    # 删除
    delete_joints = [4, 5, 6, 22, 23, 24, 34, 35, 36, 46, 47, 48, 58, 59, 60]
    data.drop(delete_joints, axis=1, inplace=True)  # 按列删数据
    data = data.T.reset_index(drop=True).T
    return data