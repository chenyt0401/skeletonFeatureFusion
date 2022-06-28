import pandas as pd

# region 得到标签
def get_label(train_data, test_data, frame_num):  # 数据， 帧数
    train_label = pd.DataFrame()
    test_label = pd.DataFrame()
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)
    for i in range(0, train_data.shape[0], frame_num):
        train_temp = train_data.iloc[i, 2]  # type类型所在列
        train_temp = pd.DataFrame([train_temp])
        train_label = pd.concat([train_label, train_temp], axis=0)
    for i in range(0, test_data.shape[0], frame_num):
        train_temp = test_data.iloc[i, 2]  # type类型所在列
        train_temp = pd.DataFrame([train_temp])
        test_label = pd.concat([test_label, train_temp], axis=0)
    train_label = pd.get_dummies(train_label[0])  # 生成独热码
    test_label = pd.get_dummies(test_label[0])
    return train_label, test_label
# endregion