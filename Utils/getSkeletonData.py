import pymssql
import pandas as pd
import numpy as np

def get_SkeletonData():
    conn = pymssql.connect(host='localhost', user='sa', password='123456', database='SkeletonDataSet', charset='utf8')
    cur = conn.cursor()
    if not cur:
        raise(NameError, '链接失败')
        return
    else:
        print('链接成功')

    sql = 'select * from Florence3DActions order by convert(int, video)'
    # sql = 'select * from UTKinectAction3D order by convert(int, TableKey)'

    try:
        cur.execute(sql)
        result = cur.fetchall()
        return result
    except Exception as e:
        raise e
# endregion


# 将数据库的有效信息提取出来
def data_set_1(Data):
    Data = pd.DataFrame(Data)
    Data[Data == Data[Data.shape[1]-1][Data.shape[0]-1]] = np.nan  # 将空的位置填nan
    Data = Data.dropna(axis=1, how='any')  # 删除为Nan的行
    # print(Data.values.shape)
    # Data = Data.to_csv('data.txt', sep='\t', header=True)

    Data[3] = Data[3].astype('int')  # 将action转换为数值类型
    Data[4] = Data[4].astype('int')  # 将type转换为数值类型
    Data[1] = Data[1].astype('int')  # 将video转换为数值类型
    Data = Data.T.reset_index(drop=True).T
    return Data
# print(Data)
# 定义每个关节位置

# region 处理关节点坐标
def data_set_2(Data):
    SpineBase, SpineMid, Neck, Head, ShoulderLeft, ElbowLeft, SpineShoulder, WristLeft, HandLeft, HandTipLeft, ShoulderRight, ElbowRight, WristRight, HandRight, HandTipRight, HipLeft, KneeLeft, AnkleLeft, FootLeft, HipRight, KneeRight, AnkleRight, FootRight, ThumbRight, ThumbLeft = (
        [] for i in range(25))
    # SpineMid, Neck, Head, ShoulderLeft, ElbowLeft, WristLeft, ShoulderRight, ElbowRight, WristRight, HipLeft, KneeLeft, AnkleLeft, HipRight, KneeRight, AnkleRight= Data[(i for i in range(7, 22))]
    SpineMid = pd.DataFrame(Data[7].str.split(',').tolist())
    Neck = pd.DataFrame(Data[8].str.split(',').tolist())
    Head = pd.DataFrame(Data[9].str.split(',').tolist())
    ShoulderLeft = pd.DataFrame(Data[10].str.split(',').tolist())
    ElbowLeft = pd.DataFrame(Data[11].str.split(',').tolist())
    WristLeft = pd.DataFrame(Data[12].str.split(',').tolist())
    ShoulderRight = pd.DataFrame(Data[13].str.split(',').tolist())
    ElbowRight = pd.DataFrame(Data[14].str.split(',').tolist())
    WristRight = pd.DataFrame(Data[15].str.split(',').tolist())
    HipLeft = pd.DataFrame(Data[16].str.split(',').tolist())
    KneeLeft = pd.DataFrame(Data[17].str.split(',').tolist())
    AnkleLeft = pd.DataFrame(Data[18].str.split(',').tolist())
    HipRight = pd.DataFrame(Data[19].str.split(',').tolist())
    KneeRight = pd.DataFrame(Data[20].str.split(',').tolist())
    AnkleRight = pd.DataFrame(Data[21].str.split(',').tolist())

    # 整合数据
    serial_number = pd.DataFrame(np.arange(4016).reshape(4016, 1))

    # new_data = Data[(i for i in range(7, 22))]  # 只取7-21的数据

    new_data = pd.concat([serial_number, Data[1], Data[3], Data[6]], axis=1)
    new_data = pd.concat([new_data, SpineMid, Neck, Head, ShoulderLeft, ElbowLeft, WristLeft, ShoulderRight, ElbowRight, WristRight, HipLeft, KneeLeft, AnkleLeft, HipRight, KneeRight, AnkleRight], axis=1)
    new_data = new_data.T.reset_index(drop=True).T  # 重新排序
    # new_data = new_data.to_csv('data.txt', sep='\t', header=True)
    return new_data
# endregion