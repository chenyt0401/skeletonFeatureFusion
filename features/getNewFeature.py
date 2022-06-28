import features
import pandas as pd

def get_new_feature(data):
    data1 = features.TLV.time_vector_frame_location(data)
    data2 = features.TGV.time_vector_frame_global(data)
    data3 = features.AP.angle_Florence(data)
    data4 = features.PRP.new_relative_position(data)
    finally_data = pd.concat([data1, data2.iloc[:, 4:], data3.iloc[:, 4:], data4.iloc[:, 4:]], axis=1, ignore_index=True)
    return finally_data