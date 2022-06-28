import time
from tensorflow.python.keras.models import load_model
import features
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import Utils

filepath = './data/MSR3D/60/weights.best.AS1.hdf5'  # 只保存最佳模型 每个部分调整
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,save_weights_only=False,save_best_only=True,mode='max')
callbacks_list = [checkpoint]

data = pd.read_pickle('./data/MSR3D/60/AS1')
# data = pd.read_pickle('./data/MSR3D/60/AS2')
# data = pd.read_pickle('./data/MSR3D/60/AS3')

train_data, test_data = Utils.trainTestSpilt.traingen_test_data(data, 60)  # 划分数据集
train_data = Utils.generate.data_gen_1(train_data, 10)  # 模拟生成不同人骨架
train_data = features.getNewFeature.get_new_feature(train_data)

train_data = Utils.generate.data_gen_2(train_data, 2)  # 模拟特征片段
train_data = Utils.generate.data_gen_3(train_data, 2)  # 模拟动作幅度

test_data = features.getNewFeature.get_new_feature(test_data)

# 打乱数据顺序以以泛化
l1 = train_data.shape[1]
l2 = test_data.shape[1]
frame_num = 60
train_data = train_data.values.reshape(-1, l1*frame_num)
test_data = test_data.values.reshape(-1, l2*frame_num)
np.random.shuffle(np.array(train_data))  # 只对第一维进行乱序
np.random.shuffle(np.array(test_data))
train_data = pd.DataFrame(train_data.reshape(-1, l1))
test_data = pd.DataFrame(test_data.reshape(-1, l2))

train_label, test_label =Utils.getLabel.get_label(train_data, test_data, frame_num)

train_data = train_data.iloc[:, 4:].values.reshape(-1, frame_num, l1-4, 1)
shape = train_data.shape
print(shape)
test_data = test_data.iloc[:, 4:].values.reshape(-1, frame_num, l1-4, 1)

import model
model_name = "Resnet18"
model = model.resNet18p.ResNet18(60, 8, shape[2])

# model = model.resNet50.ResNet50([3, 4, 6, 3])

# model = model.inceptionNet10.Inception10(num_blocks=2, num_classes=8)

# model = model.inceptionResmnet.InceptionResNet_V2((60, 147, 1), 8)

# model = model.leNet.LeNet_5(classes=8)

# model = model.alexNet8.Alexnet8(classes=8)

# model = model.VGG16.VGG16Net(classes=8)

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
time_model_start = time.time()
history = model.fit(train_data, train_label,
                    # steps_per_epoch=100,
                    validation_data=(test_data, test_label),
                    shuffle=True,
                    # validation_steps=100,
                    callbacks=callbacks_list,
                    epochs=100, batch_size=16, verbose=1)
time_model_end = time.time()
time_model = time_model_end-time_model_start

# 记录数据（训练过程的）
resultPath = './data/MSR3D/60/data.txt'
model = load_model(filepath)
loss, accuracy = model.evaluate(test_data, test_label)
Utils.recorderStr.recorderStr(resultPath, history, model_name, accuracy, time_model)  # 记录数据 resultPath, his, modelName, accuracy, timeCost

plt.plot(history.epoch, history.history.get('accuracy'))
plt.plot(history.epoch, history.history.get('val_accuracy'))
plt.show()

