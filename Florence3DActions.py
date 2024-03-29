import time
from tensorflow.python.keras.models import load_model
import features
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import Utils

filepath = './data/FLO3D/modelBest/weights.best.hdf5'  # save the best model
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,save_weights_only=False,save_best_only=True,mode='max')
callbacks_list = [checkpoint]

data = pd.read_pickle('./data/FLO3D/30/3D_coordinate_points.pkl')

train_data, test_data = Utils.trainTestSpilt.traingen_test_data(data, 30)  # Delineate the dataset
train_data = Utils.generate.data_gen_1(train_data, 10)  # generate skeleton data
train_data = features.getNewFeature.get_new_feature(train_data)

train_data = Utils.generate.data_gen_2(train_data, 2)  # generate feature data
train_data = Utils.generate.data_gen_3(train_data, 2)  # generate feature data

test_data = features.getNewFeature.get_new_feature(test_data)


l1 = train_data.shape[1]
l2 = test_data.shape[1]
frame_num = 30
train_data = train_data.values.reshape(-1, l1*frame_num)
test_data = test_data.values.reshape(-1, l2*frame_num)
np.random.shuffle(np.array(train_data))  
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
model = model.resNet18p.ResNet18(30, 9, shape[2])

# model = model.resNet50.ResNet50([3, 4, 6, 3])

# model = model.inceptionNet10.Inception10(num_blocks=2, num_classes=9)

# model = model.inceptionResmnet.InceptionResNet_V2((30, 147, 1), 9)

# model = model.leNet.LeNet_5(classes=9)

# model = model.alexNet8.Alexnet8(classes=9)

# model = model.VGG16.VGG16Net(classes=9)

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


resultPath = './data/FLO3D/30/data.txt'
model = load_model(filepath)
loss, accuracy = model.evaluate(test_data, test_label)
Utils.recorderStr.recorderStr(resultPath, history, model_name, accuracy, time_model)  #  resultPath, his, modelName, accuracy, timeCost

plt.plot(history.epoch, history.history.get('accuracy'))
plt.plot(history.epoch, history.history.get('val_accuracy'))
plt.show()

