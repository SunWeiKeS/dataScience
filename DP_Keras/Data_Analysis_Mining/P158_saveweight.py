
from random import shuffle
import pandas as pd
from keras.layers.core import Activation, Dense
from keras.models import Sequential
datafile = 'data/model.xls'
data = pd.read_excel(datafile)
data = data.as_matrix()
shuffle(data)

p = 0.8
train = data[:int(len(data) * p), :]
test = data[int(len(data) * p):, :]

# 构建LM神经网络模型
netfile = 'tmp/net.model'

net = Sequential()  # 建立神经网络
#    net.add(Dense(input_dim = 3, units = 10))
# 添加输入层（3节点）到隐藏层（10节点）的连接
net.add(Dense(10, input_shape=(3,)))
net.add(Activation('relu'))  # 隐藏层使用relu激活函数
#    net.add(Dense(input_dim = 10, units = 1))
# 添加隐藏层（10节点）到输出层（1节点）的连接
net.add(Dense(1, input_shape=(10,)))
net.add(Activation('sigmoid'))  # 输出层使用sigmoid激活函数
net.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    sample_weight_mode="binary")  # 编译模型，使用adam方法求解

net.fit(train[:, :3], train[:, 3], epochs=100, batch_size=1)
net.save_weights(netfile)

predict_result = net.predict_classes(train[:, :3]).reshape(
    len(train))  # 预测结果变形
