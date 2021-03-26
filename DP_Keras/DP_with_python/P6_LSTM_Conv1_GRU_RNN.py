# In[6-21]P163: RNN numpy实现
import numpy as np

timesteps = 100  # 输入序列的时间步数
input_features = 32  # 输入特征空间的维度
output_features = 64  # 输出特征空间的维度

inputs = np.random.random((timesteps, input_features))  # 输入数据：随机噪声
state_t = np.zeros((output_features,))  # 初始状态：全零向量

# 创建随机的权重矩阵
W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))

successive_outputs = []
for input_t in inputs:  # input_t 是形状为(input_features,)的向量
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)  # 由输入和当前状态（前一个输出）计算得到当前输出
    successive_outputs.append(output_t)  # 将这个输出保存到一个列表中
    state_t = output_t  # 更新网络的状态，用于下一个时间步

'''
当axis = 0时，np.stack的官方解释为 对指定axis增加维度
当axis = 1时，对二维平面的行进行增加，所以本来应该是1行的
'''
final_output_sequence = np.stack(successive_outputs, axis=0)  # 最终输出的是一个形状(timesteps,output_features)

# In[6.2.1]P165:RNN Keras
'''
SimpleRNN 能像其他Keras层一样处理序列批量，而不是Numpy示例那样只能处理单个完整序列
SimpleRNN 可在两种模式下允许：
    1，返回每个时间步连续输出的完整序列，即形状(batch_size,timesteps,output_features)的三维张量；True
    2，只返回每个输入序列的最终输出，即形状(batch_size,output_features)的二维张量；False
    这两种模式，由return_sequences构造函数来控制，默认是False
'''
from keras.models import Sequential

'''
对于颜色，我们可以把它拆成三个特征维度，用这三个维度的组合理论上可以表示任意一种颜色。
同理，对于词，我们也可以把它拆成指定数量的特征维度，
词表中的每一个词都可以用这些维度组合成的向量来表示，这个就是Word Embedding的含义。
'''
from keras.layers import SimpleRNN, Embedding

model = Sequential()
model.add(Embedding(10000, 32))

model.add(SimpleRNN(32, return_sequences=True))
# 多个循环层叠加可提高网络表达能力，其中中间层需返回完整的输出序列，
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32))
model.summary()
# In[6-22]P166: IMDB 数据
from keras.datasets import imdb
from keras.preprocessing import sequence  # 序列预处理

# ############################### 数据预处理 ################################
path = "/Users/vic/coder/workspace for python/pycharm/demo35/Tips/MNIST_data/imdb.npz"

max_features = 10000  # 作为特征的单词个数
max_len = 500  # 在这么多单词之后截断文本
batch_size = 32

print("loading data...")
(input_train, y_train), (input_test, y_test) = imdb.load_data(path, num_words=max_features)
print(len(input_train), 'input_train')
print(len(input_test), 'input_test')

'''
为了实现的简便，keras只能接受长度相同的序列输入。
因此如果目前序列长度参差不齐，这时需要使用pad_sequences()。
该函数是将序列转化为经过填充以后的一个长度相同的新序列新序列。
'''
print("pad_sequences (samples x time )")
input_train = sequence.pad_sequences(input_train, maxlen=max_len)
input_test = sequence.pad_sequences(input_test, maxlen=max_len)

print("input_train shape:", input_train)
print("input_test shape:", input_test)
# ######################## 模型训练--Embedding+SimpleRNN ########################
# dense:全连接层相当于添加一个层
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import SimpleRNN, Embedding

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

'''
训练集有1000个样本，batchsize=10，那么： 
训练完整个样本集需要： 
100次iteration，1次epoch。
具体的计算公式为： 
one epoch = numbers of iterations = N = 训练样本的数量/batch_size
'''

# validation_split用于在没有提供验证集的时候，按一定比例从训练集中取出一部分作为验证集
history = model.fit(input_train, y_train,
                    epochs=10, batch_size=128, validation_split=0.2)

# ######################## 绘制结果 ########################
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='validation acc')
plt.legend()  # 表示不同图形的文本标签,曲线说明
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# In[6-27]P170: LSTM

# ######################## 模型训练--Embedding+LSTM ########################
from keras.layers import LSTM, Dense, Embedding
from keras.models import Sequential

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train,
                    epochs=10, batch_size=128, validation_split=0.2)

# In[6-28]P173:  循环droup、堆叠循环层、双向循环层
# ######################## 基于耶拿天际数据集 ########################
import os

data_dir = '/Users/vic/Desktop/Dataset'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

print(header, '\n', len(lines))

import numpy as np

float_data = np.zeros((len(lines), len(header) - 1))

# enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]  # 去除时间戳
    float_data[i, :] = values

"""
import matplotlib.pyplot  as plt
temp = float_data[:, 1]  # 温度
plt.plot(range(len(temp)),temp)
plt.plot(range(1440),temp[:1440])
"""
# ############################ 数据标准化 ############################
mean = float_data[:200000].mean(axis=0)  # axis=0表示输出矩阵是1行，也就是求每一列的平均值。
float_data -= mean
std = float_data[:200000].std(axis=0)  # 求标准差
float_data /= std  # 数据标准化，使其符合正态分布


# ############################ 生成时间序列样本生成器##################
def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    """
           :param data:    浮点数数据组成的原始数组
           :param lookback:    输入数据应该包括过去多少个时间步
           :param delay:   目标应该在未来多少个时间步之后
           :param min_index,max_index: data数组的索引，用于界定需要抽取哪些时间步。这有助于一部分数据用于验证，另一部分用于测试
           :param shuffle:  打乱样本，还有按顺序抽取样本
           :param batch_size:  每个批量的样本数
           :param step:    数据采样的周期（单位:时间步）,将其设为6即每个小时抽取一个数据点
       """
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))  # np.arange()函数返回一个有终点和起点的固定步长的排列
            i += len(rows)

        samples = np.zeros((len(rows),
                            lookback // step,  # 在A//B中的返回类型与A/B的时一样的，但//取的是结果的最小整数
                            data.shape[-1]))  # 如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets  # 带有 yield 的函数不再是一个普通函数，而是一个生成器generator，可用于迭代，


# ############################ 准备训练、验证、测试生成器 ############################
lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,
                      shuffle=True,
                      step=step,
                      batch_size=batch_size)
val_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    step=step,
                    batch_size=batch_size)
test_gen = generator(float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=300001,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

# This is how many steps to draw from `val_gen`
# in order to see the whole validation set:
val_steps = (300000 - 200001 - lookback) // batch_size  # 为了查看整个验证集需要从val_gen中抽取多少次

# This is how many steps to draw from `test_gen`
# in order to see the whole test set:
test_steps = (len(float_data) - 300001 - lookback) // batch_size  # 为了查看整个测试集，需要从test_gen中抽取多少次


# In[6-35]:P178  计算MAE--平均绝对误差
def evaluate_naive_method():  # 这里用24h之后的温度等于当前温度来预测
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))


evaluate_naive_method()

# In[6-37]:P178  训练评估一个密集连接模型
"""
这是一个密集连接模型
首先将数据展平，然后通过两个Dense层并运行
最后一个Dense并没有使用激活函数，这对回归问题很常见
"""
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)

# In[6-39]:P180  训练评估一个GRU连接模型
"""
GRU 门控循环单元
工作原理与LSTM相同，但是做了一些简化，计算代价更低
"""
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)
# In[6-40]:P182 训练评估一个 dropout正则化 基于GRU的 连接模型
"""
keras的每一个循环层都有两个与 dropout 相关的参数
第一个是dropout，它是一个浮点数，它指定该层输入单元的dropout比率
另一个recurrent_dropout，指定循环单元dropout比率
因为使用dropout正则化的网络总是需要更长时间才完全收敛，所以网络训练轮次增加为原来的两倍
"""
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.2,
                     recurrent_dropout=0.2,
                     input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=40,
                              validation_data=val_gen,
                              validation_steps=val_steps)
# In[6-41]:P183 训练评估一个 dropout正则化 的堆叠GRU模型
"""
模型不再过拟合，可以通过增加网络容量
增加网络容量的做法是增加每层单元数或增加层数

所有中间层都应该返回完整的输出序列--一个3D张量，而不是只返回最后一个时间步的输出
可以用 return_sequence=True来实现
"""
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.1,
                     recurrent_dropout=0.5,
                     return_sequences=True,
                     input_train=(None, float_data.shape[-1])))
model.add(layers.GRU(64,
                     activation='relu',
                     dropout=0.1,
                     recurrent_dropout=0.5))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=40,
                              validation_data=val_gen,
                              validation_steps=val_steps)

# In[draw]: 绘制结果
import matplotlib.pyplot as plt

# acc and val_acc
acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='validation acc')
plt.legend()  # 表示不同图形的文本标签,曲线说明
plt.figure()

# loss and val_loss
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# In[6-42]:P185  使用逆序序列训练并评估一个LSTM
from keras.datasets import imdb
from keras.preprocessing import sequence  # 序列预处理
from keras.models import Sequential
from keras import layers

"""
逆序GRU的效果身子比基于常识的基准方法差很多，这说明在本例子中按时间处理正序对成功解决问题很重要
这说明： GRU层通常更善于记住最近的数据，而不是久远的数据，与更早的数据点相比，更靠后的数据点对问题自然具有更高的预测能力

逆序LSTM的性能与正序列LSTM几乎相同

"""
# ############################### 数据预处理 ################################
path = "/Users/vic/coder/workspace for python/pycharm/demo35/Tips/MNIST_data/imdb.npz"

max_features = 10000  # 作为特征的单词个数
maxlen = 500  # 在这么多单词之后截断文本

(x_train, y_train), (x_test, y_test) = imdb.load_data(path, num_words=max_features)

# 将序列反转
x_train = [x[::-1] for x in x_train]
# y_train = [x[::-1] for x in y_train]
x_test = [x[::-1] for x in x_test]
# y_test = [x[::-1] for x in y_test]

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

model = Sequential()
model.add(layers.Embedding(max_features, 128))
model.add(layers.LSTM(32))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)

# In[6-43]:P186 使用评估双向LSTM
from keras.models import Sequential
from keras import layers

"""
使用Bidirectional层 第一个参数是循环层实例，对循环层创建第二个单独实例
使用第一个实例按照正序处理输入序列，另外一个实例处理逆序输入序列
"""
model = Sequential()
model.add(layers.Embedding(max_features, 32))
model.add(layers.Bidirectional(layers.LSTM(32)))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

"""
模型表现比上一个普通LSTM略好
但是很快开始过拟合
因为双向层的参数个数是正序LSTM的2倍
添加一些正则化，双向方法在该任务上就会有很好表现
"""
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)

# In[6-44]:P186 使用评估双向GRU
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

"""
模型表现与普通GRU差不多
所有的预测能力都来自正序的一半网络
"""
model = Sequential()
model.add(layers.Bidirectional(layers.GRU(32), input_shape=([None, float_data.shape[-1]])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=40,
                              validation_data=val_gen,
                              validation_steps=val_steps)

# ############################## 用卷积神经网络处理序列 ##############################
# In[6-46]:P190 在IMDB数据上训练并评估一个简单的一维卷积神经网络
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

"""
一维卷积神经网络的架构与二维卷积神经网络相同
它是Conv1D层和MaxPooling1D层的堆叠，最后是一个全局池化层或Flatten层
将三维输出转换为二维，让你可以在模型中添加一个或多个Dense层，用于分类或回归
"""
model = Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(1))

model.summary()
model.compile(optimizer=RMSprop(lr=1e-4),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)
"""
模型的训练结果和验证结果，精度会略低于LSTM，但在CPU和GPU上的运行速度都更快
"""

# In[6-47]:P192 在耶拿上训练并评估一个简单的一维卷积神经网络

# ###################### 为数据准备更高分辨率的数据生成器 ######################
lookback = 720
step = 3  # 之前设置为6(每小时一个数据点)现在设置为3即每三十分钟一个数据点
delay = 144

train_gen = generator(float_data, lookback=lookback, delay=delay, min_index=0, max_index=200000, shuffle=True,
                      step=step)
val_gen = generator(float_data, lookback=lookback, delay=delay, min_index=200001, max_index=300000, step=step)
test_gen = generator(float_data, lookback=lookback, delay=delay, min_index=300001, max_index=None, step=step)

val_steps = (300000 - 200001 - lookback) // 128  # 为了查看整个验证集需要从val_gen中抽取多少次
test_steps = (len(float_data) - 300001 - lookback) // 128  # 为了查看整个测试集，需要从test_gen中抽取多少次

# ###################### 结合一维卷积层和GRU层的模型 ######################
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.Conv1D(32, 5, activation='relu', input_shape=(None, float_data.shape[-1])))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 5, activation='relu'))
model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5))
model.add(layers.Dense(1))

model.summary()

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=20, validation_data=val_gen,
                              validation_steps=val_steps)
