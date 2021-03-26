# In[5-1、2]: P95 实例化小型神经网络
from keras import layers
from keras import models

# sequential model就是那种最简单的结构的模型。按顺序一层一层训练，一层一层往前的那种。没有什么环的结构。
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

# In[5-13]: P113 定义一个包含droupout的新卷积神经网络
from keras import layers
from keras import models
from keras import optimizers

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))  # 池化层
model.add(layers.Conv2D(128, (3, 3), activation='relu'))  # 卷积层
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())  # 数据降维
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))  # dense ：全连接层  相当于添加一个层
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])  # metrics衡量指标

# In[5-15]: P114保存模型
model.save('namespace.h5')

# In[4]: P127冻结卷积基  --一层或多层保持训练时参数不变
from keras.applications import VGG16

'''
weights 模型初始化的权重监测点
include_top 模型最后是否包含密集连接分类器
input_shape 输入网格中的张量形状
'''
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

conv_base.trainable = False

# 冻结直到某层的所有层
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

