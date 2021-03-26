from __future__ import print_function
import tensorflow as tf
import numpy as np

"""
创建数据
搭建模型
计算误差
传播误差
训练
"""

# create data
# 生成一百个随机数，这些数字 构成 1*100一维的数组
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

### create tensorflow structure start ###
# 初始化变量的值
Weights = tf.Variable(tf.random.uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases

"""
tf.reduce_mean 函数
用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，
主要用作降维或者计算tensor（图像）的平均值。

tf.square()是对a里的每一个元素求平方
"""
loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(loss)
### create tensorflow structure end ###

sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
