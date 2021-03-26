import tensorflow as tf
# 手写数字数据集
from sklearn.datasets import load_digits
# 用于将矩阵随机划分为训练子集和测试子集
from sklearn.model_selection import train_test_split
# 标签二值化LabelBinarizer
from sklearn.preprocessing import LabelBinarizer

# load data
digits = load_digits()
x = digits.data
y = digits.target
'''
对部分数据先拟合fit，找到该part的整体指标，
如均值、方差、最大值最小值等等（根据具体转换的目的），
然后对该trainData进行转换transform，从而实现数据的标准化、归一化等等。
'''
y = LabelBinarizer().fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


# 输入 输入尺寸，输出尺寸，层名，激活函数
def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    tf.summary.histogram(layer_name + "/outputs", outputs)
    return outputs


keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 64])  # 8 x 8
ys = tf.placeholder(tf.float32, [None, 10])

"""
tf.nn ：提供神经网络相关操作的支持，包括卷积操作（conv）、池化操作（pooling）、
        归一化、loss、分类操作、embedding、RNN、Evaluation。
tf.layers：主要提供的高层的神经网络，
        主要和卷积相关的，tf.nn会更底层一些。
tf.contrib：tf.contrib.layers提供够将计算图中的 
        网络层、正则化、摘要操作、是构建计算图的高级操作，
        但是tf.contrib包含不稳定和实验代码，有可能以后API会改变。
"""
# 输入 输入尺寸，输出尺寸，层名，激活函数
l1 = add_layer(xs, 64, 50, "l1", activation_function=tf.nn.tanh)
prediction = add_layer(l1, 50, 10, "l2", activation_function=tf.nn.softmax)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))
tf.summary.scalar("loss", cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter("logs/tf11_Dropout/train", sess.graph)
test_writer = tf.summary.FileWriter("logs/tf11_Dropout/test", sess.graph)

init = tf.global_variables_initializer()
sess.run(init)

for i in range(500):
    sess.run(train_step, feed_dict={xs: x_train, ys: y_train, keep_prob: 0.5})
    if i % 50 == 0:
        train_result = sess.run(merged, feed_dict={xs: x_train, ys: y_train, keep_prob: 1})
        test_result = sess.run(merged, feed_dict={xs: x_test, ys: y_test, keep_prob: 1})
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)

"""
tensorboard --logdir="DL_Try\\tensorflow\\logs\\tf11_Dropout"
"""
