import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
# one_hot 例如在n为4的情况下，标签2对应的onehot标签就是 0.0 0.0 1.0 0.0
MNIST_DATA = "D:\\Programs\\Workspaces\\Python\\pycharm\\demo2019\\Tips\\MNIST_data"
mnist = input_data.read_data_sets(MNIST_DATA, one_hot=True)


def add_layer(inputs, in_size, out_size, activation_function=None):
    # 调用add_layer函数搭建一个最简单的训练网络结构，只有输入层和输出层
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    # arg_max 最大值
    correct_prediction = tf.equal(tf.arg_max(y_pre, 1), tf.arg_max(v_ys, 1))
    # tf.cast()函数的作用是执行tensorflow中张量数据类型转换
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])  # 28 x 28
# 每张图片都表示一个数字，所以我们的输出是数字0到9，共10类。
ys = tf.placeholder(tf.float32, [None, 10])

# add out layer nn.softmax一般用来做分类
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

# the error between prediction and read data
# 交叉熵用来衡量预测值和真实值的相似程度，如果完全相同，它们的交叉熵等于零。
# reduce_mean 函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))
# train方法（最优化算法）采用梯度下降法。
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    # 现在开始train，每次只取100张图片，免得数据太多训练太慢。
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))
