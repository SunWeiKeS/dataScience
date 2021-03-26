import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)

# fake data
# np.newaxis 的作用就是在这一位置增加一个一维，这一位置指的是np.newaxis所在的位置
x = np.linspace(-1, 1, 100)[:, np.newaxis]  # shape (100, 1)
noise = np.random.normal(0, 0.01, size=x.shape)
y = np.power(x, 2) + noise  # shape (100, 1) + some noise


def save():
    print("this is save")
    # build neural network
    tf_x = tf.placeholder(tf.float32, x.shape)  # input x
    tf_y = tf.placeholder(tf.float32, y.shape)  # input y
    l = tf.layers.dense(tf_x, 10, tf.nn.relu)  # hidden layer
    o = tf.layers.dense(l, 1)  # output layer
    loss = tf.losses.mean_squared_error(tf_y, o)  # compute cost
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())  # initialize var in graph

    saver = tf.train.Saver()  # define a saver for saving and restoring

    for step in range(100):
        sess.run(train_op, {tf_x: x, tf_y: y})

    saver.save(sess, "my_net/tf13_simple/", write_meta_graph=False)  # meta_graph is not recommend

    # plotting
    pred, l = sess.run([o, loss], {tf_x: x, tf_y: y})

    plt.figure(1, figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(x, y)
    plt.plot(x, pred, "r-", lw=5)
    plt.text(-1, 1.2, 'Save Loss=%.4f' % l, fontdict={'size': 15, 'color': 'red'})
    plt.show()


def reload():
    print("this is reload")
    # build entire net again and restore
    tf_x = tf.placeholder(tf.float32, x.shape)  # input x
    tf_y = tf.placeholder(tf.float32, y.shape)  # input y
    l_ = tf.layers.dense(tf_x, 10, tf.nn.relu)  # hidden layer
    o_ = tf.layers.dense(l_, 1)  # output layer
    loss_ = tf.losses.mean_squared_error(tf_y, o_)  # compute cost

    sess = tf.Session()
    # don't need to initialize variables,just restoring trained variables
    saver = tf.train.Saver()  # define a saver for saving and restoring
    saver.restore(sess, "my_net/tf13_simple/")

    # plotting
    pred, l = sess.run([o_, loss_], {tf_x: x, tf_y: y})
    plt.subplot(122)
    plt.scatter(x, y)
    plt.plot(x, pred, 'r-', lw=5)
    plt.text(-1, 1.2, 'Reload Loss=%.4f' % l, fontdict={'size': 15, 'color': 'red'})
    plt.show()


#
if __name__ == '__main__':
    save()
    tf.reset_default_graph()
    reload()
