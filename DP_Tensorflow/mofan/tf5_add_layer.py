import tensorflow as tf


def add_layer(inputs, in_size, out_sinze, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_sinze]))
    biases = tf.Variable(tf.zeros([1, out_sinze]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs