import tensorflow as tf
import numpy as np

# Save to file
# remember to define the same dtype and shape when restore
# W = tf.Variable([[1, 2, 3], [3, 4, 5]], dtype=tf.float32, name="weights")
# b = tf.Variable([[1, 2, 3]], dtype=tf.float32, name="biases")
#
# init=tf.global_variables_initializer()
# saver=tf.train.Saver()
#
# sess=tf.Session()
# sess.run(init)
# save_path = saver.save(sess,"my_net/tf13_Saver_Restore/tf13_saver_net.ckpt")
# print("Save to path: ",save_path)

# restore variable
# redefine the same shape and same type for your variables

W = tf.Variable(np.arange(3, 9).reshape((2, 3)), dtype=tf.float32, name="weights")
b = tf.Variable(np.arange(3, 6).reshape((1, 3)), dtype=tf.float32, name="biases")

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, "my_net/tf13_Saver_Restore/tf13_saver_net.ckpt")
print("weights:", sess.run(W))
print("biases:", sess.run(b))
