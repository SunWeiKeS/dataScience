import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import warnings
warnings.filterwarnings("ignore")


matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2],
                       [2]])
product = tf.matmul(matrix1, matrix2)  # matrix multiply np.dot(m1, m2)

# method1
sess= tf.Session()
result= sess.run(product)
print(result)
sess.close()
# method2
with tf.Session() as sess:
    result2=sess.run(product)
    print(result2)