import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

b = tf.Variable(tf.zeros((3,)))
W = tf.Variable(tf.random_uniform((5, 3), -1, 1))

x = tf.placeholder(tf.float32, (3, 5))

h = tf.nn.relu(tf.matmul(x, W) + b)
initialize = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(initialize)
    print("W", W.eval())
    print("b", b.eval())
    print("the output is", sess.run(h, {x: np.random.randn(3, 5)}))
