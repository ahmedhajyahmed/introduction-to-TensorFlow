import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

x = tf.placeholder(tf.float32, shape=(3, 5), name='x')
label = tf.placeholder(tf.float32, shape=(3, 3), name='label')

with tf.variable_scope('variable') as scope:
    b = tf.Variable(tf.zeros((3,)))
    W = tf.Variable(tf.random_uniform((5, 3), -1, 1))
    h = tf.nn.sigmoid(tf.matmul(x, W) + b)
    # prediction = tf.nn.softmax(h)
    cross_entropy = - 1/5 * tf.reduce_sum(label * tf.log(h) + (1 - label)*tf.log(1-h))

x_batch = np.random.randn(3, 5)
label_batch = np.random.randn(3, 3)
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
initialize = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(initialize)
    feed_dict = {label: label_batch, x: x_batch}
    for i in range(10):
        optimizer, loss = sess.run([train_step, cross_entropy], feed_dict)
        # print("the cross_entropy", cross_entropy.eval())
        print("the output is", loss)


