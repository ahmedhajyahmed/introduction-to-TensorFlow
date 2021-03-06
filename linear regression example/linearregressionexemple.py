# -*- coding: utf-8 -*-
"""linearRegressionExemple.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fTs6RDAGHbLfJn_WVqMhNYOrkMjwhWUJ
"""

import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
# import matplotlib
from matplotlib import pyplot as plt

'''
  linear regression find : find the best linear fit to our data
'''


def generate_dataset():
    # date is generated by y = 2x + e
    # where e is sampled form a normal distribution
    x_batch = np.linspace(-1, 1, 101)  # return 101 element in [-1,1]
    y_batch = 2 * x_batch + np.random.randn(*x_batch.shape) * 0.3
    return x_batch, y_batch


def linear_regression():
    x = tf.placeholder(tf.float32, shape=(None,), name='x')
    y = tf.placeholder(tf.float32, shape=(None,), name='y')

    with tf.variable_scope('lreg') as scope:
        w = tf.Variable(np.random.normal(), name='w')
        y_pred = tf.multiply(w, x)
        loss = tf.reduce_mean(tf.square(y - y_pred))
    return x, y, y_pred, loss


def run():
    x_batch, y_batch = generate_dataset()

    x, y, y_pred, loss = linear_regression()
    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        feed_dict = {x: x_batch, y: y_batch}
        for _ in range(30):
            loss_var, _ = sess.run([loss, optimizer], feed_dict)
            print('the loss is ', loss_var.mean())

        y_pred_batch = sess.run(y_pred, {x: x_batch})

    plt.figure(1)  # create a new figure
    plt.scatter(x_batch, y_batch)
    plt.plot(x_batch, y_pred_batch)
    plt.savefig('plot.png')


if __name__ == '__main__':
    run()
