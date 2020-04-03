import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


b = tf.Variable(tf.zeros((100,)))
#print(b)
w = tf.Variable(tf.random_uniform((784, 100), -1, 1))

x = tf.placeholder(tf.float32, (100, 784))

h = tf.nn.relu(tf.matmul(x, w)+b)
# print(w)
# tf.get_default_graph().get_operations()