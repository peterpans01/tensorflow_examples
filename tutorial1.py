import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


# First, load the image
filename = "MarshOrchid.jpg"
image = mpimg.imread(filename)

# Print out its shape
# print(image.shape)


x = tf.Variable(image, name='x')
model = tf.global_variables_initializer()
with tf.Session() as session:
    x = tf.transpose(x, perm=[1, 0, 2])
    session.run(model)
    result = session.run(x)

plt.imshow(result)
plt.show()

# data = np.random.randint(10, size=5)
# x = tf.constant(data, name = 'x')
# y = tf.Variable(5* x**2 - 3 * x + 5, name = 'y')
#
# model = tf.global_variables_initializer()
# with tf.Session() as ss:
#     ss.run(model)
#     print(ss.run(y))
# data = np.random.randint(10, size=5)
#x = tf.constant(data, name = 'x')

# data = np.random.randint(10, size=5)
# x = tf.Variable(data, name='x')
# #y = tf.Variable(x + 3, name='y')
#
#
#
#
#
# with tf.Session() as session:
#     merged = tf.summary.merge_all()
#     writer = tf.summary.FileWriter("/tmp/basic", session.graph)
#     model = tf.global_variables_initializer()
#     session.run(model)
#     for i in range(5):
#         x = (x + np.random.randint(10, size=5))/(i+ 1)
#         print(session.run(x))
