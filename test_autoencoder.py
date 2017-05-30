import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

import autoencoder

# Create TF session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

data = mnist.test.images[:10]
filtered_data = autoencoder.run(sess, data)

f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(10):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    a[1][i].imshow(np.reshape(filtered_data[i], (28, 28)))
f.savefig("ae_test.png")
