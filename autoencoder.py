from __future__ import division, print_function, absolute_import

import sys

import numpy as np
import tensorflow as tf
from skimage.util import random_noise

# Network Parameters
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 128 # 2nd layer num features
n_input = 784 # MNIST data input (img shape: 28*28)

# Training Parameters
training_epochs = int(sys.argv[1])
# training_epochs = 5
# Model
model_name = "noise_epoch{}_{}_{}".format(training_epochs, n_hidden_1, n_hidden_2)
model_path = "./ae_model/{}.model".format(model_name)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

is_restored = False
def restore(sess, model):
    global is_restored, model_name, model_path
    if not is_restored:
        is_restored = True
        model_name = model
        model_path = "./ae_model/{}.model".format(model_name)
        saver.restore(sess, model_path)
        print("Model restored from file: %s" % model_path)

def add_noise(images):
    new_images = []
    for i in range(images.shape[0]):
        new_images.append(random_noise(images[i], mode='gaussian', var=0.01))
    return np.asarray(new_images)

def train():
    # Parameters
    learning_rate = 0.01
    batch_size = 256
    display_step = 1

    # Targets (Labels) are the input data.
    y_true = Y

    # Define loss and optimizer, minimize the squared error
    cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

    # Create TF session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    # Import MNIST data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/", one_hot=True)

    total_batch = int(mnist.train.num_examples/batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, train_c = sess.run([optimizer, cost], feed_dict={X: add_noise(batch_xs),
                                                                Y: batch_xs})
        val_c = sess.run(cost, feed_dict={X: add_noise(mnist.validation.images),
                                          Y: mnist.validation.images})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch {:04d}, Train cost = {:.6f}, Val cost = {:.6f}".format(epoch+1, train_c, val_c))

    print("Optimization Finished!")
    test_c = sess.run(cost, feed_dict={X: add_noise(mnist.test.images),
                                       Y: mnist.test.images})
    log_file = "log/ae_{}.log".format(model_name)
    with open(log_file, "a") as f:
        content = "Test Cost = {:.6f}\n".format(test_c)
        print(content, end="")
        f.write(content)

    # Save model weights to disk
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)

    visualize(sess, add_noise(mnist.test.images[:10]), "train")

def visualize(sess, x, fig_mode):
    """
    sess:       tensorflow session.
    x:          mnist data. shape = (n, 784).
    fig_mode:   figure mode.

    return
    """
    import matplotlib
    matplotlib.use("agg")
    import matplotlib.pyplot as plt

    # Applying encode and decode over test set
    encode_decode = sess.run(y_pred, feed_dict={X: x})

    num_data = 10
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, num_data, figsize=(num_data, 2))
    for i in range(num_data):
        a[0][i].imshow(np.reshape(x[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    global model_name
    fig_path = "./image/ae_{}_{}.png".format(fig_mode, model_name)
    f.savefig(fig_path)


def run(sess, x):
    """
    sess:   tensorflow session.
    x:      mnist data. shape = (n, 784).

    return  filtered mnist data. shape = (n, 784).
    """
    return sess.run(y_pred, feed_dict={X: x})


if __name__ == "__main__":
    is_restored = True
    train()
