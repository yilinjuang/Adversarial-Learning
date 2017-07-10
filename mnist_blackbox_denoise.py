from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from PIL import Image, ImageDraw
import scipy.misc

import keras
from keras import backend
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Flatten, Activation, Dropout
from keras.optimizers import SGD, Adadelta

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.utils_keras import cnn_model
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, batch_eval
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks_tf import jacobian_graph, jacobian_augmentation

from gtsrb_utils import read_training_data, read_testing_data

import time
import sys

Denoise=0
DenoiseInTrain=0
dataset = "gtsrb"#"mnist"

FLAGS = flags.FLAGS
# General flags
if dataset == "mnist":
    flags.DEFINE_integer('nb_classes', 10, 'Number of classes in problem')
elif dataset == "gtsrb":
    flags.DEFINE_integer('nb_classes', 43, 'Number of classes in problem')
else:
    print("Error: Invalid dataset.")
    sys.exit(1)
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')

# Flags related to oracle
flags.DEFINE_integer('nb_epochs', 10, 'Number of epochs to train model')

# Flags related to substitute
flags.DEFINE_integer('holdout', 1000, 'Test set holdout for adversary')
flags.DEFINE_integer('data_aug', 6, 'Nb of times substitute data augmented')
flags.DEFINE_integer('nb_epochs_s', 10, 'Training epochs for each substitute')
flags.DEFINE_float('lmbda', 0.1, 'Lambda in https://arxiv.org/abs/1602.02697')


def setup_tutorial():
    """
    Helper function to check correct configuration of tf and keras for tutorial
    :return: True if setup checks completed
    """

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    if not hasattr(backend, "tf"):
        raise RuntimeError("This tutorial requires keras to be configured"
                           " to use the TensorFlow backend.")

    # Image dimensions ordering should follow the Theano convention
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' "
              "to 'th', temporarily setting to 'tf'")

    return True


def prep_bbox(sess, x, y, X_train, Y_train, X_test, Y_test):
    """
    Define and train a model that simulates the "remote"
    black-box oracle described in the original paper.
    :param sess: the TF session
    :param x: the input placeholder for MNIST
    :param y: the ouput placeholder for MNIST
    :param X_train: the training data for the oracle
    :param Y_train: the training labels for the oracle
    :param X_test: the testing data for the oracle
    :param Y_test: the testing labels for the oracle
    :return:
    """

    # Define TF model graph (for the black-box model)
    if dataset == "mnist":
        model = cnn_model(img_rows=28, img_cols=28, channels=1, nb_classes=10)
    elif dataset == "gtsrb":
        model = cnn_model(img_rows=32, img_cols=32, channels=3, nb_classes=43)
    predictions = model(x)
    print("Defined TensorFlow model graph.")

    # Train an MNIST model
    train_params = {
        'nb_epochs': FLAGS.nb_epochs,
        'batch_size': FLAGS.batch_size,
        'learning_rate': FLAGS.learning_rate
    }
    model_train(sess, x, y, predictions, X_train, Y_train,
                verbose=False, args=train_params)

    # Print out the accuracy on legitimate data
    eval_params = {'batch_size': FLAGS.batch_size}
    accuracy = model_eval(sess, x, y, predictions, X_test, Y_test,
                          args=eval_params)
    print('Test accuracy of black-box on legitimate test '
          'examples: ' + str(accuracy))

    return model, predictions


def substitute_model(img_rows=28, img_cols=28, channels=1, nb_classes=10):
    """
    Defines the model architecture to be used by the substitute
    :param img_rows: number of rows in input
    :param img_cols: number of columns in input
    :param nb_classes: number of classes in output
    :return: keras model
    """
    model = Sequential()

    # Find out the input shape ordering
    if keras.backend.image_dim_ordering() == 'th':
        input_shape = (channels, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, channels)

    # Define a fully connected model (it's different than the black-box)
    layers = [Flatten(input_shape=input_shape),
              Dense(200),
              Activation('relu'),
              Dropout(0.5),
              Dense(200),
              Activation('relu'),
              Dropout(0.5),
              Dense(nb_classes),
              Activation('softmax')]

    for layer in layers:
        model.add(layer)

    return model


def train_sub(sess, model, x, y, bbox_preds, X_sub, Y_sub):
    """
    This function creates the substitute by alternatively
    augmenting the training data and training the substitute.
    :param sess: TF session
    :param x: input TF placeholder
    :param y: output TF placeholder
    :param bbox_preds: output of black-box model predictions
    :param X_sub: initial substitute training data
    :param Y_sub: initial substitute training labels
    :return:
    """
    # Define TF model graph (for the black-box model)
    if dataset == "mnist":
        model_sub = substitute_model(img_rows=28, img_cols=28, channels=1, nb_classes=10)
    elif dataset == "gtsrb":
        model_sub = substitute_model(img_rows=32, img_cols=32, channels=3, nb_classes=43)
    preds_sub = model_sub(x)
    print("Defined TensorFlow model graph for the substitute.")

    # Define the Jacobian symbolically using TensorFlow
    grads = jacobian_graph(preds_sub, x, FLAGS.nb_classes)

    # Train the substitute and augment dataset alternatively
    for rho in range(FLAGS.data_aug):
        print("Substitute training epoch #" + str(rho))
        train_params = {
            'nb_epochs': FLAGS.nb_epochs_s,
            'batch_size': FLAGS.batch_size,
            'learning_rate': FLAGS.learning_rate
        }
        model_train(sess, x, y, preds_sub, X_sub, to_categorical(Y_sub, num_classes=FLAGS.nb_classes),
                    init_all=False, verbose=False, args=train_params)

        # If we are not at last substitute training iteration, augment dataset
        if rho < FLAGS.data_aug - 1:
            print("Augmenting substitute training data.")
            # Perform the Jacobian augmentation
            X_sub = jacobian_augmentation(sess, x, X_sub, Y_sub, grads,
                                          FLAGS.lmbda)

            print("Labeling substitute training data.")
            # Label the newly generated synthetic points using the black-box
            Y_sub = np.hstack([Y_sub, Y_sub])
            X_sub_prev = X_sub[int(len(X_sub)/2):] #(150, 28, 28, 1)
            #eval_params = {'batch_size': FLAGS.batch_size}

            if Denoise == 1:
            ##### Denoise part start #######
                with open('Denoise_large_{}.json'.format(dataset),'r') as f:
                    De_model= model_from_json(f.read())

                De_model.load_weights('Best_weights_large_{}.hdf5'.format(dataset))
                De_out=De_model.predict(X_sub_prev, verbose=1,batch_size=500)
                bbox_val=model.predict(De_out)
                print('Denoise')
                ##### Denoise part end #######
            else:
                bbox_val=model.predict(X_sub_prev)
                print('Non-Denoise')

            '''
            bbox_val = batch_eval(sess, [x], [bbox_preds], [X_sub_prev],
                                  args=eval_params)[0]
            '''
            # Note here that we take the argmax because the adversary
            # only has access to the label (not the probabilities) output
            # by the black-box model
            Y_sub[int(len(X_sub)/2):] = np.argmax(bbox_val, axis=1)

    return model_sub, preds_sub


def main(argv=None):
    """
    MNIST cleverhans tutorial
    :return:
    """
    start_time = time.time()
    keras.layers.core.K.set_learning_phase(0)

    # Perform tutorial setup
    assert setup_tutorial()

    # Create TF session and set as Keras backend session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # sess = tf.Session()
    keras.backend.set_session(sess)

    # Load data
    if dataset == "mnist":
        # Get MNIST data
        X_train, Y_train, X_test, Y_test = data_mnist()
    elif dataset == "gtsrb":
        # Get GTSRB data
        # http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset
        X_train, Y_train = read_training_data("/data/yljuang/GTSRB/Final_Training/Images/")  # (39209, 32, 32, 3), (39209, 43)
        X_test, Y_test = read_testing_data("/data/yljuang/GTSRB/Final_Test/Images/")# (12630, 32, 32, 3), (12630, 43)

    if DenoiseInTrain==1:
        print('Denoise in training...')
        with open('Denoise_large_{}.json'.format(dataset),'r') as f:
            De_model= model_from_json(f.read())
        De_model.load_weights('Best_weights_large_{}.hdf5'.format(dataset))

        X_train=De_model.predict(X_train, verbose=1,batch_size=500) #(60000, 28, 28, 1)



    # Initialize substitute training set reserved for adversary
    X_sub = X_test[:FLAGS.holdout]
    Y_sub = np.argmax(Y_test[:FLAGS.holdout], axis=1)

    # Redefine test set as remaining samples unavailable to adversaries
    X_test = X_test[FLAGS.holdout:]
    Y_test = Y_test[FLAGS.holdout:]

    # Define input and output TF placeholders
    if dataset == "mnist":
        x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
        y = tf.placeholder(tf.float32, shape=(None, 10))
    elif dataset == "gtsrb":
        x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
        y = tf.placeholder(tf.float32, shape=(None, 43))

    # Simulate the black-box model locally
    # You could replace this by a remote labeling API for instance
    print("Preparing the black-box model.")
    # model, bbox_preds = prep_bbox(sess, x, y, X_train, Y_train, X_test, Y_test)
    if dataset == "mnist":
        model = cnn_model(img_rows=28, img_cols=28, channels=1, nb_classes=10)
    elif dataset == "gtsrb":
        model = cnn_model(img_rows=32, img_cols=32, channels=3, nb_classes=43)

    ada = Adadelta(lr= 0.1, rho=0.95, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=ada)
    model.fit(X_train, Y_train, epochs=50, batch_size=128)

    print("Training the substitute model.")
    # Train substitute using method from https://arxiv.org/abs/1602.02697
    # model_sub, preds_sub = train_sub(sess, model, x, y, bbox_preds, X_sub, Y_sub)
    model_sub, preds_sub = train_sub(sess, model, x, y, "", X_sub, Y_sub)

    # Initialize the Fast Gradient Sign Method (FGSM) attack object.
    fgsm_par = {'eps': 0.3, 'ord': np.inf, 'clip_min': 0., 'clip_max': 1.}
    fgsm = FastGradientMethod(model_sub, sess=sess)

    # Craft adversarial examples using the substitute
    #eval_params = {'batch_size': FLAGS.batch_size}
    x_adv_sub = fgsm.generate_np(X_test, **fgsm_par) #(9850, 28, 28, 1)

    if Denoise == 1:
        ##### Denoise part start #######
        print('Denoise')
        with open('Denoise_large_{}.json'.format(dataset),'r') as f:
            De_model= model_from_json(f.read())
        De_model.load_weights('Best_weights_large_{}.hdf5'.format(dataset))

        Re_out=De_model.predict(X_test, verbose=1,batch_size=500) #(9850, 28, 28, 1)
        real_preds=model.predict(Re_out) #(9850, 10)


        De_out=De_model.predict(x_adv_sub, verbose=1,batch_size=500) #(9850, 28, 28, 1)
        adv_preds=model.predict(De_out) #(9850, 10)

        ##### Denoise part end #######
    else:
        print('Non-Denoise')
        real_preds=model.predict(X_test) #(9850, 10)

        adv_preds=model.predict(x_adv_sub)

    real_accuracy = np.mean(np.equal(np.argmax(real_preds, axis=1),np.argmax(Y_test, axis=1)))
    print('Test accuracy of oracle on real examples :' + str(real_accuracy))

    adv_accuracy = np.mean(np.equal(np.argmax(adv_preds, axis=1),np.argmax(Y_test, axis=1)))
    print('Test accuracy of oracle on adversarial examples generated using the substitute: ' + str(adv_accuracy))

    # Evaluate the accuracy of the "black-box" model on adversarial examples
    '''
    accuracy = model_eval(sess, x, y, model.predict(De_out), X_test, Y_test,
                          args=eval_params)
    '''
    # DIR = "denoise50eps03hold1000"
    DIR = "undenoise50eps03hold1000"
    for i in range(10):
        if Denoise == 1:
            scipy.misc.imsave('image/{}/Denoise_Clean{}.jpg'.format(DIR, i), np.squeeze(Re_out[i]))
            scipy.misc.imsave('image/{}/Denoise_Noisy{}.jpg'.format(DIR, i), np.squeeze(De_out[i]))
        scipy.misc.imsave('image/{}/Clean{}.jpg'.format(DIR, i), np.squeeze(X_test[i]))
        scipy.misc.imsave('image/{}/Noisy{}.jpg'.format(DIR, i), np.squeeze(x_adv_sub[i]))

    end_time = time.time()
    print ('The code for this file ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    app.run()
