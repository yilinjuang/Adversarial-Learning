import os
import time
import sys

if len(sys.argv) < 3:
    print("Error: Missing arguments")
    print("Usage: blackbox_denoise.py <dataset-basedir> <denoise-model-basedir>")
    sys.exit(1)

import numpy as np
import scipy.misc

import keras
from keras import backend
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, model_from_json
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.optimizers import SGD, Adadelta

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.utils_keras import cnn_model
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, batch_eval
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks_tf import jacobian_graph, jacobian_augmentation
from cleverhans.utils_keras import KerasModelWrapper

from gtsrb_utils import read_training_data, read_testing_data

FLAGS = flags.FLAGS


def oracle_model():
    print("Define oracle model")
    model = cnn_model(img_rows=FLAGS.nb_rows,
                      img_cols=FLAGS.nb_cols,
                      channels=FLAGS.nb_channels,
                      nb_classes=FLAGS.nb_classes)

    return model


def oracle_model_B_on_paper():
    print("Define oracle model on paper")

    model = Sequential()
    input_shape = (FLAGS.nb_rows, FLAGS.nb_cols, FLAGS.nb_channels)

    layers = [Conv2D(64, (2, 2), padding='same', input_shape=input_shape),
              MaxPooling2D(pool_size=(2, 2)),
              Conv2D(128, (2, 2), padding='same'),
              MaxPooling2D(pool_size=(2, 2)),
              Flatten(),
              Dense(256),
              Activation('relu'),
              Dense(256),
              Activation('relu'),
              Dense(FLAGS.nb_classes),
              Activation('softmax')]

    for layer in layers:
        model.add(layer)

    return model


def train_oracle(X_train, Y_train, X_val, Y_val):
    # model = oracle_model()
    model = oracle_model_B_on_paper()

    print("Train oracle model")
    val = (X_val, Y_val) if not X_val is None else None
    ada = Adadelta(lr=FLAGS.learning_rate,
                   rho=FLAGS.rho,
                   epsilon=FLAGS.train_eps,
                   decay=FLAGS.decay)
    model.compile(loss='categorical_crossentropy', optimizer=ada,
            metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=FLAGS.nb_epochs,
              batch_size=FLAGS.batch_size, validation_data=val)

    return model


def substitute_model():
    print("Define substitute model")

    model = Sequential()
    input_shape = (FLAGS.nb_rows, FLAGS.nb_cols, FLAGS.nb_channels)

    layers = [Flatten(input_shape=input_shape),
              Dense(200),
              Activation('relu'),
              Dropout(0.5),
              Dense(200),
              Activation('relu'),
              Dropout(0.5),
              Dense(FLAGS.nb_classes),
              Activation('softmax')]

    for layer in layers:
        model.add(layer)

    return model


def substitute_model_D_on_paper():
    print("Define substitute model on paper")

    model = Sequential()
    input_shape = (FLAGS.nb_rows, FLAGS.nb_cols, FLAGS.nb_channels)

    layers = [Conv2D(32, (2, 2), padding='same', input_shape=input_shape),
              MaxPooling2D(pool_size=(2, 2)),
              Conv2D(64, (2, 2), padding='same'),
              MaxPooling2D(pool_size=(2, 2)),
              Flatten(),
              Dense(200),
              Activation('relu'),
              Dense(200),
              Activation('relu'),
              Dense(FLAGS.nb_classes),
              Activation('softmax')]

    for layer in layers:
        model.add(layer)

    return model


def train_sub(sess, model, x, y, denoise_model, X_sub, Y_sub):
    # model_sub = substitute_model()
    model_sub = substitute_model_D_on_paper()
    preds_sub = model_sub(x)

    print("Train substitute model")
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
        keras.backend.set_learning_phase(1)
        model_train(sess, x, y, preds_sub, X_sub,
                    to_categorical(Y_sub, num_classes=FLAGS.nb_classes),
                    init_all=False, verbose=False, args=train_params)

        # If we are not at last substitute training iteration, augment dataset
        if rho < FLAGS.data_aug - 1:
            print("Augmenting substitute training data.")
            keras.backend.set_learning_phase(0)
            # Perform the Jacobian augmentation
            X_sub = jacobian_augmentation(sess, x, X_sub, Y_sub, grads,
                                          FLAGS.lmbda)

            print("Labeling substitute training data.")
            # Label the newly generated synthetic points using the black-box
            Y_sub = np.hstack([Y_sub, Y_sub])
            X_sub_prev = X_sub[int(len(X_sub)/2):]

            if DENOISE:
                X_sub_prev = denoise_model.predict(X_sub_prev,
                                                   verbose=1,
                                                   batch_size=FLAGS.batch_size)
            bbox_val = model.predict(X_sub_prev)
            Y_sub[int(len(X_sub)/2):] = np.argmax(bbox_val, axis=1)

    return model_sub, preds_sub


def main(argv=None):
    start_time = time.time()
    # keras.backend.set_learning_phase(0)
    tf.set_random_seed(1234)

    # Create TF session and set as Keras backend session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    # Load data
    print("Loading data...")
    if DATASET == "mnist":
        X_train, Y_train, X_test, Y_test = data_mnist()
        X_val = Y_val = None
    elif DATASET == "gtsrb":
        # http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset
        X_train, Y_train = read_training_data(os.path.join(sys.argv[1], "Final_Training/Images/"))
        # (39209, 32, 32, 3), (39209, 43)
        X_test, Y_test = read_testing_data(os.path.join(sys.argv[1], "Final_Test/Images/"))
        # (12630, 32, 32, 3), (12630, 43)
        X_val = X_train[35000:39000]
        Y_val = Y_train[35000:39000]
        X_train = X_train[:35000]
        Y_train = Y_train[:35000]
        X_test = X_test[:10000]
        Y_test = Y_test[:10000]

    if DENOISE or DENOISE_TRAIN:
        print("Load denoise model...")
        with open(os.path.join(sys.argv[2], ".json"), 'r') as f:
            denoise_model = model_from_json(f.read())
        denoise_model.load_weights(os.path.join(sys.argv[2], ".hdf5"))
    else:
        denoise_model = None

    if DENOISE_TRAIN:
        print("Denoise in train...")
        X_train = denoise_model.predict(X_train, verbose=1,
                                        batch_size=FLAGS.batch_size)
        if not X_val is None:
            X_val = denoise_model.predict(X_val, verbose=1,
                                          batch_size=FLAGS.batch_size)


    # Initialize substitute training set reserved for adversary
    X_sub = X_test[:FLAGS.holdout]
    Y_sub = np.argmax(Y_test[:FLAGS.holdout], axis=1)

    # Redefine test set as remaining samples unavailable to adversaries
    X_test = X_test[FLAGS.holdout:]
    Y_test = Y_test[FLAGS.holdout:]

    # Define input and output TF placeholders
    x = tf.placeholder(tf.float32, shape=(None, FLAGS.nb_rows, FLAGS.nb_cols,
                                          FLAGS.nb_channels))
    y = tf.placeholder(tf.float32, shape=(None, FLAGS.nb_classes))

    # Prepare ocacle model
    model = train_oracle(X_train, Y_train, X_val, Y_val)

    # Prepare substitute model
    model_sub, preds_sub = train_sub(sess, model, x, y, denoise_model,
                                     X_sub, Y_sub)

    # Initialize the Fast Gradient Sign Method (FGSM) attack object.
    fgsm_par = {'eps': FLAGS.attack_eps, 'ord': np.inf, 'clip_min': 0., 'clip_max': 1.}
    wrap = KerasModelWrapper(model_sub)
    fgsm = FastGradientMethod(wrap, sess=sess)

    # Craft adversarial examples using the substitute
    x_adv_sub = fgsm.generate_np(X_test, **fgsm_par)

    # Process test data if needed
    if DENOISE:
        print('Denoise')
        orig_test = denoise_model.predict(X_test, verbose=1,
                                          batch_size=FLAGS.batch_size)
        adv_test = denoise_model.predict(x_adv_sub, verbose=1,
                                         batch_size=FLAGS.batch_size)
    else:
        print('No denoise')
        orig_test = X_test
        adv_test = x_adv_sub
    orig_preds = model.predict(orig_test)
    adv_preds = model.predict(adv_test)

    # Calculate accuracy.
    def cal_accuracy(pred, targ):
        return np.mean(np.equal(np.argmax(pred, axis=1),
                                np.argmax(targ, axis=1)))

    orig_acc = cal_accuracy(orig_preds, Y_test)
    print("Test accuracy of oracle on real examples = {}".format(orig_acc))
    sub_acc = model_eval(sess, x, y, preds_sub, orig_test, Y_test,
                         args={'batch_size': FLAGS.batch_size})
    print("Test accuracy of substitute on real examples = {}".format(sub_acc))
    adv_acc = cal_accuracy(adv_preds, Y_test)
    print("Test accuracy of oracle on adv examples = {}".format(adv_acc))

    # Save images.
    base_dir = "{}_epoch{}_hold{}_eps{}".format(
            DATASET, FLAGS.nb_epochs, FLAGS.holdout, FLAGS.attack_eps)
    base_dir += "_D" if DENOISE else "_ND"
    base_dir += "_DT" if DENOISE_TRAIN else ""
    base_dir = os.path.join("image", base_dir)
    if not os.path.isdir(base_dir):
        print("Debug: directory {} doesn't exist, but created.".format(base_dir))
        os.mkdir(base_dir)
    for i in range(10):
        if DENOISE:
            scipy.misc.imsave('{}/denoised_orig{}.jpg'.format(base_dir, i), np.squeeze(orig_test[i]))
            scipy.misc.imsave('{}/denoised_adv{}.jpg'.format(base_dir, i), np.squeeze(adv_test[i]))
        scipy.misc.imsave('{}/orig{}.jpg'.format(base_dir, i), np.squeeze(X_test[i]))
        scipy.misc.imsave('{}/adv{}.jpg'.format(base_dir, i), np.squeeze(x_adv_sub[i]))

    print("Total running time: {:.2f}m".format((time.time()-start_time) / 60.))


if __name__ == '__main__':
    DENOISE = False
    DENOISE_TRAIN = False
    DATASET = "gtsrb"
    # DATASET = "mnist"

    # General flags
    flags.DEFINE_integer('batch_size', 128, 'Size of training/predicting batches')
    flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for training')

    # Flags related to dataset
    if DATASET == "mnist":
        flags.DEFINE_integer('nb_rows', 28, 'Number of rows in data image')
        flags.DEFINE_integer('nb_cols', 28, 'Number of columns in data image')
        flags.DEFINE_integer('nb_channels', 1, 'Number of channels in data image')
        flags.DEFINE_integer('nb_classes', 10, 'Number of classes in problem')
        flags.DEFINE_integer('nb_epochs', 10, 'Number of epochs to train oracle model')
        flags.DEFINE_integer('holdout', 150, 'Test set holdout for adversary')
    elif DATASET == "gtsrb":
        flags.DEFINE_integer('nb_rows', 32, 'Number of rows in data image')
        flags.DEFINE_integer('nb_cols', 32, 'Number of columns in data image')
        flags.DEFINE_integer('nb_channels', 3, 'Number of channels in data image')
        flags.DEFINE_integer('nb_classes', 43, 'Number of classes in problem')
        flags.DEFINE_integer('nb_epochs', 50, 'Number of epochs to train oracle model')  #######################
        flags.DEFINE_integer('holdout', 500, 'Test set holdout for adversary')  #######################
    else:
        print("Error: unknown dataset {}.".format(DATASET))
        sys.exit(1)

    # Flags related to oracle
    flags.DEFINE_float('rho', 0.95, '')
    flags.DEFINE_float('train_eps', 1e-08, '')
    flags.DEFINE_float('decay', 0.0, '')

    # Flags related to substitute
    flags.DEFINE_integer('data_aug', 6, 'Number of substitute data augmentations')
    flags.DEFINE_integer('nb_epochs_s', 10, 'Training epochs for substitute')
    flags.DEFINE_float('lmbda', 0.1, 'Lambda from arxiv.org/abs/1602.02697')

    # Flags related to attack
    flags.DEFINE_float('attack_eps', 0.1, 'Epsilon of FGSM attack')  #######################

    app.run()
