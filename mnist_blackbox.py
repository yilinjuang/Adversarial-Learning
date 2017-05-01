from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import pickle
import sys

import keras
from keras import backend
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.utils import cnn_model
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, batch_eval
from cleverhans.attacks import fgsm
from cleverhans.attacks_tf import jacobian_graph, jacobian_augmentation

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


# Argparse
def is_boost():
    return args.adaboost or args.gradboost or args.randforest

def check_args():
    if args.max_dep is None or args.n_est is None:
        print("Error: max_dep, n_est required.")
        sys.exit(1)
parser = argparse.ArgumentParser(description="Black-box attack against CNN/Adaptive Boosting/Gradient Boosting/Random Forest.")
classifier = parser.add_mutually_exclusive_group(required=True)
classifier.add_argument('--cnn', '-c', action='store_true', help="use cnn as black-box")
classifier.add_argument('--adaboost', '-a', action='store_true', help="use adaptive boosting as black-box")
classifier.add_argument('--gradboost', '-g', action='store_true', help="use gradient boosting as black-box")
classifier.add_argument('--randforest', '-r', action='store_true', help="use random forest as black-box")
parser.add_argument("--max_dep", "-d", type=int,
        help="the max depth of decision tree in boosting")
parser.add_argument("--n_est", "-n", type=int,
        help="the number of estimators in boosting")
args = parser.parse_args()
if args.cnn:
    model_name = "cnn"
elif args.adaboost:
    check_args()
    model_name = "adaboost_dep{}_est{}".format(args.max_dep, args.n_est)
elif args.gradboost:
    check_args()
    model_name = "gradboost_dep{}_est{}".format(args.max_dep, args.n_est)
elif args.randforest:
    check_args()
    model_name = "randforest_dep{}_est{}".format(args.max_dep, args.n_est)
else:
    print("Error: unkwown classifier.")
    sys.exit(1)
print(model_name)

FLAGS = flags.FLAGS

# General flags
flags.DEFINE_integer('nb_classes', 10, 'Number of classes in problem')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')

# Flags related to substitute
flags.DEFINE_integer('holdout', 100, 'Test set holdout for adversary')
flags.DEFINE_integer('data_aug', 6, 'Nb of times substitute data augmented')
flags.DEFINE_integer('nb_epochs_s', 6, 'Training epochs for each substitute')
flags.DEFINE_float('lmbda', 0.2, 'Lambda in https://arxiv.org/abs/1602.02697')


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


def prep_cnn_bbox(sess, x, y, X_train, Y_train, X_test, Y_test):
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
    model = cnn_model()
    predictions = model(x)

    # Train an MNIST model
    train_params = {
        'nb_epochs': 6,
        'batch_size': FLAGS.batch_size,
        'learning_rate': FLAGS.learning_rate
    }
    model_train(sess, x, y, predictions, X_train, Y_train,
                verbose=False, args=train_params)

    # Print out the accuracy on legitimate data
    eval_params = {'batch_size': FLAGS.batch_size}
    accuracy = model_eval(sess, x, y, predictions, X_test, Y_test,
                          args=eval_params)
    print("Test accuracy = {}".format(accuracy))

    return predictions


def prep_boost_bbox(X_train, Y_train, X_test, Y_test):
    if args.adaboost:
        model = AdaBoostClassifier(
                DecisionTreeClassifier(max_depth=args.max_dep),
                n_estimators=args.n_est,
                learning_rate=1.0)
    elif args.gradboost:
        model = GradientBoostingClassifier(max_depth=args.max_dep,
                                           n_estimators=args.n_est,
                                           learning_rate=1.0)
    elif args.randforest:
        model = RandomForestClassifier(max_depth=args.max_dep,
                                       n_estimators=args.n_est,
                                       max_features=None,
                                       n_jobs=-1)
    else:
        print("Error: unkwown boosting classifier.")
        sys.exit(1)

    x_train = X_train.reshape(X_train.shape[0], -1)
    y_train = np.argmax(Y_train, axis=1)
    x_test = X_test.reshape(X_test.shape[0], -1)
    y_test = np.argmax(Y_test, axis=1)

    model.fit(x_train, y_train)

    # Dump model and log.
    model_file = "model/{}.model".format(model_name)
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    log_file = "log/{}.log".format(model_name)
    if getattr(model, "staged_predict", None):
        with open(log_file, "w") as f:
            for predict in model.staged_predict(x_test):
                f.write("{}\n".format(accuracy_score(predict, y_test)))

    accuracy = model.score(x_test, y_test)
    print("Test accuracy = {}".format(accuracy))
    with open(log_file, "a") as f:
        f.write("Test accuracy = {}\n".format(accuracy))

    return model

def substitute_model(img_rows=28, img_cols=28, nb_classes=10):
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
        input_shape = (1, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 1)

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


def train_substitute(sess, x, y, bbox_preds, X_sub, Y_sub):
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
    model_sub = substitute_model()
    preds_sub = model_sub(x)

    # Define the Jacobian symbolically using TensorFlow
    grads = jacobian_graph(preds_sub, x, FLAGS.nb_classes)

    # Train the substitute and augment dataset alternatively
    for rho in range(FLAGS.data_aug):
        print("Epoch #" + str(rho))
        train_params = {
            'nb_epochs': FLAGS.nb_epochs_s,
            'batch_size': FLAGS.batch_size,
            'learning_rate': FLAGS.learning_rate
        }
        model_train(sess, x, y, preds_sub, X_sub, to_categorical(Y_sub),
                    verbose=False, args=train_params)

        # If we are not at last substitute training iteration, augment dataset
        if rho < FLAGS.data_aug - 1:
            # Perform the Jacobian augmentation
            X_sub = jacobian_augmentation(sess, x, X_sub, Y_sub, grads,
                                          FLAGS.lmbda, keras_phase=keras.
                                          backend.learning_phase())

            # Label the newly generated synthetic points using the black-box
            Y_sub = np.hstack([Y_sub, Y_sub])
            X_sub_prev = X_sub[int(len(X_sub)/2):]
            if args.cnn:
                eval_params = {'batch_size': FLAGS.batch_size}
                bbox_val = batch_eval(sess, [x], [bbox_preds], [X_sub_prev],
                                      args=eval_params)[0]
                # Note here that we take the argmax because the adversary
                # only has access to the label (not the probabilities) output
                # by the black-box model
                Y_sub_prev = np.argmax(bbox_val, axis=1)
            elif is_boost():
                x_sub_prev = X_sub_prev.reshape(X_sub_prev.shape[0], -1)
                Y_sub_prev = bbox_preds.predict(x_sub_prev)
            else:
                print("Error: unknown classifier.")
            Y_sub[int(len(X_sub)/2):] = Y_sub_prev

    return preds_sub


def main(argv=None):
    """
    MNIST cleverhans tutorial
    :return:
    """

    # Perform tutorial setup
    assert setup_tutorial()

    # Create TF session and set as Keras backend session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    # Get MNIST data
    X_train, Y_train, X_test, Y_test = data_mnist()

    # Initialize substitute training set reserved for adversary
    X_sub = X_test[:FLAGS.holdout]
    Y_sub = np.argmax(Y_test[:FLAGS.holdout], axis=1)

    # Redefine test set as remaining samples unavailable to adversaries
    X_test = X_test[FLAGS.holdout:]
    Y_test = Y_test[FLAGS.holdout:]

    # Define input and output TF placeholders
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    print("Preparing the black-box model.")
    if args.cnn:
        bbox = prep_cnn_bbox(sess, x, y, X_train, Y_train, X_test, Y_test)
    elif is_boost():
        bbox = prep_boost_bbox(X_train, Y_train, X_test, Y_test)
    else:
        print("Error: unknown classifier.")
        sys.exit(1)

    print("Training the substitute model.")
    substitute_preds = train_substitute(sess, x, y, bbox, X_sub, Y_sub)

    # Craft adversarial examples using the substitute
    print("Crafting the adversarial examples.")
    adv_x = fgsm(x, substitute_preds, eps=0.3)
    eval_params = {'batch_size': FLAGS.batch_size}
    X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test], args=eval_params)

    # Dump adversarial examples.
    example_file = "example/{}.data".format(model_name)
    with open(example_file, "wb") as f:
        pickle.dump(X_test_adv, f)

    # Evaluate the accuracy of the "black-box" model on adversarial examples
    if args.cnn:
        accuracy = model_eval(sess, x, y, bbox, X_test_adv, Y_test,
                              args=eval_params)
    elif is_boost():
        x_test_adv = X_test_adv.reshape(X_test_adv.shape[0], -1)
        y_test = np.argmax(Y_test, axis=1)
        accuracy = bbox.score(x_test_adv, y_test)
    else:
        print("Error: unknown classifier.")
        sys.exit(1)

    print("Test adversarial accuracy = {}".format(accuracy))

    log_file = "log/{}.log".format(model_name)
    with open(log_file, "a") as f:
        f.write("Test adversarial accuracy = {}\n".format(accuracy))


if __name__ == '__main__':
    app.run()
