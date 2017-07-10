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
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.utils.np_utils import to_categorical

import tensorflow as tf
from tensorflow.python.platform import app, flags

from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks_tf import jacobian_augmentation, jacobian_graph
from cleverhans.utils import cnn_model
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import batch_eval, model_eval, model_train

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Argparse.
def is_not_nn():
    return args.alg != "cnn"


parser = argparse.ArgumentParser(
        description= "Black-box attack against CNN / \
                                               Adaptive Boosting / \
                                               Gradient Boosting / \
                                               Random Forest / \
                                               Support Vector Machine.")
subparsers = parser.add_subparsers(dest='alg', help="choose black-box type")
subparsers.required = True
cnn_parser = subparsers.add_parser('cnn', help="use cnn")
ada_parser = subparsers.add_parser('adaboost', help="use adaptive boosting")
grad_parser = subparsers.add_parser('gradboost', help="use gradient boosting")
rand_parser = subparsers.add_parser('randforest', help="use random forest")
svm_parser = subparsers.add_parser('svm', help="use svm")
ada_parser.add_argument("n_est", type=int,
        help="the number of estimators")
ada_subparsers = ada_parser.add_subparsers(dest='clsfr',
        help="choose classifier for adaboost")
ada_subparsers.required = True
dt_ada_parser = ada_subparsers.add_parser('dt', help="use decision tree")
dt_ada_parser.add_argument("max_dep", type=int,
        help="the max depth of decision tree")
svm_ada_parser = ada_subparsers.add_parser('svm', help="use svm")
svm_ada_parser.add_argument("--kernel", default='rbf',
        choices=['linear', 'rbf', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        help="the kernel type")
svm_ada_parser.add_argument("C", type=float,
        help="the penalty parameter C of the error term")
grad_parser.add_argument("max_dep", type=int,
        help="the max depth of decision tree")
grad_parser.add_argument("n_est", type=int,
        help="the number of estimators")
rand_parser.add_argument("max_dep", type=int,
        help="the max depth of decision tree")
rand_parser.add_argument("n_est", type=int,
        help="the number of estimators")
svm_parser.add_argument("--kernel", default='rbf',
        choices=['linear', 'rbf', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        help="the kernel type")
svm_parser.add_argument("C", type=float,
        help="the penalty parameter C of the error term")
parser.add_argument("--ae", "--autoencoder", type=str,
        help="use the denoising autoencoder in the beginning")
args = parser.parse_args()

# Generate model name.
if args.alg == "adaboost":
    model_name = "adaboost_"
    if args.clsfr == "dt":
        model_name += "dep{}_est{}".format(args.max_dep, args.n_est)
    elif args.clsfr == "svm":
        model_name += "svm_{}_C{}_est{}".format(args.kernel, args.C, args.n_est)
    else:
        print("Error: unknown classifier.")
        sys.exit(1)
elif args.alg == "cnn":
    model_name = "cnn"
elif args.alg in ["gradboost", "randforest"]:
    model_name = "{}_dep{}_est{}".format(args.alg, args.max_dep, args.n_est)
elif args.alg == "svm":
    model_name = "svm_{}_C{}".format(args.kernel, args.C)
else:
    print("Error: unknown algorithm.")
    sys.exit(1)
if args.ae:
    model_name += "_ae"
    import autoencoder

print(model_name)

# Flags.
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
                init_all=False, verbose=False, args=train_params)
    # """
    if args.ae:
        print("Denoising...")
        num_data = X_test.shape[0]
        autoencoder.visualize(sess, X_test.reshape(num_data, -1), "bbox")
        filtered_data = autoencoder.run(sess, X_test.reshape(num_data, -1))
        X_test = filtered_data.reshape(num_data, 28, 28, 1)
    # """

    # Print out the accuracy on legitimate data
    eval_params = {'batch_size': FLAGS.batch_size}
    accuracy = model_eval(sess, x, y, predictions, X_test, Y_test,
                          args=eval_params)
    print("Test accuracy = {}".format(accuracy))

    return model, predictions


def prep_boost_bbox(X_train, Y_train, X_test, Y_test):
    if args.alg == "adaboost":
        if args.clsfr == "dt":
            alg = "SAMME.R"
            classifier = DecisionTreeClassifier(max_depth=args.max_dep)
        elif args.clsfr == "svm":
            alg = "SAMME"
            classifier = SVC(C=args.C, kernel=args.kernel, gamma=0.000005)
        model = AdaBoostClassifier(
                algorithm=alg,
                base_estimator=classifier,
                learning_rate=1.0,
                n_estimators=args.n_est)
    elif args.alg == "gradboost":
        model = GradientBoostingClassifier(loss='deviance', # {‘deviance’, ‘exponential’}
                                           max_depth=args.max_dep,
                                           n_estimators=args.n_est,
                                           learning_rate=1.0)
    elif args.alg == "randforest":
        model = RandomForestClassifier(max_depth=args.max_dep,
                                       n_estimators=args.n_est,
                                       max_features=None,
                                       n_jobs=-1)
    elif args.alg == "svm":
        model = SVC(C=args.C, kernel=args.kernel, gamma=0.000005)

    x_train = X_train.reshape(X_train.shape[0], -1)
    y_train = np.argmax(Y_train, axis=1)
    x_test = X_test.reshape(X_test.shape[0], -1)
    y_test = np.argmax(Y_test, axis=1)
    print("x_train: {}, y_train: {}, x_test: {}, y_test: {}"\
            .format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

    model.fit(x_train, y_train)

    # Dump model and log.
    model_file = "model/{}.model".format(model_name)
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    log_file = "log/{}.log".format(model_name)
    '''
    if getattr(model, "staged_predict", None):
        with open(log_file, "w") as f:
            for predict in model.staged_predict(x_test):
                f.write("{}\n".format(accuracy_score(predict, y_test)))
    '''

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
                    init_all=False, verbose=False, args=train_params)

        # If we are not at last substitute training iteration, augment dataset
        if rho < FLAGS.data_aug - 1:
            # Perform the Jacobian augmentation
            X_sub = jacobian_augmentation(sess, x, X_sub, Y_sub, grads,
                                          FLAGS.lmbda)

            # Label the newly generated synthetic points using the black-box
            Y_sub = np.hstack([Y_sub, Y_sub])
            X_sub_prev = X_sub[int(len(X_sub)/2):]
            # First feed forward a denoising autoencoder.
            if args.ae:
                print("Denoising...")
                num_data = X_sub_prev.shape[0]
                autoencoder.visualize(sess, X_sub_prev.reshape(num_data, -1), "sub{}".format(rho))
                filtered_data = autoencoder.run(sess, X_sub_prev.reshape(num_data, -1))
                X_sub_prev = filtered_data.reshape(num_data, 28, 28, 1)
            if args.alg == "cnn":
                eval_params = {'batch_size': FLAGS.batch_size}
                bbox_val = batch_eval(sess, [x], [bbox_preds], [X_sub_prev],
                                      args=eval_params)[0]
                # Note here that we take the argmax because the adversary
                # only has access to the label (not the probabilities) output
                # by the black-box model
                Y_sub_prev = np.argmax(bbox_val, axis=1)
            elif is_not_nn():
                x_sub_prev = X_sub_prev.reshape(X_sub_prev.shape[0], -1)
                Y_sub_prev = bbox_preds.predict(x_sub_prev)
            Y_sub[int(len(X_sub)/2):] = Y_sub_prev

    return model_sub, preds_sub


def main(argv=None):
    """
    MNIST cleverhans tutorial
    :return:
    """
    keras.layers.core.K.set_learning_phase(0)

    # Perform tutorial setup
    assert setup_tutorial()

    # Create TF session and set as Keras backend session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # sess = tf.Session()
    keras.backend.set_session(sess)

    if args.ae:
        autoencoder.restore(sess, args.ae) # Restore model weights from previously saved model

    # Get MNIST data
    X_train, Y_train, X_test, Y_test = data_mnist()

    # Initialize substitute training set reserved for adversary
    X_sub = X_test[:FLAGS.holdout]
    Y_sub = np.argmax(Y_test[:FLAGS.holdout], axis=1)

    # Shrink training data.
    # X_train = X_train[:10000]
    # Y_train = Y_train[:10000]
    # Redefine test set as remaining samples unavailable to adversaries
    X_test = X_test[FLAGS.holdout:]
    Y_test = Y_test[FLAGS.holdout:]

    # Define input and output TF placeholders
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    print("Preparing the black-box model.")
    if args.alg == "cnn":
        model, bbox = prep_cnn_bbox(sess, x, y, X_train, Y_train, X_test, Y_test)
    elif is_not_nn():
        bbox = prep_boost_bbox(X_train, Y_train, X_test, Y_test)

    print("Training the substitute model.")
    model_sub, preds_sub = train_substitute(sess, x, y, bbox, X_sub, Y_sub)

    # Initialize the Fast Gradient Sign Method (FGSM) attack object.
    fgsm_par = {'eps': 0.3, 'ord': np.inf, 'clip_min': 0., 'clip_max': 1.}
    fgsm = FastGradientMethod(model_sub, sess=sess)

    # Craft adversarial examples using the substitute
    print("Crafting the adversarial examples.")
    eval_params = {'batch_size': FLAGS.batch_size}
    x_adv_sub = fgsm.generate(x, **fgsm_par)
    X_test_adv, = batch_eval(sess, [x], [x_adv_sub], [X_test], args=eval_params)

    # Dump adversarial examples.
    example_file = "example/{}.data".format(model_name)
    with open(example_file, "wb") as f:
        pickle.dump(X_test_adv, f)

    if args.ae:
        print("Denoising...")
        num_data = X_test_adv.shape[0]
        autoencoder.visualize(sess, X_test_adv.reshape(num_data, -1), "adv")
        filtered_data = autoencoder.run(sess, X_test_adv.reshape(num_data, -1))
        X_test_adv = filtered_data.reshape(num_data, 28, 28, 1)
    # Evaluate the accuracy of the "black-box" model on adversarial examples
    if args.alg == "cnn":
        accuracy = model_eval(sess, x, y, bbox, X_test_adv, Y_test,
                              args=eval_params)
    elif is_not_nn():
        x_test_adv = X_test_adv.reshape(X_test_adv.shape[0], -1)
        y_test = np.argmax(Y_test, axis=1)
        accuracy = bbox.score(x_test_adv, y_test)

    print("Test adversarial accuracy = {}".format(accuracy))

    log_file = "log/{}.log".format(model_name)
    with open(log_file, "a") as f:
        if args.ae:
            f.write("{}. Test adversarial accuracy = {}\n".format(args.ae, accuracy))
        else:
            f.write("Test adversarial accuracy = {}\n".format(accuracy))


if __name__ == '__main__':
    app.run()
