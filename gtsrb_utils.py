# The German Traffic Sign Recognition Benchmark
#
# sample code for reading the traffic sign images and the
# corresponding labels
#
# example:
#
# trainImages, trainLabels = readTrafficSigns('GTSRB/Training')
# print len(trainLabels), len(trainImages)
# plt.imshow(trainImages[42])
# plt.show()
#
# have fun, Christian

import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from skimage import transform

DATANAME = "data.pickle"

def dump_to_pickle(rootpath, X, y):
    print("Dump to pickle.")
    with open(os.path.join(rootpath, DATANAME), "wb") as f:
        pickle.dump((X, y), f)

def load_from_pickle(rootpath):
    data = os.path.join(rootpath, DATANAME)
    if os.path.isfile(data):
        print("Load from pickle.")
        with open(data, "rb") as f:
            X, y = pickle.load(f)
            return X, y
    print("Load from raw data.")
    return None, None


def read_training_data(rootpath):
    X, y = load_from_pickle(rootpath)
    if not X is None and not y is None:
        return X, y

    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    for c in range(0,43):
        prefix = os.path.join(rootpath, format(c, '05d'))
        gtFile = open(os.path.join(prefix, 'GT-{:05d}.csv'.format(c))) # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            img = plt.imread(os.path.join(prefix, row[0])) # the 1th column is the filename
            min_side = min(img.shape[:-1])
            center = img.shape[0] // 2, img.shape[1] // 2
            img = img[center[0] - min_side // 2:center[0] + min_side // 2,
                      center[1] - min_side // 2:center[1] + min_side // 2,
                      :]
            img = transform.resize(img, (32, 32), mode='constant')
            images.append(img)
            labels.append(int(row[7])) # the 8th column is the label
        gtFile.close()
    X = np.array(images)
    y = np.eye(43)[labels]
    print('X_train shape:', X.shape)
    print('y_train shape:', y.shape)
    dump_to_pickle(rootpath, X, y)
    return X, y


def read_testing_data(rootpath):
    X, y = load_from_pickle(rootpath)
    if not X is None and not y is None:
        return X, y

    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    with open(rootpath + '/' + "GT-final_test.csv") as gtFile:
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            img = plt.imread(rootpath + '/' + row[0]) # the 1th column is the filename
            min_side = min(img.shape[:-1])
            center = img.shape[0] // 2, img.shape[1] // 2
            img = img[center[0] - min_side // 2:center[0] + min_side // 2,
                      center[1] - min_side // 2:center[1] + min_side // 2,
                      :]
            img = transform.resize(img, (32, 32), mode='constant')
            images.append(img)
            labels.append(int(row[7])) # the 8th column is the label
    X = np.array(images)
    y = np.eye(43)[labels]
    print('X_test shape:', X.shape)
    print('y_test shape:', y.shape)
    dump_to_pickle(rootpath, X, y)
    return X, y
