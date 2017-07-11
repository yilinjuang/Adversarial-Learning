import time
import sys

if len(sys.argv) < 2:
    print("Missing arguments")
    print("Usage: denoise.py <output-basename>")
    sys.exit(1)

import numpy as np
import numpy.matlib

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation, SpatialDropout2D, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU, PReLU, LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.local import LocallyConnected2D
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint

from cleverhans.utils_mnist import data_mnist

from gtsrb_utils import read_training_data, read_testing_data

epoch = 500
dataset = "gtsrb"#"mnist"

print('data loading...')
if dataset == "mnist":
    # Get MNIST data
    X_train, Y_train, X_test, Y_test = data_mnist()
elif dataset == "gtsrb":
    # Get GTSRB data
    # http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset
    X_train, Y_train = read_training_data("/tmp/GTSRB/Final_Training/Images/")  # (39209, 32, 32, 3), (39209, 43)
    X_test, Y_test = read_testing_data("/tmp/GTSRB/Final_Test/Images/")# (12630, 32, 32, 3), (12630, 43)
else:
    print("Error: Invalid dataset.")
    sys.exit(1)
IMG_DIM = X_train.shape[1:]
print("Image dimension: {}".format(IMG_DIM))

Cx_train=X_train[0:8000]
Cy_train=Y_train[0:8000]

Cx_test=X_train[8000:10000]
Cy_test=Y_train[8000:10000]

Nx_train1=np.zeros(np.shape(Cx_train))
Nx_train2=np.zeros(np.shape(Cx_train))
Nx_train3=np.zeros(np.shape(Cx_train))
Nx_train4=np.zeros(np.shape(Cx_train))
for i in range(np.shape(Cx_train)[0]):
    Nx_train1[i]=Cx_train[i]+ np.random.normal (0,    0.1, IMG_DIM)
    Nx_train2[i]=Cx_train[i]+ np.random.normal (0,    0.5, IMG_DIM)
    Nx_train3[i]=Cx_train[i]+ np.random.uniform(-0.1, 0.1, IMG_DIM)
    Nx_train4[i]=Cx_train[i]+ np.random.uniform(-0.5, 0.5, IMG_DIM)

Nx_train=np.concatenate((Nx_train1,Nx_train2,Nx_train3,Nx_train4,Cx_train), axis=0)
Cx_train=np.concatenate((Cx_train,Cx_train,Cx_train,Cx_train,Cx_train), axis=0)

Nx_test1=np.zeros(np.shape(Cx_test))
Nx_test2=np.zeros(np.shape(Cx_test))
Nx_test3=np.zeros(np.shape(Cx_test))
Nx_test4=np.zeros(np.shape(Cx_test))
for i in range(np.shape(Cx_test)[0]):
    Nx_test1[i]=Cx_test[i]+ np.random.normal (0,    0.1, IMG_DIM)
    Nx_test2[i]=Cx_test[i]+ np.random.normal (0,    0.5, IMG_DIM)
    Nx_test3[i]=Cx_test[i]+ np.random.uniform(-0.1, 0.1, IMG_DIM)
    Nx_test4[i]=Cx_test[i]+ np.random.uniform(-0.5, 0.5, IMG_DIM)

Nx_test=np.concatenate((Nx_test1,Nx_test2,Nx_test3,Nx_test4,Cx_test), axis=0)
Cx_test=np.concatenate((Cx_test,Cx_test,Cx_test,Cx_test,Cx_test), axis=0)

print("Nx_train: {}".format(Nx_train.shape))
print("Cx_train: {}".format(Cx_train.shape))
print("Nx_test: {}".format(Nx_test.shape))
print("Cx_test: {}".format(Cx_test.shape))

start_time = time.time()

print('model building...')
model = Sequential()

model.add(Conv2D(32, (5, 5), padding='same', data_format='channels_last', input_shape=IMG_DIM))
model.add(BatchNormalization(axis=-1))
model.add(ELU())
model.add(Dropout(0.07))

model.add(Conv2D(64, (5, 5), padding='same',  data_format='channels_last'))
model.add(BatchNormalization(axis=-1))
model.add(ELU())
model.add(Dropout(0.07))


model.add(Conv2D(64, (3, 3), padding='same',  data_format='channels_last'))
model.add(BatchNormalization(axis=-1))
model.add(ELU())
model.add(Dropout(0.07))

model.add(Conv2D(IMG_DIM[-1], (3, 3),  padding='same', data_format='channels_last'))

with open('{}.json'.format(sys.argv[1]), 'w') as f:    # save the model
    f.write(model.to_json())

sgd = SGD(lr=0.05, decay=5*1e-8, momentum=0.9, nesterov=True)
model.compile(loss='mse', optimizer="Nadam")

print('training...')
checkpointer = ModelCheckpoint(filepath='{}.hdf5'.format(sys.argv[1]), verbose=1, save_best_only=True, mode='min')
model.fit(Nx_train, Cx_train, epochs=epoch, batch_size=500, verbose=1, shuffle=True, validation_data=(Nx_test,Cx_test), callbacks=[checkpointer])

print('testing...')
model.predict(Nx_test, verbose=1, batch_size=500)

end_time = time.time()
print ('The code for this file ran for %.2fm' % ((end_time - start_time) / 60.))
