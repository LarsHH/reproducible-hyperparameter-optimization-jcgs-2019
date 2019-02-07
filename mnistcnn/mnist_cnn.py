'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
from __future__ import print_function
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
CONFIG = tf.ConfigProto(device_count = {'GPU': 1}, log_device_placement=False, allow_soft_placement=False)
CONFIG.gpu_options.allow_growth = True # Prevents tf from grabbing all gpu memory.
sess = tf.Session(config=CONFIG)
from keras import backend as K
K.set_session(sess)


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from sklearn.utils import shuffle

################## Sherpa trial ##################
import sherpa
client = sherpa.Client()
trial = client.get_trial()  # contains ID and parameters
##################################################

batch_size = 128
num_classes = 10
epochs = 15

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# def generator(X, y):
#     nbatches = X.shape[0]//batch_size
#     while True:
#         X, y = shuffle(X, y)
#         for i in range(nbatches-1):
#             yield X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size]

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(trial.parameters['dropout']))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(trial.parameters['top_dropout']))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=trial.parameters['lr'], momentum=0.7,
                                             decay=trial.parameters['lr_decay']),
              metrics=['accuracy'])


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          callbacks=[client.keras_send_metrics(trial, objective_name='val_loss', context_names=['val_acc', 'loss', 'acc'])],
          validation_data=(x_test, y_test))
# model.fit_generator(generator(x_train, y_train),
#                     steps_per_epoch=x_train.shape[0]//batch_size//15,
#                     validation_data=generator(x_test, y_test),
#                     validation_steps=x_test.shape[0]//batch_size//15,
#                     callbacks=[client.keras_send_metrics(trial, objective_name='val_loss', context_names=['val_acc'])],
#                     epochs=epochs)
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
