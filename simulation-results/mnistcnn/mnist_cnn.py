'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
from __future__ import print_function
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--notsherpa', help='Do not import Sherpa', action='store_true', default=False)
parser.add_argument('--gpu', type=str, default='')
args, unknown = parser.parse_known_args()

import sherpa
client = sherpa.Client(test_mode=args.notsherpa)
trial = client.get_trial()

gpu = os.environ.get("SHERPA_RESOURCE", '')

os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu or args.gpu)
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
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.models import load_model
import numpy as np
from sklearn.utils import shuffle

batch_size = 128
num_classes = 10
epochs = 15

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, y_train = shuffle(x_train, y_train)

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

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(trial.parameters.get('dropout', 0.1)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(trial.parameters.get('top_dropout', 0.1)))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=trial.parameters.get('lr',0.01), momentum=0.7,
                                             decay=trial.parameters.get('lr_decay',0.)), metrics=['acc'])

model_path = os.path.join(os.environ.get("SHERPA_OUTPUT_DIR", '/tmp/'), str(trial.id) + ".hdf5")
checkpointer = ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True)

print('Train...')
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_split=0.2,
                    callbacks=[checkpointer])
best_val_loss = min(history.history['val_loss'])
model = load_model(model_path)
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print("Score: ", score, "Acc: ", acc)
client.send_metrics(trial=trial, iteration=epochs, objective=best_val_loss, context={'test_loss': score, 'test_acc': acc})

os.remove(model_path)