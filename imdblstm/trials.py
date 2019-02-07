'''Trains an LSTM model on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
# Notes
- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.
- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
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


maxlen = 80  # cut texts after this number of words (among top max_features most common words)
max_features = trial.parameters.get('max_features', 20000)
batch_size = trial.parameters.get('batch_size', 128)
hidden_dim = trial.parameters.get('hidden_dim', 128)
dropout_embedding = trial.parameters.get('dropout_embedding', 0.2)
dropout_lstm = trial.parameters.get('dropout_lstm', 0.2)
embeddings_regularizer = trial.parameters.get('embeddings_regularizer', 1e-6)
kernel_regularizer = trial.parameters.get('kernel_regularizer', 1e-0)
recurrent_regularizer = trial.parameters.get('recurrent_regularizer', 1e-0)
lr = trial.parameters.get('lr', 1e-3)
rho = trial.parameters.get('rho', 0.9)
decay = trial.parameters.get('decay', 0.)
gpu = os.environ.get("SHERPA_RESOURCE", '')


os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu or args.gpu)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
CONFIG = tf.ConfigProto(device_count = {'GPU': 1}, log_device_placement=False, allow_soft_placement=False)
CONFIG.gpu_options.allow_growth = True # Prevents tf from grabbing all gpu memory.
sess = tf.Session(config=CONFIG)
from keras import backend as K
K.set_session(sess)

from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, Activation, Dropout
from keras.layers import CuDNNLSTM
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l2
from keras.datasets import imdb
from keras.callbacks import ModelCheckpoint

import numpy as np



print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, hidden_dim, embeddings_regularizer=l2(embeddings_regularizer)))
model.add(Dropout(dropout_embedding))
model.add(CuDNNLSTM(hidden_dim,
                    kernel_regularizer=l2(kernel_regularizer),
                    recurrent_regularizer=l2(recurrent_regularizer)))
model.add(Dropout(dropout_lstm))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
# optimizer = Adam(lr=lr, beta_1=0.)
optimizer = RMSprop(lr=lr, rho=rho, decay=decay)
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model_path = os.path.join(os.environ.get("SHERPA_OUTPUT_DIR", '/tmp/'), str(trial.id) + ".hdf5")
checkpointer = ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True)

print('Train...')
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=15,
                    verbose=2,
                    validation_split=0.2,
                    callbacks=[checkpointer,
                               client.keras_send_metrics(trial,
                                                         objective_name='val_acc',
                                                         context_names=['val_loss', 'loss', 'acc'])])
best_val_acc = max(history.history['val_acc'])
model = load_model(model_path)
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
client.send_metrics(trial=trial, iteration=15, objective=best_val_acc, context={'test_acc': acc})

os.remove(model_path)
print('Test score:', score)
print('Test accuracy:', acc)