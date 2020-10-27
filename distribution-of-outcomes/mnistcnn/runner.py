import os
import sherpa
from sherpa.algorithms import bayesian_optimization
from sherpa.algorithms.sequential_testing import SequentialTesting
import datetime
import gym
import time
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import random
import tempfile
import shutil



parser = argparse.ArgumentParser()
parser.add_argument("--algorithm", help="The search algorithm", type=str)
parser.add_argument("--max_num_trials", help="budget for search", type=int, default=150)
parser.add_argument("--gpu", help="GPU", type=int)
parser.add_argument("--num_final_evals", help="number of evaluations for best trial", type=int, default=25)
args, unknown = parser.parse_known_args()

GPU = int(args.gpu)

parameters = [sherpa.Continuous(name='lr', range=[0.001, 0.1], scale='log'),
              sherpa.Continuous(name='dropout', range=[0.0001, 0.7]),
              sherpa.Continuous(name='top_dropout', range=[0.0001, 0.7]),
              sherpa.Continuous(name='lr_decay', range=[1e-4, 1e-9], scale='log')]

artifacts_dir = tempfile.mkdtemp()
if args.algorithm == 'ei':
    algorithm = bayesian_optimization.GPyOpt(max_num_trials=args.max_num_trials,
                                             num_initial_data_points=5,
                                             acquisition_type='EI',
                                             max_concurrent=1)
elif args.algorithm == 'ei3':
    algorithm = bayesian_optimization.GPyOpt(max_num_trials=args.max_num_trials//3,
                                             num_initial_data_points=5,
                                             acquisition_type='EI',
                                             max_concurrent=1)
    algorithm = sherpa.algorithms.Repeat(algorithm, 3, agg=True)
elif args.algorithm == 'ei5':
    algorithm = bayesian_optimization.GPyOpt(max_num_trials=args.max_num_trials//5,
                                             num_initial_data_points=5,
                                             acquisition_type='EI',
                                             max_concurrent=1)
    algorithm = sherpa.algorithms.Repeat(algorithm, 5, agg=True)
elif args.algorithm == 'lcb':
    algorithm = bayesian_optimization.GPyOpt(max_num_trials=args.max_num_trials,
                                             num_initial_data_points=5,
                                             acquisition_type='LCB',
                                             max_concurrent=1)
elif args.algorithm == 'nei':
    algorithm = BoTorch(max_num_trials=args.max_num_trials,
                        num_initial_data_points=5)
elif args.algorithm == 'rs':
    algorithm = sherpa.algorithms.RandomSearch(max_num_trials=args.max_num_trials)
elif args.algorithm == 'rs3':
    algorithm = sherpa.algorithms.RandomSearch(max_num_trials=args.max_num_trials//3)
    algorithm = sherpa.algorithms.Repeat(algorithm, 3)
elif args.algorithm == 'rs5':
    algorithm = sherpa.algorithms.RandomSearch(max_num_trials=args.max_num_trials//5)
    algorithm = sherpa.algorithms.Repeat(algorithm, 5)
elif args.algorithm == 'gs':
    algorithm = sherpa.algorithms.RandomSearch(max_num_trials=args.max_num_trials)
    algorithm = SequentialTesting(algorithm, K=40)
elif args.algorithm == 'gs_sampling':
    algorithm = sherpa.algorithms.RandomSearch(max_num_trials=args.max_num_trials)
    algorithm = SequentialTesting(algorithm, K=40, sample_best=True)
else:
    assert False, "no algorithm provided"
best_predicted = []


study = sherpa.Study(parameters=parameters,
                     algorithm=algorithm,
                     lower_is_better=True,
                     disable_dashboard=True)

# Generate unique filename
method_name = args.algorithm
dateandtime = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
myid = str(int(random.random()*1000000))
file_name = f"{method_name}-{dateandtime}-id-{myid}"
results_path = os.path.join("./results/", file_name)
print(results_path)
log_base_dir = os.path.join(artifacts_dir, "log_dir")
os.makedirs(log_base_dir, exist_ok=True)



## Keras
print(f"Running from GPU {GPU}")
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)
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
img_rows, img_cols = 28, 28
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
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, random_state=5678, test_size=0.2)


def evaluate(params):
    """
    Function to evaluate
    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(params['dropout']))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(params['top_dropout']))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=params['lr'], momentum=0.7, decay=params['lr_decay']))

    model_path = os.path.join(artifacts_dir, str(GPU) + ".hdf5")
    checkpointer = ModelCheckpoint(filepath=model_path, verbose=0, save_best_only=True)

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0,
                        validation_data=(x_valid, y_valid),
                        callbacks=[checkpointer])
    best_val_loss = min(list(history.history['val_loss']))
    model = load_model(model_path)
    test_loss = model.evaluate(x=x_test, y=y_test,
                               batch_size=batch_size,
                               verbose=0)
    os.remove(model_path)
    return {'loss': best_val_loss, 'test_loss': test_loss}


def eval_and_save(params, name, evals_used):
    print("Evaluating best trial...")
    results = []
    for _ in range(args.num_final_evals):
        this_result = evaluate(params)
        results.append(this_result)
    # Save best results
    df = pd.DataFrame(results)
    df['evals'] = evals_used
    df.to_csv(f"{results_path}_{name}.csv", index=False)


# eval_steps = [(772, '772evals')]
eval_steps = []
training_steps_used = 0
# Sherpa Loop
for trial in study:
    t0 = time.time()

    print("-"*100)
    print(f"Trial {trial.id}\n" + "\n ".join([f"{k}={v}" for k, v in trial.parameters.items()]))
    results = evaluate(trial.parameters)
    training_steps = 1
    training_steps_used += training_steps

    objective = results['loss']
    print("This trial took ", time.time()-t0, "seconds to train...objective value=", objective)
    study.add_observation(trial, objective=objective)
    study.finalize(trial)
    
    if args.algorithm in ['ei3', 'ei5']:
        best_pred = algorithm.algorithm.get_best_pred(results=study.results,
                                                      parameters=study.parameters,
                                                      lower_is_better=True)
    elif args.algorithm not in ['rs', 'rs3', 'asha', 'rs5', 'gs', 'gs_sampling']:
        best_pred = algorithm.get_best_pred(results=study.results,
                                            parameters=study.parameters,
                                            lower_is_better=True)
    else:
        best_pred = {k:v for k, v in study.get_best_result().items() 
                     if k in trial.parameters.keys()}
        
    best_predicted.append(dict(**{'Trial-ID': trial.id}, **best_pred))
    
    if eval_steps and training_steps_used >= eval_steps[0][0]:
        eval_and_save(best_pred, name=eval_steps[0][1])
        eval_steps.pop(0)
        
    
eval_and_save(best_pred, name='final', evals_used=trial.id)

shutil.rmtree(artifacts_dir)


