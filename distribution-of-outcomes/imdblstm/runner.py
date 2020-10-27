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
parser.add_argument("--max_num_trials", help="budget for search", type=int, default=170)
parser.add_argument("--gpu", help="GPU", type=int)
parser.add_argument("--num_final_evals", help="number of evaluations for best trial", type=int, default=25)
args, unknown = parser.parse_known_args()
GPU = int(args.gpu)

batch_size = 128
hidden_dim = 128
max_features = 20000
maxlen = 80
epochs = 15

parameters = []
parameters += [sherpa.Continuous('dropout_embedding', [0.0001, 0.5])]
parameters += [sherpa.Continuous('dropout_lstm', [0.0001, 0.5])]
parameters += [sherpa.Continuous('embedding_regularizer', [1e-12, 1e-6], 'log')]
parameters += [sherpa.Continuous('kernel_regularizer', [1e-8, 1e-0], 'log')]
parameters += [sherpa.Continuous('recurrent_regularizer', [1e-8, 1e-0], 'log')]
parameters += [sherpa.Continuous('lr', [5e-4, 5e-3], scale='log')]
parameters += [sherpa.Continuous('decay', [1e-5, 1e-10], scale='log')]
parameters += [sherpa.Continuous('rho', [0.5, 0.99])]
num_initial_data_points = 2*len(parameters)

artifacts_dir = tempfile.mkdtemp()
if args.algorithm == 'ei':
    algorithm = bayesian_optimization.GPyOpt(max_num_trials=args.max_num_trials,
                                             num_initial_data_points=num_initial_data_points,
                                             acquisition_type='EI',
                                             max_concurrent=1)
elif args.algorithm == 'ei3':
    algorithm = bayesian_optimization.GPyOpt(max_num_trials=args.max_num_trials//3,
                                             num_initial_data_points=num_initial_data_points,
                                             acquisition_type='EI',
                                             max_concurrent=1)
    algorithm = sherpa.algorithms.Repeat(algorithm, 3, agg=True)
elif args.algorithm == 'ei5':
    algorithm = bayesian_optimization.GPyOpt(max_num_trials=args.max_num_trials//5,
                                             num_initial_data_points=num_initial_data_points,
                                             acquisition_type='EI',
                                             max_concurrent=1)
    algorithm = sherpa.algorithms.Repeat(algorithm, 5, agg=True)
elif args.algorithm == 'lcb':
    algorithm = bayesian_optimization.GPyOpt(max_num_trials=args.max_num_trials,
                                             num_initial_data_points=num_initial_data_points,
                                             acquisition_type='LCB',
                                             max_concurrent=1)
elif args.algorithm == 'nei':
    algorithm = BoTorch(max_num_trials=args.max_num_trials,
                        num_initial_data_points=num_initial_data_points)
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

from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, Activation, Dropout
from keras.layers import CuDNNLSTM
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l2
from keras.datasets import imdb
from keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn.utils import shuffle



def evaluate(params):
    """
    Function to evaluate
    """
    dropout_embedding = params.get('dropout_embedding', 0.2)
    dropout_lstm = params.get('dropout_lstm', 0.2)
    embeddings_regularizer = params.get('embeddings_regularizer', 1e-6)
    kernel_regularizer = params.get('kernel_regularizer', 1e-0)
    recurrent_regularizer = params.get('recurrent_regularizer', 1e-0)
    lr = params.get('lr', 1e-3)
    rho = params.get('rho', 0.9)
    decay = params.get('decay', 0.)
    
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, random_state=5678, test_size=0.2)
    
    model = Sequential()
    model.add(Embedding(max_features, hidden_dim, embeddings_regularizer=l2(embeddings_regularizer)))
    model.add(Dropout(dropout_embedding))
    model.add(CuDNNLSTM(hidden_dim,
                        kernel_regularizer=l2(kernel_regularizer),
                        recurrent_regularizer=l2(recurrent_regularizer)))
    model.add(Dropout(dropout_lstm))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    optimizer = RMSprop(lr=lr, rho=rho, decay=decay)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model_path = os.path.join(artifacts_dir, str(GPU) + ".hdf5")
    checkpointer = ModelCheckpoint(filepath=model_path, verbose=0, save_best_only=True)

    print('Train...')
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0,
                        validation_data=(x_valid, y_valid),
                        callbacks=[checkpointer])
    best_val_acc = max(history.history['val_acc'])
    model = load_model(model_path)
    test_loss, test_acc = model.evaluate(x_test, y_test,
                                batch_size=batch_size, verbose=0)

    os.remove(model_path)
    K.clear_session()
    return {'loss': 1.-best_val_acc, 'test_loss': 1.-test_acc}


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


