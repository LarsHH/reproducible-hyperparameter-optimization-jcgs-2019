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
from sklearn.datasets import load_boston
import random
import tempfile
import shutil


parser = argparse.ArgumentParser()
parser.add_argument("--algorithm", help="The search algorithm", type=str)
parser.add_argument("--max_num_trials", help="budget for search", type=int, default=160)
parser.add_argument("--num_final_evals", help="number of evaluations for best trial", type=int, default=25)
args, unknown = parser.parse_known_args()

parameters = [sherpa.Continuous('lr_scaler', [0.5, 1.5]),
              sherpa.Discrete('n_estimators', [100, 400]),
              sherpa.Discrete('max_depth', [2, 10]),
              sherpa.Continuous('subsample', [0.5, 1.])]

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
    algorithm = SequentialTesting(algorithm, K=25)
elif args.algorithm == 'gs_sampling':
    algorithm = sherpa.algorithms.RandomSearch(max_num_trials=args.max_num_trials)
    algorithm = SequentialTesting(algorithm, K=25, sample_best=True)
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

# Load dataset
data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=1234, test_size=0.2)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=5678, test_size=0.2)


def evaluate(lr_scaler,
             n_estimators,
             max_depth,
             subsample
             ):
    """
    Function to evaluate
    """
    clf = GradientBoostingRegressor(learning_rate=0.1*lr_scaler,
                                    n_estimators=int(n_estimators),
                                    max_depth=int(max_depth),
                                    subsample=subsample)
    clf.fit(X_train, y_train)

    predicted_valid = clf.predict(X_valid)
    expected_valid = y_valid
    mse_valid = np.mean((predicted_valid - expected_valid) ** 2)

    predicted_test = clf.predict(X_test)
    expected_test = y_test
    mse_test = np.mean((predicted_test - expected_test) ** 2)
    return {'mse': mse_valid, 'mse_test': mse_test}


def eval_and_save(params, name, evals_used):
    print("Evaluating best trial...")
    results = []
    for _ in range(args.num_final_evals):
        this_result = evaluate(**params)
        results.append(this_result)
    # Save best results
    df = pd.DataFrame(results)
    df['evals'] = evals_used
    df.to_csv(f"{results_path}_{name}.csv", index=False)


eval_steps = [(772, '772evals')]
training_steps_used = 0
# Sherpa Loop
for trial in study:
    t0 = time.time()

    print("-"*100)
    lr_scaler = trial.parameters['lr_scaler']
    n_estimators = trial.parameters['n_estimators']
    max_depth = trial.parameters['max_depth']
    subsample = trial.parameters['subsample']
    
    
    # Set the number of steps
    training_steps = 1
    
    print(f"Trial {trial.id}\lr_scaler={lr_scaler}\n n_estimators={n_estimators}\n max_depth={max_depth}\n subsample={subsample}")
    
    results = evaluate(lr_scaler=lr_scaler,
                       n_estimators=n_estimators,
                       max_depth=max_depth,
                       subsample=subsample)
    training_steps_used += training_steps

    
    mean_reward = results['mse']
    objective = mean_reward
    print("This trial took ", time.time()-t0, "seconds to train...objective value=", objective)
    study.add_observation(trial, objective=objective)
    study.finalize(trial)
    
    if args.algorithm in ['ei3', 'ei5']:
        best_pred = algorithm.algorithm.get_best_pred(results=study.results,
                                                      parameters=study.parameters,
                                                      lower_is_better=True)
    elif args.algorithm not in ['rs', 'rs3', 'asha', 'rs5', 'gs', 'gs_sampling']:
        # all bo without repetitions
        best_pred = algorithm.get_best_pred(results=study.results,
                                            parameters=study.parameters,
                                            lower_is_better=True)
    else:
        best_pred = {k:v for k, v in study.get_best_result().items() 
                     if k in ['lr_scaler',
                              'n_estimators',
                              'max_depth',
                              'subsample']}
        
    best_predicted.append(dict(**{'Trial-ID': trial.id}, **best_pred))
    
    if eval_steps and training_steps_used >= eval_steps[0][0]:
        eval_and_save(best_pred, name=eval_steps[0][1])
        eval_steps.pop(0)
        
    
    
eval_and_save(best_pred, name='final', evals_used=trial.id)

shutil.rmtree(artifacts_dir)


