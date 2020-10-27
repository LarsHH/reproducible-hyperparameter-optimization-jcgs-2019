import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sherpa
from sherpa.algorithms import bayesian_optimization
from sherpa.algorithms.sequential_testing import SequentialTesting
import datetime
import gym
import time
import argparse
import pandas as pd
import numpy as np
import random
import tempfile
import shutil

from stable_baselines.bench import Monitor, load_results
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy
import tensorflow as tf
tf.get_logger().setLevel('INFO')


parser = argparse.ArgumentParser()
parser.add_argument("--algorithm", help="The search algorithm", type=str)
parser.add_argument("--max_num_trials", help="budget for search", type=int, default=190)
parser.add_argument("--num_final_evals", help="number of evaluations for best trial", type=int, default=25)
parser.add_argument("--training_steps", help="number of training steps for PPO", type=int, default=30000)
parser.add_argument("--logregret", help="Optimize -1*log(500-y+0.001)", action='store_true', default=False)
args, unknown = parser.parse_known_args()


# Gamma: 1-10^(x_gamma), x in -1 to -4
# n_opt_epochs = x_n_opt_epochs x 7 + 1
parameters = [sherpa.Continuous('log10_learning_rate', [-5, 0]),
              sherpa.Continuous('log2_batch_size', [5, 8]),
              sherpa.Continuous('log2_n_steps', [4, 11]),
              sherpa.Continuous('x_n_opt_epochs', [0, 7]),
              sherpa.Continuous('log10_entcoeff', [-8, -1]),
              sherpa.Continuous('x_gamma', [-4, -1]),
              sherpa.Continuous('cliprange', [0.1, 0.4]),
              sherpa.Continuous('lam', [0.8, 1.0])]

artifacts_dir = tempfile.mkdtemp()
if args.algorithm == 'ei':
    algorithm = bayesian_optimization.GPyOpt(max_num_trials=args.max_num_trials,
                                             num_initial_data_points=2*len(parameters),
                                             acquisition_type='EI',
                                             max_concurrent=1)
elif args.algorithm == 'ei3':
    algorithm = bayesian_optimization.GPyOpt(max_num_trials=args.max_num_trials//3,
                                             num_initial_data_points=2*len(parameters),
                                             acquisition_type='EI',
                                             max_concurrent=1)
    algorithm = sherpa.algorithms.Repeat(algorithm, 3, agg=True)
elif args.algorithm == 'ei5':
    algorithm = bayesian_optimization.GPyOpt(max_num_trials=args.max_num_trials//5,
                                             num_initial_data_points=2*len(parameters),
                                             acquisition_type='EI',
                                             max_concurrent=1)
    algorithm = sherpa.algorithms.Repeat(algorithm, 5, agg=True)
elif args.algorithm == 'lcb':
    algorithm = bayesian_optimization.GPyOpt(max_num_trials=args.max_num_trials,
                                             num_initial_data_points=2*len(parameters),
                                             acquisition_type='LCB',
                                             max_concurrent=1)
elif args.algorithm == 'rs':
    algorithm = sherpa.algorithms.RandomSearch(max_num_trials=args.max_num_trials)
elif args.algorithm == 'rs3':
    algorithm = sherpa.algorithms.RandomSearch(max_num_trials=args.max_num_trials//3)
    algorithm = sherpa.algorithms.Repeat(algorithm, 3)
elif args.algorithm == 'rs5':
    algorithm = sherpa.algorithms.RandomSearch(max_num_trials=args.max_num_trials//5)
    algorithm = sherpa.algorithms.Repeat(algorithm, 5)
elif args.algorithm == 'asha':
    algorithm = sherpa.algorithms.SuccessiveHalving(r=1, R=9, eta=3, s=0, max_finished_configs=int(args.max_num_trials/2.44))
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
                     lower_is_better=False,
                     disable_dashboard=True)

# Generate unique filename
method_name = f"{args.algorithm}_logregret" if args.logregret else args.algorithm
dateandtime = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
myid = str(int(random.random()*1000000))
file_name = f"{method_name}-{dateandtime}-id-{myid}"
results_path = os.path.join("./results/", file_name)
print(results_path)
log_base_dir = os.path.join(artifacts_dir, "log_dir")
os.makedirs(log_base_dir, exist_ok=True)


def evaluate(params,
             load_from='',
             training_steps=30000,
             save_path=None,
             ):
    """
    Function to evaluate Cartpole PPO
    """
    log_dir = os.path.join(log_base_dir, "this_trial")
    os.makedirs(log_dir)
    env = gym.make('CartPole-v1')
    env = Monitor(env, log_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])
    
    
    if load_from:
        model = PPO2.load(os.path.join(artifacts_dir, load_from), env=env)
    else:
        model = PPO2(MlpPolicy,
                     env,
                     verbose=0,
                     **params)
    model.learn(total_timesteps=training_steps)
    
    df = load_results(log_dir)
    df['mean_reward'] = df.r.rolling(window=100).mean()
    convergence_episode = df.loc[df.mean_reward > 195.0].first_valid_index()
    mean_reward = df.r.mean()
    last_mean_reward = df['mean_reward'].tail(1).values.item()
    
    if save_path:
        model.save(save_path)
        
    shutil.rmtree(log_dir)
    
    return {'mean_reward': mean_reward,
            'convergence_episode': convergence_episode,
            'last_mean_reward': last_mean_reward}

def sherpa_params_to_rl_params(sherpa_params):
    n_steps = int(2**sherpa_params['log2_n_steps'])
    batch_size = int(2**sherpa_params['log2_batch_size'])
    nminibatches = 1 if n_steps < batch_size else int(n_steps / batch_size)
    return {'n_steps': n_steps//nminibatches * nminibatches,
          'nminibatches': nminibatches,
          'gamma': 1.-10**sherpa_params['x_gamma'],
          'learning_rate': 10**sherpa_params['log10_learning_rate'],
          'ent_coef': 10**sherpa_params['log10_entcoeff'],
          'cliprange': sherpa_params['cliprange'],
          'noptepochs': int(sherpa_params['x_n_opt_epochs'] * 7 + 1),
          'lam': sherpa_params['lam']}


def eval_and_save(sherpa_params, name, evals_used):
    print("Evaluating best trial...")
    results = []
    for _ in range(args.num_final_evals):
        rl_params = sherpa_params_to_rl_params(sherpa_params)
        this_result = evaluate(params=rl_params,
                               training_steps=args.training_steps)
        results.append(this_result)
    # Save best results
    df = pd.DataFrame(results)
    df['evals'] = evals_used
    df.to_csv(f"{results_path}_{name}.csv", index=False)



eval_steps = []
training_steps_used = 0
# Sherpa Loop
for trial in study:
    t0 = time.time()


    
    load_from = trial.parameters.get('load_from', '')
    
    # Set the number of steps
    max_training_steps = args.training_steps
    if 'resource' in trial.parameters:
        training_unit = max_training_steps//9
        resource = {1: 1, 3: 2, 9: 7}[trial.parameters['resource']]
        training_steps = trial.parameters['resource'] * training_unit
        save_path = os.path.join(artifacts_dir, str(trial.id))
    else:
        training_steps = max_training_steps
        save_path = None
    print("="*100)
    print(f"Trial params {trial.id}" + "".join([f"\n{k}={v}" for k,v in trial.parameters.items()]))
    rl_params = sherpa_params_to_rl_params(trial.parameters)
    print("-"*100)
    print(f"RL params {trial.id}" + "".join([f"\n{k}={v}" for k,v in rl_params.items()]))
    
    results = evaluate(params=rl_params,
                       load_from=load_from,
                       training_steps=training_steps,
                       save_path=save_path)
    training_steps_used += training_steps

    
    mean_reward = results['mean_reward']
    objective = -1*np.log(500-mean_reward+0.001) if args.logregret else mean_reward
    print("This trial took ", time.time()-t0, "seconds to train...objective value=", objective)
    study.add_observation(trial, objective=objective)
    study.finalize(trial)
    
    # Bayesopt with repetition
    if args.algorithm in ['ei3', 'ei5']:
        best_pred = algorithm.algorithm.get_best_pred(results=study.results,
                                                      parameters=study.parameters,
                                                      lower_is_better=False)
    # Bayesopt methods without repetition
    elif args.algorithm not in ['rs', 'rs3', 'asha', 'rs5', 'gs', 'gs_sampling']:
        best_pred = algorithm.get_best_pred(results=study.results,
                                            parameters=study.parameters,
                                            lower_is_better=False)
    # Model free
    else:
        best_pred = {k:v for k, v in study.get_best_result().items() 
                     if k in trial.parameters.keys()}
        
    best_predicted.append(dict(**{'Trial-ID': trial.id}, **best_pred))
    
    if eval_steps and training_steps_used >= eval_steps[0][0]:
        eval_and_save(best_pred, name=eval_steps[0][1])
        eval_steps.pop(0)
        
    
    
eval_and_save(best_pred, name='final', evals_used=trial.id)

shutil.rmtree(artifacts_dir)


