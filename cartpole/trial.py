import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sherpa
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

client = sherpa.Client()
trial = client.get_trial()

artifacts_dir = tempfile.mkdtemp()
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

# Gamma: 1-10^(x_gamma), x in -1 to -4
# n_opt_epochs = x_n_opt_epochs x 7 + 1
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



t0 = time.time()
print("="*100)
print(f"Trial params {trial.id}" + "".join([f"\n{k}={v}" for k,v in trial.parameters.items()]))
rl_params = sherpa_params_to_rl_params(trial.parameters)
print("-"*100)
print(f"RL params {trial.id}" + "".join([f"\n{k}={v}" for k,v in rl_params.items()]))
results = evaluate(params=rl_params)
objective = results['mean_reward']
print("This trial took ", time.time()-t0, "seconds to train...objective value=", objective)

client.send_metrics(trial=trial, iteration=1, objective=objective)
shutil.rmtree(artifacts_dir)


