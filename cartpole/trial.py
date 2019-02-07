from __future__ import print_function
import os
import argparse
import subprocess
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument('--notsherpa', help='Do not import Sherpa', action='store_true', default=False)
parser.add_argument('--gpu', type=str, default='')
args, unknown = parser.parse_known_args()

import sherpa
client = sherpa.Client(test_mode=args.notsherpa)
trial = client.get_trial()

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import tensorflow as tf
# CONFIG = tf.ConfigProto(device_count = {'GPU': 1}, log_device_placement=False, allow_soft_placement=False)
# CONFIG.gpu_options.allow_growth = True # Prevents tf from grabbing all gpu memory.
# sess = tf.Session(config=CONFIG)

print("Importing OpenAI Baselines...")
from baselines import run
from baselines.common import plot_util as pu

gamma = trial.parameters.get('gamma', 0.5)
entropy_coefficient = trial.parameters.get('entropy_coefficient', 0.5)
cliprange = trial.parameters.get('cliprange', 0.5)
lr = trial.parameters.get('lr', 1e-4)
cliprange = trial.parameters.get('cliprange', 0.5)
nsteps = trial.parameters.get('nsteps', 128)
num_hidden = trial.parameters.get('num_hidden', 128)
num_layers = trial.parameters.get('num_layers', 3)
gpu = os.environ.get("SHERPA_RESOURCE", args.gpu)
outputdir = os.path.join(os.environ.get("SHERPA_OUTPUT_DIR", './output_{}'.format(time.strftime("%Y-%m-%d--%H-%M-S"))), 'baselines_logs')
if trial.id == 1 and not os.path.exists(outputdir):
    os.makedirs(outputdir, exist_ok=True)
logdir = os.path.join(outputdir, "b128-gamma{}-ent_coef{}-{}".format(gamma, entropy_coefficient, trial.id))

env = {'CUDA_VISIBLE_DEVICES': str(gpu),
       'OPENAI_LOG_FORMAT': 'csv',
       'OPENAI_LOGDIR': logdir} 
# env.update(os.environ.copy())
for k, v in env.items():
    os.environ[k] = v

# command = "python -m baselines.run --alg=ppo2 --env=CartPole-v0 --num_timesteps=3e4 --seed={} --nsteps=128 --gamma={} --ent_coef={}".format(trial.id, gamma, entropy_coefficient)

args = {'alg': 'ppo2',
        'env': 'CartPole-v0',
        'num_timesteps': 1e5,
        'seed': trial.id,
        'nsteps': nsteps,
        'gamma': gamma,
        'ent_coef': entropy_coefficient,
        'cliprange': cliprange,
        'num_hidden': num_hidden,
        'num_layers': num_layers,
        'lr': lr}

t0 = time.time()
print("Running OpenAI Baselines...")
model = run.main(["--{}={}".format(k, v) for k, v in args.items()])
print("Total runtime: {}s".format(time.time() - t0))

print("Loading Logs...")
results = pu.load_results(logdir)
r = results[0]

print("Entering Logs into Sherpa...")
print("Number of iterations: {}".format(len(np.cumsum(r.monitor.l))))
obj = float(np.max(pu.smooth(r.monitor.r, radius=10)))
iteration = int(np.argmax(pu.smooth(r.monitor.r, radius=10)))
timestep = int(np.cumsum(r.monitor.l)[iteration])

converged = 0
convergence_iter = -1
convergence_timestep = -1

for i in range(0, len(r.monitor.r)-20):
    if list(r.monitor.r[i:(i+20)]) == [200.]*20:
        converged = 1
        convergence_iter = i
        convergence_timestep = int(np.cumsum(r.monitor.l)[i])
        break

print("Highest reward of {} at timestep {}".format(obj, timestep))
client.send_metrics(trial=trial, iteration=1, objective=obj,
                    context={'timestep': timestep,
                             'convergence': converged,
                             'convergence_iter': convergence_iter,
                             'convergence_timestep': convergence_timestep})


# for i, (timestep, obj) in enumerate(zip(list(np.cumsum(r.monitor.l)),pu.smooth(r.monitor.r, radius=10))):
#     client.send_metrics(trial=trial, iteration=i, objective=obj, context={'timestep': timestep})
#     if i == 100:
#         time.sleep(5)
print("Done!")