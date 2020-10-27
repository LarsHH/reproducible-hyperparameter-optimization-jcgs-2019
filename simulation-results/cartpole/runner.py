import os
import argparse
import sherpa
import datetime
import time
from sherpa.schedulers import LocalScheduler

def run_example(FLAGS):
    """
    Run parallel Sherpa optimization over a set of discrete hp combinations.
    """
    parameters = [sherpa.Continuous('log10_learning_rate', [-5, 0]),
                  sherpa.Continuous('log2_batch_size', [5, 8]),
                  sherpa.Continuous('log2_n_steps', [4, 11]),
                  sherpa.Continuous('x_n_opt_epochs', [0, 7]),
                  sherpa.Continuous('log10_entcoeff', [-8, -1]),
                  sherpa.Continuous('x_gamma', [-4, -1]),
                  sherpa.Continuous('cliprange', [0.1, 0.4]),
                  sherpa.Continuous('lam', [0.8, 1.0])]
    
    alg = sherpa.algorithms.RandomSearch(max_num_trials=300)
    alg = sherpa.algorithms.Repeat(alg, 25, agg=True)

    # Run on local machine.
    sched = LocalScheduler()

    rval = sherpa.optimize(parameters=parameters,
                           algorithm=alg,
                           lower_is_better=False,
                           filename='trial.py',
                           scheduler=sched,
                           verbose=0,
                           max_concurrent=FLAGS.max_concurrent,
                           disable_dashboard=True,
                           output_dir='./output_{}'.format(time.strftime("%Y-%m-%d--%H-%M-%S")))
    print('Best results:')
    print(rval)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_concurrent',
                        help='Number of concurrent processes',
                        type=int, default=1)
    FLAGS = parser.parse_args()
    run_example(FLAGS)  # Sherpa optimization.
