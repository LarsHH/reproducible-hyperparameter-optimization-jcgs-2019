import os
import argparse
import sherpa
import datetime
import time
from sherpa.schedulers import LocalScheduler,SGEScheduler


def run_example(FLAGS):
    """
    Run parallel Sherpa optimization over a set of discrete hp combinations.
    """
#     max_features = trial.parameters('max_features', 20000)
#     batch_size = trial.parameters('batch_size', 32)
#     hidden_dim = trial.parameters('hidden_dim', 128)
#     dropout = trial.parameters('dropout', 0.2)
#     recurrent_dropout = trial.parameters('recurrent_dropout', 0.2)
#     optimizer = trial.parameters('optimizer', 'adam')
    
    
    parameters = []
    parameters += [sherpa.Discrete('max_features', [15000, 40000])]
    parameters += [sherpa.Discrete('batch_size', [10, 150], 'log')]
    parameters += [sherpa.Discrete('hidden_dim', [100, 500], 'log')]
    parameters += [sherpa.Continuous('dropout_embedding', [0.0001, 0.5])]
    parameters += [sherpa.Continuous('dropout_lstm', [0.0001, 0.5])]
    parameters += [sherpa.Continuous('embedding_regularizer', [1e-12, 1e-6], 'log')]
    parameters += [sherpa.Continuous('kernel_regularizer', [1e-8, 1e-0], 'log')]
    parameters += [sherpa.Continuous('recurrent_regularizer', [1e-8, 1e-0], 'log')]
    parameters += [sherpa.Continuous('lr', [5e-4, 5e-3], scale='log')]
    parameters += [sherpa.Continuous('decay', [1e-5, 1e-10], scale='log')]
    parameters += [sherpa.Continuous('rho', [0.5, 0.99])]
    
    
    print('Running Random Search')
    alg = sherpa.algorithms.RandomSearch(max_num_trials=200, repeat=25)

    if FLAGS.sge:
        assert FLAGS.env, "For SGE use, you need to set an environment path."
        # Submit to SGE queue.
        env = FLAGS.env  # Script specifying environment variables.
        opt = '-N LSTMSearch -P {} -q {} -l {} -l gpu=1'.format(FLAGS.P, FLAGS.q, FLAGS.l)
        sched = SGEScheduler(environment=env, submit_options=opt)
    else:
        # Run on local machine.
        sched = LocalScheduler(resources=[0,1,2,3])

    rval = sherpa.optimize(parameters=parameters,
                           algorithm=alg,
                           lower_is_better=False,
                           filename='trial_multiple_populations.py',
                           scheduler=sched,
                           verbose=0,
                           max_concurrent=FLAGS.max_concurrent,
                           output_dir='./output_multiple_hypotheses_cudnn_gpu{}_{}'.format(FLAGS.gpu, time.strftime("%Y-%m-%d--%H:%M")))
    print('Best results:')
    print(rval)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sge', help='Run on SGE', action='store_true', default=False)
    parser.add_argument('--max_concurrent',
                        help='Number of concurrent processes',
                        type=int, default=1)
    parser.add_argument('-P',
                        help="Specifies the project to which this  job  is  assigned.",
                        default='arcus_gpu.p')
    parser.add_argument('-q',
                        help='Defines a list of cluster queues or queue instances which may be used to execute this job.',
                        default='arcus.q')
#     parser.add_argument('-l', help='the given resource list.',
#                         default="hostname=\'(arcus-1|arcus-2|arcus-3|arcus-4|arcus-5|arcus-6|arcus-7|arcus-8|arcus-9)\'")
    parser.add_argument('-l', help='the given resource list.',
                        default="hostname=\'arcus-1\'")
    parser.add_argument('--env', help='Your environment path.',
                        default='/home/lhertel/profiles/python3env.profile', type=str)
    parser.add_argument('--gpu', help='Which GPU to run on.',
                    default='0', type=str)
    parser.add_argument('--studyname', help='name for output folder', type=str, default='')
    parser.add_argument('--algorithm', type=str, default='BayesianOptimization')
    FLAGS = parser.parse_args()
    run_example(FLAGS)  # Sherpa optimization.
