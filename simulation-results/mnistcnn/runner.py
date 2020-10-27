import sherpa
import sherpa.schedulers
import argparse
import time


parser = argparse.ArgumentParser()
parser.add_argument('--local', help='Run locally', action='store_true',
                    default=True)
parser.add_argument('--max_concurrent',
                    help='Number of concurrent processes',
                    type=int, default=1)
parser.add_argument('--name',
                    help='A name for this run.',
                    type=str, default='')
parser.add_argument('--gpus',
                    help='Available gpus separated by comma.',
                    type=str, default='')
parser.add_argument('-P',
                    help="Specifies the project to which this  job  is  assigned.",
                    default='arcus_gpu.p')
parser.add_argument('-q',
                    help='Defines a list of cluster queues or queue instances which may be used to execute this job.',
                    default='arcus.q')
parser.add_argument('-l', help='the given resource list.',
                    default="hostname=\'(arcus-1|arcus-2|arcus-3|arcus-4|arcus-5|arcus-6|arcus-7|arcus-8|arcus-9)\'")
parser.add_argument('--env', help='Your environment path.',
                    default='/home/lhertel/profiles/python3env.profile',
                    type=str)
FLAGS = parser.parse_args()


# Define Hyperparameter ranges
parameters = [sherpa.Continuous(name='lr', range=[0.001, 0.1], scale='log'),
              sherpa.Continuous(name='dropout', range=[0.0001, 0.7]),
              sherpa.Continuous(name='top_dropout', range=[0.0001, 0.7]),
              sherpa.Continuous(name='lr_decay', range=[1e-4, 1e-9], scale='log')]

# parameters = [sherpa.Choice(name='lr', range=[0.09]),
#               sherpa.Choice(name='dropout', range=[0.2]),
#               sherpa.Choice(name='lr_decay', range=[1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9])]

algorithm = sherpa.algorithms.RandomSearch(max_num_trials=100, repeat=25)        
# algorithm = sherpa.algorithms.GridSearch()

# The scheduler
# if not FLAGS.local:
#     env = FLAGS.env
#     opt = '-N MNIST-H1 -P {} -q {} -l {} -l gpu=1'.format(FLAGS.P, FLAGS.q, FLAGS.l)
#     scheduler = sherpa.schedulers.SGEScheduler(environment=env, submit_options=opt)
# else:
resources = [int(x) for x in FLAGS.gpus.split(',')] * 3
#     resources = [0,0,0,1,1,1,2,2,2,3,3,3]
scheduler = sherpa.schedulers.LocalScheduler(resources=resources)

# Running it all
sherpa.optimize(algorithm=algorithm,
                scheduler=scheduler,
                parameters=parameters,
                lower_is_better=True,
                filename="mnist_cnn.py",
                max_concurrent=FLAGS.max_concurrent,
                output_dir='./output_mnistcnn_unshuffled_{}_{}_gpu-{}'.format(time.strftime("%Y-%m-%d--%H-%M-%S"), FLAGS.name, FLAGS.gpus.replace(',', '-')))