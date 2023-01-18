# precision-accuracy
import argparse
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

dir_main = os.path.join(pathlib.Path(__file__).parent.resolve(), '..')
sys.path.insert(0, dir_main)

from geneRNI.models import get_estimator_names
from geneRNI.benchmarks import Benchmark
from geneRNI import tools
from geneRNI import search_param

pd.options.mode.chained_assignment = None


def dream5_single(specs, estimator_t, network):
    print(f'Running dream5, network {network}, and estimator {estimator_t}')
    out_data = Benchmark.process_data_dream5(network=network, estimator_t=estimator_t)
    out_defaults = tools.Settings.default(estimator_t=estimator_t)
    best_scores, best_params, best_ests, sampled_permts = search_param.rand_search(
        out_data,
        param=out_defaults.param,
        param_grid=out_defaults.param_grid,
        **specs
    )
    print(f'{estimator_t} {network} -> best score, mean: {np.mean(best_scores)} std: {np.std(best_scores)}')

    # Create folder
    folder = os.path.join(dir_main, f'results/dream5/{estimator_t}')
    if not os.path.isdir(folder):
        os.makedirs(folder)

    results_dir = f'{folder}/param_search_dream_{network}.txt'

    with open(results_dir, 'w') as f:
        print({'best_scores': best_scores, 'best_params': best_params}, file=f)


def dream5(specs, estimator_t):
    networks = [1, 3, 4]
    for network in networks:
        dream5_single(specs, estimator_t, network)
    print(f'Completed dream4, estimator_t {estimator_t}')


if __name__ == '__main__':

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n-jobs', type=int, default=1, help='Number of jobs'
    )
    parser.add_argument(
        '-estimator', type=str, default='ridge',
        choices=get_estimator_names(), help='Gene expression estimator'
    )
    args = parser.parse_args()
    n_jobs = args.n_jobs
    estimator_t = args.estimator

    specs = dict(
        n_jobs=n_jobs,
        cv=4,
        output_dir=os.path.join(dir_main, 'results')
    )
    # dream4_size10(specs, estimator_t)
    t0 = time.time()
    dream5(specs, estimator_t)
    print(f'Total running time: {time.time() - t0}')
