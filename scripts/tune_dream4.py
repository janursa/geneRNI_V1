# precision-accuracy
import os
import pathlib
import sys

import numpy as np
import pandas as pd

dir_main = os.path.join(pathlib.Path(__file__).parent.resolve(), '..')
sys.path.insert(0, dir_main)

from geneRNI import tools
from geneRNI import search_param

pd.options.mode.chained_assignment = None


def dream4_single(specs, estimator_t, size, network):
    print(f'Running dream4 for network size {size}. network {network}, and estimator {estimator_t}')
    out_data = tools.Benchmark.process_data_dream4(size=size, network=network, estimator_t=estimator_t)
    out_defaults = tools.Settings.default(estimator_t=estimator_t)
    best_scores, best_params, best_ests, sampled_permts = search_param.rand_search(
        Xs=out_data.Xs_train,
        ys=out_data.ys_train,
        param=out_defaults.param,
        param_grid=out_defaults.param_grid,
        **specs
    )
    print(f'{estimator_t} {size} {network} -> best score, mean: {np.mean(best_scores)} std: {np.std(best_scores)}')

    # Create folder
    folder = os.path.join(dir_main, f'results/dream4/{estimator_t}')
    if not os.path.isdir(folder):
        os.makedirs(folder)

    results_dir = f'{folder}/param_search_dream_{size}_{network}.txt'

    with open(results_dir, 'w') as f:
        print({'best_scores': best_scores, 'best_params': best_params}, file=f)


def dream4_size10(specs, estimator_t):
    size = 10
    networks = [1, 2, 3, 4, 5]
    for network in networks:
        dream4_single(specs, estimator_t, size, network)
    print(f'Completed dream4 size {size}, estimator {estimator_t}')


def dream4_size100(specs, estimator_t):
    size = 100
    networks = [1, 2, 3, 4, 5]
    for network in networks:
        dream4_single(specs, estimator_t, size, network)
    print(f'Completed dream4 size {size}, estimator_t {estimator_t}')


if __name__ == '__main__':
    specs = dict(
        n_jobs=int(sys.argv[1]),
        cv=4,
        output_dir=os.path.join(dir_main, 'results')
    )
    estimator_t = 'ridge'  # 'HGB'
    dream4_size10(specs, estimator_t)
    # dream4_size100(specs, estimator_t)
