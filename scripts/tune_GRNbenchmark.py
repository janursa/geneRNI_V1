# precision-accuracy
import os
import sys  
import pathlib
import pandas as pd
import numpy as np

dir_main = os.path.join(pathlib.Path(__file__).parent.resolve(),'..')
sys.path.insert(0, dir_main)
from geneRNI import geneRNI as ni
from geneRNI import tools
from geneRNI import search_param
pd.options.mode.chained_assignment = None


def GRNbenchmark_single(specs, estimator_t, method, noise_level, network):
    print(f'Running single GRNbenchmark: {estimator_t} {method} {noise_level} {network}')
    out_data = tools.Benchmark.process_data_grn_benchmark(method, noise_level, network, estimator_t=estimator_t)
    out_defaults = tools.Settings.default(estimator_t=estimator_t)
    best_scores, best_params, best_ests, sampled_permts = search_param.rand_search(
        out_data, param=out_defaults.param, param_grid=out_defaults.param_grid, **specs)
    results_dir = os.path.join(dir_main, f'results/{estimator_t}/GRNbenchmark_{method}_{noise_level}_{network}.txt')
    print(f'{method}_{noise_level}_{network} -> best score, mean: {np.mean(best_scores)} std: {np.std(best_scores)}')

    with open(results_dir, 'w') as f:
        print({'best_scores': best_scores, 'best_params': best_params}, file=f)
    print(f'Completed single GRNbenchmark: {estimator_t} {method} {noise_level} {network}')


def GRNbenchmark_noise(specs, estimator_t, method, noise_level):
    networks = ['Network1', 'Network2', 'Network3', 'Network4', 'Network5']
    for network in networks:
        GRNbenchmark_single(specs, estimator_t, method, noise_level, network)


def GRNbenchmark_GeneNetWeaver(specs, estimator_t):
    method = 'GeneNetWeaver'
    noise_levels = ['HighNoise', 'LowNoise', 'MediumNoise']
    for noise_level in noise_levels:
        GRNbenchmark_noise(specs, estimator_t, method, noise_level)


def GRNbenchmark_GeneSPIDER(specs, estimator_t):
    method = 'GeneSPIDER'
    noise_levels = ['HighNoise', 'LowNoise', 'MediumNoise']
    for noise_level in noise_levels:
        GRNbenchmark_noise(specs, estimator_t, method, noise_level)


if __name__ == '__main__':
    specs = dict(
        n_jobs=int(sys.argv[1]),
        cv=4,
        output_dir=os.path.join(dir_main, 'results')
        )
    estimator_t = 'RF'  #'HGB'
    GRNbenchmark_GeneNetWeaver(specs, estimator_t)
    GRNbenchmark_GeneSPIDER(specs, estimator_t)
        