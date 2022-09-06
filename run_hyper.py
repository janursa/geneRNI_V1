# precision-accuracy
import pathlib
from geneRNI import geneRNI as ni
from geneRNI import tools
from geneRNI import search_param
import os
import pandas as pd
import numpy as np
import sys
pd.options.mode.chained_assignment = None

import importlib
importlib.reload(ni)
importlib.reload(tools)
importlib.reload(search_param)

# these are fixed 
dir_main = pathlib.Path(__file__).parent.resolve()

if __name__ == '__main__':
    specs = dict(
        n_jobs = int(sys.argv[1]),
        cv = 4,
        output_dir = os.path.join(dir_main,'results')       
        )
    estimator_t = 'RF' #'HGB'
    study = 'GRNbenchmark' #'GRNbenchmark'

    if study == 'dreams': # dream as target study
        size, network = 10, 1 # [10,100] [1-5]
        out_data = tools.Benchmark.process_data_dreams(size=size, network=network, estimator_t=estimator_t)
        results_dir = f'results/param_search_dream_{size}_{network}.txt'
    elif study == 'GRNbenchmark':
        method, noise_level, network = 'GeneNetWeaver', 'HighNoise', 'Network1'
        out_data = tools.Benchmark.process_data_GRNbenchmark(method, noise_level, network, estimator_t=estimator_t)
        results_dir = f'results/param_search_GRNbenchmark_{method}_{noise_level}_{network}.txt'
    else:
        raise ValueError('Define')
    out_defaults = tools.Settings.default(estimator_t=estimator_t)
    best_scores, best_params, best_ests, sampled_permts = search_param.rand_search(Xs=out_data.Xs_train, ys=out_data.ys_train, 
                                                                                   param=out_defaults.param, param_grid=out_defaults.param_grid, 
                                                                                       **specs)
    print(f'param search: best score, mean: {np.mean(best_scores)} std: {np.std(best_scores)}')

    with open(results_dir, 'w') as f:
            print({'best_scores':best_scores, 'best_params':best_params}, file=f)