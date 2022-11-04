import os
import sys  
import pathlib
import pandas as pd
import numpy as np
import itertools
import warnings

warnings.warn = lambda x: x

dir_main = os.path.join(pathlib.Path(__file__).parent.resolve(), '..')
sys.path.insert(0, dir_main)

from geneRNI import geneRNI as ni
from geneRNI import tools
from multiprocessing import Pool
import ast
pd.options.mode.chained_assignment = None


def dream4_single(estimator_t, size, network):
    out_defaults = tools.Settings.default(estimator_t=estimator_t)
    out_data = tools.Benchmark.process_data_dream4(size=size, network=network, estimator_t=estimator_t, verbose=False)
    results_dir = os.path.join(dir_main, f'results/dream4/{estimator_t}/param_search_dream_{size}_{network}.txt')
    with open(results_dir,'r') as f:
        out = ast.literal_eval(f.read())
    best_scores, best_params = out['best_scores'], out['best_params']
    # nework inference
    _, _, _, gene_names = tools.Benchmark.f_data_dream4(size, network)
    _, train_scores, links, oob_scores, test_scores = ni.network_inference(Xs=out_data.Xs_train, ys=out_data.ys_train, 
                                                                           gene_names=gene_names,
                                                                           param=out_defaults.param, Xs_test=out_data.Xs_test, 
                                                                           ys_test=out_data.ys_test, param_unique=best_params, verbose=False)
    if estimator_t == 'RF':
        score = oob_scores
    else:
        score = test_scores
    
    # calculate PR
    golden_links = tools.Benchmark.f_golden_dream4(size, network)
    precision, recall, average_precision, average_precision_overall = tools.GOF.calculate_PR(gene_names, links, golden_links, details=True)
    print(f'completed {size} {network}')
    return size, network, best_params, score, average_precision_overall

def map_run(args):
    return dream4_single(**args)

if __name__ == '__main__':
    n_jobs = int(sys.argv[1])
    estimator_t = 'ridge'

    sizes = [10]
    networks = [1,2,3,4,5]
    # create all the cases by combining sizes and networks
    permuts = list(itertools.product(sizes,networks))
    cases = [{'size':size, 'network':network, 'estimator_t': estimator_t} for size, network in permuts]
    # run each
    if n_jobs == 1:
        all_output = list(map(map_run, cases))
    else:
        pool = Pool(n_jobs)
        all_output = pool.map(map_run, cases)
    # extract the output in form of a dict
    scores_stack = {}
    params_stack = {}
    PR_stack = {}
    all_output.sort(key = lambda x: x[0])
    for size in sizes:
        params_stack[size] = {}
        scores_stack[size] = {}
        PR_stack[size] = {}
        output = [x for x in all_output if x[0] == size]
        for network in networks:
            output_network = next(x for x in output if x[1] == network)
            params_stack[size][network] = output_network[2]
            scores_stack[size][network] = output_network[3]
            PR_stack[size][network] = output_network[4]
    with open(os.path.join(dir_main, f'results/dream4/{estimator_t}_best_params.txt'), 'w') as ff:
        print({'best_params': params_stack}, file=ff)
    with open(os.path.join(dir_main, f'results/dream4/{estimator_t}_best_scores.txt'), 'w') as ff:
        print({'best_scores': scores_stack}, file=ff)
    with open(os.path.join(dir_main, f'results/dream4/{estimator_t}_precision_recall_AUC.txt'), 'w') as ff:
        print({'PR': PR_stack}, file=ff)
    

        