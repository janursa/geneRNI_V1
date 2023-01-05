import argparse
import os
import sys  
import pathlib
import pandas as pd
import itertools

dir_main = os.path.join(pathlib.Path(__file__).parent.resolve(), '..')
sys.path.insert(0, dir_main)

from geneRNI.models import get_estimator_names
from geneRNI import geneRNI as ni
from geneRNI import tools
from multiprocessing import Pool
import ast
pd.options.mode.chained_assignment = None


def dream4_single(estimator_t, size, network):
    out_defaults = tools.Settings.default(estimator_t=estimator_t)
    out_data = tools.Benchmark.process_data_dream4(size=size, network=network, estimator_t=estimator_t, verbose=False)

    # Load optimal hyper-parameters
    results_dir = os.path.join(dir_main, f'results/dream4/{estimator_t}/param_search_dream_{size}_{network}.txt')
    with open(results_dir, 'r') as f:
        out = ast.literal_eval(f.read())
    best_scores, best_params = out['best_scores'], out['best_params']

    # Network inference
    _, _, _, gene_names = tools.Benchmark.f_data_dream4(size, network)
    _, train_scores, links, oob_scores, test_scores = ni.network_inference(
        out_data, gene_names=gene_names, param=out_defaults.param, param_unique=best_params, verbose=False
    )
    score = oob_scores if (estimator_t == 'RF') else test_scores
    
    # calculate PR and ROC AUCs
    golden_links = tools.Benchmark.f_golden_dream4(size, network)
    average_precision = tools.GOF.calculate_PR(gene_names, links, golden_links)
    auc_roc = tools.GOF.calculate_auc_roc(gene_names, links, golden_links)
    print(f'completed {size} {network}')
    return size, network, best_params, score, average_precision, auc_roc


def map_run(args):
    return dream4_single(**args)


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

    sizes = [100]
    networks = [1, 2, 3, 4, 5]
    # create all the cases by combining sizes and networks
    permuts = list(itertools.product(sizes, networks))
    cases = [{'size': size, 'network': network, 'estimator_t': estimator_t} for size, network in permuts]
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
    ROC_stack = {}
    all_output.sort(key=lambda x: x[0])
    for size in sizes:
        params_stack[size] = {}
        scores_stack[size] = {}
        PR_stack[size] = {}
        ROC_stack[size] = {}
        output = [x for x in all_output if x[0] == size]
        for network in networks:
            output_network = next(x for x in output if x[1] == network)
            params_stack[size][network] = output_network[2]
            scores_stack[size][network] = output_network[3]
            PR_stack[size][network] = output_network[4]
            ROC_stack[size][network] = output_network[5]
    with open(os.path.join(dir_main, f'results/dream4/{estimator_t}_best_params.txt'), 'w') as ff:
        print({'best_params': params_stack}, file=ff)
    with open(os.path.join(dir_main, f'results/dream4/{estimator_t}_best_scores.txt'), 'w') as ff:
        print({'best_scores': scores_stack}, file=ff)
    with open(os.path.join(dir_main, f'results/dream4/{estimator_t}_roc_pr_auc.txt'), 'w') as ff:
        print({'PR': PR_stack, 'ROC': ROC_stack}, file=ff)
