"""This is the example module.

This module does stuff.
"""
__all__ = ['a', 'b', 'c']
__version__ = '0.1'
__author__ = 'Jalil Nourisa'

import itertools
import os
import random
import time
from typing import Tuple

import numpy as np
from pathos.pools import ParallelPool as Pool
from sklearn import model_selection

from .geneRNI import GeneEstimator


def evaluate_single(X: np.ndarray, y: np.ndarray, param: dict, cv: int = 4, train_flag=False, **specs) -> Tuple[GeneEstimator, float]:
    """ evalutes the gene estimator for a given param and returns a test score 
    for RF, the test score is oob_score. For the rest, cv is applied.
    train_flag -- to use training score 
    """
    if param['estimator_t'] == 'RF':
        use_oob_flag = True
    else:
        use_oob_flag = False
    
    est = GeneEstimator(**param)
    if use_oob_flag:
        est.fit(X, y)
        if train_flag:
            score = est.score(X,y)
        else:
            score = est.est.oob_score_
    else:
        cv_a = model_selection.ShuffleSplit(n_splits=cv, test_size=(1/cv)) 
        scores = model_selection.cross_val_score(est, X, y, cv=cv_a)
        score = np.mean(scores)
    return est, float(score)


def grid_search_single_gene(X: np.ndarray, y: np.ndarray, param: dict, permts: dict, cv: int = 4, **specs):
    """ evaluate all the permutations for a gene, and returns the best fit """
    # evaluate each permutation
    fits = []
    scores = []
    for permt in permts:  
        param_a = {**param, **permt}
        fit, score = evaluate_single(X, y, param_a, cv,**specs)
        fits.append(fit)
        scores.append(score)
    # find the best candidate. Max score is considered best score. 
    best_score = max(scores)
    index = scores.index(best_score)
    best_param = permts[index]
    best_est = fits[index]

    return best_score, best_param, best_est


def grid_search_single_permut(Xs, ys, param: dict, permt: dict, cv: int = 4, **specs):
    """ evalute all genes for the one permutation of param, and returns the best fit """
    param_a = {**param, **permt}
    fits = []
    scores = []
    for X, y in zip(Xs, ys):
        fit, score = evaluate_single(X, y, param_a, cv,**specs)
        fits.append(fit)
        scores.append(score)
    return scores, fits


def map_gene(args):
    """ maps the args to the grid search function for single target, used for multi threating """
    i = args['i']  # index of each target
    args_rest = {key: value for key, value in args.items() if key != 'i'}
    return i, grid_search_single_gene(**args_rest)


def map_permut(args):
    """ maps the args to the grid search function for single permutation of param, used for multi threating """
    i = args['i']  # index of each target
    args_rest = {key: value for key, value in args.items() if key != 'i'}
    return i, grid_search_single_permut(**args_rest)


def search(Xs, ys, param, param_grid, permts, n_jobs, output_dir=None, **specs):
    """Evaluates the permts and returns the best results for each gene """
    time_start = time.time()
    if 'n_jobs' in param: 
        del param['n_jobs']
    n_genes = len(ys)
    # run the search
    best_scores = [None for _ in range(n_genes)]
    best_params = [None for _ in range(n_genes)]
    best_ests = [None for _ in range(n_genes)]
    if n_jobs == 1:  # serial
        map_input = [{'i': i, 'X': Xs[i], 'y': ys[i], 'param': param, 'permts': permts, **specs} for i in range(n_genes)]
        all_output = list(map(map_gene, map_input))
        all_output.sort(key=lambda x: x[0])
        for i_gene, output in enumerate(all_output):  # an output for a gene
            best_score, best_param, best_est = output[1]
            best_scores[i_gene] = best_score
            best_params[i_gene] = best_param
            best_ests[i_gene] = best_est
    else:  # parallel
        # multithreading happens either on gene_n or permuts, depending which one is bigger
        pool = Pool(n_jobs)
        if n_genes >= len(permts):
            print('Gene-based multi threading')
            map_input = [{'i': i, 'X': Xs[i], 'y': ys[i], 'param': param, 'permts': permts, **specs} for i in range(n_genes)]
            all_output = pool.map(map_gene, map_input)
            all_output.sort(key=lambda x: x[0])
            for i_gene, output in enumerate(all_output):  # an output for a gene
                best_score, best_param, best_est = output[1]
                best_scores[i_gene] = best_score
                best_params[i_gene] = best_param
                best_ests[i_gene] = best_est
        else:  # when there is more permuts
            print('Permutation-based multi threading')
            input_data = [{'i': i, 'Xs': Xs, 'ys': ys, 'param': param, 'permt': permts[i], **specs} for i in range(len(permts))]
            all_output = pool.map(map_permut, input_data)
            try:
                all_output.sort(key=lambda x: x[0])
            except TypeError:
                print('Error in the outputs of map permut')
                print(all_output)
                raise TypeError()

            scores = np.empty([n_genes, 0])
            ests = np.empty([n_genes, 0])
            for output in all_output:  # each output is for a permut
                scores_all_genes, ests_all_genes = output[1]  # for all genes
                scores = np.c_[scores, scores_all_genes]
                ests = np.c_[ests, ests_all_genes]
            best_scores = list(np.max(scores, axis=1))
            best_indices = np.array(scores.argmax(1))
            best_ests = ests[range(n_genes), best_indices]
            best_params = [permts[i] for i in best_indices]
    time_end = time.time()
    print('Param search is completed in %.3f seconds' % (time_end-time_start))
    if output_dir is not None:
        with open(os.path.join(output_dir, 'best_params.txt'), 'w') as f:
            print({'best_params': best_params}, file=f)
        with open(os.path.join(output_dir, 'best_scores.txt'), 'w') as f:
            print({'best_scores': best_scores}, file=f)
    return best_scores, best_params, best_ests


def permutation(param_grid, output_dir=None):
    tags = list(param_grid.keys())
    values = list(param_grid.values())
    permts = []
    for value in list(itertools.product(*values)):
        permts.append({tag: i for tag, i in zip(tags, value)})
    if output_dir is not None:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        with open(os.path.join(output_dir, 'permts.txt'), 'w') as f:
            print({'permts': permts}, file=f)
    return permts


def grid_search(Xs, ys, param, param_grid, n_jobs=1, **specs):
    """ 
    spces -- grid search related params such as CV which is passed to single grid search
    """
    # TODO: verify the inputs
    # if n_jobs are given for param, remove it
    
    # permuation of param grid
    # print('Grid params:', param_grid)
    permts = permutation(param_grid)
    print('stats: %d genes %d permts %d threads' % (Xs[0].shape[1], len(permts), n_jobs))
    print(f'Running complete samples {len(permts)}')
    return search(Xs, ys, param, param_grid, permts=permts, n_jobs=n_jobs, **specs)


def rand_search(Xs, ys, param, param_grid, n_jobs=1, n_sample=60, output_dir=None, **specs):
    # print('Grid params:', param_grid)
    permts = permutation(param_grid, output_dir)
    print('stats: %d genes %d permts %d threads' % (Xs[0].shape[1], len(permts), n_jobs))
    if len(permts) < n_sample:
        sampled_permts = permts
    else:
        sampled_permts = random.sample(permts, n_sample)
    if output_dir is not None:
        with open(os.path.join(output_dir, 'sampled_permts.txt'), 'w') as f:
            print({'permts': sampled_permts}, file=f)
    sampled_permts_sorted = {key: [item[key] for item in sampled_permts] for key in param_grid.keys()}
    print(f'Running {len(sampled_permts)} samples randomly')

    best_scores, best_params, best_ests = search(Xs, ys, param, param_grid, permts=sampled_permts, n_jobs=n_jobs, output_dir=output_dir, **specs)
    return best_scores, best_params, best_ests, sampled_permts_sorted


def rand_search_partial(Xs, ys, param, param_grid, n_genes, n_jobs = 1, n_sample=60, output_dir=None, **specs):
    """Conducts random search on hyper-parameters, only partially on n_genes, randomly selected from all genes"""
    permts = permutation(param_grid, output_dir)
    print(f'partial rand search on {n_genes} genes with permts {len(permts)}  threads {n_jobs}')
    if len(permts) < n_sample:
        sampled_permts = permts
    else:
        sampled_permts = random.sample(permts, n_sample)
    if output_dir is not None:
        with open(os.path.join(output_dir, 'sampled_permts.txt'), 'w') as f:
            print({'permts': sampled_permts}, file=f)
    sampled_permts_sorted = {key: [item[key] for item in sampled_permts] for key in param_grid.keys()}
    print(f'Running {len(sampled_permts)} samples randomly')
    # choose n genes randomly from all genes
    choice = [random.randint(0, len(ys)-1) for i in range(n_genes)]
    best_scores, best_params, best_ests = search(Xs[choice], ys[choice], param, param_grid, permts=sampled_permts, n_jobs=n_jobs, **specs)
    return best_scores, best_params, best_ests, sampled_permts_sorted
