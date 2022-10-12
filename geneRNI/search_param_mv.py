"""This is the example module.

This module does stuff.
"""
__all__ = ['a', 'b', 'c']
__version__ = '0.1'
__author__ = 'Jalil Nourisa'

import time
import os
import numpy as np
import itertools
import operator 
import random
from pathos.pools import ParallelPool as Pool
import matplotlib.pyplot as plt

from sklearn import model_selection

from .geneRNI import GeneEstimator

def evaluate_single(est, X, y, permt, cv = 4):
    """ evalutes the estimator for a given param and returns a test score. 
    for RF, the test score is oob_score. For the rest, cv is applied.
    """
    # if param['estimator_t'] == 'RF' :
    #     use_oob_flag = True
    # else:
    #     use_oob_flag = False
    use_oob_flag = True #TODO: needs fixing
    
    # est = GeneEstimator(**param)
    params = est.get_params()
    params = {**params,**permt}
    est.set_params(**params)
    if use_oob_flag:
        fit = est.fit(X,y)
        score = est.est.oob_score_
    else:
        cv_a = model_selection.ShuffleSplit(n_splits=cv, test_size=(1/cv)) 
        scores = model_selection.cross_val_score(est, X, y, cv=cv_a)
        score = np.mean(scores)
    return score

def map_permut(args):
    """ maps the args to the grid search function for single permutation of param, used for multi threating """
    i = args['i'] # index of each target
    args_rest = {key:value for key,value in args.items() if key != 'i'}
    return (i, evaluate_single(**args_rest))

def search(est, X, y, permts, n_jobs, **specs):
    """Evaluates the permts and returns the best results for each gene """
    time_start = time.time()
    input_data =  [{'i':i,'est': est, 'X': X, 'y': y, 'permt': permts[i], **specs} for i in range(len(permts))]
    if n_jobs == 1: # serial #TODO: fix this
        all_output = list(map(map_permut, input_data))
    else: # parallel
        # multithreading happens either on gene_n or permuts, depending which one is bigger
        pool = Pool(n_jobs)
        all_output = pool.map(map_permut, input_data)
    try:
        all_output.sort(key = lambda x: x[0])
    except TypeError:
        print('Error in the outputs of map permut')
        print(all_output)
        raise TypeError()
        
    scores = []
    ests = []
    for output in all_output: # each output is for a permut
        score = output[1] # for all genes 
        scores.append(score)
    best_score = max(scores)
    best_indice = scores.index(best_score)
    best_param = permts[best_indice]
    time_end = time.time()
    print('Param search is completed in %.3f seconds' % ((time_end-time_start)))
    return best_score, best_param 
def permutation(param_grid, output_dir=None):
    tags = list(param_grid.keys())
    values = list(param_grid.values())
    permts = []
    for value in list(itertools.product(*values)):
        permts.append({tag:i for tag,i in zip(tags,value)})
    if output_dir is not None:
        with open(os.path.join(output_dir,'permts.txt'),'w') as f:
            print({'permts':permts}, file=f)
    return permts

def grid_search(X,y, param, param_grid, n_jobs = 1, **specs):
    """ 
    spces -- grid search related params such as CV which is passed to single grid search
     """
    # TODO: verify the inputs
    # if n_jobs are given for param, remove it
    
    # permuation of param grid
    # print('Grid params:', param_grid)
    permts = permutation(param_grid)
    print('stats: %d permts %d threads' % (len(permts), n_jobs))
    print(f'Running complete samples {len(permts)}')
    return search(X, y, param, param_grid, permts=permts, n_jobs=n_jobs, **specs)
   
def rand_search(est, X, y, param_grid, n_jobs = 1, n_sample=60, output_dir=None, **specs):
    # print('Grid params:', param_grid)
    permts = permutation(param_grid,output_dir)
    print('stats: %d permts %d threads' % ( len(permts), n_jobs))
    if len(permts) < n_sample:
        sampled_permts = permts
    else:
        sampled_permts = random.sample(permts, n_sample)
    if output_dir is not None:
        with open(os.path.join(output_dir, 'sampled_permts.txt'), 'w') as f:
            print({'permts':sampled_permts},file=f)
    sampled_permts_sorted = {key:[item[key] for item in sampled_permts] for key in param_grid.keys()}
    print(f'Running {len(sampled_permts)} samples randomly')

    best_score, best_param = search(est, X, y, permts=sampled_permts, n_jobs=n_jobs, **specs)
    return best_score, best_param, sampled_permts_sorted
