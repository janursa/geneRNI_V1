"""This is the main geneRNI module.


"""
__all__ = []
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
from .data import Data


def evaluate_single(X: np.ndarray, y: np.ndarray, param: dict, cv: int = 5, train_flag=False, **specs) -> Tuple[GeneEstimator, float]:
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
        test_score = est.est.oob_score_
    else:
        rs = model_selection.ShuffleSplit(n_splits=cv, test_size=0.2)
        oo = model_selection.cross_validate(est, X, y, cv=rs, return_train_score=None)
        test_score = np.mean(oo['test_score'])

    return test_score


def grid_search_single_gene(X: np.ndarray, y: np.ndarray, param: dict, permts: dict, **specs):
    """ evaluate all the permutations for one gene,  """
    #- store the data of all permutation
    testscores = []
    for permt in permts:  
        param_a = {**param, **permt}
        testscore = evaluate_single(X, y, param_a,**specs)
        testscores.append(testscore)
    # find the best candidate amongst different permutation. Max score is considered best score. 
    # best_testscore = max(testscores)
    # index = testscores.index(best_testscore)
    # best_param = permts[index]
    # best_est = fits[index]
    # best_trainscore = trainscores[index]

    # return best_trainscore, best_testscore, best_param, best_est
    return testscores


def grid_search_single_permut(Xs, ys, param: dict, permt: dict, cv: int = 4, **specs):
    """ evalute all genes for one permutation of param """
    param_a = {**param, **permt}
    testscores = []
    for X, y in zip(Xs, ys):
        testscore = evaluate_single(X, y, param_a,**specs)
        testscores.append(testscore)
    return testscores


def map_gene(args):
    """ maps the args to the grid search function for single target, used for multi threating """
    i = args['i']  # index of each target
    args['X'], args['y'] = args['data'][i]
    del args['data']
    args_rest = {key: value for key, value in args.items() if key != 'i'}
    out = i, grid_search_single_gene(**args_rest)
    return out


def map_permut(args):
    """ maps the args to the grid search function for single permutation of param, used for multi threating """
    i = args['i']  # index of each permutation

    args_rest = {key: value for key, value in args.items() if key != 'i'}
    return i, grid_search_single_permut(**args_rest)


def run(data: Data, param: dict, permts:list, n_jobs:int, **specs):
    """Evaluates all permts for each genes"""
    time_start = time.time()
    if 'n_jobs' in param: 
        del param['n_jobs']
    n_genes = len(data)
    # Run the search
    testscores = [None for _ in range(n_genes)] # for all nodes
    if n_jobs == 1:  # serial
        for i in range(n_genes):
            output = map_gene({'i': i, 'data': data, 'param': param, 'permts': permts, **specs})
            testscores_node = output[1]
            testscores[i] = testscores_node
    else:  # parallel
        # multithreading happens either on gene_n or permuts, depending which one is bigger
        pool = Pool(n_jobs)
        if n_genes >= len(permts):
            print('Gene-based multi threading')
            map_input = [{'i': i, 'data': data, 'param': param, 'permts': permts, **specs} for i in range(n_genes)]
            all_output = pool.map(map_gene, map_input)
            all_output.sort(key=lambda x: x[0])
            for i_gene, output in enumerate(all_output):  # an output for a gene
                testscores_node = output[1]
                testscores[i_gene] = testscores_node
        else:  # when there is more permuts
            print('Permutation-based multi threading')            
            Xs = []
            ys = []
            for i in range(n_genes):
                X, y = data[i]
                Xs.append(X)
                ys.append(y)

            input_data = [{'i': i, 'Xs': Xs, 'ys':ys, 'param': param, 'permt': permts[i], **specs} for i in range(len(permts))]
            all_output = pool.map(map_permut, input_data)
            
            all_output.sort(key=lambda x: x[0])

            #- extract results for all permutations for each genes
            testscores = np.empty([n_genes, 0])
            for output in all_output:  # each output is for a permut
                testscores_all_genes = output[1]  # for all genes
                testscores = np.c_[testscores, testscores_all_genes]
            
    time_end = time.time()
    print('Param search is completed in %.3f seconds' % (time_end-time_start))
    return testscores


def permutation(param_grid):
    tags = list(param_grid.keys())
    values = list(param_grid.values())
    permts = []
    for value in list(itertools.product(*values)):
        permts.append({tag: i for tag, i in zip(tags, value)})
    return permts


def rand_search(data: Data, param, param_grid, n_jobs=1, n_sample=60, random_state=None, i_start=0, i_end=1, output_dir='', **specs):
    # print('Grid params:', param_grid)
    permts = permutation(param_grid)
    print('stats: %d genes %d permts %d threads' % (len(data), len(permts), n_jobs))
    if len(permts) < n_sample:
        sampled_permts = permts
    else:
        random.seed(random_state)
        sampled_permts = random.sample(permts, n_sample)
    #- create the result dir if it doesnt exist
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    #- give warning if the dir is not empty 
    if os.listdir(output_dir):
        print('Output directory is not empty')
    #- write sampled_permts to the file
    with open(os.path.join(output_dir, 'sampled_permts.txt'), 'w') as f:
        print({'sampled_permts': sampled_permts}, file=f)
    #- run the iteration
    print(f'Running {len(sampled_permts)} samples randomly from {i_start} to {i_end} iterations')
    for i in range(i_start, i_end):
        print(f'----Run iteration {i}-----')
        testscores = run(data, param, permts=sampled_permts, n_jobs=n_jobs, **specs)
        print(f'Mean best score: {np.mean(np.max(testscores,axis=1))}')
        #- write the scores to the file
        FILE = os.path.join(output_dir, f'data_{i}.txt')
        np.savetxt(FILE, testscores, delimiter=',')

def pool(output_dir, n_repeat):
    """Pools iterative results 
    Each data in stack contains node results for all permuts, i.e. scores[i_gene][i_purmut].
    """

    #- read the data and create pool and average values
    with open(os.path.join(output_dir, 'sampled_permts.txt')) as f:
        sampled_permts = eval(f.read())['sampled_permts']
    scores_stack = [] 
    for i in range(n_repeat):
        FILE = os.path.join(output_dir, f'data_{i}.txt')
        scores = np.genfromtxt(FILE, delimiter=',')
        scores_stack.append(scores)    
    print('stack shape: n_repeat*n_genes*n_permut: ',np.array(scores_stack).shape)
    #- reformat from [n_repeat][i_gene][i_purmut] to [i_gene][i_purmut][n_repeat]
    scores_pool = np.stack(scores_stack, axis=2)
    # print(scores_pool[2][0])
    # print('scores pool shape: n_genes*n_permut*n_repeat: ', scores_pool.shape)
    #- best params
    scores_mean = np.mean(scores_pool,axis=2)
    # print('scores mean shape: n_genes*n_permut: ', scores_mean.shape)
    best_scores = np.max(scores_mean, axis=1)
    # print('scores best shape: n_genes: ', best_scores.shape)
    best_indices = np.argmax(scores_mean, axis=1) # highest scores across mutations
    best_params= [sampled_permts[i] for i in best_indices]
    print(f'Best score -> min:{np.min(best_scores)},  average: {np.mean(best_scores)}, std: {np.std(best_scores)}')
    return best_scores, best_params




