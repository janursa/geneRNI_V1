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
        train_score = est.score(X,y)
        test_score = est.est.oob_score_
    else:
        cv_a = model_selection.ShuffleSplit(n_splits=cv, test_size=(1/cv)) 
        test_scores = model_selection.cross_val_score(est, X, y, cv=cv_a)
        oo = model_selection.cross_validate(est, X, y, cv=cv_a, return_train_score=True)
        
        test_score = np.mean(oo['test_score'])
        train_score = np.mean(oo['train_score'])
    return est, train_score, test_score


def grid_search_single_gene(X: np.ndarray, y: np.ndarray, param: dict, permts: dict, cv: int = 4, **specs):
    """ evaluate all the permutations for one gene,  """
    fits = []
    trainscores = []
    testscores = []
    for permt in permts:  
        param_a = {**param, **permt}
        fit, trainscore, testscore = evaluate_single(X, y, param_a, cv,**specs)
        fits.append(fit)
        trainscores.append(trainscore)
        testscores.append(testscore)
    # find the best candidate. Max score is considered best score. 
    best_testscore = max(testscores)
    index = testscores.index(best_testscore)
    best_param = permts[index]
    best_est = fits[index]
    best_trainscore = trainscores[index]

    return best_trainscore, best_testscore, best_param, best_est


def grid_search_single_permut(Xs, ys, param: dict, permt: dict, cv: int = 4, **specs):
    """ evalute all genes for one permutation of param """
    param_a = {**param, **permt}
    fits = []
    trainscores = []
    testscores = []
    for X, y in zip(Xs, ys):
        fit, trainscore, testscore = evaluate_single(X, y, param_a, cv,**specs)
        fits.append(fit)
        trainscores.append(trainscore)
        testscores.append(testscore)
    return trainscores, testscores, fits


def map_gene(args):
    """ maps the args to the grid search function for single target, used for multi threating """
    i = args['i']  # index of each target
    data = args['data']
    X_train, _, y_train, _ = data[i]
    args['X'] = X_train
    args['y'] = y_train
    args_rest = {key: value for key, value in args.items() if key != 'i'}
    out = i, grid_search_single_gene(**args_rest)
    return out


def map_permut(args):
    """ maps the args to the grid search function for single permutation of param, used for multi threating """
    i = args['i']  # index of each permutation

    args_rest = {key: value for key, value in args.items() if key != 'i'}
    return i, grid_search_single_permut(**args_rest)


def search(data: Data, param, permts, n_jobs, output_dir=None, **specs):
    """Evaluates the permts and returns the best results for each gene """
    time_start = time.time()
    if 'n_jobs' in param: 
        del param['n_jobs']
    n_genes = len(data)

    # Run the search
    best_testscores = [None for _ in range(n_genes)]
    best_trainscores = [None for _ in range(n_genes)]
    best_params = [None for _ in range(n_genes)]
    best_ests = [None for _ in range(n_genes)]
    if n_jobs == 1:  # serial
        for i in range(n_genes):
            output = map_gene({'i': i, 'data': data, 'param': param, 'permts': permts, **specs})
            best_trainscore, best_testscore, best_param, best_est = output[1]
            best_testscores[i] = best_testscore
            best_trainscores[i] = best_trainscore
            best_params[i] = best_param
            best_ests[i] = best_est
    else:  # parallel
        # multithreading happens either on gene_n or permuts, depending which one is bigger
        pool = Pool(n_jobs)
        if n_genes >= len(permts):
            print('Gene-based multi threading')
            map_input = [{'i': i, 'data': data, 'param': param, 'permts': permts, **specs} for i in range(n_genes)]
            all_output = pool.map(map_gene, map_input)
            all_output.sort(key=lambda x: x[0])
            for i_gene, output in enumerate(all_output):  # an output for a gene
                best_trainscore, best_testscore, best_param, best_est = output[1]
                best_testscores[i_gene] = best_testscore
                best_trainscores[i_gene] = best_trainscore
                best_params[i_gene] = best_param
                best_ests[i_gene] = best_est
        else:  # when there is more permuts
            print('Permutation-based multi threading')            
            Xs = []
            ys = []
            for i in range(n_genes):
                X_train, _, y_train, _ = data[i]
                Xs.append(X_train)
                ys.append(y_train)

            input_data = [{'i': i, 'Xs': Xs, 'ys':ys, 'param': param, 'permt': permts[i], **specs} for i in range(len(permts))]
            all_output = pool.map(map_permut, input_data)
            # print(all_output)
            try:
                all_output.sort(key=lambda x: x[0])
            except TypeError:
                print('Error in the outputs of map permut')
                print(all_output)
                raise TypeError()

            testscores = np.empty([n_genes, 0])
            trainscores = np.empty([n_genes, 0])
            ests = np.empty([n_genes, 0])
            for output in all_output:  # each output is for a permut
                trainscores_all_genes, testscores_all_genes, ests_all_genes = output[1]  # for all genes
                testscores = np.c_[testscores, testscores_all_genes]
                trainscores = np.c_[trainscores, trainscores_all_genes]
                ests = np.c_[ests, ests_all_genes]
            best_trainscores = list(np.max(trainscores, axis=1))
            best_testscores = list(np.max(testscores, axis=1))
            best_indices = np.array(testscores.argmax(1))
            best_ests = ests[range(n_genes), best_indices]
            best_params = [permts[i] for i in best_indices]
    time_end = time.time()
    print('Param search is completed in %.3f seconds' % (time_end-time_start))
    return best_trainscores, best_testscores, best_params, best_ests


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
    return search(Xs, ys, param, permts=permts, n_jobs=n_jobs, **specs)



def rand_search(data: Data, param, param_grid, n_jobs=1, n_sample=60, output_dir=None, random_state=None, **specs):
    # print('Grid params:', param_grid)
    permts = permutation(param_grid, output_dir)
    print('stats: %d genes %d permts %d threads' % (len(data), len(permts), n_jobs))
    if len(permts) < n_sample:
        sampled_permts = permts
    else:
        random.seed(random_state)
        sampled_permts = random.sample(permts, n_sample)

    sampled_permts_sorted = {key: [item[key] for item in sampled_permts] for key in param_grid.keys()}

    if output_dir is not None:
        with open(os.path.join(output_dir, 'sampled_permts.txt'), 'w') as f:
            print({'permts': sampled_permts}, file=f)
    print(f'Running {len(sampled_permts)} samples randomly')

    best_trainscores, best_testscores, best_params, best_ests = search(data, param, permts=sampled_permts, n_jobs=n_jobs, output_dir=output_dir, **specs)
    return best_trainscores, best_testscores, best_params, best_ests, sampled_permts_sorted


def rand_search_partial(Xs, ys, param, param_grid, n_genes, n_jobs = 1, n_sample=60, **specs):
    """Conducts random search on hyper-parameters, only partially on n_genes, randomly selected from all genes"""
    permts = permutation(param_grid, output_dir)
    print(f'partial rand search on {n_genes} genes with permts {len(permts)}  threads {n_jobs}')
    if len(permts) < n_sample:
        sampled_permts = permts
    else:
        sampled_permts = random.sample(permts, n_sample)
    sampled_permts_sorted = {key: [item[key] for item in sampled_permts] for key in param_grid.keys()}
    print(f'Running {len(sampled_permts)} samples randomly')
    # choose n genes randomly from all genes
    choice = [random.randint(0, len(ys)-1) for i in range(n_genes)]
    best_scores, best_params, best_ests = search(Xs[choice], ys[choice], param, param_grid, permts=sampled_permts, n_jobs=n_jobs, **specs)
    return best_scores, best_params, best_ests, sampled_permts_sorted
