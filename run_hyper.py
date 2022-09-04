# precision-accuracy
import pathlib
from geneRNI import geneRNI as ni
from geneRNI import tools
from geneRNI import search_param
import os
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

import importlib
importlib.reload(ni)
importlib.reload(tools)
importlib.reload(search_param)

# these are fixed 

dir_main = dir_main = pathlib.Path(__file__).parent.resolve()
# dir_main = '/gpfs/home/nourisa/omics_estimator'

def f_golden_dream(size, network): 
    """ retreives golden links for dreams for given size and network """
    dir_ = os.path.join(dir_main,f'dynGENIE3/dynGENIE3_data/dream4/gold_standards/{size}/DREAM4_GoldStandard_InSilico_Size{size}_{network}.tsv')
    return pd.read_csv(dir_, names=['Regulator','Target','Weight'] ,sep='\t') 
def f_data_dream(size, network): 
    """ retreives train data for dreams for given size and network"""
    (TS_data, time_points, SS_data) = pd.read_pickle(os.path.join(dir_main,f'dynGENIE3/dynGENIE3_data/dream4/data/size{size}_{network}_data.pkl'))
    gene_names = [f'G{i}' for i in range(1,size+1)]
    return TS_data, time_points, SS_data, gene_names
def f_data_GRN(method, noise_level, network): 
    """ retreives train data for GRNbenchmark for given specs"""
    dir_data_benchmark = os.path.join(dir_main,'data/benchmark')
    base = method+'_'+noise_level+'_'+network 
    file_exp =  base+'_'+'GeneExpression.csv'
    file_per =  base+'_'+'Perturbations.csv'
    file_exp = os.path.join(dir_data_benchmark, file_exp)
    file_per = os.path.join(dir_data_benchmark, file_per)
    
    exp_data = pd.read_csv(file_exp)
    per_data = pd.read_csv(file_per)
    
    gene_names = exp_data['Row'].tolist()
    exp_data = np.array([exp_data[col].tolist() for col in exp_data.columns if col != 'Row'])
    per_data = [gene_names[per_data[col].tolist().index(-1)] for col in per_data.columns if col != 'Row']
    return exp_data, per_data, gene_names
def f_dir_links_dreams(size, network):
    """ returns the dir to the stored links """
    return os.path.join(dir_main,f'results/links_{size}_{network}.csv')
def prepare_data_for_study(specs):
    if specs['estimator_t'] == 'RF':
        param = dict(
            estimator_t = 'RF',
            min_samples_leaf = 1, 
            # criterion = 'absolute_error',
            n_estimators = 100, 
            # n_jobs = 10
        )
        param_grid = dict(
            min_samples_leaf = np.arange(1,10,1),
            max_depth = np.arange(10,50,5),
            alpha = np.arange(0,1,.1),
        )
        test_size = None
    elif specs['estimator_t'] =='HGB':
        param = dict(
            estimator_t = 'HGB',
            min_samples_leaf = 2, 
            # criterion = 'absolute_error',
            # max_iter = 10,
            # max_iter = 50,

        ) 
        param_grid = dict(
            learning_rate = np.arange(0.001,.2, .02),
            min_samples_leaf = np.arange(1,30,2),
            max_iter = np.arange(20,200,10),
        )
        test_size = .25
    else:
        raise ValueError('Define')
    random_state_data = 0
    random_state = None
    bootstrap_fold = 3
    
    # reformat the data
    if specs['study'] == 'dreams':
        TS_data, time_points, SS_data, gene_names = f_data_dream(specs['size'], specs['network'])
        Xs, ys = tools.process_data(TS_data=TS_data, SS_data=SS_data, gene_names=gene_names, time_points=time_points)
        if test_size is None:
            Xs_train, Xs_test, ys_train, ys_test = Xs, None, ys, None
        else:
            Xs_train, Xs_test, ys_train, ys_test = tools.train_test_split(Xs, ys, test_size = test_size, random_state=random_state_data)
 
        # Xs_train, ys_train = tools.resample(Xs_train, ys_train, n_samples = bootstrap_fold*len(ys[0]), random_state=random_state_data)
        # Xs_test, ys_test = tools.resample(Xs_test, ys_test, n_samples = bootstrap_fold*len(ys[0]), random_state=random_state)
        return dict(gene_names=gene_names, param=param, param_grid=param_grid, Xs_train=Xs_train, Xs_test=Xs_test, ys_train=ys_train, ys_test=ys_test, f_dir_links_dreams=f_dir_links_dreams, f_golden_dream=f_golden_dream)

    elif specs['study'] == 'GRNbenchmark':
        SS_data, KO, gene_names = f_data_GRN(specs['method'], specs['noise_level'], specs['network'])
        TS_data = None
        time_points = None
        KO = None #TODO: make a use of original KO
        Xs, ys = tools.process_data(TS_data=TS_data, SS_data=SS_data, gene_names=gene_names, time_points=time_points, KO=KO)
        if test_size is None:
            Xs_train, Xs_test, ys_train, ys_test = Xs, None, ys, None
        else:
            Xs_train, Xs_test, ys_train, ys_test = tools.train_test_split(Xs, ys, test_size = test_size, random_state=random_state_data)
 
        # Xs_train, ys_train = tools.resample(Xs_train, ys_train, n_samples = bootstrap_fold*len(ys[0]), random_state=random_state_data)
        # Xs_test, ys_test = tools.resample(Xs_test, ys_test, n_samples = bootstrap_fold*len(ys[0]), random_state=random_state)
        return dict(gene_names=gene_names, param=param, param_grid=param_grid, Xs_train=Xs_train, Xs_test=Xs_test, ys_train=ys_train, ys_test=ys_test)

    else:
        raise ValueError('Define first')
    
if __name__ == '__main__':
    
    specs = dict(
        n_jobs = 10,
        cv = 4,
        # n_sample=100, # for random search
        output_dir=os.path.join(dir_main,'results')
        )
    # study = 'dreams'

    study = 'GRNbenchmark'
    # estimator_t = 'RF'
    estimator_t = 'HGB'
    
    if study == 'dreams': # dream as target study
        size, network = 10, 1 # [10,100] [1-5]
        info = prepare_data_for_study(dict(size=size, network=network, study=study, estimator_t=estimator_t))
        # param search 
        best_scores, best_params, best_ests, sampled_permts = search_param.rand_search(Xs=info['Xs_train'], ys=info['ys_train'], 
                                                                                       param=info['param'], param_grid=info['param_grid'], 
                                                                                       **specs)
        print(f'param search: best score, mean: {np.mean(best_scores)} std: {np.std(best_scores)}')
        with open(f'results/param_search_dream_{size}_{network}.txt', 'w') as f:
            print({'best_scores':best_scores, 'best_params':best_params}, file=f)
    elif study == 'GRNbenchmark': # GRN as target study 
        method, noise_level, network = 'GeneNetWeaver', 'LowNoise', 'Network1'
        info = prepare_data_for_study(dict(method=method, noise_level=noise_level, network=network, study='GRNbenchmark', estimator_t=estimator_t))
        # param search 
        # best_scores, best_params, best_ests, sampled_permts = search_param.rand_search(Xs=Xs_train, ys=ys_train, param=param, param_grid=param_grid, 
        #                                                                                **specs)
        best_scores, best_params, best_ests, sampled_permts = search_param.rand_search_partial(Xs=info['Xs_train'], ys=info['ys_train'],
                                                                                       n_genes=10, param=info['param'], param_grid=info['param_grid'],
                                                                                       **specs)
        print(f'param search: best score, mean: {np.mean(best_scores)} std: {np.std(best_scores)}')
        with open(f'results/param_search_{method}_{noise_level}_{network}.txt', 'w') as f:
            print({'best_scores':best_scores, 'best_params':best_params}, file=f)
    