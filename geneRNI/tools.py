"""This is the example module.

This module does stuff.
"""
__all__ = ['a', 'b', 'c']
__version__ = '0.1'
__author__ = 'Jalil Nourisa'

import os
import sys
import time
import warnings

import numpy as np
import operator
import itertools
import pandas as pd
import pathlib

import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn import model_selection
from sklearn import utils

from scipy import sparse

from geneRNI.models import get_estimator_wrapper

dir_main = os.path.join(pathlib.Path(__file__).parent.resolve(), '..')
sys.path.insert(0, dir_main)

from geneRNI import types_


def verboseprint(flag, message):
    if flag:
        print(message)


class Links:

    @staticmethod
    def format(
            links,
            gene_names,
            regulators='all',
            KO=None,  # TODO: unused
            regulator_tag='Regulator',
            target_tag='Target',
            weight_tag='Weight',
            sign_tag='Sign'
    ):

        """Gets the regulatory links in a matrix ({gene1-gene1, gene1-gene2, ...; gene2-gene1, gene2-gene2, etc}) converts it to a df.
        
        Parameters
        ----------
        
        gene_names: list of strings, optional
            List of length p, where p is the number of rows/columns in VIM, containing the names of the genes. The i-th item of gene_names must correspond to the i-th row/column of VIM. When the gene names are not provided, the i-th gene is named Gi.
            default: None
            
        regulators: list of strings, optional
            List containing the names of the candidate regulators. When a list of regulators is provided, the names of all the genes must be provided (in gene_names), and the returned list contains only edges directed from the candidate regulators. When regulators is set to 'all', any gene can be a candidate regulator.
            default: 'all'
            
        Returns
        -------
        
        A df with the format:            
            Regulator   Target     Weight    Sign
        """

        # Check input arguments     
        VIM = np.array(links)
        # if not isinstance(VIM,ndarray):
        #     raise ValueError('VIM must be a square array')
        # elif VIM.shape[0] != VIM.shape[1]:
        #     raise ValueError('VIM must be a square array')

        ngenes = VIM.shape[0]

        nTFs = ngenes

        # remove gene to itself regulatory score
        i_j_links = [(i, j, score) for (i, j), score in np.ndenumerate(VIM) if i != j]
        # Rank the list according to the weights of the edges    
        i_j_links_sort = sorted(i_j_links, key=operator.itemgetter(2), reverse=True)
        nToWrite = len(i_j_links_sort)

        # Write the ranked list of edges
        regs = []
        targs = []
        scores = []
        for i in range(nToWrite):
            (TF_idx, target_idx, score) = i_j_links_sort[i]
            TF_idx = int(TF_idx)
            target_idx = int(target_idx)
            regs.append(gene_names[TF_idx])
            targs.append(gene_names[target_idx])
            scores.append(score)

        df = pd.DataFrame()
        df[regulator_tag] = regs
        df[target_tag] = targs
        df[weight_tag] = scores
        df[sign_tag] = ''
        df = Links.sort(df, gene_names)
        return df


    @staticmethod
    def output(links_df, file_name):
        if file_name is not None and not isinstance(file_name, str):
            raise ValueError('input argument file_name must be a string')
        links_df.to_csv(file_name, index=False)

    @staticmethod
    def sort(links, sorted_gene_names, regulator_tag='Regulator', target_tag='Target', weight_tag='Weight'):
        """ Sorts links in based on gene numbers. The output looks like:
            Regulator    Target     Weight
            G1             G2         0.5
            G1             G3         0.8
        links --  Target Regulator Weight as a database
        sorted_gene_names -- gene names sorted
        """
        # TODO: how to deal with missing genes

        for i, gene in enumerate(sorted_gene_names):
            df_gene = links.loc[links[regulator_tag] == gene]
            sorted_gene_names_a = [x for x in sorted_gene_names if x != gene]
            df_gene.loc[:, target_tag] = pd.Categorical(df_gene[target_tag], sorted_gene_names_a)
            df_gene_sorted = df_gene.sort_values(target_tag)
            if i == 0:
                sorted_links = df_gene_sorted
            else:
                sorted_links = pd.concat([sorted_links, df_gene_sorted], axis=0, ignore_index=True)
        return sorted_links


class Data:

    @staticmethod
    def process_time_series(TS_data, time_points, gene_names, regulators='all', KO=None):
        """ Reformat data for time series analysis 
        
        """
        ngenes = len(gene_names)
        # apply knockout 
        if KO is not None:
            KO_indices = []
            for gene in KO:
                KO_indices.append(gene_names.index(gene))

        # Re-order time points in increasing order
        for (i, tp) in enumerate(time_points):
            tp = np.array(tp, np.float32)
            indices = np.argsort(tp)
            time_points[i] = tp[indices]
            expr_data = TS_data[i]
            TS_data[i] = expr_data[indices, :]
        # obtain X and y for each target in a n_sample * n_feature, where n_sample is (n_exp*n_time - n_exp)
        Xs = []
        ys = []
        h = 1  # lag used for the finite approximation of the derivative of the target gene expression

        for i_gene in range(ngenes):
            if regulators == 'all':
                input_idx = list(range(ngenes))
            else:
                input_idx = regulators[i_gene]
            try:
                input_idx.remove(KO_indices[i_gene])
            except UnboundLocalError:
                pass
            nexp = len(TS_data)
            nsamples_time = sum([expr_data.shape[0] for expr_data in TS_data])
            ninputs = len(input_idx)

            # Time-series data
            input_matrix_time = np.zeros((nsamples_time - h * nexp, ninputs))
            output_vect_time = []

            for (i, exp_timeseries) in enumerate(TS_data):
                exp_time_points = time_points[i]
                n_time = exp_timeseries.shape[0]
                exp_time_diff = exp_time_points[h:] - exp_time_points[:n_time - h]
                exp_timeseries_x = exp_timeseries[:n_time - h, input_idx]
                # current_timeseries_output = (exp_timeseries[h:,i_gene] - exp_timeseries[:n_time-h,i_gene]) / exp_time_diff + alphas[i_gene]*exp_timeseries[:n_time-h,i_gene]
                for ii in range(len(exp_time_diff)):
                    f_dy_dt = lambda alpha_i, i=i, ii=ii, i_gene=i_gene: float(
                        (TS_data[i][ii + 1:ii + 2, i_gene] - TS_data[i][ii:ii + 1, i_gene]) / exp_time_diff[
                            ii] + alpha_i * TS_data[i][ii:ii + 1, i_gene])
                    output_vect_time.append(f_dy_dt)

                exp_n_samples = exp_timeseries_x.shape[0]
                input_matrix_time[i * exp_n_samples:(i + 1) * exp_n_samples, :] = exp_timeseries_x

            Xs.append(input_matrix_time)
            ys.append(output_vect_time)

        return Xs, ys

    @staticmethod
    def process_static(SS_data, gene_names, regulators='all', perturbations=None, KO=None):
        """ Reformat data for static analysis 
        SS_data -- static data in the format n_samples*n_genes
        perturbations -- initial changes to the genes such as adding certain values. n_samples*n_genes
        KO -- the list of knock-out gene names. For now, each row has 1 gene name. TODO: one gene for all samples; more than one genes for one sample
        
        output:
        Xs -- X inputs for each gene. n_genes*n_samples*n_genes 
        ys -- Y for each gene. n_genes*n_samples
        """
        ngenes = len(SS_data[0])
        # obtain X and y for each target in a n_sample * n_feature
        Xs = []
        ys = []

        ko_indices = []
        if KO is not None:
            for gene in KO:
                ko_indices.append(gene_names.index(gene))

        for i_gene in range(ngenes):
            if regulators == 'all':
                input_idx = list(range(ngenes))
            else:
                input_idx = regulators[i_gene]
            try:
                input_idx.remove(ko_indices[i_gene])
            except UnboundLocalError:
                pass
            X = SS_data[:, input_idx]
            
            Xs.append(X)
            y = []
            for i_sample, sample_data in enumerate(SS_data):
                # add perturbations
                if perturbations is not None:
                    f_dy_dt = float(SS_data[i_sample][i_gene]) - float(perturbations[i_sample][i_gene]) 
                else:
                    f_dy_dt = float(SS_data[i_sample][i_gene])
                y.append(f_dy_dt)
            ys.append(y)
        return Xs, ys

    @staticmethod
    def process(TS_data=None, SS_data=None, time_points=None, gene_names=None, regulators='all', KO=None, verbose=True):
        """ Reformats the raw data for both static and dynamic analysis

        For time series data, TS_data should be in n_exp*n_time*n_genes format. For For static analysis, 
        SS_data should be in n_samples * n_genes.
        
        Arguments:
        TS_data --
        KO -- knock-out genes. Either a single list (repeated experiments) or a list of lists (different across experiments)  

        Return: 
        Xs -- A list of training inputs for each target gene, each item has n_samples * n_regulators format. For dynamic analysis, n_samples = n_exp*n_time - n_exp.
        y -- A list of training outputs for each target gene, each item has n_samples. For dynamic analysis, n_samples = n_exp*n_time - n_exp.
        """
        # Check input arguments
        dynamic_flag = False
        static_flag = False
        if TS_data is not None:
            dynamic_flag = True
        if SS_data is not None:
            static_flag = True
        # if dynamic_flag and not isinstance(TS_data,(list,tuple)):
        #     raise ValueError('TS_data must be a list of lists')
        # if static_flag and not isinstance(SS_data,(list,tuple)):
        #     raise ValueError('SS_data must be a list of list')

        # TODO: check the inputs

        # TODO: add KO to time series data
        if TS_data is not None:
            Xs_d, ys_d = Data.process_time_series(TS_data, time_points, gene_names, regulators, KO)
            verboseprint(verbose,
                         f'dynamic data: ngenes: {len(ys_d)}, nsamples: {len(ys_d[0])}, n regulators: {len(Xs_d[0][0])}')
        if SS_data is not None:
            Xs_s, ys_s = Data.process_static(SS_data, gene_names, regulators, KO)
            verboseprint(verbose,
                         f'static data: ngenes: {len(ys_s)}, nsamples: {len(ys_s[0])}, n regulators: {len(Xs_s[0][0])}')

        # combine static and dynamic data
        if TS_data is not None and SS_data is not None:
            Xs = [np.concatenate((X_s, X_d), axis=0) for X_s, X_d in zip(Xs_s, Xs_d)]
            ys = [np.concatenate((y_s, y_d), axis=0) for y_s, y_d in zip(ys_s, ys_d)]
        elif TS_data is not None:
            Xs = Xs_d
            ys = ys_d
        elif SS_data is not None:
            Xs = Xs_s
            ys = ys_s
        else:
            raise ValueError('Static and dynamic data are both None')
        Xs = np.array(Xs)
        ys = np.array(ys)
        return Xs, ys

    @staticmethod
    def train_test_split(Xs, ys, test_size=0.25, **specs):
        """ Splits the data into train and test portion based on each gene """
        n_genes = len(Xs)
        Xs_train = [0 for i in range(n_genes)]
        ys_train = [0 for i in range(n_genes)]
        Xs_test = [0 for i in range(n_genes)]
        ys_test = [0 for i in range(n_genes)]
        if test_size == None:
            return types_.DataType(Xs, ys)
        else:
            for i, (X, y) in enumerate(zip(Xs, ys)):
                Xs_train[i], Xs_test[i], ys_train[i], ys_test[i] = model_selection.train_test_split(
                        X, y, test_size=test_size, **specs)
            return types_.DataType(Xs_train, ys_train, Xs_test, ys_test)

    @staticmethod
    def resample(Xs, ys, n_samples=None, replace=True, **specs):
        """resampling for bootstraping"""
        if n_samples is None:
            n_samples = 2 * len(ys[0])

        Xs_b, ys_b = [], []
        for X, y in zip(Xs, ys):
            X_sparse = sparse.coo_matrix(X)
            X_b, _, y_b = utils.resample(X, X_sparse, y, n_samples=n_samples, replace=replace, **specs)
            # XX = utils.resample((X,y), n_samples = n_samples, replace=replace)
            # print(len(XX[0]))
            Xs_b.append(X_b)
            ys_b.append(y_b)
        return Xs_b, ys_b


class Settings:

    @staticmethod
    def default(estimator_t):
        test_size = None
        param = get_estimator_wrapper(estimator_t).get_default_parameters()
        param_grid = get_estimator_wrapper(estimator_t).get_grid_parameters()
        if estimator_t == 'HGB':
            test_size = 0.25
        random_state_data = None
        random_state = None
        bootstrap_fold = None
        return types_.DefaultParamType(param, param_grid, test_size, bootstrap_fold, random_state, random_state_data)


class Benchmark:

    @staticmethod
    def f_data_dream5(network):
        """ retreives train data for dream5 for network"""
        data = pd.read_csv(os.path.join(dir_main, f'data/dream5/trainingData/net{network}_expression_data.tsv'),
                           sep='\t')
        transcription_factors = pd.read_csv(
            os.path.join(dir_main, f'data/dream5/trainingData/net{network}_transcription_factors.tsv'), sep='\t',
            header=None)
        gene_names = data.columns.tolist()
        SS_data = data.values
        return SS_data, gene_names, transcription_factors[0].tolist()

    @staticmethod
    def f_golden_dream4(size, network):
        """ retreives golden links for dream4 for given size and network """
        dir_ = os.path.join(dir_main,
                            f'data/dream4/gold_standards/{size}/DREAM4_GoldStandard_InSilico_Size{size}_{network}.tsv')
        return pd.read_csv(dir_, names=['Regulator', 'Target', 'Weight'], sep='\t')

    @staticmethod
    def f_data_dream4(size, network):
        """ retreives train data for dream4 for given size and network"""
        (TS_data, time_points, SS_data) = pd.read_pickle(
            os.path.join(dir_main, f'data/dream4/data/size{size}_{network}_data.pkl'))
        gene_names = [f'G{i}' for i in range(1, size + 1)]
        return TS_data, time_points, SS_data, gene_names

    @staticmethod
    def f_data_melanogaster():
        """ retreives train data for melanogaster"""
        (TS_data, time_points, genes, TFs, alphas) = pd.read_pickle(
            os.path.join(dir_main, f'data/real_networks/data/drosophila_data.pkl'))
        return TS_data, time_points, genes, TFs, alphas

    @staticmethod
    def f_data_ecoli():
        """ retreives train data for ecoli"""
        (TS_data, time_points, genes, TFs, alphas) = pd.read_pickle(
            os.path.join(dir_main, f'data/real_networks/data/ecoli_data.pkl'))
        return TS_data, time_points, genes, TFs, alphas

    @staticmethod
    def f_data_cerevisiae():
        """ retreives train data for yeast"""
        (TS_data, time_points, genes, TFs, alphas) = pd.read_pickle(
            os.path.join(dir_main, f'data/real_networks/data/cerevisiae_data.pkl'))
        return TS_data, time_points, genes, TFs, alphas

    @staticmethod
    def f_data_GRN(method, noise_level, network):
        """ retreives train data for GRNbenchmark for given specs"""
        dir_data_benchmark = os.path.join(dir_main, 'data/GRNbenchmark')
        base = method + '_' + noise_level + '_' + network
        file_exp = base + '_' + 'GeneExpression.csv'
        file_per = base + '_' + 'Perturbations.csv'
        file_exp = os.path.join(dir_data_benchmark, file_exp)
        file_per = os.path.join(dir_data_benchmark, file_per)

        exp_data = pd.read_csv(file_exp)
        per_data = pd.read_csv(file_per)

        gene_names = exp_data['Row'].tolist()
        exp_data = np.array([exp_data[col].tolist() for col in exp_data.columns if col != 'Row'])
        per_data = [gene_names[per_data[col].tolist().index(-1)] for col in per_data.columns if col != 'Row']
        return exp_data, per_data, gene_names

    @staticmethod
    def process_data(TS_data, SS_data, time_points, gene_names, estimator_t, **specs):
        Xs, ys = Data.process(TS_data=TS_data, SS_data=SS_data, gene_names=gene_names, time_points=time_points, **specs)
        pp = Settings.default(estimator_t)
        out_data = Data.train_test_split(Xs, ys, test_size=pp.test_size, random_state=pp.random_state_data, **specs)
        # Xs_train, ys_train = Data.resample(Xs_train, ys_train, n_samples = bootstrap_fold*len(ys[0]), random_state=random_state_data)
        # Xs_test, ys_test = Data.resample(Xs_test, ys_test, n_samples = bootstrap_fold*len(ys[0]), random_state=random_state)
        return out_data

    @staticmethod
    def process_data_dream4(size, network, estimator_t, **specs):
        TS_data, time_points, SS_data, gene_names = Benchmark.f_data_dream4(size, network)
        return Benchmark.process_data(TS_data, SS_data, time_points, gene_names, estimator_t, **specs)

    @staticmethod
    def process_data_GRNbenchmark(method, noise_level, network, estimator_t, **specs):
        SS_data, KO, gene_names = Benchmark.f_data_GRN(method, noise_level, network)
        return Benchmark.process_data(None, SS_data, None, gene_names, estimator_t)


class GOF:
    """Goodness of fit"""

    @staticmethod
    def boxplot_scores_groupOfgroup(scores_stack_stack, tags=None, titles=None):
        """plots scores as a box plot for a group of groups (e.g. size*network)"""
        n = len(scores_stack_stack)
        fig, axes = plt.subplots(1, n, tight_layout=True, figsize=(n * 5, 4))
        for i, scores_stack in enumerate(scores_stack_stack):
            if n == 1:
                ax = axes
            else:
                ax = axes[i]
            ax.boxplot(scores_stack, showfliers=True)
            ax.set_ylabel('Score')
            ax.set_xlabel('Network')
            if titles is not None:
                ax.set_title(titles[i])
            if tags is not None:
                ax.set_xticks(range(1, len(tags) + 1))
                ax.set_xticklabels(tags)

    @staticmethod
    def boxplot_scores_group(scores_stack, tags=None, title=None, xlabel=''):
        """plots scores as a box plot for a set"""
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        ax.boxplot(scores_stack, showfliers=True)
        ax.set_ylabel('Score')
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        if tags is not None:
            ax.set_xticks(range(1, len(tags) + 1))
            ax.set_xticklabels(tags)

    @staticmethod
    def boxplot_scores_single(scores):
        """plots scores as a box plot"""
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        ax.boxplot(scores, showfliers=True)
        ax.set_ylabel('Score')
        ax.set_title('Best scores distribution')
        ax.set_xticklabels([])

    @staticmethod
    def boxplot_params(best_params, priors=None, samples=None):
        """plots the results of grid search"""
        # TODO: check the inputs: samples should have the same keys as priors, best_params

        if priors is not None and samples is not None:
            priors = {key: list(set([item[key] for item in priors])) for key in priors.keys()}
            samples = {key: list(set([item[key] for item in samples])) for key in samples.keys()}

        def normalize(xx, priors):
            xx = {key: np.array(list(set(values))) for key, values in xx.items()}
            if priors is not None:
                return {key: (values - min(priors[key])) / (max(priors[key]) - min(priors[key])) for key, values in
                        xx.items()}
            else:
                return {key: (values - min(values)) / (max(values) - min(values)) for key, values in xx.items()}

        # sort and normalize
        sorted_best_params = {key: np.array([item[key] for item in best_params]) for key in best_params[0].keys()}
        sorted_best_params_n = normalize(sorted_best_params, priors)

        # plot best params as box plot
        fig, axs = plt.subplots(1, 2, tight_layout=True, figsize=(10, 5), gridspec_kw={'width_ratios': [3, 1]})
        axs[0].boxplot(sorted_best_params_n.values(), showfliers=True, labels=sorted_best_params_n.keys())
        axs[0].set_ylabel('Normalized quantity')
        axs[0].set_title('Estimated params stats')
        # plot samples as scatter 
        if samples is not None:
            samples_n = normalize(samples, priors)
            for i, (key, values) in enumerate(samples_n.items(), 1):
                axs[0].scatter([i for j in range(len(values))], values)

    @staticmethod
    def barplot_PR_group(PR_stack, tags=None, title=None, xlabel=''):
        """Plot the values of PR for each network"""
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        ax.bar(range(1, len(PR_stack) + 1), PR_stack)
        ax.set_ylabel('Score')
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        if tags is not None:
            ax.set_xticks(range(1, len(tags) + 1))
            ax.set_xticklabels(tags)

    @staticmethod
    def barplot_PR_groupOfgroup(PR_stack_stack, tags=None, titles=None, xlabel='Network'):
        """plots PR as a bar plot for a group of groups (e.g. size*network)"""
        n = len(PR_stack_stack)
        fig, axes = plt.subplots(1, n, tight_layout=True, figsize=(n * 5, 4))
        for i, PR_stack in enumerate(PR_stack_stack):
            if n == 1:
                ax = axes
            else:
                ax = axes[i]
            ax.bar(range(1, len(PR_stack) + 1), PR_stack)
            ax.set_ylabel('Score')
            ax.set_xlabel(xlabel)
            if titles is not None:
                ax.set_title(titles[i])
            if tags is not None:
                ax.set_xticks(range(1, len(tags) + 1))
                ax.set_xticklabels(tags)

    @staticmethod
    def calculate_auc_roc(gene_names, links, golden_links, details=True, regulator_tag='Regulator', target_tag='Target',
                     weight_tag='Weight') -> float:
        # break the array into n parts, one for each gene
        scores = np.array(np.split(np.array(links[weight_tag].tolist()), len(gene_names)))  # links
        tests = np.array(np.split(np.array(golden_links[weight_tag].tolist()), len(gene_names)))  # golden
        return metrics.roc_auc_score(tests, scores)

    @staticmethod
    def calculate_PR(gene_names, links, golden_links, details=True, regulator_tag='Regulator', target_tag='Target',
                     weight_tag='Weight'):
        """ Compute precision recall 
     
        links -- sorted links as G1->G2, in a df format
        golden_links -- sorted golden links as G1->G2, in a df format
        details -- if True, detailed precision recall is calculated. If false, only the average for overall network is outputed
     
        """
        # break the array into n parts, one for each gene
        scores = np.array(np.split(np.array(links[weight_tag].tolist()), len(gene_names)))  # links
        tests = np.array(np.split(np.array(golden_links[weight_tag].tolist()), len(gene_names)))  # golden

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            precision = dict()
            recall = dict()
            average_precision = dict()
            if details:
                for gene, score, test in zip(gene_names, scores, tests):
                    precision[gene], recall[gene], _ = metrics.precision_recall_curve(test, score)
                    average_precision[gene] = metrics.average_precision_score(test, score)

                precision["micro"], recall["micro"], _ = metrics.precision_recall_curve(
                    tests.ravel(), scores.ravel()
                )
            average_precision['micro'] = metrics.average_precision_score(tests, scores, average="micro")
        return precision, recall, average_precision, average_precision['micro']

    @staticmethod
    def PR_curve_gene(gene_names, recall, precision, average_precision):
        """ Plots PR curve for the given genes as well as the average PR combining all genes """
        colors = itertools.cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])

        _, ax = plt.subplots(figsize=(7, 8))

        f_scores = np.linspace(0.2, 0.8, num=4)
        lines, labels = [], []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
            plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

        display = metrics.PrecisionRecallDisplay(
            recall=recall["micro"],
            precision=precision["micro"],
            average_precision=average_precision["micro"],
        )
        display.plot(ax=ax, name="Micro-average precision-recall", color="gold")

        for gene, color in zip(gene_names, colors):
            display = metrics.PrecisionRecallDisplay(
                recall=recall[gene],
                precision=precision[gene],
                average_precision=average_precision[gene],
            )
            display.plot(ax=ax, name=f"Precision-recall for class {gene}", color=color)

        # add the legend for the iso-f1 curves
        handles, labels = display.ax_.get_legend_handles_labels()
        handles.extend([l])
        labels.extend(["iso-f1 curves"])
        # set the legend and the axes
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(handles=handles, labels=labels, loc="best")
        ax.set_title("Extension of Precision-Recall curve to multi-class")

        plt.show()

    @staticmethod
    def PR_curve_average(recall, precision, average_precision):
        """ Plots average precison recall curve """
        display = metrics.PrecisionRecallDisplay(
            recall=recall["micro"],
            precision=precision["micro"],
            average_precision=average_precision["micro"],
        )
        # print(precision["micro"])
        display.plot()
        _ = display.ax_.set_title("Micro-averaged over all classes")

        def param_unique_average(param_unique):
            average_param_unique = {key: int(np.mean([gene_param[key] for gene_param in param_unique])) for key in
                                    param_unique[0].keys()}
            return average_param_unique
