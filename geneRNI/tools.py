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
from typing import Tuple, List

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
from sklearn.preprocessing import StandardScaler, PowerTransformer

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

    def __init__(
            self,
            gene_names,
            ss_data,
            ts_data,
            time_points,
            regulators='all',
            perturbations=None,
            KO=None,
            test_size: float = 0.25,
            h: int = 1,
            random_state = None,
            verbose: bool = True,
            **specs
    ):
        self.gene_names = gene_names
        self.ss_data = ss_data
        self.ts_data = ts_data
        self.time_points = time_points

        # Lag used for the finite approximation of the derivative of the target gene expression
        self.h: int = int(h)

        self.test_size: float = test_size
        self.verbose: bool = verbose
        self.random_state = random_state
        self.specs = specs
        self.KO = KO
        self.regulators = regulators
        self.perturbations = perturbations
        self.n_genes = len(self.gene_names)

        # Knock-outs
        self.ko_indices = []
        if self.KO is not None:
            for gene in self.KO:
                self.ko_indices.append(gene_names.index(gene))

        # Re-order time points in increasing order
        for (i, tp) in enumerate(self.time_points):
            tp = np.array(tp, np.float32)
            indices = np.argsort(tp)
            time_points[i] = tp[indices]
            expr_data = self.ts_data[i]
            self.ts_data[i] = expr_data[indices, :]

    def process_time_series(self, i_gene: int, h: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """ Reformat data for time series analysis 
        
        """

        if self.regulators == 'all':
            input_idx = list(range(self.n_genes))
        else:
            input_idx = self.regulators[i_gene]

        # TODO: Not sure I understand what was supposed to be done here
        #try:
        #    input_idx.remove(self.ko_indices[i_gene])
        #except UnboundLocalError:
        #    pass

        nexp = len(self.ts_data)
        nsamples_time = sum([expr_data.shape[0] for expr_data in self.ts_data])
        ninputs = len(input_idx)

        # Time-series data
        X = np.zeros((nsamples_time - h * nexp, ninputs))
        y = []
        for (i, exp_timeseries) in enumerate(self.ts_data):
            exp_time_points = self.time_points[i]
            n_time = exp_timeseries.shape[0]
            exp_time_diff = exp_time_points[h:] - exp_time_points[:n_time - h]
            exp_timeseries_x = exp_timeseries[:n_time - h, input_idx]
            # current_timeseries_output = (exp_timeseries[h:,i_gene] - exp_timeseries[:n_time-h,i_gene]) / exp_time_diff + alphas[i_gene]*exp_timeseries[:n_time-h,i_gene]
            for ii in range(len(exp_time_diff)):
                f_dy_dt = lambda alpha_i, i=i, ii=ii, i_gene=i_gene: float(
                    (self.ts_data[i][ii + 1:ii + 2, i_gene] - self.ts_data[i][ii:ii + 1, i_gene]) / exp_time_diff[
                        ii] + alpha_i * self.ts_data[i][ii:ii + 1, i_gene])
                y.append(f_dy_dt)

            exp_n_samples = exp_timeseries_x.shape[0]
            X[i * exp_n_samples:(i + 1) * exp_n_samples, :] = exp_timeseries_x
        y = np.asarray(y)
        return X, y

    def process_static(self, i_gene: int) -> Tuple[np.ndarray, np.ndarray]:
        """ Reformat data for static analysis 
        SS_data -- static data in the format n_samples*n_genes
        perturbations -- initial changes to the genes such as adding certain values. n_samples*n_genes
        KO -- the list of knock-out gene names. For now, each row has 1 gene name. TODO: one gene for all samples; more than one genes for one sample
        """

        if self.regulators == 'all':
            input_idx = list(range(self.n_genes))
        else:
            input_idx = self.regulators[i_gene]

        # TODO: Not sure I understand what was supposed to be done here
        #try:
        #    input_idx.remove(self.ko_indices[i_gene])
        #except UnboundLocalError:
        #    pass

        X = self.ss_data[:, input_idx]

        y = []
        for i_sample, sample_data in enumerate(self.ss_data):
            # add perturbations
            if self.perturbations is not None:
                f_dy_dt = float(self.ss_data[i_sample][i_gene]) - float(self.perturbations[i_sample][i_gene])
            else:
                f_dy_dt = float(self.ss_data[i_sample][i_gene])
            y.append(f_dy_dt)
        y = np.asarray(y)
        return X, y

    def __getitem__(self, i_gene: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Reformats the raw data for both static and dynamic analysis

        For time series data, TS_data should be in n_exp*n_time*n_genes format. For For static analysis, 
        SS_data should be in n_samples * n_genes.
        """
        # Check input arguments
        # TODO: useless
        dynamic_flag = False
        static_flag = False
        if self.ts_data is not None:
            dynamic_flag = True
        if self.ts_data is not None:
            static_flag = True
        # if dynamic_flag and not isinstance(TS_data,(list,tuple)):
        #     raise ValueError('TS_data must be a list of lists')
        # if static_flag and not isinstance(SS_data,(list,tuple)):
        #     raise ValueError('SS_data must be a list of list')

        # TODO: check the inputs

        # TODO: add KO to time series data

        X, y = [], []
        if self.ts_data is not None:
            X_d, y_d = self.process_time_series(i_gene, h=self.h)
            #verboseprint(
            #    self.verbose,
            #    f'dynamic data: ngenes: {len(ys_d)}, nsamples: {len(ys_d[0])}, n regulators: {len(Xs_d[0][0])}'
            #)
            X.append(X_d)
            y.append(y_d)
        if self.ss_data is not None:
            X_s, y_s = self.process_static(i_gene)
            #verboseprint(
            #    self.verbose,
            #    f'static data: ngenes: {len(ys_s)}, nsamples: {len(ys_s[0])}, n regulators: {len(Xs_s[0][0])}'
            #)
            X.append(X_s)
            y.append(y_s)

        # Combine static and dynamic data
        if (self.ts_data is None) and (self.ss_data is None):
            raise ValueError('Static and dynamic data are both None')
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)

        # Split data in train/validation sets
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)

        # TODO: Resample?
        # X_train, y_train = Data.resample(X_train, y_train, n_samples = bootstrap_fold*len(y), random_state=self.random_state)
        # X_test, y_test = Data.resample(X_test, y_test, n_samples = bootstrap_fold*len(y), random_state=self.random_state)

        # Pre-processing
        transformer = PowerTransformer(method='box-cox', standardize=True, copy=False)
        transformer.fit_transform(X_train + 1e-15)
        transformer.transform(X_test + 1e-15)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def resample(X, y, n_samples, replace=True, **specs) -> Tuple[np.ndarray, np.ndarray]:
        """resampling for bootstraping"""
        if n_samples is None:
            n_samples = 2 * len(y)

        X_sparse = sparse.coo_matrix(X)
        X_b, _, y_b = utils.resample(X, X_sparse, y, n_samples=n_samples, replace=replace, **specs)
        # XX = utils.resample((X,y), n_samples = n_samples, replace=replace)
        # print(len(XX[0]))
        return X_b, y_b

    def __len__(self) -> int:
        return self.n_genes


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
    def process_data(ts_data, ss_data, time_points, gene_names, estimator_t, **specs) -> Data:
        pp = Settings.default(estimator_t)
        return Data(
            gene_names,
            ss_data,
            ts_data,
            time_points,
            test_size=pp.test_size,
            random_state=pp.random_state_data,
            **specs
        )

    @staticmethod
    def process_data_dream4(size, network, estimator_t: str, **specs) -> Data:
        ts_data, time_points, ss_data, gene_names = Benchmark.f_data_dream4(size, network)
        return Benchmark.process_data(ts_data, ss_data, time_points, gene_names, estimator_t, **specs)

    @staticmethod
    def process_data_grn_benchmark(method, noise_level, network, estimator_t: str, **specs) -> Data:
        ss_data, _, gene_names = Benchmark.f_data_GRN(method, noise_level, network)
        return Benchmark.process_data(None, ss_data, None, gene_names, estimator_t)


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
    def to_matrix(df: pd.DataFrame, gene_names: List[str]) -> np.ndarray:
        mapping = {gene_name: i for i, gene_name in enumerate(gene_names)}
        mat = np.full((len(gene_names), len(gene_names)), np.nan, dtype=float)
        idx_i = np.asarray([mapping[x] for x in df['Regulator'].tolist()], dtype=int)
        idx_j = np.asarray([mapping[x] for x in df['Target'].tolist()], dtype=int)
        weights = df['Weight'].to_numpy(dtype=float)
        mat[idx_i, idx_j] = weights
        return mat

    @staticmethod
    def calculate_auc_roc(gene_names: List[str], links: pd.DataFrame, golden_links: pd.DataFrame) -> float:
        scores = GOF.to_matrix(links, gene_names)
        tests = GOF.to_matrix(golden_links, gene_names)
        mask = ~np.isnan(tests)
        scores, tests = scores[mask], tests[mask]
        return metrics.roc_auc_score(tests, scores)

    @staticmethod
    def calculate_PR(gene_names: List[str], links: pd.DataFrame, golden_links: pd.DataFrame) -> float:
        """ Compute precision recall 
     
        links -- sorted links as G1->G2, in a df format
        golden_links -- sorted golden links as G1->G2, in a df format
        """
        scores = GOF.to_matrix(links, gene_names)
        tests = GOF.to_matrix(golden_links, gene_names)
        mask = ~np.isnan(tests)
        scores, tests = scores[mask], tests[mask]
        return metrics.average_precision_score(tests, scores)

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
