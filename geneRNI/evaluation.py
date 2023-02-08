import itertools
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from typing import Tuple, List



"""Goodness of fit"""


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

def boxplot_scores_single(scores):
    """plots scores as a box plot"""
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    ax.boxplot(scores, showfliers=True)
    ax.set_ylabel('Score')
    ax.set_title('Best scores distribution')
    ax.set_xticklabels([])

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

def to_matrix(df: pd.DataFrame) -> np.ndarray:
    
    return df.pivot(index='Regulator', columns='Target', values='Weight').fillna(0).values

def calculate_auc_roc(links: pd.DataFrame, golden_links: pd.DataFrame) -> float:
    scores = to_matrix(links)
    tests = to_matrix(golden_links)
    mask = ~np.isnan(tests)
    scores, tests = scores[mask], tests[mask]
    return metrics.roc_auc_score(tests, scores)



def calculate_PR(links: pd.DataFrame, golden_links: pd.DataFrame) -> float:
    """ Compute precision recall

    links -- sorted links as G1->G2, in a df format
    golden_links -- sorted golden links as G1->G2, in a df format
    """
    scores = to_matrix(links)
    tests = to_matrix(golden_links)
    mask = ~np.isnan(tests)
    scores, tests = scores[mask], tests[mask]
    return metrics.average_precision_score(tests, scores)
def precision_recall_curve(links: pd.DataFrame, golden_links: pd.DataFrame) -> Tuple[List, List, List]:
    """ Compute precision recall array with differnent thresholds

    links -- sorted links as G1->G2, in a df format
    golden_links -- sorted golden links as G1->G2, in a df format

    outputs: 
        precision: array
        recall: array
        threshold: array
    """
    links.to_csv('test.csv')
    scores = to_matrix(links)
    tests =  to_matrix(golden_links)
    mask = ~np.isnan(tests)
    scores, tests = scores[mask], tests[mask]
    return metrics.precision_recall_curve(tests, scores)
def roc_curve(links: pd.DataFrame, golden_links: pd.DataFrame) -> Tuple[List, List, List]:
    """ Compute Receiver operating characteristic (ROC).

    links -- sorted links as G1->G2, in a df format
    golden_links -- sorted golden links as G1->G2, in a df format

    outputs: 
        fdr: array
        tpr: array
        threshold: array
    """
    links.to_csv('test.csv')
    scores = to_matrix(links)
    tests =  to_matrix(golden_links)
    mask = ~np.isnan(tests)
    scores, tests = scores[mask], tests[mask]
    return metrics.roc_curve(tests, scores)

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

    # TODO
