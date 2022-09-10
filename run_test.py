import time
import sys
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

import importlib
import run_hyper
import inspect

from geneRNI import tools
from geneRNI import geneRNI as ni

if __name__ == '__main__':
	estimator_t = 'RF' #'HGB'
	out_default = tools.Settings.default(estimator_t)

	study = 'dream4' #'GRNbenchmark'

	if study == 'dream4': # dream as target study
	    size, network = 10, 1 # [10,100] [1-5]
	    _, _, _, gene_names = tools.Benchmark.f_data_dream(size, network)
	    out_data = tools.Benchmark.process_data_dream4(size=size, network=network, estimator_t=estimator_t)
	elif study == 'GRNbenchmark':
	    method, noise_level, network = 'GeneNetWeaver', 'LowNoise', 'Network1'
	    out_data = tools.Benchmark.process_data_GRNbenchmark(method, noise_level, network, estimator_t=estimator_t)
	else:
	    raise ValueError('Define')
	start = time.time()
	_, train_scores, links, oob_scores, test_scores = ni.network_inference(Xs=out_data.Xs_train, ys=out_data.ys_train, gene_names=gene_names,
																		   param=out_default.param, Xs_test=out_data.Xs_test, ys_test=out_data.ys_test)
	if study == 'dream4':
		golden_links = tools.Benchmark.f_golden_dream(size, network)
		precision, recall, average_precision, average_precision_overall = tools.GOF.calculate_PR(gene_names, golden_links,links, details=True)
		print('Mean PR is ', average_precision_overall)
	end = time.time()
	print('lapsed time ', (end-start))