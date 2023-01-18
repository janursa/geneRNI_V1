import time
import os 
import pathlib
import sys
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None
dir_main = os.path.join(pathlib.Path(__file__).parent.resolve(), '..')
sys.path.insert(0, dir_main)
import importlib
import inspect

from geneRNI import geneRNI as ni
from geneRNI import benchmarks, utils, evaluation

def dream4(network, size, estimator_t):
	_, _, _, gene_names = benchmarks.f_data_dream(size, network)
	out_data = benchmarks.process_data_dream4(size=size, network=network, estimator_t=estimator_t)
	return network, size, gene_names, out_data


def grn_benchmark(method, noise_level, network, estimator_t):
	_, _, gene_names = benchmarks.f_data_GRN(method, noise_level, network)
	out_data = benchmarks.process_data_grn_benchmark(method, noise_level, network, estimator_t=estimator_t)
	return gene_names, out_data


if __name__ == '__main__':
	estimator_t = 'RF'  # 'HGB'
	out_default = utils.default_settings(estimator_t)

	# choose one of the followings
	network, size, gene_names, out_data = dream4(1, 10, estimator_t)
	# gene_names, out_data = grn_benchmark('GeneNetWeaver', 'LowNoise', 'Network1', estimator_t)
	# tun
	start = time.time()
	_, train_scores, links, oob_scores, test_scores = ni.network_inference(
		out_data, gene_names=gene_names, param=out_default.param)
	if True:  # scores on the golden links
		golden_links = benchmarks.f_golden_dream(size, network)
		precision, recall, average_precision, average_precision_overall = evaluation.calculate_PR(
			gene_names, links, golden_links, details=False)
		print('Mean PR is ', average_precision_overall)
	end = time.time()
	print('lapsed time ', (end-start))
