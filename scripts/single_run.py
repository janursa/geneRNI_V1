import os
import pathlib
import sys
import time

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pd.options.mode.chained_assignment = None
dir_main = os.path.join(pathlib.Path(__file__).parent.resolve(), '..')
sys.path.insert(0, dir_main)

from geneRNI.core import network_inference
from geneRNI import benchmarks, utils, evaluation


def dream4(network, size, estimator_t):
	_, _, _, gene_names = benchmarks.Benchmark.f_data_dream4(size, network)
	out_data = benchmarks.Benchmark.process_data_dream4(size=size, network=network, estimator_t=estimator_t)
	return network, size, gene_names, out_data


def grn_benchmark(method, noise_level, network, estimator_t):
	_, _, gene_names = benchmarks.Benchmark.f_data_GRN(method, noise_level, network)
	out_data = benchmarks.Benchmark.process_data_grn_benchmark(method, noise_level, network, estimator_t=estimator_t)
	return gene_names, out_data
import random
def create_random_links(links, n=50):
	weights = links['Weight'].to_numpy(float)
	sample_links =  links.copy()
	random_links = pd.DataFrame({key:sample_links[key] for key in ['Regulator', 'Target']})
	weightpoolvector = []
	for i in range(len(weights)):
		weightpoolvector.append(random.sample(list(weights), n))
	random_links['Weight']= np.mean(weightpoolvector, axis=1)
	random_links_pool = random_links.copy()
	random_links_pool['WeightPool'] = weightpoolvector
	return random_links, random_links_pool

if __name__ == '__main__':
	estimator_t = 'RF'  # 'HGB'
	out_default = utils.default_settings(estimator_t)

	# choose one of the followings
	network, size, gene_names, out_data = dream4(5, 10, estimator_t)
	# gene_names, out_data = grn_benchmark('GeneNetWeaver', 'LowNoise', 'Network1', estimator_t)
	# tun
	start = time.time()
	_, train_scores, links, oob_scores, test_scores = network_inference(
		out_data, gene_names=gene_names, param=out_default.param)
	if True:  # scores on the golden links
		golden_links = benchmarks.Benchmark.f_golden_dream4(size, network)
		# precision, recall, average_precision, average_precision_overall = evaluation.calculate_PR(
		# 	links, golden_links)
		precision, recall, thresholds = evaluation.precision_recall_curve(links, golden_links)
		def normalize(aa):
			return (aa-min(aa))/(max(aa)-min(aa))
		plt.plot(normalize(thresholds), recall[:-1], label='trained')
		print(np.mean(recall[:-1]))

		links_random,_ = create_random_links(links)
		precision, recall, thresholds = evaluation.precision_recall_curve(links_random, golden_links)

		plt.plot(normalize(thresholds), recall[:-1], label='random')
		print(np.mean(recall[:-1]))
		plt.legend()
		plt.show()
		# print('Mean PR is ', average_precision_overall)
	end = time.time()
	print('lapsed time ', (end-start))
