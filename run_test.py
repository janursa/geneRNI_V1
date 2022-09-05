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

	study = 'dreams' #'GRNbenchmark'

	if study == 'dreams': # dream as target study
	    size, network = 10, 1 # [10,100] [1-5]
	    out_data = tools.Benchmark.process_data_dreams(size=size, network=network, estimator_t=estimator_t)
	elif study == 'GRNbenchmark':
	    method, noise_level, network = 'GeneNetWeaver', 'LowNoise', 'Network1'
	    out_data = tools.Benchmark.process_data_GRNbenchmark(method, noise_level, network, estimator_t=estimator_t)
	else:
	    raise ValueError('Define')

	out_network = ni.network_inference(Xs=out_data.Xs_train, ys=out_data.ys_train, param=out_default.param, Xs_test=out_data.Xs_test, ys_test=out_data.ys_test)
