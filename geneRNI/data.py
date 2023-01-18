from typing import Tuple, Optional

import numpy as np
from scipy.sparse import coo_matrix
from sklearn import model_selection
from sklearn import utils
from sklearn.preprocessing import PowerTransformer


class Data:

    def __init__(
            self,
            gene_names,
            ss_data,
            ts_data,
            time_points: Optional[list],
            regulators='all',
            perturbations=None,
            KO=None,
            h: int = 1,
            verbose: bool = True,
            **specs
    ):
        self.gene_names = gene_names
        self.ss_data = ss_data
        self.ts_data = ts_data
        self.time_points = time_points

        # Lag used for the finite approximation of the derivative of the target gene expression
        self.h: int = int(h)

        self.verbose: bool = verbose
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
        if self.time_points is not None:
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
        # try:
        #    input_idx.remove(self.ko_indices[i_gene])
        # except UnboundLocalError:
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
            # current_timeseries_output = (exp_timeseries[h:,i_gene] - exp_timeseries[:n_time-h,i_gene]) / exp_time_diff + decay_coeffs[i_gene]*exp_timeseries[:n_time-h,i_gene]
            for ii in range(len(exp_time_diff)):
                f_dy_dt = lambda decay_coeff_i, i=i, ii=ii, i_gene=i_gene: float(
                    (self.ts_data[i][ii + 1:ii + 2, i_gene] - self.ts_data[i][ii:ii + 1, i_gene]) / exp_time_diff[
                        ii] + decay_coeff_i * self.ts_data[i][ii:ii + 1, i_gene])
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
        # try:
        #    input_idx.remove(self.ko_indices[i_gene])
        # except UnboundLocalError:
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
            # verboseprint(
            #    self.verbose,
            #    f'dynamic data: ngenes: {len(ys_d)}, nsamples: {len(ys_d[0])}, n regulators: {len(Xs_d[0][0])}'
            # )
            X.append(X_d)
            y.append(y_d)
        if self.ss_data is not None:
            X_s, y_s = self.process_static(i_gene)
            # verboseprint(
            #    self.verbose,
            #    f'static data: ngenes: {len(ys_s)}, nsamples: {len(ys_s[0])}, n regulators: {len(Xs_s[0][0])}'
            # )
            X.append(X_s)
            y.append(y_s)

        # Combine static and dynamic data
        if (self.ts_data is None) and (self.ss_data is None):
            raise ValueError('Static and dynamic data are both None')
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)

        # Pre-processing (mostly for non-tree-based models)
        # transformer = PowerTransformer(method='box-cox', standardize=True, copy=False)
        # transformer.fit_transform(X_train + 1e-15)
        # transformer.transform(X_test + 1e-15)

        return X, y

    @staticmethod
    def resample(X, y, n_samples, replace=True, **specs) -> Tuple[np.ndarray, np.ndarray]:
        """resampling for bootstraping"""
        if n_samples is None:
            n_samples = 2 * len(y)

        X_sparse = coo_matrix(X)
        X_b, _, y_b = utils.resample(X, X_sparse, y, n_samples=n_samples, replace=replace, **specs)
        # XX = utils.resample((X,y), n_samples = n_samples, replace=replace)
        # print(len(XX[0]))
        return X_b, y_b

    def __len__(self) -> int:
        return self.n_genes