from typing import Tuple, Optional, Generator, List, Union

import numpy as np
from scipy.sparse import coo_matrix
from sklearn import utils

from geneRNI.utils import create_tf_mask, create_ko_mask


class Data:
    """Iterable data class.

    Iterating over a `Data` object will generate pairs of arrays `(X, y)` in the
    scikit-learn fashion. Each pair `(X, y)` is composed of a target gene, whose values
    are stored in `y`, and a set of putative regulators for that target gene stored
    in `X` column-wise.

    Args:
        gene_names: An arbitrary list of unique gene names.
        ss_data: An array of shape `(n_samples, n_genes)`, where `n_samples` and `n_genes`
            are the number of observations and total number of genes in the network, respectively.
        ts_data: A list of arrays of various shapes representing observations over time.
            More specifically, the shape of `ts_data[i]` is `(n_samples[i], n_genes)`,
            where `n_samples[i]` is the number of observations collected during experiment `i`,
            and `n_genes` is the total number of genes in the network. The number of genes is
            expected to be the same across experiments.
        time_points: Should be provided if `ts_data is not None`. A list of 1-dimensional arrays
            of same length as `ts_data`, where each array gives the time points of the different
            observations in `ts_data`. More specifically, `len(time_points) == len(ts_data)`
            and `ts_data[i].shape == (len(ts_data[i]),)`.
        regulators: List of putative regulators. If "all", then all genes from the network will
            be considered as candidates. If `regulators` is a list of strings, then each string element
            will be considered as candidate for any other gene in the network.
            If `regulators` is a list of lists of strings, then `len(regulators)` should be equal to the
            total number of genes in the network, and each sublist `regulators[j]` contains the
            candidates for `gene_names[j]`
        perturbations: Protein concentrations added to the experiment.
            An array of shape `(n_samples, n_genes)` identical to the shape of `ss_data`.
        ko: Nested list of knocked out genes. All genes that have been knocked out in static
            sample `ss_data[i]` should be listed in sublist `ko[i]`.
            `len(ko)` should be equal to `len(ss_data)`.
        h: Lag used for the finite approximation of the derivative of the target gene expression.
        verbose: Whether to print extra debug messages.
    """

    def __init__(
            self,
            gene_names: List[str],
            ss_data: Optional[np.ndarray] = None,
            ts_data: Optional[List[np.ndarray]] = None,
            time_points: Optional[List[np.ndarray]] = None,
            regulators: Union[str, List[str], List[List[str]]] = 'all',
            perturbations: Optional[np.ndarray] = None,
            ko: Optional[List[List[str]]] = None,
            h: int = 1,
            verbose: bool = True,
            **specs
    ):
        self.gene_names: List[str] = list(gene_names)

        # Boolean matrix of shape (n_genes, n_genes), where `is_regulator[i, j]`
        # indicates where gene `i` is a putative regulator for target gene `j`.
        self.is_regulator: np.ndarray = create_tf_mask(gene_names, regulators)

        # Boolean matrix of shape (n_samples, n_genes), identical to the shape
        # of `ss_data`, where `is_ok` indicates whether gene `j` has been
        # knocked out in experiment/sample `i`.
        if ss_data is not None:
            self.is_ko: Optional[np.ndarray] = create_ko_mask(gene_names, ko, len(ss_data))
            assert len(self.is_ko) == len(ss_data)
        else:
            self.is_ko: Optional[np.ndarray] = None

        self.ss_data: Optional[np.ndarray] = ss_data
        self.ts_data: Optional[List[np.ndarray]] = ts_data
        self.time_points: Optional[List[np.ndarray]] = time_points
        self.perturbations: Optional[np.ndarray] = perturbations
        self.n_genes: int = len(self.gene_names)
        self.h: int = int(h)

        self.verbose: bool = verbose
        self.specs = specs

        # Re-order time points in increasing order
        if self.time_points is not None:
            for (i, tp) in enumerate(self.time_points):
                tp = np.array(tp, np.float32)
                indices = np.argsort(tp)
                time_points[i] = tp[indices]
                self.ts_data[i] = self.ts_data[i][indices, :]

    def process_time_series(self, input_idx: np.ndarray, target_gene: int, h: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """ Reformat data for time series analysis

        """

        nexp = len(self.ts_data)
        nsamples_time = sum([expr_data.shape[0] for expr_data in self.ts_data])
        ninputs = len(input_idx)

        # Time-series data
        X = np.zeros((nsamples_time - h * nexp, ninputs))
        y = []
        offset = 0
        for (i, exp_timeseries) in enumerate(self.ts_data):
            exp_time_points = self.time_points[i]
            assert len(exp_timeseries) == len(exp_time_points)
            n_time = exp_timeseries.shape[0]

            # TODO: computationally slow
            for ii in range(n_time - h):
                y1 = self.ts_data[i][ii, target_gene]
                y2 = self.ts_data[i][ii + h, target_gene]
                t1 = exp_time_points[ii]
                t2 = exp_time_points[ii + h]
                der = (y2 - y1) / (t2 - t1)
                y.append(lambda decay_coef, der=der, y1=y1: float(der + decay_coef * y1))

            X[offset:offset+n_time-h, :] = exp_timeseries[:n_time-h, input_idx]
            offset += n_time - h
        y = np.asarray(y)
        return X, y

    def process_static(self, input_idx: np.ndarray, target_gene: int) -> Tuple[np.ndarray, np.ndarray]:
        """ Reformat data for static analysis
        SS_data -- static data in the format n_samples*n_genes
        perturbations -- initial changes to the genes such as adding certain values. n_samples*n_genes
        KO -- the list of knock-out gene names. For now, each row has 1 gene name. TODO: one gene for all samples; more than one genes for one sample
        """

        X = self.ss_data[:, input_idx]
        if self.perturbations is not None:
            y = self.ss_data[:, target_gene] - self.perturbations[:, target_gene]
        else:
            y = self.ss_data[:, target_gene]
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

    def __getitem__(self, i_gene: int) -> Tuple[np.ndarray, np.ndarray]:
        """ Reformats the raw data for both static and dynamic analysis

        For time series data, TS_data should be in n_exp*n_time*n_genes format. For For static analysis,
        SS_data should be in n_samples * n_genes.
        """

        # TODO: add KO to time series data

        # Determine list of regulators
        # TODO: Not sure I understand what was supposed to be done here
        # try:
        #    input_idx.remove(self.ko_indices[i_gene])
        # except UnboundLocalError:
        #    pass
        input_idx = np.where(self.is_regulator[:, i_gene])[0]
        X, y = [], []
        if self.ts_data is not None:
            X_d, y_d = self.process_time_series(input_idx, i_gene, h=self.h)
            X.append(X_d)
            y.append(y_d)
        if self.ss_data is not None:
            X_s, y_s = self.process_static(input_idx, i_gene)
            X.append(X_s)
            y.append(y_s)

        # Combine static and dynamic data
        if (self.ts_data is None) and (self.ss_data is None):
            raise ValueError('Static and dynamic data are both None')
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)

        return X, y

    def __iter__(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __len__(self) -> int:
        return self.n_genes
