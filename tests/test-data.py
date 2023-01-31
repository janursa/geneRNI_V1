# -*- coding: utf-8 -*-
#
#  test-data.py
#
#  Copyright 2023 Antoine Passemiers <antoine.passemiers@gmail.com>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.

import numpy as np
from sklearn.feature_selection import r_regression

from geneRNI.data import Data


def create_data_set(n_genes: int, n_static: int, n_dynamic: int) -> Data:
    gene_names = [f'gene-{i + 1}' for i in range(n_genes)]
    ss_data = np.random.rand(n_static, n_genes)
    if n_dynamic == 0:
        ts_data, time_points = None, None
    else:
        ts_data = np.random.rand(n_dynamic, n_genes)
        time_points = np.sort(np.random.rand(n_dynamic))
    data = Data(
        gene_names, ss_data, ts_data, time_points,
        regulators='all', perturbations=None, KO=None,  # TODO: test perturbations, regulators and KO
        h=1, verbose=False
    )
    return data


def is_data_contaminated(X: np.ndarray, y: np.ndarray, threshold: float = 0.9) -> bool:
    scores = np.abs(r_regression(X, y))
    return np.max(scores) >= threshold


def __test_data(n_genes: int = 0, n_static: int = 0, n_dynamic: int = 0):
    n_samples = n_static + n_dynamic
    data = create_data_set(n_genes, n_static, n_dynamic)
    assert len(data) == n_genes
    assert is_data_contaminated(data.ss_data, data.ss_data[:, 0])

    for X, y in data:
        assert X.shape == (n_samples, n_genes - 1)
        assert not is_data_contaminated(X, y)


def test_static():
    __test_data(n_genes=20, n_static=10, n_dynamic=0)
