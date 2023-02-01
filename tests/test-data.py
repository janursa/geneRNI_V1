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

from typing import Tuple

import numpy as np
from sklearn.feature_selection import r_regression

from geneRNI.data import Data
from geneRNI.utils import preprocess_target


def create_data_set(n_genes: int, n_static: int, n_dynamic: int) -> Data:
    gene_names = [f'gene-{i + 1}' for i in range(n_genes)]
    if n_static == 0:
        ss_data = None
    else:
        ss_data = np.random.rand(n_static, n_genes)
    if n_dynamic == 0:
        ts_data, time_points = None, None
    else:
        series_lengths = []
        n_remaining = n_dynamic
        while n_remaining > 0:
            series_length = min(n_remaining, np.random.randint(2, 10))
            if n_remaining - series_length == 1:
                series_length += 1
            n_remaining -= series_length
            series_lengths.append(series_length)
        assert sum(series_lengths) == n_dynamic
        ts_data, time_points = [], []
        for series_length in series_lengths:
            assert series_length > 1
            ts_data.append(np.random.rand(series_length, n_genes))
            time_points.append(np.sort(np.random.rand(series_length)))
    data = Data(
        gene_names, ss_data=ss_data, ts_data=ts_data, time_points=time_points,
        regulators='all', perturbations=None, ko=None,  # TODO: test perturbations, regulators and KO
        h=1, verbose=False
    )
    return data


def is_data_contaminated(X: np.ndarray, y: np.ndarray, threshold: float = 0.95) -> bool:
    scores = np.abs(r_regression(X, y))
    return np.max(scores) >= threshold


def expected_shape(data: Data) -> Tuple[int, int]:
    n_genes = data.n_genes
    n_samples = 0
    if data.ss_data is not None:
        n_samples += len(data.ss_data)
    if data.ts_data is not None:
        for time_series in data.ts_data:
            n_samples += len(time_series) - 1
    return n_samples, n_genes - 1


def __test_data(n_genes: int = 0, n_static: int = 0, n_dynamic: int = 0):
    data = create_data_set(n_genes, n_static, n_dynamic)
    assert len(data) == n_genes
    if data.ss_data is not None:
        assert is_data_contaminated(data.ss_data, data.ss_data[:, 0])

    for X, y in data:
        y = preprocess_target(y, 0.01)
        assert X.shape == expected_shape(data)
        assert not is_data_contaminated(X, y)


def test_static():
    __test_data(n_genes=20, n_static=10, n_dynamic=0)


def test_dynamic():
    __test_data(n_genes=20, n_static=0, n_dynamic=17)
