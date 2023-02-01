# -*- coding: utf-8 -*-
#
#  utils.py
#
#  Copyright 2022 Antoine Passemiers <antoine.passemiers@gmail.com>
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

import types
from typing import Any, List, Union, Optional

import numpy as np
import sklearn

from geneRNI.types_ import DefaultParamType
from geneRNI.models import get_estimator_wrapper


class HashableDict(dict):

    def __hash__(self):
        return hash(tuple(sorted(self.items())))


def preprocess_target(y: np.ndarray, decay_coef: float) -> np.ndarray:
    new_y = np.empty(len(y), dtype=float)
    for i in range(len(y)):
        if is_lambda_function(y[i]):
            new_y[i] = float(y[i](decay_coef))
        else:
            new_y[i] = float(y[i])
    sklearn.utils.indexable(new_y)
    sklearn.utils.assert_all_finite(new_y)
    return new_y


def is_lambda_function(obj: Any) -> bool:
    return isinstance(obj, types.LambdaType) and (obj.__name__ == '<lambda>')


def verbose_print(flag: bool, message: str):
    if flag:
        print(message)


def default_settings(estimator_t: str) -> DefaultParamType:
    wrapper = get_estimator_wrapper(estimator_t)
    param = wrapper.get_default_parameters()
    param_grid = wrapper.get_grid_parameters()
    return DefaultParamType(param, param_grid)


def create_tf_mask(
        gene_names: List[str],
        regulators: Union[str, List[str], List[List[str]]]
) -> np.ndarray:
    n_genes = len(gene_names)
    is_regulator = np.zeros((n_genes, n_genes), dtype=bool)
    if len(gene_names) != len(set(gene_names)):
        raise ValueError('Gene names list contains duplicate entries')
    gene_dict = {gene_name: i for i, gene_name in enumerate(gene_names)}
    if isinstance(regulators, str):
        if regulators.lower().strip() == 'all':
            is_regulator[:, :] = True
        else:
            raise ValueError(f'Invalid value for regulators list: {regulators}')
    else:
        if not isinstance(regulators, list):
            raise ValueError('Regulators list should be a list object')
        if len(regulators) > 0:
            if isinstance(regulators[0], list):
                if len(regulators) != len(gene_names):
                    raise ValueError(
                        f'Regulators nested list should be as long as '
                        f'the number of genes in the network'
                    )
                for j in range(len(regulators)):
                    for gene_name in regulators[j]:
                        i = gene_dict[gene_name]
                        is_regulator[i, j] = True
            else:
                for gene_name in regulators:
                    i = gene_dict[gene_name]
                    is_regulator[i, :] = True
    np.fill_diagonal(is_regulator, False)
    return is_regulator


def create_ko_mask(
        gene_names: List[str],
        ko_gene_names: Optional[List[List[str]]],
        n_samples: int
) -> np.ndarray:
    n_genes = len(gene_names)
    is_ko = np.zeros((n_samples, n_genes), dtype=bool)
    if ko_gene_names is None:
        return is_ko
    if len(gene_names) != len(set(gene_names)):
        raise ValueError('Gene names list contains duplicate entries')
    if len(ko_gene_names) != n_samples:
        raise ValueError(f'Nested list of KO genes should have a length {n_samples}')
    gene_dict = {gene_name: i for i, gene_name in enumerate(gene_names)}
    for i in range(len(ko_gene_names)):
        for gene_name in ko_gene_names[i]:
            j = gene_dict[gene_name]
            is_ko[i, j] = True
    return is_ko
