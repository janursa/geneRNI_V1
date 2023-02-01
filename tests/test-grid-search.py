# -*- coding: utf-8 -*-
#
#  test-grid-search.py
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

from geneRNI.search_param import gen_permutations
from geneRNI.utils import HashableDict


def create_easy_dataset() -> Tuple[np.ndarray, np.ndarray]:
    X = np.random.rand(30, 6)
    y = np.random.rand(30)
    X[:, 3] = 0.1 * X[:, 3] + 0.9 * y
    return X, y


def test_single_permutation():
    param_grid = {
        'alpha': [1.0],
        'h': ['A'],
        'j': [5.6]
    }
    permutations = list(gen_permutations(param_grid))
    assert len(permutations) == 1
    assert permutations[0] == {'alpha': 1.0, 'h': 'A', 'j': 5.6}


def test_permutations():
    param_grid = {
        'alpha': [1.0, 0.5, 0.01],
        'h': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        'j': [5.6, 23.0]
    }
    permutations = list(gen_permutations(param_grid))
    assert len(permutations) == 42
    already_seen = set()
    for permutation in permutations:
        permutation = HashableDict(permutation)
        assert permutation not in already_seen
        already_seen.add(permutation)
