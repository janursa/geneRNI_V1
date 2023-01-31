# -*- coding: utf-8 -*-
#
#  test-cv.py
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

from geneRNI.models import get_estimator_wrapper
from geneRNI.search_param import loo_cross_validate, shuffle_cross_validate


def create_easy_dataset() -> Tuple[np.ndarray, np.ndarray]:
    X = np.random.rand(30, 6)
    y = np.random.rand(30)
    X[:, 3] = 0.1 * X[:, 3] + 0.9 * y
    return X, y


def test_leave_one_out():
    X, y = create_easy_dataset()
    wrapper = get_estimator_wrapper('ridge')
    estimator = wrapper.new_estimator()
    r2 = loo_cross_validate(estimator, X, y)
    assert r2 > 0.8


def test_k_fold():
    X, y = create_easy_dataset()
    wrapper = get_estimator_wrapper('ridge')
    estimator = wrapper.new_estimator()
    r2 = shuffle_cross_validate(estimator, X, y)
    assert r2 > 0.8
