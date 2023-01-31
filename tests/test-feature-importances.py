# -*- coding: utf-8 -*-
#
#  test-feature-importances.py
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

from geneRNI.models import get_estimator_names, get_estimator_wrapper


def random_dataset() -> Tuple[np.ndarray, np.ndarray, int]:
    X = np.random.rand(100, 20)
    X[:50, 7] += 0.5
    y = np.ones(100, dtype=int)
    y[:50] = 0
    return X, y, 7


def test_feature_importance():
    X, y, best_feature_location = random_dataset()
    for est_name in set(get_estimator_names()) - {'HGB'}:
        wrapper = get_estimator_wrapper(est_name)
        estimator = wrapper.new_estimator()
        estimator.fit(X, y)
        feature_importance = wrapper.compute_feature_importances(estimator)
        assert np.argmax(feature_importance) == best_feature_location
