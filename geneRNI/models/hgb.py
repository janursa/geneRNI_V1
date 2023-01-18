# -*- coding: utf-8 -*-
#
#  hgb.py
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

from typing import Dict, Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from geneRNI.models import BaseWrapper


class HGBWrapper(BaseWrapper):

    @staticmethod
    def new_estimator(*args, **kwargs) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(**kwargs)

    @staticmethod
    def compute_feature_importances(estimator: HistGradientBoostingRegressor) -> np.array:
        # TODO: implement this
        return np.array([e.tree_.compute_feature_importances(normalize=False)
                                for e in estimator.estimators_])

    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        return dict(
            estimator_t='HGB',
            min_samples_leaf=5,
            learning_rate=.05,
            # criterion = 'absolute_error',
            max_iter=50,
        )

    @staticmethod
    def get_grid_parameters() -> Dict[str, Any]:
        return dict(
            learning_rate=np.arange(0.001, .2, .02),
            min_samples_leaf=np.arange(1, 30, 2),
            max_iter=np.arange(20, 200, 10),
        )
