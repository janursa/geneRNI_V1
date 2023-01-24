# -*- coding: utf-8 -*-
#
#  ridge.py
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
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from geneRNI.models import BaseWrapper


class RidgeWrapper(BaseWrapper):

    @staticmethod
    def new_estimator(*args, **kwargs) -> Pipeline:
        return Pipeline(steps=[
            ('pt', StandardScaler()),
            ('ridge', Ridge(**kwargs))
        ])

    @staticmethod
    def compute_feature_importances(pipeline: Pipeline) -> np.array:
        estimator = pipeline.named_steps['ridge']
        coef = np.abs(estimator.coef_)
        assert len(coef.shape) == 1
        return coef

    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        return dict(
            estimator_t='ridge',
        )

    @staticmethod
    def get_grid_parameters() -> Dict[str, Any]:
        return dict(
            # alpha=10. ** np.arange(-3, 4)
        )
