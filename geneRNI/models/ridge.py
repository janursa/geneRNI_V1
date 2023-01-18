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

from geneRNI.models import BaseWrapper


class RidgeWrapper(BaseWrapper):

    @staticmethod
    def new_estimator(*args, **kwargs) -> Ridge:
        # Pre-processing (mostly for non-tree-based models)
        # transformer = PowerTransformer(method='box-cox', standardize=True, copy=False)
        # transformer.fit_transform(X + 1e-15)
        # transformer.transform(X + 1e-15)
        #TODO: return should be the pipeline
        return Ridge(**kwargs)

    @staticmethod
    def compute_feature_importances(estimator: Ridge) -> np.array: #TODO: it also needs to be adjusted based on pipeline
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
            alpha=10. ** np.arange(-3, 4)
        )
