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
from typing import Any


from geneRNI import types_
from geneRNI.models import get_estimator_wrapper


def is_lambda_function(obj: Any) -> bool:
    return isinstance(obj, types.LambdaType) and (obj.__name__ == '<lambda>')
def verboseprint(flag, message):
    if flag:
        print(message)
def default_settings(estimator_t):
    test_size = None
    param = get_estimator_wrapper(estimator_t).get_default_parameters()
    param_grid = get_estimator_wrapper(estimator_t).get_grid_parameters()
    if estimator_t == 'HGB':
        test_size = 0.25
    random_state_data = None
    random_state = None
    bootstrap_fold = None
    return types_.DefaultParamType(param, param_grid, test_size, bootstrap_fold, random_state, random_state_data)

