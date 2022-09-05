"""This is the example module.

This module does stuff.
"""
__all__ = ['a', 'b', 'c']
__version__ = '0.1'
__author__ = 'Jalil Nourisa'

import time
import numpy as np
from typing import NamedTuple

class DataType(NamedTuple):
    Xs_train: np.array
    ys_train: np.array
    Xs_test: np.array = None
    ys_test: np.array = None
class DefaultParamType(NamedTuple):
    param: dict
    param_grid: dict
    test_size: float
    bootstrap_fold: int
    random_state: int
    random_state_data: int
