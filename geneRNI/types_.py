"""This is the example module.

This module does stuff.
"""

__all__ = ['a', 'b', 'c']
__version__ = '0.1'
__author__ = 'Jalil Nourisa'

from typing import NamedTuple, Optional

import numpy as np


class DefaultParamType(NamedTuple):
    param: dict
    param_grid: dict
    test_size: Optional[float] = None  # TODO: default value?
    bootstrap_fold: Optional[int] = None
    random_state: Optional[int] = None
    random_state_data: Optional[int] = None
