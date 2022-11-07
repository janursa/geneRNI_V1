from typing import Type

from .base import BaseWrapper
from .rf import RFWrapper
from .hgb import HGBWrapper
from .ridge import RidgeWrapper


__WRAPPERS__ = {
    'RF': RFWrapper,
    'HGB': HGBWrapper,
    'ridge': RidgeWrapper
}


def get_estimator_wrapper(estimator_t: str) -> Type[BaseWrapper]:
    if estimator_t not in __WRAPPERS__:
        raise ValueError(f'Unknown estimator "{estimator_t}"')
    return __WRAPPERS__[estimator_t]
