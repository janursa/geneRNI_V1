"""This is the example module.

This module does stuff.
"""
__all__ = ['a', 'b', 'c']
__version__ = '0.1'
__author__ = 'Jalil Nourisa'

import os
import sys
import pathlib
import pytest

dir_main = os.path.join(pathlib.Path(__file__).parent.resolve(),'..')
sys.path.insert(0, dir_main)

from geneRNI import types_

class Test:
    def test_one(self):
        x = "this"
        assert "h" in x