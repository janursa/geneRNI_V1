# -*- coding: utf-8 -*-
#
#  correction.py
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
import logging

import numpy as np


def correct_grn_matrix(
        C: np.ndarray,
        method: str = 'rcw',
        eps: float = 1e-50,
        ignore_diag: bool = True
) -> np.ndarray:
    if method not in {'none', 'rcw', 'apc', 'asc'}:
        raise NotImplementedError(f'Unknown GRN matrix correction "{method}"')
    if method == 'none':
        return C
    try:
        import portia.correction
        return portia.correction.apply_correction(C, method=method, eps=eps, ignore_diag=ignore_diag)
    except ImportError:
        logging.exception('portia-grn is not installed. No correction could be applied.')
