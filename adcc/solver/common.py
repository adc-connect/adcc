#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2020 by the adcc authors
##
## This file is part of adcc.
##
## adcc is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------
import numpy as np


def select_eigenpairs(eigenvalues, n_ep, which):
    """
    Return a numpy `bool` mask selecting the `n_ep` eigenpairs of the `which`
    criterion. It is assumed that the `eigenvalues` are sorted algebraically
    from the smallest to the largest.
    """
    mask = np.zeros(len(eigenvalues), dtype=bool)
    if which == "LA":    # Largest algebraic
        mask[-n_ep:] = True
    elif which == "SA":  # Smallest algebraic
        mask[:n_ep] = True
    elif which == "LM":  # Largest magnitude
        sorti = np.argsort(np.abs(eigenvalues))[-n_ep:]
        mask[sorti] = True
    elif which == "SM":  # Smallest magnitude
        sorti = np.argsort(np.abs(eigenvalues))[:n_ep]
        mask[sorti] = True
    else:
        raise ValueError("For now only the values 'LM', 'LA', 'SM' and 'SA' "
                         "are understood for 'which'.")
    return mask.nonzero()[0]
