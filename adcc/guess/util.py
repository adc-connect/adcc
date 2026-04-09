#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2018 by the adcc authors
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

from ..AdcMethod import AdcMethod
from typing import Optional


def determine_spin_change(method: AdcMethod, kind: str,
                          is_alpha: Optional[bool] = None) -> int | float:
    if method.adc_type == "pp":
        if kind == "spin_flip":
            return -1
        else:
            return 0
    elif method.adc_type == "ip":
        if is_alpha is None:
            raise TypeError("'is_alpha' has to be True|False for IP-ADC")
        return +0.5 - int(is_alpha)
    elif method.adc_type == "ea":
        if is_alpha is None:
            raise TypeError("'is_alpha' has to be True|False for EA-ADC")
        return -0.5 + int(is_alpha)
    else:
        raise ValueError(f"Unknown ADC method: {method.name}")


def estimate_n_guesses(matrix, n_states, n_guesses_per_state=2,
                           singles_only=True) -> int:
    """
    Implementation of a basic heuristic to find a good number of guess
    vectors to be searched for using the find_guesses function.
    Internal function called from run_adc.

    matrix             ADC matrix
    n_states           Number of states to be computed
    singles_only       Try to stay withing the singles excitation space
                    with the number of guess vectors.
    n_guesses_per_state  Number of guesses to search for for each state
    """
    # Try to use at least 4 or twice the number of states
    # to be computed as guesses
    n_guesses = n_guesses_per_state * max(2, n_states)

    if singles_only:
        # Compute the maximal number of sensible singles block guesses.
        # This is roughly the number of occupied alpha orbitals
        # times the number of virtual alpha orbitals
        #
        # If the system is core valence separated, then only the
        # core electrons count as "occupied".
        mospaces = matrix.mospaces
        sp_occ = "o2" if matrix.is_core_valence_separated else "o1"
        n_virt_a = mospaces.n_orbs_alpha("v1")
        n_occ_a = mospaces.n_orbs_alpha(sp_occ)
        estimate = n_occ_a * n_virt_a
        if matrix.method.level < 2 and matrix.method.adc_type != "pp":
            # Adjustment for IP- and EA-ADC(0/1) calculations
            estimate = (n_occ_a if matrix.method.adc_type == "ip" 
                        else n_virt_a)
        n_guesses = min(n_guesses, estimate)

    # Adjust if we overshoot the maximal number of sensible singles block
    # guesses, but make sure we get at least n_states guesses
    return max(n_states, n_guesses)