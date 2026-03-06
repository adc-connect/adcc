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
from .guess_zero import guess_symmetries, guess_zero
from .guesses_from_diagonal import guesses_from_diagonal
from .util import estimate_n_guesses, determine_spin_change

__all__ = ["guess_zero", "guesses_from_diagonal", 
           "get_spin_block_symmetrisation", "guesses_doublet",
           "guesses_singlet", "guesses_triplet", "guesses_any",
           "guesses_spin_flip", "guess_symmetries",
           "estimate_n_guesses", "determine_spin_change"]


def get_spin_block_symmetrisation(kind: str) -> str:
    """
    Return the kwargs required to be passed to `guesses_from_diagonal` to
    computed states of the passed excitation `kind`.
    """
    symmetrisation = {
        "singlet": "symmetric",
        "doublet": "none",
        "triplet": "antisymmetric",
        "spin_flip":"none",
        "any": "none"
    }
    try:
        return symmetrisation[kind]
    except KeyError:
        raise ValueError(f"Kind not known: {kind}")


def guesses_singlet(matrix, n_guesses, block="ph", **kwargs):
    """
    Obtain guesses for computing singlet states by inspecting the passed
    ADC matrix.

    matrix      The matrix for which guesses are to be constructed
    n_guesses   The number of guesses to be searched for. Less number of
                vectors are returned if this many could not be found.
    block       Diagonal block to use for obtaining the guesses
                (typically "ph" or "pphh").
    is_alpha    Is the detached/attached electron alpha spin for the respective
                IP-/EA-ADC calculation.
    kwargs      Any other argument understood by guesses_from_diagonal.
    """
    return guesses_from_diagonal(
        matrix, n_guesses, block=block, spin_change=0,
        spin_block_symmetrisation=get_spin_block_symmetrisation("singlet"),
        **kwargs
    )


def guesses_doublet(matrix, n_guesses, block="h", is_alpha=True, **kwargs):
    """
    Obtain guesses for computing doublet states by inspecting the passed
    ADC matrix.

    matrix      The matrix for which guesses are to be constructed
    n_guesses   The number of guesses to be searched for. Less number of
                vectors are returned if this many could not be found.
    block       Diagonal block to use for obtaining the guesses
                (typically "ph" or "pphh").
    is_alpha    Is the detached/attached electron alpha spin for the respective
                IP-/EA-ADC calculation.
    kwargs      Any other argument understood by guesses_from_diagonal.
    """
    if matrix.method.adc_type == "ip":
        spin_change = -0.5
    elif matrix.method.adc_type == "ea":
        spin_change = 0.5
    return guesses_from_diagonal(
        matrix, n_guesses, block=block,
        is_alpha=is_alpha, spin_change=spin_change,
        spin_block_symmetrisation= get_spin_block_symmetrisation("doublet"),
        **kwargs
    )


def guesses_triplet(matrix, n_guesses, block="ph", **kwargs):
    """
    Obtain guesses for computing triplet states by inspecting the passed
    ADC matrix.

    matrix      The matrix for which guesses are to be constructed
    n_guesses   The number of guesses to be searched for. Less number of
                vectors are returned if this many could not be found.
    block       Diagonal block to use for obtaining the guesses
                (typically "ph" or "pphh").
    is_alpha    Is the detached/attached electron alpha spin for the respective
                IP-/EA-ADC calculation.
    kwargs      Any other argument understood by guesses_from_diagonal.
    """
    return guesses_from_diagonal(
        matrix, n_guesses, block=block, spin_change=0,
        spin_block_symmetrisation= get_spin_block_symmetrisation("triplet"),
        **kwargs
    )


# guesses for computing any state (excluding spin-flip states)
guesses_any = guesses_from_diagonal


def guesses_spin_flip(matrix, n_guesses, block="ph", **kwargs):
    """
    Obtain guesses for computing spin-flip states by inspecting the passed
    ADC matrix.

    matrix      The matrix for which guesses are to be constructed
    n_guesses   The number of guesses to be searched for. Less number of
                vectors are returned if this many could not be found.
    block       Diagonal block to use for obtaining the guesses
                (typically "ph" or "pphh").
    is_alpha    Is the detached/attached electron alpha spin for the respective
                IP-/EA-ADC calculation.
    kwargs      Any other argument understood by guesses_from_diagonal.
    """
    return guesses_from_diagonal(
        matrix, n_guesses, block=block, spin_change=-1,
        spin_block_symmetrisation= get_spin_block_symmetrisation("spin_flip"),
        **kwargs
    )