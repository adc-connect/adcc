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

__all__ = ["guess_zero", "guesses_from_diagonal",
           "guesses_singlet", "guesses_triplet", "guesses_any",
           "guesses_spin_flip", "guess_symmetries"]


def guess_kwargs_kind(kind):
    """
    Return the kwargs required to be passed to `guesses_from_diagonal` to
    computed states of the passed excitation `kind`.
    """
    kwargsmap = dict(
        singlet=dict(spin_block_symmetrisation="symmetric", spin_change=0),
        triplet=dict(spin_block_symmetrisation="antisymmetric", spin_change=0),
        spin_flip=dict(spin_block_symmetrisation="none", spin_change=-1),
        any=dict(spin_block_symmetrisation="none", spin_change=0),
    )
    try:
        return kwargsmap[kind]
    except KeyError:
        raise ValueError(f"Kind not known: {kind}")


def guesses_singlet(matrix, n_guesses, block="ph", **kwargs):
    """
    Obtain guesses for computing singlet states by inspecting the passed
    ADC matrix.

    matrix       The matrix for which guesses are to be constructed
    n_guesses    The number of guesses to be searched for. Less number of
                 vectors are returned if this many could not be found.
    block        Diagonal block to use for obtaining the guesses
                 (typically "ph" or "pphh").
    kwargs       Any other argument understood by guesses_from_diagonal.
    """
    return guesses_from_diagonal(matrix, n_guesses, block=block,
                                 **guess_kwargs_kind("singlet"), **kwargs)


def guesses_triplet(matrix, n_guesses, block="ph", **kwargs):
    """
    Obtain guesses for computing triplet states by inspecting the passed
    ADC matrix.

    matrix       The matrix for which guesses are to be constructed
    n_guesses    The number of guesses to be searched for. Less number of
                 vectors are returned if this many could not be found.
    block        Diagonal block to use for obtaining the guesses
                 (typically "ph" or "pphh").
    kwargs       Any other argument understood by guesses_from_diagonal.
    """
    return guesses_from_diagonal(matrix, n_guesses, block=block,
                                 **guess_kwargs_kind("triplet"), **kwargs)


# guesses for computing any state (singlet or triplet)
guesses_any = guesses_from_diagonal


def guesses_spin_flip(matrix, n_guesses, block="ph", **kwargs):
    """
    Obtain guesses for computing spin-flip states by inspecting the passed
    ADC matrix.

    matrix       The matrix for which guesses are to be constructed
    n_guesses    The number of guesses to be searched for. Less number of
                 vectors are returned if this many could not be found.
    block        Diagonal block to use for obtaining the guesses
                 (typically "ph" or "pphh").
    kwargs       Any other argument understood by guesses_from_diagonal.
    """
    return guesses_from_diagonal(matrix, n_guesses, block=block,
                                 **guess_kwargs_kind("spin_flip"), **kwargs)
